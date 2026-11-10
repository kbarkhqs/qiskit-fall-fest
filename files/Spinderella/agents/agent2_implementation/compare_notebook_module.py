"""Task 5: prove that nmr.ipynb and src/nmr_pipeline.py compute the same numbers.

Run from the repository root (takes as long as one notebook run):
    .venv/bin/python agents/agent2_implementation/compare_notebook_module.py

A copy of the notebook is executed in memory with nbclient with ONE extra temporary cell
that dumps the notebook variables to a temporary npz. The copy is never written to disk, so
nmr.ipynb is not modified. The dump is compared with the module for molecule A, noise 5.0.
A second check runs the upstream (unguarded) time_signal_to_spectrum from git HEAD on the
cached signals to show that the guard did not change the maths.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat
import numpy as np
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import nmr_pipeline as nmr  # noqa: E402

out = {}

# ---------------------------------------------------------------------------------------
# 1. execute an in-memory copy of the notebook with a temporary dump cell
# ---------------------------------------------------------------------------------------
nb = nbformat.read(REPO / "nmr.ipynb", as_version=4)
n_cells_on_disk = len(nb.cells)
with tempfile.TemporaryDirectory() as tmp:
    dump = Path(tmp) / "notebook_dump.npz"
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "import sys as _sys\n"
            f"np.savez(r'{dump}', c_xx=c_xx, c_yx=c_yx, spectrum_full=spectrum_full,\n"
            "         ppm=np.array([float(v) for v in ppm]), ppm_dtype=str(ppm.dtype),\n"
            "         exp_x=exp_x, exp_y=exp_y, chemical_shift_hz=float(chemical_shift_hz),\n"
            "         chemical_shift_type=type(chemical_shift_hz).__name__,\n"
            "         noise_strength=noise_strength, molecule_name=molecule_name,\n"
            "         number_of_trottersteps=number_of_trottersteps, trotter_timestep=trotter_timestep,\n"
            "         reference_max=float(np.max(hqs_spectrum_tools_spectrum)),\n"
            "         reference_file_max=float(np.max(hqs_spectrum_tools_results['spectrum'])),\n"
            "         executable=_sys.executable)\n"
        )
    )
    t0 = time.perf_counter()
    NotebookClient(nb, kernel_name="python3", timeout=None, resources={"metadata": {"path": str(REPO)}}).execute()
    out["notebook_wall_time_s"] = time.perf_counter() - t0
    d = dict(np.load(dump))

problems = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    for o in cell.get("outputs", []):
        if o.output_type == "error":
            problems.append(f"cell {i}: error {o.ename}")
        text = o.get("text", "") if o.output_type == "stream" else json.dumps(o.get("data", {}).get("text/plain", ""))
        if o.output_type == "stream" and o.name == "stderr":
            problems.append(f"cell {i}: stderr {text[:120]!r}")
        if "Warning" in text:
            problems.append(f"cell {i}: warning text {text[:120]!r}")
out["in_memory_run_problems"] = problems
out["in_memory_cells_executed"] = len(nb.cells)
out["cells_on_disk"] = n_cells_on_disk
out["kernel_executable"] = str(d["executable"])
out["notebook_molecule"] = str(d["molecule_name"])
out["notebook_noise_strength"] = float(d["noise_strength"])
out["notebook_ppm_dtype"] = str(d["ppm_dtype"])
out["notebook_chemical_shift_type"] = str(d["chemical_shift_type"])
out["notebook_reference_max_after_normalisation"] = float(d["reference_max"])
out["notebook_reference_dict_max_after_cell"] = float(d["reference_file_max"])

# ---------------------------------------------------------------------------------------
# 2. the module, molecule A, noise 5.0 (official estimator route and fast route)
# ---------------------------------------------------------------------------------------
steps, dt, noise = int(d["number_of_trottersteps"]), float(d["trotter_timestep"]), float(d["noise_strength"])
ham, n, shift = nmr.load_rotating_frame_hamiltonian("molecule A")
window = np.exp(-noise * np.arange(steps) * dt)


def compare(tag, exp_x, exp_y):
    c_xx_raw, c_yx_raw = nmr.correlations(exp_x, exp_y)
    spectrum, ppm = nmr.spectrum_from_signals(c_xx_raw, c_yx_raw, noise, dt, steps, shift)
    out[f"max_abs_diff_exp_x_{tag}"] = float(np.max(np.abs(exp_x - d["exp_x"])))
    out[f"max_abs_diff_exp_y_{tag}"] = float(np.max(np.abs(exp_y - d["exp_y"])))
    out[f"max_abs_diff_c_xx_{tag}"] = float(np.max(np.abs(c_xx_raw * window - d["c_xx"])))
    out[f"max_abs_diff_c_yx_{tag}"] = float(np.max(np.abs(c_yx_raw * window - d["c_yx"])))
    out[f"max_abs_diff_spectrum_{tag}"] = float(np.max(np.abs(spectrum - d["spectrum_full"])))
    out[f"max_abs_diff_ppm_{tag}"] = float(np.max(np.abs(ppm - d["ppm"])))
    return spectrum, ppm


cache_path = REPO / "results" / "signals" / "molecule_A.npz"
for _ in range(120):  # the cache may still be written by run_official_simulation.py
    if cache_path.exists():
        break
    time.sleep(10)
cache = np.load(cache_path)
compare("module_estimator_cached", cache["exp_x"], cache["exp_y"])
fast_x, fast_y = nmr.simulate_signals_fast(ham, n, steps, dt)
spectrum, ppm = compare("module_fast", fast_x, fast_y)
out["max_abs_diff_chemical_shift"] = abs(shift - float(d["chemical_shift_hz"]))
out["notebook_spectrum_finite"] = bool(np.isfinite(d["spectrum_full"]).all())
out["notebook_spectrum_max"] = float(d["spectrum_full"].max())
out["notebook_spectrum_points_above_0.05"] = int(np.sum(d["spectrum_full"] > 0.05))
out["notebook_max_abs_c_xx"] = float(np.max(np.abs(d["c_xx"])))
out["notebook_max_abs_c_yx"] = float(np.max(np.abs(d["c_yx"])))

# ---------------------------------------------------------------------------------------
# 3. the guard did not change the maths: upstream cell 24 (git HEAD) vs module
# ---------------------------------------------------------------------------------------
upstream = json.loads(subprocess.run(["git", "show", "HEAD:nmr.ipynb"], cwd=REPO, capture_output=True, text=True, check=True).stdout)
ns = {}
exec(compile("".join(upstream["cells"][2]["source"]) + "\n" + "".join(upstream["cells"][24]["source"]), "upstream_cell_24", "exec"), ns)
guard = {}
for letter in "ABC":
    c = np.load(REPO / "results" / "signals" / f"molecule_{letter}.npz")
    st, tdt, cs = int(c["number_of_trottersteps"]), float(c["trotter_timestep"]), float(c["chemical_shift_hz"])
    worst = 0.0
    for g in (0.1, 1.0, 5.0, 20.0):
        w = np.exp(-g * np.arange(st) * tdt)
        s_up, p_up = ns["time_signal_to_spectrum"](c["c_xx_raw"] * w, c["c_yx_raw"] * w, tdt, st, Hz_to_ppm_conversion=nmr.hz_to_ppm_conversion(), chem_shift_hz=cs)
        s_mod, p_mod = nmr.spectrum_from_signals(c["c_xx_raw"], c["c_yx_raw"], g, tdt, st, cs)
        worst = max(worst, float(np.max(np.abs(s_up - s_mod))), float(np.max(np.abs(p_up - p_mod))))
    guard[letter] = worst
out["max_abs_diff_upstream_unguarded_vs_module_spectrum_and_ppm"] = guard

Path(__file__).with_name("compare_notebook_module.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
