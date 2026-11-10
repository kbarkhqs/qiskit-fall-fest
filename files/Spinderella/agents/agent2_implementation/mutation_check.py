"""Mutation check: the upstream placeholder (git HEAD, cell 14) must FAIL the state checks.

Run from the repository root:
    .venv/bin/python agents/agent2_implementation/mutation_check.py

Three independent detectors are tried on the upstream cell 14:
  1. tests/state_checks.assert_complete_x_basis (the function used by pytest),
  2. the verification cell that was added to the notebook (cell 15 of the edited notebook),
  3. the guard added to time_signal_to_spectrum (after a short estimator run).
As a control, the same three detectors are run on the edited cell 14 and must all pass.
"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
import nmr_pipeline as nmr  # noqa: E402
from state_checks import assert_complete_x_basis  # noqa: E402

upstream = json.loads(subprocess.run(["git", "show", "HEAD:nmr.ipynb"], cwd=REPO, capture_output=True, text=True, check=True).stdout)
edited = json.loads((REPO / "nmr.ipynb").read_text(encoding="utf-8"))
src = lambda nb, i: "".join(nb["cells"][i]["source"])  # noqa: E731
assert "Placeholder so the notebook runs" in src(upstream, 14)
assert "Check of the initial states" in src(edited, 15)


def no_prints(text):
    return "\n".join(line for line in text.split("\n") if not line.startswith("print("))


def run_case(label, cell14_source, molecule):
    """Execute cells 2, 4, 6, 8, 9 of the upstream notebook, then the given cell 14."""
    ns = {}
    import os
    os.chdir(REPO)
    for i in (2, 4, 6, 8, 9):
        text = src(upstream, i)
        if i == 4:
            text = text.replace('molecule_name = "molecule A"', f'molecule_name = "{molecule}"')
        exec(compile(no_prints(text), f"upstream_cell_{i}", "exec"), ns)
    exec(compile(cell14_source, "cell_14", "exec"), ns)
    n = ns["number_spins"]
    out = {"case": label, "molecule": molecule, "number_spins": n, "number_of_states": len(ns["bits_list"])}

    # detector 1: pytest helper
    try:
        assert_complete_x_basis(ns["qc_init_template"], ns["phi"], ns["bits_list"], ns["param_maps"], n)
        out["detector1_state_checks"] = "pass"
    except AssertionError as err:
        out["detector1_state_checks"] = f"AssertionError: {err}"

    # detector 2: the verification cell of the edited notebook (output suppressed)
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src(edited, 15), "edited_cell_15", "exec"), ns)
        out["detector2_notebook_cell_15"] = "pass"
    except AssertionError as err:
        out["detector2_notebook_cell_15"] = f"AssertionError: {err}"

    # detector 3: short official estimator loop (5 steps) + spectrum with the guard
    steps = 5
    circuit = nmr.trotter_step_circuit(ns["hamiltonian_qiskit"], 0.0025)
    obs_x, obs_y = nmr.magnetization_observables(n)
    from qiskit.circuit import QuantumCircuit
    c = QuantumCircuit(n).compose(ns["qc_init_template"])
    exp_x = np.zeros((len(ns["bits_list"]), steps))
    exp_y = np.zeros_like(exp_x)
    estimator = nmr.Estimator()
    for k in range(steps):
        if k > 0:
            c = c.compose(circuit)
        results = estimator.run([(c, [obs_x, obs_y], pm) for pm in ns["param_maps"]]).result()
        for m, res in enumerate(results):
            exp_x[m, k], exp_y[m, k] = res.data.evs
    c_xx, c_yx = nmr.correlations(exp_x, exp_y)
    out["max_abs_mx0"] = float(np.max(np.abs(exp_x[:, 0])))
    out["max_abs_c_xx_raw"] = float(np.max(np.abs(c_xx)))
    out["max_abs_c_yx_raw"] = float(np.max(np.abs(c_yx)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nmr.spectrum_from_signals(c_xx, c_yx, 5.0, 0.0025, steps, ns["chemical_shift_hz"])
        out["detector3_guard"] = "pass"
    except ValueError as err:
        out["detector3_guard"] = f"ValueError: {err}"

    # what the upstream (unguarded) spectrum function does with the same signals
    ns_up = {}
    exec(compile(src(upstream, 2) + "\n" + src(upstream, 24), "upstream_cell_24", "exec"), ns_up)
    window = np.exp(-5.0 * np.arange(steps) * 0.0025)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec_up, _ = ns_up["time_signal_to_spectrum"](
            c_xx * window, c_yx * window, 0.0025, steps,
            Hz_to_ppm_conversion=nmr.hz_to_ppm_conversion(), chem_shift_hz=ns["chemical_shift_hz"],
        )
    out["upstream_function_warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
    out["upstream_function_spectrum_all_nan"] = bool(np.isnan(spec_up).all())
    return out


results = []
for molecule in ("molecule A", "molecule C"):
    results.append(run_case("upstream placeholder (git HEAD cell 14)", src(upstream, 14), molecule))
    results.append(run_case("edited cell 14 (control)", src(edited, 14), molecule))

for r in results:
    mutated = r["case"].startswith("upstream")
    detectors = [r["detector1_state_checks"], r["detector2_notebook_cell_15"], r["detector3_guard"]]
    r["all_detectors_fired"] = all(d != "pass" for d in detectors)
    r["all_detectors_passed"] = all(d == "pass" for d in detectors)
    assert r["all_detectors_fired"] if mutated else r["all_detectors_passed"], r

Path(__file__).with_name("mutation_check.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print("MUTATION CHECK OK: the upstream placeholder is rejected by all three detectors; the edited cell passes.")
