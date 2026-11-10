"""Task 4: run and cache the official simulation (notebook defaults, estimator loop).

Run from the repository root, one molecule per call (A, B or C):
    .venv/bin/python agents/agent2_implementation/run_official_simulation.py A

Writes results/signals/molecule_<X>.npz and agents/agent2_implementation/task4_molecule_<X>.json.
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import qiskit
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import nmr_pipeline as nmr  # noqa: E402

# Chemical shifts quoted in docs/challenge_analysis.md (ppm), used only for the side-by-side list.
SHIFTS_PPM = {"A": [5.790, 5.970, 5.480], "B": [5.580, 5.900, 5.340], "C": [6.375, 6.302]}

letter = sys.argv[1]
steps = nmr.DEFAULT_NUMBER_OF_TROTTERSTEPS
dt = nmr.DEFAULT_TROTTER_TIMESTEP
noise = nmr.DEFAULT_NOISE_STRENGTH

ham, n, shift = nmr.load_rotating_frame_hamiltonian(f"molecule {letter}")
_, _, bits_list, _ = nmr.enumerate_x_states(n)
bits = np.array(bits_list, dtype=np.int64)

t0 = time.perf_counter()
exp_x, exp_y = nmr.simulate_signals_estimator(ham, n, steps, dt)
t_est = time.perf_counter() - t0

t0 = time.perf_counter()
fast_x, fast_y = nmr.simulate_signals_fast(ham, n, steps, dt)
t_fast = time.perf_counter() - t0

c_xx_raw, c_yx_raw = nmr.correlations(exp_x, exp_y)
times = np.arange(steps) * dt
mx0 = exp_x[:, 0].copy()

out = REPO / "results" / "signals" / f"molecule_{letter}.npz"
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    out,
    exp_x=exp_x,
    exp_y=exp_y,
    bits=bits,
    mx0=mx0,
    c_xx_raw=c_xx_raw,
    c_yx_raw=c_yx_raw,
    times=times,
    chemical_shift_hz=np.float64(shift),
    number_spins=np.int64(n),
    trotter_timestep=np.float64(dt),
    number_of_trottersteps=np.int64(steps),
)

spectrum, ppm = nmr.spectrum_from_signals(c_xx_raw, c_yx_raw, noise, dt, steps, shift)
peak_idx, props = find_peaks(spectrum, prominence=0.05)

report = {
    "molecule": letter,
    "number_spins": int(n),
    "number_of_trottersteps": steps,
    "trotter_timestep": dt,
    "noise_strength_for_peak_list": noise,
    "shape_exp_x": list(exp_x.shape),
    "shape_exp_y": list(exp_y.shape),
    "shape_bits": list(bits.shape),
    "all_finite": bool(
        np.isfinite(exp_x).all() and np.isfinite(exp_y).all()
        and np.isfinite(c_xx_raw).all() and np.isfinite(c_yx_raw).all()
    ),
    "max_abs_mx0_minus_expected": float(np.max(np.abs(mx0 - (n - 2 * bits.sum(axis=1))))),
    "mx0": mx0.tolist(),
    "c_xx_raw_0": float(c_xx_raw[0]),
    "expected_c_xx_raw_0_N_times_2powN": int(n * 2**n),
    "abs_c_xx_raw_0_minus_expected": float(abs(c_xx_raw[0] - n * 2**n)),
    "c_yx_raw_0": float(c_yx_raw[0]),
    "max_abs_exp_x": float(np.max(np.abs(exp_x))),
    "max_abs_exp_y": float(np.max(np.abs(exp_y))),
    "max_abs_c_xx_raw": float(np.max(np.abs(c_xx_raw))),
    "max_abs_c_yx_raw": float(np.max(np.abs(c_yx_raw))),
    "std_c_xx_raw": float(np.std(c_xx_raw)),
    "std_c_yx_raw": float(np.std(c_yx_raw)),
    "max_abs_exp_x_estimator_minus_fast": float(np.max(np.abs(exp_x - fast_x))),
    "max_abs_exp_y_estimator_minus_fast": float(np.max(np.abs(exp_y - fast_y))),
    "wall_time_estimator_s": t_est,
    "wall_time_fast_s": t_fast,
    "spectrum_finite": bool(np.isfinite(spectrum).all()),
    "spectrum_max": float(spectrum.max()),
    "ppm_min": float(ppm.min()),
    "ppm_max": float(ppm.max()),
    "ppm_grid_spacing": float(ppm[1] - ppm[0]),
    "peaks_ppm": [float(ppm[i]) for i in peak_idx],
    "peaks_height": [float(spectrum[i]) for i in peak_idx],
    "peaks_prominence": [float(p) for p in props["prominences"]],
    "chemical_shifts_ppm_from_analysis": SHIFTS_PPM[letter],
    "chemical_shift_hz_notebook_name_rad_s": float(shift),
    "npz": str(out.relative_to(REPO)),
    "python": platform.python_version(),
    "qiskit": qiskit.__version__,
    "numpy": np.__version__,
}
(Path(__file__).with_name(f"task4_molecule_{letter}.json")).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
