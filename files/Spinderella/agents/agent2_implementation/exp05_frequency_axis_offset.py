"""Experiment 05: which frequency does sample m of the notebook's fft_causal really hold?

Found while reading the molecule C peak list: its four peaks sit at bins -17, -12, +11, +16
around the centre of the ppm axis, i.e. mirror-symmetric about bin -0.5 instead of bin 0.

fft_causal builds two (2L-1)-point signals whose sum is the zero-padded causal signal (first
sample halved), takes the FFT and keeps every second point. A (2L-1)-point FFT has the
frequencies (j - (L-1)) / ((2L-1) dt); keeping j = 2m gives
    f_derived(m) = (2m - (L-1)) / ((2L-1) dt)
while the notebook labels sample m with fftshift(fftfreq(L, dt)) = (m - L/2) / (L dt).

Test A (identity, no fit): compare fft_causal(f) with the explicit Fourier sum of the causal
signal evaluated (1) at f_derived and (2) at the notebook labels, for random signals.
Test B (line position): one broad line (decay 40 1/s, FWHM 12.7 Hz = 9.5 grid points) from
the signal a single spin produces, <X> = cos(2 pi f0 t), <Y> = sin(2 pi f0 t); the line centre
is fitted on both axes. A spin with rotating-frame Pauli-Z coefficient h' has f0 = h'/pi.

Run from the repository root:
    .venv/bin/python agents/agent2_implementation/exp05_frequency_axis_offset.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import nmr_pipeline as nmr  # noqa: E402

L, dt = 300, 0.0025
t = np.arange(L) * dt
label_hz = np.fft.fftshift(np.fft.fftfreq(L, d=dt))
derived_hz = (2 * np.arange(L) - (L - 1)) / ((2 * L - 1) * dt)
conv = nmr.hz_to_ppm_conversion()


def fourier_sum(signal, freqs_hz):
    """sum_k s_k exp(-2 pi i f k dt) / sqrt(2L-1) with the first sample halved."""
    s = signal.astype(complex).copy()
    s[0] /= 2
    return np.exp(-2j * np.pi * np.outer(freqs_hz, t)) @ s / np.sqrt(2 * L - 1)


# ---- Test A ---------------------------------------------------------------------------
rng = np.random.default_rng(5)
diff_derived, diff_label = [], []
for _ in range(20):
    f = rng.normal(size=L) * np.exp(-rng.uniform(0, 10) * t)
    out = nmr.fft_causal(f)
    scale = np.max(np.abs(out))
    diff_derived.append(np.max(np.abs(out - fourier_sum(f, derived_hz))) / scale)
    diff_label.append(np.max(np.abs(out - fourier_sum(f, label_hz))) / scale)

# ---- Test B ---------------------------------------------------------------------------
gamma = 40.0
rows = []
for f0 in [-150.0, -60.0, -20.0, -3.0, 0.0, 3.0, 20.0, 60.0, 150.0] + [float(x) for x in rng.uniform(-150, 150, 6)]:
    c_xx = np.cos(2 * np.pi * f0 * t) * np.exp(-gamma * t)
    c_yx = np.sin(2 * np.pi * f0 * t) * np.exp(-gamma * t)
    spec = nmr.fft_causal(c_xx).real + nmr.fft_causal(c_yx).imag
    k = int(np.argmax(spec))
    sel = slice(k - 4, k + 5)
    centre = {}
    for name, axis in (("notebook", label_hz), ("derived", derived_hz)):
        coeff = np.polyfit(axis[sel], 1.0 / spec[sel], 2)  # 1/Lorentzian is a parabola
        centre[name] = float(-coeff[1] / (2 * coeff[0]))
    rows.append(
        {
            "f0_hz": f0,
            "expected_line_position_hz": -f0,
            "centre_on_notebook_axis_hz": centre["notebook"],
            "notebook_axis_minus_expected_hz": centre["notebook"] + f0,
            "centre_on_derived_axis_hz": centre["derived"],
            "derived_axis_minus_expected_hz": centre["derived"] + f0,
        }
    )
err_nb = np.array([r["notebook_axis_minus_expected_hz"] for r in rows])
err_de = np.array([r["derived_axis_minus_expected_hz"] for r in rows])

summary = {
    "L": L,
    "dt": dt,
    "testA_max_rel_diff_fft_causal_vs_fourier_sum_at_derived_axis": float(max(diff_derived)),
    "testA_min_rel_diff_fft_causal_vs_fourier_sum_at_notebook_axis": float(min(diff_label)),
    "grid_spacing_notebook_axis_hz": float(label_hz[1] - label_hz[0]),
    "grid_spacing_derived_axis_hz": float(derived_hz[1] - derived_hz[0]),
    "derived_minus_notebook_axis_hz_at_left_edge": float((derived_hz - label_hz)[0]),
    "derived_minus_notebook_axis_hz_at_centre": float((derived_hz - label_hz)[L // 2]),
    "derived_minus_notebook_axis_hz_at_right_edge": float((derived_hz - label_hz)[-1]),
    "derived_minus_notebook_axis_ppm_at_centre": float((derived_hz - label_hz)[L // 2] * conv),
    "testB_decay_rate": gamma,
    "testB_line_position_error_on_notebook_axis_hz_min": float(err_nb.min()),
    "testB_line_position_error_on_notebook_axis_hz_max": float(err_nb.max()),
    "testB_line_position_error_on_notebook_axis_hz_mean": float(err_nb.mean()),
    "testB_line_position_error_on_derived_axis_hz_max_abs": float(np.max(np.abs(err_de))),
    "testB_line_appears_at_minus_f0": bool(np.all(np.abs(err_de) < 0.5)),
    "hz_to_ppm": float(conv),
}
Path(__file__).with_name("exp05_frequency_axis_offset.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
for r in rows:
    print(f"f0={r['f0_hz']:+9.3f}  expected line at {-r['f0_hz']:+9.3f} Hz | notebook axis {r['centre_on_notebook_axis_hz']:+9.3f} "
          f"(err {r['notebook_axis_minus_expected_hz']:+.4f}) | derived axis {r['centre_on_derived_axis_hz']:+9.3f} (err {r['derived_axis_minus_expected_hz']:+.4f})")
print(json.dumps(summary, indent=2))
