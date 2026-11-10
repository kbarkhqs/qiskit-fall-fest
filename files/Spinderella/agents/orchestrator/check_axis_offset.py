"""Orchestrator check of Agent 1's claim H_c: the notebook ppm axis is offset by about half a bin.

Independent of Agent 1's code: a synthetic single line at a KNOWN offset nu_true (Hz from the mean
shift), C_xx = cos(2 pi nu t), C_yx = sin(2 pi nu t), is pushed through the official
spectrum_from_signals (src/nmr_pipeline.py, verbatim notebook code). The line centre is found by
fitting a parabola to log(spectrum) around the maximum (exact for a Gaussian, near-exact for a
well-sampled Lorentzian at Gamma = 20 s^-1, FWHM 6.4 Hz = 4.8 bins).
Sign: for H = h' Z the Heisenberg equations give C_xx ~ cos(2h't), C_yx ~ +sin(2h't); a line at
nu = -h'/pi Hz (higher ppm for nu > 0) therefore has C_yx = -sin(2 pi nu t).
Competing hypotheses: H0 the axis is exact (error ~ 0); H_c error = -0.6667 Hz - nu/600.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.nmr_pipeline import spectrum_from_signals, hz_to_ppm_conversion  # noqa: E402

Hz_to_ppm_conversion = hz_to_ppm_conversion()

L, dt, G = 300, 0.0025, 20.0
t = np.arange(L) * dt
chem = -1.5708e9  # arbitrary mean Pauli-Z coefficient (rad/s); sets the ppm origin
ppm0 = -(2 * chem / (2 * np.pi / Hz_to_ppm_conversion) + 1e6)  # ppm of zero offset on the notebook axis
rows = []
for nu in (-150.3, -60.0, -10.7, 0.0, 0.4, 25.0, 90.9, 150.0):
    s, ppm = spectrum_from_signals(np.cos(2 * np.pi * nu * t), -np.sin(2 * np.pi * nu * t), G, dt, L, chem)
    ppm = np.asarray(ppm, dtype=float)
    k = int(np.argmax(s))
    y = np.log(s[k - 1:k + 2])
    frac = 0.5 * (y[0] - y[2]) / (y[0] - 2 * y[1] + y[2])
    step = ppm[k + 1] - ppm[k]
    nu_meas = (ppm[k] + frac * step - ppm0) / Hz_to_ppm_conversion
    rows.append((nu, nu_meas, nu_meas - nu, -2 / 3 - nu / 600))
print(f"{'nu_true Hz':>10} {'measured':>10} {'error Hz':>9} {'H_c pred':>9}")
for r in rows:
    print(f"{r[0]:10.2f} {r[1]:10.3f} {r[2]:9.3f} {r[3]:9.3f}")
err = np.array([r[2] for r in rows]); pred = np.array([r[3] for r in rows])
print(f"max |error| = {np.abs(err).max():.3f} Hz ; max |error - H_c| = {np.abs(err - pred).max():.3f} Hz ; bin = {1/(L*dt):.3f} Hz")
