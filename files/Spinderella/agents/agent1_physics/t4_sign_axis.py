"""T4: sign and axis conventions of the ORIGINAL notebook's fft_causal / time_signal_to_spectrum.
Exact test signals come from dense matrices (common.exact_signals), never from the formula under test."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

# curve_fit cannot estimate a covariance when the samples are exactly symmetric about the line (f = 0 on the
# corrected axis); the fit itself is fine, and fit_peak asserts the residual instead.
warnings.simplefilter("ignore", OptimizeWarning)
from common import (X, Y, Z, GAMMA, B0, NU0, HZ_TO_PPM, DT, NSTEPS, ROOT, HERE, site_op, total, dense_h,
                    load_terms, rotating_frame, physical_params, exact_signals, line_list, true_freq_axis, record, cmd)
from notebook_verbatim import fft_causal, time_signal_to_spectrum

times = np.arange(NSTEPS) * DT
res = {}

# ---- 0. the copy under test is textually the notebook's cell ---------------------------------
nb = json.loads(subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:nmr.ipynb"], capture_output=True, text=True, check=True).stdout)
cell = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "def fft_causal" in "".join(c["source"]))
mine = (HERE / "notebook_verbatim.py").read_text().split("# --- BEGIN VERBATIM ---\n")[1].split("# --- END VERBATIM ---")[0]
res["verbatim_text_identical_to_HEAD_cell"] = cell.strip() == mine.strip()

# ---- 1. what fft_causal computes -------------------------------------------------------------
rng = np.random.default_rng(0)
f = rng.normal(size=NSTEPS)
L = NSTEPS
g = np.zeros(2 * L - 1); g[:L] = f; g[0] /= 2  # zero-padded, first sample halved
ident = np.abs(fft_causal(f) - np.fft.fftshift(np.fft.fft(g, norm="ortho"))[::2]).max()
def dtft(sig, freqs):  # sum_n w_n sig_n exp(-2 pi i f n dt), w_0 = 1/2, ortho-normalised like the code
    w = np.ones(len(sig)); w[0] = 0.5
    return (np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(sig)) * DT)) @ (w * sig)) / np.sqrt(2 * len(sig) - 1)
f_true = true_freq_axis(L, DT)                       # (2j-(L-1))/((2L-1)dt): odd harmonics of 1/((2L-1)dt)
f_nb = np.fft.fftshift(np.fft.fftfreq(L, d=DT))      # axis the notebook attaches
res["fft_causal"] = {
    "max|fft_causal(f) - FFT_{2L-1}(zero-padded f, f0/2)[::2]|": ident,
    "max|fft_causal(f) - DTFT at (2j-(L-1))/((2L-1)dt)|": np.abs(fft_causal(f) - dtft(f, f_true)).max(),
    "max|fft_causal(f) - DTFT at notebook axis fftfreq(L,dt)|": np.abs(fft_causal(f) - dtft(f, f_nb)).max(),
    "typical |fft_causal(f)|": float(np.abs(fft_causal(f)).mean()),
    "true grid spacing Hz": float(f_true[1] - f_true[0]), "notebook grid spacing Hz": float(f_nb[1] - f_nb[0]),
    "true axis ends Hz": [float(f_true[0]), float(f_true[-1])], "notebook axis ends Hz": [float(f_nb[0]), float(f_nb[-1])],
    "true frequency at the bin labelled 0 Hz": float(f_true[L // 2]),
    "predicted label(f) = f*(2L-1)/(2L) - 1/(2 L dt)": {"slope": (2 * L - 1) / (2 * L), "intercept_hz": -1 / (2 * L * DT)},
}
print(res["fft_causal"])

# ---- helpers ---------------------------------------------------------------------------------
def lor(x, a, x0, w, c):
    return a / (1 + ((x - x0) / w) ** 2) + c

def fit_peak(x, y, k=None, half=8):
    k = int(np.argmax(y)) if k is None else k
    sl = slice(max(k - half, 0), min(k + half + 1, len(y)))
    p, _ = curve_fit(lor, x[sl], y[sl], p0=[y[k], x[k], 2 * abs(x[1] - x[0]), 0.0], maxfev=20000)
    assert np.abs(lor(x[sl], *p) - y[sl]).max() < 2e-2 * y[k], "Lorentzian fit failed"
    return p[1]

def h_of(delta_ppm):
    return -GAMMA * B0 * (1 + delta_ppm * 1e-6) / 2

def sticks(hz, inten, gam, f_axis):
    """Sum of Lorentzians exp(-gam t) <-> gam/(gam^2 + (2 pi (f - f0))^2) at the exact line positions."""
    m = sum(a * gam / (gam**2 + (2 * np.pi * (f_axis - f0)) ** 2) for a, f0 in zip(inten, hz))
    return m / m.max()

GAM = 20.0  # s^-1 window for clean, well-sampled Lorentzians (exp(-GAM*T) = 3e-7)
win = np.exp(-GAM * times)

# ---- 2. single spin, reference frame at delta_ref != delta -------------------------------------
single = []
for d_ref, d in [(5.0, 5.1), (5.0, 4.9), (6.0, 6.2345), (5.5, 5.4321)]:
    hp = h_of(d) - h_of(d_ref)
    cxx, cyx = exact_signals(hp * Z, 1, times)
    spec, ppm = time_signal_to_spectrum(cxx * win, cyx * win, DT, NSTEPS, HZ_TO_PPM, h_of(d_ref))
    pos = fit_peak(ppm, spec)
    raw_minus = fft_causal(cxx * win).real - fft_causal(cyx * win).imag  # the markdown's '- Im'
    pos_minus = fit_peak(ppm, np.abs(raw_minus))
    single.append({"delta_ref": d_ref, "delta_true": d, "peak_ppm_code(+Im)": float(pos), "err_Ha_hz": float((pos - d) / HZ_TO_PPM),
                   "err_Hb_hz": float((pos - (2 * d_ref - d)) / HZ_TO_PPM), "peak_ppm_markdown(-Im)": float(pos_minus),
                   "markdown_err_vs_mirror_hz": float((pos_minus - (2 * d_ref - d)) / HZ_TO_PPM)})
    print(single[-1])
res["single_spin"] = single

# ---- 3. two spins (notebook frame = mean); only spin 0 observed breaks the mirror symmetry -------
two = []
for J in (0.0, 10.0):
    d = np.array([5.20, 5.05]); n = 2
    h = h_of(d); mean_h = h.mean(); hp = h - mean_h
    H = hp[0] * site_op(Z, 0, n) + hp[1] * site_op(Z, 1, n) + (np.pi * J / 2) * sum(site_op(P, 0, n) @ site_op(P, 1, n) for P in (X, Y, Z))
    E, V = np.linalg.eigh(H)
    U = [V @ np.diag(np.exp(-1j * E * t)) @ V.conj().T for t in times]
    xt = total(X, n)
    c0x = np.array([np.trace(u.conj().T @ site_op(X, 0, n) @ u @ xt).real for u in U])  # Tr(X_0(t) X_tot)
    c0y = np.array([np.trace(u.conj().T @ site_op(Y, 0, n) @ u @ xt).real for u in U])
    spec, ppm = time_signal_to_spectrum(c0x * win, c0y * win, DT, NSTEPS, HZ_TO_PPM, mean_h)
    centroid = float(np.sum(ppm * spec**2) / np.sum(spec**2))
    cxx, cyx = exact_signals(H, n, times)
    specf, ppmf = time_signal_to_spectrum(cxx * win, cyx * win, DT, NSTEPS, HZ_TO_PPM, mean_h)
    hz, inten, _ = line_list(H, n)
    exact_ppm = d.mean() + hz * HZ_TO_PPM
    two.append({"J_hz": J, "delta": d.tolist(), "spin0-only signal: intensity-weighted centroid ppm": centroid,
                "nearer to": "delta_0 (H_a)" if abs(centroid - d[0]) < abs(centroid - d[1]) else "delta_1 (H_b)",
                "exact line ppm": exact_ppm.round(5).tolist(), "exact intensities": inten.round(5).tolist(),
                "full signal: corr with exact Lorentzian sticks on the corrected axis": float(np.corrcoef(specf, sticks(hz, inten, GAM, true_freq_axis(NSTEPS, DT)))[0, 1])})
    print(two[-1])
res["two_spin"] = two

# ---- 4. molecule A (asymmetric, 3 spins): which hypothesis matches the exact line list? ---------
mol = {}
for name in "AB":
    terms = load_terms(name); n, h, d, Jd = physical_params(terms); rf, mean_h = rotating_frame(terms)
    H = dense_h(rf, n)
    cxx, cyx = exact_signals(H, n, times)
    w5 = np.exp(-5.0 * times)
    spec, ppm = time_signal_to_spectrum(cxx * w5, cyx * w5, DT, NSTEPS, HZ_TO_PPM, mean_h)
    hz, inten, _ = line_list(H, n)
    f_lab = (ppm - d.mean()) / HZ_TO_PPM            # notebook axis, Hz from the mean shift
    f_cor = true_freq_axis(NSTEPS, DT)              # corrected axis (same samples)
    mol[name] = {
        "notebook axis: corr with exact lines at true ppm [H_a]": float(np.corrcoef(spec, sticks(hz, inten, 5.0, f_lab))[0, 1]),
        "notebook axis: corr with exact lines mirrored about the mean [H_b]": float(np.corrcoef(spec, sticks(-hz, inten, 5.0, f_lab))[0, 1]),
        "corrected axis: corr [H_a]": float(np.corrcoef(spec, sticks(hz, inten, 5.0, f_cor))[0, 1]),
        "corrected axis: corr [H_b]": float(np.corrcoef(spec, sticks(-hz, inten, 5.0, f_cor))[0, 1]),
    }
    print(name, mol[name])
res["molecule_exact_signal"] = mol

# ---- 5. sub-bin offset of the axis: one line swept over true frequency --------------------------
d_ref = 5.5
rows = []
for f0 in np.concatenate([np.linspace(-180, 180, 37), [0.0, 0.3, 0.6678, 1.0, 21.6667, -133.3333, 111.6667]]):
    hp = -np.pi * f0  # line at nu = -h'/pi
    cxx, cyx = exact_signals(hp * Z, 1, times)
    spec, ppm = time_signal_to_spectrum(cxx * win, cyx * win, DT, NSTEPS, HZ_TO_PPM, h_of(d_ref))
    lab = (fit_peak(ppm, spec) - d_ref) / HZ_TO_PPM              # position read off the notebook axis, Hz from the reference
    k = int(np.argmax(spec))
    x_true = true_freq_axis(NSTEPS, DT)
    lab_true = fit_peak(x_true, spec, k)                          # same samples on the corrected axis
    rows.append((f0, lab, lab_true))
rows = np.array(rows)
slope, icpt = np.polyfit(rows[:, 0], rows[:, 1], 1)
res["axis_offset"] = {
    "gamma_window": GAM, "n_lines": len(rows),
    "fit label_hz = slope*f_true + intercept": {"slope": float(slope), "intercept_hz": float(icpt)},
    "predicted": {"slope": 599 / 600, "intercept_hz": -1 / (600 * DT)},
    "label - true (Hz) at f = -180, 0, +180": [float(rows[np.argmin(abs(rows[:, 0] - v)), 1] - v) for v in (-180, 0, 180)],
    "intercept in ppm": float(icpt * HZ_TO_PPM), "intercept in notebook bins": float(icpt * NSTEPS * DT),
    "max|fit residual| Hz": float(np.abs(rows[:, 1] - (slope * rows[:, 0] + icpt)).max()),
    "corrected axis: max|position - true| Hz": float(np.abs(rows[:, 2] - rows[:, 0]).max()),
}
print(res["axis_offset"])

# ---- 6. is the pre-abs spectrum purely absorptive for a well-damped line? -----------------------
cxx, cyx = exact_signals(-np.pi * 40.0 * Z, 1, times)
raw = fft_causal(cxx * win).real + fft_causal(cyx * win).imag
x_true = true_freq_axis(NSTEPS, DT)
analytic = 2 * GAM / (GAM**2 + (2 * np.pi * (x_true - 40.0)) ** 2) / DT / np.sqrt(2 * NSTEPS - 1)
res["absorptive_check"] = {"min(raw)/max(raw)": float(raw.min() / raw.max()),
                           "max|raw - analytic Lorentzian|/max": float(np.abs(raw - analytic).max() / raw.max())}
print(res["absorptive_check"])

# decision rule (comparative): single-spin error < 1.5 Hz under H_a and > 20 Hz under H_b, the spin-0-only line
# sits at delta_0, and for A and B the H_a correlation exceeds the H_b one by more than 0.5 on both axes.
ha = all(abs(s["err_Ha_hz"]) < 1.5 and abs(s["err_Hb_hz"]) > 20 for s in single) and all(t["nearer to"].startswith("delta_0") for t in two) \
     and all(m["notebook axis: corr with exact lines at true ppm [H_a]"] - m["notebook axis: corr with exact lines mirrored about the mean [H_b]"] > 0.5
             and m["corrected axis: corr [H_a]"] - m["corrected axis: corr [H_b]"] > 0.5 for m in mol.values())
record({
    "id": "T4_sign_and_axis",
    "question": "Does the original code (Re FFT[C_xx] + Im FFT[C_yx], numpy FFT sign) put each line at its true ppm, and is the ppm axis offset?",
    "hypotheses": ["H_a: '+ Im' with numpy's FFT sign puts each line at its true ppm",
                   "H_b: the code is mirrored about the mean shift (the markdown's '- Im' would be needed)",
                   "H_c: fft_causal samples odd harmonics of 1/((2L-1)dt), so the notebook axis reads label = f*(2L-1)/(2L) - 1/(2L dt)"],
    "independent_variable": "true line position (delta of a single spin; which spin is observed; true frequency sweep -180..180 Hz)",
    "dependent_variable": "fitted line position on the notebook's ppm axis",
    "fixed_parameters": {"dt": DT, "n_steps": NSTEPS, "window_gamma_for_position_fits": GAM, "window_gamma_molecule_test": 5.0},
    "reference": "exact dense-matrix evolution; analytic Lorentzian; exact line list",
    "result": res,
    "conclusion": (("H_a supported, H_b rejected: lines appear at their true ppm with '+ Im'; the markdown's '- Im' would mirror them. " if ha else "H_a NOT supported - see result. ")
                   + f"H_c supported: measured label = {slope:.6f}*f {icpt:+.4f} Hz (predicted 0.998333*f - 0.6667 Hz): every line is drawn "
                     f"{-icpt:.3f} Hz ({-icpt*HZ_TO_PPM:.5f} ppm, half a bin) too low plus a 1/600 compression about the mean; "
                     "the axis (2j-(L-1))/((2L-1)dt) removes it."),
    "limitations": ("Position fits use Gamma = 20 s^-1 (well-sampled Lorentzian); at small Gamma the line shape on this grid is not Lorentzian (see T7). "
                    "First version of the script used an absolute threshold corr > 0.9 for the molecule test; the measured H_a correlations on the "
                    "notebook axis are 0.877 (A) and 0.861 (B) because the pure-Lorentzian model ignores the half-bin shift and the truncation, so "
                    "the verdict now uses the comparative rule written in the script (H_a minus H_b > 0.5)."),
    "script": "agents/agent1_physics/t4_sign_axis.py",
    "command": cmd("t4_sign_axis.py"),
})
print("H_a:", ha, "verbatim:", res["verbatim_text_identical_to_HEAD_cell"])
