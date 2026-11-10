"""C2 + units. Run: cd agents/agent4_reviewer && ../../.venv/bin/python r2_physics_axis_trotter.py

 a. rotating frame: ||[H, Z_tot]||, and lab-frame lines (full H, ~1.6e9 rad/s) vs rotating-frame lines.
 b. sign: J = 0 copy of A (lines exactly at nu0 (delta_i - mean delta)); official '+ Im' vs '- Im'.
 c. axis: frequencies at which the pipeline actually evaluates the transform (kernel reproduced to
    rounding by r_common.spectrum_batch) vs its fftfreq labels, at every exact line of A, B, C.
 d. Trotter: Floquet lines of the Lie-Trotter step vs exact lines (positions, amplitudes, phase),
    dt, dt/2, dt/4.
 e. units: J = 2c/pi from the files, delta from h, nu0, Hz/ppm, FWHM = Gamma/pi measured.
"""
import numpy as np

from r_common import (GAMMA_1H, FIELD, NU0, HZ_PER_PPM, DT, nmr, parse_json_terms, terms_from_params,
                      rotating_frame, H_matrix, tot, op, trotter_step, exact_step, trace_correlations,
                      official_spectrum, spectrum_batch, _kernel, ppm_axis, local_maxima, save,
                      lines_from_step)


res = {}
# ---------------- e. units ----------------
units = {"nu0_Hz": NU0, "Hz_per_ppm": HZ_PER_PPM}
for m in "ABC":
    terms, n = parse_json_terms(m)
    h = [c for o, c in terms if len(o) == 1]
    J = {tuple(sorted(o)): 2 * c / np.pi for o, c in terms if len(o) == 2 and list(o.values())[0] == "X"}
    delta = [(-2 * hi / (GAMMA_1H * FIELD) - 1) * 1e6 for hi in h]
    same_xyz = all(abs(c - [cc for oo, cc in terms if set(oo) == set(o) and list(oo.values())[0] == "X"][0]) < 1e-12
                   for o, c in terms if len(o) == 2)
    units[m] = {"delta_ppm": delta, "J_Hz": {f"J{i}{j}": v for (i, j), v in J.items()}, "isotropic_XX_YY_ZZ": same_xyz}
# FWHM check: single line, long acquisition (grid 0.01 Hz)
Lf, gam, f0 = 40000, 10.0, 0.0
t = np.arange(Lf) * DT
cxx, cyx = np.cos(2 * np.pi * f0 * t), np.sin(2 * np.pi * f0 * t)
s, p = official_spectrum(cxx, cyx, gam, 0.0, L=Lf)
hz = (p - p[np.argmax(s)]) * HZ_PER_PPM
above = hz[s >= 0.5]
units["FWHM_measured_Hz_at_Gamma10"] = float(above.max() - above.min())
units["FWHM_predicted_Gamma_over_pi"] = gam / np.pi
units["grid_resolution_Hz_for_FWHM"] = float(np.median(np.diff(hz)))
res["units"] = units
print("units:", {k: v for k, v in units.items() if k not in "ABC"})
for m in "ABC":
    print("  ", m, units[m])

# ---------------- a. rotating frame ----------------
rot = {}
for m in "ABC":
    terms, n = parse_json_terms(m)
    H = H_matrix(terms, n)
    Zt = tot(n, "Z")
    comm = np.linalg.norm(H @ Zt - Zt @ H) / np.linalg.norm(H)
    rt, hbar = rotating_frame(terms, n)
    fr, cr = lines_from_step(exact_step(rt, n, 1e-9), n, 1e-9)  # tiny dt: no aliasing
    # lab frame: eigen-decomposition of the full H (frequencies ~ 5e8 Hz)
    E, V = np.linalg.eigh(H)
    X, Y = tot(n, "X"), tot(n, "Y")
    Xe, Ye = V.conj().T @ X @ V, V.conj().T @ Y @ V
    c = Xe * Xe.T - 1j * (Ye * Xe.T)
    fl = (E[:, None] - E[None, :]) / (2 * np.pi)
    keep = np.abs(c) > 1e-3 * np.abs(c).max()
    fl, cl = fl[keep], c[keep]
    # the removed rigid rotation: exp(-i hbar Z_tot t) shifts every line by -2 hbar / 2 pi Hz
    shift_hz = -2 * hbar / (2 * np.pi)
    fl_rot = np.sort(fl - shift_hz)
    fr_s = np.sort(fr)
    dev = float(np.max(np.abs(fl_rot - fr_s))) if len(fl_rot) == len(fr_s) else None
    rot[m] = {"comm_rel_norm": float(comm), "n_lines_lab": int(len(fl)), "n_lines_rot": int(len(fr)),
              "max_line_dev_Hz": dev, "removed_rigid_shift_Hz": shift_hz}
    print(f"rot frame {m}: ||[H,Ztot]||/||H|| = {comm:.1e}; lines lab {len(fl)} rot {len(fr)}; max dev {dev}")
res["rotating_frame"] = rot

# ---------------- b. sign: J = 0 copy of A ----------------
dA = units["A"]["delta_ppm"]
terms0, n = terms_from_params(dA, {})
rt0, hbar0 = rotating_frame(terms0, n)
U0 = exact_step(rt0, n)
cxx0, cyx0 = trace_correlations(U0, n, 300)
sp, pp = official_spectrum(cxx0, cyx0, 1.0, hbar0)
S_minus, _ = spectrum_batch(cxx0, -cyx0, [1.0], hbar0)  # '- Im' is Re FT Cxx - Im FT Cyx
lm = local_maxima(sp)
top = lm[np.argsort(sp[lm])[-3:]]
lm2 = local_maxima(S_minus[0])
top2 = lm2[np.argsort(S_minus[0][lm2])[-3:]]
sign = {"true_ppm": sorted(dA), "plus_Im_peaks_ppm": sorted(pp[top].tolist()),
        "minus_Im_peaks_ppm": sorted(pp[top2].tolist()),
        "mirror_prediction_ppm": sorted((2 * np.mean(dA) - np.array(dA)).tolist())}
print("sign test:", sign)
res["sign"] = sign

# ---------------- c. axis offset ----------------
_, f_eval = _kernel(300, DT)
lab = np.fft.fftshift(np.fft.fftfreq(300, DT))
axis = {"label_minus_eval_fit": np.polyfit(f_eval, lab - f_eval, 1).tolist()}
offs = []
exact_lines = {}
for m in "ABC":
    terms, n = parse_json_terms(m)
    rt, hbar = rotating_frame(terms, n)
    f, c = lines_from_step(exact_step(rt, n, 1e-9), n, 1e-9)
    strong = np.abs(c) > 0.05 * np.abs(c).max()
    f = f[strong]
    exact_lines[m] = (f, c[strong])
    o = lab[np.searchsorted(f_eval, f).clip(0, 299)] - f_eval[np.searchsorted(f_eval, f).clip(0, 299)]
    # continuous model of the offset at the line frequency: label(f) - f
    off = np.interp(f, f_eval, lab - f_eval)
    offs += off.tolist()
    axis[m] = {"line_Hz_min": float(f.min()), "line_Hz_max": float(f.max()),
               "offset_Hz_min": float(off.min()), "offset_Hz_max": float(off.max())}
axis["offset_range_all_Hz"] = [float(min(offs)), float(max(offs))]
print("axis:", axis)
res["axis"] = axis

# ---------------- d. Trotter lines ----------------
trot = {}
for m in "ABC":
    terms, n = parse_json_terms(m)
    rt, hbar = rotating_frame(terms, n)
    row = {}
    for div in (1, 2, 4):
        dt = DT / div
        fe, ce = lines_from_step(exact_step(rt, n, dt), n, dt)
        ft, ct = lines_from_step(trotter_step(rt, n, dt), n, dt)
        se = np.abs(ce) > 0.05 * np.abs(ce).max()
        fe, ce = fe[se], ce[se]
        shifts, amp_rel, phase = [], [], []
        for fi, ci in zip(fe, ce):
            j = np.argmin(np.abs(ft - fi))
            shifts.append(ft[j] - fi)
            amp_rel.append(ct[j].real / ci.real - 1)
            phase.append(np.angle(ct[j] / ci))
        row[f"dt/{div}"] = {"max_abs_shift_Hz": float(np.max(np.abs(shifts))),
                            "max_rel_absorptive_amp_change": float(np.max(np.abs(amp_rel))),
                            "max_abs_phase_rad": float(np.max(np.abs(phase))),
                            "max_imag_over_real_exact": float(np.max(np.abs(ce.imag / ce.real))),
                            "n_lines": int(len(fe))}
        if div == 1:
            row["dt/1"]["shifts_Hz"] = np.round(shifts, 4).tolist()
            row["dt/1"]["rel_amp_change"] = np.round(amp_rel, 4).tolist()
            row["dt/1"]["phase_rad"] = np.round(phase, 4).tolist()
            row["dt/1"]["exact_lines_Hz"] = np.round(fe, 3).tolist()
    trot[m] = row
    print(f"Trotter {m}: " + "; ".join(f"{k}: shift {v['max_abs_shift_Hz']:.3f} Hz, amp {v['max_rel_absorptive_amp_change']:.3f}, phase {v['max_abs_phase_rad']:.3f} rad"
                                       for k, v in row.items()))
res["trotter"] = trot
save("r2_physics.json", res)
