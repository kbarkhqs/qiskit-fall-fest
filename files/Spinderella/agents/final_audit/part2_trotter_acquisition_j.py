"""Audit part 2: Trotter convergence at fixed T = 0.75 s (item 3), acquisition length and the
0.9 Hz J01 splitting (item 4), controlled J experiments re-run independently (item 7), and the
weighting question at the spectrum level (item 5)."""
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from audit_lib import (DT, HZ_PER_PPM, L, char_peaks, evaluate, exact_lines, exact_signals, fft_causal,
                       lines_from_unitary, load_reference, official_spectrum, passes, ppm_offset,
                       rotating_frame, signals_state_sum, stick_reference, sum_ops, trotter_unitary,
                       x_basis_states)

OUT = Path(__file__).resolve().parent
ASSIGN = {"A": 3, "B": 2, "C": 1}  # from part 1 (independent matching)
TOL = 2 * (1 / (L * DT))
res = {}


def threshold(cxx, cyx, mz, centres, dt, steps, level=0.10, lo=0.01, hi=60.0):
    grid = np.round(np.arange(lo, hi + 1e-9, 0.01), 2)
    ok = np.array([passes(*evaluate(*official_spectrum(cxx, cyx, g, dt, steps, mz), centres, TOL)[:2], level) for g in grid])
    ch = np.where(ok[1:] != ok[:-1])[0]
    return (float(grid[ok].max()) if ok.any() else None), int(len(ch)), bool(ok[0])


def match_lines(nu_e, w_e, nu_t, w_t):
    """Pair every exact line with the nearest Trotter line; return max |d nu|, max |dw|/w, max |dw|/max w."""
    dn, dw_rel, dw_max = [], [], []
    for x, a in zip(nu_e, w_e):
        j = np.argmin(np.abs(nu_t - x))
        dn.append(nu_t[j] - x)
        dw_rel.append((w_t[j] - a) / a)
        dw_max.append((w_t[j] - a) / w_e.max())
    i = int(np.argmax(np.abs(dw_rel)))
    return {"max_abs_line_shift_hz": float(np.max(np.abs(dn))),
            "max_abs_intensity_change_relative_to_line": float(np.max(np.abs(dw_rel))),
            "line_with_largest_intensity_change_hz": float(nu_e[i]),
            "max_abs_intensity_change_relative_to_strongest_line": float(np.max(np.abs(dw_max)))}


# ------------------------------------------------------------------ item 3: Trotter convergence at fixed T
item3 = {}
for m in "ABC":
    n, rf, H, mz = rotating_frame(m)
    rp, rs = load_reference(ASSIGN[m])
    cen = char_peaks(rp, rs, merge_hz=TOL)
    nu_e, w_e = exact_lines(H, n)
    ex_cxx, ex_cyx = exact_signals(H, n, DT, L)
    s_ex, p_ex = official_spectrum(ex_cxx, ex_cyx, 5.0, DT, L, mz)
    rows = {}
    for steps in (300, 600, 1200):
        dt = 0.75 / steps
        U = trotter_unitary(rf, n, dt)
        nu_t, w_t = lines_from_unitary(U, n, dt)
        cxx, cyx = signals_state_sum(U, n, steps)
        s, p = official_spectrum(cxx, cyx, 5.0, dt, steps, mz)
        s_on = np.interp(p_ex, p, s)
        cos = float(s_on @ s_ex / np.linalg.norm(s_on) / np.linalg.norm(s_ex))
        thr = threshold(cxx, cyx, mz, cen, dt, steps)
        rows[f"{steps} steps, dt = {dt * 1e3:.4g} ms"] = {**match_lines(nu_e, w_e, nu_t, w_t),
                                                         "cosine_vs_exact_spectrum_noise5": cos,
                                                         "threshold_10pct": thr[0], "n_changes": thr[1]}
    rows["exact evolution (300 points, dt 2.5 ms)"] = {"threshold_10pct": threshold(ex_cxx, ex_cyx, mz, cen, DT, L)[0]}
    item3[m] = rows
res["item3_trotter_convergence_fixed_T"] = item3

# ------------------------------------------------------------------ item 4: acquisition length, J01 doublets of A
item4 = {}
n, rf, H, mz = rotating_frame("A")
nu_e, w_e = exact_lines(H, n)
U = trotter_unitary(rf, n, DT)
nu_t, w_t = lines_from_unitary(U, n, DT)
# the four J01 doublets: adjacent strong lines closer than 1.5 Hz
def doublets(nu, w):
    s = nu[w > 0.05 * w.max()]
    return [(a, b) for a, b in zip(s[:-1], s[1:]) if b - a < 1.5]
dbl = {"exact": doublets(nu_e, w_e), "trotter": doublets(nu_t, w_t)}
rp, rs = load_reference(3)
cen_official = char_peaks(rp, rs, merge_hz=TOL)


def doublet_resolved(spec, f_true, pairs, level):
    """Both lines of every doublet are separate local maxima (within 0.47 Hz of the dynamics' own
    line position, at most one maximum per line) with a dip >= level (any dip if level = 0)."""
    mx, _ = find_peaks(spec)
    for a, b in pairs:
        ia = mx[np.abs(f_true[mx] - a) <= 0.47]
        ib = mx[np.abs(f_true[mx] - b) <= 0.47]
        if len(ia) == 0 or len(ib) == 0:
            return False
        i1, i2 = ia[np.argmax(spec[ia])], ib[np.argmax(spec[ib])]
        if i2 <= i1:
            return False
        dip = 1 - spec[i1:i2 + 1].min() / min(spec[i1], spec[i2])
        if (level == 0 and dip <= 0) or (level > 0 and dip < level):
            return False
    return True


for T in (0.75, 1.5, 3.0):
    steps = int(round(T / DT))
    j = np.arange(steps)
    f_true = (2 * j - (steps - 1)) / ((2 * steps - 1) * DT)  # frequencies fft_causal evaluates
    row = {"grid_spacing_hz": 1 / T}
    for label, (cxx, cyx), pairs in (("trotter", signals_state_sum(U, n, steps), dbl["trotter"]),
                                     ("exact", exact_signals(H, n, DT, steps), dbl["exact"])):
        best = {}
        for level in (0.0, 0.10):
            ok = []
            for g in np.round(np.arange(0.0, 12.0 + 1e-9, 0.01), 2):
                t = np.arange(steps) * DT
                w = np.exp(-g * t)
                s = fft_causal(cxx * w).real + fft_causal(cyx * w).imag
                s = np.abs(s / np.max(np.abs(s)))
                ok.append((g, doublet_resolved(s, f_true, pairs, level)))
            passing = [g for g, o in ok if o]
            best["any_dip" if level == 0 else "10pct"] = max(passing) if passing else None
            best[("any_dip" if level == 0 else "10pct") + "_resolved_at_0"] = ok[0][1]
        thr = threshold(cxx, cyx, mz, cen_official, DT, steps)
        row[label] = {"J01_doublets_resolved_up_to_gamma": best, "official_criterion_threshold_10pct (doublets merged)": thr[0]}
    item4[f"T = {T} s ({steps} steps)"] = row
item4["J01_doublet_spacings_hz"] = {k: [round(b - a, 4) for a, b in v] for k, v in dbl.items()}
res["item4_acquisition_and_J01"] = item4

# ------------------------------------------------------------------ item 7: controlled J experiments, independent
item7 = []
first_order = {"A": lambda J: J[(1, 2)] - J[(0, 2)], "B": lambda J: J[(0, 1)] - J[(1, 2)], "C": lambda J: J[(0, 1)]}
base_J = {"A": {(0, 1): 0.9, (0, 2): 11.8, (1, 2): 17.9}, "B": {(0, 1): 15.6, (0, 2): -0.5, (1, 2): 8.6}, "C": {(0, 1): 7.92}}
plan = [("A", (0, 2), [9.5, 10.5, 11.8, 13.0, 14.0]), ("B", (1, 2), [8.6, 9.5, 10.5, 11.5, 12.5]),
        ("C", (0, 1), [4.0, 6.0, 7.92, 10.0, 12.0])]
notebook_values = {"A": [28.31, 24.40, 19.97, 15.42, 12.75], "B": [22.85, 18.71, 15.72, 11.72, 8.05],
                   "C": [11.19, 17.67, 23.22, 27.51, 32.00]}
for m, pair, values in plan:
    for v, nbv in zip(values, notebook_values[m]):
        n, rf, H, mz = rotating_frame(m, {pair: v})
        nu, w = exact_lines(H, n)
        rp, rs = stick_reference(nu, w, ppm_offset(mz))
        cen = char_peaks(rp, rs, merge_hz=TOL)
        U = trotter_unitary(rf, n, DT)
        cxx, cyx = signals_state_sum(U, n, L)
        thr, nch, ok0 = threshold(cxx, cyx, mz, cen, DT, L)
        J = dict(base_J[m]); J[pair] = v
        # critical pair = adjacent characteristic peaks with the smallest dip just above the threshold
        s, p = official_spectrum(cxx, cyx, thr + 0.01, DT, L, mz)
        f, d, _ = evaluate(s, p, cen, TOL)
        k = int(np.argmin(d)) if f and len(d) else None
        crit = float((cen[k + 1] - cen[k]) * HZ_PER_PPM) if k is not None else None
        item7.append({"molecule": m, "coupling": f"J{pair[0]}{pair[1]}", "J_hz": v, "threshold_independent": thr,
                      "threshold_notebook": nbv, "n_changes": nch, "critical_spacing_hz (exact lines)": crit,
                      "first_order_formula_hz": round(abs(first_order[m](J)), 3)})
res["item7_J_experiments"] = item7

# ------------------------------------------------------------------ item 5: weightings at the spectrum level
item5w = {}
for m in "AC":
    n, rf, H, mz = rotating_frame(m)
    U = trotter_unitary(rf, n, DT)
    X, Y = sum_ops(n)
    bits, S = x_basis_states(n)
    psi = S.T.copy()
    ex = np.zeros((len(bits), L)); ey = np.zeros_like(ex)
    for k in range(L):
        ex[:, k] = np.real(np.einsum("ij,ij->j", psi.conj(), X @ psi))
        ey[:, k] = np.real(np.einsum("ij,ij->j", psi.conj(), Y @ psi))
        psi = U @ psi
    wts = {"N-2w": np.array([n - 2 * sum(b) for b in bits], float),
           "N-w": np.array([n - sum(b) for b in bits], float),
           "-(N-2w)": -np.array([n - 2 * sum(b) for b in bits], float),
           "all-plus only": np.eye(len(bits))[0],
           "equal (1)": np.ones(len(bits))}
    ref_s, _ = official_spectrum(wts["N-2w"] @ ex, wts["N-2w"] @ ey, 5.0, DT, L, mz)
    row = {}
    for k, wv in wts.items():
        cx, cy = wv @ ex, wv @ ey
        if max(np.abs(cx).max(), np.abs(cy).max()) < 1e-8:
            row[k] = "signal is rounding noise (max |C| = %.1e); rejected by the guard" % max(np.abs(cx).max(), np.abs(cy).max())
            continue
        s, _ = official_spectrum(cx, cy, 5.0, DT, L, mz)
        row[k] = float(np.abs(s - ref_s).max())
    item5w[m] = row
res["item5_weightings_normalised_spectrum_max_difference"] = item5w

(OUT / "part2_results.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
