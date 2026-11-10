"""Matching error of every molecule vs its matched reference, and the pure simulation error.

Official simulation: cached signals (300 Lie-Trotter steps, dt 2.5 ms), official spectrum code.
(1) vs the reference file: peak-position error of every characteristic peak (official axis and
    axis corrected for the known sub-bin offset), cosine similarity and NRMSE on the simulation
    grid (reference averaged over each bin), at the notebook default noise 5 s^-1 and at the
    reference's own linewidth (FWHM 1.008 Hz -> Gamma = pi * 1.008 = 3.17 s^-1).
(2) vs exact evolution processed identically (same grid, same window): peak-position, peak-height
    and spectrum error caused by first-order Trotter alone.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agents" / "final_audit"))
from audit_lib import (DT, HZ_PER_PPM, L, char_peaks, exact_signals, load_reference,  # noqa: E402
                       official_spectrum, rotating_frame)

ASSIGN = {"A": 3, "B": 2, "C": 1}
TOL = 2 / (L * DT)
j = np.arange(L)
AXIS_FIX_HZ = (2 * j - (L - 1)) / ((2 * L - 1) * DT) - np.fft.fftshift(np.fft.fftfreq(L, d=DT))


def bin_avg(rp, rs, ppm):
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rs[1:] + rs[:-1]) * np.diff(rp))])
    h = 0.5 * (ppm[1] - ppm[0])
    lo, hi = np.clip(ppm - h, rp[0], rp[-1]), np.clip(ppm + h, rp[0], rp[-1])
    return (np.interp(hi, rp, cum) - np.interp(lo, rp, cum)) / (2 * h)


def peaks_near(spec, ppm, centres):
    mx, _ = find_peaks(spec)
    out = []
    for c in centres:
        near = mx[np.abs(ppm[mx] - c) * HZ_PER_PPM <= TOL]
        out.append(near[np.argmax(spec[near])] if len(near) else None)
    return out


def interp_peak(spec, ppm, i):
    """Sub-bin peak position by a parabola through the maximum and its two neighbours."""
    y0, y1, y2 = spec[i - 1], spec[i], spec[i + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return ppm[i] + d * (ppm[1] - ppm[0]), y1 - 0.25 * (y0 - y2) * d


rows, summary = [], {}
for m in "ABC":
    d = np.load(ROOT / f"results/signals/molecule_{m}.npz")
    cxx, cyx = d["c_xx_raw"], d["c_yx_raw"]
    n, rf, H, mz = rotating_frame(m)
    ex, ey = exact_signals(H, n, DT, L)
    rp, rs = load_reference(ASSIGN[m])
    cen = char_peaks(rp, rs, merge_hz=TOL)
    summ = {"matched_reference": f"spectrum_{ASSIGN[m]}", "characteristic_peaks": int(len(cen))}
    for gamma, tag in ((5.0, "noise 5"), (np.pi * 1.008, "reference linewidth (noise 3.17)")):
        s, p = official_spectrum(cxx, cyx, gamma, DT, L, mz)
        se, _ = official_spectrum(ex, ey, gamma, DT, L, mz)
        p_fix = p + AXIS_FIX_HZ / HZ_PER_PPM
        r_on = bin_avg(rp, rs, p)
        cos = float(s @ r_on / np.linalg.norm(s) / np.linalg.norm(r_on))
        nrmse = float(np.sqrt(np.mean((s - r_on) ** 2)))  # both spectra on a 0..1 scale
        cos_ex = float(s @ se / np.linalg.norm(s) / np.linalg.norm(se))
        nrmse_ex = float(np.sqrt(np.mean((s - se) ** 2)))
        idx, idx_e = peaks_near(s, p, cen), peaks_near(se, p, cen)
        pos_off, pos_fix, pos_tr, h_tr = [], [], [], []
        for c, i, ie in zip(cen, idx, idx_e):
            if i is None or ie is None:
                continue
            x, y = interp_peak(s, p, i)
            xf, _ = interp_peak(s, p_fix, i)
            xe, ye = interp_peak(se, p, ie)
            pos_off.append((x - c) * HZ_PER_PPM)
            pos_fix.append((xf - c) * HZ_PER_PPM)
            pos_tr.append((x - xe) * HZ_PER_PPM)
            h_tr.append((y - ye) / ye)
            rows.append({"molecule": m, "setting": tag, "reference_peak_ppm": round(float(c), 5),
                         "sim_peak_ppm": round(float(x), 5),
                         "position_error_hz_official_axis": round(float((x - c) * HZ_PER_PPM), 3),
                         "position_error_hz_corrected_axis": round(float((xf - c) * HZ_PER_PPM), 3),
                         "trotter_position_error_hz": round(float((x - xe) * HZ_PER_PPM), 3),
                         "trotter_height_error_pct": round(float(100 * (y - ye) / ye), 2)})
        f = lambda a: (round(float(np.mean(np.abs(a))), 3), round(float(np.max(np.abs(a))), 3))
        summ[tag] = {
            "peaks_found": f"{len(pos_off)}/{len(cen)}",
            "vs_reference": {"cosine": round(cos, 4), "1-cosine": round(1 - cos, 4), "rmse_0to1_scale": round(nrmse, 4),
                             "position_error_hz_official_axis_mean_max": f(pos_off),
                             "position_error_hz_corrected_axis_mean_max": f(pos_fix),
                             "mean_signed_position_error_hz_official_axis": round(float(np.mean(pos_off)), 3)},
            "trotter_vs_exact_same_processing": {"cosine": round(cos_ex, 5), "rmse_0to1_scale": round(nrmse_ex, 4),
                                                 "position_error_hz_mean_max": f(pos_tr),
                                                 "peak_height_error_pct_mean_max": tuple(round(100 * v, 2) for v in f(h_tr))},
        }
    summary[m] = summ

out = ROOT / "results"
with open(out / "matching_errors_per_peak.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(out / "matching_errors.json").write_text(json.dumps(summary, indent=1))
for m, sm in summary.items():
    print(f"\n=== molecule {m} -> {sm['matched_reference']} ({sm['characteristic_peaks']} characteristic peaks)")
    for tag in ("noise 5", "reference linewidth (noise 3.17)"):
        v, t = sm[tag]["vs_reference"], sm[tag]["trotter_vs_exact_same_processing"]
        print(f"  [{tag}] peaks {sm[tag]['peaks_found']} | vs reference: cosine {v['cosine']}, RMSE {v['rmse_0to1_scale']}, "
              f"position err official axis mean/max {v['position_error_hz_official_axis_mean_max']} Hz "
              f"(signed mean {v['mean_signed_position_error_hz_official_axis']}), corrected axis {v['position_error_hz_corrected_axis_mean_max']} Hz")
        print(f"  [{tag}] Trotter vs exact: cosine {t['cosine']}, RMSE {t['rmse_0to1_scale']}, position mean/max {t['position_error_hz_mean_max']} Hz, "
              f"height mean/max {t['peak_height_error_pct_mean_max']} %")
