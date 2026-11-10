"""C3: matching robustness. Run: cd agents/agent4_reviewer && ../../.venv/bin/python r3_matching.py

For every (noise, metric) choice: 3x3 score matrix, all 6 one-to-one assignments ranked by the
summed score, winner and margin (best - second best, also relative to best). File names are not
used; the byte copy of spectrum_1 is only checked at the end.
Metrics (own code): cosine with linear / nearest / bin-average resampling of the reference onto the
simulation grid (0 outside the reference window); cosine restricted to the overlap window;
cosine on the reference grid (simulation interpolated); Pearson on the overlap; negative L1 of
area-normalised spectra on a common grid; peak-position-only score (symmetric F1 with 3 Hz
tolerance) and negative capped chamfer distance; plus an axis-corrected variant (+0.667 Hz).
"""
import hashlib
from itertools import permutations

import numpy as np

from r_common import (REPO, HZ_PER_PPM, GRID_HZ, load_official_signals, load_ref, spectrum_batch,
                      local_maxima, prominence, char_peaks_from, save)

NOISES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
MOLS = "ABC"
REFS = (1, 2, 3)
refs = {k: load_ref(k) for k in REFS}


def resample(rp, rs, grid, kind):
    inside = (grid >= rp[0]) & (grid <= rp[-1])
    out = np.zeros_like(grid)
    if kind == "linear":
        out[inside] = np.interp(grid[inside], rp, rs)
    elif kind == "nearest":
        idx = np.clip(np.searchsorted(rp, grid[inside]), 1, len(rp) - 1)
        left = rp[idx - 1]
        right = rp[idx]
        pick = np.where(grid[inside] - left < right - grid[inside], idx - 1, idx)
        out[inside] = rs[pick]
    elif kind == "binavg":
        cum = np.concatenate([[0], np.cumsum(0.5 * (rs[1:] + rs[:-1]) * np.diff(rp))])
        h = 0.5 * np.median(np.diff(grid))
        lo = np.clip(grid - h, rp[0], rp[-1])
        hi = np.clip(grid + h, rp[0], rp[-1])
        out = (np.interp(hi, rp, cum) - np.interp(lo, rp, cum)) / (2 * h)
    return out, inside


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def peaks_sim(s, p):
    s = s / s.max()
    lm = local_maxima(s)
    return np.array([p[i] for i in lm if s[i] >= 0.1 and prominence(s, i) >= 0.05])


def scores(sim_s, sim_p, k, kind):
    rp, rs = refs[k]
    if kind in ("linear", "nearest", "binavg"):
        r, _ = resample(rp, rs, sim_p, kind)
        return cos(sim_s, r)
    if kind == "overlap_cos":
        r, inside = resample(rp, rs, sim_p, "linear")
        return cos(sim_s[inside], r[inside]) if inside.sum() >= 10 else 0.0
    if kind == "overlap_pearson":
        r, inside = resample(rp, rs, sim_p, "linear")
        if inside.sum() < 10:
            return -1.0
        return float(np.corrcoef(sim_s[inside], r[inside])[0, 1])
    if kind == "ref_grid_cos":
        inside = (rp >= sim_p[0]) & (rp <= sim_p[-1])
        si = np.zeros_like(rp)
        si[inside] = np.interp(rp[inside], sim_p, sim_s)
        return cos(si, rs)
    if kind == "neg_L1_area":
        lo, hi = min(sim_p[0], rp[0]), max(sim_p[-1], rp[-1])
        g = np.linspace(lo, hi, 20000)
        a = np.interp(g, sim_p, sim_s, left=0, right=0)
        b = np.interp(g, rp, rs, left=0, right=0)
        a, b = a / np.trapezoid(a, g), b / np.trapezoid(b, g)
        return -float(np.trapezoid(np.abs(a - b), g))
    if kind in ("peak_F1", "neg_chamfer"):
        ps = peaks_sim(sim_s, sim_p)
        pr = char_peaks_from(rp, rs)
        if kind == "peak_F1":
            if len(ps) == 0 or len(pr) == 0:
                return 0.0
            rec = np.mean([np.min(np.abs(ps - c)) * HZ_PER_PPM <= 3.0 for c in pr])
            pre = np.mean([np.min(np.abs(pr - c)) * HZ_PER_PPM <= 3.0 for c in ps])
            return 0.0 if rec + pre == 0 else float(2 * rec * pre / (rec + pre))
        d1 = np.mean([min(np.min(np.abs(ps - c)) * HZ_PER_PPM, 100) for c in pr])
        d2 = np.mean([min(np.min(np.abs(pr - c)) * HZ_PER_PPM, 100) for c in ps])
        return -float(0.5 * (d1 + d2))
    raise ValueError(kind)


KINDS = ["linear", "nearest", "binavg", "overlap_cos", "overlap_pearson", "ref_grid_cos",
         "neg_L1_area", "peak_F1", "neg_chamfer"]
sig = {m: load_official_signals(m) for m in MOLS}
rows = []
for shift_hz in (0.0, 2.0 / 3.0):
    for g in NOISES:
        spec = {}
        for m in MOLS:
            S, p = spectrum_batch(sig[m]["cxx"], sig[m]["cyx"], [g], sig[m]["shift"])
            spec[m] = (S[0], p + shift_hz / HZ_PER_PPM)
        for kind in KINDS:
            M = {(m, k): scores(*spec[m], k, kind) for m in MOLS for k in REFS}
            tot = sorted(((sum(M[m, k] for m, k in zip(MOLS, perm)), perm) for perm in permutations(REFS)),
                         reverse=True)
            best, second = tot[0], tot[1]
            margin = best[0] - second[0]
            # per-molecule: is the claimed reference also the row-wise best?
            rowbest = {m: max(REFS, key=lambda k: M[m, k]) for m in MOLS}
            rows.append({"axis_shift_Hz": shift_hz, "noise": g, "metric": kind,
                         "winner": "A->%d B->%d C->%d" % best[1], "best_total": best[0],
                         "second": "A->%d B->%d C->%d" % second[1], "second_total": second[0],
                         "margin": margin, "rel_margin": margin / max(abs(best[0]), 1e-12),
                         "rowwise_best": rowbest,
                         "matrix": {f"{m}{k}": M[m, k] for m in MOLS for k in REFS}})
winners = sorted({r["winner"] for r in rows})
worst = min(rows, key=lambda r: r["rel_margin"])
print("choices:", len(rows), "distinct winners:", winners)
print("worst relative margin: %.3f (%s, noise %s, axis shift %.3f Hz): best %s %.4f vs second %s %.4f" %
      (worst["rel_margin"], worst["metric"], worst["noise"], worst["axis_shift_Hz"], worst["winner"],
       worst["best_total"], worst["second"], worst["second_total"]))
for kind in KINDS:
    rs = [r for r in rows if r["metric"] == kind]
    w = min(rs, key=lambda r: r["rel_margin"])
    print("  %-16s winners %s  min rel margin %.3f (noise %s, shift %.2f); second there: %s" %
          (kind, sorted({r['winner'] for r in rs}), w["rel_margin"], w["noise"], w["axis_shift_Hz"], w["second"]))
rowwise_disagree = [r for r in rows if r["rowwise_best"] != {"A": 3, "B": 2, "C": 1}]
print("row-wise best differs from the assignment in", len(rowwise_disagree), "of", len(rows), "choices")
for r in rowwise_disagree[:8]:
    print("   ", r["metric"], r["noise"], r["axis_shift_Hz"], r["rowwise_best"], {k: round(v, 3) for k, v in r["matrix"].items()})
md5 = {f: hashlib.md5(open(REPO / "files" / f, "rb").read()).hexdigest()
       for f in ("spectrum_1.pkl", "spectrum_2.pkl", "spectrum_3.pkl", "hqs_spectrum_tools_molecule C_B11.7433T.pkl")}
print("md5:", md5)
save("r3_matching.json", {"rows": rows, "winners": winners, "worst": worst, "md5": md5,
                          "n_rowwise_disagree": len(rowwise_disagree),
                          "rowwise_disagree_examples": rowwise_disagree[:20]})
