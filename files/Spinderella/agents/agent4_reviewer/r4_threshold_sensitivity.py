"""C4: noise-tolerance thresholds, own criterion code, many reasonable variants.
Run: cd agents/agent4_reviewer && ../../.venv/bin/python r4_threshold_sensitivity.py

Official spectra (300 Lie-Trotter steps, dt 2.5 ms) from results/signals, recomputed with my own
vectorised transform (verified equal to nmr_pipeline to < 1e-13) on the grid 0.01 ... 100.00 s^-1
in steps of 0.01 (10 000 points per molecule, no bisection), plus noise 0.
Variants: dip level; position tolerance (0.5, 1, 2, 3 bins); merge distance; reference
peak-finder thresholds; characteristic peaks from the simulation's own low-noise spectrum or
from exact stick lines; no position tolerance (count of maxima per multiplet); signed spectrum
(no final np.abs); dip relative to the larger peak.
"""
import numpy as np

from r_common import (HZ_PER_PPM, GRID_HZ, MATCH, parse_json_terms, rotating_frame, exact_step,
                      load_official_signals, load_ref, spectrum_batch, local_maxima, char_peaks_from,
                      assess, passes, scan_threshold, save, check_batch_equals_pipeline,
                      lines_from_step)

GRID = np.round(np.arange(1, 10001) * 0.01, 2)
LEVELS = {"any": 0.0, "5%": 0.05, "10%": 0.10, "19%": 0.19}
MOLS = "ABC"

data = {}
for m in MOLS:
    s = load_official_signals(m)
    dev = check_batch_equals_pipeline(s["cxx"], s["cyx"], s["shift"])
    S, p = spectrum_batch(s["cxx"], s["cyx"], np.concatenate([[0.0], GRID]), s["shift"])
    Ss, _ = spectrum_batch(s["cxx"], s["cyx"], np.concatenate([[0.0], GRID]), s["shift"], signed=True)
    assert np.all(np.diff(p) > 0)
    rp, rs = load_ref(MATCH[m])
    terms, n = parse_json_terms(m)
    rt, hbar = rotating_frame(terms, n)
    fl, cl = lines_from_step(exact_step(rt, n, 1e-9), n, 1e-9)
    data[m] = dict(S=S, Ss=Ss, p=p, ref=(rp, rs), dev=dev, lines=(fl, cl.real), hbar=hbar,
                   lm=[local_maxima(x) for x in S], lms=[local_maxima(x) for x in Ss])
    print(m, "own transform vs pipeline max dev %.1e" % dev)


def stick_centres(m, merge_hz=2 * GRID_HZ, rel=0.05):
    f, w = data[m]["lines"]
    keep = w > rel * w.max()
    f, w = f[keep], w[keep]
    o = np.argsort(f)
    f, w = f[o], w[o]
    groups = [[0]]
    for i in range(1, len(f)):
        (groups[-1].append(i) if f[i] - f[groups[-1][-1]] < merge_hz else groups.append([i]))
    cen_hz = np.array([np.sum(f[g] * w[g]) / np.sum(w[g]) for g in groups])
    return np.mean(MOLDELTA[m]) + cen_hz / HZ_PER_PPM


MOLDELTA = {}
for m in MOLS:
    terms, n = parse_json_terms(m)
    MOLDELTA[m] = [(-2 * c / (267_522_128.0 * 11.7433) - 1) * 1e6 for o, c in terms if len(o) == 1]


def multiplet_windows(centres, gap_hz=25.0, pad_hz=2 * GRID_HZ):
    groups = [[centres[0]]]
    for c in centres[1:]:
        (groups[-1].append(c) if (c - groups[-1][-1]) * HZ_PER_PPM <= gap_hz else groups.append([c]))
    return [(g[0] - pad_hz / HZ_PER_PPM, g[-1] + pad_hz / HZ_PER_PPM, len(g)) for g in groups]


def count_pass(s, p, lm, windows, level, exact=False):
    for lo, hi, k in windows:
        idx = [i for i in lm if lo <= p[i] <= hi]
        if not idx:
            return False
        top = max(s[i] for i in idx)
        idx = [i for i in idx if s[i] >= 0.05 * top]  # ignore ringing bumps < 5 % of the multiplet
        # merge adjacent maxima whose separating dip is below the level (keep the higher one)
        while len(idx) > 1:
            dips = [1 - s[a:b + 1].min() / min(s[a], s[b]) for a, b in zip(idx[:-1], idx[1:])]
            j = int(np.argmin(dips))
            if (level == 0 and dips[j] > 0) or (level > 0 and dips[j] >= level):
                break
            a, b = idx[j], idx[j + 1]
            idx.pop(j + 1 if s[a] >= s[b] else j)
        if (len(idx) != k) if exact else (len(idx) < k):
            return False
    return True


def run_variant(m, centres, level, tol_bins=2.0, signed=False, mode="position", dipref="min"):
    d = data[m]
    S = d["Ss"] if signed else d["S"]
    LM = d["lms"] if signed else d["lm"]
    p = d["p"]
    flags = []
    if mode.startswith("count"):
        win = multiplet_windows(centres)
        for s, lm in zip(S, LM):
            flags.append(count_pass(s, p, lm, win, level, exact=(mode == "count_exact")))
    else:
        for s, lm in zip(S, LM):
            found, dips, _ = assess(s, p, centres, tol_hz=tol_bins * GRID_HZ, lm=lm)
            if found and dipref == "max":
                # recompute dips relative to the larger maximum
                pl = p[lm]
                ch = [lm[np.abs(pl - c) * HZ_PER_PPM <= tol_bins * GRID_HZ][np.argmax(s[lm[np.abs(pl - c) * HZ_PER_PPM <= tol_bins * GRID_HZ]])] for c in centres]
                dips = np.array([1 - s[a:b + 1].min() / max(s[a], s[b]) for a, b in zip(ch[:-1], ch[1:])])
            flags.append(passes(found, dips, level))
    flags = np.array(flags)
    at0 = bool(flags[0])
    f = flags[1:]
    mp, ff, tr = scan_threshold(f, GRID)
    tr01 = [t for t in tr if t[0] >= 0.1]
    return {"max_pass": mp, "first_fail": ff, "pass_at_0": at0, "transitions": tr,
            "n_transitions_0.01_100": len(tr), "n_transitions_0.1_100": len(tr01),
            "monotonic_0.1_100": len(tr01) == 1 and tr01[0][2] == "pass->fail" and bool(f[GRID >= 0.1][0])}


variants = {}
cent = {}
for m in MOLS:
    rp, rs = data[m]["ref"]
    cent[m] = {
        "ref default (h0.05 p0.02 merge 2 bins)": char_peaks_from(rp, rs),
        "ref merge 0.5 bin": char_peaks_from(rp, rs, merge_hz=0.5 * GRID_HZ),
        "ref merge 1 bin": char_peaks_from(rp, rs, merge_hz=1 * GRID_HZ),
        "ref merge 3 bins": char_peaks_from(rp, rs, merge_hz=3 * GRID_HZ),
        "ref h0.02 p0.005": char_peaks_from(rp, rs, height=0.02, prom=0.005),
        "ref h0.10 p0.05": char_peaks_from(rp, rs, height=0.10, prom=0.05),
        "ref h0.20 p0.10": char_peaks_from(rp, rs, height=0.20, prom=0.10),
        "ref h0.30 p0.10": char_peaks_from(rp, rs, height=0.30, prom=0.10),
        "sim own spectrum at 1 s^-1": char_peaks_from(data[m]["p"], data[m]["S"][GRID.tolist().index(1.0) + 1]),
        "sim own spectrum at 2 s^-1": char_peaks_from(data[m]["p"], data[m]["S"][GRID.tolist().index(2.0) + 1]),
        "sim own spectrum at 5 s^-1": char_peaks_from(data[m]["p"], data[m]["S"][GRID.tolist().index(5.0) + 1]),
        "exact stick lines (merge 2 bins)": stick_centres(m),
    }

summary = {m: {k: {"n": len(v), "ppm": v.tolist()} for k, v in cent[m].items()} for m in MOLS}
for m in MOLS:
    base = cent[m]["ref default (h0.05 p0.02 merge 2 bins)"]
    for k, v in cent[m].items():
        diff = (float(np.max(np.abs(v - base))) * HZ_PER_PPM) if len(v) == len(base) else None
        summary[m][k]["max_shift_vs_default_Hz"] = diff
    print(m, {k: (len(v), None if summary[m][k]["max_shift_vs_default_Hz"] is None else round(summary[m][k]["max_shift_vs_default_Hz"], 2)) for k, v in cent[m].items()})

specs = []
for lev in LEVELS:
    specs.append((f"PRIMARY-type: ref default, tol 2 bins, dip {lev}", "ref default (h0.05 p0.02 merge 2 bins)", lev, 2.0, False, "position", "min"))
for tb in (0.5, 1.0, 3.0):
    for lev in ("any", "10%"):
        specs.append((f"tol {tb} bins, dip {lev}", "ref default (h0.05 p0.02 merge 2 bins)", lev, tb, False, "position", "min"))
for ck in list(cent["A"])[1:]:
    for lev in ("any", "10%"):
        specs.append((f"centres: {ck}, dip {lev}", ck, lev, 2.0, False, "position", "min"))
for lev in LEVELS:
    specs.append((f"signed spectrum (no np.abs), dip {lev}", "ref default (h0.05 p0.02 merge 2 bins)", lev, 2.0, True, "position", "min"))
for lev in LEVELS:
    specs.append((f"no position tolerance: count >= k per multiplet, dip {lev}", "ref default (h0.05 p0.02 merge 2 bins)", lev, None, False, "count", "min"))
for lev in ("any", "10%"):
    specs.append((f"no position tolerance: count == k per multiplet, dip {lev}", "ref default (h0.05 p0.02 merge 2 bins)", lev, None, False, "count_exact", "min"))
for lev in ("5%", "10%", "19%"):
    specs.append((f"dip relative to larger peak, dip {lev}", "ref default (h0.05 p0.02 merge 2 bins)", lev, 2.0, False, "position", "max"))

rows = []
for name, ck, lev, tb, signed, mode, dipref in specs:
    row = {"variant": name}
    for m in MOLS:
        r = run_variant(m, cent[m][ck], LEVELS[lev], tol_bins=tb if tb else 2.0, signed=signed, mode=mode, dipref=dipref)
        row[m] = r
    th = {m: row[m]["max_pass"] for m in MOLS}
    if all(v is not None for v in th.values()):
        row["ranking"] = " < ".join(sorted(MOLS, key=lambda m: th[m]))
    else:
        row["ranking"] = "undefined (some molecule never passes)"
    rows.append(row)
    print("%-78s A %6s  B %6s  C %6s  | %s | transitions(0.1-100) A%d B%d C%d | pass@0 %s" % (
        name, th["A"], th["B"], th["C"], row["ranking"],
        row["A"]["n_transitions_0.1_100"], row["B"]["n_transitions_0.1_100"], row["C"]["n_transitions_0.1_100"],
        "".join("T" if row[m]["pass_at_0"] else "F" for m in MOLS)))

# critical pair for the primary (10 %) at its first failing value, own logic
crit = {}
for m in MOLS:
    c = cent[m]["ref default (h0.05 p0.02 merge 2 bins)"]
    r = [x for x in rows if x["variant"].startswith("PRIMARY") and x["variant"].endswith("10%")][0][m]
    for lab, g in (("10%", r["first_fail"]),):
        j = GRID.tolist().index(g) + 1
        found, dips, why = assess(data[m]["S"][j], data[m]["p"], c, lm=data[m]["lm"][j])
        if found:
            i = int(np.argmin(dips))
            crit[m] = {"first_fail": g, "pair_ppm": [c[i], c[i + 1]], "ref_sep_Hz": (c[i + 1] - c[i]) * HZ_PER_PPM,
                       "dip": float(dips[i]), "all_dips": dips.tolist()}
        else:
            crit[m] = {"first_fail": g, "reason": why}
    # any-dip critical
    r = [x for x in rows if x["variant"].startswith("PRIMARY") and x["variant"].endswith("any")][0][m]
    g = r["first_fail"]
    j = GRID.tolist().index(g) + 1
    found, dips, why = assess(data[m]["S"][j], data[m]["p"], c, lm=data[m]["lm"][j])
    if found:
        i = int(np.argmin(dips))
        crit[m]["any_first_fail"] = g
        crit[m]["any_pair_ppm"] = [c[i], c[i + 1]]
        crit[m]["any_ref_sep_Hz"] = (c[i + 1] - c[i]) * HZ_PER_PPM
    else:
        # identify which centre lost its maximum
        lm = data[m]["lm"][j]
        miss = [cc for cc in c if not np.any(np.abs(data[m]["p"][lm] - cc) * HZ_PER_PPM <= 2 * GRID_HZ)]
        crit[m]["any_first_fail"] = g
        crit[m]["any_reason"] = why
        crit[m]["any_missing_ppm"] = miss
print("critical:", crit)

# J01 doublets of A: can the 300-step acquisition ever show them? (official grid and zero-padded)
from r_common import trace_correlations, trotter_step, _kernel
terms, n = parse_json_terms("A")
rt, hbar = rotating_frame(terms, n)
doublets_hz = [(15.52, 16.46), (27.31, 28.25), (102.59, 103.53), (120.46, 121.40)]
j01 = {}
S0 = data["A"]["S"][0]
p = data["A"]["p"]
lm0 = data["A"]["lm"][0]
cnt = []
for a, b in doublets_hz:
    lo, hi = np.mean(MOLDELTA["A"]) + (a - 1.0) / HZ_PER_PPM, np.mean(MOLDELTA["A"]) + (b + 1.0) / HZ_PER_PPM
    cnt.append(int(np.sum((p[lm0] >= lo) & (p[lm0] <= hi))))
j01["official_noise0_maxima_per_doublet_window"] = cnt
# zero-padded exact 300-step signal (0.05 Hz grid), noise 0: information limit of T = 0.75 s
U = exact_step(rt, n)
cxx, cyx = trace_correlations(U, n, 300)
t = np.arange(300) * 0.0025
fgrid = np.arange(-160, 160, 0.05)
g = np.ones(300); g[0] = 0.5
K = np.exp(-2j * np.pi * np.outer(fgrid, t)) * g
Sf = (K @ cxx).real + (K @ cyx).imag
cntz = []
for a, b in doublets_hz:
    win = (fgrid >= a - 1.0) & (fgrid <= b + 1.0)
    s = Sf[win]
    cntz.append(int(np.sum((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]))))
j01["zero_padded_exact_300steps_noise0_maxima_per_doublet"] = cntz
# longer acquisition: how far would the doublets survive (exact dynamics, 1200 and 2400 steps)?
for L in (1200, 2400):
    cxx, cyx = trace_correlations(U, n, L)
    res_l = []
    for gam in np.round(np.arange(0.0, 8.0, 0.05), 2):
        S, pp = spectrum_batch(cxx, cyx, [gam], hbar, L=L)
        lm = local_maxima(S[0])
        ok = True
        for a, b in doublets_hz:
            lo, hi = np.mean(MOLDELTA["A"]) + (a - 0.3) / HZ_PER_PPM, np.mean(MOLDELTA["A"]) + (b + 0.3) / HZ_PER_PPM
            if np.sum((pp[lm] >= lo) & (pp[lm] <= hi)) < 2:
                ok = False
        res_l.append((gam, ok))
    oks = [x for x in res_l if x[1]]
    j01[f"exact_{L}_steps_all_4_doublets_resolved_up_to"] = max(x[0] for x in oks) if oks else None
print("J01 doublets:", j01)

save("r4_thresholds.json", {"centres": summary, "variants": rows, "critical": crit, "J01": j01,
                            "transform_dev": {m: data[m]["dev"] for m in MOLS}})
