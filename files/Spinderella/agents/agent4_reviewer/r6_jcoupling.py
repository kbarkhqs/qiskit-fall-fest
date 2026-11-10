"""C6: is the tolerance set by J itself or by the smallest resolvable line spacing (+ intensities)?
Run: cd agents/agent4_reviewer && ../../.venv/bin/python r6_jcoupling.py

PRE-REGISTERED PREDICTIONS (written before this script was run):
 E1 (A, J02 and J12 shifted together by x in [-4, 8] Hz, J12 - J02 = 6.1 Hz fixed, so mean J of the
    two strong couplings goes 10.85 -> 22.85 Hz while the inner spacing of the spin-2 multiplet stays
    ~6.1 Hz and remains the smallest resolvable spacing):
      H_J (threshold scales with J strength): threshold rises by >= 50 % from x = -4 to x = +8.
      H_s (smallest resolvable spacing): threshold stays within +-10 % of its x = 0 value.
 E2 (A, J12 varied, J02 = 11.8 fixed; inner spacing J12 - J02 = 3.1 ... 10.1 Hz):
      H_s: threshold proportional to the inner spacing while it is the smallest one (ratio constant
      within +-15 %); H_J predicts the same direction, so E2 only calibrates the ratio.
 E3 (A, J01 = 0.9 ... 5 Hz): H_J: threshold rises (or stays). H_s: unchanged while J01 < 2.667 Hz
      (merged), then FALLS once the J01 doublets become characteristic peaks.
 E5 (C, delta difference 0.03 ... 0.5 ppm at J = 7.92 Hz, mean shift fixed): the critical spacing is
      exactly J for every delta (AB quartet), so pure H_J and pure H_s both predict a constant
      threshold; the claim "lowered by the roof effect" predicts a threshold that RISES with
      delta-nu / J (outer lines gain intensity).
Simulation: official setting (Lie-Trotter, 300 x 2.5 ms, official transform), exact dynamics as a
check. Characteristic peaks: exact stick lines of the modified H (weight >= 5 % of max) merged
below 2.667 Hz (validated in r4: identical thresholds to the reference-based ones for A, B, C).
Criterion as in criterion.md (own implementation), dip levels any and 10 %.
"""
import numpy as np

from r_common import (HZ_PER_PPM, GRID_HZ, terms_from_params, rotating_frame, exact_step, trotter_step,
                      trace_correlations, lines_from_step, spectrum_batch, assess, passes, scan_threshold,
                      save)

G = np.round(np.arange(1, 7001) * 0.01, 2)
BASE = {"A": ([5.79, 5.97, 5.48], {(0, 1): 0.9, (0, 2): 11.8, (1, 2): 17.9}),
        "C": ([6.375, 6.302], {(0, 1): 7.92})}


def centres_and_spacing(delta, J):
    terms, n = terms_from_params(delta, J)
    rt, hbar = rotating_frame(terms, n)
    f, c = lines_from_step(exact_step(rt, n, 1e-9), n, 1e-9)
    w = c.real
    keep = w > 0.05 * w.max()
    f, w = f[keep], w[keep]
    o = np.argsort(f)
    f, w = f[o], w[o]
    groups = [[0]]
    for i in range(1, len(f)):
        (groups[-1].append(i) if f[i] - f[groups[-1][-1]] < 2 * GRID_HZ else groups.append([i]))
    cen = np.array([np.sum(f[g] * w[g]) / np.sum(w[g]) for g in groups])
    hgt = np.array([np.sum(w[g]) for g in groups])
    sp = np.diff(cen)
    return np.mean(delta) + cen / HZ_PER_PPM, cen, hgt, sp, rt, n, hbar


def thresholds(delta, J):
    cppm, cen, hgt, sp, rt, n, hbar = centres_and_spacing(delta, J)
    out = {"n_char": len(cen), "min_spacing_Hz": float(sp.min()),
           "outer_inner_ratio": None}
    for label, U in (("trotter", trotter_step(rt, n)), ("exact", exact_step(rt, n))):
        cxx, cyx = trace_correlations(U, n, 300)
        S, p = spectrum_batch(cxx, cyx, G, hbar)
        ev = [assess(s, p, cppm) for s in S]
        for lev_name, lev in (("any", 0.0), ("10%", 0.10)):
            mp, ff, tr = scan_threshold(np.array([passes(f, d, lev) for f, d, _ in ev]), G)
            out[f"{label}_{lev_name}"] = mp
            out[f"{label}_{lev_name}_ntrans"] = len(tr)
        if label == "trotter" and ff is not None:
            # critical pair at the first failing 10 % value
            j = int(round(ff / 0.01)) - 1
            f_, d_, why = ev[j]
            if f_:
                i = int(np.argmin(d_))
                out["critical_spacing_Hz"] = float(cen[i + 1] - cen[i])
            else:
                out["critical_spacing_Hz"] = None
                out["critical_reason"] = why
    return out, cen, hgt


if __name__ == "__main__":
    res = {}
    # E1
    rows = []
    for x in (-4, -2, 0, 2, 4, 6, 8):
        d, J = BASE["A"]
        J = dict(J)
        J[(0, 2)] += x
        J[(1, 2)] += x
        r, cen, hgt = thresholds(d, J)
        r.update({"x": x, "J02": J[(0, 2)], "J12": J[(1, 2)], "mean_J02_J12": 0.5 * (J[(0, 2)] + J[(1, 2)])})
        rows.append(r)
        print("E1 x=%+d J02=%.1f J12=%.1f n=%d minsp=%.2f crit=%s | Trotter any %s 10%% %s | exact any %s 10%% %s" % (
            x, J[(0, 2)], J[(1, 2)], r["n_char"], r["min_spacing_Hz"], r.get("critical_spacing_Hz"),
            r["trotter_any"], r["trotter_10%"], r["exact_any"], r["exact_10%"]))
    res["E1"] = rows
    # E2
    rows = []
    for j12 in (14.9, 15.9, 16.9, 17.9, 18.9, 19.9, 20.9):
        d, J = BASE["A"]
        J = dict(J)
        J[(1, 2)] = j12
        r, cen, hgt = thresholds(d, J)
        r.update({"J12": j12, "inner_spacing_first_order": j12 - 11.8})
        for k in ("trotter_any", "trotter_10%", "exact_any", "exact_10%"):
            r[k + "_over_min_spacing"] = None if r[k] is None else r[k] / r["min_spacing_Hz"]
        rows.append(r)
        print("E2 J12=%.1f minsp=%.2f crit=%s | Trotter any %s (ratio %s) 10%% %s | exact any %s (ratio %s) 10%% %s (ratio %s)" % (
            j12, r["min_spacing_Hz"], r.get("critical_spacing_Hz"), r["trotter_any"],
            None if r["trotter_any"] is None else round(r["trotter_any_over_min_spacing"], 2), r["trotter_10%"],
            r["exact_any"], None if r["exact_any"] is None else round(r["exact_any_over_min_spacing"], 2),
            r["exact_10%"], None if r["exact_10%"] is None else round(r["exact_10%_over_min_spacing"], 2)))
    res["E2"] = rows
    # E3
    rows = []
    for j01 in (0.9, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        d, J = BASE["A"]
        J = dict(J)
        J[(0, 1)] = j01
        r, cen, hgt = thresholds(d, J)
        r.update({"J01": j01})
        rows.append(r)
        print("E3 J01=%.1f n=%d minsp=%.2f crit=%s | Trotter any %s 10%% %s | exact any %s 10%% %s" % (
            j01, r["n_char"], r["min_spacing_Hz"], r.get("critical_spacing_Hz"), r["trotter_any"], r["trotter_10%"],
            r["exact_any"], r["exact_10%"]))
    res["E3"] = rows
    # E5
    rows = []
    for dd in (0.03, 0.05, 0.073, 0.1, 0.15, 0.2, 0.3, 0.5):
        mean = 0.5 * (6.375 + 6.302)
        d = [mean + dd / 2, mean - dd / 2]
        r, cen, hgt = thresholds(d, {(0, 1): 7.92})
        dnu = dd * HZ_PER_PPM
        r.update({"delta_diff_ppm": dd, "dnu_over_J": dnu / 7.92,
                  "outer_inner_ratio": float(hgt[0] / hgt[1]) if len(hgt) == 4 else None})
        rows.append(r)
        print("E5 dd=%.3f ppm dnu/J=%.2f outer/inner=%s n=%d | Trotter any %s 10%% %s | exact any %s 10%% %s" % (
            dd, dnu / 7.92, None if r["outer_inner_ratio"] is None else round(r["outer_inner_ratio"], 3), r["n_char"],
            r["trotter_any"], r["trotter_10%"], r["exact_any"], r["exact_10%"]))
    res["E5"] = rows
    save("r6_jcoupling.json", res)
