"""C4/C6: does the ranking of A, B, C survive (i) the dip level and (ii) removing Trotter error?
Run: cd agents/agent4_reviewer && ../../.venv/bin/python r5b_ranking.py

Thresholds with the stated criterion (reference characteristic peaks, +-2.667 Hz, adjacent dips) at
four dip levels, for: official Lie-Trotter 300 steps; exact dynamics 300 steps; exact 2400 steps.
Gamma grid 0.01 (300 steps: 0.01..60; 2400 steps: 10.00..45.00, monotonicity recorded).
"""
import numpy as np

from r_common import (MATCH, parse_json_terms, rotating_frame, exact_step, trotter_step, trace_correlations,
                      load_ref, char_peaks_from, assess, passes, scan_threshold, spectrum_batch,
                      official_spectrum, save)

LEVELS = {"any": 0.0, "5%": 0.05, "10%": 0.10, "19%": 0.19}
res = {}
for m in "ABC":
    terms, n = parse_json_terms(m)
    rt, hbar = rotating_frame(terms, n)
    centres = char_peaks_from(*load_ref(MATCH[m]))
    res[m] = {}
    for label, U, L in (("trotter_300", trotter_step(rt, n), 300), ("exact_300", exact_step(rt, n), 300),
                        ("exact_2400", exact_step(rt, n), 2400)):
        cxx, cyx = trace_correlations(U, n, L)
        if L == 300:
            G = np.round(np.arange(1, 6001) * 0.01, 2)
            S, p = spectrum_batch(cxx, cyx, G, hbar, L=300)
        else:
            G = np.round(np.arange(1000, 4501) * 0.01, 2)
            S = []
            for g in G:
                s, p = official_spectrum(cxx, cyx, g, hbar, L=L)
                S.append(s)
        ev = [assess(s, p, centres) for s in S]
        for lev, d in LEVELS.items():
            mp, ff, tr = scan_threshold(np.array([passes(f, dd, d) for f, dd, _ in ev]), G)
            res[m][f"{label}|{lev}"] = {"max_pass": mp, "n_transitions": len(tr)}
rank = {}
for key in res["A"]:
    th = {m: res[m][key]["max_pass"] for m in "ABC"}
    rank[key] = {"thresholds": th, "ranking": " < ".join(sorted("ABC", key=lambda m: th[m])),
                 "B_minus_C": th["B"] - th["C"], "A_minus_C": th["A"] - th["C"]}
    print("%-18s A %6.2f  B %6.2f  C %6.2f   %s   (transitions %s)" % (
        key, th["A"], th["B"], th["C"], rank[key]["ranking"], [res[m][key]["n_transitions"] for m in "ABC"]))
save("r5b_ranking.json", {"per_molecule": res, "ranking": rank})
