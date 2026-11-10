"""C6: spot-check rows of results/j_coupling_sensitivity.csv (Agent 3) with my own code.
Run: cd agents/agent4_reviewer && ../../.venv/bin/python r6b_spotcheck_agent3.py
"""
import csv
import json

from r_common import REPO, save
from r6_jcoupling import thresholds  # own code (r6), main body guarded

BASE = {"A": ([5.79, 5.97, 5.48], {(0, 1): 0.9, (0, 2): 11.8, (1, 2): 17.9}),
        "B": ([5.58, 5.90, 5.34], {(0, 1): 15.6, (0, 2): -0.5, (1, 2): 8.6}),
        "C": ([6.375, 6.302], {(0, 1): 7.92})}
PAIR = {"J01": (0, 1), "J02": (0, 2), "J12": (1, 2)}
checks = [("A_J02", 9.5), ("A_J02", 13.0), ("A_J02", 16.0), ("A_J12", 20.0), ("A_J01", 3.75),
          ("B_J12", 10.0), ("B_J12", 6.0), ("C_J01", 4.0), ("C_J01", 12.0)]
rows = list(csv.DictReader(open(REPO / "results" / "j_coupling_sensitivity.csv")))
out = []
for exp, jv in checks:
    r = [x for x in rows if x["experiment"] == exp and abs(float(x["J_hz"]) - jv) < 1e-9][0]
    mol, coup = exp.split("_")
    d, J = BASE[mol]
    J = dict(J)
    J[PAIR[coup]] = jv
    mine, _, _ = thresholds(d, J)
    rec = {"experiment": exp, "J": jv,
           "agent3_any": r["threshold_any_dip"], "mine_any": mine["trotter_any"],
           "agent3_10": r["threshold_10pct"], "mine_10": mine["trotter_10%"],
           "agent3_n_char": r["n_char_peaks"], "mine_n_char": mine["n_char"],
           "agent3_monotonic_10": r["monotonic_10pct"], "mine_ntrans_10": mine["trotter_10%_ntrans"]}
    out.append(rec)
    print(rec)
save("r6b_spotcheck.json", out)
