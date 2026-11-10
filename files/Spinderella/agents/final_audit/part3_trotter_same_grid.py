"""Audit part 3: Trotter vs exact on IDENTICAL sampling (same steps and dt), so axis-label and
grid-sampling effects cancel; isolates first-order Trotter error in spectra and thresholds."""
import json
from pathlib import Path
import numpy as np
from audit_lib import (DT, L, char_peaks, evaluate, exact_signals, load_reference, official_spectrum,
                       passes, rotating_frame, signals_state_sum, trotter_unitary)

TOL = 2 * (1 / (L * DT))
ASSIGN = {"A": 3, "B": 2, "C": 1}


def thr(cxx, cyx, mz, cen, dt, steps):
    g = np.round(np.arange(0.01, 60 + 1e-9, 0.01), 2)
    ok = np.array([passes(*evaluate(*official_spectrum(cxx, cyx, x, dt, steps, mz), cen, TOL)[:2], 0.10) for x in g])
    return float(g[ok].max())


out = {}
for m in "ABC":
    n, rf, H, mz = rotating_frame(m)
    rp, rs = load_reference(ASSIGN[m])
    cen = char_peaks(rp, rs, merge_hz=TOL)
    out[m] = {}
    for steps in (300, 600, 1200):
        dt = 0.75 / steps
        tx, ty = signals_state_sum(trotter_unitary(rf, n, dt), n, steps)
        ex, ey = exact_signals(H, n, dt, steps)
        cos = {}
        for g in (5.0, 20.0):
            a, _ = official_spectrum(tx, ty, g, dt, steps, mz)
            b, _ = official_spectrum(ex, ey, g, dt, steps, mz)
            cos[f"noise {g:g}"] = round(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), 5)
        t_tr, t_ex = thr(tx, ty, mz, cen, dt, steps), thr(ex, ey, mz, cen, dt, steps)
        out[m][f"{steps} steps"] = {"threshold_trotter": t_tr, "threshold_exact_same_grid": t_ex,
                                    "difference": round(t_tr - t_ex, 2), "cosine_trotter_vs_exact": cos}
Path("part3_results.json").write_text(json.dumps(out, indent=1))
for m, r in out.items():
    for k, v in r.items():
        print(m, k, v)
