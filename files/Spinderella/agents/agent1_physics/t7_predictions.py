"""T7: pre-registered predictions of the decay rate at which each splitting stops being resolved.

Written by the orchestrator to complete Agent 1's unfinished task T7, BEFORE any noise sweep was run.
Inputs: only the exact line list (exact_lines.json, from exact diagonalisation).  No simulation data.

Model: window exp(-Gamma t) on an infinitely long signal -> absorptive Lorentzian,
       FWHM_Hz = Gamma / pi.  Two equal Lorentzians separated by D Hz keep a dip between them
       while FWHM < sqrt(3) D, i.e. Gamma < pi sqrt(3) D = 5.441 D   (analytic, verified below).
Continuum prediction for the real multiplets: sum of Lorentzians with the exact positions and
intensities; Gamma is swept on a fine grid and every drop in the number of local maxima is recorded
together with the pair of lines that merged.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRID_HZ = 1 / (300 * 0.0025)  # notebook FFT grid spacing, 1.333 Hz


def lorentz_sum(nu, pos, inten, gamma):
    hw = gamma / (2 * np.pi)  # half width at half maximum in Hz
    return sum(a * hw**2 / ((nu - p) ** 2 + hw**2) for p, a in zip(pos, inten))


def maxima(nu, s):
    i = np.where((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]))[0] + 1
    return nu[i]


def analytic_check():
    # equal doublet, D = 10 Hz: dip must vanish at Gamma = pi*sqrt(3)*10 = 54.41
    nu = np.linspace(-60, 60, 120001)
    res = []
    for g in (54.0, 54.8):
        res.append(len(maxima(nu, lorentz_sum(nu, [-5, 5], [1, 1], g))))
    return {"Gamma=54.0 maxima": res[0], "Gamma=54.8 maxima": res[1], "predicted Gamma*": np.pi * np.sqrt(3) * 10}


def sweep(lines, min_int=1e-3):
    pos = np.array([l["hz"] for l in lines if l["intensity"] > min_int])
    inten = np.array([l["intensity"] for l in lines if l["intensity"] > min_int])
    nu = np.arange(pos.min() - 40, pos.max() + 40, 0.005)
    gammas = np.round(np.arange(0.1, 200.0, 0.05), 3)
    prev = None
    events = []
    for g in gammas:
        mx = maxima(nu, lorentz_sum(nu, pos, inten, g))
        if prev is not None and len(mx) < len(prev):
            # which previous maxima disappeared: those whose nearest new maximum is shared
            lost = [p for p in prev if np.min(np.abs(mx - p)) > 1e-9 and True]
            events.append({"gamma_s-1": float(g), "n_maxima_before": len(prev), "n_maxima_after": len(mx),
                           "maxima_before_hz": [round(float(x), 2) for x in prev],
                           "maxima_after_hz": [round(float(x), 2) for x in mx]})
        prev = mx
    return pos, inten, events


def main():
    ex = json.loads((HERE / "exact_lines.json").read_text())["molecules"]
    out = {"model": "continuum Lorentzians, FWHM = Gamma/pi Hz, exact line positions and intensities",
           "equal_doublet_rule": "Gamma* = pi*sqrt(3)*D = 5.441*D (D in Hz)",
           "analytic_check": analytic_check(), "grid_spacing_hz": GRID_HZ, "molecules": {}}
    for m in "ABC":
        pos, inten, events = sweep(ex[m]["lines"])
        order = np.argsort(pos)
        pos, inten = pos[order], inten[order]
        gaps = np.diff(pos)
        splittings = []
        for k, d in enumerate(gaps):
            splittings.append({
                "splitting_hz": round(float(d), 3),
                "lines_involved_hz": [round(float(pos[k]), 3), round(float(pos[k + 1]), 3)],
                "intensities": [round(float(inten[k]), 4), round(float(inten[k + 1]), 4)],
                "gamma_dip_vanishes_equal_pair_rule": round(float(np.pi * np.sqrt(3) * d), 2),
                "resolvable_with_default_grid": bool(d > 2 * GRID_HZ),
            })
        out["molecules"][m] = {"lines_hz": [round(float(p), 3) for p in pos],
                               "intensities": [round(float(a), 4) for a in inten],
                               "adjacent_splittings": splittings,
                               "continuum_merge_events": events}
    (HERE / "predictions.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["analytic_check"]))
    for m in "ABC":
        print(f"\n=== {m}  lines(Hz)={out['molecules'][m]['lines_hz']}")
        for s in out["molecules"][m]["adjacent_splittings"]:
            print(f"   D={s['splitting_hz']:8.3f} Hz  pair-rule Gamma*={s['gamma_dip_vanishes_equal_pair_rule']:7.2f}  grid-resolvable={s['resolvable_with_default_grid']}")
        for e in out["molecules"][m]["continuum_merge_events"]:
            print(f"   merge at Gamma={e['gamma_s-1']:6.2f}: {e['n_maxima_before']}->{e['n_maxima_after']} maxima; after={e['maxima_after_hz']}")


if __name__ == "__main__":
    main()
