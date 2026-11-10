"""Task B: maximum tolerable noise_strength per molecule (criterion in criterion.md, v1).

Run from the repository root:
    .venv/bin/python agents/agent3_spectral/task_b_noise.py
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    AGENT, DIP_LEVELS, DIP_PRIMARY, HZ_PER_PPM, MOLS, OUT, PLOTS, TOL_HZ, characteristic_peaks,
    dump_json, evaluate, exact_signals, load_official, nmr, passes, reference, scan, signals_dict,
    spectrum, threshold, transitions,
)

MATCH = {"A": 3, "B": 2, "C": 1}
PRIMARY_KEY = "10pct"
COARSE = [0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35,
          37.5, 40, 45, 50, 60, 70, 80, 90, 100]
CACHE = AGENT / "cache"
CACHE.mkdir(exist_ok=True)


def cached_signals(m, kind, nsteps):
    """kind: 'trotter' (simulate_signals_fast) or 'exact' (dense exact diagonalisation)."""
    f = CACHE / f"{m}_{kind}_{nsteps}.npz"
    ham, n, shift = nmr.load_rotating_frame_hamiltonian(m)
    if f.exists():
        z = np.load(f)
        ex, ey = z["exp_x"], z["exp_y"]
    else:
        t0 = time.time()
        if kind == "trotter":
            ex, ey = nmr.simulate_signals_fast(ham, n, nsteps, 0.0025)
        else:
            ex, ey = exact_signals(ham, n, nsteps, 0.0025)
        np.savez(f, exp_x=ex, exp_y=ey)
        print(f"  computed {m} {kind} {nsteps} in {time.time() - t0:.1f} s")
    return signals_dict(ex, ey, shift, n)


def main():
    assert DIP_LEVELS[PRIMARY_KEY] == DIP_PRIMARY
    out_rows, coarse_rows, dense_rows, ev_out = [], [], [], {}
    official = {m: load_official(m) for m in MOLS}
    # consistency: fast Trotter 300 == official
    for m in MOLS:
        s300 = cached_signals(m, "trotter", 300)
        assert np.allclose(s300["c_xx"], official[m]["c_xx"]) and np.allclose(s300["c_yx"], official[m]["c_yx"])
    for m in MOLS:
        k = MATCH[m]
        rp, rs = reference(k)
        groups = characteristic_peaks(rp, rs)
        sig = official[m]
        # lowest-noise check
        s01, p01 = spectrum(sig, 0.1)
        ev01 = evaluate(s01, p01, groups)
        print(f"molecule {m}: {len(groups)} char peaks; detected at 0.1: {ev01['found']} {ev01['reason']}; "
              f"dips at 0.1: {np.round(ev01['dips'], 3).tolist()}")

        # coarse sweep with details
        res, evs = scan(sig, groups, COARSE)
        for j, g in enumerate(COARSE):
            ev = evs[j]
            coarse_rows.append({
                "molecule": m, "noise_strength": g, "found": ev["found"], "reason": ev["reason"],
                **{f"pass_{kk}": bool(res[kk][j]) for kk in DIP_LEVELS},
                "min_dip": float(np.nanmin(ev["dips"])) if ev["found"] else np.nan,
                "dips": json.dumps([round(x, 4) for x in ev["dips"]]),
                "peak_errors_hz": json.dumps([None if q["err_hz"] is None else round(q["err_hz"], 3) for q in ev["peaks"]]),
                "peak_heights": json.dumps([None if q["height"] is None else round(q["height"], 3) for q in ev["peaks"]]),
            })
        coarse_tr = {kk: transitions(np.array(COARSE), res[kk]) for kk in DIP_LEVELS}

        # dense 0.1 grid monotonicity check
        dense = np.round(np.arange(0.1, 100.0 + 1e-9, 0.1), 6)
        resd, evd = scan(sig, groups, dense)
        dense_tr = {kk: transitions(dense, resd[kk]) for kk in DIP_LEVELS}
        for j, g in enumerate(dense):
            dense_rows.append({"molecule": m, "noise_strength": g, "found": evd[j]["found"],
                               "reason": evd[j]["reason"],
                               "min_dip": float(np.nanmin(evd[j]["dips"])) if evd[j]["found"] else np.nan,
                               **{f"pass_{kk}": bool(resd[kk][j]) for kk in DIP_LEVELS}})

        # thresholds (dense scan + 0.01 refinement)
        th = {kk: threshold(sig, groups, kk) for kk in DIP_LEVELS}
        prim = th[PRIMARY_KEY]
        ff = prim["first_fail"]
        evf = evaluate(*spectrum(sig, ff), groups)
        evp = evaluate(*spectrum(sig, prim["max_pass"]), groups)
        if evf["found"]:
            i = int(np.nanargmin(evf["dips"]))
            crit = evf["pairs"][i]
            crit_desc = f"dip {evf['dips'][i]:.4f} < {DIP_PRIMARY} at {ff}"
        else:
            i = None
            crit = None
            crit_desc = evf["reason"]

        # sensitivity checks
        sens = {}
        for label, kind, nst in (("exact_300", "exact", 300), ("trotter_600", "trotter", 600),
                                 ("trotter_1200", "trotter", 1200), ("exact_1200", "exact", 1200)):
            sg = cached_signals(m, kind, nst)
            sens[label] = {kk: threshold(sg, groups, kk) for kk in DIP_LEVELS}

        ev_out[m] = {
            "reference": f"spectrum_{k}", "n_char_peaks": len(groups),
            "char_peaks_ppm": [g["ppm"] for g in groups],
            "lowest_noise_check": {"found": ev01["found"], "reason": ev01["reason"], "dips": ev01["dips"]},
            "coarse_transitions": coarse_tr, "dense_0p1_transitions": dense_tr,
            "thresholds": th, "critical_pair": crit, "critical_index": i, "critical_desc": crit_desc,
            "eval_at_max_pass": evp, "eval_at_first_fail": evf, "sensitivity": sens,
        }
        f2 = lambda d: None if d is None else d  # noqa: E731
        out_rows.append({
            "molecule": f"molecule {m}", "matched_reference": f"spectrum_{k}",
            "max_tolerable_noise_strength": prim["max_pass"], "first_failing_noise_strength": ff,
            "search_resolution": 0.01,
            "critical_peaks_ppm": "n/a" if crit is None else f"{crit[0]:.5f};{crit[1]:.5f}",
            "critical_splitting_hz": np.nan if crit is None else round(crit[2], 3),
            "criterion": "criterion.md v1: every characteristic reference peak (components <2.667 Hz merged) "
                         "has its own local maximum within +-2.667 Hz, adjacent dips >= 10% (primary)",
            "threshold_any_dip": f2(th["any_dip"]["max_pass"]), "threshold_5pct": f2(th["5pct"]["max_pass"]),
            "threshold_19pct": f2(th["19pct"]["max_pass"]),
            "threshold_exact_signals": sens["exact_300"][PRIMARY_KEY]["max_pass"],
            "threshold_600_steps": sens["trotter_600"][PRIMARY_KEY]["max_pass"],
            "threshold_1200_steps": sens["trotter_1200"][PRIMARY_KEY]["max_pass"],
            "evidence": "agents/agent3_spectral/task_b_result.json; coarse_sweep.csv; dense_scan.csv",
        })
        print(f"  primary threshold {prim['max_pass']} first fail {ff} critical {crit} ({crit_desc}); "
              f"monotonic(fine) {prim['monotonic']} transitions {prim['transitions_fine']}")
        for kk in DIP_LEVELS:
            print(f"   {kk:8s} official {th[kk]['max_pass']}/{th[kk]['first_fail']}  "
                  + "  ".join(f"{lab} {sens[lab][kk]['max_pass']}" for lab in sens)
                  + f"  | coarse tr {coarse_tr[kk]} | dense tr {dense_tr[kk]}")

    pd.DataFrame(out_rows).to_csv(OUT / "noise_tolerance.csv", index=False)
    pd.DataFrame(coarse_rows).to_csv(AGENT / "coarse_sweep.csv", index=False)
    pd.DataFrame(dense_rows).to_csv(AGENT / "dense_scan.csv", index=False)
    dump_json(ev_out, AGENT / "task_b_result.json")
    make_plots(official, ev_out)


def make_plots(official, ev_out):
    for m in MOLS:
        k = MATCH[m]
        rp, rs = reference(k)
        groups = characteristic_peaks(rp, rs)
        cen = np.array([g["ppm"] for g in groups])
        e = ev_out[m]
        gstar = e["thresholds"][PRIMARY_KEY]["max_pass"]
        gfail = e["thresholds"][PRIMARY_KEY]["first_fail"]
        # stacked sweep
        noises = sorted({0.5, 2, 5, 10, 15, gstar, gfail, round(gstar * 1.5, 2), round(gstar * 2, 2)})
        fig, ax = plt.subplots(figsize=(11, 9))
        off = 0.0
        step = 1.15
        ax.plot(rp, rs + step * len(noises), color="k", lw=1.0)
        ax.text(cen.max() + 0.03, step * len(noises) + 0.4, f"reference spectrum_{k}", fontsize=8)
        for j, g in enumerate(noises):
            s, p = spectrum(official[m], g)
            ok = passes(evaluate(s, p, groups), DIP_PRIMARY)
            y0 = step * (len(noises) - 1 - j)
            ax.plot(p, s + y0, color="C2" if ok else "C3", lw=1.0)
            ax.text(cen.max() + 0.03, y0 + 0.4, f"noise {g} 1/s: {'PASS' if ok else 'FAIL'}", fontsize=8,
                    color="C2" if ok else "C3")
        for c in cen:
            ax.axvline(c, color="0.85", lw=0.5, zorder=0)
        ax.set_xlim(cen.max() + 0.06, cen.min() - 0.02)
        ax.set_yticks([])
        ax.set_xlabel("ppm")
        ax.set_title(f"molecule {m}: official 300-step spectra vs noise_strength (primary criterion, 10% dip); "
                     f"threshold {gstar} 1/s", fontsize=10)
        fig.tight_layout()
        fig.savefig(PLOTS / f"noise_sweep_molecule_{m}.png", dpi=110)
        plt.close(fig)

        # threshold zoom on critical pair
        crit = e["critical_pair"]
        if crit is None:
            ci = 0
            lo_c, hi_c = cen[0], cen[0]
        else:
            lo_c, hi_c = crit[0], crit[1]
        pad = 14.0 / HZ_PER_PPM
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(rp, rs, color="k", lw=1.2, label=f"reference spectrum_{k}")
        for g, c, ls in ((gstar, "C2", "-"), (gfail, "C3", "--"), (round(0.8 * gstar, 2), "C0", ":")):
            s, p = spectrum(official[m], g)
            ev = evaluate(s, p, groups)
            ok = passes(ev, DIP_PRIMARY)
            sel = (p > lo_c - pad) & (p < hi_c + pad)
            ii = None if crit is None else e["critical_index"]
            dtxt = "" if ii is None or not ev["found"] else f", critical dip {ev['dips'][ii] * 100:.2f}%"
            ax.plot(p[sel], s[sel], color=c, ls=ls, marker=".", ms=4, lw=1.2,
                    label=f"simulation noise {g} 1/s: {'PASS' if ok else 'FAIL'}{dtxt}")
        for c in cen:
            if lo_c - pad < c < hi_c + pad:
                ax.axvspan(c - TOL_HZ / HZ_PER_PPM, c + TOL_HZ / HZ_PER_PPM, color="0.9", zorder=0)
                ax.axvline(c, color="0.6", lw=0.6)
        ax.set_xlim(hi_c + pad, lo_c - pad)
        ax.set_xlabel("ppm")
        ax.set_ylabel("normalised intensity")
        cs = "n/a" if crit is None else f"{crit[2]:.2f} Hz"
        ax.set_title(f"molecule {m}: critical pair ({cs}) just below/above threshold; grey = +-2 bin tolerance",
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS / f"threshold_molecule_{m}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    import sys

    if "--plots-only" in sys.argv:
        make_plots({m: load_official(m) for m in MOLS}, json.loads((AGENT / "task_b_result.json").read_text()))
    else:
        main()
