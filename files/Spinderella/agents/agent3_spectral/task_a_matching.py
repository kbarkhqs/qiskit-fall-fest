"""Task A: global one-to-one molecule-to-spectrum matching with several metrics.

Run from the repository root:
    .venv/bin/python agents/agent3_spectral/task_a_matching.py

Resampling (documented): each reference is normalised to its own maximum over its full
window, then resampled onto the simulation's ppm grid by bin averaging (mean of the linearly
interpolated reference over each 1.333 Hz simulation bin; zero outside the reference window).
A point-interpolation variant (np.interp, zero outside) is reported as a robustness check.
Metrics are evaluated over the full simulation window; the overlap fraction is reported.
"""

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from common import (  # noqa: E402
    AGENT, HZ_PER_PPM, MOLS, OUT, PLOTS, REFS, TOL_HZ, characteristic_peaks, dump_json,
    load_official, multiplets, overlap_fraction, ref_bin_average, reference, spectrum,
)

NOISES = (0.5, 2.0, 5.0)
PRIMARY_NOISE = 2.0
SIM_PROM = 0.05  # simulated peak prominence (fraction of max) for the peak metric


def match_peaks(sim_p, sim_h, ref_p, ref_h, tol_hz=TOL_HZ):
    """Greedy one-to-one matching by distance (closest pairs first) within tol_hz."""
    pairs = []
    for i, rp in enumerate(ref_p):
        for j, sp in enumerate(sim_p):
            d = abs(sp - rp) * HZ_PER_PPM
            if d <= tol_hz:
                pairs.append((d, i, j))
    pairs.sort()
    ui, uj, out = set(), set(), []
    for d, i, j in pairs:
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        out.append((i, j, (sim_p[j] - ref_p[i]) * HZ_PER_PPM, sim_h[j] - ref_h[i]))
    return out


def multiplet_score(sim_m, ref_m, tol_hz=TOL_HZ):
    """Fraction of multiplets (max of the two counts) that agree in centre, line count and
    every internal splitting within tol_hz."""
    used, good = set(), 0
    for rm in ref_m:
        for j, sm in enumerate(sim_m):
            if j in used:
                continue
            if abs(sm["centre_ppm"] - rm["centre_ppm"]) * HZ_PER_PPM > tol_hz:
                continue
            if sm["n_lines"] != rm["n_lines"]:
                continue
            if any(abs(a - b) > tol_hz for a, b in zip(sm["splittings_hz"], rm["splittings_hz"])):
                continue
            used.add(j)
            good += 1
            break
    denom = max(len(sim_m), len(ref_m))
    return good / denom if denom else 0.0


def describe(ms):
    return "; ".join(
        f"{m['n_lines']}@{m['centre_ppm']:.4f}[" + ",".join(f"{s:.2f}" for s in m["splittings_hz"]) + "]"
        for m in ms
    )


def main():
    sigs = {m: load_official(m) for m in MOLS}
    refs = {k: reference(k) for k in REFS}
    ref_groups = {k: characteristic_peaks(*refs[k]) for k in REFS}
    rows = []
    for g in NOISES:
        for m in MOLS:
            s, p = spectrum(sigs[m], g)
            o = np.argsort(p)
            s, p = s[o], p[o]
            pk, _ = find_peaks(s, prominence=SIM_PROM)
            sim_p, sim_h = p[pk], s[pk]
            sim_m = multiplets(sim_p)
            for k in REFS:
                rp, rs = refs[k]
                gr = ref_groups[k]
                r_p = np.array([x["ppm"] for x in gr])
                r_h = np.array([x["height"] for x in gr])
                ref_m = multiplets(r_p)
                mt = match_peaks(sim_p, sim_h, r_p, r_h)
                frac = len(mt) / len(r_p)
                prec = len(mt) / len(sim_p) if len(sim_p) else 0.0
                f1 = 2 * frac * prec / (frac + prec) if frac + prec > 0 else 0.0
                for resamp in ("binavg", "point"):
                    if resamp == "binavg":
                        r = ref_bin_average(rp, rs, p)
                    else:
                        r = np.interp(p, rp, rs, left=0.0, right=0.0)
                    cos = float(s @ r / (np.linalg.norm(s) * np.linalg.norm(r))) if np.linalg.norm(r) > 0 else 0.0
                    pr = float(np.corrcoef(s, r)[0, 1]) if np.std(r) > 0 else 0.0
                    nrmse = float(np.sqrt(np.mean((s - r) ** 2)))
                    rows.append(
                        {
                            "molecule": m, "reference": f"spectrum_{k}", "noise_strength": g,
                            "resampling": resamp, "overlap_fraction": overlap_fraction(rp, p),
                            "cosine": cos, "pearson": pr, "nrmse": nrmse,
                            "n_ref_char_peaks": len(r_p), "n_sim_peaks": len(sim_p),
                            "peak_match_fraction": frac, "peak_precision": prec, "peak_f1": f1,
                            "mean_peak_error_hz": float(np.mean([abs(x[2]) for x in mt])) if mt else np.nan,
                            "mean_signed_peak_error_hz": float(np.mean([x[2] for x in mt])) if mt else np.nan,
                            "mean_abs_height_diff": float(np.mean([abs(x[3]) for x in mt])) if mt else np.nan,
                            "multiplet_score": multiplet_score(sim_m, ref_m),
                            "sim_multiplets": describe(sim_m), "ref_multiplets": describe(ref_m),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "matching_score_matrix.csv", index=False)

    # Permutations
    metrics = {  # name -> (column, maximize?)
        "cosine": ("cosine", True), "pearson": ("pearson", True), "nrmse": ("nrmse", False),
        "peak_match_fraction": ("peak_match_fraction", True), "peak_f1": ("peak_f1", True),
        "multiplet_score": ("multiplet_score", True),
    }
    perms = list(itertools.permutations(REFS))
    prow, summary = [], []
    for g in NOISES:
        for resamp in ("binavg", "point"):
            sub = df[(df.noise_strength == g) & (df.resampling == resamp)].set_index(["molecule", "reference"])
            for name, (col, mx) in metrics.items():
                if resamp == "point" and name in ("peak_match_fraction", "peak_f1", "multiplet_score"):
                    continue  # peak metrics do not depend on resampling
                totals = []
                for pm in perms:
                    tot = sum(sub.loc[(m, f"spectrum_{k}"), col] for m, k in zip(MOLS, pm))
                    totals.append(tot)
                    prow.append({"noise_strength": g, "resampling": resamp, "metric": name,
                                 "permutation": " ".join(f"{m}->{k}" for m, k in zip(MOLS, pm)),
                                 "total": tot})
                order = np.argsort(totals)[::-1] if mx else np.argsort(totals)
                best, second = order[0], order[1]
                tied = np.isclose(totals[best], totals[second])
                summary.append({
                    "noise_strength": g, "resampling": resamp, "metric": name,
                    "best": " ".join(f"{m}->{k}" for m, k in zip(MOLS, perms[best])),
                    "best_total": totals[best],
                    "second": " ".join(f"{m}->{k}" for m, k in zip(MOLS, perms[second])),
                    "second_total": totals[second],
                    "margin": abs(totals[best] - totals[second]), "tie": bool(tied),
                })
    pd.DataFrame(prow).to_csv(AGENT / "matching_permutations.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(AGENT / "matching_permutation_summary.csv", index=False)
    winners = set(sm[~sm.tie].best)
    print(sm.to_string())
    consensus = sm.best.mode()[0]
    mapping = dict(x.split("->") for x in consensus.split())
    unanimous = len(set(sm.best)) == 1 and not sm.tie.any()

    # Summary CSV (primary noise, bin-average resampling)
    out = []
    sub = df[(df.noise_strength == PRIMARY_NOISE) & (df.resampling == "binavg")].set_index(["molecule", "reference"])
    for m in MOLS:
        r = sub.loc[(m, f"spectrum_{mapping[m]}")]
        others = sub.loc[m].drop(f"spectrum_{mapping[m]}")
        mc = "pass" if r.multiplet_score == 1.0 else f"partial ({r.multiplet_score:.2f})"
        mc += f": sim {r.sim_multiplets} | ref {r.ref_multiplets}"
        out.append({
            "molecule": f"molecule {m}", "best_reference_spectrum": f"spectrum_{mapping[m]}",
            "cosine_similarity": r.cosine, "pearson_r": r.pearson, "nrmse": r.nrmse,
            "peak_match_fraction": r.peak_match_fraction, "mean_peak_error_hz": r.mean_peak_error_hz,
            "multiplet_check": mc,
            "confidence": ("high" if unanimous else "medium")
            + f" (all {len(sm)} metric/noise/resampling choices pick the same permutation: {unanimous}; "
            f"best other cosine {others.cosine.max():.3f})",
        })
    pd.DataFrame(out).to_csv(OUT / "molecule_spectrum_matching.csv", index=False)
    print(pd.DataFrame(out).to_string())

    # Plots
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=True)
    for i, m in enumerate(MOLS):
        s, p = spectrum(sigs[m], PRIMARY_NOISE)
        for j, k in enumerate(REFS):
            ax = axes[i, j]
            rp, rs = refs[k]
            ax.plot(rp, rs, color="0.55", lw=1.0, label=f"reference spectrum_{k}")
            ax.plot(p, s, color="C1" if mapping[m] == str(k) else "C0", lw=1.2, ls="--",
                    label=f"molecule {m} (noise {PRIMARY_NOISE})")
            lo, hi = min(p.min(), rp.min()), max(p.max(), rp.max())
            ax.set_xlim(hi, lo)
            c = sub.loc[(m, f"spectrum_{k}")]
            ax.set_title(f"{m} vs spectrum_{k}: cos {c.cosine:.2f}, peaks {c.peak_match_fraction:.2f}",
                         fontsize=9, fontweight="bold" if mapping[m] == str(k) else "normal")
            ax.axvspan(p.min(), p.max(), color="C0", alpha=0.05)
            if i == 2:
                ax.set_xlabel("ppm")
            if j == 0:
                ax.set_ylabel("normalised intensity")
            ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("Every molecule vs every reference (shaded: simulation window; orange = chosen match)")
    fig.tight_layout()
    fig.savefig(PLOTS / "matching_grid.png", dpi=110)
    plt.close(fig)

    for m in MOLS:
        k = int(mapping[m])
        rp, rs = refs[k]
        fig, ax = plt.subplots(2, 1, figsize=(10, 6.5))
        for a, mode in zip(ax, ["full", "zoom"]):
            a.plot(rp, rs, color="0.3", lw=1.2, label=f"reference spectrum_{k} (HQS, normalised)")
            for g, c in zip(NOISES, ("C2", "C1", "C0")):
                s, p = spectrum(sigs[m], g)
                a.plot(p, s, color=c, lw=1.0, ls="--", label=f"simulation, noise {g} 1/s")
            s, p = spectrum(sigs[m], PRIMARY_NOISE)
            if mode == "full":
                a.set_xlim(p.max(), p.min())
            else:
                # zoom on the reference multiplet that contains the smallest characteristic spacing
                gr = characteristic_peaks(rp, rs)
                cen = np.array([x["ppm"] for x in gr])
                mults = multiplets(cen)
                tight = min(mults, key=lambda q: min(q["splittings_hz"]) if q["splittings_hz"] else 1e9)
                a.set_xlim(max(tight["lines_ppm"]) + 0.012, min(tight["lines_ppm"]) - 0.012)
                a.set_title(f"zoom: multiplet at {tight['centre_ppm']:.4f} ppm "
                            f"(splittings {', '.join(f'{x:.2f}' for x in tight['splittings_hz'])} Hz)", fontsize=9)
            a.set_xlabel("ppm")
            a.set_ylabel("normalised intensity")
            a.legend(fontsize=7)
        sub_row = sub.loc[(m, f"spectrum_{k}")]
        ax[0].set_title(f"molecule {m} vs spectrum_{k}: cosine {sub_row.cosine:.3f}, "
                        f"peak match {sub_row.peak_match_fraction:.2f}, "
                        f"mean peak error {sub_row.mean_peak_error_hz:.2f} Hz (noise {PRIMARY_NOISE})")
        fig.tight_layout()
        fig.savefig(PLOTS / f"overlay_molecule_{m}.png", dpi=110)
        plt.close(fig)

    # Leaked-file consistency check (AFTER the decision)
    import hashlib

    from common import REPO

    h = lambda f: hashlib.md5((REPO / "files" / f).read_bytes()).hexdigest()  # noqa: E731
    leak_same_as = [k for k in REFS if h(f"spectrum_{k}.pkl") == h("hqs_spectrum_tools_molecule C_B11.7433T.pkl")]
    result = {
        "mapping": {f"molecule {m}": f"spectrum_{mapping[m]}" for m in MOLS},
        "unanimous": bool(unanimous), "n_choices": int(len(sm)), "winners": sorted(winners),
        "min_margin_by_metric": sm.groupby("metric").margin.min().to_dict(),
        "leaked_C_file_identical_to": [f"spectrum_{k}" for k in leak_same_as],
        "agrees_with_leaked_file": bool(leak_same_as and mapping["C"] == str(leak_same_as[0])),
        "summary": sm.to_dict(orient="records"),
    }
    dump_json(result, AGENT / "task_a_result.json")
    print(result["mapping"], "unanimous", unanimous, "leak check", result["agrees_with_leaked_file"])


if __name__ == "__main__":
    main()
