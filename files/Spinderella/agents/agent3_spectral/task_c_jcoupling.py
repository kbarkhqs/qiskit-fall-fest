"""Task C: relation between the maximum tolerable noise and the J couplings.

Part 1: couplings and adjacent line spacings of each reference multiplet, first-order
        attribution of every spacing (which coupling combination produces it).
Part 2: observational comparison (n = 3, correlation only).
Part 3: ARTIFICIAL controlled experiments: one coupling varied at a time with
        load_rotating_frame_hamiltonian(coupling_overrides=...) + simulate_signals_fast
        (official 300 steps, dt 0.0025) + the unchanged criterion (criterion.md v1). The
        reference for a modified H is the synthetic reference (exact sticks, 1 Hz Lorentzian),
        validated first on the real molecules.

Run from the repository root:
    .venv/bin/python agents/agent3_spectral/task_c_jcoupling.py
"""

import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from common import (  # noqa: E402
    AGENT, DIP_LEVELS, HZ_PER_PPM, MERGE_HZ, MOLS, OUT, PLOTS, characteristic_peaks, dump_json,
    evaluate, load_official, multiplets, nmr, ref_peaks, reference, signals_dict, spectrum,
    synthetic_reference, threshold,
)

MATCH = {"A": 3, "B": 2, "C": 1}
PRIMARY_KEY = "10pct"


# ------------------------------------------------------------------------------------------
# Hamiltonian parameters and first-order lines
# ------------------------------------------------------------------------------------------
def params(ham, n, shift):
    """Per-spin line position (ppm) and couplings J_ij (Hz) read from the rotating-frame H."""
    conv = nmr.hz_to_ppm_conversion()
    chem_shift = 2 * shift / (2.0 * np.pi / conv) + 1e6
    z = {}
    J = {(i, j): 0.0 for i in range(n) for j in range(i + 1, n)}  # zero terms are dropped from H
    for label, coeff in ham.to_list():
        ops = [(n - 1 - q, c) for q, c in enumerate(label) if c != "I"]
        if len(ops) == 1 and ops[0][1] == "Z":
            z[ops[0][0]] = coeff.real
        if len(ops) == 2 and ops[0][1] == ops[1][1] == "X":
            i, j = sorted(o[0] for o in ops)
            J[(i, j)] = 2.0 * coeff.real / np.pi
    nu = {i: (-2 * z.get(i, 0.0) / (2 * np.pi)) * conv - chem_shift for i in range(n)}
    return nu, J


def jname(i, j):
    return f"J{min(i, j)}{max(i, j)}"


def first_order_groups(nu, J, n):
    """First-order lines of every spin (centre + sum_j s_j J_ij / 2), merged like the criterion
    (closer than MERGE_HZ). Each group carries the averaged coefficient vector over couplings."""
    groups = []
    for i in range(n):
        nb = [j for j in range(n) if j != i]
        lines = []
        for signs in itertools.product((-1, 1), repeat=len(nb)):
            off = sum(s * J[tuple(sorted((i, j)))] / 2 for s, j in zip(signs, nb))
            coef = {jname(i, j): s / 2 for s, j in zip(signs, nb)}
            lines.append((nu[i] + off / HZ_PER_PPM, coef))
        lines.sort(key=lambda x: x[0])
        cur = [lines[0]]
        for ln in lines[1:]:
            if (ln[0] - cur[-1][0]) * HZ_PER_PPM < MERGE_HZ:
                cur.append(ln)
            else:
                groups.append((i, cur))
                cur = [ln]
        groups.append((i, cur))
    out = []
    for i, members in groups:
        keys = members[0][1].keys()
        out.append({"spin": i, "ppm": float(np.mean([m[0] for m in members])),
                    "coef": {k: float(np.mean([m[1][k] for m in members])) for k in keys},
                    "n_components": len(members)})
    return out


def label_pair(p1, p2, fo):
    """Express the spacing between two observed peaks as a first-order coupling combination."""
    g1 = min(fo, key=lambda g: abs(g["ppm"] - p1))
    g2 = min(fo, key=lambda g: abs(g["ppm"] - p2))
    if g1["spin"] != g2["spin"]:
        return f"different spins ({g1['spin']},{g2['spin']}): shift difference", None
    d = {k: g2["coef"][k] - g1["coef"][k] for k in g1["coef"]}
    terms = [(k, c) for k, c in d.items() if abs(c) > 1e-9]
    txt = " ".join(("+ " if c > 0 else "- ") + (f"{abs(c):g} " if abs(abs(c) - 1) > 1e-9 else "") + k
                   for k, c in sorted(terms, key=lambda x: -x[1]))
    txt = txt[2:] if txt.startswith("+ ") else txt
    return txt.strip(), (g2["ppm"] - g1["ppm"]) * HZ_PER_PPM


# ------------------------------------------------------------------------------------------
# Threshold for an arbitrary Hamiltonian with a synthetic reference
# ------------------------------------------------------------------------------------------
def artificial_point(m, overrides, levels=("any_dip", "5pct", "10pct", "19pct")):
    ham, n, shift = nmr.load_rotating_frame_hamiltonian(m, coupling_overrides=overrides)
    ex, ey = nmr.simulate_signals_fast(ham, n, 300, 0.0025)
    sig = signals_dict(ex, ey, shift, n)
    rp, rs = synthetic_reference(ham, n, shift)
    groups = characteristic_peaks(rp, rs)
    th = {k: threshold(sig, groups, k) for k in levels}
    cen = np.array([g["ppm"] for g in groups])
    seps = np.diff(cen) * HZ_PER_PPM
    ff = th[PRIMARY_KEY]["first_fail"]
    evf = evaluate(*spectrum(sig, ff), groups)
    if evf["found"]:
        i = int(np.nanargmin(evf["dips"]))
        crit = evf["pairs"][i]
    else:
        crit = None
    nu, J = params(ham, n, shift)
    fo = first_order_groups(nu, J, n)
    lab, fo_sep = label_pair(crit[0], crit[1], fo) if crit else ("detection failure", None)
    return {
        "n_char_peaks": len(groups), "min_char_spacing_hz": float(seps.min()),
        "critical_pair_ppm": None if crit is None else f"{crit[0]:.5f};{crit[1]:.5f}",
        "critical_splitting_hz": None if crit is None else float(crit[2]),
        "critical_label_first_order": lab, "critical_first_order_hz": fo_sep,
        **{f"threshold_{k}": th[k]["max_pass"] for k in levels},
        "first_fail_10pct": ff,
        "monotonic_10pct": th[PRIMARY_KEY]["monotonic"],
        "transitions_10pct": json.dumps(th[PRIMARY_KEY]["transitions_fine"]),
        "couplings_hz": json.dumps({jname(*k): round(v, 4) for k, v in J.items()}),
    }


def main():
    res_b = json.loads((AGENT / "task_b_result.json").read_text())
    rows_spacing, obs_rows = [], []
    # ---------------- Part 1 ----------------
    for m in MOLS:
        ham, n, shift = nmr.load_rotating_frame_hamiltonian(m)
        nu, J = params(ham, n, shift)
        fo = first_order_groups(nu, J, n)
        rp, rs = reference(MATCH[m])
        pk, _ = ref_peaks(rp, rs)
        for mi, mult in enumerate(multiplets(pk)):
            lines = mult["lines_ppm"]
            for a, b in zip(lines[:-1], lines[1:]):
                lab, fos = label_pair(a, b, _fo_unmerged(nu, J, n))
                rows_spacing.append({
                    "molecule": m, "reference": f"spectrum_{MATCH[m]}", "multiplet": mi,
                    "multiplet_centre_ppm": round(mult["centre_ppm"], 5), "line_left_ppm": round(a, 5),
                    "line_right_ppm": round(b, 5), "spacing_hz": round((b - a) * HZ_PER_PPM, 3),
                    "first_order_combination": lab,
                    "first_order_hz": None if fos is None else round(fos, 3),
                    "strong_coupling_correction_hz": None if fos is None else round((b - a) * HZ_PER_PPM - fos, 3),
                    "resolvable_by_300_step_acquisition": bool((b - a) * HZ_PER_PPM >= MERGE_HZ),
                })
        crit = res_b[m]["critical_pair"]
        lab, fos = label_pair(crit[0], crit[1], fo)
        th = res_b[m]["thresholds"]
        Jabs = np.abs(list(J.values()))
        obs_rows.append({
            "molecule": m, "reference": f"spectrum_{MATCH[m]}",
            "couplings_hz": json.dumps({jname(*k): round(v, 3) for k, v in J.items()}),
            "shift_ppm": json.dumps({i: round(v, 4) for i, v in nu.items()}),
            "mean_abs_J": float(Jabs.mean()), "max_abs_J": float(Jabs.max()), "min_abs_J": float(Jabs.min()),
            "critical_splitting_hz": float(crit[2]), "critical_origin": lab,
            "critical_first_order_hz": fos,
            **{f"threshold_{k}": th[k]["max_pass"] for k in DIP_LEVELS},
            "ratio_threshold_to_critical": th[PRIMARY_KEY]["max_pass"] / crit[2],
        })
    sp = pd.DataFrame(rows_spacing)
    sp.to_csv(AGENT / "reference_line_spacings.csv", index=False)
    print(sp.to_string())
    ob = pd.DataFrame(obs_rows)
    corr = {}
    for col in ("mean_abs_J", "max_abs_J", "min_abs_J", "critical_splitting_hz"):
        r, _ = pearsonr(ob[col], ob["threshold_10pct"])
        rho, _ = spearmanr(ob[col], ob["threshold_10pct"])
        corr[col] = {"pearson_r": float(r), "spearman_rho": float(rho)}
    ob.to_csv(AGENT / "observational_thresholds_vs_J.csv", index=False)
    print(ob.to_string())
    print(json.dumps(corr, indent=1))

    # ---------------- Part 3a: validate synthetic reference ----------------
    val = []
    for m in MOLS:
        ham, n, shift = nmr.load_rotating_frame_hamiltonian(m)
        g_real = characteristic_peaks(*reference(MATCH[m]))
        g_syn = characteristic_peaks(*synthetic_reference(ham, n, shift))
        same_n = len(g_real) == len(g_syn)
        dpos = float(np.max(np.abs(np.array([g["ppm"] for g in g_real]) - np.array([g["ppm"] for g in g_syn])))
                     * HZ_PER_PPM) if same_n else None
        sig = load_official(m)
        th_syn = {k: threshold(sig, g_syn, k)["max_pass"] for k in DIP_LEVELS}
        th_real = {k: res_b[m]["thresholds"][k]["max_pass"] for k in DIP_LEVELS}
        val.append({"molecule": m, "same_number_of_char_peaks": same_n, "max_position_diff_hz": dpos,
                    "thresholds_real_ref": th_real, "thresholds_synthetic_ref": th_syn,
                    "identical_thresholds": th_syn == th_real})
        print("validation", val[-1])

    # ---------------- Part 3b: artificial experiments ----------------
    experiments = [
        ("C_J01", "C", (0, 1), np.round(np.arange(3.0, 16.01, 0.5), 3), 7.92,
         "H1 and H2 both predict an increase (critical splitting = J01); intensity ratio of the AB quartet changes"),
        ("A_J02", "A", (0, 2), np.round(np.arange(4.0, 30.01, 0.5), 3), 11.8,
         "H1: threshold rises with J02. H2: critical splitting = |J12 - J02| shrinks for 11.8 < J02 < 17.9, so threshold falls"),
        ("A_J01", "A", (0, 1), np.round(np.arange(0.0, 8.01, 0.25), 3), 0.9,
         "H1: threshold rises with J01. H2: once J01 > 2.67 Hz the J01 doublets become resolvable and J01 is the new smallest spacing, so threshold falls"),
        ("A_J12", "A", (1, 2), np.round(np.arange(12.0, 30.01, 1.0), 3), 17.9,
         "H1 and H2 both predict a rise while J12 - J02 is the smallest spacing; H2 predicts saturation when another spacing becomes smallest"),
        ("B_J12", "B", (1, 2), np.round(np.arange(2.0, 15.01, 0.5), 3), 8.6,
         "H1: threshold rises with J12. H2: critical = min(J12, J01 - J12) peaks near J12 = 7.8 and falls on both sides"),
    ]
    jrows = []
    for name, m, pair, values, orig, hyp in experiments:
        for v in values:
            r = artificial_point(m, {pair: float(v)})
            jrows.append({"experiment": name, "artificial": True, "molecule": f"molecule {m} (modified)",
                          "varied_coupling": jname(*pair), "J_hz": float(v), "original_J_hz": orig,
                          "hypothesis_predictions": hyp, **r})
            print(name, v, r["threshold_10pct"], r["critical_splitting_hz"], r["critical_label_first_order"],
                  r["n_char_peaks"], r["monotonic_10pct"])
    jd = pd.DataFrame(jrows)
    jd.to_csv(OUT / "j_coupling_sensitivity.csv", index=False)

    # Statistics per experiment
    stats = {}
    for name, g in jd.groupby("experiment"):
        ok = g.threshold_10pct.notna() & g.critical_splitting_hz.notna()
        gg = g[ok]
        s = {
            "spearman_threshold_vs_J": float(spearmanr(gg.J_hz, gg.threshold_10pct)[0]),
            "spearman_threshold_vs_critical_splitting": float(spearmanr(gg.critical_splitting_hz, gg.threshold_10pct)[0]),
            "n": int(len(gg)),
            "J_range": [float(gg.J_hz.min()), float(gg.J_hz.max())],
            "threshold_range": [float(gg.threshold_10pct.min()), float(gg.threshold_10pct.max())],
        }
        stats[name] = s
    ok = jd.threshold_10pct.notna() & jd.critical_splitting_hz.notna()
    x, y = jd[ok].critical_splitting_hz.values, jd[ok].threshold_10pct.values
    slope = float(np.sum(x * y) / np.sum(x * x))
    r2 = float(1 - np.sum((y - slope * x) ** 2) / np.sum((y - y.mean()) ** 2))
    stats["pooled_fit_threshold_eq_a_times_critical_splitting"] = {"a": slope, "R2": r2, "n": int(ok.sum())}
    stats["pooled_spearman_threshold_vs_J"] = float(spearmanr(jd[ok].J_hz, y)[0])
    stats["pooled_spearman_threshold_vs_critical"] = float(spearmanr(x, y)[0])
    print(json.dumps(stats, indent=1))
    dump_json({"observational": obs_rows, "correlations_n3": corr, "synthetic_reference_validation": val,
               "experiment_stats": stats}, AGENT / "task_c_result.json")
    plot(jd, ob, slope)


def _fo_unmerged(nu, J, n):
    """First-order lines without merging (for labelling every reference spacing)."""
    out = []
    for i in range(n):
        nb = [j for j in range(n) if j != i]
        for signs in itertools.product((-1, 1), repeat=len(nb)):
            off = sum(s * J[tuple(sorted((i, j)))] / 2 for s, j in zip(signs, nb))
            out.append({"spin": i, "ppm": nu[i] + off / HZ_PER_PPM,
                        "coef": {jname(i, j): s / 2 for s, j in zip(signs, nb)}})
    return out


def plot(jd, ob, slope):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    cols = {"C_J01": "C0", "A_J02": "C3", "A_J01": "C1", "A_J12": "C2", "B_J12": "C4"}
    ax = axes[0]
    for name, g in jd.groupby("experiment"):
        ok = g.critical_splitting_hz.notna()
        ax.plot(g.J_hz.where(ok), g.threshold_10pct.where(ok), "o-", ms=3, color=cols[name], label=f"{name} (artificial)")
        ax.plot(g.J_hz[~ok], g.threshold_10pct[~ok].fillna(0.0), "x", ms=6, color=cols[name])
        orig = g.original_J_hz.iloc[0]
        ax.axvline(orig, color=cols[name], lw=0.6, ls=":")
    ax.set_xlabel("varied coupling J (Hz)")
    ax.set_ylabel("max tolerable noise_strength (1/s), 10% dip")
    ax.set_title("Threshold vs varied coupling (dotted: real value; x: detection-limited or never passes, plotted at 0)", fontsize=8)
    ax.legend(fontsize=7)
    ax = axes[1]
    for name, g in jd.groupby("experiment"):
        ax.plot(g.critical_splitting_hz, g.threshold_10pct, "o", ms=4, color=cols[name], label=name)
    for _, r in ob.iterrows():
        ax.plot(r.critical_splitting_hz, r.threshold_10pct, "k*", ms=13)
        ax.annotate(f"real {r.molecule}", (r.critical_splitting_hz, r.threshold_10pct), fontsize=8,
                    xytext=(4, -10), textcoords="offset points")
    xx = np.linspace(0, jd.critical_splitting_hz.max() * 1.05, 10)
    ax.plot(xx, slope * xx, "k--", lw=0.8, label=f"fit through 0: {slope:.2f} x splitting")
    ax.set_xlabel("critical splitting (Hz, reference separation of the failing pair)")
    ax.set_ylabel("max tolerable noise_strength (1/s)")
    ax.set_title("Threshold vs critical splitting (all points)")
    ax.legend(fontsize=7)
    ax = axes[2]
    for name, g in jd.groupby("experiment"):
        ax.plot(g.J_hz, g.critical_splitting_hz, "o-", ms=3, color=cols[name], label=name)
    ax.set_xlabel("varied coupling J (Hz)")
    ax.set_ylabel("critical splitting (Hz)")
    ax.set_title("Which spacing limits resolution")
    ax.legend(fontsize=7)
    fig.suptitle("ARTIFICIAL one-coupling-at-a-time experiments (official 300 steps, dt 0.0025, criterion v1)")
    fig.tight_layout()
    fig.savefig(PLOTS / "j_sensitivity.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if "--plot-only" in sys.argv:
        jd_ = pd.read_csv(OUT / "j_coupling_sensitivity.csv")
        ob_ = pd.read_csv(AGENT / "observational_thresholds_vs_J.csv")
        res_ = json.loads((AGENT / "task_c_result.json").read_text())
        plot(jd_, ob_, res_["experiment_stats"]["pooled_fit_threshold_eq_a_times_critical_splitting"]["a"])
    else:
        main()
