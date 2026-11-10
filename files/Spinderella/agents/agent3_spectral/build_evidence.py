"""Assemble agents/agent3_spectral/evidence.json from the measured result files.

Run from the repository root (after task_a, task_b, trotter_effective_lines, task_c):
    .venv/bin/python agents/agent3_spectral/build_evidence.py
"""

import json

import numpy as np
import pandas as pd

from common import AGENT, OUT, dump_json

PY = ".venv/bin/python"


def main():
    A = json.loads((AGENT / "task_a_result.json").read_text())
    B = json.loads((AGENT / "task_b_result.json").read_text())
    C = json.loads((AGENT / "task_c_result.json").read_text())
    T = json.loads((AGENT / "trotter_effective_lines.json").read_text())
    nt = pd.read_csv(OUT / "noise_tolerance.csv")
    jd = pd.read_csv(OUT / "j_coupling_sensitivity.csv")
    sm = pd.DataFrame(A["summary"])
    ev = []

    ev.append({
        "id": "A1_matching",
        "question": "Which reference spectrum belongs to which molecule (global one-to-one)?",
        "hypotheses": ["all 6 permutations considered; the best permutation per metric is the match"],
        "independent_variable": "permutation of (spectrum_1, spectrum_2, spectrum_3) over (A, B, C)",
        "dependent_variable": "summed cosine, Pearson r, NRMSE, peak match fraction, peak F1, multiplet score",
        "fixed_parameters": {"noise_strength": [0.5, 2.0, 5.0], "resampling": ["bin average (primary)", "point interpolation"],
                             "peak_tolerance_hz": 2.667, "official signals": "300 steps, dt 0.0025"},
        "reference": "files/spectrum_{1,2,3}.pkl",
        "result": {"mapping": A["mapping"], "unanimous_over_choices": A["unanimous"], "n_choices": A["n_choices"],
                   "min_margin_by_metric": A["min_margin_by_metric"],
                   "per_choice": sm[["noise_strength", "resampling", "metric", "best", "best_total", "second", "second_total", "margin"]].to_dict(orient="records"),
                   "leaked_file_consistency": {"identical_to": A["leaked_C_file_identical_to"], "agrees": A["agrees_with_leaked_file"]}},
        "conclusion": "All metric/noise/resampling choices select A->spectrum_3, B->spectrum_2, C->spectrum_1. "
                      "The leaked file (byte copy of spectrum_1) was checked only after the decision and agrees.",
        "limitations": "Simulation windows (+-200 Hz) do not cover every reference fully (overlap fraction of C with spectrum_1 is 0.78 of the simulation grid); "
                       "peak positions carry the known ~0.67 Hz axis offset.",
        "script": "agents/agent3_spectral/task_a_matching.py",
        "command": f"{PY} agents/agent3_spectral/task_a_matching.py",
    })

    for _, r in nt.iterrows():
        m = r.molecule.split()[-1]
        b = B[m]
        ev.append({
            "id": f"B1_threshold_{m}",
            "question": f"Highest noise_strength at which molecule {m}'s characteristic peaks stay resolved (criterion v1, 10% dip)",
            "hypotheses": ["pass/fail is monotonic in noise_strength", "the smallest characteristic splitting fails first"],
            "independent_variable": "noise_strength (1/s), 0.1..100 step 0.1, transitions refined at 0.01",
            "dependent_variable": "pass/fail of criterion v1; dips between adjacent characteristic peaks",
            "fixed_parameters": {"signals": "official, 300 Lie-Trotter steps, dt 0.0025 s", "tolerance_hz": 2.667,
                                 "merge_hz": 2.667, "primary_dip": 0.10},
            "reference": r.matched_reference,
            "result": {"max_tolerable_noise_strength": r.max_tolerable_noise_strength,
                       "first_failing_noise_strength": r.first_failing_noise_strength,
                       "critical_peaks_ppm": r.critical_peaks_ppm, "critical_splitting_hz": r.critical_splitting_hz,
                       "n_characteristic_peaks": b["n_char_peaks"],
                       "lowest_noise_check_0p1": b["lowest_noise_check"],
                       "transitions_fine_10pct": b["thresholds"]["10pct"]["transitions_fine"]},
            "conclusion": f"Threshold {r.max_tolerable_noise_strength} 1/s (first fail {r.first_failing_noise_strength}); limited by the "
                          f"{r.critical_splitting_hz:.2f} Hz pair {r.critical_peaks_ppm} ppm.",
            "limitations": "Operational criterion (10% dip) replaces 'visually resolvable'; the dip is measured on the 1.333 Hz sampled grid.",
            "script": "agents/agent3_spectral/task_b_noise.py",
            "command": f"{PY} agents/agent3_spectral/task_b_noise.py",
        })

    ev.append({
        "id": "B2_dip_depth_sensitivity",
        "question": "How much does the threshold depend on the required dip depth?",
        "hypotheses": ["threshold decreases as the required dip increases"],
        "independent_variable": "dip level: any dip, 5%, 10% (primary), 19%",
        "dependent_variable": "max tolerable noise_strength",
        "fixed_parameters": {"signals": "official 300 steps"},
        "reference": "matched references",
        "result": {m: {k: B[m]["thresholds"][k]["max_pass"] for k in B[m]["thresholds"]} for m in B},
        "conclusion": "Ordering of thresholds between dip levels is monotonic for every molecule; see report for spread.",
        "limitations": "For 'any dip' the failure is the disappearance of a local maximum (shoulder), not a position failure.",
        "script": "agents/agent3_spectral/task_b_noise.py",
        "command": f"{PY} agents/agent3_spectral/task_b_noise.py",
    })

    ev.append({
        "id": "B3_monotonicity",
        "question": "Is pass/fail monotonic in noise_strength (truncation ringing / np.abs at low noise)?",
        "hypotheses": ["H0: monotonic (single pass->fail transition)", "H1: non-monotonic at low noise"],
        "independent_variable": "noise_strength on the coarse grid (0.1..100) and on a 0.1 grid (0.1..100), refined at 0.01",
        "dependent_variable": "number and type of pass/fail transitions",
        "fixed_parameters": {"signals": "official"},
        "reference": "matched references",
        "result": {m: {"coarse": B[m]["coarse_transitions"], "dense_0p1": B[m]["dense_0p1_transitions"],
                       "fine_monotonic": {k: B[m]["thresholds"][k]["monotonic"] for k in B[m]["thresholds"]}} for m in B},
        "conclusion": "Every molecule and every dip level shows exactly one pass->fail transition; passes from 0.1 up. No bisection was needed (dense scan used).",
        "limitations": "Monotonicity verified on 0.1 grid globally and 0.01 grid only inside transition intervals.",
        "script": "agents/agent3_spectral/task_b_noise.py",
        "command": f"{PY} agents/agent3_spectral/task_b_noise.py",
    })

    ev.append({
        "id": "B4_trotter_error_sensitivity",
        "question": "How much does first-order Trotter error move the threshold?",
        "hypotheses": ["Trotter error changes line positions/intensities and therefore the threshold"],
        "independent_variable": "signals: Lie-Trotter (official) vs exact dense diagonalisation, both 300 points dt 0.0025",
        "dependent_variable": "threshold at each dip level; line positions and weights of H_eff = i log(U_step)/dt",
        "fixed_parameters": {"criterion": "v1", "reference": "matched"},
        "reference": "matched references",
        "result": {m: {"trotter": {k: B[m]["thresholds"][k]["max_pass"] for k in B[m]["thresholds"]},
                       "exact": {k: B[m]["sensitivity"]["exact_300"][k]["max_pass"] for k in B[m]["thresholds"]},
                       "critical_pair_sep_exact_vs_trotter_Heff_hz": [T[m]["char_sep_exact_hz"], T[m]["char_sep_trotter_hz"]],
                       "line_shift_trotter_minus_exact_hz": T[m]["shift_of_nearest_line_hz"],
                       "weights_exact": T[m]["exact_weights"], "weights_trotter": T[m]["trotter_weights"]} for m in B},
        "conclusion": "Trotter error barely changes the critical splittings but changes relative line intensities in A and B, which shifts their thresholds.",
        "limitations": "H_eff from the principal matrix logarithm (max |eigenphase| < pi checked); 'nearest line' separations involving merged sub-grid pairs are not meaningful.",
        "script": "agents/agent3_spectral/task_b_noise.py; agents/agent3_spectral/trotter_effective_lines.py",
        "command": f"{PY} agents/agent3_spectral/task_b_noise.py && {PY} agents/agent3_spectral/trotter_effective_lines.py",
    })

    ev.append({
        "id": "B5_acquisition_length_sensitivity",
        "question": "How much does the finite 0.75 s acquisition (300 steps) move the threshold?",
        "hypotheses": ["at thresholds of ~20 1/s the window has decayed by exp(-15), so truncation is irrelevant; only the grid spacing changes"],
        "independent_variable": "number of Lie-Trotter steps 300 / 600 / 1200 at dt 0.0025 (and exact 1200)",
        "dependent_variable": "threshold at each dip level",
        "fixed_parameters": {"criterion": "v1 (same characteristic peaks and tolerance in Hz)"},
        "reference": "matched references",
        "result": {m: {lab: {k: B[m]["sensitivity"][lab][k]["max_pass"] for k in B[m]["thresholds"]}
                       for lab in ("trotter_600", "trotter_1200", "exact_1200")} for m in B},
        "conclusion": "Longer acquisitions shift thresholds by a fraction of 1/s at the 5/10/19% levels (grid sampling effect).",
        "limitations": "Finer grids sample peaks and valleys more accurately; the change mixes grid sampling with truncation.",
        "script": "agents/agent3_spectral/task_b_noise.py",
        "command": f"{PY} agents/agent3_spectral/task_b_noise.py",
    })

    ev.append({
        "id": "C0_synthetic_reference_validation",
        "question": "Can an exact-stick + 1 Hz Lorentzian reference stand in for the supplied references?",
        "hypotheses": ["same characteristic peaks and identical thresholds on the real molecules"],
        "independent_variable": "reference type (supplied vs synthetic)",
        "dependent_variable": "characteristic peaks, thresholds at all dip levels",
        "fixed_parameters": {"signals": "official"},
        "reference": "matched references",
        "result": C["synthetic_reference_validation"],
        "conclusion": "Synthetic references reproduce the characteristic peaks within 0.04 Hz and give identical thresholds; used for the artificial experiments.",
        "limitations": "Assumes the same 1 Hz linewidth for modified Hamiltonians.",
        "script": "agents/agent3_spectral/task_c_jcoupling.py",
        "command": f"{PY} agents/agent3_spectral/task_c_jcoupling.py",
    })

    ev.append({
        "id": "C1_observational",
        "question": "Across the three real molecules, does the threshold follow J strength or the critical splitting?",
        "hypotheses": ["H1: threshold scales with J strength (mean/max/min |J|)", "H2: threshold scales with the smallest resolvable line spacing"],
        "independent_variable": "molecule (n = 3)",
        "dependent_variable": "primary threshold",
        "fixed_parameters": {"criterion": "v1"},
        "reference": "matched references",
        "result": {"table": C["observational"], "correlations_n3": C["correlations_n3"]},
        "conclusion": "Threshold is anti-ordered with mean and max |J| and ordered with the critical splitting (n = 3, correlation only).",
        "limitations": "n = 3; no causal claim from this part.",
        "script": "agents/agent3_spectral/task_c_jcoupling.py",
        "command": f"{PY} agents/agent3_spectral/task_c_jcoupling.py",
    })

    hyp = {r.experiment: r.hypothesis_predictions for r in jd.drop_duplicates("experiment").itertuples()}
    # Segments where H1 (threshold rises with J) and H2 (threshold follows the smallest spacing) disagree
    segs = {"A_J02": (9.5, 14.0, "J12 - J02 shrinks"), "A_J01": (1.0, 4.0, "J01 becomes a resolvable spacing"),
            "B_J12": (8.5, 12.5, "J01 - J12 shrinks"), "A_J12": (24.0, 28.0, "J02 = 11.8 Hz becomes the smallest spacing"),
            "C_J01": (3.5, 16.0, "H1 and H2 agree (critical splitting = J01)")}
    from scipy.stats import spearmanr
    seg_out = {}
    for name, (lo, hi, why) in segs.items():
        g = jd[(jd.experiment == name) & (jd.J_hz >= lo) & (jd.J_hz <= hi)]
        gv = g[g.threshold_10pct.notna()]
        rho = float(spearmanr(gv.J_hz, gv.threshold_10pct)[0]) if len(gv) > 2 and gv.threshold_10pct.nunique() > 1 else None
        seg_out[name] = {"J_range": [lo, hi], "why": why, "n": int(len(g)), "n_never_pass": int(g.threshold_10pct.isna().sum()),
                         "threshold_at_lo": None if g.threshold_10pct.isna().iloc[0] else float(g.threshold_10pct.iloc[0]),
                         "threshold_at_hi": None if g.threshold_10pct.isna().iloc[-1] else float(g.threshold_10pct.iloc[-1]),
                         "spearman_threshold_vs_J_in_segment": rho}
    (AGENT / "task_c_segments.json").write_text(json.dumps(seg_out, indent=2))
    print(json.dumps(seg_out, indent=1))
    for name, g in jd.groupby("experiment"):
        st = C["experiment_stats"][name]
        sg = seg_out[name]
        verdict = ("H1 and H2 agree here; threshold rises with J" if name == "C_J01" else
                   ("supports H2 (threshold falls or stays flat while J rises)"
                    if (sg["threshold_at_hi"] is None or sg["threshold_at_lo"] is None or sg["threshold_at_hi"] <= sg["threshold_at_lo"] + 1.0)
                    else "supports H1"))
        ev.append({
            "id": f"C2_artificial_{name}",
            "question": f"ARTIFICIAL: how does the threshold change when only {g.varied_coupling.iloc[0]} of molecule {name[0]} is varied?",
            "hypotheses": ["H1: threshold scales with the J-coupling strength itself",
                           "H2: threshold is set by the smallest resolvable line spacing", hyp[name]],
            "independent_variable": f"{g.varied_coupling.iloc[0]} from {g.J_hz.min()} to {g.J_hz.max()} Hz",
            "dependent_variable": "threshold (10% dip, plus any/5/19%), critical splitting",
            "fixed_parameters": {"other couplings and shifts": "as in the file", "steps": 300, "dt": 0.0025, "criterion": "v1",
                                 "reference": "synthetic (exact sticks, 1 Hz Lorentzian)"},
            "reference": "synthetic",
            "result": {"stats": st, "disagreement_segment": sg, "points": g[["J_hz", "threshold_10pct", "threshold_5pct", "threshold_19pct", "threshold_any_dip",
                                                 "critical_splitting_hz", "critical_label_first_order", "n_char_peaks"]].to_dict(orient="records")},
            "conclusion": f"Segment {sg['J_range']} ({sg['why']}): threshold {sg['threshold_at_lo']} -> {sg['threshold_at_hi']} 1/s, "
                          f"Spearman(threshold, J) = {sg['spearman_threshold_vs_J_in_segment']}; {verdict}. Whole sweep: Spearman vs J "
                          f"{st['spearman_threshold_vs_J']:.2f}, vs critical splitting {st['spearman_threshold_vs_critical_splitting']:.2f}.",
            "limitations": "Artificial Hamiltonians, not the supplied molecules; synthetic references. Splittings of 2.7-3.5 Hz pass the 2-bin merge rule "
                           "but are never resolved (no passing noise). Some points fail in narrow windows at 0.1-0.5 1/s (low-noise ringing). "
                           "A_J12 >= 29 Hz fails detection at every noise: Trotter line shifts plus the axis offset exceed the 2.667 Hz tolerance.",
            "script": "agents/agent3_spectral/task_c_jcoupling.py",
            "command": f"{PY} agents/agent3_spectral/task_c_jcoupling.py",
        })
    pooled = C["experiment_stats"]["pooled_fit_threshold_eq_a_times_critical_splitting"]
    ev.append({
        "id": "C3_pooled_fit",
        "question": "ARTIFICIAL + real: does one proportionality threshold = a x critical splitting describe all points?",
        "hypotheses": ["H2 predicts a near-universal proportionality modulated by intensity ratios"],
        "independent_variable": "critical splitting (Hz)", "dependent_variable": "threshold (10% dip)",
        "fixed_parameters": {"criterion": "v1"}, "reference": "synthetic",
        "result": {"fit": pooled, "spearman_vs_J": C["experiment_stats"]["pooled_spearman_threshold_vs_J"],
                   "spearman_vs_critical": C["experiment_stats"]["pooled_spearman_threshold_vs_critical"]},
        "conclusion": "See report.", "limitations": "Points with no passing value excluded from the fit.",
        "script": "agents/agent3_spectral/task_c_jcoupling.py",
        "command": f"{PY} agents/agent3_spectral/task_c_jcoupling.py",
    })
    dump_json(ev, AGENT / "evidence.json")
    print(f"wrote {len(ev)} evidence entries")


if __name__ == "__main__":
    main()
