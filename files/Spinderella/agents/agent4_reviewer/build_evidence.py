"""Assemble agents/agent4_reviewer/evidence.json from the outputs in out/ (numbers are read from the
files written by the r*.py scripts, not typed by hand).
Run: cd agents/agent4_reviewer && ../../.venv/bin/python build_evidence.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE / "out"
r1 = json.load(open(O / "r1_initial_states.json"))
r2 = json.load(open(O / "r2_physics.json"))
r3 = json.load(open(O / "r3_matching.json"))
r4 = json.load(open(O / "r4_thresholds.json"))
r4b = json.load(open(O / "r4b_zero_fill_J01.json"))
r5 = json.load(open(O / "r5_exact_limit.json"))
r5b = json.load(open(O / "r5b_ranking.json"))
r6 = json.load(open(O / "r6_jcoupling.json"))
r6b = json.load(open(O / "r6b_spotcheck.json"))
CMD = "cd agents/agent4_reviewer && ../../.venv/bin/python {}"
V = {v["variant"]: v for v in r4["variants"]}


def th(v):
    return {m: V[v][m]["max_pass"] for m in "ABC"}


ev = []
ev.append({
    "id": "R1a", "question": "Does H then RZ(pi b_i) give the complete orthonormal X-product basis with X_tot eigenvalue N - 2w?",
    "hypotheses": ["H0: every template state is an X_i eigenstate with eigenvalue (-1)^b_i and X_tot eigenvalue N-2w; Gram = I",
                   "H1: some state is not an eigenstate / the set is not orthonormal"],
    "independent_variable": "bit string b (all 2^N)", "dependent_variable": "||X_i|s> - (+-1)|s>||, ||X_tot|s> - (N-2w)|s>||, |Gram - I|",
    "fixed_parameters": "notebook template via nmr_pipeline.enumerate_x_states, Qiskit little-endian mapping",
    "reference": "exact eigen-equations",
    "result": {m: {"max_eigen_residual": r1[m]["template_eigen_residual_max"], "max_gram_dev": r1[m]["gram_minus_identity_max"],
                   "n_states": r1[m]["n_states"]} for m in "ABC"},
    "conclusion": "H0 holds to rounding (residual <= 2.1e-16, Gram deviation <= 4.4e-16).",
    "limitations": "Tests the template as used by the pipeline; not the Estimator path itself (the official signal files are compared with the trace in R1c).",
    "script": "r1_initial_states.py", "command": CMD.format("r1_initial_states.py")})
W = {m: {k: {"best_scale": v["best_scale"], "corr_rel_residual": v["corr_rel_residual_after_scale"],
             "spectrum_maxabs_diff": v["spectrum_maxabs_diff_vs_trace"], "note": v["note"][:40]}
         for k, v in r1[m]["weightings"].items()} for m in "ABC"}
ev.append({
    "id": "R1b", "question": "Is the weight m_s = N - 2w the only weighting of the 2^N state signals that reproduces Tr(A(t) X_tot)?",
    "hypotheses": ["H0 (claim): signed weights N-2w are required; equal weights and all-plus-only fail; it is the only correct weighting",
                   "H1: a class of weightings reproduces the trace exactly"],
    "independent_variable": "weighting (10 choices)", "dependent_variable": "relative residual of C_xx,C_yx after best scalar fit; max |spectrum difference| (noise 5, notebook normalisation)",
    "fixed_parameters": "official Lie-Trotter step (own product formula, equal to official signals to 2e-12), 300 steps, dt 2.5 ms",
    "reference": "Tr(U^-k A U^k X_tot) under the same dynamics",
    "result": W,
    "conclusion": "Equal weights give Tr(A(t)) = 0 (|signal| <= 8.9e-16; for A and B the pipeline guard does not fire and it normalises rounding noise). N-2w is correct, but NOT unique: -(N-2w), w, N-w (all non-negative), and N-2w + (N-2w)^2 reproduce the spectrum to < 3e-14, because Tr(X_tot(t)) = 0 and flip-even weight parts cancel by Z-parity symmetry. All-plus only: exact for C (N=2), off by 4.8 % (A) and 4.0 % (B) of the maximum. m_s^3 and the integer bit label fail for N = 3.",
    "limitations": "Spectral differences measured at one noise strength (5 s^-1).",
    "script": "r1_initial_states.py", "command": CMD.format("r1_initial_states.py")})
ev.append({
    "id": "R1c", "question": "Sum rules C_xx(0) = N 2^N, C_yx(0) = 0 in the official files; Pauli-vs-spin normalisation cancels?",
    "hypotheses": ["H0: sum rules hold; spin-1/2 operators with 1/2^N give the same normalised spectrum", "H1: a constant or shape difference remains"],
    "independent_variable": "operator convention (Pauli vs sigma/2 with 1/2^N)", "dependent_variable": "C(0) values; max |spectrum difference| at noise 0.1, 5, 30",
    "fixed_parameters": "official results/signals/*.npz", "reference": "N 2^N = 24, 24, 8",
    "result": {m: {k: r1[m][k] for k in ("official_cxx0", "expected_N2N", "official_cyx0", "official_mx0_minus_N_minus_2w_max",
                                          "official_vs_trace_corr_max", "pauli_over_spin_ratio", "spin_vs_pauli_spectrum_maxdiff")} for m in "ABC"},
    "conclusion": "Confirmed: C_xx(0) = 24, 24, 8; |C_yx(0)| <= 1.5e-15; official signals equal the trace correlation to 2.0e-12; Pauli/spin ratio 4*2^N (32, 32, 16); normalised spectra identical to 2e-13.",
    "limitations": "None material.", "script": "r1_initial_states.py", "command": CMD.format("r1_initial_states.py")})
ev.append({
    "id": "R2a", "question": "Is the rotating-frame subtraction exact?",
    "hypotheses": ["H0: [H, Z_tot] = 0 and lab-frame lines = rotating-frame lines + one rigid shift", "H1: the subtraction changes the spectrum"],
    "independent_variable": "frame (full lab H ~1.6e9 rad/s vs rotating frame)", "dependent_variable": "||[H,Z_tot]||/||H||; max line-position difference (Hz)",
    "fixed_parameters": "own JSON parser, dense eigendecomposition", "reference": "lab-frame transition frequencies",
    "result": r2["rotating_frame"],
    "conclusion": "Exact: commutator 0; all 12/12/4 lines agree to <= 1.8e-7 Hz after the rigid shift -2 h_mean / 2 pi.",
    "limitations": "Eigenvalue precision of the lab-frame H limits the check to ~1e-7 Hz.", "script": "r2_physics_axis_trotter.py", "command": CMD.format("r2_physics_axis_trotter.py")})
ev.append({
    "id": "R2b", "question": "Does the code's '+ Im' put lines at the true ppm (and the markdown's '- Im' mirror them)?",
    "hypotheses": ["H0: '+ Im' -> true ppm", "H1: '+ Im' mirrors"],
    "independent_variable": "sign of the Im term", "dependent_variable": "peak ppm of 3 uncoupled spins (J = 0 copy of A)",
    "fixed_parameters": "exact dynamics, 300 steps, noise 1", "reference": "delta_i = 5.48, 5.79, 5.97 ppm",
    "result": r2["sign"],
    "conclusion": "'+ Im' peaks within one grid bin of the true shifts; '- Im' peaks at the mirror image 2*mean - delta_i. Confirmed.",
    "limitations": "Peak positions limited by the 1.333 Hz grid.", "script": "r2_physics_axis_trotter.py", "command": CMD.format("r2_physics_axis_trotter.py")})
ev.append({
    "id": "R2c", "question": "How far is the ppm axis offset from the frequencies the transform actually evaluates?",
    "hypotheses": ["H0 (claim): lines drawn 0.41-0.92 Hz low (sub-bin)", "H1: offset >= 1 bin or different range"],
    "independent_variable": "line frequency", "dependent_variable": "label - evaluated frequency at each exact line",
    "fixed_parameters": "own derivation f_j = (2j-(L-1))/((2L-1) dt), kernel reproduces the pipeline output to 8e-14",
    "reference": "fftshift(fftfreq(300, dt)) labels",
    "result": r2["axis"],
    "conclusion": "label - f = -0.6667 Hz - 0.001667 f exactly; over the strong lines of A, B, C the offset is -0.42 to -0.93 Hz (B's outermost line at 159.2 Hz gives 0.93). Claim confirmed up to rounding of the upper end.",
    "limitations": "None.", "script": "r2_physics_axis_trotter.py", "command": CMD.format("r2_physics_axis_trotter.py")})
ev.append({
    "id": "R2d", "question": "What does first-order Trotter error at dt = 2.5 ms do to the lines?",
    "hypotheses": ["H0 (claim): line shifts <= 0.72 Hz", "H1: Trotter error also changes intensities / phases materially"],
    "independent_variable": "dt (2.5, 1.25, 0.625 ms)", "dependent_variable": "per-line shift, relative absorptive amplitude change, phase",
    "fixed_parameters": "Floquet decomposition of the one-step propagator (own product formula) vs exact",
    "reference": "exact lines", "result": {m: {k: {kk: vv for kk, vv in v.items() if kk in ("max_abs_shift_Hz", "max_rel_absorptive_amp_change", "max_abs_phase_rad", "n_lines")}
                                                for k, v in r2["trotter"][m].items()} for m in "ABC"},
    "conclusion": "Shifts confirmed (0.715, 0.418, 0.012 Hz; ~4-5x smaller per halving). Omitted by the claim: absorptive line amplitudes change by up to 19.1 % (A) and 15.7 % (B), 0.7 % (C); no phase twist (< 1e-3 rad). The amplitude error is what moves the thresholds (R5b).",
    "limitations": "Lines with amplitude < 5 % of the maximum not compared.", "script": "r2_physics_axis_trotter.py", "command": CMD.format("r2_physics_axis_trotter.py")})
ev.append({
    "id": "R2e", "question": "Units: J in Hz vs coefficients, nu0, Hz/ppm, FWHM = Gamma/pi",
    "hypotheses": ["H0: c = pi J / 2 (isotropic), nu0 = gamma B / 2 pi = 500 MHz, FWHM = Gamma/pi", "H1: a factor 2 or 2 pi is wrong"],
    "independent_variable": "-", "dependent_variable": "J, delta recovered from the files; measured FWHM at Gamma = 10 s^-1",
    "fixed_parameters": "single line, 40000-step acquisition (0.01 Hz grid)", "reference": "Gamma/pi = 3.183 Hz",
    "result": r2["units"],
    "conclusion": "J = 0.9/11.8/17.9, 15.6/-0.5/8.6, 7.92 Hz; deltas as stated; nu0 = 499.99999 MHz, 500.0 Hz/ppm; FWHM 3.170 Hz vs 3.183 predicted (0.4 %, discrete-time line shape and grid).",
    "limitations": "-", "script": "r2_physics_axis_trotter.py", "command": CMD.format("r2_physics_axis_trotter.py")})
ev.append({
    "id": "R3", "question": "Is A->spectrum_3, B->spectrum_2, C->spectrum_1 robust to the matching choices?",
    "hypotheses": ["H0: one assignment wins for every reasonable choice", "H1: another assignment wins for some reasonable choice"],
    "independent_variable": "noise (0.1-20), metric (9: cosine linear/nearest/bin-average, overlap-only cosine and Pearson, reference-grid cosine, -L1 of area-normalised spectra, peak-F1, -chamfer), axis correction (0 or +0.667 Hz)",
    "dependent_variable": "winning one-to-one assignment, margin best - second",
    "fixed_parameters": "official signals; references normalised; file names not used",
    "reference": "all 6 permutations",
    "result": {"n_choices": len(r3["rows"]), "distinct_winners": r3["winners"],
               "worst_relative_margin": r3["worst"]["rel_margin"], "worst_case": {k: r3["worst"][k] for k in ("metric", "noise", "axis_shift_Hz", "best_total", "second", "second_total")},
               "min_relative_margin_full_window_metrics": min(r["rel_margin"] for r in r3["rows"] if r["metric"] not in ("overlap_cos", "overlap_pearson")),
               "rowwise_disagreements": r3["n_rowwise_disagree"], "md5": r3["md5"]},
    "conclusion": "Unanimous over 126 choices. Worst relative margin 13 % (overlap-only cosine, noise 0.1) because A's simulated tails vs spectrum_1 tails in their 0.12 ppm overlap (no lines) score 0.99; with row-wise (not one-to-one) assignment the overlap metrics would map A -> spectrum_1. Full-window metrics: margin >= 52 %.",
    "limitations": "The upstream notebook overlays molecule A with spectrum_3 (cell 29, commit f27f229) and the C file is a byte copy of spectrum_1: two of three assignments are hinted by the supplied material; the numerical result does not use them.",
    "script": "r3_matching.py", "command": CMD.format("r3_matching.py")})
P = {lev: th(f"PRIMARY-type: ref default, tol 2 bins, dip {lev}") for lev in ("any", "5%", "10%", "19%")}
ev.append({
    "id": "R4a", "question": "Reproduce the official thresholds independently; are they monotonic over 0.01-100?",
    "hypotheses": ["H0 (claim): 10 % A 19.97, B 22.85, C 23.22 (any 27.32/33.36/28.58; 5 % 22.15/25.99/25.51; 19 % 17.11/19.13/20.02), monotonic", "H1: different values or non-monotonic"],
    "independent_variable": "noise_strength 0.01..100 step 0.01 (10 000 points) and 0", "dependent_variable": "pass/fail, highest pass, transitions",
    "fixed_parameters": "official signals, own transform (== pipeline to 8e-14), own peak/dip code, criterion.md v1",
    "reference": "criterion.md", "result": {"thresholds": P, "critical": r4["critical"],
                                          "transitions": {lev: {m: V[f"PRIMARY-type: ref default, tol 2 bins, dip {lev}"][m]["n_transitions_0.01_100"] for m in "ABC"} for lev in ("any", "5%", "10%", "19%")}},
    "conclusion": "All 12 numbers reproduced exactly; one transition each over 0.01-100 and pass at noise 0. Critical pairs at 10 %: A 6.05 Hz, B 7.02 Hz, C 7.92 Hz confirmed. At any-dip the first failure is a lost maximum: A inner line 5.4850 ppm, B the OUTER line 5.9250 ppm (8.58 Hz pair, not the 7.02 Hz pair), C both outer lines.",
    "limitations": "-", "script": "r4_threshold_sensitivity.py", "command": CMD.format("r4_threshold_sensitivity.py")})
sens = {v["variant"]: {"A": v["A"]["max_pass"], "B": v["B"]["max_pass"], "C": v["C"]["max_pass"], "ranking": v["ranking"],
                       "n_transitions_0.1_100": [v[m]["n_transitions_0.1_100"] for m in "ABC"]} for v in r4["variants"]}
ev.append({
    "id": "R4b", "question": "Are the thresholds sensitive to choices a reasonable analyst could make differently?",
    "hypotheses": ["H0: thresholds stable", "H1: some reasonable choice changes them or makes a molecule never pass"],
    "independent_variable": "merge distance (0.5/1/2/3 bins), tolerance (0.5/1/2/3 bins), peak-finder thresholds, centre source (reference / own low-noise spectrum / exact sticks), count-of-maxima without position, signed spectrum (no np.abs), dip relative to the larger peak",
    "dependent_variable": "highest passing noise per molecule, ranking, transitions",
    "fixed_parameters": "official signals; 0.01 grid", "reference": "primary", "result": sens,
    "conclusion": "Unchanged under merge 1-3 bins, all peak-finder thresholds, centres from own spectrum or exact sticks, no np.abs, count mode (only B any-dip 33.36 -> 33.53 where its any-dip failure is a position failure). Tolerance is fragile: 1 bin -> A and B NEVER pass (not even at noise 0), because at low noise the simulated maxima already sit up to 1.73 (A), 1.86 (B), 1.30 (C) Hz from the reference centres (axis offset + grid + Trotter shift). 'Exactly k maxima' count mode is non-monotonic below 1.5 s^-1 (ringing) for B and C, thresholds unchanged. Dip relative to the larger peak: 10 % gives 24.07/28.19/28.58.",
    "limitations": "Variants are one-at-a-time.", "script": "r4_threshold_sensitivity.py", "command": CMD.format("r4_threshold_sensitivity.py")})
ev.append({
    "id": "R4c", "question": "Is excluding A's 0.94 Hz J01 doublets legitimate?",
    "hypotheses": ["H0 (claim): the 300-step signal cannot show them as two maxima at any noise", "H1: they are resolvable with a finer frequency grid (zero filling)"],
    "independent_variable": "frequency grid spacing (1.333 official, 0.667, 0.333, 0.05 Hz), noise 0-10", "dependent_variable": "highest noise with two maxima (one within 0.3 Hz of each line) in all four doublets; with >= 10 % dip",
    "fixed_parameters": "300 steps, official Lie-Trotter and exact dynamics", "reference": "reference spectrum_3 resolves them (0.81-0.86 Hz)",
    "result": {"official_grid_noise0_maxima_per_doublet": r4["J01"]["official_noise0_maxima_per_doublet_window"],
               "zero_fill": {k: r4b[k] for k in ("trotter_official", "exact")},
               "long_acquisition_exact": {k: v for k, v in r4["J01"].items() if k.startswith("exact_")},
               "merge_0.5_bin_thresholds": th("centres: ref merge 0.5 bin, dip 10%")},
    "conclusion": "H0 holds for the official grid (0-1 maxima per doublet at every noise; A never passes if the doublets count). It is a grid/acquisition property: zero-filled to <= 0.33 Hz, the official Trotter signal shows all four doublets up to 1.25-1.95 s^-1 (>= 10 % dip up to 0.4-0.6), exact dynamics only up to 0.35 s^-1 with dips < 10 % (the Trotter signal resolves them better than the exact one: artefact). 1200/2400-step exact acquisitions keep them up to 3.85/4.65 s^-1.",
    "limitations": "The markdown of the notebook says zeros are added; the code does not. Using zero filling would change the official pipeline.",
    "script": "r4_threshold_sensitivity.py; r4b_zero_fill_J01.py", "command": CMD.format("r4b_zero_fill_J01.py")})
ev.append({
    "id": "R4d", "question": "Is the ranking of the three molecules robust to the dip level and to Trotter error?",
    "hypotheses": ["H0: same ranking for every dip level and dynamics", "H1: the ranking changes"],
    "independent_variable": "dip level (any, 5, 10, 19 %) x dynamics (official Trotter 300, exact 300, exact 2400)",
    "dependent_variable": "thresholds and ranking", "fixed_parameters": "reference centres, +-2.667 Hz",
    "reference": "official primary ranking A < B < C", "result": r5b["ranking"],
    "conclusion": "Not robust. B vs C flips with the dip level (official: C < B at any/5 %, B < C at 10/19 %; |B-C| = 0.37 s^-1 at 10 %) and with Trotter error (exact dynamics at 10 %: C 23.09 < B 23.48). A vs C at any-dip flips with Trotter error (official A 27.32 < C 28.58; exact C 28.49 < A 31.68). Robust: A lowest for dip levels >= 5 % in all dynamics.",
    "limitations": "-", "script": "r5b_ranking.py", "command": CMD.format("r5b_ranking.py")})
ev.append({
    "id": "R5a", "question": "Do the exact-limit any-dip thresholds equal the pre-registered continuum predictions?",
    "hypotheses": ["H0 (claim): exact 2400-step any-dip A 32.45, B 34.15, C 30.10 vs predictions 32.5, 34.2, 30.15", "H1: they differ by more than the scan step"],
    "independent_variable": "dynamics / acquisition", "dependent_variable": "any-dip threshold (0.01 grid)",
    "fixed_parameters": "reference centres, +-2.667 Hz", "reference": "agents/agent1_physics/predictions.json",
    "result": {m: {k: (v["max_pass"] if isinstance(v, dict) and "max_pass" in v else v) for k, v in r5[m].items()
                   if k.startswith("L2400") or k.startswith("L300") or k.startswith("continuum")} for m in "ABC"},
    "conclusion": "Confirmed: exact 2400 steps 32.49, 34.16, 30.11; own continuum (discrete-time line shape) 32.50, 34.15, 30.10; Lorentzian 32.45, 34.10, 30.10. Exact 300 steps: 31.68, 33.05, 28.49.",
    "limitations": "Orchestrator used a 0.05 grid; mine 0.01 (differences <= 0.04).", "script": "r5_exact_limit.py", "command": CMD.format("r5_exact_limit.py")})
ev.append({
    "id": "R5b", "question": "Is A's lower official any-dip value caused by Trotter error in line heights rather than spacing?",
    "hypotheses": ["H0 (claim): heights", "H1: positions/spacing", "H2: dispersive phase errors"],
    "independent_variable": "line set: exact/Trotter positions x exact/Trotter amplitudes", "dependent_variable": "continuum any-dip threshold",
    "fixed_parameters": "infinite-acquisition discrete-time line shape, reference centres", "reference": "exact 32.50, Trotter 28.25",
    "result": {m: {k: r5[m][k]["max_pass"] for k in r5[m] if k.startswith("continuum_any[") and "lorentz" not in k} for m in "ABC"},
    "conclusion": "Confirmed for A: exact positions + Trotter amplitudes 28.00 (= Trotter 28.25); Trotter positions + exact amplitudes 32.45 (= exact 32.50). Phase twist < 1e-3 rad (R2d), so H2 rejected. Decomposition of the official 27.32 vs exact-limit 32.49: Trotter 4.36 s^-1 (exact-300 31.68), finite 0.75 s acquisition 0.81 s^-1. Same mechanism raises B (34.15 -> 34.85) and C slightly (30.10 -> 30.30).",
    "limitations": "A's 0.94 Hz doublets paired by rank (nearest pairing not one-to-one); the critical multiplet is unaffected.",
    "script": "r5_exact_limit.py", "command": CMD.format("r5_exact_limit.py")})
for eid, key, q, iv in (("R6a", "E1", "Does raising J strength at fixed critical spacing raise the tolerance?", "x: J02 = 11.8 + x, J12 = 17.9 + x (J12 - J02 = 6.1 fixed), x = -4..8"),
                        ("R6b", "E2", "Is the tolerance proportional to the inner spacing J12 - J02?", "J12 = 14.9..20.9 at J02 = 11.8"),
                        ("R6c", "E3", "Does raising the small coupling J01 raise the tolerance?", "J01 = 0.9..5 Hz"),
                        ("R6d", "E5", "At fixed J, does the shift difference (roof effect) change the tolerance?", "delta difference 0.03..0.5 ppm, J = 7.92 Hz (molecule C)")):
    rows = r6[key]
    keep = ("x", "J02", "J12", "J01", "delta_diff_ppm", "dnu_over_J", "outer_inner_ratio", "n_char", "min_spacing_Hz", "critical_spacing_Hz",
            "trotter_any", "trotter_10%", "exact_any", "exact_10%")
    concl = {
        "E1": "Mean of the two strong couplings doubles (10.85 -> 22.85 Hz): exact any-dip 29.91-32.24 (-5.6..+1.8 % of x=0), exact 10 % 20.43-21.51 (-3.8..+1.3 %); official Trotter any-dip FALLS 30.26 -> 24.42 (Trotter amplitude error grows with J), 10 % 20.69 -> 19.25. Pre-registered H_J (>= +50 %) refuted; H_s (+-10 %) holds for exact dynamics and at 10 %; official any-dip drifts just outside +-10 % (-10.6/+10.8 %) in the direction opposite to H_J.",
        "E2": "Threshold proportional to the inner spacing from 4.1 to 9.1 Hz: official 10 % ratio 3.14-3.31 per Hz, exact 10 % 3.26-3.52, exact any-dip 4.64-5.24 (pi*sqrt(3) = 5.44 is the isolated-pair continuum limit), official any-dip 4.35-4.79. At 3.1 Hz (2.3 bins) the ratio drops (grid limit).",
        "E3": "J01 0.9 -> 2.5 Hz: no change (27.32 -> 27.38, merged). J01 = 3.0: never passes; 3.5-5 Hz: any-dip 13.76 -> 21.21, 10 % none -> 15.94, i.e. a LARGER coupling LOWERS the tolerance. H_J refuted.",
        "E5": "Critical spacing is exactly J = 7.92 Hz for every shift difference, yet the tolerance rises from 21.58 to 37.12 (any) and 17.18 to 27.41 (10 %) as delta-nu/J goes 1.9 -> 31.6 (outer/inner 0.36 -> 0.94). Both 'J alone' and 'spacing alone' are refuted; spacing + relative intensity (roof effect) is supported. At C's actual delta-nu/J = 4.6 the roof effect costs ~15 % (10 %) to ~23 % (any-dip) relative to the weak-coupling limit."}[key]
    ev.append({"id": eid, "question": q,
               "hypotheses": ["H_J: tolerance scales with J strength", "H_s: tolerance set by the smallest resolvable spacing (and the intensities of the two lines)"],
               "independent_variable": iv, "dependent_variable": "highest passing noise (any, 10 %), official Trotter and exact, 300 steps",
               "fixed_parameters": "other couplings and shifts as in the files; centres from exact sticks merged < 2.667 Hz (validated = reference-based for A, B, C); criterion.md v1",
               "reference": "pre-registered predictions in r6_jcoupling.py docstring",
               "result": [{k: r.get(k) for k in keep if k in r} for r in rows], "conclusion": concl,
               "limitations": "Artificial Hamiltonians, no supplied reference; a few low-noise (< 1.2 s^-1) non-monotonic windows (E1 x=6/8, E2 J12=14.9 exact) and a narrow pass window for J01 = 3.5 (12.82-13.76).",
               "script": "r6_jcoupling.py", "command": CMD.format("r6_jcoupling.py")})
ev.append({
    "id": "R6e", "question": "Are Agent 3's J-sweep numbers (results/j_coupling_sensitivity.csv) reproducible?",
    "hypotheses": ["H0: same thresholds from independent code", "H1: discrepancies"],
    "independent_variable": "9 rows (A_J02 9.5/13/16, A_J12 20, A_J01 3.75, B_J12 10/6, C_J01 4/12)",
    "dependent_variable": "any-dip and 10 % thresholds, number of characteristic peaks, monotonicity",
    "fixed_parameters": "own Hamiltonian builder, own Trotter, own criterion", "reference": "Agent 3 CSV", "result": r6b,
    "conclusion": "9/9 rows identical (any, 10 %, n_char, monotonic flag).",
    "limitations": "Spot check, not all 159 rows.", "script": "r6b_spotcheck_agent3.py", "command": CMD.format("r6b_spotcheck_agent3.py")})
(HERE / "evidence.json").write_text(json.dumps(ev, indent=1))
print(len(ev), "entries written")
