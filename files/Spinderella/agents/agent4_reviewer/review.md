# Agent 4: independent scientific review (HQS NMR challenge)

Date 2026-11-10. Reviewer: Agent 4. I took no part in the implementation. Every number below
comes from my own scripts in this folder (outputs in `out/`, collected in `evidence.json`).

**What is independent.** I wrote my own JSON parser and Hamiltonian builder, my own exact and Lie-Trotter
dynamics, my own vectorised transform, and my own peak, dip, prominence, threshold and matching code.
The only code I share with the team is `nmr_pipeline.spectrum_from_signals` (the official spectrum,
as the brief allows). Checks on the shared pieces:
- My own transform equals the pipeline to <= 8e-14.
- My Lie-Trotter correlations equal the official signal files to <= 2e-12.
- My H equals the pipeline's H to 1e-13 rad/s after reversing the qubit order.

Run everything from the repo root:
`cd agents/agent4_reviewer && for s in r1_initial_states r2_physics_axis_trotter r3_matching r4_threshold_sensitivity r4b_zero_fill_J01 r5_exact_limit r5b_ranking r6_jcoupling r6b_spotcheck_agent3 build_evidence; do ../../.venv/bin/python $s.py; done`
The full run takes about 5 min. r5 takes about 2 min and r4 about 1 min.

## Verdicts

| Claim | Verdict | Key number |
|---|---|---|
| C1 initial states and weights | CONFIRMED WITH CAVEATS | The states, sum rules and normalisation hold to 1e-14. The weight N-2w is correct but **not unique**. All-plus alone is exact for C. |
| C2 pipeline physics | CONFIRMED WITH CAVEATS | Rotating frame exact (1.8e-7 Hz). '+ Im' correct. Axis 0.42-0.93 Hz low. Trotter shifts <= 0.715 Hz, but **amplitudes change up to 19 %** (omitted). |
| C3 matching | CONFIRMED WITH CAVEATS | Unanimous over 126 choices. Worst relative margin 13 % (overlap-only cosine); >= 52 % with full-window metrics. A second hint in the upstream notebook should be disclosed. |
| C4 thresholds | numbers CONFIRMED; ranking WEAKENED | All 12 thresholds reproduced exactly and monotonic. A 1-bin tolerance makes A and B never pass. **The B-vs-C order flips** with the dip level and with Trotter error. |
| C5 exact limit and mechanism | CONFIRMED | Exact 2400-step thresholds 32.49 / 34.16 / 30.11. Swap test: Trotter **amplitudes** give 28.00 for A, Trotter positions give 32.45. |
| C6 J-coupling answer | CONFIRMED WITH CAVEATS | Pre-registered controlled tests refute "scales with J" (E1, E3) and support "spacing + intensities" (E2, E5). The pi*sqrt(3) rule is an upper bound. The cross-molecule ordering is not evidence. |

---

## C1. Initial states, weights, normalisation (`r1_initial_states.py`)

**Tests.**
1. Template states. I checked every notebook template state against X_i and X_tot. The largest eigen-equation residual is 2.1e-16 and |Gram - I| is 4.4e-16.
2. Ten weightings. I compared each with the trace Tr(U^-k A U^k X_tot) under the same official dynamics.
3. Sum rules on the official files, and the spin-vs-Pauli normalisation.

**Numbers.**

*Sum rules and normalisation*
- C_xx(0) = 24, 24, 8 = N 2^N. |C_yx(0)| <= 1.5e-15.
- The official signals equal the exact trace to 2.0e-12.
- Pauli/spin ratio = 32, 32, 16 (= 4·2^N). The normalised spectra are identical to 2e-13.

*Weightings.* Each value is the largest |difference| of the final spectrum at noise 5, as a fraction of the maximum.

| Weighting | A | B | C |
|---|---|---|---|
| N-2w (claimed) | 2e-14 | 2e-14 | 2e-14 |
| equal weights | signal 8.9e-16, i.e. Tr X_tot = 0; spectrum is normalised rounding noise (diff 0.99) | same | ValueError |
| only all-plus | 0.048 | 0.040 | **2e-14 (exact)** |
| -(N-2w) (flipped sign) | 2e-14 | 2e-14 | 2e-14 |
| w (number of \|->) | 2e-14 (scale -1/2) | 2e-14 | 2e-14 |
| N-w (all weights >= 0) | 2e-14 (scale +1/2) | 2e-14 | 2e-14 |
| (N-2w) + (N-2w)^2 | 2e-14 (scale 1) | 2e-14 | 2e-14 |
| (N-2w)^3 | 0.040 | 0.034 | 2e-14 |
| integer value of the bit label | 0.52 | 0.76 | 0.45 |
| spin 0 only | 1.00 | 1.00 | 0.93 |

**Findings.**

1. **The correct weights are not unique.** Any weight whose part that is odd under flipping all spins equals b(N-2w), with b ≠ 0, gives the exact spectrum. Two terms cancel exactly:
   - a constant, because Tr(X_tot(t)) = 0;
   - any part that is even under the global flip, because H conserves the Z parity and X_tot, Y_tot are odd under it.

   The non-negative weights N-w (the number of |+> qubits) reproduce the trace exactly. So the following statements are false as worded:
   - "signed weights are needed";
   - "the chosen one is the only weighting that reproduces the trace".

2. **The sign is invisible in the output.** Flipping the sign of all weights gives the same spectrum after `/max|.|` and `np.abs`.
3. **All-plus alone is exact for C.** For N = 2 the |++> and |--> terms are equal and the m = 0 states have weight 0. For A and B, all-plus alone gives a 4-5 % intensity error.
4. **Minor code issue.** For equal weights in A and B the signal is 8.9e-16, not exactly 0. The new guard `spectrum_maximum == 0` does not fire, so the code silently returns normalised rounding noise.

**Recommended wording.** "The 2^N states H·RZ(pi b_i)|0> form the X-product eigenbasis of X_tot, with eigenvalue m_s = N - 2w. The trace Tr(A(t) X_tot) equals sum_s m_s <s|A(t)|s>.

The weight must be m_s up to an overall factor. Two kinds of change cancel exactly:
- adding a constant, because Tr X_tot(t) = 0;
- adding any function that is unchanged when all spins are flipped, because of Z-parity symmetry.

For example, the non-negative weights N - w give the same spectrum. Equal weights give exactly zero. The all-plus state alone is exact for the two-spin molecule C but distorts the intensities of A and B by up to 4.8 % of the maximum. Pauli-vs-spin conventions multiply the signal by 4·2^N, which the normalisation removes. C_xx(0) = N 2^N and C_yx(0) = 0 are verified to 2e-15."

## C2. Pipeline physics (`r2_physics_axis_trotter.py`)

**Tests and numbers.**

*Rotating frame*
- ||[H, Z_tot]|| = 0 exactly.
- Lab-frame lines (full H, ~1.6e9 rad/s) equal the rotating-frame lines plus the rigid shift -2 h_mean / 2 pi, to <= 1.8e-7 Hz for 12, 12 and 4 lines.

*Sign.* I used a J = 0 copy of A, whose true shifts are 5.48, 5.79 and 5.97 ppm.
- '+ Im' puts the peaks at 5.4800, 5.7893 and 5.9680 ppm, within one grid bin of the true shifts.
- '- Im' puts them at 5.5227, 5.7013 and 6.0107 ppm, which is the mirror image (5.5233, 5.7033, 6.0133).

*Axis*
- I derived the evaluation frequencies f_j = (2j-(L-1))/((2L-1)dt) by hand. My kernel reproduces the pipeline to 8e-14.
- Label - f = -0.6667 Hz - 0.001667 f.
- Over the strong lines the offset is -0.42 to **-0.93** Hz. The -0.93 comes from B's outermost line at 159.2 Hz.

*Trotter.* Floquet lines of the one-step propagator compared with the exact lines:

| | max shift dt / dt/2 / dt/4 (Hz) | max change of absorptive amplitude at dt | phase |
|---|---|---|---|
| A | 0.715 / 0.140 / 0.034 | **19.1 %** | < 1e-3 rad |
| B | 0.418 / 0.083 / 0.020 | **15.7 %** | < 1e-3 rad |
| C | 0.012 / 0.003 / 0.001 | 0.7 % | < 1e-3 rad |

*Units*
- J recovered as 2c/pi: 0.9, 11.8, 17.9 / 15.6, -0.5, 8.6 / 7.92 Hz. The XX, YY and ZZ coefficients are equal.
- nu0 = 499.99999 MHz, 500.0 Hz/ppm.
- The FWHM measured at Gamma = 10 s^-1 is 3.170 Hz, against Gamma/pi = 3.183 Hz (0.4 %).

**Verdict.** CONFIRMED WITH CAVEATS. The shift bound is right, but stating only the shift is misleading. The Trotter step leaves the line positions almost unchanged but alters the relative intensities by up to 19 %, and that is what lowers A's thresholds (C5).

**Recommended wording.** "First-order Trotter error at dt = 2.5 ms moves lines by at most 0.72 Hz (sub-bin). It changes the relative line intensities by up to 19 % (A) and 16 % (B), and by 0.7 % for C. Both errors fall about 4-5 times per halving of dt. The ppm axis draws every line 0.42-0.93 Hz low."

## C3. Matching (`r3_matching.py`)

**Test.** I scored all 6 one-to-one assignments for every combination of:
- 7 noise strengths (0.1, 0.5, 1, 2, 5, 10, 20 s^-1);
- 9 metrics:
  - cosine with linear, nearest or bin-average resampling;
  - cosine restricted to the overlap window;
  - Pearson on the overlap;
  - cosine on the reference grid;
  - -L1 of area-normalised spectra;
  - peak-position-only F1 (3 Hz);
  - negative chamfer distance;
- with and without the +0.667 Hz axis correction.

That is 126 choices.

**Numbers.**
- The winner is always A->3, B->2, C->1. No other assignment wins in any of the 126 choices.
- Worst relative margin: 13 %. This is cosine restricted to the overlap at noise 0.1: best 1.959 vs second 1.697 (A->1, B->2, C->3).
- Full-window metrics: relative margin >= 52 %. Bin-average cosine: margin 1.31-1.64 (64-66 %) at every noise.
- MD5 of the C file = MD5 of spectrum_1.

**Caveats.**
1. Overlap-only metrics are unreliable here. A's simulated window overlaps spectrum_1 only over 6.03-6.15 ppm, where neither has a line. The two smooth tails give cosine 0.99. Row-wise, the overlap metrics would map A to spectrum_1 (21 of 126 choices); only the one-to-one constraint rescues them.
2. **A second hint exists in the supplied material.** The upstream notebook (commit f27f229, cell 29) loads molecule A (cell 4) and overlays `spectrum_3.pkl`. Together with the byte copy for C, two of three assignments are hinted by the organisers' files. The decision does not use them, but the report should say so.

**Recommended wording.** "A->spectrum_3, B->spectrum_2, C->spectrum_1 (expected 3/3). The assignment wins in all 126 combinations of noise strength (0.1-20 s^-1), similarity metric, resampling and axis correction. The smallest margin, 13 %, comes from a cosine restricted to the window overlap, which is unreliable for A vs spectrum_1 because the overlap contains no lines. All full-window metrics give margins >= 52 %. Two assignments are also hinted by the supplied files: the C file is a copy of spectrum_1, and the original notebook overlays molecule A with spectrum_3. Neither was used for the decision."

## C4. Noise tolerance (`r4_threshold_sensitivity.py`, `r4b_zero_fill_J01.py`, `r5b_ranking.py`)

### Reproduction

I scanned noise from 0.01 to 100 in steps of 0.01 (10 000 points per molecule), plus noise 0. The test used my own local maxima, prominence, grouping and dip code.

| Dip level | A | B | C |
|---|---|---|---|
| 10 % (primary) | 19.97 / 19.98 | 22.85 / 22.86 | 23.22 / 23.23 |
| any dip | 27.32 | 33.36 | 28.58 |
| 5 % | 22.15 | 25.99 | 25.51 |
| 19 % | 17.11 | 19.13 | 20.02 |

- All 12 numbers are reproduced exactly.
- There is exactly one pass-to-fail transition over 0.01-100 for every molecule and level. All three pass at noise 0. Nothing is non-monotonic.
- Critical pairs at 10 % are confirmed: A 6.05 Hz (5.4728 / 5.4850 ppm), B 7.02 Hz (5.8938 / 5.9079 ppm), C 7.92 Hz (6.2932 / 6.3091 ppm).
- Any-dip first failures are all lost maxima:
  - A: the inner line at 5.4850 ppm.
  - B: the **outer** line at 5.9250 ppm (the 8.58 Hz J12 pair), a position failure. With a 3-bin tolerance, B's any-dip value is 33.53.
  - C: both outer lines.

### Sensitivity

| Variant | A | B | C | Ranking |
|---|---|---|---|---|
| merge distance 1 / 3 bins | same | same | same | same |
| merge distance 0.5 bin (0.94 Hz doublets of A become peaks) | never passes | 22.85 | 23.22 | undefined |
| position tolerance 3 bins | same at 10 % | 33.53 at any dip | same | same |
| **position tolerance 1 bin** | **never passes (not even at noise 0)** | **never passes** | 23.22 | undefined |
| peak finder (height, prominence) from (0.02, 0.005) to (0.30, 0.10) | same | same | same | same |
| centres from the simulation's own spectrum at 1, 2, 5 s^-1 | same | 33.53 at any dip | same | same |
| centres from exact stick lines | same | same | same | same |
| signed spectrum (no `np.abs`) | same at all 4 levels | same | same | same |
| no position tolerance, count >= k maxima per multiplet | same at all 4 levels | 33.53 at any dip | same | same |
| count == k | same thresholds; non-monotonic below 1.5 s^-1 (ringing) | same | same | same |
| dip relative to the larger peak, 10 % | 24.07 | 28.19 | 28.58 | A<B<C |

**Why a 1-bin tolerance fails.** At low noise the simulated maxima already sit up to 1.73 Hz (A), 1.86 Hz (B) and 1.30 Hz (C) from the reference centres. Three things add up: the axis offset, the grid sampling and the Trotter shift. So 70 % of the 2.667 Hz tolerance is used before any broadening. The budget in `criterion.md` (<= ~2.0 Hz) uses the *mean* displacement (-0.54 to -0.77 Hz). It does not use the per-line maximum, which includes Trotter shifts of up to 0.72 Hz.

### Ranking (look at the alternative dip levels)

Highest passing noise strength (s^-1). Order goes from the lowest tolerance to the highest.

| Dynamics, dip level | A | B | C | Order |
|---|---|---|---|---|
| official Trotter, any | 27.32 | 33.36 | 28.58 | A < C < B |
| official Trotter, 5 % | 22.15 | 25.99 | 25.51 | A < C < B |
| official Trotter, 10 % | 19.97 | 22.85 | 23.22 | A < B < C |
| official Trotter, 19 % | 17.11 | 19.13 | 20.02 | A < B < C |
| exact 300 steps, any | 31.68 | 33.05 | 28.49 | **C < A < B** |
| exact 300 steps, 10 % | 21.23 | 23.48 | 23.09 | **A < C < B** |
| exact 2400 steps, any | 32.49 | 34.16 | 30.11 | C < A < B |
| exact 2400 steps, 10 % | 21.62 | 23.53 | 23.12 | A < C < B |

**B vs C.** The order of B and C is not determined by the data:
- it flips between the any/5 % and the 10/19 % levels;
- at the primary level, B < C (by 0.37 s^-1, 1.6 %) only because Trotter error lowers B from 23.48 to 22.85;
- in exact dynamics, C < B at 10 %.

**A vs C at any dip.** A has the lower threshold only in the official Trotter simulation.

**The only robust statement.** A has the lowest tolerance at dip levels >= 5 % in all three dynamics.

### J01 = 0.9 Hz exclusion

- **On the official grid the exclusion is legitimate.** The 300-step spectrum on its 1.333 Hz grid never shows two maxima in any of A's four 0.94 Hz doublets. At noise 0 the counts are 1, 1, 0 and 1 maxima.
- **It is a grid/acquisition property, not a law.** I zero-filled the transform: same signal, frequency evaluated on finer grids, as the notebook markdown says it does ("zeros are added").

| Signal | 0.333 Hz grid | 0.05 Hz grid |
|---|---|---|
| Official Trotter signal: all 4 doublets show two maxima up to | 1.25 s^-1 | 1.95 s^-1 |
| ... with a >= 10 % dip up to | 0.4 s^-1 | 0.6 s^-1 |
| Exact dynamics: two maxima up to | never | 0.35 s^-1 |
| ... with a >= 10 % dip | never | never |

The official signal resolves the doublets better than the exact one, which is a Trotter artefact. Longer exact acquisitions keep the doublets up to 3.85 s^-1 (1200 steps) and 4.65 s^-1 (2400 steps).

**Consequence.** If the 0.9 Hz splitting counts as a "distinct peak" (spectrum_3 resolves it), A never passes on the official grid. With zero filling, A would pass only below about 0.35-2 s^-1. The reported ~20 s^-1 refers to features the 0.75 s acquisition can show on the notebook grid. This is a defensible but consequential choice and must be stated as one.

**Recommended wording.** "With the stated criterion, the highest tolerable noise_strength is:
- 19.97 s^-1 for A (fails at 19.98);
- 22.85 s^-1 for B;
- 23.22 s^-1 for C.

Pass/fail is monotonic between 0.01 and 100 s^-1. The values do not change with the merge distance (1-3 bins), the peak-finder thresholds, the source of the characteristic peaks, a count of maxima without position tolerance, or the final np.abs.

They do depend on two choices:
- the dip level (any dip: 27.3, 33.4, 28.6; 19 %: 17.1, 19.1, 20.0);
- the position tolerance (with 1 bin, A and B never pass, because the axis offset, the grid and the Trotter shift already displace the maxima by up to 1.9 Hz).

A has the lowest tolerance at every dip level >= 5 %. B and C cannot be ranked: their order flips with the dip level and with Trotter error. The 0.9 Hz J01 doublets of A are excluded because the 300-step spectrum on the notebook grid never shows them at any noise; if they counted, A would never pass."

## C5. Exact limit and mechanism (`r5_exact_limit.py`)

**Numbers.**

*Any-dip thresholds (s^-1)*

| | A | B | C |
|---|---|---|---|
| exact dynamics, 2400 steps | 32.49 | 34.16 | 30.11 |
| own continuum, discrete-time line shape | 32.50 | 34.15 | 30.10 |
| own continuum, Lorentzian | 32.45 | 34.10 | 30.10 |
| pre-registered prediction | 32.5 | 34.15-34.2 | 30.15 |
| Trotter, 2400 steps | 28.24 | 34.83 | 30.30 |
| exact dynamics, 300 steps | 31.68 | 33.05 | 28.49 |

The orchestrator's 32.45 / 34.15 / 30.10 were computed on a 0.05 grid.

*Swap test.* Continuum any-dip thresholds when line positions and line amplitudes come from different dynamics:

| Positions + amplitudes | A | B | C |
|---|---|---|---|
| exact + exact | 32.50 | 34.15 | 30.10 |
| Trotter + Trotter | 28.25 | 34.80 | 30.30 |
| exact positions + Trotter amplitudes | **28.00** | 34.85 | 30.30 |
| Trotter positions + exact amplitudes | **32.45** | 34.15 | 30.10 |

The phase twist is < 1e-3 rad, so dispersive errors are ruled out.

**Decomposition of A's official 27.32 vs the exact limit 32.49 (any dip).**
- Trotter error: 4.36 s^-1 (exact at 300 steps gives 31.68).
- The finite 0.75 s acquisition: 0.81 s^-1.
- At 10 %: Trotter 1.26 s^-1 (21.23 → 19.97), acquisition 0.39 s^-1.

**Verdict.** CONFIRMED. The critical spacing itself barely moves: the Trotter shift is 0.7 Hz on lines 6 Hz apart, and the swap test shows positions do not matter.

**Recommended wording.** "In the exact limit (exact evolution, 6 s acquisition) the any-dip thresholds are 32.5 (A), 34.2 (B) and 30.1 (C) s^-1. These equal the continuum predictions written before the sweep. A's official value of 27.3 is lower mainly because first-order Trotter error changes the relative line intensities (4.4 s^-1 of the 5.2 s^-1 gap). A swap test confirms it: exact line positions with Trotter intensities reproduce the Trotter threshold, and the reverse does not. The remaining 0.8 s^-1 is the finite 0.75 s acquisition."

## C6. J-coupling answer (`r6_jcoupling.py`, `r6b_spotcheck_agent3.py`)

**Agent 3's experiment table** (`results/j_coupling_sensitivity.csv`, 159 rows) is present. I reproduced 9 of its rows exactly with independent code: any-dip, 10 %, number of characteristic peaks, and the monotonic flag. The rows are A_J02 9.5 / 13 / 16, A_J12 20, A_J01 3.75, B_J12 10 / 6 and C_J01 4 / 12. Their one-at-a-time sweeps already show non-monotonic dependence on J:
- A_J02 9.5 -> 14 Hz: 39.5 -> 14.7;
- A_J01 > 2.67 Hz: drop.

These sweeps are consistent with the spacing hypothesis and inconsistent with the J-strength hypothesis.

**My own controlled experiments.** I pre-registered the predictions in the script docstring. Each run uses the official Trotter setting and exact dynamics.

**E1. J02 and J12 shifted together by x = -4...+8 Hz, with J12 - J02 = 6.1 fixed.**
- The mean of the two strong couplings doubles, from 10.85 to 22.85 Hz.
- H_J predicted a rise of >= 50 %.
- Exact dynamics, any dip: 29.91-32.24, which is -5.6 % to +1.8 % of x = 0.
- Exact dynamics, 10 %: 20.43-21.51, which is -3.8 % to +1.3 %.
- Official Trotter, any dip: **falls** from 30.26 to 24.42. Official 10 %: 20.69 to 19.25.

**H_J refuted.**

**E2. J12 varied, giving an inner spacing of 4.1-9.1 Hz.** The threshold is proportional to the spacing. Ratio of threshold to spacing:
- official 10 %: 3.14-3.31;
- exact 10 %: 3.26-3.52;
- official any dip: 4.35-4.79;
- exact any dip: 4.64-5.24.

At 3.1 Hz (2.3 grid bins) the ratio drops, because the grid is the limit there.

**E3. J01 = 0.9 -> 5 Hz.**
- Up to 2.5 Hz nothing changes (27.32 -> 27.38): the doublets are still merged.
- At 3.0 Hz the spectrum never passes.
- From 3.5 to 5 Hz the any-dip threshold is 13.8 -> 21.2. At 10 % it goes from never passing to 15.9.

**A larger J lowers the tolerance. H_J refuted.**

**E5. Molecule C, J = 7.92 Hz fixed, shift difference 0.03-0.5 ppm.** The critical spacing is exactly J for every shift difference. As Δnu/J goes from 1.9 to 31.6 (outer/inner intensity 0.36 -> 0.94), the tolerance still rises:
- any dip: 21.6 -> 37.1;
- 10 %: 17.2 -> 27.4.

So neither "J alone" nor "spacing alone" holds; the intensities (roof effect) matter. At C's actual Δnu/J = 4.6, the roof effect costs about 15 % (10 % level) to 23 % (any dip) against the weak-coupling limit.

**Verdict.** CONFIRMED WITH CAVEATS. The causal claim is supported by interventions, not only by correlation. Each of E1, E3 and E5 could have falsified it:
- E1 would have shown a rise of >= 50 %;
- E3 would have shown a rise;
- E5 would have shown a constant tolerance.

None of these happened.

**Caveats.**
1. The three real molecules alone cannot support the claim. Their spacings span only 6.05-7.92 Hz, and their ordering (C4) flips with the criterion and with Trotter error. Any sentence like "A < B < C follows 6.05 < 7.02 < 7.92 Hz" must be removed: in exact dynamics the 10 % order is A < C < B.
2. "Gamma* ~ pi sqrt(3) x spacing" (5.44 x spacing) is the continuum limit for an **isolated pair of equal lines**, so it is an upper bound. Measured any-dip ratios are:
   - 4.5 (A) and 4.75 (B) official, 4.35-4.79 in E2;
   - 3.6 for C (roof effect).

   The primary 10 % level follows about 3.3 x spacing.
3. In the official Trotter setting, a larger J lowers the any-dip tolerance even at fixed spacing (E1, -19 % over the range). This is a Trotter-amplitude artefact and should be named.
4. The tolerance is discontinuous at the 2-bin merge boundary. In Agent 3's A_J02 sweep it jumps from 5.11 (J02 = 15.0) to 49.14 (J02 = 15.5) when |J12 - J02| drops below 2.667 Hz. This is a property of "unresolvable splittings are ignored" (the same rule as the J01 exclusion) and should be stated.

**Recommended wording.** "The tolerance is not set by the J values themselves. Three controlled experiments show this:
- Raising both strong couplings of A by up to 8 Hz, with their difference fixed, leaves the exact-dynamics tolerance within -6 %/+2 %.
- Raising A's small coupling J01 from 0.9 to 5 Hz lowers the tolerance, from 27 to 21 s^-1 at any dip.
- Changing only the shift difference of C at fixed J changes it by a factor 1.7.

The tolerance is set by the smallest line spacing that the acquisition can resolve, together with the relative intensity of the two lines:
- A: J12 - J02 = 6.1 Hz;
- B: J01 - J12 = 7.0 Hz;
- C: J = 7.9 Hz, with the outer line 1.54 times weaker (roof effect, Δnu/J = 4.6).

For roughly equal neighbouring lines, the 10 % criterion fails at about 3.3 x spacing in s^-1, and the any-dip criterion at about 4.4-4.8 x spacing. pi*sqrt(3) = 5.44 x spacing is the limit for an isolated equal pair. Splittings below about 2.7 Hz (two grid bins), such as J01 = 0.9 Hz in A, are invisible in the 0.75 s acquisition and do not count."

---

## Required corrections, most severe first

1. **Do not rank B against C, and do not use the A/B/C ordering as evidence about J.**
   - B vs C flips with the dip level (any/5 %: C < B; 10/19 %: B < C).
   - It also flips with Trotter error: at 10 %, the exact order is C 23.09 < B 23.48, while the official order is B 22.85 < C 23.22.
   - A < C at any dip is itself a Trotter artefact (exact: C 28.49 < A 31.68).
   - Only "A lowest at dip >= 5 %" is robust (R4d).
2. **State the J01 exclusion as a consequential choice.** If A's 0.9 Hz doublets count as distinct peaks, A never passes on the notebook grid. With zero filling (which the notebook markdown says is done), they survive only to <= 0.35 s^-1 (exact) or 1.95 s^-1 (Trotter). The ~20 s^-1 answer holds only for "features the 0.75 s acquisition can show on the notebook grid" (R4c).
3. **Trotter error changes line intensities by up to 19 % (A) and 16 % (B).** Add this to every place that says Trotter error is harmless because it shifts lines by < 0.72 Hz. The intensity error is what lowers A's any-dip threshold by 4.4 s^-1 (R2d, R5b).
4. **Position tolerance.** Report that a 1-bin tolerance makes A and B fail at every noise strength, including 0. Correct the budget in `criterion.md`: it uses mean displacements, but the per-line displacement at low noise is already 1.73 Hz (A) and 1.86 Hz (B), including Trotter shifts of up to 0.72 Hz (R4b).
5. **C1 wording on weights.**
   - The weights are not unique (constant and flip-even parts cancel), and non-negative weights N - w work.
   - "Signed weights are required" and "the only correct weighting" are false.
   - The global sign is invisible after np.abs.
   - "All-plus gives wrong intensities" is false for C (exact) and only 4-5 % for A and B (R1b).
6. **pi*sqrt(3) x spacing is an upper bound** for an isolated equal pair. Quote the measured ratios instead: about 3.3 at 10 %, 4.4-4.8 at any dip, 3.6 for C (R6b).
7. **B's any-dip threshold (33.36) is a position failure** of the outer 8.58 Hz line at 5.9250 ppm. The 7.02 Hz dip survives to 33.53. The critical splittings quoted in C4 belong to the 10 % level; label them so (R4a).
8. **Disclose the second matching hint**: the upstream notebook overlays molecule A with spectrum_3. Also say that overlap-only similarity is unreliable for A vs spectrum_1, where the tails alone give cosine 0.99 (R3).
9. **Axis offset is 0.42-0.93 Hz**, not 0.41-0.92 Hz. B's outermost line at 159.2 Hz is drawn 0.93 Hz low (R2c).
10. **Guard in `time_signal_to_spectrum`.** `spectrum_maximum == 0` does not catch a rounding-level signal: equal weights give 8.9e-16 for A and B and a normalised-noise spectrum without any error. Use a relative test, for example against C_xx(0) (R1b). This file is the team's mirror (`src/nmr_pipeline.py`) and the notebook cell. I did not edit either.
11. **Consistency between documents.** Agent 1's report table gives the C prediction as 30.2, but `predictions.json` says 30.15. It labels B's predicted critical feature as the "outer 8.58 Hz line", but the continuum merge events show the inner 7.02 Hz pair at 34.15 and the outer line at 34.3, essentially at the same time.
12. **Discontinuity at the merge boundary.** In the J discussion, state that the tolerance is discontinuous where a splitting crosses 2.667 Hz (A_J02 15.0 -> 15.5: 5.11 -> 49.14). This is a property of the criterion, not of the physics.

## Not run / limits of this review

- **NOT RUN: the Estimator path and the notebook itself.** I did not execute `nmr.ipynb` or the StatevectorEstimator loop. I used the official signal files (`results/signals/*.npz`) and my own product formula, which equals them to 2e-12.
- **NOT RUN: a check that notebook sections 8 and 9 equal `src/nmr_pipeline.py`.** That is the orchestrator's job. I only read cells 33-40.
- **NOT RUN: the full J table.** I spot-checked 9 of the 159 rows of `results/j_coupling_sensitivity.csv`. I did not audit the rows labelled "detection failure" that still carry numeric thresholds (for example A_J12 30.0: 0.23; B_J12 2.0: 51.41).
- **NOT RUN: matching under other Trotter step sizes**, and uniform scaling of all couplings.
- **Artificial-J limits.** The J experiments use artificial Hamiltonians with stick-based characteristic peaks. In r4 I checked that these give thresholds identical to the reference-based ones for the real molecules.
