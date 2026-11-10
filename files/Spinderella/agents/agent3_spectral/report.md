# Agent 3 — spectral analysis and noise tolerance

Agent 3 ran all experiments. Its harness did not allow it to write report files, so the
orchestrator transcribed this report from Agent 3's final message. Every number is in
`evidence.json` (16 entries), the CSV files here and `results/`.

## Task A — matching (one-to-one, all 6 permutations)

**A -> spectrum_3, B -> spectrum_2, C -> spectrum_1.** All 27 choices pick this permutation with no
tie: 6 metrics (cosine, Pearson, NRMSE, peak-match fraction, peak F1, multiplet score) x noise
0.5/2/5 s^-1 x 2 resampling methods (bin average, point interpolation).

| Metric | Smallest margin, best minus second-best total over 3 molecules |
|---|---|
| cosine | 1.37 |
| Pearson | 1.42 |
| NRMSE | 0.043 (0.26 vs 0.32; weakest, the window is mostly flat baseline) |
| peak-match fraction, peak F1, multiplet score | 2.0 (of 3) |

- At noise 2, cosine of the matched pair vs the best other reference: A 0.742 vs 0.007,
  B 0.767 vs 0.004, C 0.828 vs 0.000.
- Peak-match fraction is 1.0 for matched pairs and 0.0 for every other pair.
- Multiplet counts and centres agree; simulated splittings come in 1.33 Hz steps.
- Simulated peaks sit -0.54 (A), -0.77 (B), -0.67 (C) Hz from the reference on average (the known
  half-bin axis shift).
- `files/hqs_spectrum_tools_molecule C_B11.7433T.pkl` was looked at only after the decision; it is
  identical to spectrum_1, which agrees.

## Task B — noise tolerance

Criterion: `criterion.md` v1, fixed before any sweep and not changed.

| | highest pass | first fail | critical pair (reference) | origin (first-order value) | any dip / 5 % / 19 % | exact signals | 600 / 1200 steps |
|---|---|---|---|---|---|---|---|
| A | 19.97 | 19.98 | 6.053 Hz, 5.47285 / 5.48496 ppm | J12 - J02 (6.10), inner lines of the spin-2 dd | 27.32 / 22.15 / 17.11 | 21.23 | 20.05 / 20.10 |
| B | 22.85 | 22.86 | 7.025 Hz, 5.89383 / 5.90788 ppm | J01 - J12 (7.00), inner lines of the spin-1 dd | 33.36 / 25.99 / 19.13 | 23.48 | 23.36 / 23.38 |
| C | 23.22 | 23.23 | 7.915 Hz, 6.29323 / 6.30906 ppm | J01, outer doublet of the AB quartet | 28.58 / 25.51 / 20.02 | 23.09 | 23.10 / 23.23 |

- **Monotonic**: exactly one pass -> fail transition for every molecule and every dip level (coarse,
  0.1 and 0.01 grids). The smallest dip at 0.1 s^-1 is 0.61-0.79, so ringing and the final `np.abs`
  cause no failure for the real molecules.
- **Dip level**: at 5/10/19 % the binding constraint is the critical pair's dip. With "any dip" a
  weak line turns into a shoulder first; for B this is the weakest outer line of the 8.58 Hz pair.
  **Ranking changes with the level**: B > C > A for "any dip", C > B > A for 5-19 %. A is always lowest.
- **Trotter error** moves the threshold by +1.26 (A), +0.63 (B), -0.13 (C) s^-1. The effective
  Trotter Hamiltonian changes the critical splittings by only +0.05 Hz (A) and -0.005 Hz (B), but
  shifts lines by up to 0.82 Hz and redistributes intensities (A spin-2 dd: exact 0.74/0.86/0.85/1.00
  vs Trotter 0.95/1.00/0.84/0.92). That redistribution is what moves the threshold.
- **Acquisition length** moves the threshold by -0.12 to +0.53 s^-1 at 5/10/19 %. This is grid
  sampling, not truncation: the window is down to exp(-15) at 0.75 s.
- An isolated two-Lorentzian model at the critical splitting and intensity ratio predicts
  21.69 / 23.63 / 23.32. A falls furthest below it because its 11.8 Hz neighbours fill the valley.
- Plots viewed: at the threshold the critical dip is 10.01 % vs 9.98 % (A), 10.01 % vs 10.00 % (B),
  10.00 % vs 9.98 % (C); the pair is merged at 1.5 x the threshold. Automated calls match the plots.

## Task C — J couplings

**Observational (n = 3, correlation only).** Threshold vs mean |J|: Pearson -0.9997 (Spearman -1);
vs max |J|: -0.75 (-1); vs min |J|: 0.55 (0.5); vs critical splitting: 0.92 (+1). The molecule
with the largest couplings (A) has the lowest threshold.

**Artificial experiments** (one coupling varied, everything else fixed, 300 steps, criterion v1;
references are exact stick spectra with 1 Hz Lorentzians, which on the real molecules reproduce the
supplied references' thresholds exactly and peak positions within 0.04 Hz):

| Experiment | What changes | Threshold (s^-1, 10 %) | Against H1 ("scales with J")? |
|---|---|---|---|
| C: J01 3.5 -> 16 Hz | critical splitting = J01 | 9.54 -> 38.93, monotonic | H1 and H2 agree |
| A: J02 9.5 -> 14 Hz | J12 - J02 shrinks | 28.31 -> 12.75 (Spearman -1) | yes, opposite to H1 |
| A: J02 15.5-20.5 Hz | inner pair merges; critical becomes ~17 Hz | 42-49 | |
| A: J02 21 Hz | J02 - J12 = 3.17 Hz | 5.62 | |
| A: J01 1 -> 4 Hz | J01 doublets become resolvable | 19.97 -> 13.00 | yes, opposite to H1 |
| A: J01 2.75-3.5 Hz | | never passes | |
| B: J12 8.5 -> 12.5 Hz | J01 - J12 shrinks | 23.32 -> 8.05 (Spearman -1) | yes, opposite to H1 |
| A: J12 24-28 Hz | J02 now the smallest spacing | 37.74 -> 38.63 (0.2 per Hz vs 3.4 before) | saturation |

Pooled over the artificial set: Spearman (threshold, critical splitting) = 0.989, (threshold, J) =
0.538; a fit through zero gives threshold = 2.81 x critical splitting, R^2 = 0.918.
**Verdict: H2.** The threshold is set by the smallest resolvable line spacing (often a difference of
couplings), modified by intensity ratios, not by the J strength.

## Limitations found

- The 2-bin merge rule is permissive: 2.75-3.5 Hz splittings are never resolved at any noise, so the
  practical limit of the 300-step acquisition is about 3.5-4 Hz. The real molecules are unaffected
  (smallest counted splitting 6.05 Hz; excluded ones < 1 Hz).
- Some artificial points fail in narrow windows at 0.1-0.5 s^-1 (low-noise ringing); two fail
  mid-range (A J01 = 3.75 below 5.69; B J12 = 13 between 9.17 and 10.13).
- A with J12 >= 29 Hz: detection fails at every noise because Trotter line shifts plus the axis offset
  exceed the 2.667 Hz tolerance (-2.5 Hz already at 28 Hz); a Trotter artifact, not lost resolution.
- At low noise the simulated peak heights differ from the reference because lines fall at different
  positions between grid points.

## Files

Scripts (run from the repo root with `.venv/bin/python <script>`): `common.py`, `task_a_matching.py`,
`task_b_noise.py` (`--plots-only`), `trotter_effective_lines.py`, `task_c_jcoupling.py`
(`--plot-only`), `build_evidence.py`. Outputs: `results/molecule_spectrum_matching.csv`,
`results/matching_score_matrix.csv`, `results/noise_tolerance.csv`,
`results/j_coupling_sensitivity.csv`, `results/plots/*.png`, plus the CSV/JSON files in this folder.
