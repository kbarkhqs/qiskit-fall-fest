# Validation report — HQS NMR challenge (Qiskit Fall Fest 2026)

This report lists every check that was actually run, with its number and the script that ran it.
Anything not run is listed at the end. Four specialists worked under one orchestrator:
physics (Agent 1, NumPy only, no Qiskit), implementation (Agent 2), spectral analysis (Agent 3)
and an independent reviewer (Agent 4, own code, not involved in choosing the implementation). Each
folder `agents/<agent>/` holds the scripts, `evidence.json` (question, hypotheses, variables,
reference, result, limitations, command) and a report.

## 1. Results

| Molecule | Reference spectrum | Max tolerable `noise_strength` (s⁻¹, 10 % dip) | First fail | Critical spacing (10 % level) |
|---|---|---|---|---|
| A | spectrum_3 | 19.97 | 19.98 | 6.05 Hz = J12 − J02 (spin 2, 5.48 ppm) |
| B | spectrum_2 | 22.85 | 22.86 | 7.02 Hz = J01 − J12 (spin 1, 5.90 ppm) |
| C | spectrum_1 | 23.22 | 23.23 | 7.92 Hz = J01 (outer lines, AB quartet) |

**J coupling:** the tolerance follows the smallest line spacing that must stay resolved, not the
J strength (section 6 below).

## 2. Initial states and correlation functions

| Check | Result | Where |
|---|---|---|
| `H RZ(phi)` = `RX(phi) H` on \|0> for all phi | amplitude difference 0.0 (401 angles) | Agent 1 T1, Agent 2 exp01 |
| 2^N states, orthonormal, complete | Gram − 1 = 4e-16; 8 states (A, B), 4 (C) | notebook cell after section 5, `tests/` |
| Eigenvalue of sum X = N − 2w (not the bit label) | residual 0.0 (Agent 1), 2e-16 (Agent 4) | T1, notebook check cell |
| C_xx(0) = N 2^N, C_yx(0) = 0 | 24.000, 24.000, 8.000; \|C_yx(0)\| <= 1.5e-15 | Agent 2 task4, Agent 4 r1 |
| Pauli vs spin-1/2 normalisation | constant factor 4·2^N; spectra equal to 4.5e-14 | Agent 1 T2, Agent 4 |
| Weighted sum = exact trace Tr(A(t) X_tot) | 1.4e-12 | Agent 1 T6 |
| Weights unique? | **Only up to scale.** `N − w = (m + N)/2`, `w` and `−(N − 2w)` give ½, −½ and −1 times the trace; after division by the maximum and `np.abs` the normalised spectra are identical (3e-16, final audit) | Agent 4 r1, final audit |
| All-plus state alone | exact for C; 4.8 % (A) and 4.0 % (B) of the maximum off | Agent 4 r1 |
| Placeholder detected (mutation check) | upstream template: 1 state, C_xx = 0, all detectors fire | Agent 2 mutation_check |
| Equal-weights signal (8.9e-16) caught by the guard | yes, after the review fix; test `test_guard_raises_on_equal_weights_rounding_noise` | `tests/test_pipeline.py` |

## 3. Hamiltonian, circuit and time evolution

| Check | Result | Where |
|---|---|---|
| Files obey h_i = −γB(1+δ_i)/2, c_ij = πJ_ij/2 | δ to 1.1e-10 ppm, J exact | Agent 1 T3 |
| Rotating frame exact | \|\|[H, ΣZ]\|\| = 0; lab vs rotating lines agree to 1.8e-7 Hz | orchestrator `check_circuit_and_routing.py`, Agent 4 r2 |
| Trotter gate angles = 2·c·dt on the right qubits | error 0 | orchestrator |
| Circuit unitary = file-order product of exponentials | 3.3e-16 (Qiskit 2.6 moves commuting RZ gates) | orchestrator |
| Statevector step-by-step = official Estimator loop | max difference 0.0 for A, B, C (2.4 s vs 425 s) | Agent 2, notebook section 8.1 |
| Notebook = `src/nmr_pipeline.py` | <= 8.9e-16 | Agent 2 compare_notebook_module |
| Trotter line shifts at dt = 2.5 ms | max 0.715 / 0.418 / 0.012 Hz (A/B/C); about 4x smaller per halving of dt | Agent 1 T6, Agent 4 |
| Trotter intensity change | up to 19.1 % (A), 15.7 % (B) | Agent 4 r2 |
| Routing | none needed in the simulator; linear 0-1-2 chain: 1 SWAP per step for A and B (CX 18 → 21) | orchestrator |

## 4. Spectrum and frequency axis

| Check | Result | Where |
|---|---|---|
| `+ Im` (code) vs `− Im` (markdown) | code correct; markdown sign mirrors the spectrum | Agent 1 T4, Agent 4 |
| ppm axis offset | lines drawn 0.42–0.93 Hz low; formula −0.667 Hz − ν/600 matches to 0.018 Hz; under one 1.333 Hz bin; code left unchanged | Agent 1, Agent 2, orchestrator, Agent 4 |
| References = exact sticks with 1.008 Hz Lorentzians | correlation 1.0000 | Agent 1 T5 |

## 5. Matching

| Check | Result |
|---|---|
| One-to-one, all 6 permutations | A → 3, B → 2, C → 1 |
| Specialist (27 choices: 6 metrics × 3 noise × 2 resampling) | same assignment every time; smallest margins: cosine 1.37, Pearson 1.42, NRMSE 0.043 |
| Reviewer (126 choices: 7 noise 0.1–20 × 9 metrics × axis correction on/off) | same every time; worst relative margin 13 % (cosine restricted to the overlap window), ≥ 52 % for full-window metrics |
| Physics cross-check (exact lines vs reference peaks) | max distance 0.056 / 0.24 / 0.0002 Hz for the chosen pairs |
| Notebook section 8 (independent code) | cosine 0.804 / 0.869 / 0.843; all other pairs ≤ 0.005; total 2.516 vs 0.873 |
| Hints not used for the decision | upstream notebook overlays A with spectrum_3; `hqs_spectrum_tools_molecule C` file = spectrum_1; both agree |

## 6. Noise tolerance

**Criterion.** It was fixed in writing before any sweep (`agents/agent3_spectral/criterion.md`)
and implemented twice. Agent 3 wrote its code; the orchestrator wrote the notebook version from
the text alone. The reviewer wrote a third version.
1. Characteristic reference peaks are maxima with height ≥ 0.05 and prominence ≥ 0.02, with
   components closer than 2.667 Hz merged.
2. Each needs its own simulated maximum within ±2.667 Hz.
3. Each adjacent pair needs a dip of at least 10 %.

| Check | Result |
|---|---|
| Three implementations agree | all 12 thresholds (3 molecules × 4 dip levels) identical to 0.01 s⁻¹ |
| Monotonic | one pass→fail change per molecule and level over 0.01–100; all pass at noise 0 |
| Plots agree with automated calls | critical dip 10.01 % vs 9.98 % across the threshold (A), similar for B, C |
| Unchanged by | merge distance 1–3 bins, peak-finder settings, centres from simulation or exact sticks, no `np.abs`, count-of-maxima mode (Agent 4) |
| Sensitive to | dip level (19 %: 17.11/19.13/20.02; 5 %: 22.15/25.99/25.51; any: 27.32/33.36/28.58); position tolerance (1 bin: A and B never pass, even at noise 0) |
| Trotter error | exact evolution (300 points): 21.23 / 23.48 / 23.09 |
| Acquisition length | 600 / 1200 steps: −0.12 to +0.53 s⁻¹ (grid sampling, not truncation) |
| Pre-registered predictions (written before any sweep) | continuum any-dip 32.50 / 34.15 / 30.15; measured with exact evolution and 2400 steps: 32.45–32.49 / 34.15–34.16 / 30.10–30.11 |
| Why A's official value is lower | swap test: exact positions + Trotter intensities = 28.00; Trotter positions + exact intensities = 32.45. It is the intensity error |

**Robust conclusions:**
- A tolerates the least decay at every dip level ≥ 5 %.
- The B vs C order is not robust. It flips with the dip level and with Trotter error (exact: C 23.09 < B 23.48), so it is not reported.

**J-coupling experiments:**
- **Within a molecule (artificial systems, one coupling changed, everything else fixed):**
  - A, J02 from 9.5 to 14 Hz: threshold 28.31 → 12.75.
  - B, J12 from 8.6 to 12.5 Hz: 22.85 → 8.05.
  - C, J01 from 4 to 12 Hz: 11.19 → 32.00.
  - These are reproduced exactly by the notebook and by Agent 3, and spot-checked 9/9 by Agent 4.
- **Across 159 artificial Hamiltonians:** Spearman correlation 0.989 with the critical spacing, 0.538 with J.
- **Reviewer's own experiments:**
  - Raising J02 and J12 together at a fixed difference changes the exact threshold by only −6 % / +2 % (scaling with J would predict +50 %).
  - Raising A's J01 from 0.9 to 5 Hz lowers the any-dip threshold from 27.3 to 21.2.
  - Changing only C's shift difference at fixed J changes the threshold by a factor 1.7.

Verdict: hypothesis H2 (critical spacing, topology, intensities) is supported; H1 (J strength) fails
as a general rule.

## 7. Independent review: verdicts and what was changed

| Claim | Verdict | Change made |
|---|---|---|
| C1 initial states and weights | confirmed with caveats | weights described as derived but not unique; all-plus result stated; guard made scale-aware (it missed 8.9e-16 rounding noise); new test |
| C2 pipeline physics | confirmed with caveats | Trotter intensity error (up to 19 %) added; axis range corrected to 0.42–0.93 Hz |
| C3 matching | confirmed with caveats | upstream spectrum_3 hint disclosed; overlap-only similarity flagged as the weakest metric |
| C4 thresholds | numbers confirmed, ranking weakened | B vs C ranking dropped; J01 exclusion and the 2-bin tolerance stated as necessary choices; critical splittings labelled 10 % level; B's any-dip failure stated as a position failure of the outer 8.58 Hz line |
| C5 exact limit | confirmed | — |
| C6 J relation | confirmed with caveats | π√3 × spacing presented as an upper bound with measured ratios (3.3× at 10 %, 4.4–4.8× any dip); jump at the 2.667 Hz merge distance stated; predictions.json numbers aligned (30.15, 34.15) |

## 8. Notebook execution

Final run on 2026-11-10 from a clean kernel (`.venv`, Python 3.12.13, qiskit 2.6.2):

```
jupyter nbconvert --to notebook --execute --inplace nmr.ipynb   ->  exit 0, 220 s wall time
```

- All 25 code cells ran in order (execution counts 1-25).
- No error output, no stderr stream and no `RuntimeWarning` in any cell. The upstream
  `invalid value encountered in divide` is gone because its cause, the placeholder initial
  states, is fixed. The warning was not silenced.
- Section 8.1 prints a difference of 0.0 between the step-by-step evaluation and the section-6
  Estimator, and C_xx(0) = 24 / 24 / 8.
- Every number in the markdown answer cells (8.3, 9.3) equals the corresponding printed output.
- The figures were inspected: overlays (section 8), threshold zooms (section 9.1) and J dependence
  (section 9.2).
- `pytest`: 68 passed in 3.8 s.
- Individual spectra saved as data and plots: `results/spectra/*.csv`, `results/plots/simulated_spectrum_molecule_*.png`, `results/plots/reference_spectrum_*.png` (`agents/orchestrator/save_spectra.py`).

## 8b. Matching errors and circuit-parameter study (notebook sections 8.4 and 10)

**Errors at the official setting.**
- Mean / max peak-position error against the reference: A 0.73 / 1.48 Hz, B 0.83 / 1.59 Hz, C 0.67 / 0.92 Hz.
- On the corrected axis: 0.56 / 0.93, 0.34 / 0.66 and 0.12 / 0.21 Hz. The exact-evolution floor is 0.10–0.17 Hz.
- Cosine similarity with the reference: 0.804, 0.869 and 0.843. Wrong pairings score at most 0.005.

**Circuit-parameter study** (36 settings).
- First-order Trotter with 1.25 ms steps (2× the CNOTs) brings A to a 0.14 Hz line shift, a 3.7 % height error and a 0.12 Hz mean peak error, which is the exact-evolution floor.
- Second order at 2.5 ms does not reduce the line shift (0.72 Hz).
- Fourth order costs 9.4× the CNOTs.
- Changing the order of the terms in a step changes the height error by at most 1.5 percentage points.
- A longer signal (1.5–3 s) raises the cosine to 0.95–0.97.

The step-by-step unitary evaluation was checked against the official simulation (difference ≤ 6e-13).

## 9. Limitations

- `noise_strength` is an artificial exponential window on an ideal (noise-free) simulation, not
  a hardware noise model; no hardware was used and no hardware claim is made.
- The resolution criterion is an operational choice (stated in advance, sensitivity reported).
  Absolute thresholds move by −16 % to +46 % between the 19 % and "any dip" levels.
- With 300 steps of 2.5 ms (0.75 s, 1.333 Hz grid), splittings below about 3.5–4 Hz are never
  resolved. The 0.9 Hz and 0.5 Hz couplings of A and B cannot be seen at any noise strength.
- First-order Trotter error at the official step changes line intensities by up to 19 %. That
  moves A's threshold by 1.26 s⁻¹ at the 10 % level.
- Three molecules give only a correlation. The causal statement about J rests on artificial
  one-coupling-at-a-time experiments, whose references are exact spectra with 1 Hz lines; on the
  real molecules these reproduce the supplied references' peaks within 0.08 Hz and their
  thresholds exactly.

## 10. Not run

- Real quantum hardware or any noisy backend (not required by the challenge).
- The reviewer did not execute `nmr.ipynb` itself or re-check notebook = module. Agent 2 checked
  notebook = module, and the orchestrator executed the notebook (section 8).
- 150 of the 159 rows of `results/j_coupling_sensitivity.csv` were not independently re-checked
  (9 of 9 spot-checked rows matched).
- Matching at other Trotter step sizes; uniform scaling of all couplings.
