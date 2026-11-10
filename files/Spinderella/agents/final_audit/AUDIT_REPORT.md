# Final scientific audit — NMR challenge (Qiskit Fall Fest 2026)

Date 2026-11-10. **Independence:** the audit was run by the orchestrator that built the solution,
so it is not fully independent. To compensate, every key number was recomputed with new code
(`agents/final_audit/audit_lib.py`). That code builds the Hamiltonian from the JSON files with
NumPy Kronecker products, applies its own Trotter product and exact evolution (no Qiskit) and has
its own peak and dip criterion. It reuses only the supplied `fft_causal` / `time_signal_to_spectrum`,
because they define the official spectrum. `nmr.ipynb` was not modified (MD5 checked before and
after). The clean execution used a copy.

Scripts: `part1_states_matching_thresholds.py`, `part2_trotter_acquisition_j.py`,
`part3_trotter_same_grid.py`. Results: `part1_results.json`, `part2_results.json`,
`part3_results.json`, combined in `audit_results.json`.

## 1. Overall verdict: **PASS WITH CAVEATS**

## 2. Task 1, matching: independently verified

**A → spectrum_3, B → spectrum_2, C → spectrum_1.**
- All 6 one-to-one assignments were scored with a method opposite to the notebook's: the
  simulation was interpolated onto each reference's non-uniform grid, with a spacing-weighted
  cosine.
- A reference-peak fraction was used as a second metric.
- Each was run at noise 1, 2, 5 and 10 s⁻¹, on the official axis and on the axis corrected for the
  sub-bin offset.
- All 16 choices pick the same assignment. Cosine totals at noise 5: 2.416 vs 0.877 (official axis)
  and 2.673 vs 0.961 (corrected). Peak fraction: 3.0 vs 1.0.
- File names and notebook overlays were not used.

## 3. Task 2, noise thresholds: mathematically correct for the stated criterion

Reproduced exactly with new code on a **full 0.01 grid over 0.01–100 s⁻¹** (10,000 points per
molecule, with no coarse-then-refine shortcut):

| Molecule | Last pass / first fail (10 %) | Changes in 0.01–100 | Any dip | 5 % | 19 % |
|---|---|---|---|---|---|
| A | 19.97 / 19.98 | 1 (pass→fail) | 27.32 | 22.15 | 17.11 |
| B | 22.85 / 22.86 | 1 | 33.36 | 25.99 | 19.13 |
| C | 23.22 / 23.23 | 1 | 28.58 | 25.51 | 20.02 |

All values are identical to the reported ones. All three spectra pass at the lowest noise value.

**Clean execution.** A copy of `nmr.ipynb` was run from a fresh kernel: exit 0, no error or stderr
output, and the printed output of all 46 cells is byte-identical to the stored notebook.
`pytest`: 68 passed.

## 4. Criterion code inspection (item 1) and issues found

Implementations inspected:
- the notebook: `characteristic_peaks` (section 8.2 cell), `resolved` and `scan` (section 9.1 cells);
- Agent 3: `agents/agent3_spectral/common.py` (`characteristic_peaks`, `evaluate`, `passes`,
  `threshold`);
- the audit's own: `audit_lib.evaluate`.

| Question | Finding |
|---|---|
| Dip definition | Both use `1 - spec[a:b+1].min() / min(spec[a], spec[b])` = 1 − I_valley / min(I1, I2), pass if ≥ 0.10. Unequal-height test case: all three implementations give 0.557208, equal to the hand calculation. **Matches.** |
| Valley only between neighbours | Searched between the two matched maxima of adjacent characteristic peaks (ppm order); including the endpoints cannot raise the minimum. Correct. |
| Normalisation / `np.abs` | Dip is scale-invariant (×37 test: same dip). The signed spectrum is positive at every threshold (minimum 0.0025–0.0096), so `np.abs` does not change any dip there (identical to 1e-15). |
| ±2.667 Hz tolerance | Applied as \|ppm_sim − ppm_ref\| × 500 Hz/ppm ≤ 2.667. Thresholds are identical for 1.5, 2 and 3 bins; with 1 bin, A and B never pass (the axis offset and Trotter shifts already move maxima by up to about 1.9 Hz). |
| Merging within 2 bins | Justified for the official acquisition; see section 7 of this report. It only affects A's 0.94 Hz J01 doublets. |
| One maximum for two reference peaks | Rejected (`np.diff(chosen) <= 0` / shared-maximum check); unit test confirms. |
| Missing peaks, edge cases | Missing peak gives a fail with a reason (unit test). The outermost lines sit at least 40 Hz from the ±200 Hz window edge. |

**Issue 1 (low).** `scan` (notebook) and `common.threshold` (Agent 3) scan in 0.1 steps and refine
only where the result changes, so a failing window narrower than 0.1 could be missed.
Effect: none for the three molecules (the full 0.01 grid shows exactly one change each).

**Issue 2 (low, documentation; fixed).** `docs/physics_model.md`, `docs/validation_report.md` and
`results/final_results.json` said that the weightings N−w, w and −(N−2w) "give the same sum" or
"reproduce the trace". They give ½, −½ and −1 times the trace (N − w = (m + N)/2). Only the
*normalised* spectrum is identical (3e-16). No result changes; the wording was corrected because
the statement was false.

## 5. Original vs independently verified numbers

| Quantity | Original | Audit |
|---|---|---|
| Official signals C_xx, C_yx | Qiskit `StatevectorEstimator` | NumPy Trotter product: max difference 7.5e-13 |
| C_xx(0) | 24, 24, 8 | 24, 24, 8 |
| Thresholds (all 4 levels) | see table above | identical |
| Assignment | A→3, B→2, C→1 | identical (16/16 choices) |
| J experiments (15 points) | notebook section 9.2 | all 15 identical |

## 6. Trotter convergence (item 3; fixed acquisition 0.75 s, Trotter vs exact on identical sampling)

| Molecule | Steps (dt) | Max line shift (Hz) | Max intensity change (of that line) | Cosine vs exact (noise 5) | Threshold Trotter / exact (10 %) |
|---|---|---|---|---|---|
| A | 300 (2.5 ms) | 0.715 | **19.1 %** (line at −148.7 Hz) | 0.933 | 19.97 / 21.23 (**−1.26**) |
| A | 600 (1.25 ms) | 0.140 | 3.7 % | 0.997 | 20.92 / 21.46 (−0.54) |
| A | 1200 (0.625 ms) | 0.034 | 0.9 % | 0.9999 | 21.58 / 21.54 (+0.04) |
| B | 300 | 0.418 | 15.7 % | 0.975 | 22.85 / 23.48 (−0.63) |
| B | 600 | 0.083 | 2.9 % | 0.999 | 23.57 / 23.48 (+0.09) |
| B | 1200 | 0.020 | 0.7 % | 0.99995 | 23.48 / 23.44 (+0.04) |
| C | 300 | 0.012 | 0.7 % | 0.9999 | 23.22 / 23.09 (+0.13) |
| C | 1200 | 0.0007 | 0.05 % | 1.0 | 23.08 / 23.08 (0.00) |

- Line shifts and intensity errors fall about 4–5 times per halving of dt (second order).
- **Verdict.** The official 300-step result is accurate enough for the matching: lines move by less
  than one 1.333 Hz bin and the assignment margin is large. For the thresholds it is biased low by
  6 % for A and 3 % for B, through line intensities.
- The challenge specifies 300 steps of 2.5 ms, so the official numbers stay as reported. The
  converged values (A ≈ 21.5, B ≈ 23.45, C ≈ 23.08) should be quoted alongside them.
- Under converged dynamics the 10 % order of B and C reverses (C < B). This confirms that the two
  must not be ranked.

## 7. Frequency resolution and the 0.9 Hz J01 splitting (item 4; supplementary experiments)

A's J01 doublets (exact spacing 0.941 Hz; 1.007 Hz under 300-step Trotter, which widens them by 7 %):

| Acquisition (dt 2.5 ms) | Grid spacing | J01 doublets resolved? (exact / Trotter) | Official criterion, 10 % (doublets merged), Trotter / exact |
|---|---|---|---|
| **0.75 s (official)** | 1.333 Hz | **never, not even at Γ = 0** | 19.97 / 21.23 |
| 1.5 s | 0.667 Hz | never, not even at Γ = 0 | 20.05 / 21.60 |
| 3.0 s | 0.333 Hz | up to Γ = 3.86 / 4.17 (any dip), 3.03 / 3.27 (10 %) | 20.10 / 21.63 |

- The splitting is below the Fourier limit of the official acquisition (1/T = 1.33 Hz > 0.94 Hz):
  no decay rate can resolve it.
- Excluding it from the criterion is therefore **justified for the official settings**. Requiring
  it would make A fail at every noise strength, which makes the question meaningless for A.
- It must stay disclosed. With a 3 s acquisition the J01 doublets would limit A to about
  3 s⁻¹ (10 %), so the exclusion is decisive.
- Longer acquisitions change the merged-criterion thresholds by at most 0.4 s⁻¹.

## 8. J-coupling conclusion (item 7)

| Category | Content |
|---|---|
| Observation (n = 3) | A has the largest couplings (mean \|J\| 10.2 Hz) and the lowest threshold; correlation only |
| Critical-splitting equations | First order: A spin 2 inner pair = J12 − J02 (6.10), B spin 1 inner pair = J01 − J12 (7.00), C = J01 (7.92). From exact lines: 6.08, 7.02, 7.92. Every one of the 15 controlled points matches its formula within 0.03 Hz. **Correct.** |
| Controlled numerical evidence | A: J02 9.5→14 Hz, threshold 28.31→12.75. B: J12 8.6→12.5 Hz, 22.85→8.05. C: J01 4→12 Hz, 11.19→32.00. All 15 reproduced exactly with independent code, each with a single pass→fail change. Raising J lowers the tolerance whenever it shrinks the critical spacing. |
| Supported conclusion | "Noise tolerance depends more strongly on the smallest relevant spectral separation and peak intensities than on the absolute magnitude of J" is supported **for these molecules and this criterion**. It is not a universal law; it rests on single-coupling variations of three spin systems. |

## 9. Corrections

1. *(done)* Weighting wording (Issue 2).
2. *(recommended, text only)* Quote the converged thresholds (A ≈ 21.5, B ≈ 23.45, C ≈ 23.08)
   next to the official 300-step values. Added to `results/final_results.json`.
3. *(optional)* Use a full fine grid, or a bracket check, in `scan` / `common.threshold` (Issue 1).
   Not needed for the reported results.

No correction changes the matching or any official threshold.

## 10. Recommendation: **READY FOR SUBMISSION**

Submit the official 300-step results and keep the caveats already stated in the notebook (9.3)
and README:
- the dip level is an operational choice;
- B and C are not ranked;
- A's 0.9 Hz splitting is excluded because it lies below the acquisition's Fourier limit;
- Trotter error lowers A's value by about 6 %.
