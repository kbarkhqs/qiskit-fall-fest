# Phase 1 — circuit-depth optimisation of the NMR simulation (ideal simulation only)

Date: 2026-11-10. No hardware, no noisy simulation, no error mitigation was used in this phase.
`nmr.ipynb` is the unchanged baseline. The optimised work is in:
- `nmr_optimized.ipynb`;
- `src/circuit_opt.py` and `src/phase1_benchmark.py`;
- `tests/test_circuit_opt.py` (31 tests; full suite 99 passed);
- `results/phase1/`.

## 1. Result in one table

Deepest circuit per molecule, compiled to {CX, RZ, SX, X} on all-to-all connectivity. Accuracy is
measured against exact evolution e^{-iHt}.

| Molecule | Circuit | Depth | 2-qubit depth | CNOTs | Circuits on hardware | Peak error vs exact | Spectrum cosine vs exact | Accepted |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A | C0 baseline (official) | 13,460 | 5,382 | 5,382 | 4,784 | 1.12 Hz | 0.933 | baseline |
| A | **C6 exact compilation** | **80** | **19** | **19** | 2,392 | 0 | 1.000 | yes, high-accuracy tier |
| A | C5 shifts folded into pair blocks | 8,977 | 2,691 | 2,691 | 2,392 | 0.37 Hz | 0.994 | yes |
| B | C0 baseline | 13,460 | 5,382 | 5,382 | 4,784 | 0.92 Hz | 0.975 | baseline |
| B | **C6 exact compilation** | **79** | **19** | **19** | 2,392 | 0 | 1.000 | yes, high-accuracy tier |
| B | C5 | 9,274 | 2,691 | 2,691 | 2,392 | 0.23 Hz | 0.998 | yes |
| C | C0 baseline | 4,490 | 1,794 | 1,794 | 2,392 | 0.045 Hz | 0.9999 | baseline |
| C | **C6 exact compilation** (identical to C5 for 2 spins) | **16** | **3** | **3** | **598** | 0 | 1.000 | yes, high-accuracy tier |

Reductions with C6:
- **total depth:** −99.4 % (A, B) and −99.6 % (C);
- **CNOTs and two-qubit depth:** −99.6 % (A, B) and −99.8 % (C);
- **number of circuits:** −50 % (A, B) and −75 % (C).

On a 3-qubit line (hardware-like), C6 needs 27 CNOTs for A and B (best layout and seed), and 3
for C.

## 2. Baseline analysis

- One official Trotter step (`PauliEvolutionGate` + `LieTrotter`, then one `decompose`) consists of 3 RZ
  gates plus RXX, RYY and RZZ on each of the 3 pairs. Each two-qubit rotation compiles to 2 CNOTs,
  so a step costs 18 CNOTs (6 for C). Time point k applies k steps, so the deepest circuit
  (k = 299) has 5,382 CNOTs and depth 13,460 at transpiler level 1.
- Circuits needed: 8 starting states × 299 time points × 2 measurement bases (X and Y) = 4,784 for
  A and B, and 2,392 for C. The t = 0 values are known analytically.
- Accuracy of the baseline against exact evolution:
  - line shifts: 0.715 Hz (A), 0.418 Hz (B), 0.012 Hz (C);
  - line-height errors: 19 % (A), 16 % (B), 0.7 % (C);
  - correlation-function error: 0.73 (A), 0.68 (B), 0.05 (C), relative to C_xx(0).
- **Frequency axis.** The notebook labels its frequency bins with `fftfreq(L, dt)`, but `fft_causal`
  evaluates the transform at (2j−(L−1))/((2L−1)dt). Every label is therefore 0.42–0.93 Hz too low.
  In this phase all peak positions are measured on the corrected axis. Comparisons against exact
  evolution are unaffected because both spectra share the axis; positions against the reference
  move by that amount.

## 3. Exact structure of these Hamiltonians, and what it allows

`H = Σ h_i Z_i + Σ_{i<j} c_ij (X_iX_j + Y_iY_j + Z_iZ_j)` in the rotating frame.

1. **Rotating frame is exact.** [H, ΣZ] = 0, so subtracting the mean shift is an exact
   transformation, not an approximation.
2. **One pair = one exchange gate.** X_iX_j, Y_iY_j and Z_iZ_j commute with each other, and
   XX+YY+ZZ = 2·SWAP − 1. So e^{−iθ(XX+YY+ZZ)} is a partial SWAP: a single two-qubit gate that needs
   exactly 3 CNOTs, because all three of its KAK coefficients are non-zero (Vidal & Dawson 2004;
   Vatan & Williams 2004). The 18 → 9 CNOTs per step from regrouping are an exact rewrite (unitary
   difference 0).
3. **Swap networks cost nothing for Heisenberg couplings.** SWAP·e^{−iθ(XX+YY+ZZ)} is again an
   exchange gate (3 CNOTs). On a line, interactions can be routed by swapping spins "for free"
   (Kivlichan et al. 2018), and the measured sums ΣX and ΣY do not depend on where the spins end up.
4. **ΠZ symmetry halves the circuits.** ΠZ_i commutes with H and with every circuit used here,
   maps each X-basis state to its complement (weight m → −m) and flips the signs of ΣX and ΣY.
   Complementary states therefore contribute equally, and states with m = 0 not at all. So A and B
   need 4 of 8 states and C needs 1 of 4. Measured difference: ≤ 4e-15.
5. **Two spins are fully compressible.** Any two-qubit unitary needs at most 3 CNOTs, so for
   molecule C every time point, of any formula, can be compiled to 3 CNOTs. Transpiler level 3 does
   this automatically.

## 4. Methods evaluated

| Method | Verdict for these circuits | Evidence |
|---|---|---|
| Transpiler level 2/3 (block consolidation + KAK resynthesis) | use; exact | 5,382 → 2,691 CNOTs (A); C collapses to 3 |
| Explicit commuting groups (exchange blocks) | use; exact | same 2,691 CNOTs, total depth 7,480 (vs 7,782 at level 3) |
| Qiskit Rustiq Pauli-network synthesis (Goubault de Brugière & Martiel, 2024) | not better | 16 CNOTs per step vs 9 |
| Product-formula ordering / Strang splitting of shifts vs couplings | no accuracy gain | For two operator groups, Strang = similarity transform of the first-order step, so line positions are identical (0.7148 Hz both); intensities do not improve |
| Alternating pair order | fewer CNOTs, not accepted | 1,797 CNOTs (adjacent blocks merge), but correlation error 0.693 vs 0.681 (B) and peak error 1.1239 vs 1.1228 Hz (A) |
| Second-order Suzuki (Suzuki 1991) | not accepted for A and C | 3,591 CNOTs; line shift 0.722 Hz (A) |
| Fourth-order Suzuki | too costly | 170 CNOTs/step (~9.4×); at equal cost first order with smaller steps is as accurate (section 10 of `nmr.ipynb`) |
| **Shift terms folded into the pair blocks** (new formula C5) | use for product formulas | same 9 CNOTs/step; A peak error 1.12 → 0.37 Hz, line heights 19 % → 1 %; exact for C |
| Randomized formulas: qDRIFT (Campbell 2019), random ordering (Childs, Ostrander & Su 2019) | unsuitable | qDRIFT needs N ≈ 2(λt)²/ε gates with λ = Σ\|coefficients\| ≈ 1,000 rad/s and t = 0.75 s, ~10⁶ for ε ~ 1; random ordering multiplies circuits |
| Commutator-scaling error bounds (Childs et al. 2021) | explains | dominant error is [shift terms, couplings] ∝ (h_i − h_j)·c_ij |
| Qubitization / quantum signal processing (Low & Chuang 2017, 2019) | unsuitable here | needs ancillas and block encodings with multi-controlled gates; deeper for 2–3 qubits |
| **Exact compilation per time point** (Quantum Shannon Decomposition; Shende, Bullock & Markov 2006) | best depth | 19 CNOTs (3 qubits), 3 CNOTs (2 qubits), independent of t, exact |
| Fewer Trotter steps | not possible for A and B | see section 6 |
| State preparation | already minimal | X (if the bit is 1) + H per qubit; merged with the first layer by the transpiler |
| Measurement | 2 settings required | ΣX and ΣY do not commute qubit-wise; each needs its own basis (H, or S†H). Grouping cannot merge them |

## 5. Acceptance rule (written before the benchmark)

`results/phase1/acceptance_criteria.json`. For each molecule, every error against exact evolution
must be no larger than the baseline's:
- correlation function;
- spectrum RMSE and cosine at noise 5;
- peak positions and peak heights.

All characteristic peaks must also be detected, and the A→3, B→2, C→1 matching must be unchanged.
Rewrites claimed to be exact must reproduce the baseline unitaries to 1e-9 (measured: 0).

Accepted:
- C1, C2, C5 and C6 for all molecules;
- C4 for B only.

Rejected:
- C3 and C3b, for small accuracy losses;
- C4 for A and C.

## 6. Fewer Trotter steps (`results/phase1/step_size_study.csv`, `aliasing_limits.json`)

- **Longer sampling interval, same 0.75 s.** The complex signal aliases beyond 1/(2·f_max):
  - A: f_max = 148.7 Hz, so Δt ≤ 3.36 ms, at least 224 time points;
  - B: 159.2 Hz, Δt ≤ 3.14 ms, at least 239 time points;
  - C: 22.6 Hz, Δt ≤ 22 ms, at least 34 time points.
- **Same sampling, larger Trotter steps** (deepest circuit 200/150/100/75/50 steps):
  - Official formula: every larger step is rejected for A and B. At 3.75 ms molecule A even loses
    characteristic peaks.
  - C5 formula: A passes only at 15 ms (50 steps), and only marginally (heights 35 % vs 38 %),
    while 10 ms and 7.5 ms fail. Non-monotonic behaviour like this is not robust, so it is not
    selected.
  - For C every step size is exact with C5 (one block).

## 7. Selection

- **Lowest depth that passes the rule, for every molecule: C6, exact compilation of every time point**
  - A and B: 19 CNOTs, depth 79–80;
  - C: 3 CNOTs, depth 16;
  - with the ΠZ reduction: 4 states per time point for A and B, 1 for C.
- **Best genuine product formula: C5.** It halves the CNOTs and is about 3× more accurate than the
  baseline. It is the method to scale to larger molecules, where e^{-iHt} cannot be computed
  classically.

## 8. Limitations

- C6 needs e^{-iHt} from classical diagonalisation. That is trivial for 2–3 spins but exponential in
  general. It is a compilation shortcut that makes hardware runs feasible, not a scalable simulation
  algorithm, and it should be labelled as such in any hardware result.
- All metrics are ideal. Hardware gates (ECR/CZ instead of CX), connectivity and noise come in
  Phase 2. On a line, C6 needs 27 CNOTs (best of 6 layouts × 12 seeds).
- Peak-height errors are measured on spectra sampled at 1.333 Hz, where heights depend on where a
  line falls between grid points. Line-level heights (from eigenvectors) are reported alongside.
- The step-size study covers two formulas and six step sizes. Other formulas, such as
  higher-order formulas on the shift/coupling split, were not benchmarked at large steps.

## References

- M. Suzuki, J. Math. Phys. 32, 400 (1991): fractal (higher-order) product formulas.
- A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, S. Zhu, Phys. Rev. X 11, 011020 (2021): Trotter error
  with commutator scaling.
- E. Campbell, Phys. Rev. Lett. 123, 070503 (2019): qDRIFT.
- A. M. Childs, A. Ostrander, Y. Su, Quantum 3, 182 (2019): randomized product formulas.
- G. Vidal, C. M. Dawson, Phys. Rev. A 69, 010301 (2004); F. Vatan, C. P. Williams, Phys. Rev. A
  69, 032315 (2004): any two-qubit gate with at most 3 CNOTs.
- V. V. Shende, S. S. Bullock, I. L. Markov, IEEE Trans. CAD 25, 1000 (2006): Quantum Shannon
  Decomposition (used by Qiskit's unitary synthesis).
- I. D. Kivlichan et al., Phys. Rev. Lett. 120, 110501 (2018): swap networks.
- G. H. Low, I. L. Chuang, Phys. Rev. Lett. 118, 010501 (2017); Quantum 3, 163 (2019): QSP and
  qubitization.
- A. Khedri et al., arXiv:2404.18903 (2024): NMR simulation on NISQ devices (the notebook's
  reference).
- Qiskit 2.6 documentation: `PauliEvolutionGate`, `LieTrotter`, `SuzukiTrotter`, preset pass
  managers (`ConsolidateBlocks`, `UnitarySynthesis`), high-level synthesis plugins (Rustiq).
