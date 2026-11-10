# Agent 1 — physics and mathematical reasoning

T1–T6 were run by Agent 1 (dense NumPy linear algebra only, no Qiskit). Agent 1 hit a usage limit
before writing this file and T7. The orchestrator compiled this report from `evidence.json` and
wrote T7 (`t7_predictions.py`, `predictions.json`) **before any noise sweep was run**.

## T1 — X-basis initial states

- `RZ(phi) H |0> = H RX(phi) |0> = (e^{-i phi/2}|0> + e^{i phi/2}|1>)/sqrt 2` (because `H RX H = RZ`);
  measured max difference 0.0 over 401 angles. phi = 0 -> |+>, phi = pi -> -i|->.
- For phi_i = pi b_i: eigenvalue of sum_i X_i = **N - 2w** (w = Hamming weight of b), eigenvalue of
  I_x^tot = N/2 - w. The binary label is **not** the eigenvalue. Degeneracy binomial(N, w).
- Orthonormal and complete (Gram - 1 and completeness - 1 at 4e-16).
- Sum rules: C_xx(0) = sum_n m_n^2 = N 2^N (8 for N = 2, 24 for N = 3); C_yx(0) = 0.

## T2 — Pauli vs spin normalisation

Pauli observables with Pauli weights and no 1/2^N multiply both correlations by the constant 4·2^N
(16 for N = 2, 32 for N = 3). After division by the maximum the spectra agree to 4.5e-14 for A, B, C.
Only absolute intensities are lost.

## T3 — Hamiltonian conventions

Files obey `h_i = -gamma B (1 + delta_i)/2` and `c_ij = pi J_ij / 2` exactly (deviation < 1.1e-10 ppm,
0 Hz). In the rotating frame an uncoupled spin sits at `nu_i = -h'_i/pi Hz = nu0 (delta_i - mean delta)`.
The notebook's `2 * chem_shift_hz / (2 pi / Hz_to_ppm)` is the right factor (the variable is a Pauli
coefficient in rad/s, misnamed `_hz`). No missing or doubled 2 or 2 pi.

| Molecule | delta (ppm) | J (Hz) | max \|h'\| dt |
|---|---|---|---|
| A | 5.790, 5.970, 5.480 | J01 0.9, J02 11.8, J12 17.9 | 1.047 |
| B | 5.580, 5.900, 5.340 | J01 15.6, J02 -0.5, J12 8.6 | 1.152 |
| C | 6.375, 6.302 | J01 7.92 | 0.143 |

## T4 — sign and axis conventions

- H_a (code `+ Im` puts lines at the true ppm) **supported**; H_b (mirrored) rejected: a single spin
  at 5.1 ppm appears at 5.0985 ppm with `+ Im` and at the mirror image with the markdown's `- Im`.
  The markdown formula's sign belongs to the opposite FFT sign convention; the code is right.
- H_c **supported**: `fft_causal` evaluates the transform at `(2j - (L-1)) / ((2L-1) dt)`, but the
  notebook labels the bins with `fftfreq(L, dt)`. Measured label = 0.998333 f - 0.6667 Hz, so every
  line is drawn 0.42-0.93 Hz low (< 0.0019 ppm, under one 1.333 Hz bin). Independently confirmed by
  Agent 2 (`exp05`) and by the orchestrator (`agents/orchestrator/check_axis_offset.py`: max
  \|measured - H_c\| = 0.018 Hz). The official code is left unchanged; analyses tolerate one bin.

## T5 — exact lines and the reference files

Exact diagonalisation gives 12 strong lines for A and B and 4 for C (C is an AB quartet, inner/outer
intensity 1.538, analytic to 1e-8 Hz). Each reference file equals the exact sticks of exactly one
molecule broadened to a Lorentzian FWHM of 1.008 Hz (correlation 1.0000):

| | spectrum_1 | spectrum_2 | spectrum_3 |
|---|---|---|---|
| A | 0% of lines in window | corr -0.03 | **corr 1.000, max peak error 0.056 Hz** |
| B | 0% of lines in window | **corr 1.000, max peak error 0.24 Hz** | corr -0.02 |
| C | **corr 1.000, max peak error 0.0002 Hz** | 0% in window | 0% in window |

Physics cross-check of the matching: **A -> spectrum_3, B -> spectrum_2, C -> spectrum_1.**

Multiplet structure (first-order origin of each adjacent spacing):
- A, spin 0 (5.79 ppm): pairs split by |J01| = 0.94 Hz, pairs separated by 10.85 Hz (~|J02| - |J01|).
  Spin 1 (5.97 ppm): 0.94 Hz pairs separated by 16.93 Hz. Spin 2 (5.48 ppm): 4 lines, spacings
  11.79 (J02), **6.08 (J12 - J02)**, 11.79 Hz.
- B, spin 0 (5.58) and spin 2 (5.34): 0.48 Hz pairs (|J02| = 0.5) separated by 8.1 / 15.1 Hz.
  Spin 1 (5.90 ppm): 4 lines, spacings 8.58 (J12), **7.02 (J01 - J12)**, 8.58 Hz.
- C: AB quartet, spacings 7.92 (J), 29.43 Hz (sqrt(dnu^2 + J^2) - J), 7.92 Hz; roof effect.

## T6 — first-order Lie–Trotter error at dt = 0.0025 s (300 steps)

| Molecule | max \|dC\|/C_xx(0) dt, dt/2, dt/4 | max line shift (Hz) dt, dt/2, dt/4 |
|---|---|---|
| A | 0.729, 0.221, 0.056 | 0.715, 0.140, 0.034 |
| B | 0.574, 0.151, 0.039 | 0.418, 0.083, 0.020 |
| C | 0.049, 0.012, 0.003 | 0.012, 0.003, 0.0007 |

Line shifts fall about 4x per halving (second order), all below one 1.333 Hz bin. Added after the
independent review: Trotter error also changes line intensities by up to 19.1 % (A) and 15.7 % (B),
and that intensity change is what moves A's noise threshold. The large
time-domain error is accumulated phase from these sub-Hz shifts (2 pi x 0.7 Hz x 0.75 s ~ 3.3 rad),
not wrong line positions. Limitation: the NumPy emulation assumes Qiskit's LieTrotter applies the
terms in `SparsePauliOp` order (not run here, because Qiskit was excluded for independence).

## T7 — pre-registered predictions (written before any sweep)

Model: `exp(-Gamma t)` on an infinite signal gives an absorptive Lorentzian of FWHM Gamma/pi Hz.
Two equal, isolated lines D Hz apart keep a dip while Gamma < pi sqrt 3 D = 5.441 D (an upper
bound for real multiplets: unequal heights and neighbouring lines lower it; measured ratios are
about 4.4-4.8 x D for any dip and about 3.3 x D at a 10 % dip) (checked numerically:
2 maxima at 54.0, 1 at 54.8 for D = 10, prediction 54.41). With the exact positions and intensities
(continuum, no grid):

| Molecule | Sub-grid splittings (never resolvable with 300 steps) | First resolvable splitting to merge | Predicted continuum Gamma* (s^-1) |
|---|---|---|---|
| A | 0.94 Hz (merge at 4.75-5.0) | 6.08 Hz inner pair of spin 2 (5.48 ppm) | **32.50** |
| B | 0.48 Hz (merge at 2.5-2.6) | the two outer 8.58 Hz pairs of spin 1 (5.90 ppm) merge at 34.15 and 34.30; the inner 7.02 Hz pair at 38.0 | **34.15** |
| C | none | outer 7.92 Hz lines (roof effect, intensity ratio 1.54) | **30.15** |

Expected effect of the official acquisition (T = 0.75 s, 1.333 Hz grid, final `np.abs`): truncation
adds sinc ringing at low Gamma (non-monotonic pass/fail possible below about 3 s^-1); sampling on a
1.333 Hz grid can move the transition by a few s^-1 either way. `predictions.json` lists every
adjacent splitting with the equal-pair rule and every continuum merge event.

## Problems found in the notebook conventions

1. Placeholder initial states (fixed in cell 14): zero signal, 0/0 normalisation.
2. ppm axis labelled with `fftfreq(L, dt)` instead of the frequencies `fft_causal` evaluates:
   0.41-0.92 Hz shift (sub-bin). Documented, not changed.
3. Markdown `- Im` vs code `+ Im`: code is correct for NumPy's FFT sign.
4. Markdown says zeros are added for resolution; the code does no zero padding.
5. `chemical_shift_hz` holds a rad/s Pauli coefficient (name only; arithmetic correct).
