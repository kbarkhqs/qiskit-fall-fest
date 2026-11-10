# Physics model — from the NMR experiment to the notebook's sum over initial states

Every equation below is either standard liquid-state NMR (Levitt, *Spin Dynamics*; HQS Spectrum
Tools documentation; Khedri et al., arXiv:2404.18903, all cited in the notebook) or is checked by a
numerical test in this repository. The test is named next to each claim.

## 1. The experiment being simulated

1. **Thermal equilibrium.** In a static field B_z along z, the proton spins of the sample are in
   the state `rho_eq = exp(-beta H) / Z`. At room temperature `beta * hbar * gamma * B ~ 1e-5`, so
   to first order `rho_eq ~ (1 + epsilon * I_z^tot) / 2^N`, with `epsilon = beta hbar gamma B`
   (the Zeeman term dominates H by 7 orders of magnitude; couplings and shift differences do not
   matter at this order). Only the small part `epsilon I_z^tot`, the net nuclear magnetization,
   gives a signal; the identity part does not.
2. **pi/2 pulse.** A radio-frequency pulse about y turns `I_z^tot` into `I_x^tot`:
   `rho(0+) = (1 + epsilon I_x^tot) / 2^N`.
3. **Free evolution** under the spin Hamiltonian (section 2): `rho(t) = U(t) rho(0+) U(t)^dagger`.
4. **Detection** of the transverse magnetization:
   `<I_alpha^tot>(t) = Tr(rho(t) I_alpha^tot) = (epsilon / 2^N) Tr(I_alpha^tot(t) I_x^tot)`,
   because `Tr(I_alpha^tot) = 0`. Up to a constant, the free-induction decay is the
   **infinite-temperature correlation function** `C_alpha x(t) = Tr(I_alpha^tot(t) I_x^tot(0))`
   that the notebook computes.
5. **Decay.** Relaxation makes the signal decay. The notebook models this with an artificial
   envelope `exp(-noise_strength * t)` applied to an otherwise ideal simulation. It is not a model
   of quantum-hardware errors.
6. **Fourier transform** of the decaying signal gives the spectrum: each transition becomes an
   absorptive Lorentzian of full width at half maximum `noise_strength / pi` Hz.

The "500 MHz" in the presentation is the proton Larmor frequency at the notebook's field:
`gamma B / 2 pi = 267 522 128 x 11.7433 / 2 pi = 499.99999 MHz`. It is a property of this
spectrometer field, not a constant. The signal comes from the nuclear spins (protons). The
electrons enter only indirectly: their shielding gives the chemical shift `delta`, and the bonds
carry the scalar coupling `J`.

## 2. Hamiltonian, units and rotating frame

Notebook form: `H = -sum_l gamma (1 + delta_l) B I_l^z + 2 pi sum_{k<l} J_kl I_k . I_l`, stored with
Pauli operators (`I = sigma / 2`) in rad/s:

`H = sum_i h_i Z_i + sum_{i<j} c_ij (X_i X_j + Y_i Y_j + Z_i Z_j)`, with
`h_i = -gamma B (1 + delta_i) / 2` and `c_ij = 2 pi J_ij / 4 = pi J_ij / 2`.

Verified for all three files (agents/agent1_physics, T3: deviations < 1.1e-10 ppm and 0 Hz):

| Molecule | Spins | delta (ppm) | J (Hz) | Coupling topology |
|---|---|---|---|---|
| A | 3 | 5.790, 5.970, 5.480 | J01 = 0.9, J02 = 11.8, J12 = 17.9 | triangle; edge 0-1 nearly absent, so in practice a chain 0 - **2** - 1 with spin 2 in the middle |
| B | 3 | 5.580, 5.900, 5.340 | J01 = 15.6, J02 = -0.5, J12 = 8.6 | triangle; edge 0-2 nearly absent, so in practice a chain 0 - **1** - 2 with spin 1 in the middle |
| C | 2 | 6.375, 6.302 | J01 = 7.92 | one edge (an AB system) |

**Rotating frame.** `sum_i Z_i` commutes with every term of H (the isotropic coupling conserves
total z-spin): measured `||[H, sum Z]|| = 0` for A, B, C (`agents/orchestrator/check_circuit_and_routing.py`).
Hence `exp(-iHt) = exp(-i h_mean t sum Z) exp(-i H' t)` exactly, with
`H' = H - h_mean sum_i Z_i`. The removed part rotates `I_x^tot` and `I_y^tot` rigidly at one
frequency, shifting every line by the same amount. The notebook restores that shift in the ppm
axis (`chem_shift` in `time_signal_to_spectrum`). This is exact, not an approximation, and it
brings the single-spin terms from about 1.6e9 rad/s down to the size of the couplings
(|h'_i| <= 461 rad/s).

**What sets the spectrum.** In the rotating frame an uncoupled spin i gives one line at
`nu_i = nu0 (delta_i - mean delta)` Hz. Couplings split it. When `|nu_i - nu_j| >> |J_ij|`
(weak coupling), each line splits into a doublet of spacing |J_ij| per partner, giving equal-height
multiplets. When the ratio is small, the lines shift and the intensities tilt toward each other
(the "roof effect"). Ratios: A 13 to 100, B 10 to 33, C **4.6** (C's outer lines are 1.54 times
weaker than its inner lines; exact AB-quartet formula matched to 1e-8 Hz).

## 3. Why a sum over X-basis product states, and with which weights

`C_alpha x(t)` is a trace, not an expectation value in one state. Insert a complete set of
eigenstates of `X_tot = sum_i X_i` (Pauli convention, as the notebook's observables use):
`X_tot |s> = m_s |s>`. Then

`Tr(A(t) X_tot) = sum_s Tr(A(t) X_tot |s><s|) = sum_s m_s <s| A(t) |s>`.

- The product states `|s> = |s_0> (x) ... (x) |s_{N-1}>` with `s_i in {+, -}` (X eigenstates) are a
  complete orthonormal set of `2^N` eigenstates of X_tot, with `m_s = N - 2 w`
  (`w` = number of qubits in |->). Verified: agents/agent1_physics T1, notebook cell 15, `tests/`.
- **The states do not contribute equally.** The derivation gives each state the weight `m_s`, its
  X_tot eigenvalue, with its sign. For N = 3 the weights are +3, +1 (x3), -1 (x3), -3. For N = 2
  they are +2, 0, 0, -2, so the two states with one |+> and one |-> contribute nothing (and could be
  skipped). Equal weights compute `Tr(A(t)) = 0` instead of the signal: measured 8.9e-16 for A and
  B, which is rounding noise. The guard in `time_signal_to_spectrum` now rejects it; an `== 0`
  test did not.
- **The weights are not unique for the normalised spectrum, though** (independent review,
  agents/agent4_reviewer; wording corrected in the final audit). Any weighting `a * m_s + b` with
  `a != 0` gives `a` times the trace: the constant `b` drops out because `Tr A(t) = 0`, and the
  factor `a` (including its sign) disappears in the division by the maximum and the final `np.abs`.
  Examples: `N - w = (m_s + N)/2` gives half the trace, and `-(N - 2w)` gives minus the trace.
  Measured in the final audit: identical normalised spectra to 3e-16. Adding a function that is
  even under flipping every spin also leaves the sum unchanged. `m_s` is the weighting that
  reproduces the trace itself; it follows from inserting the X_tot eigenbasis and is what the
  notebook uses.
- **The all-plus state alone is not enough in general.** It is exact for C (where the zero-weight
  states drop out and the +2/-2 states give equal contributions), but it is off by 4.8 % (A) and
  4.0 % (B) of the spectrum maximum.
- The notebook takes the weight from the simulation itself: `mx_0 = exp_x[:, 0]`, the measured
  `<s| X_tot |s>` at t = 0, which equals `m_s` (measured to 1.3e-15).
- Physically, the weighted sum is the high-temperature density matrix `1 + epsilon X_tot` written in
  the X basis. Each product state is one term of that mixture, and its weight is the projection of
  `X_tot` onto it. That is why the sum reproduces the high-temperature NMR signal.
- Sum rules used as tests: `C_xx(0) = sum_s m_s^2 = N 2^N` (24 for A and B, 8 for C) and
  `C_yx(0) = 0`. Both hold to 2e-14 in the official simulation.
- Normalisation: Pauli instead of spin operators, and no `1 / 2^N`, multiply the signal by a
  constant `4 * 2^N`, which the division by the maximum removes (spectra identical to 4.5e-14,
  agents/agent1_physics T2). No arbitrary constant is needed or used.

**Initial-state circuit.** `H` then `RZ(phi)` on every qubit, with phi in {0, pi}:
`RZ(phi) H |0> = (e^{-i phi/2}|0> + e^{i phi/2}|1>)/sqrt 2`, giving |+> for phi = 0 and |-> (up to a
global phase -i) for phi = pi. It is the same state as `RX(phi)` then `H` (amplitude difference 0.0,
agents/agent2_implementation exp01).

## 4. Time evolution and circuit

- First-order Lie–Trotter: `U(dt) = prod_k exp(-i c_k P_k dt)`, dt = 2.5 ms, 300 steps (0.75 s).
  The decomposed circuit has one RZ per spin and RXX, RYY, RZZ per pair, each with angle
  `2 c_k dt`; measured angle error 0. Qiskit 2.6 moves single-qubit RZ gates past two-qubit gates on
  other qubits; they commute, and the step unitary equals the file-order product to 3.3e-16.
- First-order Trotter error at this dt (agents/agent1_physics T6): line shifts at most 0.72 Hz (A),
  0.42 Hz (B), 0.012 Hz (C), all below the 1.333 Hz frequency grid, falling about 4 times per halving
  of dt. The error also **changes line intensities, by up to 19.1 % (A) and 15.7 % (B)**
  (agents/agent4_reviewer). That intensity change, not the line shift, is what lowers A's noise
  tolerance. A swap test confirms it: exact positions with Trotter intensities give the Trotter
  threshold, and Trotter positions with exact intensities give the exact one.
- **Routing.** The statevector simulator has all-to-all connectivity, so no SWAP gate is needed and
  none is added. On hardware with a linear 0-1-2 chain, the triangle couplings of A and B need one
  SWAP per Trotter step (18 to 21 CX with trivial layout). C needs none. That routing cost is a
  hardware compilation detail, not part of the ideal simulation.

## 5. Frequency axis

`fft_causal` evaluates the transform at `(2j - (L-1)) / ((2L-1) dt)`. The notebook labels the bins
`fftfreq(L, dt)`, which draws every line about 0.4 to 0.9 Hz low (offset `-0.667 Hz - nu/600`
at offset nu from the mean shift, so 0.42-0.93 Hz over the lines of A, B and C; < 0.002 ppm, under
one 1.333 Hz bin). This was measured independently four times (Agent 1 T4, Agent 2 exp05,
`agents/orchestrator/check_axis_offset.py` with max deviation 0.018 Hz from the formula, and
Agent 4). It is too small to affect the matching or the resolution thresholds, so the supplied
code is left unchanged and the analyses tolerate one bin. The code's `+ Im` is correct for NumPy's
FFT sign; the markdown's `- Im` would mirror the spectrum.
