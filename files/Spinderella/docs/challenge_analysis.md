# Challenge analysis — HQS NMR spectroscopy (Qiskit Fall Fest 2026, intermediate track)

Written before any file in the repository was edited. Source: upstream commit `f27f229`
(`README.md`, `nmr.ipynb`, `files/`).

## 1. Actual challenge objectives

Taken from `README.md` and the notebook text, not from outside assumptions.

1. **Make `nmr.ipynb` run** (README: "open the `nmr.ipynb` and ensure that it runs").
2. **Complete the one coding TODO** (notebook section 5, cell 14): prepare *all* initial states,
   the 2^N X-basis product eigenstates of the total X magnetization, as a parameterized
   template plus one parameter map per state.
3. **Matching task** (section 8): simulate the spectrum of Molecule A, B and C and say which of
   the three reference spectra belongs to which molecule.
4. **Noise tolerance** (section 9): sweep `noise_strength` for each molecule, starting near
   0.1 s^-1, and record the highest value at which the distinct peaks are still resolvable when
   compared with the high-resolution reference.
5. **Open question** (section 9): is that maximum tolerable `noise_strength` related to the
   J-coupling strength?

Organizer briefing (received during the work, treated as authoritative next to the notebook):
- Matching is scored one point per correct individual match, at most 3 points; report the three
  matches and the total score.
- `noise_strength` is an artificial exponential decay added to an otherwise ideal simulation; no
  other noise model is to be invented.
- Explain why the sum over initial states reproduces the high-temperature NMR signal, and whether
  the states carry weights or signs (answered in `docs/physics_model.md`, section 3).
- Test two hypotheses on J: (A) larger relevant J widens multiplets and raises the tolerance;
  (B) the tolerance also depends on shift differences, coupling topology, overlap, degeneracies and
  intensities.
- Separate hardware routing (SWAP gates) from the ideal statevector simulation.

Real quantum hardware is not required. The notebook uses `StatevectorEstimator`.
The "advanced track" (avoiding the exponential number of initial states) is mentioned but is
not part of this notebook's task.

## 2. Computational workflow of the notebook

| Step | Cell | What it does |
|---|---|---|
| Load | 4 | `files/molecule X_B11.7433T.json` -> struqture `PauliHamiltonian` (Pauli operators, coefficients in rad/s) |
| Rotating frame | 6 | subtracts the mean single-spin Z coefficient; keeps it as `chemical_shift_hz` (despite the name it is a Pauli-Z coefficient in rad/s) |
| Convert | 8-9 | struqture keys -> Qiskit `SparsePauliOp`, site i -> label position `n-1-i` (little-endian, site i = qubit i) |
| Trotter step | 12 | `PauliEvolutionGate(H, 0.0025, LieTrotter())`, 300 steps, total time 0.75 s |
| **Initial states** | **14** | **TODO: placeholder** |
| Observables | 17 | sum of Pauli X and sum of Pauli Y (not spin-1/2 operators) |
| Estimation | 19 | for step k = 0..299 the circuit `init + k Trotter steps` is evaluated for every initial state |
| Correlations | 21 | `C_ax(t_k) = sum_n m_n <n| A(t_k) |n>` with weight `m_n = exp_x[n, 0]`; exponential window `exp(-noise_strength * t)` |
| Spectrum | 24-26 | causal FFT, `Re FT[C_xx] + Im FT[C_yx]`, ppm axis, division by the maximum, `np.abs` |
| Reference | 28, 30 | loads `files/spectrum_3.pkl`, normalizes, overlays |

## 3. The missing implementation

Cell 14 is a placeholder:

- the template applies only `RZ(phi)` to |0>, which leaves every qubit in |0>, so no state is
  an X eigenstate;
- `bits_list` holds one entry instead of 2^N;
- `param_maps` holds one map.

Consequence visible in the stored outputs of the upstream notebook: `<sum X>` at step 0 is 0,
so the weight `mx_0` is 0, both correlation functions are identically 0, and cell 26 prints
`RuntimeWarning: invalid value encountered in divide` (0/0). The warning is a symptom of the
placeholder, not something to silence.

Sections 8 and 9 contain no code for the matching or the noise sweep; they have to be added.

## 4. Available data

Hamiltonians (three files, Pauli convention `H = sum_i h_i Z_i + sum_{i<j} c_ij (XX+YY+ZZ)`).
With `h_i = -gamma B (1+delta_i)/2` and `c_ij = pi J_ij / 2`:

| Molecule | Spins | Chemical shifts (ppm) | J couplings (Hz) |
|---|---|---|---|
| A | 3 | 5.790, 5.970, 5.480 | J01 = 0.9, J02 = 11.8, J12 = 17.9 |
| B | 3 | 5.580, 5.900, 5.340 | J01 = 15.6, J02 = -0.5, J12 = 8.6 |
| C | 2 | 6.375, 6.302 | J01 = 7.92 |

Reference spectra: `files/spectrum_1.pkl`, `spectrum_2.pkl`, `spectrum_3.pkl`. Each is a dict
with `frequencies` (ppm) and `spectrum`, 2000 points, **non-uniform** ppm grid, a different ppm
window per file, not normalized. The pickles contain only NumPy arrays (checked with
`pickletools` before loading).

`files/hqs_spectrum_tools_molecule C_B11.7433T.pkl` is byte-identical to `spectrum_1.pkl`
(same MD5). This leaks one assignment through a filename. It is **not** used to decide the
matching; it is recorded only as a consistency check after the numerical matching.

## 5. Required output

- A notebook that runs top to bottom with the TODO completed.
- The mapping Molecule A/B/C -> spectrum_1/2/3, with numerical evidence.
- One maximum tolerable `noise_strength` per molecule, with the criterion used.
- An answer to the J-coupling question.

## 6. Environment

| Package | Notebook expects | Installed in `.venv` |
|---|---|---|
| Python | 3.14.6 (kernel "otocs") | 3.12.13 |
| qiskit | >=2.2,<3 | 2.6.2 |
| struqture_py | any | 2.6.0 |
| numpy | any (pickles need numpy >= 2) | 2.5.4 |
| scipy | any | 1.18.1 |
| matplotlib | any | 3.11.2 |

The kernel name "otocs" does not exist locally; the notebook is executed with the `.venv`
Python 3 kernel. `pip` is installed inside `.venv` so the `%pip install` cell runs unchanged.
Git state at the start: clean working tree on `main`.

## 7. Known ambiguities and potential numerical problems

1. **Resolution criterion is qualitative** ("distinct peaks still visually resolvable"). An
   operational criterion has to be defined and its sensitivity reported.
2. **Finite acquisition time.** 300 steps x 0.0025 s = 0.75 s, so the FFT grid spacing is
   1.333 Hz (0.00267 ppm) and the window is +-200 Hz (0.8 ppm). The smallest couplings
   (0.9 Hz in A, 0.5 Hz in B) are below the grid spacing: with default settings they cannot be
   resolved at any `noise_strength`. The reference for spectrum_3 does resolve a ~0.85 Hz
   splitting. Which peaks count as "characteristic" therefore matters.
3. **Truncation ringing at low noise.** At `noise_strength = 0.1` the window is still 0.93 at
   the last time point, so the spectrum carries sinc side lobes; the final `np.abs` turns
   negative lobes into positive bumps that a peak counter can mistake for peaks. A pass/fail
   criterion may be non-monotonic at the low end.
4. **Markdown vs code.** The text says zeros are added "to increase the resolution" but the code
   does no zero padding. The spectrum formula in the text has `- Im`, the code has
   `+ m_avg_y.imag`; this may be only the FFT sign convention and must be checked numerically.
5. **Linewidth units.** `exp(-Gamma t)` gives a Lorentzian of full width `Gamma/pi` Hz; this is
   the quantity to compare with J in Hz.
6. **Trotter angle.** `max|h'| * dt` is 1.05 (A), 1.15 (B), 0.14 (C); the notebook's own rule of
   thumb (`< 1`) is marginal for A and B. First-order Trotter error must be measured, not assumed.
7. **Normalization.** Dividing by the maximum on a 1.33 Hz grid makes relative peak heights
   depend on where each line falls between grid points. `hqs_spectrum_tools_spectrum /= ...`
   modifies the loaded array in place.
8. **Pauli vs spin operators.** Observables and weights use Pauli X/Y, so the correlation is
   4x the spin-1/2 one and carries no 1/2^N; whether this cancels in the normalized spectrum is
   to be verified.
9. **Cost.** The estimator loop rebuilds every step from the initial state, so cost grows
   quadratically with the number of Trotter steps; long acquisitions through Qiskit are expensive.
