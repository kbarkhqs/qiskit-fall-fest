"""Importable mirror of the official ``nmr.ipynb`` pipeline (HQS NMR challenge).

The functions ``struqture_hamiltonian_to_qiskit_op``, ``xsum_eigenstate_template``,
``time_signal_to_spectrum`` and ``fft_causal`` are verbatim copies of the notebook cells
(``tests/test_pipeline.py`` compares their source text with the notebook). The other
functions wrap the remaining notebook cells without changing the arithmetic.

Units follow the notebook: Hamiltonian coefficients are Pauli coefficients in rad/s,
``chemical_shift_hz`` is the mean Pauli-Z coefficient in rad/s (the name is the notebook's),
times are in seconds, ``noise_strength`` is a decay rate in 1/s.

Usage from the repository root::

    import sys; sys.path.insert(0, "src")
    import nmr_pipeline as nmr
    ham, n, shift = nmr.load_rotating_frame_hamiltonian("molecule A")
    exp_x, exp_y = nmr.simulate_signals_fast(ham, n)
    c_xx, c_yx = nmr.correlations(exp_x, exp_y)
    spectrum, ppm = nmr.spectrum_from_signals(c_xx, c_yx, 5.0, 0.0025, 300, shift)
"""

import json
import pickle
import re
from itertools import product
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.synthesis import LieTrotter
from struqture_py.spins import PauliHamiltonian

# ----------------------------------------------------------------------------------------
# Constants (notebook defaults)
# ----------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# Magnetic field (cell 4).
FIELD_T = 11.7433

# Default number of Trottersteps and Trotter timestep (cell 12).
DEFAULT_NUMBER_OF_TROTTERSTEPS = 300
DEFAULT_TROTTER_TIMESTEP = 0.0025

# Default decay rate of the exponential window in 1/s (cell 21).
DEFAULT_NOISE_STRENGTH = 5.0

# Gyromagnetic ratio for 1H (cell 25).
GYROMAGNETIC_RATIO_1H_RAD_S_T = 267_522_128.0

MOLECULES = ("molecule A", "molecule B", "molecule C")


def reference_energy_hz(field_T: float = FIELD_T) -> float:
    """Larmor/reference frequency in Hz at the given field (cell 25)."""
    return GYROMAGNETIC_RATIO_1H_RAD_S_T * field_T / (2 * np.pi)


def hz_to_ppm_conversion(field_T: float = FIELD_T) -> float:
    """Conversion factor to go from Hz to ppm at this field: ppm = (Hz) * (1e6 / v0) (cell 25)."""
    return 1e6 / reference_energy_hz(field_T)


def _resolve_dir(data_dir) -> Path:
    """Return ``data_dir`` as given if it exists, otherwise relative to the repository root."""
    path = Path(data_dir)
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


# ----------------------------------------------------------------------------------------
# Verbatim copies of notebook functions
# ----------------------------------------------------------------------------------------
def struqture_hamiltonian_to_qiskit_op(
    ham_struqture: PauliHamiltonian,
    n_qubits: int,
    reverse_qubit_order: bool = True,
) -> SparsePauliOp:
    """
    Convert struqture PauliHamiltonian keys like '0Z', '10X11X' into Qiskit labels.

    Args:
        ham_struqture (PauliHamiltonian): struqture Hamiltonian to transform to Qiskit
        n_qubits (int): number of qubits in the Hamiltonian
        reverse_qubit_order (bool): makes Qiskit's rightmost char qubit 0 (little-endian).
    """
    labels = []
    coeffs = []
    token_re = re.compile(r"(\d+)([XYZ])")

    for key, val in zip(ham_struqture.keys(), ham_struqture.values()):
        s = str(key)  # e.g., '0Z', '0X1X', '10X11X'
        pauli = ["I"] * n_qubits
        if s != "I":
            for m in token_re.finditer(s):
                idx = int(m.group(1))  # site index (can be multi-digit)
                op = m.group(2)  # 'X', 'Y', or 'Z'
                q = (n_qubits - 1 - idx) if reverse_qubit_order else idx
                if not (0 <= q < n_qubits):
                    raise IndexError(
                        f"Site index {idx} (mapped to qubit {q}) out of range 0..{n_qubits-1}"
                    )
                pauli[q] = op
        labels.append("".join(pauli))
        coeffs.append(complex(val))

    return SparsePauliOp(labels, coeffs)


def xsum_eigenstate_template(n_qubits: int):
    """Parameterized prep for any eigenstate of sum_i X_i."""
    phi = ParameterVector("phi", n_qubits)
    qc = QuantumCircuit(n_qubits, name="XsumEigen")

    # State preparation that makes each qubit an X-eigenstate:
    #   H|0> = |+>, then RZ(phi)|+> = (e^{-i phi/2}|0> + e^{+i phi/2}|1>) / sqrt(2)
    #   phi = 0  -> |+>  (X eigenvalue +1)
    #   phi = pi -> |->  (X eigenvalue -1, up to the global phase -i)
    for q in range(n_qubits):
        qc.h(q)
        qc.rz(phi[q], q)

    return qc, phi


def time_signal_to_spectrum(
    c_xx: np.ndarray,
    c_yx: np.ndarray,
    trotter_timestep: float,
    number_of_trottersteps: int,
    Hz_to_ppm_conversion: float,
    chem_shift_hz: float,
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Calculate the full spectrum from the computed expectation values via a Fourier Transform.

    Args:
        m_avg_z (np.ndarray): The first expectation value (c_xx)
        m_avg_y (np.ndarray): The second expectation value (c_yx)
        N_max (int): Number of Trotter steps.
        trotter_timestep (float): The Trotter timestep.
        Hz_to_ppm_conversion (float): Conversion factor from frequency to ppm.
        chem_shift_hz (float): The chemical shift of the input Hamiltonian (in Hz).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - **spectrum_full (np.ndarray):** The calculated spectrum.
            - **ppm (np.ndarray):** The ppm array to plot the full spectrum against.
    """
    spectrum_full = np.zeros(c_xx.shape[0], dtype=np.float64)

    m_avg_z = fft_causal(c_xx)
    m_avg_y = fft_causal(c_yx)

    frequencies = np.fft.fftshift(np.fft.fftfreq(number_of_trottersteps, d=trotter_timestep))

    # And now we have the frequency axis in ppm:
    # we need to shift the frequencies by twice the chemical input shift and then by the reference
    # energy, i.e. the energy of the honomuclear isotope (H) times the magnetic field.
    
    # the chemical shift:
    chem_shift = 2 * chem_shift_hz / (2.0*np.pi/Hz_to_ppm_conversion) + 1e6

    # to have the frequency axis in ppm:
    ppm = (frequencies) * Hz_to_ppm_conversion - chem_shift

    spectrum_full = m_avg_z.real + m_avg_y.imag

    # Guard: the normalisation below divides by the maximum. A zero or non-finite maximum
    # means the time signals are empty or broken (for example the initial states are not
    # X eigenstates), so stop with a clear message instead of returning NaN.
    # A correct signal has C_xx(0) = N 2^N >= 2 in these Pauli units, so a signal scale below
    # 1e-8 is rounding noise (for example equal weights for all initial states), not a spectrum.
    spectrum_maximum = np.max(np.abs(spectrum_full))
    signal_scale = max(np.max(np.abs(c_xx)), np.max(np.abs(c_yx)))
    if not np.isfinite(spectrum_maximum) or not np.isfinite(signal_scale) or signal_scale < 1e-8:
        raise ValueError(
            "Cannot normalise the spectrum: max(|spectrum|) is "
            f"{spectrum_maximum} and max(|c_xx|, |c_yx|) is {signal_scale}. The time signals are "
            "zero, rounding noise or not finite; check the initial states and their weights "
            "(section 5) and the simulated expectation values."
        )

    spectrum_full /= np.max(np.abs(spectrum_full))

    return np.abs(spectrum_full), ppm


def fft_causal(f: np.ndarray) -> np.ndarray:
    """
    Performs the Fourier Transform of a causal signal by separating its even and odd parts.

    Args:
        f (np.ndarray): Input causal signal.

    Returns:
        np.ndarray: The complex Fourier Transform of the causal signal.
    """
    L = len(f)

    # Even part (symmetrize)
    fe = np.zeros(2 * L - 1, dtype=f.dtype)
    fe[:L] = f
    fe[L:] = f[:-1][::-1]
    fe /= 2

    # Odd part (antisymmetrize)
    fo = np.zeros(2 * L - 1, dtype=f.dtype)
    fo[:L] = f
    fo[L:] = -f[:-1][::-1]
    fo /= 2
    fo[0] = 0  # Avoids jump discontinuity at t=0

    # FFT with orthogonal normalization and shift
    fft_even = np.fft.fftshift(np.fft.fft(fe, norm="ortho"))
    fft_odd = np.fft.fftshift(np.fft.fft(fo, norm="ortho"))

    # Combine and downsample to original length
    combined_fft = (fft_even + fft_odd)[::2]

    return combined_fft


# ----------------------------------------------------------------------------------------
# Thin wrappers around the remaining notebook cells
# ----------------------------------------------------------------------------------------
def load_rotating_frame_hamiltonian(
    molecule_name: str,
    data_dir="files",
    field_T: float = FIELD_T,
    coupling_overrides=None,
):
    """Cells 4, 6 and 9: load a molecule and return its rotating-frame Hamiltonian.

    Args:
        molecule_name: "molecule A" (the notebook's spelling) or just "A".
        data_dir: folder that contains the Hamiltonian JSON files.
        field_T: magnetic field in tesla (part of the file name).
        coupling_overrides: optional dict ``{(i, j): J_in_Hz}``. The XX, YY and ZZ
            coefficients of that spin pair are replaced by ``pi * J / 2`` before the
            rotating-frame step. ``None`` reproduces the file. Only for artificial
            sensitivity experiments; the result is then no longer the supplied molecule.

    Returns:
        (hamiltonian_qiskit, number_spins, chemical_shift_hz). ``chemical_shift_hz`` is the
        mean single-spin Pauli-Z coefficient (rad/s) removed by the rotating frame, as a
        plain float; the name is the notebook's.
    """
    if not molecule_name.startswith("molecule "):
        molecule_name = f"molecule {molecule_name}"

    # Build full filename.
    filename = _resolve_dir(data_dir) / f"{molecule_name}_B{field_T}T.json"

    # Open JSON file.
    with open(filename, "r") as json_file:
        hamiltonian_json = json.load(json_file)

    # Create PauliHamiltonian from JSON file.
    hamiltonian_struqture = PauliHamiltonian.from_json(hamiltonian_json)

    # Get the number of spins in the Hamiltonian.
    number_spins = hamiltonian_struqture.current_number_spins()

    # Optional artificial change of J couplings (not part of the notebook).
    if coupling_overrides:
        existing_keys = {str(key) for key in hamiltonian_struqture.keys()}
        for pair, j_hz in coupling_overrides.items():
            i, j = sorted(int(site) for site in pair)
            if i == j or not (0 <= i and j < number_spins):
                raise ValueError(f"Invalid spin pair {pair} for {number_spins} spins")
            for axis in "XYZ":
                key = f"{i}{axis}{j}{axis}"
                if key not in existing_keys:
                    raise KeyError(f"Coupling term {key} is not present in {filename}")
                hamiltonian_struqture.set(key, np.pi * float(j_hz) / 2.0)

    # We are now extracting the single-qubit Z terms from the full Hamiltonian:
    z_terms = []
    for i in range(number_spins):
        z_terms.append(hamiltonian_struqture.get(f"{i}Z"))

    # Mean of Z terms.
    mean_z_parameter = np.mean(z_terms)

    # Setting the chemical shift due to rotating frame of the QC calculation:
    chemical_shift_hz = mean_z_parameter

    # We now subtract the mean value of the Z coupling from the full Hamiltonian
    correction_hamiltonian = PauliHamiltonian()
    for i in range(number_spins):
        correction_hamiltonian.set(f"{i}Z", mean_z_parameter)

    hamiltonian_struqture = hamiltonian_struqture - correction_hamiltonian

    hamiltonian_qiskit = struqture_hamiltonian_to_qiskit_op(hamiltonian_struqture, number_spins)

    # In the notebook ``chemical_shift_hz`` is a struqture ``CalculatorFloat`` (np.mean of
    # CalculatorFloat objects), which makes the notebook's ``ppm`` an object-dtype array that
    # np.min / np.interp reject. The value is the same float64 number, so it is returned here
    # as a plain float; the resulting ppm axis is identical (measured difference 0.0).
    return hamiltonian_qiskit, number_spins, float(chemical_shift_hz)


def enumerate_x_states(number_spins: int):
    """Cell 14: template plus all 2^N X-basis product states.

    Returns:
        (qc_init_template, phi, bits_list, param_maps). ``bits_list[n][i]`` is 0 for |+> and
        1 for |-> on qubit i (= spin i); ``param_maps[n]`` binds every ``phi[i]`` to
        ``pi * bits_list[n][i]``.
    """
    # Create the parametrized circuit.
    qc_init_template, phi = xsum_eigenstate_template(number_spins)

    bits_list = list(product((0, 1), repeat=number_spins))

    param_maps = [{phi[i]: np.pi * bits[i] for i in range(number_spins)} for bits in bits_list]

    return qc_init_template, phi, bits_list, param_maps


def trotter_step_circuit(hamiltonian_qiskit: SparsePauliOp, trotter_timestep: float) -> QuantumCircuit:
    """Cell 12: one first-order (Lie-Trotter) step, decomposed once."""
    # Trotter Circuit
    trotter_step_first_order = PauliEvolutionGate(
        hamiltonian_qiskit, trotter_timestep, synthesis=LieTrotter()
    )

    circuit = QuantumCircuit(hamiltonian_qiskit.num_qubits)
    circuit.append(trotter_step_first_order, range(hamiltonian_qiskit.num_qubits))
    circuit = circuit.decompose(reps=1)
    return circuit


def magnetization_observables(number_spins: int):
    """Cell 17: sum of Pauli X and sum of Pauli Y (Pauli operators, not spin-1/2 operators)."""
    obs_x = SparsePauliOp.from_sparse_list(
        [("X", [i], 1.0) for i in range(number_spins)], num_qubits=number_spins
    )
    obs_y = SparsePauliOp.from_sparse_list(
        [("Y", [i], 1.0) for i in range(number_spins)], num_qubits=number_spins
    )
    return obs_x, obs_y


def simulate_signals_estimator(
    hamiltonian_qiskit: SparsePauliOp,
    number_spins: int,
    number_of_trottersteps: int = DEFAULT_NUMBER_OF_TROTTERSTEPS,
    trotter_timestep: float = DEFAULT_TROTTER_TIMESTEP,
):
    """Cells 12, 14, 17 and 19: the notebook's StatevectorEstimator loop (official route).

    Column k of the result belongs to k Trotter steps (time ``k * trotter_timestep``).
    The cost grows quadratically with ``number_of_trottersteps`` because every step is
    simulated again from the initial state.

    Returns:
        (exp_x, exp_y), each of shape (2^N, number_of_trottersteps).
    """
    circuit = trotter_step_circuit(hamiltonian_qiskit, trotter_timestep)
    qc_init_template, phi, bits_list, param_maps = enumerate_x_states(number_spins)
    obs_x, obs_y = magnetization_observables(number_spins)

    # Prebuild the parametric circuit at each step: U^n composed after the init template.
    circuits_per_step = []
    c = QuantumCircuit(number_spins)
    c = c.compose(qc_init_template)  # still parametric in phi
    circuits_per_step.append(c)

    for _ in range(1, number_of_trottersteps):
        c = c.compose(circuit)  # append one more Trotter step
        circuits_per_step.append(c)

    number_of_initializations = len(bits_list)
    exp_x = np.zeros((number_of_initializations, number_of_trottersteps))
    exp_y = np.zeros((number_of_initializations, number_of_trottersteps))

    estimator = Estimator()

    # Batch all initial states for each step to minimize overhead
    for k, circ in enumerate(circuits_per_step):
        pubs = []
        for m in range(number_of_initializations):
            pubs.append((circ, [obs_x, obs_y], param_maps[m]))  # pass parameter_values per pub
        job = estimator.run(pubs)
        results = job.result()  # list-like; one result per pub
        for m, res in enumerate(results):
            evs = res.data.evs  # [⟨obs_x⟩, ⟨obs_y⟩]
            exp_x[m, k] = evs[0]
            exp_y[m, k] = evs[1]

    return exp_x, exp_y


def simulate_signals_fast(
    hamiltonian_qiskit: SparsePauliOp,
    number_spins: int,
    number_of_trottersteps: int = DEFAULT_NUMBER_OF_TROTTERSTEPS,
    trotter_timestep: float = DEFAULT_TROTTER_TIMESTEP,
):
    """Same physics and same Trotter circuit as ``simulate_signals_estimator``, linear cost.

    Each initial state is prepared once as a ``Statevector`` and then pushed through the
    same decomposed Trotter-step circuit one step at a time; the expectation values are read
    after every step. No shortcut is taken in the physics (no exact exponential, no
    different Trotter order), so the result agrees with the estimator loop to rounding.

    Returns:
        (exp_x, exp_y), each of shape (2^N, number_of_trottersteps).
    """
    circuit = trotter_step_circuit(hamiltonian_qiskit, trotter_timestep)
    qc_init_template, phi, bits_list, param_maps = enumerate_x_states(number_spins)
    obs_x, obs_y = magnetization_observables(number_spins)

    number_of_initializations = len(bits_list)
    exp_x = np.zeros((number_of_initializations, number_of_trottersteps))
    exp_y = np.zeros((number_of_initializations, number_of_trottersteps))

    for m in range(number_of_initializations):
        state = Statevector(qc_init_template.assign_parameters(param_maps[m]))
        for k in range(number_of_trottersteps):
            if k > 0:
                state = state.evolve(circuit)  # one more Trotter step
            exp_x[m, k] = state.expectation_value(obs_x).real
            exp_y[m, k] = state.expectation_value(obs_y).real

    return exp_x, exp_y


def correlations(exp_x: np.ndarray, exp_y: np.ndarray):
    """Cell 21 before the window: unwindowed C_xx and C_yx.

    ``C_ax[k] = sum_n m_n * <n| A(t_k) |n>`` with the weight ``m_n = exp_x[n, 0]``, the
    measured <sum X> of initial state n before any Trotter step.

    Returns:
        (c_xx_raw, c_yx_raw), each of shape (number_of_trottersteps,).
    """
    number_of_trottersteps = exp_x.shape[1]

    # Expectation values.
    c_xx = np.zeros(number_of_trottersteps)
    c_yx = np.zeros(number_of_trottersteps)

    # Perform vectorized calculations
    mx_0 = exp_x[:, 0]
    for i in range(number_of_trottersteps):
        c_xx[i] = np.sum(mx_0 * exp_x[:, i], axis=0)
        c_yx[i] = np.sum(mx_0 * exp_y[:, i], axis=0)

    return c_xx, c_yx


def spectrum_from_signals(
    c_xx_raw: np.ndarray,
    c_yx_raw: np.ndarray,
    noise_strength: float,
    trotter_timestep: float,
    number_of_trottersteps: int,
    chemical_shift_hz: float,
    field_T: float = FIELD_T,
):
    """Cells 21 (window), 25 and 26: windowed signals -> normalised spectrum on a ppm axis.

    Args:
        c_xx_raw, c_yx_raw: unwindowed correlations from ``correlations``.
        noise_strength: decay rate of the exponential window in 1/s.
        chemical_shift_hz: third return value of ``load_rotating_frame_hamiltonian``.

    Returns:
        (spectrum, ppm). The spectrum is divided by its maximum and passed through
        ``np.abs`` exactly as in the notebook.
    """
    # --- apply a single exponential window before FFT ---
    times = np.arange(number_of_trottersteps) * trotter_timestep
    window = np.exp(-noise_strength * times)
    c_xx = c_xx_raw * window
    c_yx = c_yx_raw * window

    spectrum_full, ppm = time_signal_to_spectrum(
        c_xx,
        c_yx,
        trotter_timestep,
        number_of_trottersteps,
        Hz_to_ppm_conversion=hz_to_ppm_conversion(field_T),
        chem_shift_hz=chemical_shift_hz,
    )
    return spectrum_full, ppm


def load_reference(k: int, data_dir="files"):
    """Cell 28 without the normalisation: reference spectrum ``files/spectrum_k.pkl``.

    Returns:
        (ppm, spectrum_raw) as stored in the file (non-uniform ppm grid, not normalised).
    """
    with open(_resolve_dir(data_dir) / f"spectrum_{k}.pkl", "rb") as f:
        hqs_spectrum_tools_results = pickle.load(f)

    hqs_spectrum_tools_ppm = hqs_spectrum_tools_results["frequencies"]
    hqs_spectrum_tools_spectrum = hqs_spectrum_tools_results["spectrum"]
    return hqs_spectrum_tools_ppm, hqs_spectrum_tools_spectrum
