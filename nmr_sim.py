"""NMR spectrum simulation with Qiskit (functions ported from nmr.ipynb).

Pipeline: load struqture Hamiltonian -> rotating frame -> SparsePauliOp ->
first-order Trotter step -> time-evolve all 2^N X-basis product states ->
C_xx(t), C_yx(t) -> windowed FFT -> spectrum in ppm.

The time evolution uses exact statevector arithmetic: the Trotter step circuit is
converted to a unitary once and applied incrementally to all 2^N initial states.
This gives the same numbers as the StatevectorEstimator used in the notebook, in
O(steps) instead of O(steps^2) gate applications.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.synthesis import LieTrotter
from struqture_py.spins import PauliHamiltonian

DATA_DIR = Path(__file__).parent / "files"
FIELD_T = 11.7433
GYROMAGNETIC_RATIO_1H_RAD_S_T = 267_522_128.0
NUMBER_OF_TROTTERSTEPS = 300
TROTTER_TIMESTEP = 0.0025


# --------------------------------------------------------------------------- #
# Hamiltonian
# --------------------------------------------------------------------------- #
def load_hamiltonian(
    molecule_name: str,
    field_T: float = FIELD_T,
    data_dir: Path = DATA_DIR,
) -> PauliHamiltonian:
    """Load the struqture Hamiltonian of a molecule from its JSON file."""
    filename = Path(data_dir) / f"{molecule_name}_B{field_T}T.json"
    with open(filename, "r") as f:
        return PauliHamiltonian.from_json(json.load(f))


def to_rotating_frame(ham: PauliHamiltonian) -> Tuple[PauliHamiltonian, float]:
    """Subtract the mean Zeeman (single-Z) term.

    Returns the shifted Hamiltonian and the mean Z parameter (``chemical_shift_hz``),
    which is needed again for the ppm axis of the spectrum.
    """
    n = ham.current_number_spins()
    mean_z = float(np.mean([ham.get(f"{i}Z") for i in range(n)]))
    correction = PauliHamiltonian()
    for i in range(n):
        correction.set(f"{i}Z", mean_z)
    return ham - correction, mean_z


def struqture_hamiltonian_to_qiskit_op(
    ham_struqture: PauliHamiltonian,
    n_qubits: int,
    reverse_qubit_order: bool = True,
) -> SparsePauliOp:
    """Convert struqture keys like '0Z', '10X11X' into a Qiskit SparsePauliOp.

    Args:
        ham_struqture: struqture Hamiltonian to transform.
        n_qubits: number of qubits in the Hamiltonian.
        reverse_qubit_order: make Qiskit's rightmost char qubit 0 (little-endian).
    """
    labels, coeffs = [], []
    token_re = re.compile(r"(\d+)([XYZ])")
    for key, val in zip(ham_struqture.keys(), ham_struqture.values()):
        s = str(key)
        pauli = ["I"] * n_qubits
        if s != "I":
            for m in token_re.finditer(s):
                idx, op = int(m.group(1)), m.group(2)
                q = (n_qubits - 1 - idx) if reverse_qubit_order else idx
                if not (0 <= q < n_qubits):
                    raise IndexError(
                        f"Site index {idx} (mapped to qubit {q}) out of range 0..{n_qubits-1}"
                    )
                pauli[q] = op
        labels.append("".join(pauli))
        coeffs.append(complex(val))
    return SparsePauliOp(labels, coeffs)


# --------------------------------------------------------------------------- #
# Circuits / states / observables
# --------------------------------------------------------------------------- #
def trotter_step_circuit(
    hamiltonian_qiskit: SparsePauliOp, timestep: float = TROTTER_TIMESTEP
) -> QuantumCircuit:
    """One first-order (Lie-Trotter) step exp(-i H dt)."""
    gate = PauliEvolutionGate(hamiltonian_qiskit, timestep, synthesis=LieTrotter())
    qc = QuantumCircuit(hamiltonian_qiskit.num_qubits)
    qc.append(gate, range(hamiltonian_qiskit.num_qubits))
    return qc.decompose(reps=1)


def xbasis_initial_states(n_spins: int) -> np.ndarray:
    """All 2^N X-basis product states as columns of a (2^N, 2^N) array.

    Bit 0 -> |+>, bit 1 -> |->. Column m is the state for ``bits_list[m]`` with
    ``bits_list = list(product((0, 1), repeat=n_spins))`` (bit i acts on qubit i).
    Equivalent to the notebook's H + Rz(pi * bit) template, up to a global phase.
    """
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    cols = []
    for bits in product((0, 1), repeat=n_spins):
        psi = np.array([1.0 + 0j])
        # Qiskit little-endian: qubit 0 is the rightmost tensor factor.
        for b in reversed(bits):
            psi = np.kron(psi, minus if b else plus)
        cols.append(psi)
    return np.stack(cols, axis=1)


def total_magnetization_ops(n_spins: int) -> Tuple[SparsePauliOp, SparsePauliOp]:
    """sum_i X_i and sum_i Y_i."""
    obs_x = SparsePauliOp.from_sparse_list(
        [("X", [i], 1.0) for i in range(n_spins)], num_qubits=n_spins
    )
    obs_y = SparsePauliOp.from_sparse_list(
        [("Y", [i], 1.0) for i in range(n_spins)], num_qubits=n_spins
    )
    return obs_x, obs_y


# --------------------------------------------------------------------------- #
# Time evolution
# --------------------------------------------------------------------------- #
@dataclass
class TimeSignal:
    """Undamped correlation functions of one molecule."""

    molecule: str
    c_xx: np.ndarray
    c_yx: np.ndarray
    times: np.ndarray
    chemical_shift_hz: float
    trotter_timestep: float
    number_of_trottersteps: int


def simulate_time_signal(
    molecule_name: str,
    number_of_trottersteps: int = NUMBER_OF_TROTTERSTEPS,
    trotter_timestep: float = TROTTER_TIMESTEP,
    field_T: float = FIELD_T,
    data_dir: Path = DATA_DIR,
) -> TimeSignal:
    """Compute C_xx(t) and C_yx(t) (no damping window) for a molecule."""
    ham = load_hamiltonian(molecule_name, field_T, data_dir)
    n = ham.current_number_spins()
    ham, chemical_shift_hz = to_rotating_frame(ham)
    ham_q = struqture_hamiltonian_to_qiskit_op(ham, n)

    u_step = Operator(trotter_step_circuit(ham_q, trotter_timestep)).data
    obs_x, obs_y = (o.to_matrix() for o in total_magnetization_ops(n))

    psi = xbasis_initial_states(n)  # (2^N, 2^N), one column per initial state
    exp_x = np.zeros((psi.shape[1], number_of_trottersteps))
    exp_y = np.zeros_like(exp_x)
    for k in range(number_of_trottersteps):
        if k > 0:
            psi = u_step @ psi
        exp_x[:, k] = np.real(np.einsum("im,im->m", psi.conj(), obs_x @ psi))
        exp_y[:, k] = np.real(np.einsum("im,im->m", psi.conj(), obs_y @ psi))

    # Tr(I_a(t) I_x(0)) = sum_n m_x,n(0) <m_n(t)| I_a |m_n(t)>
    mx_0 = exp_x[:, 0]
    c_xx = mx_0 @ exp_x
    c_yx = mx_0 @ exp_y
    times = np.arange(number_of_trottersteps) * trotter_timestep
    return TimeSignal(
        molecule_name, c_xx, c_yx, times, chemical_shift_hz,
        trotter_timestep, number_of_trottersteps,
    )


# --------------------------------------------------------------------------- #
# Spectrum
# --------------------------------------------------------------------------- #
def hz_to_ppm_conversion(field_T: float = FIELD_T) -> float:
    """Factor such that ppm = Hz * factor, at the given field."""
    reference_energy_hz = GYROMAGNETIC_RATIO_1H_RAD_S_T * field_T / (2 * np.pi)
    return 1e6 / reference_energy_hz


def fft_causal(f: np.ndarray) -> np.ndarray:
    """Fourier transform of a causal signal via its even and odd parts."""
    L = len(f)
    fe = np.zeros(2 * L - 1, dtype=f.dtype)
    fe[:L] = f
    fe[L:] = f[:-1][::-1]
    fe /= 2

    fo = np.zeros(2 * L - 1, dtype=f.dtype)
    fo[:L] = f
    fo[L:] = -f[:-1][::-1]
    fo /= 2
    fo[0] = 0  # avoid jump discontinuity at t=0

    fft_even = np.fft.fftshift(np.fft.fft(fe, norm="ortho"))
    fft_odd = np.fft.fftshift(np.fft.fft(fo, norm="ortho"))
    return (fft_even + fft_odd)[::2]


def time_signal_to_spectrum(
    c_xx: np.ndarray,
    c_yx: np.ndarray,
    trotter_timestep: float,
    number_of_trottersteps: int,
    Hz_to_ppm_conversion: float,
    chem_shift_hz: float,
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Fourier-transform (already windowed) C_xx, C_yx into a spectrum.

    Returns:
        (spectrum, ppm): spectrum normalised to max 1 and the ppm axis.
    """
    m_avg_z = fft_causal(c_xx)
    m_avg_y = fft_causal(c_yx)
    frequencies = np.fft.fftshift(np.fft.fftfreq(number_of_trottersteps, d=trotter_timestep))

    chem_shift = 2 * chem_shift_hz / (2.0 * np.pi / Hz_to_ppm_conversion) + 1e6
    ppm = frequencies * Hz_to_ppm_conversion - chem_shift

    spectrum_full = m_avg_z.real + m_avg_y.imag
    spectrum_full /= np.max(np.abs(spectrum_full))
    return np.abs(spectrum_full), ppm


def spectrum_from_signal(
    sig: TimeSignal, noise_strength: float, field_T: float = FIELD_T, pad_factor: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply exp(-noise_strength * t) to the time signal and return (spectrum, ppm).

    ``pad_factor > 1`` zero-pads the signal, which samples the same spectrum on a
    ``pad_factor`` times finer frequency grid (exact interpolation, no new information).
    """
    window = np.exp(-noise_strength * sig.times)
    n = sig.number_of_trottersteps * pad_factor
    c_xx = np.zeros(n)
    c_yx = np.zeros(n)
    c_xx[: sig.number_of_trottersteps] = sig.c_xx * window
    c_yx[: sig.number_of_trottersteps] = sig.c_yx * window
    return time_signal_to_spectrum(
        c_xx,
        c_yx,
        sig.trotter_timestep,
        n,
        Hz_to_ppm_conversion=hz_to_ppm_conversion(field_T),
        chem_shift_hz=sig.chemical_shift_hz,
    )
