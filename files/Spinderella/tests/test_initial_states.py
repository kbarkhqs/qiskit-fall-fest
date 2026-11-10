"""Initial states of the notebook (cell 14): complete X-basis product basis, plus mutations."""
from itertools import product

import numpy as np
import pytest
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import Statevector
from struqture_py.spins import PauliHamiltonian

import nmr_pipeline as nmr
from state_checks import assert_complete_x_basis

SIZES = [1, 2, 3, 4]


# ------------------------------------------------------------------------------------
# The implementation
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", SIZES)
def test_states_are_the_complete_x_basis(n):
    qc, phi, bits_list, param_maps = nmr.enumerate_x_states(n)
    eigenvalues = assert_complete_x_basis(qc, phi, bits_list, param_maps, n)
    # degeneracy of eigenvalue n - 2w is binomial(n, w); the squared weights sum to n * 2^n
    values = np.array(list(eigenvalues.values()))
    assert abs(np.sum(values**2) - n * 2**n) < 1e-10
    assert abs(np.sum(values)) < 1e-10


@pytest.mark.parametrize("n", SIZES)
def test_bits_and_parameter_values(n):
    qc, phi, bits_list, param_maps = nmr.enumerate_x_states(n)
    assert bits_list == list(product((0, 1), repeat=n))
    assert len(phi) == n and qc.num_qubits == n and qc.num_parameters == n
    for bits, pm in zip(bits_list, param_maps):
        for i in range(n):
            assert pm[phi[i]] == np.pi * bits[i]


@pytest.mark.parametrize("n", SIZES)
def test_statevector_is_the_explicit_tensor_product(n):
    """Compare with kron of |+>/|-> built by hand; qubit 0 is the rightmost factor in Qiskit."""
    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    minus = np.array([1.0, -1.0]) / np.sqrt(2)
    qc, phi, bits_list, param_maps = nmr.enumerate_x_states(n)
    for bits, pm in zip(bits_list, param_maps):
        expected = np.array([1.0])
        for i in range(n):  # qubit i becomes the i-th factor from the right
            expected = np.kron(minus if bits[i] else plus, expected)
        state = Statevector(qc.assign_parameters(pm)).data
        overlap = np.vdot(expected, state)
        assert abs(abs(overlap) - 1) < 1e-12  # equal up to a global phase
        assert np.max(np.abs(state - overlap * expected)) < 1e-12


def test_bit_i_belongs_to_hamiltonian_site_i():
    """bits[i] must control the qubit that the notebook's converter uses for struqture site i."""
    n = 3
    qc, phi, bits_list, param_maps = nmr.enumerate_x_states(n)
    for site in range(n):
        ham = PauliHamiltonian()
        ham.set(f"{site}X", 1.0)
        x_site = nmr.struqture_hamiltonian_to_qiskit_op(ham, n)
        for bits, pm in zip(bits_list, param_maps):
            value = Statevector(qc.assign_parameters(pm)).expectation_value(x_site).real
            assert abs(value - (1 - 2 * bits[site])) < 1e-12


# ------------------------------------------------------------------------------------
# Mutation checks: wrong implementations must be rejected by the same checks
# ------------------------------------------------------------------------------------
def upstream_placeholder_template(n_qubits: int):
    """The template supplied in the upstream notebook: RZ(phi) on |0> (stays |0>)."""
    phi = ParameterVector("phi", n_qubits)
    qc = QuantumCircuit(n_qubits, name="XsumEigen")
    for q in range(n_qubits):
        qc.rz(phi[q], q)
    return qc, phi


def hadamard_only_template(n_qubits: int):
    """Mutation: the parameter is declared but never used, every state is all-plus."""
    phi = ParameterVector("phi", n_qubits)
    qc = QuantumCircuit(n_qubits, name="XsumEigen")
    for q in range(n_qubits):
        qc.h(q)
        qc.rz(0.0 * phi[q], q)
    return qc, phi


def full_enumeration(phi, n, angle=np.pi):
    bits_list = list(product((0, 1), repeat=n))
    param_maps = [{phi[i]: angle * bits[i] for i in range(n)} for bits in bits_list]
    return bits_list, param_maps


@pytest.mark.parametrize("n", SIZES)
def test_mutation_upstream_placeholder_fails(n):
    """Upstream cell 14 exactly: placeholder template, one bit tuple, one parameter map."""
    qc, phi = upstream_placeholder_template(n)
    bits_list = [(0,) * n]
    param_maps = [{phi[i]: 0.0 for i in range(n)}]
    with pytest.raises(AssertionError):
        assert_complete_x_basis(qc, phi, bits_list, param_maps, n)


@pytest.mark.parametrize("n", SIZES)
def test_mutation_placeholder_template_with_full_enumeration_fails(n):
    """Enumerating 2^n maps does not help if the template leaves the qubits in |0>."""
    qc, phi = upstream_placeholder_template(n)
    bits_list, param_maps = full_enumeration(phi, n)
    with pytest.raises(AssertionError, match="<X_0>"):
        assert_complete_x_basis(qc, phi, bits_list, param_maps, n)


@pytest.mark.parametrize("n", SIZES)
def test_mutation_correct_template_single_state_fails(n):
    qc, phi = nmr.xsum_eigenstate_template(n)
    with pytest.raises(AssertionError, match="expected"):
        assert_complete_x_basis(qc, phi, [(0,) * n], [{phi[i]: 0.0 for i in range(n)}], n)


@pytest.mark.parametrize("n", SIZES)
def test_mutation_parameter_ignored_fails(n):
    qc, phi = hadamard_only_template(n)
    bits_list, param_maps = full_enumeration(phi, n)
    with pytest.raises(AssertionError):
        assert_complete_x_basis(qc, phi, bits_list, param_maps, n)


@pytest.mark.parametrize("n", SIZES)
def test_mutation_wrong_angle_fails(n):
    """phi = pi/2 * bit gives a Y eigenstate, not |->."""
    qc, phi = nmr.xsum_eigenstate_template(n)
    bits_list, param_maps = full_enumeration(phi, n, angle=np.pi / 2)
    with pytest.raises(AssertionError):
        assert_complete_x_basis(qc, phi, bits_list, param_maps, n)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_mutation_reversed_bit_order_fails(n):
    """Binding phi[i] to the bit of qubit n-1-i breaks the bit <-> qubit correspondence."""
    qc, phi = nmr.xsum_eigenstate_template(n)
    bits_list = list(product((0, 1), repeat=n))
    param_maps = [{phi[i]: np.pi * bits[n - 1 - i] for i in range(n)} for bits in bits_list]
    with pytest.raises(AssertionError):
        assert_complete_x_basis(qc, phi, bits_list, param_maps, n)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_mutation_incomplete_binding_fails(n):
    qc, phi = nmr.xsum_eigenstate_template(n)
    bits_list = list(product((0, 1), repeat=n))
    param_maps = [{phi[i]: np.pi * bits[i] for i in range(n - 1)} for bits in bits_list]
    with pytest.raises(AssertionError, match="does not bind"):
        assert_complete_x_basis(qc, phi, bits_list, param_maps, n)
