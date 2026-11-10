"""Checks that a set of initial states is the complete X-basis product basis.

Used by the tests and by the mutation check. Every check raises AssertionError on failure.
The checks never look at how the circuit was built, only at the statevectors it produces.
"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

TOL = 1e-12


def single_x(i: int, n: int) -> SparsePauliOp:
    """Pauli X on qubit i of an n-qubit register (little-endian, as in the notebook)."""
    return SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n)


def sum_x(n: int) -> SparsePauliOp:
    return SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(n)], num_qubits=n)


def assert_complete_x_basis(qc_init_template, phi, bits_list, param_maps, n: int):
    """Raise AssertionError unless the states are exactly the 2^n X-basis product states."""
    dim = 2**n

    # exactly 2^n states and 2^n parameter maps, no duplicate bit tuple
    assert len(bits_list) == dim, f"expected {dim} bit tuples, got {len(bits_list)}"
    assert len(param_maps) == dim, f"expected {dim} parameter maps, got {len(param_maps)}"
    assert len(set(map(tuple, bits_list))) == dim, "duplicate bit tuples"
    assert all(len(bits) == n and set(bits) <= {0, 1} for bits in bits_list)

    # every map binds all n parameters
    for pm in param_maps:
        assert len(pm) == n and set(pm) == set(phi), "a parameter map does not bind every phi[i]"

    total_x = sum_x(n)
    total_x_squared = (total_x @ total_x).simplify()
    vectors = []
    eigenvalues = {}
    for bits, pm in zip(bits_list, param_maps):
        bound = qc_init_template.assign_parameters(pm)
        assert bound.num_parameters == 0, "unbound parameters left in the circuit"
        state = Statevector(bound)

        # product state of |+>/|->: each single-qubit <X_i> is exactly +1 or -1
        for i in range(n):
            x_i = state.expectation_value(single_x(i, n)).real
            assert abs(x_i - (1 - 2 * bits[i])) < TOL, f"bits {bits}: <X_{i}> = {x_i}"

        # eigenstate of sum_i X_i with eigenvalue n - 2 * sum(bits): zero variance
        mean = state.expectation_value(total_x).real
        variance = state.expectation_value(total_x_squared).real - mean**2
        assert abs(variance) < TOL, f"bits {bits}: variance {variance}"
        assert abs(mean - (n - 2 * sum(bits))) < TOL, f"bits {bits}: eigenvalue {mean}"

        vectors.append(state.data)
        eigenvalues[tuple(bits)] = mean

    # mutually orthonormal: Gram matrix is the identity
    vectors = np.array(vectors)
    gram = vectors.conj() @ vectors.T
    assert np.max(np.abs(gram - np.eye(dim))) < TOL, "states are not orthonormal"

    # all-plus gives +n, all-minus gives -n
    assert abs(eigenvalues[(0,) * n] - n) < TOL
    assert abs(eigenvalues[(1,) * n] + n) < TOL
    return eigenvalues
