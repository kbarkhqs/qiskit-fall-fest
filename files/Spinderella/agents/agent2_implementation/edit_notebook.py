"""Apply the Agent-2 edits to nmr.ipynb (cells 14, 24, 28 changed; one new cell after 14).

Run once from the repository root on the upstream notebook (git HEAD):
    .venv/bin/python agents/agent2_implementation/edit_notebook.py

The script refuses to run twice (it checks that cell 14 still contains the placeholder).
It writes the before/after sources to agents/agent2_implementation/cell_changes.json.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "nmr.ipynb"
OUT = Path(__file__).with_name("cell_changes.json")


def to_lines(text: str):
    """Notebook JSON stores a source as a list of lines, each but the last ending in a newline."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]
src = lambda i: "".join(cells[i]["source"])  # noqa: E731

assert len(cells) == 32, f"expected the upstream notebook with 32 cells, found {len(cells)}"
assert "Placeholder so the notebook runs" in src(14), "cell 14 is not the upstream placeholder"
assert "def time_signal_to_spectrum" in src(24)
assert 'spectrum_3.pkl' in src(28)

changes = []

# ---------------------------------------------------------------------------------------
# Cell 14: the TODO
# ---------------------------------------------------------------------------------------
before_14 = src(14)
after_14 = '''from qiskit.circuit import ParameterVector


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


# Create the parametrized circuit.
qc_init_template, phi = xsum_eigenstate_template(number_spins)

# --- Enumerate initial states and build parameter maps ---
# 1) 'bits_list' holds ALL 2^N X-basis product states, one tuple of N bits per state:
#    bits[i] = 0 puts qubit i (= spin i of the Hamiltonian) in |+>, bits[i] = 1 in |->.
# 2) 'param_maps' binds every phi[i] when running the circuits: phi[i] = pi * bits[i].

# All 2^N bit tuples, from (0, ..., 0) = all-plus to (1, ..., 1) = all-minus.
bits_list = list(product((0, 1), repeat=number_spins))

# One parameter map per state; each map binds all N parameters (0 -> |+>, pi -> |->).
param_maps = [{phi[i]: np.pi * bits[i] for i in range(number_spins)} for bits in bits_list]'''
cells[14]["source"] = to_lines(after_14)
changes.append({"cell_index_upstream": 14, "cell_index_new": 14, "kind": "modified", "before": before_14, "after": after_14})

# ---------------------------------------------------------------------------------------
# New cell directly after cell 14: verification of the initial states
# ---------------------------------------------------------------------------------------
new_15 = '''# Check of the initial states (exact statevectors, no sampling). Every assertion must hold.
from qiskit.quantum_info import Statevector

check_dim = 2**number_spins
assert len(bits_list) == check_dim and len(param_maps) == check_dim, "need exactly 2^N states"
assert all(
    len(pm) == number_spins and set(pm) == set(phi) for pm in param_maps
), "every parameter map must bind all N parameters"

check_sum_x = SparsePauliOp.from_sparse_list(
    [("X", [i], 1.0) for i in range(number_spins)], num_qubits=number_spins
)
check_sum_x_squared = (check_sum_x @ check_sum_x).simplify()

check_vectors = []
check_eigenvalues = []
print(f"{'bits (qubit 0..N-1)':>20s} {'eigenvalue of sum_i X_i':>24s} {'eigenvalue of I_x^tot':>22s}")
for check_bits, check_map in zip(bits_list, param_maps):
    check_state = Statevector(qc_init_template.assign_parameters(check_map))

    # Product state of |+> / |->: every single-qubit <X_i> is exactly +1 or -1.
    for i in range(number_spins):
        check_x_i = check_state.expectation_value(
            SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=number_spins)
        ).real
        assert abs(check_x_i - (1 - 2 * check_bits[i])) < 1e-12, (check_bits, i, check_x_i)

    # Eigenstate of sum_i X_i: zero variance, eigenvalue N - 2 * (number of |-> qubits).
    check_mean = check_state.expectation_value(check_sum_x).real
    check_variance = check_state.expectation_value(check_sum_x_squared).real - check_mean**2
    assert abs(check_variance) < 1e-12, (check_bits, check_variance)
    assert abs(check_mean - (number_spins - 2 * sum(check_bits))) < 1e-12, (check_bits, check_mean)

    check_vectors.append(check_state.data)
    check_eigenvalues.append(round(check_mean))  # integer, verified above to 1e-12
    print(f"{str(check_bits):>20s} {check_eigenvalues[-1]:>+24d} {check_eigenvalues[-1] / 2:>+22.1f}")

# The 2^N states are orthonormal (Gram matrix = identity), so none is missing or duplicated.
check_vectors = np.array(check_vectors)
check_gram = check_vectors.conj() @ check_vectors.T
assert np.allclose(check_gram, np.eye(check_dim), atol=1e-12), "states are not orthonormal"

# All-plus has the largest eigenvalue +N, all-minus the smallest -N.
assert check_eigenvalues[bits_list.index((0,) * number_spins)] == number_spins
assert check_eigenvalues[bits_list.index((1,) * number_spins)] == -number_spins

print(
    f"OK: {check_dim} = 2^{number_spins} orthonormal X-basis product states, all eigenstates of "
    f"sum_i X_i, eigenvalues from {min(check_eigenvalues):+d} to {max(check_eigenvalues):+d} "
    f"(I_x^tot from {min(check_eigenvalues) / 2:+.1f} to {max(check_eigenvalues) / 2:+.1f})."
)'''
cells.insert(
    15,
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": to_lines(new_15)},
)
changes.append({"cell_index_upstream": None, "cell_index_new": 15, "kind": "inserted", "before": None, "after": new_15})

# ---------------------------------------------------------------------------------------
# Upstream cell 24 (now 25): guard before the division in time_signal_to_spectrum
# ---------------------------------------------------------------------------------------
before_24 = src(25)
division = "    spectrum_full /= np.max(np.abs(spectrum_full))\n"
assert before_24.count(division) == 1
guard = '''    # Guard: the normalisation below divides by the maximum. A zero or non-finite maximum
    # means the time signals are empty or broken (for example the initial states are not
    # X eigenstates), so stop with a clear message instead of returning NaN.
    spectrum_maximum = np.max(np.abs(spectrum_full))
    if not np.isfinite(spectrum_maximum) or spectrum_maximum == 0:
        raise ValueError(
            "Cannot normalise the spectrum: max(|spectrum|) is "
            f"{spectrum_maximum}. The time signals c_xx and c_yx are all zero or not finite; "
            "check the initial states (section 5) and the simulated expectation values."
        )

'''
after_24 = before_24.replace(division, guard + division)
cells[25]["source"] = to_lines(after_24)
changes.append({"cell_index_upstream": 24, "cell_index_new": 25, "kind": "modified", "before": before_24, "after": after_24})

# ---------------------------------------------------------------------------------------
# Upstream cell 28 (now 29): normalise the reference into a new array
# ---------------------------------------------------------------------------------------
before_28 = src(29)
old_line = "hqs_spectrum_tools_spectrum /= np.max(np.abs(hqs_spectrum_tools_spectrum))"
assert before_28.count(old_line) == 1
new_line = (
    "# Normalise into a new array so that the loaded data is not modified in place.\n"
    "hqs_spectrum_tools_spectrum = hqs_spectrum_tools_spectrum / np.max(np.abs(hqs_spectrum_tools_spectrum))"
)
after_28 = before_28.replace(old_line, new_line)
cells[29]["source"] = to_lines(after_28)
changes.append({"cell_index_upstream": 28, "cell_index_new": 29, "kind": "modified", "before": before_28, "after": after_28})

assert len(cells) == 33
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
OUT.write_text(json.dumps(changes, indent=2, ensure_ascii=False), encoding="utf-8")
print("edited", NB, "->", len(cells), "cells;", len(changes), "changes written to", OUT)
