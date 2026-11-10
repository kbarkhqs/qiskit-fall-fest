"""Orchestrator check of the Trotter circuit and of hardware routing (SWAP) needs.

1. Every gate angle (Qiskit 2.6 moves commuting single-qubit terms, so gates are matched by type
   and qubits)  in one decomposed Trotter step equals 2 * coefficient * dt, on the right qubits
   (site i -> qubit i), for molecules A, B, C.
2. The rotating-frame shift is exact: sum_i Z_i commutes with H, so removing the mean Zeeman term only
   moves every line by the same frequency.  Checked as ||[H, sum Z]||.
3. Routing: one Trotter step transpiled to {cx, rz, sx, x} with all-to-all connectivity (ideal
   simulator) and with a linear chain 0-1-2 (restricted hardware); SWAPs are counted.
"""
import sys
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler import CouplingMap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.nmr_pipeline import load_rotating_frame_hamiltonian, trotter_step_circuit  # noqa: E402

DT = 0.0025
for m in ("molecule A", "molecule B", "molecule C"):
    H, n, chem = load_rotating_frame_hamiltonian(m)
    step = trotter_step_circuit(H, DT)
    # expected angle per term in file order
    expected = [(lbl, 2 * float(np.real(c)) * DT) for lbl, c in zip(H.paulis.to_labels(), H.coeffs)]
    got = []
    for inst in step.data:
        qs = tuple(step.find_bit(q).index for q in inst.qubits)
        got.append((inst.operation.name, qs, float(inst.operation.params[0])))
    # Qiskit may move commuting terms; compare angles term by term after matching (qubits, gate type)
    gate_of = {"X": "rxx", "Y": "ryy", "Z": "rzz"}
    max_err = 0.0
    pool = list(got)
    for lbl, ang in expected:
        acts = tuple(sorted(k for k, ch in enumerate(reversed(lbl)) if ch != "I"))
        name = "rz" if len(acts) == 1 else gate_of[[ch for ch in lbl if ch != "I"][0]]
        k = next(i for i, g in enumerate(pool) if g[0] == name and tuple(sorted(g[1])) == acts)
        max_err = max(max_err, abs(ang - pool.pop(k)[2]))
    assert not pool
    # the circuit unitary equals the file-order product prod_k exp(-i c_k P_k dt)
    from scipy.linalg import expm
    U_file = np.eye(2**n, dtype=complex)
    for P, c in zip(H.paulis, H.coeffs):
        U_file = expm(-1j * float(np.real(c)) * DT * P.to_matrix()) @ U_file
    U_circ = Operator(step).data
    u_err = float(np.max(np.abs(U_circ - U_file)))
    sumz = SparsePauliOp.from_sparse_list([("Z", [i], 1.0) for i in range(n)], num_qubits=n)
    comm = (H @ sumz - sumz @ H).simplify()
    comm_norm = float(np.max(np.abs(comm.coeffs))) if len(comm) else 0.0
    full = transpile(step, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=7)
    row = f"{m}: {len(got)} gates, angle/qubit check max err {max_err:.1e} rad; max|U_circuit - U_file_order| = {u_err:.1e}; ||[H, sum Z]|| = {comm_norm:.1e}; all-to-all: cx={full.count_ops().get('cx', 0)}"
    if n == 3:
        line = CouplingMap([[0, 1], [1, 0], [1, 2], [2, 1]])
        routed = transpile(step, basis_gates=["cx", "rz", "sx", "x"], coupling_map=line,
                           initial_layout=[0, 1, 2], routing_method="sabre",
                           layout_method="trivial", optimization_level=1, seed_transpiler=7)
        unopt = transpile(step, coupling_map=line, initial_layout=[0, 1, 2], routing_method="sabre",
                          layout_method="trivial", optimization_level=0, seed_transpiler=7)
        row += f"; linear chain 0-1-2: swaps inserted (opt 0) = {unopt.count_ops().get('swap', 0)}, cx = {routed.count_ops().get('cx', 0)}"
    print(row)
