"""Phase 1: depth-optimised circuits for the NMR time evolution (ideal simulation only).

Two layers:
  * NumPy models of every evolution formula (used for accuracy; independent of Qiskit);
  * Qiskit circuit builders for the same formulas (used for depth / gate metrics).
The two are cross-checked (``check_model_vs_circuit``).

Notation: rotating-frame Hamiltonian H = sum_i h_i Z_i + sum_{i<j} c_ij (X_i X_j + Y_i Y_j + Z_i Z_j),
qubit i = spin i (Qiskit little-endian). Circuit order = time order (first gate applied first).
"""
from __future__ import annotations

import math
from functools import lru_cache
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler import CouplingMap
from scipy.linalg import expm

import nmr_pipeline as P

BASIS = ["cx", "rz", "sx", "x"]
LINE = CouplingMap([[0, 1], [1, 0], [1, 2], [2, 1]])
DT = P.DEFAULT_TROTTER_TIMESTEP  # 2.5 ms sampling interval
L = P.DEFAULT_NUMBER_OF_TROTTERSTEPS  # 300 time points
PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]], complex),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0 + 0j, -1.0])}
SWAP4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex)


# ------------------------------------------------------------------ model of the molecule
@lru_cache(maxsize=None)
def spin_model(name: str):
    ham_q, n, chem = P.load_rotating_frame_hamiltonian(name)
    labels = ham_q.paulis.to_labels()
    coeffs = np.real(ham_q.coeffs)
    h = np.zeros(n)
    c = {}
    for lab, v in zip(labels, coeffs):
        sites = [n - 1 - k for k, ch in enumerate(lab) if ch != "I"]
        kinds = {ch for ch in lab if ch != "I"}
        if len(sites) == 1:
            h[sites[0]] = v
        elif kinds == {"X"}:
            c[tuple(sorted(sites))] = v
    return {"name": name, "ham_q": ham_q, "n": n, "chem": chem, "h": h, "c": c, "H": ham_q.to_matrix()}


def pauli_matrix(ops: dict, n: int):
    m = np.array([[1.0 + 0j]])
    for q in reversed(range(n)):
        m = np.kron(m, PAULI[ops.get(q, "I")])
    return m


def embed2(u4, i, j, n):
    """4x4 unitary on (qubit i, qubit j) with i as the low bit of the 2-qubit index, embedded in n qubits."""
    dim = 2**n
    out = np.zeros((dim, dim), complex)
    for col in range(dim):
        bi, bj = (col >> i) & 1, (col >> j) & 1
        sub_col = bi | (bj << 1)
        for sub_row in range(4):
            amp = u4[sub_row, sub_col]
            if amp == 0:
                continue
            row = col & ~(1 << i) & ~(1 << j)
            row |= (sub_row & 1) << i
            row |= ((sub_row >> 1) & 1) << j
            out[row, col] += amp
    return out


def exchange4(coeff, t):
    """exp(-i coeff t (XX+YY+ZZ)) as a 4x4 matrix."""
    heis = np.kron(PAULI["X"], PAULI["X"]) + np.kron(PAULI["Y"], PAULI["Y"]) + np.kron(PAULI["Z"], PAULI["Z"])
    return expm(-1j * coeff * t * heis)


def z_layer(model, t):
    n = model["n"]
    diag = np.ones(2**n, complex)
    for idx in range(2**n):
        e = sum(model["h"][q] * (1 - 2 * ((idx >> q) & 1)) for q in range(n))
        diag[idx] = np.exp(-1j * e * t)
    return np.diag(diag)


# ------------------------------------------------------------------ formulas (layer lists)
def pairs_of(model):
    return sorted(model["c"])


def formula_layers(formula, model, step_index):
    """Layers of one step, in circuit (time) order. ('Z', frac) or ('P', pair, frac)."""
    pairs = pairs_of(model)
    if formula == "lie":  # official: Z layer, then pairs in file order
        return [("Z", 1.0)] + [("P", p, 1.0) for p in pairs]
    if formula == "strang":  # pairs, then full Z; boundary half-Z handled by pre/post
        return [("P", p, 1.0) for p in pairs] + [("Z", 1.0)]
    if formula == "strang_alt":  # as strang, pair order reversed on every second step
        order = pairs if step_index % 2 == 0 else pairs[::-1]
        return [("P", p, 1.0) for p in order] + [("Z", 1.0)]
    if formula in ("pairfield", "pairfield_alt"):  # each spin's Z field shared among its pairs, inside the blocks
        order = pairs if (formula == "pairfield" or step_index % 2 == 0) else pairs[::-1]
        return [("F", p, 1.0) for p in order]
    if formula == "suzuki2":  # full symmetric second order over all terms
        half = [("Z", 0.5)] + [("P", p, 0.5) for p in pairs[:-1]]
        return half + [("P", pairs[-1], 1.0)] + half[::-1]
    raise ValueError(formula)


def formula_boundary(formula):
    """(pre, post) half layers applied once at the start / end of a time point with k >= 1 steps."""
    if formula in ("strang", "strang_alt"):
        return [("Z", 0.5)], [("Z", -0.5)]
    return [], []


def field_block4(model, pair, t):
    """exp(-i t (c (XX+YY+ZZ) + w_i h_i Z_i + w_j h_j Z_j)), w = 1 / (number of pairs containing the spin)."""
    i, j = pair
    share = lambda q: 1.0 / sum(q in pq for pq in model["c"])
    heis = np.kron(PAULI["X"], PAULI["X"]) + np.kron(PAULI["Y"], PAULI["Y"]) + np.kron(PAULI["Z"], PAULI["Z"])
    zi, zj = np.kron(np.eye(2), PAULI["Z"]), np.kron(PAULI["Z"], np.eye(2))  # low bit = spin i
    gen = model["c"][pair] * heis + share(i) * model["h"][i] * zi + share(j) * model["h"][j] * zj
    return expm(-1j * t * gen)


def layers_unitary(model, layers, dt):
    n = model["n"]
    u = np.eye(2**n, dtype=complex)
    for lay in layers:
        if lay[0] == "Z":
            g = z_layer(model, lay[1] * dt)
        elif lay[0] == "F":
            (i, j), frac = lay[1], lay[2]
            g = embed2(field_block4(model, (i, j), frac * dt), i, j, n)
        else:
            (i, j), frac = lay[1], lay[2]
            g = embed2(exchange4(model["c"][(i, j)], frac * dt), i, j, n)
        u = g @ u
    return u


def time_point_unitaries(formula, model, dt=DT, steps=L, step_size=None):
    """Unitary of every time point t_k = k dt (k = 0..steps-1).

    step_size=None: k Trotter steps of size dt (official scheme).
    step_size=D:    n_k = ceil(t_k / D) steps of size t_k / n_k (fewer steps, same sampling).
    formula='exact': exp(-i H t_k).
    """
    n = model["n"]
    out = [np.eye(2**n, dtype=complex)]
    if formula == "exact":
        e, v = np.linalg.eigh(model["H"])
        return [v @ np.diag(np.exp(-1j * e * k * dt)) @ v.conj().T for k in range(steps)]
    pre, post = formula_boundary(formula)
    if step_size is None:
        u_pre, u_post = layers_unitary(model, pre, dt), layers_unitary(model, post, dt)
        steps_u = [layers_unitary(model, formula_layers(formula, model, s), dt) for s in (0, 1)]
        acc = np.eye(2**n, dtype=complex)
        for k in range(1, steps):
            acc = steps_u[(k - 1) % 2] @ acc
            out.append(u_post @ acc @ u_pre)
        return out
    for k in range(1, steps):
        t = k * dt
        nk = max(1, math.ceil(t / step_size - 1e-9))
        d = t / nk
        acc = layers_unitary(model, pre, d)
        su = [layers_unitary(model, formula_layers(formula, model, s), d) for s in (0, 1)]
        for s in range(nk):
            acc = su[s % 2] @ acc
        out.append(layers_unitary(model, post, d) @ acc)
    return out


def steps_needed(k, dt=DT, step_size=None):
    if k == 0:
        return 0
    return k if step_size is None else max(1, math.ceil(k * dt / step_size - 1e-9))


# ------------------------------------------------------------------ signals and symmetry
def x_states(n, half=False):
    """X-basis product states and weights m = N - 2w. half=True keeps one state of every
    complement pair (s, not s) with doubled weight (exact by the prod-Z symmetry) and drops m = 0."""
    plus, minus = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
    bits = list(product((0, 1), repeat=n))
    keep, weights = [], []
    for b in bits:
        m = n - 2 * sum(b)
        if half:
            if m <= 0:
                continue
            m *= 2
        keep.append(b)
        weights.append(m)
    vecs = []
    for b in keep:
        v = np.array([1.0 + 0j])
        for q in reversed(range(n)):
            v = np.kron(v, minus if b[q] else plus)
        vecs.append(v)
    return keep, np.array(weights, float), np.array(vecs).T


def signals(unitaries, model, half=False):
    n = model["n"]
    X = sum(pauli_matrix({i: "X"}, n) for i in range(n))
    Y = sum(pauli_matrix({i: "Y"}, n) for i in range(n))
    _, w, psi0 = x_states(n, half)
    cxx, cyx = np.zeros(len(unitaries)), np.zeros(len(unitaries))
    for k, u in enumerate(unitaries):
        psi = u @ psi0
        cxx[k] = w @ np.real(np.einsum("ij,ij->j", psi.conj(), X @ psi))
        cyx[k] = w @ np.real(np.einsum("ij,ij->j", psi.conj(), Y @ psi))
    return cxx, cyx


# ------------------------------------------------------------------ Qiskit circuits
def _exchange_gate(coeff, t, with_swap=False, label=None):
    u = exchange4(coeff, t)
    if with_swap:
        u = SWAP4 @ u
    return UnitaryGate(u, label=label)


def append_layers(qc, model, layers, dt, style="grouped", positions=None):
    """style: 'rotations' = RXX/RYY/RZZ (as PauliEvolutionGate+LieTrotter gives),
    'grouped' = one exact 2-qubit block per pair. positions[i] = physical wire of spin i."""
    pos = positions or list(range(model["n"]))
    for lay in layers:
        if lay[0] == "Z":
            for q in range(model["n"]):
                qc.rz(2 * model["h"][q] * lay[1] * dt, pos[q])
        elif lay[0] == "F":
            (i, j), frac = lay[1], lay[2]
            qc.append(UnitaryGate(field_block4(model, (i, j), frac * dt)), [pos[i], pos[j]])
        else:
            (i, j), frac = lay[1], lay[2]
            ang = 2 * model["c"][(i, j)] * frac * dt
            if style == "rotations":
                qc.rxx(ang, pos[i], pos[j]); qc.ryy(ang, pos[i], pos[j]); qc.rzz(ang, pos[i], pos[j])
            else:
                qc.append(_exchange_gate(model["c"][(i, j)], frac * dt), [pos[i], pos[j]])


def evolution_circuit(candidate, model, k, step_size=None):
    """Evolution part (no state preparation, no measurement) for time point k."""
    n = model["n"]
    qc = QuantumCircuit(n)
    if k == 0:
        return qc
    if candidate["formula"] == "exact":
        e, v = np.linalg.eigh(model["H"])
        qc.append(UnitaryGate(v @ np.diag(np.exp(-1j * e * k * DT)) @ v.conj().T, label="exp(-iHt)"), range(n))
        return qc
    nk = steps_needed(k, DT, step_size)
    d = (k * DT) / nk
    formula, style = candidate["formula"], candidate.get("style", "grouped")
    pre, post = formula_boundary(formula)
    if candidate.get("swap_network"):
        # 3 spins on a line: exchange+SWAP blocks on wires (0,1),(1,2),(0,1); spins move, sums are invariant.
        pos = list(range(n))  # pos[spin] = wire
        append_layers(qc, model, pre, d, positions=pos)
        for s in range(nk):
            for lay in formula_layers(formula, model, s):
                if lay[0] == "Z":
                    append_layers(qc, model, [lay], d, positions=pos)
                    continue
                (i, j), frac = lay[1], lay[2]
                wi, wj = pos[i], pos[j]
                assert abs(wi - wj) == 1, "swap network needs adjacent spins"
                lo, hi = (i, j) if wi < wj else (j, i)
                qc.append(_exchange_gate(model["c"][(i, j)], frac * d, with_swap=True), [pos[lo], pos[hi]])
                pos[i], pos[j] = pos[j], pos[i]
        append_layers(qc, model, post, d, positions=pos)
        return qc
    append_layers(qc, model, pre, d, style)
    for s in range(nk):
        append_layers(qc, model, formula_layers(formula, model, s), d, style)
    append_layers(qc, model, post, d, style)
    return qc


def full_circuit(evolution, bits, basis):
    """State preparation (X then H per qubit) + evolution + basis change ('X' or 'Y') + measurement."""
    n = evolution.num_qubits
    qc = QuantumCircuit(n, n)
    for q in range(n):
        if bits[q]:
            qc.x(q)
        qc.h(q)
    qc.compose(evolution, range(n), inplace=True)
    for q in range(n):
        if basis == "Y":
            qc.sdg(q)
        qc.h(q)
    qc.measure(range(n), range(n))
    return qc


def circuit_metrics(qc):
    twoq = lambda inst: inst.operation.num_qubits == 2 and inst.operation.name != "barrier"
    ops = qc.count_ops()
    oneq = sum(v for k, v in ops.items() if k in ("rz", "sx", "x", "h", "sdg", "s", "u", "u3"))
    return {"depth": qc.depth(lambda i: i.operation.name not in ("barrier", "measure")),
            "twoq_depth": qc.depth(twoq), "cx": ops.get("cx", 0), "swap": ops.get("swap", 0),
            "oneq": oneq, "total": sum(v for k, v in ops.items() if k not in ("barrier", "measure"))}


def transpile_for(qc, level, coupling=None, layout=None, keep_swaps=False, seed=11):
    basis = BASIS + (["swap"] if keep_swaps else [])
    return transpile(qc, basis_gates=basis, optimization_level=level, coupling_map=coupling,
                     initial_layout=layout, seed_transpiler=seed)


def check_model_vs_circuit(candidate, model, ks=(1, 2, 7, 30), step_size=None):
    """Max |U_model - U_circuit| (global phase removed) over the given time points; for the swap network the
    comparison is made on the measured quantities instead (wires are permuted)."""
    units = time_point_unitaries(candidate["formula"], model, step_size=step_size)
    worst = 0.0
    n = model["n"]
    for k in ks:
        uc = Operator(evolution_circuit(candidate, model, k, step_size)).data
        if candidate.get("swap_network"):
            X = sum(pauli_matrix({i: "X"}, n) for i in range(n))
            _, _, psi0 = x_states(n)
            a = np.real(np.einsum("ij,ij->j", (uc @ psi0).conj(), X @ (uc @ psi0)))
            b = np.real(np.einsum("ij,ij->j", (units[k] @ psi0).conj(), X @ (units[k] @ psi0)))
            worst = max(worst, float(np.max(np.abs(a - b))))
            continue
        um = units[k]
        idx = np.unravel_index(np.argmax(np.abs(um)), um.shape)
        phase = uc[idx] / um[idx]
        worst = max(worst, float(np.max(np.abs(uc - phase * um))))
    return worst
