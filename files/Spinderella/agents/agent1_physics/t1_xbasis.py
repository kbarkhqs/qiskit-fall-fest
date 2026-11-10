"""T1: X-basis initial states. Explicit 2x2 matrices and Kronecker products only."""
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import X, Y, Z, I2, kron_all, total, x_basis_states, record, cmd

Hd = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
rz = lambda p: np.array([[np.exp(-1j * p / 2), 0], [0, np.exp(1j * p / 2)]])
rx = lambda p: np.array([[np.cos(p / 2), -1j * np.sin(p / 2)], [-1j * np.sin(p / 2), np.cos(p / 2)]])
ket0 = np.array([1, 0], dtype=complex)
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

# (a) the two constructions for general phi
phis = np.linspace(-2 * np.pi, 2 * np.pi, 401)
d_state = max(np.abs(rz(p) @ Hd @ ket0 - Hd @ rx(p) @ ket0).max() for p in phis)
d_formula = max(np.abs(rz(p) @ Hd @ ket0 - np.array([np.exp(-1j * p / 2), np.exp(1j * p / 2)]) / np.sqrt(2)).max() for p in phis)
bloch_err = 0.0
for p in phis:
    s = rz(p) @ Hd @ ket0
    b = [np.real(s.conj() @ P @ s) for P in (X, Y, Z)]
    bloch_err = max(bloch_err, np.abs(np.array(b) - [np.cos(p), np.sin(p), 0]).max())
s0, spi = rz(0) @ Hd @ ket0, rz(np.pi) @ Hd @ ket0
a = {
    "max|RZ(phi)H|0> - H RX(phi)|0>| over 401 phi in [-2pi,2pi]": d_state,
    "max|RZ(phi)H|0> - (e^-iphi/2,e^+iphi/2)/sqrt2|": d_formula,
    "max Bloch-vector error vs (cos phi, sin phi, 0)": bloch_err,
    "|state(phi=0) - |+>|": np.abs(s0 - plus).max(),
    "|state(phi=pi) - (-i)|->|": np.abs(spi - (-1j) * minus).max(),
    "<X> at phi=0, pi, pi/2": [np.real(s.conj() @ X @ s) for s in (s0, spi, rz(np.pi / 2) @ Hd @ ket0)],
    # the notebook placeholder: RZ only on |0>
    "placeholder RZ(phi)|0>: max |<X>| over phi": max(abs(np.real((rz(p) @ ket0).conj() @ X @ (rz(p) @ ket0))) for p in phis),
}
print("(a)", a)

res = {"a": a}
for n in (2, 3):
    psi = x_basis_states(n)  # column idx = sum_i b_i 2^i, phi_i = pi b_i
    xt, yt = total(X, n), total(Y, n)
    idx = np.arange(2**n)
    w = np.array([bin(i).count("1") for i in idx])
    # (b) eigenvalue relation
    eig_resid = np.abs(xt @ psi - psi * (n - 2 * w)).max()
    m_pauli = np.real(np.einsum("in,ij,jn->n", psi.conj(), xt, psi))
    label_equals_eig = bool(np.allclose(m_pauli, idx))
    # (c) orthonormality, completeness, degeneracies
    gram = np.abs(psi.conj().T @ psi - np.eye(2**n)).max()
    compl = np.abs(psi @ psi.conj().T - np.eye(2**n)).max()
    deg = {int(n - 2 * k): int(np.sum(w == k)) for k in range(n + 1)}
    deg_ok = all(deg[n - 2 * k] == comb(n, k) for k in range(n + 1))
    spec = np.sort(np.linalg.eigvalsh(xt))
    spec_ok = bool(np.allclose(spec, np.sort(n - 2 * w)))
    # (d) sum rules
    sum_m2 = float(np.sum(m_pauli**2))
    cxx0_trace = float(np.real(np.trace(xt @ xt)))
    cyx0_trace = float(np.abs(np.trace(yt @ xt)))
    y_each = float(np.abs(np.einsum("in,ij,jn->n", psi.conj(), yt, psi)).max())
    # what happens if the binary label is (wrongly) used as weight
    wrong = float(np.sum(idx * m_pauli))
    res[f"N={n}"] = {
        "eigen-residual |Xtot psi_b - (N-2w) psi_b|": eig_resid,
        "eigenvalues sum X (by index 0..2^N-1)": m_pauli.round(12).tolist(),
        "eigenvalues I_x^tot = N/2 - w": (m_pauli / 2).round(12).tolist(),
        "binary label equals eigenvalue": label_equals_eig,
        "gram - 1": gram,
        "completeness - 1": compl,
        "degeneracy {eigenvalue of sum X: count}": deg,
        "degeneracy equals binomial(N,w)": deg_ok,
        "spectrum of sum X matches {N-2w}": spec_ok,
        "sum_n m_n^2 (Pauli)": sum_m2,
        "N 2^N": n * 2**n,
        "Tr(Xtot Xtot)": cxx0_trace,
        "sum_n (m_n/2)^2 (spin-1/2)": sum_m2 / 4,
        "N 2^(N-2)": n * 2 ** (n - 2),
        "|Tr(Ytot Xtot)|": cyx0_trace,
        "max_n |<psi_n|Ytot|psi_n>|": y_each,
        "sum_n label_n * m_n (if label were used as weight)": wrong,
    }
    print(f"N={n}", res[f"N={n}"])

ok = (d_state < 1e-12 and all(r["eigen-residual |Xtot psi_b - (N-2w) psi_b|"] < 1e-12 and r["gram - 1"] < 1e-12
      and r["completeness - 1"] < 1e-12 and r["degeneracy equals binomial(N,w)"] and abs(r["sum_n m_n^2 (Pauli)"] - r["N 2^N"]) < 1e-10
      and r["|Tr(Ytot Xtot)|"] < 1e-12 and not r["binary label equals eigenvalue"] for r in (res["N=2"], res["N=3"])))
record({
    "id": "T1_xbasis_states",
    "question": "Which states do 'H then RZ(phi)' and 'RX(phi) then H' prepare, and what are the eigenvalues, degeneracies and sum rules of the X-basis product states?",
    "hypotheses": ["H1: both constructions give the same state vector (e^-iphi/2|0> + e^+iphi/2|1>)/sqrt2 for every phi",
                   "H2: for phi_i = pi b_i the eigenvalue of sum X is N - 2w (w = Hamming weight), not the binary label",
                   "H3: the 2^N states are orthonormal and complete, degeneracy binomial(N,w), sum m^2 = N 2^N, C_yx(0) = 0"],
    "independent_variable": "phi (401 values), bit string b, N in {2,3}",
    "dependent_variable": "state-vector differences, eigen-residuals, Gram/completeness errors, sums",
    "fixed_parameters": {"qubit_order": "site 0 = rightmost Kronecker factor (index = sum b_i 2^i)"},
    "reference": "analytic derivation in report.md section T1",
    "result": res,
    "conclusion": ("All three hypotheses confirmed to machine precision. The two constructions are the identical vector "
                   "(H RX(phi) H = RZ(phi)); phi=0 -> |+>, phi=pi -> -i|->; eigenvalue of sum X is N-2w; "
                   "C_xx(0) = N 2^N = 8 (N=2), 24 (N=3); C_yx(0) = 0.") if ok else "CHECK FAILED - see result",
    "limitations": "N = 2, 3 only (the sizes of the challenge molecules). The global phase -i of |-> is irrelevant for expectation values.",
    "script": "agents/agent1_physics/t1_xbasis.py",
    "command": cmd("t1_xbasis.py"),
})
print("ALL T1 CHECKS PASS:", ok)
