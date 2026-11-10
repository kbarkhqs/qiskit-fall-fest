"""Shared helpers for Agent 1 (independent physics reference). NumPy/SciPy only, no Qiskit.

Qubit-ordering convention used everywhere in this folder:
site 0 is the RIGHTMOST Kronecker factor, so the basis index is sum_i b_i 2^i
(little-endian). All reported numbers are independent of this choice.
"""
from __future__ import annotations

import json
import pickletools
import re
from functools import reduce
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ROOT / "files"

GAMMA = 267_522_128.0  # rad s^-1 T^-1 (value used by the notebook)
B0 = 11.7433  # T
NU0 = GAMMA * B0 / (2 * np.pi)  # reference frequency, Hz
HZ_TO_PPM = 1e6 / NU0
DT = 0.0025  # notebook default Trotter step, s
NSTEPS = 300  # notebook default number of time points

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"X": X, "Y": Y, "Z": Z}
TOKEN = re.compile(r"(\d+)([XYZ])")


def kron_all(mats):
    return reduce(np.kron, mats)


def site_op(P, i, n):
    """Single-site operator P on site i of n (site 0 = rightmost factor)."""
    mats = [I2] * n
    mats[n - 1 - i] = P
    return kron_all(mats)


def total(P, n):
    return sum(site_op(P, i, n) for i in range(n))


def load_terms(mol):
    """Terms [(key, coeff)] in file order. The file is a JSON string holding JSON."""
    with open(FILES / f"molecule {mol}_B{B0}T.json") as fh:
        obj = json.load(fh)
    if isinstance(obj, str):
        obj = json.loads(obj)
    return [(str(k), float(v)) for k, v in obj["items"]]


def parse_key(key):
    return [(int(m.group(1)), m.group(2)) for m in TOKEN.finditer(key)]


def n_spins(terms):
    return 1 + max(i for k, _ in terms for i, _ in parse_key(k))


def term_matrix(key, n):
    mats = [I2] * n
    for i, p in parse_key(key):
        mats[n - 1 - i] = PAULI[p]
    return kron_all(mats)


def dense_h(terms, n):
    return sum(c * term_matrix(k, n) for k, c in terms)


def rotating_frame(terms):
    """Notebook cell 6: subtract the mean single-Z coefficient from every iZ term."""
    zs = [c for k, c in terms if len(parse_key(k)) == 1 and parse_key(k)[0][1] == "Z"]
    mean_h = float(np.mean(zs))
    out = []
    for k, c in terms:
        t = parse_key(k)
        out.append((k, c - mean_h) if (len(t) == 1 and t[0][1] == "Z") else (k, c))
    return out, mean_h


def physical_params(terms):
    """delta_i (ppm) from h_i = -gamma B (1+delta_i)/2 ; J_ij (Hz) from c_ij = pi J_ij/2."""
    n = n_spins(terms)
    h = np.zeros(n)
    c = {}
    for k, v in terms:
        t = parse_key(k)
        if len(t) == 1:
            assert t[0][1] == "Z"
            h[t[0][0]] = v
        else:
            (i, p), (j, q) = t
            assert p == q
            c.setdefault((i, j), {})[p] = v
    for pair, d in c.items():
        assert set(d) == {"X", "Y", "Z"} and len(set(d.values())) == 1, (pair, d)
    delta_ppm = (-2 * h / (GAMMA * B0) - 1) * 1e6
    j_hz = {pair: 2 * d["X"] / np.pi for pair, d in c.items()}
    return n, h, delta_ppm, j_hz


def load_reference(i):
    """Read files/spectrum_i.pkl WITHOUT unpickling: the two float64 buffers are taken
    straight from the opcode stream (pickletools.genops executes nothing)."""
    data = (FILES / f"spectrum_{i}.pkl").read_bytes()
    keys, bufs = [], []
    for op, arg, _ in pickletools.genops(data):
        if op.name in ("BYTEARRAY8", "BINBYTES", "BINBYTES8", "SHORT_BINBYTES"):
            bufs.append(bytes(arg))
        elif op.name == "SHORT_BINUNICODE" and arg in ("frequencies", "spectrum"):
            keys.append(arg)
    assert keys == ["frequencies", "spectrum"] and len(bufs) == 2, (keys, len(bufs))
    return {k: np.frombuffer(b, dtype="<f8").copy() for k, b in zip(keys, bufs)}


def eig_sector_safe(H, n):
    """Eigen-decomposition of H with eigenvectors of definite total Z (H commutes with
    sum Z; the added multiple of sum Z only prevents mixing of accidental degeneracies)."""
    zt = total(Z, n)
    _, V = np.linalg.eigh(H + 1234.5678 * np.sqrt(2.0) * zt)
    E = np.real(np.einsum("ia,ij,ja->a", V.conj(), H, V))
    resid = np.abs(H @ V - V * E).max()
    assert resid < 1e-8 * max(1.0, np.abs(E).max()), resid
    return E, V


def line_list(H, n, rel_tol=1e-12):
    """All allowed transitions of the rotating-frame H as they appear in the notebook
    spectrum Re F[C_xx] + Im F[C_yx] (F = one-sided FT with exp(-i w t)).

    C_xx + i C_yx = (1/2) sum_ab |<a|F+|b>|^2 exp(i (E_a - E_b) t),  F+ = sum_i (X_i + i Y_i)
    -> line at  nu = (E_b - E_a)/(2 pi) Hz  with weight A = |<a|F+|b>|^2 / 2  (sum A = N 2^N).
    Returned intensity = A / 2^N (sum = N, one unit per spin). 'spin' = site with the
    largest single-site matrix element.
    """
    E, V = eig_sector_safe(H, n)
    fp_sites = [V.conj().T @ site_op(X + 1j * Y, i, n) @ V for i in range(n)]
    fp = sum(fp_sites)
    A = np.abs(fp) ** 2 / 2
    lines = []
    for a in range(2**n):
        for b in range(2**n):
            if A[a, b] > rel_tol * A.max():
                w = [abs(m[a, b]) ** 2 for m in fp_sites]
                lines.append(((E[b] - E[a]) / (2 * np.pi), A[a, b] / 2**n, int(np.argmax(w)), a, b))
    lines.sort()
    hz = np.array([l[0] for l in lines])
    inten = np.array([l[1] for l in lines])
    spin = np.array([l[2] for l in lines])
    return hz, inten, spin


def x_basis_states(n):
    """Columns = the 2^N states prod_i RZ(pi b_i) H |0>, column index = sum_i b_i 2^i."""
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    minus = -1j * np.array([1, -1], dtype=complex) / np.sqrt(2)  # RZ(pi) H |0> = -i |->
    cols = []
    for idx in range(2**n):
        bits = [(idx >> i) & 1 for i in range(n)]
        cols.append(kron_all([minus if bits[n - 1 - q] else plus for q in range(n)]))
    return np.array(cols).T


def exact_signals(H, n, times):
    """C_xx(t) = Tr(X_tot(t) X_tot), C_yx(t) = Tr(Y_tot(t) X_tot) (Pauli, no 1/2^N)."""
    E, V = np.linalg.eigh(H)
    xt = V.conj().T @ total(X, n) @ V
    yt = V.conj().T @ total(Y, n) @ V
    ph = np.exp(1j * (E[:, None] - E[None, :])[None] * np.asarray(times)[:, None, None])
    cxx = np.einsum("tab,ab,ba->t", ph, xt, xt)
    cyx = np.einsum("tab,ab,ba->t", ph, yt, xt)
    assert np.abs(cxx.imag).max() < 1e-9 and np.abs(cyx.imag).max() < 1e-9
    return cxx.real, cyx.real


def trotter_step(terms, n, dt, order=None):
    """First-order Lie-Trotter step: exp(-i c P dt) applied term by term, first term in
    the list acts first on the state (U = U_last ... U_1)."""
    U = np.eye(2**n, dtype=complex)
    seq = terms if order is None else [terms[i] for i in order]
    for k, c in seq:
        P = term_matrix(k, n)
        U = (np.cos(c * dt) * np.eye(2**n) - 1j * np.sin(c * dt) * P) @ U
    return U


def state_sum_signals(U, n, nsteps):
    """Notebook recipe: sum over the 2^N X-basis states of m_n <psi_n(k)|O|psi_n(k)>,
    m_n = <psi_n(0)|X_tot|psi_n(0)>, psi_n(k) = U^k psi_n(0)."""
    xt, yt = total(X, n), total(Y, n)
    psi = x_basis_states(n)
    m = np.real(np.einsum("in,ij,jn->n", psi.conj(), xt, psi))
    cxx, cyx = np.zeros(nsteps), np.zeros(nsteps)
    for k in range(nsteps):
        cxx[k] = np.sum(m * np.real(np.einsum("in,ij,jn->n", psi.conj(), xt, psi)))
        cyx[k] = np.sum(m * np.real(np.einsum("in,ij,jn->n", psi.conj(), yt, psi)))
        psi = U @ psi
    return cxx, cyx


def true_freq_axis(n, dt):
    """Frequencies (Hz) actually sampled by the notebook's fft_causal: odd harmonics of
    1/((2n-1) dt). Derived in report.md (T4), confirmed in t4_sign_axis.py."""
    return (2 * np.arange(n) - (n - 1)) / ((2 * n - 1) * dt)


def record(entry):
    """Insert or replace one experiment entry (by id) in evidence.json."""
    path = HERE / "evidence.json"
    items = json.loads(path.read_text()) if path.exists() else []
    items = [e for e in items if e["id"] != entry["id"]] + [entry]
    items.sort(key=lambda e: e["id"])
    path.write_text(json.dumps(items, indent=1, default=float) + "\n")


def cmd(script):
    return f".venv/bin/python -I agents/agent1_physics/{script}"
