"""Final audit: independent re-implementation (NumPy/SciPy only for the physics).

Only the supplied spectrum transform is reused (fft_causal / time_signal_to_spectrum, verbatim
copies of the notebook in src/nmr_pipeline.py), because it defines the official spectrum.
Everything else - Hamiltonian from the JSON files, initial states, Trotter product, exact
evolution, effective Trotter lines, peak finding, dip criterion, scans, matching - is written
here from scratch and does not call the notebook's or the agents' analysis functions.
"""
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import expm, schur
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.nmr_pipeline import fft_causal, time_signal_to_spectrum  # noqa: E402  (official transform)

GAMMA = 267_522_128.0
FIELD = 11.7433
NU0 = GAMMA * FIELD / (2 * np.pi)  # Hz
HZ_TO_PPM = 1e6 / NU0
HZ_PER_PPM = 1.0 / HZ_TO_PPM
DT, L = 0.0025, 300
PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]], complex),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0 + 0j, -1.0])}


# ---------------------------------------------------------------- Hamiltonian from the JSON file
def load_terms(m, overrides=None):
    """Parse files/molecule m_B11.7433T.json -> (n, [(dict site->op, coeff)] in file order)."""
    items = json.loads(json.loads((ROOT / f"files/molecule {m}_B11.7433T.json").read_text()))["items"]
    import re
    terms = []
    for key, c in items:
        ops = {int(i): p for i, p in re.findall(r"(\d+)([XYZ])", key)}
        if overrides and len(ops) == 2:
            pair = tuple(sorted(ops))
            if pair in overrides:
                c = np.pi * overrides[pair] / 2
        terms.append((ops, float(c)))
    n = 1 + max(i for ops, _ in terms for i in ops)
    return n, terms


def pauli_matrix(ops, n):
    """Little-endian (qubit i = site i, Qiskit convention): kron(op_{n-1}, ..., op_0)."""
    m = np.array([[1.0 + 0j]])
    for q in reversed(range(n)):
        m = np.kron(m, PAULI[ops.get(q, "I")])
    return m


def rotating_frame(m, overrides=None):
    n, terms = load_terms(m, overrides)
    hz = [c for ops, c in terms if len(ops) == 1]
    mean_z = float(np.mean(hz))
    rf = [(ops, c - mean_z if len(ops) == 1 else c) for ops, c in terms]
    H = sum(c * pauli_matrix(ops, n) for ops, c in rf)
    return n, rf, H, mean_z


def shifts_and_couplings(m):
    n, terms = load_terms(m)
    delta = {i: (-2 * c / (GAMMA * FIELD) - 1) * 1e6 for ops, c in terms if len(ops) == 1 for i in ops}
    J = {}
    for ops, c in terms:
        if len(ops) == 2 and set(ops.values()) == {"X"}:
            i, j = sorted(ops)
            J[f"J{i}{j}"] = 2 * c / np.pi
    return delta, J


def sum_ops(n):
    X = sum(pauli_matrix({i: "X"}, n) for i in range(n))
    Y = sum(pauli_matrix({i: "Y"}, n) for i in range(n))
    return X, Y


def x_basis_states(n):
    plus = np.array([1, 1]) / np.sqrt(2)
    minus = np.array([1, -1]) / np.sqrt(2)
    bits = list(product((0, 1), repeat=n))
    states = []
    for b in bits:
        v = np.array([1.0 + 0j])
        for q in reversed(range(n)):
            v = np.kron(v, minus if b[q] else plus)
        states.append(v)
    return bits, np.array(states)


# ---------------------------------------------------------------- dynamics
def trotter_unitary(rf, n, dt):
    U = np.eye(2**n, dtype=complex)
    for ops, c in rf:  # file order: first term applied first
        U = expm(-1j * c * dt * pauli_matrix(ops, n)) @ U
    return U


def signals_state_sum(U, n, steps):
    """Notebook recipe: sum_n m_n <n| A(t_k) |n>, states evolved by U^k (independent code)."""
    X, Y = sum_ops(n)
    bits, states = x_basis_states(n)
    m = np.array([n - 2 * sum(b) for b in bits], float)
    psi = states.T.copy()  # columns
    cxx, cyx = np.zeros(steps), np.zeros(steps)
    for k in range(steps):
        cxx[k] = m @ np.real(np.einsum("ij,ij->j", psi.conj(), X @ psi))
        cyx[k] = m @ np.real(np.einsum("ij,ij->j", psi.conj(), Y @ psi))
        psi = U @ psi
    return cxx, cyx


def exact_signals(H, n, dt, steps):
    """Tr(A(t) X_tot) in the eigenbasis of H (no Trotter error)."""
    X, Y = sum_ops(n)
    E, V = np.linalg.eigh(H)
    xe, ye = V.conj().T @ X @ V, V.conj().T @ Y @ V
    t = np.arange(steps) * dt
    ph = np.exp(1j * np.subtract.outer(E, E)[:, :, None] * t)
    return (np.einsum("ab,ba,abt->t", xe, xe, ph).real, np.einsum("ab,ba,abt->t", ye, xe, ph).real)


def lines_from_unitary(U, n, dt, min_rel=1e-3):
    """Lines of S = Re FT[C_xx - i C_yx] for evolution by U per step: frequency (Hz, rotating
    frame) and weight Re[(X - iY)'_ab X'_ba] in U's eigenbasis."""
    X, Y = sum_ops(n)
    T, Z = schur(U, output="complex")
    phi = -np.angle(np.diag(T))  # U = Z diag(e^{-i phi}) Z^dagger
    Am = Z.conj().T @ (X - 1j * Y) @ Z
    Xm = Z.conj().T @ X @ Z
    w = (Am * Xm.T).real
    nu = np.subtract.outer(phi, phi) / (2 * np.pi * dt)
    nu = (nu + 0.5 / dt) % (1 / dt) - 0.5 / dt  # alias into the Nyquist window
    keep = w > min_rel * w.max()
    order = np.argsort(nu[keep])
    return nu[keep][order], w[keep][order]


def exact_lines(H, n, min_rel=1e-3):
    X, Y = sum_ops(n)
    E, V = np.linalg.eigh(H)
    Am = V.conj().T @ (X - 1j * Y) @ V
    Xm = V.conj().T @ X @ V
    w = (Am * Xm.T).real
    nu = np.subtract.outer(E, E) / (2 * np.pi)
    keep = w > min_rel * w.max()
    order = np.argsort(nu[keep])
    return nu[keep][order], w[keep][order]


# ---------------------------------------------------------------- spectra
def ppm_offset(mean_z):
    """ppm of zero rotating-frame offset on the official axis: -(2 c/(2pi/conv) + 1e6)."""
    return -(2 * mean_z / (2.0 * np.pi / HZ_TO_PPM) + 1e6)


def official_spectrum(cxx, cyx, gamma, dt, steps, mean_z):
    t = np.arange(steps) * dt
    w = np.exp(-gamma * t)
    s, p = time_signal_to_spectrum(cxx * w, cyx * w, dt, steps, HZ_TO_PPM, mean_z)
    return s, np.asarray(p, float)


def signed_spectrum(cxx, cyx, gamma, dt, steps):
    t = np.arange(steps) * dt
    w = np.exp(-gamma * t)
    s = fft_causal(cxx * w).real + fft_causal(cyx * w).imag
    return s / np.max(np.abs(s))


def load_reference(k):
    d = pickle.load(open(ROOT / f"files/spectrum_{k}.pkl", "rb"))
    p, s = np.asarray(d["frequencies"], float), np.asarray(d["spectrum"], float)
    o = np.argsort(p)
    return p[o], s[o] / np.max(np.abs(s))


def stick_reference(nu, w, ppm0, fwhm_hz=1.0, step_hz=0.01, pad_hz=30.0):
    grid = np.arange(nu.min() - pad_hz, nu.max() + pad_hz, step_hz)
    hw = fwhm_hz / 2
    s = sum(a * hw**2 / ((grid - x) ** 2 + hw**2) for x, a in zip(nu, w))
    return ppm0 + grid / HZ_PER_PPM, s / s.max()


# ---------------------------------------------------------------- criterion (fresh implementation)
def char_peaks(r_ppm, r_s, merge_hz, height=0.05, prom=0.02):
    idx, _ = find_peaks(r_s, height=height, prominence=prom)
    groups = [[idx[0]]]
    for i in idx[1:]:
        if (r_ppm[i] - r_ppm[groups[-1][-1]]) * HZ_PER_PPM < merge_hz:
            groups[-1].append(i)
        else:
            groups.append([i])
    return np.array([np.sum(r_ppm[g] * r_s[g]) / np.sum(r_s[g]) for g in groups])


def evaluate(spec, ppm, centres, tol_hz):
    """-> (found, dips, reason). dip = 1 - min(valley between the two matched maxima) / min(peaks)."""
    mx, _ = find_peaks(spec)
    chosen = []
    for c in centres:
        near = mx[np.abs(ppm[mx] - c) * HZ_PER_PPM <= tol_hz]
        if len(near) == 0:
            return False, np.array([]), f"missing {c:.4f}"
        chosen.append(near[np.argmax(spec[near])])
    chosen = np.array(chosen)
    if np.any(np.diff(chosen) <= 0):
        return False, np.array([]), "shared or out of order"
    dips = np.array([1 - spec[a:b + 1].min() / min(spec[a], spec[b]) for a, b in zip(chosen[:-1], chosen[1:])])
    return True, dips, ""


def passes(found, dips, level):
    if not found:
        return False
    return bool(np.all(dips > 0)) if level == 0 else bool(np.all(dips >= level))


def full_scan(cxx, cyx, mean_z, centres, tol_hz, levels, dt=DT, steps=L, grid=None):
    """Evaluate every gamma on a fixed grid (default 0.01..100 step 0.01, no refinement shortcut)."""
    grid = np.round(np.arange(0.01, 100.0 + 1e-9, 0.01), 2) if grid is None else grid
    ok = {lv: np.zeros(len(grid), bool) for lv in levels}
    for i, g in enumerate(grid):
        s, p = official_spectrum(cxx, cyx, g, dt, steps, mean_z)
        f, d, _ = evaluate(s, p, centres, tol_hz)
        for lv in levels:
            ok[lv][i] = passes(f, d, lv)
    out = {}
    for lv in levels:
        o = ok[lv]
        ch = np.where(o[1:] != o[:-1])[0]
        out[lv] = {"highest_pass": float(grid[o].max()) if o.any() else None,
                   "first_fail_after": float(grid[(grid > grid[o].max()) & ~o].min()) if o.any() and np.any((grid > grid[o].max()) & ~o) else None,
                   "n_changes": int(len(ch)),
                   "changes": [(float(grid[i]), float(grid[i + 1]), "pass->fail" if o[i] else "fail->pass") for i in ch],
                   "passes_at_lowest": bool(o[0])}
    return out
