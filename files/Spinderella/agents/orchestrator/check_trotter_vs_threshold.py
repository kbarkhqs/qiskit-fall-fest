"""Is the gap between the measured any-dip threshold of A (27.3 s^-1) and the pre-registered
continuum prediction (32.5 s^-1) caused by first-order Trotter error?
Exact evolution (dense expm of the same rotating-frame H) vs the official Lie-Trotter step, both on
a long acquisition (2400 x 2.5 ms) so that truncation and grid effects are small; the same
criterion (characteristic peaks of the matched reference, +-2.667 Hz, adjacent dips) is applied.
Also measures the spacing of the critical pair in the long-acquisition spectra."""
import pickle, sys
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.nmr_pipeline as P  # noqa: E402

HZ = 1 / P.hz_to_ppm_conversion()
DT = 0.0025


def ref_peaks(k):
    d = pickle.load(open(ROOT / f"files/spectrum_{k}.pkl", "rb"))
    p, s = np.asarray(d["frequencies"], float), np.asarray(d["spectrum"], float)
    o = np.argsort(p); p, s = p[o], s[o] / s.max()
    idx, _ = find_peaks(s, height=0.05, prominence=0.02)
    groups = [[idx[0]]]
    for i in idx[1:]:
        (groups[-1].append(i) if (p[i] - p[groups[-1][-1]]) * HZ < 2.667 else groups.append([i]))
    return np.array([np.average(p[g], weights=s[g]) for g in groups])


def exact_signals(H, n, L):
    Hm = H.to_matrix()
    X = sum(P.SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n).to_matrix() for i in range(n))
    Y = sum(P.SparsePauliOp.from_sparse_list([("Y", [i], 1.0)], num_qubits=n).to_matrix() for i in range(n))
    U = expm(-1j * Hm * DT)
    # infinite-temperature correlation, same as the weighted sum over X eigenstates (Pauli units)
    cxx, cyx = np.zeros(L), np.zeros(L)
    Ut = np.eye(2**n, dtype=complex)
    for k in range(L):
        cxx[k] = np.real(np.trace(Ut.conj().T @ X @ Ut @ X))
        cyx[k] = np.real(np.trace(Ut.conj().T @ Y @ Ut @ X))
        Ut = U @ Ut
    return cxx, cyx


def threshold(cxx, cyx, chem, L, centres, lev):
    def ok(g):
        s, p = P.spectrum_from_signals(cxx, cyx, g, DT, L, chem)
        p = np.asarray(p, float)
        mx, _ = find_peaks(s); ch = []
        for c in centres:
            near = mx[np.abs(p[mx] - c) * HZ <= 2.667]
            if len(near) == 0:
                return False
            ch.append(near[np.argmax(s[near])])
        if np.any(np.diff(ch) <= 0):
            return False
        d = [1 - s[a:b + 1].min() / min(s[a], s[b]) for a, b in zip(ch[:-1], ch[1:])]
        return all(x > lev for x in d) if lev == 0 else all(x >= lev for x in d)
    gs = np.round(np.arange(10, 45, 0.05), 2)
    f = np.array([ok(g) for g in gs])
    return gs[f].max()


def pair_spacing(cxx, cyx, chem, L, a, b):
    s, p = P.spectrum_from_signals(cxx, cyx, 0.5, DT, L, chem)
    p = np.asarray(p, float); mx, _ = find_peaks(s)
    pa = p[mx[np.argmin(np.abs(p[mx] - a))]]; pb = p[mx[np.argmin(np.abs(p[mx] - b))]]
    return (pb - pa) * HZ


L = 2400
for name, k, pair in (("molecule A", 3, (5.4728, 5.4850)), ("molecule B", 2, (5.8938, 5.9079)), ("molecule C", 1, (6.2932, 6.3091))):
    H, n, chem = P.load_rotating_frame_hamiltonian(name)
    centres = ref_peaks(k)
    ex, ey = P.simulate_signals_fast(H, n, number_of_trottersteps=L, trotter_timestep=DT)
    tx, ty = P.correlations(ex, ey)
    qx, qy = exact_signals(H, n, L)
    print(f"{name}: long acquisition {L} steps; "
          f"any-dip threshold Trotter {threshold(tx, ty, chem, L, centres, 0.0):.2f} vs exact {threshold(qx, qy, chem, L, centres, 0.0):.2f}; "
          f"10% Trotter {threshold(tx, ty, chem, L, centres, 0.10):.2f} vs exact {threshold(qx, qy, chem, L, centres, 0.10):.2f}; "
          f"critical-pair spacing Trotter {pair_spacing(tx, ty, chem, L, *pair):.3f} Hz vs exact {pair_spacing(qx, qy, chem, L, *pair):.3f} Hz")
