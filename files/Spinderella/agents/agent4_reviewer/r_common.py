"""Agent 4 (independent reviewer): own helpers, written without calling any agent's analysis code.

Independent pieces (own code):
  * Hamiltonian built from the JSON file by my own parser (or from delta/J parameters),
    own Pauli matrices (qubit i = tensor factor i, ordering only has to be self-consistent).
  * Exact dynamics (eigendecomposition) and first-order Lie-Trotter (product of exact
    exponentials of the Pauli terms, in the file's term order, first term applied first).
  * Own reference loader, own local-maximum finder, own characteristic-peak grouping, own
    detection / dip / pass logic, own threshold scans.
Shared with the notebook (allowed by the brief): ``nmr_pipeline.spectrum_from_signals`` for the
OFFICIAL spectrum (window, causal FFT, ppm axis, /max, abs). ``spectrum_batch`` below is my own
vectorised re-implementation; ``check_batch_equals_pipeline`` verifies it equals the pipeline.

Units: Hamiltonian coefficients in rad/s (Pauli convention), J in Hz, nu0 = gamma B / 2 pi.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
import nmr_pipeline as nmr  # noqa: E402  (official spectrum only)

GAMMA_1H = 267_522_128.0  # rad s^-1 T^-1 (notebook cell 25)
FIELD = 11.7433  # T
NU0 = GAMMA_1H * FIELD / (2 * np.pi)  # Hz (Larmor = reference frequency)
HZ_PER_PPM = NU0 * 1e-6
DT = 0.0025
L_OFF = 300
GRID_HZ = 1.0 / (L_OFF * DT)
MATCH = {"A": 3, "B": 2, "C": 1}  # claim C3, used only AFTER r3 has tested it

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"X": PX, "Y": PY, "Z": PZ}


def op(n, ops):
    """Kronecker product; ops = {site: 'X'|'Y'|'Z'}; site 0 is the LEFTMOST factor."""
    m = np.array([[1.0 + 0j]])
    for i in range(n):
        m = np.kron(m, PAULI[ops[i]] if i in ops else I2)
    return m


def parse_json_terms(mol):
    """Own parser of files/molecule X_B11.7433T.json -> list of (dict site->pauli, coeff)."""
    raw = json.load(open(REPO / "files" / f"molecule {mol}_B11.7433T.json"))
    d = json.loads(raw) if isinstance(raw, str) else raw
    terms = []
    for key, val in d["items"]:
        ops = {}
        num = ""
        for ch in key:
            if ch.isdigit():
                num += ch
            else:
                ops[int(num)] = ch
                num = ""
        terms.append((ops, float(val)))
    n = 1 + max(max(o) for o, _ in terms)
    return terms, n


def terms_from_params(delta_ppm, J_hz):
    """H = sum h_i Z_i + sum c_ij (XX+YY+ZZ), h_i = -gamma B (1+delta_i)/2, c_ij = pi J_ij / 2.
    J_hz: dict {(i, j): J}. Term order mimics the files (all Z, then pairs (0,1),(0,2),(1,2) XX,YY,ZZ)."""
    n = len(delta_ppm)
    terms = [({i: "Z"}, -GAMMA_1H * FIELD * (1 + delta_ppm[i] * 1e-6) / 2) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in J_hz:
                c = np.pi * J_hz[(i, j)] / 2
                for p in "XYZ":
                    terms.append(({i: p, j: p}, c))
    return terms, n


def rotating_frame(terms, n):
    """Subtract the mean single-spin Z coefficient (exact because sum Z commutes with H)."""
    hz = [c for o, c in terms if len(o) == 1 and list(o.values())[0] == "Z"]
    hbar = float(np.mean(hz))
    new = [(o, c - hbar) if (len(o) == 1 and list(o.values())[0] == "Z") else (o, c) for o, c in terms]
    return new, hbar


def H_matrix(terms, n):
    return sum(c * op(n, o) for o, c in terms)


def tot(n, p):
    return sum(op(n, {i: p}) for i in range(n))


def trotter_step(terms, n, dt=DT):
    """First-order Lie-Trotter step: first term applied first (as the Qiskit circuit does)."""
    U = np.eye(2**n, dtype=complex)
    for o, c in terms:
        P = op(n, o)
        # exp(-i c P dt) = cos(c dt) I - i sin(c dt) P for a Pauli string
        U = (np.cos(c * dt) * np.eye(2**n) - 1j * np.sin(c * dt) * P) @ U
    return U


def exact_step(terms, n, dt=DT):
    H = H_matrix(terms, n)
    E, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * E * dt)) @ V.conj().T


def trace_correlations(U, n, L):
    """Infinite-temperature correlations in Pauli units: C_ax(k) = Tr(U^-k A U^k X_tot)."""
    X, Y = tot(n, "X"), tot(n, "Y")
    w, V = np.linalg.eig(U)  # U unitary, eigenvalues on the unit circle
    Vi = np.linalg.inv(V)
    Xp, Yp = Vi @ X @ V, Vi @ Y @ V
    # Tr(U^-k A U^k X) = sum_ab conj-phase... use the general formula with eigenvalues lam:
    # U^k = V lam^k V^-1 ; U^-k A U^k X -> trace = sum_ab lam_a^-k lam_b^k A'_ab X'_ba
    lam = w
    k = np.arange(L)
    ph = (lam[None, :] ** (-k[:, None]))  # (L, d) for a
    pb = (lam[None, :] ** (k[:, None]))
    Mxx = Xp * Xp.T  # A'_ab X'_ba
    Myx = Yp * Xp.T
    cxx = np.einsum("ka,ab,kb->k", ph, Mxx, pb).real
    cyx = np.einsum("ka,ab,kb->k", ph, Myx, pb).real
    return cxx, cyx


def xstates(n):
    """The 2^N X-product states |s>, bits b_i in {0,1} (0 -> |+>, 1 -> |->), with m_s = N - 2w."""
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    out = []
    for idx in range(2**n):
        bits = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
        v = np.array([1.0 + 0j])
        for b in bits:
            v = np.kron(v, minus if b else plus)
        out.append((tuple(bits), v))
    return out


def state_signals(U, n, L, psis):
    """<psi| U^-k A U^k |psi> for A = X_tot, Y_tot; psis list of vectors -> (len, L) arrays."""
    X, Y = tot(n, "X"), tot(n, "Y")
    ex = np.zeros((len(psis), L))
    ey = np.zeros((len(psis), L))
    for m, psi in enumerate(psis):
        v = psi.copy()
        for k in range(L):
            ex[m, k] = np.real(v.conj() @ X @ v)
            ey[m, k] = np.real(v.conj() @ Y @ v)
            v = U @ v
    return ex, ey


# ------------------------------------------------------------------------------------------
# Spectra
# ------------------------------------------------------------------------------------------
def official_spectrum(cxx, cyx, noise, shift, L=L_OFF, dt=DT):
    s, p = nmr.spectrum_from_signals(cxx, cyx, float(noise), dt, L, shift)
    return np.asarray(s, float), np.asarray(p, float)


def _kernel(L, dt):
    """DFT kernel of the pipeline: S_j = Re sum_k g_k c_k e^{-2 pi i f_j t_k} / sqrt(2L-1),
    f_j = (2j-(L-1))/((2L-1) dt), g_0 = 1/2. (Derived by hand from fft_causal; verified below.)"""
    M = 2 * L - 1
    j = np.arange(L)
    f = (2 * j - (L - 1)) / (M * dt)
    t = np.arange(L) * dt
    g = np.ones(L)
    g[0] = 0.5
    K = np.exp(-2j * np.pi * np.outer(f, t)) * g[None, :] / np.sqrt(M)
    return K, f


def ppm_axis(shift, L=L_OFF, dt=DT):
    """Pipeline's ppm labels: fftshift(fftfreq(L, dt)) * conv - (shift*conv/pi + 1e6)... written out."""
    conv = 1e6 / NU0
    freqs = np.fft.fftshift(np.fft.fftfreq(L, d=dt))
    chem = 2 * shift / (2.0 * np.pi / conv) + 1e6
    return freqs * conv - chem


def spectrum_batch(cxx, cyx, noises, shift, L=L_OFF, dt=DT, signed=False):
    """Own vectorised spectra for many noise values. signed=True skips the final np.abs."""
    K, _ = _kernel(L, dt)
    t = np.arange(L) * dt
    W = np.exp(-np.outer(np.asarray(noises, float), t))  # (G, L)
    Fx = (W * cxx[None, :]) @ K.T
    Fy = (W * cyx[None, :]) @ K.T
    S = Fx.real + Fy.imag
    S = S / np.max(np.abs(S), axis=1, keepdims=True)
    return (S if signed else np.abs(S)), ppm_axis(shift, L, dt)


def check_batch_equals_pipeline(cxx, cyx, shift, L=L_OFF, noises=(0.1, 1, 5, 20, 37.3, 100)):
    S, p = spectrum_batch(cxx, cyx, noises, shift, L)
    dev = 0.0
    for g, s in zip(noises, S):
        s0, p0 = official_spectrum(cxx, cyx, g, shift, L)
        dev = max(dev, float(np.max(np.abs(s - s0))), float(np.max(np.abs(p - p0))))
    return dev


# ------------------------------------------------------------------------------------------
# References and peak logic (own implementation)
# ------------------------------------------------------------------------------------------
def load_ref(k):
    d = pickle.load(open(REPO / "files" / f"spectrum_{k}.pkl", "rb"))
    p = np.asarray(d["frequencies"], float)
    s = np.asarray(d["spectrum"], float)
    o = np.argsort(p)
    return p[o], s[o] / np.max(np.abs(s[o]))


def local_maxima(s):
    """Indices of strict interior local maxima (own definition, no scipy)."""
    return np.nonzero((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]))[0] + 1


def prominence(s, i):
    """Topographic prominence of the maximum at i (own implementation)."""
    h = s[i]
    lft = i
    while lft > 0 and s[lft - 1] <= h:
        lft -= 1
    rgt = i
    while rgt < len(s) - 1 and s[rgt + 1] <= h:
        rgt += 1
    # base: highest of the two minima between i and the nearest higher point on each side
    lmin = s[lft:i + 1].min()
    rmin = s[i:rgt + 1].min()
    return h - max(lmin, rmin)


def char_peaks_from(ppm, s, height=0.05, prom=0.02, merge_hz=2 * GRID_HZ):
    """Characteristic peaks: maxima with height >= height and prominence >= prom (fractions of
    max), single-linkage merge of neighbours closer than merge_hz, height-weighted centre."""
    s = s / s.max()
    lm = local_maxima(s)
    keep = [i for i in lm if s[i] >= height and prominence(s, i) >= prom]
    groups = []
    for i in keep:
        if groups and (ppm[i] - ppm[groups[-1][-1]]) * HZ_PER_PPM < merge_hz:
            groups[-1].append(i)
        else:
            groups.append([i])
    centres = np.array([np.sum(ppm[g] * s[g]) / np.sum(s[g]) for g in groups])
    return centres


def assess(s, ppm, centres, tol_hz=2 * GRID_HZ, lm=None):
    """Detection + dips. Returns (found, dips array, reason). Own logic, mirrors the stated
    criterion: highest local maximum within tol of each centre, one-to-one and ordered;
    dip = 1 - min(valley) / min(two maxima)."""
    if lm is None:
        lm = local_maxima(s)
    pl = ppm[lm]
    chosen = []
    for c in centres:
        near = lm[np.abs(pl - c) * HZ_PER_PPM <= tol_hz]
        if len(near) == 0:
            return False, None, "missing"
        chosen.append(near[np.argmax(s[near])])
    chosen = np.array(chosen)
    if np.any(np.diff(chosen) <= 0):
        return False, None, "shared_or_order"
    dips = np.array([1 - s[a:b + 1].min() / min(s[a], s[b]) for a, b in zip(chosen[:-1], chosen[1:])])
    return True, dips, ""


def passes(found, dips, level):
    if not found:
        return False
    if dips is None or len(dips) == 0:
        return True
    return bool(np.all(dips > 0)) if level == 0 else bool(np.all(dips >= level))


def ascending(ppm, S):
    """The pipeline ppm axis is ascending already; enforce it for safety."""
    o = np.argsort(ppm)
    return ppm[o], (S[:, o] if S.ndim == 2 else S[o])


def scan_threshold(flags, grid):
    """flags over an ascending grid -> (max_pass, first_fail_after_it, transitions list)."""
    flags = np.asarray(flags, bool)
    tr = [(float(grid[i]), float(grid[i + 1]), "pass->fail" if flags[i] else "fail->pass")
          for i in np.nonzero(flags[1:] != flags[:-1])[0]]
    if not flags.any():
        return None, float(grid[0]), tr
    imax = int(np.max(np.nonzero(flags)[0]))
    ff = float(grid[imax + 1]) if imax + 1 < len(grid) else None
    return float(grid[imax]), ff, tr


def load_official_signals(m):
    z = np.load(REPO / "results" / "signals" / f"molecule_{m}.npz")
    return dict(cxx=np.asarray(z["c_xx_raw"], float), cyx=np.asarray(z["c_yx_raw"], float),
                shift=float(z["chemical_shift_hz"]), n=int(z["number_spins"]),
                exp_x=np.asarray(z["exp_x"]), exp_y=np.asarray(z["exp_y"]), bits=np.asarray(z["bits"]))


def save(name, obj):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))
    (OUT / name).write_text(json.dumps(obj, indent=1, default=conv))


def lines_from_step(U, n, dt, thresh=1e-3):
    """Lines (Hz) and complex amplitudes c = a - i b of S = Re FT Cxx + Im FT Cyx, from the
    eigen-decomposition of the one-step propagator: C_AX(k) = sum_ab e^{i(th_a - th_b)k} A'_ab X'_ba,
    U = V diag(e^{-i th}) V^-1 (exact: th = E dt)."""
    lam, V = np.linalg.eig(U)
    Vi = np.linalg.inv(V)
    th = -np.angle(lam)
    X, Y = tot(n, "X"), tot(n, "Y")
    Xp, Yp = Vi @ X @ V, Vi @ Y @ V
    a = Xp * Xp.T
    b = Yp * Xp.T
    c = a - 1j * b
    w = (th[:, None] - th[None, :])
    w = (w + np.pi) % (2 * np.pi) - np.pi  # alias into (-pi, pi]
    f = w / (2 * np.pi * dt)
    keep = np.abs(c) > thresh * np.abs(c).max()
    f, c = f[keep], c[keep]
    # merge degenerate entries (within 1e-6 Hz)
    o = np.argsort(f)
    f, c = f[o], c[o]
    F, C = [], []
    for fi, ci in zip(f, c):
        if F and abs(fi - F[-1]) < 1e-6:
            C[-1] += ci
        else:
            F.append(fi)
            C.append(ci)
    F, C = np.array(F), np.array(C)
    keep = np.abs(C) > thresh * np.abs(C).max()
    return F[keep], C[keep]
