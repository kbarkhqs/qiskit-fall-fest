"""Shared helpers for Agent 3 (spectral analysis and optimisation).

Everything goes through ``src/nmr_pipeline.py`` (verbatim mirror of the notebook) for the
official spectrum. Nothing here changes the official pipeline; extra helpers (exact
signals, stick spectra, synthetic references, criterion evaluation) are additions used for
analysis and sensitivity checks only.

Run scripts from the repository root, e.g.
    .venv/bin/python agents/agent3_spectral/task_a_matching.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import nmr_pipeline as nmr  # noqa: E402

OUT = REPO / "results"
PLOTS = OUT / "plots"
AGENT = REPO / "agents" / "agent3_spectral"
PLOTS.mkdir(parents=True, exist_ok=True)

MOLS = ("A", "B", "C")
REFS = (1, 2, 3)
DT = nmr.DEFAULT_TROTTER_TIMESTEP  # 0.0025 s
NSTEPS = nmr.DEFAULT_NUMBER_OF_TROTTERSTEPS  # 300
HZ_PER_PPM = 1.0 / nmr.hz_to_ppm_conversion()  # ~500.0 Hz per ppm at 11.7433 T
GRID_HZ = 1.0 / (NSTEPS * DT)  # 1.3333 Hz official FFT grid spacing

# Criterion constants (fixed in criterion.md before any noise sweep).
MERGE_HZ = 2.0 * GRID_HZ  # reference components closer than 2 bins are one char. peak
TOL_HZ = 2.0 * GRID_HZ  # position tolerance for a detected maximum
REF_HEIGHT_MIN = 0.05  # reference peak height (fraction of reference max)
REF_PROM_MIN = 0.02  # reference peak prominence (fraction of reference max)
DIP_PRIMARY = 0.10
DIP_LEVELS = {"any_dip": 0.0, "5pct": 0.05, "10pct": 0.10, "19pct": 0.19}
MULTIPLET_GAP_HZ = 25.0  # adjacent lines farther apart than this start a new multiplet


# ------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------
def load_official(m):
    """Official notebook signals (StatevectorEstimator, 300 steps, dt 0.0025)."""
    z = np.load(OUT / "signals" / f"molecule_{m}.npz")
    return {
        "c_xx": np.asarray(z["c_xx_raw"], float),
        "c_yx": np.asarray(z["c_yx_raw"], float),
        "shift": float(z["chemical_shift_hz"]),
        "n": int(z["number_spins"]),
        "dt": float(z["trotter_timestep"]),
        "nsteps": int(z["number_of_trottersteps"]),
    }


def spectrum(sig, noise):
    """Official spectrum (window, causal FFT, ppm axis, /max, abs) for a signal dict."""
    return nmr.spectrum_from_signals(
        sig["c_xx"], sig["c_yx"], float(noise), sig["dt"], sig["nsteps"], sig["shift"]
    )


def reference(k):
    """Reference spectrum k normalised to its maximum (copy, not in place)."""
    ppm, s = nmr.load_reference(k)
    ppm = np.asarray(ppm, float)
    s = np.asarray(s, float)
    s = s / np.max(np.abs(s))
    order = np.argsort(ppm)
    return ppm[order], s[order]


# ------------------------------------------------------------------------------------------
# Resampling of a reference onto a simulation grid
# ------------------------------------------------------------------------------------------
def ref_bin_average(ref_ppm, ref_s, grid_ppm):
    """Area-preserving resampling: mean of the (linearly interpolated) reference over each
    simulation bin [p - d/2, p + d/2]; parts of a bin outside the reference window count as 0.
    """
    d = np.median(np.diff(grid_ppm))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (ref_s[1:] + ref_s[:-1]) * np.diff(ref_ppm))])

    def F(p):
        p = np.clip(p, ref_ppm[0], ref_ppm[-1])
        i = np.clip(np.searchsorted(ref_ppm, p) - 1, 0, len(ref_ppm) - 2)
        frac = (p - ref_ppm[i]) / (ref_ppm[i + 1] - ref_ppm[i])
        si = ref_s[i] + frac * (ref_s[i + 1] - ref_s[i])
        return cum[i] + 0.5 * (ref_s[i] + si) * (p - ref_ppm[i])

    lo, hi = grid_ppm - d / 2, grid_ppm + d / 2
    return (F(hi) - F(lo)) / d


def overlap_fraction(ref_ppm, grid_ppm):
    """Fraction of simulation grid points that lie inside the reference window."""
    return float(np.mean((grid_ppm >= ref_ppm[0]) & (grid_ppm <= ref_ppm[-1])))


# ------------------------------------------------------------------------------------------
# Exact dynamics and stick spectra (dense exact diagonalisation)
# ------------------------------------------------------------------------------------------
def _ops(n):
    obs_x, obs_y = nmr.magnetization_observables(n)
    return obs_x.to_matrix(), obs_y.to_matrix()


def x_product_states(n):
    """Statevectors of the notebook's 2^N initial states (same template and parameters)."""
    from qiskit.quantum_info import Statevector

    tmpl, phi, bits_list, pmaps = nmr.enumerate_x_states(n)
    return [np.asarray(Statevector(tmpl.assign_parameters(pm)).data) for pm in pmaps], bits_list


def exact_signals(ham, n, nsteps=NSTEPS, dt=DT):
    """Exact (no Trotter error) exp_x, exp_y for the same H, initial states and observables."""
    H = ham.to_matrix()
    E, V = np.linalg.eigh(H)
    X, Y = _ops(n)
    psis, _ = x_product_states(n)
    t = np.arange(nsteps) * dt
    exp_x = np.zeros((len(psis), nsteps))
    exp_y = np.zeros((len(psis), nsteps))
    Xe = V.conj().T @ X @ V
    Ye = V.conj().T @ Y @ V
    for m, psi in enumerate(psis):
        c = V.conj().T @ psi  # coefficients in eigenbasis
        ct = c[None, :] * np.exp(-1j * np.outer(t, E))  # (nt, dim)
        exp_x[m] = np.einsum("ti,ij,tj->t", ct.conj(), Xe, ct).real
        exp_y[m] = np.einsum("ti,ij,tj->t", ct.conj(), Ye, ct).real
    return exp_x, exp_y


def signals_dict(exp_x, exp_y, shift, n, dt=DT):
    cxx, cyx = nmr.correlations(exp_x, exp_y)
    return {"c_xx": cxx, "c_yx": cyx, "shift": float(shift), "n": n, "dt": dt, "nsteps": exp_x.shape[1]}


def stick_spectrum(ham, n, shift):
    """Exact line list (ppm, weight) of the spectrum S = Re FT[C_xx] + Im FT[C_yx].

    C_ax(t) = Tr(X_tot U^dag A U) = sum_ab X_ba A_ab exp(i (E_a - E_b) t); with the FFT kernel
    exp(-i 2 pi f t) a component c exp(i w t) peaks at f = w / 2 pi with height Re(c)/Gamma for
    C_xx and Im(c)/Gamma for C_yx. Frequencies are mapped to ppm with the notebook formula.
    """
    H = ham.to_matrix()
    E, V = np.linalg.eigh(H)
    X, Y = _ops(n)
    Xe = V.conj().T @ X @ V
    Ye = V.conj().T @ Y @ V
    w = (Xe.T * Xe).real + (Xe.T * Ye).imag  # w[a, b] = Re(X_ba X_ab) + Im(X_ba Y_ab)
    f_hz = (E[:, None] - E[None, :]) / (2 * np.pi)
    conv = nmr.hz_to_ppm_conversion()
    chem_shift = 2 * shift / (2.0 * np.pi / conv) + 1e6
    ppm = f_hz * conv - chem_shift
    keep = w > 1e-9 * np.max(w)
    order = np.argsort(ppm[keep])
    return ppm[keep][order], w[keep][order]


def merge_sticks(ppm, w, tol_hz=1e-3):
    """Combine degenerate sticks (within tol_hz)."""
    out_p, out_w = [], []
    for p, ww in zip(ppm, w):
        if out_p and abs(p - out_p[-1]) * HZ_PER_PPM < tol_hz:
            tot = out_w[-1] + ww
            out_p[-1] = (out_p[-1] * out_w[-1] + p * ww) / tot
            out_w[-1] = tot
        else:
            out_p.append(p)
            out_w.append(ww)
    return np.array(out_p), np.array(out_w)


def synthetic_reference(ham, n, shift, fwhm_hz=1.0, pad_hz=30.0, step_hz=0.01):
    """High-resolution stand-in reference: exact sticks broadened by a Lorentzian.

    Only for ARTIFICIAL Hamiltonians (no provided reference). Validated against the real
    references in task_c_jcoupling.py (same characteristic peaks).
    """
    sp, sw = stick_spectrum(ham, n, shift)
    lo = sp.min() - pad_hz / HZ_PER_PPM
    hi = sp.max() + pad_hz / HZ_PER_PPM
    grid = np.arange(lo, hi, step_hz / HZ_PER_PPM)
    g = fwhm_hz / 2.0 / HZ_PER_PPM
    s = np.zeros_like(grid)
    for p, w in zip(sp, sw):
        s += w * g**2 / ((grid - p) ** 2 + g**2)
    return grid, s / s.max()


# ------------------------------------------------------------------------------------------
# Peaks and multiplets
# ------------------------------------------------------------------------------------------
def ref_peaks(ref_ppm, ref_s, height=REF_HEIGHT_MIN, prom=REF_PROM_MIN):
    pk, _ = find_peaks(ref_s, height=height, prominence=prom)
    return ref_ppm[pk], ref_s[pk]


def characteristic_peaks(ref_ppm, ref_s, merge_hz=MERGE_HZ):
    """Group reference peaks closer than merge_hz (single linkage); return groups.

    Each group: dict(ppm=height-weighted centre, height=max height, members=[ppm...]).
    """
    p, h = ref_peaks(ref_ppm, ref_s)
    groups = []
    for pi, hi in zip(p, h):
        if groups and (pi - groups[-1]["members"][-1]) * HZ_PER_PPM < merge_hz:
            groups[-1]["members"].append(float(pi))
            groups[-1]["heights"].append(float(hi))
        else:
            groups.append({"members": [float(pi)], "heights": [float(hi)]})
    for g in groups:
        hh = np.array(g["heights"])
        g["ppm"] = float(np.sum(hh * np.array(g["members"])) / hh.sum())
        g["height"] = float(hh.max())
    return groups


def multiplets(peak_ppm, gap_hz=MULTIPLET_GAP_HZ):
    """Split a sorted peak list into multiplets at gaps larger than gap_hz."""
    peak_ppm = np.sort(np.asarray(peak_ppm, float))
    if len(peak_ppm) == 0:
        return []
    out = [[peak_ppm[0]]]
    for a, b in zip(peak_ppm[:-1], peak_ppm[1:]):
        if (b - a) * HZ_PER_PPM > gap_hz:
            out.append([b])
        else:
            out[-1].append(b)
    return [
        {
            "n_lines": len(m),
            "centre_ppm": float(np.mean(m)),
            "splittings_hz": [float(x) for x in np.diff(m) * HZ_PER_PPM],
            "lines_ppm": [float(x) for x in m],
        }
        for m in out
    ]


# ------------------------------------------------------------------------------------------
# Resolution criterion (see criterion.md)
# ------------------------------------------------------------------------------------------
def evaluate(spec, ppm, groups, tol_hz=TOL_HZ):
    """Evaluate the resolution criterion on one spectrum.

    Returns dict with:
      found: bool (every characteristic peak has its own local maximum within tol, one-to-one,
             in the same order)
      reason: '' or failure reason for the detection part
      dips: list of dip depths between adjacent characteristic peaks (nan if not found)
      pairs: list of (ppm_left, ppm_right, ref_sep_hz)
      peaks: per characteristic peak (ref ppm, found ppm, error Hz, height)
    The pass decision for a dip level D is: found and min(dips) >= D (D = 0 means dip > 0).
    """
    order = np.argsort(ppm)
    ppm = ppm[order]
    spec = spec[order]
    lm, _ = find_peaks(spec)
    out = {"found": True, "reason": "", "peaks": [], "dips": [], "pairs": []}
    chosen = []
    for g in groups:
        cand = lm[np.abs(ppm[lm] - g["ppm"]) * HZ_PER_PPM <= tol_hz]
        if len(cand) == 0:
            out["found"] = False
            out["reason"] = out["reason"] or f"missing@{g['ppm']:.4f}"
            chosen.append(None)
            out["peaks"].append({"ref_ppm": g["ppm"], "sim_ppm": None, "err_hz": None, "height": None})
            continue
        c = int(cand[np.argmax(spec[cand])])
        chosen.append(c)
        out["peaks"].append(
            {
                "ref_ppm": g["ppm"],
                "sim_ppm": float(ppm[c]),
                "err_hz": float((ppm[c] - g["ppm"]) * HZ_PER_PPM),
                "height": float(spec[c]),
            }
        )
    valid = [c for c in chosen if c is not None]
    if len(set(valid)) < len(valid):
        out["found"] = False
        out["reason"] = out["reason"] or "shared_maximum"
    if any(b <= a for a, b in zip(valid[:-1], valid[1:])):
        out["found"] = False
        out["reason"] = out["reason"] or "order"
    for i in range(len(groups) - 1):
        a, b = chosen[i], chosen[i + 1]
        sep = (groups[i + 1]["ppm"] - groups[i]["ppm"]) * HZ_PER_PPM
        out["pairs"].append((groups[i]["ppm"], groups[i + 1]["ppm"], float(sep)))
        if a is None or b is None or b <= a:
            out["dips"].append(float("nan"))
            continue
        valley = spec[a : b + 1].min()
        out["dips"].append(float(1.0 - valley / min(spec[a], spec[b])))
    return out


def passes(ev, dip_min):
    if not ev["found"]:
        return False
    d = np.array(ev["dips"], float)
    if len(d) == 0:
        return True
    if np.any(np.isnan(d)):
        return False
    return bool(np.min(d) > 0.0) if dip_min <= 0.0 else bool(np.min(d) >= dip_min)


def failing_pair(ev, dip_min):
    """Pair (index) responsible for failure at dip_min (smallest dip), or detection reason."""
    if not ev["found"]:
        return ev["reason"]
    d = np.array(ev["dips"], float)
    i = int(np.nanargmin(d))
    return i


def scan(sig, groups, noises, levels=DIP_LEVELS):
    """Evaluate the criterion over a list of noise strengths; returns dict level -> bool array,
    and the list of evaluations."""
    evs = []
    res = {k: np.zeros(len(noises), bool) for k in levels}
    for j, g in enumerate(noises):
        s, p = spectrum(sig, g)
        ev = evaluate(s, p, groups)
        evs.append(ev)
        for k, d in levels.items():
            res[k][j] = passes(ev, d)
    return res, evs


def transitions(noises, ok):
    """List of (noise_before, noise_after, 'pass->fail' or 'fail->pass')."""
    out = []
    for i in range(len(ok) - 1):
        if ok[i] != ok[i + 1]:
            out.append((float(noises[i]), float(noises[i + 1]), "pass->fail" if ok[i] else "fail->pass"))
    return out


def threshold(sig, groups, dip_key, lo=0.1, hi=100.0, coarse=0.1, fine=0.01):
    """Dense scan (no bisection) at `coarse` step over [lo, hi], then every transition interval
    refined at `fine` step. Threshold = highest passing noise on the refined grid; first fail =
    next grid value above it. Returns dict."""
    d = DIP_LEVELS[dip_key] if isinstance(dip_key, str) else float(dip_key)
    lv = {"x": d}
    grid = np.round(np.arange(lo, hi + 1e-9, coarse), 6)
    res, _ = scan(sig, groups, grid, lv)
    ok = res["x"]
    tr = transitions(grid, ok)
    pts = dict(zip(grid.tolist(), ok.tolist()))
    for a, b, _ in tr:
        fg = np.round(np.arange(a, b + 1e-9, fine), 6)
        r2, _ = scan(sig, groups, fg, lv)
        pts.update(dict(zip(fg.tolist(), r2["x"].tolist())))
    xs = np.array(sorted(pts))
    oks = np.array([pts[x] for x in xs])
    if not oks.any():
        return {"max_pass": None, "first_fail": float(xs[0]), "transitions_coarse": tr,
                "transitions_fine": transitions(xs, oks), "monotonic": False, "n_points": len(xs)}
    imax = int(np.max(np.nonzero(oks)[0]))
    first_fail = float(xs[imax + 1]) if imax + 1 < len(xs) else None
    trf = transitions(xs, oks)
    return {
        "max_pass": float(xs[imax]),
        "first_fail": first_fail,
        "transitions_coarse": tr,
        "transitions_fine": trf,
        "monotonic": len(trf) == 1 and trf[0][2] == "pass->fail" and bool(oks[0]),
        "n_points": len(xs),
    }


def dump_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))

    Path(path).write_text(json.dumps(obj, indent=2, default=conv))
