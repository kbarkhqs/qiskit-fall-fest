"""T6: exact C_xx, C_yx at the notebook's 300 time points, NumPy first-order Lie-Trotter emulation
(file term order), saved npz files, and the first-order Trotter error (signals and line positions)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.linalg import expm, logm
from scipy.signal import find_peaks
from common import (HERE, HZ_TO_PPM, DT, NSTEPS, dense_h, load_terms, rotating_frame, physical_params, line_list,
                    exact_signals, trotter_step, state_sum_signals, record, cmd)

T_TOTAL = NSTEPS * DT
WEAK = 1e-3


def trotter_lines(terms, n, dt, order=None):
    """Lines of the Trotterised dynamics from the quasi-energies: H_eff = i log(U_step)/dt."""
    U = trotter_step(terms, n, dt, order)
    Heff = 1j * logm(U) / dt
    assert np.abs(Heff - Heff.conj().T).max() < 1e-6 * np.abs(Heff).max()
    Heff = (Heff + Heff.conj().T) / 2
    assert np.abs(expm(-1j * Heff * dt) - U).max() < 1e-9
    hz, inten, spin = line_list(Heff, n)
    hz = (hz + 0.5 / dt) % (1 / dt) - 0.5 / dt  # quasi-energy differences are defined modulo 1/dt
    o = np.argsort(hz)
    return hz[o], inten[o], spin[o]


def fine_spectrum(cxx, cyx, dt, gam, f):
    """Notebook quantity Re F[C_xx] + Im F[C_yx] (first sample halved) on an arbitrary fine frequency grid."""
    t = np.arange(len(cxx)) * dt
    w = np.exp(-gam * t); w[0] *= 0.5
    out = np.empty(len(f))
    for s in range(0, len(f), 4000):
        ph = np.exp(-2j * np.pi * np.outer(f[s:s + 4000], t))
        out[s:s + 4000] = (ph @ (w * cxx)).real + (ph @ (w * cyx)).imag
    return out


def peak_pos(f, y):
    pk, _ = find_peaks(y, prominence=0.02 * y.max())
    return np.array([f[k] - 0.5 * (y[k + 1] - y[k - 1]) / (y[k + 1] - 2 * y[k] + y[k - 1]) * (f[1] - f[0]) for k in pk])


res = {}
for mol in "ABC":
    terms = load_terms(mol); n, h, d, J = physical_params(terms); rf, mean_h = rotating_frame(terms)
    H = dense_h(rf, n)
    times = np.arange(NSTEPS) * DT
    cxx, cyx = exact_signals(H, n, times)                                   # trace formula
    cxx_s, cyx_s = state_sum_signals(expm(-1j * H * DT), n, NSTEPS)         # notebook recipe, exact propagator
    U1 = trotter_step(rf, n, DT)
    txx, tyx = state_sum_signals(U1, n, NSTEPS)                             # notebook recipe, Lie-Trotter
    hz, inten, spin = line_list(H, n)
    pairs = sorted(J)
    np.savez(HERE / f"exact_signals_molecule_{mol}.npz", times=times, c_xx=cxx, c_yx=cyx, trotter_c_xx=txx, trotter_c_yx=tyx,
             delta_ppm=d, J_hz=np.array([J[p] for p in pairs]), J_pairs=np.array(pairs), line_ppm=d.mean() + hz * HZ_TO_PPM,
             line_intensity=inten, line_hz=hz, line_spin=spin, mean_delta_ppm=d.mean(), mean_h_rad_s=mean_h,
             dt=DT, n_steps=NSTEPS, term_order=np.array([k for k, _ in rf]))
    r = {"C_xx(0)": float(cxx[0]), "C_yx(0)": float(cyx[0]),
         "max|trace formula - state-sum recipe| (xx, yx)": [float(np.abs(cxx - cxx_s).max()), float(np.abs(cyx - cyx_s).max())],
         "[U_step, Z_tot] norm": float(np.abs(U1 @ sum(np.kron(np.eye(1), 1) * 0 for _ in [0]) if False else 0.0))}
    del r["[U_step, Z_tot] norm"]

    # --- signal error versus dt at fixed total time ---
    sig = {}
    for div in (1, 2, 4):
        dt = DT / div; ns = NSTEPS * div
        ex = exact_signals(H, n, np.arange(ns) * dt)
        tr = state_sum_signals(trotter_step(rf, n, dt), n, ns)
        sig[f"dt/{div}"] = {"dt": dt, "n_steps": ns,
                            "max|dC_xx|/C_xx(0)": float(np.abs(tr[0] - ex[0]).max() / cxx[0]),
                            "max|dC_yx|/C_xx(0)": float(np.abs(tr[1] - ex[1]).max() / cxx[0]),
                            "rms|dC|/C_xx(0)": float(np.sqrt(np.mean((tr[0] - ex[0]) ** 2 + (tr[1] - ex[1]) ** 2)) / cxx[0])}
    e = [max(sig[f"dt/{k}"]["max|dC_xx|/C_xx(0)"], sig[f"dt/{k}"]["max|dC_yx|/C_xx(0)"]) for k in (1, 2, 4)]
    sig["order (log2 of error ratio) dt->dt/2, dt/2->dt/4"] = [float(np.log2(e[0] / e[1])), float(np.log2(e[1] / e[2]))]
    rev = state_sum_signals(trotter_step(rf, n, DT, order=list(range(len(rf)))[::-1]), n, NSTEPS)
    sig["reversed term order, dt: max|dC|/C_xx(0)"] = float(max(np.abs(rev[0] - cxx).max(), np.abs(rev[1] - cyx).max()) / cxx[0])
    r["signal_error"] = sig

    # --- line positions from quasi-energies ---
    strong = inten > WEAK
    f_ex, a_ex, s_ex = hz[strong], inten[strong], spin[strong]
    lines = {}
    for div in (1, 2, 4):
        th, ta, ts = trotter_lines(rf, n, DT / div)
        m = ta > WEAK
        assert m.sum() == strong.sum(), (m.sum(), strong.sum())
        shift = th[m] - f_ex
        assert np.all(ts[m] == s_ex), "line pairing by order failed"
        lines[f"dt/{div}"] = {"shift_hz": shift.round(5).tolist(), "max|shift| hz": float(np.abs(shift).max()),
                              "rms shift hz": float(np.sqrt(np.mean(shift**2))),
                              "max|intensity change|": float(np.abs(ta[m] - a_ex).max())}
    s = [lines[f"dt/{k}"]["max|shift| hz"] for k in (1, 2, 4)]
    lines["order (log2 of max-shift ratio) dt->dt/2, dt/2->dt/4"] = [float(np.log2(s[0] / s[1])), float(np.log2(s[1] / s[2]))]
    lines["exact line hz"] = f_ex.round(4).tolist()
    lines["line spin"] = s_ex.tolist()
    sh = np.array(lines["dt/1"]["shift_hz"])
    k = int(np.argmax(np.abs(sh)))
    lines["line that moves most at dt"] = {"exact_hz": float(f_ex[k]), "ppm": float(d.mean() + f_ex[k] * HZ_TO_PPM), "spin": int(s_ex[k]), "shift_hz": float(sh[k])}
    lines["mean shift per spin multiplet at dt (Hz)"] = {int(i): float(sh[s_ex == i].mean()) for i in range(n)}
    # splittings (neighbour spacings inside multiplets) under Trotter at dt
    th, ta, ts = trotter_lines(rf, n, DT); m = ta > WEAK
    lines["splitting change at dt (Hz) per multiplet"] = {int(i): (np.diff(th[m][ts[m] == i]) - np.diff(f_ex[s_ex == i])).round(5).tolist() for i in range(n)}
    r["line_shift"] = lines

    # --- cross-check on the spectra themselves (default 300 points, Gamma = 5, fine frequency grid) ---
    f = np.arange(-199.0, 199.0, 0.01)
    p_ex = peak_pos(f, fine_spectrum(cxx, cyx, DT, 5.0, f))
    p_tr = peak_pos(f, fine_spectrum(txx, tyx, DT, 5.0, f))
    if len(p_ex) == len(p_tr):
        dpk = p_tr - p_ex
        r["spectrum_peak_shift (Gamma=5, 300 pts, fine grid)"] = {"n_peaks": len(p_ex), "exact_peak_hz": p_ex.round(3).tolist(),
                                                                 "shift_hz": dpk.round(4).tolist(), "max|shift| hz": float(np.abs(dpk).max())}
    else:
        r["spectrum_peak_shift (Gamma=5, 300 pts, fine grid)"] = {"n_peaks_exact": len(p_ex), "n_peaks_trotter": len(p_tr)}
    res[mol] = r
    print(f"\n=== {mol}")
    for k, v in r.items():
        print(" ", k, ":", v)

record({
    "id": "T6_exact_signals_and_trotter_error",
    "question": "How large is the first-order Lie-Trotter error at dt = 0.0025 s in the time signals and in the line positions, and how does it scale with dt?",
    "hypotheses": ["H1: signal error is first order in dt (halves when dt halves)",
                   "H2: line-position error is second order in dt (quarters when dt halves), because the O(dt) term of the effective Hamiltonian has zero diagonal in the real eigenbasis",
                   "H3: line shifts at dt = 0.0025 s are below the 1.33 Hz grid spacing"],
    "independent_variable": "Trotter step dt in {0.0025, 0.00125, 0.000625} s at fixed total time 0.75 s",
    "dependent_variable": "max |C_trotter - C_exact| / C_xx(0); shift of each allowed line (quasi-energy differences of the Trotter step)",
    "fixed_parameters": {"term_order": "file order: all Z, then XX, YY, ZZ per pair (0,1),(0,2),(1,2)", "total_time_s": T_TOTAL, "weak_line_threshold": WEAK},
    "reference": "exact diagonalisation of the rotating-frame Hamiltonian",
    "result": res,
    "conclusion": "; ".join(
        f"{m}: signal error {max(res[m]['signal_error']['dt/1']['max|dC_xx|/C_xx(0)'], res[m]['signal_error']['dt/1']['max|dC_yx|/C_xx(0)']):.4f} of C_xx(0) "
        f"(order {res[m]['signal_error']['order (log2 of error ratio) dt->dt/2, dt/2->dt/4'][0]:.2f}), max line shift "
        f"{res[m]['line_shift']['dt/1']['max|shift| hz']:.4f} Hz (order {res[m]['line_shift']['order (log2 of max-shift ratio) dt->dt/2, dt/2->dt/4'][0]:.2f})" for m in res),
    "limitations": ("NumPy emulation of the term-by-term product; that Qiskit's LieTrotter synthesis uses the same order (first SparsePauliOp term first) "
                    "is assumed, NOT RUN here (Qiskit excluded by the independence rule). struqture 2.6.0 keeps the file order (checked). "
                    "Reversed order is reported as a sensitivity number."),
    "script": "agents/agent1_physics/t6_signals_trotter.py",
    "command": cmd("t6_signals_trotter.py"),
})
