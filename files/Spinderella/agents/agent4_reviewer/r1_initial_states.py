"""C1: initial states, weights, normalisation. Run: .venv/bin/python agents/agent4_reviewer/r1_initial_states.py

Tests
 1. The notebook template (H then RZ(pi b_i)) gives X eigenstates; X_tot eigenvalue N - 2w; Gram = I.
 2. Weighted sums over the 2^N X states with 9 weightings vs the exact trace correlation
    Tr(A(t) X_tot) under the SAME (official Lie-Trotter) dynamics: correlation-level error after
    the best scalar fit, and the final notebook spectrum (window 5 s^-1, /max, abs) difference.
 3. Sum rules on the official signal files; spin-vs-Pauli normalisation.
"""
import numpy as np
from qiskit.quantum_info import Statevector

from r_common import (nmr, parse_json_terms, rotating_frame, trotter_step, trace_correlations,
                      state_signals, xstates, tot, op, load_official_signals, official_spectrum, save)

L = 300
res = {}
for m in "ABC":
    terms, n = parse_json_terms(m)
    rt, hbar = rotating_frame(terms, n)
    U = trotter_step(rt, n)
    out = {"n": n}

    # --- 1. notebook template states (Qiskit little-endian: qubit i is tensor factor n-1-i) ---
    tmpl, phi, bits_list, pmaps = nmr.enumerate_x_states(n)
    Xq = [op(n, {n - 1 - i: "X"}) for i in range(n)]  # X on qiskit qubit i
    Xt = tot(n, "X")
    vecs, maxres, eig_ok = [], 0.0, True
    for bits, pm in zip(bits_list, pmaps):
        v = np.asarray(Statevector(tmpl.assign_parameters(pm)).data)
        vecs.append(v)
        for i in range(n):
            r = np.linalg.norm(Xq[i] @ v - (1 - 2 * bits[i]) * v)
            maxres = max(maxres, r)
        mval = n - 2 * sum(bits)
        maxres = max(maxres, np.linalg.norm(Xt @ v - mval * v))
    G = np.array([[np.vdot(a, b) for b in vecs] for a in vecs])
    out["template_eigen_residual_max"] = maxres
    out["gram_minus_identity_max"] = float(np.abs(G - np.eye(len(vecs))).max())
    out["n_states"] = len(vecs)

    # --- 2. weightings ---
    xs = xstates(n)
    bits = np.array([b for b, _ in xs])
    psis = [v for _, v in xs]
    w = bits.sum(1)
    mvals = n - 2 * w
    ex, ey = state_signals(U, n, L, psis)
    cxx_tr, cyx_tr = trace_correlations(U, n, L)  # exact trace, same Trotter dynamics
    integer_label = np.array([int("".join(map(str, b)), 2) for b in bits])
    weightings = {
        "m_s = N - 2w (claimed)": mvals,
        "equal weights 1": np.ones(len(xs)),
        "only all-plus state": (w == 0).astype(float),
        "flipped sign -(N - 2w)": -mvals,
        "Hamming weight w (number of |->)": w.astype(float),
        "number of |+> = N - w (all >= 0)": (n - w).astype(float),
        "m_s^3": mvals.astype(float) ** 3,
        "m_s + m_s^2 (even part added)": mvals + mvals.astype(float) ** 2,
        "integer value of the bit label": integer_label.astype(float),
        "spin 0 only: (-1)^b0": (1 - 2 * bits[:, 0]).astype(float),
    }
    s_true, _ = official_spectrum(cxx_tr, cyx_tr, 5.0, hbar)
    wres = {}
    for name, wt in weightings.items():
        cxx = wt @ ex
        cyx = wt @ ey
        ref = np.concatenate([cxx_tr, cyx_tr])
        got = np.concatenate([cxx, cyx])
        lam = float(got @ ref / (ref @ ref))
        rel = float(np.max(np.abs(got - lam * ref)) / np.max(np.abs(ref)))
        try:
            s, _ = official_spectrum(cxx, cyx, 5.0, hbar)
            sd = float(np.max(np.abs(s - s_true)))
            note = ""
        except ValueError as e:
            sd, note = None, "pipeline ValueError (zero signal): " + str(e)[:60]
        wres[name] = {"weights": wt.tolist(), "best_scale": lam,
                      "corr_rel_residual_after_scale": rel,
                      "spectrum_maxabs_diff_vs_trace": sd, "note": note,
                      "max_abs_signal": float(np.max(np.abs(got)))}
    out["weightings"] = wres

    # --- 3. sum rules on the official files, spin normalisation ---
    sig = load_official_signals(m)
    out["official_cxx0"] = float(sig["cxx"][0])
    out["expected_N2N"] = n * 2**n
    out["official_cyx0"] = float(sig["cyx"][0])
    ob = sig["bits"]
    out["official_mx0_minus_N_minus_2w_max"] = float(np.max(np.abs(sig["exp_x"][:, 0] - (n - 2 * ob.sum(1)))))
    out["official_vs_trace_corr_max"] = float(max(np.abs(sig["cxx"] - cxx_tr).max(), np.abs(sig["cyx"] - cyx_tr).max()))
    # spin operators (sigma/2), spin weights (m_s/2), 1/2^N
    cxx_s = (mvals / 2) @ (ex / 2) / 2**n
    cyx_s = (mvals / 2) @ (ey / 2) / 2**n
    out["pauli_over_spin_ratio"] = float(cxx_tr[0] / cxx_s[0])
    worst = 0.0
    for g in (0.1, 5.0, 30.0):
        a, _ = official_spectrum(cxx_tr, cyx_tr, g, hbar)
        b, _ = official_spectrum(cxx_s, cyx_s, g, hbar)
        worst = max(worst, float(np.max(np.abs(a - b))))
    out["spin_vs_pauli_spectrum_maxdiff"] = worst
    res[m] = out
    print(m, "template residual %.1e gram %.1e | Cxx(0)=%.12g (N2^N=%d) Cyx(0)=%.1e | mx0 err %.1e | official vs trace %.1e | Pauli/spin %.3f, spectra diff %.1e"
          % (maxres, out["gram_minus_identity_max"], out["official_cxx0"], n * 2**n, out["official_cyx0"],
             out["official_mx0_minus_N_minus_2w_max"], out["official_vs_trace_corr_max"],
             out["pauli_over_spin_ratio"], worst))
    for name, r in wres.items():
        print("   %-36s scale %8.3f  corr residual %.2e  spectrum diff %s %s"
              % (name, r["best_scale"], r["corr_rel_residual_after_scale"],
                 "None" if r["spectrum_maxabs_diff_vs_trace"] is None else "%.2e" % r["spectrum_maxabs_diff_vs_trace"],
                 r["note"][:40]))
save("r1_initial_states.json", res)
