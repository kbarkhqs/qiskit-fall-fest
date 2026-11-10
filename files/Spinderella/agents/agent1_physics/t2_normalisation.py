"""T2: Pauli operators without 1/2^N versus spin-1/2 operators with 1/2^N."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.linalg import expm
from common import (X, Y, HZ_TO_PPM, DT, NSTEPS, total, dense_h, load_terms, rotating_frame, n_spins,
                    state_sum_signals, record, cmd)
from notebook_verbatim import time_signal_to_spectrum

times = np.arange(NSTEPS) * DT
win = np.exp(-5.0 * times)
res = {}
ok = True
for mol in "ABC":
    terms = load_terms(mol); n = n_spins(terms); rf, mean_h = rotating_frame(terms)
    H = dense_h(rf, n)
    # (i) notebook recipe, Pauli observables and Pauli weights, exact propagator
    cxx_p, cyx_p = state_sum_signals(expm(-1j * H * DT), n, NSTEPS)
    # (ii) textbook definition with spin-1/2 operators and the 1/2^N of the infinite-temperature state
    ix, iy = total(X, n) / 2, total(Y, n) / 2
    E, V = np.linalg.eigh(H)
    cxx_s, cyx_s = np.zeros(NSTEPS), np.zeros(NSTEPS)
    for k, t in enumerate(times):
        U = V @ np.diag(np.exp(-1j * E * t)) @ V.conj().T
        cxx_s[k] = np.trace(U.conj().T @ ix @ U @ ix).real / 2**n
        cyx_s[k] = np.trace(U.conj().T @ iy @ U @ ix).real / 2**n
    factor = 4 * 2**n
    sp_p, ppm = time_signal_to_spectrum(cxx_p * win, cyx_p * win, DT, NSTEPS, HZ_TO_PPM, mean_h)
    sp_s, _ = time_signal_to_spectrum(cxx_s * win, cyx_s * win, DT, NSTEPS, HZ_TO_PPM, mean_h)
    r = {"C_xx(0) Pauli": float(cxx_p[0]), "C_xx(0) spin with 1/2^N": float(cxx_s[0]), "expected ratio 4*2^N": factor,
         "max|C_xx^Pauli - 4*2^N*C_xx^spin|": float(np.abs(cxx_p - factor * cxx_s).max()),
         "max|C_yx^Pauli - 4*2^N*C_yx^spin|": float(np.abs(cyx_p - factor * cyx_s).max()),
         "max|normalised spectrum difference|": float(np.abs(sp_p - sp_s).max())}
    ok &= r["max|normalised spectrum difference|"] < 1e-12 and r["max|C_xx^Pauli - 4*2^N*C_xx^spin|"] < 1e-9
    res[mol] = r
    print(mol, r)
record({
    "id": "T2_normalisation",
    "question": "Does using Pauli operators instead of spin-1/2 operators, and omitting 1/2^N, change the max-normalised spectrum?",
    "hypotheses": ["H1: no - both signals differ by the constant factor 4*2^N, removed by the division by the maximum",
                   "H2: yes - the spectrum shape changes"],
    "independent_variable": "operator normalisation (Pauli, no 1/2^N) vs (spin-1/2, with 1/2^N)",
    "dependent_variable": "max abs difference of the two max-normalised spectra (verbatim notebook function, Gamma = 5)",
    "fixed_parameters": {"dt": DT, "n_steps": NSTEPS, "gamma": 5.0, "evolution": "exact"},
    "reference": "Tr(I_a(t) I_x)/2^N with I = sigma/2",
    "result": res,
    "conclusion": "H1 confirmed for A, B, C: constant factor 4*2^N (16 for N=2, 32 for N=3); normalised spectra identical to rounding." if ok else "CHECK FAILED",
    "limitations": "Only absolute intensities are lost; relative intensities between molecules cannot be compared after the division by the maximum.",
    "script": "agents/agent1_physics/t2_normalisation.py",
    "command": cmd("t2_normalisation.py"),
})
print("T2 OK:", ok)
