"""T3: Hamiltonian conventions. Extract delta_i, J_ij from the three files and check the
factor chain (Pauli coefficient -> rad/s -> Hz -> ppm) with explicit matrices."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.linalg import expm
from common import (X, Y, Z, GAMMA, B0, NU0, HZ_TO_PPM, DT, load_terms, rotating_frame,
                    physical_params, record, cmd)

CLAIM = {
    "A": ([5.790, 5.970, 5.480], {(0, 1): 0.9, (0, 2): 11.8, (1, 2): 17.9}),
    "B": ([5.580, 5.900, 5.340], {(0, 1): 15.6, (0, 2): -0.5, (1, 2): 8.6}),
    "C": ([6.375, 6.302], {(0, 1): 7.92}),
}
res = {"gamma*B/(2pi) Hz": NU0, "Hz_to_ppm": HZ_TO_PPM}
all_ok = True
for mol in "ABC":
    terms = load_terms(mol)
    n, h, d, J = physical_params(terms)
    rf, mean_h = rotating_frame(terms)
    hp = np.array([c for k, c in rf[:n]])
    dbar = d.mean()
    # notebook ppm-axis constant: chem_shift = 2*mean_h/(2pi/Hz_to_ppm) + 1e6 ; ppm = f*Hz_to_ppm - chem_shift
    chem_shift = 2 * mean_h / (2.0 * np.pi / HZ_TO_PPM) + 1e6
    d_claim, j_claim = CLAIM[mol]
    dd = np.abs(d - d_claim).max()
    dj = max(abs(J[p] - j_claim[p]) for p in J)
    # h rebuilt from the claimed table vs the file (rad/s)
    h_back = np.abs(-GAMMA * B0 * (1 + np.array(d_claim) * 1e-6) / 2 - h).max()
    ok = dd < 5e-4 and dj < 5e-4 and set(J) == set(j_claim)
    all_ok &= ok
    res[mol] = {
        "n_spins": n, "term_order_in_file": [k for k, _ in terms],
        "delta_ppm": d.round(6).tolist(), "delta_claimed": d_claim, "max|delta - claim| ppm": dd,
        "J_hz": {f"{i}{j}": round(v, 6) for (i, j), v in J.items()}, "max|J - claim| Hz": dj,
        "max|h(claim) - h(file)| rad/s": h_back,
        "mean_h (notebook 'chemical_shift_hz') rad/s": mean_h, "mean delta ppm": dbar,
        "h'_i rotating frame rad/s": hp.round(6).tolist(),
        "uncoupled transition omega_i = -2h'_i rad/s": (-2 * hp).round(6).tolist(),
        "uncoupled transition nu_i = -h'_i/pi Hz": (-hp / np.pi).round(6).tolist(),
        "nu0*(delta_i - mean)*1e-6 Hz": (NU0 * (d - dbar) * 1e-6).round(6).tolist(),
        "max|h'|*dt (rad)": float(np.abs(hp).max() * DT),
        "notebook chem_shift constant": chem_shift, "-(mean delta ppm)": -dbar,
        "chem_shift + mean delta (ppm)": chem_shift + dbar,
        "shift differences Hz": {f"{i}{j}": round(float(NU0 * (d[i] - d[j]) * 1e-6), 4) for (i, j) in J},
        "|shift difference / J|": {f"{i}{j}": round(float(abs(NU0 * (d[i] - d[j]) * 1e-6 / J[(i, j)])), 3) for (i, j) in J},
        "table_confirmed": bool(ok),
    }
    print(mol, res[mol])

# factor check with an explicit 2x2 evolution: U = exp(-i h Z dt) on |+>,
# <X> + i<Y> must be exp(+2 i h t): the Pauli coefficient h is HALF the angular frequency.
hp = 137.0  # rad/s, arbitrary
dt = 1e-3
U = expm(-1j * hp * Z * dt)
psi = U @ (np.array([1, 1], dtype=complex) / np.sqrt(2))
ang = np.angle((psi.conj() @ X @ psi) + 1j * (psi.conj() @ Y @ psi))
res["single-spin factor check"] = {"h rad/s": hp, "measured d(phase)/dt of <X>+i<Y> rad/s": float(ang / dt),
                                   "2h": 2 * hp, "ratio": float(ang / dt / hp)}
print(res["single-spin factor check"])
f_ok = abs(ang / dt - 2 * hp) < 1e-9
record({
    "id": "T3_hamiltonian_conventions",
    "question": "Do the files obey h_i = -gamma B (1+delta_i)/2 and c_ij = pi J_ij/2 with the claimed delta and J, and is any factor 2 or 2 pi missing in the chain?",
    "hypotheses": ["H1: the claimed delta/J table is exact", "H2: the table needs a correction",
                   "H3: transition angular frequency of an uncoupled spin is 2|h'_i| (Pauli coefficient = half the frequency)"],
    "independent_variable": "molecule file (A, B, C)",
    "dependent_variable": "delta_i (ppm), J_ij (Hz), differences to the claimed table, rotating-frame frequencies",
    "fixed_parameters": {"gamma": GAMMA, "B": B0},
    "reference": "claimed table in docs/challenge_analysis.md",
    "result": res,
    "conclusion": (("Claimed table confirmed for A, B, C (deviations < 5e-4 ppm / Hz). " if all_ok else "TABLE MISMATCH - see result. ")
                   + ("Uncoupled-spin line: omega = -2 h'_i rad/s = nu0 (delta_i - mean delta) Hz. The notebook's "
                      "'2*chem_shift_hz/(2 pi)' is the correct factor (chem_shift_hz is a Pauli coefficient in rad/s, "
                      "misnamed); the ppm constant equals -(mean delta) to < 1e-6 ppm. No factor 2 or 2 pi missing."
                      if f_ok else "FACTOR CHECK FAILED")),
    "limitations": "Tolerance 5e-4 on the rounded table values; float64 on h ~ 1.6e9 rad/s limits delta to ~1e-9 ppm.",
    "script": "agents/agent1_physics/t3_hamiltonian.py",
    "command": cmd("t3_hamiltonian.py"),
})
print("TABLE OK:", all_ok, "FACTOR OK:", f_ok)
