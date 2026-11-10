"""Diagnostic for Task B (Trotter sensitivity): line positions of the effective Hamiltonian of
one Lie-Trotter step, H_eff = i log(U_step) / dt, compared with the exact rotating-frame H.

Run from the repository root:
    .venv/bin/python agents/agent3_spectral/trotter_effective_lines.py
"""

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.linalg import logm

from common import AGENT, DT, HZ_PER_PPM, MOLS, characteristic_peaks, dump_json, merge_sticks, nmr, reference, stick_spectrum

MATCH = {"A": 3, "B": 2, "C": 1}


def main():
    out = {}
    for m in MOLS:
        ham, n, shift = nmr.load_rotating_frame_hamiltonian(m)
        U = Operator(nmr.trotter_step_circuit(ham, DT)).data
        Heff = 1j * logm(U) / DT
        Heff = 0.5 * (Heff + Heff.conj().T)
        ham_eff = SparsePauliOp.from_operator(Heff)
        # principal branch check: eigenphases of U well inside (-pi, pi)
        ph = np.angle(np.linalg.eigvals(U))
        pe, we = merge_sticks(*stick_spectrum(ham, n, shift))
        pt, wt = merge_sticks(*stick_spectrum(ham_eff, n, shift))
        se, st = we > 0.02 * we.max(), wt > 0.02 * wt.max()
        rp, rs = reference(MATCH[m])
        cen = np.array([g["ppm"] for g in characteristic_peaks(rp, rs)])
        # nearest strong lines to each characteristic peak
        ex = [float(pe[se][np.argmin(abs(pe[se] - c))]) for c in cen]
        tr = [float(pt[st][np.argmin(abs(pt[st] - c))]) for c in cen]
        out[m] = {
            "max_abs_eigenphase_rad": float(np.max(np.abs(ph))),
            "exact_lines_ppm": pe[se].tolist(), "exact_weights": (we[se] / we.max()).tolist(),
            "trotter_lines_ppm": pt[st].tolist(), "trotter_weights": (wt[st] / wt.max()).tolist(),
            "shift_of_nearest_line_hz": [(b - a) * HZ_PER_PPM for a, b in zip(ex, tr)],
            "char_sep_exact_hz": (np.diff(ex) * HZ_PER_PPM).tolist(),
            "char_sep_trotter_hz": (np.diff(tr) * HZ_PER_PPM).tolist(),
        }
        print(m, "max|phase|", round(out[m]["max_abs_eigenphase_rad"], 3))
        print("   line shift Trotter-exact (Hz):", np.round(out[m]["shift_of_nearest_line_hz"], 3).tolist())
        print("   adjacent separations exact  :", np.round(out[m]["char_sep_exact_hz"], 3).tolist())
        print("   adjacent separations Trotter:", np.round(out[m]["char_sep_trotter_hz"], 3).tolist())
        print("   weights exact  :", np.round(out[m]["exact_weights"], 3).tolist())
        print("   weights Trotter:", np.round(out[m]["trotter_weights"], 3).tolist())
    dump_json(out, AGENT / "trotter_effective_lines.json")


if __name__ == "__main__":
    main()
