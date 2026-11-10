"""Audit part 1: initial states + Hamiltonian (item 5), official signals reproduced independently,
matching (item 6), criterion unit tests (item 1), thresholds on a full 0.01 grid (item 2)."""
import json
from itertools import permutations
from pathlib import Path

import numpy as np

from audit_lib import (DT, HZ_PER_PPM, L, ROOT, char_peaks, evaluate, exact_signals, full_scan,
                       load_reference, official_spectrum, pauli_matrix, ppm_offset, rotating_frame,
                       shifts_and_couplings, signals_state_sum, signed_spectrum, sum_ops, trotter_unitary,
                       x_basis_states, load_terms)

OUT = Path(__file__).resolve().parent
res = {}

# ------------------------------------------------------------ item 5: states and Hamiltonian
import sys
sys.path.insert(0, str(ROOT))
import src.nmr_pipeline as P  # only for comparison with the official objects
from qiskit.quantum_info import Statevector

item5 = {}
for m in "ABC":
    n, rf, H, mean_z = rotating_frame(m)
    bits, S = x_basis_states(n)
    X, Y = sum_ops(n)
    gram = S.conj() @ S.T
    eig = np.array([np.real(v.conj() @ X @ v) for v in S])
    resid = max(np.linalg.norm(X @ v - (n - 2 * sum(b)) * v) for v, b in zip(S, bits))
    # Qiskit template states (notebook construction) vs independent kron states
    tpl, phi, bl, maps = P.enumerate_x_states(n)
    overl = [abs(np.vdot(Statevector(tpl.assign_parameters(pm)).data, S[i])) for i, pm in enumerate(maps)]
    Hq, nq, cq = P.load_rotating_frame_hamiltonian(f"molecule {m}")
    Hfull = sum(c * pauli_matrix(ops, n) for ops, c in load_terms(m)[1])
    sumz = sum(pauli_matrix({i: "Z"}, n) for i in range(n))
    # weighted state sum == trace formula, and alternative weightings
    t_test = np.array([0.0, 0.0137, 0.2, 0.7])
    E, V = np.linalg.eigh(H)
    tr_err, alt = 0.0, {}
    m_w = np.array([n - 2 * sum(b) for b in bits], float)
    weightings = {"N-2w": m_w, "N-w": n - np.array([sum(b) for b in bits], float),
                  "-(N-2w)": -m_w, "equal(1)": np.ones(len(bits)), "all-plus only": np.eye(len(bits))[0] * n}
    for t in t_test:
        Ut = V @ np.diag(np.exp(-1j * E * t)) @ V.conj().T
        At = Ut.conj().T @ X @ Ut
        tr = np.real(np.trace(At @ X))
        ev = np.array([np.real(v.conj() @ At @ v) for v in S])
        tr_err = max(tr_err, abs(m_w @ ev - tr))
        for k, wv in weightings.items():
            alt.setdefault(k, 0.0)
            alt[k] = max(alt[k], abs(wv @ ev - tr))
    delta, J = shifts_and_couplings(m)
    item5[m] = {
        "n_states": len(bits), "expected": 2**n,
        "max|gram - I|": float(np.abs(gram - np.eye(2**n)).max()),
        "max eigen-residual |X_tot v - (N-2w) v|": float(resid),
        "eigenvalues_of_sum_X": sorted(set(int(round(e)) for e in eig)),
        "min |<qiskit template state|kron state>|": float(min(overl)),
        "hermiticity max|H - H^dagger| (Qiskit op)": float(np.abs(Hq.to_matrix() - Hq.to_matrix().conj().T).max()),
        "max|H_independent - H_qiskit_conversion| rad/s": float(np.abs(H - Hq.to_matrix()).max()),
        "rotating frame: ||[H_lab, sum Z]||": float(np.abs(Hfull @ sumz - sumz @ Hfull).max()),
        "rotating frame: max|eig(H_lab) - eig(H_rf + mean*sumZ)| rad/s": float(np.abs(np.sort(np.linalg.eigvalsh(Hfull)) - np.sort(np.linalg.eigvalsh(H + mean_z * sumz))).max()),
        "max|sum_n m_n<n|X(t)|n> - Tr(X(t) X)| over 4 times": float(tr_err),
        "alternative weightings, max deviation from the trace": {k: float(v) for k, v in alt.items()},
        "delta_ppm": {k: round(v, 4) for k, v in delta.items()}, "J_hz": {k: round(v, 4) for k, v in J.items()},
    }
res["item5_states_hamiltonian"] = item5

# ------------------------------------------------------------ official signals reproduced
sig = {}
item_sig = {}
for m in "ABC":
    n, rf, H, mean_z = rotating_frame(m)
    U = trotter_unitary(rf, n, DT)
    cxx, cyx = signals_state_sum(U, n, L)
    off = np.load(ROOT / f"results/signals/molecule_{m}.npz")
    item_sig[m] = {"max|c_xx(independent Trotter) - c_xx(official Qiskit)|": float(np.abs(cxx - off["c_xx_raw"]).max()),
                   "max|c_yx diff|": float(np.abs(cyx - off["c_yx_raw"]).max()),
                   "C_xx(0)": float(cxx[0]), "C_yx(0)": float(cyx[0])}
    sig[m] = (cxx, cyx, mean_z)
res["official_signals_reproduced"] = item_sig

# ------------------------------------------------------------ item 6: matching (sim -> reference grid)
refs = {k: load_reference(k) for k in (1, 2, 3)}


def corrected_axis(ppm_official, steps, dt):
    j = np.arange(steps)
    f_true = (2 * j - (steps - 1)) / ((2 * steps - 1) * dt)
    f_label = np.fft.fftshift(np.fft.fftfreq(steps, d=dt))
    return ppm_official + (f_true - f_label) / HZ_PER_PPM


def weighted_cos(a, b, w):
    na, nb = np.sqrt(np.sum(w * a * a)), np.sqrt(np.sum(w * b * b))
    return float(np.sum(w * a * b) / (na * nb)) if na > 0 and nb > 0 else 0.0


match_rows, winners = [], {}
for g in (1.0, 2.0, 5.0, 10.0):
    for axis in ("official", "corrected"):
        score, peakscore = {}, {}
        for m in "ABC":
            cxx, cyx, mz = sig[m]
            s, p = official_spectrum(cxx, cyx, g, DT, L, mz)
            if axis == "corrected":
                p = corrected_axis(p, L, DT)
            mx, _ = __import__("scipy.signal", fromlist=["find_peaks"]).find_peaks(s, prominence=0.02)
            for k, (rp, rs) in refs.items():
                sim_on_ref = np.interp(rp, p, s, left=0.0, right=0.0)
                w = np.gradient(rp)
                score[m, k] = weighted_cos(sim_on_ref, rs, w)
                cen = char_peaks(rp, rs, merge_hz=2 * (1 / (L * DT)))
                d = [np.min(np.abs(p[mx] - c)) * HZ_PER_PPM if len(mx) else 1e9 for c in cen]
                peakscore[m, k] = float(np.mean(np.array(d) <= 2 * (1 / (L * DT))))
        for metric, sc in (("cosine_on_reference_grid", score), ("peak_fraction", peakscore)):
            tot = sorted(((sum(sc[m, k] for m, k in zip("ABC", perm)), perm) for perm in permutations((1, 2, 3))), reverse=True)
            winners[(g, axis, metric)] = tot[0][1]
            match_rows.append({"noise": g, "axis": axis, "metric": metric,
                               "best": {m: f"spectrum_{k}" for m, k in zip("ABC", tot[0][1])},
                               "best_total": round(tot[0][0], 4), "second_total": round(tot[1][0], 4),
                               "matrix": {f"{m}-spectrum_{k}": round(v, 4) for (m, k), v in sc.items()}})
res["item6_matching"] = {"rows": match_rows,
                         "all_choices_agree": len(set(winners.values())) == 1,
                         "winner": {m: f"spectrum_{k}" for m, k in zip("ABC", list(winners.values())[0])},
                         "n_choices": len(winners)}
assign = {m: int(res["item6_matching"]["winner"][m][-1]) for m in "ABC"}

# ------------------------------------------------------------ item 1: criterion unit tests on synthetic spectra
ppm_t = np.linspace(5.0, 5.1, 2001)
hzax = (ppm_t - 5.05) * HZ_PER_PPM


def lor(x0, fwhm, h=1.0):
    return h * (fwhm / 2) ** 2 / ((hzax - x0) ** 2 + (fwhm / 2) ** 2)


unit = {}
s2 = lor(-4, 4.0) + lor(4, 4.0, 0.7)
f, d, _ = evaluate(s2, ppm_t, np.array([5.05 - 4 / HZ_PER_PPM, 5.05 + 4 / HZ_PER_PPM]), 2.667)
i1, i2 = np.argmin(np.abs(hzax + 4)), np.argmin(np.abs(hzax - 4))
from scipy.signal import find_peaks as fp
pk = fp(s2)[0]
manual = 1 - s2[pk[0]:pk[1] + 1].min() / min(s2[pk[0]], s2[pk[1]])
unit["dip formula equals 1 - I_valley/min(I1,I2) (unequal heights)"] = {"evaluate": float(d[0]), "manual": float(manual)}
s1 = lor(0, 4.0)
f, d, r = evaluate(s1, ppm_t, np.array([5.05 - 2 / HZ_PER_PPM, 5.05 + 2 / HZ_PER_PPM]), 2.667)
unit["one maximum for two reference peaks -> fail"] = {"found": f, "reason": r}
f, d, r = evaluate(s1, ppm_t, np.array([5.05 - 10 / HZ_PER_PPM]), 2.667)
unit["reference peak with no maximum within tolerance -> fail"] = {"found": f, "reason": r}
f, d, _ = evaluate(s2 * 37.0, ppm_t, np.array([5.05 - 4 / HZ_PER_PPM, 5.05 + 4 / HZ_PER_PPM]), 2.667)
unit["scale invariance (spectrum x 37)"] = {"dip": float(d[0])}
res["item1_unit_tests"] = unit

# notebook / Agent 3 implementations on the same synthetic case
nb_cells = __import__("importlib").import_module("notebook_cells") if False else None
sys.path.insert(0, str(ROOT / "agents" / "agent3_spectral"))
import common as A3  # noqa: E402
ev = A3.evaluate(s2, ppm_t, [{"ppm": 5.05 - 4 / HZ_PER_PPM}, {"ppm": 5.05 + 4 / HZ_PER_PPM}])
unit["Agent 3 common.evaluate on the same case"] = {"dip": ev["dips"][0]}

# ------------------------------------------------------------ item 2: thresholds, full 0.01 grid
levels = (0.0, 0.05, 0.10, 0.19)
item2 = {}
for m in "ABC":
    cxx, cyx, mz = sig[m]
    rp, rs = refs[assign[m]]
    cen = char_peaks(rp, rs, merge_hz=2 * (1 / (L * DT)))
    sc = full_scan(cxx, cyx, mz, cen, 2 * (1 / (L * DT)), levels)
    # negativity of the signed spectrum near the threshold (does np.abs matter?)
    g10 = sc[0.10]["highest_pass"]
    neg = float(signed_spectrum(cxx, cyx, g10, DT, L).min())
    s_abs, p = official_spectrum(cxx, cyx, g10, DT, L, mz)
    s_sign = signed_spectrum(cxx, cyx, g10, DT, L)
    f1, d1, _ = evaluate(s_abs, p, cen, 2.667)
    f2, d2, _ = evaluate(np.maximum(s_sign, 0) if False else s_sign, p, cen, 2.667)
    # tolerance sensitivity
    tol_sens = {}
    for tol_bins in (1.0, 1.5, 2.0, 3.0):
        r = full_scan(cxx, cyx, mz, cen, tol_bins * (1 / (L * DT)), (0.10,),
                      grid=np.round(np.arange(0.0, 60.0 + 1e-9, 0.05), 2))
        tol_sens[f"{tol_bins} bins"] = r[0.10]["highest_pass"]
    item2[m] = {"matched": f"spectrum_{assign[m]}", "n_characteristic_peaks": int(len(cen)),
                "levels": {str(k): v for k, v in sc.items()},
                "min of signed spectrum at threshold": neg,
                "min dip with abs vs without abs at threshold": [float(d1.min()), float(d2.min())],
                "threshold_10pct_vs_position_tolerance (0.05 grid)": tol_sens}
res["item2_thresholds_full_grid"] = item2

(OUT / "part1_results.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({"item5": {m: {k: v for k, v in d.items() if "alternative" not in k} for m, d in item5.items()}}, indent=1, default=str)[:3500])
print(json.dumps(item_sig, indent=1))
print("matching all agree:", res["item6_matching"]["all_choices_agree"], res["item6_matching"]["winner"],
      "choices:", res["item6_matching"]["n_choices"])
for r in match_rows:
    if r["noise"] == 5.0:
        print(r["axis"], r["metric"], r["best"], r["best_total"], r["second_total"])
print(json.dumps(unit, indent=1, default=str))
for m in "ABC":
    print(m, {k: (v["highest_pass"], v["first_fail_after"], v["n_changes"], v["passes_at_lowest"]) for k, v in item2[m]["levels"].items()},
          "signed min", round(item2[m]["min of signed spectrum at threshold"], 4), "dips abs/signed", item2[m]["min dip with abs vs without abs at threshold"],
          "tol", item2[m]["threshold_10pct_vs_position_tolerance (0.05 grid)"])
print("alt weights:", {m: item5[m]["alternative weightings, max deviation from the trace"] for m in "ABC"})
