"""Phase 1 benchmark: every candidate x molecule -> circuit metrics + accuracy vs exact evolution.

Run:  .venv/bin/python src/phase1_benchmark.py   (writes results/phase1/*.csv|json|png)
Acceptance rules: results/phase1/acceptance_criteria.json (written before this benchmark).
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit_opt as C  # noqa: E402
import nmr_pipeline as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase1"
MOLS = ("molecule A", "molecule B", "molecule C")
ASSIGN = {"molecule A": 3, "molecule B": 2, "molecule C": 1}
HZ_PER_PPM = 1.0 / P.hz_to_ppm_conversion()
GRID = 1.0 / (C.L * C.DT)
TOL = 2 * GRID
GAMMA = 5.0

CANDIDATES = {
    "C0 baseline (official)": {"formula": "lie", "style": "rotations", "level": 1, "states": "all"},
    "C1 baseline + transpiler level 3": {"formula": "lie", "style": "rotations", "level": 3, "rewrite_of_baseline": True},
    "C2 grouped exchange blocks": {"formula": "lie", "style": "grouped", "level": 3, "rewrite_of_baseline": True},
    "C3 Strang (shift/coupling split)": {"formula": "strang", "level": 3},
    "C3b Strang + alternating pairs": {"formula": "strang_alt", "level": 3},
    "C4 second-order Suzuki": {"formula": "suzuki2", "level": 3},
    "C5 shifts folded into pair blocks": {"formula": "pairfield", "level": 3},
    "C6 exact compilation per time point": {"formula": "exact", "level": 3},
}


# ------------------------------------------------------------------ spectra and accuracy
def reference(k):
    d = pickle.load(open(ROOT / f"files/spectrum_{k}.pkl", "rb"))
    p, s = np.asarray(d["frequencies"], float), np.asarray(d["spectrum"], float)
    o = np.argsort(p)
    return p[o], s[o] / np.max(np.abs(s))


def char_peaks(rp, rs):
    idx, _ = find_peaks(rs, height=0.05, prominence=0.02)
    groups = [[idx[0]]]
    for i in idx[1:]:
        (groups[-1].append(i) if (rp[i] - rp[groups[-1][-1]]) * HZ_PER_PPM < TOL else groups.append([i]))
    return np.array([np.average(rp[g], weights=rs[g]) for g in groups])


def spectrum(cxx, cyx, chem, gamma=GAMMA):
    s, p = P.spectrum_from_signals(cxx, cyx, gamma, C.DT, C.L, chem)
    p = np.asarray(p, float)
    j = np.arange(C.L)
    fix = ((2 * j - (C.L - 1)) / ((2 * C.L - 1) * C.DT) - np.fft.fftshift(np.fft.fftfreq(C.L, d=C.DT))) / HZ_PER_PPM
    return s, p, p + fix


def peaks(spec, ppm, centres):
    mx, _ = find_peaks(spec)
    pos, hgt = [], []
    for c in centres:
        near = mx[np.abs(ppm[mx] - c) * HZ_PER_PPM <= TOL]
        if len(near) == 0:
            pos.append(np.nan); hgt.append(np.nan); continue
        i = near[np.argmax(spec[near])]
        y0, y1, y2 = spec[i - 1], spec[i], spec[i + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        pos.append(ppm[i] + d * (ppm[1] - ppm[0])); hgt.append(y1 - 0.25 * (y0 - y2) * d)
    return np.array(pos), np.array(hgt)


def line_error(formula, model):
    """Largest line shift (Hz) and line-height error (%) of a uniform-step formula vs exact lines."""
    if formula == "exact":
        return 0.0, 0.0
    period = 2 if formula.endswith("_alt") else 1
    u = np.eye(2**model["n"], dtype=complex)
    for s in range(period):
        u = C.layers_unitary(model, C.formula_layers(formula, model, s), C.DT) @ u
    pre, post = C.formula_boundary(formula)
    n = model["n"]
    X = sum(C.pauli_matrix({i: "X"}, n) for i in range(n))
    Y = sum(C.pauli_matrix({i: "Y"}, n) for i in range(n))
    from scipy.linalg import schur

    def lines(unitary, dt, left=None, right=None):
        tri, z = schur(unitary, output="complex")
        ph = -np.angle(np.diag(tri))
        a = (X - 1j * Y) if left is None else right @ (X - 1j * Y) @ right.conj().T
        x = X if left is None else left.conj().T @ X @ left
        h = ((z.conj().T @ a @ z) * (z.conj().T @ x @ z).T).real
        f = np.subtract.outer(ph, ph) / (2 * np.pi * dt)
        f = (f + 0.5 / dt) % (1 / dt) - 0.5 / dt
        keep = h > 1e-3 * h.max()
        return f[keep], h[keep]

    e, v = np.linalg.eigh(model["H"])
    fe, he = lines(v @ np.diag(np.exp(-1j * e * C.DT)) @ v.conj().T, C.DT)
    ul, ur = C.layers_unitary(model, pre, C.DT), C.layers_unitary(model, post, C.DT)
    ft, ht = lines(u, period * C.DT, ul, ur)
    ft = ft  # frequencies of a period-step unitary are aliased into +-1/(2 period dt)
    near = [np.argmin(np.abs(((ft - f) + 0.5 / (period * C.DT)) % (1 / (period * C.DT)) - 0.5 / (period * C.DT))) for f in fe]
    shift = [((ft[i] - f) + 0.5 / (period * C.DT)) % (1 / (period * C.DT)) - 0.5 / (period * C.DT) for i, f in zip(near, fe)]
    return float(np.max(np.abs(shift))), float(100 * np.max(np.abs(ht[near] / he - 1)))


def accuracy(formula, step_size=None):
    """Accuracy metrics vs exact evolution for all three molecules + matching check."""
    out, spectra = {}, {}
    refs = {k: reference(k) for k in (1, 2, 3)}
    for m in MOLS:
        model = C.spin_model(m)
        ex, ey = C.signals(C.time_point_unitaries("exact", model), model, half=True)
        cx, cy = C.signals(C.time_point_unitaries(formula, model, step_size=step_size), model, half=True)
        se, _, pe_true = spectrum(ex, ey, model["chem"])
        sc, pc, pc_true = spectrum(cx, cy, model["chem"])
        spectra[m] = (sc, pc)
        cen = char_peaks(*refs[ASSIGN[m]])
        pos_c, h_c = peaks(sc, pc_true, cen)
        pos_e, h_e = peaks(se, pe_true, cen)
        ok = np.isfinite(pos_c) & np.isfinite(pos_e)
        shift_hz, height_pct = (line_error(formula, model) if step_size is None else (np.nan, np.nan))
        out[m] = {
            "E1_corr_err": float(max(np.abs(cx - ex).max(), np.abs(cy - ey).max()) / ex[0]),
            "E2_rmse": float(np.sqrt(np.mean((sc - se) ** 2))),
            "E2_cosine": float(sc @ se / np.linalg.norm(sc) / np.linalg.norm(se)),
            "E3_peak_pos_hz": float(np.max(np.abs(pos_c[ok] - pos_e[ok])) * HZ_PER_PPM),
            "E3_peak_pos_ppm": float(np.max(np.abs(pos_c[ok] - pos_e[ok]))),
            "E4_peak_height_pct": float(100 * np.max(np.abs(h_c[ok] / h_e[ok] - 1))),
            "peaks_found": int(np.sum(np.isfinite(pos_c))), "peaks_expected": int(len(cen)),
            "line_shift_hz": shift_hz, "line_height_pct": height_pct,
        }
    # matching on the candidate's spectra (cosine on the simulation grid, bin-averaged reference)
    def bin_avg(rp, rs, ppm):
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (rs[1:] + rs[:-1]) * np.diff(rp))])
        h = 0.5 * (ppm[1] - ppm[0])
        lo, hi = np.clip(ppm - h, rp[0], rp[-1]), np.clip(ppm + h, rp[0], rp[-1])
        return (np.interp(hi, rp, cum) - np.interp(lo, rp, cum)) / (2 * h)
    score = {}
    for m in MOLS:
        s, p = spectra[m]
        for k, (rp, rs) in refs.items():
            b = bin_avg(rp, rs, p)
            nb = np.linalg.norm(b)
            score[m, k] = float(s @ b / np.linalg.norm(s) / nb) if nb > 0 else 0.0
    best = max(permutations((1, 2, 3)), key=lambda perm: sum(score[m, k] for m, k in zip(MOLS, perm)))
    matching_ok = all(ASSIGN[m] == k for m, k in zip(MOLS, best))
    for m in MOLS:
        out[m]["E5_science"] = bool(matching_ok and out[m]["peaks_found"] == out[m]["peaks_expected"])
    return out


# ------------------------------------------------------------------ circuit metrics
def circuit_set(cand, model, k):
    states = C.x_states(model["n"], half=(cand.get("states") != "all"))[0]
    evo = C.evolution_circuit(cand, model, k)
    return [(b, basis, C.full_circuit(evo, b, basis)) for b in states for basis in ("X", "Y")]


def metrics_deepest(cand, model):
    """Measured metrics of the deepest circuits (all required states, both bases).
    Trotter candidates: k = 299. Exact compilation: the worst k over all time points."""
    level = cand["level"]
    ks = [C.L - 1] if cand["formula"] != "exact" else list(range(1, C.L))
    worst = None
    for k in ks:
        for b, basis, qc in (circuit_set(cand, model, k) if k == ks[-1] else
                             [(None, "X", C.full_circuit(C.evolution_circuit(cand, model, k), (0,) * model["n"], "X"))]):
            met = C.circuit_metrics(C.transpile_for(qc, level))
            if worst is None or (met["cx"], met["depth"]) > (worst["cx"], worst["depth"]):
                worst = met
    # line connectivity: best of all layouts (deepest time point, all-plus state, X basis), SWAPs kept visible
    qc = C.full_circuit(C.evolution_circuit(cand, model, ks[-1]), (0,) * model["n"], "X")
    best_line = None
    if model["n"] == 3:
        for lay in permutations(range(3)):
            t = C.transpile_for(qc, level, coupling=C.LINE, layout=list(lay), keep_swaps=True)
            met = C.circuit_metrics(t)
            met["cx_incl_swaps"] = met["cx"] + 3 * met["swap"]
            met["layout"] = list(lay)
            if best_line is None or (met["cx_incl_swaps"], met["depth"]) < (best_line["cx_incl_swaps"], best_line["depth"]):
                best_line = met
    else:
        best_line = dict(C.circuit_metrics(C.transpile_for(qc, level)), cx_incl_swaps=None, layout=[0, 1])
        best_line["cx_incl_swaps"] = best_line["cx"]
    n_states = len(C.x_states(model["n"], half=(cand.get("states") != "all"))[0])
    return worst, best_line, n_states * (C.L - 1) * 2


def max_unitary_diff(formula_a, formula_b, model):
    ua, ub = C.time_point_unitaries(formula_a, model), C.time_point_unitaries(formula_b, model)
    return float(max(np.abs(a - b).max() for a, b in zip(ua, ub)))


# ------------------------------------------------------------------ run
def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rules = json.loads((OUT / "acceptance_criteria.json").read_text())
    rows, acc_cache = [], {}
    t0 = time.time()
    for name, cand in CANDIDATES.items():
        f = cand["formula"]
        if f not in acc_cache:
            acc_cache[f] = accuracy(f)
        for m in MOLS:
            model = C.spin_model(m)
            deep, line, n_circ = metrics_deepest(cand, model)
            row = {"candidate": name, "molecule": m, "formula": f, "transpiler_level": cand["level"],
                   "circuits_on_hardware": n_circ, **{f"deep_{k}": v for k, v in deep.items()},
                   **{f"line_{k}": v for k, v in line.items()}, **acc_cache[f][m],
                   "circuit_equals_model": C.check_model_vs_circuit(cand, model) if f != "exact" else 0.0}
            if cand.get("rewrite_of_baseline"):
                row["unitary_diff_to_baseline"] = max_unitary_diff(f, "lie", model)
            rows.append(row)
        print(f"{name}: done ({time.time() - t0:.0f} s)", flush=True)
    base = {r["molecule"]: r for r in rows if r["candidate"].startswith("C0")}
    eps = 1e-9
    for r in rows:
        b = base[r["molecule"]]
        r["accepted"] = bool(r["E1_corr_err"] <= b["E1_corr_err"] + eps and r["E2_rmse"] <= b["E2_rmse"] + eps
                             and r["E2_cosine"] >= b["E2_cosine"] - eps and r["E3_peak_pos_hz"] <= b["E3_peak_pos_hz"] + eps
                             and r["E4_peak_height_pct"] <= b["E4_peak_height_pct"] + eps and r["E5_science"])
        r["high_accuracy"] = bool(r["accepted"] and r["E3_peak_pos_hz"] <= 0.1 and r["E4_peak_height_pct"] <= 2
                                  and r["E2_cosine"] >= 0.999)
        for key in ("deep_depth", "deep_twoq_depth", "deep_cx", "circuits_on_hardware"):
            r[f"{key}_reduction_pct"] = round(100 * (1 - r[key] / b[key]), 1) if b[key] else None
    import csv
    with open(OUT / "candidates.csv", "w", newline="") as fh:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # step-size study: fewer Trotter steps at the same sampling (n_k = ceil(t_k / D))
    study = []
    for f in ("lie", "pairfield"):
        for D_ms in (2.5, 3.75, 5.0, 7.5, 10.0, 15.0):
            D = D_ms * 1e-3
            acc = accuracy(f, step_size=None if D_ms == 2.5 else D)
            for m in MOLS:
                model = C.spin_model(m)
                cand = {"formula": f, "level": 3}
                deep = C.circuit_metrics(C.transpile_for(C.full_circuit(C.evolution_circuit(cand, model, C.L - 1, None if D_ms == 2.5 else D),
                                                                         (0,) * model["n"], "X"), 3))
                b = base[m]
                ok = (acc[m]["E1_corr_err"] <= b["E1_corr_err"] + eps and acc[m]["E2_rmse"] <= b["E2_rmse"] + eps
                      and acc[m]["E2_cosine"] >= b["E2_cosine"] - eps and acc[m]["E3_peak_pos_hz"] <= b["E3_peak_pos_hz"] + eps
                      and acc[m]["E4_peak_height_pct"] <= b["E4_peak_height_pct"] + eps and acc[m]["E5_science"])
                study.append({"formula": f, "trotter_step_ms": D_ms, "deepest_steps": C.steps_needed(C.L - 1, C.DT, None if D_ms == 2.5 else D),
                              "molecule": m, "deep_depth": deep["depth"], "deep_twoq_depth": deep["twoq_depth"], "deep_cx": deep["cx"],
                              **{k: acc[m][k] for k in ("E1_corr_err", "E2_rmse", "E2_cosine", "E3_peak_pos_hz", "E4_peak_height_pct", "E5_science")},
                              "accepted": ok})
        print(f"step study {f}: done ({time.time() - t0:.0f} s)", flush=True)
    with open(OUT / "step_size_study.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(study[0])); w.writeheader(); w.writerows(study)

    # aliasing limit for the fixed-length alternative (larger sampling interval, same 0.75 s)
    alias = {}
    for m in MOLS:
        model = C.spin_model(m)
        e = np.linalg.eigvalsh(model["H"])
        fe, he = None, None
        X = sum(C.pauli_matrix({i: "X"}, model["n"]) for i in range(model["n"]))
        Y = sum(C.pauli_matrix({i: "Y"}, model["n"]) for i in range(model["n"]))
        ev, vv = np.linalg.eigh(model["H"])
        h = ((vv.conj().T @ (X - 1j * Y) @ vv) * (vv.conj().T @ X @ vv).T).real
        f = np.subtract.outer(ev, ev) / (2 * np.pi)
        fmax = float(np.max(np.abs(f[h > 1e-3 * h.max()])))
        alias[m] = {"max_line_offset_hz": fmax, "max_sampling_interval_ms": 1e3 / (2 * fmax),
                    "min_time_points_for_0.75s": math.ceil(0.75 * 2 * fmax)}
    (OUT / "aliasing_limits.json").write_text(json.dumps(alias, indent=1))
    print(f"total {time.time() - t0:.0f} s")
    return rows, study, alias


if __name__ == "__main__":
    run()
