"""Phase 2 analysis, usable from notebooks WITHOUT IBM credentials.

Reads the untouched hardware outputs (results/phase2/raw/full_*_result.json), the archived circuits and
noise model (results/phase2/circuits/, kingston_noise_model.pkl), and returns
ideal / noisy-simulator / raw-hardware / mitigated-hardware correlation functions, spectra and metrics.
No smoothing or filtering is applied; the spectrum code is the notebook's (window exp(-5 t), causal FFT,
max-normalisation). Confidence intervals: parametric bootstrap with the primitive's standard errors.
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np

import circuit_opt as C
import phase1_benchmark as B
import phase2_pilot as PP

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/phase2/raw"
CIRC = ROOT / "results/phase2/circuits"
PROC = ROOT / "results/phase2/processed"
MOLS = ("molecule A", "molecule B", "molecule C")
LABELS = {"ideal": "ideal (Phase-1 best circuit, noise-free)",
          "sim": "noisy simulator (kingston calibration, 64 shots)",
          "raw": "IBM hardware, raw (64 shots)",
          "mit": "IBM hardware, mitigated (128 shots)"}


def hardware_signals(mol, variant):
    """C_xx, C_yx (k = 0..299) and their standard errors from the saved primitive results."""
    files = sorted(RAW.glob(f"full_{mol[-1]}_*_{variant}_result.json"))
    if not files:
        raise FileNotFoundError(f"no hardware results for {mol} / {variant} in {RAW}")
    cxx, cyx, vx, vy = (np.zeros(C.L) for _ in range(4))
    jobs = []
    for f in files:
        m = json.loads(f.read_text())
        if "values" not in m:
            raise RuntimeError(f"{f.name}: {m.get('error')}")
        w = m["weight"]
        for (ex, ey), (sx, sy), meta in zip(m["values"], m["stds"], m["index"]):
            cxx[meta["k"]] += w * ex; cyx[meta["k"]] += w * ey
            vx[meta["k"]] += (w * sx) ** 2; vy[meta["k"]] += (w * sy) ** 2
        jobs.append({"job_id": m["job_id"], "state_bits": m["bits"], "shots": m["shots"], "usage_s": m.get("usage"),
                     "submitted_utc": m["submitted_utc"], "finished_utc": m.get("finished_utc")})
    return cxx, cyx, np.sqrt(vx), np.sqrt(vy), jobs


def noisy_sim_signals(mol, shots=64, recompute=False):
    """Noisy-simulator C_xx, C_yx from the archived circuits and the archived kingston noise model."""
    cache = PROC / f"noisy_sim_{mol[-1]}.npz"
    if cache.exists() and not recompute:
        d = np.load(cache)
        return d["cxx"], d["cyx"]
    from qiskit import qpy
    cxx, cyx = np.zeros(C.L), np.zeros(C.L)
    for f in sorted(CIRC.glob(f"full_{mol[-1]}_*_raw.qpy")):
        idx = json.loads(Path(str(f).replace(".qpy", "_index.json")).read_text())
        with open(f, "rb") as fh:
            circuits = qpy.load(fh)
        state_seed = 1000 * int("".join(map(str, idx["index"][0]["bits"])), 2) + 11  # independent per state
        vals = PP.noisy_sim_offline(circuits, idx["physical_qubits"], ROOT / "results/phase2/kingston_noise_model.pkl", shots,
                                    seed=state_seed)
        for (vx, vy), meta in zip(vals, idx["index"]):
            cxx[meta["k"]] += meta["weight"] * vx; cyx[meta["k"]] += meta["weight"] * vy
    PROC.mkdir(parents=True, exist_ok=True)
    np.savez(cache, cxx=cxx, cyx=cyx)
    return cxx, cyx


def _compare(s, pos, hgt, s0, pos0, hgt0):
    ok = np.isfinite(pos) & np.isfinite(pos0)
    return {"rmse_vs_ideal": float(np.sqrt(np.mean((s - s0) ** 2))),
            "cosine_vs_ideal": float(s @ s0 / np.linalg.norm(s) / np.linalg.norm(s0)),
            "peaks_found": f"{int(np.sum(np.isfinite(pos)))}/{len(pos0)}",
            "max_peak_shift_hz": float(np.max(np.abs(pos[ok] - pos0[ok])) * B.HZ_PER_PPM) if ok.any() else np.nan,
            "max_peak_shift_ppm": float(np.max(np.abs(pos[ok] - pos0[ok]))) if ok.any() else np.nan,
            "max_peak_height_err_pct": float(100 * np.max(np.abs(hgt[ok] / hgt0[ok] - 1))) if ok.any() else np.nan}


def analyse(recompute_noisy_sim=False, bootstrap=200, seed=2026):
    """Returns (rows, spectra, signals, matching). rows: one dict per molecule x data set."""
    rng = np.random.default_rng(seed)
    refs = {k: B.reference(k) for k in (1, 2, 3)}
    rows, spectra, signals = [], {}, {}
    for mol in MOLS:
        model = C.spin_model(mol)
        cen = B.char_peaks(*refs[B.ASSIGN[mol]])
        ix, iy = C.signals(C.time_point_unitaries("exact", model), model, half=True)
        sets = {"ideal": (ix, iy, None, None, []), "sim": (*noisy_sim_signals(mol, recompute=recompute_noisy_sim), None, None, []),
                "raw": hardware_signals(mol, "raw"), "mit": hardware_signals(mol, "trex_twirl_dd")}
        s0, _, p0 = B.spectrum(ix, iy, model["chem"])
        pos0, hgt0 = B.peaks(s0, p0, cen)
        for key, (cx, cy, ex, ey, jobs) in sets.items():
            s, _, pt = B.spectrum(cx, cy, model["chem"])
            pos, hgt = B.peaks(s, pt, cen)
            spectra[mol, key] = (pt, s)
            signals[mol, key] = (cx, cy, ex, ey)
            ideal, meas = np.concatenate([ix, iy]), np.concatenate([cx, cy])
            scale = float(meas @ ideal / (ideal @ ideal))
            row = {"molecule": mol, "data": LABELS[key], **_compare(s, pos, hgt, s0, pos0, hgt0),
                   "corr_rms_vs_ideal": float(np.sqrt(np.mean((meas - ideal) ** 2))),
                   "corr_scale": scale, "corr_rms_after_scale": float(np.sqrt(np.mean((meas - scale * ideal) ** 2))),
                   "jobs": len(jobs), "qpu_seconds": sum(j["usage_s"] or 0 for j in jobs) if jobs else 0}
            if ex is not None and bootstrap:
                cos, shift, found = [], [], []
                for _ in range(bootstrap):
                    bs, _, bp = B.spectrum(cx + rng.normal(0, 1, cx.shape) * ex, cy + rng.normal(0, 1, cy.shape) * ey, model["chem"])
                    bpos, bh = B.peaks(bs, bp, cen)
                    c = _compare(bs, bpos, bh, s0, pos0, hgt0)
                    cos.append(c["cosine_vs_ideal"]); shift.append(c["max_peak_shift_hz"])
                    found.append(int(c["peaks_found"].split("/")[0]) == len(cen))
                row["cosine_95ci"] = [round(float(np.percentile(cos, 2.5)), 4), round(float(np.percentile(cos, 97.5)), 4)]
                row["max_peak_shift_hz_95ci"] = [round(float(np.nanpercentile(shift, 2.5)), 3), round(float(np.nanpercentile(shift, 97.5)), 3)]
                row["all_peaks_found_in_bootstrap"] = float(np.mean(found))
            rows.append(row)

    def bin_avg(rp, rs, ppm):
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (rs[1:] + rs[:-1]) * np.diff(rp))])
        h = 0.5 * (ppm[1] - ppm[0])
        lo, hi = np.clip(ppm - h, rp[0], rp[-1]), np.clip(ppm + h, rp[0], rp[-1])
        return (np.interp(hi, rp, cum) - np.interp(lo, rp, cum)) / (2 * h)

    matching = {}
    for key in ("raw", "mit"):
        score = {}
        for mol in MOLS:
            pt, s = spectra[mol, key]
            for k, (rp, rs) in refs.items():
                b = bin_avg(rp, rs, pt)
                score[mol, k] = float(s @ b / np.linalg.norm(s) / np.linalg.norm(b)) if np.linalg.norm(b) > 0 else 0.0
        ranking = sorted(((sum(score[m, k] for m, k in zip(MOLS, perm)), perm) for perm in permutations((1, 2, 3))), reverse=True)
        matching[LABELS[key]] = {"assignment": {m: f"spectrum_{k}" for m, k in zip(MOLS, ranking[0][1])},
                                 "best_total": round(ranking[0][0], 3), "second_best_total": round(ranking[1][0], 3)}
    return rows, spectra, signals, matching


def plot_spectra(spectra, ax_list=None):
    import matplotlib.pyplot as plt
    refs = {k: B.reference(k) for k in (1, 2, 3)}
    style = {"ideal": ("k", "--", 1.0), "sim": ("C7", ":", 1.2), "raw": ("C3", "-", 1.1), "mit": ("C0", "-", 1.5)}
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=100) if ax_list is None else (None, ax_list)
    for ax, mol in zip(axes, MOLS):
        rp, rs = refs[B.ASSIGN[mol]]
        ax.plot(rp, rs, color="0.82", lw=1, label=f"reference spectrum_{B.ASSIGN[mol]}")
        for key, (col, ls, lw) in style.items():
            pt, s = spectra[mol, key]
            ax.plot(pt, s, color=col, ls=ls, lw=lw, label=LABELS[key])
        ax.set_xlim(rp[0], rp[-1])
        ax.set_title(f"{mol}: spectra at noise_strength 5 s$^{{-1}}$ (red and blue = REAL IBM HARDWARE, ibm_kingston)", fontsize=10)
        ax.set_ylabel("normalised spectrum")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("frequency [ppm] (corrected axis)")
    if fig is not None:
        fig.tight_layout()
    return fig
