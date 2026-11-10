"""Phase 2 analysis: ideal vs noisy simulator vs raw hardware vs mitigated hardware, all three molecules.

Inputs : results/phase2/raw/full_*_result.json (untouched hardware outputs)
Outputs: results/phase2/processed/*.json, results/phase2/plots/*.png
No smoothing or filtering is applied; the spectrum code is the validated notebook pipeline
(noise_strength window 5 s^-1, causal FFT, max-normalisation); peak positions on the corrected axis.
Confidence intervals: parametric bootstrap with the per-time-point standard errors reported by the primitive.
"""
import json
import sys
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import circuit_opt as C  # noqa: E402
import phase1_benchmark as B  # noqa: E402
import phase2_pilot as PP  # noqa: E402

RAW = ROOT / "results/phase2/raw"
PROC = ROOT / "results/phase2/processed"; PROC.mkdir(parents=True, exist_ok=True)
PLOTS = ROOT / "results/phase2/plots"; PLOTS.mkdir(parents=True, exist_ok=True)
MOLS = ("molecule A", "molecule B", "molecule C")
REFS = {k: B.reference(k) for k in (1, 2, 3)}
rng = np.random.default_rng(2026)


def hardware_signals(mol, variant):
    files = sorted(RAW.glob(f"full_{mol[-1]}_*_{variant}_result.json"))
    cxx, cyx, var_x, var_y = np.zeros(C.L), np.zeros(C.L), np.zeros(C.L), np.zeros(C.L)
    jobs = []
    for f in files:
        m = json.loads(f.read_text())
        if "values" not in m:
            raise RuntimeError(f"{f.name}: {m.get('error')}")
        w = m["weight"]
        for (vx, vy), (sx, sy), meta in zip(m["values"], m["stds"], m["index"]):
            cxx[meta["k"]] += w * vx; cyx[meta["k"]] += w * vy
            var_x[meta["k"]] += (w * sx) ** 2; var_y[meta["k"]] += (w * sy) ** 2
        jobs.append({"job_id": m["job_id"], "bits": m["bits"], "shots": m["shots"], "usage_s": m.get("usage"),
                     "submitted_utc": m["submitted_utc"], "finished_utc": m.get("finished_utc")})
    return cxx, cyx, np.sqrt(var_x), np.sqrt(var_y), jobs


def noisy_sim_signals(mol, shots=64):
    from qiskit_ibm_runtime import QiskitRuntimeService
    backend = QiskitRuntimeService(name="nmr-hackathon").backend("ibm_kingston")
    sel = json.loads((ROOT / "results/phase2/backend_selection.json").read_text())["backends"]["ibm_kingston"]
    key = "molecule C" if mol == "molecule C" else "molecule A"
    layout, seed = sel[key][0]["layout"], sel[key][0]["seed"]
    model = C.spin_model(mol)
    pubs, index = PP.build_pubs(backend, model, list(range(C.L)), layout, seed)
    vals = PP.noisy_sim_values(backend, pubs, shots=shots, seed=11)
    cxx, cyx = PP.correlations_from(vals, index, list(range(C.L)))
    return cxx, cyx


def metrics(cxx, cyx, model, cen, ref_spec):
    s, p, p_true = B.spectrum(cxx, cyx, model["chem"])
    pos, hgt = B.peaks(s, p_true, cen)
    return s, p, p_true, pos, hgt


def compare(s, pos, hgt, s0, pos0, hgt0):
    ok = np.isfinite(pos) & np.isfinite(pos0)
    return {"rmse": float(np.sqrt(np.mean((s - s0) ** 2))),
            "cosine": float(s @ s0 / np.linalg.norm(s) / np.linalg.norm(s0)),
            "peaks_found": int(np.sum(np.isfinite(pos))), "peaks_expected": int(len(pos0)),
            "max_peak_pos_err_hz": float(np.max(np.abs(pos[ok] - pos0[ok])) * B.HZ_PER_PPM) if ok.any() else None,
            "mean_peak_pos_err_hz": float(np.mean(np.abs(pos[ok] - pos0[ok])) * B.HZ_PER_PPM) if ok.any() else None,
            "max_peak_pos_err_ppm": float(np.max(np.abs(pos[ok] - pos0[ok]))) if ok.any() else None,
            "max_peak_height_err_pct": float(100 * np.max(np.abs(hgt[ok] / hgt0[ok] - 1))) if ok.any() else None}


def bootstrap(cxx, cyx, ex, ey, model, cen, s0, pos0, hgt0, n=200):
    out = {"cosine": [], "max_peak_pos_err_hz": [], "peaks_found": []}
    for _ in range(n):
        bx = cxx + rng.normal(0, 1, cxx.shape) * ex
        by = cyx + rng.normal(0, 1, cyx.shape) * ey
        s, _, _, pos, hgt = metrics(bx, by, model, cen, None)
        c = compare(s, pos, hgt, s0, pos0, hgt0)
        out["cosine"].append(c["cosine"]); out["peaks_found"].append(c["peaks_found"])
        out["max_peak_pos_err_hz"].append(c["max_peak_pos_err_hz"] if c["max_peak_pos_err_hz"] is not None else np.nan)
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"cosine_95ci": q(out["cosine"]), "max_peak_pos_err_hz_95ci": q(out["max_peak_pos_err_hz"]),
            "all_peaks_found_fraction": float(np.mean(np.array(out["peaks_found"]) == len(pos0)))}


def main():
    summary, spectra = {}, {}
    for mol in MOLS:
        model = C.spin_model(mol)
        cen = B.char_peaks(*REFS[B.ASSIGN[mol]])
        ix, iy = C.signals(C.time_point_unitaries("exact", model), model, half=True)
        s0, p0, p0t, pos0, hgt0 = metrics(ix, iy, model, cen, None)
        sx, sy = noisy_sim_signals(mol)
        rows = {"ideal (exact = Phase-1 best circuit, noise-free)": (ix, iy, None, None, None)}
        rows["noisy simulator (kingston calibration, 64 shots, raw)"] = (sx, sy, None, None, None)
        for variant, label in (("raw", "IBM hardware raw (64 shots)"), ("trex_twirl_dd", "IBM hardware mitigated (128 shots)")):
            hx, hy, ex, ey, jobs = hardware_signals(mol, variant)
            rows[label] = (hx, hy, ex, ey, jobs)
        res = {}
        for label, (cx, cy, ex, ey, jobs) in rows.items():
            s, p, pt, pos, hgt = metrics(cx, cy, model, cen, None)
            spectra[mol, label] = (s, pt)
            entry = compare(s, pos, hgt, s0, pos0, hgt0)
            entry["corr_rms_dev"] = float(np.sqrt(np.mean(np.concatenate([cx - ix, cy - iy]) ** 2)))
            ideal = np.concatenate([ix, iy]); meas = np.concatenate([cx, cy])
            entry["corr_global_scale"] = float(meas @ ideal / (ideal @ ideal))
            entry["corr_rms_after_scale"] = float(np.sqrt(np.mean((meas - entry["corr_global_scale"] * ideal) ** 2)))
            entry["peak_positions_ppm"] = [None if not np.isfinite(v) else float(v) for v in pos]
            if ex is not None:
                entry.update(bootstrap(cx, cy, ex, ey, model, cen, s0, pos0, hgt0))
                entry["jobs"] = jobs
                entry["corr_typical_std"] = float(np.median(np.concatenate([ex[1:], ey[1:]])))
            res[label] = entry
        summary[mol] = res
        np.savez(PROC / f"signals_{mol[-1]}.npz", ideal_cxx=ix, ideal_cyx=iy, sim_cxx=sx, sim_cyx=sy,
                 **{f"{v}_{c}": rows[l][i] for v, l in (("raw", "IBM hardware raw (64 shots)"), ("mit", "IBM hardware mitigated (128 shots)"))
                    for c, i in (("cxx", 0), ("cyx", 1), ("cxx_std", 2), ("cyx_std", 3))})
        print(mol, {k: (round(v["cosine"], 4), v["peaks_found"], v["peaks_expected"], round(v["max_peak_pos_err_hz"] or -1, 2)) for k, v in res.items()}, flush=True)

    # matching task from hardware spectra alone (mitigated and raw)
    def bin_avg(rp, rs, ppm):
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (rs[1:] + rs[:-1]) * np.diff(rp))])
        h = 0.5 * (ppm[1] - ppm[0])
        lo, hi = np.clip(ppm - h, rp[0], rp[-1]), np.clip(ppm + h, rp[0], rp[-1])
        return (np.interp(hi, rp, cum) - np.interp(lo, rp, cum)) / (2 * h)
    matching = {}
    for label in ("IBM hardware raw (64 shots)", "IBM hardware mitigated (128 shots)"):
        score = {}
        for mol in MOLS:
            s, pt = spectra[mol, label]
            for k, (rp, rs) in REFS.items():
                b = bin_avg(rp, rs, pt)
                score[mol, k] = float(s @ b / np.linalg.norm(s) / np.linalg.norm(b)) if np.linalg.norm(b) > 0 else 0.0
        ranking = sorted(((sum(score[m, k] for m, k in zip(MOLS, perm)), perm) for perm in permutations((1, 2, 3))), reverse=True)
        matching[label] = {"assignment": {m: f"spectrum_{k}" for m, k in zip(MOLS, ranking[0][1])},
                           "best_total": ranking[0][0], "second_total": ranking[1][0],
                           "matrix": {f"{m}-spectrum_{k}": v for (m, k), v in score.items()}}
        print(label, "matching:", matching[label]["assignment"], f"margin {ranking[0][0] - ranking[1][0]:.3f}", flush=True)
    (PROC / "comparison.json").write_text(json.dumps({"summary": summary, "matching_from_hardware": matching}, indent=1, default=str))

    # plots: spectra on comparable axes (corrected ppm), reference on top
    colors = {"ideal (exact = Phase-1 best circuit, noise-free)": "k", "noisy simulator (kingston calibration, 64 shots, raw)": "C7",
              "IBM hardware raw (64 shots)": "C3", "IBM hardware mitigated (128 shots)": "C0"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=110)
    for ax, mol in zip(axes, MOLS):
        rp, rs = REFS[B.ASSIGN[mol]]
        ax.plot(rp, rs, color="0.8", lw=1, label=f"reference spectrum_{B.ASSIGN[mol]} (HQS)")
        for label, col in colors.items():
            s, pt = spectra[mol, label]
            ax.plot(pt, s, color=col, lw=1.4 if "hardware" in label else 1.0, ls="-" if "hardware" in label else "--", label=label)
        ax.set_xlim(rp[0], rp[-1]); ax.set_title(f"{mol}: spectra (noise_strength 5 s$^{{-1}}$), REAL IBM HARDWARE in colour", fontsize=10)
        ax.set_ylabel("normalised spectrum")
    axes[0].legend(fontsize=7, loc="upper right"); axes[-1].set_xlabel("frequency [ppm] (corrected axis)")
    fig.tight_layout(); fig.savefig(PLOTS / "spectra_all_molecules.png"); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), dpi=100, sharex=True)
    t = np.arange(C.L) * C.DT
    for row, mol in enumerate(MOLS):
        d = np.load(PROC / f"signals_{mol[-1]}.npz")
        for col, (key, name) in enumerate((("cxx", "C_xx(t)"), ("cyx", "C_yx(t)"))):
            ax = axes[row, col]
            ax.plot(t, d[f"ideal_{key}"], "k-", lw=1, label="ideal")
            ax.errorbar(t, d[f"mit_{key}"], yerr=d[f"mit_{key}_std"], fmt=".", ms=2, color="C0", elinewidth=0.4, label="hardware mitigated")
            ax.plot(t, d[f"raw_{key}"], ".", ms=2, color="C3", alpha=0.5, label="hardware raw")
            ax.set_title(f"{mol}: {name}", fontsize=9)
    axes[0, 0].legend(fontsize=7); axes[-1, 0].set_xlabel("t (s)"); axes[-1, 1].set_xlabel("t (s)")
    fig.tight_layout(); fig.savefig(PLOTS / "correlations_all_molecules.png"); plt.close(fig)
    print("plots written", flush=True)


if __name__ == "__main__":
    main()
