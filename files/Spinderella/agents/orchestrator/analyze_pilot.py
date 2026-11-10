"""Analyse the hardware pilot against the ideal values and the noisy simulator (criteria in pilot_plan.json)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import circuit_opt as C  # noqa: E402
import phase2_pilot as PP  # noqa: E402

raw_dir = ROOT / "results/phase2/raw"
model = C.spin_model("molecule A")
sim = json.loads((ROOT / "results/phase2/pilot_noisy_simulation.json").read_text())
report = {"criteria": json.loads((ROOT / "results/phase2/pilot_plan.json").read_text())["success_criteria_written_before_running"],
          "runs": {}}


def corr_with_errors(meta, values):
    ks = meta["ks"]
    cxx, cyx = PP.correlations_from(values, meta["index"], ks)
    var_x = {k: 0.0 for k in ks}
    var_y = {k: 0.0 for k in ks}
    for (sx, sy), m in zip(meta["stds"], meta["index"]):
        var_x[m["k"]] += (m["weight"] * sx) ** 2
        var_y[m["k"]] += (m["weight"] * sy) ** 2
    return cxx, cyx, np.sqrt([var_x[k] for k in ks]), np.sqrt([var_y[k] for k in ks])


for tag in ("pilot_A_raw", "pilot_A_trex_twirl_dd", "pilot_A_zne"):
    f = raw_dir / f"{tag}_result.json"
    if not f.exists():
        continue
    meta = json.loads(f.read_text())
    if "values" not in meta:
        report["runs"][tag] = {"error": meta.get("error")}
        continue
    ks = meta["ks"]
    ix, iy = PP.ideal_values(model, ks)
    cxx, cyx, ex, ey = corr_with_errors(meta, meta["values"])
    dev = np.concatenate([cxx - ix, cyx - iy])
    sig = np.concatenate([ex, ey])
    ideal = np.concatenate([ix, iy])
    meas = np.concatenate([cxx, cyx])
    scale = float(meas @ ideal / (ideal @ ideal))
    run = {"job_id": meta["job_id"], "variant": meta["variant"], "ks": ks, "shots": meta["shots"],
           "usage_seconds": meta.get("usage"),
           "qpu_seconds_per_1000_shots": (meta.get("usage") or 0) / (len(meta["index"]) * 2 * meta["shots"] / 1000),
           "cxx": cxx.tolist(), "cyx": cyx.tolist(), "cxx_std": ex.tolist(), "cyx_std": ey.tolist(),
           "ideal_cxx": ix.tolist(), "ideal_cyx": iy.tolist(),
           "rms_deviation": float(np.sqrt(np.mean(dev ** 2))),
           "rms_statistical_error": float(np.sqrt(np.mean(sig ** 2))),
           "chi2_per_point": float(np.mean((dev / np.maximum(sig, 1e-9)) ** 2)),
           "global_scale": scale,
           "rms_after_scale": float(np.sqrt(np.mean((meas - scale * ideal) ** 2))),
           "rms_after_scale_pct_of_Cxx0": float(100 * np.sqrt(np.mean((meas - scale * ideal) ** 2)) / 24)}
    if 0 in ks:
        i0 = ks.index(0)
        run["t0"] = {"cxx0": float(cxx[i0]), "cxx0_pct_dev": float(100 * (cxx[i0] - 24) / 24),
                     "cyx0": float(cyx[i0]), "cyx0_in_sigma": float(cyx[i0] / max(ey[i0], 1e-9))}
    if meta["variant"] == "zne" and meta.get("evs_extrapolated"):
        run["zne_extrapolated_available"] = True
    # same time points of the noisy simulator for reference
    sim_idx = [sim["ks"].index(k) for k in ks if k in sim["ks"]]
    run["noisy_sim_cxx"] = [sim["noisy_sim_cxx"][i] for i in sim_idx]
    run["noisy_sim_cyx"] = [sim["noisy_sim_cyx"][i] for i in sim_idx]
    report["runs"][tag] = run

(ROOT / "results/phase2/pilot_analysis.json").write_text(json.dumps(report, indent=1))
for tag, r in report["runs"].items():
    if "error" in r:
        print(tag, "ERROR", r["error"]); continue
    t0 = r.get("t0", {})
    print(f"{tag}: usage {r['usage_seconds']} s ({r['qpu_seconds_per_1000_shots']:.3f} s per 1000 shots) | rms dev {r['rms_deviation']:.3f} "
          f"(stat {r['rms_statistical_error']:.3f}, chi2/pt {r['chi2_per_point']:.1f}) | scale {r['global_scale']:.3f}, rms after scale "
          f"{r['rms_after_scale']:.3f} = {r['rms_after_scale_pct_of_Cxx0']:.1f} % of C_xx(0)"
          + (f" | t0: C_xx {t0['cxx0']:.2f} ({t0['cxx0_pct_dev']:+.1f} %), C_yx {t0['cyx0']:.2f} ({t0['cyx0_in_sigma']:+.1f} sigma)" if t0 else ""))
    print("   k      ideal Cxx  hw Cxx (+-)        ideal Cyx  hw Cyx (+-)")
    for i, k in enumerate(r["ks"]):
        print(f"   {k:3d}  {r['ideal_cxx'][i]:9.3f}  {r['cxx'][i]:7.3f} ({r['cxx_std'][i]:.2f})   {r['ideal_cyx'][i]:9.3f}  {r['cyx'][i]:7.3f} ({r['cyx_std'][i]:.2f})")
