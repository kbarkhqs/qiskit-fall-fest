"""Submit the APPROVED pilot (user approval 2026-11-10): 3 Estimator jobs on ibm_kingston, molecule A.
Saves job ids immediately, then waits for each result and saves raw primitive outputs + usage."""
import datetime, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import circuit_opt as C, phase2_pilot as PP
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2

svc = QiskitRuntimeService(name="nmr-hackathon")
backend = svc.backend("ibm_kingston")
st = backend.status()
print(f"ibm_kingston operational={st.operational} pending_jobs={st.pending_jobs} at {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
if not st.operational:
    sys.exit("backend not operational - nothing submitted")
sel = json.load(open(ROOT / "results/phase2/backend_selection.json"))["backends"]["ibm_kingston"]["molecule A"][0]
model = C.spin_model("molecule A")
plan = [("pilot_A_raw", "raw", [0, 20, 60, 100, 150, 200, 250, 299]),
        ("pilot_A_trex_twirl_dd", "trex_twirl_dd", [0, 20, 60, 100, 150, 200, 250, 299]),
        ("pilot_A_zne", "zne", [20, 100, 200, 299])]
raw = ROOT / "results/phase2/raw"; raw.mkdir(parents=True, exist_ok=True)
jobs = []
for tag, variant, ks in plan:
    pubs, index = PP.build_pubs(backend, model, ks, sel["layout"], sel["seed"])
    est = EstimatorV2(mode=backend, options=PP.estimator_options(variant, 500))
    job = est.run(pubs)
    meta = {"tag": tag, "job_id": job.job_id(), "backend": backend.name, "molecule": "molecule A", "variant": variant,
            "shots": 500, "ks": ks, "layout": sel["layout"], "seed": sel["seed"], "index": index,
            "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (raw / f"{tag}_submitted.json").write_text(json.dumps(meta, indent=1))
    print(f"submitted {tag}: job {job.job_id()}", flush=True)
    jobs.append((job, meta))
for job, meta in jobs:
    try:
        result = job.result()
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        (raw / f"{meta['tag']}_result.json").write_text(json.dumps(meta, indent=1, default=str))
        print(f"{meta['tag']} FAILED: {meta['error'][:300]}", flush=True)
        continue
    meta["values"] = [(float(r.data.evs[0]), float(r.data.evs[1])) for r in result]
    meta["stds"] = [(float(r.data.stds[0]), float(r.data.stds[1])) for r in result]
    try:
        meta["extra_data_keys"] = [list(r.data.keys()) for r in result][:1]
        if meta["variant"] == "zne" and "evs_extrapolated" in result[0].data:
            meta["evs_extrapolated"] = [r.data.evs_extrapolated.tolist() for r in result]
            meta["evs_noise_factors"] = [r.data.evs_noise_factors.tolist() for r in result] if "evs_noise_factors" in result[0].data else None
    except Exception as e:
        meta["extra_data_error"] = str(e)
    meta["result_metadata"] = result.metadata
    meta["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        meta["usage"] = job.usage()
    except Exception:
        meta["usage"] = None
    try:
        meta["metrics"] = job.metrics()
    except Exception:
        meta["metrics"] = None
    (raw / f"{meta['tag']}_result.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"{meta['tag']} done: usage {meta.get('usage')} s", flush=True)
print("all pilot jobs finished", flush=True)
