"""Submit the APPROVED full experiment (user approval 2026-11-10, option 1) on ibm_kingston:
molecules A, B, C; C6 exact-compilation circuits; k = 0..299; prod-Z-reduced states;
variant 'trex_twirl_dd' at 128 shots and 'raw' at 64 shots; one job per (molecule, state, variant).
Job ids are saved before waiting; raw primitive results are saved per job."""
import datetime, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import circuit_opt as C, phase2_pilot as PP
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2

now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
svc = QiskitRuntimeService(name="nmr-hackathon")
backend = svc.backend("ibm_kingston")
st = backend.status()
sel = json.load(open(ROOT / "results/phase2/backend_selection.json"))["backends"]["ibm_kingston"]
layouts = {"molecule A": (sel["molecule A"][0]["layout"], sel["molecule A"][0]["seed"]),
           "molecule B": (sel["molecule A"][0]["layout"], sel["molecule A"][0]["seed"]),
           "molecule C": (sel["molecule C"][0]["layout"], sel["molecule C"][0]["seed"])}
t = backend.target
snap = {"utc": now(), "pending_jobs": st.pending_jobs, "operational": st.operational, "qubits": {}}
for q in sorted(set(layouts["molecule A"][0]) | set(layouts["molecule C"][0])):
    qp = backend.qubit_properties(q)
    snap["qubits"][q] = {"readout_error": t["measure"][(q,)].error, "sx_error": t["sx"][(q,)].error,
                         "T1_us": (qp.t1 or 0) * 1e6, "T2_us": (qp.t2 or 0) * 1e6}
a, b, c = layouts["molecule A"][0]
for e in ((a, b), (b, a), (b, c), (c, b)):
    if t["cz"].get(e) is not None:
        snap.setdefault("cz_errors", {})[f"{e[0]}-{e[1]}"] = t["cz"][e].error
raw = ROOT / "results/phase2/raw"; raw.mkdir(parents=True, exist_ok=True)
(raw / "full_calibration_snapshot.json").write_text(json.dumps(snap, indent=1, default=str))
print(f"ibm_kingston operational={st.operational} pending={st.pending_jobs}; snapshot saved", flush=True)
if not st.operational:
    sys.exit("backend not operational - nothing submitted")

ks = list(range(C.L))
jobs = []
for mol in ("molecule A", "molecule B", "molecule C"):
    model = C.spin_model(mol)
    layout, seed = layouts[mol]
    states, weights, _ = C.x_states(model["n"], half=True)
    for bits, w in zip(states, weights):
        pubs, index = [], []
        for k in ks:
            isa = transpile(PP.prep_and_evolve(model, k, bits), backend=backend, initial_layout=layout,
                            optimization_level=3, seed_transpiler=seed)
            obs = [PP.sum_obs(model["n"], "X").apply_layout(isa.layout), PP.sum_obs(model["n"], "Y").apply_layout(isa.layout)]
            pubs.append((isa, obs))
            index.append({"k": k, "bits": list(bits), "weight": float(w)})
        cz = [p[0].count_ops().get("cz", 0) for p in pubs]
        for variant, shots in (("trex_twirl_dd", 128), ("raw", 64)):
            tag = f"full_{mol[-1]}_{''.join(map(str, bits))}_{variant}"
            job = EstimatorV2(mode=backend, options=PP.estimator_options(variant, shots)).run(pubs)
            meta = {"tag": tag, "job_id": job.job_id(), "backend": backend.name, "molecule": mol, "variant": variant,
                    "shots": shots, "ks": ks, "layout": layout, "seed": seed, "index": index, "bits": list(bits),
                    "weight": float(w), "cz_per_circuit_max": max(cz), "submitted_utc": now()}
            (raw / f"{tag}_submitted.json").write_text(json.dumps(meta, indent=1))
            print(f"submitted {tag}: job {job.job_id()} ({len(pubs)} pubs, <= {max(cz)} CZ)", flush=True)
            jobs.append((job, meta))
for job, meta in jobs:
    try:
        result = job.result()
        meta["values"] = [(float(r.data.evs[0]), float(r.data.evs[1])) for r in result]
        meta["stds"] = [(float(r.data.stds[0]), float(r.data.stds[1])) for r in result]
        meta["result_metadata"] = result.metadata
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
    meta["finished_utc"] = now()
    try:
        meta["usage"] = job.usage()
    except Exception:
        meta["usage"] = None
    (raw / f"{meta['tag']}_result.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"{meta['tag']} {'FAILED: ' + meta['error'][:200] if 'error' in meta else 'done'} (usage {meta.get('usage')} s)", flush=True)
u = svc.usage()
print(f"all jobs finished; account usage {u.get('usage_consumed_seconds')} s of {u.get('usage_limit_seconds')}", flush=True)
