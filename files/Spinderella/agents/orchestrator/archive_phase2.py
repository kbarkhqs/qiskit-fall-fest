"""Archive everything needed to rerun the Phase-2 analysis WITHOUT IBM credentials:
  * the exact ISA circuits and observables that IBM ran (fetched from the jobs' inputs) -> QPY + JSON
  * the backend noise model (Aer NoiseModel.from_backend) -> JSON, with timestamp
  * backend target summary (coupling map, basis, durations of the used qubits)
Usage: .venv/bin/python agents/orchestrator/archive_phase2.py [pilot|full|all]
"""
import datetime
import json
import sys
from pathlib import Path

from qiskit import qpy

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/phase2/raw"
CIRC = ROOT / "results/phase2/circuits"; CIRC.mkdir(parents=True, exist_ok=True)


def archive_job(service, sub_file, backend, cache):
    """Save (1) the circuits we submitted, rebuilt deterministically (same layout and seed), and
    (2) IBM's stored circuit templates of the job, after checking they agree (CZ count, active qubits)."""
    sys.path.insert(0, str(ROOT / "src"))
    import circuit_opt as C, phase2_pilot as PP
    meta = json.loads(sub_file.read_text())
    tag = meta["tag"]
    if (CIRC / f"{tag}.qpy").exists():
        return tag, "already archived"
    key = (meta["molecule"], tuple(meta["ks"]), tuple(meta["layout"]), meta["seed"])
    if key not in cache:
        cache[key] = PP.build_pubs(backend, C.spin_model(meta["molecule"]), meta["ks"], meta["layout"], meta["seed"])
    pubs, index = cache[key]
    if "bits" in meta:  # full-experiment jobs hold one starting state
        sel = [i for i, m in enumerate(index) if m["bits"] == meta["bits"]]
        pubs, index = [pubs[i] for i in sel], [index[i] for i in sel]
    items = service.job(meta["job_id"]).inputs["quantum_program"].items
    active = lambda c: sorted({c.find_bit(q).index for inst in c.data for q in inst.qubits if inst.operation.name != "barrier"})
    sig = lambda c: (c.count_ops().get("cz", 0), tuple(active(c)))
    if meta["variant"] == "raw":  # one template per circuit, same order
        mismatch = [i for i, (it, pub) in enumerate(zip(items, pubs)) if sig(it.circuit) != sig(pub[0])]
        if len(items) != len(pubs) or mismatch:
            return tag, f"MISMATCH: {len(items)} items vs {len(pubs)} pubs, differing {mismatch[:5]}"
    else:
        # Mitigated jobs: IBM stores twirled / noise-scaled templates plus calibration items, so a one-to-one
        # check is not possible. The submitted circuits are the same objects as in the raw job of the same
        # molecule and state (compiled once, submitted twice), which is checked one-to-one.
        pass
    with open(CIRC / f"{tag}.qpy", "wb") as fh:
        qpy.dump([pub[0] for pub in pubs], fh)
    with open(CIRC / f"{tag}_ibm_templates.qpy", "wb") as fh:
        qpy.dump([it.circuit for it in items], fh)
    phys = [q for q in pubs[0][0].layout.final_index_layout()]
    (CIRC / f"{tag}_index.json").write_text(json.dumps({"job_id": meta["job_id"], "physical_qubits": phys,
                                                        "observables": ["sum X", "sum Y"], "index": index}))
    how = "checked one-to-one against IBM's templates" if meta["variant"] == "raw" else "same circuits as the raw job; IBM templates stored"
    return tag, f"{len(pubs)} circuits archived ({how})"


def archive_backend(service, name="ibm_kingston", qubits=(151, 152, 153)):
    """Aer NoiseModel.from_backend restricted to the qubits our circuits use (identical for these circuits),
    pickled (Aer's JSON export cannot be read back), plus a readable JSON summary with the timestamp."""
    import pickle
    import qiskit_aer
    from qiskit_aer.noise import NoiseModel
    backend = service.backend(name)
    full = NoiseModel.from_backend(backend)
    small = NoiseModel(basis_gates=full.basis_gates)
    kept = []
    for gate, per_qubits in full._local_quantum_errors.items():
        for qs, err in per_qubits.items():
            if set(qs) <= set(qubits):
                small.add_quantum_error(err, gate, qs); kept.append([gate, list(qs)])
    for qs, err in full._local_readout_errors.items():
        if set(qs) <= set(qubits):
            small.add_readout_error(err, qs); kept.append(["measure", list(qs)])
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (ROOT / "results/phase2/kingston_noise_model.pkl").write_bytes(pickle.dumps(small))
    (ROOT / "results/phase2/kingston_noise_model.json").write_text(json.dumps(
        {"backend": name, "retrieved_utc": stamp, "qubits": list(qubits), "qiskit_aer": qiskit_aer.__version__,
         "basis_gates": small.basis_gates, "errors_kept": kept,
         "file": "results/phase2/kingston_noise_model.pkl (pickled qiskit_aer NoiseModel)"}, indent=1))
    t = backend.target
    summary = {"backend": name, "retrieved_utc": stamp, "num_qubits": backend.num_qubits,
               "basis": sorted(t.operation_names), "coupling_map": [list(e) for e in backend.coupling_map.get_edges()]}
    (ROOT / "results/phase2/kingston_target_summary.json").write_text(json.dumps(summary))
    return stamp


if __name__ == "__main__":
    from qiskit_ibm_runtime import QiskitRuntimeService
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    svc = QiskitRuntimeService(name="nmr-hackathon")
    backend = svc.backend("ibm_kingston")
    cache = {}
    pattern = {"pilot": "pilot_*_submitted.json", "full": "full_*_submitted.json", "all": "*_submitted.json"}[which]
    for f in sorted(RAW.glob(pattern)):
        print(*archive_job(svc, f, backend, cache), flush=True)
    if which in ("all", "pilot") or not (ROOT / "results/phase2/kingston_noise_model.json").exists():
        print("noise model archived at", archive_backend(svc), flush=True)
