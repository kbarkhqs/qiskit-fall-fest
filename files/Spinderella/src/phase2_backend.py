"""Phase 2, step 3-4: backend selection and hardware-aware transpilation (no jobs are submitted here).

For every available backend: calibration snapshot, best connected qubit chains (3 qubits for A/B,
2 for C) ranked by calibrated errors, then the Phase-1 best circuits (C6 exact compilation) are
transpiled onto the best chains with all layout orders and several seeds. Each result is scored by
its estimated success probability ESP = prod(1 - error) over all gates and measurements, using the
backend's own calibration (Target). Writes results/phase2/backend_selection.json.
"""
from __future__ import annotations

import datetime
import json
import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from qiskit import transpile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit_opt as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase2"


def calib(backend):
    t = backend.target
    twoq = [g for g in ("cz", "ecr", "cx") if g in t.operation_names][0]
    e2 = {tuple(q): p.error for q, p in t[twoq].items() if p is not None and p.error is not None}
    ro = {q[0]: p.error for q, p in t["measure"].items() if p is not None and p.error is not None}
    sx = {q[0]: p.error for q, p in t["sx"].items() if p is not None and p.error is not None}
    t1t2 = {}
    for q in range(backend.num_qubits):
        qp = backend.qubit_properties(q)
        t1t2[q] = (getattr(qp, "t1", None), getattr(qp, "t2", None))
    return twoq, e2, ro, sx, t1t2


def esp(circ, backend):
    t = backend.target
    p = 1.0
    for inst in circ.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qs = tuple(circ.find_bit(q).index for q in inst.qubits)
        props = t[name].get(qs) if name in t.operation_names else None
        if props is not None and props.error is not None:
            p *= 1 - props.error
    return p


def best_chains(backend, length, e2, ro, sx, top=6):
    edges = {tuple(sorted(e)) for e in e2}
    err = lambda a, b: max(e2.get((a, b), 1.0), e2.get((b, a), 1.0)) if ((a, b) in e2 or (b, a) in e2) else 1.0
    nbrs = {}
    for a, b in edges:
        nbrs.setdefault(a, set()).add(b); nbrs.setdefault(b, set()).add(a)
    chains = set()
    if length == 2:
        chains = edges
    else:
        for mid, ns in nbrs.items():
            for a in ns:
                for c in ns:
                    if a < c:
                        chains.add((a, mid, c))
    scored = []
    for ch in chains:
        links = [err(ch[i], ch[i + 1]) for i in range(len(ch) - 1)]
        if max(links) >= 1.0:
            continue
        # ~13 two-qubit gates per link for the 3-qubit circuit, 3 for the 2-qubit one; readout and sx once
        w = 13 if length == 3 else 3
        score = w * sum(links) + sum(ro.get(q, 1.0) for q in ch) + 10 * sum(sx.get(q, 1.0) for q in ch)
        scored.append((score, ch))
    scored.sort()
    return [c for _, c in scored[:top]]


def transpile_best(circ, backend, chain, seeds=range(6)):
    best = None
    for lay in permutations(chain):
        for seed in seeds:
            t = transpile(circ, backend=backend, initial_layout=list(lay), optimization_level=3, seed_transpiler=seed)
            twoq = sum(v for k, v in t.count_ops().items() if k in ("cz", "ecr", "cx"))
            cand = (round(-math.log(max(esp(t, backend), 1e-300)), 6), twoq, t.depth())
            if best is None or cand < best[0]:
                best = (cand, t, list(lay), seed)
    return best


def run(service):
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = {"retrieved_utc": stamp, "backends": {}}
    circuits = {}
    for m in ("molecule A", "molecule C"):
        model = C.spin_model(m)
        evo = C.evolution_circuit({"formula": "exact"}, model, C.L - 1)
        qc = C.full_circuit(evo, (0,) * model["n"], "X")
        circuits[m] = qc
    for backend in service.backends(simulator=False, operational=True):
        st = backend.status()
        twoq, e2, ro, sx, t1t2 = calib(backend)
        entry = {"pending_jobs": st.pending_jobs, "num_qubits": backend.num_qubits, "two_qubit_gate": twoq,
                 "basis": sorted(set(backend.configuration().basis_gates)),
                 "median_2q_error": float(np.median(list(e2.values()))), "median_readout_error": float(np.median(list(ro.values()))),
                 "median_sx_error": float(np.median(list(sx.values()))),
                 "median_T1_us": float(np.median([v[0] for v in t1t2.values() if v[0]]) * 1e6),
                 "median_T2_us": float(np.median([v[1] for v in t1t2.values() if v[1]]) * 1e6)}
        try:
            entry["calibration_last_update"] = str(backend.properties().last_update_date)
        except Exception:
            entry["calibration_last_update"] = None
        for m, length in (("molecule A", 3), ("molecule C", 2)):
            chains = best_chains(backend, length, e2, ro, sx)
            results = []
            for ch in chains[:3]:
                (score, twoq_n, depth), t, lay, seed = transpile_best(circuits[m], backend, ch)
                results.append({"chain": list(ch), "layout": lay, "seed": seed, "esp": math.exp(-score),
                                "two_qubit_gates": twoq_n, "depth": depth,
                                "two_qubit_depth": t.depth(lambda i: i.operation.num_qubits == 2),
                                "swap_count": t.count_ops().get("swap", 0),
                                "one_qubit_gates": sum(v for k, v in t.count_ops().items() if k in ("rz", "sx", "x")),
                                "link_errors": [max(e2.get((a, b), 0), e2.get((b, a), 0)) for a, b in zip(ch[:-1], ch[1:])],
                                "readout_errors": [ro.get(q) for q in ch],
                                "T1_T2_us": [[round((t1t2[q][0] or 0) * 1e6, 1), round((t1t2[q][1] or 0) * 1e6, 1)] for q in ch]})
            results.sort(key=lambda r: -r["esp"])
            entry[m] = results
        report["backends"][backend.name] = entry
        print(f"{backend.name}: pending {st.pending_jobs}, median CZ {entry['median_2q_error']:.2e}, readout {entry['median_readout_error']:.2e}; "
              f"A best ESP {entry['molecule A'][0]['esp']:.3f} ({entry['molecule A'][0]['two_qubit_gates']} CZ, depth {entry['molecule A'][0]['depth']}, qubits {entry['molecule A'][0]['layout']}); "
              f"C best ESP {entry['molecule C'][0]['esp']:.3f} ({entry['molecule C'][0]['two_qubit_gates']} CZ, qubits {entry['molecule C'][0]['layout']})", flush=True)
    (OUT / "backend_selection.json").write_text(json.dumps(report, indent=1, default=str))
    return report


if __name__ == "__main__":
    from qiskit_ibm_runtime import QiskitRuntimeService
    run(QiskitRuntimeService(name="nmr-hackathon"))
