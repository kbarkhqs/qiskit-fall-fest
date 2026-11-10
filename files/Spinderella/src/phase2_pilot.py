"""Phase 2 pilot: circuits, noisy-simulator dry run, and (only after explicit approval) hardware run.

Circuits: Phase-1 best (C6 exact compilation of every time point), the 4 (A, B) or 1 (C) starting
states left by the prod-Z symmetry, observables sum X and sum Y. Transpiled onto the chosen backend
with the layout and seed selected in results/phase2/backend_selection.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit_opt as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase2"


def prep_and_evolve(model, k, bits):
    n = model["n"]
    qc = QuantumCircuit(n)
    for q in range(n):
        if bits[q]:
            qc.x(q)
        qc.h(q)
    qc.compose(C.evolution_circuit({"formula": "exact"}, model, k), range(n), inplace=True)
    return qc


def sum_obs(n, pauli):
    return SparsePauliOp.from_sparse_list([(pauli, [i], 1.0) for i in range(n)], num_qubits=n)


def build_pubs(backend, model, ks, layout, seed):
    """Estimator pubs: one ISA circuit per (state, time point) with the two observables mapped to the layout."""
    states, weights, _ = C.x_states(model["n"], half=True)
    pubs, index = [], []
    for k in ks:
        for b, w in zip(states, weights):
            isa = transpile(prep_and_evolve(model, k, b), backend=backend, initial_layout=layout,
                            optimization_level=3, seed_transpiler=seed)
            obs = [sum_obs(model["n"], "X").apply_layout(isa.layout), sum_obs(model["n"], "Y").apply_layout(isa.layout)]
            pubs.append((isa, obs))
            index.append({"k": int(k), "bits": list(b), "weight": float(w)})
    return pubs, index


def ideal_values(model, ks):
    units = C.time_point_unitaries("exact", model)
    cxx, cyx = C.signals(units, model, half=True)
    return cxx[list(ks)], cyx[list(ks)]


def correlations_from(values, index, ks):
    """values[i] = (<sum X>, <sum Y>) of pub i -> C_xx(t_k), C_yx(t_k) with the half-state weights."""
    cxx = {k: 0.0 for k in ks}
    cyx = {k: 0.0 for k in ks}
    for (vx, vy), meta in zip(values, index):
        cxx[meta["k"]] += meta["weight"] * vx
        cyx[meta["k"]] += meta["weight"] * vy
    return np.array([cxx[k] for k in ks]), np.array([cyx[k] for k in ks])


def noisy_sim_values(backend, pubs, shots, seed=7):
    """Raw (unmitigated) expectation values from a noisy simulator of the backend: basis change +
    measurement on the physical qubits, counts -> <sum P> = sum_i <1 - 2 b_i>."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator.from_backend(backend, seed_simulator=seed)
    out = []
    for isa, obs in pubs:
        phys = [q for q in isa.layout.final_index_layout()]
        vals = []
        for basis in ("X", "Y"):
            qc = isa.copy()
            creg_qc = QuantumCircuit(*qc.qregs)
            from qiskit import ClassicalRegister
            cr = ClassicalRegister(len(phys), "m")
            qc.add_register(cr)
            for i, q in enumerate(phys):
                if basis == "Y":
                    qc.sdg(q)
                qc.h(q)
                qc.measure(q, cr[i])
            qc = transpile(qc, backend=backend, optimization_level=0)
            run_seed = seed + 2 * len(out) + (basis == "Y")  # own seed per circuit: no correlated shot noise
            counts = sim.run(qc, shots=shots, seed_simulator=run_seed).result().get_counts()
            total = sum(counts.values())
            ev = 0.0
            for key, c in counts.items():
                bits = key.replace(" ", "")[::-1]
                ev += c * sum(1 - 2 * int(bits[i]) for i in range(len(phys)))
            vals.append(ev / total)
        out.append(tuple(vals))
    return out


# ------------------------------------------------------------------ hardware (run ONLY after explicit approval)
VARIANTS = ("raw", "trex_twirl_dd", "zne")


def estimator_options(variant, shots):
    """Options for qiskit-ibm-runtime 0.52's EstimatorV2 (pydantic options model)."""
    from qiskit_ibm_runtime.options_models.estimator import EstimatorOptions
    if variant == "raw":
        spec = {"default_shots": shots, "resilience_level": 0,
                "twirling": {"enable_gates": False, "enable_measure": False},
                "dynamical_decoupling": {"enable": False}}
    elif variant == "trex_twirl_dd":  # TREX readout mitigation + Pauli twirling + XY4 dynamical decoupling
        spec = {"default_shots": shots, "resilience_level": 1,
                "twirling": {"enable_gates": True, "enable_measure": True},
                "dynamical_decoupling": {"enable": True, "sequence_type": "XY4"}}
    elif variant == "zne":  # TREX + zero-noise extrapolation (noise factors 1, 3, 5)
        spec = {"default_shots": shots, "resilience_level": 1,
                "resilience": {"measure_mitigation": True, "zne_mitigation": True,
                               "zne": {"noise_factors": [1, 3, 5], "extrapolator": ["exponential", "linear"]}},
                "twirling": {"enable_gates": True, "enable_measure": True},
                "dynamical_decoupling": {"enable": True, "sequence_type": "XY4"}}
    else:
        raise ValueError(variant)
    return EstimatorOptions(**spec)


def run_hardware(service, backend_name, molecule, ks, layout, seed, variant, shots, tag):
    """Submit one Estimator job and save job id + raw primitive result (no post-processing)."""
    import datetime
    from qiskit_ibm_runtime import EstimatorV2
    backend = service.backend(backend_name)
    model = C.spin_model(molecule)
    pubs, index = build_pubs(backend, model, ks, layout, seed)
    est = EstimatorV2(mode=backend, options=estimator_options(variant, shots))
    job = est.run(pubs)
    meta = {"tag": tag, "job_id": job.job_id(), "backend": backend_name, "molecule": molecule, "variant": variant,
            "shots": shots, "ks": [int(k) for k in ks], "layout": layout, "seed": seed, "index": index,
            "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{tag}_submitted.json").write_text(json.dumps(meta, indent=1))
    result = job.result()
    vals = [(float(r.data.evs[0]), float(r.data.evs[1])) for r in result]
    stds = [(float(r.data.stds[0]), float(r.data.stds[1])) for r in result]
    meta.update({"finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "values": vals, "stds": stds,
                 "usage_seconds": job.usage() if hasattr(job, "usage") else None,
                 "metrics": job.metrics() if hasattr(job, "metrics") else None})
    (raw / f"{tag}_result.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


# ------------------------------------------------------------------ offline noisy simulation (no credentials)
def noisy_sim_offline(circuits, physical_qubits, noise_model_file, shots, seed=11):
    """Raw noisy-simulator values (<sum X>, <sum Y>) for archived ISA circuits, using the archived
    Aer noise model of the backend. Basis change (S^dagger, H) and measurement are appended on the
    physical qubits; idle qubits are removed by Aer's truncation."""
    from qiskit import ClassicalRegister
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    import pickle
    nm = pickle.loads(Path(noise_model_file).read_bytes())  # archived by agents/orchestrator/archive_phase2.py
    sim = AerSimulator(noise_model=nm, seed_simulator=seed)
    out = []
    for isa in circuits:
        vals = []
        for basis in ("X", "Y"):
            qc = isa.copy()
            cr = ClassicalRegister(len(physical_qubits), "m")
            qc.add_register(cr)
            for i, q in enumerate(physical_qubits):
                if basis == "Y":
                    qc.sdg(q)
                qc.h(q)
                qc.measure(q, cr[i])
            run_seed = seed + 2 * len(out) + (basis == "Y")  # own seed per circuit: no correlated shot noise
            counts = sim.run(qc, shots=shots, seed_simulator=run_seed).result().get_counts()
            total = sum(counts.values())
            ev = sum(c * sum(1 - 2 * int(key.replace(" ", "")[::-1][i]) for i in range(len(physical_qubits)))
                     for key, c in counts.items())
            vals.append(ev / total)
        out.append(tuple(vals))
    return out
