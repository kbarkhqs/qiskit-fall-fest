"""Run the NMR Trotter circuits on IBM Quantum hardware: submit now, fetch later.

  submit  transpile for the least-busy (or given) backend, submit all Estimator jobs
          without waiting, and write a manifest JSON with the job ids.
  status  print the status of every job in a manifest.
  fetch   download finished results, build C_xx/C_yx, save .npz and a plot
          (hardware vs ideal statevector [vs FakeKingston .npz if given]).

The API key is read from IBM_QUANTUM_API_KEY in .env (never printed / saved).

Usage:
  python hardware_jobs.py submit "molecule A" --steps 60 --dt 0.003 --shots 500
  python hardware_jobs.py submit "molecule A" --steps 60 --dt 0.003 --shots 500 --dry-run
  python hardware_jobs.py status results/hardware/<manifest>.json
  python hardware_jobs.py fetch  results/hardware/<manifest>.json --reference spectrum_3.pkl \
        --fake results/fake_60steps/molecule_A_FakeKingston_exact.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

import fake_backend_sim as fbs
import nmr_sim as ns

DEFAULT_INSTANCE = (
    "crn:v1:bluemix:public:quantum-computing:us-east:a/6744467e2ccc4d768bf6ae38f7e3604b:"
    "dde0349f-4e99-451e-accd-840c690cd1d7::"
)
OUT_DIR = Path("results/hardware")


def get_service():
    from dotenv import load_dotenv
    from qiskit_ibm_runtime import QiskitRuntimeService

    load_dotenv(Path(__file__).parent / ".env")
    return QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ["IBM_QUANTUM_API_KEY"].strip(),
        instance=os.environ.get("IBM_QUANTUM_INSTANCE", DEFAULT_INSTANCE),
    )


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #
def build_pubs(molecule, n_steps, dt, backend, opt_level, seed):
    ham = ns.load_hamiltonian(molecule)
    n = ham.current_number_spins()
    ham, chemical_shift_hz = ns.to_rotating_frame(ham)
    ham_q = ns.struqture_hamiltonian_to_qiskit_op(ham, n)
    init, phi = fbs.xsum_eigenstate_template(n)
    circuits = fbs.circuits_per_step(init, ns.trotter_step_circuit(ham_q, dt), n_steps)
    observables = list(ns.total_magnetization_ops(n))
    # Same order as the notebook's bits_list; bit i -> phi[i] = pi * bit.
    param_values = np.pi * np.array(list(product((0, 1), repeat=n)), dtype=float)

    from qiskit.transpiler import generate_preset_pass_manager

    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=opt_level, seed_transpiler=seed
    )
    isa_circuits = pm.run(circuits)
    pubs, info = [], []
    for k, isa in enumerate(isa_circuits):
        obs = [[o.apply_layout(isa.layout)] for o in observables]  # (2,1) x (2^N,) -> (2, 2^N)
        order = [list(phi).index(p) for p in isa.parameters]
        pubs.append((isa, obs, param_values[:, order]))
        info.append({
            "step": k,
            "two_qubit_gates": int(sum(isa.count_ops().get(g, 0) for g in ("cz", "ecr", "cx"))),
            "depth": int(isa.depth()),
            "physical_qubits": [int(q) for q in isa.layout.final_index_layout()],
        })
    return pubs, info, n, chemical_shift_hz


def configure_estimator(backend, shots, args):
    from qiskit_ibm_runtime import EstimatorV2

    est = EstimatorV2(mode=backend)
    o = est.options
    o.default_shots = shots
    o.resilience_level = args.resilience_level  # 1 = TREX readout mitigation
    if args.dd:
        o.dynamical_decoupling.enable = True
        o.dynamical_decoupling.sequence_type = "XY4"
    if args.gate_twirling:
        o.twirling.enable_gates = True
    if args.rep_delay is not None:
        o.execution.rep_delay = args.rep_delay * 1e-6
    return est


def cmd_submit(args):
    service = get_service()
    if args.backend:
        backend = service.backend(args.backend)
    else:
        backend = service.least_busy(operational=True, simulator=False)
    print(f"backend: {backend.name} ({backend.processor_type}), "
          f"pending jobs: {backend.status().pending_jobs}")

    t0 = time.perf_counter()
    pubs, info, n, chem = build_pubs(
        args.molecule, args.steps, args.dt, backend, args.opt_level, args.seed
    )
    n_exec = len(pubs) * 2 * 2**n
    max_2q = max(i["two_qubit_gates"] for i in info)
    print(f"transpiled {len(pubs)} circuits in {time.perf_counter() - t0:.1f}s; "
          f"max 2q gates {max_2q}; qubits of last circuit {info[-1]['physical_qubits']}")
    print(f"{n_exec} executions x {args.shots} shots = {n_exec * args.shots:,} shots "
          f"(~{n_exec * args.shots * 3.0e-4 / 60:.1f} min QPU at 250 us rep_delay, plus overhead)")

    try:
        usage = service.usage()
        print(f"remaining QPU budget: {usage.get('usage_remaining_seconds')} s")
    except Exception:
        pass

    if args.dry_run:
        print("dry run: nothing submitted")
        return

    est = configure_estimator(backend, args.shots, args)
    chunks = [list(range(i, min(i + args.pubs_per_job, len(pubs))))
              for i in range(0, len(pubs), args.pubs_per_job)]
    jobs = []
    for idx in chunks:  # submit everything first, never wait
        job = est.run([pubs[i] for i in idx])
        jobs.append({"job_id": job.job_id(), "steps": idx})
        print(f"submitted job {job.job_id()} (steps {idx[0]}..{idx[-1]})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = OUT_DIR / f"{args.molecule.replace(' ', '_')}_{backend.name}_{stamp}.json"
    manifest.write_text(json.dumps({
        "molecule": args.molecule,
        "backend": backend.name,
        "n_spins": n,
        "steps": args.steps,
        "dt": args.dt,
        "shots": args.shots,
        "chemical_shift_hz": chem,
        "options": {
            "resilience_level": args.resilience_level, "dd": args.dd,
            "gate_twirling": args.gate_twirling, "rep_delay_us": args.rep_delay,
            "opt_level": args.opt_level, "seed": args.seed,
        },
        "submitted_utc": stamp,
        "jobs": jobs,
        "circuits": info,
    }, indent=1))
    print(f"manifest: {manifest}")


# --------------------------------------------------------------------------- #
# status / fetch
# --------------------------------------------------------------------------- #
def cmd_status(args):
    m = json.loads(Path(args.manifest).read_text())
    service = get_service()
    for j in m["jobs"]:
        job = service.job(j["job_id"])
        extra = ""
        try:
            extra = f" usage {job.usage()} s" if str(job.status()) == "DONE" else ""
        except Exception:
            pass
        print(f"{j['job_id']}  {job.status()}{extra}")


def cmd_fetch(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(args.manifest)
    m = json.loads(path.read_text())
    service = get_service()
    n_steps, n = m["steps"], m["n_spins"]
    evs = np.full((2, 2**n, n_steps), np.nan)
    total_usage = 0.0
    for j in m["jobs"]:
        job = service.job(j["job_id"])
        status = str(job.status())
        if status != "DONE":
            raise SystemExit(f"job {j['job_id']} is {status}; try again later")
        for step, res in zip(j["steps"], job.result()):
            evs[:, :, step] = res.data.evs
        try:
            total_usage += float(job.usage())
        except Exception:
            pass
    exp_x, exp_y = evs
    mx_0 = exp_x[:, 0]
    times = np.arange(n_steps) * m["dt"]
    sig = ns.TimeSignal(m["molecule"], mx_0 @ exp_x, mx_0 @ exp_y, times,
                        m["chemical_shift_hz"], m["dt"], n_steps)
    np.savez(path.with_suffix(".npz"), exp_x=exp_x, exp_y=exp_y, c_xx=sig.c_xx,
             c_yx=sig.c_yx, times=times, chemical_shift_hz=sig.chemical_shift_hz)
    print(f"QPU usage: {total_usage:.0f} s")

    ideal = ns.simulate_time_signal(m["molecule"], n_steps, m["dt"])
    curves = [(ideal, "C0", "ideal (statevector)")]
    if args.fake:
        d = np.load(args.fake)
        curves.append((ns.TimeSignal(m["molecule"], d["c_xx"], d["c_yx"], d["times"],
                                     float(d["chemical_shift_hz"]), m["dt"], n_steps),
                       "C2:", "FakeKingston"))
    curves.append((sig, "C3--", m["backend"]))

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(11, 4))
    for s, style, label in curves:
        ax_t.plot(s.times, s.c_xx, style, label=label)
    ax_t.set_xlabel("Time [s]")
    ax_t.set_ylabel(r"$C_{xx}(t)$ (no window)")
    ax_t.legend(fontsize=7)
    if args.reference:
        rppm, rspec = fbs.load_reference(ns.DATA_DIR / args.reference)
        ax_f.plot(rppm, rspec, "k", lw=1, label=f"HQS reference ({args.reference})")
        ax_f.set_xlim(rppm.min(), rppm.max())
    for s, style, label in curves:
        spec, ppm = ns.spectrum_from_signal(s, args.gamma, pad_factor=args.pad)
        ax_f.plot(ppm, spec, style, label=label)
    ax_f.set_xlabel("Frequency [ppm]")
    ax_f.set_ylabel("Spectrum")
    ax_f.set_title(f"{m['molecule']} on {m['backend']}, {m['shots']} shots, "
                   f"Γ = {args.gamma} s$^{{-1}}$")
    ax_f.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=130)
    err = np.max(np.abs(sig.c_xx - ideal.c_xx)) / np.max(np.abs(ideal.c_xx))
    print(f"max |C_xx(hw) - C_xx(ideal)| / max|C_xx| = {err:.3f}")
    print(f"wrote {path.with_suffix('.npz')} and {path.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit")
    s.add_argument("molecule")
    s.add_argument("--steps", type=int, default=60)
    s.add_argument("--dt", type=float, default=0.003)
    s.add_argument("--shots", type=int, default=500)
    s.add_argument("--backend", default=None, help="default: least busy")
    s.add_argument("--opt-level", type=int, default=2)
    s.add_argument("--seed", type=int, default=1234)
    s.add_argument("--resilience-level", type=int, default=1)
    s.add_argument("--no-dd", dest="dd", action="store_false")
    s.add_argument("--no-gate-twirling", dest="gate_twirling", action="store_false")
    s.add_argument("--rep-delay", type=float, default=None, help="in us (default: backend's)")
    s.add_argument("--pubs-per-job", type=int, default=60)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_submit)

    st = sub.add_parser("status")
    st.add_argument("manifest")
    st.set_defaults(func=cmd_status)

    f = sub.add_parser("fetch")
    f.add_argument("manifest")
    f.add_argument("--reference", default=None)
    f.add_argument("--fake", default=None, help="FakeKingston .npz to overlay")
    f.add_argument("--gamma", type=float, default=1.0)
    f.add_argument("--pad", type=int, default=8)
    f.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
