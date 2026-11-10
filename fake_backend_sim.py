"""Noisy NMR simulation on an IBM fake backend (script version of the last notebook cell).

Same physics as the notebook: for every Trotter depth k the circuit
``init(phi) -> U_step^k`` is run for all 2^N X-basis initial states and the total
magnetisation <sum X_i>, <sum Y_i> is evaluated on a noisy simulator of the backend.

What makes it fast compared to the notebook cell:
  * each depth-k circuit is transpiled once (not once per initial state), all
    depths in one parallel ``pm.run`` call;
  * all depths x all 2^N initial states are submitted in ONE simulator job
    (Aer parallelises over experiments);
  * no IBM account / QiskitRuntimeService is needed for a fake backend.

Modes (``--mode``):
  exact    Aer density-matrix simulation with the backend noise model; exact noisy
           expectation values (no shot noise). Default.
  sampled  What qiskit_ibm_runtime's local mode does for ``Estimator(FakeBackend)``
           in the notebook: measure in the X / Y basis with ``--shots`` shots.

Usage:
  python fake_backend_sim.py "molecule A"
  python fake_backend_sim.py "molecule C" --backend FakeKingston --steps 300 --gamma 5
  python fake_backend_sim.py "molecule B" --mode sampled --shots 1000 --reference spectrum_2.pkl
"""

from __future__ import annotations

import argparse
import pickle
import time
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

import nmr_sim as ns


# --------------------------------------------------------------------------- #
# Circuits
# --------------------------------------------------------------------------- #
def xsum_eigenstate_template(n_qubits: int) -> tuple[QuantumCircuit, ParameterVector]:
    """H then Rz(phi_q) on every qubit: phi=0 -> |+>, phi=pi -> |-> (as in the notebook)."""
    phi = ParameterVector("phi", n_qubits)
    qc = QuantumCircuit(n_qubits, name="XsumEigen")
    for q in range(n_qubits):
        qc.h(q)
        qc.rz(phi[q], q)
    return qc, phi


def circuits_per_step(init: QuantumCircuit, step: QuantumCircuit, n_steps: int) -> list:
    """[init, init+U, init+U^2, ...] (n_steps circuits)."""
    out = [init.copy()]
    c = init
    for _ in range(1, n_steps):
        c = c.compose(step)
        out.append(c)
    return out


def get_backend(name: str):
    from qiskit_ibm_runtime import fake_provider

    try:
        return getattr(fake_provider, name)()
    except AttributeError:
        raise SystemExit(f"Unknown fake backend {name!r} (see qiskit_ibm_runtime.fake_provider)")


# --------------------------------------------------------------------------- #
# Expectation values
# --------------------------------------------------------------------------- #
def expvals_exact(backend, isa_circuits, param_values, phi, observables, seed=None):
    """Exact noisy expectation values via Aer density-matrix simulation.

    All circuits and all parameter sets go into ONE ``AerSimulator.run`` call. Each
    circuit saves the reduced density matrix of the N active physical qubits (ordered
    like the virtual qubits) and Tr(rho O) is evaluated in numpy.
    Returns array (n_obs, n_params, n_circuits).
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    sim = AerSimulator(
        method="density_matrix",
        noise_model=NoiseModel.from_backend(backend),
        max_parallel_experiments=0,  # 0 = use all cores, one experiment per core
    )
    if seed is not None:
        sim.set_options(seed_simulator=seed)

    runs, binds = [], []
    for isa in isa_circuits:
        qc = isa.copy()
        # physical qubit holding virtual qubit i at the end (includes routing permutation)
        qc.save_density_matrix(qubits=isa.layout.final_index_layout(), label="rho")
        runs.append(qc)
        binds.append({p: param_values[:, list(phi).index(p)].tolist() for p in qc.parameters})
    result = sim.run(runs, parameter_binds=binds).result()

    n_par = param_values.shape[0]
    mats = np.stack([o.to_matrix() for o in observables])  # (n_obs, 2^N, 2^N)
    rhos = np.stack([np.asarray(result.data(i)["rho"]) for i in range(len(runs) * n_par)])
    rhos = rhos.reshape(len(runs), n_par, *rhos.shape[1:])  # (circ, par, 2^N, 2^N)
    return np.einsum("oij,cpji->opc", mats, rhos).real


def expvals_sampled(backend, isa_circuits, param_values, phi, observables, shots, seed=None):
    """Shot-based estimates, as ``qiskit_ibm_runtime.EstimatorV2(mode=FakeBackend)`` does
    locally (BackendEstimatorV2 on AerSimulator.from_backend, precision 1/sqrt(shots)).
    Returns array (n_obs, n_params, n_circuits)."""
    from qiskit.primitives import BackendEstimatorV2
    from qiskit_aer import AerSimulator

    sim = AerSimulator.from_backend(backend, max_parallel_experiments=0)
    if seed is not None:
        sim.set_options(seed_simulator=seed)
    est = BackendEstimatorV2(backend=sim, options={"default_precision": 1.0 / np.sqrt(shots)})
    pubs = []
    for isa in isa_circuits:
        # obs shape (n_obs, 1) broadcasts against params (n_par,) -> evs (n_obs, n_par)
        obs = [[o.apply_layout(isa.layout)] for o in observables]
        order = [list(phi).index(p) for p in isa.parameters]  # bind by name
        pubs.append((isa, obs, param_values[:, order]))
    results = est.run(pubs).result()
    return np.stack([r.data.evs for r in results], axis=-1)


# --------------------------------------------------------------------------- #
# Main simulation
# --------------------------------------------------------------------------- #
def run_fake_backend(
    molecule: str,
    backend_name: str = "FakeKingston",
    n_steps: int = ns.NUMBER_OF_TROTTERSTEPS,
    dt: float = ns.TROTTER_TIMESTEP,
    mode: str = "exact",
    shots: int = 1000,
    optimization_level: int = 2,
    seed: int | None = 1234,
) -> tuple[ns.TimeSignal, np.ndarray, np.ndarray]:
    """Return (undamped TimeSignal, exp_x, exp_y) with exp_* of shape (2^N, n_steps)."""
    ham = ns.load_hamiltonian(molecule)
    n = ham.current_number_spins()
    ham, chemical_shift_hz = ns.to_rotating_frame(ham)
    ham_q = ns.struqture_hamiltonian_to_qiskit_op(ham, n)

    init, phi = xsum_eigenstate_template(n)
    step = ns.trotter_step_circuit(ham_q, dt)
    circuits = circuits_per_step(init, step, n_steps)
    observables = list(ns.total_magnetization_ops(n))

    # All 2^N X-basis product states (same order as the notebook's bits_list):
    # bit i -> phi[i] = pi * bit.
    param_values = np.pi * np.array(list(product((0, 1), repeat=n)), dtype=float)

    backend = get_backend(backend_name)
    t0 = time.perf_counter()
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=optimization_level, seed_transpiler=seed
    )
    isa_circuits = pm.run(circuits)  # parallel over circuits
    deepest = isa_circuits[-1]
    n2q = sum(deepest.count_ops().get(g, 0) for g in ("cz", "ecr", "cx"))
    print(f"transpiled {len(isa_circuits)} circuits in {time.perf_counter() - t0:.1f}s "
          f"(deepest: depth {deepest.depth()}, {n2q} two-qubit gates)")

    t0 = time.perf_counter()
    if mode == "exact":
        evs = expvals_exact(backend, isa_circuits, param_values, phi, observables, seed)
    elif mode == "sampled":
        evs = expvals_sampled(backend, isa_circuits, param_values, phi, observables, shots, seed)
    else:
        raise ValueError(mode)
    print(f"simulated ({mode}) in {time.perf_counter() - t0:.1f}s")
    exp_x, exp_y = evs[0], evs[1]

    # Tr(I_a(t) I_x(0)) = sum_m m_x,m(0) <m(t)| I_a |m(t)>, as in the notebook.
    mx_0 = exp_x[:, 0]
    sig = ns.TimeSignal(
        molecule, mx_0 @ exp_x, mx_0 @ exp_y, np.arange(n_steps) * dt,
        chemical_shift_hz, dt, n_steps,
    )
    return sig, exp_x, exp_y


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    spec = np.abs(np.asarray(d["spectrum"], dtype=float))
    return np.asarray(d["frequencies"], dtype=float), spec / spec.max()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("molecule", help='e.g. "molecule A"')
    ap.add_argument("--backend", default="FakeKingston")
    ap.add_argument("--steps", type=int, default=ns.NUMBER_OF_TROTTERSTEPS)
    ap.add_argument("--dt", type=float, default=ns.TROTTER_TIMESTEP)
    ap.add_argument("--mode", choices=["exact", "sampled"], default="exact")
    ap.add_argument("--shots", type=int, default=1000, help="shots in sampled mode")
    ap.add_argument("--opt-level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gamma", type=float, default=5.0, help="exp(-gamma t) window [1/s]")
    ap.add_argument("--reference", default=None,
                    help="reference spectrum in files/, e.g. spectrum_1.pkl")
    ap.add_argument("--pad", type=int, default=8,
                    help="zero-padding factor for the plotted spectrum (interpolation only)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    sig, exp_x, exp_y = run_fake_backend(
        args.molecule, args.backend, args.steps, args.dt, args.mode, args.shots,
        args.opt_level, args.seed,
    )
    tag = f"{args.molecule.replace(' ', '_')}_{args.backend}_{args.mode}"
    np.savez(
        args.out / f"{tag}.npz", exp_x=exp_x, exp_y=exp_y, c_xx=sig.c_xx, c_yx=sig.c_yx,
        times=sig.times, chemical_shift_hz=sig.chemical_shift_hz,
    )

    # Noiseless statevector result for comparison.
    ideal = ns.simulate_time_signal(args.molecule, args.steps, args.dt)

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(11, 4))
    ax_t.plot(ideal.times, ideal.c_xx, "C0", label="ideal (statevector)")
    ax_t.plot(sig.times, sig.c_xx, "C1--", label=f"{args.backend} ({args.mode})")
    ax_t.set_xlabel("Time [s]")
    ax_t.set_ylabel(r"$C_{xx}(t)$ (no window)")
    ax_t.legend(fontsize=7)

    if args.reference:
        rppm, rspec = load_reference(ns.DATA_DIR / args.reference)
        ax_f.plot(rppm, rspec, "k", lw=1, label=f"HQS reference ({args.reference})")
    for s, style, label in [(ideal, "C0", "ideal"), (sig, "C1--", args.backend)]:
        spec, ppm = ns.spectrum_from_signal(s, args.gamma, pad_factor=args.pad)
        ax_f.plot(ppm, spec, style, label=label)
    if args.reference:
        ax_f.set_xlim(rppm.min(), rppm.max())
    ax_f.set_xlabel("Frequency [ppm]")
    ax_f.set_ylabel("Spectrum")
    ax_f.set_title(f"{args.molecule}, window Γ = {args.gamma} s$^{{-1}}$")
    ax_f.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / f"{tag}.png", dpi=130)

    err = np.max(np.abs(sig.c_xx - ideal.c_xx)) / np.max(np.abs(ideal.c_xx))
    print(f"max |C_xx(noisy) - C_xx(ideal)| / max|C_xx| = {err:.3f}")
    print(f"wrote {args.out / (tag + '.npz')} and {args.out / (tag + '.png')}")


if __name__ == "__main__":
    main()
