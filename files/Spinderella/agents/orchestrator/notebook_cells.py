"""Source of the cells the orchestrator adds to sections 8 and 9 of nmr.ipynb.

Written from the text of agents/agent3_spectral/criterion.md only (not from Agent 3's code), so that
the notebook is an independent second implementation of the same criterion.  Each entry is
(cell_type, source).  `SECTION8` goes after the existing overlay cell, `SECTION9` after the
section-9 markdown cell.
"""

SECTION8 = [
("markdown", r"""### 8.1 Simulate all three molecules

The cells above simulate one molecule (`molecule_name`). The function below repeats exactly the
same steps for any molecule: load the struqture Hamiltonian, subtract the mean Z term (section 2),
convert to a `SparsePauliOp` (section 3), build one first-order Lie–Trotter step (section 4.1),
prepare all $2^N$ X-basis product states (section 5), record $\langle \sum_i X_i\rangle$ and
$\langle \sum_i Y_i\rangle$ after $k$ Trotter steps (section 6) and form the weighted sums
$C_{xx}(t)$, $C_{yx}(t)$ (cell of section 6.2).

How the expectation values are computed: with `USE_ESTIMATOR = True` the section-6 Estimator loop
is run for every molecule (about 7 minutes for each three-spin molecule, because every time step
re-runs the whole circuit from $t=0$). With the default `USE_ESTIMATOR = False` each initial state
is evolved as a `Statevector` by the **same Trotter circuit**, one step at a time, and the same
observables are evaluated after every step. This is the same quantum simulation in a cheaper order;
the next cell checks it against the Estimator result of section 6."""),

("code", r"""from itertools import permutations
from qiskit.quantum_info import Statevector
from scipy.signal import find_peaks

USE_ESTIMATOR = False  # True: run the section-6 Estimator loop for every molecule (slow, same result)
MOLECULES = ["molecule A", "molecule B", "molecule C"]


def rotating_frame_hamiltonian(name, coupling_overrides_hz=None):
    # Sections 1-3: load, subtract the mean Z term, convert to Qiskit.
    # coupling_overrides_hz: optional {(k, l): J in Hz}; replaces the XX, YY, ZZ coefficients of
    # pair (k, l) by pi*J/2. Only used for the artificial J-coupling experiments of section 9.
    with open(data_dir / f"{name}_B{field_T}T.json", "r") as fh:
        ham = PauliHamiltonian.from_json(json.load(fh))
    n = ham.current_number_spins()
    for (k, l), j_hz in (coupling_overrides_hz or {}).items():
        for p in "XYZ":
            ham.set(f"{k}{p}{l}{p}", np.pi * j_hz / 2)
    mean_z = np.mean([ham.get(f"{i}Z") for i in range(n)])
    shift = PauliHamiltonian()
    for i in range(n):
        shift.set(f"{i}Z", mean_z)
    return struqture_hamiltonian_to_qiskit_op(ham - shift, n), n, mean_z


def simulate_molecule(name, coupling_overrides_hz=None):
    # Sections 1-6 for one molecule. Returns the UNWINDOWED C_xx(t), C_yx(t) and the mean Z term.
    ham_q, n, mean_z = rotating_frame_hamiltonian(name, coupling_overrides_hz)

    # Section 4.1: one Lie-Trotter step.
    step = QuantumCircuit(n)
    step.append(PauliEvolutionGate(ham_q, trotter_timestep, synthesis=LieTrotter()), range(n))
    step = step.decompose(reps=1)

    # Section 5: all 2^N X-basis product states.
    template, phi_n = xsum_eigenstate_template(n)
    bits_n = list(product((0, 1), repeat=n))
    maps_n = [{phi_n[i]: np.pi * b[i] for i in range(n)} for b in bits_n]

    # Section 6.1: observables.
    ox = SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(n)], num_qubits=n)
    oy = SparsePauliOp.from_sparse_list([("Y", [i], 1.0) for i in range(n)], num_qubits=n)

    ex = np.zeros((len(bits_n), number_of_trottersteps))
    ey = np.zeros((len(bits_n), number_of_trottersteps))
    if USE_ESTIMATOR:  # section 6.2, unchanged
        circ = QuantumCircuit(n).compose(template)
        for k in range(number_of_trottersteps):
            if k > 0:
                circ = circ.compose(step)
            res = Estimator().run([(circ, [ox, oy], pm) for pm in maps_n]).result()
            for m, r in enumerate(res):
                ex[m, k], ey[m, k] = r.data.evs[0], r.data.evs[1]
    else:  # same circuit, applied one Trotter step at a time
        for m, pm in enumerate(maps_n):
            psi = Statevector(template.assign_parameters(pm))
            for k in range(number_of_trottersteps):
                if k > 0:
                    psi = psi.evolve(step)
                ex[m, k] = psi.expectation_value(ox).real
                ey[m, k] = psi.expectation_value(oy).real

    weights = ex[:, 0]  # m_n = eigenvalue of sum_i X_i of initial state n (N - 2 * number of |->)
    return {"name": name, "n": n, "exp_x": ex, "exp_y": ey,
            "c_xx": weights @ ex, "c_yx": weights @ ey, "chem_shift": mean_z}


def spectrum_at(sim, gamma):
    # Section 6.2 window + section 7 spectrum, unchanged, for decay rate gamma (s^-1).
    t = np.arange(number_of_trottersteps) * trotter_timestep
    w = np.exp(-gamma * t)
    spec, ppm_axis = time_signal_to_spectrum(
        sim["c_xx"] * w, sim["c_yx"] * w, trotter_timestep, number_of_trottersteps,
        Hz_to_ppm_conversion=Hz_to_ppm_conversion, chem_shift_hz=sim["chem_shift"],
    )
    return spec, np.asarray(ppm_axis, dtype=float)


sims = {name: simulate_molecule(name) for name in MOLECULES}

# Check against the Estimator run of section 6 (same molecule, same circuits).
same = sims[molecule_name]
print(f"{molecule_name}: max |<sum X>| difference to the section-6 Estimator = "
      f"{np.max(np.abs(same['exp_x'] - exp_x)):.1e}, max |<sum Y>| difference = "
      f"{np.max(np.abs(same['exp_y'] - exp_y)):.1e}")
assert np.allclose(same["exp_x"], exp_x, atol=1e-9) and np.allclose(same["exp_y"], exp_y, atol=1e-9)
for name, sim in sims.items():
    n = sim["n"]
    print(f"{name}: {2**n} initial states, C_xx(0) = {sim['c_xx'][0]:.6f} (expected N 2^N = {n * 2**n}), "
          f"C_yx(0) = {sim['c_yx'][0]:.1e}")"""),

("markdown", r"""### 8.2 Compare every simulated spectrum with every reference

- References: `files/spectrum_1.pkl`, `spectrum_2.pkl`, `spectrum_3.pkl`, each divided by its
  maximum. File names are not used to decide anything.
- Similarity: cosine similarity between the simulated spectrum and the reference averaged over
  each simulation frequency bin (1.333 Hz wide), on the simulation's ppm grid; the reference is 0
  outside its own ppm window. The simulation uses the notebook's default `noise_strength` of 5.0 s⁻¹.
- Peaks: the reference's characteristic peaks (defined in section 9: maxima with height ≥ 0.05
  and prominence ≥ 0.02, components closer than two grid bins = 2.667 Hz merged) and the fraction
  of them that have a simulated local maximum within ±2.667 Hz.
- Assignment: all 3! = 6 one-to-one assignments are scored by the sum of their three cosine
  similarities, and the best one is kept."""),

("code", r"""HZ_PER_PPM = 1.0 / Hz_to_ppm_conversion  # 500.0 Hz per ppm at 11.7433 T
GRID_HZ = 1.0 / (number_of_trottersteps * trotter_timestep)  # 1.333 Hz frequency grid
MERGE_HZ = 2 * GRID_HZ  # 2.667 Hz: closer lines cannot form two maxima on this grid
TOL_HZ = 2 * GRID_HZ  # position tolerance for matching a simulated maximum to a reference peak

references = {}
for k in (1, 2, 3):
    with open(data_dir / f"spectrum_{k}.pkl", "rb") as fh:
        raw = pickle.load(fh)
    r_ppm = np.asarray(raw["frequencies"], dtype=float)
    r_spec = np.asarray(raw["spectrum"], dtype=float)
    order = np.argsort(r_ppm)
    references[f"spectrum_{k}"] = (r_ppm[order], r_spec[order] / np.max(np.abs(r_spec)))


def characteristic_peaks(r_ppm, r_spec):
    # Reference maxima (height >= 0.05, prominence >= 0.02); maxima closer than MERGE_HZ are merged
    # into one peak at their height-weighted centre with the height of the largest member.
    idx, _ = find_peaks(r_spec, height=0.05, prominence=0.02)
    groups = [[idx[0]]]
    for i in idx[1:]:
        if (r_ppm[i] - r_ppm[groups[-1][-1]]) * HZ_PER_PPM < MERGE_HZ:
            groups[-1].append(i)
        else:
            groups.append([i])
    centres = np.array([np.average(r_ppm[g], weights=r_spec[g]) for g in groups])
    heights = np.array([r_spec[g].max() for g in groups])
    return centres, heights


def bin_average(r_ppm, r_spec, ppm_axis):
    # Mean of the reference over each simulation bin (0 outside the reference window).
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (r_spec[1:] + r_spec[:-1]) * np.diff(r_ppm))])
    h = 0.5 * (ppm_axis[1] - ppm_axis[0])
    lo = np.clip(ppm_axis - h, r_ppm[0], r_ppm[-1])
    hi = np.clip(ppm_axis + h, r_ppm[0], r_ppm[-1])
    return (np.interp(hi, r_ppm, cum) - np.interp(lo, r_ppm, cum)) / (2 * h)


def cosine(a, b):
    # 0 when the reference window does not overlap the simulated window at all
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 0 else 0.0


match_noise = 5.0
score = {}
peak_fraction = {}
for name, sim in sims.items():
    spec, ppm_axis = spectrum_at(sim, match_noise)
    maxima, _ = find_peaks(spec)
    for ref_name, (r_ppm, r_spec) in references.items():
        score[name, ref_name] = cosine(spec, bin_average(r_ppm, r_spec, ppm_axis))
        centres, _ = characteristic_peaks(r_ppm, r_spec)
        found = [np.any(np.abs(ppm_axis[maxima] - c) * HZ_PER_PPM <= TOL_HZ) for c in centres]
        peak_fraction[name, ref_name] = float(np.mean(found))

ref_names = list(references)
print("cosine similarity (rows: molecules, columns: references); in brackets: fraction of the")
print("reference's characteristic peaks found in the simulation")
print(" " * 12 + "".join(f"{r:>20s}" for r in ref_names))
for name in MOLECULES:
    print(f"{name:12s}" + "".join(f"{score[name, r]:>12.3f} [{peak_fraction[name, r]:.2f}]" for r in ref_names))

ranking = sorted(
    ((sum(score[m, r] for m, r in zip(MOLECULES, perm)), perm) for perm in permutations(ref_names)),
    reverse=True,
)
print("\nall one-to-one assignments, best first (total cosine similarity):")
for total, perm in ranking:
    print(f"  {total:.3f}  " + ", ".join(f"{m[-1]} -> {r}" for m, r in zip(MOLECULES, perm)))
assignment = dict(zip(MOLECULES, ranking[0][1]))
print(f"\nmargin between best and second-best assignment: {ranking[0][0] - ranking[1][0]:.3f}")
for name in MOLECULES:
    row_best = max(ref_names, key=lambda r: score[name, r])
    print(f"{name} -> {assignment[name]}  (cosine {score[name, assignment[name]]:.3f}, "
          f"characteristic peaks found {peak_fraction[name, assignment[name]]:.0%}, "
          f"row-wise best is the same: {row_best == assignment[name]})")"""),

("code", r"""fig, axes = plt.subplots(3, 1, figsize=(8, 9), dpi=100)
for ax, name in zip(axes, MOLECULES):
    r_ppm, r_spec = references[assignment[name]]
    spec, ppm_axis = spectrum_at(sims[name], match_noise)
    ax.plot(r_ppm, r_spec, color="C0", lw=1.2, label=f"HQS Spectrum Tools ({assignment[name]})")
    ax.plot(ppm_axis, spec, color="C1", ls="--", lw=1.2, label=f"Simulation, noise_strength = {match_noise}")
    ax.set_xlim(r_ppm[0], r_ppm[-1])
    ax.set_title(f"{name} -> {assignment[name]}", fontsize=10)
    ax.set_ylabel("Spectrum")
    ax.legend(fontsize=7, loc="upper left")
axes[-1].set_xlabel("Frequency [ppm]")
fig.tight_layout()
plt.show()"""),

("markdown", r"""### 8.3 Result of the matching task

| Molecule | Reference spectrum | Cosine similarity (noise 5 s⁻¹) | Characteristic peaks found |
|---|---|---|---|
| A | **spectrum_3** | 0.804 | 8 of 8 |
| B | **spectrum_2** | 0.869 | 8 of 8 |
| C | **spectrum_1** | 0.843 | 4 of 4 |

- Every other molecule–reference pair has a cosine similarity of at most 0.005 and none of its
  characteristic peaks.
- Of the 6 one-to-one assignments, this one totals 2.516; the next best totals 0.873.
- The multiplets agree as well. A has a four-line multiplet at 5.48 ppm and two doublets at 5.79
  and 5.97 ppm. B has doublets at 5.34 and 5.58 ppm and a four-line multiplet at 5.90 ppm. C has
  two doublets at 6.30 and 6.38 ppm.
- Robustness, checked outside the notebook (`agents/`): the specialist found the same assignment for
  all 27 combinations of metric, noise level and resampling. The independent reviewer found it for
  all 126 combinations it tried. The smallest relative margin, 13 %, came from cosine similarity
  restricted to the overlapping ppm window. Full-window metrics give at least 52 %.
- Two hints in the supplied material were **not** used for the decision; both agree with it. The
  original notebook overlays molecule A with `spectrum_3.pkl`, and
  `files/hqs_spectrum_tools_molecule C_B11.7433T.pkl` is byte-identical to `spectrum_1.pkl`.

Matching answer: **A → spectrum_3, B → spectrum_2, C → spectrum_1**. That is three matches, worth
3 of 3 points if correct."""),
]

SECTION9 = [
("markdown", r"""### 9.1 A reproducible "still resolvable" criterion

The task asks for the highest `noise_strength` at which the distinct peaks are still resolvable.
We make that test explicit (it was fixed before any sweep was run):

1. **Characteristic peaks** come from the molecule's high-resolution reference (section 8):
   maxima with height ≥ 0.05 and prominence ≥ 0.02. Reference components closer than two grid bins
   (2.667 Hz) are merged into one peak, because the 300-step signal (0.75 s, grid 1.333 Hz) cannot
   show them as two maxima at any noise strength. This affects only the 0.9 Hz $J_{01}$ splitting of
   molecule A (four pairs → four peaks). That choice matters: if those doublets had to stay resolved,
   molecule A would fail at every noise strength with this acquisition. Molecule B's 0.5 Hz $J_{02}$
   splitting is not resolved even in the reference.
2. **Detection**: every characteristic peak needs its own local maximum of the simulated spectrum
   within ±2.667 Hz, in the right order.
3. **Separation**: between every two neighbouring characteristic peaks the spectrum must dip by at
   least 10 % of the smaller of the two maxima (primary level). The levels "any dip", 5 % and
   19 % (Rayleigh-like) are reported to show how much the answer depends on this choice.
4. **Search**: `noise_strength` from 0.1 to 100 s⁻¹ in steps of 0.1, every change of the
   pass/fail result re-scanned in steps of 0.01. Every change is listed, so a non-monotonic result
   would be visible."""),

("code", r"""DIP_LEVELS = {"any dip": 0.0, "5 %": 0.05, "10 % (primary)": 0.10, "19 %": 0.19}


def resolved(spec, ppm_axis, centres, dip_level):
    # Returns (passes, smallest dip, index of the left peak of the weakest pair or a failure note).
    maxima, _ = find_peaks(spec)
    chosen = []
    for c in centres:
        near = maxima[np.abs(ppm_axis[maxima] - c) * HZ_PER_PPM <= TOL_HZ]
        if len(near) == 0:
            return False, 0.0, f"no maximum near {c:.4f} ppm"
        chosen.append(near[np.argmax(spec[near])])
    if np.any(np.diff(chosen) <= 0):
        return False, 0.0, "two reference peaks share one maximum"
    dips = [1 - spec[a:b + 1].min() / min(spec[a], spec[b]) for a, b in zip(chosen[:-1], chosen[1:])]
    weakest = int(np.argmin(dips))
    ok = all(d > dip_level for d in dips) if dip_level == 0 else all(d >= dip_level for d in dips)
    return ok, dips[weakest], weakest


def scan(sim, centres, dip_level):
    # Dense scan in 0.1 steps, then 0.01 steps inside every interval where the result changes.
    coarse = np.round(np.arange(0.1, 100.0 + 1e-9, 0.1), 2)
    ok = np.array([resolved(*spectrum_at(sim, g), centres, dip_level)[0] for g in coarse])
    grid, flags = list(coarse), list(ok)
    for i in np.where(ok[1:] != ok[:-1])[0]:
        fine = np.round(np.arange(coarse[i] + 0.01, coarse[i + 1] - 1e-9, 0.01), 2)
        grid += list(fine)
        flags += [resolved(*spectrum_at(sim, g), centres, dip_level)[0] for g in fine]
    order = np.argsort(grid)
    grid, flags = np.array(grid)[order], np.array(flags)[order]
    changes = [(grid[i], grid[i + 1]) for i in np.where(flags[1:] != flags[:-1])[0]]
    best = grid[flags].max() if flags.any() else np.nan
    first_fail = grid[(grid > best) & ~flags].min() if np.any((grid > best) & ~flags) else np.nan
    return best, first_fail, changes


tolerance = {}
for name in MOLECULES:
    centres, _ = characteristic_peaks(*references[assignment[name]])
    sim = sims[name]
    base_ok, base_dip, _ = resolved(*spectrum_at(sim, 0.0), centres, 0.10)
    print(f"{name} ({assignment[name]}, {len(centres)} characteristic peaks); "
          f"noise_strength = 0 baseline passes: {base_ok}")
    for label, level in DIP_LEVELS.items():
        best, first_fail, changes = scan(sim, centres, level)
        tolerance[name, label] = (best, first_fail)
        print(f"   {label:>15s}: highest pass {best:6.2f}, first fail {first_fail:6.2f} s^-1, "
              f"pass/fail changes at {[f'{a:.2f}->{b:.2f}' for a, b in changes]}")

    best, first_fail = tolerance[name, "10 % (primary)"]
    _, dip, pair = resolved(*spectrum_at(sim, first_fail), centres, 0.10)
    if isinstance(pair, int):
        a, b = centres[pair], centres[pair + 1]
        tolerance[name, "critical"] = (a, b, (b - a) * HZ_PER_PPM)
        print(f"   critical pair at {first_fail:.2f} s^-1: {a:.4f} and {b:.4f} ppm, "
              f"{(b - a) * HZ_PER_PPM:.2f} Hz apart (dip {dip:.1%})")"""),

("code", r"""fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=100)
for ax, name in zip(axes, MOLECULES):
    a, b, split = tolerance[name, "critical"]
    best, first_fail = tolerance[name, "10 % (primary)"]
    centre, half = 0.5 * (a + b), 25.0 / HZ_PER_PPM
    r_ppm, r_spec = references[assignment[name]]
    keep = np.abs(r_ppm - centre) < half
    ax.plot(r_ppm[keep], r_spec[keep], color="0.6", lw=1, label=f"reference ({assignment[name]})")
    for g, style in ((best, "-"), (first_fail, "--"), (round(2 * first_fail), ":")):
        spec, ppm_axis = spectrum_at(sims[name], g)
        keep = np.abs(ppm_axis - centre) < half
        ax.plot(ppm_axis[keep], spec[keep], ls=style, marker=".", ms=3, label=f"noise_strength {g:g}")
    for c in (a, b):
        ax.axvline(c, color="k", lw=0.5, alpha=0.4)
    ax.set_title(f"{name}: critical pair {split:.2f} Hz apart", fontsize=9)
    ax.set_xlabel("Frequency [ppm]")
    ax.legend(fontsize=6)
axes[0].set_ylabel("Spectrum")
fig.tight_layout()
plt.show()"""),
]

SECTION9[2:2] = [
("markdown", r"""**What "distinguishable" means, shown on real data.** Below are molecule A's two closest
characteristic peaks (the inner lines of the spin-2 multiplet, 6.05 Hz apart) at four noise
strengths. For each pair of neighbouring peaks we:
1. find the two tops (orange dots);
2. find the lowest point between them, the valley (red triangle);
3. compute **dip = 1 − valley ÷ smaller top**.

The pair counts as distinguishable when the dip is at least 10 %. As `noise_strength` grows the
lines get wider, the valley fills in, and finally the two tops merge into one."""),
("code", r"""name = "molecule A"
a_ppm, b_ppm, split = tolerance[name, "critical"]
mid = 0.5 * (a_ppm + b_ppm)
limit = tolerance[name, "10 % (primary)"][0]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), dpi=100, sharey=True)
for ax, g in zip(axes, (10.0, limit, 24.0, 30.0)):
    spec, ppm_axis = spectrum_at(sims[name], g)
    x_hz = (ppm_axis - mid) * HZ_PER_PPM
    keep = np.abs(x_hz) < 9
    ax.plot(x_hz[keep], spec[keep], "-o", ms=3, color="C0", lw=1, label="simulated spectrum")
    for c in (a_ppm, b_ppm):
        ax.axvline((c - mid) * HZ_PER_PPM, color="0.75", lw=0.8, ls=":")
    maxima, _ = find_peaks(spec)
    tops = []
    for c in (a_ppm, b_ppm):
        near = maxima[np.abs(ppm_axis[maxima] - c) * HZ_PER_PPM <= TOL_HZ]
        tops.append(near[np.argmax(spec[near])] if len(near) else None)
    if None in tops or tops[0] == tops[1]:
        top = [t for t in tops if t is not None][0]
        ax.plot(x_hz[top], spec[top], "o", color="C1", ms=9)
        ax.text(0.04, 0.95, "only ONE top left:\nthe two peaks have merged", transform=ax.transAxes,
                va="top", fontsize=9)
        ax.set_title(f"noise_strength {g:g}: NOT distinguishable", color="C3", fontsize=10)
        continue
    i, j = tops
    v = i + int(np.argmin(spec[i:j + 1]))
    smaller = min(spec[i], spec[j])
    dip = 1 - spec[v] / smaller
    ax.plot(x_hz[[i, j]], spec[[i, j]], "o", color="C1", ms=9, label="tops")
    ax.plot(x_hz[v], spec[v], "v", color="C3", ms=9, label="valley")
    ax.hlines(smaller, x_hz[i], x_hz[j], colors="C1", ls="--", lw=1)
    ax.annotate("", xy=(x_hz[v], spec[v]), xytext=(x_hz[v], smaller),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1))
    ax.text(0.04, 0.95, f"smaller top = {smaller:.3f}\nvalley = {spec[v]:.3f}\n"
            f"dip = 1 - {spec[v]:.3f} / {smaller:.3f}\n    = {dip:.1%}", transform=ax.transAxes,
            va="top", fontsize=9, family="monospace")
    ok = dip >= 0.10
    ax.set_title(f"noise_strength {g:g}: {'distinguishable' if ok else 'NOT distinguishable'}"
                 f" ({'>=' if ok else '<'} 10 %)", color="C2" if ok else "C3", fontsize=10)
for ax in axes:
    ax.set_xlabel("Hz from the middle of the pair")
    ax.set_ylim(0, 0.95)
axes[0].set_ylabel("Spectrum")
axes[0].legend(fontsize=7, loc="lower center")
fig.suptitle(f"Molecule A, two closest peaks ({split:.2f} Hz apart, dotted lines = reference positions)",
             fontsize=11)
fig.tight_layout()
plt.show()"""),
]

SECTION9_J = [
("markdown", r"""### 9.2 Is the tolerance related to the $J$-coupling strength?

Two hypotheses (organizer briefing):

- **H1**: larger $J$ couplings widen the multiplets, so the tolerance grows with $J$.
- **H2**: the tolerance is set by the **smallest line spacing that has to stay resolved**, which
  depends on the coupling topology (it is often a *difference* of couplings), on the chemical-shift
  differences and on the intensity ratio of neighbouring lines.

First the observation for the three real molecules, then a controlled experiment: change **one**
coupling of a molecule, keep everything else fixed, rerun the same simulation and the same
criterion. A modified molecule has no supplied reference, so its reference is computed here by
**exact** time evolution (no Trotter error) over a long time (20 s) with 1 Hz wide lines, the
linewidth of the supplied references. The next cell first checks that this exact reference gives the
same characteristic peaks and the same thresholds as the supplied files for A, B and C. These
modified molecules are artificial test systems, not real compounds."""),

("code", r"""def coupling_table(name):
    ham_q, n, _ = rotating_frame_hamiltonian(name)
    coeff = dict(zip(ham_q.paulis.to_labels(), np.real(ham_q.coeffs)))
    couplings = {}
    for k in range(n):
        for l in range(k + 1, n):
            label = ["I"] * n
            label[n - 1 - k] = label[n - 1 - l] = "X"
            couplings[f"J{k}{l}"] = 2 * coeff["".join(label)] / np.pi  # c_kl = pi J_kl / 2
    return couplings


def exact_reference(name, coupling_overrides_hz=None, fwhm_hz=1.0, n_points=8000):
    # Exact evolution of the same rotating-frame Hamiltonian, Tr(A(t) X_tot) in its eigenbasis,
    # 8000 x 2.5 ms = 20 s, window exp(-pi * fwhm * t) (Lorentzian of FWHM fwhm_hz), then the
    # notebook's own time_signal_to_spectrum.
    ham_q, n, mean_z = rotating_frame_hamiltonian(name, coupling_overrides_hz)
    energies, vecs = np.linalg.eigh(ham_q.to_matrix())
    sx = SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(n)], num_qubits=n).to_matrix()
    sy = SparsePauliOp.from_sparse_list([("Y", [i], 1.0) for i in range(n)], num_qubits=n).to_matrix()
    xe, ye = vecs.conj().T @ sx @ vecs, vecs.conj().T @ sy @ vecs
    t = np.arange(n_points) * trotter_timestep
    phase = np.exp(1j * np.subtract.outer(energies, energies)[:, :, None] * t)  # e^{i(E_a - E_b)t}
    c_xx_exact = np.einsum("ab,ba,abt->t", xe, xe, phase).real
    c_yx_exact = np.einsum("ab,ba,abt->t", ye, xe, phase).real
    w = np.exp(-np.pi * fwhm_hz * t)
    spec, ppm_axis = time_signal_to_spectrum(c_xx_exact * w, c_yx_exact * w, trotter_timestep,
                                             n_points, Hz_to_ppm_conversion, mean_z)
    return np.asarray(ppm_axis, dtype=float), spec


print("Couplings (Hz) and the critical line spacing found above:")
for name in MOLECULES:
    j = coupling_table(name)
    a, b, split = tolerance[name, "critical"]
    best, _ = tolerance[name, "10 % (primary)"]
    print(f"  {name}: " + ", ".join(f"{k} = {v:+.2f}" for k, v in j.items())
          + f" | mean |J| = {np.mean(np.abs(list(j.values()))):.2f} | critical spacing {split:.2f} Hz"
          + f" | max tolerable noise_strength {best:.2f} s^-1")

print("\nCheck of the exact reference on the real molecules:")
for name in MOLECULES:
    c_file, _ = characteristic_peaks(*references[assignment[name]])
    c_exact, _ = characteristic_peaks(*exact_reference(name))
    same_count = len(c_file) == len(c_exact)
    shift_hz = np.max(np.abs(c_file - c_exact)) * HZ_PER_PPM if same_count else np.nan
    best_exact, _, _ = scan(sims[name], c_exact, 0.10)
    print(f"  {name}: {len(c_exact)} vs {len(c_file)} characteristic peaks, largest position "
          f"difference {shift_hz:.3f} Hz, threshold {best_exact:.2f} vs {tolerance[name, '10 % (primary)'][0]:.2f} s^-1")"""),

("code", r"""# One coupling changed at a time; every other Hamiltonian parameter fixed.
experiments = [
    ("molecule A", (0, 2), [9.5, 10.5, 11.8, 13.0, 14.0]),  # J02: raises J, shrinks J12 - J02
    ("molecule B", (1, 2), [8.6, 9.5, 10.5, 11.5, 12.5]),  # J12: raises J, shrinks J01 - J12
    ("molecule C", (0, 1), [4.0, 6.0, 7.92, 10.0, 12.0]),  # J01: raises J and the critical spacing
]
j_results = []
print(f"{'artificial system':>26s} {'J (Hz)':>7s} {'critical spacing (Hz)':>22s} {'max noise_strength':>19s}")
for name, (k, l), values in experiments:
    for j_hz in values:
        over = {(k, l): j_hz}
        c_ref, _ = characteristic_peaks(*exact_reference(name, over))
        sim_j = simulate_molecule(name, over)
        best, _, _ = scan(sim_j, c_ref, 0.10)
        _, _, pair = resolved(*spectrum_at(sim_j, best + 0.01), c_ref, 0.10)
        split = (c_ref[pair + 1] - c_ref[pair]) * HZ_PER_PPM if isinstance(pair, int) else np.nan
        j_results.append((name, f"J{k}{l}", j_hz, split, best))
        print(f"{name + f', J{k}{l} changed':>26s} {j_hz:7.2f} {split:22.2f} {best:19.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=100)
for (name, (k, l), _), color in zip(experiments, ("C0", "C1", "C2")):
    rows = [r for r in j_results if r[0] == name]
    label = f"{name[-1]}: J{k}{l} varied"
    axes[0].plot([r[2] for r in rows], [r[4] for r in rows], "o-", color=color, label=label)
    axes[1].plot([r[3] for r in rows], [r[4] for r in rows], "o", color=color, label=label)
for name in MOLECULES:
    best, _ = tolerance[name, "10 % (primary)"]
    axes[1].plot(tolerance[name, "critical"][2], best, "k*", ms=10)
axes[0].set_xlabel("the one coupling that was changed, J (Hz)")
axes[1].set_xlabel("critical line spacing (Hz)   [stars: the three real molecules]")
for ax in axes:
    ax.set_ylabel("max tolerable noise_strength (s$^{-1}$)")
    ax.legend(fontsize=7)
fig.tight_layout()
plt.show()"""),
]

SECTION9_J.append(("markdown", r"""### 9.3 Answers to the noise-tolerance exercise

**Maximum tolerable `noise_strength`** (official 300-step Lie–Trotter simulation, primary criterion
of section 9.1, search resolution 0.01 s⁻¹):

| Molecule | Highest pass (s⁻¹) | First fail (s⁻¹) | Critical line pair (10 % level) | Where it comes from | Any dip / 5 % / 19 % |
|---|---|---|---|---|---|
| A | **19.97** | 19.98 | 6.05 Hz, 5.473 / 5.485 ppm | $J_{12}-J_{02}$: inner lines of spin 2's doublet of doublets (5.48 ppm) | 27.32 / 22.15 / 17.11 |
| B | **22.85** | 22.86 | 7.02 Hz, 5.894 / 5.908 ppm | $J_{01}-J_{12}$: inner lines of spin 1's doublet of doublets (5.90 ppm) | 33.36 / 25.99 / 19.13 |
| C | **23.22** | 23.23 | 7.92 Hz, 6.293 / 6.309 ppm | $J_{01}$: outer, weaker lines of the AB quartet | 28.58 / 25.51 / 20.02 |

Every molecule and every dip level has exactly one pass→fail change between 0.1 and 100 s⁻¹, and
all three spectra pass at `noise_strength = 0`.

**What these numbers support, and what they do not.**
- Molecule A tolerates the least decay at every dip level of 5 % or more.
- B and C differ by only 0.4 s⁻¹ at the 10 % level. Their order flips with the dip level, and also
  with Trotter error: exact time evolution gives C 23.09 and B 23.48. They are not ranked.
- The values depend on the criterion: −16 % to +46 % between the 19 % level and "any dip".
- First-order Trotter error lowers A's value from 21.23 (exact evolution) to 19.97. It does this by
  changing line intensities (by up to 19 %), not line positions.

**Is it related to the $J$-coupling strength?** Not to the strength itself. It is related to the
**smallest line spacing that has to stay resolved**, and that spacing depends on how the couplings
are arranged.
- Molecule A has the largest couplings (mean $|J|$ = 10.2 Hz) and still the lowest tolerance.
- In A and B one coupling is almost zero (0.9 and −0.5 Hz), so the spins form a chain with one
  central spin. That spin's multiplet is a doublet of doublets, and its two inner lines are
  separated by the **difference** of its two couplings: 17.9 − 11.8 = 6.1 Hz in A and
  15.6 − 8.6 = 7.0 Hz in B.
- In C the critical spacing is $J$ itself. However, the two shifts are only 4.6 $J$ apart (strong
  coupling), so the outer lines are 1.54 times weaker than the inner ones and turn into shoulders
  sooner.
- Controlled test (section 9.2, one coupling changed at a time, everything else fixed):
  - **Larger $J$ lowers the tolerance** when it shrinks the critical spacing: A with $J_{02}$ raised
    from 9.5 to 14 Hz drops from 28.31 to 12.75 s⁻¹, and B with $J_{12}$ raised from 8.6 to
    12.5 Hz drops from 22.85 to 8.05 s⁻¹.
  - **Larger $J$ raises it** only when $J$ is the critical spacing: C with $J_{01}$ from 4 to 12 Hz
    rises from 11.19 to 32.00 s⁻¹.
  - Each time, the tolerance moves with the critical spacing. Hypothesis H1 (it grows with $J$)
    fails as a general rule; H2 (it is set by the critical spacing) holds.
- The wider experiments in `agents/` agree. Across 159 artificial Hamiltonians the tolerance
  correlates with the critical spacing (Spearman 0.989) far more than with the coupling that was
  changed (0.538). Raising $J_{02}$ and $J_{12}$ together at a fixed difference leaves it within
  −6 % / +2 %. Changing only C's chemical-shift difference at fixed $J$ changes it by a factor 1.7.
- Rule of thumb: a decay rate Γ gives Lorentzian lines of full width Γ/π Hz. Two equal, isolated
  lines Δ Hz apart lose their dip at Γ = π√3 Δ ≈ 5.4 Δ. That is an upper bound: real multiplets,
  with unequal heights and neighbouring lines, fail earlier (about 3.3 Δ at the 10 % level and 4.4
  to 4.8 Δ for "any dip").

**Limits.**
- `noise_strength` is an artificial decay window applied to an ideal simulation, not a model of
  hardware noise.
- The 2-bin merge rule makes the threshold jump wherever a splitting crosses 2.667 Hz.
- With 300 steps of 2.5 ms, splittings below about 3.5–4 Hz are never resolved.
- Three molecules can only show a correlation; the causal statement rests on the
  one-coupling-at-a-time experiments."""))

SECTION1_VIS = [
("markdown", r"""### 1.1 Picture of the three molecules (spin systems)

The data files do not contain the atoms and bonds of the molecules. They contain what NMR "sees":
the **protons (spins)** and how they interact. The figure below shows, for molecules A, B and C:

- **Top: the coupling network.** Each dot is one proton, labelled with its chemical shift
  $\delta$ in ppm. Each line is a scalar coupling $J$ in Hz. Thicker lines mean stronger
  couplings; dashed grey lines are very weak couplings (below 2 Hz).
- **Bottom: where each proton resonates.** Each tick is at that proton's chemical shift on the ppm
  axis, which is the axis of the spectra later in the notebook. The couplings split each tick
  into a small group of lines (a multiplet).

Values are read from the Hamiltonian files: $\delta_i = \left(-2h_i/(\gamma B) - 1\right)\times 10^6$
from the Z coefficient $h_i$, and $J_{kl} = 2c_{kl}/\pi$ from the XX coefficient $c_{kl}$."""),

("code", r"""GAMMA_1H = 267_522_128.0  # rad s^-1 T^-1 (same value as in section 7)


def spin_system(name):
    # Chemical shifts (ppm) and J couplings (Hz) read from the struqture Hamiltonian file.
    with open(data_dir / f"{name}_B{field_T}T.json", "r") as fh:
        ham = PauliHamiltonian.from_json(json.load(fh))
    n = ham.current_number_spins()
    shifts = [(-2 * float(ham.get(f"{i}Z")) / (GAMMA_1H * field_T) - 1) * 1e6 for i in range(n)]
    couplings = {(k, l): 2 * float(ham.get(f"{k}X{l}X")) / np.pi
                 for k in range(n) for l in range(k + 1, n)}
    return shifts, couplings


fig, axes = plt.subplots(2, 3, figsize=(13, 5.4), dpi=100, gridspec_kw={"height_ratios": [3, 1]})
for col, name in enumerate(["molecule A", "molecule B", "molecule C"]):
    shifts, couplings = spin_system(name)
    n = len(shifts)
    angles = np.pi / 2 + 2 * np.pi * np.arange(n) / n if n > 2 else np.array([np.pi, 0.0])
    xy = np.c_[np.cos(angles), np.sin(angles)]
    centre = xy.mean(axis=0)

    ax = axes[0, col]
    for (k, l), j_hz in couplings.items():
        weak = abs(j_hz) < 2.0
        ax.plot(*xy[[k, l]].T, color="0.6" if weak else "C0", ls="--" if weak else "-",
                lw=1.0 if weak else 1.0 + 0.35 * abs(j_hz), zorder=1)
        mid = xy[[k, l]].mean(axis=0)
        out = mid - centre if np.linalg.norm(mid - centre) > 1e-9 else np.array([0.0, 1.0])
        lab = mid + 0.28 * out / np.linalg.norm(out)
        ax.text(*lab, f"$J_{{{k}{l}}}$ = {j_hz:+.2f} Hz", ha="center", va="center", fontsize=9,
                color="0.4" if weak else "C0",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    ax.scatter(*xy.T, s=3000, color="C1", edgecolor="k", zorder=2)
    for i, (x, y) in enumerate(xy):
        ax.text(x, y + 0.09, f"H{i}", ha="center", va="center", fontsize=11, weight="bold", zorder=3)
        ax.text(x, y - 0.11, f"{shifts[i]:.3f} ppm", ha="center", va="center", fontsize=8, zorder=3)
    ax.set_title(f"{name}: {n} protons, {sum(abs(j) >= 2 for j in couplings.values())} strong coupling(s)",
                 fontsize=10)
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.0, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")

    ruler = axes[1, col]
    for i, d in enumerate(shifts):
        ruler.axvline(d, ymax=0.6, color="C1", lw=2)
        ruler.text(d, 0.75, f"H{i}", ha="center", fontsize=9, transform=ruler.get_xaxis_transform())
    ruler.set_xlim(min(shifts) - 0.15, max(shifts) + 0.15)  # same direction as the spectra below
    ruler.set_yticks([])
    ruler.set_xlabel("chemical shift [ppm]")
    for side in ("left", "right", "top"):
        ruler.spines[side].set_visible(False)

    print(f"{name}: shifts " + ", ".join(f"H{i} {d:.3f} ppm" for i, d in enumerate(shifts))
          + " | couplings " + ", ".join(f"J{k}{l} = {j:+.2f} Hz" for (k, l), j in couplings.items()))
fig.tight_layout()
plt.show()"""),
]

SECTION8.append(("markdown", r"""### 8.4 How large are the matching errors?

For each molecule we compare its simulated spectrum (official settings, `noise_strength` = 5 s⁻¹)
with its matched reference:

- **Position error**: for every characteristic reference peak, the distance in Hz to the simulated
  peak. The simulated peak position is refined between grid points with a parabola through the
  maximum and its two neighbours.
- **Cosine similarity**: the overlap of the two spectra on the simulation grid (1 = identical shape).

Three causes are separated by also computing:
1. the same spectrum on a **corrected frequency axis**. Section 7 labels the bins with
   `fftfreq(L, dt)`, but `fft_causal` evaluates the transform at `(2j-(L-1))/((2L-1)dt)`, so every
   label sits about 0.4–0.9 Hz too low;
2. the spectrum of **exact time evolution** (no Trotter error) with identical sampling and window.
   Its remaining error is the floor set by the 1.333 Hz grid and the line width."""))

SECTION8.append(("code", r"""from scipy.linalg import schur


def true_frequencies_hz(steps, dt):
    # Frequencies at which fft_causal really evaluates the spectrum (section 7 labels them with fftfreq).
    j = np.arange(steps)
    return (2 * j - (steps - 1)) / ((2 * steps - 1) * dt)


def spectrum_general(c_xx, c_yx, gamma, dt, steps, chem_shift):
    # Section 6.2 window + section 7 spectrum for any (dt, steps); returns the official and corrected ppm axes.
    t = np.arange(steps) * dt
    w = np.exp(-gamma * t)
    spec, ppm_axis = time_signal_to_spectrum(c_xx * w, c_yx * w, dt, steps, Hz_to_ppm_conversion, chem_shift)
    ppm_axis = np.asarray(ppm_axis, dtype=float)
    shift_hz = true_frequencies_hz(steps, dt) - np.fft.fftshift(np.fft.fftfreq(steps, d=dt))
    return spec, ppm_axis, ppm_axis + shift_hz / HZ_PER_PPM


def exact_signals_at(ham_q, n, dt, steps):
    # Exact Tr(A(t) X_tot), no Trotter error, sampled like the simulation.
    energies, vecs = np.linalg.eigh(ham_q.to_matrix())
    sx = SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(n)], num_qubits=n).to_matrix()
    sy = SparsePauliOp.from_sparse_list([("Y", [i], 1.0) for i in range(n)], num_qubits=n).to_matrix()
    xe, ye = vecs.conj().T @ sx @ vecs, vecs.conj().T @ sy @ vecs
    phase = np.exp(1j * np.subtract.outer(energies, energies)[:, :, None] * (np.arange(steps) * dt))
    return (np.einsum("ab,ba,abt->t", xe, xe, phase).real, np.einsum("ab,ba,abt->t", ye, xe, phase).real)


def peak_positions(spec, ppm_axis, centres):
    # Sub-bin position of the highest maximum within TOL_HZ of each centre (nan if none).
    maxima, _ = find_peaks(spec)
    out = []
    for c in centres:
        near = maxima[np.abs(ppm_axis[maxima] - c) * HZ_PER_PPM <= TOL_HZ]
        if len(near) == 0:
            out.append(np.nan)
            continue
        i = near[np.argmax(spec[near])]
        y0, y1, y2 = spec[i - 1], spec[i], spec[i + 1]
        out.append(ppm_axis[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (ppm_axis[1] - ppm_axis[0]))
    return np.array(out)


def matching_error(c_xx, c_yx, chem_shift, dt, steps, ref, gamma=5.0):
    r_ppm, r_spec = ref
    centres, _ = characteristic_peaks(r_ppm, r_spec)
    spec, p_official, p_true = spectrum_general(c_xx, c_yx, gamma, dt, steps, chem_shift)
    err_official = (peak_positions(spec, p_official, centres) - centres) * HZ_PER_PPM
    err_true = (peak_positions(spec, p_true, centres) - centres) * HZ_PER_PPM
    return {"mean_official": np.nanmean(np.abs(err_official)), "max_official": np.nanmax(np.abs(err_official)),
            "signed_official": np.nanmean(err_official),
            "mean_true": np.nanmean(np.abs(err_true)), "max_true": np.nanmax(np.abs(err_true)),
            "found": int(np.sum(np.isfinite(err_true))), "n": len(centres),
            "cosine": cosine(spec, bin_average(r_ppm, r_spec, p_official))}


official_errors, exact_errors = {}, {}
for name in MOLECULES:
    ham_q, n, chem = rotating_frame_hamiltonian(name)
    ref = references[assignment[name]]
    official_errors[name] = matching_error(sims[name]["c_xx"], sims[name]["c_yx"], chem, trotter_timestep,
                                           number_of_trottersteps, ref)
    exact_errors[name] = matching_error(*exact_signals_at(ham_q, n, trotter_timestep, number_of_trottersteps),
                                        chem, trotter_timestep, number_of_trottersteps, ref)

print(f"{'':12s} {'peaks':>6s} | {'position error (Hz), mean / max':^34s} | {'cosine with reference':^31s}")
print(f"{'':12s} {'found':>6s} | {'official axis':>15s} {'corrected axis':>17s} | {'simulation':>10s} {'exact':>7s} {'best wrong':>11s}")
for name in MOLECULES:
    e, x = official_errors[name], exact_errors[name]
    wrong = max(score[name, r] for r in ref_names if r != assignment[name])
    print(f"{name:12s} {e['found']:>3d}/{e['n']:<2d} | {e['mean_official']:6.2f} / {e['max_official']:4.2f}"
          f"     {e['mean_true']:6.2f} / {e['max_true']:4.2f}    | {e['cosine']:10.3f} {x['cosine']:7.3f} {wrong:11.3f}")
print("\nExact evolution on the corrected axis (floor set by grid and line width), mean / max:",
      ", ".join(f"{m[-1]} {exact_errors[m]['mean_true']:.2f} / {exact_errors[m]['max_true']:.2f} Hz" for m in MOLECULES))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=100)
xs = np.arange(len(MOLECULES))
bars = [("official axis", [official_errors[m]["mean_official"] for m in MOLECULES], "C3"),
        ("corrected axis", [official_errors[m]["mean_true"] for m in MOLECULES], "C1"),
        ("corrected axis, exact evolution", [exact_errors[m]["mean_true"] for m in MOLECULES], "C2")]
for k, (lab, vals, col) in enumerate(bars):
    axes[0].bar(xs + (k - 1) * 0.26, vals, 0.26, label=lab, color=col)
axes[0].axhline(GRID_HZ, color="k", ls=":", lw=1)
axes[0].text(2.45, GRID_HZ, "grid spacing 1.333 Hz", ha="right", va="bottom", fontsize=8)
axes[0].set_ylabel("mean |position error| (Hz)")
axes[0].set_title("Peak-position error vs reference", fontsize=10)
bars = [("simulation", [official_errors[m]["cosine"] for m in MOLECULES], "C1"),
        ("exact evolution", [exact_errors[m]["cosine"] for m in MOLECULES], "C2"),
        ("best wrong reference", [max(score[m, r] for r in ref_names if r != assignment[m]) for m in MOLECULES], "0.6")]
for k, (lab, vals, col) in enumerate(bars):
    axes[1].bar(xs + (k - 1) * 0.26, vals, 0.26, label=lab, color=col)
axes[1].set_ylabel("cosine similarity with reference")
axes[1].set_ylim(0, 1.18)
axes[1].set_title("Spectrum similarity (noise_strength 5)", fontsize=10)
for ax in axes:
    ax.set_xticks(xs, [f"{m}\n-> {assignment[m]}" for m in MOLECULES])
axes[0].legend(fontsize=7, loc="upper left")
axes[1].legend(fontsize=7, loc="upper center", ncol=3)
fig.tight_layout()
plt.show()"""))

SECTION10 = [
("markdown", r"""## 10. Reducing the errors: circuit parameters (supplementary study)

The official answers above use the challenge's settings: first-order Lie–Trotter, 2.5 ms steps,
300 time points. This section asks how much each circuit parameter reduces the simulation error,
and at what cost in CNOT gates. Nothing here changes the official answers.

Parameters varied, one at a time:
- **Trotter order**: 1 (`LieTrotter`), 2 and 4 (`SuzukiTrotter`);
- **time step** `dt` = 2.5, 1.25, 0.625 and 0.3125 ms, at a fixed signal length of 0.75 s (so the
  number of steps grows from 300 to 2400);
- **order of the Hamiltonian terms** inside one step;
- **signal length** T = 0.75, 1.5 and 3 s.

Error measures:
- **line error**: the exact spectral lines of one Trotter step (its eigenphases), compared with the
  exact lines of H. These give the largest line shift (Hz) and the largest relative error of a
  line's height (%);
- **matching error**: peak-position error and cosine similarity against the reference, as in 8.4.

Cost: CNOT gates of one step (compiled to CX + single-qubit gates on an all-to-all device) times
the number of steps of the deepest circuit.

For speed, every setting is evolved with the exact unitary of its Trotter circuit
(`Operator(step)`), which is the same as applying the circuit step by step. The first cell checks
that against the official simulation."""),

("code", r"""from qiskit import transpile
from qiskit.quantum_info import Operator
from qiskit.synthesis import SuzukiTrotter


def reorder_terms(ham_q, how):
    if how == "file":
        return ham_q
    labels = ham_q.paulis.to_labels()
    if how == "reversed":
        idx = list(range(len(labels)))[::-1]
    else:  # "grouped": all Z terms, then all XX, then all YY, then all ZZ
        kind = lambda s: (sum(ch != "I" for ch in s), [ch for ch in s if ch != "I"][0])
        idx = sorted(range(len(labels)), key=lambda i: kind(labels[i]))
    return SparsePauliOp([labels[i] for i in idx], ham_q.coeffs[idx])


def trotter_step(ham_q, dt, order=1, terms="file"):
    synthesis = LieTrotter() if order == 1 else SuzukiTrotter(order=order)
    qc = QuantumCircuit(ham_q.num_qubits)
    qc.append(PauliEvolutionGate(reorder_terms(ham_q, terms), dt, synthesis=synthesis), range(ham_q.num_qubits))
    return qc.decompose(reps=1)


def cnots_per_step(step_qc):
    compiled = transpile(step_qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
    return compiled.count_ops().get("cx", 0)


def signals_from_step(step_qc, n, steps):
    # Same recipe as section 6 (all 2^N X-basis states, weights = <sum X> at t = 0), evolution by the
    # circuit's unitary.
    unitary = Operator(step_qc).data
    template, phi_n = xsum_eigenstate_template(n)
    bits_n = list(product((0, 1), repeat=n))
    psi = np.array([Statevector(template.assign_parameters({phi_n[i]: np.pi * b[i] for i in range(n)})).data
                    for b in bits_n]).T
    sx = SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(n)], num_qubits=n).to_matrix()
    sy = SparsePauliOp.from_sparse_list([("Y", [i], 1.0) for i in range(n)], num_qubits=n).to_matrix()
    ex, ey = np.zeros((len(bits_n), steps)), np.zeros((len(bits_n), steps))
    for k in range(steps):
        ex[:, k] = np.real(np.einsum("ij,ij->j", psi.conj(), sx @ psi))
        ey[:, k] = np.real(np.einsum("ij,ij->j", psi.conj(), sy @ psi))
        psi = unitary @ psi
    weights = ex[:, 0]
    return weights @ ex, weights @ ey


def spectral_lines(n, ham_q=None, step_qc=None, dt=None):
    # Lines of S = Re FT[C_xx - i C_yx]: frequency (Hz) and height, from H (exact) or from one step.
    sx = SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(n)], num_qubits=n).to_matrix()
    sy = SparsePauliOp.from_sparse_list([("Y", [i], 1.0) for i in range(n)], num_qubits=n).to_matrix()
    if step_qc is None:
        phases, vecs = np.linalg.eigh(ham_q.to_matrix())
        freq = np.subtract.outer(phases, phases) / (2 * np.pi)
    else:
        tri, vecs = schur(Operator(step_qc).data, output="complex")
        phases = -np.angle(np.diag(tri))
        freq = np.subtract.outer(phases, phases) / (2 * np.pi * dt)
        freq = (freq + 0.5 / dt) % (1 / dt) - 0.5 / dt
    height = ((vecs.conj().T @ (sx - 1j * sy) @ vecs) * (vecs.conj().T @ sx @ vecs).T).real
    keep = height > 1e-3 * height.max()
    return freq[keep], height[keep]


def line_error(n, ham_q, step_qc, dt):
    f_exact, h_exact = spectral_lines(n, ham_q=ham_q)
    f_trot, h_trot = spectral_lines(n, step_qc=step_qc, dt=dt)
    nearest = [np.argmin(np.abs(f_trot - f)) for f in f_exact]
    return (np.max(np.abs(f_trot[nearest] - f_exact)), 100 * np.max(np.abs(h_trot[nearest] / h_exact - 1)))


# Check: the official setting through this route equals the section-6/8.1 simulation.
for name in MOLECULES:
    ham_q, n, chem = rotating_frame_hamiltonian(name)
    cxx, cyx = signals_from_step(trotter_step(ham_q, trotter_timestep), n, number_of_trottersteps)
    print(f"{name}: max |C_xx difference| to the official simulation = {np.max(np.abs(cxx - sims[name]['c_xx'])):.1e}, "
          f"C_yx {np.max(np.abs(cyx - sims[name]['c_yx'])):.1e}")"""),

("code", r"""# Sweep: Trotter order x time step at fixed signal length 0.75 s.
T_SIGNAL = number_of_trottersteps * trotter_timestep
sweep = []
for name in MOLECULES:
    ham_q, n, chem = rotating_frame_hamiltonian(name)
    ref = references[assignment[name]]
    for order in (1, 2, 4):
        for dt_ms in (2.5, 1.25, 0.625, 0.3125):
            dt = dt_ms * 1e-3
            steps = int(round(T_SIGNAL / dt))
            step_qc = trotter_step(ham_q, dt, order)
            shift_hz, height_pct = line_error(n, ham_q, step_qc, dt)
            cx = cnots_per_step(step_qc)
            me = matching_error(*signals_from_step(step_qc, n, steps), chem, dt, steps, ref)
            sweep.append({"molecule": name, "order": order, "dt_ms": dt_ms, "steps": steps,
                          "cx_per_step": cx, "cx_total": cx * (steps - 1), "shift_hz": shift_hz,
                          "height_pct": height_pct, "pos_mean_true": me["mean_true"], "cosine": me["cosine"]})

print(f"{'molecule':10s} {'order':>5s} {'dt (ms)':>8s} {'steps':>6s} {'CX/step':>8s} {'CX total':>9s} "
      f"{'max line shift (Hz)':>20s} {'max height error (%)':>21s} {'mean pos. error (Hz)':>21s} {'cosine':>7s}")
for r in sweep:
    print(f"{r['molecule']:10s} {r['order']:5d} {r['dt_ms']:8.4g} {r['steps']:6d} {r['cx_per_step']:8d} {r['cx_total']:9d} "
          f"{r['shift_hz']:20.4f} {r['height_pct']:21.2f} {r['pos_mean_true']:21.2f} {r['cosine']:7.3f}")"""),

("code", r"""fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=100)
styles = {"molecule A": "-", "molecule B": "--", "molecule C": ":"}
for order, color in ((1, "C3"), (2, "C0"), (4, "C2")):
    for name in MOLECULES:
        rows = [r for r in sweep if r["molecule"] == name and r["order"] == order]
        cost = [r["cx_total"] for r in rows]
        axes[0].loglog(cost, [r["shift_hz"] for r in rows], styles[name], marker="o", ms=4, color=color,
                       label=f"order {order}" if name == "molecule A" else None)
        axes[1].loglog(cost, [r["height_pct"] for r in rows], styles[name], marker="o", ms=4, color=color)
        axes[2].semilogx(cost, [r["pos_mean_true"] for r in rows], styles[name], marker="o", ms=4, color=color)
for ax in axes:
    official = [r for r in sweep if r["order"] == 1 and r["dt_ms"] == 2.5]
    ax.set_xlabel("CNOT gates in the deepest circuit")
axes[0].axhline(GRID_HZ, color="k", lw=0.8, ls=":")
axes[0].text(axes[0].get_xlim()[0] * 1.1, GRID_HZ * 1.15, "grid spacing", fontsize=8)
axes[0].set_ylabel("largest line shift vs exact (Hz)")
axes[1].set_ylabel("largest line-height error vs exact (%)")
axes[2].set_ylabel("mean peak-position error vs reference (Hz)\n(corrected axis)")
for name in MOLECULES:
    r = [x for x in sweep if x["molecule"] == name and x["order"] == 1 and x["dt_ms"] == 2.5][0]
    axes[0].plot(r["cx_total"], r["shift_hz"], "k*", ms=11)
    axes[1].plot(r["cx_total"], r["height_pct"], "k*", ms=11)
    axes[2].plot(r["cx_total"], r["pos_mean_true"], "k*", ms=11)
axes[0].plot([], [], "k*", ms=9, label="official setting")
for name, ls in styles.items():
    axes[0].plot([], [], ls, color="0.4", label=name)
axes[0].legend(fontsize=7)
axes[0].set_title("Line positions", fontsize=10)
axes[1].set_title("Line heights", fontsize=10)
axes[2].set_title("Matching error (grid floor remains)", fontsize=10)
fig.tight_layout()
plt.show()"""),

("code", r"""# Order of the terms inside one first-order step (dt = 2.5 ms).
print(f"{'term order':>10s} | " + " | ".join(f"{m}: shift (Hz), height (%)" for m in MOLECULES))
for terms in ("file", "reversed", "grouped"):
    cells = []
    for name in MOLECULES:
        ham_q, n, _ = rotating_frame_hamiltonian(name)
        s, h = line_error(n, ham_q, trotter_step(ham_q, trotter_timestep, 1, terms), trotter_timestep)
        cells.append(f"{s:14.3f}, {h:6.2f}")
    print(f"{terms:>10s} | " + " | ".join(f"{c:>30s}" for c in cells))"""),

("code", r"""# Signal length T (finer frequency grid): official circuit (2.5 ms) and first order with 1.25 ms steps.
print(f"{'setting':34s} {'T (s)':>6s} {'grid (Hz)':>10s} | " + " | ".join(f"{m[-1]}: mean/max pos. err (Hz), cosine" for m in MOLECULES))
length_rows = []
for dt_run in (trotter_timestep, trotter_timestep / 2):
    for t_sig in (0.75, 1.5, 3.0):
        steps = int(round(t_sig / dt_run))
        cells = []
        for name in MOLECULES:
            ham_q, n, chem = rotating_frame_hamiltonian(name)
            me = matching_error(*signals_from_step(trotter_step(ham_q, dt_run, 1), n, steps),
                                chem, dt_run, steps, references[assignment[name]])
            length_rows.append((dt_run, t_sig, name, me))
            cells.append(f"{me['mean_true']:5.2f} / {me['max_true']:4.2f}, {me['cosine']:.3f}")
        label = f"order 1, dt {dt_run * 1e3:g} ms" + (" (official)" if dt_run == trotter_timestep else "")
        print(f"{label:34s} {t_sig:6.2f} {1 / t_sig:10.3f} | " + " | ".join(f"{c:>33s}" for c in cells))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=100)
for dt_run, color in ((trotter_timestep, "C3"), (trotter_timestep / 2, "C0")):
    for name, ls in styles.items():
        rows = [r for r in length_rows if r[0] == dt_run and r[2] == name]
        axes[0].plot([r[1] for r in rows], [r[3]["mean_true"] for r in rows], ls, marker="o", color=color,
                     label=f"dt {dt_run * 1e3:g} ms, {name}")
        axes[1].plot([r[1] for r in rows], [r[3]["cosine"] for r in rows], ls, marker="o", color=color)
axes[0].set_ylabel("mean peak-position error (Hz), corrected axis")
axes[1].set_ylabel("cosine similarity with reference")
axes[0].set_title("Peak positions vs reference", fontsize=10)
axes[1].set_title("Spectrum shape vs reference (noise_strength 5)", fontsize=10)
for ax in axes:
    ax.set_xlabel("signal length T (s), first-order Trotter")
axes[1].legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc="lower right")
fig.tight_layout()
plt.show()"""),

("code", r"""# What the improvement looks like: molecule A's spin-2 multiplet (5.48 ppm), signal length 0.75 s.
name = "molecule A"
ham_q, n, chem = rotating_frame_hamiltonian(name)
r_ppm, r_spec = references[assignment[name]]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=100)
for ax, gamma in zip(axes, (5.0, tolerance[name, "10 % (primary)"][0])):
    keep = np.abs(r_ppm - 5.479) < 0.06
    ax.plot(r_ppm[keep], r_spec[keep], color="0.6", lw=1, label="reference (spectrum_3)")
    for label, cxx_cyx, color in (
        ("exact evolution", exact_signals_at(ham_q, n, trotter_timestep, number_of_trottersteps), "C2"),
        ("first order (official)", (sims[name]["c_xx"], sims[name]["c_yx"]), "C3"),
        ("first order, dt 1.25 ms (600 steps)", signals_from_step(trotter_step(ham_q, trotter_timestep / 2), n,
                                                                  2 * number_of_trottersteps), "C0"),
    ):
        dt_run = trotter_timestep / 2 if len(cxx_cyx[0]) == 2 * number_of_trottersteps else trotter_timestep
        spec, _, p_true = spectrum_general(*cxx_cyx, gamma, dt_run, len(cxx_cyx[0]), chem)
        keep_s = np.abs(p_true - 5.479) < 0.06
        ax.plot(p_true[keep_s], spec[keep_s], marker=".", ms=4, color=color, label=label)
    ax.set_title(f"noise_strength {gamma:g} s$^{{-1}}$ (corrected axis)", fontsize=10)
    ax.set_xlabel("Frequency [ppm]")
    ax.legend(fontsize=7)
axes[0].set_ylabel("Spectrum")
fig.tight_layout()
plt.show()

def scan_general(c_xx, c_yx, chem_shift, dt, steps, centres, level=0.10):
    # Same criterion and search as section 9.1 (0.1 steps, then 0.01 where the result changes), any (dt, steps).
    def ok(g):
        spec, p_official, _ = spectrum_general(c_xx, c_yx, g, dt, steps, chem_shift)
        return resolved(spec, p_official, centres, level)[0]
    coarse = np.round(np.arange(0.1, 60.0 + 1e-9, 0.1), 2)
    flags = np.array([ok(g) for g in coarse])
    best = coarse[flags].max()
    fine = np.round(np.arange(best, best + 0.1 + 1e-9, 0.01), 2)
    return max(g for g in fine if ok(g)), int(np.sum(flags[1:] != flags[:-1]))


# Noise thresholds (10 % criterion), signal length 0.75 s.
print("max tolerable noise_strength (10 % dip):  official (dt 2.5 ms) | first order, dt 1.25 ms | exact evolution")
for name in MOLECULES:
    ham_q, n, chem = rotating_frame_hamiltonian(name)
    centres, _ = characteristic_peaks(*references[assignment[name]])
    half = scan_general(*signals_from_step(trotter_step(ham_q, trotter_timestep / 2), n, 2 * number_of_trottersteps),
                        chem, trotter_timestep / 2, 2 * number_of_trottersteps, centres)
    exact = scan_general(*exact_signals_at(ham_q, n, trotter_timestep, number_of_trottersteps), chem,
                         trotter_timestep, number_of_trottersteps, centres)
    print(f"  {name}: {tolerance[name, '10 % (primary)'][0]:20.2f} | {half[0]:23.2f} | {exact[0]:15.2f}"
          f"   (pass/fail changes: {half[1]}, {exact[1]})")"""),
]

SECTION10[4:4] = [
("markdown", r"""### 10.0 The circuits behind the plot (molecule A, 3 qubits, no ancillas)

The drawings go from the big picture to the individual gates:

1. **Blocks: one measured circuit.** Prepare an X-basis state, apply the Trotter step k times,
   measure ⟨ΣX⟩ and ⟨ΣY⟩. The notebook runs this for k = 0 … 299 and for all 8 starting states.
2. **Blocks: one first-order Trotter step.** One block of chemical-shift rotations, then one
   coupling block for each pair of protons.
3. **Gates: the same step as the notebook builds it.** RZ on each qubit; RXX, RYY, RZZ on each pair.
4. **Where the CNOTs come from**: each RXX, RYY or RZZ is translated into CNOT, RZ, CNOT (plus basis
   changes), so one step has 18 CNOTs. The level-3 transpiler merges the three rotations of a pair
   into one block of 3 CNOTs, which gives 9 per step. The CNOT counts on the x-axis of the plot above
   use level 1 (18 per step).
5. **Gates: one second-order step**, a half step forward and then the same half step in reverse order.
6. **Blocks: one fourth-order step**, which is five second-order steps (why it costs about 5× more).
7. **The improved circuit**: the same moment in time built from twice as many steps of half the size.
   The additional CNOTs come only from the extra repetitions; drawing 7c shows that each smaller step
   has the same gates with half the angles.

Each block drawing is checked against the gate-level circuit: their unitaries must agree."""),
("code", r"""from IPython.display import display
from qiskit import QuantumRegister


def on_protons(qc):
    # Same circuit, wires named H_0, H_1, ... after the protons they represent.
    out = QuantumCircuit(QuantumRegister(qc.num_qubits, "H"), global_phase=qc.global_phase)
    return out.compose(qc)


def show_circuit(qc, title, fold=28, scale=0.7):
    # Title as a printed line, then the drawing (matplotlib needs the optional package pylatexenc;
    # without it the circuit is printed as text).
    print("\n" + title)
    try:
        fig = on_protons(qc).draw("mpl", fold=fold, scale=scale)
        display(fig)
        plt.close(fig)
    except Exception:
        print(on_protons(qc).draw("text", fold=110))


def same_unitary(a, b):
    return np.max(np.abs(Operator(a).data - Operator(b).data))


name = "molecule A"
ham_q, n, _ = rotating_frame_hamiltonian(name)
coef = dict(zip(ham_q.paulis.to_labels(), np.real(ham_q.coeffs)))
label_of = lambda ops: "".join(ops.get(q, "I") for q in reversed(range(n)))  # Qiskit order: qubit 0 last
dt = trotter_timestep
step1 = trotter_step(ham_q, dt, 1)

# 1. Blocks: the measured circuit for time point t = 3 dt.
template, phi_n = xsum_eigenstate_template(n)
full = QuantumCircuit(n)
full.append(template.to_gate(label="X-basis state"), range(n))
for k in range(3):
    full.append(step1.to_gate(label=f"Trotter step {k + 1}"), range(n))
full.append(QuantumCircuit(n).to_gate(label="measure ΣX, ΣY"), range(n))
show_circuit(full, "1. BLOCKS - the circuit for time point t = 3 dt "
                   "(the notebook runs k = 0 ... 299 Trotter steps for each of the 8 starting states)")

# 2. Blocks: one first-order step.
blocks = QuantumCircuit(n)
shifts = QuantumCircuit(n)
for q in range(n):
    shifts.rz(2 * coef[label_of({q: "Z"})] * dt, q)
blocks.append(shifts.to_gate(label="shift RZ"), range(n))
for k in range(n):
    for l in range(k + 1, n):
        c = coef[label_of({k: "X", l: "X"})]
        pair = QuantumCircuit(2)
        pair.rxx(2 * c * dt, 0, 1)
        pair.ryy(2 * c * dt, 0, 1)
        pair.rzz(2 * c * dt, 0, 1)
        blocks.append(pair.to_gate(label=f"J{k}{l}={2 * c / np.pi:.1f}Hz"), [k, l])
show_circuit(blocks, f"2. BLOCKS - one first-order Trotter step: chemical-shift rotations, then one block per "
                     f"coupled pair (same unitary as drawing 3, difference {same_unitary(blocks, step1):.0e})")

# 3. Gates: the step the notebook uses.
show_circuit(step1, "3. GATES - the same step as the notebook builds it: 3 RZ + 3 pairs x (RXX, RYY, RZZ); "
                    "angle = 2 x coefficient x dt")

# 4. Where the CNOTs come from.
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate

theta = 2 * coef[label_of({1: "X", 2: "X"})] * dt  # the H1-H2 angle, as an example
rules = QuantumCircuit(2)
for gate in (RZZGate(theta), RXXGate(theta), RYYGate(theta)):
    rules.barrier(label=f"{gate.name.upper()} =")
    rules.compose(gate.definition, inplace=True)
show_circuit(rules, "4a. WHERE THE CNOTs COME FROM - each two-qubit rotation is translated into "
                    "CNOT, RZ(angle), CNOT; RXX and RYY first turn the X or Y axis into Z (H or RX gates) "
                    "and back. So every RXX, RYY, RZZ costs exactly 2 CNOTs", fold=40)

expanded = QuantumCircuit(n)
for inst in step1.data:
    qubits = [step1.find_bit(q).index for q in inst.qubits]
    if len(qubits) == 1:
        expanded.append(inst.operation, qubits)
        continue
    expanded.barrier(label=f"{inst.operation.name.upper()} H{qubits[0]}-H{qubits[1]}")
    expanded.compose(inst.operation.definition, qubits, inplace=True)
show_circuit(expanded, f"4b. THE OFFICIAL STEP WITH EVERY ROTATION TRANSLATED - each labelled section is one "
                       f"rotation of drawing 3 and holds its 2 CNOTs: 3 pairs x 3 rotations x 2 = "
                       f"{expanded.count_ops().get('cx', 0)} CNOTs per step (difference to drawing 3: "
                       f"{same_unitary(expanded, step1):.0e})", fold=34, scale=0.6)

pair = QuantumCircuit(2)
pair.rxx(theta, 0, 1)
pair.ryy(theta, 0, 1)
pair.rzz(theta, 0, 1)
fused = transpile(pair, basis_gates=["cx", "rz", "sx", "x"], optimization_level=3, seed_transpiler=1)
show_circuit(fused, f"4c. THE LEVEL-3 SHORTCUT - for one proton pair, RXX, RYY and RZZ commute, so the "
                    f"transpiler merges them into one two-qubit block. Any two-qubit block needs at most 3 "
                    f"CNOTs: {fused.count_ops().get('cx', 0)} CNOTs instead of 6 for the pair")

compiled3 = transpile(step1, basis_gates=["cx", "rz", "sx", "x"], optimization_level=3, seed_transpiler=1)
phase_free = abs(abs(np.trace(Operator(compiled3).data.conj().T @ Operator(step1).data)) / 2**n - 1)
show_circuit(compiled3, f"4d. THE WHOLE STEP AT LEVEL 3 - 3 pairs x 3 CNOTs = {compiled3.count_ops().get('cx', 0)} "
                        f"CNOTs per step instead of 18 (same unitary up to a global phase, difference {phase_free:.0e})",
             fold=40, scale=0.6)

# 5. Gates: second-order step.
step2 = trotter_step(ham_q, dt, 2)
show_circuit(step2, "5. GATES - one second-order step: every rotation for dt/2, then the same rotations in "
                    "reverse order (the two middle ZZ gates merge into one)", fold=24)

# 6. Blocks: fourth-order step = 5 second-order steps.
p = 1 / (4 - 4 ** (1 / 3))
step4 = QuantumCircuit(n)
for frac, lab in ((p, "p"), (p, "p"), (1 - 4 * p, "1-4p"), (p, "p"), (p, "p")):
    step4.append(trotter_step(ham_q, frac * dt, 2).to_gate(label=f"2nd order {lab}"), range(n))
show_circuit(step4, f"6. BLOCKS - one fourth-order step = 5 second-order steps of p dt, p dt, (1-4p) dt, p dt, p dt "
                    f"with p = {p:.4f}; 1-4p = {1 - 4 * p:.4f} is a step backwards in time "
                    f"(same unitary as Qiskit's SuzukiTrotter(order=4), difference "
                    f"{same_unitary(step4, trotter_step(ham_q, dt, 4)):.0e})")

# 7. The improved circuit: additional CNOTs that reduce the line shift.
half_step = trotter_step(ham_q, dt / 2, 1)
cx_full, cx_half = cnots_per_step(step1), cnots_per_step(half_step)
cx_half_l3 = transpile(half_step, basis_gates=["cx", "rz", "sx", "x"], optimization_level=3,
                       seed_transpiler=1).count_ops().get("cx", 0)
shift_full, height_full = line_error(n, ham_q, step1, dt)
shift_half, height_half = line_error(n, ham_q, half_step, dt / 2)

official_t = QuantumCircuit(n)
official_t.append(template.to_gate(label="X-basis state"), range(n))
for k in range(2):
    official_t.append(step1.to_gate(label=f"2.5 ms step: {cx_full} CX"), range(n))
show_circuit(official_t, f"7a. OFFICIAL CIRCUIT for time point t = 5 ms: 2 steps of 2.5 ms = 2 x {cx_full} = "
                         f"{2 * cx_full} CNOTs. Largest line shift {shift_full:.2f} Hz, line heights off by up to "
                         f"{height_full:.1f} %")

improved_t = QuantumCircuit(n)
improved_t.append(template.to_gate(label="X-basis state"), range(n))
for k in range(4):
    improved_t.append(half_step.to_gate(label=f"1.25 ms step: {cx_half} CX"), range(n))
show_circuit(improved_t, f"7b. IMPROVED CIRCUIT for the same time point t = 5 ms: 4 steps of 1.25 ms = 4 x {cx_half} = "
                         f"{4 * cx_half} CNOTs, i.e. {4 * cx_half - 2 * cx_full} ADDITIONAL CNOTs. Largest line shift "
                         f"{shift_half:.2f} Hz ({shift_full / shift_half:.1f}x smaller), line heights off by up to "
                         f"{height_half:.1f} %")

show_circuit(half_step, "7c. GATES of one 1.25 ms step: exactly the gates of drawing 3, every angle halved "
                        "(compare the numbers under RZ, RXX, RYY, ZZ)")

print(f"\nWhole signal (deepest circuit): official 299 x {cx_full} = {299 * cx_full} CNOTs; improved 599 x {cx_half} = "
      f"{599 * cx_half} CNOTs; improved with the level-3 shortcut 599 x {cx_half_l3} = {599 * cx_half_l3} CNOTs, "
      f"about the same as the official circuit but with a {shift_full / shift_half:.1f}x smaller line shift.")"""),
]

SECTION10.append(("markdown", r"""### 10.1 Findings of the error study

**Matching errors at the official setting** (section 8.4, `noise_strength` 5):

| Molecule | Mean / max peak error, official axis | Same, corrected axis | Exact-evolution floor | Cosine: simulation / exact / best wrong reference |
|---|---|---|---|---|
| A → spectrum_3 | 0.73 / 1.48 Hz | 0.56 / 0.93 Hz | 0.10 / 0.24 Hz | 0.804 / 0.875 / 0.005 |
| B → spectrum_2 | 0.83 / 1.59 Hz | 0.34 / 0.66 Hz | 0.17 / 0.26 Hz | 0.869 / 0.858 / 0.004 |
| C → spectrum_1 | 0.67 / 0.92 Hz | 0.12 / 0.21 Hz | 0.13 / 0.18 Hz | 0.843 / 0.843 / 0.000 |

The errors come from three sources:
1. the frequency labels (0.4–0.9 Hz too low; a labelling issue, not a physics issue);
2. first-order Trotter error (only A and B);
3. the coarse 1.333 Hz grid with lines wider than the reference's (this limits the cosine for all
   three, even with exact evolution).

None of them comes close to changing the matching: every wrong pairing scores at most 0.005.

**Which circuit parameter helps** (molecule A, the largest error):

| Change | Largest line shift | Largest line-height error | Mean peak error vs reference | CNOTs (deepest circuit) |
|---|---|---|---|---|
| official: order 1, 2.5 ms, 300 steps | 0.72 Hz | 19.2 % | 0.56 Hz | 5,382 |
| order 1, 1.25 ms, 600 steps | 0.14 Hz | 3.7 % | **0.12 Hz** (floor 0.10) | 10,782 (2×) |
| order 1, 0.3125 ms, 2400 steps | 0.008 Hz | 0.2 % | 0.12 Hz | 43,182 (8×) |
| order 2, 2.5 ms | 0.72 Hz | 11.6 % | 0.56 Hz | 10,166 (1.9×) |
| order 4, 2.5 ms | 0.011 Hz | 1.1 % | 0.10 Hz | 50,830 (9.4×) |
| other order of the terms in one step | 0.71 Hz | 17.7 % (best) | — | same |

- **Smaller steps of the first-order formula are the cheapest fix.** At this step size the
  first-order line shifts already shrink like dt², so second order does not move the lines at all.
  Fourth order is accurate but about 10× more expensive, and at that cost first order with smaller
  steps is at least as accurate.
- **A longer signal improves the shape match**, because the frequency grid becomes finer. With
  1.25 ms steps the cosine rises from 0.874 (0.75 s) to 0.950 (1.5 s) and 0.971 (3 s) for A, to
  0.969 for B and 0.961 for C. Mean peak errors fall to 0.12, 0.05 and 0.01 Hz.
- **Noise thresholds** with 1.25 ms steps: A 20.92, B 23.57, C 23.11 s⁻¹. These are within 0.3 s⁻¹
  of exact evolution (21.23, 23.48, 23.09).

**Recommended for a more accurate run:** first-order Lie–Trotter with 1.25 ms steps (2× the
CNOTs), plus a longer signal (1.5–3 s) if the shape match matters. The challenge answers above
stay at the official setting (300 steps of 2.5 ms)."""))

SECTION11 = [
("markdown", r"""## 11. Circuit-depth reduction and real IBM hardware results

Both parts load saved results, so this section reruns **without IBM credentials** and spends no
quantum time:
- **Circuit reduction:** `results/phase1/candidates.csv`, produced by `src/phase1_benchmark.py`.
- **Hardware:** the untouched outputs of the 18 IBM jobs, `results/phase2/raw/*_result.json`.
  - Backend: `ibm_kingston`, 390 of 600 s used.
  - The circuits that ran are archived in `results/phase2/circuits/`, with the noise model for the
    simulator comparison.

Full details are in `nmr_optimized.ipynb`, `docs/phase1_circuit_optimization.md` and
`results/phase2/`."""),
("code", r"""import sys
import pandas as pd
from IPython.display import display
sys.path.insert(0, "src")

cand = pd.read_csv("results/phase1/candidates.csv")
cand["molecule"] = cand["molecule"].str[-1]
print("Circuit-depth reduction (deepest circuit; accuracy measured against exact evolution):")
display(cand[["candidate", "molecule", "deep_depth", "deep_twoq_depth", "deep_cx", "circuits_on_hardware",
              "E3_peak_pos_hz", "E2_cosine", "accepted"]].rename(columns={
    "deep_depth": "depth", "deep_twoq_depth": "2q depth", "deep_cx": "CNOTs", "circuits_on_hardware": "circuits",
    "E3_peak_pos_hz": "peak error (Hz)", "E2_cosine": "cosine vs exact"}).round(4))
best = cand[cand["candidate"].str.startswith("C6")].set_index("molecule")
base = cand[cand["candidate"].str.startswith("C0")].set_index("molecule")
for m in "ABC":
    print(f"molecule {m}: CNOTs {base.loc[m, 'deep_cx']} -> {best.loc[m, 'deep_cx']}, depth {base.loc[m, 'deep_depth']} -> "
          f"{best.loc[m, 'deep_depth']}, circuits {base.loc[m, 'circuits_on_hardware']} -> {best.loc[m, 'circuits_on_hardware']}")"""),
("code", r"""import phase2_analysis as P2

hw_rows, hw_spectra, hw_signals, hw_matching = P2.analyse(recompute_noisy_sim=False, bootstrap=50)
hw = pd.DataFrame(hw_rows)
hw["molecule"] = hw["molecule"].str[-1]
print("REAL IBM HARDWARE (ibm_kingston) vs ideal, noise_strength 5:")
display(hw[["molecule", "data", "cosine_vs_ideal", "rmse_vs_ideal", "peaks_found", "max_peak_shift_hz",
            "max_peak_height_err_pct", "qpu_seconds"]].round(4))
for label, m in hw_matching.items():
    print(f"matching from {label} alone: {m['assignment']} (total {m['best_total']} vs {m['second_best_total']} for the runner-up)")
fig = P2.plot_spectra(hw_spectra)
plt.show()"""),
]
