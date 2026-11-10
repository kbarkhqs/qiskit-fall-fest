"""Phase 1 circuit optimisation: correctness of the models, the circuits and the symmetry reduction."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import circuit_opt as C  # noqa: E402

MOLS = ("molecule A", "molecule B", "molecule C")
FORMULAS = ("lie", "strang", "strang_alt", "suzuki2", "pairfield")


@pytest.mark.parametrize("name", MOLS)
def test_lie_model_reproduces_official_signals(name):
    model = C.spin_model(name)
    off = np.load(ROOT / f"results/signals/molecule_{name[-1]}.npz")
    cxx, cyx = C.signals(C.time_point_unitaries("lie", model), model)
    assert np.max(np.abs(cxx - off["c_xx_raw"])) < 1e-10
    assert np.max(np.abs(cyx - off["c_yx_raw"])) < 1e-10


@pytest.mark.parametrize("name", MOLS)
def test_prod_z_symmetry_halves_the_initial_states_exactly(name):
    model = C.spin_model(name)
    n = model["n"]
    q = SparsePauliOp("Z" * n).to_matrix()
    assert np.max(np.abs(q @ model["H"] @ q - model["H"])) == 0
    units = C.time_point_unitaries("lie", model)
    full, half = C.signals(units, model), C.signals(units, model, half=True)
    assert len(C.x_states(n, half=True)[0]) == (4 if n == 3 else 1)
    assert max(np.max(np.abs(a - b)) for a, b in zip(full, half)) < 1e-12


@pytest.mark.parametrize("name", MOLS)
@pytest.mark.parametrize("formula", FORMULAS)
def test_circuit_equals_model(name, formula):
    model = C.spin_model(name)
    assert C.check_model_vs_circuit({"formula": formula}, model, ks=(1, 2, 5)) < 1e-10


@pytest.mark.parametrize("name", ("molecule A", "molecule B"))
def test_swap_network_gives_the_same_measured_values(name):
    model = C.spin_model(name)
    assert C.check_model_vs_circuit({"formula": "strang_alt", "swap_network": True}, model, ks=(1, 2, 5)) < 1e-10


@pytest.mark.parametrize("name", MOLS)
def test_grouped_blocks_are_an_exact_rewrite_of_the_baseline(name):
    model = C.spin_model(name)
    for k in (1, 3, 10):
        a = Operator(C.evolution_circuit({"formula": "lie", "style": "rotations"}, model, k)).data
        b = Operator(C.evolution_circuit({"formula": "lie", "style": "grouped"}, model, k)).data
        assert np.max(np.abs(a - b)) < 1e-10


@pytest.mark.parametrize("name,max_cx", (("molecule A", 20), ("molecule B", 20), ("molecule C", 3)))
def test_exact_compilation_is_exact_and_shallow(name, max_cx):
    model = C.spin_model(name)
    for k in (1, 150, 299):
        compiled = C.transpile_for(C.evolution_circuit({"formula": "exact"}, model, k), 3)
        assert compiled.count_ops().get("cx", 0) <= max_cx
        u = Operator(compiled).data
        target = expm(-1j * model["H"] * k * C.DT)
        overlap = abs(np.trace(target.conj().T @ u)) / 2 ** model["n"]
        assert abs(overlap - 1) < 1e-10


def test_shifts_folded_into_the_block_is_exact_for_two_spins():
    model = C.spin_model("molecule C")
    a = C.time_point_unitaries("pairfield", model)
    b = C.time_point_unitaries("exact", model)
    assert max(np.max(np.abs(x - y)) for x, y in zip(a, b)) < 1e-10


def test_acceptance_rules_were_written_and_benchmark_selects_exact_compilation():
    rules = json.loads((ROOT / "results/phase1/acceptance_criteria.json").read_text())
    assert "accept_rule" in rules
    import csv
    rows = list(csv.DictReader(open(ROOT / "results/phase1/candidates.csv")))
    exact = [r for r in rows if r["candidate"].startswith("C6")]
    assert len(exact) == 3 and all(r["accepted"] == "True" and r["high_accuracy"] == "True" for r in exact)
