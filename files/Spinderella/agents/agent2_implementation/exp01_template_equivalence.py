"""Experiment 01: are the two candidate X-eigenstate templates the same state?

(a) H then RZ(phi)      (b) RX(phi) then H      (p) supplied placeholder: RZ(phi) on |0>

Run from the repository root:
    .venv/bin/python agents/agent2_implementation/exp01_template_equivalence.py
"""
import json
from pathlib import Path

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, Statevector

OUT = Path(__file__).with_name("exp01_template_equivalence.json")


def circ_a(angle):
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(angle, 0)
    return qc


def circ_b(angle):
    qc = QuantumCircuit(1)
    qc.rx(angle, 0)
    qc.h(0)
    return qc


def circ_placeholder(angle):
    qc = QuantumCircuit(1)
    qc.rz(angle, 0)
    return qc


plus = np.array([1, 1]) / np.sqrt(2)
minus = np.array([1, -1]) / np.sqrt(2)
rng = np.random.default_rng(20261110)
generic = [float(x) for x in rng.uniform(-2 * np.pi, 2 * np.pi, size=5)]
angles = [("0", 0.0), ("pi", float(np.pi))] + [(f"generic_{i}", g) for i, g in enumerate(generic)]

rows = []
for name, ang in angles:
    sa = Statevector(circ_a(ang))
    sb = Statevector(circ_b(ang))
    sp = Statevector(circ_placeholder(ang))
    rows.append(
        {
            "angle_name": name,
            "angle_rad": ang,
            # exact vector difference (includes the global phase)
            "max_abs_amp_diff_a_minus_b": float(np.max(np.abs(sa.data - sb.data))),
            "fidelity_a_b": float(abs(np.vdot(sa.data, sb.data)) ** 2),
            "a_overlap_plus_sq": float(abs(np.vdot(plus, sa.data)) ** 2),
            "a_overlap_minus_sq": float(abs(np.vdot(minus, sa.data)) ** 2),
            "a_expX": float(sa.expectation_value(Pauli("X")).real),
            "a_expY": float(sa.expectation_value(Pauli("Y")).real),
            "b_expX": float(sb.expectation_value(Pauli("X")).real),
            "b_expY": float(sb.expectation_value(Pauli("Y")).real),
            "expected_expX_cos": float(np.cos(ang)),
            "expected_expY_sin": float(np.sin(ang)),
            "placeholder_expX": float(sp.expectation_value(Pauli("X")).real),
            "placeholder_overlap_zero_sq": float(abs(sp.data[0]) ** 2),
        }
    )

# Operator identity H . RX(phi) = RZ(phi) . H  (circuit (b) and circuit (a) as unitaries)
op_diffs = []
for _, ang in angles:
    op_diffs.append(float(np.max(np.abs(Operator(circ_a(ang)).data - Operator(circ_b(ang)).data))))

# global phase of (a) at phi = pi relative to |->
sa_pi = Statevector(circ_a(np.pi)).data
phase_pi = complex(np.vdot(minus, sa_pi))

summary = {
    "max_abs_amp_diff_a_minus_b_over_all_angles": max(r["max_abs_amp_diff_a_minus_b"] for r in rows),
    "min_fidelity_a_b": min(r["fidelity_a_b"] for r in rows),
    "max_abs_unitary_diff_a_minus_b": max(op_diffs),
    "a_at_0_is_plus_fidelity": rows[0]["a_overlap_plus_sq"],
    "a_at_pi_is_minus_fidelity": rows[1]["a_overlap_minus_sq"],
    "a_at_pi_global_phase_vs_minus": [phase_pi.real, phase_pi.imag],
    "max_abs_expX_minus_cos_a": max(abs(r["a_expX"] - r["expected_expX_cos"]) for r in rows),
    "max_abs_expY_minus_sin_a": max(abs(r["a_expY"] - r["expected_expY_sin"]) for r in rows),
    "placeholder_max_abs_expX": max(abs(r["placeholder_expX"]) for r in rows),
    "placeholder_min_overlap_zero_sq": min(r["placeholder_overlap_zero_sq"] for r in rows),
}
OUT.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
for r in rows:
    print(
        f"{r['angle_name']:>10s} phi={r['angle_rad']:+.6f}  |a-b|max={r['max_abs_amp_diff_a_minus_b']:.2e}"
        f"  F(a,b)={r['fidelity_a_b']:.15f}  <X>a={r['a_expX']:+.12f}  <X>b={r['b_expX']:+.12f}"
        f"  <X>placeholder={r['placeholder_expX']:+.1e}"
    )
print(json.dumps(summary, indent=2))
