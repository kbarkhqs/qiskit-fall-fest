# Agent 2 — quantum software implementation

Compiled by the orchestrator from Agent 2's recorded outputs: Agent 2 hit a usage limit after
finishing the work but before writing this file. Every number below is read from the JSON or log
files in this folder; `evidence.json` lists the experiments.

## Notebook changes (`nmr.ipynb`, 32 -> 33 cells)

| Cell (new index) | Change |
|---|---|
| 14 | TODO completed. Template: `H` then `RZ(phi[q])` on every qubit (phi = 0 -> \|+>, phi = pi -> \|-> up to a global phase). `bits_list = list(product((0, 1), repeat=number_spins))` (2^N tuples). `param_maps = [{phi[i]: pi * bits[i] ...}]` binds every parameter. Names and interface unchanged. |
| 15 (new) | Statevector check with assertions: 2^N states, every map binds all N parameters, each qubit's `<X_i>` = 1 - 2 bit_i, zero variance of `sum X`, eigenvalue N - 2w, Gram matrix = identity, all-plus = +N, all-minus = -N. Prints the eigenvalue table. |
| 25 | `time_signal_to_spectrum`: maths unchanged; a guard raises `ValueError` if `max|spectrum|` is 0 or not finite (the upstream 0/0 `RuntimeWarning`). **Changed by the orchestrator after the independent review**: the guard also rejects a signal scale `max(|c_xx|, |c_yx|) < 1e-8`, because equal weights give 8.9e-16 rounding noise, which an `== 0` test missed (new test `test_guard_raises_on_equal_weights_rounding_noise`). |
| 29 | Reference normalisation no longer modifies the loaded array in place. |

No other cell source changed by Agent 2. The orchestrator later added 13 cells to sections 8 and 9
(tagged `metadata.added_by = "orchestrator"`; 46 cells in total) and re-executed the notebook. Defaults kept (molecule A, 300 steps, dt = 0.0025 s, noise 5.0,
`spectrum_3.pkl`). The executed notebook has no errors and no warnings in any output.

## Results

| Check | A | B | C |
|---|---|---|---|
| C_xx(0) (expected N 2^N) | 24.000 | 24.000 | 8.000 |
| C_yx(0) (expected 0) | -1.5e-15 | -1.5e-15 | -4.9e-16 |
| max \|m_n - (N - 2w)\| | 1.3e-15 | — | 8.9e-16 |
| estimator vs fast, max diff | 0.0 | 0.0 | 0.0 |
| wall time estimator / fast (s) | 424.6 / 2.37 | 423.7 / 2.32 | 91.4 / 0.70 |

- Template equivalence: `H RZ(phi)` and `RX(phi) H` give identical amplitudes for every angle tested.
- Notebook vs module (molecule A, noise 5): max difference <= 8.9e-16; kernel = `.venv/bin/python`.
- Mutation check: the upstream placeholder makes every detector fire (1 state instead of 8,
  C_xx = 0, guard raises); the fix passes all of them.
- `pytest`: 67 passed in 5.57 s.
- Agent 2 also measured the frequency-axis offset independently (`exp05_frequency_axis_offset.json`):
  lines sit 0.67-0.92 Hz low on the notebook axis; the derived axis `(2j-(L-1))/((2L-1) dt)`
  removes it to 1e-4 Hz.

## Files
`src/nmr_pipeline.py` (verbatim mirror + `simulate_signals_fast`, `coupling_overrides`),
`tests/`, `results/signals/molecule_{A,B,C}.npz`, scripts and JSON logs in this folder.
