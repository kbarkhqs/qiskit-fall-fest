# Team Spinderella: HQS NMR challenge (Qiskit Fall Fest 2026)

**Presentation:** `Spinderella_NMR_presentation.pdf`

## Results

| Molecule | Spectrum | Max tolerable `noise_strength` (10 % dip rule) | Limiting gap |
|---|---|---|---|
| A | spectrum_3 | 19.97 s⁻¹ | 6.05 Hz = J12 − J02 |
| B | spectrum_2 | 22.85 s⁻¹ | 7.02 Hz = J01 − J12 |
| C | spectrum_1 | 23.22 s⁻¹ | 7.92 Hz = J01 |

- **Initial states (the TODO):** all 2^N X-basis states (H then RZ(0 or π)), each weighted by its ΣX
  eigenvalue N − 2w.
- **J coupling:** the noise limit follows the smallest gap between lines, not the J strength.
  Raising J02 in molecule A lowers the limit.
- **Circuit reduction** (`nmr_optimized.ipynb`, `docs/phase1_circuit_optimization.md`):
  - the deepest circuit drops from 5,382 to **19 CNOTs** (exact compilation of e^(−iHt); 3 CNOTs for
    molecule C);
  - the ΠZ symmetry halves the number of circuits.
- **Real IBM hardware** (`ibm_kingston`; 18 jobs; 390 s; raw + readout/twirling/decoupling mitigation):
  - spectra match the ideal ones with cosine 0.978 / 0.988 / 0.984;
  - all characteristic peaks are found;
  - the matching from hardware data alone is A→3, B→2, C→1.

## Rerun (no IBM credentials needed)

Run from this folder:

```bash
pip install "qiskit>=2.2,<3" struqture_py numpy scipy matplotlib pandas qiskit-aer pylatexenc nbconvert ipykernel pytest
jupyter nbconvert --to notebook --execute nmr.ipynb            # challenge + sections 8-11 (about 5 min)
jupyter nbconvert --to notebook --execute nmr_optimized.ipynb  # circuit-depth study (about 2 min)
python -m pytest -q                                            # 99 tests
```

- The hardware section reads the saved IBM outputs (`results/phase2/raw/`) and the archived circuits
  and noise model (`results/phase2/circuits/`, `results/phase2/kingston_noise_model.pkl`).
- IBM's twirled circuit templates are not included because of their size. They can be retrieved with
  the job IDs in `results/phase2/raw/*_submitted.json`.
- Validation and independent review: `docs/validation_report.md`, `agents/final_audit/AUDIT_REPORT.md`.
