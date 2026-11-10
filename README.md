# Matching molecules to NMR spectra with Trotterized time evolution

Qiskit Fall Fest 2026 hackathon (intermediate level).

- 📄 **Report:** [report.pdf](report.pdf)
- 📊 **Slides:** [slides.pdf](slides.pdf)

## Contents

| File | Purpose |
| --- | --- |
| `nmr.ipynb` | Original hackathon task |
| `nmr_sim.py` | Ideal (statevector) NMR spectrum simulation |
| `noise_search.py` | Matches molecules A/B/C to reference spectra and finds the max tolerable noise |
| `fake_backend_sim.py` | Noisy simulation on an IBM fake backend (e.g. FakeKingston) |
| `hardware_jobs.py` | Submit / check / fetch runs on IBM Quantum hardware |
| `files/` | Hamiltonians and reference spectra |
| `results/`, `figures/` | Outputs and plots used in the report |

Each script has usage examples in its docstring, e.g.

```bash
python noise_search.py
python fake_backend_sim.py "molecule A"
python hardware_jobs.py submit "molecule A" --steps 60 --dt 0.003 --shots 500
```

Hardware runs need `IBM_QUANTUM_API_KEY` in `.env`.
