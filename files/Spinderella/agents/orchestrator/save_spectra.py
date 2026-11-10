"""Save every individual spectrum as data and as a plot (Phase 5 deliverable).

Simulated: official cached signals (results/signals, 300 Lie-Trotter steps) through the verbatim
notebook pipeline, at the notebook default noise_strength 5.0 and at each molecule's threshold.
Reference: the supplied pickles, normalised to their maximum, unchanged otherwise.
"""
import pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.nmr_pipeline as P  # noqa: E402

out, plots = ROOT / "results" / "spectra", ROOT / "results" / "plots"
out.mkdir(exist_ok=True)
thresholds = {"A": 19.97, "B": 22.85, "C": 23.22}  # results/noise_tolerance.csv
for m in "ABC":
    d = np.load(ROOT / f"results/signals/molecule_{m}.npz")
    cols, header = [], []
    for g in (5.0, thresholds[m]):
        s, ppm = P.spectrum_from_signals(d["c_xx_raw"], d["c_yx_raw"], g, float(d["trotter_timestep"]),
                                         int(d["number_of_trottersteps"]), float(d["chemical_shift_hz"]))
        ppm = np.asarray(ppm, dtype=float)
        if not cols:
            cols.append(ppm); header.append("ppm")
        cols.append(s); header.append(f"spectrum_noise_{g:g}")
    np.savetxt(out / f"simulated_molecule_{m}.csv", np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="", fmt="%.8g")
    fig, ax = plt.subplots(figsize=(7, 3), dpi=110)
    ax.plot(cols[0], cols[1], lw=1.1, label="noise_strength 5 (notebook default)")
    ax.plot(cols[0], cols[2], lw=1.1, ls="--", label=f"noise_strength {thresholds[m]:g} (max tolerable)")
    ax.set_xlabel("Frequency [ppm]"); ax.set_ylabel("Spectrum"); ax.legend(fontsize=7)
    ax.set_title(f"Molecule {m}: simulated spectrum (300 Lie-Trotter steps, dt 2.5 ms)", fontsize=9)
    fig.tight_layout(); fig.savefig(plots / f"simulated_spectrum_molecule_{m}.png"); plt.close(fig)
for k in (1, 2, 3):
    raw = pickle.load(open(ROOT / f"files/spectrum_{k}.pkl", "rb"))
    p, s = np.asarray(raw["frequencies"], float), np.asarray(raw["spectrum"], float)
    o = np.argsort(p); p, s = p[o], s[o] / np.max(np.abs(s))
    np.savetxt(out / f"reference_spectrum_{k}.csv", np.column_stack([p, s]), delimiter=",",
               header="ppm,spectrum_normalised", comments="", fmt="%.8g")
    fig, ax = plt.subplots(figsize=(7, 3), dpi=110)
    ax.plot(p, s, lw=1.0, color="0.3")
    ax.set_xlabel("Frequency [ppm]"); ax.set_ylabel("Spectrum (normalised)")
    ax.set_title(f"Reference spectrum_{k}.pkl (HQS Spectrum Tools)", fontsize=9)
    fig.tight_layout(); fig.savefig(plots / f"reference_spectrum_{k}.png"); plt.close(fig)
print(sorted(x.name for x in out.iterdir()))
