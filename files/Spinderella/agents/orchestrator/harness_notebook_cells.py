"""Fast harness: run the notebook's code cells (skipping %pip and replacing the slow section-6
Estimator loop by the cached official result for molecule A), then the new section 8/9 cells."""
import json, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agents" / "orchestrator"))
from notebook_cells import SECTION1_VIS, SECTION8, SECTION9, SECTION9_J, SECTION10  # noqa: E402

nb = json.loads((ROOT / "nmr.ipynb").read_text())
ns = {}
import os
os.chdir(ROOT)
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code" or c.get("metadata", {}).get("added_by") == "orchestrator":
        continue
    src = "".join(c["source"])
    if "%pip" in src or "Section 8.1" in src or "def simulate_molecule" in src:
        continue
    if "estimator = Estimator()" in src:  # section 6.2 loop -> cached official result
        d = np.load(ROOT / "results/signals/molecule_A.npz")
        ns["exp_x"], ns["exp_y"] = d["exp_x"].copy(), d["exp_y"].copy()
        continue
    exec(compile(src, f"cell{i}", "exec"), ns)
t0 = time.time()
for kind, src in SECTION1_VIS + SECTION8 + SECTION9 + SECTION9_J + SECTION10:
    if kind == "code":
        exec(compile(src, "new", "exec"), ns)
print(f"new cells ran in {time.time() - t0:.1f} s")

# --- diagnosis: why does A's "any dip" threshold (27.3) sit below the continuum prediction (32.5)?
if "--diagnose" in sys.argv:
    from scipy.signal import find_peaks as fp
    for name, g in (("molecule A", 27.33), ("molecule B", 33.37), ("molecule C", 28.59)):
        centres, _ = ns["characteristic_peaks"](*ns["references"][ns["assignment"][name]])
        spec, ppm = ns["spectrum_at"](ns["sims"][name], g)
        print(name, g, ns["resolved"](spec, ppm, centres, 0.0))
        # count maxima inside each critical multiplet without any position tolerance
    # same criterion on longer acquisitions (dt fixed), to separate truncation from the grid
    sys.path.insert(0, str(ROOT)); import src.nmr_pipeline as P
    for name, extra in (("molecule A", (300, 600, 1200, 2400)),):
        H, n, chem = P.load_rotating_frame_hamiltonian(name)
        centres, _ = ns["characteristic_peaks"](*ns["references"][ns["assignment"][name]])
        for L in extra:
            ex, ey = P.simulate_signals_fast(H, n, number_of_trottersteps=L, trotter_timestep=0.0025)
            cxx, cyx = P.correlations(ex, ey)
            HZ = ns["HZ_PER_PPM"]
            def ok(g, lev):
                s, p = P.spectrum_from_signals(cxx, cyx, g, 0.0025, L, chem)
                p = np.asarray(p, float)
                mx, _ = fp(s); ch = []
                for c in centres:
                    near = mx[np.abs(p[mx] - c) * HZ <= 2.667]
                    if len(near) == 0: return False
                    ch.append(near[np.argmax(s[near])])
                if np.any(np.diff(ch) <= 0): return False
                d = [1 - s[a:b+1].min() / min(s[a], s[b]) for a, b in zip(ch[:-1], ch[1:])]
                return all(x > lev for x in d) if lev == 0 else all(x >= lev for x in d)
            res = {}
            for lev in (0.0, 0.10):
                gs = np.round(np.arange(10, 40, 0.05), 2)
                f = [ok(g, lev) for g in gs]
                res[lev] = gs[np.array(f)].max()
            print(f"{name} acquisition {L} steps ({L*0.0025:.2f} s): any-dip {res[0.0]:.2f}, 10% {res[0.10]:.2f}")
