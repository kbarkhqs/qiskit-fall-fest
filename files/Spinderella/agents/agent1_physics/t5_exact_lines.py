"""T5: exact-diagonalisation line lists for A, B, C and comparison with the three reference files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.signal import find_peaks
from common import (HERE, NU0, HZ_TO_PPM, dense_h, load_terms, rotating_frame, physical_params, line_list,
                    load_reference, record, cmd)

WEAK = 1e-3  # lines below this intensity (units: 1 per spin) are listed as combination lines

def analyse(mol):
    terms = load_terms(mol); n, h, d, J = physical_params(terms); rf, mean_h = rotating_frame(terms)
    hz, inten, spin = line_list(dense_h(rf, n), n)
    dbar = d.mean()
    ppm = dbar + hz * HZ_TO_PPM
    nu = NU0 * (d - dbar) * 1e-6
    out = {"n_lines": len(hz), "sum_intensity": float(inten.sum()), "mean_delta_ppm": float(dbar),
           "lines": [{"hz": round(float(f), 4), "ppm": round(float(p), 6), "intensity": round(float(a), 6), "spin": int(s)}
                     for f, p, a, s in zip(hz, ppm, inten, spin)], "multiplets": []}
    for i in range(n):
        m = (spin == i) & (inten > WEAK)
        f, a = hz[m], inten[m]
        cand = {}
        partners = [(k, J[tuple(sorted((i, k)))]) for k in range(n) if k != i]
        for k, j in partners:
            cand[f"|J{min(i,k)}{max(i,k)}|"] = abs(j)
        if len(partners) == 2:
            (k1, j1), (k2, j2) = partners
            cand[f"||J{min(i,k1)}{max(i,k1)}|-|J{min(i,k2)}{max(i,k2)}||"] = abs(abs(j1) - abs(j2))
        if n == 2:
            k, j = partners[0]
            cand["sqrt(dnu^2+J^2)-J"] = float(np.hypot(nu[i] - nu[k], j) - abs(j))
        sp = []
        for g in np.diff(f):
            name = min(cand, key=lambda c: abs(cand[c] - g))
            sp.append({"spacing_hz": round(float(g), 4), "origin": name, "first_order_hz": round(float(cand[name]), 4),
                       "deviation_hz": round(float(g - cand[name]), 4)})
        out["multiplets"].append({"spin": i, "delta_ppm": float(d[i]), "nu_first_order_hz": round(float(nu[i]), 4),
                                  "line_hz": f.round(4).tolist(), "line_ppm": (dbar + f * HZ_TO_PPM).round(6).tolist(),
                                  "intensity": a.round(5).tolist(), "centroid_minus_nu_hz": round(float(np.sum(f * a) / a.sum() - nu[i]), 4),
                                  "max/min intensity (roof)": round(float(a.max() / a.min()), 4), "spacings": sp})
    weak = inten <= WEAK
    out["combination_lines"] = [{"hz": round(float(f), 3), "intensity": float(a)} for f, a in zip(hz[weak], inten[weak])]
    strong = np.sort(hz[~weak])
    gaps = np.diff(strong)
    out["smallest_spacings_hz"] = np.sort(gaps)[:4].round(4).tolist()
    return out, hz, inten, ppm

mols = {}
arrs = {}
for mol in "ABC":
    mols[mol], hz, inten, ppm = analyse(mol)
    arrs[mol] = (hz, inten, ppm)
    print(f"\n=== molecule {mol}: {mols[mol]['n_lines']} lines, sum intensity {mols[mol]['sum_intensity']:.6f}")
    for mp in mols[mol]["multiplets"]:
        print(f" spin {mp['spin']} delta {mp['delta_ppm']:.3f}: Hz {mp['line_hz']} I {mp['intensity']} roof {mp['max/min intensity (roof)']}"
              f" centroid shift {mp['centroid_minus_nu_hz']}")
        for s in mp["spacings"]:
            print("    spacing", s)
    print(" combination lines:", mols[mol]["combination_lines"], " smallest spacings:", mols[mol]["smallest_spacings_hz"])

# analytic AB check for C
t = load_terms("C"); n, h, d, J = physical_params(t)
dn = NU0 * (d[0] - d[1]) * 1e-6; j = J[(0, 1)]; c = np.hypot(dn, j)
ab = {"dnu_hz": float(dn), "J": j, "C=sqrt(dnu^2+J^2)": float(c), "analytic lines hz": [-(c + j) / 2, -(c - j) / 2, (c - j) / 2, (c + j) / 2],
      "analytic intensities": [(1 - j / c) / 2, (1 + j / c) / 2, (1 + j / c) / 2, (1 - j / c) / 2],
      "inner/outer intensity": float((1 + j / c) / (1 - j / c)),
      "max|ED - analytic| hz": float(np.abs(np.sort(arrs["C"][0]) - np.array([-(c + j) / 2, -(c - j) / 2, (c - j) / 2, (c + j) / 2])).max()),
      "max|ED - analytic| intensity": float(np.abs(arrs["C"][1] - np.array([(1 - j / c) / 2, (1 + j / c) / 2, (1 + j / c) / 2, (1 - j / c) / 2])).max())}
print("\nAB check (C):", ab)

# ---- reference files ---------------------------------------------------------------------------
def refine(x, y, k):  # vertex of the parabola through 3 points of a non-uniform grid
    p = np.polyfit(x[k - 1:k + 2] - x[k], y[k - 1:k + 2], 2)
    return x[k] - p[1] / (2 * p[0])

refs = {}
for i in (1, 2, 3):
    r = load_reference(i)
    x, y = r["frequencies"], r["spectrum"]
    assert np.all(np.diff(x) > 0)
    pk, _ = find_peaks(y, prominence=0.01 * y.max())
    pos = np.array([refine(x, y, k) for k in pk])
    refs[i] = (x, y, pos, y[pk] / y.max())
    print(f"\nspectrum_{i}: ppm {x.min():.4f}..{x.max():.4f}, grid step {np.diff(x).min()/HZ_TO_PPM:.4f}..{np.diff(x).max()/HZ_TO_PPM:.4f} Hz,"
          f" min {y.min():.3e} max {y.max():.3e}, {len(pos)} peaks at ppm {pos.round(5).tolist()}")

def sticks(x_ppm, ppm, inten, fwhm_hz):
    w = fwhm_hz / 2 * HZ_TO_PPM
    m = sum(a * w**2 / (w**2 + (x_ppm - p) ** 2) for a, p in zip(inten, ppm))
    return m

match = {}
fwhm_grid = np.geomspace(0.05, 5.0, 93)
for mol in "ABC":
    hz, inten, ppm = arrs[mol]
    strong = inten > WEAK
    for i in (1, 2, 3):
        x, y, pos, hgt = refs[i]
        inside = float(np.mean((ppm[strong] > x.min()) & (ppm[strong] < x.max())))
        cors = [np.corrcoef(y, sticks(x, ppm, inten, w))[0, 1] if inside > 0 else np.nan for w in fwhm_grid]
        best = int(np.nanargmax(cors)) if inside > 0 else 0
        dist = np.array([np.abs(ppm[strong] - p).min() for p in pos]) / HZ_TO_PPM   # ref peak -> nearest exact line
        dist2 = np.array([np.abs(pos - p).min() for p in ppm[strong]]) / HZ_TO_PPM  # exact line -> nearest ref peak
        match[f"{mol}-spectrum_{i}"] = {"fraction of exact lines inside the file's ppm window": inside,
                                       "best corr(reference, exact sticks)": float(cors[best]) if inside > 0 else None,
                                       "best-fit Lorentzian FWHM hz": float(fwhm_grid[best]) if inside > 0 else None,
                                       "max distance ref peak -> nearest exact line (Hz)": float(dist.max()),
                                       "max distance exact line -> nearest ref peak (Hz)": float(dist2.max())}
        print(f"{mol} vs spectrum_{i}:", match[f"{mol}-spectrum_{i}"])

assign = {}
for mol in "ABC":
    sc = {i: (match[f"{mol}-spectrum_{i}"]["best corr(reference, exact sticks)"] or -1) for i in (1, 2, 3)}
    assign[mol] = max(sc, key=sc.get)
unique = len(set(assign.values())) == 3
detail = {}
for mol, i in assign.items():
    hz, inten, ppm = arrs[mol]
    x, y, pos, hgt = refs[i]
    strong = inten > WEAK
    near = np.array([ppm[strong][np.argmin(np.abs(ppm[strong] - p))] for p in pos])
    off = (pos - near) / HZ_TO_PPM
    detail[mol] = {"file": f"spectrum_{i}", "n_ref_peaks": len(pos), "n_exact_lines(>1e-3)": int(strong.sum()),
                   "ref peak ppm": pos.round(5).tolist(), "ref peak rel height": hgt.round(4).tolist(),
                   "ref peak - nearest exact line (Hz)": off.round(4).tolist(), "mean offset hz": float(off.mean()),
                   "max|offset| hz": float(np.abs(off).max()),
                   "smallest spacing between reference peaks (Hz)": float(np.diff(pos).min() / HZ_TO_PPM)}
    print(f"\n{mol} -> spectrum_{i}:", detail[mol])

(HERE / "exact_lines.json").write_text(json.dumps({"molecules": mols, "AB_check_C": ab, "reference_match": match,
                                                    "assignment": {m: f"spectrum_{i}" for m, i in assign.items()},
                                                    "assignment_detail": detail}, indent=1, default=float) + "\n")
record({
    "id": "T5_exact_lines_and_reference_match",
    "question": "What are the exact transition frequencies/intensities of A, B, C, and which reference file contains which molecule's lines?",
    "hypotheses": ["each reference file contains the exact line list of exactly one molecule (3! = 6 possible assignments)"],
    "independent_variable": "molecule (A, B, C) x reference file (1, 2, 3)",
    "dependent_variable": "correlation of the reference spectrum with the exact stick spectrum (Lorentzian width fitted); peak-to-line distances",
    "fixed_parameters": {"weak_line_threshold": WEAK, "peak_prominence": "1% of the file maximum", "file names not used": True},
    "reference": "files/spectrum_{1,2,3}.pkl (read without unpickling)",
    "result": {"assignment": {m: f"spectrum_{i}" for m, i in assign.items()}, "assignment_is_a_permutation": unique,
               "match_matrix": match, "assignment_detail": detail, "AB_check_C": ab,
               "smallest_spacings_hz": {m: mols[m]["smallest_spacings_hz"] for m in mols},
               "multiplets": {m: mols[m]["multiplets"] for m in mols}, "combination_lines": {m: mols[m]["combination_lines"] for m in mols}},
    "conclusion": ("Assignment by line positions: " + ", ".join(f"{m} -> spectrum_{i}" for m, i in assign.items())
                   + (" (a permutation; " if unique else " (NOT a permutation; ")
                   + "max |reference peak - exact line| = " + ", ".join(f"{m}: {detail[m]['max|offset| hz']:.3f} Hz" for m in detail) + ")."),
    "limitations": "Reference peaks located by a 3-point parabola on a non-uniform grid; unresolved reference doublets give one peak for two lines.",
    "script": "agents/agent1_physics/t5_exact_lines.py",
    "command": cmd("t5_exact_lines.py"),
})
