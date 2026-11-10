"""C5: exact-limit thresholds and the cause of A's lower official value.
Run: cd agents/agent4_reviewer && ../../.venv/bin/python r5_exact_limit.py

1. Long acquisition (2400 x 2.5 ms): exact (eigendecomposition) and Lie-Trotter (own product
   formula, equal to the official signal to 2e-12 at 300 steps) signals, official transform,
   own criterion (reference characteristic peaks, +-2.667 Hz, adjacent dips), Gamma 0.01..60 in
   0.01 steps; monotonicity reported.
2. Continuum: lines of the one-step propagator (exact or Trotter, Floquet), each turned into the
   infinite-acquisition line shape of the sampled signal, sum_k g_k e^{-Gamma t_k} e^{i(w-2 pi f)t_k}
   = 1/(1-z) - 1/2, on a 0.01 Hz grid; same criterion. Also the continuous-time Lorentzian.
3. Causal decomposition (A, B, C): exact positions + Trotter amplitudes, Trotter positions + exact
   amplitudes. If the threshold follows the amplitudes, the Trotter effect is through line heights.
"""
import numpy as np

from r_common import (HZ_PER_PPM, GRID_HZ, MATCH, DT, nmr, parse_json_terms, rotating_frame, exact_step,
                      trotter_step, trace_correlations, lines_from_step, load_ref, char_peaks_from,
                      local_maxima, assess, passes, scan_threshold, official_spectrum, save)

MOLS = "ABC"
L = 2400
GAM = np.round(np.arange(1, 6001) * 0.01, 2)
res = {}


def thr_from_spectra(spectra, p, centres, level):
    flags = []
    for s in spectra:
        found, dips, _ = assess(s, p, centres)
        flags.append(passes(found, dips, level))
    mp, ff, tr = scan_threshold(np.array(flags), GAM[:len(flags)])
    return mp, ff, tr


def continuum(f_lines, c_lines, gam, fgrid, kind="discrete"):
    S = np.zeros_like(fgrid)
    for fl, cl in zip(f_lines, c_lines):
        if kind == "discrete":
            z = np.exp((-gam + 1j * 2 * np.pi * (fl - fgrid)) * DT)
            K = 1 / (1 - z) - 0.5
        else:
            K = 1 / (gam - 1j * 2 * np.pi * (fl - fgrid))
        S += np.real(cl * K)
    S = np.abs(S / np.abs(S).max())
    return S


for m in MOLS:
    terms, n = parse_json_terms(m)
    rt, hbar = rotating_frame(terms, n)
    centres = char_peaks_from(*load_ref(MATCH[m]))
    dbar = hbar  # for the ppm axis of the continuum we work in Hz relative to the mean shift
    out = {}
    # ---------- 1. long acquisition ----------
    for label, U in (("exact", exact_step(rt, n)), ("trotter", trotter_step(rt, n))):
        cxx, cyx = trace_correlations(U, n, L)
        spectra = []
        p = None
        for g in GAM:
            s, p = official_spectrum(cxx, cyx, g, hbar, L=L)
            spectra.append(s)
        for lev_name, lev in (("any", 0.0), ("10%", 0.10)):
            mp, ff, tr = thr_from_spectra(spectra, p, centres, lev)
            out[f"L2400_{label}_{lev_name}"] = {"max_pass": mp, "first_fail": ff, "transitions": tr}
    # ---------- 1b. official length (300 steps), exact vs Trotter, own transform ----------
    from r_common import spectrum_batch
    for label, U in (("exact", exact_step(rt, n)), ("trotter", trotter_step(rt, n))):
        cxx, cyx = trace_correlations(U, n, 300)
        S3, p3 = spectrum_batch(cxx, cyx, GAM, hbar, L=300)
        for lev_name, lev in (("any", 0.0), ("10%", 0.10)):
            mp, ff, tr = thr_from_spectra(S3, p3, centres, lev)
            out[f"L300_{label}_{lev_name}"] = {"max_pass": mp, "first_fail": ff, "transitions": tr}
    # ---------- 2./3. continuum with exact / Trotter / swapped lines ----------
    fE, cE = lines_from_step(exact_step(rt, n), n, DT)
    fT, cT = lines_from_step(trotter_step(rt, n), n, DT)
    sE = cE.real > 1e-3 * cE.real.max()
    sT = np.abs(cT) > 1e-3 * np.abs(cT).max()
    fE, cE, fT, cT = fE[sE], cE[sE], fT[sT], cT[sT]
    # match every exact line to its Trotter counterpart (nearest, one-to-one check)
    idx = np.array([np.argmin(np.abs(fT - x)) for x in fE])
    one_to_one = len(set(idx.tolist())) == len(idx)
    if not one_to_one and len(fT) == len(fE):
        # nearest-line pairing is ambiguous inside 0.94 Hz doublets (shifts up to 0.72 Hz):
        # pair by rank instead (both lists sorted ascending) and record the largest shift
        idx = np.arange(len(fE))
        out["pairing"] = "rank (nearest was not one-to-one)"
    else:
        out["pairing"] = "nearest"
    extra_T = [float(fT[i]) for i in range(len(fT)) if i not in set(idx.tolist())]
    out["n_lines_exact"], out["n_lines_trotter"], out["one_to_one"] = len(fE), len(fT), one_to_one
    out["unmatched_trotter_lines_Hz"] = extra_T
    out["max_rel_amp_change"] = float(np.max(np.abs(cT[idx].real / cE.real - 1)))
    out["max_shift_Hz"] = float(np.max(np.abs(fT[idx] - fE)))
    # mean delta (ppm) for the ppm axis: centres are in ppm; continuum is in Hz relative to mean
    dmean = np.mean([(-2 * c / (267_522_128.0 * 11.7433) - 1) * 1e6 for o, c in terms if len(o) == 1])
    fgrid = np.arange(min(fE.min(), fT.min()) - 40, max(fE.max(), fT.max()) + 40, 0.01)
    pgrid = dmean + fgrid / HZ_PER_PPM
    combos = {"exact pos + exact amp": (fE, cE), "trotter pos + trotter amp": (fT[idx], cT[idx]),
              "exact pos + trotter amp": (fE, cT[idx]), "trotter pos + exact amp": (fT[idx], cE)}
    gscan = np.round(np.arange(15.0, 45.0, 0.05), 2)
    for name, (fl, cl) in combos.items():
        for kind in (("discrete",) if name != "exact pos + exact amp" else ("discrete", "lorentz")):
            flags = []
            for g in gscan:
                S = continuum(fl, cl, g, fgrid, kind)
                found, dips, _ = assess(S, pgrid, centres)
                flags.append(passes(found, dips, 0.0))
            mp, ff, tr = scan_threshold(np.array(flags), gscan)
            out[f"continuum_any[{name}|{kind}]"] = {"max_pass": mp, "first_fail": ff, "n_transitions": len(tr)}
    # spacing of the critical inner pair (sorted strong lines) exact vs Trotter
    res[m] = out
    print(m, {k: (v["max_pass"] if isinstance(v, dict) else v) for k, v in out.items()})
save("r5_exact_limit.json", res)
