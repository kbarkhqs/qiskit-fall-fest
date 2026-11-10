"""C4 side question: is excluding A's 0.94 Hz J01 doublets a property of the 0.75 s signal or of
the pipeline's frequency grid (no zero filling)?
Run: cd agents/agent4_reviewer && ../../.venv/bin/python r4b_zero_fill_J01.py

Same official Lie-Trotter signal (300 steps, dt 2.5 ms, window exp(-Gamma t), k=0 weight 1/2 as in
fft_causal), but the transform is evaluated on finer frequency grids (zero filling): spacing
1.333 Hz (official), 0.667 Hz (the 2L-1 FFT before the [::2] downsampling), 0.333 Hz and 0.05 Hz.
For each Gamma: number of maxima inside each of the four doublet windows and the smallest dip.
"""
import numpy as np

from r_common import (parse_json_terms, rotating_frame, trotter_step, exact_step, trace_correlations,
                      lines_from_step, save)

terms, n = parse_json_terms("A")
rt, hbar = rotating_frame(terms, n)
L, dt = 300, 0.0025
t = np.arange(L) * dt
g = np.ones(L)
g[0] = 0.5
fl, cl = lines_from_step(exact_step(rt, n, 1e-9), n, 1e-9)
strong = cl.real > 0.05 * cl.real.max()
fl = np.sort(fl[strong])
pairs = [(a, b) for a, b in zip(fl[:-1], fl[1:]) if b - a < 1.5]
print("J01 doublets (exact, Hz):", [(round(a, 2), round(b, 2)) for a, b in pairs])

out = {"doublets_Hz_exact": pairs}
for label, U in (("trotter_official", trotter_step(rt, n)), ("exact", exact_step(rt, n))):
    cxx, cyx = trace_correlations(U, n, L)
    fo, co = lines_from_step(U, n, dt)
    so = co.real > 0.05 * co.real.max()
    fo = np.sort(fo[so])
    pairs = [(a, b) for a, b in zip(fo[:-1], fo[1:]) if b - a < 1.5]
    print(label, "doublets:", [(round(a, 2), round(b, 2)) for a, b in pairs])
    res = {}
    for spacing in (4 / 3, 2 / 3, 1 / 3, 0.05):
        f = np.arange(-160, 160, spacing)
        K = np.exp(-2j * np.pi * np.outer(f, t)) * g
        best_any, best_10 = None, None
        dips0 = None
        for gam in np.round(np.arange(0, 10.001, 0.05), 2):
            w = np.exp(-gam * t)
            S = (K @ (cxx * w)).real + (K @ (cyx * w)).imag
            S = np.abs(S / np.abs(S).max())
            ok_any, ok_10, dmins = True, True, []
            for a, b in pairs:
                # one local maximum within 0.3 Hz of EACH line of the doublet (Trotter lines move by
                # <= 0.72 Hz, so the window is centred on the line positions of the signal's own dynamics)
                mxs = np.nonzero((S[1:-1] > S[:-2]) & (S[1:-1] > S[2:]))[0] + 1
                ia = mxs[np.abs(f[mxs] - a) <= 0.3]
                ib = mxs[np.abs(f[mxs] - b) <= 0.3]
                if len(ia) == 0 or len(ib) == 0:
                    ok_any = ok_10 = False
                    dmins.append(0.0)
                    continue
                i1, i2 = ia[np.argmax(S[ia])], ib[np.argmax(S[ib])]
                if i2 <= i1:
                    ok_any = ok_10 = False
                    dmins.append(0.0)
                    continue
                d = 1 - S[i1:i2 + 1].min() / min(S[i1], S[i2])
                dmins.append(float(d))
                ok_any &= d > 0
                ok_10 &= d >= 0.10
            if gam == 0:
                dips0 = dmins
            if ok_any:
                best_any = gam
            if ok_10:
                best_10 = gam
        res[f"{spacing:.3f} Hz"] = {"highest_Gamma_all_doublets_two_maxima": best_any,
                                    "highest_Gamma_all_doublets_dip_10pct": best_10, "dips_at_Gamma0": dips0}
        print(label, f"grid {spacing:.3f} Hz: all 4 doublets show two maxima up to Gamma = {best_any}, with >=10% dip up to {best_10}; dips at Gamma 0: {np.round(dips0, 3).tolist()}")
    out[label] = res
save("r4b_zero_fill_J01.json", out)
