# Resolution criterion (version 1, fixed before any noise sweep)

Written 2026-11-10 after Task A (matching) and BEFORE any noise-strength sweep was run.
Implemented in `agents/agent3_spectral/common.py` (`characteristic_peaks`, `evaluate`,
`passes`, `threshold`). Constants: `MERGE_HZ`, `TOL_HZ`, `REF_HEIGHT_MIN`, `REF_PROM_MIN`,
`DIP_LEVELS`, `DIP_PRIMARY`. If this file is ever changed, version 1 stays here unchanged and
the new version is added below with the reason.

## Inputs

- Simulation: the OFFICIAL spectrum, `nmr_pipeline.spectrum_from_signals` on the official
  signals `results/signals/molecule_X.npz` (300 Lie-Trotter steps, dt = 0.0025 s), with its own
  ppm axis (grid 1.333 Hz). No re-gridding, no interpolation, no zero padding.
- Reference: the molecule's matched high-resolution reference from Task A
  (A -> spectrum_3, B -> spectrum_2, C -> spectrum_1), divided by its maximum.

## Step 1: characteristic peaks of the reference

1. Reference peaks: `scipy.signal.find_peaks` on the native reference grid with
   height >= 0.05 and prominence >= 0.02 (fractions of the reference maximum).
2. Merge rule: reference peaks closer than **2 official grid bins = 2.667 Hz** (single
   linkage) form ONE characteristic peak, located at the height-weighted centre, height = the
   largest member. Reason: two lines closer than about two bins of the 300-step acquisition
   (bin 1.333 Hz, 1/T resolution limit) cannot appear as two separate local maxima of the
   sampled spectrum at any noise strength.
3. Excluded splittings (determined from the references, before any sweep):
   - Molecule A (spectrum_3): four doublets with 0.81-0.86 Hz spacing (the J01 = 0.9 Hz
     splitting of the spin-0 and spin-1 lines at 5.7787, 5.8022, 5.9528, 5.9886 ppm). Each pair
     is one characteristic peak. Result: 8 characteristic peaks.
   - Molecule B (spectrum_2): the J02 = -0.5 Hz splitting is not resolved even in the reference
     (stick spectrum: 0.47 Hz pairs); nothing to merge. Result: 8 characteristic peaks.
   - Molecule C (spectrum_1): nothing excluded. Result: 4 characteristic peaks.
4. Check at the lowest tested noise (0.1 s^-1): every characteristic peak must be detected
   there. If one is not, it is reported (the criterion is not changed silently).

## Step 2: pass/fail of one simulated spectrum

1. Local maxima of the sampled simulated spectrum: `find_peaks(spectrum)` without thresholds.
2. Detection: for each characteristic peak, take the HIGHEST local maximum whose ppm lies
   within **+-2.667 Hz (2 grid bins)** of the reference position. Fail if none exists, if two
   characteristic peaks take the same maximum, or if the order of the maxima differs from the
   order of the characteristic peaks.
3. Dip: for every pair of ADJACENT characteristic peaks (sorted by ppm), with matched maxima
   at grid indices a < b: `dip = 1 - min(spectrum[a..b]) / min(spectrum[a], spectrum[b])`.
4. Pass at dip level D: detection succeeded and every adjacent dip >= D.
   - **Primary: D = 10 %.**
   - Alternatives reported: any dip (dip > 0, i.e. two distinct maxima), D = 5 %, and the
     Rayleigh-like D = 19 % (valley 81 % of the smaller peak).

Tolerance justification (+-2 bins): measured in Task A at noise 0.5-5 s^-1, the official axis
puts simulated maxima on average -0.54 (A), -0.77 (B), -0.67 (C) Hz from the reference lines
(the known half-bin axis shift); peak sampling adds up to half a bin (0.67 Hz); the inward pull
of two overlapping equal Lorentzians 6 Hz apart is 0.44 Hz at a 10 % dip (0.68 Hz at 5 %,
0.24 Hz at 19 %). Sum <= ~2.0 Hz < 2.667 Hz. For "any dip" the pull can reach 2 Hz, so a
position failure can occur before the dip vanishes; the failure reason is always recorded.

## Step 3: threshold search

- Dense scan, no bisection: noise_strength from 0.1 to 100 s^-1 in steps of 0.1; every interval
  where pass/fail changes is re-scanned in steps of 0.01.
- Maximum tolerable noise = the highest passing value on this grid; first failing = the next
  grid value above it. All transitions are listed (monotonicity check).
- A coarse sweep (0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, ..., 100) is recorded with
  per-peak details.
- Critical peaks = the adjacent pair with the smallest dip at the first failing value (or the
  peak that is not detected); critical splitting = their reference separation in Hz.

## Sensitivity checks (same characteristic peaks, same tolerance in Hz, same dip levels)

- Exact signals (dense exact diagonalisation of the same rotating-frame H, same weights and
  observables), 300 points, dt = 0.0025 s: Trotter-error effect.
- Lie-Trotter signals with 600 and 1200 steps (dt = 0.0025 s) via `simulate_signals_fast`:
  acquisition-length effect.
These are labelled sensitivity checks; the official answer is the 300-step Trotter value.

## Artificial J experiments (Task C)

No reference exists for a modified Hamiltonian. The stand-in reference is the exact stick
spectrum of the modified H broadened by a 1.0 Hz FWHM Lorentzian (the measured linewidth of
the supplied references is 0.97-1.0 Hz) on a 0.01 Hz grid. Before use it is validated on the
three real molecules: it must give the same characteristic peaks (count and positions within
0.1 Hz) and the same thresholds as the supplied references. The criterion is otherwise
identical.

## Review note (orchestrator, after the independent review; criterion v1 itself is unchanged)

- The tolerance budget above uses mean displacements. Per line, the simulated maxima at low noise
  already sit up to 1.73 Hz (A) and 1.86 Hz (B) from the reference centres (Agent 4). The
  +-2.667 Hz tolerance still holds, with less margin than stated. With a 1-bin tolerance, A and B
  never pass, even at noise 0. The 2-bin tolerance is therefore necessary, not just convenient.
- Excluding A's 0.9 Hz J01 doublets changes the outcome. If they had to stay resolved, A would never
  pass on the 300-step grid (0-1 maxima per doublet at every noise). With zero filling they appear
  only up to 1.25-1.95 s^-1 (Trotter) or 0.35 s^-1 (exact, with dips under 10 %).
- The merge rule makes the threshold jump where a splitting crosses 2.667 Hz (artificial A with
  J02 = 15.0 -> 15.5 Hz: 5.11 -> 49.14 s^-1). Splittings of 2.75-3.5 Hz are never resolved by this
  acquisition, so its practical limit is about 3.5-4 Hz.
- For "any dip", B fails first by a position failure of the outer 8.58 Hz line (the dip survives
  to 33.53), not by the 7.02 Hz pair. The critical splittings in `results/noise_tolerance.csv`
  belong to the 10 % level.
