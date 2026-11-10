"""Pipeline mirror (src/nmr_pipeline.py): fast route, cached official signals, guard, I/O."""
import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import nmr_pipeline as nmr

REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNALS = REPO_ROOT / "results" / "signals"
MOLECULES = ["A", "B", "C"]
SPINS = {"A": 3, "B": 3, "C": 2}


def cached(letter):
    path = SIGNALS / f"molecule_{letter}.npz"
    assert path.exists(), f"{path} is missing; run agents/agent2_implementation/run_official_simulation.py {letter}"
    return np.load(path)


# ------------------------------------------------------------------------------------
# fast route == official estimator loop
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("letter", MOLECULES)
def test_fast_equals_estimator_on_short_run(letter):
    ham, n, _ = nmr.load_rotating_frame_hamiltonian(f"molecule {letter}")
    steps = 12
    est_x, est_y = nmr.simulate_signals_estimator(ham, n, steps, nmr.DEFAULT_TROTTER_TIMESTEP)
    fast_x, fast_y = nmr.simulate_signals_fast(ham, n, steps, nmr.DEFAULT_TROTTER_TIMESTEP)
    assert est_x.shape == fast_x.shape == (2**n, steps)
    assert np.max(np.abs(est_x - fast_x)) < 1e-9
    assert np.max(np.abs(est_y - fast_y)) < 1e-9
    # the comparison is not between two constant arrays: <sum X> moves away from its start value
    assert np.max(np.abs(est_x - est_x[:, :1])) > 1e-2


@pytest.mark.parametrize("letter", MOLECULES)
def test_fast_full_run_reproduces_cached_official_signals(letter):
    """Independent recomputation of the cached estimator result with the fast route."""
    data = cached(letter)
    ham, n, shift = nmr.load_rotating_frame_hamiltonian(f"molecule {letter}")
    steps = int(data["number_of_trottersteps"])
    dt = float(data["trotter_timestep"])
    fast_x, fast_y = nmr.simulate_signals_fast(ham, n, steps, dt)
    assert np.max(np.abs(fast_x - data["exp_x"])) < 1e-9
    assert np.max(np.abs(fast_y - data["exp_y"])) < 1e-9
    assert float(data["chemical_shift_hz"]) == shift


# ------------------------------------------------------------------------------------
# sanity identities on the cached official signals
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("letter", MOLECULES)
def test_cached_signals_sanity(letter):
    data = cached(letter)
    n = SPINS[letter]
    steps = 300
    assert int(data["number_spins"]) == n
    assert int(data["number_of_trottersteps"]) == steps
    assert float(data["trotter_timestep"]) == 0.0025
    assert data["exp_x"].shape == (2**n, steps)
    assert data["exp_y"].shape == (2**n, steps)
    assert data["bits"].shape == (2**n, n)
    assert data["mx0"].shape == (2**n,)
    assert data["c_xx_raw"].shape == data["c_yx_raw"].shape == data["times"].shape == (steps,)
    for key in ("exp_x", "exp_y", "c_xx_raw", "c_yx_raw", "times", "mx0"):
        assert np.isfinite(data[key]).all(), key

    # bits are stored in the order used by the notebook
    _, _, bits_list, _ = nmr.enumerate_x_states(n)
    assert np.array_equal(data["bits"], np.array(bits_list))

    # weight of state n = measured <sum X> at step 0 = N - 2 * (number of minus qubits)
    expected_mx0 = n - 2 * data["bits"].sum(axis=1)
    assert np.max(np.abs(data["exp_x"][:, 0] - expected_mx0)) < 1e-12
    assert np.array_equal(data["mx0"], data["exp_x"][:, 0])
    # <sum Y> of an X eigenstate is zero
    assert np.max(np.abs(data["exp_y"][:, 0])) < 1e-12

    # C_xx(0) = sum_n m_n^2 = N * 2^N and C_yx(0) = 0
    assert abs(data["c_xx_raw"][0] - n * 2**n) < 1e-10
    assert abs(data["c_yx_raw"][0]) < 1e-10

    # the signals are not identically zero and they do evolve in time
    assert np.std(data["c_xx_raw"]) > 1.0
    if n == 3:
        assert np.max(np.abs(data["c_yx_raw"])) > 1.0
    else:
        # Molecule C: two spins whose rotating-frame offsets are +h and -h. The spectrum is
        # mirror-symmetric about the centre, so C_yx vanishes at all times (a property of
        # this Hamiltonian, not a failure of the simulation): the two states with non-zero
        # weight (all-plus, all-minus) keep <sum Y> = 0, the other two have weight 0.
        assert np.max(np.abs(data["c_yx_raw"])) < 1e-9
        weighted = np.abs(data["mx0"]) > 0.5
        assert np.max(np.abs(data["exp_y"][weighted])) < 1e-9
        assert np.max(np.abs(data["exp_y"][~weighted])) > 1.0
    # |<sum X>| can never exceed N
    assert np.max(np.abs(data["exp_x"])) <= n + 1e-12
    assert np.max(np.abs(data["exp_y"])) <= n + 1e-12
    assert np.allclose(data["times"], np.arange(steps) * 0.0025, rtol=0, atol=0)


@pytest.mark.parametrize("letter", MOLECULES)
def test_correlations_reproduce_cached_raw_signals(letter):
    data = cached(letter)
    c_xx, c_yx = nmr.correlations(data["exp_x"], data["exp_y"])
    assert np.array_equal(c_xx, data["c_xx_raw"])
    assert np.array_equal(c_yx, data["c_yx_raw"])
    # same numbers from an independent matrix expression
    assert np.max(np.abs(c_xx - data["mx0"] @ data["exp_x"])) < 1e-10
    assert np.max(np.abs(c_yx - data["mx0"] @ data["exp_y"])) < 1e-10


@pytest.mark.parametrize("letter", MOLECULES)
def test_spectrum_from_cached_signals(letter):
    data = cached(letter)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        spectrum, ppm = nmr.spectrum_from_signals(
            data["c_xx_raw"], data["c_yx_raw"], 5.0, 0.0025, 300, float(data["chemical_shift_hz"])
        )
    assert spectrum.shape == ppm.shape == (300,)
    assert np.isfinite(spectrum).all() and np.isfinite(ppm).all()
    assert spectrum.max() == 1.0 and spectrum.min() >= 0.0
    assert np.all(np.diff(ppm) > 0)
    # the raw signals are not modified by the windowing
    again = cached(letter)
    assert np.array_equal(again["c_xx_raw"], data["c_xx_raw"])


# ------------------------------------------------------------------------------------
# guard in time_signal_to_spectrum
# ------------------------------------------------------------------------------------
def _call_spectrum(c_xx, c_yx):
    return nmr.time_signal_to_spectrum(
        c_xx, c_yx, 0.0025, len(c_xx), Hz_to_ppm_conversion=nmr.hz_to_ppm_conversion(), chem_shift_hz=-1.57e9
    )


def test_guard_raises_on_all_zero_signal():
    zeros = np.zeros(300)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the 0/0 RuntimeWarning must not be reached
        with pytest.raises(ValueError, match="Cannot normalise"):
            _call_spectrum(zeros, zeros)


def test_guard_raises_on_equal_weights_rounding_noise():
    # Reviewer finding: weighting every initial state by 1 gives Tr(A(t)) = 0 up to ~1e-15
    # rounding, which an "== 0" guard missed. Built from the official cached signals.
    sig = np.load(REPO_ROOT / "results" / "signals" / "molecule_A.npz")
    c_xx = np.ones(len(sig["mx0"])) @ sig["exp_x"]
    c_yx = np.ones(len(sig["mx0"])) @ sig["exp_y"]
    assert np.max(np.abs(c_xx)) < 1e-12
    with pytest.raises(ValueError, match="rounding noise"):
        _call_spectrum(c_xx, c_yx)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_guard_raises_on_non_finite_signal(bad):
    signal = np.ones(300)
    signal[10] = bad
    with np.errstate(invalid="ignore"):
        with pytest.raises(ValueError, match="Cannot normalise"):
            _call_spectrum(signal, np.zeros(300))


def test_guard_passes_a_single_lorentzian_at_the_right_place():
    """Known input: exp(-i 2 pi f0 t - G t) must give one peak at f0 (in ppm units)."""
    steps, dt, f0, gamma = 300, 0.0025, 40.0, 5.0
    t = np.arange(steps) * dt
    c_xx = np.cos(2 * np.pi * f0 * t) * np.exp(-gamma * t)
    c_yx = np.sin(2 * np.pi * f0 * t) * np.exp(-gamma * t)
    conv = nmr.hz_to_ppm_conversion()
    spectrum, ppm = nmr.time_signal_to_spectrum(c_xx, c_yx, dt, steps, Hz_to_ppm_conversion=conv, chem_shift_hz=0.0)
    freq_hz = (ppm + 1e6) / conv  # chem_shift_hz = 0 leaves only the 1e6 offset
    peak_hz = freq_hz[np.argmax(spectrum)]
    assert abs(abs(peak_hz) - f0) <= 1.0 / (steps * dt)  # within one frequency bin
    assert spectrum.max() == 1.0


# ------------------------------------------------------------------------------------
# module == notebook, loaders
# ------------------------------------------------------------------------------------
def test_module_functions_are_verbatim_notebook_code():
    nb = json.loads((REPO_ROOT / "nmr.ipynb").read_text(encoding="utf-8"))
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for function in (
        nmr.struqture_hamiltonian_to_qiskit_op,
        nmr.xsum_eigenstate_template,
        nmr.time_signal_to_spectrum,
        nmr.fft_causal,
    ):
        assert inspect.getsource(function).rstrip() in code, function.__name__


def test_constants_match_the_notebook_defaults():
    nb = json.loads((REPO_ROOT / "nmr.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert f"number_of_trottersteps = {nmr.DEFAULT_NUMBER_OF_TROTTERSTEPS}\n" in code
    assert f"trotter_timestep = {nmr.DEFAULT_TROTTER_TIMESTEP}\n" in code
    assert f"noise_strength = {nmr.DEFAULT_NOISE_STRENGTH} " in code
    assert f"field_T = {nmr.FIELD_T}\n" in code
    assert "GYROMAGNETIC_RATIO_1H_RAD_S_T = 267_522_128.0" in code
    assert nmr.GYROMAGNETIC_RATIO_1H_RAD_S_T == 267_522_128.0


@pytest.mark.parametrize("letter", MOLECULES)
def test_hamiltonian_loader(letter):
    ham, n, shift = nmr.load_rotating_frame_hamiltonian(f"molecule {letter}")
    short, n_short, shift_short = nmr.load_rotating_frame_hamiltonian(letter)
    assert n == n_short == SPINS[letter] == ham.num_qubits
    assert shift == shift_short and ham == short
    labels = ham.paulis.to_labels()
    z_coeffs = [ham.coeffs[labels.index("I" * (n - 1 - i) + "Z" + "I" * i)].real for i in range(n)]
    # rotating frame: the single-spin Z coefficients sum to zero up to float64 rounding of 1.5e9
    assert abs(sum(z_coeffs)) < 1e-5
    assert np.max(np.abs(ham.coeffs.imag)) == 0.0


def test_coupling_overrides():
    base, n, shift = nmr.load_rotating_frame_hamiltonian("molecule A")
    labels = base.paulis.to_labels()
    # qubit 0 is the rightmost label character: pair (0, 1) is 'IXX', 'IYY', 'IZZ'
    pair_labels = ["IXX", "IYY", "IZZ"]
    j_file_hz = base.coeffs[labels.index("IXX")].real * 2 / np.pi

    same, _, _ = nmr.load_rotating_frame_hamiltonian("molecule A", coupling_overrides={(0, 1): j_file_hz})
    assert same.paulis == base.paulis and np.max(np.abs(same.coeffs - base.coeffs)) < 1e-12

    changed, _, shift_changed = nmr.load_rotating_frame_hamiltonian("molecule A", coupling_overrides={(1, 0): 5.0})
    assert changed.paulis == base.paulis and shift_changed == shift
    for label, c_base, c_new in zip(labels, base.coeffs, changed.coeffs):
        if label in pair_labels:
            assert abs(c_new - np.pi * 5.0 / 2) < 1e-12
        else:
            assert c_new == c_base

    with pytest.raises(ValueError):
        nmr.load_rotating_frame_hamiltonian("molecule C", coupling_overrides={(0, 2): 1.0})


@pytest.mark.parametrize("k", [1, 2, 3])
def test_load_reference_is_raw_and_not_mutated(k):
    ppm, spectrum = nmr.load_reference(k)
    assert ppm.shape == spectrum.shape == (2000,)
    assert np.isfinite(ppm).all() and np.isfinite(spectrum).all()
    assert np.max(np.abs(spectrum)) != 1.0  # not normalised
    _ = spectrum / np.max(np.abs(spectrum))  # the notebook's (non-in-place) normalisation
    ppm_again, spectrum_again = nmr.load_reference(k)
    assert np.array_equal(spectrum, spectrum_again) and np.array_equal(ppm, ppm_again)
