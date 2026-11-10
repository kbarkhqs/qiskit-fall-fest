"""Verbatim copy of fft_causal and time_signal_to_spectrum from the ORIGINAL notebook
(git HEAD:nmr.ipynb, cell 24). t4_sign_axis.py checks this text against the notebook."""
import numpy as np
from typing import Tuple, Any

# --- BEGIN VERBATIM ---
def time_signal_to_spectrum(
    c_xx: np.ndarray,
    c_yx: np.ndarray,
    trotter_timestep: float,
    number_of_trottersteps: int,
    Hz_to_ppm_conversion: float,
    chem_shift_hz: float,
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Calculate the full spectrum from the computed expectation values via a Fourier Transform.

    Args:
        m_avg_z (np.ndarray): The first expectation value (c_xx)
        m_avg_y (np.ndarray): The second expectation value (c_yx)
        N_max (int): Number of Trotter steps.
        trotter_timestep (float): The Trotter timestep.
        Hz_to_ppm_conversion (float): Conversion factor from frequency to ppm.
        chem_shift_hz (float): The chemical shift of the input Hamiltonian (in Hz).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - **spectrum_full (np.ndarray):** The calculated spectrum.
            - **ppm (np.ndarray):** The ppm array to plot the full spectrum against.
    """
    spectrum_full = np.zeros(c_xx.shape[0], dtype=np.float64)

    m_avg_z = fft_causal(c_xx)
    m_avg_y = fft_causal(c_yx)

    frequencies = np.fft.fftshift(np.fft.fftfreq(number_of_trottersteps, d=trotter_timestep))

    # And now we have the frequency axis in ppm:
    # we need to shift the frequencies by twice the chemical input shift and then by the reference
    # energy, i.e. the energy of the honomuclear isotope (H) times the magnetic field.
    
    # the chemical shift:
    chem_shift = 2 * chem_shift_hz / (2.0*np.pi/Hz_to_ppm_conversion) + 1e6

    # to have the frequency axis in ppm:
    ppm = (frequencies) * Hz_to_ppm_conversion - chem_shift

    spectrum_full = m_avg_z.real + m_avg_y.imag

    spectrum_full /= np.max(np.abs(spectrum_full))

    return np.abs(spectrum_full), ppm


def fft_causal(f: np.ndarray) -> np.ndarray:
    """
    Performs the Fourier Transform of a causal signal by separating its even and odd parts.

    Args:
        f (np.ndarray): Input causal signal.

    Returns:
        np.ndarray: The complex Fourier Transform of the causal signal.
    """
    L = len(f)

    # Even part (symmetrize)
    fe = np.zeros(2 * L - 1, dtype=f.dtype)
    fe[:L] = f
    fe[L:] = f[:-1][::-1]
    fe /= 2

    # Odd part (antisymmetrize)
    fo = np.zeros(2 * L - 1, dtype=f.dtype)
    fo[:L] = f
    fo[L:] = -f[:-1][::-1]
    fo /= 2
    fo[0] = 0  # Avoids jump discontinuity at t=0

    # FFT with orthogonal normalization and shift
    fft_even = np.fft.fftshift(np.fft.fft(fe, norm="ortho"))
    fft_odd = np.fft.fftshift(np.fft.fft(fo, norm="ortho"))

    # Combine and downsample to original length
    combined_fft = (fft_even + fft_odd)[::2]

    return combined_fft
# --- END VERBATIM ---
