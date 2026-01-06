"""
Frequency-domain HRV feature extraction

Extracted features:
VLF, ULF, LF, HF Band Powers
LF/HF Ratio
HF Peak Frequency
LF Peak Frequency
Total Power
LFnu, HFnu
"""

import numpy as np
from scipy.signal import welch


def extract_frequency_domain_features(rri, fs=4.0):
    """
    Frequency-domain HRV features from RR intervals
    RR intervals are interpolated at 4 Hz (standard HRV practice)
    """

    if len(rri) < 4:
        return None

    # Power Spectral Density using Welch
    freqs, psd = welch(
        rri,
        fs=fs,
        nperseg=min(256, len(rri))
    )

    # Frequency bands (Hz)
    ulf_band = (freqs < 0.003)
    vlf_band = (freqs >= 0.003) & (freqs < 0.04)
    lf_band  = (freqs >= 0.04) & (freqs < 0.15)
    hf_band  = (freqs >= 0.15) & (freqs < 0.4)

    # Band powers
    ulf_power = np.trapz(psd[ulf_band], freqs[ulf_band])
    vlf_power = np.trapz(psd[vlf_band], freqs[vlf_band])
    lf_power  = np.trapz(psd[lf_band], freqs[lf_band])
    hf_power  = np.trapz(psd[hf_band], freqs[hf_band])

    total_power = ulf_power + vlf_power + lf_power + hf_power

    # Peak frequencies
    lf_peak_freq = freqs[lf_band][np.argmax(psd[lf_band])] if np.any(lf_band) else 0
    hf_peak_freq = freqs[hf_band][np.argmax(psd[hf_band])] if np.any(hf_band) else 0

    return {
        "ulf_power": ulf_power,
        "vlf_power": vlf_power,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_power / hf_power if hf_power > 0 else 0,
        "lf_peak_freq": lf_peak_freq,
        "hf_peak_freq": hf_peak_freq,
        "total_power": total_power,
        "lfnu": lf_power / total_power if total_power > 0 else 0,
        "hfnu": hf_power / total_power if total_power > 0 else 0
    }
