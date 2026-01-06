"""
Non-linear HRV feature extraction

Extracted features:
SD1, SD2, SD Ratio
PSE
CSI, CVI
Modified CVI1, Modified CVI2
Shannon Entropy
Fractal Dimension
Lempel-Ziv Complexity
"""

import numpy as np
from scipy.stats import entropy


def lempel_ziv_complexity(signal):
    """
    Compute Lempel-Ziv Complexity
    """
    binary_signal = (signal > np.mean(signal)).astype(int)
    s = ''.join(binary_signal.astype(str))
    i, l, k = 0, 1, 1
    c = 1

    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > len(s):
                c += 1
                break
        else:
            if k > 1:
                i += 1
                k -= 1
            else:
                c += 1
                l += 1
                if l == len(s):
                    break
                i = 0
                k = 1
    return c


def fractal_dimension(signal):
    """
    Katz Fractal Dimension
    """
    n = len(signal)
    L = np.sum(np.abs(np.diff(signal)))
    d = np.max(np.abs(signal - signal[0]))
    return np.log10(n) / (np.log10(d / L) + np.log10(n)) if L > 0 and d > 0 else 0


def extract_nonlinear_features(rri):
    """
    Non-linear HRV features from RR intervals
    """

    if len(rri) < 2:
        return None

    diff_rri = np.diff(rri)

    sd1 = np.std(diff_rri) / np.sqrt(2)
    sd2 = np.std(rri) * np.sqrt(2)
    sd_ratio = sd1 / sd2 if sd2 > 0 else 0

    # Poincaré plot measures
    csi = sd2 / sd1 if sd1 > 0 else 0
    cvi = np.log10(sd1 * sd2) if sd1 > 0 and sd2 > 0 else 0

    modified_cvi1 = np.log10(sd1**2 * sd2) if sd1 > 0 and sd2 > 0 else 0
    modified_cvi2 = np.log10(sd1 * sd2**2) if sd1 > 0 and sd2 > 0 else 0

    # Shannon Entropy
    hist, _ = np.histogram(rri, bins=20, density=True)
    shannon_entropy = entropy(hist + 1e-10)

    # Power Spectral Entropy (PSE)
    psd = np.abs(np.fft.fft(rri))**2
    psd_norm = psd / np.sum(psd)
    pse = entropy(psd_norm + 1e-10)

    return {
        "sd1": sd1,
        "sd2": sd2,
        "sd_ratio": sd_ratio,
        "pse": pse,
        "csi": csi,
        "cvi": cvi,
        "modified_cvi1": modified_cvi1,
        "modified_cvi2": modified_cvi2,
        "shannon_entropy": shannon_entropy,
        "fractal_dimension": fractal_dimension(rri),
        "lempel_ziv_complexity": lempel_ziv_complexity(rri)
    }
