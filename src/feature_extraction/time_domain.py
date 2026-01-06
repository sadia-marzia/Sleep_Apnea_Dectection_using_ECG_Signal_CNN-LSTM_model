"""
Time-domain HRV feature extraction from ECG segments
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks


def extract_time_domain_features(segment, fs=100):
    """
    Extract time-domain HRV features from an ECG segment
    """

    peaks, _ = find_peaks(
        segment,
        distance=fs / 2.5,
        height=np.mean(segment)
    )

    rri = np.diff(peaks) / fs
    rr_amp = np.diff(segment[peaks])

    if len(rri) == 0 or len(rr_amp) == 0:
        return None

    nn50 = np.sum(np.abs(np.diff(rri)) > 0.05)
    nn20 = np.sum(np.abs(np.diff(rri)) > 0.02)

    features = {
        "mean_rri": np.mean(rri),
        "std_rri": np.std(rri),
        "mean_rr_amp": np.mean(rr_amp),
        "std_rr_amp": np.std(rr_amp),
        "min_rr": np.min(rri),
        "range_rr": np.ptp(rri),
        "median_rr": np.median(rri),
        "sdnn": np.std(rri),
        "skewness_rr": skew(rri),
        "kurtosis_rr": kurtosis(rri),
        "sdsd": np.std(np.diff(rri)),
        "nn50": nn50,
        "nn20": nn20,
        "pnn50": (nn50 / len(rri)) * 100,
        "pnn20": (nn20 / len(rri)) * 100,
        "rmssd": np.sqrt(np.mean(np.diff(rri) ** 2)),
        "rms_rr_amp": np.sqrt(np.mean(rr_amp ** 2)),
        "rms_rri": np.sqrt(np.mean(rri ** 2)),
        "dif_rri": np.sqrt(np.mean(rri ** 2)) - np.mean(rri),
        "dif_rr_amp": np.sqrt(np.mean(rr_amp ** 2)) - np.mean(rr_amp),
    }

    return features
