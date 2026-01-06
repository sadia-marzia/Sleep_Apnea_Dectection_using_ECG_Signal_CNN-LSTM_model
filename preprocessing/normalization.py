"""
ECG Normalization Module

Applies Z-score normalization to ECG signals
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_ecg(signal):
    """
    Apply Z-score normalization.

    Parameters:
        signal (np.ndarray): ECG signal

    Returns:
        np.ndarray: Normalized ECG signal
    """
    scaler = StandardScaler()
    normalized_signal = scaler.fit_transform(signal.reshape(-1, 1))
    return normalized_signal.flatten()
