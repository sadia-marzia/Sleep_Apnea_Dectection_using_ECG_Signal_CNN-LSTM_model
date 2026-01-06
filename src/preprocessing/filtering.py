"""
ECG Filtering Module

Applies:
- 50 Hz IIR notch filter for powerline interference removal
- 0.5 Hz high-pass Butterworth filter for baseline wander removal
"""

import numpy as np
import wfdb
from scipy.signal import iirnotch, filtfilt, butter


def notch_filter(signal, fs=100, f0=50.0, q=30.0):
    """Apply 50 Hz notch filter."""
    b, a = iirnotch(f0, q, fs)
    return filtfilt(b, a, signal)


def high_pass_filter(signal, fs=100, cutoff=0.5, order=5):
    """Apply high-pass filter to remove baseline wander."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, signal)


def filter_ecg(record_path):
    """
    Load ECG record and apply filtering.

    Parameters:
        record_path (str): Path to ECG record (without .dat extension)

    Returns:
        np.ndarray: Filtered ECG signal
        int: Sampling frequency
    """
    record = wfdb.rdrecord(record_path)
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs

    ecg_notched = notch_filter(ecg_signal, fs)
    ecg_filtered = high_pass_filter(ecg_notched, fs)

    return ecg_filtered, fs


