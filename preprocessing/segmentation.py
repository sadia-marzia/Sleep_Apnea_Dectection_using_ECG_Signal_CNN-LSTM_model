"""
ECG Segmentation Module

Segments ECG signals into fixed-length epochs (default: 1 minute)
"""

import numpy as np


def segment_signal(ecg_signal, fs=100, segment_duration_sec=60):
    """
    Segment ECG signal into fixed-length segments.

    Parameters:
        ecg_signal (np.ndarray): ECG signal
        fs (int): Sampling frequency
        segment_duration_sec (int): Segment length in seconds

    Returns:
        list[np.ndarray]: List of ECG segments
    """
    segment_length = fs * segment_duration_sec
    num_segments = len(ecg_signal) // segment_length
    segments = np.array_split(
        ecg_signal[: num_segments * segment_length],
        num_segments
    )
    return segments
