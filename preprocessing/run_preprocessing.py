"""
ECG Preprocessing Pipeline Runner

Pipeline:
1. Filtering (Notch + High-pass)
2. Z-score Normalization
3. 1-minute Segmentation
"""

import os
import wfdb
import pandas as pd

from filtering import filter_ecg
from normalization import normalize_ecg
from segmentation import segment_signal


RAW_DATA_DIR = "data/raw"
OUTPUT_DIR = "data/segmented"
FS = 100
SEGMENT_DURATION = 60


def run_preprocessing():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for record in os.listdir(RAW_DATA_DIR):
        if record.endswith(".dat"):
            record_name = record.replace(".dat", "")
            record_path = os.path.join(RAW_DATA_DIR, record_name)

            try:
                # Step 1: Filtering
                filtered_signal, fs = filter_ecg(record_path)

                # Step 2: Normalization
                normalized_signal = normalize_ecg(filtered_signal)

                # Step 3: Segmentation
                segments = segment_signal(
                    normalized_signal,
                    fs=fs,
                    segment_duration_sec=SEGMENT_DURATION
                )

                # Save segments
                for idx, segment in enumerate(segments):
                    segment_file = f"{record_name}_segment_{idx}.csv"
                    segment_path = os.path.join(OUTPUT_DIR, segment_file)
                    pd.DataFrame(segment).to_csv(
                        segment_path,
                        index=False,
                        header=False
                    )

                print(f"Processed: {record_name}")

            except Exception as e:
                print(f"Failed processing {record_name}: {e}")


if __name__ == "__main__":
    run_preprocessing()
