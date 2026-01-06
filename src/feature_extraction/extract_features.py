"""
Main feature extraction runner
"""

import os
import numpy as np
import pandas as pd

from time_domain import extract_time_domain_features
from frequency_domain import extract_frequency_domain_features
from nonlinear_domain import extract_nonlinear_features


SEGMENT_DIR = "data/segmented"
ANNOTATION_DIR = "data/annotations"
OUTPUT_FILE = "data/features/ecg_features.csv"
FS = 100


def read_annotations(annotation_file):
    labels = []
    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            labels.append(parts[2])
    return labels


def run_feature_extraction():
    all_features = []

    for file in os.listdir(SEGMENT_DIR):
        if file.endswith(".csv"):
            record_name, seg_id = file.replace(".csv", "").split("_segment_")
            seg_id = int(seg_id)

            segment = pd.read_csv(
                os.path.join(SEGMENT_DIR, file),
                header=None
            ).values.flatten()

            annotations = read_annotations(
                os.path.join(ANNOTATION_DIR, record_name + ".txt")
            )

            if seg_id >= len(annotations):
                continue

            label = 1 if annotations[seg_id] == "A" else 0

            time_feats = extract_time_domain_features(segment, FS)
            if time_feats is None:
                continue

            rri = np.diff(
                np.where(segment > np.mean(segment))[0]
            ) / FS

            freq_feats = extract_frequency_domain_features(rri)
            nonlin_feats = extract_nonlinear_features(rri)

            features = {
                **time_feats,
                **freq_feats,
                **nonlin_feats,
                "record_name": record_name,
                "segment_number": seg_id,
                "apnea_label": label,
            }

            all_features.append(features)

    os.makedirs("data/features", exist_ok=True)
    pd.DataFrame(all_features).to_csv(OUTPUT_FILE, index=False)
    print("Feature extraction completed.")


if __name__ == "__main__":
    run_feature_extraction()
