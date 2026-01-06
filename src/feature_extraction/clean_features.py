"""
Remove samples with missing labels or feature values
"""

import pandas as pd


INPUT_FILE = "data/features/ecg_features.csv"
OUTPUT_FILE = "data/features/ecg_features_cleaned.csv"


def clean_feature_file():
    df = pd.read_csv(INPUT_FILE)

    df_cleaned = df.dropna()
    df_cleaned.to_csv(OUTPUT_FILE, index=False)

    print("Cleaned feature file saved.")


if __name__ == "__main__":
    clean_feature_file()
