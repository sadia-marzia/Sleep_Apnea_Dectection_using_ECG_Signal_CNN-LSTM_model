import pandas as pd
import joblib
from training.preprocessing import preprocess_features
from training.trainer import train_kfold

# Load data
df = pd.read_csv("data/all_features_file_extended_extra_cleaned.csv")

X = df[feature_columns].values
y = df["apnea_label"].values

# Save feature order
joblib.dump(feature_columns, "artifacts/feature_order.pkl")

# Preprocess
X, y = preprocess_features(X, y)

# Reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train
model, metrics = train_kfold(X, y)

# Save model
model.save("artifacts/cnn_lstm_apneamodel.keras")

print("\nAverage Metrics:")
for k, v in metrics.items():
    print(f"{k}: {sum(v)/len(v):.4f}")
