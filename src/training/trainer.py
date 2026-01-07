import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from training.model import build_cnn_lstm
from training.metrics import compute_metrics
from training.config import *


def train_kfold(X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    metrics = {"spec": [], "sens": [], "f1": [], "auc": []}
    final_model = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_cnn_lstm((X.shape[1], 1))
        model.compile(
            optimizer=Adam(LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=4, factor=0.5)
            ],
            verbose=1
        )

        y_val_pred = model.predict(X_val)
        spec, sens, f1, auc = compute_metrics(y_val, y_val_pred)

        metrics["spec"].append(spec)
        metrics["sens"].append(sens)
        metrics["f1"].append(f1)
        metrics["auc"].append(auc)

        final_model = model

    return final_model, metrics
