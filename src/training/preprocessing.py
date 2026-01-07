import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


def preprocess_features(X, y):
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X, y = smote.fit_resample(X, y)

    joblib.dump(imputer, "artifacts/mean_imputer.pkl")
    joblib.dump(scaler, "artifacts/minmax_scaler.pkl")

    return X, y
