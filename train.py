# train.py
# Ready-to-run script for predicting Ultimate tensile strength (MPa)

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# -------- CONFIG --------
CSV_PATH = "Cu_alloys_database.csv"
TARGET_COL = "Ultimate tensile strength (MPa)"
DROP_COLUMNS = [
    "Alloy formula", "Alloy class", "DOI",
    "Aging", "Secondary thermo-mechanical process"
]
RANDOM_STATE = 42
TEST_SIZE = 0.2
# ------------------------

def load_data(path):
    print("Loading:", path)
    return pd.read_csv(path, sep=';', encoding='latin1')

def prepare_df(df):
    print("Initial shape:", df.shape)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
    df = df[df[TARGET_COL].notna()].copy()
    print("After keeping only rows with target present:", df.shape)
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        print("Dropping columns:", cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    non_numeric = [c for c in df.columns if c not in numeric_df.columns]
    if non_numeric:
        print("Note: non-numeric columns dropped:", non_numeric)
    if TARGET_COL not in numeric_df.columns:
        raise ValueError("Target column missing after numeric selection.")
    return numeric_df

def build_features_and_target(numeric_df):
    X = numeric_df.drop(columns=[TARGET_COL]).copy()
    y = numeric_df[TARGET_COL].values
    print("Feature columns (count):", len(X.columns))
    print(X.columns.tolist())
    return X, y

def impute_and_scale(X_train, X_test):
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_test_s = scaler.transform(X_test_imp)
    return X_train_s, X_test_s, imputer, scaler

def train_and_evaluate(X_train, X_test, y_train, y_test, scaler):
    results = {}

    # ---- Linear Regression ----
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    rmse_lr = mean_squared_error(y_test, y_pred_lr) ** 0.5
    r2_lr = r2_score(y_test, y_pred_lr)
    results["LinearRegression"] = {
        "model": lr, "rmse": rmse_lr, "r2": r2_lr, "y_pred": y_pred_lr
    }
    print(f"LinearRegression -> RMSE: {rmse_lr:.3f}, R2: {r2_lr:.3f}")

    # ---- Random Forest ----
    rf = RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
    r2_rf = r2_score(y_test, y_pred_rf)
    results["RandomForest"] = {
        "model": rf, "rmse": rmse_rf, "r2": r2_rf, "y_pred": y_pred_rf
    }
    print(f"RandomForest -> RMSE: {rmse_rf:.3f}, R2: {r2_rf:.3f}")

    # ---- Plots ----
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_lr, s=20)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "k--")
    plt.xlabel("Actual UTS (MPa)")
    plt.ylabel("Predicted UTS (MPa) - LinearRegression")
    plt.title("Actual vs Predicted — Linear Regression")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("actual_vs_pred_lr.png")
    plt.show()

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_rf, s=20)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "k--")
    plt.xlabel("Actual UTS (MPa)")
    plt.ylabel("Predicted UTS (MPa) - RandomForest")
    plt.title("Actual vs Predicted — Random Forest")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("actual_vs_pred_rf.png")
    plt.show()

    return results

def save_artifacts(best_model, scaler, imputer):
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/random_forest_uts.joblib")
    joblib.dump(scaler, "models/feature_scaler.joblib")
    joblib.dump(imputer, "models/feature_imputer.joblib")
    print("Saved model and preprocessing artifacts in ./models/")

def main():
    df = load_data(CSV_PATH)
    numeric_df = prepare_df(df)
    X, y = build_features_and_target(numeric_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("Train/test sizes:", X_train.shape, X_test.shape)

    X_train_s, X_test_s, imputer, scaler = impute_and_scale(X_train, X_test)
    results = train_and_evaluate(X_train_s, X_test_s, y_train, y_test, scaler)

    best_name = min(results.keys(), key=lambda k: results[k]["rmse"])
    best_model = results[best_name]["model"]
    print(f"Best model: {best_name} | "
          f"RMSE: {results[best_name]['rmse']:.3f} | "
          f"R2: {results[best_name]['r2']:.3f}")

    save_artifacts(best_model, scaler, imputer)

if __name__ == "__main__":
    main()
