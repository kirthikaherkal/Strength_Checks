# ==============================
# tune_yield_strength_xgb.py
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib
import os

# ==============================
# Step 1: Load Dataset
# ==============================
print("✅ Loading dataset...")
df = pd.read_csv("Cu_alloys_database.csv", sep=';',encoding="latin1")

# Drop rows where Yield strength is missing
df = df.dropna(subset=["Yield strength (MPa)"])
print(f"Dataset after dropping missing target: {df.shape}")

# ==============================
# Step 2: Split features and target
# ==============================
X = df.drop(columns=["Yield strength (MPa)"])
y = df["Yield strength (MPa)"]

# Convert non-numeric columns if any
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Step 3: Define Pipeline
# ==============================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        tree_method="hist"
    ))
])

# ==============================
# Step 4: Parameter Search Space
# ==============================
param_distributions = {
    "xgb__n_estimators": np.arange(100, 500, 50),
    "xgb__max_depth": np.arange(3, 10),
    "xgb__learning_rate": np.linspace(0.01, 0.3, 20),
    "xgb__subsample": np.linspace(0.7, 1.0, 10),
    "xgb__colsample_bytree": np.linspace(0.7, 1.0, 10),
    "xgb__min_child_weight": np.arange(1, 10),
}

# ==============================
# Step 5: Randomized Search
# ==============================
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_distributions,
    n_iter=50,
    cv=3,
    verbose=2,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

# ==============================
# Step 6: Evaluate Model
# ==============================
best_model = search.best_estimator_

y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Best Parameters ===")
print(search.best_params_)
print(f"\nTest RMSE: {rmse:.2f} MPa")
print(f"Test R²: {r2:.3f}")

# ==============================
# Step 7: Feature Importance
# ==============================
xgb_model = best_model.named_steps["xgb"]
importances = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb_model.feature_importances_
}).sort_values(by="importance", ascending=False)

os.makedirs("models", exist_ok=True)
importances.to_csv("models/feature_importances_yield_xgb.csv", index=False)
joblib.dump(best_model, "models/best_xgb_yield_model.joblib")

# ==============================
# Step 8: Plot Feature Importance
# ==============================
plt.figure(figsize=(8, 6))
plt.barh(importances["feature"][:15], importances["importance"][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Features Affecting Yield Strength (XGBoost)")
plt.tight_layout()
plt.savefig("models/top_features_yield_xgb.png")

print("\n✅ Model saved to models/best_xgb_yield_model.joblib")
print("✅ Feature importances saved to models/feature_importances_yield_xgb.csv")
print("✅ Top feature plot saved to models/top_features_yield_xgb.png")

print("\n🎯 Goal: Try to achieve RMSE < 50 MPa and R² ≈ 1.0")
