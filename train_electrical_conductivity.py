import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Step 1: Load dataset
print("✅ Loading dataset...")
df = pd.read_csv("Cu_alloys_database.csv", sep=';',encoding="latin1")

# Step 2: Drop rows with missing target values
target_col = "Electrical conductivity (%IACS)"
df = df.dropna(subset=[target_col])
print(f"Dataset after dropping missing '{target_col}':", df.shape)

# Step 3: Separate features and target
X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
y = df[target_col]

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Define XGBoost pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42))
])

# Step 6: Define hyperparameter grid
param_grid = {
    "xgb__n_estimators": np.arange(100, 600, 50),
    "xgb__max_depth": np.arange(3, 12),
    "xgb__learning_rate": np.linspace(0.01, 0.3, 20),
    "xgb__subsample": np.linspace(0.7, 1.0, 10),
    "xgb__colsample_bytree": np.linspace(0.7, 1.0, 10),
    "xgb__min_child_weight": np.arange(1, 10)
}

# Step 7: Run Randomized Search
print("🚀 Running RandomizedSearchCV...")
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    n_iter=50,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=2
)

search.fit(X_train, y_train)

# Step 8: Evaluate
best = search.best_estimator_
y_pred = best.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Best parameters ===")
print(search.best_params_)
print(f"\nTest RMSE: {rmse:.2f}")
print(f"Test R²: {r2:.3f}")

# Step 9: Save model & feature importances
os.makedirs("models", exist_ok=True)
joblib.dump(best, "models/best_xgb_electrical_model.joblib")
print("✅ Model saved to models/best_xgb_electrical_model.joblib")

importances = pd.DataFrame({
    "feature": X.columns,
    "importance": best.named_steps["xgb"].feature_importances_
}).sort_values("importance", ascending=False)
importances.to_csv("models/feature_importances_electrical_xgb.csv", index=False)
print("✅ Feature importances saved to models/feature_importances_electrical_xgb.csv")

# Step 10: Plot feature importances
plt.figure(figsize=(8, 6))
plt.barh(importances["feature"][:15], importances["importance"][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Features Affecting Electrical Conductivity (XGBoost)")
plt.tight_layout()
plt.savefig("models/top_features_electrical_xgb.png")
print("✅ Feature plot saved to models/top_features_electrical_xgb.png")

# Step 11: Plot predicted vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Electrical Conductivity (%IACS)")
plt.ylabel("Predicted Electrical Conductivity (%IACS)")
plt.title(f"Predicted vs Actual (RMSE={rmse:.2f}, R²={r2:.3f})")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.tight_layout()
plt.savefig("models/pred_vs_actual_electrical_xgb.png")
print("✅ Predicted vs Actual plot saved to models/pred_vs_actual_electrical_xgb.png")

print("\n🎯 Recommendation:")
if rmse < 50 and r2 > 0.95:
    print("Excellent! Model meets target performance (RMSE < 50, R² ≈ 1).")
else:
    print("Good! Try tuning learning_rate or increasing estimators for even better performance.")
