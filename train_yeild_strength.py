import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# === Load dataset ===
df = pd.read_csv("Cu_alloys_database.csv", sep=';',encoding="latin1")
print("✅ Dataset loaded successfully!")

target_col = "Yield strength (MPa)"
df = df.dropna(subset=[target_col])
print(f"Dataset after dropping missing target: {df.shape}")

X = df.drop(columns=[
    "Ultimate tensile strength (MPa)",
    "Yield strength (MPa)",
    "Electrical conductivity (%IACS)",
    "DOI"
], errors="ignore")
y = df[target_col]

categorical_cols = ["Aging", "Alloy class", "Alloy formula", "Secondary thermo-mechanical process"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# === Preprocessor ===
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# === XGBoost model ===
xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

# === Pipeline ===
pipe = Pipeline(steps=[("preprocessor", preprocessor), ("xgb", xgb)])

# === Hyperparameter tuning ===
param_dist = {
    "xgb__n_estimators": [200, 300, 400],
    "xgb__max_depth": [5, 8, 10],
    "xgb__learning_rate": [0.03, 0.05, 0.1],
    "xgb__subsample": [0.8, 1],
    "xgb__colsample_bytree": [0.8, 1],
}

search = RandomizedSearchCV(pipe, param_distributions=param_dist, cv=3, n_iter=20,
                             scoring="neg_root_mean_squared_error", random_state=42, verbose=1)

search.fit(X, y)

print("\n=== Best parameters ===")
print(search.best_params_)

# === Evaluation ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nTest RMSE: {rmse:.2f} MPa")
print(f"Test R²: {r2:.3f}")

# === Save model ===
joblib.dump(best_model, "models/best_xgb_yield_model.joblib")
print("✅ Model saved to models/best_xgb_yield_model.joblib")

# === Feature Importances ===
xgb_model = best_model.named_steps["xgb"]
ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"]
cat_features = ohe.get_feature_names_out(categorical_cols)
all_features = numeric_cols + list(cat_features)

importances = pd.DataFrame({
    "Feature": all_features,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importances.to_csv("models/feature_importances_yield_xgb.csv", index=False)
print("✅ Feature importances saved to models/feature_importances_yield_xgb.csv")

# === Plot Top 15 Features ===
top_features = importances.head(15)
plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=top_features)
plt.title("Top 15 Features for Yield Strength (XGB)")
plt.tight_layout()
plt.savefig("models/top_features_yield_xgb.png")
plt.close()
print("✅ Top feature plot saved to models/top_features_yield_xgb.png")

# === Predicted vs Actual ===
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Yield Strength (MPa)")
plt.ylabel("Predicted Yield Strength (MPa)")
plt.title("Predicted vs Actual Yield Strength (XGB)")
plt.tight_layout()
plt.savefig("models/pred_vs_actual_yield_xgb.png")
plt.close()
print("✅ Predicted vs Actual plot saved to models/pred_vs_actual_yield_xgb.png")

print("\n🎯 Training completed successfully for Yield Strength (XGB)")
