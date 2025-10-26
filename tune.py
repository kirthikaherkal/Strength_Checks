import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform
import joblib
import matplotlib.pyplot as plt
import os

# ======================================================
# 1️⃣ Load dataset
# ======================================================
df = pd.read_csv("Cu_alloys_database.csv", sep=';',encoding="latin1")
print("✅ Dataset loaded successfully!")

# Strip spaces from column names to avoid hidden whitespace
df.columns = df.columns.str.strip()

# ======================================================
# 2️⃣ Define features and target
# ======================================================
numeric_features = [
    "Cu", "Al", "Ag", "Be", "Cr", "Fe", "Ni", "Si", "Ti", "Zn",
    "Tss (K)", "tss (h)", "Tag (K)", "tag (h)",
    "Hardness (HV)", "Yield strength (MPa)",
    "Electrical conductivity (%IACS)", "CR reduction (%)"
]

categorical_features = ["Aging", "Secondary thermo-mechanical process"]

TARGET_COL = "Ultimate tensile strength (MPa)"

# Drop rows with missing target
df = df.dropna(subset=[TARGET_COL])
print(f"Dataset after dropping missing target: {df.shape}")

# Keep only relevant columns
df = df[numeric_features + categorical_features + [TARGET_COL]]

# Split features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ======================================================
# 3️⃣ Split data
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================================
# 4️⃣ Preprocessing
# ======================================================
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ======================================================
# 5️⃣ Model Pipeline
# ======================================================
rf = RandomForestRegressor(random_state=42)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("rf", rf)])

# ======================================================
# 6️⃣ Hyperparameter tuning
# ======================================================
param_dist = {
    "rf__n_estimators": randint(100, 300),
    "rf__max_depth": randint(10, 60),
    "rf__min_samples_split": randint(2, 10),
    "rf__min_samples_leaf": randint(1, 10),
    "rf__max_features": uniform(0.3, 0.7),
    "rf__bootstrap": [True, False],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=30,
    cv=3,
    scoring="neg_root_mean_squared_error",
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train, y_train)

# ======================================================
# 7️⃣ Evaluate model
# ======================================================
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)

print("\n=== Best parameters ===")
print(search.best_params_)
print(f"\nTest RMSE: {rmse:.2f} MPa")
print(f"Test R2: {r2:.3f}")

# ======================================================
# 8️⃣ Feature importance (from Random Forest)
# ======================================================
rf_model = best_model.named_steps["rf"]

# Get transformed feature names
num_features_out = numeric_features
cat_features_out = list(
    best_model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
    .get_feature_names_out(categorical_features)
)

all_features = num_features_out + cat_features_out
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# ======================================================
# 9️⃣ Save model and feature importances
# ======================================================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_rf_uts_with_cat.joblib")
feature_importance_df.to_csv("models/feature_importances_with_cat.csv", index=False)

# ======================================================
# 🔟 Plot feature importance
# ======================================================
top_features = feature_importance_df.head(15)
plt.figure(figsize=(8, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances (with categorical features)")
plt.tight_layout()
plt.savefig("models/top_features_with_cat.png", dpi=300)
plt.close()

print("\n✅ Model saved to models/best_rf_uts_with_cat.joblib")
print("✅ Feature importances saved to models/feature_importances_with_cat.csv")
print("✅ Top 15 feature plot saved to models/top_features_with_cat.png")
print("\nRecommendation:\n- Check if RMSE improved.\n- If not, we can try gradient boosting or XGBoost next.")
