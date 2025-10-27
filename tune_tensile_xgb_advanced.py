import pandas as pd, numpy as np, joblib
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# === Load dataset ===
df = pd.read_csv("Cu_alloys_database.csv", sep=';', encoding="latin1")
target = "Ultimate tensile strength (MPa)"
df = df.dropna(subset=[target])
print("✅ Dataset loaded successfully!")

# === Feature Engineering ===
numeric_cols = [c for c in df.columns if df[c].dtype != 'object' and c not in [target]]
df["Total_alloying_content"] = df[numeric_cols].sum(axis=1) - df["Cu"]
df["Heat_treatment_factor"] = df[["Tss (K)", "Tag (K)", "tss (h)", "tag (h)"]].mean(axis=1)
df["Cu_ratio"] = df["Cu"] / (df[numeric_cols].sum(axis=1) + 1e-6)
df["log_Hardness"] = np.log(df["Hardness (HV)"] + 1)

categorical_cols = ["Aging", "Alloy class", "Alloy formula", "Secondary thermo-mechanical process"]

X = df.drop(columns=[target, "DOI"], errors="ignore")
y = df[target]

# === Preprocessing ===
preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder='passthrough'
)

xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

param_dist = {
    "xgb__n_estimators": [200, 400, 600, 800],
    "xgb__max_depth": [5, 7, 9, 11],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
    "xgb__colsample_bytree": [0.7, 0.8, 1.0],
    "xgb__reg_lambda": [1, 2, 3],
}

pipe = Pipeline(steps=[("preprocessor", preprocess), ("xgb", xgb)])

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Hyperparameter Tuning ===
search = RandomizedSearchCV(pipe, param_distributions=param_dist,
                            n_iter=25, cv=3, verbose=1, scoring='r2', n_jobs=-1)

search.fit(X_train, y_train)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

# === Metrics ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Best Params: {search.best_params_}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R²: {r2:.3f}")

# === Save Model ===
joblib.dump(best_model, "models/best_xgb_tensile_model_advanced.joblib")
print("✅ Saved: models/best_xgb_tensile_model_advanced.joblib")

# === Save feature names for comparison consistency ===
# (Extract from the pipeline’s preprocessing output)
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
joblib.dump(feature_names, "models/tensile_feature_names.joblib")
print("✅ Saved feature names to models/tensile_feature_names.joblib")
