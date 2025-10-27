import pandas as pd, numpy as np, joblib
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

print("✅ Dataset loaded successfully!")
df = pd.read_csv("Cu_alloys_database.csv", sep=';', encoding="latin1")

# Drop NaN targets for each case dynamically
targets = {
    "Tensile Strength (XGB)": "Ultimate tensile strength (MPa)",
    "Yield Strength (XGB)": "Yield strength (MPa)",
    "Electrical Conductivity (XGB)": "Electrical conductivity (%IACS)"
}

results = []

for model_name, target_col in targets.items():
    print(f"\n🔹 Evaluating {model_name}...")

    try:
        # Try to load model and preprocessor
        if "Tensile" in model_name:
            model = joblib.load("models/best_xgb_tensile_model_advanced.joblib")
            preprocessor = joblib.load("models/tensile_preprocessor.joblib")
        elif "Yield" in model_name:
            model = joblib.load("models/best_xgb_yield_with_cat.joblib")
            preprocessor = joblib.load("models/yield_preprocessor.joblib")
        else:
            model = joblib.load("models/best_xgb_electrical_model.joblib")
            preprocessor = joblib.load("models/electrical_preprocessor.joblib")

        df_eval = df.dropna(subset=[target_col])

        X = df_eval.drop(columns=[target_col, "DOI"], errors="ignore")
        y = df_eval[target_col]

        # Apply SAME preprocessor as used during training
        X_transformed = preprocessor.transform(X)

        # Predict and evaluate
        y_pred = model.predict(X_transformed)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        results.append({
            "Model": model_name,
            "Target": target_col,
            "RMSE": rmse,
            "R²": r2
        })

        print(f"✅ {model_name}: Evaluation completed successfully.")

    except Exception as e:
        print(f"⚠️ Skipping {model_name} due to mismatch: {e}")
        results.append({
            "Model": model_name,
            "Target": target_col,
            "RMSE": np.nan,
            "R²": np.nan
        })

# -----------------------------
# Save Results
# -----------------------------
results_df = pd.DataFrame(results)
print("\n📊 Model Comparison Results:")
print(results_df)

results_df.to_csv("models/model_comparison_summary.csv", index=False)
print("\n✅ Results saved to models/model_comparison_summary.csv")
