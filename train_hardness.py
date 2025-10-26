import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import os

# ======================================
# 1️⃣ Load dataset
# ======================================
try:
    df = pd.read_csv("Cu_alloys_database.csv",sep=';', encoding="latin1")
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

# ======================================
# 2️⃣ Target selection
# ======================================
target_col = "Hardness (HV)"
if target_col not in df.columns:
    print(f"❌ Target column '{target_col}' not found.")
    exit()

# Drop missing target rows
df = df.dropna(subset=[target_col])
print(f"Dataset after dropping missing target: {df.shape}")

# ======================================
# 3️⃣ Define features and target
# ======================================
X = df.drop(columns=[
    "Yield strength (MPa)",
    "Ultimate tensile strength (MPa)",
    "Electrical conductivity (%IACS)",
    "DOI",  # remove irrelevant column
    target_col  # target removed from features
])
y = df[target_col]

# ======================================
# 4️⃣ Identify numeric and categorical columns
# ======================================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = ["Aging", "Secondary thermo-mechanical process", "Alloy class"]

# ======================================
# 5️⃣ Preprocessor
# ======================================
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# ======================================
# 6️⃣ Define pipeline
# ======================================
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=42))
])

# ======================================
# 7️⃣ Split dataset
# ======================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================================
# 8️⃣ Hyperparameter tuning
# ======================================
param_distributions = {
    'rf__n_estimators': np.arange(50, 200, 10),
    'rf__max_depth': np.arange(10, 60, 5),
    'rf__min_samples_split': np.arange(2, 10),
    'rf__min_samples_leaf': np.arange(1, 8),
    'rf__max_features': np.linspace(0.3, 1.0, 10),
    'rf__bootstrap': [True, False]
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_distributions,
    n_iter=30,
    cv=3,
    scoring='r2',
    random_state=42,
    verbose=2,
    n_jobs=-1
)

search.fit(X_train, y_train)

# ======================================
# 9️⃣ Evaluate
# ======================================
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Best parameters ===")
print(search.best_params_)
print(f"\nTest RMSE: {rmse:.2f}")
print(f"Test R2: {r2:.3f}")

# ======================================
# 🔟 Save model
# ======================================
os.makedirs("models", exist_ok=True)
model_path = "models/best_rf_hardness.joblib"
dump(best_model, model_path)
print(f"\n✅ Model saved to {model_path}")
