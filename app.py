import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Alloy Property Predictor", initial_sidebar_state="expanded")

# -------------------
# Helper functions
# -------------------
@st.cache_data
def load_data(path="Cu_alloys_database.csv"):
    return pd.read_csv(path, sep=';', encoding="latin1")

@st.cache_data
def load_models(models_dir="models"):
    models = {}
    expected = {
        "Hardness": "best_rf_hardness.joblib",
        "Yield Strength": "best_xgb_yield_model.joblib",
        "Tensile Strength": "best_xgb_tensile_model_advanced.joblib",
        "Electrical Conductivity": "best_xgb_electrical_model.joblib"
    }
    for key, fname in expected.items():
        p = os.path.join(models_dir, fname)
        if os.path.exists(p):
            try:
                models[key] = joblib.load(p)
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")
        else:
            st.warning(f"Model file not found: {p}")
    return models

def infer_input_fields(df):
    drop_cols = ["DOI", "Hardness (HV)", "Yield strength (MPa)",
                 "Ultimate tensile strength (MPa)", "Electrical conductivity (%IACS)"]
    features = [c for c in df.columns if c not in drop_cols]
    numeric = [c for c in features if df[c].dtype in [np.float64, np.int64]]
    categorical = [c for c in features if df[c].dtype == object]
    return numeric, categorical

def prefill_example():
    return {
        "Cu": 99.2,
        "Sn": 0.0,
        "Ni": 0.4,
        "Zn": 0.1,
        "Cr": 0.1,
        "Zr": 0.06,
        "Tss (K)": 720,
        "Tag (K)": 480,
        "tss (h)": 2,
        "tag (h)": 5,
        "Aging": "Y",
        "Alloy class": "Cu-Ti alloys",
        "Alloy formula": "Cu-0.1Cr-0.06Zr",
        "Secondary thermo-mechanical process": "Cold rolled"
    }

def make_input_row(numeric, categorical, df):
    st.markdown("### 🧩 Enter material composition & processing parameters")

    # Prefill toggle
    if st.button("✨ Prefill Example Input"):
        st.session_state.prefilled = True
    else:
        if "prefilled" not in st.session_state:
            st.session_state.prefilled = False

    inp = {}
    pre = prefill_example() if st.session_state.prefilled else {}

    with st.form("input_form"):
        st.subheader("Numeric features")
        cols = st.columns(3)
        for i, col in enumerate(numeric):
            default = pre.get(col, float(df[col].median()) if pd.notna(df[col].median()) else 0.0)
            with cols[i % 3]:
                inp[col] = st.number_input(col, value=float(default), format="%.4f", step=0.01)

        st.subheader("Categorical features")
        for col in categorical:
            choices = sorted(df[col].dropna().unique().astype(str).tolist())
            if len(choices) > 0:
                if st.session_state.prefilled and col in pre and pre[col] in choices:
                    idx = choices.index(pre[col])
                else:
                    idx = 0
                inp[col] = st.selectbox(col, options=choices, index=idx)
            else:
                inp[col] = st.text_input(col, value=pre.get(col, "Unknown"))

        submitted = st.form_submit_button("Predict")

    row = pd.DataFrame([inp])
    for col in numeric:
        row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0.0)
    return submitted, row

def get_feature_names_from_pipeline(model, sample_df):
    try:
        if hasattr(model, "named_steps"):
            pre = model.named_steps.get("preprocessor") or model.named_steps.get("preprocess")
            if pre is not None and hasattr(pre, "get_feature_names_out"):
                return list(pre.get_feature_names_out(sample_df.columns))
    except Exception:
        pass
    return list(sample_df.columns)

# -------------------
# Streamlit UI
# -------------------
st.title("🔬 Copper Alloy Property Predictor")

df = load_data()
models = load_models()
numeric_cols, categorical_cols = infer_input_fields(df)

col1, col2 = st.columns([2, 1])
with col1:
    submitted, input_row = make_input_row(numeric_cols, categorical_cols, df)

with col2:
    st.markdown("### 🧠 Choose Property to Predict")
    property_choice = st.selectbox("Property", ["Hardness", "Yield Strength", "Tensile Strength", "Electrical Conductivity"])
    st.markdown("### 🧩 Model Status")
    if property_choice in models:
        st.success(f"Model loaded for: {property_choice}")
    else:
        st.error(f"No model found for: {property_choice}")

# -------------------
# Prediction Section
# -------------------
if submitted:
    if property_choice not in models:
        st.error("Selected model not loaded. Check models/ folder.")
    else:
        model = models[property_choice]
        drop_cols = ["DOI", "Hardness (HV)", "Yield strength (MPa)",
                     "Ultimate tensile strength (MPa)", "Electrical conductivity (%IACS)"]
        features = [c for c in df.columns if c not in drop_cols]
        for c in features:
            if c not in input_row.columns:
                input_row[c] = df[c].median() if df[c].dtype in [np.float64, np.int64] else str(df[c].dropna().astype(str).mode().iloc[0]) if not df[c].dropna().empty else "Unknown"
        input_row = input_row[features]

        # ✅ Universal prediction handler
        try:
            if property_choice == "Tensile Strength":
                # Add missing columns expected by model
                missing_cols = ["Hardness (HV)", "Electrical conductivity (%IACS)", "Yield strength (MPa)"]
                for col in missing_cols:
                    if col not in input_row.columns:
                        input_row[col] = 0.0  # dummy numeric

                numeric_cols_model = [c for c in input_row.columns if c not in
                                      ["Aging", "Alloy class", "Alloy formula", "Secondary thermo-mechanical process"]]
                input_row["Total_alloying_content"] = input_row[numeric_cols_model].sum(axis=1) - input_row["Cu"]
                input_row["Heat_treatment_factor"] = input_row[["Tss (K)", "Tag (K)", "tss (h)", "tag (h)"]].mean(axis=1)
                input_row["Cu_ratio"] = input_row["Cu"] / (input_row[numeric_cols_model].sum(axis=1) + 1e-6)
                input_row["log_Hardness"] = np.log(input_row["Hardness (HV)"] + 1)
                pred = model.predict(input_row)[0]

            elif property_choice == "Electrical Conductivity":
                df_all = load_data()
                drop_cols_ec = ["Electrical conductivity (%IACS)"]
                X_all = df_all.drop(columns=drop_cols_ec, errors="ignore")
                combined = pd.concat([X_all, input_row], ignore_index=True)
                X_enc = pd.get_dummies(combined, drop_first=True)
                input_encoded = X_enc.tail(1)
                pred = model.predict(input_encoded)[0]

            else:
                pred = model.predict(input_row)[0]

        except Exception as e:
            st.error("⚠️ Prediction failed due to feature mismatch.")
            st.code(str(e))
            pred = None

        st.markdown("## 🧾 Prediction")
        if pred is not None:
            unit_map = {"Hardness": "HV", "Yield Strength": "MPa",
                        "Tensile Strength": "MPa", "Electrical Conductivity": "%IACS"}
            st.metric(label=f"Predicted {property_choice}", value=f"{pred:.3f} {unit_map[property_choice]}")

            st.markdown("### Input Values Used")
            st.dataframe(input_row.T.astype(str), width='stretch')
        else:
            st.error("Prediction could not be completed.")
