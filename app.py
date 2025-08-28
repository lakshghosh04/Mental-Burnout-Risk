import streamlit as st
import pandas as pd
import numpy as np
import os, re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Burnout Risk Predictor", layout="wide")
st.title("🧠 Burnout Risk Predictor")

default_path = "Data/burnout_unified.csv"

# ---------- Load data ----------
if os.path.exists(default_path):
    df = pd.read_csv(default_path)
    st.caption(f"Loaded {default_path}")
else:
    st.error("Data/burnout_unified.csv not found. Please add it to the repo or upload one.")
    up = st.file_uploader("Upload unified CSV", type=["csv"])
    if not up:
        st.stop()
    df = pd.read_csv(up)
    st.caption("Using uploaded file")

st.write(f"Rows: {len(df)}")
st.dataframe(df.head(), use_container_width=True)

# ---------- Expect these columns ----------
expected = ["age","gender","hours_social","sleep_hours","work_hours","target"]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ---------- Coerce & clean ----------
# Coerce numerics
for c in ["age","hours_social","sleep_hours","work_hours","target"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Replace inf with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with missing target
df = df.dropna(subset=["target"])

# Ensure binary int target
df["target"] = df["target"].round().astype(int)

# Basic sanity: need at least 2 classes
classes = sorted(df["target"].unique().tolist())
if len(classes) < 2:
    st.error(f"Target has only one class present: {classes}. You need both 0 and 1.")
    st.stop()

# Fill numerics with median (column-wise)
num_cols = ["age","hours_social","sleep_hours","work_hours"]
for c in num_cols:
    if c in df.columns:
        med = df[c].median(skipna=True)
        df[c] = df[c].fillna(med)

# Guard against columns entirely NaN (median = NaN). Fill safe defaults.
defaults = {"age": 25.0, "hours_social": 2.0, "sleep_hours": 7.0, "work_hours": 8.0}
for c in num_cols:
    if df[c].isna().all():
        df[c] = defaults[c]

# Clean categoricals
df["gender"] = df["gender"].astype(str).fillna("unknown").replace({"nan":"unknown","None":"unknown"})

# Optional: clamp extreme outliers to reduce numerical issues (very simple winsorization)
df["age"] = df["age"].clip(lower=10, upper=100)
df["hours_social"] = df["hours_social"].clip(lower=0, upper=24)
df["sleep_hours"] = df["sleep_hours"].clip(lower=0, upper=24)
df["work_hours"] = df["work_hours"].clip(lower=0, upper=24)

# ---------- Model data ----------
X = df[["age","gender","hours_social","sleep_hours","work_hours"]]
y = df["target"]

cat_cols = ["gender"]

pre = ColumnTransformer([
    ("num", StandardScaler(), ["age","hours_social","sleep_hours","work_hours"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipe = Pipeline([
    ("prep", pre),
    ("clf", LogisticRegression(max_iter=500))
])

st.markdown("---")
st.subheader("Train")
test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

# Try stratified split; if it fails (e.g., class imbalance), fall back to plain split.
try:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
except Exception:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    st.warning("Stratified split failed; using plain split (class imbalance or too few samples).")

if st.button("Train model"):
    # Final guard: confirm X has no NaN/inf
    if not np.isfinite(Xtr.select_dtypes(include=[np.number]).values).all():
        st.error("Training data still contains non-finite numbers after cleaning.")
        st.stop()

    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat)
    st.success(f"Accuracy: {acc:.3f}")
    st.code(classification_report(yte, yhat, zero_division=0), language="text")

    st.markdown("---")
    st.subheader("Try a Prediction")

    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 10, 100, int(np.nanmedian(df["age"])))
    hours = c1.number_input("Hours on social/day", 0.0, 24.0, float(np.nanmedian(df["hours_social"])))
    sleep = c1.number_input("Sleep hours/day", 0.0, 24.0, float(np.nanmedian(df["sleep_hours"])))
    work = c1.number_input("Work/Study hours/day", 0.0, 24.0, float(np.nanmedian(df["work_hours"])))
    gender_options = sorted(df["gender"].astype(str).unique())
    gender = c2.selectbox("Gender", gender_options)

    if st.button("Predict"):
        row = pd.DataFrame([{
            "age": age,
            "gender": gender or "unknown",
            "hours_social": hours,
            "sleep_hours": sleep,
            "work_hours": work
        }])

        pred = int(pipe.predict(row)[0])
        st.success(f"Predicted risk (0 = No risk, 1 = At risk): {pred}")

        try:
            proba = pipe.predict_proba(row)[0]
            st.write({"No risk": round(float(proba[0]),3), "At risk": round(float(proba[1]),3)})
        except Exception:
            pass
else:
    st.info("Adjust test size if needed, then click **Train model**.")
