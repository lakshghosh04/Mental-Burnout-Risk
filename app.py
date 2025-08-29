import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Burnout Risk Predictor", layout="wide")
st.title("🧠 Burnout Risk Predictor")

DEFAULT_PATH = "Data/burnout_unified.csv"

def load_df():
    if os.path.exists(DEFAULT_PATH):
        st.caption(f"Loaded {DEFAULT_PATH}")
        return pd.read_csv(DEFAULT_PATH)
    st.error("Data/burnout_unified.csv not found.")
    up = st.file_uploader("Upload unified CSV (columns: age, gender, hours_social, sleep_hours, work_hours, target)", type=["csv"])
    if up is None:
        st.stop()
    st.caption("Using uploaded file")
    return pd.read_csv(up)

df = load_df()
st.write(f"Rows: {len(df)}")
st.dataframe(df.head(), use_container_width=True)

expected = ["age","gender","hours_social","sleep_hours","work_hours","target"]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

for c in ["age","hours_social","sleep_hours","work_hours","target"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=["target"])
df["target"] = df["target"].round().astype(int)

all_num = ["age","hours_social","sleep_hours","work_hours"]
num_cols = [c for c in all_num if df[c].notna().sum() > 0]
if len(num_cols) == 0:
    st.error("No usable numeric columns found.")
    st.stop()

for c in num_cols:
    med = df[c].median(skipna=True)
    df[c] = df[c].fillna(med if not np.isnan(med) else 0.0)

if "gender" in df.columns:
    df["gender"] = df["gender"].astype(str).fillna("unknown").replace({"nan":"unknown","None":"unknown"})
    cat_cols = ["gender"]
else:
    cat_cols = []

if "age" in df.columns: df["age"] = df["age"].clip(10, 100)
for c in ["hours_social","sleep_hours","work_hours"]:
    if c in df.columns: df[c] = df[c].clip(0, 24)

X = df[num_cols + cat_cols]
y = df["target"]

classes = sorted(y.unique().tolist())
if len(classes) < 2:
    st.error(f"Target has only one class present: {classes}. Need both 0 and 1.")
    st.stop()

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

pipe = Pipeline([
    ("prep", pre),
    ("clf", LogisticRegression(max_iter=500))
])

st.markdown("---")
st.subheader("Train")
test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

try:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
except Exception:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
    st.warning("Stratified split failed; using plain split.")

if "model" not in st.session_state:
    st.session_state.model = None
if "num_cols" not in st.session_state:
    st.session_state.num_cols = num_cols
if "cat_cols" not in st.session_state:
    st.session_state.cat_cols = cat_cols

c_train, c_status = st.columns([1,1])
if c_train.button("Train model"):
    if not np.isfinite(Xtr.select_dtypes(include=[np.number]).values).all():
        st.error("Training data contains non-finite numbers after cleaning.")
        st.stop()
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat)
    st.session_state.model = pipe
    st.session_state.num_cols = num_cols
    st.session_state.cat_cols = cat_cols
    c_status.success(f"Accuracy: {acc:.3f}")
    st.code(classification_report(yte, yhat, zero_division=0), language="text")
else:
    if st.session_state.model is None:
        st.info("Click **Train model** to fit on the dataset.")
    else:
        st.success("Model is trained and ready.")

st.markdown("---")
st.subheader("Try a Prediction")

model_ready = st.session_state.model is not None
disabled = not model_ready

c1, c2 = st.columns(2)
inputs = {}
if "age" in num_cols:
    inputs["age"] = c1.number_input("Age", 10, 100, int(np.nanmedian(df["age"])), disabled=disabled)
if "hours_social" in num_cols:
    inputs["hours_social"] = c1.number_input("Hours on social/day", 0.0, 24.0, float(np.nanmedian(df["hours_social"])), disabled=disabled)
if "sleep_hours" in num_cols:
    inputs["sleep_hours"] = c1.number_input("Sleep hours/day", 0.0, 24.0, float(np.nanmedian(df["sleep_hours"])), disabled=disabled)
if "work_hours" in num_cols:
    inputs["work_hours"] = c1.number_input("Work/Study hours/day", 0.0, 24.0, float(np.nanmedian(df["work_hours"])), disabled=disabled)
if "gender" in cat_cols:
    inputs["gender"] = c2.selectbox("Gender", sorted(df["gender"].astype(str).unique()), disabled=disabled)

th = st.slider("Decision threshold for 'At risk' (class 1)", 0.1, 0.9, 0.5, 0.05, disabled=disabled)
predict_clicked = st.button("Predict", disabled=disabled)

if predict_clicked and model_ready:
    row = pd.DataFrame([inputs])
    proba = st.session_state.model.predict_proba(row)[0]
    risk = float(proba[1])
    pred = 1 if risk >= th else 0
    st.success(f"Predicted risk (0 = No risk, 1 = At risk): {pred}")
    st.metric("At-risk probability", f"{risk*100:.1f}%")
    st.progress(risk)

    if risk < 0.3:
        st.success("Low burnout risk. Maintain balanced social time and 7–8 hours sleep.")
    elif risk < 0.6:
        st.warning("Moderate risk. Consider reducing social media time and improving sleep consistency.")
    else:
        st.error("High risk. Prioritize sleep hygiene and limit screen time; consider wellness support.")
