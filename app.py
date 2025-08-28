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

def num_from_text(x):
    if pd.isna(x): return np.nan
    m = re.search(r"\d+(\.\d+)?", str(x))
    return float(m.group()) if m else np.nan

def build_unified_from_raw(sm_df, sl_df):
    sm2 = pd.DataFrame({
        "age": pd.to_numeric(sm_df.get("1. What is your age?"), errors="coerce"),
        "gender": sm_df.get("2. Gender").astype(str) if "2. Gender" in sm_df.columns else np.nan,
        "hours_social": sm_df.get("8. What is the average time you spend on social media every day?").apply(num_from_text) if "8. What is the average time you spend on social media every day?" in sm_df.columns else np.nan,
        "sleep_hours": np.nan,
        "work_hours": np.nan
    })
    for col in [
        "13. On a scale of 1 to 5, how much are you bothered by worries?",
        "18. How often do you feel depressed or down?",
        "20. On a scale of 1 to 5, how often do you face issues regarding sleep?"
    ]:
        if col in sm_df.columns:
            sm_df[col] = pd.to_numeric(sm_df[col], errors="coerce")
        else:
            sm_df[col] = np.nan
    sm2["target"] = (
        (sm_df["13. On a scale of 1 to 5, how much are you bothered by worries?"] >= 4) |
        (sm_df["18. How often do you feel depressed or down?"] >= 4) |
        (sm_df["20. On a scale of 1 to 5, how often do you face issues regarding sleep?"] >= 4)
    ).astype(int)

    sl2 = pd.DataFrame({
        "age": pd.to_numeric(sl_df.get("Age"), errors="coerce"),
        "gender": sl_df.get("Gender").astype(str) if "Gender" in sl_df.columns else np.nan,
        "hours_social": pd.to_numeric(sl_df.get("Screen Time"), errors="coerce") if "Screen Time" in sl_df.columns else np.nan,
        "sleep_hours": pd.to_numeric(sl_df.get("Sleep Duration"), errors="coerce") if "Sleep Duration" in sl_df.columns else np.nan,
        "work_hours": pd.to_numeric(sl_df.get("Work Hours"), errors="coerce") if "Work Hours" in sl_df.columns else np.nan,
        "target": pd.to_numeric(sl_df.get("Stress Level"), errors="coerce").apply(lambda x: 1 if x>=7 else 0) if "Stress Level" in sl_df.columns else np.nan
    })

    df = pd.concat([sm2, sl2], ignore_index=True)
    df = df.dropna(subset=["target"])
    for c in ["age","hours_social","sleep_hours","work_hours","target"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["gender"] = df["gender"].astype(str).fillna("unknown")
    return df

st.caption("App will try to load **Data/burnout_unified.csv**. If not found, upload the unified file or the two raw CSVs below.")

df = None
if os.path.exists(default_path):
    try:
        df = pd.read_csv(default_path)
        st.success(f"Loaded {default_path}")
    except Exception as e:
        st.warning(f"Could not read {default_path}: {e}")

tab1, tab2 = st.tabs(["Use unified CSV", "Upload two raw CSVs"])

with tab1:
    up_unified = st.file_uploader("Upload unified CSV (burnout_unified.csv)", type=["csv"], key="unified")
    if up_unified is not None:
        df = pd.read_csv(up_unified)
        st.success("Unified CSV uploaded.")

with tab2:
    sm_file = st.file_uploader("Upload Social Media & Mental Health CSV", type=["csv"], key="sm")
    sl_file = st.file_uploader("Upload Sleep Health & Lifestyle CSV", type=["csv"], key="sl")
    if st.button("Build unified from two files"):
        if sm_file and sl_file:
            sm_df = pd.read_csv(sm_file)
            sl_df = pd.read_csv(sl_file)
            df = build_unified_from_raw(sm_df, sl_df)
            st.success(f"Built unified dataset in-memory. Rows: {len(df)}")
        else:
            st.error("Please upload both CSVs.")

if df is None:
    st.stop()

st.write(f"Rows: {len(df)}")
st.dataframe(df.head(), use_container_width=True)

expected = ["age","gender","hours_social","sleep_hours","work_hours","target"]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing columns in dataframe: {missing}")
    st.stop()

for c in ["age","hours_social","sleep_hours","work_hours","target"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["target"])
df["gender"] = df["gender"].astype(str).fillna("unknown")

X = df[["age","gender","hours_social","sleep_hours","work_hours"]]
y = df["target"].astype(int)

num_cols = ["age","hours_social","sleep_hours","work_hours"]
cat_cols = ["gender"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipe = Pipeline([
    ("prep", pre),
    ("clf", LogisticRegression(max_iter=300))
])

if st.button("Train model"):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat)
    st.success(f"Accuracy: {acc:.3f}")
    st.code(classification_report(yte, yhat, zero_division=0), language="text")

    st.subheader("Try a Prediction")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 10, 100, int(np.nanmedian(df["age"])) if df["age"].notna().any() else 25)
    hours = c1.number_input("Hours on social/day", 0.0, 24.0, float(np.nanmedian(df["hours_social"])) if df["hours_social"].notna().any() else 3.0)
    sleep = c1.number_input("Sleep hours/day", 0.0, 24.0, float(np.nanmedian(df["sleep_hours"])) if df["sleep_hours"].notna().any() else 7.0)
    work = c1.number_input("Work/Study hours/day", 0.0, 24.0, float(np.nanmedian(df["work_hours"])) if df["work_hours"].notna().any() else 8.0)
    gender_options = sorted(df["gender"].astype(str).unique())
    gender = c2.selectbox("Gender", gender_options)

    if st.button("Predict"):
        row = pd.DataFrame([{
            "age": age, "gender": gender,
            "hours_social": hours, "sleep_hours": sleep, "work_hours": work
        }])
        pred = int(pipe.predict(row)[0])
        st.success(f"Predicted risk (0 = No risk, 1 = At risk): {pred}")
        try:
            proba = pipe.predict_proba(row)[0]
            st.write({"No risk": round(float(proba[0]),3), "At risk": round(float(proba[1]),3)})
        except:
            pass
else:
    st.info("Click **Train model** to fit on the dataset.")
