import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Burnout Risk Predictor", layout="wide")
st.title("🧠 Burnout Risk Predictor")

default_path = "Data/burnout_unified.csv"
st.caption("Loads Data/burnout_unified.csv by default. You can also upload another CSV with the same columns.")

uploaded = st.file_uploader("Upload unified CSV (optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(default_path)

st.write(f"Rows: {len(df)}")
st.dataframe(df.head(), use_container_width=True)

expected = ["age","gender","hours_social","sleep_hours","work_hours","target"]
present = [c for c in expected if c in df.columns]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
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
            "age": age,
            "gender": gender,
            "hours_social": hours,
            "sleep_hours": sleep,
            "work_hours": work
        }])
        pred = int(pipe.predict(row)[0])
        st.success(f"Predicted risk (0 = No risk, 1 = At risk): {pred}")
        try:
            proba = pipe.predict_proba(row)[0]
            st.write({"No risk": round(float(proba[0]),3), "At risk": round(float(proba[1]),3)})
        except:
            pass
else:
    st.info("Click **Train model** to fit on the unified dataset.")
