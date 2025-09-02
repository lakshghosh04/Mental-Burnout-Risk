import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, roc_curve, average_precision_score, confusion_matrix
import joblib

st.set_page_config(layout="wide", page_title="Burnout Risk")

NUM_COLS_BASE = ["age", "hours_social", "sleep_hours", "work_hours"]
CAT_COLS_BASE = ["gender"]

if "models" not in st.session_state:
    st.session_state.models = []
if "active_model_id" not in st.session_state:
    st.session_state.active_model_id = None
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = {"risk": None}
if "data_version" not in st.session_state:
    st.session_state.data_version = "-"
if "last_train" not in st.session_state:
    st.session_state.last_train = None

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def load_default_csv():
    try:
        df = pd.read_csv("Data/unified_with_productivity.csv")
        return df
    except Exception:
        return None

def hash_bytes(b):
    return hashlib.md5(b).hexdigest()[:8]

@st.cache_data(show_spinner=False)
def load_data(file):
    if file is None:
        df = load_default_csv()
        if df is None:
            return None, None
        b = df.to_csv(index=False).encode()
        return df, hash_bytes(b)
    else:
        b = file.read()
        df = pd.read_csv(io.BytesIO(b))
        return df, hash_bytes(b)

def normalize_schema(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    alias = {"burnout": "target", "burnout_risk": "target", "label": "target", "risk": "target"}
    for a, b in alias.items():
        if a in df.columns and b not in df.columns:
            df = df.rename(columns={a: b})
    if "productivity" not in df.columns:
        prod_guess = next((c for c in df.columns if c.startswith("product") and c != "target"), None)
        if prod_guess:
            df = df.rename(columns={prod_guess: "productivity"})
    return df

def clean_and_filter(df, target_col):
    use = df[df[target_col].notna()].copy()
    use["gender"] = use["gender"].astype(str)
    for c in ["age", "hours_social", "sleep_hours", "work_hours"]:
        if c in use.columns:
            use[c] = pd.to_numeric(use[c], errors="coerce")
            if c == "age":
                use[c] = use[c].clip(0, 100)
            else:
                use[c] = use[c].clip(0, 24)
    num_cols = [c for c in NUM_COLS_BASE if c in use.columns and not use[c].isna().all()]
    cat_cols = [c for c in CAT_COLS_BASE if c in use.columns]
    X = use[num_cols + cat_cols]
    y = use[target_col].astype(int).values
    return X, y, num_cols, cat_cols

def build_pipeline(num_cols, cat_cols, model_type, class_weight, calibration_method):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])
    if model_type == "Random Forest":
        base = RandomForestClassifier(n_estimators=300, random_state=42, class_weight=("balanced" if class_weight else None))
    else:
        base = LogisticRegression(max_iter=500, class_weight=("balanced" if class_weight else None))
    if calibration_method == "Platt (sigmoid)":
        clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    elif calibration_method == "Isotonic":
        clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    else:
        clf = base
    return Pipeline([("prep", pre), ("clf", clf)])

def cv_scores(pipe, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1": "f1"}
    out = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=None)
    return {k: float(np.mean(v)) for k, v in out.items() if k.startswith("test_")}

def get_active_model():
    if st.session_state.active_model_id is None:
        return None
    for m in st.session_state.models:
        if m["id"] == st.session_state.active_model_id:
            return m
    return None

def set_active_model(mid):
    st.session_state.active_model_id = mid

def save_artifact(model_record):
    mid = model_record["id"]
    joblib.dump(model_record, os.path.join(ARTIFACTS_DIR, f"{mid}.joblib"))
    meta = {k: v for k, v in model_record.items() if k != "pipeline"}
    pd.Series(meta).to_json(os.path.join(ARTIFACTS_DIR, f"{mid}.json"))

def proba_for_row(pipe, row):
    return float(pipe.predict_proba(row)[0, 1])

def ensure_reasonable_threshold():
    if st.session_state.threshold < 0.1:
        st.warning("Threshold too low, resetting to 0.10")
        st.session_state.threshold = 0.1

st.markdown(
    """
    <style>
    .tiny {font-size:12px;color:#666}
    .metric-card {padding:0.5rem 1rem;border:1px solid #eee;border-radius:12px;background:#fafafa}
    </style>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Predicted Risk", "-" if st.session_state.last_prediction["risk"] is None else f"{st.session_state.last_prediction['risk']*100:.1f}%")
with m2:
    am = get_active_model()
    st.metric("Model", "-" if am is None else am.get("name", "Model"))
with m3:
    st.metric("Data version", st.session_state.data_version)

predict_tab, train_tab, explain_tab, fairness_tab, models_tab, about_tab, batch_tab = st.tabs(
    ["Predict", "Train", "Explain", "Fairness", "Models", "About", "Batch Scoring"]
)

with predict_tab:
    model = get_active_model()
    if model is None:
        st.info("Train a model first in the Train tab.")
    else:
        label_name = model.get("label_name", "target")
        st.caption(f"Active target: {label_name}")
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=0, max_value=120, value=25)
                sleep_h = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            with c2:
                social_h = st.number_input("Social hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
                work_h = st.number_input("Work or Study hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
            with c3:
                gender = st.selectbox("Gender", ["male", "female", "other"])
            adv = st.expander("Advanced")
            with adv:
                thr = st.slider("Decision threshold", 0.0, 1.0, float(st.session_state.threshold), 0.01)
            sub = st.form_submit_button("Predict")
        if sub:
            st.session_state.threshold = thr
            ensure_reasonable_threshold()
            pipe = model["pipeline"]
            row = pd.DataFrame([{"age": age, "gender": gender, "hours_social": social_h, "sleep_hours": sleep_h, "work_hours": work_h}])
            prob1 = proba_for_row(pipe, row)
            st.session_state.last_prediction = {"risk": prob1}
            left, right = st.columns([1, 1])
            if label_name == "target":
                pred = int(prob1 >= st.session_state.threshold)
                st.success(f"Burnout risk: {prob1*100:.1f}% ({'At risk' if pred==1 else 'Not at risk'})")
                with left:
                    st.metric("Burnout risk", f"{prob1*100:.1f}%")
                    st.metric("Label", "At risk" if pred==1 else "Not at risk")
                    st.progress(prob1)
            else:
                pred = int(prob1 >= st.session_state.threshold)
                low_prod_risk = 1.0 - prob1
                st.success(f"Productivity: {prob1*100:.1f}% ({'Good productivity' if pred==1 else 'Low productivity'})")
                with left:
                    st.metric("Good productivity probability", f"{prob1*100:.1f}%")
                    st.metric("Low productivity risk", f"{low_prod_risk*100:.1f}%")
                    st.progress(low_prod_risk)
            with right:
                st.caption("Model card")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Model", model["name"])
                mc2.metric("Data version", model["data_version"])
                mc3.metric("Threshold", f"{st.session_state.threshold:.2f}")
                mc4, mc5, mc6 = st.columns(3)
                mc4.metric("CV Acc", f"{model['cv']['test_accuracy']:.3f}")
                mc5.metric("CV ROC-AUC", f"{model['cv']['test_roc_auc']:.3f}")
                mc6.metric("CV F1", f"{model['cv']['test_f1']:.3f}")
                st.caption(f"Target: {model.get('label_name','target')} • Trained: {model['trained_at']} • Rows: {model['rows']} • Calibration: {model['options'].get('calibration','None')} • Class weight: {model['options']['class_weight']}")

        st.markdown("#### Small changes that flip the decision")
        model = get_active_model()
        if model is not None:
            ensure_reasonable_threshold()
            pipe = model["pipeline"]
            label_name = model.get("label_name", "target")
            current = pd.DataFrame([{
                "age": age if "age" in model["num_cols"] else 0,
                "gender": gender,
                "hours_social": social_h if "hours_social" in model["num_cols"] else 0,
                "sleep_hours": sleep_h if "sleep_hours" in model["num_cols"] else 0,
                "work_hours": work_h if "work_hours" in model["num_cols"] else 0
            }])
            base_p = proba_for_row(pipe, current)
            base_pred = 1 if base_p >= st.session_state.threshold else 0
            if label_name == "target":
                want_pred = 0 if base_pred == 1 else 1
            else:
                want_pred = 1 if base_pred == 0 else 0
            actionable = [c for c in ["sleep_hours","work_hours","hours_social"] if c in model["num_cols"]]
            if len(actionable) == 0:
                st.info("No adjustable features available for this model.")
            else:
                steps = np.arange(0.0, 5.5, 0.5)
                best_flip = None
                best_improve = {"delta": 1.0, "sleep": 0.0, "work": 0.0, "social": 0.0, "new_p": base_p}
                for ds in steps:
                    for dw in steps:
                        for dso in steps:
                            trial = current.copy()
                            if "sleep_hours" in actionable:
                                trial.loc[0, "sleep_hours"] = np.clip(trial.loc[0, "sleep_hours"] + ds, 0, 24)
                            if "work_hours" in actionable:
                                trial.loc[0, "work_hours"] = np.clip(trial.loc[0, "work_hours"] + dw, 0, 24)
                            if "hours_social" in actionable:
                                trial.loc[0, "hours_social"] = np.clip(trial.loc[0, "hours_social"] - dso, 0, 24)
                            p = proba_for_row(pipe, trial)
                            pred = 1 if p >= st.session_state.threshold else 0
                            if pred == want_pred:
                                change = ds + dw + dso
                                if best_flip is None or change < best_flip["change"]:
                                    best_flip = {"sleep": ds, "work": dw, "social": dso, "new_p": p, "change": change}
                            else:
                                if label_name == "target":
                                    delta = p - base_p
                                else:
                                    delta = (1 - p) - (1 - base_p)
                                if delta < best_improve["delta"]:
                                    best_improve = {"delta": delta, "sleep": ds, "work": dw, "social": dso, "new_p": p}
                if best_flip is not None:
                    if label_name == "target":
                        st.success(f"Do this: sleep +{best_flip['sleep']:.1f} h, work +{best_flip['work']:.1f} h, social -{best_flip['social']:.1f} h. New burnout risk: {best_flip['new_p']*100:.1f}%.")
                    else:
                        st.success(f"Do this: sleep +{best_flip['sleep']:.1f} h, work +{best_flip['work']:.1f} h, social -{best_flip['social']:.1f} h. New good productivity: {best_flip['new_p']*100:.1f}%.")
                else:
                    if label_name == "target":
                        st.info(f"No small change flips decision. Closest: sleep +{best_improve['sleep']:.1f} h, work +{best_improve['work']:.1f} h, social -{best_improve['social']:.1f} h. Risk: {best_improve['new_p']*100:.1f}%.")
                    else:
                        st.info(f"No small change flips decision. Closest: sleep +{best_improve['sleep']:.1f} h, work +{best_improve['work']:.1f} h, social -{best_improve['social']:.1f} h. Good productivity: {best_improve['new_p']*100:.1f}%.")
