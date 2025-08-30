import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import io
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
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_recall_curve,
    roc_curve, average_precision_score, confusion_matrix
)
import joblib

st.set_page_config(layout="wide", page_title="Burnout Risk")

if "models" not in st.session_state:
    st.session_state.models = []
if "active_model_id" not in st.session_state:
    st.session_state.active_model_id = None
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = {"risk": None, "conf": None}
if "data_version" not in st.session_state:
    st.session_state.data_version = "—"
if "data_rows" not in st.session_state:
    st.session_state.data_rows = 0
if "last_train" not in st.session_state:
    st.session_state.last_train = None

@st.cache_data(show_spinner=False)
def load_default_csv():
    try:
        df = pd.read_csv("Data/burnout_unified.csv")
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

def prepare(df):
    cols = ["age","gender","hours_social","sleep_hours","work_hours","target"]
    df = df.copy()
    df = df[cols]
    num_cols = ["age","hours_social","sleep_hours","work_hours"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["gender"] = df["gender"].astype(str)
    for c in num_cols:
        df[c] = df[c].clip(lower=0)
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    X = df.drop("target", axis=1)
    y = df["target"].values
    return X, y

def build_pipeline(model_type, class_weight, calibrate):
    num_cols = ["age","hours_social","sleep_hours","work_hours"]
    cat_cols = ["gender"]
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
    if calibrate:
        clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    else:
        clf = base
    pipe = Pipeline([
        ("prep", pre),
        ("clf", clf)
    ])
    return pipe

def cv_scores(pipe, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy":"accuracy","roc_auc":"roc_auc","f1":"f1"}
    out = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=None)
    return {k: float(np.mean(v)) for k,v in out.items() if k.startswith("test_")}

def get_active_model():
    if st.session_state.active_model_id is None:
        return None
    for m in st.session_state.models:
        if m["id"] == st.session_state.active_model_id:
            return m
    return None

def set_active_model(mid):
    st.session_state.active_model_id = mid

st.markdown(
    """
    <style>
    .tiny {font-size:12px;color:#666}
    .metric-card {padding:0.5rem 1rem;border:1px solid #eee;border-radius:12px;background:#fafafa}
    </style>
    """,
    unsafe_allow_html=True,
)

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Predicted Risk", "—" if st.session_state.last_prediction["risk"] is None else f"{st.session_state.last_prediction['risk']*100:.1f}%")
with colB:
    st.metric("Confidence", "—" if st.session_state.last_prediction["conf"] is None else f"{st.session_state.last_prediction['conf']*100:.1f}%")
with colC:
    am = get_active_model()
    st.metric("Model", "—" if am is None else am.get("name","Model"))
with colD:
    st.metric("Data version", st.session_state.data_version)

predict_tab, train_tab, explain_tab, models_tab, about_tab = st.tabs(["Predict","Train","Explain","Models","About"])

with predict_tab:
    src = st.radio("Data source", ["Default","Upload"], horizontal=True)
    file = None
    if src == "Upload":
        file = st.file_uploader("CSV", type=["csv"])
    df, dv = load_data(file)
    if df is not None:
        st.session_state.data_version = dv
        st.session_state.data_rows = len(df)
        X, y = prepare(df)
        st.caption(f"Rows: {len(df)} | Features: {X.shape[1]}")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=25, help="Years")
            sleep_h = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5, help="Per day")
        with c2:
            social_h = st.number_input("Social hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5, help="Per day")
            work_h = st.number_input("Work/Study hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5, help="Per day")
        with c3:
            gender = st.selectbox("Gender", ["male","female","other"], help="Optional category")
        adv = st.expander("Advanced")
        with adv:
            thr = st.slider("Decision threshold", 0.0, 1.0, float(st.session_state.threshold), 0.01)
        sub = st.form_submit_button("Predict")
    if sub:
        model = get_active_model()
        if model is None and len(st.session_state.models) > 0:
            model = st.session_state.models[-1]
            set_active_model(model["id"])
        if model is None:
            st.warning("Train a model first in the Train tab.")
        else:
            pipe = model["pipeline"]
            row = pd.DataFrame([{"age":age,"gender":gender,"hours_social":social_h,"sleep_hours":sleep_h,"work_hours":work_h}])
            prob = float(pipe.predict_proba(row)[0,1])
            label = int(prob >= thr)
            conf = prob if label==1 else 1-prob
            st.session_state.threshold = thr
            st.session_state.last_prediction = {"risk": prob, "conf": conf}
            st.subheader(f"Risk: {prob*100:.1f}% | Label: {'At-risk' if label==1 else 'Not at-risk'}")
            if model:
                with st.container():
                    st.markdown("**Model card**")
                    meta = {
                        "trained": model["trained_at"],
                        "Data version": model["data_version"],
                        "rows": model["rows"],
                        "features": model["features"],
                        "cv_accuracy": round(model["cv"]["test_accuracy"],3),
                        "cv_roc_auc": round(model["cv"]["test_roc_auc"],3),
                        "cv_f1": round(model["cv"]["test_f1"],3),
                        "threshold": round(st.session_state.threshold,2),
                        "calibrated": model["options"]["calibrate"],
                        "class_weight": model["options"]["class_weight"],
                        "fairness_note": "Gender optional; review bias before deployment."
                    }
                    st.json(meta)

with train_tab:
    src = st.radio("Training data", ["Default","Upload"], horizontal=True)
    file = None
    if src == "Upload":
        file = st.file_uploader("CSV for training", type=["csv"], key="train_csv")
    df, dv = load_data(file)
    if df is None:
        st.info("Provide a CSV or keep Default once available.")
    else:
        X, y = prepare(df)
        st.caption(f"Rows: {len(df)} | Features: {X.shape[1]}")
        c0, c1, c2, c3 = st.columns(4)
        with c0:
            model_type = st.selectbox("Model", ["Logistic Regression","Random Forest"])
        with c1:
            test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
        with c2:
            random_state = st.number_input("Random state", 0, 9999, 42)
        with c3:
            class_weight_flag = st.checkbox("Use class_weight='balanced'", value=False)
        calibrate_flag = st.checkbox("Calibrate probabilities", value=False)
        go = st.button("Train model")
        if go:
            cw = True if class_weight_flag else False
            pipe = build_pipeline(model_type, cw, calibrate_flag)
            scores = cv_scores(pipe, X, y)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y)
            pipe.fit(Xtr, ytr)
            proba = pipe.predict_proba(Xte)[:,1]
            preds = (proba >= st.session_state.threshold).astype(int)
            acc = accuracy_score(yte, preds)
            roc = roc_auc_score(yte, proba)
            f1 = f1_score(yte, preds)
            fpr, tpr, _ = roc_curve(yte, proba)
            p, r, _ = precision_recall_curve(yte, proba)
            ap = average_precision_score(yte, proba)
            cm = confusion_matrix(yte, preds)
            frac_pos, mean_pred = calibration_curve(yte, proba, n_bins=10)
            mid = f"M{len(st.session_state.models)+1}"
            model_record = {
                "id": mid,
                "name": f"Model {len(st.session_state.models)+1}",
                "pipeline": pipe,
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "data_version": dv,
                "rows": len(df),
                "features": list(X.columns),
                "cv": scores,
                "options": {"model_type": model_type, "calibrate": calibrate_flag, "class_weight": ("balanced" if cw else None)},
            }
            st.session_state.models.append(model_record)
            set_active_model(mid)
            st.session_state.last_train = {
                "acc": float(acc), "roc": float(roc), "f1": float(f1),
                "fpr": fpr, "tpr": tpr, "precision": p, "recall": r, "ap": float(ap),
                "cm": cm, "mean_pred": mean_pred, "frac_pos": frac_pos
            }
        res = st.session_state.last_train
        if res is not None:
            st.success(f"Held-out: acc {res['acc']:.3f} | roc_auc {res['roc']:.3f} | f1 {res['f1']:.3f}")
            fig1, ax1 = plt.subplots()
            ax1.plot(res['fpr'], res['tpr'])
            ax1.plot([0,1],[0,1], linestyle='--')
            ax1.set_xlabel('FPR')
            ax1.set_ylabel('TPR')
            ax1.set_title('ROC curve')
            st.pyplot(fig1)
            fig2, ax2 = plt.subplots()
            ax2.plot(res['recall'], res['precision'])
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title(f"PR curve (AP={res['ap']:.3f})")
            st.pyplot(fig2)
            fig3, ax3 = plt.subplots()
            im = ax3.imshow(res['cm'])
            ax3.set_title('Confusion matrix')
            ax3.set_xticks([0,1])
            ax3.set_yticks([0,1])
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('True')
            for (i,j), v in np.ndenumerate(res['cm']):
                ax3.text(j, i, str(v), ha='center', va='center')
            st.pyplot(fig3)
            fig4, ax4 = plt.subplots()
            ax4.plot(res['mean_pred'], res['frac_pos'], marker='o')
            ax4.plot([0,1],[0,1], linestyle='--')
            ax4.set_xlabel('Mean predicted prob')
            ax4.set_ylabel('Fraction positive')
            ax4.set_title('Calibration curve')
            st.pyplot(fig4)

with explain_tab:
    model = get_active_model()
    if model is None:
        st.info("Train or select a model first.")
    else:
        pipe = model["pipeline"]
        clf_step = pipe.named_steps["clf"]
        base = getattr(clf_step, "base_estimator", None) or getattr(clf_step, "estimator", None) or clf_step
        num_cols = ["age","hours_social","sleep_hours","work_hours"]
        cat_cols = ["gender"]
        cat_pipe = pipe.named_steps["prep"].named_transformers_["cat"]
        ohe = cat_pipe.named_steps["ohe"]
        cat_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names = num_cols + cat_names
        if hasattr(base, "coef_"):
            coefs = base.coef_.ravel()
            w = pd.DataFrame({"feature": feature_names, "weight": coefs}).sort_values("weight", ascending=False)
            st.dataframe(w, use_container_width=True)
        elif hasattr(base, "feature_importances_"):
            imps = base.feature_importances_
            w = pd.DataFrame({"feature": feature_names, "importance": imps}).sort_values("importance", ascending=False)
            st.dataframe(w, use_container_width=True)
        else:
            st.info("No explainability attributes available for this model.")

with models_tab:
    if len(st.session_state.models) == 0:
        st.info("No models yet.")
    else:
        names = [f"{m['id']} | {m['name']} | {m['data_version']}" for m in st.session_state.models]
        idx = 0
        if st.session_state.active_model_id is not None:
            for i,m in enumerate(st.session_state.models):
                if m["id"] == st.session_state.active_model_id:
                    idx = i
                    break
        choice = st.radio("Select active model", names, index=idx)
        picked = st.session_state.models[names.index(choice)]
        set_active_model(picked["id"])
        c1, c2 = st.columns(2)
        with c1:
            bio = io.BytesIO()
            joblib.dump(picked, bio)
            bio.seek(0)
            st.download_button("Download model", data=bio, file_name=f"{picked['id']}.joblib")
        with c2:
            up = st.file_uploader("Upload model file", type=["joblib"], key="upload_model")
            if up is not None:
                try:
                    obj = joblib.load(up)
                    if isinstance(obj, dict) and "pipeline" in obj:
                        st.session_state.models.append(obj)
                        set_active_model(obj["id"])
                        st.success("Model added.")
                        
                    else:
                        st.error("Invalid model file.")
                except Exception as e:
                    st.error("Could not load model.")
        with st.expander("Active model card"):
            meta = {
                "trained": picked["trained_at"],
                "Data version": picked["data_version"],
                "rows": picked["rows"],
                "features": picked["features"],
                "cv_accuracy": round(picked["cv"]["test_accuracy"],3),
                "cv_roc_auc": round(picked["cv"]["test_roc_auc"],3),
                "cv_f1": round(picked["cv"]["test_f1"],3),
                "threshold": round(st.session_state.threshold,2),
                "options": picked["options"],
                "fairness_note": "Gender optional; review bias before deployment."
            }
            st.json(meta)

with about_tab:
    st.markdown("### About")
    st.markdown("This app predicts burnout risk from age, gender, social-media hours, sleep hours, and work/study hours.")
    st.markdown("Data sources: Sleep Health & Lifestyle; Social Media & Mental Health. Target labels follow stress/wellbeing heuristics used in the unified dataset.")
    st.markdown("Use ethically. Do not use for medical diagnosis.")
