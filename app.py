with predict_tab:
    model = get_active_model()
    if model is None:
        st.info("Train a model first in the Train tab.")
    else:
        label_name = model.get("label_name", "target")  # "target" (burnout) or "productivity"
        st.caption(f"Active target: {label_name}")

        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=0, max_value=120, value=25, help="Years")
                sleep_h = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5, help="Per day")
            with c2:
                social_h = st.number_input("Social hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5, help="Per day")
                work_h = st.number_input("Work/Study hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5, help="Per day")
            with c3:
                gender = st.selectbox("Gender", ["male", "female", "other"], help="Optional category")
            adv = st.expander("Advanced")
            with adv:
                thr = st.slider("Decision threshold", 0.0, 1.0, float(st.session_state.threshold), 0.01)
            sub = st.form_submit_button("Predict")

        if sub:
            pipe = model["pipeline"]
            row = pd.DataFrame([{
                "age": age, "gender": gender,
                "hours_social": social_h, "sleep_hours": sleep_h, "work_hours": work_h
            }])
            prob1 = float(pipe.predict_proba(row)[0, 1])  # probability of class 1
            st.session_state.threshold = thr
            st.session_state.last_prediction = {"risk": prob1}

            if label_name == "target":
                # Burnout: class 1 = at risk
                label = int(prob1 >= thr)
                st.success(f"Burnout risk: {prob1*100:.1f}% ({'At risk' if label==1 else 'Not at risk'})")
                left, right = st.columns([1, 1])
                with left:
                    st.metric("Burnout risk", f"{prob1*100:.1f}%")
                    st.metric("Label", "At risk" if label==1 else "Not at risk")
                    st.progress(prob1)
            else:
                # Productivity: class 1 = good productivity -> risk of low productivity is 1 - prob1
                label = int(prob1 >= thr)
                low_prod_risk = 1.0 - prob1
                st.success(f"Productivity: {prob1*100:.1f}% ({'Good productivity' if label==1 else 'Low productivity'})")
                left, right = st.columns([1, 1])
                with left:
                    st.metric("Good productivity probability", f"{prob1*100:.1f}%")
                    st.metric("Low productivity risk", f"{low_prod_risk*100:.1f}%")
                    st.progress(low_prod_risk)

            # Model card
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
                st.caption(
                    f"Target: {model.get('label_name','target')} • "
                    f"Trained: {model['trained_at']} • Rows: {model['rows']} • "
                    f"Calibration: {model['options'].get('calibration','None')} • "
                    f"Class weight: {model['options']['class_weight']}"
                )
