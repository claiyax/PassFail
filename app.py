from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from features import add_features  # needed so joblib can load the pipeline

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="wide")

HERE = Path(__file__).parent

# column name, label, max value, help text
FIELDS = [
    ("Days.Logged.In", "Days logged in", 100, "Number of different days the student logged in"),
    ("Read.Actions", "Read actions", 1_000_000, "Pages, files and posts viewed"),
    ("Submission.Actions", "Submission actions", 1_000, "Assignments submitted"),
    ("Create.Actions", "Create actions", 1_000, "Items created"),
    ("Quiz.Actions", "Quiz actions", 5_000, "Quiz attempts and interactions"),
    ("Forum.Actions", "Forum actions", 1_000, "Forum posts and replies"),
    ("Update.Actions", "Update actions", 1_000, "Edits to existing items"),
    ("Delete.Actions", "Delete actions", 100, "Items deleted"),
]
# Must match the column order used when the model was trained
COLS = ["Submission.Actions", "Quiz.Actions", "Forum.Actions", "Create.Actions",
        "Read.Actions", "Update.Actions", "Delete.Actions", "Days.Logged.In"]
DEFAULTS = {"Submission.Actions": 4, "Quiz.Actions": 0, "Forum.Actions": 0,
            "Create.Actions": 8, "Read.Actions": 130, "Update.Actions": 3,
            "Delete.Actions": 0, "Days.Logged.In": 14}

# Real example rows from the dataset
PRESETS = {
    "Example: High rating": {"Submission.Actions": 4, "Quiz.Actions": 0, "Forum.Actions": 0,
                             "Create.Actions": 4, "Read.Actions": 154, "Update.Actions": 2,
                             "Delete.Actions": 0, "Days.Logged.In": 23},
    "Example: Low rating": {"Submission.Actions": 8, "Quiz.Actions": 0, "Forum.Actions": 5,
                            "Create.Actions": 10, "Read.Actions": 135, "Update.Actions": 3,
                            "Delete.Actions": 0, "Days.Logged.In": 7},
}


@st.cache_resource
def load_model():
    """Load the saved model; if that fails, train one from the CSV at startup."""
    try:
        return joblib.load(HERE / "model.joblib"), "saved model file"
    except Exception:
        csv = HERE / "augmented_balanced_dataset_m.csv"
        if not csv.exists():
            return None, None
        df = pd.read_csv(csv).drop_duplicates()
        y = (df["rating"] == "High").astype(int)
        pipe = Pipeline([
            ("features", FunctionTransformer(add_features)),
            ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                             random_state=0, n_jobs=-1)),
        ])
        return pipe.fit(df.drop(columns="rating"), y), "trained at startup"


def p_high(model, df):
    return model.predict_proba(df[COLS])[:, 1]


def verdict(p):
    if p >= 0.65:
        return "success", "Likely **High** rating"
    if p <= 0.35:
        return "error", "At risk of a **Low** rating"
    return "warning", "**Borderline**: the model is not confident either way"


def set_preset(values):
    for c in COLS:
        st.session_state[c] = values[c]


def clear_inputs():
    set_preset(DEFAULTS)


model, model_source = load_model()
for c in COLS:
    st.session_state.setdefault(c, DEFAULTS[c])
st.session_state.setdefault("history", [])

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🎓 About")
    st.write("Estimates whether a student's performance rating is likely to be "
             "**High** or **Low** from learning-platform activity.")
    st.info("This is a risk indicator to support advising, not a final judgment "
            "about any student.")
    st.caption("Inputs are not saved to a database. The recent-predictions table "
               "lives only in your browser session.")

st.title("Student Success Predictor")

if model is None:
    st.error("Model not found. Add model.joblib (or augmented_balanced_dataset_m.csv) "
             "to the repository next to app.py.")
    st.stop()

tab_single, tab_batch, tab_about = st.tabs(["Single student", "Batch (CSV)", "About the model"])

# ---------- Single student ----------
with tab_single:
    st.write("Fill in the student's activity or start from an example.")
    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    for col, (name, vals) in zip((b1, b2), PRESETS.items()):
        col.button(name, on_click=set_preset, args=(vals,), use_container_width=True)
    b3.button("Reset", on_click=clear_inputs, use_container_width=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        grid = st.columns(2)
        for i, (c, label, mx, hlp) in enumerate(FIELDS):
            with grid[i % 2]:
                st.number_input(label, min_value=0, max_value=mx, step=1, key=c, help=hlp)
        go = st.button("Predict", type="primary")

    with right:
        if go:
            row = pd.DataFrame([{c: st.session_state[c] for c in COLS}])
            if row["Days.Logged.In"].iloc[0] == 0 and row.drop(columns="Days.Logged.In").iloc[0].sum() > 0:
                st.warning("Activity recorded but 0 days logged in. Check the inputs.")
            p = float(p_high(model, row)[0])
            kind, text = verdict(p)
            getattr(st, kind)(text)
            st.metric("Probability of High rating", f"{p:.0%}")
            st.progress(p)
            st.caption("Below 35% is flagged as at risk, above 65% as likely High, "
                       "and in between as borderline.")
            st.session_state["history"].insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"), **row.iloc[0].to_dict(),
                "P(High)": round(p, 3), "Result": text.replace("**", ""),
            })
        else:
            st.caption("Results appear here after you click Predict.")

    if st.session_state["history"]:
        st.subheader("Recent predictions")
        hist = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist, use_container_width=True, hide_index=True)
        d1, d2 = st.columns([1, 1])
        d1.download_button("Download history (CSV)", hist.to_csv(index=False), "history.csv", "text/csv")
        if d2.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

# ---------- Batch ----------
with tab_batch:
    st.write("Upload a CSV with one student per row to score many students at once.")
    template = pd.DataFrame([DEFAULTS])[COLS]
    st.download_button("Download CSV template", template.to_csv(index=False),
                       "template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception:
            st.error("Could not read that file as a CSV.")
            data = None
        if data is not None:
            missing = [c for c in COLS if c not in data.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            elif data[COLS].isna().any().any() or (data[COLS] < 0).any().any():
                st.error("Columns must be numeric, non-negative and without blanks.")
            else:
                out = data.copy()
                out["P(High)"] = p_high(model, out).round(3)
                out["Result"] = ["Likely High" if p >= 0.65 else "At risk (Low)" if p <= 0.35
                                 else "Borderline" for p in out["P(High)"]]
                m1, m2, m3 = st.columns(3)
                m1.metric("Likely High", int((out["Result"] == "Likely High").sum()))
                m2.metric("Borderline", int((out["Result"] == "Borderline").sum()))
                m3.metric("At risk (Low)", int((out["Result"] == "At risk (Low)").sum()))
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("Download results (CSV)", out.to_csv(index=False),
                                   "predictions.csv", "text/csv")

# ---------- About ----------
with tab_about:
    st.subheader("How reliable is it?")
    a1, a2, a3 = st.columns(3)
    a1.metric("Accuracy (cross-validated)", "~82%")
    a2.metric("High recall", "~81%")
    a3.metric("Low recall", "~81%")
    st.write("Roughly 1 in 5 students is misclassified, so use the result alongside "
             "advisor judgment, never on its own.")
    try:
        names = add_features(pd.DataFrame([DEFAULTS])[COLS]).columns
        imp = pd.Series(model.named_steps["model"].feature_importances_, index=names)
        st.subheader("What the model pays attention to")
        st.bar_chart(imp.sort_values(ascending=True).tail(8), horizontal=True)
        st.caption("Importance shows what the model uses, not what causes success.")
    except Exception:
        pass
    st.subheader("Limitations")
    st.markdown(
        "- Trained on an augmented dataset with a limited set of activity features.\n"
        "- Activity counts do not capture effort, circumstances or learning quality.\n"
        "- The model should be retrained periodically and checked for fairness before "
        "any real deployment."
    )
    st.caption(f"Model source: {model_source}")
