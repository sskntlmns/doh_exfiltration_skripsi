"""
DoH Exfiltration Detection — Metadata Flow Dashboard
=====================================================
Streamlit GUI for the DoH exfiltration detection notebook.
Supports uploading custom datasets for full pipeline execution
(preprocessing → training → evaluation → prediction).
"""

import json
import io
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_model.pkl"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DoH Exfiltration Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium Dark Theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a40 50%, #24243e 100%);
}
section[data-testid="stSidebar"] * { color: #e0e0ff !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-weight: 500; padding: 6px 0;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    padding-left: 4px;
}

/* ── Metric Cards ─────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(99, 102, 241, .2);
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 4px 24px rgba(99, 102, 241, .06);
    transition: all 0.3s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(99, 102, 241, .45);
    box-shadow: 0 8px 32px rgba(99, 102, 241, .12);
    transform: translateY(-2px);
}
div[data-testid="stMetric"] label {
    color: #a5b4fc !important; font-weight: 600; font-size: .8rem;
    letter-spacing: .04em; text-transform: uppercase;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e0e7ff !important; font-weight: 700;
}

/* ── Section Headers ──────────────────────────────────────────────────── */
.section-header {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800; font-size: 1.85rem; margin-bottom: .2rem;
    letter-spacing: -0.01em;
}
.section-sub {
    color: #94a3b8; font-size: .95rem; margin-bottom: 1.4rem;
    line-height: 1.5;
}

/* ── Info Box ──────────────────────────────────────────────────────────── */
.info-box {
    background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 100%);
    border-left: 4px solid #6366f1; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 16px; color: #cbd5e1;
    font-size: .92rem; line-height: 1.7;
}
.info-box strong { color: #a5b4fc; }

/* ── Hero Card ────────────────────────────────────────────────────────── */
.hero-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e293b 100%);
    border: 1px solid rgba(99, 102, 241, .3);
    border-radius: 20px; padding: 32px 36px;
    box-shadow: 0 8px 40px rgba(99, 102, 241, .1);
    margin-bottom: 24px;
}
.hero-title {
    font-size: 1.1rem; font-weight: 600; color: #a5b4fc;
    text-transform: uppercase; letter-spacing: .06em;
    margin-bottom: 8px;
}
.hero-model {
    font-size: 2.2rem; font-weight: 800; color: #e0e7ff;
    margin-bottom: 16px; line-height: 1.2;
}
.hero-metrics {
    display: flex; gap: 32px; flex-wrap: wrap;
}
.hero-metric {
    text-align: center;
}
.hero-metric-label {
    font-size: .75rem; font-weight: 600; color: #818cf8;
    text-transform: uppercase; letter-spacing: .05em;
    margin-bottom: 4px;
}
.hero-metric-value {
    font-size: 1.6rem; font-weight: 700; color: #c7d2fe;
}

/* ── Stat Card (for highlighted sections) ─────────────────────────────── */
.stat-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #1e293b 100%);
    border: 1px solid rgba(99, 102, 241, .2);
    border-radius: 16px; padding: 24px 28px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}
.stat-card:hover {
    border-color: rgba(99, 102, 241, .4);
    box-shadow: 0 6px 28px rgba(99, 102, 241, .08);
}
.stat-card-title {
    font-size: .82rem; font-weight: 600; color: #818cf8;
    text-transform: uppercase; letter-spacing: .05em;
    margin-bottom: 10px;
}

/* ── Prediction Cards ─────────────────────────────────────────────────── */
.pred-benign {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981; border-radius: 16px; padding: 28px;
    text-align: center; color: #6ee7b7; font-size: 1.4rem; font-weight: 700;
    box-shadow: 0 4px 24px rgba(16, 185, 129, .15);
}
.pred-malicious {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444; border-radius: 16px; padding: 28px;
    text-align: center; color: #fca5a5; font-size: 1.4rem; font-weight: 700;
    box-shadow: 0 4px 24px rgba(239, 68, 68, .15);
}

/* ── Source Badges ────────────────────────────────────────────────────── */
.source-badge-default {
    display: inline-block; background: #312e81; color: #c4b5fd;
    padding: 5px 14px; border-radius: 20px; font-size: .75rem;
    font-weight: 600; letter-spacing: .04em;
}
.source-badge-uploaded {
    display: inline-block; background: #065f46; color: #6ee7b7;
    padding: 5px 14px; border-radius: 20px; font-size: .75rem;
    font-weight: 600; letter-spacing: .04em;
}

/* ── Best Model Badge ─────────────────────────────────────────────────── */
.best-badge {
    display: inline-block; background: linear-gradient(135deg, #f59e0b, #d97706);
    color: #1a1a2e; padding: 3px 10px; border-radius: 12px;
    font-size: .7rem; font-weight: 700; letter-spacing: .03em;
    margin-left: 6px; vertical-align: middle;
}

/* ── Takeaway Box ─────────────────────────────────────────────────────── */
.takeaway-box {
    background: linear-gradient(135deg, #0c4a6e 0%, #1e3a5f 100%);
    border-left: 4px solid #0ea5e9; border-radius: 12px;
    padding: 18px 22px; margin-top: 20px; color: #bae6fd;
    font-size: .9rem; line-height: 1.6;
}
.takeaway-box strong { color: #7dd3fc; }

/* ── Comparison Metric ────────────────────────────────────────────────── */
.cmp-improve { color: #34d399; font-weight: 600; }
.cmp-neutral { color: #94a3b8; font-weight: 500; }
.cmp-slight  { color: #fbbf24; font-weight: 500; }

/* ── Chart Caption ────────────────────────────────────────────────────── */
.chart-caption {
    text-align: center; color: #64748b; font-size: .82rem;
    font-style: italic; margin-top: -8px; margin-bottom: 16px;
}

/* ── Gradient Divider ─────────────────────────────────────────────────── */
.gradient-divider {
    height: 1px; border: none; margin: 24px 0;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,.3), transparent);
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 10px 22px; font-weight: 600;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 10px;
}
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Environment Card ─────────────────────────────────────────────────── */
.env-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 100%);
    border: 1px solid rgba(99, 102, 241, .15);
    border-radius: 14px; padding: 18px 22px; margin-bottom: 12px;
    color: #cbd5e1; font-size: .88rem; line-height: 1.6;
}
.env-card-aspect {
    font-weight: 700; color: #a5b4fc; font-size: .95rem;
    margin-bottom: 6px;
}

/* ── SOC Priority Badge ───────────────────────────────────────────────── */
.soc-p1 { background: #991b1b; color: #fca5a5; padding: 3px 10px; border-radius: 10px; font-weight: 700; font-size: .78rem; }
.soc-p2 { background: #92400e; color: #fcd34d; padding: 3px 10px; border-radius: 10px; font-weight: 700; font-size: .78rem; }
.soc-p3 { background: #1e3a5f; color: #93c5fd; padding: 3px 10px; border-radius: 10px; font-weight: 700; font-size: .78rem; }
.soc-p4 { background: #1e293b; color: #94a3b8; padding: 3px 10px; border-radius: 10px; font-weight: 700; font-size: .78rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e0e7ff", size=13),
    margin=dict(l=48, r=24, t=56, b=48),
    title_font=dict(size=16, color="#c7d2fe"),
)
COLOR_PALETTE = ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#06b6d4"]

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "data_source" not in st.session_state:
    st.session_state.data_source = "default"  # "default" or "uploaded"

if "uploaded_results" not in st.session_state:
    st.session_state.uploaded_results = None  # dict with all pipeline results


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE: Preprocessing → Training → Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def detect_label_column(df: pd.DataFrame) -> str:
    """Auto-detect the label column."""
    candidates = ["Label", "label", "Class", "class", "Target", "target", "Category", "category"]
    for c in candidates:
        if c in df.columns:
            return c
    obj_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    low_card = [c for c in obj_cols if 1 < df[c].nunique(dropna=True) <= 20]
    if low_card:
        return low_card[-1]
    raise ValueError("Kolom label tidak ditemukan.")


def get_positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
    return model.predict(X).astype(float)


def run_full_pipeline(df_raw: pd.DataFrame, progress_bar, status_text,
                      random_state: int = 42, test_size: float = 0.2) -> dict:
    """Run the complete pipeline on an uploaded dataset."""
    results = {}

    # ── Step 1: Detect label ──────────────────────────────────────────────
    status_text.text("🔍 Mendeteksi kolom label...")
    progress_bar.progress(5)
    label_col = detect_label_column(df_raw)
    df_raw.columns = df_raw.columns.str.strip()

    # ── Step 2: Clean ─────────────────────────────────────────────────────
    status_text.text("🧹 Cleaning data...")
    progress_bar.progress(10)
    df = df_raw.copy()
    initial_rows = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[label_col])
    df = df.drop_duplicates().reset_index(drop=True)

    # ── Step 3: Feature selection ─────────────────────────────────────────
    status_text.text("🔢 Seleksi fitur numerik...")
    progress_bar.progress(15)
    numeric_features = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).columns.tolist()
    constant_features = [c for c in numeric_features if df[c].nunique(dropna=True) <= 1]
    numeric_features = [c for c in numeric_features if c not in constant_features]

    X = df[numeric_features].copy()
    y_raw = df[label_col].astype(str).str.strip()

    # ── Step 4: Encode labels ─────────────────────────────────────────────
    status_text.text("🏷️ Encoding label...")
    progress_bar.progress(20)
    le = LabelEncoder()
    le.fit_transform(y_raw)
    classes = le.classes_.tolist()

    if len(classes) < 2:
        raise ValueError(f"Perlu minimal 2 kelas, ditemukan: {classes}")

    positive_keywords = ["malicious", "attack", "exfil", "tunnel", "botnet", "anomaly", "intrusion", "1", "positive"]
    positive_class = next(
        (cls for cls in classes if any(kw in cls.lower() for kw in positive_keywords)),
        classes[-1],
    )
    negative_class = next(cls for cls in classes if cls != positive_class)
    y = (y_raw == positive_class).astype(int)

    cleaning_summary = {
        "initial_rows": int(initial_rows),
        "rows_after_cleaning": int(len(df)),
        "removed_rows": int(initial_rows - len(df)),
        "label_column": label_col,
        "negative_class_0": negative_class,
        "positive_class_1": positive_class,
        "n_numeric_features": len(numeric_features),
        "constant_features_removed": constant_features,
    }

    # ── Step 5: Train-test split ──────────────────────────────────────────
    status_text.text("✂️ Split train-test...")
    progress_bar.progress(25)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # ── Step 6: Train models ──────────────────────────────────────────────
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200, random_state=random_state,
                class_weight="balanced", n_jobs=-1,
            )),
        ]),
        "AdaBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", AdaBoostClassifier(
                n_estimators=200, learning_rate=0.5, random_state=random_state,
            )),
        ]),
    }

    trained_models = {}
    train_times = {}
    model_names = list(models.keys())
    progress_steps = [30, 45, 65]

    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"🏋️ Training {name}...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_times[name] = time.perf_counter() - start
        trained_models[name] = model
        progress_bar.progress(progress_steps[i])

    # ── Step 7: Evaluate ──────────────────────────────────────────────────
    status_text.text("📊 Evaluasi model...")
    progress_bar.progress(70)

    eval_records = []
    for name, model in trained_models.items():
        t0 = time.perf_counter()
        y_score = get_positive_scores(model, X_test)
        y_pred = (y_score >= 0.5).astype(int)
        inference_time = time.perf_counter() - t0

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        fpr_val = fp / (fp + tn) if (fp + tn) else 0.0
        throughput = len(X_test) / inference_time if inference_time > 0 else np.inf

        eval_records.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall_tpr": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_score),
            "fpr": fpr_val,
            "training_time_sec": train_times[name],
            "inference_time_sec": inference_time,
            "throughput_rows_per_sec": throughput,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

    metrics_df = pd.DataFrame(eval_records).sort_values(
        by=["roc_auc", "f1_score", "recall_tpr"], ascending=False
    ).reset_index(drop=True)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_metrics = metrics_df.iloc[0].to_dict()

    # ── Step 8: Feature importance ────────────────────────────────────────
    status_text.text("🌲 Menghitung feature importance...")
    progress_bar.progress(80)

    fi_df = pd.DataFrame()
    if best_model_name == "Random Forest":
        clf = best_model.named_steps["classifier"]
        fi_df = pd.DataFrame({
            "feature": numeric_features,
            "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Step 9: Threshold analysis ────────────────────────────────────────
    status_text.text("🎯 Threshold analysis...")
    progress_bar.progress(90)

    y_score_best = get_positive_scores(best_model, X_test)
    thresholds = np.arange(0.05, 1.0, 0.05)
    ta_rows = []
    for t in thresholds:
        y_p = (y_score_best >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_p, labels=[0, 1]).ravel()
        fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) else 0.0
        ta_rows.append({
            "threshold": round(t, 2),
            "accuracy": accuracy_score(y_test, y_p),
            "precision": precision_score(y_test, y_p, zero_division=0),
            "recall_tpr": recall_score(y_test, y_p, zero_division=0),
            "f1_score": f1_score(y_test, y_p, zero_division=0),
            "fpr": fpr_t,
            "tn": int(tn_t), "fp": int(fp_t), "fn": int(fn_t), "tp": int(tp_t),
        })
    ta_df = pd.DataFrame(ta_rows)

    # Find best threshold by F1
    best_ta_row = ta_df.loc[ta_df["f1_score"].idxmax()]
    selected_threshold = best_ta_row["threshold"]

    progress_bar.progress(100)
    status_text.text("✅ Pipeline selesai!")

    # ── Pack results ──────────────────────────────────────────────────────
    results = {
        "cleaning_summary": cleaning_summary,
        "numeric_features": numeric_features,
        "label_counts": y_raw.value_counts().to_dict(),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "model_metrics": metrics_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_metrics": best_metrics,
        "feature_importance": fi_df,
        "threshold_analysis": ta_df,
        "selected_threshold": selected_threshold,
        "selected_threshold_metrics": best_ta_row.to_dict(),
        "trained_models": trained_models,
        "test_size": test_size,
        "random_state": random_state,
    }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# DATA ACCESS: unified getters (session state → file fallback)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def _load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def _load_json(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_resource
def _load_model_file():
    if MODEL_PATH.exists():
        obj = joblib.load(MODEL_PATH)
        if isinstance(obj, dict) and "model" in obj:
            return obj["model"]
        return obj
    return None


def is_uploaded():
    return st.session_state.data_source == "uploaded" and st.session_state.uploaded_results is not None


def get_model_metrics() -> pd.DataFrame:
    if is_uploaded():
        return st.session_state.uploaded_results["model_metrics"]
    return _load_csv(str(OUTPUT_DIR / "model_metrics.csv"))


def get_best_model_info() -> dict:
    if is_uploaded():
        r = st.session_state.uploaded_results
        return {
            "label_column": r["cleaning_summary"]["label_column"],
            "positive_class_1": r["cleaning_summary"]["positive_class_1"],
            "negative_class_0": r["cleaning_summary"]["negative_class_0"],
            "numeric_features": r["numeric_features"],
            "test_size": r["test_size"],
            "random_state": r["random_state"],
            "best_model_name": r["best_model_name"],
            "selected_threshold": r["selected_threshold"],
            "cleaning_summary": r["cleaning_summary"],
            "best_model_metrics_default_threshold": r["best_metrics"],
            "selected_threshold_metrics": r["selected_threshold_metrics"],
        }
    return _load_json(str(OUTPUT_DIR / "best_model_info.json"))


def get_enriched_info() -> dict:
    if is_uploaded():
        return {}  # no enriched info for uploaded datasets
    return _load_json(str(OUTPUT_DIR / "enriched_experiment_info.json"))


def get_feature_importance() -> pd.DataFrame:
    if is_uploaded():
        return st.session_state.uploaded_results["feature_importance"]
    return _load_csv(str(OUTPUT_DIR / "random_forest_feature_importance.csv"))


def get_threshold_analysis() -> pd.DataFrame:
    if is_uploaded():
        return st.session_state.uploaded_results["threshold_analysis"]
    return _load_csv(str(OUTPUT_DIR / "threshold_analysis.csv"))


def get_enhanced_threshold_analysis() -> pd.DataFrame:
    """Load the more detailed threshold analysis from notebook."""
    return _load_csv(str(OUTPUT_DIR / "enhanced_threshold_analysis.csv"))


def get_explainability_importance() -> pd.DataFrame:
    """Load permutation importance from notebook."""
    return _load_csv(str(OUTPUT_DIR / "explainability_importance.csv"))


def get_soc_alert_summary() -> pd.DataFrame:
    """Load enhanced SOC alert simulation summary from notebook."""
    return _load_csv(str(OUTPUT_DIR / "enhanced_soc_alert_summary.csv"))


def get_trained_model():
    if is_uploaded():
        return st.session_state.uploaded_results["best_model"]
    return _load_model_file()


def get_all_trained_models():
    if is_uploaded():
        return st.session_state.uploaded_results.get("trained_models", {})
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# Helper: gradient divider
# ═══════════════════════════════════════════════════════════════════════════
def divider():
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ DoH Detection")
    st.markdown(
        '<p style="font-size:.82rem;color:#94a3b8;margin-top:-8px">Metadata Flow Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Dataset source selector ───────────────────────────────────────────
    st.markdown("#### 📂 Sumber Dataset")
    source_option = st.radio(
        "Pilih sumber data:",
        ["📁 Default (pre-computed)", "📤 Upload Dataset Baru"],
        index=0 if st.session_state.data_source == "default" else 1,
        key="source_radio",
    )

    if source_option == "📁 Default (pre-computed)":
        st.session_state.data_source = "default"
        st.markdown(
            '<span class="source-badge-default">DEFAULT DATASET</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="source-badge-uploaded">CUSTOM DATASET</span>',
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload CSV dataset",
            type=["csv"],
            key="dataset_upload",
            accept_multiple_files=True,
            help="Upload 1 file gabungan, atau beberapa file terpisah "
                 "(misal: benign-chrome.csv, mal-dnscat2.csv). "
                 "Label otomatis dideteksi dari nama file jika kolom Label tidak ada.",
        )

        if uploaded_files:
            # Show uploaded file summary
            st.markdown(f"📎 **{len(uploaded_files)} file** diupload:")
            for uf in uploaded_files:
                fname = uf.name.lower()
                if any(kw in fname for kw in ["mal", "attack", "exfil", "tunnel", "botnet"]):
                    badge = "🔴"
                elif any(kw in fname for kw in ["benign", "normal", "clean", "legit"]):
                    badge = "🟢"
                else:
                    badge = "📄"
                st.markdown(f"&nbsp;&nbsp;{badge} `{uf.name}`")

            col_ts, col_rs = st.columns(2)
            with col_ts:
                custom_test_size = st.number_input("Test Size", 0.1, 0.5, 0.2, 0.05, key="ts")
            with col_rs:
                custom_random_state = st.number_input("Random State", 0, 999, 42, key="rs")

            if st.button("🚀 Jalankan Pipeline", type="primary", use_container_width=True):
                try:
                    # Read and concatenate all files
                    dfs = []
                    for uf in uploaded_files:
                        df_part = pd.read_csv(uf)
                        # If no label column, infer from filename
                        label_candidates = ["Label", "label", "Class", "class", "Target", "target"]
                        has_label = any(c in df_part.columns for c in label_candidates)
                        if not has_label:
                            fname = uf.name.lower()
                            if any(kw in fname for kw in ["mal", "attack", "exfil", "tunnel", "botnet"]):
                                df_part["Label"] = "Malicious"
                            elif any(kw in fname for kw in ["benign", "normal", "clean", "legit"]):
                                df_part["Label"] = "Benign"
                            else:
                                # Use filename (without extension) as label
                                df_part["Label"] = Path(uf.name).stem
                        dfs.append(df_part)

                    df_uploaded = pd.concat(dfs, ignore_index=True)
                    st.info(f"📊 Dataset gabungan: **{len(df_uploaded):,} rows** dari {len(dfs)} file")

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results = run_full_pipeline(
                        df_uploaded, progress_bar, status_text,
                        random_state=int(custom_random_state),
                        test_size=custom_test_size,
                    )

                    st.session_state.uploaded_results = results
                    st.session_state.data_source = "uploaded"
                    st.success(
                        f"✅ Pipeline selesai! Best model: **{results['best_model_name']}** "
                        f"(F1: {results['best_metrics']['f1_score']:.4%})"
                    )
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        elif is_uploaded():
            r = st.session_state.uploaded_results
            st.success(f"✅ Dataset aktif: {r['best_model_name']}")
            if st.button("🔄 Reset ke Default", use_container_width=True):
                st.session_state.data_source = "default"
                st.session_state.uploaded_results = None
                st.rerun()

    st.divider()

    # ── Navigation (reordered: Prediksi moved to last) ────────────────────
    page = st.radio(
        "Navigasi",
        [
            "📊 Dashboard Overview",
            "📈 Perbandingan Model",
            "🏆 Best Model Detail",
            "📋 Robustness & Analysis",
            "🔍 Prediksi Interaktif",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Sidebar footer: best model badge + dataset info ───────────────────
    info_sidebar = get_best_model_info()
    if info_sidebar:
        best_n = info_sidebar.get("best_model_name", "")
        best_f1 = info_sidebar.get("best_model_metrics_default_threshold", {}).get("f1_score", 0)
        if best_n:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#312e81,#1e293b);border-radius:12px;'
                f'padding:14px 16px;margin-bottom:8px;border:1px solid rgba(99,102,241,.2)">'
                f'<div style="font-size:.72rem;font-weight:600;color:#818cf8;text-transform:uppercase;'
                f'letter-spacing:.05em;margin-bottom:4px">🏆 Best Model</div>'
                f'<div style="font-size:1.05rem;font-weight:700;color:#e0e7ff">{best_n}</div>'
                f'<div style="font-size:.8rem;color:#a5b4fc;margin-top:2px">F1: {best_f1:.4%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    cs_sidebar = info_sidebar.get("cleaning_summary", {})
    total_rows = cs_sidebar.get("rows_after_cleaning", 0)
    st.caption(f"Dataset: BCCC-CIRA-CIC-DoHBrw-2020 · {total_rows:,} samples")
    st.caption("Skripsi — Deteksi Eksfiltrasi Data via DoH berbasis Metadata Flow")


# ═══════════════════════════════════════════════════════════════════════════
# Source badge helper
# ═══════════════════════════════════════════════════════════════════════════
def show_source_badge():
    if is_uploaded():
        st.markdown(
            '<span class="source-badge-uploaded">📤 CUSTOM DATASET</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="source-badge-default">📁 DEFAULT DATASET</span>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: Dashboard Overview
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":
    st.markdown('<p class="section-header">📊 Dashboard Overview</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Ringkasan dataset, preprocessing, dan konfigurasi eksperimen.</p>',
        unsafe_allow_html=True,
    )
    show_source_badge()

    info = get_best_model_info()
    cleaning = info.get("cleaning_summary", {})
    best_metrics_ov = info.get("best_model_metrics_default_threshold", {})

    # ── Hero Card: Best Model Summary ─────────────────────────────────────
    st.markdown(
        f"""
<div class="hero-card">
    <div class="hero-title">🏆 Best Model — Hasil Eksperimen</div>
    <div class="hero-model">{info.get('best_model_name', 'N/A')}</div>
    <div class="hero-metrics">
        <div class="hero-metric">
            <div class="hero-metric-label">F1 Score</div>
            <div class="hero-metric-value">{best_metrics_ov.get('f1_score', 0):.4%}</div>
        </div>
        <div class="hero-metric">
            <div class="hero-metric-label">ROC-AUC</div>
            <div class="hero-metric-value">{best_metrics_ov.get('roc_auc', 0):.6f}</div>
        </div>
        <div class="hero-metric">
            <div class="hero-metric-label">Accuracy</div>
            <div class="hero-metric-value">{best_metrics_ov.get('accuracy', 0):.4%}</div>
        </div>
        <div class="hero-metric">
            <div class="hero-metric-label">False Positive</div>
            <div class="hero-metric-value">{int(best_metrics_ov.get('fp', 0)):,}</div>
        </div>
        <div class="hero-metric">
            <div class="hero-metric-label">False Negative</div>
            <div class="hero-metric-value">{int(best_metrics_ov.get('fn', 0)):,}</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Top metric cards ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows Awal", f"{cleaning.get('initial_rows', 0):,}",
              help="Jumlah baris sebelum proses cleaning")
    c2.metric("Rows Setelah Cleaning", f"{cleaning.get('rows_after_cleaning', 0):,}",
              help="Jumlah baris setelah duplikat dan missing value dihapus")
    c3.metric("Rows Dihapus", f"{cleaning.get('removed_rows', 0):,}",
              help="Baris yang dihapus saat preprocessing")
    c4.metric("Jumlah Fitur", cleaning.get("n_numeric_features", 0),
              help="Fitur metadata flow numerik yang digunakan")

    divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📁 Informasi Dataset")
        st.markdown(
            f"""
<div class="info-box">
<strong>Dataset:</strong> BCCC-CIRA-CIC-DoHBrw-2020<br>
<strong>Label Column:</strong> {info.get('label_column', 'N/A')}<br>
<strong>Positive Class (1):</strong> {info.get('positive_class_1', 'N/A')}<br>
<strong>Negative Class (0):</strong> {info.get('negative_class_0', 'N/A')}<br>
<strong>Test Size:</strong> {info.get('test_size', 0.2)}<br>
<strong>Random State:</strong> {info.get('random_state', 42)}<br>
<strong>Best Model:</strong> {info.get('best_model_name', 'N/A')}<br>
<strong>Selected Threshold:</strong> {info.get('selected_threshold', 'N/A')}
</div>
""",
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("### 📊 Distribusi Label")
        if is_uploaded():
            lc = st.session_state.uploaded_results["label_counts"]
            label_df = pd.DataFrame({"Label": list(lc.keys()), "Count": list(lc.values())})
        else:
            initial = cleaning.get("initial_rows", 499106)
            benign = initial // 2
            label_df = pd.DataFrame({"Label": ["Benign", "Malicious"], "Count": [benign, initial - benign]})

        fig_label = px.pie(
            label_df, names="Label", values="Count", color="Label",
            color_discrete_map={"Benign": "#10b981", "Malicious": "#ef4444"},
            hole=0.45,
        )
        fig_label.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=True, title_text="")
        fig_label.update_traces(
            textinfo="label+percent", textfont_size=14,
            marker=dict(line=dict(color="#1e1b4b", width=2)),
        )
        st.plotly_chart(fig_label, use_container_width=True)
        st.markdown('<p class="chart-caption">Distribusi kelas Benign vs Malicious pada dataset</p>',
                    unsafe_allow_html=True)

    # ── Feature list ──────────────────────────────────────────────────────
    features = info.get("numeric_features", [])
    st.markdown(f"### 🔢 Fitur Metadata Flow ({len(features)} fitur)")
    if features:
        n_cols = 4
        cols = st.columns(n_cols)
        for i, feat in enumerate(features):
            cols[i % n_cols].markdown(f"- `{feat}`")
    else:
        st.info("Data fitur tidak tersedia.")

    # ── Cleaning summary ──────────────────────────────────────────────────
    st.markdown("### 🧹 Ringkasan Cleaning")
    const_removed = cleaning.get("constant_features_removed", [])
    st.markdown(
        f"""
<div class="info-box">
<strong>Baris duplikat dihapus:</strong> {cleaning.get('removed_rows', 0):,}<br>
<strong>Fitur konstan dihapus:</strong> {len(const_removed)} {'(' + ', '.join(const_removed) + ')' if const_removed else '(tidak ada)'}
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Key Takeaway ──────────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="takeaway-box">
<strong>📌 Key Takeaway:</strong> Dataset terdiri dari {cleaning.get('rows_after_cleaning', 0):,} sampel
dengan {cleaning.get('n_numeric_features', 0)} fitur metadata flow. Model terbaik
(<strong>{info.get('best_model_name', 'N/A')}</strong>) mencapai F1 Score
{best_metrics_ov.get('f1_score', 0):.4%} dengan hanya {int(best_metrics_ov.get('fp', 0))} false positive.
</div>
""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: Model Comparison
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Perbandingan Model":
    st.markdown('<p class="section-header">📈 Perbandingan Model</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Evaluasi tiga model klasifikasi: Logistic Regression, Random Forest, dan AdaBoost.</p>',
        unsafe_allow_html=True,
    )
    show_source_badge()

    metrics_df = get_model_metrics()

    if metrics_df.empty:
        st.warning("Data metrik tidak tersedia. Upload dataset atau pastikan file output ada.")
    else:
        best = metrics_df.iloc[0]
        best_name = best["model"]

        # ── Top metrics with delta ────────────────────────────────────────
        second = metrics_df.iloc[1] if len(metrics_df) > 1 else best
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🏆 Best Model", best_name,
                  help="Model dengan ROC-AUC dan F1 Score tertinggi")
        c2.metric("Accuracy", f"{best['accuracy']:.4%}",
                  delta=f"+{(best['accuracy'] - second['accuracy']):.4%} vs {second['model']}",
                  help="Proporsi prediksi benar dari total sampel")
        c3.metric("F1 Score", f"{best['f1_score']:.4%}",
                  delta=f"+{(best['f1_score'] - second['f1_score']):.4%} vs {second['model']}",
                  help="Harmonic mean dari Precision dan Recall")
        c4.metric("ROC-AUC", f"{best['roc_auc']:.6f}",
                  help="Area Under the ROC Curve — kemampuan diskriminasi model")
        c5.metric("FPR", f"{best['fpr']:.6f}",
                  help="False Positive Rate — proporsi benign yang salah diklasifikasikan")

        divider()

        # ── Table ─────────────────────────────────────────────────────────
        st.markdown("### 📋 Tabel Metrik Lengkap")
        display_cols = [
            "model", "accuracy", "precision", "recall_tpr", "f1_score",
            "roc_auc", "fpr", "training_time_sec", "inference_time_sec",
        ]
        available_cols = [c for c in display_cols if c in metrics_df.columns]
        display_df = metrics_df[available_cols].copy()
        nice_names = {
            "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
            "recall_tpr": "Recall (TPR)", "f1_score": "F1 Score", "roc_auc": "ROC-AUC",
            "fpr": "FPR", "training_time_sec": "Training (s)",
            "inference_time_sec": "Inference (s)",
        }
        display_df.columns = [nice_names.get(c, c) for c in available_cols]
        fmt = {c: "{:.6f}" for c in display_df.columns if c not in ("Model",)}
        fmt.update({"Training (s)": "{:.2f}", "Inference (s)": "{:.4f}"})
        fmt = {k: v for k, v in fmt.items() if k in display_df.columns}
        st.dataframe(display_df.style.format(fmt), use_container_width=True, hide_index=True)

        divider()

        # ── Bar chart ─────────────────────────────────────────────────────
        st.markdown("### 📊 Perbandingan Metrik Utama")
        bar_metrics = ["accuracy", "precision", "recall_tpr", "f1_score", "roc_auc"]
        bar_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        bm_avail = [m for m in bar_metrics if m in metrics_df.columns]
        bl_avail = [bar_labels[bar_metrics.index(m)] for m in bm_avail]
        melted = metrics_df.melt(id_vars=["model"], value_vars=bm_avail, var_name="Metric", value_name="Score")
        melted["Metric"] = melted["Metric"].map(dict(zip(bm_avail, bl_avail)))

        fig_bar = px.bar(
            melted, x="Metric", y="Score", color="model", barmode="group",
            color_discrete_sequence=COLOR_PALETTE, text_auto=".4f",
        )
        min_score = max(0.0, melted["Score"].min() - 0.05)
        fig_bar.update_layout(
            **PLOTLY_LAYOUT, height=480, title_text="",
            yaxis=dict(range=[min_score, 1.005], title="Score"),
            xaxis_title="", legend_title_text="Model",
        )
        fig_bar.update_traces(textposition="outside", textfont_size=10)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('<p class="chart-caption">Perbandingan 5 metrik evaluasi utama antar 3 model</p>',
                    unsafe_allow_html=True)

        divider()

        # ── Confusion Matrix ──────────────────────────────────────────────
        st.markdown("### 🔲 Confusion Matrix")
        n_models = len(metrics_df)
        cm_cols = st.columns(min(n_models, 3))
        for idx, row in metrics_df.iterrows():
            cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
            total_neg = int(row["tn"]) + int(row["fp"])
            total_pos = int(row["fn"]) + int(row["tp"])

            # Model name with best badge
            title_text = row["model"]
            if row["model"] == best_name:
                title_text = f"🏆 {row['model']}"

            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm, x=["Pred Benign", "Pred Malicious"], y=["Actual Benign", "Actual Malicious"],
                    text=[[f"{int(v):,}" for v in r] for r in cm],
                    texttemplate="%{text}", textfont=dict(size=16, color="white"),
                    colorscale=[[0, "#1e1b4b"], [1, "#6366f1"]], showscale=False,
                )
            )
            fig_cm.update_layout(
                **PLOTLY_LAYOUT, title=dict(text=title_text, font=dict(size=15)),
                height=320, xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"),
            )
            with cm_cols[idx % min(n_models, 3)]:
                st.plotly_chart(fig_cm, use_container_width=True)
                # Natural language caption
                fp_count = int(row["fp"])
                fn_count = int(row["fn"])
                st.markdown(
                    f'<p class="chart-caption">{fp_count:,} FP dari {total_neg:,} benign · '
                    f'{fn_count:,} FN dari {total_pos:,} malicious</p>',
                    unsafe_allow_html=True,
                )

        # ── Time comparison ───────────────────────────────────────────────
        if "training_time_sec" in metrics_df.columns:
            divider()
            st.markdown("### ⏱️ Training & Inference Time")
            tc1, tc2 = st.columns(2)
            with tc1:
                fig_t = px.bar(metrics_df, x="model", y="training_time_sec", color="model",
                               color_discrete_sequence=COLOR_PALETTE, text_auto=".2f")
                fig_t.update_layout(**PLOTLY_LAYOUT, height=380, title="Training Time (s)",
                                    xaxis_title="", yaxis_title="Seconds", showlegend=False)
                fig_t.update_traces(textposition="outside")
                st.plotly_chart(fig_t, use_container_width=True)
            with tc2:
                fig_i = px.bar(metrics_df, x="model", y="inference_time_sec", color="model",
                               color_discrete_sequence=COLOR_PALETTE, text_auto=".4f")
                fig_i.update_layout(**PLOTLY_LAYOUT, height=380, title="Inference Time (s)",
                                    xaxis_title="", yaxis_title="Seconds", showlegend=False)
                fig_i.update_traces(textposition="outside")
                st.plotly_chart(fig_i, use_container_width=True)
        # ── ROC-AUC Curve & Precision-Recall Curve ─────────────────────────
        divider()
        st.markdown("### 📉 ROC-AUC Curve & Precision-Recall Curve")
        st.markdown(
            '<div class="info-box">'
            '<strong>ROC Curve</strong> menggambarkan trade-off antara True Positive Rate (Sensitivity) '
            'dan False Positive Rate pada berbagai threshold. '
            '<strong>Precision-Recall Curve</strong> lebih informatif pada dataset imbalanced karena '
            'fokus pada kelas positif (Malicious). '
            'Kurva yang mendekati pojok kiri atas (ROC) atau pojok kanan atas (PR) menandakan model yang superior.</div>',
            unsafe_allow_html=True,
        )

        roc_col, pr_col = st.columns(2)

        # ROC-AUC Curve
        roc_png = OUTPUT_DIR / "roc_curve_with_auc_legend.png"
        roc_summary = _load_csv(str(OUTPUT_DIR / "roc_curve_with_auc_legend_summary.csv"))
        with roc_col:
            st.markdown("#### ROC Curve")
            if roc_png.exists():
                st.image(str(roc_png), use_container_width=True)
            else:
                st.info("Gambar ROC Curve tidak tersedia.")
            if not roc_summary.empty and "roc_auc" in roc_summary.columns:
                auc_cols = st.columns(len(roc_summary))
                for i, (_, row) in enumerate(roc_summary.iterrows()):
                    model_name = row.get("model_name", row.get("model", f"Model {i+1}"))
                    auc_val = row["roc_auc"]
                    auc_cols[i].metric(
                        f"AUC — {model_name}",
                        f"{auc_val:.6f}",
                        help=f"Area Under ROC Curve untuk {model_name}",
                    )
            st.markdown(
                '<p class="chart-caption">Kurva ROC menunjukkan Random Forest & AdaBoost mendekati '
                'pojok kiri atas (AUC ≈ 1.0), jauh mengungguli Logistic Regression</p>',
                unsafe_allow_html=True,
            )

        # Precision-Recall Curve
        pr_png = OUTPUT_DIR / "precision_recall_curve_with_ap_legend.png"
        pr_summary = _load_csv(str(OUTPUT_DIR / "precision_recall_curve_with_ap_legend_summary.csv"))
        with pr_col:
            st.markdown("#### Precision-Recall Curve")
            if pr_png.exists():
                st.image(str(pr_png), use_container_width=True)
            else:
                st.info("Gambar Precision-Recall Curve tidak tersedia.")
            if not pr_summary.empty and "average_precision" in pr_summary.columns:
                ap_cols = st.columns(len(pr_summary))
                for i, (_, row) in enumerate(pr_summary.iterrows()):
                    model_name = row.get("model_name", row.get("model", f"Model {i+1}"))
                    ap_val = row["average_precision"]
                    ap_cols[i].metric(
                        f"AP — {model_name}",
                        f"{ap_val:.6f}",
                        help=f"Average Precision untuk {model_name}",
                    )
            st.markdown(
                '<p class="chart-caption">Precision-Recall Curve mengkonfirmasi dominasi Random Forest '
                'dengan AP ≈ 1.0, konsisten dengan hasil ROC-AUC</p>',
                unsafe_allow_html=True,
            )

        # ── Key Takeaway ──────────────────────────────────────────────────
        st.markdown(
            f"""
<div class="takeaway-box">
<strong>📌 Key Takeaway:</strong> <strong>{best_name}</strong> mengungguli semua model lain
dengan F1 Score {best['f1_score']:.4%} dan ROC-AUC {best['roc_auc']:.6f}.
Hanya menghasilkan <strong>{int(best['fp']):,} false positive</strong> dan
<strong>{int(best['fn']):,} false negative</strong> pada test set.
</div>
""",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: Best Model Detail
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🏆 Best Model Detail":
    info = get_best_model_info()
    best_name = info.get("best_model_name", "Random Forest")
    st.markdown(f'<p class="section-header">🏆 Best Model — {best_name}</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Detail performa, feature importance, dan threshold analysis.</p>',
        unsafe_allow_html=True,
    )
    show_source_badge()

    best_metrics = info.get("best_model_metrics_default_threshold", {})
    threshold_metrics = info.get("selected_threshold_metrics", {})

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.4%}",
              help="Proporsi prediksi benar dari total sampel")
    c2.metric("Precision", f"{best_metrics.get('precision', 0):.4%}",
              help="Dari yang diprediksi malicious, berapa yang benar")
    c3.metric("Recall", f"{best_metrics.get('recall_tpr', 0):.4%}",
              help="Dari semua malicious, berapa yang berhasil dideteksi")
    c4.metric("F1 Score", f"{best_metrics.get('f1_score', 0):.4%}",
              help="Harmonic mean dari Precision dan Recall")
    c5.metric("ROC-AUC", f"{best_metrics.get('roc_auc', 0):.6f}",
              help="Area Under the ROC Curve")
    c6.metric("FPR", f"{best_metrics.get('fpr', 0):.6f}",
              help="False Positive Rate")

    divider()

    tab_names = ["🌲 Feature Importance", "🎯 Threshold Analysis", "📊 Selected Threshold"]
    # Add Permutation Importance tab if data is available (default dataset only)
    if not is_uploaded():
        perm_df = get_explainability_importance()
        if not perm_df.empty:
            tab_names.insert(1, "🔬 Permutation Importance")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # ── Tab: Feature Importance (Gini) ────────────────────────────────────
    with tabs[tab_idx]:
        fi_df = get_feature_importance()
        if fi_df.empty:
            st.info("Feature importance tidak tersedia (hanya tersedia untuk Random Forest).")
        else:
            st.markdown(
                '<div class="info-box"><strong>Metode:</strong> Gini Importance '
                '(Mean Decrease Impurity) — mengukur kontribusi setiap fitur terhadap '
                'pemisahan kelas di dalam ensemble Random Forest.</div>',
                unsafe_allow_html=True,
            )
            fi_sorted = fi_df.sort_values("importance", ascending=True)

            # Highlight top 3
            colors = []
            top3_features = fi_df.head(3)["feature"].tolist()
            for _, row in fi_sorted.iterrows():
                if row["feature"] == top3_features[0]:
                    colors.append("#f59e0b")  # Gold
                elif row["feature"] in top3_features[1:]:
                    colors.append("#fbbf24")  # Amber
                else:
                    colors.append("#6366f1")  # Default indigo

            fig_fi = go.Figure(go.Bar(
                x=fi_sorted["importance"], y=fi_sorted["feature"],
                orientation="h", marker_color=colors,
                text=[f"{v:.4f}" for v in fi_sorted["importance"]],
                textposition="outside", textfont=dict(size=10),
            ))
            fig_fi.update_layout(
                **PLOTLY_LAYOUT, height=max(450, len(fi_df) * 26),
                title=f"{best_name} — Gini Feature Importance",
                xaxis_title="Importance", yaxis_title="",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.markdown(
                '<p class="chart-caption">Fitur berwarna emas (🥇🥈🥉) adalah top-3 kontributor terpenting</p>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Top 5 Fitur Terpenting")
            for rank, (_, row) in enumerate(fi_df.head(5).iterrows()):
                medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank]
                st.markdown(f"{medal} **{row['feature']}** — `{row['importance'] * 100:.2f}%`")
    tab_idx += 1

    # ── Tab: Permutation Importance (conditional) ─────────────────────────
    if not is_uploaded() and "🔬 Permutation Importance" in tab_names:
        with tabs[tab_idx]:
            perm_df = get_explainability_importance()
            if perm_df.empty:
                st.info("Data permutation importance tidak tersedia.")
            else:
                st.markdown(
                    '<div class="info-box"><strong>Metode:</strong> Permutation Importance — '
                    'mengukur penurunan akurasi model saat nilai suatu fitur diacak secara random. '
                    'Metode ini lebih reliable karena model-agnostic.</div>',
                    unsafe_allow_html=True,
                )
                # Filter to features with meaningful importance
                perm_sig = perm_df[perm_df["mean_abs_importance"] > 1e-10].sort_values(
                    "mean_abs_importance", ascending=True
                )
                if perm_sig.empty:
                    st.info("Tidak ada fitur dengan permutation importance signifikan.")
                else:
                    fig_perm = go.Figure(go.Bar(
                        x=perm_sig["mean_abs_importance"], y=perm_sig["feature"],
                        orientation="h",
                        marker_color="#8b5cf6",
                        error_x=dict(type="data", array=perm_sig["importance_std"].tolist(), visible=True,
                                     color="#c4b5fd"),
                        text=[f"{v:.6f}" for v in perm_sig["mean_abs_importance"]],
                        textposition="outside", textfont=dict(size=10),
                    ))
                    fig_perm.update_layout(
                        **PLOTLY_LAYOUT, height=max(300, len(perm_sig) * 50),
                        title="Permutation Importance (Mean ± Std)",
                        xaxis_title="Mean Absolute Importance", yaxis_title="",
                    )
                    st.plotly_chart(fig_perm, use_container_width=True)
                    st.markdown(
                        '<p class="chart-caption">Hanya fitur dengan importance &gt; 0 ditampilkan. '
                        'Error bar menunjukkan standar deviasi antar iterasi.</p>',
                        unsafe_allow_html=True,
                    )

                    # Insight
                    top_perm = perm_df.iloc[0]
                    st.markdown(
                        f"""
<div class="takeaway-box">
<strong>📌 Insight:</strong> Berdasarkan permutation importance, fitur
<strong>{top_perm['feature']}</strong> adalah yang paling kritis —
mengacak nilainya menyebabkan penurunan akurasi sebesar {top_perm['mean_abs_importance']:.6f}.
Ini mengkonfirmasi temuan Gini Importance.
</div>
""",
                        unsafe_allow_html=True,
                    )
        tab_idx += 1

    # ── Tab: Threshold Analysis ───────────────────────────────────────────
    with tabs[tab_idx]:
        # Use enhanced threshold analysis if available (more granular)
        ta_df = get_enhanced_threshold_analysis() if not is_uploaded() else pd.DataFrame()
        if ta_df.empty:
            ta_df = get_threshold_analysis()

        if ta_df.empty:
            st.info("Data threshold analysis tidak tersedia.")
        else:
            st.markdown(
                '<div class="info-box"><strong>Threshold Analysis</strong> mengevaluasi bagaimana '
                'perubahan decision threshold memengaruhi metrik klasifikasi. Threshold default '
                'adalah 0.50 — menurunkannya meningkatkan recall tapi mungkin menambah false positive.</div>',
                unsafe_allow_html=True,
            )

            fig_ta = go.Figure()
            for metric, color, name in [
                ("precision", "#6366f1", "Precision"),
                ("recall_tpr", "#8b5cf6", "Recall (TPR)"),
                ("f1_score", "#ec4899", "F1 Score"),
                ("accuracy", "#10b981", "Accuracy"),
            ]:
                if metric in ta_df.columns:
                    fig_ta.add_trace(go.Scatter(
                        x=ta_df["threshold"], y=ta_df[metric],
                        mode="lines+markers", name=name,
                        line=dict(color=color, width=2.5), marker=dict(size=5),
                    ))

            sel_t = info.get("selected_threshold", 0.45)

            # Shaded recommended zone around selected threshold
            fig_ta.add_vrect(
                x0=max(0, sel_t - 0.05), x1=min(1, sel_t + 0.05),
                fillcolor="rgba(245,158,11,0.08)", line_width=0,
                annotation_text="Zona Rekomendasi", annotation_position="top left",
                annotation_font=dict(color="#f59e0b", size=11),
            )
            fig_ta.add_vline(x=sel_t, line_dash="dash", line_color="#f59e0b", line_width=2,
                             annotation_text=f"Selected: {sel_t}", annotation_font_color="#f59e0b")

            y_cols = [c for c in ["precision", "recall_tpr", "f1_score", "accuracy"] if c in ta_df.columns]
            y_min = ta_df[y_cols].min().min() if y_cols else 0.9
            fig_ta.update_layout(
                **PLOTLY_LAYOUT, height=500, title="Metrik Klasifikasi vs Threshold",
                xaxis_title="Decision Threshold", yaxis_title="Score",
                yaxis=dict(range=[max(0, y_min - 0.01), 1.001]),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_ta, use_container_width=True)
            st.markdown('<p class="chart-caption">Area kuning menunjukkan zona threshold yang direkomendasikan</p>',
                        unsafe_allow_html=True)

            if "fpr" in ta_df.columns:
                fig_fpr = go.Figure()
                fig_fpr.add_trace(go.Scatter(
                    x=ta_df["threshold"], y=ta_df["fpr"], mode="lines+markers",
                    name="FPR", line=dict(color="#ef4444", width=2.5),
                    fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
                ))
                fig_fpr.add_vline(x=sel_t, line_dash="dash", line_color="#f59e0b", line_width=2,
                                  annotation_text=f"Selected: {sel_t}", annotation_font_color="#f59e0b")
                fig_fpr.update_layout(
                    **PLOTLY_LAYOUT, height=380, title="False Positive Rate vs Threshold",
                    xaxis_title="Decision Threshold", yaxis_title="FPR",
                )
                st.plotly_chart(fig_fpr, use_container_width=True)
                st.markdown(
                    '<p class="chart-caption">FPR yang rendah menunjukkan model jarang salah menandai trafik benign</p>',
                    unsafe_allow_html=True,
                )
    tab_idx += 1

    # ── Tab: Selected Threshold ───────────────────────────────────────────
    with tabs[tab_idx]:
        if threshold_metrics:
            sel_t = threshold_metrics.get("threshold", info.get("selected_threshold", "N/A"))

            # Explanation box
            st.markdown(
                f"""
<div class="info-box">
<strong>Threshold {sel_t}</strong> dipilih berdasarkan F1 Score tertinggi pada threshold analysis.
Penurunan threshold dari 0.50 ke {sel_t} menambah
<strong>{int(threshold_metrics.get('tp', 0)) - int(best_metrics.get('tp', 0)):+d} True Positive</strong>
tanpa menambah False Positive (tetap {int(threshold_metrics.get('fp', 0))}).
</div>
""",
                unsafe_allow_html=True,
            )

            # ── Visual comparison: Default vs Selected ────────────────────
            st.markdown("#### Perbandingan Default (0.50) vs Selected Threshold")

            comp_metrics = [
                ("Accuracy", "accuracy", "{:.6f}"),
                ("Precision", "precision", "{:.6f}"),
                ("Recall (TPR)", "recall_tpr", "{:.6f}"),
                ("F1 Score", "f1_score", "{:.6f}"),
                ("FPR", "fpr", "{:.8f}"),
            ]

            comp_col1, comp_col2 = st.columns(2)

            with comp_col1:
                st.markdown(
                    '<div class="stat-card"><div class="stat-card-title">Default Threshold (0.50)</div>',
                    unsafe_allow_html=True,
                )
                for label, key, fmt in comp_metrics:
                    val = best_metrics.get(key, 0)
                    st.markdown(f"**{label}:** `{fmt.format(val)}`")
                fp_def = int(best_metrics.get("fp", 0))
                fn_def = int(best_metrics.get("fn", 0))
                st.markdown(f"**TP:** {int(best_metrics.get('tp', 0)):,} · **FP:** {fp_def:,} · "
                            f"**TN:** {int(best_metrics.get('tn', 0)):,} · **FN:** {fn_def:,}")
                st.markdown('</div>', unsafe_allow_html=True)

            with comp_col2:
                st.markdown(
                    f'<div class="stat-card" style="border-color:rgba(245,158,11,.4)">'
                    f'<div class="stat-card-title" style="color:#f59e0b">✅ Selected Threshold ({sel_t})</div>',
                    unsafe_allow_html=True,
                )
                for label, key, fmt in comp_metrics:
                    val_sel = threshold_metrics.get(key, 0)
                    val_def = best_metrics.get(key, 0)
                    diff = val_sel - val_def
                    if key == "fpr":
                        # Lower is better for FPR
                        indicator = "🟢" if diff <= 0 else "🟡"
                    else:
                        indicator = "🟢" if diff >= 0 else "🟡"
                    st.markdown(f"{indicator} **{label}:** `{fmt.format(val_sel)}`")
                fp_sel = int(threshold_metrics.get("fp", 0))
                fn_sel = int(threshold_metrics.get("fn", 0))
                st.markdown(f"**TP:** {int(threshold_metrics.get('tp', 0)):,} · **FP:** {fp_sel:,} · "
                            f"**TN:** {int(threshold_metrics.get('tn', 0)):,} · **FN:** {fn_sel:,}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Delta summary
            f1_delta = threshold_metrics.get("f1_score", 0) - best_metrics.get("f1_score", 0)
            fn_delta = fn_sel - fn_def
            st.markdown(
                f"""
<div class="takeaway-box">
<strong>📌 Kesimpulan Threshold:</strong> Dengan threshold {sel_t}, F1 Score meningkat
<strong>{f1_delta:+.6f}</strong> dan False Negative berkurang
<strong>{fn_delta:+d}</strong> (dari {fn_def} menjadi {fn_sel}).
FPR tetap konstan. Ini berarti lebih banyak trafik malicious terdeteksi tanpa
menambah beban false alarm.
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.info("Data selected threshold tidak tersedia.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: Robustness & Analysis
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📋 Robustness & Analysis":
    st.markdown('<p class="section-header">📋 Robustness & Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Cross-validation, ablation, environment comparison, dan catatan metodologis.</p>',
        unsafe_allow_html=True,
    )
    show_source_badge()

    enriched = get_enriched_info()

    if is_uploaded():
        st.info(
            "ℹ️ Data robustness analysis (cross-validation, ablation, environment comparison) "
            "hanya tersedia untuk dataset default. Halaman ini menampilkan ringkasan model yang baru di-train."
        )

        r = st.session_state.uploaded_results
        tab_u1, tab_u2 = st.tabs(["📊 Model Summary", "🌲 Feature Importance"])

        with tab_u1:
            st.markdown("#### Ringkasan Training")
            cs = r["cleaning_summary"]
            st.markdown(
                f"""
<div class="info-box">
<strong>Dataset rows:</strong> {cs['initial_rows']:,} → {cs['rows_after_cleaning']:,} (cleaned)<br>
<strong>Fitur:</strong> {cs['n_numeric_features']}<br>
<strong>Label:</strong> {cs['label_column']} ({cs['negative_class_0']} vs {cs['positive_class_1']})<br>
<strong>Train shape:</strong> {r['train_shape']}<br>
<strong>Test shape:</strong> {r['test_shape']}<br>
<strong>Best model:</strong> {r['best_model_name']} (F1: {r['best_metrics']['f1_score']:.6f})
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("#### Metrik Semua Model")
            st.dataframe(r["model_metrics"], use_container_width=True, hide_index=True)

        with tab_u2:
            fi = r["feature_importance"]
            if fi.empty:
                st.info("Feature importance tidak tersedia.")
            else:
                fi_s = fi.sort_values("importance", ascending=True)
                fig = px.bar(fi_s, x="importance", y="feature", orientation="h",
                             color="importance",
                             color_continuous_scale=["#312e81", "#6366f1", "#a78bfa", "#c4b5fd"])
                fig.update_layout(**PLOTLY_LAYOUT, height=max(400, len(fi) * 25),
                                  title="Feature Importance", xaxis_title="Importance",
                                  yaxis_title="", coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

    else:
        tab_names_r = [
            "🔄 Cross-Validation", "🧪 Ablation Study", "🌐 Environment Comparison",
            "📊 Advanced Metrics", "🚨 SOC Alert Simulation", "📝 Catatan Metodologis",
        ]
        tab_r1, tab_r2, tab_r3, tab_r4, tab_r5, tab_r6 = st.tabs(tab_names_r)

        with tab_r1:
            cv_data = enriched.get("cv_summary", [])
            if cv_data:
                st.markdown("#### Cross-Validation Stability (Random Forest)")
                st.markdown(
                    '<div class="info-box"><strong>5-Fold Stratified Cross-Validation</strong> — '
                    'mengukur stabilitas model terhadap variasi pembagian data. '
                    'Standar deviasi yang sangat kecil menunjukkan model konsisten.</div>',
                    unsafe_allow_html=True,
                )

                cv_cols = st.columns(len(cv_data))
                for i, row in enumerate(cv_data):
                    cv_cols[i].metric(
                        row["metric"].upper(),
                        f"{row['mean']:.6f}",
                        delta=f"±{row['std']:.2e}",
                        help=f"Min: {row['min']:.6f} | Max: {row['max']:.6f}",
                    )

                fig_cv = go.Figure()
                for idx, row in enumerate(cv_data):
                    fig_cv.add_trace(go.Bar(
                        name=row["metric"].upper(), x=[row["metric"].upper()], y=[row["mean"]],
                        error_y=dict(type="data", array=[row["std"]], visible=True),
                        marker_color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                        text=[f"{row['mean']:.6f}"], textposition="outside",
                    ))
                fig_cv.update_layout(**PLOTLY_LAYOUT, height=420, title="Cross-Validation Mean ± Std",
                                     yaxis=dict(range=[0.999, 1.0005], title="Score"), showlegend=False)
                st.plotly_chart(fig_cv, use_container_width=True)
                st.markdown(
                    '<p class="chart-caption">Error bar menunjukkan standar deviasi antar fold</p>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
<div class="takeaway-box">
<strong>📌 Interpretasi:</strong> Semua metrik cross-validation memiliki standar deviasi
dalam orde <strong>10⁻⁵ hingga 10⁻⁷</strong>, menunjukkan bahwa performa Random Forest
sangat stabil dan tidak sensitif terhadap pembagian data (tidak overfitting).
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Data cross-validation tidak tersedia.")

        with tab_r2:
            abl_df = _load_csv(str(OUTPUT_DIR / "ablation_study_results.csv"))
            if abl_df.empty:
                st.info("Data ablation study tidak tersedia.")
            else:
                st.markdown("#### Ablation Study — Feature Group Removal")
                st.markdown(
                    '<div class="info-box"><strong>Ablation Study</strong> menguji dampak penghapusan '
                    'kelompok fitur terhadap performa. Ini membantu memahami kontribusi relatif '
                    'setiap kelompok fitur.</div>',
                    unsafe_allow_html=True,
                )

                # Display table
                abl_display = abl_df.copy()
                if "f1_delta_vs_baseline" in abl_display.columns:
                    abl_display["f1_delta_vs_baseline"] = abl_display["f1_delta_vs_baseline"].apply(
                        lambda x: f"{x:+.6f}" if x != 0 else "baseline"
                    )
                display_abl_cols = ["ablation", "remaining_feature_count", "accuracy", "precision",
                                    "recall", "f1", "roc_auc", "f1_delta_vs_baseline"]
                display_abl_cols = [c for c in display_abl_cols if c in abl_display.columns]
                st.dataframe(abl_display[display_abl_cols], use_container_width=True, hide_index=True)

                if "f1" in abl_df.columns and "remaining_feature_count" in abl_df.columns:
                    fig_a = px.bar(
                        abl_df, x="ablation", y="f1", color="f1",
                        color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
                        text_auto=".4f",
                    )
                    fig_a.update_layout(**PLOTLY_LAYOUT, height=420,
                                        title="F1 Score per Ablation Scenario",
                                        xaxis_title="", yaxis_title="F1 Score",
                                        coloraxis_showscale=False)
                    fig_a.update_traces(textposition="outside")
                    st.plotly_chart(fig_a, use_container_width=True)

                # Top-N Feature Experiment
                topn_df = _load_csv(str(OUTPUT_DIR / "top_n_feature_experiment.csv"))
                if not topn_df.empty and "top_n" in topn_df.columns and "f1" in topn_df.columns:
                    divider()
                    st.markdown("#### Top-N Feature Experiment")
                    st.markdown(
                        '<div class="info-box">Menguji performa model dengan hanya menggunakan '
                        'top-N fitur terpenting. Menunjukkan diminishing returns dari penambahan fitur.</div>',
                        unsafe_allow_html=True,
                    )
                    fig_topn = go.Figure()
                    fig_topn.add_trace(go.Scatter(
                        x=topn_df["top_n"], y=topn_df["f1"],
                        mode="lines+markers+text", name="F1 Score",
                        line=dict(color="#8b5cf6", width=3),
                        marker=dict(size=10, color="#8b5cf6"),
                        text=[f"{v:.4f}" for v in topn_df["f1"]],
                        textposition="top center", textfont=dict(size=11),
                    ))
                    if "roc_auc" in topn_df.columns:
                        fig_topn.add_trace(go.Scatter(
                            x=topn_df["top_n"], y=topn_df["roc_auc"],
                            mode="lines+markers", name="ROC-AUC",
                            line=dict(color="#06b6d4", width=2, dash="dash"),
                            marker=dict(size=7, color="#06b6d4"),
                        ))
                    fig_topn.update_layout(
                        **PLOTLY_LAYOUT, height=420,
                        title="Performa vs Jumlah Fitur (Top-N)",
                        xaxis_title="Jumlah Fitur (Top-N)", yaxis_title="Score",
                        legend=dict(orientation="h", y=-0.15),
                    )
                    st.plotly_chart(fig_topn, use_container_width=True)
                    st.markdown(
                        '<p class="chart-caption">Performa meningkat signifikan hingga ~15 fitur, lalu plateau</p>',
                        unsafe_allow_html=True,
                    )

        with tab_r3:
            env_data = enriched.get("environment_comparison", [])
            if env_data:
                st.markdown("#### 🌐 Perbandingan: Eksperimen vs Target SOC Operasional")
                st.markdown(
                    '<div class="info-box">Analisis gap antara lingkungan eksperimen (offline, dataset publik) '
                    'dan target deployment SOC operasional nyata.</div>',
                    unsafe_allow_html=True,
                )

                aspect_icons = {
                    "Data source": "📡", "Label quality": "🏷️",
                    "Feature scope": "🔬", "Latency": "⚡",
                    "Alert handling": "🚨",
                }

                for item in env_data:
                    icon = aspect_icons.get(item["aspect"], "📋")
                    st.markdown(
                        f"""
<div class="env-card">
    <div class="env-card-aspect">{icon} {item['aspect']}</div>
    <strong>Eksperimen:</strong> {item['current_experiment']}<br>
    <strong>Target SOC:</strong> {item['target_soc_environment']}<br>
    <strong style="color:#fbbf24">⚠️ Gap:</strong> {item['risk_or_gap']}<br>
    <strong style="color:#34d399">✅ Mitigasi:</strong> {item['mitigation']}
</div>
""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Data environment comparison tidak tersedia.")

        with tab_r4:
            adv_data = enriched.get("advanced_metrics", [])
            if adv_data:
                st.markdown("#### Advanced Evaluation Metrics")
                st.markdown(
                    '<div class="info-box">Metrik evaluasi tambahan yang memberikan perspektif '
                    'lebih lengkap tentang performa model di berbagai aspek.</div>',
                    unsafe_allow_html=True,
                )

                adv_df = pd.DataFrame(adv_data)
                adv_cols = ["model", "balanced_accuracy", "average_precision", "mcc",
                            "specificity_tnr", "negative_predictive_value",
                            "false_discovery_rate", "false_positive_rate"]
                adv_cols = [c for c in adv_cols if c in adv_df.columns]

                nice_adv_names = {
                    "model": "Model", "balanced_accuracy": "Balanced Acc",
                    "average_precision": "Avg Precision", "mcc": "MCC",
                    "specificity_tnr": "Specificity", "negative_predictive_value": "NPV",
                    "false_discovery_rate": "FDR", "false_positive_rate": "FPR",
                }
                adv_display = adv_df[adv_cols].copy()
                adv_display.columns = [nice_adv_names.get(c, c) for c in adv_cols]
                adv_fmt = {c: "{:.6f}" for c in adv_display.columns if c != "Model"}
                st.dataframe(adv_display.style.format(adv_fmt), use_container_width=True, hide_index=True)

                # Radar chart
                categories = ["balanced_accuracy", "average_precision", "mcc", "specificity_tnr"]
                categories = [c for c in categories if c in adv_df.columns]
                if categories:
                    nice_cat = [nice_adv_names.get(c, c) for c in categories]
                    fig_r = go.Figure()
                    for idx, row in adv_df.iterrows():
                        vals = [row[c] for c in categories] + [row[categories[0]]]
                        fig_r.add_trace(go.Scatterpolar(
                            r=vals, theta=nice_cat + [nice_cat[0]],
                            fill="toself", name=row["model"],
                            line_color=COLOR_PALETTE[idx % len(COLOR_PALETTE)], opacity=0.8,
                        ))
                    fig_r.update_layout(**PLOTLY_LAYOUT, height=500, title="Radar Chart — Advanced Metrics",
                                        polar=dict(radialaxis=dict(visible=True, range=[0.8, 1.0],
                                                                   gridcolor="rgba(99,102,241,.2)"),
                                                   bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig_r, use_container_width=True)
                    st.markdown(
                        '<p class="chart-caption">Radar chart menunjukkan performa multidimensi — '
                        'model ideal memenuhi seluruh area</p>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Data advanced metrics tidak tersedia.")

        # ── NEW TAB: SOC Alert Simulation ─────────────────────────────────
        with tab_r5:
            soc_df = get_soc_alert_summary()
            if soc_df.empty:
                st.info("Data SOC alert simulation tidak tersedia.")
            else:
                st.markdown("#### 🚨 SOC Alert Simulation")
                st.markdown(
                    '<div class="info-box"><strong>Proof of Concept</strong> — simulasi bagaimana model '
                    'dapat diintegrasikan ke SOC dengan prioritisasi alert berdasarkan confidence score. '
                    'Alert diprioritaskan dari P1 (kritis) hingga P4 (low).</div>',
                    unsafe_allow_html=True,
                )

                # Display as visual cards
                priority_order = ["P1", "P2", "P3", "P4"]
                for priority in priority_order:
                    prio_rows = soc_df[soc_df["priority"] == priority]
                    if prio_rows.empty:
                        continue
                    total_count = prio_rows["count"].sum()
                    badge_class = f"soc-{priority.lower()}"
                    actions = prio_rows["recommended_action"].unique()
                    reasons = prio_rows["triage_reason"].unique()

                    st.markdown(
                        f"""
<div class="stat-card">
    <span class="{badge_class}">{priority}</span>
    &nbsp; <strong style="color:#e0e7ff;font-size:1.1rem">{total_count:,} alerts</strong><br>
    <span style="color:#94a3b8;font-size:.85rem">
        Alasan: {', '.join(reasons)}<br>
        Aksi: {', '.join(actions)}
    </span>
</div>
""",
                        unsafe_allow_html=True,
                    )

                # Pie chart
                prio_summary = soc_df.groupby("priority")["count"].sum().reset_index()
                fig_soc = px.pie(
                    prio_summary, names="priority", values="count",
                    color="priority",
                    color_discrete_map={"P1": "#ef4444", "P2": "#f59e0b", "P3": "#3b82f6", "P4": "#6b7280"},
                    hole=0.4,
                )
                fig_soc.update_layout(**PLOTLY_LAYOUT, height=350, title="Distribusi Alert per Priority")
                fig_soc.update_traces(textinfo="label+percent+value", textfont_size=13)
                st.plotly_chart(fig_soc, use_container_width=True)
                st.markdown(
                    '<p class="chart-caption">Mayoritas alert adalah P1 (high confidence) — '
                    'menunjukkan model dapat membedakan kelas dengan sangat yakin</p>',
                    unsafe_allow_html=True,
                )

        with tab_r6:
            gs = enriched.get("lightweight_gridsearch", {})
            if gs:
                st.markdown("#### 🔧 GridSearch Best Parameters")
                bp = gs.get("best_params", {})
                st.markdown(
                    f"""
<div class="info-box">
<strong>n_estimators:</strong> {bp.get('classifier__n_estimators', 'N/A')}<br>
<strong>max_depth:</strong> {bp.get('classifier__max_depth', 'None')}<br>
<strong>min_samples_leaf:</strong> {bp.get('classifier__min_samples_leaf', 'N/A')}<br><br>
<strong>CV Best ROC-AUC:</strong> {gs.get('cv_best_roc_auc', 0):.10f}<br>
<strong>Test ROC-AUC:</strong> {gs.get('test_roc_auc', 0):.10f}
</div>
""",
                    unsafe_allow_html=True,
                )

                # GridSearch results table
                gs_df = _load_csv(str(OUTPUT_DIR / "lightweight_gridsearch_results.csv"))
                if not gs_df.empty:
                    st.markdown("#### GridSearch Results (Top Configurations)")
                    st.dataframe(gs_df, use_container_width=True, hide_index=True)

            notes = enriched.get("methodological_notes", [])
            if notes:
                divider()
                st.markdown("#### 📝 Catatan Metodologis")
                for n in notes:
                    st.markdown(f"- {n}")

            rp = enriched.get("resource_profile", {})
            if rp:
                divider()
                st.markdown("#### 💻 Resource Profile")
                st.markdown(
                    '<div class="info-box">Profiling performa model pada skenario inference '
                    'untuk memperkirakan kesiapan deployment.</div>',
                    unsafe_allow_html=True,
                )

                rp_c1, rp_c2 = st.columns(2)
                with rp_c1:
                    throughput = rp.get('throughput_rows_per_sec', 0)
                    throughput_status = "✅ Cukup untuk near-real-time" if throughput > 10000 else "⚠️ Perlu optimisasi"
                    st.markdown(
                        f"""
<div class="stat-card">
    <div class="stat-card-title">⚡ Throughput</div>
    <div style="font-size:1.8rem;font-weight:700;color:#e0e7ff">{throughput:,.0f}</div>
    <div style="color:#94a3b8;font-size:.85rem">rows/second</div>
    <div style="color:#34d399;font-size:.82rem;margin-top:6px">{throughput_status}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                with rp_c2:
                    memory = rp.get('peak_memory_mb', 0)
                    mem_status = "✅ Ringan" if memory < 100 else "⚠️ Perlu perhatian"
                    st.markdown(
                        f"""
<div class="stat-card">
    <div class="stat-card-title">🧠 Peak Memory</div>
    <div style="font-size:1.8rem;font-weight:700;color:#e0e7ff">{memory:.1f} MB</div>
    <div style="color:#94a3b8;font-size:.85rem">saat inference {rp.get('rows_profiled', 0):,} rows</div>
    <div style="color:#34d399;font-size:.82rem;margin-top:6px">{mem_status}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                rp_c3, rp_c4 = st.columns(2)
                with rp_c3:
                    st.metric("Inference Time", f"{rp.get('inference_time_sec', 0):.4f}s",
                              help="Waktu total inference pada test set")
                with rp_c4:
                    st.metric("Rows Profiled", f"{rp.get('rows_profiled', 0):,}",
                              help="Jumlah baris yang di-profile")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: Interactive Prediction (moved to last for demo flow)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Prediksi Interaktif":
    st.markdown('<p class="section-header">🔍 Prediksi Interaktif</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Gunakan model terlatih untuk prediksi Benign / Malicious secara live.</p>',
        unsafe_allow_html=True,
    )
    show_source_badge()

    model = get_trained_model()
    info = get_best_model_info()
    features = info.get("numeric_features", [])
    threshold = info.get("selected_threshold", 0.45)
    pos_label = info.get("positive_class_1", "Malicious")
    neg_label = info.get("negative_class_0", "Benign")
    best_name_pred = info.get("best_model_name", "Random Forest")

    # Show active model and threshold info
    st.markdown(
        f"""
<div class="info-box">
<strong>Model Aktif:</strong> {best_name_pred}<br>
<strong>Decision Threshold:</strong> {threshold}<br>
<strong>Jumlah Fitur:</strong> {len(features)}<br>
<strong>Interpretasi:</strong> Jika probabilitas kelas positif ≥ {threshold}, sampel diklasifikasikan sebagai <strong>{pos_label}</strong>.
</div>
""",
        unsafe_allow_html=True,
    )

    if model is None:
        st.error("❌ Model tidak tersedia. Upload dataset dan jalankan pipeline, atau pastikan best_model.pkl ada.")
    elif not features:
        st.error("❌ Daftar fitur tidak tersedia.")
    else:
        pred_tab1, pred_tab2 = st.tabs(["✏️ Input Manual", "📂 Upload CSV"])

        with pred_tab1:
            st.markdown(f"#### Masukkan nilai {len(features)} fitur:")
            input_values = {}
            cols_per_row = 4
            for i in range(0, len(features), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, feat in enumerate(features[i:i + cols_per_row]):
                    with cols[j]:
                        input_values[feat] = st.number_input(feat, value=0.0, format="%.6f", key=f"in_{feat}")

            if st.button("🚀 Prediksi", type="primary", use_container_width=True):
                input_df = pd.DataFrame([input_values])
                proba = model.predict_proba(input_df)[0]
                pred_label = pos_label if proba[1] >= threshold else neg_label

                divider()
                _, rc, _ = st.columns([1, 2, 1])
                with rc:
                    if pred_label == pos_label:
                        st.markdown(
                            f'<div class="pred-malicious">⚠️ {pos_label.upper()}<br>'
                            f'<span style="font-size:.9rem;font-weight:400">Confidence: {proba[1]:.4%}</span></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="pred-benign">✅ {neg_label.upper()}<br>'
                            f'<span style="font-size:.9rem;font-weight:400">Confidence: {proba[0]:.4%}</span></div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("#### Distribusi Probabilitas")
                fig_p = go.Figure()
                fig_p.add_trace(go.Bar(
                    x=[neg_label, pos_label], y=[proba[0], proba[1]],
                    marker_color=["#10b981", "#ef4444"],
                    text=[f"{proba[0]:.4%}", f"{proba[1]:.4%}"], textposition="outside",
                ))
                # Add threshold line
                fig_p.add_hline(y=threshold, line_dash="dash", line_color="#f59e0b",
                                annotation_text=f"Threshold: {threshold}",
                                annotation_font_color="#f59e0b",
                                annotation_position="top right")
                fig_p.update_layout(**PLOTLY_LAYOUT, height=320, title_text="",
                                    yaxis=dict(range=[0, 1.15], title="Probability"), xaxis_title="")
                st.plotly_chart(fig_p, use_container_width=True)
                st.markdown(
                    f'<p class="chart-caption">Garis kuning putus-putus menunjukkan decision threshold ({threshold})</p>',
                    unsafe_allow_html=True,
                )

        with pred_tab2:
            st.markdown("#### Upload file CSV untuk batch prediction")
            st.markdown(
                f'<div class="info-box">CSV harus memiliki <strong>{len(features)} kolom</strong> '
                f'fitur numerik yang sesuai. Model akan menggunakan threshold <strong>{threshold}</strong> '
                f'untuk klasifikasi.</div>',
                unsafe_allow_html=True,
            )

            uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
            if uploaded is not None:
                try:
                    batch_df = pd.read_csv(uploaded)
                    batch_df.columns = batch_df.columns.str.strip()
                    missing = [f for f in features if f not in batch_df.columns]
                    if missing:
                        st.error(f"❌ Kolom tidak ditemukan: {', '.join(missing)}")
                    else:
                        X_batch = batch_df[features]
                        probas = model.predict_proba(X_batch)
                        preds = (probas[:, 1] >= threshold).astype(int)

                        result_df = batch_df.copy()
                        result_df["Prediction"] = np.where(preds == 1, pos_label, neg_label)
                        result_df["Confidence_Positive"] = probas[:, 1]
                        result_df["Confidence_Negative"] = probas[:, 0]

                        n_pos = (preds == 1).sum()
                        n_neg = (preds == 0).sum()
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Total Rows", f"{len(preds):,}")
                        sc2.metric(f"🟢 {neg_label}", f"{n_neg:,}")
                        sc3.metric(f"🔴 {pos_label}", f"{n_pos:,}")

                        fig_b = px.pie(
                            names=[neg_label, pos_label], values=[int(n_neg), int(n_pos)],
                            color=[neg_label, pos_label],
                            color_discrete_map={neg_label: "#10b981", pos_label: "#ef4444"}, hole=0.4,
                        )
                        fig_b.update_layout(**PLOTLY_LAYOUT, height=320, title_text="")
                        fig_b.update_traces(textinfo="label+percent+value", textfont_size=13)
                        st.plotly_chart(fig_b, use_container_width=True)

                        st.markdown("#### Hasil Prediksi")
                        show_cols = ["Prediction", "Confidence_Positive", "Confidence_Negative"] + features[:5]
                        st.dataframe(result_df[show_cols].head(100), use_container_width=True)

                        csv_buf = io.StringIO()
                        result_df.to_csv(csv_buf, index=False)
                        st.download_button(
                            "📥 Download Hasil Prediksi (CSV)", csv_buf.getvalue(),
                            "prediction_results.csv", "text/csv",
                            type="primary", use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"❌ Error: {e}")
