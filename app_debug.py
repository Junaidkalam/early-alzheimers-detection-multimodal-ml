import os

# Must be set BEFORE any tensorflow import
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF C++ warnings

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────────────────────
# TensorFlow (optional — only needed for MRI)
# ─────────────────────────────────────────────────────────────────────────────
TF_AVAILABLE = False
TF_ERROR     = ""
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.applications.efficientnet import preprocess_input
    TF_AVAILABLE = True
except Exception as _e:
    TF_ERROR = str(_e)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #050d1a; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #1a2d45;
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 13px;
}

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }

/* ── Custom card ── */
.ns-card {
    background: #0d1f35;
    border: 1px solid #1a2d45;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.ns-card-accent {
    background: linear-gradient(135deg, #0d1f35 0%, #0a1a2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
}

/* ── Section title ── */
.ns-section-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.ns-title {
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 6px;
}
.ns-subtitle {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 20px;
    line-height: 1.55;
}

/* ── Risk meter ── */
.risk-bar-container {
    background: #0a1628;
    border-radius: 8px;
    height: 14px;
    overflow: hidden;
    border: 1px solid #1a2d45;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}

/* ── Result box ── */
.result-high {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px;
    padding: 20px 24px;
}
.result-moderate {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 12px;
    padding: 20px 24px;
}
.result-low {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 12px;
    padding: 20px 24px;
}

/* ── Metric pill ── */
.metric-pill {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 12px;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

/* ── Stacked metric ── */
.ns-metric {
    background: #0a1628;
    border: 1px solid #1a2d45;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.ns-metric .val {
    font-size: 24px;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1;
}
.ns-metric .lbl {
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Model status dot ── */
.dot-on  { display:inline-block; width:7px; height:7px; border-radius:50%; background:#34d399; margin-right:6px; }
.dot-off { display:inline-block; width:7px; height:7px; border-radius:50%; background:#ef4444; margin-right:6px; }

/* ── Input labels ── */
.stTextInput label, .stNumberInput label, .stSelectbox label, .stFileUploader label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    filter: brightness(1.1) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.35) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1628 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid #1a2d45 !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    color: #64748b !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
}

/* ── Number inputs ── */
input[type="number"] {
    background: #0a1628 !important;
    border: 1px solid #1a2d45 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Divider ── */
.ns-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1a2d45, transparent);
    margin: 20px 0;
}

/* ── Warning banner ── */
.ns-warning {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px;
    padding: 12px 16px;
    color: #fbbf24;
    font-size: 13px;
}
.ns-info {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    color: #7dd3fc;
    font-size: 13px;
}

/* ── Sidebar nav items ── */
.nav-item {
    padding: 10px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    color: #94a3b8;
    transition: all 0.15s;
}
.nav-item:hover { background: #1a2d45; color: #e2e8f0; }
.nav-item.active { background: rgba(56,189,248,0.12); color: #38bdf8; border-left: 2px solid #38bdf8; }
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    mdls = {}

    # Clinical
    for key, fname in [
        ("clin_model",    "models/clinical_xgb.pkl"),
        ("clin_scaler",   "models/clinical_scaler.pkl"),
        ("clin_features", "models/clinical_features.pkl"),
    ]:
        try:    mdls[key] = joblib.load(fname)
        except: mdls[key] = None

    # Biomarker
    for key, fname in [
        ("bio_model",    "models/biomarker_xgb_model.pkl"),
        ("bio_scaler",   "models/biomarker_scaler.pkl"),
        ("bio_features", "models/biomarker_features.pkl"),
    ]:
        try:    mdls[key] = joblib.load(fname)
        except: mdls[key] = None

    # Genetic
    for key, fname in [
        ("gen_pipeline", "models/gene_xgb_pipeline.pkl"),
        ("gen_genes",    "models/gene_feature_names.pkl"),
    ]:
        try:    mdls[key] = joblib.load(fname)
        except: mdls[key] = None

    # MRI – XGBoost
    try:    mdls["mri_xgb"] = joblib.load("models/mri_xgb_model.pkl")
    except: mdls["mri_xgb"] = None

    # MRI – EfficientNetB0 feature extractor
    mdls["mri_extractor"]  = None
    mdls["mri_load_error"] = ""

    if not TF_AVAILABLE:
        mdls["mri_load_error"] = f"TensorFlow not available: {TF_ERROR}"
    else:
        keras_path = "models/mri_efficientnet_finetuned.keras"
        if not os.path.exists(keras_path):
            mdls["mri_load_error"] = f"File not found: {os.path.abspath(keras_path)}"
        else:
            def _find_gap_layer(base):
                """Find GlobalAveragePooling2D — the 1280-dim bottleneck layer."""
                # Pass 1: search by layer name
                for layer in reversed(base.layers):
                    if "global_average_pooling" in layer.name.lower():
                        return layer
                # Pass 2: search by output shape == (None, 1280)
                for layer in reversed(base.layers):
                    try:
                        s = layer.output_shape
                        if isinstance(s, (list, tuple)) and len(s) == 2 and s[-1] == 1280:
                            return layer
                    except Exception:
                        pass
                # Pass 3: known position in the architecture (GAP is -3 from end)
                return base.layers[-3]

            def _make_extractor(base, KModel=None):
                """Build a sub-model that stops at the 1280-dim GAP layer.
                NEVER falls back to the full model — that would return 4 outputs."""
                if KModel is None:
                    KModel = Model
                gap = _find_gap_layer(base)
                extractor = KModel(inputs=base.input, outputs=gap.output)
                # Verify output shape is (None, 1280) before returning
                out_shape = extractor.output_shape
                if isinstance(out_shape, (list, tuple)) and out_shape[-1] != 1280:
                    raise RuntimeError(
                        f"Extractor output shape is {out_shape} — expected (None, 1280). "
                        f"GAP layer found: '{gap.name}'. "
                        "The wrong layer was selected as the feature bottleneck."
                    )
                return extractor

            errors = []

            # ── Strategy A: standalone keras package (keras>=3, installed separately)
            if mdls["mri_extractor"] is None:
                try:
                    import keras as _keras
                    base = _keras.saving.load_model(keras_path, compile=False)
                    mdls["mri_extractor"] = _make_extractor(base, _keras.Model)
                except Exception as e:
                    errors.append(f"A (keras.saving): {e}")

            # ── Strategy B: tf_keras package (pip install tf_keras)
            if mdls["mri_extractor"] is None:
                try:
                    import tf_keras as _tfk
                    base = _tfk.models.load_model(keras_path, compile=False)
                    mdls["mri_extractor"] = _make_extractor(base, _tfk.Model)
                except Exception as e:
                    errors.append(f"B (tf_keras): {e}")

            # ── Strategy C: plain tf.keras load (works on TF ≤ 2.15 / keras 2)
            if mdls["mri_extractor"] is None:
                try:
                    base = load_model(keras_path, compile=False)
                    mdls["mri_extractor"] = _make_extractor(base)
                except Exception as e:
                    errors.append(f"C (load_model compile=False): {e}")

            # ── Strategy D: safe_mode=False (keras 3 sometimes requires this)
            if mdls["mri_extractor"] is None:
                try:
                    base = load_model(keras_path, compile=False, safe_mode=False)
                    mdls["mri_extractor"] = _make_extractor(base)
                except Exception as e:
                    errors.append(f"D (safe_mode=False): {e}")

            # ── Strategy E: tf.keras.saving alias
            if mdls["mri_extractor"] is None:
                try:
                    base = tf.keras.saving.load_model(keras_path, compile=False)
                    mdls["mri_extractor"] = _make_extractor(base)
                except Exception as e:
                    errors.append(f"E (tf.keras.saving): {e}")

            # ── Strategy F: rebuild architecture + load weights from .keras zip.
            # Explicitly wire the extractor output to the GAP layer (NOT Dense(4)).
            if mdls["mri_extractor"] is None:
                try:
                    import zipfile, tempfile

                    # Prefer tf_keras for rebuild; fall back to tensorflow.keras
                    try:
                        import tf_keras as _tfk2
                        _EfficientNetB0 = _tfk2.applications.EfficientNetB0
                        _GAP            = _tfk2.layers.GlobalAveragePooling2D
                        _Dense          = _tfk2.layers.Dense
                        _Dropout        = _tfk2.layers.Dropout
                        _KModel         = _tfk2.Model
                    except ImportError:
                        from tensorflow.keras.applications import EfficientNetB0 as _EfficientNetB0
                        from tensorflow.keras.layers import (
                            GlobalAveragePooling2D as _GAP,
                            Dense as _Dense,
                            Dropout as _Dropout,
                        )
                        from tensorflow.keras.models import Model as _KModel

                    # Build the full architecture exactly as in the notebook
                    base_net   = _EfficientNetB0(
                        weights=None, include_top=False, input_shape=(224, 224, 3)
                    )
                    gap_output = _GAP()(base_net.output)   # ← 1280-dim bottleneck
                    dropped    = _Dropout(0.3)(gap_output)
                    out        = _Dense(4, activation="softmax")(dropped)
                    full_model = _KModel(inputs=base_net.input, outputs=out)

                    # Load saved weights from the .keras zip archive
                    with zipfile.ZipFile(keras_path, "r") as zf:
                        names  = zf.namelist()
                        w_file = next((n for n in names if n.endswith(".weights.h5")), None)
                        if w_file is None:
                            w_file = next((n for n in names if n.endswith(".h5")), None)
                        if not w_file:
                            raise RuntimeError("No .h5 weights file inside .keras archive")
                        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                            tmp.write(zf.read(w_file))
                            tmp_path = tmp.name
                    full_model.load_weights(tmp_path)
                    os.unlink(tmp_path)

                    # Build extractor that outputs the GAP layer (1280-dim), not Dense(4)
                    mdls["mri_extractor"] = _KModel(
                        inputs=full_model.input,
                        outputs=full_model.get_layer(index=-3).output,  # GAP layer
                    )
                    # Sanity-check the output shape
                    if mdls["mri_extractor"].output_shape[-1] != 1280:
                        raise RuntimeError(
                            f"Rebuilt extractor output shape = "
                            f"{mdls['mri_extractor'].output_shape}, expected (None, 1280)"
                        )
                    errors.append("F (architecture rebuild) succeeded")
                except Exception as e:
                    mdls["mri_extractor"] = None
                    errors.append(f"F (rebuild): {e}")

            if mdls["mri_extractor"] is None:
                mdls["mri_load_error"] = (
                    "All loading strategies failed. "
                    "Fix: pip install tf_keras  OR  pip install keras  "
                    "then restart the app.\n\n"
                    "Details: " + " | ".join(errors)
                )

    return mdls

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS – RISK DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def risk_color(prob):
    if prob >= 0.65:  return "#ef4444"
    if prob >= 0.40:  return "#f59e0b"
    return "#34d399"

def risk_label(prob):
    if prob >= 0.65:  return "High Risk", "result-high"
    if prob >= 0.40:  return "Moderate Risk", "result-moderate"
    return "Low Risk", "result-low"

def render_risk_bar(prob):
    pct   = int(prob * 100)
    color = risk_color(prob)
    st.markdown(f"""
    <div class="risk-bar-container">
      <div class="risk-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    <div style="display:flex; justify-content:space-between; margin-top:4px;">
      <span style="font-size:11px; color:#64748b;">Low</span>
      <span style="font-size:12px; color:{color}; font-weight:700;">{pct}%</span>
      <span style="font-size:11px; color:#64748b;">High</span>
    </div>
    """, unsafe_allow_html=True)

def render_result_card(label, css_cls, prob, model_name, extra=""):
    st.markdown(f"""
    <div class="{css_cls}" style="margin-top:16px;">
      <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; font-family:'JetBrains Mono',monospace;">
        {model_name}
      </div>
      <div style="font-size:26px; font-weight:700; color:#e2e8f0; margin:6px 0 2px;">
        {label}
      </div>
      <div style="font-size:13px; color:#94a3b8;">
        Confidence: <strong style="color:#e2e8f0;">{prob*100:.1f}%</strong>
        {"&nbsp;·&nbsp;" + extra if extra else ""}
      </div>
    </div>
    """, unsafe_allow_html=True)

def model_status_dot(mdls, key):
    ok = mdls.get(key) is not None
    cls = "dot-on" if ok else "dot-off"
    txt = "Loaded" if ok else "Missing"
    return f'<span class="{cls}"></span>{txt}'



# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(mdls):
    with st.sidebar:
        st.markdown("""
        <div style="padding: 16px 0 20px;">
          <div style="font-size:20px; font-weight:700; color:#e2e8f0;">🧠 NeuroScan AI</div>
          <div style="font-size:11px; color:#475569; margin-top:2px;">Alzheimer's Risk Platform</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🔬 Diagnosis"],
            key="nav",
            label_visibility="collapsed",
        )

        st.markdown("<div class='ns-divider'></div>", unsafe_allow_html=True)

        # Model status
        st.markdown("""
        <div style="font-size:10px; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
                    color:#475569; margin-bottom:10px; font-family:'JetBrains Mono',monospace;">
          Model Status
        </div>
        """, unsafe_allow_html=True)

        status_rows = [
            ("Clinical XGB",    "clin_model"),
            ("Biomarker XGB",   "bio_model"),
            ("Genetic Pipeline","gen_pipeline"),
            ("MRI EfficientNet","mri_extractor"),
            ("MRI XGBoost",     "mri_xgb"),
        ]
        for label, key in status_rows:
            dot = model_status_dot(mdls, key)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; font-size:12px;
                        color:#94a3b8; padding:4px 0;">
              <span>{label}</span>
              <span>{dot}</span>
            </div>
            """, unsafe_allow_html=True)

        # Show MRI error detail if it failed to load
        mri_err = mdls.get("mri_load_error", "")
        if mri_err:
            with st.expander("⚠ MRI load error", expanded=True):
                st.markdown("""
                <div style="font-size:12px; color:#fbbf24; font-weight:600; margin-bottom:8px;">
                  Quick fix — run ONE of these in your terminal, then click Reload Models:
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                            background:#050d1a; border-radius:6px; padding:8px 10px;
                            color:#34d399; margin-bottom:8px; line-height:1.8;">
                  pip install tf_keras<br>
                  <span style="color:#64748b;"># or if that doesn't work:</span><br>
                  pip install keras
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="font-size:10px; color:#475569; word-break:break-word;
                            line-height:1.5; font-family:'JetBrains Mono',monospace;
                            margin-top:6px;">
                  {mri_err}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='ns-divider'></div>", unsafe_allow_html=True)

        if st.button("🔄 Reload Models", use_container_width=True, key="reload_models"):
            st.cache_resource.clear()
            st.rerun()

    return page

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────────────────────────────────────
def dashboard_page(mdls):
    st.markdown("""
    <div class="ns-card-accent" style="margin-bottom:24px;">
      <div class="ns-section-title">Overview</div>
      <div class="ns-title">NeuroScan AI</div>
      <div class="ns-subtitle">
        Integrates four independent machine learning models to provide
        a comprehensive Alzheimer's disease risk assessment from clinical scores,
        CSF biomarkers, gene expression, and MRI brain scans.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model cards
    cols = st.columns(4)
    model_info = [
        ("🧪", "Clinical",   "XGBoost",            "14 cognitive & demographic features",           "clin_model"),
        ("🔬", "Biomarker",  "XGBoost",            "CSF Amyloid-β, p-Tau, t-Tau + demographics",   "bio_model"),
        ("🧬", "Genetic",    "SelectKBest+PCA+XGB", "Gene expression (GEO microarray data)",         "gen_pipeline"),
        ("🧠", "MRI Scan",   "EfficientNetB0+XGB",  "4-class dementia severity from brain MRI",      "mri_xgb"),
    ]
    for col, (icon, name, arch, desc, key) in zip(cols, model_info):
        loaded = mdls.get(key) is not None
        dot = "🟢" if loaded else "🔴"
        with col:
            st.markdown(f"""
            <div class="ns-card" style="height:180px; position:relative;">
              <div style="font-size:26px; margin-bottom:8px;">{icon}</div>
              <div style="font-size:14px; font-weight:700; color:#e2e8f0;">{name}</div>
              <div class="metric-pill" style="margin:6px 0;">{arch}</div>
              <div style="font-size:11px; color:#475569; margin-top:6px; line-height:1.45;">{desc}</div>
              <div style="position:absolute; top:16px; right:16px; font-size:11px;">{dot}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ns-info" style="margin-top:24px;">
      Head to <strong>Diagnosis</strong> in the sidebar to run a multi-modal analysis.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSIS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def diagnosis_page(mdls):
    st.markdown("""
    <div class="ns-section-title">Analysis</div>
    <div class="ns-title" style="margin-bottom:4px;">Multi-Modal Diagnosis</div>
    <div class="ns-subtitle">
      Provide data for one or more modalities. Each available model will run independently,
      and a weighted ensemble risk score is computed.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 Clinical", "🔬 Biomarker", "🧬 Genetic", "🧠 MRI Scan"
    ])

    # ── Collect inputs ────────────────────────────────────────────────────────
    gen_file = mri_file = None

    # ═══ TAB 1 — CLINICAL ════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
        <div class="ns-card">
          <div class="ns-section-title">Clinical & Cognitive Features</div>
          <div class="ns-subtitle" style="margin-bottom:0;">
            Upload the clinical CSV file used during training. Required columns:
            <code>NACCAGE, SEX, EDUC, NACCAPOE, CDRGLOB, CDRSUM, NACCMOCA,
            TRAILA, TRAILB, ANIMALS, VEG, MEMUNITS, DIGIF, DIGIB</code>.
            Any missing columns will be imputed with population medians.
          </div>
        </div>
        """, unsafe_allow_html=True)

        loaded = mdls.get("clin_model") is not None
        if not loaded:
            st.markdown('<div class="ns-warning">⚠ Clinical model not loaded — place <code>clinical_xgb.pkl</code> and <code>clinical_scaler.pkl</code> in <code>models/</code></div>', unsafe_allow_html=True)

        clin_file = st.file_uploader(
            "Upload Clinical CSV",
            type=["csv"],
            key="clin_upload",
        )

        CLIN_FEATURES = [
            'NACCAGE', 'SEX', 'EDUC', 'NACCAPOE',
            'CDRGLOB', 'CDRSUM', 'NACCMOCA',
            'TRAILA', 'TRAILB', 'ANIMALS', 'VEG',
            'MEMUNITS', 'DIGIF', 'DIGIB',
        ]

        if clin_file:
            try:
                clin_df_raw = pd.read_csv(clin_file, low_memory=False)
                missing_cols = [c for c in CLIN_FEATURES if c not in clin_df_raw.columns]
                if missing_cols:
                    st.markdown(f'<div class="ns-warning">⚠ Missing columns: <code>{", ".join(missing_cols)}</code></div>', unsafe_allow_html=True)
                else:
                    clin_df = clin_df_raw[CLIN_FEATURES].copy()
                    clin_df = clin_df.fillna(clin_df.median(numeric_only=True))
                    st.session_state["clin_df"] = clin_df
                    st.success(f"✓ Loaded {len(clin_df):,} rows × {len(CLIN_FEATURES)} features from {clin_file.name}")
                    st.dataframe(clin_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")

    # ═══ TAB 2 — BIOMARKER ═══════════════════════════════════════════════════
    with tab2:
        st.markdown("""
        <div class="ns-card">
          <div class="ns-section-title">CSF Biomarker Panel</div>
          <div class="ns-subtitle" style="margin-bottom:0;">
            Upload the biomarker CSV file used during training. Required columns:
            <code>CSFABETA, CSFPTAU, CSFTTAU, NACCAGE, NACCSEX, EDUC, NACCAPOE</code>.
            These are the strongest biochemical predictors of Alzheimer's pathology.
          </div>
        </div>
        """, unsafe_allow_html=True)

        loaded_bio = mdls.get("bio_model") is not None
        if not loaded_bio:
            st.markdown('<div class="ns-warning">⚠ Biomarker model not loaded — place <code>biomarker_xgb_model.pkl</code> and <code>biomarker_scaler.pkl</code> in <code>models/</code></div>', unsafe_allow_html=True)

        bio_file = st.file_uploader(
            "Upload Biomarker CSV",
            type=["csv"],
            key="bio_upload",
        )

        BIO_FEATURES = ['CSFABETA', 'CSFPTAU', 'CSFTTAU', 'NACCAGE', 'NACCSEX', 'EDUC', 'NACCAPOE']

        if bio_file:
            try:
                bio_df_raw = pd.read_csv(bio_file, low_memory=False)
                missing_cols = [c for c in BIO_FEATURES if c not in bio_df_raw.columns]
                if missing_cols:
                    st.markdown(f'<div class="ns-warning">⚠ Missing columns: <code>{", ".join(missing_cols)}</code></div>', unsafe_allow_html=True)
                else:
                    bio_df = bio_df_raw[BIO_FEATURES].copy()
                    bio_df = bio_df.fillna(bio_df.median(numeric_only=True))
                    st.session_state["bio_df"] = bio_df
                    st.success(f"✓ Loaded {len(bio_df):,} rows × {len(BIO_FEATURES)} features from {bio_file.name}")
                    st.dataframe(bio_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")

    # ═══ TAB 3 — GENETIC ═════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
        <div class="ns-card">
          <div class="ns-section-title">Gene Expression</div>
          <div class="ns-subtitle" style="margin-bottom:0;">
            Upload a GEO series matrix (.txt) or a pre-processed gene expression CSV.
            The pipeline (SelectKBest → StandardScaler → PCA → XGBoost) handles
            dimensionality reduction automatically.
          </div>
        </div>
        """, unsafe_allow_html=True)

        loaded_gen = mdls.get("gen_pipeline") is not None
        if not loaded_gen:
            st.markdown('<div class="ns-warning">⚠ Genetic model not loaded — place <code>gene_xgb_pipeline.pkl</code> in <code>models/</code></div>', unsafe_allow_html=True)

        gen_file = st.file_uploader(
            "Upload Gene Expression File (.txt GEO series matrix  or  .csv with gene columns)",
            type=["txt", "csv"],
            key="gen_upload",
        )
        if gen_file:
            st.session_state["gen_file_name"] = gen_file.name
            st.session_state["gen_file_bytes"] = gen_file.read()
            st.success(f"✓ File received: {gen_file.name}")

    # ═══ TAB 4 — MRI ═════════════════════════════════════════════════════════
    with tab4:
        st.markdown("""
        <div class="ns-card">
          <div class="ns-section-title">MRI Brain Scan</div>
          <div class="ns-subtitle" style="margin-bottom:0;">
            Upload an axial MRI slice (JPG / PNG). The EfficientNetB0 feature extractor
            feeds into XGBoost to classify dementia severity into 4 classes.
          </div>
        </div>
        """, unsafe_allow_html=True)

        mri_xgb_ok  = mdls.get("mri_xgb")       is not None
        mri_cnn_ok  = mdls.get("mri_extractor") is not None
        loaded_mri  = mri_xgb_ok and mri_cnn_ok
        mri_err     = mdls.get("mri_load_error", "")

        if not loaded_mri:
            xgb_icon = "✅" if mri_xgb_ok  else "❌"
            cnn_icon = "✅" if mri_cnn_ok  else "❌"
            tf_icon  = "✅" if TF_AVAILABLE else "❌"

            st.markdown(f"""
            <div class="ns-warning">
              <strong>⚠ MRI model not fully loaded.</strong>
              Check the checklist below and fix each ❌ item.
            </div>
            <div style="background:#050d1a; border:1px solid #1a2d45; border-radius:10px;
                        padding:16px 20px; margin-top:12px; font-size:13px; line-height:2;">
              <div>{tf_icon} &nbsp;<strong style="color:#94a3b8;">TensorFlow installed</strong>
                {"" if TF_AVAILABLE else f'<span style="color:#f87171; margin-left:8px;">— run: <code>pip install tensorflow</code></span>'}
              </div>
              <div>{xgb_icon} &nbsp;<strong style="color:#94a3b8;"><code>models/mri_xgb_model.pkl</code></strong>
                {"" if mri_xgb_ok else '<span style="color:#f87171; margin-left:8px;">— run the last cell of MRI-CNN.ipynb to generate this file</span>'}
              </div>
              <div>{cnn_icon} &nbsp;<strong style="color:#94a3b8;"><code>models/mri_efficientnet_finetuned.keras</code></strong>
                {"" if mri_cnn_ok else '<span style="color:#f87171; margin-left:8px;">— run the last cell of MRI-CNN.ipynb to generate this file</span>'}
              </div>
              {f'<div style="margin-top:10px; color:#f87171; font-family:monospace; font-size:11px; word-break:break-all;"><strong>Error:</strong> {mri_err}</div>' if mri_err else ""}
            </div>
            """, unsafe_allow_html=True)

        mri_file = st.file_uploader("Upload MRI Image (JPG / PNG)", type=["jpg","jpeg","png"], key="mri_upload")

        if mri_file:
            col_img, col_info = st.columns([1, 2])
            img_bytes = mri_file.read()
            with col_img:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                st.image(img, caption="Uploaded MRI", use_column_width=True)
            with col_info:
                w, h = img.size
                st.markdown(f"""
                <div class="ns-metric" style="text-align:left; margin-bottom:10px;">
                  <div class="lbl">File</div>
                  <div style="font-size:14px; color:#e2e8f0; font-weight:600; margin-top:4px;">
                    {mri_file.name}
                  </div>
                </div>
                <div class="ns-metric" style="text-align:left;">
                  <div class="lbl">Original Resolution</div>
                  <div style="font-size:14px; color:#e2e8f0; font-weight:600; margin-top:4px;">
                    {w} × {h} px → resized to 224 × 224
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.session_state["mri_bytes"] = img_bytes

    # ═══ RUN FULL DIAGNOSIS ═══════════════════════════════════════════════════
    st.markdown("<div class='ns-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:12px;">
      <div class="ns-section-title">Ensemble Prediction</div>
      <div class="ns-subtitle" style="margin-bottom:0;">
        Click below to run all available models and compute a weighted risk score.
        At least one modality must be provided.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶  Run Full Diagnosis", key="run_diag", use_container_width=False):
        _run_diagnosis(mdls)


def _run_diagnosis(mdls):
    probabilities, weights, modalities, details = [], [], [], []

    with st.spinner("Running models…"):
        time.sleep(0.3)  # brief pause so spinner is visible

        # ─── CLINICAL ─────────────────────────────────────────────────────
        clin_df = st.session_state.get("clin_df")
        if clin_df is not None and mdls.get("clin_model") and mdls.get("clin_scaler"):
            try:
                feats = [
                    'NACCAGE','SEX','EDUC','NACCAPOE',
                    'CDRGLOB','CDRSUM','NACCMOCA',
                    'TRAILA','TRAILB','ANIMALS','VEG',
                    'MEMUNITS','DIGIF','DIGIB'
                ]
                X = clin_df[feats].values
                X_scaled = mdls["clin_scaler"].transform(X)
                proba_all = mdls["clin_model"].predict_proba(X_scaled)
                p = float(proba_all[:, 1].mean())
                probabilities.append(p); weights.append(0.35); modalities.append("Clinical")
                details.append(("🧪 Clinical Model", p,
                                 f"Non-AD vs Alzheimer's · {len(clin_df):,} rows"))
            except Exception as e:
                st.warning(f"Clinical model error: {e}")

        # ─── BIOMARKER ────────────────────────────────────────────────────
        bio_df = st.session_state.get("bio_df")
        if bio_df is not None and mdls.get("bio_model") and mdls.get("bio_scaler"):
            try:
                feats = ['CSFABETA','CSFPTAU','CSFTTAU','NACCAGE','NACCSEX','EDUC','NACCAPOE']
                X = bio_df[feats].values
                X_scaled = mdls["bio_scaler"].transform(X)
                proba_all = mdls["bio_model"].predict_proba(X_scaled)
                p = float(proba_all[:, 1].mean())
                probabilities.append(p); weights.append(0.25); modalities.append("Biomarker")
                details.append(("🔬 CSF Biomarker Model", p,
                                 f"Normal vs Alzheimer's · {len(bio_df):,} rows"))
            except Exception as e:
                st.warning(f"Biomarker model error: {e}")

        # ─── GENETIC ──────────────────────────────────────────────────────
        gen_bytes = st.session_state.get("gen_file_bytes")
        gen_name  = st.session_state.get("gen_file_name", "")
        if gen_bytes and mdls.get("gen_pipeline"):
            try:
                if gen_name.endswith(".txt"):
                    df = pd.read_csv(io.BytesIO(gen_bytes), sep="\t", comment="!")
                    if "ID_REF" in df.columns:
                        df = df.set_index("ID_REF").T
                else:
                    df = pd.read_csv(io.BytesIO(gen_bytes))
                    for drop_col in ["Diagnosis","label","target"]:
                        if drop_col in df.columns:
                            df = df.drop(columns=[drop_col])

                gene_names = mdls.get("gen_genes")
                if gene_names:
                    for g in gene_names:
                        if g not in df.columns:
                            df[g] = 0.0
                    df = df[gene_names]

                df = df.astype(float)
                proba = mdls["gen_pipeline"].predict_proba(df.head(1))[0]
                p = float(proba[1])
                probabilities.append(p); weights.append(0.10); modalities.append("Genetic")
                details.append(("🧬 Gene Expression Model", p, "Control vs Alzheimer's"))
            except Exception as e:
                st.warning(f"Genetic model error: {e}")

        # ─── MRI ──────────────────────────────────────────────────────────
        MRI_CLASSES = ["NonDemented","VeryMildDemented","MildDemented","ModerateDemented"]
        mri_bytes = st.session_state.get("mri_bytes")
        print("[v0] ═══════════════════════════════════════════════════════")
        print(f"[v0] MRI check: mri_bytes={mri_bytes is not None}, extractor={mdls.get('mri_extractor') is not None}, xgb={mdls.get('mri_xgb') is not None}")
        if mri_bytes and mdls.get("mri_extractor") and mdls.get("mri_xgb"):
            try:
                print("[v0] Starting MRI prediction pipeline...")
                img = Image.open(io.BytesIO(mri_bytes)).convert("RGB").resize((224, 224))
                arr = np.array(img, dtype=np.float32)
                arr = np.expand_dims(arr, 0)
                arr = preprocess_input(arr)
                feats = mdls["mri_extractor"].predict(arr, verbose=0)
                print(f"[v0] Features extracted: shape={feats.shape}")
                # Guard: extractor must output 1280-dim features, not 4-class softmax
                if feats.shape[-1] != 1280:
                    raise RuntimeError(
                        f"Feature shape mismatch — extractor returned {feats.shape[-1]} features "
                        f"but XGBoost expects 1280. The model file may need to be re-saved, "
                        f"or click 'Reload Models' and try again."
                    )
                proba_raw = mdls["mri_xgb"].predict_proba(feats)[0]
                print(f"[v0] Raw probabilities from XGBoost: {proba_raw}")

                # ── Class alignment ────────────────────────────────────────────
                # XGBoost classes_ are int32 0-3, assigned alphabetically by
                # flow_from_directory:
                #   0 → MildDemented   1 → ModerateDemented
                #   2 → NonDemented    3 → VeryMildDemented
                #
                # CRITICAL FIX: numpy int32 silently breaks Python `in` set/list
                # comparisons. Sort by int(class_value) for a reliable positional
                # mapping instead of any alias-matching approach.
                print(f"[v0] XGBoost classes_: {mdls['mri_xgb'].classes_}")
                sorted_pairs = sorted(
                    zip([int(c) for c in mdls["mri_xgb"].classes_], proba_raw),
                    key=lambda x: x[0]
                )
                print(f"[v0] Sorted class pairs: {sorted_pairs}")
                # proba[i] now reliably matches alphabetical class index i
                proba = np.array([p for _, p in sorted_pairs], dtype=np.float32)
                print(f"[v0] Aligned probabilities: {proba}")

                # Severity weights per alphabetical class:
                #   idx0 MildDemented(0.6)  idx1 ModerateDemented(1.0)
                #   idx2 NonDemented(0.0)   idx3 VeryMildDemented(0.3)
                SEVERITY = [
                    ("Mild Demented",      0.6),
                    ("Moderate Demented",  1.0),
                    ("Non Demented",       0.0),
                    ("Very Mild Demented", 0.3),
                ]
                SEVERITY_WEIGHTS = np.array([w for _, w in SEVERITY])

                print("[v0] ───────────────────────────────────────────────────")
                print("[v0] CLASS PROBABILITIES:")
                for idx, (class_name, weight) in enumerate(SEVERITY):
                    print(f"[v0]   [{idx}] {class_name:20s} | Prob: {proba[idx]:.6f} | Weight: {weight:.1f}")
                
                pred_idx = int(np.argmax(proba))
                p_ad = float(np.dot(proba, SEVERITY_WEIGHTS))
                p_ad = min(p_ad, 1.0)
                pred_class = SEVERITY[pred_idx][0]
                pred_conf = float(proba[pred_idx])
                
                print("[v0] ───────────────────────────────────────────────────")
                print(f"[v0] PREDICTION: {pred_class}")
                print(f"[v0]   - Predicted class index: {pred_idx}")
                print(f"[v0]   - Confidence: {pred_conf:.6f} ({pred_conf*100:.1f}%)")
                print(f"[v0]   - Final AD risk (weighted): {p_ad:.6f}")
                print("[v0] ═══════════════════════════════════════════════════════")
                
                probabilities.append(p_ad); weights.append(0.40); modalities.append("MRI")
                details.append(("🧠 MRI Model", p_ad,
                                 f"Class: **{pred_class}**  (confidence {pred_conf*100:.1f}%)"))
            except Exception as e:
                print(f"[v0] MRI MODEL ERROR: {e}")
                import traceback
                print(f"[v0] Traceback: {traceback.format_exc()}")
                st.warning(f"MRI model error: {e}")
        else:
            print("[v0] MRI pipeline skipped (missing components)")

    # ═══ RESULTS ═════════════════════════════════════════════════════════════
    if not probabilities:
        st.markdown('<div class="ns-warning">⚠ No modality data detected. Submit at least one form or upload a file first.</div>', unsafe_allow_html=True)
        return

    final = float(np.average(probabilities, weights=weights))
    rlabel, rcls = risk_label(final)
    rcolor = risk_color(final)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Main risk card
    st.markdown(f"""
    <div class="{rcls}">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
        <div>
          <div style="font-size:11px; color:#94a3b8; text-transform:uppercase;
                      letter-spacing:.1em; font-family:'JetBrains Mono',monospace; margin-bottom:6px;">
            Ensemble Alzheimer's Risk Score
          </div>
          <div style="font-size:48px; font-weight:800; color:{rcolor}; line-height:1;">
            {final*100:.1f}%
          </div>
          <div style="font-size:16px; font-weight:600; color:{rcolor}; margin-top:4px;">
            {rlabel}
          </div>
        </div>
        <div style="text-align:right;">
          <div class="metric-pill">
            {len(modalities)} modali{'ty' if len(modalities)==1 else 'ties'}
          </div>
          <div style="font-size:11px; color:#64748b; margin-top:8px;">
            {" · ".join(modalities)}
          </div>
        </div>
      </div>
      <div style="margin-top:20px;">
    """, unsafe_allow_html=True)
    render_risk_bar(final)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Per-model breakdown
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="ns-section-title" style="margin-bottom:12px;">Per-Model Breakdown</div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(details))
    for col, (mname, prob, note) in zip(cols, details):
        color = risk_color(prob)
        with col:
            st.markdown(f"""
            <div class="ns-metric">
              <div class="val" style="color:{color};">{prob*100:.0f}%</div>
              <div class="lbl">{mname}</div>
              <div style="font-size:10px; color:#475569; margin-top:6px; line-height:1.3;">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="margin-top:20px; padding:12px 16px; background:#050d1a; border-radius:8px;
                font-size:11px; color:#475569; line-height:1.6; border:1px solid #0d1f35;">
      ⚠ <strong style="color:#64748b;">Medical Disclaimer:</strong> This tool is for
      research and educational purposes only. Results do not constitute medical advice,
      diagnosis, or treatment. Always consult a qualified neurologist for clinical decisions.
    </div>
    """, unsafe_allow_html=True)

    # Clear cached inputs so next run is fresh
    for k in ["clin_df", "bio_df", "gen_file_bytes", "gen_file_name", "mri_bytes"]:
        st.session_state.pop(k, None)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    mdls = load_models()
    page = render_sidebar(mdls)

    if page == "🏠 Dashboard":
        dashboard_page(mdls)
    elif page == "🔬 Diagnosis":
        diagnosis_page(mdls)


if __name__ == "__main__":
    main()
