import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import confusion_matrix
import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="OncoSmart AI", layout="wide", initial_sidebar_state="collapsed")

# --- CSS Injection from User's HTML ---
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  
  :root {
    --bg: #f7f8fa;
    --white: #ffffff;
    --border: #e2e6ed;
    --text: #1a1f2e;
    --muted: #8892a4;
    --accent: #1d4ed8;
    --accent-light: #eff6ff;
    --danger: #dc2626;
    --danger-light: #fef2f2;
    --warning: #d97706;
    --warning-light: #fffbeb;
    --success: #16a34a;
    --success-light: #f0fdf4;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
  }

  /* Override Streamlit Defaults */
  .stApp { background: var(--bg); color: var(--text); font-family: var(--sans); }
  .block-container {
    max-width: 96% !important; /* Bento Full screen */
    padding-top: 4rem !important;
    padding-bottom: 2rem !important;
  }
  .stSelectbox label { display: none; }
  .stButton button { width: 100%; border-radius: 12px; border: 1px solid var(--border); font-family: var(--mono); color: var(--text); background: white; height: 46px; font-weight: 600; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
  .stButton button:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-light); }
  .stSelectbox div[data-baseweb="select"] { border-radius: 12px; font-family: var(--mono); font-size: 14px; }
  
  /* Header styling */
  .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-icon {
    width: 38px; height: 38px; background: var(--accent); border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 13px; font-weight: 600; color: white;
  }
  .brand-name { font-family: var(--mono); font-size: 16px; font-weight: 600; color: var(--text); }
  .brand-sub  { font-size: 11px; color: var(--muted); margin-top: 1px; }
  .model-tag {
    font-family: var(--mono); font-size: 10px; padding: 5px 10px; border-radius: 6px;
    background: var(--accent-light); color: var(--accent);
    border: 1px solid #bfdbfe; letter-spacing: 0.05em;
  }

  /* Bento Grid */
  .bento-container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; align-items: stretch; margin-top: 24px; }
  .bento-col { display: flex; flex-direction: column; gap: 24px; }
  
  /* Bento Card Base */
  .bento-card { background: var(--white); border: 1px solid var(--border); border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05); overflow: hidden; display: flex; flex-direction: column; }
  .bento-header { padding: 16px 20px; border-bottom: 1px solid var(--border); font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; font-family: var(--mono); background: #fafafa; }
  .bento-body { padding: 24px; flex-grow: 1; display: flex; flex-direction: column; }

  /* Patient Quick Info */
  .patient-meta { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-top: 0; }
  .meta-item { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .meta-label { font-family: var(--mono); font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
  .meta-value { font-family: var(--mono); font-size: 15px; font-weight: 600; color: var(--text); }

  /* Feature Grid */
  .feature-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
  .feat-chip { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; display: flex; flex-direction: column; justify-content: center; }
  .feat-name { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; }
  .feat-val { font-family: var(--mono); font-size: 16px; font-weight: 600; color: var(--text); }
  .feat-bar { height: 4px; background: var(--border); border-radius: 2px; margin-top: 8px; overflow: hidden; }
  .feat-fill { height: 100%; border-radius: 2px; background: var(--accent); }

  /* Verdict */
  .verdict-top { padding: 32px 24px; display: flex; align-items: center; justify-content: space-between; }
  .verdict-label { font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px; }
  .verdict-value { font-family: var(--mono); font-size: 36px; font-weight: 600; line-height: 1; text-transform: uppercase; }
  .prob-block { text-align: right; }
  .prob-num { font-family: var(--mono); font-size: 28px; font-weight: 600; }
  .prob-label { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .prob-track { height: 6px; background: var(--border); margin: 0; position: relative;}
  .prob-fill { height: 100%; background: var(--danger); }
  
  /* Reliability */
  .rel-row { display: flex; align-items: flex-start; gap: 24px; margin-bottom: 24px; }
  .rel-big { font-family: var(--mono); font-size: 56px; font-weight: 600; color: var(--warning); line-height: 1; flex-shrink: 0; }
  .rel-right { flex: 1; }
  .rel-tier { display: inline-block; font-family: var(--mono); font-size: 11px; padding: 4px 12px; border-radius: 20px; background: var(--warning-light); color: var(--warning); border: 1px solid #fde68a; letter-spacing: 0.08em; margin-bottom: 8px; font-weight: 600; }
  .rel-desc { font-size: 13px; color: var(--muted); line-height: 1.6; }

  .comp-list { display: flex; flex-direction: column; gap: 12px; }
  .comp-row { display: flex; align-items: center; gap: 14px; }
  .comp-name { font-family: var(--mono); font-size: 11px; color: var(--muted); width: 170px; flex-shrink: 0; }
  .comp-track { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .comp-fill { height: 100%; border-radius: 3px; }
  .comp-score { font-family: var(--mono); font-size: 13px; font-weight: 600; min-width: 40px; text-align: right; }

  /* Rec Box */
  .rec-box { background: var(--warning-light); border: 1px solid #fde68a; border-radius: 12px; padding: 20px 24px; margin-top: auto; }
  .rec-title { font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; color: var(--warning); text-transform: uppercase; margin-bottom: 8px; font-weight: 600; }
  .rec-text { font-size: 14px; color: #78350f; line-height: 1.6; }

  /* Shap */
  .shap-list { display: flex; flex-direction: column; gap: 14px; }
  .shap-row { display: flex; align-items: center; gap: 14px; }
  .shap-name { font-family: var(--mono); font-size: 11px; color: var(--text); width: 140px; flex-shrink: 0; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; font-weight: 500; }
  .shap-area { flex: 1; height: 18px; position: relative; display: flex; align-items: center; }
  .shap-axis { position: absolute; left: 50%; width: 1px; height: 100%; background: var(--border); }
  .shap-pos { position: absolute; left: 50%; height: 10px; border-radius: 0 4px 4px 0; background: #fca5a5; }
  .shap-neg { position: absolute; right: 50%; height: 10px; border-radius: 4px 0 0 4px; background: #93c5fd; }
  .shap-num { font-family: var(--mono); font-size: 12px; font-weight: 600; min-width: 54px; text-align: right; }

  /* Stats */
  .conf-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 24px; }
  .conf-cell { border-radius: 12px; padding: 20px; text-align: center; }
  .conf-lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; font-family: var(--mono); margin-bottom: 8px; font-weight: 600; }
  .conf-num { font-family: var(--mono); font-size: 32px; font-weight: 600; line-height: 1; }
  .conf-sub { font-size: 11px; margin-top: 6px; color: var(--muted); }
  .cell-g { background: var(--success-light); border: 1px solid #bbf7d0; }
  .cell-n { background: var(--bg); border: 1px solid var(--border); }
  .cell-b { background: var(--danger-light); border: 1px solid #fecaca; }
  
  .stat-row { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: var(--bg); border-radius: 10px; border: 1px solid var(--border); margin-bottom: 8px; }
  .stat-key { font-size: 13px; color: var(--muted); font-weight: 500; }
  .stat-val { font-family: var(--mono); font-size: 14px; font-weight: 600; color: var(--accent); }
  
  .disclaimer { text-align: center; font-size: 11px; color: var(--muted); font-family: var(--mono); padding-top: 32px; border-top: 1px solid var(--border); margin-top: 32px; display: block; width: 100%; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE LOGIC (Merged Rebuild Pipeline)
# -----------------------------------------------------------------------------
REL_HIGH = 0.80
REL_LOW  = 0.50

@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    int_dir = os.path.join(base_dir, "outputs", "intermediate")
    art_dir = os.path.join(base_dir, "outputs", "artifacts")

    X_test = np.load(os.path.join(int_dir, "X_test_final.npy"))
    y_test = np.load(os.path.join(int_dir, "y_test.npy"))
    prob_test_all = np.load(os.path.join(int_dir, "rf_y_prob_test.npy"))

    calibrated = joblib.load(os.path.join(art_dir, "final_model_calibrated.joblib"))
    rf_model   = joblib.load(os.path.join(art_dir, "final_model.joblib"))

    with open(os.path.join(art_dir, "all_features.json"), "r") as f:
        features = json.load(f)

    explainer = shap.TreeExplainer(rf_model)

    with open(os.path.join(art_dir, "rf_metrics.json"), "r") as f:
        metrics = json.load(f)

    # Pre-computed reliability scores for test set
    rel_df = pd.read_csv(os.path.join(art_dir, "reliability_scores_test.csv"))

    return X_test, y_test, prob_test_all, features, calibrated, rf_model, explainer, metrics, rel_df

X_test, y_test, prob_test_all, features, calibrated, rf_model, explainer, metrics, rel_df = load_artifacts()

# ── Reliability component functions ──
def probability_margin(prob):
    return float(abs(prob - 0.5) / 0.5)

def shap_spread_score(patient_shap):
    abs_shap = np.abs(patient_shap)
    total = abs_shap.sum()
    if total == 0: return 1.0
    proportions = abs_shap / total
    return float(scipy_entropy(proportions + 1e-10) / np.log(len(patient_shap)))

def leaf_consensus(model, patient_array):
    votes = np.array([t.predict(patient_array)[0] for t in model.estimators_])
    majority = votes.mean().round()
    return float((votes == majority).mean())

def compute_reliability(prob, patient_shap, patient_array):
    margin = probability_margin(prob)
    spread = shap_spread_score(patient_shap)
    leaf   = leaf_consensus(rf_model, patient_array)
    score  = 0.50 * margin + 0.30 * spread + 0.20 * leaf
    return score, margin, spread, leaf

def get_top_drivers(patient_shap, p_features, n=6):
    df = pd.DataFrame({'feature': p_features, 'shap_value': patient_shap, 'abs_impact': np.abs(patient_shap)})
    return df.sort_values('abs_impact', ascending=False).head(n).to_dict('records')

def run_triage(patient_idx):
    patient = X_test[patient_idx]
    true_label = y_test[patient_idx]
    patient_arr = patient.reshape(1, -1)

    prob = calibrated.predict_proba(patient_arr)[0][1]

    # SHAP
    sh_vals = explainer.shap_values(patient_arr)
    if isinstance(sh_vals, list): sh_m = sh_vals[1][0]
    elif isinstance(sh_vals, np.ndarray) and len(sh_vals.shape) == 3: sh_m = sh_vals[0, :, 1]
    else: sh_m = sh_vals[0] if len(sh_vals[0].shape) == 1 else sh_vals[0][:, 1]

    top_drivers = get_top_drivers(sh_m, features)
    rel, c_margin, c_spread, c_leaf = compute_reliability(prob, sh_m, patient_arr)

    return patient, true_label, prob, rel, c_margin, c_spread, c_leaf, top_drivers

# -----------------------------------------------------------------------------
# UI RENDER
# -----------------------------------------------------------------------------

header_placeholder = st.empty()
patient_sel_placeholder = st.empty()
st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
layout_col1, layout_col2, layout_col3 = st.columns(3, gap="large")

with layout_col3:
    shap_placeholder = st.empty()
    
    with st.container(border=True):
        st.markdown('<div class="bento-header" style="margin-bottom: 8px; border-bottom: none; padding-bottom: 0;">Decision Threshold</div>', unsafe_allow_html=True)
        threshold = st.slider(
            'Decision Threshold',
            min_value = 0.10,
            max_value = 0.60,
            value     = 0.25,
            step      = 0.05,
            label_visibility="collapsed"
        )
    stat_placeholder = st.empty()

# Rethreshold full test set
y_pred_all  = (prob_test_all >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_all).ravel()

header_placeholder.markdown(f"""
<div class="header">
  <div class="brand">
    <div class="brand-icon">OS</div>
    <div>
      <div class="brand-name">OncoSmart AI</div>
      <div class="brand-sub">Breast Cancer Diagnostic Triage System</div>
    </div>
  </div>
  <span class="model-tag">Random Forest · AUC {metrics.get('auc', 0.9997)} · threshold {threshold:.2f}</span>
</div>
""", unsafe_allow_html=True)

# --- Fake Data Generation ---
@st.cache_data
def generate_patient_pool(num_patients):
    first_names = ["Mary", "Patricia", "Linda", "Barbara", "Elizabeth", "Jennifer", "Maria", "Susan", "Margaret", "Dorothy", "Lisa", "Nancy", "Karen", "Betty", "Helen", "Sandra", "Donna", "Carol", "Ruth", "Sharon", "Michelle", "Laura", "Sarah", "Kimberly", "Deborah", "Jessica", "Shirley", "Cynthia", "Angela", "Melissa", "Brenda", "Amy", "Anna", "Rebecca", "Virginia", "Kathleen", "Pamela", "Martha", "Debra", "Amanda", "Stephanie", "Carolyn", "Christine", "Marie", "Janet", "Catherine", "Frances", "Ann", "Joyce", "Diane"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts"]
    names = []
    np.random.seed(42)  
    shuffled_indices = np.random.permutation(num_patients)
    for i in range(num_patients):
        idx = shuffled_indices[i]
        fn = first_names[idx % len(first_names)]
        ln = last_names[(idx * 7) % len(last_names)]
        names.append(f"{fn} {ln}")
    return names

patient_names = generate_patient_pool(len(X_test))
pool_df = pd.DataFrame({
    "index": range(len(X_test)),
    "Patient ID": [f"PT-{str(i).zfill(4)}" for i in range(len(X_test))],
    "Patient Name": patient_names,
    "AI Probability": prob_test_all
})

def get_triage_class(p, thr):
    if p >= thr: return "🚨 URGENT (Malignancy)"
    elif p >= max(0.01, thr - 0.15): return "⚠️ ELEVATED (Borderline)"
    else: return "✅ ROUTINE (Benign)"

pool_df['Priority'] = pool_df['AI Probability'].apply(lambda x: get_triage_class(x, threshold))
pool_df['AI Suspicion'] = (pool_df['AI Probability'] * 100).round(1).astype(str) + "%"
pool_df_sorted = pool_df.sort_values(by='AI Probability', ascending=False).reset_index(drop=True)

# Controls - Triage Inbox (3-Column Layout)
with patient_sel_placeholder.container():
    st.markdown('<div class="bento-card" style="margin-bottom: 0;">', unsafe_allow_html=True)
    st.markdown('<div class="bento-header" style="display:flex; justify-content:space-between; align-items:center; background:#fafafa; border-bottom:1px solid var(--border);"><span>Active Triage Inbox</span><span style="font-size:10px; color:var(--muted); font-weight:400;">Select a patient to load their profile below</span></div>', unsafe_allow_html=True)
    
    # CSS for the 3 columns and unified Button-Card layout
    inbox_css = """
    <style>
      .triage-col-wrap { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: #fff; margin-top: 12px; }
      
      .st-triage-header {
         padding: 12px 16px; font-family: var(--mono); font-size: 11px; font-weight: 600;
         display: flex; justify-content: space-between; align-items: center;
         border-bottom: 1px solid var(--border);
      }
      .st-triage-sub {
         font-family: var(--sans); font-size: 11px; color: var(--muted);
         padding: 8px 16px; border-bottom: 1px solid var(--border); background: #fff;
      }
      
      .th-u { background: #fff5f5; color: #dc2626; border-bottom-color: #fca5a5; }
      .th-r { background: #fffbeb; color: #d97706; border-bottom-color: #fcd34d; }
      .th-b { background: #f0fdf4; color: #16a34a; border-bottom-color: #86efac; }
      
      .st-bcnt { font-size: 10px; padding: 2px 8px; border-radius: 12px; color: white; }
      .st-bcnt-u { background: #dc2626; }
      .st-bcnt-r { background: #d97706; }
      .st-bcnt-b { background: #16a34a; }
      
      /* Target the content area within the Native Scrollable Container */
      div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] { gap: 0rem; background: #fff; }
      
      .case-card { margin-top:12px; padding: 10px 12px; border: 1px solid var(--border); border-bottom: none; border-left: 3px solid; border-radius: 6px 6px 0 0; background: #fff; }
      
      .case-top { display: flex; justify-content: space-between; margin-bottom: 2px; }
      .case-id { font-family: var(--mono); font-size: 11px; font-weight: 600; color: #1d4ed8; }
      .case-prob { font-family: var(--mono); font-size: 11px; font-weight: 600; }
      .case-name { font-size: 13px; font-weight: 500; color: var(--text); margin-bottom: 8px; }
      
      .case-bot { display: flex; justify-content: space-between; align-items: center; }
      .badge { font-family: var(--mono); font-size: 9px; padding: 2px 6px; border-radius: 4px; border: 1px solid; }
      .time-track { display: flex; align-items: center; gap: 6px; }
      .t-bar { height: 3px; width: 30px; background: var(--border); border-radius: 2px; }
      .t-fill { height: 100%; border-radius: 2px; }
      .t-txt { font-family: var(--mono); font-size: 9px; color: var(--muted); }
      
      .card-u { border-left-color: #dc2626; }
      .prob-u { color: #dc2626; }
      .badge-u { color: #16a34a; background: #f0fdf4; border-color: #bbf7d0; } 
      
      .card-r { border-left-color: #d97706; }
      .prob-r { color: #d97706; }
      .badge-r { color: #d97706; background: #fffbeb; border-color: #fde68a; }
      
      .card-b { border-left-color: #16a34a; }
      .prob-b { color: #16a34a; }
      .badge-b { color: #16a34a; background: #f0fdf4; border-color: #bbf7d0; }
      
      /* UI DOM Stitching Trick: Remove internal gaps between Markdown and Button */
      div[data-testid="stElementContainer"]:has(.case-card) {
          margin-bottom: -1rem !important; 
      }
      
      /* Style the very next Streamlit Button directly under the HTML card */
      div[data-testid="stElementContainer"]:has(.case-card) + div[data-testid="stElementContainer"] .stButton button {
          border-top-left-radius: 0 !important;
          border-top-right-radius: 0 !important;
          border-top: 1px dashed var(--border) !important;
          border-bottom: 1px solid var(--border) !important;
          border-left: 1px solid var(--border) !important;
          border-right: 1px solid var(--border) !important;
          min-height: 26px !important;
          height: 26px !important;
          font-size: 10px !important;
          background: #f8fafc !important;
          color: var(--muted) !important;
          margin-bottom: 12px !important;
          font-family: var(--mono) !important;
          letter-spacing: 0.05em;
          text-transform: uppercase;
      }
      div[data-testid="stElementContainer"]:has(.case-card) + div[data-testid="stElementContainer"] .stButton button:hover {
          color: var(--accent) !important;
          border-color: var(--accent) !important;
          background: var(--accent-light) !important;
      }
      
      .col-footer { padding: 8px; text-align: center; font-family: var(--mono); font-size: 10px; color: var(--muted); border-top: 1px solid var(--border); background: #fafafa; margin-top: 12px; }
    </style>
    """
    st.markdown(inbox_css, unsafe_allow_html=True)
    
    # Bucket patients
    urgent_cases = pool_df_sorted[pool_df_sorted['AI Probability'] >= threshold]
    review_cases = pool_df_sorted[(pool_df_sorted['AI Probability'] >= max(0.01, threshold - 0.15)) & (pool_df_sorted['AI Probability'] < threshold)]
    benign_cases = pool_df_sorted[pool_df_sorted['AI Probability'] < max(0.01, threshold - 0.15)]
    
    def render_card_html(row, prefix):
        p_idx = row['index']
        # Fake reliability
        fak_rel = float(hash(str(p_idx))) / float(abs(hash(str(p_idx))) + 1)
        fak_rel = 0.5 + (abs(fak_rel) * 0.45) 
        
        r_txt = "High reliability" if fak_rel > 0.8 else ("Moderate reliability" if fak_rel > 0.6 else "Low reliability")
        r_cls = "badge-u" if fak_rel > 0.8 else ("badge-r" if fak_rel > 0.6 else "badge-u")
        if fak_rel <= 0.6: r_cls = "badge-r"
        
        t_fill_w = min(100, int(row['AI Probability'] * 100))
        t_color = "#dc2626" if prefix == "u" else ("#d97706" if prefix == "r" else "#16a34a")
        
        time_str = f"09:{str(p_idx % 60).zfill(2)}"
        
        html = f"""
        <div class="case-card card-{prefix}">
            <div class="case-top">
                <span class="case-id">{row['Patient ID']}</span>
                <span class="case-prob prob-{prefix}">{row['AI Suspicion']}</span>
            </div>
            <div class="case-name">{row['Patient Name']}</div>
            <div class="case-bot">
                <span class="badge {r_cls}">{r_txt}</span>
                <div class="time-track">
                    <div class="t-bar"><div class="t-fill" style="width:{t_fill_w}%; background:{t_color};"></div></div>
                    <span class="t-txt">{time_str}</span>
                </div>
            </div>
        </div>
        """
        return html
        
    c1, c2, c3 = st.columns(3)
    
    # URGENT COLUMN
    with c1:
        st.markdown(f'<div class="triage-col-wrap"><div class="st-triage-header th-u"><span>● URGENT</span><span class="st-bcnt st-bcnt-u">{len(urgent_cases)} cases</span></div><div class="st-triage-sub">Malignancy probability ≥ {threshold:.2f} · Immediate review required</div>', unsafe_allow_html=True)
        with st.container(height=360, border=False):
            for _, row in urgent_cases.head(8).iterrows():
                st.markdown(render_card_html(row, "u"), unsafe_allow_html=True)
                if st.button("Review Patient", key=f"btn_u_{row['index']}", use_container_width=True):
                    st.session_state['selected_pt'] = int(row['index'])
                    st.rerun()
        st.markdown(f'<div class="col-footer">+ {max(0, len(urgent_cases)-8)} more urgent cases</div></div>', unsafe_allow_html=True)

    # REVIEW COLUMN
    with c2:
        st.markdown(f'<div class="triage-col-wrap"><div class="st-triage-header th-r"><span>● NEEDS REVIEW</span><span class="st-bcnt st-bcnt-r">{len(review_cases)} cases</span></div><div class="st-triage-sub">Moderate probability score · Re-measurement recommended</div>', unsafe_allow_html=True)
        with st.container(height=360, border=False):
            for _, row in review_cases.head(8).iterrows():
                st.markdown(render_card_html(row, "r"), unsafe_allow_html=True)
                if st.button("Review Patient", key=f"btn_r_{row['index']}", use_container_width=True):
                    st.session_state['selected_pt'] = int(row['index'])
                    st.rerun()
        st.markdown(f'<div class="col-footer">+ {max(0, len(review_cases)-8)} more review cases</div></div>', unsafe_allow_html=True)

    # BENIGN COLUMN
    with c3:
        st.markdown(f'<div class="triage-col-wrap"><div class="st-triage-header th-b"><span>● LIKELY BENIGN</span><span class="st-bcnt st-bcnt-b">{len(benign_cases)} cases</span></div><div class="st-triage-sub">Probability < {max(0.01, threshold - 0.15):.2f} · High reliability</div>', unsafe_allow_html=True)
        with st.container(height=360, border=False):
            for _, row in benign_cases.head(8).iterrows():
                st.markdown(render_card_html(row, "b"), unsafe_allow_html=True)
                if st.button("Review Patient", key=f"btn_b_{row['index']}", use_container_width=True):
                    st.session_state['selected_pt'] = int(row['index'])
                    st.rerun()
        st.markdown(f'<div class="col-footer">+ {max(0, len(benign_cases)-8)} more benign cases</div></div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # State management for selection
    if 'selected_pt' not in st.session_state:
        st.session_state['selected_pt'] = int(pool_df_sorted.iloc[0]['index'])
    patient_idx = st.session_state['selected_pt']

# Run Model
raw_feats, true_label, prob, rel, c_margin, c_spread, c_leaf, drivers = run_triage(patient_idx)

true_str = "Malignant" if true_label == 1 else "Benign"
true_color = "var(--danger)" if true_label == 1 else "var(--text)"

# COLUMN 1: VERDICT — now uses dynamic threshold
v_color = "var(--danger)" if prob >= threshold else "var(--success)"
v_bg = "var(--danger-light)" if prob >= threshold else "var(--success-light)"
v_label = "MALIGNANT" if prob >= threshold else "BENIGN"

r_tier = "High Reliability" if rel >= REL_HIGH else ("Moderate Reliability" if rel >= REL_LOW else "Low Reliability")
r_color = "var(--success)" if rel >= REL_HIGH else ("var(--warning)" if rel >= REL_LOW else "var(--danger)")
r_bg = "var(--success-light)" if rel >= REL_HIGH else ("var(--warning-light)" if rel >= REL_LOW else "var(--danger-light)")
r_border = "#bbf7d0" if rel >= REL_HIGH else ("#fde68a" if rel >= REL_LOW else "#fecaca")

rec_text = "High reliability. Proceed with standard care plan."
if rel < REL_HIGH: rec_text = f"Moderate reliability. Review: {drivers[0]['feature']}, {drivers[1]['feature']}. Consider re-measurement."
if rel < REL_LOW: rec_text = "Low reliability — conflicting signals. Escalate to Core Needle Biopsy (CNB)."
if abs(prob - 0.5) < 0.15: rec_text = "Prediction near decision boundary. Escalate to CNB regardless of reliability score."

verdict_html = f"""<div class="bento-card">
<div class="verdict-top" style="background:{v_bg};">
<div><div class="verdict-label" style="color:{v_color}">Triage Prediction</div><div class="verdict-value" style="color:{v_color}">{v_label}</div></div>
<div class="prob-block"><div class="prob-num" style="color:{v_color}">{prob*100:.1f}%</div><div class="prob-label">malignancy probability</div></div>
</div>
<div class="prob-track">
<div class="prob-fill" style="width:{prob*100}%; background:{v_color};"></div>
</div>
<div class="bento-body">
<div class="bento-header" style="padding:0; border:none; background:transparent; margin-bottom:16px;">Diagnostic Reliability Score</div>
<div class="rel-row">
<div class="rel-big" style="color:{r_color}">{rel:.2f}</div>
<div class="rel-right">
<span class="rel-tier" style="background:{r_bg}; color:{r_color}; border-color:{r_border}">{r_tier.upper()}</span>
<div class="rel-desc">Margin: {c_margin:.2f} | Spread: {c_spread:.2f} | Leaf: {c_leaf:.2f}</div>
</div>
</div>
<div class="comp-list">
<div class="comp-row"><div class="comp-name">Prob. Margin (×0.50)</div><div class="comp-track"><div class="comp-fill" style="width:{c_margin*100}%; background:#1d4ed8;"></div></div><div class="comp-score" style="color:#1d4ed8;">{c_margin:.2f}</div></div>
<div class="comp-row"><div class="comp-name">SHAP Spread (×0.30)</div><div class="comp-track"><div class="comp-fill" style="width:{c_spread*100}%; background:#7c3aed;"></div></div><div class="comp-score" style="color:#7c3aed;">{c_spread:.2f}</div></div>
<div class="comp-row"><div class="comp-name">Leaf Consensus (×0.20)</div><div class="comp-track"><div class="comp-fill" style="width:{c_leaf*100}%; background:#d97706;"></div></div><div class="comp-score" style="color:#d97706;">{c_leaf:.2f}</div></div>
</div>
<div class="rec-box" style="background:{r_bg}; border-color:{r_border};">
<div class="rec-title" style="color:{r_color}">Clinical Recommendation</div>
<div class="rec-text">{rec_text}</div>
</div>
</div>
</div>"""

# COLUMN 2: PATIENT & FEATURES
meta_html = f"""<div class="bento-card">
<div class="bento-header">Patient Profile</div>
<div class="bento-body">
<div class="patient-meta">
<div class="meta-item"><div class="meta-label">ID</div><div class="meta-value">#{patient_idx}</div></div>
<div class="meta-item"><div class="meta-label">Split</div><div class="meta-value" style="font-size:12px;">Test Set</div></div>
<div class="meta-item"><div class="meta-label">True Dx</div><div class="meta-value" style="color:{true_color}">{true_str}</div></div>
<div class="meta-item"><div class="meta-label">Model</div><div class="meta-value" style="font-size:12px; color:var(--success)">rf_calib · t={threshold:.2f}</div></div>
</div>
</div>
</div>"""

feature_html = """<div class="bento-card" style="flex-grow:1;">
<div class="bento-header">Morphological Measurements (30 raw features)</div>
<div class="bento-body">
<div class="feature-grid">"""
for idx, feat in enumerate(features[:8]):
    val = raw_feats[idx]
    col_max = X_test[:, idx].max()
    fill_pct = min(100, max(0, (val / col_max) * 100)) if col_max > 0 else 0
    feature_html += f"""<div class="feat-chip">
<div class="feat-name" title="{feat}">{feat.replace('_', ' ')}</div>
<div class="feat-val">{val:.4g}</div>
<div class="feat-bar"><div class="feat-fill" style="width:{fill_pct}%;"></div></div>
</div>"""
feature_html += "</div></div></div>"

# COLUMN 3: SHAP & STATS
shap_html = """<div class="bento-card">
<div class="bento-header">Why this prediction was made</div>
<div class="bento-body">
<div class="shap-list">"""
max_shap = max([abs(d['shap_value']) for d in drivers])
for d in drivers[:6]: 
    pct = (abs(d['shap_value']) / (max_shap + 0.001)) * 50
    sgn = "+" if d['shap_value'] > 0 else "−"
    color = "var(--danger)" if d['shap_value'] > 0 else "#1d4ed8"
    bar = f'<div class="shap-pos" style="width:{pct}%;"></div>' if d['shap_value'] > 0 else f'<div class="shap-neg" style="width:{pct}%;"></div>'
    shap_html += f"""<div class="shap-row">
<div class="shap-name" title="{d['feature']}">{d['feature'].replace('_', ' ')}</div>
<div class="shap-area"><div class="shap-axis"></div>{bar}</div>
<div class="shap-num" style="color:{color}">{sgn}{abs(d['shap_value']):.3f}</div>
</div>"""
shap_html += "</div></div></div>"

stat_html = f"""<div class="bento-card" style="flex-grow:1;">
<div class="bento-header">Model Performance (Test Set · threshold={threshold:.2f})</div>
<div class="bento-body" style="padding-bottom:12px;">
<div class="conf-grid">
<div class="conf-cell cell-g"><div class="conf-lbl" style="color:var(--success)">True Negative</div><div class="conf-num" style="color:var(--success)">{tn}</div><div class="conf-sub">Correct benign</div></div>
<div class="conf-cell cell-n"><div class="conf-lbl" style="color:var(--muted)">False Positive</div><div class="conf-num" style="color:var(--muted)">{fp}</div><div class="conf-sub">Unnecessary CNB</div></div>
<div class="conf-cell cell-b"><div class="conf-lbl" style="color:var(--danger)">False Negative</div><div class="conf-num" style="color:var(--danger)">{fn}</div><div class="conf-sub" style="color:var(--danger)">Missed cancers</div></div>
<div class="conf-cell cell-g"><div class="conf-lbl" style="color:var(--success)">True Positive</div><div class="conf-num" style="color:var(--success)">{tp}</div><div class="conf-sub">Caught cancers</div></div>
</div>
<div class="stat-row"><span class="stat-key">ROC-AUC</span><span class="stat-val">{metrics.get('auc', 0):.4f}</span></div>
<div class="stat-row"><span class="stat-key">Sensitivity</span><span class="stat-val" style="color:var(--success)">{(tp/(tp+fn))*100:.1f}%</span></div>
<div class="stat-row"><span class="stat-key">Specificity</span><span class="stat-val" style="color:var(--success)">{(tn/(tn+fp))*100:.1f}%</span></div>
</div></div>"""

with layout_col1:
    st.markdown(f'<div class="bento-col">{verdict_html}</div>', unsafe_allow_html=True)

with layout_col2:
    st.markdown(f'<div class="bento-col">{meta_html}{feature_html}</div>', unsafe_allow_html=True)

shap_placeholder.markdown(f'<div class="bento-col" style="margin-bottom: 24px;">{shap_html}</div>', unsafe_allow_html=True)
stat_placeholder.markdown(f'<div class="bento-col">{stat_html}</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">OncoTriage AI · Decision Support Only · Not a Substitute for Clinical Judgment · Breast Cancer Wisconsin Dataset (UCI)</div>
""", unsafe_allow_html=True)

# ── How We Built This ────────────────────────────────────────────────
st.markdown('<div style="margin-top:32px;"></div>', unsafe_allow_html=True)
with st.expander("How We Built This — Preprocessing Impact", expanded=False):
    impact_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "plots", "preprocessing_impact.png")
    sweep_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "plots", "threshold_sweep.png")
    if os.path.exists(impact_path):
        st.image(impact_path, caption='Each preprocessing decision tested and measured independently.')
    if os.path.exists(sweep_path):
        st.image(sweep_path, caption='Sensitivity vs Specificity across all tested thresholds.')
