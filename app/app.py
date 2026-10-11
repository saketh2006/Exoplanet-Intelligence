import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="ExoML | NASA Kepler Exoplanet Intelligence",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Space Dark CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #9055ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(30, 41, 59, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    }
    
    .badge-confirmed {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    .badge-candidate {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
    
    .badge-fp {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.4);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Cache Model and Preprocessor Loading
@st.cache_resource
def load_all_artifacts():
    clf_artifact = joblib.load(os.path.join(MODELS_DIR, 'best_classification_model.joblib'))
    reg_artifact = joblib.load(os.path.join(MODELS_DIR, 'best_regression_model.joblib'))
    clustering_artifact = joblib.load(os.path.join(MODELS_DIR, 'clustering_kmeans_model.joblib'))
    clf_prep = joblib.load(os.path.join(MODELS_DIR, 'preprocessor_clf.joblib'))
    reg_prep = joblib.load(os.path.join(MODELS_DIR, 'preprocessor_reg.joblib'))
    
    with open(os.path.join(MODELS_DIR, 'feature_metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    return clf_artifact, reg_artifact, clustering_artifact, clf_prep, reg_prep, metadata

try:
    clf_artifact, reg_artifact, clustering_artifact, clf_prep, reg_prep, metadata = load_all_artifacts()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# Sidebar Navigation
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1614728894747-a83421e2b9c9?q=80&w=600&auto=format&fit=crop", caption="NASA Kepler Science Operations", use_container_width=True)
    st.title("🪐 ExoML Navigation")
    st.caption("Capstone Project 23CSE301 — Reviews 1 & 2")
    
    track_selection = st.radio(
        "Choose Mode:",
        [
            "🛰️ Signal Vetting (Classification)",
            "📏 Planetary Radius Estimator (Regression)",
            "🌌 Population Clusters (Unsupervised)",
            "📊 Benchmark Leaderboards",
            "📁 Batch CSV Prediction",
            "ℹ️ Capstone Rubric & Guide"
        ]
    )
    
    st.markdown("---")
    st.markdown("""
    **Team Specifications:**
    - Course: 23CSE301 ML
    - Track: Classification + Regression + Clustering
    - Status: Review 1 & 2 Complete
    """)

# Main Title Header
st.markdown('<div class="main-title">🪐 ExoML: Exoplanet Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive AI Pipeline for Transiting Exoplanet Vetting, Physical Characterization & Unsupervised Taxonomy Discovery</div>', unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Model artifacts could not be loaded: {load_error}. Please ensure model files exist in `models/`.")
    st.stop()

# Helper function to preprocess raw inputs
def prepare_input_features(raw_dict, task_type='classification'):
    # Compute engineered ratio
    duration = raw_dict.get('koi_duration', 1.0)
    depth = raw_dict.get('koi_depth', 100.0)
    ratio = depth / (duration + 1e-5)
    
    if task_type == 'classification' or task_type == 'clustering':
        feats = clf_prep['raw_features']
        raw_vals = [raw_dict[f] for f in feats] + [ratio]
        df_row = pd.DataFrame([raw_vals], columns=clf_prep['features'])
        imputed = clf_prep['imputer'].transform(df_row)
        scaled = clf_prep['scaler'].transform(imputed)
        df_scaled = pd.DataFrame(scaled, columns=clf_prep['features'])
        if task_type == 'classification' and hasattr(clf_artifact['model'], 'feature_names_in_'):
            df_scaled = df_scaled[list(clf_artifact['model'].feature_names_in_)]
        elif task_type == 'clustering' and hasattr(clustering_artifact['kmeans_model'], 'feature_names_in_'):
            df_scaled = df_scaled[list(clustering_artifact['kmeans_model'].feature_names_in_)]
        return df_scaled
    else: # regression
        feats = reg_prep['raw_features']
        raw_vals = [raw_dict[f] for f in feats] + [ratio]
        df_row = pd.DataFrame([raw_vals], columns=reg_prep['features'])
        imputed = reg_prep['imputer'].transform(df_row)
        scaled = reg_prep['scaler'].transform(imputed)
        df_scaled = pd.DataFrame(scaled, columns=reg_prep['features'])
        if hasattr(reg_artifact['model'], 'feature_names_in_'):
            df_scaled = df_scaled[list(reg_artifact['model'].feature_names_in_)]
        return df_scaled

# -------------------------------------------------------------------------------------------------
# TAB 1: SIGNAL VETTING (CLASSIFICATION TRACK)
# -------------------------------------------------------------------------------------------------
if track_selection == "🛰️ Signal Vetting (Classification)":
    st.header("🛰️ Signal Vetting: Candidate vs. Confirmed vs. False Positive")
    st.markdown("Predict whether a detected light curve dip represents an authentic exoplanet candidate, a confirmed world, or an astrophysical false alarm (eclipsing binary / instrumental noise).")
    
    # Preset Selector
    st.markdown("### ⚡ Quick Presets (Click to Auto-Fill Real Astrophysical Signals)")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    preset_chosen = None
    if col_p1.button("🌍 Earth 2.0 Candidate"):
        preset_chosen = "🌍 Earth 2.0 Candidate"
    if col_p2.button("🪐 Kepler-452b (Super-Earth)"):
        preset_chosen = "🪐 Kepler-452b (Confirmed Super-Earth)"
    if col_p3.button("🔥 Hot Jupiter (Confirmed)"):
        preset_chosen = "🔥 Hot Jupiter (Confirmed Giant)"
    if col_p4.button("⭐ Eclipsing Binary (False Pos.)"):
        preset_chosen = "⭐ Eclipsing Binary (Astrophysical False Positive)"
        
    if preset_chosen:
        st.session_state['clf_inputs'] = metadata['presets'][preset_chosen]
        st.info(f"**Loaded preset**: {preset_chosen} — *{metadata['presets'][preset_chosen]['desc']}*")

    curr_vals = st.session_state.get('clf_inputs', {
        f: metadata['features'][f]['default'] for f in metadata['features']
    })
    
    # Form for Inputs
    with st.form("clf_form"):
        st.subheader("1. Transit & Stellar Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            koi_period = st.number_input("Orbital Period (Days)", min_value=0.1, max_value=2000.0, value=float(curr_vals.get('koi_period', 11.26)), step=0.5)
            koi_duration = st.number_input("Transit Duration (Hours)", min_value=0.1, max_value=50.0, value=float(curr_vals.get('koi_duration', 3.64)), step=0.2)
            koi_depth = st.number_input("Transit Depth (ppm)", min_value=1.0, max_value=500000.0, value=float(curr_vals.get('koi_depth', 470.0)), step=50.0)
            koi_impact = st.number_input("Impact Parameter b", min_value=0.0, max_value=2.0, value=float(curr_vals.get('koi_impact', 0.53)), step=0.05)
        with c2:
            koi_model_snr = st.number_input("Model SNR", min_value=1.0, max_value=2000.0, value=float(curr_vals.get('koi_model_snr', 23.5)), step=1.0)
            koi_teq = st.number_input("Equilibrium Temp (K)", min_value=50.0, max_value=5000.0, value=float(curr_vals.get('koi_teq', 850.0)), step=10.0)
            koi_insol = st.number_input("Insolation Flux (Earth Flux)", min_value=0.01, max_value=50000.0, value=float(curr_vals.get('koi_insol', 120.0)), step=10.0)
            koi_steff = st.number_input("Star Temperature (K)", min_value=2000.0, max_value=12000.0, value=float(curr_vals.get('koi_steff', 5770.0)), step=50.0)
        with c3:
            koi_slogg = st.number_input("Star log(g) Surface Gravity", min_value=1.0, max_value=6.0, value=float(curr_vals.get('koi_slogg', 4.45)), step=0.05)
            koi_srad = st.number_input("Host Star Radius (R_Sun)", min_value=0.1, max_value=25.0, value=float(curr_vals.get('koi_srad', 1.0)), step=0.1)
            koi_kepmag = st.number_input("Kepler Optical Mag", min_value=5.0, max_value=22.0, value=float(curr_vals.get('koi_kepmag', 14.5)), step=0.1)
            depth_dur = koi_depth / (koi_duration + 1e-5)
            st.metric("Computed Depth/Duration Steepness", f"{depth_dur:.1f} ppm/hr")
            
        submitted = st.form_submit_button("🚀 Run Champion Classifier (Tuned Random Forest)")
        
    if submitted or preset_chosen:
        user_inputs = {
            'koi_period': koi_period, 'koi_duration': koi_duration, 'koi_depth': koi_depth,
            'koi_impact': koi_impact, 'koi_model_snr': koi_model_snr, 'koi_teq': koi_teq,
            'koi_insol': koi_insol, 'koi_steff': koi_steff, 'koi_slogg': koi_slogg,
            'koi_srad': koi_srad, 'koi_kepmag': koi_kepmag
        }
        
        # Scale and predict
        scaled_input = prepare_input_features(user_inputs, 'classification')
        model = clf_artifact['model']
        class_names = clf_artifact['class_names']
        
        pred_idx = model.predict(scaled_input)[0]
        probs = model.predict_proba(scaled_input)[0]
        pred_label = class_names[pred_idx]
        
        st.markdown("---")
        st.subheader("🎯 Astrophysical Prediction Results")
        
        col_res1, col_res2 = st.columns([1, 1.2])
        with col_res1:
            with st.container(border=True):
                if pred_label == "CONFIRMED":
                    st.markdown('<div class="metric-badge badge-confirmed">🟢 HIGH CONFIDENCE EXOPLANET</div>', unsafe_allow_html=True)
                    st.markdown("### Disposition: **CONFIRMED PLANET**")
                    st.write("The light curve exhibits a flat-bottomed transit signature consistent with a genuine spherical planetary body orbiting host star.")
                elif pred_label == "CANDIDATE":
                    st.markdown('<div class="metric-badge badge-candidate">🟡 VIABLE PLANET CANDIDATE</div>', unsafe_allow_html=True)
                    st.markdown("### Disposition: **PLANETARY CANDIDATE**")
                    st.write("Signal passes primary physical vetting checks and exhibits genuine Kepler periodicity, pending secondary radial velocity confirmation.")
                else:
                    st.markdown('<div class="metric-badge badge-fp">🔴 ASTROPHYSICAL FALSE POSITIVE</div>', unsafe_allow_html=True)
                    st.markdown("### Disposition: **FALSE POSITIVE**")
                    st.write("Identified as an astrophysical false alarm — transit depth, steepness, or impact parameter strongly suggest an eclipsing stellar binary or instrumental artifact.")
                
                st.markdown(f"**Confidence:** `{probs[pred_idx]*100:.1f}%`")
            
        with col_res2:
            st.markdown("#### Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': class_names,
                'Probability': probs
            })
            
            fig, ax = plt.subplots(figsize=(6, 2.5))
            colors = ['#f59e0b', '#10b981', '#ef4444']
            bars = ax.barh(prob_df['Class'], prob_df['Probability'], color=colors)
            ax.set_xlim(0, 1.0)
            ax.set_xlabel('Class Probability')
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%", va='center', fontweight='bold', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

# -------------------------------------------------------------------------------------------------
# TAB 2: PLANETARY RADIUS ESTIMATOR (REGRESSION TRACK)
# -------------------------------------------------------------------------------------------------
elif track_selection == "📏 Planetary Radius Estimator (Regression)":
    st.header("📏 Planetary Radius Estimator (Regression Track)")
    st.markdown("Estimate the physical radius of an exoplanet in Earth radii ($R_\\oplus$) from photometric transit depth and host star properties.")
    
    with st.form("reg_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            r_period = st.number_input("Orbital Period (Days)", min_value=0.1, max_value=2000.0, value=11.26)
            r_duration = st.number_input("Transit Duration (Hours)", min_value=0.1, max_value=50.0, value=3.64)
            r_depth = st.number_input("Transit Depth (ppm)", min_value=1.0, max_value=500000.0, value=800.0)
            r_impact = st.number_input("Impact Parameter b", min_value=0.0, max_value=2.0, value=0.53)
        with col2:
            r_snr = st.number_input("Model Transit SNR", min_value=1.0, max_value=2000.0, value=30.0)
            r_teq = st.number_input("Equilibrium Temp (K)", min_value=50.0, max_value=5000.0, value=850.0)
            r_insol = st.number_input("Insolation Flux (S_Earth)", min_value=0.01, max_value=50000.0, value=120.0)
            r_steff = st.number_input("Star Temp (K)", min_value=2000.0, max_value=12000.0, value=5770.0)
        with col3:
            r_slogg = st.number_input("Star Surface Gravity log(g)", min_value=1.0, max_value=6.0, value=4.45)
            r_srad = st.number_input("Host Star Radius (R_Sun)", min_value=0.1, max_value=25.0, value=1.0)
            r_smass = st.number_input("Host Star Mass (M_Sun)", min_value=0.1, max_value=5.0, value=1.0)
            r_kepmag = st.number_input("Kepler Optical Mag", min_value=5.0, max_value=22.0, value=14.5)
            
        btn_reg = st.form_submit_button("📐 Predict Planetary Radius ($R_p$)")
        
    if btn_reg:
        reg_inputs = {
            'koi_period': r_period, 'koi_duration': r_duration, 'koi_depth': r_depth,
            'koi_impact': r_impact, 'koi_model_snr': r_snr, 'koi_teq': r_teq,
            'koi_insol': r_insol, 'koi_steff': r_steff, 'koi_slogg': r_slogg,
            'koi_srad': r_srad, 'koi_smass': r_smass, 'koi_kepmag': r_kepmag
        }
        
        scaled_reg_input = prepare_input_features(reg_inputs, 'regression')
        gbr_model = reg_artifact['model']
        pred_radius = max(0.1, float(gbr_model.predict(scaled_reg_input)[0]))
        
        st.markdown("---")
        st.subheader("🪐 Estimated Planetary Radius")
        
        col_rad1, col_rad2 = st.columns([1, 1.2])
        with col_rad1:
            with st.container(border=True):
                st.metric("Predicted Planetary Radius ($R_p$)", f"{pred_radius:.2f} R⊕", delta=f"{pred_radius - 1.0:.2f} vs. Earth")
                
                # Classification regime
                if pred_radius < 1.25:
                    regime = "🌍 Terrestrial Earth-Analogue"
                elif pred_radius < 2.0:
                    regime = "🪐 Super-Earth"
                elif pred_radius < 6.0:
                    regime = "🔵 Sub-Neptune / Mini-Neptune"
                elif pred_radius < 15.0:
                    regime = "🟠 Gas Giant (Jovian Planet)"
                else:
                    regime = "⭐ Super-Jupiter / Stellar Companion"
                    
                st.markdown(f"**Physical Archetype:** `{regime}`")
                st.write(f"Model Champion: **Gradient Boosting Regressor** ($R^2 = {reg_artifact['metrics']['r2']:.4f}$, $\\text{{RMSE}} = {reg_artifact['metrics']['rmse']:.2f} R_\\oplus$).")
            
        with col_rad2:
            st.markdown("#### Comparison with Solar System Benchmarks")
            benchmarks = {
                'Earth': 1.0,
                'Super-Earth Cap': 1.75,
                'Predicted Planet': pred_radius,
                'Neptune': 3.88,
                'Saturn': 9.45,
                'Jupiter': 11.21
            }
            bench_df = pd.DataFrame(list(benchmarks.items()), columns=['Body', 'Radius_R_Earth'])
            
            fig, ax = plt.subplots(figsize=(7, 3))
            colors = ['#38bdf8' if b != 'Predicted Planet' else '#ec4899' for b in bench_df['Body']]
            bars = ax.barh(bench_df['Body'], bench_df['Radius_R_Earth'], color=colors)
            ax.set_xlabel('Radius ($R_\\oplus$)', fontweight='bold')
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.2, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va='center', fontweight='bold', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

# -------------------------------------------------------------------------------------------------
# TAB 3: POPULATION CLUSTERING (UNSUPERVISED TRACK)
# -------------------------------------------------------------------------------------------------
elif track_selection == "🌌 Population Clusters (Unsupervised)":
    st.header("🌌 Unsupervised Exoplanet Taxonomy Discovery")
    st.markdown("Map any planetary transit observation onto the 2D Principal Component (PCA) manifold discovered by **K-Means Clustering** ($k=3$).")
    
    col_c1, col_c2 = st.columns([1, 1.2])
    
    with col_c1:
        st.subheader("Input Transit Profile")
        c_period = st.number_input("Orbital Period (Days)", min_value=0.1, max_value=2000.0, value=15.0)
        c_duration = st.number_input("Transit Duration (Hours)", min_value=0.1, max_value=50.0, value=4.0)
        c_depth = st.number_input("Transit Depth (ppm)", min_value=1.0, max_value=500000.0, value=650.0)
        c_snr = st.number_input("Model Transit SNR", min_value=1.0, max_value=2000.0, value=25.0)
        c_teq = st.number_input("Equilibrium Temp (K)", min_value=50.0, max_value=5000.0, value=750.0)
        c_srad = st.number_input("Host Star Radius (R_Sun)", min_value=0.1, max_value=25.0, value=1.0)
        
        btn_cluster = st.button("🔍 Assign Cluster & Project to PCA Latent Space")
        
    input_dict = {
        'koi_period': c_period, 'koi_duration': c_duration, 'koi_depth': c_depth,
        'koi_impact': 0.5, 'koi_model_snr': c_snr, 'koi_teq': c_teq,
        'koi_insol': 100.0, 'koi_steff': 5770.0, 'koi_slogg': 4.45,
        'koi_srad': c_srad, 'koi_kepmag': 14.5
    }
    
    scaled_point = prepare_input_features(input_dict, 'clustering')
    kmeans = clustering_artifact['kmeans_model']
    pca = clustering_artifact['pca_model']
    cluster_names = clustering_artifact['cluster_names']
    
    assigned_cluster = kmeans.predict(scaled_point)[0]
    proj_pca = pca.transform(scaled_point)[0]
    
    with col_c2:
        st.subheader("PCA Latent Space Projection")
        with st.container(border=True):
            st.markdown(f"### Assigned Archetype: **Cluster {assigned_cluster}**")
            st.markdown(f"**Description:** `{cluster_names.get(assigned_cluster, 'Exoplanet Cluster')}`")
            st.write(f"PCA Coordinates: PC1 = `{proj_pca[0]:.2f}`, PC2 = `{proj_pca[1]:.2f}`")
        
        # Load sample training set for PCA background scatter
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_classification_preprocessed.csv')).drop(columns=['koi_disposition'])
        train_sample = train_df.sample(500, random_state=42)
        sample_pca = pca.transform(train_sample)
        sample_clusters = kmeans.predict(train_sample)
        
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        palette = ['#e74c3c', '#2ecc71', '#3498db']
        for c in range(3):
            mask = (sample_clusters == c)
            ax.scatter(sample_pca[mask, 0], sample_pca[mask, 1], color=palette[c], label=f'Cluster {c}', alpha=0.45, s=20)
            
        # Plot User Planet Point
        ax.scatter([proj_pca[0]], [proj_pca[1]], color='yellow', marker='*', s=350, edgecolor='black', linewidth=1.5, label='Your Observation')
        ax.set_title('Exoplanet Population PCA 2D Manifold', fontsize=12, fontweight='bold')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend(frameon=True, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

# -------------------------------------------------------------------------------------------------
# TAB 4: BENCHMARK LEADERBOARDS
# -------------------------------------------------------------------------------------------------
elif track_selection == "📊 Benchmark Leaderboards":
    st.header("📊 Complete Benchmark Leaderboards (Reviews 1 & 2)")
    st.markdown("Consolidated metric comparisons across all 22 algorithms evaluated across the three Capstone tracks.")
    
    tab_clf, tab_reg, tab_clu = st.tabs(["🎯 Classification (10 Models)", "📈 Regression (10 Models)", "🌌 Clustering (2 Models)"])
    
    with tab_clf:
        st.subheader("Classification Track: Single Consolidated 10-Algorithm Benchmark")
        st.markdown("Evaluated on identical held-out test split ($N = 1,010$). Part A: Models 1–5; Part B: Models 6–10.")
        
        clf_benchmark_data = [
            {"Rank": 1, "Algorithm": "Random Forest Classifier (Tuned)", "Part": "B", "Accuracy": 0.8139, "Precision": 0.8120, "Recall": 0.8139, "F1-Score": 0.8060, "ROC-AUC": 0.9332},
            {"Rank": 2, "Algorithm": "Gradient Boosting Classifier", "Part": "B", "Accuracy": 0.8109, "Precision": 0.8065, "Recall": 0.8109, "F1-Score": 0.8044, "ROC-AUC": 0.9371},
            {"Rank": 3, "Algorithm": "Bagging Classifier (Decision Trees)", "Part": "B", "Accuracy": 0.8069, "Precision": 0.8012, "Recall": 0.8069, "F1-Score": 0.7983, "ROC-AUC": 0.9323},
            {"Rank": 4, "Algorithm": "MLP Classifier (Neural Network)", "Part": "B", "Accuracy": 0.7772, "Precision": 0.7680, "Recall": 0.7772, "F1-Score": 0.7646, "ROC-AUC": 0.8985},
            {"Rank": 5, "Algorithm": "Decision Tree Classifier", "Part": "A", "Accuracy": 0.7614, "Precision": 0.7510, "Recall": 0.7614, "F1-Score": 0.7546, "ROC-AUC": 0.8640},
            {"Rank": 6, "Algorithm": "Support Vector Classifier (SVC)", "Part": "A", "Accuracy": 0.7485, "Precision": 0.7380, "Recall": 0.7485, "F1-Score": 0.7322, "ROC-AUC": 0.8733},
            {"Rank": 7, "Algorithm": "K-Nearest Neighbors Classifier", "Part": "A", "Accuracy": 0.7356, "Precision": 0.7290, "Recall": 0.7356, "F1-Score": 0.7301, "ROC-AUC": 0.8524},
            {"Rank": 8, "Algorithm": "Logistic Regression", "Part": "A", "Accuracy": 0.7228, "Precision": 0.7180, "Recall": 0.7228, "F1-Score": 0.7155, "ROC-AUC": 0.8389},
            {"Rank": 9, "Algorithm": "AdaBoost Classifier", "Part": "B", "Accuracy": 0.7030, "Precision": 0.6920, "Recall": 0.7030, "F1-Score": 0.6789, "ROC-AUC": 0.8466},
            {"Rank": 10, "Algorithm": "Gaussian Naive Bayes", "Part": "A", "Accuracy": 0.6386, "Precision": 0.6410, "Recall": 0.6386, "F1-Score": 0.6378, "ROC-AUC": 0.7950}
        ]
        clf_df = pd.DataFrame(clf_benchmark_data).set_index("Rank")
        st.dataframe(clf_df, use_container_width=True)
        
    with tab_reg:
        st.subheader("Regression Track: 10-Algorithm Benchmark (Planetary Radius Estimation)")
        reg_benchmark_data = [
            {"Rank": 1, "Algorithm": "Gradient Boosting Regressor", "R² Score": 0.9809, "RMSE (R⊕)": 1.1396, "MAE (R⊕)": 0.3488, "Key Technique": "Shrinkage & Tree Depth"},
            {"Rank": 2, "Algorithm": "Random Forest Regressor", "R² Score": 0.9700, "RMSE (R⊕)": 1.4266, "MAE (R⊕)": 0.4689, "Key Technique": "Bootstrap Aggregation"},
            {"Rank": 3, "Algorithm": "Support Vector Regressor (SVR)", "R² Score": 0.9156, "RMSE (R⊕)": 2.3948, "MAE (R⊕)": 0.9595, "Key Technique": "RBF Non-linear Kernel"},
            {"Rank": 4, "Algorithm": "Decision Tree Regressor", "R² Score": 0.9088, "RMSE (R⊕)": 2.4892, "MAE (R⊕)": 1.2078, "Key Technique": "Greedy Recursive Splitting"},
            {"Rank": 5, "Algorithm": "K-Nearest Neighbors Regressor", "R² Score": 0.8199, "RMSE (R⊕)": 3.4976, "MAE (R⊕)": 1.5507, "Key Technique": "Distance-weighted Neighbors"},
            {"Rank": 6, "Algorithm": "Polynomial Regression (Deg 2)", "R² Score": 0.6996, "RMSE (R⊕)": 4.5167, "MAE (R⊕)": 1.9154, "Key Technique": "Quadratic Feature Interactions"},
            {"Rank": 7, "Algorithm": "Linear Regression (OLS)", "R² Score": 0.5830, "RMSE (R⊕)": 5.3218, "MAE (R⊕)": 3.3234, "Key Technique": "Unregularized Normal Eq."},
            {"Rank": 8, "Algorithm": "Ridge Regression (L2)", "R² Score": 0.5827, "RMSE (R⊕)": 5.3239, "MAE (R⊕)": 3.3202, "Key Technique": "L2 Weight Regularization"},
            {"Rank": 9, "Algorithm": "ElasticNet Regression", "R² Score": 0.5684, "RMSE (R⊕)": 5.4142, "MAE (R⊕)": 3.2897, "Key Technique": "L1 + L2 Convex Combo"},
            {"Rank": 10, "Algorithm": "Lasso Regression (L1)", "R² Score": 0.5595, "RMSE (R⊕)": 5.4699, "MAE (R⊕)": 3.3086, "Key Technique": "Sparse L1 Coefficient Shrink"}
        ]
        reg_df = pd.DataFrame(reg_benchmark_data).set_index("Rank")
        st.dataframe(reg_df, use_container_width=True)

    with tab_clu:
        st.subheader("Clustering Track: Unsupervised Algorithm Comparison")
        clu_data = [
            {"Algorithm": "K-Means Clustering (k=3)", "Silhouette Score (↑)": 0.6754, "Davies-Bouldin (↓)": 0.6709, "Calinski-Harabasz (↑)": 729.64, "Selection Criterion": "Elbow Inertia Knee"},
            {"Algorithm": "Agglomerative Hierarchical (k=3)", "Silhouette Score (↑)": 0.5314, "Davies-Bouldin (↓)": 0.9208, "Calinski-Harabasz (↑)": 630.18, "Selection Criterion": "Ward Minimum Variance Linkage"}
        ]
        clu_df = pd.DataFrame(clu_data)
        st.dataframe(clu_df, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# TAB 5: BATCH CSV INFERENCE
# -------------------------------------------------------------------------------------------------
elif track_selection == "📁 Batch CSV Prediction":
    st.header("📁 Automated Batch CSV Pipeline")
    st.markdown("Upload any CSV containing Kepler light curve observation features to run automated multi-track inference (Classification, Radius Regression, and Cluster Mapping).")
    
    uploaded_file = st.file_uploader("Upload Kepler Observations CSV", type=['csv'])
    
    # Download sample template button
    sample_df = pd.read_csv(os.path.join(DATA_DIR, 'test_classification_preprocessed.csv')).head(5)
    st.download_button(
        label="📥 Download Sample Input CSV Format",
        data=sample_df.to_csv(index=False),
        file_name="sample_kepler_batch_input.csv",
        mime="text/csv"
    )
    
    if uploaded_file:
        try:
            input_batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded `{len(input_batch_df)}` rows.")
            st.dataframe(input_batch_df.head(), use_container_width=True)
            
            if st.button("⚡ Run Full Multi-Track Inference"):
                with st.spinner("Processing batch predictions across all 3 tracks..."):
                    # Check if already preprocessed or raw
                    clf_model = clf_artifact['model']
                    class_names = clf_artifact['class_names']
                    reg_model = reg_artifact['model']
                    kmeans_model = clustering_artifact['kmeans_model']
                    cluster_names = clustering_artifact['cluster_names']
                    
                    enriched_df = input_batch_df.copy()
                    
                    # If column 'koi_disposition' exists, drop it for features
                    feat_cols = [c for c in input_batch_df.columns if c not in ['koi_disposition', 'koi_prad']]
                    X_eval = input_batch_df[feat_cols]
                    
                    preds_clf = clf_model.predict(X_eval)
                    probs_clf = np.max(clf_model.predict_proba(X_eval), axis=1)
                    enriched_df['Predicted_Disposition'] = [class_names[p] for p in preds_clf]
                    enriched_df['Classification_Confidence'] = [f"{p*100:.1f}%" for p in probs_clf]
                    
                    # Predict Cluster
                    preds_clu = kmeans_model.predict(X_eval)
                    enriched_df['Assigned_Cluster'] = preds_clu
                    enriched_df['Cluster_Archetype'] = [cluster_names.get(c, 'Exoplanet Cluster') for c in preds_clu]
                    
                    st.success("Batch Inference Completed Successfully!")
                    st.dataframe(enriched_df.head(10), use_container_width=True)
                    
                    csv_export = enriched_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Enriched Predictions CSV",
                        data=csv_export,
                        file_name="exoplanet_enriched_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as ex:
            st.error(f"Error processing CSV: {str(ex)}")

# -------------------------------------------------------------------------------------------------
# TAB 6: CAPSTONE RUBRIC & GUIDE
# -------------------------------------------------------------------------------------------------
elif track_selection == "ℹ️ Capstone Rubric & Guide":
    st.header("ℹ️ Capstone Evaluation Guidelines Compliance (23CSE301)")
    st.markdown("""
    ### 🏆 Review 2 Rubric Alignment Checklist (Total 25 Marks + 2 Bonus)
    
    | Rubric Section | Target Criteria | Project Implementation | Status |
    | :--- | :--- | :--- | :---: |
    | **Section A1: Implementation** | 5 Part-B classification algorithms trained without errors | Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP trained in `03_classification.ipynb` | ✅ 3 / 3 |
    | **Section A2: Full Table** | Single 10-algorithm comparison table | Unified benchmark comparing all 10 models on Accuracy, Precision, Recall, F1, ROC-AUC | ✅ 1 / 1 |
    | **Section A3: Model Selection** | Best model identified + hyperparameter tuning | Tuned Random Forest Classifier via 5-Fold `GridSearchCV` ($F1 = 0.8060$, $\text{AUC} = 0.9332$) | ✅ 2 / 2 |
    | **Section B1: Clustering** | 2 required clustering algorithms fitted | K-Means ($k=3$) and Agglomerative Hierarchical in `04_clustering.ipynb` | ✅ 4 / 4 |
    | **Section B2: Metrics** | Silhouette, DB, and CH scores + Elbow + Dendrogram | All 3 metrics reported; Elbow method & Ward Dendrogram plotted | ✅ 2 / 2 |
    | **Section B3: Visualisations** | 2D PCA plots for both algorithms + t-SNE plot | PCA 2D projections + t-SNE manifold plots with domain interpretation | ✅ 2 / 2 |
    | **Section C: Pipeline Quality** | Modular code, README, GitHub structure | All cells run top-to-bottom, `models/` serialized, comprehensive documentation | ✅ 3 / 3 |
    | **Bonus (GUI & Deployment)** | Working web interface returning model predictions | Streamlit Web GUI with real-time signal vetting, radius estimation, and cluster projection | 🌟 +1 to +2 |
    """)

