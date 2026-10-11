import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# Load raw Kepler dataset
raw_df = pd.read_csv('data/q1_q16_koi_2026.09.12_10.49.05.csv', comment='#')

# Clean base dataset according to notebook 01 protocol
df_clean = raw_df[raw_df['koi_disposition'] != 'NOT DISPOSITIONED'].copy()

# Feature lists
CLASSIFICATION_FEATURES = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
    'koi_model_snr', 'koi_teq', 'koi_insol', 'koi_steff',
    'koi_slogg', 'koi_srad', 'koi_kepmag'
]

REGRESSION_FEATURES = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
    'koi_model_snr', 'koi_teq', 'koi_insol', 'koi_steff',
    'koi_slogg', 'koi_srad', 'koi_smass', 'koi_kepmag'
]

# Engineered feature: depth_duration_ratio
df_clean['depth_duration_ratio'] = df_clean['koi_depth'] / (df_clean['koi_duration'] + 1e-5)

# Encode disposition
disposition_map = {'CANDIDATE': 0, 'CONFIRMED': 1, 'FALSE POSITIVE': 2}
df_clean['koi_disposition_encoded'] = df_clean['koi_disposition'].map(disposition_map)

# 1. CLASSIFICATION PREPROCESSOR
CLF_FINAL = CLASSIFICATION_FEATURES + ['depth_duration_ratio']
X_clf = df_clean[CLF_FINAL]
y_clf = df_clean['koi_disposition_encoded']

X_clf_tr, _, y_clf_tr, _ = train_test_split(
    X_clf, y_clf, test_size=0.20, random_state=RANDOM_STATE, stratify=y_clf
)

imp_clf = SimpleImputer(strategy='median')
imp_clf.fit(X_clf_tr)
X_clf_tr_imp = imp_clf.transform(X_clf_tr)

scaler_clf = StandardScaler()
scaler_clf.fit(X_clf_tr_imp)

clf_preprocessor = {
    'imputer': imp_clf,
    'scaler': scaler_clf,
    'features': CLF_FINAL,
    'raw_features': CLASSIFICATION_FEATURES
}
joblib.dump(clf_preprocessor, 'models/preprocessor_clf.joblib')

# 2. REGRESSION PREPROCESSOR
reg_df = df_clean.dropna(subset=['koi_prad']).copy()
reg_df = reg_df[reg_df['koi_prad'] <= 50].copy()

REG_FINAL = REGRESSION_FEATURES + ['depth_duration_ratio']
X_reg = reg_df[REG_FINAL]
y_reg = reg_df['koi_prad']

X_reg_tr, _, y_reg_tr, _ = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=RANDOM_STATE
)

imp_reg = SimpleImputer(strategy='median')
imp_reg.fit(X_reg_tr)
X_reg_tr_imp = imp_reg.transform(X_reg_tr)

scaler_reg = StandardScaler()
scaler_reg.fit(X_reg_tr_imp)

reg_preprocessor = {
    'imputer': imp_reg,
    'scaler': scaler_reg,
    'features': REG_FINAL,
    'raw_features': REGRESSION_FEATURES
}
joblib.dump(reg_preprocessor, 'models/preprocessor_reg.joblib')

# 3. FEATURE METADATA AND PRESETS
metadata = {
    'features': {
        'koi_period': {'name': 'Orbital Period', 'unit': 'days', 'min': 0.2, 'max': 500.0, 'default': 11.26, 'desc': 'Time taken to complete one orbit around host star'},
        'koi_duration': {'name': 'Transit Duration', 'unit': 'hours', 'min': 0.5, 'max': 30.0, 'default': 3.64, 'desc': 'Time host star appears dimmed during transit'},
        'koi_depth': {'name': 'Transit Depth', 'unit': 'ppm', 'min': 10.0, 'max': 50000.0, 'default': 470.0, 'desc': 'Fraction of stellar brightness dimmed (parts per million)'},
        'koi_impact': {'name': 'Impact Parameter', 'unit': 'ratio', 'min': 0.0, 'max': 1.5, 'default': 0.53, 'desc': 'Sky-projected distance of transit chord from star center'},
        'koi_model_snr': {'name': 'Model Transit SNR', 'unit': 'ratio', 'min': 5.0, 'max': 500.0, 'default': 23.5, 'desc': 'Signal-to-noise ratio of transit detection'},
        'koi_teq': {'name': 'Equilibrium Temp', 'unit': 'K', 'min': 100.0, 'max': 3000.0, 'default': 850.0, 'desc': 'Estimated blackbody surface temperature of exoplanet'},
        'koi_insol': {'name': 'Insolation Flux', 'unit': 'S_Earth', 'min': 0.01, 'max': 2000.0, 'default': 120.0, 'desc': 'Stellar radiation energy flux relative to Earth'},
        'koi_steff': {'name': 'Host Star Temp', 'unit': 'K', 'min': 2500.0, 'max': 9000.0, 'default': 5770.0, 'desc': 'Effective surface temperature of the host star'},
        'koi_slogg': {'name': 'Host Star Surface Gravity', 'unit': 'log10(cm/s²)', 'min': 2.0, 'max': 5.5, 'default': 4.45, 'desc': 'Surface gravity of host star in log scale'},
        'koi_srad': {'name': 'Host Star Radius', 'unit': 'R_Sun', 'min': 0.2, 'max': 10.0, 'default': 1.0, 'desc': 'Physical radius of host star relative to our Sun'},
        'koi_smass': {'name': 'Host Star Mass', 'unit': 'M_Sun', 'min': 0.2, 'max': 3.0, 'default': 1.0, 'desc': 'Physical mass of host star relative to our Sun (Regression Only)'},
        'koi_kepmag': {'name': 'Kepler Optical Magnitude', 'unit': 'mag', 'min': 7.0, 'max': 20.0, 'default': 14.5, 'desc': 'Apparent optical brightness in Kepler photometric band'}
    },
    'presets': {
        '🌍 Earth 2.0 Candidate': {
            'koi_period': 355.2, 'koi_duration': 9.8, 'koi_depth': 84.0, 'koi_impact': 0.32,
            'koi_model_snr': 18.4, 'koi_teq': 260.0, 'koi_insol': 1.05, 'koi_steff': 5750.0,
            'koi_slogg': 4.44, 'koi_srad': 0.98, 'koi_smass': 0.99, 'koi_kepmag': 13.8,
            'desc': 'Earth-sized temperate candidate orbiting a G-type Sun-like star in habitable zone.'
        },
        '🪐 Kepler-452b (Confirmed Super-Earth)': {
            'koi_period': 384.8, 'koi_duration': 10.5, 'koi_depth': 210.0, 'koi_impact': 0.16,
            'koi_model_snr': 24.2, 'koi_teq': 265.0, 'koi_insol': 1.10, 'koi_steff': 5757.0,
            'koi_slogg': 4.32, 'koi_srad': 1.11, 'koi_smass': 1.04, 'koi_kepmag': 13.4,
            'desc': 'Known confirmed Super-Earth in habitable zone with ~1.6 Earth radii.'
        },
        '🔥 Hot Jupiter (Confirmed Giant)': {
            'koi_period': 3.52, 'koi_duration': 3.1, 'koi_depth': 12500.0, 'koi_impact': 0.45,
            'koi_model_snr': 145.0, 'koi_teq': 1520.0, 'koi_insol': 850.0, 'koi_steff': 6050.0,
            'koi_slogg': 4.38, 'koi_srad': 1.25, 'koi_smass': 1.15, 'koi_kepmag': 12.2,
            'desc': 'Gas giant orbiting in extreme proximity with deep transit depth and intense stellar flux.'
        },
        '⭐ Eclipsing Binary (Astrophysical False Positive)': {
            'koi_period': 1.85, 'koi_duration': 2.4, 'koi_depth': 85000.0, 'koi_impact': 0.88,
            'koi_model_snr': 420.0, 'koi_teq': 2200.0, 'koi_insol': 3200.0, 'koi_steff': 6400.0,
            'koi_slogg': 4.10, 'koi_srad': 1.80, 'koi_smass': 1.40, 'koi_kepmag': 11.5,
            'desc': 'Stellar binary eclipsing event characterized by huge transit depth and steep V-shaped ingress.'
        }
    }
}

with open('models/feature_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Preprocessors and metadata successfully saved!")
