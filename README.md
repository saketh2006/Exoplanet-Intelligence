# ExoML: Exoplanet Exploration Using Machine Learning
### **Machine Learning Capstone Project (23CSE301) — Review 1**
**Academic Year:** 2026-27 | **Program:** B.Tech Computer Science & Engineering  
**Primary Tools:** Python 3, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn  
**Dataset:** NASA Kepler Objects of Interest (KOI) Cumulative Dataset (`q1_q16_koi_2026.09.12_10.49.05.csv`)

---

## 1. Project Overview & Problem Statement
The discovery of extrasolar planets (exoplanets) is one of the most transformative endeavors in modern astrophysics. NASA's Kepler Space Telescope continuously monitored over 150,000 stars to detect minute periodic dips in stellar brightness caused by transiting planets. However, these candidate signals—termed **Kepler Objects of Interest (KOIs)**—are heavily contaminated by astrophysical false positives (such as eclipsing binary star systems and background variable stars) as well as instrumental noise artifacts.

### The Two Core Machine Learning Objectives in Review 1:
1. **Regression Track (Planetary Radius Estimation)**:
   Predict the physical **Planetary Radius** ($R_p$ in Earth radii, `koi_prad`) from observable photometric transit parameters and host star properties. Accurately determining planetary radius is essential to differentiate rocky terrestrial worlds ($R_p < 1.75 R_\oplus$) from gaseous sub-Neptunes and Jovian giants.
2. **Classification Track Part A (Astrophysical Signal Vetting)**:
   Automate the classification of detected transit signals into three vetted categories (`CONFIRMED` exoplanets, planetary `CANDIDATE`, and `FALSE POSITIVE`), eliminating the need for labor-intensive manual inspection.

---

## 2. Dataset Description
We utilize the official **NASA Kepler Objects of Interest (KOI) Cumulative Table** (`q1_q16_koi_2026.09.12_10.49.05.csv`).
- **Total Records:** 7,348 observations (5,049 vetted records with dispositions `CONFIRMED`, `CANDIDATE`, and `FALSE POSITIVE`; `NOT DISPOSITIONED` records are safely excluded).
- **Features Used:**
  - **12 Regression Features:** `koi_period`, `koi_duration`, `koi_depth`, `koi_impact`, `koi_model_snr`, `koi_teq`, `koi_insol`, `koi_steff`, `koi_slogg`, `koi_srad`, `koi_smass`, `koi_kepmag`.
  - **11 Classification Features:** The same feature set, omitting `koi_smass` per Part A specifications.
  - **Engineered Feature:** `depth_duration_ratio` ($\text{koi\_depth} / \text{koi\_duration}$), representing the mean ingress/egress transit steepness.
- **Strict Leakage Prevention:**
  - Photometric error columns (`*_err1`, `*_err2`), false-positive flags (`koi_fpflag_*`), and derived target columns (`koi_score`, `koi_ror`, `koi_pdisposition`) are strictly excluded.
  - All scalers and imputers are fitted **strictly on the 80% training split**.

---

## 3. Results Summary Tables

### 3.1 Regression Track Results (All 10 Algorithms)
*Evaluated on the identical held-out 20% test split, ranked by $R^2$ descending:*

| Rank | Algorithm | $R^2$ Score | RMSE ($R_\oplus$) | MAE ($R_\oplus$) | Key Note |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Gradient Boosting Regressor** | **0.9760** | **1.2768** | **0.4214** | Best overall model; captures non-linear transit geometry |
| **2** | **Random Forest Regressor** | **0.9700** | **1.4266** | **0.4689** | Robust bagging ensemble; high stability |
| **3** | Support Vector Regressor (SVR) | 0.9156 | 2.3948 | 0.9595 | Effective non-linear RBF kernel boundary |
| **4** | Decision Tree Regressor | 0.9088 | 2.4892 | 1.2078 | Interpretable single tree baseline |
| **5** | K-Nearest Neighbors Regressor | 0.8199 | 3.4976 | 1.5507 | Distance-weighted neighborhood predictor |
| **6** | Polynomial Regression (Degree 2) | 0.6996 | 4.5167 | 1.9154 | Interaction terms capture curvature |
| **7** | Linear Regression (OLS) | 0.5830 | 5.3218 | 3.3234 | Unregularized baseline |
| **8** | Ridge Regression ($L_2$) | 0.5827 | 5.3239 | 3.3202 | Stabilizes correlated stellar parameters |
| **9** | ElasticNet Regression | 0.5684 | 5.4142 | 3.2897 | Balanced $L_1 + L_2$ regularizer |
| **10** | Lasso Regression ($L_1$) | 0.5595 | 5.4699 | 3.3086 | Sparse feature selection |

#### Cross-Validation & Hyperparameter Tuning:
- **5-Fold Cross-Validation:**
  - Gradient Boosting Regressor: $R^2 = 0.9684 \pm 0.0063$
  - Random Forest Regressor: $R^2 = 0.9550 \pm 0.0085$
- **Hyperparameter Tuning via `GridSearchCV`:**
  - Tuned Gradient Boosting parameters: `{'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8}`
  - Tuned Random Forest parameters: `{'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1}`

---

### 3.2 Classification Track Part A Results (All 5 Algorithms)
*Target: `koi_disposition` (CONFIRMED, CANDIDATE, FALSE POSITIVE). Evaluated on stratified 20% test split:*

| Rank | Algorithm | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Multi-Class ROC-AUC (OvR) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Decision Tree Classifier** | **0.7673** | **0.7513** | **0.7673** | **0.7546** | **0.8582** |
| **2** | **KNN Classifier** | 0.7525 | 0.7348 | 0.7525 | 0.7356 | 0.8484 |
| **3** | **Support Vector Classifier (SVC)** | 0.7574 | 0.7373 | 0.7574 | 0.7322 | **0.8733** |
| **4** | Logistic Regression | 0.6960 | 0.6617 | 0.6960 | 0.6546 | 0.8211 |
| **5** | Gaussian Naive Bayes | 0.6297 | 0.6131 | 0.6297 | 0.5589 | 0.7822 |

---

## 4. Repository Structure
```
.
├── notebooks/
│   ├── 01_preprocessing.ipynb    # Notebook 1: Dataset Audit, Complete EDA & Preprocessing (Rubric A & B)
│   ├── 02_regression.ipynb       # Notebook 2: Full Regression Track - All 10 Models, Tuning, Plots (Rubric C)
│   ├── 03_classification.ipynb   # Notebook 3: Classification Part A - All 5 Models, Metrics, Viva Guide (Rubric D & E)
│   └── ExoML_Review_1.ipynb       # Consolidated Single-Notebook Version (All Sections Combined)
├── data/                          # Cleaned, partitioned train/test feature matrices (zero leakage)
├── models/                        # Serialized best model weights (.pkl)
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # Comprehensive project documentation
└── q1_q16_koi_2026.09.12_10.49.05.csv # NASA Kepler Cumulative Dataset
```

---

## 5. Environment Setup & How to Run

### 5.1 Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Pip package manager

### 5.2 Installation
Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 5.3 Running the Notebooks
You can run either the **three modular notebooks** (recommended for presentation division among team members) or the **single master notebook**:

#### Option A: Three Modular Notebooks (Track by Track)
1. **Preprocessing & EDA**: Open `notebooks/01_preprocessing.ipynb` and run all cells to audit the data, generate EDA distributions/heatmaps, and export partitions.
2. **Regression Track**: Open `notebooks/02_regression.ipynb` and run all cells to train all 10 regression models, run 5-fold CV, tune hyperparameters, and generate evaluation plots.
3. **Classification Track**: Open `notebooks/03_classification.ipynb` and run all cells to train all 5 Part-A classifiers, generate confusion matrices and ROC curves, and study the Viva Voce defense guide.

#### Option B: Consolidated Master Notebook
Open `notebooks/ExoML_Review_1.ipynb` and select **Kernel -> Restart & Run All Cells**. All 39 cells will execute top-to-bottom without warnings or errors.

---

## 6. Review 1 Checklist & Rubric Verification (25 Marks)
- [x] **A1: Dataset Loading & Audit (1 Mark)**: Shape, dtypes, missing percentages, and class distributions reported.
- [x] **A2: EDA Visualisations (2 Marks)**: Feature distribution plots for all 12 features, correlation heatmap, target distribution, and 3 feature-target scatter plots.
- [x] **A3: Insight Commentary (1 Mark)**: Comprehensive Markdown commentary interpreting each major visualization with astrophysical context.
- [x] **B1: Data Cleaning (1 Mark)**: Median imputation strictly on train split; physical outlier filtering ($R_p \le 50 R_\oplus$) fully justified.
- [x] **B2: Encoding, Scaling & Splitting (1 Mark)**: `StandardScaler` fitted on train only; stratified train-test split (80:20, `random_state=42`).
- [x] **B3: Feature Engineering (1 Mark)**: `depth_duration_ratio` constructed and justified.
- [x] **C1: Regression Implementation (4 Marks)**: All 10 required regression algorithms trained and tested without errors.
- [x] **C2: Comparative Evaluation (2 Marks)**: Consolidated summary table sorted by $R^2$ descending.
- [x] **C3: Hyperparameter Tuning (2 Marks)**: 5-fold CV on top 2 models; `GridSearchCV` on Random Forest & Gradient Boosting.
- [x] **C4: Regression Visualisation (1 Mark)**: Actual vs. Predicted, Residual plot, and Feature Importance plot included.
- [x] **D1: Classification Implementation (2 Marks)**: All 5 Part-A classifiers trained and producing predictions.
- [x] **D2: Classification Evaluation (1 Mark)**: Accuracy, Precision, Recall, Weighted F1, Confusion Matrices, and ROC-AUC (OvR) reported in a consolidated table.
- [x] **E1 & Viva (6 Marks)**: Structured presentation story arc with dedicated Viva Defense Q&A guide.
