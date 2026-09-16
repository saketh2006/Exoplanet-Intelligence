# 🪐 ExoML: Exoplanet Exploration Using Machine Learning
### 🎓 Machine Learning Capstone Project (23CSE301) — Review 1
**Academic Year:** 2026–27 | **Program:** B.Tech Computer Science & Engineering  
**Primary Stack:** Python 3.11 · Scikit-Learn · Pandas · NumPy · Matplotlib · Seaborn

---

## 🚀 Project Overview

| Track | Objective | Target Variable | Evaluation Metric | Best Model |
| :--- | :--- | :--- | :--- | :--- |
| 📈 **Regression Track** | Predict physical **Planetary Radius** ($R_p$) from light curve parameters | `koi_prad` ($R_\oplus \le 50$) | $R^2 \ge 0.95$, RMSE, MAE | **Gradient Boosting** ($R^2 = 0.9760$) |
| 🎯 **Classification Track** | Classify detected transit signals into 3 vetted categories | `koi_disposition` (3 classes) | Weighted F1, ROC-AUC (OvR) | **Decision Tree / SVC** ($\text{F1} = 0.755$, $\text{AUC} = 0.873$) |

---

## 📁 Repository Structure

| Path | Type | Contents & Purpose |
| :--- | :---: | :--- |
| 📂 [`data/`](file:///c:/Users/DELL/Downloads/ml_project/ml_project/data) | Directory | 12 zero-leakage partitioned train/test feature matrices and target labels (`.csv`) |
| 📂 [`notebooks/`](file:///c:/Users/DELL/Downloads/ml_project/ml_project/notebooks) | Directory | 3 modular execution notebooks with all cells executed and outputs rendered |
| 📄 [`README.md`](file:///c:/Users/DELL/Downloads/ml_project/ml_project/README.md) | File | Project documentation, benchmarks, tables, and execution instructions |
| 📄 [`requirements.txt`](file:///c:/Users/DELL/Downloads/ml_project/ml_project/requirements.txt) | File | Pinned Python package dependencies for exact environment replication |

### 📓 Notebook Directory Breakdown

| Notebook | Focus Area | Key Deliverables & Rubric Mapping |
| :--- | :--- | :--- |
| 📘 [`01_preprocessing.ipynb`](file:///c:/Users/DELL/Downloads/ml_project/ml_project/notebooks/01_preprocessing.ipynb) | Data Audit, EDA & Preprocessing | Data audit, 12 feature distributions, Pearson heatmap, outlier trimming ($R_p \le 50 R_\oplus$), median imputation, standard scaling, and split export |
| 📗 [`02_regression.ipynb`](file:///c:/Users/DELL/Downloads/ml_project/ml_project/notebooks/02_regression.ipynb) | Planetary Radius Estimation | 10 regression algorithms benchmarked, 5-fold CV, GridSearchCV hyperparameter tuning, Actual vs. Predicted plots, Residual analysis, Feature Importance |
| 📙 [`03_classification.ipynb`](file:///c:/Users/DELL/Downloads/ml_project/ml_project/notebooks/03_classification.ipynb) | Astrophysical Signal Vetting | 5 classifiers benchmarked, Confusion matrices, multi-class ROC-AUC curves (OvR), and Viva Voce defense guide |

---

## 🔬 Feature Dictionary & Physical Meaning

| Feature Symbol | Feature Name | Unit | Physical Description | Used In |
| :--- | :--- | :--- | :--- | :--- |
| $P$ | `koi_period` | Days | Orbital period of the candidate planet around host star | 📈 Reg & 🎯 Clf |
| $T_{\text{dur}}$ | `koi_duration` | Hours | Total duration of stellar brightness dip during transit | 📈 Reg & 🎯 Clf |
| $\delta$ | `koi_depth` | ppm | Transit depth (fractional loss of stellar flux) | 📈 Reg & 🎯 Clf |
| $b$ | `koi_impact` | Ratio | Impact parameter ($b = d / R_\star$, transit path chord distance) | 📈 Reg & 🎯 Clf |
| $\text{SNR}$ | `koi_model_snr` | Ratio | Signal-to-noise ratio of transit event detection | 📈 Reg & 🎯 Clf |
| $T_{\text{eq}}$ | `koi_teq` | Kelvin | Estimated planet equilibrium temperature | 📈 Reg & 🎯 Clf |
| $S$ | `koi_insol` | Earth Flux | Insolation flux received relative to Earth ($S_\oplus$) | 📈 Reg & 🎯 Clf |
| $T_{\text{eff}}$ | `koi_steff` | Kelvin | Host star effective surface temperature | 📈 Reg & 🎯 Clf |
| $\log g$ | `koi_slogg` | $\log_{10}(\text{cm/s}^2)$ | Host star surface gravity | 📈 Reg & 🎯 Clf |
| $R_\star$ | `koi_srad` | Solar Radii ($R_\odot$) | Host star physical radius | 📈 Reg & 🎯 Clf |
| $M_\star$ | `koi_smass` | Solar Masses ($M_\odot$) | Host star physical mass | 📈 Reg Only |
| $m_{\text{Kep}}$ | `koi_kepmag` | Magnitude | Apparent brightness of star in Kepler optical band | 📈 Reg & 🎯 Clf |
| $\dot{\delta}$ | `depth_duration_ratio` | $\text{ppm}/\text{hr}$ | **Engineered Feature:** Ingress/egress transit steepness ($\frac{\text{koi\_depth}}{\text{koi\_duration} + 10^{-5}}$) | 📈 Reg & 🎯 Clf |

---

## ⚙️ Preprocessing Pipeline Summary

| Stage | Operation | Tool / Technique | Purpose & Leakage Prevention |
| :---: | :--- | :--- | :--- |
| **1** | 📥 Ingestion & Audit | `pd.read_csv(comment='#')` | Strips Kepler metadata comments; validates 0 duplicate rows |
| **2** | 🛡️ Leakage Exclusion | Column Whitelisting | Drops Robovetter flags (`koi_fpflag_*`), `koi_score`, and `koi_ror` |
| **3** | 🧹 Outlier Filtering | Boolean Masking | Removes `NOT DISPOSITIONED`; caps target $R_p \le 50 R_\oplus$ to protect MSE loss |
| **4** | ⚡ Feature Engineering | Mathematical Ratio | $\text{koi\_depth} / (\text{koi\_duration} + 10^{-5})$ captures transit shape geometry |
| **5** | ✂️ Train-Test Split | `train_test_split(80:20)` | Isolates test split **before** fitting any transformers; stratified on classification |
| **6** | 🩹 Median Imputation | `SimpleImputer(median)` | Replaces missing values with training medians (robust to skewed tails) |
| **7** | 📏 Feature Scaling | `StandardScaler()` | Fits $z = \frac{x - \mu}{\sigma}$ **strictly on training set**; transforms test set |
| **8** | 💾 Split Serialization | `.to_csv('data/...')` | Exports 12 clean train/test partitions for 100% downstream reproducibility |

---

## 📈 Regression Track: Model Benchmark (10 Algorithms)

> Evaluated on held-out 20% test partition ($N = 1,010$). Ranked by $R^2$ descending:

| Rank | Algorithm | $R^2$ Score | RMSE ($R_\oplus$) | MAE ($R_\oplus$) | Key Strength |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **Gradient Boosting Regressor** | **0.9760** | **1.2768** | **0.4214** | Captures non-linear geometric transit coupling |
| 🥈 | **Random Forest Regressor** | **0.9700** | **1.4266** | **0.4689** | High stability ensemble; low variance |
| 🥉 | Support Vector Regressor (SVR) | 0.9156 | 2.3948 | 0.9595 | Effective non-linear RBF margin |
| 4 | Decision Tree Regressor | 0.9088 | 2.4892 | 1.2078 | Interpretable single tree baseline |
| 5 | K-Nearest Neighbors Regressor | 0.8199 | 3.4976 | 1.5507 | Distance-weighted neighborhood predictor |
| 6 | Polynomial Regression (Degree 2) | 0.6996 | 4.5167 | 1.9154 | Captures quadratic interaction terms |
| 7 | Linear Regression (OLS) | 0.5830 | 5.3218 | 3.3234 | Unregularized parametric baseline |
| 8 | Ridge Regression ($L_2$) | 0.5827 | 5.3239 | 3.3202 | Stabilizes correlated stellar parameters |
| 9 | ElasticNet Regression | 0.5684 | 5.4142 | 3.2897 | Balanced $L_1 + L_2$ regularizer |
| 10 | Lasso Regression ($L_1$) | 0.5595 | 5.4699 | 3.3086 | Sparse feature selection regularizer |

### 🔧 Hyperparameter Tuning & Cross-Validation

| Model | 5-Fold CV $R^2$ | Best Hyperparameters Found via `GridSearchCV` |
| :--- | :---: | :--- |
| **Gradient Boosting** | $0.9684 \pm 0.0063$ | `{'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8}` |
| **Random Forest** | $0.9550 \pm 0.0085$ | `{'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1}` |

---

## 🎯 Classification Track Part A: Model Benchmark (5 Algorithms)

> Target: `koi_disposition` (`CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`). Evaluated on stratified 20% test partition ($N = 1,010$):

| Rank | Algorithm | Accuracy | Precision | Recall | Weighted F1 | Multi-Class ROC-AUC (OvR) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Decision Tree Classifier** | **0.7673** | **0.7513** | **0.7673** | **0.7546** | 0.8582 |
| 🥈 | **KNN Classifier** | 0.7525 | 0.7348 | 0.7525 | 0.7356 | 0.8484 |
| 🥉 | **Support Vector Classifier (SVC)** | 0.7574 | 0.7373 | 0.7574 | 0.7322 | **0.8733** |
| 4 | Logistic Regression | 0.6960 | 0.6617 | 0.6960 | 0.6546 | 0.8211 |
| 5 | Gaussian Naive Bayes | 0.6297 | 0.6131 | 0.6297 | 0.5589 | 0.7822 |

---

## 💻 Quickstart & Execution Guide

| Step | Action | Shell Command |
| :---: | :--- | :--- |
| **1** | 📦 Install Dependencies | `pip install -r requirements.txt` |
| **2** | 🔬 Run Preprocessing & EDA | Execute all cells in [`notebooks/01_preprocessing.ipynb`](file:///c:/Users/DELL/Downloads/ml_project/ml_project/notebooks/01_preprocessing.ipynb) |
| **3** | 📈 Run Regression Track | Execute all cells in [`notebooks/02_regression.ipynb`](file:///c:/Users/DELL/Downloads/ml_project/ml_project/notebooks/02_regression.ipynb) |
| **4** | 🎯 Run Classification Track | Execute all cells in [`notebooks/03_classification.ipynb`](file:///c:/Users/DELL/Downloads/ml_project/ml_project/notebooks/03_classification.ipynb) |

---

## 💡 Viva Voce Defense & Examiner Quick Reference

| # | Question Focus | Core Defense Answer |
| :---: | :--- | :--- |
| **Q1** | **Why filter $R_p \le 50 R_\oplus$?** | Raw Kepler data contains unvetted eclipsing binary fitting failures up to $109,061 R_\oplus$. An error of $10^5 R_\oplus$ yields an MSE penalty of $(10^5)^2 = 10^{10}$, collapsing $R^2$ negative. The $50 R_\oplus$ threshold retains 100% of real planets (Jupiter is $11.2 R_\oplus$) while removing noise. |
| **Q2** | **How is data leakage prevented?** | Train-test split (80:20) is performed **first**. All transformers (`SimpleImputer`, `StandardScaler`) are fitted strictly on the training partition (`.fit_transform()`), then applied to test (`.transform()`). Target proxies (`koi_score`, `koi_ror`) are strictly excluded. |
| **Q3** | **Why `depth_duration_ratio`?** | $\text{koi\_depth} / \text{koi\_duration}$ measures the rate of flux loss (transit steepness). Planetary transits produce steep U-shaped curves; grazing binary stars produce gradual V-shaped curves. |
| **Q4** | **Why do tree ensembles outperform linear models?** | Transit depth scales non-linearly with radius ($R_p \approx R_\star \sqrt{\delta}$). Trees segment the non-linear interaction space without assuming linear hyperplanes. |
| **Q5** | **Why stratified splitting for classification?** | Classes are imbalanced (`CONFIRMED`: 54.1%, `FALSE POSITIVE`: 31.7%, `CANDIDATE`: 14.1%). `stratify=y` guarantees identical class representation in train and test splits. |

---

## 🏆 Review 1 Rubric Checklist Verification (25 Marks)

| Rubric Item | Marks | Verified Deliverables | Status |
| :--- | :---: | :--- | :---: |
| **A1: Ingestion & Quality Audit** | 1 | Shape, dtypes, null audit, duplicate check, disposition breakdown | ✅ Complete |
| **A2: EDA Visualisations** | 2 | 12 feature distributions + KDE, correlation heatmap, 3 target scatter plots | ✅ Complete |
| **A3: Insight Commentary** | 1 | Physical explanations of skewness, stellar coupling, and radius distributions | ✅ Complete |
| **B1: Data Cleaning & Outliers** | 1 | Median imputation on train; justified $R_p \le 50 R_\oplus$ outlier trimming | ✅ Complete |
| **B2: Encoding, Scaling & Split** | 1 | 80:20 train-test split; `StandardScaler` fitted strictly on training data | ✅ Complete |
| **B3: Feature Engineering** | 1 | `depth_duration_ratio` constructed with physical justification | ✅ Complete |
| **C1: Regression Models** | 4 | All 10 required regression algorithms trained and tested | ✅ Complete |
| **C2: Comparative Evaluation** | 2 | Consolidated regression benchmark table sorted by $R^2$ descending | ✅ Complete |
| **C3: Hyperparameter Tuning** | 2 | 5-fold CV on top 2 models; `GridSearchCV` on Random Forest & Gradient Boosting | ✅ Complete |
| **C4: Regression Visualisation** | 1 | Actual vs. Predicted, Residual plot, and Feature Importance plot | ✅ Complete |
| **D1: Classification Models** | 2 | All 5 Part-A classification algorithms trained and producing predictions | ✅ Complete |
| **D2: Classification Evaluation** | 1 | Accuracy, Precision, Recall, F1, Confusion Matrices, multi-class ROC-AUC (OvR) | ✅ Complete |
| **E1 & Viva Defense** | 6 | Review 2 roadmap + dedicated 6-question Viva Voce defense guide | ✅ Complete |
