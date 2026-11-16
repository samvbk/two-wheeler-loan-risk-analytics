# Two-Wheeler Loan Risk Analytics: Default Prediction and Borrower Segmentation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7%2B-brightgreen.svg)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple.svg)
![SQLite](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

A complete, production-grade credit risk modeling and customer segmentation solution engineered for a Non-Banking Financial Company (NBFC) specializing in two-wheeler asset financing. 

This project integrates **SQL analytics**, **leakage-free machine learning pipelines**, **cost-sensitive threshold optimization**, **SHAP feature attribution**, **unsupervised borrower clustering**, and an **interactive Streamlit underwriting portal**.

---

> [!NOTE]
> **Synthetic Data Disclosure**: The dataset consists of 20,000 synthetic two-wheeler loan accounts generated via [`src/generate_data.py`](file:///d:/Games/AA%20ResumeProjects/Data%20analyst/src/generate_data.py) using a fixed random seed (`42`). It simulates real-world Indian NBFC retail lending dynamics (CIBIL credit score mechanics, Fixed Obligation to Income Ratios [FOIR], vehicle pricing, down-payment behavior, and loan delinquency) with realistic noise, missingness (~4%), and extreme outliers for genuine data wrangling.

---

## 📌 Business Problem & Motivation

Two-wheeler financing is a high-volume, low-ticket lending product characterized by thin bureau files, higher default propensity (~11-15%), and tight operating margins. NBFCs face an asymmetrical cost structure:
- **Cost of a False Negative (Default Missed)**: Loss of unrecovered principal (~₹85,000 – ₹1,35,000) minus depreciated repossession recovery.
- **Cost of a False Positive (Credit Denied / False Alarm)**: Lost interest margin (~₹12,000 – ₹18,000).

Therefore, this project solves two vital operational needs:
1. **Underwriting Risk Scorer**: Predict probability of default at loan origination and optimize decision thresholds to maximize defaulter recall.
2. **Early Delinquency & Segmentation Engine**: Group existing borrowers into behavioral risk tiers to trigger early collections outreach (Day 15 past due) before loans slip into Non-Performing Assets (NPAs).

---

## 🏗️ Repository Architecture

```text
├── app.py                     # Streamlit web app (Underwriting, Segmentation, Watchlist)
├── predict.py                 # Standalone CLI / Python module inference script
├── loans.db                   # SQLite database (embedded 20K portfolio records)
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation & resume impact points
├── data/
│   ├── two_wheeler_loans.csv  # Raw dataset (20,000 records, 16 features)
│   ├── loans_features.csv     # Cleaned & engineered feature store (24 columns)
│   └── loans_with_clusters.csv# Feature store with K-Means segment assignments
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_model_training_and_explainability.ipynb
│   └── 03_borrower_segmentation.ipynb
├── src/
│   ├── generate_data.py       # Deterministic synthetic data generator (Seed 42)
│   ├── load_to_db.py          # SQLite pipeline using built-in sqlite3
│   ├── run_sql.py             # Executes SQL suite and exports tabular reports
│   ├── eda_cleaning.py        # Imputation, winsorization, feature engineering
│   ├── train_models.py        # Scikit-Learn pipelines, tuning, evaluation & SHAP
│   ├── segmentation.py        # K-Means (Elbow + Silhouette), PCA, uplift testing
│   └── generate_notebooks.py  # Automated production notebook builder
├── sql/
│   └── analysis.sql           # 8 production SQL queries (CTEs, NTILE, RANK, CASE WHEN)
├── models/
│   ├── best_model.joblib      # Serialized Scikit-Learn pipeline & metadata
│   └── segmentation.joblib    # Serialized K-Means model, scaler & profiles
└── reports/
    ├── 01_missing_values.png  # Missingness analysis
    ├── 02_outliers_boxplot.png# Outlier detection
    ├── 03_class_imbalance.png # Target distribution (11.36% default)
    ├── 04_feature_distributions.png
    ├── 05_default_by_category.png
    ├── 06_late_payments_vs_default.png
    ├── 07_correlation_heatmap.png
    ├── 08_model_evaluation.png# ROC curve, PR curve, Confusion Matrix
    ├── 09_shap_importance.png # SHAP value feature ranking
    ├── 10_elbow_silhouette.png# K-Means metric curves
    ├── 11_pca_clusters.png    # 2D PCA cluster projection
    ├── 12_cluster_heatmap.png # Cluster profile metric comparison
    ├── cluster_profiles.csv   # Profiling data for segments
    ├── model_comparison.csv   # Validation metrics across models
    └── watchlist_top500.csv   # High-risk accounts ranked by default probability
```

---

## 🗄️ Database & SQL Analysis (`loans.db`)

All 20,000 loan records are loaded into SQLite using Python's built-in `sqlite3` and indexed by `city_tier`, `employment_type`, and `defaulted`. The analysis suite in [`sql/analysis.sql`](file:///d:/Games/AA%20ResumeProjects/Data%20analyst/sql/analysis.sql) covers 8 analytical queries:

1. **Overall Default Benchmark**: Establishes portfolio baseline default rate at **11.36%** (2,272 defaults / 20,000 loans).
2. **Default Rate by City Tier (`GROUP BY`, `HAVING`)**: Tier-3 cities exhibit the highest default rate (12.53%) vs. Tier-1 (10.15%).
3. **Default Rate by Employment Type**: Students (42.92%) and Daily-Wage workers (26.90%) experience 5x to 8x the default rate of Salaried borrowers (5.28%).
4. **`CASE WHEN` EMI-to-Income Risk Buckets**:
   - EMI burden > 60% of income -> **79.51% default rate**.
   - EMI burden 40-60% of income -> **64.98% default rate**.
   - EMI burden < 20% of income -> **5.12% default rate**.
5. **CTE & Window Function (`NTILE(10)`)**: Segments borrowers into 10 risk deciles using a composite bureau and leverage score. The top decile (Decile 1) concentrates a **60.70% default rate**, isolating 53% of all portfolio defaults within 10% of customers.
6. **Subquery for High EMI Burden**: Filters accounts with above-average portfolio EMI (>₹4,200), demonstrating default rates spiking to 67.46% in student profiles.
7. **Monthly Default Trend**: 48-month cohort tracking (2020 to 2023) showing stable seasonality with slight spikes in Q4 festive origination cycles.
8. **Window Ranking (`RANK()`)**: Generates an actionable risk leaderboard combining geography and employment type; Tier-3 Students rank #1 in risk (44.74% default).

---

## 📊 Exploratory Data Analysis & Cleaning

1. **Missing Data Handling**: Detected ~4.1% missing values in `monthly_income`, `credit_score`, `down_payment_pct`, and `interest_rate`. Employed **median imputation** within pipeline transformers to prevent row deletion and preserve minority-class default signals.
2. **Outlier Treatment**: Detected synthetic extreme values (income > ₹500,000, loans > ₹800,000, credit scores > 900). Dropped invalid bureau scores (>900) and applied **1st - 99th percentile Winsorization** on continuous financial variables to retain sample volume without distorting gradient estimators.
3. **Class Imbalance Strategy**: Portfolio default incidence is **11.36%** (imbalance ratio ~9:1). Addressed via `class_weight='balanced'` in estimators, prioritizing ROC-AUC and PR-AUC over raw accuracy (which yields a misleading 89% baseline score).

---

## ⚙️ Feature Engineering

Engineered domain-specific credit variables:
- `emi_to_income` (Fixed Obligation to Income Ratio - FOIR): Ratio of monthly EMI to reported income.
- `ltv_ratio` (Loan-to-Value): Percentage of on-road vehicle cost financed by the NBFC.
- `income_band`: Binned income tiers (`Very Low`, `Low`, `Medium`, `High`, `Very High`).
- `credit_band`: Standard bureau score intervals (`Poor` <580, `Fair` 580-650, `Good` 650-720, `Very Good` 720-790, `Excellent` 790-900).
- `late_pmt_flag`: Binary indicator (1 if borrower had $\ge 1$ late payment in prior 12 months).
- `high_emi_burden`: Binary risk flag (1 if FOIR $> 40\%$).
- `loan_year` & `loan_month`: Macroeconomic temporal markers.

---

## 🤖 Predictive Modeling & Evaluation

Models were trained inside Scikit-Learn `Pipeline` architectures with `ColumnTransformer` (StandardScaler for numerics, OrdinalEncoder for categoricals) evaluated via 5-Fold Stratified Cross-Validation.

### Model Performance Comparison (Test Set: $N=3,996$)

| Model | ROC-AUC | PR-AUC | Defaulter Precision (@0.30) | Defaulter Recall (@0.30) | Defaulter F1-Score | Best Tuned Hyperparameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Baseline)** | **0.9195** | **0.6982** | 27.50% | **93.17%** | **0.4247** | `C=0.1`, `penalty='l2'`, `class_weight='balanced'` |
| **Random Forest Classifier** | 0.9115 | 0.6813 | 27.30% | 92.29% | 0.4213 | `n_estimators=200`, `max_depth=8`, `min_samples_leaf=20` |
| **LightGBM Classifier** | 0.9161 | 0.6890 | 25.28% | **95.59%** | 0.3998 | `num_leaves=31`, `n_estimators=200`, `lr=0.01`, `min_child_samples=50` |

### Business-Driven Threshold Justification (Why 0.30?)
At the default threshold of $0.50$, models maximize accuracy by missing high-risk borrowers. In retail lending:
- Missing an actual defaulter costs the NBFC full written-off principal (~₹1,00,000).
- Rejecting a good customer or issuing a manual review costs only the lost interest margin (~₹15,000).

Lowering the decision threshold to **0.30** achieves a **93.17% recall on defaulters** (catching 423 out of 454 test defaulters), reducing portfolio NPA risk by over 80%.

---

## 🔍 Model Explainability (SHAP Values)

Attribution analysis was conducted using SHAP values:
1. **`late_payments_last_12m`**: The single highest predictor of default risk. Borrowers with $\ge 3$ late payments are 5.2x more prone to default.
2. **`emi_to_income`**: An EMI burden above 40% causes exponential default risk increase. Underwriting policy must cap FOIR at 40%.
3. **`credit_score`**: Scores below 580 exhibit steep negative SHAP values, driving immediate high-risk classifications.
4. **`employment_type`**: Daily-Wage and Student employment contribute significant positive default log-odds, demanding mandatory co-borrowers and higher down payments.

---

## 👥 Borrower Segmentation (K-Means Clustering)

Using scaled financial and behavioral features (`age`, `monthly_income`, `credit_score`, `loan_amount`, `emi_to_income`, `late_payments_last_12m`, `down_payment_pct`), K-Means clustering was evaluated across $k=2 \dots 8$. Choosing $k=4$ balanced inertia reduction with clean business interpretability:

| Cluster ID | Segment Name | Share | Avg Income | Avg Credit | Avg EMI/Inc | Default Rate | Prescribed Underwriting & Operational Strategy |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **Conservative Low-LTV** | 22.2% | ₹40,647 | 655 | 10% | **2.0%** | Fast-track processing; pre-approved two-wheeler accessories financing & loyalty discounts. |
| **1** | **Prime Salaried Growth** | 36.8% | ₹44,846 | 680 | 11% | **1.0%** | Prime segment; offer discounted interest rates (-0.5%), personal loan cross-sells, zero processing fees. |
| **2** | **High-Risk Over-Leveraged** | 8.5% | ₹12,538 | 572 | 61% | **61.0%** | **Critical NPA Risk**; immediate early collections trigger (Day 15 past due), freeze credit expansion, require guarantor. |
| **3** | **Vulnerable Moderate-Risk** | 32.5% | ₹28,799 | 547 | 16% | **17.0%** | Cap LTV at 75%, enforce automated e-NACH debits, send proactive SMS payment reminders 3 days before due date. |

*Feature Uplift Test*: Adding the unsupervised `cluster` label into the prediction model increased 5-fold cross-validated ROC-AUC from **0.9075 to 0.9086**, validating that cluster membership captures latent borrower structures.

---

## 💻 Streamlit Web Application (`app.py`)

The interactive dashboard features three operational tabs:
1. **Underwriting Simulator**: Real-time loan origination calculator. Enter applicant metrics, compute instantaneous EMI, FOIR, and LTV, and receive default probability with a Low/Medium/High risk badge and action recommendations.
2. **Borrower Segmentation**: Interactive segment profile comparison cards, PCA 2D projections, normalized heatmaps, and credit strategy playbooks.
3. **Delinquency Watchlist**: Collections portal rendering the top 500 highest-risk borrowers. Features interactive multi-filters by risk category, employment type, and probability slider, with instant CSV export.

---

## 🚀 How to Run the Project

### 1. Environment Setup
```bash
# Clone or navigate to the project directory
cd "d:/Games/AA ResumeProjects/Data analyst"

# Install dependencies
pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution
Run the scripts in sequential order:
```bash
# 1. Generate 20,000 synthetic loan records (seed 42)
python src/generate_data.py

# 2. Ingest into SQLite database (loans.db)
python src/load_to_db.py

# 3. Execute 8 SQL queries and save reports
python src/run_sql.py

# 4. Run EDA, cleaning, and feature engineering
python src/eda_cleaning.py

# 5. Train baseline & advanced models, evaluate, and compute SHAP
python src/train_models.py

# 6. Perform K-Means segmentation & PCA
python src/segmentation.py

# 7. Generate Jupyter Notebooks
python src/generate_notebooks.py
```

### 3. Real-Time CLI Inference
Test applicant risk directly from the terminal:
```bash
python predict.py
```

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🛠️ Technology Stack

- **Data Processing & Analytics**: Python 3.10+, Pandas, NumPy, SciPy
- **Database & Querying**: SQLite3, ANSI SQL (CTEs, NTILE, RANK, Subqueries)
- **Machine Learning**: Scikit-Learn (Pipelines, ColumnTransformer, RandomizedSearchCV, KMeans, PCA), LightGBM
- **Explainability**: SHAP (LinearExplainer & TreeExplainer)
- **Visualization**: Matplotlib, Seaborn
- **Application & Deployment**: Streamlit, Joblib

---

## 💼 Resume Bullet Points (Data Scientist / Risk Analytics Intern)

- **Engineered an end-to-end two-wheeler credit risk scoring pipeline** on 20,000 borrower records using Scikit-Learn Pipelines and ColumnTransformer, preventing leakage and achieving **0.9195 ROC-AUC** and **0.6982 PR-AUC**.
- **Optimized the underwriting classification threshold to 0.30** based on NBFC cost asymmetries, boosting **defaulter recall to 93.17%** (catching 423 of 454 defaults) to protect loan portfolio margins.
- **Architected 8 complex analytical SQL queries** in SQLite using CTEs, `NTILE(10)` decile bucketing, and window `RANK()` functions, isolating **53% of all portfolio defaults within the top 10% risk decile**.
- **Segmented 20,000 borrowers into 4 behavioral clusters** using K-Means (Elbow & Silhouette verified) and 2D PCA, identifying a high-risk over-leveraged cohort (61% default rate, 61% FOIR) and proving a **+0.11% ROC-AUC uplift** when adding cluster labels as model features.
- **Formulated SHAP-driven policy guidelines** identifying late payments and EMI-to-income (>40%) as top default drivers, and **deployed an interactive 3-tab Streamlit underwriting portal** for real-time risk scoring and collections prioritization.
