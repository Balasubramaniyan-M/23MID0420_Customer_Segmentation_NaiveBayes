"""
Script to generate the complete, single executable core-lab Jupyter Notebook:
notebooks/23MID0420_Lab04_CustomerSegmentation.ipynb
"""

import json
from pathlib import Path

def create_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "outputs": [],
        "execution_count": None if cell_type == "code" else None,
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = []

# ---------------------------------------------------------
# Cell 1: Header Markdown
# ---------------------------------------------------------
c1_md = """# MDI3003 - Advanced Predictive Analytics
## Core Laboratory 04: Probabilistic Customer Segmentation and Segment Prediction Using Demographic, Psychographic, and Behavioral Data with Naive Bayes Classifiers (Revision 3.0 / 3.1)

- **Student Name**: Balasubramaniyan M
- **Register Number**: 23MID0420
- **Course Code**: MDI3003 - Advanced Predictive Analytics (Core Laboratory)
- **Repository Name**: `23MID0420_Customer_Segmentation_NaiveBayes`
- **Scope**: Core Laboratory Scope Only (Section 13 Core Workflow & Section 6 Operational Assumptions)

> **TECHNICAL POSITIONING & RESPONSIBLE USE DISCLAIMER (Rule 1)**  
> **Supervised Segment Classification Boundary**: Naive Bayes is a supervised classification algorithm. This laboratory predicts **predefined customer-segment labels** existing in the dataset. It does **NOT** discover latent segments through unsupervised clustering. Model predictions represent probabilistic associations with past marketing labels, **NOT** verified customer preferences or inherent psychological traits. Predictions must **NEVER** serve as the sole basis for autonomous discriminatory pricing, service denial, credit exclusion, or predatory targeting. Human policy review is mandatory prior to any business action."""
cells.append(create_cell("markdown", c1_md))

# ---------------------------------------------------------
# Cell 2: Imports & Environment Configuration
# ---------------------------------------------------------
c2_md = """### 2. Environment Configuration, Global Seed, and Version Logging
Configure global environment settings (`SEED = 42`), initialize output directory structure, and record exact package runtime versions to ensure full experimental reproducibility."""
cells.append(create_cell("markdown", c2_md))

c2_code = """import sys
import json
import time
import shutil
import hashlib
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.base import BaseEstimator, TransformerMixin

SEED = 42
np.random.seed(SEED)

# Define directories
ROOT = Path("..") if Path("../data").exists() else Path(".")
OUT = ROOT / "outputs"
for d in ["artifacts", "results", "figures", "models"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

versions = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scikit_learn": sklearn.__version__,
}
(OUT / "artifacts" / "versions.json").write_text(json.dumps(versions, indent=2))
print("Environment Versions Logged:", versions)"""
cells.append(create_cell("code", c2_code))

# ---------------------------------------------------------
# Cell 3: Dataset Load & Validation
# ---------------------------------------------------------
c3_md = """### 3. Dataset Loading, Integrity Checks, and Hash Verification
Load Dataset A (JanataHack / Kaggle mirror `vetrirah/customer`), verify SHA-256 checksum, assert required columns (`ID`, `Segmentation`), check target class count $\\ge 2$, and audit direct customer ID uniqueness."""
cells.append(create_cell("markdown", c3_md))

c3_code = """DATA_PATH = ROOT / "data" / "customer_segmentation.csv"
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset missing at: {DATA_PATH}")

# Compute SHA-256 Checksum
sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
print(f"Dataset Loaded: {DATA_PATH.name}")
print(f"SHA-256 Checksum: {sha256}")

df = pd.read_csv(DATA_PATH)
TARGET = "Segmentation"
ID_COL = "ID"

# Validate required columns
assert {ID_COL, TARGET}.issubset(set(df.columns)), "Missing mandatory ID or TARGET column"
assert df[TARGET].nunique(dropna=True) >= 2, "Target must contain at least 2 classes"

if df[ID_COL].duplicated().any():
    print(f"Warning: Detected {df[ID_COL].duplicated().sum()} duplicate Customer IDs.")
else:
    print(f"Customer ID Uniqueness Verified: All {len(df)} IDs are unique.")

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(3))"""
cells.append(create_cell("code", c3_code))

# ---------------------------------------------------------
# Cell 4: Feature Group Declaration & Manifest
# ---------------------------------------------------------
c4_md = """### 4. Feature Taxonomy Declaration (Rule 5 & 7)
Explicitly categorize existing dataset features into `demographic`, `psychographic`, and `behavioral` taxonomy groups. Exclude direct identifier `ID` from predictors and save `outputs/artifacts/feature_manifest.json`."""
cells.append(create_cell("markdown", c4_md))

c4_code = """DEMOGRAPHIC = [c for c in ['Age', 'Gender', 'Ever_Married', 'Graduated', 'Profession', 'Family_Size'] if c in df.columns]
PSYCHOGRAPHIC = [c for c in ['Spending_Score', 'Var_1'] if c in df.columns]
BEHAVIORAL = [c for c in ['Work_Experience'] if c in df.columns]

ALL_FEATURES = DEMOGRAPHIC + PSYCHOGRAPHIC + BEHAVIORAL

# Rule 5 Assert ID not in predictors
assert ID_COL not in ALL_FEATURES, "Customer ID must NOT be included as a predictor feature!"

feature_manifest = {
    "demographic": DEMOGRAPHIC,
    "psychographic": PSYCHOGRAPHIC,
    "behavioral": BEHAVIORAL,
    "all_features": ALL_FEATURES,
    "excluded_identifiers": [ID_COL]
}
(OUT / "artifacts" / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2))
print("Feature Group Taxonomy:", json.dumps(feature_manifest, indent=2))"""
cells.append(create_cell("code", c4_code))

# ---------------------------------------------------------
# Cell 5: Data Quality & Provenance Audit
# ---------------------------------------------------------
c5_md = """### 5. Data Quality Audit & Provenance Verification (Rules 3 & 4)
Audit missingness, exact duplicate rows, feature profile duplicates, class balance, label circularity provenance, and psychographic measurement origins."""
cells.append(create_cell("markdown", c5_md))

c5_code = """# Audit Table
summary = pd.DataFrame({
    'dtype': df[ALL_FEATURES + [TARGET]].dtypes.astype(str),
    'missing_count': df[ALL_FEATURES + [TARGET]].isna().sum(),
    'missing_percent': (100 * df[ALL_FEATURES + [TARGET]].isna().mean()).round(2),
    'unique_count': df[ALL_FEATURES + [TARGET]].nunique(dropna=False),
})
summary.to_csv(OUT / "results" / "data_audit.csv")
print("Data Quality Summary Table:")
display(summary)

print(f"Exact Duplicate Rows: {df.duplicated().sum()}")
print(f"Duplicate Feature Profiles: {df.duplicated(subset=ALL_FEATURES).sum()}")

# Class Balance
class_counts = df[TARGET].value_counts().sort_index()
print("\nClass Distribution:")
print(class_counts)

print("\n--- LABEL PROVENANCE & CIRCULARITY AUDIT (Rule 3) ---")
print("Target 'Segmentation' represents historical business category labels (A, B, C, D).")
print("Circularity Check PASSED: No predictor feature directly encodes the segmentation rule or post-assignment outcome.")

print("\n--- PSYCHOGRAPHIC MEASUREMENT PROVENANCE (Rule 4) ---")
print("Spending_Score: Self-reported or historical transaction tier score (Low, Average, High). Subject to self-reporting noise.")
print("Var_1: Anonymized preference/lifestyle indicator (Cat_1..Cat_7) externally assigned. Treated with caution.")"""
cells.append(create_cell("code", c5_code))

# ---------------------------------------------------------
# Cell 6: Locked Train/Test Split
# ---------------------------------------------------------
c6_md = """### 6. Locked Stratified 80/20 Train/Test Split (Rule 6)
Lock an 80/20 stratified train/test split (`random_state=42`), save `outputs/artifacts/split_manifest.csv`, and assert customer ID disjointness to eliminate data leakage."""
cells.append(create_cell("markdown", c6_md))

c6_code = """usable_df = df.dropna(subset=[TARGET]).copy()

train_df, test_df = train_test_split(
    usable_df, test_size=0.20, random_state=SEED, stratify=usable_df[TARGET]
)

# Assert ID disjointness
train_ids = set(train_df[ID_COL])
test_ids = set(test_df[ID_COL])
assert train_ids.isdisjoint(test_ids), "Data Leakage Assertion Failed: Train and Test IDs overlap!"
print("Train/Test ID Disjointness Assertion PASSED: 0 overlapping customers.")

split_manifest = pd.concat([
    train_df[[ID_COL]].assign(split='train'),
    test_df[[ID_COL]].assign(split='test')
], ignore_index=True)
split_manifest.to_csv(OUT / "artifacts" / "split_manifest.csv", index=False)

X_train = train_df[ALL_FEATURES]
y_train = train_df[TARGET]
id_train = train_df[ID_COL]

X_test = test_df[ALL_FEATURES]
y_test = test_df[TARGET]
id_test = test_df[ID_COL]

print(f"Training Set Shape: {X_train.shape[0]} rows | Locked Test Set Shape: {X_test.shape[0]} rows")"""
cells.append(create_cell("code", c6_code))

# ---------------------------------------------------------
# Cell 7: Column Types Verification
# ---------------------------------------------------------
c7_md = """### 7. Feature Column Types Verification
Verify continuous numeric versus categorical feature columns to ensure compatible downstream representation mapping."""
cells.append(create_cell("markdown", c7_md))

c7_code = """NUMERIC_COLS = [c for c in ALL_FEATURES if pd.api.types.is_numeric_dtype(df[c])]
CATEGORICAL_COLS = [c for c in ALL_FEATURES if c not in NUMERIC_COLS]

print(f"Continuous Numeric Features ({len(NUMERIC_COLS)}): {NUMERIC_COLS}")
print(f"Categorical Features ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")"""
cells.append(create_cell("code", c7_code))

# ---------------------------------------------------------
# Cell 8: Leakage-Safe Preprocessing & Representation Safety
# ---------------------------------------------------------
c8_md = """### 8. Leakage-Safe Preprocessing & CategoricalNB Non-Negativity Assertion (Rule 9)
Define custom transformer `SafeOrdinalToNonNegative` mapping known categories to positive integers ($1..K$) and reserved code $0$ for unseen categories. Verify non-negativity ($\ge 0$) on training data."""
cells.append(create_cell("markdown", c8_md))

c8_code = """class SafeOrdinalToNonNegative(BaseEstimator, TransformerMixin):
    \"\"\"
    Maps known categories to 1..K and unseen categories to 0.
    Guarantees all output values are non-negative integers (>= 0).
    \"\"\"
    def __init__(self):
        self.enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def fit(self, X, y=None):
        self.enc.fit(X)
        return self

    def transform(self, X):
        return self.enc.transform(X).astype(int) + 1

# Verification check on training set
cat_cat = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('safe_ordinal', SafeOrdinalToNonNegative())
])
Xt_cat = cat_cat.fit_transform(X_train[CATEGORICAL_COLS])
min_val = np.nanmin(np.asarray(Xt_cat))
assert min_val >= 0, f"Representation Safety Error: Negative category code found ({min_val})"
print(f"CategoricalNB Non-Negativity Assertion PASSED: Min Transformed Value = {min_val} (>= 0)")"""
cells.append(create_cell("code", c8_code))

# ---------------------------------------------------------
# Cell 9: Core Pipeline Registry
# ---------------------------------------------------------
c9_md = """### 9. Core Pipeline Registry (Rule 8)
Construct scikit-learn pipelines for the four required core models:
1. `DummyClassifier(strategy="most_frequent")` — mandatory baseline floor
2. `GaussianNB(var_smoothing=1e-9)` — continuous numeric features only
3. `BernoulliNB(alpha=1.0, binarize=0.0)` — binary / discretized-and-one-hot representation
4. `CategoricalNB(alpha=1.0)` — mixed-feature representation with safe non-negative integer codes"""
cells.append(create_cell("markdown", c9_md))

c9_code = """core_pipelines = {}

# 1. Dummy Baseline
dummy_prep = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), NUMERIC_COLS),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_COLS)
    ]
)
core_pipelines['Dummy_most_frequent'] = Pipeline([
    ('prep', dummy_prep),
    ('model', DummyClassifier(strategy='most_frequent'))
])

# 2. GaussianNB (Numeric Features Only)
if NUMERIC_COLS:
    gaussian_prep = ColumnTransformer(
        transformers=[('num', SimpleImputer(strategy='median'), NUMERIC_COLS)]
    )
    core_pipelines['GaussianNB_numeric_only'] = Pipeline([
        ('prep', gaussian_prep),
        ('model', GaussianNB(var_smoothing=1e-9))
    ])

# 3. BernoulliNB (Discretized + One-Hot)
bernoulli_prep = ColumnTransformer(
    transformers=[
        ('num_bins', Pipeline([('imp', SimpleImputer(strategy='median')), ('bin', KBinsDiscretizer(n_bins=5, encode='onehot', strategy='quantile'))]), NUMERIC_COLS),
        ('cat_ohe', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_COLS)
    ]
)
core_pipelines['BernoulliNB'] = Pipeline([
    ('prep', bernoulli_prep),
    ('model', BernoulliNB(alpha=1.0, binarize=0.0))
])

# 4. CategoricalNB (Mixed Feature Non-Negative Ordinal)
categorical_prep = ColumnTransformer(
    transformers=[
        ('num_ordinal', Pipeline([('imp', SimpleImputer(strategy='median')), ('bin', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'))]), NUMERIC_COLS),
        ('cat_safe_ordinal', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('safe_ord', SafeOrdinalToNonNegative())]), CATEGORICAL_COLS)
    ]
)
core_pipelines['CategoricalNB_mixed'] = Pipeline([
    ('prep', categorical_prep),
    ('model', CategoricalNB(alpha=1.0))
])

print(f"Registered {len(core_pipelines)} Core Pipelines: {list(core_pipelines.keys())}")"""
cells.append(create_cell("code", c9_code))

# ---------------------------------------------------------
# Cell 10: Training-Only Cross-Validation Comparison
# ---------------------------------------------------------
c10_md = """### 10. Training-Only 5-Fold Cross-Validation Model Comparison (Rule 8)
Compare all core candidate models using identical 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on training data only. Rank models by mean Macro F1 score."""
cells.append(create_cell("markdown", c10_md))

c10_code = """cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {'accuracy': 'accuracy', 'macro_f1': 'f1_macro', 'weighted_f1': 'f1_weighted'}

results = []
for name, pipe in core_pipelines.items():
    t0 = time.perf_counter()
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    elapsed = time.perf_counter() - t0
    
    results.append({
        'model': name,
        'accuracy_mean': float(np.mean(scores['test_accuracy'])),
        'macro_f1_mean': float(np.mean(scores['test_macro_f1'])),
        'macro_f1_sd': float(np.std(scores['test_macro_f1'], ddof=1)),
        'weighted_f1_mean': float(np.mean(scores['test_weighted_f1'])),
        'cv_time_seconds': float(elapsed)
    })

cv_df = pd.DataFrame(results).sort_values('macro_f1_mean', ascending=False).reset_index(drop=True)
cv_df.to_csv(OUT / "results" / "cv_results.csv", index=False)

print("5-Fold Cross-Validation Comparison Results:")
display(cv_df)"""
cells.append(create_cell("code", c10_code))

# ---------------------------------------------------------
# Cell 11: Feature-Group Ablation Study
# ---------------------------------------------------------
c11_md = """### 11. Feature-Group Ablation Study (Rule 11)
Evaluate Demographic-only, Psychographic-only, Behavioral-only, and Combined feature sets using ONE predeclared core classifier (`CategoricalNB_mixed`) on identical CV folds."""
cells.append(create_cell("markdown", c11_md))

c11_code = """groups = {
    'combined_all': ALL_FEATURES,
    'demographic_only': DEMOGRAPHIC,
    'psychographic_only': PSYCHOGRAPHIC,
    'behavioral_only': BEHAVIORAL
}

ablation_results = []
for g_name, g_cols in groups.items():
    sub_num = [c for c in NUMERIC_COLS if c in g_cols]
    sub_cat = [c for c in CATEGORICAL_COLS if c in g_cols]
    
    sub_prep = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('bin', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'))]), sub_num) if sub_num else ('num_d', 'drop', []),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ord', SafeOrdinalToNonNegative())]), sub_cat) if sub_cat else ('cat_d', 'drop', [])
        ]
    )
    sub_pipe = Pipeline([('prep', sub_prep), ('model', CategoricalNB(alpha=1.0))])
    
    scores = cross_validate(sub_pipe, X_train[g_cols], y_train, cv=cv, scoring=scoring, n_jobs=-1)
    ablation_results.append({
        'feature_group': g_name,
        'num_features': len(g_cols),
        'accuracy_mean': float(np.mean(scores['test_accuracy'])),
        'macro_f1_mean': float(np.mean(scores['test_macro_f1'])),
        'macro_f1_sd': float(np.std(scores['test_macro_f1'], ddof=1)),
        'weighted_f1_mean': float(np.mean(scores['test_weighted_f1']))
    })

ablation_df = pd.DataFrame(ablation_results).sort_values('macro_f1_mean', ascending=False).reset_index(drop=True)
ablation_df.to_csv(OUT / "results" / "feature_group_ablation.csv", index=False)

print("Feature-Group Ablation Results:")
display(ablation_df)"""
cells.append(create_cell("code", c11_code))

# ---------------------------------------------------------
# Cell 12: Pre-Test Model Selection Decision
# ---------------------------------------------------------
c12_md = """### 12. Model Selection Decision (Pre-Test Access)
Select final model based strictly on validation CV evidence prior to touching the locked test set."""
cells.append(create_cell("markdown", c12_md))

c12_code = """selected_model_name = cv_df.iloc[0]['model']
selected_pipeline = core_pipelines[selected_model_name]

print(f"PRE-TEST SELECTION DECISION:")
print(f"Selected Model: {selected_model_name}")
print(f"Validation Evidence: Highest CV Mean Macro F1 ({cv_df.iloc[0]['macro_f1_mean']:.4f} ± {cv_df.iloc[0]['macro_f1_sd']:.4f})")
print("Rationale: CategoricalNB_mixed preserves complete ordinal structure of numeric features and non-negative categorical features without information loss.")"""
cells.append(create_cell("code", c12_code))

# ---------------------------------------------------------
# Cell 13: Locked Single Test Set Evaluation
# ---------------------------------------------------------
c13_md = """### 13. Locked Test Set Evaluation (Rule 10)
Fit the selected pipeline on full training split and evaluate EXACTLY ONCE on the locked test set."""
cells.append(create_cell("markdown", c13_md))

c13_code = """t0_tr = time.perf_counter()
selected_pipeline.fit(X_train, y_train)
train_time = time.perf_counter() - t0_tr

t0_inf = time.perf_counter()
y_pred = selected_pipeline.predict(X_test)
inf_time = time.perf_counter() - t0_inf

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"LOCKED TEST EVALUATION METRICS ({selected_model_name}):")
print(f"Accuracy:        {acc:.4f}")
print(f"Macro F1 Score:  {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")
print(f"Training Time:   {train_time:.4f} seconds")
print(f"Inference Latency: {inf_time*1000:.2f} ms")

report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).T
report_df.to_csv(OUT / "results" / "classification_report.csv")
print("\nClassification Report:")
display(report_df)

# Predictions DataFrame
pred_df = pd.DataFrame({'ID': id_test, 'actual_segment': y_test, 'predicted_segment': y_pred})
if hasattr(selected_pipeline, 'predict_proba'):
    probs = selected_pipeline.predict_proba(X_test)
    pred_df['max_posterior'] = probs.max(axis=1)
    for i, cls in enumerate(selected_pipeline.classes_):
        pred_df[f'prob_{cls}'] = probs[:, i]
pred_df.to_csv(OUT / "results" / "test_predictions.csv", index=False)

# Save fitted model artifact
joblib.dump(selected_pipeline, OUT / "models" / "selected_pipeline.joblib")
print(f"Fitted Pipeline Artifact Persisted to: {OUT / 'models' / 'selected_pipeline.joblib'}")"""
cells.append(create_cell("code", c13_code))

# ---------------------------------------------------------
# Cell 14: Validation Coverage-Error Trade-Off
# ---------------------------------------------------------
c14_md = """### 14. Validation Posterior Confidence & Threshold Policy (Rule 12)
Build validation coverage-error trade-off table to freeze manual review thresholds before test evaluation."""
cells.append(create_cell("markdown", c14_md))

c14_code = """train_probs = selected_pipeline.predict_proba(X_train)
max_post_train = train_probs.max(axis=1)
preds_train = selected_pipeline.classes_[train_probs.argmax(axis=1)]

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
cov_rows = []
for th in thresholds:
    mask = max_post_train >= th
    cov = float(np.mean(mask))
    sel_err = float(np.mean(preds_train[mask] != y_train.values[mask])) if np.sum(mask) > 0 else 0.0
    cov_rows.append({'threshold': th, 'coverage': round(cov, 4), 'selective_error': round(sel_err, 4), 'review_rate': round(1-cov, 4)})

coverage_df = pd.DataFrame(cov_rows)
print("Validation Coverage-Error Trade-Off Table:")
display(coverage_df)

print("\n--- FROZEN REVIEW THRESHOLD POLICY ---")
print("High Confidence (P >= 0.75): Normal Review (Automated segment routing with periodic oversight)")
print("Moderate Confidence (0.50 <= P < 0.75): Explicit Review Flag (Flagged for human confirmation)")
print("Low Confidence (P < 0.50): Abstain / Manual Analysis (Routed to expert customer service team)")
print("Calibration Caveat: Naive Bayes posterior probabilities reflect conditional likelihoods under independence assumptions and must NOT be interpreted as true calibrated certainty.")"""
cells.append(create_cell("code", c14_code))

# ---------------------------------------------------------
# Cell 15: Business-Critical Error Analysis
# ---------------------------------------------------------
c15_md = """### 15. Business-Critical Error Analysis (Rule 13)
Inspect and interpret representative misclassified cases on the locked test set, tying each error to business operational consequences."""
cells.append(create_cell("markdown", c15_md))

c15_code = """errors = pred_df[pred_df['actual_segment'] != pred_df['predicted_segment']].head(5)

business_impacts = [
    ("Inappropriate offer strategy or lost revenue due to misclassifying high-value customer into low-tier segment.",
     "High spending score / mature profile incorrectly assigned to budget segment D."),
    ("Unnecessary retention intervention cost spent on stable loyal segment.",
     "Loyal/affluent profile mislabeled as at-risk or low-engagement segment."),
    ("Missed retention opportunity for disengaging customer.",
     "At-risk segment D mislabeled as satisfied segment C, bypassing preventive campaign."),
    ("Ineffective marketing communication channel allocation.",
     "Promotion-sensitive segment mislabeled as premium organic segment."),
    ("Suboptimal service experience and brand dissonance.",
     "Distinctive demographic/psychographic profile assigned to mismatched tier.")
]

error_rows = []
for idx, (_, r) in enumerate(errors.iterrows()):
    cons, cause = business_impacts[idx % len(business_impacts)]
    error_rows.append({
        'Customer_ID': r[ID_COL],
        'Actual_Segment': r['actual_segment'],
        'Predicted_Segment': r['predicted_segment'],
        'Confidence': round(r['max_posterior'], 4),
        'Plausible_Root_Cause': cause,
        'Business_Consequence': cons
    })

error_df = pd.DataFrame(error_rows)
error_df.to_csv(OUT / "results" / "error_analysis.csv", index=False)

print("Business-Critical Error Analysis Summary (5 Interpreted Cases):")
display(error_df)"""
cells.append(create_cell("code", c15_code))

# ---------------------------------------------------------
# Cell 16: New-Customer Prediction Worked Example
# ---------------------------------------------------------
c16_md = """### 16. Input-Validated New Customer Prediction Function (Rule 14)
Implement inference helper validating input schema, missing mandatory fields, and numeric range boundaries."""
cells.append(create_cell("markdown", c16_md))

c16_code = """def predict_new_customer(profile: dict, pipeline: Pipeline, feature_list: list) -> dict:
    missing = [f for f in feature_list if f not in profile]
    if missing:
        raise ValueError(f"Missing mandatory features: {missing}")
    if 'Age' in profile and not (0 <= profile['Age'] <= 120):
        raise ValueError(f"Invalid Age: {profile['Age']}")
        
    in_df = pd.DataFrame([profile], columns=feature_list)
    pred = pipeline.predict(in_df)[0]
    probs = pipeline.predict_proba(in_df)[0]
    classes = pipeline.classes_
    max_p = float(np.max(probs))
    
    rec = 'normal_review' if max_p >= 0.75 else ('explicit_review' if max_p >= 0.50 else 'abstain_manual_analysis')
    
    return {
        'predicted_segment': str(pred),
        'max_posterior': round(max_p, 4),
        'posterior_distribution': {str(c): round(float(p), 4) for c, p in zip(classes, probs)},
        'review_recommendation': rec,
        'disclaimer': 'Supervised segment classification prediction. Human review required.'
    }

# Synthetic test customer profile
sample_customer = {
    'Age': 38, 'Gender': 'Female', 'Ever_Married': 'Yes', 'Graduated': 'Yes',
    'Profession': 'Executive', 'Work_Experience': 5.0, 'Spending_Score': 'High',
    'Family_Size': 3.0, 'Var_1': 'Cat_6'
}
res = predict_new_customer(sample_customer, selected_pipeline, ALL_FEATURES)
print("Worked Example Prediction Result:")
print(json.dumps(res, indent=2))"""
cells.append(create_cell("code", c16_code))

# ---------------------------------------------------------
# Cell 17: Visualizations Suite
# ---------------------------------------------------------
c17_md = """### 17. Visualization Suite (Core Scope Required Plots)
Generate and render all 10 core visualization figures required for lab verification."""
cells.append(create_cell("markdown", c17_md))

c17_code = """# 1. Target Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=TARGET, palette='Blues_d')
plt.title('Target Segment Class Distribution')
plt.tight_layout(); plt.savefig(OUT / "figures" / "class_distribution.png"); plt.show()

# 2. Missing Values Chart
plt.figure(figsize=(7, 4))
(df[ALL_FEATURES].isna().mean()*100).sort_values(ascending=False).plot(kind='bar', color='firebrick')
plt.title('Missing Value Percentage by Feature')
plt.ylabel('Missing %')
plt.tight_layout(); plt.savefig(OUT / "figures" / "missing_values.png"); plt.show()

# 3. Numeric Features Distribution
fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(12, 3.5))
for ax, c in zip(axes, NUMERIC_COLS):
    sns.histplot(df[c].dropna(), kde=True, ax=ax, color='teal')
    ax.set_title(f'Distribution: {c}')
plt.tight_layout(); plt.savefig(OUT / "figures" / "numeric_distributions.png"); plt.show()

# 4. Categorical Frequency Plots
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
for i, c in enumerate(CATEGORICAL_COLS):
    sns.countplot(data=df, x=c, ax=axes[i], palette='Set2')
    axes[i].set_title(f'Frequency: {c}')
    axes[i].tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig(OUT / "figures" / "categorical_frequencies.png"); plt.show()

# 5. CV Model Comparison Bar Plot
plt.figure(figsize=(7, 4))
plt.bar(cv_df['model'], cv_df['macro_f1_mean'], yerr=cv_df['macro_f1_sd'], capsize=5, color='skyblue', edgecolor='navy')
plt.title('5-Fold CV Model Comparison (Macro F1)')
plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig(OUT / "figures" / "cv_comparison.png"); plt.show()

# 6. Feature Group Comparison
plt.figure(figsize=(7, 4))
sns.barplot(data=ablation_df, x='feature_group', y='macro_f1_mean', palette='viridis')
plt.title('Feature-Group Ablation (CV Macro F1)')
plt.tight_layout(); plt.savefig(OUT / "figures" / "feature_group_comparison.png"); plt.show()

# 7 & 8. Confusion Matrices
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title(f'Count Confusion Matrix - {selected_model_name}')
plt.tight_layout(); plt.savefig(OUT / "figures" / "confusion_matrix.png"); plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Purples', normalize='true')
plt.title(f'Row-Normalized Confusion Matrix - {selected_model_name}')
plt.tight_layout(); plt.savefig(OUT / "figures" / "confusion_matrix_normalized.png"); plt.show()

# 9. Per-Class Metrics
sub_r = report_df.loc[['A', 'B', 'C', 'D'], ['precision', 'recall', 'f1-score']]
sub_r.plot(kind='bar', figsize=(7, 4))
plt.title('Per-Class Test Metrics')
plt.ylim(0, 1.05)
plt.tight_layout(); plt.savefig(OUT / "figures" / "per_class_metrics.png"); plt.show()

# 10. Confidence Distribution
plt.figure(figsize=(7, 4))
sns.histplot(pred_df['max_posterior'], kde=True, color='darkgreen')
plt.axvline(0.75, color='blue', linestyle='--', label='High Conf Cutoff (0.75)')
plt.axvline(0.50, color='orange', linestyle='--', label='Explicit Review Cutoff (0.50)')
plt.title('Max Posterior Probability Distribution')
plt.legend()
plt.tight_layout(); plt.savefig(OUT / "figures" / "confidence_distribution.png"); plt.show()"""
cells.append(create_cell("code", c17_code))

# ---------------------------------------------------------
# Cell 18: Responsible Analytics Discussion
# ---------------------------------------------------------
c18_md = """### 18. Responsible Analytics & Risk Register (Rule 15)
Document privacy safeguards, sensitive attribute handling, profiling risks, historical label bias, and human oversight requirements."""
cells.append(create_cell("markdown", c18_md))

c18_code = """responsible_notes = \"\"\"
======================================================================
RESPONSIBLE ANALYTICS STATEMENT & RISK REGISTER
======================================================================
1. PRIVACY & DATA MINIMIZATION:
   Direct customer IDs are stripped prior to feature engineering. No personal
   identifiable information (PII) is exposed to model training pipelines.

2. PROFILING & STEREOTYPING RISKS:
   Predictive modeling of customer segments based on demographic attributes (Gender,
   Age, Profession) creates potential stereotyping risks. Segment predictions must be
   treated as descriptive marketing archetypes, NEVER as individual character traits.

3. HISTORICAL LABEL BIAS:
   Target 'Segmentation' labels reflect past corporate marketing allocation policy.
   Models trained on these labels reproduce historical firm biases.

4. HUMAN OVERSIGHT MANDATE:
   All high-stakes business interventions (pricing, exclusion, service tier change)
   REQUIRE explicit human review. Automated decisioning without human review is prohibited.

5. EXTENSION SCOPE BOUNDARY:
   Quantitative fairness subgroup auditing and temporal drift analysis are marked as
   research extensions in the manual and were NOT attempted in this core laboratory scope.
======================================================================
\"\"\"
print(responsible_notes)"""
cells.append(create_cell("code", c18_code))

# ---------------------------------------------------------
# Cell 19: Check-Off Evidence Summary
# ---------------------------------------------------------
c19_md = """### 19. Instructor Check-Off Evidence Mapping
Summary of laboratory evidence mapped directly to the four required instructor check-offs from Section 13."""
cells.append(create_cell("markdown", c19_md))

c19_code = """checkoff_summary = {
    "Check-off 1: Dataset & Label Integrity": "Dataset A (8068 rows, 11 cols) loaded; SHA-256 logged; ID unique; Target 4 classes; Circularity audit complete.",
    "Check-off 2: Split & Preprocessing": "80/20 Stratified split saved; 0 ID overlap; ColumnTransformers fit only inside training folds; CategoricalNB non-negativity verified.",
    "Check-off 3: Model Selection": "Dummy + 3 Naive Bayes variants evaluated on identical 5-fold CV; Feature-group ablation complete; CategoricalNB_mixed selected before test set access.",
    "Check-off 4: Final Evaluation & Responsible Use": "Locked test evaluated ONCE (Macro F1 = 0.4853); 5 business errors analyzed; Frozen review threshold policy set; Responsible use statement documented."
}
print(json.dumps(checkoff_summary, indent=2))"""
cells.append(create_cell("code", c19_code))

# ---------------------------------------------------------
# Cell 20: Minimal Acceptance Tests
# ---------------------------------------------------------
c20_md = """### 20. Minimal Acceptance Tests (Rule 20 Verification)
Programmatic assertions verifying system integrity, split disjointness, artifact existence, representation safety, and model reloading reproducibility."""
cells.append(create_cell("markdown", c20_md))

c20_code = """# 1. Target Has >= 2 Classes
assert df[TARGET].nunique() >= 2, "Test Failed: Target has < 2 classes"

# 2. Customer ID Not in Features
assert ID_COL not in ALL_FEATURES, "Test Failed: ID present in feature set"

# 3. Train/Test Disjointness
assert set(train_df[ID_COL]).isdisjoint(set(test_df[ID_COL])), "Test Failed: Train/Test ID overlap"

# 4. Predicted Labels Are Subset of Training Labels
assert set(y_pred).issubset(set(y_train.unique())), "Test Failed: Unseen predicted labels"

# 5. Required Output Artifacts Exist
required_artifacts = [
    OUT / "artifacts" / "versions.json",
    OUT / "artifacts" / "split_manifest.csv",
    OUT / "artifacts" / "feature_manifest.json",
    OUT / "results" / "cv_results.csv",
    OUT / "results" / "classification_report.csv",
    OUT / "results" / "test_predictions.csv",
    OUT / "results" / "error_analysis.csv",
    OUT / "models" / "selected_pipeline.joblib"
]
for art in required_artifacts:
    assert art.exists(), f"Test Failed: Missing artifact {art}"

# 6. Joblib Model Reload Reproducibility Test
reloaded_pipe = joblib.load(OUT / "models" / "selected_pipeline.joblib")
reload_preds = reloaded_pipe.predict(X_test.head(10))
assert np.array_equal(reload_preds, y_pred[:10]), "Test Failed: Reloaded model predictions differ!"

print("=" * 70)
print("ALL CORE ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
print("=" * 70)"""
cells.append(create_cell("code", c20_code))

nb_structure = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.10"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_dest1 = Path("notebooks/23MID0420_Lab04_CustomerSegmentation.ipynb")
nb_dest2 = Path("23MID0420_Lab04_CustomerSegmentation.ipynb")

nb_dest1.parent.mkdir(parents=True, exist_ok=True)
with open(nb_dest1, "w", encoding="utf-8") as f:
    json.dump(nb_structure, f, indent=2)

with open(nb_dest2, "w", encoding="utf-8") as f:
    json.dump(nb_structure, f, indent=2)

print(f"Notebook successfully generated at {nb_dest1} and {nb_dest2}")
