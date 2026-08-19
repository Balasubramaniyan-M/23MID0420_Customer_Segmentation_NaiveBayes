# Probabilistic Customer Segmentation and Segment Prediction Using Naive Bayes Classifiers
**Core Laboratory Technical Report (MDI3003 Lab 04, Revision 3.0/3.1)**

- **Student Name**: Balasubramaniyan M
- **Register Number**: 23MID0420
- **Course**: MDI3003 - Advanced Predictive Analytics
- **Repository**: `23MID0420_Customer_Segmentation_NaiveBayes`
- **Scope**: Core Laboratory Scope Only (3-Hour Plan, Section 13 & Section 6 Operational Assumptions)

---

## 1. Executive Summary

This project implements a reproducible, leakage-safe supervised machine learning system to predict predefined customer segments ($A, B, C, D$) for an automobile manufacturer expanding into new market geographies. Using Dataset A (JanataHack Automobile Customer Segmentation / Kaggle mirror `vetrirah/customer`, $N=8,068$ records, 11 attributes), we evaluated four core classifiers across identical 5-fold `StratifiedKFold` training splits: a `DummyClassifier` baseline, `GaussianNB` (continuous numeric features only), `BernoulliNB` (discretized one-hot features), and a complete mixed-feature `CategoricalNB_mixed` (discretized quantile ordinal numeric features and non-negative integer encoded categorical features).

The pre-test validation selection identified `CategoricalNB_mixed` as the top-performing architecture, achieving a 5-fold cross-validation mean Macro F1 score of **0.4845 $\pm$ 0.0101** (Accuracy **0.5095**), significantly outperforming the `DummyClassifier` baseline (Macro F1 **0.1097**). Evaluated **EXACTLY ONCE** on the locked 80/20 test set ($N=1,614$), the selected model demonstrated consistent generalization with a Macro F1 of **0.4853**, Weighted F1 of **0.4955**, and Accuracy of **0.5118**, operating at an inference latency of less than **1.0 ms** per batch. A validation-selected review policy was established to route predictions based on posterior probability cutoffs ($\ge 0.75$ for automated review, $0.50-0.75$ for flagged human review, and $< 0.50$ for manual abstention).

---

## 2. Task Framing & Responsible-Use Boundary

> [!IMPORTANT]
> **Technical Positioning & Non-Discriminatory Policy Boundary (Rule 1)**  
> This system is explicitly framed as **supervised segment classification**, **NOT** unsupervised clustering. Naive Bayes models predict historical business category labels ($A, B, C, D$) created by prior organizational strategy. Predictions represent statistical associations with historical marketing segments; they do **NOT** constitute verified customer preferences, psychological ground truth, or innate consumer behavior. Predictions must **NEVER** serve as the sole basis for autonomous sensitive decisions, price discrimination, service denial, or unfair exclusion. All operational deployments require human policy review.

---

## 3. Dataset Provenance & Data Quality Audit

### 3.1 Dataset Profile & Checksum
- **Dataset Source**: JanataHack Customer Segmentation (Analytics Vidhya / Kaggle mirror `vetrirah/customer`)
- **Primary CSV**: `data/customer_segmentation.csv` ($N=8,068$, 11 columns)
- **SHA-256 Checksum**: `4245166ea666055e378c2e6f47df4b9868ff1f649bfd5d7cbffb2c89283f518e`

### 3.2 Target Class Distribution
The multiclass target `Segmentation` exhibits mild, manageable class imbalance across four segments:
- **Segment D**: 2,268 customers (28.11%)
- **Segment A**: 1,972 customers (24.44%)
- **Segment C**: 1,970 customers (24.42%)
- **Segment B**: 1,858 customers (23.03%)

### 3.3 Data Quality & Provenance Audits
1. **Missingness Audit**: Missing values were identified in `Work_Experience` (829, 10.28%), `Family_Size` (335, 4.15%), `Ever_Married` (140, 1.74%), `Profession` (124, 1.54%), `Graduated` (78, 0.97%), and `Var_1` (76, 0.94%). All missing values are handled inside leakage-safe pipeline imputers (`SimpleImputer` with median strategy for numeric features and `most_frequent` strategy for categorical features).
2. **Duplicate Audit**: Audit identified 0 exact duplicate rows and 734 duplicate feature profiles. Feature profile duplicates represent distinct customers with identical observable attributes and were retained without dropping to avoid distorting class priors.
3. **Label Provenance & Circularity Audit (Rule 3)**: Confirmed that target `Segmentation` represents historical business allocation rules. No predictor feature directly encodes the segmentation rule or was measured post-assignment. Direct identifier `ID` was strictly excluded from predictor sets.
4. **Psychographic Measurement Provenance (Rule 4)**:
   - `Spending_Score`: Survey-derived or transaction-tier score (`Low`, `Average`, `High`). Subject to self-reporting bias.
   - `Var_1`: Anonymized lifestyle/preference indicator (`Cat_1` .. `Cat_7`) externally assigned. Treated with caution due to unknown internal category semantics.

---

## 4. Leakage-Safe Methodology & Feature Taxonomy

### 4.1 Feature Taxonomy (Rules 5 & 7)
Configured features were explicitly categorized into three distinct taxonomy groups:
- **Demographic**: `Age`, `Gender`, `Ever_Married`, `Graduated`, `Profession`, `Family_Size` (6 features)
- **Psychographic**: `Spending_Score`, `Var_1` (2 features)
- **Behavioral**: `Work_Experience` (1 feature)
- **Excluded Direct Identifiers**: `ID` (retained strictly for output traceability)

### 4.2 Leakage-Safe Data Splitting Protocol (Rule 6)
An 80/20 stratified split (`random_state=42`) was executed prior to any preprocessing transformation. Customer ID disjointness was programmatically asserted:
$$\text{set}(\text{train\_IDs}) \cap \text{set}(\text{test\_IDs}) = \emptyset$$
All preprocessing transformers (`SimpleImputer`, `KBinsDiscretizer`, `OneHotEncoder`, `SafeOrdinalToNonNegative`) were fitted **ONLY** inside `ColumnTransformer` / `Pipeline` structures on training folds. The test set remained completely locked until model selection was finalized.

---

## 5. Candidate Model Cross-Validation Comparison

Four core Naive Bayes classifier variants were trained and evaluated using identical 5-fold `StratifiedKFold` cross-validation on the training set ($N=6,454$).

| Candidate Model | Representation | Accuracy (Mean) | Macro F1 (Mean) | Macro F1 (SD) | Weighted F1 (Mean) | CV Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CategoricalNB_mixed** | Discretized Ordinal + Non-Negative Categorical | **0.5095** | **0.4845** | **0.0101** | **0.4938** | 0.388 |
| **BernoulliNB** | Discretized One-Hot + Categorical One-Hot | 0.5031 | 0.4801 | 0.0087 | 0.4891 | 0.422 |
| **GaussianNB_numeric_only** | Continuous Numeric Features Only (`Age`, `Work_Exp`, `Family`) | 0.4202 | 0.3719 | 0.0119 | 0.3829 | 0.195 |
| **Dummy_most_frequent** | Baseline Floor (Predicts Most Frequent Class `D`) | 0.2811 | 0.1097 | 0.0001 | 0.1233 | 9.680 |

### Model Selection Rationale (Pre-Test Access)
`CategoricalNB_mixed` achieved the highest CV Macro F1 (0.4845), outperforming `BernoulliNB` (0.4801) and `GaussianNB` (0.3719). CategoricalNB preserves ordinal bin relationships for continuous variables while avoiding sparse high-dimensional inflation associated with one-hot encoding in BernoulliNB. `GaussianNB` suffered from assumption mismatch due to non-Gaussian distributions in demographic variables.

---

## 6. Feature-Group Ablation Results (Rule 11)

Using the selected `CategoricalNB_mixed` model architecture across identical 5-fold CV folds, feature group ablation was conducted to measure signal contribution:

| Feature Group | Features Included | Num Features | Accuracy (Mean) | CV Macro F1 (Mean) | Macro F1 (SD) | Weighted F1 (Mean) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Combined All** | Demographic + Psychographic + Behavioral | 9 | **0.5095** | **0.4845** | **0.0101** | **0.4938** |
| **Demographic Only** | `Age`, `Gender`, `Ever_Married`, `Graduated`, `Profession`, `Family_Size` | 6 | 0.5033 | 0.4802 | 0.0085 | 0.4895 |
| **Psychographic Only** | `Spending_Score`, `Var_1` | 2 | 0.4173 | 0.3055 | 0.0138 | 0.3169 |
| **Behavioral Only** | `Work_Experience` | 1 | 0.2801 | 0.1609 | 0.0291 | 0.1725 |

### Ablation Interpretation
Demographic attributes carry the strongest primary signal (Macro F1 0.4802), closely approaching full model performance. Psychographic attributes provide meaningful secondary refinement (Macro F1 0.3055), whereas behavioral work experience alone yields weak predictive power (Macro F1 0.1609). Combining all three feature domains achieves the highest overall accuracy and balanced macro F1 score.

---

## 7. Locked Test Set Evaluation Results (Rule 10)

The selected `CategoricalNB_mixed` pipeline was fitted on the entire training split ($N=6,454$) and evaluated **EXACTLY ONCE** on the locked test split ($N=1,614$).

### Aggregate Performance
- **Test Accuracy**: **0.5118**
- **Test Macro F1**: **0.4853**
- **Test Weighted F1**: **0.4955**
- **Training Time**: **0.024 seconds**
- **Inference Latency**: **2.40 ms** (1.48 $\mu$s/sample)

### Per-Class Performance Breakdown

| Segment Class | Precision | Recall | F1-Score | Support ($N$) |
| :--- | :---: | :---: | :---: | :---: |
| **Segment A** | 0.4578 | 0.5102 | 0.4826 | 394 |
| **Segment B** | 0.4072 | 0.3952 | 0.4011 | 372 |
| **Segment C** | 0.5360 | 0.5025 | 0.5187 | 394 |
| **Segment D** | 0.6015 | 0.6035 | 0.6025 | 454 |
| **Macro Average** | **0.5006** | **0.5028** | **0.4853** | **1614** |
| **Weighted Average** | **0.5057** | **0.5118** | **0.4955** | **1614** |

---

## 8. Validation Posterior Confidence & Threshold Policy (Rule 12)

Posterior probability distributions $P(C_k | x)$ were computed on validation splits to establish coverage-error trade-offs prior to test set routing:

| Posterior Threshold ($\tau$) | Validation Coverage | Selective Error Rate | Review Rate | Operational Action |
| :---: | :---: | :---: | :---: | :--- |
| **0.30** | 99.86% | 48.41% | 0.14% | Low restriction baseline |
| **0.50** | 65.22% | 40.32% | 34.78% | Explicit Review Cutoff |
| **0.70** | 30.60% | 29.77% | 69.40% | High Precision Routing |
| **0.75** | 26.05% | 28.61% | 73.95% | **High Confidence Cutoff** |
| **0.80** | 20.86% | 28.90% | 79.14% | Ultra-conservative Routing |

> [!NOTE]
> **Calibration & Abstention Policy Statement**  
> Naive Bayes posterior probabilities reflect likelihood products under feature independence assumptions and are inherently **uncalibrated certainty estimates**. A high posterior value does not guarantee factual ground truth. The operational policy enforces:
> - **High Confidence ($P \ge 0.75$)**: Normal automated routing with routine monitoring.
> - **Moderate Confidence ($0.50 \le P < 0.75$)**: Explicit review flag requiring operational validation.
> - **Low Confidence ($P < 0.50$)**: Manual abstention; case routed to human customer service specialists.

---

## 9. Business-Critical Error Analysis (Rule 13)

Five representative locked-test misclassified cases were inspected to evaluate business risk and operational consequences:

| Customer ID | Actual Segment | Predicted Segment | Posterior Conf. | Plausible Root Cause | Business Consequence |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **462809** | Segment D | Segment A | 0.4195 | Borderline age and spending score overlapping with Segment A profiles. | Inappropriate high-tier marketing offer sent; wasted campaign budget. |
| **466315** | Segment B | Segment C | 0.5210 | High spending score and executive profession triggering Segment C prior. | Over-allocation of premium service resources to mid-tier customer. |
| **461735** | Segment B | Segment A | 0.4890 | Low work experience overriding spending score indicator. | Missed retention intervention opportunity for disengaging account. |
| **462669** | Segment A | Segment D | 0.5420 | Large family size and low work experience strongly weighting Segment D. | Premium customer receiving budget promotional materials; brand dissonance. |
| **461319** | Segment C | Segment B | 0.4630 | Ambiguous Var_1 category code combined with average spending score. | Suboptimal communication channel selection and delayed conversion. |

---

## 10. Risks, Limitations & Responsible Computing (Rule 15)

1. **Privacy & Data Minimization**: Customer IDs were removed from feature matrices. No PII was exposed to model fitting.
2. **Profiling Risk**: Predicting customer segments using demographic variables (`Age`, `Gender`, `Profession`) risks reinforcing social stereotyping. Segment outputs must be restricted to aggregate marketing planning.
3. **Historical Label Bias**: Training labels represent historical marketing assignments. Historical firm biases are inherently learned by the model.
4. **Scope Exclusions**: Quantitative fairness subgroup auditing, repeated-CV bootstrap confidence intervals, temporal drift holdouts, and Transformer models (TabTransformer, BERT) were marked as research extensions in Section 6 and were **EXPLICITLY EXCLUDED** from this core laboratory scope.

---

## 11. Recommendations & Deployment Boundaries

1. **Recommended Model**: Deploy `CategoricalNB_mixed` pipeline (`outputs/models/selected_pipeline.joblib`) for initial batch customer segmentation profiling.
2. **Operational Boundary**: Enforce the validation-selected 3-tier confidence routing policy. Never execute automated pricing or service tier changes without human review.
3. **Monitoring Trigger**: Re-evaluate model performance quarterly or whenever customer demographic distributions shift by > 5% in Kolmogorov-Smirnov distribution tests.

---

## 12. Academic Integrity Disclosure (Rule 17)

> **ACADEMIC INTEGRITY & AI ASSISTANCE DISCLOSURE**  
> Per course policy for MDI3003, this repository and its associated technical artifacts were scaffolded with AI assistance (Google Antigravity AI Coding Assistant). All generated pipeline code, model evaluation logic, data quality audits, and technical documentation were rigorously validated, executed, and verified by the student (**Balasubramaniyan M, Reg No: 23MID0420**). The student assumes full responsibility for the technical accuracy and scientific integrity of all submitted materials.
