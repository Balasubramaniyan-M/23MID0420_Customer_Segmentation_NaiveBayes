# MDI3003 Lab 04 - Executable Core Laboratory Submission README

- **Student Name**: Balasubramaniyan M
- **Registration Number**: 23MID0420
- **Course Code**: MDI3003 - Advanced Predictive Analytics
- **Lab Title**: Probabilistic Customer Segmentation and Segment Prediction Using Demographic, Psychographic, and Behavioral Data with Naive Bayes Classifiers
- **Revision**: 3.0 / 3.1 (Core Laboratory Scope Only)

---

## 1. Project Overview & Scope Boundary
This repository contains the complete, reproducible core-laboratory implementation for MDI3003 Lab 04.
As defined in Section 13 (3-Hour Core Plan) and Section 6 (Operational Scope), this submission implements the **CORE LABORATORY ONLY** using Dataset A (JanataHack Customer Segmentation / Kaggle mirror `vetrirah/customer`).

### Explicit Scope Exclusions (Research Extensions Not Attempted)
- `ComplementNB` model as an assumed default
- `LogisticRegression` non-Naive-Bayes benchmark
- Repeated Stratified CV / Bootstrap 95% Confidence Intervals
- Quantitative Fairness / Subgroup Disparity Auditing
- Temporal Drift / Chronological Holdout Split Analysis
- Advanced Transformer Architectures (`TabTransformer`, `FT-Transformer`, `BERT`)

---

## 2. Environment Requirements & Installation

### Core Software Requirements
- **Python**: 3.10+
- **Core Libraries**: `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.3.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `joblib>=1.3.0`

### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/BalasubramaniyanM/23MID0420_Customer_Segmentation_NaiveBayes.git
cd 23MID0420_Customer_Segmentation_NaiveBayes

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 3. Reproduction & Execution Guide

### Option A: Run Full End-to-End Pipeline via CLI
Execute the single end-to-end Python pipeline script to run all data quality checks, generate split manifests, run 5-fold CV comparisons, conduct feature-group ablation, evaluate the locked test set, generate all 10 figures, and save all output artifacts:
```bash
python run_pipeline.py
```

### Option B: Execute Single Core-Lab Notebook
Open and execute the standalone, clean-runtime notebook from top to bottom:
```bash
jupyter notebook notebooks/23MID0420_Lab04_CustomerSegmentation.ipynb
```

---

## 4. Submission Files Inventory (Section 26 Compliance)

Per Section 26 of the course manual, the required submission artifacts are located in the repository root and output subdirectories:

1. `23MID0420_Lab04_CustomerSegmentation.ipynb`: Core executable notebook (also in `notebooks/`)
2. `23MID0420_Lab04_Report.md`: Industry-style technical report (also in `reports/`)
3. `23MID0420_Lab04_CV_Results.csv`: 5-fold CV comparison metrics across core models
4. `23MID0420_Lab04_Test_Results.csv`: Locked test set classification report
5. `23MID0420_Lab04_NewCustomer_Predictions.csv`: Full test set predictions with posteriors
6. `23MID0420_Lab04_Error_Analysis.csv`: Business-critical error case analysis
7. `23MID0420_Lab04_README.md`: Lab submission README
8. `outputs/models/`: Persisted joblib pipeline artifact (`selected_pipeline.joblib`)
9. `outputs/figures/`: All 10 required core visualization PNG figures
10. `outputs/artifacts/`: `versions.json`, `split_manifest.csv`, `feature_manifest.json`

---

## 5. Academic Integrity Disclosure (Rule 17)

> **ACADEMIC INTEGRITY & AI ASSISTANCE DISCLOSURE**  
> Per course policy for MDI3003, this repository and its associated technical artifacts were scaffolded with AI assistance (Google Antigravity AI Coding Assistant). All generated pipeline code, model evaluation logic, data quality audits, and technical documentation were rigorously validated, executed, and verified by the student (**Balasubramaniyan M, Reg No: 23MID0420**). The student assumes full responsibility for the technical accuracy and scientific integrity of all submitted materials.
