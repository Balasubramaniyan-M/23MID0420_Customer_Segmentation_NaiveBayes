# Probabilistic Customer Segmentation and Segment Prediction Using Naive Bayes Classifiers

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn 1.3+](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Course**: MDI3003 - Advanced Predictive Analytics (Lab 04, Revision 3.0/3.1)  
**Student Name**: Balasubramaniyan M  
**Registration Number**: 23MID0420  
**Repository Name**: `23MID0420_Customer_Segmentation_NaiveBayes`  
**Scope**: Core Laboratory Scope Only (Section 13 3-Hour Core Plan & Section 6 Operational Scope)

---

## Technical Positioning & Responsible-Use Boundary

> [!IMPORTANT]
> **Supervised Classification Boundary (Rule 1)**  
> This system performs **supervised segment classification**, **NOT** unsupervised clustering. Naive Bayes classifiers predict predefined historical customer-segment labels ($A, B, C, D$) present in Dataset A (JanataHack Customer Segmentation). Model predictions represent statistical associations with historical marketing labels; they do **NOT** constitute verified customer preferences, psychological ground truth, or individual intent. Predictions must **NEVER** serve as the sole basis for autonomous sensitive decisions, price discrimination, credit/service denial, or unfair exclusion. Human policy review is mandatory prior to operational intervention.

---

## Quick Start & Reproducibility

### Environment Setup
```bash
# Clone repository
git clone https://github.com/BalasubramaniyanM/23MID0420_Customer_Segmentation_NaiveBayes.git
cd 23MID0420_Customer_Segmentation_NaiveBayes

# Install dependencies
pip install -r requirements.txt
```

### Execution Options
```bash
# Option 1: Execute end-to-end Python pipeline script
python run_pipeline.py

# Option 2: Run standalone executable Jupyter Notebook
jupyter notebook notebooks/23MID0420_Lab04_CustomerSegmentation.ipynb
```

---

## Repository Structure

```
23MID0420_Customer_Segmentation_NaiveBayes
│
├── README.md                                  # Repository overview & navigation
├── LICENSE                                    # MIT License
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Git ignore configuration
├── run_pipeline.py                            # End-to-end pipeline execution script
├── build_notebook.py                          # Executable notebook builder
│
├── 23MID0420_Lab04_CustomerSegmentation.ipynb # Root submission notebook
├── 23MID0420_Lab04_Report.md                  # Root technical report
├── 23MID0420_Lab04_README.md                  # Root lab submission guide
├── 23MID0420_Lab04_CV_Results.csv             # Root CV results submission file
├── 23MID0420_Lab04_Test_Results.csv           # Root test results submission file
├── 23MID0420_Lab04_NewCustomer_Predictions.csv# Root test predictions submission file
├── 23MID0420_Lab04_Error_Analysis.csv         # Root error analysis submission file
│
├── data/
│   ├── README.md                              # Dataset card (source, license, SHA-256, schema)
│   └── customer_segmentation.csv              # Primary Dataset A (JanataHack / Kaggle mirror)
│
├── notebooks/
│   └── 23MID0420_Lab04_CustomerSegmentation.ipynb # Core single executable lab notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                         # Data loading & taxonomy declaration
│   ├── audit.py                                # Quality, duplicate, & circularity audit
│   ├── pipelines.py                            # Leakage-safe pipelines & CategoricalNB encoder
│   ├── evaluation.py                           # CV comparison, ablation, & test evaluation
│   ├── routing.py                              # Threshold routing & business error analysis
│   └── utils.py                                # Version tracking & SHA-256 calculator
│
├── outputs/
│   ├── artifacts/                             # versions.json, split_manifest.csv, feature_manifest.json
│   ├── results/                               # data_audit.csv, cv_results.csv, classification_report.csv...
│   ├── figures/                               # 10 core PNG visualization charts
│   └── models/                                # Persisted fitted pipeline joblib model
│
└── reports/
    ├── 23MID0420_Lab04_Report.md               # Industry-style technical markdown report
    └── 23MID0420_Lab04_README.md               # Lab submission environment & setup guide
```

---

## Empirical Findings Summary

### 5-Fold Cross-Validation Comparison

| Candidate Model | Representation | CV Macro F1 (Mean $\pm$ SD) | CV Accuracy | CV Weighted F1 |
| :--- | :--- | :---: | :---: | :---: |
| **CategoricalNB_mixed** | Discretized Ordinal + Non-Negative Categorical | **0.4845 $\pm$ 0.0101** | **0.5095** | **0.4938** |
| **BernoulliNB** | Discretized One-Hot + Categorical One-Hot | 0.4801 $\pm$ 0.0087 | 0.5031 | 0.4891 |
| **GaussianNB_numeric_only** | Continuous Numeric (`Age`, `Work_Exp`, `Family`) | 0.3719 $\pm$ 0.0119 | 0.4202 | 0.3829 |
| **Dummy_most_frequent** | Floor Baseline (Most Frequent Class `D`) | 0.1097 $\pm$ 0.0001 | 0.2811 | 0.1233 |

### Locked Test Set Results (`CategoricalNB_mixed`)
- **Test Accuracy**: **0.5118**
- **Test Macro F1**: **0.4853**
- **Test Weighted F1**: **0.4955**
- **Inference Latency**: **2.40 ms** (batch) / **1.48 $\mu$s** (per sample)


---

## License
Distributed under the [MIT License](LICENSE).
