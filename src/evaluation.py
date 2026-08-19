"""
Cross-validation comparison, feature-group ablation, locked test set evaluation, and visualization suite.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


def evaluate_cv_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    seed: int = 42,
    output_csv: Path = None
) -> pd.DataFrame:
    """
    Perform training-only 5-fold Stratified K-Fold cross-validation across core models
    using identical folds.
    Returns sorted CV summary dataframe and saves to CSV.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    scoring = {
        'accuracy': 'accuracy',
        'macro_f1': 'f1_macro',
        'weighted_f1': 'f1_weighted'
    }
    
    results = []
    for name, pipe in pipelines.items():
        start_time = time.perf_counter()
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1
        )
        elapsed = time.perf_counter() - start_time
        
        results.append({
            'model': name,
            'accuracy_mean': float(np.mean(scores['test_accuracy'])),
            'macro_f1_mean': float(np.mean(scores['test_macro_f1'])),
            'macro_f1_sd': float(np.std(scores['test_macro_f1'], ddof=1)),
            'weighted_f1_mean': float(np.mean(scores['test_weighted_f1'])),
            'cv_time_seconds': float(elapsed)
        })
        
    cv_df = pd.DataFrame(results).sort_values('macro_f1_mean', ascending=False).reset_index(drop=True)
    
    if output_csv is not None:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv_df.to_csv(out, index=False)
        
    return cv_df


def evaluate_feature_group_ablation(
    model_factory,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_manifest: Dict[str, List[str]],
    numeric_cols: List[str],
    categorical_cols: List[str],
    cv_splits: int = 5,
    seed: int = 42,
    output_csv: Path = None
) -> pd.DataFrame:
    """
    Feature-Group Ablation (Rule 11):
    Compare Demographic-only, Psychographic-only, Behavioral-only, and Combined feature sets
    using ONE predeclared core model and identical CV folds.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    scoring = {'macro_f1': 'f1_macro', 'weighted_f1': 'f1_weighted', 'accuracy': 'accuracy'}
    
    groups = {
        'demographic_only': feature_manifest['demographic'],
        'psychographic_only': feature_manifest['psychographic'],
        'behavioral_only': feature_manifest['behavioral'],
        'combined_all': feature_manifest['all_features']
    }
    
    ablation_results = []
    for group_name, group_cols in groups.items():
        if not group_cols:
            continue
            
        sub_num = [c for c in numeric_cols if c in group_cols]
        sub_cat = [c for c in categorical_cols if c in group_cols]
        
        # Build specific pipeline for this subset of columns
        pipe = model_factory(sub_num, sub_cat)
        
        scores = cross_validate(
            pipe, X_train[group_cols], y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        
        ablation_results.append({
            'feature_group': group_name,
            'num_features': len(group_cols),
            'accuracy_mean': float(np.mean(scores['test_accuracy'])),
            'macro_f1_mean': float(np.mean(scores['test_macro_f1'])),
            'macro_f1_sd': float(np.std(scores['test_macro_f1'], ddof=1)),
            'weighted_f1_mean': float(np.mean(scores['test_weighted_f1']))
        })
        
    ablation_df = pd.DataFrame(ablation_results).sort_values('macro_f1_mean', ascending=False).reset_index(drop=True)
    
    if output_csv is not None:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        ablation_df.to_csv(out, index=False)
        
    return ablation_df


def evaluate_locked_test(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    id_test: pd.Series,
    output_dir: Path
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Rule 10: Fit the selected pipeline on full training split and evaluate ONCE on locked test set.
    Saves classification report, test predictions, and returns metrics dict.
    """
    out_dir = Path(output_dir)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fit on full training split
    t0_train = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0_train
    
    # 2. Predict on locked test split
    t0_inf = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    inference_seconds = time.perf_counter() - t0_inf
    
    acc = float(accuracy_score(y_test, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': acc,
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_p),
        'weighted_recall': float(weighted_r),
        'weighted_f1': float(weighted_f1),
        'training_seconds': float(train_seconds),
        'inference_seconds': float(inference_seconds)
    }
    
    # Per-class classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(results_dir / "classification_report.csv")
    
    # Predictions DataFrame
    pred_df = pd.DataFrame({
        'ID': id_test,
        'actual_segment': y_test,
        'predicted_segment': y_pred
    })
    
    if hasattr(pipeline, 'predict_proba'):
        probs = pipeline.predict_proba(X_test)
        classes = pipeline.classes_
        pred_df['max_posterior'] = probs.max(axis=1)
        for i, cls in enumerate(classes):
            pred_df[f'prob_{cls}'] = probs[:, i]
            
    pred_df.to_csv(results_dir / "test_predictions.csv", index=False)
    
    return metrics, report_df, pred_df


# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def plot_class_distribution(df: pd.DataFrame, target_col: str, output_path: Path) -> None:
    """Plot target segment class distribution bar chart."""
    plt.figure(figsize=(7, 4.5))
    counts = df[target_col].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, palette='Blues_d')
    plt.title('Customer Segment Distribution (Target: Segmentation)', fontsize=12, fontweight='bold')
    plt.xlabel('Segment Class')
    plt.ylabel('Number of Customers')
    for i, v in enumerate(counts.values):
        plt.text(i, v + 20, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_missing_values(df: pd.DataFrame, features: List[str], output_path: Path) -> None:
    """Plot percentage of missing values per feature."""
    plt.figure(figsize=(8, 4.5))
    missing_pct = (df[features].isna().mean() * 100).sort_values(ascending=False)
    sns.barplot(x=missing_pct.values, y=missing_pct.index, palette='Reds_r')
    plt.title('Missing Value Percentage by Feature', fontsize=12, fontweight='bold')
    plt.xlabel('Missing Percentage (%)')
    plt.ylabel('Feature')
    for i, v in enumerate(missing_pct.values):
        plt.text(v + 0.1, i, f"{v:.1f}%", va='center')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: List[str], output_path: Path) -> None:
    """Plot histograms/KDEs for continuous numeric features."""
    if not numeric_cols:
        return
    n_cols = len(numeric_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color='teal')
        ax.set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_categorical_frequencies(df: pd.DataFrame, categorical_cols: List[str], output_path: Path) -> None:
    """Plot count plots for categorical variables."""
    if not categorical_cols:
        return
    n_cols = len(categorical_cols)
    nrows = (n_cols + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 3.8 * nrows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    for i, col in enumerate(categorical_cols):
        sns.countplot(data=df, x=col, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Frequency of {col}')
        axes[i].tick_params(axis='x', rotation=30)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cv_comparison(cv_df: pd.DataFrame, output_path: Path) -> None:
    """Plot cross-validation model comparison with error bars."""
    plt.figure(figsize=(8, 4.5))
    models = cv_df['model']
    means = cv_df['macro_f1_mean']
    sds = cv_df['macro_f1_sd']
    
    plt.bar(models, means, yerr=sds, capsize=5, color='skyblue', edgecolor='navy')
    plt.title('5-Fold CV Model Comparison (Macro F1 Mean ± SD)', fontsize=12, fontweight='bold')
    plt.xlabel('Candidate Model')
    plt.ylabel('Macro F1 Score')
    plt.xticks(rotation=15)
    for i, (m, s) in enumerate(zip(means, sds)):
        plt.text(i, m / 2, f"{m:.4f}\n(±{s:.4f})", ha='center', color='black', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_group_comparison(ablation_df: pd.DataFrame, output_path: Path) -> None:
    """Plot feature group ablation Macro F1 scores."""
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=ablation_df, x='feature_group', y='macro_f1_mean', palette='viridis')
    plt.title('Feature-Group Ablation Study (Macro F1 Score)', fontsize=12, fontweight='bold')
    plt.xlabel('Feature Group')
    plt.ylabel('CV Mean Macro F1')
    for i, row in ablation_df.iterrows():
        plt.text(i, row['macro_f1_mean'] + 0.01, f"{row['macro_f1_mean']:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrices(
    y_test: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    output_count_path: Path,
    output_norm_path: Path
) -> None:
    """Plot count and row-normalized confusion matrix heatmaps."""
    classes = np.unique(y_test)
    
    # 1. Count Confusion Matrix
    cm_count = confusion_matrix(y_test, y_pred, labels=classes)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_count, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp1.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(f'Count Confusion Matrix - {model_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_count_path, dpi=200)
    plt.close()
    
    # 2. Row-Normalized Confusion Matrix
    cm_norm = confusion_matrix(y_test, y_pred, labels=classes, normalize='true')
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp2.plot(cmap='Purples', ax=ax, values_format='.2f')
    plt.title(f'Row-Normalized Confusion Matrix - {model_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_norm_path, dpi=200)
    plt.close()


def plot_per_class_metrics(report_df: pd.DataFrame, output_path: Path) -> None:
    """Plot per-class Precision, Recall, and F1 bar plot."""
    # Filter out macro avg, weighted avg, accuracy
    class_rows = [r for r in report_df.index if r not in ['accuracy', 'macro avg', 'weighted avg']]
    sub_df = report_df.loc[class_rows, ['precision', 'recall', 'f1-score']]
    
    plt.figure(figsize=(8, 4.5))
    sub_df.plot(kind='bar', figsize=(8, 4.5))
    plt.title('Per-Class Performance Metrics on Locked Test Set', fontsize=12, fontweight='bold')
    plt.xlabel('Customer Segment Class')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.legend(title='Metric', loc='lower right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confidence_distribution(pred_df: pd.DataFrame, output_path: Path) -> None:
    """Plot max posterior confidence score distribution."""
    if 'max_posterior' not in pred_df.columns:
        return
    plt.figure(figsize=(7, 4.5))
    sns.histplot(pred_df['max_posterior'], kde=True, bins=20, color='darkgreen')
    plt.axvline(0.75, color='blue', linestyle='--', label='High Confidence Cutoff (0.75)')
    plt.axvline(0.50, color='orange', linestyle='--', label='Explicit Review Cutoff (0.50)')
    plt.title('Max Posterior Probability Distribution on Test Set', fontsize=12, fontweight='bold')
    plt.xlabel('Max Posterior Probability P(C_k | x)')
    plt.ylabel('Customer Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
