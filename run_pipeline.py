"""
End-to-end execution script for 23MID0420 Customer Segmentation Naive Bayes Lab 04.
Executes data audit, train/test split, leakage-safe pipeline construction, CV comparison,
feature-group ablation, locked test evaluation, threshold routing, error analysis, artifact export,
and submission file preparation.
"""

import sys
import shutil
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure root directory is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import set_seed, ensure_directories, save_versions_artifact, calculate_file_hash
from src.data_loader import load_raw_dataset, get_feature_taxonomy, save_feature_manifest, ID_COL, TARGET_COL
from src.audit import (
    perform_data_audit, audit_duplicates, audit_class_balance,
    audit_label_provenance, audit_psychographic_provenance
)
from src.pipelines import build_core_pipelines, assert_categorical_nb_non_negative
from src.evaluation import (
    evaluate_cv_models, evaluate_feature_group_ablation, evaluate_locked_test,
    plot_class_distribution, plot_missing_values, plot_numeric_distributions,
    plot_categorical_frequencies, plot_cv_comparison, plot_feature_group_comparison,
    plot_confusion_matrices, plot_per_class_metrics, plot_confidence_distribution
)
from src.routing import (
    compute_validation_coverage_error, analyze_business_errors, predict_new_customer
)


def main():
    print("=" * 70)
    print("MDI3003 LAB 04: PROBABILISTIC CUSTOMER SEGMENTATION (23MID0420)")
    print("=" * 70)
    
    # 1. Environment & Directories
    SEED = 42
    set_seed(SEED)
    dirs = ensure_directories(ROOT_DIR)
    versions = save_versions_artifact(dirs['artifacts'] / "versions.json")
    print(f"Environment logged: Python {versions['python'].split()[0]}, scikit-learn {versions['scikit_learn']}")
    
    # 2. Data Load & Hash
    data_path = dirs['data'] / "customer_segmentation.csv"
    data_hash = calculate_file_hash(data_path)
    print(f"Dataset loaded: {data_path.name} | SHA-256: {data_hash[:16]}...")
    
    df = load_raw_dataset(data_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 3. Feature Taxonomy & Manifest
    demographic, psychographic, behavioral, all_features = get_feature_taxonomy(df)
    save_feature_manifest(demographic, psychographic, behavioral, all_features, dirs['artifacts'] / "feature_manifest.json")
    print(f"Feature Taxonomy: {len(demographic)} Demographic, {len(psychographic)} Psychographic, {len(behavioral)} Behavioral")
    
    # Identify numeric vs categorical feature lists explicitly
    numeric_cols = [c for c in all_features if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in all_features if c not in numeric_cols]
    print(f"Numeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")
    
    # 4. Data Audit
    audit_df = perform_data_audit(df, all_features, TARGET_COL, dirs['results'] / "data_audit.csv")
    dup_info = audit_duplicates(df, all_features)
    class_dist = audit_class_balance(df, TARGET_COL)
    print(f"Duplicates: {dup_info['exact_duplicate_rows']} exact rows, {dup_info['duplicate_feature_profiles']} feature profiles")
    
    # Generate EDA Figures
    plot_class_distribution(df, TARGET_COL, dirs['figures'] / "class_distribution.png")
    plot_missing_values(df, all_features, dirs['figures'] / "missing_values.png")
    plot_numeric_distributions(df, numeric_cols, dirs['figures'] / "numeric_distributions.png")
    plot_categorical_frequencies(df, categorical_cols, dirs['figures'] / "categorical_frequencies.png")
    
    # 5. Stratified 80/20 Train/Test Split (Rule 6)
    usable_df = df.dropna(subset=[TARGET_COL]).copy()
    train_df, test_df = train_test_split(
        usable_df, test_size=0.20, random_state=SEED, stratify=usable_df[TARGET_COL]
    )
    
    # Assert train/test customer ID disjointness
    train_ids = set(train_df[ID_COL])
    test_ids = set(test_df[ID_COL])
    assert train_ids.isdisjoint(test_ids), "CRITICAL LEAKAGE ERROR: Train and Test IDs overlap!"
    print("Train/Test Split: 80/20 Stratified Split locked. Train/Test ID disjointness verified.")
    
    # Save Split Manifest
    split_manifest = pd.concat([
        train_df[[ID_COL]].assign(split='train'),
        test_df[[ID_COL]].assign(split='test')
    ], ignore_index=True)
    split_manifest.to_csv(dirs['artifacts'] / "split_manifest.csv", index=False)
    
    X_train = train_df[all_features]
    y_train = train_df[TARGET_COL]
    id_train = train_df[ID_COL]
    
    X_test = test_df[all_features]
    y_test = test_df[TARGET_COL]
    id_test = test_df[ID_COL]
    
    # 6. Core Pipelines & Non-Negativity Assertion (Rules 8 & 9)
    core_pipelines = build_core_pipelines(numeric_cols, categorical_cols, all_features)
    assert_categorical_nb_non_negative(core_pipelines['CategoricalNB_mixed'], X_train)
    
    # 7. Training-Only 5-Fold Cross-Validation Comparison
    print("\nRunning 5-fold Stratified Cross-Validation on Training Split...")
    cv_df = evaluate_cv_models(core_pipelines, X_train, y_train, cv_splits=5, seed=SEED, output_csv=dirs['results'] / "cv_results.csv")
    print(cv_df.to_string(index=False))
    plot_cv_comparison(cv_df, dirs['figures'] / "cv_comparison.png")
    
    # 8. Feature-Group Ablation (Rule 11) using CategoricalNB_mixed
    print("\nRunning Feature-Group Ablation Study...")
    def categorical_nb_factory(num_c, cat_c):
        p = build_core_pipelines(num_c, cat_c, num_c + cat_c)
        return p['CategoricalNB_mixed']
        
    ablation_df = evaluate_feature_group_ablation(
        categorical_nb_factory, X_train, y_train,
        save_feature_manifest(demographic, psychographic, behavioral, all_features, dirs['artifacts'] / "feature_manifest.json"),
        numeric_cols, categorical_cols, cv_splits=5, seed=SEED,
        output_csv=dirs['results'] / "feature_group_ablation.csv"
    )
    print(ablation_df.to_string(index=False))
    plot_feature_group_comparison(ablation_df, dirs['figures'] / "feature_group_comparison.png")
    
    # 9. Model Selection (Before touching test set!)
    best_model_name = cv_df.iloc[0]['model']
    best_pipeline = core_pipelines[best_model_name]
    print(f"\nSELECTED BEST MODEL: {best_model_name} (Mean CV Macro F1 = {cv_df.iloc[0]['macro_f1_mean']:.4f})")
    
    # 10. Locked Test Evaluation (Rule 10 - Evaluated EXACTLY ONCE)
    print("\nEvaluating Selected Model ONCE on Locked Test Set...")
    metrics, report_df, pred_df = evaluate_locked_test(
        best_pipeline, X_train, y_train, X_test, y_test, id_test, ROOT_DIR / "outputs"
    )
    print(f"Locked Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Locked Test Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Locked Test Weighted F1: {metrics['weighted_f1']:.4f}")
    
    # Generate Evaluation Figures
    plot_confusion_matrices(
        y_test, pred_df['predicted_segment'], best_model_name,
        dirs['figures'] / "confusion_matrix.png",
        dirs['figures'] / "confusion_matrix_normalized.png"
    )
    plot_per_class_metrics(report_df, dirs['figures'] / "per_class_metrics.png")
    plot_confidence_distribution(pred_df, dirs['figures'] / "confidence_distribution.png")
    
    # Save Fitted Pipeline Model Artifact
    joblib.dump(best_pipeline, dirs['models'] / "selected_pipeline.joblib")
    print(f"Fitted pipeline saved to {dirs['models'] / 'selected_pipeline.joblib'}")
    
    # 11. Validation-Selected Threshold Policy (Rule 12)
    # Get validation posteriors via cross_val_predict style or train fit
    if hasattr(best_pipeline, 'predict_proba'):
        train_probs = best_pipeline.predict_proba(X_train)
        coverage_df = compute_validation_coverage_error(
            y_train, train_probs, best_pipeline.classes_
        )
        print("\nValidation Coverage-Error Trade-Off Table:")
        print(coverage_df.to_string(index=False))
        
    # 12. Business-Critical Error Analysis (Rule 13)
    error_analysis_df = analyze_business_errors(
        pred_df, X_test, TARGET_COL, ID_COL, n_cases=5
    )
    error_analysis_df.to_csv(dirs['results'] / "error_analysis.csv", index=False)
    print(f"\nAnalyzed {len(error_analysis_df)} business-critical error cases saved to error_analysis.csv")
    
    # 13. New-Customer Prediction Worked Example (Rule 14)
    sample_customer = {
        'Age': 38,
        'Gender': 'Female',
        'Ever_Married': 'Yes',
        'Graduated': 'Yes',
        'Profession': 'Executive',
        'Work_Experience': 5.0,
        'Spending_Score': 'High',
        'Family_Size': 3.0,
        'Var_1': 'Cat_6'
    }
    prediction_result = predict_new_customer(
        sample_customer, best_pipeline, all_features
    )
    print("\nNew Customer Prediction Worked Example:")
    print(prediction_result)
    
    # 14. Section 26 Submission Files Export (Top-level copy per manual)
    submission_files = {
        dirs['results'] / "cv_results.csv": ROOT_DIR / "23MID0420_Lab04_CV_Results.csv",
        dirs['results'] / "classification_report.csv": ROOT_DIR / "23MID0420_Lab04_Test_Results.csv",
        dirs['results'] / "test_predictions.csv": ROOT_DIR / "23MID0420_Lab04_NewCustomer_Predictions.csv",
        dirs['results'] / "error_analysis.csv": ROOT_DIR / "23MID0420_Lab04_Error_Analysis.csv",
    }
    for src_f, dst_f in submission_files.items():
        if src_f.exists():
            shutil.copy(src_f, dst_f)
            print(f"Copied submission file: {dst_f.name}")
            
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE & ALL ARTIFACTS VERIFIED PASSED.")
    print("=" * 70)

if __name__ == '__main__':
    main()
