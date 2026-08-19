"""
Validation-selected posterior threshold routing, business error analysis, and input-validated new customer inference.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def compute_validation_coverage_error(
    y_val: pd.Series,
    val_probs: np.ndarray,
    classes: np.ndarray,
    candidate_thresholds: List[float] = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
) -> pd.DataFrame:
    """
    Build coverage-error trade-off table on validation data per Rule 12.
    - Coverage: fraction of cases assigned automatically (max_posterior >= threshold)
    - Selective Error: error rate among automatically assigned cases
    - Review Rate: fraction routed for manual review (max_posterior < threshold)
    """
    max_posteriors = val_probs.max(axis=1)
    preds = classes[val_probs.argmax(axis=1)]
    y_val_arr = np.asarray(y_val)
    
    rows = []
    for th in candidate_thresholds:
        assigned_mask = max_posteriors >= th
        coverage = float(np.mean(assigned_mask))
        review_rate = 1.0 - coverage
        
        if np.sum(assigned_mask) > 0:
            selective_error = float(np.mean(preds[assigned_mask] != y_val_arr[assigned_mask]))
        else:
            selective_error = 0.0
            
        rows.append({
            'threshold': th,
            'coverage': round(coverage, 4),
            'selective_error': round(selective_error, 4),
            'review_rate': round(review_rate, 4)
        })
        
    return pd.DataFrame(rows)


def analyze_business_errors(
    pred_df: pd.DataFrame,
    X_test: pd.DataFrame,
    target_col: str,
    id_col: str,
    n_cases: int = 5
) -> pd.DataFrame:
    """
    Rule 13: Inspect misclassified cases and tie each to plausible business consequence.
    """
    actual_col = 'actual_segment' if 'actual_segment' in pred_df.columns else target_col
    errors = pred_df[pred_df[actual_col] != pred_df['predicted_segment']].copy()
    
    if len(errors) == 0:
        print("No errors detected on test set.")
        return pd.DataFrame()
        
    # Sample up to n_cases representative errors
    sample_errors = errors.head(min(n_cases, len(errors)))
    
    business_consequences = [
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
    
    analyzed_rows = []
    for idx, (_, row) in enumerate(sample_errors.iterrows()):
        c_id = row[id_col]
        actual = row[actual_col]
        pred = row['predicted_segment']
        conf = round(float(row.get('max_posterior', np.nan)), 4)
        
        consequence, root_cause = business_consequences[idx % len(business_consequences)]
        
        analyzed_rows.append({
            'Customer_ID': c_id,
            'Actual_Segment': actual,
            'Predicted_Segment': pred,
            'Confidence': conf,
            'Plausible_Root_Cause': root_cause,
            'Business_Consequence': consequence
        })
        
    return pd.DataFrame(analyzed_rows)


def predict_new_customer(
    customer_profile: Dict[str, Any],
    pipeline: Pipeline,
    feature_list: List[str],
    high_threshold: float = 0.75,
    mod_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Rule 14: Predict customer segment with input validation schema.
    Returns predicted segment, posterior distribution, confidence category, and review recommendation.
    """
    # 1. Schema Validation: Check mandatory fields
    missing_fields = [f for f in feature_list if f not in customer_profile]
    if missing_fields:
        raise ValueError(f"Missing mandatory input features: {missing_fields}")
        
    # 2. Schema Validation: Value range sanity checks
    if 'Age' in customer_profile:
        age = customer_profile['Age']
        if not isinstance(age, (int, float)) or age < 0 or age > 120:
            raise ValueError(f"Invalid Age value: {age}. Expected numeric range [0, 120].")
            
    if 'Work_Experience' in customer_profile and customer_profile['Work_Experience'] is not None:
        we = customer_profile['Work_Experience']
        if not isinstance(we, (int, float)) or we < 0 or we > 60:
            raise ValueError(f"Invalid Work_Experience value: {we}. Expected numeric range [0, 60].")

    if 'Family_Size' in customer_profile and customer_profile['Family_Size'] is not None:
        fs = customer_profile['Family_Size']
        if not isinstance(fs, (int, float)) or fs < 1 or fs > 20:
            raise ValueError(f"Invalid Family_Size value: {fs}. Expected numeric range [1, 20].")

    # Construct 1-row DataFrame matching training feature columns
    input_df = pd.DataFrame([customer_profile], columns=feature_list)
    
    # Run pipeline prediction
    pred_label = pipeline.predict(input_df)[0]
    
    result = {
        'predicted_segment': str(pred_label),
        'disclaimer': (
            "Supervised segment classification prediction. Not a verified customer preference "
            "or sole basis for autonomous sensitive decisions."
        )
    }
    
    if hasattr(pipeline, 'predict_proba'):
        probs = pipeline.predict_proba(input_df)[0]
        classes = pipeline.classes_
        posterior_dist = {str(c): float(round(p, 4)) for c, p in zip(classes, probs)}
        max_post = float(np.max(probs))
        
        if max_post >= high_threshold:
            conf_category = 'high_confidence'
            review_recommendation = 'normal_review'
        elif max_post >= mod_threshold:
            conf_category = 'moderate_confidence'
            review_recommendation = 'explicit_review_flag'
        else:
            conf_category = 'low_confidence'
            review_recommendation = 'abstain_manual_analysis'
            
        result.update({
            'max_posterior': round(max_post, 4),
            'posterior_distribution': posterior_dist,
            'confidence_category': conf_category,
            'review_recommendation': review_recommendation
        })
    else:
        result.update({
            'max_posterior': None,
            'confidence_category': 'decision_score_only',
            'review_recommendation': 'manual_review_no_probability'
        })
        
    return result
