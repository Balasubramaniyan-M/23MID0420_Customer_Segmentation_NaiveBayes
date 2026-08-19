"""
Data loading, schema validation, feature taxonomy definition, and feature manifest generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


# Default taxonomy mapping for Dataset A (JanataHack Customer Segmentation)
DEMOGRAPHIC_COLS = ['Age', 'Gender', 'Ever_Married', 'Graduated', 'Profession', 'Family_Size']
PSYCHOGRAPHIC_COLS = ['Spending_Score', 'Var_1']
BEHAVIORAL_COLS = ['Work_Experience']
ID_COL = 'ID'
TARGET_COL = 'Segmentation'


def load_raw_dataset(data_path: Path) -> pd.DataFrame:
    """Load dataset CSV and perform initial schema checks."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
    
    df = pd.read_csv(path)
    
    # 1. Required column validation
    required_cols = {ID_COL, TARGET_COL}
    missing_req = required_cols - set(df.columns)
    if missing_req:
        raise ValueError(f"Missing mandatory columns: {sorted(missing_req)}")
    
    # 2. Target class validation
    n_classes = df[TARGET_COL].nunique(dropna=True)
    if n_classes < 2:
        raise ValueError(f"Target '{TARGET_COL}' must contain at least 2 distinct classes. Found: {n_classes}")
    
    # 3. Customer ID uniqueness check
    if df[ID_COL].duplicated().any():
        dup_count = df[ID_COL].duplicated().sum()
        print(f"WARNING: Detected {dup_count} duplicate Customer ID records. Investigate before modeling.")
        
    return df


def get_feature_taxonomy(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Extract existing features present in dataset and map them to explicit taxonomy groups:
    - Demographic
    - Psychographic
    - Behavioral
    Returns (demographic_features, psychographic_features, behavioral_features, all_features)
    """
    existing_cols = set(df.columns)
    
    demographic = [c for c in DEMOGRAPHIC_COLS if c in existing_cols]
    psychographic = [c for c in PSYCHOGRAPHIC_COLS if c in existing_cols]
    behavioral = [c for c in BEHAVIORAL_COLS if c in existing_cols]
    
    all_features = demographic + psychographic + behavioral
    
    if not all_features:
        raise ValueError("No valid predictor features found in the dataset.")
        
    # Rule 5: Assert direct identifier ID is not in model predictors
    if ID_COL in all_features:
        raise AssertionError("Direct identifier 'ID' must NEVER be included as a predictor feature.")
        
    return demographic, psychographic, behavioral, all_features


def save_feature_manifest(
    demographic: List[str],
    psychographic: List[str],
    behavioral: List[str],
    all_features: List[str],
    output_path: Path
) -> Dict[str, List[str]]:
    """Save feature taxonomy breakdown to feature_manifest.json."""
    manifest = {
        "demographic": demographic,
        "psychographic": psychographic,
        "behavioral": behavioral,
        "all_features": all_features,
        "excluded_identifiers": [ID_COL]
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
