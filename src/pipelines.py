"""
Leakage-safe preprocessing pipelines and core Naive Bayes model registry.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer


class SafeOrdinalToNonNegative(BaseEstimator, TransformerMixin):
    """
    Leakage-safe Ordinal Encoder for CategoricalNB.
    Known categories are mapped to positive integers (1..K).
    Unseen/unknown categories are mapped to reserved code 0 (via unknown_value=-1 shifted by +1).
    Guarantees all transformed output values are strictly non-negative integers (>= 0).
    """
    def __init__(self):
        self.enc = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )

    def fit(self, X, y=None):
        self.enc.fit(X)
        return self

    def transform(self, X):
        X_trans = self.enc.transform(X).astype(int)
        return X_trans + 1


def build_core_pipelines(
    numeric_cols: List[str],
    categorical_cols: List[str],
    all_features: List[str]
) -> Dict[str, Pipeline]:
    """
    Construct leakage-safe pipelines for the four required core models:
    1. DummyClassifier (most_frequent)
    2. GaussianNB (continuous numeric features only)
    3. BernoulliNB (discretized-and-one-hot numeric + one-hot categorical)
    4. CategoricalNB (discretized ordinal numeric + safe non-negative categorical)
    """
    pipelines: Dict[str, Pipeline] = {}
    
    # 1. Baseline: DummyClassifier (most_frequent)
    dummy_prep = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
            ]), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), categorical_cols)
        ],
        remainder='drop'
    )
    pipelines['Dummy_most_frequent'] = Pipeline([
        ('prep', dummy_prep),
        ('model', DummyClassifier(strategy='most_frequent'))
    ])
    
    # 2. GaussianNB (continuous numeric features only)
    if numeric_cols:
        gaussian_prep = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                ]), numeric_cols)
            ],
            remainder='drop'
        )
        pipelines['GaussianNB_numeric_only'] = Pipeline([
            ('prep', gaussian_prep),
            ('model', GaussianNB(var_smoothing=1e-9))
        ])

    # 3. BernoulliNB (binary/discretized-and-one-hot representation)
    bernoulli_num = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('bins', KBinsDiscretizer(n_bins=5, encode='onehot', strategy='quantile'))
    ])
    bernoulli_cat = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    bernoulli_prep = ColumnTransformer(
        transformers=[
            ('num_bins', bernoulli_num, numeric_cols) if numeric_cols else ('num_dummy', 'drop', []),
            ('cat_ohe', bernoulli_cat, categorical_cols) if categorical_cols else ('cat_dummy', 'drop', [])
        ],
        remainder='drop'
    )
    pipelines['BernoulliNB'] = Pipeline([
        ('prep', bernoulli_prep),
        ('model', BernoulliNB(alpha=1.0, binarize=0.0))
    ])

    # 4. CategoricalNB (complete mixed-feature version with safe non-negative integer encoding)
    cat_num = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('bins', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'))
    ])
    cat_cat = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('safe_ordinal', SafeOrdinalToNonNegative())
    ])
    categorical_nb_prep = ColumnTransformer(
        transformers=[
            ('num_ordinal', cat_num, numeric_cols) if numeric_cols else ('num_dummy', 'drop', []),
            ('cat_safe_ordinal', cat_cat, categorical_cols) if categorical_cols else ('cat_dummy', 'drop', [])
        ],
        remainder='drop'
    )
    pipelines['CategoricalNB_mixed'] = Pipeline([
        ('prep', categorical_nb_prep),
        ('model', CategoricalNB(alpha=1.0))
    ])

    return pipelines


def assert_categorical_nb_non_negative(pipeline: Pipeline, X_train: pd.DataFrame) -> None:
    """
    Transform training data using CategoricalNB preprocessor and assert
    that all values are strictly non-negative integers (>= 0).
    """
    preprocessor = pipeline.named_steps['prep']
    Xt = preprocessor.fit_transform(X_train)
    min_val = np.nanmin(np.asarray(Xt))
    if min_val < 0:
        raise AssertionError(f"CategoricalNB preprocessor produced negative value: {min_val}")
    print(f"CategoricalNB non-negativity assertion PASSED: min transformed value = {min_val} (>= 0)")
