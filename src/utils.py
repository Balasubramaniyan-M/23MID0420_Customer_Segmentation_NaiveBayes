"""
Utility functions for environment setup, version tracking, file hashing, and random seed control.
"""

import sys
import json
import hashlib
import platform
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import sklearn


def set_seed(seed: int = 42) -> None:
    """Set global random seed for reproducibility."""
    np.random.seed(seed)


def calculate_file_hash(file_path: Path) -> str:
    """Compute SHA-256 checksum of a target file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")
    
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_environment_versions() -> Dict[str, str]:
    """Record current Python runtime and key package versions."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def ensure_directories(base_dir: Path) -> Dict[str, Path]:
    """Ensure all required output directories exist."""
    base = Path(base_dir)
    dirs = {
        "data": base / "data",
        "artifacts": base / "outputs" / "artifacts",
        "results": base / "outputs" / "results",
        "figures": base / "outputs" / "figures",
        "models": base / "outputs" / "models",
        "reports": base / "reports",
        "notebooks": base / "notebooks",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_versions_artifact(output_path: Path) -> Dict[str, str]:
    """Save environment versions to JSON artifact."""
    versions = get_environment_versions()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2)
    return versions
