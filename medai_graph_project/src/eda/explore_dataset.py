"""Main EDA script for exploring CheXpert dataset structure."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from typing import Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_csv_files(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and valid.csv files."""
    train_csv = data_dir / "train.csv"
    valid_csv = data_dir / "valid.csv"
    
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv}")
    if not valid_csv.exists():
        raise FileNotFoundError(f"valid.csv not found at {valid_csv}")
    
    logger.info(f"Loading {train_csv}")
    train_df = pd.read_csv(train_csv)
    
    logger.info(f"Loading {valid_csv}")
    valid_df = pd.read_csv(valid_csv)
    
    return train_df, valid_df


def analyze_dataset_structure(train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                              data_dir: Path) -> Dict:
    """Analyze basic dataset structure."""
    logger.info("Analyzing dataset structure...")
    
    # Basic statistics
    stats = {
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "total_samples": len(train_df) + len(valid_df),
        "train_columns": list(train_df.columns),
        "valid_columns": list(valid_df.columns),
    }
    
    # Check if Path column exists
    if "Path" not in train_df.columns:
        raise ValueError("'Path' column not found in train.csv")
    
    # Count unique patients (if Patient column exists)
    if "Patient" in train_df.columns:
        stats["train_patients"] = train_df["Patient"].nunique()
        stats["valid_patients"] = valid_df["Patient"].nunique() if "Patient" in valid_df.columns else None
    
    # Check for missing values in Path column
    stats["train_missing_paths"] = train_df["Path"].isna().sum()
    stats["valid_missing_paths"] = valid_df["Path"].isna().sum()
    
    # Verify image paths exist
    logger.info("Verifying image paths...")
    train_image_dir = data_dir / "train"
    valid_image_dir = data_dir / "valid"
    
    train_existing = 0
    train_missing = 0
    
    # Sample check for train (check first 100 to avoid long runtime)
    sample_size = min(100, len(train_df))
    for idx, path in enumerate(train_df["Path"].head(sample_size)):
        if pd.isna(path):
            train_missing += 1
            continue
        full_path = data_dir / path
        if full_path.exists():
            train_existing += 1
        else:
            train_missing += 1
        if (idx + 1) % 10 == 0:
            logger.info(f"Checked {idx + 1}/{sample_size} train images...")
    
    stats["train_path_check_sample"] = sample_size
    stats["train_existing_images"] = train_existing
    stats["train_missing_images"] = train_missing
    
    # Check valid set
    valid_existing = 0
    valid_missing = 0
    sample_size = min(100, len(valid_df))
    for idx, path in enumerate(valid_df["Path"].head(sample_size)):
        if pd.isna(path):
            valid_missing += 1
            continue
        full_path = data_dir / path
        if full_path.exists():
            valid_existing += 1
        else:
            valid_missing += 1
        if (idx + 1) % 10 == 0:
            logger.info(f"Checked {idx + 1}/{sample_size} valid images...")
    
    stats["valid_path_check_sample"] = sample_size
    stats["valid_existing_images"] = valid_existing
    stats["valid_missing_images"] = valid_missing
    
    return stats


def get_pathology_columns(df: pd.DataFrame) -> list:
    """Get list of pathology label columns (exclude Path, Patient, etc.)."""
    exclude_cols = ["Path", "Patient", "Sex", "Age", "Frontal/Lateral", 
                    "AP/PA", "No Finding"]
    pathology_cols = [col for col in df.columns if col not in exclude_cols]
    return pathology_cols


def check_data_consistency(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> Dict:
    """Check for data inconsistencies."""
    logger.info("Checking data consistency...")
    
    issues = []
    
    # Check if columns match
    train_cols = set(train_df.columns)
    valid_cols = set(valid_df.columns)
    
    if train_cols != valid_cols:
        issues.append({
            "type": "column_mismatch",
            "train_only": list(train_cols - valid_cols),
            "valid_only": list(valid_cols - train_cols)
        })
    
    # Check for duplicate paths
    train_duplicates = train_df["Path"].duplicated().sum()
    valid_duplicates = valid_df["Path"].duplicated().sum()
    
    if train_duplicates > 0:
        issues.append({
            "type": "duplicate_paths",
            "dataset": "train",
            "count": int(train_duplicates)
        })
    
    if valid_duplicates > 0:
        issues.append({
            "type": "duplicate_paths",
            "dataset": "valid",
            "count": int(valid_duplicates)
        })
    
    return {
        "issues": issues,
        "num_issues": len(issues)
    }


def run_dataset_exploration(data_dir: Path, output_dir: Path) -> Dict:
    """Main function to run dataset exploration."""
    logger.info("=" * 60)
    logger.info("Starting Dataset Exploration")
    logger.info("=" * 60)
    
    # Load CSV files
    train_df, valid_df = load_csv_files(data_dir)
    
    # Analyze structure
    structure_stats = analyze_dataset_structure(train_df, valid_df, data_dir)
    
    # Check consistency
    consistency_check = check_data_consistency(train_df, valid_df)
    
    # Get pathology columns
    pathology_cols = get_pathology_columns(train_df)
    structure_stats["pathology_columns"] = pathology_cols
    structure_stats["num_pathology_classes"] = len(pathology_cols)
    
    # Combine results
    results = {
        "dataset_structure": structure_stats,
        "consistency_check": consistency_check,
        "pathology_columns": pathology_cols
    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eda_report.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(report_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Dataset exploration report saved to {report_path}")
    
    return results


if __name__ == "__main__":
    # Default paths (relative to project root)
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw" / "CheXpert-v1.0-small"
    output_dir = project_root / "outputs"
    
    results = run_dataset_exploration(data_dir, output_dir)
    logger.info("Dataset exploration completed!")
