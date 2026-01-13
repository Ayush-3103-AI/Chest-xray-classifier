"""Label distribution analysis for CheXpert dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_pathology_columns(df: pd.DataFrame) -> list:
    """Get list of pathology label columns."""
    exclude_cols = ["Path", "Patient", "Sex", "Age", "Frontal/Lateral", 
                    "AP/PA", "No Finding"]
    pathology_cols = [col for col in df.columns if col not in exclude_cols]
    return pathology_cols


def analyze_label_distribution(df: pd.DataFrame, pathology_cols: list) -> dict:
    """Analyze label distribution for each pathology."""
    logger.info("Analyzing label distributions...")
    
    label_stats = {}
    
    for col in pathology_cols:
        if col not in df.columns:
            continue
        
        # Count label values
        value_counts = df[col].value_counts(dropna=False)
        
        stats = {
            "positive": int(value_counts.get(1.0, 0)),
            "negative": int(value_counts.get(0.0, 0)),
            "uncertain": int(value_counts.get(-1.0, 0)),
            "not_mentioned": int(value_counts.get(np.nan, 0)) if np.nan in value_counts.index else int(df[col].isna().sum()),
            "total": len(df)
        }
        
        # Calculate percentages
        stats["positive_pct"] = (stats["positive"] / stats["total"]) * 100
        stats["negative_pct"] = (stats["negative"] / stats["total"]) * 100
        stats["uncertain_pct"] = (stats["uncertain"] / stats["total"]) * 100
        stats["not_mentioned_pct"] = (stats["not_mentioned"] / stats["total"]) * 100
        
        label_stats[col] = stats
    
    return label_stats


def analyze_class_imbalance(label_stats: dict) -> dict:
    """Analyze class imbalance."""
    logger.info("Analyzing class imbalance...")
    
    imbalance_info = {}
    
    for pathology, stats in label_stats.items():
        positive_ratio = stats["positive"] / stats["total"]
        
        # Classify imbalance level
        if positive_ratio < 0.01:
            level = "very_rare"
        elif positive_ratio < 0.05:
            level = "rare"
        elif positive_ratio < 0.20:
            level = "uncommon"
        elif positive_ratio < 0.50:
            level = "common"
        else:
            level = "very_common"
        
        imbalance_info[pathology] = {
            "positive_ratio": float(positive_ratio),
            "imbalance_level": level
        }
    
    return imbalance_info


def analyze_cooccurrence(df: pd.DataFrame, pathology_cols: list) -> pd.DataFrame:
    """Analyze co-occurrence patterns between pathologies."""
    logger.info("Analyzing co-occurrence patterns...")
    
    # Create binary matrix (1 for positive, 0 otherwise)
    binary_df = df[pathology_cols].copy()
    binary_df = binary_df.replace({-1: 1, np.nan: 0})  # Treat uncertain as positive for co-occurrence
    binary_df = (binary_df == 1).astype(int)
    
    # Calculate co-occurrence matrix
    cooccurrence = binary_df.T.dot(binary_df)
    
    # Normalize by number of samples
    cooccurrence_pct = (cooccurrence / len(df)) * 100
    
    return cooccurrence_pct


def visualize_label_distributions(label_stats: dict, output_dir: Path):
    """Create visualizations for label distributions."""
    logger.info("Creating label distribution visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    pathologies = list(label_stats.keys())
    positive_counts = [label_stats[p]["positive"] for p in pathologies]
    negative_counts = [label_stats[p]["negative"] for p in pathologies]
    uncertain_counts = [label_stats[p]["uncertain"] for p in pathologies]
    
    # Bar chart: Counts
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(pathologies))
    width = 0.25
    
    ax.bar(x - width, positive_counts, width, label='Positive', color='#2ecc71')
    ax.bar(x, negative_counts, width, label='Negative', color='#e74c3c')
    ax.bar(x + width, uncertain_counts, width, label='Uncertain', color='#f39c12')
    
    ax.set_xlabel('Pathology', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Label Distribution by Pathology', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pathologies, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution_counts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bar chart: Percentages
    fig, ax = plt.subplots(figsize=(14, 8))
    
    positive_pcts = [label_stats[p]["positive_pct"] for p in pathologies]
    uncertain_pcts = [label_stats[p]["uncertain_pct"] for p in pathologies]
    
    x = np.arange(len(pathologies))
    width = 0.35
    
    ax.bar(x - width/2, positive_pcts, width, label='Positive %', color='#2ecc71')
    ax.bar(x + width/2, uncertain_pcts, width, label='Uncertain %', color='#f39c12')
    
    ax.set_xlabel('Pathology', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Positive and Uncertain Label Percentages', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pathologies, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution_percentages.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Label distribution visualizations saved to {output_dir}")


def visualize_cooccurrence(cooccurrence_matrix: pd.DataFrame, output_dir: Path):
    """Create heatmap for co-occurrence patterns."""
    logger.info("Creating co-occurrence heatmap...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(cooccurrence_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Pathology Co-occurrence Matrix (% of samples)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Pathology', fontsize=12)
    ax.set_ylabel('Pathology', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "cooccurrence_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Co-occurrence heatmap saved to {output_dir}")


def run_label_analysis(train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                       output_dir: Path) -> dict:
    """Main function to run label analysis."""
    logger.info("=" * 60)
    logger.info("Starting Label Distribution Analysis")
    logger.info("=" * 60)
    
    pathology_cols = get_pathology_columns(train_df)
    logger.info(f"Found {len(pathology_cols)} pathology classes: {pathology_cols}")
    
    # Analyze train set
    logger.info("\n--- Train Set Analysis ---")
    train_label_stats = analyze_label_distribution(train_df, pathology_cols)
    
    # Analyze valid set
    logger.info("\n--- Validation Set Analysis ---")
    valid_label_stats = analyze_label_distribution(valid_df, pathology_cols)
    
    # Class imbalance analysis
    train_imbalance = analyze_class_imbalance(train_label_stats)
    valid_imbalance = analyze_class_imbalance(valid_label_stats)
    
    # Co-occurrence analysis (on train set)
    logger.info("\n--- Co-occurrence Analysis ---")
    cooccurrence_matrix = analyze_cooccurrence(train_df, pathology_cols)
    
    # Create visualizations
    viz_dir = output_dir / "eda_visualizations"
    visualize_label_distributions(train_label_stats, viz_dir)
    visualize_cooccurrence(cooccurrence_matrix, viz_dir)
    
    # Compile results
    results = {
        "train_label_stats": train_label_stats,
        "valid_label_stats": valid_label_stats,
        "train_imbalance": train_imbalance,
        "valid_imbalance": valid_imbalance,
        "cooccurrence_matrix": cooccurrence_matrix.to_dict()
    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update existing EDA report or create new one
    report_path = output_dir / "eda_report.json"
    if report_path.exists():
        with open(report_path, 'r') as f:
            existing_report = json.load(f)
    else:
        existing_report = {}
    
    existing_report["label_analysis"] = {
        "train_label_stats": train_label_stats,
        "valid_label_stats": valid_label_stats,
        "train_imbalance": train_imbalance,
        "valid_imbalance": valid_imbalance
    }
    
    # Convert cooccurrence matrix to serializable format
    existing_report["cooccurrence_matrix"] = cooccurrence_matrix.to_dict()
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return obj
    
    serializable_report = convert_to_serializable(existing_report)
    
    with open(report_path, 'w') as f:
        json.dump(serializable_report, f, indent=2)
    
    logger.info(f"Label analysis results saved to {report_path}")
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw" / "CheXpert-v1.0-small"
    output_dir = project_root / "outputs"
    
    # Load data
    train_df = pd.read_csv(data_dir / "train.csv")
    valid_df = pd.read_csv(data_dir / "valid.csv")
    
    results = run_label_analysis(train_df, valid_df, output_dir)
    logger.info("Label analysis completed!")
