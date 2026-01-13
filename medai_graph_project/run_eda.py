"""Entry point for Exploratory Data Analysis (EDA) of CheXpert dataset."""

import sys
from pathlib import Path
import logging

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from eda.explore_dataset import run_dataset_exploration
from eda.analyze_labels import run_label_analysis
from eda.visualize_samples import run_image_visualization
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "outputs" / "eda.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run complete EDA pipeline."""
    logger.info("=" * 80)
    logger.info("CHEXPERT DATASET - EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 80)
    
    # Define paths
    data_dir = project_root / "data" / "raw" / "CheXpert-v1.0-small"
    output_dir = project_root / "outputs"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists
    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        logger.error("Please ensure the CheXpert dataset is placed in the correct location.")
        return 1
    
    train_csv = data_dir / "train.csv"
    valid_csv = data_dir / "valid.csv"
    
    if not train_csv.exists() or not valid_csv.exists():
        logger.error(f"CSV files not found in {data_dir}")
        logger.error("Expected files: train.csv, valid.csv")
        return 1
    
    try:
        # Phase 1: Dataset Structure Exploration
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: Dataset Structure Exploration")
        logger.info("=" * 80)
        structure_results = run_dataset_exploration(data_dir, output_dir)
        logger.info("✓ Dataset structure exploration completed")
        
        # Phase 2: Label Distribution Analysis
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: Label Distribution Analysis")
        logger.info("=" * 80)
        
        # Load data for label analysis
        train_df = pd.read_csv(train_csv)
        valid_df = pd.read_csv(valid_csv)
        
        label_results = run_label_analysis(train_df, valid_df, output_dir)
        logger.info("✓ Label distribution analysis completed")
        
        # Phase 3: Image Visualization and Quality Assessment
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: Image Visualization and Quality Assessment")
        logger.info("=" * 80)
        
        image_results = run_image_visualization(train_df, valid_df, data_dir, output_dir)
        logger.info("✓ Image visualization and quality assessment completed")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("EDA SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total train samples: {structure_results['dataset_structure']['train_samples']}")
        logger.info(f"Total valid samples: {structure_results['dataset_structure']['valid_samples']}")
        logger.info(f"Pathology classes: {structure_results['dataset_structure']['num_pathology_classes']}")
        logger.info(f"\nEDA Report saved to: {output_dir / 'eda_report.json'}")
        logger.info(f"Visualizations saved to: {output_dir / 'eda_visualizations'}")
        logger.info(f"Log file saved to: {output_dir / 'eda.log'}")
        logger.info("\n✓ EDA completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during EDA: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
