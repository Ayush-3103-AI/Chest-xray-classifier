"""Entry point for data preparation and validation."""

import sys
from pathlib import Path
import logging
import torch

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from config import (
    RAW_DATA_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    PATHOLOGY_CLASSES, U_ONES_CLASSES
)
from training.dataset import create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "outputs" / "data_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run data preparation and validation."""
    logger.info("=" * 80)
    logger.info("DATA PREPARATION AND VALIDATION")
    logger.info("=" * 80)
    
    # Check if dataset exists
    if not RAW_DATA_DIR.exists():
        logger.error(f"Dataset directory not found: {RAW_DATA_DIR}")
        logger.error("Please ensure the CheXpert dataset is placed in the correct location.")
        return 1
    
    train_csv = RAW_DATA_DIR / "train.csv"
    valid_csv = RAW_DATA_DIR / "valid.csv"
    
    if not train_csv.exists() or not valid_csv.exists():
        logger.error(f"CSV files not found in {RAW_DATA_DIR}")
        logger.error("Expected files: train.csv, valid.csv")
        return 1
    
    try:
        # Create DataLoaders
        logger.info("\nCreating DataLoaders...")
        logger.info(f"Batch size: {BATCH_SIZE}")
        logger.info(f"Num workers: {NUM_WORKERS}")
        logger.info(f"Pin memory: {PIN_MEMORY}")
        logger.info(f"U-Ones classes: {U_ONES_CLASSES}")
        
        train_loader, valid_loader = create_dataloaders(
            train_csv=train_csv,
            valid_csv=valid_csv,
            data_dir=RAW_DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle_train=True
        )
        
        # Test DataLoader iteration
        logger.info("\n" + "=" * 80)
        logger.info("Testing DataLoader iteration...")
        logger.info("=" * 80)
        
        # Test train loader
        logger.info("\nTesting train DataLoader...")
        train_batch = next(iter(train_loader))
        logger.info(f"Train batch keys: {train_batch.keys()}")
        logger.info(f"Train image shape: {train_batch['image'].shape}")
        logger.info(f"Train labels shape: {train_batch['labels'].shape}")
        logger.info(f"Expected image shape: [batch_size, 3, 224, 224]")
        logger.info(f"Expected labels shape: [batch_size, {len(PATHOLOGY_CLASSES)}]")
        
        # Verify shapes
        batch_size_actual = train_batch['image'].shape[0]
        expected_image_shape = (batch_size_actual, 3, 224, 224)
        expected_labels_shape = (batch_size_actual, len(PATHOLOGY_CLASSES))
        
        if train_batch['image'].shape == expected_image_shape:
            logger.info("✓ Train image shape is correct")
        else:
            logger.error(f"✗ Train image shape mismatch: {train_batch['image'].shape} != {expected_image_shape}")
            return 1
        
        if train_batch['labels'].shape == expected_labels_shape:
            logger.info("✓ Train labels shape is correct")
        else:
            logger.error(f"✗ Train labels shape mismatch: {train_batch['labels'].shape} != {expected_labels_shape}")
            return 1
        
        # Test valid loader
        logger.info("\nTesting validation DataLoader...")
        valid_batch = next(iter(valid_loader))
        logger.info(f"Valid image shape: {valid_batch['image'].shape}")
        logger.info(f"Valid labels shape: {valid_batch['labels'].shape}")
        
        if valid_batch['image'].shape == expected_image_shape:
            logger.info("✓ Valid image shape is correct")
        else:
            logger.error(f"✗ Valid image shape mismatch")
            return 1
        
        if valid_batch['labels'].shape == expected_labels_shape:
            logger.info("✓ Valid labels shape is correct")
        else:
            logger.error(f"✗ Valid labels shape mismatch")
            return 1
        
        # Check label values
        logger.info("\nChecking label values...")
        unique_labels = torch.unique(train_batch['labels'])
        logger.info(f"Unique label values in batch: {unique_labels.tolist()}")
        logger.info("Expected: [0.0, 1.0] (after U-Ones policy applied)")
        
        if set(unique_labels.tolist()).issubset({0.0, 1.0}):
            logger.info("✓ Label values are correct (0.0 and 1.0 only)")
        else:
            logger.warning(f"⚠ Unexpected label values found: {unique_labels.tolist()}")
        
        # Test multiple batches
        logger.info("\nTesting multiple batch iterations...")
        num_test_batches = min(5, len(train_loader))
        for i, batch in enumerate(train_loader):
            if i >= num_test_batches:
                break
            logger.info(f"  Batch {i+1}/{num_test_batches}: OK")
        
        logger.info("\n" + "=" * 80)
        logger.info("DATA PREPARATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✓ Train DataLoader: {len(train_loader)} batches")
        logger.info(f"✓ Validation DataLoader: {len(valid_loader)} batches")
        logger.info(f"✓ Batch size: {BATCH_SIZE}")
        logger.info(f"✓ Image shape: [batch_size, 3, 224, 224]")
        logger.info(f"✓ Labels shape: [batch_size, {len(PATHOLOGY_CLASSES)}]")
        logger.info(f"✓ Label policy applied: U-Ones for {U_ONES_CLASSES}")
        logger.info("\n✓ Data preparation completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during data preparation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
