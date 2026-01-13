"""Custom PyTorch Dataset for CheXpert dataset with U-Ones label policy."""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional

from ..config import (
    RAW_DATA_DIR, IMAGE_SIZE, PATHOLOGY_CLASSES, 
    U_ONES_CLASSES, U_ZEROS_CLASSES
)
from ..utils import preprocess_image, apply_label_policy

logger = logging.getLogger(__name__)


class CheXpertDataset(Dataset):
    """
    Custom Dataset for CheXpert chest X-ray images.
    
    Handles:
    - DICOM and standard image formats
    - U-Ones/U-Zeros label policy for uncertain labels
    - Image preprocessing for BioViL-T
    """
    
    def __init__(
        self,
        csv_path: Path,
        data_dir: Path = RAW_DATA_DIR,
        transform: Optional[callable] = None,
        u_ones_classes: list = None,
        u_zeros_classes: list = None
    ):
        """
        Initialize CheXpert dataset.
        
        Args:
            csv_path: Path to CSV file (train.csv or valid.csv)
            data_dir: Root directory of CheXpert dataset
            transform: Optional transform to apply (not used, preprocessing is built-in)
            u_ones_classes: List of classes to apply U-Ones policy
            u_zeros_classes: List of classes to apply U-Zeros policy
        """
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.transform = transform
        
        # Label policy
        self.u_ones_classes = u_ones_classes if u_ones_classes is not None else U_ONES_CLASSES
        self.u_zeros_classes = u_zeros_classes if u_zeros_classes is not None else U_ZEROS_CLASSES
        
        # Load CSV
        logger.info(f"Loading dataset from {csv_path}")
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} samples")
        
        # Get pathology columns
        self.pathology_cols = [col for col in PATHOLOGY_CLASSES if col in self.df.columns]
        if len(self.pathology_cols) != len(PATHOLOGY_CLASSES):
            missing = set(PATHOLOGY_CLASSES) - set(self.pathology_cols)
            logger.warning(f"Missing pathology columns: {missing}")
        
        # Filter out rows with missing paths
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['Path'])
        if len(self.df) < initial_len:
            logger.warning(f"Removed {initial_len - len(self.df)} samples with missing paths")
        
        logger.info(f"Dataset initialized with {len(self.df)} valid samples")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - 'image': Preprocessed image tensor [3, 224, 224]
            - 'labels': Label tensor [14]
            - 'path': Image path (for debugging)
        """
        row = self.df.iloc[idx]
        
        # Get image path
        image_path = self.data_dir / row['Path']
        
        # Load and preprocess image
        try:
            image = preprocess_image(image_path, target_size=IMAGE_SIZE)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        
        # Get labels
        labels_dict = {pathology: row.get(pathology, np.nan) 
                      for pathology in self.pathology_cols}
        
        # Apply label policy
        labels_array = apply_label_policy(
            labels_dict, 
            self.u_ones_classes, 
            self.u_zeros_classes
        )
        
        # Convert to tensor
        labels = torch.from_numpy(labels_array)
        
        return {
            'image': image,
            'labels': labels,
            'path': str(row['Path'])
        }
    
    def get_label_statistics(self) -> Dict:
        """Get statistics about label distribution."""
        stats = {}
        
        for pathology in self.pathology_cols:
            if pathology not in self.df.columns:
                continue
            
            col = self.df[pathology]
            stats[pathology] = {
                'positive': int((col == 1.0).sum()),
                'negative': int((col == 0.0).sum()),
                'uncertain': int((col == -1.0).sum()),
                'not_mentioned': int(col.isna().sum()),
                'total': len(col)
            }
        
        return stats


def create_dataloaders(
    train_csv: Path,
    valid_csv: Path,
    data_dir: Path = RAW_DATA_DIR,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        train_csv: Path to train.csv
        valid_csv: Path to valid.csv
        data_dir: Root directory of CheXpert dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (for GPU)
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, valid_loader)
    """
    # Create datasets
    train_dataset = CheXpertDataset(train_csv, data_dir=data_dir)
    valid_dataset = CheXpertDataset(valid_csv, data_dir=data_dir)
    
    # Print label statistics
    logger.info("Training set label statistics:")
    train_stats = train_dataset.get_label_statistics()
    for pathology, stats in train_stats.items():
        logger.info(f"  {pathology}: +{stats['positive']} -{stats['negative']} "
                   f"?{stats['uncertain']} N/A{stats['not_mentioned']}")
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"Created DataLoaders: train={len(train_loader)} batches, "
               f"valid={len(valid_loader)} batches")
    
    return train_loader, valid_loader
