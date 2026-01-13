"""Utility functions for image preprocessing, metrics calculation, and logging."""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import logging
from typing import Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import json

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from .config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD

logger = logging.getLogger(__name__)


def load_dicom_image(dicom_path: Path) -> Image.Image:
    """Load DICOM image and convert to PIL Image."""
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required. Install with: pip install pydicom")
    
    try:
        ds = pydicom.dcmread(str(dicom_path))
        image_array = ds.pixel_array
        
        # Normalize to 0-255 range
        if image_array.max() > 255:
            image_array = ((image_array - image_array.min()) / 
                          (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
        
        # Convert to PIL Image (grayscale)
        image = Image.fromarray(image_array, mode='L')
        
        # Convert to RGB (BioViL expects 3 channels)
        image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        logger.error(f"Error loading DICOM file {dicom_path}: {e}")
        raise


def preprocess_image(image_path: Path, target_size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Preprocess image for BioViL-T model.
    
    Args:
        image_path: Path to image file (DICOM or standard format)
        target_size: Target size for resizing (default: 224)
    
    Returns:
        Preprocessed image tensor of shape [3, 224, 224]
    """
    # Load image
    if image_path.suffix.lower() in ['.dcm', '.dicom']:
        image = load_dicom_image(image_path)
    else:
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    # Resize with aspect ratio preservation (center crop)
    image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Create square image with center crop
    width, height = image.size
    if width != height:
        # Center crop to square
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        image = image.crop((left, top, right, bottom))
    
    # Resize to target size
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to tensor and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize using BioViL mean/std
    mean = np.array(IMAGE_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGE_STD).reshape(1, 1, 3)
    image_array = (image_array - mean) / std
    
    # Convert to tensor [C, H, W]
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    
    return image_tensor


def apply_label_policy(labels: dict, u_ones_classes: list, 
                       u_zeros_classes: list = None) -> np.ndarray:
    """
    Apply U-Ones/U-Zeros policy to uncertain labels.
    
    Args:
        labels: Dictionary with pathology names as keys and label values
        u_ones_classes: List of classes to apply U-Ones policy (-1 -> 1)
        u_zeros_classes: List of classes to apply U-Zeros policy (-1 -> 0)
    
    Returns:
        Numpy array of processed labels
    """
    if u_zeros_classes is None:
        u_zeros_classes = []
    
    processed_labels = []
    
    for pathology, value in labels.items():
        if pd.isna(value) or value is None:
            # Not mentioned -> 0
            processed_labels.append(0.0)
        elif value == -1.0:
            # Uncertain label
            if pathology in u_ones_classes:
                processed_labels.append(1.0)  # U-Ones
            elif pathology in u_zeros_classes:
                processed_labels.append(0.0)  # U-Zeros
            else:
                # Default: U-Ones
                processed_labels.append(1.0)
        else:
            # Positive (1) or Negative (0)
            processed_labels.append(float(value))
    
    return np.array(processed_labels, dtype=np.float32)


def calculate_auc_scores(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: list) -> dict:
    """
    Calculate AUC scores for each class and macro AUC.
    
    Args:
        y_true: Ground truth labels [N, num_classes]
        y_pred: Predicted probabilities [N, num_classes]
        class_names: List of class names
    
    Returns:
        Dictionary with per-class AUC and macro AUC
    """
    results = {}
    per_class_aucs = []
    
    for idx, class_name in enumerate(class_names):
        try:
            # Skip if all labels are the same (no variance)
            if len(np.unique(y_true[:, idx])) < 2:
                results[class_name] = None
                continue
            
            auc = roc_auc_score(y_true[:, idx], y_pred[:, idx])
            results[class_name] = float(auc)
            per_class_aucs.append(auc)
        except Exception as e:
            logger.warning(f"Could not calculate AUC for {class_name}: {e}")
            results[class_name] = None
    
    # Calculate macro AUC (average of all valid AUCs)
    if per_class_aucs:
        results["macro_auc"] = float(np.mean(per_class_aucs))
    else:
        results["macro_auc"] = None
    
    return results


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_binary: np.ndarray, class_names: list) -> dict:
    """
    Calculate comprehensive metrics (AUC, precision, recall, F1).
    
    Args:
        y_true: Ground truth labels [N, num_classes]
        y_pred: Predicted probabilities [N, num_classes]
        y_pred_binary: Binary predictions [N, num_classes]
        class_names: List of class names
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Calculate AUC scores
    auc_scores = calculate_auc_scores(y_true, y_pred, class_names)
    metrics["auc"] = auc_scores
    
    # Calculate precision, recall, F1 for each class
    precision_list = []
    recall_list = []
    f1_list = []
    
    for idx, class_name in enumerate(class_names):
        try:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true[:, idx], y_pred_binary[:, idx], 
                average='binary', zero_division=0
            )
            metrics[class_name] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            }
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
        except Exception as e:
            logger.warning(f"Could not calculate metrics for {class_name}: {e}")
            metrics[class_name] = {
                "precision": None,
                "recall": None,
                "f1": None
            }
    
    # Calculate macro averages
    if precision_list:
        metrics["macro_precision"] = float(np.mean(precision_list))
        metrics["macro_recall"] = float(np.mean(recall_list))
        metrics["macro_f1"] = float(np.mean(f1_list))
    
    return metrics


def save_metrics(metrics: dict, output_path: Path):
    """Save metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {output_path}")
