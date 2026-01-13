"""Global configuration constants for the medical image classification project."""

from pathlib import Path
import torch

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "CheXpert-v1.0-small"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TEST_IMAGES_DIR = DATA_DIR / "test_images"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Create directories if they don't exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_AVAILABLE = torch.cuda.is_available()

# Image preprocessing parameters
IMAGE_SIZE = 224  # BioViL-T standard input size
IMAGE_MEAN = [0.485, 0.456, 0.406]  # BioViL normalization mean
IMAGE_STD = [0.229, 0.224, 0.225]   # BioViL normalization std

# CheXpert pathology classes (14 classes)
PATHOLOGY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices"
]

# Label policy: U-Ones (map uncertain -1 to 1) for specific classes
# As per PRD: Atelectasis, Edema, Cardiomegaly use U-Ones
U_ONES_CLASSES = ["Atelectasis", "Edema", "Cardiomegaly"]

# U-Zeros (map uncertain -1 to 0) for other classes (optional, can be changed)
# Default: Use U-Ones for all classes except those specified
U_ZEROS_CLASSES = []  # Can add "Consolidation", "Pleural Effusion" if needed

# DataLoader parameters
BATCH_SIZE = 16  # Adjust based on GPU memory
NUM_WORKERS = 4  # Adjust based on system
PIN_MEMORY = True if CUDA_AVAILABLE else False

# Training parameters
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
GRADIENT_ACCUMULATION_STEPS = 4  # For laptop GPU efficiency
MIXED_PRECISION = True  # Use torch.amp for memory efficiency

# Model parameters
HIDDEN_DIM = 768  # BioViL-T feature dimension
NUM_ATTENTION_HEADS = 8
DROPOUT = 0.2

# Graph parameters
NUM_PATHOLOGY_NODES = 14
NUM_ANATOMY_NODES = 5
NUM_TOTAL_NODES = NUM_PATHOLOGY_NODES + NUM_ANATOMY_NODES

# Validation and checkpointing
VALIDATION_INTERVAL = 1  # Validate every N epochs
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
BEST_MODEL_METRIC = "macro_auc"  # Metric to track for best model

# Logging
LOG_INTERVAL = 10  # Log every N batches during training

# Inference parameters
INFERENCE_THRESHOLD = 0.5  # Threshold for binary classification
ATTENTION_THRESHOLD = 0.2  # Threshold for graph visualization (ignore edges < 0.2)

# Print configuration on import
if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"U-Ones Classes: {U_ONES_CLASSES}")
    print(f"Pathology Classes: {len(PATHOLOGY_CLASSES)}")
    print("=" * 60)
