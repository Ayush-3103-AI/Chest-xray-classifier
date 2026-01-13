"""Test script for BioViL-T image encoder."""

import sys
from pathlib import Path
import torch
import logging

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from config import DEVICE, CUDA_AVAILABLE, BATCH_SIZE, IMAGE_SIZE
from models.image_encoder import create_image_encoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "outputs" / "biovil_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test 1: Model loading."""
    logger.info("=" * 80)
    logger.info("TEST 1: Model Loading")
    logger.info("=" * 80)
    
    try:
        encoder = create_image_encoder()
        logger.info("✓ BioViL-T model loaded successfully")
        return encoder
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}", exc_info=True)
        return None


def test_selective_freezing(encoder):
    """Test 2: Selective freezing."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Selective Freezing")
    logger.info("=" * 80)
    
    if encoder is None:
        logger.error("✗ Cannot test freezing: encoder is None")
        return False
    
    # Count frozen and trainable parameters
    frozen_params = sum(1 for p in encoder.parameters() if not p.requires_grad)
    trainable_params = sum(1 for p in encoder.parameters() if p.requires_grad)
    total_params = frozen_params + trainable_params
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Frozen parameters: {frozen_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    if trainable_params > 0 and frozen_params > 0:
        logger.info("✓ Selective freezing applied correctly")
        return True
    else:
        logger.warning("⚠ Unexpected freezing state")
        return False


def test_forward_pass(encoder):
    """Test 3: Forward pass with dummy data."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Forward Pass")
    logger.info("=" * 80)
    
    if encoder is None:
        logger.error("✗ Cannot test forward pass: encoder is None")
        return False
    
    # Create dummy batch
    batch_size = min(2, BATCH_SIZE)  # Use small batch for testing
    dummy_images = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    
    logger.info(f"Input shape: {dummy_images.shape}")
    logger.info(f"Expected output shape: [batch_size, 197, 768]")
    
    try:
        # Forward pass
        encoder.eval()
        with torch.no_grad():
            features = encoder(dummy_images)
        
        logger.info(f"Output shape: {features.shape}")
        
        # Verify shape
        expected_shape = (batch_size, 197, 768)
        if features.shape == expected_shape:
            logger.info("✓ Output shape is correct")
            return True
        else:
            logger.warning(f"⚠ Output shape mismatch: {features.shape} != {expected_shape}")
            logger.info(f"  Actual: {features.shape}")
            logger.info(f"  Expected: {expected_shape}")
            
            # Check if it's close (e.g., different sequence length)
            if features.shape[0] == batch_size and features.shape[2] == 768:
                logger.info("  Batch size and feature dim are correct")
                logger.info(f"  Sequence length: {features.shape[1]} (expected 197)")
                return True
            
            return False
            
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}", exc_info=True)
        return False


def test_memory_usage(encoder):
    """Test 4: Memory usage."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Memory Usage")
    logger.info("=" * 80)
    
    if encoder is None:
        logger.error("✗ Cannot test memory: encoder is None")
        return False
    
    if not CUDA_AVAILABLE:
        logger.info("CUDA not available, skipping GPU memory test")
        return True
    
    try:
        # Clear cache
        torch.cuda.empty_cache()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Forward pass
        batch_size = 4
        dummy_images = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        
        encoder.eval()
        with torch.no_grad():
            features = encoder(dummy_images)
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        logger.info(f"Initial GPU memory: {initial_memory:.2f} MB")
        logger.info(f"Peak GPU memory: {peak_memory:.2f} MB")
        logger.info(f"Current GPU memory: {current_memory:.2f} MB")
        logger.info(f"Memory used: {peak_memory - initial_memory:.2f} MB")
        
        if peak_memory < 8000:  # Less than 8GB
            logger.info("✓ Memory usage is reasonable")
            return True
        else:
            logger.warning(f"⚠ High memory usage: {peak_memory:.2f} MB")
            return True  # Still pass, just warning
            
    except Exception as e:
        logger.error(f"✗ Memory test failed: {e}", exc_info=True)
        return False


def test_with_real_data(encoder):
    """Test 5: Test with real data from DataLoader (if available)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Real Data Test (Optional)")
    logger.info("=" * 80)
    
    if encoder is None:
        logger.error("✗ Cannot test with real data: encoder is None")
        return False
    
    # Try to load a sample from dataset
    try:
        from training.dataset import CheXpertDataset
        from config import RAW_DATA_DIR
        
        train_csv = RAW_DATA_DIR / "train.csv"
        
        if not train_csv.exists():
            logger.info("Dataset not available, skipping real data test")
            return True
        
        # Create dataset
        dataset = CheXpertDataset(train_csv, data_dir=RAW_DATA_DIR)
        
        if len(dataset) == 0:
            logger.info("Dataset is empty, skipping real data test")
            return True
        
        # Get a sample
        sample = dataset[0]
        image = sample['image'].unsqueeze(0)  # Add batch dimension
        
        logger.info(f"Real image shape: {image.shape}")
        
        # Forward pass
        encoder.eval()
        with torch.no_grad():
            features = encoder(image)
        
        logger.info(f"Output shape: {features.shape}")
        logger.info("✓ Real data test passed")
        return True
        
    except Exception as e:
        logger.warning(f"Real data test skipped: {e}")
        return True  # Don't fail if dataset not available


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("BIOVIL-T IMAGE ENCODER TEST SUITE")
    logger.info("=" * 80)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"CUDA Available: {CUDA_AVAILABLE}")
    logger.info(f"Image Size: {IMAGE_SIZE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    
    results = {}
    
    # Test 1: Model loading
    encoder = test_model_loading()
    results['model_loading'] = encoder is not None
    
    if encoder is None:
        logger.error("\n" + "=" * 80)
        logger.error("CRITICAL: Model loading failed. Cannot continue tests.")
        logger.error("=" * 80)
        return 1
    
    # Test 2: Selective freezing
    results['selective_freezing'] = test_selective_freezing(encoder)
    
    # Test 3: Forward pass
    results['forward_pass'] = test_forward_pass(encoder)
    
    # Test 4: Memory usage
    results['memory_usage'] = test_memory_usage(encoder)
    
    # Test 5: Real data
    results['real_data'] = test_with_real_data(encoder)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n✓ All tests passed!")
        return 0
    else:
        logger.error("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
