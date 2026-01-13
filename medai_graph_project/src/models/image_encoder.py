"""BioViL-T image encoder with selective freezing."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from typing import Tuple, Optional
import logging

from ..config import DEVICE, HIDDEN_DIM

logger = logging.getLogger(__name__)


class BioViLImageEncoder(nn.Module):
    """
    BioViL-T image encoder wrapper with selective freezing.
    
    Freezes first 3 ResNet blocks to prevent catastrophic forgetting
    and save memory. Only fine-tunes Transformer layers and last ResNet block.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedVLP-BioViL-T",
        freeze_resnet_blocks: int = 3,
        output_dim: int = HIDDEN_DIM
    ):
        """
        Initialize BioViL-T image encoder.
        
        Args:
            model_name: Hugging Face model name
            freeze_resnet_blocks: Number of ResNet blocks to freeze (default: 3)
            output_dim: Output feature dimension (default: 768)
        """
        super().__init__()
        
        self.model_name = model_name
        self.freeze_resnet_blocks = freeze_resnet_blocks
        self.output_dim = output_dim
        
        logger.info(f"Loading BioViL-T model: {model_name}")
        
        try:
            # Load BioViL-T model from Hugging Face
            self.model = AutoModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            logger.info("BioViL-T model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BioViL-T model: {e}")
            raise
        
        # Access the vision encoder
        # BioViL-T structure: model.vision_model contains the vision encoder
        if hasattr(self.model, 'vision_model'):
            self.vision_model = self.model.vision_model
        elif hasattr(self.model, 'image_encoder'):
            self.vision_model = self.model.image_encoder
        else:
            # Try to find vision-related attributes
            logger.warning("Could not find vision_model or image_encoder attribute")
            logger.info(f"Available attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
            # Use the model directly as fallback
            self.vision_model = self.model
        
        # Apply selective freezing
        self._apply_selective_freezing()
        
        # Move to device
        self.to(DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {frozen_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    def _apply_selective_freezing(self):
        """Apply selective freezing to ResNet blocks."""
        logger.info(f"Applying selective freezing: freezing first {self.freeze_resnet_blocks} ResNet blocks")
        
        # Try to find ResNet blocks in the vision model
        # BioViL-T uses a hybrid architecture with ResNet50 + ViT
        
        # Method 1: Look for ResNet blocks in the model
        resnet_blocks_found = 0
        
        # Check if vision_model has a resnet or backbone attribute
        if hasattr(self.vision_model, 'resnet') or hasattr(self.vision_model, 'backbone'):
            resnet = getattr(self.vision_model, 'resnet', None) or getattr(self.vision_model, 'backbone', None)
            
            # ResNet50 typically has 4 blocks (layer1, layer2, layer3, layer4)
            if hasattr(resnet, 'layer1'):
                # Freeze first N blocks
                blocks_to_freeze = ['layer1', 'layer2', 'layer3'][:self.freeze_resnet_blocks]
                
                for block_name in blocks_to_freeze:
                    if hasattr(resnet, block_name):
                        block = getattr(resnet, block_name)
                        for param in block.parameters():
                            param.requires_grad = False
                        resnet_blocks_found += 1
                        logger.info(f"Frozen {block_name}")
                
                # Keep layer4 trainable
                if hasattr(resnet, 'layer4'):
                    for param in resnet.layer4.parameters():
                        param.requires_grad = True
                    logger.info("layer4 kept trainable")
        
        # If no ResNet blocks found, try alternative approach
        # Freeze early layers of the vision model
        if resnet_blocks_found == 0:
            logger.warning("Could not find ResNet blocks. Attempting to freeze early layers...")
            
            # Get all named modules
            modules = list(self.vision_model.named_modules())
            
            # Freeze first portion of modules (excluding normalization and final layers)
            num_modules = len(modules)
            freeze_ratio = self.freeze_resnet_blocks / 4.0  # Assuming 4 blocks total
            freeze_until = int(num_modules * freeze_ratio)
            
            for idx, (name, module) in enumerate(modules[:freeze_until]):
                # Skip normalization layers and final layers
                if 'norm' not in name.lower() and 'head' not in name.lower():
                    for param in module.parameters():
                        param.requires_grad = False
        
        # Ensure Transformer layers are trainable
        # Look for transformer or encoder layers
        for name, module in self.vision_model.named_modules():
            if 'transformer' in name.lower() or 'encoder' in name.lower():
                for param in module.parameters():
                    param.requires_grad = True
        
        logger.info("Selective freezing applied")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BioViL-T image encoder.
        
        Args:
            images: Input images tensor [batch_size, 3, 224, 224]
        
        Returns:
            Feature maps [batch_size, 197, 768]
            (197 = 196 image patches + 1 CLS token)
        """
        # Move images to device
        images = images.to(DEVICE)
        
        # BioViL-T expects images in [0, 1] range and normalized
        # Our preprocessing already handles this, but verify
        
        # Get image features
        # BioViL-T vision model forward pass
        with torch.set_grad_enabled(self.training):
            if hasattr(self.vision_model, 'forward'):
                # Try standard forward
                outputs = self.vision_model(pixel_values=images)
            else:
                # Try alternative forward methods
                outputs = self.vision_model(images)
        
        # Extract features
        # BioViL-T typically returns last_hidden_state with shape [batch, seq_len, hidden_dim]
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
            features = outputs.hidden_states[-1]  # Use last layer
        elif isinstance(outputs, torch.Tensor):
            features = outputs
        elif isinstance(outputs, tuple):
            features = outputs[0]  # Use first element
        else:
            logger.error(f"Unexpected output type: {type(outputs)}")
            logger.error(f"Output attributes: {dir(outputs)}")
            raise ValueError(f"Cannot extract features from model output: {type(outputs)}")
        
        # Verify feature shape
        # Expected: [batch_size, 197, 768] or [batch_size, seq_len, hidden_dim]
        if len(features.shape) != 3:
            logger.warning(f"Unexpected feature shape: {features.shape}")
            # Try to reshape if needed
            if len(features.shape) == 2:
                # Add sequence dimension
                features = features.unsqueeze(1)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim
    
    def freeze_all(self):
        """Freeze all parameters (for inference only)."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("All parameters frozen")
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")


def create_image_encoder(
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    freeze_resnet_blocks: int = 3
) -> BioViLImageEncoder:
    """
    Factory function to create BioViL-T image encoder.
    
    Args:
        model_name: Hugging Face model name
        freeze_resnet_blocks: Number of ResNet blocks to freeze
    
    Returns:
        BioViLImageEncoder instance
    """
    return BioViLImageEncoder(
        model_name=model_name,
        freeze_resnet_blocks=freeze_resnet_blocks
    )
