"""Main pipeline combining image encoder, graph encoder, and cross-attention."""

import torch
import torch.nn as nn
import logging
from typing import Tuple

from .image_encoder import BioViLImageEncoder
from .graph_encoder import MedicalGraphEncoder
from .cross_attention import CrossAttention
from ..config import DEVICE

logger = logging.getLogger(__name__)


class MedAIPipeline(nn.Module):
    """
    Main neuro-symbolic pipeline for medical image classification.
    
    Combines:
    - Image encoder (BioViL-T)
    - Graph encoder (GATv2)
    - Cross-attention (image-graph fusion)
    """
    
    def __init__(
        self,
        image_encoder: BioViLImageEncoder = None,
        graph_encoder: MedicalGraphEncoder = None,
        cross_attention: CrossAttention = None
    ):
        """
        Initialize MedAI pipeline.
        
        Args:
            image_encoder: BioViL-T image encoder (created if None)
            graph_encoder: GATv2 graph encoder (created if None)
            cross_attention: Cross-attention module (created if None)
        """
        super().__init__()
        
        # Image encoder
        if image_encoder is None:
            from .image_encoder import create_image_encoder
            self.image_encoder = create_image_encoder()
        else:
            self.image_encoder = image_encoder
        
        # Graph encoder
        if graph_encoder is None:
            from .graph_encoder import create_graph_encoder
            self.graph_encoder = create_graph_encoder()
        else:
            self.graph_encoder = graph_encoder
        
        # Cross-attention
        if cross_attention is None:
            from .cross_attention import create_cross_attention
            self.cross_attention = create_cross_attention()
        else:
            self.cross_attention = cross_attention
        
        logger.info("MedAI Pipeline initialized")
    
    def forward(
        self,
        images: torch.Tensor,
        graph_data: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the pipeline.
        
        Args:
            images: Input images [batch_size, 3, 224, 224]
            graph_data: Graph node features [num_nodes, hidden_dim]
            edge_index: Graph edge indices [2, num_edges]
        
        Returns:
            Tuple of (logits [batch_size, num_pathology_nodes], attention_maps)
        """
        # Image branch: extract features
        # Output: [batch_size, 197, 768]
        image_features = self.image_encoder(images)
        
        # Graph branch: refine disease prototypes
        # Output: [num_pathology_nodes, 768]
        graph_prototypes = self.graph_encoder(graph_data, edge_index)
        
        # Cross-attention: fuse image and graph
        # Output: [batch_size, num_pathology_nodes]
        logits, attention_maps = self.cross_attention(graph_prototypes, image_features)
        
        return logits, attention_maps
    
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features only."""
        return self.image_encoder(images)
    
    def get_graph_prototypes(self, graph_data: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Extract graph prototypes only."""
        return self.graph_encoder(graph_data, edge_index)


def create_pipeline(
    image_encoder: BioViLImageEncoder = None,
    graph_encoder: MedicalGraphEncoder = None,
    cross_attention: CrossAttention = None
) -> MedAIPipeline:
    """
    Factory function to create MedAI pipeline.
    
    Args:
        image_encoder: Optional image encoder
        graph_encoder: Optional graph encoder
        cross_attention: Optional cross-attention module
    
    Returns:
        MedAIPipeline instance
    """
    return MedAIPipeline(
        image_encoder=image_encoder,
        graph_encoder=graph_encoder,
        cross_attention=cross_attention
    )
