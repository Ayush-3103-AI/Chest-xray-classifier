"""GATv2 graph encoder for medical knowledge graph."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import logging

from ..config import HIDDEN_DIM, NUM_PATHOLOGY_NODES, NUM_TOTAL_NODES, DROPOUT

logger = logging.getLogger(__name__)


class MedicalGraphEncoder(nn.Module):
    """
    Graph Attention Network (GATv2) encoder for medical knowledge graph.
    
    Architecture:
    - First GATv2 layer: 4 heads, output channels
    - Second GATv2 layer: 1 head, final output
    - Input: [num_nodes, hidden_dim] node embeddings
    - Output: [num_pathology_nodes, hidden_dim] refined disease prototypes
    """
    
    def __init__(
        self,
        in_channels: int = HIDDEN_DIM,
        out_channels: int = HIDDEN_DIM,
        num_heads: int = 4,
        dropout: float = DROPOUT
    ):
        """
        Initialize GATv2 graph encoder.
        
        Args:
            in_channels: Input feature dimension (default: 768)
            out_channels: Output feature dimension (default: 768)
            num_heads: Number of attention heads for first layer (default: 4)
            dropout: Dropout rate (default: 0.2)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # First GATv2 layer: 4 heads
        # Output will be [num_nodes, out_channels * num_heads]
        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True  # Concatenate head outputs
        )
        
        # Second GATv2 layer: 1 head, final output
        # Input is [num_nodes, out_channels * num_heads]
        self.gat2 = GATv2Conv(
            in_channels=out_channels * num_heads,
            out_channels=out_channels,
            heads=1,
            dropout=dropout,
            concat=False  # Don't concatenate (single head)
        )
        
        logger.info(f"Initialized GATv2 encoder:")
        logger.info(f"  Input dim: {in_channels}")
        logger.info(f"  Output dim: {out_channels}")
        logger.info(f"  First layer: {num_heads} heads")
        logger.info(f"  Second layer: 1 head")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GATv2 encoder.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            Refined disease prototypes [num_pathology_nodes, out_channels]
            (Only pathology nodes, not anatomy nodes)
        """
        # First GATv2 layer with ReLU activation
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)
        
        # Second GATv2 layer
        x = self.gat2(x, edge_index)
        
        # Extract only pathology nodes (indices 0-13)
        # Anatomy nodes are indices 14-18
        pathology_prototypes = x[:NUM_PATHOLOGY_NODES, :]
        
        return pathology_prototypes


def create_graph_encoder(
    in_channels: int = HIDDEN_DIM,
    out_channels: int = HIDDEN_DIM,
    num_heads: int = 4
) -> MedicalGraphEncoder:
    """
    Factory function to create graph encoder.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        num_heads: Number of attention heads
    
    Returns:
        MedicalGraphEncoder instance
    """
    return MedicalGraphEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        num_heads=num_heads
    )
