"""Multi-head cross-attention mechanism for image-graph fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from ..config import HIDDEN_DIM, NUM_ATTENTION_HEADS, DROPOUT

logger = logging.getLogger(__name__)


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism.
    
    Query (Q): Graph prototypes [num_pathology_nodes, hidden_dim]
    Key/Value (K/V): Image features [batch_size, seq_len, hidden_dim]
    
    Output: 14 class logits [batch_size, num_pathology_nodes]
    """
    
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_ATTENTION_HEADS,
        dropout: float = DROPOUT
    ):
        """
        Initialize cross-attention module.
        
        Args:
            hidden_dim: Feature dimension (default: 768)
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout rate (default: 0.2)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, 1)  # Binary classification per class
        
        logger.info(f"Initialized CrossAttention:")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Num heads: {num_heads}")
        logger.info(f"  Head dim: {self.head_dim}")
    
    def forward(
        self,
        graph_prototypes: torch.Tensor,
        image_features: torch.Tensor
    ) -> tuple:
        """
        Forward pass through cross-attention.
        
        Args:
            graph_prototypes: Graph prototypes [num_pathology_nodes, hidden_dim]
            image_features: Image features [batch_size, seq_len, hidden_dim]
        
        Returns:
            Tuple of (logits [batch_size, num_pathology_nodes], attention_maps)
        """
        batch_size, seq_len, _ = image_features.shape
        num_prototypes = graph_prototypes.shape[0]
        
        # Expand graph prototypes for batch
        # [num_pathology_nodes, hidden_dim] -> [batch_size, num_pathology_nodes, hidden_dim]
        Q = graph_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Prepare image features
        K = image_features  # [batch_size, seq_len, hidden_dim]
        V = image_features  # [batch_size, seq_len, hidden_dim]
        
        # Project to Q, K, V
        Q = self.q_proj(Q)  # [batch_size, num_pathology_nodes, hidden_dim]
        K = self.k_proj(K)  # [batch_size, seq_len, hidden_dim]
        V = self.v_proj(V)  # [batch_size, seq_len, hidden_dim]
        
        # Reshape for multi-head attention
        # [batch_size, num_nodes/seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, num_prototypes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # Q: [batch_size, num_heads, num_prototypes, head_dim]
        # K: [batch_size, num_heads, seq_len, head_dim]
        # Attention scores: [batch_size, num_heads, num_prototypes, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Apply attention to values
        # [batch_size, num_heads, num_prototypes, head_dim]
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        # [batch_size, num_prototypes, hidden_dim]
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, num_prototypes, self.hidden_dim
        )
        
        # Output projection
        output = self.out_proj(attended)
        output = self.dropout_layer(output)
        
        # Classification: binary classification per pathology
        # [batch_size, num_prototypes, 1] -> [batch_size, num_prototypes]
        logits = self.classifier(output).squeeze(-1)
        
        # Attention maps for visualization
        # Average over heads: [batch_size, num_prototypes, seq_len]
        attention_maps = attention_weights.mean(dim=1)
        
        return logits, attention_maps


def create_cross_attention(
    hidden_dim: int = HIDDEN_DIM,
    num_heads: int = NUM_ATTENTION_HEADS
) -> CrossAttention:
    """
    Factory function to create cross-attention module.
    
    Args:
        hidden_dim: Feature dimension
        num_heads: Number of attention heads
    
    Returns:
        CrossAttention instance
    """
    return CrossAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads
    )
