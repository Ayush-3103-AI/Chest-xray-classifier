"""Loss functions for training: BCE + Contrastive loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    Combined loss function: Binary Cross-Entropy + Contrastive loss.
    
    BCE: Multi-label classification loss
    Contrastive: Graph-image alignment loss (optional)
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        use_contrastive: bool = True
    ):
        """
        Initialize combined loss.
        
        Args:
            bce_weight: Weight for BCE loss (default: 1.0)
            contrastive_weight: Weight for contrastive loss (default: 0.1)
            use_contrastive: Whether to use contrastive loss (default: True)
        """
        super().__init__()
        
        self.bce_weight = bce_weight
        self.contrastive_weight = contrastive_weight
        self.use_contrastive = use_contrastive
        
        # BCE with logits (includes sigmoid)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        logger.info(f"Initialized CombinedLoss:")
        logger.info(f"  BCE weight: {bce_weight}")
        logger.info(f"  Contrastive weight: {contrastive_weight}")
        logger.info(f"  Use contrastive: {use_contrastive}")
    
    def contrastive_loss(
        self,
        image_features: torch.Tensor,
        graph_prototypes: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Contrastive loss for graph-image alignment.
        
        Args:
            image_features: Image features [batch_size, seq_len, hidden_dim]
            graph_prototypes: Graph prototypes [num_pathology_nodes, hidden_dim]
            labels: Ground truth labels [batch_size, num_pathology_nodes]
            temperature: Temperature parameter for softmax
        
        Returns:
            Contrastive loss value
        """
        batch_size = image_features.shape[0]
        num_prototypes = graph_prototypes.shape[0]
        
        # Get CLS token or average pool image features
        # Use first token (CLS token) or average pool
        if image_features.shape[1] > 0:
            # Use CLS token (first token)
            image_embeddings = image_features[:, 0, :]  # [batch_size, hidden_dim]
        else:
            # Average pool
            image_embeddings = image_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        graph_prototypes_norm = F.normalize(graph_prototypes, p=2, dim=1)
        
        # Compute similarity matrix
        # [batch_size, num_prototypes]
        similarity = torch.matmul(image_embeddings, graph_prototypes_norm.t()) / temperature
        
        # Create positive pairs based on labels
        # Positive: label == 1, Negative: label == 0
        positive_mask = (labels == 1.0).float()  # [batch_size, num_prototypes]
        negative_mask = (labels == 0.0).float()  # [batch_size, num_prototypes]
        
        # Contrastive loss: maximize similarity for positive pairs, minimize for negative
        # For each sample, compute loss over all prototypes
        loss_per_sample = []
        
        for i in range(batch_size):
            pos_prototypes = positive_mask[i]  # [num_prototypes]
            neg_prototypes = negative_mask[i]  # [num_prototypes]
            
            if pos_prototypes.sum() > 0:
                # Positive pairs: maximize similarity
                pos_similarities = similarity[i] * pos_prototypes
                pos_loss = -torch.log(torch.sigmoid(pos_similarities).sum() + 1e-8)
            else:
                pos_loss = torch.tensor(0.0, device=similarity.device)
            
            if neg_prototypes.sum() > 0:
                # Negative pairs: minimize similarity
                neg_similarities = similarity[i] * neg_prototypes
                neg_loss = torch.log(torch.sigmoid(neg_similarities).sum() + 1e-8)
            else:
                neg_loss = torch.tensor(0.0, device=similarity.device)
            
            loss_per_sample.append(pos_loss + neg_loss)
        
        contrastive_loss = torch.stack(loss_per_sample).mean()
        
        return contrastive_loss
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        image_features: torch.Tensor = None,
        graph_prototypes: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Model predictions [batch_size, num_pathology_nodes]
            labels: Ground truth labels [batch_size, num_pathology_nodes]
            image_features: Image features (for contrastive loss)
            graph_prototypes: Graph prototypes (for contrastive loss)
        
        Returns:
            Combined loss value
        """
        # BCE loss
        bce = self.bce_loss(logits, labels)
        
        # Contrastive loss (optional)
        if self.use_contrastive and image_features is not None and graph_prototypes is not None:
            contrastive = self.contrastive_loss(image_features, graph_prototypes, labels)
            total_loss = self.bce_weight * bce + self.contrastive_weight * contrastive
        else:
            contrastive = torch.tensor(0.0, device=logits.device)
            total_loss = self.bce_weight * bce
        
        return total_loss, {
            'bce': bce.item(),
            'contrastive': contrastive.item() if isinstance(contrastive, torch.Tensor) else contrastive,
            'total': total_loss.item()
        }


def create_loss(
    bce_weight: float = 1.0,
    contrastive_weight: float = 0.1,
    use_contrastive: bool = True
) -> CombinedLoss:
    """
    Factory function to create loss function.
    
    Args:
        bce_weight: Weight for BCE loss
        contrastive_weight: Weight for contrastive loss
        use_contrastive: Whether to use contrastive loss
    
    Returns:
        CombinedLoss instance
    """
    return CombinedLoss(
        bce_weight=bce_weight,
        contrastive_weight=contrastive_weight,
        use_contrastive=use_contrastive
    )
