"""Training loop with mixed precision and gradient accumulation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional

from ..config import (
    DEVICE, CUDA_AVAILABLE, GRADIENT_ACCUMULATION_STEPS,
    MIXED_PRECISION, CHECKPOINTS_DIR, LOG_INTERVAL
)
from ..utils import calculate_auc_scores
from .loss import CombinedLoss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for MedAI pipeline with mixed precision and gradient accumulation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: CombinedLoss,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 50,
        accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
        use_amp: bool = MIXED_PRECISION,
        checkpoint_dir: Path = CHECKPOINTS_DIR,
        best_metric: str = "macro_auc"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            valid_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler (optional)
            num_epochs: Number of training epochs
            accumulation_steps: Gradient accumulation steps
            use_amp: Use mixed precision training
            checkpoint_dir: Directory to save checkpoints
            best_metric: Metric to track for best model
        """
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp and CUDA_AVAILABLE
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = best_metric
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'valid_auc': [],
            'best_auc': 0.0
        }
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {DEVICE}")
        logger.info(f"  Mixed precision: {self.use_amp}")
        logger.info(f"  Gradient accumulation: {accumulation_steps}")
        logger.info(f"  Best metric: {best_metric}")
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        loss_details = {'bce': 0.0, 'contrastive': 0.0, 'total': 0.0}
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, attention_maps = self.model(images, self.graph_data, self.edge_index)
                    loss, loss_dict = self.criterion(logits, labels)
                    loss = loss / self.accumulation_steps
            else:
                logits, attention_maps = self.model(images, self.graph_data, self.edge_index)
                loss, loss_dict = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Accumulate loss
            total_loss += loss.item() * self.accumulation_steps
            for key in loss_details:
                loss_details[key] += loss_dict.get(key, 0.0)
            num_batches += 1
            
            # Update progress bar
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                pbar.set_postfix({
                    'loss': f"{total_loss/num_batches:.4f}",
                    'bce': f"{loss_details['bce']/num_batches:.4f}"
                })
        
        avg_loss = total_loss / num_batches
        for key in loss_details:
            loss_details[key] /= num_batches
        
        return {
            'loss': avg_loss,
            **loss_details
        }
    
    def validate(self, epoch: int) -> Dict:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Valid]")
            
            for batch in pbar:
                images = batch['image'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits, _ = self.model(images, self.graph_data, self.edge_index)
                        loss, _ = self.criterion(logits, labels)
                else:
                    logits, _ = self.model(images, self.graph_data, self.edge_index)
                    loss, _ = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Collect predictions and labels
                all_logits.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate all predictions
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Calculate AUC
        from ..config import PATHOLOGY_CLASSES
        auc_scores = calculate_auc_scores(all_labels, all_logits, PATHOLOGY_CLASSES)
        
        avg_loss = total_loss / len(self.valid_loader)
        
        return {
            'loss': avg_loss,
            'auc': auc_scores
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model (AUC: {metrics.get('macro_auc', 0):.4f})")
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"epoch_{epoch+1}.pth"
        torch.save(checkpoint, epoch_path)
    
    def train(self, graph_data: torch.Tensor, edge_index: torch.Tensor):
        """
        Main training loop.
        
        Args:
            graph_data: Graph node features [num_nodes, hidden_dim]
            edge_index: Graph edge indices [2, num_edges]
        """
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)
        
        # Move graph data to device
        graph_data = graph_data.to(DEVICE)
        edge_index = edge_index.to(DEVICE)
        
        # Store for use in train_epoch and validate
        self.graph_data = graph_data
        self.edge_index = edge_index
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            valid_metrics = self.validate(epoch)
            self.history['valid_loss'].append(valid_metrics['loss'])
            
            # Get macro AUC
            macro_auc = valid_metrics['auc'].get('macro_auc', 0.0)
            self.history['valid_auc'].append(macro_auc)
            
            # Check if best
            is_best = macro_auc > self.history['best_auc']
            if is_best:
                self.history['best_auc'] = macro_auc
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Valid Loss: {valid_metrics['loss']:.4f}")
            logger.info(f"Valid Macro AUC: {macro_auc:.4f}")
            if is_best:
                logger.info("✓ New best model!")
            
            # Save checkpoint
            self.save_checkpoint(epoch, valid_metrics, is_best)
        
        logger.info("\n" + "=" * 80)
        logger.info("Training Completed")
        logger.info("=" * 80)
        logger.info(f"Best Macro AUC: {self.history['best_auc']:.4f}")
