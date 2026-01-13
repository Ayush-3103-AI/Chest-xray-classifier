"""Entry point for training the MedAI pipeline."""

import sys
from pathlib import Path
import torch
import torch.optim as optim
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR,
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    DEVICE, CUDA_AVAILABLE
)
from models.medai_pipeline import create_pipeline
from training.dataset import create_dataloaders
from training.loss import create_loss
from training.trainer import Trainer
from visualization.graph_builder import MedicalKnowledgeGraph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_knowledge_graph() -> tuple:
    """Load knowledge graph from file."""
    graph_path = PROCESSED_DATA_DIR / "knowledge_graph.pt"
    
    if not graph_path.exists():
        logger.error(f"Knowledge graph not found at {graph_path}")
        logger.error("Please run run_build_graph.py first to create the knowledge graph.")
        return None, None
    
    logger.info(f"Loading knowledge graph from {graph_path}")
    graph_data = torch.load(graph_path)
    
    # Extract node features and edge index
    node_features = graph_data.x  # [num_nodes, hidden_dim]
    edge_index = graph_data.edge_index  # [2, num_edges]
    
    logger.info(f"Graph loaded: {node_features.shape[0]} nodes, {edge_index.shape[1]} edges")
    
    return node_features, edge_index


def main():
    """Run training."""
    logger.info("=" * 80)
    logger.info("MEDAI PIPELINE TRAINING")
    logger.info("=" * 80)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"CUDA Available: {CUDA_AVAILABLE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    
    # Check dataset
    train_csv = RAW_DATA_DIR / "train.csv"
    valid_csv = RAW_DATA_DIR / "valid.csv"
    
    if not train_csv.exists() or not valid_csv.exists():
        logger.error("Dataset CSV files not found. Please ensure CheXpert dataset is set up.")
        return 1
    
    # Load knowledge graph
    graph_data, edge_index = load_knowledge_graph()
    if graph_data is None:
        return 1
    
    try:
        # Create DataLoaders
        logger.info("\nCreating DataLoaders...")
        train_loader, valid_loader = create_dataloaders(
            train_csv=train_csv,
            valid_csv=valid_csv,
            data_dir=RAW_DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        
        # Create model
        logger.info("\nCreating model...")
        model = create_pipeline()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # Create loss function
        criterion = create_loss(
            bce_weight=1.0,
            contrastive_weight=0.1,
            use_contrastive=True
        )
        
        # Create scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize AUC
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            num_epochs=NUM_EPOCHS
        )
        
        # Train
        logger.info("\nStarting training...")
        trainer.train(graph_data, edge_index)
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Best Macro AUC: {trainer.history['best_auc']:.4f}")
        logger.info(f"Checkpoints saved to: {CHECKPOINTS_DIR}")
        logger.info(f"Training log saved to: {OUTPUTS_DIR / 'training.log'}")
        logger.info("\n✓ Training completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
