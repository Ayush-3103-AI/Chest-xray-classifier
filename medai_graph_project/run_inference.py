"""Entry point for running inference on test/validation images."""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import argparse
import logging

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from config import (
    CHECKPOINTS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR,
    DEVICE, PATHOLOGY_CLASSES, INFERENCE_THRESHOLD
)
from models.medai_pipeline import create_pipeline
from training.dataset import CheXpertDataset
from utils import calculate_metrics, save_metrics
from visualization.exporter import export_inference_to_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / "inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path):
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    # Create model
    model = create_pipeline()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    
    logger.info(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    if 'metrics' in checkpoint:
        logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    return model


def load_knowledge_graph():
    """Load knowledge graph."""
    graph_path = PROCESSED_DATA_DIR / "knowledge_graph.pt"
    
    if not graph_path.exists():
        logger.error(f"Knowledge graph not found: {graph_path}")
        return None, None
    
    graph_data = torch.load(graph_path, map_location=DEVICE)
    node_features = graph_data.x
    edge_index = graph_data.edge_index
    
    logger.info(f"Knowledge graph loaded: {node_features.shape[0]} nodes")
    
    return node_features, edge_index


def run_inference_on_dataset(
    model,
    graph_data: torch.Tensor,
    edge_index: torch.Tensor,
    dataset: CheXpertDataset,
    output_dir: Path
):
    """Run inference on entire dataset."""
    logger.info(f"Running inference on {len(dataset)} samples...")
    
    all_logits = []
    all_predictions = []
    all_labels = []
    all_attention_maps = []
    image_paths = []
    
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(DEVICE)
            labels = sample['labels'].unsqueeze(0)
            
            # Forward pass
            logits, attention_maps = model(image, graph_data, edge_index)
            
            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)
            
            # Binary predictions
            predictions = (probs > INFERENCE_THRESHOLD).float()
            
            # Store results
            all_logits.append(probs.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.numpy())
            all_attention_maps.append(attention_maps.cpu().numpy())
            image_paths.append(sample['path'])
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(dataset)} samples...")
    
    # Concatenate all results
    all_logits = np.concatenate(all_logits, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_attention_maps = np.concatenate(all_attention_maps, axis=0)
    
    # Calculate metrics
    logger.info("\nCalculating metrics...")
    metrics = calculate_metrics(all_labels, all_logits, all_predictions, PATHOLOGY_CLASSES)
    
    # Save metrics
    metrics_path = output_dir / "inference_metrics.json"
    save_metrics(metrics, metrics_path)
    
    # Save predictions CSV
    predictions_df = pd.DataFrame(
        all_logits,
        columns=PATHOLOGY_CLASSES
    )
    predictions_df['image_path'] = image_paths
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    
    # Export visualization for first sample
    logger.info("\nExporting visualization for first sample...")
    snapshot_path = output_dir / "inference_snapshot.json"
    export_inference_to_json(
        attention_scores=all_attention_maps[0],
        predictions=all_predictions[0],
        image_path=image_paths[0],
        output_path=snapshot_path
    )
    
    return metrics, all_attention_maps


def run_inference_on_single_image(
    model,
    graph_data: torch.Tensor,
    edge_index: torch.Tensor,
    image_path: Path,
    output_dir: Path
):
    """Run inference on a single image."""
    from utils import preprocess_image
    
    logger.info(f"Running inference on single image: {image_path}")
    
    # Preprocess image
    image_tensor = preprocess_image(image_path).unsqueeze(0).to(DEVICE)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, attention_maps = model(image_tensor, graph_data, edge_index)
        probs = torch.sigmoid(logits)
        predictions = (probs > INFERENCE_THRESHOLD).float()
    
    # Export visualization
    snapshot_path = output_dir / "inference_snapshot.json"
    export_inference_to_json(
        attention_scores=attention_maps[0].cpu().numpy(),
        predictions=predictions[0].cpu().numpy(),
        image_path=str(image_path),
        output_path=snapshot_path
    )
    
    # Print predictions
    logger.info("\nPredictions:")
    for idx, pathology in enumerate(PATHOLOGY_CLASSES):
        prob = probs[0, idx].item()
        pred = predictions[0, idx].item()
        logger.info(f"  {pathology}: {prob:.4f} ({'Positive' if pred > 0.5 else 'Negative'})")
    
    return probs, attention_maps


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference on CheXpert images')
    parser.add_argument('--checkpoint', type=str, default='best.pth',
                       help='Checkpoint file name (default: best.pth)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for inference')
    parser.add_argument('--dataset', type=str, default='valid',
                       choices=['train', 'valid'],
                       help='Dataset to run inference on (default: valid)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (default: all)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("MEDAI PIPELINE INFERENCE")
    logger.info("=" * 80)
    
    # Load model
    checkpoint_path = CHECKPOINTS_DIR / args.checkpoint
    model = load_model(checkpoint_path)
    if model is None:
        return 1
    
    # Load knowledge graph
    graph_data, edge_index = load_knowledge_graph()
    if graph_data is None:
        return 1
    
    graph_data = graph_data.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    
    # Run inference
    if args.image:
        # Single image inference
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return 1
        
        run_inference_on_single_image(
            model, graph_data, edge_index, image_path, OUTPUTS_DIR
        )
    else:
        # Dataset inference
        csv_path = RAW_DATA_DIR / f"{args.dataset}.csv"
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 1
        
        dataset = CheXpertDataset(csv_path, data_dir=RAW_DATA_DIR)
        
        if args.max_samples:
            # Limit dataset size
            from torch.utils.data import Subset
            indices = list(range(min(args.max_samples, len(dataset))))
            dataset = Subset(dataset, indices)
            logger.info(f"Limited to {len(dataset)} samples")
        
        metrics, attention_maps = run_inference_on_dataset(
            model, graph_data, edge_index, dataset, OUTPUTS_DIR
        )
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("INFERENCE SUMMARY")
        logger.info("=" * 80)
        macro_auc = metrics['auc'].get('macro_auc', None)
        if macro_auc:
            logger.info(f"Macro AUC: {macro_auc:.4f}")
        logger.info(f"Metrics saved to: {OUTPUTS_DIR / 'inference_metrics.json'}")
        logger.info(f"Predictions saved to: {OUTPUTS_DIR / 'predictions.csv'}")
        logger.info(f"Visualization saved to: {OUTPUTS_DIR / 'inference_snapshot.json'}")
    
    logger.info("\n✓ Inference completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
