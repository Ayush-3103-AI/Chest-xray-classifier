"""Export inference results to JSON for 3D graph visualization."""

import json
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List

from ..config import PATHOLOGY_CLASSES, ATTENTION_THRESHOLD, OUTPUTS_DIR

logger = logging.getLogger(__name__)


def export_inference_to_json(
    attention_scores: np.ndarray,
    predictions: np.ndarray,
    image_path: str,
    output_path: Path = None,
    threshold: float = ATTENTION_THRESHOLD
) -> Dict:
    """
    Export inference results to JSON format for 3D force-directed graph.
    
    Args:
        attention_scores: Attention scores [num_pathology_nodes] or [batch, num_pathology_nodes]
        predictions: Binary predictions [num_pathology_nodes] or [batch, num_pathology_nodes]
        image_path: Path to input image
        output_path: Path to save JSON file
        threshold: Minimum attention score to include edge (default: 0.2)
    
    Returns:
        Dictionary with nodes and links for 3D visualization
    """
    # Handle batch dimension
    if len(attention_scores.shape) > 1:
        attention_scores = attention_scores[0]  # Take first sample
    if len(predictions.shape) > 1:
        predictions = predictions[0]  # Take first sample
    
    # Normalize attention scores to [0, 1]
    if attention_scores.max() > 1.0 or attention_scores.min() < 0.0:
        attention_scores = (attention_scores - attention_scores.min()) / (
            attention_scores.max() - attention_scores.min() + 1e-8
        )
    
    # Create nodes
    nodes = []
    
    # Add pathology nodes
    for idx, pathology in enumerate(PATHOLOGY_CLASSES):
        # Node value based on prediction confidence
        node_val = float(predictions[idx]) * 20 + 10  # Scale for visualization
        
        nodes.append({
            "id": pathology,
            "group": 1,  # Pathology group
            "val": node_val
        })
    
    # Add patient node
    patient_node = {
        "id": "Patient_Xray",
        "group": 2,  # Patient group
        "val": 15
    }
    nodes.append(patient_node)
    
    # Create links (edges) based on attention scores
    links = []
    
    for idx, pathology in enumerate(PATHOLOGY_CLASSES):
        attention_score = float(attention_scores[idx])
        
        # Only create link if attention score exceeds threshold
        if attention_score >= threshold:
            links.append({
                "source": "Patient_Xray",
                "target": pathology,
                "value": attention_score
            })
    
    # Create JSON structure
    graph_data = {
        "nodes": nodes,
        "links": links
    }
    
    # Save to file if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        logger.info(f"Inference snapshot saved to {output_path}")
        logger.info(f"  Nodes: {len(nodes)}")
        logger.info(f"  Links: {len(links)} (threshold: {threshold})")
    
    return graph_data


def export_batch_inference(
    attention_scores_batch: np.ndarray,
    predictions_batch: np.ndarray,
    image_paths: List[str],
    output_dir: Path = OUTPUTS_DIR,
    threshold: float = ATTENTION_THRESHOLD
) -> List[Dict]:
    """
    Export batch inference results.
    
    Args:
        attention_scores_batch: Attention scores [batch, num_pathology_nodes]
        predictions_batch: Predictions [batch, num_pathology_nodes]
        image_paths: List of image paths
        output_dir: Output directory
        threshold: Attention threshold
    
    Returns:
        List of graph data dictionaries
    """
    batch_graphs = []
    
    for idx in range(len(image_paths)):
        graph_data = export_inference_to_json(
            attention_scores=attention_scores_batch[idx],
            predictions=predictions_batch[idx],
            image_path=image_paths[idx],
            threshold=threshold
        )
        batch_graphs.append(graph_data)
    
    # Save combined snapshot (for single image inference)
    if len(batch_graphs) == 1:
        snapshot_path = output_dir / "inference_snapshot.json"
        with open(snapshot_path, 'w') as f:
            json.dump(batch_graphs[0], f, indent=2)
        logger.info(f"Single inference snapshot saved to {snapshot_path}")
    
    return batch_graphs
