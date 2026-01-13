"""Knowledge graph builder for medical hierarchy using Fleischner Society structure."""

import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List
from torch_geometric.data import Data
import json

from ..config import (
    PATHOLOGY_CLASSES, NUM_PATHOLOGY_NODES, NUM_ANATOMY_NODES,
    NUM_TOTAL_NODES, HIDDEN_DIM, PROCESSED_DATA_DIR
)

logger = logging.getLogger(__name__)


class MedicalKnowledgeGraph:
    """
    Medical knowledge graph builder.
    
    Creates a graph with:
    - 14 pathology nodes (indices 0-13)
    - 5 anatomical region nodes (indices 14-18)
    - Edges based on medical hierarchy (Fleischner Society)
    - Node embeddings initialized with BioViL-T text encoder
    """
    
    def __init__(self):
        """Initialize the knowledge graph builder."""
        self.pathology_nodes = PATHOLOGY_CLASSES.copy()
        self.anatomy_nodes = ["Lung", "Heart", "Pleura", "Mediastinum", "Bones"]
        self.all_nodes = self.pathology_nodes + self.anatomy_nodes
        
        logger.info(f"Initialized knowledge graph with {len(self.pathology_nodes)} pathologies "
                   f"and {len(self.anatomy_nodes)} anatomical regions")
    
    def get_node_index(self, node_name: str) -> int:
        """Get index of a node by name."""
        try:
            return self.all_nodes.index(node_name)
        except ValueError:
            raise ValueError(f"Node '{node_name}' not found in graph")
    
    def build_adjacency_matrix(self) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Build adjacency matrix based on medical hierarchy.
        
        Hierarchy:
        - Lung Opacity → parent of: Edema, Consolidation, Pneumonia, Atelectasis
        - Enlarged Cardiomediastinum → parent of: Cardiomegaly
        - Pleural Effusion → connected to: Pleura
        
        Returns:
            Tuple of (edge_index tensor [2, num_edges], list of edge tuples)
        """
        logger.info("Building adjacency matrix...")
        
        edges = []
        edge_descriptions = []
        
        # Define medical hierarchy edges
        # Lung Opacity is parent of several pathologies
        lung_opacity_idx = self.get_node_index("Lung Opacity")
        
        # Edema → Lung Opacity (child to parent)
        edema_idx = self.get_node_index("Edema")
        edges.append((edema_idx, lung_opacity_idx))
        edge_descriptions.append(("Edema", "Lung Opacity"))
        
        # Consolidation → Lung Opacity
        consolidation_idx = self.get_node_index("Consolidation")
        edges.append((consolidation_idx, lung_opacity_idx))
        edge_descriptions.append(("Consolidation", "Lung Opacity"))
        
        # Pneumonia → Lung Opacity
        pneumonia_idx = self.get_node_index("Pneumonia")
        edges.append((pneumonia_idx, lung_opacity_idx))
        edge_descriptions.append(("Pneumonia", "Lung Opacity"))
        
        # Atelectasis → Lung Opacity
        atelectasis_idx = self.get_node_index("Atelectasis")
        edges.append((atelectasis_idx, lung_opacity_idx))
        edge_descriptions.append(("Atelectasis", "Lung Opacity"))
        
        # Enlarged Cardiomediastinum → parent of Cardiomegaly
        enlarged_cardiomediastinum_idx = self.get_node_index("Enlarged Cardiomediastinum")
        cardiomegaly_idx = self.get_node_index("Cardiomegaly")
        edges.append((cardiomegaly_idx, enlarged_cardiomediastinum_idx))
        edge_descriptions.append(("Cardiomegaly", "Enlarged Cardiomediastinum"))
        
        # Pleural Effusion → connected to Pleura
        pleural_effusion_idx = self.get_node_index("Pleural Effusion")
        pleura_idx = self.get_node_index("Pleura")
        edges.append((pleural_effusion_idx, pleura_idx))
        edge_descriptions.append(("Pleural Effusion", "Pleura"))
        
        # Additional anatomical connections
        # Lung pathologies connected to Lung anatomy
        lung_idx = self.get_node_index("Lung")
        edges.append((lung_opacity_idx, lung_idx))
        edge_descriptions.append(("Lung Opacity", "Lung"))
        edges.append((edema_idx, lung_idx))
        edge_descriptions.append(("Edema", "Lung"))
        edges.append((consolidation_idx, lung_idx))
        edge_descriptions.append(("Consolidation", "Lung"))
        edges.append((pneumonia_idx, lung_idx))
        edge_descriptions.append(("Pneumonia", "Lung"))
        edges.append((atelectasis_idx, lung_idx))
        edge_descriptions.append(("Atelectasis", "Lung"))
        
        # Heart pathologies connected to Heart anatomy
        heart_idx = self.get_node_index("Heart")
        edges.append((cardiomegaly_idx, heart_idx))
        edge_descriptions.append(("Cardiomegaly", "Heart"))
        edges.append((enlarged_cardiomediastinum_idx, heart_idx))
        edge_descriptions.append(("Enlarged Cardiomediastinum", "Heart"))
        
        # Make edges bidirectional for GAT message passing
        # (GATs benefit from bidirectional edges)
        bidirectional_edges = []
        for src, dst in edges:
            bidirectional_edges.append((src, dst))
            bidirectional_edges.append((dst, src))  # Reverse edge
        
        # Remove duplicates
        bidirectional_edges = list(set(bidirectional_edges))
        
        # Convert to edge_index format [2, num_edges]
        if bidirectional_edges:
            edge_index = torch.tensor(bidirectional_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        logger.info(f"Created {len(bidirectional_edges)} bidirectional edges")
        logger.info(f"Edge index shape: {edge_index.shape}")
        
        return edge_index, bidirectional_edges
    
    def initialize_node_embeddings(self, text_encoder) -> torch.Tensor:
        """
        Initialize node embeddings using BioViL-T text encoder.
        
        Args:
            text_encoder: BioViL-T text encoder model
        
        Returns:
            Node embeddings tensor [num_nodes, hidden_dim]
        """
        logger.info("Initializing node embeddings with BioViL-T text encoder...")
        
        embeddings = []
        
        # Generate embeddings for each node name
        for node_name in self.all_nodes:
            try:
                # Use text encoder to get embedding
                # BioViL-T text encoder expects text input
                with torch.no_grad():
                    # Get text embedding
                    # The exact method depends on BioViL-T API
                    if hasattr(text_encoder, 'encode_text'):
                        # If there's an encode_text method
                        embedding = text_encoder.encode_text(node_name)
                    elif hasattr(text_encoder, 'forward'):
                        # Try forward with text input
                        # BioViL-T might use tokenizer first
                        if hasattr(text_encoder, 'tokenizer'):
                            tokens = text_encoder.tokenizer(
                                node_name, 
                                return_tensors="pt",
                                padding=True,
                                truncation=True
                            )
                            embedding = text_encoder(**tokens)
                        else:
                            # Direct forward
                            embedding = text_encoder(node_name)
                    else:
                        # Fallback: try calling directly
                        embedding = text_encoder(node_name)
                    
                    # Extract embedding vector
                    if isinstance(embedding, dict):
                        # Get last_hidden_state or pooler_output
                        if 'last_hidden_state' in embedding:
                            # Use CLS token (first token)
                            emb = embedding['last_hidden_state'][0, 0, :]
                        elif 'pooler_output' in embedding:
                            emb = embedding['pooler_output'][0]
                        else:
                            # Use first available tensor
                            emb = list(embedding.values())[0]
                            if len(emb.shape) > 1:
                                emb = emb[0]  # Take first item
                    elif isinstance(embedding, torch.Tensor):
                        if len(embedding.shape) > 1:
                            emb = embedding[0]  # Take first item if batch dimension exists
                        else:
                            emb = embedding
                    else:
                        raise ValueError(f"Unexpected embedding type: {type(embedding)}")
                    
                    # Ensure correct dimension
                    if emb.shape[0] != HIDDEN_DIM:
                        logger.warning(f"Embedding dimension mismatch for {node_name}: "
                                     f"{emb.shape[0]} != {HIDDEN_DIM}")
                        # Project to correct dimension if needed
                        if emb.shape[0] < HIDDEN_DIM:
                            # Pad
                            padding = torch.zeros(HIDDEN_DIM - emb.shape[0])
                            emb = torch.cat([emb, padding])
                        else:
                            # Truncate
                            emb = emb[:HIDDEN_DIM]
                    
                    embeddings.append(emb.cpu())
                    
            except Exception as e:
                logger.error(f"Error generating embedding for {node_name}: {e}")
                # Fallback: use random embedding (should not happen)
                logger.warning(f"Using random embedding for {node_name} as fallback")
                embeddings.append(torch.randn(HIDDEN_DIM))
        
        # Stack embeddings
        node_embeddings = torch.stack(embeddings)
        
        logger.info(f"Node embeddings shape: {node_embeddings.shape}")
        logger.info(f"Expected shape: [{NUM_TOTAL_NODES}, {HIDDEN_DIM}]")
        
        return node_embeddings
    
    def build_graph(self, text_encoder) -> Data:
        """
        Build complete knowledge graph.
        
        Args:
            text_encoder: BioViL-T text encoder
        
        Returns:
            PyTorch Geometric Data object
        """
        logger.info("=" * 60)
        logger.info("Building Medical Knowledge Graph")
        logger.info("=" * 60)
        
        # Build adjacency matrix
        edge_index, edges = self.build_adjacency_matrix()
        
        # Initialize node embeddings
        node_embeddings = self.initialize_node_embeddings(text_encoder)
        
        # Create PyTorch Geometric Data object
        graph_data = Data(
            x=node_embeddings,  # Node features [num_nodes, hidden_dim]
            edge_index=edge_index,  # Edge indices [2, num_edges]
            num_nodes=NUM_TOTAL_NODES
        )
        
        # Store metadata
        graph_data.node_names = self.all_nodes
        graph_data.pathology_nodes = self.pathology_nodes
        graph_data.anatomy_nodes = self.anatomy_nodes
        
        logger.info("✓ Knowledge graph built successfully")
        logger.info(f"  Nodes: {NUM_TOTAL_NODES}")
        logger.info(f"  Edges: {edge_index.shape[1]}")
        logger.info(f"  Node features: {node_embeddings.shape}")
        
        return graph_data
    
    def save_graph(self, graph_data: Data, output_path: Path):
        """Save graph to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as PyTorch file
        torch.save(graph_data, output_path)
        logger.info(f"Graph saved to {output_path}")
        
        # Also save metadata as JSON
        metadata_path = output_path.with_suffix('.json')
        metadata = {
            "num_nodes": NUM_TOTAL_NODES,
            "num_edges": graph_data.edge_index.shape[1],
            "node_names": graph_data.node_names,
            "pathology_nodes": graph_data.pathology_nodes,
            "anatomy_nodes": graph_data.anatomy_nodes,
            "hidden_dim": HIDDEN_DIM
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Graph metadata saved to {metadata_path}")
    
    def load_graph(self, graph_path: Path) -> Data:
        """Load graph from file."""
        graph_data = torch.load(graph_path)
        logger.info(f"Graph loaded from {graph_path}")
        return graph_data


def get_biovil_text_encoder(model_name: str = "microsoft/BiomedVLP-BioViL-T"):
    """
    Get BioViL-T text encoder.
    
    Args:
        model_name: Hugging Face model name
    
    Returns:
        Text encoder model
    """
    from transformers import AutoModel, AutoProcessor
    
    logger.info(f"Loading BioViL-T text encoder from {model_name}")
    
    try:
        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Access text encoder
        # BioViL-T structure may vary
        if hasattr(model, 'text_model'):
            text_encoder = model.text_model
        elif hasattr(model, 'text_encoder'):
            text_encoder = model.text_encoder
        else:
            # Use processor for text encoding
            logger.warning("Could not find text_model/text_encoder. Using processor.")
            text_encoder = processor
        
        return text_encoder, processor
        
    except Exception as e:
        logger.error(f"Error loading text encoder: {e}")
        raise


def create_knowledge_graph(output_path: Path = None) -> Data:
    """
    Factory function to create knowledge graph.
    
    Args:
        output_path: Optional path to save graph
    
    Returns:
        Knowledge graph Data object
    """
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "knowledge_graph.pt"
    
    # Get text encoder
    text_encoder, processor = get_biovil_text_encoder()
    
    # Create graph builder
    graph_builder = MedicalKnowledgeGraph()
    
    # Build graph
    graph_data = graph_builder.build_graph(text_encoder)
    
    # Save graph
    graph_builder.save_graph(graph_data, output_path)
    
    return graph_data
