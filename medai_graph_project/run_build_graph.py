"""Entry point for building knowledge graph."""

import sys
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import networkx as nx
import torch

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from config import PROCESSED_DATA_DIR, OUTPUTS_DIR, NUM_TOTAL_NODES, HIDDEN_DIM
from visualization.graph_builder import create_knowledge_graph, MedicalKnowledgeGraph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "outputs" / "graph_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def visualize_graph_structure(graph_data, output_path: Path):
    """Create 2D visualization of graph structure."""
    logger.info("Creating graph visualization...")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for idx, node_name in enumerate(graph_data.node_names):
        G.add_node(idx, name=node_name)
    
    # Add edges
    edge_index = graph_data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        G.add_edge(int(src), int(dst))
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    # Color pathology nodes differently from anatomy nodes
    pathology_indices = list(range(len(graph_data.pathology_nodes)))
    anatomy_indices = list(range(len(graph_data.pathology_nodes), NUM_TOTAL_NODES))
    
    nx.draw_networkx_nodes(G, pos, nodelist=pathology_indices, 
                          node_color='lightblue', node_size=1000, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=anatomy_indices, 
                          node_color='lightcoral', node_size=1500, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
    
    # Draw labels
    labels = {idx: name for idx, name in enumerate(graph_data.node_names)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    plt.title("Medical Knowledge Graph Structure\n(Blue: Pathologies, Red: Anatomy)", 
             fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Graph visualization saved to {output_path}")


def verify_graph_structure(graph_data):
    """Verify graph structure is correct."""
    logger.info("=" * 80)
    logger.info("Verifying Graph Structure")
    logger.info("=" * 80)
    
    # Check node features
    logger.info(f"Node features shape: {graph_data.x.shape}")
    expected_shape = (NUM_TOTAL_NODES, HIDDEN_DIM)
    if graph_data.x.shape == expected_shape:
        logger.info("✓ Node features shape is correct")
    else:
        logger.error(f"✗ Node features shape mismatch: {graph_data.x.shape} != {expected_shape}")
        return False
    
    # Check edge index
    logger.info(f"Edge index shape: {graph_data.edge_index.shape}")
    if graph_data.edge_index.shape[0] == 2:
        logger.info("✓ Edge index format is correct")
    else:
        logger.error("✗ Edge index should have 2 rows")
        return False
    
    # Check node names
    logger.info(f"Number of node names: {len(graph_data.node_names)}")
    if len(graph_data.node_names) == NUM_TOTAL_NODES:
        logger.info("✓ Node names count is correct")
    else:
        logger.error(f"✗ Node names count mismatch: {len(graph_data.node_names)} != {NUM_TOTAL_NODES}")
        return False
    
    # Check that embeddings are not all zeros or random-looking
    node_emb_mean = graph_data.x.mean().item()
    node_emb_std = graph_data.x.std().item()
    logger.info(f"Node embeddings - Mean: {node_emb_mean:.4f}, Std: {node_emb_std:.4f}")
    
    # Check if embeddings look reasonable (not all zeros, not too random)
    if abs(node_emb_mean) < 0.1 and node_emb_std < 0.1:
        logger.warning("⚠ Node embeddings seem too uniform (might be zeros)")
    elif node_emb_std > 10:
        logger.warning("⚠ Node embeddings have very high variance (might be random)")
    else:
        logger.info("✓ Node embeddings look reasonable")
    
    # Verify specific edges exist
    logger.info("\nVerifying key edges...")
    
    # Check Lung Opacity → Edema connection
    edge_index = graph_data.edge_index.numpy()
    lung_opacity_idx = graph_data.node_names.index("Lung Opacity")
    edema_idx = graph_data.node_names.index("Edema")
    
    has_edge = False
    for i in range(edge_index.shape[1]):
        if (edge_index[0, i] == lung_opacity_idx and edge_index[1, i] == edema_idx) or \
           (edge_index[0, i] == edema_idx and edge_index[1, i] == lung_opacity_idx):
            has_edge = True
            break
    
    if has_edge:
        logger.info("✓ Lung Opacity ↔ Edema edge exists")
    else:
        logger.warning("⚠ Lung Opacity ↔ Edema edge not found")
    
    return True


def main():
    """Run knowledge graph construction."""
    logger.info("=" * 80)
    logger.info("KNOWLEDGE GRAPH CONSTRUCTION")
    logger.info("=" * 80)
    
    # Output path
    graph_path = PROCESSED_DATA_DIR / "knowledge_graph.pt"
    viz_path = OUTPUTS_DIR / "graph_structure.png"
    
    try:
        # Create knowledge graph
        logger.info("\nBuilding knowledge graph...")
        graph_data = create_knowledge_graph(output_path=graph_path)
        
        # Verify structure
        if not verify_graph_structure(graph_data):
            logger.error("Graph structure verification failed")
            return 1
        
        # Create visualization
        logger.info("\nCreating visualization...")
        visualize_graph_structure(graph_data, viz_path)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("KNOWLEDGE GRAPH SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✓ Graph saved to: {graph_path}")
        logger.info(f"✓ Visualization saved to: {viz_path}")
        logger.info(f"✓ Number of nodes: {NUM_TOTAL_NODES}")
        logger.info(f"✓ Number of edges: {graph_data.edge_index.shape[1]}")
        logger.info(f"✓ Node feature dimension: {HIDDEN_DIM}")
        logger.info(f"✓ Pathology nodes: {len(graph_data.pathology_nodes)}")
        logger.info(f"✓ Anatomy nodes: {len(graph_data.anatomy_nodes)}")
        logger.info("\n✓ Knowledge graph construction completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during graph construction: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
