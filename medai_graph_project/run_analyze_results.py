"""Results analysis script: Generate comprehensive results report."""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from config import OUTPUTS_DIR, PATHOLOGY_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / "results_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_metrics() -> dict:
    """Load inference metrics."""
    metrics_path = OUTPUTS_DIR / "inference_metrics.json"
    
    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        logger.error("Please run run_inference.py first to generate metrics.")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def generate_results_report(metrics: dict) -> str:
    """Generate comprehensive results report in Markdown format."""
    report = []
    
    report.append("# Medical AI Pipeline - Results Report\n")
    report.append("=" * 80 + "\n\n")
    
    # Summary
    report.append("## Summary\n\n")
    macro_auc = metrics.get('auc', {}).get('macro_auc', None)
    if macro_auc:
        report.append(f"**Macro AUC Score: {macro_auc:.4f}**\n\n")
        if macro_auc >= 0.85:
            report.append("✓ **Target achieved!** (Macro AUC > 0.85)\n\n")
        else:
            report.append("⚠ Target not yet achieved (Macro AUC < 0.85)\n\n")
    
    # Per-class metrics
    report.append("## Per-Class Performance Metrics\n\n")
    report.append("| Pathology | AUC | Precision | Recall | F1 Score |\n")
    report.append("|-----------|-----|-----------|--------|----------|\n")
    
    for pathology in PATHOLOGY_CLASSES:
        if pathology in metrics:
            path_metrics = metrics[pathology]
            auc = metrics.get('auc', {}).get(pathology, 'N/A')
            if auc != 'N/A':
                auc = f"{auc:.4f}"
            
            prec = path_metrics.get('precision', 'N/A')
            if prec != 'N/A':
                prec = f"{prec:.4f}"
            
            rec = path_metrics.get('recall', 'N/A')
            if rec != 'N/A':
                rec = f"{rec:.4f}"
            
            f1 = path_metrics.get('f1', 'N/A')
            if f1 != 'N/A':
                f1 = f"{f1:.4f}"
            
            report.append(f"| {pathology} | {auc} | {prec} | {rec} | {f1} |\n")
    
    # Macro averages
    report.append("\n### Macro Averages\n\n")
    macro_prec = metrics.get('macro_precision', None)
    macro_rec = metrics.get('macro_recall', None)
    macro_f1 = metrics.get('macro_f1', None)
    
    if macro_prec:
        report.append(f"- **Macro Precision:** {macro_prec:.4f}\n")
    if macro_rec:
        report.append(f"- **Macro Recall:** {macro_rec:.4f}\n")
    if macro_f1:
        report.append(f"- **Macro F1 Score:** {macro_f1:.4f}\n")
    
    # Visualization section
    report.append("\n## Visualizations\n\n")
    report.append("The following visualizations have been generated:\n\n")
    report.append("- `per_class_auc.png`: Per-class AUC scores\n")
    report.append("- `confusion_matrices.png`: Confusion matrices for each class\n")
    report.append("- `attention_maps/`: Attention map visualizations\n")
    report.append("- `3d_graph.html`: Interactive 3D graph visualization\n\n")
    
    # Model information
    report.append("## Model Information\n\n")
    report.append("- **Architecture:** Neuro-Symbolic Pipeline (BioViL-T + GATv2 + Cross-Attention)\n")
    report.append("- **Image Encoder:** BioViL-T (microsoft/BiomedVLP-BioViL-T)\n")
    report.append("- **Graph Encoder:** GATv2 (2 layers, 4 heads)\n")
    report.append("- **Fusion:** Multi-head Cross-Attention\n")
    report.append("- **Training:** Mixed precision, gradient accumulation\n\n")
    
    # Conclusion
    report.append("## Conclusion\n\n")
    if macro_auc and macro_auc >= 0.85:
        report.append("The model has successfully achieved the target macro AUC score of >0.85.\n")
        report.append("The neuro-symbolic approach combining vision transformers with knowledge graphs\n")
        report.append("demonstrates effective performance on the CheXpert dataset.\n")
    else:
        report.append("The model performance is below the target. Consider:\n")
        report.append("- Additional training epochs\n")
        report.append("- Hyperparameter tuning\n")
        report.append("- Data augmentation\n")
        report.append("- Model architecture adjustments\n")
    
    return "".join(report)


def create_visualizations(metrics: dict, output_dir: Path):
    """Create visualization plots."""
    logger.info("Creating visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Per-class AUC bar chart
    if 'auc' in metrics:
        auc_scores = metrics['auc']
        pathologies = []
        aucs = []
        
        for pathology in PATHOLOGY_CLASSES:
            if pathology in auc_scores and auc_scores[pathology] is not None:
                pathologies.append(pathology)
                aucs.append(auc_scores[pathology])
        
        if aucs:
            plt.figure(figsize=(14, 8))
            bars = plt.bar(range(len(pathologies)), aucs, color='steelblue')
            plt.xlabel('Pathology', fontsize=12)
            plt.ylabel('AUC Score', fontsize=12)
            plt.title('Per-Class AUC Scores', fontsize=14, fontweight='bold')
            plt.xticks(range(len(pathologies)), pathologies, rotation=45, ha='right')
            plt.ylim([0, 1])
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, auc) in enumerate(zip(bars, aucs)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_dir / "per_class_auc.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved per-class AUC plot to {output_dir / 'per_class_auc.png'}")


def main():
    """Run results analysis."""
    logger.info("=" * 80)
    logger.info("RESULTS ANALYSIS")
    logger.info("=" * 80)
    
    # Load metrics
    metrics = load_metrics()
    if metrics is None:
        return 1
    
    # Generate report
    logger.info("Generating results report...")
    report = generate_results_report(metrics)
    
    # Save report
    report_path = OUTPUTS_DIR / "results_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Results report saved to {report_path}")
    
    # Save summary JSON
    summary = {
        "macro_auc": metrics.get('auc', {}).get('macro_auc', None),
        "macro_precision": metrics.get('macro_precision', None),
        "macro_recall": metrics.get('macro_recall', None),
        "macro_f1": metrics.get('macro_f1', None),
        "per_class_auc": {k: v for k, v in metrics.get('auc', {}).items() 
                         if k != 'macro_auc'}
    }
    
    summary_path = OUTPUTS_DIR / "results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results summary saved to {summary_path}")
    
    # Create visualizations
    create_visualizations(metrics, OUTPUTS_DIR)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    macro_auc = metrics.get('auc', {}).get('macro_auc', None)
    if macro_auc:
        logger.info(f"Macro AUC: {macro_auc:.4f}")
        if macro_auc >= 0.85:
            logger.info("✓ Target achieved! (Macro AUC > 0.85)")
        else:
            logger.info("⚠ Target not achieved (Macro AUC < 0.85)")
    
    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("\n✓ Results analysis completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
