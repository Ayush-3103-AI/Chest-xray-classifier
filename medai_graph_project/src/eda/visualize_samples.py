"""Image visualization and quality assessment for CheXpert dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from PIL import Image
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    logger.warning("pydicom not installed. DICOM file reading will be limited.")


def load_dicom_image(dicom_path: Path) -> np.ndarray:
    """Load DICOM image and convert to numpy array."""
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required for DICOM file reading. Install with: pip install pydicom")
    
    try:
        ds = pydicom.dcmread(str(dicom_path))
        image = ds.pixel_array
        
        # Normalize to 0-255 range if needed
        if image.max() > 255:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        return image
    except Exception as e:
        logger.error(f"Error loading DICOM file {dicom_path}: {e}")
        raise


def load_image(image_path: Path) -> np.ndarray:
    """Load image (DICOM or standard format) and return as numpy array."""
    if image_path.suffix.lower() in ['.dcm', '.dicom']:
        return load_dicom_image(image_path)
    else:
        # Try loading as standard image
        try:
            img = Image.open(image_path)
            return np.array(img.convert('L'))  # Convert to grayscale
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise


def calculate_image_statistics(image: np.ndarray) -> dict:
    """Calculate statistics for an image."""
    return {
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "min": int(np.min(image)),
        "max": int(np.max(image)),
        "shape": list(image.shape),
        "dtype": str(image.dtype)
    }


def sample_images_by_class(df: pd.DataFrame, pathology_cols: list, 
                           data_dir: Path, samples_per_class: int = 3) -> dict:
    """Sample images for each pathology class."""
    logger.info("Sampling images by pathology class...")
    
    sampled_images = {}
    
    for pathology in pathology_cols:
        if pathology not in df.columns:
            continue
        
        # Get positive samples (label == 1)
        positive_samples = df[df[pathology] == 1.0]
        
        if len(positive_samples) == 0:
            logger.warning(f"No positive samples found for {pathology}")
            sampled_images[pathology] = []
            continue
        
        # Sample up to samples_per_class images
        n_samples = min(samples_per_class, len(positive_samples))
        sampled = positive_samples.sample(n=n_samples, random_state=42)
        
        image_paths = []
        for idx, row in sampled.iterrows():
            if pd.isna(row["Path"]):
                continue
            
            full_path = data_dir / row["Path"]
            if full_path.exists():
                image_paths.append({
                    "path": str(row["Path"]),
                    "full_path": str(full_path),
                    "index": int(idx)
                })
        
        sampled_images[pathology] = image_paths
        logger.info(f"Sampled {len(image_paths)} images for {pathology}")
    
    return sampled_images


def visualize_sample_images(sampled_images: dict, df: pd.DataFrame, 
                           data_dir: Path, output_dir: Path, max_per_class: int = 3):
    """Create visualization grid of sample images."""
    logger.info("Creating sample image visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a grid for each pathology
    for pathology, image_list in sampled_images.items():
        if len(image_list) == 0:
            continue
        
        n_images = min(len(image_list), max_per_class)
        if n_images == 0:
            continue
        
        # Create subplot grid
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for idx, img_info in enumerate(image_list[:n_images]):
            try:
                image_path = Path(img_info["full_path"])
                image = load_image(image_path)
                
                ax = axes[idx] if n_images > 1 else axes[0]
                ax.imshow(image, cmap='gray')
                ax.set_title(f"{pathology}\n{Path(img_info['path']).name}", 
                           fontsize=10, wrap=True)
                ax.axis('off')
                
            except Exception as e:
                logger.warning(f"Could not load image {img_info['path']}: {e}")
                if n_images > 1:
                    axes[idx].text(0.5, 0.5, f"Error loading\n{Path(img_info['path']).name}", 
                                  ha='center', va='center')
                    axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Sample Images: {pathology}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        safe_pathology = pathology.replace('/', '_').replace(' ', '_')
        plt.savefig(output_dir / f"sample_images_{safe_pathology}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Sample image visualizations saved to {output_dir}")


def analyze_image_quality(df: pd.DataFrame, pathology_cols: list, 
                         data_dir: Path, sample_size: int = 100) -> dict:
    """Analyze image quality and dimensions."""
    logger.info(f"Analyzing image quality (sampling {sample_size} images)...")
    
    # Sample random images
    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    image_stats = []
    corrupted_images = []
    missing_images = []
    
    for idx, row in sampled_df.iterrows():
        if pd.isna(row["Path"]):
            continue
        
        image_path = data_dir / row["Path"]
        
        if not image_path.exists():
            missing_images.append(str(row["Path"]))
            continue
        
        try:
            image = load_image(image_path)
            stats = calculate_image_statistics(image)
            stats["path"] = str(row["Path"])
            image_stats.append(stats)
            
        except Exception as e:
            corrupted_images.append({
                "path": str(row["Path"]),
                "error": str(e)
            })
            logger.warning(f"Error processing {row['Path']}: {e}")
    
    # Aggregate statistics
    if len(image_stats) > 0:
        mean_values = [s["mean"] for s in image_stats]
        std_values = [s["std"] for s in image_stats]
        shapes = [tuple(s["shape"]) for s in image_stats]
        
        # Get unique shapes
        unique_shapes = list(set(shapes))
        shape_counts = {str(shape): shapes.count(shape) for shape in unique_shapes}
        
        quality_summary = {
            "total_analyzed": len(image_stats),
            "mean_pixel_mean": float(np.mean(mean_values)),
            "mean_pixel_std": float(np.mean(std_values)),
            "unique_shapes": unique_shapes,
            "shape_distribution": shape_counts,
            "corrupted_count": len(corrupted_images),
            "missing_count": len(missing_images)
        }
    else:
        quality_summary = {
            "total_analyzed": 0,
            "error": "No images could be analyzed"
        }
    
    return {
        "summary": quality_summary,
        "corrupted_images": corrupted_images[:10],  # Limit to first 10
        "missing_images": missing_images[:10]  # Limit to first 10
    }


def visualize_image_statistics(image_stats: dict, output_dir: Path):
    """Create visualizations for image statistics."""
    logger.info("Creating image statistics visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if image_stats["summary"]["total_analyzed"] == 0:
        logger.warning("No image statistics to visualize")
        return
    
    # Load detailed stats if available
    # For now, we'll create a summary visualization
    summary = image_stats["summary"]
    
    # Create a simple bar chart for shape distribution
    if "shape_distribution" in summary and summary["shape_distribution"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        shapes = list(summary["shape_distribution"].keys())
        counts = list(summary["shape_distribution"].values())
        
        ax.bar(range(len(shapes)), counts)
        ax.set_xticks(range(len(shapes)))
        ax.set_xticklabels(shapes, rotation=45, ha='right')
        ax.set_xlabel('Image Shape (Height, Width)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Image Dimension Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "image_dimension_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Image statistics visualization saved to {output_dir}")


def run_image_visualization(train_df: pd.DataFrame, valid_df: pd.DataFrame,
                            data_dir: Path, output_dir: Path) -> dict:
    """Main function to run image visualization and quality analysis."""
    logger.info("=" * 60)
    logger.info("Starting Image Visualization and Quality Analysis")
    logger.info("=" * 60)
    
    pathology_cols = [col for col in train_df.columns 
                     if col not in ["Path", "Patient", "Sex", "Age", 
                                   "Frontal/Lateral", "AP/PA", "No Finding"]]
    
    # Sample images by class
    sampled_images = sample_images_by_class(train_df, pathology_cols, data_dir, 
                                           samples_per_class=3)
    
    # Visualize samples
    viz_dir = output_dir / "eda_visualizations"
    visualize_sample_images(sampled_images, train_df, data_dir, viz_dir)
    
    # Analyze image quality
    quality_stats = analyze_image_quality(train_df, pathology_cols, data_dir, 
                                         sample_size=100)
    
    # Visualize statistics
    visualize_image_statistics(quality_stats, viz_dir)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eda_report.json"
    
    # Update existing report
    if report_path.exists():
        with open(report_path, 'r') as f:
            existing_report = json.load(f)
    else:
        existing_report = {}
    
    existing_report["image_quality"] = quality_stats
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_report = convert_to_serializable(existing_report)
    
    with open(report_path, 'w') as f:
        json.dump(serializable_report, f, indent=2)
    
    logger.info(f"Image quality analysis saved to {report_path}")
    
    return {
        "sampled_images": {k: len(v) for k, v in sampled_images.items()},
        "quality_stats": quality_stats
    }


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw" / "CheXpert-v1.0-small"
    output_dir = project_root / "outputs"
    
    # Load data
    train_df = pd.read_csv(data_dir / "train.csv")
    valid_df = pd.read_csv(data_dir / "valid.csv")
    
    results = run_image_visualization(train_df, valid_df, data_dir, output_dir)
    logger.info("Image visualization completed!")
