# Medical Image Classifier - Neuro-Symbolic Chest X-Ray Diagnosis

A "Glass-Box" Medical AI pipeline that classifies Chest X-Rays into 14 pathologies while generating a 3D interactive graph visualization.

## 🎯 Project Overview

This project implements a neuro-symbolic AI system that:
- **Diagnoses**: Uses Domain-Adapted Vision Transformers (BioViL-T) for SOTA classification
- **Reasons**: Employs Graph Attention Networks (GATv2) encoding medical hierarchy
- **Visualizes**: Generates 3D Force-Directed Graphs showing diagnosis reasoning

## 📋 Requirements

- Python 3.10+
- CUDA 11.8/12.1 compatible GPU (for training)
- 16GB+ VRAM recommended
- CheXpert-v1.0-small dataset

## 🚀 Setup Instructions

### 1. Clone Repository
```bash
git clone <repository-url>
cd Medical_image_classifier
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note for GPU systems:** If you have CUDA installed, PyTorch should detect it automatically. Verify with:
```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 4. Dataset Setup

1. Download CheXpert-v1.0-small dataset
2. Extract it to: `data/raw/CheXpert-v1.0-small/`
3. Verify the structure:
   ```
   data/raw/CheXpert-v1.0-small/
   ├── train/
   ├── valid/
   ├── train.csv
   └── valid.csv
   ```

### 5. Verify Installation
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check dataset
python -c "import os; print(os.path.exists('data/raw/CheXpert-v1.0-small/valid.csv'))"
```

## 📁 Project Structure

```
medai_graph_project/
├── data/
│   ├── raw/              # CheXpert dataset (not in git)
│   ├── processed/        # Generated graph files
│   └── test_images/      # Single images for inference
├── src/
│   ├── config.py         # Global constants
│   ├── utils.py          # Metrics, logging
│   ├── models/           # Model architectures
│   ├── training/         # Training scripts
│   └── visualization/    # Graph visualization
├── frontend/             # 3D graph viewer
├── checkpoints/          # Model weights
├── outputs/              # Logs and inference JSONs
├── run_train.py          # Training entry point
└── run_inference.py      # Inference entry point
```

## 🏃 Usage

### Training
```bash
python run_train.py
```

### Inference
```bash
python run_inference.py --image_path data/test_images/example.jpg
```

### View Visualization
```bash
# Start web server
cd frontend
python -m http.server 8000

# Open browser to http://localhost:8000
```

## 👥 Development Workflow

This project uses a collaborative workflow:
- **Development System**: Code development (no GPU/dataset needed)
- **Execution System**: Training and inference (GPU + dataset required)
- **GitHub**: Code synchronization between systems

See `WORKFLOW_GUIDE.md` for detailed workflow instructions.

## 📊 Expected Results

- **Macro AUC Score**: Target >0.85 on validation set
- **Visualization**: Interactive 3D graph showing patient X-ray interaction with disease clusters

## 🐛 Troubleshooting

### CUDA Not Available
- Verify CUDA installation: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version
- Code will fallback to CPU (much slower)

### Dataset Not Found
- Verify dataset path: `data/raw/CheXpert-v1.0-small/`
- Check CSV files exist: `train.csv`, `valid.csv`

### Out of Memory
- Reduce batch size in `src/config.py`
- Use gradient accumulation
- Enable mixed precision training

## 📝 License

[Add your license here]

## 👨‍💻 Contributors

[Add contributor names]
