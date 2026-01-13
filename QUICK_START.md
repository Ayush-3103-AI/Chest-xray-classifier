# Quick Start Checklist

## ✅ Immediate Next Steps

### 1. Initialize Git Repository (Your System)
```bash
cd "D:\5_sem\Mini Project\Medical_image_classifier"
git init
git branch -M main
git add .
git commit -m "Initial project setup with structure and documentation"
```

### 2. Create GitHub Repository
1. Go to GitHub.com
2. Create a new repository (name: `Medical_image_classifier` or similar)
3. **DO NOT** initialize with README (you already have one)
4. Copy the repository URL

### 3. Connect Local Repo to GitHub
```bash
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 4. Share Repository with Teammate
- Add teammate as collaborator in GitHub repository settings
- Or make repository public (if allowed)
- Share the repository URL

### 5. Teammate Setup (On GPU System)
```bash
# Clone repository
git clone <repository-url>
cd Medical_image_classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Place dataset
# Extract CheXpert-v1.0-small to: medai_graph_project/data/raw/CheXpert-v1.0-small/

# Verify setup
python verify_setup.py
```

## 🔄 Daily Workflow

### When You Develop Code:
1. Make changes
2. Test basic imports/logic (CPU mode)
3. Commit: `git add . && git commit -m "Description"`
4. Push: `git push origin main`
5. Notify teammate to pull

### When Teammate Executes:
1. Pull latest: `git pull origin main`
2. Run training/inference
3. Report any errors back to you
4. (Optional) Push results/logs if needed

## 📋 Development Priority (Based on PRD)

1. **Phase 1**: Data Ingestion & Policy Handling
   - Create `src/config.py`
   - Create `src/training/dataset.py`
   - Implement U-Ones policy

2. **Phase 2**: Knowledge Graph Construction
   - Create `src/visualization/graph_builder.py`
   - Define adjacency matrix
   - Initialize node embeddings

3. **Phase 3**: Model Architecture
   - Create `src/models/image_encoder.py`
   - Create `src/models/graph_encoder.py`
   - Create `src/models/cross_attention.py`
   - Create `src/models/medai_pipeline.py`

4. **Phase 4**: Training
   - Create `src/training/trainer.py`
   - Create `src/training/loss.py`
   - Create `run_train.py`

5. **Phase 5**: Inference & Visualization
   - Create `src/visualization/exporter.py`
   - Create `run_inference.py`
   - Create `frontend/index.html`

## 🚨 Important Reminders

- ✅ Never commit dataset files
- ✅ Never commit large checkpoint files (unless using Git LFS)
- ✅ Use relative paths in code
- ✅ Test imports before committing
- ✅ Write clear commit messages
- ✅ Pull before pushing (to avoid conflicts)

## 📞 Communication Protocol

- **Before major changes**: Notify teammate
- **After pushing**: Message teammate to pull
- **When errors occur**: Share full traceback
- **Documentation**: Update README with new steps
