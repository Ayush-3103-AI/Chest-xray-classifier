# Collaborative Development Workflow Guide

## Overview
This document outlines the workflow for developing code on a system without GPU/dataset and executing it on a teammate's system with GPU and dataset via GitHub.

---

## Phase 1: Initial Setup (Both Systems)

### Step 1: Initialize Git Repository
```bash
# On your system (development machine)
git init
git branch -M main
```

### Step 2: Create .gitignore
Create a `.gitignore` file to exclude:
- Dataset files (too large for GitHub)
- Model checkpoints (can be large)
- Python cache files
- Virtual environments
- Output files (logs, inference JSONs can be regenerated)

### Step 3: Create Initial Project Structure
Based on the PRD, create the directory structure:
```
medai_graph_project/
├── data/
│   ├── raw/          # .gitkeep only (dataset excluded)
│   ├── processed/    # .gitkeep only
│   └── test_images/  # .gitkeep only
├── src/
├── frontend/
├── checkpoints/      # .gitkeep only
├── outputs/          # .gitkeep only
├── run_train.py
├── run_inference.py
├── requirements.txt
└── README.md
```

### Step 4: Create README.md
Document:
- Project description
- Setup instructions
- How to download/place the dataset
- How to run training/inference
- Environment requirements

---

## Phase 2: Development Workflow (Your System)

### Step 5: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 6: Install Dependencies
```bash
# Create requirements.txt with all dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install transformers
pip install pillow numpy pandas scikit-learn
pip install matplotlib seaborn
# ... add all other dependencies

# Freeze requirements
pip freeze > requirements.txt
```

### Step 7: Develop Code (CPU Mode)
- Write all code with GPU checks: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Test basic functionality on CPU (syntax, imports, logic)
- Use small mock data for testing
- Add proper error handling

### Step 8: Commit and Push Regularly
```bash
# Stage changes
git add .

# Commit with descriptive messages
git commit -m "Add image encoder module with BioViL-T wrapper"

# Push to GitHub
git push origin main
```

**Best Practices:**
- Commit frequently (after each module/feature)
- Write clear commit messages
- Test imports and basic logic before committing
- Never commit dataset files or large checkpoints

---

## Phase 3: Teammate Setup (GPU System)

### Step 9: Clone Repository
```bash
git clone <your-github-repo-url>
cd Medical_image_classifier
```

### Step 10: Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 11: Place Dataset
```bash
# Teammate places CheXpert dataset in:
data/raw/CheXpert-v1.0-small/
```

### Step 12: Verify Setup
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify dataset path
python -c "import os; print(os.path.exists('data/raw/CheXpert-v1.0-small/valid.csv'))"
```

---

## Phase 4: Iterative Development Cycle

### Daily Workflow:

**On Your System (Development):**
1. Pull latest changes (if teammate made fixes)
   ```bash
   git pull origin main
   ```

2. Develop new features/fix bugs
3. Test basic functionality (CPU mode)
4. Commit and push
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin main
   ```

**On Teammate's System (Execution):**
1. Pull latest code
   ```bash
   git pull origin main
   ```

2. Run training/inference
   ```bash
   python run_train.py
   # or
   python run_inference.py
   ```

3. If issues found:
   - Fix locally OR
   - Report back to you with error logs
   - You fix and push, teammate pulls again

4. (Optional) Commit results/logs if needed
   ```bash
   git add outputs/training_log.txt
   git commit -m "Add training results"
   git push origin main
   ```

---

## Phase 5: Branch Strategy (Recommended)

### For Feature Development:
```bash
# Create feature branch
git checkout -b feature/image-encoder

# Develop and commit
git add .
git commit -m "Implement image encoder"

# Push branch
git push origin feature/image-encoder

# Create Pull Request on GitHub (optional)
# Or merge directly:
git checkout main
git merge feature/image-encoder
git push origin main
```

---

## Important Files to Create

### 1. `.gitignore`
```
# Dataset (too large)
data/raw/CheXpert-v1.0-small/
data/processed/*.pt

# Model checkpoints (optional - can exclude if too large)
checkpoints/*.pth

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Outputs (can be regenerated)
outputs/*.json
outputs/*.log
outputs/*.png

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### 2. `requirements.txt`
```
torch>=2.1.0
torchvision>=0.16.0
torch-geometric>=2.4.0
transformers>=4.35.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.66.0
```

### 3. `README.md`
Should include:
- Project overview
- Installation steps
- Dataset setup instructions
- How to run training
- How to run inference
- Project structure
- Troubleshooting

### 4. `config.py` (in src/)
Should have:
- Paths that work on both systems
- Relative paths (not absolute)
- Easy-to-modify hyperparameters

---

## Troubleshooting Common Issues

### Issue: Path Differences (Windows vs Linux)
**Solution:** Use `pathlib` or `os.path.join()` for all paths
```python
from pathlib import Path
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
```

### Issue: CUDA Version Mismatch
**Solution:** Document required CUDA version in README, use compatible PyTorch version

### Issue: Large Files in Git
**Solution:** Use Git LFS for large files, or exclude them entirely

### Issue: Merge Conflicts
**Solution:** 
- Communicate before editing same files
- Use branches for parallel development
- Pull before pushing

---

## Communication Protocol

1. **Before Major Changes:** Notify teammate
2. **After Pushing:** Message teammate to pull
3. **When Errors Occur:** Share full error traceback
4. **Documentation:** Update README with any new setup steps

---

## Quick Reference Commands

```bash
# Your system (development)
git status                    # Check changes
git add .                     # Stage all
git commit -m "message"       # Commit
git push origin main          # Push

# Teammate system (execution)
git pull origin main          # Get latest code
python run_train.py           # Run training
```

---

## Next Steps

1. ✅ Initialize Git repository
2. ✅ Create project structure
3. ✅ Setup .gitignore
4. ✅ Create requirements.txt
5. ✅ Start developing Phase 1 (Data Ingestion)
6. ✅ Push to GitHub
7. ✅ Teammate clones and tests
