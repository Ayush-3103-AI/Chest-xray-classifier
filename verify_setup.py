"""
Setup verification script.
Run this to check if the environment is properly configured.
"""
import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version >= 3.10"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor}.{version.micro} is too old. Need >= 3.10")
        return False

def check_imports():
    """Check if required packages can be imported"""
    required_packages = [
        'torch',
        'torchvision',
        'torch_geometric',
        'transformers',
        'PIL',
        'numpy',
        'pandas',
        'sklearn',
    ]
    
    failed = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ {package} import failed: {e}")
            failed.append(package)
    
    return len(failed) == 0

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA is available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA is not available (will use CPU - much slower)")
        return True  # Not a failure, just a warning
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_dataset():
    """Check if dataset is in correct location"""
    dataset_path = Path("medai_graph_project/data/raw/CheXpert-v1.0-small")
    valid_csv = dataset_path / "valid.csv"
    train_csv = dataset_path / "train.csv"
    
    if valid_csv.exists() and train_csv.exists():
        print(f"✅ Dataset found at: {dataset_path}")
        return True
    else:
        print(f"⚠️  Dataset not found at: {dataset_path}")
        print("   Place CheXpert-v1.0-small dataset in the above location")
        return False  # This is expected on development system

def check_project_structure():
    """Check if project directories exist"""
    required_dirs = [
        "medai_graph_project/src",
        "medai_graph_project/src/models",
        "medai_graph_project/src/training",
        "medai_graph_project/src/visualization",
        "medai_graph_project/data/raw",
        "medai_graph_project/data/processed",
        "medai_graph_project/frontend",
        "medai_graph_project/checkpoints",
        "medai_graph_project/outputs",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 60)
    print("Medical Image Classifier - Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    
    print("1. Checking Python version...")
    results.append(check_python_version())
    print()
    
    print("2. Checking project structure...")
    results.append(check_project_structure())
    print()
    
    print("3. Checking package imports...")
    results.append(check_imports())
    print()
    
    print("4. Checking CUDA availability...")
    check_cuda()  # Warning only, not a failure
    print()
    
    print("5. Checking dataset...")
    dataset_ok = check_dataset()
    # Dataset check is optional (expected to fail on dev system)
    print()
    
    print("=" * 60)
    if all(results):
        print("✅ Setup verification PASSED")
        if not dataset_ok:
            print("   (Dataset check failed - expected on development system)")
        print("\nYou're ready to start development!")
    else:
        print("❌ Setup verification FAILED")
        print("   Please fix the issues above before proceeding")
    print("=" * 60)

if __name__ == "__main__":
    main()
