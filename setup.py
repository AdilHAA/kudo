import os
import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Set up a virtual environment and install necessary packages."""
    venv_dir = Path("venv")
    
    # Create virtual environment if it doesn't exist
    if not venv_dir.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Determine the pip path
    if os.name == "nt":  # Windows
        pip_path = venv_dir / "Scripts" / "pip"
    else:  # Linux, macOS
        pip_path = venv_dir / "bin" / "pip"
    
    # Install packages
    print("Installing dependencies...")
    packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",
        "shap",
        "prophet",
        "autogluon.timeseries",
    ]
    
    # Upgrade pip first
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    
    # Install each package
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([str(pip_path), "install", package], check=True)
    
    print("Virtual environment setup complete!")
    print(f"Activate it with:")
    if os.name == "nt":  # Windows
        print(r"    venv\Scripts\activate")
    else:  # Linux, macOS
        print("    source venv/bin/activate")

if __name__ == "__main__":
    setup_environment() 