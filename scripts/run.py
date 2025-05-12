import os
import sys
import subprocess
import platform

def run_setup():
    """Set up the virtual environment and install dependencies"""
    print("Setting up the environment...")
    
    # Добавляем пути к директориям
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Create virtual environment if it doesn't exist
    venv_dir = os.path.join(project_root, "venv")
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    
    # Determine the pip path
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_dir, "Scripts", "pip")
        python_path = os.path.join(venv_dir, "Scripts", "python")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
        python_path = os.path.join(venv_dir, "bin", "python")
    
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
    subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    
    # Install each package
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([pip_path, "install", package], check=True)
    
    print("Environment setup complete!")
    return python_path, project_root

def run_analysis(python_path, project_root):
    """Run the sales analysis script"""
    print("\nRunning sales analysis...")
    script_path = os.path.join(project_root, "scripts", "main.py")
    subprocess.run([python_path, script_path], check=True)

if __name__ == "__main__":
    try:
        python_path, project_root = run_setup()
        run_analysis(python_path, project_root)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 