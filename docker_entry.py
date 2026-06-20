"""Docker entry point - starts Jupyter Lab"""
import subprocess
import sys

if __name__ == "__main__":
    print("Starting HomiCSx Jupyter environment...")
    subprocess.run([
        "jupyter", "lab", 
        "--ip=0.0.0.0", 
        "--port=8888", 
        "--no-browser", 
        "--allow-root",
        "--NotebookApp.token=''",
        "--notebook-dir=/root/examples"
    ])