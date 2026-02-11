# Hugging Face Spaces entry point
import subprocess
import sys

if __name__ == "__main__":
    # Start the FastAPI server
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "similarity_service_enhanced:app",
        "--host", "0.0.0.0",
        "--port", "7860"  # Hugging Face default port
    ])
