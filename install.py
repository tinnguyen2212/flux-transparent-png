#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Installation script for the Transparent PNG Training and Generation Pipeline.
This script installs the required dependencies and sets up the project.
"""

import os
import sys
import subprocess
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing dependencies...")
    
    # Core dependencies
    dependencies = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.24.0",
        "transformers>=4.30.0",
        "pillow>=9.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0"
    ]
    
    # Install dependencies
    for dep in dependencies:
        logger.info(f"Installing {dep}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {dep}: {e}")
            sys.exit(1)
    
    logger.info("All dependencies installed successfully")

def create_directories(args):
    """Create necessary directories."""
    logger.info("Creating directories...")
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "OUT"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    logger.info("Directories created successfully")

def setup_comfyui(args):
    """Set up ComfyUI integration."""
    if not args.comfyui_dir:
        logger.info("Skipping ComfyUI setup (no directory provided)")
        return
    
    logger.info(f"Setting up ComfyUI integration in {args.comfyui_dir}")
    
    # Check if ComfyUI directory exists
    if not os.path.exists(args.comfyui_dir):
        logger.error(f"ComfyUI directory not found: {args.comfyui_dir}")
        return
    
    # Create custom_nodes directory if it doesn't exist
    custom_nodes_dir = os.path.join(args.comfyui_dir, "custom_nodes")
    os.makedirs(custom_nodes_dir, exist_ok=True)
    
    # Create transparent_vae directory in models
    models_dir = os.path.join(args.comfyui_dir, "models")
    transparent_vae_dir = os.path.join(models_dir, "transparent_vae")
    os.makedirs(transparent_vae_dir, exist_ok=True)
    
    # Copy ComfyUI nodes
    src_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfyui_transparent_png_nodes.py")
    dst_file = os.path.join(custom_nodes_dir, "comfyui_transparent_png_nodes.py")
    
    try:
        import shutil
        shutil.copy2(src_file, dst_file)
        logger.info(f"Copied ComfyUI nodes to {dst_file}")
    except Exception as e:
        logger.error(f"Failed to copy ComfyUI nodes: {e}")
    
    logger.info("ComfyUI integration set up successfully")
    logger.info(f"Place your trained models in: {transparent_vae_dir}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Install Transparent PNG Training and Generation Pipeline")
    
    # Directory parameters
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/SD-Data/TrainData/4000_PNG/TEST",
                        help="Directory for training data")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/VAE-DECODER",
                        help="Directory for output models and images")
    parser.add_argument("--comfyui_dir", type=str, default="",
                        help="ComfyUI installation directory (optional)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run installation steps
    check_python_version()
    install_dependencies()
    create_directories(args)
    setup_comfyui(args)
    
    logger.info("Installation completed successfully")
    logger.info("You can now use the Transparent PNG Training and Generation Pipeline")
    logger.info("See README.md for usage instructions")

if __name__ == "__main__":
    main()
