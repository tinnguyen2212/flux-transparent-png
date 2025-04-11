#!/bin/bash

# Script to create a GitHub repository and push the Flux Transparent PNG Generator code

# Configuration
REPO_NAME="flux-transparent-png"
DESCRIPTION="Train and generate transparent PNG images using Flux.1 dev"
USERNAME="tinnguyen2212"  # Replace with your GitHub username

# Create directory structure
mkdir -p $REPO_NAME/python
mkdir -p $REPO_NAME/colab
mkdir -p $REPO_NAME/web

# Copy Python files to python directory
cp /home/ubuntu/flux_png_project/*.py $REPO_NAME/python/
cp /home/ubuntu/flux_png_project/README.md $REPO_NAME/python/

# Copy Colab notebooks
cp /home/ubuntu/flux_png_web/colab/*.ipynb $REPO_NAME/colab/

# Copy web files
cp /home/ubuntu/flux_png_web/index.html $REPO_NAME/web/
cp /home/ubuntu/flux_png_web/package.json $REPO_NAME/web/
cp /home/ubuntu/flux_png_web/server.js $REPO_NAME/web/

# Copy main README and usage instructions
cp /home/ubuntu/flux_png_web/README.md $REPO_NAME/
cp /home/ubuntu/flux_png_web/HUONG_DAN_SU_DUNG.md $REPO_NAME/

# Create .gitignore file
cat > $REPO_NAME/.gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Node.js
node_modules/
npm-debug.log
yarn-debug.log
yarn-error.log
package-lock.json

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Checkpoints and models
*.pt
*.pth
*.ckpt
*.safetensors

# Generated images
*.png
*.jpg
*.jpeg
*.gif
EOL

# Create README for the repository root
cat > $REPO_NAME/README.md << EOL
# Flux Transparent PNG Generator

This repository contains a comprehensive solution for training and generating transparent PNG images without backgrounds using the Flux.1 dev model. The implementation is based on the Microsoft ART-MSRA project architecture but adapted specifically for single-layer transparent PNG generation.

## Features

- Train a modified VAE on transparent PNG images
- Save trained VAE and decoder models
- Generate new transparent PNG images with transparency
- ComfyUI integration with custom nodes
- Google Colab notebooks for easy usage

## Repository Structure

- \`python/\`: Python implementation of the training and generation pipeline
- \`colab/\`: Google Colab notebooks for training and generation
- \`web/\`: Web interface for the project

## Quick Links

- [Usage Instructions in English](README.md)
- [Hướng dẫn sử dụng bằng tiếng Việt](HUONG_DAN_SU_DUNG.md)
- [Training Notebook](https://colab.research.google.com/github/$USERNAME/flux-transparent-png/blob/main/colab/train_transparent_png.ipynb)
- [Generation Notebook](https://colab.research.google.com/github/$USERNAME/flux-transparent-png/blob/main/colab/generate_transparent_png.ipynb)

## Getting Started

The easiest way to get started is to use the Google Colab notebooks:

1. [Training Notebook](https://colab.research.google.com/github/$USERNAME/flux-transparent-png/blob/main/colab/train_transparent_png.ipynb) - Use this to train a VAE model on transparent PNG images
2. [Generation Notebook](https://colab.research.google.com/github/$USERNAME/flux-transparent-png/blob/main/colab/generate_transparent_png.ipynb) - Use this to generate transparent PNG images from the trained model

## Installation

For local installation:

\`\`\`bash
# Clone the repository
git clone https://github.com/$USERNAME/flux-transparent-png.git
cd flux-transparent-png/python

# Install dependencies
python install.py
\`\`\`

## Web Interface

To run the web interface locally:

\`\`\`bash
cd web
npm install
npm start
\`\`\`

Then open your browser to http://localhost:3000

## License

This project is released under the same license as Flux.1 dev. Please refer to the Flux.1 dev license for details.
EOL

echo "Repository structure created at $REPO_NAME"
echo "To create and push to GitHub, run the following commands:"
echo "cd $REPO_NAME"
echo "git init"
echo "git add ."
echo "git commit -m \"Initial commit\""
echo "gh repo create $REPO_NAME --public --description \"$DESCRIPTION\" --source=. --push"
echo ""
echo "Or create a repository manually on GitHub and push with:"
echo "git remote add origin https://github.com/$USERNAME/$REPO_NAME.git"
echo "git push -u origin main"
