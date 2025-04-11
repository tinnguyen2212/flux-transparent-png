# Flux Transparent PNG Generator

This repository contains a comprehensive solution for training and generating transparent PNG images without backgrounds using the Flux.1 dev model. The implementation is based on the Microsoft ART-MSRA project architecture but adapted specifically for single-layer transparent PNG generation.

## Features

- Train a modified VAE on transparent PNG images
- Save trained VAE and decoder models
- Generate new transparent PNG images with transparency
- ComfyUI integration with custom nodes
- Google Colab notebooks for easy usage

## Repository Structure

- `python/`: Python implementation of the training and generation pipeline
- `colab/`: Google Colab notebooks for training and generation
- `web/`: Web interface for the project

## Quick Links

- [Usage Instructions in English](README.md)
- [Hướng dẫn sử dụng bằng tiếng Việt](HUONG_DAN_SU_DUNG.md)
- [Training Notebook](https://colab.research.google.com/github/tinnguyen2212/flux-transparent-png/blob/main/colab/train_transparent_png.ipynb)
- [Generation Notebook](https://colab.research.google.com/github/tinnguyen2212/flux-transparent-png/blob/main/colab/generate_transparent_png.ipynb)

## Getting Started

The easiest way to get started is to use the Google Colab notebooks:

1. [Training Notebook](https://colab.research.google.com/github/tinnguyen2212/flux-transparent-png/blob/main/colab/train_transparent_png.ipynb) - Use this to train a VAE model on transparent PNG images
2. [Generation Notebook](https://colab.research.google.com/github/tinnguyen2212/flux-transparent-png/blob/main/colab/generate_transparent_png.ipynb) - Use this to generate transparent PNG images from the trained model

## Installation

For local installation:

```bash
# Clone the repository
git clone https://github.com/tinnguyen2212/flux-transparent-png.git
cd flux-transparent-png/python

# Install dependencies
python install.py
```

## Web Interface

To run the web interface locally:

```bash
cd web
npm install
npm start
```

Then open your browser to http://localhost:3000

## License

This project is released under the same license as Flux.1 dev. Please refer to the Flux.1 dev license for details.
