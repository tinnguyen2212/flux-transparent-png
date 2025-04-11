#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running the transparent PNG training and generation pipeline.
This script provides a unified interface for all pipeline components.
"""

import os
import argparse
import logging
import torch
from train_transparent_png import main as train_main
from save_vae_decoder import main as save_main
from generate_transparent_png import main as generate_main
from test_pipeline import run_all_tests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_train_parser(subparsers):
    """
    Set up the argument parser for the train command.
    
    Args:
        subparsers: Subparsers object from argparse
    """
    parser = subparsers.add_parser('train', help='Train VAE for transparent PNG images')
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/SD-Data/TrainData/4000_PNG/TEST",
                        help="Directory containing transparent PNG images")
    
    # Model parameters
    parser.add_argument("--pretrained_model", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="Pretrained model to use as base")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Size to resize images to")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--alpha_weight", type=float, default=2.0,
                        help="Weight for alpha channel loss")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/VAE-DECODER",
                        help="Directory to save trained models")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/VAE-DECODER/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--vis_frequency", type=int, default=10,
                        help="Frequency of visualization (in epochs)")

def setup_save_parser(subparsers):
    """
    Set up the argument parser for the save command.
    
    Args:
        subparsers: Subparsers object from argparse
    """
    parser = subparsers.add_parser('save', help='Save VAE and Decoder models')
    
    # Input parameters
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to the checkpoint to load")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/VAE-DECODER/checkpoints",
                        help="Directory containing checkpoints (used if checkpoint_path is not specified)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/VAE-DECODER",
                        help="Directory to save models")
    parser.add_argument("--vae_filename", type=str, default="transparent_vae.pt",
                        help="Filename for the saved VAE")
    parser.add_argument("--decoder_filename", type=str, default="transparent_decoder.pt",
                        help="Filename for the saved decoder")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load the model on (cuda or cpu)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify that saved models can be loaded correctly")

def setup_generate_parser(subparsers):
    """
    Set up the argument parser for the generate command.
    
    Args:
        subparsers: Subparsers object from argparse
    """
    parser = subparsers.add_parser('generate', help='Generate transparent PNG images')
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="/content/drive/MyDrive/VAE-DECODER/transparent_vae.pt",
                        help="Path to the trained VAE or decoder model")
    parser.add_argument("--use_decoder_only", action="store_true",
                        help="Use only the decoder for generation")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="A beautiful flower on a transparent background",
                        help="Text prompt for image generation")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Path to a file containing prompts (one per line)")
    parser.add_argument("--height", type=int, default=512,
                        help="Height of the generated image")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of the generated image")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Guidance scale for generation")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images to generate in parallel")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/VAE-DECODER/OUT",
                        help="Directory to save generated images")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")

def setup_test_parser(subparsers):
    """
    Set up the argument parser for the test command.
    
    Args:
        subparsers: Subparsers object from argparse
    """
    parser = subparsers.add_parser('test', help='Test transparent PNG training and generation pipeline')
    
    # Test selection
    parser.add_argument("--test_dataset", action="store_true", help="Test dataset loading and visualization")
    parser.add_argument("--test_vae_modification", action="store_true", help="Test VAE modification")
    parser.add_argument("--test_vae_forward_pass", action="store_true", help="Test VAE forward pass")
    parser.add_argument("--test_model_saving_loading", action="store_true", help="Test model saving and loading")
    parser.add_argument("--test_image_generation", action="store_true", help="Test image generation")
    parser.add_argument("--run_all", action="store_true", help="Run all tests")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/SD-Data/TrainData/4000_PNG/TEST",
                        help="Directory containing transparent PNG images")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/flux_png_project/test_results",
                        help="Directory to save test results")

def main():
    """
    Main function to run the transparent PNG training and generation pipeline.
    """
    parser = argparse.ArgumentParser(description="Transparent PNG Training and Generation Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Set up subparsers
    setup_train_parser(subparsers)
    setup_save_parser(subparsers)
    setup_generate_parser(subparsers)
    setup_test_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if args.command == 'train':
        logger.info("Running training pipeline")
        train_main(args)
    elif args.command == 'save':
        logger.info("Running model saving")
        save_main(args)
    elif args.command == 'generate':
        logger.info("Running image generation")
        generate_main(args)
    elif args.command == 'test':
        logger.info("Running tests")
        run_all_tests(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
