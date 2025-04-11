#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for validating the transparent PNG training and generation pipeline.
This script tests the functionality of the training, saving, and generation modules.
"""

import os
import torch
import argparse
import logging
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from train_transparent_png import TransparentVAE, TransparentPNGDataset
from save_vae_decoder import load_vae, load_decoder, verify_saved_models
from generate_transparent_png import generate_transparent_image, generate_transparent_image_with_decoder
from diffusers import FluxPipeline
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dataset(data_dir, output_dir, num_samples=5):
    """
    Test the TransparentPNGDataset by loading and visualizing samples.
    
    Args:
        data_dir (str): Directory containing transparent PNG images
        output_dir (str): Directory to save visualizations
        num_samples (int, optional): Number of samples to visualize
    """
    logger.info(f"Testing dataset with data from {data_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = TransparentPNGDataset(data_dir)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Visualize samples
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        
        # Get image
        image = batch['image'][0]  # Remove batch dimension
        
        # Convert to numpy for visualization
        image_np = image.numpy()
        
        # Scale from [-1, 1] to [0, 1]
        image_np = (image_np + 1) / 2
        
        # Clip to valid range
        image_np = np.clip(image_np, 0, 1)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # RGB channels
        axes[0].imshow(np.transpose(image_np[:3], (1, 2, 0)))
        axes[0].set_title("RGB Channels")
        axes[0].axis('off')
        
        # Alpha channel
        axes[1].imshow(image_np[3], cmap='gray')
        axes[1].set_title("Alpha Channel")
        axes[1].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dataset_sample_{i}.png"))
        plt.close()
        
        # Also save as RGBA PNG
        rgba = np.transpose(image_np, (1, 2, 0)) * 255
        rgba_pil = Image.fromarray(rgba.astype(np.uint8), mode='RGBA')
        rgba_pil.save(os.path.join(output_dir, f"dataset_sample_{i}_rgba.png"))
    
    logger.info(f"Dataset test completed. Visualizations saved to {output_dir}")

def test_vae_modification(output_dir):
    """
    Test the VAE modification by creating a TransparentVAE and checking its architecture.
    
    Args:
        output_dir (str): Directory to save test results
    """
    logger.info("Testing VAE modification")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create VAE
    vae = TransparentVAE()
    
    # Check encoder input channels
    encoder_in_channels = vae.encoder.conv_in.in_channels
    logger.info(f"Encoder input channels: {encoder_in_channels}")
    
    # Check decoder output channels
    decoder_out_channels = vae.decoder.conv_out.out_channels
    logger.info(f"Decoder output channels: {decoder_out_channels}")
    
    # Save architecture summary
    with open(os.path.join(output_dir, "vae_architecture.txt"), 'w') as f:
        f.write(f"Encoder input channels: {encoder_in_channels}\n")
        f.write(f"Decoder output channels: {decoder_out_channels}\n")
        f.write("\nEncoder architecture:\n")
        f.write(str(vae.encoder))
        f.write("\n\nDecoder architecture:\n")
        f.write(str(vae.decoder))
    
    logger.info(f"VAE modification test completed. Results saved to {output_dir}")

def test_vae_forward_pass(output_dir):
    """
    Test the VAE forward pass by encoding and decoding a random tensor.
    
    Args:
        output_dir (str): Directory to save test results
    """
    logger.info("Testing VAE forward pass")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create VAE
    vae = TransparentVAE()
    
    # Create random input
    batch_size = 1
    channels = 4
    height = 64
    width = 64
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    try:
        recon, mean, logvar = vae(x)
        
        # Check shapes
        logger.info(f"Input shape: {x.shape}")
        logger.info(f"Reconstructed shape: {recon.shape}")
        logger.info(f"Mean shape: {mean.shape}")
        logger.info(f"LogVar shape: {logvar.shape}")
        
        # Save results
        with open(os.path.join(output_dir, "vae_forward_pass.txt"), 'w') as f:
            f.write(f"Input shape: {x.shape}\n")
            f.write(f"Reconstructed shape: {recon.shape}\n")
            f.write(f"Mean shape: {mean.shape}\n")
            f.write(f"LogVar shape: {logvar.shape}\n")
        
        logger.info("VAE forward pass test passed")
    except Exception as e:
        logger.error(f"VAE forward pass test failed: {e}")
        with open(os.path.join(output_dir, "vae_forward_pass.txt"), 'w') as f:
            f.write(f"VAE forward pass test failed: {e}\n")

def test_model_saving_loading(output_dir):
    """
    Test saving and loading the VAE and decoder models.
    
    Args:
        output_dir (str): Directory to save test results
    """
    logger.info("Testing model saving and loading")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create VAE
    vae = TransparentVAE()
    
    # Save models
    vae_path = os.path.join(output_dir, "test_vae.pt")
    decoder_path = os.path.join(output_dir, "test_decoder.pt")
    
    torch.save(vae.state_dict(), vae_path)
    
    # Create decoder state dict
    decoder_state = {}
    full_state = vae.state_dict()
    for key in full_state:
        if key.startswith('decoder'):
            decoder_state[key] = full_state[key]
    
    torch.save(decoder_state, decoder_path)
    
    # Verify models
    try:
        verification = verify_saved_models(vae_path, decoder_path, "cpu")
        
        if verification:
            logger.info("Model saving and loading test passed")
        else:
            logger.error("Model verification failed")
        
        # Save results
        with open(os.path.join(output_dir, "model_saving_loading.txt"), 'w') as f:
            f.write(f"VAE saved to: {vae_path}\n")
            f.write(f"Decoder saved to: {decoder_path}\n")
            f.write(f"Verification result: {verification}\n")
    
    except Exception as e:
        logger.error(f"Model saving and loading test failed: {e}")
        with open(os.path.join(output_dir, "model_saving_loading.txt"), 'w') as f:
            f.write(f"Model saving and loading test failed: {e}\n")

def test_image_generation(output_dir, use_decoder_only=False):
    """
    Test image generation using a mock VAE/decoder.
    
    Args:
        output_dir (str): Directory to save test results
        use_decoder_only (bool, optional): Whether to test decoder-only generation
    """
    logger.info(f"Testing image generation (decoder_only={use_decoder_only})")
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip if CUDA is not available (this test requires GPU)
    if not torch.cuda.is_available():
        logger.warning("Skipping image generation test as CUDA is not available")
        with open(os.path.join(output_dir, "image_generation.txt"), 'w') as f:
            f.write("Skipped image generation test as CUDA is not available\n")
        return
    
    try:
        # Create VAE
        vae = TransparentVAE()
        
        # Initialize Flux pipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        
        # Generate image
        prompt = "A simple test image on a transparent background"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_decoder_only:
            output_path = os.path.join(output_dir, "generated_image_decoder_only.png")
            generate_transparent_image_with_decoder(
                vae.decoder, prompt, pipe, device, output_path,
                height=256, width=256, num_inference_steps=10
            )
        else:
            output_path = os.path.join(output_dir, "generated_image_vae.png")
            generate_transparent_image(
                vae, prompt, pipe, device, output_path,
                height=256, width=256, num_inference_steps=10
            )
        
        logger.info(f"Image generation test completed. Image saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Image generation test failed: {e}")
        with open(os.path.join(output_dir, "image_generation.txt"), 'w') as f:
            f.write(f"Image generation test failed: {e}\n")

def run_all_tests(args):
    """
    Run all tests.
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run tests
    if args.test_dataset and os.path.exists(args.data_dir):
        test_dataset(args.data_dir, os.path.join(args.output_dir, "dataset_test"))
    
    if args.test_vae_modification:
        test_vae_modification(os.path.join(args.output_dir, "vae_modification_test"))
    
    if args.test_vae_forward_pass:
        test_vae_forward_pass(os.path.join(args.output_dir, "vae_forward_pass_test"))
    
    if args.test_model_saving_loading:
        test_model_saving_loading(os.path.join(args.output_dir, "model_saving_loading_test"))
    
    if args.test_image_generation:
        test_image_generation(os.path.join(args.output_dir, "image_generation_test"))
        test_image_generation(os.path.join(args.output_dir, "image_generation_test"), use_decoder_only=True)
    
    logger.info("All tests completed")

def main():
    parser = argparse.ArgumentParser(description="Test transparent PNG training and generation pipeline")
    
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
    
    args = parser.parse_args()
    
    # If no specific test is selected, run all tests
    if not any([args.test_dataset, args.test_vae_modification, args.test_vae_forward_pass,
                args.test_model_saving_loading, args.test_image_generation]) or args.run_all:
        args.test_dataset = True
        args.test_vae_modification = True
        args.test_vae_forward_pass = True
        args.test_model_saving_loading = True
        args.test_image_generation = True
    
    run_all_tests(args)

if __name__ == "__main__":
    main()
