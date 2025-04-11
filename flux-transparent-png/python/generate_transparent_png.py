#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for generating transparent PNG images using trained VAE and Decoder models.
This script provides functionality to generate transparent PNG images without backgrounds.
"""

import os
import torch
import numpy as np
from PIL import Image
import argparse
import logging
from diffusers import FluxPipeline
from train_transparent_png import TransparentVAE
from save_vae_decoder import load_vae, load_decoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_transparent_image(vae, prompt, pipe, device, output_path=None, height=512, width=512, 
                              guidance_scale=3.5, num_inference_steps=50, seed=None):
    """
    Generate a transparent PNG image using the trained VAE.
    
    Args:
        vae (TransparentVAE): Trained VAE model
        prompt (str): Text prompt for image generation
        pipe (FluxPipeline): Flux pipeline for latent generation
        device (torch.device): Device to run on
        output_path (str, optional): Path to save the generated image
        height (int, optional): Height of the generated image
        width (int, optional): Width of the generated image
        guidance_scale (float, optional): Guidance scale for generation
        num_inference_steps (int, optional): Number of inference steps
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        PIL.Image: Generated transparent PNG image
    """
    # Set model to evaluation mode
    vae.eval()
    
    # Set up generator for reproducibility
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    else:
        generator = None
    
    # Generate latent using Flux pipeline
    logger.info(f"Generating latent for prompt: '{prompt}'")
    latents = pipe(
        prompt=prompt,
        output_type="latent",
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    
    # Decode latents to RGBA image using our trained VAE
    logger.info("Decoding latent to RGBA image")
    with torch.no_grad():
        latents = torch.tensor(latents).unsqueeze(0).to(device)
        rgba_image = vae.decode(latents)
        
        # Scale from [-1, 1] to [0, 1]
        rgba_image = (rgba_image + 1) / 2
        rgba_image = rgba_image.clamp(0, 1)
        
        # Convert to numpy array
        rgba_image = rgba_image.cpu().permute(0, 2, 3, 1).numpy()[0] * 255
        rgba_image = rgba_image.astype(np.uint8)
    
    # Convert to PIL image
    pil_image = Image.fromarray(rgba_image, mode='RGBA')
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pil_image.save(output_path)
        logger.info(f"Image saved to {output_path}")
    
    return pil_image

def generate_transparent_image_with_decoder(decoder, prompt, pipe, device, output_path=None, height=512, width=512, 
                                           guidance_scale=3.5, num_inference_steps=50, seed=None):
    """
    Generate a transparent PNG image using only the trained decoder.
    
    Args:
        decoder (nn.Module): Trained decoder model
        prompt (str): Text prompt for image generation
        pipe (FluxPipeline): Flux pipeline for latent generation
        device (torch.device): Device to run on
        output_path (str, optional): Path to save the generated image
        height (int, optional): Height of the generated image
        width (int, optional): Width of the generated image
        guidance_scale (float, optional): Guidance scale for generation
        num_inference_steps (int, optional): Number of inference steps
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        PIL.Image: Generated transparent PNG image
    """
    # Set model to evaluation mode
    decoder.eval()
    
    # Set up generator for reproducibility
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    else:
        generator = None
    
    # Generate latent using Flux pipeline
    logger.info(f"Generating latent for prompt: '{prompt}'")
    latents = pipe(
        prompt=prompt,
        output_type="latent",
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    
    # Decode latents to RGBA image using our trained decoder
    logger.info("Decoding latent to RGBA image")
    with torch.no_grad():
        latents = torch.tensor(latents).unsqueeze(0).to(device)
        
        # For decoder-only, we need to process the latent through post_quant_conv
        # Create a temporary VAE to get the post_quant_conv
        temp_vae = TransparentVAE()
        post_quant_conv = temp_vae.post_quant_conv.to(device)
        
        # Process latent
        processed_latent = post_quant_conv(latents)
        
        # Decode
        rgba_image = decoder(processed_latent)
        
        # Scale from [-1, 1] to [0, 1]
        rgba_image = (rgba_image + 1) / 2
        rgba_image = rgba_image.clamp(0, 1)
        
        # Convert to numpy array
        rgba_image = rgba_image.cpu().permute(0, 2, 3, 1).numpy()[0] * 255
        rgba_image = rgba_image.astype(np.uint8)
    
    # Convert to PIL image
    pil_image = Image.fromarray(rgba_image, mode='RGBA')
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pil_image.save(output_path)
        logger.info(f"Image saved to {output_path}")
    
    return pil_image

def batch_generate_images(model_path, output_dir, prompts, use_decoder_only=False, 
                         device="cuda", batch_size=1, **generation_kwargs):
    """
    Generate multiple transparent PNG images.
    
    Args:
        model_path (str): Path to the trained VAE or decoder model
        output_dir (str): Directory to save generated images
        prompts (list): List of text prompts for image generation
        use_decoder_only (bool, optional): Whether to use only the decoder
        device (str, optional): Device to run on
        batch_size (int, optional): Number of images to generate in parallel
        **generation_kwargs: Additional arguments for image generation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    if use_decoder_only:
        logger.info(f"Loading decoder from {model_path}")
        model = load_decoder(model_path, device)
    else:
        logger.info(f"Loading VAE from {model_path}")
        model = load_vae(model_path, device)
    
    # Initialize Flux pipeline
    logger.info("Initializing Flux pipeline")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    
    # Generate images
    for i, prompt in enumerate(prompts):
        logger.info(f"Generating image {i+1}/{len(prompts)}")
        
        # Create output path
        output_filename = f"{i+1:04d}_{prompt[:20].replace(' ', '_')}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Generate image
        if use_decoder_only:
            generate_transparent_image_with_decoder(
                model, prompt, pipe, device, output_path, **generation_kwargs
            )
        else:
            generate_transparent_image(
                model, prompt, pipe, device, output_path, **generation_kwargs
            )

def main(args):
    """
    Main function to generate transparent PNG images.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [args.prompt]
    
    # Generate images
    batch_generate_images(
        model_path=args.model_path,
        output_dir=args.output_dir,
        prompts=prompts,
        use_decoder_only=args.use_decoder_only,
        device=args.device,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed
    )
    
    logger.info(f"Generated {len(prompts)} images in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate transparent PNG images")
    
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
    
    args = parser.parse_args()
    main(args)
