#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for saving trained VAE and Decoder models for transparent PNG generation.
This script provides functionality to save and load VAE and Decoder models.
"""

import os
import torch
import logging
import argparse
from train_transparent_png import TransparentVAE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_vae(model, path):
    """
    Save the complete VAE model.
    
    Args:
        model (TransparentVAE): Model to save
        path (str): Path to save model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"VAE model saved to {path}")

def save_decoder(model, path):
    """
    Save only the decoder part of the VAE model.
    
    Args:
        model (TransparentVAE): Model containing the decoder
        path (str): Path to save decoder
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Create a state dict with only decoder parameters
    decoder_state = {}
    full_state = model.state_dict()
    
    for key in full_state:
        if key.startswith('decoder'):
            decoder_state[key] = full_state[key]
    
    torch.save(decoder_state, path)
    logger.info(f"Decoder saved to {path}")

def save_vae_and_decoder(checkpoint_path, vae_path, decoder_path, device="cuda"):
    """
    Load a checkpoint and save the VAE and decoder separately.
    
    Args:
        checkpoint_path (str): Path to the checkpoint
        vae_path (str): Path to save the VAE
        decoder_path (str): Path to save the decoder
        device (str): Device to load the model on
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return False
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = TransparentVAE()
    model.to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Save VAE and decoder
    save_vae(model, vae_path)
    save_decoder(model, decoder_path)
    
    return True

def load_vae(path, device="cuda"):
    """
    Load a saved VAE model.
    
    Args:
        path (str): Path to the saved model
        device (str): Device to load the model on
        
    Returns:
        TransparentVAE: Loaded model
    """
    # Check if model exists
    if not os.path.exists(path):
        logger.error(f"Model not found at {path}")
        return None
    
    # Create model
    model = TransparentVAE()
    model.to(device)
    
    # Load state dict
    logger.info(f"Loading VAE from {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    
    return model

def load_decoder(path, device="cuda"):
    """
    Load a saved decoder model.
    
    Args:
        path (str): Path to the saved decoder
        device (str): Device to load the model on
        
    Returns:
        nn.Module: Loaded decoder
    """
    # Check if decoder exists
    if not os.path.exists(path):
        logger.error(f"Decoder not found at {path}")
        return None
    
    # Create model to get decoder structure
    model = TransparentVAE()
    
    # Load state dict
    logger.info(f"Loading decoder from {path}")
    decoder_state = torch.load(path, map_location=device)
    
    # Create a new state dict for the full model
    model_state = model.state_dict()
    
    # Update only decoder parameters
    for key in decoder_state:
        if key in model_state:
            model_state[key] = decoder_state[key]
    
    # Load the updated state dict
    model.load_state_dict(model_state)
    
    # Return only the decoder
    decoder = model.decoder.to(device)
    
    return decoder

def verify_saved_models(vae_path, decoder_path, device="cuda"):
    """
    Verify that saved models can be loaded correctly.
    
    Args:
        vae_path (str): Path to the saved VAE
        decoder_path (str): Path to the saved decoder
        device (str): Device to load the models on
        
    Returns:
        bool: True if verification is successful
    """
    try:
        # Try loading VAE
        vae = load_vae(vae_path, device)
        if vae is None:
            return False
        
        # Try loading decoder
        decoder = load_decoder(decoder_path, device)
        if decoder is None:
            return False
        
        logger.info("Model verification successful!")
        return True
    
    except Exception as e:
        logger.error(f"Error during model verification: {e}")
        return False

def main(args):
    """
    Main function to save VAE and decoder from a checkpoint.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine paths
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Find the latest checkpoint if not specified
        checkpoint_dir = args.checkpoint_dir
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
        if not checkpoints:
            logger.error(f"No checkpoints found in {checkpoint_dir}")
            return
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
    
    vae_path = os.path.join(args.output_dir, args.vae_filename)
    decoder_path = os.path.join(args.output_dir, args.decoder_filename)
    
    # Save VAE and decoder
    success = save_vae_and_decoder(checkpoint_path, vae_path, decoder_path, device)
    
    if success and args.verify:
        # Verify saved models
        verify_saved_models(vae_path, decoder_path, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save VAE and Decoder models")
    
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
    
    args = parser.parse_args()
    main(args)
