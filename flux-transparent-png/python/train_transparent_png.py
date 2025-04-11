#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training module for transparent PNG images using Flux.1 dev.
This script implements the training pipeline for transparent PNG images without backgrounds.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from diffusers import FluxPipeline, FluxVAE
import argparse
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransparentPNGDataset(Dataset):
    """Dataset for transparent PNG images."""
    
    def __init__(self, data_dir, transform=None, image_size=512):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing transparent PNG images
            transform (callable, optional): Optional transform to be applied on a sample
            image_size (int, optional): Size to resize images to
        """
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.lower().endswith('.png')]
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2 - 1)  # Scale to [-1, 1]
            ])
        else:
            self.transform = transform
            
        logger.info(f"Found {len(self.image_paths)} PNG images in {data_dir}")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load image with alpha channel
            image = Image.open(image_path).convert('RGBA')
            
            # Apply transformations
            if self.transform:
                # Handle RGBA images with transform
                r, g, b, a = image.split()
                rgb = Image.merge("RGB", (r, g, b))
                rgb_tensor = self.transform(rgb)
                
                # Transform alpha separately and combine
                a_transform = transforms.Compose([
                    transforms.Resize((rgb_tensor.shape[1], rgb_tensor.shape[2])),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * 2 - 1)  # Scale to [-1, 1]
                ])
                alpha_tensor = a_transform(a)
                
                # Combine RGB and alpha
                rgba_tensor = torch.cat([rgb_tensor, alpha_tensor], dim=0)
            else:
                # Convert PIL image to tensor manually
                rgba_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 127.5 - 1
            
            return {
                'image': rgba_tensor,
                'path': image_path
            }
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a placeholder if image loading fails
            return {
                'image': torch.zeros(4, 512, 512),
                'path': image_path
            }

class TransparentVAE(nn.Module):
    """VAE model modified to handle transparent PNG images."""
    
    def __init__(self, base_vae=None, pretrained_model_name="black-forest-labs/FLUX.1-dev"):
        """
        Initialize the transparent VAE.
        
        Args:
            base_vae (FluxVAE, optional): Base VAE model to modify
            pretrained_model_name (str, optional): Name of pretrained model to load if base_vae is None
        """
        super().__init__()
        
        # Load base VAE if not provided
        if base_vae is None:
            logger.info(f"Loading base VAE from {pretrained_model_name}")
            base_vae = FluxVAE.from_pretrained(pretrained_model_name)
        
        # Store original components
        self.encoder = base_vae.encoder
        self.decoder = base_vae.decoder
        self.quant_conv = base_vae.quant_conv
        self.post_quant_conv = base_vae.post_quant_conv
        
        # Modify first layer of encoder to accept 4 channels (RGBA)
        in_channels = self.encoder.conv_in.in_channels
        if in_channels != 4:
            logger.info(f"Modifying encoder input from {in_channels} to 4 channels")
            new_conv_in = nn.Conv2d(
                4, 
                self.encoder.conv_in.out_channels, 
                kernel_size=self.encoder.conv_in.kernel_size,
                stride=self.encoder.conv_in.stride,
                padding=self.encoder.conv_in.padding
            )
            
            # Initialize new weights with pretrained weights
            with torch.no_grad():
                new_conv_in.weight[:, :3] = self.encoder.conv_in.weight
                # Initialize alpha channel weights with average of RGB weights
                new_conv_in.weight[:, 3:] = torch.mean(self.encoder.conv_in.weight, dim=1, keepdim=True)
                new_conv_in.bias = nn.Parameter(self.encoder.conv_in.bias.clone())
            
            self.encoder.conv_in = new_conv_in
            
        # Modify last layer of decoder to output 4 channels (RGBA)
        out_channels = self.decoder.conv_out.out_channels
        if out_channels != 4:
            logger.info(f"Modifying decoder output from {out_channels} to 4 channels")
            new_conv_out = nn.Conv2d(
                self.decoder.conv_out.in_channels, 
                4,
                kernel_size=self.decoder.conv_out.kernel_size,
                stride=self.decoder.conv_out.stride,
                padding=self.decoder.conv_out.padding
            )
            
            # Initialize new weights with pretrained weights
            with torch.no_grad():
                # Copy RGB weights
                new_conv_out.weight[:3] = self.decoder.conv_out.weight
                # Initialize alpha channel weights with average of RGB weights
                new_conv_out.weight[3:] = torch.mean(self.decoder.conv_out.weight, dim=0, keepdim=True)
                # Copy RGB bias
                new_conv_out.bias[:3] = nn.Parameter(self.decoder.conv_out.bias.clone())
                # Initialize alpha bias to 0
                new_conv_out.bias[3:] = 0.0
            
            self.decoder.conv_out = new_conv_out
    
    def encode(self, x):
        """
        Encode input images to latent space.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 4, H, W]
            
        Returns:
            tuple: Mean and log variance of latent distribution
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return torch.chunk(moments, 2, dim=1)
    
    def decode(self, z):
        """
        Decode latent representation to RGBA image.
        
        Args:
            z (torch.Tensor): Latent representation
            
        Returns:
            torch.Tensor: Reconstructed RGBA image
        """
        z = self.post_quant_conv(z)
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 4, H, W]
            
        Returns:
            tuple: Reconstructed image, mean, and log variance
        """
        mean, logvar = self.encode(x)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        # Decode
        x_recon = self.decode(z)
        
        return x_recon, mean, logvar

def vae_loss_function(recon_x, x, mean, logvar, alpha_weight=2.0):
    """
    VAE loss function with weighted alpha channel.
    
    Args:
        recon_x (torch.Tensor): Reconstructed image
        x (torch.Tensor): Original image
        mean (torch.Tensor): Mean of latent distribution
        logvar (torch.Tensor): Log variance of latent distribution
        alpha_weight (float, optional): Weight for alpha channel loss
        
    Returns:
        torch.Tensor: Total loss
    """
    # Split into RGB and alpha components
    rgb_original = x[:, :3]
    alpha_original = x[:, 3:4]
    rgb_recon = recon_x[:, :3]
    alpha_recon = recon_x[:, 3:4]
    
    # Compute reconstruction losses
    rgb_loss = F.mse_loss(rgb_recon, rgb_original, reduction='sum')
    alpha_loss = F.mse_loss(alpha_recon, alpha_original, reduction='sum')
    
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # Combined loss with higher weight for alpha to preserve transparency
    return rgb_loss + alpha_weight * alpha_loss + kld

def train_epoch(model, dataloader, optimizer, device, alpha_weight=2.0):
    """
    Train the model for one epoch.
    
    Args:
        model (TransparentVAE): Model to train
        dataloader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        alpha_weight (float, optional): Weight for alpha channel loss
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Get RGBA image
        image = batch['image'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        recon_batch, mean, logvar = model(image)
        
        # Compute loss
        loss = vae_loss_function(recon_batch, image, mean, logvar, alpha_weight)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item() / len(image)})
    
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, device, alpha_weight=2.0):
    """
    Validate the model.
    
    Args:
        model (TransparentVAE): Model to validate
        dataloader (DataLoader): Validation data loader
        device (torch.device): Device to validate on
        alpha_weight (float, optional): Weight for alpha channel loss
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            image = batch['image'].to(device)
            recon_batch, mean, logvar = model(image)
            loss = vae_loss_function(recon_batch, image, mean, logvar, alpha_weight)
            total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save a training checkpoint.
    
    Args:
        model (TransparentVAE): Model to save
        optimizer (torch.optim.Optimizer): Optimizer
        epoch (int): Current epoch
        loss (float): Current loss
        path (str): Path to save checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    
    logger.info(f"Checkpoint saved to {path}")

def save_model(model, path):
    """
    Save the trained model.
    
    Args:
        model (TransparentVAE): Model to save
        path (str): Path to save model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def save_decoder(model, path):
    """
    Save only the decoder part of the model.
    
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

def visualize_reconstructions(model, dataloader, device, output_dir, num_samples=5):
    """
    Visualize original and reconstructed images.
    
    Args:
        model (TransparentVAE): Trained model
        dataloader (DataLoader): Data loader
        device (torch.device): Device to run on
        output_dir (str): Directory to save visualizations
        num_samples (int, optional): Number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Get samples
    samples = []
    with torch.no_grad():
        for batch in dataloader:
            samples.append(batch)
            if len(samples) >= num_samples:
                break
    
    # Generate reconstructions
    for i, batch in enumerate(samples):
        if i >= num_samples:
            break
            
        # Get original image
        original = batch['image'].to(device)
        
        # Reconstruct
        with torch.no_grad():
            recon, _, _ = model(original)
        
        # Convert to numpy for visualization
        original_np = original.cpu().numpy()
        recon_np = recon.cpu().numpy()
        
        # Scale from [-1, 1] to [0, 1]
        original_np = (original_np + 1) / 2
        recon_np = (recon_np + 1) / 2
        
        # Clip to valid range
        original_np = np.clip(original_np, 0, 1)
        recon_np = np.clip(recon_np, 0, 1)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original RGB
        axes[0, 0].imshow(np.transpose(original_np[0, :3], (1, 2, 0)))
        axes[0, 0].set_title("Original RGB")
        axes[0, 0].axis('off')
        
        # Original Alpha
        axes[0, 1].imshow(original_np[0, 3], cmap='gray')
        axes[0, 1].set_title("Original Alpha")
        axes[0, 1].axis('off')
        
        # Reconstructed RGB
        axes[1, 0].imshow(np.transpose(recon_np[0, :3], (1, 2, 0)))
        axes[1, 0].set_title("Reconstructed RGB")
        axes[1, 0].axis('off')
        
        # Reconstructed Alpha
        axes[1, 1].imshow(recon_np[0, 3], cmap='gray')
        axes[1, 1].set_title("Reconstructed Alpha")
        axes[1, 1].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"reconstruction_{i}.png"))
        plt.close()
        
        # Also save as RGBA PNG
        original_rgba = np.transpose(original_np[0], (1, 2, 0)) * 255
        recon_rgba = np.transpose(recon_np[0], (1, 2, 0)) * 255
        
        original_pil = Image.fromarray(original_rgba.astype(np.uint8), mode='RGBA')
        recon_pil = Image.fromarray(recon_rgba.astype(np.uint8), mode='RGBA')
        
        original_pil.save(os.path.join(output_dir, f"original_{i}.png"))
        recon_pil.save(os.path.join(output_dir, f"reconstructed_{i}.png"))

def main(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # Scale to [-1, 1]
    ])
    
    dataset = TransparentPNGDataset(args.data_dir, transform=None, image_size=args.image_size)
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = TransparentVAE(pretrained_model_name=args.pretrained_model)
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.alpha_weight)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device, args.alpha_weight)
        val_losses.append(val_loss)
        
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "best_transparent_vae.pt")
            best_decoder_path = os.path.join(args.output_dir, "best_transparent_decoder.pt")
            save_model(model, best_model_path)
            save_decoder(model, best_decoder_path)
            logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
        
        # Visualize reconstructions periodically
        if (epoch + 1) % args.vis_frequency == 0:
            vis_dir = os.path.join(args.output_dir, f"visualizations_epoch_{epoch+1}")
            visualize_reconstructions(model, val_loader, device, vis_dir)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_transparent_vae.pt")
    final_decoder_path = os.path.join(args.output_dir, "final_transparent_decoder.pt")
    save_model(model, final_model_path)
    save_decoder(model, final_decoder_path)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE for transparent PNG images")
    
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
    
    args = parser.parse_args()
    main(args)
