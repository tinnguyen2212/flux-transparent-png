#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ComfyUI nodes for loading and using trained VAE and Decoder models for transparent PNG generation.
This script provides custom nodes for ComfyUI to integrate with the trained models.
"""

import os
import torch
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management as model_management
from comfy.sd import VAE

# Add model paths to ComfyUI's model paths
TRANSPARENT_VAE_PATH = os.path.join(folder_paths.models_dir, "transparent_vae")
os.makedirs(TRANSPARENT_VAE_PATH, exist_ok=True)
folder_paths.add_model_folder_path("transparent_vae", TRANSPARENT_VAE_PATH)

class TransparentVAE(torch.nn.Module):
    """VAE model modified to handle transparent PNG images."""
    
    def __init__(self, base_vae=None):
        """
        Initialize the transparent VAE.
        
        Args:
            base_vae (VAE, optional): Base VAE model to modify
        """
        super().__init__()
        
        # Store original components
        self.encoder = base_vae.encoder if base_vae else None
        self.decoder = base_vae.decoder if base_vae else None
        self.quant_conv = base_vae.quant_conv if base_vae else None
        self.post_quant_conv = base_vae.post_quant_conv if base_vae else None
    
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

class TransparentVAELoader:
    """ComfyUI node for loading a trained transparent VAE model."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("transparent_vae"), ),
            }
        }
    
    RETURN_TYPES = ("TRANSPARENT_VAE", )
    FUNCTION = "load_vae"
    CATEGORY = "Transparent PNG"
    
    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path("transparent_vae", vae_name)
        
        # Create base VAE
        base_vae = VAE()
        
        # Create transparent VAE
        vae = TransparentVAE(base_vae)
        
        # Load state dict
        vae.load_state_dict(torch.load(vae_path, map_location=model_management.get_torch_device()))
        
        # Move to appropriate device
        vae = model_management.load_model_gpu(vae)
        
        return (vae, )

class TransparentDecoderLoader:
    """ComfyUI node for loading a trained transparent decoder model."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder_name": (folder_paths.get_filename_list("transparent_vae"), ),
            }
        }
    
    RETURN_TYPES = ("TRANSPARENT_DECODER", )
    FUNCTION = "load_decoder"
    CATEGORY = "Transparent PNG"
    
    def load_decoder(self, decoder_name):
        decoder_path = folder_paths.get_full_path("transparent_vae", decoder_name)
        
        # Create base VAE
        base_vae = VAE()
        
        # Create transparent VAE
        vae = TransparentVAE(base_vae)
        
        # Load decoder state dict
        decoder_state = torch.load(decoder_path, map_location=model_management.get_torch_device())
        
        # Create a new state dict for the full model
        model_state = vae.state_dict()
        
        # Update only decoder parameters
        for key in decoder_state:
            if key in model_state:
                model_state[key] = decoder_state[key]
        
        # Load the updated state dict
        vae.load_state_dict(model_state)
        
        # Get only the decoder
        decoder = vae.decoder
        
        # Move to appropriate device
        decoder = model_management.load_model_gpu(decoder)
        
        return (decoder, )

class TransparentVAEEncode:
    """ComfyUI node for encoding images with a transparent VAE."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("TRANSPARENT_VAE", ),
                "image": ("IMAGE", ),
            }
        }
    
    RETURN_TYPES = ("LATENT", )
    FUNCTION = "encode"
    CATEGORY = "Transparent PNG"
    
    def encode(self, vae, image):
        # Convert image to RGBA if it's not already
        if image.shape[3] == 3:
            # Add alpha channel (fully opaque)
            alpha = torch.ones((image.shape[0], image.shape[1], image.shape[2], 1), device=image.device)
            image = torch.cat([image, alpha], dim=3)
        
        # Move to appropriate device
        vae = model_management.get_torch_device()
        image = image.to(vae.device)
        
        # Prepare image for VAE
        image = image.permute(0, 3, 1, 2)  # NHWC -> NCHW
        image = image * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        
        # Encode
        with torch.no_grad():
            mean, logvar = vae.encode(image)
            std = torch.exp(0.5 * logvar)
            latent = mean + std * torch.randn_like(std)
        
        return ({"samples": latent}, )

class TransparentVAEDecode:
    """ComfyUI node for decoding latents with a transparent VAE."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("TRANSPARENT_VAE", ),
                "latent": ("LATENT", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "decode"
    CATEGORY = "Transparent PNG"
    
    def decode(self, vae, latent):
        # Get latent samples
        latent_samples = latent["samples"]
        
        # Move to appropriate device
        vae = model_management.get_torch_device()
        latent_samples = latent_samples.to(vae.device)
        
        # Decode
        with torch.no_grad():
            image = vae.decode(latent_samples)
        
        # Post-process image
        image = (image + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        image = image.clamp(0, 1)
        image = image.permute(0, 2, 3, 1)  # NCHW -> NHWC
        
        return (image, )

class TransparentDecoderDecode:
    """ComfyUI node for decoding latents with a transparent decoder."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder": ("TRANSPARENT_DECODER", ),
                "latent": ("LATENT", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "decode"
    CATEGORY = "Transparent PNG"
    
    def decode(self, decoder, latent):
        # Get latent samples
        latent_samples = latent["samples"]
        
        # Move to appropriate device
        decoder = model_management.get_torch_device()
        latent_samples = latent_samples.to(decoder.device)
        
        # Create a temporary VAE to get the post_quant_conv
        base_vae = VAE()
        vae = TransparentVAE(base_vae)
        post_quant_conv = vae.post_quant_conv.to(decoder.device)
        
        # Process latent
        with torch.no_grad():
            processed_latent = post_quant_conv(latent_samples)
            image = decoder(processed_latent)
        
        # Post-process image
        image = (image + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        image = image.clamp(0, 1)
        image = image.permute(0, 2, 3, 1)  # NCHW -> NHWC
        
        return (image, )

class SaveTransparentPNG:
    """ComfyUI node for saving images as transparent PNGs."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "transparent"}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "/content/drive/MyDrive/VAE-DECODER/OUT"}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_png"
    OUTPUT_NODE = True
    CATEGORY = "Transparent PNG"
    
    def save_png(self, image, filename_prefix, output_dir=None):
        # Use ComfyUI's output directory if not specified
        if output_dir is None or output_dir == "":
            output_dir = folder_paths.get_output_directory()
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert to numpy array
        image_np = image.cpu().numpy()
        
        # Ensure image is in [0, 1] range
        image_np = np.clip(image_np, 0, 1)
        
        # Convert to uint8
        image_np = (image_np * 255).astype(np.uint8)
        
        # Save each image in the batch
        results = []
        for i, img in enumerate(image_np):
            # Create PIL image
            if img.shape[2] == 4:
                pil_image = Image.fromarray(img, mode='RGBA')
            else:
                # If image doesn't have alpha channel, add one (fully opaque)
                rgb_img = Image.fromarray(img, mode='RGB')
                pil_image = rgb_img.convert('RGBA')
            
            # Create filename
            filename = f"{filename_prefix}_{i:05}.png"
            save_path = os.path.join(output_dir, filename)
            
            # Save image
            pil_image.save(save_path)
            results.append(save_path)
        
        return {"ui": {"images": results}}

class TransparentPNGGenerate:
    """ComfyUI node for generating transparent PNG images from prompts."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("TRANSPARENT_VAE", ),
                "model": ("MODEL", ),
                "prompt": ("STRING", {"default": "A beautiful flower on a transparent background"}),
                "negative_prompt": ("STRING", {"default": "background, ugly, blurry"}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "generate"
    CATEGORY = "Transparent PNG"
    
    def generate(self, vae, model, prompt, negative_prompt, height, width, guidance_scale, num_inference_steps, seed):
        # Import here to avoid circular imports
        import comfy.samplers
        import comfy.sample
        import comfy.sd
        
        # Set up generator for reproducibility
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        # Create conditioning
        cond = comfy.sd.prompt_to_cond(model, [prompt])
        uncond = comfy.sd.prompt_to_cond(model, [negative_prompt])
        
        # Set up sampler
        sampler = comfy.samplers.KSampler(model, steps=num_inference_steps, device=model_management.get_torch_device())
        
        # Generate latent
        latent = comfy.sample.sample(
            model, sampler, cond, uncond, 
            width // 8, height // 8, 
            cfg_scale=guidance_scale, 
            seed=seed, 
            sampler_name="euler_ancestral"
        )
        
        # Decode latent to RGBA image using our trained VAE
        with torch.no_grad():
            image = vae.decode(latent)
            
            # Post-process image
            image = (image + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            image = image.clamp(0, 1)
            image = image.permute(0, 2, 3, 1)  # NCHW -> NHWC
        
        return (image, {"samples": latent})

# Register nodes
NODE_CLASS_MAPPINGS = {
    "TransparentVAELoader": TransparentVAELoader,
    "TransparentDecoderLoader": TransparentDecoderLoader,
    "TransparentVAEEncode": TransparentVAEEncode,
    "TransparentVAEDecode": TransparentVAEDecode,
    "TransparentDecoderDecode": TransparentDecoderDecode,
    "SaveTransparentPNG": SaveTransparentPNG,
    "TransparentPNGGenerate": TransparentPNGGenerate,
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "TransparentVAELoader": "Load Transparent VAE",
    "TransparentDecoderLoader": "Load Transparent Decoder",
    "TransparentVAEEncode": "Encode with Transparent VAE",
    "TransparentVAEDecode": "Decode with Transparent VAE",
    "TransparentDecoderDecode": "Decode with Transparent Decoder",
    "SaveTransparentPNG": "Save Transparent PNG",
    "TransparentPNGGenerate": "Generate Transparent PNG",
}
