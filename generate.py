"""
Generate images using a trained Latent Diffusion Model.
Loads a checkpoint from the training process and runs the reverse diffusion process.
"""

import torch
import os
import argparse
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm
import torchvision.utils as vutils

# ============================================================================
# STEP 1: Main Function with Configuration
# ============================================================================

def main():
    # --- CONFIGURATION ---
    # Manually set parameters here instead of using the command line.
    checkpoint_path = "checkpoint_epoch_20.pt"  # REQUIRED: Update this path to your model checkpoint.
    num_images = 4
    output_dir = "output"
    image_size = 256  # Must match the image_size used during training.
    # ---------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Derived from image_size and VAE downsampling
    latent_height = image_size // 8
    latent_width = image_size // 8
    latent_channels = 4

    # ============================================================================
    # STEP 2: Load Models and Scheduler
    # ============================================================================

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    # Initialize UNet with the same architecture as in training
    unet = UNet2DModel(
        sample_size=latent_height,
        in_channels=latent_channels,
        out_channels=latent_channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        add_attention=False,
    ).to(device)

    # Load the trained UNet weights from the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    print("UNet model loaded successfully.")

    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler.from_config({
        "num_train_timesteps": 1000,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "beta_end": 0.012,
    })

    # ============================================================================
    # STEP 3: Generation (Inference)
    # ============================================================================
    print(f"\nGenerating {num_images} images...")
    with torch.no_grad():
        # 1. Start with random noise in the latent space
        latents = torch.randn((num_images, latent_channels, latent_height, latent_width), device=device)

        # 2. Denoise the latents over a series of timesteps
        for t in tqdm(noise_scheduler.timesteps):
            model_output = unet(latents, t, return_dict=False)[0]
            latents = noise_scheduler.step(model_output, t, latents, return_dict=False)[0]

        # 3. Decode the final latents back to image space
        images = vae.decode(latents / 0.18215).sample
        images = (images / 2 + 0.5).clamp(0, 1) # Rescale to [0, 1]

    # ============================================================================
    # STEP 4: Save Images
    # ============================================================================
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_from_checkpoint.png")
    vutils.save_image(images, output_path, nrow=int(num_images**0.5))
    print(f"Generated images saved to '{output_path}'")

if __name__ == "__main__":
    main()