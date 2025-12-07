"""
Training a Latent Diffusion Model on CIFAR-10 Dataset
Uses diffusers library with corrected architecture
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

# ============================================================================
# STEP 1: Setup and Configuration
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters
batch_size = 64
num_epochs = 100
learning_rate = 1e-4
image_size = 256  # Use 256x256 for better VAE compatibility
latent_channels = 4

# ============================================================================
# STEP 2: Load and Prepare Dataset
# ============================================================================

# Download CelebA or use CIFAR-10 with resizing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Using CIFAR-10 but resized to 256x256
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

print(f"Dataset loaded: {len(train_dataset)} images")
print(f"Batches per epoch: {len(train_dataloader)}")

# ============================================================================
# STEP 3: Initialize Model Components & Debug
# ============================================================================

# Load pre-trained VAE
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float32
).to(device)
vae.eval()
for param in vae.parameters():
    param.requires_grad = False

# DEBUG: Check actual latent dimensions
print("\n=== DEBUGGING LATENT DIMENSIONS ===")
with torch.no_grad():
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    latent_dist = vae.encode(dummy_input).latent_dist
    latent_sample = latent_dist.sample()
    print(f"Input shape: {dummy_input.shape}")
    print(f"Latent shape: {latent_sample.shape}")
    latent_height = latent_sample.shape[-2]
    latent_width = latent_sample.shape[-1]
    print(f"Latent spatial dims: {latent_height}x{latent_width}")
    print(f"Latent channels: {latent_sample.shape[1]}")

# Create UNet with correct dimensions
# For Stable Diffusion VAE: 256x256 images -> 32x32 latents
unet = UNet2DModel(
    sample_size=latent_height,  # Actual latent spatial dimension
    in_channels=latent_channels,
    out_channels=latent_channels,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),  # Simpler architecture
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    add_attention=False,  # Disable attention to avoid complexity
).to(device)

print(f"\nUNet initialized with sample_size={latent_height}")
print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()) / 1e6:.2f}M")

# Noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear",
    beta_start=0.00085,
    beta_end=0.012,
)

# Optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# Learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * num_epochs,
)

# ============================================================================
# STEP 4: Training Loop
# ============================================================================

unet.train()
print("\n=== STARTING TRAINING ===\n")

for epoch in range(num_epochs):
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
    loss_sum = 0
    
    for batch_idx, (images, _) in enumerate(train_dataloader):
        images = images.to(device)
        
        try:
            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()
            
            # Add noise (forward diffusion)
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            model_output = unet(noisy_latents, timesteps, return_dict=False)[0]
            
            # Compute loss
            loss = F.mse_loss(model_output, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            loss_sum += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        except RuntimeError as e:
            print(f"\nError in batch {batch_idx}: {str(e)}")
            print(f"Latent shape: {latents.shape}")
            print(f"Noisy latents shape: {noisy_latents.shape}")
            raise
    
    avg_loss = loss_sum / len(train_dataloader)
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}\n")
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoint_epoch_{epoch+1}.pt')
        print(f"Checkpoint saved at epoch {epoch+1}\n")

# ============================================================================
# STEP 5: Inference
# ============================================================================

def generate_images(num_images=4):
    unet.eval()
    
    with torch.no_grad():
        latents = torch.randn(
            (num_images, latent_channels, latent_height, latent_width),
            device=device
        )
        
        for t in tqdm(noise_scheduler.timesteps):
            timestep = torch.tensor([t] * num_images, device=device).long()
            model_output = unet(latents, timestep, return_dict=False)[0]
            latents = noise_scheduler.step(model_output, t, latents, return_dict=False)[0]
        
        images = vae.decode(latents / 0.18215).sample
        images = (images / 2 + 0.5).clamp(0, 1)
    
    return images

print("Generating sample images...")
generated = generate_images(num_images=4)

# Save and display generated images
import torchvision.utils as vutils
import os

output_dir = 'output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

vutils.save_image(generated, f'{output_dir}generated_images.png', nrow=2)
print("Generated images saved to 'generated_images.png'")