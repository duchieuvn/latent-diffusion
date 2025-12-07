import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import numpy as np

# ============================================================================
# STEP 1: Setup and Configuration
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
num_epochs = 100
learning_rate = 1e-4
image_size = 32
latent_channels = 4

# ============================================================================
# STEP 2: Load and Prepare Dataset
# ============================================================================

# Download CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

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
    num_workers=4,
    pin_memory=True
)

print(f"Dataset loaded: {len(train_dataset)} images")
print(f"Batches per epoch: {len(train_dataloader)}")

# ============================================================================
# STEP 3: Initialize Model Components
# ============================================================================

# Autoencoder for encoding images to latent space
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float32
).to(device)
vae.eval()  # VAE is frozen
for param in vae.parameters():
    param.requires_grad = False

# UNet for learning the diffusion process in latent space
unet = UNet2DModel(
    sample_size=image_size // 8,  # 32 / 8 = 4 (downsampled by VAE)
    in_channels=latent_channels,
    out_channels=latent_channels,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(device)

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

train_output_dir = 'train_output/'

unet.train()
for epoch in range(num_epochs):
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
    loss_sum = 0
    
    for batch_idx, (images, _) in enumerate(train_dataloader):
        images = images.to(device)
        
        # Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215  # Scaling factor for stability
        
        # Sample random timesteps for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device
        ).long()
        
        # Add noise to latents (forward diffusion process)
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        model_output = unet(noisy_latents, timesteps, return_dict=False)[0]
        
        # Compute loss (predicting noise)
        loss = F.mse_loss(model_output, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        
        loss_sum += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = loss_sum / len(train_dataloader)
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{train_output_dir}checkpoint_epoch_{epoch+1}.pt')
        print(f"Checkpoint saved at epoch {epoch+1}")

# ============================================================================
# STEP 5: Inference (Generate Images)
# ============================================================================

def generate_images(num_images=4):
    unet.eval()
    
    with torch.no_grad():
        # Start with random noise
        latents = torch.randn(
            (num_images, latent_channels, image_size // 8, image_size // 8),
            device=device
        )
        
        # Reverse diffusion process
        for t in tqdm(noise_scheduler.timesteps):
            timestep = torch.tensor([t] * num_images, device=device).long()
            model_output = unet(latents, timestep, return_dict=False)[0]
            latents = noise_scheduler.step(model_output, t, latents, return_dict=False)[0]
        
        # Decode from latent space to image space
        images = vae.decode(latents / 0.18215).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        
    return images

# Generate sample images
print("Generating sample images...")
generated = generate_images(num_images=4)
print(f"Generated images shape: {generated.shape}")

# Save and display generated images
import torchvision.utils as vutils
import os

output_dir = 'output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

vutils.save_image(generated, f'{output_dir}generated_images.png', nrow=2)
print("Generated images saved to 'generated_images.png'")