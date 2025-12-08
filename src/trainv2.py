import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import AutoencoderKL, DDPMScheduler
from model import SimpleLatentDiffusion
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

# Organized output directory structure
experiment_name = 'default_experiment'  # Change as needed
base_output_dir = os.path.join('..', 'output', experiment_name)
samples_dir = os.path.join(base_output_dir, 'samples')
checkpoints_dir = os.path.join(base_output_dir, 'checkpoints')
plots_dir = os.path.join(base_output_dir, 'plots')

for d in [base_output_dir, samples_dir, checkpoints_dir, plots_dir]:
    os.makedirs(d, exist_ok=True)

# Track losses for plotting
train_losses = []
val_losses = []
best_epoch = 0
best_val_loss = float('inf')
best_epoch = 0
# Early stopping parameters
patience = 10  # Number of epochs to wait for improvement
epochs_no_improve = 0

# ============================================================================
# STEP 1: Setup and Configuration
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters
batch_size = 16
num_epochs = 50
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


# Load full CIFAR-10 dataset
full_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Select a subset of 6000 samples
# subset_indices = list(range(6000))
subset_indices = list(range(12))
subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)

# Split into train (5000) and validation (1000)
train_dataset, val_dataset = torch.utils.data.random_split(
    subset_dataset,
    [10, 2],
    generator=torch.Generator().manual_seed(42)
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"Train dataset: {len(train_dataset)} images")
print(f"Validation dataset: {len(val_dataset)} images")
print(f"Train batches per epoch: {len(train_dataloader)}")
print(f"Validation batches per epoch: {len(val_dataloader)}")

# ============================================================================
# STEP 3: Initialize Model Components & Debug
# ============================================================================

latent_size = 32  # VAE reduces 256 to 32
latent_channels = 4  # VAE output 4 channels

model = SimpleLatentDiffusion(
    sample_size=latent_size,
    in_channels=latent_channels,
    out_channels=latent_channels,
    device=device
).to(device)

print(f"\nModel initialized with sample_size={latent_size}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear",
    beta_start=0.00085,
    beta_end=0.012,
)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * num_epochs,
)

# ============================================================================
# STEP 4: Training Loop
# ============================================================================
def generate_images(num_images=4):
    model.eval()
    with torch.no_grad():
        latents = torch.randn(
            (num_images, latent_channels, latent_size, latent_size),
            device=device
        )
        for t in tqdm(noise_scheduler.timesteps):
            timestep = torch.tensor([t] * num_images, device=device).long()
            model_output = model(latents, timestep, return_dict=False)[0]
            latents = noise_scheduler.step(model_output, t, latents, return_dict=False)[0]
        images = model.decode(latents)
    return images


model.train()
print("\n=== STARTING TRAINING ===\n")

for epoch in range(num_epochs):
    # ====== TRAINING PHASE ======
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    train_loss_sum = 0
    
    for batch_idx, (images, _) in enumerate(train_dataloader):
        images = images.to(device)
        
        try:
            # Encode images to latent space
            latents = model.encode(images)
            
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
            model_output = model(noisy_latents, timesteps, return_dict=False)[0]
            
            # Compute loss
            loss = F.mse_loss(model_output, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            train_loss_sum += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        except RuntimeError as e:
            print(f"\nError in batch {batch_idx}: {str(e)}")
            print(f"Noisy latents shape: {noisy_latents.shape}")
            raise
    
    avg_train_loss = train_loss_sum / len(train_dataloader)
    progress_bar.close()
    
    # ====== VALIDATION PHASE ======
    model.eval()
    val_loss_sum = 0
    
    with torch.no_grad():
        val_progress_bar = tqdm(total=len(val_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        for images, _ in val_dataloader:
            images = images.to(device)
            
            # Encode to latent space
            latents = model.encode(images)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()
            
            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            model_output = model(noisy_latents, timesteps, return_dict=False)[0]
            
            # Compute loss
            loss = F.mse_loss(model_output, noise)
            val_loss_sum += loss.item()
            val_progress_bar.update(1)

        val_progress_bar.close()

        # Save sample images after each epoch for monitoring
        train_gen = generate_images(num_images=4)
        train_sample_path = os.path.join(samples_dir, f'samples_train_epoch_{epoch+1}.png')
        vutils.save_image(train_gen, train_sample_path, nrow=2)
        print(f"Generated train samples saved: {train_sample_path}")

    
    avg_val_loss = val_loss_sum / len(val_dataloader)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
    

    # Track losses
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        epochs_no_improve = 0
        best_model_path = os.path.join(base_output_dir, 'best_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, best_model_path)
        print(f"✓ Best model saved: {best_model_path} (Val Loss: {avg_val_loss:.4f})\n")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
            break

    # Save training plot after each epoch
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'training_plot.png')
    plt.savefig(plot_path, dpi=100)
    plt.close()
    
    model.train()  # Switch back to train mode
    
    # Save regular checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}\n")

    
# ============================================================================
# STEP 5: Inference
# ============================================================================


print("Generating sample images...")
generated = generate_images(num_images=4)
print(f"Generated images shape: {generated.shape}")