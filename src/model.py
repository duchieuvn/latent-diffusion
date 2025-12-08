
import torch
from diffusers import UNet2DModel, AutoencoderKL


class SimpleLatentDiffusion(torch.nn.Module):
    def __init__(self, sample_size, in_channels, out_channels, device):
        super().__init__()
        # Load pre-trained VAE
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float32
        ).to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

        self.vae = vae
        self.unet = UNet2DModel(
            sample_size=sample_size,  # Actual latent spatial dimension
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512),  # Simpler architecture
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
            add_attention=False,  # Disable attention to avoid complexity
        )
        self.device = device

    def encode(self, images):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
        return latents

    def decode(self, latents):
        with torch.no_grad():
            images = self.vae.decode(latents / 0.18215).sample
            images = (images / 2 + 0.5).clamp(0, 1)
        return images

    def forward(self, x, timesteps, return_dict=False):
        return self.unet(x, timesteps, return_dict=return_dict)
