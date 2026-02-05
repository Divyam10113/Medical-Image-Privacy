import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler

class DiffusionModel(nn.Module):
    def __init__(self, image_size=64):
        super().__init__()
        
        # 1. The Neural Network (UNet)
        # We use a pre-built UNet from the `diffusers` library to save time.
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=1, # Grayscale input
            out_channels=1, # Grayscale output
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"
            ),
        )
        
        # 2. The Noise Scheduler (The "Diffusion" part)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def forward(self, x, timesteps, noise):
        """
        Predict the noise added to the image.
        """

        return self.unet(x, timesteps, noise).sample

    def training_step(self, batch):
        """
        Executes a single forward pass training step.
        """
        noise = torch.randn(batch.shape, device=batch.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch.shape[0],), device=batch.device).long()
        noisy_images = self.noise_scheduler.add_noise(batch, noise, timesteps)
        noise_pred = self(noisy_images, timesteps, noise=None)
        return noise_pred, noise    
