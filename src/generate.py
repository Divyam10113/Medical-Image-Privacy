import sys
import os
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from src.models.diffusion import DiffusionModel
from src.data.loader import get_dataloader
from src.eval.metrics import calculate_fid

def generate_images(batch_size=8, save_path="generated_xrays.png"):
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🎨 Generating on {device}...")
    
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    
    # 2. Start from Pure Noise
    # Shape: [Batch, Channels, Height, Width]
    image = torch.randn((batch_size, 1, 64, 64)).to(device)
    
    # 3. Denoising Loop (Reverse Diffusion)
    scheduler = model.noise_scheduler
    
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            # Predict noise
            # Note: We pass noise=None for the third argument
            model_output = model(image, t, noise=None)
            
            # Step backward (remove noise)
            step_output = scheduler.step(model_output, t, image)
            image = step_output.prev_sample

    # 4. Save Image Grid
    # Normalize from [-1, 1] back to [0, 1] for saving
    image_saved = (image / 2 + 0.5).clamp(0, 1)
    save_image(image_saved, save_path, nrow=4)
    print(f"✅ Saved generated images to {save_path}")
    
    return image # Return raw tensors [-1, 1] for FID

def evaluate_fid(generated_images):
    # Load real images for comparison
    print("📊 Calculating FID Score...")
    # NOTE: In a real research paper, we'd use the full test set (1000+ images).
    # Here we just use a small batch for demonstration.
    device = generated_images.device
    dataloader = get_dataloader("data/raw", batch_size=len(generated_images))
    real_images = next(iter(dataloader)).to(device)
    
    # FID expects uint8 [0, 255]
    real_uint8 = ((real_images / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
    fake_uint8 = ((generated_images / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
    
    # FID expects 3 channels (RGB Inception Mode)
    real_rgb = real_uint8.repeat(1, 3, 1, 1)
    fake_rgb = fake_uint8.repeat(1, 3, 1, 1)
    
    score = calculate_fid(real_rgb, fake_rgb)
    print(f"📈 FID Score: {score.item():.2f}")
    print("(Lower is Better. 0 = Perfect, >100 = Bad)")

if __name__ == "__main__":
    # Generate more images for a better statistical comparison
    fake_images = generate_images(batch_size=16)
    evaluate_fid(fake_images)
