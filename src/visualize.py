import sys
import os
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
from src.models.diffusion import DiffusionModel

def make_diffusion_video(save_path="diffusion_process.gif"):
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🎥 Rendering video on {device}...")
    
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    
    # 2. Start from Pure Noise (Single Image)
    # Shape: [1, 1, 64, 64]
    image = torch.randn((1, 1, 64, 64)).to(device)
    
    # 3. Denoising Loop
    scheduler = model.noise_scheduler
    frames = []
    
    print("Capturing frames...")
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        with torch.no_grad():
            model_output = model(image, t, noise=None)
            step_output = scheduler.step(model_output, t, image)
            image = step_output.prev_sample
            
        # Capture frame every 20 steps (to keep GIF size reasonable)
        # Also strictly capture the last frame
        if i % 20 == 0 or i == len(scheduler.timesteps) - 1:
            # Normalize to [0, 255] uint8
            frame = (image.clamp(-1, 1) + 1) / 2
            frame = (frame * 255).type(torch.uint8).cpu().squeeze().numpy()
            
            # Convert to PIL Image
            pil_img = Image.fromarray(frame, mode='L')
            frames.append(pil_img)

    # 4. Save GIF
    print(f"Saving GIF with {len(frames)} frames...")
    
    # PAUSE EFFECT: Add the last frame 100 times (100 * 50ms = 5 seconds)
    last_frame = frames[-1]
    frames.extend([last_frame] * 100)
    
    # duration is ms per frame. 50ms = 20fps.
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0
    )
    print(f"✅ Video saved to {save_path}")

if __name__ == "__main__":
    make_diffusion_video()
