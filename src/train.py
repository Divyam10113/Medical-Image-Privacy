import sys
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

# Add the project root to path so we can import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.diffusion import DiffusionModel
from src.data.loader import get_dataloader # Uncomment when loader is ready

def train(epochs=10, batch_size=32, lr=1e-4, debug=False, max_samples=None):
    # 1. Setup Device
    # This automatically picks the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available(): # For Mac
        device = "mps"
    else:
        device = "cpu"
    print(f"🚀 Training on device: {device}")

    # 2. Setup Model & Optimizer
    model = DiffusionModel().to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # 3. Setup Data (Mocking it for now if debug is True)
    if debug:
        print("⚠️ DEBUG MODE: Using random noise data")
        dataloader = [torch.randn(batch_size, 1, 64, 64) for _ in range(5)]
    else:
        dataloader = get_dataloader("data/raw", batch_size, max_samples=max_samples)

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            batch = batch.to(device)
            
            # Optimization Step
            optimizer.zero_grad()
            noise_pred, noise = model.training_step(batch)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())
    
    # 5. Save Model
    save_path = "model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete! Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run with fake data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Input batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training images")
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, debug=args.debug, max_samples=args.max_samples)
