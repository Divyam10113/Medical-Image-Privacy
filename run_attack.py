import torch
from src.models.diffusion import DiffusionModel
from src.eval.privacy import PrivacyAttacker
from src.data.loader import get_dataloader

def run_attack():
    # 1. Load Model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🕵️‍♀️ Running Privacy Attack on {device}...")
    
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    
    # 2. Analyze a batch of real images
    # We want to see the distribution of how well the model reconstructs data
    BATCH_SIZE = 32
    dataloader = get_dataloader("data/raw", batch_size=BATCH_SIZE, max_samples=BATCH_SIZE)
    images = next(iter(dataloader)).to(device)
    
    print(f"📊 Analyzing reconstruction error on {BATCH_SIZE} validation images...")
    
    # 3. Calculate Loss for each image
    attacker = PrivacyAttacker(model)
    losses = []
    
    for i in range(BATCH_SIZE):
        loss = attacker.calculate_loss(images[i:i+1], device=device)
        losses.append(loss)
    
    # 4. Report scientific statistics
    import numpy as np
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    min_loss = np.min(losses)
    
    print("\n📈 Privacy Audit Results:")
    print(f"   Mean Reconstruction Loss: {mean_loss:.5f} ± {std_loss:.5f}")
    print(f"   Min Reconstruction Loss:  {min_loss:.5f}")
    print("-" * 40)
    print("INTERPRETATION:")
    print("   Lower Loss = Model fits the data curve better.")
    print("   To strictly prove privacy violations, you would compare this")
    print("   against a 'Held-Out' test set (not available in this demo).")

if __name__ == "__main__":
    run_attack()
