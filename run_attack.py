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
    
    # 2. Get 1 real image (Training Member)
    dataloader = get_dataloader("data/raw", batch_size=1)
    real_image = next(iter(dataloader)).to(device)
    
    # 3. Attack!
    attacker = PrivacyAttacker(model)
    loss = attacker.calculate_loss(real_image, device=device)
    is_member = attacker.predict_membership(loss, threshold=0.1)
    
    print(f"📉 Reconstruction Loss: {loss:.4f}")
    if is_member:
        print("🚨 ATTACK SUCCESSFUL: Model MEMORIZED this patient! (Privacy Risk ⚠️)")
    else:
        print("✅ ATTACK FAILED: Model did not memorize this patient. (Safe 🛡️)")

if __name__ == "__main__":
    run_attack()
