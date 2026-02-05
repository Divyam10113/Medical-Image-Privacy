import torch
import torch.nn.functional as F

class PrivacyAttacker:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def calculate_loss(self, image, device="cpu"):
        """Calculates how 'surprised' the model is by an image."""
        # 1. Add noise (just like training)
        noise = torch.randn_like(image).to(device)
        timesteps = torch.randint(0, 1000, (1,), device=device).long()
        
        # 2. Get prediction
        with torch.no_grad():
            noisy_image = self.model.noise_scheduler.add_noise(image, noise, timesteps)
            noise_pred = self.model(noisy_image, timesteps, noise=None)
            
        # 3. Return Error (MSE)
        loss = F.mse_loss(noise_pred, noise)
        return loss.item()

    def predict_membership(self, loss, threshold=0.1):
        """
        Predicts if an image was in the training set based on reconstruction loss.
        """
        if loss < threshold:
            return True
        else:
            return False
