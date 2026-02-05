import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MedicalDataset(Dataset):
    def __init__(self, image_dir, image_size=64, max_samples=None):
        """
        Args:
            image_dir (str): Path to the folder containing images.
            image_size (int): Size to resize images to (e.g., 64x64).
            max_samples (int): Optional limit on number of images to load.
        """
        self.image_dir = image_dir
        self.image_size = image_size
        
        self.image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(root, f))
        
        # Optimization: Limit dataset size for faster debugging/training
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
            print(f"⚠️ Dataset limited to {len(self.image_paths)} images.")
        
        # Standard transforms for Diffusion Models
        # We need images to be in range [-1, 1] for the DDPM model.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # Converts [0, 255] -> [0, 1]
            transforms.Normalize([0.5], [0.5]) # Converts [0, 1] -> [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert("L")
        image = self.transform(image)
        return image

def get_dataloader(image_dir, batch_size=32, image_size=64, max_samples=None):
    """Factory function to get the DataLoader."""
    dataset = MedicalDataset(image_dir, image_size, max_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Quick Test
    print("🧪 Testing Data Loader...")
    # Assume data is in data/raw relative to project root
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    loader = get_dataloader(data_dir, batch_size=2)
    
    try:
        images = next(iter(loader))
        print(f"✅ Success! Loaded batch of shape: {images.shape}")
        print(f"   Value range: [{images.min():.2f}, {images.max():.2f}]")
        if images.shape[1] != 1:
            print("❌ Error: Expected 1 channel (Grayscale), got", images.shape[1])
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
