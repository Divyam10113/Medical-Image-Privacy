import torch
import torchmetrics

def calculate_fid(real_images, generated_images):
    """
    Calculates Fréchet Inception Distance (FID) between real and generated images.
    """
    # Fix: MPS (Mac) doesn't support float64, which FID uses internally.
    # We force the calculation to run on CPU to be safe.
    fid = torchmetrics.image.fid.FrechetInceptionDistance(feature=64)
    
    # Move inputs to CPU
    real_images = real_images.cpu()
    generated_images = generated_images.cpu()
    
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    return fid.compute()
