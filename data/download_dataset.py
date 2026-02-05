import kagglehub

# Download latest version
path = kagglehub.dataset_download("nih-chest-xrays/sample")

print("Path to dataset files:", path)