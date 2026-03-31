import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU NOT FOUND")



# def load_da3_model(model_name="depth-anything/DA3-LARGE"):
#     """Initialize Depth-Anything-3 model on available device."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}") 