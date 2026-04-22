import torch
from torchvision import models
import os

print("Downloading MobileNetV2 weights...")

# Download and save weights
model = models.mobilenet_v2(weights='IMAGENET1K_V1')

# Save to cache location
cache_dir = os.path.join(os.path.dirname(torch.hub.get_dir()), 'checkpoints')
os.makedirs(cache_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(cache_dir, 'mobilenet_v2-weights.pth'))

print("Download complete!")
print(f"Saved to: {cache_dir}")