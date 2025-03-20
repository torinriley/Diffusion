import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from config import Config, default_config, DeviceConfig

# Device configuration
ALLOW_CUDA = False
ALLOW_MPS = False

device = "cpu"
if torch.cuda.is_available() and ALLOW_CUDA:
    device = "cuda"
elif (torch.backends.mps.is_built() or torch.backends.mps.is_available()) and ALLOW_MPS:
    device = "mps"
print(f"Using device: {device}")

# Initialize configuration
config = Config(
    device=DeviceConfig(device=device),
    seed=42,
    tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
)

# Update diffusion parameters
config.diffusion.strength = 0.75
config.diffusion.cfg_scale = 8.0
config.diffusion.n_inference_steps = 50

# Load models with SE blocks enabled
model_file = "data/v1-5-pruned-emaonly.ckpt"
config.models = model_loader.load_models(model_file, device, use_se=True)

# Generate image
prompt = "A ultra sharp photorealtici painting of a futuristic cityscape at night with neon lights and flying cars"
uncond_prompt = ""

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    config=config
)

# Save output
output_image = Image.fromarray(output_image)
output_image.save("output.png")
