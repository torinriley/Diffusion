import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

DEVICE = "cpu"

ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.backends.mps.is_built() or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model_file = "data/v1-5-pruned-emaonly.ckpt"
models = model_loader.load_models(model_file, DEVICE)


prompt = "A painting of a planet from space"
uncond_prompt = "" 
do_cfg = True
cfg_scale = 8 


sampler_name = "ddpm"
num_inference_steps = 50
seed = 42 
strength = 0.75

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler_name,
    n_inference_steps=num_inference_steps,
    seed=seed,
    strength=strength, 
    models=models,
    device=DEVICE,
    tokenizer=tokenizer
)

output_image = Image.fromarray(output_image)
output_image.save("output.png")
