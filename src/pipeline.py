import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .ddpm import DDPMSampler
import logging
from .config import Config, default_config

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

logging.basicConfig(level=logging.INFO)

def validate_strength(strength):
    if not 0 < strength <= 1:
        raise ValueError("Strength must be between 0 and 1")

def initialize_generator(seed, device):
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator

def encode_prompt(prompt, uncond_prompt, do_cfg, tokenizer, clip, device):
    clip.to(device)
    if do_cfg:
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        cond_context = clip(cond_tokens)
        uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt or ""], padding="max_length", max_length=77).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        uncond_context = clip(uncond_tokens)
        context = torch.cat([cond_context, uncond_context])
    else:
        tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        context = clip(tokens)
    return context

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def preprocess_image(input_image):
    input_image_tensor = input_image.resize((WIDTH, HEIGHT))
    input_image_tensor = np.array(input_image_tensor)
    input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
    input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
    input_image_tensor = input_image_tensor.unsqueeze(0)
    input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
    return input_image_tensor

def encode_image(input_image, models, device):
    # Preprocess the input image
    image_tensor = preprocess_image(input_image).to(device)
    
    # Encode the image using the VAE encoder
    encoder = models["encoder"]
    encoder.to(device)
    with torch.no_grad():
        # Create deterministic noise (zeros) since we want exact reconstruction
        noise = torch.zeros((1, 4, LATENTS_WIDTH, LATENTS_HEIGHT), device=device)
        latents = encoder(image_tensor, noise)
    
    return latents

def initialize_latents(input_image, strength, generator, models, device, sampler_name, n_inference_steps, mask_image=None):
    if input_image is None:
        # Initialize with random noise
        latents = torch.randn((1, 4, LATENTS_WIDTH, LATENTS_HEIGHT), generator=generator, device=device)
    else:
        # Initialize with encoded input image
        latents = encode_image(input_image, models, device)
        
        # If mask is provided for inpainting
        if mask_image is not None:
            # Process mask
            mask = mask_image.resize((WIDTH, HEIGHT))
            mask = np.array(mask)
            mask = torch.tensor(mask, dtype=torch.float32).to(device)
            mask = mask / 255.0  # Normalize to 0-1
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            mask = F.interpolate(mask, (LATENTS_WIDTH, LATENTS_HEIGHT))
            mask = mask.repeat(1, 4, 1, 1)  # Repeat for all latent channels
            
            # Create masked noise - torch.randn_like doesn't accept generator
            noise = torch.randn(latents.shape, device=device)
            masked_latents = latents * (1 - mask) + noise * mask
            latents = masked_latents
        
        # Add noise based on strength (for img2img)
        # torch.randn_like doesn't accept generator
        noise = torch.randn(latents.shape, device=device)
        latents = (1 - strength) * latents + strength * noise
    
    return latents

def get_sampler(sampler_name, generator, n_inference_steps):
    if sampler_name == "ddpm":
        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(n_inference_steps)
    else:
        raise ValueError(f"Unknown sampler value {sampler_name}.")
    return sampler

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def run_diffusion(latents, context, do_cfg, cfg_scale, models, device, sampler_name, n_inference_steps, generator):
    diffusion = models["diffusion"]
    diffusion.to(device)
    sampler = get_sampler(sampler_name, generator, n_inference_steps)
    timesteps = tqdm(sampler.timesteps)
    for timestep in timesteps:
        time_embedding = get_time_embedding(timestep).to(device)
        model_input = latents.repeat(2, 1, 1, 1) if do_cfg else latents
        model_output = diffusion(model_input, context, time_embedding)
        if do_cfg:
            output_cond, output_uncond = model_output.chunk(2)
            model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
        latents = sampler.step(timestep, latents, model_output)
    decoder = models["decoder"]
    decoder.to(device)
    images = decoder(latents)
    return images

def postprocess_images(images):
    images = rescale(images, (-1, 1), (0, 255), clamp=True)
    images = images.permute(0, 2, 3, 1)
    images = images.to("cpu", torch.uint8).numpy()
    return images[0]

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    mask_image=None,
    config: Config = default_config,
):
    with torch.no_grad():
        # Validate inputs and parameters
        if prompt is None or prompt.strip() == "":
            raise ValueError("Prompt cannot be empty")
            
        if uncond_prompt is None:
            uncond_prompt = ""
            
        validate_strength(config.diffusion.strength)
        
        # Initialize generator for reproducibility
        generator = initialize_generator(config.seed, config.device.device)
        
        # Encode text prompt
        context = encode_prompt(prompt, uncond_prompt, config.diffusion.do_cfg, 
                               config.tokenizer, config.models["clip"], config.device.device)
        
        # Initialize latents (either from noise or from input image)
        latents = initialize_latents(input_image, config.diffusion.strength, generator, 
                                    config.models, config.device.device, 
                                    config.diffusion.sampler_name, 
                                    config.diffusion.n_inference_steps, 
                                    mask_image)
        
        # Run diffusion process
        images = run_diffusion(latents, context, config.diffusion.do_cfg, 
                              config.diffusion.cfg_scale, config.models, 
                              config.device.device, config.diffusion.sampler_name, 
                              config.diffusion.n_inference_steps, generator)
        
        # Post-process and return the images
        return postprocess_images(images)
