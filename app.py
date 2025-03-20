import gradio as gr
import numpy as np
import random
import torch
from PIL import Image
import os
from huggingface_hub import hf_hub_download
from pathlib import Path
import sys

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import model_loader
from src import pipeline
from src.config import Config, DeviceConfig
from transformers import CLIPTokenizer

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Model configuration
MODEL_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_FILENAME = "v1-5-pruned-emaonly.ckpt"
model_file = data_dir / MODEL_FILENAME

# Download model if it doesn't exist
if not model_file.exists():
    print(f"Downloading model from {MODEL_REPO}...")
    model_file = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=data_dir,
        local_dir_use_symlinks=False
    )
    print("Model downloaded successfully!")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize configuration
config = Config(
    device=DeviceConfig(device=device),
    tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
)

# Load models
config.models = model_loader.load_models(str(model_file), device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def txt2img(
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    # Update config with user settings
    config.seed = seed
    config.diffusion.cfg_scale = guidance_scale
    config.diffusion.n_inference_steps = num_inference_steps
    config.model.width = width
    config.model.height = height
    
    # Generate image
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=negative_prompt,
        input_image=None,
        config=config
    )
    
    # Convert numpy array to PIL Image
    image = Image.fromarray(output_image)
    
    return image, seed

def img2img(
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    input_image,
    strength,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        
        if input_image is None:
            return None, seed
            
        # Update config with user settings
        config.seed = seed
        config.diffusion.cfg_scale = guidance_scale
        config.diffusion.n_inference_steps = num_inference_steps
        config.model.width = width
        config.model.height = height
        config.diffusion.strength = strength
        
        # Generate image
        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt=negative_prompt,
            input_image=input_image,
            config=config
        )
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(output_image)
        
        return image, seed
    except Exception as e:
        print(f"Error in img2img: {str(e)}")
        gr.Warning(f"Error: {str(e)}")
        return None, seed

def inpaint(
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    input_image,
    mask_image,
    strength,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        
        if input_image is None or mask_image is None:
            gr.Warning("Both input image and mask are required for inpainting")
            return None, seed
        
        # Ensure mask is in the right format
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
            
        # Update config with user settings
        config.seed = seed
        config.diffusion.cfg_scale = guidance_scale
        config.diffusion.n_inference_steps = num_inference_steps
        config.model.width = width
        config.model.height = height
        config.diffusion.strength = strength
        
        # Generate image with mask
        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt=negative_prompt,
            input_image=input_image,
            mask_image=mask_image,
            config=config
        )
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(output_image)
        
        return image, seed
    except Exception as e:
        print(f"Error in inpainting: {str(e)}")
        gr.Warning(f"Error: {str(e)}")
        return None, seed

examples = [
    "A ultra sharp photorealtici painting of a futuristic cityscape at night with neon lights and flying cars",
    "A serene mountain landscape at sunset with snow-capped peaks and a clear lake reflection",
    "A detailed portrait of a cyberpunk character with glowing neon implants and holographic tattoos",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 640px;
}

.tabs {
    margin-top: 10px;
    margin-bottom: 10px;
}

.disclaimer {
    font-size: 0.8em;
    color: #666;
    margin-top: 20px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(" # LiteDiffusion")

        with gr.Tabs(elem_classes="tabs") as tabs:
            with gr.TabItem("Text-to-Image"):
                txt2img_prompt = gr.Text(
                    label="Prompt",
                    max_lines=1,
                    placeholder="Enter your prompt",
                )
                txt2img_run = gr.Button("Generate", variant="primary")
                txt2img_result = gr.Image(label="Result")

            with gr.TabItem("Image-to-Image"):
                img2img_prompt = gr.Text(
                    label="Prompt",
                    max_lines=1,
                    placeholder="Enter your prompt",
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(label="Input Image", type="pil")
                        strength_slider = gr.Slider(
                            label="Strength",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.8,
                        )
                        img2img_run = gr.Button("Generate", variant="primary")
                    
                    with gr.Column(scale=1):
                        img2img_result = gr.Image(label="Result")
            
            with gr.TabItem("Inpainting"):
                inpaint_prompt = gr.Text(
                    label="Prompt",
                    max_lines=1,
                    placeholder="Enter your prompt",
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        inpaint_image = gr.Image(label="Input Image", type="pil")
                        inpaint_mask = gr.Image(label="Mask (White areas will be inpainted)", type="pil")
                        inpaint_strength = gr.Slider(
                            label="Strength",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.8,
                        )
                        inpaint_run = gr.Button("Generate", variant="primary")
                    
                    with gr.Column(scale=1):
                        inpaint_result = gr.Image(label="Result")

        with gr.Accordion("Advanced Settings", open=False):
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=42,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=7.5,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=50,
                )
                
        gr.Markdown(
            "By using LiteDiffusion, you agree to the terms in our [disclaimer](disclaimer.md).", 
            elem_classes="disclaimer"
        )
        
        # Example prompts for text to image
        gr.Examples(examples=examples, inputs=[txt2img_prompt])
        
    # Text-to-Image generation
    txt2img_run.click(
        fn=txt2img,
        inputs=[
            txt2img_prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[txt2img_result, seed],
    )
    
    # Image-to-Image generation
    img2img_run.click(
        fn=img2img,
        inputs=[
            img2img_prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            input_image,
            strength_slider,
        ],
        outputs=[img2img_result, seed],
    )
    
    # Inpainting
    inpaint_run.click(
        fn=inpaint,
        inputs=[
            inpaint_prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            inpaint_image,
            inpaint_mask,
            inpaint_strength,
        ],
        outputs=[inpaint_result, seed],
    )

if __name__ == "__main__":
    demo.launch() 