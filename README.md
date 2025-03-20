# Custom Diffusion Model Text-to-Image Generator

A custom implementation of a text-to-image generation model with Squeeze-Excitation blocks for improved feature selection.

## Features

- Custom diffusion model implementation
- Squeeze-Excitation blocks for enhanced feature selection
- Configurable generation parameters
- User-friendly Gradio interface
- Automatic model downloading

## Usage

1. **Enter your prompt** in the text box.
2. **(Optional) Adjust advanced settings:**
   - Negative prompt
   - Seed
   - Image dimensions
   - Guidance scale
   - Number of inference steps
3. **Click "Run"** to generate the image.

## Technical Details

- **Model:** Stable Diffusion v1.5 with custom modifications
- **Architecture:** Custom UNet with Squeeze-Excitation blocks
- **Default settings:**
  - Image size: 512x512
  - Guidance scale: 7.5
  - Inference steps: 50
  - Seed: 42 (randomizable)

## Examples

Try these example prompts:
- "A ultra sharp photorealistic painting of a futuristic cityscape at night with neon lights and flying cars"
- "A serene mountain landscape at sunset with snow-capped peaks and a clear lake reflection"
- "A detailed portrait of a cyberpunk character with glowing neon implants and holographic tattoos"

## Installation

The model will be automatically downloaded on first run. No manual installation required.

## License

This project is available under the MIT License.

## Try it out on Hugging Face Spaces

Check out the live demo on Hugging Face Spaces: [torinriley/Diffusion](https://huggingface.co/spaces/torinriley/Diffusion)
