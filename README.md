# Diffusion Pipeline from Sratch
This project implements a text-to-image generation pipeline inspired by the Stable Diffusion architecture. The pipeline was built entirely from scratch in PyTorch. It integrates a Variational Autoencoder (VAE) for latent space compression, Denoising Diffusion Probabilistic Models (DDPM) for iterative denoising, and CLIP-based text embeddings for aligning text and images effectively.

•	Custom Variational Autoencoder (VAE): Compresses images into latent representations for efficient generation.

•	DDPM Sampling: Implements iterative denoising to generate high-quality images from noise.

•	Text Embedding with CLIP: Ensures precise alignment of text and generated images.



## Thanks to the following resources
| Resource                           | Description                              |
|------------------------------------|------------------------------------------|
| [Tokenizer](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer) | Tokenizer files for Stable Diffusion.    |
| [Model Repository](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) | Main Hugging Face repository for Stable Diffusion v1.5. |
| [Research Paper](https://arxiv.org/pdf/2112.10752) | Original Stable Diffusion research paper. |
