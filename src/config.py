from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch

@dataclass
class ModelConfig:
    # Image dimensions
    width: int = 512
    height: int = 512
    latents_width: int = 64  # width // 8
    latents_height: int = 64  # height // 8
    
    # Model architecture parameters
    n_embd: int = 1280
    n_head: int = 8
    d_context: int = 768
    
    # UNet parameters
    n_time: int = 1280
    n_channels: int = 4
    n_residual_blocks: int = 2
    
    # Attention parameters
    attention_heads: int = 8
    attention_dim: int = 1280

@dataclass
class DiffusionConfig:
    # Sampling parameters
    n_inference_steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.8
    
    # Sampler configuration
    sampler_name: str = "ddpm"
    beta_start: float = 0.00085
    beta_end: float = 0.0120
    beta_schedule: str = "linear"
    
    # Conditioning parameters
    do_cfg: bool = True
    cfg_scale: float = 7.5

@dataclass
class DeviceConfig:
    device: Optional[str] = None
    idle_device: Optional[str] = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.idle_device is None:
            self.idle_device = "cpu"

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    
    # Additional settings
    seed: Optional[int] = None
    tokenizer: Optional[Any] = None
    models: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Update latent dimensions based on image dimensions
        self.model.latents_width = self.model.width // 8
        self.model.latents_height = self.model.height // 8

# Default configuration instance
default_config = Config() 