"""Minimal configuration system using Pydantic + OmegaConf."""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from omegaconf import OmegaConf


class Config(BaseModel):
    """Main configuration class."""
    
    # Core parameters
    dimension: int = Field(default=1000, ge=2)
    kappa: float = Field(default=1000.0, gt=0)
    num_samples: int = Field(default=1000, gt=0)
    
    # Implementation settings
    implementation: str = Field(default="torch", pattern="^(numpy|scipy|torch)$")
    device: str = Field(default="auto")
    dtype: str = Field(default="float32", pattern="^(float16|bfloat16|float32|float64)$")
    
    # Generation settings
    random_mean_direction: bool = True
    seed: Optional[int] = 42
    output_dir: str = "generated_samples/"
    
    # Logging
    verbosity: str = Field(default="info", pattern="^(debug|info|warning|error|critical)$")
    
    # Wandb
    wandb_project: str = "vmf-sampling"
    wandb_offline: bool = False

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load config from file, merging with base defaults."""
        # Load base config
        base_config = OmegaConf.load('configs/base.yaml')
        
        # Merge with provided config if specified
        if config_path:
            user_config = OmegaConf.load(config_path)
            merged_config = OmegaConf.merge(base_config, user_config)
        else:
            merged_config = base_config
            
        return cls(**OmegaConf.to_container(merged_config))