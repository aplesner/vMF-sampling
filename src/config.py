import json
import yaml
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from .vmf_sampler import Implementation


class VMFConfig(BaseModel):
    """Configuration class for vMF sampling experiments."""
    
    mu_dim: int = Field(default=3, description="Dimension of mu", ge=2)
    kappa: float = Field(default=1.0, description="Concentration parameter", gt=0)
    num_samples: int = Field(default=1000, description="Number of samples to generate", gt=0)
    implementation: str = Field(default="numpy", description="Implementation: 'scipy', 'numpy', 'torch'")
    wandb_project: str = Field(default="vmf-sampling", description="Weights & Biases project name")
    seed: int | None = Field(default=None, description="Random seed")
    device: str = Field(default="auto", description="Torch device ('cpu', 'cuda', 'auto')")
    wandb_offline: bool = Field(default=True, description="Whether to run wandb in offline mode")
    
    
    @field_validator('implementation')
    @classmethod
    def validate_implementation(cls, v):
        try:
            Implementation(v)
        except ValueError:
            valid_values = [impl.value for impl in Implementation]
            raise ValueError(f"implementation must be one of: {valid_values}")
        return v
    
    @classmethod
    def from_json(cls, filepath: str|Path):
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: str|Path):
        """Load configuration from YAML file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    
    def __repr__(self):
        return f"VMFConfig({self.to_dict()})"