"""
================================================================================
CHEST AI: TEXT-TO-CHEST-XRAY CONFIGURATION MODULE
================================================================================
Centralized configuration for diffusion model selection, memory optimizations,
output directories, and research safety disclaimers.
================================================================================
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Root paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)

# Default synthetic output root (isolated from clinical patient data)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "synthetic_outputs")
DEFAULT_METADATA_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "metadata")
DEFAULT_IMAGES_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "images")

# Primary and alternative pretrained model repositories
PRIMARY_MODEL_ID = "P-RAJIV/cxr_stable_diffusion_v1_5"
ALTERNATIVE_MODEL_ID = "danyalmalik/stable-diffusion-chest-xray"

# Prominently displayed synthetic research safety disclaimer
RESEARCH_SAFETY_DISCLAIMER = (
    "SYNTHETIC AI-GENERATED IMAGE — FOR RESEARCH AND EDUCATIONAL USE ONLY. "
    "NOT AN AUTHENTIC PATIENT RADIOGRAPH. NOT FOR CLINICAL DIAGNOSIS OR TREATMENT."
)

@dataclass
class DiffusionConfig:
    """Configuration dataclass for report-to-chest-X-ray diffusion generation."""
    model_id: str = PRIMARY_MODEL_ID
    model_revision: Optional[str] = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    images_dir: str = DEFAULT_IMAGES_DIR
    metadata_dir: str = DEFAULT_METADATA_DIR
    
    # Inference generation defaults
    num_inference_steps: int = 25
    guidance_scale: float = 7.0
    height: int = 512
    width: int = 512
    default_seed: int = 42
    
    # Supported projections
    supported_projections: List[str] = field(default_factory=lambda: [
        "Frontal PA", "Frontal AP", "Lateral", "Unknown"
    ])
    
    # Device and GPU memory optimization
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype: torch.dtype = field(default_factory=lambda: torch.float16 if torch.cuda.is_available() else torch.float32)
    enable_attention_slicing: bool = True
    enable_model_cpu_offload: bool = True
    
    # Safety label
    safety_disclaimer: str = RESEARCH_SAFETY_DISCLAIMER

    def __post_init__(self):
        # Create output directories if they do not exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "height": self.height,
            "width": self.width,
            "enable_attention_slicing": self.enable_attention_slicing,
            "enable_model_cpu_offload": self.enable_model_cpu_offload,
            "safety_disclaimer": self.safety_disclaimer
        }


# Global singleton configuration
default_config = DiffusionConfig()
