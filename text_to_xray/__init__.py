"""
================================================================================
CHEST AI: TEXT-TO-CHEST-XRAY REVERSE GENERATION PACKAGE
================================================================================
"""

from .config import (
    DiffusionConfig,
    default_config,
    PRIMARY_MODEL_ID,
    ALTERNATIVE_MODEL_ID,
    RESEARCH_SAFETY_DISCLAIMER
)
from .prompt_processor import (
    parse_chest_ai_report,
    adapt_report_to_diffusion_prompt,
    is_term_negated
)
from .model_loader import (
    load_diffusion_pipeline,
    clear_pipeline_cache,
    get_gpu_vram_gb
)
from .image_validator import (
    validate_synthetic_image,
    apply_synthetic_watermark
)
from .diffusion_engine import (
    generate_synthetic_xray
)

__all__ = [
    "DiffusionConfig",
    "default_config",
    "PRIMARY_MODEL_ID",
    "ALTERNATIVE_MODEL_ID",
    "RESEARCH_SAFETY_DISCLAIMER",
    "parse_chest_ai_report",
    "adapt_report_to_diffusion_prompt",
    "is_term_negated",
    "load_diffusion_pipeline",
    "clear_pipeline_cache",
    "get_gpu_vram_gb",
    "validate_synthetic_image",
    "apply_synthetic_watermark",
    "generate_synthetic_xray"
]
