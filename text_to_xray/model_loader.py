"""
================================================================================
CHEST AI: DIFFUSION MODEL LAZY-LOADER & MEMORY MANAGER
================================================================================
Provides on-demand, lazy-loading of Hugging Face diffusion pipelines with
device auto-detection, precision selection, and VRAM-constrained offloading
optimized for 4GB-8GB GPUs (e.g. RTX 2050) and safe CPU fallback.
================================================================================
"""

import os
import gc
import logging
from typing import Optional, Tuple, Any
import torch

from .config import DiffusionConfig, default_config

logger = logging.getLogger("CHEST_AI.text_to_xray.model_loader")

# In-memory singleton cache to prevent repeated re-loads
_PIPELINE_CACHE = {
    "model_id": None,
    "pipeline": None,
    "device_used": None,
    "precision_used": None
}


def get_gpu_vram_gb() -> float:
    """Returns total VRAM of the primary GPU in Gigabytes, or 0.0 if unavailable."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return 0.0


def clear_pipeline_cache():
    """Unloads cached diffusion pipeline and forces garbage collection."""
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE["pipeline"] is not None:
        logger.info("Unloading cached diffusion pipeline...")
        del _PIPELINE_CACHE["pipeline"]
        _PIPELINE_CACHE["pipeline"] = None
        _PIPELINE_CACHE["model_id"] = None
        _PIPELINE_CACHE["device_used"] = None
        _PIPELINE_CACHE["precision_used"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_diffusion_pipeline(
    config: Optional[DiffusionConfig] = None,
    force_reload: bool = False
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Lazy-loads the pretrained StableDiffusionPipeline from Hugging Face Hub.
    
    Returns:
        (pipeline, error_message): A tuple containing the initialized pipeline
        (or None if failed), and an error message string (or None if successful).
    """
    global _PIPELINE_CACHE

    cfg = config or default_config
    target_model_id = cfg.model_id

    # Return cached pipeline if already loaded with matching configuration
    if not force_reload and _PIPELINE_CACHE["pipeline"] is not None and _PIPELINE_CACHE["model_id"] == target_model_id:
        return _PIPELINE_CACHE["pipeline"], None

    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as e:
        err_msg = (
            f"The 'diffusers' library is required for report-to-image generation: {e}. "
            "Please ensure diffusers is installed in the active environment."
        )
        logger.error(err_msg)
        return None, err_msg

    # Detect hardware capabilities
    cuda_available = torch.cuda.is_available()
    vram_gb = get_gpu_vram_gb()
    logger.info(f"Loading diffusion model: {target_model_id} (CUDA: {cuda_available}, VRAM: {vram_gb:.2f}GB)")

    # Select precision
    torch_dtype = torch.float16 if cuda_available else torch.float32

    pipeline = None
    device_used = "cpu"

    try:
        # Load from Hugging Face Hub in safetensors format
        pipeline = StableDiffusionPipeline.from_pretrained(
            target_model_id,
            torch_dtype=torch_dtype,
            revision=cfg.model_revision,
            use_safetensors=True,
            safety_checker=None  # Radiology CXRs can trigger generic NSFW false positives
        )

        if cuda_available:
            # Memory optimization for 4GB VRAM cards (e.g. RTX 2050)
            if cfg.enable_attention_slicing:
                try:
                    pipeline.enable_attention_slicing()
                    logger.info("Attention slicing enabled for memory efficiency.")
                except Exception as ex:
                    logger.warning(f"Could not enable attention slicing: {ex}")

            # Apply CPU offload if VRAM <= 6.0GB to prevent CUDA OOM
            if cfg.enable_model_cpu_offload and vram_gb <= 6.0:
                try:
                    pipeline.enable_model_cpu_offload()
                    device_used = "cuda (model cpu offload)"
                    logger.info("Model CPU offload enabled for VRAM-constrained GPU.")
                except Exception as ex:
                    logger.warning(f"Model CPU offload failed ({ex}), falling back to direct GPU allocation.")
                    pipeline.to("cuda")
                    device_used = "cuda"
            else:
                pipeline.to("cuda")
                device_used = "cuda"
        else:
            pipeline.to("cpu")
            device_used = "cpu"

        # Update cache
        _PIPELINE_CACHE["pipeline"] = pipeline
        _PIPELINE_CACHE["model_id"] = target_model_id
        _PIPELINE_CACHE["device_used"] = device_used
        _PIPELINE_CACHE["precision_used"] = str(torch_dtype)

        logger.info(f"Diffusion pipeline ready on {device_used}.")
        return pipeline, None

    except torch.cuda.OutOfMemoryError as oom_err:
        clear_pipeline_cache()
        err_msg = (
            f"GPU Memory Exhausted (CUDA OOM): {oom_err}. "
            "Consider switching to CPU execution or enabling sequential CPU offloading."
        )
        logger.error(err_msg)
        return None, err_msg

    except Exception as exc:
        clear_pipeline_cache()
        err_msg = (
            f"Failed to load diffusion pipeline '{target_model_id}': {exc}. "
            "Please verify network connectivity, repository availability, or local cache."
        )
        logger.error(err_msg)
        return None, err_msg
