# CHEST AI: Hugging Face Report-to-Chest-X-Ray Reverse Diffusion Pipeline
## Research Specification, Architecture & Evaluation Framework

---

### 1. Executive Summary & Objective

In medical imaging AI, **bidirectional cycle consistency** is a fundamental research paradigm:
1. **Forward Path ($I \to T$):** Chest Radiograph $\to$ Diagnostic Report & Pathology Probabilities.
2. **Reverse Path ($T \to I$):** Diagnostic Report $\to$ Synthetic Radiograph.

Previously, `bidirectional_engine.py` contained an OpenCV edge-filter heuristic (`cv2.Laplacian` + `cv2.addWeighted`). This component has now been replaced with a **genuine, model-based Latent Diffusion generation pipeline** powered by Hugging Face (`text_to_xray`), integrating pretrained weights specialized for chest radiographs while strictly separating synthetic artifacts from authentic clinical data.

---

### 2. Candidate Model Evaluation & Selection

Four candidate repositories were audited against the project's clinical, technical, and licensing criteria:

| Candidate Repository | Architecture | Conditioning Type | Gated / Open | License | Audit Decision & Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`microsoft/radedit`** | Latent Diffusion UNet | Image + Multiple Binary Masks + Text | **Gated** (Manual Approval) | Microsoft Research | **Excluded for text-to-image.** RadEdit is an *image-editing* framework requiring an existing radiograph and manual binary segmentation masks, not a free-text report-to-image generator. |
| **`mahmoudibra98/compdiff-chest-xray`** | SD 2.1-base + HCN Conditioner | Demographics `[age, sex, race, class]` | Open | Research (arXiv:2603.16551) | **Excluded.** Requires mandatory demographic inputs `[t_age, t_sex, t_race]`. CHEST AI reports avoid fabricating demographic metadata. |
| **`danyalmalik/stable-diffusion-chest-xray`** | SD 2.1-512px DreamBooth | Free text | Open | CreativeML OpenRAIL-M | **Secondary / Fallback.** Dreambooth checkpoint trained on 1k ChestXray14 images in legacy `.bin` format. |
| **`P-RAJIV/cxr_stable_diffusion_v1_5`** | Fine-tuned Stable Diffusion v1.5 | Free-text clinical prompts | **Open** | **MIT License** | 🥇 **SELECTED AS PRIMARY.** Native `StableDiffusionPipeline`, formatted in modern `.safetensors` format, fine-tuned specifically on chest radiographs, supports FP16 with attention slicing & CPU offloading for 4GB VRAM. |

#### Selected Model Details: `P-RAJIV/cxr_stable_diffusion_v1_5`
* **Underlying Architecture:** Latent Diffusion Model (UNet 512×512, CLIP ViT-L/14 text encoder, AutoencoderKL VAE, DPMSolver / PNDM scheduler).
* **Weights Format:** `.safetensors` (prevents arbitrary pickle execution; 3.28 GB UNet, 235 MB Text Encoder, 160 MB VAE).
* **Intended Scope:** Research and educational synthesis of chest radiographs conditioned on natural language pathology descriptions.
* **License:** MIT License. Commercial and academic use permitted with attribution.

---

### 3. Pipeline Architecture (`text_to_xray`)

The new generation module is located at `medical_image_report/text_to_xray/`:

```
medical_image_report/text_to_xray/
├── __init__.py           # Package exports
├── config.py             # Model IDs, device detection, inference hyper-parameters, safety banners
├── prompt_processor.py   # Clinical findings extractor, zero-inversion negation & laterality parser
├── model_loader.py       # Lazy loader with attention slicing and CPU offload for 4GB GPUs
├── image_validator.py    # Technical integrity checks (non-degenerate, dimensions) & watermarking
└── diffusion_engine.py   # Main generation entrypoint and reproducible metadata auditor
```

#### Key Technical Innovations:

1. **Scoped Negation Preservation (Zero-Inversion):**
   * *The Problem:* Standard CLIP text encoders frequently ignore small function words like `"no"` or `"without"`, dangerously converting `"no pleural effusion"` into an effusion-positive image.
   * *Our Solution:* `prompt_processor.py` parses grammatical clauses. Negated terms (e.g., `pneumothorax`, `edema`, `effusion`) are stripped from positive prompts and **explicitly routed to the negative prompt**, discouraging the diffusion model from generating them.

2. **Laterality & Projection Preservation:**
   * Laterality (`left-sided`, `right-sided`, `bilateral`) is linked to its specific finding rather than applied globally.
   * Standard projections (`Frontal PA`, `Frontal AP`, `Lateral`) condition the perspective.

3. **VRAM-Constrained Memory Management:**
   * On consumer GPUs (e.g., NVIDIA GeForce RTX 2050 with 4.0 GB VRAM), standard Stable Diffusion will crash with CUDA OOM.
   * `model_loader.py` dynamically applies:
     - FP16 mixed precision (`torch.float16`).
     - `enable_attention_slicing()` to split self-attention computation into memory-efficient slices.
     - `enable_model_cpu_offload()`, offloading UNet/VAE/Text-Encoder modules to system RAM when idle.

4. **Mandatory Research Watermarking & Non-Clinical Separation:**
   * Generated images receive an indelible dark warning header:
     `"SYNTHETIC AI-GENERATED — RESEARCH ONLY — NOT FOR CLINICAL USE"`
   * Full JSON audit metadata is saved alongside the image in `synthetic_outputs/metadata/`.
   * Patient uploads and synthetic outputs are strictly isolated in separate directories.

---

### 4. Hardware and Software Specifications

* **Python:** 3.13.1
* **PyTorch:** 2.13.0+cu126 (CUDA enabled)
* **GPU Tested:** NVIDIA GeForce RTX 2050 (4.0 GB VRAM)
* **Key Libraries:** `diffusers==0.41.0`, `transformers==5.16.1`, `accelerate==1.14.0`, `safetensors==0.8.0`, `huggingface-hub==1.33.0`
* **Disk Footprint:** ~3.68 GB (model weights cached in `~/.cache/huggingface/hub/`)

---

### 5. Execution Commands

#### 1. Running Unit & Integration Regression Tests (Mocked, Fast):
```powershell
.\.venv\Scripts\python.exe test_text_to_xray.py
```
*(Runs 17 tests verifying parsing, zero-inversion negation, laterality, image validation, OOM recovery, and existing pipeline stability without downloading weights).*

#### 2. Running the Dry-Run Diagnostic Smoke Test:
```powershell
.\.venv\Scripts\python.exe smoke_test_diffusion.py
```

#### 3. Running Live Diffusion Inference (Downloads weights on first run):
```powershell
.\.venv\Scripts\python.exe smoke_test_diffusion.py --live --steps 25 --seed 42
```

#### 4. Launching the Interactive Streamlit Web Application:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

---

### 6. Known Failure Modes & Clinical Limitations

1. **Text-Conditioned Diffusion vs. Radiographic Ground Truth:**
   * The diffusion model samples from high-dimensional latent space. It generates an anatomically plausible *representation* of a condition; it is **NOT** a reconstruction of a specific patient's ribs, spine, or parenchymal tissue.
2. **Subtle Lesion Blindness:**
   * Fine-grained findings such as sub-centimeter pulmonary nodules, hair-line rib fractures, or apical pneumothorax margins are difficult for 512×512 latent diffusion to render reliably.
3. **Hardware Latency:**
   * On 4GB VRAM with sequential CPU offload, generating a 512×512 radiograph at 25 steps requires ~30–60 seconds per image.

---

### 7. Comprehensive Research Evaluation Protocol

For future academic defense or journal publication, evaluate synthetic outputs using this 5-dimension protocol:

1. **Anatomical Plausibility:** Radiologist visual scoring (1–5 Likert scale) assessing clavicle alignment, rib count, diaphragmatic contours, and cardiac silhouette shape.
2. **Report-Image Consistency:** Passing synthetic radiographs back into CHEST AI's forward DenseNet121 classifier to assess whether predicted label probabilities match the input report's positive findings (Closed-Loop Cycle Consistency).
3. **Negation Robustness:** Verifying that synthetic images generated from normal or negated reports produce zero false-positive disease activations.
4. **Laterality Fidelity:** Ensuring that a prompt requesting `"left pleural effusion"` does not generate a right-sided opacity.
5. **Distributional Realism:** Computing Fréchet Inception Distance (FID) and Multi-Scale Structural Similarity (MS-SSIM) against real MIMIC-CXR test partitions.
