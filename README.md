# Chest X-Ray Diagnostic Report Generator (V3)

An end-to-end automated radiology report generation pipeline using a DenseNet121 encoder, a Spatial Transformer Encoder, a Multi-Head Cross-Attention Transformer Decoder, and a 14-disease multi-task classification head. Built with Python, Keras 3, TensorFlow, and Streamlit.

---

## 🚀 Key Features in V3

1. **Spatial Transformer Encoder**: Preserves raw $7\times 7$ grid visual feature tokens with 2D positional embeddings instead of global pooling, keeping spatial relationship context intact.
2. **Medical Pre-trained Initialization**: Transfers visual encoder weights layer-by-layer from chest X-ray models pre-trained on MIMIC-CXR.
3. **Multi-Task Learning**: Shared DenseNet121 + Spatial Transformer encoder branch into dual heads:
   - **Report Decoder Head**: Generates clinical text report tokens autoregressively.
   - **Classification Head**: Output probability predictions for 14 CheXpert clinical pathologies.
4. **Resumable 3-Phase Training**: 
   - **Phase 1**: Frozen visual backbone, warmup schedule.
   - **Phase 2**: Fine-tune top 80 backbone layers.
   - **Phase 3**: Fine-tune all layers.
   - Fully resumable from any epoch checkpoint with complete optimizer and state recovery.
5. **Configurable Beam Search**: Supports both Sampling/Greedy search and Beam Search decoding (beam widths 1-5) with length normalization and loop/repetition prevention.
6. **Unified Streamlit Frontend**: Beautiful dark-mode dashboard allowing interactive uploads, Greedy vs Beam search configuration, model selection, and real-time inference.

---

## 📂 Project Directory Structure

```text
medical_image_report/
├── app.py                     # Streamlit frontend application & unified custom layers
├── datasets/                  # Datasets and processed training/validation CSVs
├── models/                    # Model directories (01_efficientnet, 02_densenet_v2, v3 checkpoints)
└── scripts/
    ├── preprocessing/         # Data audit and cleaning scripts
    ├── training/
    │   └── train_densenet121_mimic_v3.py   # 3-Phase Resumable Multi-Task training script
    └── testing/
        └── evaluate_model.py  # Validation evaluation (NLP + pathology F1 metrics)
```

---

## 🛠️ Usage Commands

### 1. Verification & Sanity Checks
Run a quick model forward-pass check on a single batch to verify tensor shapes, losses, and pipeline consistency:
```bash
python scripts/training/train_densenet121_mimic_v3.py --test-only --num-samples 100
```

### 2. Model Training
Train the model from scratch on MIMIC-CXR:
```bash
python scripts/training/train_densenet121_mimic_v3.py --num-samples 50000 --batch-size 16
```
Configure spatial encoder layers, heads, or classification loss weights using CLI parameters:
```bash
python scripts/training/train_densenet121_mimic_v3.py --encoder-layers 2 --encoder-heads 4 --class-loss-weight 0.5
```

### 3. Resuming Training
Resume training automatically from the latest checkpoint (`latest_checkpoint.keras` + metadata):
```bash
python scripts/training/train_densenet121_mimic_v3.py --resume auto
```
Or resume from a specific epoch checkpoint:
```bash
python scripts/training/train_densenet121_mimic_v3.py --resume models/02_densenet121_mimic_v3/checkpoint_epoch_12.keras
```

### 4. Running Model Evaluation
Run the evaluation script to calculate NLP metrics (BLEU-1 to BLEU-4, ROUGE-L) and clinical correctness F1 scores for both Greedy and Beam Search:
```bash
python scripts/testing/evaluate_model.py --model-dir models/02_densenet121_mimic_v3 --num-samples 100 --beam-width 3
```

### 5. Launching the Web App
Run the Streamlit frontend interface locally:
```bash
streamlit run app.py
```

---

## 📦 Prerequisites & Setup

1. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/MacOS:
   source .venv/bin/activate
   ```
2. **Install requirements:**
   ```bash
   pip install -r requirements_app.txt
   ```

---

## 🛡️ License

MIT License
