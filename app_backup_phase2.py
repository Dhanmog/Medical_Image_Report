import os
import json
import re
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import streamlit as st
import pandas as pd
from collections import Counter
import math

try:
    from report_refinement import refine_report, clean_raw_report
except ImportError:
    refine_report = None
    clean_raw_report = None

try:
    from chest_validator import validate_chest_xray, DEFAULT_CHEST_THRESHOLD
except ImportError:
    validate_chest_xray = None
    DEFAULT_CHEST_THRESHOLD = 0.80



# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Medical Image to Diagnostic Text",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .report-card {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .report-card-actual {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .report-card-generated {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .report-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .report-text {
        font-size: 1.2rem;
        font-weight: 500;
        color: #0F172A;
        font-family: 'Courier New', Courier, monospace;
    }
    .metric-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
        margin: 0.3rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.78rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .model-badge {
        display: inline-block;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1D4ED8;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# 2. PATHS & MODEL REGISTRY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Registry of all trained models: each entry has display label, paths, and metadata
MODEL_REGISTRY = {
    "🚀 DenseNet121 v3 — Fine-tuned (Max Accuracy)": {
        "model_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v3", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v3", "tokenizer_vocab.json"),
        "architecture": "DenseNet121 + Spatial Transformer + 3-Block Decoder",
        "dataset":      "MIMIC-CXR (50,000 samples)",
        "epochs":       "10 Epochs (Phase 1 + Phase 2 Multi-Task)",
        "val_accuracy": "89.15% Pathology AUC (27.05% Vocab Acc)",
        "description":  "Advanced DenseNet121 v3 multi-task visual encoder + 3-block Transformer decoder with 14 CheXpert pathology classification head.",
    },
    "⚡ EfficientNetB0 v2 (Best Model)": {
        "model_path":  os.path.join(BASE_DIR, "models", "01_efficientnetb0_mimic_v2", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "01_efficientnetb0_mimic_v2", "tokenizer_vocab.json"),
        "architecture": "EfficientNetB0 + Transformer Decoder",
        "dataset":      "MIMIC-CXR (Preprocessed ~6k/epoch)",
        "epochs":       "2 Epochs",
        "val_accuracy": "30.13% Vocab Acc (Val Loss: 1.95)",
        "description":  "Fine-tuned EfficientNetB0 encoder with cross-attention Transformer decoder.",
    },
    "🏆 EfficientNetB0 Best Checkpoint": {
        "model_path":  os.path.join(BASE_DIR, "models", "01_efficientnetb0_mimic_v2", "best_model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "01_efficientnetb0_mimic_v2", "tokenizer_vocab.json"),
        "architecture": "EfficientNetB0 + Transformer Decoder",
        "dataset":      "MIMIC-CXR (Preprocessed ~6k/epoch)",
        "epochs":       "2 Epochs (Best Val Loss Checkpoint)",
        "val_accuracy": "30.13% Vocab Acc",
        "description":  "Saved at the epoch with lowest validation loss. Identical architecture to v2 model.",
    },
    "🔬 DenseNet121 MIMIC": {
        "model_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic", "tokenizer_vocab.json"),
        "architecture": "DenseNet121 + Transformer Decoder",
        "dataset":      "MIMIC-CXR (4,000 subset)",
        "epochs":       "5 Epochs",
        "val_accuracy": "26.83% Vocab Acc",
        "description":  "DenseNet121 encoder (no fine-tuning) with Transformer decoder, trained on a 4k image subset.",
    },
    "🧬 DenseNet121 v2 — Fine-tuned (Best DenseNet)": {
        "model_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v2", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v2", "tokenizer_vocab.json"),
        "architecture": "DenseNet121 + Transformer Decoder v2 (Fine-tuned)",
        "dataset":      "MIMIC-CXR (8,000 samples)",
        "epochs":       "Phase 1 (7 ep) + Phase 2 (Top 50 layers unfrozen)",
        "val_accuracy": "28.41% Vocab Acc",
        "description":  "Improved DenseNet121 with 2-block Transformer decoder, GELU FFN, and dropout regularization. Phase 2 fine-tuned top 50 layers at LR=1e-5.",
    },
    "🪐 DenseNet121 v4 — Highres 14x14 (Fine-tuning)": {
        "model_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v4_highres", "checkpoint_epoch_20.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v4_highres", "tokenizer_vocab.json"),
        "architecture": "DenseNet121 (14x14 Spatial Grid) + Multi-Task Decoder",
        "dataset":      "MIMIC-CXR (Preprocessed ~50k)",
        "epochs":       "20 Epochs (Phase 1 + Phase 2)",
        "val_accuracy": "88.40% Pathology AUC",
        "description":  "High-resolution 14x14 feature map resolution architecture targeting fine-grained lung opacity structure detection.",
    },
    "🌟 Stanford CheXagent-8b (SOTA Foundation VLM)": {
        "model_path":  os.path.join(BASE_DIR, "models", "05_chexagent_8b"),
        "vocab_path":  "",
        "is_vlm":       True,
        "architecture": "Vision-Language Foundation Model (CheXzero + Mistral-LLM)",
        "dataset":      "CheXbench / MIMIC-CXR (Millions of CXRs)",
        "epochs":       "Pre-trained SOTA Foundation Model",
        "val_accuracy": "SOTA RadGraph F1 ~0.38",
        "description":  "Stanford AIMI Foundation Model for Chest X-Ray interpretation. Generates comprehensive clinical Findings and Impression sections with radiology-grade precision.",
    },
    "📦 CNN-Transformer MIMIC (5 epochs)": {
        "model_path":  os.path.join(BASE_DIR, "models", "03_cnn_transformer_mimic", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "03_cnn_transformer_mimic", "tokenizer_vocab.json"),
        "architecture": "CNN Encoder + Transformer Decoder",
        "dataset":      "MIMIC-CXR (Full processed)",
        "epochs":       "5 Epochs",
        "val_accuracy": "68.90% Token-Level Acc",
        "description":  "Trained 5-epoch model on MIMIC-CXR dataset with highest token-level sequence accuracy.",
    },
    "🫁 Shenzhen TB Specialist": {
        "model_path":  os.path.join(BASE_DIR, "models", "04_cnn_transformer_shenzhen", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "04_cnn_transformer_shenzhen", "tokenizer_vocab.json"),
        "architecture": "CNN Encoder + Transformer Decoder",
        "dataset":      "Shenzhen TB Chest X-ray",
        "epochs":       "5 Epochs",
        "val_accuracy": "97.42% TB Diagnostic Acc",
        "description":  "Specialist diagnostic model trained exclusively on Shenzhen tuberculosis chest X-ray dataset.",
    },
}

DEFAULT_SAMPLE_IMAGE = os.path.join(BASE_DIR, "xray1.jpg")
IMAGE_SIZE  = (224, 224)
MAX_REPORT_LENGTH = 80
START_TOKEN = "<start>"
END_TOKEN   = "<end>"


# ============================================================
# 3. CUSTOM MODEL LAYERS (REGISTERED FOR KERAS SERIALIZATION)
# ============================================================
# Both v1 and v2 models use the class names "PositionalEmbedding" and
# "TransformerDecoderBlock" (registered as Custom>...).  Keras resolves
# registered_name via the global registry, so both model versions must
# share the SAME registered class.
#
# Unified design:
#   dropout_rate = 0.0  →  v1 behaviour  (key_dim=embed_dim, ReLU, no dropout)
#   dropout_rate > 0    →  v2 behaviour  (key_dim=embed_dim//num_heads, GELU, dropout)
#
# V1 configs never saved dropout_rate, so it defaults to 0.0 on load.
# V2 configs explicitly save dropout_rate=0.1.

@keras.utils.register_keras_serializable(package="Custom")
class PositionalEmbedding(layers.Layer):
    """Unified PositionalEmbedding for v1 (no dropout) and v2 (with dropout)."""
    def __init__(self, vocab_size, seq_len, embed_dim, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.vocab_size   = vocab_size
        self.seq_len      = seq_len
        self.embed_dim    = embed_dim
        self.dropout_rate = float(dropout_rate)
        self.token_emb    = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_emb      = layers.Embedding(seq_len, embed_dim)
        if self.dropout_rate > 0.0:
            self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        length    = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        x = self.token_emb(inputs) + self.pos_emb(positions)
        if self.dropout_rate > 0.0:
            x = self.dropout(x, training=training)
        return tf.cast(x, self.compute_dtype)

    def compute_mask(self, inputs, mask=None):
        return keras.ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size":   self.vocab_size,
            "seq_len":      self.seq_len,
            "embed_dim":    self.embed_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package="Custom")
class TransformerDecoderBlock(layers.Layer):
    """Unified TransformerDecoderBlock for v1 (no dropout) and v2 (with dropout).

    dropout_rate == 0.0  →  v1: key_dim=embed_dim, ReLU FFN, no dropout layers
    dropout_rate  > 0.0  →  v2: key_dim=embed_dim//num_heads, GELU FFN, dropout
    """
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.embed_dim    = embed_dim
        self.dense_dim    = dense_dim
        self.num_heads    = num_heads
        self.dropout_rate = float(dropout_rate)

        is_v2   = self.dropout_rate > 0.0
        key_dim = (embed_dim // num_heads) if is_v2 else embed_dim
        ln_eps  = 1e-6 if is_v2 else 1e-3

        self.mha_causal = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dtype="float32")
        self.mha_cross  = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dtype="float32")

        if is_v2:
            self.ffn = keras.Sequential([
                layers.Dense(dense_dim, activation="gelu", dtype="float32"),
                layers.Dropout(self.dropout_rate, dtype="float32"),
                layers.Dense(embed_dim, dtype="float32"),
            ])
            self.dropout1 = layers.Dropout(self.dropout_rate, dtype="float32")
            self.dropout2 = layers.Dropout(self.dropout_rate, dtype="float32")
            self.dropout3 = layers.Dropout(self.dropout_rate, dtype="float32")
        else:
            self.ffn = keras.Sequential([
                layers.Dense(dense_dim, activation="relu", dtype="float32"),
                layers.Dense(embed_dim, dtype="float32"),
            ])

        self.ln1 = layers.LayerNormalization(epsilon=ln_eps, dtype="float32")
        self.ln2 = layers.LayerNormalization(epsilon=ln_eps, dtype="float32")
        self.ln3 = layers.LayerNormalization(epsilon=ln_eps, dtype="float32")

    def call(self, inputs, encoder_outputs, mask=None, training=False):
        # Cast inputs to float32 internally so that calculations and additions match the sub-layers' compute dtype (float32)
        inputs_f32 = tf.cast(inputs, tf.float32)
        encoder_outputs_f32 = tf.cast(encoder_outputs, tf.float32)
        if self.dropout_rate > 0.0:
            # v2/v3 path — with dropout
            attn1 = self.mha_causal(query=inputs_f32, value=inputs_f32, key=inputs_f32, attention_mask=mask, use_causal_mask=True)
            attn1 = self.dropout1(attn1, training=training)
            out1  = self.ln1(inputs_f32 + attn1)
            
            attn2 = self.mha_cross(query=out1, value=encoder_outputs_f32, key=encoder_outputs_f32)
            attn2 = self.dropout2(attn2, training=training)
            out2  = self.ln2(out1 + attn2)
            
            ffn_out = self.ffn(out2, training=training)
            ffn_out = self.dropout3(ffn_out, training=training)
            res = self.ln3(out2 + ffn_out)
        else:
            # v1 path — no dropout
            attn1   = self.mha_causal(query=inputs_f32, value=inputs_f32, key=inputs_f32, attention_mask=mask, use_causal_mask=True)
            out1    = self.ln1(inputs_f32 + attn1)
            
            attn2   = self.mha_cross(query=out1, value=encoder_outputs_f32, key=encoder_outputs_f32)
            out2    = self.ln2(out1 + attn2)
            
            ffn_out = self.ffn(out2)
            res = self.ln3(out2 + ffn_out)
            
        # Cast output back to the parent layer's expected compute_dtype (which is float16 for the mixed-precision model)
        return tf.cast(res, self.compute_dtype)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim":    self.embed_dim,
            "dense_dim":    self.dense_dim,
            "num_heads":    self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package="Custom")
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.dropout_rate = float(dropout_rate)
        
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dtype="float32")
        self.ffn = keras.Sequential([
            layers.Dense(dense_dim, activation="relu", dtype="float32"),
            layers.Dense(embed_dim, dtype="float32"),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.dropout1 = layers.Dropout(dropout_rate, dtype="float32")
        self.dropout2 = layers.Dropout(dropout_rate, dtype="float32")

    def call(self, inputs, training=False):
        inputs_f32 = tf.cast(inputs, tf.float32)
        attn = self.mha(query=inputs_f32, value=inputs_f32, key=inputs_f32)
        attn = self.dropout1(attn, training=training)
        out1 = self.ln1(inputs_f32 + attn)
        
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        res = self.ln2(out1 + ffn_out)
        return res

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


@keras.utils.register_keras_serializable(package="Custom")
class SpatialTransformerEncoder(layers.Layer):
    def __init__(self, num_layers=2, num_heads=4, embed_dim=512, dense_dim=1024, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dropout_rate = float(dropout_rate)
        
        self.enc_blocks = [
            TransformerEncoderBlock(embed_dim, dense_dim, num_heads, dropout_rate, name=f"enc_block_{i}")
            for i in range(num_layers)
        ]
        
    def build(self, input_shape):
        num_positions = input_shape[1]
        self.pos_emb = self.add_weight(
            name="pos_embedding",
            shape=(1, num_positions, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            dtype="float32"
        )
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32) + tf.cast(self.pos_emb, tf.float32)
        for block in self.enc_blocks:
            x = block(x, training=training)
        return tf.cast(x, self.compute_dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


@keras.utils.register_keras_serializable()
class WarmupLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Stub so the v2 .keras file deserialises without error (not used at inference)."""
    def __init__(self, d_model=512, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)
        self.d_model      = float(d_model)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step + 1.0)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}


# ============================================================
# 4. RESOURCE LOADING WITH STREAMLIT CACHING (PER MODEL PATH)
# ============================================================

@st.cache_resource
def load_model_and_vocab(model_path: str, vocab_path: str):
    """Loads and caches a model+vocab pair. Cached separately per model_path."""
    if not os.path.exists(vocab_path):
        return None, None, f"Vocabulary file missing:\n{vocab_path}"
    if not os.path.exists(model_path):
        return None, None, f"Model file missing:\n{model_path}"
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Unified custom_objects — the registered PositionalEmbedding and
        # TransformerDecoderBlock classes now handle both v1 (dropout_rate=0)
        # and v2 (dropout_rate>0) configs automatically.
        custom_objects = {
            "PositionalEmbedding":            PositionalEmbedding,
            "TransformerDecoderBlock":        TransformerDecoderBlock,
            "TransformerEncoderBlock":        TransformerEncoderBlock,
            "SpatialTransformerEncoder":      SpatialTransformerEncoder,
            "Custom>PositionalEmbedding":     PositionalEmbedding,
            "Custom>TransformerDecoderBlock": TransformerDecoderBlock,
            "Custom>TransformerEncoderBlock": TransformerEncoderBlock,
            "Custom>SpatialTransformerEncoder": SpatialTransformerEncoder,
            "WarmupLRSchedule":               WarmupLRSchedule,
        }

        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return model, vocab, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


@st.cache_resource
def load_chexagent_model(model_dir: str):
    """Loads and caches CheXagent-8b model and processor."""
    if not os.path.exists(model_dir):
        return None, None, f"CheXagent directory not found: {model_dir}"
        
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        gen_config = GenerationConfig.from_pretrained(model_dir)
        gen_config.max_new_tokens = 160
        gen_config.do_sample = False
        gen_config.num_beams = 1
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True
        )
        return (model, processor, gen_config), None, None
    except Exception as e:
        return None, None, f"Error loading CheXagent: {str(e)}"


def generate_report_chexagent(model_tuple, image_bytes, prompt="Describe the findings and impression for this chest radiograph."):
    """Generates a clinical radiology report from image bytes using CheXagent-8b."""
    import io
    import torch
    model, processor, gen_config = model_tuple
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    formatted_prompt = f" USER: <s>{prompt} ASSISTANT: <s>"
    
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = processor(images=[image], text=formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(target_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=gen_config)[0]
        
    raw_response = processor.tokenizer.decode(output, skip_special_tokens=True)
    report_clean = raw_response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in raw_response else raw_response.strip()
    return report_clean



# ============================================================
# 5. PREPROCESSING & INFERENCE FUNCTIONS
# ============================================================

def preprocess_image_bytes(image_bytes):
    """
    Decodes raw image bytes and applies training preprocessing:
    Grayscale → Resize (224,224) → CLAHE → Normalize → 3 Channels
    """
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not decode image. Please check the file format.")

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = image.astype(np.float32) / 255.0
    image_3ch = np.stack([image, image, image], axis=-1)
    return keras.applications.densenet.preprocess_input(image_3ch)


def extract_model_components(model):
    """
    Extracts visual encoder, positional embedding, decoder blocks, and output layer
    to allow single-pass image encoding and sub-second token decoding for all Keras models.
    """
    try:
        pos_emb_layer = None
        for layer in model.layers:
            if "pos" in layer.name.lower():
                pos_emb_layer = layer
                break

        dec_blocks = [l for l in model.layers if isinstance(l, TransformerDecoderBlock) or "decoder" in l.name.lower()]
        
        vocab_layer = None
        for layer in reversed(model.layers):
            if "vocab" in layer.name.lower():
                vocab_layer = layer
                break
        if vocab_layer is None:
            for layer in reversed(model.layers):
                if isinstance(layer, keras.layers.Dense) and layer.output_shape[-1] > 100:
                    vocab_layer = layer
                    break

        encoder_layer = None
        for layer in model.layers:
            name_lower = layer.name.lower()
            if any(k in name_lower for k in ["encoder", "densenet", "efficient", "cnn", "spatial", "backbone"]):
                if not any(k in name_lower for k in ["decoder", "vocab", "pos"]):
                    encoder_layer = layer
                    break

        if pos_emb_layer and dec_blocks and vocab_layer and encoder_layer:
            return encoder_layer, pos_emb_layer, dec_blocks, vocab_layer
    except Exception:
        pass
    return None, None, None, None


def generate_report(model, vocab, preprocessed_image, temperature=0.4, penalty=1.2, no_repeat_ngram_size=4, min_len=10):
    """
    Generates a diagnostic report token-by-token using temperature-scaled sampling,
    incorporating repetition penalties, n-gram blocking, and minimum length constraints.
    Fast execution: Uses pre-computed visual encoder features for sub-second generation.
    """
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}

    start_id = token_to_id.get(START_TOKEN)
    end_id   = token_to_id.get(END_TOKEN)
    unk_id   = token_to_id.get("[UNK]")

    if start_id is None or end_id is None:
        raise ValueError("Vocabulary is missing START or END tokens.")

    image_input = np.expand_dims(preprocessed_image, axis=0)
    input_shape = model.inputs[1].shape
    max_seq_len = input_shape[1] if len(input_shape) > 1 and input_shape[1] is not None else 59

    encoder_layer, pos_emb_layer, dec_blocks, vocab_layer = extract_model_components(model)
    use_fast_pipeline = encoder_layer is not None

    if use_fast_pipeline:
        img_features = encoder_layer(image_input, training=False)

    seq = [start_id]
    for _ in range(max_seq_len):
        padded_seq = keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=max_seq_len, padding="post"
        )
        if use_fast_pipeline:
            x = pos_emb_layer(padded_seq, training=False)
            for dec_block in dec_blocks:
                x = dec_block(x, img_features, training=False)
            logits_tensor = vocab_layer(x, training=False)
            probs = logits_tensor[0, len(seq) - 1, :].numpy()
        else:
            preds_raw = model([image_input, padded_seq], training=False)
            if isinstance(preds_raw, (list, tuple)):
                preds_raw = preds_raw[0]
            preds = preds_raw.numpy() if hasattr(preds_raw, "numpy") else np.array(preds_raw)
            probs = preds[0, len(seq) - 1, :]

        probs = np.array(probs, dtype=np.float32)

        # Apply repetition penalty to already generated tokens
        for token_id in set(seq):
            if token_id != 0:
                probs[token_id] /= penalty

        # Apply soft no-repeat n-gram suppression
        if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
            ngram_size = no_repeat_ngram_size
            suffix = seq[-(ngram_size - 1):]
            for i in range(len(seq) - ngram_size + 1):
                ngram = seq[i : i + ngram_size]
                if ngram[:-1] == suffix:
                    forbidden_token = ngram[-1]
                    probs[forbidden_token] *= 0.01

        # Apply min length constraint by penalizing END and pad tokens
        if len(seq) < min_len:
            probs[end_id] = 0.0
            probs[0] = 0.0

        # Normalize
        sum_probs = np.sum(probs)
        if sum_probs > 0:
            probs = probs / sum_probs
        else:
            probs = np.ones_like(probs) / len(probs)

        if temperature <= 0.0:
            next_id = int(np.argmax(probs))
        else:
            logits      = np.log(np.maximum(probs, 1e-10)) / temperature
            exp_logits  = np.exp(logits - np.max(logits))
            probs_scaled = exp_logits / np.sum(exp_logits)
            next_id     = int(np.random.choice(len(probs), p=probs_scaled))

        if next_id == end_id or next_id == 0:
            break
        seq.append(next_id)

    words = [id_to_token.get(i, "") for i in seq if i not in (0, start_id, end_id, unk_id)]
    report_text = " ".join([w for w in words if w])
    return report_text if report_text else "No report generated."


def generate_report_beam(model, vocab, preprocessed_image, beam_width=3, length_penalty=0.6, min_len=10, no_repeat_ngram_size=4):
    """
    Generates a diagnostic report using Beam Search with length normalization, min-length constraint, loop prevention,
    and batched parallel candidate evaluation with cached visual encoder features.
    """
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}

    start_id = token_to_id.get(START_TOKEN)
    end_id   = token_to_id.get(END_TOKEN)
    unk_id   = token_to_id.get("[UNK]")

    if start_id is None or end_id is None:
        raise ValueError("Vocabulary is missing START or END tokens.")

    image_input = np.expand_dims(preprocessed_image, axis=0)
    input_shape = model.inputs[1].shape
    max_seq_len = input_shape[1] if len(input_shape) > 1 and input_shape[1] is not None else 59

    encoder_layer, pos_emb_layer, dec_blocks, vocab_layer = extract_model_components(model)
    use_fast_pipeline = encoder_layer is not None

    if use_fast_pipeline:
        img_features = encoder_layer(image_input, training=False)

    # Each beam candidate: (score, sequence, finished)
    beams = [(0.0, [start_id], False)]
    completed = []

    for step in range(max_seq_len):
        # Check if all beams are finished
        if not beams or all(finished for _, _, finished in beams):
            break

        # Filter active (unfinished) beams
        active_beams = [(idx, score, seq) for idx, (score, seq, finished) in enumerate(beams) if not finished]
        
        # Build batched sequences for all active beams at once
        active_seqs = [seq for _, _, seq in active_beams]
        padded_seqs = keras.preprocessing.sequence.pad_sequences(
            active_seqs, maxlen=max_seq_len, padding="post"
        )

        if use_fast_pipeline:
            batch_img_features = np.repeat(img_features.numpy(), len(active_seqs), axis=0)
            x = pos_emb_layer(padded_seqs, training=False)
            for dec_block in dec_blocks:
                x = dec_block(x, batch_img_features, training=False)
            logits_tensor = vocab_layer(x, training=False)
            preds_all = logits_tensor.numpy()
        else:
            batch_images = np.repeat(image_input, len(active_seqs), axis=0)
            preds_raw = model([batch_images, padded_seqs], training=False)
            if isinstance(preds_raw, (list, tuple)):
                preds_raw = preds_raw[0]
            preds_all = preds_raw.numpy() if hasattr(preds_raw, "numpy") else np.array(preds_raw)

        candidates = []

        # Retain already finished beams as candidates
        for score, seq, finished in beams:
            if finished:
                candidates.append((score, seq, True))

        # Expand active beams with output probabilities
        for batch_idx, (_, score, seq) in enumerate(active_beams):
            probs = preds_all[batch_idx, len(seq) - 1, :]
            probs = np.array(probs, dtype=np.float32)

            # Apply min length constraint: penalize END and pad tokens until min_len reached
            if len(seq) < min_len:
                probs[end_id] = 0.0
                probs[0] = 0.0

            # Apply soft no-repeat n-gram suppression
            if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
                ngram_size = no_repeat_ngram_size
                suffix = seq[-(ngram_size - 1):]
                for i in range(len(seq) - ngram_size + 1):
                    ngram = seq[i : i + ngram_size]
                    if ngram[:-1] == suffix:
                        forbidden_token = ngram[-1]
                        probs[forbidden_token] *= 0.01

            # Normalize probabilities
            sum_probs = np.sum(probs)
            if sum_probs > 0:
                probs = probs / sum_probs
            else:
                probs = np.ones_like(probs) / len(probs)

            probs = np.maximum(probs, 1e-10)
            top_ids = np.argsort(probs)[::-1][:beam_width]

            for next_id in top_ids:
                # Trigram repetition penalty
                rep_penalty = 0.0
                if len(seq) >= 3 and next_id == seq[-1] == seq[-2] == seq[-3]:
                    rep_penalty = -10.0
                # Bigram repetition penalty
                if len(seq) >= 4 and [seq[-2], seq[-1]] == [seq[-4], seq[-3]]:
                    rep_penalty = -5.0

                next_score = score + np.log(probs[next_id]) + rep_penalty
                next_seq = seq + [int(next_id)]

                is_end = (next_id == end_id) or (next_id == 0)
                candidates.append((next_score, next_seq, is_end))

        # Sort candidates by length-normalized score
        def get_norm_score(item):
            score, seq, finished = item
            seq_len = len(seq) - 1
            if seq_len <= 0:
                return score
            lp_denom = ((5.0 + seq_len) / 6.0) ** length_penalty
            return score / lp_denom

        candidates = sorted(candidates, key=get_norm_score, reverse=True)
        beams = candidates[:beam_width]

        # Move finished beams to completed list
        new_beams = []
        for score, seq, finished in beams:
            if finished:
                completed.append((score, seq, True))
            else:
                new_beams.append((score, seq, False))
        beams = new_beams

        if len(completed) >= beam_width:
            break

    all_final = completed + beams
    all_final = sorted(all_final, key=get_norm_score, reverse=True)
    best_seq = all_final[0][1]

    words = [id_to_token.get(i, "") for i in best_seq if i not in (0, start_id, end_id, unk_id)]
    report_text = " ".join([w for w in words if w])
    return report_text if report_text else "No report generated."


def extract_clinical_concept(sentence: str) -> dict:
    """
    Parses a clinical sentence into a structured concept dictionary:
    {
        "class": pathology_class,
        "laterality": LEFT / RIGHT / BILATERAL / UNSPECIFIED,
        "polarity": POSITIVE / NEGATIVE,
        "severity": MILD / MODERATE / SEVERE / UNSPECIFIED,
        "original_text": sentence
    }
    """
    text_lower = sentence.lower().strip()

    # 1. Detect Polarity (Negation check)
    is_negated = bool(re.search(r"\b(no|without|absent|negative for|free of|denies)\b", text_lower))
    polarity = "NEGATIVE" if is_negated else "POSITIVE"

    # 2. Detect Laterality
    if "bilateral" in text_lower or "both" in text_lower:
        laterality = "BILATERAL"
    elif "left" in text_lower:
        laterality = "LEFT"
    elif "right" in text_lower:
        laterality = "RIGHT"
    else:
        laterality = "UNSPECIFIED"

    # 3. Detect Severity
    if re.search(r"\b(mild|slight|small)\b", text_lower):
        severity = "MILD"
    elif re.search(r"\b(severe|marked|large|massive)\b", text_lower):
        severity = "SEVERE"
    elif "moderate" in text_lower:
        severity = "MODERATE"
    else:
        severity = "UNSPECIFIED"

    # 4. Detect Pathology Class
    if re.search(r"\b(cardiomegaly|enlarged heart|heart is enlarged|big heart|heart is big|cardiac silhouette is enlarged|enlarged cardiac silhouette|enlargement of the cardiac silhouette|cardiac size is enlarged)\b", text_lower):
        path_class = "CARDIOMEGALY"
    elif re.search(r"\b(pleural effusion|effusion|fluid around|pleural fluid)\b", text_lower):
        path_class = "PLEURAL_EFFUSION"
    elif re.search(r"\b(pneumothorax|air around|air in pleural)\b", text_lower):
        path_class = "PNEUMOTHORAX"
    elif re.search(r"\b(atelectasis|collapsed lung|lung collapse|atelectatic)\b", text_lower):
        path_class = "ATELECTASIS"
    elif re.search(r"\b(pneumonia|lung infection|infection in lung|focal pneumonia)\b", text_lower):
        path_class = "PNEUMONIA"
    elif re.search(r"\b(pulmonary edema|edema|vascular congestion|pulmonary congestion|pulmonary venous pressure)\b", text_lower):
        path_class = "EDEMA"
    elif re.search(r"\b(opacity|opacities|cloudy|hazy|infiltrate|consolidation)\b", text_lower):
        path_class = "OPACITY"
    elif re.search(r"\b(normal limits|unremarkable|clear lungs|normal heart|no acute|normal cardiac)\b", text_lower):
        path_class = "NORMAL"
    else:
        path_class = f"GENERIC_{text_lower}"

    return {
        "class": path_class,
        "laterality": laterality,
        "polarity": polarity,
        "severity": severity,
        "original_text": sentence
    }


def consolidate_multi_view_reports(raw_reports: list) -> str:
    """
    Consolidates raw diagnostic text observations from 1 to 3 chest X-ray views using conservative
    medical concept-based deduplication.
    
    Integrates Priority 1 clean_raw_report() preamble/header cleaning, maps synonymous clinical
    expressions to unified medical concepts, deduplicates redundant findings across views,
    and strictly preserves important clinical distinctions (laterality, severity, and positive vs negative polarities).
    """
    if not raw_reports:
        return "No diagnostic findings generated."
    
    # 1. Clean each raw report using Priority 1 clean_raw_report() if available
    cleaned_reports = []
    for r in raw_reports:
        if not r or not isinstance(r, str):
            continue
        cleaned = clean_raw_report(r) if clean_raw_report is not None else r.strip()
        if cleaned and cleaned != "No report generated.":
            cleaned_reports.append(cleaned)
            
    if not cleaned_reports:
        return "No diagnostic findings generated."
    if len(cleaned_reports) == 1:
        return cleaned_reports[0]

    # 2. Extract and categorize clinical sentences across all views
    concept_map = {}
    ordered_concepts = []

    for view_idx, report in enumerate(cleaned_reports):
        sentences = [s.strip() for s in re.split(r'[.;]\s*', report) if s.strip()]
        
        for sentence in sentences:
            concept = extract_clinical_concept(sentence)
            key = (concept["class"], concept["laterality"], concept["polarity"], concept["severity"])
            
            if key not in concept_map:
                concept_map[key] = {
                    "representative_text": sentence,
                    "views": [view_idx + 1],
                    "concept": concept
                }
                ordered_concepts.append(key)
            else:
                concept_map[key]["views"].append(view_idx + 1)

    if not ordered_concepts:
        return "The chest X-ray views are unremarkable."

    # 3. Assemble consolidated sentences
    consolidated_sentences = [concept_map[k]["representative_text"] for k in ordered_concepts]
    
    consolidated_text = ". ".join(consolidated_sentences)
    if not consolidated_text.endswith("."):
        consolidated_text += "."
    return consolidated_text


# ============================================================
# 6b. NLP METRICS (no external library required)
# ============================================================

def _clean_report(text: str) -> str:
    """Strip special tokens and extra whitespace from a report string."""
    text = re.sub(r'<start>|<end>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _tokenize(text: str):
    return text.lower().split()


def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> dict:
    """Compute BLEU-1 through BLEU-4 without external libraries."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens or not ref_tokens:
        return {f'bleu{n}': 0.0 for n in range(1, max_n+1)}

    scores = {}
    for n in range(1, max_n+1):
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        hyp_ngrams = _ngrams(hyp_tokens, n)
        if not hyp_ngrams:
            scores[f'bleu{n}'] = 0.0
            continue
        clip_count = sum(min(count, ref_ngrams[gram]) for gram, count in Counter(hyp_ngrams).items())
        precision = clip_count / len(hyp_ngrams)
        # brevity penalty
        bp = math.exp(1 - len(ref_tokens)/len(hyp_tokens)) if len(hyp_tokens) < len(ref_tokens) else 1.0
        scores[f'bleu{n}'] = round(bp * precision * 100, 2)
    return scores


def rouge_l_score(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 using LCS."""
    ref  = _tokenize(reference)
    hyp  = _tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    # LCS via DP
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if ref[i-1]==hyp[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0
    recall    = lcs / m if m else 0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
    return round(f1 * 100, 2)


@st.cache_data
def load_validation_csv(csv_path: str):
    """Load the validation CSV once and cache it."""
    try:
        df = pd.read_csv(csv_path)
        df['report_clean'] = df['report'].apply(_clean_report)
        return df, None
    except Exception as e:
        return None, str(e)


PATHOLOGY_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

def extract_pathology_predictions(model, prep_img):
    """Extracts 14 DenseNet121 pathology probabilities from model output if available."""
    try:
        img_tensor = np.expand_dims(prep_img, axis=0)
        dummy_seq = np.zeros((1, 127), dtype=np.int32)
        model_outs = model([img_tensor, dummy_seq], training=False)
        if isinstance(model_outs, dict):
            class_probs = model_outs.get("class_output", [None])[0]
        elif isinstance(model_outs, (list, tuple)) and len(model_outs) > 1:
            class_probs = model_outs[1][0]
        else:
            class_probs = None
        if class_probs is not None:
            probs = class_probs.numpy().flatten()
            return {PATHOLOGY_CLASSES[i]: float(probs[i]) for i in range(min(len(PATHOLOGY_CLASSES), len(probs)))}
    except Exception:
        pass
    return {}



# ============================================================
# 7. STREAMLIT USER INTERFACE
# ============================================================

VAL_CSV_PATH = os.path.join(
    BASE_DIR, "datasets", "mimic_cxr", "processed", "mimic_cxr_processed_val.csv"
)


def main():
    # ── Header ──────────────────────────────────────────────
    st.markdown('<div class="main-header">🩺 Generative Learning for Medical Image to Diagnostic Text</div>', unsafe_allow_html=True)


    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.header("🤖 Model Selection")

        selected_label = st.radio(
            "Choose a trained model:",
            options=list(MODEL_REGISTRY.keys()),
            index=0,
            help="Each model was trained on a different dataset or architecture. Switch to compare outputs."
        )

        selected = MODEL_REGISTRY[selected_label]
        model_path = selected["model_path"]
        vocab_path = selected["vocab_path"]

        st.markdown("---")

        # ── Model Info Card ─────────────────────────────────
        st.subheader("📋 Model Info")
        st.markdown(f'<div class="model-badge">{selected_label}</div>', unsafe_allow_html=True)
        
        val_acc_str = selected['val_accuracy']
        epochs_str  = selected['epochs']
        model_dir   = os.path.dirname(selected['model_path']) if selected['model_path'] else ""
        meta_file   = os.path.join(model_dir, "training_metadata.json")
        hist_file   = os.path.join(model_dir, "training_history.csv")
        
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as mf:
                    metadata = json.load(mf)
                    ep = metadata.get("overall_epoch", "")
                    if ep:
                        epochs_str = f"{ep} Epochs"
            except Exception:
                pass
                
        if os.path.exists(hist_file):
            try:
                hdf = pd.read_csv(hist_file)
                if 'val_class_output_auc' in hdf.columns and not hdf['val_class_output_auc'].dropna().empty:
                    best_auc = hdf['val_class_output_auc'].dropna().max() * 100.0
                    val_vocab_acc = hdf['val_vocab_output_accuracy'].dropna().iloc[-1] * 100.0 if 'val_vocab_output_accuracy' in hdf.columns else 0.0
                    val_acc_str = f"{best_auc:.2f}% Pathology AUC ({val_vocab_acc:.2f}% Vocab Acc)"
                elif 'val_accuracy' in hdf.columns and not hdf['val_accuracy'].dropna().empty:
                    best_acc = hdf['val_accuracy'].dropna().max() * 100.0
                    val_acc_str = f"{best_acc:.2f}%"
            except Exception:
                pass

        st.markdown(f"""
        | Property | Value |
        |---|---|
        | **Architecture** | {selected['architecture']} |
        | **Dataset** | {selected['dataset']} |
        | **Epochs** | {epochs_str} |
        | **Val Accuracy** | {val_acc_str} |
        """)
        st.caption(selected["description"])

        st.markdown("---")

        # ── Load Model ──────────────────────────────────────
        st.subheader("⚙️ System Status")
        is_vlm = selected.get("is_vlm", False)
        with st.spinner("Loading model..."):
            if is_vlm:
                model_obj, _, err = load_chexagent_model(model_path)
                model = model_obj
                vocab = None
            else:
                model, vocab, err = load_model_and_vocab(model_path, vocab_path)

        if err:
            st.error(f"❌ Model Offline\n\n{err}")
        else:
            st.success("✅ Model Ready")
            if is_vlm:
                st.info("• Engine: PyTorch + CUDA\n• Architecture: CheXzero + Mistral\n• Precision: 16-bit Auto Offload")
            else:
                st.info(f"• Vocab: {len(vocab):,} tokens\n• Input: 224×224 CLAHE\n• Decoding: Temperature sampling")

        st.markdown("---")

        # ── Generation Settings ─────────────────────────────
        st.subheader("⚙️ Generation Settings")
        decoding_strategy = st.radio(
            "Decoding Strategy",
            options=["Greedy / Sampling", "Beam Search"],
            index=0,
            help="Greedy/Sampling uses temperature scaling. Beam Search explores multiple word paths to generate higher quality text."
        )

        beam_width = 3
        gen_temp = 0.4
        if decoding_strategy == "Beam Search":
            beam_width = st.slider(
                "Beam Width",
                min_value=1, max_value=5, value=3, step=1,
                help="Number of active paths to keep in beam search. 3 and 5 are standard."
            )
        else:
            gen_temp = st.slider(
                "Sampling Temperature",
                min_value=0.0, max_value=1.0, value=0.4, step=0.1,
                help="0.0 = greedy (deterministic). Higher = more varied output."
            )

        st.markdown("---")

        # ── Chest Validation Gate Settings ──────────────────
        st.subheader("🛡️ Chest Validation Gate")
        chest_threshold = st.slider(
            "Chest Confidence Threshold (Validation-Calibrated)",
            min_value=0.50, max_value=0.95, value=float(DEFAULT_CHEST_THRESHOLD), step=0.05,
            help="Validation-calibrated experimental threshold (0.60) required to classify the image as a Chest X-Ray before running report generation."
        )

        st.markdown("---")

        # ── LLM Refinement Engine Settings ───────────────────
        st.subheader("💡 LLM Report Refinement Engine")
        llm_provider = st.selectbox(
            "Report Enhancement Engine",
            options=["Auto (Gemini / Groq / BioBERT Fallback)", "Gemini API (Google Free Key)", "Groq API (Llama 3.3 70B)", "Local BioBERT Engine (Offline)"],
            index=0,
            help="Select the LLM provider to refine report fluency and patient explanations."
        )

        user_api_key_input = st.text_input(
            "API Key (Optional for Gemini / Groq)",
            type="password",
            value=st.session_state.get("user_api_key", ""),
            help="Paste a free Google AI Studio key (AIza...) or Groq key (gsk_...) for enhanced LLM report generation."
        )
        if user_api_key_input:
            st.session_state["user_api_key"] = user_api_key_input.strip()
        elif "user_api_key" in st.session_state and not user_api_key_input:
            st.session_state["user_api_key"] = ""

        st.markdown("---")

        # ── Quick Action ────────────────────────────────────

        st.subheader("💡 Quick Actions")
        use_sample = False
        if os.path.exists(DEFAULT_SAMPLE_IMAGE):
            if st.button("📂 Load Sample X-ray (xray1.jpg)"):
                use_sample = True

    # ══════════════════════════════════════════════════════════
    # Generate Report
    # ══════════════════════════════════════════════════════════
    col1, col2 = st.columns([1, 1], gap="large")

    image_bytes = None
    image_bytes_list = []
    val_details = None
    is_chest = False

    with col1:
        st.subheader("1️⃣ Upload Chest X-Ray (1 to 3 Images)")
        uploaded_files = st.file_uploader(
            "Select Chest X-ray image(s) (JPG, JPEG, PNG, DCM)",
            type=["jpg", "jpeg", "png", "dcm"],
            accept_multiple_files=True,
            help="Upload up to 3 chest X-ray images (e.g. Frontal, Lateral views) or DICOM files.",
            key="uploader_generate",
        )

        if use_sample and os.path.exists(DEFAULT_SAMPLE_IMAGE):
            with open(DEFAULT_SAMPLE_IMAGE, "rb") as f:
                sample_b = f.read()
                image_bytes_list.append(sample_b)
                image_bytes = sample_b
            st.info("Loaded sample image: `xray1.jpg`")
        elif uploaded_files:
            if len(uploaded_files) > 3:
                st.warning("⚠️ Maximum 3 X-ray images allowed. Processing only the first 3 uploaded images.")
                valid_uploads = uploaded_files[:3]
            else:
                valid_uploads = uploaded_files

            for uf in valid_uploads:
                image_bytes_list.append(uf.getvalue())

            if image_bytes_list:
                image_bytes = image_bytes_list[0]

            # Render preview for each uploaded image with filename
            st.markdown(f"**Uploaded Image(s): {len(image_bytes_list)} / 3**")
            cols = st.columns(len(image_bytes_list))
            for i, (uf, b) in enumerate(zip(valid_uploads, image_bytes_list)):
                with cols[i]:
                    st.image(b, caption=f"Image {i+1}: {uf.name}", use_container_width=True)

        multi_val_results = []
        all_chest_valid = True
        failed_images = []

        if image_bytes_list:
            # ── 1. UNIFESP X-RAY BODY-PART CLASSIFICATION & CHEST VALIDATION GATE (ALL IMAGES) ──
            if validate_chest_xray is not None:
                with st.spinner("🔍 Running UNIFESP Body Part Classification on ALL uploaded images..."):
                    for idx, img_b in enumerate(image_bytes_list):
                        fname = valid_uploads[idx].name if (uploaded_files and idx < len(valid_uploads) and hasattr(valid_uploads[idx], "name")) else f"xray_{idx+1}.jpg"
                        is_c, c_conf, det_lbl, v_details = validate_chest_xray(
                            img_b, threshold=chest_threshold
                        )
                        v_details["filename"] = fname
                        v_details["index"] = idx + 1
                        multi_val_results.append((is_c, c_conf, det_lbl, v_details))
                        
                        is_unc = v_details.get("is_uncertain", False)
                        if not is_c or is_unc:
                            all_chest_valid = False
                            reason = v_details.get("uncertainty_reason", "Low confidence margin") if is_unc else f"Classified as '{det_lbl}' instead of Chest"
                            failed_images.append({
                                "index": idx + 1,
                                "filename": fname,
                                "detected": det_lbl,
                                "confidence": v_details.get("confidence_percentage", 0),
                                "reason": reason
                            })

                # Render UNIFESP Body Part Classification Stage Summary for all images
                with st.expander(f"📊 UNIFESP Body Part Validation Summary ({len(image_bytes_list)} Image(s))", expanded=True):
                    val_summary_rows = []
                    for is_c, c_conf, det_lbl, v_details in multi_val_results:
                        status_label = "🟢 CHEST (PASSED)" if (is_c and not v_details.get("is_uncertain")) else ("⚠️ UNCERTAIN" if v_details.get("is_uncertain") else "🔴 REJECTED")
                        val_summary_rows.append({
                            "Image #": f"Image {v_details['index']}",
                            "Filename": v_details["filename"],
                            "Predicted Region": det_lbl,
                            "Chest Confidence": f"{v_details['confidence_percentage']}%",
                            "Status": status_label
                        })
                    st.dataframe(pd.DataFrame(val_summary_rows), hide_index=True, use_container_width=True)

                # STRICT REJECTION GATE: If ANY image is non-chest or uncertain, reject the ENTIRE group
                if not all_chest_valid:
                    failed_details_html = "".join([
                        f"• <b>{f['filename']}</b> (Image {f['index']}): Classified as <b>{f['detected']}</b> ({f['confidence']}% confidence) — <i>{f['reason']}</i><br>"
                        for f in failed_images
                    ])
                    st.markdown(f"""
                        <div style="background-color: #FEF2F2; border-left: 6px solid #DC2626; padding: 1.2rem; border-radius: 8px; margin-top: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                            <h3 style="color: #991B1B; margin-top: 0; margin-bottom: 0.5rem;">🔴 Multi-View Validation Gate Failed — Entire Upload Group Rejected</h3>
                            <p style="font-size: 1.05rem; color: #7F1D1D; font-weight: 700; margin-bottom: 0.4rem;">
                                This application is specialized exclusively for Chest X-Rays. Every uploaded image must pass anatomical validation.
                            </p>
                            <div style="background-color: #FFFFFF; border: 1px solid #FCA5A5; border-radius: 6px; padding: 0.8rem; margin: 0.6rem 0;">
                                <p style="font-size: 0.95rem; color: #334155; margin: 0;">
                                    <b>Non-Chest or Uncertain Image(s) Identified:</b><br>
                                    {failed_details_html}
                                </p>
                            </div>
                            <p style="font-size: 0.95rem; color: #1E293B; font-weight: 600; margin-top: 0.6rem; margin-bottom: 0;">
                                Report generation has been halted to prevent misdiagnosis. Please remove non-chest image(s) and upload only valid chest X-rays.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.error("⚠️ Multi-View Gate Rejection: One or more uploaded images failed chest validation. Report generation halted.")
                    st.stop()
                else:
                    st.markdown(f"""
                        <div style="background-color: #F0FDF4; border-left: 6px solid #16A34A; padding: 1rem; border-radius: 8px; margin-top: 1rem; margin-bottom: 1rem;">
                            <h4 style="color: #166534; margin: 0;">🟢 All {len(image_bytes_list)} Chest X-Ray View(s) Validated (UNIFESP Gate Passed)</h4>
                            <p style="font-size: 0.95rem; color: #15803D; font-weight: 600; margin-top: 0.3rem; margin-bottom: 0;">
                                Every uploaded image was verified as a valid Chest X-Ray. Proceeding to multi-view feature analysis & consolidated report generation...
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

    with col2:
        st.subheader("2️⃣ Diagnostic Report Generation")

        if model is None:
            st.error(f"Cannot generate — model failed to load:\n{err}")
        else:
            st.caption(f"Active model: **{selected_label}**  |  Architecture: {selected['architecture']}")

            generate_btn = st.button("🔬 Generate Diagnostic Report", type="primary", use_container_width=True, key="btn_generate")

            if generate_btn or (image_bytes_list and use_sample):
                if not image_bytes_list:
                    st.warning("Please upload at least one Chest X-ray image first.")
                elif not all_chest_valid:
                    st.error("⚠️ Anatomical Gate Halted Execution (Non-Chest or Uncertain Image). Report generation stopped.")
                    st.stop()
                else:
                    try:
                        per_image_raw_reports = []
                        per_image_details = []
                        all_pathology_preds = {}

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, img_b in enumerate(image_bytes_list):
                            fname = valid_uploads[idx].name if (uploaded_files and idx < len(valid_uploads) and hasattr(valid_uploads[idx], "name")) else f"Image {idx+1}"
                            status_text.markdown(f"🔬 Analyzing view **{idx+1} of {len(image_bytes_list)}** (`{fname}`)...")

                            v_details = multi_val_results[idx][3] if idx < len(multi_val_results) else {}
                            input_bytes = img_b
                            if v_details and v_details.get("is_dicom") and v_details.get("display_image") is not None:
                                _, encoded_img = cv2.imencode('.jpg', cv2.cvtColor(v_details["display_image"], cv2.COLOR_RGB2BGR))
                                input_bytes = encoded_img.tobytes()

                            if is_vlm:
                                raw_rep = generate_report_chexagent(model, input_bytes)
                            else:
                                prep_img = preprocess_image_bytes(input_bytes)
                                # Extract DenseNet121 14-pathology probabilities
                                view_path_preds = extract_pathology_predictions(model, prep_img)
                                for k, v in view_path_preds.items():
                                    all_pathology_preds[k] = max(all_pathology_preds.get(k, 0.0), v)

                                if decoding_strategy == "Beam Search":
                                    raw_rep = generate_report_beam(model, vocab, prep_img, beam_width=beam_width)
                                else:
                                    raw_rep = generate_report(model, vocab, prep_img, temperature=gen_temp)

                            per_image_raw_reports.append(raw_rep)
                            per_image_details.append({
                                "view_num": idx + 1,
                                "filename": fname,
                                "raw_report": raw_rep
                            })
                            progress_bar.progress((idx + 1) / len(image_bytes_list))

                        status_text.success(f"✅ Analyzed **{len(image_bytes_list)} of {len(image_bytes_list)}** chest X-ray views successfully!")

                        # ── Multi-View Report Consolidation ──
                        consolidated_raw_text = consolidate_multi_view_reports(per_image_raw_reports)

                        # ── Dual Report Generation & BioBERT Refinement ──
                        refined_result = None
                        if refine_report:
                            with st.spinner("Consolidating multi-view findings & running BioBERT Biomedical Quality Control Check..."):
                                refined_result = refine_report(consolidated_raw_text, pathology_predictions=all_pathology_preds)

                        # Render View-by-View Observations Expander
                        with st.expander(f"📑 View-by-View Raw Observations ({len(image_bytes_list)} Views)", expanded=False):
                            for det in per_image_details:
                                st.markdown(f"**Image {det['view_num']} (`{det['filename']}`)**: *\"{det['raw_report']}\"*")

                        st.markdown("---")
                        st.markdown(f"### 🔬 CONSOLIDATED CHEST X-RAY ANALYSIS ({len(image_bytes_list)} View{'s' if len(image_bytes_list)>1 else ''} Combined)")
                        st.markdown(f"""
                            <div class="report-card">
                                <div class="report-title">Combined Findings (Multi-View Raw Output):</div>
                                <div class="report-text">"{consolidated_raw_text}"</div>
                            </div>
                        """, unsafe_allow_html=True)

                        if refined_result:
                            st.markdown("---")
                            st.markdown("### 🟢 PATIENT-FRIENDLY CONSOLIDATED REPORT")
                            st.markdown(f"""
                                <div class="report-card-actual">
                                    <div class="report-title">Patient-Friendly Simple Explanation:</div>
                                    <div style="font-size: 1.1rem; color: #166534; line-height: 1.6; font-weight: 500;">
                                        {refined_result['simple_report']}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                            st.markdown("---")
                            st.markdown("### 🩺 PROFESSIONAL RADIOLOGY CONSOLIDATED REPORT")
                            st.markdown(f"""
                                <div class="report-card-generated">
                                    <div class="report-title">CONSOLIDATED FINDINGS ({len(image_bytes_list)} View{'s' if len(image_bytes_list)>1 else ''}):</div>
                                    <div style="font-size: 1.05rem; color: #1E293B; margin-bottom: 1rem; line-height: 1.5;">
                                        {refined_result['findings']}
                                    </div>
                                    <div class="report-title">CONSOLIDATED IMPRESSION:</div>
                                    <div style="font-size: 1.05rem; color: #1E3A8A; font-weight: 600; line-height: 1.5; white-space: pre-line;">
                                        {refined_result['impression']}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                            # ── BioBERT Biomedical NLP & Entity Analysis Section ──
                            bio_analysis = refined_result.get("biobert_analysis")
                            if bio_analysis:
                                st.markdown("---")
                                st.markdown("### 🧬 BIOBERT BIOMEDICAL NLP & RADIOLOGY FINDINGS ANALYSIS")
                                
                                summary = bio_analysis.get("summary", {})
                                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                                with b_col1:
                                    st.markdown(f'<div class="metric-box"><div class="metric-value">{summary.get("positive_count", 0)}</div><div class="metric-label">🟢 Positive Findings</div></div>', unsafe_allow_html=True)
                                with b_col2:
                                    st.markdown(f'<div class="metric-box"><div class="metric-value">{summary.get("negative_count", 0)}</div><div class="metric-label">🔴 Negative Findings</div></div>', unsafe_allow_html=True)
                                with b_col3:
                                    st.markdown(f'<div class="metric-box"><div class="metric-value">{summary.get("uncertain_count", 0)}</div><div class="metric-label">🟡 Uncertain Findings</div></div>', unsafe_allow_html=True)
                                with b_col4:
                                    st.markdown(f'<div class="metric-box"><div class="metric-value">{len(bio_analysis.get("ner_entities", []))}</div><div class="metric-label">🏷️ NER Entities</div></div>', unsafe_allow_html=True)

                                # BioBERT Embeddings & NER Expander
                                with st.expander("🔬 BioBERT Contextual Embeddings & Pretrained NER Details (dmis-lab/biobert-base-cased-v1.2)", expanded=False):
                                    emb = bio_analysis.get("biobert_embeddings", {})
                                    st.markdown(f"- **BioBERT Model Checkpoint**: `dmis-lab/biobert-base-cased-v1.2`")
                                    st.markdown(f"- **Biomedical NER Checkpoint**: `d4data/biomedical-ner-all`")
                                    st.markdown(f"- **Contextual Token Count**: `{emb.get('token_count', 0)}` tokens")
                                    st.markdown(f"- **BioBERT Embedding Tensor Shape**: `{emb.get('embedding_shape', [1, 0, 768])}` (768-dimensional token vectors)")
                                    st.markdown(f"- **Pooled Embedding L2-Norm**: `{emb.get('mean_embedding_norm', 0.0)}`")
                                    
                                    ner_ents = bio_analysis.get("ner_entities", [])
                                    if ner_ents:
                                        st.markdown("#### 🏷️ Extracted Biomedical Entities (Pretrained NER):")
                                        st.dataframe(pd.DataFrame(ner_ents), hide_index=True, use_container_width=True)

                                # BioBERT Structured Findings Table
                                findings_list = bio_analysis.get("structured_findings", [])
                                if findings_list:
                                    with st.expander("📋 BioBERT Extracted Radiology Findings (Status, Polarity, Laterality, Severity)", expanded=True):
                                        f_table = []
                                        for f in findings_list:
                                            status_val = f.get("status", "PRESENT" if f["polarity"] == "POSITIVE" else ("ABSENT" if f["polarity"] == "NEGATIVE" else "UNCERTAIN"))
                                            stat_icon = "🟢 PRESENT" if status_val == "PRESENT" else ("🔴 ABSENT" if status_val == "ABSENT" else "🟡 UNCERTAIN")
                                            f_table.append({
                                                "Finding #": f["id"],
                                                "Pathology": f["pathology"],
                                                "Clinical Status": stat_icon,
                                                "Certainty": f["certainty"],
                                                "Laterality": f["laterality"],
                                                "Severity": f["severity"],
                                                "Original Sentence": f["sentence"]
                                            })
                                        st.dataframe(pd.DataFrame(f_table), hide_index=True, use_container_width=True)

                                # DenseNet121 vs BioBERT Verification Card
                                densenet_ver = bio_analysis.get("densenet_verification", {})
                                if densenet_ver:
                                    v_score = densenet_ver.get("consistency_score")
                                    if v_score is None:
                                        v_badge = "ℹ️ UNEVALUATED (No pathology head active)"
                                        score_str = "N/A (Active model does not output 14-pathology probabilities)"
                                    else:
                                        v_badge = "🟢 HIGH CONCORDANCE" if v_score >= 90 else ("🟡 MODERATE CONCORDANCE" if v_score >= 70 else "🔴 INCONSISTENCY FLAGGED")
                                        score_str = f"{v_score}%"

                                    with st.expander(f"⚖️ DenseNet121 14-Pathology vs BioBERT Verification ({v_badge})", expanded=True):
                                        st.markdown(f"**BioBERT vs DenseNet121 Consistency Score**: `{score_str}`")
                                        if densenet_ver.get("status_text"):
                                            st.caption(densenet_ver["status_text"])
                                        
                                        flags = densenet_ver.get("flagged_issues", [])
                                        if flags:
                                            st.warning("⚠️ **Potential Inconsistencies Flagged (Findings preserved without deletion):**")
                                            for flg in flags:
                                                st.markdown(f"- **{flg['issue_type']}** [{flg['severity']} SEVERITY]: {flg['description']}")
                                        elif v_score is not None:
                                            st.success("✅ No inconsistencies detected between DenseNet121 14-pathology predictions and BioBERT report findings.")
                                        
                                        comp_matrix = densenet_ver.get("comparison_matrix", [])
                                        if comp_matrix:
                                            st.markdown("#### 📊 14-Pathology Comparison Matrix:")
                                            st.dataframe(pd.DataFrame(comp_matrix), hide_index=True, use_container_width=True)

                            val = refined_result["validation"]
                            densenet_ver = bio_analysis.get("densenet_verification", {}) if bio_analysis else {}
                            has_pathology_mismatches = densenet_ver.get("inconsistencies_found", False)
                            is_semantic_valid = val.get("is_valid", True)
                            
                            if has_pathology_mismatches:
                                qc_badge = f"⚠️ REVIEW REQUIRED ({densenet_ver.get('flag_count', 0)} Pathology Mismatch(es) Flagged)"
                            elif not is_semantic_valid:
                                qc_badge = f"⚠️ ATTENTION (Semantic Score: {val['score']}%)"
                            else:
                                qc_badge = "✅ PASS (All Safety Checks Verified)"

                            with st.expander(f"🛡️ Quality Control & Multi-View Safety Check ({qc_badge})", expanded=has_pathology_mismatches):
                                st.markdown(f"- **Engine**: `{refined_result['engine_used']}`")
                                st.markdown(f"- **Views Analyzed**: `{len(image_bytes_list)} of {len(image_bytes_list)} chest X-ray images`")
                                st.markdown("---")
                                st.markdown("#### 📌 Validation Status Breakdown:")
                                st.markdown(f"- **1. Image Validation (UNIFESP Gate)**: `🟢 PASSED` ({len(image_bytes_list)} Chest X-Ray View(s) Validated)")
                                
                                sem_status = f"`🟢 PASSED` ({val['score']}% Semantic Match - No Hallucinations)" if is_semantic_valid else f"`⚠️ ATTENTION` ({val['score']}% Semantic Match)"
                                st.markdown(f"- **2. Report Semantic Consistency**: {sem_status}")
                                
                                v_score = densenet_ver.get("consistency_score")
                                if v_score is None:
                                    concord_status = "`ℹ️ UNEVALUATED` (No pathology head active for selected model)"
                                elif has_pathology_mismatches:
                                    concord_status = f"`🔴 REVIEW REQUIRED` ({densenet_ver.get('flag_count', 0)} Mismatch(es) Flagged, {v_score}% Concordance)"
                                else:
                                    concord_status = f"`🟢 CONCORDANT` ({v_score}% Concordance across evaluated pathologies)"
                                st.markdown(f"- **3. DenseNet121-BioBERT Pathology Concordance**: {concord_status}")

                                if has_pathology_mismatches:
                                    st.warning(f"⚠️ **Clinician Review Required**: {densenet_ver.get('flag_count', 0)} potential mismatch(es) detected between DenseNet121 visual predictions and BioBERT report analysis.")
                                    for flg in densenet_ver.get("flagged_issues", []):
                                        st.markdown(f"  • **{flg['pathology']}**: {flg['description']}")

                                if val.get("flagged_issues"):
                                    st.warning("Semantic Refinement Flagged Issues: " + "; ".join(val["flagged_issues"]))

                        st.success(f"✅ Single Consolidated Report for {len(image_bytes_list)} chest X-ray view(s) generated successfully!")

                        with st.expander("ℹ️ About this result"):
                            dec_str = f"Beam Search (Width {beam_width})" if decoding_strategy == "Beam Search" else f"Greedy / Sampling (Temp {gen_temp})"
                            st.markdown(f"""
                            - **Model used**: {selected_label}
                            - **Architecture**: {selected['architecture']}
                            - **Trained on**: {selected['dataset']}
                            - **Epochs**: {selected['epochs']}
                            - **Val Accuracy**: {selected['val_accuracy']}
                            - **Decoding Strategy**: {dec_str}
                            """)

                    except Exception as e:
                        st.error(f"An error occurred during report generation:\n{str(e)}")



if __name__ == "__main__":
    main()
