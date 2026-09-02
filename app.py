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
    "🌟 Stanford CheXagent-8b (SOTA Foundation VLM)": {
        "model_path":  os.path.join(BASE_DIR, "models", "05_chexagent_8b"),
        "vocab_path":  "",
        "is_vlm":       True,
        "architecture": "Vision-Language Foundation Model (CheXzero + Mistral-LLM)",
        "dataset":      "CheXbench / MIMIC-CXR (Millions of CXRs)",
        "epochs":       "Pre-trained SOTA Foundation Model",
        "val_accuracy": "SOTA (RadGraph F1 ~0.38)",
        "description":  "Stanford AIMI Foundation Model for Chest X-Ray interpretation. Generates comprehensive clinical Findings and Impression sections with radiology-grade precision.",
    },
    "⚡ EfficientNetB0 v2  (Best Model)": {
        "model_path":  os.path.join(BASE_DIR, "models", "01_efficientnetb0_mimic_v2", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "01_efficientnetb0_mimic_v2", "tokenizer_vocab.json"),
        "architecture": "EfficientNetB0 + Transformer Decoder",
        "dataset":      "MIMIC-CXR (Preprocessed ~6k/epoch)",
        "epochs":       "2 epochs",
        "val_accuracy": "~30.1%",
        "description":  "Fine-tuned EfficientNetB0 encoder with cross-attention Transformer decoder. Best general-purpose model.",
    },
    "🏆 EfficientNetB0 Best Checkpoint": {
        "model_path":  os.path.join(BASE_DIR, "models", "01_efficientnetb0_mimic_v2", "best_model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "01_efficientnetb0_mimic_v2", "tokenizer_vocab.json"),
        "architecture": "EfficientNetB0 + Transformer Decoder",
        "dataset":      "MIMIC-CXR (Preprocessed ~6k/epoch)",
        "epochs":       "2 epochs (best val_loss checkpoint)",
        "val_accuracy": "~30.1%",
        "description":  "Saved at the epoch with lowest validation loss. Identical architecture to v2 model.",
    },
    "🔬 DenseNet121 MIMIC": {
        "model_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic", "tokenizer_vocab.json"),
        "architecture": "DenseNet121 + Transformer Decoder",
        "dataset":      "MIMIC-CXR (4,000 subset)",
        "epochs":       "5 epochs",
        "val_accuracy": "~26.8%",
        "description":  "DenseNet121 encoder (no fine-tuning) with Transformer decoder, trained on a 4k image subset.",
    },
    "🧬 DenseNet121 v2 — Fine-tuned (Best DenseNet)": {
        "model_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v2", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v2", "tokenizer_vocab.json"),
        "architecture": "DenseNet121 + Transformer Decoder v2 (Fine-tuned)",
        "dataset":      "MIMIC-CXR (8,000 samples)",
        "epochs":       "Phase 1 (7 ep, early stop) + Phase 2 (fine-tune top 50 layers)",
        "val_accuracy": "~28.41%",
        "description":  "Improved DenseNet121 with 2-block Transformer decoder, GELU FFN, dropout regularisation. "
                        "Phase 2 fine-tuned top 50 DenseNet layers at LR=1e-5. Best DenseNet checkpoint saved by ModelCheckpoint.",
    },
    "🚀 DenseNet121 v3 — Fine-tuned (Max Accuracy)": {
        "model_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v3", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v3", "tokenizer_vocab.json"),
        "architecture": "DenseNet121 + 3-Block Transformer Decoder (Fine-tuned)",
        "dataset":      "MIMIC-CXR (50,000 samples)",
        "epochs":       "Phase 1 (8 ep) + Phase 2 (7 ep) + Phase 3 (6 ep)",
        "val_accuracy": "~26.49%",
        "description":  "Advanced DenseNet121 v3 fine-tuned on a large 50,000 image subset. Features a deeper 3-block Transformer decoder, mixed-precision training, and a 3-phase training process with all backbone layers unfrozen in the final phase.",
    },
    "🪐 DenseNet121 v4 — Highres 14x14 (Fine-tuning)": {
        "model_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v4_highres", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "02_densenet121_mimic_v4_highres", "tokenizer_vocab.json"),
        "architecture": "DenseNet121 (14x14 pool4_relu) + Transformer Decoder",
        "dataset":      "MIMIC-CXR (Preprocessed ~50k)",
        "epochs":       "Phase 1 (10 ep) + Phase 2 (15 ep) + Phase 3 (8 ep)",
        "val_accuracy": "~55.78%",
        "description":  "New 14x14 feature map resolution architecture targeting fine-grained structures and device placement.",
    },
    "📦 CNN-Transformer MIMIC (5 epochs)": {
        "model_path":  os.path.join(BASE_DIR, "models", "03_cnn_transformer_mimic", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "03_cnn_transformer_mimic", "tokenizer_vocab.json"),
        "architecture": "CNN Encoder + Transformer Decoder",
        "dataset":      "MIMIC-CXR (Full processed)",
        "epochs":       "5 epochs",
        "val_accuracy": "~68.9% (token-level)",
        "description":  "Fully trained 5-epoch model on MIMIC-CXR. Highest token-level training accuracy.",
    },
    "🫁 Shenzhen TB Specialist": {
        "model_path":  os.path.join(BASE_DIR, "models", "04_cnn_transformer_shenzhen", "model.keras"),
        "vocab_path":  os.path.join(BASE_DIR, "models", "04_cnn_transformer_shenzhen", "tokenizer_vocab.json"),
        "architecture": "CNN Encoder + Transformer Decoder",
        "dataset":      "Shenzhen TB Chest X-ray",
        "epochs":       "5 epochs",
        "val_accuracy": "~68.9%",
        "description":  "Specialist model trained exclusively on Shenzhen tuberculosis chest X-ray dataset.",
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


def generate_report(model, vocab, preprocessed_image, temperature=0.4, penalty=1.2, no_repeat_ngram_size=4, min_len=10):
    """
    Generates a diagnostic report token-by-token using temperature-scaled sampling,
    incorporating repetition penalties, n-gram blocking, and minimum length constraints.
    """
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}

    start_id = token_to_id.get(START_TOKEN)
    end_id   = token_to_id.get(END_TOKEN)

    if start_id is None or end_id is None:
        raise ValueError("Vocabulary is missing START or END tokens.")

    image_input = np.expand_dims(preprocessed_image, axis=0)
    input_shape = model.inputs[1].shape
    max_seq_len = input_shape[1] if len(input_shape) > 1 and input_shape[1] is not None else 59

    seq = [start_id]
    for _ in range(max_seq_len):
        padded_seq = keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=max_seq_len, padding="post"
        )
        preds_raw = model.predict([image_input, padded_seq], verbose=0)
        preds = preds_raw[0] if isinstance(preds_raw, (list, tuple)) else preds_raw
        probs = preds[0, len(seq) - 1, :]
        probs = np.array(probs, dtype=np.float32)

        # Apply repetition penalty to already generated tokens
        for token_id in set(seq):
            if token_id != 0:
                probs[token_id] /= penalty

        # Apply no-repeat n-gram blocking
        if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
            ngram_size = no_repeat_ngram_size
            suffix = seq[-(ngram_size - 1):]
            for i in range(len(seq) - ngram_size + 1):
                ngram = seq[i : i + ngram_size]
                if ngram[:-1] == suffix:
                    forbidden_token = ngram[-1]
                    probs[forbidden_token] = 0.0

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

    words = [id_to_token.get(i, "") for i in seq if i not in (0, start_id, end_id)]
    report_text = " ".join([w for w in words if w])
    return report_text if report_text else "No report generated."


def generate_report_beam(model, vocab, preprocessed_image, beam_width=3, length_penalty=0.6, min_len=10, no_repeat_ngram_size=4):
    """
    Generates a diagnostic report using Beam Search with length normalization, min-length constraint, and loop prevention.
    """
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}

    start_id = token_to_id.get(START_TOKEN)
    end_id   = token_to_id.get(END_TOKEN)

    if start_id is None or end_id is None:
        raise ValueError("Vocabulary is missing START or END tokens.")

    image_input = np.expand_dims(preprocessed_image, axis=0)
    input_shape = model.inputs[1].shape
    max_seq_len = input_shape[1] if len(input_shape) > 1 and input_shape[1] is not None else 59

    # Each beam candidate: (score, sequence, finished)
    beams = [(0.0, [start_id], False)]
    completed = []

    for step in range(max_seq_len):
        candidates = []
        all_finished = True
        for score, seq, finished in beams:
            if not finished:
                all_finished = False
                break
        if all_finished:
            break

        for score, seq, finished in beams:
            if finished:
                candidates.append((score, seq, True))
                continue

            padded_seq = keras.preprocessing.sequence.pad_sequences(
                [seq], maxlen=max_seq_len, padding="post"
            )
            preds_raw = model.predict([image_input, padded_seq], verbose=0)
            preds = preds_raw[0] if isinstance(preds_raw, (list, tuple)) else preds_raw
            probs = preds[0, len(seq) - 1, :]
            probs = np.array(probs, dtype=np.float32)

            # Apply min length constraint: penalize END and pad tokens until min_len reached
            if len(seq) < min_len:
                probs[end_id] = 0.0
                probs[0] = 0.0

            # Apply no-repeat n-gram blocking
            if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
                ngram_size = no_repeat_ngram_size
                suffix = seq[-(ngram_size - 1):]
                for i in range(len(seq) - ngram_size + 1):
                    ngram = seq[i : i + ngram_size]
                    if ngram[:-1] == suffix:
                        forbidden_token = ngram[-1]
                        probs[forbidden_token] = 0.0

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

    words = [id_to_token.get(i, "") for i in best_seq if i not in (0, start_id, end_id)]
    report_text = " ".join([w for w in words if w])
    return report_text if report_text else "No report generated."


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
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | **Architecture** | {selected['architecture']} |
        | **Dataset** | {selected['dataset']} |
        | **Epochs** | {selected['epochs']} |
        | **Val Accuracy** | {selected['val_accuracy']} |
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

    with col1:
        st.subheader("1️⃣ Upload Chest X-Ray")
        uploaded_file = st.file_uploader(
            "Select a Chest X-ray image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a standard frontal chest X-ray image.",
            key="uploader_generate",
        )

        if use_sample and os.path.exists(DEFAULT_SAMPLE_IMAGE):
            with open(DEFAULT_SAMPLE_IMAGE, "rb") as f:
                image_bytes = f.read()
            st.info("Loaded sample image: `xray1.jpg`")
        elif uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()

        if image_bytes is not None:
            st.image(image_bytes, caption="Input Chest X-Ray", use_container_width=True)

    with col2:
        st.subheader("2️⃣ Diagnostic Report Generation")

        if model is None:
            st.error(f"Cannot generate — model failed to load:\n{err}")
        else:
            st.caption(f"Active model: **{selected_label}**  |  Architecture: {selected['architecture']}")

            generate_btn = st.button("🔬 Generate Diagnostic Report", type="primary", use_container_width=True, key="btn_generate")

            if generate_btn or (image_bytes is not None and use_sample):
                if image_bytes is None:
                    st.warning("Please upload a Chest X-ray image first.")
                else:
                    try:
                        with st.spinner("Preprocessing X-ray..."):
                            if not is_vlm:
                                preprocessed_img = preprocess_image_bytes(image_bytes)

                        with st.spinner(f"Generating report using {selected['architecture']}..."):
                            if is_vlm:
                                report_text = generate_report_chexagent(model, image_bytes)
                            elif decoding_strategy == "Beam Search":
                                report_text = generate_report_beam(model, vocab, preprocessed_img, beam_width=beam_width)
                            else:
                                report_text = generate_report(model, vocab, preprocessed_img, temperature=gen_temp)

                        st.markdown("""
                            <div class="report-card">
                                <div class="report-title">📋 Generated Diagnostic Report:</div>
                                <div class="report-text">"{}"</div>
                            </div>
                        """.format(report_text), unsafe_allow_html=True)

                        st.success("✅ Report generated successfully!")

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
