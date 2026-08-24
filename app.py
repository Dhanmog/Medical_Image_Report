import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Medical Image to Diagnostic Text",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
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
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. PATHS & CONFIGURATION
# ============================================================

MODEL_PAIRS = [
    (
        os.path.join("trained_models", "densenet121_mimic", "densenet121_mimic_model.keras"),
        os.path.join("trained_models", "densenet121_mimic", "tokenizer_vocab.json"),
    ),
    (
        os.path.join("trained_models", "mimic_cxr", "mimic_cxr_v2_model.keras"),
        os.path.join("trained_models", "mimic_cxr", "tokenizer_vocab.json"),
    ),
    (
        os.path.join("trained_models", "mimic_cxr", "checkpoints", "best_model.keras"),
        os.path.join("trained_models", "mimic_cxr", "tokenizer_vocab.json"),
    ),
    (
        os.path.join("results_model", "report_generator_model.keras"),
        os.path.join("results_model", "tokenizer_vocab.json"),
    ),
    (
        os.path.join("results_shenzhen", "shenzhen_model.keras"),
        os.path.join("results_shenzhen", "tokenizer_vocab.json"),
    ),
]

# Find first valid model + vocab pair
_valid_pair = next(((m, v) for m, v in MODEL_PAIRS if os.path.exists(m) and os.path.exists(v)), MODEL_PAIRS[0])
MODEL_PATH, VOCAB_PATH = _valid_pair[0], _valid_pair[1]

DEFAULT_SAMPLE_IMAGE = "xray1.jpg"

IMAGE_SIZE = (224, 224)
MAX_REPORT_LENGTH = 80
START_TOKEN = "<start>"
END_TOKEN = "<end>"

# ============================================================
# 3. CUSTOM MODEL LAYERS (REGISTERED FOR KERAS SERIALIZATION)
# ============================================================

@keras.utils.register_keras_serializable(package="Custom")
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(seq_len, embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_emb(inputs) + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "embed_dim": self.embed_dim,
        })
        return config


@keras.utils.register_keras_serializable(package="Custom")
class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.mha_causal = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.mha_cross = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()
        self.ln3 = layers.LayerNormalization()

    def call(self, inputs, encoder_outputs):
        attn1 = self.mha_causal(query=inputs, value=inputs, key=inputs, use_causal_mask=True)
        out1 = self.ln1(inputs + attn1)
        attn2 = self.mha_cross(query=out1, value=encoder_outputs, key=encoder_outputs)
        out2 = self.ln2(out1 + attn2)
        ffn_out = self.ffn(out2)
        return self.ln3(out2 + ffn_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config


if hasattr(keras, "saving") and hasattr(keras.saving, "register_keras_serializable"):
    try:
        keras.saving.register_keras_serializable(package="Custom")(PositionalEmbedding)
        keras.saving.register_keras_serializable(package="Custom")(TransformerDecoderBlock)
        keras.saving.register_keras_serializable()(PositionalEmbedding)
        keras.saving.register_keras_serializable()(TransformerDecoderBlock)
    except Exception:
        pass


# ============================================================
# 4. RESOURCE LOADING WITH STREAMLIT CACHING
# ============================================================

@st.cache_resource
def load_model_and_vocab():
    """Loads and caches the Transformer model and vocabulary JSON."""
    if not os.path.exists(VOCAB_PATH):
        return None, None, f"Vocabulary file missing at: {VOCAB_PATH}"
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Trained model missing at: {MODEL_PATH}"

    try:
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        custom_objects = {
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerDecoderBlock": TransformerDecoderBlock,
            "Custom>PositionalEmbedding": PositionalEmbedding,
            "Custom>TransformerDecoderBlock": TransformerDecoderBlock
        }

        model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        return model, vocab, None
    except Exception as e:
        return None, None, f"Error loading resources: {str(e)}"


# ============================================================
# 5. PREPROCESSING & INFERENCE FUNCTIONS
# ============================================================

def preprocess_image_bytes(image_bytes):
    """
    Decodes raw image bytes and applies exact training preprocessing:
    Grayscale -> Resize (224,224) -> CLAHE -> Normalize float32/255.0 -> 3 Channels
    """
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not decode image. Please check file format.")

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    image = image.astype(np.float32) / 255.0
    image = np.stack([image, image, image], axis=-1)
    return image


def generate_report(model, vocab, preprocessed_image, temperature=0.4):
    """
    Generates report token-by-token using temperature-scaled sampling.
    """
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}

    start_id = token_to_id.get(START_TOKEN)
    end_id = token_to_id.get(END_TOKEN)

    if start_id is None or end_id is None:
        raise ValueError("Vocabulary missing START or END tokens.")

    image_input = np.expand_dims(preprocessed_image, axis=0)

    # Dynamically detect max sequence length expected by model input 1
    input_shape = model.inputs[1].shape
    max_seq_len = input_shape[1] if len(input_shape) > 1 and input_shape[1] is not None else 59

    seq = [start_id]
    for step in range(max_seq_len):
        padded_seq = keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=max_seq_len, padding="post"
        )
        preds = model.predict([image_input, padded_seq], verbose=0)
        probs = preds[0, len(seq) - 1, :]

        if temperature <= 0.0:
            next_id = int(np.argmax(probs))
        else:
            logits = np.log(np.maximum(probs, 1e-10)) / temperature
            exp_logits = np.exp(logits - np.max(logits))
            probs_scaled = exp_logits / np.sum(exp_logits)
            next_id = int(np.random.choice(len(probs), p=probs_scaled))

        if next_id == end_id or next_id == 0:
            break
        seq.append(next_id)

    words = [id_to_token.get(i, "") for i in seq if i not in (0, start_id, end_id)]
    report_text = " ".join([w for w in words if w])
    return report_text if report_text else "No report generated."


# ============================================================
# 6. STREAMLIT USER INTERFACE
# ============================================================

def main():
    st.markdown('<div class="main-header">Generative Learning for Medical Image to Diagnostic Text</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">DenseNet121 + MIMIC-CXR Diagnostic Report Generator</div>', unsafe_allow_html=True)

    # Mandatory Medical Disclaimer
    st.warning("⚠️ **Research & Educational Disclaimer**: This application is built for research and demonstration purposes only. The generated text is produced by a generative AI model and must NOT be used for clinical medical diagnosis or patient treatment.")

    # Sidebar Information
    with st.sidebar:
        st.header("⚙️ System Status")
        model, vocab, err = load_model_and_vocab()
        if err:
            st.error(f"Status: Offline\n{err}")
        else:
            st.success("Status: Model Ready")
            st.info(f"• Vocab Size: {len(vocab)} tokens\n• Model: DenseNet121 + Transformer\n• Dataset: MIMIC-CXR (4000 Subset)\n• Input: 224x224 (3-channel CLAHE)")

        st.markdown("---")
        st.subheader("⚙️ Generation Settings")
        gen_temp = st.slider(
            "Sampling Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Higher values allow distinct X-ray features to generate image-specific reports. 0.0 uses greedy decoding."
        )

        st.markdown("---")
        st.subheader("💡 Quick Actions")
        use_sample = False
        if os.path.exists(DEFAULT_SAMPLE_IMAGE):
            if st.button("Load Sample X-ray (xray1.jpg)"):
                use_sample = True

    # Main Grid Layout (2 Columns)
    col1, col2 = st.columns([1, 1], gap="large")

    uploaded_file = None
    image_bytes = None

    with col1:
        st.subheader("1. Upload Chest X-Ray")
        uploaded_file = st.file_uploader(
            "Select a Chest X-ray image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a standard frontal chest X-ray image."
        )

        if use_sample and os.path.exists(DEFAULT_SAMPLE_IMAGE):
            with open(DEFAULT_SAMPLE_IMAGE, "rb") as f:
                image_bytes = f.read()
            st.info(f"Loaded sample image: `{DEFAULT_SAMPLE_IMAGE}`")
        elif uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()

        if image_bytes is not None:
            st.image(image_bytes, caption="Input Chest X-Ray", use_container_width=True)

    with col2:
        st.subheader("2. Diagnostic Report Generation")

        if model is None:
            st.error(f"Cannot generate report because model loading failed:\n{err}")
            return

        generate_btn = st.button("🔬 Generate Diagnostic Report", type="primary", use_container_width=True)

        if generate_btn or (image_bytes is not None and use_sample):
            if image_bytes is None:
                st.warning("Please upload a Chest X-ray image first.")
            else:
                try:
                    with st.spinner("Preprocessing X-ray image (CLAHE & Normalization)..."):
                        preprocessed_img = preprocess_image_bytes(image_bytes)

                    with st.spinner("Generating diagnostic report with Transformer model..."):
                        report_text = generate_report(model, vocab, preprocessed_img, temperature=gen_temp)

                    st.markdown("""
                        <div class="report-card">
                            <div class="report-title">📋 Generated Diagnostic Report:</div>
                            <div class="report-text">"{}"</div>
                        </div>
                    """.format(report_text), unsafe_allow_html=True)

                    st.success("Report generated successfully!")

                except Exception as e:
                    st.error(f"An error occurred during report generation: {str(e)}")


if __name__ == "__main__":
    main()
