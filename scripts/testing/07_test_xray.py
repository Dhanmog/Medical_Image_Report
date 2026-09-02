import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
EMBED_DIM = 256
NUM_HEADS = 4
FF_DIM = 512
MAX_REPORT_LENGTH = 60

MODEL_PATH = os.path.join("results_shenzhen", "shenzhen_model.keras")
VOCAB_PATH = os.path.join("results_shenzhen", "tokenizer_vocab.json")

# Test X-ray
IMAGE_PATH = "xray1.jpg"

START_TOKEN = "<start>"
END_TOKEN = "<end>"

# ============================================================
# 2. PREPROCESSING USED DURING TRAINING
# ============================================================

def preprocess_xray(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not find or open image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # CLAHE preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    image = image.astype(np.float32) / 255.0

    # Convert grayscale -> 3 channels
    image = np.stack([image, image, image], axis=-1)

    return image


# ============================================================
# 3. CUSTOM LAYERS USED DURING TRAINING
# ============================================================

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


# ============================================================
# 4. MAIN TESTING PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("SHENZHEN CHEST X-RAY DIAGNOSTIC REPORT GENERATION")
    print("=" * 70)

    # --- Step 1: Load Vocabulary ---
    print("\n[1/5] Loading vocabulary...")
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_PATH}")

    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    print(f"Vocabulary loaded successfully. Size: {len(vocab)}")

    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}

    start_id = token_to_id.get(START_TOKEN)
    end_id = token_to_id.get(END_TOKEN)

    print(f"START token '{START_TOKEN}' ID: {start_id}")
    print(f"END token '{END_TOKEN}' ID  : {end_id}")

    if start_id is None:
        raise ValueError(f"'{START_TOKEN}' token was not found in tokenizer_vocab.json")
    if end_id is None:
        raise ValueError(f"'{END_TOKEN}' token was not found in tokenizer_vocab.json")

    # --- Step 2: Load Model ---
    print("\n[2/5] Loading trained Shenzhen model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    custom_objects = {
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoderBlock": TransformerDecoderBlock
    }

    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects,
        compile=False
    )
    print("Model loaded successfully!")

    # --- Step 3: Load Test Image & Display Properties ---
    print("\n[3/5] Loading test X-ray image...")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"X-ray image not found: {IMAGE_PATH}")

    original_image = Image.open(IMAGE_PATH)
    print(f"Image size: {original_image.size}")
    print(f"Image mode: {original_image.mode}")

    plt.figure(figsize=(6, 6))
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")
    plt.title("Test Shenzhen Chest X-ray")
    preview_filename = "test_xray_preview.png"
    plt.savefig(preview_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved image preview to: {preview_filename}")

    # --- Step 4: Preprocess Image ---
    print("\n[4/5] Preprocessing X-ray image...")
    image = preprocess_xray(IMAGE_PATH)
    image_input = np.expand_dims(image, axis=0)
    print(f"Preprocessed image batch shape: {image_input.shape}")

    # --- Step 5: Report Generation ---
    print("\n[5/5] Generating diagnostic report...")
    print("-" * 70)

    sequence = [start_id]

    for step in range(MAX_REPORT_LENGTH - 2):
        padded_sequence = keras.preprocessing.sequence.pad_sequences(
            [sequence],
            maxlen=MAX_REPORT_LENGTH - 1,
            padding="post"
        )

        predictions = model.predict([image_input, padded_sequence], verbose=0)
        current_position = len(sequence) - 1
        next_id = int(np.argmax(predictions[0, current_position, :]))

        if next_id == end_id or next_id == 0:
            break

        sequence.append(next_id)

    # Map Token IDs to Words
    generated_words = []
    for token_id in sequence:
        if token_id in (0, start_id, end_id):
            continue
        token = id_to_token.get(token_id, "")
        if token:
            generated_words.append(token)

    generated_report = " ".join(generated_words)

    # Print Final Generated Report
    print("\n" + "=" * 70)
    print("GENERATED DIAGNOSTIC REPORT:")
    print("=" * 70)
    if generated_report.strip():
        print(generated_report)
    else:
        print("[No text generated by model]")
    print("=" * 70)
    print("\nTesting completed successfully!")

if __name__ == "__main__":
    main()