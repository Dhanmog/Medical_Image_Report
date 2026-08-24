import os
import sys
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Custom layers with dual registration for Keras 2 & 3 compatibility
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

# Register with keras.saving if available in Keras 3
if hasattr(keras, "saving") and hasattr(keras.saving, "register_keras_serializable"):
    try:
        keras.saving.register_keras_serializable(package="Custom")(PositionalEmbedding)
        keras.saving.register_keras_serializable(package="Custom")(TransformerDecoderBlock)
        keras.saving.register_keras_serializable()(PositionalEmbedding)
        keras.saving.register_keras_serializable()(TransformerDecoderBlock)
    except Exception:
        pass

def get_custom_objects():
    return {
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoderBlock": TransformerDecoderBlock,
        "Custom>PositionalEmbedding": PositionalEmbedding,
        "Custom>TransformerDecoderBlock": TransformerDecoderBlock
    }

def test_model_loading_and_inference():
    print("=" * 70)
    print("TESTING MODEL LOADING & SINGLE-IMAGE INFERENCE")
    print("=" * 70)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths = [
        os.path.join(project_dir, "results_model", "report_generator_model.keras"),
        os.path.join(project_dir, "results_shenzhen", "shenzhen_model.keras"),
        os.path.join(project_dir, "trained_models", "mimic_cxr", "mimic_cxr_model.keras"),
        os.path.join(project_dir, "trained_models", "mimic_cxr", "mimic_cxr_checkpoint.keras")
    ]
    vocab_paths = [
        os.path.join(project_dir, "results_model", "tokenizer_vocab.json"),
        os.path.join(project_dir, "results_shenzhen", "tokenizer_vocab.json"),
        os.path.join(project_dir, "trained_models", "mimic_cxr", "tokenizer_vocab.json")
    ]

    chosen_model_path = next((p for p in model_paths if os.path.exists(p)), None)
    chosen_vocab_path = next((p for p in vocab_paths if os.path.exists(p)), None)

    if not chosen_model_path or not chosen_vocab_path:
        raise FileNotFoundError("Could not locate existing model or vocab file.")

    print(f"Loading Model Path: {chosen_model_path}")
    print(f"Loading Vocab Path: {chosen_vocab_path}")

    model = keras.models.load_model(chosen_model_path, custom_objects=get_custom_objects(), compile=False)
    print(" [OK] Model loaded successfully!")

    with open(chosen_vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f" [OK] Vocabulary loaded ({len(vocab)} tokens).")

    # Load 1 valid X-ray image (e.g. sample image or first image from official data)
    test_img_path = None
    iccv_files = os.path.join(project_dir, "official_data_iccv_final", "files", "p10")
    if os.path.exists(iccv_files):
        for root, _, files in os.walk(iccv_files):
            for f in files:
                if f.endswith(".jpg"):
                    cand = os.path.join(root, f)
                    if os.path.getsize(cand) > 100:
                        test_img_path = cand
                        break
            if test_img_path:
                break

    if not test_img_path and os.path.exists("xray1.jpg"):
        test_img_path = "xray1.jpg"

    if not test_img_path:
        print("  [Warning] No test image found, creating a dummy X-ray array for test.")
        image_np = np.zeros((224, 224, 3), dtype=np.float32)
    else:
        print(f"Loading test X-ray: {test_img_path}")
        buf = np.fromfile(test_img_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = img.astype(np.float32) / 255.0
        image_np = np.stack([img, img, img], axis=-1)

    # Perform greedy inference
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}
    start_id = token_to_id.get("<start>", 1)
    end_id = token_to_id.get("<end>", 2)

    image_input = np.expand_dims(image_np, axis=0)
    seq = [start_id]
    for _ in range(60 - 2):
        padded_seq = keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=60 - 1, padding="post"
        )
        preds = model.predict([image_input, padded_seq], verbose=0)
        next_id = int(np.argmax(preds[0, len(seq) - 1, :]))
        if next_id == end_id or next_id == 0:
            break
        seq.append(next_id)

    gen_tokens = [id_to_token.get(i, "") for i in seq[1:] if i not in (0, start_id, end_id)]
    generated_report = " ".join([w for w in gen_tokens if w])

    print("\nGENERATED DIAGNOSTIC REPORT:")
    print(f'"{generated_report}"')
    print("\n" + "=" * 70)
    print(" [OK] MODEL LOADING AND INFERENCE TEST PASSED WITH ZERO ERRORS!")
    print("=" * 70)

if __name__ == "__main__":
    test_model_loading_and_inference()
