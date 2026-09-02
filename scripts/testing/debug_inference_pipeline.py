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

# Custom layers
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

def preprocess_image_file(image_path):
    if not os.path.exists(image_path) or os.path.getsize(image_path) < 100:
        raise ValueError(f"File missing or invalid: {image_path}")
    buf = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to decode: {image_path}")
    
    raw_min = float(image.min())
    raw_max = float(image.max())
    raw_mean = float(image.mean())

    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_enhanced = clahe.apply(image_resized)

    image_normalized = image_enhanced.astype(np.float32) / 255.0
    tensor_3ch = np.stack([image_normalized, image_normalized, image_normalized], axis=-1)
    
    return image, raw_min, raw_max, raw_mean, tensor_3ch

def main():
    print("=" * 80)
    print("INFERENCE PIPELINE DEEP DEBUGGING & VERIFICATION")
    print("=" * 80)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Locate 2 distinct test X-ray images
    images_found = []
    iccv_dir = os.path.join(project_dir, "official_data_iccv_final", "files", "p10")
    if os.path.exists(iccv_dir):
        for root, _, files in os.walk(iccv_dir):
            for f in files:
                if f.endswith(".jpg"):
                    cand = os.path.join(root, f)
                    if os.path.getsize(cand) > 100:
                        images_found.append(cand)
                    if len(images_found) >= 5:
                        break
            if len(images_found) >= 5:
                break

    if len(images_found) < 2 and os.path.exists("xray1.jpg"):
        images_found.append(os.path.abspath("xray1.jpg"))

    print(f"Found {len(images_found)} test image paths.")
    img_path_A = images_found[0]
    img_path_B = images_found[1]

    print(f"\n--- STEP 2: IMAGE PREPROCESSING DEBUG ---")
    raw_A, min_A, max_A, mean_A, tensor_A = preprocess_image_file(img_path_A)
    raw_B, min_B, max_B, mean_B, tensor_B = preprocess_image_file(img_path_B)

    print(f"IMAGE A: {os.path.basename(img_path_A)}")
    print(f"  • Raw Shape      : {raw_A.shape}")
    print(f"  • Min/Max/Mean   : {min_A:.1f} / {max_A:.1f} / {mean_A:.4f}")
    print(f"  • Tensor Shape   : {tensor_A.shape}")
    print(f"  • Tensor Mean    : {tensor_A.mean():.6f}")

    print(f"\nIMAGE B: {os.path.basename(img_path_B)}")
    print(f"  • Raw Shape      : {raw_B.shape}")
    print(f"  • Min/Max/Mean   : {min_B:.1f} / {max_B:.1f} / {mean_B:.4f}")
    print(f"  • Tensor Shape   : {tensor_B.shape}")
    print(f"  • Tensor Mean    : {tensor_B.mean():.6f}")

    pixel_diff = np.abs(tensor_A - tensor_B).mean()
    print(f"\nMean Absolute Pixel Difference between Tensor A and Tensor B: {pixel_diff:.6f}")
    assert pixel_diff > 1e-5, "ERROR: Image A and Image B produced identical image tensors!"
    print(" [OK] Image A and Image B produce distinct image tensors.")

    # 2. Model & Encoder Debug
    print(f"\n--- STEP 3 & 4: CNN ENCODER & DECODER FEATURE TRACING ---")
    model_path = os.path.join(project_dir, "results_model", "report_generator_model.keras")
    vocab_path = os.path.join(project_dir, "results_model", "tokenizer_vocab.json")

    custom_objects = {
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoderBlock": TransformerDecoderBlock,
        "Custom>PositionalEmbedding": PositionalEmbedding,
        "Custom>TransformerDecoderBlock": TransformerDecoderBlock
    }

    model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    print(f"Loaded Model : {model_path}")
    print(f"Model Summary inputs : {[i.name for i in model.inputs]}")

    # Extract Encoder Model/Sublayer
    encoder_layer = None
    for layer in model.layers:
        if "encoder" in layer.name or isinstance(layer, keras.Model):
            encoder_layer = layer
            break

    if encoder_layer is not None:
        print(f"Found Encoder Layer: {encoder_layer.name}")
        input_A = np.expand_dims(tensor_A, axis=0)
        input_B = np.expand_dims(tensor_B, axis=0)

        feat_A = encoder_layer(input_A).numpy()
        feat_B = encoder_layer(input_B).numpy()

        print(f"Feature A Shape : {feat_A.shape}")
        print(f"Feature A Mean  : {feat_A.mean():.6f} | Std: {feat_A.std():.6f}")
        print(f"Feature B Shape : {feat_B.shape}")
        print(f"Feature B Mean  : {feat_B.mean():.6f} | Std: {feat_B.std():.6f}")

        feat_diff = np.abs(feat_A - feat_B).mean()
        print(f"Mean Absolute Encoder Feature Difference (feat_A vs feat_B): {feat_diff:.6f}")
        print(f"Feature A sample (first 5 vals): {feat_A.flatten()[:5]}")
        print(f"Feature B sample (first 5 vals): {feat_B.flatten()[:5]}")
    else:
        print("  [Warning] Encoder is integrated into full Functional model graph.")

    # 3. Token Generation Debug
    print(f"\n--- STEP 5: TOKEN GENERATION & DECODER BEHAVIOR ---")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}
    start_id = token_to_id.get("<start>", 1)
    end_id = token_to_id.get("<end>", 2)

    max_seq_len = model.inputs[1].shape[1] if model.inputs[1].shape[1] is not None else 59

    def generate_tokens_and_logits(tensor_img):
        inp_img = np.expand_dims(tensor_img, axis=0)
        seq = [start_id]
        token_logits_log = []

        for step in range(max_seq_len):
            padded_seq = keras.preprocessing.sequence.pad_sequences(
                [seq], maxlen=max_seq_len, padding="post"
            )
            preds = model.predict([inp_img, padded_seq], verbose=0)
            step_logits = preds[0, len(seq) - 1, :]
            next_id = int(np.argmax(step_logits))
            
            top_3_ids = np.argsort(step_logits)[-3:][::-1]
            top_3_words = [(id_to_token.get(i, ""), float(step_logits[i])) for i in top_3_ids]
            token_logits_log.append((next_id, id_to_token.get(next_id, ""), top_3_words))

            if next_id == end_id or next_id == 0:
                break
            seq.append(next_id)

        report_words = [id_to_token.get(i, "") for i in seq if i not in (0, start_id, end_id)]
        return seq, " ".join([w for w in report_words if w]), token_logits_log

    seq_A, report_A, log_A = generate_tokens_and_logits(tensor_A)
    seq_B, report_B, log_B = generate_tokens_and_logits(tensor_B)

    print(f"\nIMAGE A Generated Tokens ({len(seq_A)} tokens):")
    print(f"  Token IDs: {seq_A[:10]}...")
    print(f"  Report   : {report_A}\n")

    print(f"IMAGE B Generated Tokens ({len(seq_B)} tokens):")
    print(f"  Token IDs: {seq_B[:10]}...")
    print(f"  Report   : {report_B}\n")

    print(f"\n--- LOGITS COMPARISON AT FIRST 3 STEPS ---")
    for step_idx in range(min(3, len(log_A), len(log_B))):
        print(f"Step {step_idx + 1}:")
        print(f"  Image A Top 3: {log_A[step_idx][2]}")
        print(f"  Image B Top 3: {log_B[step_idx][2]}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
