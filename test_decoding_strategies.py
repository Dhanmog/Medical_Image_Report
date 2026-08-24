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
    buf = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_enhanced = clahe.apply(image_resized)
    image_normalized = image_enhanced.astype(np.float32) / 255.0
    return np.stack([image_normalized, image_normalized, image_normalized], axis=-1)

def sample_with_temperature(probs, temperature=0.7):
    if temperature <= 0.0:
        return int(np.argmax(probs))
    logits = np.log(np.maximum(probs, 1e-10)) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    probs_scaled = exp_logits / np.sum(exp_logits)
    return int(np.random.choice(len(probs), p=probs_scaled))

def generate_report_temp(model, vocab, tensor_img, temperature=0.7):
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}
    start_id = token_to_id.get("<start>", 1)
    end_id = token_to_id.get("<end>", 2)

    max_seq_len = model.inputs[1].shape[1] if model.inputs[1].shape[1] is not None else 59
    image_input = np.expand_dims(tensor_img, axis=0)

    seq = [start_id]
    for step in range(max_seq_len):
        padded_seq = keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=max_seq_len, padding="post"
        )
        preds = model.predict([image_input, padded_seq], verbose=0)
        probs = preds[0, len(seq) - 1, :]
        next_id = sample_with_temperature(probs, temperature=temperature)
        if next_id == end_id or next_id == 0:
            break
        seq.append(next_id)

    words = [id_to_token.get(i, "") for i in seq if i not in (0, start_id, end_id)]
    return " ".join([w for w in words if w])

def main():
    print("=" * 80)
    print("TESTING INFERENCE SAMPLING & MODEL CHECKPOINTS")
    print("=" * 80)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Locate 5 distinct X-ray images
    images = []
    iccv_dir = os.path.join(project_dir, "official_data_iccv_final", "files", "p10")
    for root, _, files in os.walk(iccv_dir):
        for f in files:
            if f.endswith(".jpg") and os.path.getsize(os.path.join(root, f)) > 100:
                images.append(os.path.join(root, f))
                if len(images) == 5:
                    break
        if len(images) == 5:
            break

    custom_objects = {
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoderBlock": TransformerDecoderBlock,
        "Custom>PositionalEmbedding": PositionalEmbedding,
        "Custom>TransformerDecoderBlock": TransformerDecoderBlock
    }

    model_paths = [
        os.path.join(project_dir, "results_model", "report_generator_model.keras"),
        os.path.join(project_dir, "trained_models", "mimic_cxr", "mimic_cxr_model.keras")
    ]

    for mpath in model_paths:
        if not os.path.exists(mpath):
            continue
        print(f"\n==================================================")
        print(f"Testing Model: {os.path.basename(mpath)}")
        print(f"==================================================")
        
        vocab_path = os.path.join(os.path.dirname(mpath), "tokenizer_vocab.json")
        if not os.path.exists(vocab_path):
            vocab_path = os.path.join(project_dir, "results_model", "tokenizer_vocab.json")

        model = keras.models.load_model(mpath, custom_objects=custom_objects, compile=False)
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        for temp in [0.0, 0.4, 0.7]:
            print(f"\n--- Temperature = {temp} ---")
            reports = []
            for idx, img_p in enumerate(images, 1):
                tensor_img = preprocess_image_file(img_p)
                rep = generate_report_temp(model, vocab, tensor_img, temperature=temp)
                reports.append(rep)
                print(f"  Image #{idx} ({os.path.basename(img_p)[:15]}...): {rep[:90]}...")
            
            unique_reports = len(set(reports))
            print(f"  --> Unique Reports generated across 5 images: {unique_reports}/5")

if __name__ == "__main__":
    main()
