import os
import sys
import json
import re
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, 'trained_models', 'densenet121_mimic', 'densenet121_mimic_model.keras')
VOCAB_PATH = os.path.join(PROJECT_DIR, 'trained_models', 'densenet121_mimic', 'tokenizer_vocab.json')
PROC_VAL_CSV = os.path.join(PROJECT_DIR, 'mimic_cxr_processed_val.csv')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'trained_models', 'densenet121_mimic')

IMAGE_SIZE = (224, 224)
MAX_REPORT_LENGTH = 80
START_TOKEN = "<start>"
END_TOKEN = "<end>"

@keras.utils.register_keras_serializable()
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.token_emb = keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_emb = keras.layers.Embedding(seq_len, embed_dim)

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

@keras.utils.register_keras_serializable()
class TransformerDecoderBlock(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.mha_causal = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.mha_cross = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(dense_dim, activation="relu"),
            keras.layers.Dense(embed_dim)
        ])
        self.ln1 = keras.layers.LayerNormalization()
        self.ln2 = keras.layers.LayerNormalization()
        self.ln3 = keras.layers.LayerNormalization()

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

def preprocess_xray(image_path):
    if not isinstance(image_path, str):
        image_path = str(image_path)
    try:
        if not os.path.exists(image_path) or os.path.getsize(image_path) < 100:
            return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
        img_np = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = image.astype(np.float32) / 255.0
        return np.stack([image, image, image], axis=-1)
    except Exception:
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

def clean_report_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9.,;:!?'\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_report(model, vocab, preprocessed_image):
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}

    start_id = token_to_id.get(START_TOKEN, 1)
    end_id = token_to_id.get(END_TOKEN, 2)

    image_input = np.expand_dims(preprocessed_image, axis=0)
    seq = [start_id]

    for _ in range(MAX_REPORT_LENGTH - 2):
        padded_seq = keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=MAX_REPORT_LENGTH - 1, padding="post"
        )
        preds = model.predict([image_input, padded_seq], verbose=0)
        next_id = int(np.argmax(preds[0, len(seq) - 1, :]))
        if next_id == end_id or next_id == 0:
            break
        seq.append(next_id)

    words = [id_to_token.get(idx, "") for idx in seq[1:] if idx in id_to_token and id_to_token[idx] not in [START_TOKEN, END_TOKEN, "<pad>"]]
    return " ".join([w for w in words if w])

def main():
    print("=" * 70)
    print("EVALUATING DENSENET121 MIMIC-CXR MODEL")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print(f"Model or vocab missing: {MODEL_PATH}")
        return

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    custom_objects = {
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoderBlock": TransformerDecoderBlock
    }

    model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    print("Loaded DenseNet121 model successfully.")

    val_df = pd.read_csv(PROC_VAL_CSV)
    valid_mask = val_df['image_path'].apply(lambda p: isinstance(p, str) and os.path.exists(p) and os.path.getsize(p) > 100)
    val_df = val_df[valid_mask].reset_index(drop=True).head(100)

    print(f"Evaluating on {len(val_df)} validation samples...")

    smooth = SmoothingFunction().method1
    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    b1_scores, b2_scores, b3_scores, b4_scores = [], [], [], []
    from tqdm import tqdm

    sample_predictions = []

    print("Generating predictions and computing BLEU/ROUGE metrics...", flush=True)
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating", unit="sample"):
        img_path = row["image_path"]
        ref_text = clean_report_text(row["report"])

        img = preprocess_xray(img_path)
        gen_text = generate_report(model, vocab, img)

        ref_tokens = ref_text.split()
        gen_tokens = gen_text.split()

        b1 = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        b2 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        b3 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
        b4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

        r_res = r_scorer.score(ref_text, gen_text if gen_text else "empty")

        b1_scores.append(b1)
        b2_scores.append(b2)
        b3_scores.append(b3)
        b4_scores.append(b4)

        r1_scores.append(r_res['rouge1'].fmeasure)
        r2_scores.append(r_res['rouge2'].fmeasure)
        rl_scores.append(r_res['rougeL'].fmeasure)

        if idx < 5:
            sample_predictions.append(f"Sample {idx+1}:\nReference : {ref_text[:120]}...\nGenerated : {gen_text}\n" + "-"*50)

    avg_b1 = np.mean(b1_scores)
    avg_b2 = np.mean(b2_scores)
    avg_b3 = np.mean(b3_scores)
    avg_b4 = np.mean(b4_scores)

    avg_r1 = np.mean(r1_scores)
    avg_r2 = np.mean(r2_scores)
    avg_rl = np.mean(rl_scores)

    metrics_str = f"""DenseNet121 + MIMIC-CXR (4000 Subset) Evaluation Results
=========================================================
Number of test samples: {len(val_df)}
BLEU-1 Score: {avg_b1:.4f}
BLEU-2 Score: {avg_b2:.4f}
BLEU-3 Score: {avg_b3:.4f}
BLEU-4 Score: {avg_b4:.4f}
ROUGE-1 F1  : {avg_r1:.4f}
ROUGE-2 F1  : {avg_r2:.4f}
ROUGE-L F1  : {avg_rl:.4f}

Sample Outputs:
""" + "\n".join(sample_predictions)

    print("\n" + metrics_str)

    eval_out_path = os.path.join(RESULTS_DIR, "evaluation_metrics.txt")
    with open(eval_out_path, "w", encoding="utf-8") as f:
        f.write(metrics_str)
    print(f"Saved evaluation metrics to: {eval_out_path}")

if __name__ == "__main__":
    main()
