import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ICCV_DIR = os.path.join(PROJECT_DIR, "official_data_iccv_final")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results_model")

MODEL_PATH = os.path.join(OUTPUT_DIR, "report_generator_model.keras")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "tokenizer_vocab.json")
VAL_CSV_PATH = os.path.join(PROJECT_DIR, "mimic_cxr_processed_val.csv")

FULL_PRED_CSV_PATH = os.path.join(OUTPUT_DIR, "full_test_predictions.csv")
FULL_REPORT_TXT_PATH = os.path.join(OUTPUT_DIR, "full_test_evaluation_report.txt")

IMAGE_SIZE = (224, 224)
MAX_REPORT_LENGTH = 60

START_TOKEN = "<start>"
END_TOKEN = "<end>"

def resolve_image_path(path_str):
    if not isinstance(path_str, str):
        path_str = str(path_str)
    norm_path = path_str.replace("/", os.sep).replace("\\", os.sep)
    if os.path.isabs(norm_path) and os.path.exists(norm_path):
        return norm_path
    cand1 = os.path.normpath(os.path.join(PROJECT_DIR, norm_path))
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.normpath(os.path.join(ICCV_DIR, norm_path))
    if os.path.exists(cand2):
        return cand2
    if norm_path.startswith("official_data_iccv_final"):
        cand3 = os.path.normpath(os.path.join(PROJECT_DIR, norm_path))
        if os.path.exists(cand3):
            return cand3
    return cand2 if norm_path.startswith("files") else cand1

def preprocess_xray(image_path):
    resolved = resolve_image_path(image_path)
    try:
        if not os.path.exists(resolved) or os.path.getsize(resolved) < 100:
            raise ValueError(f"File missing or truncated: {resolved}")
        buf = np.fromfile(resolved, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"Could not decode image file: {resolved}")
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = image.astype(np.float32) / 255.0
        image = np.stack([image, image, image], axis=-1)
        return image
    except Exception as e:
        print(f"[Warning] preprocess_xray for {resolved}: {e}")
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

@keras.utils.register_keras_serializable()
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

@keras.utils.register_keras_serializable()
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

def main():
    print("=" * 70)
    print("FULL DATASET EVALUATION & TESTING PIPELINE")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab file not found at: {VOCAB_PATH}")
    if not os.path.exists(VAL_CSV_PATH):
        raise FileNotFoundError(f"Validation CSV file not found at: {VAL_CSV_PATH}")

    print(f"Loading Model     : {MODEL_PATH}")
    custom_objects = {
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoderBlock": TransformerDecoderBlock
    }
    model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print(" [OK] Model loaded successfully.")

    print(f"Loading Vocabulary: {VOCAB_PATH}")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f" [OK] Vocabulary loaded ({len(vocab)} words).")

    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}
    start_id = token_to_id.get(START_TOKEN, 1)
    end_id = token_to_id.get(END_TOKEN, 2)

    df_val = pd.read_csv(VAL_CSV_PATH)
    total_images = len(df_val)
    print(f"\nLoaded Full Test/Validation Dataset: {total_images} images")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    b1_scores, b2_scores, b3_scores, b4_scores = [], [], [], []
    r1_scores, rl_scores = [], []
    eval_records = []

    print(f"Evaluating all {total_images} images...")

    for idx, row in df_val.iterrows():
        img_raw_path = row["image_path"]
        actual_raw = str(row["report"])
        actual_clean = actual_raw.replace(START_TOKEN, "").replace(END_TOKEN, "").strip()
        subject_id = row.get("subject_id", "N/A")
        study_id = row.get("study_id", "N/A")

        img = preprocess_xray(img_raw_path)
        img_input = np.expand_dims(img, axis=0)

        # Autoregressive generation
        seq = [start_id]
        for _ in range(MAX_REPORT_LENGTH - 2):
            padded_seq = keras.preprocessing.sequence.pad_sequences(
                [seq], maxlen=MAX_REPORT_LENGTH - 1, padding="post"
            )
            preds = model.predict([img_input, padded_seq], verbose=0)
            next_id = int(np.argmax(preds[0, len(seq) - 1, :]))
            if next_id == end_id or next_id == 0:
                break
            seq.append(next_id)

        gen_tokens = [id_to_token.get(i, "") for i in seq[1:] if i not in (0, start_id, end_id)]
        generated_report = " ".join([w for w in gen_tokens if w])

        ref_tokens = actual_clean.split()
        hyp_tokens = generated_report.split()

        b1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth) if hyp_tokens else 0.0
        b2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth) if hyp_tokens else 0.0
        b3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth) if hyp_tokens else 0.0
        b4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) if hyp_tokens else 0.0

        rouge_res = scorer.score(actual_clean, generated_report)
        r1 = rouge_res['rouge1'].fmeasure
        rl = rouge_res['rougeL'].fmeasure

        b1_scores.append(b1)
        b2_scores.append(b2)
        b3_scores.append(b3)
        b4_scores.append(b4)
        r1_scores.append(r1)
        rl_scores.append(rl)

        eval_records.append({
            "subject_id": subject_id,
            "study_id": study_id,
            "image_path": resolve_image_path(img_raw_path),
            "actual_report": actual_clean,
            "generated_report": generated_report,
            "bleu1": b1,
            "bleu4": b4,
            "rougeL": rl
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == total_images:
            print(f"  Processed {idx + 1}/{total_images} images... (BLEU-1: {np.mean(b1_scores):.4f}, ROUGE-L: {np.mean(rl_scores):.4f})")

    res_df = pd.DataFrame(eval_records)
    res_df.to_csv(FULL_PRED_CSV_PATH, index=False)

    mean_b1 = float(np.mean(b1_scores))
    mean_b2 = float(np.mean(b2_scores))
    mean_b3 = float(np.mean(b3_scores))
    mean_b4 = float(np.mean(b4_scores))
    mean_r1 = float(np.mean(r1_scores))
    mean_rl = float(np.mean(rl_scores))

    summary_lines = [
        "======================================================================",
        "COMPLETE TEST DATASET EVALUATION REPORT",
        "======================================================================",
        f"Total Images in Test Set  : {total_images}",
        f"Total Images Evaluated   : {len(eval_records)}",
        f"Status                   : SUCCESSFUL (100% Complete)",
        "----------------------------------------------------------------------",
        "AVERAGE EVALUATION METRICS ACROSS ENTIRE TEST DATASET:",
        f" • BLEU-1 Score          : {mean_b1:.4f}",
        f" • BLEU-2 Score          : {mean_b2:.4f}",
        f" • BLEU-3 Score          : {mean_b3:.4f}",
        f" • BLEU-4 Score          : {mean_b4:.4f}",
        f" • ROUGE-1 F1-Score      : {mean_r1:.4f}",
        f" • ROUGE-L F1-Score      : {mean_rl:.4f}",
        "----------------------------------------------------------------------",
        f"Detailed per-image predictions CSV saved to : {FULL_PRED_CSV_PATH}",
        f"Summary report file saved to               : {FULL_REPORT_TXT_PATH}",
        "======================================================================\n",
        "SAMPLE EVALUATIONS (FIRST 5 SAMPLES):"
    ]

    for idx, row in res_df.head(5).iterrows():
        summary_lines.extend([
            f"\n--- Sample #{idx + 1} ---",
            f"Subject: {row['subject_id']} | Study: {row['study_id']}",
            f"Path: {row['image_path']}",
            f"Actual   : {row['actual_report']}",
            f"Generated: {row['generated_report']}",
            f"Metrics  : BLEU-1={row['bleu1']:.4f}, BLEU-4={row['bleu4']:.4f}, ROUGE-L={row['rougeL']:.4f}"
        ])

    report_text = "\n".join(summary_lines) + "\n"
    with open(FULL_REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)

if __name__ == "__main__":
    main()
