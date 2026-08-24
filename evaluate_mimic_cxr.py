import os
import sys
import re
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

# ============================================================
# 1. CONFIGURATION & PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "trained_models", "mimic_cxr")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "mimic_cxr_checkpoint.keras")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "tokenizer_vocab.json")
VAL_CSV_PATH = os.path.join(PROJECT_DIR, "mimic_cxr_processed_val.csv")

RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
METRICS_TXT_PATH = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
SAMPLE_PRED_PATH = os.path.join(OUTPUT_DIR, "sample_predictions.txt")

IMAGE_SIZE = (224, 224)
MAX_REPORT_LENGTH = 100

START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# ============================================================
# 2. IMAGE PREPROCESSING (EXACT MATCH TO TRAINING)
# ============================================================

def preprocess_xray(image_path):
    if not isinstance(image_path, str):
        image_path = str(image_path)
    try:
        if not os.path.exists(image_path) or os.path.getsize(image_path) < 100:
            raise ValueError(f"File missing or truncated: {image_path}")
        img_np = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image file: {image_path}")

        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = image.astype(np.float32) / 255.0
        image = np.stack([image, image, image], axis=-1)
        return image
    except Exception as e:
        print(f"  [Warning] preprocess_xray for {image_path}: {e}")
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

# ============================================================
# 3. CUSTOM MODEL LAYERS
# ============================================================

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

def clean_raw_report(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9.,;:!?'\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    print("=" * 70)
    print("MIMIC-CXR MODEL EVALUATION & TESTING PIPELINE")
    print("=" * 70)

    print("\n[Step 1/5] Loading validation dataset...")
    if not os.path.exists(VAL_CSV_PATH):
        raise FileNotFoundError(f"Validation dataset not found: {VAL_CSV_PATH}")

    val_df = pd.read_csv(VAL_CSV_PATH)
    print(f"  • Total Validation Samples : {len(val_df)}")
    print(f"  • Unique Patients          : {val_df['subject_id'].nunique()}")
    print(f"  • Unique Studies           : {val_df['study_id'].nunique()}")

    print("\n[Step 2/5] Loading Epoch 1 checkpoint model & vocabulary...")
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint model not found: {MODEL_PATH}")

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}
    start_id = token_to_id.get(START_TOKEN, 1)
    end_id = token_to_id.get(END_TOKEN, 2)

    custom_objects = {
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoderBlock": TransformerDecoderBlock
    }

    model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print(f"  • Epoch 1 Checkpoint loaded successfully from: {MODEL_PATH}")
    print(f"  • Vocabulary Size: {len(vocab)} tokens")

    # Evaluate batch loss & accuracy using TextVectorization tokenizer
    tokenizer = layers.TextVectorization(
        max_tokens=len(vocab),
        output_mode="int",
        output_sequence_length=MAX_REPORT_LENGTH,
        vocabulary=vocab,
        standardize=None
    )

    print("\n[Step 3/5] Evaluating test loss & accuracy on validation dataset...")
    images_list = []
    decoder_inputs = []
    decoder_targets = []

    for idx, row in val_df.iterrows():
        img = preprocess_xray(row["image_path"])
        tokens = tokenizer(row["report"]).numpy()
        dec_in = tokens[:-1]
        dec_tgt = tokens[1:]

        images_list.append(img)
        decoder_inputs.append(dec_in)
        decoder_targets.append(dec_tgt)

    X_img = np.array(images_list, dtype=np.float32)
    X_seq = np.array(decoder_inputs, dtype=np.int32)
    Y_tgt = np.array(decoder_targets, dtype=np.int32)

    eval_results = model.evaluate([X_img, X_seq], Y_tgt, batch_size=16, verbose=1)
    
    metrics_dict = dict(zip(model.metrics_names, eval_results))
    test_loss = metrics_dict.get("loss", eval_results[0])
    test_acc = metrics_dict.get("accuracy", eval_results[1])

    print(f"\n  • Test Evaluation Loss     : {test_loss:.4f}")
    print(f"  • Test Evaluation Accuracy : {test_acc * 100:.2f}%")

    print("\n[Step 4/5] Generating reports & computing BLEU / ROUGE evaluation metrics...")
    N = len(val_df)
    batch_size = 32
    generated_reports = []

    for b_start in range(0, N, batch_size):
        b_end = min(b_start + batch_size, N)
        print(f"  • Generating reports batch {b_start // batch_size + 1} / {(N + batch_size - 1) // batch_size} (samples {b_start} to {b_end})...", flush=True)
        batch_imgs = X_img[b_start:b_end]
        B = len(batch_imgs)

        seqs = np.zeros((B, MAX_REPORT_LENGTH - 1), dtype=np.int32)
        seqs[:, 0] = start_id
        finished = np.zeros(B, dtype=bool)

        for step in range(MAX_REPORT_LENGTH - 2):
            preds = model([batch_imgs, seqs], training=False).numpy()
            next_ids = np.argmax(preds[:, step, :], axis=-1)
            
            for i in range(B):
                if not finished[i]:
                    nid = int(next_ids[i])
                    if nid == end_id or nid == 0:
                        finished[i] = True
                    else:
                        seqs[i, step + 1] = nid

            if np.all(finished):
                break

        for i in range(B):
            row_ids = seqs[i]
            words = [id_to_token.get(t, "") for t in row_ids if t not in (0, start_id, end_id)]
            generated_reports.append(" ".join([w for w in words if w]))

    evaluation_records = []
    sample_lines = ["MIMIC-CXR EVALUATION SAMPLE REPORTS", "=" * 65]

    for idx, row in val_df.iterrows():
        study_id = row["study_id"]
        sub_id = row["subject_id"]
        image_path = row["image_path"]
        raw_ref_report = row["report"]
        ref_report = clean_raw_report(raw_ref_report)
        gen_report = generated_reports[idx]

        evaluation_records.append({
            "subject_id": sub_id,
            "study_id": study_id,
            "image_path": image_path,
            "reference_report": ref_report,
            "generated_report": gen_report
        })

        if idx < 10:
            sample_lines.append(f"Subject ID : {sub_id}")
            sample_lines.append(f"Study ID   : {study_id}")
            sample_lines.append(f"Actual     : {ref_report}")
            sample_lines.append(f"Generated  : {gen_report}")
            sample_lines.append("-" * 65)

    results_df = pd.DataFrame(evaluation_records)

    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
    rouge1_f1, rouge2_f1, rougeL_f1 = [], [], []
    exact_matches = 0

    for _, r in results_df.iterrows():
        ref = r["reference_report"]
        gen = r["generated_report"]

        ref_tokens = ref.split()
        gen_tokens = gen.split()

        if ref.strip() == gen.strip():
            exact_matches += 1

        b1 = sentence_bleu([ref_tokens], gen_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
        b2 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        b3 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
        b4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

        bleu1_scores.append(b1)
        bleu2_scores.append(b2)
        bleu3_scores.append(b3)
        bleu4_scores.append(b4)

        r_scores = scorer.score(ref, gen)
        rouge1_f1.append(r_scores['rouge1'].fmeasure)
        rouge2_f1.append(r_scores['rouge2'].fmeasure)
        rougeL_f1.append(r_scores['rougeL'].fmeasure)

    avg_b1 = np.mean(bleu1_scores)
    avg_b2 = np.mean(bleu2_scores)
    avg_b3 = np.mean(bleu3_scores)
    avg_b4 = np.mean(bleu4_scores)

    avg_r1 = np.mean(rouge1_f1)
    avg_r2 = np.mean(rouge2_f1)
    avg_rL = np.mean(rougeL_f1)

    exact_pct = (exact_matches / len(results_df)) * 100.0

    print(f"  • BLEU-1 : {avg_b1:.4f}")
    print(f"  • BLEU-2 : {avg_b2:.4f}")
    print(f"  • BLEU-3 : {avg_b3:.4f}")
    print(f"  • BLEU-4 : {avg_b4:.4f}")
    print(f"  • ROUGE-1: {avg_r1:.4f}")
    print(f"  • ROUGE-2: {avg_r2:.4f}")
    print(f"  • ROUGE-L: {avg_rL:.4f}")

    print("\n[Step 5/5] Saving evaluation outputs & predictions...")
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"  • Saved test evaluation table to: {RESULTS_CSV_PATH}")

    metrics_text = f"""MIMIC-CXR Epoch 1 Evaluation Summary
====================================
Model Checkpoint  : {MODEL_PATH}
Evaluation Data   : {VAL_CSV_PATH} ({len(val_df)} test samples)

Accuracy & Loss:
  - Validation Loss     : {test_loss:.4f}
  - Validation Accuracy : {test_acc * 100:.2f}%

NLP Sequence Overlap Metrics:
  - BLEU-1  : {avg_b1:.4f}
  - BLEU-2  : {avg_b2:.4f}
  - BLEU-3  : {avg_b3:.4f}
  - BLEU-4  : {avg_b4:.4f}
  - ROUGE-1 : {avg_r1:.4f}
  - ROUGE-2 : {avg_r2:.4f}
  - ROUGE-L : {avg_rL:.4f}
  - Exact Match Rate : {exact_pct:.2f}%
"""

    with open(METRICS_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"  • Saved metrics summary to: {METRICS_TXT_PATH}")

    with open(SAMPLE_PRED_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_lines) + "\n")
    print(f"  • Saved sample report predictions to: {SAMPLE_PRED_PATH}")

    print("\n" + "=" * 70)
    print("MIMIC-CXR EPOCH 1 EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(metrics_text)

if __name__ == "__main__":
    main()
