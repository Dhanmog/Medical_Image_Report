import os
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

# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_CSV = "shenzhen_processed.csv"
OUTPUT_DIR = "results_shenzhen"
MODEL_PATH = os.path.join(OUTPUT_DIR, "shenzhen_model.keras")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "tokenizer_vocab.json")

RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
METRICS_TXT_PATH = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")

IMAGE_SIZE = (224, 224)
MAX_REPORT_LENGTH = 60

START_TOKEN = "<start>"
END_TOKEN = "<end>"

# ============================================================
# 2. IMAGE PREPROCESSING (EXACT MATCH TO TRAINING)
# ============================================================

def preprocess_xray(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        image = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    else:
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    image = image.astype(np.float32) / 255.0
    image = np.stack([image, image, image], axis=-1)
    return image


# ============================================================
# 3. CUSTOM MODEL LAYERS
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


def clean_raw_report(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9.,;:!?'\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("=" * 70)
    print("SHENZHEN CHEST X-RAY MODEL EVALUATION PIPELINE")
    print("=" * 70)

    # --- Step 1: Load Data & Reproduce Test Split ---
    print("\n[Step 1/6] Inspecting dataset and creating test set...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Processed CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    total_samples = len(df)
    
    # Reproduce training split (random_state=42, 85% train, 15% val/test)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(total_samples * 0.85)
    
    train_df = df_shuffled.iloc[:split_idx]
    test_df = df_shuffled.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Total Shenzhen samples : {total_samples}")
    print(f"Training samples       : {len(train_df)}")
    print(f"Validation samples     : 0 (Combined into held-out test split)")
    print(f"Test samples           : {len(test_df)}")

    # Verify image files and reference reports in test set
    valid_images = sum(os.path.exists(p) for p in test_df["image_path"])
    valid_reports = sum(bool(str(r).strip()) for r in test_df["report"])
    print(f"Test images with valid files: {valid_images} / {len(test_df)}")
    print(f"Test samples with valid reports: {valid_reports} / {len(test_df)}")

    # --- Step 2: Verify Data Leakage ---
    print("\n[Step 2/6] Verifying data leakage...")
    train_ids = set(train_df["study_id"])
    test_ids = set(test_df["study_id"])
    overlap = train_ids.intersection(test_ids)

    if len(overlap) == 0:
        print("Verification PASS: Test set contains no training images.")
    else:
        raise ValueError(f"Data leakage detected! {len(overlap)} samples overlap between train and test.")

    # --- Step 3: Load Model & Vocabulary ---
    print("\n[Step 3/6] Loading vocabulary and trained model...")
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocabulary not found: {VOCAB_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}

    start_id = token_to_id.get(START_TOKEN)
    end_id = token_to_id.get(END_TOKEN)

    custom_objects = {
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoderBlock": TransformerDecoderBlock
    }

    model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    print(f"Model and vocabulary (size={len(vocab)}) loaded successfully!")

    # --- Step 4: Generate Diagnostic Reports for Test Set ---
    print(f"\n[Step 4/6] Generating reports for {len(test_df)} test images...")
    
    evaluation_results = []
    
    for idx, row in test_df.iterrows():
        study_id = row["study_id"]
        image_path = row["image_path"]
        raw_ref_report = row["report"]
        ref_report = clean_raw_report(raw_ref_report)

        img = preprocess_xray(image_path)
        img_input = np.expand_dims(img, axis=0)

        # Greedy decoding
        seq = [start_id]
        for step in range(MAX_REPORT_LENGTH - 2):
            padded_seq = keras.preprocessing.sequence.pad_sequences(
                [seq], maxlen=MAX_REPORT_LENGTH - 1, padding="post"
            )
            preds = model.predict([img_input, padded_seq], verbose=0)
            next_id = int(np.argmax(preds[0, len(seq) - 1, :]))

            if next_id == end_id or next_id == 0:
                break
            seq.append(next_id)

        gen_words = [id_to_token.get(i, "") for i in seq if i not in (0, start_id, end_id)]
        gen_report = " ".join([w for w in gen_words if w])

        evaluation_results.append({
            "study_id": study_id,
            "image_path": image_path,
            "reference_report": ref_report,
            "generated_report": gen_report
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(test_df):
            print(f"  Processed {idx + 1}/{len(test_df)} test samples...")

    results_df = pd.DataFrame(evaluation_results)

    # --- Step 5: Compute Automatic Evaluation Metrics ---
    print("\n[Step 5/6] Computing evaluation metrics...")
    
    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    bleu_scores = []
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []

    rouge1_f1 = []
    rouge2_f1 = []
    rougeL_f1 = []

    exact_matches = 0
    gen_lengths = []
    ref_lengths = []

    for _, row in results_df.iterrows():
        ref = row["reference_report"]
        gen = row["generated_report"]

        ref_tokens = ref.split()
        gen_tokens = gen.split()

        ref_lengths.append(len(ref_tokens))
        gen_lengths.append(len(gen_tokens))

        if ref.strip() == gen.strip():
            exact_matches += 1

        # BLEU scores
        b1 = sentence_bleu([ref_tokens], gen_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
        b2 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        b3 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
        b4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

        bleu1_scores.append(b1)
        bleu2_scores.append(b2)
        bleu3_scores.append(b3)
        bleu4_scores.append(b4)
        bleu_scores.append(b4)  # Cumulative BLEU-4 standard

        # ROUGE scores
        r_scores = scorer.score(ref, gen)
        rouge1_f1.append(r_scores['rouge1'].fmeasure)
        rouge2_f1.append(r_scores['rouge2'].fmeasure)
        rougeL_f1.append(r_scores['rougeL'].fmeasure)

    avg_bleu = np.mean(bleu_scores)
    avg_bleu1 = np.mean(bleu1_scores)
    avg_bleu2 = np.mean(bleu2_scores)
    avg_bleu3 = np.mean(bleu3_scores)
    avg_bleu4 = np.mean(bleu4_scores)

    avg_rouge1 = np.mean(rouge1_f1)
    avg_rouge2 = np.mean(rouge2_f1)
    avg_rougeL = np.mean(rougeL_f1)

    exact_match_pct = (exact_matches / len(results_df)) * 100.0
    avg_gen_len = np.mean(gen_lengths)
    avg_ref_len = np.mean(ref_lengths)

    # --- Step 6: Save Results & Display Terminal Output ---
    print("\n[Step 6/6] Saving evaluation outputs & displaying samples...")
    
    # Save CSV
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"Saved test evaluation results to: {RESULTS_CSV_PATH}")

    # Prepare Metrics Summary TXT
    metrics_content = f"""Shenzhen Test Set Evaluation
============================

Number of test samples: {len(results_df)}
BLEU: {avg_bleu:.4f} (BLEU-1: {avg_bleu1:.4f}, BLEU-2: {avg_bleu2:.4f}, BLEU-3: {avg_bleu3:.4f}, BLEU-4: {avg_bleu4:.4f})
ROUGE-1: {avg_rouge1:.4f}
ROUGE-2: {avg_rouge2:.4f}
ROUGE-L: {avg_rougeL:.4f}
Exact Match: {exact_match_pct:.2f}%
Average Generated Report Length: {avg_gen_len:.2f} words
Average Reference Report Length: {avg_ref_len:.2f} words

Note: These automated n-gram / sequence overlap metrics measure textual similarity to ground-truth reports.
They do not directly quantify clinical diagnostic accuracy.
"""

    with open(METRICS_TXT_PATH, "w") as f:
        f.write(metrics_content)

    print(f"Saved evaluation metrics summary to: {METRICS_TXT_PATH}")

    # Display 10 Test Examples in Terminal
    print("\n" + "=" * 70)
    print("DISPLAYING 10 TEST SAMPLE GENERATIONS")
    print("=" * 70)

    for i in range(min(10, len(results_df))):
        sample = results_df.iloc[i]
        print(f"----------------------------------------")
        print(f"TEST SAMPLE {i + 1}")
        print(f"Image            : {sample['image_path']}")
        print(f"Study ID         : {sample['study_id']}")
        print(f"\nREFERENCE REPORT:\n{sample['reference_report']}")
        print(f"\nGENERATED REPORT :\n{sample['generated_report']}")
        print(f"----------------------------------------\n")

    print("=" * 70)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 70)
    print(metrics_content)
    print("=" * 70)
    print("\nEvaluation pipeline executed successfully!")

if __name__ == "__main__":
    main()
