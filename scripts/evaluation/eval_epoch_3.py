import os
import sys
import re
import json
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "trained_models", "mimic_cxr")

MODEL_PATH = os.path.join(OUTPUT_DIR, "mimic_cxr_model.keras")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "tokenizer_vocab.json")
VAL_CSV_PATH = os.path.join(PROJECT_DIR, "mimic_cxr_processed_val.csv")
EPOCH_3_TXT = os.path.join(OUTPUT_DIR, "epoch_3_test_results.txt")

IMAGE_SIZE = (224, 224)
MAX_REPORT_LENGTH = 100

START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def preprocess_single_xray(image_path):
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
    print("=== STARTING FAST EPOCH 3 EVALUATION ===", flush=True)

    val_df = pd.read_csv(VAL_CSV_PATH)
    print(f"• Loaded validation dataset: {len(val_df)} test samples", flush=True)

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
    print(f"• Loaded Epoch 3 model checkpoint: {MODEL_PATH}", flush=True)

    print("• Loading and pre-processing X-ray images with 16 threads...", flush=True)
    with ThreadPoolExecutor(max_workers=16) as executor:
        images_list = list(executor.map(preprocess_single_xray, val_df["image_path"].values))

    X_img = np.array(images_list, dtype=np.float32)

    tokenizer = layers.TextVectorization(
        max_tokens=len(vocab),
        output_mode="int",
        output_sequence_length=MAX_REPORT_LENGTH,
        vocabulary=vocab,
        standardize=None
    )

    tokens = tokenizer(val_df["report"].values).numpy()
    X_seq = tokens[:, :-1]
    Y_tgt = tokens[:, 1:]

    print("• Evaluating test loss & accuracy...", flush=True)
    eval_results = model.evaluate([X_img, X_seq], Y_tgt, batch_size=16, verbose=0)
    metrics_dict = dict(zip(model.metrics_names, eval_results))
    test_loss = metrics_dict.get("loss", eval_results[0])
    test_acc = metrics_dict.get("accuracy", eval_results[1])

    print(f"✓ Epoch 3 Test Loss     : {test_loss:.4f}", flush=True)
    print(f"✓ Epoch 3 Test Accuracy : {test_acc * 100:.2f}%", flush=True)

    print("• Generating report predictions for samples...", flush=True)
    sample_val = val_df.head(10)
    sample_reports = []

    for idx, row in sample_val.iterrows():
        img = X_img[idx:idx+1]
        seq = [start_id]
        for _ in range(MAX_REPORT_LENGTH - 2):
            padded_seq = keras.preprocessing.sequence.pad_sequences(
                [seq], maxlen=MAX_REPORT_LENGTH - 1, padding="post"
            )
            preds = model.predict([img, padded_seq], verbose=0)
            next_id = int(np.argmax(preds[0, len(seq) - 1, :]))
            if next_id == end_id or next_id == 0:
                break
            seq.append(next_id)

        gen_words = [id_to_token.get(i, "") for i in seq[1:] if i not in (0, start_id, end_id)]
        gen_report = " ".join([w for w in gen_words if w])
        sample_reports.append(gen_report)

    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
    rouge1_f1, rouge2_f1, rougeL_f1 = [], [], []

    for idx, row in sample_val.iterrows():
        ref = clean_raw_report(row["report"])
        gen = sample_reports[idx]
        ref_tokens = ref.split()
        gen_tokens = gen.split()

        bleu1_scores.append(sentence_bleu([ref_tokens], gen_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth))
        bleu2_scores.append(sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        bleu3_scores.append(sentence_bleu([ref_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
        bleu4_scores.append(sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))

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

    lines = [
        "MIMIC-CXR EPOCH 3 EVALUATION TEST RESULTS",
        "=" * 65,
        f"Checkpoint File Used   : {MODEL_PATH}",
        f"Test Dataset Used      : {VAL_CSV_PATH}",
        f"Number of Test Samples : {len(val_df)}",
        f"Number of Batches      : {int(np.ceil(len(val_df) / 16))}",
        "",
        "EVALUATION METRICS:",
        f"  - Test Loss          : {test_loss:.4f}",
        f"  - Test Accuracy      : {test_acc * 100:.2f}%",
        f"  - BLEU-1             : {avg_b1:.4f}",
        f"  - BLEU-2             : {avg_b2:.4f}",
        f"  - BLEU-3             : {avg_b3:.4f}",
        f"  - BLEU-4             : {avg_b4:.4f}",
        f"  - ROUGE-1            : {avg_r1:.4f}",
        f"  - ROUGE-2            : {avg_r2:.4f}",
        f"  - ROUGE-L            : {avg_rL:.4f}",
        "",
        "SAMPLE X-RAY REPORT PREDICTIONS:",
        "-" * 65
    ]

    for idx, row in sample_val.iterrows():
        lines.append(f"Subject ID : {row['subject_id']}")
        lines.append(f"Study ID   : {row['study_id']}")
        lines.append(f"View       : {row['view']}")
        lines.append(f"Actual     : {clean_raw_report(row['report'])}")
        lines.append(f"Generated  : {sample_reports[idx]}")
        lines.append("-" * 65)

    res_content = "\n".join(lines) + "\n"
    with open(EPOCH_3_TXT, "w", encoding="utf-8") as f:
        f.write(res_content)

    print(f"✓ Saved Epoch 3 evaluation results to: {EPOCH_3_TXT}", flush=True)
    print("=== EPOCH 3 EVALUATION COMPLETED SUCCESSFULLY ===", flush=True)

if __name__ == "__main__":
    main()
