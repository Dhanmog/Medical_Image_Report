"""
================================================================================
AUTOMATED MODEL ACCURACY TESTING & CLINICAL EVALUATION SUITE
================================================================================
This script evaluates the Chest X-Ray Diagnostic Report Generation model on the
validation dataset and calculates:
  1. Token-Level Prediction Accuracy (Teacher-Forcing)
  2. Natural Language Generation Accuracy: BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-L
  3. Clinical Pathology Accuracy & F1-Scores across 5 major chest pathologies:
     - Cardiomegaly, Pleural Effusion, Pneumonia, Pneumothorax, Atelectasis
  4. Side-by-Side Sample Diagnostic Report Comparisons

Usage:
  python test_accuracy.py
  python test_accuracy.py --samples 20 --beam-width 3
  python test_accuracy.py --model models/02_densenet121_mimic_v3
================================================================================
"""

import os
import sys
import json
import re
import math
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import custom registered layers from app.py
from app import (
    PositionalEmbedding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
    SpatialTransformerEncoder,
    WarmupLRSchedule,
    preprocess_image_bytes,
    generate_report,
    generate_report_beam
)

# ============================================================
# 1. CLINICAL PATHOLOGY CLASSIFICATION DICTIONARY
# ============================================================

PATHOLOGIES = {
    "cardiomegaly": {
        "positive": [r"cardiomegaly", r"enlarged heart", r"heart is enlarged", r"prominent heart size", r"cardiomegaly is present"],
        "negative": [r"no cardiomegaly", r"heart size is normal", r"normal heart size", r"heart size normal", r"without cardiomegaly", r"cardiomegaly is absent"]
    },
    "pleural effusion": {
        "positive": [r"pleural effusion", r"effusion", r"fluid in the pleural space"],
        "negative": [r"no pleural effusion", r"no effusion", r"without pleural effusion", r"effusion is absent", r"no effusions", r"clear pleural spaces"]
    },
    "pneumonia": {
        "positive": [r"pneumonia", r"consolidation", r"infiltrate", r"infectious process"],
        "negative": [r"no pneumonia", r"without pneumonia", r"no consolidation", r"no infiltrates", r"no evidence of pneumonia"]
    },
    "pneumothorax": {
        "positive": [r"pneumothorax", r"air in the pleural cavity", r"collapsed lung"],
        "negative": [r"no pneumothorax", r"no evidence of pneumothorax", r"without pneumothorax", r"pneumothorax is absent", r"no air in pleural cavity"]
    },
    "atelectasis": {
        "positive": [r"atelectasis", r"lobar collapse", r"subsegmental atelectasis", r"volume loss"],
        "negative": [r"no atelectasis", r"no evidence of atelectasis", r"without atelectasis", r"atelectasis is absent"]
    }
}

def classify_report_pathologies(report_text):
    report_clean = report_text.lower()
    classifications = {}
    for pathology, keywords in PATHOLOGIES.items():
        negated = any(re.search(pat, report_clean) for pat in keywords["negative"])
        if negated:
            classifications[pathology] = -1
            continue
        affirmed = any(re.search(pat, report_clean) for pat in keywords["positive"])
        classifications[pathology] = 1 if affirmed else 0
    return classifications

# ============================================================
# 2. NLP ACCURACY FUNCTIONS (BLEU & ROUGE-L)
# ============================================================

def _tokenize(text: str):
    text = re.sub(r'<start>|<end>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower().split()

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> dict:
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens or not ref_tokens:
        return {f'bleu{n}': 0.0 for n in range(1, max_n+1)}

    scores = {}
    for n in range(1, max_n+1):
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        hyp_ngrams = _ngrams(hyp_tokens, n)
        if not hyp_ngrams:
            scores[f'bleu{n}'] = 0.0
            continue
        clip_count = sum(min(count, ref_ngrams[gram]) for gram, count in Counter(hyp_ngrams).items())
        precision = clip_count / len(hyp_ngrams)
        bp = math.exp(1 - len(ref_tokens)/len(hyp_tokens)) if len(hyp_tokens) < len(ref_tokens) else 1.0
        scores[f'bleu{n}'] = bp * precision
    return scores

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if ref[i-1]==hyp[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0
    recall = lcs / m if m else 0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# ============================================================
# 3. PATH RESOLUTION HELPER
# ============================================================

def resolve_image_path(image_path):
    if not isinstance(image_path, str):
        image_path = str(image_path)
    image_path = image_path.replace('\\', '/')
    match = re.search(r'medical_image_report/(.*)', image_path)
    if match:
        rel_path = os.path.join(PROJECT_DIR, match.group(1).replace('/', os.sep))
        if os.path.exists(rel_path):
            return os.path.abspath(rel_path)
    if not os.path.exists(image_path):
        cleaned_path = re.sub(r'^[a-zA-Z]:', '', image_path).lstrip('/')
        path_alt = os.path.join(PROJECT_DIR, cleaned_path.replace('/', os.sep))
        if os.path.exists(path_alt):
            return os.path.abspath(path_alt)
    return os.path.abspath(image_path)

# ============================================================
# 4. MAIN TESTING PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Automated Accuracy Testing for Chest X-Ray Diagnostic Report Generator")
    parser.add_argument("--model", type=str, default=None, help="Directory or path of model to evaluate")
    parser.add_argument("--samples", type=int, default=10, help="Number of validation images to test (default: 10)")
    parser.add_argument("--beam-width", type=int, default=3, help="Beam width for beam search (default: 3)")
    args = parser.parse_args()

    # Find model directory
    candidate_dirs = [
        args.model if args.model else "",
        os.path.join(PROJECT_DIR, "models", "02_densenet121_mimic_v3"),
        os.path.join(PROJECT_DIR, "models", "02_densenet121_mimic_v2"),
        os.path.join(PROJECT_DIR, "models", "01_efficientnetb0_mimic_v2"),
        os.path.join(PROJECT_DIR, "models", "02_densenet121_mimic"),
    ]

    model_dir = None
    model_path = None
    vocab_path = None

    for cdir in candidate_dirs:
        if not cdir:
            continue
        mpath = os.path.join(cdir, "model.keras") if os.path.isdir(cdir) else cdir
        vpath = os.path.join(os.path.dirname(mpath), "tokenizer_vocab.json")
        if os.path.exists(mpath) and os.path.exists(vpath):
            model_dir = os.path.dirname(mpath)
            model_path = mpath
            vocab_path = vpath
            break

    if not model_path:
        print("[ERROR] No trained model.keras found in candidate directories.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("      CHEST X-RAY DIAGNOSTIC REPORT GENERATOR — ACCURACY TEST")
    print("=" * 70)
    print(f"  • Active Model : {os.path.basename(model_dir)} ({model_path})")
    print(f"  • Vocabulary   : {vocab_path}")
    print(f"  • Test Samples : {args.samples}")
    print(f"  • Beam Width   : {args.beam_width}")
    print("=" * 70)

    # 1. Load Vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"✓ Loaded vocabulary containing {len(vocab):,} tokens.")

    # 2. Load Model
    custom_objects = {
        "PositionalEmbedding":              PositionalEmbedding,
        "TransformerDecoderBlock":          TransformerDecoderBlock,
        "TransformerEncoderBlock":          TransformerEncoderBlock,
        "SpatialTransformerEncoder":        SpatialTransformerEncoder,
        "Custom>PositionalEmbedding":       PositionalEmbedding,
        "Custom>TransformerDecoderBlock":   TransformerDecoderBlock,
        "Custom>TransformerEncoderBlock":   TransformerEncoderBlock,
        "Custom>SpatialTransformerEncoder": SpatialTransformerEncoder,
        "WarmupLRSchedule":                 WarmupLRSchedule,
    }
    print("✓ Loading Keras model weights...")
    model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    print("✓ Model successfully loaded!")

    # 3. Load Validation Dataset
    val_csv_candidates = [
        os.path.join(PROJECT_DIR, "datasets", "mimic_cxr", "processed", "mimic_cxr_processed_val.csv"),
        os.path.join(PROJECT_DIR, "datasets", "mimic_cxr", "processed", "processed_dataset_v1", "validation.csv"),
    ]
    val_csv_path = next((p for p in val_csv_candidates if os.path.exists(p)), None)
    if not val_csv_path:
        print("[ERROR] No validation dataset CSV found.")
        sys.exit(1)

    df_val = pd.read_csv(val_csv_path).dropna(subset=["report", "image_path"]).reset_index(drop=True)
    df_val = df_val.iloc[:args.samples]
    print(f"✓ Loaded {len(df_val)} records from validation set ({os.path.basename(val_csv_path)}).\n")

    # 4. Evaluation Loop
    nlp_results = {
        "greedy": {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "rouge_l": []},
        "beam":   {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "rouge_l": []}
    }

    clinical_matrix = {
        pathology: {
            "greedy": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
            "beam":   {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        } for pathology in PATHOLOGIES.keys()
    }

    sample_comparisons = []

    print("Running automated report generation & evaluation...")
    for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Evaluating Images"):
        img_path = resolve_image_path(row["image_path"])
        ground_truth = str(row["report"]).strip()

        if not os.path.exists(img_path) or os.path.getsize(img_path) < 100:
            continue

        # Read and preprocess
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        preprocessed_img = preprocess_image_bytes(image_bytes)

        # Generate Reports
        pred_greedy = generate_report(model, vocab, preprocessed_img, temperature=0.0)
        pred_beam   = generate_report_beam(model, vocab, preprocessed_img, beam_width=args.beam_width, min_len=10)

        # NLP metrics
        scores_g = compute_bleu(ground_truth, pred_greedy)
        rouge_g = compute_rouge_l(ground_truth, pred_greedy)
        scores_b = compute_bleu(ground_truth, pred_beam)
        rouge_b = compute_rouge_l(ground_truth, pred_beam)

        for k in ["bleu1", "bleu2", "bleu3", "bleu4"]:
            nlp_results["greedy"][k].append(scores_g[k])
            nlp_results["beam"][k].append(scores_b[k])
        nlp_results["greedy"]["rouge_l"].append(rouge_g)
        nlp_results["beam"]["rouge_l"].append(rouge_b)

        # Clinical metrics
        class_gt = classify_report_pathologies(ground_truth)
        class_g  = classify_report_pathologies(pred_greedy)
        class_b  = classify_report_pathologies(pred_beam)

        for path in PATHOLOGIES.keys():
            gt_val = class_gt[path]
            for strat, strat_class in [("greedy", class_g), ("beam", class_b)]:
                pred_val = strat_class[path]
                if gt_val == 1:
                    if pred_val == 1:
                        clinical_matrix[path][strat]["tp"] += 1
                    else:
                        clinical_matrix[path][strat]["fn"] += 1
                else:
                    if pred_val == 1:
                        clinical_matrix[path][strat]["fp"] += 1
                    else:
                        clinical_matrix[path][strat]["tn"] += 1

        if len(sample_comparisons) < 3:
            sample_comparisons.append({
                "image": os.path.basename(img_path),
                "ground_truth": ground_truth,
                "pred_greedy": pred_greedy,
                "pred_beam": pred_beam
            })

    # ============================================================
    # 5. DISPLAY RESULTS DASHBOARD
    # ============================================================
    g_avg = {k: np.mean(v) * 100 for k, v in nlp_results["greedy"].items()}
    b_avg = {k: np.mean(v) * 100 for k, v in nlp_results["beam"].items()}

    print("\n" + "=" * 70)
    print("                     📊 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print("\n1️⃣  NATURAL LANGUAGE GENERATION (NLG) ACCURACY:")
    print("-" * 70)
    print(f"{'Metric':<12} | {'Greedy Search':<18} | {'Beam Search (Width ' + str(args.beam_width) + ')':<20}")
    print("-" * 70)
    print(f"{'BLEU-1':<12} | {g_avg['bleu1']:>15.2f}%  | {b_avg['bleu1']:>18.2f}%")
    print(f"{'BLEU-2':<12} | {g_avg['bleu2']:>15.2f}%  | {b_avg['bleu2']:>18.2f}%")
    print(f"{'BLEU-3':<12} | {g_avg['bleu3']:>15.2f}%  | {b_avg['bleu3']:>18.2f}%")
    print(f"{'BLEU-4':<12} | {g_avg['bleu4']:>15.2f}%  | {b_avg['bleu4']:>18.2f}%")
    print(f"{'ROUGE-L':<12} | {g_avg['rouge_l']:>15.2f}%  | {b_avg['rouge_l']:>18.2f}%")
    print("-" * 70)

    print("\n2️⃣  CLINICAL PATHOLOGY ACCURACY & F1-SCORES (Beam Search):")
    print("-" * 70)
    print(f"{'Pathology':<20} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
    print("-" * 70)

    all_f1 = []
    for path in PATHOLOGIES.keys():
        stats = clinical_matrix[path]["beam"]
        tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        all_f1.append(f1)
        print(f"{path.capitalize():<20} | {prec*100:>9.1f}%   | {rec*100:>9.1f}%   | {f1*100:>9.1f}%")

    macro_f1 = np.mean(all_f1) * 100
    print("-" * 70)
    print(f"{'Macro-Average F1':<20} | {'-':<12} | {'-':<12} | {macro_f1:>9.1f}%")
    print("-" * 70)

    print("\n3️⃣  QUALITATIVE SAMPLE PREVIEWS:")
    print("-" * 70)
    for i, s in enumerate(sample_comparisons, 1):
        print(f"Sample #{i} [{s['image']}]:")
        print(f"  • Ground Truth : {s['ground_truth'][:120]}...")
        print(f"  • Generated    : {s['pred_beam'][:120]}...")
        print()
    print("=" * 70)
    print("✅ Testing completed successfully!\n")

if __name__ == "__main__":
    main()
