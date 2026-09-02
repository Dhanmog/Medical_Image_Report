import os
import sys
import json
import re
import math
import argparse
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
from tqdm import tqdm

# Calculate PROJECT_DIR and append it to sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import registered custom layers from app
from app import PositionalEmbedding, TransformerDecoderBlock, TransformerEncoderBlock, SpatialTransformerEncoder, WarmupLRSchedule

def normalize_and_resolve_path(image_path):
    if not isinstance(image_path, str):
        image_path = str(image_path)
    # Replace backslashes with forward slashes for uniform processing
    image_path = image_path.replace('\\', '/')
    
    # If running on posix (WSL/Linux) and path starts with a Windows drive letter
    if os.name == 'posix' and re.match(r'^[a-zA-Z]:', image_path):
        drive = image_path[0].lower()
        image_path = f"/mnt/{drive}{image_path[2:]}"
        
    # Extract portion after "medical_image_report" if present, to make it relative to PROJECT_DIR
    match = re.search(r'medical_image_report/(.*)', image_path)
    if match:
        rel_path = os.path.join(PROJECT_DIR, match.group(1).replace('/', os.sep))
        if os.path.exists(rel_path):
            image_path = rel_path
        else:
            path_alt = os.path.join(PROJECT_DIR, match.group(1))
            if os.path.exists(path_alt):
                image_path = path_alt
    else:
        # If "medical_image_report" is not in path, check if it's already relative or can be found in PROJECT_DIR
        if not os.path.exists(image_path):
            cleaned_path = re.sub(r'^[a-zA-Z]:', '', image_path).lstrip('/')
            path_alt = os.path.join(PROJECT_DIR, cleaned_path.replace('/', os.sep))
            if os.path.exists(path_alt):
                image_path = path_alt
            else:
                path_alt2 = os.path.join(PROJECT_DIR, image_path.replace('/', os.sep))
                if os.path.exists(path_alt2):
                    image_path = path_alt2

    return os.path.abspath(image_path)

# ============================================================
# 1. PATHOLOGY CLASSIFIERS (Clinical Correctness)
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
    """
    Classifies a report for 5 main pathologies as:
      1 (Affirmed/Positive)
     -1 (Negated/Negative)
      0 (Unmentioned)
    """
    report_clean = report_text.lower()
    classifications = {}
    
    for pathology, keywords in PATHOLOGIES.items():
        # First check negation
        negated = False
        for pattern in keywords["negative"]:
            if re.search(pattern, report_clean):
                negated = True
                break
        
        if negated:
            classifications[pathology] = -1
            continue
            
        # Then check affirmation
        affirmed = False
        for pattern in keywords["positive"]:
            if re.search(pattern, report_clean):
                affirmed = True
                break
                
        if affirmed:
            classifications[pathology] = 1
        else:
            classifications[pathology] = 0
            
    return classifications

# ============================================================
# 2. NLP METRICS FUNCTIONS
# ============================================================

def _tokenize(text: str):
    # Remove special chars and tokenize
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
    return (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0

# ============================================================
# 3. AUTOREGRESSIVE GENERATORS
# ============================================================

def generate_greedy(model, vocab, preprocessed_image):
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}
    start_id = token_to_id.get("<start>", 1)
    end_id   = token_to_id.get("<end>", 2)

    image_input = np.expand_dims(preprocessed_image, axis=0)
    input_shape = model.inputs[1].shape
    max_seq_len = input_shape[1] if len(input_shape) > 1 and input_shape[1] is not None else 59

    seq = [start_id]
    for _ in range(max_seq_len):
        padded_seq = keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=max_seq_len, padding="post"
        )
        preds_raw = model.predict([image_input, padded_seq], verbose=0)
        preds = preds_raw[0] if isinstance(preds_raw, (list, tuple)) else preds_raw
        probs = preds[0, len(seq) - 1, :]
        next_id = int(np.argmax(probs))

        if next_id == end_id or next_id == 0:
            break
        seq.append(next_id)

    words = [id_to_token.get(i, "") for i in seq if i not in (0, start_id, end_id)]
    return " ".join([w for w in words if w])

def generate_beam(model, vocab, preprocessed_image, beam_width=3, length_penalty=0.6, min_len=10, no_repeat_ngram_size=4):
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}
    start_id = token_to_id.get("<start>", 1)
    end_id   = token_to_id.get("<end>", 2)

    image_input = np.expand_dims(preprocessed_image, axis=0)
    input_shape = model.inputs[1].shape
    max_seq_len = input_shape[1] if len(input_shape) > 1 and input_shape[1] is not None else 59

    beams = [(0.0, [start_id], False)]
    completed = []

    for step in range(max_seq_len):
        candidates = []
        all_finished = True
        for score, seq, finished in beams:
            if not finished:
                all_finished = False
                break
        if all_finished:
            break

        for score, seq, finished in beams:
            if finished:
                candidates.append((score, seq, True))
                continue

            padded_seq = keras.preprocessing.sequence.pad_sequences(
                [seq], maxlen=max_seq_len, padding="post"
            )
            preds_raw = model.predict([image_input, padded_seq], verbose=0)
            preds = preds_raw[0] if isinstance(preds_raw, (list, tuple)) else preds_raw
            probs = preds[0, len(seq) - 1, :]
            probs = np.array(probs, dtype=np.float32)

            # Apply min length constraint: penalize END and pad tokens until min_len reached
            if len(seq) < min_len:
                probs[end_id] = 0.0
                probs[0] = 0.0

            # Apply no-repeat n-gram blocking
            if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
                ngram_size = no_repeat_ngram_size
                suffix = seq[-(ngram_size - 1):]
                for i in range(len(seq) - ngram_size + 1):
                    ngram = seq[i : i + ngram_size]
                    if ngram[:-1] == suffix:
                        forbidden_token = ngram[-1]
                        probs[forbidden_token] = 0.0

            # Normalize probabilities
            sum_probs = np.sum(probs)
            if sum_probs > 0:
                probs = probs / sum_probs
            else:
                probs = np.ones_like(probs) / len(probs)

            probs = np.maximum(probs, 1e-10)
            top_ids = np.argsort(probs)[::-1][:beam_width]
            
            for next_id in top_ids:
                rep_penalty = 0.0
                if len(seq) >= 3 and next_id == seq[-1] == seq[-2] == seq[-3]:
                    rep_penalty = -10.0
                if len(seq) >= 4 and [seq[-2], seq[-1]] == [seq[-4], seq[-3]]:
                    rep_penalty = -5.0
                
                next_score = score + np.log(probs[next_id]) + rep_penalty
                next_seq = seq + [int(next_id)]
                
                is_end = (next_id == end_id) or (next_id == 0)
                candidates.append((next_score, next_seq, is_end))

        def get_norm_score(item):
            score, seq, finished = item
            seq_len = len(seq) - 1
            if seq_len <= 0:
                return score
            lp_denom = ((5.0 + seq_len) / 6.0) ** length_penalty
            return score / lp_denom

        candidates = sorted(candidates, key=get_norm_score, reverse=True)
        beams = candidates[:beam_width]
        
        new_beams = []
        for score, seq, finished in beams:
            if finished:
                completed.append((score, seq, True))
            else:
                new_beams.append((score, seq, False))
        beams = new_beams
        
        if len(completed) >= beam_width:
            break

    all_final = completed + beams
    all_final = sorted(all_final, key=get_norm_score, reverse=True)
    best_seq = all_final[0][1]

    words = [id_to_token.get(i, "") for i in best_seq if i not in (0, start_id, end_id)]
    return " ".join([w for w in words if w])

# ============================================================
# 4. MAIN EVALUATION LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Chest X-Ray Report Generator (NLP + Clinical path metrics)")
    parser.add_argument("--model-dir", type=str, default="models/02_densenet121_mimic_v3", help="Directory where model.keras resides")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of validation samples to evaluate on")
    parser.add_argument("--beam-width", type=int, default=3, help="Beam width for decoding")
    args = parser.parse_args()

    model_dir_path = os.path.join(PROJECT_DIR, args.model_dir)
    model_keras_path = os.path.join(model_dir_path, "model.keras")
    vocab_path = os.path.join(model_dir_path, "tokenizer_vocab.json")

    # If v3 dir is empty, fall back to baseline model
    if not os.path.exists(model_keras_path):
        print(f"[WARNING] Model not found at {model_keras_path}. Falling back to baseline model...")
        model_dir_path = os.path.join(PROJECT_DIR, "models", "02_densenet121_mimic")
        model_keras_path = os.path.join(model_dir_path, "model.keras")
        vocab_path = os.path.join(model_dir_path, "tokenizer_vocab.json")

    print(f"\nEvaluating Model:")
    print(f"  • Model Path: {model_keras_path}")
    print(f"  • Vocab Path: {vocab_path}")
    print(f"  • Evaluation Samples: {args.num_samples}")

    if not os.path.exists(model_keras_path) or not os.path.exists(vocab_path):
        print("[ERROR] Model files or vocab not found!")
        sys.exit(1)

    # 1. Load Vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Loaded vocab containing {len(vocab)} tokens.")

    # 2. Load Model
    custom_objects = {
        "PositionalEmbedding":            PositionalEmbedding,
        "TransformerDecoderBlock":        TransformerDecoderBlock,
        "TransformerEncoderBlock":        TransformerEncoderBlock,
        "SpatialTransformerEncoder":      SpatialTransformerEncoder,
        "Custom>PositionalEmbedding":     PositionalEmbedding,
        "Custom>TransformerDecoderBlock": TransformerDecoderBlock,
        "Custom>TransformerEncoderBlock": TransformerEncoderBlock,
        "Custom>SpatialTransformerEncoder": SpatialTransformerEncoder,
        "WarmupLRSchedule":               WarmupLRSchedule,
    }
    print("Loading Keras model...")
    model = keras.models.load_model(model_keras_path, custom_objects=custom_objects, compile=False)
    print("Model loaded successfully!")

    # 3. Load Validation Data
    val_csv_path = os.path.join(PROJECT_DIR, "datasets", "mimic_cxr", "processed", "mimic_cxr_processed_val.csv")
    if not os.path.exists(val_csv_path):
        print(f"[ERROR] Validation CSV not found: {val_csv_path}")
        sys.exit(1)

    df_val = pd.read_csv(val_csv_path)
    df_val = df_val.dropna(subset=["report", "image_path"]).reset_index(drop=True)
    df_val = df_val.iloc[:args.num_samples]
    print(f"Loaded {len(df_val)} records from validation set.")

    # 4. Metrics accumulators
    nlp_results = {
        "greedy": {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "rouge_l": []},
        "beam":   {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "rouge_l": []}
    }

    # Clinical correctness pathology matrix
    # Format: {pathology: {"greedy": {"tp":0, "fp":0, "fn":0, "tn":0}, "beam": {...}}}
    clinical_matrix = {
        pathology: {
            "greedy": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
            "beam":   {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        } for pathology in PATHOLOGIES.keys()
    }

    # Loop through validation samples
    print("\nRunning inference & metric calculations...", flush=True)
    for idx, row in tqdm(df_val.iterrows(), total=len(df_val)):
        img_path = normalize_and_resolve_path(row["image_path"])
        ground_truth = row["report"]
        
        # Load and preprocess image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        if os.path.getsize(img_path) < 100:
            raise ValueError(f"Image file is too small or empty: {img_path} (size={os.path.getsize(img_path)} bytes)")
            
        buf = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to decode image as grayscale: {img_path}")
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_resized)
        img_normalized = img_enhanced.astype(np.float32) / 255.0
        preprocessed_img = np.stack([img_normalized, img_normalized, img_normalized], axis=-1)

        # Generate Reports
        pred_greedy = generate_greedy(model, vocab, preprocessed_img)
        pred_beam   = generate_beam(model, vocab, preprocessed_img, beam_width=args.beam_width)

        # NLP Evaluation - Greedy
        scores_g = compute_bleu(ground_truth, pred_greedy)
        rouge_g = compute_rouge_l(ground_truth, pred_greedy)
        nlp_results["greedy"]["bleu1"].append(scores_g["bleu1"])
        nlp_results["greedy"]["bleu2"].append(scores_g["bleu2"])
        nlp_results["greedy"]["bleu3"].append(scores_g["bleu3"])
        nlp_results["greedy"]["bleu4"].append(scores_g["bleu4"])
        nlp_results["greedy"]["rouge_l"].append(rouge_g)

        # NLP Evaluation - Beam
        scores_b = compute_bleu(ground_truth, pred_beam)
        rouge_b = compute_rouge_l(ground_truth, pred_beam)
        nlp_results["beam"]["bleu1"].append(scores_b["bleu1"])
        nlp_results["beam"]["bleu2"].append(scores_b["bleu2"])
        nlp_results["beam"]["bleu3"].append(scores_b["bleu3"])
        nlp_results["beam"]["bleu4"].append(scores_b["bleu4"])
        nlp_results["beam"]["rouge_l"].append(rouge_b)

        # Clinical evaluation path classifications
        class_gt = classify_report_pathologies(ground_truth)
        class_greedy = classify_report_pathologies(pred_greedy)
        class_beam = classify_report_pathologies(pred_beam)

        for path in PATHOLOGIES.keys():
            gt_val = class_gt[path]
            
            # Evaluate Greedy path
            greedy_val = class_greedy[path]
            if gt_val == 1:
                if greedy_val == 1:
                    clinical_matrix[path]["greedy"]["tp"] += 1
                else:
                    clinical_matrix[path]["greedy"]["fn"] += 1
            else: # negative or unmentioned
                if greedy_val == 1:
                    clinical_matrix[path]["greedy"]["fp"] += 1
                else:
                    clinical_matrix[path]["greedy"]["tn"] += 1

            # Evaluate Beam path
            beam_val = class_beam[path]
            if gt_val == 1:
                if beam_val == 1:
                    clinical_matrix[path]["beam"]["tp"] += 1
                else:
                    clinical_matrix[path]["beam"]["fn"] += 1
            else:
                if beam_val == 1:
                    clinical_matrix[path]["beam"]["fp"] += 1
                else:
                    clinical_matrix[path]["beam"]["tn"] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("                 EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    
    # NLP metrics calculations
    greedy_avg = {k: np.mean(v) * 100 for k, v in nlp_results["greedy"].items()}
    beam_avg = {k: np.mean(v) * 100 for k, v in nlp_results["beam"].items()}

    print(f"\nNLP METRICS:")
    print(f"  Greedy Search:")
    print(f"    • BLEU-1: {greedy_avg['bleu1']:.2f}% | BLEU-2: {greedy_avg['bleu2']:.2f}%")
    print(f"    • BLEU-3: {greedy_avg['bleu3']:.2f}% | BLEU-4: {greedy_avg['bleu4']:.2f}%")
    print(f"    • ROUGE-L: {greedy_avg['rouge_l']:.2f}%")
    
    print(f"  Beam Search (Width {args.beam_width}):")
    print(f"    • BLEU-1: {beam_avg['bleu1']:.2f}% | BLEU-2: {beam_avg['bleu2']:.2f}%")
    print(f"    • BLEU-3: {beam_avg['bleu3']:.2f}% | BLEU-4: {beam_avg['bleu4']:.2f}%")
    print(f"    • ROUGE-L: {beam_avg['rouge_l']:.2f}%")

    # Clinical correctness calculations
    print(f"\nCLINICAL CORRECTNESS (Pathology F1-Scores):")
    
    clinical_summary_md = "| Pathology | Strategy | TP | FP | FN | TN | Precision | Recall | F1-Score |\n"
    clinical_summary_md += "|---|---|---|---|---|---|---|---|---|\n"

    for path in PATHOLOGIES.keys():
        for strategy in ["greedy", "beam"]:
            stats = clinical_matrix[path][strategy]
            tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            clinical_summary_md += f"| {path} | {strategy.capitalize()} | {tp} | {fp} | {fn} | {tn} | {prec*100:.1f}% | {rec*100:.1f}% | {f1*100:.1f}% |\n"
            
            if strategy == "beam":
                print(f"  • {path.capitalize():<18}: Precision: {prec*100:5.1f}% | Recall: {rec*100:5.1f}% | F1: {f1*100:5.1f}%")

    # Generate Markdown report artifact
    md_content = f"""# Model Evaluation Report

- **Evaluated Model**: `{os.path.basename(model_keras_path)}`
- **Total Samples Evaluated**: {args.num_samples}
- **Beam Width**: {args.beam_width}

## 1. NLP Performance Metrics

| Metric | Greedy Search | Beam Search (Width {args.beam_width}) |
|---|---|---|
| **BLEU-1** | {greedy_avg['bleu1']:.2f}% | {beam_avg['bleu1']:.2f}% |
| **BLEU-2** | {greedy_avg['bleu2']:.2f}% | {beam_avg['bleu2']:.2f}% |
| **BLEU-3** | {greedy_avg['bleu3']:.2f}% | {beam_avg['bleu3']:.2f}% |
| **BLEU-4** | {greedy_avg['bleu4']:.2f}% | {beam_avg['bleu4']:.2f}% |
| **ROUGE-L** | {greedy_avg['rouge_l']:.2f}% | {beam_avg['rouge_l']:.2f}% |

## 2. Clinical Correctness Pathologies

{clinical_summary_md}
"""
    output_report_path = os.path.join(model_dir_path, "evaluation_results.md")
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[OK] Complete evaluation report generated at: {output_report_path}")

if __name__ == "__main__":
    main()
