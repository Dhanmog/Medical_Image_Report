import os
import sys
import json
import re
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter

# Project path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)

from app import PositionalEmbedding, TransformerDecoderBlock, TransformerEncoderBlock, SpatialTransformerEncoder, WarmupLRSchedule
from scripts.training.train_densenet121_mimic_v3 import build_model_v3, create_tf_dataset, clean_report_text, extract_disease_labels

def main():
    model_dir_v3 = os.path.join(PROJECT_DIR, "models", "02_densenet121_mimic_v3")
    vocab_path_v3 = os.path.join(model_dir_v3, "tokenizer_vocab.json")

    model_dir_v2 = os.path.join(PROJECT_DIR, "models", "02_densenet121_mimic_v2")
    vocab_path_v2 = os.path.join(model_dir_v2, "tokenizer_vocab.json")
    model_path_v2 = os.path.join(model_dir_v2, "model.keras")

    # ----------------------------------------------------
    # 1. LOAD STABLE VOCABULARY FOR V3
    # ----------------------------------------------------
    with open(vocab_path_v3, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}

    pad_id = token_to_id.get("", 0)
    start_id = token_to_id.get("<start>", -1)
    end_id = token_to_id.get("<end>", -1)
    unk_id = token_to_id.get("[UNK]", -1)

    print("=" * 80)
    print("DIAGNOSTIC CHECKS 1 & 2: SPECIAL TOKENS & TOKENIZER")
    print("=" * 80)
    print(f"Vocab size: {vocab_size}")
    print(f"Special IDs: PAD={pad_id}, START={start_id}, END={end_id}, UNK={unk_id}")

    # Verify every report starts with <start> and ends with <end>
    val_csv = os.path.join(PROJECT_DIR, "datasets", "mimic_cxr", "processed", "mimic_cxr_processed_val.csv")
    df_val = pd.read_csv(val_csv).dropna(subset=["report", "image_path"]).reset_index(drop=True)
    
    tokenizer = keras.layers.TextVectorization(
        max_tokens=8000,
        output_mode="int",
        output_sequence_length=128,
        standardize=None
    )
    tokenizer.set_vocabulary(vocab)

    reports_start_with_start = True
    reports_end_with_end = True
    for r in df_val["report"].iloc[:100]:
        cleaned = clean_report_text(r)
        if not cleaned.startswith("<start>"):
            reports_start_with_start = False
        if not cleaned.endswith("<end>"):
            reports_end_with_end = False

    print(f"Report start/end token check (100 val samples):")
    print(f"  • Every report starts with <START>: {reports_start_with_start}")
    print(f"  • Every report ends with <END>    : {reports_end_with_end}")

    # ----------------------------------------------------
    # 3. CHECK SEQUENCE ALIGNMENT (Using 16 samples to avoid batch drop)
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("DIAGNOSTIC CHECK 3: SEQUENCE ALIGNMENT & SHIFTING")
    print("=" * 80)

    test_df = df_val.iloc[:16].copy()
    test_df["cleaned_report"] = test_df["report"].apply(clean_report_text)
    labels = np.stack(test_df["cleaned_report"].apply(extract_disease_labels).values)
    
    test_ds = create_tf_dataset(test_df, labels, tokenizer, shuffle=False)
    for inputs, targets in test_ds.take(1):
        seq_in = inputs["seq_input"][0].numpy()
        seq_out = targets["vocab_output"][0].numpy()
        
        print(f"Decoder Input (first 10):  {seq_in[:10]}")
        print(f"Decoder Target (first 10): {seq_out[:10]}")
        
        # Verify shifting: target[i] is what model predicts when input is input[i]
        # Meaning target[i] is the next token, which matches input[i+1]
        shifted_correctly = True
        for i in range(len(seq_in) - 1):
            if seq_in[i+1] == 0: # stop checking at padding
                break
            if seq_in[i+1] != seq_out[i]:
                shifted_correctly = False
                break
                
        print(f"Target shifting verification: {'[CORRECT] Target is shifted by exactly 1 token' if shifted_correctly else '[ERROR] Target is not shifted correctly!'}")
        print(f"Reconstructed Input : {[id_to_token.get(t) for t in seq_in[:8] if t != 0]}")
        print(f"Reconstructed Target: {[id_to_token.get(t) for t in seq_out[:8] if t != 0]}")

    # ----------------------------------------------------
    # 4. CHECK LOSS FUNCTION & GRADIENT FLOW
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("DIAGNOSTIC CHECK 4: LOSS FUNCTION & GRADIENTS")
    print("=" * 80)

    class ArgsMock:
        encoder_layers = 2
        encoder_heads = 4
        class_loss_weight = 0.5
    
    import scripts.training.train_densenet121_mimic_v3 as train_module
    train_module.args = ArgsMock()

    model, encoder_model = build_model_v3(vocab_size)
    
    loss_fn_vocab = keras.losses.SparseCategoricalCrossentropy(ignore_class=0)
    loss_fn_class = keras.losses.BinaryCrossentropy()
    
    for inputs, targets in test_ds.take(1):
        images = inputs["image_input"]
        seqs = inputs["seq_input"]
        
        with tf.GradientTape() as tape:
            preds_vocab, preds_class = model([images, seqs], training=True)
            loss_vocab = loss_fn_vocab(targets["vocab_output"], preds_vocab)
            loss_class = loss_fn_class(targets["class_output"], preds_class)
            total_loss = loss_vocab + 0.5 * loss_class
            
        print("Tensor Shapes:")
        print(f"  • Vocab Prediction Shape: {preds_vocab.shape}")
        print(f"  • Vocab Target Shape    : {targets['vocab_output'].shape}")
        print(f"  • Total Loss Value      : {total_loss.numpy():.4f}")
        
        # Test gradient propagation
        trainable_vars = model.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        grads_none_count = sum(1 for g in grads if g is None)
        print(f"Gradient Propagation Check:")
        print(f"  • Trainable variables count: {len(trainable_vars)}")
        print(f"  • Variables with None grads: {grads_none_count}")
        if grads_none_count == 0:
            print("  • [SUCCESS] Gradients reach all parameters correctly.")
        else:
            print("  • [WARNING] Some layers are disconnected from gradient flow!")

    # ----------------------------------------------------
    # 6. CHECK IMAGE FEATURES SWAP TEST
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("DIAGNOSTIC CHECK 6: VISUAL CONDITIONING SWAP TEST")
    print("=" * 80)

    def resolve_image_path(path_str):
        path_str = path_str.replace('\\', '/')
        if os.name == 'posix' and re.match(r'^[a-zA-Z]:', path_str):
            drive = path_str[0].lower()
            path_str = f"/mnt/{drive}{path_str[2:]}"
        match = re.search(r'medical_image_report/(.*)', path_str)
        if match:
            rel_path = os.path.join(PROJECT_DIR, match.group(1))
            if os.path.exists(rel_path):
                return rel_path
        return path_str

    img_path_a = resolve_image_path(df_val.iloc[0]["image_path"])
    img_path_b = resolve_image_path(df_val.iloc[1]["image_path"])
    
    def get_preprocessed_img(p):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (224, 224))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_resized)
        img_normalized = img_enhanced.astype(np.float32) / 255.0
        return np.stack([img_normalized, img_normalized, img_normalized], axis=-1)

    img_a = get_preprocessed_img(img_path_a)
    img_b = get_preprocessed_img(img_path_b)
    
    feats_a = encoder_model(np.expand_dims(img_a, axis=0)).numpy()
    feats_b = encoder_model(np.expand_dims(img_b, axis=0)).numpy()
    
    print(f"Image A Feature Mean: {np.mean(feats_a):.4f} | Std: {np.std(feats_a):.4f}")
    print(f"Image B Feature Mean: {np.mean(feats_b):.4f} | Std: {np.std(feats_b):.4f}")
    diff = np.abs(feats_a - feats_b)
    print(f"Visual Feature Difference - Mean Absolute: {np.mean(diff):.4f} | Max Absolute: {np.max(diff):.4f}")

    # Forward pass outputs
    seq_test = [start_id]
    padded_seq = keras.preprocessing.sequence.pad_sequences([seq_test], maxlen=127, padding="post")
    
    preds_a, _ = model([np.expand_dims(img_a, axis=0), padded_seq], training=False)
    probs_a = preds_a.numpy()[0, 0, :]
    
    preds_b, _ = model([np.expand_dims(img_b, axis=0), padded_seq], training=False)
    probs_b = preds_b.numpy()[0, 0, :]
    
    prob_diff = np.mean(np.abs(probs_a - probs_b))
    print(f"Decoder output absolute difference under swap: {prob_diff:.8f}")

    # ----------------------------------------------------
    # 8. COMPARE DECODING STRATEGIES ON TRAINED MODEL (V2)
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("DIAGNOSTIC CHECK 8: GREEDY VS BEAM DECODING COMPARISON")
    print("=" * 80)

    # Load custom objects for v2 model loading
    custom_objects = {
        "PositionalEmbedding":            PositionalEmbedding,
        "TransformerDecoderBlock":        TransformerDecoderBlock,
        "Custom>PositionalEmbedding":     PositionalEmbedding,
        "Custom>TransformerDecoderBlock": TransformerDecoderBlock,
        "WarmupLRSchedule":               WarmupLRSchedule,
    }
    
    if os.path.exists(model_path_v2):
        print("Loading trained v2 model...")
        model_v2 = keras.models.load_model(model_path_v2, custom_objects=custom_objects, compile=False)
        with open(vocab_path_v2, "r", encoding="utf-8") as f:
            vocab_v2 = json.load(f)
        
        # Test 1 sample image
        from app import generate_report, generate_report_beam
        report_greedy = generate_report(model_v2, vocab_v2, img_a, temperature=0.0)
        report_beam_3 = generate_report_beam(model_v2, vocab_v2, img_a, beam_width=3)
        report_beam_5 = generate_report_beam(model_v2, vocab_v2, img_a, beam_width=5)
        
        print("\nGenerated Reports for Image 1 using Trained v2 Model:")
        print(f"  • Greedy Output      : {report_greedy}")
        print(f"  • Beam Search Width 3: {report_beam_3}")
        print(f"  • Beam Search Width 5: {report_beam_5}")
    else:
        print("[WARNING] Trained v2 model not found, skipping decoding strategy comparison.")

    # ----------------------------------------------------
    # 9. OVERFITTING TEST ON 10 SAMPLES
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("DIAGNOSTIC CHECK 9: OVERFITTING MEMORISATION TEST")
    print("=" * 80)

    # Prepare 10 validation samples
    overfit_df = df_val.iloc[:10].copy()
    overfit_df["cleaned_report"] = overfit_df["report"].apply(clean_report_text)
    overfit_labels = np.stack(overfit_df["cleaned_report"].apply(extract_disease_labels).values)
    
    train_module.BATCH_SIZE = 10
    overfit_ds = create_tf_dataset(overfit_df, overfit_labels, tokenizer, shuffle=False)
    
    # Train on these 10 samples for 20 epochs with learning rate 1e-3
    print("Training model on 10 samples for 20 epochs...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "vocab_output": keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
            "class_output": keras.losses.BinaryCrossentropy()
        },
        loss_weights={
            "vocab_output": 1.0,
            "class_output": 0.5
        }
    )
    
    history = model.fit(
        overfit_ds,
        epochs=20,
        verbose=1
    )
    
    final_loss = history.history["loss"][-1]
    print(f"\nFinal training loss on overfit dataset after 20 epochs: {final_loss:.4f}")
    
    # Check if the model can reproduce reports for these 10 training images
    print("\nGenerating report for overfit Image 1:")
    img_first = get_preprocessed_img(resolve_image_path(overfit_df.iloc[0]["image_path"]))
    from app import generate_report as gr_temp
    gen_text = gr_temp(model, vocab, img_first, temperature=0.0)
    print(f"  • Ground Truth: {overfit_df.iloc[0]['cleaned_report']}")
    print(f"  • Generated   : {gen_text}")

if __name__ == "__main__":
    main()
