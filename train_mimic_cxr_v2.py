import os
import sys
import json
import re
import ast
import argparse
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4

MAX_VOCAB_SIZE = 10000
MAX_REPORT_LENGTH = 100

EMBED_DIM = 256
NUM_HEADS = 4
FF_DIM = 512

START_TOKEN = '<start>'
END_TOKEN = '<end>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'trained_models', 'mimic_cxr')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'mimic_cxr_v2_model.keras')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.keras')
EPOCH_CHECKPOINT_PATTERN = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_{epoch:02d}.keras')
CHECKPOINT_SAVE_PATH = os.path.join(OUTPUT_DIR, 'mimic_cxr_v2_checkpoint.keras')
VOCAB_SAVE_PATH = os.path.join(OUTPUT_DIR, 'tokenizer_vocab.json')
HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, 'training_v2_history.csv')
EVAL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'sample_v2_predictions.txt')

PROC_DIR = os.path.join(PROJECT_DIR, 'processed_dataset_v1')
PROC_TRAIN_CSV = os.path.join(PROC_DIR, 'train.csv')
PROC_VAL_CSV = os.path.join(PROC_DIR, 'validation.csv')
PROC_TEST_CSV = os.path.join(PROC_DIR, 'test.csv')

AUTOTUNE = tf.data.AUTOTUNE

def is_valid_image(image_path):
    if not isinstance(image_path, str):
        image_path = str(image_path)
    if not os.path.exists(image_path):
        return False
    if os.path.getsize(image_path) < 100:
        return False
    try:
        with open(image_path, "rb") as f:
            header = f.read(3)
        return header == b'\xff\xd8\xff'
    except Exception:
        return False

def filter_dataset_images(df, dataset_name="train"):
    paths = df["image_path"].values
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as executor:
        valid_flags = list(executor.map(is_valid_image, paths))

    valid_flags = np.array(valid_flags)
    return df[valid_flags].reset_index(drop=True)

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

def preprocess_xray_tf(image_path_tensor):
    def _py_preprocess(path_bytes):
        if hasattr(path_bytes, 'numpy'):
            path_bytes = path_bytes.numpy()
        if isinstance(path_bytes, bytes):
            path_str = path_bytes.decode('utf-8')
        elif isinstance(path_bytes, np.ndarray):
            raw = path_bytes.item()
            path_str = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
        else:
            path_str = str(path_bytes)
        return preprocess_xray(path_str)

    img_tensor = tf.py_function(
        func=_py_preprocess,
        inp=[image_path_tensor],
        Tout=tf.float32
    )
    img_tensor.set_shape((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    return img_tensor

def clean_report_text(report):
    if not isinstance(report, str):
        return ""
    report = report.lower()
    report = re.sub(r"[^a-z0-9.,;:!?'\- ]+", " ", report)
    report = re.sub(r"\s+", " ", report).strip()
    if not report.startswith(START_TOKEN):
        report = f"{START_TOKEN} {report}"
    if not report.endswith(END_TOKEN):
        report = f"{report} {END_TOKEN}"
    return report

def parse_mimic_csv(csv_path, output_csv_path):
    if os.path.exists(output_csv_path):
        return pd.read_csv(output_csv_path)

    df = pd.read_csv(csv_path)
    records = []

    for idx, row in df.iterrows():
        sub_id = row.get("subject_id", f"sub_{idx}")
        st_id = row.get("study_id", f"st_{idx}")
        img_path = row.get("image_path", "")
        view = row.get("view", "UNKNOWN")
        report = row.get("report", "")

        if isinstance(report, str) and len(report.strip()) > 10 and os.path.exists(img_path):
            cleaned_rep = clean_report_text(report)
            records.append({
                "subject_id": sub_id,
                "study_id": st_id,
                "image_path": img_path,
                "view": view,
                "report": cleaned_rep
            })

    proc_df = pd.DataFrame(records)
    proc_df.to_csv(output_csv_path, index=False)
    return proc_df

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

def build_image_encoder_v2():
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    # Unfreeze top 40 layers for fine-tuning visual features
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="encoder_image_input")
    features = base_model(inputs, training=True)
    features = layers.Reshape((-1, features.shape[-1]))(features)
    features = layers.Dense(EMBED_DIM)(features)
    features = layers.LayerNormalization()(features)
    return keras.Model(inputs, features, name="image_encoder")

def build_mimic_cxr_model_v2(vocab_size):
    encoder = build_image_encoder_v2()
    img_inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image_input")
    seq_inputs = layers.Input(shape=(MAX_REPORT_LENGTH - 1,), name="seq_input")

    img_features = encoder(img_inputs)
    pos_emb = PositionalEmbedding(vocab_size, MAX_REPORT_LENGTH - 1, EMBED_DIM)(seq_inputs)
    decoder_out = TransformerDecoderBlock(EMBED_DIM, FF_DIM, NUM_HEADS)(pos_emb, img_features)
    outputs = layers.Dense(vocab_size, activation="softmax", name="vocab_output")(decoder_out)

    return keras.Model(inputs=[img_inputs, seq_inputs], outputs=outputs, name="mimic_cxr_report_generator_v2")

def create_tf_dataset(df, tokenizer, batch_size=BATCH_SIZE, shuffle=True):
    paths = df["image_path"].values
    reports = df["cleaned_report"].values if "cleaned_report" in df.columns else df["report"].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, reports))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(df), 5000), reshuffle_each_iteration=True)

    def process(image_path, report):
        image = preprocess_xray_tf(image_path)
        tokens = tokenizer(report)
        decoder_input = tokens[:-1]
        decoder_target = tokens[1:]
        return (image, decoder_input), decoder_target

    dataset = dataset.map(process, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Train Image-Conditioned MIMIC-CXR Model (v2)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--resume", nargs="?", const="AUTO", default=None, help="Resume training from checkpoint (path or AUTO)")
    args = parser.parse_args()

    print("=" * 70)
    print("MIMIC-CXR MODEL V2 TRAINING PIPELINE (FINE-TUNED VISION ENCODER)")
    print("=" * 70)

    print("\n[1/5] Loading verified MIMIC-CXR split datasets...")
    train_df = pd.read_csv(PROC_TRAIN_CSV)
    val_df = pd.read_csv(PROC_VAL_CSV)
    test_df = pd.read_csv(PROC_TEST_CSV)

    print(f"  • Training Samples   : {len(train_df):,} (train.csv)")
    print(f"  • Validation Samples : {len(val_df):,} (validation.csv)")
    print(f"  • Test Samples       : {len(test_df):,} (test.csv)")

    with open(VOCAB_SAVE_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    print(f"  • Vocabulary Size : {vocab_size} tokens")

    tokenizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=MAX_REPORT_LENGTH,
        vocabulary=vocab,
        standardize=None
    )

    initial_epoch = 0
    resume_ckpt_path = None

    if args.resume:
        if args.resume != "AUTO" and os.path.exists(args.resume):
            resume_ckpt_path = args.resume
        else:
            # Look for latest epoch checkpoint in CHECKPOINT_DIR
            existing_ckpts = [
                os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR)
                if f.startswith("checkpoint_epoch_") and f.endswith(".keras")
            ]
            if existing_ckpts:
                existing_ckpts.sort()
                resume_ckpt_path = existing_ckpts[-1]

    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        print(f"\n[2/5] Resuming Model v2 from Checkpoint: {resume_ckpt_path}...")
        custom_objects = {
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerDecoderBlock": TransformerDecoderBlock
        }
        model = keras.models.load_model(resume_ckpt_path, custom_objects=custom_objects)
        
        # Determine starting epoch from CSV history or filename
        if os.path.exists(HISTORY_SAVE_PATH):
            try:
                hist_df = pd.read_csv(HISTORY_SAVE_PATH)
                if not hist_df.empty and "epoch" in hist_df.columns:
                    initial_epoch = int(hist_df["epoch"].max()) + 1
            except Exception:
                pass
        if initial_epoch == 0:
            # Parse from filename e.g. checkpoint_epoch_03.keras
            import re
            m = re.search(r"checkpoint_epoch_(\d+)", resume_ckpt_path)
            if m:
                initial_epoch = int(m.group(1))
        print(f"  ✓ Resuming execution at Epoch {initial_epoch + 1} (Completed Epochs: {initial_epoch})")
    else:
        print("\n[2/5] Building Fine-Tuned EfficientNetB0 + Transformer Model v2...")
        model = build_mimic_cxr_model_v2(vocab_size)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(ignore_class=0)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss=loss_fn,
            metrics=["accuracy"]
        )
        print("  ✓ Compiled Model v2 with ignore_class=0 padding mask.")

    print("\n[3/5] Creating TensorFlow dataset pipelines & verifying sample targets...")
    train_ds = create_tf_dataset(train_df, tokenizer, batch_size=args.batch_size, shuffle=True)
    val_ds = create_tf_dataset(val_df, tokenizer, batch_size=args.batch_size, shuffle=False)

    # Explicit Target Diversity Verification (Confirming 1-to-1 Image -> Report mapping)
    for (sample_imgs, sample_dec_in), sample_targets in train_ds.take(1):
        target_0 = sample_targets[0].numpy()
        target_1 = sample_targets[1].numpy()
        is_different = not np.array_equal(target_0, target_1)
        print(f"  • Sample Target 0 (first 10 tokens): {target_0[:10]}")
        print(f"  • Sample Target 1 (first 10 tokens): {target_1[:10]}")
        print(f"  ✓ Target Diversity Verification: {'PASSED (Different images produce different targets)' if is_different else 'FAILED'}")
        break

    callbacks = [
        # Save complete checkpoint after EVERY epoch
        keras.callbacks.ModelCheckpoint(
            filepath=EPOCH_CHECKPOINT_PATTERN,
            save_best_only=False,
            verbose=1
        ),
        # Save best validation model separately
        keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        # Append history if resuming
        keras.callbacks.CSVLogger(HISTORY_SAVE_PATH, append=bool(args.resume and initial_epoch > 0))
    ]

    print(f"\n[4/5] Starting Model v2 Training Run (Target Epochs: {args.epochs}, Batch Size: {args.batch_size})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\n✓ Training completed! Latest model saved to: {MODEL_SAVE_PATH}")
    print(f"✓ Training history saved to: {HISTORY_SAVE_PATH}")

    # Summary report for Epoch completion
    train_loss = history.history["loss"][-1] if "loss" in history.history else "N/A"
    val_loss = history.history["val_loss"][-1] if "val_loss" in history.history else "N/A"
    train_acc = history.history.get("accuracy", [None])[-1]
    val_acc = history.history.get("val_accuracy", [None])[-1]

    print("\n" + "=" * 70)
    print(f"TRAINING STEP FINISHED SUCCESSFULLY")
    print(f"  • Training Samples Processed   : {len(train_df):,}")
    print(f"  • Validation Samples Processed : {len(val_df):,}")
    print(f"  • Training Loss                : {train_loss:.4f}" if isinstance(train_loss, float) else f"  • Training Loss: {train_loss}")
    print(f"  • Validation Loss              : {val_loss:.4f}" if isinstance(val_loss, float) else f"  • Validation Loss: {val_loss}")
    if isinstance(train_acc, float):
        print(f"  • Training Accuracy            : {train_acc*100:.2f}%")
    if isinstance(val_acc, float):
        print(f"  • Validation Accuracy          : {val_acc*100:.2f}%")
    print(f"  • Per-Epoch Checkpoints Dir    : {CHECKPOINT_DIR}")
    print(f"  • Best Model Checkpoint Path   : {BEST_MODEL_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()
