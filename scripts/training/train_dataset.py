import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ICCV_DIR = os.path.join(PROJECT_DIR, "official_data_iccv_final")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Medical Image Report Generator on Custom Dataset")
    parser.add_argument("--csv", type=str, default="mimic_cxr_processed_train.csv", help="Path to input dataset CSV")
    parser.add_argument("--image-col", type=str, default="image_path", help="CSV column name for image file paths")
    parser.add_argument("--text-col", type=str, default="report", help="CSV column name for text reports or findings")
    parser.add_argument("--output-dir", type=str, default="results_model", help="Directory to save model & outputs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-vocab", type=int, default=5000, help="Maximum vocabulary size")
    parser.add_argument("--max-length", type=int, default=60, help="Maximum sequence length")
    parser.add_argument("--num-samples", type=int, default=0, help="Subsample N rows from CSV (0 = use all rows)")
    parser.add_argument("--test-only", action="store_true", help="Perform data loading and single step test only")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if present")
    return parser.parse_args()

args = parse_args()

# Configuration Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr

MAX_VOCAB_SIZE = args.max_vocab
MAX_REPORT_LENGTH = args.max_length

EMBED_DIM = 256
NUM_HEADS = 4
FF_DIM = 512

START_TOKEN = "<start>"
END_TOKEN = "<end>"

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCH_DIR = os.path.join(OUTPUT_DIR, "epoch_checkpoints")
os.makedirs(EPOCH_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "report_generator_model.keras")
CHECKPOINT_SAVE_PATH = os.path.join(OUTPUT_DIR, "report_generator_checkpoint.keras")
VOCAB_SAVE_PATH = os.path.join(OUTPUT_DIR, "tokenizer_vocab.json")
HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, "training_history.csv")
EVAL_SAVE_PATH = os.path.join(OUTPUT_DIR, "sample_predictions.txt")
INVALID_REPORT_PATH = os.path.join(OUTPUT_DIR, "invalid_images_report.txt")

# Standard app directory sync path
APP_RESULTS_DIR = os.path.join(PROJECT_DIR, 'results_shenzhen')
os.makedirs(APP_RESULTS_DIR, exist_ok=True)
APP_MODEL_PATH = os.path.join(APP_RESULTS_DIR, "shenzhen_model.keras")
APP_VOCAB_PATH = os.path.join(APP_RESULTS_DIR, "tokenizer_vocab.json")

AUTOTUNE = tf.data.AUTOTUNE

print("=" * 70)
print("MEDICAL IMAGE REPORT GENERATOR - TRAINING PIPELINE")
print("=" * 70)
print(f"Dataset File   : {args.csv}")
print(f"Image Column   : {args.image_col}")
print(f"Report Column  : {args.text_col}")
print(f"Output Folder  : {OUTPUT_DIR}")
print(f"Epochs / Batch : {EPOCHS} / {BATCH_SIZE}")
print("=" * 70)

def resolve_image_path(path_str):
    """
    Normalizes path slashes and resolves relative dataset image paths against
    official_data_iccv_final and project root.
    """
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

def validate_image(path_str):
    """
    Multi-stage image verification:
    1. Resolve path.
    2. Check existence.
    3. Check non-zero size (>= 100 bytes).
    4. Check image header bytes.
    5. Check OpenCV decoding via np.fromfile + cv2.imdecode.
    6. Check PIL verification.
    Returns (status_str, raw_path, resolved_path)
    """
    resolved = resolve_image_path(path_str)
    
    if not os.path.exists(resolved):
        return "missing", path_str, resolved
        
    if os.path.getsize(resolved) < 100:
        return "zero_byte", path_str, resolved
        
    try:
        with open(resolved, "rb") as f:
            header = f.read(3)
        if header != b"\xff\xd8\xff" and header[:2] != b"\x89P":
            return "corrupted", path_str, resolved
            
        buf = np.fromfile(resolved, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return "corrupted", path_str, resolved
            
        with Image.open(resolved) as im:
            im.verify()
    except Exception:
        return "corrupted", path_str, resolved

    return "valid", path_str, resolved

def validate_and_filter_dataset(df, image_col="image_path", report_col="report", report_save_path=INVALID_REPORT_PATH):
    print("Verifying and validating image file integrity with multi-threading...")
    paths = df[image_col].values
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(validate_image, paths))
        
    valid_flags = []
    resolved_paths = []
    
    status_counts = {"valid": 0, "missing": 0, "zero_byte": 0, "corrupted": 0, "invalid_path": 0}
    problematic_entries = []
    
    for idx, (status, raw_p, res_p) in enumerate(results):
        if status == "valid":
            valid_flags.append(True)
            resolved_paths.append(res_p)
            status_counts["valid"] += 1
        else:
            valid_flags.append(False)
            resolved_paths.append(res_p)
            status_counts[status] = status_counts.get(status, 0) + 1
            subj = df.iloc[idx].get("subject_id", "N/A")
            stud = df.iloc[idx].get("study_id", "N/A")
            problematic_entries.append(f"Status: {status.upper()} | Subject: {subj} | Study: {stud} | Path: {res_p}")

    total_checked = len(df)
    valid_count = status_counts["valid"]
    missing_count = status_counts["missing"]
    corrupted_count = status_counts["zero_byte"] + status_counts["corrupted"]
    invalid_paths_count = status_counts["invalid_path"]

    print("=" * 70)
    print("DATASET VALIDATION & INTEGRITY SUMMARY:")
    print(f" * Total Images Checked : {total_checked}")
    print(f" * Valid Images         : {valid_count}")
    print(f" * Missing Images       : {missing_count}")
    print(f" * Corrupted/Unreadable : {corrupted_count}")
    print(f" * Invalid Paths        : {invalid_paths_count}")
    print("=" * 70)

    if report_save_path:
        log_lines = [
            "DATASET IMAGE VALIDATION & INTEGRITY REPORT",
            "=" * 70,
            f"Total Checked    : {total_checked}",
            f"Valid Images     : {valid_count}",
            f"Missing Images   : {missing_count}",
            f"Corrupted Images : {corrupted_count}",
            f"Invalid Paths    : {invalid_paths_count}",
            "=" * 70,
            f"\nPROBLEMATIC FILES LIST ({len(problematic_entries)} entries):"
        ] + problematic_entries
        
        with open(report_save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"Logged validation details to: {report_save_path}")

    valid_mask = np.array(valid_flags)
    valid_df = df[valid_mask].copy().reset_index(drop=True)
    valid_df[image_col] = np.array(resolved_paths)[valid_mask]
    
    if len(valid_df) == 0:
        raise ValueError("No valid readable image files found after filtering! Check image paths and dataset files.")

    return valid_df, status_counts, problematic_entries

# 1. Image Preprocessing (CLAHE Contrast Enhancement + Normalization)
def preprocess_xray(image_path):
    if hasattr(image_path, 'numpy'):
        image_path = image_path.numpy()
    if isinstance(image_path, bytes):
        image_path = image_path.decode('utf-8')
    image_path = str(image_path)
    
    resolved_path = resolve_image_path(image_path)
    
    try:
        if not os.path.exists(resolved_path) or os.path.getsize(resolved_path) < 100:
            raise ValueError(f"Image missing or zero-byte: {resolved_path}")
            
        buf = np.fromfile(resolved_path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"OpenCV could not decode image: {resolved_path}")
            
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = image.astype(np.float32) / 255.0
        image = np.stack([image, image, image], axis=-1)
        return image
    except Exception as e:
        print(f"[Warning] preprocess_xray error for {resolved_path}: {e}")
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

def preprocess_xray_tf(image_path):
    def _py_preprocess(path_bytes):
        return preprocess_xray(path_bytes)

    image = tf.py_function(func=_py_preprocess, inp=[image_path], Tout=tf.float32)
    image.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return image

# 2. Text Normalization
def clean_report(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9.,;:!?'\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f"{START_TOKEN} {text} {END_TOKEN}"

# 3. Data Loading & Filtering
csv_input_path = os.path.join(PROJECT_DIR, args.csv) if not os.path.isabs(args.csv) else args.csv
if not os.path.exists(csv_input_path):
    raise FileNotFoundError(f"Dataset CSV file not found: {csv_input_path}")

df = pd.read_csv(csv_input_path)

if args.image_col not in df.columns or args.text_col not in df.columns:
    raise ValueError(f"CSV must contain '{args.image_col}' and '{args.text_col}' columns. Found: {df.columns.tolist()}")

df[args.text_col] = df[args.text_col].apply(clean_report)

# Subsample if --num-samples requested
if args.num_samples > 0 and len(df) > args.num_samples:
    print(f"Subsampling {args.num_samples} rows from total {len(df)} dataset rows...")
    df = df.sample(n=args.num_samples, random_state=42).reset_index(drop=True)

# Filter dataset safely while keeping image-report pairs matched
valid_df, status_counts, problematic_entries = validate_and_filter_dataset(
    df, image_col=args.image_col, report_col=args.text_col, report_save_path=INVALID_REPORT_PATH
)

# 85% Train / 15% Validation Split
valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(valid_df) * 0.85)
train_df = valid_df.iloc[:split_idx].reset_index(drop=True)
val_df = valid_df.iloc[split_idx:].reset_index(drop=True)

steps_per_epoch = len(train_df) // BATCH_SIZE
print("=" * 70)
print(f"DATASET METRICS BREAKDOWN:")
print(f" * Total Valid Images in Dataset : {len(valid_df)}")
print(f" * Training Set (85%)            : {len(train_df)} images")
print(f" * Validation Set (15%)          : {len(val_df)} images")
print(f" * Batch Size                    : {BATCH_SIZE} images per step")
print(f" * Steps per Epoch               : {steps_per_epoch} steps")
print("=" * 70)

# 4. Tokenizer & Vocabulary
tokenizer = keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_REPORT_LENGTH,
    standardize=None
)

tokenizer.adapt(tf.data.Dataset.from_tensor_slices(train_df[args.text_col].values).batch(32))
vocab = tokenizer.get_vocabulary()
print(f"Vocabulary Size: {len(vocab)} words")

with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)
print(f"Saved vocabulary to: {VOCAB_SAVE_PATH}")

# 5. tf.data Pipeline
def create_tf_dataset(dataframe, shuffle=True):
    paths = dataframe[args.image_col].values
    reports = dataframe[args.text_col].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, reports))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe), reshuffle_each_iteration=True)

    def process(image_path, report):
        image = preprocess_xray_tf(image_path)
        tokens = tokenizer(report)
        decoder_input = tokens[:-1]
        decoder_target = tokens[1:]
        return (image, decoder_input), decoder_target

    dataset = dataset.map(process, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

train_ds = create_tf_dataset(train_df, shuffle=True)
val_ds = create_tf_dataset(val_df, shuffle=False)

# 6. Model Architecture (CNN Encoder + Transformer Decoder)
def build_encoder():
    base_model = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    features = base_model(inputs, training=False)
    features = layers.Reshape((-1, features.shape[-1]))(features)
    features = layers.Dense(EMBED_DIM)(features)
    features = layers.LayerNormalization()(features)
    return keras.Model(inputs, features, name="image_encoder")

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

def build_model(vocab_size):
    encoder = build_encoder()
    img_inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image_input")
    seq_inputs = layers.Input(shape=(MAX_REPORT_LENGTH - 1,), name="seq_input")

    img_features = encoder(img_inputs)
    pos_emb = PositionalEmbedding(vocab_size, MAX_REPORT_LENGTH - 1, EMBED_DIM)(seq_inputs)
    decoder_out = TransformerDecoderBlock(EMBED_DIM, FF_DIM, NUM_HEADS)(pos_emb, img_features)
    outputs = layers.Dense(vocab_size, activation="softmax")(decoder_out)

    return keras.Model(inputs=[img_inputs, seq_inputs], outputs=outputs, name="report_generator")

# 7. Model Setup & Sanity Test
custom_objects = {
    "PositionalEmbedding": PositionalEmbedding,
    "TransformerDecoderBlock": TransformerDecoderBlock,
    "Custom>PositionalEmbedding": PositionalEmbedding,
    "Custom>TransformerDecoderBlock": TransformerDecoderBlock
}

if args.resume and os.path.exists(CHECKPOINT_SAVE_PATH):
    print(f"Resuming training from checkpoint: {CHECKPOINT_SAVE_PATH}")
    model = keras.models.load_model(CHECKPOINT_SAVE_PATH, custom_objects=custom_objects)
else:
    model = build_model(len(vocab))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

model.summary()

print("\nExecuting Data Loading & Model Batch Sanity Test...")
for (test_img, test_seq), test_target in train_ds.take(1):
    print(f" * Input Image Batch Shape  : {test_img.shape}")
    print(f" * Input Sequence Shape     : {test_seq.shape}")
    print(f" * Target Sequence Shape    : {test_target.shape}")
    test_preds = model.predict([test_img, test_seq], verbose=0)
    print(f" * Prediction Output Shape  : {test_preds.shape}")
    test_loss, test_acc = model.evaluate([test_img, test_seq], test_target, verbose=0)
    print(f" * Test Batch Loss          : {test_loss:.4f}")
    print(f" * Test Batch Accuracy      : {test_acc * 100:.2f}%")
    print(" [OK] Sanity Test PASSED SUCCESSFULLY! No loading errors encountered.")
    break

if args.test_only:
    print("\n--test-only flag specified. Data validation and single-batch sanity check passed cleanly. Exiting before multi-epoch training.")
    exit(0)

class EpochSaveAndSyncCallback(keras.callbacks.Callback):
    def __init__(self, epoch_dir, vocab_save_path, app_model_path, app_vocab_path):
        super().__init__()
        self.epoch_dir = epoch_dir
        self.vocab_save_path = vocab_save_path
        self.app_model_path = app_model_path
        self.app_vocab_path = app_vocab_path

    def on_epoch_end(self, epoch, logs=None):
        import shutil
        epoch_num = epoch + 1
        epoch_file = os.path.join(self.epoch_dir, f"model_epoch_{epoch_num:02d}.keras")
        try:
            self.model.save(epoch_file)
            print(f"\n [Epoch {epoch_num} Saved] Per-epoch model saved to: {epoch_file}")
            
            # Sync latest model & vocab to web app directory
            shutil.copy(epoch_file, self.app_model_path)
            if os.path.exists(self.vocab_save_path):
                shutil.copy(self.vocab_save_path, self.app_vocab_path)
            print(f" [Epoch {epoch_num} Synced] Synced latest model to web app: {os.path.dirname(self.app_model_path)}")
        except Exception as e:
            print(f"\n [Epoch Save Warning]: {e}")

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_SAVE_PATH, save_best_only=False, verbose=1),
    keras.callbacks.ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1),
    keras.callbacks.CSVLogger(HISTORY_SAVE_PATH, append=args.resume),
    EpochSaveAndSyncCallback(EPOCH_DIR, VOCAB_SAVE_PATH, APP_MODEL_PATH, APP_VOCAB_PATH)
]

initial_epoch = 0
if args.resume and os.path.exists(HISTORY_SAVE_PATH):
    try:
        df_hist = pd.read_csv(HISTORY_SAVE_PATH)
        initial_epoch = len(df_hist)
        print(f" [Resume Notice] Found {initial_epoch} completed epochs in history.")
    except Exception as e:
        print(f" [Resume Warning] Could not read history file: {e}")

print("=" * 70)
print(f"Existing checkpoint : {CHECKPOINT_SAVE_PATH}")
print(f"Starting epoch      : Epoch {initial_epoch + 1}")
print(f"Target final epoch  : Epoch {EPOCHS}")
print(f"Training samples    : {len(train_df)}")
print(f"Validation samples  : {len(val_df)}")
print("=" * 70)

print(f"\nStarting Model Training for Epochs {initial_epoch + 1} to {EPOCHS}...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=initial_epoch,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(MODEL_SAVE_PATH)
print(f"Training completed successfully! Saved model to: {MODEL_SAVE_PATH}")

import shutil
try:
    shutil.copy(MODEL_SAVE_PATH, APP_MODEL_PATH)
    shutil.copy(VOCAB_SAVE_PATH, APP_VOCAB_PATH)
    print(f"Synced model & vocabulary to web app directory: {APP_RESULTS_DIR}")
except Exception as e:
    print(f"Sync warning: {e}")

# 8. Sample Inference & Evaluation
def generate_sample_reports(num_samples=5):
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}
    start_id = token_to_id.get(START_TOKEN, 1)
    end_id = token_to_id.get(END_TOKEN, 2)

    lines = ["MEDICAL IMAGE REPORT GENERATION EVALUATION\n", "=" * 60 + "\n"]
    sample_val = val_df.head(num_samples)

    for idx, row in sample_val.iterrows():
        img_path = row[args.image_col]
        actual_report = row[args.text_col]

        img = preprocess_xray(img_path)
        img_input = np.expand_dims(img, axis=0)

        seq = [start_id]
        for _ in range(MAX_REPORT_LENGTH - 2):
            padded_seq = keras.preprocessing.sequence.pad_sequences(
                [seq], maxlen=MAX_REPORT_LENGTH - 1, padding="post"
            )
            preds = model.predict([img_input, padded_seq], verbose=0)
            next_id = np.argmax(preds[0, len(seq) - 1, :])
            if next_id == end_id or next_id == 0:
                break
            seq.append(next_id)

        gen_words = [id_to_token.get(i, "") for i in seq[1:] if i not in (0, start_id, end_id)]
        gen_report = " ".join(gen_words)

        lines.append(f"Image Path : {img_path}\n")
        lines.append(f"Actual     : {actual_report}\n")
        lines.append(f"Generated  : {gen_report}\n")
        lines.append("-" * 60 + "\n")

    with open(EVAL_SAVE_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Sample evaluations saved to: {EVAL_SAVE_PATH}")

try:
    generate_sample_reports()
except Exception as e:
    print(f"Evaluation note: {e}")

print("Done.")
