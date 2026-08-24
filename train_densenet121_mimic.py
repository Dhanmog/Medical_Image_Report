import os
import sys
import json
import re
import argparse
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Force unbuffered output so progress bars display continuously in terminal
sys.stdout.flush()

# ============================================================
# 1. CONFIGURATION & DIRECTORY SETUP
# ============================================================

parser = argparse.ArgumentParser(description="Train DenseNet121 + Transformer on MIMIC-CXR (4000 images)")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--num-samples", type=int, default=4000, help="Number of training samples to use")
args = parser.parse_args()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'trained_models', 'densenet121_mimic')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Also ensure results_shenzhen or app fallback dir exists for compatibility
APP_RESULTS_DIR = os.path.join(PROJECT_DIR, 'results_shenzhen')
os.makedirs(APP_RESULTS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'densenet121_mimic_model.keras')
APP_MODEL_SAVE_PATH = os.path.join(APP_RESULTS_DIR, 'shenzhen_model.keras')
VOCAB_SAVE_PATH = os.path.join(OUTPUT_DIR, 'tokenizer_vocab.json')
APP_VOCAB_SAVE_PATH = os.path.join(APP_RESULTS_DIR, 'tokenizer_vocab.json')
HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, 'training_history.csv')

PROC_DIR = os.path.join(PROJECT_DIR, 'processed_dataset_v1')
PROC_TRAIN_CSV = os.path.join(PROC_DIR, 'train.csv')
PROC_VAL_CSV = os.path.join(PROC_DIR, 'validation.csv')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr
NUM_SAMPLES = args.num_samples

MAX_VOCAB_SIZE = 8000
MAX_REPORT_LENGTH = 80

EMBED_DIM = 256
NUM_HEADS = 4
FF_DIM = 512

START_TOKEN = '<start>'
END_TOKEN = '<end>'
AUTOTUNE = tf.data.AUTOTUNE

print("=" * 75)
print("TRAINING DENSENET121 + TRANSFORMER MODEL ON MIMIC-CXR DATASET")
print("=" * 75)
print(f"Target Samples : {NUM_SAMPLES}")
print(f"Epochs         : {EPOCHS}")
print(f"Batch Size     : {BATCH_SIZE}")
print(f"Learning Rate  : {LEARNING_RATE}")
print(f"Output Path    : {MODEL_SAVE_PATH}")
print("=" * 75, flush=True)

# ============================================================
# 2. DATA PROCESSING & SAMPLING
# ============================================================

if not os.path.exists(PROC_TRAIN_CSV):
    raise FileNotFoundError(f"Training dataset file not found: {PROC_TRAIN_CSV}")

raw_df = pd.read_csv(PROC_TRAIN_CSV)
print(f"Loaded total training CSV records: {len(raw_df)}")

# Filter valid images existing on disk with progress bar
print("Validating X-ray image files on disk...", flush=True)
valid_mask = []
for p in tqdm(raw_df['image_path'].values, desc="Checking Images", unit="img"):
    valid_mask.append(isinstance(p, str) and os.path.exists(p) and os.path.getsize(p) > 100)

df_valid = raw_df[valid_mask].reset_index(drop=True)
print(f"Valid images found on disk: {len(df_valid)}")

# Take exact specified subset of 4000 images
df_subset = df_valid.iloc[:NUM_SAMPLES].reset_index(drop=True)
print(f"Selected dataset subset size: {len(df_subset)}")

# 90% Train / 10% Val split from the 4000 samples
split_idx = int(len(df_subset) * 0.9)
train_df = df_subset.iloc[:split_idx].reset_index(drop=True)
val_df = df_subset.iloc[split_idx:].reset_index(drop=True)

print(f"Training set split size   : {len(train_df)}")
print(f"Validation set split size : {len(val_df)}", flush=True)

# ============================================================
# 3. PREPROCESSING FUNCTIONS
# ============================================================

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

    image = tf.py_function(func=_py_preprocess, inp=[image_path_tensor], Tout=tf.float32)
    image.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return image

def clean_report_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9.,;:!?'\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f"{START_TOKEN} {text} {END_TOKEN}"

train_reports = train_df["report"].astype(str).apply(clean_report_text).values
val_reports = val_df["report"].astype(str).apply(clean_report_text).values

train_df["cleaned_report"] = train_reports
val_df["cleaned_report"] = val_reports

# ============================================================
# 4. TOKENIZER & VOCABULARY
# ============================================================

tokenizer = keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_REPORT_LENGTH,
    standardize=None
)

tokenizer.adapt(tf.data.Dataset.from_tensor_slices(train_reports).batch(64))
vocab = tokenizer.get_vocabulary()
print(f"Vocabulary Size (MIMIC 4000 Subset): {len(vocab)}")

with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)
with open(APP_VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

print(f"Saved vocabulary to: {VOCAB_SAVE_PATH} and {APP_VOCAB_SAVE_PATH}")

# ============================================================
# 5. TENSORFLOW DATASETS
# ============================================================

def create_tf_dataset(df, shuffle=True):
    paths = df["image_path"].values
    reports = df["cleaned_report"].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, reports))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

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

# ============================================================
# 6. DENSENET121 ENCODER & TRANSFORMER DECODER ARCHITECTURE
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

def build_densenet_encoder():
    # DenseNet121 pretrained on ImageNet
    base_model = keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    # Freeze backbone initially for stable training
    base_model.trainable = False

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="encoder_image_input")
    features = base_model(inputs, training=False)  # Shape: (batch, 7, 7, 1024)
    features = layers.Reshape((-1, features.shape[-1]))(features)  # Shape: (batch, 49, 1024)
    features = layers.Dense(EMBED_DIM)(features)
    features = layers.LayerNormalization()(features)
    return keras.Model(inputs, features, name="densenet121_encoder")

def build_densenet_mimic_model(vocab_size):
    encoder = build_densenet_encoder()
    img_inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image_input")
    seq_inputs = layers.Input(shape=(MAX_REPORT_LENGTH - 1,), name="seq_input")

    img_features = encoder(img_inputs)
    pos_emb = PositionalEmbedding(vocab_size, MAX_REPORT_LENGTH - 1, EMBED_DIM)(seq_inputs)
    decoder_out = TransformerDecoderBlock(EMBED_DIM, FF_DIM, NUM_HEADS)(pos_emb, img_features)
    outputs = layers.Dense(vocab_size, activation="softmax", name="vocab_output")(decoder_out)

    return keras.Model(inputs=[img_inputs, seq_inputs], outputs=outputs, name="densenet121_mimic_report_generator")

model = build_densenet_mimic_model(len(vocab))
model.summary()

# ============================================================
# 7. MODEL COMPILATION & TRAINING
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
]

print("\nStarting Training on DenseNet121 + MIMIC CXR (4000 images subset)...", flush=True)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks
)

# Save final model to primary model paths
model.save(MODEL_SAVE_PATH)
model.save(APP_MODEL_SAVE_PATH)
print(f"\nSaved trained DenseNet121 model to:\n  - {MODEL_SAVE_PATH}\n  - {APP_MODEL_SAVE_PATH}")

# Save training history
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(HISTORY_SAVE_PATH, index=False)
print(f"Saved training history to: {HISTORY_SAVE_PATH}")
print("=" * 75)
print("DENSENET121 MIMIC-CXR TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 75)
