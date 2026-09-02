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

sys.stdout.flush()

# ============================================================
# 1. CONFIGURATION & DIRECTORY SETUP
# ============================================================
# NOTE: This is the IMPROVED v2 script. Saves to a SEPARATE folder:
#       models/02_densenet121_mimic_v2/
# The original model in models/02_densenet121_mimic/ is NOT touched.
# Review results and manually replace only if satisfied.

parser = argparse.ArgumentParser(
    description="[V2] Improved DenseNet121 + Transformer on MIMIC-CXR"
)
parser.add_argument("--phase1-epochs",   type=int,   default=10,   help="Phase 1 epochs (frozen backbone)")
parser.add_argument("--phase2-epochs",   type=int,   default=20,   help="Phase 2 epochs (fine-tune backbone)")
parser.add_argument("--batch-size",      type=int,   default=16,   help="Batch size")
parser.add_argument("--lr",              type=float, default=1e-4, help="Phase 1 base learning rate")
parser.add_argument("--finetune-lr",     type=float, default=1e-5, help="Phase 2 fine-tune learning rate")
parser.add_argument("--num-samples",     type=int,   default=8000, help="Number of training samples")
parser.add_argument("--unfreeze-layers", type=int,   default=50,   help="Number of DenseNet layers to unfreeze in Phase 2")
args = parser.parse_args()

# scripts/training/ is 2 levels below project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- SEPARATE output directory so original model is preserved ----
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'models', '02_densenet121_mimic_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH        = os.path.join(OUTPUT_DIR, 'model.keras')
PHASE1_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'model_phase1.keras')
VOCAB_SAVE_PATH        = os.path.join(OUTPUT_DIR, 'tokenizer_vocab.json')
HISTORY_SAVE_PATH      = os.path.join(OUTPUT_DIR, 'training_history.csv')
PHASE1_HISTORY_PATH    = os.path.join(OUTPUT_DIR, 'training_history_phase1.csv')
PHASE2_HISTORY_PATH    = os.path.join(OUTPUT_DIR, 'training_history_phase2.csv')

PROC_DIR       = os.path.join(PROJECT_DIR, 'datasets', 'mimic_cxr', 'processed', 'processed_dataset_v1')
PROC_TRAIN_CSV = os.path.join(PROC_DIR, 'train.csv')

IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = args.batch_size
NUM_SAMPLES  = args.num_samples

# ---- Improved hyperparameters ----
MAX_VOCAB_SIZE    = 8000
MAX_REPORT_LENGTH = 128   # increased from 80

EMBED_DIM          = 512   # increased from 256
NUM_HEADS          = 8     # increased from 4
FF_DIM             = 2048  # increased from 512
NUM_DECODER_BLOCKS = 2     # stacked decoder blocks (was 1)
DROPOUT_RATE       = 0.1   # new: dropout for regularisation

START_TOKEN = '<start>'
END_TOKEN   = '<end>'
AUTOTUNE    = tf.data.AUTOTUNE

print("=" * 75)
print("  [V2] IMPROVED DENSENET121 + TRANSFORMER  —  MIMIC-CXR TRAINING")
print("=" * 75)
print(f"  Output Dir        : {OUTPUT_DIR}")
print(f"  Samples           : {NUM_SAMPLES}")
print(f"  Phase 1 Epochs    : {args.phase1_epochs}  (backbone frozen,  LR warmup)")
print(f"  Phase 2 Epochs    : {args.phase2_epochs}  (fine-tune top {args.unfreeze_layers} layers, LR={args.finetune_lr})")
print(f"  Batch Size        : {BATCH_SIZE}")
print(f"  Embed Dim         : {EMBED_DIM}  |  Heads: {NUM_HEADS}  |  FF: {FF_DIM}")
print(f"  Decoder Blocks    : {NUM_DECODER_BLOCKS}  |  Dropout: {DROPOUT_RATE}")
print(f"  Max Report Len    : {MAX_REPORT_LENGTH}")
print("=" * 75, flush=True)

# ============================================================
# 2. DATA LOADING & SAMPLING
# ============================================================

if not os.path.exists(PROC_TRAIN_CSV):
    raise FileNotFoundError(f"Training CSV not found: {PROC_TRAIN_CSV}")

raw_df = pd.read_csv(PROC_TRAIN_CSV)
print(f"\nLoaded {len(raw_df)} records from training CSV.")

# Also try augmented CSV if it exists and we need more samples
AUG_TRAIN_CSV = os.path.join(PROJECT_DIR, 'datasets', 'mimic_cxr', 'processed', 'mimic_cxr_aug_train.csv')
if os.path.exists(AUG_TRAIN_CSV) and len(raw_df) < NUM_SAMPLES:
    aug_df = pd.read_csv(AUG_TRAIN_CSV)
    raw_df = pd.concat([raw_df, aug_df], ignore_index=True).drop_duplicates()
    print(f"Merged augmented CSV -> total records: {len(raw_df)}")

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

# Validate images on disk
print("Validating image files on disk...", flush=True)
valid_mask = []
for p in tqdm(raw_df['image_path'].values, desc="Checking images", unit="img"):
    if isinstance(p, str):
        normalized_p = normalize_and_resolve_path(p)
        valid_mask.append(os.path.exists(normalized_p) and os.path.getsize(normalized_p) > 100)
    else:
        valid_mask.append(False)

df_valid  = raw_df[valid_mask].reset_index(drop=True)
print(f"Valid images: {len(df_valid)}")

df_subset = df_valid.iloc[:NUM_SAMPLES].reset_index(drop=True)
print(f"Using {len(df_subset)} samples.")

split_idx = int(len(df_subset) * 0.9)
train_df  = df_subset.iloc[:split_idx].reset_index(drop=True)
val_df    = df_subset.iloc[split_idx:].reset_index(drop=True)
print(f"Train: {len(train_df)}  |  Val: {len(val_df)}", flush=True)

# ============================================================
# 3. PREPROCESSING
# ============================================================

def preprocess_xray(image_path):
    image_path = normalize_and_resolve_path(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    if os.path.getsize(image_path) < 100:
        raise ValueError(f"Image file is too small or empty: {image_path} (size={os.path.getsize(image_path)} bytes)")
        
    img_np = np.fromfile(image_path, dtype=np.uint8)
    image  = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to decode image as grayscale: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = image.astype(np.float32) / 255.0
    return np.stack([image, image, image], axis=-1)


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


train_df["cleaned_report"] = train_df["report"].astype(str).apply(clean_report_text)
val_df["cleaned_report"]   = val_df["report"].astype(str).apply(clean_report_text)

# ============================================================
# 4. TOKENIZER & VOCABULARY
# ============================================================

train_reports = train_df["cleaned_report"].values
val_reports   = val_df["cleaned_report"].values

tokenizer = keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_REPORT_LENGTH,
    standardize=None
)
tokenizer.adapt(tf.data.Dataset.from_tensor_slices(train_reports).batch(64))
vocab = tokenizer.get_vocabulary()
print(f"\nVocabulary size: {len(vocab)}")

with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)
print(f"Saved vocabulary -> {VOCAB_SAVE_PATH}")

# ============================================================
# 5. TF DATASETS
# ============================================================

def create_tf_dataset(df, shuffle=True):
    paths   = df["image_path"].values
    reports = df["cleaned_report"].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, reports))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    def process(image_path, report):
        image          = preprocess_xray_tf(image_path)
        tokens         = tokenizer(report)
        decoder_input  = tokens[:-1]
        decoder_target = tokens[1:]
        return (image, decoder_input), decoder_target

    dataset = (dataset
               .map(process, num_parallel_calls=AUTOTUNE)
               .batch(BATCH_SIZE, drop_remainder=True)
               .prefetch(AUTOTUNE))
    return dataset


train_ds = create_tf_dataset(train_df, shuffle=True)
val_ds   = create_tf_dataset(val_df,   shuffle=False)

# ============================================================
# 6. WARMUP LR SCHEDULE
# ============================================================

@keras.utils.register_keras_serializable()
class WarmupLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Transformer-style warmup + inverse-sqrt decay."""
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)
        self.d_model      = float(d_model)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step + 1.0)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

# ============================================================
# 7. MODEL ARCHITECTURE (Improved)
# ============================================================

@keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.embed_dim  = embed_dim
        self.token_emb  = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_emb    = layers.Embedding(seq_len, embed_dim)
        self.dropout    = layers.Dropout(DROPOUT_RATE)

    def call(self, inputs, training=False):
        length    = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        x = self.token_emb(inputs) + self.pos_emb(positions)
        return self.dropout(x, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"vocab_size": self.vocab_size, "seq_len": self.seq_len, "embed_dim": self.embed_dim})
        return cfg


@keras.utils.register_keras_serializable()
class TransformerDecoderBlock(layers.Layer):
    """Transformer decoder block with dropout + GELU FFN."""
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim    = embed_dim
        self.dense_dim    = dense_dim
        self.num_heads    = num_heads
        self.dropout_rate = dropout_rate

        self.mha_causal = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.mha_cross  = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dense_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])
        self.ln1      = layers.LayerNormalization(epsilon=1e-6)
        self.ln2      = layers.LayerNormalization(epsilon=1e-6)
        self.ln3      = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_outputs, training=False):
        attn1 = self.mha_causal(query=inputs, value=inputs, key=inputs, use_causal_mask=True)
        attn1 = self.dropout1(attn1, training=training)
        out1  = self.ln1(inputs + attn1)

        attn2 = self.mha_cross(query=out1, value=encoder_outputs, key=encoder_outputs)
        attn2 = self.dropout2(attn2, training=training)
        out2  = self.ln2(out1 + attn2)

        ffn_out = self.ffn(out2, training=training)
        ffn_out = self.dropout3(ffn_out, training=training)
        return self.ln3(out2 + ffn_out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim":    self.embed_dim,
            "dense_dim":    self.dense_dim,
            "num_heads":    self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


def build_densenet_encoder():
    """DenseNet121 pretrained on ImageNet — frozen during Phase 1."""
    base_model = keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    base_model.trainable = False

    inputs   = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="encoder_image_input")
    features = base_model(inputs, training=False)             # (batch, 7, 7, 1024)
    features = layers.Reshape((-1, features.shape[-1]))(features)  # (batch, 49, 1024)
    features = layers.Dense(EMBED_DIM)(features)
    features = layers.LayerNormalization(epsilon=1e-6)(features)
    features = layers.Dropout(DROPOUT_RATE)(features)
    return keras.Model(inputs, features, name="densenet121_encoder")


def build_model_v2(vocab_size):
    encoder    = build_densenet_encoder()
    img_inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image_input")
    seq_inputs = layers.Input(shape=(MAX_REPORT_LENGTH - 1,), name="seq_input")

    img_features = encoder(img_inputs)

    x = PositionalEmbedding(vocab_size, MAX_REPORT_LENGTH - 1, EMBED_DIM, name="pos_embedding")(seq_inputs)
    for i in range(NUM_DECODER_BLOCKS):
        x = TransformerDecoderBlock(
            EMBED_DIM, FF_DIM, NUM_HEADS, DROPOUT_RATE,
            name=f"decoder_block_{i+1}"
        )(x, img_features)

    outputs = layers.Dense(vocab_size, activation="softmax", name="vocab_output")(x)

    return keras.Model(
        inputs=[img_inputs, seq_inputs],
        outputs=outputs,
        name="densenet121_mimic_v2"
    ), encoder

# ============================================================
# 8. PHASE 1 — FROZEN BACKBONE TRAINING
# ============================================================

print("\n" + "=" * 75)
print("  PHASE 1: Training with frozen DenseNet121 backbone")
print("=" * 75, flush=True)

model, encoder_model = build_model_v2(len(vocab))
model.summary()

lr_schedule_p1 = WarmupLRSchedule(d_model=EMBED_DIM, warmup_steps=2000)

model.compile(
    optimizer=keras.optimizers.Adam(lr_schedule_p1),
    loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
    metrics=["accuracy"]
)

callbacks_p1 = [
    keras.callbacks.ModelCheckpoint(
        PHASE1_MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
    ),
    # NOTE: No ReduceLROnPlateau here — it conflicts with LearningRateSchedule.
    # The WarmupLRSchedule already handles LR decay automatically.
]

history_p1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.phase1_epochs,
    verbose=1,
    callbacks=callbacks_p1
)

hist_p1_df = pd.DataFrame(history_p1.history)
hist_p1_df.to_csv(PHASE1_HISTORY_PATH, index=False)
best_p1_acc = max(history_p1.history.get('val_accuracy', [0]))
print(f"\nPhase 1 complete. Best val_accuracy: {best_p1_acc:.4f}")
print(f"Phase 1 model -> {PHASE1_MODEL_SAVE_PATH}")

# ============================================================
# 9. PHASE 2 — FINE-TUNE (unfreeze top N DenseNet layers)
# ============================================================

print("\n" + "=" * 75)
print(f"  PHASE 2: Fine-tuning top {args.unfreeze_layers} DenseNet121 layers")
print("=" * 75, flush=True)

densenet_base = encoder_model.get_layer("densenet121")
densenet_base.trainable = True
total_layers  = len(densenet_base.layers)
freeze_until  = total_layers - args.unfreeze_layers
for i, layer in enumerate(densenet_base.layers):
    layer.trainable = (i >= freeze_until)

trainable_count = sum(1 for l in densenet_base.layers if l.trainable)
print(f"DenseNet layers unfrozen: {trainable_count} / {total_layers}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=args.finetune_lr),
    loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
    metrics=["accuracy"]
)

callbacks_p2 = [
    keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1
    )
]

history_p2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.phase2_epochs,
    verbose=1,
    callbacks=callbacks_p2
)

# ============================================================
# 10. SAVE ALL RESULTS
# ============================================================

hist_p2_df   = pd.DataFrame(history_p2.history)
hist_p2_df.to_csv(PHASE2_HISTORY_PATH, index=False)

combined_df = pd.concat([hist_p1_df, hist_p2_df], ignore_index=True)
combined_df.to_csv(HISTORY_SAVE_PATH, index=False)

model.save(MODEL_SAVE_PATH)
best_p2_acc = max(history_p2.history.get('val_accuracy', [0]))

print("\n" + "=" * 75)
print("  TRAINING COMPLETE — V2 RESULTS SUMMARY")
print("=" * 75)
print(f"  Phase 1 best val_accuracy : {best_p1_acc:.4f}")
print(f"  Phase 2 best val_accuracy : {best_p2_acc:.4f}")
print(f"\n  All outputs in: {OUTPUT_DIR}")
print(f"    model.keras                  <- final fine-tuned model (best)")
print(f"    model_phase1.keras           <- phase 1 checkpoint")
print(f"    tokenizer_vocab.json         <- vocabulary")
print(f"    training_history.csv         <- combined history")
print(f"    training_history_phase1.csv  <- phase 1 only")
print(f"    training_history_phase2.csv  <- phase 2 only")
print("=" * 75)
print("\nTo replace the existing model with this improved one, copy:")
print(f"  {OUTPUT_DIR}/model.keras")
print(f"  {OUTPUT_DIR}/tokenizer_vocab.json")
print("To:")
print(f"  {os.path.join(PROJECT_DIR, 'models', '02_densenet121_mimic')}/")
print("Only do this after confirming v2 accuracy > 26.8%")
print("=" * 75)
