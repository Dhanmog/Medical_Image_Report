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

parser = argparse.ArgumentParser(description="Train Shenzhen Chest X-ray Report Generator")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--test-only", action="store_true", help="Perform data loading and single step test only")
parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint if available")
args = parser.parse_args()

# ============================================================
# 1. CONFIGURATION & ISOLATED OUTPUT SETUP
# ============================================================

# scripts/training/ is 2 levels below project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(PROJECT_DIR, 'datasets', 'shenzhen', 'shenzhen_processed.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'models', '04_cnn_transformer_shenzhen')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

MAX_VOCAB_SIZE = 5000
MAX_REPORT_LENGTH = 60

EMBED_DIM = 256
NUM_HEADS = 4
FF_DIM = 512

AUTOTUNE = tf.data.AUTOTUNE

START_TOKEN = "<start>"
END_TOKEN = "<end>"

print("=" * 70)
print("TRAINING MODEL ON NEW SHENZHEN CHEST X-RAY DATASET ONLY")
print("=" * 70)
print(f"Dataset File : {INPUT_CSV}")
print(f"Output Dir   : {OUTPUT_DIR}")
print("=" * 70)

# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

def preprocess_xray(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        # Fallback if image path has issue
        image = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    else:
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    image = image.astype(np.float32) / 255.0
    image = np.stack([image, image, image], axis=-1)
    return image

def preprocess_xray_tf(image_path):
    def _py_preprocess(path_bytes):
        if hasattr(path_bytes, "numpy"):
            path_bytes = path_bytes.numpy()
        if isinstance(path_bytes, bytes):
            path_str = path_bytes.decode('utf-8')
        else:
            path_str = str(path_bytes)
        return preprocess_xray(path_str)

    image = tf.py_function(func=_py_preprocess, inp=[image_path], Tout=tf.float32)
    image.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return image

# ============================================================
# 3. REPORT CLEANING & DATASET LOADING
# ============================================================

def clean_report(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9.,;:!?'\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f"{START_TOKEN} {text} {END_TOKEN}"

df = pd.read_csv(INPUT_CSV)
df["report"] = df["report"].apply(clean_report)

# Train/Val Split (85% train, 15% val)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(df) * 0.85)
train_df = df.iloc[:split_idx]
val_df = df.iloc[split_idx:]

print(f"Total Shenzhen Samples: {len(df)}")
print(f"Training Samples      : {len(train_df)}")
print(f"Validation Samples    : {len(val_df)}")

# ============================================================
# 4. TOKENIZER & VOCABULARY
# ============================================================

tokenizer = keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_REPORT_LENGTH,
    standardize=None
)

tokenizer.adapt(tf.data.Dataset.from_tensor_slices(train_df["report"].values).batch(32))

vocab = tokenizer.get_vocabulary()
print(f"Shenzhen Dataset Vocab Size: {len(vocab)}")

# Save Vocab to Output Directory
vocab_path = os.path.join(OUTPUT_DIR, "tokenizer_vocab.json")
with open(vocab_path, "w") as f:
    json.dump(vocab, f, indent=2)
print(f"Saved vocabulary to: {vocab_path}")

# ============================================================
# 5. TENSORFLOW DATASETS
# ============================================================

def create_tf_dataset(dataframe, shuffle=True):
    paths = dataframe["image_path"].values
    reports = dataframe["report"].values

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

# ============================================================
# 6. MODEL ARCHITECTURE (CNN Encoder + Transformer Decoder)
# ============================================================

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

    return keras.Model(inputs=[img_inputs, seq_inputs], outputs=outputs, name="shenzhen_report_generator")

# ============================================================
# 7. TRAINING & CHECKPOINTING
# ============================================================

model_save_path = os.path.join(OUTPUT_DIR, "shenzhen_model.keras")
checkpoint_save_path = os.path.join(OUTPUT_DIR, "shenzhen_checkpoint.keras")
history_save_path = os.path.join(OUTPUT_DIR, "training_history.csv")

start_epoch = 0
if args.resume or os.path.exists(checkpoint_save_path):
    if os.path.exists(history_save_path):
        try:
            hist_df = pd.read_csv(history_save_path)
            if not hist_df.empty and "epoch" in hist_df.columns:
                hist_df["epoch"] = pd.to_numeric(hist_df["epoch"], errors="coerce")
                valid_epochs = hist_df["epoch"].dropna()
                if not valid_epochs.empty:
                    start_epoch = int(valid_epochs.max()) + 1
        except Exception:
            pass

resume_ckpt = None
if args.resume:
    if os.path.exists(checkpoint_save_path):
        resume_ckpt = checkpoint_save_path
    elif os.path.exists(model_save_path):
        resume_ckpt = model_save_path

if resume_ckpt:
    print(f"  • Resuming Shenzhen training from checkpoint: {resume_ckpt} (Starting at epoch {start_epoch})")
    custom_objects = {
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoderBlock": TransformerDecoderBlock
    }
    model = keras.models.load_model(resume_ckpt, custom_objects=custom_objects)
else:
    if args.resume:
        print("  • --resume specified, but no existing checkpoint found. Initializing new model.")
    model = build_model(len(vocab))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

model.summary()

if args.test_only:
    print("\nTest-only flag set. Skipping training execution.")
else:
    callbacks = [
        # Save latest epoch checkpoint for resuming
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_path,
            save_best_only=False,
            verbose=1
        ),
        # Save best model based on validation loss
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        keras.callbacks.CSVLogger(history_save_path, append=bool(args.resume and start_epoch > 0))
    ]

    if start_epoch >= args.epochs:
        print(f"\nTarget epochs ({args.epochs}) already completed (current progress: epoch {start_epoch}).")
        print(f"To train further, increase total epochs (e.g. --epochs {start_epoch + 5}).")
    else:
        print(f"\nStarting Shenzhen Training Run (Epoch {start_epoch + 1} to {args.epochs}, Batch Size {args.batch_size})...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            initial_epoch=start_epoch,
            callbacks=callbacks
        )

        model.save(model_save_path)
        print(f"\nTraining completed! Model saved to: {model_save_path}")
        print(f"Training history saved to: {history_save_path}")

# ============================================================
# 8. GENERATE SAMPLE REPORT EVALUATION
# ============================================================

def generate_sample_reports(num_samples=5):
    eval_path = os.path.join(OUTPUT_DIR, "sample_predictions.txt")
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}
    start_id = token_to_id.get(START_TOKEN, 1)
    end_id = token_to_id.get(END_TOKEN, 2)

    lines = ["SHENZHEN DATASET SAMPLE REPORT GENERATION EVALUATION\n", "=" * 60 + "\n"]

    sample_val = val_df.head(num_samples)
    for idx, row in sample_val.iterrows():
        img_path = row["image_path"]
        actual_report = row["report"]
        
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

        lines.append(f"Image File : {row['study_id']}\n")
        lines.append(f"Actual     : {actual_report}\n")
        lines.append(f"Generated  : {gen_report}\n")
        lines.append("-" * 60 + "\n")

    with open(eval_path, "w") as f:
        f.writelines(lines)

    print(f"Sample evaluations saved to: {eval_path}")

try:
    generate_sample_reports()
except Exception as e:
    print(f"Evaluation note: {e}")

print("=" * 70)
print("SHENZHEN WORKFLOW COMPLETED SUCCESSFULLY!")
print("=" * 70)
