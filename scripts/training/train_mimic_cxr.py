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
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCH_DIR = os.path.join(OUTPUT_DIR, 'epoch_checkpoints')
os.makedirs(EPOCH_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'mimic_cxr_model.keras')
CHECKPOINT_SAVE_PATH = os.path.join(OUTPUT_DIR, 'mimic_cxr_checkpoint.keras')
VOCAB_SAVE_PATH = os.path.join(OUTPUT_DIR, 'tokenizer_vocab.json')
HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, 'training_history.csv')
EVAL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'sample_predictions.txt')
INVALID_REPORT_PATH = os.path.join(OUTPUT_DIR, 'invalid_images_report.txt')

PROC_TRAIN_CSV = os.path.join(PROJECT_DIR, 'mimic_cxr_processed_train.csv')
PROC_VAL_CSV = os.path.join(PROJECT_DIR, 'mimic_cxr_processed_val.csv')

AUTOTUNE = tf.data.AUTOTUNE

def resolve_image_path(path_str):
    if not isinstance(path_str, str):
        path_str = str(path_str)
    norm_path = path_str.replace("/", os.sep).replace("\\", os.sep)
    if os.path.isabs(norm_path) and os.path.exists(norm_path):
        return norm_path
    cand1 = os.path.normpath(os.path.join(PROJECT_DIR, norm_path))
    if os.path.exists(cand1):
        return cand1
    iccv_dir = os.path.join(PROJECT_DIR, "official_data_iccv_final")
    cand2 = os.path.normpath(os.path.join(iccv_dir, norm_path))
    if os.path.exists(cand2):
        return cand2
    if norm_path.startswith("official_data_iccv_final"):
        cand3 = os.path.normpath(os.path.join(PROJECT_DIR, norm_path))
        if os.path.exists(cand3):
            return cand3
    return cand2 if norm_path.startswith("files") else cand1

def is_valid_image(image_path):
    resolved = resolve_image_path(image_path)
    if not os.path.exists(resolved) or os.path.getsize(resolved) < 100:
        return False
    try:
        with open(resolved, "rb") as f:
            header = f.read(3)
        if header != b"\xff\xd8\xff" and header[:2] != b"\x89P":
            return False
        buf = np.fromfile(resolved, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return False
    except Exception:
        return False
    return True

filter_dataset_images = None # defined below

def filter_dataset_images(df, dataset_name="train"):
    paths = df["image_path"].values
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=32) as executor:
        valid_flags = list(executor.map(is_valid_image, paths))

    valid_flags = np.array(valid_flags)
    valid_df = df[valid_flags].reset_index(drop=True)
    invalid_df = df[~valid_flags].reset_index(drop=True)

    if len(invalid_df) > 0:
        print(f"  • Warning: Found {len(invalid_df)} invalid/corrupted image(s) in {dataset_name} set.", flush=True)
        log_lines = []
        if os.path.exists(INVALID_REPORT_PATH):
            with open(INVALID_REPORT_PATH, "r", encoding="utf-8") as f:
                log_lines = f.read().splitlines()
        else:
            log_lines = ["INVALID / UNREADABLE MIMIC-CXR IMAGES REPORT", "=" * 65]

        log_lines.append(f"\n--- {dataset_name.upper()} DATASET INVALID IMAGES ({len(invalid_df)} files) ---")
        for idx, row in invalid_df.iterrows():
            sub_id = row.get("subject_id", "N/A")
            st_id = row.get("study_id", "N/A")
            img_path = row.get("image_path", "N/A")
            log_lines.append(f"Subject: {sub_id} | Study: {st_id} | Path: {img_path}")

        with open(INVALID_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"  • Logged invalid images report to: {INVALID_REPORT_PATH}", flush=True)

    return valid_df

def preprocess_xray(image_path):
    if not isinstance(image_path, str):
        image_path = str(image_path)
    resolved = resolve_image_path(image_path)
    try:
        if not os.path.exists(resolved) or os.path.getsize(resolved) < 100:
            raise ValueError(f"File missing or truncated: {resolved}")
        img_np = np.fromfile(resolved, dtype=np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"Could not decode image file: {resolved}")

        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = image.astype(np.float32) / 255.0
        image = np.stack([image, image, image], axis=-1)
        return image
    except Exception as e:
        print(f"  ⚠️ Warning in preprocess_xray for {resolved}: {e}")
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

def preprocess_xray_tf(image_path_tensor):
    def _py_preprocess(path_obj):
        if hasattr(path_obj, 'numpy'):
            path_obj = path_obj.numpy()
        if isinstance(path_obj, bytes):
            path_str = path_obj.decode('utf-8')
        else:
            path_str = str(path_obj)
        return preprocess_xray(path_str)

    image = tf.py_function(
        func=_py_preprocess,
        inp=[image_path_tensor],
        Tout=tf.float32
    )
    image.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return image

def clean_report(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9.,;:!?\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f"{START_TOKEN} {text} {END_TOKEN}"

def parse_mimic_csv(csv_path, save_proc_path=None):
    if save_proc_path and os.path.exists(save_proc_path):
        print(f"  • Loading pre-processed CSV from: {save_proc_path}")
        return pd.read_csv(save_proc_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    records = []

    for idx, row in df.iterrows():
        subj_id = row.get("subject_id", f"patient_{idx}")
        try:
            imgs = ast.literal_eval(row["image"]) if isinstance(row["image"], str) else []
            texts = ast.literal_eval(row["text"]) if isinstance(row["text"], str) else []
            views = ast.literal_eval(row["view"]) if ("view" in row and isinstance(row["view"], str)) else []
        except Exception:
            continue

        if not imgs or not texts:
            continue

        study_dict = {}
        for i, img_rel in enumerate(imgs):
            parts = img_rel.replace("\\", "/").split("/")
            st_id = parts[3] if len(parts) >= 4 else f"study_{i}"
            if st_id not in study_dict:
                study_dict[st_id] = []
            view = views[i] if i < len(views) else "FRONTAL"
            study_dict[st_id].append((img_rel, view))

        study_keys = list(study_dict.keys())

        for s_idx, st_id in enumerate(study_keys):
            report_text = texts[s_idx] if s_idx < len(texts) else (texts[0] if len(texts) > 0 else "")
            if not report_text or not str(report_text).strip():
                continue

            cleaned_rep = clean_report(report_text)
            img_tuples = study_dict[st_id]

            for img_rel, view in img_tuples:
                full_path = os.path.join(PROJECT_DIR, "official_data_iccv_final", img_rel.replace("/", os.sep))
                if os.path.exists(full_path):
                    records.append({
                        "subject_id": subj_id,
                        "study_id": st_id,
                        "image_path": full_path,
                        "view": view,
                        "report": cleaned_rep
                    })

    res_df = pd.DataFrame(records)
    if save_proc_path:
        res_df.to_csv(save_proc_path, index=False)
        print(f"  • Saved processed dataset cache to: {save_proc_path}")
    return res_df

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

def build_image_encoder():
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="encoder_image_input")
    features = base_model(inputs, training=False)
    features = layers.Reshape((-1, features.shape[-1]))(features)
    features = layers.Dense(EMBED_DIM)(features)
    features = layers.LayerNormalization()(features)
    return keras.Model(inputs, features, name="image_encoder")

def build_mimic_cxr_model(vocab_size):
    encoder = build_image_encoder()
    img_inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image_input")
    seq_inputs = layers.Input(shape=(MAX_REPORT_LENGTH - 1,), name="seq_input")

    img_features = encoder(img_inputs)
    pos_emb = PositionalEmbedding(vocab_size, MAX_REPORT_LENGTH - 1, EMBED_DIM)(seq_inputs)
    decoder_out = TransformerDecoderBlock(EMBED_DIM, FF_DIM, NUM_HEADS)(pos_emb, img_features)
    outputs = layers.Dense(vocab_size, activation="softmax", name="vocab_output")(decoder_out)

    return keras.Model(inputs=[img_inputs, seq_inputs], outputs=outputs, name="mimic_cxr_report_generator")

def create_tf_dataset(df, tokenizer, batch_size=BATCH_SIZE, shuffle=True):
    paths = df["image_path"].values
    reports = df["report"].values

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

def generate_sample_reports(model, val_df, vocab, num_samples=5):
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}
    start_id = token_to_id.get(START_TOKEN, 1)
    end_id = token_to_id.get(END_TOKEN, 2)

    lines = ["MIMIC-CXR VAL DATASET SAMPLE REPORT EVALUATION", "=" * 65]
    sample_val = val_df.head(num_samples)

    for idx, row in sample_val.iterrows():
        img_path = row["image_path"]
        actual_report = row["report"]

        try:
            img = preprocess_xray(img_path)
            img_input = np.expand_dims(img, axis=0)

            seq = [start_id]
            for _ in range(MAX_REPORT_LENGTH - 2):
                padded_seq = keras.preprocessing.sequence.pad_sequences(
                    [seq], maxlen=MAX_REPORT_LENGTH - 1, padding="post"
                )
                preds = model.predict([img_input, padded_seq], verbose=0)
                next_id = int(np.argmax(preds[0, len(seq) - 1, :]))
                if next_id == end_id or next_id == 0:
                    break
                seq.append(next_id)

            gen_words = [id_to_token.get(i, "") for i in seq[1:] if i not in (0, start_id, end_id)]
            gen_report = " ".join([w for w in gen_words if w])

            sub_id = row["subject_id"]
            st_id = row["study_id"]
            vw = row["view"]
            lines.append(f"Subject ID : {sub_id}")
            lines.append(f"Study ID   : {st_id}")
            lines.append(f"View       : {vw}")
            lines.append(f"Actual     : {actual_report}")
            lines.append(f"Generated  : {gen_report}")
            lines.append("-" * 65)
        except Exception as e:
            st_id = row["study_id"]
            lines.append(f"Error evaluating sample {st_id}: {e}")

    with open(EVAL_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved sample predictions evaluation to: {EVAL_SAVE_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Train MIMIC-CXR Medical Image Report Generation Model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--test-only", action="store_true", help="Perform data loading and single step test only")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint if available")
    args = parser.parse_args()

    print("=" * 70)
    print("MIMIC-CXR MEDICAL IMAGE REPORT GENERATION TRAINING PIPELINE")
    print("=" * 70)

    gpus = tf.config.list_physical_devices("GPU")
    gpu_available = len(gpus) > 0
    print(f"TensorFlow Version : {tf.__version__}")
    print("GPU Hardware Mode  : CPU Mode (Windows Native)")

    print("\n[1/6] Loading and verifying MIMIC-CXR datasets...")
    PROC_DIR = os.path.join(PROJECT_DIR, "processed_dataset_v1")
    train_csv = os.path.join(PROC_DIR, "train.csv")
    val_csv = os.path.join(PROC_DIR, "validation.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    tr_pts_n = train_df["subject_id"].nunique()
    tr_st_n = train_df["study_id"].nunique()
    val_pts_n = val_df["subject_id"].nunique()
    val_st_n = val_df["study_id"].nunique()

    print(f"  • Training Samples   : {len(train_df)} (Unique Patients: {tr_pts_n}, Studies: {tr_st_n})")
    print(f"  • Validation Samples : {len(val_df)} (Unique Patients: {val_pts_n}, Studies: {val_st_n})")

    tr_pts = set(train_df["subject_id"].unique())
    val_pts = set(val_df["subject_id"].unique())
    patient_overlap = len(tr_pts.intersection(val_pts))
    print(f"  • Patient Overlap    : {patient_overlap} (Zero overlap enforced)")

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Error: Train or validation dataset is empty after matching existing image paths!")

    print("\n[2/6] Building Text Tokenizer & Vocabulary...")
    tokenizer = layers.TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_REPORT_LENGTH,
        standardize=None
    )
    tokenizer.adapt(tf.data.Dataset.from_tensor_slices(train_df["report"].values.astype(str)).batch(64))
    vocab = tokenizer.get_vocabulary()
    vocab_size = len(vocab)
    print(f"  • Vocabulary Size : {vocab_size} tokens")

    with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"  • Saved Vocabulary to: {VOCAB_SAVE_PATH}")

    print("\n[3/6] Initializing EfficientNetB0 + Transformer Decoder Model...")
    start_epoch = 0
    if args.resume or os.path.exists(CHECKPOINT_SAVE_PATH):
        if os.path.exists(HISTORY_SAVE_PATH):
            try:
                hist_df = pd.read_csv(HISTORY_SAVE_PATH)
                if not hist_df.empty and "epoch" in hist_df.columns:
                    hist_df["epoch"] = pd.to_numeric(hist_df["epoch"], errors="coerce")
                    valid_epochs = hist_df["epoch"].dropna()
                    if not valid_epochs.empty:
                        start_epoch = int(valid_epochs.max()) + 1
            except Exception:
                pass

    resume_ckpt = None
    if args.resume:
        if os.path.exists(CHECKPOINT_SAVE_PATH):
            resume_ckpt = CHECKPOINT_SAVE_PATH
        elif os.path.exists(MODEL_SAVE_PATH):
            resume_ckpt = MODEL_SAVE_PATH

    if resume_ckpt:
        print(f"  • Resuming from existing checkpoint: {resume_ckpt} (Starting at epoch {start_epoch})")
        custom_objects = {
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerDecoderBlock": TransformerDecoderBlock
        }
        model = keras.models.load_model(resume_ckpt, custom_objects=custom_objects)
    else:
        if args.resume:
            print("  • --resume specified, but no existing checkpoint found. Initializing new model.")
        model = build_mimic_cxr_model(vocab_size)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    print("\n[4/6] Executing Data Loading & Model Sanity Test...")
    test_batch_size = 4
    test_df = train_df.head(test_batch_size)
    test_ds = create_tf_dataset(test_df, tokenizer, batch_size=test_batch_size, shuffle=False)

    for (test_img, test_seq), test_target in test_ds.take(1):
        print(f"  • Input Image Batch Shape  : {test_img.shape}")
        print(f"  • Input Sequence Shape     : {test_seq.shape}")
        print(f"  • Target Sequence Shape    : {test_target.shape}")

        test_preds = model.predict([test_img, test_seq], verbose=0)
        print(f"  • Prediction Output Shape  : {test_preds.shape}")

        test_loss, test_acc = model.evaluate([test_img, test_seq], test_target, verbose=0)
        print(f"  • Test Batch Loss          : {test_loss:.4f}")
        print(f"  • Test Batch Accuracy      : {test_acc * 100:.2f}%")
        print("  ✓ Sanity Test PASSED SUCCESSFULLY!")
        break

    if args.test_only:
        print("\nTest-only flag set. Skipping full training run.")
        return

    print("\n[5/6] Creating full TensorFlow training and validation pipelines...")
    train_ds = create_tf_dataset(train_df, tokenizer, batch_size=args.batch_size, shuffle=True)
    val_ds = create_tf_dataset(val_df, tokenizer, batch_size=args.batch_size, shuffle=False)

    callbacks = [
        # Save latest epoch model checkpoint for resuming
        keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_SAVE_PATH,
            save_best_only=False,
            verbose=1
        ),
        # Save best model based on validation loss
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        keras.callbacks.CSVLogger(HISTORY_SAVE_PATH, append=bool(args.resume and start_epoch > 0)),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        ),
        # Per-epoch checkpoint callback
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(EPOCH_DIR, "model_epoch_{epoch:02d}.keras"),
            save_best_only=False,
            verbose=1
        )
    ]

    if start_epoch >= args.epochs:
        print(f"\n[6/6] Target epochs ({args.epochs}) already completed (current progress: epoch {start_epoch}).")
        print(f"      To train further, increase total epochs (e.g. --epochs {start_epoch + 5}).")
    else:
        print(f"\n[6/6] Starting MIMIC-CXR Training Run (Epoch {start_epoch + 1} to {args.epochs}, Batch Size {args.batch_size})...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            initial_epoch=start_epoch,
            callbacks=callbacks
        )

        model.save(MODEL_SAVE_PATH)
        print(f"\n✓ Training completed! Final model saved to: {MODEL_SAVE_PATH}")
        print(f"✓ Training history saved to: {HISTORY_SAVE_PATH}")

    try:
        generate_sample_reports(model, val_df, vocab, num_samples=5)
    except Exception as e:
        print(f"Sample evaluation note: {e}")

    print("\n" + "=" * 70)
    print("MIMIC-CXR TRAINING WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()