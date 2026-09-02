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
from tensorflow.keras import mixed_precision

from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.stdout.flush()

# ============================================================
# 1. OPTIMIZATIONS & GPU SETUP
# ============================================================

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print("Error enabling GPU memory growth:", e)

# Enable Mixed Precision for faster training on RTX GPU (GA107 Ampere architecture)
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision policy 'mixed_float16' enabled.")
except Exception as e:
    print("Could not enable mixed precision:", e)

# ============================================================
# 2. CONFIGURATION & DIRECTORY SETUP
# ============================================================

parser = argparse.ArgumentParser(
    description="[V3] Maximum Accuracy DenseNet121 + Transformer on MIMIC-CXR"
)
parser.add_argument("--phase1-epochs",   type=int,   default=10,   help="Phase 1 epochs (frozen backbone)")
parser.add_argument("--phase2-epochs",   type=int,   default=15,   help="Phase 2 epochs (fine-tune top N backbone layers)")
parser.add_argument("--phase3-epochs",   type=int,   default=8,    help="Phase 3 epochs (fine-tune all backbone layers)")
parser.add_argument("--batch-size",      type=int,   default=4,    help="Batch size (using 4 to fit 4GB VRAM)")
parser.add_argument("--lr",              type=float, default=1e-4, help="Phase 1 base learning rate")
parser.add_argument("--finetune-lr",     type=float, default=1e-5, help="Phase 2 fine-tune learning rate")
parser.add_argument("--polish-lr",       type=float, default=5e-7, help="Phase 3 fine-tune learning rate")
parser.add_argument("--num-samples",     type=int,   default=50000, help="Number of training samples")
parser.add_argument("--unfreeze-layers", type=int,   default=80,   help="Number of DenseNet layers to unfreeze in Phase 2")
parser.add_argument("--decoder-blocks",  type=int,   default=3,    help="Number of transformer decoder blocks")
parser.add_argument("--encoder-layers",  type=int,   default=2,    help="Number of spatial transformer encoder layers")
parser.add_argument("--encoder-heads",   type=int,   default=4,    help="Number of spatial transformer encoder heads")
parser.add_argument("--class-loss-weight", type=float, default=2.0,  help="Loss weight for 14-disease classification head")
parser.add_argument("--no-balance-data", dest="balance_data", action="store_false", default=True, help="Disable pathology balancing of normal/generic template reports")
parser.add_argument("--resume",          type=str,   default='auto', nargs='?', const='auto', help="Resume training from latest_checkpoint.keras ('auto') or a specific checkpoint path.")
parser.add_argument("--test-only",       action="store_true", help="Perform sanity check forward pass on a test batch and exit.")
args = parser.parse_args()

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'models', '02_densenet121_mimic_v4_highres')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH        = os.path.join(OUTPUT_DIR, 'model.keras')
PHASE1_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'model_phase1.keras')
PHASE2_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'model_phase2.keras')
VOCAB_SAVE_PATH        = os.path.join(OUTPUT_DIR, 'tokenizer_vocab.json')
HISTORY_SAVE_PATH      = os.path.join(OUTPUT_DIR, 'training_history.csv')
PHASE1_HISTORY_PATH    = os.path.join(OUTPUT_DIR, 'training_history_phase1.csv')
PHASE2_HISTORY_PATH    = os.path.join(OUTPUT_DIR, 'training_history_phase2.csv')
PHASE3_HISTORY_PATH    = os.path.join(OUTPUT_DIR, 'training_history_phase3.csv')

PROC_DIR       = os.path.join(PROJECT_DIR, 'datasets', 'mimic_cxr', 'processed', 'processed_dataset_v1')
PROC_TRAIN_CSV = os.path.join(PROC_DIR, 'train.csv')

IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = args.batch_size
NUM_SAMPLES  = args.num_samples

# Hyperparameters for maximum accuracy
MAX_VOCAB_SIZE     = 8000
MAX_REPORT_LENGTH  = 128
EMBED_DIM          = 512
NUM_HEADS          = 8
FF_DIM             = 2048
NUM_DECODER_BLOCKS = args.decoder_blocks
DROPOUT_RATE       = 0.1

START_TOKEN = '<start>'
END_TOKEN   = '<end>'
AUTOTUNE    = tf.data.AUTOTUNE

# (Dataset loading and prints moved to main())

# ============================================================
# 4. PREPROCESSING & AUGMENTATION
# ============================================================

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
    image_3ch = np.stack([image, image, image], axis=-1)
    return keras.applications.densenet.preprocess_input(image_3ch)

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

# (Tokenizer and cleaned report logic moved to main())

# ============================================================
# 6. TF DATASETS WITH DATA AUGMENTATION (Multi-Task)
# ============================================================

def extract_disease_labels(report):
    """
    Extracts 14-disease binary labels on-the-fly from reports.
    """
    report_clean = str(report).lower()
    labels = []
    # Atelectasis
    labels.append(1 if re.search(r"atelectasis", report_clean) else 0)
    # Cardiomegaly
    labels.append(1 if re.search(r"cardiomegaly|enlarged heart", report_clean) else 0)
    # Effusion
    labels.append(1 if re.search(r"effusion", report_clean) else 0)
    # Infiltration
    labels.append(1 if re.search(r"infiltration|infiltrate", report_clean) else 0)
    # Mass
    labels.append(1 if re.search(r"mass", report_clean) else 0)
    # Nodule
    labels.append(1 if re.search(r"nodule", report_clean) else 0)
    # Pneumonia
    labels.append(1 if re.search(r"pneumonia", report_clean) else 0)
    # Pneumothorax
    labels.append(1 if re.search(r"pneumothorax", report_clean) else 0)
    # Consolidation
    labels.append(1 if re.search(r"consolidation", report_clean) else 0)
    # Edema
    labels.append(1 if re.search(r"edema", report_clean) else 0)
    # Emphysema
    labels.append(1 if re.search(r"emphysema", report_clean) else 0)
    # Fibrosis
    labels.append(1 if re.search(r"fibrosis", report_clean) else 0)
    # Pleural Thickening
    labels.append(1 if re.search(r"pleural thickening", report_clean) else 0)
    # Hernia
    labels.append(1 if re.search(r"hernia", report_clean) else 0)
    return np.array(labels, dtype=np.float32)


def balance_dataset_split(df, labels, target_normal_ratio=0.35):
    """
    Downsamples records that contain no active pathologies in training labels.
    """
    label_sums = np.sum(labels, axis=1)
    
    abnormal_indices = np.where(label_sums > 0.0)[0]
    normal_indices = np.where(label_sums == 0.0)[0]
    
    num_abnormal = len(abnormal_indices)
    num_normal = len(normal_indices)
    
    if num_normal == 0:
        return df, labels
        
    target_normal = int(num_abnormal * target_normal_ratio / (1.0 - target_normal_ratio))
    target_normal = min(target_normal, num_normal)
    
    print(f"\n[INFO] Balancing Dataset:")
    print(f"  • Original Abnormal Samples: {num_abnormal:,}")
    print(f"  • Original Normal Samples  : {num_normal:,}")
    
    sampled_normal_indices = np.random.choice(normal_indices, size=target_normal, replace=False)
    balanced_indices = np.sort(np.concatenate([abnormal_indices, sampled_normal_indices]))
    
    balanced_df = df.iloc[balanced_indices].reset_index(drop=True)
    balanced_labels = labels[balanced_indices]
    
    print(f"  • Balanced Normal Samples  : {len(sampled_normal_indices):,} (Ratio: {target_normal_ratio:.1%})")
    print(f"  • Total Balanced Dataset   : {len(balanced_df):,}\n")
    
    return balanced_df, balanced_labels


def create_tf_dataset(df, labels, tokenizer, shuffle=True, augment=False):
    paths   = df["image_path"].values
    reports = df["cleaned_report"].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, reports, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    def process(image_path, report, label):
        image          = preprocess_xray_tf(image_path)
        tokens         = tokenizer(report)
        decoder_input  = tokens[:-1]
        decoder_target = tokens[1:]
        
        inputs = {"image_input": image, "seq_input": decoder_input}
        outputs = {"vocab_output": decoder_target, "class_output": label}
        return inputs, outputs

    def augment_fn(inputs, outputs):
        image = inputs["image_input"]
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        new_inputs = {"image_input": image, "seq_input": inputs["seq_input"]}
        return new_inputs, outputs

    dataset = dataset.map(process, num_parallel_calls=AUTOTUNE)
    if augment:
        dataset = dataset.map(augment_fn, num_parallel_calls=AUTOTUNE)

    dataset = (dataset
               .batch(BATCH_SIZE, drop_remainder=True)
               .prefetch(AUTOTUNE))
    return dataset

# ============================================================
# 7. WARMUP LR SCHEDULE
# ============================================================

@keras.utils.register_keras_serializable()
class WarmupLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
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
# 8. MODEL ARCHITECTURE
# ============================================================

@keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
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

    def compute_mask(self, inputs, mask=None):
        return keras.ops.not_equal(inputs, 0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"vocab_size": self.vocab_size, "seq_len": self.seq_len, "embed_dim": self.embed_dim})
        return cfg


@keras.utils.register_keras_serializable()
class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.embed_dim    = embed_dim
        self.dense_dim    = dense_dim
        self.num_heads    = num_heads
        self.dropout_rate = dropout_rate

        self.mha_causal = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dtype="float32")
        self.mha_cross  = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dtype="float32")
        self.ffn = keras.Sequential([
            layers.Dense(dense_dim, activation="gelu", dtype="float32"),
            layers.Dropout(dropout_rate, dtype="float32"),
            layers.Dense(embed_dim, dtype="float32")
        ])
        self.ln1      = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.ln2      = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.ln3      = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.dropout1 = layers.Dropout(dropout_rate, dtype="float32")
        self.dropout2 = layers.Dropout(dropout_rate, dtype="float32")
        self.dropout3 = layers.Dropout(dropout_rate, dtype="float32")

    def call(self, inputs, encoder_outputs, mask=None, training=False):
        inputs_f32 = tf.cast(inputs, tf.float32)
        encoder_outputs_f32 = tf.cast(encoder_outputs, tf.float32)
        
        attn1 = self.mha_causal(query=inputs_f32, value=inputs_f32, key=inputs_f32, attention_mask=mask, use_causal_mask=True)
        attn1 = self.dropout1(attn1, training=training)
        out1  = self.ln1(inputs_f32 + attn1)

        attn2 = self.mha_cross(query=out1, value=encoder_outputs_f32, key=encoder_outputs_f32)
        attn2 = self.dropout2(attn2, training=training)
        out2  = self.ln2(out1 + attn2)

        ffn_out = self.ffn(out2, training=training)
        ffn_out = self.dropout3(ffn_out, training=training)
        res = self.ln3(out2 + ffn_out)
        return tf.cast(res, self.compute_dtype)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim":    self.embed_dim,
            "dense_dim":    self.dense_dim,
            "num_heads":    self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


@keras.utils.register_keras_serializable(package="Custom")
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.dropout_rate = float(dropout_rate)
        
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dtype="float32")
        self.ffn = keras.Sequential([
            layers.Dense(dense_dim, activation="relu", dtype="float32"),
            layers.Dense(embed_dim, dtype="float32"),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.dropout1 = layers.Dropout(dropout_rate, dtype="float32")
        self.dropout2 = layers.Dropout(dropout_rate, dtype="float32")

    def call(self, inputs, training=False):
        inputs_f32 = tf.cast(inputs, tf.float32)
        attn = self.mha(query=inputs_f32, value=inputs_f32, key=inputs_f32)
        attn = self.dropout1(attn, training=training)
        out1 = self.ln1(inputs_f32 + attn)
        
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        res = self.ln2(out1 + ffn_out)
        return res

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


@keras.utils.register_keras_serializable(package="Custom")
class SpatialTransformerEncoder(layers.Layer):
    def __init__(self, num_layers=2, num_heads=4, embed_dim=512, dense_dim=1024, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dropout_rate = float(dropout_rate)
        
        self.enc_blocks = [
            TransformerEncoderBlock(embed_dim, dense_dim, num_heads, dropout_rate, name=f"enc_block_{i}")
            for i in range(num_layers)
        ]
        
    def build(self, input_shape):
        num_positions = input_shape[1]
        self.pos_emb = self.add_weight(
            name="pos_embedding",
            shape=(1, num_positions, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            dtype="float32"
        )
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32) + tf.cast(self.pos_emb, tf.float32)
        for block in self.enc_blocks:
            x = block(x, training=training)
        return tf.cast(x, self.compute_dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


def build_densenet_encoder(num_layers=2, num_heads=4):
    base_model = keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    base_model.trainable = False

    # Extract intermediate layer for higher spatial resolution (14x14x1024)
    intermediate_layer = base_model.get_layer("pool4_relu")
    base_sub_model = keras.Model(inputs=base_model.input, outputs=intermediate_layer.output)

    inputs   = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="encoder_image_input")
    features = base_sub_model(inputs, training=False)
    features = layers.Reshape((-1, features.shape[-1]))(features)
    features = layers.Dense(EMBED_DIM)(features)
    
    # Run through Spatial Transformer Encoder
    features = SpatialTransformerEncoder(
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=EMBED_DIM,
        dense_dim=FF_DIM,
        dropout_rate=DROPOUT_RATE,
        name="spatial_transformer_encoder"
    )(features)
    
    features = layers.LayerNormalization(epsilon=1e-6)(features)
    features = layers.Dropout(DROPOUT_RATE)(features)
    return keras.Model(inputs, features, name="densenet121_encoder")


def build_model_v3(vocab_size):
    encoder    = build_densenet_encoder(num_layers=args.encoder_layers, num_heads=args.encoder_heads)
    img_inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image_input")
    seq_inputs = layers.Input(shape=(MAX_REPORT_LENGTH - 1,), name="seq_input")

    img_features = encoder(img_inputs)

    x = PositionalEmbedding(vocab_size, MAX_REPORT_LENGTH - 1, EMBED_DIM, name="pos_embedding")(seq_inputs)
    for i in range(NUM_DECODER_BLOCKS):
        x = TransformerDecoderBlock(
            EMBED_DIM, FF_DIM, NUM_HEADS, DROPOUT_RATE,
            name=f"decoder_block_{i+1}"
        )(x, img_features)

    # Ensure float32 output layer for numerical stability with mixed precision
    vocab_outputs = layers.Dense(vocab_size, activation="softmax", dtype="float32", name="vocab_output")(x)

    # Branch 2: 14-disease classification head
    pooled_features = layers.GlobalAveragePooling1D(name="pooling_classification")(img_features)
    class_outputs = layers.Dense(14, activation="sigmoid", dtype="float32", name="class_output")(pooled_features)

    return keras.Model(
        inputs=[img_inputs, seq_inputs],
        outputs=[vocab_outputs, class_outputs],
        name="densenet121_mimic_v3"
    ), encoder

# ============================================================
# 9. RESUME / CHECKPOINT MANAGEMENT & HELPER FUNCTIONS
# ============================================================

class CheckpointAndMetadataCallback(keras.callbacks.Callback):
    def __init__(self, output_dir, phase_name, start_epoch_offset, total_phase_epochs, total_epochs, vocab=None, fixed_val_images=None, fixed_val_filenames=None, fixed_val_gts=None):
        super().__init__()
        self.output_dir = output_dir
        self.phase_name = phase_name
        self.start_epoch_offset = start_epoch_offset
        self.total_phase_epochs = total_phase_epochs
        self.total_epochs = total_epochs
        self.vocab = vocab
        self.fixed_val_images = fixed_val_images
        self.fixed_val_filenames = fixed_val_filenames
        self.fixed_val_gts = fixed_val_gts

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        overall_epoch = self.start_epoch_offset + epoch + 1
        phase_epoch = epoch + 1
        
        # Save metadata
        metadata = {
            "overall_epoch": overall_epoch,
            "phase": self.phase_name,
            "phase_epoch": phase_epoch,
            "train_loss": float(logs.get("loss", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "train_accuracy": float(logs.get("vocab_output_accuracy", logs.get("accuracy", 0.0))),
            "val_accuracy": float(logs.get("val_vocab_output_accuracy", logs.get("val_accuracy", 0.0))),
        }
        
        metadata_path = os.path.join(self.output_dir, "training_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        # Print epoch summary statistics
        print(f"\n[EPOCH SUMMARY] Epoch: {overall_epoch}/{self.total_epochs} | Phase: {self.phase_name} "
              f"| Loss: {metadata['train_loss']:.4f} | Val Loss: {metadata['val_loss']:.4f} "
              f"| Val Accuracy: {metadata['val_accuracy']:.4f}\n")
            
        # Save epoch-specific checkpoint (model weights + optimizer state, zero-padded as requested)
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{overall_epoch:02d}.keras")
        self.model.save(checkpoint_path)
        
        # Also save as latest_checkpoint.keras
        latest_path = os.path.join(self.output_dir, "latest_checkpoint.keras")
        self.model.save(latest_path)

        # Track best validation loss and save best model
        val_loss = logs.get("val_loss", float('inf'))
        if not hasattr(self, "best_val_loss"):
            self.best_val_loss = float('inf')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = os.path.join(self.output_dir, "best_model.keras")
            self.model.save(best_model_path)
            
        # Append to training_history.csv
        history_csv_path = os.path.join(self.output_dir, "training_history.csv")
        epoch_history = {
            "epoch": overall_epoch,
            "loss": float(logs.get("loss", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "vocab_output_loss": float(logs.get("vocab_output_loss", 0.0)),
            "val_vocab_output_loss": float(logs.get("val_vocab_output_loss", 0.0)),
            "class_output_loss": float(logs.get("class_output_loss", 0.0)),
            "val_class_output_loss": float(logs.get("val_class_output_loss", 0.0)),
            "vocab_output_accuracy": float(logs.get("vocab_output_accuracy", logs.get("accuracy", 0.0))),
            "val_vocab_output_accuracy": float(logs.get("val_vocab_output_accuracy", logs.get("val_accuracy", 0.0))),
            "class_output_auc": float(logs.get("class_output_auc", 0.0)),
            "val_class_output_auc": float(logs.get("val_class_output_auc", 0.0))
        }
        
        if os.path.exists(history_csv_path):
            try:
                hist_df = pd.read_csv(history_csv_path)
            except Exception:
                hist_df = pd.DataFrame()
        else:
            hist_df = pd.DataFrame()
            
        new_row_df = pd.DataFrame([epoch_history])
        hist_df = pd.concat([hist_df, new_row_df], ignore_index=True)
        hist_df = hist_df.drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)
        hist_df.to_csv(history_csv_path, index=False)

        # Generate reports for the same fixed 5 validation X-rays
        if self.vocab and self.fixed_val_images:
            try:
                from app import generate_report
                comparison_file_path = os.path.join(self.output_dir, "val_reports_comparison.txt")
                epoch_file_path = os.path.join(self.output_dir, f"epoch_{overall_epoch:02d}_reports.txt")
                
                epoch_lines = []
                epoch_lines.append("=" * 80)
                epoch_lines.append(f"VAL REPORTS FOR EPOCH {overall_epoch:02d} (Phase: {self.phase_name})")
                epoch_lines.append("=" * 80)
                
                for j in range(len(self.fixed_val_images)):
                    img = self.fixed_val_images[j]
                    filename = self.fixed_val_filenames[j]
                    gt = self.fixed_val_gts[j]
                    
                    # Generate report (using temperature=0.4 to avoid greedy end-token collapse)
                    generated = generate_report(self.model, self.vocab, img, temperature=0.4)
                    
                    epoch_lines.append(f"Image #{j+1} Path: {filename}")
                    epoch_lines.append(f"  • Ground Truth: {gt}")
                    epoch_lines.append(f"  • Generated   : {generated}")
                    epoch_lines.append("-" * 80)
                    
                epoch_text = "\n".join(epoch_lines) + "\n\n"
                
                # Save epoch specific file
                with open(epoch_file_path, "w", encoding="utf-8") as f_out:
                    f_out.write(epoch_text)
                    
                # Append to comparison file
                with open(comparison_file_path, "a", encoding="utf-8") as f_out:
                    f_out.write(epoch_text)
                    
                print(f"Generated validation reports saved to: {epoch_file_path}")
            except Exception as e_gen:
                print(f"[WARNING] Error generating validation reports: {e_gen}")
        
        # Clearly print information as requested
        print(f"Epoch {overall_epoch}/{self.total_epochs}")
        print(f"Training Loss: {logs.get('loss'):.4f}")
        print(f"Validation Loss: {logs.get('val_loss'):.4f}")
        val_acc = logs.get('val_vocab_output_accuracy', logs.get('val_accuracy', 0.0))
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
        print(f"Checkpoint Saved: {checkpoint_path}\n", flush=True)

def load_or_append_history(history_data, history_csv_path):
    if not history_data:
        if os.path.exists(history_csv_path):
            return pd.read_csv(history_csv_path)
        return pd.DataFrame()
    new_df = pd.DataFrame(history_data)
    if os.path.exists(history_csv_path):
        try:
            old_df = pd.read_csv(history_csv_path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined.to_csv(history_csv_path, index=False)
            return combined
        except Exception:
            pass
    new_df.to_csv(history_csv_path, index=False)
    return new_df


def patch_model_stability(model):
    import types
    import tensorflow as tf

    def custom_train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            vocab_pred, class_pred = y_pred
            
            # Check for NaN/Inf in model outputs (JIT/XLA-safe check)
            vocab_nan = tf.reduce_any(tf.math.is_nan(vocab_pred))
            vocab_inf = tf.reduce_any(tf.math.is_inf(vocab_pred))
            class_nan = tf.reduce_any(tf.math.is_nan(class_pred))
            class_inf = tf.reduce_any(tf.math.is_inf(class_pred))

            # Clip predictions to prevent float16 overflow/underflow from producing NaN in loss
            vocab_pred_clipped = tf.clip_by_value(vocab_pred, 1e-7, 1.0 - 1e-7)
            class_pred_clipped = tf.clip_by_value(class_pred, 1e-7, 1.0 - 1e-7)
            y_pred_clipped = (vocab_pred_clipped, class_pred_clipped)

            loss = self.compute_loss(x, y, y_pred_clipped, sample_weight=sample_weight)

            # Update the metric state for the loss tracker with the unscaled loss
            self._loss_tracker.update_state(loss)

            # Scale loss inside the tape block if mixed precision optimizer is active
            if hasattr(self.optimizer, "scale_loss"):
                loss = self.optimizer.scale_loss(loss)

        loss_nan = tf.math.is_nan(loss)
        loss_inf = tf.math.is_inf(loss)

        # Compute gradients (applying gradients dynamically handles unscaling inside Keras 3 apply_gradients)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        metrics = self.compute_metrics(x, y, y_pred_clipped, sample_weight=sample_weight)
        metrics["vocab_nan"] = tf.cast(vocab_nan, tf.float32)
        metrics["vocab_inf"] = tf.cast(vocab_inf, tf.float32)
        metrics["class_nan"] = tf.cast(class_nan, tf.float32)
        metrics["class_inf"] = tf.cast(class_inf, tf.float32)
        metrics["loss_nan"] = tf.cast(loss_nan, tf.float32)
        metrics["loss_inf"] = tf.cast(loss_inf, tf.float32)
        return metrics

    def custom_test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        y_pred = self(x, training=False)
        vocab_pred, class_pred = y_pred
        
        vocab_nan = tf.reduce_any(tf.math.is_nan(vocab_pred))
        vocab_inf = tf.reduce_any(tf.math.is_inf(vocab_pred))
        class_nan = tf.reduce_any(tf.math.is_nan(class_pred))
        class_inf = tf.reduce_any(tf.math.is_inf(class_pred))

        vocab_pred_clipped = tf.clip_by_value(vocab_pred, 1e-7, 1.0 - 1e-7)
        class_pred_clipped = tf.clip_by_value(class_pred, 1e-7, 1.0 - 1e-7)
        y_pred_clipped = (vocab_pred_clipped, class_pred_clipped)

        loss = self.compute_loss(x, y, y_pred_clipped, sample_weight=sample_weight)
        
        # Update the metric state for the loss tracker
        self._loss_tracker.update_state(loss)
        
        loss_nan = tf.math.is_nan(loss)
        loss_inf = tf.math.is_inf(loss)

        metrics = self.compute_metrics(x, y, y_pred_clipped, sample_weight=sample_weight)
        metrics["vocab_nan"] = tf.cast(vocab_nan, tf.float32)
        metrics["vocab_inf"] = tf.cast(vocab_inf, tf.float32)
        metrics["class_nan"] = tf.cast(class_nan, tf.float32)
        metrics["class_inf"] = tf.cast(class_inf, tf.float32)
        metrics["loss_nan"] = tf.cast(loss_nan, tf.float32)
        metrics["loss_inf"] = tf.cast(loss_inf, tf.float32)
        return metrics

    model.train_step = types.MethodType(custom_train_step, model)
    model.test_step = types.MethodType(custom_test_step, model)


class NaNAlertCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        import sys
        if logs.get("vocab_nan", 0.0) > 0.0:
            print("[WARNING] NaN detected in train model vocab outputs!", file=sys.stderr)
        if logs.get("vocab_inf", 0.0) > 0.0:
            print("[WARNING] Inf detected in train model vocab outputs!", file=sys.stderr)
        if logs.get("class_nan", 0.0) > 0.0:
            print("[WARNING] NaN detected in train model class outputs!", file=sys.stderr)
        if logs.get("class_inf", 0.0) > 0.0:
            print("[WARNING] Inf detected in train model class outputs!", file=sys.stderr)
        if logs.get("loss_nan", 0.0) > 0.0:
            print("[WARNING] NaN detected in train loss!", file=sys.stderr)
        if logs.get("loss_inf", 0.0) > 0.0:
            print("[WARNING] Inf detected in train loss!", file=sys.stderr)

    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}
        import sys
        if logs.get("vocab_nan", 0.0) > 0.0:
            print("[WARNING] NaN detected in val model vocab outputs!", file=sys.stderr)
        if logs.get("vocab_inf", 0.0) > 0.0:
            print("[WARNING] Inf detected in val model vocab outputs!", file=sys.stderr)
        if logs.get("class_nan", 0.0) > 0.0:
            print("[WARNING] NaN detected in val model class outputs!", file=sys.stderr)
        if logs.get("class_inf", 0.0) > 0.0:
            print("[WARNING] Inf detected in val model class outputs!", file=sys.stderr)
        if logs.get("loss_nan", 0.0) > 0.0:
            print("[WARNING] NaN detected in validation loss!", file=sys.stderr)
        if logs.get("loss_inf", 0.0) > 0.0:
            print("[WARNING] Inf detected in validation loss!", file=sys.stderr)

# ============================================================
# 10. MAIN TRAINING FLOW
# ============================================================

def main():
    import sys
    print("=" * 75)
    print("  [V3] MAXIMUM ACCURACY DENSENET121 + TRANSFORMER — MIMIC-CXR")
    print("=" * 75)
    print(f"  Output Dir        : {OUTPUT_DIR}")
    print(f"  Samples           : {NUM_SAMPLES}")
    print(f"  Phase 1 Epochs    : {args.phase1_epochs}  (frozen backbone, Warmup LR)")
    print(f"  Phase 2 Epochs    : {args.phase2_epochs}  (fine-tune top {args.unfreeze_layers} layers, LR={args.finetune_lr})")
    print(f"  Phase 3 Epochs    : {args.phase3_epochs}  (fine-tune all layers, LR={args.polish_lr})")
    print(f"  Batch Size        : {BATCH_SIZE}")
    print(f"  Embed Dim         : {EMBED_DIM}  |  Heads: {NUM_HEADS}  |  FF: {FF_DIM}")
    print(f"  Decoder Blocks    : {NUM_DECODER_BLOCKS}  |  Dropout: {DROPOUT_RATE}")
    print(f"  Max Report Len    : {MAX_REPORT_LENGTH}")
    print(f"  Balance Data      : {args.balance_data}")
    print("=" * 75, flush=True)

    # 10.1 Load & Split CSV
    if not os.path.exists(PROC_TRAIN_CSV):
        raise FileNotFoundError(f"Training CSV not found: {PROC_TRAIN_CSV}")

    raw_df = pd.read_csv(PROC_TRAIN_CSV)
    print(f"\nLoaded {len(raw_df)} records from training CSV.")

    df_subset = raw_df.iloc[:NUM_SAMPLES].reset_index(drop=True)
    print(f"Using {len(df_subset)} samples (image-loading errors will be reported dynamically during training).")

    split_idx = int(len(df_subset) * 0.9)
    train_df  = df_subset.iloc[:split_idx].reset_index(drop=True)
    val_df    = df_subset.iloc[split_idx:].reset_index(drop=True)
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}", flush=True)
    total_epochs = args.phase1_epochs + args.phase2_epochs + args.phase3_epochs

    # 10.2 Preprocessing Texts
    train_df["cleaned_report"] = train_df["report"].astype(str).apply(clean_report_text)
    val_df["cleaned_report"]   = val_df["report"].astype(str).apply(clean_report_text)

    # Prepare same fixed 5 validation X-rays for after-epoch reporting
    print("Preparing 5 fixed validation images for epoch report comparison...", flush=True)
    fixed_val_df = val_df.iloc[:5].copy()
    fixed_val_images = []
    fixed_val_filenames = []
    fixed_val_gts = []
    for idx, row in fixed_val_df.iterrows():
        p = row["image_path"]
        norm_p = normalize_and_resolve_path(p)
        fixed_val_filenames.append(p)
        fixed_val_gts.append(row["report"])
        
        # Load and preprocess
        preprocessed = preprocess_xray(norm_p)
        fixed_val_images.append(preprocessed)

    # 10.3 Tokenizer Build (Stable and Consistent Vocabulary)
    tokenizer = keras.layers.TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_REPORT_LENGTH,
        standardize=None
    )
    
    if os.path.exists(VOCAB_SAVE_PATH):
        print(f"\n[INFO] Loading pre-existing vocabulary from {VOCAB_SAVE_PATH}...")
        with open(VOCAB_SAVE_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        tokenizer.set_vocabulary(vocab)
        print(f"Loaded vocabulary size: {len(vocab)} tokens")
    else:
        print("\n[INFO] Vocabulary file not found. Building vocabulary from training dataset...")
        # Adapt on a large subset of raw_df to build a stable and comprehensive medical vocabulary
        adapt_samples = min(len(raw_df), 50000)
        print(f"Adapting tokenizer on {adapt_samples} samples for a stable vocabulary...")
        adapt_df = raw_df.iloc[:adapt_samples].copy()
        adapt_df["cleaned_report"] = adapt_df["report"].astype(str).apply(clean_report_text)
        tokenizer.adapt(tf.data.Dataset.from_tensor_slices(adapt_df["cleaned_report"].values).batch(128))
        vocab = tokenizer.get_vocabulary()
        print(f"Created vocabulary of size: {len(vocab)} tokens")
        
        with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2)
        print(f"Saved vocabulary -> {VOCAB_SAVE_PATH}")

    # 10.3.1 Pre-extract pathology labels for multi-task learning
    print("Extracting 14-disease labels for multi-task training...", flush=True)
    train_labels = np.stack(train_df["cleaned_report"].apply(extract_disease_labels).values)
    val_labels   = np.stack(val_df["cleaned_report"].apply(extract_disease_labels).values)

    # Apply dataset balancing if enabled to prevent mode collapse
    if args.balance_data:
        train_df, train_labels = balance_dataset_split(train_df, train_labels, target_normal_ratio=0.35)

    # 10.4 TF Datasets
    train_ds = create_tf_dataset(train_df, train_labels, tokenizer, shuffle=True, augment=True)
    val_ds   = create_tf_dataset(val_df, val_labels, tokenizer, shuffle=False, augment=False)

    # 10.5 Model Build or Checkpoint Loading
    model = None
    encoder_model = None
    start_phase = 1
    phase1_start_epoch = 0
    phase2_start_epoch = 0
    phase3_start_epoch = 0
    checkpoint_to_load = None

    if args.resume:
        if args.resume == 'auto':
            latest_path = os.path.join(OUTPUT_DIR, "latest_checkpoint.keras")
            metadata_path = os.path.join(OUTPUT_DIR, "training_metadata.json")
            if os.path.exists(latest_path) and os.path.exists(metadata_path):
                checkpoint_to_load = latest_path
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    overall_epoch = meta["overall_epoch"]
                    phase = meta["phase"]
                    phase_epoch = meta["phase_epoch"]
                    print(f"\n[INFO] Found auto-resume metadata: Overall Epoch finished: {overall_epoch}, Phase: {phase}, Phase Epoch: {phase_epoch}")
                    
                    if phase == "phase1":
                        start_phase = 1
                        phase1_start_epoch = phase_epoch
                    elif phase == "phase2":
                        start_phase = 2
                        phase2_start_epoch = phase_epoch
                    elif phase == "phase3":
                        start_phase = 3
                        phase3_start_epoch = phase_epoch
                except Exception as e:
                    print(f"[WARNING] Error reading resume metadata: {e}. Starting from scratch.")
                    checkpoint_to_load = None
        else:
            if os.path.exists(args.resume):
                checkpoint_to_load = args.resume
                base = os.path.basename(args.resume)
                if "model_phase2.keras" in base:
                    start_phase = 3
                    phase3_start_epoch = 0
                    print("\n[INFO] Explicitly resuming from model_phase2.keras. Phase 3 starts directly from epoch 0.")
                elif "model_phase1.keras" in base or "checkpoint_epoch_06" in base:
                    start_phase = 2
                    phase2_start_epoch = 0
                    print(f"\n[INFO] Explicitly resuming from Phase 1 output: {base}. Phase 2 starts directly from epoch 0.")
                else:
                    match = re.search(r"checkpoint_epoch_(\d+)", base)
                    if match:
                        overall_epoch = int(match.group(1))
                        print(f"\n[INFO] Parsing checkpoint path: overall epoch finished is {overall_epoch}")
                        p1_limit = args.phase1_epochs
                        p2_limit = args.phase1_epochs + args.phase2_epochs
                        
                        if overall_epoch < p1_limit:
                            start_phase = 1
                            phase1_start_epoch = overall_epoch
                        elif overall_epoch < p2_limit:
                            start_phase = 2
                            phase2_start_epoch = overall_epoch - p1_limit
                        else:
                            start_phase = 3
                            phase3_start_epoch = overall_epoch - p2_limit
                    else:
                        metadata_path = os.path.join(OUTPUT_DIR, "training_metadata.json")
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, "r", encoding="utf-8") as f:
                                    meta = json.load(f)
                                overall_epoch = meta["overall_epoch"]
                                phase = meta["phase"]
                                phase_epoch = meta["phase_epoch"]
                                
                                if phase == "phase1":
                                    start_phase = 1
                                    phase1_start_epoch = phase_epoch
                                elif phase == "phase2":
                                    start_phase = 2
                                    phase2_start_epoch = phase_epoch
                                elif phase == "phase3":
                                    start_phase = 3
                                    phase3_start_epoch = phase_epoch
                            except Exception:
                                pass
            else:
                print(f"[ERROR] Specified checkpoint not found: {args.resume}")
                sys.exit(1)

    if checkpoint_to_load:
        print(f"Resuming model training from checkpoint: {checkpoint_to_load}")
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
        try:
            model = keras.models.load_model(checkpoint_to_load, custom_objects=custom_objects, compile=True)
            encoder_model = model.get_layer("densenet121_encoder")
            
            # Verify spatial resolution compatibility
            spatial_encoder = encoder_model.get_layer("spatial_transformer_encoder")
            loaded_positions = spatial_encoder.pos_emb.shape[1]
            if loaded_positions != 196:
                raise ValueError(
                    f"Loaded checkpoint is incompatible! It has {loaded_positions} spatial positions, "
                    f"but the new 14x14 architecture requires exactly 196 positions. Please make sure "
                    f"you are not loading an old 7x7 checkpoint."
                )
            
            print("Successfully loaded model and optimizer state from checkpoint!")
            patch_model_stability(model)

            # For Phase 3 stability, keep BatchNormalization layers frozen and restore stable moving stats from Epoch 12
            if start_phase == 3:
                # 1. Proactively freeze all BatchNormalization layers in densenet_base
                densenet_base = None
                for layer in encoder_model.layers:
                    if (isinstance(layer, keras.Model) or hasattr(layer, 'layers')) and len(getattr(layer, 'layers', [])) > 100:
                        densenet_base = layer
                        break
                if densenet_base is not None:
                    frozen_bn = 0
                    for layer in densenet_base.layers:
                        if isinstance(layer, keras.layers.BatchNormalization) or layer.__class__.__name__ == "BatchNormalization":
                            layer.trainable = False
                            frozen_bn += 1
                    print(f"[INFO] Freezing {frozen_bn} BatchNormalization layers inside the base model for Phase 3.")

                # 2. Restore moving statistics from checkpoint_epoch_12.keras
                epoch12_path = os.path.join(OUTPUT_DIR, "checkpoint_epoch_12.keras")
                if os.path.exists(epoch12_path):
                    print(f"[INFO] Restoring BatchNormalization running statistics from Epoch 12 checkpoint ({epoch12_path})...")
                    try:
                        model_12 = keras.models.load_model(epoch12_path, custom_objects=custom_objects, compile=False)
                        restored_count = 0
                        
                        def restore_bn_recursive(layer_curr, layer_base):
                            nonlocal restored_count
                            if hasattr(layer_curr, 'layers') and hasattr(layer_base, 'layers'):
                                for subl_curr in layer_curr.layers:
                                    try:
                                        subl_base = layer_base.get_layer(subl_curr.name)
                                        restore_bn_recursive(subl_curr, subl_base)
                                    except ValueError:
                                        pass
                            elif isinstance(layer_curr, keras.layers.BatchNormalization) or layer_curr.__class__.__name__ == "BatchNormalization":
                                w_base = layer_base.get_weights()
                                w_curr = layer_curr.get_weights()
                                if len(w_base) == 4 and len(w_curr) == 4:
                                    new_w = [w_curr[0], w_curr[1], w_base[2], w_base[3]]
                                    layer_curr.set_weights(new_w)
                                    restored_count += 1

                        restore_bn_recursive(model, model_12)
                        print(f"[INFO] Successfully restored running statistics for {restored_count} BN layers.")
                    except Exception as bn_err:
                        print(f"[WARNING] Failed to restore BN running statistics: {bn_err}")
                else:
                    print("[WARNING] Epoch 12 checkpoint not found; BatchNormalization statistics cannot be automatically restored.")
        except Exception as e:
            print(f"Error loading model from checkpoint: {e}")
            sys.exit(1)
    else:
        print("Initializing new model from scratch...")
        model, encoder_model = build_model_v3(len(vocab))
        patch_model_stability(model)
        
        # Running training starting from default ImageNet weights for DenseNet121 as requested
        print("[INFO] Initializing DenseNet121 base using standard ImageNet pre-trained weights.")

    # 10.5.1 Sanity Check / Forward Pass Test
    # Run sanity check forward pass test unconditionally to verify model loading
    print("\n" + "=" * 75)
    print("  SANITY CHECK FORWARD PASS TEST")
    print("=" * 75)
    
    # Compile if not already compiled
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "vocab_output": keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
            "class_output": keras.losses.BinaryCrossentropy()
        },
        loss_weights={
            "vocab_output": 1.0,
            "class_output": args.class_loss_weight
        },
        metrics={
            "vocab_output": ["accuracy"],
            "class_output": ["auc"]
        }
    )
    
    print("Taking 1 batch from train dataset...")
    for inputs, targets in train_ds.take(1):
        images = inputs["image_input"]
        seqs = inputs["seq_input"]
        targets_vocab = targets["vocab_output"]
        targets_class = targets["class_output"]
        
        print(f"  • Input Images Shape : {images.shape}")
        print(f"  • Input Sequence Shape: {seqs.shape}")
        print(f"  • Target Vocab Shape  : {targets_vocab.shape}")
        print(f"  • Target Class Shape  : {targets_class.shape}")
        
        print("Running forward pass...")
        preds_vocab, preds_class = model([images, seqs], training=False)
        print(f"  • Preds Vocab Shape   : {preds_vocab.shape}")
        print(f"  • Preds Class Shape   : {preds_class.shape}")
        
        eval_metrics = model.evaluate([images, seqs], targets, verbose=0)
        metrics_dict = dict(zip(model.metrics_names, eval_metrics))
        print(f"  • Sanity Batch Loss   : {eval_metrics[0]:.4f}")
        print(f"  • Vocab Output Loss   : {eval_metrics[1]:.4f} | Accuracy: {eval_metrics[3]*100:.2f}%")
        print(f"  • Class Output Loss   : {eval_metrics[2]:.4f} | AUC: {eval_metrics[4]:.4f}")
        
        # Print alerts if NaN/Inf metrics are detected in sanity check
        import sys
        if metrics_dict.get("vocab_nan", 0.0) > 0.0:
            print("[WARNING] NaN detected in sanity check vocab outputs!", file=sys.stderr)
        if metrics_dict.get("vocab_inf", 0.0) > 0.0:
            print("[WARNING] Inf detected in sanity check vocab outputs!", file=sys.stderr)
        if metrics_dict.get("class_nan", 0.0) > 0.0:
            print("[WARNING] NaN detected in sanity check class outputs!", file=sys.stderr)
        if metrics_dict.get("class_inf", 0.0) > 0.0:
            print("[WARNING] Inf detected in sanity check class outputs!", file=sys.stderr)
        if metrics_dict.get("loss_nan", 0.0) > 0.0:
            print("[WARNING] NaN detected in sanity check loss!", file=sys.stderr)
        if metrics_dict.get("loss_inf", 0.0) > 0.0:
            print("[WARNING] Inf detected in sanity check loss!", file=sys.stderr)

        # Run one training batch step to verify custom_train_step and LossScaleOptimizer
        print("Running one training step to verify custom_train_step & optimizer compatibility...")
        train_metrics = model.train_on_batch([images, seqs], targets)
        print(f"  • Training Step Loss  : {train_metrics[0]:.4f}")
        print(f"  • Training Step Vocab Loss: {train_metrics[1]:.4f}")
        print(f"  • Training Step Class Loss: {train_metrics[2]:.4f}")

        print("\n[OK] Forward pass and training step completed, optimizer compatibility verified!")
        break
        
    if args.test_only:
        print("\nExiting since --test-only was specified.")
        sys.exit(0)

    # ============================================================
    # 10.6 PHASE 1 — FROZEN BACKBONE TRAINING
    # ============================================================
    history_p1 = None

    if start_phase == 1:
        print("\n" + "=" * 75)
        print("  PHASE 1: Training with frozen DenseNet121 backbone")
        print("=" * 75, flush=True)

        if phase1_start_epoch == 0:
            model.summary()
            lr_schedule_p1 = WarmupLRSchedule(d_model=EMBED_DIM, warmup_steps=2000)
            model.compile(
                optimizer=keras.optimizers.Adam(lr_schedule_p1, clipnorm=1.0),
                loss={
                    "vocab_output": keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
                    "class_output": keras.losses.BinaryCrossentropy()
                },
                loss_weights={
                    "vocab_output": 1.0,
                    "class_output": args.class_loss_weight
                },
                metrics={
                    "vocab_output": ["accuracy"],
                    "class_output": ["auc"]
                }
            )

        cb_meta_p1 = CheckpointAndMetadataCallback(
            output_dir=OUTPUT_DIR,
            phase_name="phase1",
            start_epoch_offset=0,
            total_phase_epochs=args.phase1_epochs,
            total_epochs=total_epochs,
            vocab=vocab,
            fixed_val_images=fixed_val_images,
            fixed_val_filenames=fixed_val_filenames,
            fixed_val_gts=fixed_val_gts
        )
        callbacks_p1 = [
            keras.callbacks.ModelCheckpoint(
                PHASE1_MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
            ),
            cb_meta_p1,
            NaNAlertCallback()
        ]

        history_p1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.phase1_epochs,
            initial_epoch=phase1_start_epoch,
            verbose=1,
            callbacks=callbacks_p1
        )

        hist_p1_df = load_or_append_history(history_p1.history, PHASE1_HISTORY_PATH)
        best_p1_acc = max(hist_p1_df.get('val_vocab_output_accuracy', hist_p1_df.get('val_accuracy', [0])))
        print(f"\nPhase 1 complete. Best val_accuracy: {best_p1_acc:.4f}")
        print(f"Phase 1 model -> {PHASE1_MODEL_SAVE_PATH}")
    else:
        print("\n[INFO] Skipping Phase 1 (already completed or resuming in later phase).")

    # ============================================================
    # 10.7 PHASE 2 — FINE-TUNE (unfreeze top N DenseNet layers)
    # ============================================================
    history_p2 = None

    if start_phase <= 2:
        print("\n" + "=" * 75)
        print(f"  PHASE 2: Fine-tuning top {args.unfreeze_layers} DenseNet121 layers")
        print("=" * 75, flush=True)

        # Dynamically find the nested DenseNet base model inside encoder_model
        densenet_base = None
        for layer in encoder_model.layers:
            if (isinstance(layer, keras.Model) or hasattr(layer, 'layers')) and len(getattr(layer, 'layers', [])) > 100:
                densenet_base = layer
                break
        
        if densenet_base is None:
            raise ValueError("Could not find the nested DenseNet121 base layer inside the encoder model!")
            
        densenet_base.trainable = True
        total_layers  = len(densenet_base.layers)
        freeze_until  = total_layers - args.unfreeze_layers
        for i, layer in enumerate(densenet_base.layers):
            layer.trainable = (i >= freeze_until)

        trainable_count = sum(1 for l in densenet_base.layers if l.trainable)
        print(f"DenseNet layers unfrozen: {trainable_count} / {total_layers}")

        if start_phase < 2 or phase2_start_epoch == 0:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=args.finetune_lr, clipnorm=1.0),
                loss={
                    "vocab_output": keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
                    "class_output": keras.losses.BinaryCrossentropy()
                },
                loss_weights={
                    "vocab_output": 1.0,
                    "class_output": args.class_loss_weight
                },
                metrics={
                    "vocab_output": ["accuracy"],
                    "class_output": ["auc"]
                }
            )

        cb_meta_p2 = CheckpointAndMetadataCallback(
            output_dir=OUTPUT_DIR,
            phase_name="phase2",
            start_epoch_offset=args.phase1_epochs,
            total_phase_epochs=args.phase2_epochs,
            total_epochs=total_epochs,
            vocab=vocab,
            fixed_val_images=fixed_val_images,
            fixed_val_filenames=fixed_val_filenames,
            fixed_val_gts=fixed_val_gts
        )
        callbacks_p2 = [
            keras.callbacks.ModelCheckpoint(
                PHASE2_MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
            ),
            cb_meta_p2,
            NaNAlertCallback()
        ]

        history_p2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.phase2_epochs,
            initial_epoch=phase2_start_epoch,
            verbose=1,
            callbacks=callbacks_p2
        )

        hist_p2_df = load_or_append_history(history_p2.history, PHASE2_HISTORY_PATH)
        best_p2_acc = max(hist_p2_df.get('val_vocab_output_accuracy', hist_p2_df.get('val_accuracy', [0])))
        print(f"\nPhase 2 complete. Best val_accuracy: {best_p2_acc:.4f}")
        print(f"Phase 2 model -> {PHASE2_MODEL_SAVE_PATH}")
    else:
        print("\n[INFO] Skipping Phase 2 (already completed or resuming in later phase).")

    # ============================================================
    # 10.8 PHASE 3 — FULL BACKBONE POLISH
    # ============================================================
    history_p3 = None

    if start_phase <= 3:
        print("\n" + "=" * 75)
        print("  PHASE 3: Polishing with all DenseNet121 layers unfrozen")
        print("=" * 75, flush=True)

        # Dynamically find the nested DenseNet base model inside encoder_model
        densenet_base = None
        for layer in encoder_model.layers:
            if (isinstance(layer, keras.Model) or hasattr(layer, 'layers')) and len(getattr(layer, 'layers', [])) > 100:
                densenet_base = layer
                break
        
        if densenet_base is None:
            raise ValueError("Could not find the nested DenseNet121 base layer inside the encoder model!")
            
        densenet_base.trainable = True
        total_layers  = len(densenet_base.layers)
        unfrozen_count = 0
        frozen_bn_count = 0
        for layer in densenet_base.layers:
            if isinstance(layer, keras.layers.BatchNormalization) or layer.__class__.__name__ == "BatchNormalization":
                layer.trainable = False
                frozen_bn_count += 1
            else:
                layer.trainable = True
                unfrozen_count += 1

        print(f"DenseNet layers: unfroze {unfrozen_count} layers, kept {frozen_bn_count} BatchNormalization layers frozen.")

        if start_phase < 3 or phase3_start_epoch == 0:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=args.polish_lr, clipnorm=1.0),
                loss={
                    "vocab_output": keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
                    "class_output": keras.losses.BinaryCrossentropy()
                },
                loss_weights={
                    "vocab_output": 1.0,
                    "class_output": args.class_loss_weight
                },
                metrics={
                    "vocab_output": ["accuracy"],
                    "class_output": ["auc"]
                }
            )

        cb_meta_p3 = CheckpointAndMetadataCallback(
            output_dir=OUTPUT_DIR,
            phase_name="phase3",
            start_epoch_offset=args.phase1_epochs + args.phase2_epochs,
            total_phase_epochs=args.phase3_epochs,
            total_epochs=total_epochs,
            vocab=vocab,
            fixed_val_images=fixed_val_images,
            fixed_val_filenames=fixed_val_filenames,
            fixed_val_gts=fixed_val_gts
        )
        callbacks_p3 = [
            keras.callbacks.ModelCheckpoint(
                MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
            ),
            cb_meta_p3,
            NaNAlertCallback()
        ]

        history_p3 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.phase3_epochs,
            initial_epoch=phase3_start_epoch,
            verbose=1,
            callbacks=callbacks_p3
        )

        hist_p3_df = load_or_append_history(history_p3.history, PHASE3_HISTORY_PATH)
        best_p3_acc = max(hist_p3_df.get('val_accuracy', [0]))
        print(f"\nPhase 3 complete. Best val_accuracy: {best_p3_acc:.4f}")
    else:
        print("\n[INFO] Skipping Phase 3 (already completed).")

    # ============================================================
    # 10.9 SAVE ALL COMBINED RESULTS
    # ============================================================
    hist_p1_df = load_or_append_history(None, PHASE1_HISTORY_PATH)
    hist_p2_df = load_or_append_history(None, PHASE2_HISTORY_PATH)
    hist_p3_df = load_or_append_history(None, PHASE3_HISTORY_PATH)

    combined_list = []
    if not hist_p1_df.empty:
        combined_list.append(hist_p1_df)
    if not hist_p2_df.empty:
        combined_list.append(hist_p2_df)
    if not hist_p3_df.empty:
        combined_list.append(hist_p3_df)

    if combined_list:
        combined_df = pd.concat(combined_list, ignore_index=True)
        combined_df.to_csv(HISTORY_SAVE_PATH, index=False)

    model.save(MODEL_SAVE_PATH)

    bp1 = max(hist_p1_df.get('val_accuracy', [0])) if not hist_p1_df.empty else 0.0
    bp2 = max(hist_p2_df.get('val_accuracy', [0])) if not hist_p2_df.empty else 0.0
    bp3 = max(hist_p3_df.get('val_accuracy', [0])) if not hist_p3_df.empty else 0.0

    print("\n" + "=" * 75)
    print("  TRAINING COMPLETE — V3 RESULTS SUMMARY")
    print("=" * 75)
    print(f"  Phase 1 best val_accuracy : {bp1:.4f}")
    print(f"  Phase 2 best val_accuracy : {bp2:.4f}")
    print(f"  Phase 3 best val_accuracy : {bp3:.4f}")
    print(f"\n  All outputs in: {OUTPUT_DIR}")
    print(f"    model.keras                  <- final fine-tuned model (best)")
    print(f"    latest_checkpoint.keras      <- latest training state")
    print(f"    training_history.csv         <- combined history")
    print(f"    training_history_phase1.csv  <- phase 1 only")
    print(f"    training_history_phase2.csv  <- phase 2 only")
    print(f"    training_history_phase3.csv  <- phase 3 only")
    print("=" * 75)
    print("\nTo replace the existing model with this improved one, copy:")
    print(f"  {OUTPUT_DIR}/model.keras")
    print(f"  {OUTPUT_DIR}/tokenizer_vocab.json")
    print("To:")
    print(f"  {os.path.join(PROJECT_DIR, 'models', '02_densenet121_mimic')}/")
    print("=" * 75)

if __name__ == "__main__":
    main()

