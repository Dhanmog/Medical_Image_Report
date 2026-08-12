
import os
import re
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# 1. CONFIGURATION


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

MAX_VOCAB_SIZE = 12000
MAX_REPORT_LENGTH = 120

EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 512

AUTOTUNE = tf.data.AUTOTUNE

START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"



# 2. IMAGE PREPROCESSING


def preprocess_xray(image_path):
    """
    Load an X-ray, convert to grayscale, resize, apply CLAHE,
    normalize to [0, 1], and convert to 3 channels.

    Returns:
        np.ndarray of shape (224, 224, 3)
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # CLAHE
    # Contrast Limited Adaptive Histogram Equalization
    # --------------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------
    image = image.astype(np.float32) / 255.0

    # X-rays are grayscale, but most CNNs expect 3 channels.
    image = np.stack([image, image, image], axis=-1)

    return image


def preprocess_xray_tf(image_path):
    """
    TensorFlow-compatible wrapper.
    """

    image = tf.numpy_function(
        func=preprocess_xray,
        inp=[image_path],
        Tout=tf.float32
    )

    image.set_shape(
        [IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
    )

    return image


# ============================================================
# 3. REPORT CLEANING
# ============================================================

def clean_report(text):
    """
    Basic report cleaning.
    """

    text = str(text).lower()

    # Remove unnecessary characters
    text = re.sub(r"[^a-z0-9.,;:!?'\- ]+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Add sequence markers
    text = f"{START_TOKEN} {text} {END_TOKEN}"

    return text


# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_dataset(csv_file):
    df = pd.read_csv(csv_file)

    if "image_path" not in df.columns:
        raise ValueError(
            "CSV must contain an 'image_path' column."
        )

    if "report" not in df.columns:
        raise ValueError(
            "CSV must contain a 'report' column."
        )

    df["report"] = df["report"].apply(clean_report)

    return df


# Example:
#
# df = load_dataset("dataset.csv")
#
# print(df.head())


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

def split_dataset(df, validation_fraction=0.15):

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    split_index = int(
        len(df) * (1 - validation_fraction)
    )

    train_df = df.iloc[:split_index]
    val_df = df.iloc[split_index:]

    return train_df, val_df


# ============================================================
# 6. TEXT TOKENIZATION
# ============================================================

def create_tokenizer(train_reports):

    tokenizer = keras.layers.TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_REPORT_LENGTH,
        standardize=None
    )

    tokenizer.adapt(
        tf.data.Dataset.from_tensor_slices(
            train_reports
        ).batch(64)
    )

    return tokenizer


# ============================================================
# 7. DATASET PIPELINE
# ============================================================

def create_tf_dataset(
    dataframe,
    tokenizer,
    batch_size=BATCH_SIZE,
    shuffle=True
):

    image_paths = dataframe["image_path"].values
    reports = dataframe["report"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, reports)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            reshuffle_each_iteration=True
        )

    def process(image_path, report):

        # Image preprocessing
        image = preprocess_xray_tf(image_path)

        # Convert text into integer token IDs
        tokens = tokenizer(report)

        # Teacher forcing:
        #
        # Input:
        # <start> patient has ...
        #
        # Target:
        # patient has ... <end>
        #
        decoder_input = tokens[:, :-1]
        decoder_target = tokens[:, 1:]

        return (
            image,
            decoder_input,
            decoder_target
        )

    dataset = dataset.batch(batch_size)

    dataset = dataset.map(
        process,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# ============================================================
# 8. CNN IMAGE ENCODER
# ============================================================

def build_image_encoder():

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    )

    # Initially freeze CNN
    base_model.trainable = False

    inputs = layers.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    )

    # EfficientNet preprocessing is already incorporated
    # into the model for modern Keras versions.
    features = base_model(
        inputs,
        training=False
    )

    # Convert CNN feature map to sequence
    # Example:
    #
    # (batch, 7, 7, 1280)
    #
    # ->
    #
    # (batch, 49, 1280)

    features = layers.Reshape(
        (-1, features.shape[-1])
    )(features)

    features = layers.Dense(
        EMBED_DIM
    )(features)

    features = layers.LayerNormalization()(
        features
    )

    return keras.Model(
        inputs,
        features,
        name="image_encoder"
    )


# ============================================================
# 9. POSITIONAL EMBEDDING
# ============================================================

class PositionalEmbedding(layers.Layer):

    def __init__(
        self,
        vocabulary_size,
        sequence_length,
        embedding_dim
    ):
        super().__init__()

        self.token_embedding = layers.Embedding(
            vocabulary_size,
            embedding_dim,
            mask_zero=True
        )

        self.position_embedding = layers.Embedding(
            sequence_length,
            embedding_dim
        )

    def call(self, inputs):

        length = tf.shape(inputs)[-1]

        positions = tf.range(
            start=0,
            limit=length,
            delta=1
        )

        positions = self.position_embedding(
            positions
        )

        token_embeddings = self.token_embedding(
            inputs
        )

        return token_embeddings + positions


# ============================================================
# 10. TRANSFORMER DECODER BLOCK
# ============================================================

class TransformerDecoder(layers.Layer):

    def __init__(
        self,
        embed_dim,
        dense_dim,
        num_heads
    ):
        super().__init__()

        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = keras.Sequential([
            layers.Dense(
                dense_dim,
                activation="relu"
            ),
            layers.Dense(embed_dim)
        ])

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.1)
        self.dropout_3 = layers.Dropout(0.1)

    def call(
        self,
        inputs,
        encoder_outputs,
        training=False,
        mask=None
    ):

        # ----------------------------------------------------
        # Causal self-attention
        # ----------------------------------------------------

        attention_output = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            use_causal_mask=True,
            training=training
        )

        attention_output = self.dropout_1(
            attention_output,
            training=training
        )

        out_1 = self.layernorm_1(
            inputs + attention_output
        )

        # ----------------------------------------------------
        # Cross attention
        # Image features -> text decoder
        # ----------------------------------------------------

        cross_attention = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            training=training
        )

        cross_attention = self.dropout_2(
            cross_attention,
            training=training
        )

        out_2 = self.layernorm_2(
            out_1 + cross_attention
        )

        # ----------------------------------------------------
        # Feed-forward network
        # ----------------------------------------------------

        ffn_output = self.ffn(
            out_2,
            training=training
        )

        ffn_output = self.dropout_3(
            ffn_output,
            training=training
        )

        return self.layernorm_3(
            out_2 + ffn_output
        )


# ============================================================
# 11. COMPLETE IMAGE-TO-REPORT MODEL
# ============================================================

class XRayReportGenerator(keras.Model):

    def __init__(
        self,
        image_encoder,
        vocab_size,
        max_length
    ):

        super().__init__()

        self.image_encoder = image_encoder

        self.embedding = PositionalEmbedding(
            vocabulary_size=vocab_size,
            sequence_length=max_length,
            embedding_dim=EMBED_DIM
        )

        self.decoder = TransformerDecoder(
            embed_dim=EMBED_DIM,
            dense_dim=FF_DIM,
            num_heads=NUM_HEADS
        )

        self.output_layer = layers.Dense(
            vocab_size
        )

        self.loss_tracker = keras.metrics.Mean(
            name="loss"
        )

        self.accuracy_tracker = keras.metrics.Mean(
            name="accuracy"
        )

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.accuracy_tracker
        ]

    def compute_masked_loss(
        self,
        target,
        predictions
    ):

        loss = keras.losses.sparse_categorical_crossentropy(
            target,
            predictions,
            from_logits=True
        )

        mask = tf.cast(
            target != 0,
            loss.dtype
        )

        loss *= mask

        return tf.reduce_sum(loss) / (
            tf.reduce_sum(mask) + 1e-8
        )

    def compute_masked_accuracy(
        self,
        target,
        predictions
    ):

        predicted_tokens = tf.argmax(
            predictions,
            axis=-1
        )

        correct = tf.cast(
            tf.equal(
                target,
                predicted_tokens
            ),
            tf.float32
        )

        mask = tf.cast(
            target != 0,
            tf.float32
        )

        return tf.reduce_sum(
            correct * mask
        ) / (
            tf.reduce_sum(mask) + 1e-8
        )

    def train_step(self, data):

        images, decoder_input, decoder_target = data

        with tf.GradientTape() as tape:

            image_features = self.image_encoder(
                images,
                training=True
            )

            embedded_text = self.embedding(
                decoder_input
            )

            decoder_output = self.decoder(
                embedded_text,
                image_features,
                training=True
            )

            predictions = self.output_layer(
                decoder_output
            )

            loss = self.compute_masked_loss(
                decoder_target,
                predictions
            )

        gradients = tape.gradient(
            loss,
            self.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.trainable_variables
            )
        )

        accuracy = self.compute_masked_accuracy(
            decoder_target,
            predictions
        )

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()
        }

    def test_step(self, data):

        images, decoder_input, decoder_target = data

        image_features = self.image_encoder(
            images,
            training=False
        )

        embedded_text = self.embedding(
            decoder_input
        )

        decoder_output = self.decoder(
            embedded_text,
            image_features,
            training=False
        )

        predictions = self.output_layer(
            decoder_output
        )

        loss = self.compute_masked_loss(
            decoder_target,
            predictions
        )

        accuracy = self.compute_masked_accuracy(
            decoder_target,
            predictions
        )

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()
        }


# ============================================================
# 12. BUILD VOCABULARY
# ============================================================

def get_vocabulary(tokenizer):

    vocabulary = tokenizer.get_vocabulary()

    token_to_id = {
        token: index
        for index, token in enumerate(vocabulary)
    }

    id_to_token = {
        index: token
        for index, token in enumerate(vocabulary)
    }

    return vocabulary, token_to_id, id_to_token


# ============================================================
# 13. REPORT GENERATION
# ============================================================

def generate_report(
    model,
    image_path,
    tokenizer,
    max_length=MAX_REPORT_LENGTH
):

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image = preprocess_xray(
        image_path
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    # --------------------------------------------------------
    # Image encoder
    # --------------------------------------------------------

    image_features = model.image_encoder(
        image,
        training=False
    )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    vocabulary = tokenizer.get_vocabulary()

    token_to_id = {
        token: i
        for i, token in enumerate(vocabulary)
    }

    start_id = token_to_id.get(
        START_TOKEN
    )

    end_id = token_to_id.get(
        END_TOKEN
    )

    if start_id is None or end_id is None:
        raise ValueError(
            "Start/end tokens were not found."
        )

    # --------------------------------------------------------
    # Start sequence
    # --------------------------------------------------------

    generated_tokens = [
        start_id
    ]

    for _ in range(max_length - 1):

        decoder_input = tf.constant(
            [generated_tokens],
            dtype=tf.int32
        )

        embedded = model.embedding(
            decoder_input
        )

        decoder_output = model.decoder(
            embedded,
            image_features,
            training=False
        )

        predictions = model.output_layer(
            decoder_output
        )

        # Take prediction for last token
        next_token_logits = predictions[
            0, -1, :
        ]

        next_token = int(
            tf.argmax(
                next_token_logits
            ).numpy()
        )

        if next_token == end_id:
            break

        generated_tokens.append(
            next_token
        )

    # --------------------------------------------------------
    # Convert IDs -> words
    # --------------------------------------------------------

    words = []

    for token_id in generated_tokens:

        token = vocabulary[token_id]

        if token in [
            START_TOKEN,
            END_TOKEN,
            PAD_TOKEN
        ]:
            continue

        words.append(token)

    return " ".join(words)


# ============================================================
# 14. MAIN TRAINING FUNCTION
# ============================================================

def train_model(csv_file):

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset(
        csv_file
    )

    print(
        f"Total samples: {len(df)}"
    )

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    train_df, val_df = split_dataset(
        df
    )

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = create_tokenizer(
        train_df["report"].values
    )

    vocabulary = tokenizer.get_vocabulary()

    vocab_size = len(vocabulary)

    print(
        f"Vocabulary size: {vocab_size}"
    )

    # --------------------------------------------------------
    # TensorFlow datasets
    # --------------------------------------------------------

    train_dataset = create_tf_dataset(
        train_df,
        tokenizer,
        shuffle=True
    )

    val_dataset = create_tf_dataset(
        val_df,
        tokenizer,
        shuffle=False
    )

    # --------------------------------------------------------
    # CNN encoder
    # --------------------------------------------------------

    image_encoder = build_image_encoder()

    # --------------------------------------------------------
    # Complete model
    # --------------------------------------------------------

    model = XRayReportGenerator(
        image_encoder=image_encoder,
        vocab_size=vocab_size,
        max_length=MAX_REPORT_LENGTH
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = keras.optimizers.Adam(
        learning_rate=1e-4
    )

    model.compile(
        optimizer=optimizer
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        keras.callbacks.ModelCheckpoint(
            "xray_report_generator.keras",
            monitor="val_loss",
            save_best_only=True
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2
        )
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # Save tokenizer vocabulary
    # --------------------------------------------------------

    np.save(
        "vocabulary.npy",
        np.array(vocabulary)
    )

    return model, tokenizer, history


# ============================================================
# 15. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    CSV_FILE = "dataset.csv"

    model, tokenizer, history = train_model(
        CSV_FILE
    )

    # --------------------------------------------------------
    # Generate a report for a new X-ray
    # --------------------------------------------------------

    test_image = "test_xray.png"

    report = generate_report(
        model=model,
        image_path=test_image,
        tokenizer=tokenizer
    )

    print("\nGenerated Report:")
    print("----------------")
    print(report)