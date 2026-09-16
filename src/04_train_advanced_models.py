"""
04_train_advanced_models.py

Two additional architectures beyond the baseline CNN from Step 7:

  1. CSN (Capsule Network) -- uses "capsules" (small groups of numbers,
     i.e. vectors) instead of single numbers to represent features. This
     preserves more detail about HOW a pattern appears, not just whether
     it's present, using a process called "dynamic routing" instead of
     simple pooling.

  2. Attention-CNN (CBAM-style) -- adds a lightweight attention module to
     the baseline CNN that learns WHICH channels and WHICH spatial
     regions of the spectrogram matter most, and re-weights the features
     accordingly before passing them to the next layer.

Both reuse the same train/val/test .npz features from Step 3-4, and the
same song-level evaluation approach as the baseline CNN script (03).

Set MODEL_TYPE below to choose which one to train and run this script.
CSN is noticeably slower to train than a plain CNN (routing is
computationally heavier) -- a GPU (e.g. Google Colab) is strongly
recommended for it.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

# ---- CONFIG ----
FEATURES_DIR = "features"
MODEL_TYPE = "csn"   # "csn" or "attention_cnn"
MODEL_OUTPUT_PATH = f"models/{MODEL_TYPE}.keras"
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
NUM_CLASSES = len(GENRES)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3


# ---------------------------------------------------------------------
# Shared data utilities (same as 03_train_model.py)
# ---------------------------------------------------------------------

def load_split(name):
    data = np.load(os.path.join(FEATURES_DIR, f"{name}.npz"))
    return data["X"], data["y"], data["groups"]


def normalize(X, mean, std):
    return (X - mean) / (std + 1e-8)


def aggregate_to_song_level(y_true_segments, y_pred_probs, groups):
    song_true, song_probs_sum, song_probs_count = {}, {}, {}
    for true_label, probs, song_id in zip(y_true_segments, y_pred_probs, groups):
        if song_id not in song_probs_sum:
            song_probs_sum[song_id] = np.zeros_like(probs)
            song_probs_count[song_id] = 0
            song_true[song_id] = true_label
        song_probs_sum[song_id] += probs
        song_probs_count[song_id] += 1
    song_ids = list(song_true.keys())
    y_true_songs = np.array([song_true[s] for s in song_ids])
    y_pred_songs = np.array([np.argmax(song_probs_sum[s] / song_probs_count[s]) for s in song_ids])
    return y_true_songs, y_pred_songs


# ---------------------------------------------------------------------
# MODEL 1: CSN (Capsule Network)
# ---------------------------------------------------------------------

def squash(vectors, axis=-1):
    """
    Squashes a capsule's vector length into [0, 1) without changing its
    direction -- short vectors shrink toward 0, long vectors get pushed
    toward 1. This is a capsule network's equivalent of ReLU.
    """
    squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + tf.keras.backend.epsilon())
    return scale * vectors


class PrimaryCaps(layers.Layer):
    """Turns ordinary CNN feature maps into the first layer of capsules."""
    def __init__(self, num_channels, dim_capsules, kernel_size, strides, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            num_channels * dim_capsules, kernel_size, strides=strides, padding="valid"
        )
        self.dim_capsules = dim_capsules

    def call(self, inputs):
        x = self.conv(inputs)
        batch_size = tf.shape(x)[0]
        num_capsules = (x.shape[1] * x.shape[2] * x.shape[3]) // self.dim_capsules
        x = tf.reshape(x, [batch_size, num_capsules, self.dim_capsules])
        return squash(x)


class DigitCaps(layers.Layer):
    """
    The class-level capsules -- one capsule per genre. Uses "dynamic
    routing by agreement": each lower-level capsule votes for which
    class capsule it thinks matches, and capsules that agree with each
    other get their connection strengthened over a few routing rounds.
    """
    def __init__(self, num_classes, dim_capsules, routing_iters=3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.dim_capsules = dim_capsules
        self.routing_iters = routing_iters

    def build(self, input_shape):
        num_input_caps = input_shape[1]
        input_dim_caps = input_shape[2]
        self.W = self.add_weight(
            shape=[1, num_input_caps, self.num_classes, self.dim_capsules, input_dim_caps],
            initializer="glorot_uniform", trainable=True, name="routing_weights"
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_input_caps = inputs.shape[1]

        inputs_expand = inputs[:, :, tf.newaxis, :, tf.newaxis]              # (B, Nin, 1, Din, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_classes, 1, 1])  # (B, Nin, Nclass, Din, 1)
        # NOTE: self.W has shape (1, Nin, Nclass, Dout, Din) -- we let matmul
        # BROADCAST the leading "1" against the batch dimension instead of
        # manually tf.tile-ing a full copy per batch item. Tiling explicitly
        # would allocate a huge extra copy of W in memory for every single
        # sample -- broadcasting gets the same math for a fraction of the RAM,
        # which matters a lot when training on CPU.

        # "votes": what each input capsule predicts each class capsule should look like
        u_hat = tf.squeeze(tf.matmul(self.W, inputs_tiled), axis=-1)          # (B, Nin, Nclass, Dout)

        b = tf.zeros([batch_size, num_input_caps, self.num_classes])  # routing logits, start neutral

        for i in range(self.routing_iters):
            c = tf.nn.softmax(b, axis=2)                            # routing weights (agreement so far)
            s = tf.reduce_sum(c[..., tf.newaxis] * u_hat, axis=1)   # weighted sum of votes per class
            v = squash(s)                                            # final class capsule vectors
            if i < self.routing_iters - 1:
                agreement = tf.reduce_sum(u_hat * v[:, tf.newaxis, :, :], axis=-1)
                b += agreement                                       # capsules that agree get boosted

        return v  # (batch, num_classes, dim_capsules)


def capsule_length(v):
    """Turns each class capsule vector into a single probability-like number (its length)."""
    return tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1) + tf.keras.backend.epsilon())


def margin_loss(y_true, y_pred_lengths, m_plus=0.9, m_minus=0.1, lam=0.5):
    """
    CapsNet's standard loss. For the correct class it wants the capsule
    length near 1 (at least m_plus); for wrong classes it wants the
    length near 0 (below m_minus). `lam` softens the penalty for wrong
    classes so training doesn't collapse all capsules toward zero early on.
    """
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred_lengths.shape[-1])
    pos = y_true_onehot * tf.square(tf.maximum(0.0, m_plus - y_pred_lengths))
    neg = lam * (1 - y_true_onehot) * tf.square(tf.maximum(0.0, y_pred_lengths - m_minus))
    return tf.reduce_mean(tf.reduce_sum(pos + neg, axis=1))


def build_csn_model(input_shape, num_classes):
    """
    CPU-friendly CSN design. The original CapsNet paper (on tiny 28x28 MNIST
    digits) ends up with ~1,150 primary capsules. Our spectrograms are much
    bigger (128x130), so without extra downsampling first, this architecture
    would produce ~14,000 primary capsules -- routing between that many
    capsules and 10 class capsules is the computational bottleneck of a
    capsule network, and 14,000 is roughly 12x more than the original design
    ever had to handle. That's what would make CPU training crawl (or run
    out of RAM).

    Fix: add more conv+pooling BEFORE the capsules, so by the time we reach
    PrimaryCaps the spatial resolution is much smaller -- ending up with
    ~800 primary capsules instead, which keeps the routing math manageable
    on a laptop CPU while still giving the capsules meaningful features to
    work with.
    """
    inputs = layers.Input(shape=input_shape)

    # Ordinary conv+pooling layers first, doing the heavy downsampling here
    # (regular convolution is much cheaper per-pixel than capsule routing).
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)          # 128x130 -> 64x65

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)          # 64x65 -> 32x32

    x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="valid")(x)
    x = layers.BatchNormalization()(x)          # 32x32 -> ~15x15

    primary_caps = PrimaryCaps(num_channels=16, dim_capsules=8, kernel_size=3, strides=2)(x)  # ~7x7x16 = ~784 capsules
    class_caps = DigitCaps(num_classes=num_classes, dim_capsules=16, routing_iters=3)(primary_caps)
    outputs = layers.Lambda(capsule_length)(class_caps)  # capsule lengths, used as "probabilities"

    return models.Model(inputs, outputs)


# ---------------------------------------------------------------------
# MODEL 2: Attention-CNN (CBAM-style)
# ---------------------------------------------------------------------

def channel_attention(x, ratio=8):
    """Learns which CHANNELS (filters) matter most for this input."""
    channels = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    shared_dense1 = layers.Dense(channels // ratio, activation="relu")
    shared_dense2 = layers.Dense(channels)

    avg_out = shared_dense2(shared_dense1(avg_pool))
    max_out = shared_dense2(shared_dense1(max_pool))

    attention = layers.Activation("sigmoid")(layers.Add()([avg_out, max_out]))
    attention = layers.Reshape((1, 1, channels))(attention)
    return layers.Multiply()([x, attention])


def spatial_attention(x, kernel_size=7):
    """Learns which TIME/FREQUENCY regions of the spectrogram matter most."""
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid")(concat)
    return layers.Multiply()([x, attention])


def cbam_block(x, ratio=8, kernel_size=7):
    x = channel_attention(x, ratio)
    x = spatial_attention(x, kernel_size)
    return x


def conv_attention_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)                       # <-- the attention module
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    return x


def build_attention_cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = conv_attention_block(inputs, 32)
    x = conv_attention_block(x, 64)
    x = conv_attention_block(x, 128)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


# ---------------------------------------------------------------------
# Training (shared for both model types)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print("Loading features...")
    X_train, y_train, groups_train = load_split("train")
    X_val, y_val, groups_val = load_split("val")
    X_test, y_test, groups_test = load_split("test")

    train_mean, train_std = X_train.mean(), X_train.std()
    X_train = normalize(X_train, train_mean, train_std)[..., np.newaxis]
    X_val = normalize(X_val, train_mean, train_std)[..., np.newaxis]
    X_test = normalize(X_test, train_mean, train_std)[..., np.newaxis]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Building model: {MODEL_TYPE}")

    if MODEL_TYPE == "csn":
        model = build_csn_model(X_train.shape[1:], NUM_CLASSES)
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                      loss=margin_loss, metrics=["accuracy"])
    elif MODEL_TYPE == "attention_cnn":
        model = build_attention_cnn_model(X_train.shape[1:], NUM_CLASSES)
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:
        raise ValueError("MODEL_TYPE must be 'csn' or 'attention_cnn'")

    model.summary()

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(MODEL_OUTPUT_PATH, monitor="val_loss", save_best_only=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
    )

    y_pred_probs = model.predict(X_test)
    y_pred_segments = np.argmax(y_pred_probs, axis=1)
    seg_acc = (y_pred_segments == y_test).mean()
    print(f"\nSegment-level test accuracy: {seg_acc:.4f}")
    print(classification_report(y_test, y_pred_segments, target_names=GENRES))

    y_true_songs, y_pred_songs = aggregate_to_song_level(y_test, y_pred_probs, groups_test)
    song_acc = (y_true_songs == y_pred_songs).mean()
    print(f"\nSong-level test accuracy: {song_acc:.4f}  <- report this as your headline result")
    print(classification_report(y_true_songs, y_pred_songs, target_names=GENRES))
    print(confusion_matrix(y_true_songs, y_pred_songs))

    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")
