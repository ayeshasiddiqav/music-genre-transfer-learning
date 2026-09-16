"""
Step 7-9: Train a baseline CNN on the GTZAN log-mel spectrogram features
produced in Step 3, and evaluate at both the segment level and the song level.

Song-level evaluation matters here: since each song was chopped into several
3-second segments, one test song is represented by multiple rows in X_test.
Segment-level accuracy alone can look a bit inflated/noisy, so we also
aggregate predictions per song (by averaging predicted probabilities across
its segments) for a more realistic final number -- this mirrors how genre
classification would actually be used in practice: classifying a whole song,
not a 3-second snippet.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

# ---- CONFIG ----
FEATURES_DIR = "features"
MODEL_OUTPUT_PATH = "models/baseline_cnn.keras"
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
NUM_CLASSES = len(GENRES)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3


def load_split(name):
    """Loads one of train/val/test.npz produced by the feature extraction script."""
    data = np.load(os.path.join(FEATURES_DIR, f"{name}.npz"))
    return data["X"], data["y"], data["groups"]


def normalize(X, mean, std):
    """Applies z-score normalization using statistics computed on the TRAIN set only."""
    return (X - mean) / (std + 1e-8)


def build_model(input_shape, num_classes):
    """
    A standard small CNN for spectrogram classification: three conv blocks
    (conv -> batchnorm -> maxpool -> dropout) increasing in filter count,
    followed by dense layers for classification.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def aggregate_to_song_level(y_true_segments, y_pred_probs, groups):
    """
    Averages predicted probabilities across all segments belonging to the
    same song, then takes the argmax. Returns song-level true labels and
    predicted labels (one entry per unique song instead of per segment).
    """
    song_true = {}
    song_probs_sum = {}
    song_probs_count = {}

    for true_label, probs, song_id in zip(y_true_segments, y_pred_probs, groups):
        if song_id not in song_probs_sum:
            song_probs_sum[song_id] = np.zeros_like(probs)
            song_probs_count[song_id] = 0
            song_true[song_id] = true_label
        song_probs_sum[song_id] += probs
        song_probs_count[song_id] += 1

    song_ids = list(song_true.keys())
    y_true_songs = np.array([song_true[s] for s in song_ids])
    y_pred_songs = np.array([
        np.argmax(song_probs_sum[s] / song_probs_count[s]) for s in song_ids
    ])
    return y_true_songs, y_pred_songs


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # ---- Load data ----
    print("Loading features...")
    X_train, y_train, groups_train = load_split("train")
    X_val, y_val, groups_val = load_split("val")
    X_test, y_test, groups_test = load_split("test")

    # ---- Normalize using TRAIN statistics only (never fit normalization on val/test) ----
    train_mean = X_train.mean()
    train_std = X_train.std()
    X_train = normalize(X_train, train_mean, train_std)
    X_val = normalize(X_val, train_mean, train_std)
    X_test = normalize(X_test, train_mean, train_std)

    # ---- Add channel dimension: (N, n_mels, time_frames) -> (N, n_mels, time_frames, 1) ----
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ---- Build and compile model ----
    model = build_model(input_shape=X_train.shape[1:], num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ---- Callbacks ----
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_OUTPUT_PATH, monitor="val_loss", save_best_only=True
    )

    # ---- Train ----
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
    )

    # ---- Segment-level evaluation ----
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nSegment-level test accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred_segments = np.argmax(y_pred_probs, axis=1)

    print("\nSegment-level classification report:")
    print(classification_report(y_test, y_pred_segments, target_names=GENRES))

    # ---- Song-level evaluation (averaged probabilities across a song's segments) ----
    y_true_songs, y_pred_songs = aggregate_to_song_level(y_test, y_pred_probs, groups_test)
    song_acc = (y_true_songs == y_pred_songs).mean()
    print(f"\nSong-level test accuracy: {song_acc:.4f}  <- report this as your headline result")

    print("\nSong-level classification report:")
    print(classification_report(y_true_songs, y_pred_songs, target_names=GENRES))

    print("\nSong-level confusion matrix:")
    print(confusion_matrix(y_true_songs, y_pred_songs))

    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")
