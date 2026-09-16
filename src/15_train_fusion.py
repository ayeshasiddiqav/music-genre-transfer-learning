"""
15_train_fusion.py

FEATURE FUSION: instead of training separate models on YAMNet and OpenL3
and combining their predictions at the end (ensembling), this
concatenates both embedding sets into ONE input vector and trains a
single classifier on the combined representation.

Why this can beat ensembling: an ensemble only combines final answers,
so each model had to make its decision using half the information.
Fusion lets the classifier see BOTH feature sets at once and learn
relationships between them -- e.g. "when YAMNet says X and OpenL3 says
Y simultaneously, it's usually rock."

Input: 3072 (yamnet_rich) + 1536 (openl3_rich) = 4608 dims

Requires scripts 12 and 14 to have been run first.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_DIR = "features"
MODEL_OUTPUT_PATH = "models/fusion_classifier.keras"
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
NUM_CLASSES = len(GENRES)
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-3


def load_and_fuse(name):
    """Loads both feature sets and concatenates them per segment.
    Both come from the same song-level splits in the same order, so
    rows line up -- we verify that with an assert rather than assuming."""
    a = np.load(os.path.join(FEATURES_DIR, f"{name}_yamnet_rich.npz"))
    b = np.load(os.path.join(FEATURES_DIR, f"{name}_openl3_rich.npz"))

    assert np.array_equal(a["y"], b["y"]), f"{name}: label mismatch between feature sets"
    assert np.array_equal(a["groups"], b["groups"]), f"{name}: group mismatch between feature sets"

    X = np.concatenate([a["X"], b["X"]], axis=1)
    return X, a["y"], a["groups"]


def normalize(X, mean, std):
    return (X - mean) / (std + 1e-8)


def build_classifier(input_dim, num_classes):
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(768, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(384, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(192, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])


def aggregate_to_song_level(y_true_seg, y_pred_probs, groups):
    song_true, psum, pcount = {}, {}, {}
    for true_label, probs, sid in zip(y_true_seg, y_pred_probs, groups):
        if sid not in psum:
            psum[sid] = np.zeros_like(probs)
            pcount[sid] = 0
            song_true[sid] = true_label
        psum[sid] += probs
        pcount[sid] += 1
    sids = list(song_true.keys())
    return (np.array([song_true[s] for s in sids]),
            np.array([np.argmax(psum[s] / pcount[s]) for s in sids]))


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    X_train, y_train, _ = load_and_fuse("train")
    X_val, y_val, _ = load_and_fuse("val")
    X_test, y_test, groups_test = load_and_fuse("test")

    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train, X_val, X_test = normalize(X_train, mean, std), normalize(X_val, mean, std), normalize(X_test, mean, std)

    print(f"Fused feature dims: {X_train.shape[1]}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = build_classifier(X_train.shape[1], NUM_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-6),
        callbacks.ModelCheckpoint(MODEL_OUTPUT_PATH, monitor="val_loss", save_best_only=True),
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=cbs)

    y_pred_probs = model.predict(X_test)
    seg_acc = (np.argmax(y_pred_probs, axis=1) == y_test).mean()
    print(f"\nSegment-level test accuracy: {seg_acc:.4f}")

    y_true_songs, y_pred_songs = aggregate_to_song_level(y_test, y_pred_probs, groups_test)
    song_acc = (y_true_songs == y_pred_songs).mean()
    print(f"\nSong-level test accuracy: {song_acc:.4f}  <- headline result")
    print(classification_report(y_true_songs, y_pred_songs, target_names=GENRES))
    print(confusion_matrix(y_true_songs, y_pred_songs))
