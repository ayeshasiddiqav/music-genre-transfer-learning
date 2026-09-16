"""
13_train_yamnet_rich.py

Classifier for the 3072-dim rich YAMNet features (mean+max+std pooling).

Slightly wider first layer and stronger regularization than script 07,
since the input is 3x higher-dimensional -- more capacity to use the
extra information, more dropout/weight-decay to avoid overfitting on it.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_DIR = "features"
MODEL_OUTPUT_PATH = "models/yamnet_rich_classifier.keras"
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
NUM_CLASSES = len(GENRES)
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3


def load_split(name):
    d = np.load(os.path.join(FEATURES_DIR, f"{name}_yamnet_rich.npz"))
    return d["X"], d["y"], d["groups"]


def normalize(X, mean, std):
    return (X - mean) / (std + 1e-8)


def build_classifier(input_dim, num_classes):
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
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

    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, groups_test = load_split("test")

    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train, X_val, X_test = normalize(X_train, mean, std), normalize(X_val, mean, std), normalize(X_test, mean, std)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = build_classifier(X_train.shape[1], NUM_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
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
