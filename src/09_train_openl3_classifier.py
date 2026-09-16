"""
09_train_openl3_classifier.py

Same idea as script 07 (the YAMNet classifier), but trained on OpenL3
embeddings instead. A small MLP head on top of OpenL3's frozen,
music-pretrained 512-number embeddings.

Same song-level evaluation (segment-level + song-level via averaged
probabilities) as every other script in this project.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_DIR = "features"
MODEL_OUTPUT_PATH = "models/openl3_classifier.keras"
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
NUM_CLASSES = len(GENRES)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3


def load_split(name):
    data = np.load(os.path.join(FEATURES_DIR, f"{name}_openl3.npz"))
    return data["X"], data["y"], data["groups"]


def normalize(X, mean, std):
    return (X - mean) / (std + 1e-8)


def build_classifier(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


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


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print("Loading OpenL3 embeddings...")
    X_train, y_train, groups_train = load_split("train")
    X_val, y_val, groups_val = load_split("val")
    X_test, y_test, groups_test = load_split("test")

    train_mean, train_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = normalize(X_train, train_mean, train_std)
    X_val = normalize(X_val, train_mean, train_std)
    X_test = normalize(X_test, train_mean, train_std)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = build_classifier(X_train.shape[1], NUM_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
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
