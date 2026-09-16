"""
10_ensemble_predict.py

Combines predictions from multiple ALREADY-TRAINED models by averaging
their song-level probability estimates, then taking the genre with the
highest combined probability. This is one of the most reliable,
well-evidenced ways to beat any single model's accuracy -- especially
when the models are meaningfully different from each other (mel-
spectrogram CNN vs. YAMNet embeddings vs. OpenL3 embeddings), since
different models tend to make different mistakes that partly cancel out
when averaged together.

This script trains NOTHING -- it only loads models you've already built
and combines their existing test-set predictions. Zero leakage risk:
every prediction still comes from the same honest, song-level-split test
set every other script in this project has used.

Works with whatever subset of these you've already trained (skips any
that are missing, with a message):
  models/attention_cnn.keras     (needs features/train.npz, test.npz)
  models/yamnet_classifier.keras (needs features/train_embeddings.npz, test_embeddings.npz)
  models/openl3_classifier.keras (needs features/train_openl3.npz, test_openl3.npz)

Needs at least 2 trained models present to run.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
NUM_CLASSES = len(GENRES)


# ---------------------------------------------------------------------
# Attention-CNN architecture, copied from 04_train_advanced_models.py.
# We rebuild the model structure here directly from code, then load only
# the saved WEIGHTS onto it (model.load_weights), rather than reloading
# the whole saved architecture (model.load_model). This sidesteps a
# Keras 3 quirk where Lambda layers inside the attention module don't
# always reload cleanly from a saved architecture -- rebuilding from the
# same code we trained with and loading weights is more robust.
# ---------------------------------------------------------------------

def channel_attention(x, ratio=8):
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
    x = cbam_block(x)
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


def normalize(X, mean, std):
    return (X - mean) / (std + 1e-8)


def song_level_probs(y_true_segments, y_pred_probs, groups):
    """Averages segment-level probabilities into one probability vector per
    song. Returns {song_id: (true_label, averaged_probability_vector)}."""
    song_true, song_probs_sum, song_probs_count = {}, {}, {}
    for true_label, probs, song_id in zip(y_true_segments, y_pred_probs, groups):
        if song_id not in song_probs_sum:
            song_probs_sum[song_id] = np.zeros_like(probs)
            song_probs_count[song_id] = 0
            song_true[song_id] = true_label
        song_probs_sum[song_id] += probs
        song_probs_count[song_id] += 1

    return {
        song_id: (song_true[song_id], song_probs_sum[song_id] / song_probs_count[song_id])
        for song_id in song_true
    }


def get_cnn_style_song_probs(model_path):
    """For baseline_cnn / attention_cnn: mel-spectrogram input, scalar normalization."""
    if not os.path.exists(model_path):
        print(f"  [skip] {model_path} not found")
        return None

    train_data = np.load("features/train.npz")
    test_data = np.load("features/test.npz")

    train_mean, train_std = train_data["X"].mean(), train_data["X"].std()
    X_test = normalize(test_data["X"], train_mean, train_std)[..., np.newaxis]

    model = build_attention_cnn_model(input_shape=X_test.shape[1:], num_classes=NUM_CLASSES)
    model.load_weights(model_path)
    probs = model.predict(X_test, verbose=0)

    print(f"  [loaded] {model_path}")
    return song_level_probs(test_data["y"], probs, test_data["groups"])


def get_embedding_style_song_probs(model_path, embeddings_suffix):
    """For yamnet_classifier / openl3_classifier: embedding input, per-feature normalization."""
    if not os.path.exists(model_path):
        print(f"  [skip] {model_path} not found")
        return None

    train_data = np.load(f"features/train_{embeddings_suffix}.npz")
    test_data = np.load(f"features/test_{embeddings_suffix}.npz")

    train_mean = train_data["X"].mean(axis=0)
    train_std = train_data["X"].std(axis=0)
    X_test = normalize(test_data["X"], train_mean, train_std)

    model = tf.keras.models.load_model(model_path, safe_mode=False)
    probs = model.predict(X_test, verbose=0)

    print(f"  [loaded] {model_path}")
    return song_level_probs(test_data["y"], probs, test_data["groups"])


if __name__ == "__main__":
    print("Gathering predictions from each trained model...")

    all_model_probs = {}

    r = get_cnn_style_song_probs("models/attention_cnn.keras")
    if r is not None:
        all_model_probs["attention_cnn"] = r

    r = get_embedding_style_song_probs("models/yamnet_classifier.keras", "embeddings")
    if r is not None:
        all_model_probs["yamnet"] = r

    r = get_embedding_style_song_probs("models/openl3_classifier.keras", "openl3")
    if r is not None:
        all_model_probs["openl3"] = r

    if len(all_model_probs) < 2:
        raise RuntimeError(
            "Need at least 2 trained models to ensemble. "
            "Train more models first (e.g. run 07 and/or 09)."
        )

    print(f"\nEnsembling {len(all_model_probs)} models: {list(all_model_probs.keys())}")

    # Only ensemble songs every included model actually has a prediction for
    common_songs = sorted(set.intersection(*[set(d.keys()) for d in all_model_probs.values()]))
    print(f"Songs common to all models: {len(common_songs)}")

    y_true, y_pred = [], []
    for song_id in common_songs:
        true_label = all_model_probs[list(all_model_probs.keys())[0]][song_id][0]
        avg_probs = np.mean([all_model_probs[m][song_id][1] for m in all_model_probs], axis=0)
        y_true.append(true_label)
        y_pred.append(np.argmax(avg_probs))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    ensemble_acc = (y_true == y_pred).mean()
    print(f"\nEnsemble song-level test accuracy: {ensemble_acc:.4f}  <- report this as your headline result")
    print(classification_report(y_true, y_pred, target_names=GENRES))
    print(confusion_matrix(y_true, y_pred))
