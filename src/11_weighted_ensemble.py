"""
11_weighted_ensemble.py

An improved version of 10_ensemble_predict.py.

What we learned from the last run: naive equal-weight averaging can
actually HURT you if one model is meaningfully weaker than the others --
a weak model's vote drags the average down, even when the strongest
model alone was doing better on its own. The last ensemble (Attention-CNN
+ YAMNet, equal weight) landed at 84.56%, slightly BELOW YAMNet alone
(85.23%), because Attention-CNN (78.52%) pulled the average down.

The fix: WEIGHTED ensembling. Instead of every model getting an equal
vote, each model's vote is weighted by how well IT does on the
VALIDATION set (never the test set -- using test accuracy to set weights
would itself be a form of leakage, since it would let test performance
influence the final prediction). Stronger models get proportionally more
influence on the final answer; weaker models get less say, rather than
an equal, undeserved vote.

This still trains nothing new -- it only re-uses your already-trained
models, combined more intelligently based on real, honestly-measured
performance differences.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
NUM_CLASSES = len(GENRES)


# ---------------------------------------------------------------------
# Attention-CNN architecture (same as 04_train_advanced_models.py) --
# rebuilt from code + load_weights, to avoid the Lambda-layer reload
# issue from before.
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


def song_level_accuracy(song_probs_dict):
    correct = sum(1 for true_label, probs in song_probs_dict.values() if np.argmax(probs) == true_label)
    return correct / len(song_probs_dict)


def get_cnn_style_probs(model_path):
    """Returns (val_song_probs, test_song_probs), or (None, None) if missing."""
    if not os.path.exists(model_path):
        print(f"  [skip] {model_path} not found")
        return None, None

    train_data = np.load("features/train.npz")
    val_data = np.load("features/val.npz")
    test_data = np.load("features/test.npz")

    train_mean, train_std = train_data["X"].mean(), train_data["X"].std()
    X_val = normalize(val_data["X"], train_mean, train_std)[..., np.newaxis]
    X_test = normalize(test_data["X"], train_mean, train_std)[..., np.newaxis]

    model = build_attention_cnn_model(input_shape=X_test.shape[1:], num_classes=NUM_CLASSES)
    model.load_weights(model_path)

    val_probs = model.predict(X_val, verbose=0)
    test_probs = model.predict(X_test, verbose=0)

    print(f"  [loaded] {model_path}")
    return (song_level_probs(val_data["y"], val_probs, val_data["groups"]),
            song_level_probs(test_data["y"], test_probs, test_data["groups"]))


def build_simple_head(input_dim, num_classes):
    """Architecture used by 07 (yamnet) and 09 (openl3) classifiers."""
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])


def build_rich_head(input_dim, num_classes):
    """Architecture used by 13 (yamnet_rich) classifier."""
    from tensorflow.keras import regularizers
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


def get_embedding_style_probs(model_path, suffix, head="simple"):
    if not os.path.exists(model_path):
        print(f"  [skip] {model_path} not found")
        return None, None

    train_data = np.load(f"features/train_{suffix}.npz")
    val_data = np.load(f"features/val_{suffix}.npz")
    test_data = np.load(f"features/test_{suffix}.npz")

    train_mean = train_data["X"].mean(axis=0)
    train_std = train_data["X"].std(axis=0)
    X_val = normalize(val_data["X"], train_mean, train_std)
    X_test = normalize(test_data["X"], train_mean, train_std)

    # Rebuild from code + load weights, rather than load_model -- avoids
    # Keras version mismatches when models were saved under a different
    # Keras version than the one currently installed.
    builder = build_rich_head if head == "rich" else build_simple_head
    model = builder(X_test.shape[1], NUM_CLASSES)
    model.load_weights(model_path)

    val_probs = model.predict(X_val, verbose=0)
    test_probs = model.predict(X_test, verbose=0)

    print(f"  [loaded] {model_path}")
    return (song_level_probs(val_data["y"], val_probs, val_data["groups"]),
            song_level_probs(test_data["y"], test_probs, test_data["groups"]))


if __name__ == "__main__":
    print("Gathering validation + test predictions from each trained model...")

    candidates = {}

    val_p, test_p = get_cnn_style_probs("models/attention_cnn.keras")
    if val_p is not None:
        candidates["attention_cnn"] = (val_p, test_p)

    val_p, test_p = get_embedding_style_probs("models/yamnet_classifier.keras", "embeddings")
    if val_p is not None:
        candidates["yamnet"] = (val_p, test_p)

    val_p, test_p = get_embedding_style_probs("models/yamnet_rich_classifier.keras", "yamnet_rich", head="rich")
    if val_p is not None:
        candidates["yamnet_rich"] = (val_p, test_p)

    val_p, test_p = get_embedding_style_probs("models/openl3_classifier.keras", "openl3")
    if val_p is not None:
        candidates["openl3"] = (val_p, test_p)

    if len(candidates) < 2:
        raise RuntimeError("Need at least 2 trained models to ensemble.")

    # ---- Each model's weight comes from its VALIDATION accuracy ----
    # We raise accuracy to a power before normalizing. Plain linear
    # weighting barely separates models (0.87 vs 0.78 is only a 1.1x
    # difference in influence), which is why a weak model can still drag
    # a strong one down. Raising to a power sharpens the gap so clearly
    # better models dominate, while weak ones still contribute a little
    # rather than being dropped entirely.
    POWER = 8

    weights = {}
    print("\nPer-model validation accuracy (used as ensemble weight):")
    for name, (val_probs, _) in candidates.items():
        acc = song_level_accuracy(val_probs)
        weights[name] = acc ** POWER
        print(f"  {name}: {acc:.4f}")

    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    print("Normalized weights:", {k: round(v, 3) for k, v in weights.items()})

    # ---- Weighted combination, evaluated on the TEST set ----
    test_probs_by_model = {name: test_p for name, (_, test_p) in candidates.items()}
    common_songs = sorted(set.intersection(*[set(d.keys()) for d in test_probs_by_model.values()]))
    print(f"\nSongs common to all models: {len(common_songs)}")

    y_true, y_pred = [], []
    for song_id in common_songs:
        true_label = test_probs_by_model[list(test_probs_by_model.keys())[0]][song_id][0]
        combined = np.zeros(NUM_CLASSES)
        for name, probs_dict in test_probs_by_model.items():
            combined += weights[name] * probs_dict[song_id][1]
        y_true.append(true_label)
        y_pred.append(np.argmax(combined))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = (y_true == y_pred).mean()
    print(f"\nWeighted ensemble song-level test accuracy: {acc:.4f}  <- report this as your headline result")
    print(classification_report(y_true, y_pred, target_names=GENRES))
    print(confusion_matrix(y_true, y_pred))
