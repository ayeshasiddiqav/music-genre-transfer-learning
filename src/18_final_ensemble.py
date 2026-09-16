"""
18_final_ensemble.py   -- run LOCALLY.

Combines the fine-tuned AST's exported predictions (ast_probs.npz, from
Colab) with your locally-trained models, weighted by validation accuracy.

AST (89.93%) and Fusion (89.93%) are tied on overall accuracy but fail
differently -- AST is much stronger on rock (0.93 F1 vs 0.84), Fusion is
stronger on pop. Combining models that make DIFFERENT mistakes is where
ensembling actually gains points, rather than just averaging two models
that were already wrong in the same places.

Weights come from VALIDATION accuracy only; the test set never
influences model selection or weighting. Song IDs are filenames, shared
across all models via the same song-level split CSVs -- so nothing
crosses splits.

Needs: ast_probs.npz placed in this folder (downloaded from Colab).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
NUM_CLASSES = len(GENRES)
POWER = 8   # sharpens weighting so stronger models dominate


def normalize(X, mean, std):
    return (X - mean) / (std + 1e-8)


def song_level_probs(y_seg, probs, groups):
    psum, pcount, ptrue = {}, {}, {}
    for true_label, p, sid in zip(y_seg, probs, groups):
        if sid not in psum:
            psum[sid] = np.zeros_like(p)
            pcount[sid] = 0
            ptrue[sid] = true_label
        psum[sid] += p
        pcount[sid] += 1
    return {s: (ptrue[s], psum[s] / pcount[s]) for s in ptrue}


def acc_of(d):
    return sum(1 for t, p in d.values() if np.argmax(p) == t) / len(d)


# ---- Fusion model (yamnet_rich + openl3_rich) ----
def build_fusion_head(input_dim, num_classes):
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(768, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(384, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(192, activation="relu"),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])


def get_fusion():
    path = "models/fusion_classifier.keras"
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None, None

    def load_fused(name):
        a = np.load(f"features/{name}_yamnet_rich.npz")
        b = np.load(f"features/{name}_openl3_rich.npz")
        return np.concatenate([a["X"], b["X"]], axis=1), a["y"], a["groups"]

    X_train, _, _ = load_fused("train")
    X_val, y_val, g_val = load_fused("val")
    X_test, y_test, g_test = load_fused("test")
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)

    model = build_fusion_head(X_test.shape[1], NUM_CLASSES)
    model.load_weights(path)
    print(f"  [loaded] {path}")
    return (song_level_probs(y_val, model.predict(normalize(X_val, mean, std), verbose=0), g_val),
            song_level_probs(y_test, model.predict(normalize(X_test, mean, std), verbose=0), g_test))


def get_ast():
    path = "ast_probs.npz"
    if not os.path.exists(path):
        print(f"  [skip] {path} not found -- export it from Colab first")
        return None, None
    d = np.load(path, allow_pickle=True)
    val = {str(s): (int(y), p) for s, y, p in zip(d["val_songs"], d["val_y"], d["val_probs"])}
    test = {str(s): (int(y), p) for s, y, p in zip(d["test_songs"], d["test_y"], d["test_probs"])}
    print(f"  [loaded] {path}")
    return val, test


if __name__ == "__main__":
    candidates = {}

    v, t = get_fusion()
    if v is not None:
        candidates["fusion"] = (v, t)

    v, t = get_ast()
    if v is not None:
        candidates["ast"] = (v, t)

    if len(candidates) < 2:
        raise RuntimeError("Need both fusion and AST predictions.")

    weights = {}
    print("\nValidation accuracy per model:")
    for name, (val_p, _) in candidates.items():
        a = acc_of(val_p)
        weights[name] = a ** POWER
        print(f"  {name}: {a:.4f}")
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    print("Normalized weights:", {k: round(v, 3) for k, v in weights.items()})

    test_by_model = {n: t for n, (_, t) in candidates.items()}
    common = sorted(set.intersection(*[set(d.keys()) for d in test_by_model.values()]))
    print(f"\nSongs common to all models: {len(common)}")

    y_true, y_pred = [], []
    for sid in common:
        true_label = test_by_model[list(test_by_model)[0]][sid][0]
        combined = sum(weights[n] * test_by_model[n][sid][1] for n in test_by_model)
        y_true.append(true_label)
        y_pred.append(np.argmax(combined))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean()

    print(f"\nFINAL ensemble song-level test accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=GENRES))
    print(confusion_matrix(y_true, y_pred))
