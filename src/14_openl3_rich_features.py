"""
14_openl3_rich_features.py

Same mean+max+std pooling upgrade we applied to YAMNet in script 12,
now applied to OpenL3. That change took YAMNet from 85.23% -> 87%, and
OpenL3 is currently your single strongest model by validation accuracy
(0.9333), so it's the most promising place to apply the same idea.

Output: features/{train,val,test}_openl3_rich.npz  (1536-dim = 512 x 3)
"""

import os
import csv
import numpy as np
import librosa
import openl3

DATA_DIR = "GTZAN/genres_original"
SPLIT_DIR = "splits"
OUTPUT_DIR = "features"
SR = 22050
SEGMENT_DURATION = 3.0
HOP_SIZE = 0.5

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

print("Loading OpenL3 (music)...")
openl3_model = openl3.models.load_audio_embedding_model(
    input_repr="mel128", content_type="music", embedding_size=512
)


def read_split_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["genre"], row["filename"]))
    return rows


def rich_embedding(audio, sr):
    emb, _ = openl3.get_audio_embedding(
        audio, sr, model=openl3_model, hop_size=HOP_SIZE, verbose=0
    )
    return np.concatenate([emb.mean(axis=0), emb.max(axis=0), emb.std(axis=0)])  # (1536,)


def extract_segments(filepath):
    y, _ = librosa.load(filepath, sr=SR)
    seg_len = int(SEGMENT_DURATION * SR)
    n = len(y) // seg_len
    return [rich_embedding(y[i * seg_len:(i + 1) * seg_len], SR) for i in range(n)]


def process_split(rows):
    X, y, groups = [], [], []
    for genre, filename in rows:
        path = os.path.join(DATA_DIR, genre, filename)
        try:
            embs = extract_segments(path)
        except Exception as e:
            print(f"  [skipped] {path}: {e}")
            continue
        for emb in embs:
            X.append(emb)
            y.append(GENRE_TO_IDX[genre])
            groups.append(filename)
        print(f"  {filename}: {len(embs)} embeddings")
    return np.array(X), np.array(y), np.array(groups)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split}...")
        rows = read_split_csv(os.path.join(SPLIT_DIR, f"{split}_songs.csv"))
        X, y, groups = process_split(rows)
        np.savez(os.path.join(OUTPUT_DIR, f"{split}_openl3_rich.npz"), X=X, y=y, groups=groups)
        print(f"Saved {split}_openl3_rich.npz -> {X.shape}")
    print("\nDone.")
