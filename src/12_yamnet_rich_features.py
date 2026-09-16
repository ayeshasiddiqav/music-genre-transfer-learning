"""
12_yamnet_rich_features.py

Improved version of 06_yamnet_extract_features.py.

The change: instead of summarizing each segment's YAMNet frames by
AVERAGE alone (which throws away a lot of information), we concatenate
three different statistics across frames:

  mean -> the typical sound of the segment
  max  -> the most extreme/prominent moments (transients, hits, peaks)
  std  -> how MUCH the sound varies over time within the segment

That last one matters a lot for genre: e.g. classical tends to have high
dynamic variation while electronic/disco tends to be more uniform --
information the mean alone completely hides.

Result: 3072-dim features (1024 x 3) instead of 1024, carrying strictly
more information for the classifier to work with.

Same song-level splits, same leakage-safe rigor.
Output: features/{train,val,test}_yamnet_rich.npz
"""

import os
import csv
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

DATA_DIR = "GTZAN/genres_original"
SPLIT_DIR = "splits"
OUTPUT_DIR = "features"
SR = 16000
SEGMENT_DURATION = 3.0

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")


def read_split_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["genre"], row["filename"]))
    return rows


def rich_embedding(audio):
    """mean + max + std pooling across YAMNet's per-frame embeddings."""
    audio = audio.astype(np.float32)
    _, embeddings, _ = yamnet_model(audio)      # (num_frames, 1024)
    e = embeddings.numpy()
    return np.concatenate([e.mean(axis=0), e.max(axis=0), e.std(axis=0)])  # (3072,)


def extract_segments(filepath):
    y, _ = librosa.load(filepath, sr=SR)
    seg_len = int(SEGMENT_DURATION * SR)
    n = len(y) // seg_len
    return [rich_embedding(y[i * seg_len:(i + 1) * seg_len]) for i in range(n)]


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
        np.savez(os.path.join(OUTPUT_DIR, f"{split}_yamnet_rich.npz"), X=X, y=y, groups=groups)
        print(f"Saved {split}_yamnet_rich.npz -> {X.shape}")
    print("\nDone.")
