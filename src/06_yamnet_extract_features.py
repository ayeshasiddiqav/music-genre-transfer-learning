"""
06_yamnet_extract_features.py

TRANSFER LEARNING approach: instead of training a CNN/CapsNet from scratch
on only ~700 GTZAN songs, this uses YAMNet -- a model Google pretrained on
AudioSet, a dataset of over 2 million labeled audio clips covering music,
speech, and environmental sounds. YAMNet already "knows" a huge amount
about general audio structure before it ever sees a single GTZAN song.

We do NOT train YAMNet itself (it stays frozen). Instead, we run every
GTZAN segment through it and grab its internal 1024-number "embedding" --
a rich numerical summary of that audio clip. Those embeddings become the
input features for a small classifier (built in the next script), instead
of raw spectrograms.

This is the technique most consistently shown in published research to
push GTZAN genre classification into the 90%+ range, because it borrows
knowledge from millions of external clips instead of learning everything
from scratch out of 700 songs.

Uses the SAME song-level splits from Step 2 (splits/train_songs.csv etc.)
-- same leakage-safe rigor as everything else in this project.

Output: features/train_embeddings.npz, val_embeddings.npz, test_embeddings.npz
  X      -> (num_segments, 1024) YAMNet embeddings
  y      -> (num_segments,) genre labels
  groups -> (num_segments,) source filename, for song-level evaluation

First run needs internet access once, to download YAMNet (~15MB, cached
afterward). Requires: pip install tensorflow_hub
"""

import os
import csv
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

# ---- CONFIG ----
DATA_DIR = "GTZAN/genres_original"
SPLIT_DIR = "splits"
OUTPUT_DIR = "features"
SR = 16000                  # YAMNet requires 16kHz mono audio specifically
SEGMENT_DURATION = 3.0

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

print("Loading YAMNet from TensorFlow Hub (downloads once, then cached locally)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")


def read_split_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["genre"], row["filename"]))
    return rows


def get_yamnet_embedding(audio):
    """
    Runs one segment of audio through YAMNet and returns ONE embedding
    vector (1024 numbers) summarizing the whole segment. YAMNet internally
    splits the segment into short ~0.96s frames and returns one embedding
    per frame -- we average across frames to get a single fixed-size
    vector per segment, which is what our classifier expects as input.
    """
    audio = audio.astype(np.float32)
    _, embeddings, _ = yamnet_model(audio)              # (num_frames, 1024)
    return tf.reduce_mean(embeddings, axis=0).numpy()    # -> (1024,)


def extract_segment_embeddings(filepath, sr, segment_duration):
    y, _ = librosa.load(filepath, sr=sr)
    segment_samples = int(segment_duration * sr)
    num_segments = len(y) // segment_samples

    embeddings = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        embeddings.append(get_yamnet_embedding(y[start:end]))

    return embeddings


def process_split(split_rows, data_dir, sr, segment_duration):
    X, y, groups = [], [], []
    for genre, filename in split_rows:
        filepath = os.path.join(data_dir, genre, filename)
        try:
            embeddings = extract_segment_embeddings(filepath, sr, segment_duration)
        except Exception as e:
            print(f"  [skipped] {filepath}: {e}")
            continue

        label = GENRE_TO_IDX[genre]
        for emb in embeddings:
            X.append(emb)
            y.append(label)
            groups.append(filename)

        print(f"  {filename}: {len(embeddings)} embeddings")

    return np.array(X), np.array(y), np.array(groups)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        print(f"\nProcessing {split_name} split with YAMNet...")
        split_rows = read_split_csv(os.path.join(SPLIT_DIR, f"{split_name}_songs.csv"))
        X, y, groups = process_split(split_rows, DATA_DIR, SR, SEGMENT_DURATION)

        output_path = os.path.join(OUTPUT_DIR, f"{split_name}_embeddings.npz")
        np.savez(output_path, X=X, y=y, groups=groups)
        print(f"Saved {split_name}_embeddings.npz -> X shape: {X.shape}")

    print("\nDone. YAMNet embeddings saved in 'features/'.")
