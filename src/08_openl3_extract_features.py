"""
08_openl3_extract_features.py

Like script 06 (YAMNet), but using OpenL3 instead -- a pretrained audio
embedding model trained SPECIFICALLY on AudioSet videos that are mostly
musical performances, using content_type="music". YAMNet was trained on
a general mix of speech, environmental sounds, AND music -- OpenL3's
music-specific training makes it a more targeted match for a genre
classification task like this one, and research has generally found it
performs as well as or better than general-purpose embeddings on
music-specific downstream tasks.

Same song-level splits, same leakage-safe rigor, same overall pipeline
shape as the YAMNet version -- only the embedding model changes.

Output: features/train_openl3.npz, val_openl3.npz, test_openl3.npz
  X      -> (num_segments, 512) OpenL3 embeddings
  y      -> (num_segments,) genre labels
  groups -> (num_segments,) source filename, for song-level evaluation

Requires: pip install openl3
First run downloads the OpenL3 model weights once (cached afterward).
"""

import os
import csv
import numpy as np
import librosa
import openl3

# ---- CONFIG ----
DATA_DIR = "GTZAN/genres_original"
SPLIT_DIR = "splits"
OUTPUT_DIR = "features"
SR = 22050
SEGMENT_DURATION = 3.0
HOP_SIZE = 0.5   # seconds between OpenL3 analysis frames within a segment

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

print("Loading OpenL3 model (music-trained, downloads once, then cached)...")
openl3_model = openl3.models.load_audio_embedding_model(
    input_repr="mel128", content_type="music", embedding_size=512
)


def read_split_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["genre"], row["filename"]))
    return rows


def get_openl3_embedding(audio, sr):
    """
    Runs one segment through OpenL3 and returns ONE embedding vector
    (512 numbers). OpenL3 produces one embedding per short analysis
    frame within the segment -- we average across frames to get a
    single fixed-size vector per segment, same idea as the YAMNet script.
    """
    emb, _ = openl3.get_audio_embedding(
        audio, sr, model=openl3_model, hop_size=HOP_SIZE, verbose=0
    )
    return emb.mean(axis=0)  # -> (512,)


def extract_segment_embeddings(filepath, sr, segment_duration):
    y, _ = librosa.load(filepath, sr=sr)
    segment_samples = int(segment_duration * sr)
    num_segments = len(y) // segment_samples

    embeddings = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        embeddings.append(get_openl3_embedding(y[start:end], sr))

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
        print(f"\nProcessing {split_name} split with OpenL3...")
        split_rows = read_split_csv(os.path.join(SPLIT_DIR, f"{split_name}_songs.csv"))
        X, y, groups = process_split(split_rows, DATA_DIR, SR, SEGMENT_DURATION)

        output_path = os.path.join(OUTPUT_DIR, f"{split_name}_openl3.npz")
        np.savez(output_path, X=X, y=y, groups=groups)
        print(f"Saved {split_name}_openl3.npz -> X shape: {X.shape}")

    print("\nDone. OpenL3 embeddings saved in 'features/'.")
