"""
Step 3-4: Feature extraction + segment splitting for GTZAN.

Reads the song-level splits from Step 2 (splits/train_songs.csv, val_songs.csv,
test_songs.csv), extracts log-mel spectrograms, and chops each 30-second clip
into shorter segments to multiply the amount of training data.

Because segmenting happens AFTER the song-level split (not before), no segment
from a song in train can ever leak into val or test.

Output: one .npz file per split (features/train.npz, val.npz, test.npz), each
containing:
  X       -> array of log-mel spectrograms, shape (num_segments, n_mels, time_frames)
  y       -> array of integer genre labels, shape (num_segments,)
  groups  -> array of song filenames, shape (num_segments,) -- lets you trace
             which segments came from which song (needed later for song-level
             evaluation / majority voting)
"""

import os
import csv
import numpy as np
import librosa

# ---- CONFIG ----
DATA_DIR = "GTZAN/genres_original"
SPLIT_DIR = "splits"
OUTPUT_DIR = "features"
SR = 22050                 # sample rate GTZAN is recorded at
SEGMENT_DURATION = 3.0     # seconds per segment
N_MELS = 128                # number of mel frequency bins
HOP_LENGTH = 512

# Fixed genre -> integer label mapping, used consistently across all scripts
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}


def read_split_csv(path):
    """Reads a split CSV (from Step 2) into a list of (genre, filename) tuples."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["genre"], row["filename"]))
    return rows


def extract_segments(filepath, sr, segment_duration, n_mels, hop_length):
    """
    Loads one audio file and splits it into fixed-length segments, returning
    a log-mel spectrogram for each segment. All segments get exactly the same
    shape, since they're cut to the same duration.
    """
    y, _ = librosa.load(filepath, sr=sr)  # resample to SR regardless of original rate

    segment_samples = int(segment_duration * sr)
    num_segments = len(y) // segment_samples  # drop any leftover partial segment

    spectrograms = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment_audio = y[start:end]

        mel = librosa.feature.melspectrogram(
            y=segment_audio, sr=sr, n_mels=n_mels, hop_length=hop_length
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)  # power -> decibel scale
        spectrograms.append(log_mel)

    return spectrograms


def process_split(split_rows, data_dir, sr, segment_duration, n_mels, hop_length):
    """Runs extract_segments() over every song in a split, collecting X/y/groups."""
    X, y, groups = [], [], []

    for genre, filename in split_rows:
        filepath = os.path.join(data_dir, genre, filename)

        try:
            segments = extract_segments(filepath, sr, segment_duration, n_mels, hop_length)
        except Exception as e:
            # GTZAN has a couple of known corrupt files (e.g. jazz.00054.wav) -- skip them
            print(f"  [skipped] {filepath}: {e}")
            continue

        label = GENRE_TO_IDX[genre]
        for seg in segments:
            X.append(seg)
            y.append(label)
            groups.append(filename)  # keep track of which song this segment came from

        print(f"  {filename}: {len(segments)} segments")

    return np.array(X), np.array(y), np.array(groups)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        print(f"\nProcessing {split_name} split...")
        split_rows = read_split_csv(os.path.join(SPLIT_DIR, f"{split_name}_songs.csv"))

        X, y, groups = process_split(
            split_rows, DATA_DIR, SR, SEGMENT_DURATION, N_MELS, HOP_LENGTH
        )

        output_path = os.path.join(OUTPUT_DIR, f"{split_name}.npz")
        np.savez(output_path, X=X, y=y, groups=groups)

        print(f"Saved {split_name}.npz -> X shape: {X.shape}, y shape: {y.shape}")

    print(f"\nGenre label mapping: {GENRE_TO_IDX}")
    print("Done. Features saved in 'features/'")
