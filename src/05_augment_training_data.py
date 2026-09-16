"""
05_augment_training_data.py

Adds data augmentation to the TRAINING set only (never touch val/test --
augmenting those would make your evaluation numbers meaningless, since
you'd be testing on artificially modified data instead of real songs).

For every training segment, this creates 3 extra augmented copies on top
of the original:
  1. Pitch shifted up 2 semitones
  2. Pitch shifted down 2 semitones
  3. Light background noise added

This roughly QUADRUPLES your training data (1 original + 3 augmented
versions per segment) -- usually the single biggest lever for improving
accuracy on a small dataset like GTZAN.

This OVERWRITES features/train.npz with the new, larger, augmented
version. features/val.npz and features/test.npz are left completely
untouched. Run this AFTER 02_feature_extraction.py, then (re)run
03_train_model.py or 04_train_advanced_models.py afterward -- they'll
automatically pick up the bigger training set.
"""

import os
import csv
import numpy as np
import librosa

# ---- CONFIG (must match 02_feature_extraction.py) ----
DATA_DIR = "GTZAN/genres_original"
SPLIT_DIR = "splits"
OUTPUT_DIR = "features"
SR = 22050
SEGMENT_DURATION = 3.0
N_MELS = 128
HOP_LENGTH = 512
PITCH_SHIFT_STEPS = 2       # semitones up/down
NOISE_FACTOR = 0.005         # how loud the added noise is

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}


def read_split_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["genre"], row["filename"]))
    return rows


def to_log_mel(audio, sr, n_mels, hop_length):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    return librosa.power_to_db(mel, ref=np.max)


def augment_segment(audio, sr):
    """Returns a list of augmented audio versions (NOT including the original)."""
    versions = []
    versions.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=PITCH_SHIFT_STEPS))
    versions.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-PITCH_SHIFT_STEPS))
    noise = np.random.randn(len(audio))
    versions.append(audio + NOISE_FACTOR * noise)
    return versions


def extract_segments_augmented(filepath, sr, segment_duration, n_mels, hop_length):
    """Like the original extract_segments, but returns the original PLUS
    augmented log-mel spectrograms for every segment."""
    y, _ = librosa.load(filepath, sr=sr)
    segment_samples = int(segment_duration * sr)
    num_segments = len(y) // segment_samples

    spectrograms = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment_audio = y[start:end]

        spectrograms.append(to_log_mel(segment_audio, sr, n_mels, hop_length))  # original
        for aug_audio in augment_segment(segment_audio, sr):                    # augmented copies
            spectrograms.append(to_log_mel(aug_audio, sr, n_mels, hop_length))

    return spectrograms


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Processing train split WITH augmentation (slower than script 2, be patient)...")
    split_rows = read_split_csv(os.path.join(SPLIT_DIR, "train_songs.csv"))

    X, y, groups = [], [], []
    for genre, filename in split_rows:
        filepath = os.path.join(DATA_DIR, genre, filename)
        try:
            segments = extract_segments_augmented(filepath, SR, SEGMENT_DURATION, N_MELS, HOP_LENGTH)
        except Exception as e:
            print(f"  [skipped] {filepath}: {e}")
            continue

        label = GENRE_TO_IDX[genre]
        for seg in segments:
            X.append(seg)
            y.append(label)
            groups.append(filename)

        print(f"  {filename}: {len(segments)} spectrograms (original + augmented)")

    X, y, groups = np.array(X), np.array(y), np.array(groups)
    output_path = os.path.join(OUTPUT_DIR, "train.npz")
    np.savez(output_path, X=X, y=y, groups=groups)

    print(f"\nSaved augmented train.npz -> X shape: {X.shape}, y shape: {y.shape}")
    print("val.npz and test.npz were NOT touched -- still the original, un-augmented versions.")
    print("Done. Now (re)run 03_train_model.py or 04_train_advanced_models.py to train on this bigger dataset.")
