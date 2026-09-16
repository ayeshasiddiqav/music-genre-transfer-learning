"""
Step 2: Song-level train/val/test split for GTZAN.

WHY THIS MATTERS:
GTZAN clips will later be chopped into short segments (Step 4) to create
more training data. If you split segments randomly into train/val/test,
segments from the SAME original song can end up on both sides of the split.
The model then partly memorizes the song instead of learning genre features,
which inflates accuracy artificially. Splitting whole songs FIRST avoids this.

WHAT THIS SCRIPT DOES:
1. Scans the GTZAN folder structure (genre -> list of .wav files)
2. Splits the 100 songs per genre into train/val/test (stratified by genre,
   so each split gets a proportional number of songs from every genre)
3. Saves the split as three CSV files listing which song belongs to which split

Expected folder structure (default GTZAN Kaggle download):
GTZAN/
  genres_original/
    blues/    blues.00000.wav ... blues.00099.wav
    classical/
    country/
    disco/
    hiphop/
    jazz/
    metal/
    pop/
    reggae/
    rock/
"""

import os
import csv
import random

# ---- CONFIG ----
DATA_DIR = "GTZAN/genres_original"   # change this to wherever you extracted GTZAN
OUTPUT_DIR = "splits"                # where the split CSVs will be saved
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15                    # should sum to 1.0 with the above
RANDOM_SEED = 42                     # fixed seed = reproducible split


def get_songs_by_genre(data_dir):
    """
    Walks the GTZAN folder and returns a dict:
    { "blues": ["blues.00000.wav", "blues.00001.wav", ...], "rock": [...], ... }
    """
    songs_by_genre = {}
    genres = sorted(os.listdir(data_dir))  # e.g. ['blues', 'classical', ...]

    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_path):
            continue  # skip any stray non-folder files
        files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
        songs_by_genre[genre] = sorted(files)  # sorted = deterministic order before shuffling

    return songs_by_genre


def split_songs(songs, train_ratio, val_ratio, test_ratio, seed):
    """
    Takes a list of song filenames for ONE genre and splits it into
    train/val/test lists. Shuffling happens with a fixed seed so the
    split is the same every time you run this script.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    songs = songs.copy()
    random.Random(seed).shuffle(songs)  # shuffle using a local Random instance (doesn't affect global state)

    n = len(songs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # test gets whatever remains, so rounding doesn't drop any files
    train_songs = songs[:n_train]
    val_songs = songs[n_train:n_train + n_val]
    test_songs = songs[n_train + n_val:]

    return train_songs, val_songs, test_songs


def build_full_split(data_dir, train_ratio, val_ratio, test_ratio, seed):
    """
    Applies split_songs() to every genre separately (this is the
    "stratified by genre" part) and combines the results into
    three flat lists: all train songs, all val songs, all test songs.
    Each entry is a (genre, filename) tuple.
    """
    songs_by_genre = get_songs_by_genre(data_dir)

    train_set, val_set, test_set = [], [], []

    for genre, songs in songs_by_genre.items():
        train_songs, val_songs, test_songs = split_songs(
            songs, train_ratio, val_ratio, test_ratio, seed
        )
        train_set.extend((genre, fname) for fname in train_songs)
        val_set.extend((genre, fname) for fname in val_songs)
        test_set.extend((genre, fname) for fname in test_songs)

        print(f"{genre}: {len(train_songs)} train / {len(val_songs)} val / {len(test_songs)} test")

    return train_set, val_set, test_set


def save_split_csv(split_list, output_path):
    """Saves a list of (genre, filename) tuples to a CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["genre", "filename"])
        writer.writerows(split_list)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_set, val_set, test_set = build_full_split(
        DATA_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    save_split_csv(train_set, os.path.join(OUTPUT_DIR, "train_songs.csv"))
    save_split_csv(val_set, os.path.join(OUTPUT_DIR, "val_songs.csv"))
    save_split_csv(test_set, os.path.join(OUTPUT_DIR, "test_songs.csv"))

    print(f"\nTotal songs -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    print(f"Split files saved in '{OUTPUT_DIR}/'")
