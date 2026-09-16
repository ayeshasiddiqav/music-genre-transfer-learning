"""
17_export_ast_probs_colab.py   -- run in COLAB, after 16 finished.

Loads the fine-tuned AST checkpoint and exports its per-song predicted
probabilities for the val and test splits to a small .npz file.

Why: your AST model lives on Colab, but your fusion model lives on your
laptop. Rather than moving either model, we just export AST's
PREDICTIONS (a tiny file, a few KB) and ensemble locally.

Song IDs are filenames, which match exactly across both models since
both read the same song-level split CSVs -- so the two sets of
predictions line up correctly with no leakage.

Output: ast_probs.npz  -- download this from the Colab Files panel.
"""

import os
import csv
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification

SPLIT_DIR = "."
DATA_DIR = "GTZAN/Data/genres_original"
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
CHECKPOINT = "ast_gtzan.pt"
SR = 16000
SEGMENT_DURATION = 5.0
BATCH_SIZE = 8

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
device = "cuda" if torch.cuda.is_available() else "cpu"


def read_split_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["genre"], row["filename"]))
    return rows


class GTZANSegments(Dataset):
    def __init__(self, split_rows, feature_extractor):
        self.feature_extractor = feature_extractor
        self.items = []
        seg_samples = int(SEGMENT_DURATION * SR)
        for genre, filename in split_rows:
            path = os.path.join(DATA_DIR, genre, filename)
            if not os.path.exists(path):
                continue
            try:
                duration = librosa.get_duration(path=path)
            except Exception:
                continue
            for i in range(int(duration * SR) // seg_samples):
                self.items.append((path, i * seg_samples, GENRE_TO_IDX[genre], filename))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, start, label, filename = self.items[idx]
        seg_samples = int(SEGMENT_DURATION * SR)
        audio, _ = librosa.load(path, sr=SR, offset=start / SR, duration=SEGMENT_DURATION)
        if len(audio) < seg_samples:
            audio = np.pad(audio, (0, seg_samples - len(audio)))
        inputs = self.feature_extractor(audio, sampling_rate=SR, return_tensors="pt")
        return {"input_values": inputs["input_values"].squeeze(0),
                "label": torch.tensor(label), "filename": filename}


def collate(batch):
    return (torch.stack([b["input_values"] for b in batch]),
            torch.stack([b["label"] for b in batch]),
            [b["filename"] for b in batch])


def song_probs(model, loader):
    model.eval()
    psum, ptrue, pcount = {}, {}, {}
    with torch.no_grad():
        for input_values, labels, filenames in loader:
            logits = model(input_values.to(device)).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for p, lbl, fn in zip(probs, labels.numpy(), filenames):
                if fn not in psum:
                    psum[fn] = np.zeros(len(GENRES))
                    pcount[fn] = 0
                    ptrue[fn] = lbl
                psum[fn] += p
                pcount[fn] += 1
    songs = sorted(ptrue.keys())
    return (np.array(songs),
            np.array([ptrue[s] for s in songs]),
            np.array([psum[s] / pcount[s] for s in songs]))


if __name__ == "__main__":
    fe = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME, num_labels=len(GENRES), ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    print("Loaded fine-tuned AST checkpoint.")

    out = {}
    for split in ["val", "test"]:
        ds = GTZANSegments(read_split_csv(os.path.join(SPLIT_DIR, f"{split}_songs.csv")), fe)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate, num_workers=2)
        songs, y, probs = song_probs(model, loader)
        acc = (np.argmax(probs, axis=1) == y).mean()
        print(f"{split}: {len(songs)} songs, song-level acc {acc:.4f}")
        out[f"{split}_songs"] = songs
        out[f"{split}_y"] = y
        out[f"{split}_probs"] = probs

    np.savez("ast_probs.npz", **out)
    print("\nSaved ast_probs.npz -- download it from the Files panel.")
