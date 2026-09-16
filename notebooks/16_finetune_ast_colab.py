"""
16_finetune_ast_colab.py

Fine-tunes AST (Audio Spectrogram Transformer) on GTZAN. Run on Colab GPU.

WHY THIS IS DIFFERENT FROM EVERYTHING SO FAR:
Our YAMNet/OpenL3 work used FROZEN pretrained models -- we ran audio
through them and trained a small classifier on the output. The model
itself never changed; it stayed a general-purpose audio model.

Fine-tuning UNFREEZES the pretrained model and continues training it on
GTZAN, so its internal representations actually adapt to music genre
specifically. This is what the published 90%+ GTZAN results did, and
it's the main technique we hadn't tried. It needs a GPU because we're
now updating ~87 million parameters instead of ~1 million.

AST is pretrained on AudioSet (2M+ clips) and is a transformer rather
than a CNN -- so this also gives your paper a genuinely different
architecture family to report alongside CNN/CapsNet/attention.

LEAKAGE SAFETY: reads the exact same song-level split CSVs from Step 2.
Songs never cross between train/val/test. Same rigor as every other
script in this project.

SETUP IN COLAB -- run these in a cell FIRST:
    !pip install -q transformers datasets torchaudio kaggle

Then upload via the Files panel:
    - kaggle.json        (from kaggle.com -> Settings -> Create New Token)
    - train_songs.csv, val_songs.csv, test_songs.csv   (from your splits/ folder)

Then run this script. It downloads GTZAN itself -- you never upload the
1.2GB dataset.
"""

import os
import csv
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import classification_report, confusion_matrix

# ---- CONFIG ----
SPLIT_DIR = "."                       # where the CSVs were uploaded
DATA_DIR = "GTZAN/Data/genres_original"
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000                            # AST expects 16kHz
SEGMENT_DURATION = 5.0                # longer than 3s -- AST handles more context well
BATCH_SIZE = 8                        # keep modest; AST is memory-hungry on a T4
EPOCHS = 6                            # fine-tuning converges fast; more risks overfitting
LEARNING_RATE = 5e-5                  # small LR -- we're nudging a pretrained model, not training fresh

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    print("WARNING: no GPU detected. In Colab: Runtime -> Change runtime type -> T4 GPU.")


# ---------------------------------------------------------------------
# Step 1: download GTZAN directly inside Colab (no 1.2GB upload)
# ---------------------------------------------------------------------
def download_gtzan():
    if os.path.exists(DATA_DIR):
        print("GTZAN already present, skipping download.")
        return

    print("Downloading GTZAN from Kaggle...")
    os.makedirs("/root/.kaggle", exist_ok=True)
    os.system("cp kaggle.json /root/.kaggle/kaggle.json")
    os.system("chmod 600 /root/.kaggle/kaggle.json")
    os.system("kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification -p GTZAN --unzip")
    print("Download complete.")


# ---------------------------------------------------------------------
# Step 2: dataset that reads the SAME song-level splits
# ---------------------------------------------------------------------
def read_split_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["genre"], row["filename"]))
    return rows


class GTZANSegments(Dataset):
    """Chops each song into fixed-length segments. Because the song-level
    split already happened upstream, every segment of a given song stays
    entirely within one split -- no leakage."""

    def __init__(self, split_rows, feature_extractor):
        self.feature_extractor = feature_extractor
        self.items = []   # (filepath, start_sample, label, filename)

        seg_samples = int(SEGMENT_DURATION * SR)
        for genre, filename in split_rows:
            path = os.path.join(DATA_DIR, genre, filename)
            if not os.path.exists(path):
                continue
            try:
                duration = librosa.get_duration(path=path)
            except Exception as e:
                print(f"  [skipped] {filename}: {e}")
                continue

            n_segments = int(duration * SR) // seg_samples
            for i in range(n_segments):
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
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "label": torch.tensor(label),
            "filename": filename,
        }


def collate(batch):
    return (
        torch.stack([b["input_values"] for b in batch]),
        torch.stack([b["label"] for b in batch]),
        [b["filename"] for b in batch],
    )


# ---------------------------------------------------------------------
# Step 3: song-level evaluation (same approach as all other scripts)
# ---------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    song_probs, song_true = {}, {}

    with torch.no_grad():
        for input_values, labels, filenames in loader:
            input_values = input_values.to(device)
            logits = model(input_values).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for p, lbl, fn in zip(probs, labels.numpy(), filenames):
                if fn not in song_probs:
                    song_probs[fn] = np.zeros(len(GENRES))
                    song_true[fn] = lbl
                song_probs[fn] += p

    songs = list(song_true.keys())
    y_true = np.array([song_true[s] for s in songs])
    y_pred = np.array([np.argmax(song_probs[s]) for s in songs])
    return y_true, y_pred


if __name__ == "__main__":
    download_gtzan()

    print("\nLoading AST feature extractor + pretrained model...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(GENRES),
        ignore_mismatched_sizes=True,   # replaces AudioSet's 527-class head with our 10-class one
    ).to(device)

    print("\nBuilding datasets from song-level splits...")
    train_ds = GTZANSegments(read_split_csv(os.path.join(SPLIT_DIR, "train_songs.csv")), feature_extractor)
    val_ds = GTZANSegments(read_split_csv(os.path.join(SPLIT_DIR, "val_songs.csv")), feature_extractor)
    test_ds = GTZANSegments(read_split_csv(os.path.join(SPLIT_DIR, "test_songs.csv")), feature_extractor)
    print(f"Segments -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, None

    for epoch in range(EPOCHS):
        model.train()
        running_loss, seen = 0.0, 0

        for step, (input_values, labels, _) in enumerate(train_loader):
            input_values, labels = input_values.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(input_values).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * len(labels)
            seen += len(labels)

            if step % 50 == 0:
                print(f"  epoch {epoch+1}/{EPOCHS}  step {step}/{len(train_loader)}  loss {running_loss/seen:.4f}")

        y_true_val, y_pred_val = evaluate(model, val_loader)
        val_acc = (y_true_val == y_pred_val).mean()
        print(f"Epoch {epoch+1}: train loss {running_loss/seen:.4f} | val song-level acc {val_acc:.4f}")

        # Keep the best epoch by VALIDATION accuracy -- test set never
        # influences model selection.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  new best (val {val_acc:.4f}) -- checkpointed")

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nEvaluating best checkpoint on TEST set...")
    y_true, y_pred = evaluate(model, test_loader)
    acc = (y_true == y_pred).mean()

    print(f"\nAST song-level test accuracy: {acc:.4f}  <- headline result")
    print(classification_report(y_true, y_pred, target_names=GENRES))
    print(confusion_matrix(y_true, y_pred))

    torch.save(model.state_dict(), "ast_gtzan.pt")
    print("\nSaved ast_gtzan.pt -- download it from the Files panel to keep it.")
