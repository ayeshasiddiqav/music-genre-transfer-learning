"""
19_ast_v2_colab.py   -- run in COLAB with T4 GPU.

Optimized AST fine-tuning. Five concrete fixes over script 16:

1. FULL-LENGTH CONTEXT (10s instead of 5s)
   AST's pretrained input is 1024 spectrogram frames (~10.24s). Feeding
   it 5s clips meant ~half the input was silent padding -- the model was
   running at roughly half capacity. 10s segments fill the window.

2. SpecAugment
   Randomly masks bands of time and frequency in the spectrogram during
   training. Forces the model to not depend on any single cue, which is
   the standard regularizer for audio transformers. Applied ONLY to
   training batches.

3. MIXUP
   Blends pairs of training examples (and their labels) together. On a
   small dataset like GTZAN this is one of the most effective ways to
   stop a 87M-parameter model from memorizing 700 songs.

4. TEST-TIME AUGMENTATION (overlapping windows)
   At evaluation, segments overlap by 50% instead of being disjoint, so
   each song gets ~2x more views to average over. Costs nothing in
   training and reliably stabilizes song-level predictions.

5. MULTI-SEED ENSEMBLING
   Transformer fine-tuning genuinely varies 1-3% run to run just from
   random seed. Training N models and averaging their probabilities
   removes that luck factor instead of hoping for a good draw.

Plus: more epochs (12) with cosine schedule and warmup, and layer-wise
learning rate decay (earlier layers get smaller updates, since generic
low-level audio features transfer better than high-level ones).

LEAKAGE SAFETY: unchanged. Same song-level split CSVs. Overlapping
windows only ever overlap WITHIN one song, which already sits entirely
in one split. Model selection uses validation only.

SETUP (Colab cell, before running):
    !pip install -q transformers torchaudio kaggle
Upload: kaggle.json, train/val/test_songs.csv, this file.

Runtime: ~45 min per seed on a T4. NUM_SEEDS=3 -> ~2.5 hours.
Set NUM_SEEDS=1 first if you want a quick read on whether the other
fixes helped before committing to the full run.
"""

import os
import csv
import math
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import classification_report, confusion_matrix

# ---- CONFIG ----
SPLIT_DIR = "."
DATA_DIR = "GTZAN/Data/genres_original"
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000

SEGMENT_DURATION = 10.0     # matches AST's ~10.24s pretrained window
EVAL_HOP = 5.0              # 50% overlap at eval time (TTA)
BATCH_SIZE = 8              # 10s segments use more memory than 5s
ACCUM_STEPS = 1             # effective batch = 8, without the memory cost
EPOCHS = 8
BASE_LR = 5e-5
WARMUP_FRAC = 0.1
LLRD = 0.9                  # layer-wise lr decay factor
MIXUP_ALPHA = 0.3
SPECAUG_TIME_MASKS, SPECAUG_TIME_WIDTH = 2, 80
SPECAUG_FREQ_MASKS, SPECAUG_FREQ_WIDTH = 2, 16
NUM_SEEDS = 1

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    print("WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU.")


def download_gtzan():
    if os.path.exists(DATA_DIR):
        print("GTZAN already present.")
        return
    print("Downloading GTZAN...")
    os.makedirs("/root/.kaggle", exist_ok=True)
    os.system("cp kaggle.json /root/.kaggle/kaggle.json && chmod 600 /root/.kaggle/kaggle.json")
    os.system("kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification -p GTZAN --unzip")


def read_split_csv(path):
    with open(path, newline="") as f:
        return [(r["genre"], r["filename"]) for r in csv.DictReader(f)]


class GTZANSegments(Dataset):
    """hop_duration < SEGMENT_DURATION gives overlapping windows (used at
    eval time for TTA). Overlap is always within a single song, and that
    song lives entirely in one split, so no leakage is introduced."""

    def __init__(self, split_rows, feature_extractor, hop_duration=None):
        self.fe = feature_extractor
        self.items = []
        hop = hop_duration if hop_duration is not None else SEGMENT_DURATION
        hop_samples = int(hop * SR)
        seg_samples = int(SEGMENT_DURATION * SR)

        for genre, filename in split_rows:
            path = os.path.join(DATA_DIR, genre, filename)
            if not os.path.exists(path):
                continue
            try:
                total = int(librosa.get_duration(path=path) * SR)
            except Exception as e:
                print(f"  [skipped] {filename}: {e}")
                continue

            start = 0
            while start + seg_samples <= total:
                self.items.append((path, start, GENRE_TO_IDX[genre], filename))
                start += hop_samples
            if not self.items or self.items[-1][0] != path:
                self.items.append((path, 0, GENRE_TO_IDX[genre], filename))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, start, label, filename = self.items[idx]
        seg_samples = int(SEGMENT_DURATION * SR)
        audio, _ = librosa.load(path, sr=SR, offset=start / SR, duration=SEGMENT_DURATION)
        if len(audio) < seg_samples:
            audio = np.pad(audio, (0, seg_samples - len(audio)))
        x = self.fe(audio, sampling_rate=SR, return_tensors="pt")["input_values"].squeeze(0)
        return {"x": x, "y": torch.tensor(label), "fn": filename}


def collate(batch):
    return (torch.stack([b["x"] for b in batch]),
            torch.stack([b["y"] for b in batch]),
            [b["fn"] for b in batch])


def spec_augment(x):
    """x: (B, time_frames, freq_bins). Masks random time and frequency bands."""
    x = x.clone()
    B, T, Fq = x.shape
    for b in range(B):
        for _ in range(SPECAUG_TIME_MASKS):
            w = np.random.randint(0, SPECAUG_TIME_WIDTH)
            if w and T > w:
                t0 = np.random.randint(0, T - w)
                x[b, t0:t0 + w, :] = 0
        for _ in range(SPECAUG_FREQ_MASKS):
            w = np.random.randint(0, SPECAUG_FREQ_WIDTH)
            if w and Fq > w:
                f0 = np.random.randint(0, Fq - w)
                x[b, :, f0:f0 + w] = 0
    return x


def mixup(x, y, alpha):
    """Blends pairs of examples. Returns mixed input and both label sets
    plus the blend ratio, so loss can be computed against both."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def build_optimizer(model):
    """Layer-wise LR decay: deeper layers get closer to BASE_LR, early
    layers get progressively smaller LRs, since low-level audio features
    transfer well and shouldn't be disturbed much."""
    encoder_layers = model.audio_spectrogram_transformer.encoder.layer
    n_layers = len(encoder_layers)
    groups = []

    groups.append({"params": model.audio_spectrogram_transformer.embeddings.parameters(),
                   "lr": BASE_LR * (LLRD ** (n_layers + 1))})
    for i, layer in enumerate(encoder_layers):
        groups.append({"params": layer.parameters(), "lr": BASE_LR * (LLRD ** (n_layers - i))})
    groups.append({"params": model.audio_spectrogram_transformer.layernorm.parameters(), "lr": BASE_LR})
    groups.append({"params": model.classifier.parameters(), "lr": BASE_LR * 5})  # fresh head, larger LR

    return torch.optim.AdamW(groups, weight_decay=0.05)


@torch.no_grad()
def song_probs(model, loader):
    model.eval()
    psum, pcount, ptrue = {}, {}, {}
    for x, y, fns in loader:
        probs = torch.softmax(model(x.to(device)).logits, dim=-1).float().cpu().numpy()
        for p, lbl, fn in zip(probs, y.numpy(), fns):
            if fn not in psum:
                psum[fn] = np.zeros(len(GENRES))
                pcount[fn] = 0
                ptrue[fn] = int(lbl)
            psum[fn] += p
            pcount[fn] += 1
    songs = sorted(ptrue)
    return (np.array(songs),
            np.array([ptrue[s] for s in songs]),
            np.array([psum[s] / pcount[s] for s in songs]))


def train_one_seed(seed, fe, train_ds, val_loader, test_loader):
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME, num_labels=len(GENRES), ignore_mismatched_sizes=True
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=2, drop_last=True)

    optimizer = build_optimizer(model)
    steps_per_epoch = len(train_loader) // ACCUM_STEPS
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val, best_state = 0.0, None

    for epoch in range(EPOCHS):
        model.train()
        running, seen = 0.0, 0
        optimizer.zero_grad()

        for step, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x = spec_augment(x)
            x, y_a, y_b, lam = mixup(x, y, MIXUP_ALPHA)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x).logits
                loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running += loss.item() * ACCUM_STEPS * len(y)
            seen += len(y)
            if step % 100 == 0:
                print(f"  epoch {epoch+1}/{EPOCHS}  step {step}/{len(train_loader)}  loss {running/seen:.4f}")

        _, y_val, p_val = song_probs(model, val_loader)
        val_acc = (np.argmax(p_val, axis=1) == y_val).mean()
        print(f"Epoch {epoch+1}: loss {running/seen:.4f} | val song acc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  new best (val {val_acc:.4f})")

    model.load_state_dict(best_state)
    songs, y_test, p_test = song_probs(model, test_loader)
    acc = (np.argmax(p_test, axis=1) == y_test).mean()
    print(f"Seed {seed}: val {best_val:.4f} | test {acc:.4f}")

    _, y_v, p_v = song_probs(model, val_loader)
    del model
    torch.cuda.empty_cache()
    return {"songs": songs, "y": y_test, "probs": p_test,
            "val_y": y_v, "val_probs": p_v, "val_acc": best_val}


if __name__ == "__main__":
    download_gtzan()

    fe = ASTFeatureExtractor.from_pretrained(MODEL_NAME)

    train_rows = read_split_csv(os.path.join(SPLIT_DIR, "train_songs.csv"))
    val_rows = read_split_csv(os.path.join(SPLIT_DIR, "val_songs.csv"))
    test_rows = read_split_csv(os.path.join(SPLIT_DIR, "test_songs.csv"))

    train_ds = GTZANSegments(train_rows, fe)                      # disjoint windows
    val_ds = GTZANSegments(val_rows, fe, hop_duration=EVAL_HOP)   # overlapping (TTA)
    test_ds = GTZANSegments(test_rows, fe, hop_duration=EVAL_HOP)
    print(f"Segments -> train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate, num_workers=2)

    results = [train_one_seed(s, fe, train_ds, val_loader, test_loader) for s in range(NUM_SEEDS)]

    # ---- Average probabilities across seeds ----
    songs = results[0]["songs"]
    y_true = results[0]["y"]
    avg = np.mean([r["probs"] for r in results], axis=0)
    y_pred = np.argmax(avg, axis=1)
    acc = (y_pred == y_true).mean()

    print(f"\n{'='*60}")
    for i, r in enumerate(results):
        print(f"Seed {i}: test {(np.argmax(r['probs'],axis=1)==r['y']).mean():.4f}")
    print(f"\nMULTI-SEED AST song-level test accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=GENRES))
    print(confusion_matrix(y_true, y_pred))

    np.savez("ast_v2_probs.npz",
             test_songs=songs, test_y=y_true, test_probs=avg,
             val_songs=results[0]["songs"], val_y=results[0]["val_y"],
             val_probs=np.mean([r["val_probs"] for r in results], axis=0))
    print("\nSaved ast_v2_probs.npz -- download it and ensemble with your fusion model.")
