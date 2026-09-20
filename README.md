# Music Genre Classification on GTZAN

A comparative study of CNN, attention, capsule, transfer-learning and transformer
architectures for music genre classification, with leakage-safe evaluation.


## Results

All numbers are **song-level accuracy on a held-out test set**, using song-level
train/val/test splits (see Methodology).

| Model | Song-level accuracy |
|---|---|
| Baseline CNN | 76.51% |
| CNN + pitch/noise augmentation | 73.83% |
| Attention-CNN (CBAM) | 78.52% |
| YAMNet embeddings + MLP | 85.23% |
| OpenL3 embeddings + MLP | 85.23% |
| YAMNet, rich pooling (mean+max+std) | 87.00% |
| Weighted ensemble (4 models) | 88.59% |
| **Feature fusion (YAMNet + OpenL3, rich pooling)** | **89.93%** |
| **AST fine-tuned (Audio Spectrogram Transformer)** | **89.93%** |

## Key findings

**Transfer learning beat training from scratch by a wide margin.** Every
from-scratch architecture (CNN, Attention-CNN, CapsNet) plateaued below 79%.
Simply swapping to embeddings from models pretrained on AudioSet (2M+ clips)
jumped accuracy to 85%+. With only ~700 training songs, borrowing external
knowledge mattered more than architectural sophistication.

**Pooling strategy mattered as much as architecture.** Summarising each segment
with mean + max + std of its frame embeddings, rather than mean alone, added
~2 points (85.23% → 87.00%) at zero training cost.

**Naive ensembling can hurt.** Equal-weight averaging of four models scored
84.56% — *below* the best single model (85.23%), because weaker models diluted
stronger ones. Weighting each model by its validation accuracy fixed this
(88.59%).

**Feature fusion beat prediction ensembling.** Concatenating YAMNet and OpenL3
features into one classifier (89.93%) outperformed averaging their separate
predictions (88.59%), since the classifier could learn cross-feature
relationships rather than only combining final answers.

**Pitch-shift augmentation reduced accuracy** (76.51% → 73.83%), with rock and
country degrading most. Shifting pitch appears to disturb the tonal cues that
separate genres with overlapping instrumentation.

**Rock was the hardest genre across every architecture**, consistently confused
with country, blues and metal. Fine-tuned AST handled it best (0.93 F1 vs 0.84
for feature fusion).

## Methodology

### Leakage-safe splitting

Each 30-second GTZAN clip is chopped into shorter segments to increase training
data. Splitting those *segments* randomly would place segments of the same song
in both train and test — letting the model recognise the song rather than the
genre, and inflating accuracy.

This project splits at the **song level first** (70/15/15, stratified by genre),
then segments within each split. No song's audio ever crosses splits.

### Evaluation

- **Segment-level**: accuracy per short segment.
- **Song-level** (reported above): a song's segment probabilities are averaged,
  then argmaxed. Closer to real-world use, where the unit of interest is a whole
  track.

### Dataset ceiling

Prior research estimates a practical accuracy ceiling of roughly **94.5%** on
GTZAN, caused by mislabeled, duplicated and distorted tracks in the dataset
itself. Published results above this figure should be treated with caution —
several are plausibly affected by segment-level leakage.

## Setup

```bash
# Python 3.11 required (TensorFlow does not support 3.14)
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download [GTZAN from Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
and extract so the structure is:

```
GTZAN/genres_original/{blues,classical,...,rock}/*.wav
```

## Running

```bash
python 01_song_level_split.py        # create leakage-safe splits
python 02_feature_extraction.py      # mel-spectrograms
python 03_train_model.py             # baseline CNN

python 12_yamnet_rich_features.py    # YAMNet embeddings, rich pooling
python 13_train_yamnet_rich.py

python 14_openl3_rich_features.py    # OpenL3 embeddings, rich pooling
python 15_train_fusion.py            # best local model

python 11_weighted_ensemble.py       # weighted ensemble
```

`16_finetune_ast_colab.py` and `19_ast_v2_colab.py` require a GPU and are
intended for Google Colab. `18_final_ensemble.py` combines exported AST
predictions with local models.

## Notes

- GTZAN contains a known corrupt file (`jazz.00054.wav`), skipped automatically.
- Training ran on CPU (Intel Core Ultra 7); AST fine-tuning on a Colab T4 GPU.
- The dataset, extracted features and trained models are gitignored — all are
  regenerable by running the scripts.

## Credits

Dataset: GTZAN (Tzanetakis & Cook, 2002).
