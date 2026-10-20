# AIMI Lab1 - Pneumonia Classification

PyTorch implementation of NYCU AIMI Lab1. The project implements a custom
chest X-ray `Dataset`/`DataLoader`, fine-tunes **ResNet18** and **ResNet50**,
records Accuracy and F1-score every epoch, and reports final test confusion
matrices.

## Final results

The official Kaggle test split contains 624 images. All models use the
default decision threshold 0.5.

| Model | Training seed | Best epoch | Accuracy | F1-score | Confusion matrix `[[TN, FP], [FN, TP]]` |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet18 | 7 | 11 | 88.62% | 91.62% | `[[165, 69], [2, 388]]` |
| ResNet50 | 7 | 9 | **91.35%** | **93.45%** | `[[185, 49], [5, 385]]` |

Both models use training seed 7 and the same seed-42 data split. ResNet18 stopped
early at epoch 17 (best epoch 11). ResNet50 was trained for 9 epochs on an Apple
MPS laptop, and its best validation checkpoint is epoch 9.

## Files

```text
.
├── data.py         # custom Dataset, transforms, and reproducible split
├── models.py       # ResNet18/ResNet50 builders
├── metrics.py      # Accuracy, F1, specificity, and confusion matrix
├── train.py        # training, validation, checkpointing, final test
├── inference.py    # test-set evaluation or single-image prediction
├── run_all.py      # trains both models and draws all figures
└── compare.py      # accuracy/F1 curves and confusion matrices
```

The dataset, checkpoints, and generated figures are not committed; running the
commands below regenerates them locally.

## Setup

Python 3.11 was used.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
and extract it as:

```text
chest_xray/
├── train/NORMAL/
├── train/PNEUMONIA/
├── test/NORMAL/
└── test/PNEUMONIA/
```

The original 16-image Kaggle `val/` folder is not used. `data.py` creates a
seeded (42), stratified, filename-group-aware 15% validation split from
`train/`.

## Training

Settings: ImageNet pretrained weights with a re-initialized 2-class head,
full fine-tuning, 224 x 224 input, mild affine/intensity augmentation,
weighted cross entropy with label smoothing 0.05, AdamW (backbone LR 1e-4,
head LR 5e-4, weight decay 1e-4), dropout 0.3, frozen BatchNorm statistics,
ReduceLROnPlateau, early stopping patience 6, checkpoint selection by
validation balanced accuracy, training seed 7, data split seed 42.

ResNet18 (up to 20 epochs with early stopping):

```bash
python train.py --data-dir ./chest_xray --model resnet18 --output-dir runs \
  --seed 7 --split-seed 42 --epochs 20 --batch-size 16 --lr 1e-4 --patience 6 \
  --dropout 0.3 --label-smoothing 0.05 --head-lr-multiplier 5 --scheduler plateau \
  --selection-metric balanced_accuracy --resize-mode stretch --hflip-prob 0.1 \
  --group-aware-split --freeze-bn --decision-threshold 0.5
```

ResNet50 (9 epochs):

```bash
python train.py --data-dir ./chest_xray --model resnet50 --output-dir runs \
  --seed 7 --split-seed 42 --epochs 9 --batch-size 16 --lr 1e-4 --patience 6 \
  --dropout 0.3 --label-smoothing 0.05 --head-lr-multiplier 5 --scheduler plateau \
  --selection-metric balanced_accuracy --resize-mode stretch --hflip-prob 0.1 \
  --group-aware-split --freeze-bn --decision-threshold 0.5
```

Draw the accuracy/F1 curves and confusion matrices:

```bash
python compare.py --runs-dir runs --figures-dir figures
```

`run_all.py` trains both models with these settings (up to 20 epochs each, so ResNet50 is not limited to 9) and then draws the figures.

## Inference

Evaluate a checkpoint on the official 624-image test split:

```bash
python inference.py --data-dir ./chest_xray --checkpoint runs/resnet50/best.pt
```

Or classify one X-ray:

```bash
python inference.py --checkpoint runs/resnet50/best.pt \
  --image chest_xray/test/NORMAL/IM-0001-0001.jpeg
```

If the Kaggle archive contains an extra nested `chest_xray/` directory, use
`chest_xray/chest_xray/...` in the paths above.

This is a course benchmark on a single pediatric dataset, not a clinical
diagnostic system.
