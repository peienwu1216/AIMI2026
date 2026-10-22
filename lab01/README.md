# AIMI Lab1: Pneumonia Classification

Binary classification (NORMAL / PNEUMONIA) of chest X-rays with ResNet18 and
ResNet50, using a custom PyTorch `Dataset` and `DataLoader`.

## Results

Official Kaggle test set (624 images), threshold 0.5.

| Model | Best epoch | Accuracy | F1-score | Confusion matrix `[[TN, FP], [FN, TP]]` |
| --- | ---: | ---: | ---: | --- |
| ResNet18 | 11 | 88.62% | 91.62% | `[[165, 69], [2, 388]]` |
| ResNet50 | 9 | **91.35%** | **93.45%** | `[[185, 49], [5, 385]]` |

Both models use training seed 7. ResNet18 early-stopped at epoch 17.
ResNet50 was trained for 9 epochs on an Apple MPS laptop.

## Files

| File | Purpose |
| --- | --- |
| `data.py` | Dataset, augmentation, train/validation split |
| `models.py` | ResNet18 / ResNet50 with a new 2-class head |
| `metrics.py` | Accuracy, F1, specificity, confusion matrix |
| `train.py` | Training, validation, checkpoint, test |
| `inference.py` | Test-set evaluation or single-image prediction |
| `compare.py` | Accuracy/F1 curves and confusion matrices |
| `run_all.py` | Trains both models, then draws the figures |

## Setup

Python 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the [Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
and place it at `./chest_xray` with `train/` and `test/` inside. The 16-image
Kaggle `val/` folder is not used. `data.py` splits 15% of `train/` into a
validation set (seed 42), keeping images with the same filename-based patient
ID on the same side.

## Training

ImageNet-pretrained, full fine-tuning, 224 x 224 input, weighted cross entropy
with label smoothing 0.05, AdamW (LR 1e-4 backbone, 5e-4 head), dropout 0.3,
early stopping patience 6. The checkpoint with the best validation balanced
accuracy is kept.

```bash
# ResNet18
python train.py --data-dir ./chest_xray --model resnet18 --epochs 20 \
  --seed 7 --split-seed 42 --batch-size 16 --lr 1e-4 --patience 6 \
  --dropout 0.3 --label-smoothing 0.05 --head-lr-multiplier 5 --scheduler plateau \
  --selection-metric balanced_accuracy --resize-mode stretch --hflip-prob 0.1 \
  --group-aware-split --freeze-bn --decision-threshold 0.5

# ResNet50: same command with --model resnet50 --epochs 9

# Curves and confusion matrices
python compare.py --runs-dir runs --figures-dir figures
```

## Inference

```bash
# Whole test set
python inference.py --data-dir ./chest_xray --checkpoint runs/resnet50/best.pt

# One image
python inference.py --checkpoint runs/resnet50/best.pt --image path/to/xray.jpeg
```
