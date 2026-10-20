"""Reproducible custom X-ray Dataset and data loaders.

Original Kaggle train set is stratified into training + validation.
Original Kaggle val/ is intentionally excluded because it is extremely small.
Original Kaggle test/ is untouched until final evaluation.
"""
from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES = ('NORMAL', 'PNEUMONIA')
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PadToSquare:
    """Preserve image aspect ratio instead of stretching thoracic anatomy."""
    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        side = max(w, h)
        left = (side - w) // 2
        top = (side - h) // 2
        return ImageOps.expand(image, border=(left, top, side-w-left, side-h-top), fill=(0, 0, 0))


class ChestXrayDataset(Dataset):
    """Custom Dataset: deterministic file paths, explicit labels, RGB handling."""
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = list(samples)
        self.transform = transform
        if not self.samples:
            raise ValueError('Dataset contains no images')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as im:
            image = im.convert('RGB')  # Grayscale chest X-rays -> 3 repeated channels
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


def enumerate_samples(split_dir: Path) -> list[tuple[Path, int]]:
    samples = []
    for name, label in CLASS_TO_IDX.items():
        folder = split_dir / name
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing class directory: {folder}')
        samples.extend((p, label) for p in sorted(folder.rglob('*'))
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not samples:
        raise ValueError(f'No image files found under {split_dir}')
    return samples


def infer_group_id(path: Path, label: int) -> str:
    """Best-effort patient/study group from the public dataset filenames."""
    stem = path.stem.upper()
    if label == CLASS_TO_IDX['PNEUMONIA']:
        match = re.search(r'PERSON(\d+)', stem)
        if match:
            return f'PNEUMONIA_PERSON_{match.group(1)}'
    else:
        match = re.search(r'(NORMAL2-IM|IM)-(\d+)', stem)
        if match:
            return f'NORMAL_{match.group(1)}_{match.group(2)}'
    return f'UNPARSED_{path.resolve()}'


def build_transforms(image_size: int, augment: bool, resize_mode='pad', hflip_prob=0.0,
                     autocontrast_prob=0.0, equalize_prob=0.0):
    if resize_mode not in ('pad', 'stretch'):
        raise ValueError('resize_mode must be pad or stretch')
    if not all(0 <= p <= 1 for p in (hflip_prob, autocontrast_prob, equalize_prob)):
        raise ValueError('augmentation probabilities must be in [0, 1]')
    ops = []
    if resize_mode == 'pad':
        ops.append(PadToSquare())
    ops.append(transforms.Resize((image_size, image_size)))
    if augment:
        # Small geometric/intensity changes only; never use unrealistic 90-degree rotations.
        ops.extend([
            transforms.RandomHorizontalFlip(p=hflip_prob),
            transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=0),
            transforms.RandomAutocontrast(p=autocontrast_prob),
            transforms.RandomEqualize(p=equalize_prob),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
        ])
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


def make_loaders(data_root: str | Path, image_size=224, batch_size=16,
                 val_ratio=0.15, seed=42, num_workers=0, split_csv=None,
                 group_aware=False, resize_mode='pad', hflip_prob=0.0,
                 autocontrast_prob=0.0, equalize_prob=0.0):
    root = Path(data_root).expanduser().resolve()
    # Accept either /path/chest_xray or /path/chest_xray/chest_xray
    if not (root / 'train').is_dir() and (root / 'chest_xray' / 'train').is_dir():
        root = root / 'chest_xray'
    train_all = enumerate_samples(root / 'train')
    test_samples = enumerate_samples(root / 'test')
    labels = [label for _, label in train_all]
    indices = np.arange(len(train_all))
    groups = np.asarray([infer_group_id(path, label) for path, label in train_all])
    if group_aware:
        # n_splits=7 is close to a 15% holdout. Select the fold whose size and
        # class ratio most closely match the requested validation set.
        n_splits = max(2, round(1 / val_ratio))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        prevalence = float(np.mean(labels))
        candidates = []
        for candidate_train, candidate_val in splitter.split(indices, labels, groups):
            ratio_gap = abs(len(candidate_val) / len(indices) - val_ratio)
            prevalence_gap = abs(float(np.mean(np.asarray(labels)[candidate_val])) - prevalence)
            candidates.append((ratio_gap + prevalence_gap, candidate_train, candidate_val))
        _, train_idx, val_idx = min(candidates, key=lambda item: item[0])
    else:
        train_idx, val_idx = train_test_split(indices, test_size=val_ratio,
                                             random_state=seed, stratify=labels)
    train_samples = [train_all[int(i)] for i in sorted(train_idx)]
    val_samples = [train_all[int(i)] for i in sorted(val_idx)]
    train_paths = {str(p.resolve()) for p, _ in train_samples}
    val_paths = {str(p.resolve()) for p, _ in val_samples}
    test_paths = {str(p.resolve()) for p, _ in test_samples}
    assert not (train_paths & val_paths or train_paths & test_paths or val_paths & test_paths)
    if group_aware:
        train_groups = {infer_group_id(p, y) for p, y in train_samples}
        val_groups = {infer_group_id(p, y) for p, y in val_samples}
        assert not train_groups & val_groups

    if split_csv:
        dest = Path(split_csv)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('w', newline='', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            writer.writerow(['split', 'path_relative_to_root', 'label', 'group_id'])
            for split_name, collection in [('train', train_samples), ('validation', val_samples), ('test', test_samples)]:
                for path, label in collection:
                    writer.writerow([split_name, path.relative_to(root).as_posix(), CLASS_NAMES[label],
                                     infer_group_id(path, label)])
    train_ds = ChestXrayDataset(
        train_samples, build_transforms(image_size, True, resize_mode, hflip_prob,
                                        autocontrast_prob, equalize_prob))
    val_ds = ChestXrayDataset(
        val_samples, build_transforms(image_size, False, resize_mode, 0.0))
    test_ds = ChestXrayDataset(
        test_samples, build_transforms(image_size, False, resize_mode, 0.0))
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available()),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=torch.cuda.is_available()),
    }
    counts = {key: dict(Counter(y for _, y in ds.samples)) for key, ds in
              [('train', train_ds), ('val', val_ds), ('test', test_ds)]}
    return loaders, counts, root
