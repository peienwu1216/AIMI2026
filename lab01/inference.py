"""Evaluate a trained checkpoint on the official test split.

Run: python inference.py --data-dir ./chest_xray --checkpoint runs/resnet50/best.pt
 or: python inference.py --checkpoint runs/resnet50/best.pt --image path/to/xray.jpeg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import CLASS_NAMES, ChestXrayDataset, build_transforms, enumerate_samples
from metrics import summarize
from models import build_model
from train import get_device, predict_probabilities


def main(argv=None):
    ap = argparse.ArgumentParser(description='Lab1 test-set inference')
    ap.add_argument('--data-dir', help='Dataset root; evaluates the official test split')
    ap.add_argument('--image', help='Classify a single X-ray image instead')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    ap.add_argument('--output-json', default=None, help='Optionally save the test metrics')
    args = ap.parse_args(argv)
    if not (args.data_dir or args.image):
        ap.error('provide --data-dir or --image')

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model(checkpoint['model_name'], pretrained=False,
                        dropout=checkpoint.get('dropout', 0.0))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    transform = build_transforms(checkpoint['image_size'], False,
                                 checkpoint.get('resize_mode', 'pad'), 0.0)

    if args.image:
        image = transform(Image.open(args.image).convert('RGB')).unsqueeze(0).to(device)
        with torch.inference_mode():
            prob = torch.softmax(model(image), dim=1)[0].cpu().tolist()
        label = CLASS_NAMES[int(prob[1] >= args.threshold)]
        print(f'P(NORMAL)={prob[0]:.4f}  P(PNEUMONIA)={prob[1]:.4f}  -> {label}')
        return

    root = Path(args.data_dir).expanduser().resolve()
    if not (root / 'test').is_dir() and (root / 'chest_xray' / 'test').is_dir():
        root = root / 'chest_xray'
    test_ds = ChestXrayDataset(
        enumerate_samples(root / 'test'), transform)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    y_true, probabilities = predict_probabilities(model, loader, device, desc='test')
    result = summarize(y_true, probabilities >= args.threshold)

    config_path = Path(args.checkpoint).with_name('config.json')
    seed = json.loads(config_path.read_text())['seed'] if config_path.exists() else '-'
    rows = [
        ('Model', checkpoint['model_name']),
        ('Training seed', seed),
        ('Checkpoint epoch', checkpoint.get('best_epoch', '-')),
        ('Decision threshold', f'{args.threshold:.2f}'),
        ('Test samples', result['support']),
        ('Accuracy', f'{result["accuracy"]:.3%}'),
        ('Precision', f'{result["precision"]:.4f}'),
        ('Recall', f'{result["recall"]:.4f}'),
        ('Specificity', f'{result["specificity"]:.4f}'),
        ('F1-score', f'{result["f1"]:.4f}'),
        ('Confusion matrix', f'{result["confusion_matrix"]}  [[TN, FP], [FN, TP]]'),
    ]
    if args.output_json:
        result.update(model=checkpoint['model_name'], training_seed=seed,
                      best_validation_epoch=checkpoint.get('best_epoch'),
                      decision_threshold=args.threshold)
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('AIMI Lab1 | Pneumonia Classification | Test Result')
    for key, value in rows:
        print(f'{key:<19}: {value}')


if __name__ == '__main__':
    main()
