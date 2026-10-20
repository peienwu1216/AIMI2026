"""Train one required ResNet and evaluate the final operating point.

Run: python train.py --data-dir path/to/chest_xray --model resnet18
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm.auto import tqdm

from data import make_loaders, CLASS_NAMES, CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD
from models import build_model, set_frozen_batchnorm_eval
from metrics import summarize


def get_device(name='auto'):
    if name != 'auto':
        if name == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available')
        if name == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but not available')
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pass_epoch(model, loader, criterion, device, optimizer=None, desc='', freeze_bn=False,
               threshold=0.5):
    training = optimizer is not None
    if training:
        model.train()
        set_frozen_batchnorm_eval(model)
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
    else:
        model.eval()
    total_loss, n_total, y_true, y_pred = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            bs = len(labels)
            total_loss += float(loss.item()) * bs  # correct sample-weighted aggregation
            n_total += bs
            y_true.extend(labels.detach().cpu().tolist())
            pneumonia_probability = torch.softmax(logits, dim=1)[:, 1]
            y_pred.extend((pneumonia_probability >= threshold).long().detach().cpu().tolist())
    metrics = summarize(y_true, y_pred)
    metrics['loss'] = total_loss / n_total
    return metrics


@torch.inference_mode()
def predict_probabilities(model, loader, device, desc=''):
    model.eval()
    y_true, probabilities = [], []
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        prob = torch.softmax(logits, dim=1)[:, 1]
        y_true.extend(labels.tolist())
        probabilities.extend(prob.detach().cpu().tolist())
    return np.asarray(y_true, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def select_threshold(y_true, probabilities, metric='balanced_accuracy'):
    """Choose a decision threshold using validation predictions only."""
    if metric not in ('accuracy', 'balanced_accuracy', 'macro_f1', 'f1'):
        raise ValueError(f'Unsupported threshold metric: {metric}')
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 181), [0.5]]))
    best = None
    for threshold in candidates:
        result = summarize(y_true, probabilities >= threshold)
        rank = (result[metric], result['accuracy'], result['macro_f1'],
                -abs(float(threshold) - 0.5))
        if best is None or rank > best[0]:
            best = (rank, float(threshold), result)
    return best[1], best[2]


def save_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Lab1 train + validation + final test')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--model', choices=['resnet18','resnet50'], required=True)
    ap.add_argument('--output-dir', default='runs')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--label-smoothing', type=float, default=0.0)
    ap.add_argument('--class-weighting', choices=['balanced','none'], default='balanced',
                    help='balanced uses inverse-frequency CrossEntropy weights; none is an ablation')
    ap.add_argument('--normal-weight-multiplier', type=float, default=1.0,
                    help='Additional multiplier for NORMAL loss after balanced weighting')
    ap.add_argument('--head-lr-multiplier', type=float, default=1.0)
    ap.add_argument('--scheduler', choices=['none','plateau','cosine'], default='none')
    ap.add_argument('--selection-metric',
                    choices=['accuracy','balanced_accuracy','macro_f1','f1'],
                    default='f1')
    ap.add_argument('--threshold-metric',
                    choices=['accuracy','balanced_accuracy','macro_f1','f1'],
                    default='balanced_accuracy')
    ap.add_argument('--decision-threshold', type=float, default=None,
                    help='Fixed final operating point; omit to select it on validation')
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split-seed', type=int, default=None,
                    help='Keep the data split fixed while varying the training seed')
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    ap.add_argument('--trainable-stages', choices=['all','layer4','head'], default='all')
    ap.add_argument('--group-aware-split', action='store_true')
    ap.add_argument('--resize-mode', choices=['pad','stretch'], default='pad')
    ap.add_argument('--hflip-prob', type=float, default=0.0)
    ap.add_argument('--autocontrast-prob', type=float, default=0.0)
    ap.add_argument('--equalize-prob', type=float, default=0.0)
    ap.add_argument('--no-pretrained', action='store_true', help='For offline smoke tests only')
    ap.add_argument('--freeze-bn', action='store_true', help='Optionally keep all BatchNorm running stats fixed')
    ap.add_argument('--skip-test', action='store_true', help='For debugging only; final report requires test metrics')
    args = ap.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or not 0 < args.val_ratio < 1
            or not 0 <= args.dropout < 1 or not 0 <= args.label_smoothing < 1
            or args.head_lr_multiplier <= 0 or args.normal_weight_multiplier <= 0
            or (args.decision_threshold is not None
                and not 0 < args.decision_threshold < 1)
            or not all(0 <= p <= 1 for p in
                       (args.hflip_prob, args.autocontrast_prob, args.equalize_prob))):
        ap.error('epochs, batch-size must be positive and val-ratio must be (0,1)')
    out = Path(args.output_dir) / args.model
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    device = get_device(args.device)
    print(f'>> Model={args.model}, Device={device}, pretrained={not args.no_pretrained}', flush=True)
    loaders, counts, actual_root = make_loaders(
        args.data_dir, image_size=args.image_size, batch_size=args.batch_size,
        val_ratio=args.val_ratio, seed=split_seed, num_workers=args.num_workers,
        split_csv=out / 'split_manifest.csv', group_aware=args.group_aware_split,
        resize_mode=args.resize_mode, hflip_prob=args.hflip_prob,
        autocontrast_prob=args.autocontrast_prob, equalize_prob=args.equalize_prob)
    print(f'>> Class mapping: {CLASS_TO_IDX}; split counts: {counts}', flush=True)
    model = build_model(args.model, pretrained=not args.no_pretrained,
                        trainable_stages=args.trainable_stages,
                        dropout=args.dropout).to(device)
    count_train = counts['train']
    n = sum(count_train.values())
    weights = None
    if args.class_weighting == 'balanced':
        weights = torch.tensor([n/(2 * count_train[i]) for i in [0,1]],
                               dtype=torch.float32, device=device)
        weights[0] *= args.normal_weight_multiplier
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    head_parameters = [p for name, p in model.named_parameters()
                       if p.requires_grad and name.startswith('fc.')]
    backbone_parameters = [p for name, p in model.named_parameters()
                           if p.requires_grad and not name.startswith('fc.')]
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({'params':backbone_parameters, 'lr':args.lr})
    if head_parameters:
        parameter_groups.append({'params':head_parameters,
                                 'lr':args.lr * args.head_lr_multiplier})
    optimizer = AdamW(parameter_groups, weight_decay=args.weight_decay)
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                     patience=2, min_lr=1e-6)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None
    cfg = vars(args).copy()
    cfg.update(data_dir_resolved=str(actual_root), class_to_idx=CLASS_TO_IDX,
               image_normalization={'mean':IMAGENET_MEAN,'std':IMAGENET_STD},
               split_counts=counts,
               data_split_seed=split_seed,
               loss_class_weights=(weights.detach().cpu().tolist() if weights is not None else None),
               device_selected=str(device))
    save_json(out / 'config.json', cfg)
    history = []
    best_score, best_f1, best_acc, best_epoch, stale = -1, -1, -1, -1, 0
    start = time.perf_counter()
    ckpt_path = out / 'best.pt'
    for epoch in range(1, args.epochs+1):
        train_result = pass_epoch(model, loaders['train'], criterion, device, optimizer,
                                  desc=f'{args.model} train {epoch}/{args.epochs}', freeze_bn=args.freeze_bn)
        val_result = pass_epoch(model, loaders['val'], criterion, device,
                                desc=f'{args.model} valid {epoch}/{args.epochs}')
        history.append({'epoch':epoch, 'train':train_result, 'val':val_result,
                        'learning_rates':[group['lr'] for group in optimizer.param_groups]})
        save_json(out / 'history.json', history)
        print(f'Epoch {epoch:02d} | train ACC={train_result["accuracy"]:.3%} '
              f'F1={train_result["f1"]:.4f} | val ACC={val_result["accuracy"]:.3%} '
              f'F1={val_result["f1"]:.4f} BACC={val_result["balanced_accuracy"]:.4f}',
              flush=True)
        score = val_result[args.selection_metric]
        improved = score > best_score + 1e-8 or (
            abs(score-best_score)<1e-8 and val_result['accuracy'] > best_acc + 1e-8)
        if improved:
            best_score, best_f1, best_acc, best_epoch, stale = (
                score, val_result['f1'], val_result['accuracy'], epoch, 0)
            torch.save({'model_state_dict':model.state_dict(), 'model_name':args.model,
                        'class_to_idx':CLASS_TO_IDX, 'image_size':args.image_size,
                        'best_epoch':best_epoch, 'val_f1':best_f1,
                        'trainable_stages':args.trainable_stages,
                        'dropout':args.dropout, 'resize_mode':args.resize_mode}, ckpt_path)
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f'Early stopping after {epoch} epochs. Best epoch {best_epoch}.', flush=True)
                break
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_result['loss'])
        elif scheduler is not None:
            scheduler.step()
    elapsed = time.perf_counter()-start
    print(f'Best validation epoch: {best_epoch}; F1={best_f1:.4f}; '
          f'ACC={best_acc:.3%}; training time={elapsed/60:.1f} min', flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    y_val, p_val = predict_probabilities(
        model, loaders['val'], device, desc=f'{args.model} threshold calibration')
    if args.decision_threshold is None:
        threshold, threshold_validation = select_threshold(
            y_val, p_val, metric=args.threshold_metric)
        threshold_source = f'validation_{args.threshold_metric}'
    else:
        threshold = float(args.decision_threshold)
        threshold_validation = summarize(y_val, p_val >= threshold)
        threshold_source = 'fixed_operating_point'
    selection = {'best_epoch':best_epoch, 'selection_metric':args.selection_metric,
                 'best_selection_score':best_score, 'val_f1':best_f1,
                 'val_accuracy':best_acc, 'epochs_run':len(history),
                 'elapsed_seconds':elapsed, 'threshold':threshold,
                 'threshold_source':threshold_source,
                 'threshold_metric':args.threshold_metric,
                 'threshold_validation_metrics':threshold_validation}
    save_json(out / 'selection.json', selection)
    checkpoint['threshold'] = threshold
    torch.save(checkpoint, ckpt_path)
    print(f'Final decision threshold={threshold:.3f} [{threshold_source}] '
          f'(validation {args.threshold_metric}='
          f'{threshold_validation[args.threshold_metric]:.4f})',
          flush=True)
    if args.skip_test:
        return
    test_result = pass_epoch(model, loaders['test'], criterion, device,
                             desc=f'{args.model} FINAL TEST', threshold=threshold)
    test_result.update(model=args.model, best_validation_epoch=best_epoch,
                       decision_threshold=threshold)
    save_json(out / 'test_metrics.json', test_result)
    print(f'FINAL TEST [{args.model}] Accuracy={test_result["accuracy"]:.3%} '
          f'Precision={test_result["precision"]:.4f} '
              f'Recall={test_result["recall"]:.4f} Specificity={test_result["specificity"]:.4f} '
              f'F1={test_result["f1"]:.4f}', flush=True)
    print(f'Confusion matrix [[TN,FP],[FN,TP]] = {test_result["confusion_matrix"]}', flush=True)

if __name__ == '__main__':
    main()
