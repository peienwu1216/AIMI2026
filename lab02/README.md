# Lab 2: EEG Classification

Two-class motor imagery classification on BCI Competition III dataset IIIb
(subjects S4 and X11, channels C3 and C4, 125 Hz). EEGNet is the main model, and
DeepConvNet and ShallowConvNet are the bonus models.

## Results

| Model | Parameters | Epochs | Last-10 test acc. (%) | Peak test acc. (%) | Best run (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 17,874 | 500 | 87.55 ± 0.57 | 88.61 ± 0.40 | **89.26** |
| ShallowConvNet | 11,922 | 150 | 82.29 ± 0.23 | 83.98 ± 0.23 | 84.26 |
| DeepConvNet | 150,977 | 150 | 80.01 ± 0.33 | 81.57 ± 0.74 | 82.59 |

Values are mean ± SD over 5 seeds for EEGNet and 3 seeds for the bonus models.
Peak accuracy selects the epoch on the test set, following the course definition.
Last-10 is the mean over the final 10 epochs, with no selection.

## Setup

Requires Python 3.11.

```bash
pip install -r requirements.txt
```

For Linux with CUDA 12.6, use `requirements-workstation.txt` instead.

The dataset is course material and is not included. Copy these four files into
`data/`, then check the environment:

```text
data/S4b_train.npz    data/S4b_test.npz
data/X11b_train.npz   data/X11b_test.npz
```

```bash
python check_setup.py
```

The last line should be `READY`.

## Reproduce the best result

This reloads the saved checkpoint and scores all 1,080 test trials.

```bash
python main.py --eval-checkpoint results/raw/final500-leaky-relu-s113/test_best.pt --device cpu
```

Expected output: `TEST_ACCURACY: 89.26%`.

## Train

Final EEGNet recipe:

```bash
python main.py --activation leaky_relu --dropout 0.5 --lr 1e-3 --num-epochs 500 \
  --val-ratio 0 --select-by test_accuracy --seed 113
```

Use `--model deepconvnet` or `--model shallowconvnet` for the bonus models.
Run `python main.py --help` for all options. Each run writes its configuration,
history, curves, checkpoints and predictions to `results/raw/<run_id>/`.

## Experiments

| Path | Purpose |
| --- | --- |
| `sweeps/` | Manifests and runner for the activation, ELU alpha, dropout and optimization sweeps |
| `experiments/run_main_grid.py` | Architecture ablations and bonus models |
| `experiments/local_experiments.py` | Subject scope, normalization, augmentation and time windows |
| `analysis/` | Classical baselines, statistics and report figures |
| `results/summary/` | Metrics of all 540 runs |

```bash
python sweeps/run_manifest.py sweeps/p1_core_matrix.json --device cuda
```

Raw run folders are not tracked, so `analysis/report_figures.py` needs a new set of
runs. The figures used in the report are in `figures/report/`.

## Code layout

| Path | Contents |
| --- | --- |
| `models/EEGNet.py` | EEGNet, its true separable variant, and DeepConvNet |
| `models/ShallowConvNet.py` | ShallowConvNet |
| `main.py` | Training, evaluation and checkpoint reloading |
| `dataloader.py` | Course data loader |
| `tests/` | Unit tests, run with `python -m pytest -q` |

## References

- V. J. Lawhern et al., "EEGNet: a compact convolutional neural network for EEG-based
  brain–computer interfaces," *J. Neural Eng.*, 2018.
- R. T. Schirrmeister et al., "Deep learning with convolutional neural networks for
  EEG decoding and visualization," *Hum. Brain Mapp.*, 2017.
