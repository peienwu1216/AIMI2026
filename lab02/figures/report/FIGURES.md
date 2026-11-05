# Report figures

IEEE two-column style: Times New Roman embedded as TrueType (no Type 3 fonts), 8 pt
minimum, 3.5 in single column and 7.16 in double column. Use the PDF files at 100%
scale. Do not resize them in the report. Regenerate with:

```bash
python project/analysis/report_figures.py
```

Error bars are mean ± population SD unless a caption says otherwise. Accuracy axes
on dot plots do not start at zero, so each caption states the plotted range.

## Coverage of the spec

| Spec item | Requirement | Figure |
| --- | --- | --- |
| Requirement 2, Report 2.c.i | Train/test loss and accuracy curves (**necessary**) | Fig. 2 |
| Report 2.b | Highest testing accuracy **screenshot** | `highest_accuracy_screenshot.png` |
| Requirement 3, Report 2.a.i | Compare activation functions | Fig. 4 |
| Requirement 3, Report 2.a.i | Compare ELU alpha values | Fig. 5 |
| Requirement 3, Report 2.a.i | Compare dropout rates | Fig. 6 |
| Requirement 3, Report 2.a.i | Other settings | Figs. 7, 8, 9 |
| Report 3.b | Explain how the model is implemented | Fig. 1 |
| Report 3.c | Explain the ELU function | Fig. 10 |
| Report 3.d | Explain depthwise and separable convolution | Fig. 11 |
| Report 2.d, 3.a | Anything else worth presenting | Figs. 3, 12, 13, 14, 15, 16 |
| Report 5, Bonus | Accuracy of DeepConvNet or another model | Figs. 17, 18 |

The spec's plotting page links `seaborn.heatmap`, so a confusion-matrix heatmap
(Fig. 3) is included.

## Screenshot

`highest_accuracy_screenshot.png` is a real terminal screenshot, taken in VS Code at
21:51 on 2026-11-05. It reloads the saved checkpoint and scores all 1,080 test
trials. To reproduce it:

```bash
cd lab02/project
../_env/.venv/bin/python main.py --eval-checkpoint results/raw/final500-leaky-relu-s113/test_best.pt --data-dir ../_env/data --device cpu --batch-size 64
```

Caption: Highest test accuracy, reloaded from the saved checkpoint: 89.26% on 1,080
trials at epoch 484. The epoch was chosen on the test set, following the course
definition.

## Captions

**Fig. 1** (`fig01_eegnet_architecture`, double column). EEGNet as specified in the
course, with output tensor shapes and parameter counts per block. The block named
`separableConv` in the spec is an ordinary convolution (see Fig. 11).

**Fig. 2** (`fig02_training_curves`, double column). Training curves of the best
EEGNet (LeakyReLU, dropout 0.5, lr 10⁻³, 500 epochs, seed 113). Lines show this run.
Shaded bands show the mean ± SD of 5 seeds. The peak is chosen on the test set, so
it is optimistic.

**Fig. 3** (`fig03_confusion_matrix`, single column). Test confusion matrices of the
best run at its peak epoch, for all trials and for each subject. Cells show counts
and row percentages.

**Fig. 4** (`fig04_activation`, single column). Seven activation functions at the
spec and tuned baselines (5 seeds, 20% validation split). The two baselines differ
in both dropout and learning rate. Open markers are seeds. Axis range 70–90%.

**Fig. 5** (`fig05_elu_alpha_and_slope`, double column). (a) ELU α and (b) LeakyReLU
negative slope at the tuned baseline (5 seeds). Axis range 70–90%.

**Fig. 6** (`fig06_dropout`, single column). Dropout rate at the tuned baseline
(5 seeds). Train accuracy is measured with dropout active, which lowers it. Both
curves are means of the last 10 epochs. Axis range 75–101%.

**Fig. 7** (`fig07_optimization`, double column). (a) Learning rate, (b) batch size,
(c) weight decay and (d) optimizer at the tuned baseline (3 seeds). Axis range
70–90%.

**Fig. 8** (`fig08_training_duration`, single column). Effect of training length
with all 1,080 training trials (5 seeds). Peak is chosen on the test set. Last-10 is
the mean of the final 10 epochs. Markers are offset horizontally for legibility.
Axis range 85–90%.

**Fig. 9** (`fig09_architecture_ablation`, double column). (a) Temporal kernel
length, (b) second-block kernel length, (c) width and (d) block type, trained on all
trials and on an 80% split (3 seeds). Axis range 80–90%.

**Fig. 10** (`fig10_elu_function`, single column). ELU and its derivative. The
derivative is discontinuous at zero unless α = 1.

**Fig. 11** (`fig11_convolution_types`, double column). Standard, depthwise and
depthwise-separable convolution, drawn with four channels. Numbers are the weights
in this model.

**Fig. 12** (`fig12_negative_floor`, single column). Last-10 accuracy against the
lowest output of each activation at the tuned baseline (5 seeds). This pattern was
found after inspecting the results and shows a correlation only. Axis range 70–90%.

**Fig. 13** (`fig13_subject_scope`, single column). One shared model against one
model per subject. EEGNet values are last-10 accuracy over 5 seeds. Classical
pipelines are single fits tuned by cross-validation on the training set.

**Fig. 14** (`fig14_normalization_augmentation`, double column). Paired change in
last-10 accuracy against the same seeds without the change (5 seeds for
normalization and the swap, 3 for the others). Shifts are circular.

**Fig. 15** (`fig15_time_window`, double column). (a) EEGNet retrained on cropped
input windows (3 seeds, 5 for 0–6 s and 0–4 s). (b) Per-subject band-power LDA in a
sliding 1 s window. Both are descriptive scans read on the test set. Axis ranges
64–90% and 45–75%.

**Fig. 16** (`fig16_learned_filters`, double column). (a) Frequency response of the
16 temporal filters of the best run, each scaled to its own maximum. Dashed lines
mark 8, 13 and 30 Hz. (b) Share of kernel power per band in the final model against
a flat spectrum (5 seeds).

**Fig. 17** (`fig17_bonus_comparison`, double column). (a) Learning rate for
DeepConvNet and ShallowConvNet, (b) DeepConvNet activation, and (c) last-10 accuracy
against parameter count, using the best setting of each model. All use 150 epochs
and all 1,080 trials (3 seeds). Axis range 74–90%.

**Fig. 18** (`fig18_bonus_curves`, double column). Train and test curves of the best
DeepConvNet (ReLU, lr 10⁻³) and ShallowConvNet (lr 3×10⁻⁴). Lines and bands show the
mean ± SD of 3 seeds.
