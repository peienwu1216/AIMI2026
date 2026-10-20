"""Run the two required ResNet experiments and generate all report figures."""
import argparse
import subprocess
import sys


def run(cmd):
    print('\n$', ' '.join(str(x) for x in cmd),flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--image-size',type=int,default=224)
    ap.add_argument('--num-workers',type=int,default=0)
    ap.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    ap.add_argument('--trainable-stages',default='all',choices=['all','layer4','head'])
    ap.add_argument('--runs-dir',default='runs')
    ap.add_argument('--figures-dir',default='figures')
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--split-seed',type=int,default=42)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--patience',type=int,default=6)
    ap.add_argument('--dropout',type=float,default=0.3)
    ap.add_argument('--label-smoothing',type=float,default=0.05)
    ap.add_argument('--class-weighting',default='balanced',choices=['balanced','none'])
    ap.add_argument('--head-lr-multiplier',type=float,default=5.0)
    ap.add_argument('--scheduler',default='plateau',choices=['none','plateau','cosine'])
    ap.add_argument('--selection-metric',default='balanced_accuracy',
                    choices=['accuracy','balanced_accuracy','macro_f1','f1'])
    ap.add_argument('--threshold-metric',default='balanced_accuracy',
                    choices=['accuracy','balanced_accuracy','macro_f1','f1'])
    ap.add_argument('--resnet18-threshold',type=float,default=0.5)
    ap.add_argument('--resnet50-threshold',type=float,default=0.5)
    ap.add_argument('--resize-mode',default='stretch',choices=['pad','stretch'])
    ap.add_argument('--hflip-prob',type=float,default=0.1)
    ap.add_argument('--no-group-aware-split',action='store_true')
    ap.add_argument('--no-freeze-bn',action='store_true')
    args=ap.parse_args()
    thresholds = {'resnet18':args.resnet18_threshold,
                  'resnet50':args.resnet50_threshold}
    for model in ('resnet18','resnet50'):
        run([sys.executable,'train.py','--data-dir',args.data_dir,'--model',model,
             '--epochs',str(args.epochs),'--batch-size',str(args.batch_size),
             '--image-size',str(args.image_size),
             '--num-workers',str(args.num_workers),'--device',args.device,
             '--trainable-stages',args.trainable_stages,'--output-dir',args.runs_dir,
             '--seed',str(args.seed),'--lr',str(args.lr),'--patience',str(args.patience),
             '--dropout',str(args.dropout),'--label-smoothing',str(args.label_smoothing),
             '--class-weighting',args.class_weighting,
             '--head-lr-multiplier',str(args.head_lr_multiplier),
             '--scheduler',args.scheduler,'--selection-metric',args.selection_metric,
             '--threshold-metric',args.threshold_metric,'--resize-mode',args.resize_mode,
             '--hflip-prob',str(args.hflip_prob),
             '--decision-threshold',str(thresholds[model])]
            + ([] if args.no_group_aware_split else ['--group-aware-split'])
            + ([] if args.split_seed is None else ['--split-seed',str(args.split_seed)])
            + ([] if args.no_freeze_bn else ['--freeze-bn']))
    run([sys.executable,'compare.py','--runs-dir',args.runs_dir,'--figures-dir',args.figures_dir])
    print('Training, test evaluation, and figures completed.')

if __name__=='__main__':
    main()
