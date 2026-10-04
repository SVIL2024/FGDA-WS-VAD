"""Multi-source (XD+UCF) training entrypoint for one Freq-Text Aug variant.

    python union_train.py --variant E3 --max-epoch 10
    python union_train.py --variant E5 --max-epoch 10     # multi-source baseline

Trains on ``list/union_CLIP_rgb.csv`` (XD rows in native codes + UCF rows
remapped to union codes, built by ``build_union_list.py``) with the 15-class
union prompt table. The XD hyper-parameter family applies (visual_layers 1,
attn_window 64, batch 96) - classes_num 15 is the only architectural
difference, so checkpoints stay CLIPVAD-compatible.

Checkpoint selection monitors the XD test set (source-domain validation):
selecting on a cross-domain target would be target leakage. Cross-domain
numbers come from ``cross_eval.py --target <held-out>`` AFTER training.
"""

import datetime
import os

import torch

import freq_text_options
from freq_text_prompts import vadclip_label_map
from freq_text_test import make_test_fn
from freq_text_trainer import build_loaders, build_test_loader, run_train, setup_seed
from model import CLIPVAD


def make_logger(log_dir: str, tag: str):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f'{tag}.log')

    def log(msg):
        line = f'{datetime.datetime.now():%H:%M:%S} {msg}'
        print(line, flush=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    return log


def main():
    parser = freq_text_options.build_parser('union')
    args = freq_text_options.apply_variant(parser.parse_args())
    args = freq_text_options.resolve_paths(args, dataset='union')
    tag = args.run_name or f'union_{args.variant}'
    log = make_logger(args.log_dir, tag)
    setup_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'device={device} variant={args.variant} aug_enabled={args.aug_weight > 0}')
    log('args: ' + ' | '.join(f'{k}={v}' for k, v in sorted(vars(args).items())))

    label_map = vadclip_label_map('union')
    train_loader, anomaly_loader = build_loaders(args, 'union', log)
    test_loader = build_test_loader(args, 'union', log)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    # monitoring stays on the XD test protocol (source-domain validation); the
    # cross-domain numbers are produced afterwards by cross_eval.py
    best, _ = run_train(model, train_loader, anomaly_loader, make_test_fn('xd'),
                        test_loader, args, label_map, device, log, tag=tag)
    log(f'FINAL {tag} best AP = {best:.4f}')
    return best


if __name__ == '__main__':
    main()
