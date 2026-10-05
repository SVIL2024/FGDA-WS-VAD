"""UCF-Crime training entrypoint for one Freq-Text Aug ablation variant.

    python ucf_train.py --variant E1 --max-epoch 10

Mirrors VadCLIP's UCF trainer: two balanced loaders (normal / anomaly, one batch
drawn from each per step) so the MIL top-k pooling is not diluted by the ~50%
normal fraction. Metric: frame-level AUC (VadCLIP's UCF headline number).
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
    parser = freq_text_options.build_parser('ucf')
    args = freq_text_options.apply_variant(parser.parse_args())
    args = freq_text_options.resolve_paths(args, dataset='ucf')
    tag = args.run_name or f'ucf_{args.variant}'
    log = make_logger(args.log_dir, tag)
    setup_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'device={device} variant={args.variant} aug_enabled={args.aug_weight > 0}')
    log('args: ' + ' | '.join(f'{k}={v}' for k, v in sorted(vars(args).items())))

    label_map = vadclip_label_map('ucf')
    normal_loader, anomaly_loader = build_loaders(args, 'ucf', log)
    test_loader = build_test_loader(args, 'ucf', log)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    best, _ = run_train(model, normal_loader, anomaly_loader, make_test_fn('ucf'),
                        test_loader, args, label_map, device, log, tag=tag)
    log(f'FINAL {tag} best AUC = {best:.4f}')
    return best


if __name__ == '__main__':
    main()
