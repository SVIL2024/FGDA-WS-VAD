"""Cross-domain evaluation: score a source-trained CLIPVAD checkpoint on a TARGET
dataset's test set with the published pooled protocol.

    python src/cross_eval.py --source xd --target ucf
    python src/cross_eval.py --source xd --target ucf --max-clips 20 --out-dir <scratch>   # smoke
    -> runs/cross_eval_<source>2<target>.json

Why this exists. The cross-domain plan (跨域VAD_顶刊研究方案.md, setting S2) needs
its first measured baseline: how much does a source-domain weakly-supervised
detector keep when the target domain is never seen? The official VadCLIP
checkpoints (referCode/VadCLIP-main/data/model_{xd,ucf}.pth, verified to load
into this repo's model with strict=True, 2026-10-02) provide that detector
without any training, so the cross-domain floor is a measurement, not a
literature quote.

Scoring head. CLIPVAD has two heads (model.py: forward):

* ``logits1`` - the class-agnostic binary head, ``Linear(512, 1)`` on
  ``visual + mlp2(visual)``. No parameter depends on ``num_class`` (also
  ``text_prompt_embeddings`` is a fixed ``77 x embed_dim`` table), so one
  architecture serves both datasets and the binary head transfers by
  construction.
* ``logits2`` - the class-aligned head, a matmul against the prompt-tuned text
  features. The prompt table was tuned on the SOURCE classes; feeding it the
  TARGET label map is the model's own zero-shot transfer path.

Primary metric follows the vad-training dual-head convention: XD targets select
``max(AP1, AP2)``; UCF targets select the larger ROC (``max(AUC1, AUC2)``),
with the selected head recorded in ``primary_head``. All four numbers are
always stored - the selection is a reporting convention, not a refit.

Architecture note: XD and UCF differ in ``visual_layers`` (1 vs 2) and
``attn_window`` (64 vs 8), which shape the temporal transformer. The model MUST
be built with the SOURCE dataset's hyper-parameters or the checkpoint will not
load; the TARGET contributes only its test list, GT and prompt map.

Protocol. Identical to ``freq_text_test`` / VadCLIP's published loop: one pooled
metric call over every frame of every target video in list order, batch 1,
no per-video normalisation; the XD test list references only chunk ``__0``
(reference convention, do not fix). A ``--max-clips`` capped run truncates the
GT to the covered prefix and prints that its numbers are NOT comparable.

This is the program the remote-GPU gate runs (plan Stage 0, §六): train a
variant on the source with ``xd_train.py``/``ucf_train.py``, then evaluate the
best checkpoint here. Same program, local GPU for verification, server for the
real sweep.
"""

import argparse
import json
import os
import platform
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import freq_text_options  # noqa: E402
from freq_text_dataset import ClipVideoDataset, collate  # noqa: E402
from freq_text_prompts import vadclip_label_map  # noqa: E402
from freq_text_test import GT_FRAME_REPEAT, _f, _i, _within_video_auc, _frame_probs  # noqa: E402
from utils.tools import get_prompt_text  # noqa: E402

def build_target_prompt_text(target, style):
    """Target-side prompt text for the text-aligned head.

    bare    - VadCLIP's class names ('normal', 'fighting', ...): the published
              protocol;
    generic - ``a video of {name}``: the template measured at +6.6 (UCF) /
              +14.5 (XD) AUC over bare words in training-free scoring
              (vad-exp §5); the text-aligned head consumes the same CLIP text
              tower, so the wording carries over;
    scenes  - the repo's own scene templates (freq_text_prompts), first
              paraphrase per class.

    One string per class, index-aligned with the label map - exactly the
    format ``get_prompt_text`` returns, so the model path is unchanged.
    """
    if target == 'sht':
        return SHT_PROMPTS[style]
    label_map = vadclip_label_map(target)
    if style == 'bare':
        return get_prompt_text(label_map)
    if style == 'generic':
        return [f'a video of {name}' for name in label_map.values()]
    if style == 'scenes':
        from freq_text_prompts import get_class_prompts
        return [get_class_prompts(i, target)[0] for i in range(len(label_map))]
    raise ValueError(f'unknown prompt style: {style}')


class _CenteredDataset:
    """Subtracts one fixed per-feature mean from every item (M4 probe).

    Transductive: the mean is computed over the target test features
    themselves - the UPPER BOUND of the few-normal-reference calibration the
    cross-domain plan's M4 proposes. Every number produced under this flag
    must be labelled transductive in writing.
    """

    def __init__(self, ds):
        self.ds = ds
        total, n = None, 0
        for i in range(len(ds)):
            feat = ds[i][0]
            length = int(ds[i][2])
            x = feat[:length].double()
            total = x.sum(0) if total is None else total + x.sum(0)
            n += length
        self.mean = (total / max(n, 1)).float()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        feat, label, length, cls, binary = self.ds[i]
        return feat - self.mean, label, length, cls, binary


SHT_FEAT_ROOT = freq_text_options.sht_feat_root()
# SHT carries no typed anomaly classes; the text-aligned head scores against a
# binary prompt pair (index 0 = normal, consumed as 1 - p(normal))
SHT_PROMPTS = {
    'bare': ['normal scene', 'anomalous event'],
    'generic': ['a video of a normal scene', 'a video of an anomalous event'],
    'scenes': ['a video of a normal scene', 'a video of an anomalous event'],
}


def gt_path_for(target):
    if target == 'sht':
        return os.path.join(SHT_FEAT_ROOT, 'gt_sht.npy')
    return os.path.join(freq_text_options.repo_root(), 'list',
                        'gt.npy' if target == 'xd' else 'gt_ucf.npy')


def test_list_for(target):
    if target == 'sht':
        return os.path.join(SHT_FEAT_ROOT, 'sht_CLIP_rgbtest.csv')
    return os.path.join(freq_text_options.repo_root(), 'list',
                        f'{target}_CLIP_rgbtest.csv')


def resolve_checkpoint(source, checkpoint):
    """``official`` -> first existing VadCLIP release checkpoint; else a path.

    Resolution order is documented in freq_text_options.official_checkpoints().
    """
    if checkpoint in ('official', None):
        candidates = freq_text_options.official_checkpoints(source)
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f'no official checkpoint for {source}; set '
            f'RSI_VADCLIP_{source.upper()} (or FGDA_VADCLIP_{source.upper()}) '
            f'to the downloaded model_{source}.pth, or pass '
            f'--checkpoint <path>. Looked in: {candidates}')
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint}')
    return checkpoint


def run(source, target, checkpoint, max_clips, save_scores, out_dir, device,
        eval_tag=None, prompt_style='bare', center_target=False):
    args = freq_text_options.apply_variant(
        freq_text_options.build_parser(source).parse_args([]))
    args = freq_text_options.resolve_paths(args, dataset=source)
    ckpt_path = resolve_checkpoint(source, checkpoint)

    from torch.utils.data import DataLoader, Subset

    from model import CLIPVAD  # noqa: E402  (import after options: needs src/clip)

    target_list = test_list_for(target)
    gt = np.load(gt_path_for(target)).astype(np.float32)
    prompt_text = build_target_prompt_text(target, prompt_style)

    dataset = ClipVideoDataset(args.visual_length, target_list, test_mode=True,
                               # sht lists carry XD's 'A' normal code so the
                               # shared label parsing works; labels play no
                               # role in evaluation
                               dataset='xd' if target == 'sht' else target)
    if center_target:
        print('  computing target feature mean (transductive M4 probe)...', flush=True)
        dataset = _CenteredDataset(dataset)
    if max_clips is not None and max_clips < len(dataset):
        dataset = Subset(dataset, range(max_clips))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix,
                    device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)

    p1, p2, _stack, _names, lens = _frame_probs(model, loader, args.visual_length,
                                                prompt_text, device, stack_logits=False)
    s1, s2 = np.repeat(p1, GT_FRAME_REPEAT), np.repeat(p2, GT_FRAME_REPEAT)

    # capped runs cover a prefix of the GT; score only what was predicted, and
    # label the numbers non-comparable (same rule as freq_text_test.make_test_fn)
    expected = len(s1)
    gt_frames_total = _i(len(gt))
    if len(gt) < expected:
        raise ValueError(f'predicted {expected} frames but target GT has {len(gt)}')
    capped = len(gt) != expected
    if capped:
        print(f'  [warn] capped run covered {expected} of {len(gt)} GT frames '
              f'- metrics below are NOT comparable')
        gt = gt[:expected]

    auc1, ap1 = _f(roc_auc_score(gt, s1)), _f(average_precision_score(gt, s1))
    auc2, ap2 = _f(roc_auc_score(gt, s2)), _f(average_precision_score(gt, s2))
    wv1, n_wv = _within_video_auc(p1, lens, gt)
    wv2, _ = _within_video_auc(p2, lens, gt)

    # primary per the vad-training dual-head convention: XD max(AP1, AP2),
    # UCF the larger ROC. Capped runs keep the numbers but are flagged.
    if target == 'xd':
        primary_head = 'AP2' if ap2 >= ap1 else 'AP1'
        primary_value = ap2 if ap2 >= ap1 else ap1
    else:
        primary_head = 'AUC2' if auc2 >= auc1 else 'AUC1'
        primary_value = auc2 if auc2 >= auc1 else auc1

    # the true variant lives in the run dir's history JSON, not in the parser
    # defaults (the audit finding: E1 checkpoints were labelled 'E5')
    true_variant = None
    try:
        import glob as _glob
        hists = _glob.glob(os.path.join(os.path.dirname(ckpt_path), '*_history.json'))
        if hists:
            with open(hists[0], encoding='utf-8') as f:
                true_variant = json.load(f).get('args', {}).get('variant')
    except (OSError, ValueError, AttributeError):
        true_variant = None

    rep = {
        'source': source, 'target': target, 'checkpoint': ckpt_path,
        'prompt_style': prompt_style, 'center_target': bool(center_target),
        'capped': bool(capped), 'n_clips': len(dataset),
        'target_frames': expected, 'gt_frames': gt_frames_total,
        'source_args': {k: getattr(args, k) for k in
                        ('classes_num', 'embed_dim', 'visual_length', 'visual_width',
                         'visual_head', 'visual_layers', 'attn_window',
                         'prompt_prefix', 'prompt_postfix', 'variant', 'seed')},
        'checkpoint_variant': true_variant,
        'metrics': {'AUC1': round(auc1, 4), 'AP1': round(ap1, 4),
                    'AUC2': round(auc2, 4), 'AP2': round(ap2, 4),
                    'WV_AUC1': _f(wv1), 'WV_AUC2': _f(wv2), 'WV_n_videos': n_wv},
        'primary_head': primary_head,
        'primary_value': round(_f(primary_value), 4) if not capped else None,
        'device': device,
        'env': {'python': platform.python_version(), 'torch': torch.__version__},
    }

    out_dir = out_dir or os.path.join(freq_text_options.repo_root(), 'runs')
    os.makedirs(out_dir, exist_ok=True)
    # per-checkpoint tag keeps variant/seed evals from overwriting each other
    # (disk discipline: every result file must identify its own run)
    tag = eval_tag or os.path.basename(os.path.dirname(ckpt_path))
    if not eval_tag and tag == 'data':
        tag = 'official'
    suffix = '_smoke' if max_clips is not None else ''
    path = os.path.join(out_dir, f'cross_eval_{source}2{target}_{tag}{suffix}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    if save_scores:
        np.savez_compressed(os.path.join(out_dir, f'cross_scores_{source}2{target}.npz'),
                            prob1=p1, prob2=p2, gt=gt)
    print(f'saved -> {path}', flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, choices=['xd', 'ucf', 'union'],
                    help='dataset the checkpoint was trained on (fixes the '
                         'architecture; union = the 15-class XD+UCF multi-source '
                         'space, same hyper-parameter family as xd)')
    ap.add_argument('--target', required=True, choices=['xd', 'ucf', 'sht'],
                    help='dataset to evaluate on (sht = ShanghaiTech, binary '
                         'prompt pair, gt built from label/test.csv)')
    ap.add_argument('--checkpoint', default='official',
                    help="'official' = the vendored VadCLIP release checkpoint for "
                         'the source, or a path to runs/<...>/model_best.pth')
    ap.add_argument('--max-clips', type=int, default=None,
                    help='cap on target videos (smoke only; metrics then NOT comparable)')
    ap.add_argument('--save-scores', action='store_true',
                    help='save per-frame scores npz (disk discipline: off by default)')
    ap.add_argument('--out-dir', default=None,
                    help='output dir override; default runs/ - NEVER smoke over runs/')
    ap.add_argument('--tag', default=None,
                    help='result-file tag; defaults to the checkpoint parent dir '
                         'name, so runs/<variant>_s<seed> checkpoints stay distinct')
    ap.add_argument('--prompt-style', default='bare', choices=['bare', 'generic', 'scenes'],
                    help="target-side prompt wording for the text-aligned head; "
                         "generic = 'a video of {name}' (the template that beat bare "
                         'words by +6.6/+14.5 AUC in training-free scoring)')
    ap.add_argument('--center-target', action='store_true',
                    help='subtract the target test-set feature mean before scoring '
                         '(transductive M4 probe; label results as transductive)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # run-command first line: the whole experiment is recoverable from the log
    print(f'[RunCommand] python src/cross_eval.py --source {args.source} '
          f'--target {args.target} --checkpoint {args.checkpoint} '
          f'--prompt-style {args.prompt_style} --center-target {args.center_target} '
          f'--max-clips {args.max_clips} --out-dir {args.out_dir}', flush=True)
    print(f'[Config] device={device} source={args.source} target={args.target} '
          f'prompt_style={args.prompt_style} center_target={args.center_target} '
          f'smoke={"yes" if args.max_clips else "no"}', flush=True)
    rep = run(args.source, args.target, args.checkpoint, args.max_clips,
              args.save_scores, args.out_dir, device, eval_tag=args.tag,
              prompt_style=args.prompt_style, center_target=args.center_target)
    m = rep['metrics']
    print(f'\nRESULT {args.source} -> {args.target} '
          f'(head1=class-agnostic binary, head2=prompt-aligned vs TARGET classes)')
    print(f'  head1: AUC={m["AUC1"]:.4f} AP={m["AP1"]:.4f}  wvAUC={m["WV_AUC1"]:.4f}')
    print(f'  head2: AUC={m["AUC2"]:.4f} AP={m["AP2"]:.4f}  wvAUC={m["WV_AUC2"]:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
