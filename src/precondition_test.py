"""Pre-flight gate: is the CLIP text direction a usable anomaly cue?

    python precondition_test.py --dataset xd
    python precondition_test.py --dataset ucf --n-videos 800

Writes ``runs/precondition_<dataset>.json`` and exits non-zero on a STOP verdict,
so a driver script can refuse to launch E1-E5 on a dead premise.

The whole scheme assumes that ``e_class - e_normal`` is an axis along which the
frozen CLIP feature bank separates anomalous from normal frames. The probe scores
that direction the way the trained model is graded - ranking anomalous frames
above normal ones *within* a test video against the ground-truth segments - and
compares it against a null of random unit directions and against the free
feature-magnitude cue. If the text direction cannot beat those, E1 cannot beat E3
and the correct response is to change the direction source, not to tune alpha.
"""

import argparse
import json
import os
import sys

import freq_text_options
from freq_text_text import precondition_report


def main():
    ap = argparse.ArgumentParser(
        description='Check whether the text direction is aligned with the empirical '
                    'anomaly direction before running the E1-E5 ablation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--dataset', default='xd', choices=['xd', 'ucf'])
    ap.add_argument('--test-list', default=None,
                    help='default: list/<dataset>_CLIP_rgbtest.csv')
    ap.add_argument('--gt-path', default=None,
                    help='default: list/gt.npy (xd) or list/gt_ucf.npy (ucf)')
    ap.add_argument('--clip-arch', default='ViT-B-16',
                    help='keep equal to the architecture CLIPVAD loads internally')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-videos', type=int, default=400,
                    help='max test videos sampled (only mixed-GT videos are usable)')
    ap.add_argument('--threshold', type=float, default=0.2,
                    help='unused by the decisive probe; kept for the doc comparison')
    ap.add_argument('--sep-margin', type=float, default=0.01,
                    help='AUC the text mean must beat the random-direction p95 by')
    ap.add_argument('--n-random', type=int, default=200)
    ap.add_argument('--no-separability', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    test_list = freq_text_options.resolve_data_path(
        args.test_list or f'list/{args.dataset}_CLIP_rgbtest.csv')
    gt_path = freq_text_options.resolve_data_path(
        args.gt_path or ('list/gt.npy' if args.dataset == 'xd' else 'list/gt_ucf.npy'))
    for pth, what in ((test_list, 'test list'), (gt_path, 'gt file')):
        if not os.path.exists(pth):
            sys.exit(f'{what} not found: {pth}')
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print('cuda requested but unavailable -> cpu')
            args.device = 'cpu'

    out = args.out or os.path.join(freq_text_options.repo_root(),
                                   f'runs/precondition_{args.dataset}.json')
    rep = precondition_report(args.clip_arch, args.dataset, test_list, gt_path,
                              n_videos=args.n_videos, device=args.device,
                              threshold=args.threshold, write_json=out,
                              separability=not args.no_separability,
                              n_random=args.n_random,
                              margin_threshold=args.sep_margin)

    _print(rep, out)
    # non-zero exit on STOP lets run_ablation.ps1 skip the sweep
    return 0 if rep.get('proceed') else 2


def _print(rep, out_path):
    sep = rep.get('linear_separability', {})
    ts = rep['text_embedding_structure']

    print()
    print('=' * 78)
    print(f"PRECONDITION  dataset={rep['dataset']}  clip={rep['clip_arch']}")
    print('=' * 78)

    m = rep['class_direction_mutual_cos']
    print(f"  class-direction mutual cos      mean={m['mean']:+.4f} "
          f"min={m['min']:+.4f} max={m['max']:+.4f}")
    print(f"  cos(e_class, e_normal)          mean={ts['cos_e_anom_normal_mean']:+.4f} "
          f"max={ts['cos_e_anom_normal_max']:+.4f}")
    print(f"  ||e_class - e_normal|| (raw)    mean={ts['norm_d_raw_mean']:.4f}")

    if sep.get('ok'):
        print()
        print('  WITHIN-VIDEO GT PROBE  (decisive: rank anomalous frames above normal')
        print('  frames inside a test video, per-video normalised, vs the GT segments)')
        print(f"    videos used                     {sep['n_test_videos_used']} "
              f"({sep['frames_per_video_mean']:.0f} frames each)")
        print(f"    supervised ceiling AUC          {sep['supervised_ceiling_auc']:.4f}")
        print(f"    free ||f|| magnitude cue        {sep['magnitude_cue_auc']:.4f}   "
              f"<- must be beaten")
        print(f"    random dirs ({sep['n_random']})              mean={sep['random_auc_mean']:.4f} "
              f"p95={sep['random_auc_p95']:.4f} max={sep['random_auc_max']:.4f}")
        print(f"    text dirs                     mean={sep['text_auc_mean']:.4f} "
              f"max={sep['text_auc_max']:.4f} ({sep['text_auc_best_class']})")
        print(f"    margin over random mean       {sep['margin_over_random_mean']:+.4f} "
              f"(need > {sep['margin_threshold']:.4f}, p={sep['p_value_vs_random_mean']:.3f})")
        print(f"    beats random p95 / max        {sep['beats_random_mean']} / "
              f"{sep['beats_random_max']}")
        print(f"    beats magnitude cue           {sep['beats_magnitude_cue']} "
              f"(class_max basis)   mean basis: {sep.get('beats_magnitude_cue_mean')}")
        print(f"    headroom text -> ceiling      {sep['headroom_text_to_ceiling']:+.4f}")
        top = list(sep['per_class_within_video_auc'].items())[:6]
        print('    per-class within-video AUC: ' +
              ', '.join(f'{k} {v:.3f}' for k, v in top))
    elif sep:
        print(f"  within-video probe unavailable: {sep.get('reason')}")

    print('-' * 78)
    print(f"  PROCEED: {rep['proceed']}")
    print(f'  VERDICT: {rep["verdict"]}')
    print(f'  saved -> {out_path}')
    print('=' * 78)


if __name__ == '__main__':
    sys.exit(main())
