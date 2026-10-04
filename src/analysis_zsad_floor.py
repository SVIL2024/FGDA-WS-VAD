"""Stage 0: the training-free text-anchored floor on the pooled (published) protocol.

    python src/analysis_zsad_floor.py
        -> runs/stage0_zsad_floor_{dataset}.json

    python src/analysis_zsad_floor.py --dataset xd --max-train-sign 20 --out-dir <scratch>
        # smoke - NEVER over real artifacts

Why this exists. The cross-domain plan needs a reference point no training run
has produced yet: what does a plain CLIP text-anchored score reach on the exact
test protocol the leaderboard publishes, with zero training and the features
already on disk? Published CLIP-based zero-shot VAD sits around UCF AUC 0.76-0.90
depending on machinery; a training-free margin cue computed in ten lines is the
honest floor any cross-domain method must clear before claiming its gains come
from training rather than from the text anchors alone.

Cues (all training-free, per video over cached CLIP frame features):
  zsad_margin          L2-normalised frames dotted with (mean anomaly prompt
                       embedding - normal prompt embedding) - the standard
                       CLIP-ZSAD margin, no centering;
  zsad_margin_centred  same, on per-video centred features (closer to the
                       repo's own cue family, which always centres);
  zsad_max             max over the frozen class-direction bank of the
                       normalised-frame response - the deployable class-agnostic
                       readout;
  zsad_max_centred     same on centred features.

Protocol. Identical to ``analysis_official_metric``: the pooled call
``roc_auc_score(gt, np.repeat(scores, 16))`` over every test frame in list
order, XD test list referencing only chunk ``__0`` (reference convention, do
not fix), missing feature files zero-filled so list-order alignment with GT is
never broken. Sign is NOT free in the pooled metric; it is fixed from the
training split using video-level labels only - legitimate weak supervision -
here on a capped subsample (recorded n), because the full XD train list load
costs minutes for a statistic that is stable at n=400. The oracle-sign value is
recorded next to it; a large gap means the orientation does not transfer
between splits, which is itself a finding.

These numbers are the *floor* reference for the cross-domain plan (plan §五
setting S2), not a contribution: quote them with the protocol note, never
against within-video probe numbers (the two-metrics rule in AGENTS.md).
"""

import argparse
import json
import os
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import freq_text_options as opt  # noqa: E402
from analysis_official_metric import load_full, pooled, is_anomalous  # noqa: E402
from freq_text_prompts import DATASET_CLASS_NAMES, get_class_prompts, get_normal_prompt  # noqa: E402
from freq_text_text import GT_FRAME_REPEAT, _f, _i, _read_list, build_class_directions  # noqa: E402
from probe_events import encode_prompts  # noqa: E402


def make_cues(dirs, d_margin):
    """dirs: (n_cls-1, C) unit class directions; d_margin: (C,) abnormal-normal."""
    d_margin = np.asarray(d_margin, dtype=np.float64).ravel()

    def _norm(v):
        x = v.astype(np.float64)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

    def _centred_norm(v):
        xc = v.astype(np.float64) - v.mean(0, keepdims=True)
        return xc / (np.linalg.norm(xc, axis=1, keepdims=True) + 1e-9)

    def zsad_margin(v):
        return _norm(v) @ d_margin

    def zsad_margin_centred(v):
        return _centred_norm(v) @ d_margin

    def zsad_max(v):
        return (_norm(v) @ dirs.T).max(axis=1)

    def zsad_max_centred(v):
        return (_centred_norm(v) @ dirs.T).max(axis=1)

    return {'zsad_margin': zsad_margin, 'zsad_margin_centred': zsad_margin_centred,
            'zsad_max': zsad_max, 'zsad_max_centred': zsad_max_centred}


def train_sign_capped(train_list, cue_fn, dataset, cap=200, seed=0):
    """Cue orientation from video-level training labels, capped for speed.

    Same rule as ``analysis_official_metric.train_sign`` (video-level labels are
    legitimate weak supervision), but stops once ``cap`` positives and negatives
    are collected - the sign is a coarse statistic and the full XD train pass
    costs minutes. The capped n is recorded so nobody mistakes it for the full
    split.
    """
    rows = _read_list(train_list)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(rows))
    pos, neg = [], []
    for i in order:
        if len(pos) >= cap and len(neg) >= cap:
            break
        p, lab = rows[i]
        if not os.path.exists(p):
            continue
        try:
            v = np.load(p).astype(np.float32)
        except (OSError, ValueError, EOFError):
            continue
        if v.ndim != 2 or v.shape[0] == 0:
            continue
        try:
            m = _f(np.mean(cue_fn(v)))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(m):
            continue
        (pos if is_anomalous(lab, dataset) else neg).append(m)
    if len(pos) < 10 or len(neg) < 10:
        return None
    mp, mn = _f(np.mean(pos)), _f(np.mean(neg))
    return {'positive': mp, 'negative': mn, 'sign': 1.0 if mp >= mn else -1.0,
            'n_pos': len(pos), 'n_neg': len(neg)}


def pooled_scores_for(feats, n_frames, fn, sign, mode):
    """Mirror analysis_official_metric's assembly exactly (zero-fill, trim,
    optional per-video z, repeat by GT_FRAME_REPEAT)."""
    parts = []
    for k in range(len(feats)):
        arr, n = feats[k], n_frames[k]
        nf = _i(n, 0)
        if arr is None:
            parts.append(np.zeros(nf, dtype=np.float64))
            continue
        t = min(_i(len(arr), 0), nf)
        s = np.asarray(fn(arr[:t]), dtype=np.float64).ravel()
        if s.ndim != 1 or s.size != t:
            s = np.resize(s, t) if s.size else np.zeros(t)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        if mode == 'pv_z':
            s = (s - _f(np.mean(s))) / (_f(np.std(s)) + 1e-8)
        pad = np.zeros(max(nf - t, 0), dtype=np.float64)
        parts.append(np.concatenate([s, pad]) * sign)
    return np.concatenate([np.repeat(p, GT_FRAME_REPEAT) for p in parts])


def run(dataset, args):
    base = os.path.join(opt.repo_root(), 'list')
    test_list = os.path.join(base, f'{dataset}_CLIP_rgbtest.csv')
    train_list = os.path.join(base, f'{dataset}_CLIP_rgb.csv')
    gt_path = os.path.join(base, 'gt.npy' if dataset == 'xd' else 'gt_ucf.npy')

    print(f'\n=== dataset={dataset} (pooled published protocol) ===', flush=True)
    feats, n_frames, gt, missing, aligned = load_full(test_list, gt_path)
    total = sum(_i(n, 0) for n in n_frames)
    print(f'  {len(feats)} rows, {total} frames x{GT_FRAME_REPEAT} vs gt={len(gt)} '
          f'aligned={aligned} unreadable={missing}', flush=True)
    if not aligned:
        print('  ABORT: list/GT misalignment would silently corrupt the pooled AUC')
        return None

    dirs_t, _ = build_class_directions('ViT-B/16', dataset, device='cpu')
    dirs = dirs_t.numpy().astype(np.float64)[1:]          # anomaly classes only
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

    prompts = [p for c in range(1, len(DATASET_CLASS_NAMES[dataset]))
               for p in get_class_prompts(c, dataset)]
    emb = encode_prompts('ViT-B/16', prompts, device='cpu').cpu().numpy().astype(np.float64)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    emb_abn = emb.mean(0)
    e_nrm = encode_prompts('ViT-B/16', [get_normal_prompt(dataset)],
                           device='cpu').cpu().numpy().astype(np.float64)[0]
    e_nrm = e_nrm / (np.linalg.norm(e_nrm) + 1e-12)
    d_margin = emb_abn - e_nrm

    cues = make_cues(dirs, d_margin)
    rep = {'dataset': dataset, 'rows': len(feats), 'feature_frames': total,
           'gt_len': _i(len(gt), 0), 'aligned': bool(aligned), 'unreadable': _i(missing, 0),
           'max_train_sign': args.max_train_sign}
    results = {}
    for name, fn in cues.items():
        sinfo = train_sign_capped(train_list, fn, dataset,
                                  cap=args.max_train_sign, seed=args.seed)
        sign = _f((sinfo or {}).get('sign'), 1.0)
        if sinfo:
            print(f"  sign[{name:<20}] train pos {sinfo['positive']:+.4f} vs "
                  f"neg {sinfo['negative']:+.4f} -> {sign:+.0f} "
                  f"({sinfo['n_pos']}/{sinfo['n_neg']} videos)", flush=True)
        else:
            print(f'  sign[{name:<20}] UNDETERMINED, defaulting to +1', flush=True)
        for mode in ('raw', 'pv_z'):
            scores = pooled_scores_for(feats, n_frames, fn, sign, mode)
            r = pooled(scores, gt, f'{name}/{mode}')
            r['sign'] = _f(sign, 1.0)
            results[f'{name}/{mode}'] = r
            print(f'    {name:<22} {mode:<6} AUC {_f(r["auc"], 0):.4f}  '
                  f'AP {_f(r["ap"], 0):.4f}', flush=True)
    rep['cues'] = results

    # context: the repo's own training-free cue numbers on the same protocol
    ref_path = os.path.join(opt.repo_root(), 'runs', f'official_metric_{dataset}.json')
    if os.path.exists(ref_path):
        with open(ref_path, encoding='utf-8') as f:
            ref = json.load(f)
        rep['repo_official_metric_reference'] = ref.get('cues', {})
    rep['literature_reference'] = {
        'vadclip_published': {'xd_ap': 0.8451, 'ucf_auc': 0.8802,
                              'provenance': 'VadCLIP (AAAI 2024) paper-reported, '
                                            'same official CLIP features; not reproduced here'},
        'zsad_paper_reported': {'ucf_auc': 0.8986, 'xd_auc': 0.9507, 'xd_ap': 0.8482,
                                'provenance': 'NOVA (arXiv:2609.06360) paper-reported, '
                                              'training-free ZSAD, full pipeline with '
                                              'normal-side modelling; not reproduced here'}}
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='both', choices=['xd', 'ucf', 'both'])
    ap.add_argument('--max-train-sign', type=int, default=200,
                    help='cap per class for the train-split sign calibration')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default=None,
                    help='output dir override; default runs/ - NEVER smoke over runs/')
    args = ap.parse_args()

    out = {'args': vars(args),
           'env': {'python': platform.python_version(), 'numpy': np.__version__}}
    for ds in (['xd', 'ucf'] if args.dataset == 'both' else [args.dataset]):
        r = run(ds, args)
        if r:
            out[ds] = r

    out_dir = args.out_dir or os.path.join(opt.repo_root(), 'runs')
    os.makedirs(out_dir, exist_ok=True)
    done = [d for d in ('xd', 'ucf')
            if isinstance(out.get(d), dict) and 'cues' in out[d]]
    for ds in done:
        path = os.path.join(out_dir, f'stage0_zsad_floor_{ds}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'args': out['args'], 'env': out['env'], ds: out[ds]},
                      f, indent=2, ensure_ascii=False)
        print(f'saved -> {path}', flush=True)

    print('\n' + '=' * 78)
    print('STAGE 0 / ZSAD FLOOR (pooled protocol; UCF primary=AUC, XD primary=AP)')
    print('=' * 78)
    for ds in ('xd', 'ucf'):
        if ds not in out or 'cues' not in out.get(ds, {}):
            continue
        key = 'ap' if ds == 'xd' else 'auc'
        print(f'\n[{ds}] ranked by {key.upper()} (raw pooling):')
        ranked = sorted(((k, v) for k, v in out[ds]['cues'].items() if k.endswith('/raw')),
                        key=lambda kv: -_f(kv[1].get(key), 0.0))
        for k, v in ranked:
            print(f"  {k:<28} {key.upper()}={_f(v.get(key), 0):.4f} "
                  f"AUC={_f(v.get('auc'), 0):.4f} AP={_f(v.get('ap'), 0):.4f}")
    print("""
  Read: these are the training-free floors the cross-domain plan's S2 setting
  must beat, on the exact protocol the leaderboard publishes. VadCLIP published
  XD AP 84.51 / UCF AUC 88.02 on these same features; a floor within a few
  points of a trained number means the plan's gains must be shown against the
  floor, not against zero.""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
