"""Stage 0 (plan §六 gate): do anomaly directions transfer across domains at all?

    python src/analysis_crossdomain_transfer.py
        -> runs/stage0_direction_transfer.json

    python src/analysis_crossdomain_transfer.py --max-videos 40 --n-random 20 \
        --n-random-class 10 --out-dir <scratch>       # smoke - NEVER over real artifacts

The cross-domain plan (cross-domain repo, 跨域VAD_顶刊研究方案.md §三 H1) bets that
cross-domain generalisation has a geometric basis: a direction that separates
anomalous from normal frames in one domain still separates them in another, so a
source-domain detector - or a text-authored direction, which moves for free -
can cover target-domain anomalies it never saw. Before any GPU time goes into
the cross-domain training gate, this script measures the two readouts that
decide whether that bet can pay, on features that are already on disk:

  R1  text-direction portability. The CLIP text bank is the only part of the
      method that can move to an unseen domain for free. Direction for concept c
      built from dataset A's prompt bank, probed on dataset B's test videos -
      both class-matched (oracle: the video's own label picks the concept) and
      over the whole mixed-GT pool.
  R2  visual-direction portability. The anomaly-vs-normal centroid difference
      computed on dataset A's test videos of class c, probed on dataset B's
      test videos of the same class. Fighting / shooting / abuse / explosion
      exist in both UCF-Crime and XD-Violence, which is what makes this
      measurable at all. If R2 fails, "source anomalies cover target anomalies"
      is false at the feature level and the plan's M2 synthesis premise dies
      with it; if R2 holds but R1 fails, the fix is direction *authoring*
      (learned or LLM-authored), not the cross-domain framing.

Statistic. The repo's within-video probe (freq_text_text.probe_linear_separability):
score each frame, z-score inside the video, AUC against that video's GT segment
labels, free sign, averaged over videos containing both classes. Every probe
runs against a random-direction null pushed through the identical pipeline on
the identical pool, because AGENTS.md records that raw cosine alignment cannot
separate a useful direction from a useless one and CLIP anisotropy makes a bare
AUC > 0.5 vacuous. Cosines are reported but labelled descriptive.

Centering. Visual directions are computed in RAW feature space (no dataset-mean
subtraction): a direction that is meant to travel between domains must not
carry the origin dataset's mean offset. The probe still cleans every direction
against the probe pool's own mean - that step is part of the probe protocol and
is applied identically to text, visual and random directions.

Conventions kept from the repo probes (do not "improve" them silently):
* free sign ``max(a, 1-a)`` per video - the within-video probe statistic;
* the null is compared to the *mean over videos*, never to pooled per-video
  values (the p95-of-pooled-values sample-size trap recorded in AGENTS.md);
* XD test list references only chunk ``__0`` of each video - reference
  convention, GT lengths confirm it, do not fix;
* oracle class-matched readouts use the video's own test label to pick the
  concept. That is a diagnostic of direction quality, not a deployable score -
  every such number is labelled ``matched``.
"""

import argparse
import json
import os
import platform
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import freq_text_options as opt  # noqa: E402
from freq_text_prompts import DATASET_CLASS_NAMES, primary_class  # noqa: E402
from freq_text_text import (GT_FRAME_REPEAT, _f, _gt_offsets, _i,  # noqa: E402
                            _read_list, build_class_directions)

# shared semantic classes: (concept, xd class idx, ucf class idx) - index tables
# from freq_text_prompts, which are pinned to VadCLIP's label_map order
SHARED_CONCEPTS = [('fighting', 1, 7), ('shooting', 2, 10),
                   ('abuse', 4, 1), ('explosion', 6, 6)]
OTHER = {'xd': 'ucf', 'ucf': 'xd'}
DATASETS = ('xd', 'ucf')


def load_test_videos(dataset, max_videos=None, seed=0):
    """Mixed-GT test videos with primary-class labels, in list order.

    Same alignment rules as ``probe_linear_separability``: GT is frame-level,
    concatenated in list order at ``GT_FRAME_REPEAT`` times the feature rate;
    a video enters the pool only if it contains both classes, which is what
    makes a within-video AUC defined. Videos whose feature length disagrees
    with their GT block are *skipped and counted*, never silently scored
    against a shifted GT array - that failure mode is why ``load_full`` in
    ``analysis_official_metric`` exists.
    """
    base = os.path.join(opt.repo_root(), 'list')
    test_list = os.path.join(base, f'{dataset}_CLIP_rgbtest.csv')
    gt_path = os.path.join(base, 'gt.npy' if dataset == 'xd' else 'gt_ucf.npy')
    rows = _read_list(test_list)
    gt = np.load(gt_path).astype(np.float32)
    offs = _gt_offsets(rows, test_list)
    rng = np.random.RandomState(seed)
    videos, skipped_missing, skipped_len = [], 0, 0
    for i, (p, lab) in enumerate(rows):
        n = _i(offs[i + 1] - offs[i], 0)
        if n <= 0:
            continue
        seg = gt[offs[i] * GT_FRAME_REPEAT: offs[i + 1] * GT_FRAME_REPEAT]
        g = seg.reshape(-1, GT_FRAME_REPEAT).mean(1) > 0.5
        if not (0 < int(g.sum()) < len(g)):
            continue
        if not os.path.exists(p):
            skipped_missing += 1
            continue
        v = np.load(p).astype(np.float32)
        if v.ndim != 2 or v.shape[0] != len(g):
            skipped_len += 1
            continue
        videos.append({'name': os.path.basename(p), 'feat': v, 'gt': g,
                       'cls': primary_class(lab, dataset)})
    if max_videos is not None and len(videos) > max_videos:
        pick = rng.choice(len(videos), size=max_videos, replace=False)
        videos = [videos[i] for i in sorted(pick)]
    return videos, {'missing_feature': skipped_missing, 'length_mismatch': skipped_len}


def make_probe(pool):
    """Within-video probe over one pool, with the pool's own centering.

    Mirrors ``probe_linear_separability.per_video``: the pool mean mu is
    subtracted from frames, the direction is cleaned against mu's own
    direction, scores are z-scored per video, sign is free. Features are
    converted to float64 once - the probe runs thousands of direction
    evaluations over the same pool.
    """
    V = [x['feat'].astype(np.float64) for x in pool]
    G = [x['gt'] for x in pool]
    mu = np.concatenate(V, 0).mean(0)
    mu_n = mu / (np.linalg.norm(mu) + 1e-9)

    def per_video(u, signed=False):
        u = np.asarray(u, dtype=np.float64).ravel()
        uu = u - (u @ mu_n) * mu_n
        nu = np.linalg.norm(uu)
        if nu < 1e-9:
            return None
        uu = uu / nu
        aucs, effects = [], []
        for v, g in zip(V, G):
            s = (v - mu) @ uu
            s = (s - s.mean()) / (s.std() + 1e-8)
            a = _f(roc_auc_score(g, s))
            aucs.append(max(a, 1.0 - a))
            effects.append(s[g].mean() - s[~g].mean())
        out = np.asarray(aucs, dtype=np.float64)
        return out if not signed else (out, np.asarray(effects, dtype=np.float64))

    return per_video


def null_stats(per_video, dim, n_random, rng):
    """Random-direction null means over the pool - same statistic, same videos."""
    vals = []
    for _ in range(int(n_random)):
        r = per_video(rng.randn(dim).astype(np.float64))
        if r is not None:
            vals.append(r.mean())
    return np.asarray(vals, dtype=np.float64)


def p_above_null(null_vals, observed_mean, n_reps=2000, rng=None):
    """p for one direction's mean AUC being a null draw this large."""
    if len(null_vals) == 0:
        return None
    return _f((np.asarray(null_vals) >= observed_mean).mean())


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return _f((a @ b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def _summarise(per_video_result, null_vals, n_videos, rng):
    """Pack one single-direction probe result with its matched-pool null."""
    if per_video_result is None:
        return {'ok': False, 'n_videos': int(n_videos), 'reason': 'degenerate direction'}
    mean = _f(per_video_result.mean())
    if len(null_vals) == 0:
        return {'ok': True, 'n_videos': int(n_videos), 'auc_mean': round(mean, 4),
                'null_mean': None, 'null_p95': None, 'beats_null_p95': None,
                'p_vs_null': None}
    return {'ok': True, 'n_videos': int(n_videos), 'auc_mean': round(mean, 4),
            'null_mean': round(_f(null_vals.mean()), 4),
            'null_p95': round(_f(np.percentile(null_vals, 95)), 4),
            'beats_null_p95': bool(mean > np.percentile(null_vals, 95)),
            'p_vs_null': p_above_null(null_vals, mean, rng=rng)}


def prepare(dataset, n_random, n_random_class, seed, min_class_videos, min_class_frames,
            max_videos=None):
    """One dataset's pools, probes, nulls and own-dataset visual directions.

    Visual directions are centroid differences in raw feature space (see module
    docstring for why no dataset mean is subtracted here). Class pools below
    ``min_class_videos`` or ``min_class_frames`` are skipped and reported, not
    silently probed on a handful of videos. ``max_videos`` caps the pool for
    smoke runs only.
    """
    rng = np.random.RandomState(seed)
    classes = DATASET_CLASS_NAMES[dataset]
    videos, skip = load_test_videos(dataset, max_videos=max_videos, seed=seed)
    if not videos:
        return None
    dim = videos[0]['feat'].shape[1]
    pool_all = videos
    prep = {'dataset': dataset, 'classes': classes, 'videos': videos, 'skip': skip,
            'dim': int(dim), 'pool_all': pool_all}
    prep['probe_all'] = make_probe(pool_all)
    prep['null_all'] = null_stats(prep['probe_all'], dim, n_random, rng)

    pos = np.concatenate([x['feat'][x['gt']] for x in pool_all], 0).mean(0)
    neg = np.concatenate([x['feat'][~x['gt']] for x in pool_all], 0).mean(0)
    prep['global_vis'] = pos - neg

    class_pools, class_vis, class_meta = {}, {}, {}
    for c in range(1, len(classes)):
        pool_c = [x for x in pool_all if x['cls'] == c]
        n_pos = int(sum(int(x['gt'].sum()) for x in pool_c))
        n_neg = int(sum(int((~x['gt']).sum()) for x in pool_c))
        meta = {'n_videos': len(pool_c), 'n_pos_frames': n_pos, 'n_neg_frames': n_neg}
        if len(pool_c) >= min_class_videos and n_pos >= min_class_frames \
                and n_neg >= min_class_frames:
            p = np.concatenate([x['feat'][x['gt']] for x in pool_c], 0).mean(0)
            q = np.concatenate([x['feat'][~x['gt']] for x in pool_c], 0).mean(0)
            class_pools[c] = pool_c
            class_vis[c] = p - q
        else:
            meta['skipped'] = f'below min_class_videos={min_class_videos} ' \
                              f'or min_class_frames={min_class_frames}'
        class_meta[classes[c]] = meta
    prep['class_pools'] = class_pools
    prep['class_vis'] = class_vis
    prep['class_meta'] = class_meta
    prep['null_class'] = {c: null_stats(make_probe(pool_c), dim, n_random_class, rng)
                          for c, pool_c in class_pools.items()}
    return prep


def concept_indices(name, dataset):
    """(own-dataset class idx, other-dataset class idx) for a shared concept."""
    for n, xd_i, ucf_i in SHARED_CONCEPTS:
        if n == name:
            return (xd_i, ucf_i) if dataset == 'xd' else (ucf_i, xd_i)
    raise KeyError(f'unknown shared concept: {name}')


def text_transfer(prep, banks, seed):
    """R1: own-bank vs other-bank text directions on this dataset's pools."""
    dataset = prep['dataset']
    rng = np.random.RandomState(seed + 11)
    own_bank, other_bank = banks[dataset], banks[OTHER[dataset]]
    other_classes = DATASET_CLASS_NAMES[OTHER[dataset]]
    probe_all, null_all = prep['probe_all'], prep['null_all']
    out = {}
    for name, _, _ in SHARED_CONCEPTS:
        idx_own, idx_other = concept_indices(name, dataset)
        own_dir, other_dir = own_bank[idx_own], other_bank[idx_other]
        entry = {'own_class': prep['classes'][idx_own],
                 'other_class': other_classes[idx_other],
                 'cos_text_own_vs_cross_descriptive': _cos(own_dir, other_dir)}
        if idx_own in prep['class_pools']:
            pool_c = prep['class_pools'][idx_own]
            probe_c, null_c = make_probe(pool_c), prep['null_class'][idx_own]
            entry['text_own_matched'] = _summarise(probe_c(own_dir), null_c, len(pool_c), rng)
            entry['text_cross_matched'] = _summarise(probe_c(other_dir), null_c, len(pool_c), rng)
        entry['text_own_all'] = _summarise(probe_all(own_dir), null_all, len(prep['pool_all']), rng)
        entry['text_cross_all'] = _summarise(probe_all(other_dir), null_all, len(prep['pool_all']), rng)
        out[name] = entry
    return out


def visual_transfer(prep, preps, seed):
    """R2: visual directions of the OTHER dataset, probed here, class-matched."""
    dataset = prep['dataset']
    rng = np.random.RandomState(seed + 22)
    other = preps[OTHER[dataset]]
    out = {}
    for name, _, _ in SHARED_CONCEPTS:
        idx_own, idx_other = concept_indices(name, dataset)
        entry = {'own_class': prep['classes'][idx_own],
                 'other_class': other['classes'][idx_other]}
        if idx_own not in prep['class_pools'] or idx_other not in other['class_vis']:
            entry['skipped'] = 'own class pool or other class direction unavailable'
            out[name] = entry
            continue
        pool_c = prep['class_pools'][idx_own]
        probe_c, null_c = make_probe(pool_c), prep['null_class'][idx_own]
        own_vis = prep['class_vis'][idx_own]
        other_vis = other['class_vis'][idx_other]
        entry['visual_own_matched'] = _summarise(probe_c(own_vis), null_c, len(pool_c), rng)
        entry['visual_cross_matched'] = _summarise(probe_c(other_vis), null_c, len(pool_c), rng)
        entry['cos_visual_cross_descriptive'] = _cos(own_vis, other_vis)
        a_own = (entry['visual_own_matched'] or {}).get('auc_mean')
        a_cross = (entry['visual_cross_matched'] or {}).get('auc_mean')
        if a_own is not None and a_cross is not None:
            entry['transfer_cost_own_minus_cross'] = round(_f(a_own - a_cross), 4)
        # specificity control: the other dataset's direction of a DIFFERENT
        # shared concept on this concept's pool - at or below the matched
        # transfer if directions carry class information, not just an
        # anomaly-generic axis
        mismatch = next((n2 for n2, _, _ in SHARED_CONCEPTS
                         if n2 != name and concept_indices(n2, dataset)[1] in other['class_vis']),
                        None)
        if mismatch is not None:
            mis_other_idx = concept_indices(mismatch, dataset)[1]
            entry['visual_cross_mismatch'] = _summarise(
                probe_c(other['class_vis'][mis_other_idx]), null_c, len(pool_c), rng)
            entry['mismatch_concept'] = mismatch
        out[name] = entry
    return out


def global_transfer(prep, preps, seed):
    """Does the other domain's GLOBAL anomaly-vs-normal direction work here?"""
    dataset = prep['dataset']
    rng = np.random.RandomState(seed + 33)
    other = preps[OTHER[dataset]]
    probe_all, null_all = prep['probe_all'], prep['null_all']
    out = {'own_all': _summarise(probe_all(prep['global_vis']), null_all,
                                 len(prep['pool_all']), rng),
           'cross_all': _summarise(probe_all(other['global_vis']), null_all,
                                   len(prep['pool_all']), rng),
           'cos_global_cross_descriptive': _cos(prep['global_vis'], other['global_vis'])}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-random', type=int, default=200,
                    help='random-direction null draws per pool_all probe')
    ap.add_argument('--n-random-class', type=int, default=100,
                    help='random-direction null draws per class-pool probe')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-videos', type=int, default=None,
                    help='cap per dataset (smoke testing only)')
    ap.add_argument('--min-class-videos', type=int, default=4)
    ap.add_argument('--min-class-frames', type=int, default=200)
    ap.add_argument('--out-dir', default=None,
                    help='output dir override; default runs/ - NEVER smoke over runs/')
    args = ap.parse_args()

    print('building frozen CLIP text banks (cpu, vendored fork)...', flush=True)
    banks, banks_info = {}, {}
    for ds in DATASETS:
        dirs, info = build_class_directions('ViT-B/16', ds, device='cpu')
        banks[ds] = dirs.numpy().astype(np.float64)
        banks_info[ds] = {'class_direction_mutual_cos': info['class_direction_mutual_cos'],
                          'num_class': info['num_class']}
        print(f'  [{ds}] bank built: {info["num_class"]} classes', flush=True)

    print('preparing pools / nulls per dataset...', flush=True)
    preps = {}
    for ds in DATASETS:
        preps[ds] = prepare(ds, args.n_random, args.n_random_class, args.seed,
                            args.min_class_videos, args.min_class_frames,
                            max_videos=args.max_videos)
        p = preps[ds]
        if p is None:
            print(f'  [{ds}] NO mixed-GT test videos - dataset skipped', flush=True)
            continue
        print(f'  [{ds}] mixed-GT videos={len(p["pool_all"])} '
              f'class pools={sorted(p["class_pools"])} skipped={p["skip"]}', flush=True)

    out = {'args': vars(args),
           'env': {'python': platform.python_version(),
                   'numpy': np.__version__, 'machine': platform.machine()},
           'banks_info': banks_info}
    for ds in DATASETS:
        p = preps[ds]
        if p is None:
            out[ds] = {'ok': False, 'reason': 'no mixed-GT test videos'}
            continue
        out[ds] = {
            'n_mixed_gt_videos': len(p['pool_all']),
            'skipped': p['skip'],
            'class_pools': p['class_meta'],
            'pool_all_null': {'mean': round(_f(p['null_all'].mean()), 4),
                              'p95': round(_f(np.percentile(p['null_all'], 95)), 4),
                              'n_draws': int(args.n_random)},
            'text_transfer': text_transfer(p, banks, args.seed),
            'visual_transfer': visual_transfer(p, preps, args.seed),
            'global_transfer': global_transfer(p, preps, args.seed),
        }

    out_dir = args.out_dir or os.path.join(opt.repo_root(), 'runs')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'stage0_direction_transfer.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'saved -> {path}', flush=True)

    print('\n' + '=' * 78)
    print('STAGE 0 / H1 SUMMARY (within-video probe, free sign, matched nulls)')
    print('=' * 78)
    for ds in DATASETS:
        r = out[ds]
        print(f'\n[{ds}]  mixed-GT videos={r["n_mixed_gt_videos"]}  '
              f'pool null mean={r["pool_all_null"]["mean"]} p95={r["pool_all_null"]["p95"]}')
        g = r['global_transfer']
        print(f'  global visual dir: own AUC={g["own_all"].get("auc_mean")} '
              f'cross AUC={g["cross_all"].get("auc_mean")} '
              f'(null p95={r["pool_all_null"]["p95"]}, '
              f'cos={g["cos_global_cross_descriptive"]})')
        for name, e in r['visual_transfer'].items():
            if 'visual_cross_matched' not in e:
                print(f'  R2 {name:<10} SKIPPED ({e.get("skipped", "?")})')
                continue
            own = e['visual_own_matched'].get('auc_mean')
            cross = e['visual_cross_matched'].get('auc_mean')
            mis = e.get('visual_cross_mismatch', {}).get('auc_mean')
            print(f'  R2 {name:<10} own={own} cross={cross} mismatch={mis} '
                  f'cost={e.get("transfer_cost_own_minus_cross")} '
                  f'cos={e.get("cos_visual_cross_descriptive")}')
        for name, e in r['text_transfer'].items():
            ca = e.get('text_cross_all', {}).get('auc_mean')
            cm = e.get('text_cross_matched', {}).get('auc_mean')
            print(f'  R1 {name:<10} cross_all={ca} cross_matched={cm} '
                  f'cos_banks={e.get("cos_text_own_vs_cross_descriptive")}')
    print("""
  Read: R2 cross > own-dataset null p95 => anomaly geometry is domain-shared
  (the plan's premise holds at feature level). R1 cross_matched close to
  text_own_matched => the text bank is portable. Cosines are descriptive only -
  AGENTS.md: they cannot separate a useful direction from a useless one.""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
