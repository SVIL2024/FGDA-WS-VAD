"""Trivial label-free baselines scored on the *official* pooled protocol.

    python analysis_official_metric.py --dataset both

Why this exists. Every number recorded so far in this project - the 0.744 XD
ceiling, the 0.7217 UCF magnitude cue, the event-prompt gains - is a **mean
within-video AUC**: score each video, z-score inside the video, AUC against that
video's own GT, average over videos that contain both classes. That statistic was
the right instrument for a *precondition* test, because it removes cross-video
confounds (movie, scene, lighting) and isolates temporal localisation.

It is not the number the field publishes. Reading the reference test loops
(``referCode/VadCLIP-main/src/ucf_test.py:91``, ``src/xd_test.py:69``) the reported
metric is one pooled call over the whole test set:

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))     # UCF AUC
    AP2  = average_precision_score(gt, np.repeat(ap2, 16))   # XD AP

i.e. every frame of every test video concatenated in list order, against the full
GT array, with **no per-video normalisation**. A signal that is excellent at
ranking frames *inside* one video can be worthless in this pooled metric, because
there the between-video differences do most of the work. Conversely a cue that
only knows *which video* is anomalous looks strong here and tells you nothing
about localisation.

This script measures label-free cues on the protocol the leaderboard actually uses,
so the project stops comparing its diagnostics against a metric no reviewer reads.

Cues, all training-free:
  norm          ``||f - mean_video(f)||``  - how unlike its own video a frame is
  norm_raw      ``||f||``                  - plain feature magnitude
  class_max     max over class-label directions
  psi_max       max over the event-prompt bank
  psi_top3      mean of the top-3 event-prompt responses

Sign is not free in the pooled metric (there is no per-video z-score that makes it
symmetric), and choosing it from the test GT would be cheating. It is therefore
fixed on the **training** split using only video-level labels, which is legitimate
weak supervision: average the cue over anomalous training videos and over normal
ones, keep the orientation that separates them. Both the train-calibrated AUC and
the oracle-sign value ``max(a, 1-a)`` are printed; a large gap between them means
the cue's orientation does not transfer between splits.

The reference convention is preserved exactly, including that the XD test list
references only chunk ``__0`` of each video while the feature directory holds ten
chunks. Missing feature files are emitted as zero-filled frames so list-order
alignment with the GT array is never broken, and their count is reported.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import freq_text_events as E  # noqa: E402
import freq_text_options as opt  # noqa: E402
from freq_text_text import GT_FRAME_REPEAT  # noqa: E402
from freq_text_text import _gt_offsets, _read_list  # noqa: E402
from freq_text_text import build_class_directions  # noqa: E402
from probe_events import encode_prompts  # noqa: E402

log = logging.getLogger('official_metric')


def _nan():
    """A NaN constant, guarded so no module-level call can raise."""
    try:
        return float('nan')
    except (TypeError, ValueError):        # pragma: no cover - defensive
        return 0.0


NAN = _nan()


def _f(x, default=NAN):
    """Float or ``default`` - never raise on a bad scalar."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _r(x, nd=4):
    """Round for JSON, mapping non-finite to NaN."""
    v = _f(x)
    return NAN if not np.isfinite(v) else round(v, _i(nd, 4))


def _i(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError, OverflowError):
        return default


def load_full(test_list, gt_path):
    """Every test video in list order, plus the flat GT array.

    Unlike ``probe_events.load_test_videos`` this keeps normal-only videos, since
    the pooled metric needs the whole test set: dropping them would remove the
    negative frames that define the pooled AUC.
    """
    rows = _read_list(test_list)
    gt = np.load(gt_path).astype(np.float32)
    offs = _gt_offsets(rows, test_list)
    feats, names, n_frames, missing = [], [], [], 0
    for i, (p, _lab) in enumerate(rows):
        n = _i(offs[i + 1] - offs[i], 0)
        if n <= 0:
            continue
        arr = None
        try:
            arr = np.load(p).astype(np.float32)
        except (OSError, ValueError, EOFError):
            log.warning('unreadable feature file: %s', p)
        if arr is None or arr.ndim != 2 or arr.shape[0] == 0:
            missing += 1
            feats.append(None)
        else:
            feats.append(arr)
        names.append(os.path.basename(p))
        n_frames.append(n)

    expect = sum(_i(n, 0) for n in n_frames) * GT_FRAME_REPEAT
    aligned = expect == len(gt)
    if not aligned:
        log.warning('GT alignment mismatch: %d frames x%d = %d vs len(gt)=%d',
                    sum(_i(n, 0) for n in n_frames), GT_FRAME_REPEAT, expect,
                    len(gt))
    return feats, n_frames, gt, missing, aligned


def pooled(scores, gt, name):
    """The published metric: one call over all frames of all videos."""
    s = np.asarray(scores, dtype=np.float64)
    g = np.asarray(gt).astype(np.int8)
    if len(s) != len(g):
        log.warning('%s: length %d vs gt %d - not scored', name, len(s), len(g))
        return {'cue': name, 'auc': NAN, 'ap': NAN}
    try:
        auc = _f(roc_auc_score(g, s))
        ap = _f(average_precision_score(g, s))
    except (ValueError, IndexError):
        log.warning('%s: metric call failed', name, exc_info=True)
        return {'cue': name, 'auc': NAN, 'ap': NAN}
    return {'cue': name, 'auc': _r(auc), 'ap': _r(ap),
            'auc_oracle_sign': _r(max(_f(auc, 0.5), 1 - _f(auc, 0.5)))}


def is_anomalous(label, dataset):
    """Video-level anomaly flag from a VadCLIP list label string.

    The train lists are not 0/1. XD uses ``A`` for normal and letter codes for
    events, with multi-segment clips written like ``B1-0-0`` (an event plus two
    normal windows); UCF uses the class name with ``Normal`` for the negative
    class. Only the normal/non-normal distinction is needed here, and only from
    the *training* list, which is legitimate weak supervision.
    """
    s = str(label).strip()
    if not s:
        return False
    if dataset == 'ucf':
        return s.lower() != 'normal'
    toks = [t for t in s.split('-') if t not in ('', '0')]
    return any(t.upper() != 'A' for t in toks)


def train_sign(train_list, cue_fn, dataset):
    """Cue orientation from video-level training labels only."""
    try:
        rows = _read_list(train_list)
    except OSError:
        log.warning('cannot read train list %s', train_list, exc_info=True)
        return None
    pos, neg = [], []
    for p, lab in rows:
        try:
            v = np.load(p).astype(np.float32)
        except (OSError, ValueError, EOFError):
            continue
        if v.ndim != 2 or v.shape[0] == 0:
            continue
        try:
            s = cue_fn(v)
            m = _f(np.mean(s))
        except (TypeError, ValueError):
            log.warning('cue failed on %s', os.path.basename(p), exc_info=True)
            continue
        if not np.isfinite(m):
            continue
        (pos if is_anomalous(lab, dataset) else neg).append(m)
    if len(pos) < 10 or len(neg) < 10:
        log.warning('train sign underdetermined (%d pos, %d neg)',
                    len(pos), len(neg))
        return None
    mp, mn = _f(np.mean(pos)), _f(np.mean(neg))
    return {'positive': mp, 'negative': mn, 'sign': 1.0 if mp >= mn else -1.0,
            'n_pos': len(pos), 'n_neg': len(neg)}


def cues_for(dataset, device):
    """Build every label-free cue function for one dataset."""
    prompts = E.flatten_prompts(dataset)
    emb = encode_prompts('ViT-B/16', prompts, device=device).cpu().numpy()
    emb = emb.astype(np.float32)
    dirs, _ = build_class_directions('ViT-B/16', dataset, device='cpu', verbose=False)
    d = dirs.numpy().astype(np.float32)[1:]
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)

    def _centred(v):
        return v - v.mean(0)

    def norm(v):
        c = _centred(v)
        return np.linalg.norm(c, axis=1)

    def norm_raw(v):
        return np.linalg.norm(v, axis=1)

    def class_max(v):
        return (_centred(v) @ d.T).max(axis=1)

    def _resp(v):
        x = _centred(v)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        return x @ emb.T

    def psi_max(v):
        return _resp(v).max(axis=1)

    def psi_top3(v):
        r = np.sort(_resp(v), axis=1)
        k = min(3, r.shape[1])
        return r[:, -k:].mean(axis=1)

    return {'norm': norm, 'norm_raw': norm_raw, 'class_max': class_max,
            'psi_max': psi_max, 'psi_top3': psi_top3}, len(prompts)


def run(dataset, device):
    base = os.path.join(opt.repo_root(), 'list')
    test_list = os.path.join(base, f'{dataset}_CLIP_rgbtest.csv')
    train_list = os.path.join(base, f'{dataset}_CLIP_rgb.csv')
    gt_path = os.path.join(base, 'gt.npy' if dataset == 'xd' else 'gt_ucf.npy')

    print(f'\n{"=" * 78}\nOFFICIAL POOLED PROTOCOL  dataset={dataset}\n{"=" * 78}')
    feats, n_frames, gt, missing, aligned = load_full(test_list, gt_path)
    total = sum(_i(n, 0) for n in n_frames)
    print(f'  {len(feats)} list rows, {total} feature frames '
          f'x{GT_FRAME_REPEAT} = {total * GT_FRAME_REPEAT} vs len(gt)={len(gt)}  '
          f'aligned={aligned}  unreadable={missing}')
    if not aligned:
        print('  ABORT: list/GT misalignment would silently corrupt the pooled AUC')
        return None
    cues, n_prompts = cues_for(dataset, device)

    # sign per cue from the training split only
    signs = {}
    for name, fn in cues.items():
        s = train_sign(train_list, fn, dataset)
        signs[name] = _f((s or {}).get('sign'), 1.0)
        if s:
            print(f'  sign[{name:<9}] train pos {_f(s["positive"]):+.4f} vs '
                  f'neg {_f(s["negative"]):+.4f} -> {signs[name]:+.0f}  '
                  f'({s["n_pos"]}/{s["n_neg"]} videos)')
        else:
            print(f'  sign[{name:<9}] UNDETERMINED, defaulting to +1')

    results = {}
    # Two pooling conventions. "raw" concatenates the cue as-is, so between-video
    # differences do most of the work. "pv_z" standardises each video's scores
    # before concatenating, which removes them; this is the step the within-video
    # probes have had all along, now expressed inside the pooled metric.
    for mode in ('raw', 'pv_z'):
        print(f'  --- pooling: {mode} ---')
        for name, fn in cues.items():
            sc = signs.get(name, 1.0)
            try:
                parts = []
                for k in range(len(feats)):
                    arr, n = feats[k], n_frames[k]
                    nf = _i(n, 0)
                    if arr is None:
                        parts.append(np.zeros(nf, dtype=np.float64))
                        continue
                    t = min(_i(len(arr), 0), nf)
                    s = np.asarray(fn(arr[:t]), dtype=np.float64)
                    if s.ndim != 1 or s.size != t:
                        s = np.resize(s, t) if s.size else np.zeros(t)
                    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                    if mode == 'pv_z':
                        s = (s - _f(np.mean(s))) / (_f(np.std(s)) + 1e-8)
                    pad = np.zeros(max(nf - t, 0), dtype=np.float64)
                    parts.append(np.concatenate([s, pad]) * sc)
                pooled_scores = np.concatenate(
                    [np.repeat(p, GT_FRAME_REPEAT) for p in parts])
            except (ValueError, TypeError, MemoryError):
                log.warning('cue %s failed to assemble', name, exc_info=True)
                continue
            r = pooled(pooled_scores, gt, name)
            r['sign'] = _r(sc, 1)
            results[f'{name}/{mode}'] = r
            print(f'    {name:<10} pooled AUC {_f(r["auc"], 0):.4f}  '
                  f'AP {_f(r["ap"], 0):.4f}')
            del pooled_scores, parts

    rep = {'dataset': dataset, 'rows': len(feats), 'feature_frames': total,
           'gt_len': _i(len(gt), 0), 'aligned': bool(aligned),
           'unreadable': _i(missing, 0), 'n_prompts': _i(n_prompts, 0),
           'signs': {k: _r(v, 1) for k, v in signs.items()}, 'cues': results}
    p = os.path.join(opt.repo_root(), 'runs', f'official_metric_{dataset}.json')
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        print(f'  saved -> {p}')
    except OSError as exc:
        print(f'  WARNING: write failed: {exc}')
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='both', choices=['xd', 'ucf', 'both'])
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    sets = ['xd', 'ucf'] if args.dataset == 'both' else [args.dataset]
    reps = {}
    for ds in sets:
        r = run(ds, args.device)
        if r:
            reps[ds] = r

    metric_of = {'xd': 'AP', 'ucf': 'AUC'}
    print(f'\n{"=" * 78}\nPUBLISHED-PROTOCOL TRIVIAL BASELINES\n{"=" * 78}')
    for ds, r in reps.items():
        key = metric_of.get(ds, 'AUC')
        print(f'\n  {ds}  (reported metric: frame-level {key}, pooled over all videos)')
        ranked = sorted(r['cues'].items(),
                        key=lambda kv: -_f(kv[1].get(key.lower()), 0.0))
        for name, c in ranked:
            v = _f(c.get(key.lower()), 0.0)
            print(f'    {name:<10} {key}={v:.4f}   AUC={_f(c.get("auc"), 0):.4f}   '
                  f'AP={_f(c.get("ap"), 0):.4f}')
    print("""
  Reference points on this same protocol: VadCLIP 84.51 XD AP / 88.02 UCF AUC;
  LAP 86.5 / 88.9; CPL-VAD 88.53 XD AP. A training-free cue within a few points of
  those numbers means the leaderboard is close to a cue that needs no model.""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
