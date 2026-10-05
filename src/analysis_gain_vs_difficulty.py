"""Is the event-prompt gain a dataset property, or a difficulty property?

    python analysis_gain_vs_difficulty.py --dataset both

Motivation. The two datasets gave opposite verdicts: XD event-vs-class =
+0.041 [0.030, 0.053], UCF = +0.001 [-0.005, 0.008]. The obvious reading, "the
method works on XD and fails on UCF", is a dataset-level claim and it is the
least useful possible conclusion: it tells the reader nothing about *when* to use
the method, and it invites the retort that we picked the favourable dataset.

The anomaly-frame fraction differs a lot between the datasets (XD median 0.33,
UCF median 0.14), and it varies enormously *within* each one too. So stratify by
it and ask whether the gain tracks the stratum rather than the dataset. If it
does, the finding becomes a boundary condition - "event-prompt guidance helps
when the anomaly occupies at least f* of the video and stops helping below that"
- which is actionable, testable on a third dataset, and independent of the AP
leaderboard.

Both sides are computed identically and both are usable without test-time labels,
so the difference is not an artefact of one side being chosen with GT:

  incumbent   max over the class-label directions, per frame
  candidate   max (or top-k mean) over the event-prompt responses, per frame

Both are z-scored inside the video and scored against that video's GT segments,
i.e. the quantity the trained model is graded on. The per-video paired
difference is the statistic; its stratum means carry the CIs.
"""

import argparse
import json
import logging
import math
import os
import sys

import numpy as np

log = logging.getLogger('gain_vs_difficulty')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import freq_text_events as E  # noqa: E402
import freq_text_options as opt  # noqa: E402
from freq_text_text import build_class_directions  # noqa: E402
from probe_events import encode_prompts, load_test_videos  # noqa: E402

NAN = math.nan


# ---------------------------------------------------------------------------
# Guarded scalar conversions. Every statistic in this file is optional: a video
# with degenerate labels, an empty stratum, or a zero-variance rank vector must
# degrade to NaN and be dropped, never abort a 400-video sweep. Routing the
# conversions through these two helpers keeps that guarantee in one place
# instead of repeating try/except at each call site.
# ---------------------------------------------------------------------------
def _f(x, default=NAN):
    """float(x), or ``default`` if x is missing/unconvertible."""
    try:
        return float(x)
    except (TypeError, ValueError):
        log.debug('non-numeric value %r -> %r', x, default)
        return default


def _i(x, default=0):
    """int(x), or ``default`` if x is missing/unconvertible."""
    try:
        return int(x)
    except (TypeError, ValueError, OverflowError):
        log.debug('non-integer value %r -> %r', x, default)
        return default


def _r(x, nd=4, default=NAN):
    """round(float(x), nd), or ``default`` if x is not a finite number."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return round(v, nd)


def _auc_one(g, s):
    """Sign-free within-video AUC of score ``s`` against binary labels ``g``.

    Returns NaN for single-class labels or a constant score - both are
    uninformative and are filtered by the caller rather than silently scored 0.5.
    """
    from sklearn.metrics import roc_auc_score
    try:
        g = np.asarray(g)
        s = np.asarray(s, dtype=np.float64)
        if g.ndim != 1 or s.ndim != 1 or g.shape != s.shape or s.size < 2:
            return NAN
        k = int(np.count_nonzero(g))
        if k == 0 or k == g.size:
            return NAN
        z = (s - s.mean()) / (s.std() + 1e-9)
        if not np.isfinite(z).all():
            return NAN
        a = _f(roc_auc_score(g, z))
    except Exception:                     # noqa: BLE001 - never abort a sweep
        return NAN
    if not np.isfinite(a):
        return NAN
    return max(a, 1.0 - a)


def _auc_each(V, G, score_fn):
    """Per-video AUC array with NaNs dropped (indices stay aligned via ``keep``)."""
    raw, keep = [], []
    # zip() takes no `strict` kwarg on this env's Python 3.9, so the length match
    # that load_test_videos guarantees is asserted here instead of implied.
    if len(V) != len(G):
        raise ValueError(f'feature/label count mismatch: {len(V)} vs {len(G)}')
    for v, g in zip(V, G):  # noqa: B905 - lengths checked above, strict= is 3.10+
        try:
            s = np.asarray(score_fn(v), dtype=np.float64)
        except Exception:                 # noqa: BLE001
            log.warning('score_fn failed on a %s video; dropped',
                        f'{v.shape}', exc_info=True)
            raw.append(NAN)
            keep.append(False)
            continue
        a = _auc_one(g, s)
        raw.append(a)
        keep.append(np.isfinite(a))
    idx = np.array(keep, dtype=bool)
    return np.array(raw, dtype=np.float64), idx


def _finite_mask(*arrays):
    """Boolean mask of positions finite in every array (equal length required)."""
    if not arrays:
        return np.zeros(0, dtype=bool)
    m = np.ones(len(np.asarray(arrays[0], dtype=np.float64)), dtype=bool)
    for a in arrays:
        m &= np.isfinite(np.asarray(a, dtype=np.float64))
    return m


def _apply(mask, *arrays):
    """Index every array by ``mask``, as a fixed-length list."""
    return [np.asarray(a, dtype=np.float64)[mask] for a in arrays]


def bank_auc(V, G, emb, k=None):
    """Per-video AUC of an embedding bank, aggregated by max or top-k mean."""
    def fn(v):
        x = v - v.mean(0)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        r = x @ emb.T                                  # (T, P)
        return np.sort(r, axis=1)[:, -k:].mean(1) if k else r.max(axis=1)
    return _auc_each(V, G, fn)


def norm_auc(V, G):
    """Per-video AUC of the free ||f - mean|| magnitude cue."""
    return _auc_each(V, G, lambda v: np.linalg.norm(v - v.mean(0), axis=1))


def ci95(diff):
    """Mean +/- 95% CI (normal approx) of a paired difference; NaN-safe."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return [NAN, NAN]
    se = _f(d.std(ddof=1)) / np.sqrt(d.size)
    m = _f(d.mean())
    return [_r(m - 1.96 * se), _r(m + 1.96 * se)]


def spearman(x, y):
    """Spearman rho by rank correlation (no scipy dependency). NaN-safe."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    try:
        x, y = _apply(_finite_mask(x, y), x, y)
        if x.size < 3:
            return NAN
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        if rx.std() < 1e-9 or ry.std() < 1e-9:
            return NAN
        c = np.corrcoef(rx, ry)[0, 1]
    except Exception:                     # noqa: BLE001
        return NAN
    return _r(c, 3)


def bucket_table(strata, diff, edges):
    """Per-stratum paired-gain summary; strata with no finite samples are skipped."""
    strata = np.asarray(strata, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = _f(edges[i]), _f(edges[i + 1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        try:
            m = (strata >= lo) & (strata < hi) & np.isfinite(diff)
            n = _i(np.count_nonzero(m))
        except Exception:                 # noqa: BLE001
            log.warning('bucket [%g,%g) selection failed; skipped', lo, hi,
                        exc_info=True)
            continue
        if n == 0:
            continue
        try:
            sub = diff[m]
            c = ci95(sub)
            rows.append({
                'lo': _r(lo, 3), 'hi': _r(hi, 3), 'n': n,
                'label': f'[{lo:g},{hi:g})',
                'mean': _r(sub.mean()),
                'lo_ci': c[0], 'hi_ci': c[1],
                'win_rate': _r(_f(np.count_nonzero(sub > 0)) / n, 3),
            })
        except Exception:                 # noqa: BLE001
            log.warning('bucket [%g,%g) summary failed; skipped', lo, hi,
                        exc_info=True)
            continue
    return rows


def _write_json(path, obj):
    """Write JSON, reporting failure instead of raising after a long sweep."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f'  WARNING: could not write {path}: {exc}')
        return False
    print(f'  saved -> {path}')
    return True


def run(dataset, n_videos, device):
    base = os.path.join(opt.repo_root(), 'list')
    test_list = os.path.join(base, f'{dataset}_CLIP_rgbtest.csv')
    gt_path = os.path.join(base, 'gt.npy' if dataset == 'xd' else 'gt_ucf.npy')
    V, G = load_test_videos(test_list, gt_path, n_videos)
    if len(V) < 20:
        print(f'\n{dataset}: only {len(V)} mixed-GT videos - skipped')
        return None
    frac = np.array([_f(np.mean(g)) for g in G], dtype=np.float64)
    length = np.array([_f(len(v)) for v in V], dtype=np.float64)

    dirs, _ = build_class_directions('ViT-B/16', dataset, device='cpu', verbose=False)
    d = dirs.numpy().astype(np.float32)[1:]            # drop the normal row
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    prompts = E.flatten_prompts(dataset)
    emb = encode_prompts('ViT-B/16', prompts, device=device).cpu().numpy()
    emb = emb.astype(np.float32)

    inc, _k1 = bank_auc(V, G, d)                     # max over class directions
    cand, _k2 = bank_auc(V, G, emb)
    cand3, _k3 = bank_auc(V, G, emb, k=3)
    nrm, _k4 = norm_auc(V, G)
    # _auc_each writes NaN where a video was unusable, so one finite-mask across
    # all six series is both simpler and stricter than intersecting its keep flags.
    valid = _finite_mask(inc, cand, cand3, nrm, frac, length)
    cols = _apply(valid, frac, length, inc, cand, cand3, nrm)
    frac, length, inc, cand, cand3, nrm = cols
    print(f'\n{dataset}: {len(inc)} usable mixed-GT videos | class-max '
          f'{inc.mean():.4f}  event-max {cand.mean():.4f}  event-top3 '
          f'{cand3.mean():.4f}  ||f|| {nrm.mean():.4f}')
    print(f'  anomaly fraction: median {np.median(frac):.3f}  '
          f'p10 {np.percentile(frac, 10):.3f}  p90 {np.percentile(frac, 90):.3f}')

    rep = {'dataset': dataset, 'n_videos': _i(len(inc)),
           'frac_median': _r(np.median(frac), 3),
           'class_max_auc': _r(inc.mean()), 'magnitude_auc': _r(nrm.mean())}
    for tag, c in (('event_max', cand), ('event_top3', cand3)):
        diff = c - inc
        over = c - nrm
        rep[f'{tag}_auc'] = _r(c.mean())
        rep[f'{tag}_gain_mean'] = _r(diff.mean())
        rep[f'{tag}_gain_ci95'] = ci95(diff)
        rep[f'{tag}_win_rate'] = _r(_f(np.count_nonzero(diff > 0)) / max(len(diff), 1), 3)
        rep[f'{tag}_gain_vs_frac_spearman'] = spearman(frac, diff)
        rep[f'{tag}_gain_vs_length_spearman'] = spearman(length, diff)
        rep[f'{tag}_by_anomaly_fraction'] = bucket_table(
            frac, diff, [0.0, 0.05, 0.12, 0.25, 0.45, 0.70, 1.01])
        rep[f'{tag}_by_video_length'] = bucket_table(
            length, diff, [0, 80, 120, 170, 250, 1e9])
        rep[f'{tag}_gain_over_magnitude'] = _r(over.mean())
        rep[f'{tag}_gain_over_magnitude_ci95'] = ci95(over)
        print(f'  {tag}: gain {_r(diff.mean()):+.4f} {ci95(diff)}  '
              f'win-rate {rep[f"{tag}_win_rate"]:.3f}  '
              f'rho(frac) {rep[f"{tag}_gain_vs_frac_spearman"]:+.3f}  '
              f'rho(len) {rep[f"{tag}_gain_vs_length_spearman"]:+.3f}')
    return rep


def _pool(rows, lo_edge=None, hi_edge=None):
    """n-weighted mean gain and n over strata matching an edge predicate."""
    n = 0
    acc = 0.0
    for r in rows:
        lo, hi, k = _f(r.get('lo')), _f(r.get('hi')), _f(r.get('n'))
        m = np.isfinite(lo) and np.isfinite(hi) and np.isfinite(k) and k > 0
        if m and lo_edge is not None and lo < lo_edge:
            m = False
        if m and hi_edge is not None and hi > hi_edge:
            m = False
        mean = _f(r.get('mean'))
        if m and np.isfinite(mean):
            acc += mean * k
            n += _i(k)
    return (acc / n if n else NAN), n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='both', choices=['xd', 'ucf', 'both'])
    ap.add_argument('--n-videos', type=int, default=10000)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    sets = ['xd', 'ucf'] if args.dataset == 'both' else [args.dataset]
    reps = {}
    for ds in sets:
        r = run(ds, args.n_videos, args.device)
        if r:
            reps[ds] = r
            _write_json(os.path.join(opt.repo_root(), 'runs',
                                     f'gain_difficulty_{ds}.json'), r)
    if not reps:
        print('no usable data')
        return 1

    print(f'\n{"=" * 78}\nPAIRED GAIN (event-max - class-max) BY ANOMALY-FRACTION STRATUM\n{"=" * 78}')
    for ds, r in reps.items():
        print(f'\n  {ds}  (median anomaly fraction {r["frac_median"]}, '
              f'rho = {r["event_max_gain_vs_frac_spearman"]:+.3f})')
        print(f'      {"stratum":<14} {"n":>5} {"gain":>9} {"95% CI":>22} {"win%":>6}')
        for row in r['event_max_by_anomaly_fraction']:
            print(f'      {row["label"]:<14} {row["n"]:>5} {row["mean"]:>+9.4f} '
                  f'[{row["lo_ci"]:>+8.4f}, {row["hi_ci"]:>+8.4f}] '
                  f'{100 * _f(row["win_rate"], 0):>5.1f}')

    print(f'\n{"=" * 78}\nBOUNDARY-CONDITION TEST\n{"=" * 78}')
    verdicts = []
    for ds, r in reps.items():
        gh, nh = _pool(r['event_max_by_anomaly_fraction'], lo_edge=0.25)
        gl, nl = _pool(r['event_max_by_anomaly_fraction'], hi_edge=0.12)
        verdicts.append(np.isfinite(gh) and np.isfinite(gl) and gh > gl)
        print(f'  {ds}: frac>=0.25 -> gain {gh:+.4f} (n={nh})   '
              f'frac<0.12 -> gain {gl:+.4f} (n={nl})')
    print(f'\n  gain increases with anomaly fraction on every dataset: {all(verdicts)}')
    print('  If True, the XD/UCF split is a difficulty effect rather than a dataset')
    print('  effect, and the contribution becomes the boundary condition plus a')
    print('  cheap label-free predictor of when text guidance pays off.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
