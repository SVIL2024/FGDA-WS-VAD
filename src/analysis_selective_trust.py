"""Can the model know, without labels, when text guidance is trustworthy?

    python analysis_selective_trust.py --dataset both

Background. ``analysis_gain_vs_difficulty.py`` rejected the predicted boundary
condition: rho(anomaly_fraction, gain) = -0.046 on XD and +0.155 on UCF, opposite
signs and both near zero. Both of its candidate predictors were functions of the
same ground truth being scored, so they were poor predictors by construction.

What *is* available at inference time is the response itself. This asks the
question the failed test leaves open, and it is answerable label-free:

  is a video whose event-prompt response is sharply structured one where the text
  signal is actually helping?

Four confidence statistics, each computed from the test video alone - no labels,
no training, no knowledge of the answer:

  spread       std over frames of the best-prompt response. A flat response means
               no frame matches any event, so the induced ranking is arbitrary.
  margin       mean over frames of (best prompt - runner-up). A small margin means
               the winning prompt is decided by noise.
  event_share  fraction of frames whose best event response exceeds that video's
               own best NORMAL-prompt response. How much of the video looks like a
               named event rather than like ordinary activity.
  agree        rank correlation between the text ranking and the free magnitude
               cue. High agreement means the two cues tell one story.

Videos are bucketed per quartile of each statistic and the reported quantity is
the per-bucket **win rate** of event-max over class-max, alongside the paired
gain. Win rate is the headline because the mean misleads here: XD event-max gains
+0.0230 with a CI excluding zero yet wins on only 54.9% of videos, so the average
is carried by a minority of cases. A predictor with a monotone win-rate gradient
would let the method apply text guidance only where it earns its keep - a
reliability contribution that does not depend on beating an AP leaderboard.
"""

import argparse
import json
import logging
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import freq_text_events as E  # noqa: E402
import freq_text_options as opt  # noqa: E402
from analysis_gain_vs_difficulty import (_auc_each, _f, _i, _r, bank_auc, ci95,  # noqa: E402
                                         norm_auc, spearman)
from freq_text_text import build_class_directions  # noqa: E402
from probe_events import encode_prompts, load_test_videos  # noqa: E402

log = logging.getLogger('selective_trust')
NAN = math.nan
STATS = ('spread', 'margin', 'event_share', 'agree')


def _vmean(v):
    """Per-video mean feature. This centring is the probes' LOCALISATION
    convention; VadCLIP's own test loop applies no per-video normalisation at
    all - its published AP/AUC pool every frame (analysis_official_metric.py
    measures both protocols for the same cues)."""
    return v.mean(0)


def _max_proj(v, d):
    """Max over a direction bank of the centred per-frame projection."""
    x = v - _vmean(v)
    return (x @ d.T).max(axis=1)


def _responses(v, emb):
    """(T, P) cosine responses of one video's frames against a prompt bank."""
    x = v - _vmean(v)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return x @ emb.T


def confidence_stats(v, emb, normal_cols):
    """Label-free confidence statistics for one video's prompt response.

    ``normal_cols`` indexes the bank's normal-class prompts, used as a per-video
    baseline. A previous version collapsed those embeddings into one global scalar
    mean and compared cosine responses against it, which is not the same quantity:
    the mean of L2-normalised text vectors is not itself a response magnitude.
    The baseline must be a response, so it is taken per video here.
    """
    out = dict.fromkeys(STATS, NAN)
    try:
        r = _responses(v, emb)
        if r.ndim != 2 or r.shape[0] < 3:
            return out
        s = np.sort(r, axis=1)
        top, second = s[:, -1], s[:, -2]
        out['spread'] = _r(np.std(top), 6)
        out['margin'] = _r(np.mean(top - second), 6)
        if normal_cols is not None and len(normal_cols):
            floor = r[:, normal_cols].max(axis=1)     # (T,) best normal response
            beats = _f(np.count_nonzero(top > floor)) / max(len(top), 1)
        else:
            beats = NAN
        out['event_share'] = _r(beats, 6)
        out['agree'] = spearman(top, np.linalg.norm(v - _vmean(v), axis=1))
    except Exception:                     # noqa: BLE001 - one bad video must not end a sweep
        log.warning('confidence_stats failed on one video', exc_info=True)
    return out


def normal_columns(dataset, owners):
    """Indices of the bank's normal-class prompts, or None if none are labelled."""
    names = E.class_names(dataset)
    try:
        cols = [i for i, o in enumerate(owners)
                if names[_i(o, 0)].lower().startswith('normal')]
    except (IndexError, TypeError, ValueError):
        log.warning('could not identify normal prompts for %s', dataset,
                    exc_info=True)
        return None
    return cols or None


def quartile_table(x, win, gain, q=4):
    """Per-quartile win rate and paired gain of a candidate predictor."""
    rows = []
    try:
        edges = np.percentile(np.asarray(x, dtype=np.float64),
                              np.linspace(0, 100, q + 1))
    except (TypeError, ValueError):
        log.warning('percentiles failed for a predictor; skipped', exc_info=True)
        return rows
    arr = np.asarray(x, dtype=np.float64)
    win = np.asarray(win, dtype=bool)
    gain = np.asarray(gain, dtype=np.float64)
    for i in range(_i(q)):
        lo, hi = _f(edges[i]), _f(edges[i + 1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        try:
            m = (arr >= lo) & ((arr <= hi) if i == q - 1 else (arr < hi))
            m = np.asarray(m, dtype=bool) & np.isfinite(gain)
            n = _i(np.count_nonzero(m))
            if n < 5:
                continue
            c = ci95(gain[m])
            rows.append({'lo': _r(lo, 5), 'hi': _r(hi, 5), 'n': n,
                         'win_rate': _r(_f(np.count_nonzero(win[m])) / n, 3),
                         'gain': _r(np.mean(gain[m])),
                         'lo_ci': c[0], 'hi_ci': c[1]})
        except (TypeError, ValueError):
            log.warning('bucket [%g,%g) failed; skipped', lo, hi, exc_info=True)
    return rows


def monotone(rows):
    """+1 if win rate rises across buckets, -1 if it falls, 0 if non-monotone."""
    w = [_f(r.get('win_rate')) for r in rows]
    w = [x for x in w if np.isfinite(x)]
    if len(w) < 3:
        return 0
    d = np.diff(np.array(w, dtype=np.float64))
    if bool(np.all(d >= -1e-9)):
        return 1
    if bool(np.all(d <= 1e-9)):
        return -1
    return 0


def run(dataset, n_videos, device):
    base = os.path.join(opt.repo_root(), 'list')
    V, G = load_test_videos(os.path.join(base, f'{dataset}_CLIP_rgbtest.csv'),
                            os.path.join(base, 'gt.npy' if dataset == 'xd'
                                         else 'gt_ucf.npy'), n_videos)
    if len(V) < 40:
        print(f'\n{dataset}: {len(V)} videos - skipped')
        return None

    prompts = E.flatten_prompts(dataset)
    owners = E.get_prompt_owner(dataset)
    emb = encode_prompts('ViT-B/16', prompts, device=device).cpu().numpy()
    emb = emb.astype(np.float32)
    ncols = normal_columns(dataset, owners)
    dirs, _ = build_class_directions('ViT-B/16', dataset, device='cpu', verbose=False)
    d = dirs.numpy().astype(np.float32)[1:]
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)

    inc, _k1 = _auc_each(V, G, lambda v: _max_proj(v, d))
    cand, _k2 = bank_auc(V, G, emb)
    nrm, _k3 = norm_auc(V, G)
    stats = [confidence_stats(v, emb, ncols) for v in V]
    cols = {k: np.array([_f(s[k]) for s in stats], dtype=np.float64) for k in STATS}

    gain, mag_gain = cand - inc, cand - nrm
    win = cand > inc
    keep = np.isfinite(inc) & np.isfinite(cand) & np.isfinite(nrm) & np.isfinite(gain)
    for k in STATS:
        keep &= np.isfinite(cols[k])
    sel = np.where(keep)[0]
    if sel.size < 40:
        print(f'\n{dataset}: only {sel.size} fully-observed videos - skipped')
        return None
    g, w, m_ = gain[sel], win[sel], mag_gain[sel]
    print(f'\n{dataset}: {sel.size}/{len(V)} usable | class-max '
          f'{np.mean(inc[sel]):.4f}  event-max {np.mean(cand[sel]):.4f}  '
          f'||f|| {np.mean(nrm[sel]):.4f}  win-rate {np.mean(w):.3f}')

    rep = {'dataset': dataset, 'n': _i(sel.size),
           'win_rate': _r(np.mean(w), 3),
           'gain_mean': _r(np.mean(g)), 'gain_ci95': ci95(g),
           'gain_over_magnitude': _r(np.mean(m_))}
    print(f'  {"predictor":<12} {"rho(win)":>9} {"mono":>5}   win rate by quartile')
    for k in STATS:
        x = cols[k][sel]
        rows = quartile_table(x, w, g)
        rho = spearman(x, w.astype(np.float64))
        rep[k] = {'rho_win_rate': rho, 'rho_gain': spearman(x, g),
                  'rho_mag_gain': spearman(x, m_),
                  'buckets': rows, 'monotone': monotone(rows)}
        cells = '  '.join(f'{_f(b["win_rate"], 0):.2f}(n={b["n"]})' for b in rows)
        print(f'  {k:<12} {rho:>+9.3f} {rep[k]["monotone"]:>+5d}   {cells}')
    return rep


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
            p = os.path.join(opt.repo_root(), 'runs', f'selective_trust_{ds}.json')
            try:
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(r, f, indent=2, ensure_ascii=False)
                print(f'  saved -> {p}')
            except OSError as exc:
                print(f'  WARNING: write failed: {exc}')
    if not reps:
        print('no usable data')
        return 1

    print(f'\n{"=" * 76}\nIS TEXT GUIDANCE SELF-AWARE?\n{"=" * 76}')
    found = False
    for ds, r in reps.items():
        best = max(STATS, key=lambda k, r=r: abs(_f(r[k]['rho_win_rate'], 0)))
        b = r[best]
        found = found or bool(b['monotone'])
        print(f'  {ds}: strongest label-free predictor = {best}  '
              f'rho(win)={_f(b["rho_win_rate"], 0):+.3f}  '
              f'monotone={b["monotone"]:+d}')
        for row in b['buckets']:
            print(f'      [{row["lo"]}, {row["hi"]}]  n={row["n"]:>4}  '
                  f'win={_f(row["win_rate"], 0):.3f}  gain {row["gain"]:+.4f} '
                  f'[{row["lo_ci"]:+.4f}, {row["hi_ci"]:+.4f}]')
    print('\n  Monotone predictor found on some dataset: '
          f'{found}')
    print('  If False, the gain cannot be predicted from the response itself, so')
    print('  selective application is not available and the average has to be sold')
    print('  as an average - which is the honest worst case for this line of work.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
