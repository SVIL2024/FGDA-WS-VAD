"""Held-out test: can a label-free rule decide *when* to trust text guidance?

    python analysis_selective_routing.py --dataset both

Why a separate script. analysis_selective_trust.py picked "spread" as the best of
four predictors using rho computed on the full set, then reported that predictor's
top-quartile win rate of 0.769 and gain +0.086 on that same data. Those numbers
carry the winner's curse: maximising over four candidates inflates whichever won,
so nothing may be concluded from them. This script re-tests the idea the only way
that makes such a number mean anything - choose on one half, measure on the other.

Protocol, repeated over shuffled splits with all statistics pooled over folds:

  1. Split the videos into A / B.
  2. On A only: choose the predictor (spread, margin, event_share, agree) and the
     threshold t maximising the selected-subset mean gain over the class-direction
     incumbent, subject to a minimum coverage so the rule cannot degenerate.
  3. On B only, with the predictor and t frozen: report coverage, win rate, and the
     paired gain.

The gain on the videos the rule *rejects* matters as much as the retained ones. A
rule is only useful if the rejected videos are the ones where text guidance would
have hurt; otherwise it is just discarding hard cases and flattering itself. Two
failure modes are therefore reported rather than hidden:

  degenerate      coverage collapses, or the gain rides on a few outliers
  no separation   the rejected subset also gains, i.e. the rule learned nothing
                  about which videos to trust

The claim under test is a reliability claim, not an accuracy claim: "apply event
guidance conditionally, and say when from the response alone, with no labels and
no training". That is worth knowing even if the headline AP never moves.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analysis_gain_vs_difficulty as ga  # noqa: E402
import analysis_selective_trust as st  # noqa: E402
import freq_text_events as E  # noqa: E402
import freq_text_options as opt  # noqa: E402
from freq_text_text import build_class_directions  # noqa: E402
from probe_events import encode_prompts, load_test_videos  # noqa: E402

log = logging.getLogger('selective_routing')

# numeric guards live in the shared analysis helper
_f, _i, _r = ga._f, ga._i, ga._r
ci95, spearman = ga.ci95, ga.spearman
_auc_each, bank_auc, norm_auc = ga._auc_each, ga.bank_auc, ga.norm_auc
NAN, STATS = st.NAN, st.STATS
_max_proj, confidence_stats = st._max_proj, st.confidence_stats
ARRAY_KEYS = ('inc', 'cand', 'nrm', 'gain', 'mag_gain')


def normal_columns(dataset, owners):
    """Indices of the bank's normal-class prompts, or None if unlabelled."""
    names = E.class_names(dataset)
    try:
        cols = [i for i, o in enumerate(owners)
                if names[_i(o, 0)].lower().startswith('normal')]
    except (IndexError, TypeError, ValueError, AttributeError):
        log.warning('normal prompts unidentifiable for %s', dataset, exc_info=True)
        return None
    return cols or None


def load_matrix(dataset, n_videos, device):
    """AUC triplets plus label-free predictors for every usable video."""
    base = os.path.join(opt.repo_root(), 'list')
    test_list = os.path.join(base, f'{dataset}_CLIP_rgbtest.csv')
    gt = os.path.join(base, 'gt.npy' if dataset == 'xd' else 'gt_ucf.npy')
    V, G = load_test_videos(test_list, gt, n_videos)
    if len(V) < 60:
        print(f'{dataset}: only {len(V)} mixed-GT videos - skipped')
        return None
    prompts = E.flatten_prompts(dataset)
    owners = E.get_prompt_owner(dataset)
    emb = encode_prompts('ViT-B/16', prompts, device=device).cpu().numpy()
    emb = emb.astype(np.float32)
    ncols = normal_columns(dataset, owners)
    dirs, _ = build_class_directions('ViT-B/16', dataset, device='cpu', verbose=False)
    d = dirs.numpy().astype(np.float32)[1:]
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)

    inc, _a = _auc_each(V, G, lambda v: _max_proj(v, d))
    cand, _b = bank_auc(V, G, emb)
    nrm, _c = norm_auc(V, G)
    stats = [confidence_stats(v, emb, ncols) for v in V]
    pred = {k: np.array([_f(s[k]) for s in stats], dtype=np.float64) for k in STATS}

    keep = (np.isfinite(inc) & np.isfinite(cand) & np.isfinite(nrm)
            & np.isfinite(cand - inc))
    for k in STATS:
        keep &= np.isfinite(pred[k])
    idx = np.where(keep)[0]
    if idx.size < 60:
        print(f'{dataset}: only {_i(idx.size)} fully-observed videos - skipped')
        return None
    out = {key: arr[idx] for key, arr in
           (('inc', inc), ('cand', cand), ('nrm', nrm))}
    out['gain'] = out['cand'] - out['inc']
    out['mag_gain'] = out['cand'] - out['nrm']
    out['n'] = idx.size
    out['pred'] = {k: pred[k][idx] for k in STATS}
    win = np.count_nonzero(out['cand'] > out['inc']) / max(out['n'], 1)
    print(f'\n{dataset}: {out["n"]}/{len(V)} usable | class-max '
          f'{_f(np.mean(out["inc"])):.4f}  event-max '
          f'{_f(np.mean(out["cand"])):.4f}  ||f|| '
          f'{_f(np.mean(out["nrm"])):.4f}  win-rate {_f(win):.3f}')
    return out


def subset(data, idx):
    """Index every per-video array in ``data``; ``pred`` is a dict of arrays."""
    out = {k: data[k][idx] for k in ARRAY_KEYS if k in data}
    out['pred'] = {k: v[idx] for k, v in data['pred'].items()}
    out['n'] = _i(len(idx), 0)
    return out


def fit_rule(data, min_cov=0.15):
    """Choose (predictor, threshold) on this subset by maximising selected gain."""
    g = data['gain']
    n = max(_i(len(g), 0), 1)
    lo_n = max(10, _i(round(min_cov * n), 10))
    best = None
    for k in STATS:
        x = data['pred'][k]
        if len(np.unique(x)) < 6:
            continue
        try:
            qs = np.linspace(0.95, 1.0 - min_cov, 9)
            cuts = np.unique(np.round(np.quantile(x, qs), 8))
        except (TypeError, ValueError):
            log.warning('quantiles failed for %s', k, exc_info=True)
            continue
        for t in np.atleast_1d(cuts):
            tv = _f(t)
            if not np.isfinite(tv):
                continue
            try:
                m = x >= tv
                cnt = _i(np.count_nonzero(m), 0)
                if cnt < lo_n:
                    continue
                mean_sel = _f(np.mean(g[m]))
            except (TypeError, ValueError, IndexError):
                log.warning('threshold scan failed for %s', k, exc_info=True)
                continue
            if not np.isfinite(mean_sel):
                continue
            if best is None or mean_sel > best['sel_gain']:
                best = {'feature': k, 't': tv, 'sel_gain': mean_sel,
                        'coverage': _f(cnt) / n}
    return best


def apply_rule(data, rule):
    """Subset statistics under a frozen rule, scored on held-out videos."""
    g = data['gain']
    win_all = data['cand'] > data['inc']
    if not rule:
        return {'feature': 'none', 't': NAN, 'coverage': 1.0,
                'n_sel': _i(len(g), 0), 'n_rej': 0,
                'sel_gain': _r(np.mean(g)), 'sel_win': _r(np.mean(win_all)),
                'sel_mag_gain': _r(np.mean(data['mag_gain'])), 'rej_gain': NAN}
    x = data['pred'].get(rule['feature'])
    if x is None:
        log.warning('rule names unknown feature %r; falling back to keep-all',
                    rule.get('feature'))
        return apply_rule(data, None)
    try:
        m = x >= _f(rule['t'])
        inv = ~m
        sel, rej = g[m], g[inv]
        win = win_all[m]
        return {'feature': rule['feature'], 't': _f(rule['t']),
                'coverage': _r(np.mean(m)),
                'n_sel': _i(np.count_nonzero(m), 0),
                'n_rej': _i(np.count_nonzero(inv), 0),
                'sel_gain': _r(np.mean(sel)) if sel.size else NAN,
                'sel_win': _r(np.mean(win)) if win.size else NAN,
                'sel_mag_gain': _r(np.mean(data['mag_gain'][m])) if m.any() else NAN,
                'rej_gain': _r(np.mean(rej)) if rej.size else NAN}
    except (TypeError, ValueError, IndexError):
        log.warning('applying rule failed; scored as keep-all', exc_info=True)
        return apply_rule(data, None)


def cross_fit(data, n_splits=5, seed=0):
    """Average held-out rule performance over several random half-splits."""
    rng = np.random.RandomState(seed)
    n = _i(len(data['gain']), 0)
    rows = []
    for k in range(max(_i(n_splits, 5), 1)):
        perm = rng.permutation(n)
        half = n // 2
        A, B = subset(data, perm[:half]), subset(data, perm[half:])
        rule = fit_rule(A)
        s = apply_rule(B, rule)
        rows.append(s)
        print(f'  fold {k}: chose {s["feature"]:<11} '
              f't={_f(s["t"], 0):+.4f}  coverage {_f(s["coverage"], 0):.2f}  '
              f'held-out win {_f(s["sel_win"], 0):.3f}  gain '
              f'{_f(s["sel_gain"], 0):+.4f}  rejected-subset gain '
              f'{_f(s["rej_gain"], 0):+.4f}')
    return rows


def summarise(rows):
    """Pool held-out folds: coverage, win rate, gain, and rejected-subset gain."""
    def col(key):
        return np.array([_f(r.get(key)) for r in rows], dtype=np.float64)

    cov, win, sg, rg = col('coverage'), col('sel_win'), col('sel_gain'), col('rej_gain')
    cov, win, sg = cov[np.isfinite(cov)], win[np.isfinite(win)], sg[np.isfinite(sg)]
    feats = {}
    for r in rows:
        key = str(r.get('feature'))
        feats[key] = feats.get(key, 0) + 1
    rg_ok = rg[np.isfinite(rg)]
    return {'n_folds': _i(len(rows), 0),
            'coverage_mean': _r(np.mean(cov)) if cov.size else NAN,
            'sel_win_mean': _r(np.mean(win)) if win.size else NAN,
            'sel_gain_mean': _r(np.mean(sg)) if sg.size else NAN,
            'sel_gain_ci95': ci95(sg),
            'sel_gain_min_fold': _r(np.min(sg)) if sg.size else NAN,
            'sel_gain_max_fold': _r(np.max(sg)) if sg.size else NAN,
            'rej_gain_mean': _r(np.mean(rg_ok)) if rg_ok.size else NAN,
            'feature_counts': feats}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='both', choices=['xd', 'ucf', 'both'])
    ap.add_argument('--n-videos', type=int, default=10000)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    sets = ['xd', 'ucf'] if args.dataset == 'both' else [args.dataset]
    reps = {}
    for ds in sets:
        data = load_matrix(ds, args.n_videos, args.device)
        if data is None:
            continue
        print('  held-out folds (rule fitted on A, scored on B):')
        rows = cross_fit(data, n_splits=args.folds)
        s = summarise(rows)
        s['dataset'] = ds
        s['overall_win_rate'] = _r(np.mean(data['cand'] > data['inc']))
        s['overall_gain'] = _r(np.mean(data['gain']))
        s['per_fold'] = rows
        reps[ds] = s
        p = os.path.join(opt.repo_root(), 'runs', f'selective_routing_{ds}.json')
        try:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(s, f, indent=2, ensure_ascii=False)
            print(f'  saved -> {p}')
        except OSError as exc:
            print(f'  WARNING: write failed: {exc}')

    if not reps:
        print('no usable data')
        return 1
    print(f'\n{"=" * 78}\nHELD-OUT SELECTIVE ROUTING (mean over folds)\n{"=" * 78}')
    print(f'  {"dataset":<7} {"cov":>6} {"win":>7} {"sel gain":>10} '
          f'{"95% CI":>22} {"rej gain":>10}  fold features')
    for ds, s in reps.items():
        ci = s['sel_gain_ci95']
        txt = f'[{_f(ci[0]):+.4f}, {_f(ci[1]):+.4f}]' if len(ci) == 2 else 'n/a'
        print(f'  {ds:<7} {_f(s["coverage_mean"]):>6.2f} '
              f'{_f(s["sel_win_mean"]):>7.3f} {_f(s["sel_gain_mean"]):>+10.4f} '
              f'{txt:>22} {_f(s["rej_gain_mean"]):>+10.4f}  {s["feature_counts"]}')
    print(f'\n  {"dataset":<7} {"uncond win":>11} {"uncond gain":>12} '
          f'{"routing win delta":>19}')
    for ds, s in reps.items():
        delta = _f(s['sel_win_mean'], 0) - _f(s['overall_win_rate'], 0)
        print(f'  {ds:<7} {_f(s["overall_win_rate"]):>11.3f} '
              f'{_f(s["overall_gain"]):>+12.4f} {delta:>+19.3f}')
    print("""
  Reading this table:
    routing useful   coverage > 0.2, sel-gain CI excludes 0, sel gain above the
                     unconditional gain, and rejected-subset gain <= 0 outside CI
    routing useless  sel-gain CI spans 0, or the rejected subset also gains (the
                     rule learned nothing about which videos to distrust)""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
