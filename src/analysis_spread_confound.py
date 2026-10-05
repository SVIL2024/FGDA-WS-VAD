"""Is the label-free "spread" rule real, or a proxy for something boring?

    python analysis_spread_confound.py --dataset xd

analysis_selective_routing.py found one held-out-surviving result: on XD, keeping
the top ~15% of videos by response *spread* raises the event-vs-class win rate
from 0.558 to 0.769, with spread chosen in 8/8 folds at a threshold stable to
+-0.001. Before that is worth a sentence in a paper it has to survive three
attacks.

  1. PROXY FOR LENGTH. spread is a std over T frames, and std estimates shrink
     with sample size, so spread could simply be counting frames. Length is known
     at inference so this would not be leakage, but a length rule is uninteresting
     and would explain the effect without any role for text.
  2. PROXY FOR ANOMALY FRACTION. Videos with many anomaly frames are easier to
     rank, so a high-spread subset could just be an easy-GT subset. Not leakage
     (fraction is not used), but it would mean the rule detects "this video has
     lots of anomaly" rather than "the text signal is sharp here".
  3. PROXY FOR EASE. If spread tracks the magnitude cue's own AUC, high-spread
     videos are simply ones where *any* readout works, and event guidance gains
     nothing specific.

Each attack is a rank correlation plus a stratified re-test: within strata of the
suspected confound, does high-spread still beat low-spread? If the effect is a
proxy it collapses inside strata; if it is real it survives.

Also reported for context: the ORACLE ceiling. Routing on the true per-video gain
is the best any selector could do, so the fraction of oracle gain the spread rule
captures bounds what is achievable here without inventing a better predictor.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analysis_selective_routing as sr  # noqa: E402
import freq_text_options as opt  # noqa: E402

log = logging.getLogger('spread_confound')
_f, _i, _r = sr._f, sr._i, sr._r
NAN = sr.NAN
spearman = sr.spearman


def per_video_meta(dataset, n_videos):
    """Per-video length and GT anomaly fraction, aligned to usable-video order.

    Re-derives the usable subset with the same filter as ``load_matrix`` so the
    rows line up. Both quantities are used here as *explanations only*; neither is
    available to the routing rule, which sees the response spread alone.
    """
    base = os.path.join(opt.repo_root(), 'list')
    from probe_events import load_test_videos
    V, G = load_test_videos(os.path.join(base, f'{dataset}_CLIP_rgbtest.csv'),
                            os.path.join(base, 'gt.npy' if dataset == 'xd'
                                         else 'gt_ucf.npy'), n_videos)
    lens, fracs = [], []
    for i in range(min(len(V), len(G))):
        try:
            lens.append(_i(len(V[i]), 0))
            g = G[i]
            fracs.append(_r(float(np.mean(g)), 5) if len(g) else NAN)
        except (TypeError, ValueError):
            log.warning('metadata failed for one video', exc_info=True)
            lens.append(0)
            fracs.append(NAN)
    return np.array(lens, dtype=np.float64), np.array(fracs, dtype=np.float64)


def stratified(x, effect, q=3):
    """Inside each quartile of a confound, does high x still beat low x?

    ``effect`` is the per-video gain. Within a stratum the videos are split at the
    stratum median of x and the two halves' mean gains compared, so a confound that
    fully explains the effect leaves the within-stratum difference at ~0.
    """
    rows = []
    try:
        edges = np.quantile(np.asarray(x, dtype=np.float64),
                            np.linspace(0, 1, q + 1))
    except (TypeError, ValueError):
        log.warning('strata edges failed', exc_info=True)
        return rows
    x = np.asarray(x, dtype=np.float64)
    eff = np.asarray(effect, dtype=np.float64)
    for i in range(max(_i(q, 3), 1)):
        lo, hi = _f(edges[i]), _f(edges[i + 1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        m = (x >= lo) & ((x <= hi) if i == q - 1 else (x < hi))
        m &= np.isfinite(eff)
        n = _i(np.count_nonzero(m), 0)
        if n < 20:
            continue
        xs = x[m]
        try:
            med = float(np.median(xs))
            hi_m, lo_m = xs >= med, xs < med
            nh, nl = _i(np.count_nonzero(hi_m), 0), _i(np.count_nonzero(lo_m), 0)
            if nh < 8 or nl < 8:
                continue
            diff = _f(np.mean(eff[m][hi_m])) - _f(np.mean(eff[m][lo_m]))
        except (TypeError, ValueError):
            log.warning('within-stratum split failed', exc_info=True)
            continue
        if not (np.isfinite(diff) and np.isfinite(_f(np.mean(eff[m][hi_m])))
                and np.isfinite(_f(np.mean(eff[m][lo_m])))):
            continue
        rows.append({'lo': _r(lo, 5), 'hi': _r(hi, 5), 'n': n,
                     'hi_gain': _r(np.mean(eff[m][hi_m])),
                     'lo_gain': _r(np.mean(eff[m][lo_m])),
                     'diff': _r(diff), 'n_hi': nh, 'n_lo': nl})
    return rows


def oracle(data, coverage):
    """Best attainable gain by selecting on the true gain (an upper bound)."""
    g = np.sort(np.asarray(data['gain'], dtype=np.float64))[::-1]
    n = max(_i(len(g) * _f(coverage, 0.15)), 1)
    return _r(np.mean(g[:n])) if g.size else NAN


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='xd', choices=['xd', 'ucf'])
    ap.add_argument('--n-videos', type=int, default=10000)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    data = sr.load_matrix(args.dataset, args.n_videos, args.device)
    if data is None:
        return 1
    lens, fracs = per_video_meta(args.dataset, args.n_videos)
    n = _i(data['n'], 0)
    if len(lens) < n:
        print(f'cannot align metadata ({len(lens)} rows vs {n} usable videos) '
              '- run with --n-videos large enough to cover the test list')
        return 1
    lens, fracs = lens[:n], fracs[:n]

    spread = data['pred']['spread']
    gain, mag_gain = data['gain'], data['mag_gain']
    mag_auc, cand_auc = data['nrm'], data['cand']

    print(f'\n{args.dataset}: {n} videos | unconditional win '
          f'{_r(np.mean(cand_auc > data["inc"])):.3f}  gain {_r(np.mean(gain)):+.4f}')
    print('\n  is spread a proxy?  (rank correlation with spread)')
    confounds = {'video length': lens, 'anomaly fraction': fracs,
                 '||f|| AUC (ease)': mag_auc, 'event AUC': cand_auc}
    rep = {'dataset': args.dataset, 'n': n, 'confounds': {}}
    for name, x in confounds.items():
        rho = spearman(spread, x)
        rep['confounds'][name] = rho
        print(f'    rho(spread, {name:<17}) = {rho:+.3f}')

    thr = _f(np.quantile(spread, 0.85))
    hi = spread >= thr
    print(f'\n  spread rule (top 15%, t={thr:.4f}): coverage {np.mean(hi):.3f}  '
          f'gain {_r(np.mean(gain[hi])):+.4f}  win '
          f'{_r(np.mean(cand_auc[hi] > data["inc"][hi])):.3f}  '
          f'||f||-relative gain {_r(np.mean(mag_gain[hi])):+.4f}')
    orc = oracle(data, 0.15)
    sel = _f(np.mean(gain[hi]))
    base = _f(np.mean(gain))
    if np.isfinite(orc) and orc > base:
        print(f'  oracle at 15% coverage: {_f(orc):+.4f}  -> spread captures '
              f'{(sel - base) / (orc - base):.2f} of the attainable lift')

    print('\n  within-stratum re-test of the spread rule')
    for name, x in (('length', lens), ('anomaly fraction', fracs),
                    ('||f|| AUC', mag_auc)):
        rows = stratified(x, gain)
        if not rows:
            print(f'    {name:<17}: no usable strata')
            continue
        diffs = [_f(r['diff']) for r in rows if np.isfinite(_f(r['diff']))]
        pos = sum(1 for d in diffs if d > 0)
        rep['confounds'][name + ' (strata)'] = {'rows': rows,
                                                'strata_positive': pos,
                                                'strata_n': len(diffs)}
        if diffs:
            print(f'    {name:<17}: high-spread wins in {pos}/{len(diffs)} strata, '
                  f'mean within-stratum lift {np.mean(diffs):+.4f}')
        else:
            print(f'    {name:<17}: no usable within-stratum splits')
        for r in rows:
            print(f'        [{r["lo"]}, {r["hi"]}] n={r["n"]:>4}  '
                  f'hi {_f(r["hi_gain"], 0):+.4f} vs lo {_f(r["lo_gain"], 0):+.4f} '
                  f'-> {r["diff"]:+.4f}')

    p = os.path.join(opt.repo_root(), 'runs', f'spread_confound_{args.dataset}.json')
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        print(f'\n  saved -> {p}')
    except OSError as exc:
        print(f'\n  WARNING: write failed: {exc}')
    print("""
  Verdict guide: rho <= ~0.3 with all strata positive means spread carries
  information the confound does not. A large rho plus strata that vanish means the
  "predictor" was the confound all along and there is no reliability story.""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
