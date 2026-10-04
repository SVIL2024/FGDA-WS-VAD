"""Stage 0 (plan §三 H2): is anomaly evidence carried by domain-shared temporal bands?

    python src/analysis_spectral_bands.py
        -> runs/stage0_spectral_bands.json

    python src/analysis_spectral_bands.py --max-videos 40 --out-dir <scratch>
        # smoke - NEVER over real artifacts

The cross-domain plan bets (H2) that on cached CLIP frame-feature trajectories
the two things a detector needs are separated along the temporal-frequency axis:

  * anomaly evidence lives in HIGH temporal-frequency bands (sudden motion,
    short irregular bursts), and this band's evidence is domain-shared;
  * domain style lives in LOW bands (sustained scene context), so low-band
    statistics are domain-specific and must be aligned or suppressed.

If that split is real, a spectral branch trained on source domains transfers,
and fusing it with the semantic branch covers exactly the failure mode
Alert-CLIP attributes to CLIP's semantic blind spot. If it is not real - if
anomaly evidence sits in the same bands that carry domain style - the plan's
M3 spectral anchor loses its theoretical story and the B1-style band-gain
variants are the wrong family to invest in.

Two readouts, both on cached features, no GPU:

  A  anomaly effect per cue, per dataset. Each per-frame scalar cue is scored
     with the repo's within-video statistic (z-score inside the video, AUC
     against that video's GT, free sign) AND a *signed* effect (mean z over
     anomalous frames minus mean z over normal frames). The signed effect is
     what makes cross-dataset comparison possible: free-sign AUC throws away
     the orientation, and H2's claim is about WHERE the evidence sits, which is
     an orientation claim. ``norm_dev`` is the repo's ||f - mean_video|| cue -
     on UCF the magnitude cue is a documented negative asset, so a low-band /
     magnitude cue with opposite signed effects on the two datasets is the
     expected shape of the result, not a bug.

  B  domain separability per band. Per video, the energy FRACTION of each
     band is one feature; a logistic regression classifies UCF vs XD from it
     (5-fold CV AUC). H2 predicts: low-band fractions separate the domains
     strongly, high-band fractions weakly. The informative reading is the
     CONTRAST across bands, not any single AUC - two datasets drawn from
     different scenes differ somewhere in every band.

Band convention: the rFFT of the mean-centred (T,512) trajectory, bands given
as fractions of the Nyquist bin index. Bin 0 is ~0 by construction (centred),
and the definition is recorded in the JSON so the dual-mamba cutoff sweeps
(0.15-0.65) can be compared later.
"""

import argparse
import json
import os
import platform
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import freq_text_options as opt  # noqa: E402
from analysis_crossdomain_transfer import load_test_videos  # noqa: E402

# bands as fractions of the Nyquist bin index of the centred trajectory
BANDS = [('b1_low', 0.00, 0.05), ('b2_mid', 0.05, 0.15),
         ('b3_high', 0.15, 0.40), ('b4_top', 0.40, 1.00)]
MA_WINDOW = 16  # ~0.25 s of feature time at 16-frame feature stride


def _moving_average(x, window):
    """Trailing moving average along time, edges shrunk (no phase shift)."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    idx = np.arange(1, x.shape[0] + 1)
    lo = np.maximum(idx - window, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


def video_cues(feat):
    """Per-frame scalar cues for one video; ``None`` entries where undefined.

    Returns (names, list of (T,) arrays or None). Short videos (< 2 * MA window)
    get the moving-average cues as None rather than a degenerate average over
    one frame - a cue that is undefined on a video must not silently become a
    constant, because a constant z-scores to 0 everywhere and quietly dilutes
    the pooled effect.
    """
    x = feat.astype(np.float64)
    xc = x - x.mean(0, keepdims=True)
    t = xc.shape[0]
    spec = np.fft.rfft(xc, axis=0)
    n_bins = spec.shape[0]

    names, cues = [], []
    for bname, lo, hi in BANDS:
        k_lo, k_hi = int(np.ceil(lo * (n_bins - 1))), int(np.floor(hi * (n_bins - 1)))
        mask = np.zeros(n_bins)
        mask[max(k_lo, 1): k_hi + 1] = 1.0          # bin 0 is DC, centred away
        eb = np.linalg.norm(np.fft.irfft(spec * mask[:, None], n=t, axis=0), axis=1)
        names.append(f'band_{bname}')
        cues.append(eb)

    diff = np.diff(xc, axis=0)
    hf_diff = np.concatenate([[0.0], np.linalg.norm(diff, axis=1)])
    names.append('hf_diff')
    cues.append(hf_diff)

    if t >= 2 * MA_WINDOW:
        ma = _moving_average(xc, MA_WINDOW)
        resid = xc - ma
        names.append('hf_resid')
        cues.append(np.linalg.norm(resid, axis=1))
        names.append('lf_dev')
        cues.append(np.linalg.norm(ma, axis=1))
    else:
        names.extend(['hf_resid', 'lf_dev'])
        cues.extend([None, None])
    names.append('norm_dev')
    cues.append(np.linalg.norm(xc, axis=1))
    return names, cues


def band_fractions(feat):
    """Per-video energy fraction of each band - the domain-classification features."""
    xc = feat.astype(np.float64) - feat.astype(np.float64).mean(0, keepdims=True)
    spec = np.fft.rfft(xc, axis=0)
    n_bins = spec.shape[0]
    energies = []
    for _, lo, hi in BANDS:
        k_lo, k_hi = int(np.ceil(lo * (n_bins - 1))), int(np.floor(hi * (n_bins - 1)))
        mask = np.zeros(n_bins)
        mask[max(k_lo, 1): k_hi + 1] = 1.0
        eb = np.fft.irfft(spec * mask[:, None], n=xc.shape[0], axis=0)
        energies.append(float((eb ** 2).sum()))
    total = sum(energies)
    if total <= 0:
        return None
    return [e / total for e in energies]


def cue_stats(pool, names_getter):
    """Within-video AUC (free sign) and signed effect per cue, aggregated."""
    per_name = {}
    for x in pool:
        names, cues = names_getter(x['feat'])
        for nm, s in zip(names, cues):
            if s is None:
                continue
            s = np.asarray(s, dtype=np.float64)
            if s.std() < 1e-12:
                continue
            z = (s - s.mean()) / (s.std() + 1e-8)
            g = x['gt']
            try:
                a = _auc_safe(g, z)
            except ValueError:
                continue
            d = per_name.setdefault(nm, {'auc': [], 'effect': []})
            d['auc'].append(max(a, 1.0 - a))
            d['effect'].append(z[g].mean() - z[~g].mean())
    out = {}
    for nm, d in per_name.items():
        auc = np.asarray(d['auc'])
        eff = np.asarray(d['effect'])
        sem = eff.std(ddof=1) / np.sqrt(len(eff)) if len(eff) > 1 else float('nan')
        out[nm] = {'n_videos': int(len(auc)),
                   'auc_free_sign_mean': round(float(auc.mean()), 4),
                   'effect_signed_mean': round(float(eff.mean()), 4),
                   'effect_signed_sem': round(float(sem), 4),
                   'effect_ci95': [round(float(eff.mean() - 1.96 * sem), 4),
                                   round(float(eff.mean() + 1.96 * sem), 4)],
                   'effect_positive_frac': round(float((eff > 0).mean()), 4)}
    return out


def _auc_safe(g, s):
    return float(roc_auc_score(g, s))


def domain_separability(preps, seed=0):
    """Per-band and joint logistic AUC for UCF-vs-XD from band energy fractions."""
    feats, ys = [], []
    for yi, ds in enumerate(('xd', 'ucf')):
        for x in preps[ds]['videos']:
            fr = band_fractions(x['feat'])
            if fr is not None:
                feats.append(fr)
                ys.append(yi)
    X = np.asarray(feats)
    y = np.asarray(ys)
    if len(np.unique(y)) < 2 or len(X) < 20:
        return {'reason': f'insufficient videos ({len(X)})'}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=1000)
    out = {}
    for bi, (bname, _, _) in enumerate(BANDS):
        scores = cross_val_score(clf, X[:, [bi]], y, cv=cv, scoring='roc_auc')
        out[f'band_{bname}'] = round(float(scores.mean()), 4)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    out['all_bands'] = round(float(scores.mean()), 4)
    out['n_videos'] = int(len(X))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max-videos', type=int, default=None,
                    help='cap per dataset (smoke testing only)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default=None,
                    help='output dir override; default runs/ - NEVER smoke over runs/')
    args = ap.parse_args()

    preps = {}
    for ds in ('xd', 'ucf'):
        videos, skip = load_test_videos(ds, max_videos=args.max_videos, seed=args.seed)
        preps[ds] = {'videos': videos, 'skip': skip}
        print(f'[{ds}] mixed-GT videos={len(videos)} skipped={skip}', flush=True)

    out = {'args': vars(args),
           'env': {'python': platform.python_version(), 'numpy': np.__version__},
           'band_definition': 'fractions of the Nyquist bin index of the '
                              'mean-centred (T,512) CLIP-feature trajectory',
           'bands': [{'name': n, 'lo': lo, 'hi': hi} for n, lo, hi in BANDS]}
    for ds in ('xd', 'ucf'):
        out[ds] = {'n_videos': len(preps[ds]['videos']),
                   'skipped': preps[ds]['skip'],
                   'cues': cue_stats(preps[ds]['videos'], video_cues)}
    out['domain_separability_per_band'] = domain_separability(preps, seed=args.seed)

    # cross-dataset consistency of the signed effect: H2's claim is an
    # orientation claim, so this is the readout that can falsify it
    consistency = {}
    for nm in out['xd']['cues']:
        a, b = out['xd']['cues'].get(nm), out['ucf']['cues'].get(nm)
        if not a or not b:
            continue
        same_sign = (a['effect_signed_mean'] > 0) == (b['effect_signed_mean'] > 0)
        both_excl_0 = all(lo > 0 or hi < 0 for lo, hi in
                          (a['effect_ci95'], b['effect_ci95']))
        consistency[nm] = {'xd_effect': a['effect_signed_mean'],
                           'ucf_effect': b['effect_signed_mean'],
                           'same_sign': bool(same_sign),
                           'both_ci_exclude_zero': bool(both_excl_0)}
    out['cross_dataset_consistency'] = consistency

    out_dir = args.out_dir or os.path.join(opt.repo_root(), 'runs')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'stage0_spectral_bands.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'saved -> {path}', flush=True)

    print('\n' + '=' * 78)
    print('STAGE 0 / H2 SUMMARY (within-video signed effects + band domain AUC)')
    print('=' * 78)
    for ds in ('xd', 'ucf'):
        print(f'\n[{ds}]')
        for nm, st in sorted(out[ds]['cues'].items(),
                             key=lambda kv: -abs(kv[1]['effect_signed_mean'])):
            print(f"  {nm:<14} effect={st['effect_signed_mean']:+.4f} "
                  f"CI[{st['effect_ci95'][0]:+.4f},{st['effect_ci95'][1]:+.4f}] "
                  f"auc(free)={st['auc_free_sign_mean']:.4f} n={st['n_videos']}")
    print('\ndomain separability per band (UCF vs XD, logistic 5-fold AUC):')
    for k, v in out['domain_separability_per_band'].items():
        print(f'  {k:<16} {v}')
    print('\ncross-dataset signed-effect consistency:')
    for nm, c in consistency.items():
        print(f"  {nm:<14} xd={c['xd_effect']:+.4f} ucf={c['ucf_effect']:+.4f} "
              f"same_sign={c['same_sign']} both_ci_excl_0={c['both_ci_exclude_zero']}")
    print("""
  Read: H2 wants band_{b3,b4} anomaly effects positive in BOTH datasets while
  band_{b1,b2} fractions carry the domain separability. If instead the same
  bands carry both the anomaly effect and the domain signal, the spectral
  anchor must be re-designed (e.g. condition on the semantic score) before
  any training run spends GPU time on it.""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
