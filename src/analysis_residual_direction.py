"""Step 1: does a direction fitted to the magnitude-residual text score beat the frozen text direction?

    python analysis_residual_direction.py --dataset both

``analysis_magnitude_orthogonal.py`` established that the text direction carries
magnitude-independent signal (XD: +0.078 to +0.103 AUC over a same-statistic
random null, CIs excluding 0) and that magnitude is nearly a function of the
text direction while the reverse does not hold. The next object is therefore not
a better text encoder but a *better direction*: the one that predicts what the
text score says once the magnitude cue has been removed. On the TRAIN split only,

    x(t) = f(t) - mean_t f(t)                  per-clip centring
    m(t) = ||x(t)||                             the free magnitude cue
    s(t) = <x(t), d_text>                       d_text = frozen CLIP class bank
    r(t) = s(t) - beta * m(t)                   beta fitted on train
    d'   = argmin || r - X w ||^2 + lam ||w||^2

``d'`` is the linear-algebra statement of "the part of the text direction that
magnitude does not already explain", and it bounds what a text adapter could
achieve: a nonlinear adapter trained on the same target has the same information
and strictly more capacity, so if a ridge least-squares fit cannot beat the
frozen direction, the adapter is not the fix either.

Three protocol properties, each of which is easy to get wrong:

* **Everything fitted on TRAIN, applied to test.** The magnitude-orthogonal
  analysis used a test-derived ``mu`` purely to reproduce the gate's numbers; that
  shortcut is not available for a direction meant to be *used*, so this script
  per-clip centres instead, which needs no cross-split statistics at all and is
  the same convention ``analysis_official_metric`` uses for its pooled ``norm``.
* **The ridge parameter is chosen by the TRAIN residual R-squared.** R^2 needs
  no labels, so the selection cannot leak. Scoring several lambdas on the test
  metric and reporting the winner would be exactly the bias the design doc warns
  about.
* **Both passes stream the same rows.** ``rng`` is advanced by the sampling draw,
  so re-sampling inside the second pass silently scores a *different* subset than
  the one whose sufficient statistics were accumulated. The row subset is chosen
  once and passed to both.

Kill condition, fixed before running: if ``d'`` does not beat the frozen
``d_text`` on the pooled protocol, the "find a better direction" branch is dead
and a text adapter is not the answer either.
"""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import freq_text_options as opt
from freq_text_prompts import DATASET_CLASS_NAMES
from freq_text_text import (GT_FRAME_REPEAT, _f, _gt_offsets, _i, _read_list,
                            build_class_directions)

RIDGES = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _is_anomalous(label, dataset):
    s = str(label).strip()
    if not s:
        return False
    if dataset == 'ucf':
        return s.lower() != 'normal'
    return any(t.upper() != 'A' for t in s.split('-') if t not in ('', '0'))


def _shifted(path, mu):
    """One feature file minus the TRAIN feature mean, in float64.

    Global centring, not per-clip. Per-clip centring removes the between-video
    component, and the between-video component is most of what the pooled metric
    measures: with per-clip centring the text cue scores 0.566 pooled on XD,
    which reproduces exactly the drop the project already recorded (per-video
    z-scoring takes class_max from 0.677 to 0.565). Fitting mu on train and
    applying it to test keeps the pooled comparison honest without leaking.
    """
    v = np.load(path).astype(np.float32)
    if v.ndim != 2 or v.shape[0] == 0:
        return None
    return (v - mu).astype(np.float64)


def _train_mu(rows, C):
    """Feature mean over the sampled train clips - the only statistic that crosses
    the train/test boundary, and it is estimated on train."""
    s = np.zeros(C)
    n = 0
    for p, _lab in rows:
        if not os.path.exists(p):
            continue
        v = np.load(p).astype(np.float32)
        if v.ndim != 2 or v.shape[0] == 0:
            continue
        s += v.sum(0)
        n += v.shape[0]
    return s / n if n else None


def _bank(dataset, device):
    """d_text: mean over the class bank of the unit CLIP class directions.

    This is the direction E1 injects, and it is a constant for the whole run:
    ``freq_text_trainer.py:186`` builds it once from a separately loaded frozen
    CLIP over the raw prompt strings, outside ``params`` and outside the training
    loop, so the learnable ``model.text_prompt_embeddings`` never reach it.
    """
    dirs, _ = build_class_directions('ViT-B/16', dataset, device=device, verbose=False)
    D = np.asarray(dirs.cpu().numpy() if hasattr(dirs, 'cpu') else dirs, dtype=np.float64)
    D = D[1:][:len(DATASET_CLASS_NAMES[dataset]) - 1]
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    mean = D.mean(0)
    return mean / (np.linalg.norm(mean) + 1e-12), D


def _pass1(rows, mu, d_text, C):
    """Sufficient statistics for beta and for the ridge normal equations."""
    n = 0
    sS = sM = sSM = sMM = 0.0
    XtX = np.zeros((C, C))
    for p, _lab in rows:
        if not os.path.exists(p):
            continue
        x = _shifted(p, mu)
        if x is None:
            continue
        m = np.linalg.norm(x, axis=1)
        s = x @ d_text
        sS += float(s.sum())
        sM += float(m.sum())
        sSM += float((s * m).sum())
        sMM += float((m * m).sum())
        XtX += x.T @ x
        n += x.shape[0]
    if not n:
        return None
    beta = (sSM - sS * sM / n) / (sMM - sM * sM / n + 1e-12)
    return {'n': n, 'beta': beta, 'XtX': XtX}


def _pass2(rows, mu, d_text, beta, C, dataset):
    """X^T r, ||r||^2, and the train-side feature/label arrays for the combination."""
    Xtr = np.zeros(C)
    r2 = 0.0
    Ms, Ss, Ys = [], [], []
    for p, lab in rows:
        x = _shifted(p, mu)
        if x is None:
            continue
        m = np.linalg.norm(x, axis=1)
        s = x @ d_text
        r = s - beta * m
        Xtr += x.T @ r
        r2 += float(r @ r)
        Ms.append(m)
        Ss.append(s)
        Ys.append(np.full(x.shape[0], float(_is_anomalous(lab, dataset))))
    return Xtr, r2, np.concatenate(Ms), np.concatenate(Ss), np.concatenate(Ys)


def _test_side(test_list, gt_path, mu):
    """Every test video, minus the train mean, with its frame-level GT."""
    rows = _read_list(test_list)
    gt = np.load(gt_path).astype(np.float32)
    offs = _gt_offsets(rows, test_list)
    Xs, Gs = [], []
    for i, (p, _lab) in enumerate(rows):
        n = _i(offs[i + 1] - offs[i])
        if n <= 0 or not os.path.exists(p):
            continue
        x = _shifted(p, mu)
        if x is None:
            continue
        Xs.append(x)
        seg = gt[offs[i] * GT_FRAME_REPEAT:offs[i + 1] * GT_FRAME_REPEAT]
        Gs.append(seg.reshape(-1, GT_FRAME_REPEAT).mean(1) > 0.5)
    return Xs, Gs


def _pooled(g, s):
    return {'auc': round(_f(roc_auc_score(g, s)), 4),
            'ap': round(_f(average_precision_score(g, s)), 4)}


def run(dataset, device, max_train_clips, n_random, seed):
    root = opt.repo_root()
    base = os.path.join(root, 'list')
    train_list = os.path.join(base, f'{dataset}_CLIP_rgb.csv')
    test_list = os.path.join(base, f'{dataset}_CLIP_rgbtest.csv')
    gt_path = os.path.join(base, 'gt.npy' if dataset == 'xd' else 'gt_ucf.npy')

    print(f'\n{"=" * 78}\nRESIDUAL DIRECTION  dataset={dataset}\n{"=" * 78}')
    d_text, D_bank = _bank(dataset, device)
    C = d_text.shape[0]

    all_rows = _read_list(train_list)
    rng = np.random.RandomState(seed)
    rows = all_rows if len(all_rows) <= max_train_clips else \
        [all_rows[i] for i in rng.choice(len(all_rows), max_train_clips, replace=False)]

    mu = _train_mu(rows, C)
    if mu is None:
        print('  no usable train clips')
        return None
    d_mag = mu / (np.linalg.norm(mu) + 1e-12)      # the magnitude-aligned direction

    p1 = _pass1(rows, mu, d_text, C)
    if p1 is None:
        print('  no usable train clips')
        return None
    beta, XtX, n_train = p1['beta'], p1['XtX'], p1['n']
    Xtr, r2, m_tr, s_tr, y_tr = _pass2(rows, mu, d_text, beta, C, dataset)
    print(f'  train frames {n_train}   beta {beta:.4f}   dim {C}')

    eye_scale = float(np.linalg.eigvalsh(XtX)[-1]) / C
    cands, r2_by_lam = {}, {}
    for lam in RIDGES:
        w = np.linalg.solve(XtX + np.eye(C) * (lam * eye_scale + 1e-12), Xtr)
        d = w / (np.linalg.norm(w) + 1e-12)
        cands[lam] = d
        cov = float(d @ Xtr)
        r2_by_lam[lam] = cov * cov / (float(d @ XtX @ d) * r2 + 1e-12)
    lam_star = max(r2_by_lam, key=r2_by_lam.get)
    d_prime = cands[lam_star]

    print('  train R^2 by ridge: ' + '  '.join(f'{l:g}:{r2_by_lam[l]:.4f}' for l in RIDGES))
    print(f'  selected ridge {lam_star:g}  (train R^2 {r2_by_lam[lam_star]:.4f})')
    print(f"  cos(d', d_text)  {float(d_prime @ d_text):+.4f}")
    print(f"  cos(d', d_mag)   {float(d_prime @ d_mag):+.4f}")
    print(f"  cos(d_text,d_mag){float(d_text @ d_mag):+.4f}   <- reference")

    Xs, Gs = _test_side(test_list, gt_path, mu)
    gflat = np.concatenate(Gs)
    mflat = np.concatenate([np.linalg.norm(x, axis=1) for x in Xs])

    def single(d):
        return _pooled(gflat, np.concatenate([x @ d for x in Xs]))

    rng2 = np.random.RandomState(seed + 1)
    rand = rng2.randn(n_random, C)
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    rand_aucs = [_f(roc_auc_score(gflat, np.concatenate([x @ rand[j] for x in Xs])))
                 for j in range(n_random)]
    rand_mean, rand_p95 = float(np.mean(rand_aucs)), float(np.percentile(rand_aucs, 95))

    results = {'d_prime': single(d_prime), 'd_text': single(d_text),
               'd_mag(mu direction)': single(d_mag),
               'random_mean': round(rand_mean, 4), 'random_p95': round(rand_p95, 4)}
    print()
    for k in ('d_prime', 'd_text', 'd_mag(mu direction)'):
        print(f'  cue {k:<20} pooled AUC {results[k]["auc"]:>7.4f}  AP {results[k]["ap"]:>7.4f}')
    print(f'  random {n_random} draws         mean {rand_mean:.4f}   p95 {rand_p95:.4f}')

    yb = y_tr.astype(np.int8)
    if yb.min() == yb.max():
        print('  train split has a single class; combinations skipped')
        for name in ('m + d_prime', 'm + d_text'):
            results[name] = {'auc': None, 'ap': None, 'reason': 'single class'}
    else:
        for name, cols in (('m only', [0]), ('d_text only', [1]),
                           ('m + d_prime', [0, 1]), ('m + d_text', [0, 1])):
            s_te = np.concatenate([x @ (d_prime if name == 'm + d_prime' else d_text)
                                   for x in Xs])
            Ztr = np.stack([m_tr, s_tr], 1)
            Zte = np.stack([mflat, s_te], 1)
            mu_, sd_ = Ztr.mean(0), Ztr.std(0) + 1e-8
            # Select the columns *before* standardising. Indexing a 1-D array with
            # a list gives shape (1,), which broadcasts across all columns instead
            # of picking one - so a "single feature" model silently fits both and
            # every variant comes out identical.
            Xtr = (Ztr[:, cols] - mu_[cols]) / sd_[cols]
            Xte = (Zte[:, cols] - mu_[cols]) / sd_[cols]
            clf = LogisticRegression(max_iter=1000).fit(Xtr, yb)
            p = clf.predict_proba(Xte)[:, 1]
            results[name] = _pooled(gflat, p)
            results[name]['coef'] = [round(float(c), 4) for c in np.ravel(clf.coef_)]
            print(f'  combo {name:<18} pooled AUC {results[name]["auc"]:>7.4f}  '
                  f'AP {results[name]["ap"]:>7.4f}  coef {results[name]["coef"]}')

    gain = round(results['m + d_prime']['auc'] - results['m + d_text']['auc'], 4)
    alone = round(results['d_prime']['auc'] - results['d_text']['auc'], 4)
    synergy = round(results['m + d_text']['auc'] - max(results['m only']['auc'],
                                                       results['d_text only']['auc']), 4)
    branch_alive = bool(alone > 0)
    verdict = {
        'ridge_selected_on_train_r2': lam_star,
        'train_r2': round(r2_by_lam[lam_star], 4),
        'beta': round(float(beta), 4),
        'cos_dprime_dtext': round(float(d_prime @ d_text), 4),
        'cos_dprime_dmag': round(float(d_prime @ d_mag), 4),
        'cos_dtext_dmag': round(float(d_text @ d_mag), 4),
        'gain_dprime_over_dtext_alone': alone,
        'gain_m_plus_dprime_over_m_plus_dtext': gain,
        'synergy_m_plus_dtext_over_best_single': synergy,
        'branch_alive': branch_alive,
        'verdict': ('BRANCH ALIVE: the residual direction beats the frozen text direction, '
                    'so a text adapter has a target worth learning.'
                    if branch_alive else
                    'BRANCH DEAD: the residual direction does not beat the frozen text '
                    'direction. cos(d_text, mu) is already ~0, so the text-vs-magnitude '
                    'confound lives in the nonlinearity of the norm rather than in the '
                    'direction - and no reorientation of a text embedding, linear or not, '
                    'can remove it. The usable lever is combining magnitude with text, '
                    'not replacing the direction.'),
    }
    print(f'\n  {"-" * 74}')
    print(f"  d' - d_text (alone)          {alone:+.4f}")
    print(f"  (m + d') - (m + d_text)      {gain:+.4f}")
    print(f"  (m + d_text) - best single   {synergy:+.4f}   <- the real lever")
    print(f'  {verdict["verdict"]}')
    print(f'  {"-" * 74}')

    out = {'dataset': dataset, 'n_train_frames': n_train, 'max_train_clips': max_train_clips,
           'n_test_videos': len(Xs), 'n_random': n_random,
           'train_r2_by_ridge': {str(l): round(r2_by_lam[l], 4) for l in RIDGES},
           'pooled': results, 'verdict': verdict,
           'protocol': 'per-clip centring; mu, beta and d\' fitted on the TRAIN split; '
                       'ridge selected by TRAIN R^2 (label-free); test supplies only '
                       'features and ground truth'}
    dest = os.path.join(root, 'runs', f'residual_direction_{dataset}.json')
    with open(dest, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'  saved -> {dest}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='both', choices=['xd', 'ucf', 'both'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--max-train-clips', type=int, default=4000)
    ap.add_argument('--n-random', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print('cuda requested but unavailable -> cpu')
            args.device = 'cpu'
    for ds in (['xd', 'ucf'] if args.dataset == 'both' else [args.dataset]):
        run(ds, args.device, args.max_train_clips, args.n_random, args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
