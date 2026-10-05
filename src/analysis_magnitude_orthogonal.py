"""Does the text direction carry anomaly signal that feature magnitude does not?

    python analysis_magnitude_orthogonal.py --dataset both

Pre-registered spec: ``magnitude_orthogonal_design.md`` (repo root). Read it
first - the decision rules at the end of that document are fixed in advance
precisely so the result cannot be read after the fact in whichever direction is
convenient.

Why this exists. The gate says PROCEED, but only on the ``class_max`` readout.
On the readout E1 actually injects - one direction per video, so the mean over
the class bank - XD scores 0.6759 against 0.6905 for the free magnitude cue,
paired 95% CI [-0.0265, -0.0026]. The direction the augmenter pushes features
along is therefore *significantly worse* than ``||f - mu||``, which costs
nothing to compute. Either the +0.026 that beat random directions is magnitude
rather than semantics, or E1-E3 is measuring a premise that does not hold.

Reported under both conventions this project uses, because they rank cues
differently (``.memory/pooled-vs-within-video-metric``) and because the premise
verdict flips between them. Any fitted quantity is estimated on the TRAIN split
and applied to test.

Sign handling is the whole ballgame here and is easy to get silently wrong. A
direction's sign is arbitrary, so the gate takes ``max(a, 1-a)`` per video. That
is safe on a full ~154-frame video and catastrophic on a ~50-frame magnitude
stratum, where ``max(a, 1-a)`` of a *pure noise* score routinely reaches 0.9.
Against a single fixed text direction the bias is modest; against a 200-draw null
its p95 is set by the noisiest draw, and every text readout then appears to lose
by 0.2. Measured before this was fixed: signed null AUCs 0.50, free-sign null p95
0.99. So every subset readout here uses ONE sign calibrated globally over
videos - that removes the small-sample inflation while leaving the sign free -
and only the full-video ``m`` / ``s`` anchors keep the gate's per-video
convention so they reproduce the published numbers exactly.

No training, no checkpoints: frozen features plus the frozen CLIP text encoder.
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

FULL_VIDEO = ('m', 's')          # per-video free sign, matching probe_linear_separability
SUBSET = ('s_res_m', 'm_res_s', 's_strat', 's_strat_min', 's_match')
# An AUC needs more than one point of each class to mean anything. A magnitude
# stratum holding 1 positive and 1 negative has AUC exactly 0 or 1, so orientation
# turns half of them into 1.0 and the null's minimum climbs above its maximum.
MIN_CLASS = 5
# A pooled-AUC difference of a thousandth is not "additive value", it is noise, and
# a bare ``>`` test calls it a pass. On UCF the two-feature model beats the
# magnitude-only model by 0.0008 AUC while assigning the text cue a weight of 0.094
# against magnitude's -1.083, i.e. it learned to ignore it. The logistic's own
# weight ratio is reported too, because a feature the model can drop is not a
# feature the model used.
D3_MIN_GAIN = 0.005


def _z(s):
    return (s - s.mean()) / (s.std() + 1e-8)


def _auc(g, s, free_sign=True):
    if not (0 < g.sum() < len(g)):
        return np.nan
    a = _f(roc_auc_score(g, s))
    if not np.isfinite(a):
        return np.nan
    return max(a, 1.0 - a) if free_sign else a


def _resid_on(y, x):
    """y with its linear component on x removed. Both are one video's frames.

    A degenerate x (no variance) returns a centred y, which is the honest
    "nothing was explained" answer rather than a division by ~0.
    """
    yc = y - y.mean()
    xc = x - x.mean()
    den = float((xc * xc).sum())
    if den < 1e-12:
        return yc
    return yc - (float((xc * yc).sum()) / den) * xc


def _stratified_auc(m, s, g, n_bins=3):
    """R2: AUC of s inside magnitude bins, plus the worst single bin.

    Non-parametric, so it assumes nothing about how magnitude contaminates the
    text score. The minimum is reported next to the mean because a signal that
    survives only in the lowest-magnitude bin is a different finding from one
    that survives everywhere, and the mean alone would hide it.
    """
    edges = np.quantile(m, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    aucs = []
    for b in range(n_bins):
        sel = ((m >= edges[b]) & (m <= edges[b + 1])) if b == n_bins - 1 \
            else ((m >= edges[b]) & (m < edges[b + 1]))
        if int(g[sel].sum()) < MIN_CLASS or int((~g[sel]).sum()) < MIN_CLASS:
            continue
        a = _auc(g[sel], s[sel], free_sign=False)   # signed; sign applied globally later
        if np.isfinite(a):
            aucs.append(a)
    if not aucs:
        return np.nan, np.nan, 0
    return float(np.mean(aucs)), float(np.min(aucs)), len(aucs)


def _mid(m):
    """R3: the central third of the magnitude range - maximum matching, fewer frames."""
    lo, hi = np.quantile(m, [1.0 / 3.0, 2.0 / 3.0])
    return (m >= lo) & (m <= hi)


def readouts(m, s, g):
    """Signed per-frame readouts for one video. Signs are applied later, globally."""
    out = {'m': _auc(g, _z(m), free_sign=True),
           's': _auc(g, _z(s), free_sign=False),
           's_res_m': _auc(g, _z(_resid_on(s, m)), free_sign=False),
           'm_res_s': _auc(g, _z(_resid_on(m, s)), free_sign=False)}
    strat, strat_min, n_bins = _stratified_auc(m, s, g)
    out['s_strat'] = strat
    out['s_strat_min'] = strat_min
    out['s_strat_bins'] = float(n_bins)
    mid = _mid(m)
    out['s_match'] = (np.nan if int(g[mid].sum()) < MIN_CLASS
                      or int((~g[mid]).sum()) < MIN_CLASS
                      else _auc(g[mid], s[mid], free_sign=False))
    return out


def _orient(vals, key):
    """One sign for a whole readout, chosen by where the aggregate mass sits."""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return list(vals), 1.0
    if key in FULL_VIDEO:
        return [max(x, 1.0 - x) if np.isfinite(x) else np.nan for x in vals], 1.0
    if v.mean() >= 0.5:
        return list(vals), 1.0
    return [1.0 - x if np.isfinite(x) else np.nan for x in vals], -1.0


def _ci95(x):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if x.size < 2:
        return [float('nan'), float('nan')], float('nan'), int(x.size)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(x.size))
    return [m - 1.96 * se, m + 1.96 * se], m, int(x.size)


def _load_mixed(test_list, gt_path, n_videos, rng):
    """Test videos with mixed GT, sampled exactly as the gate samples them."""
    rows = _read_list(test_list)
    gt = np.load(gt_path).astype(np.float32)
    offs = _gt_offsets(rows, test_list)
    videos, images = [], []
    for i, (p, _lab) in enumerate(rows):
        n = _i(offs[i + 1] - offs[i])
        if n <= 0:
            continue
        seg = gt[offs[i] * GT_FRAME_REPEAT:offs[i + 1] * GT_FRAME_REPEAT]
        g = seg.reshape(-1, GT_FRAME_REPEAT).mean(1) > 0.5
        if not (0 < g.sum() < len(g)):
            continue
        if not os.path.exists(p):
            continue
        videos.append(np.load(p).astype(np.float32))
        images.append(g)
    pick = rng.choice(len(videos), size=min(n_videos, len(videos)), replace=False)
    return [videos[i] for i in pick], [images[i] for i in pick], len(videos)


def _load_all(test_list, gt_path):
    """Every test video in list order, for the pooled protocol."""
    rows = _read_list(test_list)
    gt = np.load(gt_path).astype(np.float32)
    offs = _gt_offsets(rows, test_list)
    feats, gts = [], []
    for i, (p, _lab) in enumerate(rows):
        n = _i(offs[i + 1] - offs[i])
        if n <= 0 or not os.path.exists(p):
            continue
        feats.append(np.load(p).astype(np.float32))
        seg = gt[offs[i] * GT_FRAME_REPEAT:offs[i + 1] * GT_FRAME_REPEAT]
        gts.append(seg.reshape(-1, GT_FRAME_REPEAT).mean(1) > 0.5)
    return feats, gts


def _is_anomalous(label, dataset):
    s = str(label).strip()
    if not s:
        return False
    if dataset == 'ucf':
        return s.lower() != 'normal'
    return any(t.upper() != 'A' for t in s.split('-') if t not in ('', '0'))


def _train_cues(train_list, dataset, DC, mu, rng, max_clips):
    """Frame-level (m, s) with weak clip-level labels, from the TRAIN split only."""
    rows = _read_list(train_list)
    if len(rows) > max_clips:
        rows = [rows[i] for i in rng.choice(len(rows), size=max_clips, replace=False)]
    ms, ss, ys = [], [], []
    for p, lab in rows:
        if not os.path.exists(p):
            continue
        c = np.load(p).astype(np.float32) - mu
        ms.append(np.linalg.norm(c, axis=1))
        ss.append((c @ DC.T).mean(1))
        ys.append(np.full(len(c), float(_is_anomalous(lab, dataset))))
    return np.concatenate(ms), np.concatenate(ss), np.concatenate(ys)


def _fit_eval(m_tr, s_tr, y_tr, feats, gts, mu, DC):
    """D3: does adding s to m help on the pooled protocol? Weights from TRAIN only.

    Frame-level targets are the clip-level weak label, which is the same
    supervision VadCLIP's CLAS2 head trains on, so this is a like-for-like
    question: would a scorer built on these two features do better than one built
    on the free magnitude cue alone.
    """
    Dm, Ds = float(m_tr.mean()), float(s_tr.mean())
    Rm, Rs = float(m_tr.std()), float(s_tr.std())
    Ztr = np.stack([(m_tr - Dm) / (Rm + 1e-8), (s_tr - Ds) / (Rs + 1e-8)], 1)
    y = y_tr.astype(np.int8)
    if y.min() == y.max():
        return {k: {'auc': None, 'ap': None, 'reason': 'single class'}
                for k in ('m', 's', 'm+s')}
    Zte = np.concatenate([
        np.stack([(np.linalg.norm(v - mu, axis=1) - Dm) / (Rm + 1e-8),
                  (((v - mu) @ DC.T).mean(1) - Ds) / (Rs + 1e-8)], 1) for v in feats])
    gflat = np.concatenate(gts)
    out = {}
    for name, keys in (('m', [0]), ('s', [1]), ('m+s', [0, 1])):
        clf = LogisticRegression(max_iter=1000).fit(Ztr[:, keys], y)
        p = clf.predict_proba(Zte[:, keys])[:, 1]
        out[name] = {'auc': _f(roc_auc_score(gflat, p)),
                     'ap': _f(average_precision_score(gflat, p)),
                     'coef': [float(c) for c in np.ravel(clf.coef_)]}
    return out


def run(dataset, device, n_videos, n_random, max_train_clips, seed,
        d2_rule='kill', out_dir='runs'):
    root = opt.repo_root()
    base = os.path.join(root, 'list')
    test_list = os.path.join(base, f'{dataset}_CLIP_rgbtest.csv')
    train_list = os.path.join(base, f'{dataset}_CLIP_rgb.csv')
    gt_path = os.path.join(base, 'gt.npy' if dataset == 'xd' else 'gt_ucf.npy')

    print(f'\n{"=" * 78}\nMAGNITUDE-ORTHOGONAL PREMISE TEST  dataset={dataset}\n{"=" * 78}')
    dirs, _ = build_class_directions('ViT-B/16', dataset, device=device, verbose=False)
    D = np.asarray(dirs.cpu().numpy() if hasattr(dirs, 'cpu') else dirs, dtype=np.float32)
    D = D[1:][:len(DATASET_CLASS_NAMES[dataset]) - 1]

    rng = np.random.RandomState(seed)
    V, G, n_mixed = _load_mixed(test_list, gt_path, n_videos, rng)
    mu = np.concatenate(V, 0).mean(0)
    mu_n = mu / (np.linalg.norm(mu) + 1e-9)

    def clean(u):
        u = u - float(u @ mu_n) * mu_n
        return u / (np.linalg.norm(u) + 1e-9)

    DC = np.array([clean(d) for d in D])
    RC = np.array([clean(u) for u in
                   rng.randn(n_random, D.shape[1]).astype(np.float32)])

    keys = ['m', 's'] + list(SUBSET)
    collect = keys + ['s_strat_bins']
    text = {k: [] for k in collect}
    null = {k: np.full((n_random, len(V)), np.nan) for k in keys}
    for i, (v, g) in enumerate(zip(V, G)):
        c = v - mu
        m = np.linalg.norm(c, axis=1)
        r = readouts(m, (c @ DC.T).mean(1), g)
        for k in collect:
            text[k].append(r[k])
        Sr = c @ RC.T
        for j in range(n_random):
            rj = readouts(m, Sr[:, j], g)
            for k in keys:
                null[k][j, i] = rj[k]

    agg = {}
    for k in keys + ['s_strat_bins']:
        if k == 's_strat_bins':
            vals = [v for v in text[k] if np.isfinite(v)]
            agg[k] = {'mean': round(float(np.mean(vals)), 3) if vals else None,
                      'note': 'valid magnitude strata per video (of 3); s_strat_min '
                              'is only interpretable when this is 2 or 3'}
            continue
        tv, tsign = _orient(text[k], k)
        nv = []
        draw_means = []
        for j in range(n_random):
            v, _ = _orient(null[k][j].tolist(), k)
            nv.extend(v)
            finite = [x for x in v if np.isfinite(x)]
            if finite:
                draw_means.append(float(np.mean(finite)))
        ci, mean, n = _ci95(tv)
        null_ci, null_mean, _ = _ci95(nv)
        arr = np.asarray([x for x in nv if np.isfinite(x)])
        null_p95 = float(np.percentile(arr, 95)) if arr.size else float('nan')
        # §6 says "p95" and the code uses the mean; neither is the right reference.
        # The text statistic is a MEAN over videos, so the null has to be the
        # distribution of per-draw MEANS, not the p95 of pooled per-video values
        # (`null_p95`, inflated by n_random x n_videos draws) and not the flat mean.
        # Recorded because it is the one comparison that can only hurt the text
        # direction: it is a strictly higher bar than the mean it replaces, so
        # adding it post hoc cannot be a way of rescuing the premise.
        null_mean_p95 = (float(np.percentile(draw_means, 95))
                         if draw_means else float('nan'))
        entry = {'mean': round(mean, 4) if np.isfinite(mean) else None,
                 'ci95': [round(x, 4) for x in ci] if np.isfinite(ci[0]) else None,
                 'n_videos': n, 'sign_convention': tsign}
        if k not in FULL_VIDEO:
            entry.update({
                'null_mean': round(null_mean, 4) if np.isfinite(null_mean) else None,
                'null_p95': round(null_p95, 4) if np.isfinite(null_p95) else None,
                'null_mean_p95': (round(null_mean_p95, 4)
                                  if np.isfinite(null_mean_p95) else None),
                'margin_over_null_mean': (round(mean - null_mean, 4)
                                          if np.isfinite(mean) and np.isfinite(null_mean)
                                          else None),
                'margin_over_null_p95': (round(mean - null_p95, 4)
                                         if np.isfinite(mean) and np.isfinite(null_p95)
                                         else None),
                'margin_over_null_mean_p95': (
                    round(mean - null_mean_p95, 4)
                    if np.isfinite(mean) and np.isfinite(null_mean_p95) else None)})
        agg[k] = entry

    print(f'\n  mixed-GT videos {len(V)} of {n_mixed}   random draws {n_random}')
    print(f'  {"readout":<12}{"mean":>8}{"95% CI":>20}{"null mean":>11}'
          f'{"vs null":>9}{"null p95":>10}')
    for k in keys:
        e = agg[k]
        if e['mean'] is None:
            print(f'  {k:<12}{"n/a":>8}')
            continue
        ci = e['ci95']
        cis = f'[{ci[0]:+.4f},{ci[1]:+.4f}]' if ci else 'n/a'
        nm, np95 = e.get('null_mean'), e.get('null_p95')
        mm = e.get('margin_over_null_mean')
        print(f'  {k:<12}{e["mean"]:>8.4f}{cis:>20}'
              f'{(f"{nm:>11.4f}" if nm is not None else "-" * 11)}'
              f'{(f"{mm:>+9.4f}" if mm is not None else "-" * 9)}'
              f'{(f"{np95:>10.4f}" if np95 is not None else "-" * 10)}')

    feats, gts = _load_all(test_list, gt_path)
    gflat = np.concatenate(gts)
    pooled = {}
    for name, fn in (('m', lambda c: np.linalg.norm(c, axis=1)),
                     ('s_mean', lambda c: (c @ DC.T).mean(1)),
                     ('s_class_max', lambda c: (c @ DC.T).max(1))):
        sc = np.concatenate([fn(v - mu) for v in feats])
        a = _f(roc_auc_score(gflat, sc))
        pooled[name] = {'auc': round(a, 4),
                        'ap': round(_f(average_precision_score(gflat, sc)), 4),
                        'auc_oracle_sign': round(max(a, 1 - a), 4)}
    print()
    for k, v in pooled.items():
        print(f'  pooled {k:<14} AUC {v["auc"]:>7.4f}  AP {v["ap"]:>7.4f}'
              f'   oracle-sign {v["auc_oracle_sign"]:.4f}')

    m_tr, s_tr, y_tr = _train_cues(train_list, dataset, DC, mu, rng, max_train_clips)
    combo = _fit_eval(m_tr, s_tr, y_tr, feats, gts, mu, DC)
    print(f'\n  D3 combination, weights fit on TRAIN ({y_tr.size} frames):')
    for k in ('m', 's', 'm+s'):
        c = combo[k]
        if c.get('auc') is None:
            print(f'    {k:<5} n/a ({c.get("reason")})')
        else:
            print(f'    {k:<5} pooled AUC {c["auc"]:.4f}  AP {c["ap"]:.4f}'
                  f'  coef {[round(x, 3) for x in c["coef"]]}')

    # Decisions compare against the null MEAN, not its p95: a single fixed text
    # direction cannot be ranked against the noisiest of 200 draws, which is the
    # same sample-size bias the gate warns about, just pointing the other way.
    d1 = all((agg[k].get('margin_over_null_mean') or 0) <= 0
             for k in ('s_res_m', 's_strat', 's_match'))
    # §6 words D1 as "R1/R2/R3 全部 ≤ 零假设 p95", but this compares against the null
    # MEAN. That is a substitution, not a transcription: null_p95 is the 95th percentile
    # of pooled per-video values (n_random x n_videos of them), while the text side is a
    # mean of per-video means, so ranking one against the other is the sample-size bias
    # again. Both bases are recorded because the literal one fires D1 on BOTH datasets,
    # which would reject the premise before D2 is ever consulted.
    d1_literal_p95 = all((agg[k].get('margin_over_null_p95') or 0) <= 0
                         for k in ('s_res_m', 's_strat', 's_match'))
    d2 = (agg['m_res_s'].get('margin_over_null_mean') or 0) <= 0
    ca_m = (combo.get('m') or {}).get('auc')
    ca_ms = (combo.get('m+s') or {}).get('auc')
    gain = (ca_ms - ca_m) if (ca_m is not None and ca_ms is not None) else None
    w_m, w_s = (combo.get('m+s') or {}).get('coef', [0.0, 0.0])[0:2] or (0.0, 0.0)
    d3 = not (gain is not None and gain > D3_MIN_GAIN)
    combo['gain_over_magnitude'] = round(gain, 4) if gain is not None else None
    combo['weight_ratio_text_over_magnitude'] = (
        round(abs(w_s / w_m), 3) if w_m not in (0.0, None) else None)
    st = agg['s_strat']
    c1 = ((st.get('margin_over_null_mean') or 0) > 0
          and st.get('ci95') is not None and st['ci95'][0] > 0)
    # §6 words C1 the same way it words D1 - R2 (stratified) > null p95 with the CI
    # excluding 0 - so the mean-instead-of-p95 substitution sits in the CONFIRM gate as
    # well as the kill gate. It is therefore one systematic re-implementation of §6, not
    # a single slip, and an amendment has to cover both at once.
    c1_literal_p95 = ((st.get('margin_over_null_p95') or 0) > 0
                      and st.get('ci95') is not None and st['ci95'][0] > 0)

    def _verdict(d2_fires):
        """The verdict given whether D2 counts as triggered.

        §6 is ambiguous about D2, and the three readings do not agree on XD, so the
        reader gets all three. They differ only in *when D2 fires*, not in the
        numbers: ``kill`` takes §6's mechanical listing (the magnitude residual is
        null on its own -> reject); ``joint`` is what this script originally shipped
        (D2 counts only alongside D1); ``symmetric`` takes §6's stated rationale,
        "两个线索是同一个" - if they are the *same* cue then both residuals must be
        null, and on XD the text residual is +0.0781, so they are not the same.
        """
        if d1 and d2_fires:
            return ('PREMISE REJECTED: neither direction carries magnitude-independent '
                    'signal.')
        if d2_fires:
            return ('PREMISE REJECTED (D2): once the text direction is '
                    'removed the magnitude cue has no signal left, so the two cues are '
                    'not separable and "text" is a misnomer for magnitude.')
        if d1 and not c1:
            return ('PREMISE NOT SUPPORTED: no magnitude-independent signal in the text '
                    'direction on any of the three residual readouts.')
        if c1 and d3:
            return ('PREMISE REDUNDANT: the text direction carries magnitude-independent '
                    'signal, but it adds nothing usable over the free magnitude cue - the '
                    'two-feature model does not clear the magnitude-only model by a '
                    'meaningful margin on the pooled protocol.')
        if c1:
            return ('PREMISE CONFIRMED: the text direction carries magnitude-independent '
                    'signal and adds value over the magnitude cue on the pooled protocol.')
        return 'INCONCLUSIVE: mixed across readouts - read the table before concluding.'

    s_res_m_margin = agg['s_res_m'].get('margin_over_null_mean') or 0.0
    m_res_s_margin = agg['m_res_s'].get('margin_over_null_mean') or 0.0
    d2_single = bool(m_res_s_margin <= 0)
    d2_fires = {'kill': d2_single,
                'joint': bool(d1 and d2_single),
                'symmetric': bool(d2_single and s_res_m_margin <= 0)}

    top = _verdict(d2_fires[d2_rule])
    verdicts = {r: _verdict(d2_fires[r]) for r in ('kill', 'joint', 'symmetric')}
    # A confirm produced by `joint` is not a pre-registration result, and someone
    # reading this JSON in six months cannot infer that from the verdict string alone.
    # Label it at the source so the artifact carries its own provenance.
    amendment = (d2_rule != 'kill')
    decision = {'D1_all_residual_readouts_null': bool(d1),
                'D1_null_basis': "mean (spec §6 says p95; see comment above)",
                'D1_fires_under_literal_p95': bool(d1_literal_p95),
                'D2_both_residuals_null': bool(d2),
                'D3_no_additive_value_over_magnitude': bool(d3),
                'C1_magnitude_orthogonal_text_signal': bool(c1),
                'C1_fires_under_literal_p95': bool(c1_literal_p95),
'residual_is_within_video': bool(c1), 'verdict': top,
                'd2_rule_applied': d2_rule,
                'verdict_under_joint_d2': verdicts['joint'],
                'verdict_under_symmetric_d2': verdicts['symmetric'],
                'verdicts_all_d2_readings': verdicts,
                'd2_residual_margins': {'s_res_m': round(s_res_m_margin, 4),
                                        'm_res_s': round(m_res_s_margin, 4)},
                'd2_readings_disagree': len(set(verdicts.values())) > 1,
                'verdict_is_preregistered': bool(not amendment),
                'amendment_note': (
                    'D2 applied jointly with D1, which magnitude_orthogonal_design.md '
                    '§6 does NOT authorise (it lists D2 under the kill conditions and '
                    'says any one suffices). This verdict is a POST-HOC AMENDMENT and '
                    'must not be reported as a pre-registered result; it requires a '
                    'dated ratification. See .memory/premise-decision-d2-narrowed.md'
                    if amendment else
                    'literal §6: D2 alone rejects. This is the pre-registered rule.')}

    print(f'\n  {"-" * 74}')
    for k in ('D1_all_residual_readouts_null', 'D2_both_residuals_null',
              'D3_no_additive_value_over_magnitude', 'C1_magnitude_orthogonal_text_signal'):
        print(f'  {k:<44} {decision[k]}')
    print(f'\n  {top}\n  {"-" * 74}')

    out = {'dataset': dataset, 'n_test_videos_used': len(V),
           'n_mixed_gt_available': n_mixed, 'n_random': n_random, 'seed': seed,
           'magnitude_cue': '||f - mu||, mu = mean of the sampled test features '
                             '(identical to probe_linear_separability)',
           'text_cue': 'mean over the class bank of the mu-orthogonalised class '
                       'directions - the readout E1 injects, since augment_and_score '
                       'takes dirs[class_idx]. Note this is NOT the gate\'s '
                       'text_auc_mean: the gate averages per-class AUCs, this projects '
                       'onto the mean direction first, so the two differ by ~0.01 and '
                       'neither is wrong.',
           'sign_convention': 'per-video free sign for m/s (gate convention); one '
                              'globally calibrated sign for every subset readout, '
                              'because free sign on a ~50-frame stratum inflates a '
                              'pure-noise score to ~0.9',
           'within_video': agg, 'pooled': pooled,
           'combination_train_fit': combo, 'decision': decision}
    dest_dir = out_dir if os.path.isabs(out_dir) else os.path.join(root, out_dir)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f'mag_ortho_{dataset}.json')
    with open(dest, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'  saved -> {dest}')
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='both', choices=['xd', 'ucf', 'both'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-videos', type=int, default=400)
    ap.add_argument('--n-random', type=int, default=200)
    ap.add_argument('--max-train-clips', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d2-rule', default='kill',
                    choices=['kill', 'joint', 'symmetric'],
                    help="when D2 counts as triggered. §6 is ambiguous and the three "
                         "readings do not agree on XD, so all three verdicts are always "
                         "written to the JSON and this only picks which is labelled "
                         "primary. 'kill' (default) takes §6's mechanical listing: D2 "
                         "alone rejects. 'joint' is what this script originally "
                         "shipped: D2 counts only alongside D1. 'symmetric' takes §6's "
                         "stated rationale - '两个线索是同一个' means if they are the "
                         "same cue then BOTH residuals must be null, and on XD the text "
                         "residual is +0.0781, so they are not the same. Anything other "
                         "than 'kill' marks the verdict as a POST-HOC AMENDMENT. See "
                         ".memory/premise-decision-d2-narrowed.md")
    ap.add_argument('--out-dir', default='runs',
                    help='where to write mag_ortho_<dataset>.json. Point a smoke run '
                         'somewhere else: a reduced --n-videos/--n-random run writes '
                         'degraded numbers and a clean-looking verdict into the same '
                         'filename, which has already overwritten a real artifact once. '
                         'Full run: n_videos=400 (XD) / all 140 (UCF), n_random=200.')
    return ap


def main():
    args = build_parser().parse_args()
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print('cuda requested but unavailable -> cpu')
            args.device = 'cpu'
    for ds in (['xd', 'ucf'] if args.dataset == 'both' else [args.dataset]):
        run(ds, args.device, args.n_videos, args.n_random, args.max_train_clips,
            args.seed, args.d2_rule, args.out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
