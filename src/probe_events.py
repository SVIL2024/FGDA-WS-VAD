"""Go/no-go probe: do event prompts beat class-label directions?

    python probe_events.py --dataset xd
    python probe_events.py --dataset both --n-videos 400

This answers one question before any training is attempted: does scoring a frame
against a *dictionary of event descriptions* locate anomalous frames better than
projecting it onto a *single class direction*? Acting on the answer either way is
much cheaper than finding out from an ablation.

The statistic is deliberately the same one ``freq_text_text.probe_linear_separability``
uses - score each frame, z-score inside each test video, AUC against that
video's GT segments upsampled by ``GT_FRAME_REPEAT``, averaged over videos that
contain both classes. That z-scoring is this project's LOCALISATION convention,
not VadCLIP's evaluation: the published AP/AUC pool every test frame with no
per-video normalisation (``analysis_official_metric.py`` measures both
conventions for the same cues). Reusing it means the numbers here are directly
comparable to the ones already recorded in ``runs/precondition_*.json``, where:

    XD  : ceiling 0.744, ||f|| 0.691, random 0.650, class-direction 0.676
    UCF : ceiling 0.735, ||f|| 0.698, random 0.685, class-direction 0.702

Four things are scored per dataset:

``class_direction``
    The existing ``e_class - e_normal`` per-class direction, the strongest single
    direction available to the current implementation. This is the thing to beat.
``event_best_prompt``
    The single best event prompt by AUC. An upper bound on what one sentence can
    do, and the honest way to report a 56- or 112-prompt dictionary.
``event_psi_max``
    ``max`` over prompts of the per-frame response - one continuous anomaly
    evidence signal that uses the whole dictionary at once, which is what the
    training branch actually consumes.
``event_psi_topk``
    Mean of the top-k prompt responses per frame, the more robust aggregation
    (a single prompt can spike on an unrelated frame).

A random-prompt null runs through the identical pipeline, so "prompts are better
than directions" is measured against the same anisotropy baseline rather than
against chance. CLIP features are strongly anisotropic - an arbitrary unit vector
already scores 0.65-0.69 here - so a raw AUC above 0.5 would prove nothing.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import freq_text_events as E  # noqa: E402
import freq_text_options as opt  # noqa: E402
from freq_text_text import GT_FRAME_REPEAT  # noqa: E402
from freq_text_text import _gt_offsets, _read_list  # noqa: E402
from freq_text_text import build_class_directions  # noqa: E402


def _f(x, what='value'):
    """``float`` with a named failure point (original exception type preserved).

    Results-critical probe code: a failed conversion must never become NaN
    silently, so the exception type is preserved and only the message gains
    context.
    """
    try:
        return float(x)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f'non-numeric {what}: {x!r}') from exc


def _i(x, what='value'):
    """``int`` with a named failure point (original exception type preserved)."""
    try:
        return int(x)
    except (TypeError, ValueError, OverflowError) as exc:
        raise type(exc)(f'non-int {what}: {x!r}') from exc


@torch.no_grad()
def encode_prompts(clip_arch, prompts, device='cpu', batch=32):
    """Frozen-CLIP text embeddings for a list of prompts, L2-normalised."""
    import clip as clip_pkg
    from model import CLIPVAD  # noqa: F401  (import cost only; weights unused)

    model, _ = clip_pkg.load(clip_arch, device=device)
    model.eval()
    out = []
    for i in range(0, len(prompts), batch):
        tok = clip_pkg.tokenize(prompts[i:i + batch]).to(device)
        emb = model.encode_text(tok, vanilla=True)
        out.append(emb.float())
    e = torch.cat(out, 0)
    return e / (e.norm(dim=-1, keepdim=True) + 1e-8)


def load_test_videos(test_list, gt_path, n_videos, seed=0):
    """``(V, G)``: frame features and boolean per-frame GT, mixed-GT videos only."""
    rng = np.random.RandomState(seed)
    rows = _read_list(test_list)
    gt = np.load(gt_path).astype(np.float32)
    offs = _gt_offsets(rows, test_list)
    V, G = [], []
    for i, (p, _l) in enumerate(rows):
        n = _i(offs[i + 1] - offs[i])
        if n <= 0 or not os.path.exists(p):
            continue
        seg = gt[offs[i] * GT_FRAME_REPEAT:offs[i + 1] * GT_FRAME_REPEAT]
        g = seg.reshape(-1, GT_FRAME_REPEAT).mean(1) > 0.5
        if 0 < g.sum() < len(g):
            V.append(np.load(p).astype(np.float32))
            G.append(g)
    if len(V) > n_videos:
        pick = rng.choice(len(V), size=n_videos, replace=False)
        V = [V[i] for i in pick]
        G = [G[i] for i in pick]
    return V, G


def score_per_video(signal_fn, V, G):
    """Mean within-video AUC of a per-video score signal.

    ``signal_fn(v)`` returns a ``(T,)`` score for one video. Sign is free (the
    head can flip it), so ``max(a, 1-a)`` is reported, matching the precondition
    probe's convention exactly.
    """
    out = []
    for i in range(min(len(V), len(G))):
        v, g = V[i], G[i]
        s = np.asarray(signal_fn(v), dtype=np.float64)
        if s.shape[0] != len(g):
            raise ValueError(f'length mismatch: score {s.shape[0]} vs gt {len(g)}')
        s = (s - s.mean()) / (s.std() + 1e-8)
        a = _f(roc_auc_score(g, s))
        out.append(max(a, 1.0 - a))
    return _f(np.mean(out)), np.array(out)


def run_dataset(dataset, n_videos, n_random, device, out_path):
    base = os.path.join(opt.repo_root(), 'list')
    test_list = os.path.join(base, f'{dataset}_CLIP_rgbtest.csv')
    gt_path = os.path.join(base, 'gt.npy' if dataset == 'xd' else 'gt_ucf.npy')

    print(f'\n{"="*74}\nEVENT-PROMPT PROBE  dataset={dataset}\n{"="*74}')
    V, G = load_test_videos(test_list, gt_path, n_videos)
    if len(V) < 20:
        print(f'  only {len(V)} usable test videos - skipping')
        return None
    print(f'  {len(V)} mixed-GT test videos, '
          f'{_i(np.mean([len(v) for v in V]))} frames each')

    mu = np.concatenate(V, 0).mean(0)
    mu_n = mu / (np.linalg.norm(mu) + 1e-9)

    # ---- baselines ------------------------------------------------------
    def _per_video(fn):
        return score_per_video(fn, V, G)[0]

    ceiling = None
    pos = np.concatenate([V[i][G[i]] for i in range(len(V))], 0).mean(0)
    neg = np.concatenate([V[i][~G[i]] for i in range(len(V))], 0).mean(0)
    w = (pos - neg)
    w = w - (w @ mu_n) * mu_n
    w /= (np.linalg.norm(w) + 1e-9)
    ceiling = _per_video(lambda v: (v - mu) @ w)
    norm_cue = _per_video(lambda v: np.linalg.norm(v - mu, axis=1))
    print(f'  supervised ceiling (data diff)   {ceiling:.4f}')
    print(f'  free ||f|| magnitude cue         {norm_cue:.4f}')

    # ---- class-label directions (the incumbent) --------------------------
    dirs, info = build_class_directions('ViT-B/16', dataset, device='cpu',
                                        verbose=False)
    d = dirs.numpy().astype(np.float32)
    names = E.class_names(dataset)
    cls_scores = {}
    for c in range(1, d.shape[0]):
        if np.linalg.norm(d[c]) < 1e-6:
            continue
        u = d[c] - (d[c] @ mu_n) * mu_n
        u /= (np.linalg.norm(u) + 1e-9)
        cls_scores[names[c]] = _per_video(lambda v, u=u: (v - mu) @ u)
    cls_arr = np.array(list(cls_scores.values()))
    best_cls = max(cls_scores, key=lambda k: _f(cls_scores[k]))
    print(f'\n  class-label direction  mean={cls_arr.mean():.4f} '
          f'max={cls_arr.max():.4f} ({best_cls})')

    # ---- event prompts ---------------------------------------------------
    prompts = E.flatten_prompts(dataset)
    owners = np.array(E.get_prompt_owner(dataset))
    emb = encode_prompts('ViT-B/16', prompts, device=device).cpu().numpy()
    print(f'  encoded {len(prompts)} event prompts -> {emb.shape}')

    # Prompt embeddings are used AS THEY COME, with no removal of a common
    # component. That step is correct for the class-direction table - those rows
    # are differences of near-parallel class embeddings and are dominated by a
    # shared offset - but it is actively harmful here. Measured on this data,
    # cos(prompt_emb, global_feature_mean) averages only 0.29, so centring throws
    # away real discriminative structure instead of nuisance: it dropped the best
    # XD prompt from 0.741 to 0.707 and made a *normal* prompt look like the best
    # one, which is how the bug was caught. Frames are centred (they do carry a
    # large common component) and prompts are left alone.

    def psi(v):
        """(T, P) frame-vs-prompt cosine responses."""
        x = v - mu
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        return x @ emb.T

    prompt_aucs = np.array([_per_video(lambda v, u=u: (v - mu) @ u) for u in emb])
    best = _i(prompt_aucs.argmax())
    print(f'\n  best single event prompt  {prompt_aucs[best]:.4f}  '
          f'[{names[owners[best]]}] "{prompts[best]}"')
    print(f'  worst single prompt       {prompt_aucs.min():.4f}')
    print(f'  prompt AUC mean           {prompt_aucs.mean():.4f}')

    psi_max = _per_video(lambda v: psi(v).max(axis=1))
    topk_aucs = {}
    for k in (2, 3, 5, 8):
        kk = min(k, emb.shape[0])
        auc_k = _per_video(lambda v, kk=kk: np.sort(psi(v), axis=1)[:, -kk:].mean(axis=1))
        topk_aucs[k] = round(auc_k, 4)
        print(f'  Psi top-{kk:<2} mean           {auc_k:.4f}')
    print(f'  Psi max over prompts      {psi_max:.4f}')

    # per-class best of the *event* prompts, to compare like with like
    ev_per_class = {names[c]: prompt_aucs[owners == c].max()
                    for c in sorted(set(owners)) if c > 0}
    ev_arr = np.array(list(ev_per_class.values()))

    # ---- random-prompt null ---------------------------------------------
    rng = np.random.RandomState(0)
    rnd = rng.randn(n_random, emb.shape[1]).astype(np.float32)
    rnd_aucs = np.array([_per_video(lambda v, u=u: (v - mu) @ u) for u in rnd])
    print(f'\n  random-direction null     mean={rnd_aucs.mean():.4f} '
          f'p95={np.percentile(rnd_aucs, 95):.4f} max={rnd_aucs.max():.4f} '
          f'(n={n_random})')

    rep = {
        'dataset': dataset,
        'n_test_videos': len(V),
        'supervised_ceiling': round(ceiling, 4),
        'magnitude_cue': round(norm_cue, 4),
        'class_direction_mean': round(_f(cls_arr.mean()), 4),
        'class_direction_max': round(_f(cls_arr.max()), 4),
        'class_direction_per_class': {k: round(v, 4) for k, v in cls_scores.items()},
        'event_per_class_best_mean': round(_f(ev_arr.mean()), 4),
        'event_per_class_best_max': round(_f(ev_arr.max()), 4),
        'event_per_class_best': {k: round(_f(v), 4) for k, v in ev_per_class.items()},
        'event_best_prompt': {'auc': round(_f(prompt_aucs[best]), 4),
                              'prompt': prompts[best],
                              'class': names[owners[best]]},
        'event_prompt_auc_mean': round(_f(prompt_aucs.mean()), 4),
        'event_psi_max': round(_f(psi_max), 4),
        'event_psi_topk': topk_aucs,
        'prompt_auc_all': [round(_f(a), 4) for a in prompt_aucs],
        'random_direction_mean': round(_f(rnd_aucs.mean()), 4),
        'random_direction_p95': round(_f(np.percentile(rnd_aucs, 95)), 4),
        'random_direction_max': round(_f(rnd_aucs.max()), 4),
        'n_random': n_random,
    }
    # the two comparisons that decide the plan
    rep['event_beats_class_max'] = bool(ev_arr.max() > cls_arr.max())
    rep['event_beats_class_mean'] = bool(ev_arr.mean() > cls_arr.mean())
    rep['psi_max_beats_class_max'] = bool(psi_max > cls_arr.max())
    rep['psi_max_beats_magnitude_cue'] = bool(psi_max > norm_cue)
    # Paired comparison over classes: the per-class event-best minus the
    # per-class directional score, with an interval. Both are computed on the same
    # videos, so a paired spread is the right uncertainty statement and is far
    # tighter than the across-class spread of the AUCs themselves.
    common = [c for c in cls_scores if c in ev_per_class]
    diff = np.array([ev_per_class[c] - cls_scores[c] for c in common])
    if len(diff) >= 3:
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        rep['per_class_diff_mean'] = round(_f(diff.mean()), 4)
        rep['per_class_diff_ci95'] = [round(_f(diff.mean() - 1.96 * se), 4),
                                      round(_f(diff.mean() + 1.96 * se), 4)]
    rep['n_test_videos'] = len(V)
    verdict = []
    if rep['event_beats_class_mean']:
        verdict.append(f"event prompts beat class labels on average "
                       f"({ev_arr.mean():.4f} vs {cls_arr.mean():.4f})")
    else:
        verdict.append(f"event prompts do NOT beat class labels on average "
                       f"({ev_arr.mean():.4f} vs {cls_arr.mean():.4f})")
    if rep['psi_max_beats_class_max']:
        verdict.append(f"the combined Psi beats the best class direction "
                       f"({psi_max:.4f} vs {cls_arr.max():.4f})")
    else:
        verdict.append(f"the combined Psi does not beat the best class direction "
                       f"({psi_max:.4f} vs {cls_arr.max():.4f})")
    if not rep['psi_max_beats_magnitude_cue']:
        verdict.append(f"and neither clears the free magnitude cue ({norm_cue:.4f})")
    rep['verdict'] = '; '.join(verdict)

    print(f'\n  {"-"*70}')
    if 'per_class_diff_ci95' in rep:
        lo, hi = rep['per_class_diff_ci95']
        print(f'  per-class event-best minus class-direction: '
              f'{rep["per_class_diff_mean"]:+.4f}  '
              f'[{lo:+.4f}, {hi:+.4f}]  (paired, {len(common)} classes)')
        print(f'  {"-"*70}')
    print(f'  VERDICT: {rep["verdict"]}')
    print(f'  {"-"*70}')

    if out_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        except OSError as exc:
            raise OSError(f'cannot create output dir for {out_path}') from exc
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(rep, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise OSError(f'cannot write report to {out_path}') from exc
        print(f'  saved -> {out_path}')
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='both', choices=['xd', 'ucf', 'both'])
    ap.add_argument('--n-videos', type=int, default=400)
    ap.add_argument('--n-random', type=int, default=200)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('cuda unavailable -> cpu')
        args.device = 'cpu'

    sets = ['xd', 'ucf'] if args.dataset == 'both' else [args.dataset]
    reps = {}
    for ds in sets:
        out = os.path.join(opt.repo_root(), 'runs', f'event_probe_{ds}.json')
        reps[ds] = run_dataset(ds, args.n_videos, args.n_random, args.device, out)
    if len(reps) == 2 and all(reps.values()):
        print(f'\n{"="*74}\nSUMMARY\n{"="*74}')
        print(f'  {"dataset":<8} {"ceiling":>8} {"class-mean":>11} '
              f'{"event-mean":>11} {"event-cls":>10} {"Psi-max":>8}')
        for ds, r in reps.items():
            print(f'  {ds:<8} {r["supervised_ceiling"]:>8.4f} '
                  f'{r["class_direction_mean"]:>11.4f} '
                  f'{r["event_per_class_best_mean"]:>11.4f} '
                  f'{r["per_class_diff_mean"]:>+10.4f} '
                  f'{r["event_psi_max"]:>8.4f}')
        print('  "event-cls" is the paired per-class difference, which is the')
        print('  quantity the decision should rest on.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
