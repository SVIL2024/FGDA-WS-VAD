"""CLIP text directions for Freq-Text Aug + the precondition experiment.

The plan document assumes ``batch['text_prompts']`` and calls ``clip.tokenize``
inside the training loop. Neither survives contact with the real code:

* VadCLIP's list files carry a *category string* (XD's is composite, e.g.
  ``B1-B2-0``), not text, so the direction must come from
  ``freq_text_prompts.parse_label`` -> a prompt table;
* encoding a handful of prompts on every step is pure waste: the direction is a
  constant per class, so :func:`build_class_directions` produces a
  ``(num_class, C)`` table once and training is an index lookup;
* the document builds the direction from the *raw* CLIP text encoder while
  VadCLIP's ``logits2`` is computed against the *prompt-tuned* text encoder
  (``CLIPVAD.encode_textprompt``). The precondition check therefore reports both,
  because agreement with the raw encoder is a necessary condition for the
  augmentation to interact with the classifier at all.

Direction construction follows the document: ``d = normalize(e_anom - e_normal)``.
"""

import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from freq_text_prompts import DATASET_CLASS_NAMES, DATASET_NUM_CLASS
from freq_text_prompts import get_class_prompts, get_normal_prompt, parse_label


# VadCLIP subsamples features, so one feature index covers this many GT
# frame labels. Mirrors the np.repeat(..., 16) in the test loop.
GT_FRAME_REPEAT = 16


def _f(x, what='value'):
    """``float`` with a named failure point.

    Results-critical module: a failed conversion must never become NaN
    silently. The original exception type is preserved; only the message gains
    context, so existing callers' except clauses keep working unchanged.
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


def label_to_index(label, dataset: str):
    """Primary class index for a raw CSV label (None if no recognised class)."""
    return parse_label(label, dataset)[0]


def label_is_anomaly(label, dataset: str) -> bool:
    return parse_label(label, dataset)[1]


# ---------------------------------------------------------------------------
# CLIP loading + direction table
# ---------------------------------------------------------------------------
def _clip_tokenize():
    import clip
    return clip.tokenize


@torch.no_grad()
def build_class_directions(clip_arch: str, dataset: str, device='cpu', verbose: bool = False):
    """``d_c = normalize(e_c - e_normal)`` for every class, computed once.

    Args:
        clip_arch: ``'ViT-B/16'`` or ``'ViT-B-16'`` - matches ``args.clip_arch`` /
            the architecture ``CLIPVAD`` loads internally.
        dataset: ``'xd'`` or ``'ucf'``.

    Returns:
        dirs: ``(num_class, C)`` unit directions; row 0 (Normal) is all zeros
            because ``e_normal - e_normal`` carries no direction.
        info: per-class diagnostics used by the precondition report.
    """
    import clip
    name = 'ViT-B/16' if 'B-16' in clip_arch or 'B/16' in clip_arch else clip_arch
    model, _ = clip.load(name, device=device, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenize = clip.tokenize

    def encode(prompts):
        # vanilla=True: this repo's CLIP fork overrides encode_text to take
        # (embedding, token_ids) for the prompt-tuned path. Raw prompts need the
        # upstream path, which performs token_embedding internally.
        tok = tokenize(list(prompts), truncate=True).to(device)
        e = model.encode_text(tok, vanilla=True).float()
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
        return e.mean(dim=0)

    normal_prompt = get_normal_prompt(dataset)
    e_normal = encode([normal_prompt] if isinstance(normal_prompt, str)
                      else list(normal_prompt))
    e_normal = e_normal / e_normal.norm()

    n_cls = DATASET_NUM_CLASS[dataset.lower()]
    C = e_normal.shape[0]
    dirs = torch.zeros(n_cls, C, device=device)
    info = {'normal_emb': e_normal.detach().cpu(), 'per_class': {}, 'clip_arch': name}

    for c in range(1, n_cls):
        prompts = get_class_prompts(c, dataset)
        e = encode(prompts)
        d = e - e_normal
        nd = d.norm()
        dirs[c] = d / (nd + 1e-8)
        info['per_class'][DATASET_CLASS_NAMES[dataset.lower()][c]] = {
            'idx': c,
            'n_prompts': len(prompts),
            'cos_e_anom_normal': _f((e * e_normal).sum()),
            'norm_d_raw': _f(nd),
        }
        if verbose:
            v = info['per_class'][DATASET_CLASS_NAMES[dataset.lower()][c]]
            print(f"  [{c:2d}] {DATASET_CLASS_NAMES[dataset.lower()][c]:14s} "
                  f"cos(e,norm)={v['cos_e_anom_normal']:+.4f}  |d_raw|={v['norm_d_raw']:.4f}")

    # how similar are the class directions to each other? if they are nearly
    # parallel the "class-specific text direction" carries no class information.
    off = []
    for i in range(1, n_cls):
        for j in range(i + 1, n_cls):
            off.append(_f((dirs[i] * dirs[j]).sum()))
    info['class_direction_mutual_cos'] = {'mean': _f(np.mean(off)),
                                          'min': _f(np.min(off)),
                                          'max': _f(np.max(off))}
    info['num_class'] = n_cls
    return dirs, info


def directions_for_batch(dirs: torch.Tensor, labels, dataset: str):
    """Resolve a batch of raw labels into augmentation directions.

    Returns
        mean_dir: ``(C,)`` mean over the batch's anomalous classes (the
            document's single global direction);
        per_video: ``(B, C)`` the per-video class direction, zero for normal
            videos - use this with ``aug_per_video``.
    """
    idxs = [label_to_index(lab, dataset) for lab in labels]
    per_video = torch.zeros(len(idxs), dirs.shape[1], device=dirs.device)
    for b, ix in enumerate(idxs):
        if ix is not None and ix > 0:
            per_video[b] = dirs[ix]
    anom = per_video.norm(dim=1) > 0
    if bool(anom.any()):
        m = per_video[anom].mean(dim=0)
        m = m / (m.norm() + 1e-8)
    else:
        m = torch.zeros_like(per_video[0])
    return m, per_video


# ---------------------------------------------------------------------------
# precondition experiment
# ---------------------------------------------------------------------------
def _read_list(list_csv: str):
    """``(path, label)`` rows from a VadCLIP list CSV."""
    rows = []
    try:
        with open(list_csv) as f:
            header = f.readline().strip().split(',')
            try:
                pi, li = header.index('path'), header.index('label')
            except ValueError:
                pi, li = 0, 1
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip('\n').split(',')
                if len(parts) <= max(pi, li):
                    continue
                rows.append((parts[pi], parts[li]))
    except OSError as exc:
        raise OSError(f'cannot read list file: {list_csv}') from exc
    return rows
def _gt_offsets(rows, test_list):
    """Start index of each list row's frames inside the flat GT array.

    VadCLIP's GT files are the *frame-level* labels of every test video
    concatenated in list order, at ``GT_FRAME_REPEAT`` times the feature rate
    (the feature extractor subsamples, so a feature index covers a block of GT
    entries). Reproducing the alignment here is what lets the probe measure the
    model's real task - ranking anomalous frames above normal frames *within* a
    video - instead of the far easier "which video is this" question.
    """
    counts = [np.load(p).shape[0] if os.path.exists(p) else 0 for p, _ in rows]
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


@torch.no_grad()
def probe_linear_separability(test_list: str, gt_path: str, dataset: str, dirs,
                             n_videos: int = 400, n_random: int = 200, seed: int = 0,
                             margin_threshold: float = 0.01):
    """Is the text direction a usable anomaly cue *within a video*?

    This is the decision-relevant premise test, and it took three attempts to
    get right - each earlier version would have licensed a wrong conclusion:

    * **Frame-level pooled AUC (wrong).** Pooling frames from anomaly and normal
      videos and scoring a direction separates *videos*, not anomalous frames
      from normal ones. Because clip identity (movie, scene, lighting) dominates
      CLIP features, a direction can score 0.78 pooled while carrying no
      frame-level information at all. Both text and random directions pass.
    * **Video-mean AUC (wrong the other way).** Averaging frames per video makes
      the between-video variance the signal, and *everything* collapses to ~0.51,
      including the supervised ceiling. A statistic that cannot rank videos
      cannot rank directions either.
    * **Within-video GT contrast (used here).** Score each frame, z-score the
      scores inside each video, and compute AUC against the ground-truth segment
      labels of that video, then average over videos. The z-score is this
      project's convention, NOT VadCLIP's: the reference test loops
      (``referCode/VadCLIP-main/src/xd_test.py:69``, ``ucf_test.py:91``) score
      one POOLED ``roc_auc_score(gt, np.repeat(scores, 16))`` over all test
      frames with no per-video normalisation, so the published AP/AUC are
      cross-video metrics. This statistic strips the between-video component on
      purpose, to isolate temporal localisation. Comparisons made under the
      same statistic are internally valid; they do not rank cues the same way
      the published metric does - ``runs/official_metric_*.json`` measures both
      conventions for the same cues.

    Two baselines are reported alongside, both essential:

    * a **random-direction null** of ``n_random`` unit vectors pushed through the
      identical pipeline - CLIP anisotropy means AUC > 0.5 is not evidence;
    * the **feature-magnitude cue** ``||f||``, which is free and scores ~0.69
      here. A text direction that cannot beat it has demonstrated nothing about
      language supervision.

    The text direction is compared against the null with the *same statistic on
    both sides* (mean of text AUCs vs mean of the null, and each text AUC's
    percentile in the null). Comparing max-of-6-classes against max-of-200
    random draws, as a naive gate would, is biased against the text purely
    through the number of draws.
    """
    rng = np.random.RandomState(seed)
    rows = _read_list(test_list)
    if not os.path.exists(gt_path):
        return {'ok': False, 'reason': f'gt file not found: {gt_path}'}
    gt = np.load(gt_path).astype(np.float32)
    offs = _gt_offsets(rows, test_list)

    videos, images = [], []
    for i, (p, _l) in enumerate(rows):
        n = _i(offs[i + 1] - offs[i])
        if n <= 0:
            continue
        seg = gt[offs[i] * GT_FRAME_REPEAT:offs[i + 1] * GT_FRAME_REPEAT]
        g = seg.reshape(-1, GT_FRAME_REPEAT).mean(1) > 0.5
        if not (0 < g.sum() < len(g)):     # needs both classes to define an AUC
            continue
        if not os.path.exists(p):
            continue
        videos.append(np.load(p).astype(np.float32))
        images.append(g)

    if len(videos) < 20:
        return {'ok': False, 'reason': f'only {len(videos)} test videos have mixed GT'}

    pick = rng.choice(len(videos), size=min(n_videos, len(videos)), replace=False)
    V = [videos[i] for i in pick]
    G = [images[i] for i in pick]
    mu = np.concatenate(V, 0).mean(0)
    mu_n = mu / (np.linalg.norm(mu) + 1e-9)

    def clean(u):
        u = u - (u @ mu_n) * mu_n
        return u / (np.linalg.norm(u) + 1e-9)

    def per_video(u):
        uu = clean(u)
        out = []
        for i in range(len(V)):
            v, g = V[i], G[i]
            s = (v - mu) @ uu
            s = (s - s.mean()) / (s.std() + 1e-8)
            a = _f(roc_auc_score(g, s))
            out.append(max(a, 1.0 - a))     # sign is free
        return np.array(out)

    def per_video_scalar(fn):
        """Same pipeline but for a per-frame scalar (not a projection)."""
        out = []
        for i in range(len(V)):
            v, g = V[i], G[i]
            s = fn(v)
            s = (s - s.mean()) / (s.std() + 1e-8)
            a = _f(roc_auc_score(g, s))
            out.append(max(a, 1.0 - a))
        return np.array(out)

    d = np.asarray(dirs.cpu().numpy() if hasattr(dirs, 'cpu') else dirs, dtype=np.float32)
    names = DATASET_CLASS_NAMES[dataset.lower()]
    per_class = {}
    for c in range(1, min(d.shape[0], len(names))):
        if np.linalg.norm(d[c]) < 1e-6:
            continue
        per_class[names[c]] = per_video(d[c])

    null = np.array([per_video(u).mean()
                     for u in rng.randn(n_random, d.shape[1]).astype(np.float32)])
    text_mean = np.array([a.mean() for a in per_class.values()])

    # supervised ceiling and the free magnitude baseline, same pipeline
    pos = np.concatenate([V[i][G[i]] for i in range(len(V))], 0).mean(0)
    neg = np.concatenate([V[i][~G[i]] for i in range(len(V))], 0).mean(0)
    ceiling = _f(max(per_video(pos - neg).mean(), 1 - per_video(pos - neg).mean()))
    # the magnitude cue is ||f|| itself - projecting onto the mean direction would
    # be a different (and here degenerate) statistic
    norm_per_video = per_video_scalar(lambda v: np.linalg.norm(v - mu, axis=1))
    norm_cue = _f(norm_per_video.mean())

    # The mean-over-classes readout, kept per video so it can be paired against the
    # magnitude cue on the same videos. This comparison is the paper's premise, and
    # the unpaired means are not enough to settle it: measured at n=60 the mean beat
    # the magnitude cue (0.6786 vs 0.6755) and at n=400 it lost (0.6759 vs 0.6905).
    # A 0.015 gap that flips with sample size needs an interval, not a point
    # comparison. Both sides are sign-corrected per video (max(a, 1-a)), so the
    # pairing compares "which readout separates the frames better on this video",
    # with the same optimistic bias on both sides.
    text_per_video = np.array([per_video(d[c]) for c in range(1, min(d.shape[0], len(names)))
                               if np.linalg.norm(d[c]) >= 1e-6]).mean(0)
    _diff = text_per_video - norm_per_video
    _n = len(_diff)
    _se = _diff.std(ddof=1) / np.sqrt(_n) if _n > 1 else float('nan')
    mean_margin = _f(_diff.mean())
    mean_margin_lo = _f(mean_margin - 1.96 * _se) if _n > 1 else float('nan')
    mean_margin_hi = _f(mean_margin + 1.96 * _se) if _n > 1 else float('nan')

    p95 = _f(np.percentile(null, 95))
    # mean-over-classes vs the null distribution of the same mean: draw as many
    # random directions per replicate as there are text classes, so the two
    # sides of the comparison share the statistic *and* the sample size
    k = len(text_mean)
    reps = np.array([rng.choice(null, size=k, replace=False).mean() for _ in range(2000)])
    p_value = _f((reps >= text_mean.mean()).mean())

    out = {
        'ok': True,
        'n_test_videos_used': len(V),
        'frames_per_video_mean': round(_f(np.mean([len(v) for v in V])), 1),
        'per_class_within_video_auc': {kk: round(_f(v.mean()), 4)
                                       for kk, v in
                                       sorted(per_class.items(), key=lambda kv: -kv[1].mean())},
        'text_auc_mean': round(_f(text_mean.mean()), 4),
        'text_auc_max': round(_f(text_mean.max()), 4),
        'text_auc_best_class': max(per_class, key=lambda kk: per_class[kk].mean()),
        'random_auc_mean': round(_f(null.mean()), 4),
        'random_auc_p95': round(p95, 4),
        'random_auc_max': round(_f(null.max()), 4),
        'n_random': n_random,
        'magnitude_cue_auc': round(norm_cue, 4),
        'supervised_ceiling_auc': round(ceiling, 4),
        'margin_over_random_mean': round(_f(text_mean.mean() - null.mean()), 4),
        'margin_threshold': margin_threshold,
        'p_value_vs_random_mean': p_value,
        'beats_random_mean': bool(text_mean.mean() > p95 or
                                  (text_mean.mean() - null.mean()) > margin_threshold),
        'beats_random_max': bool(text_mean.max() > null.max()),
        # Two different readouts of the same bank, kept separate on purpose.
        # ``text_auc_max`` is the class_max readout (max over the bank). That is a
        # deployable training-free baseline and it is what the magnitude comparison
        # has always used. ``text_auc_mean`` is the expected quality of a *per-video*
        # class direction, which is what E1 actually injects - ``augment_and_score``
        # takes ``dirs[class_idx]``, one direction per video, so a video's expected
        # injected signal is the mean over the bank, not its best entry. The two
        # readouts can disagree, and when they do the disagreement is the finding.
        'beats_magnitude_cue': bool(text_mean.max() > norm_cue),
        'beats_magnitude_cue_mean': bool(text_mean.mean() > norm_cue),
        # paired, per video, mean-over-classes readout minus magnitude cue. The
        # interval is the honest version of 'does the text beat the magnitude';
        # the point comparison above is not stable to the sample size.
        'mean_basis_margin': mean_margin,
        'mean_basis_ci95': [mean_margin_lo, mean_margin_hi],
        'mean_basis_beats_magnitude': bool(mean_margin_lo > 0),
        'headroom_text_to_ceiling': round(_f(ceiling - text_mean.max()), 4),
        'headroom_text_to_ceiling_mean': round(_f(ceiling - text_mean.mean()), 4),
    }
    out['carries_semantic_signal'] = bool(out['beats_random_mean'] and out['beats_magnitude_cue'])
    out['proceed'] = bool(out['carries_semantic_signal'])
    # The same question recomputed on the mean-over-classes basis: "is the direction
    # E1 injects a usable cue" rather than "is the best direction in the bank a usable
    # cue". Recorded next to ``proceed`` instead of replacing it, because adopting it
    # as the gate flips XD to STOP and aborts the sweep - a research decision about
    # which readout the ablation is about, not a bug fix.
    out['proceed_on_mean_basis'] = bool(out['beats_random_mean'] and
                                        out['beats_magnitude_cue_mean'])

    if out['proceed']:
        # The magnitude comparison is decided on class_max, so the claim has to be
        # stated on class_max too. Quoting text_auc_mean next to a magnitude number
        # it loses to is what made the XD verdict self-contradictory on disk
        # (0.6759 printed against "beats ... 0.6905").
        mag_verb = 'beats' if out['beats_magnitude_cue'] else 'does not beat'
        mean_note = (
            'the mean class direction also beats it'
            if out['beats_magnitude_cue_mean'] else
            f'the mean class direction does NOT on the point comparison '
            f'({out["text_auc_mean"]:.4f} < {norm_cue:.4f}) - and E1 injects the '
            f'video\'s own class direction, so the signal the augmenter actually '
            f'uses is the weaker of the two')
        ci = out['mean_basis_ci95']
        above = 'demonstrably above' if out['mean_basis_beats_magnitude'] else 'NOT demonstrably above'
        out['verdict'] = (
            f"PROCEED ON THE CLASS-MAX READOUT, WEAK BUT REAL: over the {k} class "
            f"directions the best ({out['text_auc_best_class']}) reaches within-video "
            f"AUC {out['text_auc_max']:.4f} and {mag_verb} the free magnitude cue "
            f"({norm_cue:.4f}); the mean over classes is {out['text_auc_mean']:.4f} vs "
            f"a random-direction mean of {out['random_auc_mean']:.4f} "
            f"(p95 {p95:.4f}, p={p_value:.3f}). CAVEAT: {mean_note}. Paired per video, "
            f"the mean-over-classes readout minus the magnitude cue is "
            f"{mean_margin:+.4f} (95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] over "
            f"{_n} videos), so on the basis E1 actually uses the text direction is "
            f"{above} a cue that costs nothing. Expect E1 vs E3 to "
            f"differ by roughly {out['margin_over_random_mean']:.3f} AUC at best, "
            f"not more: the supervised ceiling is {ceiling:.4f}, so the text direction "
            f"captures only a small share of the available signal. Plan the ablation "
            f"with enough seeds to resolve a difference this small, and report the E3 "
            f"random-direction control prominently - it is the whole argument.")
    else:
        why = []
        if not out['beats_random_mean']:
            why.append(f"it does not clear the random-direction p95 "
                       f"({out['text_auc_mean']:.4f} vs {p95:.4f}, p={p_value:.3f})")
        if not out['beats_magnitude_cue']:
            # decided on class_max, so report class_max here too
            why.append(f"no class direction beats the free feature-magnitude cue "
                       f"(best {out['text_auc_max']:.4f}, mean "
                       f"{out['text_auc_mean']:.4f} vs {norm_cue:.4f})")
        out['verdict'] = (
            'STOP: the text direction is not a demonstrated anomaly cue - ' +
            '; '.join(why) +
            f". The supervised ceiling is {ceiling:.4f}, so signal exists in the "
            f"features; it is not reachable through the CLIP text direction. Change "
            f"the direction source (learned, or data-derived) before spending GPU "
            f"time on E1-E5.")
    return out


@torch.no_grad()
def precondition_report(clip_arch: str, dataset: str, test_list: str, gt_path: str,
                        n_videos: int = 400, device='cpu', seed: int = 0,
                        threshold: float = 0.2, write_json=None,
                        separability: bool = True, n_random: int = 200,
                        margin_threshold: float = 0.01) -> dict:
    """Gate the E1-E5 ablation on whether the text direction carries any signal.

    Read this as a go/no-go, not as a result. The decisive probe is
    :func:`probe_linear_separability`, which scores the direction as a temporal
    localisation problem: ranking anomalous frames above normal frames *within* a
    test video against the ground-truth segments, z-scored per video. That
    z-scoring is this project's convention - VadCLIP's reported AP/AUC pool every
    test frame with no per-video normalisation - so this gate measures
    localisation, the property the E1 augmentation is supposed to change, not
    the published number itself.

    Reported diagnostics that are *not* used for the decision:

    ``cos_text_direction_vs_data_direction`` - the plan document's own "前置实验".
    It compares the text direction against a hand-built centroid difference on the
    *training* clips with no null distribution, and it reads near zero here even
    though the direction does carry signal; a statistic that cannot separate a
    useful direction from a useless one is not a gate.
    ``text_embedding_structure.cos_e_anom_normal_mean``
        CLIP text embeddings are mutually close. Values ~0.98+ mean the class
        contrast is a small difference between near-parallel vectors, so
        ``e_anom - e_normal`` is largely prompt noise.
    ``class_direction_mutual_cos``
        if the per-class directions are nearly parallel to *each other*, the
        "class-specific" qualifier in the title is not doing any work.
    """
    dirs, info = build_class_directions(clip_arch, dataset, device=device)

    report = {'dataset': dataset, 'clip_arch': info['clip_arch'], 'threshold': threshold,
              'num_class': info['num_class'], 'per_class': info['per_class'],
              'class_direction_mutual_cos': info['class_direction_mutual_cos'],
              'test_list': test_list, 'gt_path': gt_path}

    sep = None
    if separability:
        sep = probe_linear_separability(test_list, gt_path, dataset, dirs,
                                        n_videos=n_videos, seed=seed,
                                        n_random=n_random,
                                        margin_threshold=margin_threshold)
        report['linear_separability'] = sep

    class_cos = [v['cos_e_anom_normal'] for v in info['per_class'].values()]
    d_raw = [v['norm_d_raw'] for v in info['per_class'].values()]
    report['text_embedding_structure'] = {
        'cos_e_anom_normal_mean': _f(np.mean(class_cos)),
        'cos_e_anom_normal_max': _f(np.max(class_cos)),
        'norm_d_raw_mean': _f(np.mean(d_raw)),
    }

    if sep is not None and sep.get('ok'):
        report['proceed'] = bool(sep['proceed'])
        report['verdict'] = sep['verdict']
        report['gate_meaning'] = (
            'proceed=True means the text direction ranks anomalous frames above '
            'normal frames within a test video better than the 95th percentile of '
            'random unit directions AND better than the free feature-magnitude cue. '
            'The magnitude comparison is made on the class_max readout (the best '
            'direction in the bank), which is the deployable training-free baseline; '
            'proceed_on_mean_basis reports the same question on the mean-over-classes '
            'basis, which is what E1 actually injects since it takes the video\'s own '
            'class direction. Where the two disagree, the paper\'s premise is weaker '
            'than proceed alone suggests. Only then can a gain over the E3 '
            'random-direction control be attributed to the text rather than to CLIP '
            'anisotropy or to feature magnitude.'
        )
    else:
        report['proceed'] = None
        report['verdict'] = ('INCONCLUSIVE: ' +
                             str((sep or {}).get('reason', 'probe not run')))

    if write_json:
        import json
        try:
            os.makedirs(os.path.dirname(os.path.abspath(write_json)), exist_ok=True)
        except OSError as exc:
            raise OSError(f'cannot create output dir for {write_json}') from exc
        try:
            with open(write_json, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise OSError(f'cannot write report to {write_json}') from exc
    return report
