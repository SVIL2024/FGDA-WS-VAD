"""Shared Freq-Text Aug trainer for XD-Violence (AP) and UCF-Crime (AUC).

Mirrors the VadCLIP training loop so a run is directly comparable to the
published VadCLIP numbers: same model, same ``CLAS2`` + ``CLASM`` + prompt
dispersion losses, same AdamW/MultiStepLR schedule, same top-k MIL pooling, same
checkpointing rule. The augmentation enters as exactly one extra term

    loss = loss1 + loss2 + loss3 + aug_weight * loss_aug

so any difference against E5 is attributable to the augmentation branch alone.

The five ablations are *configuration*, not code paths - see ``VARIANTS`` in
``freq_text_options.py``.
"""

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from freq_text_aug import augment_and_score
from freq_text_dataset import ClipVideoDataset, collate
from freq_text_text import build_class_directions
from utils.lr_warmup import WarmupMultiStepLR

# Result-affecting knobs that go into the self-describing checkpoint filename, in
# this order. The same list drives the [Config] log header so the two cannot drift.
FILENAME_KEYS = ('variant', 'seed', 'lr', 'batch_size', 'visual_layers', 'attn_window',
                 'warmup_pct', 'scheduler_milestones', 'scheduler_rate',
                 'early_stop_patience')


def config_keyvals(args) -> str:
    """The ordered ``key=value`` string shared by the log header and the filename."""
    parts = []
    for k in FILENAME_KEYS:
        v = getattr(args, k, None)
        if v is None:
            continue
        parts.append(f'{k}={v}')
    return '_'.join(parts)


def log_header(args, device, logger, tag, warmup_note, threshold):
    """The three header lines, emitted before any training output.

    Order is fixed: the exact run command with defaults resolved, how the launch
    was authorised, then the full resolved config in the same ``key=value`` order
    the checkpoint filename uses. A log that omits these cannot be replayed, and
    the shared ``config_keyvals`` helper is what keeps the config and the filename
    from drifting apart.
    """
    cmd = (f'python {args.dataset}_train.py --variant {args.variant} --seed {args.seed} '
           f'--run-name {tag} --lr {args.lr} --batch-size {args.batch_size} '
           f'--max-epoch {args.max_epoch} --visual-layers {args.visual_layers} '
           f'--attn-window {args.attn_window} --warmup-pct {args.warmup_pct} '
           f'--early-stop-patience {args.early_stop_patience} '
           f'--save-threshold {args.save_threshold} --best-metric {args.best_metric}')
    logger(f'[RunCommand] {cmd}')
    logger('[Launch] baseline run: training recipe is the upstream VadCLIP recipe, '
           'unmodified. Any deviation (warmup, dual-head selection, early stopping) '
           'must be an explicit flag, never a default.')
    logger(f'[Threshold] dataset={args.dataset.upper()} selection='
           f'{args.best_metric} threshold={threshold:.4f}')
    logger(f'[Config] {config_keyvals(args)} | device={device} | aug_weight='
           f'{args.aug_weight} | aug_mode={args.aug_mode} | aug_direction='
           f'{args.aug_direction} | {warmup_note}')


def select_head(dataset: str, best_metric: str, extra: dict):
    """(selection metric, AUC paired with it, which head won).

    The model has two heads - ``logits1`` (CLAS2, frame level) and ``logits2``
    (CLASM, class level) - and each reports its own AP and AUC. The AUC quoted in a
    checkpoint filename must come from the head that produced the winning AP;
    pairing AP2 with AUC1 would make the filename describe a model that does not
    exist.

    ``test_fn`` is the default because it is what upstream actually selects on. In
    nwpu-zxr/VadCLIP ``xd_test.test`` ends in ``return ROC1, AP2, 0``, and the training
    loop then does ``if AP > ap_best`` - so on XD the selected epoch is the best
    **AP2**, and on UCF it is the best **AUC1**. ``AP``/``AUC`` remain available as
    explicit opt-in deviations: on XD they pick the larger of AP1/AP2, which reports a
    different epoch and breaks comparability with the published numbers.
    """
    ap1, ap2 = extra.get('AP1'), extra.get('AP2')
    auc1, auc2 = extra.get('AUC1'), extra.get('AUC2')
    if best_metric == 'test_fn':
        if dataset == 'xd':
            return (None, None, 'head?') if ap2 is None else (ap2, auc2, 'head2')
        return (None, None, 'head?') if auc1 is None else (auc1, ap1, 'head1')
    if best_metric == 'AP':
        if ap1 is None or ap2 is None:
            return None, None, 'head?'
        if ap1 >= ap2:
            return ap1, auc1, 'head1'
        return ap2, auc2, 'head2'
    if auc1 is None or auc2 is None:
        return None, None, 'head?'
    if auc1 >= auc2:
        return auc1, ap1, 'head1'
    return auc2, ap2, 'head2'


def early_stop_state(patience, patience_left, improved):
    """Advance the early-stop counter and say whether the run should end.

    Returns ``(patience_left, should_stop)``.

    ``patience <= 0`` means early stopping is **off**, which is the upstream default
    and the only setting under which a run is comparable with the published numbers.
    It must not be read as "the counter is already exhausted": the counter starts at
    0 and is reset to 0 on every improvement, so a bare ``patience_left <= 0`` test
    fires at epoch 1 and silently trains a single epoch while the log header still
    advertises the full schedule. This is a function rather than three inline lines
    because that bug shipped once and no test could reach it inside the loop.
    """
    if patience <= 0:
        return patience_left, False
    if improved:
        return int(patience), False
    left = patience_left - 1
    return left, left <= 0


# ---------------------------------------------------------------------------
# MIL losses (identical maths to VadCLIP's ucf_train/xd_train, vectorised where
# the loop was pure overhead)
# ---------------------------------------------------------------------------
def _topk_mean_rows(scores, lengths, divisor=16):
    """Row-wise mean of the top ``L/divisor + 1`` entries of ``scores``."""
    B, T = scores.shape[0], scores.shape[1]
    out = []
    for i in range(B):
        L = int(lengths[i])
        k = max(1, int(L / divisor + 1))
        tmp = torch.topk(scores[i, :L], k=min(k, L), largest=True).values
        out.append(tmp.mean())
    return torch.stack(out)


def CLAS2(logits, labels, lengths, device):
    """Binary MIL loss on ``logits1`` - VadCLIP's CLAS2.

    logits: ``(B, T, 1)``; labels: ``(B, num_class)`` one-hot/multi-hot with
    column 0 == normal.
    """
    labels = (1.0 - labels[:, 0]).reshape(-1).to(device)
    prob = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    inst = _topk_mean_rows(prob, lengths.to(device))
    return F.binary_cross_entropy(inst, labels.float())


def CLASM(logits, labels, lengths, device):
    """Multi-class MIL loss on ``logits2`` - VadCLIP's CLASM.

    logits: ``(B, T, num_class)``; labels: ``(B, num_class)``. Top-k is taken over
    the *time* axis (``dim=0`` of the per-video slice), giving ``(k, num_class)``
    which averages to one logit row per video, exactly as upstream does.
    """
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    instance_logits = torch.zeros(0, logits.shape[-1], device=device)
    for i in range(logits.shape[0]):
        k = max(1, int(lengths[i] / 16 + 1))
        tmp = torch.topk(logits[i, 0:int(lengths[i])], k=min(k, int(lengths[i])),
                         largest=True, dim=0).values          # (k, num_class)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


def prompt_dispersion(text_features, num_class):
    """VadCLIP's loss3: push the normal prompt away from the anomaly prompts."""
    dev = text_features.device
    loss = torch.zeros(1, device=dev)
    n = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    for j in range(1, text_features.shape[0]):
        a = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
        loss = loss + torch.abs(n @ a)
    return loss / max(1, num_class - 1)


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_loaders(args, dataset: str, logger):
    """Normal/anomaly twin loaders (UCF) or one unfiltered loader (XD)."""
    common = dict(test_mode=False, dataset=dataset)
    if dataset == 'ucf':
        normal_ds = ClipVideoDataset(args.visual_length, args.train_list, normal=True, **common)
        anom_ds = ClipVideoDataset(args.visual_length, args.train_list, normal=False, **common)
        normal_loader = DataLoader(normal_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, drop_last=True, collate_fn=collate)
        anom_loader = DataLoader(anom_ds, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.workers, drop_last=True, collate_fn=collate)
        logger(f'train loaders: normal={len(normal_ds)} anomaly={len(anom_ds)}')
        return normal_loader, anom_loader
    train_ds = ClipVideoDataset(args.visual_length, args.train_list, normal=None, **common)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate)
    logger(f'train loader: {len(train_ds)} clips')
    return train_loader, None


def build_test_loader(args, dataset: str, logger):
    """Test loader on the same collate, so the eval loop reads ``item[0]``/``item[2]``."""
    test_ds = ClipVideoDataset(args.visual_length, args.test_list, test_mode=True,
                               dataset=dataset)
    logger(f'test loader: {len(test_ds)} clips')
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate)
    limit = getattr(args, 'max_test_clips', 0)
    if limit:
        logger(f'test loader capped at {limit} clips -> metric is MEANINGLESS, '
               f'only the plumbing is validated')
        loader = CappedLoader(loader, limit)
    return loader


class CappedLoader:
    """Yield the first ``n`` batches of a loader (smoke runs)."""

    def __init__(self, loader, n):
        self.loader, self.n = loader, int(n)

    def __iter__(self):
        from itertools import islice
        return islice(iter(self.loader), self.n)

    def __len__(self):
        return min(self.n, len(self.loader))


def merge_batch(items):
    """Concatenate the normal and anomaly halves of a UCF-style batch.

    Each item is ``(feat, label_tuple, lengths, class_idx, binary)`` as produced by
    :func:`freq_text_dataset.collate`. Labels stay a flat tuple of raw strings
    because ``utils.tools.get_batch_label`` expects strings, not indices.
    """
    feats = torch.cat([it[0] for it in items], dim=0)
    labels = tuple(s for it in items for s in it[1])
    lengths = torch.cat([it[2] for it in items], dim=0)
    class_idx = torch.cat([it[3] for it in items], dim=0)
    binary = torch.cat([it[4] for it in items], dim=0)
    return feats, labels, lengths, class_idx, binary


def run_train(model, train_loader, anomaly_loader, test_fn, test_loader, args,
              label_map, device, logger, tag: str = 'run'):
    """One training run of one ablation variant. Returns the best metric."""
    from utils.tools import get_batch_label, get_prompt_text

    model.to(device)
    prompt_text = get_prompt_text(label_map)
    num_class = len(prompt_text)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    params = list(model.parameters())
    augmenter = None
    dirs = None
    if args.aug_weight > 0.0:
        from freq_text_aug import FreqTextAugmenter
        augmenter = FreqTextAugmenter(
            dim=args.embed_dim, block_len=args.aug_block_len, mode=args.aug_mode,
            use_freq_gate=args.aug_freq_gate, alpha_range=(args.aug_alpha_lo, args.aug_alpha_hi),
            band_gain=args.aug_band_gain, gate_bands=args.aug_gate_bands,
            exclude_dc=args.aug_exclude_dc, delta_norm=args.aug_delta_norm,
            direction_source=args.aug_direction).to(device)
        params = params + list(augmenter.parameters())
        dirs, dinfo = build_class_directions(args.clip_arch, args.dataset, device=device)
        logger(f'direction table {tuple(dirs.shape)} from {args.clip_arch}; '
               f'class-direction mutual cos mean='
               f'{dinfo["class_direction_mutual_cos"]["mean"]:.4f}')
        logger(f'augmenter mode={args.aug_mode} gate={args.aug_freq_gate} '
               f'direction={args.aug_direction} delta_norm={args.aug_delta_norm} '
               f'exclude_dc={args.aug_exclude_dc}')
        if augmenter is not None:
            logger(f'envelope stats at init: {augmenter.envelope_stats()}')

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    best = 0.0
    history = []
    os.makedirs(args.log_dir, exist_ok=True)
    metric_name = args.best_metric
    if metric_name == 'test_fn':
        # The history key has to be the metric that was actually optimised, not the
        # name of the selection mode, or the logs stop lining up with published tables.
        metric_name = 'AP' if args.dataset == 'xd' else 'AUC'

    # Steps per epoch are needed before the scheduler can be built: the warmup
    # scheduler counts global iterations, and VadCLIP's milestones are given in
    # epochs, so they have to be converted.
    twin = anomaly_loader is not None
    steps_per_epoch = min(len(train_loader), len(anomaly_loader)) if twin else len(train_loader)
    if getattr(args, 'max_train_steps', 0):
        steps_per_epoch = min(steps_per_epoch, args.max_train_steps)
    max_iter = max(1, steps_per_epoch * args.max_epoch)
    if getattr(args, 'warmup_pct', 0.0) > 0.0:
        scheduler = WarmupMultiStepLR(
            optimizer, max_iter,
            [m * steps_per_epoch for m in args.scheduler_milestones],
            gamma=args.scheduler_rate, pct_start=args.warmup_pct)
        step_per_iter = True
        warmup_note = (f'warmup {args.warmup_pct:.0%} of {max_iter} iters '
                       f'({int(args.warmup_pct * max_iter)}), milestones in iters, '
                       f'stepped per iteration')
    else:
        scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
        step_per_iter = False
        warmup_note = 'no warmup, milestones in epochs, stepped per epoch'

    threshold = float(args.save_threshold)
    best_path = None
    best_epoch = 0
    stopped_epoch = 0
    patience_left = int(args.early_stop_patience)
    log_header(args, device, logger, tag, warmup_note, threshold)

    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        ck = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        best = ck.get('metric', 0.0)
        logger(f'resumed from {args.checkpoint_path} (best {metric_name}={best:.4f})')

    twin = anomaly_loader is not None
    gate_grad = None
    for e in range(args.max_epoch):
        model.train()
        acc = {'l1': 0.0, 'l2': 0.0, 'l3': 0.0, 'la': 0.0, 'n_aug': 0.0, 'alpha': 0.0}
        n_batches = min(len(train_loader), len(anomaly_loader)) if twin else len(train_loader)
        if getattr(args, 'max_train_steps', 0):
            n_batches = min(n_batches, args.max_train_steps)
        normal_iter = iter(train_loader)
        anom_iter = iter(anomaly_loader) if twin else None
        t0 = time.time()

        for i in range(n_batches):
            items = [next(normal_iter)] + ([next(anom_iter)] if twin else [])
            visual_feat, text_labels, feat_lengths, class_idx, _binary = merge_batch(items)
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual_feat, None, prompt_text, feat_lengths)
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            # reuse the text features the forward pass already produced rather than
            # re-running the CLIP text tower for the dispersion term
            loss3 = prompt_dispersion(text_features, num_class) * args.loss3_scale

            loss_aug, astats = augment_and_score(
                model, augmenter, dirs, visual_feat, feat_lengths, class_idx, args,
                prompt_text=prompt_text, logits_ref=logits1.squeeze(-1).detach())

            loss = loss1 + loss2 + loss3 + loss_aug
            optimizer.zero_grad()
            loss.backward()
            # Capture the gate gradient *before* clipping, and only on the first
            # step: E1 is a learned-gate experiment, so "the gate received no
            # gradient" is a silent failure that would turn E1 into E4 while every
            # loss still looks healthy.
            if i == 0 and augmenter is not None and \
                    isinstance(augmenter.freq_gate, torch.nn.Parameter):
                g = augmenter.freq_gate.grad
                gate_grad = float(g.norm()) if g is not None else 0.0
                if gate_grad == 0.0:
                    logger(f'[{tag}] WARNING: freq_gate received zero gradient on '
                           f'step 1 - the augmentation is detached from the gate, '
                           f'so E1 is running as E4')
            clip_grad_value_(params, 10)
            optimizer.step()
            # The warmup scheduler counts global iterations, so it must be stepped
            # here rather than once per epoch; the plain MultiStepLR path keeps its
            # epoch-level milestones and is stepped below instead.
            if step_per_iter:
                scheduler.step()

            acc['l1'] += loss1.item(); acc['l2'] += loss2.item()
            acc['l3'] += loss3.item(); acc['la'] += float(loss_aug.item())
            acc['n_aug'] += astats['n_aug']; acc['alpha'] += astats['alpha']

            if args.log_every and (i + 1) % args.log_every == 0:
                k = i + 1
                logger(f'[{tag}] ep {e+1}/{args.max_epoch} step {k}/{n_batches} '
                       f'loss1 {acc["l1"]/k:.4f} loss2 {acc["l2"]/k:.4f} '
                       f'loss3 {acc["l3"]/k:.5f} loss_aug {acc["la"]/k:.4f} '
                       f'aug_frames/step {acc["n_aug"]/k:.1f} alpha {acc["alpha"]/k:.3f}')
        if not step_per_iter:
            scheduler.step()

        mean_aug = acc['n_aug'] / max(1, n_batches)
        logger(f'[{tag}] epoch {e+1} done in {time.time()-t0:.0f}s | '
               f'loss1 {acc["l1"]/max(1,n_batches):.4f} loss2 {acc["l2"]/max(1,n_batches):.4f} '
               f'loss_aug {acc["la"]/max(1,n_batches):.4f} | mean aug frames/step {mean_aug:.1f}' +
               (f' | first-step gate grad {gate_grad:.3e}' if gate_grad is not None else ''))

        _raw_metric, extra = test_fn(model, test_loader, args.visual_length, prompt_text,
                                    gt, gtsegments, gtlabels, device)
        # Select on the configured metric, taking the AUC from the SAME head that
        # won. test_fn's own primary metric is used only as the fallback when a head
        # reports nothing, so a missing head can never silently become a 0.0.
        metric, auc_for_name, head = select_head(args.dataset, metric_name, extra)
        if metric is None:
            metric, head = float(_raw_metric), 'fallback'
            auc_for_name = extra.get('AUC1', float('nan'))
        improved = (metric - best) > float(args.early_stop_min_delta)
        best_so_far = best if not improved else metric
        best_ep_so_far = (e + 1) if improved else best_epoch
        stopped_epoch = e + 1
        history.append({'epoch': e + 1, metric_name: float(metric),
                        'head': head, 'paired_auc': None if auc_for_name is None
                        else float(auc_for_name),
                        'improved': bool(improved),
                        'lr': float(optimizer.param_groups[0]['lr']),
                        'loss1': acc['l1'] / max(1, n_batches),
                        'loss_aug': acc['la'] / max(1, n_batches),
                        'aug_frames_per_step': mean_aug, **extra})
        logger(f'[{tag}] ep {e+1}/{args.max_epoch} | {time.time()-t0:.0f}s | '
               f'loss1 {acc["l1"]/max(1,n_batches):.4f} | '
               f'{metric_name}({head}) {metric:.4f} | '
               f'BEST {metric_name} {best_so_far:.4f}(e{best_ep_so_far}) | '
               f'lr {optimizer.param_groups[0]["lr"]:.2e} | '
               f'threshold {threshold:.4f}')

        if improved:
            best, best_epoch = metric, e + 1
            if metric > threshold:
                # Self-describing name: the whole experiment is recoverable from the
                # filename, and the AUC is the winning head's own AUC.
                fname = (f'{args.dataset.upper()}_auc{float(auc_for_name):.4f}_'
                         f'ap{float(metric):.4f}_e{e+1}_{config_keyvals(args)}.pth')
                new_path = os.path.join(args.log_dir, fname)
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)      # never keep more than one best per run
                state_dict = model.state_dict()
                torch.save(state_dict, new_path)
                # Stable alias for the documented workflow: freq_text_test's
                # default model_path and the README examples resolve to
                # runs/<run>/model_best.pth, not to the self-describing name.
                torch.save(state_dict, args.model_path)
                best_path = new_path
                with open(os.path.join(args.log_dir, 'latest_best.txt'), 'w') as f:
                    f.write(fname + '\n')
                logger(f'[{tag}] new best {metric_name}={metric:.4f} (above '
                       f'threshold {threshold:.4f}) -> {fname}')
            else:
                logger(f'[{tag}] {metric_name}={metric:.4f} is a new best but below '
                       f'threshold {threshold:.4f} - not saved')

        # The resume checkpoint is ~700 MB because it carries the optimizer state, and
        # writing it every epoch is how a 10-epoch run fills a disk for no benefit: it
        # is only ever read by an explicit `--use-checkpoint` resume, which the user
        # asked for by name. Defaulting it to off is the difference between 7 GB and 0.
        if args.save_resume_checkpoint or args.use_checkpoint:
            os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)
            torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metric': best, 'aug_state': None if augmenter is None
                        else augmenter.state_dict()}, args.checkpoint_path)
        else:
            logger(f'[{tag}] resume checkpoint not written (pass '
                   f'--save-resume-checkpoint to keep one; it is ~700 MB)')

        with open(os.path.join(args.log_dir, f'{tag}_history.json'), 'w') as f:
            json.dump({'tag': tag, 'args': vars(args), 'best': best,
                       'best_epoch': best_epoch, 'best_model_path': best_path,
                       'save_threshold': threshold, 'metric_name': metric_name,
                       'history': history}, f, indent=2)

        patience_left, stop_now = early_stop_state(
            args.early_stop_patience, patience_left, improved)
        if stop_now:
            logger(f'[{tag}] [EarlyStop] no {metric_name} improvement for '
                   f'{args.early_stop_patience} evals (min_delta '
                   f'{args.early_stop_min_delta}) - stopping at epoch {e+1}')
            break

    if best_path and os.path.exists(best_path):
        # Restore the best qualifying weights so the in-memory model matches the
        # file that will be reported. The threshold gate and this reload are in
        # tension by design: the skill wants a run never to end model-less, while
        # the disk rule forbids keeping weights that missed the target. When the
        # threshold was never cleared there is nothing to restore, and that is
        # stated explicitly rather than papered over.
        model.load_state_dict(torch.load(best_path, map_location=device))
        logger(f'[{tag}] best model restored: {best_path}')
    else:
        logger(f'[{tag}] No checkpoint saved - metric never exceeded threshold '
               f'{threshold:.4f} (best seen {best:.4f} at e{best_epoch}); '
               f'no weights on disk by design')

    summary_dir = os.path.dirname(os.path.abspath(args.log_dir))
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, 'results_summary.txt'), 'a') as f:
        f.write(f'[Total] {tag} metric={metric_name} best={best:.4f} '
                f'epoch={best_epoch} threshold={threshold:.4f} '
                f'saved={"yes" if best_path else "no"} '
                f'stopped_epoch={stopped_epoch} of {args.max_epoch} '
                f'config={config_keyvals(args)}\n')

    logger(f'[{tag}] best {metric_name} = {best:.4f}')
    return best, history


# ---------------------------------------------------------------------------
# smoke test: one step on synthetic data, no dataset required
# ---------------------------------------------------------------------------
def smoke_train_step(model, augmenter, dirs, device, batch=4, T=256, C=512, n_class=7):
    """One forward/backward through the *real* augmentation path.

    Catches shape/device/gradient plumbing errors before a multi-hour run.
    Returns a dict of the observed quantities.
    """
    from freq_text_aug import augment_and_score

    class _Cfg:
        aug_weight = 1.0
        aug_ratio = 0.2
        aug_target = 0.9
        aug_alpha = 0.3

    visual = torch.randn(batch, T, C, device=device)
    lengths = torch.full((batch,), T, device=device, dtype=torch.long)
    lengths[-1] = 70
    class_idx = torch.tensor([0, 1, 2, 3], device=device)[:batch]
    prompt = ['normal'] * n_class
    _, logits1, _ = model(visual, None, prompt, lengths)
    loss, stats = augment_and_score(model, augmenter, dirs, visual, lengths, class_idx,
                                    _Cfg(), prompt_text=prompt,
                                    logits_ref=logits1.squeeze(-1).detach())
    loss.backward()
    gnorm = None
    if isinstance(augmenter.freq_gate, torch.nn.Parameter):
        gnorm = float(augmenter.freq_gate.grad.norm())
    return {'loss_aug': float(loss.item()), 'stats': stats, 'gate_grad': gnorm,
            'logits1_shape': tuple(logits1.shape)}
