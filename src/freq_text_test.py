"""Evaluation for Freq-Text Aug, numerically identical to VadCLIP's test loops.

Copied from ``xd_test.py`` / ``ucf_test.py`` with three mechanical changes:

* the batch unpacks through ``freq_text_dataset.collate`` (``item[0]`` features,
  ``item[2]`` length) instead of the upstream 3-tuple;
* ``model.eval()`` and ``torch.no_grad()`` stay, but the function *returns*
  ``(metric, extra)`` instead of only printing, so the trainer can log JSON;
* frame scores are also returned as arrays so an offline paired comparison
  between two variants can be run without retraining.

XD is scored by frame-level AP (the metric VadCLIP reports for XD); UCF by frame
level AUC. Detection mAP is computed alongside for both because an augmentation
that synthesises *frames* should show up in localisation first.

Two conventions are reported, because they measure different things and rank
methods differently (see .memory/pooled-vs-within-video-metric and
src/analysis_official_metric.py):

* ``AUC1/AP1/AUC2/AP2`` are the PUBLISHED pooled metrics - one metric call over
  all frames of all videos, exactly VadCLIP's protocol (``xd_test.py:69``).
  Cross-video numbers; these stay the primary metric.
* ``WV_AUC1/WV_AUC2`` are within-video AUCs - each video's scores z-scored and
  scored against that video's own GT, averaged over mixed-GT videos. This is
  the localisation convention every probe in this repo uses. Unlike the
  probes, the sign is NOT free here: the head is trained to score anomalies
  high, so the raw orientation is the honest one.
"""

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from freq_text_text import GT_FRAME_REPEAT
from utils.tools import get_batch_mask


def _f(x, what='value'):
    """``float`` with a named failure point (original exception type preserved)."""
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


def _within_video_auc(scores, lens, gt, repeat=GT_FRAME_REPEAT):
    """Mean AUC of per-video z-scored scores against that video's own GT.

    The pooled metrics are cross-video; this is the localisation convention the
    probes use, computed on the same model outputs so an ablation can be read
    under both conventions. Only videos containing both classes define an AUC;
    single-class videos are skipped. A GT slice shorter than the video's score
    count (only possible in a capped run) ends the loop rather than scoring a
    partial video.
    """
    offs = np.concatenate([[0], np.cumsum(np.asarray(lens, dtype=np.int64))])
    aucs = []
    for i in range(len(lens)):
        lo, hi = _i(offs[i]), _i(offs[i + 1])
        seg = np.asarray(gt)[lo * repeat: hi * repeat]
        if len(seg) < (hi - lo) * repeat:
            break
        g = seg.reshape(-1, repeat).mean(1) > 0.5
        if not (0 < g.sum() < len(g)):
            continue
        s = np.asarray(scores[lo:hi], dtype=np.float64)
        sd = s.std()
        if sd < 1e-12:
            continue
        z = (s - s.mean()) / sd
        aucs.append(roc_auc_score(g, z))
    if not aucs:
        return math.nan, 0
    return _f(np.mean(aucs)), len(aucs)


def _frame_probs(model, test_loader, maxlen, prompt_text, device, stack_logits=True):
    """Run the split-window test pass and return per-frame probs + mAP inputs."""
    model.to(device)
    model.eval()
    ap1 = None
    ap2 = None
    stack = []
    names = []
    lens = []

    with torch.no_grad():
        for item in test_loader:
            visual = item[0].squeeze(0)
            length = _i(item[2])
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)

            lengths = torch.zeros(_i(length / maxlen) + 1)
            # identical bookkeeping to VadCLIP's test loop (it consumes `length`
            # in place); kept verbatim so scores are comparable run-to-run
            for j in range(_i(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:    # noqa: SIM114 - verbatim
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            # normal class is index 0 in prompt_text, so 1 - p(normal) is the
            # text-discriminated anomaly score
            prob2 = 1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)

            ap1 = prob1 if ap1 is None else torch.cat([ap1, prob1], dim=0)
            ap2 = prob2 if ap2 is None else torch.cat([ap2, prob2], dim=0)

            if stack_logits:
                e = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
                stack.append(np.repeat(e, 16, 0))
            names.append(str(item[1][0]))
            lens.append(len_cur)

    if ap1 is None or ap2 is None:
        raise ValueError('test loader produced no videos - nothing to score')

    return ap1.cpu().numpy(), ap2.cpu().numpy(), stack, names, lens


def make_test_fn(dataset: str, save_scores: bool = False, out_path=None):
    """Build ``test(model, loader, maxlen, prompt_text, gt, gtseg, gtlabel, device)``.

    Signature matches VadCLIP's ``test`` so it can be passed straight into
    :func:`freq_text_trainer.run_train`. Returns ``(primary_metric, extra_dict)``
    with ``primary_metric`` = AP for XD and AUC1 for UCF.
    """
    from utils.ucf_detectionMAP import getDetectionMAP as ucf_map
    from utils.xd_detectionMAP import getDetectionMAP as xd_map
    dmAP = xd_map if dataset == 'xd' else ucf_map

    def test(model, test_loader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
        p1, p2, stack, _names, lens = _frame_probs(model, test_loader, maxlen,
                                                   prompt_text, device)
        s1, s2 = np.repeat(p1, 16), np.repeat(p2, 16)

        # A capped test loader (--max-test-clips, used by the smoke runs) predicts
        # fewer frames than the GT array holds. Left alone this crashes inside
        # roc_auc_score with two 7-digit lengths; worse, a partial-truncation bug
        # would instead score a *prefix* of the videos against the *whole* GT.
        # Truncate the GT to match what was actually predicted and say so.
        gt = np.asarray(gt)
        expected = len(s1)
        capped = len(gt) != expected
        if len(gt) < expected:
            raise ValueError(f'predicted {expected} frames but GT only has {len(gt)}')
        if capped:
            print(f'    [warn] test loader covered {expected} of {len(gt)} GT frames '
                  f'(capped run) - truncating GT; the metric below is NOT comparable')
            gt = gt[:expected]

        auc1 = roc_auc_score(gt, s1)
        ap1 = average_precision_score(gt, s1)
        auc2 = roc_auc_score(gt, s2)
        ap2 = average_precision_score(gt, s2)

        if capped:
            # gtsegments/gtlabels are per-video structures (a list of segment
            # index ranges and a list of labels), not flat arrays - slicing them by
            # frame count would corrupt them silently and produce a plausible
            # mAP number. Skip the detection metric instead: the smoke run's
            # purpose is to prove the AUC/AP plumbing works.
            avg_map = math.nan
            dmap, iou = [], []
        else:
            dmap, iou = dmAP(stack, gtsegments, gtlabels, excludeNormal=False)
            avg_map = _f(np.mean(dmap[:5]))

        # within-video (localisation) convention, on the same model outputs
        wv1, n_wv = _within_video_auc(p1, lens, gt)
        wv2, _ = _within_video_auc(p2, lens, gt)

        extra = {'AUC1': _f(auc1), 'AP1': _f(ap1),
                 'AUC2': _f(auc2), 'AP2': _f(ap2),
                 'det_mAP': avg_map,
                 'det_mAP_iou': {f'{v:.1f}': _f(dmap[i]) for i, v in enumerate(iou[:5])},
                 'WV_AUC1': wv1, 'WV_AUC2': wv2, 'WV_n_videos': n_wv}
        if save_scores and out_path:
            np.savez_compressed(out_path, prob1=p1, prob2=p2, gt=np.asarray(gt))
        primary = ap2 if dataset == 'xd' else auc1
        print(f'    AUC1 {auc1:.4f} AP1 {ap1:.4f} | AUC2 {auc2:.4f} AP2 {ap2:.4f} '
              f'| det mAP {avg_map:.2f} | wvAUC1 {wv1:.4f} wvAUC2 {wv2:.4f} (n={n_wv})')
        return _f(primary), extra

    return test


def main(dataset: str):
    """Load ``runs/<dataset>_<variant>/model_best.pth`` and evaluate it."""
    import json
    import os

    from torch.utils.data import DataLoader

    import freq_text_options
    from freq_text_dataset import ClipVideoDataset, collate
    from freq_text_prompts import vadclip_label_map
    from model import CLIPVAD
    from utils.tools import get_prompt_text

    # build_parser(dataset) already fixed every dataset-dependent default and
    # exposes no --dataset flag, so an entrypoint cannot desync from its lists
    args = freq_text_options.apply_variant(freq_text_options.build_parser(dataset).parse_args())
    args = freq_text_options.resolve_paths(args, dataset=dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'no checkpoint at {args.model_path}; train it first with '
                                f'{dataset}_train.py --variant {args.variant}')

    prompt_text = get_prompt_text(vadclip_label_map(dataset))
    test_loader = DataLoader(ClipVideoDataset(args.visual_length, args.test_list,
                                              test_mode=True, dataset=dataset),
                             batch_size=1, shuffle=False, collate_fn=collate)
    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    args.prompt_prefix, args.prompt_postfix, device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    gt = np.load(args.gt_path)
    gtseg = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabel = np.load(args.gt_label_path, allow_pickle=True)
    metric, extra = make_test_fn(dataset, save_scores=True,
                                 out_path=os.path.join(args.log_dir, 'scores.npz'))(
        model, test_loader, args.visual_length, prompt_text, gt, gtseg, gtlabel, device)
    print(f'primary metric ({"AP" if dataset == "xd" else "AUC"}): {metric:.4f}')
    print(json.dumps(extra, indent=2))
    return metric


if __name__ == '__main__':
    # convenience: python freq_text_test.py [xd|ucf] --variant E1
    # (the normal entrypoints are xd_test.py / ucf_test.py)
    import sys

    ds = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('xd', 'ucf') else 'xd'
    if ds in sys.argv[1:]:
        sys.argv.remove(ds)
    main(ds)
