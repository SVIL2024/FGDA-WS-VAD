"""Build the SHT test list + flat frame GT aligned to stride-16 features.

    python src/sht_build_gt.py

Inputs (already on disk):
* ``E:\\dataset\\shanghaitech\\label\\test.csv`` - one line per test video:
  ``<frames dir> <0/1 per native frame>`` (frame-level GT, verified format);
* ``E:\\dataset\\SHTClipFeatures\\test\\<video>.npy`` - stride-16 features,
  ``T = ceil(n_frames / 16)``.

Outputs:
* ``E:\\dataset\\SHTClipFeatures\\sht_CLIP_rgbtest.csv`` - ``path,label`` rows
  in sorted video order; the label column uses XD's normal code ``A`` because
  cross_eval loads the list with ``dataset='xd'`` label parsing and the label
  plays no role in evaluation;
* ``E:\\dataset\\SHTClipFeatures\\gt_sht.npy`` - flat int8 array, length
  ``sum(T_i) * 16``: each video's per-block label (block = 1 if ANY native
  frame in its 16-frame window is anomalous) expanded 16x, concatenated in
  list order - the exact alignment contract of the pooled metric
  (``roc_auc_score(gt, np.repeat(scores, 16))``).

Validation is the whole point: a length mismatch between a video's GT line and
its frame directory, or between the feature array and ceil(n/16), is reported
and the video is DROPPED from both list and GT (counted), never silently
scored against a shifted array.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import freq_text_options  # noqa: E402

SHT_ROOT = freq_text_options.sht_root()
FEAT_ROOT = freq_text_options.sht_feat_root()
LABEL_CSV = os.path.join(SHT_ROOT, 'label', 'test.csv')
STRIDE = 16


def main():
    rows = []
    with open(LABEL_CSV) as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) != 2 or not parts[0]:
                continue
            rows.append((parts[0], np.fromstring(parts[1], dtype=np.int8, sep=' ')))
    print(f'label csv rows: {len(rows)}')

    out_rows, segs = [], []
    dropped = []
    for path, values in sorted(rows):
        vid = os.path.basename(path.rstrip('\\/'))
        n = len(values)
        frames_dir = path if os.path.isdir(path) else os.path.join(SHT_ROOT, 'testing', 'frames', vid)
        n_png = len([f for f in os.listdir(frames_dir)
                     if f.lower().endswith(('.png', '.jpg'))]) if os.path.isdir(frames_dir) else -1
        feat_path = os.path.join(FEAT_ROOT, 'test', vid + '.npy')
        if not os.path.exists(feat_path):
            dropped.append((vid, 'no feature file'))
            continue
        feat = np.load(feat_path)
        t_feat = feat.shape[0]
        if feat.ndim != 2 or feat.shape[1] != 512:
            dropped.append((vid, f'bad feature shape {feat.shape}'))
            continue
        if t_feat != (n + STRIDE - 1) // STRIDE:
            dropped.append((vid, f'feature T={t_feat} vs ceil({n}/16)'))
            continue
        if n_png != n:
            dropped.append((vid, f'{n_png} png frames vs {n} gt frames'))
            continue
        padded = np.zeros(t_feat * STRIDE, dtype=np.int8)
        padded[:n] = values          # tail block: frames beyond the GT line are
        blocks = padded.reshape(t_feat, STRIDE)   # unlabelled -> 0 (no anomaly)
        block_gt = (blocks.sum(1) > 0).astype(np.int8)
        segs.append(np.repeat(block_gt, STRIDE))
        out_rows.append((feat_path, 'A'))          # 'A' = XD normal code: the
        # label column is unused by evaluation; parse_label needs a valid token
    gt = np.concatenate(segs) if segs else np.zeros(0, dtype=np.int8)

    csv_path = os.path.join(FEAT_ROOT, 'sht_CLIP_rgbtest.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label'])
        w.writerows(out_rows)
    gt_path = os.path.join(FEAT_ROOT, 'gt_sht.npy')
    np.save(gt_path, gt)

    n_anom_frames = int(gt.sum())
    print(f'list rows: {len(out_rows)}  dropped: {len(dropped)}')
    for vid, why in dropped[:10]:
        print(f'  DROPPED {vid}: {why}')
    print(f'gt frames: {len(gt)} (= sum(T_i) x 16)  anomalous frames: {n_anom_frames} '
          f'({n_anom_frames / max(len(gt), 1):.1%})')
    print(f'saved -> {csv_path}')
    print(f'saved -> {gt_path}')
    print('SHT_GT_DONE')


if __name__ == '__main__':
    main()
