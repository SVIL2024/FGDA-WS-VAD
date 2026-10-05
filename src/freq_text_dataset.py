"""Datasets for Freq-Text Aug, aligned with VadCLIP's list files.

Split out of ``utils/dataset.py`` because the augmentation needs one field the
upstream datasets do not return: the *category* of each video, so the trainer can
look up a per-class CLIP text direction.

Three things are normalised here rather than patched at every call site:

* **dtype.** The CLIP feature banks are stored ``float16``. ``tools.process_feat``
  returns ``float32`` when it has to resample but keeps ``float16`` on the padding
  branch, so a batch can mix dtypes and ``default_collate``/``torch.stack`` will
  promote unpredictably. Everything is cast to float32 here.
* **label.** ``utils.tools.get_batch_label`` wants the raw string (XD's is
  composite, ``B1-B2-0``), and the augmentation wants an integer class index.
  Both are returned. ``utils.tools`` itself is left untouched.
* **train split.** VadCLIP's UCF trainer builds two loaders (normal / anomaly) by
  filtering the CSV; the XD trainer uses the unfiltered CSV. ``mode`` reproduces
  both without duplicating the filtering logic.
"""

import numpy as np
import torch
import torch.utils.data as data

import freq_text_options

import utils.tools as tools
from freq_text_prompts import parse_label


class ClipVideoDataset(data.Dataset):
    """Rows of a VadCLIP list CSV: ``path,label``.

    Args:
        clip_dim: sequence length ``T`` (``args.visual_length``, 256).
        file_path: list CSV.
        test_mode: use ``process_split`` (test) instead of ``process_feat``.
        normal: keep only normal (True) or only anomalous (False) rows; ``None``
            keeps everything, which is what the XD trainer does.
        dataset: ``'xd'`` / ``'ucf'``, used for label parsing.
    """

    def __init__(self, clip_dim: int, file_path: str, test_mode: bool = False,
                 normal=None, dataset: str = 'xd'):
        self.dataset = dataset
        # cwd-independent: list files may be given relative to the project root
        file_path = freq_text_options.resolve_data_path(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.paths, self.labels = [], []
        with open(file_path) as f:
            f.readline()  # header: path,label
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip('\n').split(',')
                if len(parts) < 2:
                    continue
                self.paths.append(parts[0])
                self.labels.append(parts[1])
        if normal is not None and not test_mode:
            # normal=True -> keep normals, normal=False -> keep anomalies
            keep = [i for i, l in enumerate(self.labels)
                    if (not parse_label(l, dataset)[1]) == normal]
            self.paths = [self.paths[i] for i in keep]
            self.labels = [self.labels[i] for i in keep]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        feat = np.load(self.paths[index])
        if self.test_mode:
            feat, length = tools.process_split(feat, self.clip_dim)
        else:
            feat, length = tools.process_feat(feat, self.clip_dim)
        feat = torch.from_numpy(np.ascontiguousarray(feat)).float()

        label = self.labels[index]
        class_idx, anom, _ = parse_label(label, self.dataset)
        # an unmapped label degrades to "anomaly, no specific class" -> index 0
        # keeps the video in the MIL loss without contributing a text direction
        class_idx = 0 if class_idx is None else class_idx
        binary = 1.0 if anom else 0.0
        return feat, label, int(length), class_idx, binary


def collate(batch):
    """``(feat, raw_label_str, lengths, class_idx, binary_label)``.

    ``raw_label_str`` stays a tuple of strings for ``utils.tools.get_batch_label``;
    ``BaseTool.collate_fn`` cannot be used because it does ``float()`` on it.
    """
    feats = torch.stack([b[0] for b in batch], dim=0)
    labels = tuple(b[1] for b in batch)
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    class_idx = torch.tensor([b[3] for b in batch], dtype=torch.long)
    binary = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    return feats, labels, lengths, class_idx, binary
