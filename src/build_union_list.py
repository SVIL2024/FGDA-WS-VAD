"""Build the XD+UCF merged training list for multi-source (S1) training.

    python src/build_union_list.py

Reads the two native training lists (absolute paths, verified on both machines)
and writes ``list/union_CLIP_rgb.csv``: XD rows pass through unchanged (their
composite codes are already union codes); UCF rows are remapped to union codes
via ``freq_text_prompts.UCF_NAME_TO_UNION_CODE`` ('Normal' -> 'A'). Any UCF
label without a mapping is a hard error - a silently mislabelled row would
poison the MIL supervision.

The merged list is written once and then is a static artifact (the anti-rerun
inventory covers it): rebuilding is only needed if a source list changes.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import freq_text_options  # noqa: E402
from freq_text_prompts import UCF_NAME_TO_UNION_CODE  # noqa: E402
from freq_text_text import _read_list  # noqa: E402


def main():
    base = os.path.join(freq_text_options.repo_root(), 'list')
    xd_rows = _read_list(os.path.join(base, 'xd_CLIP_rgb.csv'))
    ucf_rows = _read_list(os.path.join(base, 'ucf_CLIP_rgb.csv'))
    print(f'xd rows: {len(xd_rows)}  ucf rows: {len(ucf_rows)}')

    out = []
    unmapped = {}
    for p, lab in ucf_rows:
        name = str(lab).strip()
        if name.lower() == 'normal':
            out.append((p, 'A'))
            continue
        code = UCF_NAME_TO_UNION_CODE.get(name)
        if code is None:
            unmapped[name] = unmapped.get(name, 0) + 1
            continue
        out.append((p, code))
    if unmapped:
        raise SystemExit(f'unmapped UCF labels (fix UCF_NAME_TO_UNION_CODE): {unmapped}')

    out_path = os.path.join(base, 'union_CLIP_rgb.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label'])
        w.writerows(xd_rows + out)

    n_normal = sum(1 for _p, lab in out if lab == 'A')
    print(f'ucf remapped: {len(out)} rows ({n_normal} normal, '
          f'{len(out) - n_normal} anomaly)')
    print(f'union total: {len(xd_rows) + len(out)} rows -> {out_path}')
    print('UNION_LIST_DONE')


if __name__ == '__main__':
    main()
