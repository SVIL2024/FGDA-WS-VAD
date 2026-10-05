"""Repair stale absolute paths in the VadCLIP list CSVs.

The lists copied from VadCLIP were generated on the original author's machine and
36732 of XD's 39540 training rows still point at ``/home/xbgydx/Desktop/...``,
which does not exist here. Without a repair the loader silently drops them and XD
trains on 2808 clips - 7% of the data - while looking completely healthy.

Strategy: for every row whose path does not exist, look up its basename inside the
matching local feature directory. This is unambiguous because
``list/make_list_*.py`` names each feature file
``<video>__#<seg>_label_<label>__<idx>.npy``, and the label embedded in the name is
verified against the CSV's label column before a row is accepted (a basename
collision across two classes would otherwise inject label noise).

    python list/repair_list_paths.py --check-only
    python list/repair_list_paths.py                 # writes *.csv.bak + rewrites
    python list/repair_list_paths.py --lists list/xd_CLIP_rgb.csv

Rows that cannot be resolved are reported and *dropped* (with their count printed),
never left as a dangling path.
"""

import argparse
import csv
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import freq_text_options  # noqa: E402

LABEL_IN_NAME = re.compile(r'_label_(.+?)__\d+\.npy$')

# dataset -> the local directory that holds its feature .npy files
DEFAULT_ROOTS = [freq_text_options.data_root()]


def _read(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and r[0].strip()]
    return header, rows


def _embedded_label(name: str):
    m = LABEL_IN_NAME.search(name)
    return m.group(1) if m else None


def build_index(roots):
    """Index feature files under ``roots``.

    Two levels, because ``E:/dataset`` also holds unrelated datasets (I3D features
    for other VAD benchmarks share filenames with CLIP features - 28k collisions):

    * ``by_dir_name``: ``(parent_dir_name, basename) -> path`` - unambiguous, and
      correct here because the stale paths keep their original parent directory
      name (``/home/xbgydx/Desktop/XDTrainClipFeatures/x.npy`` -> local
      ``E:/dataset/XDTrainClipFeatures/x.npy``);
    * ``by_name``: ``basename -> path`` - a fallback used only when it is unique.
    """
    by_dir_name, by_name, ambiguous, collisions = {}, {}, set(), 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            parent = os.path.basename(dirpath)
            for fn in filenames:
                if not fn.endswith('.npy'):
                    continue
                full = os.path.join(dirpath, fn).replace('\\', '/')
                by_dir_name[(parent, fn)] = full
                prev = by_name.get(fn)
                if prev is not None and prev != full:
                    ambiguous.add(fn)
                    collisions += 1
                else:
                    by_name[fn] = full
    for fn in ambiguous:
        by_name.pop(fn, None)
    return by_dir_name, by_name, collisions


def resolve(path, by_dir_name, by_name):
    """Local path for a possibly-stale list entry, or None."""
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path.replace('\\', '/')))
    cand = by_dir_name.get((parent, base)) or by_name.get(base)
    return cand if cand and os.path.exists(cand) else None


def repair(list_path, by_dir_name, by_name, check_only=False, drop_unresolved=True):
    header, rows = _read(list_path)
    kept, fixed, dropped, label_conflicts = [], 0, [], 0

    for row in rows:
        path, label = row[0], row[1]
        if os.path.exists(path):
            kept.append(row)
            continue
        new = resolve(path, by_dir_name, by_name)
        base = os.path.basename(path)
        if new is None:
            dropped.append((path, label))
            continue
        # the CSV label must agree with the label baked into the filename
        emb = _embedded_label(base)
        if emb is not None and emb != label:
            label_conflicts += 1
            dropped.append((path, label))
            continue
        kept.append([new] + row[1:])
        fixed += 1

    total = len(rows)
    print(f'{os.path.basename(list_path)}: total={total} ok={total - len(dropped) - fixed} '
          f'remapped={fixed} unresolved={len(dropped)} label_conflicts={label_conflicts}')
    for path, label in dropped[:3]:
        print(f'    dropped label={label!r} {path}')

    if check_only:
        return len(kept), total, len(dropped)

    tmp = list_path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)
    if not drop_unresolved and len(dropped):
        os.remove(tmp)
        raise RuntimeError(f'{len(dropped)} unresolved rows in {list_path}; pass '
                           f'--keep-unresolved or fix the roots')
    shutil.copy2(list_path, list_path + '.bak')
    os.replace(tmp, list_path)
    return len(kept), total, len(dropped)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--lists', nargs='*', default=[
        os.path.join(here, n) for n in
        ('xd_CLIP_rgb.csv', 'xd_CLIP_rgbtest.csv', 'ucf_CLIP_rgb.csv', 'ucf_CLIP_rgbtest.csv')])
    ap.add_argument('--roots', nargs='*', default=DEFAULT_ROOTS,
                    help='directories scanned for feature .npy files')
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()

    print(f'scanning roots for .npy: {args.roots}')
    by_dir_name, by_name, collisions = build_index(args.roots)
    print(f'  indexed {len(by_dir_name)} (dir, file) pairs, {len(by_name)} unique '
          f'basenames, {collisions} ambiguous basenames excluded\n')
    if not by_dir_name:
        sys.exit('no .npy files found - pass --roots pointing at your feature directories')

    missing_total = 0
    for lp in args.lists:
        if not os.path.exists(lp):
            print(f'MISSING LIST {lp}')
            missing_total += 1
            continue
        kept, total, dropped = repair(lp, by_dir_name, by_name, check_only=args.check_only)
        missing_total += dropped
        if kept == 0:
            sys.exit(f'{lp}: nothing resolved, refusing to continue')

    if missing_total:
        print(f'\nWARNING {missing_total} rows could not be resolved and were dropped')
    else:
        print('\nall rows resolve')


if __name__ == '__main__':
    main()
