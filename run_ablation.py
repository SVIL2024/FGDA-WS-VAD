"""Run the E1-E5 (+B1) ablation sweep, gated on the precondition check.

    python run_ablation.py --dataset xd                  # full sweep
    python run_ablation.py --dataset xd --variants E5 E1 E3
    python run_ablation.py --dataset ucf --seed 234 235 236
    python run_ablation.py --dataset xd --dry-run        # print the plan only

Design notes, because a sweep script that is merely convenient is not worth
having:

* **The gate is enforced, not advised.** The precondition report is re-read (or
  produced) before any training starts, and a STOP verdict aborts the sweep. The
  whole point of the precondition experiment is to avoid spending GPU days on a
  premise that cannot hold. ``--force`` exists for the case where you have
  deliberately changed the direction source and want to override, and it prints
  exactly what is being overridden.

* **One directory per (variant, seed).** Results are read back from each run's
  ``result.json`` rather than scraped from stdout, so a crashed run is visibly
  missing instead of silently absent from a table.

* **Resumable.** A (variant, seed) whose ``result.json`` already exists is
  skipped unless ``--overwrite``. Multi-day sweeps get interrupted; re-running
  should not throw away completed work.

* **Seeds are explicit and repeated.** The precondition probe says the expected
  E1-E3 gap is on the order of 0.02 AUC, which is inside seed noise for a single
  run. ``--seed`` takes a list, and the summary reports the spread across seeds
  rather than a single number, so a difference smaller than the seed spread is
  visible as what it is.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src')
sys.path.insert(0, SRC)

import freq_text_options as opt  # noqa: E402

# E5 first (the baseline everything is compared to), then the proposal, then the
# control that decides whether the proposal means anything, then the weaker ones.
DEFAULT_ORDER = ['E5', 'E1', 'E3', 'B1', 'E2', 'E4']
# variants that actually use a text direction, i.e. ones the precondition gate
# applies to. E5 has no augmentation and E3 uses a random direction by design, so
# a dead text direction does not invalidate them.
NEEDS_TEXT = {'E1', 'E2', 'E4', 'B1'}


def parse_args():
    ap = argparse.ArgumentParser(
        description='Run the Freq-Text Aug ablation sweep (precondition-gated).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['xd', 'ucf'])
    ap.add_argument('--variants', nargs='+', default=None, choices=DEFAULT_ORDER,
                    help=f'default: {DEFAULT_ORDER}')
    ap.add_argument('--seed', nargs='+', type=int, default=[234],
                    help='one or more seeds; the summary reports the spread')
    ap.add_argument('--max-epoch', type=int, default=None,
                    help='omit for the dataset default (VadCLIP schedule)')
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--run-prefix', default=None,
                    help='run dir is <prefix><variant>_s<seed>; default <dataset>_')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--overwrite', action='store_true',
                    help='re-run (variant, seed) pairs that already have a result')
    ap.add_argument('--force', action='store_true',
                    help='run even if the precondition verdict is STOP')
    ap.add_argument('--skip-precondition', action='store_true',
                    help='do not run the probe; still reads an existing report')
    ap.add_argument('--shutdown', action='store_true',
                    help='power the instance off once the whole plan finishes; the '
                         'driver is the only thing allowed to do this')
    ap.add_argument('--force-unlock', action='store_true',
                    help='clear a lock left behind by a killed driver')
    # Unknown flags are forwarded verbatim to the trainer, so a smoke run can be
    # written as `... -- --max-train-steps 2` without this script needing an
    # option for every trainer knob (several of which exist only for smoke runs).
    args, extra = ap.parse_known_args()
    if extra and extra[0] == '--':
        extra = extra[1:]
    args.extra = extra
    return args


def acquire_lock(root, force_unlock=False):
    """One driver at a time: two concurrent drivers double-book the GPU and OOM.

    The lock is created with O_CREAT|O_EXCL, which is atomic on every platform.
    An earlier version probed liveness with ``os.kill(pid, 0)``: that is a harmless
    existence check on POSIX, but on Windows CPython maps it to
    ``TerminateProcess``, so the probe killed whichever process held the lock -
    including the test runner, which is the only reason this is worth writing down.

    Reclaiming an existing lock therefore requires *proving* its owner is dead. A
    first attempt instead treated "could not probe" as "stale" on non-POSIX, which
    is the exact inverse of the safe direction: every live lock was being stolen by
    the second driver, so the guard protected nothing. Unknown now means refuse.
    """
    lock = os.path.join(root, 'runs', '.ablation.lock')
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pass
    except OSError as exc:
        print(f'[lock] cannot create {lock}: {exc}')
        return None
    else:
        with os.fdopen(fd, 'w') as f:
            f.write(str(os.getpid()))
        return lock

    pid = None
    try:
        with open(lock) as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        pid = None
    if pid == os.getpid():
        if force_unlock:
            print('[lock] --force-unlock: re-acquiring our own lock')
        else:
            print('[lock] this process already holds the lock')
            return None
    elif pid is not None and os.name == 'posix':
        try:
            os.kill(pid, 0)
        except OSError:
            print(f'[lock] removing stale lock (pid {pid} is gone)')
        else:
            if not force_unlock:
                print(f'[lock] a driver is already running (pid {pid}); refusing to '
                      f'launch a second one. If it is dead, use --force-unlock.')
                return None
            print(f'[lock] --force-unlock: taking over from live pid {pid}')
    else:
        # Either no readable pid, or a platform where a liveness probe would kill
        # the owner. Unknown is not dead.
        if not force_unlock:
            print(f'[lock] {lock} exists (pid {pid}) and liveness cannot be probed '
                  f'safely on this platform; refusing. Use --force-unlock if no '
                  f'driver is running.')
            return None
        print(f'[lock] --force-unlock: removing lock (pid {pid})')
    try:
        os.remove(lock)
    except OSError as exc:
        print(f'[lock] cannot remove stale lock: {exc}')
        return None
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as exc:
        print(f'[lock] lost the race to re-create the lock: {exc}')
        return None
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    return lock


def already_done(tag, root):
    """True when this (variant, seed) already recorded its [Total] verdict.

    The trainer appends that line the moment a run finishes, so a driver that
    dies mid-sweep still leaves behind exactly which experiments completed.
    """
    summary = os.path.join(root, 'runs', 'results_summary.txt')
    if not os.path.exists(summary):
        return False
    with open(summary) as f:
        return any(line.startswith(f'[Total] {tag} ') for line in f)


def precondition_verdict(dataset, skip_probe):
    """(proceed, verdict_text) from runs/precondition_<dataset>.json."""
    path = os.path.join(opt.repo_root(), 'runs', f'precondition_{dataset}.json')
    if skip_probe or not os.path.exists(path):
        cmd = [sys.executable, os.path.join(SRC, 'precondition_test.py'),
               '--dataset', dataset, '--device', 'cpu']
        print(f'[gate] running precondition probe: {" ".join(cmd)}')
        r = subprocess.run(cmd, cwd=SRC)
        if not os.path.exists(path):
            return False, f'probe produced no report at {path} (exit {r.returncode})'
    try:
        with open(path, encoding='utf-8') as f:
            rep = json.load(f)
    except Exception as exc:                       # noqa: BLE001
        return False, f'could not read {path}: {exc}'
    return bool(rep.get('proceed')), str(rep.get('verdict', 'no verdict recorded'))


def _result_path(run_name):
    """Where a completed run's numbers live, or None if the run has not finished.

    The trainer writes ``<tag>_history.json``. ``tag`` is ``run_name`` when the
    caller set one and ``<dataset>_<variant>`` otherwise, so both spellings are
    accepted. Getting this wrong is not a crash - it silently yields an empty
    summary table after a sweep that reported every run as successful, which is
    the worst possible failure mode for a results script.
    """
    run_dir = os.path.join(opt.repo_root(), 'runs', run_name)
    if not os.path.isdir(run_dir):
        return None
    direct = os.path.join(run_dir, f'{run_name}_history.json')
    if os.path.exists(direct):
        return direct
    for fname in sorted(os.listdir(run_dir)):
        if fname.endswith('_history.json'):
            return os.path.join(run_dir, fname)
    return None


def run_one(dataset, variant, seed, args):
    run_name = f'{args.run_prefix or (dataset + "_")}{variant}_s{seed}'
    out_dir = os.path.join(opt.repo_root(), 'runs', run_name)
    result = _result_path(run_name)
    # The [Total] marker is the skip test, not the presence of a history file: the
    # marker is written the instant a run finishes, so a driver killed between runs
    # still knows what completed. A history file can exist for a run that died
    # mid-way, and skipping on that would silently lose the experiment.
    if already_done(run_name, opt.repo_root()) and not args.overwrite:
        print(f'[skip] {run_name} already recorded a [Total] verdict '
              f'(use --overwrite to redo)')
        return run_name, True
    if result and not args.overwrite and not args.dry_run:
        print(f'[warn] {run_name} has a history file but no [Total] marker - '
              f'treating it as incomplete and re-running')

    # -u keeps the child's stdout line-buffered. Without it the driver log stays
    # empty for minutes and a monitoring pass cannot tell "still starting" from
    # "hung".
    cmd = [sys.executable, '-u', os.path.join(SRC, f'{dataset}_train.py'),
           '--variant', variant, '--seed', str(seed), '--run-name', run_name]
    if args.max_epoch is not None:
        cmd += ['--max-epoch', str(args.max_epoch)]
    if args.batch_size is not None:
        cmd += ['--batch-size', str(args.batch_size)]
    if args.extra:
        extra = [a for a in args.extra if a != '--']
        cmd += extra

    print(f'[run ] {run_name}\n       {" ".join(cmd)}')
    if args.dry_run:
        return run_name, None
    t0 = time.time()
    log_path = os.path.join(out_dir, 'stdout.log')
    os.makedirs(out_dir, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, cwd=SRC, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f'[FAIL] {run_name} exited {proc.returncode} after {dt:.0f}s; '
              f'see {log_path}')
        return run_name, False
    if not already_done(run_name, opt.repo_root()):
        print(f'[warn] {run_name} exited 0 but wrote no [Total] marker - the run '
              f'probably died before its first eval; not counting it as done')
        return run_name, False
    print(f'[done] {run_name} in {dt:.0f}s')
    return run_name, True


def summarize(dataset, run_names, seeds):
    """Read each run's result.json and print a table plus the seed spread."""
    rows = []
    for name in run_names:
        path = _result_path(name)
        if not path:
            rows.append((name, None, None, 'MISSING'))
            continue
        try:
            with open(path, encoding='utf-8') as f:
                r = json.load(f)
        except Exception as exc:                   # noqa: BLE001
            rows.append((name, None, None, f'unreadable ({exc})'))
            continue
        variant = (r.get('args') or {}).get('variant', '')
        rows.append((name, r.get('best'), r.get('metric_name'), variant))

    print()
    print('=' * 74)
    print(f'ABLATION SUMMARY  dataset={dataset}')
    print('=' * 74)
    metric = next((m for _, _, m, _ in rows if m), 'metric')
    print(f'{"run":<26} {"variant":<8} {metric:>10}')
    print('-' * 74)
    per_variant = {}
    for name, val, _m, variant in rows:
        variant = variant or name.rsplit('_s', 1)[0].split('_')[-1]
        if val is None:
            print(f'{name:<26} {variant:<8} {"-":>10}   {_m}')
            continue
        print(f'{name:<26} {variant:<8} {val:>10.4f}')
        per_variant.setdefault(variant, []).append(val)

    if not per_variant:
        print('\nNo results found. Every run above reported success, so this means')
        print('the summary could not locate the history files - check that')
        print('runs/<run_name>/ contains a *_history.json.')
    if len(seeds) > 1:
        print('-' * 74)
        print(f'{"variant":<8} {"n":>3} {"mean":>9} {"std":>9} {"min":>9} {"max":>9}')
        for v in DEFAULT_ORDER:
            vals = per_variant.get(v, [])
            if not vals:
                continue
            m = sum(vals) / len(vals)
            sd = (sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5
            print(f'{v:<8} {len(vals):>3} {m:>9.4f} {sd:>9.4f} '
                  f'{min(vals):>9.4f} {max(vals):>9.4f}')
        print('-' * 74)
        print('A difference smaller than the std above is inside seed noise. The')
        print('precondition probe predicts an E1-E3 gap near 0.02, so a single run')
        print('cannot settle it.')
    print('=' * 74)

    base = per_variant.get('E5')
    prop = per_variant.get('E1')
    ctrl = per_variant.get('E3')
    if base and prop and ctrl:
        b, p, c = (sum(x) / len(x) for x in (base, prop, ctrl))
        print(f'E5 {b:.4f} | E1 {p:.4f} ({p - b:+.4f}) | E3 {c:.4f} ({c - b:+.4f})')
        print(f'E1 - E3 = {p - c:+.4f}   <- this is the number the paper turns on.')
        print('If it is not clearly positive, the text direction is not doing the')
        print('work and the augmentation is a generic perturbation.')


def shutdown_instance(reason):
    """Driver-level power off. Never called by a training script.

    Only reached when --shutdown was passed, so a plain sweep for a single
    experiment cannot take the machine down. AutoDL preserves the data disk
    across a stop, which is what makes this safe to automate.
    """
    print(f'[Shutdown] {reason}')
    for cmd in (['shutdown', '-h', 'now'], ['poweroff']):
        try:
            subprocess.run(cmd, check=False, timeout=30)
            print(f'[Shutdown] ran: {" ".join(cmd)}')
            return
        except (OSError, subprocess.SubprocessError) as exc:
            print(f'[Shutdown] WARN: {" ".join(cmd)} failed ({exc})')
    print('[Shutdown] WARN: instance not powered off - stop it manually')


def main():
    args = parse_args()
    variants = args.variants or DEFAULT_ORDER
    root = opt.repo_root()

    if not args.dry_run:
        if acquire_lock(root, args.force_unlock) is None:
            return 4
    try:
        print(f'[plan] dataset={args.dataset} variants={variants} seeds={args.seed}')
        if not args.dry_run:
            needs = [v for v in variants if v in NEEDS_TEXT]
            if needs:
                ok, verdict = precondition_verdict(args.dataset, args.skip_precondition)
                print(f'[gate] proceed={ok} for variants {needs}')
                print(f'[gate] {verdict}')
                if not ok and not args.force:
                    print('\nAborting: the precondition gate did not pass. Do not tune '
                          'alpha around this - change the direction source, or pass '
                          '--force to record the negative result deliberately.')
                    return 3
                if not ok and args.force:
                    print('\n[!] --force: running the sweep against a failing '
                          'precondition. Any E1 gain over E3 would contradict the '
                          'probe; treat the outcome as a sanity check, not a result.')

        run_names, failed = [], 0
        for variant in variants:
            for seed in args.seed:
                name, ok = run_one(args.dataset, variant, seed, args)
                run_names.append(name)
                if ok is False:
                    failed += 1
                    print('[!] continuing after a failed run; the summary will show a gap')

        if not args.dry_run:
            summarize(args.dataset, run_names, args.seed)
        return 1 if failed else 0
    finally:
        lock = os.path.join(root, 'runs', '.ablation.lock')
        if os.path.exists(lock):
            try:
                os.remove(lock)
            except OSError:
                pass
        if args.shutdown and not args.dry_run:
            shutdown_instance('sweep finished, results recorded')


if __name__ == '__main__':
    sys.exit(main())
