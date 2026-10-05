"""CPU tests for the training-protocol changes: head selection, threshold gating,
the log header, the warmup schedule and the driver's crash-safe skip.

    python tests/test_training_protocol.py      (no pytest required)

These run on CPU with no dataset, no checkpoint and no GPU, and they execute the
real functions rather than re-implementing their logic. They exist because the
protocol changes are almost entirely about *which* number is reported and *when*
a file is written - the class of bug that produces a plausible, wrong experiment
rather than an exception.

Layer 1 (py_compile) and layer 2 (pyflakes) cannot see any of this: the head
pairing rule, the "better but below threshold, so not saved" branch, and a
warmup scheduler that is stepped in the wrong place all type-check perfectly and
all silently produce a wrong number.
"""

import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
# run_ablation.py is a root-level driver, not a src module, so it needs the repo
# root on the path too. Importing it is what lets the skip/lock helpers be tested
# on CPU instead of only being exercised on a GPU box.
sys.path.insert(0, ROOT)

import freq_text_options as opt                                    # noqa: E402
import run_ablation as ra                                          # noqa: E402
from analysis_magnitude_orthogonal import build_parser as mag_parser  # noqa: E402
from freq_text_trainer import (config_keyvals, early_stop_state,  # noqa: E402
                               log_header, select_head)
from utils.lr_warmup import WarmupMultiStepLR                     # noqa: E402

# The E5 baseline actually measured on the remote server: 0.8468 best AP across
# three seeds, all peaking at epoch 1 and then overfitting. If the default
# threshold is ever lowered below this, the gate stops protecting anything and
# the 600 MB non-qualifying checkpoints come back.
MEASURED_E5_BEST_AP = 0.8468
BASE_LR = 1e-5                               # VadCLIP's XD learning rate


def _both(ap1, ap2, auc1, auc2):
    return {'AP1': ap1, 'AP2': ap2, 'AUC1': auc1, 'AUC2': auc2}


def test_xd_selects_the_larger_ap_and_never_mixes_heads():
    m, auc, head = select_head('xd', 'AP', _both(0.90, 0.70, 0.95, 0.80))
    assert (m, auc, head) == (0.90, 0.95, 'head1'), (m, auc, head)
    m, auc, head = select_head('xd', 'AP', _both(0.70, 0.90, 0.95, 0.80))
    assert (m, auc, head) == (0.90, 0.80, 'head2'), (m, auc, head)


def test_the_paired_auc_always_belongs_to_the_winning_head():
    # head1 wins on AP but head2 has the higher AUC: quoting head2's AUC would
    # make the filename describe a model that does not exist.
    extra = _both(0.91, 0.50, 0.60, 0.99)
    m, auc, head = select_head('xd', 'AP', extra)
    assert head == 'head1' and m == 0.91 and auc == 0.60, (head, m, auc)
    extra = _both(0.50, 0.91, 0.99, 0.60)
    m, auc, head = select_head('xd', 'AP', extra)
    assert head == 'head2' and m == 0.91 and auc == 0.60, (head, m, auc)


def test_ucf_selects_the_larger_auc_paired_with_its_own_ap():
    m, ap, head = select_head('ucf', 'AUC', _both(0.30, 0.32, 0.91, 0.88))
    assert (m, ap, head) == (0.91, 0.30, 'head1'), (m, ap, head)
    m, ap, head = select_head('ucf', 'AUC', _both(0.30, 0.32, 0.88, 0.93))
    assert (m, ap, head) == (0.93, 0.32, 'head2'), (m, ap, head)


def test_a_missing_head_returns_none_so_the_fallback_engages():
    # A silent 0.0 here would be counted as "the run scored zero" and would also
    # fail the threshold gate, hiding the real reason.
    assert select_head('xd', 'AP', {'AP1': 0.5, 'AUC1': 0.6}) == (None, None, 'head?')
    assert select_head('ucf', 'AUC', {}) == (None, None, 'head?')


def test_dataset_defaults_are_the_published_thresholds():
    for dataset, thr in (('xd', 0.855), ('ucf', 0.875)):
        a = opt.build_parser(dataset).parse_args([])
        a = opt.apply_variant(a)
        a = opt.resolve_paths(a, dataset=dataset)
        assert a.save_threshold == thr, (dataset, a.save_threshold)


def test_baseline_defaults_must_equal_the_upstream_recipe():
    """E5 is VadCLIP. Its defaults may not drift from the authors' code.

    Every knob below changes the training trajectory or which epoch gets reported.
    A previous version of this file had warmup on by default, dual-head selection
    on by default and early stopping at patience 3, all introduced as "baseline
    repair". That silently made E5 something other than VadCLIP, which invalidates
    any claim of the form "we beat VadCLIP" - the whole point of running a baseline
    is that it is the thing being compared against, unmodified. Deviations are
    allowed, but only as explicit flags with a stated reason.
    """
    for dataset in ('xd', 'ucf'):
        a = opt.build_parser(dataset).parse_args([])
        a = opt.apply_variant(a)
        a = opt.resolve_paths(a, dataset=dataset)
        assert a.warmup_pct == 0.0, (
            f'{dataset}: warmup must be opt-in; on by default it changes the '
            f'optimisation trajectory and E5 stops being VadCLIP')
        assert a.early_stop_patience == 0, (
            f'{dataset}: early stopping must be opt-in; it changes which epoch is '
            f'reported, so an E5 number from an early-stopped run is not comparable')
        assert a.best_metric == 'test_fn', (
            f'{dataset}: selection must default to the upstream test loop (AP2 on XD, '
            f'AUC1 on UCF). Switching to max(AP1,AP2) picks a different epoch and '
            f'breaks comparability with the published numbers')


def test_test_fn_reproduces_the_epoch_upstream_actually_selects():
    """Upstream ends xd_test.test in `return ROC1, AP2, 0` and the loop does
    `if AP > ap_best`, so on XD the selected epoch is best AP2 - not max(AP1, AP2).
    On UCF it is best AUC1. This was the default path and nothing covered it, which
    is how 'test_fn' could silently fall through to the AUC branch and start
    reporting the wrong head."""
    extra = {'AP1': 0.80, 'AP2': 0.86, 'AUC1': 0.79, 'AUC2': 0.85}
    assert select_head('xd', 'test_fn', extra) == (0.86, 0.85, 'head2')
    assert select_head('ucf', 'test_fn', extra) == (0.79, 0.80, 'head1')
    assert select_head('xd', 'test_fn', {'AP1': 0.8}) == (None, None, 'head?')
    assert select_head('ucf', 'test_fn', {}) == (None, None, 'head?')


def test_the_history_key_is_the_metric_not_the_selection_mode():
    for dataset, expected in (('xd', 'AP'), ('ucf', 'AUC')):
        a = opt.build_parser(dataset).parse_args(['--variant', 'E5', '--seed', '234'])
        a = opt.apply_variant(a)
        a = opt.resolve_paths(a, dataset=dataset)
        key = a.best_metric
        if key == 'test_fn':
            key = 'AP' if a.dataset == 'xd' else 'AUC'
        assert key == expected, (dataset, key)


def test_the_pre_registered_d2_rule_is_the_default():
    """§6 of magnitude_orthogonal_design.md lists D2 under the kill conditions and
    says any one suffices, so D2 alone must reject the text branch.

    The script originally shipped the narrower reading, where D2 only counts
    together with D1, and that narrowing moved XD from kill to PREMISE CONFIRMED
    after the data was in. The default is now the rule that was written down
    first; the narrowing is reachable only as an explicit `--d2-rule joint`. This
    test exists so the favourable default cannot quietly come back."""
    assert mag_parser().parse_args([]).d2_rule == 'kill'
    for rule in ('kill', 'joint', 'symmetric'):
        assert mag_parser().parse_args(['--d2-rule', rule]).d2_rule == rule, rule


def test_no_unbacked_loose_checkpoint_is_sitting_in_runs():
    """A `.pth` directly in runs/ is not a result, and clean.py cannot remove one.

    clean.py:92 `continue`s on every non-directory, so only RUN_DIRS ever reach
    shutil.rmtree (:132) and os.remove (:125) is reserved for pycache and smoke logs.
    A stray checkpoint is therefore invisible to `--runs` *and* `--all-runs` and
    persists forever. One did: a 1-byte file named
    `XD_auc0.8612_ap0.8591_...warmup_pct=0.3..._early_stop_patience=3.pth`, i.e. an E1
    result that clears the 0.855 threshold and beats the published 84.51, with no
    `_history.json`, no `stdout.log` and empty run dirs behind it.

    Size is checked rather than deserialised: a real CLIPVAD checkpoint is ~607 MB, and
    loading one per test would cost more than the guard is worth.
    """
    runs = os.path.join(ROOT, 'runs')
    if not os.path.isdir(runs):
        return
    real = os.listdir(runs)
    loose = [n for n in real if n.lower().endswith('.pth')]
    assert not loose, (
        f'loose checkpoint(s) directly in runs/ with no run dir: {loose}. These are not '
        f'results and clean.py will never delete them - quarantine or delete by hand.')
    # A results dir whose only checkpoint is implausibly small means a truncated write
    # survived under a name that still looks authoritative.
    for name in real:
        if not os.path.isdir(os.path.join(runs, name)):
            continue
        tag = name
        hist = os.path.join(runs, name, f'{tag}_history.json')
        if not os.path.exists(hist):
            continue
        for n in os.listdir(os.path.join(runs, name)):
            if n.lower().endswith('.pth'):
                size = os.path.getsize(os.path.join(runs, name, n))
                assert size > 1_000_000, (
                    f'{name}/{n} is {size} bytes; a CLIPVAD checkpoint is ~607 MB, so '
                    f'this is a truncated or stub write, not a result')


def test_premise_smoke_runs_can_be_redirected_away_from_the_real_artifact():
    """A reduced --n-videos/--n-random run writes degraded numbers under the same
    mag_ortho_<dataset>.json filename and still prints a clean-looking verdict. That
    has already overwritten a real full-quality artifact once, so --out-dir exists to
    make 'write it somewhere else' actually possible rather than just advice."""
    assert mag_parser().parse_args([]).out_dir == 'runs'
    assert mag_parser().parse_args(
        ['--out-dir', 'scratch']).out_dir == 'scratch'


def test_early_stop_patience_zero_is_disabled_not_exhausted():
    """patience 0 is the upstream default and means OFF.

    The bug this pins: the counter starts at 0 and is reset to 0 on every improvement,
    so the original inline test was `patience_left <= 0` - true at epoch 1. Every
    default run trained a single epoch while the log header still advertised the
    upstream 10-epoch schedule. Nothing caught it because the logic sat inside the
    training loop, unreachable from a test.
    """
    # disabled: never stops, however many epochs pass without improvement
    left, stop = early_stop_state(0, 0, True)
    assert stop is False
    for _ in range(50):
        left, stop = early_stop_state(0, left, False)
        assert stop is False, 'patience=0 must never stop the run'

    # enabled: an improvement resets the counter, patience non-improvements end it
    left, stop = early_stop_state(3, 3, True)
    assert (left, stop) == (3, False)
    left, stop = early_stop_state(3, left, False)
    assert (left, stop) == (2, False)
    left, stop = early_stop_state(3, left, False)
    assert (left, stop) == (1, False)
    left, stop = early_stop_state(3, left, False)
    assert (left, stop) == (0, True), 'third non-improvement must stop at patience=3'


def test_explicit_flags_beat_the_dataset_defaults():
    a = opt.build_parser('xd').parse_args(['--save-threshold', '0.5', '--best-metric', 'AUC'])
    a = opt.apply_variant(a)
    a = opt.resolve_paths(a, dataset='xd')
    assert a.save_threshold == 0.5 and a.best_metric == 'AUC'


def test_threshold_default_still_excludes_the_measured_baseline():
    # The guard only means something while the target sits above what a real run
    # produced. This is the regression check on the number itself.
    assert opt.SAVE_THRESHOLD['xd'] > MEASURED_E5_BEST_AP, (
        f'xd threshold {opt.SAVE_THRESHOLD["xd"]} no longer excludes the measured '
        f'E5 best {MEASURED_E5_BEST_AP}, so the gate would save it')


def test_keyvals_is_stable_and_carries_the_result_affecting_knobs():
    a = opt.build_parser('xd').parse_args(['--variant', 'E1', '--seed', '7'])
    a = opt.apply_variant(a)
    a = opt.resolve_paths(a, dataset='xd')
    kv = config_keyvals(a)
    for key in ('variant=E1', 'seed=7', 'lr=', 'batch_size=', 'warmup_pct=',
                'early_stop_patience='):
        assert key in kv, (key, kv)
    assert config_keyvals(a) == kv, 'key ordering must be deterministic'


def test_log_header_states_threshold_and_command():
    a = opt.build_parser('xd').parse_args(['--variant', 'E5', '--seed', '234'])
    a = opt.apply_variant(a)
    a = opt.resolve_paths(a, dataset='xd')
    lines = []
    log_header(a, 'cpu', lines.append, 'xd_E5_s234', 'no warmup, milestones in epochs, stepped per epoch', 0.855)
    head = '\n'.join(lines)
    for token in ('[RunCommand]', '[Launch]', '[Threshold]', '[Config]',
                  'threshold=0.8550', 'dataset=XD', 'selection=test_fn', 'xd_E5_s234'):
        assert token in head, (token, head)
    # the header must precede any training output, so the first line is the command
    assert lines[0].startswith('[RunCommand]'), lines[0]
    # and it must say out loud that the recipe is upstream's, so a log that is later
    # read as "our improved baseline" can be spotted from the header alone
    assert 'unmodified' in head, head


def test_warmup_ramps_from_half_lr_to_base_then_fires_milestones_in_iterations():
    p = torch.nn.Parameter(torch.zeros(1))
    opt_ = torch.optim.AdamW([p], lr=BASE_LR)
    steps_per_epoch, epochs = 10, 5
    max_iter = steps_per_epoch * epochs
    milestones = [2, 4]                       # in epochs, as VadCLIP specifies
    sched = WarmupMultiStepLR(opt_, max_iter, [m * steps_per_epoch for m in milestones],
                              gamma=0.1, pct_start=0.3)
    lrs = []
    for _ in range(max_iter):
        lrs.append(opt_.param_groups[0]['lr'])
        sched.step()

    warm = int(0.3 * max_iter)
    assert warm == 15, warm
    # ramps from warmup_factor (0.5) up to the base lr, strictly increasing
    assert abs(lrs[0] - 0.5 * BASE_LR) < 1e-12, lrs[0]
    assert all(lrs[i] < lrs[i + 1] for i in range(warm - 1)), lrs[:warm]
    assert abs(lrs[warm] - BASE_LR) < 1e-12, lrs[warm]
    # never overshoots the base lr at any point
    assert max(lrs) <= BASE_LR + 1e-12, max(lrs)
    # milestones were converted to iterations, so the drop lands at 20 and 40
    assert abs(lrs[19] - BASE_LR) < 1e-12, lrs[19]
    assert abs(lrs[20] - BASE_LR * 0.1) < 1e-12, lrs[20]
    assert abs(lrs[40] - BASE_LR * 0.01) < 1e-12, lrs[40]
    assert abs(lrs[-1] - BASE_LR * 0.01) < 1e-12, lrs[-1]


def test_stopping_the_scheduler_still_decays_on_epoch_milestones():
    # The no-warmup path keeps epoch-unit milestones and is stepped once per epoch.
    # Stepping that scheduler per iteration instead would decay the lr ~200x too
    # fast and nothing would look wrong except the final number.
    p = torch.nn.Parameter(torch.zeros(1))
    opt_ = torch.optim.AdamW([p], lr=BASE_LR)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt_, [2, 4], 0.1)
    seen = []
    for _ in range(5):
        seen.append(opt_.param_groups[0]['lr'])
        sched.step()
    assert abs(seen[0] - BASE_LR) < 1e-12
    assert abs(seen[1] - BASE_LR) < 1e-12
    assert abs(seen[2] - BASE_LR * 0.1) < 1e-12, seen
    assert abs(seen[4] - BASE_LR * 0.01) < 1e-12, seen


def test_driver_skips_only_on_the_total_marker():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, 'proj')
        os.makedirs(os.path.join(root, 'runs'))
        summary = os.path.join(root, 'runs', 'results_summary.txt')
        assert ra.already_done('xd_E5_s234', root) is False
        # a history file alone must NOT count as done: that run may have died
        with open(os.path.join(root, 'runs', 'xd_E5_s234_history.json'), 'w') as f:
            f.write('{"best": 0.5}')
        assert ra.already_done('xd_E5_s234', root) is False
        with open(summary, 'w') as f:
            f.write('[Total] xd_E5_s235 metric=AP best=0.8 epoch=1 threshold=0.8550 '
                    'saved=no stopped_epoch=4 of 10 config=variant=E5\n')
        assert ra.already_done('xd_E5_s235', root) is True
        assert ra.already_done('xd_E5_s234', root) is False, 'must not match a prefix'


def test_lock_refuses_a_second_driver_and_survives_a_stale_one():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, 'proj')
        os.makedirs(os.path.join(root, 'runs'))
        lock = ra.acquire_lock(root)
        assert lock and os.path.exists(lock)
        # A second driver must be refused. It must also be refused *without killing
        # anything*: an earlier version probed liveness with os.kill(pid, 0), which
        # is a no-op on POSIX but calls TerminateProcess on Windows, so the probe
        # killed this very test runner with no traceback. Surviving the call is
        # part of what this asserts.
        assert ra.acquire_lock(root) is None, 'a held lock must block a second driver'
        assert os.path.exists(lock), 'the held lock must survive the refused attempt'

        with open(lock, 'w') as f:
            f.write('999999')          # a pid that is certainly not running
        if os.name == 'posix':
            assert ra.acquire_lock(root) is not None, 'a dead pid is reclaimable'
        else:
            # Liveness cannot be probed without killing the owner, so an unreadable
            # signal must not be read as "dead". Treating unknown as stale would let
            # a second driver steal a *live* lock, which is the exact failure the
            # guard exists to prevent.
            assert ra.acquire_lock(root) is None
        with open(lock, 'w') as f:
            f.write('999999')
        assert ra.acquire_lock(root, force_unlock=True) is not None
        assert os.path.exists(lock), 'force_unlock must leave a usable lock behind'


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f'  ok    {name}')
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, exc))
            print(f'  FAIL  {name}: {type(exc).__name__}: {exc}')
    print(f'\n{len(tests) - len(failed)}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
