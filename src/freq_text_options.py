"""Options + ablation variant table for Freq-Text Aug experiments.

Base hyper-parameters are VadCLIP's (see ``xd_option.py`` / ``ucf_option.py`` in
the reference repo) so E5 is a faithful VadCLIP baseline and E1-E4 differ from it
by the augmentation branch only.

The five variants are pure data: each maps to ``FreqTextAugmenter`` constructor
arguments plus ``aug_weight``. They are *not* separate code paths.

Known degeneracy, recorded up front so the table is not misread as five
independent hypotheses (see ``freq_text_aug`` module docstring, points 5-6):
in ``shift`` mode the injected update is exactly ``alpha * h(t) * d`` with
``h = irfft(gate)``. A frozen flat gate collapses ``h`` to a constant, and an
empty gate collapses it to zero - so E2 and E4 are the *same* update up to a
scalar, and both are close to "add a constant along d". E3 differs from E1 only
in ``d``. Read the table as: E1 the proposal, E5 the control, E3 the decisive
control, E2/E4 documented-but-weak controls.
"""

import argparse
import os

# ---------------------------------------------------------------------------
# ablation table
# ---------------------------------------------------------------------------
VARIANTS = {
    # E5: no augmentation. Reference point; everything is judged against it.
    'E5': dict(aug_weight=0.0),
    # E1: the proposal - frequency-domain gate + CLIP text direction.
    'E1': dict(aug_weight=1.0, aug_mode='shift', aug_freq_gate=True,
               aug_direction='text', aug_exclude_dc=True),
    # E3: decisive control - identical machinery, random fixed direction. If E1
    #     does not beat E3, the *text* is not what helps.
    'E3': dict(aug_weight=1.0, aug_mode='shift', aug_freq_gate=True,
               aug_direction='random', aug_exclude_dc=True),
    # E4: gate frozen flat -> h becomes a constant -> a pure DC offset along d.
    'E4': dict(aug_weight=1.0, aug_mode='shift', aug_freq_gate=False,
               aug_direction='text', aug_exclude_dc=False),
    # E2: the document's time-domain fallback (no spectral step at all).
    'E2': dict(aug_weight=1.0, aug_mode='timedomain', aug_freq_gate=False,
               aug_direction='text', aug_exclude_dc=False),
    # Extra: the non-degenerate spectral form (multiplicative band gain on the
    # anomaly-projection spectrum, data-dependent envelope). Recommended next step.
    'B1': dict(aug_weight=1.0, aug_mode='band', aug_freq_gate=True,
               aug_direction='text', aug_exclude_dc=False),
    # N1: the plain-noise control - iid Gaussian per (frame, dim), RMS-matched
    #     through the shared relative normalisation. Isolates whether the
    #     frequency-domain STRUCTURE (rank-1 direction, temporal envelope)
    #     matters vs structureless perturbation of the same magnitude.
    'N1': dict(aug_weight=1.0, aug_mode='noise', aug_freq_gate=False,
               aug_direction='random', aug_exclude_dc=True),
    # E3a: gate frozen at the designed bump spectrum -> separates "the frequency
    #      shape matters" from "learning that shape matters".
    'E3a': dict(aug_weight=1.0, aug_mode='shift', aug_freq_gate=True,
                aug_direction='random', aug_exclude_dc=True,
                aug_gate_learnable=False, aug_gate_random=False),
    # E3b: gate frozen at a fixed RANDOM spectrum of the same scale -> null
    #      model for the claim that the frequency gate is a real mechanism.
    'E3b': dict(aug_weight=1.0, aug_mode='shift', aug_freq_gate=True,
                aug_direction='random', aug_exclude_dc=True,
                aug_gate_learnable=False, aug_gate_random=True),
}

VARIANT_HELP = {
    'E5': 'baseline, no augmentation (VadCLIP as published)',
    'E1': 'frequency gate + CLIP text direction (the proposal)',
    'E2': 'time-domain text interpolation, no spectral step (degenerate, see docstring)',
    'E3': 'frequency gate + fixed random direction (the control that matters)',
    'E4': 'gate frozen flat -> constant offset along d (degenerate)',
    'B1': 'multiplicative band gain on the anomaly-projection spectrum',
    'N1': 'plain iid-noise control, RMS-matched (no direction, no spectral structure)',
    'E3a': 'frozen structured gate, random direction (isolates gate adaptivity)',
    'E3b': 'frozen random spectrum, random direction (null model for the gate)',
}


# Neutral defaults for every augmentation knob. The variant table overrides a
# subset; anything the user passes on the command line overrides both.
AUG_DEFAULTS = dict(
    aug_weight=0.0,
    aug_mode='shift',
    aug_freq_gate=True,
    aug_direction='text',
    aug_exclude_dc=True,
    aug_delta_norm='relative',
    aug_alpha_lo=0.1,
    aug_alpha_hi=0.5,
    aug_alpha=0.0,
    aug_ratio=0.2,
    aug_target=0.9,
    aug_block_len=51,
    aug_gate_bands=3,
    aug_band_gain=1.0,
    aug_gate_learnable=True,
    aug_gate_random=False,
)

_BOOL = lambda s: str(s).lower() in ('1', 'true', 'yes', 'y', 'on')


def add_augment_arguments(parser):
    """Every knob defaults to ``None`` = "unspecified", so the variant table can
    tell an omitted flag apart from a deliberate override."""
    g = parser.add_argument_group('freq-text augmentation')
    g.add_argument('--variant', default='E5', choices=sorted(VARIANTS),
                   help='ablation id: ' + '; '.join(f'{k}={VARIANT_HELP[k]}' for k in sorted(VARIANTS)))
    g.add_argument('--aug-weight', default=None, type=float,
                   help='0 disables the branch; 1.0 = the document weight')
    g.add_argument('--aug-mode', default=None, choices=['shift', 'band', 'timedomain'],
                   help="shift = the document's additive spectrum; band = "
                        'multiplicative band gain; timedomain = constant offset')
    g.add_argument('--aug-freq-gate', default=None, type=_BOOL,
                   help='False freezes the gate at its neutral value')
    g.add_argument('--aug-direction', default=None, choices=['text', 'random'],
                   help="'random' substitutes a fixed unit Gaussian vector (E3)")
    g.add_argument('--aug-exclude-dc', default=None, type=_BOOL,
                   help='subtract the gate mean so E1 is not a scaled copy of E2')
    g.add_argument('--aug-delta-norm', default=None, choices=['relative', 'absolute'],
                   help='relative makes alpha a fraction of the feature RMS so E1/E3 '
                        'perturb equally (a unit CLIP direction is ~20x smaller than '
                        'a CLIP feature, which would otherwise confound the ablation)')
    g.add_argument('--aug-alpha-lo', default=None, type=float)
    g.add_argument('--aug-alpha-hi', default=None, type=float)
    g.add_argument('--aug-alpha', default=None, type=float,
                   help='>0 pins alpha instead of sampling in [lo, hi]')
    g.add_argument('--aug-ratio', default=None, type=float,
                   help='fraction of lowest-score frames synthesised')
    g.add_argument('--aug-target', default=None, type=float,
                   help='BCE target probability for synthesised frames')
    g.add_argument('--aug-block-len', default=None, type=int,
                   help='K, the basis block length (document uses T//5)')
    g.add_argument('--aug-gate-bands', default=None, type=int,
                   help='number of Gaussian bumps initialising the band gate')
    g.add_argument('--aug-band-gain', default=None, type=float)
    g.add_argument('--aug-gate-learnable', default=None, type=_BOOL,
                   help='False freezes the gate at its initial spectrum '
                        '(E3a/E3b: structure without adaptivity)')
    g.add_argument('--aug-gate-random', default=None, type=_BOOL,
                   help='True initialises the gate from fixed random noise instead '
                        'of Gaussian bumps (E3b null model)')
    return parser


# Publication thresholds, keyed by dataset. The project targets are deliberately
# stricter than the skill's defaults (XD AP 0.85 / UCF AUC 0.87): 0.855 / 0.875.
# A checkpoint is written only when the selection metric is BOTH strictly better
# than the previous best AND above this number, so a run that never reaches the
# target leaves no weight behind instead of a 600 MB file that cannot be reported.
SAVE_THRESHOLD = {'xd': 0.855, 'ucf': 0.875, 'union': 0.80}
BEST_METRIC = {'xd': 'AP', 'ucf': 'AUC', 'union': 'AP'}


def _add_reporting_arguments(p, dataset: str):
    """Checkpointing, early stopping and LR warmup.

    All of these are result-affecting, so they are CLI args with dataset-aware
    defaults rather than constants inside the training loop.
    """
    g = p.add_argument_group('reporting / optimisation controls')
    g.add_argument('--save-threshold', default=None, type=float,
                   help=f'minimum selection metric required before any checkpoint is '
                        f'written; default {SAVE_THRESHOLD[dataset]} for {dataset.upper()}')
    g.add_argument('--best-metric', default=None, choices=['AP', 'AUC', 'test_fn'],
                   help='which metric selects the best checkpoint. DEFAULT \'test_fn\' '
                        '= whatever the upstream test loop returns (AP2 for XD, AUC1 '
                        'for UCF), which is the convention the published numbers were '
                        'produced under. Only pass AP/AUC to switch to the '
                        'max(AP1,AP2) dual-head rule, which changes which epoch is '
                        'reported and therefore breaks comparability with upstream.')
    g.add_argument('--early-stop-patience', default=0, type=int,
                   help='consecutive evals without improvement before stopping. '
                        'DEFAULT 0 = disabled, so a baseline run trains the full '
                        'upstream schedule; pass 2-3 to iterate, 5+ for a final run. '
                        'Enabling it changes which epoch is reported.')
    g.add_argument('--early-stop-min-delta', default=0.0, type=float,
                   help='an improvement smaller than this does not reset the patience counter')
    g.add_argument('--warmup-pct', default=0.0, type=float,
                   help='fraction of total steps spent ramping the learning rate up. '
                        'DEFAULT 0 = off, which keeps the training trajectory identical '
                        'to the upstream VadCLIP code. E5 is supposed to BE VadCLIP, so '
                        'any deviation from the upstream recipe must be an explicit '
                        'opt-in flag with a reason, never a default. '
                        'utils/lr_warmup.WarmupMultiStepLR is vendored but referenced '
                        'by nothing in src/; whether upstream used it has NOT been '
                        'verified against the authors\' repository, so it is off until '
                        'someone checks.')
    return p


def apply_variant(args):
    """Resolve the augmentation config. Precedence: CLI flag > variant > default."""
    name = args.variant
    if name not in VARIANTS:
        raise KeyError(f'unknown variant {name!r}, expected one of {sorted(VARIANTS)}')
    override = VARIANTS[name]
    for key, neutral in AUG_DEFAULTS.items():
        current = getattr(args, key, None)
        if current is None:
            setattr(args, key, override.get(key, neutral))
    if args.aug_alpha_lo >= args.aug_alpha_hi:
        raise ValueError(f'aug_alpha_lo ({args.aug_alpha_lo}) must be < '
                         f'aug_alpha_hi ({args.aug_alpha_hi})')
    args.variant = name
    return args


def build_parser(dataset: str) -> argparse.ArgumentParser:
    """VadCLIP base options for ``dataset`` plus the augmentation group.

    ``--dataset`` is deliberately *not* an option: the entrypoint (``xd_train.py``,
    ``ucf_train.py``, ``xd_test.py``, ``ucf_test.py``) fixes it, which is what keeps
    the list files, gt arrays, hyper-parameters and output dir mutually consistent.
    Passing ``--dataset`` is an argparse error rather than a silent override.
    """
    p = argparse.ArgumentParser(description=f'Freq-Text Aug ({dataset})')
    p.set_defaults(dataset=dataset)
    p.add_argument('--seed', default=234, type=int)

    p.add_argument('--embed-dim', default=512, type=int)
    p.add_argument('--visual-length', default=256, type=int)
    p.add_argument('--visual-width', default=512, type=int)
    p.add_argument('--visual-head', default=1, type=int)
    p.add_argument('--attn-window', default=64 if dataset in ('xd', 'union') else 8, type=int)
    p.add_argument('--visual-layers', default=1 if dataset in ('xd', 'union') else 2, type=int)
    p.add_argument('--prompt-prefix', default=10, type=int)
    p.add_argument('--prompt-postfix', default=10, type=int)
    p.add_argument('--classes-num', default={'xd': 7, 'union': 15}.get(dataset, 14), type=int)
    p.add_argument('--clip-arch', default='ViT-B-16',
                   help='architecture used for the text direction table; keep it '
                        'equal to the ViT-B/16 that CLIPVAD loads internally')
    p.add_argument('--loss3-scale', default=1e-4 if dataset in ('xd', 'union') else 1e-1,
                   type=float,
                   help='prompt dispersion weight (VadCLIP: 1e-4 XD, /13*1e-1 UCF)')

    p.add_argument('--max-epoch', default=10, type=int)
    p.add_argument('--batch-size', default=96 if dataset in ('xd', 'union') else 64, type=int)
    p.add_argument('--lr', default=1e-5 if dataset in ('xd', 'union') else 2e-5, type=float)
    p.add_argument('--scheduler-rate', default=0.1, type=float)
    p.add_argument('--scheduler-milestones',
                   default=[3, 6, 10] if dataset in ('xd', 'union') else [4, 8],
                   type=int, nargs='+')

    # union monitors checkpoints on the XD test set (source-domain validation:
    # selecting on a cross-domain target would be target leakage)
    suffix = {'xd': 'xd', 'union': 'union'}.get(dataset, 'ucf')
    p.add_argument('--train-list', default=f'list/{suffix}_CLIP_rgb.csv')
    p.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv'
                   if dataset == 'union' else f'list/{suffix}_CLIP_rgbtest.csv')
    p.add_argument('--gt-path', default='list/gt.npy' if dataset in ('xd', 'union')
                   else 'list/gt_ucf.npy')
    p.add_argument('--gt-segment-path', default='list/gt_segment.npy' if dataset in ('xd', 'union')
                   else 'list/gt_segment_ucf.npy')
    p.add_argument('--gt-label-path', default='list/gt_label.npy' if dataset in ('xd', 'union')
                   else 'list/gt_label_ucf.npy')

    p.add_argument('--run-name', default=None,
                   help='output subdir under runs/; defaults to <dataset>_<variant>')
    p.add_argument('--log-dir', default=None)
    p.add_argument('--model-path', default=None)
    p.add_argument('--checkpoint-path', default=None)
    p.add_argument('--use-checkpoint', default=False,
                   type=lambda s: s.lower() != 'false')
    p.add_argument('--save-resume-checkpoint', action='store_true',
                   help='write checkpoint.pth every epoch so the run can be resumed. '
                        'Off by default: the file carries the optimizer state and is '
                        '~700 MB, so a 10-epoch run would write ~7 GB that nothing '
                        'reads unless you also pass --use-checkpoint.')
    p.add_argument('--workers', default=4, type=int)
    p.add_argument('--log-every', default=50, type=int)
    p.add_argument('--max-train-steps', default=0, type=int,
                   help='>0 truncates every epoch (smoke runs)')
    p.add_argument('--max-test-clips', default=0, type=int,
                   help='>0 truncates the test loader (smoke runs; the metric is '
                        'then meaningless, only the plumbing is validated)')

    add_augment_arguments(p)
    _add_reporting_arguments(p, dataset)
    return p


def repo_root() -> str:
    """The project root (the directory holding ``list/``), inferred from this file.

    VadCLIP keeps its defaults relative (``list/xd_CLIP_rgb.csv``) while the
    convention is to *run* from ``src/``, so the same string is only correct when
    the process starts in the repo root. Deriving the root from ``__file__`` makes
    every default correct regardless of the caller's cwd.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_data_path(path: str) -> str:
    """Resolve a data path against cwd first, then the project root.

    Absolute paths and paths that already exist relative to cwd pass through
    untouched, so an explicit ``--train-list`` always wins.
    """
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(repo_root(), path)
    return candidate if os.path.exists(candidate) else path


def resolve_paths(args, dataset: str = None):
    """Place all artefacts of a variant in their own directory - never shared.

    ``dataset`` re-asserts the entrypoint's dataset: ``build_parser`` already picked
    the XD or UCF defaults from it, so a stray ``--dataset`` flag would otherwise
    train an XD model on UCF lists with UCF hyper-parameters.
    """
    if dataset is not None:
        args.dataset = dataset
    if getattr(args, 'save_threshold', None) is None:
        args.save_threshold = SAVE_THRESHOLD[args.dataset]
    if getattr(args, 'best_metric', None) is None:
        args.best_metric = 'test_fn'      # upstream's own selection; see the flag help
    for key in ('train_list', 'test_list', 'gt_path', 'gt_segment_path', 'gt_label_path'):
        setattr(args, key, resolve_data_path(getattr(args, key)))
    if args.run_name is None:
        args.run_name = f'{args.dataset}_{args.variant}'
    root = os.path.join(repo_root(), 'runs', args.run_name)
    if args.log_dir is None:
        args.log_dir = root
    if args.model_path is None:
        args.model_path = os.path.join(root, 'model_best.pth')
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(root, 'checkpoint.pth')
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    return args


# Development-machine locations of the released VadCLIP checkpoints, kept only
# as the last fallback -- see official_checkpoints() for the resolution order.
_DEV_OFFICIAL_CHECKPOINTS = {
    'xd': [r'E:\program\VadCLIP-main\data\model_xd.pth',
           r'E:\program\referCode\VadCLIP-main\data\model_xd.pth'],
    'ucf': [r'E:\program\VadCLIP-main\data\model_ucf.pth',
            r'E:\program\referCode\VadCLIP-main\data\model_ucf.pth'],
}


def env_path(*names: str):
    """First non-empty environment variable among ``names``, else ``None``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def data_root() -> str:
    """Directory holding the CLIP feature banks (``XDTrainClipFeatures`` etc.)."""
    return env_path('RSI_DATA_ROOT', 'FGDA_DATA_ROOT') or 'E:/dataset'


def sht_root() -> str:
    """Directory holding the raw ShanghaiTech frames."""
    return (env_path('RSI_SHT_ROOT', 'FGDA_SHT_ROOT')
            or r'E:\dataset\shanghaitech')


def sht_feat_root() -> str:
    """Directory holding the extracted ShanghaiTech CLIP features."""
    return (env_path('RSI_SHT_FEAT_ROOT', 'FGDA_SHT_FEAT_ROOT')
            or os.path.join(data_root(), 'SHTClipFeatures'))


def official_checkpoints(source: str):
    """Ordered candidate paths for the released VadCLIP ``source`` checkpoint.

    ``RSI_VADCLIP_<SOURCE>`` -> ``FGDA_VADCLIP_<SOURCE>`` (the pre-rename name,
    still honoured) -> repo-local ``checkpoints/model_<source>.pth`` -> the
    development-machine paths in ``_DEV_OFFICIAL_CHECKPOINTS``.
    """
    override = env_path(f'RSI_VADCLIP_{source.upper()}',
                        f'FGDA_VADCLIP_{source.upper()}')
    if override:
        return [override]
    return [os.path.join(repo_root(), 'checkpoints',
                         f'model_{source}.pth')] + _DEV_OFFICIAL_CHECKPOINTS[source]
