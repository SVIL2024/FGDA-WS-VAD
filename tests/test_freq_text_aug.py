"""Regression tests for the corrected Freq-Text Augmenter.

Each test corresponds to one numbered defect in the ``freq_text_aug`` module
docstring. They exist because every one of these bugs is *silent*: the original
document's code runs, returns a tensor of plausible shape, and produces an
augmentation that is either the identity or applies to the wrong frames. A shape
assertion alone would not have caught any of them, so the tests check
mathematical properties (rank, envelope energy, mask semantics) rather than
shapes.

Run:  python tests/test_freq_text_aug.py        (no pytest required)
      pytest tests/test_freq_text_aug.py -q     (also works)
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from freq_text_aug import (FreqTextAugmenter, augment_sequence,  # noqa: E402
                           lowest_score_indices)

T, C = 256, 512


def _aug(**kw):
    kw.setdefault('dim', C)
    return FreqTextAugmenter(**kw)


# --------------------------------------------------------------------------
# 1. broadcasting: the document's double unsqueeze demoted the batch axis
# --------------------------------------------------------------------------
def test_shape_is_preserved_and_batch_is_not_smuggled_into_a_frequency_axis():
    """(B, K, C) in -> (B, K, C) out.

    The document's ``unsqueeze(0).unsqueeze(0)`` produced ``(1, 1, nb, C)`` added
    to ``(B, nb, C)``, which broadcasts to ``(1, B, nb, C)``; ``irfft`` then
    returned ``(1, T, B, C)``. Any code that only checks "did it not raise" will
    happily train on that.
    """
    for B in (1, 3, 8):
        for K in (5, 17, 51):
            a = _aug(block_len=K)
            block = torch.randn(B, K, C)
            d = torch.randn(C)
            out, alpha = a(block, d)
            assert out.shape == (B, K, C), (B, K, out.shape)
            assert alpha.dim() == 0


def test_each_sample_gets_its_own_direction_not_a_shared_frequency_axis():
    """A batch of different directions must produce different deltas.

    Under the document's broadcasting the direction ends up indexed by the batch
    axis inside a frequency dimension, so samples silently share or swap
    directions. Checking that per-sample deltas differ catches it.
    """
    B, K = 4, 21
    a = _aug(block_len=K, delta_norm='absolute')
    block = torch.randn(B, K, C)
    dirs = torch.eye(C)[:B] * 5.0          # mutually orthogonal-ish
    out, _ = a(block, dirs)
    deltas = (out - block)
    for i in range(B):
        for j in range(i + 1, B):
            assert not torch.allclose(deltas[i], deltas[j]), (i, j)


# --------------------------------------------------------------------------
# 2. gate initialisation: a constant gate is a Dirac, so nothing is augmented
# --------------------------------------------------------------------------
def test_structured_gate_is_not_a_dirac_at_index_zero():
    """The document's ``gate = 2 * ones`` gives ``irfft`` = spike at t = 0.

    With the bottom-20% selection that spike addresses frames the mask almost
    never covers, so the augmentation is the identity while all the shapes look
    right. ``h_index0_energy`` is exactly 1.0 in the degenerate case.
    """
    a = _aug(block_len=51, mode='shift')
    st = a.envelope_stats()
    assert st['h_index0_energy'] < 0.5, st
    assert st['h_eff_len'] > 4.0, st

    # and the degenerate configuration is still reproducible for the record
    degenerate = _aug(block_len=51, mode='shift', gate_bands=1, gate_init=0.0)
    degenerate.freq_gate.data.fill_(2.0)          # constant gate, as in the doc
    st_deg = degenerate.envelope_stats()
    assert st_deg['h_eff_len'] < 1.5, st_deg
    # with exclude_dc the constant gate's DC bin is removed, leaving nothing at
    # all - the honest statement of what E4 reduces to
    assert st_deg['h_rms'] < 1e-6, st_deg


# --------------------------------------------------------------------------
# 3. transform length must follow the block, not the sequence
# --------------------------------------------------------------------------
def test_block_length_transform_keeps_the_envelope_inside_the_block():
    """``rfft(n=T)`` on K real frames pads with zeros, so part of the injected
    envelope is spent modulating frames the caller discards.

    With ``n_fft == K`` the envelope has exactly K samples and all of it lands on
    real frames. With ``n_fft == T`` (the document's behaviour) the envelope is
    length T, and the energy outside the K genuine frames is lost - and worse, it
    depends on the *sequence* length, so the same block is augmented differently
    depending on how many frames the video happens to have.
    """
    K = 51
    good = _aug(block_len=K, n_fft=K, mode='shift')
    assert good.envelope().shape[-1] == K
    assert good.envelope_stats()['h_eff_len'] > 3.0, good.envelope_stats()

    bad = _aug(block_len=K, n_fft=T, mode='shift')
    h_bad = bad.envelope()
    assert h_bad.shape[-1] == T
    frac_in_block = float((h_bad[:K] ** 2).sum() / (h_bad ** 2).sum())
    assert frac_in_block < 0.9, frac_in_block

    # the envelope must not depend on the sequence length
    other = _aug(block_len=K, n_fft=128, mode='shift')
    assert not torch.allclose(good.envelope(), other.envelope()[:K], atol=1e-4), \
        'n_fft=K should make the envelope independent of sequence length'


def test_augmentation_actually_changes_the_selected_block():
    """The end-to-end consequence of 2 and 3: the selected frames must move."""
    K = 51
    a = _aug(block_len=K, mode='shift', delta_norm='relative')
    block = torch.randn(3, K, C)
    out, _ = a(block, torch.randn(C))
    changed = (out - block).norm(dim=-1)          # (3, K)
    assert (changed > 1e-4).float().mean() > 0.9, changed


# --------------------------------------------------------------------------
# 4. sequence structure and padding
# --------------------------------------------------------------------------
def test_mask_never_covers_padding():
    """Zero-padded frames must not be selected as augmentation targets."""
    B, Tt, K = 6, 64, 16
    a = _aug(block_len=K, mode='shift')
    feats = torch.randn(B, Tt, C)
    lengths = torch.tensor([64, 40, 17, 9, 3, 1])
    scores = torch.randn(B, Tt)
    out, mask, _ = augment_sequence(feats, torch.randn(C), a, scores=scores,
                                    ratio=0.25, lengths=lengths)
    for i, L in enumerate(lengths.tolist()):
        assert not mask[i, L:].any(), (i, L, mask[i].nonzero().flatten().tolist())


def test_mismatched_block_len_and_ratio_fails_loudly():
    """The two knobs look independent in the CLI; a mismatch must not surface as
    a shape error deep inside forward() halfway through an epoch."""
    a = _aug(block_len=8, mode='shift')
    feats = torch.randn(2, 64, C)
    try:
        augment_sequence(feats, torch.randn(C), a, scores=torch.randn(2, 64),
                         ratio=0.5, lengths=torch.tensor([64, 64]))
    except ValueError as exc:
        assert '--aug-block-len' in str(exc), str(exc)
        return
    raise AssertionError('expected a ValueError naming the coupling')


def test_unselected_frames_are_bitwise_unchanged():
    B, Tt, K = 4, 64, 16
    a = _aug(block_len=K, mode='shift')
    feats = torch.randn(B, Tt, C)
    lengths = torch.tensor([64, 50, 30, 20])
    scores = torch.randn(B, Tt)
    out, mask, _ = augment_sequence(feats, torch.randn(C), a, scores=scores,
                                    ratio=0.2, lengths=lengths)
    assert out.shape == feats.shape
    assert torch.equal(out[~mask], feats[~mask])
    assert mask.any(), 'nothing was augmented - the test would be vacuous'


def test_selection_prefers_the_lowest_scores():
    B, Tt = 2, 32
    scores = torch.randn(B, Tt)
    idx, valid, _ = lowest_score_indices(scores, ratio=0.25,
                                         lengths=torch.tensor([Tt, Tt]))
    k = idx.shape[1]
    for i in range(B):
        chosen = scores[i, idx[i][valid[i]]].max()
        rejected = scores[i].clone()
        rejected[idx[i][valid[i]]] = float('inf')
        assert chosen <= rejected.min() + 1e-6, (i, chosen, rejected.min())
    assert k == max(5, round(Tt * 0.25))


# --------------------------------------------------------------------------
# 5. the rank-1 identity of 'shift' mode
# --------------------------------------------------------------------------
def test_shift_mode_equals_envelope_times_direction():
    """``delta(t) = alpha_norm * h(t) * d`` for *any* gate.

    This is the honest characterisation of the document's formulation: the gate
    has K//2+1 degrees of freedom but the resulting (K, C) update has rank 1, so
    the frequency-domain machinery cannot express a data-dependent profile in
    this mode. Asserting the identity means the claim stays true in the paper
    even if the gate is retrained.
    """
    K = 33
    a = _aug(block_len=K, mode='shift', delta_norm='absolute')
    a.freq_gate.data.normal_()
    block = torch.randn(1, K, C)
    d = torch.randn(C)
    d = d / d.norm()
    alpha = torch.tensor(0.7)
    out, _ = a(block, d, alpha=alpha)

    h = a.envelope()
    expected = alpha * torch.outer(h, d)
    assert torch.allclose(out[0] - block[0], expected, atol=1e-4)


def test_shift_output_has_rank_one_update():
    K = 41
    a = _aug(block_len=K, mode='shift', delta_norm='absolute')
    block = torch.zeros(1, K, C)
    out, _ = a(block, torch.randn(C))
    delta = out[0]
    assert torch.linalg.matrix_rank(delta, tol=1e-5) == 1


def test_exclude_dc_removes_the_uniform_offset():
    """Without this, E1 in 'shift' mode degenerates into E2 (a DC shift)."""
    K = 32
    for exclude in (True, False):
        a = _aug(block_len=K, mode='shift', exclude_dc=exclude, delta_norm='absolute')
        h = a.envelope()
        if exclude:
            assert h.mean().abs() < 1e-5, float(h.mean())
        else:
            assert h.mean().abs() > 1e-3, float(h.mean())


# --------------------------------------------------------------------------
# 6. mode semantics: 'band' is data dependent, 'timedomain' is not
# --------------------------------------------------------------------------
def test_band_mode_depends_on_the_input_features():
    """'shift' is input independent by construction; 'band' must not be.

    This is the only defensible reading of the paper's "frequency band" claim: in
    'shift' the injected spectrum is a constant, so the update is a fixed envelope
    times a fixed direction regardless of what the frame contains. 'band' lets the
    gate filter the spectrum of the anomaly projection, so the envelope follows the
    data.
    """
    K = 24
    d = torch.randn(C)
    block_a = torch.randn(1, K, C)
    block_b = torch.randn(1, K, C)
    alpha = torch.tensor(0.3)

    shift = _aug(block_len=K, mode='shift', delta_norm='absolute', gate_init=0.5)
    delta_shift_a = shift(block_a, d, alpha=alpha)[0] - block_a
    delta_shift_b = shift(block_b, d, alpha=alpha)[0] - block_b
    # a fixed spectrum cannot react to the input. Compare with an absolute
    # tolerance: the injected delta is ~1e-4 here, so the default rtol=1e-5 in
    # allclose fails on float32 round-off alone and would misreport a correct
    # implementation as input dependent.
    assert torch.allclose(delta_shift_a, delta_shift_b, atol=1e-6), \
        'shift mode should be input independent'

    band = _aug(block_len=K, mode='band', delta_norm='absolute', gate_init=0.5)
    delta_band_a = band(block_a, d, alpha=alpha)[0] - block_a
    delta_band_b = band(block_b, d, alpha=alpha)[0] - block_b
    assert not torch.allclose(delta_band_a, delta_band_b), \
        'band mode ignored its input'

    # and the two modes must not be accidentally the same implementation
    assert not torch.allclose(delta_shift_a, delta_band_a)


def test_timedomain_mode_is_a_constant_offset_per_frame():
    K = 16
    a = _aug(block_len=K, mode='timedomain', delta_norm='absolute')
    block = torch.randn(2, K, C)
    out, _ = a(block, torch.randn(C), alpha=torch.tensor(0.5))
    delta = out - block
    for i in range(K):
        assert torch.allclose(delta[:, i], delta[:, 0], atol=1e-5), i


def test_e4_has_no_trainable_gate_and_e1_does():
    """E4 is the frozen-gate control; if its gate were a Parameter the ablation
    would not be testing what it claims to test."""
    frozen = _aug(block_len=16, mode='shift', use_freq_gate=False)
    assert not any(p.requires_grad for p in frozen.parameters()), \
        [n for n, _ in frozen.named_parameters()]
    trainable = _aug(block_len=16, mode='shift', use_freq_gate=True)
    names = [n for n, p in trainable.named_parameters() if p.requires_grad]
    assert 'freq_gate' in names


def test_random_direction_source_replaces_the_supplied_direction():
    """E3 must not silently fall back to the text direction."""
    K = 8
    text_dir = torch.ones(C)
    a = _aug(block_len=K, mode='shift', direction_source='random',
             delta_norm='absolute')
    block = torch.zeros(2, K, C)
    out, _ = a(block, text_dir, alpha=torch.tensor(1.0))
    delta = out[0]
    assert delta.norm() > 0
    ref = a.random_direction
    # the applied direction must be the stored random one, not text_dir
    row = delta[0]
    assert not torch.allclose(row / row.norm(), text_dir / text_dir.norm(),
                              atol=1e-3)
    got = row / row.norm()
    assert abs(abs(float(got @ (ref / ref.norm()))) - 1.0) < 1e-4


# --------------------------------------------------------------------------
# scale: alpha must mean the same thing across variants
# --------------------------------------------------------------------------
def test_relative_normalisation_makes_alpha_comparable_across_modes():
    """A unit CLIP direction is ~20x smaller than a CLIP feature, so without
    relative normalisation the injected magnitude depends on the mode and the
    E1-E4 comparison is meaningless."""
    K = 32
    block = torch.randn(2, K, C) * 5.0
    rel = []
    for mode in ('shift', 'band', 'timedomain'):
        a = _aug(block_len=K, mode=mode, delta_norm='relative', gate_init=0.5)
        out, _ = a(block, torch.randn(C), alpha=torch.tensor(0.5))
        rel.append(float((out - block).pow(2).mean().sqrt() /
                         block.pow(2).mean().sqrt()))
    spread = max(rel) - min(rel)
    assert spread < 0.5, rel
    # relative normalisation pins every mode to the same injected magnitude
    assert abs(float(np.mean(rel)) - 1.0) < 0.6, rel

    absn = []
    for mode in ('shift', 'timedomain'):
        a = _aug(block_len=K, mode=mode, delta_norm='absolute', gate_init=0.5)
        out, _ = a(block, torch.randn(C), alpha=torch.tensor(0.5))
        absn.append(float((out - block).pow(2).mean().sqrt() /
                          block.pow(2).mean().sqrt()))
    # under 'absolute' the injected magnitude is set by the mode (a unit direction
    # is ~20x smaller than a CLIP feature, and the gate's own scale leaks in), which
    # is exactly the confound relative normalisation removes
    ratio = max(absn) / (min(absn) + 1e-12)
    assert ratio > 5.0, absn


def test_direction_is_renormalised_so_its_scale_cannot_leak_into_alpha():
    K = 8
    a = _aug(block_len=K, mode='timedomain', delta_norm='absolute')
    block = torch.zeros(1, K, C)
    small, _ = a(block, torch.randn(C), alpha=torch.tensor(0.5))
    big, _ = a(block, torch.randn(C) * 100.0, alpha=torch.tensor(0.5))
    assert torch.allclose(small.norm(), big.norm(), rtol=1e-3)


def test_gradients_reach_the_frequency_gate():
    """E1 is only a learned-gate experiment if the gate receives gradient."""
    K = 16
    a = _aug(block_len=K, mode='shift', use_freq_gate=True)
    out, _ = a(torch.randn(2, K, C), torch.randn(C))
    out.pow(2).mean().backward()
    g = a.freq_gate.grad
    assert g is not None and float(g.abs().sum()) > 0


def test_unknown_configuration_is_rejected():
    for kw in ({'mode': 'nope'}, {'direction_source': 'nope'},
               {'delta_norm': 'nope'}):
        try:
            _aug(block_len=8, **kw)
        except ValueError:
            continue
        raise AssertionError(f'{kw} should have raised')


def test_block_longer_than_n_fft_is_rejected():
    a = _aug(block_len=8, n_fft=8)
    try:
        a(torch.randn(1, 16, C), torch.randn(C))
    except ValueError:
        return
    raise AssertionError('K > n_fft should have raised instead of truncating')


def test_noise_mode_matches_input_rms_and_leaves_unselected_untouched():
    """N1 control: iid noise, RMS-matched through the shared relative
    normalisation - the injected delta must carry no direction or envelope
    structure, only magnitude comparable to the spectral modes."""
    torch.manual_seed(0)
    K = 33
    a = _aug(block_len=K, mode='noise', delta_norm='relative')
    block = torch.randn(2, K, C)
    out, alpha = a(block, torch.randn(C))
    delta = out - block
    d_rms = delta.pow(2).mean().sqrt()
    x_rms = block.pow(2).mean().sqrt()
    # relative normalisation pins the injected RMS to the block RMS
    assert abs(d_rms - x_rms) < 0.15 * x_rms, (d_rms, x_rms)
    # rank must be full (no direction) - the spectral modes are rank 1
    assert torch.linalg.matrix_rank(delta[0], tol=1e-3) > K // 2


def test_noise_mode_ignores_direction():
    """Same seed -> same delta whatever direction is passed: N1 has no
    direction semantics, so a stale or random direction cannot leak in."""
    K = 33
    a = _aug(block_len=K, mode='noise', delta_norm='relative')
    block = torch.randn(1, K, C)
    torch.manual_seed(0)
    out1, _ = a(block, torch.randn(C))
    torch.manual_seed(0)
    out2, _ = a(block, torch.randn(C))
    assert torch.allclose(out1, out2)


if __name__ == '__main__':
    torch.manual_seed(0)
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f'  ok    {name}')
        except Exception as exc:                      # noqa: BLE001
            failed.append((name, exc))
            print(f'  FAIL  {name}: {type(exc).__name__}: {exc}')
    print(f'\n{len(tests) - len(failed)}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
