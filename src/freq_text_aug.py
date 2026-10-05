"""Freq-Text Aug: frequency-domain, text-guided pseudo-anomaly synthesis.

Corrected implementation of the ``FreqTextAugmenter`` in
``Freq-Text_Aug_最小可行版_频域文本引导异常样本合成方案.md``. Every deviation from
the document is a bug fix, and each one is asserted in
``tests/test_freq_text_aug.py``.

1. **Broadcasting bug (silent shape error).** The document computes
   ``freq_shift.unsqueeze(0).unsqueeze(0)`` -> ``(1, 1, 129, 512)`` and adds it to
   ``normal_freq`` of shape ``(B, 129, 512)``. Broadcasting *demotes the batch
   axis into a frequency axis*: the sum becomes ``(1, B, 129, 512)`` and the
   ``irfft`` returns ``(1, T, B, 512)`` instead of ``(B, T, 512)``. One
   ``unsqueeze`` is correct, not two.

2. **The gate initialisation silently disables the augmentation.**
   ``freq_gate = 2.0 * ones(129)`` is constant across bands, and the inverse
   transform of a constant spectrum is a Dirac: ``irfft(c * ones, n=T)`` is
   ``c`` at ``t = 0`` and ``0`` everywhere else. So the document's delta is
   non-zero only at frame 0, which the bottom-20%-score selection almost never
   contains - the augmenter returns its input unchanged and E1/E2/E3/E4 all
   collapse onto E5. Measured as ``h_frame0_energy == 1.0``.
   Fix: ``gate_init_std > 0`` perturbs the gate so the envelope is band-shaped,
   and the transform length is tied to the block length (point 3) so the
   envelope spreads over the selected frames instead of frame 0.

3. **Zero-padding K frames to T before ``rfft`` puts the modulation in the wrong
   place.** The document gathers K = T/5 frames and calls ``rfft(n=T)``, i.e. it
   transforms ``[K frames | 205 zeros]``. A spectrum whose phase is all-zero
   (real, non-negative gate) is by construction concentrated at index 0 of that
   padded window, so the injected bump lands outside the K real frames. Here the
   augmenter transforms the K-frame block at its own length (``n_fft = K``) and a
   separate helper scatters the result back into the ``(B, T, C)`` sequence, so
   every selected frame is actually modified. ``n_fft = T`` is still selectable
   purely to reproduce and demonstrate the document's behaviour.

4. **Temporal structure must survive.** The document flattens the output to
   ``(M*K, 1, C)`` and feeds that to the model, which expects ``(B, 256, C)`` and
   runs a transformer + GCN + attention-MIL pooling over the sequence. Here the
   augmented tensor stays ``(B, T, C)`` and is consumed by one extra forward pass.

5. **The rank-1 degeneracy is stated, not hidden.** For *any* gate the additive
   formulation collapses to

       delta(t) = alpha * h(t) * d,     h = irfft_K(gate),  d = anomaly direction

   a single scalar temporal envelope times a single fixed feature-space
   direction; the gate has ``K//2+1`` degrees of freedom but the resulting
   ``(K, C)`` update has rank 1. Two modes therefore exist:

   * ``mode='shift'``  - the document's additive formulation, honestly labelled
     rank-1 (reproduces E1-E5 as specified).
   * ``mode='band'``   - the gate multiplies the *spectrum of the anomaly
     projection* ``a(t) = <x(t), d>``, so the synthesised envelope is data
     dependent. The output is still collinear with ``d`` (an additive shift along
     one direction is rank-1 by construction); what ``'band'`` genuinely buys is
     a data-dependent envelope, which is the only defensible reading of the
     "frequency band" claim.

6. **E2 and E4 are degenerate against E1 and are labelled as such.** In
   ``'shift'`` mode a frozen constant gate reduces the update to a uniform offset
   (E4) or, with an empty gate, to the document's time-domain interpolation (E2).
   The runner keeps the switches separate for reproducibility and logs
   ``envelope_stats`` so the equivalence is visible instead of being misread as
   an ablation result.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# basis-frame selection
# ---------------------------------------------------------------------------
def lowest_score_indices(scores: torch.Tensor, ratio: float = 0.2, min_k: int = 5,
                         lengths=None):
    """Indices of the lowest-scoring (most confidently normal) frames per video.

    Args:
        scores: ``(B, T)`` frame-level anomaly scores.
        ratio: fraction of frames to select (the document uses the bottom 20%).
        min_k: floor on the number of selected frames.
        lengths: ``(B,)`` valid lengths. Zero-padding is never *selected* (it is
            pushed to the end of the ordering and dropped by ``valid``).

    Returns:
        idx: ``(B, k)`` long - block positions, fixed width ``k``. For a video
            whose valid length is below ``k`` the tail of ``idx`` points at
            padding positions; ``valid`` marks them and callers must ignore them.
        valid: ``(B, k)`` bool - ``True`` where ``idx`` addresses a real frame.
        mask: ``(B, T)`` bool - the same information as a frame mask.
    """
    B, T = scores.shape
    dev = scores.device
    if lengths is None:
        lengths = torch.full((B,), T, device=dev, dtype=torch.long)
    lengths = lengths.to(dev).view(-1).clamp(min=1, max=T)

    valid_t = torch.arange(T, device=dev).unsqueeze(0) < lengths.unsqueeze(1)
    # padding -> +inf so an ascending sort places it last: the first k entries
    # are the k lowest-scoring *valid* frames whenever k <= length.
    s = scores.masked_fill(~valid_t, float('inf'))
    order = s.argsort(dim=1, descending=False)

    k = int(min(T, max(min_k, round(T * ratio))))
    idx = order[:, :k]
    valid = valid_t.gather(1, idx)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, valid)
    return idx, valid, mask


# ---------------------------------------------------------------------------
# the augmenter
# ---------------------------------------------------------------------------
def _structured_gate_init(n_fft: int, n_bins: int, bands: int, seed: int) -> torch.Tensor:
    """Gaussian-band initial spectrum whose *inverse transform* is spread out.

    Initialising the spectrum directly (a few Gaussian bumps over the frequency
    bins) does not control the object that matters. The envelope is
    ``irfft(profile)``, and a bump sitting near the DC bin - or a constant
    spectrum - transforms to a spike at frame 0, which is precisely the
    degeneracy this initialisation is supposed to avoid. The document's
    ``2.0 * ones`` is the extreme case of that mistake.

    So the desired *temporal* envelope is designed first and its spectrum
    derived: ``profile = rfft(env).real`` (the real part keeps the gate real and
    the round trip yields the even part of ``env``, spread whenever ``env`` is).

    The DC bin is removed analytically. ``rfft(env)[0] = sum(env)`` dwarfs every
    other bin for any non-negative envelope, so leaving it in and then squashing
    through a sigmoid saturates the tail of the spectrum into a near-constant
    gate - which transforms straight back into the frame-0 spike we are trying to
    escape. That failure is silent and was observed, hence the explicit removal
    and the bounded amplitude (``DC_AMPLITUDE``) that keeps the gate inside the
    linear region of the activation.
    """
    gen = torch.Generator().manual_seed(seed + 104729)
    t = torch.arange(n_fft, dtype=torch.float32)
    env = torch.zeros(n_fft)
    for _ in range(max(1, bands)):
        c = float(torch.rand(1, generator=gen)) * n_fft
        w = max(1.0, n_fft / (4.0 + 6.0 * float(torch.rand(1, generator=gen))))
        env = env + torch.exp(-0.5 * ((t - c) / w) ** 2)
    prof = torch.fft.rfft(env).real[:n_bins]
    # Centre over the *non-DC* bins. Subtracting prof[0] (the largest value) leaves
    # a one-sided, always-negative spectrum; subtracting the mean of the rest keeps
    # the gate centred on its identity value so the untrained augmenter is close to
    # a no-op in mean but still shaped in time.
    prof = prof - prof[1:].mean()
    prof[0] = 0.0
    scale = prof.abs().max() + 1e-8
    return prof / scale


# Amplitude of the normalised profile, in pre-activation units. Kept small
# enough that sigmoid/tanh stay in their linear region, so the initialised
# spectrum keeps its shape instead of being flattened into a constant.
DC_AMPLITUDE = 0.35


def _random_gate_init(n_fft: int, n_bins: int, seed: int) -> torch.Tensor:
    """Null model for the designed band gate: a fixed, *unstructured* spectrum.

    Same post-processing as :func:`_structured_gate_init` -- DC zeroed, non-DC
    bins mean-removed, max-abs normalised -- so the injected field has the same
    scale and the same ``exclude_dc`` handling. The only difference is that the
    spectrum itself is i.i.d. Gaussian noise from a fixed seed instead of a sum
    of Gaussian bumps, i.e. the *shape* information is destroyed while the
    magnitude statistics are preserved.

    This is the control that answers "is the learned frequency shape doing any
    work, or would a random gate of the same scale do the same job?". Paired
    with a *frozen structured* gate it separates the two things the learnable
    gate could be contributing: its frequency shape, and its adaptivity.
    """
    gen = torch.Generator().manual_seed(seed + 611953)
    prof = torch.randn(n_bins, generator=gen)
    prof = prof - prof[1:].mean()
    prof[0] = 0.0
    scale = prof.abs().max() + 1e-8
    return prof / scale


class FreqTextAugmenter(nn.Module):
    """Frequency-domain text-guided pseudo-anomaly synthesiser.

    Operates on the gathered block of "normal basis" frames, which is exactly the
    tensor the document feeds the augmenter (``normal_feats`` of shape
    ``(B, K, C)``). The block is transformed at its own length, so the band gate
    modulates the spectrum of the selected frames rather than of a zero-padded
    window. Use :func:`augment_sequence` to go from ``(B, T, C)`` to ``(B, T, C)``.

    Args:
        dim: feature dim (512 for CLIP ViT-B/16).
        block_len: block length K (the document uses ``T // 5``).
        mode: ``'shift'`` (document's additive form) or ``'band'`` (multiplicative
            band gain on the anomaly-projection spectrum, see module docstring).
        use_freq_gate: False freezes the gate at its neutral value (E4).
        alpha_range: ``(lo, hi)`` for the per-batch interpolation strength.
        band_gain: strength of the multiplicative gain in ``'band'`` mode.
        gate_init: raw gate mean. ``shift`` keeps the document's 2.0; ``band``
            uses 0.0 so training starts at identity gain.
        gate_bands: number of random Gaussian bumps used to initialise the gate in
            band space. A structured (band-pass) profile is required: a *flat*
            gate transforms to a Dirac (point 2), while a localised frequency bump
            transforms to an envelope spread over the block, which is what
            "frequency-band modulation" is supposed to mean.
        exclude_dc: subtract the gate's mean before injecting, so the synthesised
            signal carries **no** uniform offset. This is what separates E1 from
            E2: a time-domain interpolation along ``d`` *is* the DC component, so
            leaving DC in the spectrum makes E1 a scaled copy of E2 instead of a
            different hypothesis. Default True.
        delta_norm: ``'relative'`` rescales the injected update so its RMS equals
            ``alpha * rms(block)``; ``'absolute'`` keeps the document's raw
            ``alpha * unit_direction``. Relative scaling is required for a fair
            E1-vs-E3 comparison, because a unit-norm CLIP direction is ~20x
            smaller than a CLIP feature (norm ~10) and the two variants would
            otherwise perturb the features by different amounts.
        n_fft: transform length. Defaults to ``block_len``; setting it to ``T``
            reproduces the document's zero-pad-to-T behaviour for diagnostics.
        direction_source: ``'text'`` uses the caller's CLIP direction;
            ``'random'`` substitutes a fixed unit Gaussian vector (E3).
        gate_learnable: when False the spectral gate keeps its initial spectrum
            and is excluded from gradient updates (E3a: structure present but
            not adapted; E3b: structure destroyed and not adapted). The gate
            still appears in ``parameters()``; AdamW leaves ``.grad is None``
            parameters untouched, so no call site has to special-case it.
        gate_random: when True the gate is initialised from a fixed i.i.d.
            Gaussian spectrum instead of the designed Gaussian bumps (E3b).
            Same scale and same DC handling, so the only difference vs a
            structured gate is the shape information itself.
    """

    def __init__(self, dim: int = 512, block_len: int = 51, mode: str = 'shift',
                 use_freq_gate: bool = True, alpha_range=(0.1, 0.5),
                 band_gain: float = 1.0, gate_init: float = None,
                 gate_bands: int = 3, exclude_dc: bool = True,
                 delta_norm: str = 'relative', n_fft: int = None,
                 direction_source: str = 'text', seed: int = 234,
                 gate_learnable: bool = True, gate_random: bool = False):
        super().__init__()
        if mode not in ('shift', 'band', 'timedomain', 'noise'):
            raise ValueError(f"unknown mode '{mode}', expected 'shift', 'band', "
                             f"'timedomain' or 'noise'")
        if direction_source not in ('text', 'random'):
            raise ValueError(f"unknown direction_source '{direction_source}'")
        if delta_norm not in ('relative', 'absolute'):
            raise ValueError(f"unknown delta_norm '{delta_norm}'")

        self.dim = dim
        self.block_len = block_len
        self.mode = mode
        self.use_freq_gate = use_freq_gate
        self.alpha_range = tuple(alpha_range)
        self.band_gain = band_gain
        self.exclude_dc = exclude_dc
        self.delta_norm = delta_norm
        self.direction_source = direction_source
        self.gate_learnable = bool(gate_learnable)
        self.gate_random = bool(gate_random)
        self.n_fft = int(block_len if n_fft is None else n_fft)
        if gate_init is None:
            # deviation multiplier applied to the structured profile. 1.0 == the
            # designed band shape; 0.0 == a constant gate (the document's case,
            # which reduces to a frame-0 spike and is kept reproducible).
            gate_init = 1.0

        n_bins = self.n_fft // 2 + 1
        if use_freq_gate:
            # Structured init. The gate is stored in pre-activation space, so the
            # desired spectrum is mapped into the sigmoid/tanh range and inverted.
            # Without this the gate is a constant and the envelope is a spike at
            # frame 0, which the bottom-20% selection never touches - the
            # augmentation becomes the identity while every shape still checks out.
            base = (_random_gate_init(self.n_fft, n_bins, seed) if self.gate_random
                    else _structured_gate_init(self.n_fft, n_bins, gate_bands, seed)) * gate_init
            if mode == 'shift':
                target = (0.5 + DC_AMPLITUDE * base).clamp(1e-3, 1 - 1e-3)
                pre = torch.log(target / (1 - target))                 # logit
            else:
                target = (DC_AMPLITUDE * base).clamp(-1 + 1e-3, 1 - 1e-3)
                pre = 0.5 * torch.log((1 + target) / (1 - target))     # atanh
            pre[0] = 0.0      # DC stays at the identity; exclude_dc zeroes it anyway
            self.freq_gate = nn.Parameter(pre, requires_grad=self.gate_learnable)
        else:
            neutral = 2.0 if mode == 'shift' else 0.0
            self.register_buffer('freq_gate', torch.ones(n_bins) * neutral)

        # E3: one fixed random unit direction, own generator so the global seed
        # stays reproducible.
        gen = torch.Generator().manual_seed(seed + 7919)
        r = torch.randn(dim, generator=gen)
        self.register_buffer('random_direction', r / (r.norm() + 1e-8))

    # -- gate --------------------------------------------------------------
    def gate(self) -> torch.Tensor:
        """Band strength, shape ``(n_fft//2+1,)``.

        ``'shift'`` -> ``sigmoid(w)`` in ``(0,1)`` (document semantics).
        ``'band'``  -> ``tanh(w)`` in ``(-1,1)``, centred at 0 == identity gain.
        """
        w = self.freq_gate
        return torch.sigmoid(w) if self.mode == 'shift' else torch.tanh(w)

    def profile(self) -> torch.Tensor:
        """Spectral profile actually injected in ``'shift'`` mode.

        Two separate conditions must hold for the injected delta to be both
        non-trivial and free of the time-domain fallback's effect, and getting only
        one of them is a silent failure:

        * ``profile[0] = 0`` removes the *uniform temporal offset* - the mean of
          ``irfft(profile)`` over time is exactly the DC bin. Skipping this makes
          E1 contain E2's constant shift.
        * ``mean(profile[1:]) = 0`` removes the near-DC concentration. A gate with
          a large common offset across the non-DC bins (which is what a sigmoid
          gate naturally produces, since it is centred at 0.5) transforms to a
          spike at frame 0. Zeroing only bin 0 leaves that spike intact - observed
          as ``h_eff_len`` ~= 1.0 with the augmentation effectively addressing a
          single frame.

        ``profile[0]`` is therefore zeroed first and the remaining bins centred
        afterwards, in that order; the reverse order re-introduces an offset.
        """
        g = self.gate()
        if self.mode in ('shift', 'timedomain') and self.exclude_dc:
            g = g.clone()
            g[0] = 0.0
            if g.numel() > 1:
                g[1:] = g[1:] - g[1:].mean()
        return g

    def envelope(self) -> torch.Tensor:
        """Time-domain envelope ``h`` (diagnostic + the rank-1 identity)."""
        with torch.no_grad():
            if self.mode == 'timedomain':
                return torch.ones(self.n_fft)
            return torch.fft.irfft(self.profile().to(torch.cfloat), n=self.n_fft, dim=0)

    def envelope_stats(self) -> dict:
        """Diagnostics that expose the degeneracies described in the docstring."""
        h = self.envelope()
        e = h ** 2
        rms = h.pow(2).mean().sqrt()
        return {
            'h_peak': float(h.abs().max()),
            'h_index0_energy': float(e[0] / (e.sum() + 1e-12)),
            'h_rms': float(rms),
            'h_mean': float(h.mean()),
            # ~1 means a Dirac at index 0 (degenerate); ~K means spread over the block
            'h_eff_len': float(e.sum() / (e.max() + 1e-12)),
        }

    def sample_alpha(self, device) -> torch.Tensor:
        lo, hi = self.alpha_range
        return torch.empty((), device=device).uniform_(lo, hi)

    def resolve_direction(self, direction: torch.Tensor) -> torch.Tensor:
        if self.direction_source == 'random':
            return self.random_direction.to(direction.dtype)
        return direction

    # -- forward -----------------------------------------------------------
    def forward(self, block: torch.Tensor, direction: torch.Tensor, alpha=None):
        """Synthesise pseudo-anomaly frames inside one basis block.

        Args:
            block: ``(B, K, C)`` normal-basis frame features.
            direction: ``(C,)`` or ``(B, C)`` anomaly direction. L2-normalised
                internally. Pass the *final* direction (``e_anom - e_normal`` or a
                random unit vector).
            alpha: strength; sampled from ``alpha_range`` when None.

        Returns:
            ``(aug_block, alpha)`` with ``aug_block`` of shape ``(B, K, C)``.
        """
        B, K, C = block.shape
        if C != self.dim:
            raise ValueError(f"augmenter built for C={self.dim}, got C={C}")
        if K > self.n_fft:
            raise ValueError(f"block length {K} exceeds n_fft {self.n_fft}")

        direction = self.resolve_direction(direction)
        d = direction.unsqueeze(0).expand(B, -1) if direction.dim() == 1 else direction
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)

        if alpha is None:
            alpha = self.sample_alpha(block.device)

        x = block.to(torch.float32)

        profile = self.profile()
        if self.mode == 'noise':
            # N1: the plain-noise control for the "does the frequency-domain
            # structure matter" question. iid Gaussian per (frame, dim): no
            # direction, no temporal envelope, no spectral structure. Block
            # selection, ratio, alpha sampling and the shared relative-RMS
            # matching below are identical to the spectral modes, so the only
            # difference vs E3 is the structure of the injected delta.
            delta = torch.randn(B, K, C, device=x.device, dtype=x.dtype)
        elif self.mode == 'timedomain':
            # E2: the document's fallback - no spectral step at all, every selected
            # frame gets the same offset along d. Provided so the "frequency
            # domain matters" claim can actually be tested; under 'relative'
            # normalisation it is E4 with the gate mean removed.
            delta = alpha * d.unsqueeze(1).expand(B, K, C)
        elif self.mode == 'shift':
            # document formulation, broadcasting corrected. The added spectrum is
            # real and data-independent, so this round trip is exactly the rank-1
            # update alpha * irfft(profile)(t) * d (asserted in the tests).
            freq_shift = profile.view(1, -1, 1) * d.unsqueeze(1)          # (B,nb,C)
            delta = torch.fft.irfft(alpha * freq_shift, n=self.n_fft, dim=1)[:, :K]
        else:
            # 'band': the gate filters the spectrum of the anomaly projection, so
            # the reconstructed envelope depends on the data.
            a = (x * d.unsqueeze(1)).sum(-1)                              # (B,K)
            A = torch.fft.rfft(a, n=self.n_fft, dim=1)                    # (B,nb)
            gain = 1.0 + self.band_gain * profile.view(1, -1)              # (1,nb)
            a_aug = torch.fft.irfft(A * gain, n=self.n_fft, dim=1)[:, :K]  # (B,K)
            delta = alpha * (a_aug - a).unsqueeze(-1) * d.unsqueeze(1)

        if self.delta_norm == 'relative':
            # make alpha a *relative* perturbation magnitude so E1/E2/E3/E4 inject
            # the same amount of change; a unit-norm CLIP direction is otherwise
            # ~20x smaller than a CLIP feature and confounds the ablation.
            d_rms = delta.pow(2).mean().sqrt()
            x_rms = x.pow(2).mean().sqrt()
            delta = delta * (x_rms / (d_rms + 1e-8))

        return (x + delta).to(block.dtype), alpha


# ---------------------------------------------------------------------------
# (B, T, C) <-> block plumbing
# ---------------------------------------------------------------------------
def augment_sequence(feats: torch.Tensor, direction: torch.Tensor, augmenter: 'FreqTextAugmenter',
                     scores: torch.Tensor = None, ratio: float = 0.2, lengths=None,
                     alpha=None, generator=None):
    """Apply the augmenter to the selected frames of a full sequence.

    Args:
        feats: ``(B, T, C)``.
        direction: ``(C,)`` or ``(B, C)``.
        augmenter: the module.
        scores: ``(B, T)`` frame scores used to pick the basis frames. If None the
            first ``K`` frames of each video are used (deterministic smoke mode).
        ratio: bottom-score fraction.
        lengths: ``(B,)`` valid lengths.
        alpha: optional fixed strength.
        generator: unused hook kept for reproducibility experiments.

    Returns:
        ``(aug_feats, mask, alpha)``:
        ``aug_feats`` ``(B, T, C)`` - identical to ``feats`` outside ``mask``;
        ``mask`` ``(B, T)`` bool - the synthesised frames;
        ``alpha`` scalar tensor.
    """
    B, T, C = feats.shape
    dev = feats.device
    if lengths is None:
        lengths = torch.full((B,), T, device=dev, dtype=torch.long)
    lengths = lengths.to(dev).view(-1).clamp(min=1, max=T)

    if scores is None:
        k = int(min(T, max(5, round(T * ratio))))
        idx = torch.arange(k, device=dev).unsqueeze(0).expand(B, -1).contiguous()
        valid = torch.arange(k, device=dev).unsqueeze(0) < lengths.unsqueeze(1)
    else:
        idx, valid, _ = lowest_score_indices(scores, ratio=ratio, lengths=lengths)

    # block_len (== n_fft) must cover every selected frame. If it does not, the
    # augmenter raises deep inside forward(); fail here instead, with the numbers,
    # because the two knobs (--aug-block-len, --aug-ratio) look independent in the
    # CLI and a mismatch otherwise only shows up as a shape error mid-epoch.
    if idx.shape[1] > augmenter.n_fft:
        raise ValueError(
            f'augmentation selects {idx.shape[1]} frames per video (ratio={ratio}, '
            f'T={T}) but the augmenter transforms blocks of {augmenter.n_fft} '
            f'(aug_block_len). Raise --aug-block-len to at least {idx.shape[1]} or '
            f'lower --aug-ratio to <= {augmenter.n_fft / T:.4f}.')

    # (B, K, C) block -> augmenter -> write back only at valid positions
    block = torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, C))
    aug_block, alpha = augmenter(block, direction, alpha=alpha)
    aug_block = torch.where(valid.unsqueeze(-1), aug_block, block)

    aug_feats = feats.scatter(1, idx.unsqueeze(-1).expand(-1, -1, C), aug_block)
    mask = torch.zeros(B, T, device=dev, dtype=torch.bool)
    mask.scatter_(1, idx, valid)
    return aug_feats, mask, alpha


# ---------------------------------------------------------------------------
# losses (VadCLIP-compatible)
# ---------------------------------------------------------------------------
def clas2(scores: torch.Tensor, labels: torch.Tensor, lengths) -> torch.Tensor:
    """VadCLIP's ``CLAS2`` MIL loss on frame logits.

    Args:
        scores: ``(B, T)`` pre-sigmoid frame logits.
        labels: ``(B,)`` binary anomaly label.
        lengths: ``(B,)`` valid lengths.
    """
    B, T = scores.shape
    dev = scores.device
    lengths = lengths.to(dev).view(-1).clamp(min=1, max=T)
    prob = torch.sigmoid(scores)
    inst = torch.zeros(0, device=dev)
    for i in range(B):
        L = int(lengths[i])
        k = max(1, int(L / 16 + 1))
        tmp, _ = torch.topk(prob[i, :L], k=min(k, L), largest=True)
        inst = torch.cat([inst, torch.mean(tmp).view(1)], dim=0)
    return F.binary_cross_entropy(inst, labels.to(dev).view(-1).float())


def topk_prob(scores: torch.Tensor, lengths, ratio: float = 1 / 16) -> torch.Tensor:
    """Mean of the top ``L*ratio`` frame probabilities, per video ``(B,)``."""
    B, T = scores.shape
    lengths = lengths.to(scores.device).view(-1).clamp(min=1, max=T)
    out = []
    for i in range(B):
        L = int(lengths[i])
        k = max(1, int(L * ratio + 1))
        tmp, _ = torch.topk(torch.sigmoid(scores[i, :L]), k=min(k, L))
        out.append(tmp.mean())
    return torch.stack(out)


def text_logit_gap(logits2: torch.Tensor, lengths, normal_idx: int = 0) -> torch.Tensor:
    """Mean ``logit(anomaly) - logit(normal)`` over each video's top frames, ``(B,)``.

    ``logits2`` is ``(B, T, num_class)`` with column ``normal_idx`` == normal, the
    layout used by VadCLIP's prompt encoder. This is the ``logits2`` diagnostic:
    it tells us whether the *text-discriminated* anomaly score moves, which is
    what an AP change would have to come from.
    """
    B, T, N = logits2.shape
    lengths = lengths.to(logits2.device).view(-1).clamp(min=1, max=T)
    out = []
    for i in range(B):
        L = int(lengths[i])
        k = max(1, int(L / 16 + 1))
        top, _ = torch.topk(logits2[i, :L], k=min(k, L), dim=0)
        mean = top.mean(dim=0)
        out.append(mean.max(dim=0).values - mean[normal_idx])
    return torch.stack(out)


# ---------------------------------------------------------------------------
# one augmentation step, shared by the XD and UCF trainers
# ---------------------------------------------------------------------------
def augment_and_score(model, augmenter, dirs, visual_feats, feat_lengths,
                      class_idx, cfg, prompt_text=None, logits_ref=None):
    """Build the augmented pseudo-anomaly branch for one training batch.

    Replaces the document's ``train_step_with_aug``:

    * no per-video Python loop and no per-step ``clip.tokenize`` - directions come
      from a precomputed ``(num_class, C)`` table via one index lookup;
    * the augmented tensor stays ``(B, T, C)`` and costs one extra forward pass,
      instead of becoming ``(M*K, 1, C)`` single-frame "videos";
    * supervision lands on the synthesised frames - the actual hypothesis - rather
      than on the whole pseudo-video, which would also re-label genuinely anomalous
      frames as normal.

    Args:
        model: ``CLIPVAD``.
        augmenter: ``FreqTextAugmenter``, or ``None`` for E5.
        dirs: ``(num_class, C)`` table from ``build_class_directions``; row 0 is
            Normal (zero) and therefore excluded automatically.
        visual_feats: ``(B, T, C)``.
        feat_lengths: ``(B,)``.
        class_idx: ``(B,)`` long, 0 == Normal.
        cfg: namespace with ``aug_weight``/``aug_ratio``/``aug_target``/``aug_alpha``.
        prompt_text: forwarded to ``model``.
        logits_ref: optional ``(B, T)`` frame logits reused from the main forward
            pass so the reference scoring costs nothing.

    Returns:
        ``(loss_aug, stats)``. ``loss_aug`` is a 0-d tensor, exactly ``0`` when the
        branch is off or nothing was synthesised.
    """
    dev = visual_feats.device
    zero = torch.zeros((), device=dev)
    stats = {'n_aug': 0, 'alpha': 0.0, 'n_seq': 0}
    weight = float(getattr(cfg, 'aug_weight', 0.0))
    if augmenter is None or dirs is None or weight <= 0.0:
        return zero, stats

    B, T, C = visual_feats.shape
    lengths = feat_lengths.to(dev).view(-1).clamp(min=1, max=T)
    class_idx = class_idx.to(dev).view(-1).clamp(min=0, max=dirs.shape[0] - 1)

    # Synthesise only from anomalous videos: their lowest-score frames are the
    # normal-looking content *inside* an anomaly video, which is the hard pool the
    # hypothesis targets. Normal videos have no class direction to inject.
    sel = class_idx > 0
    if not bool(sel.any()):
        return zero, stats

    d = dirs[class_idx[sel]]
    keep = d.norm(dim=1) > 0          # classes whose direction table row is empty
    pos = sel.nonzero(as_tuple=True)[0][keep]
    if pos.numel() == 0:
        return zero, stats

    feats_sel = visual_feats[pos]
    len_sel = lengths[pos]
    d = d[keep]

    if logits_ref is None:
        with torch.no_grad():
            _, logits1_ref, _ = model(feats_sel, None, prompt_text, len_sel)
            logits_ref = logits1_ref.squeeze(-1).float()
    else:
        logits_ref = logits_ref[pos].float()

    alpha = float(getattr(cfg, 'aug_alpha', 0.0))
    fixed = None if alpha <= 0.0 else torch.tensor(alpha, device=dev)
    aug_feats, mask, alpha = augment_sequence(
        feats_sel, d, augmenter, scores=logits_ref, ratio=getattr(cfg, 'aug_ratio', 0.2),
        lengths=len_sel, alpha=fixed)

    _, logits1_aug, _ = model(aug_feats, None, prompt_text, len_sel)
    scores_aug = logits1_aug.squeeze(-1).float()

    n_sel = int(mask.sum().item())
    stats.update({'n_aug': n_sel, 'alpha': float(alpha.item()),
                  'n_seq': int(mask.shape[0])})
    if n_sel == 0:
        return zero, stats

    target = torch.full((n_sel,), float(getattr(cfg, 'aug_target', 0.9)), device=dev)
    loss_aug = F.binary_cross_entropy_with_logits(scores_aug[mask], target)
    return weight * loss_aug, stats
