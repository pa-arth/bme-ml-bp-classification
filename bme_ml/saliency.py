"""Gradient-based saliency for the 1D CNN.

Per the writeup template, we want to know "what temporal regions of the
waveform did the deep model focus on." We compute vanilla gradient
saliency — the absolute gradient of the predicted-class logit with
respect to the input — averaged over the test loader. Output shape is
`(2, T)` (PPG and ECG channels separately), so the result can be plotted
on top of the mean waveform for interpretation.

This is the simplest faithful attribution; if downstream interpretation
needs more (Grad-CAM 1D, integrated gradients), this module is the place
to add it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_saliency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    n_batches: int | None = None,
    target: str = "predicted",
) -> np.ndarray:
    """Average `|d logit / d input|` across the loader.

    `target="predicted"` uses each sample's argmax class as the target
    (i.e. "what does the model think it's seeing, and why"). `target=int`
    uses a fixed class index for every sample (e.g. 2 for "show me what
    drives the Hypertensive prediction across the whole test set").

    Returns a numpy array of shape `(C, T)` where `C` is the number of
    input channels (2 for PPG+ECG) and `T` is the segment length.
    """
    model = model.to(device).eval()
    accum: np.ndarray | None = None
    n_seen = 0

    for i, (x, _y) in enumerate(loader):
        if n_batches is not None and i >= n_batches:
            break
        x = x.to(device).requires_grad_(True)
        logits = model(x)
        if target == "predicted":
            cls = logits.argmax(dim=1)
        elif isinstance(target, int):
            cls = torch.full((x.size(0),), int(target), device=device, dtype=torch.long)
        else:
            raise ValueError(f"target must be 'predicted' or an int, got {target!r}")

        # Sum the predicted-class logit over the batch, then take grad —
        # this gives a per-sample gradient (linearity of d/dx of a sum).
        selected = logits.gather(1, cls.unsqueeze(1)).sum()
        grad = torch.autograd.grad(selected, x, retain_graph=False)[0]
        sal = grad.detach().abs().cpu().numpy()  # (B, C, T)
        batch_sum = sal.sum(axis=0)               # (C, T)
        if accum is None:
            accum = batch_sum
        else:
            accum += batch_sum
        n_seen += sal.shape[0]

    if accum is None or n_seen == 0:
        raise RuntimeError("loader yielded no batches")
    return accum / n_seen


def mean_waveform(loader: DataLoader, *, n_batches: int | None = None) -> np.ndarray:
    """Average input across the loader. Returns `(C, T)`. Useful as a
    backdrop for the saliency overlay so a reader can match peaks in the
    saliency curve to features in the actual waveform.
    """
    accum: np.ndarray | None = None
    n_seen = 0
    for i, (x, _y) in enumerate(loader):
        if n_batches is not None and i >= n_batches:
            break
        arr = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        batch_sum = arr.sum(axis=0)
        if accum is None:
            accum = batch_sum
        else:
            accum += batch_sum
        n_seen += arr.shape[0]
    if accum is None or n_seen == 0:
        raise RuntimeError("loader yielded no batches")
    return accum / n_seen


def plot_saliency(
    saliency: np.ndarray,
    waveform: np.ndarray | None = None,
    *,
    sample_rate_hz: float = 125.0,
    channel_names: tuple[str, ...] = ("PPG", "ECG"),
    axes=None,
):
    """Render saliency overlaid on the mean waveform per channel.

    If `waveform` is given (same shape as `saliency`), the channel's mean
    signal is plotted on the primary axis and saliency on a twin axis,
    so the reader can match saliency peaks to actual waveform features.
    """
    import matplotlib.pyplot as plt

    n_channels, n_samples = saliency.shape
    if axes is None:
        _fig, axes = plt.subplots(n_channels, 1, figsize=(11, 2.6 * n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]

    t = np.arange(n_samples) / sample_rate_hz
    for c in range(n_channels):
        ax = axes[c]
        if waveform is not None:
            ax.plot(t, waveform[c], color="0.6", lw=0.9, label=f"{channel_names[c]} (mean)")
            ax.set_ylabel(channel_names[c], color="0.4")
            twin = ax.twinx()
            twin.fill_between(t, 0, saliency[c], color="C3", alpha=0.35, label="saliency")
            twin.set_ylabel("|grad|", color="C3")
        else:
            ax.fill_between(t, 0, saliency[c], color="C3", alpha=0.5)
            ax.set_ylabel(f"{channel_names[c]} |grad|")
    axes[-1].set_xlabel("time (s)")
    return axes
