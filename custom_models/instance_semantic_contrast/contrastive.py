"""Exact supervised-contrastive sum with bounded pair-matrix memory.

Positive pairs are distinct GT rows with the same class; self-pairs are excluded.
An anchor with no positive contributes zero but remains a negative candidate for
other anchors. Normalization by ALL valid GTs is done by the caller. The custom
first-order backward includes both anchor and candidate derivatives; chunking
does not sample, truncate or detach either side of a feature pair.
"""
import math

import torch
from torch.autograd.function import once_differentiable


def _pair_chunk(features, labels, start, end, temperature):
    logits = features[start:end] @ features.T / temperature
    local = torch.arange(end - start, device=features.device)
    global_index = torch.arange(start, end, device=features.device)
    logits[local, global_index] = -torch.inf
    positives = labels[start:end, None] == labels[None, :]
    positives[local, global_index] = False
    counts = positives.sum(dim=1)
    return logits, positives, counts


class _SupervisedContrastiveSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, labels, temperature, chunk_size):
        ctx.save_for_backward(features, labels)
        ctx.temperature, ctx.chunk_size = temperature, chunk_size
        loss = features.new_zeros(())
        if len(features) < 2:
            return loss
        for start in range(0, len(features), chunk_size):
            end = min(start + chunk_size, len(features))
            logits, positives, counts = _pair_chunk(
                features, labels, start, end, temperature)
            positive_mean = logits.masked_fill(~positives, 0).sum(1) / counts.clamp(min=1)
            per_anchor = torch.logsumexp(logits, dim=1) - positive_mean
            loss = loss + per_anchor[counts > 0].sum()
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        features, labels = ctx.saved_tensors
        gradient = torch.zeros_like(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            if len(features) >= 2:
                for start in range(0, len(features), ctx.chunk_size):
                    end = min(start + ctx.chunk_size, len(features))
                    logits, positives, counts = _pair_chunk(
                        features, labels, start, end, ctx.temperature)
                    coefficients = logits.softmax(dim=1)
                    coefficients -= positives.to(features.dtype) / counts.clamp(min=1)[:, None]
                    coefficients *= (counts > 0)[:, None]
                    coefficients /= ctx.temperature
                    gradient[start:end] += coefficients @ features
                    gradient += coefficients.T @ features[start:end]
        return gradient * grad_output, None, None, None


def supervised_contrastive_sum(features, labels, temperature=.2, chunk_size=128):
    """Features are normalized by the caller; returns an unnormalized sum."""
    if features.ndim != 2 or labels.shape != features.shape[:1]:
        raise ValueError('Expected [GT, embedding] features and [GT] labels')
    if features.dtype not in (torch.float32, torch.float64):
        raise ValueError('Pairwise contrast must be computed in FP32 or FP64')
    if labels.dtype != torch.long or labels.device != features.device:
        raise ValueError('Class labels must be int64 on the feature device')
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Contrast temperature must be finite and positive')
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('Pair chunk size must be a positive integer')
    # Also protect direct calls made under autocast: pair similarities and their
    # recomputed backward must use the same precision.
    with torch.autocast(device_type=features.device.type, enabled=False):
        return _SupervisedContrastiveSum.apply(features, labels, temperature, chunk_size)
