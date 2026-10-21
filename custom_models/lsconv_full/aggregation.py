"""LSNet small-kernel aggregation, with Torch 2.0 / Triton 2.0 support.

The equations and cyclic channel indexing follow THU-MIG/lsnet, commit
cbe737c92b7c43ecf02d08545a07f03f1010177c, model/ska.py. These kernels
implement the operator independently; accumulation uses FP32 for FP16/BF16.
CPU/double inputs use the differentiable unfold reference.
"""
import math

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import functional as F

import triton
import triton.language as tl


def _validate(x, weights):
    if x.ndim != 4 or weights.ndim != 5:
        raise ValueError('Expected x[B,C,H,W] and weights[B,G,K*K,H,W]')
    b, c, h, w = x.shape
    wb, groups, neighbors, wh, ww = weights.shape
    kernel = math.isqrt(neighbors)
    if ((wb, wh, ww) != (b, h, w) or groups < 1 or c % groups
            or kernel * kernel != neighbors or kernel % 2 != 1):
        raise ValueError('Invalid aggregation dimensions, groups, or kernel')
    if x.device != weights.device or not x.is_floating_point() or not weights.is_floating_point():
        raise ValueError('Features and weights must be floating tensors on one device')
    return kernel


def aggregation_reference(x, weights):
    """Differentiable reference; shared kernel index is channel % G."""
    kernel = _validate(x, weights)
    b, c, h, w = x.shape
    neighbors = F.unfold(x, kernel_size=kernel, padding=kernel // 2)
    neighbors = neighbors.reshape(b, c, kernel * kernel, h, w)
    selected = weights[:, torch.arange(c, device=x.device) % weights.shape[1]]
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    # Match the kernel's casting of aggregation weights to feature precision.
    selected = selected.to(x.dtype)
    return (neighbors.to(dtype) * selected.to(dtype)).sum(2).to(x.dtype)


@triton.jit
def _forward_kernel(X, W, Y, C: tl.constexpr, H: tl.constexpr,
                    WIDTH: tl.constexpr, G: tl.constexpr, K: tl.constexpr,
                    COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    pixel = index % (H * WIDTH)
    channel = (index // (H * WIDTH)) % C
    batch = index // (C * H * WIDTH)
    row, col = pixel // WIDTH, pixel % WIDTH
    group = channel % G
    value = tl.full((BLOCK,), 0, tl.float32)
    for neighbor in tl.static_range(K * K):
        iy = row + neighbor // K - K // 2
        ix = col + neighbor % K - K // 2
        inside = valid & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < WIDTH)
        x_offset = (batch * C + channel) * H * WIDTH + iy * WIDTH + ix
        w_offset = ((batch * G + group) * K * K + neighbor) * H * WIDTH + pixel
        xv = tl.load(X + x_offset, mask=inside, other=0)
        wv = tl.load(W + w_offset, mask=valid, other=0).to(xv.dtype)
        value += xv.to(tl.float32) * wv.to(tl.float32)
    tl.store(Y + index, value, mask=valid)


@triton.jit
def _backward_x_kernel(DY, W, DX, C: tl.constexpr, H: tl.constexpr,
                       WIDTH: tl.constexpr, G: tl.constexpr, K: tl.constexpr,
                       COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    pixel = index % (H * WIDTH)
    channel = (index // (H * WIDTH)) % C
    batch = index // (C * H * WIDTH)
    row, col = pixel // WIDTH, pixel % WIDTH
    group = channel % G
    value = tl.full((BLOCK,), 0, tl.float32)
    for neighbor in tl.static_range(K * K):
        oy = row - neighbor // K + K // 2
        ox = col - neighbor % K + K // 2
        inside = valid & (oy >= 0) & (oy < H) & (ox >= 0) & (ox < WIDTH)
        out_pixel = oy * WIDTH + ox
        dy_offset = (batch * C + channel) * H * WIDTH + out_pixel
        w_offset = ((batch * G + group) * K * K + neighbor) * H * WIDTH + out_pixel
        dy = tl.load(DY + dy_offset, mask=inside, other=0)
        wv = tl.load(W + w_offset, mask=inside, other=0).to(dy.dtype)
        value += dy.to(tl.float32) * wv.to(tl.float32)
    tl.store(DX + index, value, mask=valid)


@triton.jit
def _backward_w_kernel(DY, X, DW, C: tl.constexpr, H: tl.constexpr,
                       WIDTH: tl.constexpr, G: tl.constexpr, K: tl.constexpr,
                       COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    pixel = index % (H * WIDTH)
    neighbor = (index // (H * WIDTH)) % (K * K)
    group = (index // (H * WIDTH * K * K)) % G
    batch = index // (G * K * K * H * WIDTH)
    row, col = pixel // WIDTH, pixel % WIDTH
    iy = row + neighbor // K - K // 2
    ix = col + neighbor % K - K // 2
    inside = valid & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < WIDTH)
    value = tl.full((BLOCK,), 0, tl.float32)
    for shared in tl.static_range(C // G):
        channel = group + shared * G
        x_offset = (batch * C + channel) * H * WIDTH + iy * WIDTH + ix
        dy_offset = (batch * C + channel) * H * WIDTH + pixel
        xv = tl.load(X + x_offset, mask=inside, other=0)
        dy = tl.load(DY + dy_offset, mask=inside, other=0)
        value += xv.to(tl.float32) * dy.to(tl.float32)
    tl.store(DW + index, value, mask=valid)


class _AggregationFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, weights):
        x, weights = x.contiguous(), weights.contiguous()
        b, c, h, w = x.shape
        groups, kernel = weights.shape[1], math.isqrt(weights.shape[2])
        result = torch.empty_like(x)
        _forward_kernel[(triton.cdiv(x.numel(), 256),)](
            x, weights, result, c, h, w, groups, kernel, x.numel(), 256)
        ctx.save_for_backward(x, weights)
        ctx.kernel = kernel
        return result

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x, weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        b, c, h, w = x.shape
        groups, kernel = weights.shape[1], ctx.kernel
        grad_x = grad_weights = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.empty_like(x)
            _backward_x_kernel[(triton.cdiv(x.numel(), 256),)](
                grad_output, weights, grad_x, c, h, w, groups, kernel, x.numel(), 256)
        if ctx.needs_input_grad[1]:
            grad_weights = torch.empty_like(weights)
            _backward_w_kernel[(triton.cdiv(weights.numel(), 256),)](
                grad_output, x, grad_weights, c, h, w, groups, kernel, weights.numel(), 256)
        return grad_x, grad_weights


def small_kernel_aggregation(x, weights):
    _validate(x, weights)
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return _AggregationFunction.apply(x, weights)
    return aggregation_reference(x, weights)
