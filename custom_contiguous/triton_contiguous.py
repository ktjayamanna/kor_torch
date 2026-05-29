from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl


def _contiguous_strides(size: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    strides: list[int] = []
    for dim in reversed(size):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


@triton.jit
def _contiguous_5d_kernel(
    src,
    dst,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    d4: tl.constexpr,
    s0: tl.constexpr,
    s1: tl.constexpr,
    s2: tl.constexpr,
    s3: tl.constexpr,
    s4: tl.constexpr,
    n_elements: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements

    i4 = offsets % d4
    tmp = offsets // d4
    i3 = tmp % d3
    tmp = tmp // d3
    i2 = tmp % d2
    tmp = tmp // d2
    i1 = tmp % d1
    i0 = tmp // d1

    src_offsets = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4
    values = tl.load(src + src_offsets, mask=mask)
    tl.store(dst + offsets, values, mask=mask)


def custom_contiguous_triton(
    x: torch.Tensor,
    block_size: int = 256,
) -> torch.Tensor:
    if x.layout != torch.strided:
        raise NotImplementedError("custom_contiguous_triton only supports strided tensors")
    if not x.is_cuda:
        raise NotImplementedError("custom_contiguous_triton only supports CUDA tensors")
    if x.dim() != 5:
        raise NotImplementedError("custom_contiguous_triton only supports rank-5 tensors")
    if x.is_contiguous():
        return x

    size = tuple(x.size())
    out = torch.empty_strided(
        size,
        _contiguous_strides(size),
        dtype=x.dtype,
        layout=x.layout,
        device=x.device,
    )
    if x.numel() == 0:
        return out

    grid = (triton.cdiv(x.numel(), block_size),)
    _contiguous_5d_kernel[grid](
        x,
        out,
        size[0],
        size[1],
        size[2],
        size[3],
        size[4],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        x.stride(4),
        x.numel(),
        block_size,
    )
    return out
