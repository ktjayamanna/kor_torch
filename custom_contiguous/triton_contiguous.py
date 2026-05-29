from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(
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
    tl.store(dst + offsets, tl.load(src + src_offsets, mask=mask), mask=mask)


@triton.jit
def _dim0_kernel(
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
    block_size: tl.constexpr,
):
    outer = tl.program_id(0)
    i4 = outer % d4
    tmp = outer // d4
    i3 = tmp % d3
    tmp = tmp // d3
    i2 = tmp % d2
    i1 = tmp // d2

    i0 = tl.arange(0, block_size)
    mask = i0 < d0
    src_offsets = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4
    dst_offsets = (((i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4 + i4
    tl.store(dst + dst_offsets, tl.load(src + src_offsets, mask=mask), mask=mask)


def _empty_contiguous_like(x: torch.Tensor) -> torch.Tensor:
    stride = []
    running = 1
    for dim in reversed(x.shape):
        stride.append(running)
        running *= dim
    return torch.empty_strided(
        tuple(x.shape),
        tuple(reversed(stride)),
        dtype=x.dtype,
        layout=x.layout,
        device=x.device,
    )


def custom_contiguous_triton(x: torch.Tensor) -> torch.Tensor:
    if x.is_contiguous():
        return x
    if x.dim() != 5 or not x.is_cuda or x.layout != torch.strided:
        raise NotImplementedError("only rank-5 CUDA strided tensors are supported")

    out = _empty_contiguous_like(x)
    if x.numel() == 0:
        return out

    size = tuple(x.shape)
    stride = tuple(x.stride())
    if stride[0] == 1 and x.numel() >= 4 * 1024 * 1024:
        _dim0_kernel[(size[1] * size[2] * size[3] * size[4],)](
            x,
            out,
            size[0],
            size[1],
            size[2],
            size[3],
            size[4],
            stride[0],
            stride[1],
            stride[2],
            stride[3],
            stride[4],
            triton.next_power_of_2(size[0]),
        )
    else:
        block_size = 256
        _linear_kernel[(triton.cdiv(x.numel(), block_size),)](
            x,
            out,
            size[0],
            size[1],
            size[2],
            size[3],
            size[4],
            stride[0],
            stride[1],
            stride[2],
            stride[3],
            stride[4],
            x.numel(),
            block_size,
        )
    return out
