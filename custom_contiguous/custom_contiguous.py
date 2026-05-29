from __future__ import annotations

from typing import Sequence

import torch


def _contiguous_strides(size: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    strides: list[int] = []
    for dim in reversed(size):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def custom_contiguous(
    x: torch.Tensor,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> torch.Tensor:
    if memory_format != torch.contiguous_format:
        raise NotImplementedError("custom_contiguous only emulates default contiguous layout")
    if x.layout != torch.strided:
        raise NotImplementedError("custom_contiguous only emulates dense strided tensors")

    if x.is_contiguous(memory_format=memory_format):
        return x

    size = tuple(x.size())
    stride = tuple(x.stride())
    out = torch.empty_strided(
        size,
        _contiguous_strides(size),
        dtype=x.dtype,
        layout=x.layout,
        device=x.device,
        pin_memory=x.is_pinned() if x.device.type == "cpu" else False,
    )

    if x.numel() == 0:
        return out

    if x.dim() == 0:
        out.fill_(x.item())
        return out

    index = [0] * x.dim()
    flat_out = out.reshape(-1)
    base_offset = x.storage_offset()
    for linear_index in range(x.numel()):
        source_offset = base_offset
        for dim, dim_index in enumerate(index):
            source_offset += dim_index * stride[dim]
        flat_out[linear_index] = x.as_strided((), (), storage_offset=source_offset)

        for dim in range(x.dim() - 1, -1, -1):
            index[dim] += 1
            if index[dim] != size[dim]:
                break
            index[dim] = 0

    return out
