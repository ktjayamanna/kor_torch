from __future__ import annotations

import argparse
import sys
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_contiguous import custom_contiguous, custom_contiguous_triton_auto


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    repeats: int,
) -> tuple[float, torch.Tensor]:
    result = fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        _sync(device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), result


def _bandwidth_gbps(x: torch.Tensor, seconds: float) -> float:
    bytes_moved = 2 * x.numel() * x.element_size()
    return bytes_moved / seconds / 1e9


def _dtype(name: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "int32": torch.int32,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    return dtypes[name]


def _base_tensor(
    shape: tuple[int, int, int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dtype.is_floating_point:
        return torch.randn(shape, device=device, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device, dtype=dtype)
    return torch.randint(0, 100, shape, device=device, dtype=dtype)


def _cases(
    device: torch.device,
    shape: tuple[int, int, int, int, int],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    base = _base_tensor(shape, device, dtype)
    return {
        "transpose_like": base.transpose(1, 3),
        "inner_contiguous": base.permute(1, 0, 2, 3, 4),
        "inner_badly_strided": base.permute(4, 0, 1, 2, 3),
    }


def run(
    device: torch.device,
    repeats: int,
    shape: tuple[int, int, int, int, int],
    dtype: torch.dtype,
) -> None:
    print(f"shape={shape} dtype={dtype} device={device}")
    for name, x in _cases(device, shape, dtype).items():
        torch_time, torch_result = _time(lambda: x.contiguous(), device, repeats)
        custom_time, custom_result = _time(lambda: custom_contiguous(x), device, repeats)
        triton_time = None
        triton_result = None
        if device.type == "cuda":
            triton_time, triton_result = _time(
                lambda: custom_contiguous_triton_auto(x),
                device,
                repeats,
            )

        if torch_result.data_ptr() == x.data_ptr() or custom_result.data_ptr() == x.data_ptr():
            raise AssertionError(f"{name}: non-contiguous path returned an alias")
        torch.testing.assert_close(custom_result, torch_result, rtol=0, atol=0)
        if triton_result is not None:
            if triton_result.data_ptr() == x.data_ptr():
                raise AssertionError(f"{name}: triton path returned an alias")
            torch.testing.assert_close(triton_result, torch_result, rtol=0, atol=0)
        if custom_result.stride() != torch_result.stride():
            raise AssertionError(
                f"{name}: stride mismatch {custom_result.stride()} != {torch_result.stride()}"
            )
        if triton_result is not None and triton_result.stride() != torch_result.stride():
            raise AssertionError(
                f"{name}: triton stride mismatch {triton_result.stride()} != {torch_result.stride()}"
            )

        ratio = custom_time / torch_time
        line = (
            f"{name:20s} "
            f"torch={torch_time * 1e3:8.3f} ms {_bandwidth_gbps(x, torch_time):8.2f} GB/s  "
            f"python={custom_time * 1e3:8.3f} ms {_bandwidth_gbps(x, custom_time):8.2f} GB/s  "
            f"py/torch={ratio:7.2f}x"
        )
        if triton_time is not None:
            line += (
                f"  triton={triton_time * 1e3:8.3f} ms "
                f"{_bandwidth_gbps(x, triton_time):8.2f} GB/s  "
                f"tri/torch={triton_time / torch_time:5.2f}x"
            )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--shape",
        type=int,
        nargs=5,
        default=(2, 4, 8, 16, 32),
        metavar=("D0", "D1", "D2", "D3", "D4"),
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "float32", "int32", "uint8", "bool"),
    )
    args = parser.parse_args()
    run(torch.device(args.device), args.repeats, tuple(args.shape), _dtype(args.dtype))


if __name__ == "__main__":
    main()
