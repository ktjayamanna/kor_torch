from __future__ import annotations

import argparse
import sys
import statistics
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from custom_contiguous import custom_contiguous


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


def _cases(device: torch.device, shape: tuple[int, int, int, int, int]) -> dict[str, torch.Tensor]:
    base = torch.randn(shape, device=device)
    return {
        "transpose_like": base.transpose(1, 3),
        "inner_contiguous": base.permute(1, 0, 2, 3, 4),
        "inner_badly_strided": base.permute(4, 0, 1, 2, 3),
    }


def run(device: torch.device, repeats: int, shape: tuple[int, int, int, int, int]) -> None:
    for name, x in _cases(device, shape).items():
        torch_time, torch_result = _time(lambda: x.contiguous(), device, repeats)
        custom_time, custom_result = _time(lambda: custom_contiguous(x), device, repeats)

        if torch_result.data_ptr() == x.data_ptr() or custom_result.data_ptr() == x.data_ptr():
            raise AssertionError(f"{name}: non-contiguous path returned an alias")
        torch.testing.assert_close(custom_result, torch_result, rtol=0, atol=0)
        if custom_result.stride() != torch_result.stride():
            raise AssertionError(
                f"{name}: stride mismatch {custom_result.stride()} != {torch_result.stride()}"
            )

        ratio = custom_time / torch_time
        print(
            f"{name:20s} "
            f"torch={torch_time * 1e3:8.3f} ms {_bandwidth_gbps(x, torch_time):8.2f} GB/s  "
            f"custom={custom_time * 1e3:8.3f} ms {_bandwidth_gbps(x, custom_time):8.2f} GB/s  "
            f"ratio={ratio:5.2f}x"
        )


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
    args = parser.parse_args()
    run(torch.device(args.device), args.repeats, tuple(args.shape))


if __name__ == "__main__":
    main()
