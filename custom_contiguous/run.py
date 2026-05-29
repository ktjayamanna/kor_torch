from __future__ import annotations

import time

import torch

from triton_contiguous import custom_contiguous_triton


def _mib(n: int) -> float:
    return n / 1024**2


def _measure(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return elapsed, out, before, after, peak, reserved


def check(shape: tuple[int, int, int, int, int], perm: tuple[int, int, int, int, int]) -> None:
    x = torch.randn(shape, device="cuda", dtype=torch.float16).permute(perm)

    x.contiguous()
    custom_contiguous_triton(x)

    torch_time, expected, pt_before, pt_after, pt_peak, pt_reserved = _measure(
        lambda: x.contiguous()
    )
    triton_time, actual, tr_before, tr_after, tr_peak, tr_reserved = _measure(
        lambda: custom_contiguous_triton(x)
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.is_contiguous()
    assert actual.data_ptr() != x.data_ptr()

    moved_gb = 2 * x.numel() * x.element_size() / 1e9
    pt_aux = max(0, pt_peak - pt_before)
    tr_aux = max(0, tr_peak - tr_before)
    print(
        f"shape={shape} perm={perm} "
        f"out_shape={tuple(actual.shape)}"
    )
    print(
        f"  torch : {torch_time * 1e3:8.3f} ms "
        f"{moved_gb / torch_time:8.2f} GB/s "
        f"aux={_mib(pt_aux):7.2f} MiB "
        f"alloc_delta={_mib(pt_after - pt_before):8.2f} MiB "
        f"peak_delta={_mib(pt_peak - pt_before):8.2f} MiB "
        f"reserved={_mib(pt_reserved):8.2f} MiB"
    )
    print(
        f"  triton: {triton_time * 1e3:8.3f} ms "
        f"{moved_gb / triton_time:8.2f} GB/s "
        f"aux={_mib(tr_aux):7.2f} MiB "
        f"alloc_delta={_mib(tr_after - tr_before):8.2f} MiB "
        f"peak_delta={_mib(tr_peak - tr_before):8.2f} MiB "
        f"reserved={_mib(tr_reserved):8.2f} MiB "
        f"time={triton_time / torch_time:.2f}x torch"
    )


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")

    check((2, 4, 8, 16, 32), (1, 0, 2, 3, 4))
    check((4, 8, 16, 32, 64), (1, 3, 0, 4, 2))
    check((8, 16, 32, 64, 128), (4, 0, 1, 2, 3))

    print("all checks passed")


if __name__ == "__main__":
    main()
