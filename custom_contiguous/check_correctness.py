from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_contiguous import custom_contiguous_triton, custom_contiguous_triton_auto


def _check(name: str, x: torch.Tensor) -> None:
    ref = x.contiguous()

    for label, out in (
        ("triton", custom_contiguous_triton(x)),
        ("auto", custom_contiguous_triton_auto(x)),
    ):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert out.shape == ref.shape, (name, label, out.shape, ref.shape)
        assert out.stride() == ref.stride(), (name, label, out.stride(), ref.stride())
        assert out.dtype == ref.dtype, (name, label, out.dtype, ref.dtype)
        assert out.device == ref.device, (name, label, out.device, ref.device)
        assert out.is_contiguous(), (name, label, out.stride())
        if x.numel() > 0 and not x.is_contiguous():
            assert out.data_ptr() != x.data_ptr(), (name, label, "unexpected alias")

    print(f"OK {name}")


def main() -> None:
    assert torch.cuda.is_available()

    for dtype in (torch.float32, torch.int32):
        for shape in ((2, 3, 4, 5, 6), (2, 4, 8, 16, 32), (4, 8, 16, 32, 64)):
            base = torch.arange(
                torch.tensor(shape).prod().item(),
                device="cuda",
                dtype=dtype,
            ).reshape(shape)
            _check(f"{dtype}-transpose_like-{shape}", base.transpose(1, 3))
            _check(f"{dtype}-inner_contiguous-{shape}", base.permute(1, 0, 2, 3, 4))
            _check(f"{dtype}-inner_badly_strided-{shape}", base.permute(4, 0, 1, 2, 3))
            _check(f"{dtype}-sliced_offset-{shape}", base[:, 1:, :, :, :].transpose(1, 3))

    shape = (8, 16, 32, 64, 128)
    base = torch.arange(
        torch.tensor(shape).prod().item(),
        device="cuda",
        dtype=torch.float32,
    ).reshape(shape)
    _check("large_auto_dim0", base.permute(4, 0, 1, 2, 3))

    _check("empty", torch.empty((2, 0, 3, 4, 5), device="cuda").transpose(1, 3))
    print("all assert checks passed")


if __name__ == "__main__":
    main()
