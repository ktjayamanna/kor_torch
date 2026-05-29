# The Real Question: Why Is The Copy Fast?

Ignore the already-contiguous fast path for this investigation. The only part
we care about is the forced materialization path:

```text
non-contiguous tensor
  -> allocate new contiguous tensor
  -> copy values in logical index order
  -> return contiguous tensor
```

In PyTorch this path is:

```text
aten/src/ATen/native/TensorProperties.cpp::contiguous
  -> self.clone(memory_format)
  -> aten/src/ATen/native/TensorFactories.cpp::clone
  -> allocate destination
  -> self.copy_(src)
```

The core implementation is tiny:

```cpp
Tensor contiguous(const Tensor& self, MemoryFormat memory_format) {
  return self.clone(memory_format);
}
```

after excluding the fast-path check and error handling.

## What Must Happen

For a dense rank-5 tensor, contiguous copy has to do this:

```text
shape:      [D0, D1, D2, D3, D4]
src stride: [S0, S1, S2, S3, S4]
dst stride: [D1*D2*D3*D4, D2*D3*D4, D3*D4, D4, 1]
```

For every logical index:

```text
i0, i1, i2, i3, i4
```

read:

```text
src[src_offset + i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4]
```

write:

```text
dst[((((i0*D1 + i1)*D2 + i2)*D3 + i3)*D4 + i4)]
```

That is the whole problem.

## What Makes It Fast

The copy is fast when PyTorch avoids treating it like arbitrary 5D indexing for
every element.

The speed comes from:

1. Destination writes are contiguous.
2. Adjacent dimensions can often be collapsed into larger linear runs.
3. Inner loops can become simple pointer increments.
4. Copy kernels are compiled, vectorized, and parallelized.
5. CPU copies can use cache-friendly loops and multiple threads.
6. CUDA copies can use coalesced writes and many threads.
7. Allocation comes from PyTorch's allocator, not repeated per-element growth.

The important enemy is per-element index math:

```text
offset = i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4
```

If that calculation happens naively for 1B elements in Python, the emulator
cannot match PyTorch. If the loop is compiled and dimensions are collapsed, the
work becomes close to a memory-bandwidth problem.

## First-Principles Copy Model

Start with the dumb truth:

```python
for i0 in range(D0):
  for i1 in range(D1):
    for i2 in range(D2):
      for i3 in range(D3):
        for i4 in range(D4):
          dst[k] = src[base + i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4]
          k += 1
```

Then optimize by asking:

```text
Which dimensions are already contiguous in source order?
Can the innermost loop copy a full contiguous run?
Can several dimensions collapse into one loop?
Can the outer loops only compute a base pointer once per block?
Can the inner loop become memcpy or vectorized loads/stores?
```

Example: if the last dimension is contiguous in source:

```text
S4 == 1
```

then the innermost loop reads and writes contiguous memory. That is much faster
than a fully strided gather.

If more dimensions match contiguous layout:

```text
S3 == D4
S2 == D3*D4
S1 == D2*D3*D4
```

then those dimensions can collapse into larger linear copies.

## Experiment Target

Use a dense rank-5 `float32` tensor.

Benchmark cases:

1. Simple transpose-like non-contiguous layout.
2. Permuted layout where the last logical dimension is still source-contiguous.
3. Layout where the innermost logical dimension is badly strided.

For each case compare:

```text
torch_result = x.contiguous()
emulated_result = custom_contiguous(x)
```

Measure:

```text
bytes_moved = input_bytes + output_bytes
bandwidth = bytes_moved / seconds
```

Matching PyTorch means matching effective bandwidth, not just returning the
right values.

## Files To Read

Only these matter for this question:

```text
aten/src/ATen/native/TensorProperties.cpp
aten/src/ATen/native/TensorFactories.cpp
aten/src/ATen/native/Copy.cpp
aten/src/ATen/native/cpu/CopyKernel.cpp
aten/src/ATen/native/cuda/Copy.cu
```

The investigation should move downward from:

```text
contiguous -> clone -> copy_ -> backend copy kernel
```

That backend copy kernel is where the speed truth lives.
