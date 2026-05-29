# How `Tensor.contiguous()` Is Structured

This note is a map for reverse engineering `tensor.contiguous()` and rebuilding
it from first principles in `custom_contiguous/emulated`.

At a high level, `.contiguous()` is not a single kernel. It is a method wrapper
around this behavior:

1. Check whether the tensor already has the requested contiguous memory format.
2. If yes, return the same tensor object.
3. If no, allocate a new tensor with the requested contiguous layout.
4. Copy values from the original logical indexing order into the new layout.

The core logic is small, but PyTorch routes it through generated bindings,
ATen operator metadata, C++ Tensor wrappers, native implementation code, and
low-level layout metadata.

## End-to-End Call Path

For normal Python use:

```text
x.contiguous(memory_format=torch.contiguous_format)
  -> Python Tensor method binding
  -> C++ Tensor::contiguous
  -> TensorBase::contiguous fast path
  -> aten operator dispatch: at::_ops::contiguous::call
  -> native implementation: at::native::contiguous
  -> clone(memory_format) if a copy is needed
```

Important files:

```text
tools/autograd/templates/python_variable_methods.cpp
aten/src/ATen/native/native_functions.yaml
aten/src/ATen/templates/TensorBody.h
aten/src/ATen/core/TensorBase.h
aten/src/ATen/core/Tensor.cpp
aten/src/ATen/native/TensorProperties.cpp
aten/src/ATen/native/TensorFactories.cpp
c10/core/TensorImpl.cpp
c10/core/Contiguity.h
torch/_refs/__init__.py
```

## Python Binding Layer

File:

```text
tools/autograd/templates/python_variable_methods.cpp
```

Key function:

```cpp
THPVariable_contiguous
```

This is the Python-facing `torch.Tensor.contiguous` method. It:

- parses the optional `memory_format` keyword,
- handles `__torch_function__`,
- unwraps the Python object into an ATen `Tensor`,
- checks if the tensor is already contiguous,
- returns `self` directly for the fast path,
- otherwise calls `self.contiguous(memory_format)`.

This file is a template used by PyTorch code generation, so the final built
binding is generated from it.

## Operator Schema

File:

```text
aten/src/ATen/native/native_functions.yaml
```

Schema:

```yaml
- func: contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)
  variants: method
  manual_cpp_binding: True
```

This declares the ATen operator. The important details are:

- `variants: method` means it exists as a tensor method.
- `manual_cpp_binding: True` means the C++ method binding is hand-written
  instead of only generated.
- The alias annotation `Tensor(a) -> Tensor(a)` allows returning the same
  tensor when no copy is needed.

## C++ Tensor Method Layer

Files:

```text
aten/src/ATen/templates/TensorBody.h
aten/src/ATen/core/TensorBase.h
aten/src/ATen/core/Tensor.cpp
```

`TensorBody.h` exposes:

```cpp
Tensor contiguous(MemoryFormat memory_format=MemoryFormat::Contiguous) const
```

That forwards to `TensorBase::contiguous`.

`TensorBase::contiguous` contains an important fast path:

```cpp
if (is_contiguous_or_false(memory_format)) {
  return *this;
} else {
  return __dispatch_contiguous(memory_format);
}
```

`Tensor.cpp` defines the slow path dispatch:

```cpp
TensorBase TensorBase::__dispatch_contiguous(c10::MemoryFormat memory_format) const {
  OptionalTensorRef self(*this);
  return at::_ops::contiguous::call(*self, memory_format);
}
```

For an emulation, this layer can be reduced to:

```text
if already_contiguous(tensor, memory_format):
    return tensor
return contiguous_copy(tensor, memory_format)
```

## Native Implementation

File:

```text
aten/src/ATen/native/TensorProperties.cpp
```

Core implementation:

```cpp
Tensor contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous_or_false(memory_format)) {
    return self;
  }
  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");

  return self.clone(memory_format);
}
```

This is the main behavior to rebuild. The only real work happens inside:

- `is_contiguous_or_false(memory_format)`
- `clone(memory_format)`

`MemoryFormat::Preserve` is explicitly rejected for `.contiguous()`, even
though `clone()` itself can accept preserve-format semantics.

## Contiguity Predicate

Files:

```text
c10/core/TensorImpl.cpp
c10/core/Contiguity.h
```

For default row-major contiguous layout, the key rule is in
`c10/core/Contiguity.h`:

```cpp
expected_stride = 1
for d from last dimension to first dimension:
    if size[d] == 1:
        continue
    if stride[d] != expected_stride:
        return false
    expected_stride *= size[d]
return true
```

Important behavior:

- A tensor with `numel == 0` is considered contiguous.
- Dimensions of size `1` do not constrain stride.
- Sparse tensors return false from the dense contiguity computation.
- Channels-last and channels-last-3d have separate stride rules.

For a first emulation, start with default contiguous format only:

```text
sizes   = tensor shape
strides = tensor strides
numel   = product(sizes)
```

Then implement the algorithm above.

## Copy Path Through `clone`

Files:

```text
aten/src/ATen/native/native_functions.yaml
aten/src/ATen/native/TensorFactories.cpp
aten/src/ATen/native/Copy.cpp
```

`contiguous()` uses:

```cpp
self.clone(memory_format)
```

The dense clone implementation does roughly:

```text
memory_format = requested format or Preserve
allocate output tensor with that layout
copy src values into output
return output
```

For default contiguous format, rebuilding this means:

1. Compute contiguous strides for the same shape.
2. Allocate storage of `numel` elements.
3. Iterate over logical indices of the source tensor.
4. Read from source using source strides and source storage offset.
5. Write into destination using contiguous strides.

## Python Reference Implementation

File:

```text
torch/_refs/__init__.py
```

Function:

```python
def contiguous(a, *, memory_format=torch.contiguous_format):
    ...
```

This is useful as a readable model. It rejects `torch.preserve_format`, checks
contiguity with `is_contiguous_for_memory_format_or_false`, and otherwise calls
`torch.clone(a, memory_format=memory_format)`.

For emulation, this Python reference is easier to mirror than the generated
C++ binding path.

## What To Rebuild First

Suggested order for `custom_contiguous/emulated`:

1. A small tensor metadata object with `sizes`, `strides`, `storage_offset`,
   and flat `storage`.
2. `numel(sizes)`.
3. `default_contiguous_strides(sizes)`.
4. `is_default_contiguous(sizes, strides, numel)`.
5. Logical-index iteration over an N-dimensional shape.
6. `clone_default_contiguous(tensor)`.
7. `custom_contiguous(tensor)`:

```python
def custom_contiguous(t):
    if is_default_contiguous(t.sizes, t.strides, t.numel()):
        return t
    return clone_default_contiguous(t)
```

After that works, extend toward:

- `memory_format=torch.channels_last`
- `memory_format=torch.channels_last_3d`
- size-1 dimension stride edge cases
- zero-numel tensors
- aliasing behavior when returning `self`

## Minimal Behavior Contract

For default `.contiguous()`:

- If the input is already contiguous, return the same logical tensor.
- If the input is not contiguous, return a new tensor.
- The new tensor has the same shape and values.
- The new tensor has default row-major contiguous strides.
- Logical values are preserved, storage layout is changed.
- `preserve_format` is not valid for `.contiguous()`.

This is the core contract to emulate before worrying about PyTorch dispatch,
autograd, devices, dtypes, sparse tensors, quantized tensors, or named tensors.
