---
title: "PyTorch 101"
date: 2022-08-13
tags: ["llm", "pytorch"]
---

### `Tensor`

PyTorch represents data with `Tensor`s, its n-dimensional array type. A tensor has a shape, a data type, and storage placed on a device such as the CPU or GPU. Many tensor operations create new values, while simple view operations such as `reshape` or `split` can reuse the same underlying storage without copying data.

**Init**

A simple way to create a tensor is with `torch.arange()`, which generates evenly spaced values over a range. For example, `torch.arange(0, 8, 2, dtype=torch.float)` creates a 1D tensor with values `[0., 2., 4., 6.]`, starting at `0`, stopping before `8`, and stepping by `2`.

Another common initializer is `torch.ones()`, which creates a tensor filled with `1`s. For example, `torch.ones(2, 3)` creates a tensor of shape `(2, 3)` where every element is `1`.

**Reshape**

`reshape()` changes the shape of a tensor without changing the total number of elements. For example, if `x.shape == (12,)`, then `x.reshape(3, 4)` gives `(3, 4)`.

`unsqueeze()` adds a dimension of size `1` to a tensor, while `squeeze()` removes a dimension of size `1`. For example, if `x.shape == (8, 512)`, then `x.unsqueeze(0)` gives `(1, 8, 512)`, and if `y.shape == (1, 8, 512)`, then `y.squeeze(0)` gives `(8, 512)`. These operations are often used to make tensor shapes line up for batching or broadcasting.

`tensor.view()` reshapes a tensor without changing its underlying data, as long as the new shape is compatible with the number of elements. For example, if `x = torch.arange(12)`, then `x.view(3, 4)` reshapes it from shape `(12,)` to `(3, 4)`. This is commonly used to reorganize tensor dimensions before operations such as matrix multiplication or attention.

### `Module`

A `Module` packages a computation from inputs to outputs. Its behavior is defined by the `forward` method, and it can own parameter tensors that are updated during training. 

`nn.Linear(in_dim, out_dim, bias=False)` contains a weight and an optional bias, and its forward pass applies them to the input to produce the output.

To build your own module, inherit from `nn.Module`, define an `__init__()` method to create and register the layers you need, call `super().__init__()`, and then define a `forward()` method that describes how the input flows through those layers. For example, a custom attention block could use the following skeleton:

```python
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
		# Define and register child modules here.

    def forward(self, q, k, v, mask=None):
        # Describe how inputs move through the module here.
        return output
```

If a module needs to store several repeated child modules, PyTorch provides `nn.ModuleList`, which registers each child module so its parameters are tracked correctly. For example:

```python
self.residual_connections = nn.ModuleList([ResidualConnection(dim, dropout) for _ in range(2)])
```

### DeviceMesh

`DeviceMesh` is PyTorch’s abstraction for device topology in distributed execution. It defines how devices are organized so APIs such as FSDP2 and DTensor know how tensors should be sharded or replicated. Below is a simple example:

```python
import torch
from torch.distributed.device_mesh import init_device_mesh

# 4 GPUs organized as a 1D mesh
mesh_1d = init_device_mesh("cuda", (4,))
print(mesh_1d)

# 8 GPUs organized as a 2D mesh
mesh_2d = init_device_mesh("cuda", (2, 4))
print(mesh_2d)
```