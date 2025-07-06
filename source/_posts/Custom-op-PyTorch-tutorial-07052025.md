---
title: Registering custom C++/CUDA operators using modern PyTorch APIs
layout: post
categories: blog
tag: PyTorch CUDA 
date: 2025-07-05
---


Nowadays, most ML inference engines are built on top of PyTorch‘s echosystem. By leveraging custom operators,  they deliver state-of-the-art throughput on large workloads—ranging from LLMs to Stable Diffusion. Compared to PyTorch’s native kernels, these operators usually offer lower latency and enable optimized implementations for cutting-edge operations that aren’t yet supported out of the box.

This tutorial shows you how to build and integrate a simple PyTorch custom operator that runs on both CPU and NVIDIA GPUs. For the PyTorch APIs used in this tutorial, check out the [official PyTorch documentation](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) and this [Google Doc](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.ptttacy8y1u9). An official example project can be found on [GitHub](https://github.com/pytorch/extension-cpp/blob/38ec45e3d8f908b22a9d462f776cf80fc9ab921a/pyproject.toml). Here, we provide a more compact, self-contained version to illustrate the same concepts in a smaller, easier-to-follow project. 

This simple custom operator is provided purely for demonstration; in real-world projects，PyTorch’s built-in operators can already handle simple workloads — like element-wise multiplication efficiently.

The repository of this tutorial:
<a href="https://github.com/HsChen-sys/torch-custom-op" target="_blank"
   style="display:inline-block; text-decoration:none;">
  <div style="
      width: 600px;             /* card width */
      height: 180px;            /* only show top 180px */
      overflow: hidden;         /* crop anything below */
      background: url('https://opengraph.githubassets.com/1/HsChen-sys/torch-custom-op') 
                  no-repeat top center;
      background-size: 600px auto;  /* scale width to 600px, height auto */
      border: 1px solid #ddd;
      border-radius: 6px;
    ">
  </div>
</a>

### Kernel Implementation

 The kernel simply multiply the input by 1.23 — equivalent to this Python snippet:


```python
def short_op(x: torch.Tensor):

	return x * 1.23
```
We define our simple kernel in the simple_ops.cu file.
```cpp
// my_ops/csrc/simple_ops.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h> // __half, __float2half, etc.
#include <torch/extension.h>
#include <ATen/ATen.h> // at::empty_like, at::ScalarType, etc.
#include <ATen/cuda/CUDAContext.h> // at::cuda::getCurrentCUDAStream()
#include <Python.h>

// ----------------------------------------
//  Device kernel template
//    - Works for float and __half via template dispatch
// ----------------------------------------
template <typename scalar_t> 
__global__ void shortKernel(
    const scalar_t* __restrict__ in,
    scalar_t* __restrict__ out,
    size_t total_elems
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elems) {
        if constexpr (std::is_same<scalar_t, __half>::value) {
		    // half-precision needs explicit conversion of the constant
            out[idx] = __float2half(1.23f) * in[idx];
        } else {
            out[idx] = scalar_t(1.23) * in[idx];
        }
    }
}

// ----------------------------------------
//CPU implementation 
// ----------------------------------------
template <typename scalar_t>
void short_kernel_cpu_impl(
    const scalar_t* __restrict__ in,
    scalar_t*       __restrict__ out,
    int64_t          total
) {
    for (int64_t i = 0; i < total; ++i) {
        out[i] = scalar_t(1.23) * in[i];
    }
}

```

We also provide two binding functions that sit between PyTorch and our low-level C++/CUDA code. These wrappers map PyTorch’s tensor dtypes to the correct C++/CUDA scalar types and then launch the appropriate kernel.

### PyTorch Binding: CUDA Side

```cpp
// my_ops/csrc/simple_ops.cu
// ----------------------------------------
//  CUDA-side PyTorch binding function
//    - Checks that input is on CUDA
//    - Allocates output tensor of same shape & dtype
//    - Computes grid/block sizes
//    - Retrieves current CUDA stream
//    - Dispatches to the proper instantiation of shortKernel<scalar_t>
// ----------------------------------------

torch::Tensor short_kernel(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    size_t total = x.numel();
    auto x_out = torch::empty_like(x);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "short_kernel", [&] {
        using scalar_t = scalar_t;  
        const scalar_t* in_ptr  = x.data_ptr<scalar_t>();
        scalar_t*       out_ptr = x_out.data_ptr<scalar_t>();
        // Launch kernel
        shortKernel<scalar_t><<<blocks, threads, 0, stream>>>(
            in_ptr, out_ptr, total
        );
    });

    return x_out;
}

```

### PyTorch Binding: CPU side

```cpp
// ----------------------------------------
//  CPU-side PyTorch binding function
//    - Ensures input is on CPU
//    - Makes tensor contiguous
//    - Allocates output
//    - Dispatches to short_kernel_cpu_impl<scalar_t>
// ----------------------------------------
torch::Tensor short_kernel_cpu(at::Tensor x) {
    // Making sure x is a CPU tensor
    TORCH_CHECK(!x.is_cuda(), "short_kernel_cpu: expected CPU tensor");
    auto x_contig = x.contiguous();
    auto y        = at::empty_like(x_contig);
    int64_t total = x_contig.numel();

    // Dispatch to different instantiation according to scalar_t
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x_contig.scalar_type(), "short_kernel_cpu", [&] {
            const auto* in_ptr  = x_contig.data_ptr<scalar_t>();
            auto*       out_ptr = y.data_ptr<scalar_t>();
            short_kernel_cpu_impl<scalar_t>(in_ptr, out_ptr, total);
        }
    );

    return y;
}
```

<details> <summary> What is AT_DISPATCH_FLOATING_TYPES_AND_HALF? </summary>
    
Here, `AT_DISPATCH_FLOATING_TYPES_AND_HALF` becomes roughly

```cpp
switch (x.scalar_type()) {
    case at::ScalarType::Float: {
    // 1) Pick C++ type float for this branch
    using scalar_t = float;

    // 2) “Insert” your lambda body here, with scalar_t = float:
    //    - read pointers as float*
    //    - launch the CUDA kernel instantiation shortKernel<float>
    //    shortKernel<float><<<blocks, threads, 0, stream>>>(
    //        reinterpret_cast<const float*>(x.data_ptr<float>()),
    //        reinterpret_cast<float*>(x_out.data_ptr<float>()),
    //        total
    //    );
    break;
    }
    case at::ScalarType::Double: {
    // 1) Pick C++ type double for this branch
    using scalar_t = double;

    // 2) Run the exact same code, but now:
    //    - in_ptr and out_ptr are double*
    //    - kernel invocation becomes shortKernel<double><<<…>>>(…)
    break;
    }
    case at::ScalarType::Half: {
    // 1) Pick C++ type at::Half (alias for CUDA __half)
    using scalar_t = at::Half;

    // 2) Again run the same code, but now scalar_t = at::Half:
    //    - in_ptr and out_ptr are at::Half*
    //    - kernel invocation becomes shortKernel<at::Half><<<…>>>(…)
    //    - inside the kernel the constexpr branch uses __float2half for the constant
    break;
    }
    default:
    // If the tensor’s dtype isn’t float/double/half, error out
    AT_ERROR("short_kernel not implemented for this scalar type");
}

```

This macro eliminates boilerplate and guarantees you cover all the major floating-point types seamlessly.
</details>

### Kernel registration:

Finally, we wire everything into PyTorch’s dispatcher:

```cpp
// my_ops/csrc/simple_ops.cu

TORCH_LIBRARY(my_ops, m) {
    m.def("short_kernel(Tensor x) -> Tensor"); //schema
}

TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
		// For CUDA tensors, use the above short_kernel()
    m.impl("short_kernel", &short_kernel);
}

TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
		// For CPU tensors, use short_kernel_cpu()
    m.impl("short_kernel", &short_kernel_cpu);
}

```

`TORCH_LIBRARY(my_ops, m)` registers operators to the namespace ‘my_ops’ in PyTorch, so we can use torch.ops.my_ops.short_kernel to call our implementations. To register this op, we need to pass a schema `"short_kernel(Tensor x) -> Tensor"`  to tell PyTorch how this op can be called. Please see [The Custom Operators Manual](https://pytorch.org/docs/main/notes/custom_operators.html) for more details.

- **`TORCH_LIBRARY_IMPL(my_ops, CUDA, m)`** says “for the `my_ops` operator library, here are the implementations to use when the *dispatch key* is CUDA.”
- **`m.impl("short_kernel", &short_kernel)`** binds the C++ function `short_kernel(at::Tensor)` as the CUDA‐backend kernel for that op. So if you do:

```cpp
x = torch.randn(..., device="cuda")
torch.ops.my_ops.short_kernel(x)
```

the dispatcher will route the call into your `short_kernel`  CUDA function.

- **`TORCH_LIBRARY_IMPL(my_ops, CPU, m)`** does the same thing for the CPU dispatch key.
- If you call

```cpp
x = torch.randn(..., device="cpu")
torch.ops.my_ops.short_kernel(x)
```

then PyTorch will invoke your `short_kernel_cpu(at::Tensor)` function instead.

### Setting Up the Build System

Following Python’s convention, we use **setuptools** to configure the build system.

Our code is as simple as the following:

```python
# setup.py
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="my_ops",
      packages=['my_ops'],
      ext_modules=[
          cpp_extension.CUDAExtension(
            "my_ops._C",
            ["my_ops/csrc/simple_ops.cu"],
            # define Py_LIMITED_API with min version 3.9 to expose only the stable
            # limited API subset from Python.h
            extra_compile_args={
                "cxx": ["-DPy_LIMITED_API=0x03090000", "-O2"],
                "nvcc": [
                    "-O3"
                ]},
            py_limited_api=True)],  # Build 1 wheel across multiple Python versions
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      options={"bdist_wheel": {"py_limited_api": "cp39"}}  # 3.9 is minimum supported Python version
)
```

By adding the compiler flag `-DPy_LIMITED_API=0x03090000` , we can assure that the built extension can be run on any Python environment with version ≥ 3.9. It helps verify that the extension is in fact only using the [CPython Stable Limited API](https://docs.python.org/3/c-api/stable.html) which ensures forward compatibility. According to PyTorch documentation,

> *If this requirement (Defining the `Py_LIMITED_API` flag) is not met, it is possible to build a wheel that looks CPython agnostic but will crash, or worse, be silently incorrect, in another CPython environment.*
> 

We need to register an empty _C module as the following code block shows to let python directly import our built .so library. By configuring `ext_modules.cpp_extension.CUDAExtension.name = “my_ops._C”` , the output file of our built module would be **my_ops/_C.abi3.so.** By executing `from my_ops import _C` *,* our custom ops can be registered to PyTorch since our simple_ops.cu file contains calls to `TORCH_LIBRARY`  and`TORCH_LIBRARY_IMPL` .

```cpp
// my_ops/csrc/simple_ops.cu
extern "C" PyObject *PyInit__C(void) {
    static struct PyModuleDef def = {
        PyModuleDef_HEAD_INIT,
        "_C",   // <— the name of the module
        nullptr,
        -1,
        nullptr
    };
    return PyModule_Create(&def);
}
```

<details> <summary>How to  export a c++ function to Python?</summary>
    
The `extern "C" PyObject *PyInit__C(void)` is a standard way to expose a cpp function to Python. 

To register a cpp function, we need to do the following things:

- **Write a real C++ func** (`add_one`).
- **Wrap it** in a `py_add_one` that speaks the Python/C API: parsing args & building return values.
- **Fill out** a `PyMethodDef[]` with your method(s).
- **Point your `PyModuleDef`** at that table.
- **Keep** your same `PyInit__C()` entrypoint.

A complete example:

```cpp
// my_ops/csrc/simple_ops.cu
#include <Python.h>

// 1) Your “real” C++ function you’d like to expose
int add_one(int x) {
    return x + 1;
}

// 2) A thin C‐API wrapper that parses Python args and calls your C++
//    function, then builds a Python return value.
static PyObject *py_add_one(PyObject *self, PyObject *args) {
    int input;
    // Parse a single integer argument
    if (!PyArg_ParseTuple(args, "i", &input)) {
        return nullptr;  // on failure, Python exception is set for you
    }
    int result = add_one(input);
    // Build a Python integer to return
    return PyLong_FromLong(result);
}

// 3) Method table: tell Python which names map to which C functions
static PyMethodDef SimpleOpsMethods[] = {
    // name in Python,      C wrapper,          argument style, doc‐string
    { "add_one",           py_add_one,         METH_VARARGS,  "Add one to an integer" },
    { nullptr,             nullptr,            0,             nullptr }   // sentinel
};

// 4) Module definition: plug in that table
static struct PyModuleDef simpleops_module = {
    PyModuleDef_HEAD_INIT,
    "_C",                // the module name
    "My simple ops",     // optional doc‐string
    -1,                  // per‐interpreter state size (−1 means “global”)
    SimpleOpsMethods     // the method table
};

// 5) Module init: Python will call this when you do “import my_ops._C”
extern "C" PyObject *PyInit__C(void) {
    return PyModule_Create(&simpleops_module);
}

```

Here we use `extern “C”` to disable the name mangling feature of C++. Without `extern "C"`, the compiler might emit a symbol like `_Z10PyInit__Cv` (or worse), and Python wouldn’t be able to locate the entry point it expects.
</details> 

To register our custom ops at import time,  `__init__.py` should be written as

```cpp
// my_ops/__init__.py

from . import _C
```

### Project File Organization

The current file organization:

```cpp
my_ops/
|-- csrc/
|   |-- simple_ops.cu
|-- __init__.py
graph_pytorch.py
setup.py
```

### Building Python Module and Importing

From the project’s root directory, run:

`pip install -e .` 

This command tells **pip** to install the current directory (`.` ) as an **editable** package.

During installation, pip will compile the C++/CUDA source code (as specified in **setup.py**) and place the resulting `.so` extension inside the package directory `./my_ops` .

Once that’s done, you can simply:

```python
import my_ops
y = torch.ops.my_ops.short_kernel(x)
```

to invoke your custom operator.