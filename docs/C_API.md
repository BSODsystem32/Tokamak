# Using Tokamak from a C Host Application

Tokamak exports a small host-side C ABI through `include/tkm.h`. A caller includes
that header and passes ordinary host pointers to the runtime API; the caller does
not need CUDA headers or CUDA types. The dynamic and static libraries both
require CUDA runtime libraries, NVRTC,
the CUDA driver, and cuBLAS at run time. The static SDK release bundle includes
the redistributable CUDA runtime/NVRTC/cuBLAS libraries and host-link libraries,
so consumers do not need to install the full CUDA Toolkit. An NVIDIA driver
must still be installed on the target machine.

## Release Packages

The executable/runtime bundles (`tkm-windows-x64.zip` and `tkm-linux-x64.tar.gz`)
include the CLI and model files. The separate static SDK bundles
(`tkm-sdk-windows-x64.zip` and `tkm-sdk-linux-x64.tar.gz`) contain the public
header, Tokamak static archive, required CUDA runtime libraries, and host-link
libraries, without the CLI or model files:

| Platform | Static library |
|---|---|
| Windows x64 | `lib/tkm_static.lib` plus CUDA host-link libraries; required DLLs are in `bin/` |
| Linux x64 | `lib/libtkm.a` plus CUDA runtime/link libraries in `lib/` |

The static archive removes the Tokamak DLL/shared-library deployment step. The
SDK bundle includes the CUDA runtime pieces needed to link and run without a
full Toolkit installation; it does not include the NVIDIA driver.

## Build

On Windows, run `build.bat` from an x64 Native Tools Command Prompt for VS. The
script also attempts to initialize the MSVC x64 environment automatically when
run from a normal shell. It builds `tkm.dll`, its import library `tkm.lib`, the
static archive `tkm_static.lib`, and C ABI example programs linked against both.

On Linux, build both libraries and the C ABI examples with:

```sh
make -j
make mnist_idx
make mnist_idx_static
```

To link an application against a downloaded static SDK, use the platform host
compiler and the libraries included in the SDK:

```sh
# Linux
gcc app.c -Iinclude lib/libtkm.a \
  -Llib -lcudart -lnvrtc -lcuda -lcublas -lcublasLt \
  -ldl -lpthread -lstdc++ -Wl,-rpath,'$ORIGIN/lib' -o app

# Windows (from the SDK root and an x64 Native Tools Command Prompt)
cl app.c /I include lib\tkm_static.lib lib\cudart.lib lib\nvrtc.lib \
  lib\cuda.lib lib\cublas.lib lib\cublasLt.lib /Fe:app.exe
```

On Windows, put the SDK's `bin/` directory next to the executable or on `PATH`.
On Linux, retain the shown rpath (run the command from the SDK root) or set
`LD_LIBRARY_PATH` to the SDK's `lib/` directory. The Linux SDK includes the
CUDA `libcuda.so` stub required for linking; it is not a driver runtime. The
actual driver (`nvcuda.dll` / `libcuda.so.1`) comes from the installed NVIDIA
driver and is intentionally not bundled.

Keep the built library, CUDA runtime dependencies, and model file available when
running an application that uses the ABI. The example loads
`models/resnet_mnist.tkm` relative to the current working directory.

## MNIST IDX Example

`examples/mnist_idx.c` reads the standard uncompressed IDX image and label files,
checks their headers and dimensions, converts image bytes from `[0, 255]` to
host-side `float` values in `[0, 1]`, then trains and evaluates the ResNet model
through `tkm_load_file`, `tkm_runtime_step`, and `tkm_runtime_eval`.

Download and decompress the four standard MNIST IDX files, then run:

```cmd
mnist_idx.exe <train-images-idx3-ubyte> <train-labels-idx1-ubyte> <t10k-images-idx3-ubyte> <t10k-labels-idx1-ubyte> [batch=64] [epochs=1]
```

For example, with the files in `data\`:

```cmd
mnist_idx.exe data\train-images-idx3-ubyte data\train-labels-idx1-ubyte data\t10k-images-idx3-ubyte data\t10k-labels-idx1-ubyte 1024 30
```

The current training API consumes fixed-size batches, so the example trains only
on complete batches and reports any samples left over at the end of the training
set as skipped. Evaluation accepts any positive sample count and handles a
partial final batch without including padded examples in the accuracy count.

One observed run with batch size 1024 for 30 epochs reported a final training
loss of `0.1206` and test accuracy of `9846/10000 (98.46%)`. This is a sample
result, not a guaranteed accuracy; it can vary with initialization, hardware,
and training settings.

## Minimal Host-Side API Call

```c
#include "tkm.h"

int main(void) {
    TkmRuntime *rt = tkm_load_file("models/resnet_mnist.tkm", 64);
    if (!rt) return 1;

    float images[64 * 28 * 28] = {0};
    uint8_t labels[64] = {0};
    float loss = tkm_runtime_step(rt, images, labels, 1e-3f);
    (void)loss;

    tkm_runtime_free(rt);
    return 0;
}
```

The input and label buffers are host memory. Provide `batch_size * H * W * C`
floats and `batch_size` labels for each training step, matching the model input
shape and runtime batch size.
