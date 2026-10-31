# Using Tokamak from a C Host Application

Tokamak exports a small host-side C ABI through `include/tkm.h`. A caller includes
that header and passes ordinary host pointers to the runtime API; the caller does
not need CUDA headers or CUDA types. The Tokamak DLL/shared library itself still
requires the CUDA runtime, NVRTC, the CUDA driver, and cuBLAS at run time.

## Build

On Windows, run `build.bat` from an x64 Native Tools Command Prompt for VS. The
script also attempts to initialize the MSVC x64 environment automatically when
run from a normal shell. It builds `tkm.dll`, `tkm.lib`, and `mnist_idx.exe`.

On Linux, build the shared library and example with:

```sh
make -j
make mnist_idx
```

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
