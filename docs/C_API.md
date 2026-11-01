# Tokamak C ABI and Host Integration Guide

This document is the reference for embedding Tokamak from C, C++, or another
FFI-capable host language. The public declarations are in
[`include/tkm.h`](../include/tkm.h). `examples/mnist_idx.c` is a runnable
example, but it is not the API contract; the sections below define ownership,
buffer sizes, callback rules, return values, and lifecycle requirements.

## 1. ABI scope

Tokamak exposes two layers:

1. **Graph/compiler APIs** operate on the public `TkmGraph` IR and are useful
   for applications that build or inspect graphs themselves.
2. **Runtime APIs** expose the normal embedding path: load a `.tkm` DSL,
   provide host input buffers, run training or inference, and save/load
   weights.

The ABI uses C linkage when included from C++, and does not expose CUDA types
in `tkm.h`. Host applications pass ordinary CPU pointers. Device allocation,
NVRTC compilation, CUDA streams, cuBLAS handles, and generated kernels remain
private to `TkmRuntime`.

The ABI is currently a synchronous, single-runtime interface. Calls that
transfer data to or from the GPU return after the required CUDA work has
completed. A runtime must not be used concurrently from multiple host threads;
create one runtime per independently synchronized execution stream.

## 2. Deployment and linking

### Runtime packages and SDK packages

The executable/runtime bundles (`tkm-windows-x64.zip` and
`tkm-linux-x64.tar.gz`) include the CLI and model files. Static SDK bundles
(`tkm-sdk-windows-x64.zip` and `tkm-sdk-linux-x64.tar.gz`) contain the public
header, Tokamak static archive, CUDA runtime/NVRTC/cuBLAS libraries, and host
link libraries. The SDK does not include an NVIDIA driver.

| Platform | Tokamak library |
|---|---|
| Windows x64 | `lib/tkm_static.lib` or the shared `tkm.dll` + `tkm.lib` |
| Linux x64 | `lib/libtkm.a` or the shared `libtkm.so` |

### Building this repository

On Windows, run `build.bat` from an x64 Native Tools Command Prompt for
Visual Studio. It builds the CLI, DLL/import library, static library, and C
ABI examples.

On Linux:

```sh
make -j
make mnist_idx
make mnist_idx_static
```

### Linking a host application

```sh
# Linux static SDK
gcc app.c -Iinclude lib/libtkm.a \
  -Llib -lcudart -lnvrtc -lcuda -lcublas -lcublasLt \
  -ldl -lpthread -lm -lstdc++ -Wl,-rpath,'$ORIGIN/lib' -o app
```

```cmd
rem Windows static SDK, from an x64 Native Tools Prompt
cl app.c /I include lib\tkm_static.lib lib\cudart.lib lib\nvrtc.lib ^
  lib\cuda.lib lib\cublas.lib lib\cublasLt.lib /Fe:app.exe
```

For a DLL build, compile the caller with `TKM_USE_DLL` and link against
`tkm.lib`; place `tkm.dll` and the CUDA runtime DLLs beside the executable or
on `PATH`. On Linux, keep the shared library and CUDA libraries discoverable
through the shown rpath or `LD_LIBRARY_PATH`.

The Linux SDK's `libcuda.so` is a link-time driver stub. The actual
`libcuda.so.1`/`nvcuda.dll` is supplied by the installed NVIDIA driver.

## 3. Runtime lifecycle

The normal lifecycle is:

```text
load DSL -> create TkmRuntime -> optionally load weights
         -> step/eval/forward or dataset helpers
         -> optionally save weights -> tkm_runtime_free
```

Use exactly one destroy call for every non-null runtime. The runtime owns all
GPU allocations and generated CUDA state. The caller owns:

- the DSL bytes passed to `tkm_load_buffer` or `tkm_load_string`;
- input, label, output, and checkpoint buffers;
- the `user` object passed to a `TkmDatasetNext` callback;
- strings and file paths passed to path-based APIs.

The caller may release or reuse input buffers after the API call returns.
Callbacks are invoked synchronously and must not retain Tokamak's temporary
input or label pointers after returning.

## 4. Tensor and batch conventions

### Input layout

For a graph whose input shape is `H x W x C` and runtime batch size `B`,
allocate at least:

```text
B * H * W * C floats
```

Input samples are contiguous host `float` values in NHWC order:

```c
input[((sample * H + y) * W + x) * C + channel]
```

Labels are one `uint8_t` per sample. Their interpretation is determined by the
graph's loss operator; for the standard classification models they are class
indices.

### Fixed runtime batch size

`tkm_runtime_step` and `tkm_runtime_forward` always consume exactly `B`
samples. `tkm_runtime_eval` accepts any positive sample count and internally
processes complete batches plus a zero-padded final GPU batch; only the
requested samples contribute to the returned accuracy.

The dataset training callback must return exactly `B` samples. The dataset
evaluation callback may return `1..B` samples for its final batch.

### Output layout

For an output shape `H_out x W_out x C_out`, inference writes
`B * H_out * W_out * C_out` host floats in the same contiguous NHWC order.
For `models/llama3_style.tkm` with `B == 1`, the output is
`[sequence, vocabulary]`; the logits for the last token begin at
`(sequence - 1) * vocabulary`.

`tkm_runtime_forward` is only valid for graphs ending in a tensor output. A
graph ending in `SoftmaxCE`, `SoftmaxFocal`, or `SoftmaxMSE` is a training/loss
graph and has no inference output through this function.

## 5. Error and return-value contract

There is no thread-local error object or error-code query in the current ABI.
Use the return value of each API and treat diagnostics written to `stderr` as
informational details.

| API category | Failure result |
|---|---|
| Runtime creation | `NULL` |
| Size query | `0` |
| `tkm_runtime_step` | `0.0f` can mean failure or a valid zero loss |
| `tkm_runtime_eval` | `0` can mean failure or zero correct samples |
| `tkm_runtime_forward` | `0` |
| Weight buffer/file APIs | `0` |
| Dataset helpers | `0` |
| Successful integer/boolean operations | non-zero |

Because some CUDA and NVRTC failures are fatal backend errors, an application
must not assume every GPU failure can be recovered from by checking a return
code. Validate paths, sizes, model compatibility, and callback results before
entering the runtime.

## 6. API reference

All functions in this section are declared with `TKM_API`.

### 6.1 Runtime creation

#### `tkm_runtime_ignite`

```c
TkmRuntime *tkm_runtime_ignite(const TkmGraph *g, const char *cuda_src);
```

Creates a runtime from an already parsed, optimized, and memory-planned graph
plus the CUDA source generated for that graph. This is the lower-level
equivalent of `tkm_load_string`:

- `g`: fully prepared graph; it is copied during the call.
- `cuda_src`: NUL-terminated generated CUDA source.
- Returns: owned runtime, or `NULL` only for allocation-level failure. Invalid
  graph/source data can instead surface as an NVRTC/backend diagnostic.

The caller retains ownership of `g` and `cuda_src` and may release them after
the call returns. Most host applications should use `tkm_load_file`,
`tkm_load_string`, or `tkm_load_buffer`.

#### `tkm_load_file`

```c
TkmRuntime *tkm_load_file(const char *tkm_path, int batch_size);
```

Loads a UTF-8/ASCII `.tkm` DSL file, parses and optimizes the graph, generates
CUDA source, compiles it with NVRTC, and allocates the runtime.

- `tkm_path`: readable model path.
- `batch_size`: positive fixed runtime batch size.
- Returns: owned runtime, or `NULL` on invalid arguments/file/model failure.

The file is read during the call; the caller may close or replace it after the
function returns.

#### `tkm_load_string`

```c
TkmRuntime *tkm_load_string(const char *dsl_text, int batch_size);
```

Equivalent to `tkm_load_buffer(dsl_text, strlen(dsl_text), batch_size)`.
The DSL must be NUL-terminated. Use `tkm_load_buffer` when the model is
received as a byte range or may contain embedded NUL bytes.

#### `tkm_load_buffer`

```c
TkmRuntime *tkm_load_buffer(const char *dsl_data, size_t dsl_size,
                            int batch_size);
```

Loads a model from an explicit byte range. Tokamak copies the range before
parsing, so the caller may release the source buffer after return.

- `dsl_data`: model bytes; must be non-null.
- `dsl_size`: non-zero byte count.
- `batch_size`: positive fixed runtime batch size.
- Returns: owned runtime, or `NULL`.

### 6.2 Direct execution

#### `tkm_runtime_step`

```c
float tkm_runtime_step(TkmRuntime *rt, const float *h_input,
                       const uint8_t *h_labels, float lr);
```

Runs one complete training step: host-to-device input/label transfer,
forward pass, backward pass, and SGD update.

- `h_input`: `B * H * W * C` host floats. May be `NULL` for internal/demo
  execution, but external callers should provide it.
- `h_labels`: `B` host labels. May be `NULL` only for graphs that do not read
  labels.
- `lr`: learning rate for this step.
- Returns: scalar loss copied back to host when input is supplied.

The call uses the runtime's fixed batch size. Do not pass a partial batch.

#### `tkm_runtime_eval`

```c
int tkm_runtime_eval(TkmRuntime *rt, const float *h_input,
                     const uint8_t *h_labels, int count);
```

Runs evaluation without applying an optimizer update and returns the number of
correct predictions.

- `h_input`: at least `count * H * W * C` host floats.
- `h_labels`: at least `count` labels.
- `count`: positive sample count, including a partial final batch.
- Returns: number correct, or `0` for invalid input/failure.

The result is an integer count, not a percentage. Compute the percentage in
the host application.

#### `tkm_runtime_forward`

```c
size_t tkm_runtime_forward(TkmRuntime *rt, const float *h_input,
                           float *h_output, size_t output_capacity);
```

Runs inference for exactly one runtime batch and converts the final GPU tensor
to host `float` values.

- `h_input`: `B * H * W * C` host floats.
- `h_output`: caller-owned output buffer.
- `output_capacity`: number of floats available in `h_output`, not bytes.
- Returns: number of floats written, or `0` if the graph is a loss graph,
  capacity is insufficient, or an argument/backend check fails.

Call `tkm_runtime_output_elements` before allocating the output buffer.

### 6.3 Runtime size queries

```c
size_t tkm_runtime_input_elements(const TkmRuntime *rt);
size_t tkm_runtime_output_elements(const TkmRuntime *rt);
size_t tkm_runtime_weight_bytes(const TkmRuntime *rt);
```

These queries return allocation sizes in **elements or bytes**, as named:

- `tkm_runtime_input_elements`: number of input `float` elements for one
  complete runtime batch.
- `tkm_runtime_output_elements`: number of output `float` elements for one
  complete runtime batch, or `0` for a loss graph.
- `tkm_runtime_weight_bytes`: exact size of the current opaque checkpoint
  representation, or `0` on invalid runtime/overflow.

The queries do not allocate memory and remain valid until the runtime is freed.

### 6.4 Weight checkpoints

#### File APIs

```c
int tkm_save_weights(TkmRuntime *rt, const char *bin_path);
int tkm_load_weights(TkmRuntime *rt, const char *bin_path);
```

These APIs save or load all learnable tensors in the runtime. Loading requires
the same tensor count and tensor sizes as the destination runtime; a
checkpoint from an incompatible graph is rejected.

The checkpoint format is an opaque Tokamak binary. It is not a general model
format and should not be modified, parsed, or assumed portable across
incompatible Tokamak builds, architectures, or floating-point ABIs.

#### Memory APIs

```c
int tkm_save_weights_buffer(TkmRuntime *rt, void *buffer, size_t capacity,
                            size_t *out_size);
int tkm_load_weights_buffer(TkmRuntime *rt, const void *buffer, size_t size);
```

`tkm_save_weights_buffer` supports a two-call allocation pattern:

```c
size_t bytes = tkm_runtime_weight_bytes(rt);
void *checkpoint = malloc(bytes);
size_t written = 0;
int ok = checkpoint &&
         tkm_save_weights_buffer(rt, checkpoint, bytes, &written);
```

When `buffer` is `NULL` or `capacity` is too small, the function returns `0`
and still writes the required size to `out_size` when that pointer is
non-null. On success, `written` equals the exact checkpoint size.

`tkm_load_weights_buffer` validates the checkpoint header, tensor count, and
each tensor length before copying weights to the GPU. The caller retains
ownership of the input bytes and may release them after return.

#### Training and inference lifecycle

Weights are loaded **after** the model DSL creates a runtime and **before** the
first forward, evaluation, or training call. Saving is normally done after a
training step or epoch:

```c
TkmRuntime *rt = tkm_load_file("models/resnet_mnist.tkm", 64);
if (!rt) return 1;

/* Optional: resume from a previous checkpoint before consuming data. */
if (resume_path && !tkm_load_weights(rt, resume_path)) {
    tkm_runtime_free(rt);
    return 1;
}

for (int epoch = 0; epoch < epochs; ++epoch) {
    float average_loss = 0.0f;
    if (!tkm_runtime_train_dataset(rt, next_batch, &loader,
                                   batches_per_epoch, 1e-3f,
                                   &average_loss)) {
        tkm_runtime_free(rt);
        return 1;
    }

    /* Save only after the epoch has completed successfully. */
    if (checkpoint_path && !tkm_save_weights(rt, checkpoint_path)) {
        tkm_runtime_free(rt);
        return 1;
    }
}

tkm_runtime_free(rt);
```

For pure inference, create the runtime, load the checkpoint, then call
`tkm_runtime_forward` for a tensor-output graph or `tkm_runtime_eval` for a
loss/classification graph:

```c
TkmRuntime *rt = tkm_load_file("models/llama3_style.tkm", 1);
if (!rt || !tkm_load_weights(rt, "llama3-trained.bin")) {
    tkm_runtime_free(rt);
    return 1;
}

size_t input_n = tkm_runtime_input_elements(rt);
size_t output_n = tkm_runtime_output_elements(rt);
float *input = calloc(input_n, sizeof(float));
float *logits = malloc(output_n * sizeof(float));
size_t written = tkm_runtime_forward(rt, input, logits, output_n);
if (written != output_n) {
    /* Report inference failure. */
}
free(logits);
free(input);
tkm_runtime_free(rt);
```

The current checkpoint contains the learnable master tensors and restores
their device copies. It does **not** contain optimizer velocity, random
generator state, or an epoch/data-loader cursor. Loading a checkpoint resumes
from the same model weights, but not from a bit-for-bit identical optimizer
trajectory. Store epoch, learning-rate schedule, RNG seed, and dataset cursor
in the host application's own metadata if exact job resumption is required.

Do not load a checkpoint from one graph into another graph with a different
learnable tensor count or shape. The runtime rejects those mismatches.

### 6.5 Dataset callbacks

```c
typedef int (*TkmDatasetNext)(void *user, float *h_input,
                              uint8_t *h_labels, size_t capacity,
                              size_t *out_count);
```

Tokamak allocates reusable host buffers and invokes the callback synchronously.
The callback must:

1. write no more than `capacity` samples;
2. write `*out_count` with the number of samples produced;
3. return a positive value when a batch was produced;
4. return `0` at end of dataset;
5. return a negative value on loader/I/O/decode error.

On a callback error, stop producing data and return failure from the dataset
helper. `user` is passed through unchanged and is owned by the caller.

#### `tkm_runtime_train_dataset`

```c
int tkm_runtime_train_dataset(TkmRuntime *rt, TkmDatasetNext next_batch,
                              void *user, size_t max_batches, float lr,
                              float *out_average_loss);
```

Consumes batches until end-of-dataset or `max_batches`. `max_batches == 0`
means no explicit batch limit. Every successful training callback invocation
must return exactly the runtime batch size. A short final training batch is an
error, not implicit padding.

Returns non-zero only if at least one batch was trained and no callback/runtime
error occurred. `out_average_loss` receives the arithmetic mean of the
returned per-batch losses when non-null.

#### `tkm_runtime_eval_dataset`

```c
int tkm_runtime_eval_dataset(TkmRuntime *rt, TkmDatasetNext next_batch,
                             void *user, size_t max_batches,
                             int *out_correct, size_t *out_samples);
```

Consumes evaluation batches until end-of-dataset or `max_batches`. Each
callback may return `1..B` samples, including a short final batch.

Returns non-zero only if at least one batch was evaluated and no callback/runtime
error occurred. `out_correct` receives the total number of correct predictions
and `out_samples` receives the total number of evaluated samples.

### 6.6 Runtime destruction

```c
void tkm_runtime_free(TkmRuntime *rt);
```

Synchronizes and releases the runtime, CUDA graph objects, generated module,
device buffers, optimizer state, and host bookkeeping. Passing `NULL` is safe.
Do not use any pointer or callback state associated with the runtime after this
call.

## 7. Graph/compiler APIs

The following exported APIs are available when an application wants to
construct and inspect a `TkmGraph` directly instead of using `tkm_load_*`:

```c
int   tkm_verify_registry(void);
void  tkm_parse_dsl(TkmGraph *g, const char *dsl_text, int batch_size);
void  tkm_infer_shapes(TkmGraph *g);
void  tkm_optimize_and_bind(TkmGraph *g);
void  tkm_plan_memory(TkmGraph *g);
void  tkm_dump_summary(const TkmGraph *g, FILE *out);
char *tkm_codegen_cuda_string(const TkmGraph *g);
```

The expected order is:

```text
zero-initialize TkmGraph
-> tkm_parse_dsl
-> tkm_infer_shapes / tkm_optimize_and_bind
-> tkm_plan_memory
-> tkm_codegen_cuda_string
-> tkm_runtime_ignite
```

`tkm_codegen_cuda_string` returns heap-allocated CUDA source; release it with
the C runtime's `free`. `tkm_runtime_ignite` copies the graph and compiles the
provided generated source, so the caller may release the source and graph
after it returns. These lower-level APIs expose the internal graph capacity
limits in `tkm.h` and are more sensitive to malformed or manually edited IR
than the `tkm_load_*` helpers.

The registry inspection helpers (`tkm_find_spec_by_*`,
`tkm_get_spec_*`, `tkm_eval_math_expr`, `tkm_get_fusion_rules`, and RNG name
helpers) are intended for tooling and diagnostics. Their returned strings and
specification pointers are library-owned and must not be freed.

## 8. Numerical gradient probe APIs

```c
float tkm_get_grad_sample(TkmRuntime *rt, int node_id, int elem_idx);
void  tkm_perturb_tensor(TkmRuntime *rt, int node_id, int elem_idx,
                         float delta);
float tkm_eval_current_loss(TkmRuntime *rt);
```

These functions support Tokamak's finite-difference checks. `node_id` and
`elem_idx` refer to internal graph tensor storage, not stable model parameter
names. They are diagnostic APIs and should not be used as a model-serving
contract.

## 9. Complete minimal example

```c
#include "tkm.h"
#include <stdint.h>
#include <stdlib.h>

int main(void) {
    TkmRuntime *rt = tkm_load_file("models/resnet_mnist.tkm", 64);
    if (!rt) return 1;

    size_t input_count = tkm_runtime_input_elements(rt);
    float *images = (float *)calloc(input_count, sizeof(*images));
    uint8_t *labels = (uint8_t *)calloc(64, sizeof(*labels));
    if (!images || !labels) {
        free(images);
        free(labels);
        tkm_runtime_free(rt);
        return 1;
    }

    float loss = tkm_runtime_step(rt, images, labels, 1e-3f);
    (void)loss;

    free(images);
    free(labels);
    tkm_runtime_free(rt);
    return 0;
}
```

## 10. External loader example

```c
static int next_batch(void *user, float *input, uint8_t *labels,
                      size_t capacity, size_t *count) {
    MyLoader *loader = (MyLoader *)user;
    return my_loader_read(loader, input, labels, capacity, count);
}

TkmRuntime *rt = tkm_load_buffer(model_bytes, model_size, 64);
if (!rt) return 1;

size_t checkpoint_bytes = tkm_runtime_weight_bytes(rt);
void *checkpoint = external_checkpoint_bytes;
if (external_checkpoint_size != checkpoint_bytes ||
    !tkm_load_weights_buffer(rt, checkpoint, external_checkpoint_size)) {
    tkm_runtime_free(rt);
    return 1;
}

float average_loss = 0.0f;
if (!tkm_runtime_train_dataset(rt, next_batch, &loader, 0,
                               1e-3f, &average_loss)) {
    tkm_runtime_free(rt);
    return 1;
}

tkm_runtime_free(rt);
```

## 11. MNIST IDX example

`examples/mnist_idx.c` demonstrates a complete external C caller. It reads
standard uncompressed IDX images and labels, validates headers and dimensions,
converts image bytes from `[0, 255]` to host floats in `[0, 1]`, then calls
`tkm_load_file`, `tkm_runtime_step`, and `tkm_runtime_eval`.

```cmd
mnist_idx.exe <train-images-idx3> <train-labels-idx1> ^
  <t10k-images-idx3> <t10k-labels-idx1> [batch=64] [epochs=1]
```

Training requires complete fixed-size batches. Evaluation accepts a partial
final batch. Any reported accuracy depends on initialization, hardware, and
training settings and is not a guaranteed benchmark.

## 12. FFI examples in other host languages

The ABI is intentionally C-shaped: opaque runtime handles, fixed-width
integer types, `size_t` counts, and caller-owned flat buffers. The same
ownership rules apply to every language binding:

1. keep the native library loaded while the runtime exists;
2. allocate input/output buffers in host memory;
3. keep callback/user state alive for the entire dataset operation;
4. call `tkm_runtime_free` exactly once.

These snippets use the dynamic library name `tkm.dll` on Windows and
`libtkm.so` on Linux. Adjust the path and calling convention for the target
platform.

### C: inference and one backward/update step

```c
#include "tkm.h"
#include <stdlib.h>

int main(void) {
    TkmRuntime *rt = tkm_load_file("models/resnet_mnist.tkm", 1);
    if (!rt) return 1;

    size_t input_n = tkm_runtime_input_elements(rt);
    size_t output_n = tkm_runtime_output_elements(rt);
    float *input = (float *)calloc(input_n, sizeof(float));
    float *output = (float *)calloc(output_n ? output_n : 1, sizeof(float));
    uint8_t label = 0;
    if (!input || !output) return 1;

    /* Pure inference: only valid for a graph ending in a tensor output. */
    if (output_n != 0)
        tkm_runtime_forward(rt, input, output, output_n);

    /*
     * Training path: this performs forward + backward + SGD update.
     * The public ABI does not expose a backward-only/no-update call.
     */
    tkm_runtime_step(rt, input, &label, 1e-3f);
    if (!tkm_save_weights(rt, "resnet-after-step.bin")) return 1;

    free(output);
    free(input);
    tkm_runtime_free(rt);
    return 0;
}
```

### C++: RAII wrapper around the opaque handle

```cpp
#include "tkm.h"
#include <vector>

struct Runtime {
    TkmRuntime *p;
    explicit Runtime(const char *model, int batch)
        : p(tkm_load_file(model, batch)) {}
    ~Runtime() { tkm_runtime_free(p); }
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;
};

int main() {
    Runtime model("models/llama3_style.tkm", 1);
    if (!model.p) return 1;
    if (!tkm_load_weights(model.p, "llama3-trained.bin")) return 1;

    const size_t input_n = tkm_runtime_input_elements(model.p);
    const size_t output_n = tkm_runtime_output_elements(model.p);
    std::vector<float> embeddings(input_n);
    std::vector<float> logits(output_n);

    const size_t written =
        tkm_runtime_forward(model.p, embeddings.data(), logits.data(),
                            logits.size());
    if (!tkm_save_weights(model.p, "llama3-after-inference.bin")) return 1;
    return written == output_n ? 0 : 1;
}
```

### Python: `ctypes` inference and training call

`ctypes` must use `c_size_t` for all `size_t` arguments and results. Do not
let Python's garbage collector move or release an array while a native call
is using it; keeping the array in a local variable for the call is sufficient
for these synchronous APIs.

```python
import ctypes as C

lib = C.CDLL("./tkm.dll")          # "./libtkm.so" on Linux
Runtime = C.c_void_p

lib.tkm_load_file.argtypes = [C.c_char_p, C.c_int]
lib.tkm_load_file.restype = Runtime
lib.tkm_runtime_input_elements.argtypes = [Runtime]
lib.tkm_runtime_input_elements.restype = C.c_size_t
lib.tkm_runtime_output_elements.argtypes = [Runtime]
lib.tkm_runtime_output_elements.restype = C.c_size_t
lib.tkm_runtime_forward.argtypes = [
    Runtime, C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_size_t
]
lib.tkm_runtime_forward.restype = C.c_size_t
lib.tkm_runtime_step.argtypes = [
    Runtime, C.POINTER(C.c_float), C.POINTER(C.c_uint8), C.c_float
]
lib.tkm_runtime_step.restype = C.c_float
lib.tkm_load_weights.argtypes = [Runtime, C.c_char_p]
lib.tkm_load_weights.restype = C.c_int
lib.tkm_save_weights.argtypes = [Runtime, C.c_char_p]
lib.tkm_save_weights.restype = C.c_int
lib.tkm_runtime_free.argtypes = [Runtime]
lib.tkm_runtime_free.restype = None

rt = lib.tkm_load_file(b"models/resnet_mnist.tkm", 1)
if not rt:
    raise RuntimeError("Tokamak runtime creation failed")

try:
    if not lib.tkm_load_weights(rt, b"resnet-start.bin"):
        raise RuntimeError("checkpoint load failed")
    input_n = lib.tkm_runtime_input_elements(rt)
    output_n = lib.tkm_runtime_output_elements(rt)
    x = (C.c_float * input_n)()

    if output_n:
        y = (C.c_float * output_n)()
        written = lib.tkm_runtime_forward(rt, x, y, output_n)
        assert written == output_n

    labels = (C.c_uint8 * 1)(0)
    loss = lib.tkm_runtime_step(rt, x, labels, C.c_float(1e-3))
    if not lib.tkm_save_weights(rt, b"resnet-after-step.bin"):
        raise RuntimeError("checkpoint save failed")
finally:
    lib.tkm_runtime_free(rt)
```

### Rust: explicit FFI declarations

```rust
use std::{ffi::CString, os::raw::{c_char, c_float, c_int, c_uchar}, ptr};

#[repr(C)]
struct TkmRuntime { _private: [u8; 0] }

#[link(name = "tkm")]
extern "C" {
    fn tkm_load_file(path: *const c_char, batch: c_int) -> *mut TkmRuntime;
    fn tkm_runtime_input_elements(rt: *const TkmRuntime) -> usize;
    fn tkm_runtime_output_elements(rt: *const TkmRuntime) -> usize;
    fn tkm_runtime_forward(rt: *mut TkmRuntime, input: *const c_float,
                           output: *mut c_float, capacity: usize) -> usize;
    fn tkm_load_weights(rt: *mut TkmRuntime, path: *const c_char) -> c_int;
    fn tkm_save_weights(rt: *mut TkmRuntime, path: *const c_char) -> c_int;
    fn tkm_runtime_free(rt: *mut TkmRuntime);
}

fn main() {
    let path = CString::new("models/llama3_style.tkm").unwrap();
    unsafe {
        let rt = tkm_load_file(path.as_ptr(), 1);
        assert!(!rt.is_null());
        let checkpoint = CString::new("llama3-start.bin").unwrap();
        assert_eq!(tkm_load_weights(rt, checkpoint.as_ptr()), 1);
        let input = vec![0.0f32; tkm_runtime_input_elements(rt)];
        let mut output = vec![0.0f32; tkm_runtime_output_elements(rt)];
        let written = tkm_runtime_forward(
            rt, input.as_ptr(), output.as_mut_ptr(), output.len());
        assert_eq!(written, output.len());
        let saved = CString::new("llama3-after-forward.bin").unwrap();
        assert_eq!(tkm_save_weights(rt, saved.as_ptr()), 1);
        tkm_runtime_free(rt);
    }
}
```

For production Rust code, wrap the raw pointer in a `Drop` type and mark the
wrapper `!Send`/`!Sync` unless access is externally synchronized. The runtime
is not safe for concurrent calls.

### C#: P/Invoke declarations

```csharp
using System;
using System.Runtime.InteropServices;

static class Tokamak {
    [DllImport("tkm.dll", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr tkm_load_file(
        [MarshalAs(UnmanagedType.LPStr)] string path, int batch);

    [DllImport("tkm.dll", CallingConvention = CallingConvention.Cdecl)]
    internal static extern nuint tkm_runtime_input_elements(IntPtr rt);

    [DllImport("tkm.dll", CallingConvention = CallingConvention.Cdecl)]
    internal static extern nuint tkm_runtime_output_elements(IntPtr rt);

    [DllImport("tkm.dll", CallingConvention = CallingConvention.Cdecl)]
    internal static extern nuint tkm_runtime_forward(
        IntPtr rt, float[] input, float[] output, nuint capacity);

    [DllImport("tkm.dll", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int tkm_load_weights(
        IntPtr rt, [MarshalAs(UnmanagedType.LPStr)] string path);

    [DllImport("tkm.dll", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int tkm_save_weights(
        IntPtr rt, [MarshalAs(UnmanagedType.LPStr)] string path);

    [DllImport("tkm.dll", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void tkm_runtime_free(IntPtr rt);
}

using var model = new RuntimeHandle("models/resnet_mnist.tkm", 1);
if (Tokamak.tkm_load_weights(model.Ptr, "resnet-start.bin") == 0)
    throw new InvalidOperationException("checkpoint load failed");
float[] input = new float[(int)Tokamak.tkm_runtime_input_elements(model.Ptr)];
int outputCount = checked((int)Tokamak.tkm_runtime_output_elements(model.Ptr));
float[] output = new float[Math.Max(outputCount, 1)];
if (outputCount != 0)
    Tokamak.tkm_runtime_forward(model.Ptr, input, output, (nuint)outputCount);
if (Tokamak.tkm_save_weights(model.Ptr, "resnet-after-forward.bin") == 0)
    throw new InvalidOperationException("checkpoint save failed");
```

The `RuntimeHandle` in a real C# binding should implement `IDisposable` and
call `tkm_runtime_free` from `Dispose`; a finalizer is only a safety net.
`nuint` is available on modern .NET and matches native `size_t`.

## 13. Model patterns: inference versus backward/update

The three checked-in models intentionally cover different kernel families.
Use them as templates for how a host loader should shape its buffers:

| Model | Input graph shape | End of graph | Inference API | Training/backward API |
|---|---:|---|---|---|
| `resnet_mnist.tkm` | `28x28x1` | `FC(10) -> SoftmaxCE` | not a tensor-output graph; use `tkm_runtime_eval` | `tkm_runtime_step` with `B` images + `B` labels |
| `exotic_llama.tkm` | `1x256x512` | `FC(10) -> SoftmaxCE` | not a tensor-output graph; use `tkm_runtime_eval` | `tkm_runtime_step` with `B` embedding sequences + `B` labels |
| `llama3_style.tkm` | `1x128x4096` | `FC(128256)` logits | `tkm_runtime_forward` | no loss sink; add a host loss or a training graph before using `tkm_runtime_step` |

`tkm_runtime_step` is the public backward path: it executes the generated
forward and backward kernels and then applies SGD. Tokamak does not currently
export a backward-only function that returns gradients without updating
weights. For a numerical gradient probe, use
`tkm_get_grad_sample`, `tkm_perturb_tensor`, and `tkm_eval_current_loss`; those
are diagnostic tensor-index APIs, not stable named-parameter APIs.

The probe pattern is:

```c
/* rt must be a loss graph, such as resnet_mnist.tkm. */
float analytical = tkm_get_grad_sample(rt, node_id, element);
float epsilon = 1e-3f;
tkm_perturb_tensor(rt, node_id, element, +epsilon);
float plus = tkm_eval_current_loss(rt);
tkm_perturb_tensor(rt, node_id, element, -2.0f * epsilon);
float minus = tkm_eval_current_loss(rt);
tkm_perturb_tensor(rt, node_id, element, +epsilon); /* restore */
float numerical = (plus - minus) / (2.0f * epsilon);
```

This compares an analytical gradient already present in the runtime's
backward arena with a finite-difference estimate. The caller must establish
the same input/label state before each loss evaluation. `node_id` and
`element` are internal graph coordinates and should be discovered from
tooling/debug output rather than persisted as a long-term checkpoint ABI.

### 13.1 `resnet_mnist.tkm`: spatial CNN training and evaluation

The graph begins with augmentation, then two convolution/normalization/pooling
stages, a residual branch, and a gated fully connected head:

```text
NHWC [B,28,28,1]
  -> RandShift/RandRotate/RandScale/RandElastic/RandNoise
  -> Conv32 -> BN -> SiLU -> MixPool
  -> Conv64 -> BN -> SiLU -> MixPool
  -> residual Conv64 -> BN -> SiLU + skip
  -> Flatten -> (FC512 -> SiLU) * (FC512)
  -> LN -> SiLU -> Drop -> FC10 -> SoftmaxCE
```

```c
TkmRuntime *rt = tkm_load_file("models/resnet_mnist.tkm", 64);
size_t image_count = tkm_runtime_input_elements(rt); /* 64*28*28*1 */
float *images = calloc(image_count, sizeof(float));
uint8_t labels[64] = {0};

/* Forward + backward + SGD. */
float loss = tkm_runtime_step(rt, images, labels, 1e-3f);

/* Evaluation can consume a non-multiple-of-64 dataset. */
int correct = tkm_runtime_eval(rt, test_images, test_labels, test_count);

free(images);
tkm_runtime_free(rt);
```

The augmentation and dropout operators are runtime-aware and are bypassed or
adjusted during evaluation. This model therefore demonstrates why a host
should use `tkm_runtime_eval` rather than calling a training step with a zero
learning rate when it wants evaluation semantics.

### 13.2 `exotic_llama.tkm`: composable attention and FFN training

This model has four repeated transformer-like layers. Each layer combines
RMSNorm, Q/K/V projections, QK normalization, RoPE, GQA attention, residual
addition, a cosine-gated twin-branch FFN, DyT, and dropout:

```text
NHWC [B,1,256,512]
  -> repeat 4:
       RMSNorm -> { Q:FC512->QKNorm->RoPE,
                    K:FC128->QKNorm->RoPE,
                    V:FC128 }
       -> GQA(q=8,kv=2,d=64) -> FC512 -> SoftCap -> Drop
       -> + skip
       RMSNorm -> { gate:FC1024->SoftCap->CosGate->SiLU,
                    up:FC1024->ReLU2 }
       -> Mul -> FC512 -> DyT -> Drop
       -> + skip
  -> RMSNorm -> FC10 -> SoftmaxCE
```

It is a classification/training graph, not a decoder that emits token logits.
The host supplies precomputed `1x256x512` embeddings for each sample; the
tokenizer and embedding table remain outside Tokamak. Use the same
`tkm_runtime_step`/`tkm_runtime_eval` pattern as the CNN, but allocate
`B*1*256*512` floats per batch.

### 13.3 `llama3_style.tkm`: pure decoder inference

This graph is deliberately a tensor-output graph:

```text
NHWC [B,1,128,4096] precomputed embeddings
  -> repeat 32:
       RMSNorm -> Q/K/V FC -> RoPE -> causal GQA(q=32,kv=8,d=128)
       -> FC4096 -> + skip
       RMSNorm -> (FC14336 -> SiLU) * FC14336 -> FC4096 -> + skip
  -> RMSNorm -> FC128256 logits
```

The host owns tokenization, embedding lookup, sampling, and KV-cache policy.
For `B == 1`, the output contains `128 * 128256` floats. The final token's
logits begin at:

```c
size_t last = (128 - 1) * 128256;
```

```c
TkmRuntime *rt = tkm_load_file("models/llama3_style.tkm", 1);
size_t input_n = tkm_runtime_input_elements(rt);   /* 1*128*4096 */
size_t logits_n = tkm_runtime_output_elements(rt); /* 128*128256 */
float *embeddings = calloc(input_n, sizeof(float));
float *logits = malloc(logits_n * sizeof(float));

size_t written = tkm_runtime_forward(rt, embeddings, logits, logits_n);
if (written == logits_n) {
    float *next_token_logits = logits + (127u * 128256u);
    sample_next_token(next_token_logits, 128256);
}

free(logits);
free(embeddings);
tkm_runtime_free(rt);
```

This model has no `SoftmaxCE` loss sink, so `tkm_runtime_step` is not a valid
language-model training entry point for it. To train it through the current
ABI, define a training DSL with a loss sink (or add a host-side training
integration that supplies the required loss/gradient graph); do not infer
backward support merely from the presence of backward kernels for individual
operators.
