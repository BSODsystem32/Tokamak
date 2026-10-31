NVCC   := nvcc
ARCH   ?= -arch=sm_86
CFLAGS := -O3 -Iinclude
LDFLAGS := -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lnvrtc -lcuda -lcublas

LIB_SRCS := src/tkm_registry.c \
        src/tkm_core.c \
        src/tkm_codegen.c \
        src/tkm_gpu.cu

SRCS := $(LIB_SRCS) src/main.c

all: tkm libtkm.so

tkm: $(SRCS) include/tkm.h
	$(NVCC) $(CFLAGS) $(ARCH) $(SRCS) $(LDFLAGS) -o tkm

libtkm.so: $(LIB_SRCS) include/tkm.h
	$(NVCC) $(CFLAGS) -Xcompiler -fPIC -shared $(ARCH) $(LIB_SRCS) $(LDFLAGS) -o libtkm.so

mnist_idx: examples/mnist_idx.c include/tkm.h libtkm.so
	$(CC) -O2 -Iinclude examples/mnist_idx.c -L. -ltkm -Wl,-rpath,'$$ORIGIN' -o mnist_idx

clean:
	rm -f tkm mnist_idx *.so *.cu.out *.ptx

.PHONY: all clean mnist_idx