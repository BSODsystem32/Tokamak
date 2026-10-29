NVCC   := nvcc
ARCH   ?= -arch=sm_86
CFLAGS := -O3 -Iinclude
LDFLAGS := -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lnvrtc -lcuda -lcublas

SRCS := src/tkm_registry.c \
        src/tkm_core.c \
        src/tkm_codegen.c \
        src/tkm_gpu.cu \
        src/main.c

tkm: $(SRCS) include/tkm.h
	$(NVCC) $(CFLAGS) $(ARCH) $(SRCS) $(LDFLAGS) -o tkm

clean:
	rm -f tkm *.cu.out *.ptx