CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

CFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
RT_CORE_DIR := runtime-core
RT_CORE_SRC := $(RT_CORE_DIR)/src/runtime.c
RT_CORE_HDR := $(RT_CORE_DIR)/include/rt_runtime.h
RT_CPPFLAGS := -I$(RT_CORE_DIR)/include
DS4_MODEL_DIR := models/deepseek-v4-flash
DS4_ENGINE_SRC := $(DS4_MODEL_DIR)/engine/ds4.c
DS4_PUBLIC_HDR := $(DS4_MODEL_DIR)/include/ds4.h
DS4_RUNTIME_SRC := $(DS4_MODEL_DIR)/runtime/ds4_runtime.c
DS4_RUNTIME_HDR := $(DS4_MODEL_DIR)/include/ds4_runtime.h
DS4_GPU_HDR := $(DS4_MODEL_DIR)/backends/ds4_gpu.h
DS4_METAL_SRC := $(DS4_MODEL_DIR)/backends/metal/ds4_metal.m
DS4_CUDA_SRC := $(DS4_MODEL_DIR)/backends/cuda/ds4_cuda.cu
DS4_CUDA_TABLES := $(DS4_MODEL_DIR)/backends/cuda/ds4_iq2_tables_cuda.inc
DS4_CPPFLAGS := $(RT_CPPFLAGS) -I$(DS4_MODEL_DIR)/include -I$(DS4_MODEL_DIR)/backends -I.
METAL_SRCS := $(wildcard $(DS4_MODEL_DIR)/backends/metal/kernels/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = rt_runtime.o ds4_runtime.o ds4.o ds4_metal.o
CPU_CORE_OBJS = rt_runtime.o ds4_runtime_cpu.o ds4_cpu.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CORE_OBJS = rt_runtime.o ds4_runtime.o ds4.o ds4_cuda.o
CPU_CORE_OBJS = rt_runtime.o ds4_runtime_cpu.o ds4_cpu.o
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, and ./ds4-eval"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, and ./ds4-eval"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, and ./ds4-eval"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH=

cuda-generic:
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH="$(CUDA_ARCH)"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

rt_runtime.o: $(RT_CORE_SRC) $(RT_CORE_HDR)
	$(CC) $(CFLAGS) $(RT_CPPFLAGS) -c -o $@ $(RT_CORE_SRC)

ds4_runtime.o: $(DS4_RUNTIME_SRC) $(DS4_RUNTIME_HDR) $(DS4_PUBLIC_HDR) $(RT_CORE_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -c -o $@ $(DS4_RUNTIME_SRC)

ds4_runtime_cpu.o: $(DS4_RUNTIME_SRC) $(DS4_RUNTIME_HDR) $(DS4_PUBLIC_HDR) $(RT_CORE_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -DDS4_NO_GPU -c -o $@ $(DS4_RUNTIME_SRC)

ds4.o: $(DS4_ENGINE_SRC) $(DS4_PUBLIC_HDR) $(DS4_GPU_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -c -o $@ $(DS4_ENGINE_SRC)

ds4_cli.o: ds4_cli.c $(DS4_PUBLIC_HDR) linenoise.h
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c $(DS4_PUBLIC_HDR) rax.h
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c $(DS4_PUBLIC_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c $(DS4_PUBLIC_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -c -o $@ ds4_eval.c

ds4_test.o: tests/ds4_test.c ds4_server.c $(DS4_PUBLIC_HDR) $(DS4_RUNTIME_HDR) $(DS4_GPU_HDR) $(RT_CORE_HDR) rax.h
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c $(DS4_GPU_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: $(DS4_ENGINE_SRC) $(DS4_PUBLIC_HDR) $(DS4_GPU_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -DDS4_NO_GPU -c -o $@ $(DS4_ENGINE_SRC)

ds4_cli_cpu.o: ds4_cli.c $(DS4_PUBLIC_HDR) linenoise.h
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c $(DS4_PUBLIC_HDR) rax.h
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c $(DS4_PUBLIC_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c $(DS4_PUBLIC_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_metal.o: $(DS4_METAL_SRC) $(DS4_GPU_HDR) $(DS4_PUBLIC_HDR) $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) $(DS4_CPPFLAGS) -c -o $@ $(DS4_METAL_SRC)

ds4_cuda.o: $(DS4_CUDA_SRC) $(DS4_GPU_HDR) $(DS4_CUDA_TABLES)
	$(NVCC) $(NVCCFLAGS) $(DS4_CPPFLAGS) -c -o $@ $(DS4_CUDA_SRC)

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4_cpu ds4_native ds4_server_test ds4_test *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
