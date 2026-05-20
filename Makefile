CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

CFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread -lz
DS4_MEDIA_AV_SRC := ds4_media_av.m
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
QWEN36_MODEL_DIR := models/qwen3.6-27b
QWEN36_RUNTIME_SRC := $(QWEN36_MODEL_DIR)/runtime/qwen36_runtime.c
QWEN36_RUNTIME_HDR := $(QWEN36_MODEL_DIR)/include/qwen36_runtime.h
QWEN36_METAL_SRC := $(QWEN36_MODEL_DIR)/backends/metal/qwen36_metal.m
QWEN36_BACKEND_HDR := $(QWEN36_MODEL_DIR)/backends/qwen36_metal.h
QWEN36_CPPFLAGS := $(RT_CPPFLAGS) -I$(QWEN36_MODEL_DIR)/include -I$(QWEN36_MODEL_DIR)/backends
MISTRAL35_MODEL_DIR := models/mistral-medium-3.5-128b
MISTRAL35_RUNTIME_SRC := $(MISTRAL35_MODEL_DIR)/runtime/mistral35_runtime.c
MISTRAL35_RUNTIME_HDR := $(MISTRAL35_MODEL_DIR)/include/mistral35_runtime.h
MISTRAL35_CPPFLAGS := $(RT_CPPFLAGS) -I$(MISTRAL35_MODEL_DIR)/include
MODEL_TEST_CPPFLAGS := $(DS4_CPPFLAGS) -I$(QWEN36_MODEL_DIR)/include -I$(QWEN36_MODEL_DIR)/backends -I$(MISTRAL35_MODEL_DIR)/include
SERVER_CPPFLAGS := $(MODEL_TEST_CPPFLAGS)
METAL_SRCS := $(wildcard $(DS4_MODEL_DIR)/backends/metal/kernels/*.metal)

ifeq ($(UNAME_S),Darwin)
DARWIN_MEDIA_LDLIBS := $(LDLIBS) -framework AVFoundation -framework CoreFoundation -framework CoreGraphics -framework CoreMedia -framework CoreVideo -framework Foundation -framework ImageIO
METAL_LDLIBS := $(DARWIN_MEDIA_LDLIBS) -framework Metal
CPU_TEST_LDLIBS := $(DARWIN_MEDIA_LDLIBS)
CORE_OBJS = rt_runtime.o ds4_runtime.o qwen36_runtime.o qwen36_metal.o mistral35_runtime.o ds4.o ds4_metal.o
CPU_CORE_OBJS = rt_runtime.o ds4_runtime_cpu.o qwen36_runtime_cpu.o mistral35_runtime_cpu.o ds4_cpu.o
MEDIA_OBJS = ds4_media_av.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -lz -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CPU_TEST_LDLIBS := $(LDLIBS)
CORE_OBJS = rt_runtime.o ds4_runtime.o qwen36_runtime.o mistral35_runtime.o ds4.o ds4_cuda.o
CPU_CORE_OBJS = rt_runtime.o ds4_runtime_cpu.o qwen36_runtime_cpu.o mistral35_runtime_cpu.o ds4_cpu.o
METAL_LDLIBS := $(LDLIBS)
MEDIA_OBJS =
endif

.PHONY: all help clean test qwen36-gates qwen36-gates-strict qwen36-gates-cpu qwen36-text-gates qwen36-text-gates-strict qwen36-text-gates-cpu cpu cuda cuda-spark cuda-generic cuda-regression

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, and ./ds4-eval"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, and ./ds4-eval"
	@echo "  make test         Build and run tests"
	@echo "  make qwen36-gates       Build and run Qwen full local gates"
	@echo "  make qwen36-gates-strict  Require pinned Qwen official vectors"
	@echo "  make qwen36-gates-cpu   Run Qwen full local gates with DS4_NO_GPU"
	@echo "  make qwen36-text-gates  Build and run Qwen text-only gates"
	@echo "  make qwen36-text-gates-strict  Require pinned Qwen official vectors"
	@echo "  make qwen36-text-gates-cpu  Run Qwen text gates with DS4_NO_GPU"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS) $(MEDIA_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(MEDIA_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o linenoise.o rax.o $(CPU_CORE_OBJS) $(MEDIA_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(DARWIN_MEDIA_LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(MEDIA_OBJS) $(DARWIN_MEDIA_LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(DARWIN_MEDIA_LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(DARWIN_MEDIA_LDLIBS)

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
	@echo "  make qwen36-gates        Build and run Qwen full local gates"
	@echo "  make qwen36-gates-strict Require pinned Qwen official vectors"
	@echo "  make qwen36-gates-cpu    Run Qwen full local gates with DS4_NO_GPU"
	@echo "  make qwen36-text-gates   Build and run Qwen text-only gates"
	@echo "  make qwen36-text-gates-strict  Require pinned Qwen official vectors"
	@echo "  make qwen36-text-gates-cpu  Run Qwen text gates with DS4_NO_GPU"
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

qwen36_runtime.o: $(QWEN36_RUNTIME_SRC) $(QWEN36_RUNTIME_HDR) $(QWEN36_BACKEND_HDR) $(RT_CORE_HDR)
	$(CC) $(CFLAGS) $(QWEN36_CPPFLAGS) -c -o $@ $(QWEN36_RUNTIME_SRC)

qwen36_runtime_cpu.o: $(QWEN36_RUNTIME_SRC) $(QWEN36_RUNTIME_HDR) $(QWEN36_BACKEND_HDR) $(RT_CORE_HDR)
	$(CC) $(CFLAGS) $(QWEN36_CPPFLAGS) -DDS4_NO_GPU -c -o $@ $(QWEN36_RUNTIME_SRC)

mistral35_runtime.o: $(MISTRAL35_RUNTIME_SRC) $(MISTRAL35_RUNTIME_HDR) $(RT_CORE_HDR)
	$(CC) $(CFLAGS) $(MISTRAL35_CPPFLAGS) -c -o $@ $(MISTRAL35_RUNTIME_SRC)

mistral35_runtime_cpu.o: $(MISTRAL35_RUNTIME_SRC) $(MISTRAL35_RUNTIME_HDR) $(RT_CORE_HDR)
	$(CC) $(CFLAGS) $(MISTRAL35_CPPFLAGS) -c -o $@ $(MISTRAL35_RUNTIME_SRC)

ds4.o: $(DS4_ENGINE_SRC) $(DS4_PUBLIC_HDR) $(DS4_GPU_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -c -o $@ $(DS4_ENGINE_SRC)

ds4_cli.o: ds4_cli.c $(DS4_PUBLIC_HDR) linenoise.h
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c $(DS4_PUBLIC_HDR) $(DS4_RUNTIME_HDR) $(QWEN36_RUNTIME_HDR) $(MISTRAL35_RUNTIME_HDR) rax.h
	$(CC) $(CFLAGS) $(SERVER_CPPFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c $(DS4_PUBLIC_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c $(DS4_PUBLIC_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -c -o $@ ds4_eval.c

ds4_test.o: tests/ds4_test.c ds4_server.c $(DS4_PUBLIC_HDR) $(DS4_RUNTIME_HDR) $(QWEN36_RUNTIME_HDR) $(QWEN36_BACKEND_HDR) $(MISTRAL35_RUNTIME_HDR) $(DS4_GPU_HDR) $(RT_CORE_HDR) rax.h
	$(CC) $(CFLAGS) $(MODEL_TEST_CPPFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_test_cpu.o: tests/ds4_test.c ds4_server.c $(DS4_PUBLIC_HDR) $(DS4_RUNTIME_HDR) $(QWEN36_RUNTIME_HDR) $(QWEN36_BACKEND_HDR) $(MISTRAL35_RUNTIME_HDR) $(DS4_GPU_HDR) $(RT_CORE_HDR) rax.h
	$(CC) $(CFLAGS) $(MODEL_TEST_CPPFLAGS) -DDS4_NO_GPU -Wno-unused-function -c -o $@ tests/ds4_test.c

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

ds4_server_cpu.o: ds4_server.c $(DS4_PUBLIC_HDR) $(DS4_RUNTIME_HDR) $(QWEN36_RUNTIME_HDR) $(MISTRAL35_RUNTIME_HDR) rax.h
	$(CC) $(CFLAGS) $(SERVER_CPPFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c $(DS4_PUBLIC_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c $(DS4_PUBLIC_HDR)
	$(CC) $(CFLAGS) $(DS4_CPPFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_metal.o: $(DS4_METAL_SRC) $(DS4_GPU_HDR) $(DS4_PUBLIC_HDR) $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) $(DS4_CPPFLAGS) -c -o $@ $(DS4_METAL_SRC)

qwen36_metal.o: $(QWEN36_METAL_SRC) $(QWEN36_BACKEND_HDR) $(RT_CORE_HDR)
	$(CC) $(OBJCFLAGS) $(QWEN36_CPPFLAGS) -c -o $@ $(QWEN36_METAL_SRC)

ds4_media_av.o: $(DS4_MEDIA_AV_SRC)
	$(CC) $(OBJCFLAGS) -c -o $@ $(DS4_MEDIA_AV_SRC)

ds4_cuda.o: $(DS4_CUDA_SRC) $(DS4_GPU_HDR) $(DS4_CUDA_TABLES)
	$(NVCC) $(NVCCFLAGS) $(DS4_CPPFLAGS) -c -o $@ $(DS4_CUDA_SRC)

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: ds4_test.o rax.o $(CORE_OBJS) $(MEDIA_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(MEDIA_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

ds4_test_cpu: ds4_test_cpu.o rax.o $(CPU_CORE_OBJS) $(MEDIA_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_test_cpu.o rax.o $(CPU_CORE_OBJS) $(MEDIA_OBJS) $(CPU_TEST_LDLIBS)

test: ds4_test
	./ds4_test

qwen36-gates: ds4_test ds4-server
	./ds4_test --runtime-core --qwen36-vectors

qwen36-gates-strict: ds4_test ds4-server
	QWEN36_REQUIRE_OFFICIAL_VECTORS=1 ./ds4_test --runtime-core --qwen36-vectors

qwen36-gates-cpu: ds4_test_cpu
	./ds4_test_cpu --runtime-core --qwen36-vectors

qwen36-text-gates: ds4_test ds4-server
	QWEN36_TEST_TEXT_ONLY=1 ./ds4_test --runtime-core --qwen36-vectors

qwen36-text-gates-strict: ds4_test ds4-server
	QWEN36_TEST_TEXT_ONLY=1 QWEN36_REQUIRE_OFFICIAL_VECTORS=1 ./ds4_test --runtime-core --qwen36-vectors

qwen36-text-gates-cpu: ds4_test_cpu
	QWEN36_TEST_TEXT_ONLY=1 ./ds4_test_cpu --runtime-core --qwen36-vectors

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4_cpu ds4_native ds4_server_test ds4_test ds4_test_cpu *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
