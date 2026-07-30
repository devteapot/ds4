/*
 * Blackwell-only implementation for the official Laguna NVFP4 checkpoint.
 *
 * Keep this translation unit separate from ds4_cuda.cu: the canonical
 * Q4_K_M/DFlash CUDA implementation is fastest on GB10 with nvcc's generic
 * code generation, while the native checkpoint benefits from explicit
 * SM121 block-scaled MMA and DP4A code generation.
 */

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "../../../ds4_gpu.h"
#include "../../../ds4_gpu_mgpu.h"

typedef struct {
    uint8_t qs[32];
    uint8_t d[4];
} cuda_block_nvfp4;

static_assert(sizeof(cuda_block_nvfp4) == 36u,
              "NVFP4 compatibility block layout mismatch");

static_assert(sizeof(ds4_gpu_nvfp4_matrix_desc) == 48u,
              "NVFP4 matrix descriptor ABI mismatch");
static_assert(sizeof(ds4_gpu_laguna_moe_desc) == 256u,
              "Laguna MoE descriptor ABI mismatch");

#if defined(DS4_CUDA_NVFP4_MMA)

static inline int ds4_tensor_device_idx(const ds4_gpu_tensor *tensor) {
    if (!tensor || tensor->device_id < 0) return 0;
    return tensor->device_id;
}

static int cuda_ok(cudaError_t error, const char *what) {
    if (error == cudaSuccess) return 1;
    fprintf(stderr, "ds4: CUDA %s failed: %s\n",
            what, cudaGetErrorString(error));
    return 0;
}

extern "C" const char *ds4_cuda_nvfp4_resolve_weight_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        int logical_tier,
        const char *label);
extern "C" void *ds4_cuda_nvfp4_tmp_alloc_on(
        int logical_tier, uint64_t bytes, const char *what);
extern "C" int ds4_cuda_nvfp4_prepare_sorted_tiles16(
        uint32_t *counts,
        uint32_t *offsets,
        uint32_t *cursors,
        uint32_t *sorted_pairs,
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert);

#define cuda_resolve_weight_ptr ds4_cuda_nvfp4_resolve_weight_ptr
#define cuda_tmp_alloc_on ds4_cuda_nvfp4_tmp_alloc_on
#define DS4_CUDA_NVFP4_NATIVE_ONLY 1
#include "nvfp4.inc"
#undef DS4_CUDA_NVFP4_NATIVE_ONLY
#undef cuda_tmp_alloc_on
#undef cuda_resolve_weight_ptr

extern "C" void
ds4_cuda_laguna_native_nvfp4_cache_release_all(void) {
    cuda_native_nvfp4_cache_release_all();
}

#else

extern "C" int ds4_cuda_laguna_native_nvfp4_routed_moe_tensor(
        ds4_gpu_tensor                *,
        ds4_gpu_tensor                *,
        const void                    *,
        uint64_t,
        const ds4_gpu_laguna_moe_desc *,
        uint32_t,
        uint32_t,
        uint32_t,
        const ds4_gpu_tensor          *,
        const ds4_gpu_tensor          *,
        uint32_t,
        uint32_t,
        const ds4_gpu_tensor          *,
        uint32_t) {
    fprintf(stderr,
            "ds4: this CUDA build has no native Blackwell NVFP4 "
            "implementation; build with make cuda-spark\n");
    return 0;
}

extern "C" void
ds4_cuda_laguna_native_nvfp4_cache_release_all(void) {
}

#endif
