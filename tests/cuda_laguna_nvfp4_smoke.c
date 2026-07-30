#include "ds4_gpu.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "cuda-laguna-nvfp4: FAIL: %s (line %d)\n", \
                msg, __LINE__); \
        return 1; \
    } \
} while (0)

typedef struct {
    uint8_t d[4];
    uint8_t qs[32];
} block_nvfp4;

typedef struct {
    uint8_t *data;
    uint64_t size;
    uint64_t used;
} model_blob;

#ifndef DS4_NVFP4_SMOKE_TOKENS
#define DS4_NVFP4_SMOKE_TOKENS 256
#endif

static uint64_t blob_alloc(model_blob *blob, uint64_t bytes) {
    const uint64_t offset = (blob->used + 255u) & ~255ull;
    if (offset > blob->size || bytes > blob->size - offset)
        return UINT64_MAX;
    blob->used = offset + bytes;
    return offset;
}

static int close_enough(float got, float expected, float atol, float rtol) {
    return isfinite(got) &&
           fabsf(got - expected) <= atol + rtol * fabsf(expected);
}

static int check_bf16_qkvg_shared_activation(model_blob *blob) {
    enum {
        in_dim = 128,
        q_dim = 256,
        kv_dim = 64,
        gate_dim = 8,
        max_tokens = 8
    };
    const uint32_t rows[4] = {q_dim, kv_dim, kv_dim, gate_dim};
    const uint16_t bf16_value[4] = {
        0x3f80u, /*  1.0 */
        0x3f00u, /*  0.5 */
        0x4000u, /*  2.0 */
        0xbf80u, /* -1.0 */
    };
    const float expected[4] = {32.0f, 16.0f, 64.0f, -32.0f};
    uint64_t weight_offset[4];
    for (uint32_t matrix = 0u; matrix < 4u; matrix++) {
        const uint64_t values = (uint64_t)rows[matrix] * in_dim;
        const uint64_t bytes = values * sizeof(uint16_t);
        weight_offset[matrix] = blob_alloc(blob, bytes);
        CHECK(weight_offset[matrix] != UINT64_MAX,
              "allocate BF16 QKVG weights");
        uint16_t *weight =
            (uint16_t *)(blob->data + weight_offset[matrix]);
        for (uint64_t i = 0u; i < values; i++)
            weight[i] = bf16_value[matrix];
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(
        (uint64_t)max_tokens * in_dim * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(
        (uint64_t)max_tokens * q_dim * sizeof(float));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(
        (uint64_t)max_tokens * kv_dim * sizeof(float));
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(
        (uint64_t)max_tokens * kv_dim * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(
        (uint64_t)max_tokens * gate_dim * sizeof(float));
    CHECK(x && q && k && v && gate, "BF16 QKVG tensor allocation");
    CHECK(ds4_gpu_tensor_fill_f32(
              x, 0.25f, (uint64_t)max_tokens * in_dim),
          "write BF16 QKVG activation");

    const uint32_t token_cases[2] = {1u, max_tokens};
    ds4_gpu_tensor *outputs[4] = {q, k, v, gate};
    for (uint32_t tc = 0u; tc < 2u; tc++) {
        const uint32_t n_tokens = token_cases[tc];
        CHECK(ds4_gpu_laguna_qkvg_bf16_tensor(
                  q, k, v, gate, blob->data, blob->size,
                  weight_offset[0], weight_offset[1],
                  weight_offset[2], weight_offset[3],
                  in_dim, q_dim, kv_dim, gate_dim, n_tokens, x),
              "Laguna BF16 QKVG shared-activation projection");
        CHECK(cudaDeviceSynchronize() == cudaSuccess,
              "synchronize Laguna BF16 QKVG projection");
        for (uint32_t matrix = 0u; matrix < 4u; matrix++) {
            const uint64_t count = (uint64_t)n_tokens * rows[matrix];
            float *result = (float *)malloc((size_t)count * sizeof(float));
            CHECK(result != NULL, "allocate BF16 QKVG result");
            CHECK(ds4_gpu_tensor_read(
                      outputs[matrix], 0, result,
                      count * sizeof(float)),
                  "read BF16 QKVG result");
            for (uint64_t i = 0u; i < count; i++) {
                if (!close_enough(result[i], expected[matrix],
                                  0.01f, 0.001f)) {
                    fprintf(stderr,
                            "cuda-laguna: BF16 QKVG matrix=%u "
                            "result[%llu]=%.9g expected=%.9g\n",
                            matrix, (unsigned long long)i,
                            result[i], expected[matrix]);
                    free(result);
                    CHECK(0, "Laguna BF16 QKVG numeric");
                }
            }
            free(result);
        }
    }

    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(x);
    return 0;
}

static int check_native_nvfp4_moe_real_shape(model_blob *blob) {
    enum {
        n_total_expert = 256,
        n_expert = 10,
        n_tokens = DS4_NVFP4_SMOKE_TOKENS,
        expert_in_dim = 3072,
        expert_mid_dim = 1024,
        out_dim = 3072
    };
    const uint32_t rows[3] = {
        expert_mid_dim, expert_mid_dim, out_dim
    };
    const uint32_t cols[3] = {
        expert_in_dim, expert_in_dim, expert_mid_dim
    };
    uint64_t packed[3][n_total_expert];
    uint64_t scale[3][n_total_expert];
    uint64_t weight_global[3][n_total_expert];
    uint64_t input_global[3][n_total_expert];
    const float output_scale[3] = {0.5f, 2.0f, 1.25f};
    for (uint32_t p = 0u; p < 3u; p++) {
        const uint64_t packed_bytes =
            (uint64_t)rows[p] * cols[p] / 2u;
        const uint64_t scale_bytes =
            (uint64_t)rows[p] * cols[p] / 16u;
        const uint64_t packed_offset = blob_alloc(blob, packed_bytes);
        const uint64_t scale_offset = blob_alloc(blob, scale_bytes);
        const uint64_t weight_global_offset =
            blob_alloc(blob, sizeof(float));
        const uint64_t input_global_offset =
            blob_alloc(blob, sizeof(float));
        CHECK(packed_offset != UINT64_MAX &&
              scale_offset != UINT64_MAX &&
              weight_global_offset != UINT64_MAX &&
              input_global_offset != UINT64_MAX,
              "allocate real-shape native NVFP4 tensors");
        memset(blob->data + packed_offset, 0x22, packed_bytes);
        memset(blob->data + scale_offset, 0x38, scale_bytes);
        const float raw_weight_global = 1.0f / output_scale[p];
        const float raw_input_global = p < 2u ? 600.0f : 256.0f;
        memcpy(blob->data + weight_global_offset,
               &raw_weight_global, sizeof(raw_weight_global));
        memcpy(blob->data + input_global_offset,
               &raw_input_global, sizeof(raw_input_global));
        for (uint32_t e = 0u; e < n_total_expert; e++) {
            packed[p][e] = packed_offset;
            scale[p][e] = scale_offset;
            weight_global[p][e] = weight_global_offset;
            input_global[p][e] = input_global_offset;
        }
    }

    ds4_gpu_laguna_moe_desc routed = {0};
    routed.gate_type = 40u;
    routed.up_type = 40u;
    routed.down_type = 40u;
    routed.gate_row_bytes =
        routed.up_row_bytes =
            (uint64_t)(expert_in_dim / 64u) * sizeof(block_nvfp4);
    routed.down_row_bytes =
        (uint64_t)(expert_mid_dim / 64u) * sizeof(block_nvfp4);
    routed.gate_expert_bytes =
        routed.up_expert_bytes =
            (uint64_t)expert_mid_dim * routed.gate_row_bytes;
    routed.down_expert_bytes =
        (uint64_t)out_dim * routed.down_row_bytes;
    ds4_gpu_nvfp4_matrix_desc *matrix[3] = {
        &routed.gate_nvfp4, &routed.up_nvfp4, &routed.down_nvfp4
    };
    for (uint32_t p = 0u; p < 3u; p++) {
        matrix[p]->packed_offsets = packed[p];
        matrix[p]->scale_offsets = scale[p];
        matrix[p]->weight_global_scale_offsets = weight_global[p];
        matrix[p]->input_global_scale_offsets = input_global[p];
        matrix[p]->packed_bytes = (uint64_t)rows[p] * cols[p] / 2u;
        matrix[p]->scale_bytes = (uint64_t)rows[p] * cols[p] / 16u;
    }

    int32_t selected_host[n_tokens * n_expert];
    float weights_host[n_tokens * n_expert];
    for (uint32_t token = 0u; token < n_tokens; token++) {
        for (uint32_t e = 0u; e < n_expert; e++) {
            const uint64_t pair = (uint64_t)token * n_expert + e;
            selected_host[pair] =
                (int32_t)((token * 17u + e * 23u) % n_total_expert);
            weights_host[pair] = 1.0f / (float)n_expert;
        }
    }
    ds4_gpu_tensor *x =
        ds4_gpu_tensor_alloc(
            (uint64_t)n_tokens * expert_in_dim * sizeof(float));
    ds4_gpu_tensor *selected =
        ds4_gpu_tensor_alloc(sizeof(selected_host));
    ds4_gpu_tensor *weights =
        ds4_gpu_tensor_alloc(sizeof(weights_host));
    ds4_gpu_tensor *out =
        ds4_gpu_tensor_alloc(
            (uint64_t)n_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *reference =
        ds4_gpu_tensor_alloc(
            (uint64_t)n_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *mid =
        ds4_gpu_tensor_alloc(
            (uint64_t)n_tokens * n_expert *
            expert_mid_dim * sizeof(float));
    CHECK(x && selected && weights && out && reference && mid,
          "real-shape NVFP4 MoE tensor allocation");
    CHECK(ds4_gpu_tensor_fill_f32(
              x, 0.0001f, (uint64_t)n_tokens * expert_in_dim) &&
          ds4_gpu_tensor_write(selected, 0, selected_host,
                               sizeof(selected_host)) &&
          ds4_gpu_tensor_write(weights, 0, weights_host,
                               sizeof(weights_host)),
          "write real-shape NVFP4 MoE tensors");
    CHECK(setenv("DS4_CUDA_NVFP4_GROUPED_MIN_TOKENS",
                 "4294967295", 1) == 0,
          "select native NVFP4 DP4A reference path");
    CHECK(ds4_gpu_laguna_routed_moe_tensor(
              reference, mid, blob->data, blob->size, &routed,
              expert_in_dim, expert_mid_dim, out_dim,
              selected, weights, n_total_expert, n_expert, x, n_tokens),
          "real-shape native Laguna NVFP4 DP4A reference");
    CHECK(cudaDeviceSynchronize() == cudaSuccess,
          "synchronize real-shape native Laguna NVFP4 DP4A reference");
    CHECK(setenv("DS4_CUDA_NVFP4_GROUPED_MIN_TOKENS", "1", 1) == 0,
          "select native NVFP4 grouped MMA path");
    CHECK(ds4_gpu_laguna_routed_moe_tensor(
              out, mid, blob->data, blob->size, &routed,
              expert_in_dim, expert_mid_dim, out_dim,
              selected, weights, n_total_expert, n_expert, x, n_tokens),
          "real-shape Blackwell native Laguna NVFP4 MoE");
    CHECK(cudaDeviceSynchronize() == cudaSuccess,
          "synchronize real-shape Blackwell native Laguna NVFP4 MoE");

    const uint64_t result_count = (uint64_t)n_tokens * out_dim;
    float *result = (float *)malloc(
        (size_t)result_count * sizeof(float));
    float *reference_result = (float *)malloc(
        (size_t)result_count * sizeof(float));
    CHECK(result != NULL && reference_result != NULL,
          "allocate real-shape NVFP4 results");
    CHECK(ds4_gpu_tensor_read(
              out, 0, result, result_count * sizeof(float)),
          "read real-shape NVFP4 result");
    CHECK(ds4_gpu_tensor_read(
              reference, 0, reference_result,
              result_count * sizeof(float)),
          "read real-shape NVFP4 DP4A reference");
    const float projection = 0.0001f * (float)expert_in_dim;
    const float gate = projection * output_scale[0];
    const float up = projection * output_scale[1];
    const float one_mid =
        (gate / (1.0f + expf(-gate))) * up / (float)n_expert;
    const float expected =
        one_mid * (float)n_expert * (float)expert_mid_dim *
        output_scale[2];
    for (uint64_t i = 0u; i < result_count; i++) {
        if (!close_enough(result[i], reference_result[i],
                          0.01f, 0.002f)) {
            fprintf(stderr,
                    "cuda-laguna: grouped NVFP4 result[%llu]=%.9g "
                    "DP4A=%.9g\n",
                    (unsigned long long)i,
                    result[i], reference_result[i]);
            CHECK(0, "grouped Blackwell NVFP4 matches DP4A");
        }
        if (!close_enough(result[i], expected, 1.0f, 0.20f)) {
            fprintf(stderr,
                    "cuda-laguna: real-shape NVFP4 result[%llu]=%.9g "
                    "expected=%.9g\n",
                    (unsigned long long)i, result[i], expected);
            CHECK(0, "real-shape Blackwell native Laguna NVFP4 MoE numeric");
        }
    }
    free(reference_result);
    free(result);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(x);
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init(), "ds4_gpu_init");
    const uint64_t blob_size = 16u * 1024u * 1024u;
    void *host = NULL;
    CHECK(cudaMallocHost(&host, blob_size) == cudaSuccess,
          "allocate pinned native model blob");
    memset(host, 0, blob_size);
    model_blob blob = {(uint8_t *)host, blob_size, 0u};
    CHECK(ds4_gpu_set_model_map(blob.data, blob.size),
          "set synthetic native model map");
    int rc = check_bf16_qkvg_shared_activation(&blob);
    if (rc == 0) rc = check_native_nvfp4_moe_real_shape(&blob);
    ds4_gpu_cleanup();
    (void)cudaFreeHost(host);
    if (rc == 0) puts("cuda Laguna native NVFP4 regression: OK");
    return rc;
}
