/* CUDA-only BF16 projection and norm fusions for official Laguna checkpoints. */

enum {
    CUDA_LAGUNA_BF16_LT_PLAN_COUNT = 5,
};

static const uint32_t cuda_laguna_bf16_lt_out_dims[
        CUDA_LAGUNA_BF16_LT_PLAN_COUNT] = {
    1024u, 6144u, 9216u, 12288u, 100352u,
};

typedef struct {
    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t a;
    cublasLtMatrixLayout_t b;
    cublasLtMatrixLayout_t c;
    cublasLtMatmulAlgo_t algorithm;
    int state; /* 0: unprepared, 1: ready, -1: unavailable */
} cuda_laguna_bf16_lt_plan;

static cuda_laguna_bf16_lt_plan
    g_cuda_laguna_bf16_lt_plans[DS4_MAX_GPUS]
                                    [CUDA_LAGUNA_BF16_LT_PLAN_COUNT];

static int32_t cuda_laguna_bf16_lt_config_i32(
        const cublasLtMatmulAlgo_t *algorithm,
        cublasLtMatmulAlgoConfigAttributes_t attribute) {
    int32_t value = -1;
    size_t written = 0u;
    if (cublasLtMatmulAlgoConfigGetAttribute(
            algorithm, attribute, &value, sizeof(value), &written) !=
            CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    return value;
}

static uint32_t cuda_laguna_bf16_lt_config_u32(
        const cublasLtMatmulAlgo_t *algorithm,
        cublasLtMatmulAlgoConfigAttributes_t attribute) {
    uint32_t value = UINT32_MAX;
    size_t written = 0u;
    if (cublasLtMatmulAlgoConfigGetAttribute(
            algorithm, attribute, &value, sizeof(value), &written) !=
            CUBLAS_STATUS_SUCCESS) {
        return UINT32_MAX;
    }
    return value;
}

static void cuda_laguna_bf16_lt_destroy_plan(
        cuda_laguna_bf16_lt_plan *plan) {
    if (plan->c) (void)cublasLtMatrixLayoutDestroy(plan->c);
    if (plan->b) (void)cublasLtMatrixLayoutDestroy(plan->b);
    if (plan->a) (void)cublasLtMatrixLayoutDestroy(plan->a);
    if (plan->operation) {
        (void)cublasLtMatmulDescDestroy(plan->operation);
    }
    memset(plan, 0, sizeof(*plan));
}

static int cuda_laguna_bf16_lt_prepare_plan(
        int logical_tier,
        int slot) {
    cuda_laguna_bf16_lt_plan *plan =
        &g_cuda_laguna_bf16_lt_plans[logical_tier][slot];
    if (plan->state != 0) return plan->state > 0;
    plan->state = -1;

    const cublasLtHandle_t handle =
        (cublasLtHandle_t)g_gpu[logical_tier].cublaslt;
    if (!handle) return 0;

    constexpr uint32_t in_dim = 3072u;
    const uint32_t out_dim = cuda_laguna_bf16_lt_out_dims[slot];
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    cublasLtMatmulPreference_t preference = NULL;
    constexpr uint64_t max_workspace = 0u;
    constexpr uint32_t min_alignment_a = 64u;
    bool descriptors_ok =
        cublasLtMatmulDescCreate(
            &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_TRANSA,
            &transa, sizeof(transa)) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_TRANSB,
            &transb, sizeof(transb)) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(
            &plan->a, CUDA_R_16BF, in_dim, out_dim, in_dim) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(
            &plan->b, CUDA_R_16BF, in_dim, 1u, in_dim) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(
            &plan->c, CUDA_R_32F, out_dim, 1u, out_dim) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulPreferenceCreate(&preference) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &max_workspace, sizeof(max_workspace)) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            &min_alignment_a, sizeof(min_alignment_a)) ==
            CUBLAS_STATUS_SUCCESS;
    if (!descriptors_ok) {
        if (preference) {
            (void)cublasLtMatmulPreferenceDestroy(preference);
        }
        cuda_laguna_bf16_lt_destroy_plan(plan);
        plan->state = -1;
        return 0;
    }

    constexpr int requested = 32;
    cublasLtMatmulHeuristicResult_t heuristics[requested] = {};
    int returned = 0;
    const cublasStatus_t heuristic_status =
        cublasLtMatmulAlgoGetHeuristic(
            handle, plan->operation, plan->a, plan->b,
            plan->c, plan->c, preference, requested,
            heuristics, &returned);
    (void)cublasLtMatmulPreferenceDestroy(preference);
    if (heuristic_status != CUBLAS_STATUS_SUCCESS) {
        cuda_laguna_bf16_lt_destroy_plan(plan);
        plan->state = -1;
        return 0;
    }

    for (int i = 0; i < returned; i++) {
        const cublasLtMatmulAlgo_t *algorithm =
            &heuristics[i].algo;
        if (heuristics[i].state != CUBLAS_STATUS_SUCCESS ||
            heuristics[i].workspaceSize != 0u ||
            cuda_laguna_bf16_lt_config_i32(
                algorithm, CUBLASLT_ALGO_CONFIG_ID) != 13 ||
            cuda_laguna_bf16_lt_config_u32(
                algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID) != 0 ||
            cuda_laguna_bf16_lt_config_u32(
                algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID) != 0 ||
            cuda_laguna_bf16_lt_config_i32(
                algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) != 1 ||
            cuda_laguna_bf16_lt_config_u32(
                algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION) != 75 ||
            cuda_laguna_bf16_lt_config_u32(
                algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING) != 0 ||
            cuda_laguna_bf16_lt_config_u32(
                algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) != 0) {
            continue;
        }
        cublasLtMatmulHeuristicResult_t check = {};
        if (cublasLtMatmulAlgoCheck(
                handle, plan->operation, plan->a, plan->b,
                plan->c, plan->c, algorithm, &check) !=
                CUBLAS_STATUS_SUCCESS ||
            check.state != CUBLAS_STATUS_SUCCESS ||
            check.workspaceSize != 0u) {
            continue;
        }
        plan->algorithm = *algorithm;
        plan->state = 1;
        if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr,
                    "ds4: Laguna BF16 Lt plan ready "
                    "K=%u M=%u algo=13 custom=75\n",
                    in_dim, out_dim);
        }
        return 1;
    }

    cuda_laguna_bf16_lt_destroy_plan(plan);
    plan->state = -1;
    return 0;
}

static int cuda_laguna_bf16_lt_matmul(
        float *out,
        const __nv_bfloat16 *weights,
        const __nv_bfloat16 *input,
        uint32_t in_dim,
        uint32_t out_dim,
        int logical_tier) {
    if (!g_cuda_laguna_bf16_lt_enabled ||
        !out || !weights || !input ||
        in_dim != 3072u ||
        (reinterpret_cast<uintptr_t>(weights) & 63u) != 0u ||
        (reinterpret_cast<uintptr_t>(input) & 255u) != 0u ||
        (reinterpret_cast<uintptr_t>(out) & 255u) != 0u ||
        logical_tier < 0 || logical_tier >= g_n_gpus ||
        g_gpu[logical_tier].sm_major != 12 ||
        g_gpu[logical_tier].sm_minor != 1) {
        return 0;
    }
    int slot = -1;
    for (int i = 0; i < CUDA_LAGUNA_BF16_LT_PLAN_COUNT; i++) {
        if (cuda_laguna_bf16_lt_out_dims[i] == out_dim) {
            slot = i;
            break;
        }
    }
    if (slot < 0) return 0;
    cuda_laguna_bf16_lt_plan *plan =
        &g_cuda_laguna_bf16_lt_plans[logical_tier][slot];
    if (plan->state == 0 &&
        !cuda_laguna_bf16_lt_prepare_plan(logical_tier, slot)) {
        return 0;
    }
    if (plan->state != 1) return 0;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasLtMatmul(
        (cublasLtHandle_t)g_gpu[logical_tier].cublaslt,
        plan->operation,
        &alpha,
        weights,
        plan->a,
        input,
        plan->b,
        &beta,
        out,
        plan->c,
        out,
        plan->c,
        &plan->algorithm,
        NULL,
        0u,
        NULL);
    if (status == CUBLAS_STATUS_SUCCESS) return 1;
    plan->state = -1;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr,
                "ds4: Laguna BF16 Lt launch failed "
                "K=%u M=%u status=%d; disabling plan\n",
                in_dim, out_dim, (int)status);
    }
    return 0;
}

static void cuda_laguna_bf16_lt_cleanup(void) {
    for (int tier = 0; tier < DS4_MAX_GPUS; tier++) {
        if (g_cuda_laguna_bf16_lt_plans[tier][0].state != 0 &&
            tier < g_n_gpus) {
            (void)cudaSetDevice(g_gpu[tier].device_id);
        }
        for (int slot = 0;
             slot < CUDA_LAGUNA_BF16_LT_PLAN_COUNT;
             slot++) {
            cuda_laguna_bf16_lt_destroy_plan(
                &g_cuda_laguna_bf16_lt_plans[tier][slot]);
        }
    }
}

__global__ static void dflash_aux_rms_norm_bf16_kernel(
        float *features,
        const __nv_bfloat16 *weights,
        uint32_t n_rows,
        uint32_t n_embd,
        uint32_t n_aux,
        float eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t aux = blockIdx.y;
    if (row >= n_rows || aux >= n_aux || n_embd == 0u) return;
    float *x = features + ((uint64_t)row * n_aux + aux) * n_embd;
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t col = tid; col < n_embd; col += blockDim.x) {
        const float value = x[col];
        sum += value * value;
    }
    __shared__ float scratch[256];
    scratch[tid] = sum;
    __syncthreads();
    for (uint32_t step = blockDim.x >> 1u; step != 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] / (float)n_embd + eps);
    const __nv_bfloat16 *weight = weights + (uint64_t)aux * n_embd;
    for (uint32_t col = tid; col < n_embd; col += blockDim.x) {
        x[col] = x[col] * scale * __bfloat162float(weight[col]);
    }
}

extern "C" int ds4_gpu_dflash_aux_norm_bf16_tensor(
        ds4_gpu_tensor *features,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        n_rows,
        uint32_t        n_embd,
        uint32_t        n_aux,
        float           eps) {
    if (!features || !model_map || n_rows == 0u || n_embd == 0u ||
        n_aux == 0u || !isfinite(eps) || eps < 0.0f) {
        return 0;
    }
    const uint64_t feature_values = (uint64_t)n_rows * n_aux * n_embd;
    const uint64_t weight_values = (uint64_t)n_aux * n_embd;
    const uint64_t weight_bytes =
        weight_values * sizeof(__nv_bfloat16);
    if (feature_values > UINT64_MAX / sizeof(float) ||
        weight_values > UINT64_MAX / sizeof(__nv_bfloat16) ||
        features->bytes < feature_values * sizeof(float) ||
        weight_offset > model_size ||
        weight_bytes > model_size - weight_offset) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(features);
    const __nv_bfloat16 *weights =
        (const __nv_bfloat16 *)cuda_resolve_weight_ptr(
            model_map, weight_offset, weight_bytes, logical_tier,
            "DFlash BF16 auxiliary norm");
    if (!weights) return 0;
    const dim3 grid(n_rows, n_aux, 1u);
    dflash_aux_rms_norm_bf16_kernel<<<grid, 256>>>(
        (float *)features->ptr, weights, n_rows, n_embd, n_aux, eps);
    return cuda_ok(cudaGetLastError(),
                   "DFlash BF16 auxiliary norm launch");
}

__global__ static void dflash_unpack_qkv_kernel(
        float *q,
        float *k,
        float *v,
        const float *qkv,
        uint32_t n_tokens) {
    constexpr uint32_t q_dim = 72u * 128u;
    constexpr uint32_t kv_dim = 8u * 128u;
    constexpr uint32_t qkv_dim = q_dim + 2u * kv_dim;
    const uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t values = (uint64_t)n_tokens * qkv_dim;
    if (index >= values) return;
    const uint32_t token = (uint32_t)(index / qkv_dim);
    const uint32_t col =
        (uint32_t)(index - (uint64_t)token * qkv_dim);
    const float value = qkv[index];
    if (col < q_dim) {
        q[(uint64_t)token * q_dim + col] = value;
    } else if (col < q_dim + kv_dim) {
        k[(uint64_t)token * kv_dim + col - q_dim] = value;
    } else {
        v[(uint64_t)token * kv_dim + col - q_dim - kv_dim] = value;
    }
}


__device__ __forceinline__ static float laguna_norm_weight_f32(float v) {
    return v;
}

__device__ __forceinline__ static float laguna_norm_weight_f32(
        __nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename Weight>
__global__ static void laguna_head_rms_norm_rope_kernel(
        float *x,
        const Weight *weight,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (head >= n_head || token >= n_tokens || tid >= 128u) return;
    float *row = x + ((uint64_t)token * n_head + head) * head_dim;
    __shared__ float partial[128];
    float sum = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += 128u) {
        const float v = row[i];
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = 64u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = tid; i < head_dim; i += 128u) {
        row[i] = row[i] * inv * laguna_norm_weight_f32(weight[i]);
    }
    __syncthreads();

    const uint32_t half_rot = n_rot >> 1u;
    if (tid >= half_rot) return;
    const uint32_t rel_i0 = tid * 2u;
    const uint32_t pos = pos0 + token;
    const float theta_extrap =
        (float)pos * powf(freq_base, -(float)rel_i0 / (float)n_rot);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float magnitude = attn_factor;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        const float corr0 = fmaxf(
                0.0f,
                floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) / denom));
        const float corr1 = fminf(
                (float)(n_rot - 1u),
                ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) / denom));
        const float mix = rope_yarn_ramp_cpu_equiv_dev(
                corr0, corr1, (int)rel_i0) * ext_factor;
        theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
        magnitude *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const float c = cosf(theta) * magnitude;
    const float s = sinf(theta) * magnitude;
    const float x0 = row[tid];
    const float x1 = row[tid + half_rot];
    row[tid] = x0 * c - x1 * s;
    row[tid + half_rot] = x0 * s + x1 * c;
}

/* Q and K use the same per-head operation. Keep each head in its own block,
 * as above, but put both block sets in one launch. */
template <typename Weight>
__global__ static void laguna_qk_head_rms_norm_rope_kernel(
        float *q,
        float *k,
        const Weight *q_weight,
        const Weight *k_weight,
        uint32_t n_tokens,
        uint32_t n_q_head,
        uint32_t n_k_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    const bool is_q = blockIdx.x < n_q_head;
    const uint32_t head =
        is_q ? blockIdx.x : blockIdx.x - n_q_head;
    const uint32_t n_head = is_q ? n_q_head : n_k_head;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (head >= n_head || token >= n_tokens || tid >= 128u) return;
    float *row = (is_q ? q : k) +
        ((uint64_t)token * n_head + head) * head_dim;
    const Weight *weight = is_q ? q_weight : k_weight;
    __shared__ float partial[128];
    float sum = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += 128u) {
        const float value = row[i];
        sum += value * value;
    }
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = 64u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = tid; i < head_dim; i += 128u) {
        row[i] =
            row[i] * inv * laguna_norm_weight_f32(weight[i]);
    }
    __syncthreads();

    const uint32_t half_rot = n_rot >> 1u;
    if (tid >= half_rot) return;
    const uint32_t rel_i0 = tid * 2u;
    const uint32_t pos = pos0 + token;
    const float theta_extrap =
        (float)pos * powf(freq_base, -(float)rel_i0 / (float)n_rot);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float magnitude = attn_factor;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        const float corr0 = fmaxf(
                0.0f,
                floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) / denom));
        const float corr1 = fminf(
                (float)(n_rot - 1u),
                ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) / denom));
        const float mix = rope_yarn_ramp_cpu_equiv_dev(
                corr0, corr1, (int)rel_i0) * ext_factor;
        theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
        magnitude *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const float c = cosf(theta) * magnitude;
    const float s = sinf(theta) * magnitude;
    const float x0 = row[tid];
    const float x1 = row[tid + half_rot];
    row[tid] = x0 * c - x1 * s;
    row[tid + half_rot] = x0 * s + x1 * c;
}


extern "C" int ds4_gpu_laguna_qkvg_bf16_tensor(
        ds4_gpu_tensor *q, ds4_gpu_tensor *k, ds4_gpu_tensor *v,
        ds4_gpu_tensor *gate, const void *model_map, uint64_t model_size,
        uint64_t q_weight_offset, uint64_t k_weight_offset,
        uint64_t v_weight_offset, uint64_t gate_weight_offset,
        uint32_t in_dim, uint32_t q_dim, uint32_t kv_dim,
        uint32_t gate_dim, uint32_t n_tokens,
        const ds4_gpu_tensor *x) {
    const bool grouped =
        cuda_laguna_blackwell_ok() && g_cublas_ready &&
        getenv("DS4_CUDA_LAGUNA_NO_BF16_QKVG_FUSION") == NULL;
    if (grouped && q && k && v && gate && model_map && x &&
        n_tokens != 0u && n_tokens <= 16u && n_tokens <= INT_MAX &&
        in_dim != 0u && in_dim <= INT_MAX &&
        q_dim != 0u && q_dim <= INT_MAX &&
        kv_dim != 0u && kv_dim <= INT_MAX &&
        gate_dim != 0u && gate_dim <= INT_MAX) {
        const uint64_t q_values = (uint64_t)in_dim * q_dim;
        const uint64_t kv_values = (uint64_t)in_dim * kv_dim;
        const uint64_t gate_values = (uint64_t)in_dim * gate_dim;
        const bool sizes_ok =
            q_values <= UINT64_MAX / sizeof(__nv_bfloat16) &&
            kv_values <= UINT64_MAX / sizeof(__nv_bfloat16) &&
            gate_values <= UINT64_MAX / sizeof(__nv_bfloat16);
        const uint64_t q_weight_bytes =
            sizes_ok ? q_values * sizeof(__nv_bfloat16) : 0u;
        const uint64_t kv_weight_bytes =
            sizes_ok ? kv_values * sizeof(__nv_bfloat16) : 0u;
        const uint64_t gate_weight_bytes =
            sizes_ok ? gate_values * sizeof(__nv_bfloat16) : 0u;
        const int logical_tier = ds4_tensor_device_idx(q);
        if (sizes_ok &&
            q_weight_offset <= model_size &&
            q_weight_bytes <= model_size - q_weight_offset &&
            k_weight_offset <= model_size &&
            kv_weight_bytes <= model_size - k_weight_offset &&
            v_weight_offset <= model_size &&
            kv_weight_bytes <= model_size - v_weight_offset &&
            gate_weight_offset <= model_size &&
            gate_weight_bytes <= model_size - gate_weight_offset &&
            x->bytes >= (uint64_t)n_tokens * in_dim * sizeof(float) &&
            q->bytes >= (uint64_t)n_tokens * q_dim * sizeof(float) &&
            k->bytes >= (uint64_t)n_tokens * kv_dim * sizeof(float) &&
            v->bytes >= (uint64_t)n_tokens * kv_dim * sizeof(float) &&
            gate->bytes >=
                (uint64_t)n_tokens * gate_dim * sizeof(float) &&
            ds4_tensor_device_idx(k) == logical_tier &&
            ds4_tensor_device_idx(v) == logical_tier &&
            ds4_tensor_device_idx(gate) == logical_tier &&
            ds4_tensor_device_idx(x) == logical_tier) {
            const void *weights[4] = {
                cuda_resolve_weight_ptr(
                    model_map, q_weight_offset, q_weight_bytes,
                    logical_tier, "Laguna BF16 Q"),
                cuda_resolve_weight_ptr(
                    model_map, k_weight_offset, kv_weight_bytes,
                    logical_tier, "Laguna BF16 K"),
                cuda_resolve_weight_ptr(
                    model_map, v_weight_offset, kv_weight_bytes,
                    logical_tier, "Laguna BF16 V"),
                cuda_resolve_weight_ptr(
                    model_map, gate_weight_offset, gate_weight_bytes,
                    logical_tier, "Laguna BF16 attention gate"),
            };
            __nv_bfloat16 *xb =
                (__nv_bfloat16 *)cuda_tmp_alloc_on(
                    logical_tier,
                    (uint64_t)n_tokens * in_dim *
                        sizeof(__nv_bfloat16),
                    "Laguna grouped BF16 QKVG activation");
            if (weights[0] && weights[1] && weights[2] && weights[3] && xb) {
                /* Native NVFP4 keeps attention in BF16. The target supplies
                 * one activation column; its paired DFlash checkpoint
                 * supplies a short 2-16-column proposal block. Convert those
                 * token rows once, then reuse them across the four independent
                 * projections without flattening either matrix dimension. */
                const uint64_t activation_values =
                    (uint64_t)n_tokens * in_dim;
                f32_to_bf16_kernel<<<
                        (activation_values + 255u) / 256u, 256>>>(
                    xb, (const float *)x->ptr, activation_values);
                if (!cuda_ok(
                        cudaGetLastError(),
                        "Laguna grouped BF16 QKVG activation convert")) {
                    return 0;
                }
                const int rows[4] = {
                    (int)q_dim, (int)kv_dim, (int)kv_dim, (int)gate_dim
                };
                void *outputs[4] = {
                    q->ptr, k->ptr, v->ptr, gate->ptr
                };
                const char *labels[4] = {
                    "Laguna BF16 Q matmul",
                    "Laguna BF16 K matmul",
                    "Laguna BF16 V matmul",
                    "Laguna BF16 attention gate matmul",
                };
                const float alpha = 1.0f;
                const float beta = 0.0f;
                const cublasHandle_t handle =
                    cuda_cublas_for_tier(logical_tier);
                for (uint32_t i = 0u; i < 4u; i++) {
                    if (n_tokens == 1u &&
                        cuda_laguna_bf16_lt_matmul(
                            (float *)outputs[i],
                            (const __nv_bfloat16 *)weights[i],
                            xb,
                            in_dim,
                            (uint32_t)rows[i],
                            logical_tier)) {
                        continue;
                    }
                    const cublasStatus_t st = cublasGemmEx(
                        handle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        rows[i],
                        (int)n_tokens,
                        (int)in_dim,
                        &alpha,
                        weights[i],
                        CUDA_R_16BF,
                        (int)in_dim,
                        xb,
                        CUDA_R_16BF,
                        (int)in_dim,
                        &beta,
                        outputs[i],
                        CUDA_R_32F,
                        rows[i],
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (!cublas_ok(st, labels[i])) return 0;
                }
                return 1;
            }
        }
    }
    return ds4_gpu_matmul_bf16_tensor(
                   q, model_map, model_size, q_weight_offset,
                   in_dim, q_dim, x, n_tokens) &&
           ds4_gpu_matmul_bf16_tensor(
                   k, model_map, model_size, k_weight_offset,
                   in_dim, kv_dim, x, n_tokens) &&
           ds4_gpu_matmul_bf16_tensor(
                   v, model_map, model_size, v_weight_offset,
                   in_dim, kv_dim, x, n_tokens) &&
           ds4_gpu_matmul_bf16_tensor(
                   gate, model_map, model_size, gate_weight_offset,
                   in_dim, gate_dim, x, n_tokens);
}

extern "C" int ds4_gpu_dflash_qkvg_bf16_tensor(
        ds4_gpu_tensor *q, ds4_gpu_tensor *k, ds4_gpu_tensor *v,
        ds4_gpu_tensor *gate, ds4_gpu_tensor *qkv,
        const void *model_map, uint64_t model_size,
        uint64_t qkv_weight_offset, uint64_t gate_weight_offset,
        uint32_t n_tokens, const ds4_gpu_tensor *x) {
    constexpr uint32_t in_dim = 3072u;
    constexpr uint32_t q_dim = 72u * 128u;
    constexpr uint32_t kv_dim = 8u * 128u;
    constexpr uint32_t qkv_dim = q_dim + 2u * kv_dim;
    constexpr uint32_t gate_dim = 72u;
    if (!cuda_laguna_blackwell_ok() || !g_cublas_ready ||
        getenv("DS4_CUDA_LAGUNA_NO_BF16_QKVG_FUSION") != NULL ||
        !q || !k || !v || !gate || !qkv || !model_map || !x ||
        n_tokens == 0u || n_tokens > 16u) {
        return 0;
    }
    const uint64_t input_values = (uint64_t)n_tokens * in_dim;
    const uint64_t q_values = (uint64_t)n_tokens * q_dim;
    const uint64_t kv_values = (uint64_t)n_tokens * kv_dim;
    const uint64_t qkv_values = (uint64_t)n_tokens * qkv_dim;
    const uint64_t gate_values = (uint64_t)n_tokens * gate_dim;
    const uint64_t qkv_weight_bytes =
        (uint64_t)qkv_dim * in_dim * sizeof(__nv_bfloat16);
    const uint64_t gate_weight_bytes =
        (uint64_t)gate_dim * in_dim * sizeof(__nv_bfloat16);
    const int logical_tier = ds4_tensor_device_idx(q);
    if (qkv_weight_offset > model_size ||
        qkv_weight_bytes > model_size - qkv_weight_offset ||
        gate_weight_offset > model_size ||
        gate_weight_bytes > model_size - gate_weight_offset ||
        x->bytes < input_values * sizeof(float) ||
        q->bytes < q_values * sizeof(float) ||
        k->bytes < kv_values * sizeof(float) ||
        v->bytes < kv_values * sizeof(float) ||
        gate->bytes < gate_values * sizeof(float) ||
        qkv->bytes < qkv_values * sizeof(float) ||
        ds4_tensor_device_idx(k) != logical_tier ||
        ds4_tensor_device_idx(v) != logical_tier ||
        ds4_tensor_device_idx(gate) != logical_tier ||
        ds4_tensor_device_idx(qkv) != logical_tier ||
        ds4_tensor_device_idx(x) != logical_tier) {
        return 0;
    }
    const void *weights[2] = {
        cuda_resolve_weight_ptr(
            model_map, qkv_weight_offset, qkv_weight_bytes,
            logical_tier, "DFlash fused BF16 QKV"),
        cuda_resolve_weight_ptr(
            model_map, gate_weight_offset, gate_weight_bytes,
            logical_tier, "DFlash BF16 attention gate"),
    };
    __nv_bfloat16 *xb =
        (__nv_bfloat16 *)cuda_tmp_alloc_on(
            logical_tier, input_values * sizeof(__nv_bfloat16),
            "DFlash fused BF16 QKVG activation");
    if (!weights[0] || !weights[1] || !xb) return 0;
    f32_to_bf16_kernel<<<(input_values + 255u) / 256u, 256>>>(
        xb, (const float *)x->ptr, input_values);
    if (!cuda_ok(
            cudaGetLastError(),
            "DFlash fused BF16 QKVG activation convert")) {
        return 0;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t qkv_status = cublasGemmEx(
        cuda_cublas_for_tier(logical_tier),
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)qkv_dim, (int)n_tokens, (int)in_dim,
        &alpha,
        weights[0], CUDA_R_16BF, (int)in_dim,
        xb, CUDA_R_16BF, (int)in_dim,
        &beta,
        qkv->ptr, CUDA_R_32F, (int)qkv_dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (!cublas_ok(qkv_status, "DFlash fused BF16 QKV matmul")) {
        return 0;
    }
    const cublasStatus_t gate_status = cublasGemmEx(
        cuda_cublas_for_tier(logical_tier),
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)gate_dim, (int)n_tokens, (int)in_dim,
        &alpha,
        weights[1], CUDA_R_16BF, (int)in_dim,
        xb, CUDA_R_16BF, (int)in_dim,
        &beta,
        gate->ptr, CUDA_R_32F, (int)gate_dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (!cublas_ok(gate_status, "DFlash BF16 attention gate matmul")) {
        return 0;
    }
    dflash_unpack_qkv_kernel<<<
            (qkv_values + 255u) / 256u, 256>>>(
        (float *)q->ptr, (float *)k->ptr, (float *)v->ptr,
        (const float *)qkv->ptr, n_tokens);
    return cuda_ok(cudaGetLastError(), "DFlash fused BF16 QKV unpack");
}


extern "C" int ds4_gpu_laguna_head_rms_norm_rope_bf16_tensor(
        ds4_gpu_tensor *x, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t n_tokens, uint32_t n_head,
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow, float eps) {
    if (!x || !model_map || n_tokens == 0u || n_head == 0u ||
        head_dim == 0u || head_dim > 128u || n_rot == 0u ||
        n_rot > head_dim || (n_rot & 1u) != 0u ||
        pos0 > UINT32_MAX - n_tokens ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow) ||
        !isfinite(eps) || eps <= 0.0f) {
        return 0;
    }
    const uint64_t values = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t weight_bytes =
        (uint64_t)head_dim * sizeof(__nv_bfloat16);
    if (x->bytes < values * sizeof(float) ||
        weight_offset > model_size ||
        weight_bytes > model_size - weight_offset) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(x);
    const __nv_bfloat16 *weight =
        (const __nv_bfloat16 *)cuda_resolve_weight_ptr(
            model_map, weight_offset, weight_bytes, logical_tier,
            "Laguna BF16 head norm");
    if (!weight) return 0;
    laguna_head_rms_norm_rope_kernel<<<
            dim3(n_head, n_tokens, 1u), 128>>>(
            (float *)x->ptr, weight,
            n_tokens, n_head, head_dim, n_rot, pos0, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, eps);
    return cuda_ok(cudaGetLastError(),
                   "Laguna BF16 head RMSNorm/RoPE launch");
}


extern "C" int ds4_gpu_laguna_qk_head_rms_norm_rope_bf16_tensor(
        ds4_gpu_tensor *q, ds4_gpu_tensor *k, const void *model_map,
        uint64_t model_size, uint64_t q_weight_offset,
        uint64_t k_weight_offset, uint32_t n_tokens, uint32_t n_q_head,
        uint32_t n_k_head, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t n_ctx_orig, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, float eps) {
    if (n_tokens != 0u && n_tokens <= 16u &&
        cuda_laguna_blackwell_ok() &&
        getenv("DS4_CUDA_LAGUNA_NO_QK_NORM_ROPE_FUSION") == NULL &&
        q && k && model_map && n_q_head != 0u && n_k_head != 0u &&
        head_dim != 0u && head_dim <= 128u &&
        n_rot != 0u && n_rot <= head_dim && (n_rot & 1u) == 0u &&
        pos0 <= UINT32_MAX - n_tokens &&
        isfinite(freq_base) && freq_base > 0.0f &&
        isfinite(freq_scale) && freq_scale > 0.0f &&
        isfinite(ext_factor) && isfinite(attn_factor) &&
        isfinite(beta_fast) && isfinite(beta_slow) &&
        isfinite(eps) && eps > 0.0f) {
        const uint64_t q_values =
            (uint64_t)n_tokens * n_q_head * head_dim;
        const uint64_t k_values =
            (uint64_t)n_tokens * n_k_head * head_dim;
        const uint64_t weight_bytes =
            (uint64_t)head_dim * sizeof(__nv_bfloat16);
        if (q->bytes >= q_values * sizeof(float) &&
            k->bytes >= k_values * sizeof(float) &&
            q_weight_offset <= model_size &&
            weight_bytes <= model_size - q_weight_offset &&
            k_weight_offset <= model_size &&
            weight_bytes <= model_size - k_weight_offset &&
            ds4_tensor_device_idx(q) == ds4_tensor_device_idx(k)) {
            const int logical_tier = ds4_tensor_device_idx(q);
            const __nv_bfloat16 *q_weight =
                (const __nv_bfloat16 *)cuda_resolve_weight_ptr(
                    model_map, q_weight_offset, weight_bytes, logical_tier,
                    "Laguna BF16 Q head norm");
            const __nv_bfloat16 *k_weight =
                (const __nv_bfloat16 *)cuda_resolve_weight_ptr(
                    model_map, k_weight_offset, weight_bytes, logical_tier,
                    "Laguna BF16 K head norm");
            if (q_weight && k_weight) {
                laguna_qk_head_rms_norm_rope_kernel<<<
                        dim3(n_q_head + n_k_head, n_tokens, 1u), 128>>>(
                        (float *)q->ptr,
                        (float *)k->ptr,
                        q_weight,
                        k_weight,
                        n_tokens,
                        n_q_head,
                        n_k_head,
                        head_dim,
                        n_rot,
                        pos0,
                        n_ctx_orig,
                        freq_base,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        beta_fast,
                        beta_slow,
                        eps);
                return cuda_ok(
                    cudaGetLastError(),
                    "Laguna fused BF16 Q/K head RMSNorm/RoPE launch");
            }
        }
    }
    return ds4_gpu_laguna_head_rms_norm_rope_bf16_tensor(
                   q, model_map, model_size, q_weight_offset,
                   n_tokens, n_q_head, head_dim, n_rot, pos0, n_ctx_orig,
                   freq_base, freq_scale, ext_factor, attn_factor,
                   beta_fast, beta_slow, eps) &&
           ds4_gpu_laguna_head_rms_norm_rope_bf16_tensor(
                   k, model_map, model_size, k_weight_offset,
                   n_tokens, n_k_head, head_dim, n_rot, pos0, n_ctx_orig,
                   freq_base, freq_scale, ext_factor, attn_factor,
                   beta_fast, beta_slow, eps);
}
