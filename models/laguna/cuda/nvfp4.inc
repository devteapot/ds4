/* CUDA-only W4A4 path for the official Laguna NVFP4 checkpoint. */

static int cuda_laguna_blackwell_ok(void) {
    static int cached = -1;
    if (cached < 0) {
        if (getenv("DS4_CUDA_LAGUNA_NO_BLACKWELL") != NULL) {
            cached = 0;
        } else {
            int dev = 0;
            int major = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(
                &major, cudaDevAttrComputeCapabilityMajor, dev);
            cached = major >= 12 ? 1 : 0;
        }
    }
    return cached;
}

#if defined(DS4_CUDA_NVFP4_MMA)
static int cuda_laguna_nvfp4_hw_ok(void) {
    static int cached = -1;
    if (cached < 0) {
        int dev = 0;
        int major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, dev);
        cached = major >= 12 ? 1 : 0;
    }
    return cached;
}

static uint32_t cuda_laguna_nvfp4_grouped_min_tokens(void) {
    const char *value = getenv("DS4_CUDA_NVFP4_GROUPED_MIN_TOKENS");
    if (!value || !value[0]) return 256u;
    char *end = NULL;
    const unsigned long parsed = strtoul(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0u &&
           parsed <= UINT32_MAX ? (uint32_t)parsed : 256u;
}
#endif

struct cuda_native_nvfp4_cache {
    const uint64_t *key;
    const uint8_t **gate_packed;
    const uint8_t **gate_scale;
    const uint8_t **up_packed;
    const uint8_t **up_scale;
    const uint8_t **down_packed;
    const uint8_t **down_scale;
    float *gate_global;
    float *up_global;
    float *down_global;
    float *gate_input_global;
    float *up_input_global;
    float *down_input_global;
    void *storage;
};
static std::vector<cuda_native_nvfp4_cache> g_native_nvfp4_caches;

static void cuda_native_nvfp4_cache_release_all(void) {
    for (const cuda_native_nvfp4_cache &c : g_native_nvfp4_caches)
        if (c.storage) (void)cudaFree(c.storage);
    g_native_nvfp4_caches.clear();
}

__device__ __forceinline__ static float dev_nvfp4_ue4m3_to_f32(
        uint8_t value) {
    value &= 0x7fu;
    if (value == 0u || value == 0x7fu) return 0.0f;
    const uint32_t exponent = (value >> 3u) & 15u;
    const uint32_t mantissa = value & 7u;
    /*
     * E2M1 values below are represented doubled, so return half of the
     * UE4M3 value.  Constructing the normal float directly avoids an ldexpf
     * sequence in every block-scale dot product.
     */
    if (exponent == 0u) return (float)mantissa * 0x1p-10f;
    const uint32_t bits =
        ((exponent + 119u) << 23u) | (mantissa << 20u);
    return __uint_as_float(bits);
}

__device__ __forceinline__ static int8_t dev_nvfp4_e2m1_i8(
        uint32_t value) {
    static const int8_t table[16] = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12
    };
    return table[value & 15u];
}

__device__ __forceinline__ static int32_t dev_nvfp4_pack4(
        const uint8_t *packed,
        uint32_t high) {
    const uint32_t shift = high ? 4u : 0u;
    return (uint32_t)(uint8_t)dev_nvfp4_e2m1_i8(
               packed[0] >> shift) |
           ((uint32_t)(uint8_t)dev_nvfp4_e2m1_i8(
               packed[1] >> shift) << 8u) |
           ((uint32_t)(uint8_t)dev_nvfp4_e2m1_i8(
               packed[2] >> shift) << 16u) |
           ((uint32_t)(uint8_t)dev_nvfp4_e2m1_i8(
               packed[3] >> shift) << 24u);
}

/* compressed-tensors stores adjacent FP4 values in each byte:
 * low nibble = even element, high nibble = odd element. */
__device__ __forceinline__ static int32_t dev_nvfp4_pack4_adjacent(
        const uint8_t *packed) {
    return (uint32_t)(uint8_t)dev_nvfp4_e2m1_i8(packed[0]) |
           ((uint32_t)(uint8_t)dev_nvfp4_e2m1_i8(
               packed[0] >> 4u) << 8u) |
           ((uint32_t)(uint8_t)dev_nvfp4_e2m1_i8(
               packed[1]) << 16u) |
           ((uint32_t)(uint8_t)dev_nvfp4_e2m1_i8(
               packed[1] >> 4u) << 24u);
}

__device__ __forceinline__ static uint8_t dev_nvfp4_f32_to_e2m1(
        float value, float inverse_scale) {
    const float a = fabsf(value) * inverse_scale;
    uint8_t code =
        a < 0.25f ? 0u :
        a < 0.75f ? 1u :
        a < 1.25f ? 2u :
        a < 1.75f ? 3u :
        a < 2.50f ? 4u :
        a < 3.50f ? 5u :
        a < 5.00f ? 6u : 7u;
    if (value < 0.0f && code != 0u) code |= 8u;
    return code;
}

/* Quantize one selected expert input per 16-value block. compressed-tensors
 * stores the local block scale as E4M3 after multiplying by the calibrated
 * input global scale; kernels dequantize with its reciprocal. */
__global__ static void quantize_nvfp4_selected_f32_kernel(
        uint8_t *packed,
        uint8_t *block_scale,
        const float *x,
        const int32_t *selected,
        const float *input_global_recip,
        uint32_t dim,
        uint32_t groups,
        uint32_t n_expert,
        uint32_t n_pair,
        uint32_t n_selected,
        uint32_t selected_stride,
        uint32_t pair_base,
        bool source_is_pair,
        bool output_int8) {
    const uint32_t group = blockIdx.x;
    const uint32_t pair = pair_base + blockIdx.y;
    if (group >= groups || pair >= n_pair) return;
    const int32_t expert_i =
        selected[(uint64_t)pair * selected_stride];
    if (expert_i < 0 || (uint32_t)expert_i >= n_expert) return;
    const uint32_t source_row =
        source_is_pair ? pair : pair / n_selected;
    const uint32_t lane = threadIdx.x;
    if (lane >= 16u) return;
    const uint32_t col = group * 16u + lane;
    const float value = col < dim ?
        x[(uint64_t)source_row * dim + col] : 0.0f;

    const uint32_t active = 0x0000ffffu;
    float amax = fabsf(value);
#pragma unroll
    for (uint32_t stride = 8u; stride != 0u; stride >>= 1u) {
        amax = fmaxf(
            amax, __shfl_down_sync(active, amax, stride, 16));
    }

    float global_recip = lane == 0u ?
        input_global_recip[(uint32_t)expert_i] : 0.0f;
    global_recip = __shfl_sync(active, global_recip, 0, 16);
    uint32_t encoded_scale = 0u;
    float local_scale = 0.0f;
    if (lane == 0u) {
        local_scale = amax / 6.0f;
        const float scaled = global_recip > 0.0f ?
            local_scale / global_recip : 0.0f;
        encoded_scale = __nv_cvt_float_to_fp8(
            scaled, __NV_SATFINITE, __NV_E4M3);
        block_scale[(uint64_t)pair * groups + group] =
            (uint8_t)encoded_scale;
        local_scale =
            2.0f * dev_nvfp4_ue4m3_to_f32((uint8_t)encoded_scale) *
            global_recip;
    }
    local_scale = __shfl_sync(active, local_scale, 0, 16);
    const float inverse_scale =
        local_scale > 0.0f ? 1.0f / local_scale : 0.0f;
    const uint32_t code =
        dev_nvfp4_f32_to_e2m1(value, inverse_scale);
    const uint32_t packed_lane = lane & 7u;
    const uint32_t even =
        __shfl_sync(active, code, packed_lane * 2u, 16);
    const uint32_t odd =
        __shfl_sync(active, code, packed_lane * 2u + 1u, 16);
    if (output_int8) {
        ((int8_t *)packed)[
            ((uint64_t)pair * groups + group) * 16u + lane] =
                dev_nvfp4_e2m1_i8(code);
    } else if (lane < 8u) {
        packed[((uint64_t)pair * groups + group) * 8u + lane] =
            (uint8_t)(even | (odd << 4u));
    }
}

/* One GGUF NVFP4 super-block spans two Q8_0 activation blocks. Keeping the
 * activation scale per 32 values matches the generic GGUF dot convention,
 * while every E2M1 integer product uses DP4A. */
__device__ __forceinline__ static float dev_dot_nvfp4_q8_0_block(
        const cuda_block_nvfp4 *weight,
        const int8_t *xq,
        const float *xscale) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t sub = 0u; sub < 4u; sub++) {
        const uint8_t *wq = weight->qs + sub * 8u;
        const uint32_t q8_block = sub >> 1u;
        const uint32_t q8_offset = (sub & 1u) * 16u;
        const int8_t *x = xq + q8_block * 32u + q8_offset;
        int32_t dot = 0;
        dot = __dp4a(dev_nvfp4_pack4(wq + 0u, 0u),
                     *(const int32_t *)(x + 0u), dot);
        dot = __dp4a(dev_nvfp4_pack4(wq + 4u, 0u),
                     *(const int32_t *)(x + 4u), dot);
        dot = __dp4a(dev_nvfp4_pack4(wq + 0u, 1u),
                     *(const int32_t *)(x + 8u), dot);
        dot = __dp4a(dev_nvfp4_pack4(wq + 4u, 1u),
                     *(const int32_t *)(x + 12u), dot);
        total += dev_nvfp4_ue4m3_to_f32(weight->d[sub]) *
                 xscale[q8_block] * (float)dot;
    }
    return total;
}

__device__ __forceinline__ static float dev_dot_nvfp4_native_w4a4_group(
        const uint8_t *weight_packed,
        uint8_t weight_scale,
        const int8_t *activation_i8,
        uint8_t activation_scale) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0u; i < 16u; i += 4u) {
        dot = __dp4a(dev_nvfp4_pack4_adjacent(weight_packed + i / 2u),
                     *(const int32_t *)(activation_i8 + i),
                     dot);
    }
    /* dev_nvfp4_ue4m3_to_f32 includes the 1/2 factor for the
     * doubled-integer E2M1 representation. Applying it to both operands
     * supplies the required 1/4 product correction. */
    return (float)dot *
           dev_nvfp4_ue4m3_to_f32(weight_scale) *
           dev_nvfp4_ue4m3_to_f32(activation_scale);
}

__device__ __forceinline__ static void
dev_dot_nvfp4_native_w4a4_gate_up_group(
        const uint8_t *gate_packed,
        uint8_t gate_scale,
        const uint8_t *up_packed,
        uint8_t up_scale,
        const int8_t *activation_i8,
        uint8_t activation_scale,
        float &gate,
        float &up) {
    int32_t gate_dot = 0;
    int32_t up_dot = 0;
#pragma unroll
    for (uint32_t i = 0u; i < 16u; i += 4u) {
        const int32_t activation =
            *(const int32_t *)(activation_i8 + i);
        gate_dot = __dp4a(
            dev_nvfp4_pack4_adjacent(gate_packed + i / 2u),
            activation, gate_dot);
        up_dot = __dp4a(
            dev_nvfp4_pack4_adjacent(up_packed + i / 2u),
            activation, up_dot);
    }
    const float activation_factor =
        dev_nvfp4_ue4m3_to_f32(activation_scale);
    gate += (float)gate_dot *
            dev_nvfp4_ue4m3_to_f32(gate_scale) *
            activation_factor;
    up += (float)up_dot *
          dev_nvfp4_ue4m3_to_f32(up_scale) *
          activation_factor;
}

/*
 * Blackwell GeForce/GB10 exposes NVFP4 through the SM120-family warp MMA:
 * 16 output rows x 8 routed columns x 64 reduction values.  Laguna's official
 * weights and dynamically quantized activations already use the instruction's
 * adjacent E2M1 packing and four UE4M3 scales per K=64 tile, so the operands
 * can be loaded directly into fragments without dequantizing or repacking.
 *
 * The family-specific target is deliberate. Generic sm_12x code cannot issue
 * block-scaled MMA and pre-Blackwell devices should use Laguna Q4_K_M.
 */
#if defined(DS4_CUDA_NVFP4_MMA)
__device__ __forceinline__ static void laguna_nvfp4_mma_m16n8k64(
        float &d0, float &d1, float &d2, float &d3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1,
        uint32_t scale_a, uint32_t scale_b) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale."
        "scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
}

__device__ __forceinline__ static uint32_t laguna_nvfp4_load_u32(
        const uint8_t *p) {
    return *(const uint32_t *)p;
}

__device__ __forceinline__ static void laguna_nvfp4_load_grouped_activation(
        const uint8_t *packed,
        const uint8_t *scales,
        uint32_t source_lo,
        uint32_t source_hi,
        bool valid_lo,
        bool valid_hi,
        uint32_t packed_row,
        uint32_t scale_row,
        uint32_t k0,
        uint32_t lane,
        uint32_t &a0,
        uint32_t &a1,
        uint32_t &a2,
        uint32_t &a3,
        uint32_t &scale_a) {
    const uint32_t tid = lane & 3u;
    const uint32_t byte0 = k0 / 2u + tid * 4u;
    const uint32_t byte1 = byte0 + 16u;
    a0 = valid_lo ? laguna_nvfp4_load_u32(
        packed + (uint64_t)source_lo * packed_row + byte0) : 0u;
    a1 = valid_hi ? laguna_nvfp4_load_u32(
        packed + (uint64_t)source_hi * packed_row + byte0) : 0u;
    a2 = valid_lo ? laguna_nvfp4_load_u32(
        packed + (uint64_t)source_lo * packed_row + byte1) : 0u;
    a3 = valid_hi ? laguna_nvfp4_load_u32(
        packed + (uint64_t)source_hi * packed_row + byte1) : 0u;
    scale_a = 0u;
    if (tid == 0u && valid_lo) {
        scale_a = laguna_nvfp4_load_u32(
            scales + (uint64_t)source_lo * scale_row + k0 / 16u);
    } else if (tid == 1u && valid_hi) {
        scale_a = laguna_nvfp4_load_u32(
            scales + (uint64_t)source_hi * scale_row + k0 / 16u);
    }
}

__device__ __forceinline__ static void laguna_nvfp4_load_grouped_weight(
        const uint8_t *packed,
        const uint8_t *scales,
        uint32_t packed_row,
        uint32_t scale_row,
        uint32_t col0,
        uint32_t k0,
        uint32_t lane,
        uint32_t &b0,
        uint32_t &b1,
        uint32_t &scale_b) {
    const uint32_t col = col0 + (lane >> 2u);
    const uint32_t tid = lane & 3u;
    const uint32_t byte0 = k0 / 2u + tid * 4u;
    b0 = laguna_nvfp4_load_u32(
        packed + (uint64_t)col * packed_row + byte0);
    b1 = laguna_nvfp4_load_u32(
        packed + (uint64_t)col * packed_row + byte0 + 16u);
    scale_b = tid == 0u ? laguna_nvfp4_load_u32(
        scales + (uint64_t)col * scale_row + k0 / 16u) : 0u;
}

/*
 * Prefill groups up to sixteen routes for the same expert into MMA's M
 * dimension and computes eight output channels in N.  Unlike the decode
 * kernel's unavoidable GEMV shape, this consumes the complete 16x8 result
 * tile and reuses each expert weight fragment across sixteen prompt tokens.
 */
__global__ static void laguna_grouped_nvfp4_gate_up_mma_kernel(
        float *mid,
        const uint8_t *const *gate_packed,
        const uint8_t *const *gate_scale,
        const uint8_t *const *up_packed,
        const uint8_t *const *up_scale,
        const float *gate_global,
        const float *up_global,
        const float *input_global,
        const uint8_t *x4,
        const uint8_t *x4scale,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *router_weights,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim) {
    const uint32_t warp_in_block = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t warp =
        (uint64_t)blockIdx.x * (blockDim.x / 32u) + warp_in_block;
    const uint32_t col_tiles = expert_mid_dim / 8u;
    const uint32_t tile = (uint32_t)(warp / col_tiles);
    if (tile >= *tile_total) return;
    const uint32_t col0 =
        (uint32_t)(warp - (uint64_t)tile * col_tiles) * 8u;
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    const uint32_t pair_count = counts[expert];
    const uint32_t sorted_offset = offsets[expert];
    const uint32_t packed_row = expert_in_dim / 2u;
    const uint32_t scale_row = expert_in_dim / 16u;
    const uint32_t group = lane >> 2u;
    const uint32_t tid = lane & 3u;
    const uint32_t local_lo = local_start + group;
    const uint32_t local_hi = local_lo + 8u;
    const bool valid_lo = local_lo < pair_count;
    const bool valid_hi = local_hi < pair_count;
    const uint32_t pair_lo =
        valid_lo ? sorted_pairs[sorted_offset + local_lo] : 0u;
    const uint32_t pair_hi =
        valid_hi ? sorted_pairs[sorted_offset + local_hi] : 0u;
    /* The official checkpoint always routes ten experts per token. */
    const uint32_t source_lo = pair_lo / 10u;
    const uint32_t source_hi = pair_hi / 10u;
    const uint8_t *gate_weight = gate_packed[expert];
    const uint8_t *gate_sf = gate_scale[expert];
    const uint8_t *up_weight = up_packed[expert];
    const uint8_t *up_sf = up_scale[expert];

    float gd[4][4] = {};
    float ud[4][4] = {};
    for (uint32_t kbase = 0u; kbase < expert_in_dim; kbase += 256u) {
#pragma unroll
        for (uint32_t stage = 0u; stage < 4u; stage++) {
            const uint32_t k0 = kbase + stage * 64u;
            uint32_t a0, a1, a2, a3, sa;
            uint32_t gb0, gb1, gsb;
            uint32_t ub0, ub1, usb;
            laguna_nvfp4_load_grouped_activation(
                x4, x4scale, source_lo, source_hi, valid_lo, valid_hi,
                packed_row, scale_row, k0, lane,
                a0, a1, a2, a3, sa);
            laguna_nvfp4_load_grouped_weight(
                gate_weight, gate_sf,
                packed_row, scale_row, col0, k0, lane,
                gb0, gb1, gsb);
            laguna_nvfp4_load_grouped_weight(
                up_weight, up_sf,
                packed_row, scale_row, col0, k0, lane,
                ub0, ub1, usb);
            laguna_nvfp4_mma_m16n8k64(
                gd[stage][0], gd[stage][1],
                gd[stage][2], gd[stage][3],
                a0, a1, a2, a3, gb0, gb1, sa, gsb);
            laguna_nvfp4_mma_m16n8k64(
                ud[stage][0], ud[stage][1],
                ud[stage][2], ud[stage][3],
                a0, a1, a2, a3, ub0, ub1, sa, usb);
        }
    }
#pragma unroll
    for (uint32_t stage = 1u; stage < 4u; stage++) {
#pragma unroll
        for (uint32_t i = 0u; i < 4u; i++) {
            gd[0][i] += gd[stage][i];
            ud[0][i] += ud[stage][i];
        }
    }

    const float gate_factor =
        gate_global[expert] * input_global[expert];
    const float up_factor =
        up_global[expert] * input_global[expert];
    if (valid_lo) {
        const uint32_t pair = pair_lo;
        float gate_value = gd[0][0] * gate_factor;
        float up_value = ud[0][0] * up_factor;
        mid[(uint64_t)pair * expert_mid_dim + col0 + tid * 2u] =
            (gate_value / (1.0f + expf(-gate_value))) *
            up_value * router_weights[pair];
        gate_value = gd[0][1] * gate_factor;
        up_value = ud[0][1] * up_factor;
        mid[(uint64_t)pair * expert_mid_dim + col0 + tid * 2u + 1u] =
            (gate_value / (1.0f + expf(-gate_value))) *
            up_value * router_weights[pair];
    }
    if (valid_hi) {
        const uint32_t pair = pair_hi;
        float gate_value = gd[0][2] * gate_factor;
        float up_value = ud[0][2] * up_factor;
        mid[(uint64_t)pair * expert_mid_dim + col0 + tid * 2u] =
            (gate_value / (1.0f + expf(-gate_value))) *
            up_value * router_weights[pair];
        gate_value = gd[0][3] * gate_factor;
        up_value = ud[0][3] * up_factor;
        mid[(uint64_t)pair * expert_mid_dim + col0 + tid * 2u + 1u] =
            (gate_value / (1.0f + expf(-gate_value))) *
            up_value * router_weights[pair];
    }
}

__global__ static void laguna_grouped_nvfp4_down_mma_kernel(
        float *terms,
        const uint8_t *const *down_packed,
        const uint8_t *const *down_scale,
        const float *down_global,
        const float *input_global,
        const uint8_t *mid4,
        const uint8_t *mid4scale,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint32_t expert_mid_dim,
        uint32_t term_stride,
        uint32_t row0,
        uint32_t chunk_rows) {
    const uint32_t warp_in_block = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t warp =
        (uint64_t)blockIdx.x * (blockDim.x / 32u) + warp_in_block;
    const uint32_t col_tiles = chunk_rows / 8u;
    const uint32_t tile = (uint32_t)(warp / col_tiles);
    if (tile >= *tile_total) return;
    const uint32_t local_col0 =
        (uint32_t)(warp - (uint64_t)tile * col_tiles) * 8u;
    const uint32_t col0 = row0 + local_col0;
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    const uint32_t pair_count = counts[expert];
    const uint32_t sorted_offset = offsets[expert];
    const uint32_t packed_row = expert_mid_dim / 2u;
    const uint32_t scale_row = expert_mid_dim / 16u;
    const uint32_t group = lane >> 2u;
    const uint32_t tid = lane & 3u;
    const uint32_t local_lo = local_start + group;
    const uint32_t local_hi = local_lo + 8u;
    const bool valid_lo = local_lo < pair_count;
    const bool valid_hi = local_hi < pair_count;
    const uint32_t pair_lo =
        valid_lo ? sorted_pairs[sorted_offset + local_lo] : 0u;
    const uint32_t pair_hi =
        valid_hi ? sorted_pairs[sorted_offset + local_hi] : 0u;
    const uint8_t *down_weight = down_packed[expert];
    const uint8_t *down_sf = down_scale[expert];

    float d[4][4] = {};
    for (uint32_t kbase = 0u; kbase < expert_mid_dim; kbase += 256u) {
#pragma unroll
        for (uint32_t stage = 0u; stage < 4u; stage++) {
            const uint32_t k0 = kbase + stage * 64u;
            uint32_t a0, a1, a2, a3, sa;
            uint32_t b0, b1, sb;
            laguna_nvfp4_load_grouped_activation(
                mid4, mid4scale, pair_lo, pair_hi, valid_lo, valid_hi,
                packed_row, scale_row, k0, lane,
                a0, a1, a2, a3, sa);
            laguna_nvfp4_load_grouped_weight(
                down_weight, down_sf,
                packed_row, scale_row, col0, k0, lane,
                b0, b1, sb);
            laguna_nvfp4_mma_m16n8k64(
                d[stage][0], d[stage][1],
                d[stage][2], d[stage][3],
                a0, a1, a2, a3, b0, b1, sa, sb);
        }
    }
#pragma unroll
    for (uint32_t stage = 1u; stage < 4u; stage++) {
#pragma unroll
        for (uint32_t i = 0u; i < 4u; i++)
            d[0][i] += d[stage][i];
    }

    const float factor =
        down_global[expert] * input_global[expert];
    if (valid_lo) {
        const uint32_t pair = pair_lo;
        terms[(uint64_t)pair * term_stride + local_col0 + tid * 2u] =
            d[0][0] * factor;
        terms[(uint64_t)pair * term_stride + local_col0 + tid * 2u + 1u] =
            d[0][1] * factor;
    }
    if (valid_hi) {
        const uint32_t pair = pair_hi;
        terms[(uint64_t)pair * term_stride + local_col0 + tid * 2u] =
            d[0][2] * factor;
        terms[(uint64_t)pair * term_stride + local_col0 + tid * 2u + 1u] =
            d[0][3] * factor;
    }
}

#endif

template <uint32_t LANES>
__device__ __forceinline__ static float laguna_subwarp_sum_f32(float value) {
    static_assert(LANES == 8u || LANES == 16u,
                  "unsupported Laguna NVFP4 reduction width");
#pragma unroll
    for (uint32_t offset = LANES >> 1u; offset != 0u; offset >>= 1u) {
        value += __shfl_down_sync(0xffffffffu, value, offset, LANES);
    }
    return value;
}

template <uint32_t LANES>
__global__ static void laguna_moe_gate_up_nvfp4_kernel(
        float *mid,
        const cuda_block_nvfp4 *gate_base,
        const cuda_block_nvfp4 *up_base,
        const uint8_t *const *gate_native_packed,
        const uint8_t *const *gate_native_scale,
        const uint8_t *const *up_native_packed,
        const uint8_t *const *up_native_scale,
        const float *gate_scales,
        const float *up_scales,
        const int8_t *xq,
        const float *xscale,
        const uint8_t *x4,
        const uint8_t *x4scale,
        const float *input_global,
        const int32_t *selected,
        const float *router_weights,
        uint64_t gate_expert_blocks,
        uint64_t gate_row_blocks,
        uint64_t up_expert_blocks,
        uint64_t up_row_blocks,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t n_tokens,
        bool native_input_by_token) {
    const uint32_t rows_per_block = 256u / LANES;
    const uint32_t row_tiles =
        (expert_mid_dim + rows_per_block - 1u) / rows_per_block;
    const uint64_t flat = blockIdx.x;
    const uint64_t pair = flat / row_tiles;
    const uint32_t row_tile =
        (uint32_t)(flat - pair * row_tiles);
    const uint32_t token = (uint32_t)(pair / n_expert);
    const uint32_t slot =
        (uint32_t)(pair - (uint64_t)token * n_expert);
    if (token >= n_tokens || slot >= n_expert) return;
    const int32_t expert_i = selected[pair];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) return;
    const uint32_t lane = threadIdx.x & (LANES - 1u);
    const uint32_t row_lane = threadIdx.x / LANES;
    const uint32_t row = row_tile * rows_per_block + row_lane;
    if (row >= expert_mid_dim) return;

    const uint32_t expert = (uint32_t)expert_i;
    const bool native = gate_native_packed != NULL;
    const cuda_block_nvfp4 *gate = native ? NULL :
        gate_base + (uint64_t)expert * gate_expert_blocks +
        (uint64_t)row * gate_row_blocks;
    const cuda_block_nvfp4 *up = native ? NULL :
        up_base + (uint64_t)expert * up_expert_blocks +
        (uint64_t)row * up_row_blocks;
    const uint64_t native_row_packed =
        (uint64_t)xq_blocks * 16u;
    const uint64_t native_row_scale =
        (uint64_t)xq_blocks / 2u;
    const uint8_t *gate_packed = native ?
        gate_native_packed[expert] + (uint64_t)row * native_row_packed : NULL;
    const uint8_t *gate_scale = native ?
        gate_native_scale[expert] + (uint64_t)row * native_row_scale : NULL;
    const uint8_t *up_packed = native ?
        up_native_packed[expert] + (uint64_t)row * native_row_packed : NULL;
    const uint8_t *up_scale = native ?
        up_native_scale[expert] + (uint64_t)row * native_row_scale : NULL;
    const int8_t *xq_row =
        xq + (uint64_t)token * xq_blocks * 32u;
    const float *xscale_row =
        xscale + (uint64_t)token * xq_blocks;
    float gate_value = 0.0f;
    float up_value = 0.0f;
    if (native) {
        const uint32_t groups = xq_blocks * 2u;
        const uint64_t input_row =
            native_input_by_token ? token : pair;
        const int8_t *x4_row =
            (const int8_t *)x4 +
            input_row * (uint64_t)groups * 16u;
        const uint8_t *x4scale_row =
            x4scale + input_row * (uint64_t)groups;
        for (uint32_t b = lane; b < groups; b += LANES) {
            dev_dot_nvfp4_native_w4a4_gate_up_group(
                gate_packed + b * 8u, gate_scale[b],
                up_packed + b * 8u, up_scale[b],
                x4_row + b * 16u, x4scale_row[b],
                gate_value, up_value);
        }
    } else {
        const uint32_t nv_blocks = xq_blocks / 2u;
        for (uint32_t b = lane; b < nv_blocks; b += LANES) {
            gate_value += dev_dot_nvfp4_q8_0_block(
                gate + b, xq_row + b * 64u, xscale_row + b * 2u);
            up_value += dev_dot_nvfp4_q8_0_block(
                up + b, xq_row + b * 64u, xscale_row + b * 2u);
        }
    }
    gate_value = laguna_subwarp_sum_f32<LANES>(gate_value);
    up_value = laguna_subwarp_sum_f32<LANES>(up_value);
    if (lane == 0u) {
        const float activation_global =
            native ? input_global[expert] : 1.0f;
        gate_value *= gate_scales[expert] * activation_global;
        up_value *= up_scales[expert] * activation_global;
        mid[pair * expert_mid_dim + row] =
            (gate_value / (1.0f + expf(-gate_value))) *
            up_value * router_weights[pair];
    }
}

template <uint32_t LANES>
__global__ static void laguna_moe_down_nvfp4_kernel(
        float *out,
        const cuda_block_nvfp4 *down_base,
        const uint8_t *const *down_native_packed,
        const uint8_t *const *down_native_scale,
        const float *down_scales,
        const int8_t *midq,
        const float *midscale,
        const uint8_t *mid4,
        const uint8_t *mid4scale,
        const float *input_global,
        const int32_t *selected,
        uint64_t down_expert_blocks,
        uint64_t down_row_blocks,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t rows_per_block = 256u / LANES;
    const uint32_t row_tiles =
        (out_dim + rows_per_block - 1u) / rows_per_block;
    const uint64_t flat = blockIdx.x;
    const uint32_t token = (uint32_t)(flat / row_tiles);
    const uint32_t row_tile =
        (uint32_t)(flat - (uint64_t)token * row_tiles);
    if (token >= n_tokens) return;
    const uint32_t lane = threadIdx.x & (LANES - 1u);
    const uint32_t row_lane = threadIdx.x / LANES;
    const uint32_t row = row_tile * rows_per_block + row_lane;
    if (row >= out_dim) return;

    float total = 0.0f;
    for (uint32_t slot = 0u; slot < n_expert; slot++) {
        const uint64_t pair = (uint64_t)token * n_expert + slot;
        const int32_t expert_i = selected[pair];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) continue;
        const uint32_t expert = (uint32_t)expert_i;
        const bool native = down_native_packed != NULL;
        const cuda_block_nvfp4 *weight = native ? NULL :
            down_base + (uint64_t)expert * down_expert_blocks +
            (uint64_t)row * down_row_blocks;
        const uint64_t native_row_packed =
            (uint64_t)midq_blocks * 16u;
        const uint64_t native_row_scale =
            (uint64_t)midq_blocks / 2u;
        const uint8_t *packed = native ?
            down_native_packed[expert] +
                (uint64_t)row * native_row_packed : NULL;
        const uint8_t *block_scale = native ?
            down_native_scale[expert] +
                (uint64_t)row * native_row_scale : NULL;
        float part = 0.0f;
        if (native) {
            const uint32_t groups = midq_blocks * 2u;
            const int8_t *mid4_row =
                (const int8_t *)mid4 + pair * (uint64_t)groups * 16u;
            const uint8_t *mid4scale_row =
                mid4scale + pair * (uint64_t)groups;
            for (uint32_t b = lane; b < groups; b += LANES) {
                part += dev_dot_nvfp4_native_w4a4_group(
                    packed + b * 8u, block_scale[b],
                    mid4_row + b * 16u, mid4scale_row[b]);
            }
        } else {
            const uint32_t nv_blocks = midq_blocks / 2u;
            const int8_t *midq_row =
                midq + pair * midq_blocks * 32u;
            const float *midscale_row =
                midscale + pair * midq_blocks;
            for (uint32_t b = lane; b < nv_blocks; b += LANES) {
                part += dev_dot_nvfp4_q8_0_block(
                    weight + b, midq_row + b * 64u,
                    midscale_row + b * 2u);
            }
        }
        part = laguna_subwarp_sum_f32<LANES>(part);
        if (lane == 0u) {
            total += down_scales[expert] *
                     (native ? input_global[expert] : 1.0f) * part;
        }
    }
    if (lane == 0u) {
        out[(uint64_t)token * out_dim + row] = total;
    }
}

__global__ static void laguna_moe_down_chunk_reduce_kernel(
        float *out,
        const float *terms,
        uint32_t n_tokens,
        uint32_t n_expert,
        uint32_t out_dim,
        uint32_t term_stride,
        uint32_t row0,
        uint32_t chunk_rows) {
    const uint64_t i =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * chunk_rows;
    if (i >= n) return;
    const uint32_t token = (uint32_t)(i / chunk_rows);
    const uint32_t local_row =
        (uint32_t)(i - (uint64_t)token * chunk_rows);
    float total = 0.0f;
    for (uint32_t slot = 0; slot < n_expert; slot++) {
        const uint64_t pair = (uint64_t)token * n_expert + slot;
        total += terms[pair * term_stride + local_row];
    }
    out[(uint64_t)token * out_dim + row0 + local_row] = total;
}

static const cuda_native_nvfp4_cache *cuda_native_nvfp4_prepare(
        const void *model_map, uint64_t model_size,
        const ds4_gpu_laguna_moe_desc *desc,
        uint32_t n_total_expert, int logical_tier) {
    const ds4_gpu_nvfp4_matrix_desc *matrix[3] = {
        &desc->gate_nvfp4, &desc->up_nvfp4, &desc->down_nvfp4
    };
    if (!matrix[0]->packed_offsets) return NULL;
    for (const cuda_native_nvfp4_cache &cached : g_native_nvfp4_caches)
        if (cached.key == matrix[0]->packed_offsets) return &cached;

    for (uint32_t p = 0; p < 3u; p++) {
        if (!matrix[p]->packed_offsets || !matrix[p]->scale_offsets ||
            !matrix[p]->weight_global_scale_offsets ||
            !matrix[p]->input_global_scale_offsets ||
            matrix[p]->packed_bytes == 0u ||
            matrix[p]->scale_bytes == 0u) {
            return NULL;
        }
    }
    std::vector<const uint8_t *> packed[3];
    std::vector<const uint8_t *> scale[3];
    std::vector<float> global[3];
    std::vector<float> input_global[3];
    for (uint32_t p = 0; p < 3u; p++) {
        packed[p].resize(n_total_expert);
        scale[p].resize(n_total_expert);
        global[p].resize(n_total_expert);
        input_global[p].resize(n_total_expert);
        for (uint32_t e = 0; e < n_total_expert; e++) {
            const uint64_t po = matrix[p]->packed_offsets[e];
            const uint64_t so = matrix[p]->scale_offsets[e];
            const uint64_t go = matrix[p]->weight_global_scale_offsets[e];
            const uint64_t io = matrix[p]->input_global_scale_offsets[e];
            if (po > model_size ||
                matrix[p]->packed_bytes > model_size - po ||
                so > model_size ||
                matrix[p]->scale_bytes > model_size - so ||
                go > model_size || sizeof(float) > model_size - go ||
                io > model_size || sizeof(float) > model_size - io) {
                return NULL;
            }
            packed[p][e] = (const uint8_t *)cuda_resolve_weight_ptr(
                model_map, po, matrix[p]->packed_bytes, logical_tier,
                p == 0u ? "native NVFP4 gate packed" :
                p == 1u ? "native NVFP4 up packed" :
                          "native NVFP4 down packed");
            scale[p][e] = (const uint8_t *)cuda_resolve_weight_ptr(
                model_map, so, matrix[p]->scale_bytes, logical_tier,
                p == 0u ? "native NVFP4 gate scales" :
                p == 1u ? "native NVFP4 up scales" :
                          "native NVFP4 down scales");
            float raw_global = 0.0f;
            float raw_input_global = 0.0f;
            memcpy(&raw_global, (const uint8_t *)model_map + go,
                   sizeof(raw_global));
            memcpy(&raw_input_global, (const uint8_t *)model_map + io,
                   sizeof(raw_input_global));
            if (!packed[p][e] || !scale[p][e] ||
                !isfinite(raw_global) || raw_global <= 0.0f ||
                !isfinite(raw_input_global) || raw_input_global <= 0.0f) {
                return NULL;
            }
            /* compressed-tensors serializes the quantization multiplier.
             * CUDA kernels consume the dequantization multiplier. */
            global[p][e] = 1.0f / raw_global;
            input_global[p][e] = 1.0f / raw_input_global;
        }
    }
    /* Gate and up consume the same source activation and official Laguna
     * checkpoints calibrate them identically, allowing one W4A4 input. */
    for (uint32_t e = 0; e < n_total_expert; e++)
        if (input_global[0][e] != input_global[1][e]) return NULL;
    /*
     * This backend supports Poolside's fixed Laguna-S-2.1-NVFP4 checkpoint,
     * whose routed gate/up input calibration is one scalar per layer (not
     * per expert). Enforce that invariant before quantizing each token once
     * and sharing it across the model's ten selected experts.
     */
    for (uint32_t e = 1; e < n_total_expert; e++)
        if (input_global[0][e] != input_global[0][0]) return NULL;

    const uint64_t pointer_bytes =
        (uint64_t)n_total_expert * sizeof(const uint8_t *);
    const uint64_t global_bytes =
        (uint64_t)n_total_expert * sizeof(float);
    const uint64_t total = pointer_bytes * 6u + global_bytes * 6u;
    void *storage = NULL;
    if (!cuda_ok(cudaMalloc(&storage, (size_t)total),
                 "native NVFP4 descriptor cache allocation")) {
        return NULL;
    }
    uint8_t *cursor = (uint8_t *)storage;
    cuda_native_nvfp4_cache cached = {};
    cached.key = matrix[0]->packed_offsets;
    cached.storage = storage;
#define DS4_NVFP4_COPY_PTRS(dst_field_, src_) do { \
        cached.dst_field_ = (const uint8_t **)cursor; \
        if (!cuda_ok(cudaMemcpy(cursor, (src_).data(), \
                                (size_t)pointer_bytes, \
                                cudaMemcpyHostToDevice), \
                     "native NVFP4 pointer table upload")) { \
            (void)cudaFree(storage); \
            return NULL; \
        } \
        cursor += pointer_bytes; \
    } while (0)
    DS4_NVFP4_COPY_PTRS(gate_packed, packed[0]);
    DS4_NVFP4_COPY_PTRS(gate_scale, scale[0]);
    DS4_NVFP4_COPY_PTRS(up_packed, packed[1]);
    DS4_NVFP4_COPY_PTRS(up_scale, scale[1]);
    DS4_NVFP4_COPY_PTRS(down_packed, packed[2]);
    DS4_NVFP4_COPY_PTRS(down_scale, scale[2]);
#undef DS4_NVFP4_COPY_PTRS
#define DS4_NVFP4_COPY_GLOBAL(dst_field_, src_) do { \
        cached.dst_field_ = (float *)cursor; \
        if (!cuda_ok(cudaMemcpy(cursor, (src_).data(), \
                                (size_t)global_bytes, \
                                cudaMemcpyHostToDevice), \
                     "native NVFP4 global-scale upload")) { \
            (void)cudaFree(storage); \
            return NULL; \
        } \
        cursor += global_bytes; \
    } while (0)
    DS4_NVFP4_COPY_GLOBAL(gate_global, global[0]);
    DS4_NVFP4_COPY_GLOBAL(up_global, global[1]);
    DS4_NVFP4_COPY_GLOBAL(down_global, global[2]);
    DS4_NVFP4_COPY_GLOBAL(gate_input_global, input_global[0]);
    DS4_NVFP4_COPY_GLOBAL(up_input_global, input_global[1]);
    DS4_NVFP4_COPY_GLOBAL(down_input_global, input_global[2]);
#undef DS4_NVFP4_COPY_GLOBAL
    g_native_nvfp4_caches.push_back(cached);
    return &g_native_nvfp4_caches.back();
}

static int cuda_laguna_nvfp4_routed_moe(
        ds4_gpu_tensor                *out,
        ds4_gpu_tensor                *mid,
        const void                    *model_map,
        uint64_t                       model_size,
        const ds4_gpu_laguna_moe_desc *desc,
        uint32_t                       expert_in_dim,
        uint32_t                       expert_mid_dim,
        uint32_t                       out_dim,
        const ds4_gpu_tensor          *selected,
        const ds4_gpu_tensor          *weights,
        uint32_t                       n_total_expert,
        uint32_t                       n_expert,
        const ds4_gpu_tensor          *x,
        uint32_t                       n_tokens) {
    if (!out || !mid || !model_map || !desc || !selected || !weights || !x ||
        n_tokens == 0u || n_total_expert == 0u || n_expert == 0u ||
        n_expert > n_total_expert ||
        desc->gate_type != 40u || desc->up_type != 40u ||
        desc->down_type != 40u ||
        (expert_in_dim & 63u) != 0u ||
        (expert_mid_dim & 63u) != 0u ||
        (out_dim & 31u) != 0u) {
        return 0;
    }
    const uint64_t nvfp4_block_bytes = sizeof(cuda_block_nvfp4);
    const uint64_t gate_row_bytes =
        (uint64_t)(expert_in_dim / 64u) * nvfp4_block_bytes;
    const uint64_t down_row_bytes =
        (uint64_t)(expert_mid_dim / 64u) * nvfp4_block_bytes;
    if (desc->gate_row_bytes != gate_row_bytes ||
        desc->up_row_bytes != gate_row_bytes ||
        desc->down_row_bytes != down_row_bytes ||
        gate_row_bytes > UINT64_MAX / expert_mid_dim ||
        down_row_bytes > UINT64_MAX / out_dim ||
        desc->gate_expert_bytes !=
            gate_row_bytes * expert_mid_dim ||
        desc->up_expert_bytes !=
            gate_row_bytes * expert_mid_dim ||
        desc->down_expert_bytes != down_row_bytes * out_dim) {
        return 0;
    }
    const uint64_t pairs = (uint64_t)n_tokens * n_expert;
    const bool native = desc->gate_nvfp4.packed_offsets != NULL;
    const uint64_t scale_bytes =
        (uint64_t)n_total_expert * sizeof(float);
    if (pairs > UINT32_MAX ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < pairs * sizeof(int32_t) ||
        weights->bytes < pairs * sizeof(float) ||
        mid->bytes < pairs * expert_mid_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float) ||
        n_total_expert > UINT64_MAX / desc->gate_expert_bytes ||
        n_total_expert > UINT64_MAX / desc->up_expert_bytes ||
        n_total_expert > UINT64_MAX / desc->down_expert_bytes) {
        return 0;
    }
    const uint64_t gate_bytes =
        (uint64_t)n_total_expert * desc->gate_expert_bytes;
    const uint64_t up_bytes =
        (uint64_t)n_total_expert * desc->up_expert_bytes;
    const uint64_t down_bytes =
        (uint64_t)n_total_expert * desc->down_expert_bytes;
#define DS4_NVFP4_MODEL_RANGE_OK(offset_, bytes_) \
    ((offset_) <= model_size && (bytes_) <= model_size - (offset_))
    if (!native &&
        (!DS4_NVFP4_MODEL_RANGE_OK(desc->gate_offset, gate_bytes) ||
         !DS4_NVFP4_MODEL_RANGE_OK(desc->up_offset, up_bytes) ||
         !DS4_NVFP4_MODEL_RANGE_OK(desc->down_offset, down_bytes) ||
         !DS4_NVFP4_MODEL_RANGE_OK(desc->gate_scale_offset, scale_bytes) ||
         !DS4_NVFP4_MODEL_RANGE_OK(desc->up_scale_offset, scale_bytes) ||
         !DS4_NVFP4_MODEL_RANGE_OK(desc->down_scale_offset, scale_bytes))) {
#undef DS4_NVFP4_MODEL_RANGE_OK
        return 0;
    }
#undef DS4_NVFP4_MODEL_RANGE_OK

    const int logical_tier = ds4_tensor_device_idx(out);
    if (ds4_tensor_device_idx(mid) != logical_tier ||
        ds4_tensor_device_idx(selected) != logical_tier ||
        ds4_tensor_device_idx(weights) != logical_tier ||
        ds4_tensor_device_idx(x) != logical_tier) {
        return 0;
    }
    const cuda_block_nvfp4 *gate = NULL;
    const cuda_block_nvfp4 *up = NULL;
    const cuda_block_nvfp4 *down = NULL;
    const float *gate_scales = NULL;
    const float *up_scales = NULL;
    const float *down_scales = NULL;
    const cuda_native_nvfp4_cache *native_cache = NULL;
    if (native) {
        if (desc->gate_nvfp4.packed_bytes !=
                (uint64_t)expert_mid_dim * expert_in_dim / 2u ||
            desc->up_nvfp4.packed_bytes !=
                (uint64_t)expert_mid_dim * expert_in_dim / 2u ||
            desc->down_nvfp4.packed_bytes !=
                (uint64_t)out_dim * expert_mid_dim / 2u ||
            desc->gate_nvfp4.scale_bytes !=
                (uint64_t)expert_mid_dim * expert_in_dim / 16u ||
            desc->up_nvfp4.scale_bytes !=
                (uint64_t)expert_mid_dim * expert_in_dim / 16u ||
            desc->down_nvfp4.scale_bytes !=
                (uint64_t)out_dim * expert_mid_dim / 16u) {
            return 0;
        }
        native_cache = cuda_native_nvfp4_prepare(
            model_map, model_size, desc, n_total_expert, logical_tier);
        if (!native_cache) return 0;
        gate_scales = native_cache->gate_global;
        up_scales = native_cache->up_global;
        down_scales = native_cache->down_global;
    } else {
        gate = (const cuda_block_nvfp4 *)cuda_resolve_weight_ptr(
            model_map, desc->gate_offset, gate_bytes, logical_tier,
            "Laguna NVFP4 MoE gate");
        up = (const cuda_block_nvfp4 *)cuda_resolve_weight_ptr(
            model_map, desc->up_offset, up_bytes, logical_tier,
            "Laguna NVFP4 MoE up");
        down = (const cuda_block_nvfp4 *)cuda_resolve_weight_ptr(
            model_map, desc->down_offset, down_bytes, logical_tier,
            "Laguna NVFP4 MoE down");
        gate_scales = (const float *)cuda_resolve_weight_ptr(
            model_map, desc->gate_scale_offset, scale_bytes, logical_tier,
            "Laguna NVFP4 MoE gate scales");
        up_scales = (const float *)cuda_resolve_weight_ptr(
            model_map, desc->up_scale_offset, scale_bytes, logical_tier,
            "Laguna NVFP4 MoE up scales");
        down_scales = (const float *)cuda_resolve_weight_ptr(
            model_map, desc->down_scale_offset, scale_bytes, logical_tier,
            "Laguna NVFP4 MoE down scales");
        if (!gate || !up || !down ||
            !gate_scales || !up_scales || !down_scales) {
            return 0;
        }
    }

    const uint32_t xq_blocks = expert_in_dim / 32u;
    const uint32_t midq_blocks = expert_mid_dim / 32u;
    const bool grouped_native =
        native && n_expert == 10u && pairs <= UINT32_MAX &&
#if defined(DS4_CUDA_NVFP4_MMA)
        n_tokens >= cuda_laguna_nvfp4_grouped_min_tokens();
#else
        n_tokens >= 256u;
#endif
    const bool shared_native_dp4a_input =
        native && !grouped_native &&
        expert_in_dim == 3072u && expert_mid_dim == 1024u &&
        out_dim == 3072u && n_total_expert == 256u &&
        n_expert == 10u;
    const bool shared_native_gate_input =
        grouped_native || shared_native_dp4a_input;
    const uint64_t native_input_rows =
        shared_native_gate_input ? n_tokens : pairs;
    const uint64_t xq_bytes =
        native ? native_input_rows * expert_in_dim /
                     (grouped_native ? 2u : 1u) :
                 (uint64_t)n_tokens * expert_in_dim * sizeof(int8_t);
    const uint64_t xscale_offset = (xq_bytes + 255u) & ~255ull;
    const uint64_t xscale_bytes =
        native ? native_input_rows * expert_in_dim / 16u :
                 (uint64_t)n_tokens * xq_blocks * sizeof(float);
    if (xscale_offset > UINT64_MAX - xscale_bytes) return 0;
    const uint64_t midq_offset =
        (xscale_offset + xscale_bytes + 255u) & ~255ull;
    const uint64_t midq_bytes =
        native ? pairs * expert_mid_dim /
                     (grouped_native ? 2u : 1u) :
                 pairs * expert_mid_dim * sizeof(int8_t);
    if (midq_offset > UINT64_MAX - midq_bytes) return 0;
    const uint64_t midscale_offset =
        (midq_offset + midq_bytes + 255u) & ~255ull;
    const uint64_t midscale_bytes =
        native ? pairs * expert_mid_dim / 16u :
                 pairs * midq_blocks * sizeof(float);
    if (midscale_offset > UINT64_MAX - midscale_bytes) return 0;
    const uint64_t sorted_base =
        (midscale_offset + midscale_bytes + 255u) & ~255ull;
    const uint64_t counts_bytes = grouped_native ?
        (uint64_t)n_total_expert * sizeof(uint32_t) : 0u;
    const uint64_t offsets_bytes = grouped_native ?
        ((uint64_t)n_total_expert + 1u) * sizeof(uint32_t) : 0u;
    const uint64_t cursors_bytes = counts_bytes;
    const uint64_t sorted_pairs_bytes = grouped_native ?
        pairs * sizeof(uint32_t) : 0u;
    const uint64_t tile_offsets_bytes = offsets_bytes;
    /*
     * Each non-empty expert needs one tile; another tile requires sixteen
     * routes beyond that expert's first.  This is a tight upper bound and
     * avoids launching hundreds of empty expert tiles for verifier batches.
     */
    const uint64_t first_tiles =
        grouped_native ?
            (pairs < n_total_expert ? pairs : n_total_expert) : 0u;
    const uint64_t tile_capacity = grouped_native ?
        first_tiles + (pairs - first_tiles) / 16u : 0u;
    const uint64_t tile_total_bytes = grouped_native ?
        sizeof(uint32_t) : 0u;
    const uint64_t tile_experts_bytes =
        tile_capacity * sizeof(uint32_t);
    const uint64_t tile_starts_bytes = tile_experts_bytes;
    const uint64_t sorted_bytes =
        counts_bytes + offsets_bytes + cursors_bytes +
        sorted_pairs_bytes + tile_offsets_bytes + tile_total_bytes +
        tile_experts_bytes + tile_starts_bytes;
    if (grouped_native &&
        (tile_capacity > UINT32_MAX ||
         sorted_base > UINT64_MAX - sorted_bytes)) {
        return 0;
    }
    const uint64_t scratch_bytes = grouped_native ?
        sorted_base + sorted_bytes : midscale_offset + midscale_bytes;
    uint8_t *scratch = (uint8_t *)cuda_tmp_alloc_on(
        logical_tier, scratch_bytes,
        "Laguna NVFP4 MoE");
    if (!scratch) return 0;
    int8_t *xq = (int8_t *)scratch;
    float *xscale = (float *)(scratch + xscale_offset);
    int8_t *midq = (int8_t *)(scratch + midq_offset);
    float *midscale = (float *)(scratch + midscale_offset);
    uint8_t *x4 = native ? scratch : NULL;
    uint8_t *x4scale = native ? scratch + xscale_offset : NULL;
    uint8_t *mid4 = native ? scratch + midq_offset : NULL;
    uint8_t *mid4scale = native ? scratch + midscale_offset : NULL;
    uint32_t *sorted_counts = NULL;
    uint32_t *sorted_offsets = NULL;
    uint32_t *sorted_pairs = NULL;
    uint32_t *tile_total = NULL;
    uint32_t *tile_experts = NULL;
    uint32_t *tile_starts = NULL;

    if (native) {
        const uint32_t input_rows = (uint32_t)native_input_rows;
        for (uint32_t pair0 = 0u; pair0 < input_rows;) {
            uint32_t chunk = input_rows - pair0;
            if (chunk > 65535u) chunk = 65535u;
            quantize_nvfp4_selected_f32_kernel
                <<<dim3(expert_in_dim / 16u, chunk, 1u), 32>>>(
                    x4, x4scale, (const float *)x->ptr,
                    (const int32_t *)selected->ptr,
                    native_cache->gate_input_global,
                    expert_in_dim, expert_in_dim / 16u,
                    n_total_expert, input_rows,
                    shared_native_gate_input ? 1u : n_expert,
                    shared_native_gate_input ? n_expert : 1u,
                    pair0, false,
                    !grouped_native);
            if (!cuda_ok(cudaGetLastError(),
                         "Laguna native NVFP4 input quantize launch")) {
                return 0;
            }
            pair0 += chunk;
        }
    } else {
        for (uint32_t token0 = 0u; token0 < n_tokens;) {
            uint32_t chunk = n_tokens - token0;
            if (chunk > 65535u) chunk = 65535u;
            quantize_q8_0_f32_kernel<<<dim3(xq_blocks, chunk, 1u), 32>>>(
                xq + (uint64_t)token0 * expert_in_dim,
                xscale + (uint64_t)token0 * xq_blocks,
                (const float *)x->ptr +
                    (uint64_t)token0 * expert_in_dim,
                expert_in_dim, xq_blocks);
            if (!cuda_ok(cudaGetLastError(),
                         "Laguna NVFP4 input quantize launch")) {
                return 0;
            }
            token0 += chunk;
        }
    }

    if (grouped_native) {
        uint8_t *cursor = scratch + sorted_base;
        sorted_counts = (uint32_t *)cursor;
        cursor += counts_bytes;
        sorted_offsets = (uint32_t *)cursor;
        cursor += offsets_bytes;
        uint32_t *cursors = (uint32_t *)cursor;
        cursor += cursors_bytes;
        sorted_pairs = (uint32_t *)cursor;
        cursor += sorted_pairs_bytes;
        uint32_t *tile_offsets = (uint32_t *)cursor;
        cursor += tile_offsets_bytes;
        tile_total = (uint32_t *)cursor;
        cursor += tile_total_bytes;
        tile_experts = (uint32_t *)cursor;
        cursor += tile_experts_bytes;
        tile_starts = (uint32_t *)cursor;
        if (!cuda_ok(cudaMemsetAsync(
                         sorted_counts, 0, (size_t)counts_bytes),
                     "Laguna native NVFP4 sorted counts clear")) {
            return 0;
        }
        moe_count_sorted_pairs_kernel<<<
            ((uint32_t)pairs + 255u) / 256u, 256>>>(
                sorted_counts, (const int32_t *)selected->ptr,
                (uint32_t)pairs, n_total_expert);
        if (!cuda_ok(cudaGetLastError(),
                     "Laguna native NVFP4 sorted count launch")) {
            return 0;
        }
        moe_prefix_sorted_pairs_kernel<<<1, 1>>>(
            sorted_offsets, cursors, sorted_counts, n_total_expert);
        if (!cuda_ok(cudaGetLastError(),
                     "Laguna native NVFP4 sorted prefix launch")) {
            return 0;
        }
        moe_scatter_sorted_pairs_kernel<<<
            ((uint32_t)pairs + 255u) / 256u, 256>>>(
                sorted_pairs, cursors,
                (const int32_t *)selected->ptr,
                (uint32_t)pairs, n_total_expert);
        if (!cuda_ok(cudaGetLastError(),
                     "Laguna native NVFP4 sorted scatter launch")) {
            return 0;
        }
        moe_build_expert_tile_offsets_kernel<<<1, 1>>>(
            tile_offsets, tile_total, sorted_counts, 16u,
            n_total_expert);
        if (!cuda_ok(cudaGetLastError(),
                     "Laguna native NVFP4 sorted tile-offset launch")) {
            return 0;
        }
        moe_build_expert_tiles_kernel<<<
            (n_total_expert + 255u) / 256u, 256>>>(
                tile_experts, tile_starts, tile_offsets,
                sorted_counts, 16u, n_total_expert);
        if (!cuda_ok(cudaGetLastError(),
                     "Laguna native NVFP4 sorted tile launch")) {
            return 0;
        }
    }

    if (native) {
#if defined(DS4_CUDA_NVFP4_MMA)
        if (!cuda_laguna_nvfp4_hw_ok()) {
            fprintf(stderr,
                    "ds4: native Laguna NVFP4 requires a Blackwell GPU; "
                    "use the official Q4_K_M model on older CUDA devices\n");
            return 0;
        }
        if (grouped_native) {
            const uint64_t gate_warps =
                tile_capacity * (expert_mid_dim / 8u);
            const uint64_t gate_blocks = (gate_warps + 7u) / 8u;
            if (gate_blocks > UINT32_MAX) return 0;
            laguna_grouped_nvfp4_gate_up_mma_kernel<<<
                (uint32_t)gate_blocks, 256>>>(
                    (float *)mid->ptr,
                    native_cache->gate_packed,
                    native_cache->gate_scale,
                    native_cache->up_packed,
                    native_cache->up_scale,
                    gate_scales,
                    up_scales,
                    native_cache->gate_input_global,
                    x4,
                    x4scale,
                    sorted_pairs,
                    sorted_offsets,
                    sorted_counts,
                    tile_total,
                    tile_experts,
                    tile_starts,
                    (const float *)weights->ptr,
                    expert_in_dim,
                    expert_mid_dim);
            if (!cuda_ok(cudaGetLastError(),
                         "Laguna Blackwell grouped NVFP4 gate/up MMA launch")) {
                return 0;
            }

            for (uint32_t pair0 = 0u; pair0 < (uint32_t)pairs;) {
                uint32_t chunk = (uint32_t)pairs - pair0;
                if (chunk > 65535u) chunk = 65535u;
                quantize_nvfp4_selected_f32_kernel
                    <<<dim3(expert_mid_dim / 16u, chunk, 1u), 32>>>(
                        mid4, mid4scale, (const float *)mid->ptr,
                        (const int32_t *)selected->ptr,
                        native_cache->down_input_global,
                        expert_mid_dim, expert_mid_dim / 16u,
                        n_total_expert, (uint32_t)pairs, n_expert,
                        1u, pair0, true, false);
                if (!cuda_ok(cudaGetLastError(),
                             "Laguna Blackwell NVFP4 mid quantize launch")) {
                    return 0;
                }
                pair0 += chunk;
            }

            for (uint32_t row0 = 0u; row0 < out_dim;
                 row0 += expert_mid_dim) {
                uint32_t chunk_rows = out_dim - row0;
                if (chunk_rows > expert_mid_dim)
                    chunk_rows = expert_mid_dim;
                const uint64_t down_warps =
                    tile_capacity * (chunk_rows / 8u);
                const uint64_t down_blocks =
                    (down_warps + 7u) / 8u;
                if (down_blocks > UINT32_MAX) return 0;
                laguna_grouped_nvfp4_down_mma_kernel<<<
                    (uint32_t)down_blocks, 256>>>(
                        (float *)mid->ptr,
                        native_cache->down_packed,
                        native_cache->down_scale,
                        down_scales,
                        native_cache->down_input_global,
                        mid4,
                        mid4scale,
                        sorted_pairs,
                        sorted_offsets,
                        sorted_counts,
                        tile_total,
                        tile_experts,
                        tile_starts,
                        expert_mid_dim,
                        expert_mid_dim,
                        row0,
                        chunk_rows);
                if (!cuda_ok(cudaGetLastError(),
                             "Laguna Blackwell grouped NVFP4 down MMA launch")) {
                    return 0;
                }
                const uint64_t reduce_values =
                    (uint64_t)n_tokens * chunk_rows;
                if (reduce_values >
                    (uint64_t)UINT32_MAX * 256u) {
                    return 0;
                }
                laguna_moe_down_chunk_reduce_kernel<<<
                    (uint32_t)((reduce_values + 255u) / 256u), 256>>>(
                        (float *)out->ptr,
                        (const float *)mid->ptr,
                        n_tokens,
                        n_expert,
                        out_dim,
                        expert_mid_dim,
                        row0,
                        chunk_rows);
                if (!cuda_ok(cudaGetLastError(),
                             "Laguna Blackwell grouped NVFP4 down reduce launch")) {
                    return 0;
                }
            }
            return cuda_ok(cudaGetLastError(),
                           "Laguna Blackwell grouped NVFP4 down MMA launch");
        }
#else
        fprintf(stderr,
                "ds4: this CUDA build has no native Blackwell NVFP4 MMA; "
                "build with make cuda-spark\n");
        return 0;
#endif
    }

    /*
     * SM120-family MMA has no GEMV-sized NVFP4 instruction: m16n8k64 would
     * discard seven of eight columns for a single routed vector.  Keep the
     * native integer-dot decode kernel for that shape; prompt batches above
     * use the fully occupied grouped FP4 MMA path and have already returned.
     */
    const bool blackwell = native || cuda_laguna_blackwell_ok();
    const uint32_t lanes = blackwell ? 16u : 8u;
    const uint32_t rows_per_block = 256u / lanes;
    const uint64_t gate_blocks =
        pairs * ((expert_mid_dim + rows_per_block - 1u) / rows_per_block);
    if (gate_blocks > UINT32_MAX) return 0;
#define DS4_LAUNCH_NVFP4_GATE_UP(LANES_) \
    laguna_moe_gate_up_nvfp4_kernel<LANES_> \
        <<<(uint32_t)gate_blocks, 256>>>( \
            (float *)mid->ptr, \
            gate, up, \
            native_cache ? native_cache->gate_packed : NULL, \
            native_cache ? native_cache->gate_scale : NULL, \
            native_cache ? native_cache->up_packed : NULL, \
            native_cache ? native_cache->up_scale : NULL, \
            gate_scales, up_scales, \
            xq, xscale, x4, x4scale, \
            native_cache ? native_cache->gate_input_global : NULL, \
            (const int32_t *)selected->ptr, \
            (const float *)weights->ptr, \
            desc->gate_expert_bytes / nvfp4_block_bytes, \
            desc->gate_row_bytes / nvfp4_block_bytes, \
            desc->up_expert_bytes / nvfp4_block_bytes, \
            desc->up_row_bytes / nvfp4_block_bytes, \
            xq_blocks, expert_mid_dim, n_total_expert, n_expert, n_tokens, \
            shared_native_dp4a_input)
    if (blackwell) {
        DS4_LAUNCH_NVFP4_GATE_UP(16u);
    } else {
        DS4_LAUNCH_NVFP4_GATE_UP(8u);
    }
#undef DS4_LAUNCH_NVFP4_GATE_UP
    if (!cuda_ok(cudaGetLastError(),
                 "Laguna NVFP4 gate/up launch")) {
        return 0;
    }

    for (uint32_t pair0 = 0u; pair0 < (uint32_t)pairs;) {
        uint32_t chunk = (uint32_t)pairs - pair0;
        if (chunk > 65535u) chunk = 65535u;
        if (native) {
            quantize_nvfp4_selected_f32_kernel
                <<<dim3(expert_mid_dim / 16u, chunk, 1u), 32>>>(
                    mid4, mid4scale, (const float *)mid->ptr,
                    (const int32_t *)selected->ptr,
                    native_cache->down_input_global,
                    expert_mid_dim, expert_mid_dim / 16u,
                    n_total_expert, (uint32_t)pairs, n_expert,
                    1u, pair0, true, true);
        } else {
            quantize_q8_0_f32_kernel<<<dim3(midq_blocks, chunk, 1u), 32>>>(
                midq + (uint64_t)pair0 * expert_mid_dim,
                midscale + (uint64_t)pair0 * midq_blocks,
                (const float *)mid->ptr +
                    (uint64_t)pair0 * expert_mid_dim,
                expert_mid_dim, midq_blocks);
        }
        if (!cuda_ok(cudaGetLastError(),
                     native ? "Laguna native NVFP4 mid quantize launch" :
                              "Laguna NVFP4 mid quantize launch")) {
            return 0;
        }
        pair0 += chunk;
    }

    const uint64_t down_blocks =
        (uint64_t)n_tokens *
        ((out_dim + rows_per_block - 1u) / rows_per_block);
    if (down_blocks > UINT32_MAX) return 0;
#define DS4_LAUNCH_NVFP4_DOWN(LANES_) \
    laguna_moe_down_nvfp4_kernel<LANES_> \
        <<<(uint32_t)down_blocks, 256>>>( \
            (float *)out->ptr, down, \
            native_cache ? native_cache->down_packed : NULL, \
            native_cache ? native_cache->down_scale : NULL, \
            down_scales, midq, midscale, mid4, mid4scale, \
            native_cache ? native_cache->down_input_global : NULL, \
            (const int32_t *)selected->ptr, \
            desc->down_expert_bytes / nvfp4_block_bytes, \
            desc->down_row_bytes / nvfp4_block_bytes, \
            midq_blocks, out_dim, n_total_expert, n_expert, n_tokens)
    if (blackwell) {
        DS4_LAUNCH_NVFP4_DOWN(16u);
    } else {
        DS4_LAUNCH_NVFP4_DOWN(8u);
    }
#undef DS4_LAUNCH_NVFP4_DOWN
    return cuda_ok(cudaGetLastError(), "Laguna NVFP4 down launch");
}


extern "C" int ds4_gpu_laguna_routed_moe_tensor(
        ds4_gpu_tensor                *out,
        ds4_gpu_tensor                *mid,
        const void                    *model_map,
        uint64_t                       model_size,
        const ds4_gpu_laguna_moe_desc *routed,
        uint32_t                       expert_in_dim,
        uint32_t                       expert_mid_dim,
        uint32_t                       out_dim,
        const ds4_gpu_tensor          *selected,
        const ds4_gpu_tensor          *weights,
        uint32_t                       n_total_expert,
        uint32_t                       n_expert,
        const ds4_gpu_tensor          *x,
        uint32_t                       n_tokens) {
    if (!routed || routed->gate_type != 40u ||
        routed->up_type != 40u || routed->down_type != 40u) {
        return 0;
    }
    return cuda_laguna_nvfp4_routed_moe(
        out, mid, model_map, model_size, routed,
        expert_in_dim, expert_mid_dim, out_dim,
        selected, weights, n_total_expert, n_expert, x, n_tokens);
}
