/* Laguna and DFlash Metal host dispatch. Included once by ds4_metal.m. */

typedef struct {
    uint32_t in_dim;
    uint32_t q_dim;
    uint32_t kv_dim;
    uint32_t gate_dim;
} ds4_gpu_laguna_qkvg_args;

int ds4_gpu_laguna_qkvg_f16_tensor(
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        ds4_gpu_tensor       *gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint64_t              k_weight_offset,
        uint64_t              v_weight_offset,
        uint64_t              gate_weight_offset,
        uint32_t              in_dim,
        uint32_t              q_dim,
        uint32_t              kv_dim,
        uint32_t              gate_dim,
        const ds4_gpu_tensor *x) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!q || !k || !v || !gate || !model_map || !x || in_dim == 0u ||
        q_dim == 0u || kv_dim == 0u || gate_dim == 0u ||
        (in_dim % 32u) != 0u || (q_dim & 1u) != 0u ||
        (kv_dim & 1u) != 0u || (gate_dim & 1u) != 0u ||
        kv_dim > UINT32_MAX / 2u ||
        q_dim > UINT32_MAX - 2u * kv_dim ||
        q_dim + 2u * kv_dim > UINT32_MAX - gate_dim) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> kbuf = ds4_gpu_tensor_buffer(k);
        id<MTLBuffer> vbuf = ds4_gpu_tensor_buffer(v);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
        const uint64_t q_out_bytes = (uint64_t)q_dim * sizeof(float);
        const uint64_t kv_out_bytes = (uint64_t)kv_dim * sizeof(float);
        const uint64_t gate_out_bytes = (uint64_t)gate_dim * sizeof(float);
        if (!qbuf || !kbuf || !vbuf || !gatebuf || !xbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(q) < q_out_bytes ||
            ds4_gpu_tensor_bytes(k) < kv_out_bytes ||
            ds4_gpu_tensor_bytes(v) < kv_out_bytes ||
            ds4_gpu_tensor_bytes(gate) < gate_out_bytes) {
            return 0;
        }

        const uint64_t row_bytes = (uint64_t)in_dim * sizeof(uint16_t);
        if (q_dim > UINT64_MAX / row_bytes ||
            kv_dim > UINT64_MAX / row_bytes ||
            gate_dim > UINT64_MAX / row_bytes) {
            return 0;
        }
        const uint64_t q_weight_bytes = (uint64_t)q_dim * row_bytes;
        const uint64_t kv_weight_bytes = (uint64_t)kv_dim * row_bytes;
        const uint64_t gate_weight_bytes = (uint64_t)gate_dim * row_bytes;
#define DS4_LAGUNA_WEIGHT_RANGE_OK(offset_, bytes_) \
        ((offset_) <= model_size && (bytes_) <= model_size - (offset_))
        if (!DS4_LAGUNA_WEIGHT_RANGE_OK(q_weight_offset, q_weight_bytes) ||
            !DS4_LAGUNA_WEIGHT_RANGE_OK(k_weight_offset, kv_weight_bytes) ||
            !DS4_LAGUNA_WEIGHT_RANGE_OK(v_weight_offset, kv_weight_bytes) ||
            !DS4_LAGUNA_WEIGHT_RANGE_OK(gate_weight_offset, gate_weight_bytes)) {
            return 0;
        }
#undef DS4_LAGUNA_WEIGHT_RANGE_OK

        uint64_t q_inner = 0, k_inner = 0, v_inner = 0, gate_inner = 0;
        id<MTLBuffer> qwbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, q_weight_offset, q_weight_bytes, &q_inner);
        id<MTLBuffer> kwbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, k_weight_offset, kv_weight_bytes, &k_inner);
        id<MTLBuffer> vwbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, v_weight_offset, kv_weight_bytes, &v_inner);
        id<MTLBuffer> gwbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, gate_weight_offset, gate_weight_bytes,
            &gate_inner);
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_mul_mv_pipeline(
            "kernel_laguna_qkvg_f16_f32", 8);
        if (!qwbuf || !kwbuf || !vwbuf || !gwbuf || !pipeline) return 0;

        const ds4_gpu_laguna_qkvg_args args = {
            .in_dim = in_dim,
            .q_dim = q_dim,
            .kv_dim = kv_dim,
            .gate_dim = gate_dim,
        };
        const NSUInteger rows =
            (NSUInteger)q_dim + 2u * kv_dim + gate_dim;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qwbuf offset:(NSUInteger)q_inner atIndex:1];
        [enc setBuffer:kwbuf offset:(NSUInteger)k_inner atIndex:2];
        [enc setBuffer:vwbuf offset:(NSUInteger)v_inner atIndex:3];
        [enc setBuffer:gwbuf offset:(NSUInteger)gate_inner atIndex:4];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:5];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:6];
        [enc setBuffer:kbuf offset:ds4_gpu_tensor_offset(k) atIndex:7];
        [enc setBuffer:vbuf offset:ds4_gpu_tensor_offset(v) atIndex:8];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:9];
        [enc setThreadgroupMemoryLength:32u * 2u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((rows + 1u) / 2u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned,
                                            "Laguna Q/K/V/gate projection")) {
            return 0;
        }
    }
    return 1;
}
int ds4_gpu_laguna_attn_output_residual_f16_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !model_map || !x || !residual || in_dim == 0u ||
        out_dim == 0u || (in_dim % 32u) != 0u || (out_dim & 1u) != 0u) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> residualbuf = ds4_gpu_tensor_buffer(residual);
        const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!outbuf || !xbuf || !residualbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(residual) < out_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            return 0;
        }

        const uint64_t row_bytes = (uint64_t)in_dim * sizeof(uint16_t);
        if (out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
        if (weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            return 0;
        }

        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf = ds4_gpu_wrap_f32_decode_model_range(
            model_map, model_size, weight_offset, weight_bytes, 1,
            &weight_inner);
        ds4_gpu_mv_dispatch dispatch =
            ds4_gpu_make_plain_mv_dispatch(in_dim, 0);
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_mul_mv_pipeline(
            "kernel_laguna_attn_output_residual_f16_f32", dispatch.nsg);
        if (!weightbuf || !pipeline) return 0;

        ds4_gpu_f16_matvec_args args =
            ds4_gpu_make_f16_mv_args(in_dim, out_dim);
        args.nr0 = 2;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:residualbuf offset:ds4_gpu_tensor_offset(residual)
                atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * 2u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + 1u) / 2u,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(
                cb, owned, "Laguna attention output/residual")) {
            return 0;
        }
    }
    return 1;
}

int ds4_gpu_laguna_head_rms_norm_rope_tensor(
        ds4_gpu_tensor *x,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        n_rot,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow,
        float           eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || !model_map || n_tokens == 0 || n_head == 0 ||
        head_dim == 0 || head_dim > 128u || n_rot == 0 ||
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
    const uint64_t bytes = values * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    if (ds4_gpu_tensor_bytes(x) < bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        fprintf(stderr, "ds4: Metal Laguna head norm/RoPE received an invalid buffer or weight range\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf = ds4_gpu_wrap_model_range(model_map,
                                                            model_size,
                                                            weight_offset,
                                                            weight_bytes,
                                                            &weight_inner);
        id<MTLComputePipelineState> pipeline = ds4_gpu_hot_pipeline(
            g_laguna_head_norm_rope_pipeline,
            "kernel_laguna_head_rms_norm_rope_neox");
        if (!xbuf || !weightbuf || !pipeline) return 0;

        ds4_gpu_laguna_norm_rope_args args = {
            .n_tokens = n_tokens,
            .n_head = n_head,
            .head_dim = head_dim,
            .n_rot = n_rot,
            .pos0 = pos0,
            .n_ctx_orig = n_ctx_orig,
            .eps = eps,
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:2];
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_head, n_tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned, "Laguna head norm/RoPE")) return 0;
    }
    return 1;
}

int ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
        ds4_gpu_tensor *q,
        ds4_gpu_tensor *k,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        q_weight_offset,
        uint64_t        k_weight_offset,
        uint32_t        n_tokens,
        uint32_t        n_q_head,
        uint32_t        n_k_head,
        uint32_t        head_dim,
        uint32_t        n_rot,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow,
        float           eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!q || !k || !model_map || n_tokens == 0 || n_q_head == 0 ||
        n_k_head == 0 || n_q_head > UINT32_MAX - n_k_head ||
        head_dim == 0 || head_dim > 128u || n_rot == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0u ||
        pos0 > UINT32_MAX - n_tokens ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow) ||
        !isfinite(eps) || eps <= 0.0f) {
        return 0;
    }

    const uint64_t q_values = (uint64_t)n_tokens * n_q_head * head_dim;
    const uint64_t k_values = (uint64_t)n_tokens * n_k_head * head_dim;
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    if (ds4_gpu_tensor_bytes(q) < q_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(k) < k_values * sizeof(float) ||
        q_weight_offset > model_size ||
        weight_bytes > model_size - q_weight_offset ||
        k_weight_offset > model_size ||
        weight_bytes > model_size - k_weight_offset) {
        fprintf(stderr, "ds4: Metal Laguna Q/K norm/RoPE received an invalid buffer or weight range\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> kbuf = ds4_gpu_tensor_buffer(k);
        uint64_t q_weight_inner = 0;
        uint64_t k_weight_inner = 0;
        id<MTLBuffer> q_weightbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, q_weight_offset, weight_bytes,
            &q_weight_inner);
        id<MTLBuffer> k_weightbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, k_weight_offset, weight_bytes,
            &k_weight_inner);
        id<MTLComputePipelineState> pipeline = ds4_gpu_hot_pipeline(
            g_laguna_qk_head_norm_rope_pipeline,
            "kernel_laguna_qk_head_rms_norm_rope_neox");
        if (!qbuf || !kbuf || !q_weightbuf || !k_weightbuf || !pipeline) {
            return 0;
        }

        ds4_gpu_laguna_norm_rope_args args = {
            .n_tokens = n_tokens,
            .n_head = n_q_head + n_k_head,
            .head_dim = head_dim,
            .n_rot = n_rot,
            .pos0 = pos0,
            .n_ctx_orig = n_ctx_orig,
            .eps = eps,
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:kbuf offset:ds4_gpu_tensor_offset(k) atIndex:2];
        [enc setBuffer:q_weightbuf offset:(NSUInteger)q_weight_inner atIndex:3];
        [enc setBuffer:k_weightbuf offset:(NSUInteger)k_weight_inner atIndex:4];
        [enc setBytes:&n_q_head length:sizeof(n_q_head) atIndex:5];
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_q_head + n_k_head, n_tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(
                cb, owned, "Laguna Q/K head norm/RoPE")) {
            return 0;
        }
    }
    return 1;
}

static int ds4_gpu_encode_laguna_flash_attention_decode(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        headsbuf,
        NSUInteger           heads_offset,
        id<MTLBuffer>        qbuf,
        NSUInteger           q_offset,
        id<MTLBuffer>        gatebuf,
        NSUInteger           gate_offset,
        id<MTLBuffer>        keybuf,
        NSUInteger           key_offset,
        id<MTLBuffer>        valuebuf,
        NSUInteger           value_offset,
        uint32_t              cache_cap,
        uint32_t              key_count,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale) {
    if (!cb || !headsbuf || !qbuf || !gatebuf || !keybuf || !valuebuf ||
        key_count == 0u || key_count > cache_cap || n_head == 0u ||
        n_head_kv == 0u || n_head % n_head_kv != 0u || head_dim != 128u) {
        return 0;
    }

    const uint32_t ncpsg = 32u;
    const uint32_t nwg = 32u;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(key_count, nwg, ncpsg);
    const bool has_pad = (key_count % ncpsg) != 0u;
    const bool use_gqa3 =
        cache_cap > 512u && (n_head % 3u) == 0u &&
        ((n_head / n_head_kv) % 3u) == 0u;
    const NSUInteger head_bytes = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger cache_row_bytes =
        (NSUInteger)n_head_kv * head_bytes;
    const NSUInteger mask_bytes = (NSUInteger)key_count * sizeof(uint16_t);
    const NSUInteger pad_bytes = has_pad ?
        2u * (NSUInteger)ncpsg * cache_row_bytes * n_head_kv +
        (NSUInteger)ncpsg * sizeof(uint16_t) : 1u;
    const NSUInteger nrows = (NSUInteger)n_head;
    const NSUInteger tmp_bytes =
        nrows * head_dim * nwg * sizeof(float) +
        nrows * 2u * nwg * sizeof(float);

    if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                       &g_flash_attn_tmp_bytes,
                                       tmp_bytes,
                                       "ds4_laguna_flash_attn_tmp")) {
        return 0;
    }
    if (!use_gqa3 &&
        (!ds4_gpu_ensure_zero_attention_mask(mask_bytes) ||
         !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_laguna_flash_attn_pad"))) {
        return 0;
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_pad && !use_gqa3) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline = use_gqa3 ?
        ds4_gpu_get_pipeline("kernel_laguna_attention_decode_gqa3_split_f16") :
        ds4_gpu_get_flash_attn_vec_pipeline(
            "kernel_flash_attn_ext_vec_qf32_f16_dk128_dv128",
            true, false, false, false, has_pad, false,
            (int32_t)head_dim, (int32_t)head_dim,
            (int32_t)nsg, (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_laguna_flash_attn_reduce_gate_pipeline(
            (int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    id<MTLComputeCommandEncoder> enc = nil;
    if (has_pad && !use_gqa3) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)key_count,
            .ne_12_2 = (int32_t)n_head_kv,
            .ne_12_3 = 1,
            .nb11 = cache_row_bytes,
            .nb12 = head_bytes,
            .nb13 = (uint64_t)cache_cap * cache_row_bytes,
            .nb21 = cache_row_bytes,
            .nb22 = head_bytes,
            .nb23 = (uint64_t)cache_cap * cache_row_bytes,
            .ne31 = 1,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = mask_bytes,
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };
        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:keybuf offset:key_offset atIndex:1];
        [enc setBuffer:valuebuf offset:value_offset atIndex:2];
        [enc setBuffer:g_flash_attn_zero_mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, n_head_kv, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    const NSUInteger q_row_bytes =
        (NSUInteger)n_head * head_dim * sizeof(float);
    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = 1,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = q_row_bytes,
        .nb02 = (uint64_t)head_dim * sizeof(float),
        .nb03 = q_row_bytes,
        .ne11 = (int32_t)key_count,
        .ne_12_2 = (int32_t)n_head_kv,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = cache_row_bytes,
        .nb12 = head_bytes,
        .nb13 = (uint64_t)cache_cap * cache_row_bytes,
        .ns20 = (int32_t)head_dim,
        .nb21 = cache_row_bytes,
        .nb22 = head_bytes,
        .nb23 = (uint64_t)cache_cap * cache_row_bytes,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = mask_bytes,
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = 1,
        .ne3 = 1,
        .scale = scale,
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };
    const NSUInteger shared_elems =
        (2u * ds4_gpu_align_up_ns(head_dim, 128u) + 4u * ncpsg +
         2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes =
        ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBuffer:qbuf offset:q_offset atIndex:1];
    [enc setBuffer:keybuf offset:key_offset atIndex:2];
    [enc setBuffer:valuebuf offset:value_offset atIndex:3];
    if (use_gqa3) {
        const ds4_gpu_laguna_gqa3_decode_args grouped_args = {
            .n_head = n_head,
            .n_head_kv = n_head_kv,
            .head_dim = head_dim,
            .key_count0 = key_count,
            .n_tokens = 1u,
            .nsg = nsg,
            .nwg = nwg,
            .scale = scale,
        };
        const NSUInteger grouped_shared_bytes =
            3u * nsg * (2u + head_dim) * sizeof(float);
        [enc setBytes:&grouped_args length:sizeof(grouped_args) atIndex:0];
        [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:grouped_shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_head / 3u, 1, nwg)
             threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    } else {
        [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
        [enc setBuffer:g_flash_attn_zero_mask_buffer offset:0 atIndex:4];
        [enc setBuffer:gatebuf offset:gate_offset atIndex:5];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
        [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
        [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, n_head, nwg)
             threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    }
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:heads_offset atIndex:2];
    [enc setBuffer:gatebuf offset:gate_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_laguna_store_attention_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *gate,
        uint32_t              pos,
        uint32_t              cache_cap,
        uint32_t              key_start,
        uint32_t              key_count,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !key_cache || !value_cache || !q || !k || !v || !gate ||
        cache_cap == 0 || key_count == 0 || key_count > cache_cap ||
        n_head == 0 || n_head_kv == 0 || n_head % n_head_kv != 0 ||
        head_dim != 128u || !isfinite(scale) || scale <= 0.0f) {
        return 0;
    }
    const uint64_t q_values = (uint64_t)n_head * head_dim;
    const uint64_t kv_values = (uint64_t)n_head_kv * head_dim;
    const uint64_t cache_values = (uint64_t)cache_cap * kv_values;
    if (ds4_gpu_tensor_bytes(q) < q_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(k) < kv_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(v) < kv_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(gate) < (uint64_t)n_head * sizeof(float) ||
        ds4_gpu_tensor_bytes(heads) < q_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(key_cache) < cache_values * sizeof(uint16_t) ||
        ds4_gpu_tensor_bytes(value_cache) < cache_values * sizeof(uint16_t)) {
        fprintf(stderr, "ds4: Metal Laguna attention received undersized buffers\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valuebuf = ds4_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> kbuf = ds4_gpu_tensor_buffer(k);
        id<MTLBuffer> vbuf = ds4_gpu_tensor_buffer(v);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLComputePipelineState> store_pipeline = ds4_gpu_hot_pipeline(
            g_laguna_store_kv_pipeline, "kernel_laguna_store_kv_f16");
        id<MTLComputePipelineState> attention_pipeline = ds4_gpu_hot_pipeline(
            g_laguna_attention_pipeline,
            "kernel_laguna_attention_decode_gqa_f16");
        if (!headsbuf || !keybuf || !valuebuf || !qbuf || !kbuf || !vbuf ||
            !gatebuf || !store_pipeline || !attention_pipeline) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        ds4_gpu_laguna_kv_store_args store_args = {
            .cache_cap = cache_cap,
            .cache_row = pos % cache_cap,
            .n_head_kv = n_head_kv,
            .head_dim = head_dim,
        };
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:store_pipeline];
        [enc setBytes:&store_args length:sizeof(store_args) atIndex:0];
        [enc setBuffer:kbuf offset:ds4_gpu_tensor_offset(k) atIndex:1];
        [enc setBuffer:vbuf offset:ds4_gpu_tensor_offset(v) atIndex:2];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:3];
        [enc setBuffer:valuebuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_values, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        /* The global-attention cache is contiguous until it reaches capacity.
         * A full sliding ring also contains exactly the active key set; its
         * physical rotation does not affect an unmasked softmax reduction.
         * An unaligned full global cache uses the old kernel so tail padding
         * never reads beyond the interleaved KV allocation. */
        const bool full_sliding_ring =
            cache_cap == 512u && key_count == cache_cap;
        const bool use_flash_decode =
            full_sliding_ring ||
            (cache_cap > 512u && key_start == 0u && key_count >= 1024u &&
             (key_count < cache_cap || key_count % 32u == 0u));
        if (use_flash_decode) {
            if (!ds4_gpu_encode_laguna_flash_attention_decode(
                    cb,
                    headsbuf,
                    ds4_gpu_tensor_offset(heads),
                    qbuf,
                    ds4_gpu_tensor_offset(q),
                    gatebuf,
                    ds4_gpu_tensor_offset(gate),
                    keybuf,
                    ds4_gpu_tensor_offset(key_cache),
                    valuebuf,
                    ds4_gpu_tensor_offset(value_cache),
                    cache_cap,
                    key_count,
                    n_head,
                    n_head_kv,
                    head_dim,
                    scale)) {
                return 0;
            }
        } else {
            ds4_gpu_laguna_attention_args attention_args = {
                .n_head = n_head,
                .n_head_kv = n_head_kv,
                .head_dim = head_dim,
                .cache_cap = cache_cap,
                .key_start = key_start,
                .key_count = key_count,
                .scale = scale,
            };
            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:attention_pipeline];
            [enc setBytes:&attention_args length:sizeof(attention_args) atIndex:0];
            [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
            [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:2];
            [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:3];
            [enc setBuffer:valuebuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:4];
            [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:5];
            [enc setThreadgroupMemoryLength:(8u + 8u + 8u * 128u) * sizeof(float)
                                    atIndex:0];
            const NSUInteger attention_threads = key_count > 256u ? 256u : 32u;
            [enc dispatchThreadgroups:MTLSizeMake(n_head, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(attention_threads, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        }
        if (!ds4_gpu_finish_command_buffer(cb, owned, "Laguna KV store + attention")) return 0;
    }
    return 1;
}

static int ds4_gpu_encode_laguna_flash_attention_decode_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        headsbuf,
        NSUInteger           heads_offset,
        id<MTLBuffer>        qbuf,
        NSUInteger           q_offset,
        id<MTLBuffer>        gatebuf,
        NSUInteger           gate_offset,
        id<MTLBuffer>        keybuf,
        NSUInteger           key_offset,
        id<MTLBuffer>        valuebuf,
        NSUInteger           value_offset,
        uint32_t              cache_cap,
        uint32_t              key_count0,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale) {
    if (!cb || !headsbuf || !qbuf || !gatebuf || !keybuf || !valuebuf ||
        cache_cap <= 512u || key_count0 == 0u || n_tokens < 2u ||
        n_tokens > 16u || key_count0 > cache_cap - n_tokens + 1u ||
        n_head == 0u || n_head_kv == 0u ||
        (n_head % 3u) != 0u || n_head % n_head_kv != 0u ||
        ((n_head / n_head_kv) % 3u) != 0u || head_dim != 128u) {
        return 0;
    }

    const uint32_t nwg = 32u;
    const uint32_t ncpsg = 32u;
    const uint32_t key_count_max = key_count0 + n_tokens - 1u;
    const uint32_t nsg =
        ds4_gpu_flash_attn_vec_nsg(key_count_max, nwg, ncpsg);
    if (ds4_gpu_flash_attn_vec_nsg(key_count0, nwg, ncpsg) != nsg) {
        return 0;
    }
    const NSUInteger nrows = (NSUInteger)n_tokens * n_head;
    const NSUInteger tmp_bytes =
        nrows * head_dim * nwg * sizeof(float) +
        nrows * 2u * nwg * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                       &g_flash_attn_tmp_bytes,
                                       tmp_bytes,
                                       "ds4_laguna_flash_attn_tmp")) {
        return 0;
    }

    id<MTLComputePipelineState> attention_pipeline =
        ds4_gpu_get_pipeline("kernel_laguna_attention_decode_gqa3_split_f16");
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_laguna_flash_attn_reduce_gate_pipeline(
            (int32_t)head_dim, (int32_t)nwg);
    if (!attention_pipeline || !reduce_pipeline) return 0;

    const ds4_gpu_laguna_gqa3_decode_args args = {
        .n_head = n_head,
        .n_head_kv = n_head_kv,
        .head_dim = head_dim,
        .key_count0 = key_count0,
        .n_tokens = n_tokens,
        .nsg = nsg,
        .nwg = nwg,
        .scale = scale,
    };
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attention_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:q_offset atIndex:1];
    [enc setBuffer:keybuf offset:key_offset atIndex:2];
    [enc setBuffer:valuebuf offset:value_offset atIndex:3];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:
        3u * nsg * (2u + head_dim) * sizeof(float)
                            atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_head / 3u, n_tokens, nwg)
         threadsPerThreadgroup:MTLSizeMake(32u, nsg, 1u)];
    ds4_gpu_end_compute_encoder(cb, enc);

    const ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:heads_offset atIndex:2];
    [enc setBuffer:gatebuf offset:gate_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1u, 1u)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1u, 1u)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_laguna_attention_prefill_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        ds4_gpu_tensor       *staged_key,
        ds4_gpu_tensor       *staged_value,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *gate,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale,
        int                   split_decode_rows) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !key_cache || !value_cache || !staged_key ||
        !staged_value || !q || !k || !v || !gate || n_tokens == 0 ||
        pos0 > UINT32_MAX - n_tokens || cache_cap == 0 || n_head == 0 ||
        n_head_kv == 0 || n_head % n_head_kv != 0 || head_dim != 128u ||
        !isfinite(scale) || scale <= 0.0f) {
        return 0;
    }

    const uint64_t q_values = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_values =
        (uint64_t)n_tokens * n_head_kv * head_dim;
    const uint64_t cache_values =
        (uint64_t)cache_cap * n_head_kv * head_dim;
    if (ds4_gpu_tensor_bytes(q) < q_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(k) < kv_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(v) < kv_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(gate) <
            (uint64_t)n_tokens * n_head * sizeof(float) ||
        ds4_gpu_tensor_bytes(heads) < q_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(staged_key) < kv_values * sizeof(uint16_t) ||
        ds4_gpu_tensor_bytes(staged_value) < kv_values * sizeof(uint16_t) ||
        ds4_gpu_tensor_bytes(key_cache) < cache_values * sizeof(uint16_t) ||
        ds4_gpu_tensor_bytes(value_cache) < cache_values * sizeof(uint16_t)) {
        fprintf(stderr, "ds4: Metal Laguna prefill attention received undersized buffers\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valuebuf = ds4_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> stagedkeybuf = ds4_gpu_tensor_buffer(staged_key);
        id<MTLBuffer> stagedvaluebuf = ds4_gpu_tensor_buffer(staged_value);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> kbuf = ds4_gpu_tensor_buffer(k);
        id<MTLBuffer> vbuf = ds4_gpu_tensor_buffer(v);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLComputePipelineState> store_pipeline = ds4_gpu_hot_pipeline(
            g_laguna_store_kv_pipeline, "kernel_laguna_store_kv_f16");
        id<MTLComputePipelineState> stage_pipeline = ds4_gpu_hot_pipeline(
            g_laguna_stage_kv_pipeline, "kernel_laguna_stage_kv_f16");
        const uint32_t heads_per_kv = n_head / n_head_kv;
        const bool use_gqa6 =
            (n_head % 6u) == 0u && (heads_per_kv % 6u) == 0u;
        const bool use_gqa3 =
            !use_gqa6 && (n_head % 3u) == 0u && (heads_per_kv % 3u) == 0u;
        id<MTLComputePipelineState> attention_pipeline = use_gqa6 ?
            ds4_gpu_get_pipeline("kernel_laguna_attention_prefill_gqa6_f16") :
            use_gqa3 ?
                ds4_gpu_get_pipeline("kernel_laguna_attention_prefill_gqa3_f16") :
                ds4_gpu_hot_pipeline(g_laguna_prefill_attention_pipeline,
                                     "kernel_laguna_attention_prefill_gqa_f16");
        id<MTLComputePipelineState> decode_attention_pipeline =
            ds4_gpu_hot_pipeline(g_laguna_attention_pipeline,
                                 "kernel_laguna_attention_decode_gqa_f16");
        id<MTLComputePipelineState> commit_pipeline = ds4_gpu_hot_pipeline(
            g_laguna_commit_kv_pipeline, "kernel_laguna_commit_kv_f16");
        if (!headsbuf || !keybuf || !valuebuf || !stagedkeybuf ||
            !stagedvaluebuf || !qbuf || !kbuf || !vbuf || !gatebuf ||
            !store_pipeline || !stage_pipeline || !attention_pipeline ||
            !decode_attention_pipeline || !commit_pipeline) {
            return 0;
        }

        /*
         * A speculative verifier has only a handful of query rows.  At long
         * context, assigning one SIMD group to each query makes that group
         * walk thousands of keys serially.  Encode the established split-key
         * decode attention once per row instead.  The rows still share one
         * command buffer, and storing them in order preserves causal and SWA
         * ring semantics exactly.
         */
        const bool use_split_decode_rows =
            split_decode_rows != 0 && n_tokens <= 16u;
        if (use_split_decode_rows) {
            const NSUInteger q_row_bytes =
                (NSUInteger)n_head * head_dim * sizeof(float);
            const NSUInteger kv_row_bytes =
                (NSUInteger)n_head_kv * head_dim * sizeof(float);
            const NSUInteger gate_row_bytes =
                (NSUInteger)n_head * sizeof(float);
            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;

            const uint32_t first_key_count =
                MIN(pos0 + 1u, cache_cap);
            const bool use_batched_global_rows =
                n_tokens >= 2u && n_tokens <= 16u &&
                cache_cap > 512u &&
                pos0 + n_tokens <= cache_cap &&
                (n_head % 3u) == 0u &&
                (heads_per_kv % 3u) == 0u &&
                ds4_gpu_flash_attn_vec_nsg(first_key_count, 32u, 32u) ==
                ds4_gpu_flash_attn_vec_nsg(
                    first_key_count + n_tokens - 1u, 32u, 32u);
            if (use_batched_global_rows) {
                for (uint32_t row = 0; row < n_tokens; row++) {
                    const uint32_t pos = pos0 + row;
                    const ds4_gpu_laguna_kv_store_args store_args = {
                        .cache_cap = cache_cap,
                        .cache_row = pos,
                        .n_head_kv = n_head_kv,
                        .head_dim = head_dim,
                    };
                    id<MTLComputeCommandEncoder> enc =
                        ds4_gpu_compute_encoder(cb);
                    [enc setComputePipelineState:store_pipeline];
                    [enc setBytes:&store_args
                           length:sizeof(store_args)
                          atIndex:0];
                    [enc setBuffer:kbuf
                            offset:ds4_gpu_tensor_offset(k) +
                                   (NSUInteger)row * kv_row_bytes
                           atIndex:1];
                    [enc setBuffer:vbuf
                            offset:ds4_gpu_tensor_offset(v) +
                                   (NSUInteger)row * kv_row_bytes
                           atIndex:2];
                    [enc setBuffer:keybuf
                            offset:ds4_gpu_tensor_offset(key_cache)
                           atIndex:3];
                    [enc setBuffer:valuebuf
                            offset:ds4_gpu_tensor_offset(value_cache)
                           atIndex:4];
                    [enc dispatchThreads:
                            MTLSizeMake((NSUInteger)n_head_kv * head_dim,
                                        1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ds4_gpu_end_compute_encoder(cb, enc);
                }
                if (!ds4_gpu_encode_laguna_flash_attention_decode_rows(
                        cb,
                        headsbuf,
                        ds4_gpu_tensor_offset(heads),
                        qbuf,
                        ds4_gpu_tensor_offset(q),
                        gatebuf,
                        ds4_gpu_tensor_offset(gate),
                        keybuf,
                        ds4_gpu_tensor_offset(key_cache),
                        valuebuf,
                        ds4_gpu_tensor_offset(value_cache),
                        cache_cap,
                        first_key_count,
                        n_tokens,
                        n_head,
                        n_head_kv,
                        head_dim,
                        scale)) {
                    return 0;
                }
                if (!ds4_gpu_finish_command_buffer(
                        cb, owned, "Laguna batched-row split attention")) {
                    return 0;
                }
                return 1;
            }

            const bool use_batched_swa_rows =
                n_tokens >= 2u && n_tokens <= 16u &&
                cache_cap == 512u &&
                pos0 + n_tokens <= cache_cap;
            if (use_batched_swa_rows) {
                id<MTLComputePipelineState> rows_store_pipeline =
                    ds4_gpu_get_pipeline(
                        "kernel_laguna_store_kv_rows_f16");
                id<MTLComputePipelineState> rows_attention_pipeline =
                    ds4_gpu_get_pipeline(
                        "kernel_laguna_attention_decode_rows_gqa_f16");
                if (!rows_store_pipeline || !rows_attention_pipeline) {
                    return 0;
                }
                const ds4_gpu_laguna_prefill_attention_args rows_args = {
                    .n_tokens = n_tokens,
                    .pos0 = pos0,
                    .cache_cap = cache_cap,
                    .n_head = n_head,
                    .n_head_kv = n_head_kv,
                    .head_dim = head_dim,
                    .scale = scale,
                    .pad0 = 0u,
                };
                id<MTLComputeCommandEncoder> enc =
                    ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:rows_store_pipeline];
                [enc setBytes:&rows_args
                       length:sizeof(rows_args)
                      atIndex:0];
                [enc setBuffer:kbuf
                        offset:ds4_gpu_tensor_offset(k)
                       atIndex:1];
                [enc setBuffer:vbuf
                        offset:ds4_gpu_tensor_offset(v)
                       atIndex:2];
                [enc setBuffer:keybuf
                        offset:ds4_gpu_tensor_offset(key_cache)
                       atIndex:3];
                [enc setBuffer:valuebuf
                        offset:ds4_gpu_tensor_offset(value_cache)
                       atIndex:4];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_values, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);

                enc = ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:rows_attention_pipeline];
                [enc setBytes:&rows_args
                       length:sizeof(rows_args)
                      atIndex:0];
                [enc setBuffer:qbuf
                        offset:ds4_gpu_tensor_offset(q)
                       atIndex:1];
                [enc setBuffer:gatebuf
                        offset:ds4_gpu_tensor_offset(gate)
                       atIndex:2];
                [enc setBuffer:keybuf
                        offset:ds4_gpu_tensor_offset(key_cache)
                       atIndex:3];
                [enc setBuffer:valuebuf
                        offset:ds4_gpu_tensor_offset(value_cache)
                       atIndex:4];
                [enc setBuffer:headsbuf
                        offset:ds4_gpu_tensor_offset(heads)
                       atIndex:5];
                [enc setThreadgroupMemoryLength:
                        (8u + 8u + 8u * 128u) * sizeof(float)
                                        atIndex:0];
                const uint32_t last_key_count = pos0 + n_tokens;
                const NSUInteger threads =
                    last_key_count > 256u ? 256u : 32u;
                [enc dispatchThreadgroups:
                        MTLSizeMake(n_head, n_tokens, 1)
                     threadsPerThreadgroup:
                        MTLSizeMake(threads, 1, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);

                if (!ds4_gpu_finish_command_buffer(
                        cb, owned,
                        "Laguna batched sliding-window attention")) {
                    return 0;
                }
                return 1;
            }

            for (uint32_t row = 0; row < n_tokens; row++) {
                const uint32_t pos = pos0 + row;
                const uint32_t key_count = MIN(pos + 1u, cache_cap);
                const uint32_t key_start = pos + 1u - key_count;
                const ds4_gpu_laguna_kv_store_args store_args = {
                    .cache_cap = cache_cap,
                    .cache_row = pos % cache_cap,
                    .n_head_kv = n_head_kv,
                    .head_dim = head_dim,
                };
                id<MTLComputeCommandEncoder> enc =
                    ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:store_pipeline];
                [enc setBytes:&store_args
                       length:sizeof(store_args)
                      atIndex:0];
                [enc setBuffer:kbuf
                        offset:ds4_gpu_tensor_offset(k) +
                               (NSUInteger)row * kv_row_bytes
                       atIndex:1];
                [enc setBuffer:vbuf
                        offset:ds4_gpu_tensor_offset(v) +
                               (NSUInteger)row * kv_row_bytes
                       atIndex:2];
                [enc setBuffer:keybuf
                        offset:ds4_gpu_tensor_offset(key_cache)
                       atIndex:3];
                [enc setBuffer:valuebuf
                        offset:ds4_gpu_tensor_offset(value_cache)
                       atIndex:4];
                [enc dispatchThreads:
                        MTLSizeMake((NSUInteger)n_head_kv * head_dim, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);

                const bool full_sliding_ring =
                    cache_cap == 512u && key_count == cache_cap;
                const bool use_flash_decode =
                    full_sliding_ring ||
                    (cache_cap > 512u && key_start == 0u &&
                     key_count >= 1024u &&
                     (key_count < cache_cap || key_count % 32u == 0u));
                if (use_flash_decode) {
                    if (!ds4_gpu_encode_laguna_flash_attention_decode(
                            cb,
                            headsbuf,
                            ds4_gpu_tensor_offset(heads) +
                                (NSUInteger)row * q_row_bytes,
                            qbuf,
                            ds4_gpu_tensor_offset(q) +
                                (NSUInteger)row * q_row_bytes,
                            gatebuf,
                            ds4_gpu_tensor_offset(gate) +
                                (NSUInteger)row * gate_row_bytes,
                            keybuf,
                            ds4_gpu_tensor_offset(key_cache),
                            valuebuf,
                            ds4_gpu_tensor_offset(value_cache),
                            cache_cap,
                            key_count,
                            n_head,
                            n_head_kv,
                            head_dim,
                            scale)) {
                        return 0;
                    }
                } else {
                    const ds4_gpu_laguna_attention_args attention_args = {
                        .n_head = n_head,
                        .n_head_kv = n_head_kv,
                        .head_dim = head_dim,
                        .cache_cap = cache_cap,
                        .key_start = key_start,
                        .key_count = key_count,
                        .scale = scale,
                    };
                    enc = ds4_gpu_compute_encoder(cb);
                    [enc setComputePipelineState:decode_attention_pipeline];
                    [enc setBytes:&attention_args
                           length:sizeof(attention_args)
                          atIndex:0];
                    [enc setBuffer:qbuf
                            offset:ds4_gpu_tensor_offset(q) +
                                   (NSUInteger)row * q_row_bytes
                           atIndex:1];
                    [enc setBuffer:gatebuf
                            offset:ds4_gpu_tensor_offset(gate) +
                                   (NSUInteger)row * gate_row_bytes
                           atIndex:2];
                    [enc setBuffer:keybuf
                            offset:ds4_gpu_tensor_offset(key_cache)
                           atIndex:3];
                    [enc setBuffer:valuebuf
                            offset:ds4_gpu_tensor_offset(value_cache)
                           atIndex:4];
                    [enc setBuffer:headsbuf
                            offset:ds4_gpu_tensor_offset(heads) +
                                   (NSUInteger)row * q_row_bytes
                           atIndex:5];
                    [enc setThreadgroupMemoryLength:
                            (8u + 8u + 8u * 128u) * sizeof(float)
                                            atIndex:0];
                    const NSUInteger threads =
                        key_count > 256u ? 256u : 32u;
                    [enc dispatchThreadgroups:MTLSizeMake(n_head, 1, 1)
                         threadsPerThreadgroup:
                            MTLSizeMake(threads, 1, 1)];
                    ds4_gpu_end_compute_encoder(cb, enc);
                }
            }
            if (!ds4_gpu_finish_command_buffer(
                    cb, owned, "Laguna split-row attention")) {
                return 0;
            }
            return 1;
        }

        const ds4_gpu_laguna_prefill_attention_args args = {
            .n_tokens = n_tokens,
            .pos0 = pos0,
            .cache_cap = cache_cap,
            .n_head = n_head,
            .n_head_kv = n_head_kv,
            .head_dim = head_dim,
            .scale = scale,
            .pad0 = 0,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:stage_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kbuf offset:ds4_gpu_tensor_offset(k) atIndex:1];
        [enc setBuffer:vbuf offset:ds4_gpu_tensor_offset(v) atIndex:2];
        [enc setBuffer:stagedkeybuf offset:ds4_gpu_tensor_offset(staged_key)
                atIndex:3];
        [enc setBuffer:stagedvaluebuf offset:ds4_gpu_tensor_offset(staged_value)
                atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_values, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:attention_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:2];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:3];
        [enc setBuffer:valuebuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:4];
        [enc setBuffer:stagedkeybuf offset:ds4_gpu_tensor_offset(staged_key)
                atIndex:5];
        [enc setBuffer:stagedvaluebuf offset:ds4_gpu_tensor_offset(staged_value)
                atIndex:6];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:7];
        const uint32_t attention_groups = use_gqa6 ?
            n_head / 6u : use_gqa3 ? n_head / 3u : n_head;
        [enc dispatchThreadgroups:MTLSizeMake(attention_groups,
                                              n_tokens,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:commit_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:stagedkeybuf offset:ds4_gpu_tensor_offset(staged_key)
                atIndex:1];
        [enc setBuffer:stagedvaluebuf offset:ds4_gpu_tensor_offset(staged_value)
                atIndex:2];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:3];
        [enc setBuffer:valuebuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_values, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned,
                                            "Laguna prefill attention")) {
            return 0;
        }
    }
    return 1;
}

typedef struct {
    uint32_t n_rows;
    uint32_t n_embd;
    uint32_t n_aux;
    uint32_t aux_index;
    uint32_t src_row0;
    uint32_t dst_row0;
} ds4_gpu_dflash_capture_args;

int ds4_gpu_dflash_capture_rows_tensor(
        ds4_gpu_tensor       *features,
        const ds4_gpu_tensor *src,
        uint32_t              src_row0,
        uint32_t              dst_row0,
        uint32_t              n_rows,
        uint32_t              n_embd,
        uint32_t              n_aux,
        uint32_t              aux_index) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!features || !src || n_rows == 0u || n_embd == 0u ||
        n_aux == 0u || aux_index >= n_aux) {
        return 0;
    }
    const uint64_t src_rows = (uint64_t)src_row0 + n_rows;
    const uint64_t dst_rows = (uint64_t)dst_row0 + n_rows;
    const uint64_t src_values = src_rows * n_embd;
    const uint64_t dst_values = dst_rows * n_aux * n_embd;
    if (src_values > UINT64_MAX / sizeof(float) ||
        dst_values > UINT64_MAX / sizeof(float) ||
        ds4_gpu_tensor_bytes(src) < src_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(features) < dst_values * sizeof(float)) {
        fprintf(stderr, "ds4: Metal DFlash capture received undersized buffers\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> srcbuf = ds4_gpu_tensor_buffer(src);
        id<MTLBuffer> dstbuf = ds4_gpu_tensor_buffer(features);
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dflash_capture_rows");
        if (!srcbuf || !dstbuf || !pipeline) return 0;

        const ds4_gpu_dflash_capture_args args = {
            .n_rows = n_rows,
            .n_embd = n_embd,
            .n_aux = n_aux,
            .aux_index = aux_index,
            .src_row0 = src_row0,
            .dst_row0 = dst_row0,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:srcbuf offset:ds4_gpu_tensor_offset(src) atIndex:1];
        [enc setBuffer:dstbuf offset:ds4_gpu_tensor_offset(features) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n_rows * n_embd, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned,
                                            "DFlash feature capture")) {
            return 0;
        }
    }
    return 1;
}

typedef struct {
    uint32_t n_rows;
    uint32_t n_embd;
    uint32_t n_aux;
    float eps;
} ds4_gpu_dflash_aux_norm_args;

int ds4_gpu_dflash_aux_norm_tensor(
        ds4_gpu_tensor *features,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        n_rows,
        uint32_t        n_embd,
        uint32_t        n_aux,
        float           eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!features || !model_map || n_rows == 0u || n_embd == 0u ||
        n_aux == 0u || !isfinite(eps) || eps < 0.0f) {
        return 0;
    }
    const uint64_t feature_values = (uint64_t)n_rows * n_aux * n_embd;
    const uint64_t weight_values = (uint64_t)n_aux * n_embd;
    if (feature_values > UINT64_MAX / sizeof(float) ||
        weight_values > UINT64_MAX / sizeof(float) ||
        ds4_gpu_tensor_bytes(features) <
            feature_values * sizeof(float) ||
        weight_offset > model_size ||
        weight_values * sizeof(float) > model_size - weight_offset) {
        fprintf(stderr, "ds4: Metal DFlash auxiliary norm received invalid ranges\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> featurebuf = ds4_gpu_tensor_buffer(features);
        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf = ds4_gpu_wrap_model_range(
            model_map,
            model_size,
            weight_offset,
            weight_values * sizeof(float),
            &weight_inner);
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dflash_aux_rms_norm");
        if (!featurebuf || !weightbuf || !pipeline) return 0;

        const ds4_gpu_dflash_aux_norm_args args = {
            .n_rows = n_rows,
            .n_embd = n_embd,
            .n_aux = n_aux,
            .eps = eps,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:featurebuf
                offset:ds4_gpu_tensor_offset(features)
               atIndex:1];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:2];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_rows, n_aux, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned,
                                            "DFlash auxiliary norm")) {
            return 0;
        }
    }
    return 1;
}

typedef struct {
    uint32_t n_rows;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t width;
} ds4_gpu_dflash_commit_kv_args;

int ds4_gpu_dflash_commit_kv_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        uint32_t              pos0,
        uint32_t              n_rows,
        uint32_t              cache_cap,
        uint32_t              n_head_kv,
        uint32_t              head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!key_cache || !value_cache || !k || !v || n_rows == 0u ||
        cache_cap == 0u || n_head_kv == 0u || head_dim == 0u) {
        return 0;
    }
    const uint64_t width = (uint64_t)n_head_kv * head_dim;
    const uint64_t src_values = (uint64_t)n_rows * width;
    const uint64_t cache_values = (uint64_t)cache_cap * width;
    if (width > UINT32_MAX ||
        src_values > UINT64_MAX / sizeof(float) ||
        cache_values > UINT64_MAX / sizeof(uint16_t) ||
        ds4_gpu_tensor_bytes(k) < src_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(v) < src_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(key_cache) <
            cache_values * sizeof(uint16_t) ||
        ds4_gpu_tensor_bytes(value_cache) <
            cache_values * sizeof(uint16_t)) {
        fprintf(stderr, "ds4: Metal DFlash KV commit received undersized buffers\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valuebuf = ds4_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> kbuf = ds4_gpu_tensor_buffer(k);
        id<MTLBuffer> vbuf = ds4_gpu_tensor_buffer(v);
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dflash_commit_kv_f16");
        if (!keybuf || !valuebuf || !kbuf || !vbuf || !pipeline) return 0;

        const ds4_gpu_dflash_commit_kv_args args = {
            .n_rows = n_rows,
            .pos0 = pos0,
            .cache_cap = cache_cap,
            .width = (uint32_t)width,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kbuf offset:ds4_gpu_tensor_offset(k) atIndex:1];
        [enc setBuffer:vbuf offset:ds4_gpu_tensor_offset(v) atIndex:2];
        [enc setBuffer:keybuf
                offset:ds4_gpu_tensor_offset(key_cache)
               atIndex:3];
        [enc setBuffer:valuebuf
                offset:ds4_gpu_tensor_offset(value_cache)
               atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)src_values, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned,
                                            "DFlash KV commit")) {
            return 0;
        }
    }
    return 1;
}

typedef struct {
    uint32_t n_rows;
    uint32_t n_vocab;
} ds4_gpu_dflash_probabilities_args;

int ds4_gpu_dflash_probabilities_tensor(
        ds4_gpu_tensor       *probabilities,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *argmax,
        uint32_t              n_rows,
        uint32_t              n_vocab) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!probabilities || !logits || !argmax ||
        n_rows == 0u || n_vocab == 0u ||
        ds4_gpu_tensor_bytes(probabilities) <
            (uint64_t)n_rows * sizeof(float) ||
        ds4_gpu_tensor_bytes(logits) <
            (uint64_t)n_rows * n_vocab * sizeof(float) ||
        ds4_gpu_tensor_bytes(argmax) <
            (uint64_t)n_rows * sizeof(int32_t)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> probabilitiesbuf =
            ds4_gpu_tensor_buffer(probabilities);
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> argmaxbuf = ds4_gpu_tensor_buffer(argmax);
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dflash_probabilities");
        if (!probabilitiesbuf || !logitsbuf || !argmaxbuf || !pipeline) {
            return 0;
        }

        const ds4_gpu_dflash_probabilities_args args = {
            .n_rows = n_rows,
            .n_vocab = n_vocab,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:logitsbuf
                offset:ds4_gpu_tensor_offset(logits)
               atIndex:1];
        [enc setBuffer:argmaxbuf
                offset:ds4_gpu_tensor_offset(argmax)
               atIndex:2];
        [enc setBuffer:probabilitiesbuf
                offset:ds4_gpu_tensor_offset(probabilities)
               atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(
                cb, owned, "DFlash draft probabilities")) {
            return 0;
        }
    }
    return 1;
}

int ds4_gpu_laguna_routed_shared_moe_one_tensor(
        ds4_gpu_tensor                *routed_out,
        ds4_gpu_tensor                *routed_mid,
        ds4_gpu_tensor                *shared_out,
        ds4_gpu_tensor                *shared_mid,
        const void                    *model_map,
        uint64_t                       model_size,
        const ds4_gpu_laguna_moe_desc *routed,
        const ds4_gpu_laguna_moe_desc *shared,
        uint32_t                       expert_in_dim,
        uint32_t                       expert_mid_dim,
        uint32_t                       out_dim,
        const ds4_gpu_tensor          *selected,
        const ds4_gpu_tensor          *weights,
        uint32_t                       n_total_expert,
        uint32_t                       n_expert,
        const ds4_gpu_tensor          *shared_selected,
        const ds4_gpu_tensor          *shared_weight,
        const ds4_gpu_tensor          *x) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!routed_out || !routed_mid || !shared_out || !shared_mid ||
        !model_map || !routed || !shared || !selected || !weights ||
        !shared_selected || !shared_weight || !x || n_total_expert == 0u ||
        n_expert == 0u || n_expert > n_total_expert ||
        expert_in_dim == 0u || expert_mid_dim == 0u || out_dim == 0u ||
        routed->gate_row_bytes == 0u || routed->up_row_bytes == 0u ||
        routed->down_row_bytes == 0u || shared->gate_row_bytes == 0u ||
        shared->up_row_bytes == 0u || shared->down_row_bytes == 0u ||
        n_expert > UINT32_MAX / expert_mid_dim ||
        (expert_in_dim % 256u) != 0u || (expert_mid_dim % 256u) != 0u ||
        routed->gate_type != DS4_METAL_TENSOR_Q4_K ||
        routed->up_type != DS4_METAL_TENSOR_Q4_K ||
        shared->gate_type != DS4_METAL_TENSOR_Q4_K ||
        shared->up_type != DS4_METAL_TENSOR_Q4_K ||
        routed->down_type != shared->down_type ||
        (routed->down_type != DS4_METAL_TENSOR_Q4_K &&
         routed->down_type != DS4_METAL_TENSOR_Q6_K)) {
        return 0;
    }

    if (routed->gate_row_bytes > UINT64_MAX / expert_mid_dim ||
        routed->up_row_bytes > UINT64_MAX / expert_mid_dim ||
        routed->down_row_bytes > UINT64_MAX / out_dim ||
        shared->gate_row_bytes > UINT64_MAX / expert_mid_dim ||
        shared->up_row_bytes > UINT64_MAX / expert_mid_dim ||
        shared->down_row_bytes > UINT64_MAX / out_dim ||
        routed->gate_expert_bytes !=
            (uint64_t)expert_mid_dim * routed->gate_row_bytes ||
        routed->up_expert_bytes !=
            (uint64_t)expert_mid_dim * routed->up_row_bytes ||
        routed->down_expert_bytes !=
            (uint64_t)out_dim * routed->down_row_bytes ||
        shared->gate_expert_bytes !=
            (uint64_t)expert_mid_dim * shared->gate_row_bytes ||
        shared->up_expert_bytes !=
            (uint64_t)expert_mid_dim * shared->up_row_bytes ||
        shared->down_expert_bytes !=
            (uint64_t)out_dim * shared->down_row_bytes ||
        n_total_expert > UINT64_MAX / routed->gate_expert_bytes ||
        n_total_expert > UINT64_MAX / routed->up_expert_bytes ||
        n_total_expert > UINT64_MAX / routed->down_expert_bytes) {
        return 0;
    }

    const uint64_t routed_gate_bytes =
        (uint64_t)n_total_expert * routed->gate_expert_bytes;
    const uint64_t routed_up_bytes =
        (uint64_t)n_total_expert * routed->up_expert_bytes;
    const uint64_t routed_down_bytes =
        (uint64_t)n_total_expert * routed->down_expert_bytes;
    if (routed->gate_offset > model_size ||
        routed_gate_bytes > model_size - routed->gate_offset ||
        routed->up_offset > model_size ||
        routed_up_bytes > model_size - routed->up_offset ||
        routed->down_offset > model_size ||
        routed_down_bytes > model_size - routed->down_offset ||
        shared->gate_offset > model_size ||
        shared->gate_expert_bytes > model_size - shared->gate_offset ||
        shared->up_offset > model_size ||
        shared->up_expert_bytes > model_size - shared->up_offset ||
        shared->down_offset > model_size ||
        shared->down_expert_bytes > model_size - shared->down_offset) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> routedoutbuf = ds4_gpu_tensor_buffer(routed_out);
        id<MTLBuffer> routedmidbuf = ds4_gpu_tensor_buffer(routed_mid);
        id<MTLBuffer> sharedoutbuf = ds4_gpu_tensor_buffer(shared_out);
        id<MTLBuffer> sharedmidbuf = ds4_gpu_tensor_buffer(shared_mid);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> sharedselectedbuf = ds4_gpu_tensor_buffer(shared_selected);
        id<MTLBuffer> sharedweightbuf = ds4_gpu_tensor_buffer(shared_weight);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t routed_mid_bytes =
            (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t shared_mid_bytes =
            (uint64_t)expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!routedoutbuf || !routedmidbuf || !sharedoutbuf || !sharedmidbuf ||
            !selectedbuf || !weightsbuf || !sharedselectedbuf ||
            !sharedweightbuf || !xbuf ||
            ds4_gpu_tensor_bytes(routed_out) < out_bytes ||
            ds4_gpu_tensor_bytes(shared_out) < out_bytes ||
            ds4_gpu_tensor_bytes(routed_mid) < routed_mid_bytes ||
            ds4_gpu_tensor_bytes(shared_mid) < shared_mid_bytes ||
            ds4_gpu_tensor_bytes(selected) <
                (uint64_t)n_expert * sizeof(int32_t) ||
            ds4_gpu_tensor_bytes(weights) <
                (uint64_t)n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(shared_selected) < sizeof(int32_t) ||
            ds4_gpu_tensor_bytes(shared_weight) < sizeof(float) ||
            ds4_gpu_tensor_bytes(x) <
                (uint64_t)expert_in_dim * sizeof(float)) {
            return 0;
        }

        uint64_t routed_gate_inner = 0, routed_up_inner = 0;
        uint64_t routed_down_inner = 0, shared_gate_inner = 0;
        uint64_t shared_up_inner = 0, shared_down_inner = 0;
        id<MTLBuffer> routedgatebuf = ds4_gpu_wrap_model_range(
            model_map, model_size, routed->gate_offset, routed_gate_bytes,
            &routed_gate_inner);
        id<MTLBuffer> routedupbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, routed->up_offset, routed_up_bytes,
            &routed_up_inner);
        id<MTLBuffer> routeddownbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, routed->down_offset, routed_down_bytes,
            &routed_down_inner);
        id<MTLBuffer> sharedgatebuf = ds4_gpu_wrap_model_range(
            model_map, model_size, shared->gate_offset,
            shared->gate_expert_bytes, &shared_gate_inner);
        id<MTLBuffer> sharedupbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, shared->up_offset,
            shared->up_expert_bytes, &shared_up_inner);
        id<MTLBuffer> shareddownbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, shared->down_offset,
            shared->down_expert_bytes, &shared_down_inner);
        id<MTLComputePipelineState> pair_pipeline = ds4_gpu_hot_pipeline(
            g_laguna_routed_shared_pair_pipeline,
            "kernel_laguna_q4_K_routed_shared_pair_swiglu_f32");
        id<MTLComputePipelineState> down_pipeline =
            routed->down_type == DS4_METAL_TENSOR_Q4_K ?
            ds4_gpu_hot_pipeline(
                g_laguna_routed_shared_q4_down_pipeline,
                "kernel_laguna_q4_K_routed_shared_down_f32") :
            ds4_gpu_hot_pipeline(
                g_laguna_routed_shared_q6_down_pipeline,
                "kernel_laguna_q6_K_routed_shared_down_f32");
        if (!routedgatebuf || !routedupbuf || !routeddownbuf ||
            !sharedgatebuf || !sharedupbuf || !shareddownbuf ||
            !pair_pipeline || !down_pipeline) {
            return 0;
        }

        const ds4_gpu_glm_routed_moe_args routed_args = {
            .tp_rank = 0,
            .tp_world = 1,
            .tp_expert_base = 0,
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = n_total_expert,
            .n_expert_used = n_expert,
            .n_tokens = 1,
            .mid_token_stride = n_expert * expert_mid_dim,
            .down_type = routed->down_type,
            .gate_expert_bytes = routed->gate_expert_bytes,
            .gate_row_bytes = routed->gate_row_bytes,
            .up_expert_bytes = routed->up_expert_bytes,
            .up_row_bytes = routed->up_row_bytes,
            .down_expert_bytes = routed->down_expert_bytes,
            .down_row_bytes = routed->down_row_bytes,
        };
        const ds4_gpu_glm_routed_moe_args shared_args = {
            .tp_rank = 0,
            .tp_world = 1,
            .tp_expert_base = 0,
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = 1,
            .n_expert_used = 1,
            .n_tokens = 1,
            .mid_token_stride = expert_mid_dim,
            .down_type = shared->down_type,
            .gate_expert_bytes = shared->gate_expert_bytes,
            .gate_row_bytes = shared->gate_row_bytes,
            .up_expert_bytes = shared->up_expert_bytes,
            .up_row_bytes = shared->up_row_bytes,
            .down_expert_bytes = shared->down_expert_bytes,
            .down_row_bytes = shared->down_row_bytes,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pair_pipeline];
        [enc setBytes:&routed_args length:sizeof(routed_args) atIndex:0];
        [enc setBytes:&shared_args length:sizeof(shared_args) atIndex:1];
        [enc setBuffer:routedgatebuf offset:(NSUInteger)routed_gate_inner atIndex:2];
        [enc setBuffer:routedupbuf offset:(NSUInteger)routed_up_inner atIndex:3];
        [enc setBuffer:sharedgatebuf offset:(NSUInteger)shared_gate_inner atIndex:4];
        [enc setBuffer:sharedupbuf offset:(NSUInteger)shared_up_inner atIndex:5];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:6];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:7];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:8];
        [enc setBuffer:sharedselectedbuf
                offset:ds4_gpu_tensor_offset(shared_selected) atIndex:9];
        [enc setBuffer:sharedweightbuf
                offset:ds4_gpu_tensor_offset(shared_weight) atIndex:10];
        [enc setBuffer:routedmidbuf offset:ds4_gpu_tensor_offset(routed_mid)
                atIndex:11];
        [enc setBuffer:sharedmidbuf offset:ds4_gpu_tensor_offset(shared_mid)
                atIndex:12];
        [enc dispatchThreadgroups:MTLSizeMake(
                ((NSUInteger)expert_mid_dim + 1u) / 2u,
                (NSUInteger)n_expert + 1u,
                1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&routed_args length:sizeof(routed_args) atIndex:0];
        [enc setBytes:&shared_args length:sizeof(shared_args) atIndex:1];
        [enc setBuffer:routeddownbuf offset:(NSUInteger)routed_down_inner atIndex:2];
        [enc setBuffer:shareddownbuf offset:(NSUInteger)shared_down_inner atIndex:3];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:4];
        [enc setBuffer:sharedselectedbuf
                offset:ds4_gpu_tensor_offset(shared_selected) atIndex:5];
        [enc setBuffer:routedmidbuf offset:ds4_gpu_tensor_offset(routed_mid)
                atIndex:6];
        [enc setBuffer:sharedmidbuf offset:ds4_gpu_tensor_offset(shared_mid)
                atIndex:7];
        [enc setBuffer:routedoutbuf offset:ds4_gpu_tensor_offset(routed_out)
                atIndex:8];
        [enc setBuffer:sharedoutbuf offset:ds4_gpu_tensor_offset(shared_out)
                atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(
                ((NSUInteger)out_dim + 3u) / 4u, 2u, 1u)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(
                cb, owned, "Laguna routed/shared MoE")) {
            return 0;
        }
    }
    return 1;
}
