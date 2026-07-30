/* Laguna has a conventional residual stream and KV cache, but alternates
 * full-attention and sliding-window layers with different query geometry.
 * Keep this graph separate from DeepSeek's compressed-attention graph and
 * GLM's DSA graph so their mature scheduling paths remain untouched. */
static ds4_context_memory laguna_graph_context_memory_estimate(uint32_t ctx) {
    ds4_context_memory m = {0};
    const uint64_t kv_row_bytes =
        2ull * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
    m.prefill_cap = 1;
    m.raw_cap = ctx;
    m.comp_cap = ctx < DS4_N_SWA ? ctx : DS4_N_SWA;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t cap =
            ds4_laguna_layer_is_swa(il) ? m.comp_cap : ctx;
        m.raw_bytes += (uint64_t)cap * kv_row_bytes;
    }

    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint64_t ffn_max =
        DS4_N_FF_DENSE >
                (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP
            ? DS4_N_FF_DENSE
            : (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t scratch_f32 =
        9ull * DS4_N_EMBD +
        2ull * q_dim +
        2ull * kv_dim +
        DS4_N_HEAD +
        3ull * ffn_max +
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP +
        2ull * DS4_N_EXPERT +
        2ull * DS4_N_EXPERT_USED +
        DS4_N_VOCAB;
    m.scratch_bytes = scratch_f32 * sizeof(float);
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

typedef struct {
    ds4_gpu_tensor *features;
    const uint32_t *target_layers;
    uint32_t n_aux;
    uint32_t src_row0;
    uint32_t dst_row0;
    uint32_t n_rows;
} ds4_laguna_feature_capture;

typedef struct {
    uint32_t feature_cap;
    uint32_t block_cap;
    uint32_t cache_cap;
    uint64_t scratch_bytes;
    uint64_t kv_bytes;

    ds4_gpu_tensor *features;
    ds4_gpu_tensor *encoder;
    ds4_gpu_tensor *encoder_norm;
    ds4_gpu_tensor *norm;
    ds4_gpu_tensor *tokens;
    ds4_gpu_tensor *cur;
    ds4_gpu_tensor *next;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *k;
    ds4_gpu_tensor *v;
    ds4_gpu_tensor *qkv;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *ffn_gate;
    ds4_gpu_tensor *ffn_up;
    ds4_gpu_tensor *ffn_mid;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *staged_key;
    ds4_gpu_tensor *staged_value;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *argmax;
    ds4_gpu_tensor *probabilities;
    ds4_gpu_tensor *key_cache[DS4_DFLASH_N_LAYER];
    ds4_gpu_tensor *value_cache[DS4_DFLASH_N_LAYER];
} ds4_dflash_gpu_graph;

typedef struct {
    uint32_t ctx_size;
    uint32_t prefill_cap;
    uint64_t scratch_bytes;
    uint64_t kv_bytes;

    ds4_gpu_tensor *tokens;
    ds4_gpu_tensor *cur;
    ds4_gpu_tensor *next;
    ds4_gpu_tensor *attn_norm;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *k;
    ds4_gpu_tensor *v;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *ffn_gate;
    ds4_gpu_tensor *ffn_up;
    ds4_gpu_tensor *ffn_mid;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *shared_out;
    ds4_gpu_tensor *routed_mid;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *shared_selected;
    ds4_gpu_tensor *shared_weight;
    ds4_gpu_tensor *staged_key;
    ds4_gpu_tensor *staged_value;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *spec_output_norm;
    ds4_gpu_tensor *spec_logits;
    ds4_gpu_tensor *spec_argmax;
    ds4_gpu_tensor *spec_key_backup[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_value_backup[DS4_MAX_LAYER];
    ds4_gpu_tensor *key_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *value_cache[DS4_MAX_LAYER];
    uint32_t cache_cap[DS4_MAX_LAYER];
} ds4_laguna_gpu_graph;

static void laguna_graph_free(ds4_laguna_gpu_graph *g) {
    if (!g) return;
#define DS4_LAGUNA_FREE(name) do { \
        ds4_gpu_tensor_free(g->name); \
        g->name = NULL; \
    } while (0)
    DS4_LAGUNA_FREE(tokens);
    DS4_LAGUNA_FREE(cur);
    DS4_LAGUNA_FREE(next);
    DS4_LAGUNA_FREE(attn_norm);
    DS4_LAGUNA_FREE(q);
    DS4_LAGUNA_FREE(k);
    DS4_LAGUNA_FREE(v);
    DS4_LAGUNA_FREE(gate);
    DS4_LAGUNA_FREE(heads);
    DS4_LAGUNA_FREE(attn_out);
    DS4_LAGUNA_FREE(after_attn);
    DS4_LAGUNA_FREE(ffn_norm);
    DS4_LAGUNA_FREE(ffn_gate);
    DS4_LAGUNA_FREE(ffn_up);
    DS4_LAGUNA_FREE(ffn_mid);
    DS4_LAGUNA_FREE(ffn_out);
    DS4_LAGUNA_FREE(shared_out);
    DS4_LAGUNA_FREE(routed_mid);
    DS4_LAGUNA_FREE(router_logits);
    DS4_LAGUNA_FREE(router_probs);
    DS4_LAGUNA_FREE(router_selected);
    DS4_LAGUNA_FREE(router_weights);
    DS4_LAGUNA_FREE(shared_selected);
    DS4_LAGUNA_FREE(shared_weight);
    DS4_LAGUNA_FREE(staged_key);
    DS4_LAGUNA_FREE(staged_value);
    DS4_LAGUNA_FREE(output_norm);
    DS4_LAGUNA_FREE(logits);
    DS4_LAGUNA_FREE(spec_output_norm);
    DS4_LAGUNA_FREE(spec_logits);
    DS4_LAGUNA_FREE(spec_argmax);
#undef DS4_LAGUNA_FREE
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(g->spec_key_backup[il]);
        ds4_gpu_tensor_free(g->spec_value_backup[il]);
        ds4_gpu_tensor_free(g->key_cache[il]);
        ds4_gpu_tensor_free(g->value_cache[il]);
        g->spec_key_backup[il] = NULL;
        g->spec_value_backup[il] = NULL;
        g->key_cache[il] = NULL;
        g->value_cache[il] = NULL;
        g->cache_cap[il] = 0;
    }
    memset(g, 0, sizeof(*g));
}

static bool laguna_graph_alloc(ds4_laguna_gpu_graph *g, uint32_t ctx_size) {
    if (!g || ctx_size == 0 || ctx_size > DS4_CONTEXT_LENGTH ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_LAGUNA) {
        return false;
    }
    memset(g, 0, sizeof(*g));
    g->ctx_size = ctx_size;
    g->prefill_cap = ctx_size < 16384u ? ctx_size : 16384u;

    const uint64_t f32 = sizeof(float);
    const uint64_t rows = g->prefill_cap;
    const uint64_t embd = DS4_N_EMBD;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint64_t ffn_max = DS4_N_FF_DENSE >
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP ?
        DS4_N_FF_DENSE : (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;

#define DS4_LAGUNA_ALLOC(name, bytes) do { \
        const uint64_t ds4_laguna_bytes_ = (uint64_t)(bytes); \
        g->name = ds4_gpu_tensor_alloc(ds4_laguna_bytes_); \
        if (!g->name) goto fail; \
        g->scratch_bytes += ds4_laguna_bytes_; \
    } while (0)
    DS4_LAGUNA_ALLOC(tokens, rows * sizeof(uint32_t));
    DS4_LAGUNA_ALLOC(cur, rows * embd * f32);
    DS4_LAGUNA_ALLOC(next, rows * embd * f32);
    DS4_LAGUNA_ALLOC(attn_norm, rows * embd * f32);
    DS4_LAGUNA_ALLOC(q, rows * q_dim * f32);
    DS4_LAGUNA_ALLOC(k, rows * kv_dim * f32);
    DS4_LAGUNA_ALLOC(v, rows * kv_dim * f32);
    DS4_LAGUNA_ALLOC(gate, rows * DS4_N_HEAD * f32);
    DS4_LAGUNA_ALLOC(heads, rows * q_dim * f32);
    DS4_LAGUNA_ALLOC(attn_out, rows * embd * f32);
    DS4_LAGUNA_ALLOC(after_attn, rows * embd * f32);
    DS4_LAGUNA_ALLOC(ffn_norm, rows * embd * f32);
    DS4_LAGUNA_ALLOC(ffn_gate, rows * ffn_max * f32);
    DS4_LAGUNA_ALLOC(ffn_up, rows * ffn_max * f32);
    DS4_LAGUNA_ALLOC(ffn_mid, rows * ffn_max * f32);
    DS4_LAGUNA_ALLOC(ffn_out, rows * embd * f32);
    DS4_LAGUNA_ALLOC(shared_out, rows * embd * f32);
    DS4_LAGUNA_ALLOC(routed_mid,
                     rows * DS4_N_EXPERT_USED * DS4_N_FF_EXP * f32);
    DS4_LAGUNA_ALLOC(router_logits, rows * DS4_N_EXPERT * f32);
    DS4_LAGUNA_ALLOC(router_probs, rows * DS4_N_EXPERT * f32);
    DS4_LAGUNA_ALLOC(router_selected,
                     rows * DS4_N_EXPERT_USED * sizeof(int32_t));
    DS4_LAGUNA_ALLOC(router_weights, rows * DS4_N_EXPERT_USED * f32);
    DS4_LAGUNA_ALLOC(shared_selected, sizeof(int32_t));
    DS4_LAGUNA_ALLOC(shared_weight, sizeof(float));
    DS4_LAGUNA_ALLOC(staged_key, rows * kv_dim * sizeof(uint16_t));
    DS4_LAGUNA_ALLOC(staged_value, rows * kv_dim * sizeof(uint16_t));
    DS4_LAGUNA_ALLOC(output_norm, embd * f32);
    DS4_LAGUNA_ALLOC(logits, DS4_N_VOCAB * f32);
#undef DS4_LAGUNA_ALLOC

    const int32_t shared_id = 0;
    const float shared_weight = 1.0f;
    if (!ds4_gpu_tensor_write(g->shared_selected,
                              0,
                              &shared_id,
                              sizeof(shared_id)) ||
        !ds4_gpu_tensor_write(g->shared_weight,
                              0,
                              &shared_weight,
                              sizeof(shared_weight))) {
        goto fail;
    }

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint32_t cap = ds4_laguna_layer_is_swa(il) ? DS4_N_SWA : ctx_size;
        if (cap > ctx_size) cap = ctx_size;
        if (cap == 0) cap = 1;
        const uint64_t bytes =
            (uint64_t)cap * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
        g->key_cache[il] = ds4_gpu_tensor_alloc(bytes);
        g->value_cache[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->key_cache[il] || !g->value_cache[il]) goto fail;
        g->cache_cap[il] = cap;
        g->kv_bytes += 2u * bytes;
    }

    fprintf(stderr,
            "ds4: Laguna GPU graph: ctx=%u, prefill=%u, KV %.2f GiB, scratch %.2f MiB\n",
            ctx_size,
            g->prefill_cap,
            (double)g->kv_bytes / 1073741824.0,
            (double)g->scratch_bytes / 1048576.0);
    return true;

fail:
    fprintf(stderr, "ds4: failed to allocate Laguna GPU graph\n");
    laguna_graph_free(g);
    return false;
}

static bool laguna_graph_ensure_spec_scratch(ds4_laguna_gpu_graph *g) {
    if (!g) return false;
    const uint64_t rows = DS4_DFLASH_BLOCK_SIZE;
    if (!g->spec_output_norm) {
        g->spec_output_norm = ds4_gpu_tensor_alloc(
            rows * DS4_N_EMBD * sizeof(float));
    }
    if (!g->spec_logits) {
        g->spec_logits = ds4_gpu_tensor_alloc(
            rows * DS4_N_VOCAB * sizeof(float));
    }
    if (!g->spec_argmax) {
        g->spec_argmax =
            ds4_gpu_tensor_alloc(rows * sizeof(int32_t));
    }
    if (!g->spec_output_norm || !g->spec_logits || !g->spec_argmax) {
        fprintf(stderr,
                "ds4: failed to allocate Laguna speculative verifier scratch\n");
        return false;
    }

    const uint64_t row_bytes =
        (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
    const uint64_t backup_bytes = rows * row_bytes;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!ds4_laguna_layer_is_swa(il)) continue;
        if (!g->spec_key_backup[il]) {
            g->spec_key_backup[il] =
                ds4_gpu_tensor_alloc(backup_bytes);
        }
        if (!g->spec_value_backup[il]) {
            g->spec_value_backup[il] =
                ds4_gpu_tensor_alloc(backup_bytes);
        }
        if (!g->spec_key_backup[il] || !g->spec_value_backup[il]) {
            fprintf(stderr,
                    "ds4: failed to allocate Laguna speculative KV backup\n");
            return false;
        }
    }
    return true;
}

static bool laguna_graph_capture_feature(
        const ds4_laguna_feature_capture *capture,
        const ds4_gpu_tensor             *src,
        uint32_t                          target_layer) {
    if (!capture) return true;
    if (!capture->features || !capture->target_layers ||
        capture->n_aux == 0u || capture->n_rows == 0u) {
        return false;
    }
    for (uint32_t aux = 0; aux < capture->n_aux; aux++) {
        if (capture->target_layers[aux] != target_layer) continue;
        return ds4_gpu_dflash_capture_rows_tensor(
                   capture->features,
                   src,
                   capture->src_row0,
                   capture->dst_row0,
                   capture->n_rows,
                   DS4_N_EMBD,
                   capture->n_aux,
                   aux) != 0;
    }
    return true;
}

/* Copy a run of sliding-window ring rows in at most two transfers per layer
 * instead of one per row.  A speculative block is short and the ring wraps at
 * most once inside it, and the per-transfer cost dominates at 2 KiB a row:
 * doing this per row cost more than the draft pass it protects. */
static bool laguna_graph_spec_ring_copy(
        ds4_gpu_tensor *cache,
        ds4_gpu_tensor *backup,
        uint32_t        pos0,
        uint32_t        first_row,
        uint32_t        n_rows,
        uint32_t        cache_cap,
        uint64_t        row_bytes,
        bool            to_backup) {
    if (!cache || !backup || cache_cap == 0u) return false;
    uint32_t row = first_row;
    while (row < n_rows) {
        const uint32_t ring = (pos0 + row) % cache_cap;
        uint32_t run = n_rows - row;
        if (run > cache_cap - ring) run = cache_cap - ring;
        const uint64_t bytes = (uint64_t)run * row_bytes;
        const uint64_t cache_off = (uint64_t)ring * row_bytes;
        const uint64_t backup_off = (uint64_t)row * row_bytes;
        const int ok = to_backup ?
            ds4_gpu_tensor_copy(backup, backup_off, cache, cache_off, bytes) :
            ds4_gpu_tensor_copy(cache, cache_off, backup, backup_off, bytes);
        if (!ok) return false;
        row += run;
    }
    return true;
}

static bool laguna_graph_spec_snapshot(
        ds4_laguna_gpu_graph *g,
        uint32_t              pos0,
        uint32_t              n_rows) {
    if (!g || n_rows == 0u || n_rows > DS4_DFLASH_BLOCK_SIZE ||
        !laguna_graph_ensure_spec_scratch(g)) {
        return false;
    }
    const uint64_t row_bytes =
        (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (!ds4_laguna_layer_is_swa(il)) continue;
        ok = laguna_graph_spec_ring_copy(g->key_cache[il],
                                         g->spec_key_backup[il],
                                         pos0, 0, n_rows,
                                         g->cache_cap[il],
                                         row_bytes, true) &&
             laguna_graph_spec_ring_copy(g->value_cache[il],
                                         g->spec_value_backup[il],
                                         pos0, 0, n_rows,
                                         g->cache_cap[il],
                                         row_bytes, true);
    }
    /* Drafting and verification append to this batch. Keeping it open avoids
     * a completion whose only purpose was to delimit the KV backup. */
    if (!ok && ds4_gpu_commands_active()) {
        (void)ds4_gpu_end_commands();
    }
    return ok;
}

static bool laguna_graph_spec_restore(
        ds4_laguna_gpu_graph *g,
        uint32_t              pos0,
        uint32_t              first_row,
        uint32_t              n_rows) {
    if (!g || first_row > n_rows || n_rows > DS4_DFLASH_BLOCK_SIZE) {
        return false;
    }
    if (first_row == n_rows) return true;
    const uint64_t row_bytes =
        (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (!ds4_laguna_layer_is_swa(il)) continue;
        ok = laguna_graph_spec_ring_copy(g->key_cache[il],
                                         g->spec_key_backup[il],
                                         pos0, first_row, n_rows,
                                         g->cache_cap[il],
                                         row_bytes, false) &&
             laguna_graph_spec_ring_copy(g->value_cache[il],
                                         g->spec_value_backup[il],
                                         pos0, first_row, n_rows,
                                         g->cache_cap[il],
                                         row_bytes, false);
    }
    if (ds4_gpu_commands_active() &&
#if defined(__APPLE__)
        ds4_gpu_submit_commands() == 0
#else
        ds4_gpu_end_commands() == 0
#endif
    ) {
        ok = false;
    }
    return ok;
}

static bool laguna_graph_read_spec_logits(
        const ds4_laguna_gpu_graph *g,
        uint32_t                    row,
        float                      *logits_out) {
    if (!g || !g->spec_logits || !logits_out ||
        row >= DS4_DFLASH_BLOCK_SIZE) {
        return false;
    }
    const uint64_t row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    return ds4_gpu_tensor_read(g->spec_logits,
                               (uint64_t)row * row_bytes,
                               logits_out,
                               row_bytes) != 0;
}

#ifdef DS4_ROCM_BUILD
extern int ds4_gpu_matmul_q8_0_decode_preq_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok);
extern int ds4_gpu_matmul_q8_0_pair_decode_preq8_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x);
#endif

static bool laguna_graph_matmul(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        const ds4_tensor     *weight,
        const ds4_gpu_tensor *x,
        uint64_t              n_tokens) {
    if (!out || !model || !weight || !x || weight->ndim < 2) return false;
#ifdef DS4_ROCM_BUILD
    if (weight->type == DS4_TENSOR_Q8_0 && n_tokens == 1u) {
        return ds4_gpu_matmul_q8_0_decode_preq_tensor(
                   out,
                   model->map,
                   model->size,
                   weight->abs_offset,
                   weight->dim[0],
                   weight->dim[1],
                   x,
                   n_tokens) != 0;
    }
#endif
    if (weight->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out,
                                         model->map,
                                         model->size,
                                         weight->abs_offset,
                                         weight->dim[0],
                                         weight->dim[1],
                                         x,
                                         n_tokens) != 0;
    }
    if (weight->type == DS4_TENSOR_BF16) {
        return ds4_gpu_matmul_bf16_tensor(out,
                                          model->map,
                                          model->size,
                                          weight->abs_offset,
                                          weight->dim[0],
                                          weight->dim[1],
                                          x,
                                          n_tokens) != 0;
    }
    if (weight->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out,
                                         model->map,
                                         model->size,
                                         weight->abs_offset,
                                         weight->dim[0],
                                         weight->dim[1],
                                         x,
                                         n_tokens) != 0;
    }
    if (weight->type == DS4_TENSOR_Q6_K) {
        return ds4_gpu_matmul_q6_K_tensor(out,
                                          model->map,
                                          model->size,
                                          weight->abs_offset,
                                          weight->dim[0],
                                          weight->dim[1],
                                          x,
                                          n_tokens) != 0;
    }
    return ds4_gpu_matmul_quant_tensor(out,
                                       model->map,
                                       model->size,
                                       weight->abs_offset,
                                       weight->type,
                                       weight->dim[0],
                                       weight->dim[1],
                                       x,
                                       n_tokens) != 0;
}

static bool laguna_graph_rms_norm(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const ds4_model      *model,
        const ds4_tensor     *weight,
        uint32_t              n,
        uint32_t              rows) {
    if (!out || !x || !model || !weight || rows == 0u) return false;
    if (weight->type == DS4_TENSOR_BF16) {
        return ds4_gpu_rms_norm_bf16_weight_tensor(
                   out,
                   x,
                   model->map,
                   model->size,
                   weight->abs_offset,
                   n,
                   rows,
                   DS4_RMS_EPS) != 0;
    }
    if (rows == 1u) {
        return ds4_gpu_rms_norm_weight_tensor(
                   out,
                   x,
                   model->map,
                   model->size,
                   weight->abs_offset,
                   n,
                   DS4_RMS_EPS) != 0;
    }
    return ds4_gpu_rms_norm_weight_rows_tensor(
               out,
               x,
               model->map,
               model->size,
               weight->abs_offset,
               n,
               rows,
               DS4_RMS_EPS) != 0;
}

static ds4_gpu_nvfp4_matrix_desc laguna_native_nvfp4_desc(
        const ds4_tensor *tensor) {
    ds4_gpu_nvfp4_matrix_desc desc = {0};
    if (!tensor || !tensor->native_nvfp4) return desc;
    const ds4_native_nvfp4_matrix *native = tensor->native_nvfp4;
    desc.packed_offsets = native->packed_offset;
    desc.scale_offsets = native->scale_offset;
    desc.weight_global_scale_offsets =
        native->weight_global_scale_offset;
    desc.input_global_scale_offsets =
        native->input_global_scale_offset;
    desc.packed_bytes = native->packed_bytes;
    desc.scale_bytes = native->scale_bytes;
    return desc;
}

static ds4_gpu_laguna_moe_desc laguna_graph_moe_desc(
        const ds4_layer_weights *layer) {
    const uint64_t gate_row_bytes =
        routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t up_row_bytes =
        routed_expert_row_bytes(layer->ffn_up_exps);
    const uint64_t down_row_bytes =
        routed_expert_row_bytes(layer->ffn_down_exps);
    const ds4_gpu_laguna_moe_desc desc = {
        .gate_offset = layer->ffn_gate_exps->abs_offset,
        .up_offset = layer->ffn_up_exps->abs_offset,
        .down_offset = layer->ffn_down_exps->abs_offset,
        .gate_scale_offset = layer->ffn_gate_exps_scale ?
            layer->ffn_gate_exps_scale->abs_offset : 0u,
        .up_scale_offset = layer->ffn_up_exps_scale ?
            layer->ffn_up_exps_scale->abs_offset : 0u,
        .down_scale_offset = layer->ffn_down_exps_scale ?
            layer->ffn_down_exps_scale->abs_offset : 0u,
        .gate_type = layer->ffn_gate_exps->type,
        .up_type = layer->ffn_up_exps->type,
        .down_type = layer->ffn_down_exps->type,
        .gate_expert_bytes =
            layer->ffn_gate_exps->dim[1] * gate_row_bytes,
        .gate_row_bytes = gate_row_bytes,
        .up_expert_bytes =
            layer->ffn_up_exps->dim[1] * up_row_bytes,
        .up_row_bytes = up_row_bytes,
        .down_expert_bytes =
            layer->ffn_down_exps->dim[1] * down_row_bytes,
        .down_row_bytes = down_row_bytes,
        .gate_nvfp4 =
            laguna_native_nvfp4_desc(layer->ffn_gate_exps),
        .up_nvfp4 =
            laguna_native_nvfp4_desc(layer->ffn_up_exps),
        .down_nvfp4 =
            laguna_native_nvfp4_desc(layer->ffn_down_exps),
    };
    return desc;
}

static bool laguna_graph_matmul_decode_rows(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        const ds4_tensor     *weight,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows,
        bool                  exact_q8_rows) {
    if (exact_q8_rows && weight && weight->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                   out,
                   model->map,
                   model->size,
                   weight->abs_offset,
                   weight->dim[0],
                   weight->dim[1],
                   x,
                   n_rows) != 0;
    }
    return laguna_graph_matmul(out, model, weight, x, n_rows);
}

static bool laguna_graph_routed_moe_decode_rows(
        ds4_laguna_gpu_graph   *g,
        const ds4_model        *model,
        const ds4_layer_weights *l,
        uint32_t                layer,
        uint32_t                n_rows,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes) {
    if (!g || !model || !l || n_rows == 0u) return false;
    const uint64_t embd_bytes =
        (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t mid_elems =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t mid_bytes = mid_elems * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t weight_bytes =
        (uint64_t)DS4_N_EXPERT_USED * sizeof(float);

    if (l->ffn_gate_exps->type == DS4_TENSOR_NVFP4) {
        const ds4_gpu_laguna_moe_desc routed =
            laguna_graph_moe_desc(l);
        return ds4_gpu_laguna_routed_moe_tensor(
                   g->ffn_out,
                   g->routed_mid,
                   model->map,
                   model->size,
                   &routed,
                   DS4_N_EMBD,
                   DS4_N_FF_EXP,
                   DS4_N_EMBD,
                   g->router_selected,
                   g->router_weights,
                   DS4_N_EXPERT,
                   DS4_N_EXPERT_USED,
                   g->ffn_norm,
                   n_rows) != 0;
    }

#if defined(__APPLE__)
    if (l->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        l->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
        l->ffn_down_exps->type == DS4_TENSOR_Q4_K) {
        return ds4_gpu_glm_routed_moe_batch_decode_exact_q4_tensor(
                   g->ffn_out,
                   g->routed_mid,
                   model->map,
                   model->size,
                   l->ffn_gate_exps->abs_offset,
                   l->ffn_up_exps->abs_offset,
                   l->ffn_down_exps->abs_offset,
                   l->ffn_gate_exps->type,
                   l->ffn_up_exps->type,
                   l->ffn_down_exps->type,
                   gate_expert_bytes,
                   gate_row_bytes,
                   up_expert_bytes,
                   up_row_bytes,
                   down_expert_bytes,
                   down_row_bytes,
                   DS4_N_EMBD,
                   DS4_N_FF_EXP,
                   DS4_N_EMBD,
                   g->router_selected,
                   g->router_weights,
                   DS4_N_EXPERT,
                   DS4_N_EXPERT_USED,
                   layer,
                   g->ffn_norm,
                   n_rows,
                   (uint32_t)mid_elems) != 0;
    }
    if ((l->ffn_gate_exps->type == DS4_TENSOR_Q2_K ||
         l->ffn_gate_exps->type == DS4_TENSOR_Q3_K) &&
        l->ffn_up_exps->type == l->ffn_gate_exps->type &&
        l->ffn_down_exps->type == l->ffn_gate_exps->type &&
        n_rows <= DS4_DFLASH_BLOCK_SIZE) {
        /*
         * The Q2_K/Q3_K batch dispatch uses the same independent per-token
         * kernels as one-token decode. Encoding all verifier rows together
         * removes repeated encoder boundaries without changing reductions.
         */
        return ds4_gpu_glm_routed_moe_batch_decode_exact_q2_q3_tensor(
                   g->ffn_out,
                   g->routed_mid,
                   model->map,
                   model->size,
                   l->ffn_gate_exps->abs_offset,
                   l->ffn_up_exps->abs_offset,
                   l->ffn_down_exps->abs_offset,
                   l->ffn_gate_exps->type,
                   l->ffn_up_exps->type,
                   l->ffn_down_exps->type,
                   gate_expert_bytes,
                   gate_row_bytes,
                   up_expert_bytes,
                   up_row_bytes,
                   down_expert_bytes,
                   down_row_bytes,
                   DS4_N_EMBD,
                   DS4_N_FF_EXP,
                   DS4_N_EMBD,
                   g->router_selected,
                   g->router_weights,
                   DS4_N_EXPERT,
                   DS4_N_EXPERT_USED,
                   layer,
                   g->ffn_norm,
                   n_rows,
                   (uint32_t)mid_elems) != 0;
    }
#endif
#if defined(DS4_ROCM_BUILD) || defined(DS4_NATIVE_CUDA_BUILD)
#ifdef DS4_ROCM_BUILD
    /* The ROCm Q4_K batch path keeps decode's per-token kernels when the row
     * block is short, so it sweeps the selected experts once for the whole
     * block while every row stays bit-identical to single-row decode.  Falling
     * back to the row loop below would cost a launch storm and one full expert
     * sweep per row, which is exactly what speculation is meant to avoid. */
    if (l->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        l->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
        l->ffn_down_exps->type == DS4_TENSOR_Q4_K &&
        n_rows <= DS4_DFLASH_BLOCK_SIZE) {
        return ds4_gpu_glm_routed_moe_batch_tensor(
                   g->ffn_out,
                   g->routed_mid,
                   model->map,
                   model->size,
                   l->ffn_gate_exps->abs_offset,
                   l->ffn_up_exps->abs_offset,
                   l->ffn_down_exps->abs_offset,
                   l->ffn_gate_exps->type,
                   l->ffn_up_exps->type,
                   l->ffn_down_exps->type,
                   gate_expert_bytes,
                   gate_row_bytes,
                   up_expert_bytes,
                   up_row_bytes,
                   down_expert_bytes,
                   down_row_bytes,
                   DS4_N_EMBD,
                   DS4_N_FF_EXP,
                   DS4_N_EMBD,
                   g->router_selected,
                   g->router_weights,
                   DS4_N_EXPERT,
                   DS4_N_EXPERT_USED,
                   layer,
                   g->ffn_norm,
                   n_rows,
                   (uint32_t)mid_elems,
                   true) != 0;
    }
#endif
    if (
#ifdef DS4_NATIVE_CUDA_BUILD
        getenv("DS4_CUDA_DFLASH_ROW_MOE") == NULL &&
#endif
        (l->ffn_gate_exps->type == DS4_TENSOR_Q2_K ||
         l->ffn_gate_exps->type == DS4_TENSOR_Q3_K) &&
        l->ffn_up_exps->type == l->ffn_gate_exps->type &&
        l->ffn_down_exps->type == l->ffn_gate_exps->type &&
        n_rows <= DS4_DFLASH_BLOCK_SIZE) {
        return ds4_gpu_glm_routed_moe_batch_decode_exact_q2_q3_tensor(
                   g->ffn_out,
                   g->routed_mid,
                   model->map,
                   model->size,
                   l->ffn_gate_exps->abs_offset,
                   l->ffn_up_exps->abs_offset,
                   l->ffn_down_exps->abs_offset,
                   l->ffn_gate_exps->type,
                   l->ffn_up_exps->type,
                   l->ffn_down_exps->type,
                   gate_expert_bytes,
                   gate_row_bytes,
                   up_expert_bytes,
                   up_row_bytes,
                   down_expert_bytes,
                   down_row_bytes,
                   DS4_N_EMBD,
                   DS4_N_FF_EXP,
                   DS4_N_EMBD,
                   g->router_selected,
                   g->router_weights,
                   DS4_N_EXPERT,
                   DS4_N_EXPERT_USED,
                   layer,
                   g->ffn_norm,
                   n_rows,
                   (uint32_t)mid_elems) != 0;
    }
#endif

    for (uint32_t row = 0; row < n_rows; row++) {
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(
            g->ffn_out, (uint64_t)row * embd_bytes, embd_bytes);
        ds4_gpu_tensor *mid = ds4_gpu_tensor_view(
            g->routed_mid, (uint64_t)row * mid_bytes, mid_bytes);
        ds4_gpu_tensor *selected = ds4_gpu_tensor_view(
            g->router_selected,
            (uint64_t)row * selected_bytes,
            selected_bytes);
        ds4_gpu_tensor *weights = ds4_gpu_tensor_view(
            g->router_weights,
            (uint64_t)row * weight_bytes,
            weight_bytes);
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(
            g->ffn_norm, (uint64_t)row * embd_bytes, embd_bytes);
        const int ok =
            out && mid && selected && weights && x &&
            ds4_gpu_glm_routed_moe_one_tensor(
                out,
                mid,
                model->map,
                model->size,
                l->ffn_gate_exps->abs_offset,
                l->ffn_up_exps->abs_offset,
                l->ffn_down_exps->abs_offset,
                l->ffn_gate_exps->type,
                l->ffn_up_exps->type,
                l->ffn_down_exps->type,
                gate_expert_bytes,
                gate_row_bytes,
                up_expert_bytes,
                up_row_bytes,
                down_expert_bytes,
                down_row_bytes,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                DS4_N_EMBD,
                selected,
                weights,
                DS4_N_EXPERT,
                DS4_N_EXPERT_USED,
                layer,
                x,
                true);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(mid);
        ds4_gpu_tensor_free(out);
        if (!ok) return false;
    }
    return true;
}

static bool laguna_graph_router_decode_rows(
        ds4_laguna_gpu_graph    *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 n_rows) {
    if (!g || !model || !l || n_rows == 0u) return false;
    if (l->ffn_gate_inp->type == DS4_TENSOR_BF16) {
        return laguna_graph_matmul(g->router_logits,
                                   model,
                                   l->ffn_gate_inp,
                                   g->ffn_norm,
                                   n_rows) &&
               ds4_gpu_glm_router_select_batch_tensor(
                   g->router_selected,
                   g->router_weights,
                   g->router_probs,
                   model->map,
                   model->size,
                   l->ffn_exp_probs_b->abs_offset,
                   g->router_logits,
                   DS4_N_EXPERT,
                   DS4_N_EXPERT_USED,
                   DS4_EXPERT_WEIGHT_SCALE,
                   n_rows) != 0;
    }
#if defined(__APPLE__) || defined(DS4_ROCM_BUILD)
    return ds4_gpu_matmul_f32_decode_rows_exact_tensor(
               g->router_logits,
               model->map,
               model->size,
               l->ffn_gate_inp->abs_offset,
               DS4_N_EMBD,
               DS4_N_EXPERT,
               g->ffn_norm,
               n_rows) != 0 &&
           ds4_gpu_glm_router_select_batch_tensor(
               g->router_selected,
               g->router_weights,
               g->router_probs,
               model->map,
               model->size,
               l->ffn_exp_probs_b->abs_offset,
               g->router_logits,
               DS4_N_EXPERT,
               DS4_N_EXPERT_USED,
               DS4_EXPERT_WEIGHT_SCALE,
               n_rows) != 0;
#elif defined(DS4_NATIVE_CUDA_BUILD)
    if (getenv("DS4_CUDA_DFLASH_ROW_ROUTER") == NULL) {
        return ds4_gpu_matmul_f32_decode_rows_exact_tensor(
                   g->router_logits,
                   model->map,
                   model->size,
                   l->ffn_gate_inp->abs_offset,
                   DS4_N_EMBD,
                   DS4_N_EXPERT,
                   g->ffn_norm,
                   n_rows) != 0 &&
               ds4_gpu_glm_router_select_batch_tensor(
                   g->router_selected,
                   g->router_weights,
                   g->router_probs,
                   model->map,
                   model->size,
                   l->ffn_exp_probs_b->abs_offset,
                   g->router_logits,
                   DS4_N_EXPERT,
                   DS4_N_EXPERT_USED,
                   DS4_EXPERT_WEIGHT_SCALE,
                   n_rows) != 0;
    }
#endif
    const uint64_t embd_bytes =
        (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t expert_f32_bytes =
        (uint64_t)DS4_N_EXPERT * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t selected_weight_bytes =
        (uint64_t)DS4_N_EXPERT_USED * sizeof(float);

    for (uint32_t row = 0; row < n_rows; row++) {
        ds4_gpu_tensor *logits = ds4_gpu_tensor_view(
            g->router_logits,
            (uint64_t)row * expert_f32_bytes,
            expert_f32_bytes);
        ds4_gpu_tensor *probs = ds4_gpu_tensor_view(
            g->router_probs,
            (uint64_t)row * expert_f32_bytes,
            expert_f32_bytes);
        ds4_gpu_tensor *selected = ds4_gpu_tensor_view(
            g->router_selected,
            (uint64_t)row * selected_bytes,
            selected_bytes);
        ds4_gpu_tensor *weights = ds4_gpu_tensor_view(
            g->router_weights,
            (uint64_t)row * selected_weight_bytes,
            selected_weight_bytes);
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(
            g->ffn_norm, (uint64_t)row * embd_bytes, embd_bytes);
        const int ok =
            logits && probs && selected && weights && x &&
            ds4_gpu_matmul_f32_tensor(
                logits,
                model->map,
                model->size,
                l->ffn_gate_inp->abs_offset,
                DS4_N_EMBD,
                DS4_N_EXPERT,
                x,
                1u) &&
            ds4_gpu_glm_router_select_tensor(
                selected,
                weights,
                probs,
                model->map,
                model->size,
                l->ffn_exp_probs_b->abs_offset,
                logits,
                DS4_N_EXPERT,
                DS4_N_EXPERT_USED,
                DS4_EXPERT_WEIGHT_SCALE);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(probs);
        ds4_gpu_tensor_free(logits);
        if (!ok) return false;
    }
    return true;
}

static void dflash_graph_free(ds4_dflash_gpu_graph *g) {
    if (!g) return;
#define DS4_DFLASH_FREE(name) do { \
        ds4_gpu_tensor_free(g->name); \
        g->name = NULL; \
    } while (0)
    DS4_DFLASH_FREE(features);
    DS4_DFLASH_FREE(encoder);
    DS4_DFLASH_FREE(encoder_norm);
    DS4_DFLASH_FREE(norm);
    DS4_DFLASH_FREE(tokens);
    DS4_DFLASH_FREE(cur);
    DS4_DFLASH_FREE(next);
    DS4_DFLASH_FREE(q);
    DS4_DFLASH_FREE(k);
    DS4_DFLASH_FREE(v);
    DS4_DFLASH_FREE(qkv);
    DS4_DFLASH_FREE(gate);
    DS4_DFLASH_FREE(heads);
    DS4_DFLASH_FREE(attn_out);
    DS4_DFLASH_FREE(after_attn);
    DS4_DFLASH_FREE(ffn_norm);
    DS4_DFLASH_FREE(ffn_gate);
    DS4_DFLASH_FREE(ffn_up);
    DS4_DFLASH_FREE(ffn_mid);
    DS4_DFLASH_FREE(ffn_out);
    DS4_DFLASH_FREE(staged_key);
    DS4_DFLASH_FREE(staged_value);
    DS4_DFLASH_FREE(output_norm);
    DS4_DFLASH_FREE(logits);
    DS4_DFLASH_FREE(argmax);
    DS4_DFLASH_FREE(probabilities);
#undef DS4_DFLASH_FREE
    for (uint32_t il = 0; il < DS4_DFLASH_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->key_cache[il]);
        ds4_gpu_tensor_free(g->value_cache[il]);
    }
    memset(g, 0, sizeof(*g));
}

static bool dflash_graph_alloc(ds4_dflash_gpu_graph *g) {
    if (!g || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_LAGUNA) return false;
    memset(g, 0, sizeof(*g));
    g->feature_cap = DS4_DFLASH_CACHE_CAP;
    g->block_cap = DS4_DFLASH_BLOCK_SIZE;
    g->cache_cap = DS4_DFLASH_CACHE_CAP;

    const uint64_t f32 = sizeof(float);
    const uint64_t embd = DS4_SHAPE_LAGUNA_S21.n_embd;
    const uint64_t q_dim =
        (uint64_t)DS4_SHAPE_LAGUNA_S21.n_head *
        DS4_SHAPE_LAGUNA_S21.n_head_dim;
    const uint64_t kv_dim =
        (uint64_t)DS4_SHAPE_LAGUNA_S21.n_head_kv *
        DS4_SHAPE_LAGUNA_S21.n_head_dim;
    const uint64_t ff = DS4_SHAPE_LAGUNA_S21.n_ff_dense;
    const uint64_t feature_rows = g->feature_cap;
    const uint64_t block_rows = g->block_cap;

#define DS4_DFLASH_ALLOC(name, bytes) do { \
        const uint64_t ds4_dflash_bytes_ = (uint64_t)(bytes); \
        g->name = ds4_gpu_tensor_alloc(ds4_dflash_bytes_); \
        if (!g->name) goto fail; \
        g->scratch_bytes += ds4_dflash_bytes_; \
    } while (0)
    DS4_DFLASH_ALLOC(features,
                     feature_rows * DS4_DFLASH_N_AUX * embd * f32);
    DS4_DFLASH_ALLOC(encoder, feature_rows * embd * f32);
    DS4_DFLASH_ALLOC(encoder_norm, feature_rows * embd * f32);
    DS4_DFLASH_ALLOC(norm, feature_rows * embd * f32);
    DS4_DFLASH_ALLOC(tokens, block_rows * sizeof(uint32_t));
    DS4_DFLASH_ALLOC(cur, block_rows * embd * f32);
    DS4_DFLASH_ALLOC(next, block_rows * embd * f32);
    DS4_DFLASH_ALLOC(q, block_rows * q_dim * f32);
    DS4_DFLASH_ALLOC(k, feature_rows * kv_dim * f32);
    DS4_DFLASH_ALLOC(v, feature_rows * kv_dim * f32);
    DS4_DFLASH_ALLOC(qkv,
                     block_rows * (q_dim + 2u * kv_dim) * f32);
    DS4_DFLASH_ALLOC(gate,
                     block_rows * DS4_SHAPE_LAGUNA_S21.n_head * f32);
    DS4_DFLASH_ALLOC(heads, block_rows * q_dim * f32);
    DS4_DFLASH_ALLOC(attn_out, block_rows * embd * f32);
    DS4_DFLASH_ALLOC(after_attn, block_rows * embd * f32);
    DS4_DFLASH_ALLOC(ffn_norm, block_rows * embd * f32);
    DS4_DFLASH_ALLOC(ffn_gate, block_rows * ff * f32);
    DS4_DFLASH_ALLOC(ffn_up, block_rows * ff * f32);
    DS4_DFLASH_ALLOC(ffn_mid, block_rows * ff * f32);
    DS4_DFLASH_ALLOC(ffn_out, block_rows * embd * f32);
    DS4_DFLASH_ALLOC(staged_key,
                     block_rows * kv_dim * sizeof(uint16_t));
    DS4_DFLASH_ALLOC(staged_value,
                     block_rows * kv_dim * sizeof(uint16_t));
    DS4_DFLASH_ALLOC(output_norm, block_rows * embd * f32);
    DS4_DFLASH_ALLOC(logits, block_rows * DS4_N_VOCAB * f32);
    DS4_DFLASH_ALLOC(argmax, block_rows * sizeof(int32_t));
    DS4_DFLASH_ALLOC(probabilities, block_rows * f32);
#undef DS4_DFLASH_ALLOC

    const uint64_t cache_bytes =
        (uint64_t)g->cache_cap * kv_dim * sizeof(uint16_t);
    for (uint32_t il = 0; il < DS4_DFLASH_N_LAYER; il++) {
        g->key_cache[il] = ds4_gpu_tensor_alloc(cache_bytes);
        g->value_cache[il] = ds4_gpu_tensor_alloc(cache_bytes);
        if (!g->key_cache[il] || !g->value_cache[il]) {
            goto fail;
        }
        g->kv_bytes += 2u * cache_bytes;
    }

    fprintf(stderr,
            "ds4: DFlash graph: block=%u, history=%u, KV %.2f MiB, "
            "scratch %.2f MiB\n",
            g->block_cap,
            g->cache_cap,
            (double)g->kv_bytes / 1048576.0,
            (double)g->scratch_bytes / 1048576.0);
    return true;

fail:
    fprintf(stderr, "ds4: failed to allocate DFlash GPU graph\n");
    dflash_graph_free(g);
    return false;
}

static bool dflash_graph_matmul(
        ds4_gpu_tensor       *out,
        const ds4_engine     *e,
        const ds4_tensor     *weight,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows) {
    if (!out || !e || !weight || !x || weight->ndim < 2) {
        return false;
    }
    if (weight->type == DS4_TENSOR_BF16 && e->dflash_f16_map) {
        return ds4_gpu_matmul_f16_tensor(out,
                                         e->dflash_f16_map,
                                         e->dflash_f16_map_size,
                                         weight->abs_offset,
                                         weight->dim[0],
                                         weight->dim[1],
                                         x,
                                         n_rows) != 0;
    }
    if (e->mtp_model.native_dflash &&
        weight->type == DS4_TENSOR_BF16) {
        return ds4_gpu_matmul_bf16_tensor(out,
                                          e->mtp_model.map,
                                          e->mtp_model.size,
                                          weight->abs_offset,
                                          weight->dim[0],
                                          weight->dim[1],
                                          x,
                                          n_rows) != 0;
    }
    if (weight->type == DS4_TENSOR_Q8_0) {
#ifdef __APPLE__
        return ds4_gpu_matmul_q8_0_dflash_tensor(
                out,
                e->mtp_model.map,
                e->mtp_model.size,
                weight->abs_offset,
                weight->dim[0],
                weight->dim[1],
                x,
                n_rows) != 0;
#else
        return ds4_gpu_matmul_q8_0_tensor(out,
                                          e->mtp_model.map,
                                          e->mtp_model.size,
                                          weight->abs_offset,
                                          weight->dim[0],
                                          weight->dim[1],
                                          x,
                                          n_rows) != 0;
#endif
    }
    if (tensor_type_is_dense_quant(weight->type)) {
        return ds4_gpu_matmul_quant_tensor(out,
                                           e->mtp_model.map,
                                           e->mtp_model.size,
                                           weight->abs_offset,
                                           weight->type,
                                           weight->dim[0],
                                           weight->dim[1],
                                           x,
                                           n_rows) != 0;
    }
    return false;
}

static const void *dflash_graph_weight_map(const ds4_engine *e) {
    return e->dflash_f16_map ? e->dflash_f16_map : e->mtp_model.map;
}

static uint64_t dflash_graph_weight_map_size(const ds4_engine *e) {
    return e->dflash_f16_map ?
        e->dflash_f16_map_size : e->mtp_model.size;
}

static bool dflash_graph_rms_norm(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const ds4_engine     *e,
        const ds4_tensor     *weight,
        uint32_t              n,
        uint32_t              rows) {
    if (!out || !x || !e || !weight || rows == 0u) return false;
    if (e->mtp_model.native_dflash &&
        weight->type == DS4_TENSOR_BF16) {
        return ds4_gpu_rms_norm_bf16_weight_tensor(
                   out,
                   x,
                   e->mtp_model.map,
                   e->mtp_model.size,
                   weight->abs_offset,
                   n,
                   rows,
                   DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
    }
    const void *weight_map = dflash_graph_weight_map(e);
    const uint64_t weight_map_size = dflash_graph_weight_map_size(e);
    if (rows == 1u) {
        return ds4_gpu_rms_norm_weight_tensor(
                   out,
                   x,
                   weight_map,
                   weight_map_size,
                   weight->abs_offset,
                   n,
                   DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
    }
    return ds4_gpu_rms_norm_weight_rows_tensor(
               out,
               x,
               weight_map,
               weight_map_size,
               weight->abs_offset,
               n,
               rows,
               DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
}

static bool dflash_graph_encode_inject(
        ds4_dflash_gpu_graph *g,
        const ds4_engine     *e,
        uint32_t              pos0,
        uint32_t              n_rows) {
    if (!g || !e || !e->dflash_ready || n_rows == 0u ||
        n_rows > g->feature_cap) {
        return false;
    }
    const ds4_dflash_weights *w = &e->dflash_weights;
    const uint32_t embd = DS4_SHAPE_LAGUNA_S21.n_embd;
    const uint32_t n_head_kv = DS4_SHAPE_LAGUNA_S21.n_head_kv;
    const uint32_t head_dim = DS4_SHAPE_LAGUNA_S21.n_head_dim;
    const bool native_dflash = e->mtp_model.native_dflash;
    const uint32_t rope_context =
        native_dflash ? 262144u : DS4_CONTEXT_LENGTH;
    const float rope_theta = native_dflash ? 10000.0f : 500000.0f;
    const void *weight_map = dflash_graph_weight_map(e);
    const uint64_t weight_map_size = dflash_graph_weight_map_size(e);

    bool ok = ds4_gpu_commands_active() ||
              ds4_gpu_begin_commands() != 0;
    if (ok) {
        if (native_dflash &&
            w->aux_norm->type == DS4_TENSOR_BF16) {
            ok = ds4_gpu_dflash_aux_norm_bf16_tensor(
                     g->features,
                     e->mtp_model.map,
                     e->mtp_model.size,
                     w->aux_norm->abs_offset,
                     n_rows,
                     embd,
                     DS4_DFLASH_N_AUX,
                     DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
        } else {
            ok = ds4_gpu_dflash_aux_norm_tensor(
                     g->features,
                     weight_map,
                     weight_map_size,
                     w->aux_norm->abs_offset,
                     n_rows,
                     embd,
                     DS4_DFLASH_N_AUX,
                     DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
        }
    }
    if (ok) {
        ok = dflash_graph_matmul(g->encoder,
                                 e,
                                 w->fc,
                                 g->features,
                                 n_rows);
    }
    if (ok) {
        ok = dflash_graph_rms_norm(g->encoder_norm,
                                   g->encoder,
                                   e,
                                   w->encoder_output_norm,
                                   embd,
                                   n_rows);
    }
    for (uint32_t il = 0; ok && il < DS4_DFLASH_N_LAYER; il++) {
        const ds4_dflash_layer_weights *l = &w->layer[il];
        ok = dflash_graph_rms_norm(g->norm,
                                   g->encoder_norm,
                                   e,
                                   l->attn_norm,
                                   embd,
                                   n_rows);
        if (ok) {
            ok = dflash_graph_matmul(g->k, e, l->attn_k,
                                     g->norm, n_rows) &&
                 dflash_graph_matmul(g->v, e, l->attn_v,
                                     g->norm, n_rows);
        }
        if (ok) {
            if (native_dflash &&
                l->attn_k_norm->type == DS4_TENSOR_BF16) {
                ok = ds4_gpu_laguna_head_rms_norm_rope_bf16_tensor(
                         g->k,
                         e->mtp_model.map,
                         e->mtp_model.size,
                         l->attn_k_norm->abs_offset,
                         n_rows,
                         n_head_kv,
                         head_dim,
                         head_dim,
                         pos0,
                         rope_context,
                         rope_theta,
                         1.0f,
                         0.0f,
                         1.0f,
                         0.0f,
                         0.0f,
                         DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
            } else {
                ok = ds4_gpu_laguna_head_rms_norm_rope_tensor(
                         g->k,
                         weight_map,
                         weight_map_size,
                         l->attn_k_norm->abs_offset,
                         n_rows,
                         n_head_kv,
                         head_dim,
                         DS4_SHAPE_LAGUNA_S21.n_rot_swa,
                         pos0,
                         rope_context,
                         rope_theta,
                         1.0f,
                         0.0f,
                         1.0f,
                         0.0f,
                         0.0f,
                         DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
            }
        }
        if (ok) {
            ok = ds4_gpu_dflash_commit_kv_tensor(
                     g->key_cache[il],
                     g->value_cache[il],
                     g->k,
                     g->v,
                     pos0,
                     n_rows,
                     g->cache_cap,
                     n_head_kv,
                     head_dim) != 0;
        }
    }
    if (ds4_gpu_commands_active() && ds4_gpu_end_commands() == 0) ok = false;
    return ok;
}

static bool dflash_graph_draft_block(
        ds4_dflash_gpu_graph *g,
        const ds4_engine     *e,
        int                   first_token,
        uint32_t              pos0,
        uint32_t              n_draft) {
    if (!g || !e || !e->dflash_ready ||
        first_token < 0 || first_token >= (int)DS4_N_VOCAB ||
        n_draft == 0u || n_draft + 1u > g->block_cap) {
        return false;
    }
    const uint32_t n_rows = n_draft + 1u;
    uint32_t token_ids[DS4_DFLASH_BLOCK_SIZE];
    token_ids[0] = (uint32_t)first_token;
    for (uint32_t row = 1; row < n_rows; row++) {
        token_ids[row] = e->dflash_weights.mask_token_id;
    }
    if (!ds4_gpu_tensor_write(g->tokens,
                              0,
                              token_ids,
                              (uint64_t)n_rows * sizeof(token_ids[0]))) {
        return false;
    }

    const ds4_dflash_weights *w = &e->dflash_weights;
    const uint32_t embd = DS4_SHAPE_LAGUNA_S21.n_embd;
    const uint32_t n_head = DS4_SHAPE_LAGUNA_S21.n_head;
    const uint32_t n_head_kv = DS4_SHAPE_LAGUNA_S21.n_head_kv;
    const uint32_t head_dim = DS4_SHAPE_LAGUNA_S21.n_head_dim;
    const uint32_t q_dim = n_head * head_dim;
    const uint32_t kv_dim = n_head_kv * head_dim;
    const uint32_t ff = DS4_SHAPE_LAGUNA_S21.n_ff_dense;
    const bool native_dflash = e->mtp_model.native_dflash;
    const uint32_t rope_context =
        native_dflash ? 262144u : DS4_CONTEXT_LENGTH;
    const float rope_theta = native_dflash ? 10000.0f : 500000.0f;
    const void *weight_map = dflash_graph_weight_map(e);
    const uint64_t weight_map_size = dflash_graph_weight_map_size(e);

    bool ok = ds4_gpu_commands_active() ||
              ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_embed_tokens_quant_tensor(
                 g->cur,
                 g->tokens,
                 e->model.map,
                 e->model.size,
                 e->weights.token_embd->abs_offset,
                 e->weights.token_embd->type,
                 DS4_N_VOCAB,
                 n_rows,
                 embd) != 0;
    }
    for (uint32_t il = 0; ok && il < DS4_DFLASH_N_LAYER; il++) {
        const ds4_dflash_layer_weights *l = &w->layer[il];
        ok = dflash_graph_rms_norm(g->norm,
                                   g->cur,
                                   e,
                                   l->attn_norm,
                                   embd,
                                   n_rows);
        if (ok) {
            bool projections_done = false;
            const uint64_t q_weight_bytes =
                (uint64_t)q_dim * embd * sizeof(uint16_t);
            const uint64_t kv_weight_bytes =
                (uint64_t)kv_dim * embd * sizeof(uint16_t);
            const bool native_fused_qkv =
                native_dflash &&
                l->attn_k->abs_offset ==
                    l->attn_q->abs_offset + q_weight_bytes &&
                l->attn_v->abs_offset ==
                    l->attn_k->abs_offset + kv_weight_bytes;
            const bool all_bf16 =
                l->attn_q->type == DS4_TENSOR_BF16 &&
                l->attn_k->type == DS4_TENSOR_BF16 &&
                l->attn_v->type == DS4_TENSOR_BF16 &&
                l->attn_gate->type == DS4_TENSOR_BF16;
            if (native_fused_qkv && all_bf16 &&
                getenv("DS4_CUDA_DFLASH_NO_BF16_QKVG_FUSION") == NULL) {
                projections_done =
                    ds4_gpu_dflash_qkvg_bf16_tensor(
                        g->q,
                        g->k,
                        g->v,
                        g->gate,
                        g->qkv,
                        e->mtp_model.map,
                        e->mtp_model.size,
                        l->attn_q->abs_offset,
                        l->attn_gate->abs_offset,
                        n_rows,
                        g->norm) != 0;
            } else if (native_dflash && all_bf16 &&
                       getenv(
                           "DS4_CUDA_DFLASH_NO_BF16_QKVG_FUSION") == NULL) {
                projections_done =
                    ds4_gpu_laguna_qkvg_bf16_tensor(
                        g->q,
                        g->k,
                        g->v,
                        g->gate,
                        e->mtp_model.map,
                        e->mtp_model.size,
                        l->attn_q->abs_offset,
                        l->attn_k->abs_offset,
                        l->attn_v->abs_offset,
                        l->attn_gate->abs_offset,
                        embd,
                        q_dim,
                        kv_dim,
                        n_head,
                        n_rows,
                        g->norm) != 0;
            }
            ok = projections_done ||
                 (dflash_graph_matmul(g->q, e, l->attn_q,
                                      g->norm, n_rows) &&
                  dflash_graph_matmul(g->k, e, l->attn_k,
                                      g->norm, n_rows) &&
                  dflash_graph_matmul(g->v, e, l->attn_v,
                                      g->norm, n_rows) &&
                  dflash_graph_matmul(g->gate, e, l->attn_gate,
                                      g->norm, n_rows));
        }
        if (ok) {
            if (native_dflash &&
                l->attn_q_norm->type == DS4_TENSOR_BF16 &&
                getenv(
                    "DS4_CUDA_DFLASH_NO_QK_NORM_ROPE_FUSION") == NULL) {
                ok = ds4_gpu_laguna_qk_head_rms_norm_rope_bf16_tensor(
                         g->q,
                         g->k,
                         e->mtp_model.map,
                         e->mtp_model.size,
                         l->attn_q_norm->abs_offset,
                         l->attn_k_norm->abs_offset,
                         n_rows,
                         n_head,
                         n_head_kv,
                         head_dim,
                         head_dim,
                         pos0,
                         rope_context,
                         rope_theta,
                         1.0f,
                         0.0f,
                         1.0f,
                         0.0f,
                         0.0f,
                         DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
            } else if (native_dflash &&
                       l->attn_q_norm->type == DS4_TENSOR_BF16) {
                ok = ds4_gpu_laguna_head_rms_norm_rope_bf16_tensor(
                         g->q,
                         e->mtp_model.map,
                         e->mtp_model.size,
                         l->attn_q_norm->abs_offset,
                         n_rows,
                         n_head,
                         head_dim,
                         head_dim,
                         pos0,
                         rope_context,
                         rope_theta,
                         1.0f,
                         0.0f,
                         1.0f,
                         0.0f,
                         0.0f,
                         DS4_SHAPE_LAGUNA_S21.rms_eps) != 0 &&
                     ds4_gpu_laguna_head_rms_norm_rope_bf16_tensor(
                         g->k,
                         e->mtp_model.map,
                         e->mtp_model.size,
                         l->attn_k_norm->abs_offset,
                         n_rows,
                         n_head_kv,
                         head_dim,
                         head_dim,
                         pos0,
                         rope_context,
                         rope_theta,
                         1.0f,
                         0.0f,
                         1.0f,
                         0.0f,
                         0.0f,
                         DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
            } else {
                ok = ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                         g->q,
                         g->k,
                         weight_map,
                         weight_map_size,
                         l->attn_q_norm->abs_offset,
                         l->attn_k_norm->abs_offset,
                         n_rows,
                         n_head,
                         n_head_kv,
                         head_dim,
                         DS4_SHAPE_LAGUNA_S21.n_rot_swa,
                         pos0,
                         rope_context,
                         rope_theta,
                         1.0f,
                         0.0f,
                         1.0f,
                         0.0f,
                         0.0f,
                         DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
            }
        }
        if (ok) {
            ok = ds4_gpu_laguna_attention_prefill_tensor(
                     g->heads,
                     g->key_cache[il],
                     g->value_cache[il],
                     g->staged_key,
                     g->staged_value,
                     g->q,
                     g->k,
                     g->v,
                     g->gate,
                     pos0,
                     n_rows,
                     g->cache_cap,
                     n_head,
                     n_head_kv,
                     head_dim,
                     1.0f / sqrtf((float)head_dim),
                     1) != 0;
        }
        if (ok) {
            ok = dflash_graph_matmul(g->attn_out,
                                     e,
                                     l->attn_output,
                                     g->heads,
                                     n_rows) &&
                 ds4_gpu_add_tensor(g->after_attn,
                                    g->cur,
                                    g->attn_out,
                                    (uint64_t)n_rows * embd) != 0;
        }
        if (ok) {
            ok = dflash_graph_rms_norm(g->ffn_norm,
                                       g->after_attn,
                                       e,
                                       l->ffn_norm,
                                       embd,
                                       n_rows);
        }
        if (ok) {
            ok = dflash_graph_matmul(g->ffn_gate,
                                     e,
                                     l->ffn_gate,
                                     g->ffn_norm,
                                     n_rows) &&
                 dflash_graph_matmul(g->ffn_up,
                                     e,
                                     l->ffn_up,
                                     g->ffn_norm,
                                     n_rows);
        }
        if (ok) {
            ok = ds4_gpu_swiglu_tensor(g->ffn_mid,
                                        g->ffn_gate,
                                        g->ffn_up,
                                        (uint64_t)n_rows * ff,
                                        0.0f,
                                        1.0f) != 0;
        }
        if (ok) {
            ok = dflash_graph_matmul(g->ffn_out,
                                     e,
                                     l->ffn_down,
                                     g->ffn_mid,
                                     n_rows) &&
                 ds4_gpu_add_tensor(g->next,
                                    g->after_attn,
                                    g->ffn_out,
                                    (uint64_t)n_rows * embd) != 0;
        }
        if (ok) {
            ds4_gpu_tensor *tmp = g->cur;
            g->cur = g->next;
            g->next = tmp;
        }
    }
    if (ok) {
        ok = dflash_graph_rms_norm(g->output_norm,
                                   g->cur,
                                   e,
                                   w->output_norm,
                                   embd,
                                   n_rows);
    }
    if (ok) {
#ifdef __APPLE__
        if (e->weights.output->type == DS4_TENSOR_Q8_0) {
            ok = ds4_gpu_matmul_q8_0_dflash_tensor(
                     g->logits,
                     e->model.map,
                     e->model.size,
                     e->weights.output->abs_offset,
                     e->weights.output->dim[0],
                     e->weights.output->dim[1],
                     g->output_norm,
                     n_rows) != 0;
        } else
#endif
        {
            ok = laguna_graph_matmul(g->logits,
                                     &e->model,
                                     e->weights.output,
                                     g->output_norm,
                                     n_rows);
        }
    }
    if (ok) {
#ifdef DS4_ROCM_BUILD
        ok = ds4_gpu_argmax_rows_tensor(g->argmax,
                                        g->logits,
                                        DS4_N_VOCAB,
                                        n_rows) != 0;
#else
        ok = ds4_gpu_indexer_topk_tensor(g->argmax,
                                          g->logits,
                                          DS4_N_VOCAB,
                                          n_rows,
                                          1) != 0;
#endif
    }
    if (ok && e->dflash_p_min > 0.0f) {
        ok = ds4_gpu_dflash_probabilities_tensor(
                 g->probabilities,
                 g->logits,
                 g->argmax,
                 n_rows,
                 DS4_N_VOCAB) != 0;
    }
    if (!ok && ds4_gpu_commands_active()) {
        (void)ds4_gpu_end_commands();
    }
    (void)q_dim;
    (void)kv_dim;
    return ok;
}

static bool laguna_graph_forward_token(
        ds4_laguna_gpu_graph *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        int                   token,
        uint32_t              pos,
        const ds4_laguna_feature_capture *capture,
        float                *logits_out) {
    if (!g || !model || !weights || token < 0 ||
        token >= (int)DS4_N_VOCAB || pos >= g->ctx_size) {
        return false;
    }

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_embed_token_quant_tensor(g->cur,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              weights->token_embd->type,
                                              DS4_N_VOCAB,
                                              (uint32_t)token,
                                              DS4_N_EMBD) != 0;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t n_head = ds4_layer_head_count(il);
        const uint32_t q_dim = n_head * DS4_N_HEAD_DIM;
        const bool is_swa = ds4_laguna_layer_is_swa(il);
        const uint32_t n_rot = is_swa ? DS4_N_ROT_SWA : DS4_N_ROT;
        const float freq_base = is_swa ? DS4_ROPE_FREQ_BASE_SWA : DS4_ROPE_FREQ_BASE;
        const float freq_scale = is_swa ? 1.0f : 1.0f / DS4_ROPE_SCALE_FACTOR;
        const float ext_factor = is_swa ? 0.0f : 1.0f;
        /* rope_yarn applies the YaRN magnitude multiplier internally.  The
         * reference runtime passes the configured attention factor here; its
         * context-level adjustment first constructs, then cancels, that same
         * internal multiplier.  Dividing once more suppresses YaRN mscale. */
        const float attn_factor = is_swa ? 1.0f :
            DS4_ROPE_YARN_ATTN_FACTOR;
        const float beta_fast = is_swa ? 0.0f : DS4_ROPE_YARN_BETA_FAST;
        const float beta_slow = is_swa ? 0.0f : DS4_ROPE_YARN_BETA_SLOW;
        const uint32_t rope_ctx = is_swa ?
            (uint32_t)DS4_CONTEXT_LENGTH : (uint32_t)DS4_ROPE_ORIG_CTX;

        ok = laguna_graph_capture_feature(capture, g->cur, il);
        if (!ok) break;
        ok = laguna_graph_rms_norm(g->attn_norm,
                                   g->cur,
                                   model,
                                   l->attn_norm,
                                   DS4_N_EMBD,
                                   1u);
        if (ok) {
            if (l->attn_q->type == DS4_TENSOR_F16) {
                ok = ds4_gpu_laguna_qkvg_f16_tensor(
                        g->q,
                        g->k,
                        g->v,
                        g->gate,
                        model->map,
                        model->size,
                        l->attn_q->abs_offset,
                        l->attn_k->abs_offset,
                        l->attn_v->abs_offset,
                        l->attn_gate->abs_offset,
                        DS4_N_EMBD,
                        q_dim,
                        DS4_N_HEAD_KV * DS4_N_HEAD_DIM,
                        n_head,
                        g->attn_norm) != 0;
            } else if (l->attn_q->type == DS4_TENSOR_BF16) {
                ok = ds4_gpu_laguna_qkvg_bf16_tensor(
                        g->q,
                        g->k,
                        g->v,
                        g->gate,
                        model->map,
                        model->size,
                        l->attn_q->abs_offset,
                        l->attn_k->abs_offset,
                        l->attn_v->abs_offset,
                        l->attn_gate->abs_offset,
                        DS4_N_EMBD,
                        q_dim,
                        DS4_N_HEAD_KV * DS4_N_HEAD_DIM,
                        n_head,
                        1u,
                        g->attn_norm) != 0;
            } else {
#ifdef DS4_ROCM_BUILD
                ok = ds4_gpu_matmul_q8_0_pair_decode_preq8_tensor(
                        g->q,
                        g->k,
                        model->map,
                        model->size,
                        l->attn_q->abs_offset,
                        l->attn_k->abs_offset,
                        DS4_N_EMBD,
                        q_dim,
                        DS4_N_HEAD_KV * DS4_N_HEAD_DIM,
                        g->attn_norm) != 0 &&
                     ds4_gpu_matmul_q8_0_pair_decode_preq8_tensor(
                        g->v,
                        g->gate,
                        model->map,
                        model->size,
                        l->attn_v->abs_offset,
                        l->attn_gate->abs_offset,
                        DS4_N_EMBD,
                        DS4_N_HEAD_KV * DS4_N_HEAD_DIM,
                        n_head,
                        g->attn_norm) != 0;
#else
                ok = ds4_gpu_matmul_q8_0_pair_tensor(
                        g->q,
                        g->k,
                        model->map,
                        model->size,
                        l->attn_q->abs_offset,
                        l->attn_k->abs_offset,
                        DS4_N_EMBD,
                        q_dim,
                        DS4_N_HEAD_KV * DS4_N_HEAD_DIM,
                        g->attn_norm,
                        1) != 0 &&
                     ds4_gpu_matmul_q8_0_pair_tensor(
                        g->v,
                        g->gate,
                        model->map,
                        model->size,
                        l->attn_v->abs_offset,
                        l->attn_gate->abs_offset,
                        DS4_N_EMBD,
                        DS4_N_HEAD_KV * DS4_N_HEAD_DIM,
                        n_head,
                        g->attn_norm,
                        1) != 0;
#endif
            }
        }
        if (ok) {
            if (l->attn_q_norm->type == DS4_TENSOR_BF16) {
                ok = ds4_gpu_laguna_qk_head_rms_norm_rope_bf16_tensor(
                        g->q,
                        g->k,
                        model->map,
                        model->size,
                        l->attn_q_norm->abs_offset,
                        l->attn_k_norm->abs_offset,
                        1u,
                        n_head,
                        DS4_N_HEAD_KV,
                        DS4_N_HEAD_DIM,
                        n_rot,
                        pos,
                        rope_ctx,
                        freq_base,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        beta_fast,
                        beta_slow,
                        DS4_RMS_EPS) != 0;
            } else {
                ok = ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                        g->q,
                        g->k,
                        model->map,
                        model->size,
                        l->attn_q_norm->abs_offset,
                        l->attn_k_norm->abs_offset,
                        1u,
                        n_head,
                        DS4_N_HEAD_KV,
                        DS4_N_HEAD_DIM,
                        n_rot,
                        pos,
                        rope_ctx,
                        freq_base,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        beta_fast,
                        beta_slow,
                        DS4_RMS_EPS) != 0;
            }
        }
        uint32_t key_count = pos + 1u;
        if (key_count > g->cache_cap[il]) key_count = g->cache_cap[il];
        const uint32_t key_start = pos + 1u - key_count;
        if (ok) {
            ok = ds4_gpu_laguna_store_attention_tensor(
                    g->heads,
                    g->key_cache[il],
                    g->value_cache[il],
                    g->q,
                    g->k,
                    g->v,
                    g->gate,
                    pos,
                    g->cache_cap[il],
                    key_start,
                    key_count,
                    n_head,
                    DS4_N_HEAD_KV,
                    DS4_N_HEAD_DIM,
                    1.0f / sqrtf((float)DS4_N_HEAD_DIM)) != 0;
        }
        if (ok) {
            if (l->attn_output->type == DS4_TENSOR_F16) {
                ok = ds4_gpu_laguna_attn_output_residual_f16_tensor(
                        g->after_attn,
                        model->map,
                        model->size,
                        l->attn_output->abs_offset,
                        q_dim,
                        DS4_N_EMBD,
                        g->heads,
                        g->cur) != 0;
            } else {
                ok = laguna_graph_matmul(g->attn_out,
                                         model,
                                         l->attn_output,
                                         g->heads,
                                         1) &&
                     ds4_gpu_add_tensor(g->after_attn,
                                        g->cur,
                                        g->attn_out,
                                        DS4_N_EMBD) != 0;
            }
        }
        if (ok) {
            ok = laguna_graph_rms_norm(g->ffn_norm,
                                       g->after_attn,
                                       model,
                                       l->ffn_norm,
                                       DS4_N_EMBD,
                                       1u);
        }
        if (ok && il < DS4_N_LEADING_DENSE) {
            ok = laguna_graph_matmul(g->ffn_gate,
                                     model,
                                     l->ffn_gate,
                                     g->ffn_norm,
                                     1) &&
                 laguna_graph_matmul(g->ffn_up,
                                     model,
                                     l->ffn_up,
                                     g->ffn_norm,
                                     1);
            if (ok) {
                ok = ds4_gpu_swiglu_tensor(g->ffn_mid,
                                            g->ffn_gate,
                                            g->ffn_up,
                                            DS4_N_FF_DENSE,
                                            0.0f,
                                            1.0f) != 0;
            }
            if (ok) {
                ok = laguna_graph_matmul(g->ffn_out,
                                         model,
                                         l->ffn_down,
                                         g->ffn_mid,
                                         1);
            }
            if (ok) {
                ok = ds4_gpu_add_tensor(g->next,
                                        g->after_attn,
                                        g->ffn_out,
                                        DS4_N_EMBD) != 0;
            }
        } else if (ok) {
            ok = laguna_graph_matmul(g->router_logits,
                                     model,
                                     l->ffn_gate_inp,
                                     g->ffn_norm,
                                     1u);
            if (ok) {
                ok = ds4_gpu_glm_router_select_tensor(
                        g->router_selected,
                        g->router_weights,
                        g->router_probs,
                        model->map,
                        model->size,
                        l->ffn_exp_probs_b->abs_offset,
                        g->router_logits,
                        DS4_N_EXPERT,
                        DS4_N_EXPERT_USED,
                        DS4_EXPERT_WEIGHT_SCALE) != 0;
            }

            const uint64_t gate_row_bytes =
                routed_expert_row_bytes(l->ffn_gate_exps);
            const uint64_t up_row_bytes =
                routed_expert_row_bytes(l->ffn_up_exps);
            const uint64_t down_row_bytes =
                routed_expert_row_bytes(l->ffn_down_exps);
            const uint64_t gate_expert_bytes =
                l->ffn_gate_exps->dim[1] * gate_row_bytes;
            const uint64_t up_expert_bytes =
                l->ffn_up_exps->dim[1] * up_row_bytes;
            const uint64_t down_expert_bytes =
                l->ffn_down_exps->dim[1] * down_row_bytes;
            const uint64_t shared_gate_row_bytes =
                tensor_is_routed_expert_type(
                    l->ffn_gate_shexp->type) ?
                    routed_expert_row_bytes(l->ffn_gate_shexp) : 0u;
            const uint64_t shared_up_row_bytes =
                tensor_is_routed_expert_type(
                    l->ffn_up_shexp->type) ?
                    routed_expert_row_bytes(l->ffn_up_shexp) : 0u;
            const uint64_t shared_down_row_bytes =
                tensor_is_routed_expert_type(
                    l->ffn_down_shexp->type) ?
                    routed_expert_row_bytes(l->ffn_down_shexp) : 0u;
            const ds4_gpu_laguna_moe_desc routed_moe =
                laguna_graph_moe_desc(l);
            const ds4_gpu_laguna_moe_desc shared_moe = {
                .gate_offset = l->ffn_gate_shexp->abs_offset,
                .up_offset = l->ffn_up_shexp->abs_offset,
                .down_offset = l->ffn_down_shexp->abs_offset,
                .gate_type = l->ffn_gate_shexp->type,
                .up_type = l->ffn_up_shexp->type,
                .down_type = l->ffn_down_shexp->type,
                .gate_expert_bytes =
                    l->ffn_gate_shexp->dim[1] * shared_gate_row_bytes,
                .gate_row_bytes = shared_gate_row_bytes,
                .up_expert_bytes =
                    l->ffn_up_shexp->dim[1] * shared_up_row_bytes,
                .up_row_bytes = shared_up_row_bytes,
                .down_expert_bytes =
                    l->ffn_down_shexp->dim[1] * shared_down_row_bytes,
                .down_row_bytes = shared_down_row_bytes,
            };
            /* The legacy recipe can co-dispatch routed and shared Q4 work.
             * Revised layouts keep one routed quantization but make the
             * shared expert Q8, so each uses its native fast kernel family. */
            if (ok &&
                l->ffn_gate_exps->type == DS4_TENSOR_NVFP4) {
                ok = ds4_gpu_laguna_routed_moe_tensor(
                        g->ffn_out,
                        g->routed_mid,
                        model->map,
                        model->size,
                        &routed_moe,
                        DS4_N_EMBD,
                        DS4_N_FF_EXP,
                        DS4_N_EMBD,
                        g->router_selected,
                        g->router_weights,
                        DS4_N_EXPERT,
                        DS4_N_EXPERT_USED,
                        g->ffn_norm,
                        1u) != 0;
                if (ok) {
                    ok = laguna_graph_matmul(g->ffn_gate,
                                             model,
                                             l->ffn_gate_shexp,
                                             g->ffn_norm,
                                             1u) &&
                         laguna_graph_matmul(g->ffn_up,
                                             model,
                                             l->ffn_up_shexp,
                                             g->ffn_norm,
                                             1u);
                }
                if (ok) {
                    ok = ds4_gpu_swiglu_tensor(g->ffn_mid,
                                               g->ffn_gate,
                                               g->ffn_up,
                                               DS4_N_FF_SHARED,
                                               0.0f,
                                               1.0f) != 0;
                }
                if (ok) {
                    ok = laguna_graph_matmul(g->shared_out,
                                             model,
                                             l->ffn_down_shexp,
                                             g->ffn_mid,
                                             1u);
                }
            } else if (ok &&
                       l->ffn_gate_shexp->type == DS4_TENSOR_Q4_K) {
                ok = ds4_gpu_laguna_routed_shared_moe_one_tensor(
                        g->ffn_out,
                        g->routed_mid,
                        g->shared_out,
                        g->ffn_mid,
                        model->map,
                        model->size,
                        &routed_moe,
                        &shared_moe,
                        DS4_N_EMBD,
                        DS4_N_FF_EXP,
                        DS4_N_EMBD,
                        g->router_selected,
                        g->router_weights,
                        DS4_N_EXPERT,
                        DS4_N_EXPERT_USED,
                        g->shared_selected,
                        g->shared_weight,
                        g->ffn_norm) != 0;
            } else if (ok) {
                ok = ds4_gpu_glm_routed_moe_one_tensor(
                        g->ffn_out,
                        g->routed_mid,
                        model->map,
                        model->size,
                        l->ffn_gate_exps->abs_offset,
                        l->ffn_up_exps->abs_offset,
                        l->ffn_down_exps->abs_offset,
                        l->ffn_gate_exps->type,
                        l->ffn_up_exps->type,
                        l->ffn_down_exps->type,
                        gate_expert_bytes,
                        gate_row_bytes,
                        up_expert_bytes,
                        up_row_bytes,
                        down_expert_bytes,
                        down_row_bytes,
                        DS4_N_EMBD,
                        DS4_N_FF_EXP,
                        DS4_N_EMBD,
                        g->router_selected,
                        g->router_weights,
                        DS4_N_EXPERT,
                        DS4_N_EXPERT_USED,
                        il,
                        g->ffn_norm,
                        true) != 0;
                if (ok) {
                    ok = ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                            g->ffn_mid,
                            model->map,
                            model->size,
                            l->ffn_gate_shexp->abs_offset,
                            l->ffn_up_shexp->abs_offset,
                            DS4_N_EMBD,
                            DS4_N_FF_SHARED,
                            g->ffn_norm,
                            0.0f) != 0;
                }
                if (ok) {
                    ok = laguna_graph_matmul(g->shared_out,
                                             model,
                                             l->ffn_down_shexp,
                                             g->ffn_mid,
                                             1);
                }
            }
            if (ok) {
                ok = ds4_gpu_add3_tensor(g->next,
                                         g->after_attn,
                                         g->ffn_out,
                                         g->shared_out,
                                         DS4_N_EMBD) != 0;
            }
        }

        if (ok) {
            ds4_gpu_tensor *tmp = g->cur;
            g->cur = g->next;
            g->next = tmp;
        }
    }

    if (ok) {
        ok = laguna_graph_capture_feature(capture, g->cur, DS4_N_LAYER);
    }
    if (ok && logits_out) {
        ok = laguna_graph_rms_norm(g->output_norm,
                                   g->cur,
                                   model,
                                   weights->output_norm,
                                   DS4_N_EMBD,
                                   1u);
        if (ok) {
            ok = laguna_graph_matmul(g->logits,
                                     model,
                                     weights->output,
                                     g->output_norm,
                                     1);
        }
    }
    if (ds4_gpu_commands_active() && ds4_gpu_end_commands() == 0) ok = false;
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits,
                                 0,
                                 logits_out,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    return ok;
}

static void laguna_graph_report_prefill_display_progress(
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                layer_done,
        int                     total,
        bool                    allow_complete) {
    if (!display_progress || n_tokens == 0) return;
    if (layer_done > (uint32_t)DS4_N_LAYER) {
        layer_done = (uint32_t)DS4_N_LAYER;
    }

    uint64_t done = (uint64_t)n_tokens * layer_done /
                    (uint32_t)DS4_N_LAYER;
    if (layer_done == (uint32_t)DS4_N_LAYER) done = n_tokens;
    if (!allow_complete && done >= n_tokens) done = n_tokens - 1u;
    display_progress(display_progress_ud,
                     "prefill_display",
                     (int)((uint64_t)pos0 + done),
                     total);
}

static bool laguna_graph_capture_final_feature(
        ds4_laguna_gpu_graph             *g,
        const ds4_model                  *model,
        const ds4_layer_weights          *layer,
        uint32_t                          n_tokens,
        const ds4_laguna_feature_capture *capture) {
    if (!capture) return true;
    if (!g || !model || !layer ||
        capture->src_row0 > n_tokens ||
        capture->n_rows > n_tokens - capture->src_row0) {
        return false;
    }

    /*
     * The fast Q8 batch kernel stages FP32 activations as FP16. Laguna's final
     * shared-expert SwiGLU can legitimately exceed the FP16 range on early
     * prompt rows; normal inference does not consume those final hidden rows,
     * but DFlash does. Recompute only the rows DFlash captures with the
     * decode-row reduction, which reads FP32 directly. The ordinary target
     * result in g->cur remains untouched, so enabling DFlash cannot alter
     * target logits.
     */
    if (n_tokens <= 16u ||
        layer->ffn_down_shexp->type != DS4_TENSOR_Q8_0) {
        return laguna_graph_capture_feature(capture,
                                            g->cur,
                                            DS4_N_LAYER);
    }

    const uint64_t rows = capture->n_rows;
    const uint64_t embd_bytes = rows * DS4_N_EMBD * sizeof(float);
    const uint64_t shared_bytes =
        rows * DS4_N_FF_SHARED * sizeof(float);
    const uint64_t src_embd_off =
        (uint64_t)capture->src_row0 * DS4_N_EMBD * sizeof(float);
    const uint64_t src_shared_off =
        (uint64_t)capture->src_row0 * DS4_N_FF_SHARED * sizeof(float);

    ds4_gpu_tensor *mid = ds4_gpu_tensor_view(g->ffn_mid,
                                               src_shared_off,
                                               shared_bytes);
    ds4_gpu_tensor *after_attn = ds4_gpu_tensor_view(g->after_attn,
                                                     src_embd_off,
                                                     embd_bytes);
    ds4_gpu_tensor *routed = ds4_gpu_tensor_view(g->ffn_out,
                                                 src_embd_off,
                                                 embd_bytes);
    ds4_gpu_tensor *shared = ds4_gpu_tensor_view(g->attn_out,
                                                 0,
                                                 embd_bytes);
    ds4_gpu_tensor *final = ds4_gpu_tensor_view(g->q,
                                                0,
                                                embd_bytes);
    bool ok = mid && after_attn && routed && shared && final;
    if (ok) {
        ok = laguna_graph_matmul_decode_rows(
                 shared,
                 model,
                 layer->ffn_down_shexp,
                 mid,
                 capture->n_rows,
                 true);
    }
    if (ok) {
        ok = ds4_gpu_add3_tensor(final,
                                 after_attn,
                                 routed,
                                 shared,
                                 rows * DS4_N_EMBD) != 0;
    }
    if (ok) {
        ds4_laguna_feature_capture adjusted = *capture;
        adjusted.src_row0 = 0;
        ok = laguna_graph_capture_feature(&adjusted,
                                          final,
                                          DS4_N_LAYER);
    }
    ds4_gpu_tensor_free(final);
    ds4_gpu_tensor_free(shared);
    ds4_gpu_tensor_free(routed);
    ds4_gpu_tensor_free(after_attn);
    ds4_gpu_tensor_free(mid);
    return ok;
}

static bool laguna_graph_forward_batch(
        ds4_laguna_gpu_graph *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        const int            *tokens,
        const ds4_gpu_tensor *gpu_draft_tokens,
        uint32_t              n_tokens,
        uint32_t              pos0,
        float                *logits_out,
        int                  *row_argmax_out,
        const ds4_laguna_feature_capture *capture,
        ds4_session_progress_fn display_progress,
        void                 *display_progress_ud,
        int                   display_total) {
    if (!g || !model || !weights || !tokens || n_tokens == 0 ||
        n_tokens > g->prefill_cap || pos0 > g->ctx_size - n_tokens) {
        return false;
    }
    if (row_argmax_out &&
        (n_tokens > DS4_DFLASH_BLOCK_SIZE ||
         !laguna_graph_ensure_spec_scratch(g))) {
        return false;
    }

    bool ok = true;
#if defined(DS4_ROCM_BUILD) || defined(DS4_NATIVE_CUDA_BUILD)
    const bool gpu_draft_pipeline_ready = true;
#else
    const bool gpu_draft_pipeline_ready =
        ds4_gpu_commands_active() != 0;
#endif
    if (gpu_draft_tokens) {
        /* Copy proposals on-GPU so drafting does not add an intermediate
         * completion and CPU readback. Metal keeps both graphs in one command
         * buffer; CUDA and ROCm get the same ordering from the default stream. */
        if (!gpu_draft_pipeline_ready ||
            tokens[0] < 0 || tokens[0] >= (int)DS4_N_VOCAB ||
            ds4_gpu_tensor_bytes(gpu_draft_tokens) <
                (uint64_t)n_tokens * sizeof(uint32_t)) {
            return false;
        }
        const uint32_t first_token = (uint32_t)tokens[0];
        ok = ds4_gpu_tensor_write(g->tokens,
                                  0,
                                  &first_token,
                                  sizeof(first_token)) != 0;
        if (ok && n_tokens > 1u) {
            ok = ds4_gpu_tensor_copy(
                     g->tokens,
                     sizeof(uint32_t),
                     gpu_draft_tokens,
                     sizeof(uint32_t),
                     (uint64_t)(n_tokens - 1u) * sizeof(uint32_t)) != 0;
        }
    } else {
        uint32_t *token_ids =
            xmalloc((size_t)n_tokens * sizeof(*token_ids));
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (tokens[i] < 0 || tokens[i] >= (int)DS4_N_VOCAB) {
                free(token_ids);
                return false;
            }
            token_ids[i] = (uint32_t)tokens[i];
        }
        ok = ds4_gpu_tensor_write(
                 g->tokens,
                 0,
                 token_ids,
                 (uint64_t)n_tokens * sizeof(*token_ids)) != 0;
        free(token_ids);
    }
    if (!ok) return false;

    laguna_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  pos0,
                                                  n_tokens,
                                                  0,
                                                  display_total,
                                                  true);

    /* Metal 4 Tensor matmuls introduce enough low-bit accumulation drift in
     * Laguna prefill to change sliding-attention routes after a few hundred
     * tokens. The legacy batched kernels are both stable against the Poolside
     * reference and slightly faster on these shapes. */
    ds4_gpu_set_tensor_matmul_suppressed(true);
    /* A long Laguna prefill otherwise lives in one command buffer and cannot
     * report real progress until the whole chunk completes.  When a frontend
     * asks for display progress, finish one layer at a time so each callback
     * reflects GPU work that has actually completed.  Tiny suffixes keep the
     * lower-overhead single-command path. */
    const bool live_progress = display_progress != NULL && n_tokens >= 32u;
    const bool exact_q8_rows = row_argmax_out != NULL;
    if (gpu_draft_tokens) {
        ok = gpu_draft_pipeline_ready;
    } else {
        ok = ds4_gpu_begin_commands() != 0;
    }
    if (ok) {
        ok = ds4_gpu_embed_tokens_quant_tensor(g->cur,
                                                g->tokens,
                                                model->map,
                                                model->size,
                                                weights->token_embd->abs_offset,
                                                weights->token_embd->type,
                                                DS4_N_VOCAB,
                                                n_tokens,
                                                DS4_N_EMBD) != 0;
    }

    uint32_t completed_layers = 0;
    const char *failed_stage = "embedding";
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t n_head = ds4_layer_head_count(il);
        const bool is_swa = ds4_laguna_layer_is_swa(il);
        const uint32_t n_rot = is_swa ? DS4_N_ROT_SWA : DS4_N_ROT;
        const float freq_base = is_swa ?
            DS4_ROPE_FREQ_BASE_SWA : DS4_ROPE_FREQ_BASE;
        const float freq_scale = is_swa ?
            1.0f : 1.0f / DS4_ROPE_SCALE_FACTOR;
        const float ext_factor = is_swa ? 0.0f : 1.0f;
        const float attn_factor = is_swa ? 1.0f : DS4_ROPE_YARN_ATTN_FACTOR;
        const float beta_fast = is_swa ? 0.0f : DS4_ROPE_YARN_BETA_FAST;
        const float beta_slow = is_swa ? 0.0f : DS4_ROPE_YARN_BETA_SLOW;
        const uint32_t rope_ctx = is_swa ?
            (uint32_t)DS4_CONTEXT_LENGTH : (uint32_t)DS4_ROPE_ORIG_CTX;

        failed_stage = "attention norm";
        ok = laguna_graph_capture_feature(capture, g->cur, il);
        if (!ok) {
            failed_stage = "DFlash feature capture";
            break;
        }
        ok = laguna_graph_rms_norm(g->attn_norm,
                                   g->cur,
                                   model,
                                   l->attn_norm,
                                   DS4_N_EMBD,
                                   n_tokens);
        if (ok) {
            failed_stage = "Q/K/V/gate projection";
            bool projections_done = false;
            if (l->attn_q->type == DS4_TENSOR_BF16 &&
                l->attn_k->type == DS4_TENSOR_BF16 &&
                l->attn_v->type == DS4_TENSOR_BF16 &&
                l->attn_gate->type == DS4_TENSOR_BF16) {
                projections_done =
                    ds4_gpu_laguna_qkvg_bf16_tensor(
                        g->q,
                        g->k,
                        g->v,
                        g->gate,
                        model->map,
                        model->size,
                        l->attn_q->abs_offset,
                        l->attn_k->abs_offset,
                        l->attn_v->abs_offset,
                        l->attn_gate->abs_offset,
                        DS4_N_EMBD,
                        n_head * DS4_N_HEAD_DIM,
                        DS4_N_HEAD_KV * DS4_N_HEAD_DIM,
                        n_head,
                        n_tokens,
                        g->attn_norm) != 0;
            }
            ok = projections_done ||
                 (laguna_graph_matmul_decode_rows(
                      g->q,
                      model,
                      l->attn_q,
                      g->attn_norm,
                      n_tokens,
                      exact_q8_rows) &&
                  laguna_graph_matmul_decode_rows(
                      g->k,
                      model,
                      l->attn_k,
                      g->attn_norm,
                      n_tokens,
                      exact_q8_rows) &&
                  laguna_graph_matmul_decode_rows(
                      g->v,
                      model,
                      l->attn_v,
                      g->attn_norm,
                      n_tokens,
                      exact_q8_rows) &&
                  laguna_graph_matmul_decode_rows(
                      g->gate,
                      model,
                      l->attn_gate,
                      g->attn_norm,
                      n_tokens,
                      exact_q8_rows));
        }
        if (ok && exact_q8_rows) {
            failed_stage = "Q/K norm/RoPE";
            if (l->attn_q_norm->type == DS4_TENSOR_BF16) {
                ok = ds4_gpu_laguna_qk_head_rms_norm_rope_bf16_tensor(
                        g->q,
                        g->k,
                        model->map,
                        model->size,
                        l->attn_q_norm->abs_offset,
                        l->attn_k_norm->abs_offset,
                        n_tokens,
                        n_head,
                        DS4_N_HEAD_KV,
                        DS4_N_HEAD_DIM,
                        n_rot,
                        pos0,
                        rope_ctx,
                        freq_base,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        beta_fast,
                        beta_slow,
                        DS4_RMS_EPS) != 0;
            } else {
                ok = ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                        g->q,
                        g->k,
                        model->map,
                        model->size,
                        l->attn_q_norm->abs_offset,
                        l->attn_k_norm->abs_offset,
                        n_tokens,
                        n_head,
                        DS4_N_HEAD_KV,
                        DS4_N_HEAD_DIM,
                        n_rot,
                        pos0,
                        rope_ctx,
                        freq_base,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        beta_fast,
                        beta_slow,
                        DS4_RMS_EPS) != 0;
            }
        } else {
            if (ok) {
                failed_stage = "Q norm/RoPE";
                if (l->attn_q_norm->type == DS4_TENSOR_BF16) {
                    ok = ds4_gpu_laguna_head_rms_norm_rope_bf16_tensor(
                            g->q,
                            model->map,
                            model->size,
                            l->attn_q_norm->abs_offset,
                            n_tokens,
                            n_head,
                            DS4_N_HEAD_DIM,
                            n_rot,
                            pos0,
                            rope_ctx,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            beta_fast,
                            beta_slow,
                            DS4_RMS_EPS) != 0;
                } else {
                    ok = ds4_gpu_laguna_head_rms_norm_rope_tensor(
                            g->q,
                            model->map,
                            model->size,
                            l->attn_q_norm->abs_offset,
                            n_tokens,
                            n_head,
                            DS4_N_HEAD_DIM,
                            n_rot,
                            pos0,
                            rope_ctx,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            beta_fast,
                            beta_slow,
                            DS4_RMS_EPS) != 0;
                }
            }
            if (ok) {
                failed_stage = "K norm/RoPE";
                if (l->attn_k_norm->type == DS4_TENSOR_BF16) {
                    ok = ds4_gpu_laguna_head_rms_norm_rope_bf16_tensor(
                            g->k,
                            model->map,
                            model->size,
                            l->attn_k_norm->abs_offset,
                            n_tokens,
                            DS4_N_HEAD_KV,
                            DS4_N_HEAD_DIM,
                            n_rot,
                            pos0,
                            rope_ctx,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            beta_fast,
                            beta_slow,
                            DS4_RMS_EPS) != 0;
                } else {
                    ok = ds4_gpu_laguna_head_rms_norm_rope_tensor(
                            g->k,
                            model->map,
                            model->size,
                            l->attn_k_norm->abs_offset,
                            n_tokens,
                            DS4_N_HEAD_KV,
                            DS4_N_HEAD_DIM,
                            n_rot,
                            pos0,
                            rope_ctx,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            beta_fast,
                            beta_slow,
                            DS4_RMS_EPS) != 0;
                }
            }
        }
        if (ok) {
            failed_stage = "causal attention";
            /* Ask the backend to replay decode's split-key attention per
             * verifier row.  On Metal that is a long-context throughput choice;
             * on ROCm the batched causal kernel accumulates its softmax in a
             * different order than the decode kernel, so the verifier needs the
             * replay at every context length to accept the same tokens plain
             * decoding would have produced. */
            const int split_decode_rows = row_argmax_out != NULL &&
#ifndef DS4_ROCM_BUILD
                (uint64_t)pos0 + n_tokens > 256u &&
#endif
                true;
            ok = ds4_gpu_laguna_attention_prefill_tensor(
                    g->heads,
                    g->key_cache[il],
                    g->value_cache[il],
                    g->staged_key,
                    g->staged_value,
                    g->q,
                    g->k,
                    g->v,
                    g->gate,
                    pos0,
                    n_tokens,
                    g->cache_cap[il],
                    n_head,
                    DS4_N_HEAD_KV,
                    DS4_N_HEAD_DIM,
                    1.0f / sqrtf((float)DS4_N_HEAD_DIM),
                    split_decode_rows) != 0;
        }
        if (ok) {
            failed_stage = "attention output projection";
            ok = laguna_graph_matmul_decode_rows(g->attn_out,
                                                 model,
                                                 l->attn_output,
                                                 g->heads,
                                                 n_tokens,
                                                 exact_q8_rows);
        }
        if (ok) {
            failed_stage = "attention residual";
            ok = ds4_gpu_add_tensor(g->after_attn,
                                    g->cur,
                                    g->attn_out,
                                    (uint64_t)n_tokens * DS4_N_EMBD) != 0;
        }
        if (ok) {
            failed_stage = "FFN norm";
            ok = laguna_graph_rms_norm(g->ffn_norm,
                                       g->after_attn,
                                       model,
                                       l->ffn_norm,
                                       DS4_N_EMBD,
                                       n_tokens);
        }
        if (ok && il < DS4_N_LEADING_DENSE) {
            failed_stage = "dense FFN gate/up";
            ok = laguna_graph_matmul_decode_rows(g->ffn_gate,
                                                 model,
                                                 l->ffn_gate,
                                                 g->ffn_norm,
                                                 n_tokens,
                                                 exact_q8_rows) &&
                 laguna_graph_matmul_decode_rows(g->ffn_up,
                                                 model,
                                                 l->ffn_up,
                                                 g->ffn_norm,
                                                 n_tokens,
                                                 exact_q8_rows);
            if (ok) {
                failed_stage = "dense FFN SwiGLU";
                ok = ds4_gpu_swiglu_tensor(
                        g->ffn_mid,
                        g->ffn_gate,
                        g->ffn_up,
                        (uint64_t)n_tokens * DS4_N_FF_DENSE,
                        0.0f,
                        1.0f) != 0;
            }
            if (ok) {
                failed_stage = "dense FFN down";
                ok = laguna_graph_matmul_decode_rows(g->ffn_out,
                                                     model,
                                                     l->ffn_down,
                                                     g->ffn_mid,
                                                     n_tokens,
                                                     exact_q8_rows);
            }
            if (ok) {
                failed_stage = "dense FFN residual";
                ok = ds4_gpu_add_tensor(g->next,
                                        g->after_attn,
                                        g->ffn_out,
                                        (uint64_t)n_tokens * DS4_N_EMBD) != 0;
            }
        } else if (ok) {
            if (exact_q8_rows) {
                failed_stage = "decode-row router";
                ok = laguna_graph_router_decode_rows(
                        g, model, l, n_tokens);
            } else {
                failed_stage = "router projection";
                ok = laguna_graph_matmul(g->router_logits,
                                         model,
                                         l->ffn_gate_inp,
                                         g->ffn_norm,
                                         n_tokens);
            }
            if (ok && !exact_q8_rows) {
                failed_stage = "router selection";
                ok = ds4_gpu_glm_router_select_batch_tensor(
                        g->router_selected,
                        g->router_weights,
                        g->router_probs,
                        model->map,
                        model->size,
                        l->ffn_exp_probs_b->abs_offset,
                        g->router_logits,
                        DS4_N_EXPERT,
                        DS4_N_EXPERT_USED,
                        DS4_EXPERT_WEIGHT_SCALE,
                        n_tokens) != 0;
            }

            const uint64_t gate_row_bytes =
                routed_expert_row_bytes(l->ffn_gate_exps);
            const uint64_t up_row_bytes =
                routed_expert_row_bytes(l->ffn_up_exps);
            const uint64_t down_row_bytes =
                routed_expert_row_bytes(l->ffn_down_exps);
            const uint64_t gate_expert_bytes =
                l->ffn_gate_exps->dim[1] * gate_row_bytes;
            const uint64_t up_expert_bytes =
                l->ffn_up_exps->dim[1] * up_row_bytes;
            const uint64_t down_expert_bytes =
                l->ffn_down_exps->dim[1] * down_row_bytes;
            if (ok) {
                failed_stage = "routed experts";
                if (exact_q8_rows) {
                    ok = laguna_graph_routed_moe_decode_rows(
                            g,
                            model,
                            l,
                            il,
                            n_tokens,
                            gate_expert_bytes,
                            gate_row_bytes,
                            up_expert_bytes,
                            up_row_bytes,
                            down_expert_bytes,
                            down_row_bytes);
                } else if (l->ffn_gate_exps->type ==
                           DS4_TENSOR_NVFP4) {
                    const ds4_gpu_laguna_moe_desc routed =
                        laguna_graph_moe_desc(l);
                    ok = ds4_gpu_laguna_routed_moe_tensor(
                            g->ffn_out,
                            g->routed_mid,
                            model->map,
                            model->size,
                            &routed,
                            DS4_N_EMBD,
                            DS4_N_FF_EXP,
                            DS4_N_EMBD,
                            g->router_selected,
                            g->router_weights,
                            DS4_N_EXPERT,
                            DS4_N_EXPERT_USED,
                            g->ffn_norm,
                            n_tokens) != 0;
                } else {
                    ok = ds4_gpu_glm_routed_moe_batch_tensor(
                            g->ffn_out,
                            g->routed_mid,
                            model->map,
                            model->size,
                            l->ffn_gate_exps->abs_offset,
                            l->ffn_up_exps->abs_offset,
                            l->ffn_down_exps->abs_offset,
                            l->ffn_gate_exps->type,
                            l->ffn_up_exps->type,
                            l->ffn_down_exps->type,
                            gate_expert_bytes,
                            gate_row_bytes,
                            up_expert_bytes,
                            up_row_bytes,
                            down_expert_bytes,
                            down_row_bytes,
                            DS4_N_EMBD,
                            DS4_N_FF_EXP,
                            DS4_N_EMBD,
                            g->router_selected,
                            g->router_weights,
                            DS4_N_EXPERT,
                            DS4_N_EXPERT_USED,
                            il,
                            g->ffn_norm,
                            n_tokens,
                            DS4_N_EXPERT_USED * DS4_N_FF_EXP,
                            true) != 0;
                }
            }
            const bool exact_q8_shared =
                exact_q8_rows &&
                l->ffn_gate_shexp->type == DS4_TENSOR_Q8_0 &&
                l->ffn_up_shexp->type == DS4_TENSOR_Q8_0;
            if (ok && exact_q8_shared) {
                failed_stage = "shared expert fused gate/up";
                ok = ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                         g->ffn_gate,
                         g->ffn_up,
                         g->ffn_mid,
                         model->map,
                         model->size,
                         l->ffn_gate_shexp->abs_offset,
                         l->ffn_up_shexp->abs_offset,
                         DS4_N_EMBD,
                         DS4_N_FF_SHARED,
                         g->ffn_norm,
                         n_tokens,
                         0.0f) != 0;
            } else if (ok) {
                failed_stage = "shared expert gate/up";
                ok = laguna_graph_matmul_decode_rows(
                         g->ffn_gate,
                         model,
                         l->ffn_gate_shexp,
                         g->ffn_norm,
                         n_tokens,
                         exact_q8_rows) &&
                     laguna_graph_matmul_decode_rows(
                         g->ffn_up,
                         model,
                         l->ffn_up_shexp,
                         g->ffn_norm,
                         n_tokens,
                         exact_q8_rows);
                if (ok) {
                    failed_stage = "shared expert SwiGLU";
                    ok = ds4_gpu_swiglu_tensor(
                            g->ffn_mid,
                            g->ffn_gate,
                            g->ffn_up,
                            (uint64_t)n_tokens * DS4_N_FF_SHARED,
                            0.0f,
                            1.0f) != 0;
                }
            }
            if (ok) {
                failed_stage = "shared expert down";
                ok = laguna_graph_matmul_decode_rows(
                         g->shared_out,
                         model,
                         l->ffn_down_shexp,
                         g->ffn_mid,
                         n_tokens,
                         exact_q8_rows);
            }
            if (ok) {
                failed_stage = "MoE residual";
                ok = ds4_gpu_add3_tensor(
                        g->next,
                        g->after_attn,
                        g->ffn_out,
                        g->shared_out,
                        (uint64_t)n_tokens * DS4_N_EMBD) != 0;
            }
        }

        if (ok) {
            ds4_gpu_tensor *tmp = g->cur;
            g->cur = g->next;
            g->next = tmp;
            completed_layers = il + 1u;
        }
        if (ok && live_progress) {
            ok = ds4_gpu_end_commands() != 0;
            if (ok) {
                const bool layer_is_all_work =
                    completed_layers == (uint32_t)DS4_N_LAYER &&
                    logits_out == NULL;
                laguna_graph_report_prefill_display_progress(
                        display_progress,
                        display_progress_ud,
                        pos0,
                        n_tokens,
                        completed_layers,
                        display_total,
                        layer_is_all_work);
            }
            if (ok && (completed_layers < (uint32_t)DS4_N_LAYER ||
                       logits_out != NULL)) {
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
    }

    if (!ok) {
        fprintf(stderr,
                "ds4: Laguna batch prefill failed in %s after %u/%u layers\n",
                failed_stage,
                completed_layers,
                (unsigned)DS4_N_LAYER);
    }

    if (ok) {
        ok = laguna_graph_capture_final_feature(
                g,
                model,
                &weights->layer[DS4_N_LAYER - 1u],
                n_tokens,
                capture);
        if (!ok) failed_stage = "DFlash final feature capture";
    }

    if (ok && row_argmax_out) {
        failed_stage = "speculative output norm";
        ok = laguna_graph_rms_norm(g->spec_output_norm,
                                   g->cur,
                                   model,
                                   weights->output_norm,
                                   DS4_N_EMBD,
                                   n_tokens);
        if (ok) {
            failed_stage = "speculative output projection";
            ok = laguna_graph_matmul_decode_rows(g->spec_logits,
                                                 model,
                                                 weights->output,
                                                 g->spec_output_norm,
                                                 n_tokens,
                                                 exact_q8_rows);
        }
        if (ok) {
            failed_stage = "speculative row argmax";
#ifdef DS4_ROCM_BUILD
            ok = ds4_gpu_argmax_rows_tensor(g->spec_argmax,
                                            g->spec_logits,
                                            DS4_N_VOCAB,
                                            n_tokens) != 0;
#else
            ok = ds4_gpu_indexer_topk_tensor(g->spec_argmax,
                                              g->spec_logits,
                                              DS4_N_VOCAB,
                                              n_tokens,
                                              1) != 0;
#endif
        }
    }

    ds4_gpu_tensor *last = NULL;
    if (ok && logits_out) {
        last = ds4_gpu_tensor_view(
                g->cur,
                (uint64_t)(n_tokens - 1u) * DS4_N_EMBD * sizeof(float),
                (uint64_t)DS4_N_EMBD * sizeof(float));
        ok = last != NULL;
        if (ok) {
            ok = laguna_graph_rms_norm(g->output_norm,
                                       last,
                                       model,
                                       weights->output_norm,
                                       DS4_N_EMBD,
                                       1u);
        }
        if (ok) {
            ok = laguna_graph_matmul(g->logits,
                                     model,
                                     weights->output,
                                     g->output_norm,
                                     1);
        }
    }
    /* A speculative cycle appends support-cache injection before completing
     * the shared snapshot/draft/verify command stream. */
    const bool defer_completion = ok && gpu_draft_tokens != NULL;
    if (!defer_completion &&
        ds4_gpu_commands_active() && ds4_gpu_end_commands() == 0) {
        ok = false;
    }
    ds4_gpu_tensor_free(last);
    if (ok && row_argmax_out && !defer_completion) {
        ok = ds4_gpu_tensor_read(g->spec_argmax,
                                 0,
                                 row_argmax_out,
                                 (uint64_t)n_tokens *
                                     sizeof(row_argmax_out[0])) != 0;
    }
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits,
                                 0,
                                 logits_out,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok && (!live_progress || logits_out != NULL)) {
        laguna_graph_report_prefill_display_progress(display_progress,
                                                      display_progress_ud,
                                                      pos0,
                                                      n_tokens,
                                                      DS4_N_LAYER,
                                                      display_total,
                                                      true);
    }
    ds4_gpu_set_tensor_matmul_suppressed(false);
    return ok;
}

static int generate_laguna_metal_argmax(
        const ds4_model   *model,
        const ds4_vocab   *vocab,
        const ds4_weights *weights,
        const token_vec   *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud) {
    if (!prompt || prompt->len <= 0 || prompt->len >= ctx_size) {
        fprintf(stderr, "ds4: Laguna prompt is empty or leaves no context room\n");
        return 1;
    }
    ds4_laguna_gpu_graph g;
    if (!laguna_graph_alloc(&g, (uint32_t)ctx_size)) return 1;
    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    bool ok = true;
    const double prefill_t0 = now_sec();
    for (int i = 0; ok && i < prompt->len;) {
        uint32_t n = (uint32_t)(prompt->len - i);
        if (n > g.prefill_cap) n = g.prefill_cap;
        ok = laguna_graph_forward_batch(&g,
                                        model,
                                        weights,
                                        prompt->v + i,
                                        NULL,
                                        n,
                                        (uint32_t)i,
                                        i + (int)n == prompt->len ?
                                            logits : NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        prompt->len);
        i += (int)n;
        if (progress) progress(progress_ud, "prefill_chunk", i, prompt->len);
    }
    const double prefill_t1 = now_sec();

    int generated = 0;
    uint32_t pos = (uint32_t)prompt->len;
    const double decode_t0 = now_sec();
    for (int i = 0; ok && i < n_predict && pos < (uint32_t)ctx_size; i++) {
        if (getenv("DS4_TRACE_TOP") != NULL) {
            char label[64];
            snprintf(label, sizeof(label), "Laguna step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }
        const int token = sample_argmax(logits, DS4_N_VOCAB);
        if (vocab_token_is_generation_stop(vocab, token)) break;
        if (emit) emit(emit_ud, token);
        generated++;
        if (i + 1 == n_predict || pos + 1u >= (uint32_t)ctx_size) break;
        ok = laguna_graph_forward_token(
            &g, model, weights, token, pos, NULL, logits);
        pos++;
    }
    const double decode_t1 = now_sec();
    if (done) done(emit_ud);
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: Laguna prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_t1 > prefill_t0 ?
                (double)prompt->len / (prefill_t1 - prefill_t0) : 0.0,
            decode_t1 > decode_t0 ?
                (double)generated / (decode_t1 - decode_t0) : 0.0);
    free(logits);
    laguna_graph_free(&g);
    return ok ? 0 : 1;
}
