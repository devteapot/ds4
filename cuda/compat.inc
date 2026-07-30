/* Compatibility surface shared with the canonical Metal/ROCm graph. CUDA
 * either delegates to its equivalent primitive or reports an unavailable
 * optional fast path so the graph can use its established fallback. */
extern "C" int ds4_gpu_commit_and_wait_selected_readback(
        uint64_t event_value, const char *label) {
    (void)event_value;
    return cuda_ok(cudaDeviceSynchronize(),
                   label ? label : "selected readback wait");
}

extern "C" int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    const int ok = ds4_gpu_set_model_fd(fd);
    if (ok) g_model_fd_host_base = model_map;
    return ok;
}

extern "C" int ds4_gpu_pro_q4_expert_table_auto_available(void) {
    return 0;
}

extern "C" int ds4_gpu_preload_q4_expert_tables(
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
        uint32_t n_total_expert) {
    (void)model_map; (void)model_size;
    (void)gate_offset; (void)up_offset; (void)down_offset;
    (void)gate_expert_bytes; (void)down_expert_bytes;
    (void)n_total_expert;
    return 1;
}

extern "C" void ds4_gpu_set_glm_model(bool enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    g_ssd_streaming_mode = enabled ? 1 : 0;
    cuda_stream_selected_cache_invalidate();
    if (!g_ssd_streaming_mode) cuda_stream_selected_cache_release();
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    (void)experts;
}

extern "C" void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    (void)bytes;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    return 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return g_stream_selected_cache.valid ?
        g_stream_selected_cache.compact_count : 0;
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
}

extern "C" void ds4_gpu_stream_expert_cache_release_resident(void) {
    cuda_stream_selected_cache_release();
}

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *selected_ids,
        uint32_t n_selected) {
    (void)table; (void)selected_ids; (void)n_selected;
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *selected_ids,
        uint32_t n_tokens,
        uint32_t n_selected) {
    if (n_tokens == 0 || n_selected == 0 ||
        (uint64_t)n_tokens * n_selected > UINT32_MAX) {
        return 0;
    }
    return cuda_stream_selected_cache_begin_load(
            table, selected_ids, n_tokens * n_selected);
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *expert_ids,
        const uint32_t *expert_priorities,
        uint32_t n_experts) {
    (void)table; (void)expert_ids; (void)expert_priorities; (void)n_experts;
    return 1;
}

extern "C" int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t n_vocab) {
    return ds4_gpu_indexer_topk_tensor(out_idx, logits, n_vocab, 1u, 1u);
}

extern "C" int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_vocab,
        uint32_t token,
        uint32_t n_embd) {
    return ds4_gpu_embed_token_quant_tensor(out, model_map, model_size,
                                            weight_offset, 8u, n_vocab,
                                            token, n_embd);
}

extern "C" int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *tokens,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd) {
    return ds4_gpu_embed_tokens_quant_tensor(out, tokens, model_map,
                                             model_size, weight_offset, 8u,
                                             n_vocab, n_tokens, n_embd);
}

extern "C" int ds4_gpu_glm_k_b_project_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *kv_norm,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_tokens,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t n_head) {
    return ds4_gpu_glm_k_b_project_typed_tensor(
            out, kv_norm, model_map, model_size, weight_offset, 8u,
            n_tokens, kv_lora_dim, qk_nope, n_head);
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t full_in_dim,
        uint64_t k_off,
        uint64_t k_cnt,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (!x || x_elem_off > x->bytes / sizeof(float) ||
        k_cnt > x->bytes / sizeof(float) - x_elem_off) {
        return 0;
    }
    ds4_gpu_tensor x_slice = *x;
    x_slice.ptr = (char *)x->ptr + x_elem_off * sizeof(float);
    x_slice.bytes = k_cnt * sizeof(float);
    x_slice.owner = 0;
    return ds4_gpu_matmul_q8_0_kslice_rows_tensor(
            out, model_map, model_size, weight_offset,
            full_in_dim, out_dim, k_off, k_cnt, &x_slice, 1u);
}

extern "C" int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t weight_type,
        uint64_t full_in_dim,
        uint64_t k_off,
        uint64_t k_cnt,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_matmul_q8_0_kslice_tensor(
            out, model_map, model_size, weight_offset,
            full_in_dim, k_off, k_cnt, out_dim, x, x_elem_off);
}

extern "C" int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor *out_h,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    (void)out_h; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}

extern "C" int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *q_half,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float eps) {
    (void)out; (void)q_half; (void)model_map; (void)model_size;
    (void)weight_offset; (void)in_dim; (void)out_dim; (void)x;
    (void)n_tok; (void)n_head; (void)head_dim; (void)n_rot; (void)pos0;
    (void)n_ctx_orig; (void)inverse; (void)freq_base; (void)freq_scale;
    (void)ext_factor; (void)attn_factor; (void)beta_fast; (void)beta_slow;
    (void)eps;
    return 0;
}

extern "C" int ds4_gpu_attention_prefill_raw_heads_range_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t q_row0, uint32_t n_q,
        uint32_t n_kv, uint32_t window, uint32_t n_head,
        uint32_t head_dim) {
    (void)heads; (void)model_map; (void)model_size; (void)sinks_offset;
    (void)q; (void)raw_kv; (void)q_row0; (void)n_q; (void)n_kv;
    (void)window; (void)n_head; (void)head_dim;
    return 0;
}

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, uint32_t q_row0, uint32_t n_q,
        uint32_t n_tokens, uint32_t n_comp, uint32_t window,
        uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    (void)heads; (void)model_map; (void)model_size; (void)sinks_offset;
    (void)q; (void)raw_kv; (void)comp_kv; (void)comp_kv_f16;
    (void)q_row0; (void)n_q; (void)n_tokens; (void)n_comp; (void)window;
    (void)ratio; (void)n_head; (void)head_dim;
    return 0;
}

extern "C" int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor *out_h, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups,
        uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    (void)out_h; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim; (void)rank;
    (void)n_groups; (void)out_dim; (void)heads; (void)n_tokens;
    return 0;
}

extern "C" int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        ds4_gpu_tensor *group_tmp, ds4_gpu_tensor *low_tmp,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset, uint32_t out_b_type,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups,
        uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    (void)out; (void)low; (void)group_tmp; (void)low_tmp;
    (void)model_map; (void)model_size; (void)out_a_offset;
    (void)out_b_offset; (void)out_b_type; (void)group_dim; (void)rank;
    (void)n_groups; (void)out_dim; (void)heads; (void)n_tokens;
    return 0;
}

extern "C" int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t group0, uint32_t group_cnt,
        const ds4_gpu_tensor *heads) {
    (void)low; (void)model_map; (void)model_size; (void)out_a_offset;
    (void)group_dim; (void)rank; (void)group0; (void)group_cnt;
    (void)heads;
    return 0;
}

extern "C" int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out_h; (void)residual_hc; (void)split;
    (void)n_embd; (void)n_hc;
    return 0;
}

extern "C" int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out; (void)block_add_h;
    (void)residual_hc; (void)split; (void)n_embd; (void)n_hc;
    return 0;
}

extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    (void)suspend;
}

extern "C" void ds4_gpu_tp_keepalive_pause(int paused) {
    (void)paused;
}

extern "C" void ds4_gpu_model_residency_skip(int skip) {
    (void)skip;
}

extern "C" uint64_t ds4_gpu_tp_big_gate_kick(
        uint32_t layer, uint32_t rows, const ds4_gpu_tensor *out_t,
        ds4_gpu_tensor *in_t, uint64_t bytes) {
    (void)layer; (void)rows; (void)out_t; (void)in_t; (void)bytes;
    return 0;
}

extern "C" int ds4_gpu_tp_big_gate_wait(uint64_t seq) {
    (void)seq;
    return 0;
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    (void)layer; (void)rows;
    return 0;
}

extern "C" int ds4_gpu_matmul_q6_K_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}

extern "C" int ds4_gpu_laguna_routed_shared_moe_one_tensor(
        ds4_gpu_tensor *routed_out, ds4_gpu_tensor *routed_mid,
        ds4_gpu_tensor *shared_out, ds4_gpu_tensor *shared_mid,
        const void *model_map, uint64_t model_size,
        const ds4_gpu_laguna_moe_desc *routed,
        const ds4_gpu_laguna_moe_desc *shared,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert,
        const ds4_gpu_tensor *shared_selected,
        const ds4_gpu_tensor *shared_weight, const ds4_gpu_tensor *x) {
    (void)routed_out; (void)routed_mid; (void)shared_out; (void)shared_mid;
    (void)model_map; (void)model_size; (void)routed; (void)shared;
    (void)expert_in_dim; (void)expert_mid_dim; (void)out_dim;
    (void)selected; (void)weights; (void)n_total_expert; (void)n_expert;
    (void)shared_selected; (void)shared_weight; (void)x;
    return 0;
}

extern "C" int ds4_gpu_laguna_qkvg_f16_tensor(
        ds4_gpu_tensor *q, ds4_gpu_tensor *k, ds4_gpu_tensor *v,
        ds4_gpu_tensor *gate, const void *model_map, uint64_t model_size,
        uint64_t q_weight_offset, uint64_t k_weight_offset,
        uint64_t v_weight_offset, uint64_t gate_weight_offset,
        uint32_t in_dim, uint32_t q_dim, uint32_t kv_dim,
        uint32_t gate_dim, const ds4_gpu_tensor *x) {
    (void)q; (void)k; (void)v; (void)gate; (void)model_map; (void)model_size;
    (void)q_weight_offset; (void)k_weight_offset; (void)v_weight_offset;
    (void)gate_weight_offset; (void)in_dim; (void)q_dim; (void)kv_dim;
    (void)gate_dim; (void)x;
    return 0;
}

extern "C" int ds4_gpu_laguna_attn_output_residual_f16_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *residual) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)residual;
    return 0;
}

#pragma GCC diagnostic pop
