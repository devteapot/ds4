void ds4_gpu_set_glm_mtp_verify_mode(bool enabled) {
    (void)enabled;
}

/* Official Laguna BF16/NVFP4 checkpoints are CUDA-only. */
int ds4_gpu_laguna_routed_moe_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *mid,
        const void *model_map, uint64_t model_size,
        const ds4_gpu_laguna_moe_desc *routed,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert,
        const ds4_gpu_tensor *x, uint32_t n_tokens) {
    (void)out; (void)mid; (void)model_map; (void)model_size; (void)routed;
    (void)expert_in_dim; (void)expert_mid_dim; (void)out_dim;
    (void)selected; (void)weights; (void)n_total_expert; (void)n_expert;
    (void)x; (void)n_tokens;
    return 0;
}

int ds4_gpu_matmul_bf16_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}

int ds4_gpu_rms_norm_bf16_weight_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    (void)out; (void)x; (void)model_map; (void)model_size;
    (void)weight_offset; (void)n; (void)rows; (void)eps;
    return 0;
}

int ds4_gpu_laguna_qkvg_bf16_tensor(
        ds4_gpu_tensor *q, ds4_gpu_tensor *k, ds4_gpu_tensor *v,
        ds4_gpu_tensor *gate, const void *model_map, uint64_t model_size,
        uint64_t qo, uint64_t ko, uint64_t vo, uint64_t go,
        uint32_t in_dim, uint32_t q_dim, uint32_t kv_dim,
        uint32_t gate_dim, uint32_t n_tokens,
        const ds4_gpu_tensor *x) {
    (void)q; (void)k; (void)v; (void)gate; (void)model_map;
    (void)model_size; (void)qo; (void)ko; (void)vo; (void)go;
    (void)in_dim; (void)q_dim; (void)kv_dim; (void)gate_dim;
    (void)n_tokens; (void)x;
    return 0;
}

int ds4_gpu_dflash_qkvg_bf16_tensor(
        ds4_gpu_tensor *q, ds4_gpu_tensor *k, ds4_gpu_tensor *v,
        ds4_gpu_tensor *gate, ds4_gpu_tensor *qkv,
        const void *model_map, uint64_t model_size,
        uint64_t qkv_offset, uint64_t gate_offset,
        uint32_t n_tokens, const ds4_gpu_tensor *x) {
    (void)q; (void)k; (void)v; (void)gate; (void)qkv;
    (void)model_map; (void)model_size; (void)qkv_offset; (void)gate_offset;
    (void)n_tokens; (void)x;
    return 0;
}

int ds4_gpu_dflash_aux_norm_bf16_tensor(
        ds4_gpu_tensor *features, const void *model_map,
        uint64_t model_size, uint64_t weight_offset,
        uint32_t n_rows, uint32_t n_embd, uint32_t n_aux, float eps) {
    (void)features; (void)model_map; (void)model_size; (void)weight_offset;
    (void)n_rows; (void)n_embd; (void)n_aux; (void)eps;
    return 0;
}

int ds4_gpu_laguna_head_rms_norm_rope_bf16_tensor(
        ds4_gpu_tensor *x, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t n_tokens, uint32_t n_head,
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow, float eps) {
    (void)x; (void)model_map; (void)model_size; (void)weight_offset;
    (void)n_tokens; (void)n_head; (void)head_dim; (void)n_rot;
    (void)pos0; (void)n_ctx_orig; (void)freq_base; (void)freq_scale;
    (void)ext_factor; (void)attn_factor; (void)beta_fast;
    (void)beta_slow; (void)eps;
    return 0;
}

int ds4_gpu_laguna_qk_head_rms_norm_rope_bf16_tensor(
        ds4_gpu_tensor *q, ds4_gpu_tensor *k, const void *model_map,
        uint64_t model_size, uint64_t qo, uint64_t ko,
        uint32_t n_tokens, uint32_t n_q_head, uint32_t n_k_head,
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow, float eps) {
    (void)q; (void)k; (void)model_map; (void)model_size; (void)qo; (void)ko;
    (void)n_tokens; (void)n_q_head; (void)n_k_head; (void)head_dim;
    (void)n_rot; (void)pos0; (void)n_ctx_orig; (void)freq_base;
    (void)freq_scale; (void)ext_factor; (void)attn_factor;
    (void)beta_fast; (void)beta_slow; (void)eps;
    return 0;
}
