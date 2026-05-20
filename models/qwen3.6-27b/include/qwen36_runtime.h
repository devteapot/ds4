#ifndef QWEN36_RUNTIME_H
#define QWEN36_RUNTIME_H

#include "rt_runtime.h"

#include <stdbool.h>
#include <stdint.h>

#define QWEN36_RUNTIME_FAMILY "qwen3.6-27b"
#define QWEN36_RUNTIME_HF_REPO "Qwen/Qwen3.6-27B"

typedef enum {
    QWEN36_THINK_AUTO = 0,
    QWEN36_THINK_DISABLED,
    QWEN36_THINK_ENABLED,
} qwen36_think_mode;

typedef struct {
    const char *mmproj_path;
    const char *mtp_path;
    bool language_model_only;
    bool enable_mtp;
    int mtp_draft_tokens;
    float mtp_margin;
} qwen36_runtime_engine_options;

typedef struct {
    qwen36_think_mode think_mode;
    bool preserve_thinking;
} qwen36_runtime_chat_options;

typedef struct {
    uint64_t proposed;
    uint64_t accepted;
    uint64_t rejected;
    uint64_t skipped;
    int draft_len;
    int draft_pos;
} qwen36_runtime_mtp_stats;

typedef struct {
    uint32_t n_tokens;
    uint32_t hidden_size;
    float *data;
} qwen36_runtime_vision_embedding;

typedef struct {
    bool loaded;
    uint32_t image_size;
    uint32_t patch_size;
    uint32_t spatial_merge_size;
    uint64_t min_pixels;
    uint64_t max_pixels;
    uint32_t hidden_size;
    int image_pad_token;
    int video_pad_token;
} qwen36_runtime_vision_config;

typedef struct {
    rt_backend backend;
    bool metal_loaded;
    uint64_t metal_matvec_calls;
    uint64_t metal_matvec_fallbacks;
    uint64_t metal_rms_norm_calls;
    uint64_t metal_rms_norm_fallbacks;
    uint64_t metal_silu_mul_calls;
    uint64_t metal_silu_mul_fallbacks;
    uint64_t metal_l2_norm_calls;
    uint64_t metal_l2_norm_fallbacks;
    uint64_t metal_gated_delta_calls;
    uint64_t metal_gated_delta_fallbacks;
    uint64_t metal_full_attention_calls;
    uint64_t metal_full_attention_fallbacks;
} qwen36_runtime_backend_stats;

typedef struct {
    int token_pos;
    uint32_t n_tokens;
    uint32_t hidden_size;
    const float *data;
} qwen36_runtime_embedding_span;

typedef struct {
    const char *hf_repo;
    const char *display_name;
    uint32_t parameter_count_b;
    uint32_t vocab_size;
    uint32_t max_context;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t n_layers;
    uint32_t full_attention_interval;
    uint32_t attention_heads;
    uint32_t attention_kv_heads;
    uint32_t attention_head_dim;
    uint32_t linear_qk_heads;
    uint32_t linear_v_heads;
    uint32_t linear_head_dim;
    bool dense_weights;
    bool multimodal_checkpoint;
} qwen36_model_spec;

const qwen36_model_spec *qwen36_runtime_model_spec(void);
const rt_model_ops *qwen36_runtime_ops(void);
int qwen36_runtime_register(void);
bool qwen36_runtime_session_mtp_stats(const rt_session *session,
                                      qwen36_runtime_mtp_stats *out);
bool qwen36_runtime_backend_stats_get(const rt_engine *engine,
                                      qwen36_runtime_backend_stats *out);
bool qwen36_runtime_vision_config_get(const rt_engine *engine,
                                      qwen36_runtime_vision_config *out);
void qwen36_runtime_vision_embedding_free(qwen36_runtime_vision_embedding *embedding);
/* `patches` is a row-major grid of RGB patch tensors:
 * grid_h * grid_w * 3 * patch_size * patch_size float32 values. */
int qwen36_runtime_embed_image_patches_f32(rt_engine *engine,
                                           const float *patches,
                                           uint32_t grid_h,
                                           uint32_t grid_w,
                                           qwen36_runtime_vision_embedding *out,
                                           char *err,
                                           size_t errlen);
int qwen36_runtime_session_sync_embeddings(rt_session *session,
                                           const rt_tokens *prompt,
                                           const qwen36_runtime_embedding_span *spans,
                                           size_t n_spans,
                                           char *err,
                                           size_t errlen);
bool qwen36_runtime_session_read_last_hidden(const rt_session *session,
                                             float *out,
                                             uint32_t cap);

#endif
