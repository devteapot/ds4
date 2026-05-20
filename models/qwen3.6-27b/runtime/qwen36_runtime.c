#include "qwen36_runtime.h"
#include "qwen36_metal.h"

#include <ctype.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const qwen36_model_spec QWEN36_MODEL_SPEC = {
    .hf_repo = QWEN36_RUNTIME_HF_REPO,
    .display_name = "Qwen3.6-27B",
    .parameter_count_b = 27,
    .vocab_size = 248320,
    .max_context = 262144,
    .hidden_size = 5120,
    .intermediate_size = 17408,
    .n_layers = 64,
    .full_attention_interval = 4,
    .attention_heads = 24,
    .attention_kv_heads = 4,
    .attention_head_dim = 256,
    .linear_qk_heads = 16,
    .linear_v_heads = 48,
    .linear_head_dim = 128,
    .dense_weights = true,
    .multimodal_checkpoint = true,
};

#define QWEN36_MAX_LAYERS 72u
#define QWEN36_MAX_MTP_LAYERS 8u
#define QWEN36_MMPROJ_MAX_LAYERS 128u
#define QWEN36_MAX_TENSOR_NAME 96u
#define QWEN36_DEFAULT_SSM_D_CONV 4u

typedef struct {
    const char *ptr;
    uint64_t len;
} qwen36_str;

typedef struct {
    qwen36_str key;
    int value;
    bool used;
} qwen36_table_entry;

typedef struct {
    qwen36_table_entry *entry;
    uint64_t cap;
    uint64_t used;
} qwen36_str_i32_table;

typedef struct {
    qwen36_str *token;
    int n_vocab;
    int eos_id;
    int im_start_id;
    int im_end_id;
    int think_start_id;
    int think_end_id;
    int vision_start_id;
    int vision_end_id;
    int image_pad_id;
    int video_pad_id;
    qwen36_str_i32_table token_to_id;
    qwen36_str_i32_table merge_rank;
} qwen36_vocab;

typedef struct {
    const rt_gguf_tensor *attn_norm;
    const rt_gguf_tensor *attn_post_norm;
    const rt_gguf_tensor *ffn_gate;
    const rt_gguf_tensor *ffn_down;
    const rt_gguf_tensor *ffn_up;

    const rt_gguf_tensor *attn_q;
    const rt_gguf_tensor *attn_k;
    const rt_gguf_tensor *attn_v;
    const rt_gguf_tensor *attn_output;
    const rt_gguf_tensor *attn_q_norm;
    const rt_gguf_tensor *attn_k_norm;

    const rt_gguf_tensor *attn_qkv;
    const rt_gguf_tensor *attn_gate;
    const rt_gguf_tensor *ssm_conv1d;
    const rt_gguf_tensor *ssm_dt;
    const rt_gguf_tensor *ssm_a;
    const rt_gguf_tensor *ssm_beta;
    const rt_gguf_tensor *ssm_alpha;
    const rt_gguf_tensor *ssm_norm;
    const rt_gguf_tensor *ssm_out;

    bool recurrent;
} qwen36_layer_tensors;

typedef struct {
    qwen36_layer_tensors block;
    const rt_gguf_file *source;
    const rt_gguf_tensor *eh_proj;
    const rt_gguf_tensor *shared_head_norm;
    const rt_gguf_tensor *enorm;
    const rt_gguf_tensor *hnorm;
    uint32_t layer_index;
} qwen36_mtp_layer_tensors;

typedef struct {
    const rt_gguf_tensor *tok_embd;
    const rt_gguf_tensor *output_norm;
    const rt_gguf_tensor *output;
    qwen36_layer_tensors layer[QWEN36_MAX_LAYERS];
    qwen36_mtp_layer_tensors mtp[QWEN36_MAX_MTP_LAYERS];
    uint32_t n_layers_total;
    uint32_t n_layers_main;
    uint32_t nextn_predict_layers;
    uint32_t n_mtp_layers_bound;
    uint32_t ssm_d_conv;
    uint32_t ssm_d_inner;
    uint32_t ssm_d_state;
    uint32_t ssm_dt_rank;
    uint32_t ssm_n_group;
    uint32_t n_bound;
    uint32_t n_mtp_bound;
} qwen36_weights;

typedef struct {
    const rt_gguf_tensor *ln1;
    const rt_gguf_tensor *ln2;
    const rt_gguf_tensor *attn_qkv;
    const rt_gguf_tensor *attn_q;
    const rt_gguf_tensor *attn_k;
    const rt_gguf_tensor *attn_v;
    const rt_gguf_tensor *attn_out;
    const rt_gguf_tensor *ffn_up;
    const rt_gguf_tensor *ffn_down;
    const rt_gguf_tensor *ffn_gate;
} qwen36_mmproj_layer_tensors;

typedef struct {
    bool loaded;
    char projector_type[32];
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t projection_dim;
    uint32_t image_size;
    uint32_t patch_size;
    uint32_t spatial_merge_size;
    uint64_t min_pixels;
    uint64_t max_pixels;
    uint32_t n_bound;
    const rt_gguf_tensor *patch_embd;
    const rt_gguf_tensor *patch_bias;
    const rt_gguf_tensor *mm_0_w;
    const rt_gguf_tensor *mm_0_b;
    const rt_gguf_tensor *mm_1_w;
    const rt_gguf_tensor *mm_1_b;
    qwen36_mmproj_layer_tensors layer[QWEN36_MMPROJ_MAX_LAYERS];
} qwen36_mmproj;

typedef struct {
    rt_gguf_file *model;
    rt_gguf_file *mmproj;
    rt_gguf_file *mtp_model;
    qwen36_vocab vocab;
    qwen36_weights weights;
    qwen36_mmproj mmproj_info;
    const qwen36_model_spec *spec;
    rt_backend backend;
    qwen36_metal_backend *metal;
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
    bool mtp_enabled;
    int mtp_draft_tokens;
    float mtp_margin;
} qwen36_engine;

typedef struct {
    qwen36_engine *engine;
    int ctx_size;
    rt_tokens tokens;
    float *logits;
    bool logits_valid;
    bool state_valid;
    float *last_hidden;
    bool last_hidden_valid;
    int mtp_draft[QWEN36_MAX_MTP_LAYERS];
    float mtp_draft_margin[QWEN36_MAX_MTP_LAYERS];
    int mtp_draft_len;
    int mtp_draft_pos;
    uint64_t mtp_draft_proposed;
    uint64_t mtp_draft_accepted;
    uint64_t mtp_draft_rejected;
    uint64_t mtp_draft_skipped;
    uint8_t *full_kv_state;
    uint64_t full_kv_state_bytes;
    uint8_t *recurrent_state;
    uint64_t recurrent_state_bytes;
    uint8_t *conv_state;
    uint64_t conv_state_bytes;
    rt_session_progress_fn progress_fn;
    void *progress_ud;
} qwen36_session;

typedef struct {
    int id;
    float logit;
    double prob;
} qwen36_sample_candidate;

bool qwen36_runtime_session_mtp_stats(const rt_session *session,
                                      qwen36_runtime_mtp_stats *out) {
    if (!session || !session->engine || !session->impl || !out ||
        session->engine->ops != qwen36_runtime_ops()) {
        return false;
    }
    const qwen36_session *s = session->impl;
    out->proposed = s->mtp_draft_proposed;
    out->accepted = s->mtp_draft_accepted;
    out->rejected = s->mtp_draft_rejected;
    out->skipped = s->mtp_draft_skipped;
    out->draft_len = s->mtp_draft_len;
    out->draft_pos = s->mtp_draft_pos;
    return true;
}

bool qwen36_runtime_vision_config_get(const rt_engine *engine,
                                      qwen36_runtime_vision_config *out) {
    if (!engine || !engine->impl || !out ||
        engine->ops != qwen36_runtime_ops()) {
        return false;
    }
    const qwen36_engine *e = engine->impl;
    const qwen36_mmproj *mm = &e->mmproj_info;
    memset(out, 0, sizeof(*out));
    out->loaded = mm->loaded;
    out->image_size = mm->image_size;
    out->patch_size = mm->patch_size;
    out->spatial_merge_size = mm->spatial_merge_size;
    out->min_pixels = mm->min_pixels;
    out->max_pixels = mm->max_pixels;
    out->hidden_size = e->spec->hidden_size;
    out->image_pad_token = e->vocab.image_pad_id;
    out->video_pad_token = e->vocab.video_pad_id;
    return true;
}

bool qwen36_runtime_backend_stats_get(const rt_engine *engine,
                                      qwen36_runtime_backend_stats *out) {
    if (!engine || !engine->impl || !out ||
        engine->ops != qwen36_runtime_ops()) {
        return false;
    }
    const qwen36_engine *e = engine->impl;
    memset(out, 0, sizeof(*out));
    out->backend = e->backend;
    out->metal_loaded = e->metal != NULL;
    out->metal_matvec_calls = e->metal_matvec_calls;
    out->metal_matvec_fallbacks = e->metal_matvec_fallbacks;
    out->metal_rms_norm_calls = e->metal_rms_norm_calls;
    out->metal_rms_norm_fallbacks = e->metal_rms_norm_fallbacks;
    out->metal_silu_mul_calls = e->metal_silu_mul_calls;
    out->metal_silu_mul_fallbacks = e->metal_silu_mul_fallbacks;
    out->metal_l2_norm_calls = e->metal_l2_norm_calls;
    out->metal_l2_norm_fallbacks = e->metal_l2_norm_fallbacks;
    out->metal_gated_delta_calls = e->metal_gated_delta_calls;
    out->metal_gated_delta_fallbacks = e->metal_gated_delta_fallbacks;
    out->metal_full_attention_calls = e->metal_full_attention_calls;
    out->metal_full_attention_fallbacks = e->metal_full_attention_fallbacks;
    return true;
}

typedef struct {
    float *hidden;
    float *norm;
    float *mix;
    float *mix2;
    float *ffn_gate;
    float *ffn_up;
    float *ffn_act;
    float *attn;
    float *k;
    float *v;
    float *qkv;
    float *qkv_conv;
    float *z;
    float *beta;
    float *alpha;
    float *core;
    float *scores;
} qwen36_cpu_scratch;

const qwen36_model_spec *qwen36_runtime_model_spec(void) {
    return &QWEN36_MODEL_SPEC;
}

void qwen36_runtime_vision_embedding_free(qwen36_runtime_vision_embedding *embedding) {
    if (!embedding) return;
    free(embedding->data);
    memset(embedding, 0, sizeof(*embedding));
}

static void *qwen36_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) abort();
    return p;
}

static void *qwen36_xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) abort();
    return p;
}

static void *qwen36_xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) abort();
    return p;
}

static void qwen36_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static bool qwen36_str_eq(qwen36_str a, qwen36_str b) {
    return a.len == b.len && (a.len == 0 || memcmp(a.ptr, b.ptr, (size_t)a.len) == 0);
}

static uint64_t qwen36_hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t qwen36_next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n && p <= (UINT64_MAX >> 1)) p <<= 1;
    return p < n ? 0 : p;
}

static bool qwen36_table_init(qwen36_str_i32_table *t, uint64_t expected) {
    memset(t, 0, sizeof(*t));
    if (expected > (UINT64_MAX - 16) / 2) return false;
    t->cap = qwen36_next_pow2(expected * 2 + 16);
    if (t->cap == 0 || t->cap > SIZE_MAX / sizeof(t->entry[0])) return false;
    t->entry = qwen36_xcalloc((size_t)t->cap, sizeof(t->entry[0]));
    return true;
}

static void qwen36_table_free(qwen36_str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void qwen36_table_put(qwen36_str_i32_table *t, qwen36_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = qwen36_hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (qwen36_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

static bool qwen36_table_get(const qwen36_str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (!t || t->cap == 0) return false;

    uint64_t mask = t->cap - 1;
    uint64_t i = qwen36_hash_bytes(ptr, len) & mask;
    while (t->entry[i].used) {
        qwen36_str key = t->entry[i].key;
        if (key.len == len && (len == 0 || memcmp(key.ptr, ptr, (size_t)len) == 0)) {
            if (value) *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

static int qwen36_vocab_lookup(const qwen36_vocab *vocab, const char *text) {
    int token = -1;
    if (text && qwen36_table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        return token;
    }
    return -1;
}

static int qwen36_vocab_lookup_len(const qwen36_vocab *vocab, const char *text, uint64_t len) {
    int token = -1;
    return qwen36_table_get(&vocab->token_to_id, text, len, &token) ? token : -1;
}

static bool qwen36_metadata_token_id(const rt_gguf_metadata *meta, const char *key, int *out) {
    uint32_t value = 0;
    if (!rt_gguf_metadata_get_u32(meta, key, &value) || value > INT32_MAX) return false;
    if (out) *out = (int)value;
    return true;
}

static bool qwen36_vocab_load(qwen36_vocab *vocab, const rt_gguf_file *file) {
    memset(vocab, 0, sizeof(*vocab));
    vocab->eos_id = -1;
    vocab->im_start_id = -1;
    vocab->im_end_id = -1;
    vocab->think_start_id = -1;
    vocab->think_end_id = -1;
    vocab->vision_start_id = -1;
    vocab->vision_end_id = -1;
    vocab->image_pad_id = -1;
    vocab->video_pad_id = -1;

    rt_gguf_array_cursor tokens = {0};
    if (!rt_gguf_file_array_cursor(file, "tokenizer.ggml.tokens", &tokens) ||
        tokens.array.type != RT_GGUF_VALUE_STRING ||
        tokens.array.len > INT32_MAX ||
        tokens.array.len > SIZE_MAX / sizeof(vocab->token[0])) {
        return false;
    }

    vocab->n_vocab = (int)tokens.array.len;
    vocab->token = qwen36_xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    if (!qwen36_table_init(&vocab->token_to_id, tokens.array.len)) return false;

    for (int i = 0; i < vocab->n_vocab; i++) {
        const char *ptr = NULL;
        size_t len = 0;
        if (!rt_gguf_array_cursor_next_string(&tokens, &ptr, &len)) return false;
        vocab->token[i] = (qwen36_str){ ptr, (uint64_t)len };
        qwen36_table_put(&vocab->token_to_id, vocab->token[i], i);
    }

    rt_gguf_array_cursor merges = {0};
    if (rt_gguf_file_array_cursor(file, "tokenizer.ggml.merges", &merges) &&
        merges.array.type == RT_GGUF_VALUE_STRING) {
        if (!qwen36_table_init(&vocab->merge_rank, merges.array.len)) return false;
        for (uint64_t i = 0; i < merges.array.len; i++) {
            const char *ptr = NULL;
            size_t len = 0;
            if (!rt_gguf_array_cursor_next_string(&merges, &ptr, &len)) return false;
            qwen36_table_put(&vocab->merge_rank, (qwen36_str){ ptr, (uint64_t)len }, (int)i);
        }
    } else if (!qwen36_table_init(&vocab->merge_rank, 0)) {
        return false;
    }

    const rt_gguf_metadata *meta = rt_gguf_file_metadata(file);
    if (!qwen36_metadata_token_id(meta, "tokenizer.ggml.eos_token_id", &vocab->eos_id)) {
        vocab->eos_id = qwen36_vocab_lookup(vocab, "<|endoftext|>");
    }
    vocab->im_start_id = qwen36_vocab_lookup(vocab, "<|im_start|>");
    vocab->im_end_id = qwen36_vocab_lookup(vocab, "<|im_end|>");
    vocab->think_start_id = qwen36_vocab_lookup(vocab, "<think>");
    vocab->think_end_id = qwen36_vocab_lookup(vocab, "</think>");
    vocab->vision_start_id = qwen36_vocab_lookup(vocab, "<|vision_start|>");
    vocab->vision_end_id = qwen36_vocab_lookup(vocab, "<|vision_end|>");
    vocab->image_pad_id = qwen36_vocab_lookup(vocab, "<|image_pad|>");
    vocab->video_pad_id = qwen36_vocab_lookup(vocab, "<|video_pad|>");

    return vocab->eos_id >= 0 &&
           vocab->im_start_id >= 0 &&
           vocab->im_end_id >= 0 &&
           vocab->think_start_id >= 0 &&
           vocab->think_end_id >= 0;
}

static void qwen36_vocab_free(qwen36_vocab *vocab) {
    if (!vocab) return;
    free(vocab->token);
    qwen36_table_free(&vocab->token_to_id);
    qwen36_table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

static bool qwen36_contains_ci(const char *haystack, const char *needle) {
    if (!haystack || !needle || !needle[0]) return false;
    size_t nlen = strlen(needle);
    for (const char *p = haystack; *p; p++) {
        size_t i = 0;
        while (i < nlen &&
               p[i] &&
               tolower((unsigned char)p[i]) == tolower((unsigned char)needle[i])) {
            i++;
        }
        if (i == nlen) return true;
    }
    return false;
}

static bool qwen36_probe_filename(const char *model_path) {
    return qwen36_contains_ci(model_path, "qwen3.6-27b") ||
           qwen36_contains_ci(model_path, "qwen3-6-27b") ||
           qwen36_contains_ci(model_path, "qwen3_6-27b") ||
           qwen36_contains_ci(model_path, "qwen36-27b");
}

static bool qwen36_arch_name_supported(const char *arch) {
    return arch &&
           (!strcmp(arch, "qwen35") ||
            !strcmp(arch, "qwen3_5") ||
            !strcmp(arch, "qwen3_5_text"));
}

static bool qwen36_metadata_get_prefixed_u32(
        const rt_gguf_metadata *meta,
        const char *field,
        uint32_t *out) {
    static const char *prefixes[] = {
        "qwen35",
        "qwen3_5",
        "qwen3_5_text",
    };
    char key[128];

    for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) {
        int n = snprintf(key, sizeof(key), "%s.%s", prefixes[i], field);
        if (n > 0 && (size_t)n < sizeof(key) &&
            rt_gguf_metadata_get_u32(meta, key, out)) {
            return true;
        }
    }
    return false;
}

static uint32_t qwen36_metadata_get_prefixed_u32_default(
        const rt_gguf_metadata *meta,
        const char *field,
        uint32_t fallback) {
    uint32_t value = fallback;
    qwen36_metadata_get_prefixed_u32(meta, field, &value);
    return value;
}

static bool qwen36_metadata_has_u32(
        const rt_gguf_metadata *meta,
        const char *field,
        uint32_t expected) {
    uint32_t got = 0;
    return qwen36_metadata_get_prefixed_u32(meta, field, &got) && got == expected;
}

static bool qwen36_metadata_name_allows_27b(const rt_gguf_metadata *meta) {
    const char *name = NULL;
    if (!rt_gguf_metadata_get_string(meta, "general.name", &name) || !name[0]) {
        return true;
    }
    if (qwen36_contains_ci(name, "qwen") &&
        (qwen36_contains_ci(name, "3.6") || qwen36_contains_ci(name, "36"))) {
        return qwen36_contains_ci(name, "27b");
    }
    return true;
}

static bool qwen36_metadata_matches(const rt_gguf_metadata *meta) {
    const char *arch = NULL;
    uint32_t block_count = 0;
    uint32_t nextn_predict_layers = 0;
    uint32_t context = 0;
    uint32_t full_attention_interval = 0;

    if (!rt_gguf_metadata_get_string(meta, "general.architecture", &arch) ||
        !qwen36_arch_name_supported(arch)) {
        return false;
    }
    if (!qwen36_metadata_name_allows_27b(meta)) return false;
    if (!qwen36_metadata_get_prefixed_u32(meta, "block_count", &block_count)) return false;
    qwen36_metadata_get_prefixed_u32(meta, "nextn_predict_layers", &nextn_predict_layers);
    if (block_count != QWEN36_MODEL_SPEC.n_layers &&
        block_count != QWEN36_MODEL_SPEC.n_layers + nextn_predict_layers) {
        return false;
    }
    if (!qwen36_metadata_has_u32(meta, "embedding_length", QWEN36_MODEL_SPEC.hidden_size)) return false;
    if (!qwen36_metadata_has_u32(meta, "feed_forward_length", QWEN36_MODEL_SPEC.intermediate_size)) return false;
    if (!qwen36_metadata_has_u32(meta, "attention.head_count", QWEN36_MODEL_SPEC.attention_heads)) return false;
    if (!qwen36_metadata_has_u32(meta, "attention.head_count_kv", QWEN36_MODEL_SPEC.attention_kv_heads)) return false;

    if (qwen36_metadata_get_prefixed_u32(meta, "context_length", &context) &&
        context < QWEN36_MODEL_SPEC.max_context) {
        return false;
    }
    if (qwen36_metadata_get_prefixed_u32(meta, "full_attention_interval", &full_attention_interval) &&
        full_attention_interval != QWEN36_MODEL_SPEC.full_attention_interval) {
        return false;
    }
    return true;
}

static bool qwen36_path_is_readable(const char *model_path) {
    if (!model_path || !model_path[0]) return false;
    FILE *fp = fopen(model_path, "rb");
    if (!fp) return false;
    fclose(fp);
    return true;
}

static bool qwen36_tensor_has_dims(const rt_gguf_tensor *t,
                                   uint32_t ndim,
                                   const uint64_t *dims) {
    if (!t || t->ndim != ndim) return false;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] != dims[i]) return false;
    }
    return true;
}

static bool qwen36_tensor_has_shape(const rt_gguf_tensor *t,
                                    uint32_t ndim,
                                    uint64_t d0,
                                    uint64_t d1) {
    uint64_t dims[2] = {d0, d1};
    return qwen36_tensor_has_dims(t, ndim, dims);
}

static const rt_gguf_tensor *qwen36_bind_tensor(
        const rt_gguf_file *file,
        const char *name,
        uint32_t ndim,
        uint64_t d0,
        uint64_t d1,
        uint32_t *n_bound) {
    const rt_gguf_tensor *t = rt_gguf_file_find_tensor(file, name);
    if (!qwen36_tensor_has_shape(t, ndim, d0, d1) ||
        !rt_gguf_file_tensor_data(file, t)) {
        return NULL;
    }
    if (n_bound) (*n_bound)++;
    return t;
}

static bool qwen36_bind_optional_tensor(
        const rt_gguf_file *file,
        const char *name,
        uint32_t ndim,
        uint64_t d0,
        uint64_t d1,
        const rt_gguf_tensor **out,
        uint32_t *n_bound) {
    const rt_gguf_tensor *t = rt_gguf_file_find_tensor(file, name);
    if (out) *out = NULL;
    if (!t) return true;
    if (!qwen36_tensor_has_shape(t, ndim, d0, d1) ||
        !rt_gguf_file_tensor_data(file, t)) {
        return false;
    }
    if (n_bound) (*n_bound)++;
    if (out) *out = t;
    return true;
}

static bool qwen36_tensor_2d_matches_pair(const rt_gguf_tensor *t,
                                          uint64_t d0,
                                          uint64_t d1) {
    return t &&
           t->ndim == 2 &&
           ((t->dim[0] == d0 && t->dim[1] == d1) ||
            (t->dim[0] == d1 && t->dim[1] == d0));
}

static const rt_gguf_tensor *qwen36_bind_tensor_2d_pair(
        const rt_gguf_file *file,
        const char *name,
        uint64_t d0,
        uint64_t d1,
        uint32_t *n_bound) {
    const rt_gguf_tensor *t = rt_gguf_file_find_tensor(file, name);
    if (!qwen36_tensor_2d_matches_pair(t, d0, d1) ||
        !rt_gguf_file_tensor_data(file, t)) {
        return NULL;
    }
    if (n_bound) (*n_bound)++;
    return t;
}

static bool qwen36_bind_optional_tensor_2d_pair(
        const rt_gguf_file *file,
        const char *name,
        uint64_t d0,
        uint64_t d1,
        const rt_gguf_tensor **out,
        uint32_t *n_bound) {
    const rt_gguf_tensor *t = rt_gguf_file_find_tensor(file, name);
    if (out) *out = NULL;
    if (!t) return true;
    if (!qwen36_tensor_2d_matches_pair(t, d0, d1) ||
        !rt_gguf_file_tensor_data(file, t)) {
        return false;
    }
    if (n_bound) (*n_bound)++;
    if (out) *out = t;
    return true;
}

static bool qwen36_tensor_2d_has_dim(const rt_gguf_tensor *t, uint64_t dim) {
    return t && t->ndim == 2 && (t->dim[0] == dim || t->dim[1] == dim);
}

static const rt_gguf_tensor *qwen36_bind_tensor_2d_has_dim(
        const rt_gguf_file *file,
        const char *name,
        uint64_t dim,
        uint32_t *n_bound) {
    const rt_gguf_tensor *t = rt_gguf_file_find_tensor(file, name);
    if (!qwen36_tensor_2d_has_dim(t, dim) ||
        t->dim[0] == 0 ||
        t->dim[1] == 0 ||
        !rt_gguf_file_tensor_data(file, t)) {
        return NULL;
    }
    if (n_bound) (*n_bound)++;
    return t;
}

static const rt_gguf_tensor *qwen36_bind_tensor_ndim_nonzero(
        const rt_gguf_file *file,
        const char *name,
        uint32_t ndim,
        uint32_t *n_bound) {
    const rt_gguf_tensor *t = rt_gguf_file_find_tensor(file, name);
    if (!t || t->ndim != ndim || !rt_gguf_file_tensor_data(file, t)) return NULL;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == 0) return NULL;
    }
    if (n_bound) (*n_bound)++;
    return t;
}

static bool qwen36_format_tensor_name(char *dst, size_t dstlen,
                                      const char *fmt, uint32_t il) {
    int n = snprintf(dst, dstlen, fmt, il);
    return n > 0 && (size_t)n < dstlen;
}

static bool qwen36_bind_layer_common(
        const rt_gguf_file *file,
        qwen36_layer_tensors *layer,
        uint32_t il,
        uint32_t n_embd,
        uint32_t n_ff,
        uint32_t *n_bound) {
    char name[QWEN36_MAX_TENSOR_NAME];

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.attn_norm.weight", il)) return false;
    layer->attn_norm = qwen36_bind_tensor(file, name, 1, n_embd, 0, n_bound);
    if (!layer->attn_norm) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.post_attention_norm.weight", il)) return false;
    layer->attn_post_norm = qwen36_bind_tensor(file, name, 1, n_embd, 0, n_bound);
    if (!layer->attn_post_norm) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.ffn_gate.weight", il)) return false;
    layer->ffn_gate = qwen36_bind_tensor(file, name, 2, n_embd, n_ff, n_bound);
    if (!layer->ffn_gate) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.ffn_down.weight", il)) return false;
    layer->ffn_down = qwen36_bind_tensor(file, name, 2, n_ff, n_embd, n_bound);
    if (!layer->ffn_down) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.ffn_up.weight", il)) return false;
    layer->ffn_up = qwen36_bind_tensor(file, name, 2, n_embd, n_ff, n_bound);
    return layer->ffn_up != NULL;
}

static bool qwen36_bind_full_attention_layer(
        const rt_gguf_file *file,
        qwen36_layer_tensors *layer,
        uint32_t il,
        const qwen36_model_spec *spec,
        uint32_t *n_bound) {
    char name[QWEN36_MAX_TENSOR_NAME];
    uint64_t q_gate_dim = (uint64_t)spec->attention_head_dim * spec->attention_heads * 2u;
    uint64_t kv_dim = (uint64_t)spec->attention_head_dim * spec->attention_kv_heads;
    uint64_t out_in_dim = (uint64_t)spec->attention_head_dim * spec->attention_heads;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.attn_q.weight", il)) return false;
    layer->attn_q = qwen36_bind_tensor(file, name, 2, spec->hidden_size, q_gate_dim, n_bound);
    if (!layer->attn_q) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.attn_k.weight", il)) return false;
    layer->attn_k = qwen36_bind_tensor(file, name, 2, spec->hidden_size, kv_dim, n_bound);
    if (!layer->attn_k) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.attn_v.weight", il)) return false;
    layer->attn_v = qwen36_bind_tensor(file, name, 2, spec->hidden_size, kv_dim, n_bound);
    if (!layer->attn_v) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.attn_output.weight", il)) return false;
    layer->attn_output = qwen36_bind_tensor(file, name, 2, out_in_dim, spec->hidden_size, n_bound);
    if (!layer->attn_output) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.attn_q_norm.weight", il)) return false;
    layer->attn_q_norm = qwen36_bind_tensor(file, name, 1, spec->attention_head_dim, 0, n_bound);
    if (!layer->attn_q_norm) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.attn_k_norm.weight", il)) return false;
    layer->attn_k_norm = qwen36_bind_tensor(file, name, 1, spec->attention_head_dim, 0, n_bound);
    return layer->attn_k_norm != NULL;
}

static bool qwen36_bind_recurrent_layer(
        const rt_gguf_file *file,
        qwen36_layer_tensors *layer,
        uint32_t il,
        const qwen36_weights *weights,
        uint32_t n_embd,
        uint32_t *n_bound) {
    char name[QWEN36_MAX_TENSOR_NAME];
    uint64_t key_dim = (uint64_t)weights->ssm_d_state * weights->ssm_n_group;
    uint64_t value_dim = weights->ssm_d_inner;
    uint64_t qkv_dim = key_dim * 2u + value_dim;

    layer->recurrent = true;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.attn_qkv.weight", il)) return false;
    layer->attn_qkv = qwen36_bind_tensor(file, name, 2, n_embd, qkv_dim, n_bound);
    if (!layer->attn_qkv) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.attn_gate.weight", il)) return false;
    layer->attn_gate = qwen36_bind_tensor(file, name, 2, n_embd, value_dim, n_bound);
    if (!layer->attn_gate) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.ssm_conv1d.weight", il)) return false;
    layer->ssm_conv1d = qwen36_bind_tensor(file, name, 2, weights->ssm_d_conv, qkv_dim, n_bound);
    if (!layer->ssm_conv1d) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.ssm_dt.bias", il)) return false;
    layer->ssm_dt = qwen36_bind_tensor(file, name, 1, weights->ssm_dt_rank, 0, n_bound);
    if (!layer->ssm_dt) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.ssm_a", il)) return false;
    layer->ssm_a = qwen36_bind_tensor(file, name, 1, weights->ssm_dt_rank, 0, n_bound);
    if (!layer->ssm_a) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.ssm_beta.weight", il)) return false;
    layer->ssm_beta = qwen36_bind_tensor(file, name, 2, n_embd, weights->ssm_dt_rank, n_bound);
    if (!layer->ssm_beta) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.ssm_alpha.weight", il)) return false;
    layer->ssm_alpha = qwen36_bind_tensor(file, name, 2, n_embd, weights->ssm_dt_rank, n_bound);
    if (!layer->ssm_alpha) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.ssm_norm.weight", il)) return false;
    layer->ssm_norm = qwen36_bind_tensor(file, name, 1, weights->ssm_d_state, 0, n_bound);
    if (!layer->ssm_norm) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.ssm_out.weight", il)) return false;
    layer->ssm_out = qwen36_bind_tensor(file, name, 2, value_dim, n_embd, n_bound);
    return layer->ssm_out != NULL;
}

static bool qwen36_bind_mtp_layer(
        qwen36_weights *weights,
        const rt_gguf_file *file,
        uint32_t layer_index,
        uint32_t mtp_index,
        const qwen36_model_spec *spec) {
    if (!weights || !file || !spec || mtp_index >= QWEN36_MAX_MTP_LAYERS) return false;

    char name[QWEN36_MAX_TENSOR_NAME];
    qwen36_mtp_layer_tensors mtp = {0};
    uint32_t n_bound = 0;

    if (!qwen36_bind_layer_common(file, &mtp.block, layer_index, spec->hidden_size,
                                  spec->intermediate_size, &n_bound)) {
        return false;
    }
    if (!qwen36_bind_full_attention_layer(file, &mtp.block, layer_index, spec, &n_bound)) {
        return false;
    }
    mtp.source = file;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.nextn.eh_proj.weight",
                                   layer_index)) {
        return false;
    }
    mtp.eh_proj = qwen36_bind_tensor_2d_pair(file, name,
                                             (uint64_t)spec->hidden_size * 2u,
                                             spec->hidden_size, &n_bound);
    if (!mtp.eh_proj) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.nextn.shared_head_norm.weight",
                                   layer_index)) {
        return false;
    }
    mtp.shared_head_norm = qwen36_bind_tensor(file, name, 1, spec->hidden_size, 0, &n_bound);
    if (!mtp.shared_head_norm) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.nextn.enorm.weight",
                                   layer_index)) {
        return false;
    }
    mtp.enorm = qwen36_bind_tensor(file, name, 1, spec->hidden_size, 0, &n_bound);
    if (!mtp.enorm) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "blk.%u.nextn.hnorm.weight",
                                   layer_index)) {
        return false;
    }
    mtp.hnorm = qwen36_bind_tensor(file, name, 1, spec->hidden_size, 0, &n_bound);
    if (!mtp.hnorm) return false;

    mtp.layer_index = layer_index;
    weights->mtp[mtp_index] = mtp;
    weights->n_bound += n_bound;
    weights->n_mtp_bound += n_bound;
    weights->n_mtp_layers_bound++;
    return true;
}

static bool qwen36_bind_mtp_layers(
        qwen36_weights *weights,
        const rt_gguf_file *file,
        uint32_t first_layer_index,
        uint32_t count,
        const qwen36_model_spec *spec) {
    if (!weights || !file || !spec) return false;
    if (count == 0) return true;
    if (count > QWEN36_MAX_MTP_LAYERS ||
        weights->n_mtp_layers_bound > QWEN36_MAX_MTP_LAYERS - count) {
        return false;
    }
    if (first_layer_index > UINT32_MAX - (count - 1u)) return false;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t mtp_index = weights->n_mtp_layers_bound;
        if (!qwen36_bind_mtp_layer(weights, file, first_layer_index + i,
                                   mtp_index, spec)) {
            return false;
        }
    }
    return true;
}

static bool qwen36_mtp_layer_present(const rt_gguf_file *file,
                                     uint32_t layer_index) {
    char name[128];
    if (!qwen36_format_tensor_name(name, sizeof(name),
                                   "blk.%u.nextn.eh_proj.weight",
                                   layer_index)) {
        return false;
    }
    const rt_gguf_tensor *t = rt_gguf_file_find_tensor(file, name);
    return t && rt_gguf_file_tensor_data(file, t);
}

static bool qwen36_bind_available_mtp_layers(
        qwen36_weights *weights,
        const rt_gguf_file *file,
        uint32_t first_layer_index,
        uint32_t max_count,
        const qwen36_model_spec *spec) {
    if (!weights || !file || !spec || max_count == 0) return false;
    if (max_count > QWEN36_MAX_MTP_LAYERS) max_count = QWEN36_MAX_MTP_LAYERS;
    bool bound_any = false;
    for (uint32_t i = 0; i < max_count; i++) {
        uint32_t layer_index = first_layer_index + i;
        if (layer_index < first_layer_index) return false;
        if (!qwen36_mtp_layer_present(file, layer_index)) break;
        if (!qwen36_bind_mtp_layers(weights, file, layer_index, 1, spec)) {
            return false;
        }
        bound_any = true;
    }
    return bound_any;
}

static bool qwen36_bind_weights(qwen36_weights *weights,
                                const rt_gguf_file *file,
                                const qwen36_model_spec *spec) {
    memset(weights, 0, sizeof(*weights));
    const rt_gguf_metadata *meta = rt_gguf_file_metadata(file);

    weights->n_layers_total =
        qwen36_metadata_get_prefixed_u32_default(meta, "block_count", spec->n_layers);
    weights->nextn_predict_layers =
        qwen36_metadata_get_prefixed_u32_default(meta, "nextn_predict_layers", 0);
    if (weights->n_layers_total > QWEN36_MAX_LAYERS ||
        weights->n_layers_total < spec->n_layers) {
        return false;
    }
    if (weights->nextn_predict_layers != 0 &&
        weights->n_layers_total == spec->n_layers + weights->nextn_predict_layers) {
        weights->n_layers_main = spec->n_layers;
    } else {
        weights->n_layers_main = weights->n_layers_total;
    }
    if (weights->n_layers_main != spec->n_layers) return false;

    weights->ssm_d_conv =
        qwen36_metadata_get_prefixed_u32_default(meta, "ssm.conv_kernel",
                                                 QWEN36_DEFAULT_SSM_D_CONV);
    weights->ssm_d_inner =
        qwen36_metadata_get_prefixed_u32_default(meta, "ssm.inner_size",
                                                 spec->linear_v_heads * spec->linear_head_dim);
    weights->ssm_d_state =
        qwen36_metadata_get_prefixed_u32_default(meta, "ssm.state_size",
                                                 spec->linear_head_dim);
    weights->ssm_dt_rank =
        qwen36_metadata_get_prefixed_u32_default(meta, "ssm.time_step_rank",
                                                 spec->linear_v_heads);
    weights->ssm_n_group =
        qwen36_metadata_get_prefixed_u32_default(meta, "ssm.group_count",
                                                 spec->linear_qk_heads);

    if (weights->ssm_d_conv != QWEN36_DEFAULT_SSM_D_CONV ||
        weights->ssm_d_inner != spec->linear_v_heads * spec->linear_head_dim ||
        weights->ssm_d_state != spec->linear_head_dim ||
        weights->ssm_dt_rank != spec->linear_v_heads ||
        weights->ssm_n_group != spec->linear_qk_heads) {
        return false;
    }

    weights->tok_embd = qwen36_bind_tensor(file, "token_embd.weight", 2,
                                           spec->hidden_size, spec->vocab_size,
                                           &weights->n_bound);
    if (!weights->tok_embd) return false;

    weights->output_norm = qwen36_bind_tensor(file, "output_norm.weight", 1,
                                              spec->hidden_size, 0,
                                              &weights->n_bound);
    if (!weights->output_norm) return false;

    if (!qwen36_bind_optional_tensor(file, "output.weight", 2,
                                     spec->hidden_size, spec->vocab_size,
                                     &weights->output, &weights->n_bound)) {
        return false;
    }
    if (!weights->output) weights->output = weights->tok_embd;

    for (uint32_t il = 0; il < weights->n_layers_main; il++) {
        qwen36_layer_tensors *layer = &weights->layer[il];
        bool recurrent = (il + 1u) % spec->full_attention_interval != 0;
        if (!qwen36_bind_layer_common(file, layer, il, spec->hidden_size,
                                      spec->intermediate_size, &weights->n_bound)) {
            return false;
        }
        if (recurrent) {
            if (!qwen36_bind_recurrent_layer(file, layer, il, weights,
                                             spec->hidden_size, &weights->n_bound)) {
                return false;
            }
        } else if (!qwen36_bind_full_attention_layer(file, layer, il, spec, &weights->n_bound)) {
            return false;
        }
    }

    uint32_t embedded_mtp_layers = weights->n_layers_total - weights->n_layers_main;
    if (embedded_mtp_layers != 0 &&
        !qwen36_bind_mtp_layers(weights, file, weights->n_layers_main,
                                embedded_mtp_layers, spec)) {
        return false;
    }

    return true;
}

static bool qwen36_mmproj_projector_supported(const char *projector_type) {
    return projector_type &&
           (!strcmp(projector_type, "qwen3vl_merger") ||
            !strcmp(projector_type, "qwen2.5vl_merger") ||
            !strcmp(projector_type, "qwen2vl_merger"));
}

static bool qwen36_mmproj_metadata_string(const rt_gguf_metadata *meta,
                                          const char **out) {
    if (rt_gguf_metadata_get_string(meta, "clip.projector_type", out)) return true;
    return rt_gguf_metadata_get_string(meta, "clip.vision.projector_type", out);
}

static bool qwen36_mmproj_patch_shape_ok(const rt_gguf_tensor *t,
                                         uint32_t n_embd,
                                         uint32_t patch_size) {
    if (!t || t->ndim != 4) return false;
    return (t->dim[0] == patch_size &&
            t->dim[1] == patch_size &&
            t->dim[2] == 3 &&
            t->dim[3] == n_embd) ||
           (t->dim[0] == n_embd &&
            t->dim[1] == 3 &&
            t->dim[2] == patch_size &&
            t->dim[3] == patch_size);
}

static bool qwen36_bind_mmproj_attention(qwen36_mmproj *mm,
                                         const rt_gguf_file *file,
                                         qwen36_mmproj_layer_tensors *layer,
                                         uint32_t il) {
    char name[QWEN36_MAX_TENSOR_NAME];
    const rt_gguf_tensor *qkv = NULL;

    if (!qwen36_format_tensor_name(name, sizeof(name), "v.blk.%u.attn_qkv.weight", il)) return false;
    qkv = rt_gguf_file_find_tensor(file, name);
    if (qkv) {
        layer->attn_qkv = qwen36_bind_tensor_2d_pair(file, name, mm->n_embd,
                                                     (uint64_t)mm->n_embd * 3u,
                                                     &mm->n_bound);
        return layer->attn_qkv != NULL;
    }

    if (!qwen36_format_tensor_name(name, sizeof(name), "v.blk.%u.attn_q.weight", il)) return false;
    layer->attn_q = qwen36_bind_tensor_2d_pair(file, name, mm->n_embd, mm->n_embd,
                                               &mm->n_bound);
    if (!layer->attn_q) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "v.blk.%u.attn_k.weight", il)) return false;
    layer->attn_k = qwen36_bind_tensor_2d_pair(file, name, mm->n_embd, mm->n_embd,
                                               &mm->n_bound);
    if (!layer->attn_k) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "v.blk.%u.attn_v.weight", il)) return false;
    layer->attn_v = qwen36_bind_tensor_2d_pair(file, name, mm->n_embd, mm->n_embd,
                                               &mm->n_bound);
    return layer->attn_v != NULL;
}

static bool qwen36_bind_mmproj_layer(qwen36_mmproj *mm,
                                     const rt_gguf_file *file,
                                     uint32_t il) {
    qwen36_mmproj_layer_tensors *layer = &mm->layer[il];
    char name[QWEN36_MAX_TENSOR_NAME];

    if (!qwen36_format_tensor_name(name, sizeof(name), "v.blk.%u.ln1.weight", il)) return false;
    layer->ln1 = qwen36_bind_tensor(file, name, 1, mm->n_embd, 0, &mm->n_bound);
    if (!layer->ln1) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "v.blk.%u.ln2.weight", il)) return false;
    layer->ln2 = qwen36_bind_tensor(file, name, 1, mm->n_embd, 0, &mm->n_bound);
    if (!layer->ln2) return false;

    if (!qwen36_bind_mmproj_attention(mm, file, layer, il)) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "v.blk.%u.attn_out.weight", il)) return false;
    layer->attn_out = qwen36_bind_tensor_2d_pair(file, name, mm->n_embd, mm->n_embd,
                                                 &mm->n_bound);
    if (!layer->attn_out) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "v.blk.%u.ffn_up.weight", il)) return false;
    layer->ffn_up = qwen36_bind_tensor_2d_pair(file, name, mm->n_embd, mm->n_ff,
                                               &mm->n_bound);
    if (!layer->ffn_up) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "v.blk.%u.ffn_down.weight", il)) return false;
    layer->ffn_down = qwen36_bind_tensor_2d_pair(file, name, mm->n_ff, mm->n_embd,
                                                 &mm->n_bound);
    if (!layer->ffn_down) return false;

    if (!qwen36_format_tensor_name(name, sizeof(name), "v.blk.%u.ffn_gate.weight", il)) return false;
    return qwen36_bind_optional_tensor_2d_pair(file, name, mm->n_embd, mm->n_ff,
                                               &layer->ffn_gate, &mm->n_bound);
}

static bool qwen36_bind_mmproj(qwen36_mmproj *mm,
                               const rt_gguf_file *file,
                               const qwen36_model_spec *spec) {
    memset(mm, 0, sizeof(*mm));
    const rt_gguf_metadata *meta = rt_gguf_file_metadata(file);
    const char *projector_type = NULL;
    bool has_vision = false;

    if (!rt_gguf_metadata_get_bool(meta, "clip.has_vision_encoder", &has_vision) ||
        !has_vision ||
        !qwen36_mmproj_metadata_string(meta, &projector_type) ||
        !qwen36_mmproj_projector_supported(projector_type)) {
        return false;
    }

    if (!rt_gguf_metadata_get_u32(meta, "clip.vision.embedding_length", &mm->n_embd) ||
        !rt_gguf_metadata_get_u32(meta, "clip.vision.feed_forward_length", &mm->n_ff) ||
        !rt_gguf_metadata_get_u32(meta, "clip.vision.block_count", &mm->n_layer) ||
        !rt_gguf_metadata_get_u32(meta, "clip.vision.projection_dim", &mm->projection_dim) ||
        !rt_gguf_metadata_get_u32(meta, "clip.vision.attention.head_count", &mm->n_head) ||
        !rt_gguf_metadata_get_u32(meta, "clip.vision.image_size", &mm->image_size) ||
        !rt_gguf_metadata_get_u32(meta, "clip.vision.patch_size", &mm->patch_size)) {
        return false;
    }
    rt_gguf_metadata_get_u32(meta, "clip.vision.attention.head_count_kv", &mm->n_head_kv);
    if (!rt_gguf_metadata_get_u32(meta, "clip.vision.spatial_merge_size", &mm->spatial_merge_size)) {
        mm->spatial_merge_size = 2;
    }

    if (mm->n_embd == 0 ||
        mm->n_ff == 0 ||
        mm->n_layer == 0 ||
        mm->n_layer > QWEN36_MMPROJ_MAX_LAYERS ||
        mm->n_head == 0 ||
        mm->projection_dim != spec->hidden_size ||
        mm->image_size == 0 ||
        mm->patch_size == 0 ||
        mm->spatial_merge_size == 0) {
        return false;
    }
    uint64_t patch_factor = (uint64_t)mm->patch_size * mm->spatial_merge_size;
    if (patch_factor == 0 || patch_factor > UINT32_MAX) return false;
    uint64_t factor_pixels = patch_factor * patch_factor;
    if (factor_pixels / patch_factor != patch_factor) return false;
    if (factor_pixels > UINT64_MAX / 16384u) return false;
    mm->min_pixels = 4u * factor_pixels;
    mm->max_pixels = 16384u * factor_pixels;
    rt_gguf_metadata_get_u64(meta, "clip.vision.min_pixels", &mm->min_pixels);
    rt_gguf_metadata_get_u64(meta, "clip.vision.max_pixels", &mm->max_pixels);
    if (mm->min_pixels < factor_pixels) mm->min_pixels = factor_pixels;
    if (mm->max_pixels < mm->min_pixels) return false;

    int n = snprintf(mm->projector_type, sizeof(mm->projector_type), "%s", projector_type);
    if (n <= 0 || (size_t)n >= sizeof(mm->projector_type)) return false;

    mm->patch_embd = rt_gguf_file_find_tensor(file, "v.patch_embd.weight");
    if (!qwen36_mmproj_patch_shape_ok(mm->patch_embd, mm->n_embd, mm->patch_size) ||
        !rt_gguf_file_tensor_data(file, mm->patch_embd)) {
        return false;
    }
    mm->n_bound++;

    if (!qwen36_bind_optional_tensor(file, "v.patch_embd.bias", 1, mm->n_embd, 0,
                                     &mm->patch_bias, &mm->n_bound)) {
        return false;
    }

    for (uint32_t il = 0; il < mm->n_layer; il++) {
        if (!qwen36_bind_mmproj_layer(mm, file, il)) return false;
    }

    mm->mm_0_w = qwen36_bind_tensor_ndim_nonzero(file, "mm.0.weight", 2, &mm->n_bound);
    if (!mm->mm_0_w) return false;
    mm->mm_0_b = qwen36_bind_tensor_ndim_nonzero(file, "mm.0.bias", 1, &mm->n_bound);
    if (!mm->mm_0_b) return false;
    mm->mm_1_w = qwen36_bind_tensor_2d_has_dim(file, "mm.2.weight",
                                               spec->hidden_size, &mm->n_bound);
    if (!mm->mm_1_w) return false;
    mm->mm_1_b = qwen36_bind_tensor(file, "mm.2.bias", 1,
                                    spec->hidden_size, 0, &mm->n_bound);
    if (!mm->mm_1_b) return false;

    mm->loaded = true;
    return true;
}

static bool qwen36_rt_probe_model_path(const char *model_path) {
    rt_gguf_metadata meta = {0};
    char err[256];

    if (rt_gguf_metadata_read(model_path, &meta, err, sizeof(err)) == 0) {
        bool ok = qwen36_metadata_matches(&meta);
        rt_gguf_metadata_free(&meta);
        return ok;
    }

    if (qwen36_path_is_readable(model_path)) return false;
    return qwen36_probe_filename(model_path);
}

static bool qwen36_backend_supported(rt_backend backend) {
    return backend == RT_BACKEND_AUTO ||
           backend == RT_BACKEND_CPU ||
           (backend == RT_BACKEND_METAL && qwen36_metal_available());
}

static void qwen36_engine_destroy(qwen36_engine *engine) {
    if (!engine) return;
    qwen36_metal_destroy(engine->metal);
    qwen36_vocab_free(&engine->vocab);
    rt_gguf_file_close(engine->mtp_model);
    rt_gguf_file_close(engine->mmproj);
    rt_gguf_file_close(engine->model);
    free(engine);
}

static int qwen36_rt_engine_open(void **out, const rt_engine_options *opt) {
    if (!out || !opt || !opt->model_path) return 1;
    *out = NULL;

    if (!qwen36_backend_supported(opt->backend)) return 1;

    const qwen36_runtime_engine_options *extra = opt->model_options;
    bool mtp_requested = extra &&
        (extra->enable_mtp ||
         (extra->mtp_path && extra->mtp_path[0]) ||
         extra->mtp_draft_tokens > 1);

    qwen36_engine *engine = calloc(1, sizeof(*engine));
    if (!engine) return 1;
    engine->spec = qwen36_runtime_model_spec();
    engine->backend = opt->backend == RT_BACKEND_AUTO ? RT_BACKEND_CPU : opt->backend;
    engine->mtp_draft_tokens = 1;
    engine->mtp_margin = 0.0f;
    if (engine->backend == RT_BACKEND_METAL) {
        char metal_err[256] = {0};
        engine->metal = qwen36_metal_create(metal_err, sizeof(metal_err));
        if (!engine->metal) {
            qwen36_engine_destroy(engine);
            return 1;
        }
    }

    char err[256];
    if (rt_gguf_file_open(&engine->model, opt->model_path, err, sizeof(err)) != 0) {
        qwen36_engine_destroy(engine);
        return 1;
    }
    const rt_gguf_metadata *meta = rt_gguf_file_metadata(engine->model);
    if (!qwen36_metadata_matches(meta)) {
        qwen36_engine_destroy(engine);
        return 1;
    }
    if (!qwen36_vocab_load(&engine->vocab, engine->model)) {
        qwen36_engine_destroy(engine);
        return 1;
    }
    if (!qwen36_bind_weights(&engine->weights, engine->model, engine->spec)) {
        qwen36_engine_destroy(engine);
        return 1;
    }

    if (extra && extra->mmproj_path && extra->mmproj_path[0]) {
        if (rt_gguf_file_open(&engine->mmproj, extra->mmproj_path, err, sizeof(err)) != 0) {
            qwen36_engine_destroy(engine);
            return 1;
        }
        if (!qwen36_bind_mmproj(&engine->mmproj_info, engine->mmproj, engine->spec)) {
            qwen36_engine_destroy(engine);
            return 1;
        }
    } else if (extra && !extra->language_model_only && engine->spec->multimodal_checkpoint) {
        /*
         * The text GGUF is enough for text-only usage.  Server multimodal
         * requests will reject media until a mmproj file is provided.
         */
    }

    if (extra && extra->mtp_path && extra->mtp_path[0]) {
        if (engine->weights.n_mtp_layers_bound != 0) {
            qwen36_engine_destroy(engine);
            return 1;
        }
        if (rt_gguf_file_open(&engine->mtp_model, extra->mtp_path, err, sizeof(err)) != 0) {
            qwen36_engine_destroy(engine);
            return 1;
        }
        uint32_t max_mtp_layers = 1;
        if (extra->mtp_draft_tokens > 1) {
            max_mtp_layers = (uint32_t)(extra->mtp_draft_tokens - 1);
        }
        if (!qwen36_bind_available_mtp_layers(&engine->weights,
                                              engine->mtp_model,
                                              engine->spec->n_layers,
                                              max_mtp_layers,
                                              engine->spec)) {
            qwen36_engine_destroy(engine);
            return 1;
        }
    }

    if (mtp_requested && engine->weights.n_mtp_layers_bound == 0) {
        qwen36_engine_destroy(engine);
        return 1;
    }
    engine->mtp_enabled = mtp_requested && engine->weights.n_mtp_layers_bound != 0;
    if (extra && extra->mtp_draft_tokens > 0) {
        engine->mtp_draft_tokens = extra->mtp_draft_tokens;
        if (engine->mtp_draft_tokens > 16) engine->mtp_draft_tokens = 16;
    }
    if (extra && isfinite(extra->mtp_margin) && extra->mtp_margin > 0.0f) {
        engine->mtp_margin = extra->mtp_margin;
    }

    *out = engine;
    return 0;
}

static void qwen36_rt_engine_close(void *engine) {
    qwen36_engine_destroy(engine);
}

static void qwen36_rt_engine_summary(void *engine) {
    qwen36_engine *e = engine;
    const qwen36_model_spec *s = qwen36_runtime_model_spec();
    if (!e) {
        fprintf(stderr,
                "%s runtime scaffold\n"
                "  repo: %s\n"
                "  status: engine not open\n"
                "  text model: %uB dense, %u layers, hidden %u, vocab %u, native ctx %u\n",
                s->display_name,
                s->hf_repo,
                s->parameter_count_b,
                s->n_layers,
                s->hidden_size,
                s->vocab_size,
                s->max_context);
        return;
    }

    const rt_gguf_metadata *meta = rt_gguf_file_metadata(e->model);
    const char *arch = NULL;
    const char *name = NULL;
    rt_gguf_metadata_get_string(meta, "general.architecture", &arch);
    rt_gguf_metadata_get_string(meta, "general.name", &name);
    fprintf(stderr,
            "%s runtime\n"
            "  repo: %s\n"
            "  model: %s\n"
            "  gguf: v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors, %.2f GiB\n"
            "  arch: %s%s%s\n"
            "  backend: %s diagnostic loader/session path\n"
            "  text model: %uB dense, %u layers, hidden %u, vocab %u, native ctx %u\n"
            "  bound tensors: %u, recurrent layers: %u, full-attention layers: %u\n"
            "  mmproj: %s\n",
            s->display_name,
            s->hf_repo,
            rt_gguf_file_path(e->model) ? rt_gguf_file_path(e->model) : "(unknown)",
            meta ? meta->version : 0,
            meta ? meta->n_kv : 0,
            rt_gguf_file_tensor_count(e->model),
            (double)rt_gguf_file_size(e->model) / (1024.0 * 1024.0 * 1024.0),
            arch ? arch : "(unknown)",
            name ? ", " : "",
            name ? name : "",
            rt_backend_name(e->backend),
            s->parameter_count_b,
            s->n_layers,
            s->hidden_size,
            s->vocab_size,
            s->max_context,
            e->weights.n_bound,
            e->weights.n_layers_main - (e->weights.n_layers_main / s->full_attention_interval),
            e->weights.n_layers_main / s->full_attention_interval,
            e->mmproj_info.loaded ? "loaded" : "not loaded");
    if (e->mmproj_info.loaded) {
        fprintf(stderr,
                "    projector: %s, vision layers: %u, hidden %u, ff %u, image %u, patch %u, proj %u, bound tensors: %u\n",
                e->mmproj_info.projector_type,
                e->mmproj_info.n_layer,
                e->mmproj_info.n_embd,
                e->mmproj_info.n_ff,
                e->mmproj_info.image_size,
                e->mmproj_info.patch_size,
                e->mmproj_info.projection_dim,
                e->mmproj_info.n_bound);
    }
    if (e->weights.n_mtp_layers_bound != 0) {
        fprintf(stderr,
                "  mtp: %u layer(s), bound tensors: %u, draft tokens: %d%s\n",
                e->weights.n_mtp_layers_bound,
                e->weights.n_mtp_bound,
                e->mtp_draft_tokens,
                e->mtp_enabled ? " (CPU draft-logit probe + target verification)" : " (disabled)");
    } else {
        fprintf(stderr, "  mtp: not loaded\n");
    }
}

static uint32_t qwen36_rt_engine_vocab_size(void *engine) {
    qwen36_engine *e = engine;
    rt_gguf_array_ref tokens = {0};
    if (e && rt_gguf_metadata_get_array(rt_gguf_file_metadata(e->model),
                                        "tokenizer.ggml.tokens",
                                        &tokens) &&
        tokens.len <= UINT32_MAX) {
        return (uint32_t)tokens.len;
    }
    return QWEN36_MODEL_SPEC.vocab_size;
}

static void qwen36_utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t qwen36_gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) {
        return b;
    }

    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174) continue;
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

static char *qwen36_byte_encode(qwen36_str in, uint64_t *out_len) {
    char *out = qwen36_xmalloc((size_t)in.len * 4 + 1);
    char *p = out;
    for (uint64_t i = 0; i < in.len; i++) {
        qwen36_utf8_put(&p, qwen36_gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int qwen36_utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

static uint64_t qwen36_next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = qwen36_utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

typedef struct {
    char *ptr;
    uint64_t len;
} qwen36_owned_str;

static qwen36_owned_str qwen36_owned_copy(const char *ptr, uint64_t len) {
    qwen36_owned_str s = {0};
    s.ptr = qwen36_xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

static int qwen36_bpe_rank(const qwen36_vocab *vocab,
                           const qwen36_owned_str *a,
                           const qwen36_owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : qwen36_xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    qwen36_table_get(&vocab->merge_rank, buf, len, &rank);
    if (buf != stack) free(buf);
    return rank;
}

static bool qwen36_bpe_emit_piece(const qwen36_vocab *vocab, qwen36_str raw_piece, rt_tokens *out) {
    if (!raw_piece.ptr || raw_piece.len == 0) return true;

    uint64_t encoded_len = 0;
    char *encoded = qwen36_byte_encode(raw_piece, &encoded_len);
    bool ok = true;

    int n_sym = 0;
    int cap_sym = 32;
    qwen36_owned_str *sym = qwen36_xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = qwen36_utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = qwen36_xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = qwen36_owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = qwen36_bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;

        qwen36_owned_str merged = {0};
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = qwen36_xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;
        for (int j = best_i + 1; j + 1 < n_sym; j++) sym[j] = sym[j + 1];
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = qwen36_vocab_lookup_len(vocab, sym[i].ptr, sym[i].len);
        if (token >= 0) {
            rt_tokens_push(out, token);
        } else {
            bool emitted = false;
            for (uint64_t j = 0; j < sym[i].len;) {
                uint64_t next = qwen36_next_utf8_char(sym[i].ptr, sym[i].len, j);
                token = qwen36_vocab_lookup_len(vocab, sym[i].ptr + j, next - j);
                if (token >= 0) {
                    rt_tokens_push(out, token);
                    emitted = true;
                } else {
                    ok = false;
                }
                j = next;
            }
            if (!emitted) ok = false;
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
    return ok;
}

static bool qwen36_ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool qwen36_ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool qwen36_ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool qwen36_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static bool qwen36_ascii_crlf(uint8_t c) {
    return c == '\r' || c == '\n';
}

static bool qwen36_ascii_starts_ci(const char *text,
                                   uint64_t len,
                                   uint64_t pos,
                                   const char *pat) {
    uint64_t n = strlen(pat);
    if (pos > len || n > len - pos) return false;
    for (uint64_t i = 0; i < n; i++) {
        if (tolower((unsigned char)text[pos + i]) !=
            tolower((unsigned char)pat[i])) {
            return false;
        }
    }
    return true;
}

static uint64_t qwen36_match_contraction(const char *text,
                                         uint64_t len,
                                         uint64_t pos) {
    static const char *const contractions[] = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
    };
    if (pos >= len || text[pos] != '\'') return 0;
    for (size_t i = 0; i < sizeof(contractions) / sizeof(contractions[0]); i++) {
        if (qwen36_ascii_starts_ci(text, len, pos, contractions[i])) {
            return (uint64_t)strlen(contractions[i]);
        }
    }
    return 0;
}

static bool qwen36_letter_mark_at(const char *text, uint64_t len, uint64_t pos) {
    if (!text || pos >= len) return false;
    uint8_t c = (uint8_t)text[pos];
    if (c < 0x80) return qwen36_ascii_alpha(c);
    /*
     * Qwen's tokenizer config uses \p{L}/\p{M}.  Keep the native path
     * dependency-free by treating non-ASCII UTF-8 scalars as letter/mark-like;
     * official vector fixtures remain the authority for exact Unicode edge
     * cases.
     */
    return true;
}

static bool qwen36_number_at(const char *text, uint64_t len, uint64_t pos) {
    return text && pos < len && qwen36_ascii_digit((uint8_t)text[pos]);
}

static bool qwen36_space_at(const char *text, uint64_t len, uint64_t pos) {
    return text && pos < len && qwen36_ascii_space((uint8_t)text[pos]);
}

static bool qwen36_punct_at(const char *text, uint64_t len, uint64_t pos) {
    return text && pos < len &&
           qwen36_ascii_punct_symbol((uint8_t)text[pos]);
}

static uint64_t qwen36_consume_letter_mark(const char *text,
                                           uint64_t len,
                                           uint64_t pos) {
    while (pos < len && qwen36_letter_mark_at(text, len, pos)) {
        pos = qwen36_next_utf8_char(text, len, pos);
    }
    return pos;
}

static uint64_t qwen36_consume_punct(const char *text,
                                     uint64_t len,
                                     uint64_t pos) {
    while (pos < len && qwen36_punct_at(text, len, pos)) pos++;
    while (pos < len && qwen36_ascii_crlf((uint8_t)text[pos])) pos++;
    return pos;
}

static uint64_t qwen36_consume_space_run(const char *text,
                                         uint64_t len,
                                         uint64_t pos) {
    while (pos < len && qwen36_space_at(text, len, pos)) pos++;
    return pos;
}

static uint64_t qwen36_next_pretoken_piece(const char *text,
                                           uint64_t len,
                                           uint64_t pos) {
    uint64_t n = qwen36_match_contraction(text, len, pos);
    if (n != 0) return pos + n;

    if (qwen36_letter_mark_at(text, len, pos)) {
        return qwen36_consume_letter_mark(text, len, pos);
    }

    uint64_t after_first = qwen36_next_utf8_char(text, len, pos);
    if (after_first < len &&
        !qwen36_ascii_crlf((uint8_t)text[pos]) &&
        !qwen36_letter_mark_at(text, len, pos) &&
        !qwen36_number_at(text, len, pos) &&
        qwen36_letter_mark_at(text, len, after_first)) {
        return qwen36_consume_letter_mark(text, len, after_first);
    }

    if (qwen36_number_at(text, len, pos)) {
        return qwen36_next_utf8_char(text, len, pos);
    }

    if ((uint8_t)text[pos] == ' ' && pos + 1 < len &&
        qwen36_punct_at(text, len, pos + 1)) {
        return qwen36_consume_punct(text, len, pos + 1);
    }

    if (qwen36_punct_at(text, len, pos)) {
        return qwen36_consume_punct(text, len, pos);
    }

    if (qwen36_space_at(text, len, pos)) {
        return qwen36_consume_space_run(text, len, pos);
    }

    return qwen36_next_utf8_char(text, len, pos);
}

static bool qwen36_bpe_tokenize_text(const qwen36_vocab *vocab, const char *text, rt_tokens *out) {
    if (!text) text = "";
    uint64_t len = strlen(text);
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        pos = qwen36_next_pretoken_piece(text, len, pos);
        if (pos == start) pos = qwen36_next_utf8_char(text, len, pos);
        if (!qwen36_bpe_emit_piece(vocab,
                                   (qwen36_str){ text + start, pos - start },
                                   out)) {
            return false;
        }
    }
    return true;
}

static bool qwen36_special_at(const qwen36_vocab *vocab, const char *p, int *token, size_t *len) {
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<|endoftext|>",  vocab->eos_id},
        {"<|im_start|>",   vocab->im_start_id},
        {"<|im_end|>",     vocab->im_end_id},
        {"<think>",        vocab->think_start_id},
        {"</think>",       vocab->think_end_id},
        {"<|vision_start|>", vocab->vision_start_id},
        {"<|vision_end|>", vocab->vision_end_id},
        {"<|image_pad|>",  vocab->image_pad_id},
        {"<|video_pad|>",  vocab->video_pad_id},
    };

    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        if (specials[i].token < 0) continue;
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}

static bool qwen36_tokenize_span(const qwen36_vocab *vocab, const char *p, size_t n, rt_tokens *out) {
    if (n == 0) return true;
    char *tmp = qwen36_xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bool ok = qwen36_bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
    return ok;
}

static bool qwen36_tokenize_rendered_text(const qwen36_vocab *vocab, const char *text, rt_tokens *out) {
    if (!text) text = "";

    const char *span = text;
    const char *p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (qwen36_special_at(vocab, p, &token, &len)) {
            if (!qwen36_tokenize_span(vocab, span, (size_t)(p - span), out)) {
                return false;
            }
            rt_tokens_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    return qwen36_tokenize_span(vocab, span, (size_t)(p - span), out);
}

static uint32_t qwen36_utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int qwen36_gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) continue;
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}

static char *qwen36_rt_token_text(void *engine, int token, size_t *len) {
    qwen36_engine *e = engine;
    if (!e || token < 0 || token >= e->vocab.n_vocab) {
        if (len) *len = 0;
        char *out = qwen36_xmalloc(1);
        out[0] = '\0';
        return out;
    }

    qwen36_str s = e->vocab.token[token];
    char *out = qwen36_xmalloc((size_t)s.len + 1);
    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = qwen36_utf8_decode_one(s.ptr, s.len, &pos);
        int b = qwen36_gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

static int qwen36_rt_tokenize_text(void *engine, const char *text, rt_tokens *out) {
    qwen36_engine *e = engine;
    if (!e || !out) return 1;
    rt_tokens tmp = {0};
    if (!qwen36_tokenize_rendered_text(&e->vocab, text ? text : "", &tmp)) {
        rt_tokens_free(&tmp);
        return 1;
    }
    for (int i = 0; i < tmp.len; i++) rt_tokens_push(out, tmp.v[i]);
    rt_tokens_free(&tmp);
    return 0;
}

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} qwen36_buf;

static void qwen36_buf_reserve(qwen36_buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) abort();
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) cap *= 2;
    b->ptr = qwen36_xrealloc(b->ptr, cap);
    b->cap = cap;
}

static void qwen36_buf_append(qwen36_buf *b, const char *p, size_t n) {
    qwen36_buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void qwen36_buf_puts(qwen36_buf *b, const char *s) {
    qwen36_buf_append(b, s, strlen(s));
}

static const char *qwen36_chat_role(const char *role) {
    if (!role || !role[0]) return "user";
    if (!strcmp(role, "developer")) return "system";
    if (!strcmp(role, "function")) return "tool";
    return role;
}

static const char *qwen36_visible_assistant_content(const char *content, bool preserve_thinking) {
    if (!content) return "";
    if (preserve_thinking) return content;
    const char *end = strstr(content, "</think>");
    if (!end) return content;
    end += strlen("</think>");
    while (*end == '\n' || *end == '\r') end++;
    return end;
}

static const char *qwen36_skip_ascii_space(const char *p) {
    while (p && *p && isspace((unsigned char)*p)) p++;
    return p ? p : "";
}

static bool qwen36_content_is_tool_response(const char *content) {
    static const char open[] = "<tool_response>";
    static const char close[] = "</tool_response>";
    const size_t open_len = sizeof(open) - 1;
    const size_t close_len = sizeof(close) - 1;
    const char *p = qwen36_skip_ascii_space(content ? content : "");
    bool saw_response = false;
    while (*p) {
        if (strncmp(p, open, open_len) != 0) return false;
        p += open_len;
        const char *end = strstr(p, close);
        if (!end) return false;
        p = qwen36_skip_ascii_space(end + close_len);
        saw_response = true;
    }
    return saw_response;
}

static bool qwen36_last_real_user_query_index(const rt_chat_message *messages,
                                              size_t n_messages,
                                              size_t *out_index) {
    for (size_t remain = n_messages; remain > 0; remain--) {
        const size_t i = remain - 1;
        const char *role = qwen36_chat_role(messages[i].role);
        if (!strcmp(role, "user") &&
            !qwen36_content_is_tool_response(messages[i].content)) {
            if (out_index) *out_index = i;
            return true;
        }
    }
    return false;
}

typedef struct {
    bool vision_start;
    bool vision_end;
    bool image_pad;
    bool video_pad;
} qwen36_media_markers;

static qwen36_media_markers qwen36_content_media_markers(const char *content) {
    if (!content) return (qwen36_media_markers){0};
    return (qwen36_media_markers){
        .vision_start = strstr(content, "<|vision_start|>") != NULL,
        .vision_end = strstr(content, "<|vision_end|>") != NULL,
        .image_pad = strstr(content, "<|image_pad|>") != NULL,
        .video_pad = strstr(content, "<|video_pad|>") != NULL,
    };
}

static bool qwen36_media_markers_any(qwen36_media_markers m) {
    return m.vision_start || m.vision_end || m.image_pad || m.video_pad;
}

static bool qwen36_engine_can_render_media(const qwen36_engine *e,
                                           qwen36_media_markers m) {
    return e &&
           e->mmproj_info.loaded &&
           (!m.vision_start || e->vocab.vision_start_id >= 0) &&
           (!m.vision_end || e->vocab.vision_end_id >= 0) &&
           (!m.image_pad || e->vocab.image_pad_id >= 0) &&
           (!m.video_pad || e->vocab.video_pad_id >= 0);
}

static int qwen36_rt_render_chat(
        void *engine,
        const rt_chat_message *messages,
        size_t n_messages,
        const rt_chat_render_options *options,
        rt_tokens *out) {
    qwen36_engine *e = engine;
    if (!e || (!messages && n_messages != 0) || !out) return 1;

    const qwen36_runtime_chat_options *qwen_chat =
        options ? options->model_options : NULL;
    qwen36_think_mode think_mode = qwen_chat ? qwen_chat->think_mode : QWEN36_THINK_AUTO;
    bool preserve_thinking = qwen_chat ? qwen_chat->preserve_thinking : false;
    bool add_generation_prompt = options ? options->add_generation_prompt : true;

    size_t last_query_index = 0;
    bool has_last_query = qwen36_last_real_user_query_index(messages, n_messages,
                                                           &last_query_index);

    qwen36_buf rendered = {0};
    for (size_t i = 0; i < n_messages; i++) {
        qwen36_media_markers media = qwen36_content_media_markers(messages[i].content);
        if (qwen36_media_markers_any(media) &&
            !qwen36_engine_can_render_media(e, media)) {
            free(rendered.ptr);
            return 1;
        }
        const char *role = qwen36_chat_role(messages[i].role);
        const char *content = messages[i].content ? messages[i].content : "";
        if (!strcmp(role, "assistant")) {
            bool replay_after_query = has_last_query && i > last_query_index;
            content = qwen36_visible_assistant_content(content,
                                                       preserve_thinking ||
                                                       replay_after_query);
        }
        qwen36_buf_puts(&rendered, "<|im_start|>");
        qwen36_buf_puts(&rendered, role);
        qwen36_buf_puts(&rendered, "\n");
        qwen36_buf_puts(&rendered, content);
        qwen36_buf_puts(&rendered, "<|im_end|>\n");
    }

    if (add_generation_prompt) {
        qwen36_buf_puts(&rendered, "<|im_start|>assistant\n");
        if (think_mode == QWEN36_THINK_DISABLED) {
            qwen36_buf_puts(&rendered, "<think>\n\n</think>\n\n");
        } else {
            qwen36_buf_puts(&rendered, "<think>\n");
        }
    }

    rt_tokens tmp = {0};
    bool ok = qwen36_tokenize_rendered_text(&e->vocab,
                                            rendered.ptr ? rendered.ptr : "",
                                            &tmp);
    free(rendered.ptr);
    if (!ok) {
        rt_tokens_free(&tmp);
        return 1;
    }
    for (int i = 0; i < tmp.len; i++) rt_tokens_push(out, tmp.v[i]);
    rt_tokens_free(&tmp);
    return 0;
}

static int qwen36_rt_token_eos(void *engine) {
    qwen36_engine *e = engine;
    return e ? e->vocab.eos_id : -1;
}

static bool qwen36_u64_add_overflows(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return true;
    if (out) *out = a + b;
    return false;
}

static bool qwen36_u64_mul_overflows(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return true;
    if (out) *out = a * b;
    return false;
}

static bool qwen36_scale_bytes(uint64_t count, uint64_t elem_bytes, uint64_t *out) {
    return !qwen36_u64_mul_overflows(count, elem_bytes, out);
}

static bool qwen36_state_layout_bytes(const qwen36_model_spec *spec,
                                      int ctx_size,
                                      uint64_t *full_kv_bytes,
                                      uint64_t *recurrent_state_bytes,
                                      uint64_t *conv_state_bytes) {
    if (!spec || ctx_size <= 0) return false;
    if (spec->full_attention_interval == 0) return false;

    uint64_t full_layers = spec->n_layers / spec->full_attention_interval;
    uint64_t recurrent_layers = spec->n_layers - full_layers;
    uint64_t full_kv_width = (uint64_t)spec->attention_kv_heads * spec->attention_head_dim;
    uint64_t recurrent_value_width = (uint64_t)spec->linear_v_heads * spec->linear_head_dim;
    uint64_t recurrent_key_width = (uint64_t)spec->linear_qk_heads * spec->linear_head_dim;
    uint64_t qkv_width = recurrent_key_width * 2u + recurrent_value_width;
    uint64_t n = 0;

    if (full_layers == 0) return false;
    if (qwen36_u64_mul_overflows(full_layers, (uint64_t)ctx_size, &n) ||
        qwen36_u64_mul_overflows(n, 2u, &n) ||
        qwen36_u64_mul_overflows(n, full_kv_width, &n) ||
        !qwen36_scale_bytes(n, sizeof(float), full_kv_bytes)) {
        return false;
    }

    if (qwen36_u64_mul_overflows(recurrent_layers, (uint64_t)spec->linear_head_dim, &n) ||
        qwen36_u64_mul_overflows(n, recurrent_value_width, &n) ||
        !qwen36_scale_bytes(n, sizeof(float), recurrent_state_bytes)) {
        return false;
    }

    if (qwen36_u64_mul_overflows(recurrent_layers, QWEN36_DEFAULT_SSM_D_CONV, &n) ||
        qwen36_u64_mul_overflows(n, qkv_width, &n) ||
        !qwen36_scale_bytes(n, sizeof(float), conv_state_bytes)) {
        return false;
    }

    return true;
}

static rt_context_memory qwen36_rt_estimate_context_memory(rt_backend backend, int ctx_size) {
    if (!qwen36_backend_supported(backend) ||
        ctx_size <= 0 || ctx_size > (int)QWEN36_MODEL_SPEC.max_context) {
        return (rt_context_memory){0};
    }
    uint64_t token_bytes = (uint64_t)ctx_size * sizeof(int);
    uint64_t logits_bytes = (uint64_t)QWEN36_MODEL_SPEC.vocab_size * sizeof(float);
    uint64_t hidden_bytes = (uint64_t)QWEN36_MODEL_SPEC.hidden_size * sizeof(float);
    uint64_t full_kv_bytes = 0;
    uint64_t recurrent_state_bytes = 0;
    uint64_t conv_state_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t total_bytes = 0;
    if (!qwen36_state_layout_bytes(&QWEN36_MODEL_SPEC, ctx_size,
                                   &full_kv_bytes,
                                   &recurrent_state_bytes,
                                   &conv_state_bytes) ||
        qwen36_u64_add_overflows(token_bytes, full_kv_bytes, &state_bytes) ||
        qwen36_u64_add_overflows(state_bytes, recurrent_state_bytes, &state_bytes) ||
        qwen36_u64_add_overflows(state_bytes, conv_state_bytes, &state_bytes) ||
        qwen36_u64_add_overflows(state_bytes, hidden_bytes, &state_bytes) ||
        qwen36_u64_add_overflows(state_bytes, logits_bytes, &total_bytes)) {
        return (rt_context_memory){0};
    }
    return (rt_context_memory){
        .total_bytes = total_bytes,
        .state_bytes = state_bytes,
        .scratch_bytes = logits_bytes,
        .model_private_bytes = recurrent_state_bytes + conv_state_bytes,
        .prefill_cap = 0,
        .model_caps = {0, 0, 0},
    };
}

#define QWEN36_SESSION_PAYLOAD_MAGIC UINT32_C(0x36335751) /* QW36 */
#define QWEN36_SESSION_PAYLOAD_LEGACY_VERSION 1u
#define QWEN36_SESSION_PAYLOAD_V2_VERSION 2u
#define QWEN36_SESSION_PAYLOAD_VERSION 3u
#define QWEN36_SESSION_PAYLOAD_LEGACY_WORDS 12u
#define QWEN36_SESSION_PAYLOAD_WORDS 16u
#define QWEN36_SESSION_PAYLOAD_SECTION_WORDS 4u
#define QWEN36_SESSION_PAYLOAD_MAX_SECTIONS 8u
#define QWEN36_PAYLOAD_SECTION_TOKENS UINT32_C(1)
#define QWEN36_PAYLOAD_SECTION_LOGITS UINT32_C(2)
#define QWEN36_PAYLOAD_SECTION_FULL_KV_F32 UINT32_C(3)
#define QWEN36_PAYLOAD_SECTION_RECURRENT_F32 UINT32_C(4)
#define QWEN36_PAYLOAD_SECTION_CONV_F32 UINT32_C(5)
#define QWEN36_PAYLOAD_SECTION_LAST_HIDDEN_F32 UINT32_C(6)

typedef struct {
    uint32_t type;
    uint32_t flags;
    uint64_t bytes;
} qwen36_payload_section;

static uint64_t qwen36_session_payload_bytes_impl(const qwen36_session *s) {
    if (!s || s->tokens.len < 0) return 0;

    uint64_t bytes = QWEN36_SESSION_PAYLOAD_WORDS * sizeof(uint32_t);
    bool have_hidden = s->last_hidden_valid && s->last_hidden;
    uint32_t sections = 1u + (s->logits_valid ? 1u : 0u) +
                        (have_hidden ? 1u : 0u) +
                        (s->full_kv_state ? 1u : 0u) +
                        (s->recurrent_state ? 1u : 0u) +
                        (s->conv_state ? 1u : 0u);
    uint64_t section_bytes = (uint64_t)sections *
                             QWEN36_SESSION_PAYLOAD_SECTION_WORDS *
                             sizeof(uint32_t);
    if (qwen36_u64_add_overflows(bytes, section_bytes, &bytes)) return 0;
    uint64_t token_bytes = (uint64_t)s->tokens.len * sizeof(uint32_t);
    if (qwen36_u64_add_overflows(bytes, token_bytes, &bytes)) return 0;
    if (s->logits_valid) {
        uint64_t logits_bytes = (uint64_t)s->engine->vocab.n_vocab * sizeof(float);
        if (qwen36_u64_add_overflows(bytes, logits_bytes, &bytes)) return 0;
    }
    if (have_hidden) {
        uint64_t hidden_bytes = (uint64_t)s->engine->spec->hidden_size * sizeof(float);
        if (qwen36_u64_add_overflows(bytes, hidden_bytes, &bytes)) return 0;
    }
    if (s->full_kv_state &&
        qwen36_u64_add_overflows(bytes, s->full_kv_state_bytes, &bytes)) return 0;
    if (s->recurrent_state &&
        qwen36_u64_add_overflows(bytes, s->recurrent_state_bytes, &bytes)) return 0;
    if (s->conv_state &&
        qwen36_u64_add_overflows(bytes, s->conv_state_bytes, &bytes)) return 0;
    return bytes;
}

static void qwen36_le_put32(uint8_t *dst, uint32_t v) {
    dst[0] = (uint8_t)(v & 0xffu);
    dst[1] = (uint8_t)((v >> 8) & 0xffu);
    dst[2] = (uint8_t)((v >> 16) & 0xffu);
    dst[3] = (uint8_t)((v >> 24) & 0xffu);
}

static void qwen36_le_put64_words(uint8_t *dst, uint64_t v) {
    qwen36_le_put32(dst, (uint32_t)(v & UINT64_C(0xffffffff)));
    qwen36_le_put32(dst + 4, (uint32_t)(v >> 32));
}

static uint32_t qwen36_le_get32(const uint8_t *src) {
    return (uint32_t)src[0] |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

static uint64_t qwen36_le_get64_words(const uint8_t *src) {
    return (uint64_t)qwen36_le_get32(src) |
           ((uint64_t)qwen36_le_get32(src + 4) << 32);
}

static bool qwen36_write_exact(FILE *fp, const void *ptr, size_t n) {
    return n == 0 || fwrite(ptr, 1, n, fp) == n;
}

static bool qwen36_read_exact(FILE *fp, void *ptr, size_t n) {
    return n == 0 || fread(ptr, 1, n, fp) == n;
}

static bool qwen36_write_section_header(FILE *fp,
                                        uint32_t type,
                                        uint64_t bytes,
                                        char *err,
                                        size_t errlen) {
    uint8_t section[QWEN36_SESSION_PAYLOAD_SECTION_WORDS * sizeof(uint32_t)];
    qwen36_le_put32(section + 0, type);
    qwen36_le_put32(section + 4, 0);
    qwen36_le_put64_words(section + 8, bytes);
    if (!qwen36_write_exact(fp, section, sizeof(section))) {
        qwen36_set_err(err, errlen, "failed to write Qwen session payload section table");
        return false;
    }
    return true;
}

static bool qwen36_read_payload_blob(FILE *fp, uint64_t bytes, uint8_t **out) {
    if (!out || bytes > SIZE_MAX) return false;
    uint8_t *buf = malloc((size_t)(bytes ? bytes : 1));
    if (!buf) return false;
    if (!qwen36_read_exact(fp, buf, (size_t)bytes)) {
        free(buf);
        return false;
    }
    *out = buf;
    return true;
}

static void qwen36_session_clear_mtp_draft(qwen36_session *s) {
    if (!s) return;
    memset(s->mtp_draft, 0, sizeof(s->mtp_draft));
    memset(s->mtp_draft_margin, 0, sizeof(s->mtp_draft_margin));
    s->mtp_draft_len = 0;
    s->mtp_draft_pos = 0;
}

static void qwen36_session_reset_mtp_accounting(qwen36_session *s) {
    if (!s) return;
    qwen36_session_clear_mtp_draft(s);
    s->mtp_draft_proposed = 0;
    s->mtp_draft_accepted = 0;
    s->mtp_draft_rejected = 0;
    s->mtp_draft_skipped = 0;
}

static void qwen36_session_invalidate_logits(qwen36_session *s) {
    if (!s) return;
    s->logits_valid = false;
    qwen36_session_clear_mtp_draft(s);
}

static void qwen36_session_invalidate_hidden(qwen36_session *s) {
    if (!s) return;
    s->last_hidden_valid = false;
    qwen36_session_clear_mtp_draft(s);
}

static void qwen36_session_clear_state(qwen36_session *s) {
    if (!s) return;
    qwen36_session_clear_mtp_draft(s);
    s->state_valid = false;
    free(s->full_kv_state);
    free(s->recurrent_state);
    free(s->conv_state);
    s->full_kv_state = NULL;
    s->recurrent_state = NULL;
    s->conv_state = NULL;
    s->full_kv_state_bytes = 0;
    s->recurrent_state_bytes = 0;
    s->conv_state_bytes = 0;
}

static int qwen36_sample_candidate_cmp_desc(const void *a, const void *b) {
    const qwen36_sample_candidate *ca = a;
    const qwen36_sample_candidate *cb = b;
    if (ca->logit < cb->logit) return 1;
    if (ca->logit > cb->logit) return -1;
    return ca->id - cb->id;
}

static bool qwen36_logits_logsumexp(const float *logits, int n_vocab, double *out) {
    if (!logits || n_vocab <= 0 || !out) return false;

    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    if (!isfinite(max_logit)) return false;

    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        sum += exp((double)logits[i] - (double)max_logit);
    }
    if (sum <= 0.0 || !isfinite(sum)) return false;
    *out = (double)max_logit + log(sum);
    return true;
}

static bool qwen36_session_token_score(const qwen36_session *s, int token, rt_token_score *out) {
    if (!s || !s->logits_valid || token < 0 || token >= s->engine->vocab.n_vocab || !out) {
        return false;
    }
    double logsum = 0.0;
    if (!qwen36_logits_logsumexp(s->logits, s->engine->vocab.n_vocab, &logsum)) return false;
    out->id = token;
    out->logit = s->logits[token];
    out->logprob = (float)((double)s->logits[token] - logsum);
    return true;
}

static bool qwen36_vec_all_zero(const float *x, uint64_t n) {
    if (!x) return true;
    for (uint64_t i = 0; i < n; i++) {
        if (x[i] != 0.0f) return false;
    }
    return true;
}

static float qwen36_sigmoid_f32(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static float qwen36_silu_f32(float x) {
    return x * qwen36_sigmoid_f32(x);
}

static bool qwen36_silu_mul_checked(qwen36_engine *e,
                                    const float *gate,
                                    const float *up,
                                    uint64_t dim,
                                    float *out,
                                    char *err,
                                    size_t errlen) {
    if (!gate || !up || !out) {
        qwen36_set_err(err, errlen, "invalid Qwen SiLU-mul request");
        return false;
    }
    if (dim == 0) return true;
    if (e && e->backend == RT_BACKEND_METAL && e->metal &&
        qwen36_metal_silu_mul(e->metal, gate, up, dim, out, err, errlen)) {
        e->metal_silu_mul_calls++;
        return true;
    }
    if (e && e->backend == RT_BACKEND_METAL && e->metal) {
        e->metal_silu_mul_fallbacks++;
    }
    for (uint64_t i = 0; i < dim; i++) {
        out[i] = qwen36_silu_f32(gate[i]) * up[i];
    }
    return true;
}

static float qwen36_softplus_f32(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void qwen36_cpu_scratch_free(qwen36_cpu_scratch *sc) {
    if (!sc) return;
    free(sc->hidden);
    free(sc->norm);
    free(sc->mix);
    free(sc->mix2);
    free(sc->ffn_gate);
    free(sc->ffn_up);
    free(sc->ffn_act);
    free(sc->attn);
    free(sc->k);
    free(sc->v);
    free(sc->qkv);
    free(sc->qkv_conv);
    free(sc->z);
    free(sc->beta);
    free(sc->alpha);
    free(sc->core);
    free(sc->scores);
    memset(sc, 0, sizeof(*sc));
}

static bool qwen36_cpu_scratch_init(qwen36_cpu_scratch *sc,
                                    const qwen36_session *s,
                                    char *err,
                                    size_t errlen) {
    memset(sc, 0, sizeof(*sc));
    const qwen36_model_spec *spec = s->engine->spec;
    uint64_t q_gate_dim = (uint64_t)spec->attention_head_dim * spec->attention_heads * 2u;
    uint64_t kv_dim = (uint64_t)spec->attention_head_dim * spec->attention_kv_heads;
    uint64_t attn_dim = (uint64_t)spec->attention_head_dim * spec->attention_heads;
    uint64_t qkv_dim =
        (uint64_t)spec->linear_qk_heads * spec->linear_head_dim * 2u +
        (uint64_t)spec->linear_v_heads * spec->linear_head_dim;
    uint64_t value_dim = (uint64_t)spec->linear_v_heads * spec->linear_head_dim;
    uint64_t mix_dim = q_gate_dim;
    if (mix_dim < spec->hidden_size * 2ull) mix_dim = spec->hidden_size * 2ull;

    if (mix_dim > SIZE_MAX / sizeof(float) ||
        kv_dim > SIZE_MAX / sizeof(float) ||
        attn_dim > SIZE_MAX / sizeof(float) ||
        qkv_dim > SIZE_MAX / sizeof(float) ||
        value_dim > SIZE_MAX / sizeof(float) ||
        (uint64_t)s->ctx_size > SIZE_MAX / sizeof(float)) {
        qwen36_set_err(err, errlen, "Qwen CPU scratch size overflow");
        return false;
    }

    sc->hidden = calloc(spec->hidden_size, sizeof(float));
    sc->norm = calloc(spec->hidden_size, sizeof(float));
    sc->mix = calloc(mix_dim, sizeof(float));
    sc->mix2 = calloc(attn_dim > qkv_dim ? attn_dim : qkv_dim, sizeof(float));
    sc->ffn_gate = calloc(spec->intermediate_size, sizeof(float));
    sc->ffn_up = calloc(spec->intermediate_size, sizeof(float));
    sc->ffn_act = calloc(spec->intermediate_size, sizeof(float));
    sc->attn = calloc(attn_dim, sizeof(float));
    sc->k = calloc(kv_dim, sizeof(float));
    sc->v = calloc(kv_dim, sizeof(float));
    sc->qkv = calloc(qkv_dim, sizeof(float));
    sc->qkv_conv = calloc(qkv_dim, sizeof(float));
    sc->z = calloc(value_dim, sizeof(float));
    sc->beta = calloc(spec->linear_v_heads, sizeof(float));
    sc->alpha = calloc(spec->linear_v_heads, sizeof(float));
    sc->core = calloc(value_dim, sizeof(float));
    sc->scores = calloc((size_t)s->ctx_size, sizeof(float));

    if (!sc->hidden || !sc->norm || !sc->mix || !sc->mix2 ||
        !sc->ffn_gate || !sc->ffn_up || !sc->ffn_act || !sc->attn ||
        !sc->k || !sc->v || !sc->qkv || !sc->qkv_conv || !sc->z ||
        !sc->beta || !sc->alpha || !sc->core || !sc->scores) {
        qwen36_cpu_scratch_free(sc);
        qwen36_set_err(err, errlen, "failed to allocate Qwen CPU scratch");
        return false;
    }

    return true;
}

static bool qwen36_tensor_read_file_checked(const qwen36_engine *e,
                                            const rt_gguf_file *file,
                                            const rt_gguf_tensor *tensor,
                                            uint64_t index,
                                            float *out,
                                            char *err,
                                            size_t errlen) {
    (void)e;
    if (rt_gguf_tensor_read_f32(file, tensor, index, out)) return true;
    qwen36_set_err(err, errlen,
                   "Qwen CPU reference does not support tensor read type %s for '%s'",
                   tensor ? rt_gguf_tensor_type_name(tensor->type) : "(null)",
                   tensor && tensor->name ? tensor->name : "(unknown)");
    return false;
}

static bool qwen36_tensor_read_checked(const qwen36_engine *e,
                                       const rt_gguf_tensor *tensor,
                                       uint64_t index,
                                       float *out,
                                       char *err,
                                       size_t errlen) {
    return qwen36_tensor_read_file_checked(e, e ? e->model : NULL,
                                           tensor, index, out, err, errlen);
}

static bool qwen36_matvec_file_checked(qwen36_engine *e,
                                       const rt_gguf_file *file,
                                       const rt_gguf_tensor *tensor,
                                       const float *x,
                                       uint64_t in_dim,
                                       float *out,
                                       uint64_t out_dim,
                                       char *err,
                                       size_t errlen) {
    (void)e;
    if (!tensor || !x || !out) {
        qwen36_set_err(err, errlen, "invalid Qwen CPU matvec request");
        return false;
    }
    if (qwen36_vec_all_zero(x, in_dim)) {
        memset(out, 0, (size_t)out_dim * sizeof(out[0]));
        return true;
    }
    if (e && e->backend == RT_BACKEND_METAL && e->metal &&
        qwen36_metal_matvec(e->metal, file, tensor, x, in_dim, out, out_dim,
                            err, errlen)) {
        e->metal_matvec_calls++;
        return true;
    }
    if (e && e->backend == RT_BACKEND_METAL && e->metal) {
        e->metal_matvec_fallbacks++;
    }
    if (rt_gguf_tensor_matvec_f32(file, tensor, x, in_dim, out, out_dim)) return true;
    if (file && tensor->ndim == 2 && tensor->dim[0] == in_dim &&
        tensor->dim[1] >= out_dim && out_dim <= UINT64_MAX / in_dim) {
        for (uint64_t row = 0; row < out_dim; row++) {
            double sum = 0.0;
            uint64_t base = row * in_dim;
            for (uint64_t col = 0; col < in_dim; col++) {
                float w = 0.0f;
                if (!qwen36_tensor_read_file_checked(e, file, tensor,
                                                     base + col, &w,
                                                     err, errlen)) {
                    return false;
                }
                sum += (double)w * (double)x[col];
            }
            out[row] = (float)sum;
        }
        return true;
    }
    qwen36_set_err(err, errlen,
                   "Qwen CPU reference does not support 2D matvec type %s for tensor '%s'",
                   rt_gguf_tensor_type_name(tensor->type),
                   tensor->name ? tensor->name : "(unknown)");
    return false;
}

static bool qwen36_matvec_checked(qwen36_engine *e,
                                  const rt_gguf_tensor *tensor,
                                  const float *x,
                                  uint64_t in_dim,
                                  float *out,
                                  uint64_t out_dim,
                                  char *err,
                                  size_t errlen) {
    return qwen36_matvec_file_checked(e, e ? e->model : NULL, tensor, x,
                                      in_dim, out, out_dim, err, errlen);
}

static bool qwen36_read_embedding(qwen36_engine *e,
                                  int token,
                                  float *out,
                                  char *err,
                                  size_t errlen) {
    const uint64_t hidden = e->spec->hidden_size;
    const rt_gguf_tensor *t = e->weights.tok_embd;
    if (!t || token < 0 || (uint64_t)token >= t->dim[1] ||
        (uint64_t)token > (UINT64_MAX / hidden)) {
        qwen36_set_err(err, errlen, "invalid Qwen token embedding request");
        return false;
    }
    uint64_t base = (uint64_t)token * hidden;
    for (uint64_t i = 0; i < hidden; i++) {
        if (!qwen36_tensor_read_checked(e, t, base + i, out + i, err, errlen)) return false;
    }
    return true;
}

static bool qwen36_rms_norm_file(qwen36_engine *e,
                                 const rt_gguf_file *file,
                                 const rt_gguf_tensor *weight,
                                 const float *x,
                                 uint64_t dim,
                                 float *out,
                                 bool add_one_to_weight,
                                 char *err,
                                 size_t errlen) {
    if (!weight || !x || !out || weight->ndim != 1 || weight->dim[0] != dim) {
        qwen36_set_err(err, errlen, "invalid Qwen RMSNorm request");
        return false;
    }
    if (qwen36_vec_all_zero(x, dim)) {
        memset(out, 0, (size_t)dim * sizeof(out[0]));
        return true;
    }
    if (e && e->backend == RT_BACKEND_METAL && e->metal &&
        qwen36_metal_rms_norm(e->metal, file, weight, x, dim, out,
                              add_one_to_weight, err, errlen)) {
        e->metal_rms_norm_calls++;
        return true;
    }
    if (e && e->backend == RT_BACKEND_METAL && e->metal) {
        e->metal_rms_norm_fallbacks++;
    }
    double ss = 0.0;
    for (uint64_t i = 0; i < dim; i++) ss += (double)x[i] * (double)x[i];
    float inv = 1.0f / sqrtf((float)(ss / (double)dim) + 1.0e-6f);
    for (uint64_t i = 0; i < dim; i++) {
        float w = 0.0f;
        if (!qwen36_tensor_read_file_checked(e, file, weight, i, &w, err, errlen)) return false;
        out[i] = x[i] * inv * (add_one_to_weight ? (1.0f + w) : w);
    }
    return true;
}

static bool qwen36_rms_norm(qwen36_engine *e,
                            const rt_gguf_tensor *weight,
                            const float *x,
                            uint64_t dim,
                            float *out,
                            bool add_one_to_weight,
                            char *err,
                            size_t errlen) {
    return qwen36_rms_norm_file(e, e ? e->model : NULL, weight, x, dim, out,
                                add_one_to_weight, err, errlen);
}

static bool qwen36_l2norm_inplace(qwen36_engine *e, float *x, uint64_t dim) {
    if (!x || dim == 0) return false;
    if (qwen36_vec_all_zero(x, dim)) return false;
    if (e && e->backend == RT_BACKEND_METAL && e->metal &&
        qwen36_metal_l2_norm(e->metal, x, dim, x, NULL, 0)) {
        e->metal_l2_norm_calls++;
        return true;
    }
    if (e && e->backend == RT_BACKEND_METAL && e->metal) {
        e->metal_l2_norm_fallbacks++;
    }
    double ss = 0.0;
    for (uint64_t i = 0; i < dim; i++) ss += (double)x[i] * (double)x[i];
    if (ss == 0.0) return false;
    float inv = 1.0f / sqrtf((float)ss + 1.0e-6f);
    for (uint64_t i = 0; i < dim; i++) x[i] *= inv;
    return true;
}

static float qwen36_gelu_f32(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

static bool qwen36_add_bias_file(qwen36_engine *e,
                                 const rt_gguf_file *file,
                                 const rt_gguf_tensor *bias,
                                 float *x,
                                 uint64_t dim,
                                 char *err,
                                 size_t errlen) {
    if (!bias) return true;
    if (!x || bias->ndim != 1 || bias->dim[0] != dim) {
        qwen36_set_err(err, errlen, "invalid Qwen bias add request");
        return false;
    }
    for (uint64_t i = 0; i < dim; i++) {
        float b = 0.0f;
        if (!qwen36_tensor_read_file_checked(e, file, bias, i, &b, err, errlen)) return false;
        x[i] += b;
    }
    return true;
}

static bool qwen36_mmproj_patch_weight(qwen36_engine *e,
                                       const qwen36_mmproj *mm,
                                       uint32_t embd,
                                       uint32_t channel,
                                       uint32_t py,
                                       uint32_t px,
                                       float *out,
                                       char *err,
                                       size_t errlen) {
    const rt_gguf_tensor *t = mm->patch_embd;
    const uint64_t patch = mm->patch_size;
    uint64_t index = 0;
    if (t->dim[0] == patch && t->dim[1] == patch && t->dim[2] == 3 &&
        t->dim[3] == mm->n_embd) {
        index = (((uint64_t)embd * 3u + channel) * patch + py) * patch + px;
    } else {
        index = (((uint64_t)py * patch + px) * 3u + channel) * mm->n_embd + embd;
    }
    return qwen36_tensor_read_file_checked(e, e->mmproj, t, index, out, err, errlen);
}

static bool qwen36_mmproj_patch_embed(qwen36_engine *e,
                                      const float *patches,
                                      uint64_t n_patches,
                                      float *out,
                                      char *err,
                                      size_t errlen) {
    qwen36_mmproj *mm = &e->mmproj_info;
    uint64_t patch_elems = (uint64_t)mm->patch_size * mm->patch_size * 3u;
    for (uint64_t ip = 0; ip < n_patches; ip++) {
        const float *patch = patches + ip * patch_elems;
        float *dst = out + ip * mm->n_embd;
        if (qwen36_vec_all_zero(patch, patch_elems)) {
            memset(dst, 0, (size_t)mm->n_embd * sizeof(dst[0]));
        } else {
            for (uint32_t d = 0; d < mm->n_embd; d++) {
                double sum = 0.0;
                uint64_t pix = 0;
                for (uint32_t py = 0; py < mm->patch_size; py++) {
                    for (uint32_t px = 0; px < mm->patch_size; px++) {
                        for (uint32_t c = 0; c < 3u; c++, pix++) {
                            float w = 0.0f;
                            if (!qwen36_mmproj_patch_weight(e, mm, d, c, py, px,
                                                            &w, err, errlen)) {
                                return false;
                            }
                            sum += (double)w * (double)patch[pix];
                        }
                    }
                }
                dst[d] = (float)sum;
            }
        }
        if (!qwen36_add_bias_file(e, e->mmproj, mm->patch_bias, dst,
                                  mm->n_embd, err, errlen)) {
            return false;
        }
    }
    return true;
}

static void qwen36_softmax_inplace(float *x, uint32_t n) {
    if (!x || n == 0) return;
    float max_v = x[0];
    for (uint32_t i = 1; i < n; i++) {
        if (x[i] > max_v) max_v = x[i];
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_v);
        sum += x[i];
    }
    if (sum <= 0.0 || !isfinite(sum)) {
        float uniform = 1.0f / (float)n;
        for (uint32_t i = 0; i < n; i++) x[i] = uniform;
        return;
    }
    float inv = (float)(1.0 / sum);
    for (uint32_t i = 0; i < n; i++) x[i] *= inv;
}

static bool qwen36_mmproj_eval_layer(qwen36_engine *e,
                                     const qwen36_mmproj_layer_tensors *layer,
                                     uint32_t n_tokens,
                                     float *x,
                                     float *norm,
                                     float *q,
                                     float *k,
                                     float *v,
                                     float *attn,
                                     float *proj,
                                     float *qkv,
                                     float *scores,
                                     float *ffn_gate,
                                     float *ffn_up,
                                     float *ffn_act,
                                     char *err,
                                     size_t errlen) {
    qwen36_mmproj *mm = &e->mmproj_info;
    const uint32_t n_embd = mm->n_embd;
    if (qwen36_vec_all_zero(x, (uint64_t)n_tokens * n_embd)) return true;
    if (mm->n_head == 0 || n_embd % mm->n_head != 0) {
        qwen36_set_err(err, errlen, "invalid Qwen mmproj attention head layout");
        return false;
    }
    const uint32_t head_dim = n_embd / mm->n_head;
    const float scale = 1.0f / sqrtf((float)head_dim);

    for (uint32_t t = 0; t < n_tokens; t++) {
        if (!qwen36_rms_norm_file(e, e->mmproj, layer->ln1,
                                  x + (uint64_t)t * n_embd, n_embd,
                                  norm + (uint64_t)t * n_embd, false,
                                  err, errlen)) {
            return false;
        }
        if (layer->attn_qkv) {
            if (!qwen36_matvec_file_checked(e, e->mmproj, layer->attn_qkv,
                                            norm + (uint64_t)t * n_embd,
                                            n_embd, qkv, (uint64_t)n_embd * 3u,
                                            err, errlen)) {
                return false;
            }
            memcpy(q + (uint64_t)t * n_embd, qkv,
                   (size_t)n_embd * sizeof(q[0]));
            memcpy(k + (uint64_t)t * n_embd, qkv + n_embd,
                   (size_t)n_embd * sizeof(k[0]));
            memcpy(v + (uint64_t)t * n_embd, qkv + (uint64_t)n_embd * 2u,
                   (size_t)n_embd * sizeof(v[0]));
        } else {
            if (!qwen36_matvec_file_checked(e, e->mmproj, layer->attn_q,
                                            norm + (uint64_t)t * n_embd,
                                            n_embd, q + (uint64_t)t * n_embd,
                                            n_embd, err, errlen) ||
                !qwen36_matvec_file_checked(e, e->mmproj, layer->attn_k,
                                            norm + (uint64_t)t * n_embd,
                                            n_embd, k + (uint64_t)t * n_embd,
                                            n_embd, err, errlen) ||
                !qwen36_matvec_file_checked(e, e->mmproj, layer->attn_v,
                                            norm + (uint64_t)t * n_embd,
                                            n_embd, v + (uint64_t)t * n_embd,
                                            n_embd, err, errlen)) {
                return false;
            }
        }
    }

    memset(attn, 0, (size_t)n_tokens * n_embd * sizeof(attn[0]));
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t h = 0; h < mm->n_head; h++) {
            const float *qh = q + (uint64_t)t * n_embd + (uint64_t)h * head_dim;
            for (uint32_t u = 0; u < n_tokens; u++) {
                const float *kh = k + (uint64_t)u * n_embd + (uint64_t)h * head_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += (double)qh[d] * (double)kh[d];
                }
                scores[u] = (float)dot * scale;
            }
            qwen36_softmax_inplace(scores, n_tokens);
            float *oh = attn + (uint64_t)t * n_embd + (uint64_t)h * head_dim;
            for (uint32_t u = 0; u < n_tokens; u++) {
                const float p = scores[u];
                const float *vh = v + (uint64_t)u * n_embd + (uint64_t)h * head_dim;
                for (uint32_t d = 0; d < head_dim; d++) oh[d] += p * vh[d];
            }
        }
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        if (!qwen36_matvec_file_checked(e, e->mmproj, layer->attn_out,
                                        attn + (uint64_t)t * n_embd,
                                        n_embd, proj, n_embd, err, errlen)) {
            return false;
        }
        for (uint32_t d = 0; d < n_embd; d++) x[(uint64_t)t * n_embd + d] += proj[d];

        if (!qwen36_rms_norm_file(e, e->mmproj, layer->ln2,
                                  x + (uint64_t)t * n_embd, n_embd,
                                  norm + (uint64_t)t * n_embd, false,
                                  err, errlen)) {
            return false;
        }
        if (layer->ffn_gate) {
            if (!qwen36_matvec_file_checked(e, e->mmproj, layer->ffn_gate,
                                            norm + (uint64_t)t * n_embd,
                                            n_embd, ffn_gate, mm->n_ff,
                                            err, errlen) ||
                !qwen36_matvec_file_checked(e, e->mmproj, layer->ffn_up,
                                            norm + (uint64_t)t * n_embd,
                                            n_embd, ffn_up, mm->n_ff,
                                            err, errlen)) {
                return false;
            }
            if (!qwen36_silu_mul_checked(e, ffn_gate, ffn_up, mm->n_ff,
                                         ffn_act, err, errlen)) {
                return false;
            }
        } else {
            if (!qwen36_matvec_file_checked(e, e->mmproj, layer->ffn_up,
                                            norm + (uint64_t)t * n_embd,
                                            n_embd, ffn_act, mm->n_ff,
                                            err, errlen)) {
                return false;
            }
            for (uint32_t i = 0; i < mm->n_ff; i++) ffn_act[i] = qwen36_gelu_f32(ffn_act[i]);
        }
        if (!qwen36_matvec_file_checked(e, e->mmproj, layer->ffn_down,
                                        ffn_act, mm->n_ff, proj, n_embd,
                                        err, errlen)) {
            return false;
        }
        for (uint32_t d = 0; d < n_embd; d++) x[(uint64_t)t * n_embd + d] += proj[d];
    }
    return true;
}

static bool qwen36_mmproj_project_merged(qwen36_engine *e,
                                         const float *vision,
                                         uint32_t grid_h,
                                         uint32_t grid_w,
                                         qwen36_runtime_vision_embedding *out,
                                         char *err,
                                         size_t errlen) {
    qwen36_mmproj *mm = &e->mmproj_info;
    const uint32_t merge = mm->spatial_merge_size;
    const uint32_t out_h = grid_h / merge;
    const uint32_t out_w = grid_w / merge;
    const uint32_t n_out = out_h * out_w;
    const uint64_t merge_area = (uint64_t)merge * merge;
    const uint64_t merge_dim = merge_area * mm->n_embd;
    const uint32_t hidden = e->spec->hidden_size;
    if (mm->mm_0_b->ndim != 1 ||
        mm->mm_0_w->ndim != 2 ||
        mm->mm_0_w->dim[0] != merge_dim ||
        mm->mm_0_w->dim[1] != mm->mm_0_b->dim[0] ||
        mm->mm_1_w->ndim != 2 ||
        mm->mm_1_w->dim[0] != mm->mm_0_b->dim[0] ||
        mm->mm_1_w->dim[1] != hidden) {
        qwen36_set_err(err, errlen, "unsupported Qwen mmproj merger tensor layout");
        return false;
    }
    uint64_t mlp_dim = mm->mm_0_b->dim[0];
    if (merge_dim > SIZE_MAX / sizeof(float) ||
        mlp_dim > SIZE_MAX / sizeof(float) ||
        (uint64_t)n_out > SIZE_MAX / hidden ||
        (uint64_t)n_out * hidden > SIZE_MAX / sizeof(float)) {
        qwen36_set_err(err, errlen, "Qwen mmproj merger allocation overflow");
        return false;
    }
    float *merge_in = calloc((size_t)merge_dim, sizeof(merge_in[0]));
    float *mlp = calloc((size_t)mlp_dim, sizeof(mlp[0]));
    float *data = calloc((size_t)n_out * hidden, sizeof(data[0]));
    if (!merge_in || !mlp || !data) {
        free(merge_in);
        free(mlp);
        free(data);
        qwen36_set_err(err, errlen, "failed to allocate Qwen mmproj merger buffers");
        return false;
    }

    for (uint32_t oy = 0; oy < out_h; oy++) {
        for (uint32_t ox = 0; ox < out_w; ox++) {
            uint64_t mpos = 0;
            for (uint32_t dy = 0; dy < merge; dy++) {
                for (uint32_t dx = 0; dx < merge; dx++) {
                    uint32_t sy = oy * merge + dy;
                    uint32_t sx = ox * merge + dx;
                    const float *src = vision + ((uint64_t)sy * grid_w + sx) * mm->n_embd;
                    memcpy(merge_in + mpos, src, (size_t)mm->n_embd * sizeof(src[0]));
                    mpos += mm->n_embd;
                }
            }
            if (!qwen36_matvec_file_checked(e, e->mmproj, mm->mm_0_w,
                                            merge_in, merge_dim, mlp, mlp_dim,
                                            err, errlen) ||
                !qwen36_add_bias_file(e, e->mmproj, mm->mm_0_b, mlp, mlp_dim,
                                      err, errlen)) {
                free(merge_in);
                free(mlp);
                free(data);
                return false;
            }
            for (uint64_t i = 0; i < mlp_dim; i++) mlp[i] = qwen36_gelu_f32(mlp[i]);
            float *row = data + ((uint64_t)oy * out_w + ox) * hidden;
            if (!qwen36_matvec_file_checked(e, e->mmproj, mm->mm_1_w,
                                            mlp, mlp_dim, row, hidden,
                                            err, errlen) ||
                !qwen36_add_bias_file(e, e->mmproj, mm->mm_1_b, row, hidden,
                                      err, errlen)) {
                free(merge_in);
                free(mlp);
                free(data);
                return false;
            }
        }
    }

    free(merge_in);
    free(mlp);
    out->n_tokens = n_out;
    out->hidden_size = hidden;
    out->data = data;
    return true;
}

int qwen36_runtime_embed_image_patches_f32(rt_engine *engine,
                                           const float *patches,
                                           uint32_t grid_h,
                                           uint32_t grid_w,
                                           qwen36_runtime_vision_embedding *out,
                                           char *err,
                                           size_t errlen) {
    if (!engine || !engine->impl || engine->ops != qwen36_runtime_ops() ||
        !patches || !out) {
        qwen36_set_err(err, errlen, "invalid Qwen vision embedding request");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    qwen36_engine *e = engine->impl;
    qwen36_mmproj *mm = &e->mmproj_info;
    if (!mm->loaded) {
        qwen36_set_err(err, errlen, "Qwen vision embedding requires a loaded mmproj");
        return -1;
    }
    if (grid_h == 0 || grid_w == 0 ||
        grid_h % mm->spatial_merge_size != 0 ||
        grid_w % mm->spatial_merge_size != 0) {
        qwen36_set_err(err, errlen, "Qwen vision patch grid must be non-empty and divisible by spatial merge size");
        return -1;
    }
    uint64_t n_patches = (uint64_t)grid_h * grid_w;
    uint64_t patch_elems = (uint64_t)mm->patch_size * mm->patch_size * 3u;
    uint64_t seq_elems = n_patches * mm->n_embd;
    if ((grid_w != 0 && n_patches / grid_w != grid_h) ||
        patch_elems / 3u / mm->patch_size != mm->patch_size ||
        seq_elems / mm->n_embd != n_patches ||
        seq_elems > SIZE_MAX / sizeof(float) ||
        (uint64_t)mm->n_embd * 3u > SIZE_MAX / sizeof(float) ||
        (uint64_t)mm->n_ff > SIZE_MAX / sizeof(float)) {
        qwen36_set_err(err, errlen, "Qwen vision embedding allocation overflow");
        return -1;
    }

    float *x = calloc((size_t)seq_elems, sizeof(x[0]));
    float *norm = calloc((size_t)seq_elems, sizeof(norm[0]));
    float *q = calloc((size_t)seq_elems, sizeof(q[0]));
    float *k = calloc((size_t)seq_elems, sizeof(k[0]));
    float *v = calloc((size_t)seq_elems, sizeof(v[0]));
    float *attn = calloc((size_t)seq_elems, sizeof(attn[0]));
    float *proj = calloc((size_t)mm->n_embd, sizeof(proj[0]));
    float *qkv = calloc((size_t)mm->n_embd * 3u, sizeof(qkv[0]));
    float *scores = calloc((size_t)n_patches, sizeof(scores[0]));
    float *ffn_gate = calloc((size_t)mm->n_ff, sizeof(ffn_gate[0]));
    float *ffn_up = calloc((size_t)mm->n_ff, sizeof(ffn_up[0]));
    float *ffn_act = calloc((size_t)mm->n_ff, sizeof(ffn_act[0]));
    if (!x || !norm || !q || !k || !v || !attn || !proj || !qkv ||
        !scores || !ffn_gate || !ffn_up || !ffn_act) {
        free(x); free(norm); free(q); free(k); free(v); free(attn);
        free(proj); free(qkv); free(scores); free(ffn_gate); free(ffn_up);
        free(ffn_act);
        qwen36_set_err(err, errlen, "failed to allocate Qwen vision embedding buffers");
        return -1;
    }

    bool ok = qwen36_mmproj_patch_embed(e, patches, n_patches, x, err, errlen);
    for (uint32_t il = 0; ok && il < mm->n_layer; il++) {
        ok = qwen36_mmproj_eval_layer(e, &mm->layer[il], (uint32_t)n_patches,
                                      x, norm, q, k, v, attn, proj, qkv,
                                      scores, ffn_gate, ffn_up, ffn_act,
                                      err, errlen);
    }
    if (ok) {
        ok = qwen36_mmproj_project_merged(e, x, grid_h, grid_w, out, err, errlen);
    }

    free(x); free(norm); free(q); free(k); free(v); free(attn);
    free(proj); free(qkv); free(scores); free(ffn_gate); free(ffn_up);
    free(ffn_act);
    return ok ? 0 : -1;
}

static void qwen36_apply_rope(float *x, uint32_t head_dim, uint32_t rotary_dim, int pos) {
    const double theta = 10000000.0;
    if (!x || rotary_dim == 0 || rotary_dim > head_dim) return;
    uint32_t half = rotary_dim / 2u;
    for (uint32_t i = 0; i < half; i++) {
        double inv_freq = 1.0 / pow(theta, (double)(2u * i) / (double)rotary_dim);
        double a = (double)pos * inv_freq;
        float c = (float)cos(a);
        float s = (float)sin(a);
        float x0 = x[i];
        float x1 = x[i + half];
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

static float *qwen36_session_full_kv(qwen36_session *s, bool allocate) {
    if (s->full_kv_state) return (float *)s->full_kv_state;
    if (!allocate) return NULL;
    uint64_t full_kv = 0, recurrent = 0, conv = 0;
    if (!qwen36_state_layout_bytes(s->engine->spec, s->ctx_size,
                                   &full_kv, &recurrent, &conv) ||
        full_kv > SIZE_MAX) {
        return NULL;
    }
    s->full_kv_state = calloc(1, (size_t)full_kv);
    if (!s->full_kv_state) return NULL;
    s->full_kv_state_bytes = full_kv;
    return (float *)s->full_kv_state;
}

static float *qwen36_session_recurrent(qwen36_session *s, bool allocate) {
    if (s->recurrent_state) return (float *)s->recurrent_state;
    if (!allocate) return NULL;
    uint64_t full_kv = 0, recurrent = 0, conv = 0;
    if (!qwen36_state_layout_bytes(s->engine->spec, s->ctx_size,
                                   &full_kv, &recurrent, &conv) ||
        recurrent > SIZE_MAX) {
        return NULL;
    }
    s->recurrent_state = calloc(1, (size_t)recurrent);
    if (!s->recurrent_state) return NULL;
    s->recurrent_state_bytes = recurrent;
    return (float *)s->recurrent_state;
}

static float *qwen36_session_conv(qwen36_session *s, bool allocate) {
    if (s->conv_state) return (float *)s->conv_state;
    if (!allocate) return NULL;
    uint64_t full_kv = 0, recurrent = 0, conv = 0;
    if (!qwen36_state_layout_bytes(s->engine->spec, s->ctx_size,
                                   &full_kv, &recurrent, &conv) ||
        conv > SIZE_MAX) {
        return NULL;
    }
    s->conv_state = calloc(1, (size_t)conv);
    if (!s->conv_state) return NULL;
    s->conv_state_bytes = conv;
    return (float *)s->conv_state;
}

static bool qwen36_eval_full_attention(qwen36_session *s,
                                       const qwen36_layer_tensors *layer,
                                       uint32_t full_idx,
                                       int pos,
                                       qwen36_cpu_scratch *sc,
                                       float *out,
                                       char *err,
                                       size_t errlen) {
    qwen36_engine *e = s->engine;
    const qwen36_model_spec *spec = e->spec;
    const uint32_t head_dim = spec->attention_head_dim;
    const uint32_t n_heads = spec->attention_heads;
    const uint32_t n_kv = spec->attention_kv_heads;
    const uint32_t kv_groups = n_heads / n_kv;
    const uint64_t q_dim = (uint64_t)n_heads * head_dim;
    const uint64_t q_gate_dim = q_dim * 2u;
    const uint64_t kv_dim = (uint64_t)n_kv * head_dim;

    memset(out, 0, spec->hidden_size * sizeof(out[0]));
    if (!qwen36_matvec_checked(e, layer->attn_q, sc->norm, spec->hidden_size,
                               sc->mix, q_gate_dim, err, errlen) ||
        !qwen36_matvec_checked(e, layer->attn_k, sc->norm, spec->hidden_size,
                               sc->k, kv_dim, err, errlen) ||
        !qwen36_matvec_checked(e, layer->attn_v, sc->norm, spec->hidden_size,
                               sc->v, kv_dim, err, errlen)) {
        return false;
    }

    float *query = sc->mix;
    float *gate = sc->mix + q_dim;
    for (uint32_t h = 0; h < n_heads; h++) {
        if (!qwen36_rms_norm(e, layer->attn_q_norm, query + (uint64_t)h * head_dim,
                             head_dim, query + (uint64_t)h * head_dim, true,
                             err, errlen)) {
            return false;
        }
        qwen36_apply_rope(query + (uint64_t)h * head_dim, head_dim,
                          head_dim / 4u, pos);
    }
    for (uint32_t h = 0; h < n_kv; h++) {
        if (!qwen36_rms_norm(e, layer->attn_k_norm, sc->k + (uint64_t)h * head_dim,
                             head_dim, sc->k + (uint64_t)h * head_dim, true,
                             err, errlen)) {
            return false;
        }
        qwen36_apply_rope(sc->k + (uint64_t)h * head_dim, head_dim,
                          head_dim / 4u, pos);
    }

    bool need_kv = !qwen36_vec_all_zero(sc->k, kv_dim) || !qwen36_vec_all_zero(sc->v, kv_dim);
    float *state = qwen36_session_full_kv(s, need_kv);
    if (need_kv && !state) {
        qwen36_set_err(err, errlen, "failed to allocate Qwen full-attention KV state");
        return false;
    }
    uint64_t layer_stride = (uint64_t)s->ctx_size * 2u * kv_dim;
    float *keys = state ? state + (uint64_t)full_idx * layer_stride : NULL;
    float *values = keys ? keys + (uint64_t)s->ctx_size * kv_dim : NULL;
    if (state) {
        memcpy(keys + (uint64_t)pos * kv_dim, sc->k, (size_t)kv_dim * sizeof(float));
        memcpy(values + (uint64_t)pos * kv_dim, sc->v, (size_t)kv_dim * sizeof(float));
    } else {
        memset(sc->attn, 0, (size_t)q_dim * sizeof(sc->attn[0]));
        return qwen36_matvec_checked(e, layer->attn_output, sc->attn, q_dim,
                                     out, spec->hidden_size, err, errlen);
    }

    for (uint32_t h = 0; h < n_heads; h++) {
        uint32_t kh = h / kv_groups;
        const float *qh = query + (uint64_t)h * head_dim;
        const float *gh = gate + (uint64_t)h * head_dim;
        const float *kh_base = keys + (uint64_t)kh * head_dim;
        const float *vh_base = values + (uint64_t)kh * head_dim;
        float *oh = sc->attn + (uint64_t)h * head_dim;
        if (e->backend == RT_BACKEND_METAL && e->metal &&
            qwen36_metal_full_attention_head(e->metal, qh, gh, kh_base,
                                             vh_base, (uint64_t)pos + 1u,
                                             kv_dim, head_dim, oh,
                                             NULL, 0)) {
            e->metal_full_attention_calls++;
            continue;
        }
        if (e->backend == RT_BACKEND_METAL && e->metal) {
            e->metal_full_attention_fallbacks++;
        }
        float max_score = -FLT_MAX;
        for (int t = 0; t <= pos; t++) {
            const float *kt = keys + ((uint64_t)t * kv_dim) + (uint64_t)kh * head_dim;
            double dot = 0.0;
            for (uint32_t d = 0; d < head_dim; d++) dot += (double)qh[d] * (double)kt[d];
            float score = (float)(dot / sqrt((double)head_dim));
            sc->scores[t] = score;
            if (score > max_score) max_score = score;
        }
        double denom = 0.0;
        for (int t = 0; t <= pos; t++) {
            double p = exp((double)sc->scores[t] - (double)max_score);
            sc->scores[t] = (float)p;
            denom += p;
        }
        if (denom <= 0.0 || !isfinite(denom)) {
            qwen36_set_err(err, errlen, "invalid Qwen attention softmax");
            return false;
        }
        for (uint32_t d = 0; d < head_dim; d++) {
            double sum = 0.0;
            for (int t = 0; t <= pos; t++) {
                const float *vt = values + ((uint64_t)t * kv_dim) + (uint64_t)kh * head_dim;
                sum += ((double)sc->scores[t] / denom) * (double)vt[d];
            }
            oh[d] = (float)sum * qwen36_sigmoid_f32(gh[d]);
        }
    }

    return qwen36_matvec_checked(e, layer->attn_output, sc->attn, q_dim,
                                 out, spec->hidden_size, err, errlen);
}

static bool qwen36_gated_delta_head(qwen36_engine *e,
                                    float *state_h,
                                    const float *query,
                                    const float *key,
                                    const float *value,
                                    uint32_t head_dim,
                                    float decay,
                                    float beta,
                                    float qscale,
                                    float *out,
                                    char *err,
                                    size_t errlen) {
    if (!state_h || !query || !key || !value || !out || head_dim == 0 ||
        head_dim > 256u) {
        qwen36_set_err(err, errlen, "invalid Qwen Gated DeltaNet head request");
        return false;
    }
    if (e && e->backend == RT_BACKEND_METAL && e->metal &&
        qwen36_metal_gated_delta_head(e->metal, state_h, query, key, value,
                                      head_dim, decay, beta, qscale, out,
                                      err, errlen)) {
        e->metal_gated_delta_calls++;
        return true;
    }
    if (e && e->backend == RT_BACKEND_METAL && e->metal) {
        e->metal_gated_delta_fallbacks++;
    }

    float kv_mem[256];
    float delta[256];
    for (uint32_t kd = 0; kd < head_dim; kd++) {
        for (uint32_t vd = 0; vd < head_dim; vd++) {
            state_h[(uint64_t)kd * head_dim + vd] *= decay;
        }
    }
    for (uint32_t vd = 0; vd < head_dim; vd++) {
        double sum = 0.0;
        for (uint32_t kd = 0; kd < head_dim; kd++) {
            sum += (double)state_h[(uint64_t)kd * head_dim + vd] *
                   (double)key[kd];
        }
        kv_mem[vd] = (float)sum;
        delta[vd] = (value[vd] - kv_mem[vd]) * beta;
    }
    for (uint32_t kd = 0; kd < head_dim; kd++) {
        for (uint32_t vd = 0; vd < head_dim; vd++) {
            state_h[(uint64_t)kd * head_dim + vd] += key[kd] * delta[vd];
        }
    }
    for (uint32_t vd = 0; vd < head_dim; vd++) {
        double sum = 0.0;
        for (uint32_t kd = 0; kd < head_dim; kd++) {
            sum += (double)state_h[(uint64_t)kd * head_dim + vd] *
                   (double)(query[kd] * qscale);
        }
        out[vd] = (float)sum;
    }
    return true;
}

static bool qwen36_eval_recurrent(qwen36_session *s,
                                  const qwen36_layer_tensors *layer,
                                  uint32_t recurrent_idx,
                                  qwen36_cpu_scratch *sc,
                                  float *out,
                                  char *err,
                                  size_t errlen) {
    qwen36_engine *e = s->engine;
    const qwen36_model_spec *spec = e->spec;
    const uint32_t head_dim = spec->linear_head_dim;
    const uint32_t n_k_heads = spec->linear_qk_heads;
    const uint32_t n_v_heads = spec->linear_v_heads;
    const uint32_t repeat = n_v_heads / n_k_heads;
    const uint64_t key_dim = (uint64_t)n_k_heads * head_dim;
    const uint64_t value_dim = (uint64_t)n_v_heads * head_dim;
    const uint64_t qkv_dim = key_dim * 2u + value_dim;
    const uint64_t conv_kernel = QWEN36_DEFAULT_SSM_D_CONV;

    memset(out, 0, spec->hidden_size * sizeof(out[0]));
    if (!qwen36_matvec_checked(e, layer->attn_qkv, sc->norm, spec->hidden_size,
                               sc->qkv, qkv_dim, err, errlen) ||
        !qwen36_matvec_checked(e, layer->attn_gate, sc->norm, spec->hidden_size,
                               sc->z, value_dim, err, errlen) ||
        !qwen36_matvec_checked(e, layer->ssm_beta, sc->norm, spec->hidden_size,
                               sc->beta, n_v_heads, err, errlen) ||
        !qwen36_matvec_checked(e, layer->ssm_alpha, sc->norm, spec->hidden_size,
                               sc->alpha, n_v_heads, err, errlen)) {
        return false;
    }

    bool need_conv = !qwen36_vec_all_zero(sc->qkv, qkv_dim) || s->conv_state != NULL;
    float *conv_state = qwen36_session_conv(s, need_conv);
    if (need_conv && !conv_state) {
        qwen36_set_err(err, errlen, "failed to allocate Qwen convolution state");
        return false;
    }
    float *conv = conv_state ? conv_state + (uint64_t)recurrent_idx * conv_kernel * qkv_dim : NULL;
    if (!conv && qwen36_vec_all_zero(sc->qkv, qkv_dim)) {
        memset(sc->qkv_conv, 0, (size_t)qkv_dim * sizeof(sc->qkv_conv[0]));
    } else {
        for (uint64_t d = 0; d < qkv_dim; d++) {
            double sum = 0.0;
            for (uint64_t k = 0; k < conv_kernel; k++) {
                float x = 0.0f;
                if (k + 1u < conv_kernel) {
                    x = conv ? conv[(k + 1u) * qkv_dim + d] : 0.0f;
                } else {
                    x = sc->qkv[d];
                }
                float w = 0.0f;
                if (!qwen36_tensor_read_checked(e, layer->ssm_conv1d,
                                                d * conv_kernel + k, &w,
                                                err, errlen)) {
                    return false;
                }
                sum += (double)x * (double)w;
            }
            sc->qkv_conv[d] = qwen36_silu_f32((float)sum);
        }
        if (conv) {
            memmove(conv, conv + qkv_dim, (size_t)(conv_kernel - 1u) * (size_t)qkv_dim * sizeof(float));
            memcpy(conv + (conv_kernel - 1u) * qkv_dim, sc->qkv, (size_t)qkv_dim * sizeof(float));
        }
    }

    float *query = sc->qkv_conv;
    float *key = sc->qkv_conv + key_dim;
    float *value = sc->qkv_conv + key_dim * 2u;
    for (uint32_t h = 0; h < n_k_heads; h++) {
        qwen36_l2norm_inplace(e, query + (uint64_t)h * head_dim, head_dim);
        qwen36_l2norm_inplace(e, key + (uint64_t)h * head_dim, head_dim);
    }
    for (uint32_t h = 0; h < n_v_heads; h++) {
        sc->beta[h] = qwen36_sigmoid_f32(sc->beta[h]);
        float dt = 0.0f;
        float a = 0.0f;
        if (!qwen36_tensor_read_checked(e, layer->ssm_dt, h, &dt, err, errlen) ||
            !qwen36_tensor_read_checked(e, layer->ssm_a, h, &a, err, errlen)) {
            return false;
        }
        sc->alpha[h] = -expf(a) * qwen36_softplus_f32(sc->alpha[h] + dt);
    }

    bool current_recurrent_nonzero =
        !qwen36_vec_all_zero(query, key_dim) ||
        !qwen36_vec_all_zero(key, key_dim) ||
        !qwen36_vec_all_zero(value, value_dim) ||
        s->recurrent_state != NULL;
    float *state = qwen36_session_recurrent(s, current_recurrent_nonzero);
    if (current_recurrent_nonzero && !state) {
        qwen36_set_err(err, errlen, "failed to allocate Qwen recurrent state");
        return false;
    }
    uint64_t state_layer_stride = (uint64_t)n_v_heads * head_dim * head_dim;
    float *layer_state = state ? state + (uint64_t)recurrent_idx * state_layer_stride : NULL;
    memset(sc->core, 0, (size_t)value_dim * sizeof(sc->core[0]));

    if (layer_state) {
        float qscale = 1.0f / sqrtf((float)head_dim);
        for (uint32_t vh = 0; vh < n_v_heads; vh++) {
            uint32_t kh = vh / repeat;
            float decay = expf(sc->alpha[vh]);
            float beta = sc->beta[vh];
            const float *qh = query + (uint64_t)kh * head_dim;
            const float *khv = key + (uint64_t)kh * head_dim;
            const float *vv = value + (uint64_t)vh * head_dim;
            float *state_h = layer_state + (uint64_t)vh * head_dim * head_dim;
            if (!qwen36_gated_delta_head(e, state_h, qh, khv, vv, head_dim,
                                         decay, beta, qscale,
                                         sc->core + (uint64_t)vh * head_dim,
                                         err, errlen)) {
                return false;
            }
        }
    }

    for (uint32_t vh = 0; vh < n_v_heads; vh++) {
        float *head = sc->core + (uint64_t)vh * head_dim;
        if (!qwen36_rms_norm(e, layer->ssm_norm, head, head_dim, head, false,
                             err, errlen)) return false;
        for (uint32_t d = 0; d < head_dim; d++) {
            head[d] *= qwen36_silu_f32(sc->z[(uint64_t)vh * head_dim + d]);
        }
    }

    return qwen36_matvec_checked(e, layer->ssm_out, sc->core, value_dim,
                                 out, spec->hidden_size, err, errlen);
}

static bool qwen36_eval_ffn_file(qwen36_session *s,
                                 const rt_gguf_file *file,
                                 const qwen36_layer_tensors *layer,
                                 qwen36_cpu_scratch *sc,
                                 float *out,
                                 char *err,
                                 size_t errlen) {
    qwen36_engine *e = s->engine;
    const qwen36_model_spec *spec = e->spec;
    memset(out, 0, spec->hidden_size * sizeof(out[0]));
    if (!qwen36_matvec_file_checked(e, file, layer->ffn_gate, sc->norm,
                                    spec->hidden_size, sc->ffn_gate,
                                    spec->intermediate_size, err, errlen) ||
        !qwen36_matvec_file_checked(e, file, layer->ffn_up, sc->norm,
                                    spec->hidden_size, sc->ffn_up,
                                    spec->intermediate_size, err, errlen)) {
        return false;
    }
    if (!qwen36_silu_mul_checked(e, sc->ffn_gate, sc->ffn_up,
                                 spec->intermediate_size, sc->ffn_act,
                                 err, errlen)) {
        return false;
    }
    return qwen36_matvec_file_checked(e, file, layer->ffn_down, sc->ffn_act,
                                      spec->intermediate_size, out,
                                      spec->hidden_size, err, errlen);
}

static bool qwen36_eval_ffn(qwen36_session *s,
                            const qwen36_layer_tensors *layer,
                            qwen36_cpu_scratch *sc,
                            float *out,
                            char *err,
                            size_t errlen) {
    return qwen36_eval_ffn_file(s, s ? s->engine->model : NULL, layer,
                                sc, out, err, errlen);
}

static bool qwen36_eval_mtp_attention_stateless(qwen36_session *s,
                                                const qwen36_mtp_layer_tensors *mtp,
                                                int pos,
                                                qwen36_cpu_scratch *sc,
                                                float *out,
                                                char *err,
                                                size_t errlen) {
    qwen36_engine *e = s->engine;
    const rt_gguf_file *file = mtp->source ? mtp->source : e->model;
    const qwen36_model_spec *spec = e->spec;
    const qwen36_layer_tensors *layer = &mtp->block;
    const uint32_t head_dim = spec->attention_head_dim;
    const uint32_t n_heads = spec->attention_heads;
    const uint32_t n_kv = spec->attention_kv_heads;
    const uint32_t kv_groups = n_heads / n_kv;
    const uint64_t q_dim = (uint64_t)n_heads * head_dim;
    const uint64_t q_gate_dim = q_dim * 2u;
    const uint64_t kv_dim = (uint64_t)n_kv * head_dim;

    memset(out, 0, spec->hidden_size * sizeof(out[0]));
    if (!qwen36_matvec_file_checked(e, file, layer->attn_q, sc->norm,
                                    spec->hidden_size, sc->mix, q_gate_dim,
                                    err, errlen) ||
        !qwen36_matvec_file_checked(e, file, layer->attn_k, sc->norm,
                                    spec->hidden_size, sc->k, kv_dim,
                                    err, errlen) ||
        !qwen36_matvec_file_checked(e, file, layer->attn_v, sc->norm,
                                    spec->hidden_size, sc->v, kv_dim,
                                    err, errlen)) {
        return false;
    }

    float *query = sc->mix;
    float *gate = sc->mix + q_dim;
    for (uint32_t h = 0; h < n_heads; h++) {
        if (!qwen36_rms_norm_file(e, file, layer->attn_q_norm,
                                  query + (uint64_t)h * head_dim,
                                  head_dim, query + (uint64_t)h * head_dim,
                                  true, err, errlen)) {
            return false;
        }
        qwen36_apply_rope(query + (uint64_t)h * head_dim, head_dim,
                          head_dim / 4u, pos);
    }
    for (uint32_t h = 0; h < n_kv; h++) {
        if (!qwen36_rms_norm_file(e, file, layer->attn_k_norm,
                                  sc->k + (uint64_t)h * head_dim,
                                  head_dim, sc->k + (uint64_t)h * head_dim,
                                  true, err, errlen)) {
            return false;
        }
        qwen36_apply_rope(sc->k + (uint64_t)h * head_dim, head_dim,
                          head_dim / 4u, pos);
    }

    for (uint32_t h = 0; h < n_heads; h++) {
        uint32_t kh = h / kv_groups;
        const float *vh = sc->v + (uint64_t)kh * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) {
            uint64_t idx = (uint64_t)h * head_dim + d;
            sc->attn[idx] = vh[d] * qwen36_sigmoid_f32(gate[idx]);
        }
    }

    return qwen36_matvec_file_checked(e, file, layer->attn_output, sc->attn,
                                      q_dim, out, spec->hidden_size,
                                      err, errlen);
}

static int qwen36_argmax_logits(const float *logits, int n_vocab) {
    if (!logits || n_vocab <= 0) return -1;
    int best = 0;
    float best_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best = i;
        }
    }
    return best;
}

static bool qwen36_logits_top2(const float *logits,
                               int n_vocab,
                               int *top0,
                               float *v0,
                               int *top1,
                               float *v1) {
    if (!logits || n_vocab <= 0 || !top0 || !v0 || !top1 || !v1) return false;
    int best = 0;
    int second = -1;
    float best_logit = logits[0];
    float second_logit = -FLT_MAX;
    for (int i = 1; i < n_vocab; i++) {
        float logit = logits[i];
        if (logit > best_logit) {
            second = best;
            second_logit = best_logit;
            best = i;
            best_logit = logit;
        } else if (second < 0 || logit > second_logit) {
            second = i;
            second_logit = logit;
        }
    }
    *top0 = best;
    *v0 = best_logit;
    *top1 = second;
    *v1 = second >= 0 ? second_logit : best_logit;
    return true;
}

static float qwen36_mtp_margin_threshold(const qwen36_engine *e) {
    float threshold = e ? e->mtp_margin : 0.0f;
    const char *env = getenv("QWEN36_MTP_MIN_MARGIN");
    if (!env || !env[0]) env = getenv("DS4_MTP_MIN_MARGIN");
    if (env && env[0]) {
        char *end = NULL;
        float v = strtof(env, &end);
        if (end != env && v >= 0.0f && isfinite(v)) threshold = v;
    }
    return threshold > 0.0f && isfinite(threshold) ? threshold : 0.0f;
}

static int qwen36_mtp_draft_argmax_from_hidden(qwen36_session *s,
                                               uint32_t mtp_index,
                                               int prev_token,
                                               const float *hidden_in,
                                               float *hidden_out,
                                               float *margin_out,
                                               char *err,
                                               size_t errlen) {
    if (margin_out) *margin_out = 0.0f;
    if (!s || !s->engine->mtp_enabled ||
        s->engine->weights.n_mtp_layers_bound == 0 ||
        mtp_index >= s->engine->weights.n_mtp_layers_bound ||
        !hidden_in) {
        return -1;
    }
    qwen36_engine *e = s->engine;
    const qwen36_model_spec *spec = e->spec;
    const qwen36_mtp_layer_tensors *mtp = &e->weights.mtp[mtp_index];
    const rt_gguf_file *file = mtp->source ? mtp->source : e->model;
    qwen36_cpu_scratch sc;
    if (!qwen36_cpu_scratch_init(&sc, s, err, errlen)) return -1;
    float *draft_logits = calloc((size_t)e->vocab.n_vocab, sizeof(draft_logits[0]));
    if (!draft_logits) {
        qwen36_cpu_scratch_free(&sc);
        qwen36_set_err(err, errlen, "failed to allocate Qwen MTP logits");
        return -1;
    }

    bool ok = qwen36_read_embedding(e, prev_token, sc.hidden, err, errlen);
    if (ok) {
        ok = qwen36_rms_norm_file(e, file, mtp->enorm, sc.hidden,
                                  spec->hidden_size, sc.mix, true,
                                  err, errlen);
    }
    if (ok) {
        ok = qwen36_rms_norm_file(e, file, mtp->hnorm, hidden_in,
                                  spec->hidden_size,
                                  sc.mix + spec->hidden_size,
                                  true, err, errlen);
    }
    if (ok) {
        ok = qwen36_matvec_file_checked(e, file, mtp->eh_proj, sc.mix,
                                        (uint64_t)spec->hidden_size * 2u,
                                        sc.hidden, spec->hidden_size,
                                        err, errlen);
    }
    if (ok) {
        ok = qwen36_rms_norm_file(e, file, mtp->block.attn_norm, sc.hidden,
                                  spec->hidden_size, sc.norm, true,
                                  err, errlen);
    }
    if (ok) {
        ok = qwen36_eval_mtp_attention_stateless(s, mtp,
                                                 s->tokens.len + (int)mtp_index,
                                                 &sc, sc.mix2, err, errlen);
    }
    if (ok) {
        for (uint32_t i = 0; i < spec->hidden_size; i++) sc.hidden[i] += sc.mix2[i];
        ok = qwen36_rms_norm_file(e, file, mtp->block.attn_post_norm,
                                  sc.hidden, spec->hidden_size, sc.norm,
                                  true, err, errlen);
    }
    if (ok) {
        ok = qwen36_eval_ffn_file(s, file, &mtp->block, &sc, sc.mix2,
                                  err, errlen);
    }
    if (ok) {
        for (uint32_t i = 0; i < spec->hidden_size; i++) sc.hidden[i] += sc.mix2[i];
        if (hidden_out) {
            memcpy(hidden_out, sc.hidden,
                   (size_t)spec->hidden_size * sizeof(hidden_out[0]));
        }
        ok = qwen36_rms_norm_file(e, file, mtp->shared_head_norm,
                                  sc.hidden, spec->hidden_size, sc.norm,
                                  true, err, errlen);
    }
    if (ok) {
        ok = qwen36_matvec_checked(e, e->weights.output, sc.norm,
                                   spec->hidden_size, draft_logits,
                                   e->vocab.n_vocab, err, errlen);
    }

    int best = -1;
    if (ok) {
        int top0 = -1;
        int top1 = -1;
        float v0 = 0.0f;
        float v1 = 0.0f;
        if (qwen36_logits_top2(draft_logits, e->vocab.n_vocab,
                               &top0, &v0, &top1, &v1)) {
            best = top0;
            if (margin_out) {
                *margin_out = top1 >= 0 ? v0 - v1 : FLT_MAX;
            }
        }
    }
    free(draft_logits);
    qwen36_cpu_scratch_free(&sc);
    return best;
}

static int qwen36_mtp_build_draft_suffix(qwen36_session *s,
                                         int prev_token,
                                         int max_draft,
                                         int eos_token,
                                         char *err,
                                         size_t errlen) {
    qwen36_session_clear_mtp_draft(s);
    if (!s || !s->engine->mtp_enabled || max_draft <= 0 ||
        s->engine->weights.n_mtp_layers_bound == 0 ||
        !s->last_hidden_valid || !s->last_hidden) {
        return 0;
    }
    qwen36_engine *e = s->engine;
    const qwen36_model_spec *spec = e->spec;
    int cap = max_draft;
    if (cap > e->mtp_draft_tokens - 1) cap = e->mtp_draft_tokens - 1;
    if (cap > (int)e->weights.n_mtp_layers_bound) {
        cap = (int)e->weights.n_mtp_layers_bound;
    }
    if (cap > (int)QWEN36_MAX_MTP_LAYERS) cap = (int)QWEN36_MAX_MTP_LAYERS;
    if (cap <= 0) return 0;

    float *hidden_a = malloc((size_t)spec->hidden_size * sizeof(hidden_a[0]));
    float *hidden_b = malloc((size_t)spec->hidden_size * sizeof(hidden_b[0]));
    if (!hidden_a || !hidden_b) {
        free(hidden_a);
        free(hidden_b);
        qwen36_set_err(err, errlen, "failed to allocate Qwen MTP draft hidden state");
        return -1;
    }
    memcpy(hidden_a, s->last_hidden, (size_t)spec->hidden_size * sizeof(hidden_a[0]));
    float *current_hidden = hidden_a;
    float *next_hidden = hidden_b;
    float margin_threshold = qwen36_mtp_margin_threshold(e);

    int n = 0;
    for (int i = 0; i < cap; i++) {
        float margin = 0.0f;
        int draft = qwen36_mtp_draft_argmax_from_hidden(s, (uint32_t)i,
                                                        prev_token,
                                                        current_hidden,
                                                        next_hidden,
                                                        &margin,
                                                        err, errlen);
        if (draft < 0) {
            free(hidden_a);
            free(hidden_b);
            return -1;
        }
        s->mtp_draft_proposed++;
        if (margin_threshold > 0.0f && margin < margin_threshold) {
            s->mtp_draft_skipped++;
            break;
        }
        s->mtp_draft[n++] = draft;
        s->mtp_draft_margin[n - 1] = margin;
        s->mtp_draft_len = n;
        prev_token = draft;
        float *tmp = current_hidden;
        current_hidden = next_hidden;
        next_hidden = tmp;
        if (draft == eos_token) break;
    }

    free(hidden_a);
    free(hidden_b);
    return n;
}

static bool qwen36_session_eval_one_input(qwen36_session *s,
                                          int token,
                                          const float *input_embedding,
                                          char *err,
                                          size_t errlen) {
    if (!s || token < 0 || token >= s->engine->vocab.n_vocab || s->tokens.len >= s->ctx_size) {
        qwen36_set_err(err, errlen, "invalid Qwen eval request");
        return false;
    }
    qwen36_engine *e = s->engine;
    const qwen36_model_spec *spec = e->spec;
    qwen36_cpu_scratch sc;
    if (!qwen36_cpu_scratch_init(&sc, s, err, errlen)) return false;

    bool ok = true;
    if (input_embedding) {
        memcpy(sc.hidden, input_embedding, (size_t)spec->hidden_size * sizeof(sc.hidden[0]));
    } else {
        ok = qwen36_read_embedding(e, token, sc.hidden, err, errlen);
    }
    uint32_t full_idx = 0;
    uint32_t recurrent_idx = 0;
    int pos = s->tokens.len;

    for (uint32_t il = 0; ok && il < e->weights.n_layers_main; il++) {
        const qwen36_layer_tensors *layer = &e->weights.layer[il];
        ok = qwen36_rms_norm(e, layer->attn_norm, sc.hidden, spec->hidden_size,
                             sc.norm, true, err, errlen);
        if (!ok) break;
        if (layer->recurrent) {
            ok = qwen36_eval_recurrent(s, layer, recurrent_idx++, &sc,
                                       sc.mix2, err, errlen);
        } else {
            ok = qwen36_eval_full_attention(s, layer, full_idx++, pos, &sc,
                                            sc.mix2, err, errlen);
        }
        if (!ok) break;
        for (uint32_t i = 0; i < spec->hidden_size; i++) sc.hidden[i] += sc.mix2[i];

        ok = qwen36_rms_norm(e, layer->attn_post_norm, sc.hidden, spec->hidden_size,
                             sc.norm, true, err, errlen);
        if (!ok) break;
        ok = qwen36_eval_ffn(s, layer, &sc, sc.mix2, err, errlen);
        if (!ok) break;
        for (uint32_t i = 0; i < spec->hidden_size; i++) sc.hidden[i] += sc.mix2[i];
    }

    if (ok) {
        ok = qwen36_rms_norm(e, e->weights.output_norm, sc.hidden, spec->hidden_size,
                             sc.norm, true, err, errlen);
    }
    if (ok) {
        ok = qwen36_matvec_checked(e, e->weights.output, sc.norm, spec->hidden_size,
                                   s->logits, e->vocab.n_vocab, err, errlen);
    }
    if (ok && s->last_hidden) {
        memcpy(s->last_hidden, sc.hidden,
               (size_t)spec->hidden_size * sizeof(s->last_hidden[0]));
    }
    qwen36_cpu_scratch_free(&sc);
    if (!ok) {
        qwen36_session_invalidate_logits(s);
        qwen36_session_invalidate_hidden(s);
        return false;
    }
    rt_tokens_push(&s->tokens, token);
    s->logits_valid = true;
    s->state_valid = true;
    s->last_hidden_valid = s->last_hidden != NULL;
    return true;
}

static bool qwen36_session_eval_one(qwen36_session *s, int token, char *err, size_t errlen) {
    return qwen36_session_eval_one_input(s, token, NULL, err, errlen);
}

bool qwen36_runtime_session_read_last_hidden(const rt_session *session,
                                             float *out,
                                             uint32_t cap) {
    if (!session || !session->engine || !session->impl || !out ||
        session->engine->ops != qwen36_runtime_ops()) {
        return false;
    }
    const qwen36_session *s = session->impl;
    const uint32_t hidden = s->engine->spec->hidden_size;
    if (!s->last_hidden_valid || !s->last_hidden || cap < hidden) return false;
    memcpy(out, s->last_hidden, (size_t)hidden * sizeof(out[0]));
    return true;
}

int qwen36_runtime_session_sync_embeddings(rt_session *session,
                                           const rt_tokens *prompt,
                                           const qwen36_runtime_embedding_span *spans,
                                           size_t n_spans,
                                           char *err,
                                           size_t errlen) {
    if (!session || !session->engine || !session->impl ||
        session->engine->ops != qwen36_runtime_ops() ||
        !prompt || prompt->len < 0 || (n_spans != 0 && !spans)) {
        qwen36_set_err(err, errlen, "invalid Qwen embedding sync request");
        return -1;
    }
    qwen36_session *s = session->impl;
    const uint32_t hidden = s->engine->spec->hidden_size;
    if (prompt->len >= s->ctx_size) {
        qwen36_set_err(err, errlen, "Qwen prompt length %d exceeds context limit %d",
                       prompt->len, s->ctx_size);
        return -1;
    }
    int span_end = 0;
    for (size_t i = 0; i < n_spans; i++) {
        const qwen36_runtime_embedding_span *span = &spans[i];
        if (!span->data || span->token_pos < span_end ||
            span->token_pos < 0 || span->n_tokens == 0 ||
            span->hidden_size != hidden ||
            span->n_tokens > (uint32_t)INT_MAX ||
            span->token_pos > INT_MAX - (int)span->n_tokens ||
            span->token_pos + (int)span->n_tokens > prompt->len) {
            qwen36_set_err(err, errlen, "invalid Qwen embedding span");
            return -1;
        }
        uint64_t n_values = (uint64_t)span->n_tokens * span->hidden_size;
        for (uint64_t j = 0; j < n_values; j++) {
            if (!isfinite(span->data[j])) {
                qwen36_set_err(err, errlen, "Qwen embedding span contains a non-finite value");
                return -1;
            }
        }
        span_end = span->token_pos + (int)span->n_tokens;
    }

    rt_tokens_free(&s->tokens);
    qwen36_session_invalidate_logits(s);
    qwen36_session_invalidate_hidden(s);
    qwen36_session_clear_state(s);

    size_t span_i = 0;
    if (s->progress_fn) {
        s->progress_fn(s->progress_ud, "qwen36-sync-embeddings", 0, prompt->len);
    }
    for (int i = 0; i < prompt->len; i++) {
        while (span_i < n_spans &&
               i >= spans[span_i].token_pos + (int)spans[span_i].n_tokens) {
            span_i++;
        }
        const float *embedding = NULL;
        if (span_i < n_spans &&
            i >= spans[span_i].token_pos &&
            i < spans[span_i].token_pos + (int)spans[span_i].n_tokens) {
            uint64_t rel = (uint64_t)(i - spans[span_i].token_pos);
            embedding = spans[span_i].data + rel * hidden;
        }
        if (!qwen36_session_eval_one_input(s, prompt->v[i], embedding, err, errlen)) {
            qwen36_session_clear_state(s);
            rt_tokens_free(&s->tokens);
            qwen36_session_invalidate_logits(s);
            qwen36_session_invalidate_hidden(s);
            return -1;
        }
        if (s->progress_fn) {
            s->progress_fn(s->progress_ud, "qwen36-sync-embeddings", i + 1, prompt->len);
        }
    }
    if (s->progress_fn) {
        s->progress_fn(s->progress_ud, "qwen36-sync-embeddings", prompt->len, prompt->len);
    }
    return 0;
}

static int qwen36_rt_session_create(void **out, void *engine, int ctx_size) {
    if (!out || !engine || ctx_size <= 0 ||
        ctx_size > (int)QWEN36_MODEL_SPEC.max_context) {
        return 1;
    }
    *out = NULL;

    qwen36_session *session = calloc(1, sizeof(*session));
    if (!session) return 1;
    session->engine = engine;
    session->ctx_size = ctx_size;
    session->logits = calloc((size_t)session->engine->vocab.n_vocab, sizeof(session->logits[0]));
    session->last_hidden = calloc((size_t)session->engine->spec->hidden_size,
                                  sizeof(session->last_hidden[0]));
    if (!session->logits || !session->last_hidden) {
        free(session->last_hidden);
        free(session->logits);
        free(session);
        return 1;
    }
    *out = session;
    return 0;
}

static void qwen36_rt_session_free(void *session) {
    qwen36_session *s = session;
    if (!s) return;
    rt_tokens_free(&s->tokens);
    free(s->logits);
    free(s->last_hidden);
    qwen36_session_clear_state(s);
    free(s);
}

static void qwen36_rt_session_set_progress(void *session, rt_session_progress_fn fn, void *ud) {
    qwen36_session *s = session;
    if (!s) return;
    s->progress_fn = fn;
    s->progress_ud = ud;
}

static int qwen36_rt_session_sync(void *session, const rt_tokens *prompt, char *err, size_t errlen) {
    qwen36_session *s = session;
    if (!s || !prompt || prompt->len < 0) {
        qwen36_set_err(err, errlen, "invalid Qwen session sync arguments");
        return -1;
    }
    if (prompt->len >= s->ctx_size) {
        qwen36_set_err(err, errlen, "Qwen prompt length %d exceeds context limit %d",
                       prompt->len, s->ctx_size);
        return -1;
    }

    int start = 0;
    bool can_reuse_prefix = s->tokens.len == 0 ||
                            (s->state_valid && s->logits_valid);
    if (can_reuse_prefix && s->tokens.len > 0 && s->tokens.len <= prompt->len) {
        while (start < s->tokens.len && s->tokens.v[start] == prompt->v[start]) {
            start++;
        }
    }
    if (!can_reuse_prefix || start != s->tokens.len) {
        start = 0;
        rt_tokens_free(&s->tokens);
        qwen36_session_invalidate_logits(s);
        qwen36_session_invalidate_hidden(s);
        qwen36_session_clear_state(s);
    }
    qwen36_session_clear_mtp_draft(s);

    if (s->progress_fn) s->progress_fn(s->progress_ud, "qwen36-sync", start, prompt->len);
    for (int i = start; i < prompt->len; i++) {
        if (!qwen36_session_eval_one(s, prompt->v[i], err, errlen)) {
            qwen36_session_clear_state(s);
            rt_tokens_free(&s->tokens);
            qwen36_session_invalidate_logits(s);
            qwen36_session_invalidate_hidden(s);
            return -1;
        }
        if (s->progress_fn) s->progress_fn(s->progress_ud, "qwen36-sync", i + 1, prompt->len);
    }
    if (s->progress_fn) s->progress_fn(s->progress_ud, "qwen36-sync", prompt->len, prompt->len);
    return 0;
}

static int qwen36_rt_session_eval(void *session, int token, char *err, size_t errlen) {
    qwen36_session *s = session;
    qwen36_session_invalidate_logits(s);
    return qwen36_session_eval_one(s, token, err, errlen) ? 0 : -1;
}

static int qwen36_rt_session_argmax(void *session);

static int qwen36_rt_session_eval_speculative_argmax(void *session,
                                                     int first_token,
                                                     int max_tokens,
                                                     int eos_token,
                                                     int *out_tokens,
                                                     int out_cap,
                                                     char *err,
                                                     size_t errlen) {
    qwen36_session *s = session;
    if (!s || !out_tokens || out_cap <= 0 || max_tokens <= 0) {
        qwen36_set_err(err, errlen, "invalid Qwen speculative eval request");
        return -1;
    }
    if (first_token == eos_token) return 0;

    int limit = max_tokens;
    if (limit > out_cap) limit = out_cap;
    if (limit > 16) limit = 16;
    if (!s->engine->mtp_enabled || s->engine->mtp_draft_tokens <= 1) {
        limit = 1;
    } else if (limit > s->engine->mtp_draft_tokens) {
        limit = s->engine->mtp_draft_tokens;
    }
    int room = s->ctx_size - s->tokens.len;
    if (limit > room) limit = room;
    if (limit <= 0) {
        qwen36_set_err(err, errlen, "Qwen context is full");
        return -1;
    }

    qwen36_session_invalidate_logits(s);
    if (!qwen36_session_eval_one(s, first_token, err, errlen)) return -1;
    int n = 0;
    out_tokens[n++] = first_token;

    int drafted = qwen36_mtp_build_draft_suffix(s, first_token, limit - n,
                                                eos_token, err, errlen);
    if (drafted < 0) return -1;
    for (int i = 0; i < drafted && n < limit; i++) {
        s->mtp_draft_pos = i;
        int target_next = qwen36_rt_session_argmax(s);
        if (target_next < 0) {
            qwen36_set_err(err, errlen, "Qwen speculative target argmax failed");
            return -1;
        }
        int draft_next = s->mtp_draft[i];
        if (draft_next != target_next) {
            s->mtp_draft_rejected++;
            qwen36_session_clear_mtp_draft(s);
            return n;
        }
        if (!qwen36_session_eval_one(s, draft_next, err, errlen)) {
            qwen36_session_clear_mtp_draft(s);
            return -1;
        }
        s->mtp_draft_accepted++;
        s->mtp_draft_pos = i + 1;
        out_tokens[n++] = draft_next;
        if (draft_next == eos_token) {
            qwen36_session_clear_mtp_draft(s);
            return n;
        }
    }
    qwen36_session_clear_mtp_draft(s);

    while (drafted == 0 && n < limit) {
        int target_next = qwen36_rt_session_argmax(s);
        if (target_next < 0) {
            qwen36_set_err(err, errlen, "Qwen speculative target argmax failed");
            return -1;
        }
        if (target_next == eos_token) break;
        if (!qwen36_session_eval_one(s, target_next, err, errlen)) return -1;
        out_tokens[n++] = target_next;
    }
    return n;
}

static int qwen36_rt_session_argmax(void *session) {
    qwen36_session *s = session;
    if (!s || !s->logits_valid || s->engine->vocab.n_vocab <= 0) return -1;
    return qwen36_argmax_logits(s->logits, s->engine->vocab.n_vocab);
}

static int qwen36_rt_session_sample(
        void *session,
        float temperature,
        int top_k,
        float top_p,
        float min_p,
        uint64_t *rng) {
    qwen36_session *s = session;
    if (!s || !s->logits_valid || s->engine->vocab.n_vocab <= 0) return -1;
    if (temperature <= 0.0f || !rng) return qwen36_rt_session_argmax(session);

    int n_vocab = s->engine->vocab.n_vocab;
    qwen36_sample_candidate *c = qwen36_xmalloc((size_t)n_vocab * sizeof(c[0]));
    for (int i = 0; i < n_vocab; i++) {
        c[i].id = i;
        c[i].logit = s->logits[i] / temperature;
        c[i].prob = 0.0;
    }
    qsort(c, (size_t)n_vocab, sizeof(c[0]), qwen36_sample_candidate_cmp_desc);

    int n = n_vocab;
    if (top_k > 0 && top_k < n) n = top_k;

    double max_logit = c[0].logit;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        c[i].prob = exp((double)c[i].logit - max_logit);
        sum += c[i].prob;
    }
    if (sum <= 0.0 || !isfinite(sum)) {
        int id = c[0].id;
        free(c);
        return id;
    }
    for (int i = 0; i < n; i++) c[i].prob /= sum;

    if (min_p > 0.0f && n > 1) {
        double threshold = c[0].prob * (double)min_p;
        int keep = 1;
        while (keep < n && c[keep].prob >= threshold) keep++;
        n = keep;
    }

    if (top_p > 0.0f && top_p < 1.0f && n > 1) {
        double cumulative = 0.0;
        int keep = 0;
        while (keep < n) {
            cumulative += c[keep].prob;
            keep++;
            if (cumulative >= (double)top_p) break;
        }
        if (keep > 0) n = keep;
    }

    sum = 0.0;
    for (int i = 0; i < n; i++) sum += c[i].prob;
    if (sum <= 0.0 || !isfinite(sum)) {
        int id = c[0].id;
        free(c);
        return id;
    }

    *rng = *rng * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    double r = (double)(*rng >> 11) * (1.0 / 9007199254740992.0);
    double target = r * sum;
    double cumulative = 0.0;
    int selected = c[n - 1].id;
    for (int i = 0; i < n; i++) {
        cumulative += c[i].prob;
        if (target <= cumulative) {
            selected = c[i].id;
            break;
        }
    }
    free(c);
    return selected;
}

static int qwen36_rt_session_top_logprobs(void *session, rt_token_score *out, int k) {
    qwen36_session *s = session;
    if (!s || !s->logits_valid || !out || k <= 0) return -1;
    int n_vocab = s->engine->vocab.n_vocab;
    if (n_vocab <= 0) return -1;

    qwen36_sample_candidate *top = qwen36_xcalloc((size_t)k, sizeof(top[0]));
    int n_top = 0;
    for (int i = 0; i < n_vocab; i++) {
        float logit = s->logits[i];
        int pos = n_top;
        while (pos > 0 && logit > top[pos - 1].logit) pos--;
        if (pos >= k) continue;
        if (n_top < k) n_top++;
        for (int j = n_top - 1; j > pos; j--) top[j] = top[j - 1];
        top[pos] = (qwen36_sample_candidate){ .id = i, .logit = logit, .prob = 0.0 };
    }

    double logsum = 0.0;
    if (!qwen36_logits_logsumexp(s->logits, n_vocab, &logsum)) {
        free(top);
        return -1;
    }
    for (int i = 0; i < n_top; i++) {
        out[i].id = top[i].id;
        out[i].logit = top[i].logit;
        out[i].logprob = (float)((double)top[i].logit - logsum);
    }
    free(top);
    return n_top;
}

static int qwen36_rt_session_token_logprob(void *session, int token, rt_token_score *out) {
    qwen36_session *s = session;
    return qwen36_session_token_score(s, token, out) ? 0 : -1;
}

static int qwen36_rt_session_read_logits(void *session, float *out, uint32_t cap) {
    qwen36_session *s = session;
    if (!s || !out || !s->logits_valid || cap < (uint32_t)s->engine->vocab.n_vocab) return -1;
    memcpy(out, s->logits, (size_t)s->engine->vocab.n_vocab * sizeof(out[0]));
    return 0;
}

static void qwen36_rt_session_invalidate(void *session) {
    qwen36_session *s = session;
    if (!s) return;
    rt_tokens_free(&s->tokens);
    qwen36_session_invalidate_logits(s);
    qwen36_session_invalidate_hidden(s);
    qwen36_session_clear_state(s);
}

static void qwen36_rt_session_rewind(void *session, int pos) {
    qwen36_session *s = session;
    if (!s) return;
    if (pos < 0) pos = 0;
    if (pos < s->tokens.len) s->tokens.len = pos;
    qwen36_session_invalidate_logits(s);
    qwen36_session_invalidate_hidden(s);
    qwen36_session_clear_state(s);
}

static int qwen36_rt_session_pos(void *session) {
    qwen36_session *s = session;
    return s ? s->tokens.len : 0;
}

static int qwen36_rt_session_ctx(void *session) {
    qwen36_session *s = session;
    return s ? s->ctx_size : 0;
}

static const rt_tokens *qwen36_rt_session_tokens(void *session) {
    qwen36_session *s = session;
    return s ? &s->tokens : NULL;
}

static uint64_t qwen36_rt_session_payload_bytes(void *session) {
    return qwen36_session_payload_bytes_impl(session);
}

static int qwen36_rt_session_save_payload(void *session, FILE *fp, char *err, size_t errlen) {
    qwen36_session *s = session;
    if (!s || !fp) {
        qwen36_set_err(err, errlen, "invalid Qwen payload save arguments");
        return -1;
    }
    if (s->tokens.len < 0 || s->tokens.len > s->ctx_size ||
        s->tokens.len > INT32_MAX ||
        s->engine->vocab.n_vocab > INT32_MAX) {
        qwen36_set_err(err, errlen, "Qwen session state is not serializable");
        return -1;
    }
    uint64_t full_kv_bytes = 0;
    uint64_t recurrent_state_bytes = 0;
    uint64_t conv_state_bytes = 0;
    if (!qwen36_state_layout_bytes(s->engine->spec, s->ctx_size,
                                   &full_kv_bytes,
                                   &recurrent_state_bytes,
                                   &conv_state_bytes)) {
        qwen36_set_err(err, errlen, "Qwen session state layout is not serializable");
        return -1;
    }

    uint8_t header[QWEN36_SESSION_PAYLOAD_WORDS * sizeof(uint32_t)];
    qwen36_le_put32(header + 0, QWEN36_SESSION_PAYLOAD_MAGIC);
    qwen36_le_put32(header + 4, QWEN36_SESSION_PAYLOAD_VERSION);
    qwen36_le_put32(header + 8, QWEN36_SESSION_PAYLOAD_WORDS);
    qwen36_le_put32(header + 12, (uint32_t)s->ctx_size);
    qwen36_le_put32(header + 16, (uint32_t)s->tokens.len);
    qwen36_le_put32(header + 20, (uint32_t)s->tokens.len);
    qwen36_le_put32(header + 24, (uint32_t)s->engine->vocab.n_vocab);
    qwen36_le_put32(header + 28, s->engine->spec->hidden_size);
    qwen36_le_put32(header + 32, s->engine->spec->n_layers);
    bool have_hidden = s->last_hidden_valid && s->last_hidden;
    uint32_t section_count = 1u + (s->logits_valid ? 1u : 0u) +
                             (have_hidden ? 1u : 0u) +
                             (s->full_kv_state ? 1u : 0u) +
                             (s->recurrent_state ? 1u : 0u) +
                             (s->conv_state ? 1u : 0u);
    qwen36_le_put32(header + 36, section_count);
    qwen36_le_put64_words(header + 40, full_kv_bytes);
    qwen36_le_put64_words(header + 48, recurrent_state_bytes);
    qwen36_le_put64_words(header + 56, conv_state_bytes);

    if (!qwen36_write_exact(fp, header, sizeof(header))) {
        qwen36_set_err(err, errlen, "failed to write Qwen session payload header");
        return -1;
    }

    if (!qwen36_write_section_header(fp, QWEN36_PAYLOAD_SECTION_TOKENS,
                                     (uint64_t)s->tokens.len * sizeof(uint32_t),
                                     err, errlen)) return -1;
    if (s->logits_valid) {
        if (!qwen36_write_section_header(fp, QWEN36_PAYLOAD_SECTION_LOGITS,
                                         (uint64_t)s->engine->vocab.n_vocab * sizeof(float),
                                         err, errlen)) return -1;
    }
    if (have_hidden) {
        if (!qwen36_write_section_header(fp, QWEN36_PAYLOAD_SECTION_LAST_HIDDEN_F32,
                                         (uint64_t)s->engine->spec->hidden_size * sizeof(float),
                                         err, errlen)) return -1;
    }
    if (s->full_kv_state &&
        !qwen36_write_section_header(fp, QWEN36_PAYLOAD_SECTION_FULL_KV_F32,
                                     s->full_kv_state_bytes, err, errlen)) {
        return -1;
    }
    if (s->recurrent_state &&
        !qwen36_write_section_header(fp, QWEN36_PAYLOAD_SECTION_RECURRENT_F32,
                                     s->recurrent_state_bytes, err, errlen)) {
        return -1;
    }
    if (s->conv_state &&
        !qwen36_write_section_header(fp, QWEN36_PAYLOAD_SECTION_CONV_F32,
                                     s->conv_state_bytes, err, errlen)) {
        return -1;
    }

    for (int i = 0; i < s->tokens.len; i++) {
        uint8_t b[4];
        qwen36_le_put32(b, (uint32_t)s->tokens.v[i]);
        if (!qwen36_write_exact(fp, b, sizeof(b))) {
            qwen36_set_err(err, errlen, "failed to write Qwen session payload tokens");
            return -1;
        }
    }
    if (s->logits_valid &&
        !qwen36_write_exact(fp, s->logits,
                            (size_t)s->engine->vocab.n_vocab * sizeof(s->logits[0]))) {
        qwen36_set_err(err, errlen, "failed to write Qwen session payload logits");
        return -1;
    }
    if (have_hidden &&
        !qwen36_write_exact(fp, s->last_hidden,
                            (size_t)s->engine->spec->hidden_size * sizeof(s->last_hidden[0]))) {
        qwen36_set_err(err, errlen, "failed to write Qwen last hidden state payload");
        return -1;
    }
    if (s->full_kv_state &&
        !qwen36_write_exact(fp, s->full_kv_state, (size_t)s->full_kv_state_bytes)) {
        qwen36_set_err(err, errlen, "failed to write Qwen full-attention state payload");
        return -1;
    }
    if (s->recurrent_state &&
        !qwen36_write_exact(fp, s->recurrent_state, (size_t)s->recurrent_state_bytes)) {
        qwen36_set_err(err, errlen, "failed to write Qwen recurrent state payload");
        return -1;
    }
    if (s->conv_state &&
        !qwen36_write_exact(fp, s->conv_state, (size_t)s->conv_state_bytes)) {
        qwen36_set_err(err, errlen, "failed to write Qwen convolution state payload");
        return -1;
    }
    return 0;
}

static int qwen36_rt_session_load_payload(
        void *session,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen) {
    qwen36_session *s = session;
    if (!s || !fp) {
        qwen36_set_err(err, errlen, "invalid Qwen payload load arguments");
        return -1;
    }

    uint8_t header[QWEN36_SESSION_PAYLOAD_WORDS * sizeof(uint32_t)];
    size_t legacy_header_bytes = QWEN36_SESSION_PAYLOAD_LEGACY_WORDS * sizeof(uint32_t);
    if (payload_bytes < legacy_header_bytes ||
        !qwen36_read_exact(fp, header, legacy_header_bytes)) {
        qwen36_set_err(err, errlen, "failed to read Qwen session payload header");
        return -1;
    }

    uint32_t magic = qwen36_le_get32(header + 0);
    uint32_t version = qwen36_le_get32(header + 4);
    uint32_t header_words = qwen36_le_get32(header + 8);
    uint32_t saved_ctx = qwen36_le_get32(header + 12);
    uint32_t token_count = qwen36_le_get32(header + 16);
    uint32_t pos = qwen36_le_get32(header + 20);
    uint32_t vocab_size = qwen36_le_get32(header + 24);
    uint32_t hidden_size = qwen36_le_get32(header + 28);
    uint32_t n_layers = qwen36_le_get32(header + 32);
    uint32_t logits_valid = qwen36_le_get32(header + 36);

    if (magic != QWEN36_SESSION_PAYLOAD_MAGIC ||
        saved_ctx == 0 ||
        saved_ctx > (uint32_t)s->ctx_size ||
        token_count != pos ||
        token_count >= (uint32_t)s->ctx_size ||
        vocab_size != (uint32_t)s->engine->vocab.n_vocab ||
        hidden_size != s->engine->spec->hidden_size ||
        n_layers != s->engine->spec->n_layers) {
        qwen36_set_err(err, errlen, "invalid Qwen session payload header");
        return -1;
    }

    rt_tokens next = {0};
    float *next_logits = NULL;
    bool next_logits_valid = false;
    uint8_t *next_full_kv = NULL;
    uint64_t next_full_kv_bytes = 0;
    uint8_t *next_recurrent = NULL;
    uint64_t next_recurrent_bytes = 0;
    uint8_t *next_conv = NULL;
    uint64_t next_conv_bytes = 0;
    float *next_hidden = NULL;
    bool next_hidden_valid = false;

    if (version == QWEN36_SESSION_PAYLOAD_LEGACY_VERSION) {
        if (header_words != QWEN36_SESSION_PAYLOAD_LEGACY_WORDS || logits_valid > 1u) {
            qwen36_set_err(err, errlen, "invalid Qwen session payload header");
            return -1;
        }

        uint64_t expected = legacy_header_bytes + (uint64_t)token_count * sizeof(uint32_t);
        if (logits_valid &&
            qwen36_u64_add_overflows(expected,
                                     (uint64_t)vocab_size * sizeof(float),
                                     &expected)) {
            qwen36_set_err(err, errlen, "Qwen session payload size overflow");
            return -1;
        }
        if (payload_bytes != expected) {
            qwen36_set_err(err, errlen, "Qwen session payload size mismatch");
            return -1;
        }

        for (uint32_t i = 0; i < token_count; i++) {
            uint8_t b[4];
            if (!qwen36_read_exact(fp, b, sizeof(b))) {
                rt_tokens_free(&next);
                qwen36_set_err(err, errlen, "failed to read Qwen session payload tokens");
                return -1;
            }
            uint32_t token = qwen36_le_get32(b);
            if (token >= (uint32_t)s->engine->vocab.n_vocab || token > (uint32_t)INT32_MAX) {
                rt_tokens_free(&next);
                qwen36_set_err(err, errlen, "invalid token in Qwen session payload");
                return -1;
            }
            rt_tokens_push(&next, (int)token);
        }

        if (logits_valid) {
            next_logits = qwen36_xmalloc((size_t)vocab_size * sizeof(next_logits[0]));
            if (!qwen36_read_exact(fp, next_logits,
                                   (size_t)vocab_size * sizeof(next_logits[0]))) {
                free(next_logits);
                rt_tokens_free(&next);
                qwen36_set_err(err, errlen, "failed to read Qwen session payload logits");
                return -1;
            }
            next_logits_valid = true;
        }
    } else if (version == QWEN36_SESSION_PAYLOAD_V2_VERSION ||
               version == QWEN36_SESSION_PAYLOAD_VERSION) {
        if (header_words != QWEN36_SESSION_PAYLOAD_WORDS ||
            payload_bytes < (uint64_t)header_words * sizeof(uint32_t) ||
            !qwen36_read_exact(fp, header + legacy_header_bytes,
                               (size_t)(header_words - QWEN36_SESSION_PAYLOAD_LEGACY_WORDS) *
                               sizeof(uint32_t))) {
            qwen36_set_err(err, errlen, "failed to read Qwen session payload header");
            return -1;
        }

        uint32_t section_count = qwen36_le_get32(header + 36);
        uint64_t full_kv_bytes = qwen36_le_get64_words(header + 40);
        uint64_t recurrent_state_bytes = qwen36_le_get64_words(header + 48);
        uint64_t conv_state_bytes = qwen36_le_get64_words(header + 56);
        uint64_t expected_full_kv = 0;
        uint64_t expected_recurrent = 0;
        uint64_t expected_conv = 0;
        if (section_count == 0 ||
            section_count > QWEN36_SESSION_PAYLOAD_MAX_SECTIONS ||
            !qwen36_state_layout_bytes(s->engine->spec, (int)saved_ctx,
                                       &expected_full_kv,
                                       &expected_recurrent,
                                       &expected_conv) ||
            full_kv_bytes != expected_full_kv ||
            recurrent_state_bytes != expected_recurrent ||
            conv_state_bytes != expected_conv) {
            qwen36_set_err(err, errlen, "invalid Qwen session payload header");
            return -1;
        }

        qwen36_payload_section sections[QWEN36_SESSION_PAYLOAD_MAX_SECTIONS];
        memset(sections, 0, sizeof(sections));
        uint64_t expected = (uint64_t)header_words * sizeof(uint32_t);
        uint64_t section_table_bytes = (uint64_t)section_count *
                                       QWEN36_SESSION_PAYLOAD_SECTION_WORDS *
                                       sizeof(uint32_t);
        if (qwen36_u64_add_overflows(expected, section_table_bytes, &expected)) {
            qwen36_set_err(err, errlen, "Qwen session payload size overflow");
            return -1;
        }

        bool seen_tokens = false;
        bool seen_logits = false;
        bool seen_full_kv = false;
        bool seen_recurrent = false;
        bool seen_conv = false;
        bool seen_hidden = false;
        for (uint32_t i = 0; i < section_count; i++) {
            uint8_t raw[QWEN36_SESSION_PAYLOAD_SECTION_WORDS * sizeof(uint32_t)];
            if (!qwen36_read_exact(fp, raw, sizeof(raw))) {
                qwen36_set_err(err, errlen, "failed to read Qwen session payload section table");
                return -1;
            }
            sections[i].type = qwen36_le_get32(raw + 0);
            sections[i].flags = qwen36_le_get32(raw + 4);
            sections[i].bytes = qwen36_le_get64_words(raw + 8);
            if (sections[i].flags != 0 ||
                qwen36_u64_add_overflows(expected, sections[i].bytes, &expected)) {
                qwen36_set_err(err, errlen, "invalid Qwen session payload section");
                return -1;
            }
            switch (sections[i].type) {
            case QWEN36_PAYLOAD_SECTION_TOKENS:
                if (seen_tokens || sections[i].bytes != (uint64_t)token_count * sizeof(uint32_t)) {
                    qwen36_set_err(err, errlen, "invalid Qwen token payload section");
                    return -1;
                }
                seen_tokens = true;
                break;
            case QWEN36_PAYLOAD_SECTION_LOGITS:
                if (seen_logits || sections[i].bytes != (uint64_t)vocab_size * sizeof(float)) {
                    qwen36_set_err(err, errlen, "invalid Qwen logits payload section");
                    return -1;
                }
                seen_logits = true;
                break;
            case QWEN36_PAYLOAD_SECTION_LAST_HIDDEN_F32:
                if (version != QWEN36_SESSION_PAYLOAD_VERSION ||
                    seen_hidden ||
                    sections[i].bytes != (uint64_t)hidden_size * sizeof(float)) {
                    qwen36_set_err(err, errlen, "invalid Qwen last hidden payload section");
                    return -1;
                }
                seen_hidden = true;
                break;
            case QWEN36_PAYLOAD_SECTION_FULL_KV_F32:
                if (seen_full_kv || sections[i].bytes != full_kv_bytes) {
                    qwen36_set_err(err, errlen, "invalid Qwen full-attention payload section");
                    return -1;
                }
                seen_full_kv = true;
                break;
            case QWEN36_PAYLOAD_SECTION_RECURRENT_F32:
                if (seen_recurrent || sections[i].bytes != recurrent_state_bytes) {
                    qwen36_set_err(err, errlen, "invalid Qwen recurrent payload section");
                    return -1;
                }
                seen_recurrent = true;
                break;
            case QWEN36_PAYLOAD_SECTION_CONV_F32:
                if (seen_conv || sections[i].bytes != conv_state_bytes) {
                    qwen36_set_err(err, errlen, "invalid Qwen convolution payload section");
                    return -1;
                }
                seen_conv = true;
                break;
            default:
                qwen36_set_err(err, errlen, "unknown Qwen session payload section");
                return -1;
            }
        }
        if (!seen_tokens) {
            qwen36_set_err(err, errlen, "missing Qwen token payload section");
            return -1;
        }
        if (seen_hidden && token_count == 0) {
            qwen36_set_err(err, errlen, "invalid Qwen last hidden payload section");
            return -1;
        }
        if ((seen_full_kv || seen_recurrent || seen_conv) &&
            saved_ctx != (uint32_t)s->ctx_size) {
            qwen36_set_err(err, errlen, "Qwen state payload requires matching session context");
            return -1;
        }
        if (payload_bytes != expected) {
            qwen36_set_err(err, errlen, "Qwen session payload size mismatch");
            return -1;
        }

        for (uint32_t i = 0; i < section_count; i++) {
            switch (sections[i].type) {
            case QWEN36_PAYLOAD_SECTION_TOKENS:
                for (uint32_t j = 0; j < token_count; j++) {
                    uint8_t b[4];
                    if (!qwen36_read_exact(fp, b, sizeof(b))) {
                        free(next_logits);
                        free(next_hidden);
                        free(next_full_kv);
                        free(next_recurrent);
                        free(next_conv);
                        rt_tokens_free(&next);
                        qwen36_set_err(err, errlen, "failed to read Qwen session payload tokens");
                        return -1;
                    }
                    uint32_t token = qwen36_le_get32(b);
                    if (token >= (uint32_t)s->engine->vocab.n_vocab ||
                        token > (uint32_t)INT32_MAX) {
                        free(next_logits);
                        free(next_hidden);
                        free(next_full_kv);
                        free(next_recurrent);
                        free(next_conv);
                        rt_tokens_free(&next);
                        qwen36_set_err(err, errlen, "invalid token in Qwen session payload");
                        return -1;
                    }
                    rt_tokens_push(&next, (int)token);
                }
                break;
            case QWEN36_PAYLOAD_SECTION_LOGITS:
                next_logits = qwen36_xmalloc((size_t)vocab_size * sizeof(next_logits[0]));
                if (!qwen36_read_exact(fp, next_logits,
                                       (size_t)vocab_size * sizeof(next_logits[0]))) {
                    free(next_logits);
                    free(next_hidden);
                    rt_tokens_free(&next);
                    qwen36_set_err(err, errlen, "failed to read Qwen session payload logits");
                    return -1;
                }
                next_logits_valid = true;
                break;
            case QWEN36_PAYLOAD_SECTION_LAST_HIDDEN_F32:
                next_hidden = qwen36_xmalloc((size_t)hidden_size * sizeof(next_hidden[0]));
                if (!qwen36_read_exact(fp, next_hidden,
                                       (size_t)hidden_size * sizeof(next_hidden[0]))) {
                    free(next_logits);
                    free(next_hidden);
                    free(next_full_kv);
                    free(next_recurrent);
                    free(next_conv);
                    rt_tokens_free(&next);
                    qwen36_set_err(err, errlen, "failed to read Qwen last hidden state payload");
                    return -1;
                }
                next_hidden_valid = true;
                break;
            case QWEN36_PAYLOAD_SECTION_FULL_KV_F32:
                if (!qwen36_read_payload_blob(fp, sections[i].bytes, &next_full_kv)) {
                    free(next_logits);
                    free(next_hidden);
                    free(next_full_kv);
                    free(next_recurrent);
                    free(next_conv);
                    rt_tokens_free(&next);
                    qwen36_set_err(err, errlen, "failed to read Qwen full-attention state payload");
                    return -1;
                }
                next_full_kv_bytes = sections[i].bytes;
                break;
            case QWEN36_PAYLOAD_SECTION_RECURRENT_F32:
                if (!qwen36_read_payload_blob(fp, sections[i].bytes, &next_recurrent)) {
                    free(next_logits);
                    free(next_hidden);
                    free(next_full_kv);
                    free(next_recurrent);
                    free(next_conv);
                    rt_tokens_free(&next);
                    qwen36_set_err(err, errlen, "failed to read Qwen recurrent state payload");
                    return -1;
                }
                next_recurrent_bytes = sections[i].bytes;
                break;
            case QWEN36_PAYLOAD_SECTION_CONV_F32:
                if (!qwen36_read_payload_blob(fp, sections[i].bytes, &next_conv)) {
                    free(next_logits);
                    free(next_hidden);
                    free(next_full_kv);
                    free(next_recurrent);
                    free(next_conv);
                    rt_tokens_free(&next);
                    qwen36_set_err(err, errlen, "failed to read Qwen convolution state payload");
                    return -1;
                }
                next_conv_bytes = sections[i].bytes;
                break;
            default:
                break;
            }
        }
    } else {
        qwen36_set_err(err, errlen, "unsupported Qwen session payload version");
        return -1;
    }

    rt_tokens_free(&s->tokens);
    qwen36_session_clear_state(s);
    s->tokens = next;
    if (next_logits_valid) {
        memcpy(s->logits, next_logits, (size_t)s->engine->vocab.n_vocab * sizeof(s->logits[0]));
        free(next_logits);
        s->logits_valid = true;
    } else {
        s->logits_valid = false;
        qwen36_session_invalidate_logits(s);
    }
    if (next_hidden_valid && s->last_hidden) {
        memcpy(s->last_hidden, next_hidden,
               (size_t)s->engine->spec->hidden_size * sizeof(s->last_hidden[0]));
        s->last_hidden_valid = true;
    } else {
        qwen36_session_invalidate_hidden(s);
    }
    free(next_hidden);
    s->full_kv_state = next_full_kv;
    s->full_kv_state_bytes = next_full_kv_bytes;
    s->recurrent_state = next_recurrent;
    s->recurrent_state_bytes = next_recurrent_bytes;
    s->conv_state = next_conv;
    s->conv_state_bytes = next_conv_bytes;
    s->state_valid = next.len == 0 ||
                     (version == QWEN36_SESSION_PAYLOAD_VERSION &&
                      next_hidden_valid) ||
                     next_full_kv != NULL ||
                     next_recurrent != NULL ||
                     next_conv != NULL;
    qwen36_session_reset_mtp_accounting(s);
    return 0;
}

static const rt_model_ops QWEN36_RUNTIME_OPS = {
    .abi_version = RT_RUNTIME_ABI_VERSION,
    .family = QWEN36_RUNTIME_FAMILY,
    .display_name = "Qwen3.6-27B",
    .probe_model_path = qwen36_rt_probe_model_path,
    .engine_open = qwen36_rt_engine_open,
    .engine_close = qwen36_rt_engine_close,
    .engine_summary = qwen36_rt_engine_summary,
    .engine_vocab_size = qwen36_rt_engine_vocab_size,
    .estimate_context_memory = qwen36_rt_estimate_context_memory,
    .tokenize_text = qwen36_rt_tokenize_text,
    .token_text = qwen36_rt_token_text,
    .token_eos = qwen36_rt_token_eos,
    .render_chat = qwen36_rt_render_chat,
    .session_create = qwen36_rt_session_create,
    .session_free = qwen36_rt_session_free,
    .session_set_progress = qwen36_rt_session_set_progress,
    .session_sync = qwen36_rt_session_sync,
    .session_eval = qwen36_rt_session_eval,
    .session_eval_speculative_argmax = qwen36_rt_session_eval_speculative_argmax,
    .session_argmax = qwen36_rt_session_argmax,
    .session_sample = qwen36_rt_session_sample,
    .session_top_logprobs = qwen36_rt_session_top_logprobs,
    .session_token_logprob = qwen36_rt_session_token_logprob,
    .session_read_logits = qwen36_rt_session_read_logits,
    .session_invalidate = qwen36_rt_session_invalidate,
    .session_rewind = qwen36_rt_session_rewind,
    .session_pos = qwen36_rt_session_pos,
    .session_ctx = qwen36_rt_session_ctx,
    .session_tokens = qwen36_rt_session_tokens,
    .session_payload_bytes = qwen36_rt_session_payload_bytes,
    .session_save_payload = qwen36_rt_session_save_payload,
    .session_load_payload = qwen36_rt_session_load_payload,
};

const rt_model_ops *qwen36_runtime_ops(void) {
    return &QWEN36_RUNTIME_OPS;
}

int qwen36_runtime_register(void) {
    return rt_register_model(&QWEN36_RUNTIME_OPS);
}
