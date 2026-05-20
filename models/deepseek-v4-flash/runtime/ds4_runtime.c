#include "ds4_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ds4_backend ds4_backend_from_rt(rt_backend backend) {
    switch (backend) {
    case RT_BACKEND_CPU:
        return DS4_BACKEND_CPU;
    case RT_BACKEND_METAL:
        return DS4_BACKEND_METAL;
    case RT_BACKEND_CUDA:
        return DS4_BACKEND_CUDA;
    case RT_BACKEND_AUTO:
#ifdef DS4_NO_GPU
        return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
        return DS4_BACKEND_METAL;
#else
        return DS4_BACKEND_CUDA;
#endif
    case RT_BACKEND_ROCM:
    case RT_BACKEND_MLX:
    default:
        return DS4_BACKEND_CPU;
    }
}

static bool ds4_rt_probe_model_path(const char *model_path) {
    rt_gguf_metadata meta = {0};
    char err[256];

    if (!model_path) return false;
    if (rt_gguf_metadata_read(model_path, &meta, err, sizeof(err)) == 0) {
        const char *arch = NULL;
        bool ok = rt_gguf_metadata_get_string(&meta, "general.architecture", &arch) &&
                  arch &&
                  !strcmp(arch, "deepseek4");
        rt_gguf_metadata_free(&meta);
        return ok;
    }

    if (model_path && model_path[0]) {
        FILE *fp = fopen(model_path, "rb");
        if (fp) {
            fclose(fp);
            return false;
        }
    }
    return strstr(model_path, "ds4flash") ||
           strstr(model_path, "DeepSeek-V4") ||
           strstr(model_path, "deepseek-v4");
}

static int ds4_rt_engine_open(void **out, const rt_engine_options *opt) {
    if (!out || !opt) return 1;
    *out = NULL;

    const ds4_runtime_engine_options *extra = opt->model_options;
    rt_backend requested = opt->backend;
    if (requested == RT_BACKEND_ROCM || requested == RT_BACKEND_MLX) return 1;

    ds4_engine_options ds4_opt = {
        .model_path = opt->model_path,
        .backend = ds4_backend_from_rt(requested),
        .n_threads = opt->n_threads,
        .warm_weights = opt->warm_weights,
        .quality = opt->quality,
    };
    if (extra) {
        ds4_opt.mtp_path = extra->mtp_path;
        ds4_opt.mtp_draft_tokens = extra->mtp_draft_tokens;
        ds4_opt.mtp_margin = extra->mtp_margin;
        ds4_opt.directional_steering_file = extra->directional_steering_file;
        ds4_opt.directional_steering_attn = extra->directional_steering_attn;
        ds4_opt.directional_steering_ffn = extra->directional_steering_ffn;
    }

    ds4_engine *engine = NULL;
    int rc = ds4_engine_open(&engine, &ds4_opt);
    if (rc == 0) *out = engine;
    return rc;
}

static void ds4_rt_engine_close(void *engine) {
    ds4_engine_close(engine);
}

static void ds4_rt_engine_summary(void *engine) {
    ds4_engine_summary(engine);
}

static uint32_t ds4_rt_engine_vocab_size(void *engine) {
    return ds4_engine_vocab_size(engine);
}

static rt_context_memory ds4_rt_estimate_context_memory(rt_backend backend, int ctx_size) {
    if (backend == RT_BACKEND_ROCM || backend == RT_BACKEND_MLX) return (rt_context_memory){0};
    ds4_context_memory ds4_mem = ds4_context_memory_estimate(ds4_backend_from_rt(backend), ctx_size);
    return (rt_context_memory){
        .total_bytes = ds4_mem.total_bytes,
        .state_bytes = ds4_mem.raw_bytes + ds4_mem.compressed_bytes,
        .scratch_bytes = ds4_mem.scratch_bytes,
        .model_private_bytes = 0,
        .prefill_cap = ds4_mem.prefill_cap,
        .model_caps = { ds4_mem.raw_cap, ds4_mem.comp_cap, 0 },
    };
}

static void ds4_tokens_from_rt(ds4_tokens *dst, const rt_tokens *src) {
    dst->v = src ? src->v : NULL;
    dst->len = src ? src->len : 0;
    dst->cap = src ? src->cap : 0;
}

static void rt_tokens_from_ds4(rt_tokens *dst, const ds4_tokens *src) {
    dst->v = src->v;
    dst->len = src->len;
    dst->cap = src->cap;
}

static int ds4_rt_tokenize_text(void *engine, const char *text, rt_tokens *out) {
    if (!engine || !text || !out) return 1;
    ds4_tokens tmp;
    ds4_tokens_from_rt(&tmp, out);
    ds4_tokenize_text(engine, text, &tmp);
    rt_tokens_from_ds4(out, &tmp);
    return 0;
}

static char *ds4_rt_token_text(void *engine, int token, size_t *len) {
    return ds4_token_text(engine, token, len);
}

static int ds4_rt_token_eos(void *engine) {
    return ds4_token_eos(engine);
}

static int ds4_rt_render_chat(
        void *engine,
        const rt_chat_message *messages,
        size_t n_messages,
        const rt_chat_render_options *options,
        rt_tokens *out) {
    if (!engine || (!messages && n_messages != 0) || !out) return 1;

    const ds4_runtime_chat_options *ds4_chat =
        options ? options->model_options : NULL;
    ds4_think_mode think_mode = ds4_chat ? ds4_chat->think_mode : DS4_THINK_HIGH;
    bool add_generation_prompt = options ? options->add_generation_prompt : true;

    ds4_tokens tmp;
    ds4_tokens_from_rt(&tmp, out);
    ds4_chat_begin(engine, &tmp);
    for (size_t i = 0; i < n_messages; i++) {
        ds4_chat_append_message(engine, &tmp, messages[i].role, messages[i].content);
    }
    if (add_generation_prompt) {
        ds4_chat_append_assistant_prefix(engine, &tmp, think_mode);
    }
    rt_tokens_from_ds4(out, &tmp);
    return 0;
}

static int ds4_rt_session_create(void **out, void *engine, int ctx_size) {
    if (!out || !engine) return 1;
    *out = NULL;
    ds4_session *session = NULL;
    int rc = ds4_session_create(&session, engine, ctx_size);
    if (rc == 0) *out = session;
    return rc;
}

static void ds4_rt_session_free(void *session) {
    ds4_session_free(session);
}

static void ds4_rt_session_set_progress(void *session, rt_session_progress_fn fn, void *ud) {
    ds4_session_set_progress(session, (ds4_session_progress_fn)fn, ud);
}

static int ds4_rt_session_sync(void *session, const rt_tokens *prompt, char *err, size_t errlen) {
    ds4_tokens tmp;
    ds4_tokens_from_rt(&tmp, prompt);
    return ds4_session_sync(session, &tmp, err, errlen);
}

static int ds4_rt_session_eval(void *session, int token, char *err, size_t errlen) {
    return ds4_session_eval(session, token, err, errlen);
}

static int ds4_rt_session_eval_speculative_argmax(void *session,
                                                  int first_token,
                                                  int max_tokens,
                                                  int eos_token,
                                                  int *out_tokens,
                                                  int out_cap,
                                                  char *err,
                                                  size_t errlen) {
    return ds4_session_eval_speculative_argmax(session, first_token,
                                              max_tokens, eos_token,
                                              out_tokens, out_cap,
                                              err, errlen);
}

static int ds4_rt_session_argmax(void *session) {
    return ds4_session_argmax(session);
}

static int ds4_rt_session_sample(void *session, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    return ds4_session_sample(session, temperature, top_k, top_p, min_p, rng);
}

static int ds4_rt_session_top_logprobs(void *session, rt_token_score *out, int k) {
    if (!out || k <= 0) return 0;
    ds4_token_score *tmp = calloc((size_t)k, sizeof(tmp[0]));
    if (!tmp) return 0;
    int n = ds4_session_top_logprobs(session, tmp, k);
    for (int i = 0; i < n; i++) {
        out[i].id = tmp[i].id;
        out[i].logit = tmp[i].logit;
        out[i].logprob = tmp[i].logprob;
    }
    free(tmp);
    return n;
}

static int ds4_rt_session_token_logprob(void *session, int token, rt_token_score *out) {
    if (!out) return 1;
    ds4_token_score score = {0};
    int rc = ds4_session_token_logprob(session, token, &score);
    out->id = score.id;
    out->logit = score.logit;
    out->logprob = score.logprob;
    return rc;
}

static int ds4_rt_session_read_logits(void *session, float *out, uint32_t cap) {
    return ds4_session_read_logits(session, out, cap);
}

static void ds4_rt_session_invalidate(void *session) {
    ds4_session_invalidate(session);
}

static void ds4_rt_session_rewind(void *session, int pos) {
    ds4_session_rewind(session, pos);
}

static int ds4_rt_session_pos(void *session) {
    return ds4_session_pos(session);
}

static int ds4_rt_session_ctx(void *session) {
    return ds4_session_ctx(session);
}

static uint64_t ds4_rt_session_payload_bytes(void *session) {
    return ds4_session_payload_bytes(session);
}

static int ds4_rt_session_save_payload(void *session, FILE *fp, char *err, size_t errlen) {
    return ds4_session_save_payload(session, fp, err, errlen);
}

static int ds4_rt_session_load_payload(void *session, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    return ds4_session_load_payload(session, fp, payload_bytes, err, errlen);
}

static const rt_model_ops DS4_RUNTIME_OPS = {
    .abi_version = RT_RUNTIME_ABI_VERSION,
    .family = "deepseek-v4-flash",
    .display_name = "DeepSeek V4 Flash",
    .probe_model_path = ds4_rt_probe_model_path,
    .engine_open = ds4_rt_engine_open,
    .engine_close = ds4_rt_engine_close,
    .engine_summary = ds4_rt_engine_summary,
    .engine_vocab_size = ds4_rt_engine_vocab_size,
    .estimate_context_memory = ds4_rt_estimate_context_memory,
    .tokenize_text = ds4_rt_tokenize_text,
    .token_text = ds4_rt_token_text,
    .token_eos = ds4_rt_token_eos,
    .render_chat = ds4_rt_render_chat,
    .session_create = ds4_rt_session_create,
    .session_free = ds4_rt_session_free,
    .session_set_progress = ds4_rt_session_set_progress,
    .session_sync = ds4_rt_session_sync,
    .session_eval = ds4_rt_session_eval,
    .session_eval_speculative_argmax = ds4_rt_session_eval_speculative_argmax,
    .session_argmax = ds4_rt_session_argmax,
    .session_sample = ds4_rt_session_sample,
    .session_top_logprobs = ds4_rt_session_top_logprobs,
    .session_token_logprob = ds4_rt_session_token_logprob,
    .session_read_logits = ds4_rt_session_read_logits,
    .session_invalidate = ds4_rt_session_invalidate,
    .session_rewind = ds4_rt_session_rewind,
    .session_pos = ds4_rt_session_pos,
    .session_ctx = ds4_rt_session_ctx,
    .session_payload_bytes = ds4_rt_session_payload_bytes,
    .session_save_payload = ds4_rt_session_save_payload,
    .session_load_payload = ds4_rt_session_load_payload,
};

const rt_model_ops *ds4_runtime_ops(void) {
    return &DS4_RUNTIME_OPS;
}

int ds4_runtime_register(void) {
    return rt_register_model(&DS4_RUNTIME_OPS);
}
