#include "rt_runtime.h"

#include <stdlib.h>
#include <string.h>

#define RT_MAX_MODELS 16

static const rt_model_ops *g_models[RT_MAX_MODELS];
static size_t g_n_models;

static bool rt_model_ops_valid(const rt_model_ops *ops) {
    return ops &&
           ops->abi_version == RT_RUNTIME_ABI_VERSION &&
           ops->family &&
           ops->family[0] &&
           ops->engine_open &&
           ops->engine_close &&
           ops->session_create &&
           ops->session_free;
}

const char *rt_backend_name(rt_backend backend) {
    switch (backend) {
    case RT_BACKEND_AUTO:  return "auto";
    case RT_BACKEND_CPU:   return "cpu";
    case RT_BACKEND_METAL: return "metal";
    case RT_BACKEND_CUDA:  return "cuda";
    case RT_BACKEND_ROCM:  return "rocm";
    case RT_BACKEND_MLX:   return "mlx";
    default:               return "unknown";
    }
}

void rt_tokens_push(rt_tokens *tokens, int token) {
    if (!tokens) return;
    if (tokens->len == tokens->cap) {
        int newcap = tokens->cap ? tokens->cap * 2 : 32;
        int *newv = realloc(tokens->v, (size_t)newcap * sizeof(tokens->v[0]));
        if (!newv) abort();
        tokens->v = newv;
        tokens->cap = newcap;
    }
    tokens->v[tokens->len++] = token;
}

void rt_tokens_free(rt_tokens *tokens) {
    if (!tokens) return;
    free(tokens->v);
    tokens->v = NULL;
    tokens->len = 0;
    tokens->cap = 0;
}

void rt_tokens_copy(rt_tokens *dst, const rt_tokens *src) {
    if (!dst || !src) return;
    if (dst == src) return;
    rt_tokens_free(dst);
    if (src->len == 0) return;
    dst->v = malloc((size_t)src->len * sizeof(dst->v[0]));
    if (!dst->v) abort();
    memcpy(dst->v, src->v, (size_t)src->len * sizeof(dst->v[0]));
    dst->len = src->len;
    dst->cap = src->len;
}

bool rt_tokens_starts_with(const rt_tokens *tokens, const rt_tokens *prefix) {
    if (!tokens || !prefix || prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}

int rt_register_model(const rt_model_ops *ops) {
    if (!rt_model_ops_valid(ops)) return -1;
    for (size_t i = 0; i < g_n_models; i++) {
        if (!strcmp(g_models[i]->family, ops->family)) {
            return g_models[i] == ops ? 0 : -1;
        }
    }
    if (g_n_models == RT_MAX_MODELS) return -1;
    g_models[g_n_models++] = ops;
    return 0;
}

const rt_model_ops *rt_model_by_family(const char *family) {
    if (!family || !family[0]) return NULL;
    for (size_t i = 0; i < g_n_models; i++) {
        if (!strcmp(g_models[i]->family, family)) return g_models[i];
    }
    return NULL;
}

const rt_model_ops *rt_probe_model_path(const char *model_path) {
    if (!model_path || !model_path[0]) return NULL;
    for (size_t i = 0; i < g_n_models; i++) {
        const rt_model_ops *ops = g_models[i];
        if (ops->probe_model_path && ops->probe_model_path(model_path)) return ops;
    }
    return NULL;
}

int rt_engine_open_with_ops(rt_engine **out, const rt_model_ops *ops, const rt_engine_options *opt) {
    if (!out || !rt_model_ops_valid(ops) || !opt) return -1;
    *out = NULL;

    rt_engine *engine = calloc(1, sizeof(*engine));
    if (!engine) return -1;

    void *impl = NULL;
    if (ops->engine_open(&impl, opt) != 0 || !impl) {
        free(engine);
        return -1;
    }

    engine->ops = ops;
    engine->impl = impl;
    *out = engine;
    return 0;
}

int rt_engine_open(rt_engine **out, const rt_engine_options *opt, const char *family) {
    if (!out || !opt) return -1;
    const rt_model_ops *ops = family && family[0] ?
        rt_model_by_family(family) :
        rt_probe_model_path(opt->model_path);
    if (!ops) return -1;
    return rt_engine_open_with_ops(out, ops, opt);
}

void rt_engine_close(rt_engine *engine) {
    if (!engine) return;
    if (engine->ops && engine->ops->engine_close) {
        engine->ops->engine_close(engine->impl);
    }
    free(engine);
}

void rt_engine_summary(rt_engine *engine) {
    if (engine && engine->ops->engine_summary) engine->ops->engine_summary(engine->impl);
}

uint32_t rt_engine_vocab_size(rt_engine *engine) {
    if (!engine || !engine->ops->engine_vocab_size) return 0;
    return engine->ops->engine_vocab_size(engine->impl);
}

rt_context_memory rt_estimate_context_memory(const rt_model_ops *ops, rt_backend backend, int ctx_size) {
    if (ops && ops->estimate_context_memory) return ops->estimate_context_memory(backend, ctx_size);
    return (rt_context_memory){0};
}

int rt_tokenize_text(rt_engine *engine, const char *text, rt_tokens *out) {
    if (!engine || !engine->ops->tokenize_text) return -1;
    return engine->ops->tokenize_text(engine->impl, text, out);
}

char *rt_token_text(rt_engine *engine, int token, size_t *len) {
    if (!engine || !engine->ops->token_text) return NULL;
    return engine->ops->token_text(engine->impl, token, len);
}

int rt_token_eos(rt_engine *engine) {
    if (!engine || !engine->ops->token_eos) return -1;
    return engine->ops->token_eos(engine->impl);
}

int rt_render_chat(rt_engine *engine,
                   const rt_chat_message *messages,
                   size_t n_messages,
                   const rt_chat_render_options *options,
                   rt_tokens *out) {
    if (!engine || !engine->ops->render_chat) return -1;
    return engine->ops->render_chat(engine->impl, messages, n_messages, options, out);
}

int rt_session_create(rt_session **out, rt_engine *engine, int ctx_size) {
    if (!out || !engine || !engine->ops->session_create) return -1;
    *out = NULL;

    rt_session *session = calloc(1, sizeof(*session));
    if (!session) return -1;

    void *impl = NULL;
    if (engine->ops->session_create(&impl, engine->impl, ctx_size) != 0 || !impl) {
        free(session);
        return -1;
    }

    session->engine = engine;
    session->impl = impl;
    *out = session;
    return 0;
}

void rt_session_free(rt_session *session) {
    if (!session) return;
    if (session->engine && session->engine->ops->session_free) {
        session->engine->ops->session_free(session->impl);
    }
    free(session);
}

void rt_session_set_progress(rt_session *session, rt_session_progress_fn fn, void *ud) {
    if (session && session->engine->ops->session_set_progress) {
        session->engine->ops->session_set_progress(session->impl, fn, ud);
    }
}

int rt_session_sync(rt_session *session, const rt_tokens *prompt, char *err, size_t errlen) {
    if (!session || !session->engine->ops->session_sync) return -1;
    return session->engine->ops->session_sync(session->impl, prompt, err, errlen);
}

int rt_session_eval(rt_session *session, int token, char *err, size_t errlen) {
    if (!session || !session->engine->ops->session_eval) return -1;
    return session->engine->ops->session_eval(session->impl, token, err, errlen);
}

int rt_session_argmax(rt_session *session) {
    if (!session || !session->engine->ops->session_argmax) return -1;
    return session->engine->ops->session_argmax(session->impl);
}

int rt_session_sample(rt_session *session, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!session || !session->engine->ops->session_sample) return -1;
    return session->engine->ops->session_sample(session->impl, temperature, top_k, top_p, min_p, rng);
}

int rt_session_top_logprobs(rt_session *session, rt_token_score *out, int k) {
    if (!session || !session->engine->ops->session_top_logprobs) return -1;
    return session->engine->ops->session_top_logprobs(session->impl, out, k);
}

int rt_session_token_logprob(rt_session *session, int token, rt_token_score *out) {
    if (!session || !session->engine->ops->session_token_logprob) return -1;
    return session->engine->ops->session_token_logprob(session->impl, token, out);
}

int rt_session_read_logits(rt_session *session, float *out, uint32_t cap) {
    if (!session || !session->engine->ops->session_read_logits) return -1;
    return session->engine->ops->session_read_logits(session->impl, out, cap);
}

void rt_session_invalidate(rt_session *session) {
    if (session && session->engine->ops->session_invalidate) {
        session->engine->ops->session_invalidate(session->impl);
    }
}

void rt_session_rewind(rt_session *session, int pos) {
    if (session && session->engine->ops->session_rewind) {
        session->engine->ops->session_rewind(session->impl, pos);
    }
}

int rt_session_pos(rt_session *session) {
    if (!session || !session->engine->ops->session_pos) return -1;
    return session->engine->ops->session_pos(session->impl);
}

int rt_session_ctx(rt_session *session) {
    if (!session || !session->engine->ops->session_ctx) return -1;
    return session->engine->ops->session_ctx(session->impl);
}

uint64_t rt_session_payload_bytes(rt_session *session) {
    if (!session || !session->engine->ops->session_payload_bytes) return 0;
    return session->engine->ops->session_payload_bytes(session->impl);
}

int rt_session_save_payload(rt_session *session, FILE *fp, char *err, size_t errlen) {
    if (!session || !session->engine->ops->session_save_payload) return -1;
    return session->engine->ops->session_save_payload(session->impl, fp, err, errlen);
}

int rt_session_load_payload(rt_session *session, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!session || !session->engine->ops->session_load_payload) return -1;
    return session->engine->ops->session_load_payload(session->impl, fp, payload_bytes, err, errlen);
}
