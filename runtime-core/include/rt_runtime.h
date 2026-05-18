#ifndef RT_RUNTIME_H
#define RT_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define RT_RUNTIME_ABI_VERSION 1u

typedef enum {
    RT_BACKEND_AUTO = 0,
    RT_BACKEND_CPU,
    RT_BACKEND_METAL,
    RT_BACKEND_CUDA,
    RT_BACKEND_ROCM,
    RT_BACKEND_MLX,
} rt_backend;

typedef struct {
    int *v;
    int len;
    int cap;
} rt_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} rt_token_score;

typedef struct {
    uint64_t total_bytes;
    uint64_t state_bytes;
    uint64_t scratch_bytes;
    uint64_t model_private_bytes;
    uint32_t prefill_cap;
    uint32_t model_caps[3];
} rt_context_memory;

typedef struct {
    const char *role;
    const char *content;
} rt_chat_message;

typedef struct {
    bool add_generation_prompt;
    const void *model_options;
} rt_chat_render_options;

typedef struct {
    const char *model_path;
    rt_backend backend;
    int n_threads;
    bool warm_weights;
    bool quality;
    const void *model_options;
} rt_engine_options;

typedef struct rt_engine rt_engine;
typedef struct rt_session rt_session;
typedef struct rt_model_ops rt_model_ops;

typedef void (*rt_session_progress_fn)(void *ud, const char *event, int current, int total);

struct rt_model_ops {
    uint32_t abi_version;
    const char *family;
    const char *display_name;

    bool (*probe_model_path)(const char *model_path);

    int  (*engine_open)(void **out, const rt_engine_options *opt);
    void (*engine_close)(void *engine);
    void (*engine_summary)(void *engine);
    uint32_t (*engine_vocab_size)(void *engine);
    rt_context_memory (*estimate_context_memory)(rt_backend backend, int ctx_size);

    int   (*tokenize_text)(void *engine, const char *text, rt_tokens *out);
    char *(*token_text)(void *engine, int token, size_t *len);
    int   (*token_eos)(void *engine);
    int   (*render_chat)(void *engine,
                         const rt_chat_message *messages,
                         size_t n_messages,
                         const rt_chat_render_options *options,
                         rt_tokens *out);

    int  (*session_create)(void **out, void *engine, int ctx_size);
    void (*session_free)(void *session);
    void (*session_set_progress)(void *session, rt_session_progress_fn fn, void *ud);
    int  (*session_sync)(void *session, const rt_tokens *prompt, char *err, size_t errlen);
    int  (*session_eval)(void *session, int token, char *err, size_t errlen);
    int  (*session_argmax)(void *session);
    int  (*session_sample)(void *session,
                           float temperature,
                           int top_k,
                           float top_p,
                           float min_p,
                           uint64_t *rng);
    int  (*session_top_logprobs)(void *session, rt_token_score *out, int k);
    int  (*session_token_logprob)(void *session, int token, rt_token_score *out);
    int  (*session_read_logits)(void *session, float *out, uint32_t cap);
    void (*session_invalidate)(void *session);
    void (*session_rewind)(void *session, int pos);
    int  (*session_pos)(void *session);
    int  (*session_ctx)(void *session);

    uint64_t (*session_payload_bytes)(void *session);
    int (*session_save_payload)(void *session, FILE *fp, char *err, size_t errlen);
    int (*session_load_payload)(void *session, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
};

struct rt_engine {
    const rt_model_ops *ops;
    void *impl;
};

struct rt_session {
    rt_engine *engine;
    void *impl;
};

const char *rt_backend_name(rt_backend backend);

void rt_tokens_push(rt_tokens *tokens, int token);
void rt_tokens_free(rt_tokens *tokens);
void rt_tokens_copy(rt_tokens *dst, const rt_tokens *src);
bool rt_tokens_starts_with(const rt_tokens *tokens, const rt_tokens *prefix);

int rt_register_model(const rt_model_ops *ops);
const rt_model_ops *rt_model_by_family(const char *family);
const rt_model_ops *rt_probe_model_path(const char *model_path);

int rt_engine_open_with_ops(rt_engine **out, const rt_model_ops *ops, const rt_engine_options *opt);
int rt_engine_open(rt_engine **out, const rt_engine_options *opt, const char *family);
void rt_engine_close(rt_engine *engine);
void rt_engine_summary(rt_engine *engine);
uint32_t rt_engine_vocab_size(rt_engine *engine);
rt_context_memory rt_estimate_context_memory(const rt_model_ops *ops, rt_backend backend, int ctx_size);

int rt_tokenize_text(rt_engine *engine, const char *text, rt_tokens *out);
char *rt_token_text(rt_engine *engine, int token, size_t *len);
int rt_token_eos(rt_engine *engine);
int rt_render_chat(rt_engine *engine,
                   const rt_chat_message *messages,
                   size_t n_messages,
                   const rt_chat_render_options *options,
                   rt_tokens *out);

int rt_session_create(rt_session **out, rt_engine *engine, int ctx_size);
void rt_session_free(rt_session *session);
void rt_session_set_progress(rt_session *session, rt_session_progress_fn fn, void *ud);
int rt_session_sync(rt_session *session, const rt_tokens *prompt, char *err, size_t errlen);
int rt_session_eval(rt_session *session, int token, char *err, size_t errlen);
int rt_session_argmax(rt_session *session);
int rt_session_sample(rt_session *session, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int rt_session_top_logprobs(rt_session *session, rt_token_score *out, int k);
int rt_session_token_logprob(rt_session *session, int token, rt_token_score *out);
int rt_session_read_logits(rt_session *session, float *out, uint32_t cap);
void rt_session_invalidate(rt_session *session);
void rt_session_rewind(rt_session *session, int pos);
int rt_session_pos(rt_session *session);
int rt_session_ctx(rt_session *session);
uint64_t rt_session_payload_bytes(rt_session *session);
int rt_session_save_payload(rt_session *session, FILE *fp, char *err, size_t errlen);
int rt_session_load_payload(rt_session *session, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);

#endif
