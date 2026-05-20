#ifndef RT_RUNTIME_H
#define RT_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define RT_RUNTIME_ABI_VERSION 3u

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

typedef enum {
    RT_GGUF_VALUE_UINT8   = 0,
    RT_GGUF_VALUE_INT8    = 1,
    RT_GGUF_VALUE_UINT16  = 2,
    RT_GGUF_VALUE_INT16   = 3,
    RT_GGUF_VALUE_UINT32  = 4,
    RT_GGUF_VALUE_INT32   = 5,
    RT_GGUF_VALUE_FLOAT32 = 6,
    RT_GGUF_VALUE_BOOL    = 7,
    RT_GGUF_VALUE_STRING  = 8,
    RT_GGUF_VALUE_ARRAY   = 9,
    RT_GGUF_VALUE_UINT64  = 10,
    RT_GGUF_VALUE_INT64   = 11,
    RT_GGUF_VALUE_FLOAT64 = 12,
} rt_gguf_value_type;

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_offset;
} rt_gguf_array_ref;

typedef union {
    uint64_t u64;
    int64_t i64;
    double f64;
    bool bool_value;
    char *string;
    rt_gguf_array_ref array;
} rt_gguf_value;

typedef struct {
    char *key;
    uint32_t type;
    rt_gguf_value value;
} rt_gguf_kv;

typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint64_t alignment;
    rt_gguf_kv *kv;
} rt_gguf_metadata;

#define RT_GGUF_MAX_DIMS 8u

typedef struct {
    char *name;
    uint32_t ndim;
    uint64_t dim[RT_GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t data_offset;
    uint64_t elements;
    uint64_t bytes;
} rt_gguf_tensor;

typedef struct rt_gguf_file rt_gguf_file;

typedef struct {
    const rt_gguf_file *file;
    rt_gguf_array_ref array;
    uint64_t offset;
    uint64_t index;
} rt_gguf_array_cursor;

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
    int  (*session_eval_speculative_argmax)(void *session,
                                            int first_token,
                                            int max_tokens,
                                            int eos_token,
                                            int *out_tokens,
                                            int out_cap,
                                            char *err,
                                            size_t errlen);
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
    const rt_tokens *(*session_tokens)(void *session);

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

int rt_gguf_metadata_read(const char *path, rt_gguf_metadata *out, char *err, size_t errlen);
void rt_gguf_metadata_free(rt_gguf_metadata *meta);
const rt_gguf_kv *rt_gguf_metadata_find(const rt_gguf_metadata *meta, const char *key);
bool rt_gguf_metadata_get_string(const rt_gguf_metadata *meta, const char *key, const char **out);
bool rt_gguf_metadata_get_u32(const rt_gguf_metadata *meta, const char *key, uint32_t *out);
bool rt_gguf_metadata_get_u64(const rt_gguf_metadata *meta, const char *key, uint64_t *out);
bool rt_gguf_metadata_get_bool(const rt_gguf_metadata *meta, const char *key, bool *out);
bool rt_gguf_metadata_get_array(const rt_gguf_metadata *meta, const char *key, rt_gguf_array_ref *out);

const char *rt_gguf_tensor_type_name(uint32_t type);
bool rt_gguf_tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);
int rt_gguf_file_open(rt_gguf_file **out, const char *path, char *err, size_t errlen);
void rt_gguf_file_close(rt_gguf_file *file);
const char *rt_gguf_file_path(const rt_gguf_file *file);
uint64_t rt_gguf_file_size(const rt_gguf_file *file);
const rt_gguf_metadata *rt_gguf_file_metadata(const rt_gguf_file *file);
uint64_t rt_gguf_file_tensor_count(const rt_gguf_file *file);
const rt_gguf_tensor *rt_gguf_file_tensor(const rt_gguf_file *file, uint64_t index);
const rt_gguf_tensor *rt_gguf_file_find_tensor(const rt_gguf_file *file, const char *name);
const void *rt_gguf_file_tensor_data(const rt_gguf_file *file, const rt_gguf_tensor *tensor);
bool rt_gguf_tensor_read_f32(const rt_gguf_file *file,
                             const rt_gguf_tensor *tensor,
                             uint64_t index,
                             float *out);
bool rt_gguf_tensor_matvec_f32(const rt_gguf_file *file,
                               const rt_gguf_tensor *tensor,
                               const float *x,
                               uint64_t in_dim,
                               float *out,
                               uint64_t out_dim);
bool rt_gguf_file_array_cursor(const rt_gguf_file *file, const char *key, rt_gguf_array_cursor *out);
bool rt_gguf_array_cursor_next_string(rt_gguf_array_cursor *cursor, const char **ptr, size_t *len);
bool rt_gguf_file_array_string_at(const rt_gguf_file *file,
                                  const char *key,
                                  uint64_t index,
                                  const char **ptr,
                                  size_t *len);

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
int rt_session_eval_speculative_argmax(rt_session *session,
                                       int first_token,
                                       int max_tokens,
                                       int eos_token,
                                       int *out_tokens,
                                       int out_cap,
                                       char *err,
                                       size_t errlen);
int rt_session_argmax(rt_session *session);
int rt_session_sample(rt_session *session, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int rt_session_top_logprobs(rt_session *session, rt_token_score *out, int k);
int rt_session_token_logprob(rt_session *session, int token, rt_token_score *out);
int rt_session_read_logits(rt_session *session, float *out, uint32_t cap);
void rt_session_invalidate(rt_session *session);
void rt_session_rewind(rt_session *session, int pos);
int rt_session_pos(rt_session *session);
int rt_session_ctx(rt_session *session);
const rt_tokens *rt_session_tokens(rt_session *session);
uint64_t rt_session_payload_bytes(rt_session *session);
int rt_session_save_payload(rt_session *session, FILE *fp, char *err, size_t errlen);
int rt_session_load_payload(rt_session *session, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);

#endif
