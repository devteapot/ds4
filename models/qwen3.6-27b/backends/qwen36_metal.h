#ifndef QWEN36_METAL_H
#define QWEN36_METAL_H

#include "rt_runtime.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct qwen36_metal_backend qwen36_metal_backend;

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
bool qwen36_metal_available(void);
qwen36_metal_backend *qwen36_metal_create(char *err, size_t errlen);
void qwen36_metal_destroy(qwen36_metal_backend *backend);
bool qwen36_metal_matvec(qwen36_metal_backend *backend,
                         const rt_gguf_file *file,
                         const rt_gguf_tensor *tensor,
                         const float *x,
                         uint64_t in_dim,
                         float *out,
                         uint64_t out_dim,
                         char *err,
                         size_t errlen);
bool qwen36_metal_rms_norm(qwen36_metal_backend *backend,
                           const rt_gguf_file *file,
                           const rt_gguf_tensor *weight,
                           const float *x,
                           uint64_t dim,
                           float *out,
                           bool add_one_to_weight,
                           char *err,
                           size_t errlen);
bool qwen36_metal_silu_mul(qwen36_metal_backend *backend,
                           const float *gate,
                           const float *up,
                           uint64_t dim,
                           float *out,
                           char *err,
                           size_t errlen);
bool qwen36_metal_l2_norm(qwen36_metal_backend *backend,
                          const float *x,
                          uint64_t dim,
                          float *out,
                          char *err,
                          size_t errlen);
bool qwen36_metal_gated_delta_head(qwen36_metal_backend *backend,
                                   float *state,
                                   const float *query,
                                   const float *key,
                                   const float *value,
                                   uint64_t dim,
                                   float decay,
                                   float beta,
                                   float qscale,
                                   float *out,
                                   char *err,
                                   size_t errlen);
bool qwen36_metal_full_attention_head(qwen36_metal_backend *backend,
                                      const float *query,
                                      const float *gate,
                                      const float *keys,
                                      const float *values,
                                      uint64_t seq_len,
                                      uint64_t kv_stride,
                                      uint64_t head_dim,
                                      float *out,
                                      char *err,
                                      size_t errlen);
#else
static inline bool qwen36_metal_available(void) {
    return false;
}

static inline qwen36_metal_backend *qwen36_metal_create(char *err, size_t errlen) {
    (void)err;
    (void)errlen;
    return NULL;
}

static inline void qwen36_metal_destroy(qwen36_metal_backend *backend) {
    (void)backend;
}

static inline bool qwen36_metal_matvec(qwen36_metal_backend *backend,
                                       const rt_gguf_file *file,
                                       const rt_gguf_tensor *tensor,
                                       const float *x,
                                       uint64_t in_dim,
                                       float *out,
                                       uint64_t out_dim,
                                       char *err,
                                       size_t errlen) {
    (void)backend;
    (void)file;
    (void)tensor;
    (void)x;
    (void)in_dim;
    (void)out;
    (void)out_dim;
    (void)err;
    (void)errlen;
    return false;
}

static inline bool qwen36_metal_rms_norm(qwen36_metal_backend *backend,
                                         const rt_gguf_file *file,
                                         const rt_gguf_tensor *weight,
                                         const float *x,
                                         uint64_t dim,
                                         float *out,
                                         bool add_one_to_weight,
                                         char *err,
                                         size_t errlen) {
    (void)backend;
    (void)file;
    (void)weight;
    (void)x;
    (void)dim;
    (void)out;
    (void)add_one_to_weight;
    (void)err;
    (void)errlen;
    return false;
}

static inline bool qwen36_metal_silu_mul(qwen36_metal_backend *backend,
                                         const float *gate,
                                         const float *up,
                                         uint64_t dim,
                                         float *out,
                                         char *err,
                                         size_t errlen) {
    (void)backend;
    (void)gate;
    (void)up;
    (void)dim;
    (void)out;
    (void)err;
    (void)errlen;
    return false;
}

static inline bool qwen36_metal_l2_norm(qwen36_metal_backend *backend,
                                        const float *x,
                                        uint64_t dim,
                                        float *out,
                                        char *err,
                                        size_t errlen) {
    (void)backend;
    (void)x;
    (void)dim;
    (void)out;
    (void)err;
    (void)errlen;
    return false;
}

static inline bool qwen36_metal_gated_delta_head(qwen36_metal_backend *backend,
                                                 float *state,
                                                 const float *query,
                                                 const float *key,
                                                 const float *value,
                                                 uint64_t dim,
                                                 float decay,
                                                 float beta,
                                                 float qscale,
                                                 float *out,
                                                 char *err,
                                                 size_t errlen) {
    (void)backend;
    (void)state;
    (void)query;
    (void)key;
    (void)value;
    (void)dim;
    (void)decay;
    (void)beta;
    (void)qscale;
    (void)out;
    (void)err;
    (void)errlen;
    return false;
}

static inline bool qwen36_metal_full_attention_head(qwen36_metal_backend *backend,
                                                    const float *query,
                                                    const float *gate,
                                                    const float *keys,
                                                    const float *values,
                                                    uint64_t seq_len,
                                                    uint64_t kv_stride,
                                                    uint64_t head_dim,
                                                    float *out,
                                                    char *err,
                                                    size_t errlen) {
    (void)backend;
    (void)query;
    (void)gate;
    (void)keys;
    (void)values;
    (void)seq_len;
    (void)kv_stride;
    (void)head_dim;
    (void)out;
    (void)err;
    (void)errlen;
    return false;
}
#endif

#endif /* QWEN36_METAL_H */
