#include "rt_runtime.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define RT_MAX_MODELS 16
#define RT_GGUF_MAGIC 0x46554747u
#define RT_GGUF_DEFAULT_ALIGNMENT 32u
#define RT_GGUF_MAX_STRING_BYTES (64u * 1024u * 1024u)

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} rt_gguf_tensor_type_info;

static const rt_gguf_tensor_type_info rt_gguf_tensor_types[] = {
    [0]  = {"f32",       1,   4},
    [1]  = {"f16",       1,   2},
    [2]  = {"q4_0",     32,  18},
    [3]  = {"q4_1",     32,  20},
    [6]  = {"q5_0",     32,  22},
    [7]  = {"q5_1",     32,  24},
    [8]  = {"q8_0",     32,  34},
    [9]  = {"q8_1",     32,  36},
    [10] = {"q2_k",    256,  84},
    [11] = {"q3_k",    256, 110},
    [12] = {"q4_k",    256, 144},
    [13] = {"q5_k",    256, 176},
    [14] = {"q6_k",    256, 210},
    [15] = {"q8_k",    256, 292},
    [16] = {"iq2_xxs", 256,  66},
    [17] = {"iq2_xs",  256,  74},
    [18] = {"iq3_xxs", 256,  98},
    [19] = {"iq1_s",   256,  50},
    [20] = {"iq4_nl",   32,  18},
    [21] = {"iq3_s",   256, 110},
    [22] = {"iq2_s",   256,  82},
    [23] = {"iq4_xs",  256, 136},
    [24] = {"i8",        1,   1},
    [25] = {"i16",       1,   2},
    [26] = {"i32",       1,   4},
    [27] = {"i64",       1,   8},
    [28] = {"f64",       1,   8},
    [29] = {"iq1_m",   256,  56},
    [30] = {"bf16",      1,   2},
};

struct rt_gguf_file {
    char *path;
    int fd;
    const uint8_t *map;
    uint64_t size;
    uint64_t tensor_data_offset;
    rt_gguf_metadata meta;
    rt_gguf_tensor *tensors;
};

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

static void rt_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || errlen == 0 || err[0] != '\0') return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static char *rt_strdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *out = malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

static uint64_t rt_align_up(uint64_t value, uint64_t alignment) {
    if (alignment == 0) return value;
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

static int rt_file_offset(FILE *fp, uint64_t *out, char *err, size_t errlen) {
    long pos = ftell(fp);
    if (pos < 0) {
        rt_set_err(err, errlen, "GGUF ftell failed: %s", strerror(errno));
        return -1;
    }
    *out = (uint64_t)pos;
    return 0;
}

static int rt_read_exact(FILE *fp, void *dst, size_t n, const char *what, char *err, size_t errlen) {
    if (n == 0) return 0;
    if (fread(dst, 1, n, fp) != n) {
        rt_set_err(err, errlen, "short GGUF read while reading %s", what);
        return -1;
    }
    return 0;
}

static int rt_read_u8(FILE *fp, uint8_t *out, const char *what, char *err, size_t errlen) {
    return rt_read_exact(fp, out, sizeof(*out), what, err, errlen);
}

static int rt_read_u16(FILE *fp, uint16_t *out, const char *what, char *err, size_t errlen) {
    uint8_t b[2];
    if (rt_read_exact(fp, b, sizeof(b), what, err, errlen) != 0) return -1;
    *out = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
    return 0;
}

static int rt_read_u32(FILE *fp, uint32_t *out, const char *what, char *err, size_t errlen) {
    uint8_t b[4];
    if (rt_read_exact(fp, b, sizeof(b), what, err, errlen) != 0) return -1;
    *out = (uint32_t)b[0] |
           ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
    return 0;
}

static int rt_read_u64(FILE *fp, uint64_t *out, const char *what, char *err, size_t errlen) {
    uint8_t b[8];
    if (rt_read_exact(fp, b, sizeof(b), what, err, errlen) != 0) return -1;
    *out = (uint64_t)b[0] |
           ((uint64_t)b[1] << 8) |
           ((uint64_t)b[2] << 16) |
           ((uint64_t)b[3] << 24) |
           ((uint64_t)b[4] << 32) |
           ((uint64_t)b[5] << 40) |
           ((uint64_t)b[6] << 48) |
           ((uint64_t)b[7] << 56);
    return 0;
}

static char *rt_read_gguf_string(FILE *fp, const char *what, char *err, size_t errlen) {
    uint64_t len = 0;
    if (rt_read_u64(fp, &len, what, err, errlen) != 0) return NULL;
    if (len > RT_GGUF_MAX_STRING_BYTES || len > SIZE_MAX - 1) {
        rt_set_err(err, errlen, "GGUF string is too large while reading %s", what);
        return NULL;
    }

    char *s = malloc((size_t)len + 1);
    if (!s) {
        rt_set_err(err, errlen, "out of memory while reading GGUF string");
        return NULL;
    }
    if (rt_read_exact(fp, s, (size_t)len, what, err, errlen) != 0) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

static uint64_t rt_gguf_scalar_size(uint32_t type) {
    switch (type) {
    case RT_GGUF_VALUE_UINT8:
    case RT_GGUF_VALUE_INT8:
    case RT_GGUF_VALUE_BOOL:
        return 1;
    case RT_GGUF_VALUE_UINT16:
    case RT_GGUF_VALUE_INT16:
        return 2;
    case RT_GGUF_VALUE_UINT32:
    case RT_GGUF_VALUE_INT32:
    case RT_GGUF_VALUE_FLOAT32:
        return 4;
    case RT_GGUF_VALUE_UINT64:
    case RT_GGUF_VALUE_INT64:
    case RT_GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static int rt_skip_bytes(FILE *fp, uint64_t n, char *err, size_t errlen) {
    while (n > 0) {
        long step = n > (uint64_t)LONG_MAX ? LONG_MAX : (long)n;
        if (fseek(fp, step, SEEK_CUR) != 0) {
            rt_set_err(err, errlen, "GGUF seek failed: %s", strerror(errno));
            return -1;
        }
        n -= (uint64_t)step;
    }
    return 0;
}

static int rt_skip_gguf_value(FILE *fp, uint32_t type, int depth, char *err, size_t errlen) {
    if (depth > 8) {
        rt_set_err(err, errlen, "GGUF metadata array nesting is too deep");
        return -1;
    }

    uint64_t scalar = rt_gguf_scalar_size(type);
    if (scalar != 0) return rt_skip_bytes(fp, scalar, err, errlen);

    if (type == RT_GGUF_VALUE_STRING) {
        uint64_t len = 0;
        if (rt_read_u64(fp, &len, "GGUF string length", err, errlen) != 0) return -1;
        if (len > RT_GGUF_MAX_STRING_BYTES) {
            rt_set_err(err, errlen, "GGUF string is too large while skipping metadata");
            return -1;
        }
        return rt_skip_bytes(fp, len, err, errlen);
    }

    if (type == RT_GGUF_VALUE_ARRAY) {
        uint32_t item_type = 0;
        uint64_t len = 0;
        if (rt_read_u32(fp, &item_type, "GGUF array type", err, errlen) != 0) return -1;
        if (rt_read_u64(fp, &len, "GGUF array length", err, errlen) != 0) return -1;

        uint64_t item_size = rt_gguf_scalar_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                rt_set_err(err, errlen, "GGUF array byte size overflow");
                return -1;
            }
            return rt_skip_bytes(fp, len * item_size, err, errlen);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (rt_skip_gguf_value(fp, item_type, depth + 1, err, errlen) != 0) return -1;
        }
        return 0;
    }

    rt_set_err(err, errlen, "unknown GGUF metadata type %u", type);
    return -1;
}

static int rt_read_gguf_value(FILE *fp, rt_gguf_kv *kv, char *err, size_t errlen) {
    switch (kv->type) {
    case RT_GGUF_VALUE_UINT8: {
        uint8_t v = 0;
        if (rt_read_u8(fp, &v, kv->key, err, errlen) != 0) return -1;
        kv->value.u64 = v;
        return 0;
    }
    case RT_GGUF_VALUE_INT8: {
        uint8_t v = 0;
        if (rt_read_u8(fp, &v, kv->key, err, errlen) != 0) return -1;
        kv->value.i64 = (int8_t)v;
        return 0;
    }
    case RT_GGUF_VALUE_UINT16: {
        uint16_t v = 0;
        if (rt_read_u16(fp, &v, kv->key, err, errlen) != 0) return -1;
        kv->value.u64 = v;
        return 0;
    }
    case RT_GGUF_VALUE_INT16: {
        uint16_t v = 0;
        if (rt_read_u16(fp, &v, kv->key, err, errlen) != 0) return -1;
        kv->value.i64 = (int16_t)v;
        return 0;
    }
    case RT_GGUF_VALUE_UINT32: {
        uint32_t v = 0;
        if (rt_read_u32(fp, &v, kv->key, err, errlen) != 0) return -1;
        kv->value.u64 = v;
        return 0;
    }
    case RT_GGUF_VALUE_INT32: {
        uint32_t v = 0;
        if (rt_read_u32(fp, &v, kv->key, err, errlen) != 0) return -1;
        kv->value.i64 = (int32_t)v;
        return 0;
    }
    case RT_GGUF_VALUE_FLOAT32: {
        uint32_t bits = 0;
        float f = 0.0f;
        if (rt_read_u32(fp, &bits, kv->key, err, errlen) != 0) return -1;
        memcpy(&f, &bits, sizeof(f));
        kv->value.f64 = f;
        return 0;
    }
    case RT_GGUF_VALUE_BOOL: {
        uint8_t v = 0;
        if (rt_read_u8(fp, &v, kv->key, err, errlen) != 0) return -1;
        kv->value.bool_value = v != 0;
        return 0;
    }
    case RT_GGUF_VALUE_STRING:
        kv->value.string = rt_read_gguf_string(fp, kv->key, err, errlen);
        return kv->value.string ? 0 : -1;
    case RT_GGUF_VALUE_ARRAY:
        if (rt_read_u32(fp, &kv->value.array.type, "GGUF array type", err, errlen) != 0) return -1;
        if (rt_read_u64(fp, &kv->value.array.len, "GGUF array length", err, errlen) != 0) return -1;
        if (rt_file_offset(fp, &kv->value.array.data_offset, err, errlen) != 0) return -1;
        for (uint64_t i = 0; i < kv->value.array.len; i++) {
            if (rt_skip_gguf_value(fp, kv->value.array.type, 1, err, errlen) != 0) return -1;
        }
        return 0;
    case RT_GGUF_VALUE_UINT64:
        return rt_read_u64(fp, &kv->value.u64, kv->key, err, errlen);
    case RT_GGUF_VALUE_INT64: {
        uint64_t v = 0;
        if (rt_read_u64(fp, &v, kv->key, err, errlen) != 0) return -1;
        kv->value.i64 = (int64_t)v;
        return 0;
    }
    case RT_GGUF_VALUE_FLOAT64: {
        uint64_t bits = 0;
        if (rt_read_u64(fp, &bits, kv->key, err, errlen) != 0) return -1;
        memcpy(&kv->value.f64, &bits, sizeof(kv->value.f64));
        return 0;
    }
    default:
        rt_set_err(err, errlen, "unknown GGUF metadata type %u", kv->type);
        return -1;
    }
}

static int rt_gguf_metadata_read_fp(FILE *fp, rt_gguf_metadata *out, char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    out->alignment = RT_GGUF_DEFAULT_ALIGNMENT;

    uint32_t magic = 0;
    if (rt_read_u32(fp, &magic, "GGUF magic", err, errlen) != 0) goto fail;
    if (magic != RT_GGUF_MAGIC) {
        rt_set_err(err, errlen, "file is not GGUF");
        goto fail;
    }
    if (rt_read_u32(fp, &out->version, "GGUF version", err, errlen) != 0) goto fail;
    if (rt_read_u64(fp, &out->n_tensors, "GGUF tensor count", err, errlen) != 0) goto fail;
    if (rt_read_u64(fp, &out->n_kv, "GGUF metadata count", err, errlen) != 0) goto fail;
    if (out->n_kv > SIZE_MAX / sizeof(out->kv[0])) {
        rt_set_err(err, errlen, "GGUF metadata table is too large");
        goto fail;
    }

    out->kv = calloc((size_t)out->n_kv, sizeof(out->kv[0]));
    if (!out->kv && out->n_kv != 0) {
        rt_set_err(err, errlen, "out of memory while reading GGUF metadata");
        goto fail;
    }

    for (uint64_t i = 0; i < out->n_kv; i++) {
        rt_gguf_kv *kv = &out->kv[i];
        kv->key = rt_read_gguf_string(fp, "GGUF metadata key", err, errlen);
        if (!kv->key) goto fail;
        if (rt_read_u32(fp, &kv->type, "GGUF metadata type", err, errlen) != 0) goto fail;
        if (rt_read_gguf_value(fp, kv, err, errlen) != 0) goto fail;
        if (!strcmp(kv->key, "general.alignment") &&
            kv->type == RT_GGUF_VALUE_UINT32 &&
            kv->value.u64 != 0) {
            out->alignment = kv->value.u64;
        }
    }

    return 0;

fail:
    rt_gguf_metadata_free(out);
    return -1;
}

int rt_gguf_metadata_read(const char *path, rt_gguf_metadata *out, char *err, size_t errlen) {
    if (err && errlen > 0) err[0] = '\0';
    if (!path || !out) {
        rt_set_err(err, errlen, "invalid GGUF metadata read arguments");
        return -1;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        rt_set_err(err, errlen, "cannot open GGUF '%s': %s", path, strerror(errno));
        return -1;
    }

    int rc = rt_gguf_metadata_read_fp(fp, out, err, errlen);
    fclose(fp);
    return rc;
}

void rt_gguf_metadata_free(rt_gguf_metadata *meta) {
    if (!meta) return;
    if (meta->kv) {
        for (uint64_t i = 0; i < meta->n_kv; i++) {
            free(meta->kv[i].key);
            if (meta->kv[i].type == RT_GGUF_VALUE_STRING) {
                free(meta->kv[i].value.string);
            }
        }
    }
    free(meta->kv);
    memset(meta, 0, sizeof(*meta));
}

const rt_gguf_kv *rt_gguf_metadata_find(const rt_gguf_metadata *meta, const char *key) {
    if (!meta || !key) return NULL;
    for (uint64_t i = 0; i < meta->n_kv; i++) {
        if (meta->kv[i].key && !strcmp(meta->kv[i].key, key)) return &meta->kv[i];
    }
    return NULL;
}

bool rt_gguf_metadata_get_string(const rt_gguf_metadata *meta, const char *key, const char **out) {
    const rt_gguf_kv *kv = rt_gguf_metadata_find(meta, key);
    if (!kv || kv->type != RT_GGUF_VALUE_STRING || !kv->value.string) return false;
    if (out) *out = kv->value.string;
    return true;
}

bool rt_gguf_metadata_get_u32(const rt_gguf_metadata *meta, const char *key, uint32_t *out) {
    const rt_gguf_kv *kv = rt_gguf_metadata_find(meta, key);
    if (!kv) return false;
    uint64_t v = 0;
    if (kv->type == RT_GGUF_VALUE_UINT32 || kv->type == RT_GGUF_VALUE_UINT16 || kv->type == RT_GGUF_VALUE_UINT8) {
        v = kv->value.u64;
    } else if (kv->type == RT_GGUF_VALUE_UINT64) {
        v = kv->value.u64;
    } else {
        return false;
    }
    if (v > UINT32_MAX) return false;
    if (out) *out = (uint32_t)v;
    return true;
}

bool rt_gguf_metadata_get_u64(const rt_gguf_metadata *meta, const char *key, uint64_t *out) {
    const rt_gguf_kv *kv = rt_gguf_metadata_find(meta, key);
    if (!kv) return false;
    if (kv->type != RT_GGUF_VALUE_UINT64 &&
        kv->type != RT_GGUF_VALUE_UINT32 &&
        kv->type != RT_GGUF_VALUE_UINT16 &&
        kv->type != RT_GGUF_VALUE_UINT8) {
        return false;
    }
    if (out) *out = kv->value.u64;
    return true;
}

bool rt_gguf_metadata_get_bool(const rt_gguf_metadata *meta, const char *key, bool *out) {
    const rt_gguf_kv *kv = rt_gguf_metadata_find(meta, key);
    if (!kv || kv->type != RT_GGUF_VALUE_BOOL) return false;
    if (out) *out = kv->value.bool_value;
    return true;
}

bool rt_gguf_metadata_get_array(const rt_gguf_metadata *meta, const char *key, rt_gguf_array_ref *out) {
    const rt_gguf_kv *kv = rt_gguf_metadata_find(meta, key);
    if (!kv || kv->type != RT_GGUF_VALUE_ARRAY) return false;
    if (out) *out = kv->value.array;
    return true;
}

static const rt_gguf_tensor_type_info *rt_gguf_tensor_type_info_for(uint32_t type) {
    uint32_t n = sizeof(rt_gguf_tensor_types) / sizeof(rt_gguf_tensor_types[0]);
    if (type >= n || !rt_gguf_tensor_types[type].name) return NULL;
    return &rt_gguf_tensor_types[type];
}

const char *rt_gguf_tensor_type_name(uint32_t type) {
    const rt_gguf_tensor_type_info *info = rt_gguf_tensor_type_info_for(type);
    return info ? info->name : "unknown";
}

bool rt_gguf_tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const rt_gguf_tensor_type_info *info = rt_gguf_tensor_type_info_for(type);
    if (!info || info->block_elems == 0) return false;
    uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    if (bytes) *bytes = blocks * info->block_bytes;
    return true;
}

static int rt_gguf_file_parse_tensors(FILE *fp, rt_gguf_file *file, char *err, size_t errlen) {
    if (file->meta.n_tensors > SIZE_MAX / sizeof(file->tensors[0])) {
        rt_set_err(err, errlen, "GGUF tensor table is too large");
        return -1;
    }

    file->tensors = calloc((size_t)file->meta.n_tensors, sizeof(file->tensors[0]));
    if (!file->tensors && file->meta.n_tensors != 0) {
        rt_set_err(err, errlen, "out of memory while reading GGUF tensor table");
        return -1;
    }

    for (uint64_t i = 0; i < file->meta.n_tensors; i++) {
        rt_gguf_tensor *t = &file->tensors[i];
        t->name = rt_read_gguf_string(fp, "GGUF tensor name", err, errlen);
        if (!t->name) return -1;
        if (rt_read_u32(fp, &t->ndim, "GGUF tensor rank", err, errlen) != 0) return -1;
        if (t->ndim == 0 || t->ndim > RT_GGUF_MAX_DIMS) {
            rt_set_err(err, errlen, "GGUF tensor '%s' has unsupported rank %u", t->name, t->ndim);
            return -1;
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (rt_read_u64(fp, &t->dim[d], "GGUF tensor dimension", err, errlen) != 0) return -1;
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                rt_set_err(err, errlen, "GGUF tensor '%s' element count overflow", t->name);
                return -1;
            }
            t->elements *= t->dim[d];
        }
        if (rt_read_u32(fp, &t->type, "GGUF tensor type", err, errlen) != 0) return -1;
        if (rt_read_u64(fp, &t->rel_offset, "GGUF tensor offset", err, errlen) != 0) return -1;
        if (!rt_gguf_tensor_nbytes(t->type, t->elements, &t->bytes)) {
            rt_set_err(err, errlen, "GGUF tensor '%s' has unsupported type %u", t->name, t->type);
            return -1;
        }
    }

    uint64_t pos = 0;
    if (rt_file_offset(fp, &pos, err, errlen) != 0) return -1;
    file->tensor_data_offset = rt_align_up(pos, file->meta.alignment);

    for (uint64_t i = 0; i < file->meta.n_tensors; i++) {
        rt_gguf_tensor *t = &file->tensors[i];
        if (t->rel_offset > UINT64_MAX - file->tensor_data_offset) {
            rt_set_err(err, errlen, "GGUF tensor '%s' offset overflow", t->name);
            return -1;
        }
        t->data_offset = file->tensor_data_offset + t->rel_offset;
        if (t->bytes != 0 &&
            (t->data_offset > file->size || t->bytes > file->size - t->data_offset)) {
            rt_set_err(err, errlen, "GGUF tensor '%s' points outside file", t->name);
            return -1;
        }
    }
    return 0;
}

static void rt_gguf_tensors_free(rt_gguf_file *file) {
    if (!file || !file->tensors) return;
    for (uint64_t i = 0; i < file->meta.n_tensors; i++) {
        free(file->tensors[i].name);
    }
    free(file->tensors);
    file->tensors = NULL;
}

int rt_gguf_file_open(rt_gguf_file **out, const char *path, char *err, size_t errlen) {
    if (err && errlen > 0) err[0] = '\0';
    if (!out || !path) {
        rt_set_err(err, errlen, "invalid GGUF file open arguments");
        return -1;
    }
    *out = NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        rt_set_err(err, errlen, "cannot open GGUF '%s': %s", path, strerror(errno));
        return -1;
    }

    rt_gguf_file *file = calloc(1, sizeof(*file));
    if (!file) {
        fclose(fp);
        rt_set_err(err, errlen, "out of memory while opening GGUF");
        return -1;
    }
    file->fd = -1;
    file->path = rt_strdup(path);
    if (!file->path) {
        fclose(fp);
        rt_gguf_file_close(file);
        rt_set_err(err, errlen, "out of memory while storing GGUF path");
        return -1;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fclose(fp);
        rt_gguf_file_close(file);
        rt_set_err(err, errlen, "cannot open GGUF '%s': %s", path, strerror(errno));
        return -1;
    }
    file->fd = fd;

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        fclose(fp);
        rt_gguf_file_close(file);
        rt_set_err(err, errlen, "cannot stat GGUF '%s': %s", path, strerror(errno));
        return -1;
    }
    file->size = (uint64_t)st.st_size;

    if (rt_gguf_metadata_read_fp(fp, &file->meta, err, errlen) != 0 ||
        rt_gguf_file_parse_tensors(fp, file, err, errlen) != 0) {
        fclose(fp);
        rt_gguf_file_close(file);
        return -1;
    }
    fclose(fp);

    if (file->size != 0) {
        void *map = mmap(NULL, (size_t)file->size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map == MAP_FAILED) {
            rt_gguf_file_close(file);
            rt_set_err(err, errlen, "cannot mmap GGUF '%s': %s", path, strerror(errno));
            return -1;
        }
        file->map = map;
    }

    *out = file;
    return 0;
}

void rt_gguf_file_close(rt_gguf_file *file) {
    if (!file) return;
    rt_gguf_tensors_free(file);
    rt_gguf_metadata_free(&file->meta);
    if (file->map) munmap((void *)file->map, (size_t)file->size);
    if (file->fd >= 0) close(file->fd);
    free(file->path);
    free(file);
}

const char *rt_gguf_file_path(const rt_gguf_file *file) {
    return file ? file->path : NULL;
}

uint64_t rt_gguf_file_size(const rt_gguf_file *file) {
    return file ? file->size : 0;
}

const rt_gguf_metadata *rt_gguf_file_metadata(const rt_gguf_file *file) {
    return file ? &file->meta : NULL;
}

uint64_t rt_gguf_file_tensor_count(const rt_gguf_file *file) {
    return file ? file->meta.n_tensors : 0;
}

const rt_gguf_tensor *rt_gguf_file_tensor(const rt_gguf_file *file, uint64_t index) {
    if (!file || index >= file->meta.n_tensors) return NULL;
    return &file->tensors[index];
}

const rt_gguf_tensor *rt_gguf_file_find_tensor(const rt_gguf_file *file, const char *name) {
    if (!file || !name) return NULL;
    for (uint64_t i = 0; i < file->meta.n_tensors; i++) {
        if (file->tensors[i].name && !strcmp(file->tensors[i].name, name)) {
            return &file->tensors[i];
        }
    }
    return NULL;
}

const void *rt_gguf_file_tensor_data(const rt_gguf_file *file, const rt_gguf_tensor *tensor) {
    if (!file || !file->map || !tensor) return NULL;
    if (tensor->data_offset > file->size || tensor->bytes > file->size - tensor->data_offset) return NULL;
    return file->map + tensor->data_offset;
}

static uint16_t rt_read_le16_mem(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t rt_read_le32_mem(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static float rt_f32_from_bits(uint32_t bits) {
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

static float rt_f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    int exp = (int)(((uint32_t)h >> 10) & 0x1fu);
    uint32_t frac = (uint32_t)h & 0x03ffu;

    if (exp == 0) {
        if (frac == 0) return rt_f32_from_bits(sign);
        while ((frac & 0x0400u) == 0) {
            frac <<= 1;
            exp--;
        }
        exp++;
        frac &= 0x03ffu;
    } else if (exp == 31) {
        return rt_f32_from_bits(sign | 0x7f800000u | (frac << 13));
    }

    exp = exp + (127 - 15);
    return rt_f32_from_bits(sign | ((uint32_t)exp << 23) | (frac << 13));
}

static const int8_t rt_iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

static const uint64_t rt_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

const uint32_t rt_iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

const uint64_t rt_iq2xs_grid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

const uint64_t rt_iq2s_grid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};


const uint32_t rt_iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

const uint64_t rt_iq1s_grid[2048] = {
    0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff,
    0xffffffffffff0101, 0xffffffffff00ff00, 0xffffffffff000000, 0xffffffffff01ffff,
    0xffffffffff01ff01, 0xffffffffff0101ff, 0xffffffffff010101, 0xffffffff00ff0000,
    0xffffffff0000ff00, 0xffffffff000000ff, 0xffffffff00000001, 0xffffffff00010000,
    0xffffffff01ffffff, 0xffffffff01ffff01, 0xffffffff01ff01ff, 0xffffffff01ff0101,
    0xffffffff01000000, 0xffffffff0101ffff, 0xffffffff0101ff01, 0xffffffff010101ff,
    0xffffffff01010101, 0xffffff00ffff00ff, 0xffffff00ffff0000, 0xffffff00ff00ff00,
    0xffffff00ff0000ff, 0xffffff00ff000001, 0xffffff00ff000100, 0xffffff00ff000101,
    0xffffff00ff010000, 0xffffff0000ffff00, 0xffffff0000ff0001, 0xffffff0000ff0100,
    0xffffff000000ff01, 0xffffff0000000000, 0xffffff0000000101, 0xffffff000001ff00,
    0xffffff00000100ff, 0xffffff0000010001, 0xffffff00000101ff, 0xffffff0001ff0000,
    0xffffff000100ff00, 0xffffff00010000ff, 0xffffff0001000001, 0xffffff0001010000,
    0xffffff01ffffffff, 0xffffff01ffffff01, 0xffffff01ffff01ff, 0xffffff01ffff0101,
    0xffffff01ff000000, 0xffffff01ff01ffff, 0xffffff01ff01ff01, 0xffffff01ff0101ff,
    0xffffff01ff010101, 0xffffff0100ff0000, 0xffffff010000ff00, 0xffffff0100000100,
    0xffffff01000100ff, 0xffffff0100010100, 0xffffff0101ffffff, 0xffffff0101ffff01,
    0xffffff0101ff01ff, 0xffffff0101ff0101, 0xffffff010100ff00, 0xffffff0101000000,
    0xffffff0101000100, 0xffffff010101ffff, 0xffffff010101ff01, 0xffffff01010101ff,
    0xffffff0101010101, 0xffff00ffff00ff00, 0xffff00ffff0000ff, 0xffff00ffff000001,
    0xffff00ffff010000, 0xffff00ff00ffff00, 0xffff00ff00ff0100, 0xffff00ff00000000,
    0xffff00ff00000101, 0xffff00ff000100ff, 0xffff00ff00010000, 0xffff00ff0100ff00,
    0xffff00ff01000100, 0xffff00ff01010000, 0xffff0000ffffff00, 0xffff0000ffff00ff,
    0xffff0000ffff0000, 0xffff0000ffff0001, 0xffff0000ff000000, 0xffff0000ff0001ff,
    0xffff0000ff000101, 0xffff0000ff010100, 0xffff000000ffffff, 0xffff000000ff0000,
    0xffff000000ff0101, 0xffff00000000ffff, 0xffff00000000ff00, 0xffff0000000000ff,
    0xffff000000000000, 0xffff000000000001, 0xffff000000000100, 0xffff00000001ffff,
    0xffff00000001ff01, 0xffff000000010000, 0xffff0000000101ff, 0xffff000000010101,
    0xffff000001ffff00, 0xffff00000100ff00, 0xffff000001000000, 0xffff0000010001ff,
    0xffff000001000101, 0xffff00000101ff00, 0xffff0000010100ff, 0xffff000001010000,
    0xffff000001010001, 0xffff000001010100, 0xffff0001ff0000ff, 0xffff0001ff000100,
    0xffff000100ffff00, 0xffff000100ff00ff, 0xffff00010000ffff, 0xffff00010000ff01,
    0xffff000100000000, 0xffff0001000001ff, 0xffff00010001ffff, 0xffff00010001ff00,
    0xffff000100010001, 0xffff000100010100, 0xffff000101ff0000, 0xffff00010100ff00,
    0xffff0001010000ff, 0xffff000101000100, 0xffff01ffffffffff, 0xffff01ffffffff01,
    0xffff01ffffff01ff, 0xffff01ffffff0101, 0xffff01ffff000000, 0xffff01ffff01ffff,
    0xffff01ffff01ff01, 0xffff01ffff0101ff, 0xffff01ffff010101, 0xffff01ff00ff0000,
    0xffff01ff0000ff00, 0xffff01ff00000001, 0xffff01ff00010000, 0xffff01ff01ffffff,
    0xffff01ff01ffff01, 0xffff01ff01ff01ff, 0xffff01ff01ff0101, 0xffff01ff01000000,
    0xffff01ff0101ffff, 0xffff01ff0101ff01, 0xffff01ff010101ff, 0xffff01ff01010101,
    0xffff0100ffff0000, 0xffff0100ff00ff00, 0xffff0100ff0000ff, 0xffff0100ff000100,
    0xffff0100ff0100ff, 0xffff0100ff010000, 0xffff010000ffff00, 0xffff01000000ffff,
    0xffff01000000ff00, 0xffff010000000000, 0xffff01000001ff00, 0xffff0100000100ff,
    0xffff010000010100, 0xffff01000100ff00, 0xffff0100010000ff, 0xffff010001000001,
    0xffff010001000100, 0xffff010001010000, 0xffff0101ffffffff, 0xffff0101ffffff01,
    0xffff0101ffff01ff, 0xffff0101ffff0101, 0xffff0101ff000000, 0xffff0101ff01ffff,
    0xffff0101ff01ff01, 0xffff0101ff0101ff, 0xffff0101ff010101, 0xffff010100ff0000,
    0xffff01010000ff00, 0xffff010100000100, 0xffff01010001ff00, 0xffff010100010000,
    0xffff010101ffffff, 0xffff010101ffff01, 0xffff010101ff0000, 0xffff010101ff01ff,
    0xffff010101ff0101, 0xffff010101000000, 0xffff01010101ffff, 0xffff01010101ff01,
    0xffff0101010101ff, 0xffff010101010101, 0xff00ffffff00ffff, 0xff00ffffff00ff00,
    0xff00ffffff0000ff, 0xff00ffffff000100, 0xff00ffffff0100ff, 0xff00ffffff010000,
    0xff00ffff00ffff00, 0xff00ffff00ff00ff, 0xff00ffff0000ffff, 0xff00ffff00000000,
    0xff00ffff000001ff, 0xff00ffff0001ff00, 0xff00ffff000100ff, 0xff00ffff00010000,
    0xff00ffff00010100, 0xff00ffff0100ff00, 0xff00ffff010000ff, 0xff00ffff01000001,
    0xff00ffff0101ff00, 0xff00ffff01010000, 0xff00ff00ffffff00, 0xff00ff00ffff00ff,
    0xff00ff00ffff0001, 0xff00ff00ffff0100, 0xff00ff00ff00ffff, 0xff00ff00ff00ff01,
    0xff00ff00ff000000, 0xff00ff00ff0001ff, 0xff00ff00ff01ff00, 0xff00ff00ff0100ff,
    0xff00ff00ff010100, 0xff00ff0000ff0000, 0xff00ff0000ff0101, 0xff00ff000000ffff,
    0xff00ff000000ff00, 0xff00ff000000ff01, 0xff00ff00000000ff, 0xff00ff0000000000,
    0xff00ff0000000001, 0xff00ff0000000100, 0xff00ff000001ffff, 0xff00ff0000010000,
    0xff00ff0001ff00ff, 0xff00ff000100ff01, 0xff00ff0001000000, 0xff00ff000101ff00,
    0xff00ff00010100ff, 0xff00ff01ff00ff00, 0xff00ff01ff0000ff, 0xff00ff01ff000001,
    0xff00ff01ff010000, 0xff00ff0100ffffff, 0xff00ff0100ff0001, 0xff00ff0100ff0100,
    0xff00ff010000ff01, 0xff00ff0100000000, 0xff00ff01000001ff, 0xff00ff0100000101,
    0xff00ff01000100ff, 0xff00ff0100010001, 0xff00ff0101ff0000, 0xff00ff010100ff00,
    0xff00ff01010000ff, 0xff00ff0101000001, 0xff00ff0101010000, 0xff0000ffffffff00,
    0xff0000ffffff0001, 0xff0000ffffff0100, 0xff0000ffff0000ff, 0xff0000ffff000000,
    0xff0000ffff0001ff, 0xff0000ffff000100, 0xff0000ffff01ff00, 0xff0000ffff010001,
    0xff0000ff00ffff00, 0xff0000ff00ff0000, 0xff0000ff00ff0001, 0xff0000ff00ff01ff,
    0xff0000ff00ff0101, 0xff0000ff0000ff00, 0xff0000ff000000ff, 0xff0000ff00000000,
    0xff0000ff00000001, 0xff0000ff00000100, 0xff0000ff0001ff01, 0xff0000ff00010000,
    0xff0000ff000101ff, 0xff0000ff01ff00ff, 0xff0000ff01ff0100, 0xff0000ff0100ffff,
    0xff0000ff010000ff, 0xff0000ff01000000, 0xff0000ff010001ff, 0xff0000ff01000100,
    0xff0000ff01000101, 0xff0000ff0101ff00, 0xff0000ff010100ff, 0xff0000ff01010000,
    0xff0000ff01010100, 0xff000000ffffff01, 0xff000000ffff0000, 0xff000000ffff0101,
    0xff000000ff00ff00, 0xff000000ff0000ff, 0xff000000ff000000, 0xff000000ff000001,
    0xff000000ff000100, 0xff000000ff01ffff, 0xff000000ff01ff01, 0xff000000ff010000,
    0xff000000ff0101ff, 0xff000000ff010101, 0xff00000000ffff00, 0xff00000000ff00ff,
    0xff00000000ff0000, 0xff00000000ff0001, 0xff0000000000ff00, 0xff0000000000ff01,
    0xff000000000000ff, 0xff00000000000000, 0xff00000000000001, 0xff00000000000100,
    0xff00000000000101, 0xff0000000001ff00, 0xff000000000100ff, 0xff00000000010000,
    0xff00000000010001, 0xff00000000010100, 0xff00000001ffffff, 0xff00000001ffff01,
    0xff00000001ff00ff, 0xff00000001ff0000, 0xff00000001ff01ff, 0xff00000001ff0101,
    0xff0000000100ffff, 0xff0000000100ff00, 0xff000000010000ff, 0xff00000001000000,
    0xff00000001000001, 0xff00000001000100, 0xff00000001000101, 0xff0000000101ffff,
    0xff0000000101ff01, 0xff00000001010000, 0xff000001ffffff00, 0xff000001ffff00ff,
    0xff000001ffff0000, 0xff000001ffff0001, 0xff000001ff000000, 0xff000001ff000001,
    0xff000001ff0001ff, 0xff000001ff000101, 0xff000001ff01ff00, 0xff000001ff010001,
    0xff00000100ffffff, 0xff00000100ffff01, 0xff00000100ff00ff, 0xff00000100ff0000,
    0xff00000100ff01ff, 0xff00000100ff0101, 0xff0000010000ff00, 0xff00000100000000,
    0xff00000100000001, 0xff000001000001ff, 0xff00000100000100, 0xff0000010001ff00,
    0xff000001000100ff, 0xff00000100010000, 0xff000001000101ff, 0xff00000100010100,
    0xff00000100010101, 0xff00000101ff0001, 0xff00000101ff0101, 0xff0000010100ff01,
    0xff00000101000000, 0xff000001010100ff, 0xff00000101010100, 0xff0001ffff00ff00,
    0xff0001ffff000001, 0xff0001ffff010000, 0xff0001ff00ffff00, 0xff0001ff00ff00ff,
    0xff0001ff00ff0001, 0xff0001ff00ff0100, 0xff0001ff0000ffff, 0xff0001ff00000000,
    0xff0001ff000001ff, 0xff0001ff00000101, 0xff0001ff0001ffff, 0xff0001ff0001ff00,
    0xff0001ff000100ff, 0xff0001ff00010001, 0xff0001ff00010100, 0xff0001ff01ff0000,
    0xff0001ff0100ff00, 0xff0001ff010000ff, 0xff0001ff01010000, 0xff000100ff00ffff,
    0xff000100ff00ff01, 0xff000100ff000000, 0xff000100ff000101, 0xff000100ff01ff00,
    0xff000100ff010000, 0xff00010000ffff01, 0xff00010000ff00ff, 0xff00010000ff0000,
    0xff00010000ff01ff, 0xff0001000000ff00, 0xff000100000000ff, 0xff00010000000000,
    0xff00010000000001, 0xff00010000000100, 0xff00010000000101, 0xff0001000001ffff,
    0xff00010000010000, 0xff00010000010101, 0xff00010001ff0100, 0xff0001000100ff00,
    0xff0001000100ff01, 0xff00010001000000, 0xff000100010001ff, 0xff0001000101ff00,
    0xff00010001010001, 0xff00010001010100, 0xff000101ffff0100, 0xff000101ff000001,
    0xff000101ff0100ff, 0xff000101ff010001, 0xff00010100ff00ff, 0xff00010100ff0001,
    0xff00010100ff0100, 0xff0001010000ffff, 0xff0001010000ff01, 0xff00010100000000,
    0xff000101000001ff, 0xff0001010001ff00, 0xff00010100010001, 0xff00010100010100,
    0xff00010101ff0000, 0xff0001010100ff00, 0xff00010101000001, 0xff00010101000101,
    0xff01ffffffffffff, 0xff01ffffffffff01, 0xff01ffffffff01ff, 0xff01ffffffff0101,
    0xff01ffffff000000, 0xff01ffffff01ffff, 0xff01ffffff01ff01, 0xff01ffffff010000,
    0xff01ffffff0101ff, 0xff01ffffff010101, 0xff01ffff00ff0000, 0xff01ffff0000ff00,
    0xff01ffff00000100, 0xff01ffff0001ff00, 0xff01ffff00010000, 0xff01ffff01ffffff,
    0xff01ffff01ffff01, 0xff01ffff01ff01ff, 0xff01ffff01ff0101, 0xff01ffff01000000,
    0xff01ffff0101ffff, 0xff01ffff0101ff01, 0xff01ffff01010000, 0xff01ffff010101ff,
    0xff01ffff01010101, 0xff01ff00ffff0000, 0xff01ff00ff00ff00, 0xff01ff00ff0000ff,
    0xff01ff00ff000100, 0xff01ff00ff010000, 0xff01ff0000ffff01, 0xff01ff0000ff00ff,
    0xff01ff0000ff0100, 0xff01ff0000000000, 0xff01ff00000001ff, 0xff01ff0000000101,
    0xff01ff000001ff00, 0xff01ff00000100ff, 0xff01ff0000010000, 0xff01ff0000010001,
    0xff01ff0001ff0000, 0xff01ff000100ffff, 0xff01ff0001000001, 0xff01ff0001000100,
    0xff01ff0001010000, 0xff01ff01ffffff00, 0xff01ff01ffff01ff, 0xff01ff01ffff0101,
    0xff01ff01ff00ff00, 0xff01ff01ff000000, 0xff01ff01ff01ffff, 0xff01ff01ff01ff01,
    0xff01ff01ff0101ff, 0xff01ff01ff010101, 0xff01ff0100ff0000, 0xff01ff010000ff00,
    0xff01ff0100000001, 0xff01ff0100000100, 0xff01ff0100010000, 0xff01ff0101ffff00,
    0xff01ff0101ff01ff, 0xff01ff0101ff0101, 0xff01ff010100ff00, 0xff01ff0101000000,
    0xff01ff010101ffff, 0xff01ff010101ff01, 0xff01ff01010101ff, 0xff01ff0101010101,
    0xff0100ffffff0000, 0xff0100ffff0000ff, 0xff0100ffff000001, 0xff0100ffff000100,
    0xff0100ffff010000, 0xff0100ff00ff00ff, 0xff0100ff00ff0000, 0xff0100ff00ff0001,
    0xff0100ff00ff0100, 0xff0100ff0000ff01, 0xff0100ff00000000, 0xff0100ff000001ff,
    0xff0100ff00000101, 0xff0100ff00010001, 0xff0100ff01ff0000, 0xff0100ff0100ff00,
    0xff0100ff010000ff, 0xff0100ff01000100, 0xff0100ff0101ff00, 0xff0100ff01010000,
    0xff010000ffff0100, 0xff010000ff000000, 0xff010000ff01ff00, 0xff010000ff010100,
    0xff01000000ffffff, 0xff01000000ff0000, 0xff01000000ff01ff, 0xff0100000000ff00,
    0xff010000000000ff, 0xff01000000000000, 0xff01000000000100, 0xff0100000001ff01,
    0xff01000000010000, 0xff010000000101ff, 0xff01000001ff0100, 0xff0100000100ffff,
    0xff010000010000ff, 0xff01000001000000, 0xff010000010001ff, 0xff01000001000101,
    0xff0100000101ff00, 0xff010000010100ff, 0xff01000001010001, 0xff01000001010100,
    0xff010001ffff0000, 0xff010001ff00ffff, 0xff010001ff00ff01, 0xff010001ff000100,
    0xff010001ff010000, 0xff01000100ffff00, 0xff01000100ff0100, 0xff01000100000000,
    0xff0100010001ffff, 0xff0100010001ff00, 0xff01000100010100, 0xff01000101ff00ff,
    0xff01000101ff0001, 0xff0100010100ffff, 0xff01000101000101, 0xff0101ffffffffff,
    0xff0101ffffffff01, 0xff0101ffffff01ff, 0xff0101ffffff0101, 0xff0101ffff000000,
    0xff0101ffff01ffff, 0xff0101ffff01ff01, 0xff0101ffff0101ff, 0xff0101ffff010101,
    0xff0101ff00ff0000, 0xff0101ff0000ff00, 0xff0101ff000000ff, 0xff0101ff00010000,
    0xff0101ff01ffffff, 0xff0101ff01ffff01, 0xff0101ff01ff01ff, 0xff0101ff01ff0101,
    0xff0101ff0101ffff, 0xff0101ff0101ff01, 0xff0101ff010101ff, 0xff0101ff01010101,
    0xff010100ffff0100, 0xff010100ff00ff00, 0xff010100ff0000ff, 0xff010100ff000100,
    0xff010100ff010000, 0xff01010000ff0001, 0xff01010000ff0100, 0xff0101000000ff01,
    0xff01010000000000, 0xff0101000001ff00, 0xff010100000100ff, 0xff01010000010001,
    0xff01010000010100, 0xff01010001ff0000, 0xff0101000100ffff, 0xff01010001000001,
    0xff01010001000100, 0xff010100010100ff, 0xff01010001010000, 0xff010101ffffffff,
    0xff010101ffffff01, 0xff010101ffff01ff, 0xff010101ffff0101, 0xff010101ff01ffff,
    0xff010101ff01ff01, 0xff010101ff0101ff, 0xff010101ff010101, 0xff01010100ff0000,
    0xff0101010000ff00, 0xff01010100000001, 0xff01010100000100, 0xff01010100010000,
    0xff01010101ffffff, 0xff01010101ffff01, 0xff01010101ff01ff, 0xff01010101ff0101,
    0xff01010101000000, 0xff0101010101ffff, 0xff0101010101ff01, 0xff010101010101ff,
    0xff01010101010101, 0x00ffffffffff0000, 0x00ffffffff00ff00, 0x00ffffffff000001,
    0x00ffffffff010000, 0x00ffffff00ff0100, 0x00ffffff0000ff01, 0x00ffffff00000000,
    0x00ffffff000001ff, 0x00ffffff00000101, 0x00ffffff0001ff00, 0x00ffffff000100ff,
    0x00ffffff00010001, 0x00ffffff010000ff, 0x00ffffff01000100, 0x00ffffff0101ff00,
    0x00ffffff01010001, 0x00ffff00ffffffff, 0x00ffff00ffffff00, 0x00ffff00ffff00ff,
    0x00ffff00ffff0001, 0x00ffff00ffff0100, 0x00ffff00ff00ff01, 0x00ffff00ff000000,
    0x00ffff00ff000001, 0x00ffff00ff0001ff, 0x00ffff00ff000101, 0x00ffff00ff01ff00,
    0x00ffff00ff010001, 0x00ffff00ff010100, 0x00ffff0000ff0000, 0x00ffff0000ff01ff,
    0x00ffff0000ff0101, 0x00ffff000000ff00, 0x00ffff00000000ff, 0x00ffff0000000000,
    0x00ffff0000000001, 0x00ffff0000000100, 0x00ffff0000000101, 0x00ffff0000010000,
    0x00ffff00000101ff, 0x00ffff0000010101, 0x00ffff0001ffff00, 0x00ffff0001ff00ff,
    0x00ffff0001ff0001, 0x00ffff000100ffff, 0x00ffff000100ff01, 0x00ffff0001000000,
    0x00ffff000101ffff, 0x00ffff000101ff00, 0x00ffff000101ff01, 0x00ffff01ffff0000,
    0x00ffff01ff00ff00, 0x00ffff01ff0000ff, 0x00ffff01ff000001, 0x00ffff01ff010000,
    0x00ffff0100ffff00, 0x00ffff010000ff01, 0x00ffff0100000000, 0x00ffff0100000101,
    0x00ffff01000100ff, 0x00ffff0100010100, 0x00ffff0101ff0100, 0x00ffff01010000ff,
    0x00ffff0101010000, 0x00ff00ffffffff00, 0x00ff00ffff000000, 0x00ff00ffff000100,
    0x00ff00ffff010100, 0x00ff00ff00ff0000, 0x00ff00ff00ff01ff, 0x00ff00ff00ff0101,
    0x00ff00ff0000ff00, 0x00ff00ff000000ff, 0x00ff00ff00000000, 0x00ff00ff00000001,
    0x00ff00ff0001ff00, 0x00ff00ff0001ff01, 0x00ff00ff00010000, 0x00ff00ff000101ff,
    0x00ff00ff00010101, 0x00ff00ff01ffff00, 0x00ff00ff01ff0001, 0x00ff00ff01ff0100,
    0x00ff00ff0100ffff, 0x00ff00ff0100ff01, 0x00ff00ff01000000, 0x00ff00ff0101ffff,
    0x00ff00ff0101ff00, 0x00ff00ff01010100, 0x00ff0000ffffff00, 0x00ff0000ffffff01,
    0x00ff0000ffff0000, 0x00ff0000ffff0101, 0x00ff0000ff00ff00, 0x00ff0000ff0000ff,
    0x00ff0000ff000000, 0x00ff0000ff000001, 0x00ff0000ff000100, 0x00ff0000ff01ffff,
    0x00ff0000ff010000, 0x00ff0000ff010101, 0x00ff000000ffff00, 0x00ff000000ff00ff,
    0x00ff000000ff0000, 0x00ff000000ff0001, 0x00ff000000ff0100, 0x00ff00000000ffff,
    0x00ff00000000ff00, 0x00ff0000000000ff, 0x00ff000000000000, 0x00ff000000000001,
    0x00ff0000000001ff, 0x00ff000000000100, 0x00ff00000001ff00, 0x00ff0000000100ff,
    0x00ff000000010000, 0x00ff000000010001, 0x00ff000000010100, 0x00ff000001ffff01,
    0x00ff000001ff00ff, 0x00ff000001ff0000, 0x00ff000001ff01ff, 0x00ff00000100ff00,
    0x00ff0000010000ff, 0x00ff000001000000, 0x00ff000001000001, 0x00ff000001000100,
    0x00ff000001000101, 0x00ff000001010000, 0x00ff0000010101ff, 0x00ff000001010101,
    0x00ff0001ffffff00, 0x00ff0001ffff0000, 0x00ff0001ffff0100, 0x00ff0001ff0000ff,
    0x00ff0001ff000000, 0x00ff0001ff0001ff, 0x00ff0001ff000101, 0x00ff0001ff01ff00,
    0x00ff0001ff0100ff, 0x00ff0001ff010100, 0x00ff000100ffffff, 0x00ff000100ffff01,
    0x00ff000100ff0000, 0x00ff000100ff01ff, 0x00ff00010000ffff, 0x00ff00010000ff00,
    0x00ff00010000ff01, 0x00ff000100000000, 0x00ff000100000001, 0x00ff000100000100,
    0x00ff00010001ff01, 0x00ff000100010000, 0x00ff0001000101ff, 0x00ff000101ffff00,
    0x00ff000101ff0000, 0x00ff000101ff0101, 0x00ff0001010000ff, 0x00ff000101000000,
    0x00ff00010101ff00, 0x00ff0001010100ff, 0x00ff000101010001, 0x00ff01ffffff0000,
    0x00ff01ffff00ff00, 0x00ff01ffff000000, 0x00ff01ffff000101, 0x00ff01ffff010000,
    0x00ff01ff00ffff01, 0x00ff01ff00ff0100, 0x00ff01ff0000ffff, 0x00ff01ff00000000,
    0x00ff01ff000001ff, 0x00ff01ff0001ff00, 0x00ff01ff000100ff, 0x00ff01ff00010001,
    0x00ff01ff00010100, 0x00ff01ff01ff0000, 0x00ff01ff0100ff00, 0x00ff01ff010000ff,
    0x00ff01ff01000001, 0x00ff01ff01000100, 0x00ff01ff01010000, 0x00ff0100ffffff00,
    0x00ff0100ffff0000, 0x00ff0100ffff0001, 0x00ff0100ffff0101, 0x00ff0100ff00ffff,
    0x00ff0100ff0000ff, 0x00ff0100ff000000, 0x00ff0100ff0001ff, 0x00ff0100ff01ff00,
    0x00ff0100ff0100ff, 0x00ff0100ff010001, 0x00ff010000ffffff, 0x00ff010000ff0000,
    0x00ff010000ff0101, 0x00ff01000000ff00, 0x00ff01000000ff01, 0x00ff0100000000ff,
    0x00ff010000000000, 0x00ff010000000001, 0x00ff010000000100, 0x00ff01000001ffff,
    0x00ff01000001ff01, 0x00ff010000010000, 0x00ff010000010001, 0x00ff010000010101,
    0x00ff010001ff0001, 0x00ff010001ff0100, 0x00ff01000100ff01, 0x00ff010001000000,
    0x00ff010001000001, 0x00ff0100010001ff, 0x00ff01000101ff00, 0x00ff0100010100ff,
    0x00ff010001010001, 0x00ff010001010100, 0x00ff0101ff000001, 0x00ff010100ff00ff,
    0x00ff010100ff0001, 0x00ff010100ff0100, 0x00ff010100000000, 0x00ff0101000001ff,
    0x00ff010100000101, 0x00ff0101000100ff, 0x00ff010100010100, 0x00ff0101010000ff,
    0x00ff010101010000, 0x0000ffffffffff00, 0x0000ffffffff00ff, 0x0000ffffffff0000,
    0x0000ffffffff0001, 0x0000ffffffff0100, 0x0000ffffff00ff01, 0x0000ffffff000000,
    0x0000ffffff000101, 0x0000ffffff01ff00, 0x0000ffffff0100ff, 0x0000ffffff010100,
    0x0000ffff00ffffff, 0x0000ffff00ff0000, 0x0000ffff00ff01ff, 0x0000ffff0000ff00,
    0x0000ffff000000ff, 0x0000ffff00000000, 0x0000ffff00000001, 0x0000ffff00000100,
    0x0000ffff00010000, 0x0000ffff000101ff, 0x0000ffff01ff0001, 0x0000ffff01ff0100,
    0x0000ffff01000000, 0x0000ffff010001ff, 0x0000ffff0101ffff, 0x0000ffff0101ff00,
    0x0000ffff01010001, 0x0000ffff01010100, 0x0000ff00ffff0000, 0x0000ff00ffff01ff,
    0x0000ff00ffff0100, 0x0000ff00ffff0101, 0x0000ff00ff00ff00, 0x0000ff00ff0000ff,
    0x0000ff00ff000000, 0x0000ff00ff000001, 0x0000ff00ff0001ff, 0x0000ff00ff000100,
    0x0000ff00ff01ffff, 0x0000ff00ff010000, 0x0000ff00ff010001, 0x0000ff00ff0101ff,
    0x0000ff00ff010101, 0x0000ff0000ffff00, 0x0000ff0000ff00ff, 0x0000ff0000ff0000,
    0x0000ff0000ff0001, 0x0000ff0000ff0100, 0x0000ff000000ffff, 0x0000ff000000ff00,
    0x0000ff000000ff01, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001,
    0x0000ff00000001ff, 0x0000ff0000000100, 0x0000ff0000000101, 0x0000ff000001ff00,
    0x0000ff00000100ff, 0x0000ff0000010000, 0x0000ff0000010001, 0x0000ff0000010100,
    0x0000ff0001ffff01, 0x0000ff0001ff0000, 0x0000ff000100ff00, 0x0000ff00010000ff,
    0x0000ff0001000000, 0x0000ff0001000001, 0x0000ff0001000100, 0x0000ff000101ffff,
    0x0000ff0001010000, 0x0000ff0001010101, 0x0000ff01ffffff00, 0x0000ff01ffff0001,
    0x0000ff01ff00ff01, 0x0000ff01ff000000, 0x0000ff01ff000101, 0x0000ff01ff01ff00,
    0x0000ff01ff0100ff, 0x0000ff0100ffff01, 0x0000ff0100ff0000, 0x0000ff0100ff0101,
    0x0000ff010000ff00, 0x0000ff01000000ff, 0x0000ff0100000000, 0x0000ff0100000001,
    0x0000ff0100000100, 0x0000ff010001ff01, 0x0000ff0100010000, 0x0000ff0101ff0000,
    0x0000ff010100ffff, 0x0000ff010100ff01, 0x0000ff0101000000, 0x0000ff0101000100,
    0x0000ff0101000101, 0x0000ff01010100ff, 0x000000ffffff00ff, 0x000000ffffff0000,
    0x000000ffff00ff00, 0x000000ffff0000ff, 0x000000ffff000000, 0x000000ffff000001,
    0x000000ffff0001ff, 0x000000ffff000100, 0x000000ffff01ff00, 0x000000ffff010000,
    0x000000ffff0101ff, 0x000000ffff010101, 0x000000ff00ffff00, 0x000000ff00ff00ff,
    0x000000ff00ff0000, 0x000000ff00ff0001, 0x000000ff00ff0100, 0x000000ff00ff0101,
    0x000000ff0000ffff, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000,
    0x000000ff00000001, 0x000000ff000001ff, 0x000000ff00000100, 0x000000ff00000101,
    0x000000ff0001ff00, 0x000000ff0001ff01, 0x000000ff000100ff, 0x000000ff00010000,
    0x000000ff00010001, 0x000000ff00010100, 0x000000ff01ffffff, 0x000000ff01ff01ff,
    0x000000ff01ff0101, 0x000000ff0100ff00, 0x000000ff010000ff, 0x000000ff01000000,
    0x000000ff01000001, 0x000000ff01000100, 0x000000ff0101ff00, 0x000000ff010100ff,
    0x000000ff01010000, 0x000000ff01010101, 0x00000000ffffff00, 0x00000000ffffff01,
    0x00000000ffff00ff, 0x00000000ffff0000, 0x00000000ffff0001, 0x00000000ffff0100,
    0x00000000ff00ffff, 0x00000000ff00ff00, 0x00000000ff00ff01, 0x00000000ff0000ff,
    0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff000101,
    0x00000000ff01ff00, 0x00000000ff0100ff, 0x00000000ff010000, 0x00000000ff010001,
    0x00000000ff010100, 0x0000000000ffffff, 0x0000000000ffff00, 0x0000000000ffff01,
    0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001, 0x0000000000ff01ff,
    0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01,
    0x00000000000000ff, 0x0000000000000000, 0x0000000000000001, 0x00000000000001ff,
    0x0000000000000100, 0x0000000000000101, 0x000000000001ffff, 0x000000000001ff00,
    0x00000000000100ff, 0x0000000000010000, 0x0000000000010001, 0x00000000000101ff,
    0x0000000000010100, 0x0000000000010101, 0x0000000001ffff00, 0x0000000001ff00ff,
    0x0000000001ff0000, 0x0000000001ff0100, 0x0000000001ff0101, 0x000000000100ffff,
    0x000000000100ff00, 0x00000000010000ff, 0x0000000001000000, 0x0000000001000001,
    0x00000000010001ff, 0x0000000001000100, 0x000000000101ff00, 0x00000000010100ff,
    0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x00000001ffffffff,
    0x00000001ffffff00, 0x00000001ffffff01, 0x00000001ffff00ff, 0x00000001ffff0001,
    0x00000001ffff01ff, 0x00000001ffff0100, 0x00000001ff00ff00, 0x00000001ff0000ff,
    0x00000001ff000000, 0x00000001ff0001ff, 0x00000001ff000100, 0x00000001ff01ffff,
    0x00000001ff01ff00, 0x00000001ff01ff01, 0x00000001ff0100ff, 0x00000001ff010000,
    0x00000001ff010001, 0x00000001ff0101ff, 0x00000001ff010100, 0x0000000100ffff00,
    0x0000000100ff0000, 0x0000000100ff0001, 0x0000000100ff01ff, 0x0000000100ff0100,
    0x0000000100ff0101, 0x000000010000ffff, 0x000000010000ff00, 0x000000010000ff01,
    0x00000001000000ff, 0x0000000100000000, 0x0000000100000001, 0x00000001000001ff,
    0x0000000100000100, 0x0000000100000101, 0x000000010001ff00, 0x00000001000100ff,
    0x0000000100010000, 0x0000000100010100, 0x0000000101ffff01, 0x0000000101ff0000,
    0x0000000101ff0001, 0x0000000101ff01ff, 0x0000000101ff0100, 0x0000000101ff0101,
    0x000000010100ff00, 0x0000000101000000, 0x0000000101000101, 0x000000010101ff01,
    0x0000000101010000, 0x0000000101010001, 0x00000001010101ff, 0x0000000101010100,
    0x000001ffffff00ff, 0x000001ffffff0000, 0x000001ffffff0001, 0x000001ffffff0100,
    0x000001ffff00ffff, 0x000001ffff000000, 0x000001ffff0001ff, 0x000001ffff01ff00,
    0x000001ffff010101, 0x000001ff00ff0000, 0x000001ff00ff01ff, 0x000001ff00ff0101,
    0x000001ff0000ff00, 0x000001ff000000ff, 0x000001ff00000000, 0x000001ff00000001,
    0x000001ff000001ff, 0x000001ff00000100, 0x000001ff0001ffff, 0x000001ff0001ff01,
    0x000001ff000100ff, 0x000001ff00010000, 0x000001ff01ffff01, 0x000001ff01ff0100,
    0x000001ff0100ffff, 0x000001ff0100ff01, 0x000001ff01000000, 0x000001ff010001ff,
    0x000001ff0101ff00, 0x000001ff01010100, 0x00000100ffffff00, 0x00000100ffffff01,
    0x00000100ffff0000, 0x00000100ffff0101, 0x00000100ff00ff00, 0x00000100ff0000ff,
    0x00000100ff000000, 0x00000100ff000001, 0x00000100ff000100, 0x00000100ff010000,
    0x0000010000ffff00, 0x0000010000ff00ff, 0x0000010000ff0000, 0x0000010000ff0001,
    0x0000010000ff0100, 0x000001000000ffff, 0x000001000000ff00, 0x000001000000ff01,
    0x00000100000000ff, 0x0000010000000000, 0x0000010000000001, 0x00000100000001ff,
    0x0000010000000100, 0x0000010000000101, 0x000001000001ff00, 0x00000100000100ff,
    0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010001ffff00,
    0x0000010001ff0000, 0x0000010001ff0100, 0x000001000100ff00, 0x00000100010000ff,
    0x0000010001000000, 0x0000010001000001, 0x00000100010001ff, 0x0000010001000100,
    0x0000010001010000, 0x00000101ffff00ff, 0x00000101ffff01ff, 0x00000101ff000000,
    0x00000101ff000101, 0x00000101ff01ffff, 0x00000101ff010000, 0x00000101ff010001,
    0x00000101ff010100, 0x0000010100ff0000, 0x0000010100ff01ff, 0x0000010100ff0100,
    0x000001010000ff00, 0x0000010100000000, 0x0000010100000001, 0x00000101000001ff,
    0x0000010100000100, 0x000001010001ff01, 0x0000010100010000, 0x00000101000101ff,
    0x0000010100010101, 0x0000010101ffff00, 0x0000010101ff0101, 0x000001010100ff01,
    0x0000010101000000, 0x0000010101000001, 0x00000101010001ff, 0x0000010101000101,
    0x000001010101ff00, 0x0001ffffffff0000, 0x0001ffffff0000ff, 0x0001ffffff000001,
    0x0001ffffff000100, 0x0001ffffff010000, 0x0001ffff00ff00ff, 0x0001ffff0000ffff,
    0x0001ffff00000000, 0x0001ffff00000001, 0x0001ffff000001ff, 0x0001ffff00000101,
    0x0001ffff0001ff00, 0x0001ffff000100ff, 0x0001ffff00010001, 0x0001ffff00010100,
    0x0001ffff01ffff00, 0x0001ffff01000001, 0x0001ffff01010000, 0x0001ff00ffffff00,
    0x0001ff00ffff00ff, 0x0001ff00ffff0001, 0x0001ff00ffff0100, 0x0001ff00ff00ff01,
    0x0001ff00ff000000, 0x0001ff00ff01ff00, 0x0001ff00ff01ff01, 0x0001ff00ff010001,
    0x0001ff00ff010100, 0x0001ff0000ff0000, 0x0001ff0000ff0100, 0x0001ff000000ff00,
    0x0001ff0000000000, 0x0001ff0000000001, 0x0001ff0000000100, 0x0001ff0000010000,
    0x0001ff0000010001, 0x0001ff0000010101, 0x0001ff0001ff00ff, 0x0001ff0001ff0101,
    0x0001ff000100ff01, 0x0001ff0001000000, 0x0001ff000101ff00, 0x0001ff0001010001,
    0x0001ff0001010100, 0x0001ff01ff00ff00, 0x0001ff01ff000001, 0x0001ff01ff000100,
    0x0001ff0100ffffff, 0x0001ff0100ffff00, 0x0001ff0100ff0001, 0x0001ff0100000000,
    0x0001ff0100000001, 0x0001ff01000001ff, 0x0001ff010001ffff, 0x0001ff0101ff0000,
    0x0001ff010100ff00, 0x0001ff0101000001, 0x0001ff0101010000, 0x000100ffff00ff00,
    0x000100ffff00ff01, 0x000100ffff000000, 0x000100ffff000001, 0x000100ffff000101,
    0x000100ffff01ff00, 0x000100ffff010001, 0x000100ffff010100, 0x000100ff00ffffff,
    0x000100ff00ffff01, 0x000100ff00ff0000, 0x000100ff00ff01ff, 0x000100ff00ff0101,
    0x000100ff0000ff00, 0x000100ff000000ff, 0x000100ff00000000, 0x000100ff00000001,
    0x000100ff00000100, 0x000100ff00000101, 0x000100ff0001ffff, 0x000100ff0001ff01,
    0x000100ff00010000, 0x000100ff01ff00ff, 0x000100ff01ff0000, 0x000100ff01ff0100,
    0x000100ff0100ffff, 0x000100ff0100ff01, 0x000100ff010000ff, 0x000100ff01000000,
    0x000100ff01000001, 0x000100ff010001ff, 0x000100ff01000101, 0x000100ff0101ff00,
    0x000100ff010100ff, 0x000100ff01010100, 0x00010000ffff0000, 0x00010000ffff01ff,
    0x00010000ffff0101, 0x00010000ff00ff00, 0x00010000ff000000, 0x00010000ff000001,
    0x00010000ff000100, 0x0001000000ff00ff, 0x0001000000ff0000, 0x0001000000ff0001,
    0x0001000000ff0100, 0x000100000000ffff, 0x000100000000ff00, 0x00010000000000ff,
    0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x000100000001ff00,
    0x00010000000100ff, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100,
    0x0001000001ff0001, 0x0001000001ff0100, 0x0001000001ff0101, 0x000100000100ff00,
    0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101,
    0x000100000101ff01, 0x0001000001010000, 0x0001000001010001, 0x00010000010101ff,
    0x00010001ffffff01, 0x00010001ffff0100, 0x00010001ff000000, 0x00010001ff01ffff,
    0x00010001ff010001, 0x00010001ff0101ff, 0x00010001ff010100, 0x0001000100ffffff,
    0x0001000100ff0000, 0x0001000100ff01ff, 0x0001000100ff0101, 0x000100010000ff00,
    0x00010001000000ff, 0x0001000100000000, 0x0001000100000001, 0x00010001000001ff,
    0x0001000100000101, 0x000100010001ffff, 0x0001000100010000, 0x00010001000101ff,
    0x0001000101ffffff, 0x0001000101ffff01, 0x0001000101ff0000, 0x0001000101ff0101,
    0x00010001010000ff, 0x0001000101000001, 0x00010001010001ff, 0x0001000101000100,
    0x000100010101ffff, 0x00010001010100ff, 0x0001000101010001, 0x0001000101010101,
    0x000101ffff000001, 0x000101ffff000100, 0x000101ffff010000, 0x000101ff00ffff00,
    0x000101ff0000ff01, 0x000101ff00000000, 0x000101ff00000101, 0x000101ff0001ff00,
    0x000101ff00010100, 0x000101ff01ff0000, 0x000101ff0100ff00, 0x000101ff010001ff,
    0x000101ff01010001, 0x00010100ffffff00, 0x00010100ffff00ff, 0x00010100ff00ffff,
    0x00010100ff000000, 0x00010100ff01ff00, 0x00010100ff0100ff, 0x00010100ff010001,
    0x00010100ff010100, 0x0001010000ffffff, 0x0001010000ffff00, 0x0001010000ff0000,
    0x0001010000ff0001, 0x0001010000ff01ff, 0x000101000000ff00, 0x00010100000000ff,
    0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x000101000001ffff,
    0x0001010000010000, 0x0001010000010101, 0x0001010001ffff01, 0x0001010001ff00ff,
    0x0001010001ff0101, 0x0001010001000000, 0x000101000101ff00, 0x00010100010100ff,
    0x0001010001010000, 0x0001010001010100, 0x00010101ff00ff00, 0x00010101ff000001,
    0x00010101ff0001ff, 0x0001010100ffff00, 0x0001010100ff00ff, 0x0001010100ff0100,
    0x000101010000ffff, 0x0001010100000000, 0x00010101000001ff, 0x0001010100000101,
    0x00010101000100ff, 0x0001010100010000, 0x0001010100010100, 0x0001010101ff0001,
    0x00010101010000ff, 0x00010101010001ff, 0x0001010101000101, 0x0001010101010001,
    0x01ffffffffffffff, 0x01ffffffffffff01, 0x01ffffffffff01ff, 0x01ffffffffff0101,
    0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff, 0x01ffffffff010101,
    0x01ffffff00ff0000, 0x01ffffff0000ffff, 0x01ffffff0000ff00, 0x01ffffff000000ff,
    0x01ffffff00000001, 0x01ffffff00000100, 0x01ffffff00010000, 0x01ffffff01ffffff,
    0x01ffffff01ffff01, 0x01ffffff01ff01ff, 0x01ffffff01ff0101, 0x01ffffff01000000,
    0x01ffffff0101ffff, 0x01ffffff0101ff01, 0x01ffffff010101ff, 0x01ffffff01010101,
    0x01ffff00ffff0000, 0x01ffff00ff00ff00, 0x01ffff00ff0000ff, 0x01ffff00ff000001,
    0x01ffff00ff000100, 0x01ffff00ff010000, 0x01ffff0000ffff00, 0x01ffff0000ff00ff,
    0x01ffff0000ff0100, 0x01ffff000000ffff, 0x01ffff000000ff01, 0x01ffff0000000000,
    0x01ffff0000000001, 0x01ffff00000001ff, 0x01ffff0000000100, 0x01ffff00000100ff,
    0x01ffff0000010001, 0x01ffff0000010100, 0x01ffff0001ff0000, 0x01ffff0001ff0100,
    0x01ffff00010000ff, 0x01ffff0001000001, 0x01ffff0001000100, 0x01ffff0001010000,
    0x01ffff01ffffffff, 0x01ffff01ffffff01, 0x01ffff01ffff01ff, 0x01ffff01ffff0101,
    0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff01ff01, 0x01ffff01ff0101ff,
    0x01ffff01ff010101, 0x01ffff010000ff00, 0x01ffff01000000ff, 0x01ffff0100000100,
    0x01ffff0100010000, 0x01ffff0101ffffff, 0x01ffff0101ffff01, 0x01ffff0101ff01ff,
    0x01ffff0101ff0101, 0x01ffff0101000000, 0x01ffff010101ffff, 0x01ffff010101ff01,
    0x01ffff01010101ff, 0x01ffff0101010101, 0x01ff00ffff0000ff, 0x01ff00ffff000100,
    0x01ff00ff00ffff00, 0x01ff00ff00ff00ff, 0x01ff00ff0000ff00, 0x01ff00ff00000000,
    0x01ff00ff00000101, 0x01ff00ff0001ff00, 0x01ff00ff000100ff, 0x01ff00ff00010100,
    0x01ff00ff010000ff, 0x01ff00ff01000100, 0x01ff0000ffffff00, 0x01ff0000ffff0100,
    0x01ff0000ff00ff01, 0x01ff0000ff000000, 0x01ff0000ff000101, 0x01ff0000ff010001,
    0x01ff0000ff010100, 0x01ff000000ffffff, 0x01ff000000ffff00, 0x01ff000000ff0000,
    0x01ff000000ff01ff, 0x01ff00000000ff00, 0x01ff0000000000ff, 0x01ff000000000000,
    0x01ff000000000001, 0x01ff000000000100, 0x01ff000000000101, 0x01ff000000010000,
    0x01ff000000010001, 0x01ff0000000101ff, 0x01ff000000010101, 0x01ff000001ffff00,
    0x01ff000001ff00ff, 0x01ff000001ff0001, 0x01ff000001ff0100, 0x01ff00000100ffff,
    0x01ff00000100ff01, 0x01ff000001000000, 0x01ff0000010001ff, 0x01ff000001010001,
    0x01ff0001ff00ff00, 0x01ff0001ff000001, 0x01ff0001ff000100, 0x01ff0001ff010000,
    0x01ff000100ffff00, 0x01ff000100ff00ff, 0x01ff000100ff0100, 0x01ff000100ff0101,
    0x01ff00010000ffff, 0x01ff000100000000, 0x01ff000100000100, 0x01ff000100000101,
    0x01ff00010001ff00, 0x01ff000100010001, 0x01ff000100010101, 0x01ff000101ff0000,
    0x01ff00010100ff00, 0x01ff000101000101, 0x01ff0001010100ff, 0x01ff01ffffffffff,
    0x01ff01ffffffff01, 0x01ff01ffffff01ff, 0x01ff01ffffff0101, 0x01ff01ffff000000,
    0x01ff01ffff01ffff, 0x01ff01ffff01ff01, 0x01ff01ffff0101ff, 0x01ff01ffff010101,
    0x01ff01ff00ffff00, 0x01ff01ff00ff0000, 0x01ff01ff0000ff00, 0x01ff01ff000000ff,
    0x01ff01ff00000100, 0x01ff01ff00010000, 0x01ff01ff00010100, 0x01ff01ff01ffffff,
    0x01ff01ff01ffff01, 0x01ff01ff01ff01ff, 0x01ff01ff01ff0101, 0x01ff01ff01000000,
    0x01ff01ff0101ffff, 0x01ff01ff0101ff01, 0x01ff01ff010101ff, 0x01ff01ff01010101,
    0x01ff0100ffff0000, 0x01ff0100ffff0001, 0x01ff0100ff00ff00, 0x01ff0100ff0000ff,
    0x01ff0100ff000001, 0x01ff0100ff010000, 0x01ff010000ffff00, 0x01ff010000ff00ff,
    0x01ff010000ff0001, 0x01ff010000ff0100, 0x01ff01000000ffff, 0x01ff01000000ff01,
    0x01ff010000000000, 0x01ff010000000101, 0x01ff01000001ff00, 0x01ff0100000100ff,
    0x01ff010001ff0000, 0x01ff010001000001, 0x01ff010001000100, 0x01ff010001010000,
    0x01ff0101ffffffff, 0x01ff0101ffffff01, 0x01ff0101ffff01ff, 0x01ff0101ffff0101,
    0x01ff0101ff000000, 0x01ff0101ff01ffff, 0x01ff0101ff01ff01, 0x01ff0101ff0101ff,
    0x01ff0101ff010101, 0x01ff010100ff0000, 0x01ff01010000ff00, 0x01ff0101000000ff,
    0x01ff010100000001, 0x01ff010101ffffff, 0x01ff010101ffff01, 0x01ff010101ff01ff,
    0x01ff010101ff0101, 0x01ff010101000000, 0x01ff01010101ffff, 0x01ff01010101ff01,
    0x01ff0101010101ff, 0x01ff010101010101, 0x0100ffffffff0000, 0x0100ffffff00ff00,
    0x0100ffffff000001, 0x0100ffffff0001ff, 0x0100ffffff000100, 0x0100ffffff010000,
    0x0100ffff00ffff00, 0x0100ffff00ff0001, 0x0100ffff00ff0100, 0x0100ffff00000000,
    0x0100ffff000001ff, 0x0100ffff00000101, 0x0100ffff00010100, 0x0100ffff00010101,
    0x0100ffff01ff0000, 0x0100ffff0100ff00, 0x0100ffff010000ff, 0x0100ffff01000001,
    0x0100ffff01000100, 0x0100ffff01010000, 0x0100ff00ffffff00, 0x0100ff00ffff00ff,
    0x0100ff00ffff0001, 0x0100ff00ffff0100, 0x0100ff00ff00ffff, 0x0100ff00ff000000,
    0x0100ff00ff0001ff, 0x0100ff00ff000101, 0x0100ff00ff01ff00, 0x0100ff00ff0100ff,
    0x0100ff00ff010001, 0x0100ff00ff010100, 0x0100ff0000ffffff, 0x0100ff0000ff0000,
    0x0100ff000000ffff, 0x0100ff000000ff00, 0x0100ff00000000ff, 0x0100ff0000000000,
    0x0100ff0000000001, 0x0100ff0000000100, 0x0100ff000001ff01, 0x0100ff0000010000,
    0x0100ff0001ff00ff, 0x0100ff0001ff0001, 0x0100ff000100ff01, 0x0100ff0001000000,
    0x0100ff00010001ff, 0x0100ff000101ff00, 0x0100ff00010100ff, 0x0100ff0001010001,
    0x0100ff0001010100, 0x0100ff01ffff0000, 0x0100ff01ff00ff00, 0x0100ff01ff0000ff,
    0x0100ff01ff000100, 0x0100ff01ff010000, 0x0100ff0100ff00ff, 0x0100ff0100ff0001,
    0x0100ff0100ff0100, 0x0100ff010000ffff, 0x0100ff010000ff01, 0x0100ff0100000000,
    0x0100ff01000001ff, 0x0100ff0100010001, 0x0100ff0100010100, 0x0100ff0101ff0000,
    0x0100ff01010000ff, 0x0100ff0101000001, 0x0100ff0101010100, 0x010000ffffffff00,
    0x010000ffffff00ff, 0x010000ffffff0001, 0x010000ffff00ffff, 0x010000ffff000000,
    0x010000ffff0001ff, 0x010000ffff010001, 0x010000ff00ffffff, 0x010000ff00ff0101,
    0x010000ff0000ff00, 0x010000ff000000ff, 0x010000ff00000000, 0x010000ff00000001,
    0x010000ff000001ff, 0x010000ff00000100, 0x010000ff0001ffff, 0x010000ff0001ff00,
    0x010000ff0001ff01, 0x010000ff00010000, 0x010000ff01ff00ff, 0x010000ff01ff0001,
    0x010000ff0100ff01, 0x010000ff010000ff, 0x010000ff01000000, 0x010000ff010001ff,
    0x010000ff0101ff00, 0x010000ff01010100, 0x01000000ffffffff, 0x01000000ffff0000,
    0x01000000ffff01ff, 0x01000000ffff0101, 0x01000000ff00ffff, 0x01000000ff00ff00,
    0x01000000ff0000ff, 0x01000000ff000000, 0x01000000ff000001, 0x01000000ff000100,
    0x01000000ff01ff00, 0x01000000ff010000, 0x01000000ff010100, 0x01000000ff010101,
    0x0100000000ffff00, 0x0100000000ff00ff, 0x0100000000ff0000, 0x0100000000ff0001,
    0x0100000000ff0100, 0x010000000000ffff, 0x010000000000ff00, 0x010000000000ff01,
    0x01000000000000ff, 0x0100000000000000, 0x0100000000000001, 0x01000000000001ff,
    0x0100000000000100, 0x0100000000000101, 0x010000000001ff00, 0x01000000000100ff,
    0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000001ffff00,
    0x0100000001ff0000, 0x0100000001ff01ff, 0x010000000100ff00, 0x010000000100ff01,
    0x01000000010000ff, 0x0100000001000000, 0x0100000001000001, 0x0100000001000100,
    0x0100000001000101, 0x010000000101ffff, 0x010000000101ff01, 0x0100000001010000,
    0x01000000010101ff, 0x0100000001010101, 0x01000001ffffff00, 0x01000001ffff00ff,
    0x01000001ff00ffff, 0x01000001ff000000, 0x01000001ff000100, 0x01000001ff01ffff,
    0x01000001ff010001, 0x01000001ff010100, 0x0100000100ff0000, 0x0100000100ff01ff,
    0x0100000100ff0100, 0x010000010000ff00, 0x010000010000ff01, 0x0100000100000000,
    0x0100000100000001, 0x0100000100000100, 0x0100000100010000, 0x01000001000101ff,
    0x0100000101ffff01, 0x0100000101ff00ff, 0x0100000101ff0100, 0x0100000101ff0101,
    0x010000010100ff01, 0x01000001010000ff, 0x0100000101000000, 0x01000001010100ff,
    0x0100000101010001, 0x0100000101010100, 0x010001ffffff0000, 0x010001ffff000001,
    0x010001ffff000100, 0x010001ffff010000, 0x010001ff00ffff00, 0x010001ff00ff0001,
    0x010001ff0000ffff, 0x010001ff0000ff01, 0x010001ff00000000, 0x010001ff00000001,
    0x010001ff00000101, 0x010001ff000100ff, 0x010001ff00010000, 0x010001ff01ff0000,
    0x010001ff0100ff00, 0x010001ff01000001, 0x010001ff01000100, 0x010001ff01010000,
    0x01000100ffff00ff, 0x01000100ffff0001, 0x01000100ffff0100, 0x01000100ff00ffff,
    0x01000100ff00ff01, 0x01000100ff000000, 0x01000100ff0001ff, 0x01000100ff000101,
    0x01000100ff01ffff, 0x01000100ff01ff00, 0x01000100ff0100ff, 0x01000100ff010001,
    0x0100010000ffffff, 0x0100010000ffff01, 0x0100010000ff0000, 0x0100010000ff01ff,
    0x0100010000ff0101, 0x010001000000ff00, 0x01000100000000ff, 0x0100010000000000,
    0x0100010000000001, 0x0100010000000100, 0x010001000001ff01, 0x0100010000010000,
    0x0100010000010001, 0x0100010000010101, 0x0100010001ffff00, 0x0100010001ff00ff,
    0x010001000100ffff, 0x010001000100ff01, 0x0100010001000000, 0x0100010001000101,
    0x010001000101ff00, 0x0100010001010001, 0x01000101ffff0000, 0x01000101ff000000,
    0x01000101ff010000, 0x0100010100ff00ff, 0x0100010100ff0001, 0x0100010100ff0100,
    0x010001010000ffff, 0x0100010100000000, 0x01000101000001ff, 0x010001010001ff00,
    0x0100010101ff0000, 0x010001010100ff00, 0x01000101010000ff, 0x0100010101000000,
    0x0100010101000001, 0x0101ffffffffffff, 0x0101ffffffffff01, 0x0101ffffffff01ff,
    0x0101ffffffff0101, 0x0101ffffff000000, 0x0101ffffff01ffff, 0x0101ffffff01ff01,
    0x0101ffffff0101ff, 0x0101ffffff010101, 0x0101ffff00ff0000, 0x0101ffff0000ff00,
    0x0101ffff000000ff, 0x0101ffff00000001, 0x0101ffff00000100, 0x0101ffff01ffffff,
    0x0101ffff01ffff01, 0x0101ffff01ff01ff, 0x0101ffff01ff0101, 0x0101ffff01000000,
    0x0101ffff0101ffff, 0x0101ffff0101ff01, 0x0101ffff010101ff, 0x0101ffff01010101,
    0x0101ff00ffff0000, 0x0101ff00ffff0100, 0x0101ff00ff00ff00, 0x0101ff00ff0000ff,
    0x0101ff00ff000001, 0x0101ff00ff000100, 0x0101ff00ff000101, 0x0101ff0000ff0001,
    0x0101ff0000ff0100, 0x0101ff000000ff00, 0x0101ff0000000000, 0x0101ff00000001ff,
    0x0101ff0000000101, 0x0101ff000001ff00, 0x0101ff00000100ff, 0x0101ff0001ff0000,
    0x0101ff000100ffff, 0x0101ff000100ff01, 0x0101ff0001000001, 0x0101ff0001000100,
    0x0101ff01ffffff01, 0x0101ff01ffff01ff, 0x0101ff01ffff0101, 0x0101ff01ff00ffff,
    0x0101ff01ff000100, 0x0101ff01ff01ff01, 0x0101ff01ff0101ff, 0x0101ff01ff010101,
    0x0101ff0100ff0000, 0x0101ff010000ff00, 0x0101ff0100000001, 0x0101ff0100000100,
    0x0101ff0100010000, 0x0101ff0101ffffff, 0x0101ff0101ffff01, 0x0101ff0101ff01ff,
    0x0101ff0101ff0101, 0x0101ff0101000000, 0x0101ff010101ffff, 0x0101ff010101ff01,
    0x0101ff01010101ff, 0x0101ff0101010101, 0x010100ffff000100, 0x010100ffff010000,
    0x010100ff00ffff00, 0x010100ff00ff00ff, 0x010100ff0000ffff, 0x010100ff000000ff,
    0x010100ff00000000, 0x010100ff000001ff, 0x010100ff00000101, 0x010100ff0001ff00,
    0x010100ff00010000, 0x010100ff00010001, 0x010100ff000101ff, 0x010100ff00010100,
    0x010100ff01ff0000, 0x01010000ffff0001, 0x01010000ffff0100, 0x01010000ff00ffff,
    0x01010000ff00ff01, 0x01010000ff000000, 0x01010000ff0001ff, 0x01010000ff010001,
    0x01010000ff010100, 0x0101000000ffff01, 0x0101000000ff0000, 0x010100000000ff00,
    0x01010000000000ff, 0x0101000000000000, 0x0101000000000001, 0x0101000000000100,
    0x0101000000010000, 0x0101000000010101, 0x0101000001ffff00, 0x0101000001ff00ff,
    0x0101000001ff0000, 0x0101000001ff0001, 0x0101000001ff0100, 0x010100000100ff01,
    0x0101000001000000, 0x01010000010001ff, 0x01010001ffff0000, 0x01010001ff00ff00,
    0x01010001ff000001, 0x01010001ff000101, 0x01010001ff01ff00, 0x01010001ff010000,
    0x0101000100ff00ff, 0x0101000100ff0001, 0x0101000100ff0101, 0x010100010000ff01,
    0x0101000100000000, 0x0101000100000001, 0x01010001000001ff, 0x010100010001ffff,
    0x010100010001ff01, 0x0101000101ff0001, 0x010100010100ffff, 0x0101000101000000,
    0x0101000101000001, 0x0101000101000100, 0x010100010101ff00, 0x01010001010100ff,
    0x0101000101010001, 0x010101ffffffffff, 0x010101ffffffff01, 0x010101ffffff01ff,
    0x010101ffffff0101, 0x010101ffff01ffff, 0x010101ffff01ff01, 0x010101ffff0101ff,
    0x010101ffff010101, 0x010101ff0000ff00, 0x010101ff000000ff, 0x010101ff00000001,
    0x010101ff00000100, 0x010101ff01ffffff, 0x010101ff01ffff01, 0x010101ff01ff01ff,
    0x010101ff01ff0101, 0x010101ff01000000, 0x010101ff0101ffff, 0x010101ff0101ff01,
    0x010101ff010101ff, 0x010101ff01010101, 0x01010100ffff0000, 0x01010100ff0000ff,
    0x01010100ff000100, 0x01010100ff01ff00, 0x01010100ff010000, 0x0101010000ffff00,
    0x010101000000ffff, 0x0101010000000000, 0x0101010000000101, 0x010101000001ff00,
    0x0101010000010001, 0x0101010000010100, 0x010101000100ffff, 0x0101010001000001,
    0x01010101ffffffff, 0x01010101ffffff01, 0x01010101ffff01ff, 0x01010101ffff0101,
    0x01010101ff01ffff, 0x01010101ff01ff01, 0x01010101ff0101ff, 0x01010101ff010101,
    0x010101010000ff00, 0x01010101000000ff, 0x0101010100000001, 0x0101010101ffffff,
    0x0101010101ffff01, 0x0101010101ff01ff, 0x0101010101ff0101, 0x0101010101000000,
    0x010101010101ffff, 0x010101010101ff01, 0x01010101010101ff, 0x0101010101010101,
};

static uint8_t rt_iq2xxs_signs(uint32_t idx) {
    uint32_t v = idx & 127u;
    uint32_t p = v;
    p ^= p >> 4;
    p ^= p >> 2;
    p ^= p >> 1;
    return (uint8_t)(v | ((p & 1u) << 7));
}

static void rt_q4_k_scale_min(int group, const uint8_t *scales, uint8_t *d, uint8_t *m) {
    if (group < 4) {
        *d = scales[group] & 63u;
        *m = scales[group + 4] & 63u;
    } else {
        *d = (uint8_t)((scales[group + 4] & 0x0fu) |
                       ((scales[group - 4] >> 6) << 4));
        *m = (uint8_t)((scales[group + 4] >> 4) |
                       ((scales[group] >> 6) << 4));
    }
}

static bool rt_gguf_tensor_read_q2_k(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 84u;
    const uint8_t *scales = b;
    const uint8_t *qs = b + 16;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b + 80));
    const float dmin = rt_f16_to_f32(rt_read_le16_mem(b + 82));

    const uint32_t half = ib / 128u;
    const uint32_t local = ib & 127u;
    const uint32_t group = ib / 16u;
    const uint32_t shift = (local / 32u) * 2u;
    const uint32_t qidx = half * 32u + (local & 31u);
    const uint8_t sc = scales[group];
    const uint8_t q = (uint8_t)((qs[qidx] >> shift) & 3u);
    *out = d * (float)(sc & 0x0fu) * (float)q -
           dmin * (float)(sc >> 4);
    return true;
}

static bool rt_gguf_tensor_read_q3_k(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 110u;
    const uint8_t *hm = b;
    const uint8_t *qs = b + 32;
    const uint8_t *packed_scales = b + 96;
    const float d_all = rt_f16_to_f32(rt_read_le16_mem(b + 108));

    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    aux[0] = rt_read_le32_mem(packed_scales + 0);
    aux[1] = rt_read_le32_mem(packed_scales + 4);
    aux[2] = rt_read_le32_mem(packed_scales + 8);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    uint8_t scale_bytes[16];
    for (int i = 0; i < 4; i++) {
        scale_bytes[i * 4 + 0] = (uint8_t)(aux[i] & 0xffu);
        scale_bytes[i * 4 + 1] = (uint8_t)((aux[i] >> 8) & 0xffu);
        scale_bytes[i * 4 + 2] = (uint8_t)((aux[i] >> 16) & 0xffu);
        scale_bytes[i * 4 + 3] = (uint8_t)((aux[i] >> 24) & 0xffu);
    }

    const uint32_t half = ib / 128u;
    const uint32_t local = ib & 127u;
    const uint32_t group = ib / 16u;
    const uint32_t shift = (local / 32u) * 2u;
    const uint32_t qidx = half * 32u + (local & 31u);
    const uint8_t high_mask = (uint8_t)(1u << (group / 2u));
    const int scale = (int)(int8_t)scale_bytes[group] - 32;
    const int q = (int)((qs[qidx] >> shift) & 3u) -
                  ((hm[qidx] & high_mask) ? 0 : 4);
    *out = d_all * (float)scale * (float)q;
    return true;
}

static bool rt_gguf_tensor_read_q4_0(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 32u;
    const uint32_t ib = (uint32_t)(index % 32u);
    const uint8_t *b = data + block * 18u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const uint8_t *qs = b + 2;
    const uint8_t packed = qs[ib & 15u];
    const uint8_t q = ib < 16u ? (packed & 0x0fu) : (packed >> 4);
    *out = d * ((float)q - 8.0f);
    return true;
}

static bool rt_gguf_tensor_read_q4_1(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 32u;
    const uint32_t ib = (uint32_t)(index % 32u);
    const uint8_t *b = data + block * 20u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const float m = rt_f16_to_f32(rt_read_le16_mem(b + 2));
    const uint8_t *qs = b + 4;
    const uint8_t packed = qs[ib & 15u];
    const uint8_t q = ib < 16u ? (packed & 0x0fu) : (packed >> 4);
    *out = d * (float)q + m;
    return true;
}

static bool rt_gguf_tensor_read_q5_0(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 32u;
    const uint32_t ib = (uint32_t)(index % 32u);
    const uint8_t *b = data + block * 22u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const uint32_t qh = rt_read_le32_mem(b + 2);
    const uint8_t *qs = b + 6;
    const uint8_t packed = qs[ib & 15u];
    uint8_t q = ib < 16u ? (packed & 0x0fu) : (packed >> 4);
    q |= (uint8_t)(((qh >> ib) & 1u) << 4);
    *out = d * ((float)q - 16.0f);
    return true;
}

static bool rt_gguf_tensor_read_q5_1(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 32u;
    const uint32_t ib = (uint32_t)(index % 32u);
    const uint8_t *b = data + block * 24u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const float m = rt_f16_to_f32(rt_read_le16_mem(b + 2));
    const uint32_t qh = rt_read_le32_mem(b + 4);
    const uint8_t *qs = b + 8;
    const uint8_t packed = qs[ib & 15u];
    uint8_t q = ib < 16u ? (packed & 0x0fu) : (packed >> 4);
    q |= (uint8_t)(((qh >> ib) & 1u) << 4);
    *out = d * (float)q + m;
    return true;
}

static bool rt_gguf_tensor_read_q8_0(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 32u;
    const uint64_t ib = index % 32u;
    const uint8_t *b = data + block * 34u;
    float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const int8_t *qs = (const int8_t *)(b + 2);
    *out = d * (float)qs[ib];
    return true;
}

static bool rt_gguf_tensor_read_q8_1(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 32u;
    const uint64_t ib = index % 32u;
    const uint8_t *b = data + block * 36u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const int8_t *qs = (const int8_t *)(b + 4);
    *out = d * (float)qs[ib];
    return true;
}

static bool rt_gguf_tensor_read_q4_k(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 144u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const float dmin = rt_f16_to_f32(rt_read_le16_mem(b + 2));
    const uint8_t *scales = b + 4;
    const uint8_t *qs = b + 16;

    const int group = (int)(ib / 32u);
    uint8_t sc = 0;
    uint8_t m = 0;
    rt_q4_k_scale_min(group, scales, &sc, &m);

    const uint32_t within64 = ib & 63u;
    const uint32_t byte_index = (ib / 64u) * 32u + (within64 & 31u);
    const uint8_t packed = qs[byte_index];
    const uint8_t q = within64 < 32u ? (packed & 0x0fu) : (packed >> 4);
    *out = d * (float)sc * (float)q - dmin * (float)m;
    return true;
}

static bool rt_gguf_tensor_read_q5_k(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 176u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const float dmin = rt_f16_to_f32(rt_read_le16_mem(b + 2));
    const uint8_t *scales = b + 4;
    const uint8_t *qh = b + 16;
    const uint8_t *qs = b + 48;

    const int group = (int)(ib / 32u);
    uint8_t sc = 0;
    uint8_t m = 0;
    rt_q4_k_scale_min(group, scales, &sc, &m);

    const uint32_t within64 = ib & 63u;
    const uint32_t byte_index = (ib / 64u) * 32u + (within64 & 31u);
    const uint8_t packed = qs[byte_index];
    uint8_t q = within64 < 32u ? (packed & 0x0fu) : (packed >> 4);
    q |= (uint8_t)(((qh[ib >> 3] >> (ib & 7u)) & 1u) << 4);
    *out = d * (float)sc * (float)q - dmin * (float)m;
    return true;
}

static bool rt_gguf_tensor_read_q6_k(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 210u;
    const uint8_t *ql = b;
    const uint8_t *qh = b + 128;
    const int8_t *scales = (const int8_t *)(b + 192);
    const float d = rt_f16_to_f32(rt_read_le16_mem(b + 208));

    const uint32_t half = ib / 128u;
    const uint32_t local = ib & 127u;
    const uint32_t l = local & 31u;
    const uint8_t *qlh = ql + half * 64u;
    const uint8_t qhb = qh[half * 32u + l];
    const int scale_base = (int)half * 8;
    int8_t q = 0;
    int8_t sc = 0;

    if (local < 32u) {
        q = (int8_t)((qlh[l] & 0x0fu) | (((qhb >> 0) & 3u) << 4));
        sc = scales[scale_base + (int)(l / 16u) + 0];
    } else if (local < 64u) {
        q = (int8_t)((qlh[l + 32u] & 0x0fu) | (((qhb >> 2) & 3u) << 4));
        sc = scales[scale_base + (int)(l / 16u) + 2];
    } else if (local < 96u) {
        q = (int8_t)((qlh[l] >> 4) | (((qhb >> 4) & 3u) << 4));
        sc = scales[scale_base + (int)(l / 16u) + 4];
    } else {
        q = (int8_t)((qlh[l + 32u] >> 4) | (((qhb >> 6) & 3u) << 4));
        sc = scales[scale_base + (int)(l / 16u) + 6];
    }
    *out = d * (float)sc * (float)(q - 32);
    return true;
}

static bool rt_gguf_tensor_read_q8_k(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint64_t ib = index % 256u;
    const uint8_t *b = data + block * 292u;
    const float d = rt_f32_from_bits(rt_read_le32_mem(b));
    const int8_t *qs = (const int8_t *)(b + 4);
    *out = d * (float)qs[ib];
    return true;
}

static bool rt_gguf_tensor_read_iq2_xxs(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 66u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const uint8_t *q2 = b + 2;
    const uint32_t ib32 = ib / 32u;
    const uint32_t local = ib & 31u;
    const uint32_t group = local / 8u;
    const uint32_t item = local & 7u;
    const uint8_t *chunk = q2 + ib32 * 8u;
    const uint32_t aux_g = rt_read_le32_mem(chunk);
    const uint32_t aux_s = rt_read_le32_mem(chunk + 4);
    const uint32_t grid_idx = (aux_g >> (8u * group)) & 255u;
    const uint32_t sign_idx = (aux_s >> (7u * group)) & 127u;
    const uint64_t grid = rt_iq2xxs_grid[grid_idx];
    const uint8_t signs = rt_iq2xxs_signs(sign_idx);
    float v = (float)((grid >> (8u * item)) & 255u);
    if (signs & (1u << item)) v = -v;
    *out = d * (0.5f + (float)(aux_s >> 28)) * 0.25f * v;
    return true;
}

static bool rt_gguf_tensor_read_iq2_xs(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 74u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const uint8_t *qs = b + 2;
    const uint8_t *scales = qs + 64u;
    const uint32_t ib32 = ib / 32u;
    const uint32_t local = ib & 31u;
    const uint32_t group = local / 8u;
    const uint32_t item = local & 7u;
    const uint16_t packed = rt_read_le16_mem(qs + 2u * (ib32 * 4u + group));
    const uint64_t grid = rt_iq2xs_grid[packed & 511u];
    const uint8_t signs = rt_iq2xxs_signs(packed >> 9);
    const uint8_t scale = group < 2u ? (scales[ib32] & 15u) : (scales[ib32] >> 4);
    float v = (float)((grid >> (8u * item)) & 255u);
    if (signs & (1u << item)) v = -v;
    *out = d * (0.5f + (float)scale) * 0.25f * v;
    return true;
}

static bool rt_gguf_tensor_read_iq2_s(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 82u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const uint8_t *qs = b + 2;
    const uint8_t *qh = b + 66;
    const uint8_t *scales = b + 74;
    const uint32_t ib32 = ib / 32u;
    const uint32_t local = ib & 31u;
    const uint32_t group = local / 8u;
    const uint32_t item = local & 7u;
    const uint32_t grid_idx =
        (uint32_t)qs[ib32 * 4u + group] |
        (((uint32_t)qh[ib32] << (8u - 2u * group)) & 0x300u);
    const uint64_t grid = rt_iq2s_grid[grid_idx];
    const uint8_t signs = qs[32u + ib32 * 4u + group];
    const uint8_t scale = group < 2u ? (scales[ib32] & 15u) : (scales[ib32] >> 4);
    float v = (float)((grid >> (8u * item)) & 255u);
    if (signs & (1u << item)) v = -v;
    *out = d * (0.5f + (float)scale) * 0.25f * v;
    return true;
}

static bool rt_gguf_tensor_read_iq3_xxs(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 98u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const uint8_t *qs = b + 2;
    const uint8_t *scales_and_signs = qs + 64u;
    const uint32_t ib32 = ib / 32u;
    const uint32_t local = ib & 31u;
    const uint32_t group = local / 8u;
    const uint32_t item = local & 7u;
    const uint32_t aux = rt_read_le32_mem(scales_and_signs + 4u * ib32);
    const uint8_t signs = rt_iq2xxs_signs((aux >> (7u * group)) & 127u);
    const uint8_t q = qs[ib32 * 8u + group * 2u + (item >= 4u ? 1u : 0u)];
    const uint32_t grid = rt_iq3xxs_grid[q];
    float v = (float)((grid >> (8u * (item & 3u))) & 255u);
    if (signs & (1u << item)) v = -v;
    *out = d * (0.5f + (float)(aux >> 28)) * 0.5f * v;
    return true;
}

static bool rt_gguf_tensor_read_iq3_s(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 110u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const uint8_t *qs = b + 2;
    const uint8_t *qh = b + 66;
    const uint8_t *signs = b + 74;
    const uint8_t *scales = b + 106;
    const uint32_t ib32 = ib / 32u;
    const uint32_t pair = ib32 / 2u;
    const uint32_t second = ib32 & 1u;
    const uint32_t local = ib & 31u;
    const uint32_t group = local / 8u;
    const uint32_t item = local & 7u;
    const uint32_t qoff = pair * 16u + second * 8u + group * 2u + (item >= 4u ? 1u : 0u);
    const uint32_t qh_byte = qh[pair * 2u + second];
    const uint32_t qh_shift = item < 4u ? (8u - 2u * group) : (7u - 2u * group);
    const uint32_t grid_idx = (uint32_t)qs[qoff] | ((qh_byte << qh_shift) & 256u);
    const uint32_t grid = rt_iq3s_grid[grid_idx];
    const uint8_t sign = signs[pair * 8u + second * 4u + group];
    const uint8_t scale = second == 0 ? (scales[pair] & 15u) : (scales[pair] >> 4);
    float v = (float)((grid >> (8u * (item & 3u))) & 255u);
    if (sign & (1u << item)) v = -v;
    *out = d * (float)(1u + 2u * scale) * v;
    return true;
}

static bool rt_gguf_tensor_read_iq1_s(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 50u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const uint8_t *qs = b + 2;
    const uint8_t *qh = b + 34;
    const uint32_t ib32 = ib / 32u;
    const uint32_t local = ib & 31u;
    const uint32_t group = local / 8u;
    const uint32_t item = local & 7u;
    const uint16_t qh_word = rt_read_le16_mem(qh + 2u * ib32);
    const uint32_t grid_idx =
        (uint32_t)qs[ib32 * 4u + group] |
        ((((uint32_t)qh_word >> (3u * group)) & 7u) << 8);
    const uint64_t grid = rt_iq1s_grid[grid_idx];
    int q = (int)((grid >> (8u * item)) & 255u);
    if (q >= 128) q -= 256;
    const float dl = d * (float)(2u * ((qh_word >> 12) & 7u) + 1u);
    const float delta = (qh_word & 0x8000u) ? -0.125f : 0.125f;
    *out = dl * ((float)q + delta);
    return true;
}

static bool rt_gguf_tensor_read_iq1_m(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 56u;
    const uint8_t *qs = b;
    const uint8_t *qh = b + 32;
    const uint8_t *scales = b + 48;
    const uint16_t sc0 = rt_read_le16_mem(scales);
    const uint16_t sc1 = rt_read_le16_mem(scales + 2);
    const uint16_t sc2 = rt_read_le16_mem(scales + 4);
    const uint16_t sc3 = rt_read_le16_mem(scales + 6);
    const uint16_t scale_bits = (uint16_t)((sc0 >> 12) |
                                           ((sc1 >> 8) & 0x00f0u) |
                                           ((sc2 >> 4) & 0x0f00u) |
                                           (sc3 & 0xf000u));
    const float d = rt_f16_to_f32(scale_bits);
    const uint32_t ib32 = ib / 32u;
    const uint32_t local = ib & 31u;
    const uint32_t group = local / 8u;
    const uint32_t item = local & 7u;
    const uint16_t sc = rt_read_le16_mem(scales + 2u * (ib32 / 2u));
    const uint32_t shift = 6u * (ib32 & 1u);
    const uint8_t scale = group < 2u ? ((sc >> shift) & 7u) : ((sc >> (shift + 3u)) & 7u);
    const uint8_t qh_byte = qh[ib32 * 2u + group / 2u];
    const uint32_t grid_idx =
        (uint32_t)qs[ib32 * 4u + group] |
        ((uint32_t)(group & 1u ? ((qh_byte << 4) & 0x700u) : ((qh_byte << 8) & 0x700u)));
    const uint64_t grid = rt_iq1s_grid[grid_idx];
    int q = (int)((grid >> (8u * item)) & 255u);
    if (q >= 128) q -= 256;
    const float delta = (qh_byte & (group & 1u ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
    *out = d * (float)(2u * scale + 1u) * ((float)q + delta);
    return true;
}

static bool rt_gguf_tensor_read_iq4_nl(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 32u;
    const uint32_t ib = (uint32_t)(index % 32u);
    const uint8_t *b = data + block * 18u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const uint8_t packed = b[2u + (ib & 15u)];
    const uint8_t q = ib < 16u ? (packed & 15u) : (packed >> 4);
    *out = d * (float)rt_iq4nl_values[q];
    return true;
}

static bool rt_gguf_tensor_read_iq4_xs(const uint8_t *data, uint64_t index, float *out) {
    const uint64_t block = index / 256u;
    const uint32_t ib = (uint32_t)(index % 256u);
    const uint8_t *b = data + block * 136u;
    const float d = rt_f16_to_f32(rt_read_le16_mem(b));
    const uint16_t scales_h = rt_read_le16_mem(b + 2);
    const uint8_t *scales_l = b + 4;
    const uint8_t *qs = b + 8;
    const uint32_t group = ib / 32u;
    const uint32_t local = ib & 31u;
    const int ls = (int)((scales_l[group / 2u] >> (4u * (group & 1u))) & 15u) |
                   (int)(((scales_h >> (2u * group)) & 3u) << 4);
    const uint8_t packed = qs[group * 16u + (local & 15u)];
    const uint8_t q = local < 16u ? (packed & 15u) : (packed >> 4);
    *out = d * (float)(ls - 32) * (float)rt_iq4nl_values[q];
    return true;
}

bool rt_gguf_tensor_read_f32(const rt_gguf_file *file,
                             const rt_gguf_tensor *tensor,
                             uint64_t index,
                             float *out) {
    if (!out || !tensor || index >= tensor->elements) return false;
    const uint8_t *data = rt_gguf_file_tensor_data(file, tensor);
    if (!data) return false;

    switch (tensor->type) {
    case 0:
        *out = rt_f32_from_bits(rt_read_le32_mem(data + index * 4u));
        return true;
    case 1:
        *out = rt_f16_to_f32(rt_read_le16_mem(data + index * 2u));
        return true;
    case 2:
        return rt_gguf_tensor_read_q4_0(data, index, out);
    case 3:
        return rt_gguf_tensor_read_q4_1(data, index, out);
    case 6:
        return rt_gguf_tensor_read_q5_0(data, index, out);
    case 7:
        return rt_gguf_tensor_read_q5_1(data, index, out);
    case 8:
        return rt_gguf_tensor_read_q8_0(data, index, out);
    case 9:
        return rt_gguf_tensor_read_q8_1(data, index, out);
    case 10:
        return rt_gguf_tensor_read_q2_k(data, index, out);
    case 11:
        return rt_gguf_tensor_read_q3_k(data, index, out);
    case 12:
        return rt_gguf_tensor_read_q4_k(data, index, out);
    case 13:
        return rt_gguf_tensor_read_q5_k(data, index, out);
    case 14:
        return rt_gguf_tensor_read_q6_k(data, index, out);
    case 15:
        return rt_gguf_tensor_read_q8_k(data, index, out);
    case 16:
        return rt_gguf_tensor_read_iq2_xxs(data, index, out);
    case 17:
        return rt_gguf_tensor_read_iq2_xs(data, index, out);
    case 18:
        return rt_gguf_tensor_read_iq3_xxs(data, index, out);
    case 19:
        return rt_gguf_tensor_read_iq1_s(data, index, out);
    case 29:
        return rt_gguf_tensor_read_iq1_m(data, index, out);
    case 22:
        return rt_gguf_tensor_read_iq2_s(data, index, out);
    case 21:
        return rt_gguf_tensor_read_iq3_s(data, index, out);
    case 20:
        return rt_gguf_tensor_read_iq4_nl(data, index, out);
    case 23:
        return rt_gguf_tensor_read_iq4_xs(data, index, out);
    case 30:
        *out = rt_f32_from_bits((uint32_t)rt_read_le16_mem(data + index * 2u) << 16);
        return true;
    default:
        return false;
    }
}

static bool rt_gguf_tensor_sparse_matvec_f32(const rt_gguf_file *file,
                                             const rt_gguf_tensor *tensor,
                                             const float *x,
                                             uint64_t in_dim,
                                             float *out,
                                             uint64_t out_dim);

bool rt_gguf_tensor_matvec_f32(const rt_gguf_file *file,
                               const rt_gguf_tensor *tensor,
                               const float *x,
                               uint64_t in_dim,
                               float *out,
                               uint64_t out_dim) {
    if (!file || !tensor || !x || !out ||
        tensor->ndim != 2 ||
        tensor->dim[0] != in_dim ||
        tensor->dim[1] != out_dim ||
        in_dim == 0 ||
        out_dim == 0) {
        return false;
    }
    if (out_dim > UINT64_MAX / in_dim) return false;

    if (rt_gguf_tensor_sparse_matvec_f32(file, tensor, x, in_dim, out, out_dim)) {
        return true;
    }

    for (uint64_t row = 0; row < out_dim; row++) {
        double sum = 0.0;
        uint64_t base = row * in_dim;
        for (uint64_t col = 0; col < in_dim; col++) {
            float w = 0.0f;
            if (!rt_gguf_tensor_read_f32(file, tensor, base + col, &w)) return false;
            sum += (double)w * (double)x[col];
        }
        out[row] = (float)sum;
    }
    return true;
}

static bool rt_gguf_file_span(const rt_gguf_file *file, uint64_t offset, uint64_t len, const uint8_t **out) {
    if (!file || !file->map || offset > file->size || len > file->size - offset) return false;
    if (out) *out = file->map + offset;
    return true;
}

static bool rt_gguf_tensor_sparse_matvec_f32(const rt_gguf_file *file,
                                             const rt_gguf_tensor *tensor,
                                             const float *x,
                                             uint64_t in_dim,
                                             float *out,
                                             uint64_t out_dim) {
#if defined(SEEK_DATA) && defined(SEEK_HOLE)
    if (!file || file->fd < 0 || !tensor || !x || !out ||
        tensor->ndim != 2 ||
        tensor->dim[0] != in_dim ||
        tensor->dim[1] != out_dim ||
        in_dim == 0 ||
        out_dim == 0 ||
        (tensor->type != 0 && tensor->type != 1 && tensor->type != 30)) {
        return false;
    }
    uint64_t elem_bytes = tensor->type == 0 ? 4u : 2u;
    uint64_t start = tensor->data_offset;
    uint64_t bytes = tensor->bytes;
    if (bytes == 0) {
        memset(out, 0, (size_t)out_dim * sizeof(out[0]));
        return true;
    }
    if (start > UINT64_MAX - bytes || start > (uint64_t)LLONG_MAX ||
        start + bytes > (uint64_t)LLONG_MAX) {
        return false;
    }

    const uint64_t max_sparse_data_bytes = UINT64_C(1) << 20;
    uint64_t end = start + bytes;
    uint64_t sparse_data_bytes = 0;
    memset(out, 0, (size_t)out_dim * sizeof(out[0]));

    off_t pos = (off_t)start;
    while ((uint64_t)pos < end) {
        errno = 0;
        off_t data = lseek(file->fd, pos, SEEK_DATA);
        if (data < 0) {
            return errno == ENXIO;
        }
        if ((uint64_t)data >= end) return true;

        off_t hole = lseek(file->fd, data, SEEK_HOLE);
        if (hole < 0 || hole <= data) return false;

        uint64_t data_u = (uint64_t)data < start ? start : (uint64_t)data;
        uint64_t hole_u = (uint64_t)hole > end ? end : (uint64_t)hole;
        if (hole_u > data_u) {
            uint64_t extent_bytes = hole_u - data_u;
            if (extent_bytes > max_sparse_data_bytes ||
                sparse_data_bytes > max_sparse_data_bytes - extent_bytes) {
                return false;
            }
            sparse_data_bytes += extent_bytes;

            uint64_t rel_begin = data_u - start;
            uint64_t rel_end = hole_u - start;
            uint64_t first = rel_begin / elem_bytes;
            uint64_t last = (rel_end + elem_bytes - 1u) / elem_bytes;
            if (last > tensor->elements) last = tensor->elements;

            for (uint64_t idx = first; idx < last; idx++) {
                uint64_t col = idx % in_dim;
                if (x[col] == 0.0f) continue;
                float w = 0.0f;
                if (!rt_gguf_tensor_read_f32(file, tensor, idx, &w)) return false;
                if (w == 0.0f) continue;
                uint64_t row = idx / in_dim;
                if (row < out_dim) out[row] += w * x[col];
            }
        }

        if ((uint64_t)hole >= end) return true;
        pos = hole;
    }
    return true;
#else
    (void)file;
    (void)tensor;
    (void)x;
    (void)in_dim;
    (void)out;
    (void)out_dim;
    return false;
#endif
}

static uint64_t rt_read_le64_mem(const uint8_t *p) {
    return (uint64_t)p[0] |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}

bool rt_gguf_file_array_cursor(const rt_gguf_file *file, const char *key, rt_gguf_array_cursor *out) {
    if (!file || !key || !out) return false;

    rt_gguf_array_ref array = {0};
    if (!rt_gguf_metadata_get_array(&file->meta, key, &array)) return false;
    if (!rt_gguf_file_span(file, array.data_offset, 0, NULL)) return false;

    out->file = file;
    out->array = array;
    out->offset = array.data_offset;
    out->index = 0;
    return true;
}

bool rt_gguf_array_cursor_next_string(rt_gguf_array_cursor *cursor, const char **ptr, size_t *len) {
    if (!cursor || cursor->array.type != RT_GGUF_VALUE_STRING ||
        cursor->index >= cursor->array.len) {
        return false;
    }

    const uint8_t *p = NULL;
    if (!rt_gguf_file_span(cursor->file, cursor->offset, sizeof(uint64_t), &p)) return false;
    uint64_t n = rt_read_le64_mem(p);
    if (n > SIZE_MAX) return false;

    const uint8_t *s = NULL;
    if (!rt_gguf_file_span(cursor->file, cursor->offset + sizeof(uint64_t), n, &s)) return false;
    cursor->offset += sizeof(uint64_t) + n;
    cursor->index++;

    if (ptr) *ptr = (const char *)s;
    if (len) *len = (size_t)n;
    return true;
}

bool rt_gguf_file_array_string_at(const rt_gguf_file *file,
                                  const char *key,
                                  uint64_t index,
                                  const char **ptr,
                                  size_t *len) {
    rt_gguf_array_cursor cursor = {0};
    if (!rt_gguf_file_array_cursor(file, key, &cursor) ||
        cursor.array.type != RT_GGUF_VALUE_STRING ||
        index >= cursor.array.len) {
        return false;
    }

    const char *s = NULL;
    size_t n = 0;
    for (uint64_t i = 0; i <= index; i++) {
        if (!rt_gguf_array_cursor_next_string(&cursor, &s, &n)) return false;
    }
    if (ptr) *ptr = s;
    if (len) *len = n;
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

int rt_session_eval_speculative_argmax(rt_session *session,
                                       int first_token,
                                       int max_tokens,
                                       int eos_token,
                                       int *out_tokens,
                                       int out_cap,
                                       char *err,
                                       size_t errlen) {
    if (!session || !out_tokens || out_cap <= 0 || max_tokens <= 0) return -1;
    if (!session->engine->ops->session_eval_speculative_argmax) return 0;
    return session->engine->ops->session_eval_speculative_argmax(
        session->impl, first_token, max_tokens, eos_token,
        out_tokens, out_cap, err, errlen);
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

const rt_tokens *rt_session_tokens(rt_session *session) {
    if (!session || !session->engine->ops->session_tokens) return NULL;
    return session->engine->ops->session_tokens(session->impl);
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
