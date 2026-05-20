#include "ds4.h"
#include "ds4_kvstore.h"

/* Native Sloppy engine protocol endpoint.
 *
 * This binary keeps HTTP/OpenAI/Anthropic policy out of the inference loop.  A
 * client sends already-rendered DS4 prompt text over a Unix-socket NDJSON
 * protocol; this process owns the loaded ds4_engine and mutable ds4_session KV
 * timelines.  The socket reader stays separate from the inference worker so a
 * session.interrupt request can be handled while generation is streaming.
 */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop_requested = 0;
static volatile sig_atomic_t g_listen_fd = -1;
static const char *g_socket_path = NULL;
static bool g_socket_created = false;

#define DS4_ENGINE_SEND_STALL_TIMEOUT_MS 2000
#define DS4_ENGINE_DSML_BAR "\xef\xbd\x9c"

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf;

typedef struct {
    ds4_engine_options engine;
    const char *socket_path;
    const char *trace_path;
    const char *kv_disk_dir;
    uint64_t kv_disk_space_mb;
    ds4_kvstore_options kv_cache;
    bool kv_cache_reject_different_quant;
    int ctx_size;
    int max_sessions;
} engine_config;

typedef struct client_conn {
    int fd;
    int refs;
    pthread_mutex_t mu;
    pthread_mutex_t write_mu;
} client_conn;

typedef struct {
    char *id;
    ds4_session *session;
    ds4_kvstore kv;
    bool busy;
    bool interrupt;
    int last_sync_evaluated;
    double last_sync_seconds;
} engine_session_slot;

typedef enum {
    JOB_SESSION_CREATE,
    JOB_SESSION_SYNC,
    JOB_SESSION_GENERATE,
    JOB_SESSION_DESTROY,
} engine_job_kind;

typedef struct engine_job {
    engine_job_kind kind;
    client_conn *client;
    char *request_id;
    char *params_json;
    char *session_id;
    char *requested_session_id;
    char *prefix_text;
    int max_tokens;
    float temperature;
    float top_p;
    float min_p;
    int top_k;
    uint64_t seed;
    bool has_seed;
    struct engine_job *next;
} engine_job;

typedef struct {
    ds4_engine *engine;
    engine_config cfg;
    engine_session_slot *sessions;
    uint64_t next_session_id;
    pthread_mutex_t mu;
    pthread_cond_t cond;
    pthread_mutex_t trace_mu;
    engine_job *job_head;
    engine_job *job_tail;
    bool stop;
    FILE *trace;
    uint64_t trace_seq;
} engine_server;

typedef struct {
    char *id;
    char *method;
    char *params_json;
} engine_request;

typedef struct {
    char *session_id;
    char *prefix_text;
} sync_params;

typedef struct {
    char *session_id;
    int max_tokens;
    float temperature;
    float top_p;
    float min_p;
    int top_k;
    uint64_t seed;
    bool has_seed;
    bool has_stop;
} generate_params;

typedef struct {
    buf text;
    bool dsml_seen;
} generate_stop_detector;

typedef struct {
    engine_server *server;
    engine_session_slot *slot;
    ds4_session *session;
    client_conn *client;
    const char *request_id;
    int cached_tokens;
    int total_eval_tokens;
    int last_current;
    char ctx[48];
    const char *phase;
    double t0;
    double last_t;
    bool seen;
} sync_progress;

static void stop_signal_handler(int sig) {
    (void)sig;
    if (g_stop_requested) _exit(130);
    g_stop_requested = 1;
    if (g_listen_fd >= 0) {
        int fd = (int)g_listen_fd;
        g_listen_fd = -1;
        close(fd);
    }
}

static void die(const char *msg) {
    fprintf(stderr, "ds4-engine: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static void buf_reserve(buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) die("buffer overflow");
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) die("buffer overflow");
        cap *= 2;
    }
    b->ptr = xrealloc(b->ptr, cap);
    b->cap = cap;
}

static void buf_append(buf *b, const void *p, size_t n) {
    buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void buf_putc(buf *b, char c) {
    buf_append(b, &c, 1);
}

static void buf_puts(buf *b, const char *s) {
    buf_append(b, s, strlen(s));
}

static void buf_printf(buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) die("vsnprintf failed");
    buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static void buf_free(buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

static bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static void json_put_utf8(buf *b, uint32_t cp) {
    if (cp <= 0x7f) {
        buf_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        buf_putc(b, (char)(0xc0 | (cp >> 6)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        buf_putc(b, (char)(0xe0 | (cp >> 12)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        buf_putc(b, (char)(0xf0 | (cp >> 18)));
        buf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static bool json_u16(const char **p, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[i]);
        if (h < 0) return false;
        v = (v << 4) | (uint32_t)h;
    }
    *p += 4;
    *out = v;
    return true;
}

static bool json_string(const char **p, char **out) {
    if (**p != '"') return false;
    (*p)++;
    buf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*((*p)++);
        if (c != '\\') {
            if (c < 0x20) goto fail;
            buf_putc(&b, (char)c);
            continue;
        }
        c = (unsigned char)*((*p)++);
        switch (c) {
        case '"': buf_putc(&b, '"'); break;
        case '\\': buf_putc(&b, '\\'); break;
        case '/': buf_putc(&b, '/'); break;
        case 'b': buf_putc(&b, '\b'); break;
        case 'f': buf_putc(&b, '\f'); break;
        case 'n': buf_putc(&b, '\n'); break;
        case 'r': buf_putc(&b, '\r'); break;
        case 't': buf_putc(&b, '\t'); break;
        case 'u': {
            uint32_t cp = 0;
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                if ((*p)[0] != '\\' || (*p)[1] != 'u') goto fail;
                *p += 2;
                uint32_t lo = 0;
                if (!json_u16(p, &lo) || lo < 0xdc00 || lo > 0xdfff) goto fail;
                cp = 0x10000 + (((cp - 0xd800) << 10) | (lo - 0xdc00));
            }
            json_put_utf8(&b, cp);
            break;
        }
        default:
            goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    *out = b.ptr ? b.ptr : xstrdup("");
    return true;
fail:
    buf_free(&b);
    return false;
}

static bool json_number(const char **p, double *out) {
    char *end = NULL;
    errno = 0;
    double v = strtod(*p, &end);
    if (end == *p || errno == ERANGE || !isfinite(v)) return false;
    *p = end;
    *out = v;
    return true;
}

static bool json_skip_value_depth(const char **p, int depth);

static bool json_skip_array_depth(const char **p, int depth) {
    if (depth > 64 || **p != '[') return false;
    (*p)++;
    json_ws(p);
    if (**p == ']') {
        (*p)++;
        return true;
    }
    while (**p) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            json_ws(p);
            continue;
        }
        if (**p == ']') {
            (*p)++;
            return true;
        }
        return false;
    }
    return false;
}

static bool json_skip_object_depth(const char **p, int depth) {
    if (depth > 64 || **p != '{') return false;
    (*p)++;
    json_ws(p);
    if (**p == '}') {
        (*p)++;
        return true;
    }
    while (**p) {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        json_ws(p);
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            json_ws(p);
            continue;
        }
        if (**p == '}') {
            (*p)++;
            return true;
        }
        return false;
    }
    return false;
}

static bool json_skip_value_depth(const char **p, int depth) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null")) return true;
    double v = 0.0;
    return json_number(p, &v);
}

static bool json_skip_value(const char **p) {
    return json_skip_value_depth(p, 0);
}

static bool json_raw_value(const char **p, char **out) {
    json_ws(p);
    const char *start = *p;
    if (!json_skip_value(p)) return false;
    *out = xstrndup(start, (size_t)(*p - start));
    return true;
}

static void json_escape_n(buf *b, const char *s, size_t n) {
    buf_putc(b, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if (c == '\r') {
            buf_puts(b, "\\r");
        } else if (c == '\t') {
            buf_puts(b, "\\t");
        } else if (c < 0x20) {
            buf_printf(b, "\\u%04x", c);
        } else {
            buf_putc(b, (char)c);
        }
    }
    buf_putc(b, '"');
}

static void json_escape(buf *b, const char *s) {
    if (!s) s = "";
    json_escape_n(b, s, strlen(s));
}

static long long wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void engine_log(ds4_log_type type, const char *fmt, ...) {
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char ts[16];
    strftime(ts, sizeof(ts), "%m%d %H:%M:%S", &tm);

    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int n = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);

    fprintf(stderr, "%s ", ts);
    if (n < 0) {
        ds4_log(stderr, type, "%s", fmt);
    } else {
        char *line = xmalloc((size_t)n + 1);
        vsnprintf(line, (size_t)n + 1, fmt, ap);
        ds4_log(stderr, type, "%s", line);
        free(line);
    }
    va_end(ap);
    fputc('\n', stderr);
}

static void request_ctx_span(char *buf, size_t len, int cached, int prompt) {
    int suffix = prompt - cached;
    if (suffix < 0) suffix = 0;
    snprintf(buf, len, "%d..%d:%d", cached, prompt, suffix);
}

static void log_flags(char *buf, size_t len, bool dsml_start, bool dsml_end) {
    size_t used = 0;
    buf[0] = '\0';
#define ADD_FLAG(name) do { \
    int n = snprintf(buf + used, used < len ? len - used : 0, "%s%s", used ? " " : "", name); \
    if (n > 0) used += (size_t)n; \
} while (0)
    if (dsml_start) ADD_FLAG("DSML_START");
    if (dsml_end) ADD_FLAG("DSML_END");
#undef ADD_FLAG
}

static void log_generate_progress(const char *session_id,
                                  int start_pos,
                                  int generated,
                                  bool dsml_start,
                                  bool dsml_end,
                                  double decode_t0,
                                  double *last_t,
                                  int *last_generated) {
    const double now = now_sec();
    const double elapsed = now - decode_t0;
    const double interval_s = now - *last_t;
    const int interval_tokens = generated - *last_generated;
    const double chunk_tps = interval_s > 0.0 ? (double)interval_tokens / interval_s : 0.0;
    const double avg_tps = elapsed > 0.0 ? (double)generated / elapsed : 0.0;
    char ctx[48];
    request_ctx_span(ctx, sizeof(ctx), start_pos + *last_generated,
                     start_pos + generated);
    char flags[64];
    log_flags(flags, sizeof(flags), dsml_start, dsml_end);
    engine_log(DS4_LOG_GENERATION,
               "ds4-engine: generate session=%s ctx=%s gen=%d%s%s decoding chunk=%.2f t/s avg=%.2f t/s %.3fs",
               session_id ? session_id : "?",
               ctx,
               generated,
               flags[0] ? " " : "",
               flags,
               chunk_tps,
               avg_tps,
               elapsed);
    *last_t = now;
    *last_generated = generated;
}

static bool send_all_fd(int fd, const void *p, size_t n) {
    const char *s = p;
    long long deadline = wall_ms() + DS4_ENGINE_SEND_STALL_TIMEOUT_MS;
    while (n) {
        if (g_stop_requested) return false;
#ifdef MSG_NOSIGNAL
        ssize_t w = send(fd, s, n, MSG_NOSIGNAL);
#else
        ssize_t w = send(fd, s, n, 0);
#endif
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            long long remaining = deadline - wall_ms();
            if (remaining <= 0) return false;
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            int timeout = remaining > 50 ? 50 : (int)remaining;
            int rc;
            do {
                rc = poll(&pfd, 1, timeout);
            } while (rc < 0 && errno == EINTR);
            if (rc < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) return false;
            continue;
        }
        if (w <= 0) return false;
        s += w;
        n -= (size_t)w;
        deadline = wall_ms() + DS4_ENGINE_SEND_STALL_TIMEOUT_MS;
    }
    return true;
}

static client_conn *client_create(int fd) {
    client_conn *c = xmalloc(sizeof(*c));
    c->fd = fd;
    c->refs = 1;
    pthread_mutex_init(&c->mu, NULL);
    pthread_mutex_init(&c->write_mu, NULL);
    return c;
}

static void client_ref(client_conn *c) {
    pthread_mutex_lock(&c->mu);
    c->refs++;
    pthread_mutex_unlock(&c->mu);
}

static void client_unref(client_conn *c) {
    bool destroy = false;
    pthread_mutex_lock(&c->mu);
    c->refs--;
    if (c->refs == 0) destroy = true;
    pthread_mutex_unlock(&c->mu);
    if (!destroy) return;
    close(c->fd);
    pthread_mutex_destroy(&c->write_mu);
    pthread_mutex_destroy(&c->mu);
    free(c);
}

static bool client_write(client_conn *c, const char *p, size_t n) {
    pthread_mutex_lock(&c->write_mu);
    bool ok = send_all_fd(c->fd, p, n);
    pthread_mutex_unlock(&c->write_mu);
    return ok;
}

static bool client_send_buf(client_conn *c, buf *b) {
    return client_write(c, b->ptr ? b->ptr : "", b->len);
}

static bool send_response_ok(client_conn *c, const char *id, const char *result_json) {
    buf b = {0};
    buf_puts(&b, "{\"type\":\"response\",\"id\":");
    json_escape(&b, id);
    buf_puts(&b, ",\"ok\":true");
    if (result_json) {
        buf_puts(&b, ",\"result\":");
        buf_puts(&b, result_json);
    }
    buf_puts(&b, "}\n");
    bool ok = client_send_buf(c, &b);
    buf_free(&b);
    return ok;
}

static bool send_response_error(client_conn *c, const char *id,
                                const char *code, const char *message,
                                bool retryable) {
    buf b = {0};
    buf_puts(&b, "{\"type\":\"response\",\"id\":");
    json_escape(&b, id);
    buf_puts(&b, ",\"ok\":false,\"error\":{\"code\":");
    json_escape(&b, code);
    buf_puts(&b, ",\"message\":");
    json_escape(&b, message);
    if (retryable) buf_puts(&b, ",\"retryable\":true");
    buf_puts(&b, "}}\n");
    bool ok = client_send_buf(c, &b);
    buf_free(&b);
    return ok;
}

static bool send_event(client_conn *c, const char *id, const char *event_json) {
    buf b = {0};
    buf_puts(&b, "{\"type\":\"event\",\"id\":");
    json_escape(&b, id);
    buf_puts(&b, ",\"event\":");
    buf_puts(&b, event_json);
    buf_puts(&b, "}\n");
    bool ok = client_send_buf(c, &b);
    buf_free(&b);
    return ok;
}

static void trace_log(engine_server *s, const char *fmt, ...) {
    if (!s || !s->trace) return;
    pthread_mutex_lock(&s->trace_mu);
    double t = now_sec();
    fprintf(s->trace, "%.6f ", t);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(s->trace, fmt, ap);
    va_end(ap);
    fputc('\n', s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

#define TRACE_CACHE_BEFORE 8
#define TRACE_CACHE_AFTER  8
#define TRACE_CACHE_WINDOW (TRACE_CACHE_BEFORE + 1 + TRACE_CACHE_AFTER)

typedef struct {
    bool valid;
    int old_pos;
    int prompt_len;
    int common;
    int start;
    int count;
    int live_id[TRACE_CACHE_WINDOW];
    int prompt_id[TRACE_CACHE_WINDOW];
} trace_cache_diag;

static void trace_cache_capture(trace_cache_diag *d,
                                const ds4_tokens *live,
                                const ds4_tokens *prompt,
                                int old_pos,
                                int common) {
    memset(d, 0, sizeof(*d));
    d->valid = true;
    d->old_pos = old_pos;
    d->prompt_len = prompt ? prompt->len : 0;
    d->common = common;

    const int live_len = live ? live->len : 0;
    const int prompt_len = prompt ? prompt->len : 0;
    int max_len = live_len > prompt_len ? live_len : prompt_len;
    int start = common - TRACE_CACHE_BEFORE;
    if (start < 0) start = 0;
    int end = common + TRACE_CACHE_AFTER + 1;
    if (end > max_len) end = max_len;
    if (end < start) end = start;

    d->start = start;
    d->count = end - start;
    if (d->count > TRACE_CACHE_WINDOW) d->count = TRACE_CACHE_WINDOW;
    for (int i = 0; i < d->count; i++) {
        int pos = start + i;
        d->live_id[i] = live && pos < live->len ? live->v[pos] : -1;
        d->prompt_id[i] = prompt && pos < prompt->len ? prompt->v[pos] : -1;
    }
}

static const char *trace_cache_miss_reason(const trace_cache_diag *d) {
    if (!d || !d->valid) return "unknown";
    if (d->old_pos == 0) return "no-live-checkpoint";
    if (d->common != d->old_pos) return "token-mismatch";
    if (d->prompt_len < d->old_pos) return "incoming-prompt-shorter-than-live-checkpoint";
    return "live-prefix-match";
}

static void trace_write_escaped_bytes(FILE *fp, const char *p, size_t len) {
    static const char hex[] = "0123456789abcdef";
    fputc('"', fp);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)p[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20 || c == 0x7f) {
            fputs("\\x", fp);
            fputc(hex[c >> 4], fp);
            fputc(hex[c & 15], fp);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void trace_write_token(FILE *fp, ds4_engine *engine, int token) {
    if (token < 0) {
        fputs("- <none>", fp);
        return;
    }
    size_t len = 0;
    char *piece = ds4_token_text(engine, token, &len);
    fprintf(fp, "%d ", token);
    trace_write_escaped_bytes(fp, piece, len);
    free(piece);
}

static void trace_write_cache_diag(engine_server *s,
                                   const trace_cache_diag *d,
                                   int cached,
                                   const char *cache_source,
                                   int disk_cached,
                                   const char *disk_path) {
    fprintf(s->trace,
            "\n--- cache decision ---\n"
            "live_tokens_before: %d\n"
            "prompt_tokens: %d\n"
            "live_prompt_common: %d\n"
            "memory_token_reusable: %d\n"
            "memory_miss_reason: %s\n"
            "cache_source: %s\n"
            "cached_tokens: %d\n"
            "disk_cached_tokens: %d\n",
            d && d->valid ? d->old_pos : 0,
            d && d->valid ? d->prompt_len : 0,
            d && d->valid ? d->common : 0,
            d && d->valid && d->old_pos > 0 &&
                d->common == d->old_pos && d->prompt_len >= d->old_pos ? 1 : 0,
            trace_cache_miss_reason(d),
            cache_source ? cache_source : "none",
            cached,
            disk_cached);
    if (disk_path && disk_path[0]) fprintf(s->trace, "disk_cache_file: %s\n", disk_path);

    if (!d || !d->valid || d->old_pos == 0 ||
        (d->common == d->old_pos && d->prompt_len >= d->old_pos))
    {
        return;
    }

    fprintf(s->trace,
            "\nfirst_mismatch_token: %d\n"
            "token_window: [%d..%d)\n",
            d->common,
            d->start,
            d->start + d->count);
    for (int i = 0; i < d->count; i++) {
        int pos = d->start + i;
        int live = d->live_id[i];
        int prompt = d->prompt_id[i];
        const char *mark;
        if (live < 0) mark = "prompt-only";
        else if (prompt < 0) mark = "live-only";
        else mark = live == prompt ? "==" : "!=";

        fprintf(s->trace, "%7d %-11s live ", pos, mark);
        trace_write_token(s->trace, s->engine, live);
        fputs(" | prompt ", s->trace);
        trace_write_token(s->trace, s->engine, prompt);
        fputc('\n', s->trace);
    }
}

static void trace_time(FILE *fp) {
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char b[32];
    strftime(b, sizeof(b), "%Y-%m-%d %H:%M:%S", &tm);
    fputs(b, fp);
}

static uint64_t trace_sync_begin(engine_server *s,
                                 const engine_job *j,
                                 const ds4_tokens *prompt,
                                 const ds4_tokens *effective_prompt,
                                 int cached,
                                 int evaluated,
                                 const trace_cache_diag *cache_diag,
                                 const char *cache_source,
                                 int disk_cached,
                                 const char *disk_path) {
    if (!s || !s->trace) return 0;

    pthread_mutex_lock(&s->trace_mu);
    uint64_t id = ++s->trace_seq;
    fprintf(s->trace, "\n===== request %llu ", (unsigned long long)id);
    trace_time(s->trace);
    fprintf(s->trace,
            " =====\nmethod: session.sync\nrequest_id: %s\nsession: %s\nmodel: %s\nbackend: %s\nctx: %d\nprompt_tokens: %d\neffective_prompt_tokens: %d\ncached_tokens: %d\nevaluated_tokens: %d\n",
            j->request_id ? j->request_id : "",
            j->session_id ? j->session_id : "",
            s->cfg.engine.model_path ? s->cfg.engine.model_path : "",
            ds4_backend_name(s->cfg.engine.backend),
            s->cfg.ctx_size,
            prompt ? prompt->len : 0,
            effective_prompt ? effective_prompt->len : 0,
            cached,
            evaluated);
    trace_write_cache_diag(s, cache_diag, cached, cache_source, disk_cached, disk_path);
    if (j->params_json) {
        fputs("\n--- params json ---\n", s->trace);
        fputs(j->params_json, s->trace);
        if (!j->params_json[0] || j->params_json[strlen(j->params_json) - 1] != '\n') {
            fputc('\n', s->trace);
        }
    }
    if (j->prefix_text) {
        fputs("\n--- rendered prompt ---\n", s->trace);
        fputs(j->prefix_text, s->trace);
        if (!j->prefix_text[0] || j->prefix_text[strlen(j->prefix_text) - 1] != '\n') {
            fputc('\n', s->trace);
        }
    }
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
    return id;
}

static uint64_t trace_generate_begin(engine_server *s,
                                     const engine_job *j,
                                     int start_pos,
                                     int max_tokens,
                                     uint64_t seed) {
    if (!s || !s->trace) return 0;

    pthread_mutex_lock(&s->trace_mu);
    uint64_t id = ++s->trace_seq;
    fprintf(s->trace, "\n===== request %llu ", (unsigned long long)id);
    trace_time(s->trace);
    fprintf(s->trace,
            " =====\nmethod: session.generate\nrequest_id: %s\nsession: %s\nmodel: %s\nbackend: %s\nctx: %d\nstart_position: %d\nmax_tokens: %d\ntemperature: %.3f\ntop_k: %d\ntop_p: %.3f\nmin_p: %.3f\nseed: %s%llu\n",
            j->request_id ? j->request_id : "",
            j->session_id ? j->session_id : "",
            s->cfg.engine.model_path ? s->cfg.engine.model_path : "",
            ds4_backend_name(s->cfg.engine.backend),
            s->cfg.ctx_size,
            start_pos,
            max_tokens,
            j->temperature,
            j->top_k,
            j->top_p,
            j->min_p,
            j->has_seed ? "" : "auto:",
            (unsigned long long)seed);
    if (j->params_json) {
        fputs("\n--- params json ---\n", s->trace);
        fputs(j->params_json, s->trace);
        if (!j->params_json[0] || j->params_json[strlen(j->params_json) - 1] != '\n') {
            fputc('\n', s->trace);
        }
    }
    fputs("\n--- generated text ---\n", s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
    return id;
}

static void trace_piece(engine_server *s, uint64_t id, const char *piece, size_t len) {
    if (!s || !s->trace || !id || !piece || !len) return;
    pthread_mutex_lock(&s->trace_mu);
    fwrite(piece, 1, len, s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

static void trace_event(engine_server *s, uint64_t id, const char *fmt, ...) {
    if (!s || !s->trace || !id) return;
    pthread_mutex_lock(&s->trace_mu);
    fputs("\n\n--- trace: ", s->trace);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(s->trace, fmt, ap);
    va_end(ap);
    fputs(" ---\n\n", s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

static void trace_sync_finish(engine_server *s,
                              uint64_t id,
                              int position,
                              int cached,
                              int evaluated,
                              bool rebuilt,
                              const char *status,
                              const char *err,
                              double elapsed) {
    if (!s || !s->trace || !id) return;
    pthread_mutex_lock(&s->trace_mu);
    fprintf(s->trace,
            "\n--- sync summary ---\nstatus: %s\nposition: %d\ncached_tokens: %d\nevaluated_tokens: %d\nrebuilt: %d\nelapsed_sec: %.3f\n",
            status ? status : "ok",
            position,
            cached,
            evaluated,
            rebuilt ? 1 : 0,
            elapsed);
    if (err && err[0]) fprintf(s->trace, "error: %s\n", err);
    fprintf(s->trace, "\n===== end request %llu =====\n", (unsigned long long)id);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

static void trace_generate_finish(engine_server *s,
                                  uint64_t id,
                                  const char *reason,
                                  int generated,
                                  int position,
                                  bool dsml_start,
                                  bool dsml_end,
                                  bool invalidated,
                                  double elapsed) {
    if (!s || !s->trace || !id) return;
    pthread_mutex_lock(&s->trace_mu);
    fprintf(s->trace,
            "\n\n--- generation summary ---\nfinish: %s\ngenerated_tokens: %d\nposition: %d\ndsml_start: %d\ndsml_end: %d\ninvalidated: %d\nelapsed_sec: %.3f\n",
            reason ? reason : "",
            generated,
            position,
            dsml_start ? 1 : 0,
            dsml_end ? 1 : 0,
            invalidated ? 1 : 0,
            elapsed);
    fprintf(s->trace, "\n===== end request %llu =====\n", (unsigned long long)id);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

static void kv_log_cb(void *ud, ds4_kvstore_log_type type, const char *msg) {
    engine_server *s = ud;
    const char *level = "kvcache";
    if (type == DS4_KVSTORE_LOG_DEFAULT) level = "info";
    else if (type == DS4_KVSTORE_LOG_WARNING) level = "warning";
    ds4_log_type log_type = DS4_LOG_KVCACHE;
    if (type == DS4_KVSTORE_LOG_DEFAULT) log_type = DS4_LOG_DEFAULT;
    else if (type == DS4_KVSTORE_LOG_WARNING) log_type = DS4_LOG_WARNING;
    engine_log(log_type, "%s", msg ? msg : "");
    trace_log(s, "%s %s", level, msg ? msg : "");
}

static void engine_request_free(engine_request *r) {
    free(r->id);
    free(r->method);
    free(r->params_json);
    memset(r, 0, sizeof(*r));
}

static bool parse_engine_request(const char *line, engine_request *out,
                                 char *err, size_t errlen) {
    const char *p = line;
    bool saw_type = false;
    bool type_ok = false;
    memset(out, 0, sizeof(*out));
    json_ws(&p);
    if (*p != '{') {
        snprintf(err, errlen, "request must be a JSON object");
        return false;
    }
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto invalid;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto invalid;
        }
        p++;
        json_ws(&p);
        if (!strcmp(key, "type")) {
            char *type = NULL;
            if (!json_string(&p, &type)) {
                free(key);
                goto invalid;
            }
            saw_type = true;
            type_ok = !strcmp(type, "request");
            free(type);
        } else if (!strcmp(key, "id")) {
            free(out->id);
            if (!json_string(&p, &out->id)) {
                free(key);
                goto invalid;
            }
        } else if (!strcmp(key, "method")) {
            free(out->method);
            if (!json_string(&p, &out->method)) {
                free(key);
                goto invalid;
            }
        } else if (!strcmp(key, "params")) {
            free(out->params_json);
            if (!json_raw_value(&p, &out->params_json)) {
                free(key);
                goto invalid;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto invalid;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') {
            p++;
            json_ws(&p);
            continue;
        }
        if (*p != '}') goto invalid;
    }
    if (*p != '}') goto invalid;
    p++;
    json_ws(&p);
    if (*p != '\0') goto invalid;
    if (!saw_type || !type_ok) {
        snprintf(err, errlen, "request type must be \"request\"");
        goto fail;
    }
    if (!out->id || !out->method) {
        snprintf(err, errlen, "request requires id and method");
        goto fail;
    }
    return true;
invalid:
    snprintf(err, errlen, "invalid JSON request");
fail:
    engine_request_free(out);
    return false;
}

static bool json_object_optional_string(const char *json, const char *field, char **out,
                                        char *err, size_t errlen) {
    *out = NULL;
    if (!json) return true;
    const char *p = json;
    json_ws(&p);
    if (*p != '{') {
        snprintf(err, errlen, "params must be an object");
        return false;
    }
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto invalid;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto invalid;
        }
        p++;
        json_ws(&p);
        if (!strcmp(key, field)) {
            free(*out);
            if (!json_string(&p, out)) {
                free(key);
                goto invalid;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto invalid;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') {
            p++;
            json_ws(&p);
            continue;
        }
        if (*p != '}') goto invalid;
    }
    if (*p != '}') goto invalid;
    return true;
invalid:
    snprintf(err, errlen, "invalid params");
    free(*out);
    *out = NULL;
    return false;
}

static bool parse_session_id_params(const char *json, char **session_id,
                                    char *err, size_t errlen) {
    if (!json_object_optional_string(json, "sessionId", session_id, err, errlen))
        return false;
    if (!*session_id || !(*session_id)[0]) {
        snprintf(err, errlen, "missing sessionId");
        free(*session_id);
        *session_id = NULL;
        return false;
    }
    return true;
}

static bool parse_sync_prefix(const char **p, char **text, char *err, size_t errlen) {
    char *kind = NULL;
    *text = NULL;
    if (**p != '{') goto invalid;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) goto invalid;
        json_ws(p);
        if (**p != ':') {
            free(key);
            goto invalid;
        }
        (*p)++;
        json_ws(p);
        if (!strcmp(key, "kind")) {
            free(kind);
            if (!json_string(p, &kind)) {
                free(key);
                goto invalid;
            }
        } else if (!strcmp(key, "text")) {
            free(*text);
            if (!json_string(p, text)) {
                free(key);
                goto invalid;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            goto invalid;
        }
        free(key);
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            json_ws(p);
            continue;
        }
        if (**p != '}') goto invalid;
    }
    if (**p != '}') goto invalid;
    (*p)++;
    if (!kind || strcmp(kind, "rendered_text") != 0) {
        snprintf(err, errlen, "session.sync only supports rendered_text input");
        free(kind);
        free(*text);
        *text = NULL;
        return false;
    }
    free(kind);
    if (!*text) {
        snprintf(err, errlen, "session.sync prefix.text is required");
        return false;
    }
    return true;
invalid:
    snprintf(err, errlen, "invalid session.sync prefix");
    free(kind);
    free(*text);
    *text = NULL;
    return false;
}

static bool parse_sync_params(const char *json, sync_params *out,
                              char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    if (!json) {
        snprintf(err, errlen, "missing params");
        return false;
    }
    const char *p = json;
    json_ws(&p);
    if (*p != '{') goto invalid;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto invalid;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto invalid;
        }
        p++;
        json_ws(&p);
        if (!strcmp(key, "sessionId")) {
            free(out->session_id);
            if (!json_string(&p, &out->session_id)) {
                free(key);
                goto invalid;
            }
        } else if (!strcmp(key, "prefix")) {
            if (!parse_sync_prefix(&p, &out->prefix_text, err, errlen)) {
                free(key);
                goto fail;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto invalid;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') {
            p++;
            json_ws(&p);
            continue;
        }
        if (*p != '}') goto invalid;
    }
    if (*p != '}') goto invalid;
    if (!out->session_id || !out->session_id[0]) {
        snprintf(err, errlen, "missing sessionId");
        goto fail;
    }
    if (!out->prefix_text) {
        snprintf(err, errlen, "missing prefix");
        goto fail;
    }
    return true;
invalid:
    snprintf(err, errlen, "invalid session.sync params");
fail:
    free(out->session_id);
    free(out->prefix_text);
    memset(out, 0, sizeof(*out));
    return false;
}

static bool parse_json_int_value(const char **p, int *out) {
    double v = 0.0;
    if (!json_number(p, &v)) return false;
    if (v < 0.0 || v > (double)INT_MAX || floor(v) != v) return false;
    *out = (int)v;
    return true;
}

static bool parse_json_u64_value(const char **p, uint64_t *out) {
    double v = 0.0;
    if (!json_number(p, &v)) return false;
    if (v <= 0.0 || v > 18446744073709551615.0 || floor(v) != v) return false;
    *out = (uint64_t)v;
    return true;
}

static bool parse_stop_presence(const char **p, bool *has_stop) {
    json_ws(p);
    if (**p == '[') {
        (*p)++;
        json_ws(p);
        if (**p != ']') *has_stop = true;
        while (**p && **p != ']') {
            if (!json_skip_value(p)) return false;
            json_ws(p);
            if (**p == ',') {
                (*p)++;
                json_ws(p);
                continue;
            }
            if (**p != ']') return false;
        }
        if (**p != ']') return false;
        (*p)++;
        return true;
    }
    if (json_lit(p, "null")) return true;
    *has_stop = true;
    return json_skip_value(p);
}

static bool parse_generate_options(const char **p, generate_params *out,
                                   char *err, size_t errlen) {
    bool have_max = false;
    if (**p != '{') goto invalid;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) goto invalid;
        json_ws(p);
        if (**p != ':') {
            free(key);
            goto invalid;
        }
        (*p)++;
        json_ws(p);
        if (!strcmp(key, "maxTokens")) {
            if (!parse_json_int_value(p, &out->max_tokens)) {
                free(key);
                goto invalid;
            }
            have_max = true;
        } else if (!strcmp(key, "temperature")) {
            double v = 0.0;
            if (!json_number(p, &v) || v < 0.0 || v > 100.0) {
                free(key);
                goto invalid;
            }
            out->temperature = (float)v;
        } else if (!strcmp(key, "topP")) {
            double v = 0.0;
            if (!json_number(p, &v) || v < 0.0 || v > 1.0) {
                free(key);
                goto invalid;
            }
            out->top_p = (float)v;
        } else if (!strcmp(key, "minP")) {
            double v = 0.0;
            if (!json_number(p, &v) || v < 0.0 || v > 1.0) {
                free(key);
                goto invalid;
            }
            out->min_p = (float)v;
        } else if (!strcmp(key, "topK")) {
            if (!parse_json_int_value(p, &out->top_k)) {
                free(key);
                goto invalid;
            }
        } else if (!strcmp(key, "seed")) {
            if (!parse_json_u64_value(p, &out->seed)) {
                free(key);
                goto invalid;
            }
            out->has_seed = true;
        } else if (!strcmp(key, "stop")) {
            if (!parse_stop_presence(p, &out->has_stop)) {
                free(key);
                goto invalid;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            goto invalid;
        }
        free(key);
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            json_ws(p);
            continue;
        }
        if (**p != '}') goto invalid;
    }
    if (**p != '}') goto invalid;
    (*p)++;
    if (!have_max) {
        snprintf(err, errlen, "session.generate options.maxTokens is required");
        return false;
    }
    return true;
invalid:
    snprintf(err, errlen, "invalid session.generate options");
    return false;
}

static bool parse_generate_params(const char *json, generate_params *out,
                                  char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    out->temperature = DS4_DEFAULT_TEMPERATURE;
    out->top_p = DS4_DEFAULT_TOP_P;
    out->min_p = DS4_DEFAULT_MIN_P;
    bool have_options = false;
    if (!json) {
        snprintf(err, errlen, "missing params");
        return false;
    }
    const char *p = json;
    json_ws(&p);
    if (*p != '{') goto invalid;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto invalid;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto invalid;
        }
        p++;
        json_ws(&p);
        if (!strcmp(key, "sessionId")) {
            free(out->session_id);
            if (!json_string(&p, &out->session_id)) {
                free(key);
                goto invalid;
            }
        } else if (!strcmp(key, "options")) {
            if (!parse_generate_options(&p, out, err, errlen)) {
                free(key);
                goto fail;
            }
            have_options = true;
        } else if (!json_skip_value(&p)) {
            free(key);
            goto invalid;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') {
            p++;
            json_ws(&p);
            continue;
        }
        if (*p != '}') goto invalid;
    }
    if (*p != '}') goto invalid;
    if (!out->session_id || !out->session_id[0]) {
        snprintf(err, errlen, "missing sessionId");
        goto fail;
    }
    if (!have_options) {
        snprintf(err, errlen, "missing options");
        goto fail;
    }
    return true;
invalid:
    snprintf(err, errlen, "invalid session.generate params");
fail:
    free(out->session_id);
    memset(out, 0, sizeof(*out));
    return false;
}

static void generate_params_free(generate_params *p) {
    free(p->session_id);
    memset(p, 0, sizeof(*p));
}

static void sync_params_free(sync_params *p) {
    free(p->session_id);
    free(p->prefix_text);
    memset(p, 0, sizeof(*p));
}

static engine_session_slot *find_session_locked(engine_server *s, const char *id) {
    for (int i = 0; i < s->cfg.max_sessions; i++) {
        if (s->sessions[i].session && s->sessions[i].id &&
            !strcmp(s->sessions[i].id, id))
        {
            return &s->sessions[i];
        }
    }
    return NULL;
}

static int session_count_locked(engine_server *s) {
    int n = 0;
    for (int i = 0; i < s->cfg.max_sessions; i++) {
        if (s->sessions[i].session) n++;
    }
    return n;
}

static engine_session_slot *free_session_slot_locked(engine_server *s) {
    for (int i = 0; i < s->cfg.max_sessions; i++) {
        if (!s->sessions[i].session) return &s->sessions[i];
    }
    return NULL;
}

static void job_free(engine_job *j) {
    if (!j) return;
    client_unref(j->client);
    free(j->request_id);
    free(j->params_json);
    free(j->session_id);
    free(j->requested_session_id);
    free(j->prefix_text);
    free(j);
}

static void enqueue_job(engine_server *s, engine_job *j) {
    pthread_mutex_lock(&s->mu);
    if (s->job_tail) s->job_tail->next = j;
    else s->job_head = j;
    s->job_tail = j;
    pthread_cond_signal(&s->cond);
    pthread_mutex_unlock(&s->mu);
}

static engine_job *dequeue_job(engine_server *s) {
    pthread_mutex_lock(&s->mu);
    while (!s->stop && !s->job_head) {
        pthread_cond_wait(&s->cond, &s->mu);
    }
    if (s->stop && !s->job_head) {
        pthread_mutex_unlock(&s->mu);
        return NULL;
    }
    engine_job *j = s->job_head;
    s->job_head = j->next;
    if (!s->job_head) s->job_tail = NULL;
    j->next = NULL;
    pthread_mutex_unlock(&s->mu);
    return j;
}

static void clear_busy_locked(engine_session_slot *slot) {
    if (slot) slot->busy = false;
}

static void job_clear_busy(engine_server *s, const char *session_id) {
    if (!session_id) return;
    pthread_mutex_lock(&s->mu);
    clear_busy_locked(find_session_locked(s, session_id));
    pthread_mutex_unlock(&s->mu);
}

static void sync_progress_cb(void *ud, const char *event, int current, int total) {
    sync_progress *p = ud;
    (void)total;
    if (!p || !event || strcmp(event, "prefill_chunk")) return;
    if (p->slot && p->slot->kv.enabled && p->session &&
        current > p->cached_tokens)
    {
        char err[160] = {0};
        ds4_kvstore_maybe_store_continued(&p->slot->kv, p->server->engine,
                                          p->session, NULL, err, sizeof(err));
    }
    int relative = current - p->cached_tokens;
    if (relative < 0) relative = 0;
    if (relative > p->total_eval_tokens) relative = p->total_eval_tokens;
    if (relative == p->last_current) return;
    const double now = now_sec();
    const double elapsed = now - p->t0;
    const int interval_tokens = p->seen ? relative - p->last_current : relative;
    const double interval_s = p->seen ? now - p->last_t : elapsed;
    const double chunk_tps = interval_s > 0.0 ? (double)interval_tokens / interval_s : 0.0;
    const double avg_tps = elapsed > 0.0 ? (double)relative / elapsed : 0.0;
    const double pct = p->total_eval_tokens > 0 ?
        100.0 * (double)relative / (double)p->total_eval_tokens : 100.0;
    engine_log(DS4_LOG_PREFILL,
               "ds4-engine: sync session=%s ctx=%s %s chunk %d/%d (%.1f%%) chunk=%.2f t/s avg=%.2f t/s %.3fs",
               p->slot && p->slot->id ? p->slot->id : "?",
               p->ctx,
               p->phase ? p->phase : "prefill",
               relative,
               p->total_eval_tokens,
               pct,
               chunk_tps,
               avg_tps,
               elapsed);
    p->last_t = now;
    p->seen = true;
    p->last_current = relative;
    buf e = {0};
    buf_printf(&e, "{\"type\":\"prefill_progress\",\"current\":%d,\"total\":%d}",
               relative, p->total_eval_tokens);
    send_event(p->client, p->request_id, e.ptr);
    buf_free(&e);
}

static void send_final_prefill_progress(sync_progress *p) {
    if (!p || p->last_current == p->total_eval_tokens) return;
    buf e = {0};
    buf_printf(&e, "{\"type\":\"prefill_progress\",\"current\":%d,\"total\":%d}",
               p->total_eval_tokens, p->total_eval_tokens);
    send_event(p->client, p->request_id, e.ptr);
    buf_free(&e);
    p->last_current = p->total_eval_tokens;
}

static bool buffer_contains_literal(const buf *b, const char *needle) {
    size_t needle_len = strlen(needle);
    if (!needle_len || b->len < needle_len) return false;
    for (size_t i = 0; i <= b->len - needle_len; i++) {
        if (memcmp(b->ptr + i, needle, needle_len) == 0) return true;
    }
    return false;
}

static bool buffer_contains_any_literal(const buf *b,
                                        const char *const *needles,
                                        size_t needle_count) {
    for (size_t i = 0; i < needle_count; i++) {
        if (buffer_contains_literal(b, needles[i])) return true;
    }
    return false;
}

static void generate_stop_detector_update(generate_stop_detector *d,
                                          const char *piece,
                                          size_t piece_len) {
    static const char canonical_start[] =
        "<" DS4_ENGINE_DSML_BAR "DSML" DS4_ENGINE_DSML_BAR "tool_calls>";
    static const char missing_bar_start[] =
        "<DSML" DS4_ENGINE_DSML_BAR "tool_calls>";
    static const char plain_start[] = "<tool_calls>";
    static const char *const starts[] = {
        canonical_start,
        missing_bar_start,
        plain_start,
    };

    if (!piece_len) return;
    buf_append(&d->text, piece, piece_len);
    if (!d->dsml_seen) {
        d->dsml_seen = buffer_contains_any_literal(&d->text, starts,
                                                   sizeof(starts) / sizeof(starts[0]));
    }
}

static bool generate_stop_detector_done(const generate_stop_detector *d) {
    static const char canonical_end[] =
        "</" DS4_ENGINE_DSML_BAR "DSML" DS4_ENGINE_DSML_BAR "tool_calls>";
    static const char missing_bar_end[] =
        "</DSML" DS4_ENGINE_DSML_BAR "tool_calls>";
    static const char plain_end[] = "</tool_calls>";
    static const char *const ends[] = {
        canonical_end,
        missing_bar_end,
        plain_end,
    };

    return d->dsml_seen &&
           buffer_contains_any_literal(&d->text, ends, sizeof(ends) / sizeof(ends[0]));
}

static void generate_stop_detector_free(generate_stop_detector *d) {
    buf_free(&d->text);
    d->dsml_seen = false;
}

static bool session_kv_open(engine_server *s, engine_session_slot *slot) {
    memset(&slot->kv, 0, sizeof(slot->kv));
    if (!s->cfg.kv_disk_dir) return true;
    return ds4_kvstore_open(&slot->kv, s->cfg.kv_disk_dir,
                            s->cfg.kv_disk_space_mb,
                            s->cfg.kv_cache_reject_different_quant,
                            s->cfg.kv_cache,
                            "ds4-engine", kv_log_cb, s);
}

static void session_kv_close(engine_session_slot *slot) {
    if (!slot) return;
    ds4_kvstore_close(&slot->kv);
}

static bool session_kv_store_live(engine_server *s, engine_session_slot *slot,
                                  const ds4_tokens *tokens, int store_len,
                                  const char *reason) {
    if (!slot || !slot->kv.enabled || !slot->session) return false;
    char err[160] = {0};
    return ds4_kvstore_store_live_prefix(&slot->kv, s->engine, slot->session,
                                         tokens, store_len, reason, NULL,
                                         err, sizeof(err));
}

static void session_kv_store_current(engine_server *s, engine_session_slot *slot,
                                     const char *reason) {
    if (!slot || !slot->kv.enabled || !slot->session) return;
    const ds4_tokens *tokens = ds4_session_tokens(slot->session);
    if (!tokens || tokens->len < slot->kv.opt.min_tokens) return;
    session_kv_store_live(s, slot, tokens, tokens->len, reason);
}

static int live_text_prefix_prompt(engine_server *s, ds4_session *session,
                                   const char *prefix_text,
                                   ds4_tokens *effective_prompt) {
    if (!s || !session || !prefix_text || !effective_prompt) return 0;
    const ds4_tokens *live_tokens = ds4_session_tokens(session);
    if (!live_tokens || live_tokens->len <= 0) return 0;

    size_t live_text_len = 0;
    char *live_text = ds4_kvstore_render_tokens_text(s->engine, live_tokens,
                                                     &live_text_len);
    const size_t prefix_text_len = strlen(prefix_text);
    if (!ds4_kvstore_byte_prefix_match(prefix_text, prefix_text_len,
                                       live_text, live_text_len))
    {
        free(live_text);
        return 0;
    }

    ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, prefix_text + live_text_len, effective_prompt);
    free(live_text);
    return live_tokens->len;
}

static void worker_create_session(engine_server *s, engine_job *j) {
    pthread_mutex_lock(&s->mu);
    if (session_count_locked(s) >= s->cfg.max_sessions) {
        pthread_mutex_unlock(&s->mu);
        send_response_error(j->client, j->request_id, "busy",
                            "maximum session count reached", false);
        return;
    }
    if (j->requested_session_id &&
        find_session_locked(s, j->requested_session_id))
    {
        pthread_mutex_unlock(&s->mu);
        send_response_error(j->client, j->request_id, "invalid_request",
                            "sessionId already exists", false);
        return;
    }
    pthread_mutex_unlock(&s->mu);

    ds4_session *session = NULL;
    if (ds4_session_create(&session, s->engine, s->cfg.ctx_size) != 0) {
        send_response_error(j->client, j->request_id, "engine_error",
                            "failed to create DS4 session", false);
        return;
    }

    engine_session_slot new_slot;
    memset(&new_slot, 0, sizeof(new_slot));
    new_slot.session = session;
    if (!session_kv_open(s, &new_slot) && s->cfg.kv_disk_dir) {
        trace_log(s, "request=%s method=session.create kv_cache_disabled dir=%s",
                  j->request_id, s->cfg.kv_disk_dir);
    }

    char idbuf[64];
    const char *session_id = j->requested_session_id;
    if (!session_id || !session_id[0]) {
        snprintf(idbuf, sizeof(idbuf), "ds4-session-%llu",
                 (unsigned long long)++s->next_session_id);
        session_id = idbuf;
    }

    pthread_mutex_lock(&s->mu);
    engine_session_slot *slot = free_session_slot_locked(s);
    if (!slot || find_session_locked(s, session_id)) {
        pthread_mutex_unlock(&s->mu);
        session_kv_close(&new_slot);
        ds4_session_free(session);
        send_response_error(j->client, j->request_id, "busy",
                            "maximum session count reached", false);
        return;
    }
    slot->id = xstrdup(session_id);
    slot->session = session;
    slot->kv = new_slot.kv;
    slot->busy = false;
    slot->interrupt = false;
    slot->last_sync_evaluated = 0;
    slot->last_sync_seconds = 0.0;
    pthread_mutex_unlock(&s->mu);

    engine_log(DS4_LOG_DEFAULT,
               "ds4-engine: session.create id=%s ctx=%d",
               session_id, s->cfg.ctx_size);
    trace_log(s, "request=%s method=session.create session=%s ctx=%d",
              j->request_id, session_id, s->cfg.ctx_size);

    buf result = {0};
    buf_puts(&result, "{\"sessionId\":");
    json_escape(&result, session_id);
    buf_putc(&result, '}');
    send_response_ok(j->client, j->request_id, result.ptr);
    buf_free(&result);
}

static void worker_sync_session(engine_server *s, engine_job *j) {
    pthread_mutex_lock(&s->mu);
    engine_session_slot *slot = find_session_locked(s, j->session_id);
    ds4_session *session = slot ? slot->session : NULL;
    pthread_mutex_unlock(&s->mu);
    if (!slot || !session) {
        send_response_error(j->client, j->request_id, "session_not_found",
                            "session not found", false);
        return;
    }

    ds4_tokens tokens = {0};
    ds4_tokenize_rendered_chat(s->engine, j->prefix_text, &tokens);
    ds4_tokens effective_prompt = {0};
    const ds4_tokens *prompt_for_sync = &tokens;
    const int old_pos = ds4_session_pos(session);
    const int common = ds4_session_common_prefix(session, &tokens);
    trace_cache_diag cache_diag = {0};
    trace_cache_capture(&cache_diag, ds4_session_tokens(session), &tokens,
                        old_pos, common);
    int cached = common == old_pos && tokens.len >= old_pos ? common : 0;
    const char *cache_source = cached > 0 ? "memory-token" : "none";

    if (cached == 0 && old_pos > 0) {
        int text_cached = live_text_prefix_prompt(s, session, j->prefix_text,
                                                  &effective_prompt);
        if (text_cached > 0) {
            cached = text_cached;
            cache_source = "memory-text";
            prompt_for_sync = &effective_prompt;
        }
    }

    if (cached == 0 && slot->kv.enabled) {
        slot->kv.continued_last_store_tokens = 0;
    }
    if (slot->kv.enabled && cached == 0 && old_pos >= slot->kv.opt.min_tokens) {
        session_kv_store_current(s, slot, "evict");
    }

    int disk_cached = 0;
    ds4_kvstore_load_result load_result = {0};
    if (cached == 0 && slot->kv.enabled) {
        disk_cached = ds4_kvstore_try_load_text(&slot->kv, s->engine, session,
                                                j->prefix_text, &effective_prompt,
                                                &load_result, NULL, false);
        if (disk_cached > 0) {
            cached = disk_cached;
            cache_source = "disk-text";
            prompt_for_sync = &effective_prompt;
        }
    }

    int evaluated = prompt_for_sync->len - cached;
    if (evaluated < 0) evaluated = prompt_for_sync->len;

    char err[160];
    const double t0 = now_sec();
    sync_progress progress = {
        .server = s,
        .slot = slot,
        .session = session,
        .client = j->client,
        .request_id = j->request_id,
        .cached_tokens = cached,
        .total_eval_tokens = evaluated,
        .last_current = -1,
        .phase = "prefill",
        .t0 = t0,
        .last_t = t0,
    };
    request_ctx_span(progress.ctx, sizeof(progress.ctx), cached, prompt_for_sync->len);

    engine_log(DS4_LOG_PREFILL,
               "ds4-engine: sync session=%s ctx=%s source=%s disk=%d prompt start",
               j->session_id, progress.ctx, cache_source, disk_cached);
    uint64_t trace_id = trace_sync_begin(s, j, &tokens, prompt_for_sync,
                                         cached, evaluated, &cache_diag,
                                         cache_source, disk_cached,
                                         load_result.path);

    ds4_session_set_progress(session, sync_progress_cb, &progress);

    int cold_store_len = 0;
    if (cached == 0 &&
        slot->kv.enabled &&
        prompt_for_sync->len >= slot->kv.opt.min_tokens &&
        slot->kv.opt.cold_max_tokens > 0 &&
        prompt_for_sync->len <= slot->kv.opt.cold_max_tokens)
    {
        const int anchor = ds4_kvstore_chat_anchor_pos(&slot->kv, prompt_for_sync,
                                                       ds4_token_user(s->engine),
                                                       ds4_token_assistant(s->engine));
        cold_store_len = anchor >= slot->kv.opt.min_tokens ?
                         anchor : ds4_kvstore_store_len(&slot->kv,
                                                        prompt_for_sync->len);
    }

    int suppressed_continued_last = -1;
    if (slot->kv.enabled && cold_store_len >= slot->kv.opt.min_tokens) {
        suppressed_continued_last =
            ds4_kvstore_suppress_continued_store(&slot->kv, cold_store_len);
    }

    int rc = 0;
    if (slot->kv.enabled &&
        cold_store_len >= slot->kv.opt.min_tokens &&
        cold_store_len < prompt_for_sync->len)
    {
        ds4_tokens prefix = {0};
        ds4_kvstore_tokens_copy_prefix(&prefix, prompt_for_sync, cold_store_len);
        rc = ds4_session_sync(session, &prefix, err, sizeof(err));
        ds4_tokens_free(&prefix);
        if (rc != 0) goto sync_error;

        if (session_kv_store_live(s, slot, prompt_for_sync, cold_store_len, "cold")) {
            ds4_kvstore_note_store(&slot->kv, cold_store_len);
            suppressed_continued_last = -1;
        } else {
            ds4_kvstore_restore_suppressed_continued(&slot->kv,
                                                     suppressed_continued_last,
                                                     cold_store_len);
            suppressed_continued_last = -1;
        }
    }

    rc = ds4_session_sync(session, prompt_for_sync, err, sizeof(err));
    if (rc != 0) goto sync_error;
    ds4_session_set_progress(session, NULL, NULL);
    if (slot->kv.enabled) {
        char store_err[160] = {0};
        ds4_kvstore_maybe_store_continued(&slot->kv, s->engine, session, NULL,
                                          store_err, sizeof(store_err));
    }
    const double t1 = now_sec();

    if (slot->kv.enabled && cold_store_len == prompt_for_sync->len) {
        if (session_kv_store_live(s, slot, prompt_for_sync, cold_store_len, "cold")) {
            ds4_kvstore_note_store(&slot->kv, cold_store_len);
            suppressed_continued_last = -1;
        } else {
            ds4_kvstore_restore_suppressed_continued(&slot->kv,
                                                     suppressed_continued_last,
                                                     cold_store_len);
            suppressed_continued_last = -1;
        }
    }
    send_final_prefill_progress(&progress);

    const int position = ds4_session_pos(session);
    const bool rebuilt = old_pos > 0 && cached == 0;
    pthread_mutex_lock(&s->mu);
    slot = find_session_locked(s, j->session_id);
    if (slot) {
        slot->last_sync_evaluated = evaluated;
        slot->last_sync_seconds = t1 - t0;
        slot->busy = false;
    }
    pthread_mutex_unlock(&s->mu);

    trace_log(s, "request=%s method=session.sync session=%s position=%d cached=%d evaluated=%d rebuilt=%d source=%s disk_cached=%d disk_path=%s seconds=%.3f",
              j->request_id, j->session_id, position, cached, evaluated,
              rebuilt ? 1 : 0, cache_source, disk_cached,
              load_result.path ? load_result.path : "", t1 - t0);
    engine_log(DS4_LOG_PREFILL,
               "ds4-engine: sync session=%s ctx=%s source=%s cached=%d evaluated=%d rebuilt=%d prompt done %.3fs",
               j->session_id, progress.ctx, cache_source, cached, evaluated,
               rebuilt ? 1 : 0, t1 - t0);
    trace_sync_finish(s, trace_id, position, cached, evaluated, rebuilt,
                      "ok", NULL, t1 - t0);

    buf result = {0};
    buf_puts(&result, "{\"sessionId\":");
    json_escape(&result, j->session_id);
    buf_printf(&result,
               ",\"position\":%d,\"contextWindowTokens\":%d,"
               "\"cachedPrefixTokens\":%d,\"evaluatedTokens\":%d,\"rebuilt\":%s}",
               position, ds4_session_ctx(session), cached, evaluated,
               rebuilt ? "true" : "false");
    send_response_ok(j->client, j->request_id, result.ptr);
    buf_free(&result);
    ds4_kvstore_load_result_free(&load_result);
    ds4_tokens_free(&effective_prompt);
    ds4_tokens_free(&tokens);
    return;

sync_error:
    {
        const double t_err = now_sec();
        engine_log(DS4_LOG_ERROR,
                   "ds4-engine: sync session=%s ctx=%s source=%s error=\"%s\" %.3fs",
                   j->session_id, progress.ctx, cache_source, err, t_err - t0);
        trace_sync_finish(s, trace_id, ds4_session_pos(session), cached, evaluated,
                          old_pos > 0 && cached == 0, "error", err, t_err - t0);
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_kvstore_restore_suppressed_continued(&slot->kv, suppressed_continued_last,
                                             cold_store_len);
    ds4_kvstore_load_result_free(&load_result);
    ds4_tokens_free(&effective_prompt);
    ds4_tokens_free(&tokens);
    job_clear_busy(s, j->session_id);
    send_response_error(j->client, j->request_id, "engine_error", err, false);
    trace_log(s, "request=%s method=session.sync session=%s error=%s",
              j->request_id, j->session_id, err);
}

static bool session_interrupted(engine_server *s, const char *session_id) {
    pthread_mutex_lock(&s->mu);
    engine_session_slot *slot = find_session_locked(s, session_id);
    bool interrupted = slot && slot->interrupt;
    pthread_mutex_unlock(&s->mu);
    return interrupted;
}

static void worker_generate_session(engine_server *s, engine_job *j) {
    pthread_mutex_lock(&s->mu);
    engine_session_slot *slot = find_session_locked(s, j->session_id);
    ds4_session *session = slot ? slot->session : NULL;
    int last_sync_evaluated = slot ? slot->last_sync_evaluated : 0;
    double last_sync_seconds = slot ? slot->last_sync_seconds : 0.0;
    if (slot) slot->interrupt = false;
    pthread_mutex_unlock(&s->mu);
    if (!slot || !session) {
        send_response_error(j->client, j->request_id, "session_not_found",
                            "session not found", false);
        return;
    }

    send_response_ok(j->client, j->request_id, "{\"accepted\":true}");

    int max_tokens = j->max_tokens;
    const int start_pos = ds4_session_pos(session);
    int room = ds4_session_ctx(session) - start_pos;
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    if (max_tokens < 0) max_tokens = 0;

    uint64_t rng = j->has_seed ? j->seed :
        (((uint64_t)time(NULL) << 32) ^ ((uint64_t)getpid() << 1) ^ (uint64_t)(uintptr_t)j);
    int generated = 0;
    const double t0 = now_sec();
    const char *reason = "max_tokens";
    char err[160] = {0};
    generate_stop_detector stop_detector = {0};
    bool dsml_done = false;
    bool invalidated = false;
    double last_decode_log_t = t0;
    int last_decode_log_generated = 0;
    int next_decode_log = 50;
    uint64_t trace_id = trace_generate_begin(s, j, start_pos, max_tokens, rng);
    engine_log(DS4_LOG_GENERATION,
               "ds4-engine: generate session=%s ctx=%d..%d:%d max=%d start",
               j->session_id, start_pos, start_pos, 0, max_tokens);

    while (generated < max_tokens &&
           ds4_session_pos(session) < ds4_session_ctx(session) &&
           !session_interrupted(s, j->session_id))
    {
        int token = ds4_session_sample(session, j->temperature, j->top_k,
                                       j->top_p, j->min_p, &rng);
        if (token == ds4_token_eos(s->engine)) {
            reason = "eos";
            break;
        }

        int toks[17];
        int ntok = 0;
        if (j->temperature <= 0.0f &&
            !stop_detector.dsml_seen &&
            ds4_engine_mtp_draft_tokens(s->engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL)
        {
            ntok = ds4_session_eval_speculative_argmax(session, token,
                                                       max_tokens - generated,
                                                       ds4_token_eos(s->engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                buf e = {0};
                buf_puts(&e, "{\"type\":\"error\",\"code\":\"engine_error\",\"message\":");
                json_escape(&e, err);
                buf_putc(&e, '}');
                send_event(j->client, j->request_id, e.ptr);
                buf_free(&e);
                reason = "error";
                goto done;
            }
        } else {
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                buf e = {0};
                buf_puts(&e, "{\"type\":\"error\",\"code\":\"engine_error\",\"message\":");
                json_escape(&e, err);
                buf_putc(&e, '}');
                send_event(j->client, j->request_id, e.ptr);
                buf_free(&e);
                reason = "error";
                goto done;
            }
            toks[0] = token;
            ntok = 1;
        }

        for (int i = 0; i < ntok && generated < max_tokens; i++) {
            token = toks[i];
            if (token == ds4_token_eos(s->engine)) {
                if (i + 1 < ntok) {
                    ds4_session_invalidate(session);
                    invalidated = true;
                    trace_event(s, trace_id,
                                "invalidated speculative suffix after EOS at generated token %d",
                                generated);
                }
                reason = "eos";
                goto done;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(s->engine, token, &piece_len);

            buf tok = {0};
            buf_printf(&tok, "{\"type\":\"token\",\"id\":%d}", token);
            send_event(j->client, j->request_id, tok.ptr);
            buf_free(&tok);

            if (piece_len > 0) {
                buf txt = {0};
                buf_puts(&txt, "{\"type\":\"text\",\"text\":");
                json_escape_n(&txt, piece, piece_len);
                buf_putc(&txt, '}');
                send_event(j->client, j->request_id, txt.ptr);
                buf_free(&txt);
            }
            if (piece_len > 0) trace_piece(s, trace_id, piece, piece_len);
            bool had_dsml = stop_detector.dsml_seen;
            generate_stop_detector_update(&stop_detector, piece, piece_len);
            if (!had_dsml && stop_detector.dsml_seen) {
                trace_event(s, trace_id,
                            "entered DSML tool-call block after %d generated tokens",
                            generated + 1);
            }
            free(piece);
            generated++;
            if (generated >= next_decode_log) {
                log_generate_progress(j->session_id, start_pos, generated,
                                      stop_detector.dsml_seen, dsml_done,
                                      t0, &last_decode_log_t,
                                      &last_decode_log_generated);
                next_decode_log += 50;
            }
            if (generate_stop_detector_done(&stop_detector)) {
                dsml_done = true;
                trace_event(s, trace_id,
                            "closed DSML tool-call block after %d generated tokens",
                            generated);
                if (i + 1 < ntok) {
                    ds4_session_invalidate(session);
                    invalidated = true;
                    trace_event(s, trace_id,
                                "invalidated speculative suffix after DSML stop at generated token %d",
                                generated);
                }
                reason = "stop";
                goto done;
            }
            if (session_interrupted(s, j->session_id)) {
                reason = "interrupted";
                goto done;
            }
        }
    }
    if (session_interrupted(s, j->session_id)) reason = "interrupted";

done:
    {
        const double t_done = now_sec();
        if (generated > last_decode_log_generated) {
            log_generate_progress(j->session_id, start_pos, generated,
                                  stop_detector.dsml_seen, dsml_done,
                                  t0, &last_decode_log_t,
                                  &last_decode_log_generated);
        }
        char flags[64];
        log_flags(flags, sizeof(flags), stop_detector.dsml_seen, dsml_done);
        int final_pos = ds4_session_pos(session);
        int visible_pos = invalidated ? start_pos + generated : final_pos;
        char final_ctx[48];
        request_ctx_span(final_ctx, sizeof(final_ctx), start_pos, visible_pos);
        if (!strcmp(reason, "error") && err[0]) {
            engine_log(DS4_LOG_ERROR,
                       "ds4-engine: generate session=%s ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       j->session_id, final_ctx, generated,
                       flags[0] ? " " : "", flags, reason, err, t_done - t0);
            trace_event(s, trace_id, "engine error: %s", err);
        } else {
            engine_log(DS4_LOG_GENERATION,
                       "ds4-engine: generate session=%s ctx=%s gen=%d%s%s finish=%s %.3fs",
                       j->session_id, final_ctx, generated,
                       flags[0] ? " " : "", flags, reason, t_done - t0);
        }
        trace_generate_finish(s, trace_id, reason, generated,
                              final_pos, stop_detector.dsml_seen,
                              dsml_done, invalidated, t_done - t0);
    }
    if (strcmp(reason, "error") != 0) {
        const double t1 = now_sec();
        double gen_s = t1 - t0;
        buf metrics = {0};
        buf_puts(&metrics, "{\"type\":\"metrics\"");
        if (last_sync_evaluated > 0 && last_sync_seconds > 0.0) {
            buf_printf(&metrics, ",\"prefillTps\":%.3f",
                       (double)last_sync_evaluated / last_sync_seconds);
        }
        if (generated > 0 && gen_s > 0.0) {
            buf_printf(&metrics, ",\"generationTps\":%.3f", (double)generated / gen_s);
        }
        buf_putc(&metrics, '}');
        send_event(j->client, j->request_id, metrics.ptr);
        buf_free(&metrics);

        buf done = {0};
        buf_puts(&done, "{\"type\":\"done\",\"reason\":");
        json_escape(&done, reason);
        buf_putc(&done, '}');
        send_event(j->client, j->request_id, done.ptr);
        buf_free(&done);
    }

    pthread_mutex_lock(&s->mu);
    slot = find_session_locked(s, j->session_id);
    if (slot) {
        slot->interrupt = false;
        slot->busy = false;
    }
    pthread_mutex_unlock(&s->mu);

    trace_log(s, "request=%s method=session.generate session=%s generated=%d reason=%s",
              j->request_id, j->session_id, generated, reason);
    generate_stop_detector_free(&stop_detector);
}

static void worker_destroy_session(engine_server *s, engine_job *j) {
    ds4_session *session = NULL;
    char *id = NULL;
    pthread_mutex_lock(&s->mu);
    engine_session_slot *slot = find_session_locked(s, j->session_id);
    pthread_mutex_unlock(&s->mu);
    if (slot) {
        session_kv_store_current(s, slot, "shutdown");
    }

    pthread_mutex_lock(&s->mu);
    slot = find_session_locked(s, j->session_id);
    if (slot) {
        session_kv_close(slot);
        session = slot->session;
        id = slot->id;
        memset(slot, 0, sizeof(*slot));
    }
    pthread_mutex_unlock(&s->mu);

    if (!session) {
        send_response_error(j->client, j->request_id, "session_not_found",
                            "session not found", false);
        return;
    }
    engine_log(DS4_LOG_DEFAULT,
               "ds4-engine: session.destroy id=%s",
               id ? id : j->session_id);
    trace_log(s, "request=%s method=session.destroy session=%s",
              j->request_id, id ? id : j->session_id);
    ds4_session_free(session);
    free(id);
    send_response_ok(j->client, j->request_id, "{}");
}

static void *worker_main(void *arg) {
    engine_server *s = arg;
    while (!g_stop_requested) {
        engine_job *j = dequeue_job(s);
        if (!j) break;
        switch (j->kind) {
        case JOB_SESSION_CREATE:
            worker_create_session(s, j);
            break;
        case JOB_SESSION_SYNC:
            worker_sync_session(s, j);
            break;
        case JOB_SESSION_GENERATE:
            worker_generate_session(s, j);
            break;
        case JOB_SESSION_DESTROY:
            worker_destroy_session(s, j);
            break;
        }
        job_free(j);
    }
    return NULL;
}

static const char *path_basename(const char *path) {
    if (!path || !path[0]) return "ds4flash.gguf";
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static void handle_describe(engine_server *s, client_conn *c, const char *id) {
    char host[128] = {0};
    if (gethostname(host, sizeof(host) - 1) != 0) host[0] = '\0';
    bool spec = ds4_engine_has_mtp(s->engine) && ds4_engine_mtp_draft_tokens(s->engine) > 1;
    buf result = {0};
    buf_puts(&result, "{\"protocol\":\"sloppy.engine\",\"protocolVersion\":1,\"engine\":\"ds4\",");
    buf_puts(&result, "\"model\":{\"id\":");
    json_escape(&result, path_basename(s->cfg.engine.model_path));
    buf_printf(&result,
               ",\"family\":\"deepseek-v4-flash\",\"contextWindowTokens\":%d,"
               "\"tokenizer\":\"ds4-rendered-chat\",\"chatTemplate\":\"ds4-rendered\"},",
               s->cfg.ctx_size);
    buf_puts(&result, "\"runtime\":{\"backend\":");
    json_escape(&result, ds4_backend_name(s->cfg.engine.backend));
    if (host[0]) {
        buf_puts(&result, ",\"host\":");
        json_escape(&result, host);
    }
    buf_printf(&result, ",\"pid\":%d},", (int)getpid());
    buf_puts(&result,
             "\"capabilities\":{"
             "\"renderedTextInput\":true,"
             "\"tokenInput\":false,"
             "\"tokenization\":false,"
             "\"prefixSync\":true,"
             "\"prefillProgress\":true,"
             "\"tokenStreaming\":true,"
             "\"textStreaming\":true,"
             "\"logprobs\":false,"
             "\"rewind\":false,"
             "\"snapshots\":false,"
             "\"persistentKv\":");
    buf_puts(&result, s->cfg.kv_disk_dir ? "true" : "false");
    buf_puts(&result,
             ",\"batching\":false,");
    buf_puts(&result, "\"speculativeDecode\":");
    buf_puts(&result, spec ? "true" : "false");
    buf_puts(&result, "}}");
    send_response_ok(c, id, result.ptr);
    buf_free(&result);
}

static bool reserve_session_busy(engine_server *s, const char *session_id,
                                 bool reset_interrupt, char *err, size_t errlen,
                                 const char **code, bool *retryable) {
    *code = "engine_error";
    *retryable = false;
    pthread_mutex_lock(&s->mu);
    engine_session_slot *slot = find_session_locked(s, session_id);
    if (!slot) {
        pthread_mutex_unlock(&s->mu);
        *code = "session_not_found";
        snprintf(err, errlen, "session not found");
        return false;
    }
    if (slot->busy) {
        pthread_mutex_unlock(&s->mu);
        *code = "busy";
        *retryable = true;
        snprintf(err, errlen, "session is busy");
        return false;
    }
    slot->busy = true;
    if (reset_interrupt) slot->interrupt = false;
    pthread_mutex_unlock(&s->mu);
    return true;
}

static void queue_job_for_client(engine_server *s, client_conn *c, engine_job *j) {
    client_ref(c);
    j->client = c;
    enqueue_job(s, j);
}

static void handle_request_line(engine_server *s, client_conn *c, const char *line) {
    char err[256];
    engine_request req = {0};
    if (!parse_engine_request(line, &req, err, sizeof(err))) {
        send_response_error(c, "", "invalid_request", err, false);
        return;
    }

    trace_log(s, "request=%s method=%s received", req.id, req.method);

    if (!strcmp(req.method, "engine.describe")) {
        handle_describe(s, c, req.id);
        engine_request_free(&req);
        return;
    }

    if (!strcmp(req.method, "session.tokenize") ||
        !strcmp(req.method, "session.detokenize") ||
        !strcmp(req.method, "session.rewind") ||
        !strcmp(req.method, "session.save_snapshot") ||
        !strcmp(req.method, "session.load_snapshot"))
    {
        send_response_error(c, req.id, "unsupported",
                            "method is not supported by this ds4-engine milestone", false);
        engine_request_free(&req);
        return;
    }

    if (!strcmp(req.method, "session.interrupt")) {
        char *session_id = NULL;
        if (!parse_session_id_params(req.params_json, &session_id, err, sizeof(err))) {
            send_response_error(c, req.id, "invalid_request", err, false);
            engine_request_free(&req);
            return;
        }
        pthread_mutex_lock(&s->mu);
        engine_session_slot *slot = find_session_locked(s, session_id);
        if (slot) slot->interrupt = true;
        pthread_mutex_unlock(&s->mu);
        if (!slot) {
            send_response_error(c, req.id, "session_not_found", "session not found", false);
        } else {
            engine_log(DS4_LOG_WARNING,
                       "ds4-engine: session.interrupt id=%s",
                       session_id);
            trace_log(s, "request=%s method=session.interrupt session=%s", req.id, session_id);
            send_response_ok(c, req.id, "{}");
        }
        free(session_id);
        engine_request_free(&req);
        return;
    }

    if (!strcmp(req.method, "session.create")) {
        char *requested_id = NULL;
        if (!json_object_optional_string(req.params_json, "sessionId",
                                         &requested_id, err, sizeof(err))) {
            send_response_error(c, req.id, "invalid_request", err, false);
            engine_request_free(&req);
            return;
        }
        engine_job *j = xmalloc(sizeof(*j));
        memset(j, 0, sizeof(*j));
        j->kind = JOB_SESSION_CREATE;
        j->request_id = xstrdup(req.id);
        j->params_json = xstrdup(req.params_json ? req.params_json : "{}");
        j->requested_session_id = requested_id;
        queue_job_for_client(s, c, j);
        engine_request_free(&req);
        return;
    }

    if (!strcmp(req.method, "session.sync")) {
        sync_params p = {0};
        if (!parse_sync_params(req.params_json, &p, err, sizeof(err))) {
            send_response_error(c, req.id, "invalid_request", err, false);
            engine_request_free(&req);
            return;
        }
        const char *code = NULL;
        bool retryable = false;
        if (!reserve_session_busy(s, p.session_id, false, err, sizeof(err), &code, &retryable)) {
            send_response_error(c, req.id, code, err, retryable);
            sync_params_free(&p);
            engine_request_free(&req);
            return;
        }
        engine_job *j = xmalloc(sizeof(*j));
        memset(j, 0, sizeof(*j));
        j->kind = JOB_SESSION_SYNC;
        j->request_id = xstrdup(req.id);
        j->params_json = xstrdup(req.params_json ? req.params_json : "{}");
        j->session_id = p.session_id;
        j->prefix_text = p.prefix_text;
        p.session_id = NULL;
        p.prefix_text = NULL;
        queue_job_for_client(s, c, j);
        sync_params_free(&p);
        engine_request_free(&req);
        return;
    }

    if (!strcmp(req.method, "session.generate")) {
        generate_params p = {0};
        if (!parse_generate_params(req.params_json, &p, err, sizeof(err))) {
            send_response_error(c, req.id, "invalid_request", err, false);
            engine_request_free(&req);
            return;
        }
        if (p.has_stop) {
            send_response_error(c, req.id, "unsupported",
                                "session.generate stop sequences are not supported yet", false);
            generate_params_free(&p);
            engine_request_free(&req);
            return;
        }
        const char *code = NULL;
        bool retryable = false;
        if (!reserve_session_busy(s, p.session_id, true, err, sizeof(err), &code, &retryable)) {
            send_response_error(c, req.id, code, err, retryable);
            generate_params_free(&p);
            engine_request_free(&req);
            return;
        }
        engine_job *j = xmalloc(sizeof(*j));
        memset(j, 0, sizeof(*j));
        j->kind = JOB_SESSION_GENERATE;
        j->request_id = xstrdup(req.id);
        j->params_json = xstrdup(req.params_json ? req.params_json : "{}");
        j->session_id = p.session_id;
        j->max_tokens = p.max_tokens;
        j->temperature = p.temperature;
        j->top_p = p.top_p;
        j->min_p = p.min_p;
        j->top_k = p.top_k;
        j->seed = p.seed;
        j->has_seed = p.has_seed;
        p.session_id = NULL;
        queue_job_for_client(s, c, j);
        generate_params_free(&p);
        engine_request_free(&req);
        return;
    }

    if (!strcmp(req.method, "session.destroy")) {
        char *session_id = NULL;
        if (!parse_session_id_params(req.params_json, &session_id, err, sizeof(err))) {
            send_response_error(c, req.id, "invalid_request", err, false);
            engine_request_free(&req);
            return;
        }
        const char *code = NULL;
        bool retryable = false;
        if (!reserve_session_busy(s, session_id, true, err, sizeof(err), &code, &retryable)) {
            send_response_error(c, req.id, code, err, retryable);
            free(session_id);
            engine_request_free(&req);
            return;
        }
        engine_job *j = xmalloc(sizeof(*j));
        memset(j, 0, sizeof(*j));
        j->kind = JOB_SESSION_DESTROY;
        j->request_id = xstrdup(req.id);
        j->params_json = xstrdup(req.params_json ? req.params_json : "{}");
        j->session_id = session_id;
        queue_job_for_client(s, c, j);
        engine_request_free(&req);
        return;
    }

    send_response_error(c, req.id, "unsupported", "unsupported method", false);
    engine_request_free(&req);
}

typedef struct {
    engine_server *server;
    client_conn *client;
} client_thread_arg;

static void *client_thread_main(void *arg) {
    client_thread_arg *a = arg;
    engine_server *s = a->server;
    client_conn *c = a->client;
    free(a);

    buf line = {0};
    char tmp[4096];
    while (!g_stop_requested) {
        ssize_t n = recv(c->fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (n <= 0) break;
        size_t start = 0;
        for (size_t i = 0; i < (size_t)n; i++) {
            if (tmp[i] != '\n') continue;
            if (i > start) buf_append(&line, tmp + start, i - start);
            if (line.len > 0) handle_request_line(s, c, line.ptr);
            line.len = 0;
            if (line.ptr) line.ptr[0] = '\0';
            start = i + 1;
        }
        if (start < (size_t)n) buf_append(&line, tmp + start, (size_t)n - start);
    }
    if (line.len > 0) handle_request_line(s, c, line.ptr);
    buf_free(&line);
    client_unref(c);
    return NULL;
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-engine: invalid backend: %s\n", s);
    exit(2);
}

static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-engine: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonneg_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-engine: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_uint64_arg(const char *s, const char *opt) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (!s[0] || *end || errno == ERANGE) {
        fprintf(stderr, "ds4-engine: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!s[0] || *end || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4-engine: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-engine: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-engine --socket PATH [options]\n"
        "\n"
        "Native DS4 Sloppy engine endpoint over Unix-socket NDJSON.\n"
        "\n"
        "Options:\n"
        "  --socket PATH          Unix socket path to create.\n"
        "  -m, --model FILE       GGUF model path. Default: ds4flash.gguf\n"
        "  --mtp FILE            Optional MTP support GGUF.\n"
        "  --mtp-draft N         Maximum MTP draft tokens. Default: 1\n"
        "  --mtp-margin F        MTP verifier margin. Default: 3\n"
        "  -c, --ctx N           Context size. Default: 100000\n"
        "  --max-sessions N      Maximum live sessions. Default: 1\n"
        "  --trace FILE          Write protocol and timing trace.\n"
        "  --backend NAME        metal, cuda, or cpu.\n"
        "  --metal, --cuda, --cpu\n"
        "  -t, --threads N       CPU helper threads.\n"
        "  --quality             Prefer exact kernels where available.\n"
        "  --warm-weights        Touch mapped tensor pages before serving.\n"
        "  --dir-steering-file FILE\n"
        "  --dir-steering-ffn F\n"
        "  --dir-steering-attn F\n"
        "  --kv-disk-dir DIR     Enable disk KV checkpoints in DIR.\n"
        "  --kv-disk-space-mb N  Disk KV budget. Default: 4096\n"
        "  --kv-cache-min-tokens N\n"
        "  --kv-cache-cold-max-tokens N\n"
        "  --kv-cache-continued-interval-tokens N\n"
        "  --kv-cache-boundary-trim-tokens N\n"
        "  --kv-cache-boundary-align-tokens N\n"
        "  --kv-cache-reject-different-quant\n"
        "  -h, --help            Show this help.\n");
}

static engine_config parse_options(int argc, char **argv) {
    engine_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
        },
        .kv_cache = {0},
        .ctx_size = 100000,
        .max_sessions = 1,
    };
    c.kv_cache = ds4_kvstore_default_options();
    bool steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "--socket")) {
            c.socket_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--max-sessions")) {
            c.max_sessions = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--kv-disk-dir")) {
            c.kv_disk_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--kv-disk-space-mb")) {
            c.kv_disk_space_mb = parse_uint64_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-min-tokens")) {
            c.kv_cache.min_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-cold-max-tokens")) {
            c.kv_cache.cold_max_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-continued-interval-tokens")) {
            c.kv_cache.continued_interval_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-trim-tokens")) {
            c.kv_cache.boundary_trim_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-align-tokens")) {
            c.kv_cache.boundary_align_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-reject-different-quant")) {
            c.kv_cache_reject_different_quant = true;
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else {
            fprintf(stderr, "ds4-engine: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (!c.socket_path || !c.socket_path[0]) {
        fprintf(stderr, "ds4-engine: --socket is required\n");
        exit(2);
    }
    if (c.kv_cache.cold_max_tokens > 0 &&
        c.kv_cache.cold_max_tokens < c.kv_cache.min_tokens)
    {
        fprintf(stderr, "ds4-engine: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens\n");
        exit(2);
    }
    if (c.engine.directional_steering_file && !steering_scale_set)
        c.engine.directional_steering_ffn = 1.0f;
    return c;
}

static int listen_unix_socket(const char *path) {
    if (strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        fprintf(stderr, "ds4-engine: socket path too long: %s\n", path);
        return -1;
    }
    struct stat st;
    if (lstat(path, &st) == 0) {
        fprintf(stderr, "ds4-engine: socket path already exists: %s\n", path);
        return -1;
    }
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    g_socket_created = true;
    if (listen(fd, 64) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void engine_server_free(engine_server *s) {
    if (!s) return;
    if (s->sessions) {
        for (int i = 0; i < s->cfg.max_sessions; i++) {
            session_kv_store_current(s, &s->sessions[i], "shutdown");
            session_kv_close(&s->sessions[i]);
            ds4_session_free(s->sessions[i].session);
            free(s->sessions[i].id);
        }
        free(s->sessions);
    }
    engine_job *j = s->job_head;
    while (j) {
        engine_job *next = j->next;
        job_free(j);
        j = next;
    }
    if (s->trace) fclose(s->trace);
    ds4_engine_close(s->engine);
    pthread_mutex_destroy(&s->trace_mu);
    pthread_cond_destroy(&s->cond);
    pthread_mutex_destroy(&s->mu);
}

#ifndef DS4_ENGINE_TEST_NO_MAIN
int main(int argc, char **argv) {
    signal(SIGINT, stop_signal_handler);
    signal(SIGTERM, stop_signal_handler);
    signal(SIGPIPE, SIG_IGN);

    engine_config cfg = parse_options(argc, argv);
    g_socket_path = cfg.socket_path;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &cfg.engine) != 0) {
        engine_log(DS4_LOG_ERROR, "ds4-engine: failed to load model: %s",
                   cfg.engine.model_path);
        return 1;
    }

    engine_server server;
    memset(&server, 0, sizeof(server));
    server.engine = engine;
    server.cfg = cfg;
    server.sessions = xmalloc((size_t)cfg.max_sessions * sizeof(server.sessions[0]));
    memset(server.sessions, 0, (size_t)cfg.max_sessions * sizeof(server.sessions[0]));
    pthread_mutex_init(&server.mu, NULL);
    pthread_cond_init(&server.cond, NULL);
    pthread_mutex_init(&server.trace_mu, NULL);
    if (cfg.trace_path) {
        server.trace = fopen(cfg.trace_path, "a");
        if (!server.trace) {
            engine_log(DS4_LOG_ERROR, "ds4-engine: failed to open trace %s: %s",
                       cfg.trace_path, strerror(errno));
            engine_server_free(&server);
            return 1;
        }
        engine_log(DS4_LOG_DEFAULT, "ds4-engine: tracing session to %s",
                   cfg.trace_path);
    }

    int lfd = listen_unix_socket(cfg.socket_path);
    if (lfd < 0) {
        engine_log(DS4_LOG_ERROR, "ds4-engine: failed to listen on %s: %s",
                   cfg.socket_path, strerror(errno));
        engine_server_free(&server);
        return 1;
    }
    g_listen_fd = lfd;

    pthread_t worker;
    if (pthread_create(&worker, NULL, worker_main, &server) != 0) {
        engine_log(DS4_LOG_ERROR, "ds4-engine: failed to start worker thread");
        close(lfd);
        engine_server_free(&server);
        return 1;
    }

    engine_log(DS4_LOG_DEFAULT, "ds4-engine: listening on unix://%s", cfg.socket_path);
    trace_log(&server, "listening socket=%s model=%s backend=%s ctx=%d max_sessions=%d",
              cfg.socket_path, cfg.engine.model_path,
              ds4_backend_name(cfg.engine.backend), cfg.ctx_size, cfg.max_sessions);

    while (!g_stop_requested) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0 && errno == EINTR) continue;
        if (fd < 0) break;
        client_conn *client = client_create(fd);
        client_thread_arg *arg = xmalloc(sizeof(*arg));
        arg->server = &server;
        arg->client = client;
        pthread_t tid;
        if (pthread_create(&tid, NULL, client_thread_main, arg) != 0) {
            free(arg);
            client_unref(client);
            continue;
        }
        pthread_detach(tid);
    }

    pthread_mutex_lock(&server.mu);
    server.stop = true;
    pthread_cond_broadcast(&server.cond);
    pthread_mutex_unlock(&server.mu);
    pthread_join(worker, NULL);

    if (g_listen_fd >= 0) {
        close((int)g_listen_fd);
        g_listen_fd = -1;
    }
    if (g_socket_created && g_socket_path) unlink(g_socket_path);
    engine_server_free(&server);
    return g_stop_requested ? 130 : 0;
}
#endif
