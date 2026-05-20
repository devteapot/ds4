#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#include "ds4_runtime.h"
#include "mistral35_runtime.h"
#include "qwen36_metal.h"
#include "qwen36_runtime.h"
#include <math.h>
#ifndef DS4_NO_GPU
#include "ds4_gpu.h"

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static ds4_engine *test_get_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    ds4_engine_options opt = {
        .model_path = test_model_path(),
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = quality,
    };
    TEST_ASSERT(ds4_engine_open(slot, &opt) == 0);
    return *slot;
}

static void test_close_engines(void) {
    ds4_engine_close(test_engine_fast);
    ds4_engine_close(test_engine_quality);
    test_engine_fast = NULL;
    test_engine_quality = NULL;
}

static void test_close_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    ds4_engine_close(*slot);
    *slot = NULL;
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static void test_metal_f16_matvec_fast_nr0_4(void) {
    /*
     * This is the short regression for the long-context repetition failure.
     * Decode uses one-token F16 matvecs for several DS4 projections; the fast
     * nr0=4 variant must be numerically equivalent to the plain kernel.
     */
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 512;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            ref += w * x_host[i];
        }
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static char *test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *s = malloc((size_t)len + 1);
    if (!s) {
        fclose(fp);
        return NULL;
    }
    size_t nread = fread(s, 1, (size_t)len, fp);
    fclose(fp);
    if (nread != (size_t)len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

#ifdef __APPLE__
typedef struct {
    int listen_fd;
    pthread_t thread;
    bool thread_started;
    const unsigned char *body;
    size_t body_len;
    unsigned max_requests;
    unsigned served;
} test_http_media_server;

static void test_http_send_all(int fd, const void *ptr, size_t len) {
    const unsigned char *p = ptr;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) return;
        p += (size_t)n;
        len -= (size_t)n;
    }
}

static void *test_http_media_server_thread(void *ud) {
    test_http_media_server *srv = ud;
    unsigned served = 0;
    while (served < srv->max_requests) {
        struct pollfd pfd = {
            .fd = srv->listen_fd,
            .events = POLLIN,
        };
        if (poll(&pfd, 1, 5000) <= 0) break;
        int fd = accept(srv->listen_fd, NULL, NULL);
        if (fd < 0) break;
        char hdr[256];
        int hdr_len = snprintf(hdr, sizeof(hdr),
                               "HTTP/1.1 200 OK\r\n"
                               "Content-Type: image/x-portable-pixmap\r\n"
                               "Content-Length: %llu\r\n"
                               "Connection: close\r\n"
                               "\r\n",
                               (unsigned long long)srv->body_len);
        if (hdr_len > 0 && (size_t)hdr_len < sizeof(hdr)) {
            test_http_send_all(fd, hdr, (size_t)hdr_len);
            test_http_send_all(fd, srv->body, srv->body_len);
        }
        shutdown(fd, SHUT_RDWR);
        close(fd);
        served++;
        srv->served = served;
    }
    return NULL;
}

static bool test_http_media_server_start(test_http_media_server *srv,
                                         const unsigned char *body,
                                         size_t body_len,
                                         unsigned max_requests,
                                         char *url,
                                         size_t url_len) {
    memset(srv, 0, sizeof(*srv));
    srv->listen_fd = -1;
    srv->body = body;
    srv->body_len = body_len;
    srv->max_requests = max_requests ? max_requests : 1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return false;
    }
    if (listen(fd, 8) != 0) {
        close(fd);
        return false;
    }
    socklen_t addr_len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr *)&addr, &addr_len) != 0) {
        close(fd);
        return false;
    }
    int n = snprintf(url, url_len, "http://127.0.0.1:%u/frame.ppm",
                     (unsigned)ntohs(addr.sin_port));
    if (n <= 0 || (size_t)n >= url_len) {
        close(fd);
        return false;
    }
    srv->listen_fd = fd;
    if (pthread_create(&srv->thread, NULL,
                       test_http_media_server_thread, srv) != 0) {
        close(fd);
        srv->listen_fd = -1;
        return false;
    }
    srv->thread_started = true;
    return true;
}

static void test_http_media_server_join(test_http_media_server *srv) {
    if (srv->thread_started) pthread_join(srv->thread, NULL);
    if (srv->listen_fd >= 0) close(srv->listen_fd);
    srv->listen_fd = -1;
    srv->thread_started = false;
}
#endif

typedef struct {
    const char *name;
    int number;
} test_long_fact;

static const test_long_fact test_long_facts[] = {
    {"Bob", 34},
    {"Alice", 52},
    {"Clara", 71},
    {"Diego", 93},
    {"Elena", 16},
    {"Felix", 88},
    {"Greta", 47},
    {"Hugo", 29},
    {"Iris", 64},
    {"Jonas", 12},
    {"Kira", 81},
    {"Leo", 39},
    {"Marta", 76},
    {"Nadia", 23},
    {"Owen", 58},
    {"Priya", 97},
};

static bool test_is_name_boundary(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || !(isalnum(uc) || c == '_');
}

static bool test_parse_assignment_value(const char *p, int *value) {
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '=') return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (!isdigit((unsigned char)*p)) return false;

    int v = 0;
    while (isdigit((unsigned char)*p)) {
        v = v * 10 + (*p - '0');
        p++;
    }
    *value = v;
    return true;
}

static bool test_output_has_fact(const char *text, const test_long_fact *fact) {
    const size_t name_len = strlen(fact->name);
    const char *p = text;
    bool saw_wrong_assignment = false;
    int wrong_value = -1;

    while ((p = strstr(p, fact->name)) != NULL) {
        const bool before_ok = p == text || test_is_name_boundary(p[-1]);
        const bool after_ok = test_is_name_boundary(p[name_len]) ||
                              p[name_len] == ' ' ||
                              p[name_len] == '\t' ||
                              p[name_len] == '=';
        if (before_ok && after_ok) {
            int value = 0;
            if (test_parse_assignment_value(p + name_len, &value)) {
                if (value == fact->number) return true;
                saw_wrong_assignment = true;
                wrong_value = value;
            }
        }
        p += name_len;
    }

    if (saw_wrong_assignment) {
        fprintf(stderr,
                "ds4-test: long-context wrong assignment for %s: got %d expected %d\n",
                fact->name, wrong_value, fact->number);
    } else {
        fprintf(stderr,
                "ds4-test: long-context missing assignment for %s=%d\n",
                fact->name, fact->number);
    }
    return false;
}

static int test_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool test_hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    while (*hex && !isspace((unsigned char)*hex)) {
        int hi = test_hex_digit(hex[0]);
        int lo = test_hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static bool test_token_bytes_equal(ds4_engine *engine, int token,
                                   const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = ds4_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static void test_long_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 8192 == 0) {
        fprintf(stderr, "ds4-test: long-context prefill %d/%d\n", current, total);
    }
}

static void test_long_story_fact_recall(void) {
    const char *prompt_path = getenv("DS4_TEST_LONG_PROMPT");
    if (!prompt_path || !prompt_path[0]) {
        prompt_path = "tests/long_context_story_prompt.txt";
    }
    char *prompt_text = test_read_file(prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        free(prompt_text);
        return;
    }

    ds4_tokens prompt = {0};
    ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    TEST_ASSERT(prompt.len > 30000);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 100000) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        free(prompt_text);
        return;
    }

    char err[160];
    ds4_session_set_progress(session, test_long_prefill_progress, NULL);
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    ds4_session_set_progress(session, NULL, NULL);

    buf out = {0};
    uint64_t rng = 12345;
    int generated = 0;
    bool decode_ok = true;
    for (; generated < 350; generated++) {
        int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);

        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    const char *text = out.ptr ? out.ptr : "";
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(generated > 0);
    for (size_t i = 0; i < sizeof(test_long_facts) / sizeof(test_long_facts[0]); i++) {
        TEST_ASSERT(test_output_has_fact(text, &test_long_facts[i]));
    }

    buf_free(&out);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    free(prompt_text);
}

#define TEST_VEC_MAX_STEPS 16
#define TEST_VEC_MAX_TOP 32
#define TEST_VEC_MAX_TOKEN_BYTES 128

typedef struct {
    unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} test_vec_top;

typedef struct {
    unsigned char selected[TEST_VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    test_vec_top top[TEST_VEC_MAX_TOP];
} test_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    test_vec_step steps[TEST_VEC_MAX_STEPS];
} test_vec_case;

static char *test_trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static bool test_read_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            TEST_ASSERT(vc->nsteps > 0 && vc->nsteps <= TEST_VEC_MAX_STEPS);
            return true;
        }
        TEST_ASSERT(!"unexpected line before vector case");
    }
    return false;
}

static bool test_fill_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;

        if (!strncmp(p, "step ", 5)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                TEST_ASSERT(!"bad vector step line");
                return false;
            }
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(ntop >= 0 && ntop <= TEST_VEC_MAX_TOP);
            vc->steps[step_index].ntop = ntop;
            TEST_ASSERT(test_hex_to_bytes(hex,
                                          vc->steps[step_index].selected,
                                          TEST_VEC_MAX_TOKEN_BYTES,
                                          &vc->steps[step_index].selected_len));
            top_index = 0;
            continue;
        }

        if (!strncmp(p, "top ", 4)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index < vc->steps[step_index].ntop);
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) {
                TEST_ASSERT(!"bad vector top line");
                return false;
            }
            test_vec_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            TEST_ASSERT(test_hex_to_bytes(hex, top->bytes,
                                          TEST_VEC_MAX_TOKEN_BYTES, &top->len));
            continue;
        }

        TEST_ASSERT(!"unexpected vector line");
        return false;
    }

    TEST_ASSERT(!"unterminated vector case");
    return false;
}

static void test_logprob_vector_case(ds4_engine *engine, const test_vec_case *vc) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);

    ds4_token_score scores[20];
    for (int i = 0; i < vc->nsteps; i++) {
        const test_vec_step *step = &vc->steps[i];
        int nscore = ds4_session_top_logprobs(session, scores, 20);
        int token = ds4_session_argmax(session);
        if (!test_token_bytes_equal(engine, token, step->selected, step->selected_len)) {
            fprintf(stderr, "ds4-test: vector %s step %d selected token mismatch\n",
                    vc->id, i);
            TEST_ASSERT(false);
        }

        for (int t = 0; t < step->ntop; t++) {
            bool found = false;
            float local_lp = 0.0f;
            for (int j = 0; j < nscore; j++) {
                if (scores[j].id < 0) continue;
                if (test_token_bytes_equal(engine, scores[j].id,
                                           step->top[t].bytes,
                                           step->top[t].len)) {
                    found = true;
                    local_lp = scores[j].logprob;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "ds4-test: vector %s step %d official top token missing locally\n",
                        vc->id, i);
                TEST_ASSERT(false);
            } else if (fabsf(local_lp - step->top[t].logprob) > 4.0f) {
                fprintf(stderr,
                        "ds4-test: vector %s step %d logprob delta too high: local=%g official=%g\n",
                        vc->id, i, local_lp, step->top[t].logprob);
                TEST_ASSERT(false);
            }
        }

        if (i + 1 < vc->nsteps) {
            TEST_ASSERT(ds4_session_eval(session, token, err, sizeof(err)) == 0);
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static void test_official_logprob_vectors(void) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        fclose(fp);
        return;
    }

    test_vec_case vc;
    while (test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        fprintf(stderr, "ds4-test: vector %s\n", vc.id);
        test_logprob_vector_case(engine, &vc);
    }
    fclose(fp);
}

#endif

#ifdef DS4_NO_GPU
static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static char *test_trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static char *test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *s = malloc((size_t)len + 1);
    if (!s) {
        fclose(fp);
        return NULL;
    }
    size_t nread = fread(s, 1, (size_t)len, fp);
    fclose(fp);
    if (nread != (size_t)len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}
#endif

#define QWEN36_VEC_MAX_STEPS 16
#define QWEN36_VEC_MAX_TOP 64
#define QWEN36_VEC_MAX_FULL_LOGITS 1000000

typedef struct {
    int id;
    float logit;
} qwen36_vec_top;

typedef struct {
    int selected_id;
    float selected_logit;
    int ntop;
    qwen36_vec_top top[QWEN36_VEC_MAX_TOP];
    int full_logits_len;
    float *full_logits;
} qwen36_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    char render[32];
    int ctx;
    int nsteps;
    int prompt_expected;
    int prompt_len;
    int prompt_cap;
    int *prompt_tokens;
    float tolerance;
    qwen36_vec_step steps[QWEN36_VEC_MAX_STEPS];
} qwen36_vec_case;

typedef struct {
    bool saw_model_vocab_size;
    uint32_t model_vocab_size;
    bool saw_tokenizer_vocab_size;
    uint32_t tokenizer_vocab_size;
    bool saw_eos_token_id;
    int eos_token_id;
} qwen36_vec_fixture_meta;

static void test_qwen36_vec_case_free(qwen36_vec_case *vc) {
    if (!vc) return;
    for (int i = 0; i < QWEN36_VEC_MAX_STEPS; i++) {
        free(vc->steps[i].full_logits);
        vc->steps[i].full_logits = NULL;
        vc->steps[i].full_logits_len = 0;
    }
    free(vc->prompt_tokens);
    vc->prompt_tokens = NULL;
    vc->prompt_len = 0;
    vc->prompt_cap = 0;
}

static bool test_qwen36_vec_append_prompt_token(qwen36_vec_case *vc, int token) {
    if (token < 0) return false;
    if (vc->prompt_len == vc->prompt_cap) {
        int new_cap = vc->prompt_cap ? vc->prompt_cap * 2 : 64;
        int *new_tokens = realloc(vc->prompt_tokens,
                                  (size_t)new_cap * sizeof(new_tokens[0]));
        if (!new_tokens) return false;
        vc->prompt_tokens = new_tokens;
        vc->prompt_cap = new_cap;
    }
    vc->prompt_tokens[vc->prompt_len++] = token;
    return true;
}

static bool test_qwen36_vec_parse_token_line(qwen36_vec_case *vc, const char *p) {
    while (*p) {
        while (isspace((unsigned char)*p)) p++;
        if (!*p) break;
        char *end = NULL;
        long token = strtol(p, &end, 10);
        if (end == p || token < 0 || token > INT_MAX) return false;
        if (!test_qwen36_vec_append_prompt_token(vc, (int)token)) return false;
        p = end;
    }
    return true;
}

static int test_qwen36_hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool test_qwen36_parse_u32_hex8(const char *p, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < 8; i++) {
        int n = test_qwen36_hex_nibble(p[i]);
        if (n < 0) return false;
        v = (v << 4) | (uint32_t)n;
    }
    if (out) *out = v;
    return true;
}

static bool test_qwen36_vec_parse_logits_hex_line(qwen36_vec_step *step,
                                                  const char *p,
                                                  int *seen) {
    while (*p) {
        while (isspace((unsigned char)*p)) p++;
        if (!*p) break;
        if (*seen >= step->full_logits_len) return false;
        uint32_t bits = 0;
        if (!test_qwen36_parse_u32_hex8(p, &bits)) return false;
        float f = 0.0f;
        memcpy(&f, &bits, sizeof(f));
        step->full_logits[(*seen)++] = f;
        p += 8;
        if (*p && !isspace((unsigned char)*p)) return false;
    }
    return true;
}

static bool test_qwen36_vec_count_logits_hex_line(const char *p,
                                                  uint32_t expected,
                                                  uint32_t *seen) {
    while (*p) {
        while (isspace((unsigned char)*p)) p++;
        if (!*p) break;
        if (*seen >= expected) return false;
        uint32_t bits = 0;
        if (!test_qwen36_parse_u32_hex8(p, &bits)) return false;
        (void)bits;
        (*seen)++;
        p += 8;
        if (*p && !isspace((unsigned char)*p)) return false;
    }
    return true;
}

static bool test_qwen36_parse_u32_field(const char *p, uint32_t *out) {
    while (isspace((unsigned char)*p)) p++;
    char *end = NULL;
    unsigned long v = strtoul(p, &end, 10);
    if (end == p || v > UINT32_MAX) return false;
    while (isspace((unsigned char)*end)) end++;
    if (*end) return false;
    if (out) *out = (uint32_t)v;
    return true;
}

static bool test_qwen36_parse_int_field(const char *p, int *out) {
    while (isspace((unsigned char)*p)) p++;
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p || v < 0 || v > INT_MAX) return false;
    while (isspace((unsigned char)*end)) end++;
    if (*end) return false;
    if (out) *out = (int)v;
    return true;
}

static bool test_qwen36_validate_vector_fixture_preamble(FILE *fp,
                                                         const char *path,
                                                         bool require_pinned_revision,
                                                         qwen36_vec_fixture_meta *meta) {
    char line[4096];
    bool saw_header = false;
    bool saw_model = false;
    bool saw_revision = false;
    bool pinned_revision = false;
    bool saw_full_logits = false;
    bool saw_case = false;
    qwen36_vec_fixture_meta parsed_meta = {0};

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0]) continue;
        if (p[0] == '#') {
            if (!strcmp(p, "# qwen36-official-full-logit-vectors-v1")) {
                saw_header = true;
            } else if (!strncmp(p, "# model ", 8)) {
                saw_model = !strcmp(p + 8, QWEN36_RUNTIME_HF_REPO);
            } else if (!strncmp(p, "# revision ", 11)) {
                const char *rev = p + 11;
                saw_revision = true;
                pinned_revision = rev[0] &&
                                  strcmp(rev, "default") &&
                                  strcmp(rev, "none") &&
                                  strcmp(rev, "None");
            } else if (!strncmp(p, "# full_logits ", 14)) {
                const char *v = p + 14;
                saw_full_logits = !strcmp(v, "1") ||
                                  !strcmp(v, "true") ||
                                  !strcmp(v, "True");
            } else if (!strncmp(p, "# model_vocab_size ", 19)) {
                parsed_meta.saw_model_vocab_size =
                    test_qwen36_parse_u32_field(p + 19,
                                                &parsed_meta.model_vocab_size);
            } else if (!strncmp(p, "# tokenizer_vocab_size ", 23)) {
                parsed_meta.saw_tokenizer_vocab_size =
                    test_qwen36_parse_u32_field(
                        p + 23, &parsed_meta.tokenizer_vocab_size);
            } else if (!strncmp(p, "# eos_token_id ", 15)) {
                parsed_meta.saw_eos_token_id =
                    test_qwen36_parse_int_field(p + 15,
                                                &parsed_meta.eos_token_id);
            }
            continue;
        }
        if (!strncmp(p, "case ", 5)) {
            saw_case = true;
            break;
        }
        fprintf(stderr,
                "ds4-test: unexpected Qwen vector preamble line in %s: %s\n",
                path ? path : "(unknown)", p);
        TEST_ASSERT(false);
        return false;
    }

    if (!saw_header) {
        fprintf(stderr,
                "ds4-test: Qwen vector fixture missing format header: %s\n",
                path ? path : "(unknown)");
    }
    if (!saw_model) {
        fprintf(stderr,
                "ds4-test: Qwen vector fixture missing model provenance for %s: %s\n",
                QWEN36_RUNTIME_HF_REPO, path ? path : "(unknown)");
    }
    if (require_pinned_revision && (!saw_revision || !pinned_revision)) {
        fprintf(stderr,
                "ds4-test: strict Qwen vectors require a pinned HF revision: %s\n",
                path ? path : "(unknown)");
    }
    if (require_pinned_revision && !saw_full_logits) {
        fprintf(stderr,
                "ds4-test: strict Qwen vectors require dense full-logit provenance: %s\n",
                path ? path : "(unknown)");
    }
    if (require_pinned_revision && !parsed_meta.saw_model_vocab_size) {
        fprintf(stderr,
                "ds4-test: strict Qwen vectors require model_vocab_size provenance: %s\n",
                path ? path : "(unknown)");
    }
    if (require_pinned_revision && !parsed_meta.saw_tokenizer_vocab_size) {
        fprintf(stderr,
                "ds4-test: strict Qwen vectors require tokenizer_vocab_size provenance: %s\n",
                path ? path : "(unknown)");
    }
    if (require_pinned_revision && !parsed_meta.saw_eos_token_id) {
        fprintf(stderr,
                "ds4-test: strict Qwen vectors require eos_token_id provenance: %s\n",
                path ? path : "(unknown)");
    }
    if (!saw_case) {
        fprintf(stderr,
                "ds4-test: Qwen vector fixture contains no cases: %s\n",
                path ? path : "(unknown)");
    }

    TEST_ASSERT(saw_header);
    TEST_ASSERT(saw_model);
    TEST_ASSERT(!require_pinned_revision || (saw_revision && pinned_revision));
    TEST_ASSERT(!require_pinned_revision || saw_full_logits);
    TEST_ASSERT(!require_pinned_revision || parsed_meta.saw_model_vocab_size);
    TEST_ASSERT(!require_pinned_revision ||
                parsed_meta.saw_tokenizer_vocab_size);
    TEST_ASSERT(!require_pinned_revision || parsed_meta.saw_eos_token_id);
    TEST_ASSERT(saw_case);
    TEST_ASSERT(fseek(fp, 0, SEEK_SET) == 0);
    if (meta) *meta = parsed_meta;
    return saw_header && saw_model &&
           (!require_pinned_revision || (saw_revision && pinned_revision)) &&
           (!require_pinned_revision || saw_full_logits) &&
           (!require_pinned_revision || parsed_meta.saw_model_vocab_size) &&
           (!require_pinned_revision || parsed_meta.saw_tokenizer_vocab_size) &&
           (!require_pinned_revision || parsed_meta.saw_eos_token_id) &&
           saw_case;
}

static bool test_qwen36_validate_vector_fixture_dense_sections(FILE *fp,
                                                               const char *path) {
    char line[4096];
    const uint32_t expected_vocab = qwen36_runtime_model_spec()->vocab_size;
    bool in_case = false;
    bool in_logits = false;
    bool current_step_open = false;
    bool current_step_dense = false;
    int expected_steps = 0;
    int seen_steps = 0;
    uint32_t seen_logits = 0;
    int cases = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;

        if (in_logits) {
            if (!strcmp(p, "end_logits")) {
                if (seen_logits != expected_vocab) {
                    fprintf(stderr,
                            "ds4-test: Qwen vector dense logits length mismatch in %s: got=%u expected=%u\n",
                            path ? path : "(unknown)",
                            seen_logits, expected_vocab);
                    TEST_ASSERT(false);
                    return false;
                }
                in_logits = false;
                current_step_dense = true;
                continue;
            }
            if (!strncmp(p, "logits_hex ", 11)) p += 11;
            if (!test_qwen36_vec_count_logits_hex_line(
                    p, expected_vocab, &seen_logits)) {
                fprintf(stderr,
                        "ds4-test: malformed Qwen dense logit line in %s\n",
                        path ? path : "(unknown)");
                TEST_ASSERT(false);
                return false;
            }
            continue;
        }

        if (!strncmp(p, "case ", 5)) {
            if (in_case) {
                fprintf(stderr,
                        "ds4-test: Qwen vector case missing end before next case in %s\n",
                        path ? path : "(unknown)");
                TEST_ASSERT(false);
                return false;
            }
            char id[96];
            int ctx = 0;
            if (sscanf(p, "case %95s %d %d", id, &ctx, &expected_steps) != 3 ||
                ctx <= 0 || expected_steps <= 0 ||
                expected_steps > QWEN36_VEC_MAX_STEPS) {
                TEST_ASSERT(!"bad Qwen vector case line");
                return false;
            }
            in_case = true;
            current_step_open = false;
            current_step_dense = false;
            seen_steps = 0;
            cases++;
            continue;
        }

        if (!strncmp(p, "step ", 5)) {
            if (!in_case) {
                TEST_ASSERT(!"Qwen vector step outside case");
                return false;
            }
            if (current_step_open && !current_step_dense) {
                fprintf(stderr,
                        "ds4-test: Qwen vector step missing dense logits before next step in %s\n",
                        path ? path : "(unknown)");
                TEST_ASSERT(false);
                return false;
            }
            current_step_open = true;
            current_step_dense = false;
            seen_steps++;
            continue;
        }

        if (!strncmp(p, "logits_f32_hex ", 15)) {
            if (!current_step_open || current_step_dense) {
                TEST_ASSERT(!"Qwen dense logits outside step or duplicated");
                return false;
            }
            char *rest = p + 15;
            char *end = NULL;
            long count = strtol(rest, &end, 10);
            if (end == rest || count != (long)expected_vocab) {
                fprintf(stderr,
                        "ds4-test: Qwen dense logits count mismatch in %s: got=%ld expected=%u\n",
                        path ? path : "(unknown)", count, expected_vocab);
                TEST_ASSERT(false);
                return false;
            }
            in_logits = true;
            seen_logits = 0;
            continue;
        }

        if (!strcmp(p, "end")) {
            if (!in_case || in_logits ||
                (current_step_open && !current_step_dense) ||
                seen_steps != expected_steps) {
                fprintf(stderr,
                        "ds4-test: Qwen vector case has incomplete dense logits in %s\n",
                        path ? path : "(unknown)");
                TEST_ASSERT(false);
                return false;
            }
            in_case = false;
            current_step_open = false;
            current_step_dense = false;
            expected_steps = 0;
            seen_steps = 0;
            continue;
        }
    }

    bool rewind_ok = fseek(fp, 0, SEEK_SET) == 0;
    TEST_ASSERT(!in_case);
    TEST_ASSERT(!in_logits);
    TEST_ASSERT(cases > 0);
    TEST_ASSERT(rewind_ok);
    return !in_case && !in_logits && cases > 0 && rewind_ok;
}

static bool test_read_qwen36_vector_case(FILE *fp, qwen36_vec_case *vc) {
    char line[4096];
    memset(vc, 0, sizeof(*vc));
    vc->prompt_expected = -1;
    vc->tolerance = 0.25f;
    strcpy(vc->render, "raw");
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            TEST_ASSERT(vc->ctx > 0);
            TEST_ASSERT(vc->nsteps > 0 && vc->nsteps <= QWEN36_VEC_MAX_STEPS);
            return true;
        }
        TEST_ASSERT(!"unexpected line before Qwen vector case");
    }
    return false;
}

static bool test_fill_qwen36_vector_case(FILE *fp, qwen36_vec_case *vc) {
    char line[4096];
    int step_index = -1;
    int top_index = 0;
    bool reading_prompt = false;
    bool reading_logits = false;
    int logits_step_index = -1;
    int logits_seen = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;

        if (!strcmp(p, "end")) {
            TEST_ASSERT(!reading_prompt);
            TEST_ASSERT(!reading_logits);
            if (step_index >= 0) {
                TEST_ASSERT(top_index == vc->steps[step_index].ntop);
            }
            return true;
        }

        if (!strcmp(p, "end_logits")) {
            TEST_ASSERT(reading_logits);
            TEST_ASSERT(logits_step_index >= 0 &&
                        logits_step_index < vc->nsteps);
            TEST_ASSERT(logits_seen ==
                        vc->steps[logits_step_index].full_logits_len);
            reading_logits = false;
            logits_step_index = -1;
            logits_seen = 0;
            continue;
        }

        if (reading_logits) {
            if (!strncmp(p, "logits_hex ", 11)) p += 11;
            TEST_ASSERT(logits_step_index >= 0 &&
                        logits_step_index < vc->nsteps);
            if (!test_qwen36_vec_parse_logits_hex_line(
                    &vc->steps[logits_step_index], p, &logits_seen)) {
                TEST_ASSERT(!"bad Qwen full-logit hex line");
                return false;
            }
            continue;
        }

        if (!strcmp(p, "end_prompt_tokens")) {
            TEST_ASSERT(vc->prompt_expected >= 0);
            TEST_ASSERT(vc->prompt_len == vc->prompt_expected);
            reading_prompt = false;
            continue;
        }

        if (!strncmp(p, "render ", 7)) {
            TEST_ASSERT(sscanf(p, "render %31s", vc->render) == 1);
            continue;
        }

        if (!strncmp(p, "tolerance ", 10)) {
            TEST_ASSERT(sscanf(p, "tolerance %f", &vc->tolerance) == 1);
            TEST_ASSERT(vc->tolerance >= 0.0f);
            continue;
        }

        if (!strncmp(p, "prompt_tokens ", 14)) {
            char *rest = p + 14;
            char *end = NULL;
            long count = strtol(rest, &end, 10);
            TEST_ASSERT(end != rest && count >= 0 && count <= INT_MAX);
            vc->prompt_expected = (int)count;
            if (!test_qwen36_vec_parse_token_line(vc, end)) {
                TEST_ASSERT(!"bad Qwen prompt token line");
                return false;
            }
            reading_prompt = vc->prompt_len < vc->prompt_expected;
            TEST_ASSERT(vc->prompt_len <= vc->prompt_expected);
            continue;
        }

        if (reading_prompt) {
            if (!strncmp(p, "tokens ", 7)) p += 7;
            if (!test_qwen36_vec_parse_token_line(vc, p)) {
                TEST_ASSERT(!"bad Qwen prompt token continuation");
                return false;
            }
            TEST_ASSERT(vc->prompt_len <= vc->prompt_expected);
            if (vc->prompt_len == vc->prompt_expected) reading_prompt = false;
            continue;
        }

        if (!strncmp(p, "step ", 5)) {
            if (step_index >= 0) {
                TEST_ASSERT(top_index == vc->steps[step_index].ntop);
            }
            int ntop = 0;
            int parsed_step = -1;
            int selected_id = -1;
            float selected_logit = 0.0f;
            if (sscanf(p, "step %d %d %f %d",
                       &parsed_step, &selected_id, &selected_logit,
                       &ntop) != 4) {
                TEST_ASSERT(!"bad Qwen vector step line");
                return false;
            }
            step_index = parsed_step;
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(selected_id >= 0);
            TEST_ASSERT(ntop >= 0 && ntop <= QWEN36_VEC_MAX_TOP);
            vc->steps[step_index].selected_id = selected_id;
            vc->steps[step_index].selected_logit = selected_logit;
            vc->steps[step_index].ntop = ntop;
            top_index = 0;
            continue;
        }

        if (!strncmp(p, "logits_f32_hex ", 15)) {
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index == vc->steps[step_index].ntop);
            char *rest = p + 15;
            char *end = NULL;
            long count = strtol(rest, &end, 10);
            TEST_ASSERT(end != rest && count > 0 &&
                        count <= QWEN36_VEC_MAX_FULL_LOGITS);
            qwen36_vec_step *step = &vc->steps[step_index];
            TEST_ASSERT(step->full_logits == NULL);
            step->full_logits = malloc((size_t)count *
                                       sizeof(step->full_logits[0]));
            TEST_ASSERT(step->full_logits != NULL);
            if (!step->full_logits) return false;
            step->full_logits_len = (int)count;
            reading_logits = true;
            logits_step_index = step_index;
            logits_seen = 0;
            continue;
        }

        if (!strncmp(p, "top ", 4)) {
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index < vc->steps[step_index].ntop);
            qwen36_vec_top *top = &vc->steps[step_index].top[top_index++];
            if (sscanf(p, "top %d %f", &top->id, &top->logit) != 2) {
                TEST_ASSERT(!"bad Qwen vector top line");
                return false;
            }
            TEST_ASSERT(top->id >= 0);
            continue;
        }

        TEST_ASSERT(!"unexpected Qwen vector line");
        return false;
    }

    TEST_ASSERT(!"unterminated Qwen vector case");
    return false;
}

static bool test_qwen36_render_vector_prompt(rt_engine *engine,
                                             const qwen36_vec_case *vc,
                                             rt_tokens *out) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return false;

    int rc = -1;
    if (!strcmp(vc->render, "raw")) {
        rc = rt_tokenize_text(engine, prompt_text, out);
    } else if (!strcmp(vc->render, "chat-disabled") ||
               !strcmp(vc->render, "chat-enabled") ||
               !strcmp(vc->render, "chat-auto")) {
        rt_chat_message msg = {
            .role = "user",
            .content = prompt_text,
        };
        qwen36_runtime_chat_options chat = {0};
        if (!strcmp(vc->render, "chat-disabled")) {
            chat.think_mode = QWEN36_THINK_DISABLED;
        } else if (!strcmp(vc->render, "chat-enabled")) {
            chat.think_mode = QWEN36_THINK_ENABLED;
        } else {
            chat.think_mode = QWEN36_THINK_AUTO;
        }
        rt_chat_render_options render = {
            .add_generation_prompt = true,
            .model_options = &chat,
        };
        rc = rt_render_chat(engine, &msg, 1, &render, out);
    } else {
        TEST_ASSERT(!"unsupported Qwen vector render mode");
    }

    free(prompt_text);
    return rc == 0;
}

static void test_qwen36_vector_case(rt_engine *engine,
                                    const qwen36_vec_case *vc,
                                    bool require_full_logits) {
    rt_tokens prompt = {0};
    TEST_ASSERT(test_qwen36_render_vector_prompt(engine, vc, &prompt));
    if (vc->prompt_expected >= 0) {
        TEST_ASSERT(prompt.len == vc->prompt_expected);
        int n = prompt.len < vc->prompt_expected ? prompt.len : vc->prompt_expected;
        for (int i = 0; i < n; i++) {
            if (prompt.v[i] != vc->prompt_tokens[i]) {
                fprintf(stderr,
                        "ds4-test: Qwen vector %s prompt token %d mismatch: local=%d official=%d\n",
                        vc->id, i, prompt.v[i], vc->prompt_tokens[i]);
                TEST_ASSERT(false);
                break;
            }
        }
    }

    rt_session *session = NULL;
    TEST_ASSERT(rt_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        rt_tokens_free(&prompt);
        return;
    }

    char err[256] = {0};
    TEST_ASSERT(rt_session_sync(session, &prompt, err, sizeof(err)) == 0);

    int vocab = rt_engine_vocab_size(engine);
    TEST_ASSERT(vocab > 0);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    TEST_ASSERT(logits != NULL);
    if (!logits) {
        rt_session_free(session);
        rt_tokens_free(&prompt);
        return;
    }

    for (int i = 0; i < vc->nsteps; i++) {
        const qwen36_vec_step *step = &vc->steps[i];
        TEST_ASSERT(step->selected_id >= 0 && step->selected_id < vocab);
        TEST_ASSERT(rt_session_read_logits(session, logits, (uint32_t)vocab) == 0);
        int local_selected = rt_session_argmax(session);
        if (local_selected != step->selected_id) {
            fprintf(stderr,
                    "ds4-test: Qwen vector %s step %d selected mismatch: local=%d official=%d\n",
                    vc->id, i, local_selected, step->selected_id);
            TEST_ASSERT(false);
        }
        float delta = fabsf(logits[step->selected_id] - step->selected_logit);
        if (delta > vc->tolerance) {
            fprintf(stderr,
                    "ds4-test: Qwen vector %s step %d selected logit delta too high: local=%g official=%g tol=%g\n",
                    vc->id, i, logits[step->selected_id],
                    step->selected_logit, vc->tolerance);
            TEST_ASSERT(false);
        }
        for (int j = 0; j < step->ntop; j++) {
            int id = step->top[j].id;
            TEST_ASSERT(id >= 0 && id < vocab);
            if (id < 0 || id >= vocab) continue;
            delta = fabsf(logits[id] - step->top[j].logit);
            if (delta > vc->tolerance) {
                fprintf(stderr,
                        "ds4-test: Qwen vector %s step %d top token %d logit delta too high: local=%g official=%g tol=%g\n",
                        vc->id, i, id, logits[id], step->top[j].logit,
                        vc->tolerance);
                TEST_ASSERT(false);
            }
        }
        if (require_full_logits || step->full_logits_len > 0) {
            TEST_ASSERT(step->full_logits != NULL);
            TEST_ASSERT(step->full_logits_len == vocab);
            if (step->full_logits && step->full_logits_len == vocab) {
                float max_delta = 0.0f;
                int max_id = -1;
                bool ok = true;
                for (int id = 0; id < vocab; id++) {
                    float official = step->full_logits[id];
                    delta = fabsf(logits[id] - official);
                    if (delta > max_delta) {
                        max_delta = delta;
                        max_id = id;
                    }
                    if (!(logits[id] == logits[id]) ||
                        !(official == official) ||
                        delta > vc->tolerance) {
                        if (max_id < 0) {
                            max_delta = delta;
                            max_id = id;
                        }
                        ok = false;
                        break;
                    }
                }
                if (!ok) {
                    fprintf(stderr,
                            "ds4-test: Qwen vector %s step %d full-logit delta too high: token=%d local=%g official=%g max_delta=%g tol=%g\n",
                            vc->id, i, max_id, logits[max_id],
                            step->full_logits[max_id], max_delta,
                            vc->tolerance);
                    TEST_ASSERT(false);
                }
            }
        }
        if (i + 1 < vc->nsteps) {
            TEST_ASSERT(rt_session_eval(session, step->selected_id,
                                        err, sizeof(err)) == 0);
        }
    }

    free(logits);
    rt_session_free(session);
    rt_tokens_free(&prompt);
}

static bool test_env_truthy(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") &&
           strcmp(v, "false") && strcmp(v, "FALSE") &&
           strcmp(v, "no") && strcmp(v, "NO");
}

static bool test_qwen36_text_only_mode(void) {
    return test_env_truthy("QWEN36_TEST_TEXT_ONLY");
}

static void test_qwen36_official_vectors(void) {
    const char *path = getenv("QWEN36_TEST_VECTOR_FILE");
    bool explicit_path = path && path[0];
    bool require_vectors = test_env_truthy("QWEN36_REQUIRE_OFFICIAL_VECTORS");
    if (!explicit_path) path = "tests/qwen36-vectors/official.vec";

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        if (explicit_path || require_vectors) {
            fprintf(stderr, "ds4-test: Qwen official vector fixture not found: %s\n",
                    path);
            TEST_ASSERT(false);
        } else {
            fprintf(stderr, "ds4-test: Qwen official vector fixture not found; skipping (%s)\n",
                    path);
        }
        return;
    }

    qwen36_vec_fixture_meta meta = {0};
    if (!test_qwen36_validate_vector_fixture_preamble(
            fp, path, require_vectors, &meta)) {
        fclose(fp);
        return;
    }
    if (require_vectors &&
        !test_qwen36_validate_vector_fixture_dense_sections(fp, path)) {
        fclose(fp);
        return;
    }

    const char *model_path = getenv("QWEN36_TEST_MODEL");
    if (!model_path || !model_path[0]) model_path = getenv("DS4_TEST_MODEL");
    if (!model_path || !model_path[0]) {
        fprintf(stderr, "ds4-test: Qwen official vectors require QWEN36_TEST_MODEL; skipping\n");
        TEST_ASSERT(!explicit_path && !require_vectors);
        fclose(fp);
        return;
    }

    const rt_model_ops *qwen = qwen36_runtime_ops();
    rt_engine *engine = NULL;
    rt_engine_options opt = {
        .model_path = model_path,
        .backend = RT_BACKEND_CPU,
    };
    TEST_ASSERT(rt_engine_open_with_ops(&engine, qwen, &opt) == 0);
    if (!engine) {
        fclose(fp);
        return;
    }
    int engine_vocab = rt_engine_vocab_size(engine);
    TEST_ASSERT(engine_vocab > 0);
    if (meta.saw_model_vocab_size) {
        TEST_ASSERT(meta.model_vocab_size == (uint32_t)engine_vocab);
    }
    if (meta.saw_tokenizer_vocab_size) {
        TEST_ASSERT(meta.tokenizer_vocab_size == (uint32_t)engine_vocab);
    }
    if (meta.saw_eos_token_id) {
        TEST_ASSERT(meta.eos_token_id == rt_token_eos(engine));
    }

    qwen36_vec_case vc;
    while (test_read_qwen36_vector_case(fp, &vc)) {
        if (!test_fill_qwen36_vector_case(fp, &vc)) {
            test_qwen36_vec_case_free(&vc);
            break;
        }
        fprintf(stderr, "ds4-test: Qwen vector %s\n", vc.id);
        test_qwen36_vector_case(engine, &vc, require_vectors);
        test_qwen36_vec_case_free(&vc);
    }

    rt_engine_close(engine);
    fclose(fp);
}

#ifndef DS4_NO_GPU
static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static void test_tool_call_quality_one(bool quality) {
    ds4_engine *engine = test_get_engine(quality);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_tool_call_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(ds4_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    buf text = {0};
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    for (int i = 0; i < r.max_tokens; i++) {
        int token = ds4_session_sample(session, r.temperature, r.top_k,
                                       r.top_p, r.min_p, &rng);
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr ? text.ptr : "", &saw_tool_start, &saw_tool_end, NULL);
        if (saw_tool_end) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr ? text.ptr : "",
                                             false, &content, &reasoning, &calls);
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    ds4_session_free(session);
    request_free(&r);
}

static void test_tool_call_quality(void) {
    fprintf(stderr, "ds4-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "ds4-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}

#endif

static void test_server_unit_group(void) {
    ds4_server_unit_tests_run();
}

static void test_write_u32(FILE *fp, uint32_t v) {
    uint8_t b[4] = {
        (uint8_t)(v & 0xffu),
        (uint8_t)((v >> 8) & 0xffu),
        (uint8_t)((v >> 16) & 0xffu),
        (uint8_t)((v >> 24) & 0xffu),
    };
    TEST_ASSERT(fwrite(b, 1, sizeof(b), fp) == sizeof(b));
}

static uint32_t test_read_le32_mem(const uint8_t *b) {
    return (uint32_t)b[0] |
           ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
}

static uint64_t test_read_le64_words_mem(const uint8_t *b) {
    return (uint64_t)test_read_le32_mem(b) |
           ((uint64_t)test_read_le32_mem(b + 4) << 32);
}

static void test_write_u16(FILE *fp, uint16_t v) {
    uint8_t b[2] = {
        (uint8_t)(v & 0xffu),
        (uint8_t)((v >> 8) & 0xffu),
    };
    TEST_ASSERT(fwrite(b, 1, sizeof(b), fp) == sizeof(b));
}

static void test_write_u64(FILE *fp, uint64_t v) {
    uint8_t b[8] = {
        (uint8_t)(v & 0xffu),
        (uint8_t)((v >> 8) & 0xffu),
        (uint8_t)((v >> 16) & 0xffu),
        (uint8_t)((v >> 24) & 0xffu),
        (uint8_t)((v >> 32) & 0xffu),
        (uint8_t)((v >> 40) & 0xffu),
        (uint8_t)((v >> 48) & 0xffu),
        (uint8_t)((v >> 56) & 0xffu),
    };
    TEST_ASSERT(fwrite(b, 1, sizeof(b), fp) == sizeof(b));
}

static void test_write_f32(FILE *fp, float v) {
    TEST_ASSERT(fwrite(&v, 1, sizeof(v), fp) == sizeof(v));
}

static void test_patch_gguf_tensor_f32(const char *path,
                                       const char *tensor_name,
                                       uint64_t index,
                                       float value) {
    char err[256] = {0};
    rt_gguf_file *file = NULL;
    TEST_ASSERT(rt_gguf_file_open(&file, path, err, sizeof(err)) == 0);
    TEST_ASSERT(file != NULL);
    const rt_gguf_tensor *tensor = rt_gguf_file_find_tensor(file, tensor_name);
    TEST_ASSERT(tensor != NULL);
    TEST_ASSERT(index < tensor->elements);
    uint32_t type = tensor->type;
    uint64_t elem_bytes = type == 1 ? sizeof(uint16_t) : sizeof(float);
    TEST_ASSERT(type == 0 || type == 1);
    uint64_t offset = tensor->data_offset + index * elem_bytes;
    rt_gguf_file_close(file);

    FILE *fp = fopen(path, "r+b");
    TEST_ASSERT(fp != NULL);
    TEST_ASSERT(fseeko(fp, (off_t)offset, SEEK_SET) == 0);
    if (type == 1) {
        test_write_u16(fp, test_float_to_f16(value));
    } else {
        test_write_f32(fp, value);
    }
    TEST_ASSERT(fclose(fp) == 0);
}

static void test_write_gguf_string(FILE *fp, const char *s) {
    size_t len = strlen(s);
    test_write_u64(fp, (uint64_t)len);
    TEST_ASSERT(fwrite(s, 1, len, fp) == len);
}

static void test_write_gguf_kv_string(FILE *fp, const char *key, const char *value) {
    test_write_gguf_string(fp, key);
    test_write_u32(fp, RT_GGUF_VALUE_STRING);
    test_write_gguf_string(fp, value);
}

static void test_write_gguf_kv_u32(FILE *fp, const char *key, uint32_t value) {
    test_write_gguf_string(fp, key);
    test_write_u32(fp, RT_GGUF_VALUE_UINT32);
    test_write_u32(fp, value);
}

static void test_write_gguf_kv_bool(FILE *fp, const char *key, bool value) {
    test_write_gguf_string(fp, key);
    test_write_u32(fp, RT_GGUF_VALUE_BOOL);
    TEST_ASSERT(fputc(value ? 1 : 0, fp) != EOF);
}

static void test_write_gguf_kv_string_array(FILE *fp, const char *key, const char *const *values, size_t n_values) {
    test_write_gguf_string(fp, key);
    test_write_u32(fp, RT_GGUF_VALUE_ARRAY);
    test_write_u32(fp, RT_GGUF_VALUE_STRING);
    test_write_u64(fp, (uint64_t)n_values);
    for (size_t i = 0; i < n_values; i++) {
        test_write_gguf_string(fp, values[i]);
    }
}

static void test_write_gguf_kv_string_array_one(FILE *fp, const char *key, const char *value) {
    const char *values[] = {value};
    test_write_gguf_kv_string_array(fp, key, values, 1);
}

static void test_write_gguf_header(FILE *fp, uint64_t n_tensors, uint64_t n_kv) {
    TEST_ASSERT(fwrite("GGUF", 1, 4, fp) == 4);
    test_write_u32(fp, 3);
    test_write_u64(fp, n_tensors);
    test_write_u64(fp, n_kv);
}

static void test_write_gguf_tensor_1d(FILE *fp, const char *name, uint64_t dim, uint32_t type, uint64_t rel_offset) {
    test_write_gguf_string(fp, name);
    test_write_u32(fp, 1);
    test_write_u64(fp, dim);
    test_write_u32(fp, type);
    test_write_u64(fp, rel_offset);
}

static void test_write_gguf_tensor_2d(FILE *fp,
                                      const char *name,
                                      uint64_t dim0,
                                      uint64_t dim1,
                                      uint32_t type,
                                      uint64_t rel_offset) {
    test_write_gguf_string(fp, name);
    test_write_u32(fp, 2);
    test_write_u64(fp, dim0);
    test_write_u64(fp, dim1);
    test_write_u32(fp, type);
    test_write_u64(fp, rel_offset);
}

static void test_write_gguf_tensor_4d(FILE *fp,
                                      const char *name,
                                      uint64_t dim0,
                                      uint64_t dim1,
                                      uint64_t dim2,
                                      uint64_t dim3,
                                      uint32_t type,
                                      uint64_t rel_offset) {
    test_write_gguf_string(fp, name);
    test_write_u32(fp, 4);
    test_write_u64(fp, dim0);
    test_write_u64(fp, dim1);
    test_write_u64(fp, dim2);
    test_write_u64(fp, dim3);
    test_write_u32(fp, type);
    test_write_u64(fp, rel_offset);
}

static void test_write_q3_k_scale_bytes(FILE *fp, const uint8_t scale[16]) {
    uint32_t raw0 = 0;
    uint32_t raw1 = 0;
    uint32_t raw2 = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t s0 = scale[i];
        uint8_t s1 = scale[i + 4];
        uint8_t s2 = scale[i + 8];
        uint8_t s3 = scale[i + 12];
        uint8_t b0 = (uint8_t)((s0 & 0x0fu) | ((s2 & 0x0fu) << 4));
        uint8_t b1 = (uint8_t)((s1 & 0x0fu) | ((s3 & 0x0fu) << 4));
        uint8_t b2 = (uint8_t)(((s0 >> 4) & 3u) |
                               (((s1 >> 4) & 3u) << 2) |
                               (((s2 >> 4) & 3u) << 4) |
                               (((s3 >> 4) & 3u) << 6));
        raw0 |= (uint32_t)b0 << (i * 8);
        raw1 |= (uint32_t)b1 << (i * 8);
        raw2 |= (uint32_t)b2 << (i * 8);
    }
    test_write_u32(fp, raw0);
    test_write_u32(fp, raw1);
    test_write_u32(fp, raw2);
}

static void test_write_iq4_xs_block(FILE *fp,
                                    uint16_t d,
                                    uint16_t scales_h,
                                    const uint8_t scales_l[4],
                                    const uint8_t qs[128]) {
    test_write_u16(fp, d);
    test_write_u16(fp, scales_h);
    TEST_ASSERT(fwrite(scales_l, 1, 4, fp) == 4);
    TEST_ASSERT(fwrite(qs, 1, 128, fp) == 128);
}

static void test_write_iq3_xxs_block(FILE *fp,
                                     uint16_t d,
                                     const uint8_t qs[64],
                                     const uint32_t aux[8]) {
    test_write_u16(fp, d);
    TEST_ASSERT(fwrite(qs, 1, 64, fp) == 64);
    for (int i = 0; i < 8; i++) {
        test_write_u32(fp, aux[i]);
    }
}

static void test_write_iq2_xs_block(FILE *fp,
                                    uint16_t d,
                                    const uint16_t qs[32],
                                    const uint8_t scales[8]) {
    test_write_u16(fp, d);
    for (int i = 0; i < 32; i++) {
        test_write_u16(fp, qs[i]);
    }
    TEST_ASSERT(fwrite(scales, 1, 8, fp) == 8);
}

static void test_write_iq2_s_block(FILE *fp,
                                   uint16_t d,
                                   const uint8_t qs[64],
                                   const uint8_t qh[8],
                                   const uint8_t scales[8]) {
    test_write_u16(fp, d);
    TEST_ASSERT(fwrite(qs, 1, 64, fp) == 64);
    TEST_ASSERT(fwrite(qh, 1, 8, fp) == 8);
    TEST_ASSERT(fwrite(scales, 1, 8, fp) == 8);
}

static void test_write_iq3_s_block(FILE *fp,
                                   uint16_t d,
                                   const uint8_t qs[64],
                                   const uint8_t qh[8],
                                   const uint8_t signs[32],
                                   const uint8_t scales[4]) {
    test_write_u16(fp, d);
    TEST_ASSERT(fwrite(qs, 1, 64, fp) == 64);
    TEST_ASSERT(fwrite(qh, 1, 8, fp) == 8);
    TEST_ASSERT(fwrite(signs, 1, 32, fp) == 32);
    TEST_ASSERT(fwrite(scales, 1, 4, fp) == 4);
}

static void test_write_iq1_s_block(FILE *fp,
                                   uint16_t d,
                                   const uint8_t qs[32],
                                   const uint16_t qh[8]) {
    test_write_u16(fp, d);
    TEST_ASSERT(fwrite(qs, 1, 32, fp) == 32);
    for (int i = 0; i < 8; i++) {
        test_write_u16(fp, qh[i]);
    }
}

static void test_write_iq1_m_block(FILE *fp,
                                   const uint8_t qs[32],
                                   const uint8_t qh[16],
                                   const uint16_t scales[4]) {
    TEST_ASSERT(fwrite(qs, 1, 32, fp) == 32);
    TEST_ASSERT(fwrite(qh, 1, 16, fp) == 16);
    for (int i = 0; i < 4; i++) {
        test_write_u16(fp, scales[i]);
    }
}

static void test_write_iq2_xxs_block(FILE *fp,
                                      uint16_t d,
                                      uint32_t aux_g0,
                                      uint32_t aux_s0) {
    test_write_u16(fp, d);
    test_write_u32(fp, aux_g0);
    test_write_u32(fp, aux_s0);
    for (int i = 1; i < 8; i++) {
        test_write_u32(fp, 0);
        test_write_u32(fp, 0);
    }
}

static void test_write_qwen36_tensor_1d(FILE *fp,
                                        const char *name,
                                        uint64_t dim,
                                        uint64_t *rel_offset) {
    uint64_t bytes = 0;
    TEST_ASSERT(rt_gguf_tensor_nbytes(1, dim, &bytes));
    test_write_gguf_tensor_1d(fp, name, dim, 1, *rel_offset);
    *rel_offset += bytes;
}

static void test_write_qwen36_tensor_2d(FILE *fp,
                                        const char *name,
                                        uint64_t dim0,
                                        uint64_t dim1,
                                        uint64_t *rel_offset) {
    uint64_t elements = 0;
    uint64_t bytes = 0;
    TEST_ASSERT(dim0 == 0 || dim1 <= UINT64_MAX / dim0);
    elements = dim0 * dim1;
    TEST_ASSERT(rt_gguf_tensor_nbytes(1, elements, &bytes));
    test_write_gguf_tensor_2d(fp, name, dim0, dim1, 1, *rel_offset);
    *rel_offset += bytes;
}

static void test_write_qwen36_tensor_4d(FILE *fp,
                                        const char *name,
                                        uint64_t dim0,
                                        uint64_t dim1,
                                        uint64_t dim2,
                                        uint64_t dim3,
                                        uint64_t *rel_offset) {
    uint64_t elements = dim0;
    uint64_t bytes = 0;
    TEST_ASSERT(dim1 == 0 || elements <= UINT64_MAX / dim1);
    elements *= dim1;
    TEST_ASSERT(dim2 == 0 || elements <= UINT64_MAX / dim2);
    elements *= dim2;
    TEST_ASSERT(dim3 == 0 || elements <= UINT64_MAX / dim3);
    elements *= dim3;
    TEST_ASSERT(rt_gguf_tensor_nbytes(1, elements, &bytes));
    test_write_gguf_tensor_4d(fp, name, dim0, dim1, dim2, dim3, 1, *rel_offset);
    *rel_offset += bytes;
}

static void test_pad_gguf_data(FILE *fp, uint64_t alignment) {
    long pos = ftell(fp);
    TEST_ASSERT(pos >= 0);
    while (((uint64_t)pos % alignment) != 0) {
        TEST_ASSERT(fputc(0, fp) != EOF);
        pos++;
    }
}

static uint64_t test_current_file_offset(FILE *fp) {
    long pos = ftell(fp);
    TEST_ASSERT(pos >= 0);
    return (uint64_t)pos;
}

#define QWEN36_TEST_HIDDEN 5120u
#define QWEN36_TEST_VOCAB 248320u
#define QWEN36_TEST_FFN 17408u
#define QWEN36_TEST_LAYERS 64u
#define QWEN36_TEST_FULL_INTERVAL 4u
#define QWEN36_TEST_HEAD_DIM 256u
#define QWEN36_TEST_HEADS 24u
#define QWEN36_TEST_KV_HEADS 4u
#define QWEN36_TEST_LINEAR_QK_HEADS 16u
#define QWEN36_TEST_LINEAR_V_HEADS 48u
#define QWEN36_TEST_LINEAR_HEAD_DIM 128u
#define QWEN36_TEST_SSM_D_CONV 4u
#define QWEN36_TEST_SSM_INNER (QWEN36_TEST_LINEAR_V_HEADS * QWEN36_TEST_LINEAR_HEAD_DIM)
#define QWEN36_TEST_SSM_QKV ((QWEN36_TEST_LINEAR_QK_HEADS * QWEN36_TEST_LINEAR_HEAD_DIM * 2u) + QWEN36_TEST_SSM_INNER)
#define QWEN36_TEST_FULL_LAYERS (QWEN36_TEST_LAYERS / QWEN36_TEST_FULL_INTERVAL)
#define QWEN36_TEST_RECURRENT_LAYERS (QWEN36_TEST_LAYERS - QWEN36_TEST_FULL_LAYERS)
#define QWEN36_TEST_TENSORS \
    (2u + QWEN36_TEST_LAYERS * 5u + QWEN36_TEST_FULL_LAYERS * 6u + \
     QWEN36_TEST_RECURRENT_LAYERS * 9u)
#define QWEN36_TEST_TENSORS_WITH_OUTPUT (QWEN36_TEST_TENSORS + 1u)
#define QWEN36_TEST_VISION_EMBD 1152u
#define QWEN36_TEST_VISION_FFN 4304u
#define QWEN36_TEST_VISION_LAYERS 3u
#define QWEN36_TEST_VISION_HEADS 16u
#define QWEN36_TEST_VISION_IMAGE 448u
#define QWEN36_TEST_VISION_PATCH 16u
#define QWEN36_TEST_VISION_MERGE 2u
#define QWEN36_TEST_MMPROJ_TENSORS (2u + QWEN36_TEST_VISION_LAYERS * 7u + 4u)
#define QWEN36_TEST_MTP_TENSORS 15u
#define QWEN36_TEST_SESSION_MAGIC 0x36335751u
#define QWEN36_TEST_SESSION_LEGACY_VERSION 1u
#define QWEN36_TEST_SESSION_V2_VERSION 2u
#define QWEN36_TEST_SESSION_VERSION 3u
#define QWEN36_TEST_SESSION_LEGACY_WORDS 12u
#define QWEN36_TEST_SESSION_WORDS 16u
#define QWEN36_TEST_SESSION_SECTION_WORDS 4u
#define QWEN36_TEST_TOKEN_COUNT 109u
#define QWEN36_TEST_VISION_START_TOKEN 18
#define QWEN36_TEST_VISION_END_TOKEN 19
#define QWEN36_TEST_IMAGE_PAD_TOKEN 20
#define QWEN36_TEST_VIDEO_PAD_TOKEN 21

static bool test_tokens_contain_triplet(const int *v, int len, int a, int b, int c) {
    for (int i = 0; i + 2 < len; i++) {
        if (v[i] == a && v[i + 1] == b && v[i + 2] == c) return true;
    }
    return false;
}

static uint64_t test_qwen36_full_kv_bytes(uint32_t ctx) {
    return (uint64_t)QWEN36_TEST_FULL_LAYERS * ctx * 2u *
           QWEN36_TEST_KV_HEADS * QWEN36_TEST_HEAD_DIM * sizeof(float);
}

static uint64_t test_qwen36_recurrent_state_bytes(void) {
    return (uint64_t)QWEN36_TEST_RECURRENT_LAYERS * QWEN36_TEST_LINEAR_HEAD_DIM *
           QWEN36_TEST_SSM_INNER * sizeof(float);
}

static uint64_t test_qwen36_conv_state_bytes(void) {
    return (uint64_t)QWEN36_TEST_RECURRENT_LAYERS * QWEN36_TEST_SSM_D_CONV *
           QWEN36_TEST_SSM_QKV * sizeof(float);
}

static void test_qwen36_assert_zero_last_hidden(const rt_session *session) {
    float *hidden = calloc(QWEN36_TEST_HIDDEN, sizeof(hidden[0]));
    TEST_ASSERT(hidden != NULL);
    if (!hidden) return;
    TEST_ASSERT(qwen36_runtime_session_read_last_hidden(session, hidden,
                                                       QWEN36_TEST_HIDDEN));
    for (uint32_t i = 0; i < QWEN36_TEST_HIDDEN; i++) {
        TEST_ASSERT(hidden[i] == 0.0f);
    }
    free(hidden);
}

static float test_qwen36_read_first_last_hidden(const rt_session *session) {
    float *hidden = calloc(QWEN36_TEST_HIDDEN, sizeof(hidden[0]));
    TEST_ASSERT(hidden != NULL);
    if (!hidden) return 0.0f;
    TEST_ASSERT(qwen36_runtime_session_read_last_hidden(session, hidden,
                                                       QWEN36_TEST_HIDDEN));
    float first = hidden[0];
    free(hidden);
    return first;
}

static void test_format_qwen36_tensor_name(char *dst, size_t dstlen, const char *fmt, uint32_t il) {
    int n = snprintf(dst, dstlen, fmt, il);
    TEST_ASSERT(n > 0 && (size_t)n < dstlen);
}

static void test_write_qwen36_mtp_tensors_at(FILE *fp,
                                             uint64_t *rel_offset,
                                             uint32_t layer_index,
                                             bool corrupt_shape);

static void test_write_qwen36_required_tensors_ex(FILE *fp,
                                                  uint64_t *payload_bytes,
                                                  bool corrupt_shape,
                                                  bool include_output) {
    char name[128];
    uint64_t rel = 0;

    test_write_qwen36_tensor_2d(fp, "token_embd.weight",
                                QWEN36_TEST_HIDDEN, QWEN36_TEST_VOCAB,
                                &rel);
    if (include_output) {
        test_write_qwen36_tensor_2d(fp, "output.weight", QWEN36_TEST_HIDDEN,
                                    QWEN36_TEST_VOCAB, &rel);
    }
    test_write_qwen36_tensor_1d(fp, "output_norm.weight", QWEN36_TEST_HIDDEN, &rel);

    for (uint32_t il = 0; il < QWEN36_TEST_LAYERS; il++) {
        test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_norm.weight", il);
        test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_HIDDEN, &rel);
        test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.post_attention_norm.weight", il);
        test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_HIDDEN, &rel);
        test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ffn_gate.weight", il);
        test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN, QWEN36_TEST_FFN, &rel);
        test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ffn_down.weight", il);
        test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_FFN, QWEN36_TEST_HIDDEN, &rel);
        test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ffn_up.weight", il);
        test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN, QWEN36_TEST_FFN, &rel);

        if ((il + 1u) % QWEN36_TEST_FULL_INTERVAL == 0) {
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_q.weight", il);
            test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN,
                                        QWEN36_TEST_HEAD_DIM * QWEN36_TEST_HEADS * 2u, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_k.weight", il);
            test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN,
                                        QWEN36_TEST_HEAD_DIM * QWEN36_TEST_KV_HEADS, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_v.weight", il);
            test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN,
                                        QWEN36_TEST_HEAD_DIM * QWEN36_TEST_KV_HEADS, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_output.weight", il);
            test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HEAD_DIM * QWEN36_TEST_HEADS,
                                        QWEN36_TEST_HIDDEN, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_q_norm.weight", il);
            test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_HEAD_DIM, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_k_norm.weight", il);
            test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_HEAD_DIM, &rel);
        } else {
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_qkv.weight", il);
            test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN, QWEN36_TEST_SSM_QKV, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_gate.weight", il);
            test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN, QWEN36_TEST_SSM_INNER, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ssm_conv1d.weight", il);
            test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_SSM_D_CONV, QWEN36_TEST_SSM_QKV, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ssm_dt.bias", il);
            test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_LINEAR_V_HEADS, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ssm_a", il);
            test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_LINEAR_V_HEADS, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ssm_beta.weight", il);
            test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN, QWEN36_TEST_LINEAR_V_HEADS, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ssm_alpha.weight", il);
            test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN, QWEN36_TEST_LINEAR_V_HEADS, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ssm_norm.weight", il);
            test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_LINEAR_HEAD_DIM, &rel);
            test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ssm_out.weight", il);
            test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_SSM_INNER,
                                        corrupt_shape && il == 0 ? QWEN36_TEST_HIDDEN - 1u : QWEN36_TEST_HIDDEN,
                                        &rel);
        }
    }

    *payload_bytes = rel;
}

static void test_write_qwen36_required_tensors(FILE *fp,
                                               uint64_t *payload_bytes,
                                               bool corrupt_shape) {
    test_write_qwen36_required_tensors_ex(fp, payload_bytes, corrupt_shape, false);
}

static void test_write_qwen36_probe_gguf_ex3(const char *path,
                                             const char *name,
                                             bool corrupt_shape,
                                             bool chat_vocab,
                                             bool include_output,
                                             bool include_embedded_mtp) {
    static const char *const qwen_base_tokens[] = {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "h",
        "i",
        "hi",
        "u",
        "s",
        "e",
        "r",
        "a",
        "t",
        "n",
        "\xC4\x8A",
        "\xC4\xA0",
        "m",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "b", "c", "d", "f", "g", "j", "k", "l", "o", "p", "q", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
        ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
        "23", "'s", "!\xC4\x8A",
    };
    static const char *const qwen_chat_tokens[] = {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "h",
        "i",
        "hi",
        "u",
        "s",
        "e",
        "r",
        "a",
        "t",
        "n",
        "\xC4\x8A",
        "\xC4\xA0",
        "m",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "b", "c", "d", "f", "g", "j", "k", "l", "o", "p", "q", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
        ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
        "23", "'s", "!\xC4\x8A",
    };
    static const char *const qwen_merges[] = {
        "h i",
        "2 3",
        "' s",
        "! \xC4\x8A",
    };

    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    const char *const *qwen_tokens = chat_vocab ? qwen_chat_tokens : qwen_base_tokens;
    size_t qwen_token_count = chat_vocab
        ? sizeof(qwen_chat_tokens) / sizeof(qwen_chat_tokens[0])
        : sizeof(qwen_base_tokens) / sizeof(qwen_base_tokens[0]);
    uint64_t n_tensors = include_output ? QWEN36_TEST_TENSORS_WITH_OUTPUT
                                        : QWEN36_TEST_TENSORS;
    if (include_embedded_mtp) n_tensors += QWEN36_TEST_MTP_TENSORS;
    test_write_gguf_header(fp, n_tensors, include_embedded_mtp ? 18 : 17);
    test_write_gguf_kv_string(fp, "general.architecture", "qwen35");
    test_write_gguf_kv_string(fp, "general.name", name);
    test_write_gguf_kv_u32(fp, "qwen35.block_count",
                           QWEN36_TEST_LAYERS + (include_embedded_mtp ? 1u : 0u));
    if (include_embedded_mtp) {
        test_write_gguf_kv_u32(fp, "qwen35.nextn_predict_layers", 1);
    }
    test_write_gguf_kv_u32(fp, "qwen35.embedding_length", 5120);
    test_write_gguf_kv_u32(fp, "qwen35.feed_forward_length", 17408);
    test_write_gguf_kv_u32(fp, "qwen35.attention.head_count", 24);
    test_write_gguf_kv_u32(fp, "qwen35.attention.head_count_kv", 4);
    test_write_gguf_kv_u32(fp, "qwen35.context_length", 262144);
    test_write_gguf_kv_u32(fp, "qwen35.full_attention_interval", 4);
    test_write_gguf_kv_u32(fp, "qwen35.ssm.conv_kernel", QWEN36_TEST_SSM_D_CONV);
    test_write_gguf_kv_u32(fp, "qwen35.ssm.inner_size", QWEN36_TEST_SSM_INNER);
    test_write_gguf_kv_u32(fp, "qwen35.ssm.state_size", QWEN36_TEST_LINEAR_HEAD_DIM);
    test_write_gguf_kv_u32(fp, "qwen35.ssm.time_step_rank", QWEN36_TEST_LINEAR_V_HEADS);
    test_write_gguf_kv_u32(fp, "qwen35.ssm.group_count", QWEN36_TEST_LINEAR_QK_HEADS);
    test_write_gguf_kv_string_array(fp, "tokenizer.ggml.tokens", qwen_tokens,
                                    qwen_token_count);
    test_write_gguf_kv_string_array(fp, "tokenizer.ggml.merges", qwen_merges,
                                    sizeof(qwen_merges) / sizeof(qwen_merges[0]));
    test_write_gguf_kv_u32(fp, "tokenizer.ggml.eos_token_id", 0);
    uint64_t payload_bytes = 0;
    test_write_qwen36_required_tensors_ex(fp, &payload_bytes, corrupt_shape,
                                          include_output);
    if (include_embedded_mtp) {
        test_write_qwen36_mtp_tensors_at(fp, &payload_bytes,
                                         QWEN36_TEST_LAYERS, false);
    }
    test_pad_gguf_data(fp, 32);
    uint64_t data_start = test_current_file_offset(fp);
    TEST_ASSERT(data_start <= (uint64_t)LLONG_MAX);
    TEST_ASSERT(payload_bytes <= (uint64_t)LLONG_MAX - data_start);
    TEST_ASSERT(fflush(fp) == 0);
    TEST_ASSERT(ftruncate(fileno(fp), (off_t)(data_start + payload_bytes)) == 0);
    TEST_ASSERT(fclose(fp) == 0);
}

static void test_write_qwen36_probe_gguf_ex2(const char *path,
                                             const char *name,
                                             bool corrupt_shape,
                                             bool chat_vocab) {
    test_write_qwen36_probe_gguf_ex3(path, name, corrupt_shape, chat_vocab, false, false);
}

static void test_write_qwen36_probe_gguf_ex(const char *path, const char *name, bool corrupt_shape) {
    test_write_qwen36_probe_gguf_ex2(path, name, corrupt_shape, false);
}

static void test_write_qwen36_probe_gguf(const char *path, const char *name) {
    test_write_qwen36_probe_gguf_ex(path, name, false);
}

static void test_write_qwen36_chat_probe_gguf(const char *path, const char *name) {
    test_write_qwen36_probe_gguf_ex2(path, name, false, true);
}

static void test_write_qwen36_output_probe_gguf(const char *path, const char *name) {
    test_write_qwen36_probe_gguf_ex3(path, name, false, false, true, false);
}

static void test_write_qwen36_embedded_mtp_probe_gguf(const char *path,
                                                      const char *name) {
    test_write_qwen36_probe_gguf_ex3(path, name, false, false, false, true);
}

static void test_write_qwen36_mmproj_tensors(FILE *fp, uint64_t *payload_bytes, bool corrupt_shape) {
    char name[128];
    uint64_t rel = 0;

    test_write_qwen36_tensor_4d(fp, "v.patch_embd.weight",
                                QWEN36_TEST_VISION_PATCH,
                                QWEN36_TEST_VISION_PATCH,
                                3,
                                QWEN36_TEST_VISION_EMBD,
                                &rel);
    test_write_qwen36_tensor_1d(fp, "v.patch_embd.bias", QWEN36_TEST_VISION_EMBD, &rel);

    for (uint32_t il = 0; il < QWEN36_TEST_VISION_LAYERS; il++) {
        test_format_qwen36_tensor_name(name, sizeof(name), "v.blk.%u.ln1.weight", il);
        test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_VISION_EMBD, &rel);
        test_format_qwen36_tensor_name(name, sizeof(name), "v.blk.%u.ln2.weight", il);
        test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_VISION_EMBD, &rel);
        test_format_qwen36_tensor_name(name, sizeof(name), "v.blk.%u.attn_qkv.weight", il);
        test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_VISION_EMBD,
                                    QWEN36_TEST_VISION_EMBD * 3u, &rel);
        test_format_qwen36_tensor_name(name, sizeof(name), "v.blk.%u.attn_out.weight", il);
        test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_VISION_EMBD,
                                    QWEN36_TEST_VISION_EMBD, &rel);
        test_format_qwen36_tensor_name(name, sizeof(name), "v.blk.%u.ffn_up.weight", il);
        test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_VISION_EMBD,
                                    QWEN36_TEST_VISION_FFN, &rel);
        test_format_qwen36_tensor_name(name, sizeof(name), "v.blk.%u.ffn_down.weight", il);
        test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_VISION_FFN,
                                    QWEN36_TEST_VISION_EMBD, &rel);
        test_format_qwen36_tensor_name(name, sizeof(name), "v.blk.%u.ffn_gate.weight", il);
        test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_VISION_EMBD,
                                    QWEN36_TEST_VISION_FFN, &rel);
    }

    test_write_qwen36_tensor_2d(fp, "mm.0.weight",
                                QWEN36_TEST_VISION_EMBD * QWEN36_TEST_VISION_MERGE * QWEN36_TEST_VISION_MERGE,
                                QWEN36_TEST_VISION_EMBD, &rel);
    test_write_qwen36_tensor_1d(fp, "mm.0.bias", QWEN36_TEST_VISION_EMBD, &rel);
    test_write_qwen36_tensor_2d(fp, "mm.2.weight", QWEN36_TEST_VISION_EMBD,
                                QWEN36_TEST_HIDDEN, &rel);
    test_write_qwen36_tensor_1d(fp, "mm.2.bias",
                                corrupt_shape ? QWEN36_TEST_HIDDEN - 1u : QWEN36_TEST_HIDDEN,
                                &rel);

    *payload_bytes = rel;
}

static void test_write_qwen36_mmproj_gguf(const char *path, bool corrupt_shape) {
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    test_write_gguf_header(fp, QWEN36_TEST_MMPROJ_TENSORS, 10);
    test_write_gguf_kv_bool(fp, "clip.has_vision_encoder", true);
    test_write_gguf_kv_string(fp, "clip.vision.projector_type", "qwen3vl_merger");
    test_write_gguf_kv_u32(fp, "clip.vision.embedding_length", QWEN36_TEST_VISION_EMBD);
    test_write_gguf_kv_u32(fp, "clip.vision.feed_forward_length", QWEN36_TEST_VISION_FFN);
    test_write_gguf_kv_u32(fp, "clip.vision.block_count", QWEN36_TEST_VISION_LAYERS);
    test_write_gguf_kv_u32(fp, "clip.vision.projection_dim", QWEN36_TEST_HIDDEN);
    test_write_gguf_kv_u32(fp, "clip.vision.attention.head_count", QWEN36_TEST_VISION_HEADS);
    test_write_gguf_kv_u32(fp, "clip.vision.image_size", QWEN36_TEST_VISION_IMAGE);
    test_write_gguf_kv_u32(fp, "clip.vision.patch_size", QWEN36_TEST_VISION_PATCH);
    test_write_gguf_kv_u32(fp, "clip.vision.spatial_merge_size", QWEN36_TEST_VISION_MERGE);
    uint64_t payload_bytes = 0;
    test_write_qwen36_mmproj_tensors(fp, &payload_bytes, corrupt_shape);
    test_pad_gguf_data(fp, 32);
    uint64_t data_start = test_current_file_offset(fp);
    TEST_ASSERT(data_start <= (uint64_t)LLONG_MAX);
    TEST_ASSERT(payload_bytes <= (uint64_t)LLONG_MAX - data_start);
    TEST_ASSERT(fflush(fp) == 0);
    TEST_ASSERT(ftruncate(fileno(fp), (off_t)(data_start + payload_bytes)) == 0);
    TEST_ASSERT(fclose(fp) == 0);
}

static void test_write_qwen36_mtp_tensors_at(FILE *fp,
                                             uint64_t *rel_offset,
                                             uint32_t layer_index,
                                             bool corrupt_shape) {
    char name[128];
    uint32_t il = layer_index;

    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_norm.weight", il);
    test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_HIDDEN, rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.post_attention_norm.weight", il);
    test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_HIDDEN, rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ffn_gate.weight", il);
    test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN, QWEN36_TEST_FFN,
                                rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ffn_down.weight", il);
    test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_FFN, QWEN36_TEST_HIDDEN,
                                rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.ffn_up.weight", il);
    test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN, QWEN36_TEST_FFN,
                                rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_q.weight", il);
    test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN,
                                QWEN36_TEST_HEAD_DIM * QWEN36_TEST_HEADS * 2u,
                                rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_k.weight", il);
    test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN,
                                QWEN36_TEST_HEAD_DIM * QWEN36_TEST_KV_HEADS,
                                rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_v.weight", il);
    test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HIDDEN,
                                QWEN36_TEST_HEAD_DIM * QWEN36_TEST_KV_HEADS,
                                rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_output.weight", il);
    test_write_qwen36_tensor_2d(fp, name, QWEN36_TEST_HEAD_DIM * QWEN36_TEST_HEADS,
                                QWEN36_TEST_HIDDEN, rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_q_norm.weight", il);
    test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_HEAD_DIM, rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.attn_k_norm.weight", il);
    test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_HEAD_DIM, rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.nextn.eh_proj.weight", il);
    test_write_qwen36_tensor_2d(fp, name,
                                QWEN36_TEST_HIDDEN * 2u,
                                QWEN36_TEST_HIDDEN, rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.nextn.shared_head_norm.weight", il);
    test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_HIDDEN, rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.nextn.enorm.weight", il);
    test_write_qwen36_tensor_1d(fp, name, QWEN36_TEST_HIDDEN, rel_offset);
    test_format_qwen36_tensor_name(name, sizeof(name), "blk.%u.nextn.hnorm.weight", il);
    test_write_qwen36_tensor_1d(fp, name,
                                corrupt_shape ? QWEN36_TEST_HIDDEN - 1u : QWEN36_TEST_HIDDEN,
                                rel_offset);
}

static void test_write_qwen36_mtp_tensors(FILE *fp,
                                          uint64_t *payload_bytes,
                                          bool corrupt_shape) {
    uint64_t rel = 0;

    test_write_qwen36_mtp_tensors_at(fp, &rel, QWEN36_TEST_LAYERS,
                                     corrupt_shape);
    *payload_bytes = rel;
}

static void test_write_qwen36_mtp_tensors_layers(FILE *fp,
                                                 uint64_t *payload_bytes,
                                                 uint32_t layers,
                                                 bool corrupt_shape) {
    uint64_t rel = 0;
    for (uint32_t i = 0; i < layers; i++) {
        test_write_qwen36_mtp_tensors_at(fp, &rel, QWEN36_TEST_LAYERS + i,
                                         corrupt_shape && i + 1u == layers);
    }
    *payload_bytes = rel;
}

static void test_write_qwen36_mtp_gguf(const char *path, bool corrupt_shape) {
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    test_write_gguf_header(fp, QWEN36_TEST_MTP_TENSORS, 0);
    uint64_t payload_bytes = 0;
    test_write_qwen36_mtp_tensors(fp, &payload_bytes, corrupt_shape);
    test_pad_gguf_data(fp, 32);
    uint64_t data_start = test_current_file_offset(fp);
    TEST_ASSERT(data_start <= (uint64_t)LLONG_MAX);
    TEST_ASSERT(payload_bytes <= (uint64_t)LLONG_MAX - data_start);
    TEST_ASSERT(fflush(fp) == 0);
    TEST_ASSERT(ftruncate(fileno(fp), (off_t)(data_start + payload_bytes)) == 0);
    TEST_ASSERT(fclose(fp) == 0);
}

static void test_write_qwen36_mtp_gguf_layers(const char *path,
                                              uint32_t layers,
                                              bool corrupt_shape) {
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    TEST_ASSERT(layers > 0);
    test_write_gguf_header(fp, (uint64_t)QWEN36_TEST_MTP_TENSORS * layers, 0);
    uint64_t payload_bytes = 0;
    test_write_qwen36_mtp_tensors_layers(fp, &payload_bytes, layers,
                                         corrupt_shape);
    test_pad_gguf_data(fp, 32);
    uint64_t data_start = test_current_file_offset(fp);
    TEST_ASSERT(data_start <= (uint64_t)LLONG_MAX);
    TEST_ASSERT(payload_bytes <= (uint64_t)LLONG_MAX - data_start);
    TEST_ASSERT(fflush(fp) == 0);
    TEST_ASSERT(ftruncate(fileno(fp), (off_t)(data_start + payload_bytes)) == 0);
    TEST_ASSERT(fclose(fp) == 0);
}

static uint64_t test_write_qwen36_logits_payload(FILE *fp, uint32_t ctx, uint32_t token) {
    TEST_ASSERT(fp != NULL);
    test_write_u32(fp, QWEN36_TEST_SESSION_MAGIC);
    test_write_u32(fp, QWEN36_TEST_SESSION_LEGACY_VERSION);
    test_write_u32(fp, QWEN36_TEST_SESSION_LEGACY_WORDS);
    test_write_u32(fp, ctx);
    test_write_u32(fp, 1);
    test_write_u32(fp, 1);
    test_write_u32(fp, QWEN36_TEST_TOKEN_COUNT);
    test_write_u32(fp, QWEN36_TEST_HIDDEN);
    test_write_u32(fp, QWEN36_TEST_LAYERS);
    test_write_u32(fp, 1);
    test_write_u32(fp, 0);
    test_write_u32(fp, 0);
    test_write_u32(fp, token);

    for (uint32_t i = 0; i < QWEN36_TEST_TOKEN_COUNT; i++) {
        float logit = -4.0f - (float)i * 0.125f;
        if (i == 4) logit = 4.0f;
        if (i == 7) logit = 2.0f;
        if (i == 9) logit = 1.0f;
        test_write_f32(fp, logit);
    }
    TEST_ASSERT(fflush(fp) == 0);
    return QWEN36_TEST_SESSION_LEGACY_WORDS * sizeof(uint32_t) + sizeof(uint32_t) +
           QWEN36_TEST_TOKEN_COUNT * sizeof(float);
}

static uint64_t test_write_qwen36_full_kv_payload_version(
        FILE *fp, uint32_t ctx, uint32_t token, uint32_t version) {
    TEST_ASSERT(fp != NULL);
    uint64_t full_kv_bytes = test_qwen36_full_kv_bytes(ctx);
    uint64_t recurrent_bytes = test_qwen36_recurrent_state_bytes();
    uint64_t conv_bytes = test_qwen36_conv_state_bytes();

    test_write_u32(fp, QWEN36_TEST_SESSION_MAGIC);
    test_write_u32(fp, version);
    test_write_u32(fp, QWEN36_TEST_SESSION_WORDS);
    test_write_u32(fp, ctx);
    test_write_u32(fp, 1);
    test_write_u32(fp, 1);
    test_write_u32(fp, QWEN36_TEST_TOKEN_COUNT);
    test_write_u32(fp, QWEN36_TEST_HIDDEN);
    test_write_u32(fp, QWEN36_TEST_LAYERS);
    test_write_u32(fp, 2);
    test_write_u32(fp, (uint32_t)(full_kv_bytes & 0xffffffffu));
    test_write_u32(fp, (uint32_t)(full_kv_bytes >> 32));
    test_write_u32(fp, (uint32_t)(recurrent_bytes & 0xffffffffu));
    test_write_u32(fp, (uint32_t)(recurrent_bytes >> 32));
    test_write_u32(fp, (uint32_t)(conv_bytes & 0xffffffffu));
    test_write_u32(fp, (uint32_t)(conv_bytes >> 32));

    test_write_u32(fp, 1);
    test_write_u32(fp, 0);
    test_write_u32(fp, sizeof(uint32_t));
    test_write_u32(fp, 0);
    test_write_u32(fp, 3);
    test_write_u32(fp, 0);
    test_write_u32(fp, (uint32_t)(full_kv_bytes & 0xffffffffu));
    test_write_u32(fp, (uint32_t)(full_kv_bytes >> 32));

    test_write_u32(fp, token);
    for (uint64_t i = 0; i < full_kv_bytes; i++) {
        TEST_ASSERT(fputc((int)(i & 0xffu), fp) != EOF);
    }
    TEST_ASSERT(fflush(fp) == 0);
    return QWEN36_TEST_SESSION_WORDS * sizeof(uint32_t) +
           2u * QWEN36_TEST_SESSION_SECTION_WORDS * sizeof(uint32_t) +
           sizeof(uint32_t) +
           full_kv_bytes;
}

static uint64_t test_write_qwen36_full_kv_payload(FILE *fp, uint32_t ctx, uint32_t token) {
    return test_write_qwen36_full_kv_payload_version(
        fp, ctx, token, QWEN36_TEST_SESSION_VERSION);
}

static void test_write_arch_probe_gguf(const char *path, const char *arch) {
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    test_write_gguf_header(fp, 0, 1);
    test_write_gguf_kv_string(fp, "general.architecture", arch);
    TEST_ASSERT(fclose(fp) == 0);
}

static void test_write_scalar_tensor_gguf(const char *path) {
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);

    test_write_gguf_header(fp, 46, 0);
    test_write_gguf_tensor_1d(fp, "f32_values", 3, 0, 0);
    test_write_gguf_tensor_1d(fp, "f16_values", 4, 1, 12);
    test_write_gguf_tensor_1d(fp, "bf16_values", 2, 30, 20);
    test_write_gguf_tensor_1d(fp, "q8_values", 32, 8, 24);
    test_write_gguf_tensor_2d(fp, "f16_matrix", 2, 2, 1, 58);
    test_write_gguf_tensor_1d(fp, "q4_k_values", 256, 12, 66);
    test_write_gguf_tensor_2d(fp, "q8_matrix", 32, 2, 8, 210);
    test_write_gguf_tensor_1d(fp, "q6_k_values", 256, 14, 278);
    test_write_gguf_tensor_1d(fp, "q4_0_values", 32, 2, 488);
    test_write_gguf_tensor_1d(fp, "q4_1_values", 32, 3, 506);
    test_write_gguf_tensor_1d(fp, "q5_0_values", 32, 6, 526);
    test_write_gguf_tensor_1d(fp, "q5_1_values", 32, 7, 548);
    test_write_gguf_tensor_1d(fp, "q8_1_values", 32, 9, 572);
    test_write_gguf_tensor_1d(fp, "q5_k_values", 256, 13, 608);
    test_write_gguf_tensor_1d(fp, "q8_k_values", 256, 15, 784);
    test_write_gguf_tensor_1d(fp, "q2_k_values", 256, 10, 1076);
    test_write_gguf_tensor_1d(fp, "q3_k_values", 256, 11, 1160);
    test_write_gguf_tensor_2d(fp, "q8_k_matrix", 256, 2, 15, 1270);
    test_write_gguf_tensor_2d(fp, "q4_k_matrix", 256, 2, 12, 1854);
    test_write_gguf_tensor_2d(fp, "q5_k_matrix", 256, 2, 13, 2142);
    test_write_gguf_tensor_2d(fp, "q6_k_matrix", 256, 2, 14, 2494);
    test_write_gguf_tensor_2d(fp, "q2_k_matrix", 256, 2, 10, 2914);
    test_write_gguf_tensor_2d(fp, "q3_k_matrix", 256, 2, 11, 3082);
    test_write_gguf_tensor_2d(fp, "q4_0_matrix", 32, 2, 2, 3302);
    test_write_gguf_tensor_2d(fp, "q4_1_matrix", 32, 2, 3, 3338);
    test_write_gguf_tensor_2d(fp, "q5_0_matrix", 32, 2, 6, 3378);
    test_write_gguf_tensor_2d(fp, "q5_1_matrix", 32, 2, 7, 3422);
    test_write_gguf_tensor_2d(fp, "q8_1_matrix", 32, 2, 9, 3470);
    test_write_gguf_tensor_1d(fp, "iq2_xxs_values", 256, 16, 3542);
    test_write_gguf_tensor_2d(fp, "iq2_xxs_matrix", 256, 2, 16, 3608);
    test_write_gguf_tensor_1d(fp, "iq4_nl_values", 32, 20, 3740);
    test_write_gguf_tensor_2d(fp, "iq4_nl_matrix", 32, 2, 20, 3758);
    test_write_gguf_tensor_1d(fp, "iq4_xs_values", 256, 23, 3794);
    test_write_gguf_tensor_2d(fp, "iq4_xs_matrix", 256, 2, 23, 3930);
    test_write_gguf_tensor_1d(fp, "iq3_xxs_values", 256, 18, 4202);
    test_write_gguf_tensor_2d(fp, "iq3_xxs_matrix", 256, 2, 18, 4300);
    test_write_gguf_tensor_1d(fp, "iq2_xs_values", 256, 17, 4496);
    test_write_gguf_tensor_2d(fp, "iq2_xs_matrix", 256, 2, 17, 4570);
    test_write_gguf_tensor_1d(fp, "iq2_s_values", 256, 22, 4718);
    test_write_gguf_tensor_2d(fp, "iq2_s_matrix", 256, 2, 22, 4800);
    test_write_gguf_tensor_1d(fp, "iq3_s_values", 256, 21, 4964);
    test_write_gguf_tensor_2d(fp, "iq3_s_matrix", 256, 2, 21, 5074);
    test_write_gguf_tensor_1d(fp, "iq1_s_values", 256, 19, 5294);
    test_write_gguf_tensor_2d(fp, "iq1_s_matrix", 256, 2, 19, 5344);
    test_write_gguf_tensor_1d(fp, "iq1_m_values", 256, 29, 5444);
    test_write_gguf_tensor_2d(fp, "iq1_m_matrix", 256, 2, 29, 5500);
    test_pad_gguf_data(fp, 32);

    test_write_f32(fp, 1.0f);
    test_write_f32(fp, -2.5f);
    test_write_f32(fp, 3.25f);
    test_write_u16(fp, 0x3c00u); /* 1.0 */
    test_write_u16(fp, 0xc000u); /* -2.0 */
    test_write_u16(fp, 0x3800u); /* 0.5 */
    test_write_u16(fp, 0x0001u); /* smallest positive subnormal */
    test_write_u16(fp, 0x3f80u); /* bf16 1.0 */
    test_write_u16(fp, 0xc020u); /* bf16 -2.5 */
    test_write_u16(fp, 0x3c00u); /* q8_0 scale 1.0 */
    for (int i = 0; i < 32; i++) {
        int8_t v = 0;
        if (i == 0) v = 1;
        if (i == 1) v = -2;
        if (i == 2) v = 3;
        TEST_ASSERT(fwrite(&v, 1, 1, fp) == 1);
    }
    test_write_u16(fp, 0x3c00u); /* 1.0 */
    test_write_u16(fp, 0x4000u); /* 2.0 */
    test_write_u16(fp, 0x4200u); /* 3.0 */
    test_write_u16(fp, 0x4400u); /* 4.0 */
    test_write_u16(fp, 0x3c00u); /* q4_K d = 1.0 */
    test_write_u16(fp, 0x0000u); /* q4_K dmin = 0.0 */
    uint8_t q4_scales[12] = {0};
    q4_scales[0] = 1;
    q4_scales[1] = 1;
    TEST_ASSERT(fwrite(q4_scales, 1, sizeof(q4_scales), fp) == sizeof(q4_scales));
    uint8_t q4_qs[128] = {0};
    q4_qs[0] = (uint8_t)(2u | (3u << 4));
    TEST_ASSERT(fwrite(q4_qs, 1, sizeof(q4_qs), fp) == sizeof(q4_qs));
    test_write_u16(fp, 0x3c00u); /* row 0 q8_0 scale 1.0 */
    for (int i = 0; i < 32; i++) {
        int8_t v = 0;
        if (i == 0) v = 1;
        if (i == 1) v = 2;
        TEST_ASSERT(fwrite(&v, 1, 1, fp) == 1);
    }
    test_write_u16(fp, 0x3c00u); /* row 1 q8_0 scale 1.0 */
    for (int i = 0; i < 32; i++) {
        int8_t v = 0;
        if (i == 0) v = -1;
        if (i == 1) v = 4;
        TEST_ASSERT(fwrite(&v, 1, 1, fp) == 1);
    }
    uint8_t q6_ql[128] = {0};
    uint8_t q6_qh[64] = {0};
    int8_t q6_scales[16] = {0};
    q6_ql[0] = 1;
    q6_ql[32] = 2;
    q6_qh[0] = (uint8_t)(2u | (2u << 2));
    q6_scales[0] = 1;
    q6_scales[2] = 1;
    TEST_ASSERT(fwrite(q6_ql, 1, sizeof(q6_ql), fp) == sizeof(q6_ql));
    TEST_ASSERT(fwrite(q6_qh, 1, sizeof(q6_qh), fp) == sizeof(q6_qh));
    TEST_ASSERT(fwrite(q6_scales, 1, sizeof(q6_scales), fp) == sizeof(q6_scales));
    test_write_u16(fp, 0x3c00u); /* q6_K d = 1.0 */
    test_write_u16(fp, 0x3c00u); /* q4_0 d = 1.0 */
    uint8_t q4_0_qs[16] = {0};
    q4_0_qs[0] = (uint8_t)(10u | (5u << 4));
    TEST_ASSERT(fwrite(q4_0_qs, 1, sizeof(q4_0_qs), fp) == sizeof(q4_0_qs));
    test_write_u16(fp, 0x3800u); /* q4_1 d = 0.5 */
    test_write_u16(fp, 0xbc00u); /* q4_1 m = -1.0 */
    uint8_t q4_1_qs[16] = {0};
    q4_1_qs[0] = (uint8_t)(4u | (1u << 4));
    TEST_ASSERT(fwrite(q4_1_qs, 1, sizeof(q4_1_qs), fp) == sizeof(q4_1_qs));
    test_write_u16(fp, 0x3c00u); /* q5_0 d = 1.0 */
    test_write_u32(fp, 1u);      /* high bit for element 0 */
    uint8_t q5_0_qs[16] = {0};
    q5_0_qs[0] = (uint8_t)(2u | (15u << 4));
    TEST_ASSERT(fwrite(q5_0_qs, 1, sizeof(q5_0_qs), fp) == sizeof(q5_0_qs));
    test_write_u16(fp, 0x3400u); /* q5_1 d = 0.25 */
    test_write_u16(fp, 0xc000u); /* q5_1 m = -2.0 */
    test_write_u32(fp, 1u);      /* high bit for element 0 */
    uint8_t q5_1_qs[16] = {0};
    q5_1_qs[0] = (uint8_t)(2u | (15u << 4));
    TEST_ASSERT(fwrite(q5_1_qs, 1, sizeof(q5_1_qs), fp) == sizeof(q5_1_qs));
    test_write_u16(fp, 0x3800u); /* q8_1 d = 0.5 */
    test_write_u16(fp, 0x0000u); /* q8_1 s = 0.0 */
    for (int i = 0; i < 32; i++) {
        int8_t v = 0;
        if (i == 0) v = 4;
        if (i == 1) v = -6;
        TEST_ASSERT(fwrite(&v, 1, 1, fp) == 1);
    }
    test_write_u16(fp, 0x3c00u); /* q5_K d = 1.0 */
    test_write_u16(fp, 0x0000u); /* q5_K dmin = 0.0 */
    uint8_t q5_k_scales[12] = {0};
    q5_k_scales[0] = 1;
    q5_k_scales[1] = 1;
    TEST_ASSERT(fwrite(q5_k_scales, 1, sizeof(q5_k_scales), fp) == sizeof(q5_k_scales));
    uint8_t q5_k_qh[32] = {0};
    q5_k_qh[0] = 1u;
    TEST_ASSERT(fwrite(q5_k_qh, 1, sizeof(q5_k_qh), fp) == sizeof(q5_k_qh));
    uint8_t q5_k_qs[128] = {0};
    q5_k_qs[0] = (uint8_t)(2u | (3u << 4));
    TEST_ASSERT(fwrite(q5_k_qs, 1, sizeof(q5_k_qs), fp) == sizeof(q5_k_qs));
    test_write_f32(fp, 0.5f); /* q8_K d */
    for (int i = 0; i < 256; i++) {
        int8_t v = 0;
        if (i == 0) v = 5;
        if (i == 255) v = -2;
        TEST_ASSERT(fwrite(&v, 1, 1, fp) == 1);
    }
    int16_t q8_k_sums[16] = {0};
    TEST_ASSERT(fwrite(q8_k_sums, 1, sizeof(q8_k_sums), fp) == sizeof(q8_k_sums));
    uint8_t q2_k_scales[16] = {0};
    q2_k_scales[0] = (uint8_t)(2u | (1u << 4));
    q2_k_scales[1] = (uint8_t)(1u | (2u << 4));
    TEST_ASSERT(fwrite(q2_k_scales, 1, sizeof(q2_k_scales), fp) == sizeof(q2_k_scales));
    uint8_t q2_k_qs[64] = {0};
    q2_k_qs[0] = 3u;
    q2_k_qs[16] = 1u;
    TEST_ASSERT(fwrite(q2_k_qs, 1, sizeof(q2_k_qs), fp) == sizeof(q2_k_qs));
    test_write_u16(fp, 0x3c00u); /* q2_K d = 1.0 */
    test_write_u16(fp, 0x3800u); /* q2_K dmin = 0.5 */
    uint8_t q3_k_hmask[32] = {0};
    q3_k_hmask[0] = 1u;
    q3_k_hmask[16] = 1u;
    TEST_ASSERT(fwrite(q3_k_hmask, 1, sizeof(q3_k_hmask), fp) == sizeof(q3_k_hmask));
    uint8_t q3_k_qs[64] = {0};
    q3_k_qs[0] = 3u;
    q3_k_qs[1] = 2u;
    q3_k_qs[16] = 1u;
    TEST_ASSERT(fwrite(q3_k_qs, 1, sizeof(q3_k_qs), fp) == sizeof(q3_k_qs));
    uint8_t q3_k_scales[12] = {0};
    q3_k_scales[0] = 1u;
    q3_k_scales[1] = 2u;
    q3_k_scales[8] = 2u;
    q3_k_scales[9] = 2u;
    TEST_ASSERT(fwrite(q3_k_scales, 1, sizeof(q3_k_scales), fp) == sizeof(q3_k_scales));
    test_write_u16(fp, 0x3c00u); /* q3_K d = 1.0 */
    test_write_f32(fp, 0.5f); /* row 0 q8_K d */
    for (int i = 0; i < 256; i++) {
        int8_t v = 0;
        if (i == 0) v = 5;
        if (i == 255) v = -2;
        TEST_ASSERT(fwrite(&v, 1, 1, fp) == 1);
    }
    int16_t q8_k_matrix_sums0[16] = {0};
    TEST_ASSERT(fwrite(q8_k_matrix_sums0, 1, sizeof(q8_k_matrix_sums0), fp) ==
                sizeof(q8_k_matrix_sums0));
    test_write_f32(fp, 1.0f); /* row 1 q8_K d */
    for (int i = 0; i < 256; i++) {
        int8_t v = 0;
        if (i == 0) v = -2;
        if (i == 1) v = 3;
        TEST_ASSERT(fwrite(&v, 1, 1, fp) == 1);
    }
    int16_t q8_k_matrix_sums1[16] = {0};
    TEST_ASSERT(fwrite(q8_k_matrix_sums1, 1, sizeof(q8_k_matrix_sums1), fp) ==
                sizeof(q8_k_matrix_sums1));
    test_write_u16(fp, 0x3c00u); /* row 0 q4_K d = 1.0 */
    test_write_u16(fp, 0x0000u); /* row 0 q4_K dmin = 0.0 */
    uint8_t q4_k_matrix_scales0[12] = {0};
    q4_k_matrix_scales0[0] = 1;
    q4_k_matrix_scales0[1] = 1;
    TEST_ASSERT(fwrite(q4_k_matrix_scales0, 1, sizeof(q4_k_matrix_scales0), fp) ==
                sizeof(q4_k_matrix_scales0));
    uint8_t q4_k_matrix_qs0[128] = {0};
    q4_k_matrix_qs0[0] = (uint8_t)(2u | (3u << 4));
    TEST_ASSERT(fwrite(q4_k_matrix_qs0, 1, sizeof(q4_k_matrix_qs0), fp) ==
                sizeof(q4_k_matrix_qs0));
    test_write_u16(fp, 0x3800u); /* row 1 q4_K d = 0.5 */
    test_write_u16(fp, 0x0000u); /* row 1 q4_K dmin = 0.0 */
    uint8_t q4_k_matrix_scales1[12] = {0};
    q4_k_matrix_scales1[0] = 1;
    q4_k_matrix_scales1[1] = 1;
    TEST_ASSERT(fwrite(q4_k_matrix_scales1, 1, sizeof(q4_k_matrix_scales1), fp) ==
                sizeof(q4_k_matrix_scales1));
    uint8_t q4_k_matrix_qs1[128] = {0};
    q4_k_matrix_qs1[0] = (uint8_t)(15u | (4u << 4));
    TEST_ASSERT(fwrite(q4_k_matrix_qs1, 1, sizeof(q4_k_matrix_qs1), fp) ==
                sizeof(q4_k_matrix_qs1));
    test_write_u16(fp, 0x3c00u); /* row 0 q5_K d = 1.0 */
    test_write_u16(fp, 0x0000u); /* row 0 q5_K dmin = 0.0 */
    uint8_t q5_k_matrix_scales0[12] = {0};
    q5_k_matrix_scales0[0] = 1;
    q5_k_matrix_scales0[1] = 1;
    TEST_ASSERT(fwrite(q5_k_matrix_scales0, 1, sizeof(q5_k_matrix_scales0), fp) ==
                sizeof(q5_k_matrix_scales0));
    uint8_t q5_k_matrix_qh0[32] = {0};
    q5_k_matrix_qh0[0] = 1u;
    TEST_ASSERT(fwrite(q5_k_matrix_qh0, 1, sizeof(q5_k_matrix_qh0), fp) ==
                sizeof(q5_k_matrix_qh0));
    uint8_t q5_k_matrix_qs0[128] = {0};
    q5_k_matrix_qs0[0] = (uint8_t)(2u | (3u << 4));
    TEST_ASSERT(fwrite(q5_k_matrix_qs0, 1, sizeof(q5_k_matrix_qs0), fp) ==
                sizeof(q5_k_matrix_qs0));
    test_write_u16(fp, 0x3800u); /* row 1 q5_K d = 0.5 */
    test_write_u16(fp, 0x0000u); /* row 1 q5_K dmin = 0.0 */
    uint8_t q5_k_matrix_scales1[12] = {0};
    q5_k_matrix_scales1[0] = 1;
    q5_k_matrix_scales1[1] = 1;
    TEST_ASSERT(fwrite(q5_k_matrix_scales1, 1, sizeof(q5_k_matrix_scales1), fp) ==
                sizeof(q5_k_matrix_scales1));
    uint8_t q5_k_matrix_qh1[32] = {0};
    q5_k_matrix_qh1[0] = 1u;
    TEST_ASSERT(fwrite(q5_k_matrix_qh1, 1, sizeof(q5_k_matrix_qh1), fp) ==
                sizeof(q5_k_matrix_qh1));
    uint8_t q5_k_matrix_qs1[128] = {0};
    q5_k_matrix_qs1[0] = (uint8_t)(4u | (2u << 4));
    TEST_ASSERT(fwrite(q5_k_matrix_qs1, 1, sizeof(q5_k_matrix_qs1), fp) ==
                sizeof(q5_k_matrix_qs1));
    uint8_t q6_k_matrix_ql0[128] = {0};
    uint8_t q6_k_matrix_qh0[64] = {0};
    int8_t q6_k_matrix_scales0[16] = {0};
    q6_k_matrix_ql0[0] = 1;
    q6_k_matrix_ql0[32] = 2;
    q6_k_matrix_qh0[0] = (uint8_t)(2u | (2u << 2));
    q6_k_matrix_scales0[0] = 1;
    q6_k_matrix_scales0[2] = 1;
    TEST_ASSERT(fwrite(q6_k_matrix_ql0, 1, sizeof(q6_k_matrix_ql0), fp) ==
                sizeof(q6_k_matrix_ql0));
    TEST_ASSERT(fwrite(q6_k_matrix_qh0, 1, sizeof(q6_k_matrix_qh0), fp) ==
                sizeof(q6_k_matrix_qh0));
    TEST_ASSERT(fwrite(q6_k_matrix_scales0, 1, sizeof(q6_k_matrix_scales0), fp) ==
                sizeof(q6_k_matrix_scales0));
    test_write_u16(fp, 0x3c00u); /* row 0 q6_K d = 1.0 */
    uint8_t q6_k_matrix_ql1[128] = {0};
    uint8_t q6_k_matrix_qh1[64] = {0};
    int8_t q6_k_matrix_scales1[16] = {0};
    q6_k_matrix_ql1[0] = 15;
    q6_k_matrix_ql1[32] = 3;
    q6_k_matrix_qh1[0] = (uint8_t)(1u | (2u << 2));
    q6_k_matrix_scales1[0] = 1;
    q6_k_matrix_scales1[2] = 1;
    TEST_ASSERT(fwrite(q6_k_matrix_ql1, 1, sizeof(q6_k_matrix_ql1), fp) ==
                sizeof(q6_k_matrix_ql1));
    TEST_ASSERT(fwrite(q6_k_matrix_qh1, 1, sizeof(q6_k_matrix_qh1), fp) ==
                sizeof(q6_k_matrix_qh1));
    TEST_ASSERT(fwrite(q6_k_matrix_scales1, 1, sizeof(q6_k_matrix_scales1), fp) ==
                sizeof(q6_k_matrix_scales1));
    test_write_u16(fp, 0x3c00u); /* row 1 q6_K d = 1.0 */
    uint8_t q2_k_matrix_scales0[16] = {0};
    q2_k_matrix_scales0[0] = 1u;
    TEST_ASSERT(fwrite(q2_k_matrix_scales0, 1, sizeof(q2_k_matrix_scales0), fp) ==
                sizeof(q2_k_matrix_scales0));
    uint8_t q2_k_matrix_qs0[64] = {0};
    q2_k_matrix_qs0[0] = 3u;
    TEST_ASSERT(fwrite(q2_k_matrix_qs0, 1, sizeof(q2_k_matrix_qs0), fp) ==
                sizeof(q2_k_matrix_qs0));
    test_write_u16(fp, 0x3c00u); /* row 0 q2_K d = 1.0 */
    test_write_u16(fp, 0x0000u); /* row 0 q2_K dmin = 0.0 */
    uint8_t q2_k_matrix_scales1[16] = {0};
    q2_k_matrix_scales1[0] = 2u;
    q2_k_matrix_scales1[2] = 1u;
    TEST_ASSERT(fwrite(q2_k_matrix_scales1, 1, sizeof(q2_k_matrix_scales1), fp) ==
                sizeof(q2_k_matrix_scales1));
    uint8_t q2_k_matrix_qs1[64] = {0};
    q2_k_matrix_qs1[0] = (uint8_t)(1u | (2u << 2));
    TEST_ASSERT(fwrite(q2_k_matrix_qs1, 1, sizeof(q2_k_matrix_qs1), fp) ==
                sizeof(q2_k_matrix_qs1));
    test_write_u16(fp, 0x3c00u); /* row 1 q2_K d = 1.0 */
    test_write_u16(fp, 0x0000u); /* row 1 q2_K dmin = 0.0 */
    uint8_t q3_k_matrix_hmask0[32] = {0};
    q3_k_matrix_hmask0[0] = 3u;
    q3_k_matrix_hmask0[1] = 1u;
    TEST_ASSERT(fwrite(q3_k_matrix_hmask0, 1, sizeof(q3_k_matrix_hmask0), fp) ==
                sizeof(q3_k_matrix_hmask0));
    uint8_t q3_k_matrix_qs0[64] = {0};
    q3_k_matrix_qs0[0] = (uint8_t)(3u | (1u << 2));
    TEST_ASSERT(fwrite(q3_k_matrix_qs0, 1, sizeof(q3_k_matrix_qs0), fp) ==
                sizeof(q3_k_matrix_qs0));
    uint8_t q3_k_matrix_scales0[16];
    for (int i = 0; i < 16; i++) q3_k_matrix_scales0[i] = 32u;
    q3_k_matrix_scales0[0] = 33u;
    q3_k_matrix_scales0[2] = 34u;
    test_write_q3_k_scale_bytes(fp, q3_k_matrix_scales0);
    test_write_u16(fp, 0x3c00u); /* row 0 q3_K d = 1.0 */
    uint8_t q3_k_matrix_hmask1[32] = {0};
    q3_k_matrix_hmask1[0] = 3u;
    q3_k_matrix_hmask1[1] = 1u;
    TEST_ASSERT(fwrite(q3_k_matrix_hmask1, 1, sizeof(q3_k_matrix_hmask1), fp) ==
                sizeof(q3_k_matrix_hmask1));
    uint8_t q3_k_matrix_qs1[64] = {0};
    q3_k_matrix_qs1[0] = (uint8_t)(1u | (2u << 2));
    TEST_ASSERT(fwrite(q3_k_matrix_qs1, 1, sizeof(q3_k_matrix_qs1), fp) ==
                sizeof(q3_k_matrix_qs1));
    uint8_t q3_k_matrix_scales1[16];
    for (int i = 0; i < 16; i++) q3_k_matrix_scales1[i] = 32u;
    q3_k_matrix_scales1[0] = 34u;
    q3_k_matrix_scales1[2] = 33u;
    test_write_q3_k_scale_bytes(fp, q3_k_matrix_scales1);
    test_write_u16(fp, 0x3c00u); /* row 1 q3_K d = 1.0 */
    test_write_u16(fp, 0x3c00u); /* row 0 q4_0 d = 1.0 */
    uint8_t q4_0_matrix_qs0[16] = {0};
    q4_0_matrix_qs0[0] = 10u; /* x[0] -> 2.0 */
    q4_0_matrix_qs0[1] = 9u;  /* x[1] -> 1.0 */
    TEST_ASSERT(fwrite(q4_0_matrix_qs0, 1, sizeof(q4_0_matrix_qs0), fp) ==
                sizeof(q4_0_matrix_qs0));
    test_write_u16(fp, 0x3800u); /* row 1 q4_0 d = 0.5 */
    uint8_t q4_0_matrix_qs1[16] = {0};
    q4_0_matrix_qs1[0] = 15u; /* x[0] -> 3.5 */
    q4_0_matrix_qs1[1] = 4u;  /* x[1] -> -2.0 */
    TEST_ASSERT(fwrite(q4_0_matrix_qs1, 1, sizeof(q4_0_matrix_qs1), fp) ==
                sizeof(q4_0_matrix_qs1));
    test_write_u16(fp, 0x3800u); /* row 0 q4_1 d = 0.5 */
    test_write_u16(fp, 0xbc00u); /* row 0 q4_1 m = -1.0 */
    uint8_t q4_1_matrix_qs0[16] = {0};
    q4_1_matrix_qs0[0] = 4u;  /* x[0] -> 1.0 */
    q4_1_matrix_qs0[1] = 10u; /* x[1] -> 4.0 */
    TEST_ASSERT(fwrite(q4_1_matrix_qs0, 1, sizeof(q4_1_matrix_qs0), fp) ==
                sizeof(q4_1_matrix_qs0));
    test_write_u16(fp, 0x3c00u); /* row 1 q4_1 d = 1.0 */
    test_write_u16(fp, 0xc000u); /* row 1 q4_1 m = -2.0 */
    uint8_t q4_1_matrix_qs1[16] = {0};
    q4_1_matrix_qs1[0] = 3u; /* x[0] -> 1.0 */
    q4_1_matrix_qs1[1] = 5u; /* x[1] -> 3.0 */
    TEST_ASSERT(fwrite(q4_1_matrix_qs1, 1, sizeof(q4_1_matrix_qs1), fp) ==
                sizeof(q4_1_matrix_qs1));
    test_write_u16(fp, 0x3c00u); /* row 0 q5_0 d = 1.0 */
    test_write_u32(fp, 1u);      /* high bit for element 0 */
    uint8_t q5_0_matrix_qs0[16] = {0};
    q5_0_matrix_qs0[0] = 2u;  /* x[0] -> 2.0 */
    q5_0_matrix_qs0[1] = 15u; /* x[1] -> -1.0 */
    TEST_ASSERT(fwrite(q5_0_matrix_qs0, 1, sizeof(q5_0_matrix_qs0), fp) ==
                sizeof(q5_0_matrix_qs0));
    test_write_u16(fp, 0x3800u); /* row 1 q5_0 d = 0.5 */
    test_write_u32(fp, 3u);      /* high bit for elements 0 and 1 */
    uint8_t q5_0_matrix_qs1[16] = {0};
    q5_0_matrix_qs1[0] = 15u; /* x[0] -> 7.5 */
    q5_0_matrix_qs1[1] = 0u;  /* x[1] -> 0.0 */
    TEST_ASSERT(fwrite(q5_0_matrix_qs1, 1, sizeof(q5_0_matrix_qs1), fp) ==
                sizeof(q5_0_matrix_qs1));
    test_write_u16(fp, 0x3400u); /* row 0 q5_1 d = 0.25 */
    test_write_u16(fp, 0xc000u); /* row 0 q5_1 m = -2.0 */
    test_write_u32(fp, 1u);      /* high bit for element 0 */
    uint8_t q5_1_matrix_qs0[16] = {0};
    q5_1_matrix_qs0[0] = 2u;  /* x[0] -> 2.5 */
    q5_1_matrix_qs0[1] = 15u; /* x[1] -> 1.75 */
    TEST_ASSERT(fwrite(q5_1_matrix_qs0, 1, sizeof(q5_1_matrix_qs0), fp) ==
                sizeof(q5_1_matrix_qs0));
    test_write_u16(fp, 0x3800u); /* row 1 q5_1 d = 0.5 */
    test_write_u16(fp, 0xbc00u); /* row 1 q5_1 m = -1.0 */
    test_write_u32(fp, 0u);
    uint8_t q5_1_matrix_qs1[16] = {0};
    q5_1_matrix_qs1[0] = 4u; /* x[0] -> 1.0 */
    q5_1_matrix_qs1[1] = 2u; /* x[1] -> 0.0 */
    TEST_ASSERT(fwrite(q5_1_matrix_qs1, 1, sizeof(q5_1_matrix_qs1), fp) ==
                sizeof(q5_1_matrix_qs1));
    test_write_u16(fp, 0x3800u); /* row 0 q8_1 d = 0.5 */
    test_write_u16(fp, 0x0000u); /* row 0 q8_1 sum = 0.0 */
    for (int i = 0; i < 32; i++) {
        int8_t v = 0;
        if (i == 0) v = 4;
        if (i == 1) v = -6;
        TEST_ASSERT(fwrite(&v, 1, 1, fp) == 1);
    }
    test_write_u16(fp, 0x3e00u); /* row 1 q8_1 d = 1.5 */
    test_write_u16(fp, 0x0000u); /* row 1 q8_1 sum = 0.0 */
    for (int i = 0; i < 32; i++) {
        int8_t v = 0;
        if (i == 0) v = -2;
        if (i == 1) v = 2;
        TEST_ASSERT(fwrite(&v, 1, 1, fp) == 1);
    }
    test_write_iq2_xxs_block(fp, 0x3c00u, 1u, 0u);
    test_write_iq2_xxs_block(fp, 0x3c00u, 1u, 0u);
    test_write_iq2_xxs_block(fp, 0x4000u, 0u, 3u);
    test_write_u16(fp, 0x3c00u); /* iq4_nl d = 1.0 */
    uint8_t iq4_nl_qs[16] = {0};
    iq4_nl_qs[0] = (uint8_t)(15u | (7u << 4));
    TEST_ASSERT(fwrite(iq4_nl_qs, 1, sizeof(iq4_nl_qs), fp) == sizeof(iq4_nl_qs));
    test_write_u16(fp, 0x3c00u); /* row 0 iq4_nl d = 1.0 */
    uint8_t iq4_nl_matrix_qs0[16] = {0};
    iq4_nl_matrix_qs0[0] = 8u; /* x[0] -> 1.0 */
    iq4_nl_matrix_qs0[1] = 9u; /* x[1] -> 13.0 */
    TEST_ASSERT(fwrite(iq4_nl_matrix_qs0, 1, sizeof(iq4_nl_matrix_qs0), fp) ==
                sizeof(iq4_nl_matrix_qs0));
    test_write_u16(fp, 0x3800u); /* row 1 iq4_nl d = 0.5 */
    uint8_t iq4_nl_matrix_qs1[16] = {0};
    iq4_nl_matrix_qs1[0] = 10u; /* x[0] -> 12.5 */
    iq4_nl_matrix_qs1[1] = 7u;  /* x[1] -> -5.0 */
    TEST_ASSERT(fwrite(iq4_nl_matrix_qs1, 1, sizeof(iq4_nl_matrix_qs1), fp) ==
                sizeof(iq4_nl_matrix_qs1));
    const uint8_t iq4_xs_scales_l[4] = {1u, 0u, 0u, 0u};
    uint8_t iq4_xs_qs[128] = {0};
    iq4_xs_qs[0] = (uint8_t)(15u | (7u << 4));
    test_write_iq4_xs_block(fp, 0x3c00u, 0xaaaau, iq4_xs_scales_l, iq4_xs_qs);
    uint8_t iq4_xs_matrix_qs0[128] = {0};
    iq4_xs_matrix_qs0[0] = 8u; /* x[0] -> 1.0 */
    iq4_xs_matrix_qs0[1] = 9u; /* x[1] -> 13.0 */
    test_write_iq4_xs_block(fp, 0x3c00u, 0xaaaau, iq4_xs_scales_l, iq4_xs_matrix_qs0);
    uint8_t iq4_xs_matrix_qs1[128] = {0};
    iq4_xs_matrix_qs1[0] = 10u; /* x[0] -> 12.5 */
    iq4_xs_matrix_qs1[1] = 7u;  /* x[1] -> -5.0 */
    test_write_iq4_xs_block(fp, 0x3800u, 0xaaaau, iq4_xs_scales_l, iq4_xs_matrix_qs1);
    uint8_t iq3_xxs_qs[64] = {0};
    iq3_xxs_qs[0] = 1u;
    uint32_t iq3_xxs_aux[8] = {0};
    test_write_iq3_xxs_block(fp, 0x3c00u, iq3_xxs_qs, iq3_xxs_aux);
    test_write_iq3_xxs_block(fp, 0x3c00u, iq3_xxs_qs, iq3_xxs_aux);
    uint32_t iq3_xxs_aux_neg[8] = {1u};
    test_write_iq3_xxs_block(fp, 0x3c00u, iq3_xxs_qs, iq3_xxs_aux_neg);
    uint16_t iq2_xs_qs[32] = {0};
    iq2_xs_qs[0] = 1u;
    uint8_t iq2_xs_scales[8] = {0};
    test_write_iq2_xs_block(fp, 0x3c00u, iq2_xs_qs, iq2_xs_scales);
    test_write_iq2_xs_block(fp, 0x3c00u, iq2_xs_qs, iq2_xs_scales);
    uint16_t iq2_xs_qs_neg[32] = {0};
    iq2_xs_qs_neg[0] = (uint16_t)(1u | (1u << 9));
    test_write_iq2_xs_block(fp, 0x3c00u, iq2_xs_qs_neg, iq2_xs_scales);
    uint8_t iq2_s_qs[64] = {0};
    iq2_s_qs[0] = 1u;
    uint8_t iq2_s_qh[8] = {0};
    uint8_t iq2_s_scales[8] = {0};
    test_write_iq2_s_block(fp, 0x3c00u, iq2_s_qs, iq2_s_qh, iq2_s_scales);
    test_write_iq2_s_block(fp, 0x3c00u, iq2_s_qs, iq2_s_qh, iq2_s_scales);
    uint8_t iq2_s_qs_neg[64] = {0};
    iq2_s_qs_neg[0] = 1u;
    iq2_s_qs_neg[32] = 1u;
    test_write_iq2_s_block(fp, 0x3c00u, iq2_s_qs_neg, iq2_s_qh, iq2_s_scales);
    uint8_t iq3_s_qs[64] = {0};
    iq3_s_qs[0] = 1u;
    uint8_t iq3_s_qh[8] = {0};
    uint8_t iq3_s_signs[32] = {0};
    uint8_t iq3_s_scales[4] = {0};
    test_write_iq3_s_block(fp, 0x3c00u, iq3_s_qs, iq3_s_qh, iq3_s_signs, iq3_s_scales);
    test_write_iq3_s_block(fp, 0x3c00u, iq3_s_qs, iq3_s_qh, iq3_s_signs, iq3_s_scales);
    uint8_t iq3_s_signs_neg[32] = {0};
    iq3_s_signs_neg[0] = 1u;
    test_write_iq3_s_block(fp, 0x3c00u, iq3_s_qs, iq3_s_qh, iq3_s_signs_neg, iq3_s_scales);
    uint8_t iq1_s_qs[32] = {0};
    iq1_s_qs[0] = 1u;
    uint16_t iq1_s_qh[8] = {0};
    test_write_iq1_s_block(fp, 0x3c00u, iq1_s_qs, iq1_s_qh);
    test_write_iq1_s_block(fp, 0x3c00u, iq1_s_qs, iq1_s_qh);
    uint16_t iq1_s_qh_neg_delta[8] = {0x8000u};
    test_write_iq1_s_block(fp, 0x3c00u, iq1_s_qs, iq1_s_qh_neg_delta);
    uint8_t iq1_m_qs[32] = {0};
    iq1_m_qs[0] = 1u;
    uint8_t iq1_m_qh[16] = {0};
    uint16_t iq1_m_scales[4] = {0x0000u, 0x0000u, 0xc000u, 0x3000u};
    test_write_iq1_m_block(fp, iq1_m_qs, iq1_m_qh, iq1_m_scales);
    test_write_iq1_m_block(fp, iq1_m_qs, iq1_m_qh, iq1_m_scales);
    uint8_t iq1_m_qh_neg_delta[16] = {0};
    iq1_m_qh_neg_delta[0] = 0x08u;
    test_write_iq1_m_block(fp, iq1_m_qs, iq1_m_qh_neg_delta, iq1_m_scales);

    TEST_ASSERT(fclose(fp) == 0);
}

static void test_runtime_core_group(void) {
    bool qwen_text_only = test_qwen36_text_only_mode();

    TEST_ASSERT(ds4_runtime_register() == 0);
    TEST_ASSERT(ds4_runtime_register() == 0);
    TEST_ASSERT(qwen36_runtime_register() == 0);
    TEST_ASSERT(qwen36_runtime_register() == 0);
    TEST_ASSERT(mistral35_runtime_register() == 0);
    TEST_ASSERT(mistral35_runtime_register() == 0);

    const rt_model_ops *ops = rt_model_by_family("deepseek-v4-flash");
    TEST_ASSERT(ops == ds4_runtime_ops());
    const rt_model_ops *qwen = rt_model_by_family(QWEN36_RUNTIME_FAMILY);
    TEST_ASSERT(qwen == qwen36_runtime_ops());
    const rt_model_ops *mistral = rt_model_by_family(MISTRAL35_RUNTIME_FAMILY);
    TEST_ASSERT(mistral == mistral35_runtime_ops());
    TEST_ASSERT(rt_probe_model_path("ds4flash.gguf") == ops);
    TEST_ASSERT(rt_probe_model_path("Qwen3.6-27B-Q4_K_M.gguf") == qwen);
    TEST_ASSERT(rt_probe_model_path("Qwen3.6-35B-A3B.gguf") == NULL);
    TEST_ASSERT(rt_probe_model_path("Mistral-Medium-3.5-128B-Q4_K_M.gguf") == mistral);
    TEST_ASSERT(rt_probe_model_path("Mistral-Medium-3.1-128B.gguf") == NULL);
    TEST_ASSERT(!strcmp(rt_backend_name(RT_BACKEND_CUDA), "cuda"));

    char qwen_path[256];
    char qwen_state_path[256];
    char qwen_recurrent_path[256];
    char qwen35_path[256];
    char qwen_bad_shape_path[256];
    char qwen_mmproj_path[256];
    char qwen_mmproj_metal_path[256];
    char qwen_bad_mmproj_path[256];
    char qwen_embedded_mtp_path[256];
    char qwen_mtp_path[256];
    char qwen_mtp_multi_path[256];
    char qwen_mtp_server_model_path[256];
    char qwen_mtp_server_accept_path[256];
    char qwen_mtp_reject_model_path[256];
    char qwen_mtp_reject_path[256];
    char qwen_bad_mtp_path[256];
    char qwen_chat_path[256];
    char qwen_vec_prompt_path[256];
    char qwen_vec_path[256];
    char ds4_path[256];
    char scalar_path[256];
    snprintf(qwen_path, sizeof(qwen_path), "/tmp/ds4-test-%ld-qwen36.gguf", (long)getpid());
    snprintf(qwen_state_path, sizeof(qwen_state_path),
             "/tmp/ds4-test-%ld-qwen36-state.gguf", (long)getpid());
    snprintf(qwen_recurrent_path, sizeof(qwen_recurrent_path),
             "/tmp/ds4-test-%ld-qwen36-recurrent.gguf", (long)getpid());
    snprintf(qwen35_path, sizeof(qwen35_path), "/tmp/ds4-test-%ld-qwen35.gguf", (long)getpid());
    snprintf(qwen_bad_shape_path, sizeof(qwen_bad_shape_path),
             "/tmp/ds4-test-%ld-qwen36-bad-shape.gguf", (long)getpid());
    snprintf(qwen_mmproj_path, sizeof(qwen_mmproj_path),
             "/tmp/ds4-test-%ld-qwen36-mmproj.gguf", (long)getpid());
    snprintf(qwen_mmproj_metal_path, sizeof(qwen_mmproj_metal_path),
             "/tmp/ds4-test-%ld-qwen36-mmproj-metal.gguf", (long)getpid());
    snprintf(qwen_bad_mmproj_path, sizeof(qwen_bad_mmproj_path),
             "/tmp/ds4-test-%ld-qwen36-bad-mmproj.gguf", (long)getpid());
    snprintf(qwen_embedded_mtp_path, sizeof(qwen_embedded_mtp_path),
             "/tmp/ds4-test-%ld-qwen36-embedded-mtp.gguf", (long)getpid());
    snprintf(qwen_mtp_path, sizeof(qwen_mtp_path),
             "/tmp/ds4-test-%ld-qwen36-mtp.gguf", (long)getpid());
    snprintf(qwen_mtp_multi_path, sizeof(qwen_mtp_multi_path),
             "/tmp/ds4-test-%ld-qwen36-mtp-multi.gguf", (long)getpid());
    snprintf(qwen_mtp_server_model_path, sizeof(qwen_mtp_server_model_path),
             "/tmp/ds4-test-%ld-qwen36-mtp-server-model.gguf", (long)getpid());
    snprintf(qwen_mtp_server_accept_path, sizeof(qwen_mtp_server_accept_path),
             "/tmp/ds4-test-%ld-qwen36-mtp-server-accept.gguf", (long)getpid());
    snprintf(qwen_mtp_reject_model_path, sizeof(qwen_mtp_reject_model_path),
             "/tmp/ds4-test-%ld-qwen36-mtp-reject-model.gguf", (long)getpid());
    snprintf(qwen_mtp_reject_path, sizeof(qwen_mtp_reject_path),
             "/tmp/ds4-test-%ld-qwen36-mtp-reject.gguf", (long)getpid());
    snprintf(qwen_bad_mtp_path, sizeof(qwen_bad_mtp_path),
             "/tmp/ds4-test-%ld-qwen36-bad-mtp.gguf", (long)getpid());
    snprintf(qwen_chat_path, sizeof(qwen_chat_path),
             "/tmp/ds4-test-%ld-qwen36-chat.gguf", (long)getpid());
    snprintf(qwen_vec_prompt_path, sizeof(qwen_vec_prompt_path),
             "/tmp/ds4-test-%ld-qwen36-vector-prompt.txt", (long)getpid());
    snprintf(qwen_vec_path, sizeof(qwen_vec_path),
             "/tmp/ds4-test-%ld-qwen36-vector.vec", (long)getpid());
    snprintf(ds4_path, sizeof(ds4_path), "/tmp/ds4-test-%ld-ds4.gguf", (long)getpid());
    snprintf(scalar_path, sizeof(scalar_path), "/tmp/ds4-test-%ld-scalars.gguf", (long)getpid());

    test_write_qwen36_probe_gguf(qwen_path, "Qwen3.6-27B");
    test_write_qwen36_probe_gguf(qwen_state_path, "Qwen3.6-27B");
    test_patch_gguf_tensor_f32(qwen_state_path, "token_embd.weight",
                               (uint64_t)7u * QWEN36_TEST_HIDDEN, 1.0f);
    test_patch_gguf_tensor_f32(qwen_state_path, "blk.3.attn_k.weight",
                               0, 1.0f);
    test_patch_gguf_tensor_f32(qwen_state_path, "blk.3.attn_v.weight",
                               0, 1.0f);
    test_patch_gguf_tensor_f32(qwen_state_path, "blk.3.attn_output.weight",
                               0, 1.0f);
    test_write_qwen36_probe_gguf(qwen_recurrent_path, "Qwen3.6-27B");
    test_patch_gguf_tensor_f32(qwen_recurrent_path, "token_embd.weight",
                               (uint64_t)7u * QWEN36_TEST_HIDDEN, 1.0f);
    test_patch_gguf_tensor_f32(qwen_recurrent_path, "token_embd.weight",
                               (uint64_t)5u * QWEN36_TEST_HIDDEN + 1u, 1.0f);
    test_patch_gguf_tensor_f32(qwen_recurrent_path, "blk.0.attn_qkv.weight",
                               0, 1.0f);
    test_patch_gguf_tensor_f32(
        qwen_recurrent_path, "blk.0.attn_qkv.weight",
        (uint64_t)QWEN36_TEST_LINEAR_QK_HEADS *
            QWEN36_TEST_LINEAR_HEAD_DIM * QWEN36_TEST_HIDDEN,
        1.0f);
    test_patch_gguf_tensor_f32(
        qwen_recurrent_path, "blk.0.attn_qkv.weight",
        2ull * QWEN36_TEST_LINEAR_QK_HEADS *
            QWEN36_TEST_LINEAR_HEAD_DIM * QWEN36_TEST_HIDDEN,
        1.0f);
    test_patch_gguf_tensor_f32(qwen_recurrent_path,
                               "blk.0.ssm_conv1d.weight",
                               QWEN36_TEST_SSM_D_CONV - 2u,
                               1.0f);
    test_patch_gguf_tensor_f32(qwen_recurrent_path,
                               "blk.0.ssm_conv1d.weight",
                               QWEN36_TEST_SSM_D_CONV - 1u,
                               1.0f);
    test_patch_gguf_tensor_f32(
        qwen_recurrent_path, "blk.0.ssm_conv1d.weight",
        (uint64_t)QWEN36_TEST_LINEAR_QK_HEADS *
            QWEN36_TEST_LINEAR_HEAD_DIM * QWEN36_TEST_SSM_D_CONV +
            (QWEN36_TEST_SSM_D_CONV - 2u),
        1.0f);
    test_patch_gguf_tensor_f32(
        qwen_recurrent_path, "blk.0.ssm_conv1d.weight",
        (uint64_t)QWEN36_TEST_LINEAR_QK_HEADS *
            QWEN36_TEST_LINEAR_HEAD_DIM * QWEN36_TEST_SSM_D_CONV +
            (QWEN36_TEST_SSM_D_CONV - 1u),
        1.0f);
    test_patch_gguf_tensor_f32(
        qwen_recurrent_path, "blk.0.ssm_conv1d.weight",
        2ull * QWEN36_TEST_LINEAR_QK_HEADS *
            QWEN36_TEST_LINEAR_HEAD_DIM * QWEN36_TEST_SSM_D_CONV +
            (QWEN36_TEST_SSM_D_CONV - 2u),
        1.0f);
    test_patch_gguf_tensor_f32(
        qwen_recurrent_path, "blk.0.ssm_conv1d.weight",
        2ull * QWEN36_TEST_LINEAR_QK_HEADS *
            QWEN36_TEST_LINEAR_HEAD_DIM * QWEN36_TEST_SSM_D_CONV +
            (QWEN36_TEST_SSM_D_CONV - 1u),
        1.0f);
    test_patch_gguf_tensor_f32(qwen_recurrent_path, "blk.0.attn_gate.weight",
                               0, 1.0f);
    test_patch_gguf_tensor_f32(qwen_recurrent_path, "blk.0.attn_gate.weight",
                               1, 1.0f);
    test_patch_gguf_tensor_f32(qwen_recurrent_path, "blk.0.ssm_norm.weight",
                               0, 1.0f);
    test_patch_gguf_tensor_f32(qwen_recurrent_path, "blk.0.ssm_out.weight",
                               0, 1.0f);
    test_write_qwen36_probe_gguf(qwen35_path, "Qwen3.6-35B-A3B");
    test_write_qwen36_probe_gguf_ex(qwen_bad_shape_path, "Qwen3.6-27B", true);
    if (!qwen_text_only) {
        test_write_qwen36_mmproj_gguf(qwen_mmproj_path, false);
        test_write_qwen36_mmproj_gguf(qwen_mmproj_metal_path, false);
        test_patch_gguf_tensor_f32(qwen_mmproj_metal_path,
                                   "v.patch_embd.bias", 0, 1.0f);
        test_write_qwen36_mmproj_gguf(qwen_bad_mmproj_path, true);
    }
    test_write_qwen36_embedded_mtp_probe_gguf(qwen_embedded_mtp_path,
                                              "Qwen3.6-27B");
    test_write_qwen36_mtp_gguf(qwen_mtp_path, false);
    test_write_qwen36_mtp_gguf_layers(qwen_mtp_multi_path, 2, false);
    test_write_qwen36_output_probe_gguf(qwen_mtp_server_model_path,
                                        "Qwen3.6-27B");
    test_patch_gguf_tensor_f32(qwen_mtp_server_model_path,
                               "token_embd.weight",
                               (uint64_t)7u * QWEN36_TEST_HIDDEN, 1.0f);
    test_patch_gguf_tensor_f32(qwen_mtp_server_model_path,
                               "token_embd.weight",
                               (uint64_t)15u * QWEN36_TEST_HIDDEN, 1.0f);
    test_patch_gguf_tensor_f32(qwen_mtp_server_model_path, "output.weight",
                               (uint64_t)7u * QWEN36_TEST_HIDDEN, 1.0f);
    test_write_qwen36_mtp_gguf(qwen_mtp_server_accept_path, false);
    test_patch_gguf_tensor_f32(qwen_mtp_server_accept_path,
                               "blk.64.nextn.eh_proj.weight", 0, 1.0f);
    test_write_qwen36_output_probe_gguf(qwen_mtp_reject_model_path, "Qwen3.6-27B");
    test_patch_gguf_tensor_f32(qwen_mtp_reject_model_path, "token_embd.weight",
                               (uint64_t)7u * QWEN36_TEST_HIDDEN, 1.0f);
    test_patch_gguf_tensor_f32(qwen_mtp_reject_model_path, "output.weight",
                               (uint64_t)4u * QWEN36_TEST_HIDDEN + 1u, 1.0f);
    test_write_qwen36_mtp_gguf(qwen_mtp_reject_path, false);
    test_patch_gguf_tensor_f32(qwen_mtp_reject_path, "blk.64.nextn.eh_proj.weight",
                               (uint64_t)1u * QWEN36_TEST_HIDDEN * 2u, 1.0f);
    test_write_qwen36_mtp_gguf(qwen_bad_mtp_path, true);
    test_write_qwen36_chat_probe_gguf(qwen_chat_path, "Qwen3.6-27B");
    test_write_arch_probe_gguf(ds4_path, "deepseek4");
    test_write_scalar_tensor_gguf(scalar_path);
    FILE *qwen_vec_prompt_fp = fopen(qwen_vec_prompt_path, "wb");
    TEST_ASSERT(qwen_vec_prompt_fp != NULL);
    TEST_ASSERT(fputs("hi", qwen_vec_prompt_fp) >= 0);
    TEST_ASSERT(fclose(qwen_vec_prompt_fp) == 0);
    FILE *qwen_vec_fp = fopen(qwen_vec_path, "wb");
    TEST_ASSERT(qwen_vec_fp != NULL);
    TEST_ASSERT(fprintf(qwen_vec_fp,
                        "# qwen36-official-full-logit-vectors-v1\n"
                        "# model Qwen/Qwen3.6-27B\n"
                        "# revision synthetic-test-fixture\n"
                        "# model_vocab_size %u\n"
                        "# tokenizer_vocab_size %u\n"
                        "# eos_token_id 0\n"
                        "# full_logits 1\n"
                        "case synthetic_zero 32 1 %s\n"
                        "render raw\n"
                        "tolerance 0\n"
                        "prompt_tokens 1\n"
                        "tokens 7\n"
                        "end_prompt_tokens\n"
                        "step 0 0 0 2\n"
                        "top 0 0\n"
                        "top 1 0\n"
                        "logits_f32_hex %u\n",
                        QWEN36_TEST_TOKEN_COUNT,
                        QWEN36_TEST_TOKEN_COUNT,
                        qwen_vec_prompt_path,
                        QWEN36_TEST_TOKEN_COUNT) > 0);
    for (uint32_t i = 0; i < QWEN36_TEST_TOKEN_COUNT; i++) {
        if ((i % 16u) == 0) {
            TEST_ASSERT(fputs("logits_hex", qwen_vec_fp) >= 0);
        }
        TEST_ASSERT(fputs(" 00000000", qwen_vec_fp) >= 0);
        if ((i % 16u) == 15u || i + 1u == QWEN36_TEST_TOKEN_COUNT) {
            TEST_ASSERT(fputc('\n', qwen_vec_fp) != EOF);
        }
    }
    TEST_ASSERT(fputs("end_logits\nend\n", qwen_vec_fp) >= 0);
    TEST_ASSERT(fclose(qwen_vec_fp) == 0);

    rt_gguf_metadata meta = {0};
    char gguf_err[256];
    TEST_ASSERT(rt_gguf_metadata_read(qwen_path, &meta, gguf_err, sizeof(gguf_err)) == 0);
    const char *arch = NULL;
    uint32_t block_count = 0;
    TEST_ASSERT(meta.version == 3);
    TEST_ASSERT(meta.n_tensors == QWEN36_TEST_TENSORS);
    TEST_ASSERT(meta.n_kv == 17);
    TEST_ASSERT(rt_gguf_metadata_get_string(&meta, "general.architecture", &arch));
    TEST_ASSERT(!strcmp(arch, "qwen35"));
    TEST_ASSERT(rt_gguf_metadata_get_u32(&meta, "qwen35.block_count", &block_count));
    TEST_ASSERT(block_count == 64);
    rt_gguf_array_ref token_array = {0};
    TEST_ASSERT(rt_gguf_metadata_get_array(&meta, "tokenizer.ggml.tokens", &token_array));
    TEST_ASSERT(token_array.type == RT_GGUF_VALUE_STRING);
    TEST_ASSERT(token_array.len == QWEN36_TEST_TOKEN_COUNT);
    rt_gguf_metadata_free(&meta);

    rt_gguf_file *qwen_file = NULL;
    TEST_ASSERT(rt_gguf_file_open(&qwen_file, qwen_path, gguf_err, sizeof(gguf_err)) == 0);
    TEST_ASSERT(rt_gguf_file_tensor_count(qwen_file) == QWEN36_TEST_TENSORS);
    const char *token_view = NULL;
    size_t token_view_len = 0;
    TEST_ASSERT(rt_gguf_file_array_string_at(qwen_file, "tokenizer.ggml.tokens", 7, &token_view, &token_view_len));
    TEST_ASSERT(token_view_len == 2);
    TEST_ASSERT(!memcmp(token_view, "hi", token_view_len));
    const rt_gguf_tensor *embd = rt_gguf_file_find_tensor(qwen_file, "token_embd.weight");
    TEST_ASSERT(embd != NULL);
    TEST_ASSERT(embd->type == 1);
    TEST_ASSERT(embd->ndim == 2);
    TEST_ASSERT(embd->dim[0] == QWEN36_TEST_HIDDEN);
    TEST_ASSERT(embd->dim[1] == QWEN36_TEST_VOCAB);
    TEST_ASSERT(rt_gguf_file_tensor_data(qwen_file, embd) != NULL);
    TEST_ASSERT(!strcmp(rt_gguf_tensor_type_name(embd->type), "f16"));
    const rt_gguf_tensor *last_attn_out = rt_gguf_file_find_tensor(qwen_file, "blk.63.attn_output.weight");
    TEST_ASSERT(last_attn_out != NULL);
    TEST_ASSERT(last_attn_out->ndim == 2);
    TEST_ASSERT(last_attn_out->dim[0] == QWEN36_TEST_HEAD_DIM * QWEN36_TEST_HEADS);
    TEST_ASSERT(last_attn_out->dim[1] == QWEN36_TEST_HIDDEN);
    rt_gguf_file_close(qwen_file);

    rt_gguf_file *scalar_file = NULL;
    TEST_ASSERT(rt_gguf_file_open(&scalar_file, scalar_path, gguf_err, sizeof(gguf_err)) == 0);
    const rt_gguf_tensor *f32_values = rt_gguf_file_find_tensor(scalar_file, "f32_values");
    const rt_gguf_tensor *f16_values = rt_gguf_file_find_tensor(scalar_file, "f16_values");
    const rt_gguf_tensor *bf16_values = rt_gguf_file_find_tensor(scalar_file, "bf16_values");
    const rt_gguf_tensor *q8_values = rt_gguf_file_find_tensor(scalar_file, "q8_values");
    const rt_gguf_tensor *f16_matrix = rt_gguf_file_find_tensor(scalar_file, "f16_matrix");
    const rt_gguf_tensor *q4_k_values = rt_gguf_file_find_tensor(scalar_file, "q4_k_values");
    const rt_gguf_tensor *q8_matrix = rt_gguf_file_find_tensor(scalar_file, "q8_matrix");
    const rt_gguf_tensor *q6_k_values = rt_gguf_file_find_tensor(scalar_file, "q6_k_values");
    const rt_gguf_tensor *q4_0_values = rt_gguf_file_find_tensor(scalar_file, "q4_0_values");
    const rt_gguf_tensor *q4_1_values = rt_gguf_file_find_tensor(scalar_file, "q4_1_values");
    const rt_gguf_tensor *q5_0_values = rt_gguf_file_find_tensor(scalar_file, "q5_0_values");
    const rt_gguf_tensor *q5_1_values = rt_gguf_file_find_tensor(scalar_file, "q5_1_values");
    const rt_gguf_tensor *q8_1_values = rt_gguf_file_find_tensor(scalar_file, "q8_1_values");
    const rt_gguf_tensor *q5_k_values = rt_gguf_file_find_tensor(scalar_file, "q5_k_values");
    const rt_gguf_tensor *q8_k_values = rt_gguf_file_find_tensor(scalar_file, "q8_k_values");
    const rt_gguf_tensor *q2_k_values = rt_gguf_file_find_tensor(scalar_file, "q2_k_values");
    const rt_gguf_tensor *q3_k_values = rt_gguf_file_find_tensor(scalar_file, "q3_k_values");
    const rt_gguf_tensor *iq2_xxs_values = rt_gguf_file_find_tensor(scalar_file, "iq2_xxs_values");
    const rt_gguf_tensor *iq2_xs_values = rt_gguf_file_find_tensor(scalar_file, "iq2_xs_values");
    const rt_gguf_tensor *iq2_s_values = rt_gguf_file_find_tensor(scalar_file, "iq2_s_values");
    const rt_gguf_tensor *iq3_xxs_values = rt_gguf_file_find_tensor(scalar_file, "iq3_xxs_values");
    const rt_gguf_tensor *iq3_s_values = rt_gguf_file_find_tensor(scalar_file, "iq3_s_values");
    const rt_gguf_tensor *iq1_s_values = rt_gguf_file_find_tensor(scalar_file, "iq1_s_values");
    const rt_gguf_tensor *iq1_m_values = rt_gguf_file_find_tensor(scalar_file, "iq1_m_values");
    const rt_gguf_tensor *iq4_nl_values = rt_gguf_file_find_tensor(scalar_file, "iq4_nl_values");
    const rt_gguf_tensor *iq4_xs_values = rt_gguf_file_find_tensor(scalar_file, "iq4_xs_values");
    const rt_gguf_tensor *q8_k_matrix = rt_gguf_file_find_tensor(scalar_file, "q8_k_matrix");
    const rt_gguf_tensor *q4_k_matrix = rt_gguf_file_find_tensor(scalar_file, "q4_k_matrix");
    const rt_gguf_tensor *q5_k_matrix = rt_gguf_file_find_tensor(scalar_file, "q5_k_matrix");
    const rt_gguf_tensor *q6_k_matrix = rt_gguf_file_find_tensor(scalar_file, "q6_k_matrix");
    const rt_gguf_tensor *q2_k_matrix = rt_gguf_file_find_tensor(scalar_file, "q2_k_matrix");
    const rt_gguf_tensor *q3_k_matrix = rt_gguf_file_find_tensor(scalar_file, "q3_k_matrix");
    const rt_gguf_tensor *q4_0_matrix = rt_gguf_file_find_tensor(scalar_file, "q4_0_matrix");
    const rt_gguf_tensor *q4_1_matrix = rt_gguf_file_find_tensor(scalar_file, "q4_1_matrix");
    const rt_gguf_tensor *q5_0_matrix = rt_gguf_file_find_tensor(scalar_file, "q5_0_matrix");
    const rt_gguf_tensor *q5_1_matrix = rt_gguf_file_find_tensor(scalar_file, "q5_1_matrix");
    const rt_gguf_tensor *q8_1_matrix = rt_gguf_file_find_tensor(scalar_file, "q8_1_matrix");
    const rt_gguf_tensor *iq2_xxs_matrix = rt_gguf_file_find_tensor(scalar_file, "iq2_xxs_matrix");
    const rt_gguf_tensor *iq2_xs_matrix = rt_gguf_file_find_tensor(scalar_file, "iq2_xs_matrix");
    const rt_gguf_tensor *iq2_s_matrix = rt_gguf_file_find_tensor(scalar_file, "iq2_s_matrix");
    const rt_gguf_tensor *iq3_xxs_matrix = rt_gguf_file_find_tensor(scalar_file, "iq3_xxs_matrix");
    const rt_gguf_tensor *iq3_s_matrix = rt_gguf_file_find_tensor(scalar_file, "iq3_s_matrix");
    const rt_gguf_tensor *iq1_s_matrix = rt_gguf_file_find_tensor(scalar_file, "iq1_s_matrix");
    const rt_gguf_tensor *iq1_m_matrix = rt_gguf_file_find_tensor(scalar_file, "iq1_m_matrix");
    const rt_gguf_tensor *iq4_nl_matrix = rt_gguf_file_find_tensor(scalar_file, "iq4_nl_matrix");
    const rt_gguf_tensor *iq4_xs_matrix = rt_gguf_file_find_tensor(scalar_file, "iq4_xs_matrix");
    float scalar = 0.0f;
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, f32_values, 0, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, f32_values, 1, &scalar) && scalar == -2.5f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, f16_values, 0, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, f16_values, 1, &scalar) && scalar == -2.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, f16_values, 2, &scalar) && scalar == 0.5f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, f16_values, 3, &scalar) && scalar > 0.0f && scalar < 0.000001f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, bf16_values, 0, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, bf16_values, 1, &scalar) && scalar == -2.5f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q8_values, 0, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q8_values, 1, &scalar) && scalar == -2.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q8_values, 2, &scalar) && scalar == 3.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q4_k_values, 0, &scalar) && scalar == 2.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q4_k_values, 32, &scalar) && scalar == 3.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q4_k_values, 1, &scalar) && scalar == 0.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q6_k_values, 0, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q6_k_values, 32, &scalar) && scalar == 2.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q4_0_values, 0, &scalar) && scalar == 2.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q4_0_values, 16, &scalar) && scalar == -3.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q4_1_values, 0, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q4_1_values, 16, &scalar) && scalar == -0.5f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q5_0_values, 0, &scalar) && scalar == 2.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q5_0_values, 16, &scalar) && scalar == -1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q5_1_values, 0, &scalar) && scalar == 2.5f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q5_1_values, 16, &scalar) && scalar == 1.75f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q8_1_values, 0, &scalar) && scalar == 2.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q8_1_values, 1, &scalar) && scalar == -3.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q5_k_values, 0, &scalar) && scalar == 18.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q5_k_values, 32, &scalar) && scalar == 3.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q8_k_values, 0, &scalar) && scalar == 2.5f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q8_k_values, 255, &scalar) && scalar == -1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q2_k_values, 0, &scalar) && scalar == 5.5f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q2_k_values, 1, &scalar) && scalar == -0.5f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q2_k_values, 16, &scalar) && scalar == 0.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q3_k_values, 0, &scalar) && scalar == 3.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q3_k_values, 1, &scalar) && scalar == -2.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q3_k_values, 16, &scalar) && scalar == 2.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, q3_k_values, 17, &scalar) && scalar == -8.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq2_xxs_values, 0, &scalar) && scalar == 5.375f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq2_xxs_values, 1, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq2_xxs_values, 255, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq2_xs_values, 0, &scalar) && scalar == 5.375f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq2_xs_values, 1, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq2_s_values, 0, &scalar) && scalar == 5.375f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq2_s_values, 1, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq3_xxs_values, 0, &scalar) && scalar == 5.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq3_xxs_values, 1, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq3_s_values, 0, &scalar) && scalar == 3.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq3_s_values, 1, &scalar) && scalar == 1.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq1_s_values, 0, &scalar) && scalar == 1.125f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq1_s_values, 1, &scalar) && scalar == -0.875f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq1_m_values, 0, &scalar) && scalar == 1.125f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq1_m_values, 1, &scalar) && scalar == -0.875f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq4_nl_values, 0, &scalar) && scalar == 113.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq4_nl_values, 16, &scalar) && scalar == -10.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq4_xs_values, 0, &scalar) && scalar == 113.0f);
    TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, iq4_xs_values, 16, &scalar) && scalar == -10.0f);
    TEST_ASSERT(!rt_gguf_tensor_read_f32(scalar_file, f32_values, 3, &scalar));
    float x[2] = {10.0f, 1.0f};
    float y[2] = {0};
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, f16_matrix, x, 2, y, 2));
    TEST_ASSERT(y[0] == 12.0f);
    TEST_ASSERT(y[1] == 34.0f);
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (qwen36_metal_available()) {
        char metal_err[160] = {0};
        qwen36_metal_backend *metal = qwen36_metal_create(
            metal_err, sizeof(metal_err));
        TEST_ASSERT(metal != NULL);
        float metal_y[2] = {0};
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, f16_matrix,
                                        x, 2, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(metal_y[0] == 12.0f);
        TEST_ASSERT(metal_y[1] == 34.0f);
        float metal_x32[32] = {0};
        metal_x32[0] = 10.0f;
        metal_x32[1] = 1.0f;
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q8_matrix,
                                        metal_x32, 32, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 12.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] + 6.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q4_0_matrix,
                                        metal_x32, 32, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 21.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 33.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q4_1_matrix,
                                        metal_x32, 32, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 14.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 13.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q5_0_matrix,
                                        metal_x32, 32, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 19.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 75.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q5_1_matrix,
                                        metal_x32, 32, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 26.75f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 10.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q8_1_matrix,
                                        metal_x32, 32, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 17.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] + 27.0f) < 0.0002f);
        float metal_x256[256] = {0};
        metal_x256[0] = 10.0f;
        metal_x256[1] = 1.0f;
        metal_x256[32] = 2.0f;
        metal_x256[255] = 2.0f;
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q8_k_matrix,
                                        metal_x256, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 23.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] + 17.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q4_k_matrix,
                                        metal_x256, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 26.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 79.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q5_k_matrix,
                                        metal_x256, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 186.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 102.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q6_k_matrix,
                                        metal_x256, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] + 18.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] + 36.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q2_k_matrix,
                                        metal_x256, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 30.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 24.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, q3_k_matrix,
                                        metal_x256, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 34.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 24.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, iq2_xxs_matrix,
                                        metal_x256, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 58.75f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] + 14.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, iq4_nl_matrix,
                                        metal_x32, 32, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 23.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 120.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, iq4_xs_matrix,
                                        metal_x256, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 23.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 120.0f) < 0.0002f);
        float metal_x256_iq3[256] = {0};
        metal_x256_iq3[0] = 10.0f;
        metal_x256_iq3[1] = 1.0f;
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, iq3_xxs_matrix,
                                        metal_x256_iq3, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 51.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] + 49.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, iq2_xs_matrix,
                                        metal_x256_iq3, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 54.75f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] + 52.75f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, iq2_s_matrix,
                                        metal_x256_iq3, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 54.75f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] + 52.75f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, iq3_s_matrix,
                                        metal_x256_iq3, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 31.0f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] + 29.0f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, iq1_s_matrix,
                                        metal_x256_iq3, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 10.375f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 7.625f) < 0.0002f);
        TEST_ASSERT(qwen36_metal_matvec(metal, scalar_file, iq1_m_matrix,
                                        metal_x256_iq3, 256, metal_y, 2,
                                        metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(metal_y[0] - 10.375f) < 0.0002f);
        TEST_ASSERT(fabsf(metal_y[1] - 7.625f) < 0.0002f);
        float rms_x[4] = {2.0f, -1.0f, 0.5f, 4.0f};
        float rms_y[4] = {0};
        TEST_ASSERT(qwen36_metal_rms_norm(metal, scalar_file, f16_values,
                                          rms_x, 4, rms_y, false,
                                          metal_err, sizeof(metal_err)));
        double rms_ss = 0.0;
        for (uint64_t i = 0; i < 4; i++) {
            rms_ss += (double)rms_x[i] * (double)rms_x[i];
        }
        float rms_inv = 1.0f / sqrtf((float)(rms_ss / 4.0) + 1.0e-6f);
        for (uint64_t i = 0; i < 4; i++) {
            float w = 0.0f;
            TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, f16_values,
                                                i, &w));
            TEST_ASSERT(fabsf(rms_y[i] - (rms_x[i] * rms_inv * w)) < 0.0002f);
        }
        TEST_ASSERT(qwen36_metal_rms_norm(metal, scalar_file, f16_values,
                                          rms_x, 4, rms_y, true,
                                          metal_err, sizeof(metal_err)));
        for (uint64_t i = 0; i < 4; i++) {
            float w = 0.0f;
            TEST_ASSERT(rt_gguf_tensor_read_f32(scalar_file, f16_values,
                                                i, &w));
            TEST_ASSERT(fabsf(rms_y[i] -
                              (rms_x[i] * rms_inv * (1.0f + w))) < 0.0002f);
        }
        float silu_gate[4] = {-4.0f, -1.0f, 0.0f, 2.0f};
        float silu_up[4] = {0.5f, -3.0f, 7.0f, 4.0f};
        float silu_y[4] = {0};
        TEST_ASSERT(qwen36_metal_silu_mul(metal, silu_gate, silu_up, 4,
                                          silu_y, metal_err, sizeof(metal_err)));
        for (uint64_t i = 0; i < 4; i++) {
            float z = silu_gate[i] >= 0.0f ? expf(-silu_gate[i]) : expf(silu_gate[i]);
            float sig = silu_gate[i] >= 0.0f ? (1.0f / (1.0f + z)) : (z / (1.0f + z));
            TEST_ASSERT(fabsf(silu_y[i] - (silu_gate[i] * sig * silu_up[i])) < 0.0002f);
        }
        float l2_x[4] = {3.0f, 4.0f, 0.0f, 0.0f};
        float l2_y[4] = {0};
        TEST_ASSERT(qwen36_metal_l2_norm(metal, l2_x, 4, l2_y,
                                         metal_err, sizeof(metal_err)));
        TEST_ASSERT(fabsf(l2_y[0] - 0.6f) < 0.0002f);
        TEST_ASSERT(fabsf(l2_y[1] - 0.8f) < 0.0002f);
        TEST_ASSERT(fabsf(l2_y[2]) < 0.0002f);
        TEST_ASSERT(fabsf(l2_y[3]) < 0.0002f);
        float l2_zero[4] = {0};
        TEST_ASSERT(!qwen36_metal_l2_norm(metal, l2_zero, 4, l2_y,
                                          metal_err, sizeof(metal_err)));
        float delta_state[16] = {
            0.5f, -0.25f, 0.75f, 1.0f,
            -1.5f, 0.5f, 0.25f, -0.75f,
            0.0f, 1.25f, -0.5f, 0.5f,
            1.0f, -1.0f, 0.5f, -0.25f,
        };
        float delta_want_state[16];
        memcpy(delta_want_state, delta_state, sizeof(delta_want_state));
        float delta_query[4] = {0.25f, -0.5f, 1.0f, 0.75f};
        float delta_key[4] = {0.5f, -1.0f, 0.25f, 0.75f};
        float delta_value[4] = {1.5f, -0.5f, 0.25f, -1.0f};
        float delta_out[4] = {0};
        float delta_want[4] = {0};
        float delta_mem[4] = {0};
        for (uint64_t kd = 0; kd < 4; kd++) {
            for (uint64_t vd = 0; vd < 4; vd++) {
                delta_want_state[kd * 4 + vd] *= 0.6f;
                delta_mem[vd] += delta_want_state[kd * 4 + vd] * delta_key[kd];
            }
        }
        for (uint64_t kd = 0; kd < 4; kd++) {
            for (uint64_t vd = 0; vd < 4; vd++) {
                float delta = (delta_value[vd] - delta_mem[vd]) * 0.25f;
                delta_want_state[kd * 4 + vd] += delta_key[kd] * delta;
            }
        }
        for (uint64_t vd = 0; vd < 4; vd++) {
            for (uint64_t kd = 0; kd < 4; kd++) {
                delta_want[vd] +=
                    delta_want_state[kd * 4 + vd] * (delta_query[kd] * 0.5f);
            }
        }
        TEST_ASSERT(qwen36_metal_gated_delta_head(
            metal, delta_state, delta_query, delta_key, delta_value, 4,
            0.6f, 0.25f, 0.5f, delta_out, metal_err, sizeof(metal_err)));
        for (uint64_t i = 0; i < 16; i++) {
            TEST_ASSERT(fabsf(delta_state[i] - delta_want_state[i]) < 0.0002f);
        }
        for (uint64_t i = 0; i < 4; i++) {
            TEST_ASSERT(fabsf(delta_out[i] - delta_want[i]) < 0.0002f);
        }
        float attn_query[4] = {0.5f, -0.25f, 0.75f, 1.0f};
        float attn_gate[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
        float attn_keys[12] = {
            1.0f, 0.0f, -0.5f, 0.25f,
            0.25f, 1.0f, 0.5f, -0.75f,
            -1.0f, 0.5f, 0.25f, 1.0f,
        };
        float attn_values[12] = {
            0.5f, 1.0f, -1.0f, 0.25f,
            -0.25f, 0.75f, 0.5f, 1.25f,
            1.0f, -0.5f, 0.25f, -0.75f,
        };
        float attn_out[4] = {0};
        float attn_score[3] = {0};
        float attn_max = -FLT_MAX;
        for (uint64_t t = 0; t < 3; t++) {
            for (uint64_t d = 0; d < 4; d++) {
                attn_score[t] += attn_query[d] * attn_keys[t * 4 + d];
            }
            attn_score[t] /= sqrtf(4.0f);
            if (attn_score[t] > attn_max) attn_max = attn_score[t];
        }
        float attn_denom = 0.0f;
        for (uint64_t t = 0; t < 3; t++) {
            attn_score[t] = expf(attn_score[t] - attn_max);
            attn_denom += attn_score[t];
        }
        float attn_want[4] = {0};
        for (uint64_t d = 0; d < 4; d++) {
            for (uint64_t t = 0; t < 3; t++) {
                attn_want[d] +=
                    (attn_score[t] / attn_denom) * attn_values[t * 4 + d];
            }
            float z = attn_gate[d] >= 0.0f ? expf(-attn_gate[d]) : expf(attn_gate[d]);
            float sig = attn_gate[d] >= 0.0f ? (1.0f / (1.0f + z)) : (z / (1.0f + z));
            attn_want[d] *= sig;
        }
        TEST_ASSERT(qwen36_metal_full_attention_head(
            metal, attn_query, attn_gate, attn_keys, attn_values,
            3, 4, 4, attn_out, metal_err, sizeof(metal_err)));
        for (uint64_t i = 0; i < 4; i++) {
            TEST_ASSERT(fabsf(attn_out[i] - attn_want[i]) < 0.0002f);
        }
        qwen36_metal_destroy(metal);
    }
#endif
    TEST_ASSERT(!rt_gguf_tensor_matvec_f32(scalar_file, f16_matrix, x, 3, y, 2));
    float x32[32] = {0};
    x32[0] = 10.0f;
    x32[1] = 1.0f;
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q8_matrix, x32, 32, y, 2));
    TEST_ASSERT(y[0] == 12.0f);
    TEST_ASSERT(y[1] == -6.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q4_0_matrix, x32, 32, y, 2));
    TEST_ASSERT(y[0] == 21.0f);
    TEST_ASSERT(y[1] == 33.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q4_1_matrix, x32, 32, y, 2));
    TEST_ASSERT(y[0] == 14.0f);
    TEST_ASSERT(y[1] == 13.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q5_0_matrix, x32, 32, y, 2));
    TEST_ASSERT(y[0] == 19.0f);
    TEST_ASSERT(y[1] == 75.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q5_1_matrix, x32, 32, y, 2));
    TEST_ASSERT(y[0] == 26.75f);
    TEST_ASSERT(y[1] == 10.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q8_1_matrix, x32, 32, y, 2));
    TEST_ASSERT(y[0] == 17.0f);
    TEST_ASSERT(y[1] == -27.0f);
    float x256[256] = {0};
    x256[0] = 10.0f;
    x256[1] = 1.0f;
    x256[32] = 2.0f;
    x256[255] = 2.0f;
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q8_k_matrix, x256, 256, y, 2));
    TEST_ASSERT(y[0] == 23.0f);
    TEST_ASSERT(y[1] == -17.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q4_k_matrix, x256, 256, y, 2));
    TEST_ASSERT(y[0] == 26.0f);
    TEST_ASSERT(y[1] == 79.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q5_k_matrix, x256, 256, y, 2));
    TEST_ASSERT(y[0] == 186.0f);
    TEST_ASSERT(y[1] == 102.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q6_k_matrix, x256, 256, y, 2));
    TEST_ASSERT(y[0] == -18.0f);
    TEST_ASSERT(y[1] == -36.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q2_k_matrix, x256, 256, y, 2));
    TEST_ASSERT(y[0] == 30.0f);
    TEST_ASSERT(y[1] == 24.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, q3_k_matrix, x256, 256, y, 2));
    TEST_ASSERT(y[0] == 34.0f);
    TEST_ASSERT(y[1] == 24.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, iq2_xxs_matrix, x256, 256, y, 2));
    TEST_ASSERT(y[0] == 58.75f);
    TEST_ASSERT(y[1] == -14.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, iq4_nl_matrix, x32, 32, y, 2));
    TEST_ASSERT(y[0] == 23.0f);
    TEST_ASSERT(y[1] == 120.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, iq4_xs_matrix, x256, 256, y, 2));
    TEST_ASSERT(y[0] == 23.0f);
    TEST_ASSERT(y[1] == 120.0f);
    float x256_iq3[256] = {0};
    x256_iq3[0] = 10.0f;
    x256_iq3[1] = 1.0f;
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, iq3_xxs_matrix, x256_iq3, 256, y, 2));
    TEST_ASSERT(y[0] == 51.0f);
    TEST_ASSERT(y[1] == -49.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, iq2_xs_matrix, x256_iq3, 256, y, 2));
    TEST_ASSERT(y[0] == 54.75f);
    TEST_ASSERT(y[1] == -52.75f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, iq2_s_matrix, x256_iq3, 256, y, 2));
    TEST_ASSERT(y[0] == 54.75f);
    TEST_ASSERT(y[1] == -52.75f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, iq3_s_matrix, x256_iq3, 256, y, 2));
    TEST_ASSERT(y[0] == 31.0f);
    TEST_ASSERT(y[1] == -29.0f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, iq1_s_matrix, x256_iq3, 256, y, 2));
    TEST_ASSERT(y[0] == 10.375f);
    TEST_ASSERT(y[1] == 7.625f);
    TEST_ASSERT(rt_gguf_tensor_matvec_f32(scalar_file, iq1_m_matrix, x256_iq3, 256, y, 2));
    TEST_ASSERT(y[0] == 10.375f);
    TEST_ASSERT(y[1] == 7.625f);
    rt_gguf_file_close(scalar_file);

    if (!qwen_text_only) {
        rt_gguf_file *qwen_mmproj_file = NULL;
        TEST_ASSERT(rt_gguf_file_open(&qwen_mmproj_file, qwen_mmproj_path,
                                      gguf_err, sizeof(gguf_err)) == 0);
        TEST_ASSERT(rt_gguf_file_tensor_count(qwen_mmproj_file) ==
                    QWEN36_TEST_MMPROJ_TENSORS);
        const rt_gguf_metadata *mmproj_meta =
            rt_gguf_file_metadata(qwen_mmproj_file);
        bool has_vision = false;
        TEST_ASSERT(rt_gguf_metadata_get_bool(mmproj_meta,
                                              "clip.has_vision_encoder",
                                              &has_vision));
        TEST_ASSERT(has_vision);
        const rt_gguf_tensor *patch =
            rt_gguf_file_find_tensor(qwen_mmproj_file, "v.patch_embd.weight");
        TEST_ASSERT(patch != NULL);
        TEST_ASSERT(patch->ndim == 4);
        TEST_ASSERT(patch->dim[0] == QWEN36_TEST_VISION_PATCH);
        TEST_ASSERT(patch->dim[3] == QWEN36_TEST_VISION_EMBD);
        rt_gguf_file_close(qwen_mmproj_file);
    }

    TEST_ASSERT(rt_probe_model_path(qwen_path) == qwen);
    TEST_ASSERT(rt_probe_model_path(qwen35_path) == NULL);
    TEST_ASSERT(rt_probe_model_path(qwen_bad_shape_path) == qwen);
    TEST_ASSERT(rt_probe_model_path(ds4_path) == ops);

    const qwen36_model_spec *spec = qwen36_runtime_model_spec();
    TEST_ASSERT(!strcmp(spec->hf_repo, QWEN36_RUNTIME_HF_REPO));
    TEST_ASSERT(spec->dense_weights);
    TEST_ASSERT(spec->n_layers == 64);
    TEST_ASSERT(spec->hidden_size == 5120);
    TEST_ASSERT(spec->vocab_size == 248320);
    TEST_ASSERT(spec->full_attention_interval == 4);
    rt_context_memory qwen_cpu_mem = rt_estimate_context_memory(qwen, RT_BACKEND_CPU, 32);
    uint64_t qwen_token_state_bytes = 32u * sizeof(int);
    uint64_t qwen_full_kv_bytes = test_qwen36_full_kv_bytes(32);
    uint64_t qwen_recurrent_state_bytes = test_qwen36_recurrent_state_bytes();
    uint64_t qwen_conv_state_bytes = test_qwen36_conv_state_bytes();
    uint64_t qwen_hidden_state_bytes = QWEN36_TEST_HIDDEN * sizeof(float);
    uint64_t qwen_logits_bytes = (uint64_t)QWEN36_TEST_VOCAB * sizeof(float);
    uint64_t qwen_expected_state_bytes =
        qwen_token_state_bytes + qwen_full_kv_bytes +
        qwen_recurrent_state_bytes + qwen_conv_state_bytes +
        qwen_hidden_state_bytes;
    TEST_ASSERT(qwen_cpu_mem.total_bytes > 0);
    TEST_ASSERT(qwen_cpu_mem.state_bytes == qwen_expected_state_bytes);
    TEST_ASSERT(qwen_cpu_mem.scratch_bytes == qwen_logits_bytes);
    TEST_ASSERT(qwen_cpu_mem.total_bytes == qwen_expected_state_bytes + qwen_logits_bytes);
    TEST_ASSERT(qwen_cpu_mem.model_private_bytes == qwen_recurrent_state_bytes + qwen_conv_state_bytes);
    TEST_ASSERT(rt_estimate_context_memory(qwen, RT_BACKEND_AUTO, 32).total_bytes > 0);
    rt_context_memory qwen_metal_mem =
        rt_estimate_context_memory(qwen, RT_BACKEND_METAL, 32);
    if (qwen_metal_mem.total_bytes > 0) {
        TEST_ASSERT(qwen_metal_mem.state_bytes == qwen_cpu_mem.state_bytes);
        TEST_ASSERT(qwen_metal_mem.scratch_bytes == qwen_cpu_mem.scratch_bytes);
    }
    TEST_ASSERT(rt_estimate_context_memory(qwen, RT_BACKEND_CUDA, 32).total_bytes == 0);
    TEST_ASSERT(rt_estimate_context_memory(qwen, RT_BACKEND_CPU, (int)spec->max_context + 1).total_bytes == 0);

    const mistral35_model_spec *mistral_spec = mistral35_runtime_model_spec();
    TEST_ASSERT(!strcmp(mistral_spec->hf_repo, MISTRAL35_RUNTIME_HF_REPO));
    TEST_ASSERT(!strcmp(mistral_spec->api_model, MISTRAL35_RUNTIME_API_MODEL));
    TEST_ASSERT(mistral_spec->dense_weights);
    TEST_ASSERT(mistral_spec->multimodal_checkpoint);
    TEST_ASSERT(mistral_spec->fp8_checkpoint);
    TEST_ASSERT(mistral_spec->n_layers == 88);
    TEST_ASSERT(mistral_spec->hidden_size == 12288);
    TEST_ASSERT(mistral_spec->vocab_size == 131072);
    TEST_ASSERT(mistral_spec->max_context == 262144);
    TEST_ASSERT(mistral_spec->attention_kv_heads == 8);
    TEST_ASSERT(mistral_spec->yarn_factor == 64);

    rt_engine *qwen_engine = NULL;
    rt_engine_options qwen_opt = {
        .model_path = qwen_path,
        .backend = RT_BACKEND_AUTO,
    };
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) == 0);
    TEST_ASSERT(qwen_engine != NULL);
    TEST_ASSERT(rt_engine_vocab_size(qwen_engine) == QWEN36_TEST_TOKEN_COUNT);
    rt_engine *qwen_chat_engine = NULL;
    rt_engine_options qwen_chat_opt = qwen_opt;
    qwen_chat_opt.model_path = qwen_chat_path;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_chat_engine, qwen, &qwen_chat_opt) == 0);
    TEST_ASSERT(qwen_chat_engine != NULL);
    TEST_ASSERT(rt_engine_vocab_size(qwen_chat_engine) == QWEN36_TEST_TOKEN_COUNT);
    TEST_ASSERT(rt_token_eos(qwen_engine) == 0);

    rt_tokens qwen_text = {0};
    TEST_ASSERT(rt_tokenize_text(qwen_engine, "hi", &qwen_text) == 0);
    TEST_ASSERT(qwen_text.len == 1);
    TEST_ASSERT(qwen_text.v[0] == 7);
    size_t qwen_piece_len = 0;
    char *qwen_piece = rt_token_text(qwen_engine, qwen_text.v[0], &qwen_piece_len);
    TEST_ASSERT(qwen_piece != NULL);
    TEST_ASSERT(qwen_piece_len == 2);
    TEST_ASSERT(!memcmp(qwen_piece, "hi", qwen_piece_len));
    free(qwen_piece);
    rt_tokens_free(&qwen_text);

    rt_tokens qwen_digit_two = {0};
    rt_tokens qwen_digit_three = {0};
    rt_tokens qwen_digits = {0};
    TEST_ASSERT(rt_tokenize_text(qwen_engine, "2", &qwen_digit_two) == 0);
    TEST_ASSERT(rt_tokenize_text(qwen_engine, "3", &qwen_digit_three) == 0);
    TEST_ASSERT(rt_tokenize_text(qwen_engine, "23", &qwen_digits) == 0);
    TEST_ASSERT(qwen_digit_two.len == 1);
    TEST_ASSERT(qwen_digit_three.len == 1);
    TEST_ASSERT(qwen_digits.len == 2);
    TEST_ASSERT(qwen_digits.v[0] == qwen_digit_two.v[0]);
    TEST_ASSERT(qwen_digits.v[1] == qwen_digit_three.v[0]);
    rt_tokens_free(&qwen_digit_two);
    rt_tokens_free(&qwen_digit_three);
    rt_tokens_free(&qwen_digits);

    rt_tokens qwen_contraction = {0};
    TEST_ASSERT(rt_tokenize_text(qwen_engine, "'s", &qwen_contraction) == 0);
    TEST_ASSERT(qwen_contraction.len == 1);
    qwen_piece = rt_token_text(qwen_engine, qwen_contraction.v[0],
                               &qwen_piece_len);
    TEST_ASSERT(qwen_piece != NULL);
    TEST_ASSERT(qwen_piece_len == 2);
    TEST_ASSERT(!memcmp(qwen_piece, "'s", qwen_piece_len));
    free(qwen_piece);
    rt_tokens_free(&qwen_contraction);

    rt_tokens qwen_punct_newline = {0};
    TEST_ASSERT(rt_tokenize_text(qwen_engine, "!\n", &qwen_punct_newline) == 0);
    TEST_ASSERT(qwen_punct_newline.len == 1);
    qwen_piece = rt_token_text(qwen_engine, qwen_punct_newline.v[0],
                               &qwen_piece_len);
    TEST_ASSERT(qwen_piece != NULL);
    TEST_ASSERT(qwen_piece_len == 2);
    TEST_ASSERT(!memcmp(qwen_piece, "!\n", qwen_piece_len));
    free(qwen_piece);
    rt_tokens_free(&qwen_punct_newline);

    rt_tokens qwen_unknown = {0};
    rt_tokens_push(&qwen_unknown, 7);
    TEST_ASSERT(rt_tokenize_text(qwen_engine, "\x01", &qwen_unknown) != 0);
    TEST_ASSERT(qwen_unknown.len == 1);
    TEST_ASSERT(qwen_unknown.v[0] == 7);
    rt_tokens_free(&qwen_unknown);

    FILE *qwen_vec_read = fopen(qwen_vec_path, "rb");
    TEST_ASSERT(qwen_vec_read != NULL);
    if (qwen_vec_read) {
        TEST_ASSERT(test_qwen36_validate_vector_fixture_preamble(
                        qwen_vec_read, qwen_vec_path, true, NULL));
        qwen36_vec_case qwen_vec_case = {0};
        TEST_ASSERT(test_read_qwen36_vector_case(qwen_vec_read, &qwen_vec_case));
        TEST_ASSERT(test_fill_qwen36_vector_case(qwen_vec_read, &qwen_vec_case));
        test_qwen36_vector_case(qwen_engine, &qwen_vec_case, true);
        test_qwen36_vec_case_free(&qwen_vec_case);
        fclose(qwen_vec_read);
    }

    rt_chat_message qwen_messages[] = {
        {.role = "user", .content = "hi"},
    };
    qwen36_runtime_chat_options qwen_chat = {
        .think_mode = QWEN36_THINK_ENABLED,
    };
    rt_chat_render_options qwen_render = {
        .add_generation_prompt = true,
        .model_options = &qwen_chat,
    };
    rt_tokens qwen_prompt = {0};
    TEST_ASSERT(rt_render_chat(qwen_engine, qwen_messages, 1, &qwen_render, &qwen_prompt) == 0);
    TEST_ASSERT(qwen_prompt.len == 22);
    TEST_ASSERT(qwen_prompt.v[0] == 1);
    TEST_ASSERT(qwen_prompt.v[6] == 7);
    TEST_ASSERT(qwen_prompt.v[7] == 2);
    TEST_ASSERT(qwen_prompt.v[20] == 3);
    TEST_ASSERT(qwen_prompt.v[21] == 15);
    rt_tokens_free(&qwen_prompt);

    qwen_chat.think_mode = QWEN36_THINK_DISABLED;
    TEST_ASSERT(rt_render_chat(qwen_engine, qwen_messages, 1, &qwen_render, &qwen_prompt) == 0);
    TEST_ASSERT(qwen_prompt.len == 26);
    TEST_ASSERT(qwen_prompt.v[20] == 3);
    TEST_ASSERT(qwen_prompt.v[23] == 4);
    TEST_ASSERT(qwen_prompt.v[25] == 15);
    rt_tokens_free(&qwen_prompt);

    rt_chat_message qwen_unknown_messages[] = {
        {.role = "user", .content = "\x01"},
    };
    rt_tokens_push(&qwen_prompt, 7);
    TEST_ASSERT(rt_render_chat(qwen_engine, qwen_unknown_messages, 1,
                               &qwen_render, &qwen_prompt) != 0);
    TEST_ASSERT(qwen_prompt.len == 1);
    TEST_ASSERT(qwen_prompt.v[0] == 7);
    rt_tokens_free(&qwen_prompt);

    rt_chat_message qwen_history_messages[] = {
        {.role = "developer", .content = "hi"},
        {.role = "assistant", .content = "<think>\nhi\n</think>\n\nhi"},
        {.role = "function", .content = "hi"},
        {.role = "user", .content = "hi"},
    };
    qwen_chat.think_mode = QWEN36_THINK_DISABLED;
    qwen_chat.preserve_thinking = false;
    TEST_ASSERT(rt_render_chat(qwen_chat_engine, qwen_history_messages, 4,
                               &qwen_render, &qwen_prompt) == 0);
    size_t qwen_rendered_len = 0;
    ds4_tokens qwen_rendered_tokens = {0};
    ds4_tokens_copy_rt(&qwen_rendered_tokens, &qwen_prompt);
    char *qwen_rendered_text = render_rt_tokens_text(qwen_chat_engine,
                                                     &qwen_rendered_tokens,
                                                     &qwen_rendered_len);
    TEST_ASSERT(qwen_rendered_text != NULL);
    TEST_ASSERT(qwen_rendered_len > 0);
    TEST_ASSERT(strstr(qwen_rendered_text,
                       "<|im_start|>system\nhi<|im_end|>\n") != NULL);
    TEST_ASSERT(strstr(qwen_rendered_text,
                       "<|im_start|>assistant\nhi<|im_end|>\n") != NULL);
    TEST_ASSERT(strstr(qwen_rendered_text,
                       "<think>\nhi\n</think>") == NULL);
    TEST_ASSERT(strstr(qwen_rendered_text,
                       "<|im_start|>tool\nhi<|im_end|>\n") != NULL);
    TEST_ASSERT(strstr(qwen_rendered_text,
                       "<|im_start|>assistant\n<think>\n\n</think>\n\n") != NULL);
    free(qwen_rendered_text);
    ds4_tokens_free(&qwen_rendered_tokens);
    rt_tokens_free(&qwen_prompt);

    qwen_chat.preserve_thinking = true;
    TEST_ASSERT(rt_render_chat(qwen_chat_engine, qwen_history_messages, 4,
                               &qwen_render, &qwen_prompt) == 0);
    ds4_tokens_copy_rt(&qwen_rendered_tokens, &qwen_prompt);
    qwen_rendered_text = render_rt_tokens_text(qwen_chat_engine, &qwen_rendered_tokens,
                                               &qwen_rendered_len);
    TEST_ASSERT(qwen_rendered_text != NULL);
    TEST_ASSERT(strstr(qwen_rendered_text,
                       "<|im_start|>assistant\n<think>\nhi\n</think>\n\nhi<|im_end|>\n") != NULL);
    TEST_ASSERT(strstr(qwen_rendered_text,
                       "<|im_start|>tool\nhi<|im_end|>\n") != NULL);
    free(qwen_rendered_text);
    ds4_tokens_free(&qwen_rendered_tokens);
    rt_tokens_free(&qwen_prompt);
    qwen_chat.preserve_thinking = false;

    rt_chat_message qwen_tool_tail_messages[] = {
        {.role = "user", .content = "hi"},
        {.role = "assistant", .content = "<think>\nhi\n</think>\n\nhi"},
        {.role = "user", .content = "<tool_response>\nhi\n</tool_response>"},
    };
    TEST_ASSERT(rt_render_chat(qwen_chat_engine, qwen_tool_tail_messages, 3,
                               &qwen_render, &qwen_prompt) == 0);
    ds4_tokens_copy_rt(&qwen_rendered_tokens, &qwen_prompt);
    qwen_rendered_text = render_rt_tokens_text(qwen_chat_engine,
                                               &qwen_rendered_tokens,
                                               &qwen_rendered_len);
    TEST_ASSERT(qwen_rendered_text != NULL);
    TEST_ASSERT(strstr(qwen_rendered_text,
                       "<|im_start|>assistant\n<think>\nhi\n</think>\n\nhi<|im_end|>\n") != NULL);
    TEST_ASSERT(strstr(qwen_rendered_text,
                       "<|im_start|>user\n<tool_response>\nhi\n</tool_response><|im_end|>\n") != NULL);
    free(qwen_rendered_text);
    ds4_tokens_free(&qwen_rendered_tokens);
    rt_tokens_free(&qwen_prompt);

    rt_engine_close(qwen_chat_engine);
    qwen_chat_engine = NULL;

    rt_chat_message qwen_media_messages[] = {
        {.role = "user", .content = "<|vision_start|><|image_pad|><|vision_end|>"},
    };
    TEST_ASSERT(rt_render_chat(qwen_engine, qwen_media_messages, 1, &qwen_render,
                               &qwen_prompt) != 0);
    TEST_ASSERT(qwen_prompt.len == 0);
    rt_tokens_free(&qwen_prompt);

    request qwen_media_req;
    char qwen_err[160] = {0};
    TEST_ASSERT(!parse_chat_request_rt(qwen_engine, qwen, NULL,
        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
        "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AA==\"}}]}],"
        "\"max_tokens\":1}",
        4, 64, &qwen_media_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(strstr(qwen_err, "vision placeholders") != NULL);

    rt_session *qwen_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_session, qwen_engine, 32) == 0);
    TEST_ASSERT(qwen_session != NULL);
    TEST_ASSERT(rt_session_ctx(qwen_session) == 32);
    TEST_ASSERT(rt_session_pos(qwen_session) == 0);

    rt_tokens qwen_sync = {0};
    rt_tokens_push(&qwen_sync, 7);
    TEST_ASSERT(rt_session_sync(qwen_session, &qwen_sync, qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_session) == 1);
    test_qwen36_assert_zero_last_hidden(qwen_session);
    float qwen_zero_logits[QWEN36_TEST_TOKEN_COUNT] = {0};
    TEST_ASSERT(rt_session_read_logits(qwen_session, qwen_zero_logits,
                                       QWEN36_TEST_TOKEN_COUNT) == 0);
    TEST_ASSERT(rt_session_argmax(qwen_session) == 0);
    for (uint32_t i = 0; i < QWEN36_TEST_TOKEN_COUNT; i++) {
        TEST_ASSERT(qwen_zero_logits[i] == 0.0f);
    }
    uint64_t qwen_payload_bytes = rt_session_payload_bytes(qwen_session);
    TEST_ASSERT(qwen_payload_bytes > 48);

    FILE *qwen_payload = tmpfile();
    TEST_ASSERT(qwen_payload != NULL);
    TEST_ASSERT(rt_session_save_payload(qwen_session, qwen_payload, qwen_err, sizeof(qwen_err)) == 0);
    uint64_t qwen_expected_payload_bytes =
        QWEN36_TEST_SESSION_WORDS * sizeof(uint32_t) +
        3u * QWEN36_TEST_SESSION_SECTION_WORDS * sizeof(uint32_t) +
        sizeof(uint32_t) +
        QWEN36_TEST_TOKEN_COUNT * sizeof(float) +
        QWEN36_TEST_HIDDEN * sizeof(float);
    TEST_ASSERT(qwen_payload_bytes == qwen_expected_payload_bytes);
    TEST_ASSERT(fseek(qwen_payload, 0, SEEK_SET) == 0);
    uint8_t qwen_payload_header[QWEN36_TEST_SESSION_WORDS * sizeof(uint32_t)];
    TEST_ASSERT(fread(qwen_payload_header, 1, sizeof(qwen_payload_header), qwen_payload) ==
                sizeof(qwen_payload_header));
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 0) == QWEN36_TEST_SESSION_MAGIC);
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 4) == QWEN36_TEST_SESSION_VERSION);
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 8) == QWEN36_TEST_SESSION_WORDS);
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 36) == 3);
    TEST_ASSERT(test_read_le64_words_mem(qwen_payload_header + 40) == qwen_full_kv_bytes);
    TEST_ASSERT(test_read_le64_words_mem(qwen_payload_header + 48) == qwen_recurrent_state_bytes);
    TEST_ASSERT(test_read_le64_words_mem(qwen_payload_header + 56) == qwen_conv_state_bytes);
    uint8_t qwen_section[QWEN36_TEST_SESSION_SECTION_WORDS * sizeof(uint32_t)];
    TEST_ASSERT(fread(qwen_section, 1, sizeof(qwen_section), qwen_payload) == sizeof(qwen_section));
    TEST_ASSERT(test_read_le32_mem(qwen_section + 0) == 1);
    TEST_ASSERT(test_read_le64_words_mem(qwen_section + 8) == sizeof(uint32_t));
    TEST_ASSERT(fread(qwen_section, 1, sizeof(qwen_section), qwen_payload) == sizeof(qwen_section));
    TEST_ASSERT(test_read_le32_mem(qwen_section + 0) == 2);
    TEST_ASSERT(test_read_le64_words_mem(qwen_section + 8) ==
                QWEN36_TEST_TOKEN_COUNT * sizeof(float));
    TEST_ASSERT(fread(qwen_section, 1, sizeof(qwen_section), qwen_payload) == sizeof(qwen_section));
    TEST_ASSERT(test_read_le32_mem(qwen_section + 0) == 6);
    TEST_ASSERT(test_read_le64_words_mem(qwen_section + 8) ==
                QWEN36_TEST_HIDDEN * sizeof(float));
    TEST_ASSERT(fseek(qwen_payload, 0, SEEK_SET) == 0);

    rt_session *qwen_loaded_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_loaded_session, qwen_engine, 32) == 0);
    TEST_ASSERT(rt_session_load_payload(qwen_loaded_session, qwen_payload, qwen_payload_bytes,
                                        qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_loaded_session) == 1);
    TEST_ASSERT(rt_session_payload_bytes(qwen_loaded_session) == qwen_expected_payload_bytes);
    test_qwen36_assert_zero_last_hidden(qwen_loaded_session);
    memset(qwen_zero_logits, 42, sizeof(qwen_zero_logits));
    TEST_ASSERT(rt_session_read_logits(qwen_loaded_session, qwen_zero_logits,
                                       QWEN36_TEST_TOKEN_COUNT) == 0);
    TEST_ASSERT(qwen_zero_logits[0] == 0.0f);
    qwen_err[0] = '\0';
    TEST_ASSERT(rt_session_eval(qwen_loaded_session, 7, qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_loaded_session) == 2);
    TEST_ASSERT(rt_session_argmax(qwen_loaded_session) == 0);
    rt_session_rewind(qwen_loaded_session, 0);
    TEST_ASSERT(rt_session_pos(qwen_loaded_session) == 0);

    FILE *qwen_logits_payload = tmpfile();
    TEST_ASSERT(qwen_logits_payload != NULL);
    uint64_t qwen_logits_payload_bytes = test_write_qwen36_logits_payload(qwen_logits_payload, 32, 7);
    TEST_ASSERT(fseek(qwen_logits_payload, 0, SEEK_SET) == 0);

    rt_session *qwen_logits_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_logits_session, qwen_engine, 32) == 0);
    TEST_ASSERT(rt_session_load_payload(qwen_logits_session, qwen_logits_payload,
                                        qwen_logits_payload_bytes, qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_logits_session) == 1);
    TEST_ASSERT(rt_session_argmax(qwen_logits_session) == 4);
    TEST_ASSERT(rt_session_sample(qwen_logits_session, 0.0f, 0, 1.0f, 0.0f, NULL) == 4);
    float qwen_logits[QWEN36_TEST_TOKEN_COUNT] = {0};
    TEST_ASSERT(rt_session_read_logits(qwen_logits_session, qwen_logits,
                                       QWEN36_TEST_TOKEN_COUNT) == 0);
    TEST_ASSERT(qwen_logits[4] == 4.0f);
    TEST_ASSERT(qwen_logits[7] == 2.0f);

    rt_token_score top[3] = {0};
    TEST_ASSERT(rt_session_top_logprobs(qwen_logits_session, top, 3) == 3);
    TEST_ASSERT(top[0].id == 4);
    TEST_ASSERT(top[1].id == 7);
    TEST_ASSERT(top[2].id == 9);
    TEST_ASSERT(top[0].logit == 4.0f);
    TEST_ASSERT(top[0].logprob > top[1].logprob);
    TEST_ASSERT(top[1].logprob > top[2].logprob);

    rt_token_score token_score = {0};
    TEST_ASSERT(rt_session_token_logprob(qwen_logits_session, 7, &token_score) == 0);
    TEST_ASSERT(token_score.id == 7);
    TEST_ASSERT(token_score.logit == 2.0f);
    TEST_ASSERT(token_score.logprob == top[1].logprob);

    FILE *qwen_logits_v2_payload = tmpfile();
    TEST_ASSERT(qwen_logits_v2_payload != NULL);
    uint64_t qwen_logits_v2_bytes = rt_session_payload_bytes(qwen_logits_session);
    uint64_t qwen_expected_logits_v2_bytes =
        QWEN36_TEST_SESSION_WORDS * sizeof(uint32_t) +
        2u * QWEN36_TEST_SESSION_SECTION_WORDS * sizeof(uint32_t) +
        sizeof(uint32_t) +
        QWEN36_TEST_TOKEN_COUNT * sizeof(float);
    TEST_ASSERT(qwen_logits_v2_bytes == qwen_expected_logits_v2_bytes);
    TEST_ASSERT(rt_session_save_payload(qwen_logits_session, qwen_logits_v2_payload,
                                        qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(fseek(qwen_logits_v2_payload, 0, SEEK_SET) == 0);
    TEST_ASSERT(fread(qwen_payload_header, 1, sizeof(qwen_payload_header),
                      qwen_logits_v2_payload) == sizeof(qwen_payload_header));
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 4) == QWEN36_TEST_SESSION_VERSION);
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 36) == 2);
    TEST_ASSERT(fseek(qwen_logits_v2_payload, 0, SEEK_SET) == 0);
    rt_session *qwen_logits_v2_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_logits_v2_session, qwen_engine, 32) == 0);
    TEST_ASSERT(rt_session_load_payload(qwen_logits_v2_session, qwen_logits_v2_payload,
                                        qwen_logits_v2_bytes, qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_argmax(qwen_logits_v2_session) == 4);
    rt_session_free(qwen_logits_v2_session);
    fclose(qwen_logits_v2_payload);

    rt_tokens qwen_state_prompt = {0};
    rt_tokens_push(&qwen_state_prompt, 7);
    TEST_ASSERT(rt_session_sync(qwen_logits_session, &qwen_state_prompt,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_logits_session) == 1);
    TEST_ASSERT(rt_session_argmax(qwen_logits_session) == 0);
    TEST_ASSERT(rt_session_payload_bytes(qwen_logits_session) ==
                qwen_expected_payload_bytes);
    rt_tokens_push(&qwen_state_prompt, 7);
    TEST_ASSERT(rt_session_sync(qwen_logits_session, &qwen_state_prompt,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_logits_session) == 2);
    TEST_ASSERT(rt_session_argmax(qwen_logits_session) == 0);
    memset(qwen_logits, 42, sizeof(qwen_logits));
    TEST_ASSERT(rt_session_read_logits(qwen_logits_session, qwen_logits,
                                       QWEN36_TEST_TOKEN_COUNT) == 0);
    for (uint32_t i = 0; i < QWEN36_TEST_TOKEN_COUNT; i++) {
        TEST_ASSERT(qwen_logits[i] == 0.0f);
    }

    rt_session_rewind(qwen_logits_session, 1);
    TEST_ASSERT(rt_session_pos(qwen_logits_session) == 1);
    TEST_ASSERT(rt_session_read_logits(qwen_logits_session, qwen_logits,
                                       QWEN36_TEST_TOKEN_COUNT) != 0);
    float qwen_hidden_probe[QWEN36_TEST_HIDDEN] = {0};
    TEST_ASSERT(!qwen36_runtime_session_read_last_hidden(qwen_logits_session,
                                                        qwen_hidden_probe,
                                                        QWEN36_TEST_HIDDEN));
    TEST_ASSERT(rt_session_payload_bytes(qwen_logits_session) ==
                QWEN36_TEST_SESSION_WORDS * sizeof(uint32_t) +
                QWEN36_TEST_SESSION_SECTION_WORDS * sizeof(uint32_t) +
                sizeof(uint32_t));
    qwen_state_prompt.len = 1;
    TEST_ASSERT(rt_session_sync(qwen_logits_session, &qwen_state_prompt,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_logits_session) == 1);
    TEST_ASSERT(rt_session_argmax(qwen_logits_session) == 0);
    TEST_ASSERT(rt_session_payload_bytes(qwen_logits_session) ==
                qwen_expected_payload_bytes);

    rt_tokens qwen_divergent_prompt = {0};
    rt_tokens_push(&qwen_divergent_prompt, 5);
    rt_tokens_push(&qwen_divergent_prompt, 7);
    TEST_ASSERT(rt_session_sync(qwen_logits_session, &qwen_divergent_prompt,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_logits_session) == 2);
    const rt_tokens *qwen_state_tokens = rt_session_tokens(qwen_logits_session);
    TEST_ASSERT(qwen_state_tokens != NULL);
    TEST_ASSERT(qwen_state_tokens->len == 2);
    TEST_ASSERT(qwen_state_tokens->v[0] == 5);
    TEST_ASSERT(qwen_state_tokens->v[1] == 7);
    TEST_ASSERT(rt_session_argmax(qwen_logits_session) == 0);
    rt_tokens_free(&qwen_divergent_prompt);
    rt_tokens_free(&qwen_state_prompt);

    FILE *qwen_full_kv_payload = tmpfile();
    TEST_ASSERT(qwen_full_kv_payload != NULL);
    uint64_t qwen_full_kv_payload_bytes =
        test_write_qwen36_full_kv_payload(qwen_full_kv_payload, 2, 7);
    TEST_ASSERT(fseek(qwen_full_kv_payload, 0, SEEK_SET) == 0);
    rt_session *qwen_full_kv_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_full_kv_session, qwen_engine, 2) == 0);
    TEST_ASSERT(rt_session_load_payload(qwen_full_kv_session, qwen_full_kv_payload,
                                        qwen_full_kv_payload_bytes,
                                        qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_full_kv_session) == 1);
    TEST_ASSERT(rt_session_payload_bytes(qwen_full_kv_session) == qwen_full_kv_payload_bytes);
    FILE *qwen_full_kv_resaved = tmpfile();
    TEST_ASSERT(qwen_full_kv_resaved != NULL);
    TEST_ASSERT(rt_session_save_payload(qwen_full_kv_session, qwen_full_kv_resaved,
                                        qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(fseek(qwen_full_kv_resaved, 0, SEEK_SET) == 0);
    TEST_ASSERT(fread(qwen_payload_header, 1, sizeof(qwen_payload_header),
                      qwen_full_kv_resaved) == sizeof(qwen_payload_header));
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 36) == 2);
    TEST_ASSERT(test_read_le64_words_mem(qwen_payload_header + 40) ==
                test_qwen36_full_kv_bytes(2));
    rt_session_rewind(qwen_full_kv_session, 0);
    TEST_ASSERT(rt_session_pos(qwen_full_kv_session) == 0);
    TEST_ASSERT(rt_session_payload_bytes(qwen_full_kv_session) ==
                QWEN36_TEST_SESSION_WORDS * sizeof(uint32_t) +
                QWEN36_TEST_SESSION_SECTION_WORDS * sizeof(uint32_t));
    TEST_ASSERT(fseek(qwen_full_kv_payload, 0, SEEK_SET) == 0);
    rt_session *qwen_full_kv_mismatch = NULL;
    TEST_ASSERT(rt_session_create(&qwen_full_kv_mismatch, qwen_engine, 3) == 0);
    qwen_err[0] = '\0';
    TEST_ASSERT(rt_session_load_payload(qwen_full_kv_mismatch, qwen_full_kv_payload,
                                        qwen_full_kv_payload_bytes,
                                        qwen_err, sizeof(qwen_err)) != 0);
    TEST_ASSERT(strstr(qwen_err, "matching session context") != NULL);
    rt_session_free(qwen_full_kv_mismatch);
    rt_session_free(qwen_full_kv_session);
    fclose(qwen_full_kv_resaved);
    fclose(qwen_full_kv_payload);

    FILE *qwen_full_kv_v2_payload = tmpfile();
    TEST_ASSERT(qwen_full_kv_v2_payload != NULL);
    uint64_t qwen_full_kv_v2_payload_bytes =
        test_write_qwen36_full_kv_payload_version(
            qwen_full_kv_v2_payload, 2, 7, QWEN36_TEST_SESSION_V2_VERSION);
    TEST_ASSERT(fseek(qwen_full_kv_v2_payload, 0, SEEK_SET) == 0);
    rt_session *qwen_full_kv_v2_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_full_kv_v2_session, qwen_engine, 2) == 0);
    TEST_ASSERT(rt_session_load_payload(qwen_full_kv_v2_session,
                                        qwen_full_kv_v2_payload,
                                        qwen_full_kv_v2_payload_bytes,
                                        qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_full_kv_v2_session) == 1);
    TEST_ASSERT(rt_session_payload_bytes(qwen_full_kv_v2_session) ==
                qwen_full_kv_v2_payload_bytes);
    rt_session_free(qwen_full_kv_v2_session);
    fclose(qwen_full_kv_v2_payload);

    rt_engine *qwen_state_engine = NULL;
    rt_engine_options qwen_state_opt = {
        .model_path = qwen_state_path,
        .backend = RT_BACKEND_AUTO,
    };
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_state_engine, qwen,
                                        &qwen_state_opt) == 0);
    TEST_ASSERT(qwen_state_engine != NULL);
    rt_session *qwen_full_attn_state_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_full_attn_state_session,
                                  qwen_state_engine, 64) == 0);
    rt_tokens qwen_full_attn_prompt = {0};
    rt_tokens_push(&qwen_full_attn_prompt, 7);
    qwen_err[0] = '\0';
    TEST_ASSERT(rt_session_sync(qwen_full_attn_state_session,
                                &qwen_full_attn_prompt,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_full_attn_state_session) == 1);
    float qwen_full_attn_hidden_first =
        test_qwen36_read_first_last_hidden(qwen_full_attn_state_session);
    TEST_ASSERT(qwen_full_attn_hidden_first > 1.0f);
    rt_tokens_free(&qwen_full_attn_prompt);
    rt_session_free(qwen_full_attn_state_session);

    rt_session *qwen_long_state_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_long_state_session,
                                  qwen_state_engine, 64) == 0);
    rt_tokens qwen_long_prompt = {0};
    for (int i = 0; i < 16; i++) rt_tokens_push(&qwen_long_prompt, 7);
    qwen_err[0] = '\0';
    TEST_ASSERT(rt_session_sync(qwen_long_state_session, &qwen_long_prompt,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_long_state_session) == qwen_long_prompt.len);
    float qwen_long_hidden_first =
        test_qwen36_read_first_last_hidden(qwen_long_state_session);
    TEST_ASSERT(qwen_long_hidden_first > 30.0f);
    uint64_t qwen_long_expected_payload_bytes =
        QWEN36_TEST_SESSION_WORDS * sizeof(uint32_t) +
        4u * QWEN36_TEST_SESSION_SECTION_WORDS * sizeof(uint32_t) +
        (uint64_t)qwen_long_prompt.len * sizeof(uint32_t) +
        QWEN36_TEST_TOKEN_COUNT * sizeof(float) +
        QWEN36_TEST_HIDDEN * sizeof(float) +
        test_qwen36_full_kv_bytes(64);
    TEST_ASSERT(rt_session_payload_bytes(qwen_long_state_session) ==
                qwen_long_expected_payload_bytes);
    FILE *qwen_long_payload = tmpfile();
    TEST_ASSERT(qwen_long_payload != NULL);
    TEST_ASSERT(rt_session_save_payload(qwen_long_state_session,
                                        qwen_long_payload,
                                        qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(fseek(qwen_long_payload, 0, SEEK_SET) == 0);
    TEST_ASSERT(fread(qwen_payload_header, 1, sizeof(qwen_payload_header),
                      qwen_long_payload) == sizeof(qwen_payload_header));
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 4) ==
                QWEN36_TEST_SESSION_VERSION);
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 16) ==
                (uint32_t)qwen_long_prompt.len);
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 36) == 4);
    TEST_ASSERT(test_read_le64_words_mem(qwen_payload_header + 40) ==
                test_qwen36_full_kv_bytes(64));
    TEST_ASSERT(fseek(qwen_long_payload, 0, SEEK_SET) == 0);
    rt_session *qwen_long_loaded_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_long_loaded_session,
                                  qwen_state_engine, 64) == 0);
    TEST_ASSERT(rt_session_load_payload(qwen_long_loaded_session,
                                        qwen_long_payload,
                                        qwen_long_expected_payload_bytes,
                                        qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_long_loaded_session) ==
                qwen_long_prompt.len);
    TEST_ASSERT(rt_session_payload_bytes(qwen_long_loaded_session) ==
                qwen_long_expected_payload_bytes);
    TEST_ASSERT(fabsf(test_qwen36_read_first_last_hidden(qwen_long_loaded_session) -
                      qwen_long_hidden_first) < 0.0001f);
    qwen_err[0] = '\0';
    TEST_ASSERT(rt_session_eval(qwen_long_loaded_session, 5,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_long_loaded_session) ==
                qwen_long_prompt.len + 1);
    TEST_ASSERT(test_qwen36_read_first_last_hidden(qwen_long_loaded_session) >
                30.0f);
    TEST_ASSERT(rt_session_payload_bytes(qwen_long_loaded_session) ==
                qwen_long_expected_payload_bytes + sizeof(uint32_t));
    rt_session_free(qwen_long_loaded_session);
    fclose(qwen_long_payload);
    rt_tokens_free(&qwen_long_prompt);
    rt_session_free(qwen_long_state_session);
    rt_engine_close(qwen_state_engine);

    rt_engine *qwen_recurrent_engine = NULL;
    rt_engine_options qwen_recurrent_opt = {
        .model_path = qwen_recurrent_path,
        .backend = RT_BACKEND_AUTO,
    };
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_recurrent_engine, qwen,
                                        &qwen_recurrent_opt) == 0);
    TEST_ASSERT(qwen_recurrent_engine != NULL);
    rt_session *qwen_recurrent_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_recurrent_session,
                                  qwen_recurrent_engine, 32) == 0);
    rt_tokens qwen_recurrent_prompt = {0};
    rt_tokens_push(&qwen_recurrent_prompt, 7);
    qwen_err[0] = '\0';
    TEST_ASSERT(rt_session_sync(qwen_recurrent_session,
                                &qwen_recurrent_prompt,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_recurrent_session) == 1);
    float qwen_recurrent_hidden_first =
        test_qwen36_read_first_last_hidden(qwen_recurrent_session);
    TEST_ASSERT(qwen_recurrent_hidden_first > 1.0f);
    uint64_t qwen_recurrent_expected_payload_bytes =
        QWEN36_TEST_SESSION_WORDS * sizeof(uint32_t) +
        5u * QWEN36_TEST_SESSION_SECTION_WORDS * sizeof(uint32_t) +
        sizeof(uint32_t) +
        QWEN36_TEST_TOKEN_COUNT * sizeof(float) +
        QWEN36_TEST_HIDDEN * sizeof(float) +
        test_qwen36_recurrent_state_bytes() +
        test_qwen36_conv_state_bytes();
    TEST_ASSERT(rt_session_payload_bytes(qwen_recurrent_session) ==
                qwen_recurrent_expected_payload_bytes);
    FILE *qwen_recurrent_payload = tmpfile();
    TEST_ASSERT(qwen_recurrent_payload != NULL);
    TEST_ASSERT(rt_session_save_payload(qwen_recurrent_session,
                                        qwen_recurrent_payload,
                                        qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(fseek(qwen_recurrent_payload, 0, SEEK_SET) == 0);
    TEST_ASSERT(fread(qwen_payload_header, 1, sizeof(qwen_payload_header),
                      qwen_recurrent_payload) == sizeof(qwen_payload_header));
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 4) ==
                QWEN36_TEST_SESSION_VERSION);
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 16) == 1);
    TEST_ASSERT(test_read_le32_mem(qwen_payload_header + 36) == 5);
    TEST_ASSERT(test_read_le64_words_mem(qwen_payload_header + 48) ==
                test_qwen36_recurrent_state_bytes());
    TEST_ASSERT(test_read_le64_words_mem(qwen_payload_header + 56) ==
                test_qwen36_conv_state_bytes());
    TEST_ASSERT(fseek(qwen_recurrent_payload, 0, SEEK_SET) == 0);
    rt_session *qwen_recurrent_loaded_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_recurrent_loaded_session,
                                  qwen_recurrent_engine, 32) == 0);
    TEST_ASSERT(rt_session_load_payload(qwen_recurrent_loaded_session,
                                        qwen_recurrent_payload,
                                        qwen_recurrent_expected_payload_bytes,
                                        qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_recurrent_loaded_session) == 1);
    TEST_ASSERT(rt_session_payload_bytes(qwen_recurrent_loaded_session) ==
                qwen_recurrent_expected_payload_bytes);
    TEST_ASSERT(fabsf(test_qwen36_read_first_last_hidden(
                          qwen_recurrent_loaded_session) -
                      qwen_recurrent_hidden_first) < 0.0001f);
    qwen_err[0] = '\0';
    TEST_ASSERT(rt_session_eval(qwen_recurrent_loaded_session, 5,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_recurrent_loaded_session) == 2);
    float qwen_recurrent_next_first =
        test_qwen36_read_first_last_hidden(qwen_recurrent_loaded_session);
    TEST_ASSERT(qwen_recurrent_next_first > 1.0f);
    TEST_ASSERT(rt_session_payload_bytes(qwen_recurrent_loaded_session) ==
                qwen_recurrent_expected_payload_bytes + sizeof(uint32_t));
    rt_session_free(qwen_recurrent_loaded_session);

    char qwen_recurrent_kv_tmpl[] =
        "/tmp/ds4-qwen-recurrent-kv-test.XXXXXX";
    char *qwen_recurrent_kv_dir = mkdtemp(qwen_recurrent_kv_tmpl);
    TEST_ASSERT(qwen_recurrent_kv_dir != NULL);
    if (qwen_recurrent_kv_dir) {
        kv_cache_options qwen_recurrent_kv_opt = kv_cache_default_options();
        qwen_recurrent_kv_opt.min_tokens = 1;
        qwen_recurrent_kv_opt.cold_max_tokens = 64;

        server qwen_recurrent_cache_server = {0};
        qwen_recurrent_cache_server.rt_engine = qwen_recurrent_engine;
        qwen_recurrent_cache_server.rt_session = qwen_recurrent_session;
        qwen_recurrent_cache_server.rt_ops = qwen;
        TEST_ASSERT(pthread_mutex_init(&qwen_recurrent_cache_server.tool_mu,
                                       NULL) == 0);
        TEST_ASSERT(kv_cache_open(&qwen_recurrent_cache_server.kv,
                                  qwen_recurrent_kv_dir, 512, false,
                                  qwen_recurrent_kv_opt));
        ds4_tokens qwen_recurrent_cache_tokens = {0};
        ds4_tokens_push(&qwen_recurrent_cache_tokens, 7);
        TEST_ASSERT(kv_cache_store_runtime_current(
            &qwen_recurrent_cache_server, &qwen_recurrent_cache_tokens,
            "continued", "hi", 2, 0));
        ds4_tokens_free(&qwen_recurrent_cache_tokens);
        kv_cache_close(&qwen_recurrent_cache_server.kv);
        pthread_mutex_destroy(&qwen_recurrent_cache_server.tool_mu);

        rt_session *qwen_recurrent_cache_reload_session = NULL;
        TEST_ASSERT(rt_session_create(&qwen_recurrent_cache_reload_session,
                                      qwen_recurrent_engine, 32) == 0);
        server qwen_recurrent_cache_reload_server = {0};
        qwen_recurrent_cache_reload_server.rt_engine = qwen_recurrent_engine;
        qwen_recurrent_cache_reload_server.rt_session =
            qwen_recurrent_cache_reload_session;
        qwen_recurrent_cache_reload_server.rt_ops = qwen;
        TEST_ASSERT(pthread_mutex_init(
            &qwen_recurrent_cache_reload_server.tool_mu, NULL) == 0);
        TEST_ASSERT(kv_cache_open(&qwen_recurrent_cache_reload_server.kv,
                                  qwen_recurrent_kv_dir, 512, false,
                                  qwen_recurrent_kv_opt));
        rt_tokens qwen_recurrent_cache_effective = {0};
        uint8_t qwen_recurrent_cache_ext = 0;
        int qwen_recurrent_cache_hit = kv_cache_try_load_runtime_text(
            &qwen_recurrent_cache_reload_server, "his",
            &qwen_recurrent_cache_effective, &qwen_recurrent_cache_ext);
        TEST_ASSERT(qwen_recurrent_cache_hit == 1);
        TEST_ASSERT(qwen_recurrent_cache_ext == 0);
        TEST_ASSERT(qwen_recurrent_cache_effective.len == 2);
        TEST_ASSERT(qwen_recurrent_cache_effective.v[0] == 7);
        TEST_ASSERT(qwen_recurrent_cache_effective.v[1] == 9);
        TEST_ASSERT(rt_session_pos(qwen_recurrent_cache_reload_session) == 1);
        TEST_ASSERT(rt_session_payload_bytes(
                        qwen_recurrent_cache_reload_session) ==
                    qwen_recurrent_expected_payload_bytes);
        TEST_ASSERT(fabsf(test_qwen36_read_first_last_hidden(
                              qwen_recurrent_cache_reload_session) -
                          qwen_recurrent_hidden_first) < 0.0001f);
        qwen_err[0] = '\0';
        TEST_ASSERT(rt_session_eval(qwen_recurrent_cache_reload_session, 5,
                                    qwen_err, sizeof(qwen_err)) == 0);
        TEST_ASSERT(rt_session_pos(qwen_recurrent_cache_reload_session) == 2);
        TEST_ASSERT(test_qwen36_read_first_last_hidden(
                        qwen_recurrent_cache_reload_session) > 1.0f);
        TEST_ASSERT(rt_session_payload_bytes(
                        qwen_recurrent_cache_reload_session) ==
                    qwen_recurrent_expected_payload_bytes + sizeof(uint32_t));
        rt_tokens_free(&qwen_recurrent_cache_effective);
        kv_cache_close(&qwen_recurrent_cache_reload_server.kv);
        pthread_mutex_destroy(&qwen_recurrent_cache_reload_server.tool_mu);
        rt_session_free(qwen_recurrent_cache_reload_session);

        DIR *qwen_recurrent_kv_scan = opendir(qwen_recurrent_kv_dir);
        if (qwen_recurrent_kv_scan) {
            struct dirent *de;
            while ((de = readdir(qwen_recurrent_kv_scan)) != NULL) {
                if (de->d_name[0] == '.') continue;
                char *path = path_join(qwen_recurrent_kv_dir, de->d_name);
                unlink(path);
                free(path);
            }
            closedir(qwen_recurrent_kv_scan);
        }
        rmdir(qwen_recurrent_kv_dir);
    }
    fclose(qwen_recurrent_payload);
    rt_tokens_free(&qwen_recurrent_prompt);
    rt_session_free(qwen_recurrent_session);
    rt_engine_close(qwen_recurrent_engine);

    rt_tokens qwen_too_long = {0};
    for (int i = 0; i < 32; i++) rt_tokens_push(&qwen_too_long, 7);
    qwen_err[0] = '\0';
    TEST_ASSERT(rt_session_sync(qwen_session, &qwen_too_long, qwen_err, sizeof(qwen_err)) != 0);
    TEST_ASSERT(strstr(qwen_err, "exceeds context") != NULL);
    rt_session_invalidate(qwen_session);
    TEST_ASSERT(rt_session_pos(qwen_session) == 0);

    rt_session *qwen_server_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_server_session, qwen_engine, 32) == 0);
    server qwen_server = {0};
    qwen_server.rt_engine = qwen_engine;
    qwen_server.rt_session = qwen_server_session;
    qwen_server.rt_ops = qwen;
    qwen_server.default_tokens = 4;
    TEST_ASSERT(pthread_mutex_init(&qwen_server.tool_mu, NULL) == 0);

    buf qwen_model_json = {0};
    append_model_json(&qwen_model_json, &qwen_server);
    TEST_ASSERT(strstr(qwen_model_json.ptr, "\"id\":\"qwen3.6-27b\"") != NULL);
    TEST_ASSERT(strstr(qwen_model_json.ptr, "\"owned_by\":\"runtime-core\"") != NULL);
    TEST_ASSERT(strstr(qwen_model_json.ptr, "\"tools\"") == NULL);
    buf_free(&qwen_model_json);

    request qwen_server_req;
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, &qwen_server, "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}",
                                      4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(!strcmp(qwen_server_req.model, QWEN36_RUNTIME_FAMILY));
    TEST_ASSERT(qwen_server_req.prompt.len == 22);
    request_free(&qwen_server_req);

    qwen_err[0] = '\0';
    TEST_ASSERT(!parse_chat_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"deepseek-v4-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(strstr(qwen_err, "active model is qwen3.6-27b") != NULL);

    qwen_err[0] = '\0';
    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"x\","
        "\"description\":\"lookup\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"query\":{\"type\":\"string\"}}}}}]}",
        4, 4096, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_server_req.has_tools);
    TEST_ASSERT(qwen_server_req.prompt.len > 22);
    request_free(&qwen_server_req);

    const char *qwen_tool_text =
        "<think>\nneed a lookup\n</think>\n\n"
        "<tool_call>\n"
        "<function=x>\n"
        "<parameter=query>\n"
        "hi\n"
        "</parameter>\n"
        "<parameter=limit>\n"
        "2\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";
    tool_calls qwen_tool_calls = {0};
    char *qwen_tool_content = NULL;
    char *qwen_tool_reasoning = NULL;
    const char *qwen_tool_finish = "stop";
    bool qwen_tool_recovered = false;
    TEST_ASSERT(parse_generated_message_for_response(qwen_tool_text,
        true, true, true, &qwen_tool_finish, qwen_err, sizeof(qwen_err),
        &qwen_tool_content, &qwen_tool_reasoning, &qwen_tool_calls,
        &qwen_tool_recovered));
    TEST_ASSERT(qwen_tool_calls.len == 1);
    TEST_ASSERT(!strcmp(qwen_tool_calls.v[0].name, "x"));
    TEST_ASSERT(strstr(qwen_tool_calls.v[0].arguments, "\"query\": \"hi\"") != NULL);
    TEST_ASSERT(strstr(qwen_tool_calls.v[0].arguments, "\"limit\": 2") != NULL);
    TEST_ASSERT(qwen_tool_content != NULL);
    TEST_ASSERT(qwen_tool_reasoning != NULL);
    TEST_ASSERT(!qwen_tool_recovered);
    free(qwen_tool_content);
    free(qwen_tool_reasoning);
    tool_calls_free(&qwen_tool_calls);

    qwen_err[0] = '\0';
    TEST_ASSERT(parse_completion_request_rt(qwen_engine, qwen,
        "{\"model\":\"Qwen/Qwen3.6-27B\",\"prompt\":\"hi\",\"max_tokens\":1}",
        4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(!strcmp(qwen_server_req.model, QWEN36_RUNTIME_FAMILY));
    TEST_ASSERT(qwen_server_req.prompt.len == 1);
    TEST_ASSERT(qwen_server_req.prompt.v[0] == 7);
    request_free(&qwen_server_req);

    qwen_err[0] = '\0';
    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen36-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}",
        4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    int sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        job qwen_job = {0};
        qwen_job.fd = sv[0];
        qwen_job.req = qwen_server_req;
        generate_job_rt(&qwen_server, &qwen_job);
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "\"model\":\"qwen3.6-27b\"") != NULL);
        TEST_ASSERT(strstr(out, "\"finish_reason\":\"stop\"") != NULL);
        TEST_ASSERT(strstr(out, "\"prompt_tokens\":22") != NULL);
        free(out);
        request_free(&qwen_job.req);
        close(sv[0]);
        close(sv[1]);
    } else {
        request_free(&qwen_server_req);
    }

    rt_session_invalidate(qwen_server_session);
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_responses_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen3.6-27b\",\"input\":\"hi\","
        "\"max_output_tokens\":1,\"temperature\":0,"
        "\"reasoning\":{\"effort\":\"none\"}}",
        4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_server_req.api == API_RESPONSES);
    int resp_sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, resp_sv) == 0);
    if (resp_sv[0] >= 0 && resp_sv[1] >= 0) {
        job qwen_resp_job = {0};
        qwen_resp_job.fd = resp_sv[0];
        qwen_resp_job.req = qwen_server_req;
        generate_job_rt(&qwen_server, &qwen_resp_job);
        shutdown(resp_sv[0], SHUT_WR);
        char *out = read_socket_text(resp_sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "\"object\":\"response\"") != NULL);
        TEST_ASSERT(strstr(out, "\"model\":\"qwen3.6-27b\"") != NULL);
        TEST_ASSERT(strstr(out, "\"usage\"") != NULL);
        free(out);
        request_free(&qwen_resp_job.req);
        close(resp_sv[0]);
        close(resp_sv[1]);
    } else {
        request_free(&qwen_server_req);
    }

    tool_calls qwen_prev_resp_calls = {0};
    tool_call qwen_prev_resp_call = {0};
    qwen_prev_resp_call.id = xstrdup("call_qwen_prev");
    qwen_prev_resp_call.name = xstrdup("x");
    qwen_prev_resp_call.arguments = xstrdup("{\"query\":\"hi\"}");
    tool_calls_push(&qwen_prev_resp_calls, qwen_prev_resp_call);
    responses_object_store_remember(
        &qwen_server, "resp_qwen_prev",
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nhi<|im_end|>\n",
        &qwen_prev_resp_calls);
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_responses_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen3.6-27b\",\"previous_response_id\":\"resp_qwen_prev\","
        "\"input\":[{\"type\":\"function_call_output\","
        "\"call_id\":\"call_qwen_prev\",\"output\":\"hi\"}],"
        "\"max_output_tokens\":1,\"reasoning\":{\"effort\":\"none\"}}",
        4, 64, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(!qwen_server_req.responses_requires_live_tool_state);
    TEST_ASSERT(qwen_server_req.responses_previous_visible_text != NULL);
    TEST_ASSERT(qwen_server_req.prompt_text != NULL);
    TEST_ASSERT(strstr(qwen_server_req.prompt_text,
                       "<|im_start|>assistant\nhi<|im_end|>") != NULL);
    TEST_ASSERT(strstr(qwen_server_req.prompt_text, "<tool_response>") != NULL);
    TEST_ASSERT(strstr(qwen_server_req.prompt_text,
                       "<|im_start|>assistant\n<think>\n\n</think>") != NULL);
    request_free(&qwen_server_req);

    responses_object_store_remember(
        &qwen_server, "conv_qwen_prev",
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nhi<|im_end|>\n",
        &qwen_prev_resp_calls);
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_responses_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen3.6-27b\",\"conversation\":{\"id\":\"conv_qwen_prev\","
        "\"metadata\":{\"ignored\":true}},"
        "\"input\":[{\"type\":\"function_call_output\","
        "\"call_id\":\"call_qwen_prev\",\"output\":\"hi\"}],"
        "\"max_output_tokens\":1,\"reasoning\":{\"effort\":\"none\"}}",
        4, 64, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_server_req.responses_conversation_id != NULL);
    TEST_ASSERT(strcmp(qwen_server_req.responses_conversation_id,
                       "conv_qwen_prev") == 0);
    TEST_ASSERT(qwen_server_req.responses_previous_visible_text != NULL);
    TEST_ASSERT(strstr(qwen_server_req.prompt_text,
                       "<|im_start|>assistant\nhi<|im_end|>") != NULL);
    TEST_ASSERT(strstr(qwen_server_req.prompt_text, "<tool_response>") != NULL);
    request_free(&qwen_server_req);

    qwen_err[0] = '\0';
    TEST_ASSERT(parse_responses_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen3.6-27b\",\"conversation\":\"conv_qwen_new\","
        "\"input\":\"hi\",\"max_output_tokens\":1,"
        "\"reasoning\":{\"effort\":\"none\"}}",
        4, 64, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_server_req.responses_conversation_id != NULL);
    TEST_ASSERT(strcmp(qwen_server_req.responses_conversation_id,
                       "conv_qwen_new") == 0);
    TEST_ASSERT(qwen_server_req.responses_previous_visible_text == NULL);
    request_free(&qwen_server_req);

    TEST_ASSERT(conversation_store_upsert_full(
        &qwen_server, "conv_qwen_items", "{}",
        "[{\"id\":\"msg_qwen_seed\",\"status\":\"completed\","
        "\"type\":\"message\",\"role\":\"user\","
        "\"content\":[{\"type\":\"input_text\",\"text\":\"seed item\"}]}]",
        0));
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_responses_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen3.6-27b\",\"conversation\":\"conv_qwen_items\","
        "\"input\":\"current item\",\"max_output_tokens\":1,"
        "\"reasoning\":{\"effort\":\"none\"}}",
        4, 96, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_server_req.responses_conversation_id != NULL);
    TEST_ASSERT(qwen_server_req.responses_previous_visible_text == NULL);
    TEST_ASSERT(qwen_server_req.prompt_text != NULL);
    TEST_ASSERT(strstr(qwen_server_req.prompt_text, "seed item") != NULL);
    TEST_ASSERT(strstr(qwen_server_req.prompt_text, "current item") != NULL);
    TEST_ASSERT(strstr(qwen_server_req.prompt_text, "seed item") <
                strstr(qwen_server_req.prompt_text, "current item"));
    request_free(&qwen_server_req);

    qwen_err[0] = '\0';
    TEST_ASSERT(!parse_responses_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen3.6-27b\",\"previous_response_id\":\"resp_qwen_prev\","
        "\"conversation\":\"conv_qwen_prev\",\"input\":\"hi\","
        "\"max_output_tokens\":1,\"reasoning\":{\"effort\":\"none\"}}",
        4, 64, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(strstr(qwen_err,
                       "conversation cannot be combined with previous_response_id") != NULL);
    tool_calls_free(&qwen_prev_resp_calls);

    rt_session_invalidate(qwen_server_session);
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_anthropic_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"Qwen/Qwen3.6-27B\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        "\"max_tokens\":1,\"temperature\":0,"
        "\"thinking\":{\"type\":\"disabled\"}}",
        4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_server_req.api == API_ANTHROPIC);
    int anth_sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, anth_sv) == 0);
    if (anth_sv[0] >= 0 && anth_sv[1] >= 0) {
        job qwen_anth_job = {0};
        qwen_anth_job.fd = anth_sv[0];
        qwen_anth_job.req = qwen_server_req;
        generate_job_rt(&qwen_server, &qwen_anth_job);
        shutdown(anth_sv[0], SHUT_WR);
        char *out = read_socket_text(anth_sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"message\"") != NULL);
        TEST_ASSERT(strstr(out, "\"model\":\"qwen3.6-27b\"") != NULL);
        TEST_ASSERT(strstr(out, "\"stop_reason\":\"end_turn\"") != NULL);
        free(out);
        request_free(&qwen_anth_job.req);
        close(anth_sv[0]);
        close(anth_sv[1]);
    } else {
        request_free(&qwen_server_req);
    }

    qwen_err[0] = '\0';
    TEST_ASSERT(parse_responses_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen3.6-27b\",\"input\":\"hi\","
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"x\","
        "\"parameters\":{\"type\":\"object\"}}}]}",
        4, 4096, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_server_req.has_tools);
    TEST_ASSERT(qwen_server_req.api == API_RESPONSES);
    request_free(&qwen_server_req);

    rt_session_invalidate(qwen_server_session);
    rt_tokens qwen_live_prefix = {0};
    rt_tokens_push(&qwen_live_prefix, 7);
    qwen_err[0] = '\0';
    TEST_ASSERT(rt_session_sync(qwen_server_session, &qwen_live_prefix,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_server_session) == 1);

    tool_calls qwen_live_resp_calls = {0};
    tool_call qwen_live_resp_call = {0};
    qwen_live_resp_call.id = xstrdup("call_qwen_live");
    qwen_live_resp_call.name = xstrdup("x");
    qwen_live_resp_call.arguments = xstrdup("{}");
    tool_calls_push(&qwen_live_resp_calls, qwen_live_resp_call);
    responses_live_remember_at(&qwen_server, "visible-qwen-response",
                               &qwen_live_resp_calls,
                               rt_session_pos(qwen_server_session));
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_responses_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen3.6-27b\",\"input\":[{\"type\":\"function_call_output\","
        "\"call_id\":\"call_qwen_live\",\"output\":\"ok\"}],"
        "\"max_output_tokens\":1,\"reasoning\":{\"effort\":\"none\"}}",
        4, 64, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_server_req.responses_requires_live_tool_state);
    TEST_ASSERT(qwen_server_req.responses_live_suffix_text != NULL);
    TEST_ASSERT(strstr(qwen_server_req.responses_live_suffix_text, "<|im_end|>") != NULL);
    TEST_ASSERT(strstr(qwen_server_req.responses_live_suffix_text, "<｜end") == NULL);
    int qwen_live_ids = 0;
    rt_tokens qwen_effective = {0};
    TEST_ASSERT(responses_live_continuation_prompt_rt(
        &qwen_server, &qwen_server_req, rt_session_pos(qwen_server_session),
        &qwen_effective, &qwen_live_ids) == 1);
    TEST_ASSERT(qwen_live_ids == 1);
    TEST_ASSERT(qwen_effective.len > qwen_live_prefix.len);
    TEST_ASSERT(qwen_effective.v[0] == qwen_live_prefix.v[0]);
    rt_tokens_free(&qwen_effective);
    request_free(&qwen_server_req);

    qwen_err[0] = '\0';
    TEST_ASSERT(!parse_responses_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen3.6-27b\",\"input\":[{\"type\":\"function_call_output\","
        "\"call_id\":\"call_qwen_missing\",\"output\":\"ok\"}],"
        "\"max_output_tokens\":1,\"reasoning\":{\"effort\":\"none\"}}",
        4, 64, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(strstr(qwen_err, "Responses continuation state is not available") != NULL);
    tool_calls_free(&qwen_live_resp_calls);

    tool_calls qwen_live_anth_calls = {0};
    tool_call qwen_live_anth_call = {0};
    qwen_live_anth_call.id = xstrdup("toolu_qwen_live");
    qwen_live_anth_call.name = xstrdup("x");
    qwen_live_anth_call.arguments = xstrdup("{}");
    tool_calls_push(&qwen_live_anth_calls, qwen_live_anth_call);
    anthropic_live_remember_at(&qwen_server, &qwen_live_anth_calls,
                               rt_session_pos(qwen_server_session));
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_anthropic_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
        "\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_qwen_live\","
        "\"content\":\"ok\"}]}],\"max_tokens\":1,"
        "\"thinking\":{\"type\":\"disabled\"}}",
        4, 64, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_server_req.anthropic_requires_live_tool_state);
    TEST_ASSERT(qwen_server_req.anthropic_live_suffix_text != NULL);
    TEST_ASSERT(strstr(qwen_server_req.anthropic_live_suffix_text, "<|im_end|>") != NULL);
    TEST_ASSERT(strstr(qwen_server_req.anthropic_live_suffix_text, "<｜end") == NULL);
    qwen_live_ids = 0;
    TEST_ASSERT(anthropic_live_continuation_prompt_rt(
        &qwen_server, &qwen_server_req, rt_session_pos(qwen_server_session),
        &qwen_effective, &qwen_live_ids) == 1);
    TEST_ASSERT(qwen_live_ids == 1);
    TEST_ASSERT(qwen_effective.len > qwen_live_prefix.len);
    TEST_ASSERT(qwen_effective.v[0] == qwen_live_prefix.v[0]);
    rt_tokens_free(&qwen_effective);
    request_free(&qwen_server_req);

    qwen_err[0] = '\0';
    TEST_ASSERT(!parse_anthropic_request_rt(qwen_engine, qwen, &qwen_server,
        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
        "\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_qwen_missing\","
        "\"content\":\"ok\"}]}],\"max_tokens\":1,"
        "\"thinking\":{\"type\":\"disabled\"}}",
        4, 64, &qwen_server_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(strstr(qwen_err, "Anthropic continuation state is not available") != NULL);
    tool_calls_free(&qwen_live_anth_calls);
    rt_tokens_free(&qwen_live_prefix);

    char qwen_kv_tmpl[] = "/tmp/ds4-qwen-rt-kv-test.XXXXXX";
    char *qwen_kv_dir = mkdtemp(qwen_kv_tmpl);
    TEST_ASSERT(qwen_kv_dir != NULL);
    if (qwen_kv_dir) {
        kv_cache_options qwen_kv_opt = kv_cache_default_options();
        qwen_kv_opt.min_tokens = 1;
        qwen_kv_opt.cold_max_tokens = 64;
        TEST_ASSERT(kv_cache_open(&qwen_server.kv, qwen_kv_dir, 16, false,
                                  qwen_kv_opt));

        qwen_err[0] = '\0';
        TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, &qwen_server,
            "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}",
            4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
        int kv_first[2] = {-1, -1};
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, kv_first) == 0);
        if (kv_first[0] >= 0 && kv_first[1] >= 0) {
            job qwen_kv_job = {0};
            qwen_kv_job.fd = kv_first[0];
            qwen_kv_job.req = qwen_server_req;
            generate_job_rt(&qwen_server, &qwen_kv_job);
            shutdown(kv_first[0], SHUT_WR);
            char *out = read_socket_text(kv_first[1]);
            TEST_ASSERT(strstr(out, "\"cached_tokens\":0") != NULL);
            free(out);
            request_free(&qwen_kv_job.req);
            close(kv_first[0]);
            close(kv_first[1]);
        } else {
            request_free(&qwen_server_req);
        }

        rt_session_invalidate(qwen_server_session);
        qwen_err[0] = '\0';
        TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, &qwen_server,
            "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}",
            4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
        int kv_second[2] = {-1, -1};
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, kv_second) == 0);
        if (kv_second[0] >= 0 && kv_second[1] >= 0) {
            job qwen_kv_job = {0};
            qwen_kv_job.fd = kv_second[0];
            qwen_kv_job.req = qwen_server_req;
            generate_job_rt(&qwen_server, &qwen_kv_job);
            shutdown(kv_second[0], SHUT_WR);
            char *out = read_socket_text(kv_second[1]);
            TEST_ASSERT(strstr(out, "\"cached_tokens\":22") != NULL);
            TEST_ASSERT(strstr(out, "\"cache_write_tokens\":0") != NULL);
            free(out);
            request_free(&qwen_kv_job.req);
            close(kv_second[0]);
            close(kv_second[1]);
        } else {
            request_free(&qwen_server_req);
        }

        rt_session *qwen_text_reload_session = NULL;
        TEST_ASSERT(rt_session_create(&qwen_text_reload_session,
                                      qwen_engine, 32) == 0);
        server qwen_text_reload_server = {0};
        qwen_text_reload_server.rt_engine = qwen_engine;
        qwen_text_reload_server.rt_session = qwen_text_reload_session;
        qwen_text_reload_server.rt_ops = qwen;
        qwen_text_reload_server.default_tokens = 4;
        TEST_ASSERT(pthread_mutex_init(&qwen_text_reload_server.tool_mu,
                                       NULL) == 0);
        TEST_ASSERT(kv_cache_open(&qwen_text_reload_server.kv, qwen_kv_dir,
                                  16, false, qwen_kv_opt));
        qwen_err[0] = '\0';
        TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen,
            &qwen_text_reload_server,
            "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":0,\"temperature\":0}",
            4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
        int kv_fresh[2] = {-1, -1};
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, kv_fresh) == 0);
        if (kv_fresh[0] >= 0 && kv_fresh[1] >= 0) {
            job qwen_kv_fresh_job = {0};
            qwen_kv_fresh_job.fd = kv_fresh[0];
            qwen_kv_fresh_job.req = qwen_server_req;
            generate_job_rt(&qwen_text_reload_server, &qwen_kv_fresh_job);
            shutdown(kv_fresh[0], SHUT_WR);
            char *out = read_socket_text(kv_fresh[1]);
            TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
            TEST_ASSERT(strstr(out, "\"cached_tokens\":22") != NULL);
            TEST_ASSERT(strstr(out, "\"cache_write_tokens\":0") != NULL);
            TEST_ASSERT(rt_session_pos(qwen_text_reload_session) == 22);
            test_qwen36_assert_zero_last_hidden(qwen_text_reload_session);
            free(out);
            request_free(&qwen_kv_fresh_job.req);
            close(kv_fresh[0]);
            close(kv_fresh[1]);
        } else {
            request_free(&qwen_server_req);
        }
        kv_cache_close(&qwen_text_reload_server.kv);
        pthread_mutex_destroy(&qwen_text_reload_server.tool_mu);
        rt_session_free(qwen_text_reload_session);

        rt_session_invalidate(qwen_server_session);
        qwen_err[0] = '\0';
        TEST_ASSERT(parse_completion_request_rt(qwen_engine, qwen,
            "{\"model\":\"qwen3.6-27b\",\"prompt\":\"hi\",\"max_tokens\":1,"
            "\"temperature\":1,\"top_k\":22,\"seed\":1}",
            4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
        int kv_gen_first[2] = {-1, -1};
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, kv_gen_first) == 0);
        if (kv_gen_first[0] >= 0 && kv_gen_first[1] >= 0) {
            job qwen_kv_job = {0};
            qwen_kv_job.fd = kv_gen_first[0];
            qwen_kv_job.req = qwen_server_req;
            generate_job_rt(&qwen_server, &qwen_kv_job);
            shutdown(kv_gen_first[0], SHUT_WR);
            char *out = read_socket_text(kv_gen_first[1]);
            TEST_ASSERT(strstr(out, "\"text\":\"s\"") != NULL);
            TEST_ASSERT(strstr(out, "\"cached_tokens\":0") != NULL);
            free(out);
            request_free(&qwen_kv_job.req);
            close(kv_gen_first[0]);
            close(kv_gen_first[1]);
        } else {
            request_free(&qwen_server_req);
        }

        rt_session_invalidate(qwen_server_session);
        qwen_err[0] = '\0';
        TEST_ASSERT(parse_completion_request_rt(qwen_engine, qwen,
            "{\"model\":\"qwen3.6-27b\",\"prompt\":\"his\",\"max_tokens\":0,\"temperature\":0}",
            4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
        int kv_gen_second[2] = {-1, -1};
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, kv_gen_second) == 0);
        if (kv_gen_second[0] >= 0 && kv_gen_second[1] >= 0) {
            job qwen_kv_job = {0};
            qwen_kv_job.fd = kv_gen_second[0];
            qwen_kv_job.req = qwen_server_req;
            generate_job_rt(&qwen_server, &qwen_kv_job);
            shutdown(kv_gen_second[0], SHUT_WR);
            char *out = read_socket_text(kv_gen_second[1]);
            TEST_ASSERT(strstr(out, "\"cached_tokens\":2") != NULL);
            TEST_ASSERT(strstr(out, "\"prompt_tokens\":2") != NULL);
            free(out);
            request_free(&qwen_kv_job.req);
            close(kv_gen_second[0]);
            close(kv_gen_second[1]);
        } else {
            request_free(&qwen_server_req);
        }

        rt_session *qwen_gen_reload_session = NULL;
        TEST_ASSERT(rt_session_create(&qwen_gen_reload_session,
                                      qwen_engine, 32) == 0);
        server qwen_gen_reload_server = {0};
        qwen_gen_reload_server.rt_engine = qwen_engine;
        qwen_gen_reload_server.rt_session = qwen_gen_reload_session;
        qwen_gen_reload_server.rt_ops = qwen;
        qwen_gen_reload_server.default_tokens = 4;
        TEST_ASSERT(pthread_mutex_init(&qwen_gen_reload_server.tool_mu,
                                       NULL) == 0);
        TEST_ASSERT(kv_cache_open(&qwen_gen_reload_server.kv, qwen_kv_dir,
                                  16, false, qwen_kv_opt));
        qwen_err[0] = '\0';
        TEST_ASSERT(parse_completion_request_rt(qwen_engine, qwen,
            "{\"model\":\"qwen3.6-27b\",\"prompt\":\"his\","
            "\"max_tokens\":0,\"temperature\":0}",
            4, 32, &qwen_server_req, qwen_err, sizeof(qwen_err)));
        int kv_gen_fresh[2] = {-1, -1};
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, kv_gen_fresh) == 0);
        if (kv_gen_fresh[0] >= 0 && kv_gen_fresh[1] >= 0) {
            job qwen_kv_gen_fresh_job = {0};
            qwen_kv_gen_fresh_job.fd = kv_gen_fresh[0];
            qwen_kv_gen_fresh_job.req = qwen_server_req;
            generate_job_rt(&qwen_gen_reload_server,
                            &qwen_kv_gen_fresh_job);
            shutdown(kv_gen_fresh[0], SHUT_WR);
            char *out = read_socket_text(kv_gen_fresh[1]);
            TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
            TEST_ASSERT(strstr(out, "\"cached_tokens\":2") != NULL);
            TEST_ASSERT(strstr(out, "\"prompt_tokens\":2") != NULL);
            TEST_ASSERT(strstr(out, "\"cache_write_tokens\":0") != NULL);
            TEST_ASSERT(rt_session_pos(qwen_gen_reload_session) == 2);
            test_qwen36_assert_zero_last_hidden(qwen_gen_reload_session);
            free(out);
            request_free(&qwen_kv_gen_fresh_job.req);
            close(kv_gen_fresh[0]);
            close(kv_gen_fresh[1]);
        } else {
            request_free(&qwen_server_req);
        }
        kv_cache_close(&qwen_gen_reload_server.kv);
        pthread_mutex_destroy(&qwen_gen_reload_server.tool_mu);
        rt_session_free(qwen_gen_reload_session);

        rt_session_invalidate(qwen_server_session);
        rt_tokens qwen_disk_rt = {0};
        rt_tokens_push(&qwen_disk_rt, 7);
        qwen_err[0] = '\0';
        TEST_ASSERT(rt_session_sync(qwen_server_session, &qwen_disk_rt,
                                    qwen_err, sizeof(qwen_err)) == 0);
        ds4_tokens qwen_disk_tokens = {0};
        ds4_tokens_push(&qwen_disk_tokens, 7);
        const char *qwen_disk_raw =
            "<tool_call>\n"
            "<function=x>\n"
            "<parameter=query>\n"
            "hi\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>";
        tool_calls qwen_disk_calls = {0};
        tool_call qwen_disk_call = {0};
        qwen_disk_call.id = xstrdup("call_qwen_disk");
        qwen_disk_call.name = xstrdup("x");
        qwen_disk_call.arguments = xstrdup("{\"query\":\"hi\"}");
        tool_calls_push(&qwen_disk_calls, qwen_disk_call);
        qwen_disk_calls.raw_dsml = xstrdup(qwen_disk_raw);
        tool_memory_remember(&qwen_server, &qwen_disk_calls);
        buf qwen_disk_key = {0};
        buf_puts(&qwen_disk_key, "<|im_start|>assistant\n");
        buf_puts(&qwen_disk_key, qwen_disk_raw);
        TEST_ASSERT(kv_cache_store_runtime_current(
            &qwen_server, &qwen_disk_tokens, "continued",
            qwen_disk_key.ptr, qwen_disk_key.len, 0));
        tool_memory_free(&qwen_server.tool_mem);
        chat_msgs qwen_disk_msgs = {0};
        chat_msg qwen_disk_msg = {0};
        qwen_disk_msg.role = xstrdup("assistant");
        qwen_disk_msg.content = xstrdup("");
        tool_call qwen_disk_replay = {0};
        qwen_disk_replay.id = xstrdup("call_qwen_disk");
        qwen_disk_replay.name = xstrdup("x");
        qwen_disk_replay.arguments = xstrdup("{\"query\":\"hi\"}");
        tool_calls_push(&qwen_disk_msg.calls, qwen_disk_replay);
        chat_msgs_push(&qwen_disk_msgs, qwen_disk_msg);
        kv_cache_restore_tool_memory_for_messages(&qwen_server, &qwen_disk_msgs);
        tool_replay_stats qwen_disk_stats = {0};
        tool_memory_attach_to_messages(&qwen_server, &qwen_disk_msgs,
                                       &qwen_disk_stats);
        TEST_ASSERT(qwen_disk_stats.disk == 1);
        TEST_ASSERT(qwen_disk_msgs.v[0].calls.raw_dsml != NULL);
        TEST_ASSERT(strstr(qwen_disk_msgs.v[0].calls.raw_dsml,
                           "<tool_call>") != NULL);
        chat_msgs_free(&qwen_disk_msgs);
        tool_calls_free(&qwen_disk_calls);
        buf_free(&qwen_disk_key);
        ds4_tokens_free(&qwen_disk_tokens);
        rt_tokens_free(&qwen_disk_rt);

        rt_session_invalidate(qwen_server_session);
        rt_tokens qwen_visible_rt = {0};
        rt_tokens_push(&qwen_visible_rt, 7);
        qwen_err[0] = '\0';
        TEST_ASSERT(rt_session_sync(qwen_server_session, &qwen_visible_rt,
                                    qwen_err, sizeof(qwen_err)) == 0);
        responses_live_remember_at(&qwen_server, "visible-qwen", NULL,
                                   rt_session_pos(qwen_server_session));
        kv_cache_store_current(&qwen_server, "evict");
        responses_live_clear(&qwen_server);
        rt_session_invalidate(qwen_server_session);
        rt_tokens qwen_visible_effective = {0};
        uint8_t qwen_visible_ext = 0;
        int qwen_visible_cached = kv_cache_try_load_runtime_text(
            &qwen_server, "visible-qwens",
            &qwen_visible_effective, &qwen_visible_ext);
        TEST_ASSERT(qwen_visible_cached == 1);
        TEST_ASSERT((qwen_visible_ext & KV_EXT_RESPONSES_VISIBLE) != 0);
        TEST_ASSERT(qwen_visible_effective.len == 2);
        TEST_ASSERT(qwen_visible_effective.v[0] == 7);
        TEST_ASSERT(qwen_visible_effective.v[1] == 9);
        rt_tokens_free(&qwen_visible_effective);
        rt_tokens_free(&qwen_visible_rt);

        kv_cache_close(&qwen_server.kv);
        DIR *qwen_kv_scan = opendir(qwen_kv_dir);
        if (qwen_kv_scan) {
            struct dirent *de;
            while ((de = readdir(qwen_kv_scan)) != NULL) {
                if (de->d_name[0] == '.') continue;
                char *path = path_join(qwen_kv_dir, de->d_name);
                unlink(path);
                free(path);
            }
            closedir(qwen_kv_scan);
        }
        rmdir(qwen_kv_dir);
    }
    responses_live_clear(&qwen_server);
    anthropic_live_clear(&qwen_server);
    responses_object_store_free(&qwen_server);
    conversation_store_free(&qwen_server);
    tool_memory_free(&qwen_server.tool_mem);
    pthread_mutex_destroy(&qwen_server.tool_mu);
    rt_session_free(qwen_server_session);

    rt_tokens_free(&qwen_too_long);
    rt_tokens_free(&qwen_sync);
    rt_session_free(qwen_logits_session);
    rt_session_free(qwen_loaded_session);
    rt_session_free(qwen_session);
    fclose(qwen_logits_payload);
    fclose(qwen_payload);

    rt_engine_close(qwen_engine);
    qwen_engine = NULL;

    qwen_opt.backend = RT_BACKEND_CPU;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) == 0);
    TEST_ASSERT(qwen_engine != NULL);
    rt_engine_close(qwen_engine);
    qwen_engine = NULL;

    qwen_opt.backend = RT_BACKEND_METAL;
    if (qwen_metal_mem.total_bytes > 0) {
        TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) == 0);
        TEST_ASSERT(qwen_engine != NULL);
        qwen36_runtime_backend_stats qwen_backend_stats = {0};
        TEST_ASSERT(qwen36_runtime_backend_stats_get(qwen_engine,
                                                     &qwen_backend_stats));
        TEST_ASSERT(qwen_backend_stats.backend == RT_BACKEND_METAL);
        TEST_ASSERT(qwen_backend_stats.metal_loaded);
        rt_engine_close(qwen_engine);
        qwen_engine = NULL;

        rt_engine_options qwen_metal_text_opt = {
            .model_path = qwen_recurrent_path,
            .backend = RT_BACKEND_METAL,
        };
        TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen,
                                            &qwen_metal_text_opt) == 0);
        TEST_ASSERT(qwen_engine != NULL);
        rt_session *qwen_metal_text_session = NULL;
        TEST_ASSERT(rt_session_create(&qwen_metal_text_session, qwen_engine,
                                      32) == 0);
        rt_tokens qwen_metal_text_prompt = {0};
        rt_tokens_push(&qwen_metal_text_prompt, 7);
        qwen_err[0] = '\0';
        TEST_ASSERT(rt_session_sync(qwen_metal_text_session,
                                    &qwen_metal_text_prompt,
                                    qwen_err, sizeof(qwen_err)) == 0);
        TEST_ASSERT(rt_session_pos(qwen_metal_text_session) == 1);
        float qwen_metal_text_hidden_first =
            test_qwen36_read_first_last_hidden(qwen_metal_text_session);
        TEST_ASSERT(qwen_metal_text_hidden_first > 1.0f);
        TEST_ASSERT(fabsf(qwen_metal_text_hidden_first -
                          qwen_recurrent_hidden_first) < 0.05f);
        qwen_err[0] = '\0';
        TEST_ASSERT(rt_session_eval(qwen_metal_text_session, 5,
                                    qwen_err, sizeof(qwen_err)) == 0);
        TEST_ASSERT(rt_session_pos(qwen_metal_text_session) == 2);
        TEST_ASSERT(test_qwen36_read_first_last_hidden(
                        qwen_metal_text_session) > 1.0f);
        memset(&qwen_backend_stats, 0, sizeof(qwen_backend_stats));
        TEST_ASSERT(qwen36_runtime_backend_stats_get(qwen_engine,
                                                     &qwen_backend_stats));
        TEST_ASSERT(qwen_backend_stats.backend == RT_BACKEND_METAL);
        TEST_ASSERT(qwen_backend_stats.metal_loaded);
        TEST_ASSERT(qwen_backend_stats.metal_matvec_calls > 0);
        TEST_ASSERT(qwen_backend_stats.metal_rms_norm_calls > 0);
        TEST_ASSERT(qwen_backend_stats.metal_silu_mul_calls > 0);
        TEST_ASSERT(qwen_backend_stats.metal_l2_norm_calls > 0);
        TEST_ASSERT(qwen_backend_stats.metal_gated_delta_calls > 0);
        TEST_ASSERT(qwen_backend_stats.metal_matvec_fallbacks == 0);
        TEST_ASSERT(qwen_backend_stats.metal_rms_norm_fallbacks == 0);
        TEST_ASSERT(qwen_backend_stats.metal_silu_mul_fallbacks == 0);
        TEST_ASSERT(qwen_backend_stats.metal_l2_norm_fallbacks == 0);
        TEST_ASSERT(qwen_backend_stats.metal_gated_delta_fallbacks == 0);
        rt_tokens_free(&qwen_metal_text_prompt);
        rt_session_free(qwen_metal_text_session);
        rt_engine_close(qwen_engine);
        qwen_engine = NULL;

        rt_engine_options qwen_metal_full_opt = {
            .model_path = qwen_state_path,
            .backend = RT_BACKEND_METAL,
        };
        TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen,
                                            &qwen_metal_full_opt) == 0);
        TEST_ASSERT(qwen_engine != NULL);
        rt_session *qwen_metal_full_session = NULL;
        TEST_ASSERT(rt_session_create(&qwen_metal_full_session, qwen_engine,
                                      64) == 0);
        rt_tokens qwen_metal_full_prompt = {0};
        rt_tokens_push(&qwen_metal_full_prompt, 7);
        qwen_err[0] = '\0';
        TEST_ASSERT(rt_session_sync(qwen_metal_full_session,
                                    &qwen_metal_full_prompt,
                                    qwen_err, sizeof(qwen_err)) == 0);
        TEST_ASSERT(rt_session_pos(qwen_metal_full_session) ==
                    qwen_metal_full_prompt.len);
        float qwen_metal_full_hidden_first =
            test_qwen36_read_first_last_hidden(qwen_metal_full_session);
        TEST_ASSERT(qwen_metal_full_hidden_first > 1.0f);
        TEST_ASSERT(fabsf(qwen_metal_full_hidden_first -
                          qwen_full_attn_hidden_first) < 0.05f);
        memset(&qwen_backend_stats, 0, sizeof(qwen_backend_stats));
        TEST_ASSERT(qwen36_runtime_backend_stats_get(qwen_engine,
                                                     &qwen_backend_stats));
        TEST_ASSERT(qwen_backend_stats.backend == RT_BACKEND_METAL);
        TEST_ASSERT(qwen_backend_stats.metal_loaded);
        TEST_ASSERT(qwen_backend_stats.metal_full_attention_calls > 0);
        TEST_ASSERT(qwen_backend_stats.metal_full_attention_fallbacks == 0);
        rt_tokens_free(&qwen_metal_full_prompt);
        rt_session_free(qwen_metal_full_session);
        rt_engine_close(qwen_engine);
        qwen_engine = NULL;

        if (!qwen_text_only) {
            qwen36_runtime_engine_options qwen_metal_extra = {
                .mmproj_path = qwen_mmproj_metal_path,
            };
            qwen_opt.model_options = &qwen_metal_extra;
            TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen,
                                                &qwen_opt) == 0);
            TEST_ASSERT(qwen_engine != NULL);
            uint64_t qwen_metal_patch_elems =
                (uint64_t)QWEN36_TEST_VISION_PATCH *
                QWEN36_TEST_VISION_PATCH * 3u;
            uint64_t qwen_metal_patch_count =
                (uint64_t)QWEN36_TEST_VISION_MERGE *
                QWEN36_TEST_VISION_MERGE;
            float *qwen_metal_patches =
                calloc((size_t)(qwen_metal_patch_count *
                                qwen_metal_patch_elems),
                       sizeof(qwen_metal_patches[0]));
            TEST_ASSERT(qwen_metal_patches != NULL);
            qwen36_runtime_vision_embedding qwen_metal_vision = {0};
            qwen_err[0] = '\0';
            TEST_ASSERT(qwen36_runtime_embed_image_patches_f32(
                            qwen_engine, qwen_metal_patches,
                            QWEN36_TEST_VISION_MERGE,
                            QWEN36_TEST_VISION_MERGE,
                            &qwen_metal_vision,
                            qwen_err, sizeof(qwen_err)) == 0);
            TEST_ASSERT(qwen_metal_vision.n_tokens == 1);
            memset(&qwen_backend_stats, 0, sizeof(qwen_backend_stats));
            TEST_ASSERT(qwen36_runtime_backend_stats_get(qwen_engine,
                                                         &qwen_backend_stats));
            TEST_ASSERT(qwen_backend_stats.metal_matvec_calls > 0);
            TEST_ASSERT(qwen_backend_stats.metal_rms_norm_calls > 0);
            TEST_ASSERT(qwen_backend_stats.metal_silu_mul_calls > 0);
            qwen36_runtime_vision_embedding_free(&qwen_metal_vision);
            free(qwen_metal_patches);
            rt_engine_close(qwen_engine);
            qwen_engine = NULL;
            qwen_opt.model_options = NULL;
        }
    } else {
        TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) != 0);
        TEST_ASSERT(qwen_engine == NULL);
    }

    qwen_opt.backend = RT_BACKEND_CUDA;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) != 0);
    TEST_ASSERT(qwen_engine == NULL);

    qwen_opt.backend = RT_BACKEND_AUTO;
    qwen36_runtime_engine_options qwen_mtp = {
        .enable_mtp = true,
        .mtp_path = qwen_mtp_path,
        .mtp_draft_tokens = 4,
    };
    qwen_opt.model_options = &qwen_mtp;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) == 0);
    TEST_ASSERT(qwen_engine != NULL);
    rt_session *qwen_mtp_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_mtp_session, qwen_engine, 32) == 0);
    rt_tokens qwen_mtp_tokens = {0};
    rt_tokens_push(&qwen_mtp_tokens, 7);
    TEST_ASSERT(rt_session_sync(qwen_mtp_session, &qwen_mtp_tokens,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_mtp_session) == 1);
    rt_tokens_free(&qwen_mtp_tokens);
    rt_session_free(qwen_mtp_session);
    rt_session *qwen_mtp_spec_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_mtp_spec_session, qwen_engine, 32) == 0);
    int qwen_mtp_out[4] = {0};
    qwen_err[0] = '\0';
    int qwen_mtp_n = rt_session_eval_speculative_argmax(
        qwen_mtp_spec_session, 7, 4, -1,
        qwen_mtp_out, 4, qwen_err, sizeof(qwen_err));
    TEST_ASSERT(qwen_mtp_n == 2);
    TEST_ASSERT(qwen_mtp_out[0] == 7);
    TEST_ASSERT(qwen_mtp_out[1] == 0);
    TEST_ASSERT(rt_session_pos(qwen_mtp_spec_session) == qwen_mtp_n);
    qwen36_runtime_mtp_stats qwen_mtp_stats = {0};
    TEST_ASSERT(qwen36_runtime_session_mtp_stats(qwen_mtp_spec_session,
                                                 &qwen_mtp_stats));
    TEST_ASSERT(qwen_mtp_stats.proposed == 1);
    TEST_ASSERT(qwen_mtp_stats.accepted == 1);
    TEST_ASSERT(qwen_mtp_stats.rejected == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_len == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_pos == 0);
    FILE *qwen_mtp_payload = tmpfile();
    TEST_ASSERT(qwen_mtp_payload != NULL);
    uint64_t qwen_mtp_payload_bytes =
        rt_session_payload_bytes(qwen_mtp_spec_session);
    TEST_ASSERT(rt_session_save_payload(qwen_mtp_spec_session,
                                        qwen_mtp_payload,
                                        qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(fseek(qwen_mtp_payload, 0, SEEK_SET) == 0);
    TEST_ASSERT(rt_session_load_payload(qwen_mtp_spec_session,
                                        qwen_mtp_payload,
                                        qwen_mtp_payload_bytes,
                                        qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_mtp_spec_session) == qwen_mtp_n);
    memset(&qwen_mtp_stats, 0, sizeof(qwen_mtp_stats));
    TEST_ASSERT(qwen36_runtime_session_mtp_stats(qwen_mtp_spec_session,
                                                 &qwen_mtp_stats));
    TEST_ASSERT(qwen_mtp_stats.proposed == 0);
    TEST_ASSERT(qwen_mtp_stats.accepted == 0);
    TEST_ASSERT(qwen_mtp_stats.rejected == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_len == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_pos == 0);
    fclose(qwen_mtp_payload);
    rt_session_free(qwen_mtp_spec_session);

    rt_session *qwen_mtp_eos_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_mtp_eos_session, qwen_engine, 32) == 0);
    int qwen_mtp_eos_out[4] = {0};
    qwen_err[0] = '\0';
    int qwen_mtp_eos_n = rt_session_eval_speculative_argmax(
        qwen_mtp_eos_session, 7, 4, 0,
        qwen_mtp_eos_out, 4, qwen_err, sizeof(qwen_err));
    TEST_ASSERT(qwen_mtp_eos_n == 2);
    TEST_ASSERT(qwen_mtp_eos_out[0] == 7);
    TEST_ASSERT(qwen_mtp_eos_out[1] == 0);
    TEST_ASSERT(rt_session_pos(qwen_mtp_eos_session) == qwen_mtp_eos_n);
    memset(&qwen_mtp_stats, 0, sizeof(qwen_mtp_stats));
    TEST_ASSERT(qwen36_runtime_session_mtp_stats(qwen_mtp_eos_session,
                                                 &qwen_mtp_stats));
    TEST_ASSERT(qwen_mtp_stats.proposed == 1);
    TEST_ASSERT(qwen_mtp_stats.accepted == 1);
    TEST_ASSERT(qwen_mtp_stats.rejected == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_len == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_pos == 0);
    rt_session_free(qwen_mtp_eos_session);
    rt_engine_close(qwen_engine);
    qwen_engine = NULL;

    qwen_mtp.mtp_margin = 1.0f;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) == 0);
    TEST_ASSERT(qwen_engine != NULL);
    rt_session *qwen_mtp_margin_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_mtp_margin_session, qwen_engine, 32) == 0);
    int qwen_mtp_margin_out[2] = {0};
    qwen_err[0] = '\0';
    int qwen_mtp_margin_n = rt_session_eval_speculative_argmax(
        qwen_mtp_margin_session, 7, 2, -1,
        qwen_mtp_margin_out, 2, qwen_err, sizeof(qwen_err));
    TEST_ASSERT(qwen_mtp_margin_n == 2);
    TEST_ASSERT(qwen_mtp_margin_out[0] == 7);
    TEST_ASSERT(qwen_mtp_margin_out[1] == 0);
    TEST_ASSERT(rt_session_pos(qwen_mtp_margin_session) ==
                qwen_mtp_margin_n);
    memset(&qwen_mtp_stats, 0, sizeof(qwen_mtp_stats));
    TEST_ASSERT(qwen36_runtime_session_mtp_stats(qwen_mtp_margin_session,
                                                 &qwen_mtp_stats));
    TEST_ASSERT(qwen_mtp_stats.proposed == 1);
    TEST_ASSERT(qwen_mtp_stats.accepted == 0);
    TEST_ASSERT(qwen_mtp_stats.rejected == 0);
    TEST_ASSERT(qwen_mtp_stats.skipped == 1);
    TEST_ASSERT(qwen_mtp_stats.draft_len == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_pos == 0);
    rt_session_free(qwen_mtp_margin_session);
    rt_engine_close(qwen_engine);
    qwen_engine = NULL;
    qwen_mtp.mtp_margin = 0.0f;

    qwen_mtp.mtp_path = qwen_mtp_multi_path;
    qwen_mtp.mtp_draft_tokens = 4;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) == 0);
    TEST_ASSERT(qwen_engine != NULL);
    rt_session *qwen_mtp_multi_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_mtp_multi_session, qwen_engine, 32) == 0);
    int qwen_mtp_multi_out[4] = {0};
    qwen_err[0] = '\0';
    int qwen_mtp_multi_n = rt_session_eval_speculative_argmax(
        qwen_mtp_multi_session, 7, 4, -1,
        qwen_mtp_multi_out, 4, qwen_err, sizeof(qwen_err));
    TEST_ASSERT(qwen_mtp_multi_n == 3);
    TEST_ASSERT(qwen_mtp_multi_out[0] == 7);
    TEST_ASSERT(qwen_mtp_multi_out[1] == 0);
    TEST_ASSERT(qwen_mtp_multi_out[2] == 0);
    TEST_ASSERT(rt_session_pos(qwen_mtp_multi_session) == qwen_mtp_multi_n);
    memset(&qwen_mtp_stats, 0, sizeof(qwen_mtp_stats));
    TEST_ASSERT(qwen36_runtime_session_mtp_stats(qwen_mtp_multi_session,
                                                 &qwen_mtp_stats));
    TEST_ASSERT(qwen_mtp_stats.proposed == 2);
    TEST_ASSERT(qwen_mtp_stats.accepted == 2);
    TEST_ASSERT(qwen_mtp_stats.rejected == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_len == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_pos == 0);
    rt_session_free(qwen_mtp_multi_session);
    rt_engine_close(qwen_engine);
    qwen_engine = NULL;

    qwen_opt.model_path = qwen_mtp_server_model_path;
    qwen_mtp.mtp_path = qwen_mtp_path;
    qwen_mtp.mtp_draft_tokens = 4;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) == 0);
    TEST_ASSERT(qwen_engine != NULL);
    rt_session *qwen_mtp_server_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_mtp_server_session,
                                  qwen_engine, 32) == 0);
    server qwen_mtp_server = {0};
    qwen_mtp_server.rt_engine = qwen_engine;
    qwen_mtp_server.rt_session = qwen_mtp_server_session;
    qwen_mtp_server.rt_ops = qwen;
    qwen_mtp_server.default_tokens = 4;
    TEST_ASSERT(pthread_mutex_init(&qwen_mtp_server.tool_mu, NULL) == 0);
    request qwen_mtp_server_req = {0};
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_completion_request_rt(qwen_engine, qwen,
        "{\"model\":\"qwen3.6-27b\",\"prompt\":\"hi\","
        "\"max_tokens\":2,\"temperature\":0}",
        4, 32, &qwen_mtp_server_req, qwen_err, sizeof(qwen_err)));
    int qwen_mtp_server_sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0,
                           qwen_mtp_server_sv) == 0);
    if (qwen_mtp_server_sv[0] >= 0 && qwen_mtp_server_sv[1] >= 0) {
        job qwen_mtp_server_job = {0};
        qwen_mtp_server_job.fd = qwen_mtp_server_sv[0];
        qwen_mtp_server_job.req = qwen_mtp_server_req;
        generate_job_rt(&qwen_mtp_server, &qwen_mtp_server_job);
        shutdown(qwen_mtp_server_sv[0], SHUT_WR);
        char *out = read_socket_text(qwen_mtp_server_sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "\"model\":\"qwen3.6-27b\"") != NULL);
        TEST_ASSERT(strstr(out, "\"finish_reason\":\"length\"") != NULL);
        TEST_ASSERT(strstr(out, "\"completion_tokens\":2") != NULL);
        free(out);
        request_free(&qwen_mtp_server_job.req);
        close(qwen_mtp_server_sv[0]);
        close(qwen_mtp_server_sv[1]);
    } else {
        request_free(&qwen_mtp_server_req);
    }
    memset(&qwen_mtp_stats, 0, sizeof(qwen_mtp_stats));
    TEST_ASSERT(qwen36_runtime_session_mtp_stats(qwen_mtp_server_session,
                                                 &qwen_mtp_stats));
    TEST_ASSERT(qwen_mtp_stats.proposed == 1);
    TEST_ASSERT(qwen_mtp_stats.accepted == 0);
    TEST_ASSERT(qwen_mtp_stats.rejected == 1);
    TEST_ASSERT(qwen_mtp_stats.draft_len == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_pos == 0);
    pthread_mutex_destroy(&qwen_mtp_server.tool_mu);
    rt_session_free(qwen_mtp_server_session);
    rt_engine_close(qwen_engine);
    qwen_engine = NULL;

    qwen_mtp.mtp_path = qwen_mtp_server_accept_path;
    qwen_mtp.mtp_draft_tokens = 4;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) == 0);
    TEST_ASSERT(qwen_engine != NULL);
    rt_session *qwen_mtp_server_accept_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_mtp_server_accept_session,
                                  qwen_engine, 32) == 0);
    server qwen_mtp_server_accept = {0};
    qwen_mtp_server_accept.rt_engine = qwen_engine;
    qwen_mtp_server_accept.rt_session = qwen_mtp_server_accept_session;
    qwen_mtp_server_accept.rt_ops = qwen;
    qwen_mtp_server_accept.default_tokens = 4;
    TEST_ASSERT(pthread_mutex_init(&qwen_mtp_server_accept.tool_mu, NULL) == 0);
    request qwen_mtp_server_accept_req = {0};
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_completion_request_rt(qwen_engine, qwen,
        "{\"model\":\"qwen3.6-27b\",\"prompt\":\"hi\","
        "\"max_tokens\":2,\"temperature\":0}",
        4, 32, &qwen_mtp_server_accept_req, qwen_err, sizeof(qwen_err)));
    int qwen_mtp_server_accept_sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0,
                           qwen_mtp_server_accept_sv) == 0);
    if (qwen_mtp_server_accept_sv[0] >= 0 &&
        qwen_mtp_server_accept_sv[1] >= 0) {
        job qwen_mtp_server_accept_job = {0};
        qwen_mtp_server_accept_job.fd = qwen_mtp_server_accept_sv[0];
        qwen_mtp_server_accept_job.req = qwen_mtp_server_accept_req;
        generate_job_rt(&qwen_mtp_server_accept,
                        &qwen_mtp_server_accept_job);
        shutdown(qwen_mtp_server_accept_sv[0], SHUT_WR);
        char *out = read_socket_text(qwen_mtp_server_accept_sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "\"model\":\"qwen3.6-27b\"") != NULL);
        TEST_ASSERT(strstr(out, "\"finish_reason\":\"length\"") != NULL);
        TEST_ASSERT(strstr(out, "\"completion_tokens\":2") != NULL);
        free(out);
        request_free(&qwen_mtp_server_accept_job.req);
        close(qwen_mtp_server_accept_sv[0]);
        close(qwen_mtp_server_accept_sv[1]);
    } else {
        request_free(&qwen_mtp_server_accept_req);
    }
    memset(&qwen_mtp_stats, 0, sizeof(qwen_mtp_stats));
    TEST_ASSERT(qwen36_runtime_session_mtp_stats(
                    qwen_mtp_server_accept_session,
                    &qwen_mtp_stats));
    TEST_ASSERT(qwen_mtp_stats.proposed == 1);
    TEST_ASSERT(qwen_mtp_stats.accepted == 1);
    TEST_ASSERT(qwen_mtp_stats.rejected == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_len == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_pos == 0);
    TEST_ASSERT(rt_session_pos(qwen_mtp_server_accept_session) == 3);

    rt_session_invalidate(qwen_mtp_server_accept_session);
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_completion_request_rt(qwen_engine, qwen,
        "{\"model\":\"qwen3.6-27b\",\"prompt\":\"hi\","
        "\"max_tokens\":1,\"temperature\":0,\"stream\":true,"
        "\"stream_options\":{\"include_usage\":true}}",
        4, 32, &qwen_mtp_server_accept_req, qwen_err, sizeof(qwen_err)));
    int qwen_stream_completion_sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0,
                           qwen_stream_completion_sv) == 0);
    if (qwen_stream_completion_sv[0] >= 0 &&
        qwen_stream_completion_sv[1] >= 0) {
        job qwen_stream_completion_job = {0};
        qwen_stream_completion_job.fd = qwen_stream_completion_sv[0];
        qwen_stream_completion_job.req = qwen_mtp_server_accept_req;
        generate_job_rt(&qwen_mtp_server_accept,
                        &qwen_stream_completion_job);
        shutdown(qwen_stream_completion_sv[0], SHUT_WR);
        char *out = read_socket_text(qwen_stream_completion_sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "Content-Type: text/event-stream") != NULL);
        TEST_ASSERT(strstr(out, "\"object\":\"text_completion\"") != NULL);
        TEST_ASSERT(strstr(out, "\"text\":\"hi\"") != NULL);
        TEST_ASSERT(strstr(out, "\"finish_reason\":\"length\"") != NULL);
        TEST_ASSERT(strstr(out, "\"choices\":[],\"usage\"") != NULL);
        TEST_ASSERT(strstr(out, "\"completion_tokens\":1") != NULL);
        TEST_ASSERT(strstr(out, "data: [DONE]") != NULL);
        free(out);
        request_free(&qwen_stream_completion_job.req);
        close(qwen_stream_completion_sv[0]);
        close(qwen_stream_completion_sv[1]);
    } else {
        request_free(&qwen_mtp_server_accept_req);
    }

    rt_session_invalidate(qwen_mtp_server_accept_session);
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, &qwen_mtp_server_accept,
        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
        "\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0,"
        "\"stream\":true,\"stream_options\":{\"include_usage\":true},"
        "\"think\":false}",
        4, 64, &qwen_mtp_server_accept_req, qwen_err, sizeof(qwen_err)));
    int qwen_stream_chat_sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0,
                           qwen_stream_chat_sv) == 0);
    if (qwen_stream_chat_sv[0] >= 0 && qwen_stream_chat_sv[1] >= 0) {
        job qwen_stream_chat_job = {0};
        qwen_stream_chat_job.fd = qwen_stream_chat_sv[0];
        qwen_stream_chat_job.req = qwen_mtp_server_accept_req;
        generate_job_rt(&qwen_mtp_server_accept, &qwen_stream_chat_job);
        shutdown(qwen_stream_chat_sv[0], SHUT_WR);
        char *out = read_socket_text(qwen_stream_chat_sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "Content-Type: text/event-stream") != NULL);
        TEST_ASSERT(strstr(out, "\"object\":\"chat.completion.chunk\"") != NULL);
        TEST_ASSERT(strstr(out, "\"role\":\"assistant\"") != NULL);
        TEST_ASSERT(strstr(out, "\"content\":\"hi\"") != NULL);
        TEST_ASSERT(strstr(out, "\"finish_reason\":\"length\"") != NULL);
        TEST_ASSERT(strstr(out, "\"choices\":[],\"usage\"") != NULL);
        TEST_ASSERT(strstr(out, "\"completion_tokens\":1") != NULL);
        TEST_ASSERT(strstr(out, "data: [DONE]") != NULL);
        free(out);
        request_free(&qwen_stream_chat_job.req);
        close(qwen_stream_chat_sv[0]);
        close(qwen_stream_chat_sv[1]);
    } else {
        request_free(&qwen_mtp_server_accept_req);
    }

    rt_session_invalidate(qwen_mtp_server_accept_session);
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_responses_request_rt(qwen_engine, qwen,
        &qwen_mtp_server_accept,
        "{\"model\":\"qwen3.6-27b\",\"input\":\"hi\","
        "\"max_output_tokens\":1,\"temperature\":0,\"stream\":true,"
        "\"reasoning\":{\"effort\":\"none\"}}",
        4, 64, &qwen_mtp_server_accept_req, qwen_err, sizeof(qwen_err)));
    int qwen_stream_responses_sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0,
                           qwen_stream_responses_sv) == 0);
    if (qwen_stream_responses_sv[0] >= 0 &&
        qwen_stream_responses_sv[1] >= 0) {
        job qwen_stream_responses_job = {0};
        qwen_stream_responses_job.fd = qwen_stream_responses_sv[0];
        qwen_stream_responses_job.req = qwen_mtp_server_accept_req;
        generate_job_rt(&qwen_mtp_server_accept,
                        &qwen_stream_responses_job);
        shutdown(qwen_stream_responses_sv[0], SHUT_WR);
        char *out = read_socket_text(qwen_stream_responses_sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "Content-Type: text/event-stream") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"response.created\"") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"response.output_text.delta\"") != NULL);
        TEST_ASSERT(strstr(out, "\"delta\":\"hi\"") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"response.incomplete\"") != NULL);
        TEST_ASSERT(strstr(out, "\"output_tokens\":1") != NULL);
        free(out);
        request_free(&qwen_stream_responses_job.req);
        close(qwen_stream_responses_sv[0]);
        close(qwen_stream_responses_sv[1]);
    } else {
        request_free(&qwen_mtp_server_accept_req);
    }

    rt_session_invalidate(qwen_mtp_server_accept_session);
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_anthropic_request_rt(qwen_engine, qwen,
        &qwen_mtp_server_accept,
        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
        "\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0,"
        "\"stream\":true,\"thinking\":{\"type\":\"disabled\"}}",
        4, 64, &qwen_mtp_server_accept_req, qwen_err, sizeof(qwen_err)));
    int qwen_stream_anthropic_sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0,
                           qwen_stream_anthropic_sv) == 0);
    if (qwen_stream_anthropic_sv[0] >= 0 &&
        qwen_stream_anthropic_sv[1] >= 0) {
        job qwen_stream_anthropic_job = {0};
        qwen_stream_anthropic_job.fd = qwen_stream_anthropic_sv[0];
        qwen_stream_anthropic_job.req = qwen_mtp_server_accept_req;
        generate_job_rt(&qwen_mtp_server_accept,
                        &qwen_stream_anthropic_job);
        shutdown(qwen_stream_anthropic_sv[0], SHUT_WR);
        char *out = read_socket_text(qwen_stream_anthropic_sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "Content-Type: text/event-stream") != NULL);
        TEST_ASSERT(strstr(out, "event: message_start") != NULL);
        TEST_ASSERT(strstr(out, "event: content_block_delta") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"text_delta\"") != NULL);
        TEST_ASSERT(strstr(out, "\"text\":\"hi\"") != NULL);
        TEST_ASSERT(strstr(out, "\"stop_reason\":\"max_tokens\"") != NULL);
        TEST_ASSERT(strstr(out, "event: message_stop") != NULL);
        free(out);
        request_free(&qwen_stream_anthropic_job.req);
        close(qwen_stream_anthropic_sv[0]);
        close(qwen_stream_anthropic_sv[1]);
    } else {
        request_free(&qwen_mtp_server_accept_req);
    }

    pthread_mutex_destroy(&qwen_mtp_server_accept.tool_mu);
    rt_session_free(qwen_mtp_server_accept_session);
    rt_engine_close(qwen_engine);
    qwen_engine = NULL;

    qwen_opt.model_path = qwen_embedded_mtp_path;
    qwen_mtp.mtp_path = NULL;
    qwen_mtp.mtp_draft_tokens = 4;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) == 0);
    TEST_ASSERT(qwen_engine != NULL);
    rt_session *qwen_embedded_mtp_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_embedded_mtp_session, qwen_engine, 32) == 0);
    int qwen_embedded_mtp_out[4] = {0};
    qwen_err[0] = '\0';
    int qwen_embedded_mtp_n = rt_session_eval_speculative_argmax(
        qwen_embedded_mtp_session, 7, 4, -1,
        qwen_embedded_mtp_out, 4, qwen_err, sizeof(qwen_err));
    TEST_ASSERT(qwen_embedded_mtp_n == 2);
    TEST_ASSERT(qwen_embedded_mtp_out[0] == 7);
    TEST_ASSERT(qwen_embedded_mtp_out[1] == 0);
    TEST_ASSERT(rt_session_pos(qwen_embedded_mtp_session) ==
                qwen_embedded_mtp_n);
    memset(&qwen_mtp_stats, 0, sizeof(qwen_mtp_stats));
    TEST_ASSERT(qwen36_runtime_session_mtp_stats(qwen_embedded_mtp_session,
                                                 &qwen_mtp_stats));
    TEST_ASSERT(qwen_mtp_stats.proposed == 1);
    TEST_ASSERT(qwen_mtp_stats.accepted == 1);
    TEST_ASSERT(qwen_mtp_stats.rejected == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_len == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_pos == 0);
    rt_session_free(qwen_embedded_mtp_session);
    rt_engine_close(qwen_engine);
    qwen_engine = NULL;

    qwen_mtp.mtp_path = qwen_mtp_path;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) != 0);
    TEST_ASSERT(qwen_engine == NULL);
    qwen_opt.model_path = qwen_path;

    qwen_opt.model_path = qwen_mtp_reject_model_path;
    qwen_mtp.mtp_path = qwen_mtp_reject_path;
    qwen_mtp.mtp_draft_tokens = 4;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) == 0);
    TEST_ASSERT(qwen_engine != NULL);
    rt_session *qwen_mtp_reject_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_mtp_reject_session, qwen_engine, 32) == 0);
    int qwen_mtp_reject_out[4] = {0};
    qwen_err[0] = '\0';
    int qwen_mtp_reject_n = rt_session_eval_speculative_argmax(
        qwen_mtp_reject_session, 7, 4, -1,
        qwen_mtp_reject_out, 4, qwen_err, sizeof(qwen_err));
    TEST_ASSERT(qwen_mtp_reject_n == 1);
    TEST_ASSERT(qwen_mtp_reject_out[0] == 7);
    TEST_ASSERT(rt_session_pos(qwen_mtp_reject_session) == 1);
    const rt_tokens *qwen_mtp_reject_tokens =
        rt_session_tokens(qwen_mtp_reject_session);
    TEST_ASSERT(qwen_mtp_reject_tokens != NULL);
    TEST_ASSERT(qwen_mtp_reject_tokens->len == 1);
    TEST_ASSERT(qwen_mtp_reject_tokens->v[0] == 7);
    memset(&qwen_mtp_stats, 0, sizeof(qwen_mtp_stats));
    TEST_ASSERT(qwen36_runtime_session_mtp_stats(qwen_mtp_reject_session,
                                                 &qwen_mtp_stats));
    TEST_ASSERT(qwen_mtp_stats.proposed == 1);
    TEST_ASSERT(qwen_mtp_stats.accepted == 0);
    TEST_ASSERT(qwen_mtp_stats.rejected == 1);
    TEST_ASSERT(qwen_mtp_stats.draft_len == 0);
    TEST_ASSERT(qwen_mtp_stats.draft_pos == 0);
    TEST_ASSERT(rt_session_argmax(qwen_mtp_reject_session) == 0);
    TEST_ASSERT(rt_session_eval(qwen_mtp_reject_session, 0,
                                qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_mtp_reject_session) == 2);
    rt_session_free(qwen_mtp_reject_session);
    rt_engine_close(qwen_engine);
    qwen_engine = NULL;
    qwen_opt.model_path = qwen_path;

    qwen_mtp.mtp_path = NULL;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) != 0);
    TEST_ASSERT(qwen_engine == NULL);

    qwen_mtp.mtp_path = qwen_bad_mtp_path;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) != 0);
    TEST_ASSERT(qwen_engine == NULL);

    qwen_opt.model_options = NULL;
    if (!qwen_text_only) {
        qwen36_runtime_engine_options qwen_extra = {
            .mmproj_path = qwen_mmproj_path,
        };
        qwen_opt.model_options = &qwen_extra;
        TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) == 0);
        TEST_ASSERT(qwen_engine != NULL);
	    qwen_chat.think_mode = QWEN36_THINK_ENABLED;
	    qwen36_runtime_vision_config qwen_vision_cfg = {0};
	    TEST_ASSERT(qwen36_runtime_vision_config_get(qwen_engine, &qwen_vision_cfg));
	    TEST_ASSERT(qwen_vision_cfg.loaded);
	    TEST_ASSERT(qwen_vision_cfg.patch_size == QWEN36_TEST_VISION_PATCH);
	    TEST_ASSERT(qwen_vision_cfg.spatial_merge_size == QWEN36_TEST_VISION_MERGE);
	    TEST_ASSERT(qwen_vision_cfg.min_pixels ==
	                4ull * QWEN36_TEST_VISION_PATCH *
	                QWEN36_TEST_VISION_MERGE *
	                QWEN36_TEST_VISION_PATCH *
	                QWEN36_TEST_VISION_MERGE);
	    TEST_ASSERT(qwen_vision_cfg.max_pixels ==
	                16384ull * QWEN36_TEST_VISION_PATCH *
	                QWEN36_TEST_VISION_MERGE *
	                QWEN36_TEST_VISION_PATCH *
	                QWEN36_TEST_VISION_MERGE);
	    TEST_ASSERT(qwen_vision_cfg.hidden_size == QWEN36_TEST_HIDDEN);
	    TEST_ASSERT(qwen_vision_cfg.image_pad_token == QWEN36_TEST_IMAGE_PAD_TOKEN);
	    TEST_ASSERT(qwen_vision_cfg.video_pad_token == QWEN36_TEST_VIDEO_PAD_TOKEN);
	    TEST_ASSERT(rt_render_chat(qwen_engine, qwen_media_messages, 1, &qwen_render,
	                               &qwen_prompt) == 0);
    TEST_ASSERT(qwen_prompt.len >= 3);
    TEST_ASSERT(qwen_prompt.v[6] == QWEN36_TEST_VISION_START_TOKEN);
    TEST_ASSERT(qwen_prompt.v[7] == QWEN36_TEST_IMAGE_PAD_TOKEN);
	    TEST_ASSERT(qwen_prompt.v[8] == QWEN36_TEST_VISION_END_TOKEN);
	    rt_tokens_free(&qwen_prompt);

	    unsigned char qwen_processor_rgb[3] = {0, 128, 255};
	    runtime_image_rgb qwen_processor_image = {
	        .width = 1,
	        .height = 1,
	        .rgb = qwen_processor_rgb,
	    };
	    float *qwen_processor_patches = NULL;
	    uint32_t qwen_processor_grid_h = 0;
	    uint32_t qwen_processor_grid_w = 0;
	    qwen_err[0] = '\0';
	    TEST_ASSERT(runtime_image_to_qwen_patches(
	                    &qwen_processor_image, &qwen_vision_cfg,
	                    NULL,
	                    &qwen_processor_patches,
	                    &qwen_processor_grid_h,
	                    &qwen_processor_grid_w,
	                    qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_processor_grid_h == 4);
	    TEST_ASSERT(qwen_processor_grid_w == 4);
	    TEST_ASSERT(fabsf(qwen_processor_patches[0] -
	                      ((0.0f / 255.0f - 0.48145466f) / 0.26862954f)) <
	                0.0001f);
	    TEST_ASSERT(fabsf(qwen_processor_patches[1] -
	                      ((128.0f / 255.0f - 0.45782750f) / 0.26130258f)) <
	                0.0001f);
	    TEST_ASSERT(fabsf(qwen_processor_patches[2] -
	                      ((255.0f / 255.0f - 0.40821073f) / 0.27577711f)) <
	                0.0001f);
	    free(qwen_processor_patches);
	    qwen_processor_patches = NULL;

	    runtime_media_ref qwen_processor_min_opts = {
	        .min_pixels = (uint64_t)QWEN36_TEST_VISION_PATCH *
	                      QWEN36_TEST_VISION_MERGE *
	                      QWEN36_TEST_VISION_PATCH *
	                      QWEN36_TEST_VISION_MERGE,
	    };
	    qwen_err[0] = '\0';
	    TEST_ASSERT(runtime_image_to_qwen_patches(
	                    &qwen_processor_image, &qwen_vision_cfg,
	                    &qwen_processor_min_opts,
	                    &qwen_processor_patches,
	                    &qwen_processor_grid_h,
	                    &qwen_processor_grid_w,
	                    qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_processor_grid_h == 2);
	    TEST_ASSERT(qwen_processor_grid_w == 2);
	    free(qwen_processor_patches);
	    qwen_processor_patches = NULL;

	    runtime_media_ref qwen_processor_resize_opts = {
	        .resized_height = 64,
	        .resized_width = 96,
	    };
	    qwen_err[0] = '\0';
	    TEST_ASSERT(runtime_image_to_qwen_patches(
	                    &qwen_processor_image, &qwen_vision_cfg,
	                    &qwen_processor_resize_opts,
	                    &qwen_processor_patches,
	                    &qwen_processor_grid_h,
	                    &qwen_processor_grid_w,
	                    qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_processor_grid_h == 4);
	    TEST_ASSERT(qwen_processor_grid_w == 6);
	    free(qwen_processor_patches);

	    uint64_t qwen_patch_elems =
        (uint64_t)QWEN36_TEST_VISION_PATCH * QWEN36_TEST_VISION_PATCH * 3u;
    uint64_t qwen_patch_count =
        (uint64_t)QWEN36_TEST_VISION_MERGE * QWEN36_TEST_VISION_MERGE;
    float *qwen_patches = calloc((size_t)(qwen_patch_count * qwen_patch_elems),
                                 sizeof(qwen_patches[0]));
    TEST_ASSERT(qwen_patches != NULL);
    qwen36_runtime_vision_embedding qwen_vision = {0};
    qwen_err[0] = '\0';
    TEST_ASSERT(qwen36_runtime_embed_image_patches_f32(
                    qwen_engine, qwen_patches,
                    QWEN36_TEST_VISION_MERGE,
                    QWEN36_TEST_VISION_MERGE,
                    &qwen_vision, qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(qwen_vision.n_tokens == 1);
    TEST_ASSERT(qwen_vision.hidden_size == QWEN36_TEST_HIDDEN);
    TEST_ASSERT(qwen_vision.data != NULL);
    for (uint32_t i = 0; i < 16; i++) {
        TEST_ASSERT(qwen_vision.data[i] == 0.0f);
    }
    qwen36_runtime_vision_embedding_free(&qwen_vision);
    free(qwen_patches);

    rt_session *qwen_embed_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_embed_session, qwen_engine, 32) == 0);
    rt_tokens qwen_embed_prompt = {0};
    rt_tokens_push(&qwen_embed_prompt, QWEN36_TEST_IMAGE_PAD_TOKEN);
    float *qwen_external_hidden = calloc(QWEN36_TEST_HIDDEN,
                                         sizeof(qwen_external_hidden[0]));
    float *qwen_last_hidden = calloc(QWEN36_TEST_HIDDEN,
                                     sizeof(qwen_last_hidden[0]));
    TEST_ASSERT(qwen_external_hidden != NULL);
    TEST_ASSERT(qwen_last_hidden != NULL);
    for (uint32_t i = 0; i < 16; i++) {
        qwen_external_hidden[i] = (float)(i + 1u);
    }
    qwen36_runtime_embedding_span qwen_embed_span = {
        .token_pos = 0,
        .n_tokens = 1,
        .hidden_size = QWEN36_TEST_HIDDEN,
        .data = qwen_external_hidden,
    };
    qwen_err[0] = '\0';
    TEST_ASSERT(qwen36_runtime_session_sync_embeddings(
                    qwen_embed_session, &qwen_embed_prompt,
                    &qwen_embed_span, 1, qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_embed_session) == 1);
    TEST_ASSERT(qwen36_runtime_session_read_last_hidden(
                    qwen_embed_session, qwen_last_hidden, QWEN36_TEST_HIDDEN));
    for (uint32_t i = 0; i < 16; i++) {
        TEST_ASSERT(qwen_last_hidden[i] == qwen_external_hidden[i]);
    }
    free(qwen_last_hidden);
    free(qwen_external_hidden);
	    rt_tokens_free(&qwen_embed_prompt);
	    rt_session_free(qwen_embed_session);

	    qwen_err[0] = '\0';
	    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	        "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
	        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/x-portable-pixmap,"
	        "P3%0A1%201%0A255%0A0%200%200%0A\"}}]}],"
	        "\"max_tokens\":1}",
	        4, 128, &qwen_media_req, qwen_err, sizeof(qwen_err)));
	    rt_tokens qwen_media_rt_prompt = {
	        .v = qwen_media_req.prompt.v,
	        .len = qwen_media_req.prompt.len,
	        .cap = qwen_media_req.prompt.cap,
	    };
	    qwen_runtime_media_spans qwen_media_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_media_req, &qwen_media_rt_prompt,
	                    &qwen_media_spans, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_media_spans.len == 1);
	    TEST_ASSERT(qwen_media_spans.spans[0].n_tokens == 4);
	    TEST_ASSERT(qwen_media_spans.spans[0].hidden_size == QWEN36_TEST_HIDDEN);
	    TEST_ASSERT(qwen_media_spans.embeddings[0].n_tokens == 4);
	    TEST_ASSERT(qwen_media_spans.embeddings[0].hidden_size == QWEN36_TEST_HIDDEN);
	    TEST_ASSERT(qwen_media_spans.spans[0].token_pos >= 0);
	    TEST_ASSERT(qwen_media_req.prompt.v[qwen_media_spans.spans[0].token_pos] ==
	                QWEN36_TEST_IMAGE_PAD_TOKEN);
	    int qwen_media_expanded_len = qwen_media_spans.prompt.len;
	    qwen_runtime_media_spans_free(&qwen_media_spans);
	    rt_session *qwen_media_session = NULL;
	    TEST_ASSERT(rt_session_create(&qwen_media_session, qwen_engine, 128) == 0);
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_session_sync_request(
	                    qwen_engine, qwen_media_session, &qwen_media_req,
	                    &qwen_media_rt_prompt, qwen_err, sizeof(qwen_err)) == 0);
	    TEST_ASSERT(rt_session_pos(qwen_media_session) == qwen_media_expanded_len);
	    rt_session_free(qwen_media_session);

    request qwen_responses_media_req = {0};
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_responses_request_rt(qwen_engine, qwen, NULL,
        "{\"model\":\"qwen3.6-27b\",\"input\":[{\"type\":\"message\","
        "\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"look\"},"
        "{\"type\":\"input_image\",\"image_url\":{\"url\":\"data:image/x-portable-pixmap,"
        "P3%0A1%201%0A255%0A1%202%203%0A\"}},"
        "{\"type\":\"input_video\",\"video_url\":{\"url\":\"data:image/x-portable-pixmap,"
        "P3%0A1%201%0A255%0A4%205%206%0A\",\"nframes\":1}}]}],"
        "\"max_output_tokens\":1,\"reasoning\":{\"effort\":\"none\"}}",
        4, 128, &qwen_responses_media_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_responses_media_req.api == API_RESPONSES);
    TEST_ASSERT(qwen_responses_media_req.media.len == 2);
    TEST_ASSERT(!qwen_responses_media_req.media.v[0].video);
    TEST_ASSERT(qwen_responses_media_req.media.v[1].video);
    rt_tokens qwen_responses_media_rt_prompt = {
        .v = qwen_responses_media_req.prompt.v,
        .len = qwen_responses_media_req.prompt.len,
        .cap = qwen_responses_media_req.prompt.cap,
    };
    qwen_runtime_media_spans qwen_responses_media_spans = {0};
    qwen_err[0] = '\0';
    TEST_ASSERT(qwen_runtime_build_media_embeddings(
                    qwen_engine, &qwen_responses_media_req,
                    &qwen_responses_media_rt_prompt,
                    &qwen_responses_media_spans,
                    qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_responses_media_spans.len == 2);
    TEST_ASSERT(qwen_responses_media_spans.spans[0].n_tokens == 4);
    TEST_ASSERT(qwen_responses_media_spans.spans[1].n_tokens == 4);
    TEST_ASSERT(qwen_responses_media_spans.prompt.v[
                    qwen_responses_media_spans.spans[0].token_pos] ==
                QWEN36_TEST_IMAGE_PAD_TOKEN);
    TEST_ASSERT(qwen_responses_media_spans.prompt.v[
                    qwen_responses_media_spans.spans[1].token_pos] ==
                QWEN36_TEST_VIDEO_PAD_TOKEN);
    int qwen_responses_media_expanded_len =
        qwen_responses_media_spans.prompt.len;
    qwen_runtime_media_spans_free(&qwen_responses_media_spans);
    rt_session *qwen_responses_media_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_responses_media_session,
                                  qwen_engine, 128) == 0);
    qwen_err[0] = '\0';
    TEST_ASSERT(qwen_runtime_session_sync_request(
                    qwen_engine, qwen_responses_media_session,
                    &qwen_responses_media_req,
                    &qwen_responses_media_rt_prompt,
                    qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_responses_media_session) ==
                qwen_responses_media_expanded_len);
    rt_session_free(qwen_responses_media_session);
    request_free(&qwen_responses_media_req);

    request qwen_anthropic_media_req = {0};
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_anthropic_request_rt(qwen_engine, qwen, NULL,
        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
        "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
        "{\"type\":\"image\",\"source\":{\"type\":\"base64\","
        "\"media_type\":\"image/x-portable-pixmap\","
        "\"data\":\"UDMKMSAxCjI1NQo4IDkgMTAK\"}},"
        "{\"type\":\"video\",\"source\":{\"type\":\"base64\","
        "\"media_type\":\"image/x-portable-pixmap\","
        "\"data\":\"UDMKMSAxCjI1NQo4IDkgMTAK\"}}]}],"
        "\"max_tokens\":1,\"thinking\":{\"type\":\"disabled\"}}",
        4, 128, &qwen_anthropic_media_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_anthropic_media_req.api == API_ANTHROPIC);
    TEST_ASSERT(qwen_anthropic_media_req.media.len == 2);
    TEST_ASSERT(!qwen_anthropic_media_req.media.v[0].video);
    TEST_ASSERT(qwen_anthropic_media_req.media.v[1].video);
    rt_tokens qwen_anthropic_media_rt_prompt = {
        .v = qwen_anthropic_media_req.prompt.v,
        .len = qwen_anthropic_media_req.prompt.len,
        .cap = qwen_anthropic_media_req.prompt.cap,
    };
    qwen_runtime_media_spans qwen_anthropic_media_spans = {0};
    qwen_err[0] = '\0';
    TEST_ASSERT(qwen_runtime_build_media_embeddings(
                    qwen_engine, &qwen_anthropic_media_req,
                    &qwen_anthropic_media_rt_prompt,
                    &qwen_anthropic_media_spans,
                    qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(qwen_anthropic_media_spans.len == 2);
    TEST_ASSERT(qwen_anthropic_media_spans.spans[0].n_tokens == 4);
    TEST_ASSERT(qwen_anthropic_media_spans.spans[1].n_tokens == 4);
    TEST_ASSERT(qwen_anthropic_media_spans.prompt.v[
                    qwen_anthropic_media_spans.spans[0].token_pos] ==
                QWEN36_TEST_IMAGE_PAD_TOKEN);
    TEST_ASSERT(qwen_anthropic_media_spans.prompt.v[
                    qwen_anthropic_media_spans.spans[1].token_pos] ==
                QWEN36_TEST_VIDEO_PAD_TOKEN);
    int qwen_anthropic_media_expanded_len =
        qwen_anthropic_media_spans.prompt.len;
    qwen_runtime_media_spans_free(&qwen_anthropic_media_spans);
    rt_session *qwen_anthropic_media_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_anthropic_media_session,
                                  qwen_engine, 128) == 0);
    qwen_err[0] = '\0';
    TEST_ASSERT(qwen_runtime_session_sync_request(
                    qwen_engine, qwen_anthropic_media_session,
                    &qwen_anthropic_media_req,
                    &qwen_anthropic_media_rt_prompt,
                    qwen_err, sizeof(qwen_err)) == 0);
    TEST_ASSERT(rt_session_pos(qwen_anthropic_media_session) ==
                qwen_anthropic_media_expanded_len);
    rt_session_free(qwen_anthropic_media_session);
    request_free(&qwen_anthropic_media_req);

    rt_session *qwen_media_api_session = NULL;
    TEST_ASSERT(rt_session_create(&qwen_media_api_session, qwen_engine, 128) == 0);
    server qwen_media_api_server = {0};
    qwen_media_api_server.rt_engine = qwen_engine;
    qwen_media_api_server.rt_session = qwen_media_api_session;
    qwen_media_api_server.rt_ops = qwen;
    qwen_media_api_server.default_tokens = 4;
    TEST_ASSERT(pthread_mutex_init(&qwen_media_api_server.tool_mu, NULL) == 0);

    request qwen_responses_media_job_req = {0};
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_responses_request_rt(qwen_engine, qwen,
        &qwen_media_api_server,
        "{\"model\":\"qwen3.6-27b\",\"input\":[{\"type\":\"message\","
        "\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"look\"},"
        "{\"type\":\"input_image\",\"image_url\":{\"url\":\"data:image/x-portable-pixmap,"
        "P3%0A1%201%0A255%0A1%202%203%0A\"}}]}],"
        "\"max_output_tokens\":0,\"reasoning\":{\"effort\":\"none\"}}",
        4, 128, &qwen_responses_media_job_req,
        qwen_err, sizeof(qwen_err)));
    rt_tokens qwen_responses_media_job_prompt = {
        .v = qwen_responses_media_job_req.prompt.v,
        .len = qwen_responses_media_job_req.prompt.len,
        .cap = qwen_responses_media_job_req.prompt.cap,
    };
    qwen_runtime_media_spans qwen_responses_media_job_spans = {0};
    qwen_err[0] = '\0';
    TEST_ASSERT(qwen_runtime_build_media_embeddings(
                    qwen_engine, &qwen_responses_media_job_req,
                    &qwen_responses_media_job_prompt,
                    &qwen_responses_media_job_spans,
                    qwen_err, sizeof(qwen_err)));
    int qwen_responses_media_job_expanded_len =
        qwen_responses_media_job_spans.prompt.len;
    qwen_runtime_media_spans_free(&qwen_responses_media_job_spans);
    int qwen_responses_media_sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0,
                           qwen_responses_media_sv) == 0);
    if (qwen_responses_media_sv[0] >= 0 &&
        qwen_responses_media_sv[1] >= 0) {
        job qwen_responses_media_job = {0};
        qwen_responses_media_job.fd = qwen_responses_media_sv[0];
        qwen_responses_media_job.req = qwen_responses_media_job_req;
        generate_job_rt(&qwen_media_api_server, &qwen_responses_media_job);
        shutdown(qwen_responses_media_sv[0], SHUT_WR);
        char *out = read_socket_text(qwen_responses_media_sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "\"object\":\"response\"") != NULL);
        TEST_ASSERT(rt_session_pos(qwen_media_api_session) ==
                    qwen_responses_media_job_expanded_len);
        TEST_ASSERT(qwen_media_api_server.rt_live_media_state);
        free(out);
        request_free(&qwen_responses_media_job.req);
        close(qwen_responses_media_sv[0]);
        close(qwen_responses_media_sv[1]);
    } else {
        request_free(&qwen_responses_media_job_req);
    }
    rt_session_invalidate(qwen_media_api_session);
    qwen_media_api_server.rt_live_media_state = false;

    request qwen_anthropic_media_job_req = {0};
    qwen_err[0] = '\0';
    TEST_ASSERT(parse_anthropic_request_rt(qwen_engine, qwen,
        &qwen_media_api_server,
        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
        "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
        "{\"type\":\"image\",\"source\":{\"type\":\"base64\","
        "\"media_type\":\"image/x-portable-pixmap\","
        "\"data\":\"UDMKMSAxCjI1NQo4IDkgMTAK\"}}]}],"
        "\"max_tokens\":0,\"thinking\":{\"type\":\"disabled\"}}",
        4, 128, &qwen_anthropic_media_job_req,
        qwen_err, sizeof(qwen_err)));
    rt_tokens qwen_anthropic_media_job_prompt = {
        .v = qwen_anthropic_media_job_req.prompt.v,
        .len = qwen_anthropic_media_job_req.prompt.len,
        .cap = qwen_anthropic_media_job_req.prompt.cap,
    };
    qwen_runtime_media_spans qwen_anthropic_media_job_spans = {0};
    qwen_err[0] = '\0';
    TEST_ASSERT(qwen_runtime_build_media_embeddings(
                    qwen_engine, &qwen_anthropic_media_job_req,
                    &qwen_anthropic_media_job_prompt,
                    &qwen_anthropic_media_job_spans,
                    qwen_err, sizeof(qwen_err)));
    int qwen_anthropic_media_job_expanded_len =
        qwen_anthropic_media_job_spans.prompt.len;
    qwen_runtime_media_spans_free(&qwen_anthropic_media_job_spans);
    int qwen_anthropic_media_sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0,
                           qwen_anthropic_media_sv) == 0);
    if (qwen_anthropic_media_sv[0] >= 0 &&
        qwen_anthropic_media_sv[1] >= 0) {
        job qwen_anthropic_media_job = {0};
        qwen_anthropic_media_job.fd = qwen_anthropic_media_sv[0];
        qwen_anthropic_media_job.req = qwen_anthropic_media_job_req;
        generate_job_rt(&qwen_media_api_server, &qwen_anthropic_media_job);
        shutdown(qwen_anthropic_media_sv[0], SHUT_WR);
        char *out = read_socket_text(qwen_anthropic_media_sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"message\"") != NULL);
        TEST_ASSERT(rt_session_pos(qwen_media_api_session) ==
                    qwen_anthropic_media_job_expanded_len);
        TEST_ASSERT(qwen_media_api_server.rt_live_media_state);
        free(out);
        request_free(&qwen_anthropic_media_job.req);
        close(qwen_anthropic_media_sv[0]);
        close(qwen_anthropic_media_sv[1]);
    } else {
        request_free(&qwen_anthropic_media_job_req);
    }
    pthread_mutex_destroy(&qwen_media_api_server.tool_mu);
    rt_session_free(qwen_media_api_session);

	    request qwen_resize_req = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	        "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
	        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/x-portable-pixmap,"
	        "P3%0A1%201%0A255%0A0%200%200%0A\","
	        "\"resized_height\":64,\"resized_width\":96,"
	        "\"min_pixels\":1024,\"max_pixels\":1048576}}]}],"
	        "\"max_tokens\":1}",
	        4, 128, &qwen_resize_req, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_resize_req.media.len == 1);
	    TEST_ASSERT(qwen_resize_req.media.v[0].resized_height == 64);
	    TEST_ASSERT(qwen_resize_req.media.v[0].resized_width == 96);
	    TEST_ASSERT(qwen_resize_req.media.v[0].min_pixels == 1024);
	    TEST_ASSERT(qwen_resize_req.media.v[0].max_pixels == 1048576);
	    char qwen_resize_sha_a[41];
	    char qwen_resize_sha_b[41];
	    runtime_media_ref_digest_hex(&qwen_resize_req.media.v[0],
	                                 qwen_resize_sha_a);
	    qwen_resize_req.media.v[0].resized_width = 64;
	    runtime_media_ref_digest_hex(&qwen_resize_req.media.v[0],
	                                 qwen_resize_sha_b);
	    TEST_ASSERT(strcmp(qwen_resize_sha_a, qwen_resize_sha_b) != 0);
	    qwen_resize_req.media.v[0].resized_width = 96;
	    rt_tokens qwen_resize_rt_prompt = {
	        .v = qwen_resize_req.prompt.v,
	        .len = qwen_resize_req.prompt.len,
	        .cap = qwen_resize_req.prompt.cap,
	    };
	    qwen_runtime_media_spans qwen_resize_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_resize_req, &qwen_resize_rt_prompt,
	                    &qwen_resize_spans, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_resize_spans.len == 1);
	    TEST_ASSERT(qwen_resize_spans.spans[0].n_tokens == 6);
	    TEST_ASSERT(qwen_resize_spans.spans[0].hidden_size ==
	                QWEN36_TEST_HIDDEN);
	    qwen_runtime_media_spans_free(&qwen_resize_spans);
	    request_free(&qwen_resize_req);

	    request qwen_video_opts_req = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	        "\"content\":[{\"type\":\"text\",\"text\":\"watch\"},"
	        "{\"type\":\"video_url\",\"video_url\":{\"url\":\"data:image/x-portable-pixmap,"
	        "P3%0A1%201%0A255%0A0%200%200%0A\","
	        "\"fps\":1.5,\"video_start\":0.25,\"video_end\":0.75,"
	        "\"min_frames\":1,\"max_frames\":2}}]}],"
	        "\"max_tokens\":1}",
	        4, 128, &qwen_video_opts_req, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_video_opts_req.media.len == 1);
	    TEST_ASSERT(qwen_video_opts_req.media.v[0].video);
	    TEST_ASSERT(qwen_video_opts_req.media.v[0].has_fps);
	    TEST_ASSERT(fabs(qwen_video_opts_req.media.v[0].fps - 1.5) < 0.0001);
	    TEST_ASSERT(qwen_video_opts_req.media.v[0].has_video_start);
	    TEST_ASSERT(fabs(qwen_video_opts_req.media.v[0].video_start - 0.25) <
	                0.0001);
	    TEST_ASSERT(qwen_video_opts_req.media.v[0].has_video_end);
	    TEST_ASSERT(fabs(qwen_video_opts_req.media.v[0].video_end - 0.75) <
	                0.0001);
	    TEST_ASSERT(qwen_video_opts_req.media.v[0].min_frames == 1);
	    TEST_ASSERT(qwen_video_opts_req.media.v[0].max_frames == 2);
	    char qwen_video_opts_sha_a[41];
	    char qwen_video_opts_sha_b[41];
	    runtime_media_ref_digest_hex(&qwen_video_opts_req.media.v[0],
	                                 qwen_video_opts_sha_a);
	    qwen_video_opts_req.media.v[0].video_end = 0.5;
	    runtime_media_ref_digest_hex(&qwen_video_opts_req.media.v[0],
	                                 qwen_video_opts_sha_b);
	    TEST_ASSERT(strcmp(qwen_video_opts_sha_a, qwen_video_opts_sha_b) != 0);
	    rt_tokens qwen_video_opts_rt_prompt = {
	        .v = qwen_video_opts_req.prompt.v,
	        .len = qwen_video_opts_req.prompt.len,
	        .cap = qwen_video_opts_req.prompt.cap,
	    };
	    qwen_runtime_media_spans qwen_video_opts_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_video_opts_req,
	                    &qwen_video_opts_rt_prompt, &qwen_video_opts_spans,
	                    qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_video_opts_spans.len == 1);
	    TEST_ASSERT(qwen_video_opts_spans.spans[0].n_tokens == 4);
	    qwen_runtime_media_spans_free(&qwen_video_opts_spans);
	    request_free(&qwen_video_opts_req);

	    request qwen_many_video_req = {0};
	    qwen_many_video_req.media.v =
	        xmalloc(sizeof(qwen_many_video_req.media.v[0]));
	    qwen_many_video_req.media.len = 1;
	    qwen_many_video_req.media.cap = 1;
	    memset(qwen_many_video_req.media.v, 0,
	           sizeof(qwen_many_video_req.media.v[0]));
	    qwen_many_video_req.media.v[0].video = true;
	    qwen_many_video_req.media.v[0].media_type =
	        xstrdup("image/x-portable-pixmap");
	    qwen_many_video_req.media.v[0].nframes = 18;
	    qwen_many_video_req.media.v[0].max_frames = 18;
	    buf qwen_many_video_payload = {0};
	    for (uint32_t i = 0; i < 18; i++) {
	        buf_printf(&qwen_many_video_payload,
	                   "P3\n1 1\n255\n%u %u %u\n",
	                   i, (i + 1u) % 256u, (i + 2u) % 256u);
	    }
	    qwen_many_video_req.media.v[0].bytes =
	        (unsigned char *)buf_take(&qwen_many_video_payload);
	    qwen_many_video_req.media.v[0].bytes_len =
	        strlen((char *)qwen_many_video_req.media.v[0].bytes);
	    rt_tokens qwen_many_video_prompt = {0};
	    rt_tokens_push(&qwen_many_video_prompt,
	                   QWEN36_TEST_VIDEO_PAD_TOKEN);
	    qwen_runtime_media_spans qwen_many_video_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_many_video_req,
	                    &qwen_many_video_prompt,
	                    &qwen_many_video_spans,
	                    qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_many_video_spans.len == 1);
	    TEST_ASSERT(qwen_many_video_spans.spans[0].n_tokens == 72);
	    TEST_ASSERT(qwen_many_video_spans.prompt_expanded);
	    TEST_ASSERT(qwen_many_video_spans.prompt.len == 72);
	    for (int i = 0; i < qwen_many_video_spans.prompt.len; i++) {
	        TEST_ASSERT(qwen_many_video_spans.prompt.v[i] ==
	                    QWEN36_TEST_VIDEO_PAD_TOKEN);
	    }
	    qwen_runtime_media_spans_free(&qwen_many_video_spans);
	    rt_tokens_free(&qwen_many_video_prompt);
	    request_free(&qwen_many_video_req);

	    request qwen_png_req = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	        "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
	        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,"
	        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC\"}}]}],"
	        "\"max_tokens\":1}",
	        4, 128, &qwen_png_req, qwen_err, sizeof(qwen_err)));
	    rt_tokens qwen_png_rt_prompt = {
	        .v = qwen_png_req.prompt.v,
	        .len = qwen_png_req.prompt.len,
	        .cap = qwen_png_req.prompt.cap,
	    };
	    qwen_runtime_media_spans qwen_png_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_png_req, &qwen_png_rt_prompt,
	                    &qwen_png_spans, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_png_spans.len == 1);
	    TEST_ASSERT(qwen_png_spans.spans[0].n_tokens == 4);
	    TEST_ASSERT(qwen_png_spans.spans[0].hidden_size == QWEN36_TEST_HIDDEN);
	    TEST_ASSERT(qwen_png_req.media.len == 1);
	    TEST_ASSERT(!qwen_png_req.media.v[0].video);
	    TEST_ASSERT(qwen_png_req.media.v[0].media_type != NULL);
	    TEST_ASSERT(!strcmp(qwen_png_req.media.v[0].media_type, "image/png"));
	    qwen_runtime_media_spans_free(&qwen_png_spans);
	    request_free(&qwen_png_req);

	    request qwen_grid_req = {0};
	    qwen_grid_req.media.v = xmalloc(sizeof(qwen_grid_req.media.v[0]));
	    qwen_grid_req.media.len = 1;
	    qwen_grid_req.media.cap = 1;
	    memset(qwen_grid_req.media.v, 0, sizeof(qwen_grid_req.media.v[0]));
	    qwen_grid_req.media.v[0].media_type = xstrdup("image/x-portable-pixmap");
	    char qwen_grid_hdr[64];
	    int qwen_grid_hdr_len = snprintf(qwen_grid_hdr, sizeof(qwen_grid_hdr),
	                                     "P6\n64 64\n255\n");
	    TEST_ASSERT(qwen_grid_hdr_len > 0 &&
	                (size_t)qwen_grid_hdr_len < sizeof(qwen_grid_hdr));
	    size_t qwen_grid_pixels = 64u * 64u * 3u;
	    qwen_grid_req.media.v[0].bytes_len =
	        (size_t)qwen_grid_hdr_len + qwen_grid_pixels;
	    qwen_grid_req.media.v[0].bytes =
	        xmalloc(qwen_grid_req.media.v[0].bytes_len);
	    memcpy(qwen_grid_req.media.v[0].bytes, qwen_grid_hdr,
	           (size_t)qwen_grid_hdr_len);
	    memset(qwen_grid_req.media.v[0].bytes + qwen_grid_hdr_len, 0,
	           qwen_grid_pixels);
	    rt_tokens qwen_grid_prompt = {0};
	    rt_tokens_push(&qwen_grid_prompt, QWEN36_TEST_IMAGE_PAD_TOKEN);
	    qwen_runtime_media_spans qwen_grid_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_grid_req, &qwen_grid_prompt,
	                    &qwen_grid_spans, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_grid_spans.len == 1);
	    TEST_ASSERT(qwen_grid_spans.spans[0].token_pos == 0);
	    TEST_ASSERT(qwen_grid_spans.spans[0].n_tokens == 4);
	    TEST_ASSERT(qwen_grid_spans.prompt_expanded);
	    TEST_ASSERT(qwen_grid_spans.prompt.len == 4);
	    for (int i = 0; i < qwen_grid_spans.prompt.len; i++) {
	        TEST_ASSERT(qwen_grid_spans.prompt.v[i] ==
	                    QWEN36_TEST_IMAGE_PAD_TOKEN);
	    }
	    qwen_runtime_media_spans_free(&qwen_grid_spans);
	    rt_tokens_free(&qwen_grid_prompt);
	    request_free(&qwen_grid_req);

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
	    request qwen_jpeg_req = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	        "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
	        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,"
	        "/9j/4AAQSkZJRgABAQAASABIAAD/4QBMRXhpZgAATU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAA"
	        "A6ABAAMAAAABAAEAAKACAAQAAAABAAAAAaADAAQAAAABAAAAAQAAAAD/7QA4UGhvdG9zaG9wIDMu"
	        "MAA4QklNBAQAAAAAAAA4QklNBCUAAAAAABDUHYzZjwCyBOmACZjs+EJ+/8AAEQgAAQABAwEiAAIR"
	        "AQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAAB"
	        "fQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5"
	        "OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeo"
	        "qaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMB"
	        "AQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYS"
	        "QVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNU"
	        "VVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5"
	        "usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/bAEMAAgICAgICAwICAwUDAwMF"
	        "BgUFBQUGCAYGBgYGCAoICAgICAgKCgoKCgoKCgwMDAwMDA4ODg4ODw8PDw8PDw8PD//bAEMBAgIC"
	        "BAQEBwQEBxALCQsQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ"
	        "EBAQEP/dAAQAAf/aAAwDAQACEQMRAD8A+L6KKK/lM/38P//Z\"}}]}],"
	        "\"max_tokens\":1}",
	        4, 128, &qwen_jpeg_req, qwen_err, sizeof(qwen_err)));
	    rt_tokens qwen_jpeg_rt_prompt = {
	        .v = qwen_jpeg_req.prompt.v,
	        .len = qwen_jpeg_req.prompt.len,
	        .cap = qwen_jpeg_req.prompt.cap,
	    };
	    qwen_runtime_media_spans qwen_jpeg_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_jpeg_req, &qwen_jpeg_rt_prompt,
	                    &qwen_jpeg_spans, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_jpeg_spans.len == 1);
	    TEST_ASSERT(qwen_jpeg_spans.spans[0].n_tokens == 4);
	    TEST_ASSERT(qwen_jpeg_spans.spans[0].hidden_size == QWEN36_TEST_HIDDEN);
	    TEST_ASSERT(qwen_jpeg_req.media.len == 1);
	    TEST_ASSERT(!qwen_jpeg_req.media.v[0].video);
	    TEST_ASSERT(qwen_jpeg_req.media.v[0].media_type != NULL);
	    TEST_ASSERT(!strcmp(qwen_jpeg_req.media.v[0].media_type, "image/jpeg"));
	    qwen_runtime_media_spans_free(&qwen_jpeg_spans);
	    request_free(&qwen_jpeg_req);

	    request qwen_gif_req = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	        "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
	        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/gif;base64,"
	        "R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==\"}}]}],"
	        "\"max_tokens\":1}",
	        4, 128, &qwen_gif_req, qwen_err, sizeof(qwen_err)));
	    rt_tokens qwen_gif_rt_prompt = {
	        .v = qwen_gif_req.prompt.v,
	        .len = qwen_gif_req.prompt.len,
	        .cap = qwen_gif_req.prompt.cap,
	    };
	    qwen_runtime_media_spans qwen_gif_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_gif_req, &qwen_gif_rt_prompt,
	                    &qwen_gif_spans, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_gif_spans.len == 1);
	    TEST_ASSERT(qwen_gif_spans.spans[0].n_tokens == 4);
	    TEST_ASSERT(qwen_gif_spans.spans[0].hidden_size == QWEN36_TEST_HIDDEN);
	    TEST_ASSERT(qwen_gif_req.media.len == 1);
	    TEST_ASSERT(!qwen_gif_req.media.v[0].video);
	    TEST_ASSERT(qwen_gif_req.media.v[0].media_type != NULL);
	    TEST_ASSERT(!strcmp(qwen_gif_req.media.v[0].media_type, "image/gif"));
	    qwen_runtime_media_spans_free(&qwen_gif_spans);
	    request_free(&qwen_gif_req);

	    static const unsigned char qwen_remote_ppm[] =
	        "P3\n1 1\n255\n5 6 7\n";
	    test_http_media_server qwen_remote_server = {0};
	    char qwen_remote_url[160];
	    TEST_ASSERT(test_http_media_server_start(
	                    &qwen_remote_server, qwen_remote_ppm,
	                    sizeof(qwen_remote_ppm) - 1u, 3,
	                    qwen_remote_url, sizeof(qwen_remote_url)));
	    request qwen_remote_req = {0};
	    bool qwen_remote_started = qwen_remote_server.thread_started;
	    if (qwen_remote_started) {
	        char qwen_remote_json[512];
	        int qwen_remote_json_len = snprintf(qwen_remote_json,
	            sizeof(qwen_remote_json),
	            "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	            "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
	            "{\"type\":\"image_url\",\"image_url\":{\"url\":\"%s\"}}]}],"
	            "\"max_tokens\":1}",
	            qwen_remote_url);
	        TEST_ASSERT(qwen_remote_json_len > 0 &&
	                    (size_t)qwen_remote_json_len <
	                        sizeof(qwen_remote_json));
	        qwen_err[0] = '\0';
	        TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	            qwen_remote_json, 4, 128, &qwen_remote_req,
	            qwen_err, sizeof(qwen_err)));
	        TEST_ASSERT(qwen_remote_req.media.len == 1);
	        TEST_ASSERT(qwen_remote_req.media.v[0].url != NULL);
	        TEST_ASSERT(!strcmp(qwen_remote_req.media.v[0].url,
	                            qwen_remote_url));
	        TEST_ASSERT(qwen_remote_req.media.v[0].bytes_len == 0);
	        unsigned char *qwen_remote_bytes = NULL;
	        size_t qwen_remote_bytes_len = 0;
	        qwen_err[0] = '\0';
	        TEST_ASSERT(runtime_media_ref_url_bytes(
	                        &qwen_remote_req.media.v[0], &qwen_remote_bytes,
	                        &qwen_remote_bytes_len, qwen_err,
	                        sizeof(qwen_err)));
	        TEST_ASSERT(qwen_remote_bytes_len ==
	                    sizeof(qwen_remote_ppm) - 1u);
	        TEST_ASSERT(!memcmp(qwen_remote_bytes, qwen_remote_ppm,
	                            sizeof(qwen_remote_ppm) - 1u));
	        free(qwen_remote_bytes);
	        rt_tokens qwen_remote_rt_prompt = {
	            .v = qwen_remote_req.prompt.v,
	            .len = qwen_remote_req.prompt.len,
	            .cap = qwen_remote_req.prompt.cap,
	        };
	        qwen_runtime_media_spans qwen_remote_spans = {0};
	        qwen_err[0] = '\0';
	        TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                        qwen_engine, &qwen_remote_req,
	                        &qwen_remote_rt_prompt, &qwen_remote_spans,
	                        qwen_err, sizeof(qwen_err)));
	        TEST_ASSERT(qwen_remote_spans.len == 1);
	        TEST_ASSERT(qwen_remote_spans.spans[0].n_tokens == 4);
	        TEST_ASSERT(qwen_remote_spans.spans[0].hidden_size ==
	                    QWEN36_TEST_HIDDEN);
	        char qwen_remote_sha[41];
	        runtime_media_ref_digest_hex(&qwen_remote_req.media.v[0],
	                                     qwen_remote_sha);
	        TEST_ASSERT(strlen(qwen_remote_sha) == 40);
	        qwen_runtime_media_spans_free(&qwen_remote_spans);
	        request_free(&qwen_remote_req);
	    }
	    test_http_media_server_join(&qwen_remote_server);
	    if (qwen_remote_started) TEST_ASSERT(qwen_remote_server.served == 3);
#endif

	    request qwen_video_req = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	        "\"content\":[{\"type\":\"text\",\"text\":\"watch\"},"
	        "{\"type\":\"video_url\",\"video_url\":{\"url\":\"data:image/x-portable-pixmap,"
	        "P3%0A1%201%0A255%0A0%200%200%0A\"}}]}],"
	        "\"max_tokens\":1}",
	        4, 128, &qwen_video_req, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_video_req.media.len == 1);
	    TEST_ASSERT(qwen_video_req.media.v[0].video);
	    rt_tokens qwen_video_rt_prompt = {
	        .v = qwen_video_req.prompt.v,
	        .len = qwen_video_req.prompt.len,
	        .cap = qwen_video_req.prompt.cap,
	    };
	    qwen_runtime_media_spans qwen_video_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_video_req, &qwen_video_rt_prompt,
	                    &qwen_video_spans, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_video_spans.len == 1);
	    TEST_ASSERT(qwen_video_spans.spans[0].n_tokens == 4);
	    TEST_ASSERT(qwen_video_spans.spans[0].hidden_size == QWEN36_TEST_HIDDEN);
	    TEST_ASSERT(qwen_video_spans.spans[0].token_pos >= 0);
	    TEST_ASSERT(qwen_video_req.prompt.v[qwen_video_spans.spans[0].token_pos] ==
	                QWEN36_TEST_VIDEO_PAD_TOKEN);
	    int qwen_video_expanded_len = qwen_video_spans.prompt.len;
	    qwen_runtime_media_spans_free(&qwen_video_spans);
	    rt_session *qwen_video_session = NULL;
	    TEST_ASSERT(rt_session_create(&qwen_video_session, qwen_engine, 128) == 0);
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_session_sync_request(
	                    qwen_engine, qwen_video_session, &qwen_video_req,
	                    &qwen_video_rt_prompt, qwen_err, sizeof(qwen_err)) == 0);
	    TEST_ASSERT(rt_session_pos(qwen_video_session) == qwen_video_expanded_len);
	    rt_session_free(qwen_video_session);
	    request_free(&qwen_video_req);

#ifdef __APPLE__
	    request qwen_mp4_req = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	        "\"content\":[{\"type\":\"text\",\"text\":\"watch\"},"
	        "{\"type\":\"video_url\",\"video_url\":{\"url\":\"data:video/mp4;base64,"
	        "AAAAHGZ0eXBpc29tAAACAGlzb21pc28ybXA0MQAAA0Btb292AAAAbG12aGQAAAAAAAAAAAAAAAAA"
	        "AAPoAAAAKAABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAA"
	        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACa3RyYWsAAABcdGtoZAAAAAMAAAAAAAAA"
	        "AAAAAAEAAAAAAAAAKAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAA"
	        "AAAAAEAAAAAAEAAAABAAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAACgAAAAAAAEAAAAAAeNt"
	        "ZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAADIAAAACAFXEAAAAAAAtaGRscgAAAAAAAAAAdmlkZQAA"
	        "AAAAAAAAAAAAAFZpZGVvSGFuZGxlcgAAAAGObWluZgAAABR2bWhkAAAAAQAAAAAAAAAAAAAAJGRp"
	        "bmYAAAAcZHJlZgAAAAAAAAABAAAADHVybCAAAAABAAABTnN0YmwAAADqc3RzZAAAAAAAAAABAAAA"
	        "2m1wNHYAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAEAAQAEgAAABIAAAAAAAAAAETTGF2YzYyLjEx"
	        "LjEwMCBtcGVnNAAAAAAAAAAAAAAAAAAY//8AAABgZXNkcwAAAAADgICATwABAASAgIBBIBEAAAAA"
	        "Aw1AAAAPoAWAgIAvAAABsAEAAAG1iRMAAAEAAAABIADEjYgAzQCEAhRjAAABskxhdmM2Mi4xMS4x"
	        "MDAGgICAAQIAAAAQcGFzcAAAAAEAAAABAAAAFGJ0cnQAAAAAAAMNQAAAD6AAAAAYc3R0cwAAAAAA"
	        "AAABAAAAAQAAAgAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAEAAAABAAAAFHN0c3oAAAAAAAAAFAAA"
	        "AAEAAAAUc3RjbwAAAAAAAAABAAADbAAAAGF1ZHRhAAAAWW1ldGEAAAAAAAAAIWhkbHIAAAAAAAAA"
	        "AG1kaXJhcHBsAAAAAAAAAAAAAAAALGlsc3QAAAAkqXRvbwAAABxkYXRhAAAAAQAAAABMYXZmNjIu"
	        "My4xMDAAAAAIZnJlZQAAABxtZGF0AAABswAQBwAAAbYQYFGFBtgsgeA=\"}}]}],"
	        "\"max_tokens\":1}",
	        4, 128, &qwen_mp4_req, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_mp4_req.media.len == 1);
	    TEST_ASSERT(qwen_mp4_req.media.v[0].video);
	    TEST_ASSERT(qwen_mp4_req.media.v[0].media_type != NULL);
	    TEST_ASSERT(!strcmp(qwen_mp4_req.media.v[0].media_type, "video/mp4"));
	    rt_tokens qwen_mp4_rt_prompt = {
	        .v = qwen_mp4_req.prompt.v,
	        .len = qwen_mp4_req.prompt.len,
	        .cap = qwen_mp4_req.prompt.cap,
	    };
	    qwen_runtime_media_spans qwen_mp4_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_mp4_req, &qwen_mp4_rt_prompt,
	                    &qwen_mp4_spans, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_mp4_spans.len == 1);
	    TEST_ASSERT(qwen_mp4_spans.spans[0].n_tokens == 4);
	    TEST_ASSERT(qwen_mp4_spans.spans[0].hidden_size == QWEN36_TEST_HIDDEN);
	    TEST_ASSERT(qwen_mp4_spans.spans[0].token_pos >= 0);
	    TEST_ASSERT(qwen_mp4_req.prompt.v[qwen_mp4_spans.spans[0].token_pos] ==
	                QWEN36_TEST_VIDEO_PAD_TOKEN);
	    qwen_runtime_media_spans_free(&qwen_mp4_spans);
	    request_free(&qwen_mp4_req);

	    request qwen_mp4_multi_req = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	        "\"content\":[{\"type\":\"text\",\"text\":\"watch\"},"
	        "{\"type\":\"video_url\",\"video_url\":{\"url\":\"data:video/mp4;base64,"
	        "AAAAHGZ0eXBpc29tAAACAGlzb21pc28ybXA0MQAAA1xtb292AAAAbG12aGQAAAAAAAAAAAAAAAAA"
	        "AAPoAAAH0AABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAA"
	        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACh3RyYWsAAABcdGtoZAAAAAMAAAAAAAAA"
	        "AAAAAAEAAAAAAAAH0AAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAA"
	        "AAAAAEAAAAAAIAAAACAAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAB9AAAAAAAAEAAAAAAf9t"
	        "ZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAAEAAAACAAFXEAAAAAAAtaGRscgAAAAAAAAAAdmlkZQAA"
	        "AAAAAAAAAAAAAFZpZGVvSGFuZGxlcgAAAAGqbWluZgAAABR2bWhkAAAAAQAAAAAAAAAAAAAAJGRp"
	        "bmYAAAAcZHJlZgAAAAAAAAABAAAADHVybCAAAAABAAABanN0YmwAAADqc3RzZAAAAAAAAAABAAAA"
	        "2m1wNHYAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAIAAgAEgAAABIAAAAAAAAAAETTGF2YzYyLjEx"
	        "LjEwMCBtcGVnNAAAAAAAAAAAAAAAAAAY//8AAABgZXNkcwAAAAADgICATwABAASAgIBBIBEAAAAA"
	        "Aw1AAAAYKAWAgIAvAAABsAEAAAG1iRMAAAEAAAABIADEjYgADQEEBBRDAAABskxhdmM2Mi4xMS4x"
	        "MDAGgICAAQIAAAAQcGFzcAAAAAEAAAABAAAAFGJ0cnQAAAAAAAMNQAAAGCgAAAAYc3R0cwAAAAAA"
	        "AAABAAAAAgAAQAAAAAAUc3RzcwAAAAAAAAABAAAAAQAAABxzdHNjAAAAAAAAAAEAAAABAAAAAgAA"
	        "AAEAAAAcc3RzegAAAAAAAAAAAAAAAgAABJcAAAFzAAAAFHN0Y28AAAAAAAAAAQAAA4gAAABhdWR0"
	        "YQAAAFltZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAACxpbHN0AAAA"
	        "JKl0b28AAAAcZGF0YQAAAAEAAAAATGF2ZjYyLjMuMTAwAAAACGZyZWUAAAYSbWRhdAAAAbMAEAcA"
	        "AAG2FgTYRgeA/PwYIIkAzXghF3m1SUFDqofq1QhDmX7CpXrDSoSR21mjlgdfbSJfKqVUdspFapNo"
	        "MsXsgshkDAGAwkAyQSfCT4IYKQA4DoGgVQgaB0Rh8If2q3E8ZrVHrDIgCCIDVbxgs9bkKjgNoGA+"
	        "DMhDBk4PAfsoPAfu6cIbakHAeBS6DMCQqz6eph8EIGwehBCEXNJmk6tWHwfh98SR75VfJmwhjsSy9"
	        "pUwCLfOBUA8BA6g8J/sg8RANg+Z/1g1g8B+1gycFKXDoGEAdAfEsSB8qUUcCQXqx0OfiCkZHLP03"
	        "/f/uAp28zuqJ68Ud7X0DoIBcCkEsHgIIcRxCbH7CsHgP4sfgwHR+DDvwIvwUwQ6DDwQi4IQNipsu"
	        "WYqYIXtL0qbP5/YPRLL1d2gy2AFNgygYDgMPQZKIeiToQwbQQQg1ICmA0DAbCGPxHxgQQU8bHHh2"
	        "yrzC0s8p5Fyz1R+w8sDwH6CDwEEGEMAwdFwNQgKi8SQZgSGwNCGPQ8+yBxOI4feHwMORxPA4uHrK"
	        "b7SodxjE3kusCUnrSVOOgfB/6T4PAQOoPCf7IPEQDYPmf9YIsAwGYCEDCODeBhHBkgBrQ5EIEMSkw"
	        "lJeeTfZVDnB2IA9LgU6pKqaEsSlcaHYkJ/qmwUw7TJkuYpacBTg8B+1g2AoEiUG6PgOCMOkqZRGm"
	        "xKHyYdVRcLvFn4k81jfgclV7M/Ko/+/nLafgKEEESQDB0DwEE2IwhfYSFwMCCkBgUJcDD+RImEEA"
	        "1oGYElMIYKSl6W+A3rYHkv08Hnxz1hMPR6mgGE+ucAAMI2GgGVhCHYHAeA/ZQeA/Xy4SVQGlbYMJ"
	        "I/CH7w5Yb6OgQx6IYBoQlalV9vICKIQllyUcWiCwORAT+aV+baMAoQeAgdQeE/2QeIgGwfM/6wmA"
	        "8B+0g8BBXgxerHqYdsF4IH439OraVQvH+K/gamWtQtEjFU/8c5ni1XjXQ7b9h0O4N4HgIIMA0G+"
	        "JX8EYSQQKCmEAc62lHszqubckLBIb2bJ4Pc/ntzuLEw6DwH7yDwED6DD5WDYDgUY9B4D+HHwkCDk"
	        "4q+AYnb8IM1N6KqX4XxrP/1NzJsy3t5KFhQNgkhAB4D+TbBgUQ+BQiSwDBAEcHgIHfQZW0nEpPnw"
	        "htj9WDCSXYO2GFetRJ4QtVD5M2nVDmtgxb5pkQFGqm4YbB0A8BA6g8J/sg8RANg+Z/1ghAYdBDH"
	        "QIQQgYvANCGrBEEIDgleCGJKUcMsgaHQlJhHLy9WICbzatscKmv/HCpptn/FDXt2xwSgeAgmQeA/"
	        "fwQGB0XjoGUAGsq2GGvWMJC76mtxuwsB4L/pm2SKtulU5OEp4GYDMg8B+ogw6BoI8+2wAaJbCVSI"
	        "CsSh8mLKOIxvyodj0dMzlVzy7S+5vCpRpkIwYDoPAfxYMJYHwUcAMBSg8BA/jsGH2Jf5Gi8vAPVJ"
	        "d1qYm2q4XYXNNjjiQb9mL2Ke4eGgeA/fxCA6qaB4CBzHoMmCElBhJCEDAovAwlq6I4kQQEwkpAZO"
	        "PWhGxW3aOQYQIqHfqkVRsc6JLeJw+nC5LQufAAABtmuBHQAAwgVMGEgHgIJkGg/B4CBnBggKwaAw"
	        "k0uCGX/CECFoMqvx+q3jIQ/AHfBVEAPAf34PAQJoMAaEIHgIDMHgP8svBhLCD4S1YkW+HQkeVUf"
	        "81hWXl3gfK/+weAgOQeA/vwYfgwlgwQgagygGCADwEBuJYPAf4YQB6oBgUIkCXBIVKAU6iqi8Yg"
	        "8B/ig8B/fgwkAwQgYIANAZSDAGA8B/jiQDwH+CEFRQeCgGwhCR8SP6x4e1WXtkQVMGCGDwH7ODUf"
	        "A8B/MgwQVQNQYIXi8SC7wkgfwG/PFyvOsBCVgHzQYhB4CBDB4D+3BgDAhg8B/ng8BAblwMJAQviU"
	        "qEmRWOxLquD5pdUqL1YMCqcDwH+WDwECCDCUDCUDBBBoDKQYA8HgP80SgeAgPwhjxSDwMA2EMfA2"
	        "q9BTKfKy5OYB4CA7B4CBFBhJBghgwQQQQZQDAHg8BAdiSDwECGEDYDwX/OEMSfBDVYBlRFRc0DEb"
	        "8=\"}}]}],"
	        "\"max_tokens\":1}",
	        4, 128, &qwen_mp4_multi_req, qwen_err, sizeof(qwen_err)));
	    rt_tokens qwen_mp4_multi_rt_prompt = {
	        .v = qwen_mp4_multi_req.prompt.v,
	        .len = qwen_mp4_multi_req.prompt.len,
	        .cap = qwen_mp4_multi_req.prompt.cap,
	    };
	    qwen_runtime_media_spans qwen_mp4_multi_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_mp4_multi_req,
	                    &qwen_mp4_multi_rt_prompt,
	                    &qwen_mp4_multi_spans, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_mp4_multi_spans.len == 1);
	    TEST_ASSERT(qwen_mp4_multi_spans.spans[0].n_tokens == 8);
	    TEST_ASSERT(qwen_mp4_multi_spans.prompt_expanded);
	    TEST_ASSERT(qwen_mp4_multi_spans.prompt.len ==
	                qwen_mp4_multi_req.prompt.len + 7);
	    TEST_ASSERT(qwen_mp4_multi_spans.spans[0].token_pos >= 0);
	    TEST_ASSERT(qwen_mp4_multi_spans.prompt.v[
	                    qwen_mp4_multi_spans.spans[0].token_pos] ==
	                QWEN36_TEST_VIDEO_PAD_TOKEN);
	    TEST_ASSERT(qwen_mp4_multi_spans.prompt.v[
	                    qwen_mp4_multi_spans.spans[0].token_pos + 1] ==
	                QWEN36_TEST_VIDEO_PAD_TOKEN);
	    qwen_runtime_media_spans_free(&qwen_mp4_multi_spans);

	    request qwen_mp4_one_req = {0};
	    qwen_mp4_one_req.media.v = xmalloc(sizeof(qwen_mp4_one_req.media.v[0]));
	    qwen_mp4_one_req.media.len = 1;
	    qwen_mp4_one_req.media.cap = 1;
	    memset(qwen_mp4_one_req.media.v, 0,
	           sizeof(qwen_mp4_one_req.media.v[0]));
	    qwen_mp4_one_req.media.v[0].video = true;
	    qwen_mp4_one_req.media.v[0].media_type =
	        xstrdup(qwen_mp4_multi_req.media.v[0].media_type);
	    qwen_mp4_one_req.media.v[0].bytes_len =
	        qwen_mp4_multi_req.media.v[0].bytes_len;
	    qwen_mp4_one_req.media.v[0].bytes =
	        xmalloc(qwen_mp4_one_req.media.v[0].bytes_len);
	    memcpy(qwen_mp4_one_req.media.v[0].bytes,
	           qwen_mp4_multi_req.media.v[0].bytes,
	           qwen_mp4_one_req.media.v[0].bytes_len);
	    qwen_mp4_one_req.media.v[0].nframes = 1;
	    rt_tokens qwen_mp4_one_prompt = {0};
	    rt_tokens_push(&qwen_mp4_one_prompt, QWEN36_TEST_VIDEO_PAD_TOKEN);
	    qwen_runtime_media_spans qwen_mp4_one_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_mp4_one_req,
	                    &qwen_mp4_one_prompt,
	                    &qwen_mp4_one_spans, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_mp4_one_spans.len == 1);
	    TEST_ASSERT(qwen_mp4_one_spans.spans[0].n_tokens == 8);
	    qwen_runtime_media_spans_free(&qwen_mp4_one_spans);
	    rt_tokens_free(&qwen_mp4_one_prompt);

	    qwen_mp4_one_req.media.v[0].nframes = 0;
	    qwen_mp4_one_req.media.v[0].has_fps = true;
	    qwen_mp4_one_req.media.v[0].fps = 2.0;
	    qwen_mp4_one_req.media.v[0].has_video_start = true;
	    qwen_mp4_one_req.media.v[0].video_start = 0.0;
	    qwen_mp4_one_req.media.v[0].has_video_end = true;
	    qwen_mp4_one_req.media.v[0].video_end = 0.0;
	    qwen_mp4_one_req.media.v[0].min_frames = 1;
	    qwen_mp4_one_req.media.v[0].max_frames = 4;
	    rt_tokens qwen_mp4_range_prompt = {0};
	    rt_tokens_push(&qwen_mp4_range_prompt, QWEN36_TEST_VIDEO_PAD_TOKEN);
	    qwen_runtime_media_spans qwen_mp4_range_spans = {0};
	    qwen_err[0] = '\0';
	    TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                    qwen_engine, &qwen_mp4_one_req,
	                    &qwen_mp4_range_prompt,
	                    &qwen_mp4_range_spans, qwen_err, sizeof(qwen_err)));
	    TEST_ASSERT(qwen_mp4_range_spans.len == 1);
	    TEST_ASSERT(qwen_mp4_range_spans.spans[0].n_tokens == 4);
	    qwen_runtime_media_spans_free(&qwen_mp4_range_spans);
	    rt_tokens_free(&qwen_mp4_range_prompt);
	    request_free(&qwen_mp4_one_req);
	    request_free(&qwen_mp4_multi_req);
#endif

	    char qwen_media_kv_tmpl[] = "/tmp/ds4-qwen-media-kv-test.XXXXXX";
	    char *qwen_media_kv_dir = mkdtemp(qwen_media_kv_tmpl);
	    TEST_ASSERT(qwen_media_kv_dir != NULL);
	    if (qwen_media_kv_dir) {
	        rt_session *qwen_media_cache_session = NULL;
	        TEST_ASSERT(rt_session_create(&qwen_media_cache_session,
	                                      qwen_engine, 128) == 0);
	        server qwen_media_cache_server = {0};
	        qwen_media_cache_server.rt_engine = qwen_engine;
	        qwen_media_cache_server.rt_session = qwen_media_cache_session;
	        qwen_media_cache_server.rt_ops = qwen;
	        TEST_ASSERT(pthread_mutex_init(&qwen_media_cache_server.tool_mu,
	                                       NULL) == 0);
	        kv_cache_options qwen_media_kv_opt = kv_cache_default_options();
	        qwen_media_kv_opt.min_tokens = 1;
	        qwen_media_kv_opt.cold_max_tokens = 128;
	        TEST_ASSERT(kv_cache_open(&qwen_media_cache_server.kv,
	                                  qwen_media_kv_dir, 16, false,
	                                  qwen_media_kv_opt));
	        qwen_runtime_media_spans qwen_media_cache_spans = {0};
	        qwen_err[0] = '\0';
	        TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                        qwen_engine, &qwen_media_req,
	                        &qwen_media_rt_prompt,
	                        &qwen_media_cache_spans,
	                        qwen_err, sizeof(qwen_err)));
	        TEST_ASSERT(qwen_media_cache_spans.prompt_expanded);
	        ds4_tokens qwen_media_cache_prompt = {0};
	        ds4_tokens_copy_rt(&qwen_media_cache_prompt,
	                           &qwen_media_cache_spans.prompt);
	        qwen_err[0] = '\0';
	        TEST_ASSERT(qwen36_runtime_session_sync_embeddings(
	                        qwen_media_cache_session,
	                        &qwen_media_cache_spans.prompt,
	                        qwen_media_cache_spans.spans,
	                        qwen_media_cache_spans.len,
	                        qwen_err, sizeof(qwen_err)) == 0);
	        size_t qwen_media_prompt_text_len = 0;
	        char *qwen_media_prompt_text =
	            render_rt_tokens_text(qwen_engine, &qwen_media_req.prompt,
	                                  &qwen_media_prompt_text_len);
	        TEST_ASSERT(kv_cache_store_runtime_media_exact(
	                        &qwen_media_cache_server, &qwen_media_req,
	                        &qwen_media_cache_prompt,
	                        qwen_media_prompt_text ? qwen_media_prompt_text : "",
	                        qwen_media_prompt_text_len, "cold"));
	        rt_session_invalidate(qwen_media_cache_session);
	        uint8_t qwen_media_ext = 0;
	        int qwen_media_cached = kv_cache_try_load_runtime_media_exact(
	            &qwen_media_cache_server, &qwen_media_req,
	            &qwen_media_cache_prompt,
	            qwen_media_prompt_text ? qwen_media_prompt_text : "",
	            qwen_media_prompt_text_len, &qwen_media_ext);
	        TEST_ASSERT(qwen_media_cached == qwen_media_cache_prompt.len);
	        TEST_ASSERT((qwen_media_ext & KV_EXT_MEDIA_EXACT) != 0);
	        TEST_ASSERT(rt_session_pos(qwen_media_cache_session) ==
	                    qwen_media_cache_prompt.len);

	        rt_session *qwen_media_reload_session = NULL;
	        TEST_ASSERT(rt_session_create(&qwen_media_reload_session,
	                                      qwen_engine, 128) == 0);
	        server qwen_media_reload_server = {0};
	        qwen_media_reload_server.rt_engine = qwen_engine;
	        qwen_media_reload_server.rt_session = qwen_media_reload_session;
	        qwen_media_reload_server.rt_ops = qwen;
	        TEST_ASSERT(pthread_mutex_init(&qwen_media_reload_server.tool_mu,
	                                       NULL) == 0);
	        TEST_ASSERT(kv_cache_open(&qwen_media_reload_server.kv,
	                                  qwen_media_kv_dir, 16, false,
	                                  qwen_media_kv_opt));
	        request qwen_media_reload_req = {0};
	        qwen_err[0] = '\0';
	        TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	            "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	            "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
	            "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/x-portable-pixmap,"
	            "P3%0A1%201%0A255%0A0%200%200%0A\"}}]}],"
	            "\"max_tokens\":0}",
	            4, 128, &qwen_media_reload_req,
	            qwen_err, sizeof(qwen_err)));
	        int qwen_media_reload_sv[2] = {-1, -1};
	        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0,
	                               qwen_media_reload_sv) == 0);
	        if (qwen_media_reload_sv[0] >= 0 &&
	            qwen_media_reload_sv[1] >= 0) {
	            job qwen_media_reload_job = {0};
	            qwen_media_reload_job.fd = qwen_media_reload_sv[0];
	            qwen_media_reload_job.req = qwen_media_reload_req;
	            generate_job_rt(&qwen_media_reload_server,
	                            &qwen_media_reload_job);
	            shutdown(qwen_media_reload_sv[0], SHUT_WR);
	            char *out = read_socket_text(qwen_media_reload_sv[1]);
	            char reload_cached_fragment[64];
	            snprintf(reload_cached_fragment,
	                     sizeof(reload_cached_fragment),
	                     "\"cached_tokens\":%d",
	                     qwen_media_cache_prompt.len);
	            TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
	            TEST_ASSERT(strstr(out, reload_cached_fragment) != NULL);
	            TEST_ASSERT(strstr(out, "\"cache_write_tokens\":0") != NULL);
	            TEST_ASSERT(rt_session_pos(qwen_media_reload_session) ==
	                        qwen_media_cache_prompt.len);
	            TEST_ASSERT(qwen_media_reload_server.rt_live_media_state);
	            free(out);
	            request_free(&qwen_media_reload_job.req);
	            close(qwen_media_reload_sv[0]);
	            close(qwen_media_reload_sv[1]);
	        } else {
	            request_free(&qwen_media_reload_req);
	        }
	        kv_cache_close(&qwen_media_reload_server.kv);
	        pthread_mutex_destroy(&qwen_media_reload_server.tool_mu);
	        rt_session_free(qwen_media_reload_session);

	        rt_session_invalidate(qwen_media_cache_session);
	        request qwen_media_gen_req = {0};
	        qwen_err[0] = '\0';
	        TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	            "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	            "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
	            "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/x-portable-pixmap,"
	            "P3%0A1%201%0A255%0A0%200%200%0A\"}}]}],"
	            "\"max_tokens\":1,\"temperature\":1,\"seed\":1}",
	            4, 128, &qwen_media_gen_req, qwen_err, sizeof(qwen_err)));
	        int qwen_media_sv[2] = {-1, -1};
	        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, qwen_media_sv) == 0);
	        if (qwen_media_sv[0] >= 0 && qwen_media_sv[1] >= 0) {
	            job qwen_media_job = {0};
	            qwen_media_job.fd = qwen_media_sv[0];
	            qwen_media_job.req = qwen_media_gen_req;
	            generate_job_rt(&qwen_media_cache_server, &qwen_media_job);
	            shutdown(qwen_media_sv[0], SHUT_WR);
	            char *out = read_socket_text(qwen_media_sv[1]);
	            char cached_fragment[64];
	            snprintf(cached_fragment, sizeof(cached_fragment),
	                     "\"cached_tokens\":%d", qwen_media_cache_prompt.len);
	            TEST_ASSERT(strstr(out, cached_fragment) != NULL);
	            TEST_ASSERT(strstr(out, "\"cache_write_tokens\":0") != NULL);
	            free(out);
	            const rt_tokens *qwen_media_generated_rt =
	                rt_session_tokens(qwen_media_cache_session);
	            TEST_ASSERT(qwen_media_generated_rt != NULL);
	            TEST_ASSERT(qwen_media_generated_rt->len >
	                        qwen_media_cache_prompt.len);
	            ds4_tokens qwen_media_generated_ds = {0};
	            ds4_tokens_copy_rt(&qwen_media_generated_ds,
	                               qwen_media_generated_rt);
	            size_t qwen_media_generated_text_len = 0;
	            char *qwen_media_generated_text =
	                render_rt_tokens_text(qwen_engine,
	                                      &qwen_media_generated_ds,
	                                      &qwen_media_generated_text_len);
	            rt_session_invalidate(qwen_media_cache_session);
	            uint8_t qwen_media_generated_ext = 0;
	            int qwen_media_generated_cached =
	                kv_cache_try_load_runtime_media_exact(
	                    &qwen_media_cache_server, &qwen_media_job.req,
	                    &qwen_media_generated_ds,
	                    qwen_media_generated_text ? qwen_media_generated_text : "",
	                    qwen_media_generated_text_len,
	                    &qwen_media_generated_ext);
	            TEST_ASSERT(qwen_media_generated_cached ==
	                        qwen_media_generated_ds.len);
	            TEST_ASSERT((qwen_media_generated_ext & KV_EXT_MEDIA_EXACT) != 0);
	            TEST_ASSERT(rt_session_pos(qwen_media_cache_session) ==
	                        qwen_media_generated_ds.len);
	            free(qwen_media_generated_text);
	            ds4_tokens_free(&qwen_media_generated_ds);
	            request_free(&qwen_media_job.req);
	            close(qwen_media_sv[0]);
	            close(qwen_media_sv[1]);
	        } else {
	            request_free(&qwen_media_gen_req);
	        }

	        request qwen_media_miss_req = {0};
	        qwen_err[0] = '\0';
	        TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	            "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
	            "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
	            "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/x-portable-pixmap,"
	            "P3%0A1%201%0A255%0A1%201%201%0A\"}}]}],"
	            "\"max_tokens\":1}",
	            4, 128, &qwen_media_miss_req, qwen_err, sizeof(qwen_err)));
	        rt_tokens qwen_media_miss_rt_prompt = {
	            .v = qwen_media_miss_req.prompt.v,
	            .len = qwen_media_miss_req.prompt.len,
	            .cap = qwen_media_miss_req.prompt.cap,
	        };
	        qwen_runtime_media_spans qwen_media_miss_spans = {0};
	        qwen_err[0] = '\0';
	        TEST_ASSERT(qwen_runtime_build_media_embeddings(
	                        qwen_engine, &qwen_media_miss_req,
	                        &qwen_media_miss_rt_prompt,
	                        &qwen_media_miss_spans,
	                        qwen_err, sizeof(qwen_err)));
	        ds4_tokens qwen_media_miss_prompt = {0};
	        ds4_tokens_copy_rt(&qwen_media_miss_prompt,
	                           &qwen_media_miss_spans.prompt);
	        size_t qwen_media_miss_text_len = 0;
	        char *qwen_media_miss_text =
	            render_rt_tokens_text(qwen_engine, &qwen_media_miss_prompt,
	                                  &qwen_media_miss_text_len);
	        uint8_t qwen_media_miss_ext = 0;
	        TEST_ASSERT(kv_cache_try_load_runtime_media_exact(
	                        &qwen_media_cache_server, &qwen_media_miss_req,
	                        &qwen_media_miss_prompt,
	                        qwen_media_miss_text ? qwen_media_miss_text : "",
	                        qwen_media_miss_text_len,
	                        &qwen_media_miss_ext) == 0);
	        TEST_ASSERT(qwen_media_miss_ext == 0);
	        rt_tokens qwen_media_text_effective = {0};
	        TEST_ASSERT(kv_cache_try_load_runtime_text(
	                        &qwen_media_cache_server,
	                        qwen_media_prompt_text ? qwen_media_prompt_text : "",
	                        &qwen_media_text_effective, NULL) == 0);
	        rt_tokens_free(&qwen_media_text_effective);
	        free(qwen_media_miss_text);
	        ds4_tokens_free(&qwen_media_miss_prompt);
	        qwen_runtime_media_spans_free(&qwen_media_miss_spans);
	        request_free(&qwen_media_miss_req);
	        free(qwen_media_prompt_text);
	        ds4_tokens_free(&qwen_media_cache_prompt);
	        qwen_runtime_media_spans_free(&qwen_media_cache_spans);
	        kv_cache_close(&qwen_media_cache_server.kv);
	        DIR *qwen_media_kv_scan = opendir(qwen_media_kv_dir);
	        if (qwen_media_kv_scan) {
	            struct dirent *de;
	            while ((de = readdir(qwen_media_kv_scan)) != NULL) {
	                if (de->d_name[0] == '.') continue;
	                char *path = path_join(qwen_media_kv_dir, de->d_name);
	                unlink(path);
	                free(path);
	            }
	            closedir(qwen_media_kv_scan);
	        }
	        rmdir(qwen_media_kv_dir);
	        pthread_mutex_destroy(&qwen_media_cache_server.tool_mu);
	        rt_session_free(qwen_media_cache_session);
	    }
	    request_free(&qwen_media_req);

	    qwen_err[0] = '\0';
	    TEST_ASSERT(parse_chat_request_rt(qwen_engine, qwen, NULL,
	        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
        "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AA==\"}},"
        "{\"type\":\"video_url\",\"video_url\":{\"url\":\"file:///tmp/frame.mp4\"}}]}],"
        "\"max_tokens\":1}",
        4, 128, &qwen_media_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(test_tokens_contain_triplet(qwen_media_req.prompt.v,
                                            qwen_media_req.prompt.len,
                                            QWEN36_TEST_VISION_START_TOKEN,
                                            QWEN36_TEST_IMAGE_PAD_TOKEN,
                                            QWEN36_TEST_VISION_END_TOKEN));
    TEST_ASSERT(test_tokens_contain_triplet(qwen_media_req.prompt.v,
                                            qwen_media_req.prompt.len,
                                            QWEN36_TEST_VISION_START_TOKEN,
                                            QWEN36_TEST_VIDEO_PAD_TOKEN,
                                            QWEN36_TEST_VISION_END_TOKEN));
    TEST_ASSERT(qwen_media_req.media.len == 2);
    TEST_ASSERT(!qwen_media_req.media.v[0].video);
    TEST_ASSERT(qwen_media_req.media.v[0].url != NULL);
    TEST_ASSERT(!strcmp(qwen_media_req.media.v[0].url,
                        "data:image/png;base64,AA=="));
    TEST_ASSERT(qwen_media_req.media.v[0].media_type != NULL);
    TEST_ASSERT(!strcmp(qwen_media_req.media.v[0].media_type, "image/png"));
    TEST_ASSERT(qwen_media_req.media.v[0].bytes_len == 1);
    TEST_ASSERT(qwen_media_req.media.v[0].bytes[0] == 0);
    TEST_ASSERT(qwen_media_req.media.v[1].video);
    TEST_ASSERT(qwen_media_req.media.v[1].url != NULL);
    TEST_ASSERT(!strcmp(qwen_media_req.media.v[1].url,
                        "file:///tmp/frame.mp4"));
    TEST_ASSERT(qwen_media_req.media.v[1].bytes_len == 0);
    request_free(&qwen_media_req);

    qwen_err[0] = '\0';
    TEST_ASSERT(parse_responses_request_rt(qwen_engine, qwen, NULL,
        "{\"model\":\"qwen3.6-27b\",\"input\":[{\"type\":\"message\","
        "\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"look\"},"
        "{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,AA==\"}]}],"
        "\"max_output_tokens\":1,\"reasoning\":{\"effort\":\"none\"}}",
        4, 128, &qwen_media_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(test_tokens_contain_triplet(qwen_media_req.prompt.v,
                                            qwen_media_req.prompt.len,
                                            QWEN36_TEST_VISION_START_TOKEN,
                                            QWEN36_TEST_IMAGE_PAD_TOKEN,
                                            QWEN36_TEST_VISION_END_TOKEN));
    TEST_ASSERT(qwen_media_req.media.len == 1);
    TEST_ASSERT(!qwen_media_req.media.v[0].video);
    TEST_ASSERT(qwen_media_req.media.v[0].media_type != NULL);
    TEST_ASSERT(!strcmp(qwen_media_req.media.v[0].media_type, "image/png"));
    TEST_ASSERT(qwen_media_req.media.v[0].bytes_len == 1);
    TEST_ASSERT(qwen_media_req.media.v[0].bytes[0] == 0);
    request_free(&qwen_media_req);

    qwen_err[0] = '\0';
    TEST_ASSERT(parse_anthropic_request_rt(qwen_engine, qwen, NULL,
        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\","
        "\"content\":[{\"type\":\"text\",\"text\":\"look\"},"
        "{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\","
        "\"data\":\"AA==\"}}]}],\"max_tokens\":1,"
        "\"thinking\":{\"type\":\"disabled\"}}",
        4, 128, &qwen_media_req, qwen_err, sizeof(qwen_err)));
    TEST_ASSERT(test_tokens_contain_triplet(qwen_media_req.prompt.v,
                                            qwen_media_req.prompt.len,
                                            QWEN36_TEST_VISION_START_TOKEN,
                                            QWEN36_TEST_IMAGE_PAD_TOKEN,
                                            QWEN36_TEST_VISION_END_TOKEN));
    TEST_ASSERT(qwen_media_req.media.len == 1);
    TEST_ASSERT(!qwen_media_req.media.v[0].video);
    TEST_ASSERT(qwen_media_req.media.v[0].url != NULL);
    TEST_ASSERT(!strcmp(qwen_media_req.media.v[0].url,
                        "data:image/png;base64,AA=="));
    TEST_ASSERT(qwen_media_req.media.v[0].media_type != NULL);
    TEST_ASSERT(!strcmp(qwen_media_req.media.v[0].media_type, "image/png"));
    TEST_ASSERT(qwen_media_req.media.v[0].bytes_len == 1);
    TEST_ASSERT(qwen_media_req.media.v[0].bytes[0] == 0);
    request_free(&qwen_media_req);

    rt_engine_close(qwen_engine);
    qwen_engine = NULL;

    qwen_extra.mmproj_path = qwen_bad_mmproj_path;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) != 0);
    TEST_ASSERT(qwen_engine == NULL);
    qwen_opt.model_options = NULL;
    }

    qwen_opt.model_path = "Qwen3.6-27B.gguf";
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) != 0);
    TEST_ASSERT(qwen_engine == NULL);

    qwen_opt.model_path = qwen_bad_shape_path;
    TEST_ASSERT(rt_engine_open_with_ops(&qwen_engine, qwen, &qwen_opt) != 0);
    TEST_ASSERT(qwen_engine == NULL);

    unlink(qwen_path);
    unlink(qwen_state_path);
    unlink(qwen_recurrent_path);
    unlink(qwen35_path);
    unlink(qwen_bad_shape_path);
    unlink(qwen_mmproj_path);
    unlink(qwen_mmproj_metal_path);
    unlink(qwen_bad_mmproj_path);
    unlink(qwen_embedded_mtp_path);
    unlink(qwen_mtp_path);
    unlink(qwen_mtp_multi_path);
    unlink(qwen_mtp_server_model_path);
    unlink(qwen_mtp_server_accept_path);
    unlink(qwen_mtp_reject_model_path);
    unlink(qwen_mtp_reject_path);
    unlink(qwen_bad_mtp_path);
    unlink(qwen_chat_path);
    unlink(ds4_path);
    unlink(scalar_path);

    rt_engine *mistral_engine = NULL;
    rt_engine_options mistral_opt = {
        .model_path = "Mistral-Medium-3.5-128B.gguf",
        .backend = RT_BACKEND_AUTO,
    };
    TEST_ASSERT(rt_engine_open_with_ops(&mistral_engine, mistral, &mistral_opt) != 0);
    TEST_ASSERT(mistral_engine == NULL);

    rt_tokens a = {0};
    rt_tokens b = {0};
    rt_tokens_push(&a, 10);
    rt_tokens_push(&a, 20);
    rt_tokens_copy(&b, &a);
    TEST_ASSERT(rt_tokens_starts_with(&a, &b));
    rt_tokens_push(&a, 30);
    TEST_ASSERT(rt_tokens_starts_with(&a, &b));
    TEST_ASSERT(!rt_tokens_starts_with(&b, &a));
    rt_tokens_free(&a);
    rt_tokens_free(&b);
}

typedef void (*test_fn)(void);

typedef struct {
    const char *flag;
    const char *name;
    const char *desc;
    test_fn fn;
} ds4_test_entry;

static const ds4_test_entry test_entries[] = {
#ifndef DS4_NO_GPU
    {"--long-context", "long-context", "long-context story fact-recall regression", test_long_story_fact_recall},
    {"--tool-call-quality", "tool-call-quality", "model emits valid DSML tool calls", test_tool_call_quality},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison", test_official_logprob_vectors},
    {"--metal-kernels", "metal-kernels", "isolated Metal kernel numeric regressions", test_metal_f16_matvec_fast_nr0_4},
#endif
    {"--runtime-core", "runtime-core", "model-agnostic runtime registry and token helpers", test_runtime_core_group},
    {"--qwen36-vectors", "qwen36-vectors", "Qwen3.6 official full-logit vector comparison", test_qwen36_official_vectors},
    {"--server", "server", "server parser/rendering/cache unit tests", test_server_unit_group},
};

static void test_print_help(const char *prog) {
    printf("Usage: %s [--all | TEST...]\n\n", prog);
    puts("Tests:");
    puts("  --all");
    puts("      Run every test. This is the default, ordered from slower to faster.");
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        printf("  %-20s %s\n", test_entries[i].flag, test_entries[i].desc);
    }
    puts("  --list");
    puts("      Print test names only.");
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  DS4_TEST_MODEL=FILE        Model path. Default: ds4flash.gguf");
    puts("  DS4_TEST_LONG_PROMPT=FILE  Rendered long-context story fact prompt.");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Simple official-vector fixture.");
    puts("  QWEN36_TEST_MODEL=FILE     Qwen3.6 GGUF for --qwen36-vectors.");
    puts("  QWEN36_TEST_VECTOR_FILE=FILE  Qwen3.6 official full-logit fixture.");
    puts("  QWEN36_REQUIRE_OFFICIAL_VECTORS=1  Fail instead of skipping missing Qwen vectors.");
    puts("  QWEN36_TEST_TEXT_ONLY=1  Skip Qwen mmproj/media regressions in runtime-core gates.");
}

static const ds4_test_entry *test_find_entry(const char *arg) {
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!strcmp(arg, test_entries[i].flag)) return &test_entries[i];
    }
    return NULL;
}

static void test_run_entry(const ds4_test_entry *entry) {
    int before = test_failures;
    fprintf(stderr, "%s:\n", entry->name);
    entry->fn();
    fprintf(stderr, "%s: ", entry->name);
    ds4_log(stderr,
            test_failures == before ? DS4_LOG_OK : DS4_LOG_ERROR,
            "%s",
            test_failures == before ? "OK" : "ERR");
    fputc('\n', stderr);
}

int main(int argc, char **argv) {
    bool run_all = argc == 1;
    bool selected[sizeof(test_entries) / sizeof(test_entries[0])] = {0};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) {
            run_all = true;
        } else if (!strcmp(argv[i], "--list")) {
            for (size_t j = 0; j < sizeof(test_entries) / sizeof(test_entries[0]); j++) {
                puts(test_entries[j].flag);
            }
            return 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            test_print_help(argv[0]);
            return 0;
        } else {
            const ds4_test_entry *entry = test_find_entry(argv[i]);
            if (!entry) {
                fprintf(stderr, "ds4-test: unknown test switch: %s\n", argv[i]);
                test_print_help(argv[0]);
                return 2;
            }
            selected[(size_t)(entry - test_entries)] = true;
        }
    }

    if (run_all) {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            test_run_entry(&test_entries[i]);
        }
    } else {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            if (selected[i]) test_run_entry(&test_entries[i]);
        }
    }

#ifndef DS4_NO_GPU
    test_close_engines();
#endif

    if (test_failures) {
        fprintf(stderr, "ds4 tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4 tests: ok");
    return 0;
}
