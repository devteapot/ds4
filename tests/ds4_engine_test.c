#define DS4_ENGINE_TEST_NO_MAIN
#include "../ds4_engine.c"

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "ds4-engine-test: assertion failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_parse_request_envelope(void) {
    const char *line =
        "{\"type\":\"request\",\"id\":\"req-1\",\"method\":\"engine.describe\","
        "\"params\":{\"ignored\":true}}";
    engine_request r = {0};
    char err[128];
    TEST_ASSERT(parse_engine_request(line, &r, err, sizeof(err)));
    TEST_ASSERT(strcmp(r.id, "req-1") == 0);
    TEST_ASSERT(strcmp(r.method, "engine.describe") == 0);
    TEST_ASSERT(r.params_json != NULL);
    TEST_ASSERT(strstr(r.params_json, "\"ignored\"") != NULL);
    engine_request_free(&r);
}

static void test_parse_sync_rendered_text(void) {
    const char *json =
        "{\"sessionId\":\"s1\",\"prefix\":{\"kind\":\"rendered_text\","
        "\"text\":\"<\\uff5cbegin\\u2581of\\u2581sentence\\uff5c><\\uff5cUser\\uff5c>hi\"},"
        "\"options\":{\"allowRebuild\":true}}";
    sync_params p = {0};
    char err[128];
    TEST_ASSERT(parse_sync_params(json, &p, err, sizeof(err)));
    TEST_ASSERT(strcmp(p.session_id, "s1") == 0);
    TEST_ASSERT(strstr(p.prefix_text, "begin") != NULL);
    TEST_ASSERT(strstr(p.prefix_text, "hi") != NULL);
    sync_params_free(&p);
}

static void test_reject_token_sync(void) {
    const char *json =
        "{\"sessionId\":\"s1\",\"prefix\":{\"kind\":\"tokens\",\"tokens\":[1,2,3]}}";
    sync_params p = {0};
    char err[128];
    TEST_ASSERT(!parse_sync_params(json, &p, err, sizeof(err)));
    TEST_ASSERT(strstr(err, "rendered_text") != NULL);
}

static void test_parse_generate_options(void) {
    const char *json =
        "{\"sessionId\":\"s1\",\"options\":{\"maxTokens\":8,\"temperature\":0,"
        "\"topP\":0.9,\"minP\":0.02,\"topK\":40,\"seed\":123,\"stop\":[]}}";
    generate_params p = {0};
    char err[128];
    TEST_ASSERT(parse_generate_params(json, &p, err, sizeof(err)));
    TEST_ASSERT(strcmp(p.session_id, "s1") == 0);
    TEST_ASSERT(p.max_tokens == 8);
    TEST_ASSERT(p.temperature == 0.0f);
    TEST_ASSERT(p.top_k == 40);
    TEST_ASSERT(p.seed == 123);
    TEST_ASSERT(p.has_seed);
    TEST_ASSERT(!p.has_stop);
    generate_params_free(&p);
}

static void test_detect_stop_sequences(void) {
    const char *json =
        "{\"sessionId\":\"s1\",\"options\":{\"maxTokens\":8,\"stop\":[\"END\"]}}";
    generate_params p = {0};
    char err[128];
    TEST_ASSERT(parse_generate_params(json, &p, err, sizeof(err)));
    TEST_ASSERT(p.has_stop);
    generate_params_free(&p);
}

static void test_generate_requires_options(void) {
    const char *json = "{\"sessionId\":\"s1\"}";
    generate_params p = {0};
    char err[128];
    TEST_ASSERT(!parse_generate_params(json, &p, err, sizeof(err)));
    TEST_ASSERT(strstr(err, "options") != NULL);
}

static void test_parse_kv_disk_options(void) {
    char *argv[] = {
        "ds4-engine",
        "--socket", "/tmp/ds4-engine.sock",
        "--ctx", "140000",
        "--kv-disk-dir", "/tmp/ds4-kv",
        "--kv-disk-space-mb", "8192",
        "--kv-cache-min-tokens", "1024",
        "--kv-cache-cold-max-tokens", "0",
        "--kv-cache-continued-interval-tokens", "4096",
        "--kv-cache-boundary-trim-tokens", "16",
        "--kv-cache-boundary-align-tokens", "1024",
        "--kv-cache-reject-different-quant",
    };
    int argc = (int)(sizeof(argv) / sizeof(argv[0]));
    engine_config c = parse_options(argc, argv);
    TEST_ASSERT(strcmp(c.socket_path, "/tmp/ds4-engine.sock") == 0);
    TEST_ASSERT(c.ctx_size == 140000);
    TEST_ASSERT(strcmp(c.kv_disk_dir, "/tmp/ds4-kv") == 0);
    TEST_ASSERT(c.kv_disk_space_mb == 8192);
    TEST_ASSERT(c.kv_cache.min_tokens == 1024);
    TEST_ASSERT(c.kv_cache.cold_max_tokens == 0);
    TEST_ASSERT(c.kv_cache.continued_interval_tokens == 4096);
    TEST_ASSERT(c.kv_cache.boundary_trim_tokens == 16);
    TEST_ASSERT(c.kv_cache.boundary_align_tokens == 1024);
    TEST_ASSERT(c.kv_cache_reject_different_quant);
}

static void test_response_error_shape(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    client_conn *c = client_create(sv[0]);
    TEST_ASSERT(send_response_error(c, "req-err", "busy", "session is busy", true));

    char out[512];
    ssize_t n = read(sv[1], out, sizeof(out) - 1);
    TEST_ASSERT(n > 0);
    out[n] = '\0';
    TEST_ASSERT(strstr(out, "\"type\":\"response\"") != NULL);
    TEST_ASSERT(strstr(out, "\"id\":\"req-err\"") != NULL);
    TEST_ASSERT(strstr(out, "\"ok\":false") != NULL);
    TEST_ASSERT(strstr(out, "\"code\":\"busy\"") != NULL);
    TEST_ASSERT(strstr(out, "\"retryable\":true") != NULL);

    client_unref(c);
    close(sv[1]);
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    test_parse_request_envelope();
    test_parse_sync_rendered_text();
    test_reject_token_sync();
    test_parse_generate_options();
    test_detect_stop_sequences();
    test_generate_requires_options();
    test_parse_kv_disk_options();
    test_response_error_shape();
    fprintf(stderr, "ds4-engine-test: ok\n");
    return 0;
}
