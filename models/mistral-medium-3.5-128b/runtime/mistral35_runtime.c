#include "mistral35_runtime.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const mistral35_model_spec MISTRAL35_MODEL_SPEC = {
    .hf_repo = MISTRAL35_RUNTIME_HF_REPO,
    .api_model = MISTRAL35_RUNTIME_API_MODEL,
    .display_name = "Mistral Medium 3.5 128B",
    .text_model_type = "ministral3",
    .parameter_count_b = 128,
    .vocab_size = 131072,
    .max_context = 262144,
    .hidden_size = 12288,
    .intermediate_size = 28672,
    .n_layers = 88,
    .attention_heads = 96,
    .attention_kv_heads = 8,
    .attention_head_dim = 128,
    .original_rope_context = 4096,
    .yarn_factor = 64,
    .vision_layers = 48,
    .vision_hidden_size = 1664,
    .vision_intermediate_size = 8192,
    .vision_attention_heads = 16,
    .vision_image_size = 1540,
    .vision_patch_size = 14,
    .vision_spatial_merge_size = 2,
    .dense_weights = true,
    .multimodal_checkpoint = true,
    .fp8_checkpoint = true,
};

const mistral35_model_spec *mistral35_runtime_model_spec(void) {
    return &MISTRAL35_MODEL_SPEC;
}

static bool mistral35_contains_ci(const char *haystack, const char *needle) {
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

static bool mistral35_rt_probe_model_path(const char *model_path) {
    /*
     * This first adapter can only probe by path spelling.  Once the GGUF
     * metadata helpers are shared, this should validate architecture metadata
     * instead of trusting filenames.
     */
    return mistral35_contains_ci(model_path, "mistral-medium-3.5-128b") ||
           mistral35_contains_ci(model_path, "mistral-medium-3-5-128b") ||
           mistral35_contains_ci(model_path, "mistral_medium_3_5_128b") ||
           mistral35_contains_ci(model_path, "mistral-medium-35-128b");
}

static int mistral35_rt_engine_open(void **out, const rt_engine_options *opt) {
    (void)opt;
    if (!out) return 1;
    *out = NULL;
    return 1;
}

static void mistral35_rt_engine_close(void *engine) {
    free(engine);
}

static void mistral35_rt_engine_summary(void *engine) {
    (void)engine;
    const mistral35_model_spec *s = mistral35_runtime_model_spec();
    fprintf(stderr,
            "%s runtime scaffold\n"
            "  repo: %s\n"
            "  API model: %s\n"
            "  status: registry/probe only; engine is not implemented yet\n"
            "  text model: %uB dense, %u layers, hidden %u, vocab %u, native ctx %u\n"
            "  vision model: %u layers, hidden %u, image %u, patch %u\n",
            s->display_name,
            s->hf_repo,
            s->api_model,
            s->parameter_count_b,
            s->n_layers,
            s->hidden_size,
            s->vocab_size,
            s->max_context,
            s->vision_layers,
            s->vision_hidden_size,
            s->vision_image_size,
            s->vision_patch_size);
}

static uint32_t mistral35_rt_engine_vocab_size(void *engine) {
    (void)engine;
    return MISTRAL35_MODEL_SPEC.vocab_size;
}

static rt_context_memory mistral35_rt_estimate_context_memory(rt_backend backend, int ctx_size) {
    (void)backend;
    (void)ctx_size;
    return (rt_context_memory){0};
}

static int mistral35_rt_session_create(void **out, void *engine, int ctx_size) {
    (void)engine;
    (void)ctx_size;
    if (!out) return 1;
    *out = NULL;
    return 1;
}

static void mistral35_rt_session_free(void *session) {
    free(session);
}

static const rt_model_ops MISTRAL35_RUNTIME_OPS = {
    .abi_version = RT_RUNTIME_ABI_VERSION,
    .family = MISTRAL35_RUNTIME_FAMILY,
    .display_name = "Mistral Medium 3.5 128B",
    .probe_model_path = mistral35_rt_probe_model_path,
    .engine_open = mistral35_rt_engine_open,
    .engine_close = mistral35_rt_engine_close,
    .engine_summary = mistral35_rt_engine_summary,
    .engine_vocab_size = mistral35_rt_engine_vocab_size,
    .estimate_context_memory = mistral35_rt_estimate_context_memory,
    .session_create = mistral35_rt_session_create,
    .session_free = mistral35_rt_session_free,
};

const rt_model_ops *mistral35_runtime_ops(void) {
    return &MISTRAL35_RUNTIME_OPS;
}

int mistral35_runtime_register(void) {
    return rt_register_model(&MISTRAL35_RUNTIME_OPS);
}
