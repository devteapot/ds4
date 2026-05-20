#ifndef MISTRAL35_RUNTIME_H
#define MISTRAL35_RUNTIME_H

#include "rt_runtime.h"

#include <stdbool.h>
#include <stdint.h>

#define MISTRAL35_RUNTIME_FAMILY "mistral-medium-3.5-128b"
#define MISTRAL35_RUNTIME_HF_REPO "mistralai/Mistral-Medium-3.5-128B"
#define MISTRAL35_RUNTIME_API_MODEL "mistral-medium-3.5"

typedef enum {
    MISTRAL35_REASONING_AUTO = 0,
    MISTRAL35_REASONING_NONE,
    MISTRAL35_REASONING_HIGH,
} mistral35_reasoning_mode;

typedef struct {
    bool language_model_only;
    bool enable_vision;
} mistral35_runtime_engine_options;

typedef struct {
    mistral35_reasoning_mode reasoning_mode;
    bool preserve_reasoning;
} mistral35_runtime_chat_options;

typedef struct {
    const char *hf_repo;
    const char *api_model;
    const char *display_name;
    const char *text_model_type;
    uint32_t parameter_count_b;
    uint32_t vocab_size;
    uint32_t max_context;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t n_layers;
    uint32_t attention_heads;
    uint32_t attention_kv_heads;
    uint32_t attention_head_dim;
    uint32_t original_rope_context;
    uint32_t yarn_factor;
    uint32_t vision_layers;
    uint32_t vision_hidden_size;
    uint32_t vision_intermediate_size;
    uint32_t vision_attention_heads;
    uint32_t vision_image_size;
    uint32_t vision_patch_size;
    uint32_t vision_spatial_merge_size;
    bool dense_weights;
    bool multimodal_checkpoint;
    bool fp8_checkpoint;
} mistral35_model_spec;

const mistral35_model_spec *mistral35_runtime_model_spec(void);
const rt_model_ops *mistral35_runtime_ops(void);
int mistral35_runtime_register(void);

#endif
