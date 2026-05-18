#ifndef DS4_RUNTIME_H
#define DS4_RUNTIME_H

#include "ds4.h"
#include "rt_runtime.h"

typedef struct {
    const char *mtp_path;
    int mtp_draft_tokens;
    float mtp_margin;
    const char *directional_steering_file;
    float directional_steering_attn;
    float directional_steering_ffn;
} ds4_runtime_engine_options;

typedef struct {
    ds4_think_mode think_mode;
} ds4_runtime_chat_options;

const rt_model_ops *ds4_runtime_ops(void);
int ds4_runtime_register(void);

#endif
