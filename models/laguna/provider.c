#include "provider.h"

#include "../../ds4_model_provider_builtin.h"

static const ds4_model_provider_v1 DS4_LAGUNA_PROVIDER = {
    .abi_version = DS4_MODEL_PROVIDER_ABI_VERSION,
    .struct_size = sizeof(ds4_model_provider_v1),
    .id = "laguna-s2.1",
    .session_create = ds4_laguna_session_create,
    .session_destroy = ds4_laguna_session_destroy,
    .session_sync = ds4_laguna_session_sync,
    .session_eval = ds4_laguna_session_eval,
    .sessions_eval_batch = ds4_builtin_sessions_eval_batch,
    .sessions_eval_batch_with_prefill =
        ds4_builtin_sessions_eval_batch_with_prefill,
    .session_eval_speculative = ds4_laguna_session_eval_speculative,
    .session_invalidate = ds4_laguna_session_invalidate,
    .session_rewind = ds4_laguna_session_rewind,
    .session_layer_slice_reset = ds4_laguna_session_layer_slice_reset,
    .session_eval_output_head = ds4_laguna_session_eval_output_head,
    .session_eval_layer_slice = ds4_laguna_session_eval_layer_slice,
    .session_payload_bytes = ds4_laguna_session_payload_bytes,
    .session_save_payload = ds4_laguna_session_save_payload,
    .session_load_payload = ds4_laguna_session_load_payload,
    .session_layer_payload_bytes =
        ds4_laguna_session_layer_payload_bytes,
    .session_save_layer_payload =
        ds4_laguna_session_save_layer_payload,
    .session_load_layer_payload =
        ds4_laguna_session_load_layer_payload,
};

const ds4_model_provider_v1 *ds4_laguna_model_provider(void) {
    return &DS4_LAGUNA_PROVIDER;
}
