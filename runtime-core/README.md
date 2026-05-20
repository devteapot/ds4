# Runtime Core

This directory is the shared layer for the model-specific runtime framework.
It owns the model-agnostic C ABI in `include/rt_runtime.h` and the small
registry/wrapper implementation in `src/runtime.c`.

The core does not know specific families such as DeepSeek, Qwen, or Mistral;
it also does not know compressed attention, dense attention, MoE routing, RoPE
details, quant formats, or backend kernel names.  It only knows how to select a
registered model runtime and drive opaque engine/session objects through a
narrow vtable.

The current CLI, server, benchmark, eval, and some disk-KV code still live at
the repository root because they are DS4-shaped in a few important places. They
should move into this layer only when additional model runtimes prove which APIs
are genuinely shared.

Runtime hooks are intentionally coarse:

- model probing and engine open/close
- tokenization and token text
- chat rendering
- context memory estimation
- session create/free/sync/eval
- optional speculative argmax-span eval for model-owned verifier paths
- sampling, top-logprobs, and optional raw logits readback
- opaque session payload save/load

Model runtimes decide how those operations are optimized internally.  The core
does not expose callback slots for individual graph optimizations because those
would accidentally encode one model family's assumptions into everyone else.

## Current ABI

`include/rt_runtime.h` exposes:

- `rt_backend`: the backend selection enum shared by runtimes (`auto`, `cpu`,
  `metal`, `cuda`, `rocm`, and `mlx`).
- `rt_tokens`, `rt_token_score`, and `rt_context_memory`: small shared data
  containers used by wrappers and tests.
- `rt_engine_options` and `rt_chat_render_options`: model-agnostic option
  structs with `model_options` escape hatches for family-specific data.
- `rt_model_ops`: the registration vtable every model runtime implements.
- `rt_register_model`, `rt_model_by_family`, and `rt_probe_model_path`: the
  process-local runtime registry.
- `rt_gguf_metadata_read`, `rt_gguf_metadata_free`, and
  `rt_gguf_metadata_get_*`: lightweight GGUF metadata helpers for model-family
  probes and early validation. They read scalar/string metadata and array
  descriptors without loading tensor data.
- `rt_gguf_file_open`, `rt_gguf_file_find_tensor`, and related tensor helpers:
  a shared read-only GGUF file layer for mmap-backed tensor directory
  inspection, tensor byte offsets, and tensor type sizing.
- `rt_gguf_tensor_read_f32`: scalar CPU-reference access for plain `f32`,
  `f16`, and `bf16` tensors, plus slow reference dequantization for `q4_0`,
  `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`, `q2_K`, `q3_K`, `q4_K`, `q5_K`,
  `q6_K`, `q8_K`, `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`,
  `iq3_xxs`, `iq3_s`, `iq4_nl`, and `iq4_xs`.
- `rt_gguf_tensor_matvec_f32`: a small CPU-reference matvec for GGUF 2D tensors
  with `dim[0] == input` and `dim[1] == output`. It is intended for correctness
  gates and diagnostics, not production decode speed.
- `rt_gguf_file_array_cursor`, `rt_gguf_array_cursor_next_string`, and
  `rt_gguf_file_array_string_at`: mapped string-array access for tokenizer
  tables such as `tokenizer.ggml.tokens` and `tokenizer.ggml.merges`.
- `rt_engine_*`, `rt_session_*`, and token helpers: wrappers that own the
  generic engine/session shells and dispatch into the selected runtime.
- `rt_session_eval_speculative_argmax`: an optional model-owned hook that
  commits one or more exact argmax tokens in a session. Runtimes may implement
  this with a real speculative verifier, a low-confidence draft gate, or a
  correctness-first greedy span; the core treats a zero return as "not
  available" and falls back to ordinary eval.
- `rt_session_tokens`: an optional exact token-history view for live
  continuation paths that need to append a freshly tokenized suffix to the
  runtime's current session frontier.

Keep model-family data behind `model_options` structs declared by that runtime.
For example, DS4 uses `ds4_runtime_engine_options` and
`ds4_runtime_chat_options`; Qwen and Mistral runtimes should define their own
equivalents instead of extending the core ABI with family-specific fields.

## Adding A Runtime

To add a new optimized model family, create `models/<model-family>/runtime/`
with an adapter that returns one static `rt_model_ops` descriptor and a
registration function. The adapter should translate the core types into the
model's native engine/session API, while the tensor binder, graph schedule,
quantization decisions, prompt rules, and backend kernels remain under the
model directory.

Probe readable GGUF files with the shared metadata helpers before falling back
to filename aliases. Filename probing is useful for a model path that does not
exist yet, but readable files should be selected by `general.architecture` and
family-specific shape metadata.

The lightweight contract test is:

```sh
./ds4_test --runtime-core
```

Add equivalent model-family tests when a new runtime lands; the core test only
checks the shared registry/helpers, the DS4 adapter, and any scaffold adapters
that are registered before their engine exists.

## Documentation

When this ABI changes, update this file together with:

- `README.md` for the user-facing architecture direction.
- `models/README.md` for the expected runtime layout and adapter contract.
- `models/<model-family>/README.md` for model-specific options, backends, and
  validation.
- `CONTRIBUTING.md` when contributors need a new check.

Good candidates for future extraction:

- process entrypoints and option parsing shared by model runtimes
- protocol object storage abstractions beyond DS4's local
  Responses/Conversations object files
- broader runtime-core disk-cache policy beyond text-prefix payload reuse and
  optional exact tool-call replay maps
- shared media-aware runtime checkpoint helpers for adapters that inject
  non-text embeddings
- sampling utilities and logprob reporting
- disk-KV cache policy and file management
- benchmark and regression harnesses
- deeper GGUF helpers for tokenizer array values once multiple runtimes need
  them

Model architecture, tensor binding, graph scheduling, quantization choices, and
backend kernels should stay under `models/<model-family>/`.
