# Runtime Core

This directory is the shared layer for the model-specific runtime framework.
It owns the model-agnostic C ABI in `include/rt_runtime.h` and the small
registry/wrapper implementation in `src/runtime.c`.

The core does not know DeepSeek, Qwen, compressed attention, dense attention,
MoE routing, RoPE details, quant formats, or backend kernel names.  It only
knows how to select a registered model runtime and drive opaque engine/session
objects through a narrow vtable.

The current CLI, server, benchmark, eval, and some disk-KV code still live at
the repository root because they are DS4-shaped in a few important places. They
should move into this layer only when a second model runtime proves which APIs
are genuinely shared.

Runtime hooks are intentionally coarse:

- model probing and engine open/close
- tokenization and token text
- chat rendering
- context memory estimation
- session create/free/sync/eval
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
- `rt_engine_*`, `rt_session_*`, and token helpers: wrappers that own the
  generic engine/session shells and dispatch into the selected runtime.

Keep model-family data behind `model_options` structs declared by that runtime.
For example, DS4 uses `ds4_runtime_engine_options` and
`ds4_runtime_chat_options`; a Qwen runtime should define its own equivalents
instead of extending the core ABI with Qwen-specific fields.

## Adding A Runtime

To add a new optimized model family, create `models/<model-family>/runtime/`
with an adapter that returns one static `rt_model_ops` descriptor and a
registration function. The adapter should translate the core types into the
model's native engine/session API, while the tensor binder, graph schedule,
quantization decisions, prompt rules, and backend kernels remain under the
model directory.

The lightweight contract test is:

```sh
./ds4_test --runtime-core
```

Add equivalent model-family tests when a new runtime lands; the core test only
checks the shared registry/helpers and the currently registered DS4 adapter.

## Documentation

When this ABI changes, update this file together with:

- `README.md` for the user-facing architecture direction.
- `models/README.md` for the expected runtime layout and adapter contract.
- `models/<model-family>/README.md` for model-specific options, backends, and
  validation.
- `CONTRIBUTING.md` when contributors need a new check.

Good candidates for future extraction:

- process entrypoints and option parsing shared by model runtimes
- OpenAI/Anthropic-compatible server plumbing
- sampling utilities and logprob reporting
- disk-KV cache policy and file management
- benchmark and regression harnesses
- GGUF metadata helpers that are not model-specific

Model architecture, tensor binding, graph scheduling, quantization choices, and
backend kernels should stay under `models/<model-family>/`.
