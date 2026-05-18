# DeepSeek V4 Flash Runtime

This is the current production runtime behind the `ds4`, `ds4-server`,
`ds4-bench`, and `ds4-eval` binaries.

It is intentionally model-specific.  The runtime validates the DeepSeek V4 Flash
GGUF layout, binds known tensor names, assumes fixed model dimensions, and uses
DS4-specific graph scheduling for compressed attention, hyper-connections,
routed MoE, MTP support, KV persistence, and imatrix collection.

Layout:

- `include/ds4.h`: public engine/session API consumed by the root CLI, server,
  benchmark, eval, and tests.
- `include/ds4_runtime.h`: DS4 adapter options and registration entrypoints for
  the model-agnostic runtime core.
- `engine/ds4.c`: GGUF loading, tokenizer, tensor binding, CPU reference paths,
  graph scheduling, sessions, sampling support, KV payload serialization, and
  DeepSeek V4 Flash prompt helpers.
- `runtime/ds4_runtime.c`: `rt_model_ops` adapter that maps the core runtime
  vtable onto `ds4_engine` and `ds4_session`.
- `backends/ds4_gpu.h`: narrow tensor/backend API used by `engine/ds4.c`.
- `backends/metal/ds4_metal.m`: Metal runtime, memory management, pipeline
  setup, and wrappers around the Metal kernels.
- `backends/metal/kernels/*.metal`: DS4 Metal kernels.
- `backends/cuda/ds4_cuda.cu`: CUDA runtime and kernels.
- `backends/cuda/ds4_iq2_tables_cuda.inc`: CUDA IQ2 lookup tables.

Framework adapter:

- Register DS4 with `ds4_runtime_register()` or fetch its descriptor with
  `ds4_runtime_ops()`.
- Pass DS4-only engine options through `rt_engine_options.model_options` using
  `ds4_runtime_engine_options` for MTP and directional steering fields.
- Pass DS4-only chat options through `rt_chat_render_options.model_options`
  using `ds4_runtime_chat_options` for thinking-mode selection.
- Validate adapter changes with `./ds4_test --runtime-core`; run the model-backed
  DS4 checks in `CONTRIBUTING.md` when tokenizer, prompt, logits, KV, or backend
  behavior changes.

Future model runtimes should copy this separation but not the DeepSeek V4 Flash
mechanics. For example, a Qwen runtime should have its own tensor binder, RoPE
rules, KV layout, dense or MoE FFN schedule, quant choices, and backend kernels.
