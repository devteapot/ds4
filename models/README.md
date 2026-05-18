# Model Runtimes

This directory is the start of the model-specific runtime framework.

The rule is: each model family gets a narrow optimized runtime instead of being
forced through one generic graph.  A runtime is allowed to bake in model shape,
tensor names, attention/KV behavior, quantization recipes, prompt rendering
constraints, and backend kernels when that makes inference faster or more
reliable.

Expected shape for new model families:

```text
models/<model-family>/
  README.md
  include/          public engine boundary for current entrypoints
  engine/           loader, tensor binder, graph/session logic
  runtime/          adapter that exposes the model through rt_model_ops
  backends/
    metal/          Metal runtime and model kernels
    cuda/           CUDA runtime and model kernels
    rocm/           ROCm/HIP runtime when available
    mlx/            MLX runtime when useful
  tools/            model-specific conversion, quant, and validation helpers
  tests/            official vectors and model-family regressions
```

The existing CLI, server, benchmark, eval, and GGUF utilities still live at the
repository root or under `gguf-tools/`. They remain DS4-shaped today, but those
are the candidates for a future shared `runtime-core` once a second runtime,
such as Qwen, proves the right abstraction boundaries.

Each model runtime should expose one `rt_model_ops` descriptor through a small
adapter.  That descriptor is the only hook the core needs: model-specific
optimizations stay behind the adapter in the model's graph and backend code.

Adapter checklist:

- Declare public model-specific options in `include/<model>_runtime.h`.
- Implement `runtime/<model>_runtime.c` with `*_runtime_ops()` and
  `*_runtime_register()` entrypoints.
- Translate `rt_backend` into the model's supported backend set and reject
  unsupported backends explicitly.
- Keep shape constants, tensor names, prompt quirks, KV layout, quantization,
  and kernel choices out of `runtime-core/`.
- Document the runtime's supported backends, options, and validation commands in
  `models/<model-family>/README.md`.
