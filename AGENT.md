# Agent Notes

`models/deepseek-v4-flash/engine/ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under
`models/deepseek-v4-flash/backends/metal/kernels/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.
- Keep docs in the same patch as framework, layout, public API, build, or test
  expectation changes. Update `README.md`, `runtime-core/README.md`,
  `models/README.md`, the affected model README, `CONTRIBUTING.md`, and this
  file when their contracts move.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short Metal smoke tests for build verification.

## Layout

- `models/deepseek-v4-flash/engine/ds4.c`: model loading, tokenizer, CPU
  reference code, graph scheduling, sessions, disk-cache payload serialization.
- `models/deepseek-v4-flash/include/ds4.h`: public DS4 engine/session boundary
  used by the CLI, server, eval, and bench tools.
- `models/deepseek-v4-flash/include/ds4_runtime.h` and
  `models/deepseek-v4-flash/runtime/ds4_runtime.c`: adapter exposing DS4
  through the model-agnostic `rt_model_ops` boundary.
- `models/deepseek-v4-flash/backends/ds4_gpu.h`: narrow tensor API shared by
  the DS4 graph driver and accelerator backends.
- `models/deepseek-v4-flash/backends/metal/ds4_metal.m`: Objective-C Metal
  runtime and kernel wrappers.
- `models/deepseek-v4-flash/backends/metal/kernels/*.metal`: DS4 Metal compute
  kernels.
- `models/deepseek-v4-flash/backends/cuda/ds4_cuda.cu`: DS4 CUDA runtime and
  kernels.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `runtime-core/`: model-agnostic registry, token helpers, and engine/session
  vtable wrappers. It must not include model-specific architecture assumptions.
- `models/`: future home for one optimized runtime per model family.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.
