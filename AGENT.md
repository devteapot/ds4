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
- `models/qwen3.6-27b/include/qwen36_runtime.h` and
  `models/qwen3.6-27b/runtime/qwen36_runtime.c`: Qwen3.6 27B
  loader-bound tokenizer engine. It opens compatible GGUFs for metadata/tensor
  inspection, Qwen2-style pre-tokenization, Qwen chat tokenization, strict text
  tensor binding, and optional Qwen VL mmproj validation. It can create
  prompt/checkpoint state sessions,
  save/load v3 typed-section payloads with Qwen state-layout byte counts, last
  target hidden state, and lazy owned state buffers, estimate Qwen
  recurrent/full-attention context
  state, consume loaded logits for sampling/logprobs, and reject unimplemented
  execution backends and media placeholders without a valid mmproj.
  Mmproj-backed engines can render Qwen vision placeholders into token IDs, and
  the server maps OpenAI, Responses, and Anthropic image/video blocks to those
  placeholders. The Qwen runtime-core server path renders API tool schemas into
  Qwen XML tool prompts and maps generated Qwen XML tool calls back to
  structured API tool calls, with live Responses/Anthropic tool-result
  continuation while the worker session still owns the matching tool IDs.
  Runtime-core disk checkpoints can persist the optional tool-id map for exact
  Qwen XML tool-call replay and can key Qwen Responses-visible transcripts to
  hidden runtime payloads for restart recovery. Local Responses objects support
  both `previous_response_id` and local `conversation` IDs against that
  visible-transcript store, and `/v1/conversations` supports local conversation
  metadata create/get/update/delete plus text-only item
  create/list/retrieve/delete. Provider-hosted remote conversation sync and
  media-bearing conversation item storage remain gated. Server media decoding
  now covers
  local/base64/data/file URL image payloads, macOS HTTP(S) images, portable
  single/concatenated Netpbm frame payloads, and bounded high-frame macOS
  video-frame sampling; the server
  expands Qwen image/video placeholders to projected mmproj embeddings and syncs
  those hidden vectors into the session. Full upstream video metadata return and
  provider-level video sampling parity remain gated.
  Qwen can bind and validate MTP/nextn support layers and exposes a bounded CPU
  nextn draft suffix with exact target-model verification through runtime-core
  when MTP is enabled. Low-confidence draft suffixes are skipped through the
  Qwen MTP margin and return to exact greedy decode, plus per-session
  proposed/accepted/rejected/skipped counters for diagnostics. Checkpoints now
  persist the last target hidden state required by the verifier, reset
  non-persisted MTP draft counters on payload restore, and runtime-core server
  tests cover deterministic MTP verifier rejection and acceptance through
  `generate_job_rt`; the text gate also covers consecutive external nextn
  layers accepting a multi-token draft span and margin-gated skips with the
  skipped counter. Batched target verification and production acceleration
  remain gated.
  Token evaluation now has a CPU reference path for plain f32/f16/bf16
  tensors plus slow q8_0/q4_K/q6_K dequantization; broader quant coverage and
  production backends are still gated.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy, and runtime-core text serving for
  Qwen with text-prefix checkpoint reuse.
- `runtime-core/`: model-agnostic registry, token helpers, GGUF metadata/tensor
  probes, f32/f16/bf16/q8_0/q4_K/q6_K scalar reads and reference 2D matvec, plus
  engine/session vtable wrappers. It must not include model-specific
  architecture assumptions.
- `models/`: future home for one optimized runtime per model family.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface. Use `make qwen36-gates` for the broad local Qwen gate,
including synthetic mmproj/media prompt expansion and media-exact disk
checkpoints; use `make qwen36-text-gates` for the Qwen text milestone when
mmproj/media regressions should stay out of the pass. Use the matching
`*-strict` targets only when a pinned Qwen official-vector fixture and
`QWEN36_TEST_MODEL` are available, and use the matching `*-cpu` targets to run
through a `DS4_NO_GPU` test binary. The text gate includes
Qwen pre-tokenization regressions for contractions, single digits, and
punctuation/newline pieces, chat-template last-query thinking preservation,
synthetic full-attention KV payload
save/load/continue coverage, including a restored-KV
continuation that must affect the next hidden state, and synthetic
Gated DeltaNet/convolution payload save/load/continue coverage, including
server disk-cache reload/continue coverage for the nonzero recurrent/conv
sections, plus OpenAI completion/chat, Responses, and Anthropic SSE streaming
through the runtime-core Qwen server path, and multi-token external MTP draft
acceptance. It does not prove real long-context parity without the pinned
official vector fixture.
When Metal is available, the same text gate also opens a Qwen Metal engine on
the synthetic recurrent fixture, runs text evaluation through it, compares the
first hidden-state probe against the CPU fixture, and requires real
matvec/RMSNorm/SiLU/L2/Gated Delta/full-attention helper calls with no helper
fallbacks.
