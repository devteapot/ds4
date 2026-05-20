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

Current runtime directories:

- `deepseek-v4-flash/`: production DS4 runtime.
- `qwen3.6-27b/`: Qwen3.6 27B loader-bound tokenizer engine. It opens
  compatible GGUFs for metadata/tensor-directory inspection, Qwen chat
  tokenization, strict text tensor binding, and optional Qwen VL mmproj
  validation. It can create prompt/checkpoint state sessions, and its v3 payload
  format records Qwen state-layout byte counts plus typed token/logit/last-hidden/state
  sections that can be loaded and re-saved. Token evaluation now has a CPU
  reference path for plain f32/f16/bf16 tensors and slow
  q4_0/q4_1/q5_0/q5_1/q8_0/q8_1/q2_K/q3_K/q4_K/q5_K/q6_K/q8_K/iq1_s/iq1_m/iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq4_nl/iq4_xs
  dequantization with lazy Qwen state allocation; official-vector parity and
  production backends remain gated, with the Qwen full-logit vector
  runner/capture script now under `tests/qwen36-vectors/`.
  `ds4-server`
  can serve Qwen text `/v1/chat/completions`, `/v1/completions`,
  `/v1/responses`, and `/v1/messages` through runtime-core when launched with a
  compatible Qwen GGUF, and `--kv-disk-dir` can persist/reload Qwen prompt
  checkpoints and generated completion prefixes by rendered text prefix,
  including the optional tool-id map for exact Qwen XML tool-call replay.
  The Qwen runtime-core server path can render tool schemas into Qwen's XML
  tool prompt and map generated XML tool calls back to structured API tool
  calls, including live Responses/Anthropic tool-result continuation while the
  worker session still owns the matching tool IDs. Runtime-core disk checkpoints
  can key Qwen Responses-visible transcripts to hidden runtime payloads for
  restart recovery, and media-bearing prompt/generated states use exact media
  digest keys. Local Responses objects support `previous_response_id` and local
  `conversation` IDs by persisting visible response text and tool-call IDs
  beside the KV checkpoint directory, and `/v1/conversations` supports local
  metadata create/get/update/delete plus text-only item
  create/list/retrieve/delete. Provider-hosted remote object sync and
  media-bearing conversation item storage remain gated. Production GPU backends
  remain gated. Qwen currently opens for
  `auto`/`cpu`, plus an early explicit
  macOS `metal` backend that compiles and self-tests Qwen
  dense/scalar-block/K-quant matvec/RMSNorm/SiLU-mul/L2Norm/Gated
  DeltaNet/full-attention-head kernels; CUDA opens are
  rejected until Qwen kernels exist. Its memory estimate
  includes the planned recurrent state, convolution state, and periodic
  full-attention KV cache.
  Qwen engines without mmproj support reject image/video content and
  vision-placeholder prompts; mmproj-backed engines can translate OpenAI,
  Responses, and Anthropic image/video blocks into placeholder token IDs. Qwen
  preserves media URLs or decoded base64/data-URL bytes, resolves file URLs and
  macOS HTTP(S) media URLs to bytes, and can now decode ImageIO-supported image
  payloads on macOS, PNG and Netpbm image payloads in portable builds, Netpbm
  video-frame payloads, and up to four temporally sampled AVFoundation-decoded
  frames from video containers on macOS into Qwen-style smart-resized, bicubic-sampled,
  CLIP-normalized, aspect-preserving patch grids. Per-media
  `min_pixels`/`max_pixels` and explicit `resized_height`/`resized_width`
  controls are honored in the processor path, and video `nframes`,
  `min_frames`, `max_frames`, `fps`, `video_start`, and `video_end` controls
  bound the current macOS container decode cap and Qwen frame-factor-aligned
  time window. Qwen can project supplied
  image patch grids through the loaded mmproj on the CPU, expand image/video
  pad placeholders to the projected token count, and sync prompts with
  caller-provided hidden embeddings at selected token positions. Qwen can bind
  and validate MTP/nextn support tensors from embedded
  or separate GGUF files, and exposes a bounded CPU nextn draft suffix through
  the runtime-core argmax-span hook when MTP is enabled. Low-confidence draft
  suffixes are skipped through the Qwen MTP margin and return to exact greedy
  decode, with per-session proposed/accepted/rejected/skipped counters
  available for diagnostics. Qwen checkpoints persist the last target hidden
  state needed by the verifier; batched target verification and production
  acceleration remain gated.
- `mistral-medium-3.5-128b/`: Mistral Medium 3.5 128B registry/probe scaffold.
  Engine open is intentionally unsupported until the Mistral-specific loader,
  prompt renderer, graph, and backends land.

Each model runtime should expose one `rt_model_ops` descriptor through a small
adapter.  That descriptor is the only hook the core needs: model-specific
optimizations stay behind the adapter in the model's graph and backend code.

Adapter checklist:

- Declare public model-specific options in `include/<model>_runtime.h`.
- Implement `runtime/<model>_runtime.c` with `*_runtime_ops()` and
  `*_runtime_register()` entrypoints.
- Probe readable GGUF files through `rt_gguf_metadata_read()` before using
  filename aliases for paths that do not exist yet.
- Translate `rt_backend` into the model's supported backend set and reject
  unsupported backends explicitly.
- Keep shape constants, tensor names, prompt quirks, KV layout, quantization,
  and kernel choices out of `runtime-core/`.
- Document the runtime's supported backends, options, and validation commands in
  `models/<model-family>/README.md`.
