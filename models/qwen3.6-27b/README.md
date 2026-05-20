# Qwen3.6-27B Runtime

This directory tracks the start of a Qwen3.6-27B model family runtime for
`Qwen/Qwen3.6-27B`.

Status: loader-bound tokenizer engine with a first CPU reference path.  The adapter registers the
`qwen3.6-27b` family with `runtime-core`, exposes the model's shape constants,
validates readable GGUF files through shared metadata helpers, opens compatible
GGUFs through the shared read-only tensor directory layer, loads
`tokenizer.ggml.tokens` and `tokenizer.ggml.merges`, applies the Qwen2-style
pre-tokenizer shape used by the upstream tokenizer config, renders the Qwen
chat template into token IDs, and binds the required Qwen text tensors with
strict shape validation. Tokenization and chat rendering fail closed when a loaded
tokenizer cannot represent a byte piece, so malformed/incomplete GGUF vocabularies
do not silently drop prompt content. When `mmproj_path` is provided, it also validates Qwen VL
projector metadata and required vision/merger tensors before accepting the
engine. It can create sessions for prompt sync, position tracking, and versioned
payload save/load. Version 3 payloads include Qwen state-layout byte counts and
a typed section table for token IDs, logits, last target hidden state,
full-attention KV, Gated DeltaNet, and convolution state sections. State
sections are loaded lazily and re-saved when present; version 1 and 2 payloads
remain readable. Session sync now distinguishes reusable graph state from token
IDs alone, so rewound or legacy logits-only checkpoints are recomputed instead
of being treated as valid continuation frontiers.
Loaded logits can already drive argmax, sampling, top-logprobs, token logprob,
and raw-logit readback. Token evaluation now has a one-token CPU reference path
for plain `f32`, `f16`, and `bf16` GGUF tensors plus slow `q4_0`, `q4_1`,
`q5_0`, `q5_1`, `q8_0`, `q8_1`, `q2_K`, `q3_K`, `q4_K`, `q5_K`, `q6_K`, and
`q8_K` dequantization, plus `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_nl`, and `iq4_xs`. It runs the Qwen hybrid text stack with lazy full-attention KV,
Gated DeltaNet recurrent state, and convolution state allocation, so
zero/synthetic fixtures do not allocate the full state budget. Official-vector
parity and full backend graph scheduling still remain open, but the test suite
now has a Qwen full-logit vector runner and
capture script under `tests/qwen36-vectors/` for official HF parity fixtures.
An early Metal backend now opens explicitly on macOS, compiles Qwen
matvec/RMSNorm/SiLU-mul/L2Norm/Gated DeltaNet/full-attention-head kernels, runs
a tiny self-test, and offloads supported `f32`/`f16`/`bf16`, scalar-block
`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`, and K-quant `q2_K`,
`q3_K`, `q4_K`, `q5_K`, `q6_K`, `q8_K`, `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_nl`, and `iq4_xs` 2D matvecs, RMSNorms, FFN
SiLU-mul activations, Gated DeltaNet L2Norms, per-head recurrent updates, and
per-head full-attention reductions with CPU fallback for unsupported tensor
types. Runtime tests now also exercise a nonzero Qwen text recurrent fixture
through an explicit Metal engine and assert CPU-comparable hidden state plus
real matvec/RMSNorm/SiLU/L2/Gated Delta/full-attention helper calls with no
fallbacks. They also exercise the Qwen mmproj path through an explicit Metal
engine and assert real matvec/RMSNorm/SiLU helper calls. The root `ds4-server`
can route OpenAI-compatible chat/completion requests and
Responses/Anthropic requests through this runtime when launched with a
compatible Qwen GGUF, including image/video content blocks when mmproj support
is loaded. That server path uses the CPU reference session today and can
persist/reload Qwen prompt checkpoints and generated completion prefixes with
`--kv-disk-dir`, including across fresh server/session reloads. It can also
render OpenAI, Responses, and Anthropic tool
schemas into Qwen's XML tool prompt and map generated XML tool calls back to
structured API tool calls. Live Responses and Anthropic tool-result
continuations reuse the Qwen runtime-core session when the returned tool IDs
still match the worker-owned frontier. Runtime-core disk checkpoints also carry
the optional tool-id map for exact Qwen XML tool-call replay after a cache
restore, and can key Qwen Responses-visible transcripts to hidden runtime
payloads for restart recovery. The runtime-core server now preserves image/video
URLs and decoded base64/data-URL bytes on requests. File URLs and, on macOS,
HTTP(S) media URLs are resolved to bytes before decode and media-exact disk-key
hashing. Media-bearing prompts bypass text-prefix KV reuse, but same-media
prompt and generated-continuation states can be stored and reloaded with exact
disk keys built from rendered prompt, token, and media payload digests.
ImageIO-supported image payloads on macOS, PNG image payloads in portable
builds, Netpbm image payloads, single or concatenated Netpbm video-frame
payloads, and bounded high-frame AVFoundation-decoded frames from video
containers on macOS can be decoded,
resized with Qwen-style patch-factor pixel budgets, bicubic-sampled and
CLIP-normalized into aspect-preserving Qwen patch grids. The server honors
per-media `min_pixels`/`max_pixels` controls for smart resize, aligns explicit
`resized_height`/`resized_width` requests through Qwen's patch-factor resize
path, includes those controls in media-exact disk keys, and lets video
`nframes`/`min_frames`/`max_frames` plus `fps`/`video_start`/`video_end`
controls drive bounded frame-factor-aligned sampling up to 256 frames. Patch grids are projected through the
CPU mmproj path, expanded into the matching number of image/video pad tokens,
and fed into those positions with embedding sync. Local Responses objects now
support
`previous_response_id` and local `conversation` IDs by persisting visible
response text and tool-call IDs beside the KV checkpoint directory, and
`/v1/conversations` supports local metadata create/get/update/delete plus
text-only item create/list/retrieve/delete. Provider-hosted remote
object/conversation sync and media-bearing conversation item storage remain
gated. Batched MTP
acceleration, full Qwen Metal graph scheduling, CUDA kernels, production GPU
backends, full upstream video
metadata return, and provider-level video sampling parity remain
gated. The model layer can
project supplied image patch grids through the loaded mmproj as a CPU reference
path and can evaluate prompts with caller-provided hidden embeddings replacing
selected token embeddings. When MTP layers are loaded, Qwen exposes a runtime-core
speculative argmax-span hook backed by a bounded CPU nextn draft suffix and
exact target-model verification, so the server can exercise the verifier
surface without changing output semantics. The Qwen MTP margin skips
low-confidence suffixes back to exact greedy decode. Runtime regressions now
cover accepted, rejected, and margin-skipped draft paths; the rejected path
proves the session remains on the target frontier and can continue decoding the
verifier-selected token.
They also exercise Qwen GGUFs with embedded `nextn_predict_layers`, consecutive
external nextn layers up to the configured draft span, and the duplicate-MTP
guard when a separate MTP GGUF is supplied on top.

Supported backends today are `auto` and `cpu`, both selecting the CPU reference
path, plus explicit `metal` on macOS when a Metal device is available. The
Metal path currently accelerates supported 2D matvecs, RMSNorms, SiLU-mul
activations, Gated DeltaNet L2Norms, per-head recurrent updates, and per-head
full-attention reductions, and otherwise falls back to the CPU
reference helpers; Metal runtime regressions cover text recurrent,
full-attention, and mmproj helper offload counters, while full graph scheduling
is still pending. `cuda`,
`rocm`, and `mlx` fail engine open until Qwen-specific graph scheduling and
kernels are implemented.
Context-memory estimates already include Qwen's full-attention KV rows,
Gated DeltaNet recurrent state, convolution history, token state, and logits
scratch space so backend planning sees the real state budget.
The shared runtime core now provides CPU-reference scalar reads for plain
`f32`, `f16`, and `bf16` GGUF tensors plus slow reference dequantization for
`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`, `q2_K`, `q3_K`, `q4_K`,
`q5_K`, `q6_K`, `q8_K`, `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_nl`, and `iq4_xs`; Qwen can use that for norms, biases, and early
quantized projection checks while optimized model-specific matvecs are built.
Plain `f32`/`f16`/`bf16` matvecs also recognize sparse-hole GGUF test tensors,
which keeps the full-shape Qwen verifier fixtures cheap without changing dense
model behavior.
Text-state regressions now also cover a longer synthetic prompt that allocates
full-attention KV state, writes it through the v3 session payload, reloads it
into a fresh session, and continues decoding from the restored frontier with a
zero-embedding token that must still receive signal from the restored KV.
They also cover a synthetic Gated DeltaNet recurrent/convolution state payload
that is saved, reloaded, and continued with current q/k/v disabled so restored
state must drive the next hidden value, and the same nonzero state is stored
through the server disk cache, reopened in a fresh session, and continued.
Text-only engines reject rendered prompts containing Qwen vision placeholders.
When a validated mmproj file is loaded and the tokenizer contains the expected
vision special tokens, Qwen chat rendering accepts image/video placeholder
content and emits those special token IDs. The server path translates OpenAI
chat, Responses, and Anthropic image/video content blocks into those placeholders
when mmproj support is available, preserving media URLs and decoded base64/data
URL bytes. File URLs and, on macOS, HTTP(S) media URLs are resolved to bytes.
The server currently decodes ImageIO-supported image bytes on macOS and PNG or
Netpbm P2/P3/P5/P6 image bytes in portable builds into
Qwen-style smart-resized, bicubic-sampled, CLIP-normalized patch grids for
`qwen36_runtime_embed_image_patches_f32`, including per-media pixel-budget and
explicit resize controls, and accepts the same Netpbm payloads for video blocks
as single or concatenated frame sequences plus bounded high-frame
AVFoundation-decoded frames from video containers on macOS, then expands the
rendered image/video pad token to the projected token count and uses
`qwen36_runtime_session_sync_embeddings` to feed the projected hidden vectors
while preserving token history. Full upstream video metadata return and exact
provider-level video sampling parity remain gated. The runtime can
bind and validate MTP/nextn support layers from embedded
or separate GGUF files, bind consecutive external nextn layers up to the
configured draft span, build a bounded CPU draft-token suffix from those layers,
skip low-confidence suffixes through the Qwen MTP margin, and expose
per-session proposed, accepted, rejected, and skipped draft counters for
tests/diagnostics. The text-only MTP verifier regression covers accepted
drafts from embedded layers, separate nextn layers, and a multi-token external
draft span, margin-gated skips, EOS-limited drafts, missing/bad MTP files,
duplicate embedded-plus-external MTP configuration, and a forced rejection that
leaves only accepted target tokens in the session history. It also saves and
reloads an MTP-advanced payload in the same session and verifies that draft
scratch/counters are not persisted across restore.
Runtime-core server tests now route a deterministic completion through
`generate_job_rt` and assert both MTP verifier rejection and acceptance counters
from the worker-owned session; the text gate separately asserts a margin-gated
skip and the skipped counter.
Batched target verification and production MTP
acceleration are still gated. Payloads now persist the last target hidden state
needed by that verifier.

The official model is not a DS4-shaped graph.  It is a dense 27B multimodal
checkpoint with a hybrid text stack: 64 text layers, hidden size 5120,
intermediate size 17408, padded vocabulary 248320, native 262144-token context,
and a repeated schedule of Gated DeltaNet blocks with periodic full attention.
That means the runtime needs its own tensor binder, prompt rules, linear-state
cache, full-attention KV layout, MTP handling, and Metal/CUDA kernels.

References:

- [Qwen/Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen/Qwen3.6-27B config.json](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json)
- [Qwen/Qwen3.6-27B chat_template.jinja](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/chat_template.jinja)

Layout:

- `include/qwen36_runtime.h`: runtime options, model spec constants, and
  `rt_model_ops` registration entrypoints.
- `runtime/qwen36_runtime.c`: current registry/probe/tokenizer/chat/tensor
  binding, CPU reference eval, sampling, and session-payload adapter.
- `backends/metal/qwen36_metal.m`: early Qwen Metal dense/scalar-block/K-quant
  matvec/RMSNorm/SiLU-mul/L2Norm/Gated DeltaNet/full-attention-head backend and
  kernel self-test.
- `engine/`: future split-out graph/session logic and official-vector harness.
- `backends/cuda/`: future model-specific kernels.
- `tests/`: model-family regression fixtures; Qwen official full-logit capture
  lives at `tests/qwen36-vectors/`.

Initial implementation order:

1. Fill and version official-vector fixtures for the CPU reference path once the
   exact upstream checkpoint is pinned. The harness can now fail CI explicitly
   with `QWEN36_REQUIRE_OFFICIAL_VECTORS=1` or
   `make qwen36-text-gates-strict`; strict mode requires the fixture preamble to
   name `Qwen/Qwen3.6-27B`, a non-default HF revision, vocab/EOS provenance,
   and dense full-logit sections for vocabulary-wide comparison after the
   fixture metadata is checked against the opened GGUF.
   Use `make qwen36-gates` for the broad local Qwen gate, including current
   tokenizer/chat/text runtime coverage, synthetic mmproj/media prompt
   expansion, and media-exact checkpoint coverage. Use
   `make qwen36-text-gates` for the text milestone, including Qwen2-style
   pre-tokenization regressions, last-query assistant thinking preservation for
   tool replay, server text API, OpenAI/Responses/Anthropic SSE streaming,
   checkpoint, MTP, and vector-harness pass; it sets
   `QWEN36_TEST_TEXT_ONLY=1` so mmproj/media regressions stay out of that
   milestone. Use the matching `*-cpu` targets to run through a `DS4_NO_GPU`
   test binary for CPU-reference portability.
2. Add Metal/CUDA graph paths for Gated DeltaNet, dense FFN, and periodic full
   attention.
3. Broaden real long-context state behavior tests. Current Qwen-specific
   regressions cover thinking/no-thinking prompts, assistant reasoning stripping
   versus replay preservation, developer/function role normalization, exact
   prefix reuse, rewind invalidation, legacy logits-only recompute,
   divergent-prefix rebuilds, and synthetic full-attention KV payload
   save/load/continue behavior that proves restored KV influences the next
   hidden state, plus recurrent/conv payload and server disk-cache
   save/load/continue behavior.
