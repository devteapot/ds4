---
format: patchmd/v0.1
id: laguna-s21-nvfp4-cuda
status: ready
kind: persistent-customization
upstream:
    base: 7e3dbef7e336433f487c172a3308e26b39fa75a3
relationships:
    depends_on:
        - laguna-s21-cuda
    conflicts_with: []
authors:
    - name: Unknown
license: MIT
---
# Intent

Extend the Laguna S 2.1 CUDA backend with native loading and execution of
Poolside's official `Laguna-S-2.1-NVFP4` checkpoint.

# Motivation

The existing Laguna CUDA patch runs the official Q4_K_M GGUF. The official
NVFP4 release is instead a 15-shard compressed-tensors safetensors checkpoint:
its routed experts use packed E2M1 weights, E4M3 per-group scales, and
per-expert global scales, while the remaining model tensors are BF16. Loading
that representation directly avoids a conversion step and preserves its
calibrated W4A4 execution contract.

# Required behavior

- Accept the official checkpoint directory through `-m` without converting or
  repacking it to GGUF.
- Parse all safetensors shards and synthesize the logical Laguna tensor groups
  needed by the existing graph while retaining direct offsets to every
  expert's packed weights, block scales, weight-global scale, and input-global
  scale.
- Load the checkpoint's tokenizer JSON directly and preserve Laguna chat,
  reasoning, and tool-call behavior.
- Execute BF16 embeddings, norms, attention projections, dense FFN, shared
  experts, and output head on CUDA.
- Execute routed experts as W4A4: dynamically quantize selected activations to
  adjacent-nibble E2M1 in groups of 16, store E4M3 block scales, and apply both
  activation and weight global reciprocals in fused gate/up and down kernels.
- Require a Blackwell family-specific CUDA build for the official native
  checkpoint; direct users of older cards to the official Q4_K_M model.
- Use GB10's block-scaled NVFP4 MMA for Laguna prompt batches, grouping routes
  by expert so each `m16n8k64` result tile is fully occupied and its weight
  fragments are reused across up to 16 routes.
- Use the native W4A4 integer-dot kernel for decode, where the only SM121 FP4
  MMA shape would otherwise discard seven of eight result columns.
- Preserve existing Laguna Q4_K_M and unrelated backend behavior.

# Invariants

- The shared Laguna graph, routing, attention, KV cache, sampling, CLI, agent,
  and server behavior remain the source of model semantics.
- Only routed expert gate/up/down projections use NVFP4. Attention, layer 0's
  dense FFN, routers, shared experts, embeddings, and output remain BF16.
- Safetensors data stays file-backed and read-only; logical routed tensors do
  not pretend their per-expert shard ranges are physically contiguous.
- Correct scale conventions and stable logits are preserved across both the
  grouped FP4 MMA prefill kernel and the decode-specialized W4A4 kernel.
- Native safetensors support is CUDA-only for this patch.

# Non-goals

- Laguna DFlash or any speculative drafter.
- Converting the NVFP4 checkpoint to GGUF.
- A general-purpose safetensors or Transformers runtime.
- SSD streaming, distributed inference, Metal, or ROCm support for this
  checkpoint.
- Pre-Blackwell native NVFP4 execution.

# Assumptions

- The supported checkpoint is revision
  `07614121b31898586430f189d27a25a0be310843` of
  `poolside/Laguna-S-2.1-NVFP4`.
- Gate and up input-global scales match for a given expert, allowing one
  quantized activation to feed both projections. Loader preparation rejects a
  checkpoint that violates this.
- The attached CUDA host is a GB10 (`sm_121`) with CUDA 13 and is the primary
  Blackwell validation target.

# Open questions

None.

# Acceptance

- Build both CUDA `sm_121` and CPU targets without warnings introduced by this
  patch.
- Inspect the native directory and report its 15 shards, BF16 tensors, and
  logical NVFP4 expert groups.
- Load `tokenizer.json` and reproduce a known simple tokenization.
- Add synthetic CUDA coverage for both the pre-existing interleaved NVFP4 test
  representation and the official split/adjacent native representation.
- Validate both grouped prompt-batch FP4 MMA and decode-specialized W4A4
  routed-expert schedules at Laguna's production dimensions.
- Run the CUDA long-context and Laguna regression tests.
- Run deterministic full-model inference from the official directory on GB10.

# Adaptation guidance

Keep native-file parsing narrow and Laguna-specific. Map shards into one
reserved virtual range so existing model-offset APIs remain usable for dense
tensors, but carry routed expert physical offsets through
`ds4_gpu_nvfp4_matrix_desc`. Skip the synthetic logical native groups during
contiguous startup span preparation because their underlying shard ranges are
not contiguous. Add their physical packed-weight and block-scale ranges
instead, so CUDA resolves and caches the real tensors before first inference
and benchmark prefill does not pay a one-time expert upload.

The packed compressed-tensors layout is adjacent: each byte's low nibble is
the even element and its high nibble is the odd element. Both weights and
activations use E2M1 groups of 16. E4M3 block scales are serialized after
multiplication by their global scale, so dequantization applies the reciprocal
stored in the CUDA descriptor cache.

# Decisions

- Detect a model directory before GGUF parsing and initialize the fixed Laguna
  shape before filtering shard tensor names.
- Map all shards page-aligned into a reserved read-only address range and keep
  their file descriptors until model close.
- Expose native routed projections as logical type 40 tensors whose descriptors
  retain arrays of per-expert physical offsets.
- Parse the tokenizer's object vocab and pair-array merges instead of deriving
  tokenizer metadata from GGUF.
- Add BF16 CUDA matmul, embedding, RMSNorm, QKVG, and head-norm/RoPE entry
  points used by the native mixed-precision graph.
- Quantize one activation per selected expert because input-global scales are
  expert-specific. Reuse that activation for gate and up, then quantize the
  routed intermediate with the selected down projection's scale.
- Compile GB10 with the family-specific `compute_121f` to `sm_121` target,
  which is required for SM120-family block-scaled NVFP4 MMA.
- Group prompt routes into 16-route expert tiles, place routes in MMA's M
  dimension and eight output channels in N, and process down-projection output
  in 1024-row chunks using the existing routed-mid buffer for terms.
- Keep the 16-lane W4A4 integer-dot schedule for decode because SM121 offers
  no GEMV-sized NVFP4 MMA instruction and the direct one-column MMA path
  benchmarks slower.
- Preserve source-offset alignment when merged safetensors spans are copied
  into CUDA's range cache so 32-bit packed/scale fragment loads remain aligned.
- Add `download_model.sh laguna-nvfp4`; it uses the official Hugging Face
  client to download the complete pinned checkpoint directory.

# Validation

- `make -j2 cpu`: passes.
- `make cuda-spark`: passes with the family-specific `compute_121f`/`sm_121`
  target required by block-scaled MMA.
- The Laguna CUDA regression passes on GB10, including a 256-token,
  256-expert, top-10 native W4A4 test at the production
  3072→1024→3072 dimensions and the existing Q4_K/Q6_K coverage.
- `./ds4 --inspect --cuda -m /srv/models/poolside/Laguna-S-2.1-NVFP4`:
  reports 15 shards, 66.98 GiB, 626 BF16 logical tensors, and 141 NVFP4
  logical tensors.
- Native tokenizer smoke: `Hello` maps to token 6352.
- Deterministic live prompt `Reply with OK.` produces `OK` directly from the
  official sharded checkpoint through all 48 layers.
- CUDA startup residency covers 66.96 GiB of physical dense, packed-weight,
  and block-scale tensor ranges before inference timing.
- The grouped FP4 MMA benchmark at 2K/4K/8K reports prompt throughput of
  413.38/317.80/224.72 token/s, compared with 95.05/87.59/78.13 for the prior
  integer-dot prompt kernel, and steady decode of 14.26/14.10/13.55 token/s.
  The raw run is stored in
  `speed-bench/laguna_s21_nvfp4_gb10.csv`.

# Provenance

Explicitly requested:

- Add official Laguna S 2.1 NVFP4 support on top of the existing CUDA patch.
- Load and execute NVFP4 natively rather than converting it to GGUF.
- Optimize specifically for NVFP4 on CUDA and optionally further for
  Blackwell/GB10.
- Use existing Metal and CUDA kernels as implementation references where
  appropriate.
- Support only the raw model, without DFlash.
- Use the repository download script and open a PR targeting the original
  Laguna CUDA patch branch.

Observed from the official checkpoint:

- Safetensors metadata describes 71,898,733,760 bytes across 15 shards.
- A routed gate/up projection is U8 `[1024, 1536]` with F8_E4M3 scales
  `[1024, 192]`; a routed down projection is U8 `[3072, 512]` with scales
  `[3072, 64]`.
- Each projection has scalar F32 weight-global and input-global scales.
- The router is BF16 `[256, 3072]` and its correction bias is F32 `[256]`.

# Reference realization

The reference realization is based on upstream
7e3dbef7e336433f487c172a3308e26b39fa75a3 and stored in `reference.patch`.
