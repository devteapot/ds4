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
Poolside's official `Laguna-S-2.1-NVFP4` checkpoint and its target-specific
`Laguna-S-2.1-DFlash-NVFP4` drafter.

# Motivation

The existing Laguna CUDA patch runs the official Q4_K_M GGUF. The official
NVFP4 release is instead a 15-shard compressed-tensors safetensors checkpoint:
its routed experts use packed E2M1 weights, E4M3 per-group scales, and
per-expert global scales, while the remaining model tensors are BF16. Loading
that representation directly avoids a conversion step and preserves its
calibrated W4A4 execution contract.

Poolside also publishes a separate 1B-parameter DFlash checkpoint trained for
this exact NVFP4 target. The target qualifier describes the training pairing;
the drafter weights themselves are BF16 and should reuse the existing Laguna
DFlash Tensor Core and Blackwell attention kernels.

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
- On Blackwell, evaluate each grouped-query prompt-attention head in its own
  warp while retaining one shared K/V load per group for both the 48-head
  target and 72-head DFlash drafter.
- Use the native W4A4 integer-dot kernel for decode and narrow DFlash verifier
  batches, where the only SM121 FP4 MMA shape would otherwise leave most of
  its result tile empty.
- Predecode each dynamically quantized E2M1 activation group once for the
  integer-dot path so every output row can consume signed bytes directly.
- For the fixed official checkpoint, verify that routed gate/up input-global
  scales are identical across all 256 experts in a layer, quantize each decode
  token once, and share that activation across its ten selected experts.
- On Blackwell, evaluate Laguna's at-most-16-token DFlash verifier blocks with
  a head-dimension-128 split-history attention kernel instead of serially
  walking the full context inside each query block.
- Accept the official NVFP4 DFlash safetensors directory through `--mtp`
  without converting it to GGUF, including its fused QKV tensors, BF16 norms,
  six target residual streams, and checkpoint-specific RoPE configuration.
- Preserve target-verified greedy semantics and default the NVFP4-specific
  drafter to Poolside's recommended seven proposals while retaining an
  explicit `--mtp-draft 15` comparison mode.
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
- Blackwell prompt attention preserves the portable grouped-query kernel's
  reduction order and causal online-softmax semantics.
- Native safetensors support is CUDA-only for this patch.
- The NVFP4 target remains W4A4/BF16 mixed precision; its paired DFlash
  checkpoint is BF16 and must not be mislabeled or executed as NVFP4 weights.

# Non-goals

- Converting the NVFP4 checkpoint to GGUF.
- A general-purpose safetensors or Transformers runtime.
- DFlash checkpoints trained for other Laguna targets or precisions.
- SSD streaming, distributed inference, Metal, or ROCm support for this
  checkpoint.
- Pre-Blackwell native NVFP4 execution.

# Assumptions

- The supported checkpoint is revision
  `07614121b31898586430f189d27a25a0be310843` of
  `poolside/Laguna-S-2.1-NVFP4`.
- The paired drafter is revision
  `723794750422b3efbf3a7b3af76dffb4ba035943` of
  `poolside/Laguna-S-2.1-DFlash-NVFP4`.
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
- Compare the Blackwell warp-per-head and portable grouped-query prefill
  attention schedules at the target and drafter's production head counts.
- Run the CUDA long-context and Laguna regression tests.
- Run deterministic full-model inference from the official directory on GB10.
- Inspect and load the official native DFlash directory as one BF16
  safetensors file, then run a deterministic target-verified generation.
- Compare raw decode with DFlash depths 7 and 15 on the same 2K/256 workload,
  recording proposal acceptance and verifier-block timing.

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

Keep native DFlash parsing equally narrow: detect Poolside's
`DFlashLagunaForCausalLM` config, expose the fused QKV payload as three
zero-copy tensor views, and synthesize the contiguous six-row auxiliary norm
view. Dispatch BF16 norms through the same mixed-precision helpers used by the
native target, including the batched target verifier's output norm.

Attention remains BF16 for native NVFP4, so source-patch attention
optimizations can be reused without changing quantization. Keep the portable
128-thread grouped-query kernel as the fallback. On Blackwell, use one warp
per query head, load each K/V row once into shared memory, and explicitly
reproduce the portable reduction tree before the warp shuffle reduction.

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
- The official checkpoint's gate/up input-global scale is identical across all
  256 experts in each layer. Enforce that exact invariant and quantize one
  activation per decode token rather than ten duplicate selected-expert rows.
  Down input-global scales remain expert-specific, so quantize each routed
  intermediate with its selected down projection's scale.
- Compile GB10 with the family-specific `compute_121f` to `sm_121` target,
  which is required for SM120-family block-scaled NVFP4 MMA.
- Group prompt routes into 16-route expert tiles, place routes in MMA's M
  dimension and eight output channels in N, and process down-projection output
  in 1024-row chunks using the existing routed-mid buffer for terms.
- Reuse the source CUDA patch's Blackwell warp-per-query-head prefill
  attention for both six-head target groups and nine-head DFlash groups. Gate
  it on production-size models and retain
  `DS4_CUDA_LAGUNA_NO_WARP_GQA_PREFILL=1` as the direct portable comparison.
- Keep the 16-lane W4A4 integer-dot schedule for decode because SM121 offers
  no GEMV-sized NVFP4 MMA instruction and the direct one-column MMA path
  benchmarks slower.
- Store dynamic activations as packed E2M1 only for grouped prompt MMA. For
  decode and DFlash verification, expand E2M1 to signed bytes during
  quantization and remove repeated activation-nibble decode from every gate,
  up, and down dot product.
- Consume each signed activation group once for both routed gate and up DP4A
  accumulators.
- Borrow split-K online-softmax reduction from the major engines only for
  Laguna's fixed Blackwell verifier shape: 128-wide heads, a long cached
  history, and at most 16 staged tokens. Use 16 warps per query/head and retain
  the existing grouped-GQA kernel for large prompt prefill.
- Keep narrow decode on DP4A rather than copying a generic grouped FP4 GEMM:
  with 256 experts and top-10 routing, a DFlash block has too few repeated
  routes per expert to occupy SM121's `m16n8k64` tile efficiently.
- Preserve source-offset alignment when merged safetensors spans are copied
  into CUDA's range cache so 32-bit packed/scale fragment loads remain aligned.
- Add `download_model.sh laguna-nvfp4`; it uses the official Hugging Face
  client to download the complete pinned checkpoint directory.
- Add `download_model.sh laguna-nvfp4-dflash` for the pinned paired drafter.
  Load its unsharded safetensors file directly and use its 262K-context,
  theta-10000 sliding-attention RoPE rather than the Q4 drafter's 1M-context,
  theta-500000 configuration.
- Keep the existing Blackwell DFlash grouped-GQA kernel and BF16 cuBLAS path;
  the new work is checkpoint-native loading and mixed-precision correctness,
  not an invented FP4 drafter kernel for weights that are actually BF16.

# Validation

- `make -j2 cpu`: passes.
- `make cuda-spark`: passes with the family-specific `compute_121f`/`sm_121`
  target required by block-scaled MMA.
- The Laguna CUDA regression passes on GB10, including a 256-token,
  256-expert, top-10 native W4A4 test at the production
  3072→1024→3072 dimensions, production 72-head/8-KV-head DFlash attention,
  and the existing Q4_K/Q6_K coverage.
- `./ds4 --inspect --cuda -m /srv/models/poolside/Laguna-S-2.1-NVFP4`:
  reports 15 shards, 66.98 GiB, 626 BF16 logical tensors, and 141 NVFP4
  logical tensors.
- Native tokenizer smoke: `Hello` maps to token 6352.
- Deterministic live prompt `Reply with OK.` produces `OK` directly from the
  official sharded checkpoint through all 48 layers.
- CUDA startup residency covers 66.96 GiB of physical dense, packed-weight,
  and block-scale tensor ranges before inference timing.
- On the same 2,048-token `ds4.c` pure-prefill workload, warp-per-head
  Blackwell attention measures 578.52 token/s versus 415.27 token/s with
  `DS4_CUDA_LAGUNA_NO_WARP_GQA_PREFILL=1`, a 39.3% improvement. The paired
  result is stored in
  `speed-bench/laguna_s21_nvfp4_prefill_attention_gb10.csv`.
- The grouped FP4 MMA benchmark at 2K/4K/8K reports prompt throughput of
  413.38/317.80/224.72 token/s, compared with 95.05/87.59/78.13 for the prior
  integer-dot prompt kernel, and steady decode of 14.26/14.10/13.55 token/s.
  The raw run is stored in
  `speed-bench/laguna_s21_nvfp4_gb10.csv`.
- The official native drafter maps as one 2.08 GiB file and 76 logical BF16
  tensors. A deterministic `Reply with OK.` smoke produces exactly `OK`
  through target verification.
- On the repeated 2,048-token `ds4.c` / 256-token greedy workload, raw NVFP4
  measures 446.58/14.74 prefill/generation tok/s. DFlash depth 5 measures
  382.05/28.15 tok/s with 64.78% proposal acceptance and 4.20 committed
  tokens per verifier block; depth 7 measures 366.70/31.73 tok/s with 64.31%
  proposal acceptance and 5.45 committed tokens per 171.66 ms block. Results
  are stored in `speed-bench/laguna_s21_nvfp4_dflash_gb10.csv`.
- On an immutable common 2,048-token prompt, Q4_K_M with its official BF16
  DFlash drafter at depth 15 measures 27.81 token/s (47.06% proposal
  acceptance, 8.00 committed tokens per 287.38 ms block). Native NVFP4 with
  its checkpoint-specific official drafter at depth 7 measures 31.73 token/s
  (64.31% acceptance, 5.45 committed tokens per 171.66 ms block). The exact
  comparison is stored in
  `speed-bench/laguna_s21_dflash_quant_comparison_gb10.csv`.
- Before split-history verifier attention, depth 7 measured 21.45 token/s and
  248.14 ms per block on this workload. The fixed-shape Blackwell kernel
  improves generation by 47.9% and reduces block latency by 30.8%.
- A depth-5 control without split-history attention preserved the exact
  201/272 acceptance result and measured 20.13 token/s. Enabling the split
  schedule reduced synchronized block time from 231.26 to 149.04 ms and
  measured 28.15 token/s.
- `make cuda-regression` passes the long-context and Laguna suites, including
  F32/BF16 DFlash auxiliary feature-pack equivalence and serial-versus-split
  verifier attention over a 300-token cached history.
- `make test` passes CUDA long-context, tensor-equivalence, kernel, unit, and
  server checks, but exits nonzero on 28 pre-existing model-output fixture
  assertions (tool formatting and stale official/golden vectors for the
  configured Q4 model); those paths do not dispatch native NVFP4 or DFlash
  verifier attention.
- Compute Sanitizer memcheck reports zero errors with expected unsupported
  host-registration API diagnostics suppressed.

# Provenance

Explicitly requested:

- Add official Laguna S 2.1 NVFP4 support on top of the existing CUDA patch.
- Load and execute NVFP4 natively rather than converting it to GGUF.
- Optimize specifically for NVFP4 on CUDA and optionally further for
  Blackwell/GB10.
- Use existing Metal and CUDA kernels as implementation references where
  appropriate.
- Initially support only the raw model, then add the official target-specific
  DFlash checkpoint after rebasing the dependent patch through PatchMD.
- Use the repository download script and open a PR targeting the original
  Laguna CUDA patch branch.
- Reapply the CUDA and NVFP4 patch stack through PatchMD after the source
  branch's DFlash commits, adapt dependent patch references where necessary,
  benchmark the paired official drafter, and update the PR.
- Reuse applicable prefill improvements from the updated source CUDA patch in
  the NVFP4 realization and update its PatchMD record.

Observed from the official checkpoint:

- Safetensors metadata describes 71,898,733,760 bytes across 15 shards.
- A routed gate/up projection is U8 `[1024, 1536]` with F8_E4M3 scales
  `[1024, 192]`; a routed down projection is U8 `[3072, 512]` with scales
  `[3072, 64]`.
- Each projection has scalar F32 weight-global and input-global scales.
- The router is BF16 `[256, 3072]` and its correction bias is F32 `[256]`.

# Reference realization

The reference realization is an incremental delta on top of the
`laguna-s21-cuda` dependency realization and is stored in `reference.patch`.
