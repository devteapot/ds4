# Mistral Medium 3.5 128B Runtime

This directory tracks the start of a Mistral Medium 3.5 model family runtime
for `mistralai/Mistral-Medium-3.5-128B`.

Status: scaffold only.  The adapter registers the `mistral-medium-3.5-128b`
family with `runtime-core`, exposes the model's shape constants, and probes
obvious Mistral Medium 3.5 128B filenames.  It intentionally refuses to open an
engine until the Mistral-specific loader, prompt renderer, graph schedule, and
backend kernels exist.

The official checkpoint is not a DS4-shaped graph.  It is a dense 128B
multimodal checkpoint with a 256k-token text context, 88 text decoder layers,
hidden size 12288, intermediate size 28672, grouped-query attention with 96
query heads and 8 KV heads, 131072 vocabulary entries, YaRN-scaled RoPE from an
original 4096-token context, FP8 text weights, and a Pixtral-style vision
encoder.  Reasoning effort is request-configurable through the official API and
serving stacks.

References:

- [Mistral Medium 3.5 model card](https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04)
- [mistralai/Mistral-Medium-3.5-128B model card](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B)
- [mistralai/Mistral-Medium-3.5-128B config.json](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/main/config.json)
- [mistralai/Mistral-Medium-3.5-128B params.json](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/main/params.json)

Layout:

- `include/mistral35_runtime.h`: runtime options, model spec constants, and
  `rt_model_ops` registration entrypoints.
- `runtime/mistral35_runtime.c`: current registry/probe adapter.
- `engine/`: future loader, tokenizer/chat rendering, tensor binder, CPU
  reference path, graph/session logic, sampling, and payload serialization.
- `backends/metal/`, `backends/cuda/`, `backends/rocm/`, and `backends/mlx/`:
  future model-specific kernels where each backend proves useful.
- `tests/`: future official-vector and model-family regression fixtures.

Initial implementation order:

1. Add shared GGUF metadata helpers and make probing validate architecture
   metadata instead of filenames.
2. Land a text-only loader, tokenizer, and Mistral chat-template path with
   `reasoning_effort` support.
3. Implement a short CPU reference pass for one-token logits and official-vector
   capture.
4. Add Metal/CUDA graph paths for dense attention, grouped KV cache, RMSNorm,
   SwiGLU MLP, YaRN RoPE, and FP8/dequant handling.
5. Add Mistral-specific runtime tests, including none/high reasoning chat
   rendering, function-call rendering, and long-context KV behavior.
