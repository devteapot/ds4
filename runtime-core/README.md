# Runtime Core

This directory marks the intended shared layer for the model-specific runtime
framework.

Nothing production-critical has been moved here yet.  The current CLI, server,
benchmark, eval, sampling, and disk-KV code still live at the repository root
because they are DS4-shaped in a few important places.  They should move into
this layer only when a second model runtime proves which APIs are genuinely
shared.

Good candidates for future extraction:

- process entrypoints and option parsing shared by model runtimes
- OpenAI/Anthropic-compatible server plumbing
- sampling utilities and logprob reporting
- disk-KV cache policy and file management
- benchmark and regression harnesses
- GGUF metadata helpers that are not model-specific

Model architecture, tensor binding, graph scheduling, quantization choices, and
backend kernels should stay under `models/<model-family>/`.
