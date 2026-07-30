# Laguna S 2.1 integration

This directory owns the Laguna S 2.1 model provider and its tailored GPU
inference implementation:

- `provider.c` exposes the whole-model lifecycle to the engine core.
- `graph.inc` owns Laguna graph state, KV caches, prefill, decode, and DFlash
  orchestration.
- `cuda/` contains CUDA host integration.
- `gpu/` contains kernels shared by CUDA and ROCm.
- `metal/` contains Metal host dispatch and Laguna/DFlash shaders.

Laguna currently requires a GPU graph backend. Distributed layer snapshots and
layer-slice execution are intentionally unsupported.
