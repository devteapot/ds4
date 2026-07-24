---
format: patchmd-set/v0.1
id: laguna-s21-nvfp4-stack
status: ready
patches:
    - laguna-s21-cuda
    - laguna-s21-nvfp4-cuda
---
# Intent

Compose Laguna S 2.1 CUDA/DFlash support with native Blackwell NVFP4 execution.

# Combined acceptance

- CPU-only, generic CUDA, and Blackwell CUDA builds continue to compile.
- Laguna S 2.1 Q4_K_M raw inference and DFlash remain available through the
  CUDA backend.
- The official Laguna S 2.1 NVFP4 checkpoint loads natively without a GGUF
  conversion and uses Blackwell-specific FP4 kernels.
- The NVFP4 realization composes cleanly on top of the CUDA/DFlash patch
  without weakening either patch's existing regression coverage.
- Live inference and benchmark evidence cover raw and DFlash execution for the
  native NVFP4 checkpoint on Blackwell.
