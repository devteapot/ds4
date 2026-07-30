/* These kernels are shared verbatim with HIP: both backends use CUDA launch
 * syntax and the header selects the appropriate warp shuffle intrinsic. */
#include "../gpu/laguna.cuh"
#include "../gpu/dflash.cuh"
