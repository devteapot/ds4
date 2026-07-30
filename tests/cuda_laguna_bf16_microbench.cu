#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {

constexpr int kDefaultIn = 3072;
constexpr int kMaxIn = 12288;
constexpr int kMaxOut = 100352;
constexpr int kMaxTokens = 8;
constexpr size_t kMaxWeightValues =
    static_cast<size_t>(kMaxOut) * kDefaultIn;
#ifndef LAGUNA_BF16_LT_WORKSPACE_BYTES
#define LAGUNA_BF16_LT_WORKSPACE_BYTES 0u
#endif
constexpr size_t kLtWorkspaceBytes = LAGUNA_BF16_LT_WORKSPACE_BYTES;
static size_t gWeightOffsetBytes = 0;
static uint32_t gMinAlignmentABytes = 256;

struct Shape {
    const char *name;
    int in;
    int out;
};

constexpr Shape kShapes[] = {
    {"k3072_out1024", 3072, 1024},
    {"k3072_out6144", 3072, 6144},
    {"k3072_out9216", 3072, 9216},
    {"k3072_out12288", 3072, 12288},
    {"k3072_out100352", 3072, 100352},
    {"k3072_out256", 3072, 256},
    {"k3072_out48", 3072, 48},
    {"k3072_out72", 3072, 72},
    {"k6144_out3072", 6144, 3072},
    {"k9216_out3072", 9216, 3072},
    {"k1024_out3072", 1024, 3072},
    {"k12288_out3072", 12288, 3072},
};

static bool cuda_ok(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "FAIL CUDA %s: %s\n",
                 what, cudaGetErrorString(status));
    return false;
}

static bool cublas_ok(cublasStatus_t status, const char *what) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "FAIL cuBLAS %s: status=%d\n",
                 what, static_cast<int>(status));
    return false;
}

static bool cublaslt_ok(cublasStatus_t status, const char *what) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "FAIL cuBLASLt %s: status=%d\n",
                 what, static_cast<int>(status));
    return false;
}

__global__ void init_bf16_kernel(__nv_bfloat16 *dst, uint64_t count) {
    const uint64_t i =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t h = static_cast<uint32_t>(i) * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    const int32_t centered = static_cast<int32_t>(h & 1023u) - 512;
    dst[i] = __float2bfloat16_rn(static_cast<float>(centered) / 4096.0f);
}

__global__ void init_f32_kernel(float *dst, uint64_t count) {
    const uint64_t i =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t h = static_cast<uint32_t>(i) * 1664525u + 1013904223u;
    h ^= h >> 16u;
    const int32_t centered = static_cast<int32_t>(h & 2047u) - 1024;
    dst[i] = static_cast<float>(centered) / 2048.0f;
}

__global__ void f32_to_bf16_bench_kernel(
        __nv_bfloat16 *out, const float *in, uint64_t count) {
    const uint64_t i =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2bfloat16(in[i]);
}

template <int RowsPerWarp, int Tokens>
__global__ void bf16_full_warp_gemv_kernel(
        float *out,
        const __nv_bfloat16 *weights,
        const __nv_bfloat16 *x,
        int in_dim,
        int out_dim) {
    static_assert(RowsPerWarp >= 1 && RowsPerWarp <= 8,
                  "unsupported rows per warp");
    static_assert(Tokens == 1 || Tokens == 8, "unsupported token count");
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int warp =
        static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int row0 = warp * RowsPerWarp;
    if (row0 >= out_dim) return;

    float accum[RowsPerWarp][Tokens] = {};
    const int kPairs = in_dim / 2;
    const __nv_bfloat162 *x2 =
        reinterpret_cast<const __nv_bfloat162 *>(x);
#pragma unroll 1
    for (int pair0 = lane; pair0 < kPairs; pair0 += 32) {
        float2 xv[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; token++) {
            xv[token] = __bfloat1622float2(
                x2[static_cast<uint64_t>(token) * kPairs + pair0]);
        }
#pragma unroll
        for (int r = 0; r < RowsPerWarp; r++) {
            const int row = row0 + r;
            if (row >= out_dim) continue;
            const __nv_bfloat162 *wr =
                reinterpret_cast<const __nv_bfloat162 *>(
                    weights + static_cast<uint64_t>(row) * in_dim);
            const float2 wv = __bfloat1622float2(wr[pair0]);
#pragma unroll
            for (int token = 0; token < Tokens; token++) {
                accum[r][token] =
                    fmaf(wv.x, xv[token].x, accum[r][token]);
                accum[r][token] =
                    fmaf(wv.y, xv[token].y, accum[r][token]);
            }
        }
    }

#pragma unroll
    for (int offset = 16; offset != 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < RowsPerWarp; r++) {
#pragma unroll
            for (int token = 0; token < Tokens; token++) {
                accum[r][token] +=
                    __shfl_down_sync(0xffffffffu, accum[r][token], offset);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RowsPerWarp; r++) {
            const int row = row0 + r;
            if (row >= out_dim) continue;
#pragma unroll
            for (int token = 0; token < Tokens; token++) {
                out[static_cast<uint64_t>(token) * out_dim + row] =
                    accum[r][token];
            }
        }
    }
}

struct Buffers {
    void *weights_allocation = nullptr;
    __nv_bfloat16 *weights = nullptr;
    float *x = nullptr;
    __nv_bfloat16 *xb = nullptr;
    float *out = nullptr;
    void *workspace = nullptr;
};

static bool allocate_buffers(Buffers *b) {
    const size_t weight_bytes =
        kMaxWeightValues * sizeof(__nv_bfloat16);
    const size_t x_values = static_cast<size_t>(kMaxTokens) * kMaxIn;
    const size_t out_values =
        static_cast<size_t>(kMaxTokens) * kMaxOut;
    if (!cuda_ok(cudaMalloc(
                     &b->weights_allocation,
                     weight_bytes + 128),
                 "allocate weights")) {
        return false;
    }
    b->weights = reinterpret_cast<__nv_bfloat16 *>(
        static_cast<unsigned char *>(b->weights_allocation) +
        gWeightOffsetBytes);
    return cuda_ok(cudaMalloc(
                       reinterpret_cast<void **>(&b->x),
                       x_values * sizeof(float)),
                   "allocate input") &&
           cuda_ok(cudaMalloc(
                       reinterpret_cast<void **>(&b->xb),
                       x_values * sizeof(__nv_bfloat16)),
                   "allocate BF16 input") &&
           cuda_ok(cudaMalloc(
                       reinterpret_cast<void **>(&b->out),
                       out_values * sizeof(float)),
                   "allocate output") &&
           (kLtWorkspaceBytes == 0 ||
            cuda_ok(cudaMalloc(&b->workspace, kLtWorkspaceBytes),
                    "allocate cuBLASLt workspace"));
}

static void free_buffers(Buffers *b) {
    if (b->workspace) cudaFree(b->workspace);
    if (b->out) cudaFree(b->out);
    if (b->xb) cudaFree(b->xb);
    if (b->x) cudaFree(b->x);
    if (b->weights_allocation) cudaFree(b->weights_allocation);
    *b = {};
}

static bool initialize_buffers(const Buffers &b) {
    const uint64_t weight_values = kMaxWeightValues;
    const uint64_t x_values =
        static_cast<uint64_t>(kMaxTokens) * kMaxIn;
    init_bf16_kernel<<<
        static_cast<unsigned>((weight_values + 255u) / 256u), 256>>>(
        b.weights, weight_values);
    init_f32_kernel<<<
        static_cast<unsigned>((x_values + 255u) / 256u), 256>>>(
        b.x, x_values);
    return cuda_ok(cudaGetLastError(), "initialize buffers") &&
           cuda_ok(cudaDeviceSynchronize(), "synchronize initialization");
}

static bool launch_convert(const Buffers &b, int tokens, int in_dim) {
    const uint64_t values = static_cast<uint64_t>(tokens) * in_dim;
    f32_to_bf16_bench_kernel<<<
        static_cast<unsigned>((values + 255u) / 256u), 256>>>(
        b.xb, b.x, values);
    return cuda_ok(cudaGetLastError(), "activation conversion");
}

static bool launch_cublas(
        cublasHandle_t handle,
        const Buffers &b,
        int in_dim,
        int out_dim,
        int tokens,
        cublasGemmAlgo_t algo) {
    if (!launch_convert(b, tokens, in_dim)) return false;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublas_ok(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_dim,
            tokens,
            in_dim,
            &alpha,
            b.weights,
            CUDA_R_16BF,
            in_dim,
            b.xb,
            CUDA_R_16BF,
            in_dim,
            &beta,
            b.out,
            CUDA_R_32F,
            out_dim,
            CUBLAS_COMPUTE_32F,
            algo),
        "GemmEx");
}

template <int RowsPerWarp, int Tokens>
static bool launch_full_warp(
        const Buffers &b, int in_dim, int out_dim) {
    if (!launch_convert(b, Tokens, in_dim)) return false;
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const int row_groups = (out_dim + RowsPerWarp - 1) / RowsPerWarp;
    const int blocks = (row_groups + warps - 1) / warps;
    bf16_full_warp_gemv_kernel<RowsPerWarp, Tokens>
        <<<blocks, threads>>>(
            b.out, b.weights, b.xb, in_dim, out_dim);
    return cuda_ok(cudaGetLastError(), "full-warp GEMV");
}

template <typename Launch>
static float time_launch(
        Launch launch, int warmups, int repetitions, bool *ok) {
    *ok = false;
    for (int i = 0; i < warmups; i++) {
        if (!launch()) return 0.0f;
    }
    if (!cuda_ok(cudaDeviceSynchronize(), "warmup synchronize"))
        return 0.0f;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    if (!cuda_ok(cudaEventCreate(&begin), "create begin event") ||
        !cuda_ok(cudaEventCreate(&end), "create end event")) {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        return 0.0f;
    }
    cudaEventRecord(begin);
    for (int i = 0; i < repetitions; i++) {
        if (!launch()) {
            cudaEventDestroy(end);
            cudaEventDestroy(begin);
            return 0.0f;
        }
    }
    cudaEventRecord(end);
    if (!cuda_ok(cudaEventSynchronize(end), "timing synchronize")) {
        cudaEventDestroy(end);
        cudaEventDestroy(begin);
        return 0.0f;
    }
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, begin, end);
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    *ok = true;
    return elapsed_ms / static_cast<float>(repetitions);
}

struct Error {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    double rms = 0.0;
};

static bool copy_output(
        const Buffers &b, int out_dim, int tokens,
        std::vector<float> *host) {
    host->resize(static_cast<size_t>(out_dim) * tokens);
    return cuda_ok(
        cudaMemcpy(
            host->data(), b.out,
            host->size() * sizeof(float),
            cudaMemcpyDeviceToHost),
        "copy output");
}

static Error compare(
        const std::vector<float> &got,
        const std::vector<float> &reference) {
    Error error;
    double squared = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        const float absolute = std::fabs(got[i] - reference[i]);
        const float relative =
            absolute / std::max(1.0e-6f, std::fabs(reference[i]));
        error.max_abs = std::max(error.max_abs, absolute);
        error.max_rel = std::max(error.max_rel, relative);
        squared += static_cast<double>(absolute) * absolute;
    }
    error.rms = std::sqrt(squared / static_cast<double>(got.size()));
    return error;
}

static double effective_gbps(
        int in_dim, int out_dim, float milliseconds) {
    const double bytes =
        static_cast<double>(out_dim) * in_dim * sizeof(__nv_bfloat16);
    return bytes / (static_cast<double>(milliseconds) * 1.0e6);
}

struct LtProblem {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristics;

    void destroy() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
        *this = {};
    }
};

static bool create_lt_problem(
        cublasLtHandle_t handle,
        int in_dim,
        int out_dim,
        int tokens,
        LtProblem *p) {
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    if (!cublaslt_ok(
            cublasLtMatmulDescCreate(
                &p->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "create operation") ||
        !cublaslt_ok(
            cublasLtMatmulDescSetAttribute(
                p->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                &transa, sizeof(transa)),
            "set transa") ||
        !cublaslt_ok(
            cublasLtMatmulDescSetAttribute(
                p->operation, CUBLASLT_MATMUL_DESC_TRANSB,
                &transb, sizeof(transb)),
            "set transb") ||
        !cublaslt_ok(
            cublasLtMatrixLayoutCreate(
                &p->a, CUDA_R_16BF, in_dim, out_dim, in_dim),
            "create A layout") ||
        !cublaslt_ok(
            cublasLtMatrixLayoutCreate(
                &p->b, CUDA_R_16BF, in_dim, tokens, in_dim),
            "create B layout") ||
        !cublaslt_ok(
            cublasLtMatrixLayoutCreate(
                &p->c, CUDA_R_32F, out_dim, tokens, out_dim),
            "create C layout") ||
        !cublaslt_ok(
            cublasLtMatmulPreferenceCreate(&p->preference),
            "create preference") ||
        !cublaslt_ok(
            cublasLtMatmulPreferenceSetAttribute(
                p->preference,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &kLtWorkspaceBytes,
                sizeof(kLtWorkspaceBytes)),
            "set workspace preference") ||
        !cublaslt_ok(
            cublasLtMatmulPreferenceSetAttribute(
                p->preference,
                CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                &gMinAlignmentABytes,
                sizeof(gMinAlignmentABytes)),
            "set A alignment preference")) {
        p->destroy();
        return false;
    }
    constexpr int requested = 32;
    p->heuristics.resize(requested);
    int returned = 0;
    const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        p->operation,
        p->a,
        p->b,
        p->c,
        p->c,
        p->preference,
        requested,
        p->heuristics.data(),
        &returned);
    if (!cublaslt_ok(status, "get heuristics")) {
        p->destroy();
        return false;
    }
    p->heuristics.resize(returned);
    return true;
}

static bool launch_lt(
        cublasLtHandle_t handle,
        const LtProblem &p,
        const cublasLtMatmulAlgo_t &algo,
        const Buffers &b,
        int in_dim,
        int tokens) {
    if (!launch_convert(b, tokens, in_dim)) return false;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublaslt_ok(
        cublasLtMatmul(
            handle,
            p.operation,
            &alpha,
            b.weights,
            p.a,
            b.xb,
            p.b,
            &beta,
            b.out,
            p.c,
            b.out,
            p.c,
            &algo,
            b.workspace,
            kLtWorkspaceBytes,
            nullptr),
        "matmul");
}

static int lt_config_int(
        const cublasLtMatmulAlgo_t &algo,
        cublasLtMatmulAlgoConfigAttributes_t attribute) {
    size_t written = 0;
    if (attribute == CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID ||
        attribute == CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID) {
        uint16_t value = 0;
        if (cublasLtMatmulAlgoConfigGetAttribute(
                &algo, attribute, &value, sizeof(value), &written) !=
            CUBLAS_STATUS_SUCCESS) {
            return -1;
        }
        return static_cast<int>(value);
    }
    int32_t value = -1;
    if (cublasLtMatmulAlgoConfigGetAttribute(
            &algo, attribute, &value, sizeof(value), &written) !=
        CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    return value;
}

struct Result {
    std::string backend;
    int parameter = 0;
    float milliseconds = 0.0f;
    Error error;
    double gbps = 0.0;
    int lt_algo_id = -1;
    int lt_tile = -1;
    int lt_stages = -1;
    int lt_split_k = -1;
    int lt_custom = -1;
    int lt_swizzle = -1;
    int lt_reduction = -1;
    int lt_inner = -1;
    int lt_cluster = -1;
    size_t workspace = 0;
};

static bool error_is_acceptable(const Error &e) {
    return e.max_abs <= 5.0e-3f || e.max_rel <= 2.0e-3f;
}

static void print_result(
        const Shape &shape, int tokens, const Result &r) {
    std::printf(
        "RESULT,%s,%d,%s,%d,%.6f,%.2f,%.9g,%.9g,%.9g,"
        "%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu\n",
        shape.name,
        tokens,
        r.backend.c_str(),
        r.parameter,
        r.milliseconds,
        r.gbps,
        r.error.max_abs,
        r.error.max_rel,
        r.error.rms,
        r.lt_algo_id,
        r.lt_tile,
        r.lt_stages,
        r.lt_split_k,
        r.lt_custom,
        r.lt_swizzle,
        r.lt_reduction,
        r.lt_inner,
        r.lt_cluster,
        r.workspace);
}

static bool benchmark_shape(
        cublasHandle_t cublas,
        cublasLtHandle_t lt,
        const Buffers &buffers,
        const Shape &shape,
        int tokens) {
    constexpr int warmups = 4;
    const int repetitions = shape.out >= 1024 ? 16 : 40;
    bool ok = false;
    const float baseline_ms = time_launch(
        [&]() {
            return launch_cublas(
                cublas, buffers, shape.in, shape.out, tokens,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        },
        warmups, repetitions, &ok);
    if (!ok) return false;
    std::vector<float> reference;
    if (!launch_cublas(
            cublas, buffers, shape.in, shape.out, tokens,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP) ||
        !cuda_ok(cudaDeviceSynchronize(), "reference synchronize") ||
        !copy_output(buffers, shape.out, tokens, &reference)) {
        return false;
    }
    Result baseline;
    baseline.backend = "cublas_default_tensor";
    baseline.parameter = static_cast<int>(CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    baseline.milliseconds = baseline_ms;
    baseline.gbps =
        effective_gbps(shape.in, shape.out, baseline_ms);
    print_result(shape, tokens, baseline);

    std::vector<int> explicit_algorithms;
    explicit_algorithms.push_back(static_cast<int>(CUBLAS_GEMM_DEFAULT));
    for (int i = static_cast<int>(CUBLAS_GEMM_ALGO0);
         i <= static_cast<int>(CUBLAS_GEMM_ALGO23); i++) {
        explicit_algorithms.push_back(i);
    }
    for (int i = static_cast<int>(CUBLAS_GEMM_ALGO0_TENSOR_OP);
         i <= static_cast<int>(CUBLAS_GEMM_ALGO15_TENSOR_OP); i++) {
        explicit_algorithms.push_back(i);
    }
    for (int algorithm : explicit_algorithms) {
        const auto algo = static_cast<cublasGemmAlgo_t>(algorithm);
        if (!launch_cublas(
                cublas, buffers, shape.in, shape.out, tokens, algo) ||
            !cuda_ok(cudaDeviceSynchronize(), "explicit algo synchronize")) {
            (void)cudaGetLastError();
            continue;
        }
        std::vector<float> got;
        if (!copy_output(buffers, shape.out, tokens, &got)) return false;
        Result result;
        result.backend = "cublas_explicit";
        result.parameter = algorithm;
        result.error = compare(got, reference);
        if (!error_is_acceptable(result.error)) {
            print_result(shape, tokens, result);
            continue;
        }
        result.milliseconds = time_launch(
            [&]() {
                return launch_cublas(
                    cublas, buffers, shape.in, shape.out, tokens, algo);
            },
            2, std::max(6, repetitions / 2), &ok);
        if (!ok) continue;
        result.gbps = effective_gbps(
            shape.in, shape.out, result.milliseconds);
        print_result(shape, tokens, result);
    }

    LtProblem problem;
    if (create_lt_problem(
            lt, shape.in, shape.out, tokens, &problem)) {
        for (size_t i = 0; i < problem.heuristics.size(); i++) {
            const auto &heuristic = problem.heuristics[i];
            if (heuristic.state != CUBLAS_STATUS_SUCCESS ||
                heuristic.workspaceSize > kLtWorkspaceBytes) {
                continue;
            }
            if (!launch_lt(
                    lt, problem, heuristic.algo,
                    buffers, shape.in, tokens) ||
                !cuda_ok(cudaDeviceSynchronize(),
                         "cuBLASLt candidate synchronize")) {
                (void)cudaGetLastError();
                continue;
            }
            std::vector<float> got;
            if (!copy_output(buffers, shape.out, tokens, &got)) {
                problem.destroy();
                return false;
            }
            Result result;
            result.backend = "cublaslt";
            result.parameter = static_cast<int>(i);
            result.error = compare(got, reference);
            result.lt_algo_id = lt_config_int(
                heuristic.algo, CUBLASLT_ALGO_CONFIG_ID);
            result.lt_tile = lt_config_int(
                heuristic.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
            result.lt_stages = lt_config_int(
                heuristic.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
            result.lt_split_k = lt_config_int(
                heuristic.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
            result.lt_custom = lt_config_int(
                heuristic.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
            result.lt_swizzle = lt_config_int(
                heuristic.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
            result.lt_reduction = lt_config_int(
                heuristic.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
            result.lt_inner = lt_config_int(
                heuristic.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID);
            result.lt_cluster = lt_config_int(
                heuristic.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
            result.workspace = heuristic.workspaceSize;
            if (!error_is_acceptable(result.error)) {
                print_result(shape, tokens, result);
                continue;
            }
            result.milliseconds = time_launch(
                [&]() {
                    return launch_lt(
                        lt, problem, heuristic.algo,
                        buffers, shape.in, tokens);
                },
                2, std::max(6, repetitions / 2), &ok);
            if (!ok) continue;
            result.gbps =
                effective_gbps(
                    shape.in, shape.out, result.milliseconds);
            print_result(shape, tokens, result);
        }
        problem.destroy();
    }

    if ((reinterpret_cast<uintptr_t>(buffers.weights) &
         (alignof(__nv_bfloat162) - 1u)) != 0) {
        return true;
    }

    const int row_variants_n1[] = {1, 2, 4, 8};
    const int row_variants_n8[] = {1, 2};
    const int *row_variants =
        tokens == 1 ? row_variants_n1 : row_variants_n8;
    const int variant_count = tokens == 1 ? 4 : 2;
    for (int vi = 0; vi < variant_count; vi++) {
        const int rows_per_warp = row_variants[vi];
        auto launch = [&]() -> bool {
            if (tokens == 1) {
                switch (rows_per_warp) {
                    case 1:
                        return launch_full_warp<1, 1>(
                            buffers, shape.in, shape.out);
                    case 2:
                        return launch_full_warp<2, 1>(
                            buffers, shape.in, shape.out);
                    case 4:
                        return launch_full_warp<4, 1>(
                            buffers, shape.in, shape.out);
                    case 8:
                        return launch_full_warp<8, 1>(
                            buffers, shape.in, shape.out);
                }
            } else {
                switch (rows_per_warp) {
                    case 1:
                        return launch_full_warp<1, 8>(
                            buffers, shape.in, shape.out);
                    case 2:
                        return launch_full_warp<2, 8>(
                            buffers, shape.in, shape.out);
                }
            }
            return false;
        };
        if (!launch() ||
            !cuda_ok(cudaDeviceSynchronize(), "warp synchronize")) {
            continue;
        }
        std::vector<float> got;
        if (!copy_output(buffers, shape.out, tokens, &got)) return false;
        Result result;
        result.backend = "warp_gemv";
        result.parameter = rows_per_warp;
        result.error = compare(got, reference);
        result.milliseconds = time_launch(
            launch, 3, repetitions, &ok);
        if (!ok) continue;
        result.gbps = effective_gbps(
            shape.in, shape.out, result.milliseconds);
        print_result(shape, tokens, result);
    }

    return true;
}

static const Shape *find_shape(const char *name) {
    for (const Shape &shape : kShapes) {
        if (std::strcmp(shape.name, name) == 0) return &shape;
    }
    return nullptr;
}

static bool run_profile_mode(
        int argc,
        char **argv,
        cublasHandle_t cublas,
        cublasLtHandle_t lt,
        const Buffers &buffers) {
    if (argc < 5) {
        std::fprintf(
            stderr,
            "usage: %s profile "
            "<current|algo|lt|warp> <shape> <tokens> [parameter]\n",
            argv[0]);
        return false;
    }
    const std::string backend = argv[2];
    const Shape *shape = find_shape(argv[3]);
    const int tokens = std::atoi(argv[4]);
    const int parameter = argc > 5 ? std::atoi(argv[5]) : 0;
    if (!shape || (tokens != 1 && tokens != 8)) return false;
    constexpr int repetitions = 200;
    bool ok = false;
    float milliseconds = 0.0f;
    if (backend == "current") {
        milliseconds = time_launch(
            [&]() {
                return launch_cublas(
                    cublas, buffers, shape->in, shape->out, tokens,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            },
            10, repetitions, &ok);
    } else if (backend == "algo") {
        const auto algo = static_cast<cublasGemmAlgo_t>(parameter);
        milliseconds = time_launch(
            [&]() {
                return launch_cublas(
                    cublas, buffers, shape->in, shape->out, tokens, algo);
            },
            10, repetitions, &ok);
    } else if (backend == "lt") {
        LtProblem problem;
        if (!create_lt_problem(
                lt, shape->in, shape->out, tokens, &problem) ||
            parameter < 0 ||
            parameter >= static_cast<int>(problem.heuristics.size())) {
            problem.destroy();
            return false;
        }
        const auto algorithm = problem.heuristics[parameter].algo;
        milliseconds = time_launch(
            [&]() {
                return launch_lt(
                    lt, problem, algorithm, buffers,
                    shape->in, tokens);
            },
            10, repetitions, &ok);
        problem.destroy();
    } else if (backend == "warp") {
        auto launch = [&]() -> bool {
            if (tokens == 1) {
                switch (parameter) {
                    case 1:
                        return launch_full_warp<1, 1>(
                            buffers, shape->in, shape->out);
                    case 2:
                        return launch_full_warp<2, 1>(
                            buffers, shape->in, shape->out);
                    case 4:
                        return launch_full_warp<4, 1>(
                            buffers, shape->in, shape->out);
                    case 8:
                        return launch_full_warp<8, 1>(
                            buffers, shape->in, shape->out);
                }
            } else {
                switch (parameter) {
                    case 1:
                        return launch_full_warp<1, 8>(
                            buffers, shape->in, shape->out);
                    case 2:
                        return launch_full_warp<2, 8>(
                            buffers, shape->in, shape->out);
                }
            }
            return false;
        };
        milliseconds = time_launch(
            launch, 10, repetitions, &ok);
    }
    if (!ok) return false;
    std::printf(
        "PROFILE,%s,%s,%d,%d,%.6f,%.2f\n",
        backend.c_str(), shape->name, tokens, parameter,
        milliseconds,
        effective_gbps(shape->in, shape->out, milliseconds));
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (const char *value =
            std::getenv("LAGUNA_BF16_WEIGHT_OFFSET_BYTES")) {
        gWeightOffsetBytes =
            static_cast<size_t>(std::strtoul(value, nullptr, 10));
    }
    if (const char *value =
            std::getenv("LAGUNA_BF16_MIN_ALIGNMENT_A_BYTES")) {
        gMinAlignmentABytes =
            static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
    } else if (gWeightOffsetBytes != 0) {
        gMinAlignmentABytes = static_cast<uint32_t>(
            gWeightOffsetBytes & (~gWeightOffsetBytes + 1));
    }
    if (gWeightOffsetBytes > 128 ||
        (gWeightOffsetBytes & 1u) != 0 ||
        gMinAlignmentABytes == 0 ||
        (gMinAlignmentABytes &
         (gMinAlignmentABytes - 1u)) != 0) {
        std::fprintf(stderr, "invalid weight offset/alignment\n");
        return 1;
    }
    cudaDeviceProp prop = {};
    int device = 0;
    if (!cuda_ok(cudaGetDevice(&device), "get device") ||
        !cuda_ok(cudaGetDeviceProperties(&prop, device),
                 "get properties")) {
        return 1;
    }
    std::fprintf(
        stderr,
        "device=%s sm_%d%d weight_offset=%zu min_alignment_A=%u\n",
        prop.name, prop.major, prop.minor,
        gWeightOffsetBytes, gMinAlignmentABytes);

    Buffers buffers;
    cublasHandle_t cublas = nullptr;
    cublasLtHandle_t lt = nullptr;
    if (!allocate_buffers(&buffers) ||
        !initialize_buffers(buffers) ||
        !cublas_ok(cublasCreate(&cublas), "create handle") ||
        !cublas_ok(
            cublasSetMathMode(
                cublas, CUBLAS_TF32_TENSOR_OP_MATH),
            "set math mode") ||
        !cublaslt_ok(cublasLtCreate(&lt), "create Lt handle")) {
        if (lt) cublasLtDestroy(lt);
        if (cublas) cublasDestroy(cublas);
        free_buffers(&buffers);
        return 1;
    }

    bool success = true;
    if (argc > 1 && std::strcmp(argv[1], "profile") == 0) {
        success = run_profile_mode(
            argc, argv, cublas, lt, buffers);
    } else {
        std::puts(
            "kind,shape,tokens,backend,parameter,ms,effective_GBps,"
            "max_abs,max_rel,rms,lt_algo,lt_tile,lt_stages,lt_splitk,"
            "lt_custom,lt_swizzle,lt_reduction,lt_inner,lt_cluster,"
            "workspace");
        for (int tokens : {1, 8}) {
            for (const Shape &shape : kShapes) {
                if (!benchmark_shape(
                        cublas, lt, buffers, shape, tokens)) {
                    success = false;
                    break;
                }
            }
            if (!success) break;
        }
    }

    cublasLtDestroy(lt);
    cublasDestroy(cublas);
    free_buffers(&buffers);
    return success ? 0 : 1;
}
