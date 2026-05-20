#include "qwen36_metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const uint32_t rt_iq3xxs_grid[256];
extern const uint32_t rt_iq3s_grid[512];
extern const uint64_t rt_iq2xs_grid[512];
extern const uint64_t rt_iq2s_grid[1024];
extern const uint64_t rt_iq1s_grid[2048];

struct qwen36_metal_backend {
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLComputePipelineState> matvec_f32;
    __strong id<MTLComputePipelineState> matvec_f16;
    __strong id<MTLComputePipelineState> matvec_bf16;
    __strong id<MTLComputePipelineState> matvec_q4_0;
    __strong id<MTLComputePipelineState> matvec_q4_1;
    __strong id<MTLComputePipelineState> matvec_q5_0;
    __strong id<MTLComputePipelineState> matvec_q5_1;
    __strong id<MTLComputePipelineState> matvec_q8_0;
    __strong id<MTLComputePipelineState> matvec_q8_1;
    __strong id<MTLComputePipelineState> matvec_q2_k;
    __strong id<MTLComputePipelineState> matvec_q3_k;
    __strong id<MTLComputePipelineState> matvec_q4_k;
    __strong id<MTLComputePipelineState> matvec_q5_k;
    __strong id<MTLComputePipelineState> matvec_q6_k;
    __strong id<MTLComputePipelineState> matvec_q8_k;
    __strong id<MTLComputePipelineState> matvec_iq2_xxs;
    __strong id<MTLComputePipelineState> matvec_iq2_xs;
    __strong id<MTLComputePipelineState> matvec_iq2_s;
    __strong id<MTLComputePipelineState> matvec_iq3_xxs;
    __strong id<MTLComputePipelineState> matvec_iq3_s;
    __strong id<MTLComputePipelineState> matvec_iq1_s;
    __strong id<MTLComputePipelineState> matvec_iq1_m;
    __strong id<MTLComputePipelineState> matvec_iq4_nl;
    __strong id<MTLComputePipelineState> matvec_iq4_xs;
    __strong id<MTLComputePipelineState> rms_sum;
    __strong id<MTLComputePipelineState> rms_apply_f32;
    __strong id<MTLComputePipelineState> rms_apply_f16;
    __strong id<MTLComputePipelineState> rms_apply_bf16;
    __strong id<MTLComputePipelineState> silu_mul;
    __strong id<MTLComputePipelineState> l2_sum;
    __strong id<MTLComputePipelineState> l2_apply;
    __strong id<MTLComputePipelineState> delta_mem;
    __strong id<MTLComputePipelineState> delta_update;
    __strong id<MTLComputePipelineState> delta_core;
    __strong id<MTLComputePipelineState> attention_scores;
    __strong id<MTLComputePipelineState> attention_apply;
};

static const uint64_t qwen36_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static void qwen36_metal_set_err(char *err, size_t errlen,
                                 const char *fmt, ...) {
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static NSString *qwen36_metal_source(void) {
    return @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "constant int qwen36_iq4nl_values[16] = {\n"
    "    -127, -104, -83, -65, -49, -35, -22, -10,\n"
    "       1,   13,  25,  38,  53,  69,  89, 113,\n"
    "};\n"
    "static inline float qwen36_f16_bits_to_f32(ushort bits) {\n"
    "    return float(as_type<half>(bits));\n"
    "}\n"
    "static inline int qwen36_i8_from_u8(uchar v) {\n"
    "    int out = int(v);\n"
    "    return out >= 128 ? out - 256 : out;\n"
    "}\n"
    "static inline uint qwen36_le32(device const uchar *p) {\n"
    "    return uint(p[0]) | (uint(p[1]) << 8) |\n"
    "           (uint(p[2]) << 16) | (uint(p[3]) << 24);\n"
    "}\n"
    "static inline uchar qwen36_iq2xxs_signs(uint idx) {\n"
    "    uint v = idx & 127u;\n"
    "    uint p = v;\n"
    "    p ^= p >> 4;\n"
    "    p ^= p >> 2;\n"
    "    p ^= p >> 1;\n"
    "    return uchar(v | ((p & 1u) << 7));\n"
    "}\n"
    "static inline void qwen36_q4_k_scale_min(uint group, device const uchar *scales,\n"
    "                                        thread uchar &d, thread uchar &m) {\n"
    "    if (group < 4u) {\n"
    "        d = scales[group] & 63u;\n"
    "        m = scales[group + 4u] & 63u;\n"
    "    } else {\n"
    "        d = (scales[group + 4u] & 15u) | ((scales[group - 4u] >> 6) << 4);\n"
    "        m = (scales[group + 4u] >> 4) | ((scales[group] >> 6) << 4);\n"
    "    }\n"
    "}\n"
    "kernel void qwen36_matvec_f32(device const float *w [[buffer(0)]],\n"
    "                              device const float *x [[buffer(1)]],\n"
    "                              device float *y [[buffer(2)]],\n"
    "                              constant uint &in_dim [[buffer(3)]],\n"
    "                              constant uint &out_dim [[buffer(4)]],\n"
    "                              uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    float sum = 0.0f;\n"
    "    uint base = row * in_dim;\n"
    "    for (uint col = 0; col < in_dim; ++col) {\n"
    "        sum = fma(w[base + col], x[col], sum);\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_f16(device const half *w [[buffer(0)]],\n"
    "                              device const float *x [[buffer(1)]],\n"
    "                              device float *y [[buffer(2)]],\n"
    "                              constant uint &in_dim [[buffer(3)]],\n"
    "                              constant uint &out_dim [[buffer(4)]],\n"
    "                              uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    float sum = 0.0f;\n"
    "    uint base = row * in_dim;\n"
    "    for (uint col = 0; col < in_dim; ++col) {\n"
    "        sum = fma(float(w[base + col]), x[col], sum);\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_bf16(device const ushort *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    float sum = 0.0f;\n"
    "    uint base = row * in_dim;\n"
    "    for (uint col = 0; col < in_dim; ++col) {\n"
    "        uint bits = uint(w[base + col]) << 16;\n"
    "        sum = fma(as_type<float>(bits), x[col], sum);\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q4_0(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 32u;\n"
    "    uint row_base = row * blocks_per_row * 18u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 18u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        device const uchar *qs = w + bb + 2u;\n"
    "        uint xbase = block * 32u;\n"
    "        for (uint i = 0; i < 32u; ++i) {\n"
    "            uchar packed = qs[i & 15u];\n"
    "            uchar q = i < 16u ? (packed & 15u) : (packed >> 4);\n"
    "            sum = fma(d * (float(q) - 8.0f), x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q4_1(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 32u;\n"
    "    uint row_base = row * blocks_per_row * 20u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 20u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        float m = qwen36_f16_bits_to_f32(ushort(w[bb + 2u]) | (ushort(w[bb + 3u]) << 8));\n"
    "        device const uchar *qs = w + bb + 4u;\n"
    "        uint xbase = block * 32u;\n"
    "        for (uint i = 0; i < 32u; ++i) {\n"
    "            uchar packed = qs[i & 15u];\n"
    "            uchar q = i < 16u ? (packed & 15u) : (packed >> 4);\n"
    "            sum = fma(d * float(q) + m, x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q5_0(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 32u;\n"
    "    uint row_base = row * blocks_per_row * 22u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 22u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        uint qh = qwen36_le32(w + bb + 2u);\n"
    "        device const uchar *qs = w + bb + 6u;\n"
    "        uint xbase = block * 32u;\n"
    "        for (uint i = 0; i < 32u; ++i) {\n"
    "            uchar packed = qs[i & 15u];\n"
    "            uchar q = i < 16u ? (packed & 15u) : (packed >> 4);\n"
    "            q |= uchar(((qh >> i) & 1u) << 4);\n"
    "            sum = fma(d * (float(q) - 16.0f), x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q5_1(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 32u;\n"
    "    uint row_base = row * blocks_per_row * 24u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 24u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        float m = qwen36_f16_bits_to_f32(ushort(w[bb + 2u]) | (ushort(w[bb + 3u]) << 8));\n"
    "        uint qh = qwen36_le32(w + bb + 4u);\n"
    "        device const uchar *qs = w + bb + 8u;\n"
    "        uint xbase = block * 32u;\n"
    "        for (uint i = 0; i < 32u; ++i) {\n"
    "            uchar packed = qs[i & 15u];\n"
    "            uchar q = i < 16u ? (packed & 15u) : (packed >> 4);\n"
    "            q |= uchar(((qh >> i) & 1u) << 4);\n"
    "            sum = fma(d * float(q) + m, x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q8_0(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 32u;\n"
    "    uint row_base = row * blocks_per_row * 34u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 34u;\n"
    "        ushort dbits = ushort(w[bb]) | (ushort(w[bb + 1u]) << 8);\n"
    "        float d = qwen36_f16_bits_to_f32(dbits);\n"
    "        uint xbase = block * 32u;\n"
    "        for (uint i = 0; i < 32u; ++i) {\n"
    "            sum = fma(d * float(qwen36_i8_from_u8(w[bb + 2u + i])),\n"
    "                      x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q8_1(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 32u;\n"
    "    uint row_base = row * blocks_per_row * 36u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 36u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        uint xbase = block * 32u;\n"
    "        for (uint i = 0; i < 32u; ++i) {\n"
    "            sum = fma(d * float(qwen36_i8_from_u8(w[bb + 4u + i])),\n"
    "                      x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_iq4_nl(device const uchar *w [[buffer(0)]],\n"
    "                                 device const float *x [[buffer(1)]],\n"
    "                                 device float *y [[buffer(2)]],\n"
    "                                 constant uint &in_dim [[buffer(3)]],\n"
    "                                 constant uint &out_dim [[buffer(4)]],\n"
    "                                 uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 32u;\n"
    "    uint row_base = row * blocks_per_row * 18u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 18u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        device const uchar *qs = w + bb + 2u;\n"
    "        uint xbase = block * 32u;\n"
    "        for (uint i = 0; i < 32u; ++i) {\n"
    "            uchar packed = qs[i & 15u];\n"
    "            uchar q = i < 16u ? (packed & 15u) : (packed >> 4);\n"
    "            sum = fma(d * float(qwen36_iq4nl_values[q]), x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q2_k(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 84u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 84u;\n"
    "        device const uchar *scales = w + bb;\n"
    "        device const uchar *qs = w + bb + 16u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb + 80u]) | (ushort(w[bb + 81u]) << 8));\n"
    "        float dmin = qwen36_f16_bits_to_f32(ushort(w[bb + 82u]) | (ushort(w[bb + 83u]) << 8));\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint i = 0; i < 256u; ++i) {\n"
    "            uint half_idx = i / 128u;\n"
    "            uint local = i & 127u;\n"
    "            uint group = i / 16u;\n"
    "            uint shift = (local / 32u) * 2u;\n"
    "            uint qidx = half_idx * 32u + (local & 31u);\n"
    "            uchar sc = scales[group];\n"
    "            uchar q = (qs[qidx] >> shift) & 3u;\n"
    "            float weight = d * float(sc & 15u) * float(q) -\n"
    "                           dmin * float(sc >> 4);\n"
    "            sum = fma(weight, x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q3_k(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 110u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 110u;\n"
    "        device const uchar *hm = w + bb;\n"
    "        device const uchar *qs = w + bb + 32u;\n"
    "        device const uchar *packed_scales = w + bb + 96u;\n"
    "        uint aux0 = qwen36_le32(packed_scales + 0u);\n"
    "        uint aux1 = qwen36_le32(packed_scales + 4u);\n"
    "        uint aux2_raw = qwen36_le32(packed_scales + 8u);\n"
    "        uint tmp = aux2_raw;\n"
    "        uint aux2 = ((aux0 >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);\n"
    "        uint aux3 = ((aux1 >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);\n"
    "        aux0 = (aux0 & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);\n"
    "        aux1 = (aux1 & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);\n"
    "        float d_all = qwen36_f16_bits_to_f32(ushort(w[bb + 108u]) | (ushort(w[bb + 109u]) << 8));\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint i = 0; i < 256u; ++i) {\n"
    "            uint half_idx = i / 128u;\n"
    "            uint local = i & 127u;\n"
    "            uint group = i / 16u;\n"
    "            uint shift = (local / 32u) * 2u;\n"
    "            uint qidx = half_idx * 32u + (local & 31u);\n"
    "            uint scale_word = group < 4u ? aux0 : (group < 8u ? aux1 : (group < 12u ? aux2 : aux3));\n"
    "            uint scale_shift = (group & 3u) * 8u;\n"
    "            int scale = qwen36_i8_from_u8(uchar((scale_word >> scale_shift) & 255u)) - 32;\n"
    "            uchar high_mask = uchar(1u << (group / 2u));\n"
    "            int q = int((qs[qidx] >> shift) & 3u) - ((hm[qidx] & high_mask) ? 0 : 4);\n"
    "            float weight = d_all * float(scale) * float(q);\n"
    "            sum = fma(weight, x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q4_k(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 144u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 144u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        float dmin = qwen36_f16_bits_to_f32(ushort(w[bb + 2u]) | (ushort(w[bb + 3u]) << 8));\n"
    "        device const uchar *scales = w + bb + 4u;\n"
    "        device const uchar *qs = w + bb + 16u;\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint i = 0; i < 256u; ++i) {\n"
    "            uint group = i / 32u;\n"
    "            uchar sc;\n"
    "            uchar mn;\n"
    "            qwen36_q4_k_scale_min(group, scales, sc, mn);\n"
    "            uint within64 = i & 63u;\n"
    "            uint byte_index = (i / 64u) * 32u + (within64 & 31u);\n"
    "            uchar packed = qs[byte_index];\n"
    "            uchar q = within64 < 32u ? (packed & 15u) : (packed >> 4);\n"
    "            float weight = d * float(sc) * float(q) - dmin * float(mn);\n"
    "            sum = fma(weight, x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q5_k(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 176u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 176u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        float dmin = qwen36_f16_bits_to_f32(ushort(w[bb + 2u]) | (ushort(w[bb + 3u]) << 8));\n"
    "        device const uchar *scales = w + bb + 4u;\n"
    "        device const uchar *qh = w + bb + 16u;\n"
    "        device const uchar *qs = w + bb + 48u;\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint i = 0; i < 256u; ++i) {\n"
    "            uint group = i / 32u;\n"
    "            uchar sc;\n"
    "            uchar mn;\n"
    "            qwen36_q4_k_scale_min(group, scales, sc, mn);\n"
    "            uint within64 = i & 63u;\n"
    "            uint byte_index = (i / 64u) * 32u + (within64 & 31u);\n"
    "            uchar packed = qs[byte_index];\n"
    "            uchar q = within64 < 32u ? (packed & 15u) : (packed >> 4);\n"
    "            q |= uchar(((qh[i >> 3] >> (i & 7u)) & 1u) << 4);\n"
    "            float weight = d * float(sc) * float(q) - dmin * float(mn);\n"
    "            sum = fma(weight, x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q6_k(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 210u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 210u;\n"
    "        device const uchar *ql = w + bb;\n"
    "        device const uchar *qh = w + bb + 128u;\n"
    "        device const uchar *scales = w + bb + 192u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb + 208u]) | (ushort(w[bb + 209u]) << 8));\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint i = 0; i < 256u; ++i) {\n"
    "            uint half_idx = i / 128u;\n"
    "            uint local = i & 127u;\n"
    "            uint l = local & 31u;\n"
    "            uint qh_base = half_idx * 32u + l;\n"
    "            uchar qhb = qh[qh_base];\n"
    "            uint scale_base = half_idx * 8u;\n"
    "            int q;\n"
    "            int sc;\n"
    "            if (local < 32u) {\n"
    "                q = int((ql[half_idx * 64u + l] & 15u) | (((qhb >> 0) & 3u) << 4));\n"
    "                sc = qwen36_i8_from_u8(scales[scale_base + (l / 16u) + 0u]);\n"
    "            } else if (local < 64u) {\n"
    "                q = int((ql[half_idx * 64u + l + 32u] & 15u) | (((qhb >> 2) & 3u) << 4));\n"
    "                sc = qwen36_i8_from_u8(scales[scale_base + (l / 16u) + 2u]);\n"
    "            } else if (local < 96u) {\n"
    "                q = int((ql[half_idx * 64u + l] >> 4) | (((qhb >> 4) & 3u) << 4));\n"
    "                sc = qwen36_i8_from_u8(scales[scale_base + (l / 16u) + 4u]);\n"
    "            } else {\n"
    "                q = int((ql[half_idx * 64u + l + 32u] >> 4) | (((qhb >> 6) & 3u) << 4));\n"
    "                sc = qwen36_i8_from_u8(scales[scale_base + (l / 16u) + 6u]);\n"
    "            }\n"
    "            float weight = d * float(sc) * float(q - 32);\n"
    "            sum = fma(weight, x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_q8_k(device const uchar *w [[buffer(0)]],\n"
    "                               device const float *x [[buffer(1)]],\n"
    "                               device float *y [[buffer(2)]],\n"
    "                               constant uint &in_dim [[buffer(3)]],\n"
    "                               constant uint &out_dim [[buffer(4)]],\n"
    "                               uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 292u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 292u;\n"
    "        uint dbits = uint(w[bb]) | (uint(w[bb + 1u]) << 8) |\n"
    "                     (uint(w[bb + 2u]) << 16) | (uint(w[bb + 3u]) << 24);\n"
    "        float d = as_type<float>(dbits);\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint i = 0; i < 256u; ++i) {\n"
    "            sum = fma(d * float(qwen36_i8_from_u8(w[bb + 4u + i])),\n"
    "                      x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_iq4_xs(device const uchar *w [[buffer(0)]],\n"
    "                                 device const float *x [[buffer(1)]],\n"
    "                                 device float *y [[buffer(2)]],\n"
    "                                 constant uint &in_dim [[buffer(3)]],\n"
    "                                 constant uint &out_dim [[buffer(4)]],\n"
    "                                 uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 136u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 136u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        uint scales_h = uint(w[bb + 2u]) | (uint(w[bb + 3u]) << 8);\n"
    "        device const uchar *scales_l = w + bb + 4u;\n"
    "        device const uchar *qs = w + bb + 8u;\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint i = 0; i < 256u; ++i) {\n"
    "            uint group = i / 32u;\n"
    "            uint local = i & 31u;\n"
    "            uint ls = ((scales_l[group / 2u] >> (4u * (group & 1u))) & 15u) |\n"
    "                      (((scales_h >> (2u * group)) & 3u) << 4);\n"
    "            uchar packed = qs[group * 16u + (local & 15u)];\n"
    "            uchar q = local < 16u ? (packed & 15u) : (packed >> 4);\n"
    "            float weight = d * float(int(ls) - 32) * float(qwen36_iq4nl_values[q]);\n"
    "            sum = fma(weight, x[xbase + i], sum);\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_iq3_xxs(device const uchar *w [[buffer(0)]],\n"
    "                                  device const float *x [[buffer(1)]],\n"
    "                                  device float *y [[buffer(2)]],\n"
    "                                  constant uint &in_dim [[buffer(3)]],\n"
    "                                  constant uint &out_dim [[buffer(4)]],\n"
    "                                  device const uint *iq3_grid [[buffer(5)]],\n"
    "                                  uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 98u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 98u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        device const uchar *qs = w + bb + 2u;\n"
    "        device const uchar *scales_and_signs = qs + 64u;\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint ib32 = 0; ib32 < 8u; ++ib32) {\n"
    "            uint aux = qwen36_le32(scales_and_signs + 4u * ib32);\n"
    "            float dl = d * (0.5f + float(aux >> 28)) * 0.5f;\n"
    "            for (uint group = 0; group < 4u; ++group) {\n"
    "                uchar signs = qwen36_iq2xxs_signs((aux >> (7u * group)) & 127u);\n"
    "                for (uint item = 0; item < 8u; ++item) {\n"
    "                    uchar qidx = qs[ib32 * 8u + group * 2u + (item >= 4u ? 1u : 0u)];\n"
    "                    uint grid = iq3_grid[qidx];\n"
    "                    float q = float((grid >> (8u * (item & 3u))) & 255u);\n"
    "                    if ((signs & uchar(1u << item)) != 0) q = -q;\n"
    "                    uint col = ib32 * 32u + group * 8u + item;\n"
    "                    sum = fma(dl * q, x[xbase + col], sum);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_iq1_m(device const uchar *w [[buffer(0)]],\n"
    "                                device const float *x [[buffer(1)]],\n"
    "                                device float *y [[buffer(2)]],\n"
    "                                constant uint &in_dim [[buffer(3)]],\n"
    "                                constant uint &out_dim [[buffer(4)]],\n"
    "                                device const ulong *iq1s_grid [[buffer(5)]],\n"
    "                                uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 56u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 56u;\n"
    "        device const uchar *qs = w + bb;\n"
    "        device const uchar *qh = w + bb + 32u;\n"
    "        device const uchar *scales = w + bb + 48u;\n"
    "        uint sc0 = uint(scales[0]) | (uint(scales[1]) << 8);\n"
    "        uint sc1 = uint(scales[2]) | (uint(scales[3]) << 8);\n"
    "        uint sc2 = uint(scales[4]) | (uint(scales[5]) << 8);\n"
    "        uint sc3 = uint(scales[6]) | (uint(scales[7]) << 8);\n"
    "        ushort d_bits = ushort((sc0 >> 12) | ((sc1 >> 8) & 0x00f0u) |\n"
    "                               ((sc2 >> 4) & 0x0f00u) | (sc3 & 0xf000u));\n"
    "        float d = qwen36_f16_bits_to_f32(d_bits);\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint ib32 = 0; ib32 < 8u; ++ib32) {\n"
    "            uint sc = uint(scales[2u * (ib32 / 2u)]) | (uint(scales[2u * (ib32 / 2u) + 1u]) << 8);\n"
    "            uint shift = 6u * (ib32 & 1u);\n"
    "            for (uint group = 0; group < 4u; ++group) {\n"
    "                uint scale = group < 2u ? ((sc >> shift) & 7u) : ((sc >> (shift + 3u)) & 7u);\n"
    "                uchar qhb = qh[ib32 * 2u + group / 2u];\n"
    "                uint grid_idx = uint(qs[ib32 * 4u + group]) |\n"
    "                    (group & 1u ? ((uint(qhb) << 4) & 0x700u) : ((uint(qhb) << 8) & 0x700u));\n"
    "                ulong grid = iq1s_grid[grid_idx];\n"
    "                float dl = d * float(2u * scale + 1u);\n"
    "                float delta = (qhb & uchar(group & 1u ? 0x80u : 0x08u)) != 0 ? -0.125f : 0.125f;\n"
    "                for (uint i = 0; i < 8u; ++i) {\n"
    "                    int q = int((grid >> (8u * i)) & 255ul);\n"
    "                    if (q >= 128) q -= 256;\n"
    "                    uint col = ib32 * 32u + group * 8u + i;\n"
    "                    sum = fma(dl * (float(q) + delta), x[xbase + col], sum);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_iq1_s(device const uchar *w [[buffer(0)]],\n"
    "                                device const float *x [[buffer(1)]],\n"
    "                                device float *y [[buffer(2)]],\n"
    "                                constant uint &in_dim [[buffer(3)]],\n"
    "                                constant uint &out_dim [[buffer(4)]],\n"
    "                                device const ulong *iq1s_grid [[buffer(5)]],\n"
    "                                uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 50u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 50u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        device const uchar *qs = w + bb + 2u;\n"
    "        device const uchar *qh = w + bb + 34u;\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint ib32 = 0; ib32 < 8u; ++ib32) {\n"
    "            uint qh_word = uint(qh[2u * ib32]) | (uint(qh[2u * ib32 + 1u]) << 8);\n"
    "            float dl = d * float(2u * ((qh_word >> 12) & 7u) + 1u);\n"
    "            float delta = (qh_word & 0x8000u) != 0u ? -0.125f : 0.125f;\n"
    "            for (uint group = 0; group < 4u; ++group) {\n"
    "                uint grid_idx = uint(qs[ib32 * 4u + group]) |\n"
    "                    (((qh_word >> (3u * group)) & 7u) << 8);\n"
    "                ulong grid = iq1s_grid[grid_idx];\n"
    "                for (uint i = 0; i < 8u; ++i) {\n"
    "                    int q = int((grid >> (8u * i)) & 255ul);\n"
    "                    if (q >= 128) q -= 256;\n"
    "                    uint col = ib32 * 32u + group * 8u + i;\n"
    "                    sum = fma(dl * (float(q) + delta), x[xbase + col], sum);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_iq3_s(device const uchar *w [[buffer(0)]],\n"
    "                                device const float *x [[buffer(1)]],\n"
    "                                device float *y [[buffer(2)]],\n"
    "                                constant uint &in_dim [[buffer(3)]],\n"
    "                                constant uint &out_dim [[buffer(4)]],\n"
    "                                device const uint *iq3s_grid [[buffer(5)]],\n"
    "                                uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 110u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 110u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        device const uchar *qs = w + bb + 2u;\n"
    "        device const uchar *qh = w + bb + 66u;\n"
    "        device const uchar *signs = w + bb + 74u;\n"
    "        device const uchar *scales = w + bb + 106u;\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint pair = 0; pair < 4u; ++pair) {\n"
    "            for (uint second = 0; second < 2u; ++second) {\n"
    "                float dl = d * float(1u + 2u * (second == 0u ? (scales[pair] & 15u) : (scales[pair] >> 4)));\n"
    "                uchar qhb = qh[pair * 2u + second];\n"
    "                for (uint group = 0; group < 4u; ++group) {\n"
    "                    uchar sign = signs[pair * 8u + second * 4u + group];\n"
    "                    for (uint item = 0; item < 8u; ++item) {\n"
    "                        uint qoff = pair * 16u + second * 8u + group * 2u + (item >= 4u ? 1u : 0u);\n"
    "                        uint shift = item < 4u ? (8u - 2u * group) : (7u - 2u * group);\n"
    "                        uint grid_idx = uint(qs[qoff]) | ((uint(qhb) << shift) & 256u);\n"
    "                        uint grid = iq3s_grid[grid_idx];\n"
    "                        float q = float((grid >> (8u * (item & 3u))) & 255u);\n"
    "                        if ((sign & uchar(1u << item)) != 0) q = -q;\n"
    "                        uint col = pair * 64u + second * 32u + group * 8u + item;\n"
    "                        sum = fma(dl * q, x[xbase + col], sum);\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_iq2_s(device const uchar *w [[buffer(0)]],\n"
    "                                device const float *x [[buffer(1)]],\n"
    "                                device float *y [[buffer(2)]],\n"
    "                                constant uint &in_dim [[buffer(3)]],\n"
    "                                constant uint &out_dim [[buffer(4)]],\n"
    "                                device const ulong *iq2s_grid [[buffer(5)]],\n"
    "                                uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 82u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 82u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        device const uchar *qs = w + bb + 2u;\n"
    "        device const uchar *qh = w + bb + 66u;\n"
    "        device const uchar *scales = w + bb + 74u;\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint ib32 = 0; ib32 < 8u; ++ib32) {\n"
    "            uint scale_lo = scales[ib32] & 15u;\n"
    "            uint scale_hi = scales[ib32] >> 4;\n"
    "            for (uint group = 0; group < 4u; ++group) {\n"
    "                uint grid_idx = uint(qs[ib32 * 4u + group]) |\n"
    "                    ((uint(qh[ib32]) << (8u - 2u * group)) & 0x300u);\n"
    "                ulong grid = iq2s_grid[grid_idx];\n"
    "                uchar signs = qs[32u + ib32 * 4u + group];\n"
    "                float dl = d * (0.5f + float(group < 2u ? scale_lo : scale_hi)) * 0.25f;\n"
    "                for (uint i = 0; i < 8u; ++i) {\n"
    "                    float q = float((grid >> (8u * i)) & 255ul);\n"
    "                    if ((signs & uchar(1u << i)) != 0) q = -q;\n"
    "                    uint col = ib32 * 32u + group * 8u + i;\n"
    "                    sum = fma(dl * q, x[xbase + col], sum);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_iq2_xs(device const uchar *w [[buffer(0)]],\n"
    "                                 device const float *x [[buffer(1)]],\n"
    "                                 device float *y [[buffer(2)]],\n"
    "                                 constant uint &in_dim [[buffer(3)]],\n"
    "                                 constant uint &out_dim [[buffer(4)]],\n"
    "                                 device const ulong *iq2xs_grid [[buffer(5)]],\n"
    "                                 uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 74u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 74u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        device const uchar *qs = w + bb + 2u;\n"
    "        device const uchar *scales = qs + 64u;\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint ib32 = 0; ib32 < 8u; ++ib32) {\n"
    "            uint scale_lo = scales[ib32] & 15u;\n"
    "            uint scale_hi = scales[ib32] >> 4;\n"
    "            for (uint group = 0; group < 4u; ++group) {\n"
    "                uint packed = uint(qs[2u * (ib32 * 4u + group)]) |\n"
    "                              (uint(qs[2u * (ib32 * 4u + group) + 1u]) << 8);\n"
    "                ulong grid = iq2xs_grid[packed & 511u];\n"
    "                uchar signs = qwen36_iq2xxs_signs(packed >> 9);\n"
    "                float dl = d * (0.5f + float(group < 2u ? scale_lo : scale_hi)) * 0.25f;\n"
    "                for (uint i = 0; i < 8u; ++i) {\n"
    "                    float q = float((grid >> (8u * i)) & 255ul);\n"
    "                    if ((signs & uchar(1u << i)) != 0) q = -q;\n"
    "                    uint col = ib32 * 32u + group * 8u + i;\n"
    "                    sum = fma(dl * q, x[xbase + col], sum);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_matvec_iq2_xxs(device const uchar *w [[buffer(0)]],\n"
    "                                  device const float *x [[buffer(1)]],\n"
    "                                  device float *y [[buffer(2)]],\n"
    "                                  constant uint &in_dim [[buffer(3)]],\n"
    "                                  constant uint &out_dim [[buffer(4)]],\n"
    "                                  device const ulong *iq2_grid [[buffer(5)]],\n"
    "                                  uint row [[thread_position_in_grid]]) {\n"
    "    if (row >= out_dim) return;\n"
    "    uint blocks_per_row = in_dim / 256u;\n"
    "    uint row_base = row * blocks_per_row * 66u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint block = 0; block < blocks_per_row; ++block) {\n"
    "        uint bb = row_base + block * 66u;\n"
    "        float d = qwen36_f16_bits_to_f32(ushort(w[bb]) | (ushort(w[bb + 1u]) << 8));\n"
    "        uint xbase = block * 256u;\n"
    "        for (uint ib32 = 0; ib32 < 8u; ++ib32) {\n"
    "            uint chunk = bb + 2u + ib32 * 8u;\n"
    "            uint aux_g = qwen36_le32(w + chunk);\n"
    "            uint aux_s = qwen36_le32(w + chunk + 4u);\n"
    "            float dl = d * (0.5f + float(aux_s >> 28)) * 0.25f;\n"
    "            for (uint group = 0; group < 4u; ++group) {\n"
    "                uint grid_idx = (aux_g >> (8u * group)) & 255u;\n"
    "                uchar signs = qwen36_iq2xxs_signs((aux_s >> (7u * group)) & 127u);\n"
    "                ulong grid = iq2_grid[grid_idx];\n"
    "                for (uint i = 0; i < 8u; ++i) {\n"
    "                    float q = float((grid >> (8u * i)) & 255ul);\n"
    "                    if ((signs & uchar(1u << i)) != 0) q = -q;\n"
    "                    uint col = ib32 * 32u + group * 8u + i;\n"
    "                    sum = fma(dl * q, x[xbase + col], sum);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    y[row] = sum;\n"
    "}\n"
    "kernel void qwen36_rms_sum(device const float *x [[buffer(0)]],\n"
    "                           device float *partials [[buffer(1)]],\n"
    "                           constant uint &dim [[buffer(2)]],\n"
    "                           constant uint &group_size [[buffer(3)]],\n"
    "                           uint tid [[thread_index_in_threadgroup]],\n"
    "                           uint group [[threadgroup_position_in_grid]]) {\n"
    "    threadgroup float sums[256];\n"
    "    uint idx = group * group_size + tid;\n"
    "    float v = idx < dim ? x[idx] : 0.0f;\n"
    "    sums[tid] = v * v;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint offset = group_size >> 1; offset > 0; offset >>= 1) {\n"
    "        if (tid < offset) sums[tid] += sums[tid + offset];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if (tid == 0) partials[group] = sums[0];\n"
    "}\n"
    "kernel void qwen36_rms_apply_f32(device const float *w [[buffer(0)]],\n"
    "                                 device const float *x [[buffer(1)]],\n"
    "                                 device const float *partials [[buffer(2)]],\n"
    "                                 device float *y [[buffer(3)]],\n"
    "                                 constant uint &dim [[buffer(4)]],\n"
    "                                 constant uint &partial_count [[buffer(5)]],\n"
    "                                 constant uint &add_one [[buffer(6)]],\n"
    "                                 uint gid [[thread_position_in_grid]]) {\n"
    "    if (gid >= dim) return;\n"
    "    float ss = 0.0f;\n"
    "    for (uint i = 0; i < partial_count; ++i) ss += partials[i];\n"
    "    float scale = rsqrt(ss / float(dim) + 1.0e-6f);\n"
    "    float weight = w[gid];\n"
    "    if (add_one != 0) weight += 1.0f;\n"
    "    y[gid] = x[gid] * scale * weight;\n"
    "}\n"
    "kernel void qwen36_rms_apply_f16(device const half *w [[buffer(0)]],\n"
    "                                 device const float *x [[buffer(1)]],\n"
    "                                 device const float *partials [[buffer(2)]],\n"
    "                                 device float *y [[buffer(3)]],\n"
    "                                 constant uint &dim [[buffer(4)]],\n"
    "                                 constant uint &partial_count [[buffer(5)]],\n"
    "                                 constant uint &add_one [[buffer(6)]],\n"
    "                                 uint gid [[thread_position_in_grid]]) {\n"
    "    if (gid >= dim) return;\n"
    "    float ss = 0.0f;\n"
    "    for (uint i = 0; i < partial_count; ++i) ss += partials[i];\n"
    "    float scale = rsqrt(ss / float(dim) + 1.0e-6f);\n"
    "    float weight = float(w[gid]);\n"
    "    if (add_one != 0) weight += 1.0f;\n"
    "    y[gid] = x[gid] * scale * weight;\n"
    "}\n"
    "kernel void qwen36_rms_apply_bf16(device const ushort *w [[buffer(0)]],\n"
    "                                  device const float *x [[buffer(1)]],\n"
    "                                  device const float *partials [[buffer(2)]],\n"
    "                                  device float *y [[buffer(3)]],\n"
    "                                  constant uint &dim [[buffer(4)]],\n"
    "                                  constant uint &partial_count [[buffer(5)]],\n"
    "                                  constant uint &add_one [[buffer(6)]],\n"
    "                                  uint gid [[thread_position_in_grid]]) {\n"
    "    if (gid >= dim) return;\n"
    "    float ss = 0.0f;\n"
    "    for (uint i = 0; i < partial_count; ++i) ss += partials[i];\n"
    "    float scale = rsqrt(ss / float(dim) + 1.0e-6f);\n"
    "    uint bits = uint(w[gid]) << 16;\n"
    "    float weight = as_type<float>(bits);\n"
    "    if (add_one != 0) weight += 1.0f;\n"
    "    y[gid] = x[gid] * scale * weight;\n"
    "}\n"
    "kernel void qwen36_silu_mul(device const float *gate [[buffer(0)]],\n"
    "                            device const float *up [[buffer(1)]],\n"
    "                            device float *y [[buffer(2)]],\n"
    "                            constant uint &dim [[buffer(3)]],\n"
    "                            uint gid [[thread_position_in_grid]]) {\n"
    "    if (gid >= dim) return;\n"
    "    float x = gate[gid];\n"
    "    float z = x >= 0.0f ? exp(-x) : exp(x);\n"
    "    float sig = x >= 0.0f ? (1.0f / (1.0f + z)) : (z / (1.0f + z));\n"
    "    y[gid] = x * sig * up[gid];\n"
    "}\n"
    "kernel void qwen36_l2_sum(device const float *x [[buffer(0)]],\n"
    "                          device float *partials [[buffer(1)]],\n"
    "                          constant uint &dim [[buffer(2)]],\n"
    "                          constant uint &group_size [[buffer(3)]],\n"
    "                          uint tid [[thread_index_in_threadgroup]],\n"
    "                          uint group [[threadgroup_position_in_grid]]) {\n"
    "    threadgroup float sums[256];\n"
    "    uint idx = group * group_size + tid;\n"
    "    float v = idx < dim ? x[idx] : 0.0f;\n"
    "    sums[tid] = v * v;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint offset = group_size >> 1; offset > 0; offset >>= 1) {\n"
    "        if (tid < offset) sums[tid] += sums[tid + offset];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if (tid == 0) partials[group] = sums[0];\n"
    "}\n"
    "kernel void qwen36_l2_apply(device const float *x [[buffer(0)]],\n"
    "                            device const float *partials [[buffer(1)]],\n"
    "                            device float *y [[buffer(2)]],\n"
    "                            constant uint &dim [[buffer(3)]],\n"
    "                            constant uint &partial_count [[buffer(4)]],\n"
    "                            uint gid [[thread_position_in_grid]]) {\n"
    "    if (gid >= dim) return;\n"
    "    float ss = 0.0f;\n"
    "    for (uint i = 0; i < partial_count; ++i) ss += partials[i];\n"
    "    float inv = rsqrt(ss + 1.0e-6f);\n"
    "    y[gid] = x[gid] * inv;\n"
    "}\n"
    "kernel void qwen36_delta_mem(device float *state [[buffer(0)]],\n"
    "                              device const float *key [[buffer(1)]],\n"
    "                              device float *kv_mem [[buffer(2)]],\n"
    "                              constant uint &dim [[buffer(3)]],\n"
    "                              constant float &decay [[buffer(4)]],\n"
    "                              constant uint &group_size [[buffer(5)]],\n"
    "                              uint tid [[thread_index_in_threadgroup]],\n"
    "                              uint2 group_pos [[threadgroup_position_in_grid]]) {\n"
    "    threadgroup float sums[256];\n"
    "    uint vd = group_pos.y;\n"
    "    float v = 0.0f;\n"
    "    if (vd < dim && tid < dim) {\n"
    "        uint idx = tid * dim + vd;\n"
    "        float s = state[idx] * decay;\n"
    "        state[idx] = s;\n"
    "        v = s * key[tid];\n"
    "    }\n"
    "    sums[tid] = v;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint offset = group_size >> 1; offset > 0; offset >>= 1) {\n"
    "        if (tid < offset) sums[tid] += sums[tid + offset];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if (tid == 0 && vd < dim) kv_mem[vd] = sums[0];\n"
    "}\n"
    "kernel void qwen36_delta_update(device float *state [[buffer(0)]],\n"
    "                                device const float *key [[buffer(1)]],\n"
    "                                device const float *value [[buffer(2)]],\n"
    "                                device const float *kv_mem [[buffer(3)]],\n"
    "                                constant uint &dim [[buffer(4)]],\n"
    "                                constant float &beta [[buffer(5)]],\n"
    "                                uint gid [[thread_position_in_grid]]) {\n"
    "    uint total = dim * dim;\n"
    "    if (gid >= total) return;\n"
    "    uint kd = gid / dim;\n"
    "    uint vd = gid - kd * dim;\n"
    "    float delta = (value[vd] - kv_mem[vd]) * beta;\n"
    "    state[gid] += key[kd] * delta;\n"
    "}\n"
    "kernel void qwen36_delta_core(device const float *state [[buffer(0)]],\n"
    "                              device const float *query [[buffer(1)]],\n"
    "                              device float *out [[buffer(2)]],\n"
    "                              constant uint &dim [[buffer(3)]],\n"
    "                              constant float &qscale [[buffer(4)]],\n"
    "                              constant uint &group_size [[buffer(5)]],\n"
    "                              uint tid [[thread_index_in_threadgroup]],\n"
    "                              uint2 group_pos [[threadgroup_position_in_grid]]) {\n"
    "    threadgroup float sums[256];\n"
    "    uint vd = group_pos.y;\n"
    "    float v = 0.0f;\n"
    "    if (vd < dim && tid < dim) {\n"
    "        v = state[tid * dim + vd] * (query[tid] * qscale);\n"
    "    }\n"
    "    sums[tid] = v;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint offset = group_size >> 1; offset > 0; offset >>= 1) {\n"
    "        if (tid < offset) sums[tid] += sums[tid + offset];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if (tid == 0 && vd < dim) out[vd] = sums[0];\n"
    "}\n"
    "kernel void qwen36_attention_scores(device const float *query [[buffer(0)]],\n"
    "                                    device const float *keys [[buffer(1)]],\n"
    "                                    device float *scores [[buffer(2)]],\n"
    "                                    constant uint &seq_len [[buffer(3)]],\n"
    "                                    constant uint &kv_stride [[buffer(4)]],\n"
    "                                    constant uint &head_dim [[buffer(5)]],\n"
    "                                    uint t [[thread_position_in_grid]]) {\n"
    "    if (t >= seq_len) return;\n"
    "    float sum = 0.0f;\n"
    "    uint base = t * kv_stride;\n"
    "    for (uint d = 0; d < head_dim; ++d) {\n"
    "        sum = fma(query[d], keys[base + d], sum);\n"
    "    }\n"
    "    scores[t] = sum * rsqrt(float(head_dim));\n"
    "}\n"
    "kernel void qwen36_attention_apply(device const float *scores [[buffer(0)]],\n"
    "                                   device const float *values [[buffer(1)]],\n"
    "                                   device const float *gate [[buffer(2)]],\n"
    "                                   device float *out [[buffer(3)]],\n"
    "                                   constant uint &seq_len [[buffer(4)]],\n"
    "                                   constant uint &kv_stride [[buffer(5)]],\n"
    "                                   constant uint &head_dim [[buffer(6)]],\n"
    "                                   uint d [[thread_position_in_grid]]) {\n"
    "    if (d >= head_dim) return;\n"
    "    float max_score = scores[0];\n"
    "    for (uint t = 1; t < seq_len; ++t) max_score = max(max_score, scores[t]);\n"
    "    float denom = 0.0f;\n"
    "    float sum = 0.0f;\n"
    "    for (uint t = 0; t < seq_len; ++t) {\n"
    "        float p = exp(scores[t] - max_score);\n"
    "        denom += p;\n"
    "        sum = fma(p, values[t * kv_stride + d], sum);\n"
    "    }\n"
    "    float g = gate[d];\n"
    "    float z = g >= 0.0f ? exp(-g) : exp(g);\n"
    "    float sig = g >= 0.0f ? (1.0f / (1.0f + z)) : (z / (1.0f + z));\n"
    "    out[d] = (sum / denom) * sig;\n"
    "}\n";
}

bool qwen36_metal_available(void) {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

static id<MTLComputePipelineState> qwen36_metal_pipeline(id<MTLLibrary> lib,
                                                         NSString *name,
                                                         char *err,
                                                         size_t errlen) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        qwen36_metal_set_err(err, errlen, "Qwen Metal kernel '%s' is missing",
                             [name UTF8String]);
        return nil;
    }
    NSError *ns_err = nil;
    id<MTLComputePipelineState> pipe =
        [lib.device newComputePipelineStateWithFunction:fn error:&ns_err];
    if (!pipe) {
        qwen36_metal_set_err(err, errlen,
                             "failed to create Qwen Metal pipeline '%s': %s",
                             [name UTF8String],
                             ns_err ? [[ns_err localizedDescription] UTF8String] : "unknown error");
    }
    return pipe;
}

static NSUInteger qwen36_metal_power2_threadgroup(id<MTLComputePipelineState> pipe,
                                                  NSUInteger desired) {
    if (!pipe) return 0;
    NSUInteger max_threads = pipe.maxTotalThreadsPerThreadgroup;
    if (max_threads > desired) max_threads = desired;
    NSUInteger out = 1;
    while ((out << 1) <= max_threads) out <<= 1;
    return out;
}

static bool qwen36_metal_matvec_raw(qwen36_metal_backend *backend,
                                    const void *weights,
                                    uint64_t weight_bytes,
                                    uint32_t tensor_type,
                                    const float *x,
                                    uint64_t in_dim,
                                    float *out,
                                    uint64_t out_dim,
                                    char *err,
                                    size_t errlen) {
    if (!backend || !weights || !x || !out || in_dim == 0 || out_dim == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        weight_bytes > (uint64_t)NSUIntegerMax ||
        in_dim > UINT64_MAX / out_dim) {
        qwen36_metal_set_err(err, errlen, "invalid Qwen Metal matvec request");
        return false;
    }

    id<MTLComputePipelineState> pipe = nil;
    switch (tensor_type) {
    case 0: pipe = backend->matvec_f32; break;
    case 1: pipe = backend->matvec_f16; break;
    case 2:
        if (in_dim % 32u != 0) return false;
        pipe = backend->matvec_q4_0;
        break;
    case 3:
        if (in_dim % 32u != 0) return false;
        pipe = backend->matvec_q4_1;
        break;
    case 6:
        if (in_dim % 32u != 0) return false;
        pipe = backend->matvec_q5_0;
        break;
    case 7:
        if (in_dim % 32u != 0) return false;
        pipe = backend->matvec_q5_1;
        break;
    case 8:
        if (in_dim % 32u != 0) return false;
        pipe = backend->matvec_q8_0;
        break;
    case 9:
        if (in_dim % 32u != 0) return false;
        pipe = backend->matvec_q8_1;
        break;
    case 20:
        if (in_dim % 32u != 0) return false;
        pipe = backend->matvec_iq4_nl;
        break;
    case 23:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_iq4_xs;
        break;
    case 10:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_q2_k;
        break;
    case 11:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_q3_k;
        break;
    case 12:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_q4_k;
        break;
    case 13:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_q5_k;
        break;
    case 14:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_q6_k;
        break;
    case 15:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_q8_k;
        break;
    case 16:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_iq2_xxs;
        break;
    case 17:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_iq2_xs;
        break;
    case 18:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_iq3_xxs;
        break;
    case 19:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_iq1_s;
        break;
    case 29:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_iq1_m;
        break;
    case 22:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_iq2_s;
        break;
    case 21:
        if (in_dim % 256u != 0) return false;
        pipe = backend->matvec_iq3_s;
        break;
    case 30: pipe = backend->matvec_bf16; break;
    default:
        return false;
    }
    if (!pipe) return false;

    uint64_t needed_bytes = 0;
    switch (tensor_type) {
    case 0:
        if (in_dim > UINT64_MAX / out_dim ||
            in_dim * out_dim > UINT64_MAX / sizeof(float)) {
            return false;
        }
        needed_bytes = in_dim * out_dim * sizeof(float);
        break;
    case 1:
    case 30:
        if (in_dim > UINT64_MAX / out_dim ||
            in_dim * out_dim > UINT64_MAX / sizeof(uint16_t)) {
            return false;
        }
        needed_bytes = in_dim * out_dim * sizeof(uint16_t);
        break;
    case 2:
        if (out_dim > UINT64_MAX / (in_dim / 32u) ||
            out_dim * (in_dim / 32u) > UINT64_MAX / 18u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 32u) * 18u;
        break;
    case 3:
        if (out_dim > UINT64_MAX / (in_dim / 32u) ||
            out_dim * (in_dim / 32u) > UINT64_MAX / 20u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 32u) * 20u;
        break;
    case 6:
        if (out_dim > UINT64_MAX / (in_dim / 32u) ||
            out_dim * (in_dim / 32u) > UINT64_MAX / 22u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 32u) * 22u;
        break;
    case 7:
        if (out_dim > UINT64_MAX / (in_dim / 32u) ||
            out_dim * (in_dim / 32u) > UINT64_MAX / 24u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 32u) * 24u;
        break;
    case 8:
        if (out_dim > UINT64_MAX / (in_dim / 32u) ||
            out_dim * (in_dim / 32u) > UINT64_MAX / 34u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 32u) * 34u;
        break;
    case 9:
        if (out_dim > UINT64_MAX / (in_dim / 32u) ||
            out_dim * (in_dim / 32u) > UINT64_MAX / 36u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 32u) * 36u;
        break;
    case 20:
        if (out_dim > UINT64_MAX / (in_dim / 32u) ||
            out_dim * (in_dim / 32u) > UINT64_MAX / 18u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 32u) * 18u;
        break;
    case 10:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 84u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 84u;
        break;
    case 11:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 110u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 110u;
        break;
    case 12:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 144u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 144u;
        break;
    case 13:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 176u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 176u;
        break;
    case 14:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 210u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 210u;
        break;
    case 15:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 292u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 292u;
        break;
    case 16:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 66u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 66u;
        break;
    case 17:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 74u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 74u;
        break;
    case 18:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 98u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 98u;
        break;
    case 19:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 50u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 50u;
        break;
    case 29:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 56u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 56u;
        break;
    case 22:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 82u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 82u;
        break;
    case 21:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 110u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 110u;
        break;
    case 23:
        if (out_dim > UINT64_MAX / (in_dim / 256u) ||
            out_dim * (in_dim / 256u) > UINT64_MAX / 136u) {
            return false;
        }
        needed_bytes = out_dim * (in_dim / 256u) * 136u;
        break;
    default:
        return false;
    }
    if (needed_bytes > weight_bytes) return false;

    const size_t x_bytes = (size_t)in_dim * sizeof(float);
    const size_t out_bytes = (size_t)out_dim * sizeof(float);
    id<MTLBuffer> wbuf =
        [backend->device newBufferWithBytesNoCopy:(void *)weights
                                           length:(NSUInteger)weight_bytes
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
    if (!wbuf) {
        wbuf = [backend->device newBufferWithBytes:weights
                                            length:(NSUInteger)weight_bytes
                                           options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> xbuf = [backend->device newBufferWithBytes:x
                                                      length:x_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> ybuf = [backend->device newBufferWithLength:out_bytes
                                                      options:MTLResourceStorageModeShared];
    if (!wbuf || !xbuf || !ybuf) {
        qwen36_metal_set_err(err, errlen, "failed to allocate Qwen Metal buffers");
        return false;
    }

    uint32_t in32 = (uint32_t)in_dim;
    uint32_t out32 = (uint32_t)out_dim;
    id<MTLCommandBuffer> cb = [backend->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!cb || !enc) {
        qwen36_metal_set_err(err, errlen, "failed to encode Qwen Metal matvec");
        return false;
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:wbuf offset:0 atIndex:0];
    [enc setBuffer:xbuf offset:0 atIndex:1];
    [enc setBuffer:ybuf offset:0 atIndex:2];
    [enc setBytes:&in32 length:sizeof(in32) atIndex:3];
    [enc setBytes:&out32 length:sizeof(out32) atIndex:4];
    if (tensor_type == 16) {
        [enc setBytes:qwen36_iq2xxs_grid
               length:sizeof(qwen36_iq2xxs_grid)
              atIndex:5];
    } else if (tensor_type == 17) {
        [enc setBytes:rt_iq2xs_grid
               length:sizeof(uint64_t) * 512u
              atIndex:5];
    } else if (tensor_type == 18) {
        [enc setBytes:rt_iq3xxs_grid
               length:sizeof(uint32_t) * 256u
              atIndex:5];
    } else if (tensor_type == 19) {
        [enc setBytes:rt_iq1s_grid
               length:sizeof(uint64_t) * 2048u
              atIndex:5];
    } else if (tensor_type == 29) {
        [enc setBytes:rt_iq1s_grid
               length:sizeof(uint64_t) * 2048u
              atIndex:5];
    } else if (tensor_type == 22) {
        [enc setBytes:rt_iq2s_grid
               length:sizeof(uint64_t) * 1024u
              atIndex:5];
    } else if (tensor_type == 21) {
        [enc setBytes:rt_iq3s_grid
               length:sizeof(uint32_t) * 512u
              atIndex:5];
    }
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    if (tg == 0) tg = 1;
    MTLSize grid = MTLSizeMake((NSUInteger)out_dim, 1, 1);
    MTLSize group = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        NSError *ns_err = cb.error;
        qwen36_metal_set_err(err, errlen, "Qwen Metal matvec failed: %s",
                             ns_err ? [[ns_err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    memcpy(out, [ybuf contents], out_bytes);
    return true;
}

static bool qwen36_metal_rms_norm_raw(qwen36_metal_backend *backend,
                                      const void *weights,
                                      uint64_t weight_bytes,
                                      uint32_t tensor_type,
                                      const float *x,
                                      uint64_t dim,
                                      float *out,
                                      bool add_one_to_weight,
                                      char *err,
                                      size_t errlen) {
    if (!backend || !weights || !x || !out || dim == 0 ||
        dim > UINT32_MAX ||
        weight_bytes > (uint64_t)NSUIntegerMax ||
        dim > (uint64_t)SIZE_MAX / sizeof(float)) {
        qwen36_metal_set_err(err, errlen, "invalid Qwen Metal RMSNorm request");
        return false;
    }

    id<MTLComputePipelineState> apply_pipe = nil;
    switch (tensor_type) {
    case 0: apply_pipe = backend->rms_apply_f32; break;
    case 1: apply_pipe = backend->rms_apply_f16; break;
    case 30: apply_pipe = backend->rms_apply_bf16; break;
    default:
        return false;
    }
    if (!backend->rms_sum || !apply_pipe) return false;

    NSUInteger group = qwen36_metal_power2_threadgroup(backend->rms_sum, 256);
    if (group == 0 || group > UINT32_MAX) {
        qwen36_metal_set_err(err, errlen, "invalid Qwen Metal RMSNorm threadgroup");
        return false;
    }
    uint64_t partial_count64 = (dim + (uint64_t)group - 1) / (uint64_t)group;
    if (partial_count64 == 0 || partial_count64 > UINT32_MAX ||
        partial_count64 > (uint64_t)SIZE_MAX / sizeof(float)) {
        qwen36_metal_set_err(err, errlen, "invalid Qwen Metal RMSNorm partial count");
        return false;
    }

    size_t x_bytes = (size_t)dim * sizeof(float);
    size_t out_bytes = (size_t)dim * sizeof(float);
    size_t partial_bytes = (size_t)partial_count64 * sizeof(float);
    id<MTLBuffer> wbuf =
        [backend->device newBufferWithBytesNoCopy:(void *)weights
                                           length:(NSUInteger)weight_bytes
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
    if (!wbuf) {
        wbuf = [backend->device newBufferWithBytes:weights
                                            length:(NSUInteger)weight_bytes
                                           options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> xbuf = [backend->device newBufferWithBytes:x
                                                      length:x_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> pbuf = [backend->device newBufferWithLength:partial_bytes
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> ybuf = [backend->device newBufferWithLength:out_bytes
                                                      options:MTLResourceStorageModeShared];
    if (!wbuf || !xbuf || !pbuf || !ybuf) {
        qwen36_metal_set_err(err, errlen, "failed to allocate Qwen Metal RMSNorm buffers");
        return false;
    }

    uint32_t dim32 = (uint32_t)dim;
    uint32_t group32 = (uint32_t)group;
    uint32_t partial_count32 = (uint32_t)partial_count64;
    uint32_t add_one32 = add_one_to_weight ? 1u : 0u;
    id<MTLCommandBuffer> cb = [backend->queue commandBuffer];
    if (!cb) {
        qwen36_metal_set_err(err, errlen, "failed to create Qwen Metal RMSNorm command buffer");
        return false;
    }

    id<MTLComputeCommandEncoder> sum_enc = [cb computeCommandEncoder];
    if (!sum_enc) {
        qwen36_metal_set_err(err, errlen, "failed to encode Qwen Metal RMSNorm sum");
        return false;
    }
    [sum_enc setComputePipelineState:backend->rms_sum];
    [sum_enc setBuffer:xbuf offset:0 atIndex:0];
    [sum_enc setBuffer:pbuf offset:0 atIndex:1];
    [sum_enc setBytes:&dim32 length:sizeof(dim32) atIndex:2];
    [sum_enc setBytes:&group32 length:sizeof(group32) atIndex:3];
    MTLSize sum_grid = MTLSizeMake((NSUInteger)partial_count64 * group, 1, 1);
    MTLSize sum_group = MTLSizeMake(group, 1, 1);
    [sum_enc dispatchThreads:sum_grid threadsPerThreadgroup:sum_group];
    [sum_enc endEncoding];

    id<MTLComputeCommandEncoder> apply_enc = [cb computeCommandEncoder];
    if (!apply_enc) {
        qwen36_metal_set_err(err, errlen, "failed to encode Qwen Metal RMSNorm apply");
        return false;
    }
    [apply_enc setComputePipelineState:apply_pipe];
    [apply_enc setBuffer:wbuf offset:0 atIndex:0];
    [apply_enc setBuffer:xbuf offset:0 atIndex:1];
    [apply_enc setBuffer:pbuf offset:0 atIndex:2];
    [apply_enc setBuffer:ybuf offset:0 atIndex:3];
    [apply_enc setBytes:&dim32 length:sizeof(dim32) atIndex:4];
    [apply_enc setBytes:&partial_count32 length:sizeof(partial_count32) atIndex:5];
    [apply_enc setBytes:&add_one32 length:sizeof(add_one32) atIndex:6];
    NSUInteger apply_group = qwen36_metal_power2_threadgroup(apply_pipe, 256);
    if (apply_group == 0) apply_group = 1;
    MTLSize apply_grid = MTLSizeMake((NSUInteger)dim, 1, 1);
    MTLSize apply_tg = MTLSizeMake(apply_group, 1, 1);
    [apply_enc dispatchThreads:apply_grid threadsPerThreadgroup:apply_tg];
    [apply_enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        NSError *ns_err = cb.error;
        qwen36_metal_set_err(err, errlen, "Qwen Metal RMSNorm failed: %s",
                             ns_err ? [[ns_err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    memcpy(out, [ybuf contents], out_bytes);
    return true;
}

static bool qwen36_metal_silu_mul_raw(qwen36_metal_backend *backend,
                                      const float *gate,
                                      const float *up,
                                      uint64_t dim,
                                      float *out,
                                      char *err,
                                      size_t errlen) {
    if (!backend || !backend->silu_mul || !gate || !up || !out || dim == 0 ||
        dim > UINT32_MAX ||
        dim > (uint64_t)SIZE_MAX / sizeof(float)) {
        qwen36_metal_set_err(err, errlen, "invalid Qwen Metal SiLU-mul request");
        return false;
    }
    size_t bytes = (size_t)dim * sizeof(float);
    id<MTLBuffer> gate_buf = [backend->device newBufferWithBytes:gate
                                                          length:bytes
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> up_buf = [backend->device newBufferWithBytes:up
                                                        length:bytes
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> ybuf = [backend->device newBufferWithLength:bytes
                                                      options:MTLResourceStorageModeShared];
    if (!gate_buf || !up_buf || !ybuf) {
        qwen36_metal_set_err(err, errlen, "failed to allocate Qwen Metal SiLU-mul buffers");
        return false;
    }

    uint32_t dim32 = (uint32_t)dim;
    id<MTLCommandBuffer> cb = [backend->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!cb || !enc) {
        qwen36_metal_set_err(err, errlen, "failed to encode Qwen Metal SiLU-mul");
        return false;
    }
    [enc setComputePipelineState:backend->silu_mul];
    [enc setBuffer:gate_buf offset:0 atIndex:0];
    [enc setBuffer:up_buf offset:0 atIndex:1];
    [enc setBuffer:ybuf offset:0 atIndex:2];
    [enc setBytes:&dim32 length:sizeof(dim32) atIndex:3];
    NSUInteger tg = qwen36_metal_power2_threadgroup(backend->silu_mul, 256);
    if (tg == 0) tg = 1;
    MTLSize grid = MTLSizeMake((NSUInteger)dim, 1, 1);
    MTLSize group = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        NSError *ns_err = cb.error;
        qwen36_metal_set_err(err, errlen, "Qwen Metal SiLU-mul failed: %s",
                             ns_err ? [[ns_err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    memcpy(out, [ybuf contents], bytes);
    return true;
}

static bool qwen36_metal_l2_norm_raw(qwen36_metal_backend *backend,
                                     const float *x,
                                     uint64_t dim,
                                     float *out,
                                     char *err,
                                     size_t errlen) {
    if (!backend || !backend->l2_sum || !backend->l2_apply || !x || !out ||
        dim == 0 || dim > UINT32_MAX ||
        dim > (uint64_t)SIZE_MAX / sizeof(float)) {
        qwen36_metal_set_err(err, errlen, "invalid Qwen Metal L2Norm request");
        return false;
    }
    bool nonzero = false;
    for (uint64_t i = 0; i < dim; i++) {
        if (x[i] != 0.0f) {
            nonzero = true;
            break;
        }
    }
    if (!nonzero) {
        qwen36_metal_set_err(err, errlen, "cannot L2-normalize a zero vector");
        return false;
    }

    NSUInteger group = qwen36_metal_power2_threadgroup(backend->l2_sum, 256);
    if (group == 0 || group > UINT32_MAX) {
        qwen36_metal_set_err(err, errlen, "invalid Qwen Metal L2Norm threadgroup");
        return false;
    }
    uint64_t partial_count64 = (dim + (uint64_t)group - 1) / (uint64_t)group;
    if (partial_count64 == 0 || partial_count64 > UINT32_MAX ||
        partial_count64 > (uint64_t)SIZE_MAX / sizeof(float)) {
        qwen36_metal_set_err(err, errlen, "invalid Qwen Metal L2Norm partial count");
        return false;
    }

    size_t bytes = (size_t)dim * sizeof(float);
    size_t partial_bytes = (size_t)partial_count64 * sizeof(float);
    id<MTLBuffer> xbuf = [backend->device newBufferWithBytes:x
                                                      length:bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> pbuf = [backend->device newBufferWithLength:partial_bytes
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> ybuf = [backend->device newBufferWithLength:bytes
                                                      options:MTLResourceStorageModeShared];
    if (!xbuf || !pbuf || !ybuf) {
        qwen36_metal_set_err(err, errlen, "failed to allocate Qwen Metal L2Norm buffers");
        return false;
    }

    uint32_t dim32 = (uint32_t)dim;
    uint32_t group32 = (uint32_t)group;
    uint32_t partial_count32 = (uint32_t)partial_count64;
    id<MTLCommandBuffer> cb = [backend->queue commandBuffer];
    if (!cb) {
        qwen36_metal_set_err(err, errlen, "failed to create Qwen Metal L2Norm command buffer");
        return false;
    }

    id<MTLComputeCommandEncoder> sum_enc = [cb computeCommandEncoder];
    if (!sum_enc) {
        qwen36_metal_set_err(err, errlen, "failed to encode Qwen Metal L2Norm sum");
        return false;
    }
    [sum_enc setComputePipelineState:backend->l2_sum];
    [sum_enc setBuffer:xbuf offset:0 atIndex:0];
    [sum_enc setBuffer:pbuf offset:0 atIndex:1];
    [sum_enc setBytes:&dim32 length:sizeof(dim32) atIndex:2];
    [sum_enc setBytes:&group32 length:sizeof(group32) atIndex:3];
    MTLSize sum_grid = MTLSizeMake((NSUInteger)partial_count64 * group, 1, 1);
    MTLSize sum_group = MTLSizeMake(group, 1, 1);
    [sum_enc dispatchThreads:sum_grid threadsPerThreadgroup:sum_group];
    [sum_enc endEncoding];

    id<MTLComputeCommandEncoder> apply_enc = [cb computeCommandEncoder];
    if (!apply_enc) {
        qwen36_metal_set_err(err, errlen, "failed to encode Qwen Metal L2Norm apply");
        return false;
    }
    [apply_enc setComputePipelineState:backend->l2_apply];
    [apply_enc setBuffer:xbuf offset:0 atIndex:0];
    [apply_enc setBuffer:pbuf offset:0 atIndex:1];
    [apply_enc setBuffer:ybuf offset:0 atIndex:2];
    [apply_enc setBytes:&dim32 length:sizeof(dim32) atIndex:3];
    [apply_enc setBytes:&partial_count32 length:sizeof(partial_count32) atIndex:4];
    NSUInteger apply_group = qwen36_metal_power2_threadgroup(backend->l2_apply, 256);
    if (apply_group == 0) apply_group = 1;
    MTLSize apply_grid = MTLSizeMake((NSUInteger)dim, 1, 1);
    MTLSize apply_tg = MTLSizeMake(apply_group, 1, 1);
    [apply_enc dispatchThreads:apply_grid threadsPerThreadgroup:apply_tg];
    [apply_enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        NSError *ns_err = cb.error;
        qwen36_metal_set_err(err, errlen, "Qwen Metal L2Norm failed: %s",
                             ns_err ? [[ns_err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    memcpy(out, [ybuf contents], bytes);
    return true;
}

static bool qwen36_metal_gated_delta_head_raw(qwen36_metal_backend *backend,
                                              float *state,
                                              const float *query,
                                              const float *key,
                                              const float *value,
                                              uint64_t dim,
                                              float decay,
                                              float beta,
                                              float qscale,
                                              float *out,
                                              char *err,
                                              size_t errlen) {
    if (!backend || !backend->delta_mem || !backend->delta_update ||
        !backend->delta_core || !state || !query || !key || !value || !out ||
        dim == 0 || dim > 256u ||
        dim > (uint64_t)SIZE_MAX / dim ||
        dim * dim > (uint64_t)SIZE_MAX / sizeof(float)) {
        qwen36_metal_set_err(err, errlen, "invalid Qwen Metal Gated DeltaNet request");
        return false;
    }

    const size_t vec_bytes = (size_t)dim * sizeof(float);
    const uint64_t matrix_elems = dim * dim;
    const size_t state_bytes = (size_t)matrix_elems * sizeof(float);
    id<MTLBuffer> state_buf = [backend->device newBufferWithBytes:state
                                                           length:state_bytes
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_buf = [backend->device newBufferWithBytes:query
                                                          length:vec_bytes
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> key_buf = [backend->device newBufferWithBytes:key
                                                        length:vec_bytes
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> value_buf = [backend->device newBufferWithBytes:value
                                                          length:vec_bytes
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> kv_buf = [backend->device newBufferWithLength:vec_bytes
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> ybuf = [backend->device newBufferWithLength:vec_bytes
                                                      options:MTLResourceStorageModeShared];
    if (!state_buf || !query_buf || !key_buf || !value_buf || !kv_buf || !ybuf) {
        qwen36_metal_set_err(err, errlen, "failed to allocate Qwen Metal Gated DeltaNet buffers");
        return false;
    }

    NSUInteger reduce_group = qwen36_metal_power2_threadgroup(backend->delta_mem, 256);
    NSUInteger core_group = qwen36_metal_power2_threadgroup(backend->delta_core, 256);
    if (reduce_group < dim || core_group < dim ||
        reduce_group > UINT32_MAX || core_group > UINT32_MAX) {
        qwen36_metal_set_err(err, errlen, "invalid Qwen Metal Gated DeltaNet threadgroup");
        return false;
    }
    NSUInteger update_group = qwen36_metal_power2_threadgroup(backend->delta_update, 256);
    if (update_group == 0) update_group = 1;

    uint32_t dim32 = (uint32_t)dim;
    uint32_t reduce_group32 = (uint32_t)reduce_group;
    uint32_t core_group32 = (uint32_t)core_group;
    id<MTLCommandBuffer> cb = [backend->queue commandBuffer];
    if (!cb) {
        qwen36_metal_set_err(err, errlen, "failed to create Qwen Metal Gated DeltaNet command buffer");
        return false;
    }

    id<MTLComputeCommandEncoder> mem_enc = [cb computeCommandEncoder];
    if (!mem_enc) {
        qwen36_metal_set_err(err, errlen, "failed to encode Qwen Metal Gated DeltaNet memory read");
        return false;
    }
    [mem_enc setComputePipelineState:backend->delta_mem];
    [mem_enc setBuffer:state_buf offset:0 atIndex:0];
    [mem_enc setBuffer:key_buf offset:0 atIndex:1];
    [mem_enc setBuffer:kv_buf offset:0 atIndex:2];
    [mem_enc setBytes:&dim32 length:sizeof(dim32) atIndex:3];
    [mem_enc setBytes:&decay length:sizeof(decay) atIndex:4];
    [mem_enc setBytes:&reduce_group32 length:sizeof(reduce_group32) atIndex:5];
    MTLSize reduce_grid = MTLSizeMake(reduce_group, (NSUInteger)dim, 1);
    MTLSize reduce_tg = MTLSizeMake(reduce_group, 1, 1);
    [mem_enc dispatchThreads:reduce_grid threadsPerThreadgroup:reduce_tg];
    [mem_enc endEncoding];

    id<MTLComputeCommandEncoder> update_enc = [cb computeCommandEncoder];
    if (!update_enc) {
        qwen36_metal_set_err(err, errlen, "failed to encode Qwen Metal Gated DeltaNet update");
        return false;
    }
    [update_enc setComputePipelineState:backend->delta_update];
    [update_enc setBuffer:state_buf offset:0 atIndex:0];
    [update_enc setBuffer:key_buf offset:0 atIndex:1];
    [update_enc setBuffer:value_buf offset:0 atIndex:2];
    [update_enc setBuffer:kv_buf offset:0 atIndex:3];
    [update_enc setBytes:&dim32 length:sizeof(dim32) atIndex:4];
    [update_enc setBytes:&beta length:sizeof(beta) atIndex:5];
    MTLSize update_grid = MTLSizeMake((NSUInteger)matrix_elems, 1, 1);
    MTLSize update_tg = MTLSizeMake(update_group, 1, 1);
    [update_enc dispatchThreads:update_grid threadsPerThreadgroup:update_tg];
    [update_enc endEncoding];

    id<MTLComputeCommandEncoder> core_enc = [cb computeCommandEncoder];
    if (!core_enc) {
        qwen36_metal_set_err(err, errlen, "failed to encode Qwen Metal Gated DeltaNet core");
        return false;
    }
    [core_enc setComputePipelineState:backend->delta_core];
    [core_enc setBuffer:state_buf offset:0 atIndex:0];
    [core_enc setBuffer:query_buf offset:0 atIndex:1];
    [core_enc setBuffer:ybuf offset:0 atIndex:2];
    [core_enc setBytes:&dim32 length:sizeof(dim32) atIndex:3];
    [core_enc setBytes:&qscale length:sizeof(qscale) atIndex:4];
    [core_enc setBytes:&core_group32 length:sizeof(core_group32) atIndex:5];
    MTLSize core_grid = MTLSizeMake(core_group, (NSUInteger)dim, 1);
    MTLSize core_tg = MTLSizeMake(core_group, 1, 1);
    [core_enc dispatchThreads:core_grid threadsPerThreadgroup:core_tg];
    [core_enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        NSError *ns_err = cb.error;
        qwen36_metal_set_err(err, errlen, "Qwen Metal Gated DeltaNet failed: %s",
                             ns_err ? [[ns_err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    memcpy(state, [state_buf contents], state_bytes);
    memcpy(out, [ybuf contents], vec_bytes);
    return true;
}

static bool qwen36_metal_full_attention_head_raw(qwen36_metal_backend *backend,
                                                 const float *query,
                                                 const float *gate,
                                                 const float *keys,
                                                 const float *values,
                                                 uint64_t seq_len,
                                                 uint64_t kv_stride,
                                                 uint64_t head_dim,
                                                 float *out,
                                                 char *err,
                                                 size_t errlen) {
    if (!backend || !backend->attention_scores || !backend->attention_apply ||
        !query || !gate || !keys || !values || !out ||
        seq_len == 0 || kv_stride == 0 || head_dim == 0 ||
        head_dim > kv_stride ||
        seq_len > UINT32_MAX || kv_stride > UINT32_MAX ||
        head_dim > UINT32_MAX ||
        head_dim > (uint64_t)SIZE_MAX / sizeof(float) ||
        seq_len > (uint64_t)SIZE_MAX / sizeof(float) ||
        (seq_len - 1u) > (UINT64_MAX - head_dim) / kv_stride) {
        qwen36_metal_set_err(err, errlen,
                             "invalid Qwen Metal full-attention request");
        return false;
    }

    const uint64_t strided_elems = (seq_len - 1u) * kv_stride + head_dim;
    if (strided_elems > (uint64_t)SIZE_MAX / sizeof(float)) {
        qwen36_metal_set_err(err, errlen,
                             "Qwen Metal full-attention buffer is too large");
        return false;
    }
    const size_t vec_bytes = (size_t)head_dim * sizeof(float);
    const size_t score_bytes = (size_t)seq_len * sizeof(float);
    const size_t state_bytes = (size_t)strided_elems * sizeof(float);
    id<MTLBuffer> query_buf = [backend->device newBufferWithBytes:query
                                                           length:vec_bytes
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate_buf = [backend->device newBufferWithBytes:gate
                                                         length:vec_bytes
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> key_buf = [backend->device newBufferWithBytesNoCopy:(void *)keys
                                                               length:state_bytes
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
    if (!key_buf) {
        key_buf = [backend->device newBufferWithBytes:keys
                                               length:state_bytes
                                              options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> value_buf = [backend->device newBufferWithBytesNoCopy:(void *)values
                                                                 length:state_bytes
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
    if (!value_buf) {
        value_buf = [backend->device newBufferWithBytes:values
                                                 length:state_bytes
                                                options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> score_buf = [backend->device newBufferWithLength:score_bytes
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf = [backend->device newBufferWithLength:vec_bytes
                                                         options:MTLResourceStorageModeShared];
    if (!query_buf || !gate_buf || !key_buf || !value_buf ||
        !score_buf || !out_buf) {
        qwen36_metal_set_err(err, errlen,
                             "failed to allocate Qwen Metal full-attention buffers");
        return false;
    }

    uint32_t seq32 = (uint32_t)seq_len;
    uint32_t stride32 = (uint32_t)kv_stride;
    uint32_t head32 = (uint32_t)head_dim;
    id<MTLCommandBuffer> cb = [backend->queue commandBuffer];
    if (!cb) {
        qwen36_metal_set_err(err, errlen,
                             "failed to create Qwen Metal full-attention command buffer");
        return false;
    }

    id<MTLComputeCommandEncoder> score_enc = [cb computeCommandEncoder];
    if (!score_enc) {
        qwen36_metal_set_err(err, errlen,
                             "failed to encode Qwen Metal attention scores");
        return false;
    }
    [score_enc setComputePipelineState:backend->attention_scores];
    [score_enc setBuffer:query_buf offset:0 atIndex:0];
    [score_enc setBuffer:key_buf offset:0 atIndex:1];
    [score_enc setBuffer:score_buf offset:0 atIndex:2];
    [score_enc setBytes:&seq32 length:sizeof(seq32) atIndex:3];
    [score_enc setBytes:&stride32 length:sizeof(stride32) atIndex:4];
    [score_enc setBytes:&head32 length:sizeof(head32) atIndex:5];
    NSUInteger score_group = qwen36_metal_power2_threadgroup(
        backend->attention_scores, 256);
    if (score_group == 0) score_group = 1;
    [score_enc dispatchThreads:MTLSizeMake((NSUInteger)seq_len, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(score_group, 1, 1)];
    [score_enc endEncoding];

    id<MTLComputeCommandEncoder> apply_enc = [cb computeCommandEncoder];
    if (!apply_enc) {
        qwen36_metal_set_err(err, errlen,
                             "failed to encode Qwen Metal attention apply");
        return false;
    }
    [apply_enc setComputePipelineState:backend->attention_apply];
    [apply_enc setBuffer:score_buf offset:0 atIndex:0];
    [apply_enc setBuffer:value_buf offset:0 atIndex:1];
    [apply_enc setBuffer:gate_buf offset:0 atIndex:2];
    [apply_enc setBuffer:out_buf offset:0 atIndex:3];
    [apply_enc setBytes:&seq32 length:sizeof(seq32) atIndex:4];
    [apply_enc setBytes:&stride32 length:sizeof(stride32) atIndex:5];
    [apply_enc setBytes:&head32 length:sizeof(head32) atIndex:6];
    NSUInteger apply_group = qwen36_metal_power2_threadgroup(
        backend->attention_apply, 256);
    if (apply_group == 0) apply_group = 1;
    [apply_enc dispatchThreads:MTLSizeMake((NSUInteger)head_dim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(apply_group, 1, 1)];
    [apply_enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        NSError *ns_err = cb.error;
        qwen36_metal_set_err(err, errlen,
                             "Qwen Metal full-attention failed: %s",
                             ns_err ? [[ns_err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    memcpy(out, [out_buf contents], vec_bytes);
    return true;
}

static bool qwen36_metal_self_test(qwen36_metal_backend *backend,
                                   char *err,
                                   size_t errlen) {
    const float weights[] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.5f, 4.0f,
    };
    const float x[] = {2.0f, -1.0f, 0.5f};
    float y[2] = {0.0f, 0.0f};
    if (!qwen36_metal_matvec_raw(backend, weights, sizeof(weights), 0,
                                 x, 3, y, 2, err, errlen)) {
        return false;
    }
    if (fabsf(y[0] - 1.5f) > 0.0001f ||
        fabsf(y[1] + 0.5f) > 0.0001f) {
        qwen36_metal_set_err(err, errlen,
                             "Qwen Metal matvec self-test mismatch: %g %g",
                             (double)y[0], (double)y[1]);
        return false;
    }
    uint8_t q8_0_weights[68] = {0};
    q8_0_weights[0] = 0x00u;
    q8_0_weights[1] = 0x3cu;
    q8_0_weights[2] = 1u;
    q8_0_weights[3] = 2u;
    q8_0_weights[34] = 0x00u;
    q8_0_weights[35] = 0x3cu;
    q8_0_weights[36] = (uint8_t)-1;
    q8_0_weights[37] = 4u;
    float q8_x32[32] = {0.0f};
    q8_x32[0] = 10.0f;
    q8_x32[1] = 1.0f;
    float q8_y[2] = {0.0f, 0.0f};
    if (!qwen36_metal_matvec_raw(backend, q8_0_weights, sizeof(q8_0_weights),
                                 8, q8_x32, 32, q8_y, 2, err, errlen)) {
        return false;
    }
    if (fabsf(q8_y[0] - 12.0f) > 0.0001f ||
        fabsf(q8_y[1] + 6.0f) > 0.0001f) {
        qwen36_metal_set_err(err, errlen,
                             "Qwen Metal q8_0 matvec self-test mismatch: %g %g",
                             (double)q8_y[0], (double)q8_y[1]);
        return false;
    }
    uint8_t q8_k_weights[584] = {0};
    float q8_k_d0 = 0.5f;
    float q8_k_d1 = 1.0f;
    memcpy(q8_k_weights, &q8_k_d0, sizeof(q8_k_d0));
    q8_k_weights[4] = 5u;
    q8_k_weights[259] = (uint8_t)-2;
    memcpy(q8_k_weights + 292, &q8_k_d1, sizeof(q8_k_d1));
    q8_k_weights[292 + 4] = (uint8_t)-2;
    q8_k_weights[292 + 5] = 3u;
    float q8_x256[256] = {0.0f};
    q8_x256[0] = 10.0f;
    q8_x256[1] = 1.0f;
    q8_x256[255] = 2.0f;
    if (!qwen36_metal_matvec_raw(backend, q8_k_weights, sizeof(q8_k_weights),
                                 15, q8_x256, 256, q8_y, 2, err, errlen)) {
        return false;
    }
    if (fabsf(q8_y[0] - 23.0f) > 0.0001f ||
        fabsf(q8_y[1] + 17.0f) > 0.0001f) {
        qwen36_metal_set_err(err, errlen,
                             "Qwen Metal q8_K matvec self-test mismatch: %g %g",
                             (double)q8_y[0], (double)q8_y[1]);
        return false;
    }
    const float rms_weight[] = {1.0f, -2.0f, 0.5f, 3.0f};
    const float rms_x[] = {2.0f, -1.0f, 0.5f, 4.0f};
    float rms_y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (!qwen36_metal_rms_norm_raw(backend, rms_weight, sizeof(rms_weight), 0,
                                   rms_x, 4, rms_y, false, err, errlen)) {
        return false;
    }
    double ss = 0.0;
    for (size_t i = 0; i < 4; ++i) ss += (double)rms_x[i] * (double)rms_x[i];
    float inv = 1.0f / sqrtf((float)(ss / 4.0) + 1.0e-6f);
    for (size_t i = 0; i < 4; ++i) {
        float want = rms_x[i] * inv * rms_weight[i];
        if (fabsf(rms_y[i] - want) > 0.0001f) {
            qwen36_metal_set_err(err, errlen,
                                 "Qwen Metal RMSNorm self-test mismatch at %zu: %g != %g",
                                 i, (double)rms_y[i], (double)want);
            return false;
        }
    }
    const float silu_gate[] = {-4.0f, -1.0f, 0.0f, 2.0f};
    const float silu_up[] = {0.5f, -3.0f, 7.0f, 4.0f};
    float silu_y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (!qwen36_metal_silu_mul_raw(backend, silu_gate, silu_up, 4,
                                   silu_y, err, errlen)) {
        return false;
    }
    for (size_t i = 0; i < 4; ++i) {
        float x0 = silu_gate[i];
        float z = x0 >= 0.0f ? expf(-x0) : expf(x0);
        float sig = x0 >= 0.0f ? (1.0f / (1.0f + z)) : (z / (1.0f + z));
        float want = x0 * sig * silu_up[i];
        if (fabsf(silu_y[i] - want) > 0.0001f) {
            qwen36_metal_set_err(err, errlen,
                                 "Qwen Metal SiLU-mul self-test mismatch at %zu: %g != %g",
                                 i, (double)silu_y[i], (double)want);
            return false;
        }
    }
    const float l2_x[] = {3.0f, 4.0f, 0.0f};
    float l2_y[3] = {0.0f, 0.0f, 0.0f};
    if (!qwen36_metal_l2_norm_raw(backend, l2_x, 3, l2_y, err, errlen)) {
        return false;
    }
    const float l2_want[] = {0.6f, 0.8f, 0.0f};
    for (size_t i = 0; i < 3; ++i) {
        if (fabsf(l2_y[i] - l2_want[i]) > 0.0001f) {
            qwen36_metal_set_err(err, errlen,
                                 "Qwen Metal L2Norm self-test mismatch at %zu: %g != %g",
                                 i, (double)l2_y[i], (double)l2_want[i]);
            return false;
        }
    }
    float delta_state[] = {
        0.5f, -0.25f, 0.75f, 1.0f,
        -1.5f, 0.5f, 0.25f, -0.75f,
        0.0f, 1.25f, -0.5f, 0.5f,
        1.0f, -1.0f, 0.5f, -0.25f,
    };
    float delta_state_want[16];
    memcpy(delta_state_want, delta_state, sizeof(delta_state_want));
    const float delta_query[] = {0.25f, -0.5f, 1.0f, 0.75f};
    const float delta_key[] = {0.5f, -1.0f, 0.25f, 0.75f};
    const float delta_value[] = {1.5f, -0.5f, 0.25f, -1.0f};
    const float decay = 0.6f;
    const float beta = 0.25f;
    const float qscale = 0.5f;
    float delta_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float delta_want[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float delta_mem[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t kd = 0; kd < 4; ++kd) {
        for (size_t vd = 0; vd < 4; ++vd) {
            size_t idx = kd * 4 + vd;
            delta_state_want[idx] *= decay;
            delta_mem[vd] += delta_state_want[idx] * delta_key[kd];
        }
    }
    for (size_t kd = 0; kd < 4; ++kd) {
        for (size_t vd = 0; vd < 4; ++vd) {
            float delta = (delta_value[vd] - delta_mem[vd]) * beta;
            delta_state_want[kd * 4 + vd] += delta_key[kd] * delta;
        }
    }
    for (size_t vd = 0; vd < 4; ++vd) {
        for (size_t kd = 0; kd < 4; ++kd) {
            delta_want[vd] += delta_state_want[kd * 4 + vd] *
                              (delta_query[kd] * qscale);
        }
    }
    if (!qwen36_metal_gated_delta_head_raw(backend, delta_state, delta_query,
                                           delta_key, delta_value, 4,
                                           decay, beta, qscale,
                                           delta_out, err, errlen)) {
        return false;
    }
    for (size_t i = 0; i < 16; ++i) {
        if (fabsf(delta_state[i] - delta_state_want[i]) > 0.0002f) {
            qwen36_metal_set_err(err, errlen,
                                 "Qwen Metal Gated DeltaNet state self-test mismatch at %zu: %g != %g",
                                 i, (double)delta_state[i], (double)delta_state_want[i]);
            return false;
        }
    }
    for (size_t i = 0; i < 4; ++i) {
        if (fabsf(delta_out[i] - delta_want[i]) > 0.0002f) {
            qwen36_metal_set_err(err, errlen,
                                 "Qwen Metal Gated DeltaNet output self-test mismatch at %zu: %g != %g",
                                 i, (double)delta_out[i], (double)delta_want[i]);
            return false;
        }
    }
    const float attn_query[] = {0.5f, -0.25f, 0.75f, 1.0f};
    const float attn_gate[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    const float attn_keys[] = {
        1.0f, 0.0f, -0.5f, 0.25f,
        0.25f, 1.0f, 0.5f, -0.75f,
        -1.0f, 0.5f, 0.25f, 1.0f,
    };
    const float attn_values[] = {
        0.5f, 1.0f, -1.0f, 0.25f,
        -0.25f, 0.75f, 0.5f, 1.25f,
        1.0f, -0.5f, 0.25f, -0.75f,
    };
    float attn_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float attn_scores[3] = {0.0f, 0.0f, 0.0f};
    float attn_max = -FLT_MAX;
    for (size_t t = 0; t < 3; ++t) {
        float dot = 0.0f;
        for (size_t d = 0; d < 4; ++d) {
            dot += attn_query[d] * attn_keys[t * 4 + d];
        }
        attn_scores[t] = dot / sqrtf(4.0f);
        if (attn_scores[t] > attn_max) attn_max = attn_scores[t];
    }
    float attn_denom = 0.0f;
    for (size_t t = 0; t < 3; ++t) {
        attn_scores[t] = expf(attn_scores[t] - attn_max);
        attn_denom += attn_scores[t];
    }
    float attn_want[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t d = 0; d < 4; ++d) {
        float sum = 0.0f;
        for (size_t t = 0; t < 3; ++t) {
            sum += (attn_scores[t] / attn_denom) * attn_values[t * 4 + d];
        }
        float g = attn_gate[d];
        float z = g >= 0.0f ? expf(-g) : expf(g);
        float sig = g >= 0.0f ? (1.0f / (1.0f + z)) : (z / (1.0f + z));
        attn_want[d] = sum * sig;
    }
    if (!qwen36_metal_full_attention_head_raw(
            backend, attn_query, attn_gate, attn_keys, attn_values,
            3, 4, 4, attn_out, err, errlen)) {
        return false;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (fabsf(attn_out[i] - attn_want[i]) > 0.0002f) {
            qwen36_metal_set_err(err, errlen,
                                 "Qwen Metal attention self-test mismatch at %zu: %g != %g",
                                 i, (double)attn_out[i], (double)attn_want[i]);
            return false;
        }
    }
    return true;
}

qwen36_metal_backend *qwen36_metal_create(char *err, size_t errlen) {
    @autoreleasepool {
        qwen36_metal_backend *backend = calloc(1, sizeof(*backend));
        if (!backend) {
            qwen36_metal_set_err(err, errlen, "failed to allocate Qwen Metal backend");
            return NULL;
        }
        backend->device = MTLCreateSystemDefaultDevice();
        if (!backend->device) {
            qwen36_metal_set_err(err, errlen, "Metal is not available");
            qwen36_metal_destroy(backend);
            return NULL;
        }
        backend->queue = [backend->device newCommandQueue];
        if (!backend->queue) {
            qwen36_metal_set_err(err, errlen, "failed to create Qwen Metal command queue");
            qwen36_metal_destroy(backend);
            return NULL;
        }

        NSError *ns_err = nil;
        id<MTLLibrary> lib =
            [backend->device newLibraryWithSource:qwen36_metal_source()
                                          options:nil
                                            error:&ns_err];
        if (!lib) {
            qwen36_metal_set_err(err, errlen,
                                 "failed to compile Qwen Metal kernels: %s",
                                 ns_err ? [[ns_err localizedDescription] UTF8String] : "unknown error");
            qwen36_metal_destroy(backend);
            return NULL;
        }
        backend->matvec_f32 = qwen36_metal_pipeline(lib, @"qwen36_matvec_f32", err, errlen);
        backend->matvec_f16 = qwen36_metal_pipeline(lib, @"qwen36_matvec_f16", err, errlen);
        backend->matvec_bf16 = qwen36_metal_pipeline(lib, @"qwen36_matvec_bf16", err, errlen);
        backend->matvec_q4_0 = qwen36_metal_pipeline(lib, @"qwen36_matvec_q4_0", err, errlen);
        backend->matvec_q4_1 = qwen36_metal_pipeline(lib, @"qwen36_matvec_q4_1", err, errlen);
        backend->matvec_q5_0 = qwen36_metal_pipeline(lib, @"qwen36_matvec_q5_0", err, errlen);
        backend->matvec_q5_1 = qwen36_metal_pipeline(lib, @"qwen36_matvec_q5_1", err, errlen);
        backend->matvec_q8_0 = qwen36_metal_pipeline(lib, @"qwen36_matvec_q8_0", err, errlen);
        backend->matvec_q8_1 = qwen36_metal_pipeline(lib, @"qwen36_matvec_q8_1", err, errlen);
        backend->matvec_q2_k = qwen36_metal_pipeline(lib, @"qwen36_matvec_q2_k", err, errlen);
        backend->matvec_q3_k = qwen36_metal_pipeline(lib, @"qwen36_matvec_q3_k", err, errlen);
        backend->matvec_q4_k = qwen36_metal_pipeline(lib, @"qwen36_matvec_q4_k", err, errlen);
        backend->matvec_q5_k = qwen36_metal_pipeline(lib, @"qwen36_matvec_q5_k", err, errlen);
        backend->matvec_q6_k = qwen36_metal_pipeline(lib, @"qwen36_matvec_q6_k", err, errlen);
        backend->matvec_q8_k = qwen36_metal_pipeline(lib, @"qwen36_matvec_q8_k", err, errlen);
        backend->matvec_iq2_xxs = qwen36_metal_pipeline(lib, @"qwen36_matvec_iq2_xxs", err, errlen);
        backend->matvec_iq2_xs = qwen36_metal_pipeline(lib, @"qwen36_matvec_iq2_xs", err, errlen);
        backend->matvec_iq2_s = qwen36_metal_pipeline(lib, @"qwen36_matvec_iq2_s", err, errlen);
        backend->matvec_iq3_xxs = qwen36_metal_pipeline(lib, @"qwen36_matvec_iq3_xxs", err, errlen);
        backend->matvec_iq3_s = qwen36_metal_pipeline(lib, @"qwen36_matvec_iq3_s", err, errlen);
        backend->matvec_iq1_s = qwen36_metal_pipeline(lib, @"qwen36_matvec_iq1_s", err, errlen);
        backend->matvec_iq1_m = qwen36_metal_pipeline(lib, @"qwen36_matvec_iq1_m", err, errlen);
        backend->matvec_iq4_nl = qwen36_metal_pipeline(lib, @"qwen36_matvec_iq4_nl", err, errlen);
        backend->matvec_iq4_xs = qwen36_metal_pipeline(lib, @"qwen36_matvec_iq4_xs", err, errlen);
        backend->rms_sum = qwen36_metal_pipeline(lib, @"qwen36_rms_sum", err, errlen);
        backend->rms_apply_f32 = qwen36_metal_pipeline(lib, @"qwen36_rms_apply_f32", err, errlen);
        backend->rms_apply_f16 = qwen36_metal_pipeline(lib, @"qwen36_rms_apply_f16", err, errlen);
        backend->rms_apply_bf16 = qwen36_metal_pipeline(lib, @"qwen36_rms_apply_bf16", err, errlen);
        backend->silu_mul = qwen36_metal_pipeline(lib, @"qwen36_silu_mul", err, errlen);
        backend->l2_sum = qwen36_metal_pipeline(lib, @"qwen36_l2_sum", err, errlen);
        backend->l2_apply = qwen36_metal_pipeline(lib, @"qwen36_l2_apply", err, errlen);
        backend->delta_mem = qwen36_metal_pipeline(lib, @"qwen36_delta_mem", err, errlen);
        backend->delta_update = qwen36_metal_pipeline(lib, @"qwen36_delta_update", err, errlen);
        backend->delta_core = qwen36_metal_pipeline(lib, @"qwen36_delta_core", err, errlen);
        backend->attention_scores = qwen36_metal_pipeline(lib, @"qwen36_attention_scores", err, errlen);
        backend->attention_apply = qwen36_metal_pipeline(lib, @"qwen36_attention_apply", err, errlen);
        if (!backend->matvec_f32 || !backend->matvec_f16 ||
            !backend->matvec_bf16 || !backend->matvec_q8_0 ||
            !backend->matvec_q4_0 || !backend->matvec_q4_1 ||
            !backend->matvec_q5_0 || !backend->matvec_q5_1 ||
            !backend->matvec_q8_1 ||
            !backend->matvec_q2_k || !backend->matvec_q3_k ||
            !backend->matvec_q4_k || !backend->matvec_q5_k ||
            !backend->matvec_q6_k || !backend->matvec_q8_k ||
            !backend->matvec_iq2_xxs || !backend->matvec_iq2_xs ||
            !backend->matvec_iq2_s || !backend->matvec_iq3_xxs ||
            !backend->matvec_iq3_s || !backend->matvec_iq1_s ||
            !backend->matvec_iq1_m ||
            !backend->matvec_iq4_nl ||
            !backend->matvec_iq4_xs ||
            !backend->rms_sum || !backend->rms_apply_f32 ||
            !backend->rms_apply_f16 || !backend->rms_apply_bf16 ||
            !backend->silu_mul || !backend->l2_sum || !backend->l2_apply ||
            !backend->delta_mem || !backend->delta_update || !backend->delta_core ||
            !backend->attention_scores || !backend->attention_apply ||
            !qwen36_metal_self_test(backend, err, errlen)) {
            qwen36_metal_destroy(backend);
            return NULL;
        }
        return backend;
    }
}

void qwen36_metal_destroy(qwen36_metal_backend *backend) {
    if (!backend) return;
    backend->matvec_f32 = nil;
    backend->matvec_f16 = nil;
    backend->matvec_bf16 = nil;
    backend->matvec_q4_0 = nil;
    backend->matvec_q4_1 = nil;
    backend->matvec_q5_0 = nil;
    backend->matvec_q5_1 = nil;
    backend->matvec_q8_0 = nil;
    backend->matvec_q8_1 = nil;
    backend->matvec_q2_k = nil;
    backend->matvec_q3_k = nil;
    backend->matvec_q4_k = nil;
    backend->matvec_q5_k = nil;
    backend->matvec_q6_k = nil;
    backend->matvec_q8_k = nil;
    backend->matvec_iq2_xxs = nil;
    backend->matvec_iq2_xs = nil;
    backend->matvec_iq2_s = nil;
    backend->matvec_iq3_xxs = nil;
    backend->matvec_iq3_s = nil;
    backend->matvec_iq1_s = nil;
    backend->matvec_iq1_m = nil;
    backend->matvec_iq4_nl = nil;
    backend->matvec_iq4_xs = nil;
    backend->rms_sum = nil;
    backend->rms_apply_f32 = nil;
    backend->rms_apply_f16 = nil;
    backend->rms_apply_bf16 = nil;
    backend->silu_mul = nil;
    backend->l2_sum = nil;
    backend->l2_apply = nil;
    backend->delta_mem = nil;
    backend->delta_update = nil;
    backend->delta_core = nil;
    backend->attention_scores = nil;
    backend->attention_apply = nil;
    backend->queue = nil;
    backend->device = nil;
    free(backend);
}

bool qwen36_metal_matvec(qwen36_metal_backend *backend,
                         const rt_gguf_file *file,
                         const rt_gguf_tensor *tensor,
                         const float *x,
                         uint64_t in_dim,
                         float *out,
                         uint64_t out_dim,
                         char *err,
                         size_t errlen) {
    if (!tensor || tensor->ndim != 2 || tensor->dim[0] != in_dim ||
        tensor->dim[1] < out_dim || out_dim == 0 ||
        out_dim > UINT64_MAX / in_dim) {
        return false;
    }
    if (tensor->type != 0 && tensor->type != 1 &&
        tensor->type != 2 && tensor->type != 3 &&
        tensor->type != 6 && tensor->type != 7 &&
        tensor->type != 8 && tensor->type != 9 &&
        tensor->type != 10 &&
        tensor->type != 11 && tensor->type != 12 &&
        tensor->type != 13 && tensor->type != 14 &&
        tensor->type != 15 && tensor->type != 16 &&
        tensor->type != 17 && tensor->type != 18 &&
        tensor->type != 19 &&
        tensor->type != 20 && tensor->type != 21 &&
        tensor->type != 22 &&
        tensor->type != 23 &&
        tensor->type != 29 &&
        tensor->type != 30) {
        return false;
    }
    const void *data = rt_gguf_file_tensor_data(file, tensor);
    if (!data) return false;
    uint64_t needed_elements = in_dim * out_dim;
    uint64_t needed_bytes = 0;
    if (!rt_gguf_tensor_nbytes(tensor->type, needed_elements, &needed_bytes) ||
        needed_bytes > tensor->bytes) {
        return false;
    }
    return qwen36_metal_matvec_raw(backend, data, needed_bytes,
                                   tensor->type, x, in_dim, out, out_dim,
                                   err, errlen);
}

bool qwen36_metal_rms_norm(qwen36_metal_backend *backend,
                           const rt_gguf_file *file,
                           const rt_gguf_tensor *weight,
                           const float *x,
                           uint64_t dim,
                           float *out,
                           bool add_one_to_weight,
                           char *err,
                           size_t errlen) {
    if (!weight || weight->ndim != 1 || weight->dim[0] != dim || dim == 0) {
        return false;
    }
    if (weight->type != 0 && weight->type != 1 && weight->type != 30) {
        return false;
    }
    const void *data = rt_gguf_file_tensor_data(file, weight);
    if (!data) return false;
    uint64_t needed_bytes = 0;
    if (!rt_gguf_tensor_nbytes(weight->type, dim, &needed_bytes) ||
        needed_bytes > weight->bytes) {
        return false;
    }
    return qwen36_metal_rms_norm_raw(backend, data, needed_bytes,
                                     weight->type, x, dim, out,
                                     add_one_to_weight, err, errlen);
}

bool qwen36_metal_silu_mul(qwen36_metal_backend *backend,
                           const float *gate,
                           const float *up,
                           uint64_t dim,
                           float *out,
                           char *err,
                           size_t errlen) {
    return qwen36_metal_silu_mul_raw(backend, gate, up, dim, out, err, errlen);
}

bool qwen36_metal_l2_norm(qwen36_metal_backend *backend,
                          const float *x,
                          uint64_t dim,
                          float *out,
                          char *err,
                          size_t errlen) {
    return qwen36_metal_l2_norm_raw(backend, x, dim, out, err, errlen);
}

bool qwen36_metal_gated_delta_head(qwen36_metal_backend *backend,
                                   float *state,
                                   const float *query,
                                   const float *key,
                                   const float *value,
                                   uint64_t dim,
                                   float decay,
                                   float beta,
                                   float qscale,
                                   float *out,
                                   char *err,
                                   size_t errlen) {
    return qwen36_metal_gated_delta_head_raw(backend, state, query, key, value,
                                             dim, decay, beta, qscale, out,
                                             err, errlen);
}

bool qwen36_metal_full_attention_head(qwen36_metal_backend *backend,
                                      const float *query,
                                      const float *gate,
                                      const float *keys,
                                      const float *values,
                                      uint64_t seq_len,
                                      uint64_t kv_stride,
                                      uint64_t head_dim,
                                      float *out,
                                      char *err,
                                      size_t errlen) {
    return qwen36_metal_full_attention_head_raw(backend, query, gate, keys,
                                                values, seq_len, kv_stride,
                                                head_dim, out, err, errlen);
}
