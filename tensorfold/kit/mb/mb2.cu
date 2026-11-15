// q38 microbench 2: launch overhead and programmatic dependent launch (PDL) on a hyper-connection-like chain:
//   tiny -> qmmf mix_down (320x10240 BF16, 8 K-slices in a cluster, fp32 out) -> tiny -> qmmf mix_up (10240x320 BF16) -> tiny, 96 weight copies.
// Modes: plain launches; PDL attribute on the qmmf launches; PDL attribute on every launch. Final buffers are compared across modes.
#include "qmmf.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
using namespace tf_qmmf;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

// a glue-sized kernel: n bf16 values, out = in with the low mantissa bit cleared (keeps values bounded); PDL-aware
__global__ void tiny(const uint16_t* __restrict__ in, uint16_t* __restrict__ out, int n) {
    qmm_frag::grid_launch();
    qmm_frag::grid_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] & 0xFFFEu;
}
// fp32 partial sums -> bf16 activations (like fn_hc_reduce_act, simplified but deterministic)
__global__ void tinyf(const float* __restrict__ in, uint16_t* __restrict__ out, int n) {
    qmm_frag::grid_launch();
    qmm_frag::grid_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = in[i]; v = v > 4.f ? 4.f : v < -4.f ? -4.f : v; out[i] = (uint16_t)(__float_as_uint(v * 0.01f) >> 16); }
}
typedef void (*KFn)(const __nv_bfloat16*, const unsigned char*, const uint8_t*, float, void*, float*, int, int, int, int, int, int, int);

int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 7, L = 96, K1 = 10240, N1 = 320, K2 = 320, N2 = 10240, M = argc > 2 ? atoi(argv[2]) : 1;
    const size_t wb1 = (size_t)((N1 + 127) / 128 * 128 / 64) * (K1 / 64) * 8192, wb2 = (size_t)((N2 + 127) / 128 * 128 / 64) * (K2 / 64) * 8192;
    unsigned char *w1, *w2; uint8_t* bs; CK(cudaMalloc(&w1, wb1 * L)); CK(cudaMalloc(&w2, wb2 * L)); CK(cudaMalloc(&bs, 4096));
    { std::vector<uint16_t> h((wb1 > wb2 ? wb1 : wb2) / 2); unsigned s = 4242;
      for (int cp = 0; cp < L; ++cp) { for (size_t i = 0; i < wb1 / 2; ++i) { s = s * 1664525u + 1013904223u; h[i] = (uint16_t)(0x3A00u + ((s >> 16) & 0x1FFu)) | ((s >> 30) & 1 ? 0x8000u : 0); } CK(cudaMemcpy(w1 + wb1 * cp, h.data(), wb1, cudaMemcpyHostToDevice));
                                     for (size_t i = 0; i < wb2 / 2; ++i) { s = s * 1664525u + 1013904223u; h[i] = (uint16_t)(0x3A00u + ((s >> 16) & 0x1FFu)) | ((s >> 30) & 1 ? 0x8000u : 0); } CK(cudaMemcpy(w2 + wb2 * cp, h.data(), wb2, cudaMemcpyHostToDevice)); } }
    uint16_t *x0, *x, *x1, *act, *up; float* part;
    CK(cudaMalloc(&x0, M * K1 * 2)); CK(cudaMalloc(&x, M * K1 * 2)); CK(cudaMalloc(&x1, M * K1 * 2)); CK(cudaMalloc(&act, M * K2 * 2)); CK(cudaMalloc(&up, M * N2 * 2)); CK(cudaMalloc(&part, M * N1 * 4));
    { std::vector<uint16_t> hx((size_t)M * K1); unsigned s = 99; for (auto& v : hx) { s = s * 1664525u + 1013904223u; v = (uint16_t)(0x3C00u + ((s >> 16) & 0x3FFu)) | ((s >> 30) & 1 ? 0x8000u : 0); } CK(cudaMemcpy(x0, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice)); }
    KFn down = (KFn)qmmf_kernel<BF16, 16, 64, 1, 4, 4, true, true, false>, upk = (KFn)qmmf_kernel<BF16, 16, 64, 1, 4, 4, false, false, false>;
    CK(cudaFuncSetAttribute((const void*)down, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960)); CK(cudaFuncSetAttribute((const void*)upk, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960));
    auto cfgOf = [&](dim3 g, int thr, unsigned sh, int cluster, bool pdl, cudaLaunchAttribute* at) {
        cudaLaunchConfig_t c = {}; c.gridDim = g; c.blockDim = dim3(thr, 1, 1); c.dynamicSmemBytes = sh; c.stream = 0; int n = 0;
        if (pdl) { at[n].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[n].val.programmaticStreamSerializationAllowed = 1; ++n; }
        if (cluster) { at[n].id = cudaLaunchAttributeClusterDimension; at[n].val.clusterDim.x = 1; at[n].val.clusterDim.y = 1; at[n].val.clusterDim.z = cluster; ++n; }
        c.attrs = at; c.numAttrs = n; return c; };
    std::vector<uint16_t> ref((size_t)M * K1), got((size_t)M * K1);
    const char* names[3] = {"plain", "PDL on qmmf", "PDL on all"};
    for (int chain = 0; chain < 2; ++chain) for (int mode = 0; mode < 3; ++mode) {
        const bool pq = mode >= 1, pt = mode >= 2; double best = 1e9;
        for (int r = 0; r < reps + 1; ++r) {
            CK(cudaMemcpy(x, x0, M * K1 * 2, cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize());
            auto t0 = std::chrono::steady_clock::now();
            for (int l = 0; l < L; ++l) {
                cudaLaunchAttribute a[3];
                if (chain == 0) {
                    auto c1 = cfgOf(dim3((M * K1 + 127) / 128), 128, 0, 0, pt, a); CK(cudaLaunchKernelEx(&c1, tiny, (const uint16_t*)x, x1, M * K1));
                    auto c2 = cfgOf(dim3((N1 + 63) / 64, 1, 8), 128, 40960, 8, pq, a); CK(cudaLaunchKernelEx(&c2, down, (const __nv_bfloat16*)x1, (const unsigned char*)(w1 + wb1 * l), (const uint8_t*)bs, 1.0f, (void*)part, (float*)nullptr, M, N1, K1, 8, (N1 + 127) / 128 * 128, K1, 1));
                    auto c3 = cfgOf(dim3((M * N1 + 127) / 128), 128, 0, 0, pt, a); CK(cudaLaunchKernelEx(&c3, tinyf, (const float*)part, act, M * N1));
                    auto c4 = cfgOf(dim3((N2 + 63) / 64, 1, 1), 128, 40960, 0, pq, a); CK(cudaLaunchKernelEx(&c4, upk, (const __nv_bfloat16*)act, (const unsigned char*)(w2 + wb2 * l), (const uint8_t*)bs, 1.0f, (void*)up, (float*)nullptr, M, N2, K2, 1, (N2 + 127) / 128 * 128, K2, 1));
                    auto c5 = cfgOf(dim3((M * K1 + 127) / 128), 128, 0, 0, pt, a); CK(cudaLaunchKernelEx(&c5, tiny, (const uint16_t*)up, x, M * N2));
                } else {   // 10 glue-sized kernels in a row
                    for (int i = 0; i < 5; ++i) { auto c1 = cfgOf(dim3((M * K1 + 127) / 128), 128, 0, 0, pt, a); CK(cudaLaunchKernelEx(&c1, tiny, (const uint16_t*)x, x1, M * K1)); CK(cudaLaunchKernelEx(&c1, tiny, (const uint16_t*)x1, x, M * K1)); }
                }
            }
            CK(cudaDeviceSynchronize());
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / L;
            if (r > 0 && s < best) best = s;
        }
        CK(cudaMemcpy(got.data(), x, (size_t)M * K1 * 2, cudaMemcpyDeviceToHost));
        if (mode == 0) ref = got;
        printf("%s M=%d %-12s %7.1f us per %s  %s\n", chain == 0 ? "hc chain (tiny,down,tiny,up,tiny)" : "10 tiny kernels", M, names[mode], best * 1e6, chain == 0 ? "chain" : "10 launches",
               mode == 0 ? "ref" : memcmp(ref.data(), got.data(), got.size() * 2) == 0 ? "BITS SAME" : "BITS DIFFER");
    }
    return 0;
}
