// q38 microbench: TensorFold's qmmf lane kernel on decode-step shapes, stock vs variants, bit-compared. No model needed.
// build: nvcc -O3 -std=c++17 -arch=sm_121 -I KERNELS_DIR mb.cu -o mb     run: ./mb [reps]
#include "qmmf.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
using namespace tf_qmmf;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ uint4 mb_ld(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
// raw read ceiling: every thread streams 16-byte loads over its own contiguous chunk
__global__ void read_kernel(const uint4* __restrict__ p, size_t per_thread, unsigned* __restrict__ out) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint4* q = p + t * per_thread;
    unsigned a = 0;
    for (size_t i = 0; i < per_thread; ++i) { uint4 v = mb_ld(q + i); a ^= v.x + v.y + v.z + v.w; }
    if (a == 0x12345678u) out[0] = a;
}

typedef void (*KFn)(const __nv_bfloat16*, const unsigned char*, const uint8_t*, float, void*, float*, int, int, int, int, int, int, int);
struct Var { const char* name; KFn fn; int bm, stages; bool cluster; bool f32; int mode; };
template <int MODE, int BM, int ST, bool F32, bool CL>
Var mk(const char* name) { return Var{name, (KFn)qmmf_kernel<MODE, BM, 64, 1, 4, ST, F32, CL, false>, BM, ST, CL, F32, MODE}; }
static int splitK(int n, int k) { int tiles = (n + 63) / 64, groups = k / 64, sk = 1; while (sk < 8 && tiles * sk < 192 && groups % (sk * 2) == 0 && groups / (sk * 2) >= 8) sk *= 2; return sk; }
static unsigned sharedBytes(int mode, int bm, int st) { unsigned w = mode == BF16 ? 64 * 64 * 2 : 64 * 64, s = mode == MXFP8 ? 128 : 0; unsigned stage = (bm * 128 + w + s + 127) / 128 * 128; unsigned part = (bm / 16) * 2 * 4 * 128 * 4; unsigned v = st * stage; return v > part ? v : part; }
struct Case { const char* name; int mode, n, k, copies; };
int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 5;
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device %s SMs %d smem/SM %zu KB smem/block-optin %zu KB L2 %d MB\n", pr.name, pr.multiProcessorCount, pr.sharedMemPerMultiprocessor / 1024, pr.sharedMemPerBlockOptin / 1024, pr.l2CacheSize >> 20);
    // ceiling
    {
        size_t bytes = (size_t)1 << 31; uint4* p; unsigned* o; CK(cudaMalloc(&p, bytes)); CK(cudaMalloc(&o, 64)); CK(cudaMemset(p, 1, bytes));
        for (int bpsm : {1, 2, 4, 8}) for (int thr : {128, 256, 512}) {
            int blocks = pr.multiProcessorCount * bpsm; size_t per = bytes / 16 / ((size_t)blocks * thr);
            read_kernel<<<blocks, thr>>>(p, per, o); CK(cudaDeviceSynchronize());
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 3; ++r) read_kernel<<<blocks, thr>>>(p, per, o);
            CK(cudaDeviceSynchronize());
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("ceiling read: %d blocks/SM x %d threads: %.1f GB/s\n", bpsm, thr, 3.0 * per * 16 * blocks * thr / s / 1e9);
        }
        cudaFree(p); cudaFree(o);
    }
    // q38 mb3: decode-lane projections at 65..128 rows. Stock: two 64-row tiles side by side, so every weight tile is read twice.
    // Variant: one 128-row tile a block (weights once). Outputs compared with stock bit for bit.
    std::vector<Case> cases = {
        {"hc mix_up BF16 10240x320", BF16, 10240, 320, 96},
        {"hc mix_down BF16 320x10240", BF16, 320, 10240, 96},
        {"q_proj MXFP8 12288x2560", MXFP8, 12288, 2560, 24},
        {"o_proj MXFP8 2560x6144", MXFP8, 2560, 6144, 24},
        {"gdn proj MXFP8 6144x2560", MXFP8, 6144, 2560, 36},
        {"shared MXFP8 640x2560", MXFP8, 640, 2560, 96},
    };
    for (auto& c : cases) {
        const int KG = c.k / 64, npad = (c.n + 127) / 128 * 128, tiles = (c.n + 63) / 64, sk = splitK(c.n, c.k);
        const size_t wb = (size_t)(npad / 64) * KG * (c.mode == BF16 ? 8192 : 4096), sb = c.mode == MXFP8 ? (size_t)(npad / 64) * KG * 128 : 16;
        unsigned char* w; uint8_t* bs; CK(cudaMalloc(&w, wb * c.copies)); CK(cudaMalloc(&bs, sb * c.copies));
        {
            std::vector<unsigned char> h(wb); unsigned s = 12345;
            for (int cp = 0; cp < c.copies; ++cp) {
                if (c.mode == BF16) { uint16_t* q = (uint16_t*)h.data(); for (size_t i = 0; i < wb / 2; ++i) { s = s * 1664525u + 1013904223u; q[i] = (uint16_t)(0x3C00u + ((s >> 16) & 0x1FFu)) | ((s >> 30) & 1 ? 0x8000u : 0); } }
                else for (size_t i = 0; i < wb; ++i) { s = s * 1664525u + 1013904223u; h[i] = (unsigned char)(((s >> 16) & 0x3F) + 0x10) | ((s >> 30) & 1 ? 0x80 : 0); }
                CK(cudaMemcpy(w + wb * cp, h.data(), wb, cudaMemcpyHostToDevice));
            }
            std::vector<uint8_t> hs(sb * c.copies); for (auto& v : hs) { s = s * 1664525u + 1013904223u; v = 125 + ((s >> 20) & 3); }
            CK(cudaMemcpy(bs, hs.data(), hs.size(), cudaMemcpyHostToDevice));
        }
        for (int M : {48, 79, 97, 128}) {
            __nv_bfloat16* x; CK(cudaMalloc(&x, (size_t)M * c.k * 2));
            { std::vector<uint16_t> hx((size_t)M * c.k); unsigned s = 777; for (auto& v : hx) { s = s * 1664525u + 1013904223u; v = (uint16_t)(0x3E00u + ((s >> 16) & 0x3FFu)) | ((s >> 30) & 1 ? 0x8000u : 0); } CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice)); }
            const bool cl = sk > 1;
            const size_t ob = (size_t)M * c.n * 4;
            void *out, *ref; CK(cudaMalloc(&out, ob)); CK(cudaMalloc(&ref, ob));
            std::vector<unsigned char> href(ob), hout(ob);
            std::vector<Var> vars;
            if (c.mode == BF16) { if (cl) vars = {mk<BF16, 64, 4, true, true>("stock 64x2 st4"), mk<BF16, 128, 4, true, true>("128 st4"), mk<BF16, 128, 3, true, true>("128 st3"), mk<BF16, 128, 2, true, true>("128 st2")};
                                  else vars = {mk<BF16, 64, 4, false, false>("stock 64x2 st4"), mk<BF16, 128, 4, false, false>("128 st4"), mk<BF16, 128, 3, false, false>("128 st3"), mk<BF16, 128, 2, false, false>("128 st2")}; }
            else { if (cl) vars = {mk<MXFP8, 64, 4, true, true>("stock 64x2 st4"), mk<MXFP8, 128, 4, true, true>("128 st4"), mk<MXFP8, 128, 3, true, true>("128 st3"), mk<MXFP8, 128, 2, true, true>("128 st2")};
                   else vars = {mk<MXFP8, 64, 4, false, false>("stock 64x2 st4"), mk<MXFP8, 128, 4, false, false>("128 st4"), mk<MXFP8, 128, 3, false, false>("128 st3"), mk<MXFP8, 128, 2, false, false>("128 st2")}; }
            double tref = 0;
            for (size_t vi = 0; vi < vars.size(); ++vi) {
                Var& v = vars[vi];
                unsigned sh = sharedBytes(c.mode, v.bm, v.stages);
                if (sh > pr.sharedMemPerBlockOptin) { printf("%-28s M=%-3d %-16s skipped (smem %u > %zu)\n", c.name, M, v.name, sh, pr.sharedMemPerBlockOptin); continue; }
                CK(cudaFuncSetAttribute((const void*)v.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sh));
                int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)v.fn, 128, sh));
                const int rows_t = (M + v.bm - 1) / v.bm;
                cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(rows_t * tiles, 1, sk); cfg.blockDim = dim3(128, 1, 1); cfg.dynamicSmemBytes = sh; cfg.stream = 0;
                cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = sk;
                cfg.attrs = at; cfg.numAttrs = cl ? 1 : 0;
                auto go = [&](int cp, void* o) {
                    CK(cudaLaunchKernelEx(&cfg, v.fn, (const __nv_bfloat16*)x, (const unsigned char*)(w + wb * cp), (const uint8_t*)(bs + sb * cp), 1.0f, o, (float*)nullptr, M, c.n, c.k, sk, npad, c.k, 1));
                };
                CK(cudaMemset(out, 0, ob)); CK(cudaMemset(ref, 0, vi == 0 ? ob : 0)); go(0, vi == 0 ? ref : out); CK(cudaDeviceSynchronize());
                bool same = true;
                if (vi == 0) CK(cudaMemcpy(href.data(), ref, ob, cudaMemcpyDeviceToHost));
                else { CK(cudaMemcpy(hout.data(), out, ob, cudaMemcpyDeviceToHost)); size_t cmpb = (size_t)M * c.n * (v.f32 ? 4 : 2); same = memcmp(hout.data(), href.data(), cmpb) == 0; }
                double best = 1e9;
                for (int r = 0; r < reps; ++r) {
                    auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < c.copies; ++i) go(i % c.copies, out);
                    CK(cudaDeviceSynchronize());
                    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / c.copies;
                    if (s < best) best = s;
                }
                if (vi == 0) tref = best;
                printf("%-28s M=%-3d sk=%d %s %-16s %8.1f us  %6.1f GB/s of weights  %+6.1f %%  smem %5u occ %d  %s\n", c.name, M, sk, cl ? "cluster" : "plain  ", v.name, best * 1e6, (wb + sb) / best / 1e9, (tref / best - 1) * 100, sh, occ, vi == 0 ? "ref" : same ? "BITS SAME" : "BITS DIFFER");
            }
            cudaFree(x); cudaFree(out); cudaFree(ref);
        }
        cudaFree(w); cudaFree(bs);
    }
    return 0;
}
