// q38 expert microbench: TensorFold's NVFP4 routed-expert kernels (decode form, MiaAI's shared-memory row tile) on Flash Next
// shapes, by rows a pass and distinct experts a layer. No model needed: synthetic weights, 48 "layers" a step, experts drawn from a
// pool far larger than the GPU cache. Every variant's outputs are hashed and compared with the stock decode kernel's.
// build: nvcc -O3 -std=c++17 -arch=sm_121 -I KERNELS mbx.cu -o mbx          (stock arithmetic)
//        nvcc -O3 -std=c++17 -arch=sm_121a -DQ38_NATIVE_FP -I KERNELS mbx.cu -o mbxn   (native E2M1 / E4M3 decodes)
// run:   ./mbx [reps=3] [width=640] [pool experts=1024]
#include "nvfp4_experts.cu"
#include "nvfp4_experts_tile.cu"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

typedef void (*EFn)(const __nv_bfloat16*, int, int, const uint4*, const float*, int, int, const int*, const int*, const int*, void*, int, float, int);

static __device__ __forceinline__ uint32_t mix32(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
  return (uint32_t)x;
}
// weights: random code nibbles; the 4 scale uint4 of every 36 hold E4M3 codes 0x28..0x37 (0.25 .. 0.94), never a NaN byte
__global__ void fill_w(uint4* w, size_t n, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v;
    v.x = mix32(i * 4 + 0 + ((uint64_t)seed << 40)); v.y = mix32(i * 4 + 1 + ((uint64_t)seed << 40));
    v.z = mix32(i * 4 + 2 + ((uint64_t)seed << 40)); v.w = mix32(i * 4 + 3 + ((uint64_t)seed << 40));
    if (i % 36 >= 32) { v.x = 0x28282828u + (v.x & 0x0F0F0F0Fu); v.y = 0x28282828u + (v.y & 0x0F0F0F0Fu); v.z = 0x28282828u + (v.z & 0x0F0F0F0Fu); v.w = 0x28282828u + (v.w & 0x0F0F0F0Fu); }
    w[i] = v;
  }
}
// all E2M1 bytes through fp4pair's four shifts, and all E4M3 byte pairs without a NaN byte: native against stock
__global__ void check_fp(unsigned* bad) {
  unsigned b4 = 0, b8 = 0;
  for (uint32_t i = threadIdx.x; i < 65536; i += blockDim.x) {
    const uint32_t w = mix32(i * 7919ull + blockIdx.x * 104729ull);
    for (int s = 0; s < 16; s += 4) if (q38_fp::fp4pair(w, s) != q38_fp::sw_fp4pair(w, s)) ++b4;
    if ((i & 0x7F) == 0x7F || ((i >> 8) & 0x7F) == 0x7F) continue;
    float s0, s1; q38_fp::e4m3f2(i, s0, s1);
    if (__float_as_uint(s0) != __float_as_uint(q38_fp::sw_e4m3f(i & 0xFFu)) || __float_as_uint(s1) != __float_as_uint(q38_fp::sw_e4m3f((i >> 8) & 0xFFu))) ++b8;
  }
  if (b4) atomicAdd(bad, b4);
  if (b8) atomicAdd(bad + 1, b8);
}

struct Plan { int* items; int* counts; int* members; int nitems, valid; };   // device arrays; valid = items that are not the shared slot
struct Var { const char* name; EFn fn; int kind; int warps; int smem; int pairs; };   // kind 0 decode form, 1 tile

static uint64_t fnv(const unsigned char* p, size_t n) { uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; } return h; }

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 3, WIDTH = argc > 2 ? atoi(argv[2]) : 640, NE = argc > 3 ? atoi(argv[3]) : 1024;
  const int DIMS = 2560, SLOTS = 11, L = 48;
  const int KG_G = DIMS / 32, NB_G = WIDTH / 32, KG_D = WIDTH / 32, NB_D = DIMS / 32;
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  const int SMS = pr.multiProcessorCount;
  printf("device %s SMs %d; native FP decode compiled: %d; width %d, pool %d experts, %d layers a step\n", pr.name, SMS, Q38_FP_NATIVE, WIDTH, NE, L);
  { unsigned* bad; CK(cudaMalloc(&bad, 8)); CK(cudaMemset(bad, 0, 8)); check_fp<<<64, 256>>>(bad); CK(cudaDeviceSynchronize()); unsigned hb[2]; CK(cudaMemcpy(hb, bad, 8, cudaMemcpyDeviceToHost));
    printf("fp check (this binary's decodes against the stock arithmetic): E2M1 pairs differing %u of 16,777,216; E4M3 scale pairs differing %u\n", hb[0], hb[1]); cudaFree(bad); }
  const size_t gu_u4 = (size_t)NE * NB_G * KG_G * 2 * 36, dn_u4 = (size_t)NE * NB_D * KG_D * 36;
  const double exp_bytes = ((double)NB_G * KG_G * 2 * 36 + (double)NB_D * KG_D * 36) * 16;
  uint4 *wg, *wd; float *sg, *sd;
  CK(cudaMalloc(&wg, gu_u4 * 16)); CK(cudaMalloc(&wd, dn_u4 * 16)); CK(cudaMalloc(&sg, (size_t)(NE + 1) * 2 * 4)); CK(cudaMalloc(&sd, (size_t)(NE + 1) * 4));
  fill_w<<<SMS * 8, 256>>>(wg, gu_u4, 1); fill_w<<<SMS * 8, 256>>>(wd, dn_u4, 2); CK(cudaDeviceSynchronize());
  { std::vector<float> h((NE + 1) * 2); unsigned s = 99; for (auto& v : h) { s = s * 1664525u + 1013904223u; v = 0.01f + (float)((s >> 16) & 0xFF) / 4096.f; } CK(cudaMemcpy(sg, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sd, h.data(), (NE + 1) * 4, cudaMemcpyHostToDevice)); }
  printf("weights: gate/up %.2f GiB, down %.2f GiB; one expert %.3f MiB\n", gu_u4 * 16 / 1073741824.0, dn_u4 * 16 / 1073741824.0, exp_bytes / 1048576.0);

  std::vector<Var> gv = {
      {"decode stock", (EFn)tf_nvfp4_experts::nvfp4_expert_kernel<2, 2, 4>, 0, 4, 0, 16},
      {"decode d3", (EFn)q38_gate_up_d3, 0, 4, 0, 16},
      {"decode d4", (EFn)q38_gate_up_d4, 0, 4, 0, 16},
      {"decode 1-warp", (EFn)tf_nvfp4_experts::nvfp4_expert_kernel<2, 2, 1>, 0, 1, 0, 16},
      {"tile rt4 s3 p64", (EFn)tf_nvfp4_tile_gu_rt4_s3, 1, 4, 3 * (4096 + 2 * 2304), 64},
      {"tile rt4 s4 p64", (EFn)tf_nvfp4_tile_gu_rt4_s4, 1, 4, 4 * (4096 + 2 * 2304), 64},
      {"tile rt2 s3 p32", (EFn)tf_nvfp4_tile_gu_rt2_s3, 1, 4, 3 * (4096 + 2 * 2304), 32},
      {"tile rt2 s4 p32", (EFn)tf_nvfp4_tile_gu_rt2_s4, 1, 4, 4 * (4096 + 2 * 2304), 32},
      {"tile rt2 s4 p16", (EFn)tf_nvfp4_tile_gu_rt2_s4, 1, 4, 4 * (4096 + 2 * 2304), 16},
  };
  std::vector<Var> dv = {
      {"decode stock", (EFn)tf_nvfp4_experts::nvfp4_expert_kernel<1, 0, 4>, 0, 4, 0, 16},
      {"decode d3", (EFn)q38_down_f32_d3, 0, 4, 0, 16},
      {"decode d4", (EFn)q38_down_f32_d4, 0, 4, 0, 16},
      {"decode d8", (EFn)q38_down_f32_d8, 0, 4, 0, 16},
      {"decode 1-warp", (EFn)tf_nvfp4_experts::nvfp4_expert_kernel<1, 0, 1>, 0, 1, 0, 16},
      {"tile rt4 s3 p64", (EFn)tf_nvfp4_tile_f32_rt4_s3, 1, 4, 3 * (4096 + 2304), 64},
      {"tile rt4 s4 p64", (EFn)tf_nvfp4_tile_f32_rt4_s4, 1, 4, 4 * (4096 + 2304), 64},
      {"tile rt2 s3 p32", (EFn)tf_nvfp4_tile_f32_rt2_s3, 1, 4, 3 * (4096 + 2304), 32},
      {"tile rt2 s3 p16", (EFn)tf_nvfp4_tile_f32_rt2_s3, 1, 4, 3 * (4096 + 2304), 16},
  };
  auto occ = [&](const Var& v) { int n = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void*)v.fn, v.warps * 32, v.smem)); return std::max(1, n); };
  printf("resident CTAs an SM (warps):");
  for (auto& v : gv) printf("  gate %s %d (%d)", v.name, occ(v), occ(v) * v.warps);
  for (auto& v : dv) printf("  down %s %d (%d)", v.name, occ(v), occ(v) * v.warps);
  printf("\n");

  struct Scen { const char* name; int rows, pool; };
  std::vector<Scen> scens = {{"1 row (10 experts)", 1, 10}, {"5 rows from 30 experts (a drafted round)", 5, 30}, {"16 rows from 100", 16, 100},
                             {"97 rows from 120 (16 users)", 97, 120}, {"97 rows from 250 (16 users, wider)", 97, 250}, {"1024 rows from 512 (prompt)", 1024, 512}};
  const int pair_kinds[3] = {16, 32, 64};
  for (auto& sc : scens) {
    const int R = sc.rows, pairs = R * SLOTS;
    // plans: per layer, per pairs-an-item
    std::vector<Plan> plans[3]; double valid_sum = 0; unsigned rs = 4242 + R * 31 + sc.pool;
    auto rnd = [&]() { rs = rs * 1664525u + 1013904223u; return rs >> 8; };
    for (int l = 0; l < L; ++l) {
      std::vector<int> pool(sc.pool); { std::vector<char> used(NE, 0); for (auto& e : pool) { int c; do c = rnd() % NE; while (used[c]); used[c] = 1; e = c; } }
      std::vector<std::vector<int>> mem(NE + 1);
      for (int r = 0; r < R; ++r) {
        std::vector<int> pick; while ((int)pick.size() < SLOTS - 1) { int c = pool[rnd() % sc.pool]; if (std::find(pick.begin(), pick.end(), c) == pick.end()) pick.push_back(c); }
        for (int s = 0; s < SLOTS - 1; ++s) mem[pick[s]].push_back(r * SLOTS + s);
        mem[NE].push_back(r * SLOTS + SLOTS - 1);   // the shared slot: the kernels skip expert NE
      }
      for (int pk = 0; pk < 3; ++pk) {
        std::vector<int> items, members; int valid = 0;
        for (int e = 0; e <= NE; ++e) for (size_t o = 0; o < mem[e].size(); o += pair_kinds[pk]) {
          const int cnt = (int)std::min<size_t>(pair_kinds[pk], mem[e].size() - o);
          items.push_back(e); items.push_back((int)members.size()); items.push_back(cnt);
          for (int i = 0; i < cnt; ++i) members.push_back(mem[e][o + i]);
          if (e != NE && o == 0) ++valid;
        }
        Plan p; p.nitems = (int)items.size() / 3; p.valid = valid; int cnt1 = p.nitems;
        CK(cudaMalloc(&p.items, items.size() * 4)); CK(cudaMalloc(&p.counts, 4)); CK(cudaMalloc(&p.members, members.size() * 4));
        CK(cudaMemcpy(p.items, items.data(), items.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(p.counts, &cnt1, 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(p.members, members.data(), members.size() * 4, cudaMemcpyHostToDevice));
        plans[pk].push_back(p); if (pk == 0) valid_sum += valid;
      }
    }
    const double bytes_g = valid_sum * (double)NB_G * KG_G * 2 * 36 * 16, bytes_d = valid_sum * (double)NB_D * KG_D * 36 * 16;
    __nv_bfloat16 *x, *act; float* y; const size_t actb = (size_t)pairs * WIDTH * 2, yb = (size_t)pairs * DIMS * 4;
    CK(cudaMalloc(&x, (size_t)R * DIMS * 2)); CK(cudaMalloc(&act, actb)); CK(cudaMalloc(&y, yb));
    { std::vector<uint16_t> hx((size_t)R * DIMS); unsigned s = 777; for (auto& v : hx) { s = s * 1664525u + 1013904223u; v = (uint16_t)(0x3E00u + ((s >> 16) & 0x3FFu)) | ((s >> 30) & 1 ? 0x8000u : 0); } CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice)); }
    printf("\n== %s: %.1f distinct experts a layer, %d (row, expert) pairs; a step reads gate/up %.3f GiB + down %.3f GiB\n", sc.name, valid_sum / L, pairs, bytes_g / 1073741824.0, bytes_d / 1073741824.0);
    auto launch = [&](const Var& v, bool gate, int l) {
      const int pk = v.pairs == 16 ? 0 : v.pairs == 32 ? 1 : 2; const Plan& p = plans[pk][l];
      const int nb = gate ? NB_G : NB_D, kg = gate ? KG_G : KG_D;
      const int grid = v.kind == 0 ? std::min((p.nitems * nb + v.warps - 1) / v.warps, occ(v) * SMS) : std::min(p.nitems * ((nb + 3) / 4), occ(v) * SMS);
      cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(grid, 1, 1); cfg.blockDim = dim3(v.warps * 32, 1, 1); cfg.dynamicSmemBytes = v.smem; cfg.stream = 0;
      if (gate) CK(cudaLaunchKernelEx(&cfg, v.fn, (const __nv_bfloat16*)x, DIMS, SLOTS, (const uint4*)wg, (const float*)sg, kg, nb, (const int*)p.items, (const int*)p.counts, (const int*)p.members, (void*)act, WIDTH, 0.0f, NE));
      else CK(cudaLaunchKernelEx(&cfg, v.fn, (const __nv_bfloat16*)act, WIDTH, 0, (const uint4*)wd, (const float*)sd, kg, nb, (const int*)p.items, (const int*)p.counts, (const int*)p.members, (void*)y, DIMS, 0.0f, NE));
    };
    std::vector<unsigned char> hbuf(std::max(actb, yb));
    uint64_t ref_g = 0, ref_d = 0; double t_ref_g = 0, t_ref_d = 0, best_g = 1e9, best_d = 1e9; std::string bg, bd;
    for (int side = 0; side < 2; ++side) {
      const bool gate = side == 0; auto& vars = gate ? gv : dv;
      for (size_t vi = 0; vi < vars.size(); ++vi) {
        const Var& v = vars[vi];
        if (v.kind == 0 && v.pairs != 16) continue;
        // bits: layers 0..2, each from zeroed outputs; the down side reads the stock gate kernel's activations of the same layer
        uint64_t h = 0;
        for (int l = 0; l < 3; ++l) {
          if (gate) { CK(cudaMemset(act, 0, actb)); launch(v, true, l); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hbuf.data(), act, actb, cudaMemcpyDeviceToHost)); h ^= fnv(hbuf.data(), actb) * (l + 1); }
          else { CK(cudaMemset(act, 0, actb)); launch(gv[0], true, l); CK(cudaMemset(y, 0, yb)); launch(v, false, l); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hbuf.data(), y, yb, cudaMemcpyDeviceToHost)); h ^= fnv(hbuf.data(), yb) * (l + 1); }
        }
        if (vi == 0) (gate ? ref_g : ref_d) = h;
        double best = 1e9;
        for (int r = 0; r < reps + 1; ++r) {
          auto t0 = std::chrono::steady_clock::now();
          for (int l = 0; l < L; ++l) launch(v, gate, l);
          CK(cudaDeviceSynchronize());
          const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          if (r > 0 && s < best) best = s;
        }
        const double by = gate ? bytes_g : bytes_d;
        if (vi == 0) (gate ? t_ref_g : t_ref_d) = best;
        if (best < (gate ? best_g : best_d)) { (gate ? best_g : best_d) = best; (gate ? bg : bd) = v.name; }
        printf("  %-5s %-16s %9.3f ms a step  %6.1f GB/s  %+6.1f %%  hash %016llx %s\n", gate ? "gate" : "down", v.name, best * 1e3, by / best / 1e9, (((gate ? t_ref_g : t_ref_d) / best) - 1) * 100,
               (unsigned long long)h, h == (gate ? ref_g : ref_d) ? (vi == 0 ? "ref" : "BITS SAME") : "BITS DIFFER");
      }
    }
    printf("  SUM stock %.3f ms (%.1f GB/s); best gate '%s' + best down '%s' %.3f ms (%+.1f %%)\n", (t_ref_g + t_ref_d) * 1e3, (bytes_g + bytes_d) / (t_ref_g + t_ref_d) / 1e9, bg.c_str(), bd.c_str(), (best_g + best_d) * 1e3,
           ((t_ref_g + t_ref_d) / (best_g + best_d) - 1) * 100);
    for (int pk = 0; pk < 3; ++pk) for (auto& p : plans[pk]) { cudaFree(p.items); cudaFree(p.counts); cudaFree(p.members); }
    cudaFree(x); cudaFree(act); cudaFree(y);
  }
  return 0;
}
