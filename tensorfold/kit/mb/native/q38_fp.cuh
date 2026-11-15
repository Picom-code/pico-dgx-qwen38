// q38: E2M1 pairs and E4M3 scale pairs for the NVFP4 expert kernels. Default: the stock arithmetic (bf16x2 FMA widening, f16 cvt).
// Q38_NATIVE_FP on an arch-specific sm_120+ target: PTX ISA 9.2's one-instruction decodes (the route upstream TensorFold's
// -Dnative-fp-decode takes for other kernels, common/fp_pairs.cuh). Same values for every code except E4M3 NaN bytes (0x7F, 0xFF).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#if defined(Q38_NATIVE_FP) && (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
#define Q38_FP_NATIVE 1
#else
#define Q38_FP_NATIVE 0
#endif

namespace q38_fp {

// stock: two E2M1 codes, bits s..s+3 of each 16-bit half of w, as a bf16x2 pair
__device__ __forceinline__ uint32_t sw_fp4pair(uint32_t w, int s) {
  const uint32_t v = w >> s;
  const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
  return r;
}

// stock: one E4M3 code as a float
__device__ __forceinline__ float sw_e4m3f(uint32_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

#if Q38_FP_NATIVE
__device__ __forceinline__ uint32_t e2m1x2(uint32_t b) {
  uint32_t r;
  asm("{ .reg .b8 a; cvt.u8.u32 a, %1; cvt.rn.bf16x2.e2m1x2 %0, a; }" : "=r"(r) : "r"(b));
  return r;
}
__device__ __forceinline__ uint32_t e4m3x2(uint32_t h) {
  uint32_t r;
  asm("{ .reg .b16 a; cvt.u16.u32 a, %1; cvt.rn.bf16x2.e4m3x2 %0, a; }" : "=r"(r) : "r"(h));
  return r;
}
// s is 0, 4, 8 or 12: the nibbles at bits s..s+3 and 16+s..19+s, gathered into one byte (low nibble = low half)
__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
  const uint32_t n = (w >> (s & 4)) & 0x0F0F0F0Fu;
  const uint32_t x = n | (n >> 12);
  return e2m1x2(x >> (s & 8));
}
// the E4M3 codes in the low two bytes of v as floats (bf16 holds an E4M3 value exactly; bf16 to float is a shift)
__device__ __forceinline__ void e4m3f2(uint32_t v, float& s0, float& s1) {
  const uint32_t p = e4m3x2(v);
  s0 = __uint_as_float(p << 16);
  s1 = __uint_as_float(p & 0xFFFF0000u);
}
#else
__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) { return sw_fp4pair(w, s); }
__device__ __forceinline__ void e4m3f2(uint32_t v, float& s0, float& s1) {
  s0 = sw_e4m3f(v & 0xFFu);
  s1 = sw_e4m3f((v >> 8) & 0xFFu);
}
#endif

}  // namespace q38_fp
