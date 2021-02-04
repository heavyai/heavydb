#ifdef HAVE_DCPMM

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "Pmem.h"

extern "C" {

void PmemFlush(const void* addr, size_t len) {
  uintptr_t ptr;
  for (ptr = (uintptr_t)addr & ~(CACHELINE_SIZE - 1); ptr < (uintptr_t)addr + len;
       ptr += CACHELINE_SIZE) {
    PmemClwb((char*)ptr);
  }
}

void PmemPersist(const void* addr, size_t len) {
  uintptr_t ptr;
  for (ptr = (uintptr_t)addr & ~(CACHELINE_SIZE - 1); ptr < (uintptr_t)addr + len;
       ptr += CACHELINE_SIZE) {
    PmemClwb((char*)ptr);
  }
  PmemFence();
}

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("avx512f")
#else /* __GNUC__ */
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("avx512f"))), apply_to = function)
#endif /* __clang__ */
#endif /* __GNUC__ */

static __attribute__((always_inline)) __m128i m256_get16b(__m256i ymm) {
  return _mm256_extractf128_si256(ymm, 0);
}

static inline __attribute__((always_inline)) uint64_t m256_get8b(__m256i ymm) {
  return (uint64_t)_mm256_extract_epi64(ymm, 0);
}

static inline __attribute__((always_inline)) uint32_t m256_get4b(__m256i ymm) {
  return (uint32_t)_mm256_extract_epi32(ymm, 0);
}

static inline __attribute__((always_inline)) uint16_t m256_get2b(__m256i ymm) {
  return (uint16_t)_mm256_extract_epi16(ymm, 0);
}

static inline __attribute__((always_inline)) int util_is_pow2(uint64_t v) {
  return v && !(v & (v - 1));
}

static inline __attribute__((always_inline)) void avx_zeroupper(void) {
  _mm256_zeroupper();
}

static inline __attribute__((always_inline)) void memmove_small_avx512f(char* dest,
                                                                        const char* src,
                                                                        size_t len) {
  if (len > 32) {
    /* 33..64 */
    __m256i ymm0 = _mm256_loadu_si256((__m256i*)src);
    __m256i ymm1 = _mm256_loadu_si256((__m256i*)(src + len - 32));

    _mm256_storeu_si256((__m256i*)dest, ymm0);
    _mm256_storeu_si256((__m256i*)(dest + len - 32), ymm1);
    return;
  }

  if (len > 8) {
    if (len > 16) {
      /* 17..32 */
      __m128i xmm0 = _mm_loadu_si128((__m128i*)src);
      __m128i xmm1 = _mm_loadu_si128((__m128i*)(src + len - 16));

      _mm_storeu_si128((__m128i*)dest, xmm0);
      _mm_storeu_si128((__m128i*)(dest + len - 16), xmm1);
      return;
    }

    /* 9..16 */
    uint64_t d80 = *(uint64_t*)src;
    uint64_t d81 = *(uint64_t*)(src + len - 8);

    *(uint64_t*)dest = d80;
    *(uint64_t*)(dest + len - 8) = d81;
    return;
  }

  if (len > 2) {
    if (len > 4) {
      /* 5..8 */
      uint32_t d40 = *(uint32_t*)src;
      uint32_t d41 = *(uint32_t*)(src + len - 4);

      *(uint32_t*)dest = d40;
      *(uint32_t*)(dest + len - 4) = d41;
      return;
    }

    /* 3..4 */
    uint16_t d20 = *(uint16_t*)src;
    uint16_t d21 = *(uint16_t*)(src + len - 2);

    *(uint16_t*)dest = d20;
    *(uint16_t*)(dest + len - 2) = d21;
    return;
  }

  if (len == 2) {
    *(uint16_t*)dest = *(uint16_t*)src;
    return;
  }

  *(uint8_t*)dest = *(uint8_t*)src;
}

static inline __attribute__((always_inline)) void memmove_movnt32x64b(char* dest,
                                                                      const char* src) {
  __m512i zmm0 = _mm512_loadu_si512((__m512i*)src + 0);
  __m512i zmm1 = _mm512_loadu_si512((__m512i*)src + 1);
  __m512i zmm2 = _mm512_loadu_si512((__m512i*)src + 2);
  __m512i zmm3 = _mm512_loadu_si512((__m512i*)src + 3);
  __m512i zmm4 = _mm512_loadu_si512((__m512i*)src + 4);
  __m512i zmm5 = _mm512_loadu_si512((__m512i*)src + 5);
  __m512i zmm6 = _mm512_loadu_si512((__m512i*)src + 6);
  __m512i zmm7 = _mm512_loadu_si512((__m512i*)src + 7);
  __m512i zmm8 = _mm512_loadu_si512((__m512i*)src + 8);
  __m512i zmm9 = _mm512_loadu_si512((__m512i*)src + 9);
  __m512i zmm10 = _mm512_loadu_si512((__m512i*)src + 10);
  __m512i zmm11 = _mm512_loadu_si512((__m512i*)src + 11);
  __m512i zmm12 = _mm512_loadu_si512((__m512i*)src + 12);
  __m512i zmm13 = _mm512_loadu_si512((__m512i*)src + 13);
  __m512i zmm14 = _mm512_loadu_si512((__m512i*)src + 14);
  __m512i zmm15 = _mm512_loadu_si512((__m512i*)src + 15);
  __m512i zmm16 = _mm512_loadu_si512((__m512i*)src + 16);
  __m512i zmm17 = _mm512_loadu_si512((__m512i*)src + 17);
  __m512i zmm18 = _mm512_loadu_si512((__m512i*)src + 18);
  __m512i zmm19 = _mm512_loadu_si512((__m512i*)src + 19);
  __m512i zmm20 = _mm512_loadu_si512((__m512i*)src + 20);
  __m512i zmm21 = _mm512_loadu_si512((__m512i*)src + 21);
  __m512i zmm22 = _mm512_loadu_si512((__m512i*)src + 22);
  __m512i zmm23 = _mm512_loadu_si512((__m512i*)src + 23);
  __m512i zmm24 = _mm512_loadu_si512((__m512i*)src + 24);
  __m512i zmm25 = _mm512_loadu_si512((__m512i*)src + 25);
  __m512i zmm26 = _mm512_loadu_si512((__m512i*)src + 26);
  __m512i zmm27 = _mm512_loadu_si512((__m512i*)src + 27);
  __m512i zmm28 = _mm512_loadu_si512((__m512i*)src + 28);
  __m512i zmm29 = _mm512_loadu_si512((__m512i*)src + 29);
  __m512i zmm30 = _mm512_loadu_si512((__m512i*)src + 30);
  __m512i zmm31 = _mm512_loadu_si512((__m512i*)src + 31);

  _mm512_stream_si512((__m512i*)dest + 0, zmm0);
  _mm512_stream_si512((__m512i*)dest + 1, zmm1);
  _mm512_stream_si512((__m512i*)dest + 2, zmm2);
  _mm512_stream_si512((__m512i*)dest + 3, zmm3);
  _mm512_stream_si512((__m512i*)dest + 4, zmm4);
  _mm512_stream_si512((__m512i*)dest + 5, zmm5);
  _mm512_stream_si512((__m512i*)dest + 6, zmm6);
  _mm512_stream_si512((__m512i*)dest + 7, zmm7);
  _mm512_stream_si512((__m512i*)dest + 8, zmm8);
  _mm512_stream_si512((__m512i*)dest + 9, zmm9);
  _mm512_stream_si512((__m512i*)dest + 10, zmm10);
  _mm512_stream_si512((__m512i*)dest + 11, zmm11);
  _mm512_stream_si512((__m512i*)dest + 12, zmm12);
  _mm512_stream_si512((__m512i*)dest + 13, zmm13);
  _mm512_stream_si512((__m512i*)dest + 14, zmm14);
  _mm512_stream_si512((__m512i*)dest + 15, zmm15);
  _mm512_stream_si512((__m512i*)dest + 16, zmm16);
  _mm512_stream_si512((__m512i*)dest + 17, zmm17);
  _mm512_stream_si512((__m512i*)dest + 18, zmm18);
  _mm512_stream_si512((__m512i*)dest + 19, zmm19);
  _mm512_stream_si512((__m512i*)dest + 20, zmm20);
  _mm512_stream_si512((__m512i*)dest + 21, zmm21);
  _mm512_stream_si512((__m512i*)dest + 22, zmm22);
  _mm512_stream_si512((__m512i*)dest + 23, zmm23);
  _mm512_stream_si512((__m512i*)dest + 24, zmm24);
  _mm512_stream_si512((__m512i*)dest + 25, zmm25);
  _mm512_stream_si512((__m512i*)dest + 26, zmm26);
  _mm512_stream_si512((__m512i*)dest + 27, zmm27);
  _mm512_stream_si512((__m512i*)dest + 28, zmm28);
  _mm512_stream_si512((__m512i*)dest + 29, zmm29);
  _mm512_stream_si512((__m512i*)dest + 30, zmm30);
  _mm512_stream_si512((__m512i*)dest + 31, zmm31);
}

static inline __attribute__((always_inline)) void memmove_movnt16x64b(char* dest,
                                                                      const char* src) {
  __m512i zmm0 = _mm512_loadu_si512((__m512i*)src + 0);
  __m512i zmm1 = _mm512_loadu_si512((__m512i*)src + 1);
  __m512i zmm2 = _mm512_loadu_si512((__m512i*)src + 2);
  __m512i zmm3 = _mm512_loadu_si512((__m512i*)src + 3);
  __m512i zmm4 = _mm512_loadu_si512((__m512i*)src + 4);
  __m512i zmm5 = _mm512_loadu_si512((__m512i*)src + 5);
  __m512i zmm6 = _mm512_loadu_si512((__m512i*)src + 6);
  __m512i zmm7 = _mm512_loadu_si512((__m512i*)src + 7);
  __m512i zmm8 = _mm512_loadu_si512((__m512i*)src + 8);
  __m512i zmm9 = _mm512_loadu_si512((__m512i*)src + 9);
  __m512i zmm10 = _mm512_loadu_si512((__m512i*)src + 10);
  __m512i zmm11 = _mm512_loadu_si512((__m512i*)src + 11);
  __m512i zmm12 = _mm512_loadu_si512((__m512i*)src + 12);
  __m512i zmm13 = _mm512_loadu_si512((__m512i*)src + 13);
  __m512i zmm14 = _mm512_loadu_si512((__m512i*)src + 14);
  __m512i zmm15 = _mm512_loadu_si512((__m512i*)src + 15);

  _mm512_stream_si512((__m512i*)dest + 0, zmm0);
  _mm512_stream_si512((__m512i*)dest + 1, zmm1);
  _mm512_stream_si512((__m512i*)dest + 2, zmm2);
  _mm512_stream_si512((__m512i*)dest + 3, zmm3);
  _mm512_stream_si512((__m512i*)dest + 4, zmm4);
  _mm512_stream_si512((__m512i*)dest + 5, zmm5);
  _mm512_stream_si512((__m512i*)dest + 6, zmm6);
  _mm512_stream_si512((__m512i*)dest + 7, zmm7);
  _mm512_stream_si512((__m512i*)dest + 8, zmm8);
  _mm512_stream_si512((__m512i*)dest + 9, zmm9);
  _mm512_stream_si512((__m512i*)dest + 10, zmm10);
  _mm512_stream_si512((__m512i*)dest + 11, zmm11);
  _mm512_stream_si512((__m512i*)dest + 12, zmm12);
  _mm512_stream_si512((__m512i*)dest + 13, zmm13);
  _mm512_stream_si512((__m512i*)dest + 14, zmm14);
  _mm512_stream_si512((__m512i*)dest + 15, zmm15);
}

static inline __attribute__((always_inline)) void memmove_movnt8x64b(char* dest,
                                                                     const char* src) {
  __m512i zmm0 = _mm512_loadu_si512((__m512i*)src + 0);
  __m512i zmm1 = _mm512_loadu_si512((__m512i*)src + 1);
  __m512i zmm2 = _mm512_loadu_si512((__m512i*)src + 2);
  __m512i zmm3 = _mm512_loadu_si512((__m512i*)src + 3);
  __m512i zmm4 = _mm512_loadu_si512((__m512i*)src + 4);
  __m512i zmm5 = _mm512_loadu_si512((__m512i*)src + 5);
  __m512i zmm6 = _mm512_loadu_si512((__m512i*)src + 6);
  __m512i zmm7 = _mm512_loadu_si512((__m512i*)src + 7);

  _mm512_stream_si512((__m512i*)dest + 0, zmm0);
  _mm512_stream_si512((__m512i*)dest + 1, zmm1);
  _mm512_stream_si512((__m512i*)dest + 2, zmm2);
  _mm512_stream_si512((__m512i*)dest + 3, zmm3);
  _mm512_stream_si512((__m512i*)dest + 4, zmm4);
  _mm512_stream_si512((__m512i*)dest + 5, zmm5);
  _mm512_stream_si512((__m512i*)dest + 6, zmm6);
  _mm512_stream_si512((__m512i*)dest + 7, zmm7);
}

static inline __attribute__((always_inline)) void memmove_movnt4x64b(char* dest,
                                                                     const char* src) {
  __m512i zmm0 = _mm512_loadu_si512((__m512i*)src + 0);
  __m512i zmm1 = _mm512_loadu_si512((__m512i*)src + 1);
  __m512i zmm2 = _mm512_loadu_si512((__m512i*)src + 2);
  __m512i zmm3 = _mm512_loadu_si512((__m512i*)src + 3);

  _mm512_stream_si512((__m512i*)dest + 0, zmm0);
  _mm512_stream_si512((__m512i*)dest + 1, zmm1);
  _mm512_stream_si512((__m512i*)dest + 2, zmm2);
  _mm512_stream_si512((__m512i*)dest + 3, zmm3);
}

static inline __attribute__((always_inline)) void memmove_movnt2x64b(char* dest,
                                                                     const char* src) {
  __m512i zmm0 = _mm512_loadu_si512((__m512i*)src + 0);
  __m512i zmm1 = _mm512_loadu_si512((__m512i*)src + 1);

  _mm512_stream_si512((__m512i*)dest + 0, zmm0);
  _mm512_stream_si512((__m512i*)dest + 1, zmm1);
}

static inline __attribute__((always_inline)) void memmove_movnt1x64b(char* dest,
                                                                     const char* src) {
  __m512i zmm0 = _mm512_loadu_si512((__m512i*)src + 0);

  _mm512_stream_si512((__m512i*)dest + 0, zmm0);
}

static inline __attribute__((always_inline)) void memmove_movnt1x32b(char* dest,
                                                                     const char* src) {
  __m256i zmm0 = _mm256_loadu_si256((__m256i*)src);

  _mm256_stream_si256((__m256i*)dest, zmm0);
}

static inline __attribute__((always_inline)) void memmove_movnt1x16b(char* dest,
                                                                     const char* src) {
  __m128i ymm0 = _mm_loadu_si128((__m128i*)src);

  _mm_stream_si128((__m128i*)dest, ymm0);
}

static inline __attribute__((always_inline)) void memmove_movnt1x8b(char* dest,
                                                                    const char* src) {
  _mm_stream_si64((long long*)dest, *(long long*)src);
}

static inline __attribute__((always_inline)) void memmove_movnt1x4b(char* dest,
                                                                    const char* src) {
  _mm_stream_si32((int*)dest, *(int*)src);
}

static inline __attribute__((always_inline)) void
memmove_movnt_avx512f_fw(char* dest, const char* src, size_t len) {
  size_t cnt = (uint64_t)dest & 63;
  if (cnt > 0) {
    cnt = 64 - cnt;

    if (cnt > len)
      cnt = len;

    memmove_small_avx512f(dest, src, cnt);

    dest += cnt;
    src += cnt;
    len -= cnt;
  }

  while (len >= 32 * 64) {
    memmove_movnt32x64b(dest, src);
    dest += 32 * 64;
    src += 32 * 64;
    len -= 32 * 64;
  }

  if (len >= 16 * 64) {
    memmove_movnt16x64b(dest, src);
    dest += 16 * 64;
    src += 16 * 64;
    len -= 16 * 64;
  }

  if (len >= 8 * 64) {
    memmove_movnt8x64b(dest, src);
    dest += 8 * 64;
    src += 8 * 64;
    len -= 8 * 64;
  }

  if (len >= 4 * 64) {
    memmove_movnt4x64b(dest, src);
    dest += 4 * 64;
    src += 4 * 64;
    len -= 4 * 64;
  }

  if (len >= 2 * 64) {
    memmove_movnt2x64b(dest, src);
    dest += 2 * 64;
    src += 2 * 64;
    len -= 2 * 64;
  }

  if (len >= 1 * 64) {
    memmove_movnt1x64b(dest, src);

    dest += 1 * 64;
    src += 1 * 64;
    len -= 1 * 64;
  }

  if (len == 0)
    goto end;

  /* There's no point in using more than 1 nt store for 1 cache line. */
  if (util_is_pow2(len)) {
    if (len == 32)
      memmove_movnt1x32b(dest, src);
    else if (len == 16)
      memmove_movnt1x16b(dest, src);
    else if (len == 8)
      memmove_movnt1x8b(dest, src);
    else if (len == 4)
      memmove_movnt1x4b(dest, src);
    else
      goto nonnt;

    goto end;
  }

nonnt:
  memmove_small_avx512f(dest, src, len);
end:
  avx_zeroupper();
}

static inline __attribute__((always_inline)) void
memmove_movnt_avx512f_bw(char* dest, const char* src, size_t len) {
  dest += len;
  src += len;

  size_t cnt = (uint64_t)dest & 63;
  if (cnt > 0) {
    if (cnt > len)
      cnt = len;

    dest -= cnt;
    src -= cnt;
    len -= cnt;

    memmove_small_avx512f(dest, src, cnt);
  }

  while (len >= 32 * 64) {
    dest -= 32 * 64;
    src -= 32 * 64;
    len -= 32 * 64;
    memmove_movnt32x64b(dest, src);
  }

  if (len >= 16 * 64) {
    dest -= 16 * 64;
    src -= 16 * 64;
    len -= 16 * 64;
    memmove_movnt16x64b(dest, src);
  }

  if (len >= 8 * 64) {
    dest -= 8 * 64;
    src -= 8 * 64;
    len -= 8 * 64;
    memmove_movnt8x64b(dest, src);
  }

  if (len >= 4 * 64) {
    dest -= 4 * 64;
    src -= 4 * 64;
    len -= 4 * 64;
    memmove_movnt4x64b(dest, src);
  }

  if (len >= 2 * 64) {
    dest -= 2 * 64;
    src -= 2 * 64;
    len -= 2 * 64;
    memmove_movnt2x64b(dest, src);
  }

  if (len >= 1 * 64) {
    dest -= 1 * 64;
    src -= 1 * 64;
    len -= 1 * 64;

    memmove_movnt1x64b(dest, src);
  }

  if (len == 0)
    goto end;

  /* There's no point in using more than 1 nt store for 1 cache line. */
  if (util_is_pow2(len)) {
    if (len == 32) {
      dest -= 32;
      src -= 32;
      memmove_movnt1x32b(dest, src);
    } else if (len == 16) {
      dest -= 16;
      src -= 16;
      memmove_movnt1x16b(dest, src);
    } else if (len == 8) {
      dest -= 8;
      src -= 8;
      memmove_movnt1x8b(dest, src);
    } else if (len == 4) {
      dest -= 4;
      src -= 4;
      memmove_movnt1x4b(dest, src);
    } else {
      goto nonnt;
    }

    goto end;
  }

nonnt:
  dest -= len;
  src -= len;

  memmove_small_avx512f(dest, src, len);
end:
  avx_zeroupper();
}

void PmemMemCpy(char* dest, const char* src, const size_t len) {
  if ((uintptr_t)dest - (uintptr_t)src >= len)
    memmove_movnt_avx512f_fw(dest, src, len);
  else
    memmove_movnt_avx512f_bw(dest, src, len);
  PmemFence();
}

static inline __attribute__((always_inline)) void memset_movnt32x64b(char* dest,
                                                                     __m512i zmm) {
  _mm512_stream_si512((__m512i*)dest + 0, zmm);
  _mm512_stream_si512((__m512i*)dest + 1, zmm);
  _mm512_stream_si512((__m512i*)dest + 2, zmm);
  _mm512_stream_si512((__m512i*)dest + 3, zmm);
  _mm512_stream_si512((__m512i*)dest + 4, zmm);
  _mm512_stream_si512((__m512i*)dest + 5, zmm);
  _mm512_stream_si512((__m512i*)dest + 6, zmm);
  _mm512_stream_si512((__m512i*)dest + 7, zmm);
  _mm512_stream_si512((__m512i*)dest + 8, zmm);
  _mm512_stream_si512((__m512i*)dest + 9, zmm);
  _mm512_stream_si512((__m512i*)dest + 10, zmm);
  _mm512_stream_si512((__m512i*)dest + 11, zmm);
  _mm512_stream_si512((__m512i*)dest + 12, zmm);
  _mm512_stream_si512((__m512i*)dest + 13, zmm);
  _mm512_stream_si512((__m512i*)dest + 14, zmm);
  _mm512_stream_si512((__m512i*)dest + 15, zmm);
  _mm512_stream_si512((__m512i*)dest + 16, zmm);
  _mm512_stream_si512((__m512i*)dest + 17, zmm);
  _mm512_stream_si512((__m512i*)dest + 18, zmm);
  _mm512_stream_si512((__m512i*)dest + 19, zmm);
  _mm512_stream_si512((__m512i*)dest + 20, zmm);
  _mm512_stream_si512((__m512i*)dest + 21, zmm);
  _mm512_stream_si512((__m512i*)dest + 22, zmm);
  _mm512_stream_si512((__m512i*)dest + 23, zmm);
  _mm512_stream_si512((__m512i*)dest + 24, zmm);
  _mm512_stream_si512((__m512i*)dest + 25, zmm);
  _mm512_stream_si512((__m512i*)dest + 26, zmm);
  _mm512_stream_si512((__m512i*)dest + 27, zmm);
  _mm512_stream_si512((__m512i*)dest + 28, zmm);
  _mm512_stream_si512((__m512i*)dest + 29, zmm);
  _mm512_stream_si512((__m512i*)dest + 30, zmm);
  _mm512_stream_si512((__m512i*)dest + 31, zmm);
}

static inline __attribute__((always_inline)) void memset_movnt16x64b(char* dest,
                                                                     __m512i zmm) {
  _mm512_stream_si512((__m512i*)dest + 0, zmm);
  _mm512_stream_si512((__m512i*)dest + 1, zmm);
  _mm512_stream_si512((__m512i*)dest + 2, zmm);
  _mm512_stream_si512((__m512i*)dest + 3, zmm);
  _mm512_stream_si512((__m512i*)dest + 4, zmm);
  _mm512_stream_si512((__m512i*)dest + 5, zmm);
  _mm512_stream_si512((__m512i*)dest + 6, zmm);
  _mm512_stream_si512((__m512i*)dest + 7, zmm);
  _mm512_stream_si512((__m512i*)dest + 8, zmm);
  _mm512_stream_si512((__m512i*)dest + 9, zmm);
  _mm512_stream_si512((__m512i*)dest + 10, zmm);
  _mm512_stream_si512((__m512i*)dest + 11, zmm);
  _mm512_stream_si512((__m512i*)dest + 12, zmm);
  _mm512_stream_si512((__m512i*)dest + 13, zmm);
  _mm512_stream_si512((__m512i*)dest + 14, zmm);
  _mm512_stream_si512((__m512i*)dest + 15, zmm);
}

static inline __attribute__((always_inline)) void memset_movnt8x64b(char* dest,
                                                                    __m512i zmm) {
  _mm512_stream_si512((__m512i*)dest + 0, zmm);
  _mm512_stream_si512((__m512i*)dest + 1, zmm);
  _mm512_stream_si512((__m512i*)dest + 2, zmm);
  _mm512_stream_si512((__m512i*)dest + 3, zmm);
  _mm512_stream_si512((__m512i*)dest + 4, zmm);
  _mm512_stream_si512((__m512i*)dest + 5, zmm);
  _mm512_stream_si512((__m512i*)dest + 6, zmm);
  _mm512_stream_si512((__m512i*)dest + 7, zmm);
}

static inline __attribute__((always_inline)) void memset_movnt4x64b(char* dest,
                                                                    __m512i zmm) {
  _mm512_stream_si512((__m512i*)dest + 0, zmm);
  _mm512_stream_si512((__m512i*)dest + 1, zmm);
  _mm512_stream_si512((__m512i*)dest + 2, zmm);
  _mm512_stream_si512((__m512i*)dest + 3, zmm);
}

static inline __attribute__((always_inline)) void memset_movnt2x64b(char* dest,
                                                                    __m512i zmm) {
  _mm512_stream_si512((__m512i*)dest + 0, zmm);
  _mm512_stream_si512((__m512i*)dest + 1, zmm);
}

static inline __attribute__((always_inline)) void memset_movnt1x64b(char* dest,
                                                                    __m512i zmm) {
  _mm512_stream_si512((__m512i*)dest + 0, zmm);
}

static inline __attribute__((always_inline)) void memset_movnt1x32b(char* dest,
                                                                    __m256i ymm) {
  _mm256_stream_si256((__m256i*)dest, ymm);
}

static inline __attribute__((always_inline)) void memset_movnt1x16b(char* dest,
                                                                    __m256i ymm) {
  __m128i xmm = _mm256_extracti128_si256(ymm, 0);

  _mm_stream_si128((__m128i*)dest, xmm);
}

static inline __attribute__((always_inline)) void memset_movnt1x8b(char* dest,
                                                                   __m256i ymm) {
  uint64_t x = m256_get8b(ymm);

  _mm_stream_si64((long long*)dest, (long long)x);
}

static inline __attribute__((always_inline)) void memset_movnt1x4b(char* dest,
                                                                   __m256i ymm) {
  uint32_t x = m256_get4b(ymm);

  _mm_stream_si32((int*)dest, (int)x);
}

static inline __attribute__((always_inline)) void memset_small_avx(char* dest,
                                                                   __m256i ymm,
                                                                   size_t len) {
  if (len > 32) {
    /* 33..64 */
    _mm256_storeu_si256((__m256i*)dest, ymm);
    _mm256_storeu_si256((__m256i*)(dest + len - 32), ymm);
    return;
  }

  if (len > 16) {
    /* 17..32 */
    __m128i xmm = m256_get16b(ymm);

    _mm_storeu_si128((__m128i*)dest, xmm);
    _mm_storeu_si128((__m128i*)(dest + len - 16), xmm);
    return;
  }

  if (len > 8) {
    /* 9..16 */
    uint64_t d8 = m256_get8b(ymm);

    *(uint64_t*)dest = d8;
    *(uint64_t*)(dest + len - 8) = d8;
    return;
  }

  if (len > 4) {
    /* 5..8 */
    uint32_t d = m256_get4b(ymm);

    *(uint32_t*)dest = d;
    *(uint32_t*)(dest + len - 4) = d;
    return;
  }

  if (len > 2) {
    /* 3..4 */
    uint16_t d2 = m256_get2b(ymm);

    *(uint16_t*)dest = d2;
    *(uint16_t*)(dest + len - 2) = d2;
    return;
  }

  if (len == 2) {
    uint16_t d2 = m256_get2b(ymm);

    *(uint16_t*)dest = d2;
    return;
  }

  *(uint8_t*)dest = (uint8_t)m256_get2b(ymm);

  PmemFlush(dest, len);
}

static inline __attribute__((always_inline)) void memset_small_avx512f(char* dest,
                                                                       __m256i ymm,
                                                                       size_t len) {
  /* We can't do better than AVX here. */
  memset_small_avx(dest, ymm, len);
}

void PmemMemSet(char* dest, int c, size_t len) {
  __m512i zmm = _mm512_set1_epi8((char)c);
  /*
   * Can't use _mm512_extracti64x4_epi64, because some versions of gcc
   * crash. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=82887
   */
  __m256i ymm = _mm256_set1_epi8((char)c);

  size_t cnt = (uint64_t)dest & 63;
  if (cnt > 0) {
    cnt = 64 - cnt;

    if (cnt > len)
      cnt = len;

    memset_small_avx512f(dest, ymm, cnt);

    dest += cnt;
    len -= cnt;
  }

  while (len >= 32 * 64) {
    memset_movnt32x64b(dest, zmm);
    dest += 32 * 64;
    len -= 32 * 64;
  }

  if (len >= 16 * 64) {
    memset_movnt16x64b(dest, zmm);
    dest += 16 * 64;
    len -= 16 * 64;
  }

  if (len >= 8 * 64) {
    memset_movnt8x64b(dest, zmm);
    dest += 8 * 64;
    len -= 8 * 64;
  }

  if (len >= 4 * 64) {
    memset_movnt4x64b(dest, zmm);
    dest += 4 * 64;
    len -= 4 * 64;
  }

  if (len >= 2 * 64) {
    memset_movnt2x64b(dest, zmm);
    dest += 2 * 64;
    len -= 2 * 64;
  }

  if (len >= 1 * 64) {
    memset_movnt1x64b(dest, zmm);

    dest += 1 * 64;
    len -= 1 * 64;
  }

  if (len == 0)
    goto end;

  /* There's no point in using more than 1 nt store for 1 cache line. */
  if (util_is_pow2(len)) {
    if (len == 32)
      memset_movnt1x32b(dest, ymm);
    else if (len == 16)
      memset_movnt1x16b(dest, ymm);
    else if (len == 8)
      memset_movnt1x8b(dest, ymm);
    else if (len == 4)
      memset_movnt1x4b(dest, ymm);
    else
      goto nonnt;

    goto end;
  }

nonnt:
  memset_small_avx512f(dest, ymm, len);
end:
  avx_zeroupper();

  PmemFence();
}

#ifdef __GNUC__
#pragma GCC pop_options
#else /* __GNUC__ */
#ifdef __clang__
#pragma clang attribute pop
#endif /* __clang__ */
#endif /* __GNUC__ */
}

#endif /* HAVE_DCPMM */