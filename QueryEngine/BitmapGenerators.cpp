#include "BitmapGenerators.h"
#include <immintrin.h>

#ifdef __AVX512F__
size_t __attribute__((target("avx512bw", "avx512f"), optimize("no-tree-vectorize")))
gen_null_bitmap_8(uint8_t* dst, const uint8_t* src, size_t size, const uint8_t null_val) {
  __m512i nulls_mask = _mm512_set1_epi8(reinterpret_cast<const int8_t&>(null_val));

  size_t null_count = 0;
  while (size > 0) {
    __m512d loaded_bytes = _mm512_loadu_pd(static_cast<const void*>(src));

    __mmask64 k1 =
        _mm512_cmpneq_epi8_mask(*reinterpret_cast<__m512i*>(&loaded_bytes), nulls_mask);

    *reinterpret_cast<__mmask64*>(dst) = k1;

    null_count += _mm_popcnt_u64(~k1);

    dst += 64 / 8;
    src += 64 / sizeof(uint8_t);

    size -= 64;
  }

  return null_count;
}

size_t __attribute__((target("avx512bw", "avx512f"), optimize("no-tree-vectorize")))
gen_null_bitmap_16(uint8_t* dst,
                   const uint16_t* src,
                   size_t size,
                   const uint16_t null_val) {
  __m512i nulls_mask = _mm512_set1_epi16(reinterpret_cast<const int16_t&>(null_val));

  size_t null_count = 0;
  while (size > 0) {
    __m512d loaded_bytes = _mm512_loadu_pd(static_cast<const void*>(src));

    __mmask32 k1 =
        _mm512_cmpneq_epi16_mask(*reinterpret_cast<__m512i*>(&loaded_bytes), nulls_mask);

    *reinterpret_cast<__mmask32*>(dst) = k1;

    null_count += _mm_popcnt_u32(~k1);

    dst += 64 / 16;
    src += 64 / sizeof(uint16_t);
    size -= 64 / sizeof(uint16_t);
    ;
  }

  return null_count;
}

size_t __attribute__((target("avx512bw", "avx512f"), optimize("no-tree-vectorize")))
gen_null_bitmap_32(uint8_t* dst,
                   const uint32_t* src,
                   size_t size,
                   const uint32_t null_val) {
  __m512i nulls_mask = _mm512_set1_epi32(reinterpret_cast<const int32_t&>(null_val));

  size_t null_count = 0;
  while (size > 0) {
    __m512d loaded_bytes = _mm512_loadu_pd(static_cast<const void*>(src));

    __mmask16 k1 =
        _mm512_cmpneq_epi32_mask(*reinterpret_cast<__m512i*>(&loaded_bytes), nulls_mask);

    *reinterpret_cast<__mmask16*>(dst) = k1;

    null_count += _mm_popcnt_u32((~k1) & 0xFFFF);

    dst += 64 / 32;
    src += 64 / sizeof(uint32_t);
    size -= 64 / sizeof(uint32_t);
  }

  return null_count;
}

size_t __attribute__((target("avx512bw", "avx512f"), optimize("no-tree-vectorize")))
gen_null_bitmap_64(uint8_t* dst,
                   const uint64_t* src,
                   size_t size,
                   const uint64_t null_val) {
  __m512i nulls_mask = _mm512_set1_epi64(reinterpret_cast<const int64_t&>(null_val));

  size_t null_count = 0;
  while (size > 0) {
    __m512d loaded_bytes = _mm512_loadu_pd(static_cast<const void*>(src));

    __mmask8 k1 =
        _mm512_cmpneq_epi64_mask(*reinterpret_cast<__m512i*>(&loaded_bytes), nulls_mask);

    *reinterpret_cast<__mmask8*>(dst) = k1;

    null_count += _mm_popcnt_u32((~k1) & 0xFF);

    ++dst;
    src += 64 / sizeof(uint64_t);
    size -= 64 / sizeof(uint64_t);
  }

  return null_count;
}
#endif

template <typename TYPE>
size_t gen_null_bitmap_default(uint8_t* dst,
                               const TYPE* src,
                               size_t size,
                               const TYPE null_val) {
  size_t null_count = 0;
  TYPE loaded_data[8];

  while (size > 0) {
    uint8_t encoded_byte = 0;
    memcpy(loaded_data, src, 8 * sizeof(TYPE));

    for (size_t i = 0; i < 8; i++) {
      uint8_t is_null = loaded_data[i] == null_val;
      encoded_byte |= (!is_null) << i;
      null_count += is_null;
    }
    *dst = encoded_byte;

    dst += 1;
    src += 8;
    size -= 8;
  }
  return null_count;
}

size_t __attribute__((target("default")))
gen_null_bitmap_8(uint8_t* dst, const uint8_t* src, size_t size, const uint8_t null_val) {
  return gen_null_bitmap_default<uint8_t>(dst, src, size, null_val);
}

size_t __attribute__((target("default"))) gen_null_bitmap_16(uint8_t* dst,
                                                             const uint16_t* src,
                                                             size_t size,
                                                             const uint16_t null_val) {
  return gen_null_bitmap_default<uint16_t>(dst, src, size, null_val);
}

size_t __attribute__((target("default"))) gen_null_bitmap_32(uint8_t* dst,
                                                             const uint32_t* src,
                                                             size_t size,
                                                             const uint32_t null_val) {
  return gen_null_bitmap_default<uint32_t>(dst, src, size, null_val);
}

size_t __attribute__((target("default"))) gen_null_bitmap_64(uint8_t* dst,
                                                             const uint64_t* src,
                                                             size_t size,
                                                             const uint64_t null_val) {
  return gen_null_bitmap_default<uint64_t>(dst, src, size, null_val);
}
