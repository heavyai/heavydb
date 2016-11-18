/*
 * @file    DecodersImpl.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_DECODERSIMPL_H
#define QUERYENGINE_DECODERSIMPL_H

#include "../Shared/funcannotations.h"
#include <stdint.h>

extern "C" DEVICE ALWAYS_INLINE int64_t SUFFIX(fixed_width_int_decode)(const int8_t* byte_stream,
                                                                       const int32_t byte_width,
                                                                       const int64_t pos) {
  switch (byte_width) {
    case 1:
      return static_cast<int64_t>(byte_stream[pos * byte_width]);
    case 2:
      return *(reinterpret_cast<const int16_t*>(&byte_stream[pos * byte_width]));
    case 4:
      return *(reinterpret_cast<const int32_t*>(&byte_stream[pos * byte_width]));
    case 8:
      return *(reinterpret_cast<const int64_t*>(&byte_stream[pos * byte_width]));
    default:
// TODO(alex)
#ifdef __CUDACC__
      return -1;
#else
      return std::numeric_limits<int64_t>::min() + 1;
#endif
  }
}

extern "C" DEVICE NEVER_INLINE int64_t SUFFIX(fixed_width_int_decode_noinline)(const int8_t* byte_stream,
                                                                               const int32_t byte_width,
                                                                               const int64_t pos) {
  return SUFFIX(fixed_width_int_decode)(byte_stream, byte_width, pos);
}

extern "C" DEVICE ALWAYS_INLINE int64_t SUFFIX(diff_fixed_width_int_decode)(const int8_t* byte_stream,
                                                                            const int32_t byte_width,
                                                                            const int64_t baseline,
                                                                            const int64_t pos) {
  return SUFFIX(fixed_width_int_decode)(byte_stream, byte_width, pos) + baseline;
}

extern "C" DEVICE ALWAYS_INLINE float SUFFIX(fixed_width_float_decode)(const int8_t* byte_stream, const int64_t pos) {
  return *(reinterpret_cast<const float*>(&byte_stream[pos * sizeof(float)]));
}

extern "C" DEVICE NEVER_INLINE float SUFFIX(fixed_width_float_decode_noinline)(const int8_t* byte_stream,
                                                                               const int64_t pos) {
  return SUFFIX(fixed_width_float_decode)(byte_stream, pos);
}

extern "C" DEVICE ALWAYS_INLINE double SUFFIX(fixed_width_double_decode)(const int8_t* byte_stream, const int64_t pos) {
  return *(reinterpret_cast<const double*>(&byte_stream[pos * sizeof(double)]));
}

extern "C" DEVICE NEVER_INLINE double SUFFIX(fixed_width_double_decode_noinline)(const int8_t* byte_stream,
                                                                                 const int64_t pos) {
  return SUFFIX(fixed_width_double_decode)(byte_stream, pos);
}

#undef SUFFIX

#endif  // QUERYENGINE_DECODERSIMPL_H
