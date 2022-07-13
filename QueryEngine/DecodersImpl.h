/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    DecodersImpl.h
 * @brief
 *
 */

#ifndef QUERYENGINE_DECODERSIMPL_H
#define QUERYENGINE_DECODERSIMPL_H

#include <cstdint>
#include "../Shared/funcannotations.h"
#include "ExtractFromTime.h"

extern "C" DEVICE ALWAYS_INLINE int64_t
SUFFIX(fixed_width_int_decode)(const int8_t* byte_stream,
                               const int32_t byte_width,
                               const int64_t pos) {
#ifdef WITH_DECODERS_BOUNDS_CHECKING
  assert(pos >= 0);
#endif  // WITH_DECODERS_BOUNDS_CHECKING
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
#ifdef _WIN32
      return LLONG_MIN + 1;
#else
      return std::numeric_limits<int64_t>::min() + 1;
#endif
#endif
  }
}

extern "C" DEVICE ALWAYS_INLINE int64_t
SUFFIX(fixed_width_unsigned_decode)(const int8_t* byte_stream,
                                    const int32_t byte_width,
                                    const int64_t pos) {
#ifdef WITH_DECODERS_BOUNDS_CHECKING
  assert(pos >= 0);
#endif  // WITH_DECODERS_BOUNDS_CHECKING
  switch (byte_width) {
    case 1:
      return reinterpret_cast<const uint8_t*>(byte_stream)[pos * byte_width];
    case 2:
      return *(reinterpret_cast<const uint16_t*>(&byte_stream[pos * byte_width]));
    case 4:
      return *(reinterpret_cast<const uint32_t*>(&byte_stream[pos * byte_width]));
    case 8:
      return *(reinterpret_cast<const uint64_t*>(&byte_stream[pos * byte_width]));
    default:
// TODO(alex)
#ifdef __CUDACC__
      return -1;
#else
#ifdef _WIN32
      return LLONG_MIN + 1;
#else
      return std::numeric_limits<int64_t>::min() + 1;
#endif
#endif
  }
}

extern "C" DEVICE NEVER_INLINE int64_t
SUFFIX(fixed_width_int_decode_noinline)(const int8_t* byte_stream,
                                        const int32_t byte_width,
                                        const int64_t pos) {
  return SUFFIX(fixed_width_int_decode)(byte_stream, byte_width, pos);
}

extern "C" DEVICE NEVER_INLINE int64_t
SUFFIX(fixed_width_unsigned_decode_noinline)(const int8_t* byte_stream,
                                             const int32_t byte_width,
                                             const int64_t pos) {
  return SUFFIX(fixed_width_unsigned_decode)(byte_stream, byte_width, pos);
}

extern "C" DEVICE ALWAYS_INLINE int64_t
SUFFIX(diff_fixed_width_int_decode)(const int8_t* byte_stream,
                                    const int32_t byte_width,
                                    const int64_t baseline,
                                    const int64_t pos) {
  return SUFFIX(fixed_width_int_decode)(byte_stream, byte_width, pos) + baseline;
}

extern "C" DEVICE ALWAYS_INLINE float SUFFIX(
    fixed_width_float_decode)(const int8_t* byte_stream, const int64_t pos) {
#ifdef WITH_DECODERS_BOUNDS_CHECKING
  assert(pos >= 0);
#endif  // WITH_DECODERS_BOUNDS_CHECKING
  return *(reinterpret_cast<const float*>(&byte_stream[pos * sizeof(float)]));
}

extern "C" DEVICE NEVER_INLINE float SUFFIX(
    fixed_width_float_decode_noinline)(const int8_t* byte_stream, const int64_t pos) {
  return SUFFIX(fixed_width_float_decode)(byte_stream, pos);
}

extern "C" DEVICE ALWAYS_INLINE double SUFFIX(
    fixed_width_double_decode)(const int8_t* byte_stream, const int64_t pos) {
#ifdef WITH_DECODERS_BOUNDS_CHECKING
  assert(pos >= 0);
#endif  // WITH_DECODERS_BOUNDS_CHECKING
  return *(reinterpret_cast<const double*>(&byte_stream[pos * sizeof(double)]));
}

extern "C" DEVICE NEVER_INLINE double SUFFIX(
    fixed_width_double_decode_noinline)(const int8_t* byte_stream, const int64_t pos) {
  return SUFFIX(fixed_width_double_decode)(byte_stream, pos);
}

extern "C" DEVICE ALWAYS_INLINE int64_t
SUFFIX(fixed_width_small_date_decode)(const int8_t* byte_stream,
                                      const int32_t byte_width,
                                      const int32_t null_val,
                                      const int64_t ret_null_val,
                                      const int64_t pos) {
  auto val = SUFFIX(fixed_width_int_decode)(byte_stream, byte_width, pos);
  return val == null_val ? ret_null_val : val * kSecsPerDay;
}

extern "C" DEVICE NEVER_INLINE int64_t
SUFFIX(fixed_width_small_date_decode_noinline)(const int8_t* byte_stream,
                                               const int32_t byte_width,
                                               const int32_t null_val,
                                               const int64_t ret_null_val,
                                               const int64_t pos) {
  return SUFFIX(fixed_width_small_date_decode)(
      byte_stream, byte_width, null_val, ret_null_val, pos);
}

extern "C" DEVICE ALWAYS_INLINE int64_t
SUFFIX(fixed_width_date_encode)(const int64_t cur_col_val,
                                const int32_t ret_null_val,
                                const int64_t null_val) {
  return cur_col_val == null_val ? ret_null_val : cur_col_val / kSecsPerDay;
}

extern "C" DEVICE NEVER_INLINE int64_t
SUFFIX(fixed_width_date_encode_noinline)(const int64_t cur_col_val,
                                         const int32_t ret_null_val,
                                         const int64_t null_val) {
  return SUFFIX(fixed_width_date_encode)(cur_col_val, ret_null_val, null_val);
}

#undef SUFFIX

#endif  // QUERYENGINE_DECODERSIMPL_H
