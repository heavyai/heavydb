/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
      return std::numeric_limits<int64_t>::min() + 1;
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
      return std::numeric_limits<int64_t>::min() + 1;
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

extern "C" DEVICE ALWAYS_INLINE const int8_t* SUFFIX(segmented_column_ptr)(
    const int8_t* descriptor,
    const int64_t pos,
    const int64_t elem_size) {
  const auto header = reinterpret_cast<const uint64_t*>(descriptor);
  const auto fragment_count = header[0];
  const auto index_shift = header[1];
  const auto index_count = header[2];
  const auto entries = header + 3;
  const auto index = entries + 3 * fragment_count;
  const uint64_t target_pos = static_cast<uint64_t>(pos);
  if (index_count > 0) {
    const uint64_t last_fragment_idx = fragment_count - 1;
    uint64_t bucket = index_shift >= 63 ? uint64_t(0) : target_pos >> index_shift;
    if (bucket >= index_count) {
      bucket = index_count - 1;
    }
    uint64_t fragment_idx = index[bucket];
    if (fragment_idx > last_fragment_idx) {
      fragment_idx = last_fragment_idx;
    }
    while (fragment_idx < last_fragment_idx) {
      const auto entry = entries + 3 * fragment_idx;
      const uint64_t start = entry[1];
      const uint64_t row_count = entry[2];
      if (target_pos < start + row_count) {
        break;
      }
      ++fragment_idx;
    }
    while (fragment_idx > 0) {
      const auto entry = entries + 3 * fragment_idx;
      const uint64_t start = entry[1];
      if (target_pos >= start) {
        break;
      }
      --fragment_idx;
    }
    const auto entry = entries + 3 * fragment_idx;
    const uint64_t start = entry[1];
    const uint64_t row_count = entry[2];
    if (target_pos >= start && target_pos < start + row_count) {
      return reinterpret_cast<const int8_t*>(entry[0]) +
             (target_pos - start) * static_cast<uint64_t>(elem_size);
    }
  }
  uint64_t low = 0;
  uint64_t high = fragment_count;
  while (low < high) {
    const uint64_t mid = low + ((high - low) >> 1);
    const auto entry = entries + 3 * mid;
    const uint64_t start = entry[1];
    const uint64_t row_count = entry[2];
    if (target_pos < start) {
      high = mid;
    } else if (target_pos >= start + row_count) {
      low = mid + 1;
    } else {
      return reinterpret_cast<const int8_t*>(entry[0]) +
             (target_pos - start) * static_cast<uint64_t>(elem_size);
    }
  }
  return reinterpret_cast<const int8_t*>(entries[0]);
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

extern "C" DEVICE ALWAYS_INLINE int64_t
SUFFIX(fixed_width_date_decode)(const int64_t cur_col_val,
                                const int32_t ret_null_val,
                                const int64_t null_val) {
  return cur_col_val == null_val ? ret_null_val : cur_col_val * kSecsPerDay;
}

extern "C" DEVICE NEVER_INLINE int64_t
SUFFIX(fixed_width_date_encode_noinline)(const int64_t cur_col_val,
                                         const int32_t ret_null_val,
                                         const int64_t null_val) {
  return SUFFIX(fixed_width_date_encode)(cur_col_val, ret_null_val, null_val);
}

#undef SUFFIX

#endif  // QUERYENGINE_DECODERSIMPL_H
