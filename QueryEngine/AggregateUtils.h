#ifndef QUERYENGINE_AGGREGATEUTILS_H
#define QUERYENGINE_AGGREGATEUTILS_H

#include "BufferCompaction.h"

inline int64_t get_component(const int8_t* group_by_buffer, const size_t comp_sz, const size_t index = 0) {
  int64_t ret = std::numeric_limits<int64_t>::min();
  switch (comp_sz) {
    case 1: {
      ret = group_by_buffer[index];
      break;
    }
    case 2: {
      const int16_t* buffer_ptr = reinterpret_cast<const int16_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    case 4: {
      const int32_t* buffer_ptr = reinterpret_cast<const int32_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    case 8: {
      const int64_t* buffer_ptr = reinterpret_cast<const int64_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    default:
      CHECK(false);
  }
  return ret;
}

inline void set_component(int8_t* group_by_buffer, const size_t comp_sz, const int64_t val, const size_t index = 0) {
  switch (comp_sz) {
    case 1: {
      group_by_buffer[index] = static_cast<int8_t>(val);
      break;
    }
    case 2: {
      int16_t* buffer_ptr = reinterpret_cast<int16_t*>(group_by_buffer);
      buffer_ptr[index] = (int16_t)val;
      break;
    }
    case 4: {
      int32_t* buffer_ptr = reinterpret_cast<int32_t*>(group_by_buffer);
      buffer_ptr[index] = (int32_t)val;
      break;
    }
    case 8: {
      int64_t* buffer_ptr = reinterpret_cast<int64_t*>(group_by_buffer);
      buffer_ptr[index] = val;
      break;
    }
    default:
      CHECK(false);
  }
}

inline int64_t float_to_double_bin(int32_t val, bool nullable = false) {
  float null_float = NULL_FLOAT;
  if (nullable && val == *reinterpret_cast<int32_t*>(&null_float)) {
    double null_double = NULL_DOUBLE;
    return *reinterpret_cast<int64_t*>(&null_double);
  }
  double res = *reinterpret_cast<float*>(&val);
  return *reinterpret_cast<int64_t*>(&res);
}

inline std::vector<int64_t> compact_init_vals(const size_t cmpt_size,
                                              const std::vector<int64_t>& init_vec,
                                              const std::vector<int8_t>& col_widths) {
  CHECK_GE(init_vec.size(), col_widths.size());
  std::vector<int64_t> cmpt_res(cmpt_size, 0);
  int8_t* buffer_ptr = reinterpret_cast<int8_t*>(&cmpt_res[0]);
  for (size_t col_idx = 0, col_count = col_widths.size(); col_idx < col_count; ++col_idx) {
    const auto chosen_bytes = compact_byte_width(col_widths[col_idx]);
    if (chosen_bytes == sizeof(int64_t)) {
      buffer_ptr = align_to_int64(buffer_ptr);
    }
    set_component(buffer_ptr, chosen_bytes, init_vec[col_idx]);
    buffer_ptr += chosen_bytes;
  }
  return cmpt_res;
}
#endif  // QUERYENGINE_AGGREGATEUTILS_H
