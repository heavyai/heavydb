/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef QUERYENGINE_AGGREGATEUTILS_H
#define QUERYENGINE_AGGREGATEUTILS_H

#include "BufferCompaction.h"
#include "TypePunning.h"
#include "../Shared/sqltypes.h"

#include <glog/logging.h>

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
    case 0: {
      break;
    }
    default:
      CHECK(false);
  }
}

inline int64_t float_to_double_bin(int32_t val, bool nullable = false) {
  const float null_float = NULL_FLOAT;
  if (nullable && val == *reinterpret_cast<const int32_t*>(may_alias_ptr(&null_float))) {
    double null_double = NULL_DOUBLE;
    return *reinterpret_cast<const int64_t*>(may_alias_ptr(&null_double));
  }
  double res = *reinterpret_cast<const float*>(may_alias_ptr(&val));
  return *reinterpret_cast<const int64_t*>(may_alias_ptr(&res));
}

inline std::vector<int64_t> compact_init_vals(const size_t cmpt_size,
                                              const std::vector<int64_t>& init_vec,
                                              const std::vector<ColWidths>& col_widths) {
  std::vector<int64_t> cmpt_res(cmpt_size, 0);
  int8_t* buffer_ptr = reinterpret_cast<int8_t*>(&cmpt_res[0]);
  for (size_t col_idx = 0, init_vec_idx = 0, col_count = col_widths.size(); col_idx < col_count; ++col_idx) {
    const auto chosen_bytes = static_cast<unsigned>(col_widths[col_idx].compact);
    if (chosen_bytes == 0) {
      continue;
    }
    if (chosen_bytes == sizeof(int64_t)) {
      buffer_ptr = align_to_int64(buffer_ptr);
    }
    CHECK_LT(init_vec_idx, init_vec.size());
    set_component(buffer_ptr, chosen_bytes, init_vec[init_vec_idx++]);
    buffer_ptr += chosen_bytes;
  }
  return cmpt_res;
}
#endif  // QUERYENGINE_AGGREGATEUTILS_H
