#ifndef QUERYENGINE_AGGREGATEUTILS_H
#define QUERYENGINE_AGGREGATEUTILS_H

#include "BufferCompaction.h"

inline int64_t inline_int_null_val(const SQLTypeInfo& ti) {
  auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
  if (ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    CHECK_EQ(4, ti.get_size());
    type = kINT;
  }
  switch (type) {
    case kBOOLEAN:
      return std::numeric_limits<int8_t>::min();
    case kSMALLINT:
      return std::numeric_limits<int16_t>::min();
    case kINT:
      return std::numeric_limits<int32_t>::min();
    case kBIGINT:
      return std::numeric_limits<int64_t>::min();
    case kTIMESTAMP:
    case kTIME:
    case kDATE:
      return std::numeric_limits<int64_t>::min();
    default:
      CHECK(false);
  }
}

inline double inline_fp_null_val(const SQLTypeInfo& ti) {
  CHECK(ti.is_fp());
  const auto type = ti.get_type();
  switch (type) {
    case kFLOAT:
      return NULL_FLOAT;
    case kDOUBLE:
      return NULL_DOUBLE;
    default:
      CHECK(false);
  }
}

inline size_t get_bit_width(const SQLTypeInfo& ti) {
  const auto int_type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
  switch (int_type) {
    case kBOOLEAN:
      return 8;
    case kSMALLINT:
      return 16;
    case kINT:
      return 32;
    case kBIGINT:
      return 64;
    case kFLOAT:
      return 32;
    case kDOUBLE:
      return 64;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return sizeof(time_t) * 8;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return 32;
    case kARRAY:
      throw std::runtime_error("Projecting on array columns not supported yet.");
    default:
      CHECK(false);
  }
}

// TODO(alex): proper types for aggregate
inline int64_t get_agg_initial_val(const SQLAgg agg, const SQLTypeInfo& ti, const bool is_group_by) {
  CHECK(!ti.is_string());
  const auto byte_width = is_group_by ? compact_byte_width(get_bit_width(ti) >> 3) : sizeof(int64_t);
  CHECK_GE(byte_width, static_cast<unsigned>(ti.get_size()));
  switch (agg) {
    case kAVG:
    case kSUM:
    case kCOUNT: {
      switch (byte_width) {
        case 4: {
          const float zero_float{0.};
          return ti.is_fp() ? *reinterpret_cast<const int32_t*>(&zero_float) : 0;
        }
        case 8: {
          const double zero_double{0.};
          return ti.is_fp() ? *reinterpret_cast<const int64_t*>(&zero_double) : 0;
        }
        default:
          CHECK(false);
      }
    }
    case kMIN: {
      switch (byte_width) {
        case 4: {
          const float max_float = std::numeric_limits<float>::max();
          const float null_float = ti.is_fp() ? inline_fp_null_val(ti) : 0.;
          return ti.is_fp() ? (ti.get_notnull() ? *reinterpret_cast<const int32_t*>(&max_float)
                                                : *reinterpret_cast<const int32_t*>(&null_float))
                            : (ti.get_notnull() ? std::numeric_limits<int32_t>::max() : inline_int_null_val(ti));
        }
        case 8: {
          const double max_double = std::numeric_limits<double>::max();
          const double null_double{ti.is_fp() ? inline_fp_null_val(ti) : 0.};
          return ti.is_fp() ? (ti.get_notnull() ? *reinterpret_cast<const int64_t*>(&max_double)
                                                : *reinterpret_cast<const int64_t*>(&null_double))
                            : (ti.get_notnull() ? std::numeric_limits<int64_t>::max() : inline_int_null_val(ti));
        }
        default:
          CHECK(false);
      }
    }
    case kMAX: {
      switch (byte_width) {
        case 4: {
          const float min_float = std::numeric_limits<float>::min();
          const float null_float = ti.is_fp() ? inline_fp_null_val(ti) : 0.;
          return (ti.is_fp()) ? (ti.get_notnull() ? *reinterpret_cast<const int32_t*>(&min_float)
                                                  : *reinterpret_cast<const int32_t*>(&null_float))
                              : (ti.get_notnull() ? std::numeric_limits<int32_t>::min() : inline_int_null_val(ti));
        }
        case 8: {
          const double min_double = std::numeric_limits<double>::min();
          const double null_double{ti.is_fp() ? inline_fp_null_val(ti) : 0.};
          return ti.is_fp() ? (ti.get_notnull() ? *reinterpret_cast<const int64_t*>(&min_double)
                                                : *reinterpret_cast<const int64_t*>(&null_double))
                            : (ti.get_notnull() ? std::numeric_limits<int64_t>::min() : inline_int_null_val(ti));
        }
        default:
          CHECK(false);
      }
    }
    default:
      CHECK(false);
  }
}

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
