/**
 * @file    SqlTypesLayout.h
 * @author  Alex Suhan <alex@mapd.com>
 */

#ifndef QUERYENGINE_SQLTYPESLAYOUT_H
#define QUERYENGINE_SQLTYPESLAYOUT_H

#include "../Shared/TargetInfo.h"

#include <glog/logging.h>

#include <limits>

class OverflowOrUnderflow : public std::runtime_error {
 public:
  OverflowOrUnderflow() : std::runtime_error("Overflow or underflow") {}
};

inline const SQLTypeInfo& get_compact_type(const TargetInfo& target) {
  if (!target.is_agg) {
    return target.sql_type;
  }
  const auto agg_type = target.agg_kind;
  const auto& agg_arg = target.agg_arg_type;
  if (agg_arg.get_type() == kNULLT) {
    CHECK_EQ(kCOUNT, agg_type);
    CHECK(!target.is_distinct);
    return target.sql_type;
  }

  return agg_type != kCOUNT ? agg_arg : target.sql_type;
}

template <typename T>
inline bool detect_overflow_and_underflow(const T a,
                                          const T b,
                                          const bool nullable,
                                          const T null_val,
                                          const SQLTypeInfo& ti) {
#ifdef ENABLE_COMPACTION
  if (!ti.is_integer()) {
    return false;
  }
  if (nullable) {
    if (a == null_val || b == null_val) {
      return false;
    }
  }
  const auto max_intx = std::numeric_limits<T>::max();
  const auto min_intx = std::numeric_limits<T>::min();
  if ((b > 0 && a > (max_intx - b)) || (b < 0 && a < (min_intx - b))) {
    return true;
  }
#endif
  return false;
}

inline int64_t inline_int_null_val(const SQLTypeInfo& ti) {
  auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
  if (ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    CHECK_EQ(4, ti.get_logical_size());
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
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return std::numeric_limits<int64_t>::min();
    default:
      CHECK(false);
  }
}

inline int64_t inline_fixed_encoding_null_val(const SQLTypeInfo& ti) {
  if (ti.get_compression() == kENCODING_NONE) {
    return inline_int_null_val(ti);
  }
  if (ti.get_compression() == kENCODING_DICT) {
    CHECK(ti.is_string());
    return -(1L << (8 * ti.get_size() - 1));
  }
  CHECK_EQ(kENCODING_FIXED, ti.get_compression());
  CHECK(ti.is_integer() || ti.is_time() || ti.is_decimal());
  CHECK_EQ(0, ti.get_comp_param() % 8);
  return -(1L << (ti.get_comp_param() - 1));
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

inline uint64_t exp_to_scale(const unsigned exp) {
  uint64_t res = 1;
  for (unsigned i = 0; i < exp; ++i) {
    res *= 10;
  }
  return res;
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
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
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

#endif  // QUERYENGINE_SQLTYPESLAYOUT_H
