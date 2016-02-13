#ifndef QUERYENGINE_AGGREGATEUTILS_H
#define QUERYENGINE_AGGREGATEUTILS_H

// TODO(alex): proper types for aggregate
inline int64_t init_agg_val(const SQLAgg agg, const SQLTypeInfo& ti) {
  switch (agg) {
    case kAVG:
    case kSUM:
    case kCOUNT: {
      const double zero_double{0.};
      return ti.is_fp() ? *reinterpret_cast<const int64_t*>(&zero_double) : 0;
    }
    case kMIN: {
      const double max_double{std::numeric_limits<double>::max()};
      const double null_double{ti.is_fp() ? inline_fp_null_val(ti) : 0.};
      return ti.is_fp() ? (ti.get_notnull() ? *reinterpret_cast<const int64_t*>(&max_double)
                                            : *reinterpret_cast<const int64_t*>(&null_double))
                        : (ti.get_notnull() ? std::numeric_limits<int64_t>::max() : inline_int_null_val(ti));
    }
    case kMAX: {
      const auto min_double = std::numeric_limits<double>::min();
      const double null_double{ti.is_fp() ? inline_fp_null_val(ti) : 0.};
      return (ti.is_fp()) ? (ti.get_notnull() ? *reinterpret_cast<const int64_t*>(&min_double)
                                              : *reinterpret_cast<const int64_t*>(&null_double))
                          : (ti.get_notnull() ? std::numeric_limits<int64_t>::min() : inline_int_null_val(ti));
    }
    default:
      CHECK(false);
  }
}

#endif  // QUERYENGINE_AGGREGATEUTILS_H
