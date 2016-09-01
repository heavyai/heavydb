#include "OutputBufferInitialization.h"
#include "BufferCompaction.h"
#include "ResultRows.h"

#include "../Analyzer/Analyzer.h"

namespace {

inline std::vector<int64_t> init_agg_val_vec(const std::vector<TargetInfo>& targets,
                                             size_t agg_col_count,
                                             const bool is_group_by,
                                             const size_t min_byte_width_to_compact) {
  std::vector<int64_t> agg_init_vals(agg_col_count, 0);
  for (size_t target_idx = 0, agg_col_idx = 0; target_idx < targets.size() && agg_col_idx < agg_col_count;
       ++target_idx, ++agg_col_idx) {
    const auto agg_info = targets[target_idx];
    if (!agg_info.is_agg) {
      continue;
    }
    agg_init_vals[agg_col_idx] =
        get_agg_initial_val(agg_info.agg_kind, get_compact_type(agg_info), is_group_by, min_byte_width_to_compact);
    if (kAVG == agg_info.agg_kind) {
      agg_init_vals[++agg_col_idx] = 0;
    }
  }
  return agg_init_vals;
}

void set_compact_type(TargetInfo& target, const SQLTypeInfo& new_type) {
  if (target.is_agg) {
    const auto agg_type = target.agg_kind;
    auto& agg_arg = target.agg_arg_type;
    if (agg_type != kCOUNT || agg_arg.get_type() != kNULLT) {
      agg_arg = new_type;
      return;
    }
  }
  target.sql_type = new_type;
}

}  // namespace

std::pair<int64_t, int64_t> inline_int_max_min(const size_t byte_width) {
  switch (byte_width) {
    case 1:
      return std::make_pair(std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::min());
    case 2:
      return std::make_pair(std::numeric_limits<int16_t>::max(), std::numeric_limits<int16_t>::min());
    case 4:
      return std::make_pair(std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min());
    case 8:
      return std::make_pair(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min());
    default:
      CHECK(false);
  }
}

std::pair<uint64_t, uint64_t> inline_uint_max_min(const size_t byte_width) {
  switch (byte_width) {
    case 1:
      return std::make_pair(std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::min());
    case 2:
      return std::make_pair(std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint16_t>::min());
    case 4:
      return std::make_pair(std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::min());
    case 8:
      return std::make_pair(std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::min());
    default:
      CHECK(false);
  }
}

// TODO(alex): proper types for aggregate
int64_t get_agg_initial_val(const SQLAgg agg,
                            const SQLTypeInfo& ti,
                            const bool enable_compaction,
                            const unsigned min_byte_width_to_compact) {
  CHECK(!ti.is_string());
  const auto byte_width = enable_compaction ? compact_byte_width(static_cast<unsigned>(get_bit_width(ti) >> 3),
                                                                 unsigned(min_byte_width_to_compact))
                                            : sizeof(int64_t);
  CHECK_GE(byte_width, static_cast<unsigned>(ti.get_logical_size()));
  switch (agg) {
    case kAVG:
    case kSUM: {
      if (!ti.get_notnull()) {
        if (ti.is_fp()) {
          switch (byte_width) {
            case 4: {
              const float null_float = inline_fp_null_val(ti);
              return *reinterpret_cast<const int32_t*>(&null_float);
            }
            case 8: {
              const double null_double = inline_fp_null_val(ti);
              return *reinterpret_cast<const int64_t*>(&null_double);
            }
            default:
              CHECK(false);
          }
        } else {
          return inline_int_null_val(ti);
        }
      }
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
    case kCOUNT:
      return 0;
    case kMIN: {
      switch (byte_width) {
        case 4: {
          const float max_float = std::numeric_limits<float>::max();
          const float null_float = ti.is_fp() ? static_cast<float>(inline_fp_null_val(ti)) : 0.;
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
          const float null_float = ti.is_fp() ? static_cast<float>(inline_fp_null_val(ti)) : 0.;
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

int64_t get_initial_val(const TargetInfo& target_info, const size_t min_byte_width_to_compact) {
  if (!target_info.is_agg) {
    return 0;
  }
  const auto chosen_type = get_compact_type(target_info);
  return get_agg_initial_val(target_info.agg_kind, chosen_type, !chosen_type.is_fp(), min_byte_width_to_compact);
}

std::vector<int64_t> init_agg_val_vec(const std::vector<Analyzer::Expr*>& targets,
                                      const std::list<std::shared_ptr<Analyzer::Expr>>& quals,
                                      size_t agg_col_count,
                                      const bool is_group_by,
                                      const size_t min_byte_width_to_compact) {
  std::vector<TargetInfo> target_infos;
  target_infos.reserve(targets.size());
  for (size_t target_idx = 0, agg_col_idx = 0; target_idx < targets.size() && agg_col_idx < agg_col_count;
       ++target_idx, ++agg_col_idx) {
    const auto target_expr = targets[target_idx];
    auto target = target_info(target_expr);
    auto arg_expr = agg_arg(target_expr);
    if (arg_expr && constrained_not_null(arg_expr, quals)) {
      target.skip_null_val = false;
      auto new_type = get_compact_type(target);
      new_type.set_notnull(true);
      set_compact_type(target, new_type);
    }
    target_infos.push_back(target);
  }
  return init_agg_val_vec(target_infos, agg_col_count, is_group_by, min_byte_width_to_compact);
}

const Analyzer::Expr* agg_arg(const Analyzer::Expr* expr) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(expr);
  return agg_expr ? agg_expr->get_arg() : nullptr;
}

bool constrained_not_null(const Analyzer::Expr* expr, const std::list<std::shared_ptr<Analyzer::Expr>>& quals) {
  for (const auto qual : quals) {
    auto uoper = std::dynamic_pointer_cast<Analyzer::UOper>(qual);
    if (!uoper) {
      continue;
    }
    bool is_negated{false};
    if (uoper->get_optype() == kNOT) {
      uoper = std::dynamic_pointer_cast<Analyzer::UOper>(uoper->get_own_operand());
      is_negated = true;
    }
    if (uoper && (uoper->get_optype() == kISNOTNULL || (is_negated && uoper->get_optype() == kISNULL))) {
      if (*uoper->get_own_operand() == *expr) {
        return true;
      }
    }
  }
  return false;
}
