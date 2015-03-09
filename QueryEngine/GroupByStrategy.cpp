#include "GroupByStrategy.h"

#include <glog/logging.h>


namespace {

struct OneColKnownRangeInfo {
  GroupByStrategy::ColRangeInfo hash_type_;
  int64_t min;
  int64_t max;
};

int64_t extract_from_datum(const Datum datum, const SQLTypeInfo& ti) {
  switch (ti.get_type()) {
  case kSMALLINT:
    return datum.smallintval;
  case kINT:
  case kCHAR:
  case kVARCHAR:
  case kTEXT:
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    return datum.intval;
  case kBIGINT:
    return datum.bigintval;
  case kTIME:
  case kTIMESTAMP:
  case kDATE:
    return datum.timeval;
  default:
    CHECK(false);
  }
}

int64_t extract_min_stat(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_from_datum(stats.min, ti);
}

int64_t extract_max_stat(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_from_datum(stats.max, ti);
}

#define FIND_STAT_FRAG(stat_name)                                                             \
  const auto stat_name##_frag = std::stat_name##_element(fragments.begin(), fragments.end(),  \
    [group_col_id, group_by_ti](const Fragmenter_Namespace::FragmentInfo& lhs,                \
                                 const Fragmenter_Namespace::FragmentInfo& rhs) {             \
      auto lhs_meta_it = lhs.chunkMetadataMap.find(group_col_id);                             \
      CHECK(lhs_meta_it != lhs.chunkMetadataMap.end());                                       \
      auto rhs_meta_it = rhs.chunkMetadataMap.find(group_col_id);                             \
      CHECK(rhs_meta_it != rhs.chunkMetadataMap.end());                                       \
      return extract_##stat_name##_stat(lhs_meta_it->second.chunkStats, group_by_ti) <        \
             extract_##stat_name##_stat(rhs_meta_it->second.chunkStats, group_by_ti);         \
  });                                                                                         \
  if (stat_name##_frag == fragments.end()) {                                                  \
    return { GroupByStrategy::ColRangeInfo::OneColGuessedRange, 0, guessed_range_max };                                                                 \
  }

OneColKnownRangeInfo getColumnRangeInfo(
    const Planner::AggPlan* agg_plan,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  const int64_t guessed_range_max { 255 };  // TODO(alex): replace with educated guess
  if (!agg_plan) {
    return { GroupByStrategy::ColRangeInfo::Scan, 0, 0 };
  }
  const auto& groupby_exprs = agg_plan->get_groupby_list();
  if (groupby_exprs.size() != 1) {
    return { GroupByStrategy::ColRangeInfo::MultiCol, 0, 0 };
  }
  const auto group_col_expr = dynamic_cast<Analyzer::ColumnVar*>(groupby_exprs.front());
  if (!group_col_expr) {
    return { GroupByStrategy::ColRangeInfo::OneColGuessedRange, 0, guessed_range_max };
  }
  const int group_col_id = group_col_expr->get_column_id();
  const auto group_by_ti = group_col_expr->get_type_info();
  switch (group_by_ti.get_type()) {
  case kTEXT:
  case kCHAR:
  case kVARCHAR:
    CHECK(group_by_ti.get_compression() != kENCODING_DICT);
  case kSMALLINT:
  case kINT:
  case kBIGINT: {
    FIND_STAT_FRAG(min);
    FIND_STAT_FRAG(max);
    const auto min_it = min_frag->chunkMetadataMap.find(group_col_id);
    CHECK(min_it != min_frag->chunkMetadataMap.end());
    const auto max_it = max_frag->chunkMetadataMap.find(group_col_id);
    CHECK(max_it != max_frag->chunkMetadataMap.end());
    const auto min_val = extract_min_stat(min_it->second.chunkStats, group_by_ti);
    const auto max_val = extract_max_stat(max_it->second.chunkStats, group_by_ti);
    CHECK_GE(max_val, min_val);
    return {
      group_by_ti.is_string()
        ? GroupByStrategy::ColRangeInfo::OneColConsecutiveKeys
        : GroupByStrategy::ColRangeInfo::OneColKnownRange,
      min_val,
      max_val
    };
  }
  default:
    return { GroupByStrategy::ColRangeInfo::Unknown, 0, 0 };
  }
}

#undef FIND_STAT_FRAG

}  // namespace

GroupByStrategy::GroupByStrategy(
    const Planner::AggPlan* agg_plan,
    const Fragmenter_Namespace::QueryInfo& query_info)
  : agg_plan_(agg_plan)
  , query_info_(query_info) {
}
