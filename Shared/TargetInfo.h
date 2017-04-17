/**
 * @file    TargetInfo.h
 * @author  Alex Suhan <alex@mapd.com>
 */

#ifndef QUERYENGINE_TARGETINFO_H
#define QUERYENGINE_TARGETINFO_H

#include "sqldefs.h"
#include "sqltypes.h"

struct TargetInfo {
  bool is_agg;
  SQLAgg agg_kind;
  SQLTypeInfo sql_type;
  SQLTypeInfo agg_arg_type;
  bool skip_null_val;
  bool is_distinct;
};

inline bool is_distinct_target(const TargetInfo& target_info) {
  return target_info.is_distinct || target_info.agg_kind == kAPPROX_COUNT_DISTINCT;
}

inline bool takes_float_argument(const TargetInfo& target_info) {
  return target_info.is_agg && (target_info.agg_kind == kAVG || target_info.agg_kind == kSUM ||
                                target_info.agg_kind == kMIN || target_info.agg_kind == kMAX) &&
         target_info.agg_arg_type.get_type() == kFLOAT;
}

#endif  // QUERYENGINE_TARGETINFO_H
