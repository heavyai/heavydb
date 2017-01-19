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

#endif  // QUERYENGINE_TARGETINFO_H
