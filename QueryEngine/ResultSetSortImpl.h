#ifndef QUERYENGINE_RESULTSETSORTIMPL_H
#define QUERYENGINE_RESULTSETSORTIMPL_H

#include "CompilationOptions.h"
#include "../Shared/TargetInfo.h"

struct PodOrderEntry {
  int tle_no;       /* targetlist entry number: 1-based */
  bool is_desc;     /* true if order is DESC */
  bool nulls_first; /* true if nulls are ordered first.  otherwise last. */
};

struct GroupByBufferLayoutInfo {
  const size_t entry_count;
  const size_t col_off;
  const size_t col_bytes;
  const size_t row_bytes;
  const TargetInfo oe_target_info;
  const ssize_t target_groupby_index;
};

namespace Data_Namespace {

class DataMgr;

}  // Data_Namespace

template <class K>
std::vector<uint32_t> baseline_sort(const ExecutorDeviceType device_type,
                                    const int device_id,
                                    Data_Namespace::DataMgr* data_mgr,
                                    const int8_t* groupby_buffer,
                                    const PodOrderEntry& oe,
                                    const GroupByBufferLayoutInfo& layout,
                                    const size_t top_n,
                                    const size_t start,
                                    const size_t step);

#endif  // QUERYENGINE_RESULTSETSORTIMPL_H
