#include "ResultSetSortImpl.h"
#include "ResultSetBufferAccessors.h"
#include "GpuRtConstants.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

namespace {

template <class V, class I>
std::vector<uint32_t> do_radix_sort(const int8_t* groupby_buffer,
                                    V& dev_oe_col_buffer,
                                    I& dev_idx_buff,
                                    const PodOrderEntry& oe,
                                    const GroupByBufferLayoutInfo& layout,
                                    const size_t top_n) {
  if (oe.is_desc) {
    thrust::sort_by_key(
        dev_oe_col_buffer.begin(), dev_oe_col_buffer.end(), dev_idx_buff.begin(), thrust::greater<int64_t>());
  } else {
    thrust::sort_by_key(dev_oe_col_buffer.begin(), dev_oe_col_buffer.end(), dev_idx_buff.begin());
  }
  // Speculatively transfer only the top_n first, most of the time it'll be enough.
  thrust::host_vector<uint32_t> host_vector_result(dev_idx_buff.begin(),
                                                   dev_idx_buff.begin() + std::min(top_n, dev_idx_buff.size()));
  // Sometimes, radix sort can bring to the front entries which are empty.
  // For example, ascending sort on COUNT(*) will bring non-existent groups
  // to the front of dev_idx_buff since they're 0 in our system. Re-do the
  // transfer in that case to bring the entire dev_idx_buff; existing logic
  // in row iteration will take care of skipping the empty rows.
  for (size_t i = 0; i < host_vector_result.size(); ++i) {
    const auto entry_idx = host_vector_result[i];
    const auto key_ptr = groupby_buffer + entry_idx * layout.row_bytes;
    if (*reinterpret_cast<const int64_t*>(key_ptr) == EMPTY_KEY_64) {
      host_vector_result =
          thrust::host_vector<uint32_t>(dev_idx_buff.begin(), dev_idx_buff.begin() + dev_idx_buff.size());
      break;
    }
  }
  return std::vector<uint32_t>(host_vector_result.begin(), host_vector_result.end());
}

void add_nulls(std::vector<uint32_t>& idx_buff, const std::vector<uint32_t>& null_idx_buff, const PodOrderEntry& oe) {
  if (null_idx_buff.empty()) {
    return;
  }
  const auto insertion_point = oe.nulls_first ? idx_buff.begin() : idx_buff.end();
  idx_buff.insert(insertion_point, null_idx_buff.begin(), null_idx_buff.end());
}

std::vector<uint32_t> baseline_sort_fp(const ExecutorDeviceType device_type,
                                       const int8_t* groupby_buffer,
                                       const thrust::host_vector<int64_t>& oe_col_buffer,
                                       const PodOrderEntry& oe,
                                       const GroupByBufferLayoutInfo& layout,
                                       const size_t top_n,
                                       const size_t start,
                                       const size_t step) {
  thrust::host_vector<uint32_t> neg_idx_buff;
  thrust::host_vector<uint32_t> pos_idx_buff;
  std::vector<uint32_t> null_idx_buff;
  thrust::host_vector<int64_t> neg_oe_col_buffer;
  thrust::host_vector<int64_t> pos_oe_col_buffer;
  const auto slice_entry_count = layout.entry_count / step + (layout.entry_count % step ? 1 : 0);
  neg_idx_buff.reserve(slice_entry_count);
  pos_idx_buff.reserve(slice_entry_count);
  null_idx_buff.reserve(slice_entry_count);
  neg_oe_col_buffer.reserve(slice_entry_count);
  pos_oe_col_buffer.reserve(slice_entry_count);
  size_t oe_col_buffer_idx = 0;
  const auto col_ti =
      layout.oe_target_info.agg_kind == kAVG ? SQLTypeInfo(kDOUBLE, false) : layout.oe_target_info.sql_type;
  for (size_t i = start; i < layout.entry_count; i += step, ++oe_col_buffer_idx) {
    if (oe_col_buffer[oe_col_buffer_idx] == null_val_bit_pattern(col_ti)) {
      null_idx_buff.push_back(i);
      continue;
    }
    if (oe_col_buffer[oe_col_buffer_idx] < 0) {  // sign bit works the same for integer and floating point
      neg_idx_buff.push_back(i);
      neg_oe_col_buffer.push_back(oe_col_buffer[oe_col_buffer_idx]);
    } else {
      pos_idx_buff.push_back(i);
      pos_oe_col_buffer.push_back(oe_col_buffer[oe_col_buffer_idx]);
    }
  }
  std::vector<uint32_t> pos_result;
  if (device_type == ExecutorDeviceType::GPU) {
    thrust::device_vector<uint32_t> dev_pos_idx_buff = pos_idx_buff;
    thrust::device_vector<int64_t> dev_pos_oe_col_buffer = pos_oe_col_buffer;
    pos_result = do_radix_sort(groupby_buffer, dev_pos_oe_col_buffer, dev_pos_idx_buff, oe, layout, top_n);
  } else {
    CHECK(device_type == ExecutorDeviceType::CPU);
    pos_result = do_radix_sort(groupby_buffer, pos_oe_col_buffer, pos_idx_buff, oe, layout, top_n);
  }
  std::vector<uint32_t> neg_result;
  PodOrderEntry reverse_oe{oe.tle_no, !oe.is_desc, oe.nulls_first};
  if (device_type == ExecutorDeviceType::GPU) {
    thrust::device_vector<uint32_t> dev_neg_idx_buff = neg_idx_buff;
    thrust::device_vector<int64_t> dev_neg_oe_col_buffer = neg_oe_col_buffer;
    neg_result = do_radix_sort(groupby_buffer, dev_neg_oe_col_buffer, dev_neg_idx_buff, reverse_oe, layout, top_n);
  } else {
    CHECK(device_type == ExecutorDeviceType::CPU);
    neg_result = do_radix_sort(groupby_buffer, neg_oe_col_buffer, neg_idx_buff, reverse_oe, layout, top_n);
  }
  if (oe.is_desc) {
    pos_result.insert(pos_result.end(), neg_result.begin(), neg_result.end());
    add_nulls(pos_result, null_idx_buff, oe);
    return pos_result;
  }
  neg_result.insert(neg_result.end(), pos_result.begin(), pos_result.end());
  add_nulls(neg_result, null_idx_buff, oe);
  return neg_result;
}

std::vector<uint32_t> baseline_sort_int(const ExecutorDeviceType device_type,
                                        const int8_t* groupby_buffer,
                                        const thrust::host_vector<int64_t>& oe_col_buffer,
                                        const PodOrderEntry& oe,
                                        const GroupByBufferLayoutInfo& layout,
                                        const size_t top_n,
                                        const size_t start,
                                        const size_t step) {
  const auto& entry_ti = get_compact_type(layout.oe_target_info);
  std::vector<uint32_t> null_idx_buff;
  thrust::host_vector<uint32_t> notnull_idx_buff;
  const auto slice_entry_count = layout.entry_count / step + (layout.entry_count % step ? 1 : 0);
  null_idx_buff.reserve(slice_entry_count);
  notnull_idx_buff.reserve(slice_entry_count);
  thrust::host_vector<int64_t> notnull_oe_col_buffer;
  notnull_oe_col_buffer.reserve(slice_entry_count);
  size_t oe_col_buffer_idx = 0;
  for (size_t i = start; i < layout.entry_count; i += step, ++oe_col_buffer_idx) {
    if (oe_col_buffer[oe_col_buffer_idx] == null_val_bit_pattern(entry_ti)) {
      null_idx_buff.push_back(i);
    } else {
      notnull_idx_buff.push_back(i);
      notnull_oe_col_buffer.push_back(oe_col_buffer[oe_col_buffer_idx]);
    }
  }
  std::vector<uint32_t> notnull_result;
  if (device_type == ExecutorDeviceType::GPU) {
    thrust::device_vector<uint32_t> dev_notnull_idx_buff = notnull_idx_buff;
    thrust::device_vector<int64_t> dev_notnull_oe_col_buffer = notnull_oe_col_buffer;
    notnull_result = do_radix_sort(groupby_buffer, dev_notnull_oe_col_buffer, dev_notnull_idx_buff, oe, layout, top_n);
  } else {
    CHECK(device_type == ExecutorDeviceType::CPU);
    notnull_result = do_radix_sort(groupby_buffer, notnull_oe_col_buffer, notnull_idx_buff, oe, layout, top_n);
  }
  add_nulls(notnull_result, null_idx_buff, oe);
  return notnull_result;
}

thrust::host_vector<int64_t> collect_order_entry_column(const int8_t* groupby_buffer,
                                                        const GroupByBufferLayoutInfo& layout,
                                                        const size_t start,
                                                        const size_t step) {
  thrust::host_vector<int64_t> oe_col_buffer;
  const int8_t* crt_group_ptr1 = groupby_buffer + start * layout.row_bytes + layout.col_off;
  const int8_t* crt_group_ptr2{nullptr};
  if (layout.oe_target_info.agg_kind == kAVG) {
    crt_group_ptr2 = crt_group_ptr1 + layout.col_bytes;
  }
  const auto& entry_ti = get_compact_type(layout.oe_target_info);
  const auto step_bytes = layout.row_bytes * step;
  for (size_t i = start; i < layout.entry_count; i += step) {
    auto val1 = read_int_from_buff(crt_group_ptr1, layout.col_bytes);
    if (crt_group_ptr2) {
      const auto val2 = read_int_from_buff(crt_group_ptr2, 8);
      const auto avg_val = pair_to_double({val1, val2}, entry_ti);
      val1 = *reinterpret_cast<const int64_t*>(&avg_val);
    }
    oe_col_buffer.push_back(val1);
    crt_group_ptr1 += step_bytes;
    if (crt_group_ptr2) {
      crt_group_ptr2 += step_bytes;
    }
  }
  return oe_col_buffer;
}

}  // namespace

std::vector<uint32_t> baseline_sort(const ExecutorDeviceType device_type,
                                    const int device_id,
                                    const int8_t* groupby_buffer,
                                    const PodOrderEntry& oe,
                                    const GroupByBufferLayoutInfo& layout,
                                    const size_t top_n,
                                    const size_t start,
                                    const size_t step) {
  auto oe_col_buffer = collect_order_entry_column(groupby_buffer, layout, start, step);
  const auto& entry_ti = get_compact_type(layout.oe_target_info);
  CHECK(entry_ti.is_number());
  if (entry_ti.is_fp() || layout.oe_target_info.agg_kind == kAVG) {
    return baseline_sort_fp(device_type, groupby_buffer, oe_col_buffer, oe, layout, top_n, start, step);
  }
  // Because of how we represent nulls for integral types, they'd be at the
  // wrong position in these two cases. Separate them into a different buffer.
  if ((oe.is_desc && oe.nulls_first) || (!oe.is_desc && !oe.nulls_first)) {
    return baseline_sort_int(device_type, groupby_buffer, oe_col_buffer, oe, layout, top_n, start, step);
  }
  // Fastest path, no need to separate nulls away since they'll end up at the
  // right place as a side effect of how we're representing nulls.
  if (device_type == ExecutorDeviceType::GPU) {
    thrust::device_vector<uint32_t> dev_idx_buff(oe_col_buffer.size());
    thrust::sequence(dev_idx_buff.begin(), dev_idx_buff.end(), start, step);
    thrust::device_vector<int64_t> dev_oe_col_buffer = oe_col_buffer;
    return do_radix_sort(groupby_buffer, dev_oe_col_buffer, dev_idx_buff, oe, layout, top_n);
  }
  CHECK(device_type == ExecutorDeviceType::CPU);
  thrust::host_vector<uint32_t> host_idx_buff(oe_col_buffer.size());
  thrust::sequence(host_idx_buff.begin(), host_idx_buff.end(), start, step);
  return do_radix_sort(groupby_buffer, oe_col_buffer, host_idx_buff, oe, layout, top_n);
}
