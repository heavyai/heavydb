/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/ProcessResultsUtils.h"

#include "GfxDriver/RenderLogger.h"

namespace QueryRenderer {

RowIdStatus::RowIdStatus(const int in_rowid_idx, const RenderInfo& render_info)
    : rowid_idx{in_rowid_idx}, add_rowid{false} {
  const bool is_rowid_in_query_results = rowid_idx >= 0;
  if (!is_rowid_in_query_results) {
    // NOTE: an insitu query could be forced non-insitu, but we still want to treat it
    // as insitu for hit-testing purposes. In such a case, rowid is not auto-injected
    // here. This is checked with coundRunInSitu() below
    const bool is_insitu_query = render_info.couldRunInSitu();
    if (!is_insitu_query) {
      auto const& render_query_options = render_info.getRenderQueryOptions();
      // legacy rowid injection means rowid is automatically added if an
      // `enableHitTesting` property is not defined in a vega data block
      // Need to also add rowid for PPLL poly rendering, even if hit-testing
      // is explicitly disabled.
      if (render_query_options.isHitTestingEnabled() ||
          render_query_options.injectRowIdForPPLL()) {
        add_rowid = true;
      }
    }
  }
}

RowIdStatus get_rowid_status(const std::vector<TargetMetaInfo>& targets,
                             const RenderInfo& render_info) {
  int rowid_idx = -1;
  for (size_t i = 0; i < targets.size(); ++i) {
    if (targets[i].get_resname() == kRowIdColumnName) {
      const auto& col_ti = targets[i].get_type_info();
      CHECK_EQ(kBIGINT, col_ti.get_type());
      rowid_idx = i;
      break;
    }
  }
  return RowIdStatus(rowid_idx, render_info);
}

RenderDataQueryResult get_render_data_template(
    const std::vector<TargetMetaInfo>& targets_meta,
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets,
    const std::vector<unsigned int>& target_column_indices,
    const std::unordered_set<unsigned int>& target_column_indices_to_ignore,
    const std::unordered_map<std::string, std::string>& target_aliases,
    const QueryDataLayout::LayoutType convert_to_layout_type,
    const size_t entry_count,
    const RowIdStatus& rowid_status,
    const bool uses_result_set,
    const bool allocate_local_row_data_buffer,
    const size_t additional_col_count) {
  RENDER_LOG_SCOPE();
  // the overall stride of the SSBO
  size_t align_bytes = 0;

  const bool add_rowid = rowid_status.add_rowid && uses_result_set;
  if (add_rowid) {
    align_bytes += DefaultNumBytesPerColumnType;
  }

  std::vector<QueryDataLayout::AttrAliasInfo> attr_info;

  std::function<std::pair<int, int>(const int)> getTableIdColIdFunc = [](const int idx) {
    return std::make_pair(-1, -1);
  };
  if (targets.size()) {
    getTableIdColIdFunc = [&targets_meta, &targets](const int idx) {
      const auto& te = targets[idx];
      const auto alias = te->get_resname();
      CHECK_EQ(alias, targets_meta[idx].get_resname());
      return get_table_id_col_id_from_target_expr(te->get_expr());
    };
  }

  std::vector<unsigned int> output_target_column_indices;

  for (auto i : target_column_indices) {
    // ignore this one?
    if (target_column_indices_to_ignore.find(i) !=
        target_column_indices_to_ignore.end()) {
      continue;
    }

    // name
    auto alias = targets_meta[i].get_resname();
    if (auto itr = target_aliases.find(alias); itr != target_aliases.end()) {
      alias = itr->second;
    }

    // type
    if (static_cast<int>(i) == rowid_status.rowid_idx && uses_result_set) {
      // Adding the row index of the results as our rowid in these cases, not the rowid of
      // the primary table
      attr_info.emplace_back(alias, SQLTypeInfo(kBIGINT, true), -1, -1);
    } else {
      int table_id, col_id;
      std::tie(table_id, col_id) = getTableIdColIdFunc(i);
      attr_info.emplace_back(alias, targets_meta[i].get_type_info(), table_id, col_id);
    }

    // keep this one
    output_target_column_indices.push_back(i);

    // add it to the stride
    // TODO(croot): support smaller types. This is straight forward
    // to do in a non in-situ data case, but in the in situ case, it appears
    // that all data is encoded into 8-byte chunks. Keeping this consistent
    // in both cases for now.
    align_bytes += DefaultNumBytesPerColumnType;
  }

  if (add_rowid) {
    attr_info.emplace_back(kRowIdColumnName, SQLTypeInfo(kBIGINT, true), -1, -1);
  }

  // For some rendering queries, we may want to pad the returned result with additional
  // values (e.g. duplicate vertices at the beginning and end of each line segment for
  // line strip rendering). additional_col_count allows us to pad the data buffer with
  // additional columns.
  if (additional_col_count > 0) {
    align_bytes += additional_col_count * DefaultNumBytesPerColumnType;
  }

  size_t num_data_bytes = entry_count * align_bytes;
  std::vector<char> raw_data;
  if (allocate_local_row_data_buffer) {
    raw_data.resize(num_data_bytes, 0);
  }

  auto layout =
      std::make_shared<QueryDataLayout>(std::move(attr_info), convert_to_layout_type);

  return {std::move(layout),
          std::move(output_target_column_indices),
          std::move(raw_data),
          align_bytes};
}

struct RenderDataEntryFunctor : public boost::static_visitor<char*> {
  RenderDataEntryFunctor(bool use_executor_rowidx,
                         const size_t& resultrow_entry_idx,
                         char* dataptr)
      : use_executor_rowidx_(use_executor_rowidx)
      , resultrow_entry_idx_(resultrow_entry_idx)
      , dataptr_(dataptr) {}

  char* operator()(const int64_t& i) const {
    if (use_executor_rowidx_) {
      std::memcpy(dataptr_, &i, sizeof(int64_t));
      return dataptr_ + sizeof(int64_t);
    } else {
      std::memcpy(
          dataptr_, &resultrow_entry_idx_, sizeof(decltype(resultrow_entry_idx_)));
      return dataptr_ + sizeof(decltype(resultrow_entry_idx_));
    }
  }

  char* operator()(const float& f) const {
    double d = static_cast<double>(f);
    std::memcpy(dataptr_, &d, sizeof(double));
    return dataptr_ + sizeof(double);
  }

  char* operator()(const double& d) const {
    std::memcpy(dataptr_, &d, sizeof(double));
    return dataptr_ + sizeof(double);
  }

  char* operator()(const NullableString& n) const {
    throw std::runtime_error(
        "Unable to copy a NullableString into a vertex or uniform/ssbo buffer. Possible "
        "column type mismatch.");
    return dataptr_;
  }

  bool use_executor_rowidx_;
  const size_t resultrow_entry_idx_;
  char* dataptr_;
};

void set_non_in_situ_render_data_entry(RenderDataQueryResult& render_data,
                                       const std::vector<TargetValue>& row,
                                       const std::vector<TargetMetaInfo>& targets,
                                       const size_t rowidx,
                                       const size_t resultrow_entry_idx,
                                       const RowIdStatus& rowid_status,
                                       const size_t align_bytes) {
  CHECK_EQ(row.size(), targets.size());
  auto startoffset = rowidx * align_bytes;
  auto offset = startoffset;

  auto dataptr = render_data.data.data() + offset;
  for (auto col_idx : render_data.target_column_indices) {
    const auto tv = row[col_idx];
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    if (!scalar_tv) {
      // array value, ignore, write zero to SSBO
      const int64_t zero = 0L;
      std::memcpy(dataptr, &zero, sizeof(int64_t));
      dataptr += sizeof(int64_t);
      continue;
    }
    // TODO(croot): get attr offset per row from the layout
    // and use here
    bool use_resultrow_rowidx = static_cast<int>(col_idx) == rowid_status.rowid_idx;
    const auto renderDataEntryFunctor =
        RenderDataEntryFunctor(!use_resultrow_rowidx, resultrow_entry_idx, dataptr);
    dataptr = boost::apply_visitor(renderDataEntryFunctor, *scalar_tv);
  }

  if (rowid_status.add_rowid) {
    std::memcpy(dataptr, &resultrow_entry_idx, sizeof(decltype(resultrow_entry_idx)));
    dataptr += sizeof(decltype(resultrow_entry_idx));
  }

  CHECK(offset - startoffset <= align_bytes);
}

size_t executor_process_result_rows(
    const ResultSet& rows,
    std::function<void(std::vector<TargetValue>&&, const size_t, const size_t)> do_work,
    const bool force_singlethreaded) {
  size_t entry_count = rows.entryCount();
  if (!DISABLE_MULTI_THREADING && !force_singlethreaded && !rows.isTruncated() &&
      entry_count > kMinRowCountWorthMultiThreading) {
    const size_t worker_count = cpu_threads();
    std::vector<std::future<void>> threads;
    std::atomic<size_t> row_idx{0};
    for (size_t i = 0,
                start_entry = 0,
                stride = (entry_count + worker_count - 1) / worker_count;
         i < worker_count && start_entry < entry_count;
         ++i, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, entry_count);
      threads.push_back(std::async(
          std::launch::async,
          [&rows,
           &do_work,
           &row_idx,
           parent_thread_local_ids = logger::thread_local_ids()](const size_t start,
                                                                 const size_t end) {
            logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
            for (size_t i = start; i < end; ++i) {
              auto crt_row = rows.getRowAtNoTranslations(i);
              if (!crt_row.empty()) {
                do_work(std::move(crt_row), row_idx.fetch_add(1), i);
              }
            }
          },
          start_entry,
          end_entry));
    }
    for (auto& child : threads) {
      child.get();
    }
    return row_idx.load();
  } else {
    size_t row_idx = 0;
    rows.moveToBegin();
    while (true) {
      auto crt_row = rows.getNextRow(false, false);
      if (crt_row.empty()) {
        break;
      }
      do_work(std::move(crt_row), row_idx, rows.getCurrentRowBufferIndex());
      row_idx++;
    }
    return row_idx;
  }
}

}  // namespace QueryRenderer
