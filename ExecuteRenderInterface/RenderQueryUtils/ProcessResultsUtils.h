/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "QueryEngine/Rendering/RenderInfo.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRenderer/Interop/Enums.h"
#include "QueryRenderer/QueryDataLayout.h"
#include "QueryRenderer/Utils/AnalyzerUtils.h"
#include "Shared/thread_count.h"

#define DISABLE_MULTI_THREADING 0
#define DEBUG_RENDER_POLYGONS 0

namespace QueryRenderer {

static constexpr int DefaultNumBytesPerColumnType = 8;
static constexpr size_t kMinRowCountWorthMultiThreading = 5000;
constexpr char kRowIdColumnName[] = "rowid";

/**
 * @brief RowIdStatus oversees the current status of rowid management from query results.
 * If the results of a query includes a "rowid" column, this struct will tag the column
 * index for the "rowid" column. If a "rowid" column does not exist, and the query is a
 * non-insitu query, then a "rowid" column may be automatically added. This struct
 * maintains the logic for when a rowid column can be automatically added.
 */
struct RowIdStatus {
  explicit RowIdStatus(const int in_rowid_idx, const RenderInfo& render_info);

  // the column index in the result set of a valid rowid column. Value is -1 if such a
  // column doesn't exist.
  int rowid_idx;

  // if true, a rowid column is capable of being automatically added. If true, rowid_idx
  // above should be -1, and should only be true for non-insitu results.
  bool add_rowid;
};

RowIdStatus get_rowid_status(const std::vector<TargetMetaInfo>& targets,
                             const RenderInfo& render_info);

struct RenderDataQueryResult {
  std::shared_ptr<QueryDataLayout> render_data_layout;
  std::vector<unsigned int> target_column_indices;
  std::vector<char> data;
  const size_t align_bytes;
};

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
    const size_t additional_col_count = 0);

void set_non_in_situ_render_data_entry(RenderDataQueryResult& render_data,
                                       const std::vector<TargetValue>& row,
                                       const std::vector<TargetMetaInfo>& targets,
                                       const size_t rowidx,
                                       const size_t resultrow_entry_idx,
                                       const RowIdStatus& rowid_status,
                                       const size_t align_bytes);

size_t executor_process_result_rows(
    const ResultSet& rows,
    std::function<void(std::vector<TargetValue>&&, const size_t, const size_t)> do_work,
    const bool force_singlethreaded = false);

template <typename ThreadState, typename... Args>
size_t executor_process_result_rows(
    const ResultSet& rows,
    std::function<
        void(ThreadState& state, std::vector<TargetValue>&&, const size_t, const size_t)>
        do_work,
    std::function<void(std::vector<ThreadState>&)> do_reduction,
    Args&... Fargs) {
  std::vector<ThreadState> thread_states;
  size_t entry_count = rows.entryCount();
  size_t row_count{0};
  if (!DISABLE_MULTI_THREADING && !rows.isTruncated() &&
      entry_count > kMinRowCountWorthMultiThreading) {
    const size_t worker_count = cpu_threads();
    std::vector<std::future<void>> threads;
    std::atomic<size_t> row_idx{0};

    std::vector<std::pair<size_t, size_t>> thread_ranges;

    // run an init step first
    for (size_t i = 0,
                start_entry = 0,
                stride = (entry_count + worker_count - 1) / worker_count;
         i < worker_count && start_entry < entry_count;
         ++i, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, entry_count);
      thread_ranges.emplace_back(start_entry, end_entry);
      thread_states.emplace_back(start_entry, end_entry, Fargs...);
    }

    // now launch the threads
    for (size_t i = 0; i < thread_ranges.size(); ++i) {
      threads.push_back(std::async(
          std::launch::async,
          [&, parent_thread_local_ids = logger::thread_local_ids()](
              const size_t start, const size_t end, const size_t thread_idx) {
            logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
            for (size_t i = start; i < end; ++i) {
              auto crt_row = rows.getRowAtNoTranslations(i);
              if (!crt_row.empty()) {
                do_work(thread_states[thread_idx],
                        std::move(crt_row),
                        row_idx.fetch_add(1),
                        i);
              }
            }
          },
          thread_ranges[i].first,
          thread_ranges[i].second,
          i));
    }
    for (auto& child : threads) {
      child.get();
    }
    row_count = row_idx.load();
  } else {
    thread_states.emplace_back(0, entry_count, Fargs...);
    auto& thread_state = thread_states.back();

    rows.moveToBegin();
    while (true) {
      auto crt_row = rows.getNextRow(false, false);
      if (crt_row.empty()) {
        break;
      }
      do_work(
          thread_state, std::move(crt_row), row_count, rows.getCurrentRowBufferIndex());
      row_count++;
    }
  }
  if (do_reduction) {
    do_reduction(thread_states);
  }
  return row_count;
}

}  // namespace QueryRenderer
