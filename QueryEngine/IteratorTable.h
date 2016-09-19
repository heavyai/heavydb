/**
 * @file    IteratorTable.h
 * @author  Minggang Yu <miyu@mapd.com>
 * @brief   Basic constructors and methods of the iterator table interface.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_ITERATORTABLE_H
#define QUERYENGINE_ITERATORTABLE_H

#include "ResultRows.h"

#include <boost/variant.hpp>

class IteratorTable {
 public:
  IteratorTable(const QueryMemoryDescriptor& query_mem_desc,
                const std::vector<Analyzer::Expr*>& targets,
                const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                int64_t* group_by_buffer,
                const size_t groups_buffer_entry_count,
                const bool output_columnar,
                const std::vector<std::vector<const int8_t*>>& col_buffers,
                const ExecutorDeviceType device_type,
                const int device_id);

  void append(const IteratorTable& that) {
    group_by_buffer_frags_.insert(
        group_by_buffer_frags_.end(), that.group_by_buffer_frags_.begin(), that.group_by_buffer_frags_.end());
  }

  void fetchLazy(const std::unordered_map<size_t, ssize_t>& lazy_col_local_ids,
                 const std::vector<std::vector<const int8_t*>>& col_buffers,
                 const ssize_t frag_id);

  size_t colCount() const { return just_explain_ ? 1 : query_mem_desc_.agg_col_widths.size(); }

  bool definitelyHasNoRows() const { return group_by_buffer_frags_.empty() && !just_explain_ && !rowCount(); }

 private:
  size_t rowCount() const;

  const std::vector<TargetInfo> targets_;
  const QueryMemoryDescriptor query_mem_desc_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;

  std::vector<int64_t*> group_by_buffer_frags_;
  const size_t entry_count_per_frag_;
  const ExecutorDeviceType device_type_;

  const bool just_explain_;
  std::string explanation_;
};

typedef std::unique_ptr<IteratorTable> IterTabPtr;

typedef boost::variant<IterTabPtr, RowSetPtr> ResultPtr;

#endif  // QUERYENGINE_ITERATORTABLE_H
