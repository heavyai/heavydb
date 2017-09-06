/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

class QueryExecutionContext;

struct BufferFragment {
  int64_t* data;
  size_t row_count;
};

class IteratorTable {
 public:
  IteratorTable(const QueryMemoryDescriptor& query_mem_desc,
                const std::vector<Analyzer::Expr*>& targets,
                const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                int64_t* group_by_buffer,
                const size_t groups_buffer_entry_count,
                const std::vector<std::vector<const int8_t*>>& iter_buffers,
                const ssize_t frag_id,
                const ExecutorDeviceType device_type);

  IteratorTable(const std::vector<TargetInfo>& targets,
                const QueryMemoryDescriptor& query_mem_desc,
                const ExecutorDeviceType device_type,
                const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

  // Empty iterator table constructor
  IteratorTable();

  void append(const IteratorTable& that) {
    buffer_frags_.insert(buffer_frags_.end(), that.buffer_frags_.begin(), that.buffer_frags_.end());
  }

  void fetchLazy(const std::vector<std::vector<const int8_t*>>& iter_buffers, const ssize_t frag_id);

  size_t colCount() const { return just_explain_ ? 1 : query_mem_desc_.agg_col_widths.size(); }

  size_t fragCount() const { return buffer_frags_.size(); }

  size_t rowCount() const;

  const BufferFragment& getFragAt(const int frag_id) const {
    CHECK_LE(int(0), frag_id);
    CHECK_GT(buffer_frags_.size(), size_t(frag_id));
    return buffer_frags_[frag_id];
  }

  SQLTypeInfo getColType(const size_t col_idx) const {
    if (just_explain_) {
      return SQLTypeInfo(kTEXT, false);
    }
    return targets_[col_idx].sql_type;
  }

  bool definitelyHasNoRows() const { return buffer_frags_.empty() && !just_explain_ && !rowCount(); }

 private:
  void fuse(const IteratorTable& that);

  BufferFragment transformGroupByBuffer(const int64_t* group_by_buffer,
                                        const size_t groups_buffer_entry_count,
                                        const QueryMemoryDescriptor& query_mem_desc);

  const std::vector<TargetInfo> targets_;
  const QueryMemoryDescriptor query_mem_desc_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;

  std::vector<BufferFragment> buffer_frags_;
  const ExecutorDeviceType device_type_;

  const bool just_explain_;
  std::string explanation_;

  friend class QueryExecutionContext;
};

inline bool contains_iter_expr(const std::vector<Analyzer::Expr*>& target_exprs) {
  for (const auto& expr : target_exprs) {
    if (dynamic_cast<const Analyzer::IterExpr*>(expr)) {
      return true;
    }
  }
  return false;
}

typedef std::unique_ptr<IteratorTable> IterTabPtr;

typedef boost::variant<RowSetPtr, IterTabPtr> ResultPtr;

#endif  // QUERYENGINE_ITERATORTABLE_H
