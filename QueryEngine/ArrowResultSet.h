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

#ifndef QUERYENGINE_ARROWRESULTSET_H
#define QUERYENGINE_ARROWRESULTSET_H

#include "../Shared/sqltypes.h"
#include "ResultSet.h"
#include "Shared/SqlTypesLayout.h"
#include "TargetMetaInfo.h"
#include "TargetValue.h"

#include <arrow/api.h>

class ArrowResultSet;

class ArrowResultSetRowIterator {
 public:
  using value_type = std::vector<TargetValue>;
  using difference_type = std::ptrdiff_t;
  using pointer = std::vector<TargetValue>*;
  using reference = std::vector<TargetValue>&;
  using iterator_category = std::input_iterator_tag;

  bool operator==(const ArrowResultSetRowIterator& other) const {
    return result_set_ == other.result_set_ && crt_row_idx_ == other.crt_row_idx_;
  }
  bool operator!=(const ArrowResultSetRowIterator& other) const {
    return !(*this == other);
  }

  inline value_type operator*() const;
  ArrowResultSetRowIterator& operator++(void) {
    crt_row_idx_++;
    return *this;
  }
  ArrowResultSetRowIterator operator++(int) {
    ArrowResultSetRowIterator iter(*this);
    ++(*this);
    return iter;
  }

 private:
  const ArrowResultSet* result_set_;
  size_t crt_row_idx_;

  ArrowResultSetRowIterator(const ArrowResultSet* rs)
      : result_set_(rs), crt_row_idx_(0){};

  friend class ArrowResultSet;
};

// Expose Arrow buffers as a subset of the ResultSet interface
// to make it work within the existing execution test framework.
class ArrowResultSet {
 public:
  ArrowResultSet(const std::shared_ptr<arrow::RecordBatch>& record_batch);

  ArrowResultSetRowIterator rowIterator(size_t from_index,
                                        bool translate_strings,
                                        bool decimal_to_double) const {
    ArrowResultSetRowIterator iter(this);
    for (size_t i = 0; i < from_index; i++) {
      ++iter;
    }

    return iter;
  }

  ArrowResultSetRowIterator rowIterator(bool translate_strings,
                                        bool decimal_to_double) const {
    return rowIterator(0, translate_strings, decimal_to_double);
  }

  std::vector<TargetValue> getRowAt(const size_t index) const;

  std::vector<TargetValue> getNextRow(const bool translate_strings,
                                      const bool decimal_to_double) const;

  size_t colCount() const;

  SQLTypeInfo getColType(const size_t col_idx) const;

  bool definitelyHasNoRows() const;

  size_t rowCount() const;

 private:
  std::shared_ptr<arrow::RecordBatch> record_batch_;

  // Boxed arrays from the record batch. The result of RecordBatch::column is
  // temporary, so we cache these for better performance
  std::vector<std::shared_ptr<arrow::Array>> columns_;
  mutable size_t crt_row_idx_;
  std::vector<TargetMetaInfo> column_metainfo_;
};

ArrowResultSetRowIterator::value_type ArrowResultSetRowIterator::operator*() const {
  return result_set_->getRowAt(crt_row_idx_);
}

class ExecutionResult;

// Take results from the executor, serializes them to Arrow and then deserialize
// them to ArrowResultSet, which can then be used by the existing test framework.
std::unique_ptr<ArrowResultSet> result_set_arrow_loopback(const ExecutionResult& results);

// QUERYENGINE_// Take results from the executor, serializes them to Arrow and then
// deserialize them to ArrowResultSet, which can then be used by the existing test
// framework.
std::unique_ptr<ArrowResultSet> result_set_arrow_loopback(
    const ExecutionResult* results,
    const std::shared_ptr<ResultSet>& rows);

#endif  // QUERYENGINE_ARROWRESULTSET_H
