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

#include "SqlTypesLayout.h"
#include "TargetMetaInfo.h"
#include "TargetValue.h"
#include "../Shared/sqltypes.h"

#include <arrow/api.h>

// Expose Arrow buffers as a subset of the ResultSet interface
// to make it work within the existing execution test framework.
class ArrowResultSet {
 public:
  ArrowResultSet(const std::shared_ptr<arrow::RecordBatch>& record_batch);

  std::vector<TargetValue> getNextRow(const bool translate_strings, const bool decimal_to_double) const;

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

class ExecutionResult;

// Take results from the executor, serializes them to Arrow and then deserialize
// them to ArrowResultSet, which can then be used by the existing test framework.
std::unique_ptr<ArrowResultSet> result_set_arrow_loopback(const ExecutionResult& results);

#endif  // QUERYENGINE_ARROWRESULTSET_H
