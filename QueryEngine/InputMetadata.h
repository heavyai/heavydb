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

#ifndef QUERYENGINE_INPUTMETADATA_H
#define QUERYENGINE_INPUTMETADATA_H

#include "QueryEngine/Descriptors/InputDescriptors.h"
#include "QueryEngine/RelAlgExecutionUnit.h"

#include <unordered_map>

namespace Catalog_Namespace {
class Catalog;
}  // namespace Catalog_Namespace

class Executor;

class TemporaryTable {
 public:
  TemporaryTable() {}
  TemporaryTable(const ResultSetPtr& rs) { results_.emplace_back(rs); }
  TemporaryTable(ResultSetPtr&& rs) { results_.emplace_back(rs); }
  TemporaryTable(const std::vector<ResultSetPtr>& results) : results_(results) {}
  TemporaryTable(std::vector<ResultSetPtr>&& results) : results_(results) {}

  TemporaryTable(const TemporaryTable& other) = default;
  TemporaryTable(TemporaryTable&& other) = default;

  TemporaryTable& operator=(const TemporaryTable& other) = default;
  TemporaryTable& operator=(TemporaryTable&& other) = default;

  int getFragCount() const { return static_cast<int>(results_.size()); }

  const ResultSetPtr& getResultSet(const int frag_id) const {
    CHECK(frag_id < getFragCount());
    return results_[frag_id];
  }

  size_t getLimit() const;
  size_t rowCount() const;
  size_t colCount() const;

  SQLTypeInfo getColType(const size_t col_idx) const;

  bool empty() const { return results_.empty(); }

  void setKernelQueueTime(const int64_t kernel_queue_time);
  void addCompilationQueueTime(const int64_t compilation_queue_time);
  void setValidationOnlyRes();

  ResultSetPtr& operator[](size_t idx) { return results_[idx]; }
  const ResultSetPtr& operator[](size_t idx) const { return results_[idx]; }

 private:
  std::vector<ResultSetPtr> results_;
};

// using TemporaryTables = std::unordered_map<int, const ResultSetPtr&>;
using TemporaryTables = std::unordered_map<int, TemporaryTable>;

struct InputTableInfo {
  int table_id;
  Fragmenter_Namespace::TableInfo info;
};

class InputTableInfoCache {
 public:
  InputTableInfoCache(Executor* executor);

  Fragmenter_Namespace::TableInfo getTableInfo(const int table_id);

  void clear();

 private:
  std::unordered_map<int, Fragmenter_Namespace::TableInfo> cache_;
  Executor* executor_;
};

ChunkMetadataMap synthesize_metadata(const ResultSet* rows);

size_t get_frag_count_of_table(const int table_id, Executor* executor);

std::vector<InputTableInfo> get_table_infos(
    const std::vector<InputDescriptor>& input_descs,
    Executor* executor);

std::vector<InputTableInfo> get_table_infos(const RelAlgExecutionUnit& ra_exe_unit,
                                            Executor* executor);

Fragmenter_Namespace::TableInfo build_table_info(
    const std::vector<const TableDescriptor*>& shard_tables);

#endif  // QUERYENGINE_INPUTMETADATA_H
