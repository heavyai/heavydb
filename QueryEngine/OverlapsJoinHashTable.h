/*
 * Copyright 2018 OmniSci, Inc.
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

#ifndef QUERYENGINE_OVERLAPSHASHTABLE_H
#define QUERYENGINE_OVERLAPSHASHTABLE_H

#include "BaselineJoinHashTable.h"

class OverlapsJoinHashTable : public BaselineJoinHashTable {
 public:
  OverlapsJoinHashTable(const std::shared_ptr<Analyzer::BinOper> condition,
                        const std::vector<InputTableInfo>& query_infos,
                        const RelAlgExecutionUnit& ra_exe_unit,
                        const Data_Namespace::MemoryLevel memory_level,
                        const size_t entry_count,
                        ColumnCacheMap& column_map,
                        Executor* executor)
      : BaselineJoinHashTable(condition,
                              query_infos,
                              ra_exe_unit,
                              memory_level,
                              entry_count,
                              column_map,
                              executor) {}

  virtual ~OverlapsJoinHashTable() {}

  static std::shared_ptr<OverlapsJoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const std::vector<InputTableInfo>& query_infos,
      const RelAlgExecutionUnit& ra_exe_unit,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_count,
      ColumnCacheMap& column_map,
      Executor* executor);

 protected:
  virtual void reifyWithLayout(const int device_count,
                               const JoinHashTableInterface::HashType layout) override;

  virtual ColumnsForDevice fetchColumnsForDevice(
      const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
      const int device_id) override;

  virtual std::pair<size_t, size_t> approximateTupleCount(
      const std::vector<ColumnsForDevice>&) const override;

  virtual size_t getKeyComponentWidth() const override;

  virtual size_t getKeyComponentCount() const override;

  virtual int initHashTableOnCpu(const std::vector<JoinColumn>& join_columns,
                                 const std::vector<JoinColumnTypeInfo>& join_column_types,
                                 const std::vector<JoinBucketInfo>& join_bucket_info,
                                 const JoinHashTableInterface::HashType layout) override;

  virtual int initHashTableOnGpu(const std::vector<JoinColumn>& join_columns,
                                 const std::vector<JoinColumnTypeInfo>& join_column_types,
                                 const std::vector<JoinBucketInfo>& join_bucket_info,
                                 const JoinHashTableInterface::HashType layout,
                                 const size_t key_component_width,
                                 const size_t key_component_count,
                                 const int device_id) override;

  virtual llvm::Value* codegenKey(const CompilationOptions&) override;

 private:
  void computeBucketSizes(std::vector<double>& bucket_sizes_for_dimension,
                          const JoinColumn& join_column,
                          const std::vector<InnerOuter>& inner_outer_pairs,
                          const size_t row_count);

  std::vector<double> bucket_sizes_for_dimension_;
};

#endif  // QUERYENGINE_OVERLAPSHASHTABLE_H
