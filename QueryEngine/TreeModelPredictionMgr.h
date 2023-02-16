/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#pragma once

#include "DataMgr/MemoryLevel.h"
#include "DecisionTreeEntry.h"

#include <vector>

struct CompilationOptions;

namespace Data_Namespace {
class DataMgr;
class AbstractBuffer;
}  // namespace Data_Namespace

class Executor;

namespace llvm {
class Value;
}

class TreeModelPredictionMgr {
 public:
  TreeModelPredictionMgr(
      const Data_Namespace::MemoryLevel memory_level,
      const int device_count,
      Executor* executor,
      Data_Namespace::DataMgr* data_mgr,
      const std::vector<std::vector<DecisionTreeEntry>>& decision_trees,
      const std::vector<int64_t>& decision_tree_offsets)
      : memory_level_(memory_level)
      , device_count_(device_count)
      , executor_(executor)
      , data_mgr_(data_mgr)
      , num_trees_(decision_trees.size()) {
#ifdef HAVE_CUDA
    CHECK(memory_level_ == Data_Namespace::CPU_LEVEL ||
          memory_level_ == Data_Namespace::GPU_LEVEL);
#else
    CHECK_EQ(Data_Namespace::CPU_LEVEL, memory_level_);
#endif  // HAVE_CUDA
    allocateAndPopulateHostBuffers(decision_trees, decision_tree_offsets);
    createKernelBuffers();
  }

  llvm::Value* codegen(const std::vector<llvm::Value*>& regressor_inputs,
                       const CompilationOptions& co) const;

 private:
  void allocateAndPopulateHostBuffers(
      const std::vector<std::vector<DecisionTreeEntry>>& decision_trees,
      const std::vector<int64_t>& decision_tree_offsets);
  void createKernelBuffers();

  const Data_Namespace::MemoryLevel memory_level_;
  const int device_count_;
  Executor* executor_;
  Data_Namespace::DataMgr* data_mgr_;
  const int32_t num_trees_;
  int8_t* host_decision_tree_table_;
  int8_t* host_decision_tree_offsets_;
  int64_t decision_tree_table_size_bytes_;
  int64_t decision_tree_offsets_size_bytes_;
  std::vector<Data_Namespace::AbstractBuffer*> decision_tree_table_device_buffers_;
  std::vector<Data_Namespace::AbstractBuffer*> decision_tree_offsets_device_buffers_;
  std::vector<const int8_t*> kernel_decision_tree_tables_;
  std::vector<const int8_t*> kernel_decision_tree_offsets_;
};
