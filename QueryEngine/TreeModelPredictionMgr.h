/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
      Executor* executor,
      const std::vector<std::vector<DecisionTreeEntry>>& decision_trees,
      const std::vector<int64_t>& decision_tree_offsets,
      const bool compute_avg);
  ~TreeModelPredictionMgr();

  llvm::Value* codegen(const std::vector<llvm::Value*>& regressor_inputs,
                       const CompilationOptions& co) const;

 private:
  void allocateAndPopulateHostBuffers(
      const std::vector<std::vector<DecisionTreeEntry>>& decision_trees,
      const std::vector<int64_t>& decision_tree_offsets);
  void createKernelBuffers();

  const Data_Namespace::MemoryLevel memory_level_;
  Executor* executor_;
  Data_Namespace::DataMgr* data_mgr_;
  const std::set<int> device_ids_;
  const int32_t num_trees_;
  const bool compute_avg_;
  int8_t* host_decision_tree_table_;
  int8_t* host_decision_tree_offsets_;
  int64_t decision_tree_table_size_bytes_;
  int64_t decision_tree_offsets_size_bytes_;
  std::vector<Data_Namespace::AbstractBuffer*> decision_tree_table_device_buffers_;
  std::vector<Data_Namespace::AbstractBuffer*> decision_tree_offsets_device_buffers_;
  std::unordered_map<int, const int8_t*> kernel_decision_tree_tables_;
  std::unordered_map<int, const int8_t*> kernel_decision_tree_offsets_;
};
