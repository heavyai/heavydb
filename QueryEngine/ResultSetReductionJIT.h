/*
 * Copyright 2019 OmniSci, Inc.
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

#include "CgenState.h"
#include "CodeCache.h"
#include "ResultSetReductionOps.h"

#include "Descriptors/QueryMemoryDescriptor.h"
#include "Shared/TargetInfo.h"

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>

struct ReductionCode {
  // Function which reduces 'that_buff' into 'this_buff', for rows between
  // [start_entry_index, end_entry_index).
  using FuncPtr = int32_t (*)(int8_t* this_buff,
                              const int8_t* that_buff,
                              const int32_t start_entry_index,
                              const int32_t end_entry_index,
                              const int32_t that_entry_count,
                              const void* this_qmd,
                              const void* that_qmd,
                              const void* serialized_varlen_buffer);

  FuncPtr func_ptr;
  llvm::Function* llvm_reduce_loop;
  std::unique_ptr<CgenState> cgen_state;
  std::unique_ptr<llvm::Module> module;
  std::unique_ptr<Function> ir_is_empty;
  std::unique_ptr<Function> ir_reduce_one_entry;
  std::unique_ptr<Function> ir_reduce_one_entry_idx;
  std::unique_ptr<Function> ir_reduce_loop;

  static std::mutex s_reduction_mutex;
};

class ResultSetReductionJIT {
 public:
  ResultSetReductionJIT(const QueryMemoryDescriptor& query_mem_desc,
                        const std::vector<TargetInfo>& targets,
                        const std::vector<int64_t>& target_init_vals);

  // Generate the code for the result set reduction loop.
  virtual ReductionCode codegen() const;
  static void clearCache();

 protected:
  // Generate a function which checks whether a row is empty.
  void isEmpty(const ReductionCode& reduction_code) const;

  // Generate a function which reduces two rows given by their start pointer, for the
  // perfect hash layout.
  void reduceOneEntryNoCollisions(const ReductionCode& reduction_code) const;

  // Generate a function which reduces two rows given by the start pointer of the result
  // buffers they are part of and the indices inside those buffers.
  void reduceOneEntryNoCollisionsIdx(const ReductionCode& reduction_code) const;

  // Generate a function for the reduction of an entire result set chunk.
  void reduceLoop(const ReductionCode& reduction_code) const;

 private:
  // Used to implement 'reduceOneEntryNoCollisions'.
  void reduceOneEntryTargetsNoCollisions(Function* ir_reduce_one_entry,
                                         Value* this_targets_start_ptr,
                                         Value* that_targets_start_ptr) const;

  // Same as above, for the baseline layout.
  void reduceOneEntryBaseline(const ReductionCode& reduction_code) const;

  // Same as above, for the baseline layout.
  void reduceOneEntryBaselineIdx(const ReductionCode& reduction_code) const;

  // Generate reduction code for a single slot.
  void reduceOneSlot(Value* this_ptr1,
                     Value* this_ptr2,
                     Value* that_ptr1,
                     Value* that_ptr2,
                     const TargetInfo& target_info,
                     const size_t target_logical_idx,
                     const size_t target_slot_idx,
                     const size_t init_agg_val_idx,
                     const size_t first_slot_idx_for_target,
                     Function* ir_reduce_one_entry) const;

  // Generate reduction code for a single aggregate (with the exception of sample) slot.
  void reduceOneAggregateSlot(Value* this_ptr1,
                              Value* this_ptr2,
                              Value* that_ptr1,
                              Value* that_ptr2,
                              const TargetInfo& target_info,
                              const size_t target_logical_idx,
                              const size_t target_slot_idx,
                              const int64_t init_val,
                              const int8_t chosen_bytes,
                              Function* ir_reduce_one_entry) const;

  // Generate reduction code for a count distinct slot.
  void reduceOneCountDistinctSlot(Value* this_ptr1,
                                  Value* that_ptr1,
                                  const size_t target_logical_idx,
                                  Function* ir_reduce_one_entry) const;

  ReductionCode finalizeReductionCode(ReductionCode reduction_code,
                                      const llvm::Function* ir_is_empty,
                                      const llvm::Function* ir_reduce_one_entry,
                                      const llvm::Function* ir_reduce_one_entry_idx,
                                      const CodeCacheKey& key) const;

  std::string cacheKey() const;

  const QueryMemoryDescriptor query_mem_desc_;
  const std::vector<TargetInfo> targets_;
  const std::vector<int64_t> target_init_vals_;
  static CodeCache s_code_cache;
};

/**
 * This is a helper class for performing GPU reduction code.
 * It uses the same functions that ResultSetReductionJIT generates so that
 * it can be reused and help the reduction procedure within GPU.
 */
class GpuReductionHelperJIT : public ResultSetReductionJIT {
 public:
  GpuReductionHelperJIT(const QueryMemoryDescriptor& query_mem_desc,
                        const std::vector<TargetInfo>& targets,
                        const std::vector<int64_t>& target_init_vals)
      : ResultSetReductionJIT(query_mem_desc, targets, target_init_vals)
      , query_mem_desc_(query_mem_desc) {
    CHECK(query_mem_desc_.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash);
    CHECK(!query_mem_desc_.didOutputColumnar());
    CHECK(query_mem_desc_.hasKeylessHash());
  }
  /**
   * generates code for perfect hash group by reduction: the following functions are
   * internally created: isEmpty, reduceOneEntryNoCollision (reduce for perfect hash),
   * reduceOneEntryNoCollissionsIdx(reduce one slot for perfect hash), and reduceLoop (the
   * outer loop).
   */
  virtual ReductionCode codegen() const;

 private:
  const QueryMemoryDescriptor& query_mem_desc_;
};