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

#include "TreeModelPredictionMgr.h"
#include "CodeGenerator.h"

#ifdef HAVE_CUDA
#include "DataMgr/Allocators/CudaAllocator.h"
#include "GpuMemUtils.h"
#endif  // HAVE_CUDA
#include "Parser/ParserNode.h"

#include <tbb/parallel_for.h>

TreeModelPredictionMgr::TreeModelPredictionMgr(
    const Data_Namespace::MemoryLevel memory_level,
    Executor* executor,
    const std::vector<std::vector<DecisionTreeEntry>>& decision_trees,
    const std::vector<int64_t>& decision_tree_offsets,
    const bool compute_avg)
    : memory_level_(memory_level)
    , executor_(executor)
    , data_mgr_(executor->getDataMgr())
    , device_count_(executor->deviceCount(memory_level == Data_Namespace::GPU_LEVEL
                                              ? ExecutorDeviceType::GPU
                                              : ExecutorDeviceType::CPU))
    , num_trees_(decision_trees.size())
    , compute_avg_(compute_avg) {
#ifdef HAVE_CUDA
  CHECK(memory_level_ == Data_Namespace::CPU_LEVEL ||
        memory_level_ == Data_Namespace::GPU_LEVEL);
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, memory_level_);
#endif  // HAVE_CUDA
  allocateAndPopulateHostBuffers(decision_trees, decision_tree_offsets);
  createKernelBuffers();
}

TreeModelPredictionMgr::~TreeModelPredictionMgr() {
  CHECK(data_mgr_);
  for (auto* buffer : decision_tree_table_device_buffers_) {
    CHECK(buffer);
    data_mgr_->free(buffer);
  }
  for (auto* buffer : decision_tree_offsets_device_buffers_) {
    CHECK(buffer);
    data_mgr_->free(buffer);
  }
}

void TreeModelPredictionMgr::allocateAndPopulateHostBuffers(
    const std::vector<std::vector<DecisionTreeEntry>>& decision_trees,
    const std::vector<int64_t>& decision_tree_offsets) {
  auto timer = DEBUG_TIMER(__func__);
  const size_t num_trees = decision_trees.size();
  CHECK_EQ(num_trees, static_cast<size_t>(num_trees_));
  CHECK_EQ(num_trees, decision_tree_offsets.size() - 1);
  const size_t num_tree_entries = decision_tree_offsets[num_trees];
  decision_tree_table_size_bytes_ = num_tree_entries * sizeof(DecisionTreeEntry);
  decision_tree_offsets_size_bytes_ = decision_tree_offsets.size() * sizeof(size_t);
  host_decision_tree_table_ =
      executor_->getRowSetMemoryOwner()->allocate(decision_tree_table_size_bytes_);
  host_decision_tree_offsets_ =
      executor_->getRowSetMemoryOwner()->allocate(decision_tree_offsets_size_bytes_);
  // Take this opportunity to copy offsets buffer over
  std::memcpy(host_decision_tree_offsets_,
              reinterpret_cast<const int8_t*>(decision_tree_offsets.data()),
              decision_tree_offsets_size_bytes_);

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, num_trees), [&](const tbb::blocked_range<size_t>& r) {
        const auto start_tree_idx = r.begin();
        const auto end_tree_idx = r.end();
        for (size_t tree_idx = start_tree_idx; tree_idx < end_tree_idx; ++tree_idx) {
          std::memcpy(host_decision_tree_table_ +
                          decision_tree_offsets[tree_idx] * sizeof(DecisionTreeEntry),
                      reinterpret_cast<const int8_t*>(decision_trees[tree_idx].data()),
                      decision_trees[tree_idx].size() * sizeof(DecisionTreeEntry));
        }
      });
}

void TreeModelPredictionMgr::createKernelBuffers() {
  auto timer = DEBUG_TIMER(__func__);
#ifdef HAVE_CUDA
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      decision_tree_table_device_buffers_.emplace_back(
          CudaAllocator::allocGpuAbstractBuffer(
              data_mgr_, decision_tree_table_size_bytes_, device_id));
      decision_tree_offsets_device_buffers_.emplace_back(
          CudaAllocator::allocGpuAbstractBuffer(
              data_mgr_, decision_tree_offsets_size_bytes_, device_id));
      auto decision_tree_table_device_buffer = reinterpret_cast<const int8_t*>(
          decision_tree_table_device_buffers_.back()->getMemoryPtr());
      auto decision_tree_offsets_device_buffer = reinterpret_cast<const int8_t*>(
          decision_tree_offsets_device_buffers_.back()->getMemoryPtr());
      copy_to_nvidia_gpu(data_mgr_,
                         reinterpret_cast<CUdeviceptr>(decision_tree_table_device_buffer),
                         reinterpret_cast<const int8_t*>(host_decision_tree_table_),
                         decision_tree_table_size_bytes_,
                         device_id);
      copy_to_nvidia_gpu(
          data_mgr_,
          reinterpret_cast<CUdeviceptr>(decision_tree_offsets_device_buffer),
          reinterpret_cast<const int8_t*>(host_decision_tree_offsets_),
          decision_tree_offsets_size_bytes_,
          device_id);
      kernel_decision_tree_tables_.push_back(decision_tree_table_device_buffer);
      kernel_decision_tree_offsets_.push_back(decision_tree_offsets_device_buffer);
    }
  }
#else
  CHECK_EQ(1, device_count_);
#endif
  if (memory_level_ == Data_Namespace::CPU_LEVEL) {
    kernel_decision_tree_tables_.push_back(host_decision_tree_table_);
    kernel_decision_tree_offsets_.push_back(host_decision_tree_offsets_);
  }
}

std::pair<std::vector<std::shared_ptr<const Analyzer::Constant>>,
          std::vector<const Analyzer::Constant*>>
generate_kernel_buffer_constants(CgenState* cgen_state_ptr,
                                 const std::vector<const int8_t*>& kernel_buffers,
                                 const bool hoist_literals) {
  std::vector<std::shared_ptr<const Analyzer::Constant>> kernel_buffer_constants_owned;
  std::vector<const Analyzer::Constant*> kernel_buffer_constants;
  for (const auto kernel_buffer : kernel_buffers) {
    const int64_t kernel_buffer_handle = reinterpret_cast<int64_t>(kernel_buffer);
    const auto kernel_buffer_handle_literal =
        std::dynamic_pointer_cast<Analyzer::Constant>(
            Parser::IntLiteral::analyzeValue(kernel_buffer_handle));
    CHECK_EQ(kENCODING_NONE,
             kernel_buffer_handle_literal->get_type_info().get_compression());
    kernel_buffer_constants_owned.push_back(kernel_buffer_handle_literal);
    kernel_buffer_constants.push_back(kernel_buffer_handle_literal.get());
  }
  CHECK_GE(kernel_buffer_constants.size(), 1UL);
  CHECK(hoist_literals || kernel_buffer_constants.size() == 1UL);

  return std::make_pair(kernel_buffer_constants_owned, kernel_buffer_constants);
}

llvm::Value* TreeModelPredictionMgr::codegen(
    const std::vector<llvm::Value*>& regressor_inputs,
    const CompilationOptions& co) const {
  CHECK(kernel_decision_tree_tables_.size() == kernel_decision_tree_offsets_.size());
  CHECK(kernel_decision_tree_tables_.size() == static_cast<size_t>(device_count_));
  if (!co.hoist_literals && kernel_decision_tree_tables_.size() > 1UL) {
    CHECK(memory_level_ == Data_Namespace::GPU_LEVEL);
    CHECK(co.device_type == ExecutorDeviceType::GPU);
    throw QueryMustRunOnCpu();
  }
  CHECK(co.hoist_literals || kernel_decision_tree_tables_.size() == 1UL);

  auto cgen_state_ptr = executor_->getCgenStatePtr();
  AUTOMATIC_IR_METADATA(cgen_state_ptr);

  const auto [decision_tree_table_constants_owned, decision_tree_table_constants] =
      generate_kernel_buffer_constants(
          cgen_state_ptr, kernel_decision_tree_tables_, co.hoist_literals);

  const auto [decision_tree_offsets_constants_owned, decision_tree_offsets_constants] =
      generate_kernel_buffer_constants(
          cgen_state_ptr, kernel_decision_tree_offsets_, co.hoist_literals);

  CodeGenerator code_generator(executor_);

  const auto decision_tree_table_handle_lvs =
      co.hoist_literals
          ? code_generator.codegenHoistedConstants(
                decision_tree_table_constants, kENCODING_NONE, {})
          : code_generator.codegen(decision_tree_table_constants[0], false, co);

  const auto decision_tree_offsets_handle_lvs =
      co.hoist_literals
          ? code_generator.codegenHoistedConstants(
                decision_tree_offsets_constants, kENCODING_NONE, {})
          : code_generator.codegen(decision_tree_offsets_constants[0], false, co);

  auto& builder = cgen_state_ptr->ir_builder_;
  const int32_t num_regressors = static_cast<int32_t>(regressor_inputs.size());
  auto regressor_ty = llvm::Type::getDoubleTy(cgen_state_ptr->context_);
  llvm::ArrayType* regressor_arr_type =
      llvm::ArrayType::get(regressor_ty, num_regressors);
  auto regressor_local_storage_lv =
      builder.CreateAlloca(regressor_arr_type, nullptr, "Regressor_Local_Storage");
  auto idx_lv = cgen_state_ptr->llInt(0);
  auto regressor_local_storage_gep = llvm::GetElementPtrInst::CreateInBounds(
      regressor_local_storage_lv->getType()->getScalarType()->getPointerElementType(),
      regressor_local_storage_lv,
      {idx_lv, idx_lv},
      "",
      builder.GetInsertBlock());
  for (int32_t reg_idx = 0; reg_idx < num_regressors; ++reg_idx) {
    auto reg_ptr = builder.CreateGEP(
        regressor_local_storage_lv->getType()->getScalarType()->getPointerElementType(),
        regressor_local_storage_lv,
        {cgen_state_ptr->llInt(0), cgen_state_ptr->llInt(reg_idx)},
        "");
    builder.CreateStore(regressor_inputs[reg_idx], reg_ptr);
  }
  const double translated_null_value = inline_fp_null_value<double>();

  return cgen_state_ptr->emitCall(
      "tree_model_reg_predict",
      {regressor_local_storage_gep,
       cgen_state_ptr->castToTypeIn(decision_tree_table_handle_lvs.front(), 64),
       cgen_state_ptr->castToTypeIn(decision_tree_offsets_handle_lvs.front(), 64),
       cgen_state_ptr->llInt(num_regressors),
       cgen_state_ptr->llInt(num_trees_),
       cgen_state_ptr->llBool(compute_avg_),
       cgen_state_ptr->llFp(translated_null_value)});
}
