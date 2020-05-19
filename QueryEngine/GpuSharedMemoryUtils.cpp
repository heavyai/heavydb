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

#include "GpuSharedMemoryUtils.h"
#include "ResultSetReductionJIT.h"
#include "RuntimeFunctions.h"

GpuSharedMemCodeBuilder::GpuSharedMemCodeBuilder(
    llvm::Module* module,
    llvm::LLVMContext& context,
    const QueryMemoryDescriptor& qmd,
    const std::vector<TargetInfo>& targets,
    const std::vector<int64_t>& init_agg_values)
    : module_(module)
    , context_(context)
    , reduction_func_(nullptr)
    , init_func_(nullptr)
    , query_mem_desc_(qmd)
    , targets_(targets)
    , init_agg_values_(init_agg_values) {
  /**
   * This class currently works only with:
   * 1. row-wise output memory layout
   * 2. GroupByPerfectHash
   * 3. single-column group by
   * 4. Keyless hash strategy (no redundant group column in the output buffer)
   *
   * All conditions in 1, 3, and 4 can be easily relaxed if proper code is added to
   * support them in the future.
   */
  CHECK(!query_mem_desc_.didOutputColumnar());
  CHECK(query_mem_desc_.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash);
  CHECK(query_mem_desc_.hasKeylessHash());
}

void GpuSharedMemCodeBuilder::codegen() {
  auto timer = DEBUG_TIMER(__func__);

  // codegen the init function
  init_func_ = createInitFunction();
  CHECK(init_func_);
  codegenInitialization();
  verify_function_ir(init_func_);

  // codegen the reduction function:
  reduction_func_ = createReductionFunction();
  CHECK(reduction_func_);
  codegenReduction();
  verify_function_ir(reduction_func_);
}

/**
 * The reduction function is going to be used to reduce group by buffer
 * stored in the shared memory, back into global memory buffer. The general
 * procedure is very similar to the what we have ResultSetReductionJIT, with some
 * major differences that will be discussed below:
 *
 * The general procedure is as follows:
 * 1. the function takes three arguments: 1) dest_buffer_ptr which points to global memory
 * group by buffer (what existed before), 2) src_buffer_ptr which points to the shared
 * memory group by buffer, exclusively accessed by each specific GPU thread-block, 3)
 * total buffer size.
 * 2. We assign each thread to a specific entry (all targets within that entry), so any
 * thread with an index larger than max entries, will have an early return from this
 * function
 * 3. It is assumed here that there are at least as many threads in the GPU as there are
 * entries in the group by buffer. In practice, given the buffer sizes that we deal with,
 * this is a reasonable asumption, but can be easily relaxed in the future if needed to:
 * threads can form a loop and process all entries until all are finished. It should be
 * noted that we currently don't use shared memory if there are more entries than number
 * of threads.
 * 4. We loop over all slots corresponding to a specific entry, and use
 * ResultSetReductionJIT's reduce_one_entry_idx to reduce one slot from the destination
 * buffer into source buffer. The only difference is that we should replace all agg_*
 * funcitons within this code with their agg_*_shared counterparts, which use atomics
 * operations and are used on the GPU.
 * 5. Once all threads are done, we return from the function.
 */
void GpuSharedMemCodeBuilder::codegenReduction() {
  CHECK(reduction_func_);
  // adding names to input arguments:
  auto arg_it = reduction_func_->arg_begin();
  auto dest_buffer_ptr = &*arg_it;
  dest_buffer_ptr->setName("dest_buffer_ptr");
  arg_it++;
  auto src_buffer_ptr = &*arg_it;
  src_buffer_ptr->setName("src_buffer_ptr");
  arg_it++;
  auto buffer_size = &*arg_it;
  buffer_size->setName("buffer_size");

  auto bb_entry = llvm::BasicBlock::Create(context_, ".entry", reduction_func_);
  auto bb_body = llvm::BasicBlock::Create(context_, ".body", reduction_func_);
  auto bb_exit = llvm::BasicBlock::Create(context_, ".exit", reduction_func_);
  llvm::IRBuilder<> ir_builder(bb_entry);

  // synchronize all threads within a threadblock:
  const auto sync_threadblock = getFunction("sync_threadblock");
  ir_builder.CreateCall(sync_threadblock, {});

  const auto func_thread_index = getFunction("get_thread_index");
  const auto thread_idx = ir_builder.CreateCall(func_thread_index, {}, "thread_index");

  // branching out of out of bound:
  const auto entry_count = ll_int(query_mem_desc_.getEntryCount(), context_);
  const auto entry_count_i32 =
      ll_int(static_cast<int32_t>(query_mem_desc_.getEntryCount()), context_);
  const auto is_thread_inbound =
      ir_builder.CreateICmpSLT(thread_idx, entry_count, "is_thread_inbound");
  ir_builder.CreateCondBr(is_thread_inbound, bb_body, bb_exit);

  ir_builder.SetInsertPoint(bb_body);

  // cast src/dest buffers into byte streams:
  auto src_byte_stream = ir_builder.CreatePointerCast(
      src_buffer_ptr, llvm::Type::getInt8PtrTy(context_, 0), "src_byte_stream");
  const auto dest_byte_stream = ir_builder.CreatePointerCast(
      dest_buffer_ptr, llvm::Type::getInt8PtrTy(context_, 0), "dest_byte_stream");

  // running the result set reduction JIT code to get reduce_one_entry_idx function
  auto rs_reduction_jit = std::make_unique<GpuReductionHelperJIT>(
      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc_),
      targets_,
      initialize_target_values_for_storage(targets_));
  auto reduction_code = rs_reduction_jit->codegen();
  reduction_code.module->setDataLayout(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  reduction_code.module->setTargetTriple("nvptx64-nvidia-cuda");

  llvm::Linker linker(*module_);
  bool link_error = linker.linkInModule(std::move(reduction_code.module));
  CHECK(!link_error);

  // go through the reduction code and replace all occurances of agg functions
  // with their _shared counterparts, which are specifically used in GPUs
  auto reduce_one_entry_func = getFunction("reduce_one_entry");
  bool agg_func_found = true;
  while (agg_func_found) {
    agg_func_found = false;
    for (auto it = llvm::inst_begin(reduce_one_entry_func);
         it != llvm::inst_end(reduce_one_entry_func);
         it++) {
      if (!llvm::isa<llvm::CallInst>(*it)) {
        continue;
      }
      auto& func_call = llvm::cast<llvm::CallInst>(*it);
      std::string func_name = func_call.getCalledFunction()->getName().str();
      if (func_name.length() > 4 && func_name.substr(0, 4) == "agg_") {
        if (func_name.length() > 7 &&
            func_name.substr(func_name.length() - 7) == "_shared") {
          continue;
        }
        agg_func_found = true;
        std::vector<llvm::Value*> args;
        for (size_t i = 0; i < func_call.getNumArgOperands(); ++i) {
          args.push_back(func_call.getArgOperand(i));
        }
        auto gpu_agg_func = getFunction(func_name + "_shared");
        llvm::ReplaceInstWithInst(&func_call,
                                  llvm::CallInst::Create(gpu_agg_func, args, ""));
        break;
      }
    }
  }
  const auto reduce_one_entry_idx_func = getFunction("reduce_one_entry_idx");
  CHECK(reduce_one_entry_idx_func);

  // qmd_handles are only used with count distinct and baseline group by
  // serialized varlen buffer is only used with SAMPLE on varlen types, which we will
  // disable for current shared memory support.
  const auto null_ptr_ll =
      llvm::ConstantPointerNull::get(llvm::Type::getInt8PtrTy(context_, 0));
  const auto thread_idx_i32 = ir_builder.CreateCast(
      llvm::Instruction::CastOps::Trunc, thread_idx, get_int_type(32, context_));
  ir_builder.CreateCall(reduce_one_entry_idx_func,
                        {dest_byte_stream,
                         src_byte_stream,
                         thread_idx_i32,
                         entry_count_i32,
                         null_ptr_ll,
                         null_ptr_ll,
                         null_ptr_ll},
                        "");
  ir_builder.CreateBr(bb_exit);
  llvm::ReturnInst::Create(context_, bb_exit);
}

namespace {
// given a particular destination ptr to the beginning of an entry, this function creates
// proper cast for a specific slot index.
// it also assumes these pointers are within shared memory address space (3)
llvm::Value* codegen_smem_dest_slot_ptr(llvm::LLVMContext& context,
                                        const QueryMemoryDescriptor& query_mem_desc,
                                        llvm::IRBuilder<>& ir_builder,
                                        const size_t slot_idx,
                                        const TargetInfo& target_info,
                                        llvm::Value* dest_byte_stream,
                                        llvm::Value* byte_offset) {
  const auto sql_type = get_compact_type(target_info);
  const auto slot_bytes = query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
  auto ptr_type = [&context](const size_t slot_bytes, const SQLTypeInfo& sql_type) {
    if (slot_bytes == sizeof(int32_t)) {
      return llvm::Type::getInt32PtrTy(context, /*address_space=*/3);
    } else {
      CHECK(slot_bytes == sizeof(int64_t));
      return llvm::Type::getInt64PtrTy(context, /*address_space=*/3);
    }
    UNREACHABLE() << "Invalid slot size encountered: " << std::to_string(slot_bytes);
    return llvm::Type::getInt32PtrTy(context, /*address_space=*/3);
  };

  const auto casted_dest_slot_address =
      ir_builder.CreatePointerCast(ir_builder.CreateGEP(dest_byte_stream, byte_offset),
                                   ptr_type(slot_bytes, sql_type),
                                   "dest_slot_adr_" + std::to_string(slot_idx));
  return casted_dest_slot_address;
}
}  // namespace

/**
 * This function generates code to initialize the shared memory buffer, the way we
 * initialize the group by output buffer on the host. Similar to the reduction function,
 * it is assumed that there are at least as many threads as there are entries in the
 * buffer. Each entry is assigned to a single thread, and then all slots corresponding to
 * that entry are initialized with aggregate init values.
 */
void GpuSharedMemCodeBuilder::codegenInitialization() {
  CHECK(init_func_);
  // similar to the rest of the system, we used fixup QMD to be able to handle reductions
  // it should be removed in the future.
  auto fixup_query_mem_desc = ResultSet::fixupQueryMemoryDescriptor(query_mem_desc_);
  CHECK(!fixup_query_mem_desc.didOutputColumnar());
  CHECK(fixup_query_mem_desc.hasKeylessHash());
  CHECK_GE(init_agg_values_.size(), targets_.size());

  auto bb_entry = llvm::BasicBlock::Create(context_, ".entry", init_func_);
  auto bb_body = llvm::BasicBlock::Create(context_, ".body", init_func_);
  auto bb_exit = llvm::BasicBlock::Create(context_, ".exit", init_func_);

  llvm::IRBuilder<> ir_builder(bb_entry);
  const auto func_thread_index = getFunction("get_thread_index");
  const auto thread_idx = ir_builder.CreateCall(func_thread_index, {}, "thread_index");

  // declare dynamic shared memory:
  const auto declare_smem_func = getFunction("declare_dynamic_shared_memory");
  const auto shared_mem_buffer =
      ir_builder.CreateCall(declare_smem_func, {}, "shared_mem_buffer");

  const auto entry_count = ll_int(fixup_query_mem_desc.getEntryCount(), context_);
  const auto is_thread_inbound =
      ir_builder.CreateICmpSLT(thread_idx, entry_count, "is_thread_inbound");
  ir_builder.CreateCondBr(is_thread_inbound, bb_body, bb_exit);

  ir_builder.SetInsertPoint(bb_body);
  // compute byte offset assigned to this thread:
  const auto row_size_bytes = ll_int(fixup_query_mem_desc.getRowWidth(), context_);
  auto byte_offset_ll = ir_builder.CreateMul(row_size_bytes, thread_idx, "byte_offset");

  const auto dest_byte_stream = ir_builder.CreatePointerCast(
      shared_mem_buffer, llvm::Type::getInt8PtrTy(context_), "dest_byte_stream");

  // each thread will be responsible for one
  const auto& col_slot_context = fixup_query_mem_desc.getColSlotContext();
  size_t init_agg_idx = 0;
  for (size_t target_logical_idx = 0; target_logical_idx < targets_.size();
       ++target_logical_idx) {
    const auto& target_info = targets_[target_logical_idx];
    const auto& slots_for_target = col_slot_context.getSlotsForCol(target_logical_idx);
    for (size_t slot_idx = slots_for_target.front(); slot_idx <= slots_for_target.back();
         slot_idx++) {
      const auto slot_size = fixup_query_mem_desc.getPaddedSlotWidthBytes(slot_idx);

      auto casted_dest_slot_address = codegen_smem_dest_slot_ptr(context_,
                                                                 fixup_query_mem_desc,
                                                                 ir_builder,
                                                                 slot_idx,
                                                                 target_info,
                                                                 dest_byte_stream,
                                                                 byte_offset_ll);

      llvm::Value* init_value_ll = nullptr;
      if (slot_size == sizeof(int32_t)) {
        init_value_ll =
            ll_int(static_cast<int32_t>(init_agg_values_[init_agg_idx++]), context_);
      } else if (slot_size == sizeof(int64_t)) {
        init_value_ll =
            ll_int(static_cast<int64_t>(init_agg_values_[init_agg_idx++]), context_);
      } else {
        UNREACHABLE() << "Invalid slot size encountered.";
      }
      ir_builder.CreateStore(init_value_ll, casted_dest_slot_address);

      // if not the last loop, we compute the next offset:
      if (slot_idx != (col_slot_context.getSlotCount() - 1)) {
        byte_offset_ll = ir_builder.CreateAdd(
            byte_offset_ll, ll_int(static_cast<size_t>(slot_size), context_));
      }
    }
  }

  ir_builder.CreateBr(bb_exit);

  ir_builder.SetInsertPoint(bb_exit);
  // synchronize all threads within a threadblock:
  const auto sync_threadblock = getFunction("sync_threadblock");
  ir_builder.CreateCall(sync_threadblock, {});
  ir_builder.CreateRet(shared_mem_buffer);
}

llvm::Function* GpuSharedMemCodeBuilder::createReductionFunction() const {
  std::vector<llvm::Type*> input_arguments;
  input_arguments.push_back(llvm::Type::getInt64PtrTy(context_));
  input_arguments.push_back(llvm::Type::getInt64PtrTy(context_));
  input_arguments.push_back(llvm::Type::getInt32Ty(context_));

  llvm::FunctionType* ft =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context_), input_arguments, false);
  const auto reduction_function = llvm::Function::Create(
      ft, llvm::Function::ExternalLinkage, "reduce_from_smem_to_gmem", module_);
  return reduction_function;
}

llvm::Function* GpuSharedMemCodeBuilder::createInitFunction() const {
  std::vector<llvm::Type*> input_arguments;
  input_arguments.push_back(
      llvm::Type::getInt64PtrTy(context_));                     // a pointer to the buffer
  input_arguments.push_back(llvm::Type::getInt32Ty(context_));  // buffer size in bytes

  llvm::FunctionType* ft = llvm::FunctionType::get(
      llvm::Type::getInt64PtrTy(context_), input_arguments, false);
  const auto init_function = llvm::Function::Create(
      ft, llvm::Function::ExternalLinkage, "init_smem_func", module_);
  return init_function;
}

llvm::Function* GpuSharedMemCodeBuilder::getFunction(const std::string& func_name) const {
  const auto function = module_->getFunction(func_name);
  CHECK(function) << func_name << " is not found in the module.";
  return function;
}

namespace {
/**
 * searches through the main function for the first appearance of called function
 * "target_func_name", and if found it replaces it with replace_func while keeping the
 * same arguments
 */
void replace_called_function_with(llvm::Function* main_func,
                                  const std::string& target_func_name,
                                  llvm::Function* replace_func) {
  for (auto it = llvm::inst_begin(main_func), e = llvm::inst_end(main_func); it != e;
       ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& instruction = llvm::cast<llvm::CallInst>(*it);
    if (std::string(instruction.getCalledFunction()->getName()) == target_func_name) {
      std::vector<llvm::Value*> args;
      for (size_t i = 0; i < instruction.getNumArgOperands(); ++i) {
        args.push_back(instruction.getArgOperand(i));
      }
      llvm::ReplaceInstWithInst(&instruction,
                                llvm::CallInst::Create(replace_func, args, ""));
      return;
    }
  }
  UNREACHABLE() << "Target function " << target_func_name << " was not found in "
                << replace_func->getName().str();
}

}  // namespace

void GpuSharedMemCodeBuilder::injectFunctionsInto(llvm::Function* query_func) {
  CHECK(reduction_func_);
  CHECK(init_func_);
  replace_called_function_with(query_func, "init_shared_mem", init_func_);
  replace_called_function_with(query_func, "write_back_nop", reduction_func_);
}

std::string GpuSharedMemCodeBuilder::toString() const {
  CHECK(reduction_func_);
  CHECK(init_func_);
  return serialize_llvm_object(init_func_) + serialize_llvm_object(reduction_func_);
}
