/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GpuSharedMemoryTest.h"
#include "CudaMgr/CudaMgr.h"
#include "QueryEngine/GpuInitGroups.h"
#include "QueryEngine/JoinHashTable/RankedBitmapHashTable.h"
#include "QueryEngine/LLVMGlobalContext.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "QueryEngine/QueryEngine.h"
#include "QueryEngine/ResultSetReductionJIT.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "Shared/scope.h"
#include "Tests/DataMgrTestHelpers.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <limits>
#include <map>
#include <optional>

extern bool g_is_test_env;

extern std::unique_ptr<CudaMgr_Namespace::CudaMgr> g_cuda_mgr;
std::shared_ptr<QueryEngine> g_query_engine;

TEST(RankedBitmapHashTableHost, OverflowingResizePreservesOriginalLayout) {
  constexpr size_t max_slab_size = 64;
  RankedBitmapHashTable hash_table(ExecutorDeviceType::CPU,
                                   /*bit_count=*/1,
                                   /*payload_count=*/1,
                                   max_slab_size,
                                   nullptr,
                                   -1,
                                   HashType::OneToOne,
                                   /*distinct_count=*/0,
                                   /*force_segmented_layout=*/true);

  const auto original_bytes = hash_table.getAllocatedBytes();
  EXPECT_THROW(hash_table.resizeOneToOnePayloadBuffer(std::numeric_limits<size_t>::max(),
                                                      max_slab_size),
               std::overflow_error);
  EXPECT_EQ(hash_table.getLayout(), HashType::OneToOne);
  EXPECT_EQ(hash_table.getPayloadCount(), size_t(1));
  EXPECT_EQ(hash_table.getAllocatedBytes(), original_bytes);

  EXPECT_THROW(hash_table.allocateOneToManyBuffers(
                   std::numeric_limits<size_t>::max() / 2 + 1, max_slab_size),
               std::overflow_error);
  EXPECT_EQ(hash_table.getLayout(), HashType::OneToOne);
  EXPECT_EQ(hash_table.getDistinctCount(), size_t(0));
  EXPECT_EQ(hash_table.getPayloadCount(), size_t(1));
  EXPECT_EQ(hash_table.getAllocatedBytes(), original_bytes);
}

namespace {

void init_storage_buffer(int8_t* buffer,
                         const std::vector<TargetInfo>& targets,
                         const QueryMemoryDescriptor& query_mem_desc) {
  // get the initial values for all the aggregate columns
  const auto init_agg_vals = init_agg_val_vec(targets, query_mem_desc);
  CHECK(!query_mem_desc.didOutputColumnar());
  CHECK(query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash);

  const auto row_size = query_mem_desc.getRowSize();
  CHECK(query_mem_desc.hasKeylessHash());
  for (size_t entry_idx = 0; entry_idx < query_mem_desc.getEntryCount(); ++entry_idx) {
    const auto row_ptr = buffer + entry_idx * row_size;
    size_t init_agg_idx{0};
    int64_t init_val{0};
    // initialize each row's aggregate columns:
    auto col_ptr = row_ptr + query_mem_desc.getColOffInBytes(0);
    for (size_t slot_idx = 0; slot_idx < query_mem_desc.getSlotCount(); slot_idx++) {
      if (query_mem_desc.getPaddedSlotWidthBytes(slot_idx) > 0) {
        init_val = init_agg_vals[init_agg_idx++];
      }
      switch (query_mem_desc.getPaddedSlotWidthBytes(slot_idx)) {
        case 4:
          *reinterpret_cast<int32_t*>(col_ptr) = static_cast<int32_t>(init_val);
          break;
        case 8:
          *reinterpret_cast<int64_t*>(col_ptr) = init_val;
          break;
        case 0:
          break;
        default:
          UNREACHABLE();
      }
      col_ptr += query_mem_desc.getNextColOffInBytes(col_ptr, entry_idx, slot_idx);
    }
  }
}

}  // namespace

namespace {

constexpr int kReductionTestDeviceId = 0;
constexpr int64_t kInt64NullSentinel = std::numeric_limits<int64_t>::min() + 17;
constexpr int32_t kInt32NullSentinel = std::numeric_limits<int32_t>::min() + 17;

struct RuntimeDeviceBuffer {
  explicit RuntimeDeviceBuffer(const size_t bytes) : bytes_(bytes) {
    CHECK_GT(bytes_, size_t(0));
    CHECK_EQ(cudaSetDevice(kReductionTestDeviceId), cudaSuccess);
    CHECK_EQ(cudaMalloc(&ptr_, bytes_), cudaSuccess);
  }

  RuntimeDeviceBuffer(const RuntimeDeviceBuffer&) = delete;
  RuntimeDeviceBuffer& operator=(const RuntimeDeviceBuffer&) = delete;

  ~RuntimeDeviceBuffer() {
    if (ptr_) {
      CHECK_EQ(cudaFree(ptr_), cudaSuccess);
    }
  }

  int8_t* ptr() const { return static_cast<int8_t*>(ptr_); }

  void copyFromHost(const std::vector<int8_t>& host) const {
    CHECK_EQ(host.size(), bytes_);
    CHECK_EQ(cudaMemcpy(ptr_, host.data(), bytes_, cudaMemcpyHostToDevice), cudaSuccess);
  }

  void copyToHost(std::vector<int8_t>& host) const {
    CHECK_EQ(host.size(), bytes_);
    CHECK_EQ(cudaMemcpy(host.data(), ptr_, bytes_, cudaMemcpyDeviceToHost), cudaSuccess);
  }

 private:
  void* ptr_{nullptr};
  size_t bytes_{0};
};

void write_i32(int8_t* row, const size_t offset, const int32_t value) {
  *reinterpret_cast<int32_t*>(row + offset) = value;
}

void write_i64(int8_t* row, const size_t offset, const int64_t value) {
  *reinterpret_cast<int64_t*>(row + offset) = value;
}

int32_t read_i32(const int8_t* row, const size_t offset) {
  return *reinterpret_cast<const int32_t*>(row + offset);
}

int64_t read_i64(const int8_t* row, const size_t offset) {
  return *reinterpret_cast<const int64_t*>(row + offset);
}

void write_f32(int8_t* row, const size_t offset, const float value) {
  std::memcpy(row + offset, &value, sizeof(value));
}

void write_f64(int8_t* row, const size_t offset, const double value) {
  std::memcpy(row + offset, &value, sizeof(value));
}

float read_f32(const int8_t* row, const size_t offset) {
  float value{0.0f};
  std::memcpy(&value, row + offset, sizeof(value));
  return value;
}

double read_f64(const int8_t* row, const size_t offset) {
  double value{0.0};
  std::memcpy(&value, row + offset, sizeof(value));
  return value;
}

void init_empty_8b_composite_key_rows(std::vector<int8_t>& rows,
                                      const size_t row_size,
                                      const size_t entry_count) {
  CHECK_EQ(rows.size(), row_size * entry_count);
  for (size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
    auto* row = rows.data() + entry_idx * row_size;
    write_i64(row, 0, EMPTY_KEY_64);
    write_i64(row, 8, EMPTY_KEY_64);
    write_i64(row, 16, 0);
    write_i64(row, 24, 0);
    write_i64(row, 32, kInt64NullSentinel);
    write_i64(row, 40, kInt64NullSentinel);
  }
}

void write_8b_composite_key_row(std::vector<int8_t>& rows,
                                const size_t row_size,
                                const size_t entry_idx,
                                const int64_t key0,
                                const int64_t key1,
                                const int64_t count,
                                const int64_t sum,
                                const int64_t min,
                                const int64_t max) {
  auto* row = rows.data() + entry_idx * row_size;
  write_i64(row, 0, key0);
  write_i64(row, 8, key1);
  write_i64(row, 16, count);
  write_i64(row, 24, sum);
  write_i64(row, 32, min);
  write_i64(row, 40, max);
}

struct Int64ReductionValues {
  int64_t count;
  int64_t sum;
  int64_t min;
  int64_t max;
};

std::map<std::pair<int64_t, int64_t>, Int64ReductionValues> collect_8b_composite_key_rows(
    const std::vector<int8_t>& rows,
    const size_t row_size,
    const size_t entry_count) {
  std::map<std::pair<int64_t, int64_t>, Int64ReductionValues> out;
  for (size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
    const auto* row = rows.data() + entry_idx * row_size;
    const auto key0 = read_i64(row, 0);
    if (key0 == EMPTY_KEY_64) {
      continue;
    }
    const auto key1 = read_i64(row, 8);
    out[{key0, key1}] = Int64ReductionValues{
        read_i64(row, 16), read_i64(row, 24), read_i64(row, 32), read_i64(row, 40)};
  }
  return out;
}

void expect_int64_values(const Int64ReductionValues& actual,
                         const Int64ReductionValues& expected) {
  EXPECT_EQ(actual.count, expected.count);
  EXPECT_EQ(actual.sum, expected.sum);
  EXPECT_EQ(actual.min, expected.min);
  EXPECT_EQ(actual.max, expected.max);
}

std::vector<DeviceBaselineHashReductionSlot> make_int64_reduction_slots() {
  return {
      DeviceBaselineHashReductionSlot{
          16, 0, sizeof(int64_t), DeviceBaselineHashReductionSlot::Sum, false, false},
      DeviceBaselineHashReductionSlot{
          24, 0, sizeof(int64_t), DeviceBaselineHashReductionSlot::Sum, false, false},
      DeviceBaselineHashReductionSlot{32,
                                      kInt64NullSentinel,
                                      sizeof(int64_t),
                                      DeviceBaselineHashReductionSlot::Min,
                                      true,
                                      false},
      DeviceBaselineHashReductionSlot{40,
                                      kInt64NullSentinel,
                                      sizeof(int64_t),
                                      DeviceBaselineHashReductionSlot::Max,
                                      true,
                                      false},
  };
}

void run_baseline_hash_reduction(
    std::vector<int8_t>& destination,
    const size_t destination_entry_count,
    const std::vector<int8_t>& source,
    const size_t source_entry_count,
    const size_t row_size,
    const size_t key_width,
    const size_t key_count,
    const std::vector<DeviceBaselineHashReductionSlot>& slots) {
  RuntimeDeviceBuffer destination_device(destination.size());
  RuntimeDeviceBuffer source_device(source.size());
  RuntimeDeviceBuffer slots_device(slots.size() *
                                   sizeof(DeviceBaselineHashReductionSlot));
  RuntimeDeviceBuffer reduction_scratch(sizeof(uint64_t));
  destination_device.copyFromHost(destination);
  source_device.copyFromHost(source);
  CHECK_EQ(cudaMemcpy(slots_device.ptr(),
                      slots.data(),
                      slots.size() * sizeof(DeviceBaselineHashReductionSlot),
                      cudaMemcpyHostToDevice),
           cudaSuccess);
  ASSERT_TRUE(reduce_baseline_hash_rows_on_device(
      destination_device.ptr(),
      destination_entry_count,
      source_device.ptr(),
      source_entry_count,
      row_size,
      key_width,
      key_count,
      reinterpret_cast<DeviceBaselineHashReductionSlot*>(slots_device.ptr()),
      slots.size(),
      reinterpret_cast<int*>(reduction_scratch.ptr()),
      kReductionTestDeviceId));
  destination_device.copyToHost(destination);
}

void init_empty_4b_key_rows(std::vector<int8_t>& rows,
                            const size_t row_size,
                            const size_t entry_count) {
  CHECK_EQ(rows.size(), row_size * entry_count);
  for (size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
    auto* row = rows.data() + entry_idx * row_size;
    write_i32(row, 0, EMPTY_KEY_32);
    write_i32(row, 4, 0);
    write_i32(row, 8, 0);
    write_i32(row, 12, 0);
    write_i64(row, 16, 0);
    write_i32(row, 24, kInt32NullSentinel);
    write_i32(row, 28, kInt32NullSentinel);
  }
}

void write_4b_key_row(std::vector<int8_t>& rows,
                      const size_t row_size,
                      const size_t entry_idx,
                      const int32_t key,
                      const int32_t count,
                      const int64_t sum,
                      const int32_t min,
                      const int32_t max) {
  auto* row = rows.data() + entry_idx * row_size;
  write_i32(row, 0, key);
  write_i32(row, 4, 0);
  write_i32(row, 8, count);
  write_i32(row, 12, 0);
  write_i64(row, 16, sum);
  write_i32(row, 24, min);
  write_i32(row, 28, max);
}

std::map<int32_t, Int64ReductionValues> collect_4b_key_rows(
    const std::vector<int8_t>& rows,
    const size_t row_size,
    const size_t entry_count) {
  std::map<int32_t, Int64ReductionValues> out;
  for (size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
    const auto* row = rows.data() + entry_idx * row_size;
    const auto key = read_i32(row, 0);
    if (key == EMPTY_KEY_32) {
      continue;
    }
    out[key] = Int64ReductionValues{
        read_i32(row, 8), read_i64(row, 16), read_i32(row, 24), read_i32(row, 28)};
  }
  return out;
}

std::vector<DeviceBaselineHashReductionSlot> make_mixed_width_reduction_slots() {
  return {
      DeviceBaselineHashReductionSlot{
          8, 0, sizeof(int32_t), DeviceBaselineHashReductionSlot::Sum, false, false},
      DeviceBaselineHashReductionSlot{
          16, 0, sizeof(int64_t), DeviceBaselineHashReductionSlot::Sum, false, false},
      DeviceBaselineHashReductionSlot{24,
                                      kInt32NullSentinel,
                                      sizeof(int32_t),
                                      DeviceBaselineHashReductionSlot::Min,
                                      true,
                                      false},
      DeviceBaselineHashReductionSlot{28,
                                      kInt32NullSentinel,
                                      sizeof(int32_t),
                                      DeviceBaselineHashReductionSlot::Max,
                                      true,
                                      false},
  };
}

void run_perfect_hash_reduction(
    std::vector<int8_t>& destination,
    const std::vector<int8_t>& source,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_width,
    const size_t key_count,
    const bool keyless,
    const size_t key_slot_offset,
    const size_t key_slot_width,
    const int64_t key_init_val,
    const std::vector<DeviceBaselineHashReductionSlot>& slots) {
  RuntimeDeviceBuffer destination_device(destination.size());
  RuntimeDeviceBuffer source_device(source.size());
  RuntimeDeviceBuffer slots_device(slots.size() *
                                   sizeof(DeviceBaselineHashReductionSlot));
  RuntimeDeviceBuffer reduction_scratch(sizeof(uint64_t));
  destination_device.copyFromHost(destination);
  source_device.copyFromHost(source);
  CHECK_EQ(cudaMemcpy(slots_device.ptr(),
                      slots.data(),
                      slots.size() * sizeof(DeviceBaselineHashReductionSlot),
                      cudaMemcpyHostToDevice),
           cudaSuccess);
  ASSERT_TRUE(reduce_perfect_hash_rows_on_device(
      destination_device.ptr(),
      source_device.ptr(),
      entry_count,
      row_size,
      key_width,
      key_count,
      keyless,
      key_slot_offset,
      key_slot_width,
      key_init_val,
      reinterpret_cast<DeviceBaselineHashReductionSlot*>(slots_device.ptr()),
      slots.size(),
      reinterpret_cast<int*>(reduction_scratch.ptr()),
      kReductionTestDeviceId));
  destination_device.copyToHost(destination);
}

void init_empty_keyless_rows(std::vector<int8_t>& rows,
                             const size_t row_size,
                             const size_t entry_count,
                             const int64_t key_init_val) {
  CHECK_EQ(rows.size(), row_size * entry_count);
  for (size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
    auto* row = rows.data() + entry_idx * row_size;
    write_i64(row, 0, key_init_val);
    write_i64(row, 8, 0);
    write_i64(row, 16, 0);
    write_i64(row, 24, kInt64NullSentinel);
    write_i64(row, 32, kInt64NullSentinel);
  }
}

void write_keyless_row(std::vector<int8_t>& rows,
                       const size_t row_size,
                       const size_t entry_idx,
                       const int64_t key_value,
                       const int64_t count,
                       const int64_t sum,
                       const int64_t min,
                       const int64_t max) {
  auto* row = rows.data() + entry_idx * row_size;
  write_i64(row, 0, key_value);
  write_i64(row, 8, count);
  write_i64(row, 16, sum);
  write_i64(row, 24, min);
  write_i64(row, 32, max);
}

std::vector<DeviceBaselineHashReductionSlot> make_keyless_reduction_slots() {
  return {
      DeviceBaselineHashReductionSlot{
          8, 0, sizeof(int64_t), DeviceBaselineHashReductionSlot::Sum, false, false},
      DeviceBaselineHashReductionSlot{
          16, 0, sizeof(int64_t), DeviceBaselineHashReductionSlot::Sum, false, false},
      DeviceBaselineHashReductionSlot{24,
                                      kInt64NullSentinel,
                                      sizeof(int64_t),
                                      DeviceBaselineHashReductionSlot::Min,
                                      true,
                                      false},
      DeviceBaselineHashReductionSlot{32,
                                      kInt64NullSentinel,
                                      sizeof(int64_t),
                                      DeviceBaselineHashReductionSlot::Max,
                                      true,
                                      false},
  };
}

}  // namespace

void GpuReductionTester::codegenWrapperKernel() {
  const unsigned address_space = 0;
  auto pi8_type = llvm::Type::getInt8PtrTy(context_, address_space);
  std::vector<llvm::Type*> input_arguments;
  input_arguments.push_back(llvm::PointerType::get(pi8_type, address_space));
  input_arguments.push_back(llvm::Type::getInt64Ty(context_));  // num input buffers
  input_arguments.push_back(llvm::Type::getInt8PtrTy(context_, address_space));

  llvm::FunctionType* ft =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context_), input_arguments, false);
  wrapper_kernel_ = llvm::Function::Create(
      ft, llvm::Function::ExternalLinkage, "wrapper_kernel", module_);

  auto arg_it = wrapper_kernel_->arg_begin();
  auto input_ptrs = &*arg_it;
  input_ptrs->setName("input_pointers");
  arg_it++;
  auto num_buffers = &*arg_it;
  num_buffers->setName("num_buffers");
  arg_it++;
  auto output_buffer = &*arg_it;
  output_buffer->setName("output_buffer");

  llvm::IRBuilder<> ir_builder(context_);

  auto bb_entry = llvm::BasicBlock::Create(context_, ".entry", wrapper_kernel_);
  auto bb_body = llvm::BasicBlock::Create(context_, ".body", wrapper_kernel_);
  auto bb_exit = llvm::BasicBlock::Create(context_, ".exit", wrapper_kernel_);

  // return if blockIdx.x > num_buffers
  ir_builder.SetInsertPoint(bb_entry);
  auto get_block_index_func = getFunction("get_block_index");
  auto block_index = ir_builder.CreateCall(get_block_index_func, {}, "block_index");
  const auto is_block_inbound =
      ir_builder.CreateICmpSLT(block_index, num_buffers, "is_block_inbound");
  ir_builder.CreateCondBr(is_block_inbound, bb_body, bb_exit);

  // locate the corresponding input buffer:
  ir_builder.SetInsertPoint(bb_body);
  auto input_buffer_gep = ir_builder.CreateGEP(
      input_ptrs->getType()->getScalarType()->getPointerElementType(),
      input_ptrs,
      block_index);
  auto input_buffer = ir_builder.CreateLoad(
      llvm::Type::getInt8PtrTy(context_, address_space), input_buffer_gep);
  auto input_buffer_ptr =
      ir_builder.CreatePointerCast(input_buffer,
                                   llvm::Type::getInt64PtrTy(context_, address_space),
                                   "input_buffer_ptr");
  const auto buffer_size = ll_int(
      static_cast<int32_t>(query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU)),
      context_);

  // initializing shared memory and copy input buffer into shared memory buffer:
  auto init_smem_func = getFunction("init_shared_mem");
  auto smem_input_buffer_ptr = ir_builder.CreateCall(init_smem_func,
                                                     {
                                                         input_buffer_ptr,
                                                         buffer_size,
                                                     },
                                                     "smem_input_buffer_ptr");

  auto output_buffer_ptr =
      ir_builder.CreatePointerCast(output_buffer,
                                   llvm::Type::getInt64PtrTy(context_, address_space),
                                   "output_buffer_ptr");
  // call the reduction function
  CHECK(reduction_func_);
  std::vector<llvm::Value*> reduction_args{
      output_buffer_ptr, smem_input_buffer_ptr, buffer_size};
  ir_builder.CreateCall(reduction_func_, reduction_args);
  ir_builder.CreateBr(bb_exit);

  ir_builder.SetInsertPoint(bb_exit);
  ir_builder.CreateRet(nullptr);
}

namespace {
void prepare_generated_gpu_kernel(llvm::Module* module,
                                  llvm::LLVMContext& context,
                                  llvm::Function* kernel) {
  // might be extra, remove and clean up
  module->setDataLayout(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  module->setTargetTriple("nvptx64-nvidia-cuda");

  llvm::NamedMDNode* md = module->getOrInsertNamedMetadata("nvvm.annotations");

  llvm::Metadata* md_vals[] = {llvm::ConstantAsMetadata::get(kernel),
                               llvm::MDString::get(context, "kernel"),
                               llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                                   llvm::Type::getInt32Ty(context), 1))};

  // Append metadata to nvvm.annotations
  md->addOperand(llvm::MDNode::get(context, md_vals));
}

std::unique_ptr<GpuDeviceCompilationContext> compile_and_link_gpu_code(
    const std::string& cuda_llir,
    llvm::Module* module,
    CudaMgr_Namespace::CudaMgr* cuda_mgr,
    const std::string& kernel_name,
    const size_t gpu_device_idx = 0) {
  CHECK(module);
  CHECK(cuda_mgr);
  auto& context = module->getContext();
  std::unique_ptr<llvm::TargetMachine> nvptx_target_machine =
      CodeGenerator::initializeNVPTXBackend(cuda_mgr->getDeviceArch());
  const auto ptx =
      CodeGenerator::generatePTX(cuda_llir, nvptx_target_machine.get(), context);

  CubinResult cubin_result = ptx_to_cubin(ptx, cuda_mgr, gpu_device_idx);
  auto gpu_context =
      std::make_unique<GpuDeviceCompilationContext>(cubin_result.cubin,
                                                    cubin_result.cubin_size,
                                                    kernel_name,
                                                    gpu_device_idx,
                                                    cuda_mgr,
                                                    cubin_result.option_keys.size(),
                                                    cubin_result.option_keys.data(),
                                                    cubin_result.option_values.data());

  checkCudaErrors(cuLinkDestroy(cubin_result.link_state));
  return gpu_context;
}

std::vector<std::unique_ptr<ResultSet>> create_and_fill_input_result_sets(
    const size_t num_input_buffers,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<TargetInfo>& target_infos,
    std::vector<StrideNumberGenerator>& generators,
    const std::vector<size_t>& steps) {
  std::vector<std::unique_ptr<ResultSet>> result_sets;
  for (size_t i = 0; i < num_input_buffers; i++) {
    result_sets.push_back(std::make_unique<ResultSet>(
        target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, 0, 0));
    const auto storage = result_sets.back()->allocateStorage();
    fill_storage_buffer(storage->getUnderlyingBuffer(),
                        target_infos,
                        query_mem_desc,
                        generators[i],
                        steps[i]);
  }
  return result_sets;
}

std::pair<std::unique_ptr<ResultSet>, std::unique_ptr<ResultSet>>
create_and_init_output_result_sets(std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   const std::vector<TargetInfo>& target_infos) {
  // CPU result set, will eventually host CPU reduciton results for validations
  auto cpu_result_set = std::make_unique<ResultSet>(
      target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, 0, 0);
  auto cpu_storage_result = cpu_result_set->allocateStorage();
  init_storage_buffer(
      cpu_storage_result->getUnderlyingBuffer(), target_infos, query_mem_desc);

  // GPU result set, will eventually host GPU reduction results
  auto gpu_result_set = std::make_unique<ResultSet>(
      target_infos, ExecutorDeviceType::GPU, query_mem_desc, row_set_mem_owner, 0, 0);
  auto gpu_storage_result = gpu_result_set->allocateStorage();
  init_storage_buffer(
      gpu_storage_result->getUnderlyingBuffer(), target_infos, query_mem_desc);
  return std::make_pair(std::move(cpu_result_set), std::move(gpu_result_set));
}
void perform_reduction_on_cpu(std::vector<std::unique_ptr<ResultSet>>& result_sets,
                              const ResultSetStorage* cpu_result_storage) {
  CHECK(result_sets.size() > 0);
  ResultSetReductionJIT reduction_jit(result_sets.front()->getQueryMemDesc(),
                                      result_sets.front()->getTargetInfos(),
                                      result_sets.front()->getTargetInitVals(),
                                      Executor::UNITARY_EXECUTOR_ID);
  const auto reduction_code = reduction_jit.codegen();
  for (auto& result_set : result_sets) {
    cpu_result_storage->reduce(
        *(result_set->getStorage()), {}, reduction_code, Executor::UNITARY_EXECUTOR_ID);
  }
}

struct TestInputData {
  size_t device_id;
  size_t num_input_buffers;
  std::vector<TargetInfo> target_infos;
  int8_t suggested_agg_widths;
  size_t min_entry;
  size_t max_entry;
  size_t step_size;
  bool keyless_hash;
  int32_t target_index_for_key;
  TestInputData()
      : device_id(0)
      , num_input_buffers(0)
      , suggested_agg_widths(0)
      , min_entry(0)
      , max_entry(0)
      , step_size(2)
      , keyless_hash(false)
      , target_index_for_key(0) {}
  TestInputData& setDeviceId(const size_t id) {
    device_id = id;
    return *this;
  }
  TestInputData& setNumInputBuffers(size_t num_buffers) {
    num_input_buffers = num_buffers;
    return *this;
  }
  TestInputData& setTargetInfos(std::vector<TargetInfo> tis) {
    target_infos = tis;
    return *this;
  }
  TestInputData& setAggWidth(int8_t agg_width) {
    suggested_agg_widths = agg_width;
    return *this;
  }
  TestInputData& setMinEntry(size_t min_e) {
    min_entry = min_e;
    return *this;
  }
  TestInputData& setMaxEntry(size_t max_e) {
    max_entry = max_e;
    return *this;
  }
  TestInputData& setKeylessHash(bool is_keyless) {
    keyless_hash = is_keyless;
    return *this;
  }
  TestInputData& setTargetIndexForKey(size_t target_idx) {
    target_index_for_key = target_idx;
    return *this;
  }
  TestInputData& setStepSize(size_t step) {
    step_size = step;
    return *this;
  }
};

void perform_test_and_verify_results(TestInputData input) {
  auto executor = Executor::getExecutor(0);
  auto& context = executor->getContext();
  auto cgen_state = std::unique_ptr<CgenState>(new CgenState({}, false));
  cgen_state->set_module_shallow_copy(executor->get_rt_module());
  auto module = cgen_state->module_;
  module->setDataLayout(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  module->setTargetTriple("nvptx64-nvidia-cuda");
  auto cuda_mgr = std::make_unique<CudaMgr_Namespace::CudaMgr>(1);
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>(
      Executor::getArenaBlockSize(), executor->getExecutorId());
  auto query_mem_desc = perfect_hash_one_col_desc(
      input.target_infos, input.suggested_agg_widths, input.min_entry, input.max_entry);
  if (input.keyless_hash) {
    query_mem_desc.setHasKeylessHash(true);
    query_mem_desc.setTargetIdxForKey(input.target_index_for_key);
  }

  std::vector<StrideNumberGenerator> generators(
      input.num_input_buffers, StrideNumberGenerator(1, input.step_size));
  std::vector<size_t> steps(input.num_input_buffers, input.step_size);
  auto input_result_sets = create_and_fill_input_result_sets(input.num_input_buffers,
                                                             row_set_mem_owner,
                                                             query_mem_desc,
                                                             input.target_infos,
                                                             generators,
                                                             steps);

  const auto [cpu_result_set, gpu_result_set] = create_and_init_output_result_sets(
      row_set_mem_owner, query_mem_desc, input.target_infos);

  // performing reduciton using the GPU reduction code:
  GpuReductionTester gpu_smem_tester(module,
                                     context,
                                     query_mem_desc,
                                     input.target_infos,
                                     init_agg_val_vec(input.target_infos, query_mem_desc),
                                     cuda_mgr.get());
  gpu_smem_tester.codegen();  // generate code for gpu reduciton and initialization
  gpu_smem_tester.codegenWrapperKernel();
  gpu_smem_tester.performReductionTest(
      input_result_sets, gpu_result_set->getStorage(), input.device_id);

  // CPU reduction for validation:
  perform_reduction_on_cpu(input_result_sets, cpu_result_set->getStorage());

  const auto cmp_result =
      std::memcmp(cpu_result_set->getStorage()->getUnderlyingBuffer(),
                  gpu_result_set->getStorage()->getUnderlyingBuffer(),
                  query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU));
  ASSERT_EQ(cmp_result, 0);
}

}  // namespace

void GpuReductionTester::performReductionTest(
    const std::vector<std::unique_ptr<ResultSet>>& result_sets,
    const ResultSetStorage* gpu_result_storage,
    const size_t device_id) {
  prepare_generated_gpu_kernel(module_, context_, getWrapperKernel());

  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  module_->print(os, nullptr);
  os.flush();
  std::string module_str(ss.str());

  std::unique_ptr<GpuDeviceCompilationContext> gpu_context(compile_and_link_gpu_code(
      module_str, module_, cuda_mgr_, getWrapperKernel()->getName().str()));

  const auto buffer_size = query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU);
  const size_t num_buffers = result_sets.size();
  std::vector<int8_t*> d_input_buffers;
  for (size_t i = 0; i < num_buffers; i++) {
    d_input_buffers.push_back(cuda_mgr_->allocateDeviceMem(buffer_size, device_id));
    cuda_mgr_->copyHostToDevice(d_input_buffers[i],
                                result_sets[i]->getStorage()->getUnderlyingBuffer(),
                                buffer_size,
                                device_id,
                                "");
  }

  constexpr size_t num_kernel_params = 3;
  CHECK_EQ(getWrapperKernel()->arg_size(), num_kernel_params);

  // parameter 1: an array of device pointers
  std::vector<CUdeviceptr> h_input_buffer_dptrs;
  h_input_buffer_dptrs.reserve(num_buffers);
  std::transform(d_input_buffers.begin(),
                 d_input_buffers.end(),
                 std::back_inserter(h_input_buffer_dptrs),
                 [](int8_t* dptr) { return reinterpret_cast<CUdeviceptr>(dptr); });

  auto d_input_buffer_dptrs =
      cuda_mgr_->allocateDeviceMem(num_buffers * sizeof(CUdeviceptr), device_id);
  cuda_mgr_->copyHostToDevice(d_input_buffer_dptrs,
                              reinterpret_cast<int8_t*>(h_input_buffer_dptrs.data()),
                              num_buffers * sizeof(CUdeviceptr),
                              device_id,
                              "");

  // parameter 2: number of buffers
  auto d_num_buffers = cuda_mgr_->allocateDeviceMem(sizeof(int64_t), device_id);
  cuda_mgr_->copyHostToDevice(d_num_buffers,
                              reinterpret_cast<const int8_t*>(&num_buffers),
                              sizeof(int64_t),
                              device_id,
                              "");

  // parameter 3: device pointer to the output buffer
  auto d_result_buffer = cuda_mgr_->allocateDeviceMem(buffer_size, device_id);
  cuda_mgr_->copyHostToDevice(d_result_buffer,
                              gpu_result_storage->getUnderlyingBuffer(),
                              buffer_size,
                              device_id,
                              "");

  // collecting all kernel parameters:
  std::vector<CUdeviceptr> h_kernel_params{
      reinterpret_cast<CUdeviceptr>(d_input_buffer_dptrs),
      reinterpret_cast<CUdeviceptr>(d_num_buffers),
      reinterpret_cast<CUdeviceptr>(d_result_buffer)};

  // casting each kernel parameter to be a void* device ptr itself:
  std::vector<void*> kernel_param_ptrs;
  kernel_param_ptrs.reserve(num_kernel_params);
  std::transform(h_kernel_params.begin(),
                 h_kernel_params.end(),
                 std::back_inserter(kernel_param_ptrs),
                 [](CUdeviceptr& param) { return &param; });

  // launching a kernel:
  auto cu_func = static_cast<CUfunction>(gpu_context->kernel());
  // we launch as many threadblocks as there are input buffers:
  // in other words, each input buffer is handled by a single threadblock.

  checkCudaErrors(cuLaunchKernel(cu_func,
                                 num_buffers,
                                 1,
                                 1,
                                 1024,
                                 1,
                                 1,
                                 buffer_size,
                                 0,
                                 kernel_param_ptrs.data(),
                                 nullptr));

  // transfer back the results:
  cuda_mgr_->copyDeviceToHost(
      gpu_result_storage->getUnderlyingBuffer(), d_result_buffer, buffer_size, "");

  // release the gpu memory used:
  for (auto& d_buffer : d_input_buffers) {
    cuda_mgr_->freeDeviceMem(d_buffer);
  }
  cuda_mgr_->freeDeviceMem(d_input_buffer_dptrs);
  cuda_mgr_->freeDeviceMem(d_num_buffers);
  cuda_mgr_->freeDeviceMem(d_result_buffer);
}

TEST(SingleColumn, VariableEntries_CountQuery_4B_Group) {
  for (auto num_entries : {1, 2, 3, 5, 13, 31, 63, 126, 241, 511, 1021}) {
    TestInputData input;
    input.setDeviceId(0)
        .setNumInputBuffers(4)
        .setTargetInfos(generate_custom_agg_target_infos({4}, {kCOUNT}, {kINT}, {kINT}))
        .setAggWidth(4)
        .setMinEntry(0)
        .setMaxEntry(num_entries)
        .setStepSize(2)
        .setKeylessHash(true)
        .setTargetIndexForKey(0);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableEntries_CountQuery_8B_Group) {
  for (auto num_entries : {1, 2, 3, 5, 13, 31, 63, 126, 241, 511, 1021}) {
    TestInputData input;
    input.setDeviceId(0)
        .setNumInputBuffers(4)
        .setTargetInfos(
            generate_custom_agg_target_infos({8}, {kCOUNT}, {kBIGINT}, {kBIGINT}))
        .setAggWidth(8)
        .setMinEntry(0)
        .setMaxEntry(num_entries)
        .setStepSize(2)
        .setKeylessHash(true)
        .setTargetIndexForKey(0);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_1) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(126)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(
          generate_custom_agg_target_infos({8},
                                           {kCOUNT, kMAX, kMIN, kSUM, kAVG},
                                           {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kDOUBLE},
                                           {kINT, kINT, kINT, kINT, kINT}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_2) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(126)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(
          generate_custom_agg_target_infos({8},
                                           {kCOUNT, kAVG, kMAX, kSUM, kMIN},
                                           {kBIGINT, kDOUBLE, kBIGINT, kBIGINT, kBIGINT},
                                           {kINT, kINT, kINT, kINT, kINT}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_3) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(367)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(
          generate_custom_agg_target_infos({8},
                                           {kCOUNT, kMAX, kAVG, kSUM, kMIN},
                                           {kBIGINT, kDOUBLE, kDOUBLE, kDOUBLE, kDOUBLE},
                                           {kINT, kDOUBLE, kDOUBLE, kDOUBLE, kDOUBLE}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_4) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(517)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(
          generate_custom_agg_target_infos({8},
                                           {kCOUNT, kSUM, kMAX, kAVG, kMIN},
                                           {kBIGINT, kFLOAT, kFLOAT, kFLOAT, kFLOAT},
                                           {kSMALLINT, kFLOAT, kFLOAT, kFLOAT, kFLOAT}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableNumBuffers) {
  TestInputData input;
  input.setDeviceId(0)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(266)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(generate_custom_agg_target_infos(
          {8},
          {kCOUNT, kSUM, kAVG, kMAX, kMIN},
          {kINT, kBIGINT, kDOUBLE, kFLOAT, kDOUBLE},
          {kTINYINT, kTINYINT, kSMALLINT, kFLOAT, kDOUBLE}));

  for (auto& num_buffers : {2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128}) {
    input.setNumInputBuffers(num_buffers);
    perform_test_and_verify_results(input);
  }
}

void run_peer_access_producer_event_test(const bool use_remote_kernel) {
  constexpr int destination_device_id = 0;
  constexpr int source_device_id = 1;
  constexpr size_t entry_count = size_t(1) << 20;
  constexpr size_t buffer_bytes = entry_count * sizeof(int64_t);

  if (g_cuda_mgr->getDeviceCount() <= source_device_id ||
      !g_cuda_mgr->canAccessPeer(destination_device_id, source_device_id)) {
    GTEST_SKIP() << "Two peer-accessible GPUs are required";
  }

  auto* source_buffer = g_cuda_mgr->allocateDeviceMem(buffer_bytes, source_device_id);
  ASSERT_NE(source_buffer, nullptr);
  auto* destination_buffer =
      g_cuda_mgr->allocateDeviceMem(buffer_bytes, destination_device_id);
  ASSERT_NE(destination_buffer, nullptr);
  int64_t* host_buffer{nullptr};
  CUstream source_stream{nullptr};
  CUstream destination_stream{nullptr};
  CUevent source_ready{nullptr};
  ScopeGuard cleanup = [&] {
    if (source_ready) {
      g_cuda_mgr->setContext(source_device_id);
      (void)cuEventDestroy(source_ready);
    }
    if (source_stream) {
      g_cuda_mgr->setContext(source_device_id);
      (void)cuStreamDestroy(source_stream);
    }
    if (destination_stream) {
      g_cuda_mgr->setContext(destination_device_id);
      (void)cuStreamDestroy(destination_stream);
    }
    if (host_buffer) {
      (void)cuMemFreeHost(host_buffer);
    }
    if (source_buffer) {
      g_cuda_mgr->freeDeviceMem(source_buffer);
    }
    if (destination_buffer) {
      g_cuda_mgr->freeDeviceMem(destination_buffer);
    }
  };

  ASSERT_EQ(CUDA_SUCCESS,
            cuMemAllocHost(reinterpret_cast<void**>(&host_buffer), buffer_bytes));
  g_cuda_mgr->setContext(source_device_id);
  ASSERT_EQ(CUDA_SUCCESS, cuStreamCreate(&source_stream, CU_STREAM_NON_BLOCKING));
  ASSERT_EQ(CUDA_SUCCESS, cuEventCreate(&source_ready, CU_EVENT_DISABLE_TIMING));
  g_cuda_mgr->setContext(destination_device_id);
  ASSERT_EQ(CUDA_SUCCESS, cuStreamCreate(&destination_stream, CU_STREAM_NON_BLOCKING));
  if (use_remote_kernel) {
    g_cuda_mgr->ensurePeerAccess(destination_device_id, source_device_id);
    const bool kernel_peer_access = g_cuda_mgr->canAccessPeerMemoryFromKernel(
        destination_device_id, source_device_id);
    const bool pointer_peer_access = g_cuda_mgr->ensurePeerAccessToDevicePtr(
        destination_device_id, source_device_id, source_buffer, buffer_bytes);
    ASSERT_EQ(kernel_peer_access, pointer_peer_access);
    if (!kernel_peer_access) {
      GTEST_SKIP() << "Peer DMA is available, but this topology cannot safely "
                      "dereference remote device memory from a kernel";
    }
  }

  constexpr size_t repetitions = 20;
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    const auto base = static_cast<int64_t>(repetition * entry_count);
    for (size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
      host_buffer[entry_idx] = base + static_cast<int64_t>(entry_idx);
    }

    g_cuda_mgr->setContext(source_device_id);
    ASSERT_EQ(CUDA_SUCCESS,
              cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(source_buffer),
                                host_buffer,
                                buffer_bytes,
                                source_stream));
    ASSERT_EQ(CUDA_SUCCESS, cuEventRecord(source_ready, source_stream));

    g_cuda_mgr->setContext(destination_device_id);
    ASSERT_EQ(CUDA_SUCCESS, cuStreamWaitEvent(destination_stream, source_ready, 0));
    if (use_remote_kernel) {
      CUcontext context_before_kernel{nullptr};
      CUcontext context_after_kernel{nullptr};
      ASSERT_EQ(CUDA_SUCCESS, cuCtxGetCurrent(&context_before_kernel));
      extract_fixed_width_column_from_rows_on_device(source_buffer,
                                                     destination_buffer,
                                                     entry_count,
                                                     sizeof(int64_t),
                                                     0,
                                                     sizeof(int64_t),
                                                     sizeof(int64_t),
                                                     -1,
                                                     std::numeric_limits<int64_t>::min(),
                                                     0,
                                                     destination_device_id,
                                                     destination_stream);
      ASSERT_EQ(CUDA_SUCCESS, cuCtxGetCurrent(&context_after_kernel));
      ASSERT_EQ(context_before_kernel, context_after_kernel);
    } else {
      g_cuda_mgr->copyPeerToPeer(destination_buffer,
                                 source_buffer,
                                 buffer_bytes,
                                 destination_device_id,
                                 source_device_id,
                                 "Peer DMA event-ordering verification",
                                 destination_stream,
                                 /*synchronize=*/false);
    }
    ASSERT_EQ(CUDA_SUCCESS, cuStreamSynchronize(destination_stream));
    g_cuda_mgr->setContext(destination_device_id);
    ASSERT_EQ(CUDA_SUCCESS,
              cuMemcpyDtoH(host_buffer,
                           reinterpret_cast<CUdeviceptr>(destination_buffer),
                           buffer_bytes));
    for (size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
      ASSERT_EQ(base + static_cast<int64_t>(entry_idx), host_buffer[entry_idx]);
    }

    if (use_remote_kernel) {
      DeviceColumnFragmentStats stats;
      ASSERT_TRUE(compute_columnar_fragment_int_stats_on_device(
          source_buffer,
          entry_count,
          sizeof(int64_t),
          std::numeric_limits<int64_t>::min(),
          destination_device_id,
          stats,
          destination_stream));
      ASSERT_TRUE(stats.has_values);
      ASSERT_FALSE(stats.has_nulls);
      ASSERT_EQ(base, stats.int_min);
      ASSERT_EQ(base + static_cast<int64_t>(entry_count) - 1, stats.int_max);
    }
  }
}

TEST(GpuPeerAccessRuntime, PeerDmaHonorsProducerEvent) {
  ASSERT_NO_FATAL_FAILURE(run_peer_access_producer_event_test(false));
}

TEST(GpuPeerAccessRuntime, RemoteKernelReadsHonorProducerEvent) {
  ASSERT_NO_FATAL_FAILURE(run_peer_access_producer_event_test(true));
}

TEST(GpuResultReductionRuntime, BaselineHashCompositeKeysAndNullSkipping) {
  constexpr size_t row_size = 48;
  constexpr size_t source_entry_count = 5;
  constexpr size_t destination_entry_count = 16;
  std::vector<int8_t> destination(row_size * destination_entry_count);
  std::vector<int8_t> source_a(row_size * source_entry_count);
  std::vector<int8_t> source_b(row_size * source_entry_count);
  init_empty_8b_composite_key_rows(destination, row_size, destination_entry_count);
  init_empty_8b_composite_key_rows(source_a, row_size, source_entry_count);
  init_empty_8b_composite_key_rows(source_b, row_size, source_entry_count);

  write_8b_composite_key_row(source_a, row_size, 0, 1, 10, 1, 5, 5, 5);
  write_8b_composite_key_row(source_a, row_size, 1, 2, 20, 1, 7, 7, 7);
  write_8b_composite_key_row(
      source_a, row_size, 3, 3, 30, 1, 6, kInt64NullSentinel, kInt64NullSentinel);
  write_8b_composite_key_row(source_b, row_size, 0, 1, 10, 2, 8, 3, 8);
  write_8b_composite_key_row(
      source_b, row_size, 1, 2, 20, 1, 4, kInt64NullSentinel, kInt64NullSentinel);
  write_8b_composite_key_row(source_b, row_size, 2, 4, 40, 1, 9, 9, 9);

  const auto slots = make_int64_reduction_slots();
  ASSERT_NO_FATAL_FAILURE(run_baseline_hash_reduction(destination,
                                                      destination_entry_count,
                                                      source_a,
                                                      source_entry_count,
                                                      row_size,
                                                      sizeof(int64_t),
                                                      2,
                                                      slots));
  ASSERT_NO_FATAL_FAILURE(run_baseline_hash_reduction(destination,
                                                      destination_entry_count,
                                                      source_b,
                                                      source_entry_count,
                                                      row_size,
                                                      sizeof(int64_t),
                                                      2,
                                                      slots));

  const auto rows =
      collect_8b_composite_key_rows(destination, row_size, destination_entry_count);
  ASSERT_EQ(rows.size(), size_t(4));
  ASSERT_NO_FATAL_FAILURE(
      expect_int64_values(rows.at({1, 10}), Int64ReductionValues{3, 13, 3, 8}));
  ASSERT_NO_FATAL_FAILURE(
      expect_int64_values(rows.at({2, 20}), Int64ReductionValues{2, 11, 7, 7}));
  ASSERT_NO_FATAL_FAILURE(expect_int64_values(
      rows.at({3, 30}),
      Int64ReductionValues{1, 6, kInt64NullSentinel, kInt64NullSentinel}));
  ASSERT_NO_FATAL_FAILURE(
      expect_int64_values(rows.at({4, 40}), Int64ReductionValues{1, 9, 9, 9}));
}

TEST(GpuResultReductionRuntime, BaselineHashInt32KeysAndMixedSlotWidths) {
  constexpr size_t row_size = 32;
  constexpr size_t source_entry_count = 5;
  constexpr size_t destination_entry_count = 16;
  std::vector<int8_t> destination(row_size * destination_entry_count);
  std::vector<int8_t> source_a(row_size * source_entry_count);
  std::vector<int8_t> source_b(row_size * source_entry_count);
  init_empty_4b_key_rows(destination, row_size, destination_entry_count);
  init_empty_4b_key_rows(source_a, row_size, source_entry_count);
  init_empty_4b_key_rows(source_b, row_size, source_entry_count);

  write_4b_key_row(source_a, row_size, 0, 7, 1, 10, 10, 10);
  write_4b_key_row(source_a, row_size, 1, 11, 1, 20, 20, 20);
  write_4b_key_row(source_b, row_size, 0, 7, 2, 30, 5, 35);
  write_4b_key_row(source_b, row_size, 2, 13, 1, 40, 40, 40);
  write_4b_key_row(
      source_b, row_size, 3, 11, 1, 5, kInt32NullSentinel, kInt32NullSentinel);

  const auto slots = make_mixed_width_reduction_slots();
  ASSERT_NO_FATAL_FAILURE(run_baseline_hash_reduction(destination,
                                                      destination_entry_count,
                                                      source_a,
                                                      source_entry_count,
                                                      row_size,
                                                      sizeof(int32_t),
                                                      1,
                                                      slots));
  ASSERT_NO_FATAL_FAILURE(run_baseline_hash_reduction(destination,
                                                      destination_entry_count,
                                                      source_b,
                                                      source_entry_count,
                                                      row_size,
                                                      sizeof(int32_t),
                                                      1,
                                                      slots));

  const auto rows = collect_4b_key_rows(destination, row_size, destination_entry_count);
  ASSERT_EQ(rows.size(), size_t(3));
  ASSERT_NO_FATAL_FAILURE(
      expect_int64_values(rows.at(7), Int64ReductionValues{3, 40, 5, 35}));
  ASSERT_NO_FATAL_FAILURE(
      expect_int64_values(rows.at(11), Int64ReductionValues{2, 25, 20, 20}));
  ASSERT_NO_FATAL_FAILURE(
      expect_int64_values(rows.at(13), Int64ReductionValues{1, 40, 40, 40}));
}

TEST(GpuResultReductionRuntime, NullableFloatingSumsPreserveNullSemantics) {
  constexpr size_t row_size = 24;
  constexpr size_t source_entry_count = 4;
  constexpr size_t destination_entry_count = 16;
  constexpr int32_t float_null_bits = 0x7fc00001;
  constexpr int64_t double_null_bits = 0x7ff8000000000001LL;

  const auto init_rows = [&](std::vector<int8_t>& rows, const size_t entry_count) {
    ASSERT_EQ(rows.size(), row_size * entry_count);
    for (size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
      auto* row = rows.data() + entry_idx * row_size;
      write_i64(row, 0, EMPTY_KEY_64);
      write_i32(row, 8, float_null_bits);
      write_i32(row, 12, 0);
      write_i64(row, 16, double_null_bits);
    }
  };
  const auto write_row = [&](std::vector<int8_t>& rows,
                             const size_t entry_idx,
                             const int64_t key,
                             const std::optional<float> float_sum,
                             const std::optional<double> double_sum) {
    auto* row = rows.data() + entry_idx * row_size;
    write_i64(row, 0, key);
    if (float_sum) {
      write_f32(row, 8, *float_sum);
    }
    if (double_sum) {
      write_f64(row, 16, *double_sum);
    }
  };
  const std::vector<DeviceBaselineHashReductionSlot> slots{
      DeviceBaselineHashReductionSlot{8,
                                      float_null_bits,
                                      sizeof(float),
                                      DeviceBaselineHashReductionSlot::Sum,
                                      true,
                                      true},
      DeviceBaselineHashReductionSlot{16,
                                      double_null_bits,
                                      sizeof(double),
                                      DeviceBaselineHashReductionSlot::Sum,
                                      true,
                                      true}};
  const auto verify_rows = [&](const std::vector<int8_t>& rows,
                               const size_t entry_count) {
    std::map<int64_t, const int8_t*> rows_by_key;
    for (size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
      const auto* row = rows.data() + entry_idx * row_size;
      const auto key = read_i64(row, 0);
      if (key != EMPTY_KEY_64) {
        ASSERT_TRUE(rows_by_key.emplace(key, row).second);
      }
    }
    ASSERT_EQ(rows_by_key.size(), size_t(3));
    EXPECT_FLOAT_EQ(read_f32(rows_by_key.at(7), 8), 4.0f);
    EXPECT_DOUBLE_EQ(read_f64(rows_by_key.at(7), 16), 7.0);
    EXPECT_FLOAT_EQ(read_f32(rows_by_key.at(11), 8), -3.0f);
    EXPECT_DOUBLE_EQ(read_f64(rows_by_key.at(11), 16), 8.0);
    EXPECT_EQ(read_i32(rows_by_key.at(13), 8), float_null_bits);
    EXPECT_EQ(read_i64(rows_by_key.at(13), 16), double_null_bits);
  };

  std::vector<int8_t> baseline_destination(row_size * destination_entry_count);
  std::vector<int8_t> source_a(row_size * source_entry_count);
  std::vector<int8_t> source_b(row_size * source_entry_count);
  init_rows(baseline_destination, destination_entry_count);
  init_rows(source_a, source_entry_count);
  init_rows(source_b, source_entry_count);
  write_row(source_a, 0, 7, 1.25f, 2.5);
  write_row(source_a, 1, 11, std::nullopt, std::nullopt);
  write_row(source_a, 2, 13, std::nullopt, std::nullopt);
  write_row(source_b, 0, 7, 2.75f, 4.5);
  write_row(source_b, 1, 11, -3.0f, 8.0);
  write_row(source_b, 2, 13, std::nullopt, std::nullopt);
  ASSERT_NO_FATAL_FAILURE(run_baseline_hash_reduction(baseline_destination,
                                                      destination_entry_count,
                                                      source_a,
                                                      source_entry_count,
                                                      row_size,
                                                      sizeof(int64_t),
                                                      1,
                                                      slots));
  ASSERT_NO_FATAL_FAILURE(run_baseline_hash_reduction(baseline_destination,
                                                      destination_entry_count,
                                                      source_b,
                                                      source_entry_count,
                                                      row_size,
                                                      sizeof(int64_t),
                                                      1,
                                                      slots));
  verify_rows(baseline_destination, destination_entry_count);

  std::vector<int8_t> perfect_destination(row_size * source_entry_count);
  init_rows(perfect_destination, source_entry_count);
  write_row(perfect_destination, 0, 7, 1.25f, 2.5);
  write_row(perfect_destination, 1, 11, std::nullopt, std::nullopt);
  write_row(perfect_destination, 2, 13, std::nullopt, std::nullopt);
  ASSERT_NO_FATAL_FAILURE(run_perfect_hash_reduction(perfect_destination,
                                                     source_b,
                                                     source_entry_count,
                                                     row_size,
                                                     sizeof(int64_t),
                                                     1,
                                                     false,
                                                     0,
                                                     0,
                                                     0,
                                                     slots));
  verify_rows(perfect_destination, source_entry_count);
}

TEST(GpuResultReductionRuntime, BaselineHashRejectsPublicationSentinelKeys) {
  constexpr size_t row_size = sizeof(int64_t);
  constexpr size_t destination_entry_count = 4;

  const auto rejects_key = [&](const size_t key_width) {
    std::vector<int8_t> destination(row_size * destination_entry_count, 0);
    std::vector<int8_t> source(row_size, 0);
    for (size_t entry_idx = 0; entry_idx < destination_entry_count; ++entry_idx) {
      auto* row = destination.data() + entry_idx * row_size;
      if (key_width == sizeof(int32_t)) {
        write_i32(row, 0, EMPTY_KEY_32);
      } else {
        write_i64(row, 0, EMPTY_KEY_64);
      }
    }
    if (key_width == sizeof(int32_t)) {
      write_i32(source.data(), 0, EMPTY_KEY_32 - 1);
    } else {
      write_i64(source.data(), 0, EMPTY_KEY_64 - 1);
    }

    RuntimeDeviceBuffer destination_device(destination.size());
    RuntimeDeviceBuffer source_device(source.size());
    RuntimeDeviceBuffer reduction_scratch(sizeof(uint64_t));
    destination_device.copyFromHost(destination);
    source_device.copyFromHost(source);
    return reduce_baseline_hash_rows_on_device(
        destination_device.ptr(),
        destination_entry_count,
        source_device.ptr(),
        1,
        row_size,
        key_width,
        1,
        nullptr,
        0,
        reinterpret_cast<int*>(reduction_scratch.ptr()),
        kReductionTestDeviceId);
  };

  EXPECT_FALSE(rejects_key(sizeof(int32_t)));
  EXPECT_FALSE(rejects_key(sizeof(int64_t)));
}

TEST(GpuResultReductionRuntime, BaselineHashPublishesRowsUnderContention) {
  constexpr size_t row_size = 2 * sizeof(int64_t);
  constexpr size_t source_entry_count = 64;
  constexpr size_t source_buffer_count = 128;
  constexpr size_t destination_entry_count = 256;
  constexpr size_t source_buffer_stride = source_entry_count * row_size;

  std::vector<int8_t> destination(destination_entry_count * row_size, 0);
  for (size_t entry_idx = 0; entry_idx < destination_entry_count; ++entry_idx) {
    write_i64(destination.data() + entry_idx * row_size, 0, EMPTY_KEY_64);
  }
  std::vector<int8_t> source(source_buffer_count * source_buffer_stride, 0);
  for (size_t buffer_idx = 0; buffer_idx < source_buffer_count; ++buffer_idx) {
    for (size_t entry_idx = 0; entry_idx < source_entry_count; ++entry_idx) {
      auto* row =
          source.data() + buffer_idx * source_buffer_stride + entry_idx * row_size;
      write_i64(row, 0, static_cast<int64_t>(entry_idx));
      write_i64(row, sizeof(int64_t), 1);
    }
  }

  const std::vector<DeviceBaselineHashReductionSlot> slots{
      DeviceBaselineHashReductionSlot{sizeof(int64_t),
                                      0,
                                      sizeof(int64_t),
                                      DeviceBaselineHashReductionSlot::Sum,
                                      false,
                                      false}};
  RuntimeDeviceBuffer destination_device(destination.size());
  RuntimeDeviceBuffer source_device(source.size());
  RuntimeDeviceBuffer slots_device(slots.size() *
                                   sizeof(DeviceBaselineHashReductionSlot));
  RuntimeDeviceBuffer reduction_scratch(sizeof(uint64_t));
  destination_device.copyFromHost(destination);
  source_device.copyFromHost(source);
  CHECK_EQ(cudaMemcpy(slots_device.ptr(),
                      slots.data(),
                      slots.size() * sizeof(DeviceBaselineHashReductionSlot),
                      cudaMemcpyHostToDevice),
           cudaSuccess);

  ASSERT_TRUE(reduce_baseline_hash_buffers_on_device(
      destination_device.ptr(),
      destination_entry_count,
      source_device.ptr(),
      source_entry_count,
      source_buffer_stride,
      source_buffer_count,
      row_size,
      sizeof(int64_t),
      1,
      reinterpret_cast<DeviceBaselineHashReductionSlot*>(slots_device.ptr()),
      slots.size(),
      reinterpret_cast<int*>(reduction_scratch.ptr()),
      kReductionTestDeviceId));
  destination_device.copyToHost(destination);

  std::map<int64_t, int64_t> sums_by_key;
  for (size_t entry_idx = 0; entry_idx < destination_entry_count; ++entry_idx) {
    const auto* row = destination.data() + entry_idx * row_size;
    const auto key = read_i64(row, 0);
    if (key != EMPTY_KEY_64) {
      sums_by_key.emplace(key, read_i64(row, sizeof(int64_t)));
    }
  }
  ASSERT_EQ(source_entry_count, sums_by_key.size());
  for (size_t key = 0; key < source_entry_count; ++key) {
    ASSERT_EQ(static_cast<int64_t>(source_buffer_count),
              sums_by_key.at(static_cast<int64_t>(key)));
  }
}

TEST(GpuResultReductionRuntime, PerfectHashRowsReduceAndCopyEmptyDestinations) {
  constexpr size_t row_size = 48;
  constexpr size_t entry_count = 5;
  std::vector<int8_t> destination(row_size * entry_count);
  std::vector<int8_t> source(row_size * entry_count);
  init_empty_8b_composite_key_rows(destination, row_size, entry_count);
  init_empty_8b_composite_key_rows(source, row_size, entry_count);

  write_8b_composite_key_row(destination, row_size, 1, 1, 10, 1, 10, 10, 10);
  write_8b_composite_key_row(destination, row_size, 3, 3, 30, 2, 8, 8, 8);
  write_8b_composite_key_row(source, row_size, 1, 1, 10, 4, 5, 5, 15);
  write_8b_composite_key_row(source, row_size, 2, 2, 20, 1, 7, 7, 7);
  write_8b_composite_key_row(
      source, row_size, 3, 3, 30, 3, 3, kInt64NullSentinel, kInt64NullSentinel);

  ASSERT_NO_FATAL_FAILURE(run_perfect_hash_reduction(destination,
                                                     source,
                                                     entry_count,
                                                     row_size,
                                                     sizeof(int64_t),
                                                     2,
                                                     false,
                                                     0,
                                                     0,
                                                     0,
                                                     make_int64_reduction_slots()));

  const auto rows = collect_8b_composite_key_rows(destination, row_size, entry_count);
  ASSERT_EQ(rows.size(), size_t(3));
  ASSERT_NO_FATAL_FAILURE(
      expect_int64_values(rows.at({1, 10}), Int64ReductionValues{5, 15, 5, 15}));
  ASSERT_NO_FATAL_FAILURE(
      expect_int64_values(rows.at({2, 20}), Int64ReductionValues{1, 7, 7, 7}));
  ASSERT_NO_FATAL_FAILURE(
      expect_int64_values(rows.at({3, 30}), Int64ReductionValues{5, 11, 8, 8}));
}

TEST(GpuResultReductionRuntime, PerfectHashKeylessRowsReduceAndCopyEmptyDestinations) {
  constexpr size_t row_size = 40;
  constexpr size_t entry_count = 5;
  constexpr int64_t key_init_val = -1;
  std::vector<int8_t> destination(row_size * entry_count);
  std::vector<int8_t> source(row_size * entry_count);
  init_empty_keyless_rows(destination, row_size, entry_count, key_init_val);
  init_empty_keyless_rows(source, row_size, entry_count, key_init_val);

  write_keyless_row(destination, row_size, 1, 1, 1, 10, 10, 10);
  write_keyless_row(destination, row_size, 3, 3, 2, 8, 8, 8);
  write_keyless_row(source, row_size, 1, 1, 4, 5, 5, 15);
  write_keyless_row(source, row_size, 2, 2, 1, 7, 7, 7);
  write_keyless_row(source, row_size, 3, 3, 3, 3, kInt64NullSentinel, kInt64NullSentinel);

  ASSERT_NO_FATAL_FAILURE(run_perfect_hash_reduction(destination,
                                                     source,
                                                     entry_count,
                                                     row_size,
                                                     sizeof(int64_t),
                                                     0,
                                                     true,
                                                     0,
                                                     sizeof(int64_t),
                                                     key_init_val,
                                                     make_keyless_reduction_slots()));

  EXPECT_EQ(read_i64(destination.data() + row_size, 0), 1);
  EXPECT_EQ(read_i64(destination.data() + row_size, 8), 5);
  EXPECT_EQ(read_i64(destination.data() + row_size, 16), 15);
  EXPECT_EQ(read_i64(destination.data() + row_size, 24), 5);
  EXPECT_EQ(read_i64(destination.data() + row_size, 32), 15);

  EXPECT_EQ(read_i64(destination.data() + 2 * row_size, 0), 2);
  EXPECT_EQ(read_i64(destination.data() + 2 * row_size, 8), 1);
  EXPECT_EQ(read_i64(destination.data() + 2 * row_size, 16), 7);
  EXPECT_EQ(read_i64(destination.data() + 2 * row_size, 24), 7);
  EXPECT_EQ(read_i64(destination.data() + 2 * row_size, 32), 7);

  EXPECT_EQ(read_i64(destination.data() + 3 * row_size, 0), 3);
  EXPECT_EQ(read_i64(destination.data() + 3 * row_size, 8), 5);
  EXPECT_EQ(read_i64(destination.data() + 3 * row_size, 16), 11);
  EXPECT_EQ(read_i64(destination.data() + 3 * row_size, 24), 8);
  EXPECT_EQ(read_i64(destination.data() + 3 * row_size, 32), 8);
}

int main(int argc, char** argv) {
  g_is_test_env = true;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  TestHelpers::init_sys_catalog();

  g_cuda_mgr.reset(new CudaMgr_Namespace::CudaMgr(0));
  g_query_engine = QueryEngine::createInstance(g_cuda_mgr.get(), /*cpu_only=*/false);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_query_engine.reset();
  g_cuda_mgr.reset(nullptr);

  return err;
}
