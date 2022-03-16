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

#include "GpuSharedMemoryTest.h"
#include "QueryEngine/LLVMGlobalContext.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "QueryEngine/ResultSetReductionJIT.h"

extern bool g_is_test_env;

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
  auto input_buffer_gep = ir_builder.CreateGEP(input_ptrs, block_index);
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
    const size_t gpu_block_size = 1024,
    const size_t gpu_device_idx = 0) {
  CHECK(module);
  CHECK(cuda_mgr);
  auto& context = module->getContext();
  std::unique_ptr<llvm::TargetMachine> nvptx_target_machine =
      CodeGenerator::initializeNVPTXBackend(cuda_mgr->getDeviceArch());
  const auto ptx =
      CodeGenerator::generatePTX(cuda_llir, nvptx_target_machine.get(), context);

  auto cubin_result = ptx_to_cubin(ptx, gpu_block_size, cuda_mgr);
  auto& option_keys = cubin_result.option_keys;
  auto& option_values = cubin_result.option_values;
  auto cubin = cubin_result.cubin;
  auto link_state = cubin_result.link_state;
  const auto num_options = option_keys.size();
  auto gpu_context = std::make_unique<GpuDeviceCompilationContext>(cubin,
                                                                   kernel_name,
                                                                   gpu_device_idx,
                                                                   cuda_mgr,
                                                                   num_options,
                                                                   &option_keys[0],
                                                                   &option_values[0]);

  checkCudaErrors(cuLinkDestroy(link_state));
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
    result_sets.push_back(std::make_unique<ResultSet>(target_infos,
                                                      ExecutorDeviceType::CPU,
                                                      query_mem_desc,
                                                      row_set_mem_owner,
                                                      nullptr,
                                                      nullptr,
                                                      -1, /*fixme*/
                                                      0,
                                                      0));
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
  auto cpu_result_set = std::make_unique<ResultSet>(target_infos,
                                                    ExecutorDeviceType::CPU,
                                                    query_mem_desc,
                                                    row_set_mem_owner,
                                                    nullptr,
                                                    nullptr,
                                                    -1, /*fixme*/
                                                    0,
                                                    0);
  auto cpu_storage_result = cpu_result_set->allocateStorage();
  init_storage_buffer(
      cpu_storage_result->getUnderlyingBuffer(), target_infos, query_mem_desc);

  // GPU result set, will eventually host GPU reduction results
  auto gpu_result_set = std::make_unique<ResultSet>(target_infos,
                                                    ExecutorDeviceType::GPU,
                                                    query_mem_desc,
                                                    row_set_mem_owner,
                                                    nullptr,
                                                    nullptr,
                                                    -1, /*fixme*/
                                                    0,
                                                    0);
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
                                      result_sets.front()->getTargetInitVals());
  const auto reduction_code = reduction_jit.codegen();
  for (auto& result_set : result_sets) {
    cpu_result_storage->reduce(*(result_set->getStorage()), {}, reduction_code);
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
  auto cgen_state = std::unique_ptr<CgenState>(new CgenState({}, false));
  llvm::LLVMContext& context = cgen_state->context_;
  std::unique_ptr<llvm::Module> module(runtime_module_shallow_copy(cgen_state.get()));
  module->setDataLayout(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  module->setTargetTriple("nvptx64-nvidia-cuda");
  auto cuda_mgr = std::make_unique<CudaMgr_Namespace::CudaMgr>(1);
  const auto row_set_mem_owner =
      std::make_shared<RowSetMemoryOwner>(nullptr, Executor::getArenaBlockSize());
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
  GpuReductionTester gpu_smem_tester(module.get(),
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
                                device_id);
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
                              device_id);

  // parameter 2: number of buffers
  auto d_num_buffers = cuda_mgr_->allocateDeviceMem(sizeof(int64_t), device_id);
  cuda_mgr_->copyHostToDevice(d_num_buffers,
                              reinterpret_cast<const int8_t*>(&num_buffers),
                              sizeof(int64_t),
                              device_id);

  // parameter 3: device pointer to the output buffer
  auto d_result_buffer = cuda_mgr_->allocateDeviceMem(buffer_size, device_id);
  cuda_mgr_->copyHostToDevice(
      d_result_buffer, gpu_result_storage->getUnderlyingBuffer(), buffer_size, device_id);

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
      gpu_result_storage->getUnderlyingBuffer(), d_result_buffer, buffer_size, device_id);

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

int main(int argc, char** argv) {
  g_is_test_env = true;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
