/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "NvidiaKernel.h"

#include <boost/filesystem/operations.hpp>

#include <cstdint>
#include <cstring>
#include <sstream>

#include "Logger/Logger.h"
#include "Shared/heavyai_path.h"

unsigned int g_cuda_jit_max_parallel_threads{1};

#ifdef HAVE_CUDA

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

void GpuCompilationContext::createGpuDeviceCompilationContextForDevices(
    std::set<int> const& device_ids,
    CudaMgr_Namespace::CudaMgr const* cuda_mgr,
    bool parallel_load) {
  std::vector<int> missing_device_ids;
  missing_device_ids.reserve(device_ids.size());
  for (const auto device_id : device_ids) {
    if (contexts_per_device_.find(device_id) == contexts_per_device_.end()) {
      missing_device_ids.push_back(device_id);
    }
  }

  std::vector<std::unique_ptr<GpuDeviceCompilationContext>> new_contexts(
      missing_device_ids.size());
  const auto load_context = [&](const size_t index) {
    // A persistent-cache hit contains a fully linked CUBIN. Loading it needs no JIT
    // options, whose output buffers belong to CubinResult and cannot be shared by
    // concurrent cuModuleLoadDataEx calls.
    const unsigned int num_options =
        parallel_load ? 0U : static_cast<unsigned int>(cubin_result_.option_keys.size());
    auto* option_keys = parallel_load ? nullptr : cubin_result_.option_keys.data();
    auto* option_values = parallel_load ? nullptr : cubin_result_.option_values.data();
    new_contexts[index] =
        std::make_unique<GpuDeviceCompilationContext>(cubin_result_.moduleImage(),
                                                      cubin_result_.moduleSize(),
                                                      function_name_,
                                                      missing_device_ids[index],
                                                      cuda_mgr,
                                                      num_options,
                                                      option_keys,
                                                      option_values);
  };

  if (parallel_load && missing_device_ids.size() > 1) {
    tbb::task_arena load_arena(static_cast<int>(missing_device_ids.size()));
    load_arena.execute([&] {
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, missing_device_ids.size(), size_t{1}),
          [&](const tbb::blocked_range<size_t>& range) {
            for (size_t index = range.begin(); index != range.end(); ++index) {
              load_context(index);
            }
          });
    });
  } else {
    for (size_t index = 0; index < missing_device_ids.size(); ++index) {
      load_context(index);
    }
  }

  for (size_t index = 0; index < missing_device_ids.size(); ++index) {
    contexts_per_device_.emplace(missing_device_ids[index],
                                 std::move(new_contexts[index]));
  }
  if (!cu_link_state_destroyed_ && cubin_result_.hasLinkState() &&
      contexts_per_device_.size() == static_cast<size_t>(cuda_mgr->getDeviceCount())) {
    // All GPUs have this module; we do not need to retain the linker state anymore.
    checkCudaErrors(cuLinkDestroy(cubin_result_.link_state));
    cu_link_state_destroyed_ = true;
  }
}

CubinResult::CubinResult()
    : cubin(nullptr)
    , link_state(CUlinkState{})
    , cubin_size(0u)
    , link_state_valid(false)
    , jit_wall_time_idx(0u) {
  constexpr size_t JIT_LOG_SIZE = 8192u;
  static_assert(0u < JIT_LOG_SIZE);
  info_log.resize(JIT_LOG_SIZE - 1u);  // minus 1 for null terminator
  error_log.resize(JIT_LOG_SIZE - 1u);
  std::pair<CUjit_option, void*> options[] = {
    {CU_JIT_LOG_VERBOSE, reinterpret_cast<void*>(1)},
    // fix the minimum # threads per block to the hardware-limit maximum num threads to
    // avoid recompiling jit module even if we manipulate it via query hint (and allowed
    // `CU_JIT_THREADS_PER_BLOCK` range is between 1 and 1024 by query hint)
    {CU_JIT_THREADS_PER_BLOCK, reinterpret_cast<void*>(1024)},
#if CUDA_VERSION >= 13000
    {CU_JIT_SPLIT_COMPILE,
     reinterpret_cast<void*>(static_cast<uintptr_t>(g_cuda_jit_max_parallel_threads))},
#endif
    {CU_JIT_WALL_TIME, nullptr},  // input not read, only output
    {CU_JIT_INFO_LOG_BUFFER, reinterpret_cast<void*>(&info_log[0])},
    {CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, reinterpret_cast<void*>(JIT_LOG_SIZE)},
    {CU_JIT_ERROR_LOG_BUFFER, reinterpret_cast<void*>(&error_log[0])},
    {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, reinterpret_cast<void*>(JIT_LOG_SIZE)}
  };
  constexpr size_t n_options = sizeof(options) / sizeof(*options);
  option_keys.reserve(n_options);
  option_values.reserve(n_options);
  for (size_t i = 0; i < n_options; ++i) {
    option_keys.push_back(options[i].first);
    option_values.push_back(options[i].second);
    if (options[i].first == CU_JIT_WALL_TIME) {
      jit_wall_time_idx = i;
    }
  }
  CHECK_EQ(CU_JIT_WALL_TIME, option_keys[jit_wall_time_idx]) << jit_wall_time_idx;
}

CubinResult CubinResult::fromCubinBytes(std::vector<int8_t> bytes) {
  CubinResult result;
  result.cubin_storage = std::move(bytes);
  result.cubin = result.cubin_storage.data();
  result.cubin_size = result.cubin_storage.size();
  result.link_state_valid = false;
  return result;
}

const void* CubinResult::moduleImage() const {
  return cubin_storage.empty() ? cubin : cubin_storage.data();
}

size_t CubinResult::moduleSize() const {
  return cubin_storage.empty() ? cubin_size : cubin_storage.size();
}

namespace {

boost::filesystem::path get_gpu_rt_path() {
  boost::filesystem::path gpu_rt_path{heavyai::get_root_abs_path()};
  gpu_rt_path /= "QueryEngine";
  gpu_rt_path /= "cuda_mapd_rt.fatbin";
  if (!boost::filesystem::exists(gpu_rt_path)) {
    throw std::runtime_error("HeavyDB GPU runtime library not found at " +
                             gpu_rt_path.string());
  }
  return gpu_rt_path;
}

boost::filesystem::path get_cuda_table_functions_path() {
  boost::filesystem::path cuda_table_functions_path{heavyai::get_root_abs_path()};
  cuda_table_functions_path /= "QueryEngine";
  cuda_table_functions_path /= "CudaTableFunctions.a";
  if (!boost::filesystem::exists(cuda_table_functions_path)) {
    throw std::runtime_error("HeavyDB GPU table functions module not found at " +
                             cuda_table_functions_path.string());
  }

  return cuda_table_functions_path;
}

}  // namespace

void nvidia_jit_warmup() {
  CubinResult cubin_result{};
  CHECK_EQ(cubin_result.option_values.size(), cubin_result.option_keys.size());
  unsigned const num_options = cubin_result.option_keys.size();
  checkCudaErrors(cuLinkCreate(num_options,
                               cubin_result.option_keys.data(),
                               cubin_result.option_values.data(),
                               &cubin_result.link_state))
      << ": " << cubin_result.error_log.c_str();
  cubin_result.link_state_valid = true;
  VLOG(1) << "CUDA JIT time to create link: " << cubin_result.jitWallTime();
  boost::filesystem::path gpu_rt_path = get_gpu_rt_path();
  boost::filesystem::path cuda_table_functions_path = get_cuda_table_functions_path();
  CHECK(!gpu_rt_path.empty());
  CHECK(!cuda_table_functions_path.empty());
  checkCudaErrors(cuLinkAddFile(cubin_result.link_state,
                                CU_JIT_INPUT_FATBINARY,
                                gpu_rt_path.c_str(),
                                0,
                                nullptr,
                                nullptr))
      << ": " << cubin_result.error_log.c_str();
  VLOG(1) << "CUDA JIT time to add RT fatbinary: " << cubin_result.jitWallTime();
  checkCudaErrors(cuLinkAddFile(cubin_result.link_state,
                                CU_JIT_INPUT_LIBRARY,
                                cuda_table_functions_path.c_str(),
                                0,
                                nullptr,
                                nullptr))
      << ": " << cubin_result.error_log.c_str();
  VLOG(1) << "CUDA JIT time to add GPU table functions library: "
          << cubin_result.jitWallTime();
  checkCudaErrors(cuLinkDestroy(cubin_result.link_state))
      << ": " << cubin_result.error_log.c_str();
}

std::string add_line_numbers(const std::string& text) {
  std::stringstream iss(text);
  std::string result;
  size_t count = 1;
  while (iss.good()) {
    std::string line;
    std::getline(iss, line, '\n');
    result += std::to_string(count) + ": " + line + "\n";
    count++;
  }
  return result;
}

CubinResult ptx_to_cubin(const std::string& ptx,
                         const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                         int const device_id) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(!ptx.empty());
  CHECK(cuda_mgr && cuda_mgr->getDeviceCount() > 0);
  cuda_mgr->setContext(device_id);
  CubinResult cubin_result{};
  CHECK_EQ(cubin_result.option_values.size(), cubin_result.option_keys.size());
  checkCudaErrors(cuLinkCreate(cubin_result.option_keys.size(),
                               cubin_result.option_keys.data(),
                               cubin_result.option_values.data(),
                               &cubin_result.link_state))
      << ": " << cubin_result.error_log.c_str();
  cubin_result.link_state_valid = true;
  VLOG(1) << "CUDA JIT time to create link: " << cubin_result.jitWallTime();

  boost::filesystem::path gpu_rt_path = get_gpu_rt_path();
  boost::filesystem::path cuda_table_functions_path = get_cuda_table_functions_path();
  CHECK(!gpu_rt_path.empty());
  CHECK(!cuda_table_functions_path.empty());
  // How to create a static CUDA library:
  // 1. nvcc -std=c++11 -arch=sm_35 --device-link -c [list of .cu files]
  // 2. nvcc -std=c++11 -arch=sm_35 -lib [list of .o files generated by step 1] -o
  // [library_name.a]
  checkCudaErrors(cuLinkAddFile(cubin_result.link_state,
                                CU_JIT_INPUT_FATBINARY,
                                gpu_rt_path.c_str(),
                                0,
                                nullptr,
                                nullptr))
      << ": " << cubin_result.error_log.c_str();
  VLOG(1) << "CUDA JIT time to add RT fatbinary: " << cubin_result.jitWallTime();
  checkCudaErrors(cuLinkAddFile(cubin_result.link_state,
                                CU_JIT_INPUT_LIBRARY,
                                cuda_table_functions_path.c_str(),
                                0,
                                nullptr,
                                nullptr))
      << ": " << cubin_result.error_log.c_str();
  VLOG(1) << "CUDA JIT time to add GPU table functions library: "
          << cubin_result.jitWallTime();
  // The ptx.length() + 1 follows the example in
  // https://developer.nvidia.com/blog/discovering-new-features-in-cuda-11-4/
  checkCudaErrors(cuLinkAddData(cubin_result.link_state,
                                CU_JIT_INPUT_PTX,
                                static_cast<void*>(const_cast<char*>(ptx.c_str())),
                                ptx.length() + 1,
                                0,
                                0,
                                nullptr,
                                nullptr))
      << ": " << cubin_result.error_log.c_str() << "\nPTX:\n"
      << add_line_numbers(ptx) << "\nEOF PTX";
  VLOG(1) << "CUDA JIT time to add generated code: " << cubin_result.jitWallTime();
  checkCudaErrors(cuLinkComplete(
      cubin_result.link_state, &cubin_result.cubin, &cubin_result.cubin_size))
      << ": " << cubin_result.error_log.c_str();
  VLOG(1) << "CUDA Linker completed: " << cubin_result.info_log.c_str();
  CHECK(cubin_result.cubin);
  CHECK_LT(0u, cubin_result.cubin_size);
  const auto cubin_bytes = static_cast<const int8_t*>(cubin_result.cubin);
  cubin_result.cubin_storage.assign(cubin_bytes, cubin_bytes + cubin_result.cubin_size);
  cubin_result.cubin = cubin_result.cubin_storage.data();
  VLOG(1) << "Generated GPU binary code size: " << cubin_result.cubin_size << " bytes";
  return cubin_result;
}

GpuDeviceCompilationContext::GpuDeviceCompilationContext(const void* image,
                                                         const size_t module_size,
                                                         const std::string& kernel_name,
                                                         const int device_id,
                                                         const void* cuda_mgr,
                                                         unsigned int num_options,
                                                         CUjit_option* options,
                                                         void** option_vals)
    : module_(nullptr)
    , module_size_(module_size)
    , kernel_(nullptr)
    , kernel_name_(kernel_name)
    , device_id_(device_id)
    , cuda_mgr_(static_cast<const CudaMgr_Namespace::CudaMgr*>(cuda_mgr)) {
  LOG_IF(FATAL, cuda_mgr_ == nullptr)
      << "Unable to initialize GPU compilation context without CUDA manager";
  auto timer = timer_start();
  cuda_mgr_->loadGpuModuleData(
      &module_, image, num_options, options, option_vals, device_id_);
  CHECK(module_);
  auto const load_module_ms = timer_stop(timer);
  timer = timer_start();
  const auto function_status =
      cuModuleGetFunction(&kernel_, module_, kernel_name_.c_str());
  if (function_status != CUDA_SUCCESS) {
    cuda_mgr_->unloadGpuModuleData(&module_, device_id_);
    module_ = nullptr;
    CudaMgr_Namespace::check_error(function_status);
  }
  auto const get_module_ms = timer_stop(timer);
  VLOG(1) << "GPU device compilation context initialized, device: " << device_id_
          << ", load module: " << load_module_ms << "ms, init. kernel: " << get_module_ms
          << "ms";
}
#endif  // HAVE_CUDA

GpuDeviceCompilationContext::~GpuDeviceCompilationContext() {
#ifdef HAVE_CUDA
  CHECK(cuda_mgr_);
  cuda_mgr_->unloadGpuModuleData(&module_, device_id_);
#endif
}
