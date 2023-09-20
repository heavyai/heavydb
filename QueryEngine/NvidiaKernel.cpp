/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "NvidiaKernel.h"
#include "Logger/Logger.h"
#include "OSDependent/heavyai_path.h"

#include <boost/filesystem/operations.hpp>

#include <sstream>

#ifdef HAVE_CUDA

CubinResult::CubinResult()
    : cubin(nullptr), link_state(CUlinkState{}), cubin_size(0u), jit_wall_time_idx(0u) {
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
      {CU_JIT_WALL_TIME, nullptr},  // input not read, only output
      {CU_JIT_INFO_LOG_BUFFER, reinterpret_cast<void*>(&info_log[0])},
      {CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, reinterpret_cast<void*>(JIT_LOG_SIZE)},
      {CU_JIT_ERROR_LOG_BUFFER, reinterpret_cast<void*>(&error_log[0])},
      {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, reinterpret_cast<void*>(JIT_LOG_SIZE)}};
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
                         const CudaMgr_Namespace::CudaMgr* cuda_mgr) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(!ptx.empty());
  CHECK(cuda_mgr && cuda_mgr->getDeviceCount() > 0);
  cuda_mgr->setContext(0);
  CubinResult cubin_result{};
  CHECK_EQ(cubin_result.option_values.size(), cubin_result.option_keys.size());
  checkCudaErrors(cuLinkCreate(cubin_result.option_keys.size(),
                               cubin_result.option_keys.data(),
                               cubin_result.option_values.data(),
                               &cubin_result.link_state))
      << ": " << cubin_result.error_log.c_str();
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
  cuda_mgr_->loadGpuModuleData(
      &module_, image, num_options, options, option_vals, device_id_);
  CHECK(module_);
  checkCudaErrors(cuModuleGetFunction(&kernel_, module_, kernel_name_.c_str()));
}
#endif  // HAVE_CUDA

GpuDeviceCompilationContext::~GpuDeviceCompilationContext() {
#ifdef HAVE_CUDA
  CHECK(cuda_mgr_);
  cuda_mgr_->unloadGpuModuleData(&module_, device_id_);
#endif
}
