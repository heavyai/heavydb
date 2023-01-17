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

#include <sstream>

#include "NvidiaKernel.h"

#include "Logger/Logger.h"
#include "OSDependent/heavyai_path.h"

#include <boost/filesystem/operations.hpp>

#ifdef HAVE_CUDA
namespace {

#define JIT_LOG_SIZE 8192

void fill_options(std::vector<CUjit_option>& option_keys,
                  std::vector<void*>& option_values,
                  char* info_log,
                  char* error_log) {
  option_keys.push_back(CU_JIT_LOG_VERBOSE);
  option_values.push_back(reinterpret_cast<void*>(1));
  option_keys.push_back(CU_JIT_THREADS_PER_BLOCK);
  // fix the minimum # threads per block to the hardware-limit maximum num threads
  // to avoid recompiling jit module even if we manipulate it via query hint
  // (and allowed `CU_JIT_THREADS_PER_BLOCK` range is between 1 and 1024 by query hint)
  option_values.push_back(reinterpret_cast<void*>(1024));
  option_keys.push_back(CU_JIT_WALL_TIME);
  option_values.push_back(reinterpret_cast<void*>(0));
  option_keys.push_back(CU_JIT_INFO_LOG_BUFFER);
  option_values.push_back(reinterpret_cast<void*>(info_log));
  option_keys.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
  option_values.push_back(reinterpret_cast<void*>((long)JIT_LOG_SIZE));
  option_keys.push_back(CU_JIT_ERROR_LOG_BUFFER);
  option_values.push_back(reinterpret_cast<void*>(error_log));
  option_keys.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
  option_values.push_back(reinterpret_cast<void*>((long)JIT_LOG_SIZE));
}

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
  std::vector<CUjit_option> option_keys;
  std::vector<void*> option_values;
  char info_log[JIT_LOG_SIZE];
  char error_log[JIT_LOG_SIZE];
  fill_options(option_keys, option_values, info_log, error_log);
  CHECK_EQ(option_values.size(), option_keys.size());
  unsigned num_options = option_keys.size();
  CUlinkState link_state;
  checkCudaErrors(
      cuLinkCreate(num_options, &option_keys[0], &option_values[0], &link_state))
      << ": " << std::string(error_log);
  VLOG(1) << "CUDA JIT time to create link: "
          << *reinterpret_cast<float*>(&option_values[2]);
  boost::filesystem::path gpu_rt_path = get_gpu_rt_path();
  boost::filesystem::path cuda_table_functions_path = get_cuda_table_functions_path();
  CHECK(!gpu_rt_path.empty());
  CHECK(!cuda_table_functions_path.empty());
  checkCudaErrors(cuLinkAddFile(
      link_state, CU_JIT_INPUT_FATBINARY, gpu_rt_path.c_str(), 0, nullptr, nullptr))
      << ": " << std::string(error_log);
  VLOG(1) << "CUDA JIT time to add RT fatbinary: "
          << *reinterpret_cast<float*>(&option_values[2]);
  checkCudaErrors(cuLinkAddFile(link_state,
                                CU_JIT_INPUT_LIBRARY,
                                cuda_table_functions_path.c_str(),
                                0,
                                nullptr,
                                nullptr))
      << ": " << std::string(error_log);
  VLOG(1) << "CUDA JIT time to add GPU table functions library: "
          << *reinterpret_cast<float*>(&option_values[2]);
  checkCudaErrors(cuLinkDestroy(link_state)) << ": " << std::string(error_log);
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
  std::vector<CUjit_option> option_keys;
  std::vector<void*> option_values;
  char info_log[JIT_LOG_SIZE];
  char error_log[JIT_LOG_SIZE];
  fill_options(option_keys, option_values, info_log, error_log);
  CHECK_EQ(option_values.size(), option_keys.size());
  unsigned num_options = option_keys.size();
  CUlinkState link_state;
  checkCudaErrors(
      cuLinkCreate(num_options, &option_keys[0], &option_values[0], &link_state))
      << ": " << std::string(error_log);
  VLOG(1) << "CUDA JIT time to create link: "
          << *reinterpret_cast<float*>(&option_values[2]);

  boost::filesystem::path gpu_rt_path = get_gpu_rt_path();
  boost::filesystem::path cuda_table_functions_path = get_cuda_table_functions_path();
  CHECK(!gpu_rt_path.empty());
  CHECK(!cuda_table_functions_path.empty());
  // How to create a static CUDA library:
  // 1. nvcc -std=c++11 -arch=sm_35 --device-link -c [list of .cu files]
  // 2. nvcc -std=c++11 -arch=sm_35 -lib [list of .o files generated by step 1] -o
  // [library_name.a]
  checkCudaErrors(cuLinkAddFile(
      link_state, CU_JIT_INPUT_FATBINARY, gpu_rt_path.c_str(), 0, nullptr, nullptr))
      << ": " << std::string(error_log);
  VLOG(1) << "CUDA JIT time to add RT fatbinary: "
          << *reinterpret_cast<float*>(&option_values[2]);
  checkCudaErrors(cuLinkAddFile(link_state,
                                CU_JIT_INPUT_LIBRARY,
                                cuda_table_functions_path.c_str(),
                                0,
                                nullptr,
                                nullptr))
      << ": " << std::string(error_log);
  VLOG(1) << "CUDA JIT time to add GPU table functions library: "
          << *reinterpret_cast<float*>(&option_values[2]);
  checkCudaErrors(cuLinkAddData(link_state,
                                CU_JIT_INPUT_PTX,
                                static_cast<void*>(const_cast<char*>(ptx.c_str())),
                                ptx.length() + 1,
                                0,
                                0,
                                nullptr,
                                nullptr))
      << ": " << std::string(error_log) << "\nPTX:\n"
      << add_line_numbers(ptx) << "\nEOF PTX";
  VLOG(1) << "CUDA JIT time to add generated code: "
          << *reinterpret_cast<float*>(&option_values[2]);
  void* cubin{nullptr};
  size_t cubinSize{0};
  checkCudaErrors(cuLinkComplete(link_state, &cubin, &cubinSize))
      << ": " << std::string(error_log);
  VLOG(1) << "CUDA Linker completed: " << info_log;
  CHECK(cubin);
  CHECK_GT(cubinSize, size_t(0));
  VLOG(1) << "Generated GPU binary code size: " << cubinSize << " bytes";
  return {cubin, option_keys, option_values, link_state};
}
#endif

#ifdef HAVE_CUDA
GpuDeviceCompilationContext::GpuDeviceCompilationContext(const void* image,
                                                         const std::string& kernel_name,
                                                         const int device_id,
                                                         const void* cuda_mgr,
                                                         unsigned int num_options,
                                                         CUjit_option* options,
                                                         void** option_vals)
    : module_(nullptr)
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
