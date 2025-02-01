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

#pragma once

#include <iostream>

#include "CudaMgr/CudaMgr.h"
#include "QueryEngine/CompilationContext.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#include <ostream>
#else
#include "../Shared/nocuda.h"
#endif  // HAVE_CUDA
#include <string>
#include <vector>

#ifdef HAVE_CUDA
struct CudaErrorLog {
  CUresult cu_result;
};
inline std::ostream& operator<<(std::ostream& os, CudaErrorLog const cuda_error_log) {
  char const* error_name{nullptr};
  char const* error_string{nullptr};
  cuGetErrorName(cuda_error_log.cu_result, &error_name);
  cuGetErrorString(cuda_error_log.cu_result, &error_string);
  os << (error_name ? error_name : "Unknown CUDA error") << ' '
     << static_cast<int>(cuda_error_log.cu_result);
  if (error_string) {
    os << " (" << error_string << ')';
  }
  return os;
}

#define checkCudaErrors(ARG)                                                \
  if (CUresult const err = static_cast<CUresult>(ARG); err != CUDA_SUCCESS) \
    CHECK_EQ(err, CUDA_SUCCESS) << CudaErrorLog {                           \
      err                                                                   \
    }
#endif  // HAVE_CUDA

struct CubinResult {
  void* cubin;
  std::vector<CUjit_option> option_keys;
  std::vector<void*> option_values;
  CUlinkState link_state;
  size_t cubin_size;

  std::string info_log;
  std::string error_log;
  size_t jit_wall_time_idx;

  CubinResult();
  inline float jitWallTime() const {
    return *reinterpret_cast<float const*>(&option_values[jit_wall_time_idx]);
  }
};

/**
 * Loads the fatbin from disk which populates the nvcache. The fatbin load can take
 * several seconds, so we warmup the GPU JIT at server startup.
 */
void nvidia_jit_warmup();

/**
 * Compile and link PTX from the LLVM NVPTX backend with the CUDA runtime module and
 * device linker to create executable GPU device code.
 */
CubinResult ptx_to_cubin(const std::string& ptx,
                         const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                         int const device_id);

class GpuDeviceCompilationContext {
 public:
  GpuDeviceCompilationContext(const void* image,
                              const size_t module_size,
                              const std::string& kernel_name,
                              const int device_id,
                              const void* cuda_mgr,
                              unsigned int num_options,
                              CUjit_option* options,
                              void** option_vals);
  ~GpuDeviceCompilationContext();
  CUfunction kernel() { return kernel_; }
  CUmodule module() { return module_; }
  std::string const& name() const { return kernel_name_; }
  size_t getModuleSize() const { return module_size_; }
  int getDeviceId() const { return device_id_; }

 private:
  CUmodule module_;
  size_t module_size_;
  CUfunction kernel_;
  std::string const kernel_name_;
  const int device_id_;
#ifdef HAVE_CUDA
  const CudaMgr_Namespace::CudaMgr* cuda_mgr_;
#endif  // HAVE_CUDA
};

class GpuCompilationContext : public CompilationContext {
 public:
  GpuCompilationContext(CubinResult&& cubin_result, std::string const& function_name)
      : cubin_result_(std::move(cubin_result)), function_name_(function_name) {}

  ~GpuCompilationContext() {
#ifdef HAVE_CUDA
    if (!cu_link_state_destroyed_) {
      checkCudaErrors(cuLinkDestroy(cubin_result_.link_state));
    }
#endif
  }
#ifdef HAVE_CUDA
  void createGpuDeviceCompilationContextForDevices(
      std::set<int> const& device_ids,
      CudaMgr_Namespace::CudaMgr const* cuda_mgr) {
    for (auto device_id : device_ids) {
      auto it = contexts_per_device_.find(device_id);
      if (it == contexts_per_device_.end()) {
        auto device_context = std::make_unique<GpuDeviceCompilationContext>(
            cubin_result_.cubin,
            cubin_result_.cubin_size,
            function_name_,
            device_id,
            cuda_mgr,
            cubin_result_.option_keys.size(),
            cubin_result_.option_keys.data(),
            cubin_result_.option_values.data());
        contexts_per_device_.emplace(device_id, std::move(device_context));
        if (!cu_link_state_destroyed_ &&
            contexts_per_device_.size() ==
                static_cast<size_t>(cuda_mgr->getDeviceCount())) {
          // all GPUs have this module; we do not need to load the module anymore
          checkCudaErrors(cuLinkDestroy(cubin_result_.link_state));
          cu_link_state_destroyed_ = true;
        }
      }
    }
  }
#endif

  std::pair<void*, void*> getNativeCode(const size_t device_id) const {
    auto it = contexts_per_device_.find(device_id);
    CHECK(it != contexts_per_device_.end())
        << "Cannot find a compilation context for device " << device_id;
    auto device_context = it->second.get();
    return std::make_pair<void*, void*>(device_context->kernel(),
                                        device_context->module());
  }

  std::unordered_map<int, void*> getNativeFunctionPointers() const {
    std::unordered_map<int, void*> fn_ptrs;
    for (auto& kv : contexts_per_device_) {
      CHECK(kv.second);
      fn_ptrs.emplace(kv.first, kv.second->kernel());
    }
    return fn_ptrs;
  }

  std::string const& name(size_t const device_id) const {
    auto it = contexts_per_device_.find(device_id);
    CHECK(it != contexts_per_device_.end())
        << "Cannot find a compilation context for device " << device_id;
    return it->second->name();
  }

  size_t getMemSize() const {
    CHECK_GE(contexts_per_device_.size(), 1u);
    return contexts_per_device_.begin()->second.get()->getModuleSize();
  }

 private:
  CubinResult cubin_result_;
  std::string function_name_;
#ifdef HAVE_CUDA
  bool cu_link_state_destroyed_{false};
#endif
  std::unordered_map<int, std::unique_ptr<GpuDeviceCompilationContext>>
      contexts_per_device_;
};
