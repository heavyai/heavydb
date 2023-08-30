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

#include "CudaMgr/CudaMgr.h"
#include "QueryEngine/CompilationContext.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include "../Shared/nocuda.h"
#endif  // HAVE_CUDA
#include <string>
#include <vector>

struct CubinResult {
  void* cubin;
  std::vector<CUjit_option> option_keys;
  std::vector<void*> option_values;
  CUlinkState link_state;
  size_t cubin_size;
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
                         const CudaMgr_Namespace::CudaMgr* cuda_mgr);

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

 private:
  CUmodule module_;
  size_t module_size_;
  CUfunction kernel_;
  std::string const kernel_name_;
#ifdef HAVE_CUDA
  const int device_id_;
  const CudaMgr_Namespace::CudaMgr* cuda_mgr_;
#endif  // HAVE_CUDA
};

class GpuCompilationContext : public CompilationContext {
 public:
  GpuCompilationContext() {}

  void addDeviceCode(std::unique_ptr<GpuDeviceCompilationContext>&& device_context) {
    contexts_per_device_.push_back(std::move(device_context));
  }

  std::pair<void*, void*> getNativeCode(const size_t device_id) const {
    CHECK_LT(device_id, contexts_per_device_.size());
    auto device_context = contexts_per_device_[device_id].get();
    return std::make_pair<void*, void*>(device_context->kernel(),
                                        device_context->module());
  }

  std::vector<void*> getNativeFunctionPointers() const {
    std::vector<void*> fn_ptrs;
    for (auto& device_context : contexts_per_device_) {
      CHECK(device_context);
      fn_ptrs.push_back(device_context->kernel());
    }
    return fn_ptrs;
  }

  std::string const& name(size_t const device_id) const {
    CHECK_LT(device_id, contexts_per_device_.size());
    return contexts_per_device_[device_id]->name();
  }

  size_t getMemSize() const {
    return contexts_per_device_.begin()->get()->getModuleSize();
  }

 private:
  std::vector<std::unique_ptr<GpuDeviceCompilationContext>> contexts_per_device_;
};

#ifdef HAVE_CUDA
inline std::string ourCudaErrorStringHelper(CUresult error) {
  char const* c1;
  CUresult res1 = cuGetErrorName(error, &c1);
  char const* c2;
  CUresult res2 = cuGetErrorString(error, &c2);
  std::string text;
  if (res1 == CUDA_SUCCESS) {
    text += c1;
    text += " (";
    text += std::to_string(error);
    text += ")";
  }
  if (res2 == CUDA_SUCCESS) {
    if (!text.empty()) {
      text += ": ";
    }
    text += c2;
  }
  if (text.empty()) {
    text = std::to_string(error);  // never return an empty error string
  }
  return text;
}

#define checkCudaErrors(ARG)                                                \
  if (CUresult const err = static_cast<CUresult>(ARG); err != CUDA_SUCCESS) \
  CHECK_EQ(err, CUDA_SUCCESS) << ourCudaErrorStringHelper(err)
#endif  // HAVE_CUDA
