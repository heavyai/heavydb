/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef QUERYENGINE_NVIDIAKERNELLAUNCH_H
#define QUERYENGINE_NVIDIAKERNELLAUNCH_H

#include "../CudaMgr/CudaMgr.h"

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
};

CubinResult ptx_to_cubin(const std::string& ptx, const unsigned block_size, const CudaMgr_Namespace::CudaMgr* cuda_mgr);

class GpuCompilationContext {
 public:
  GpuCompilationContext(const void* image,
                        const std::string& kernel_name,
                        const int device_id,
                        const void* cuda_mgr,
                        unsigned int num_options,
                        CUjit_option* options,
                        void** option_vals);
  ~GpuCompilationContext();
  CUfunction kernel() { return kernel_; }
  CUmodule module() { return module_; }

 private:
  CUmodule module_;
  CUfunction kernel_;
#ifdef HAVE_CUDA
  const int device_id_;
  const void* cuda_mgr_;
#endif  // HAVE_CUDA
};

#define checkCudaErrors(err) CHECK_EQ(err, CUDA_SUCCESS);

#endif  // QUERYENGINE_NVIDIAKERNELLAUNCH_H
