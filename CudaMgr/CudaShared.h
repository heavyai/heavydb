/*
 * Copyright 2025 HEAVY.AI, Inc.
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

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include "Shared/nocuda.h"
#endif  // HAVE_CUDA

#include "Logger/Logger.h"
#include "Shared/boost_stacktrace.hpp"

namespace CudaMgr_Namespace {
inline std::string error_message(CUresult const status) {
  const char* error_string{nullptr};
#ifdef HAVE_CUDA
  cuGetErrorString(status, &error_string);
#endif
  return error_string
             ? "CUDA Error (" + std::to_string(status) + "): " + std::string(error_string)
             : "CUDA Driver API error code " + std::to_string(status);
}

class CudaErrorException : public std::runtime_error {
 public:
  CudaErrorException(CUresult status)
      : std::runtime_error(error_message(status)), status_(status) {
#ifdef HAVE_CUDA
    // cuda already de-initialized can occur during system shutdown. avoid making calls to
    // the logger to prevent failing during a standard teardown.
    if (status != CUDA_ERROR_DEINITIALIZED) {
      VLOG(1) << error_message(status);
      VLOG(1) << boost::stacktrace::stacktrace();
    }
#endif
  }

  CUresult getStatus() const {
    return status_;
  }

 private:
  CUresult const status_;
};

inline void check_error(CUresult status) {
#ifdef HAVE_CUDA
  if (status != CUDA_SUCCESS) {
    throw CudaErrorException(status);
  }
#endif
}

inline void set_context(const std::vector<CUcontext>& device_contexts,
                        int32_t device_num) {
#ifdef HAVE_CUDA
  CHECK_LT(size_t(device_num), device_contexts.size());
  cuCtxSetCurrent(device_contexts[device_num]);
#endif
}
}  // namespace CudaMgr_Namespace
