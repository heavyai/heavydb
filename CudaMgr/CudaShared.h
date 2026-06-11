/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
