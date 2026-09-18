/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxInterop/Utils/CudaErrorCheck.h"

#ifdef HAVE_CUDA

#include "GfxDriver/DeviceContext.h"

namespace gfx {

void checkRenderCudaErrors(const CUresult result,
                           const DeviceId device_id,
                           const char* file_name,
                           const int line_no) {
  if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    throw gfx::OutOfGpuMemoryError("Cuda out of memory error on gpu: " +
                                   std::to_string(device_id));
  } else {
    if (result != CUDA_SUCCESS) {
      const char* errorString{nullptr};
      cuGetErrorString(result, &errorString);
      LOG(FATAL) << "CUDA error code=" << result << " (" << errorString
                 << "): " << file_name << ":" << line_no;
    }
  }
}

}  // namespace gfx

#endif  // HAVE_CUDA
