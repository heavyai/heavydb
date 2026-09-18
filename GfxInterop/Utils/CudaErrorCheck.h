/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef HAVE_CUDA
#include <cuda.h>

#include "GfxDriver/Types.h"

namespace gfx {

void checkRenderCudaErrors(const CUresult result,
                           const DeviceId device_id,
                           const char* file_name,
                           const int line_no);

}  // namespace gfx

#define CHECK_RENDER_CUDA_ERRORS(result, device_id) \
  ::gfx::checkRenderCudaErrors(result, device_id, __FILE__, __LINE__);

#define CHECK_RENDER_CUDA_ERRORS_MEMCHECK(result, device_id, out_of_mem_err_str)     \
  try {                                                                              \
    ::gfx::checkRenderCudaErrors(result, device_id, __FILE__, __LINE__);             \
  } catch (gfx::OutOfGpuMemoryError & err) {                                         \
    LOG(ERROR) << err.what() << " during " << out_of_mem_err_str << ". " << __FILE__ \
               << ":" << __LINE__;                                                   \
    throw err;                                                                       \
  }

#endif  // HAVE_CUDA
