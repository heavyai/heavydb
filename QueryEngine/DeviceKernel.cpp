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

#include "DeviceKernel.h"

#include "CompilationContext.h"
#include "NvidiaKernel.h"
extern bool g_enable_gpu_dynamic_smem;

#ifdef HAVE_CUDA
void NvidiaKernel::launch(unsigned int grid_dim_x,
                          unsigned int grid_dim_y,
                          unsigned int grid_dim_z,
                          unsigned int block_dim_x,
                          unsigned int block_dim_y,
                          unsigned int block_dim_z,
                          CompilationResult const& compilation_result,
                          void** kernel_params,
                          KernelParamsLog const& kernel_params_log,
                          bool optimize_block_and_grid_sizes) {
  size_t const shared_memory_size =
      compilation_result.gpu_smem_context.getSharedMemorySize();
  if (g_enable_gpu_dynamic_smem) {
    auto returned_status =
        cuFuncSetAttribute(function_ptr_,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_memory_size);
    if (returned_status == CUDA_SUCCESS) {
      VLOG(1) << "Set max dynamic shared size to " << shared_memory_size << " bytes";
    } else {
      VLOG(1) << "Failed to set max dynamic shared size to " << shared_memory_size
              << " bytes (Error code: " << returned_status << ")";
    }
  }
  if (optimize_block_and_grid_sizes) {
    int recommended_block_size;
    int recommended_grid_size;
    checkCudaErrors(cuOccupancyMaxPotentialBlockSize(&recommended_grid_size,
                                                     &recommended_block_size,
                                                     function_ptr_,
                                                     nullptr,
                                                     shared_memory_size,
                                                     0));
    if (static_cast<unsigned int>(recommended_block_size) != block_dim_x) {
      VLOG(1) << "Apply a recommended CUDA block size: " << recommended_block_size
              << " (current: " << block_dim_x << ")";
      block_dim_x = recommended_block_size;
    }
    if (static_cast<unsigned int>(recommended_grid_size) != grid_dim_x) {
      VLOG(1) << "Apply a recommended CUDA grid size: " << recommended_grid_size
              << " (current: " << grid_dim_x << ")";
      grid_dim_x = recommended_grid_size;
    }
  }

  VLOG(1) << "Launch GPU kernel on device " << device_id_
          << " compiled with the following grid and block sizes: " << grid_dim_x
          << " and " << block_dim_x
          << " (shared memory used per CUDA block: " << shared_memory_size << " bytes)";
  checkCudaErrors(cuLaunchKernel(function_ptr_,
                                 grid_dim_x,
                                 grid_dim_y,
                                 grid_dim_z,
                                 block_dim_x,
                                 block_dim_y,
                                 block_dim_z,
                                 shared_memory_size,
                                 cuda_stream_,
                                 kernel_params,
                                 nullptr));
  CUresult const cu_result = cuStreamSynchronize(cuda_stream_);
  LOG_IF(FATAL, cu_result != CUDA_SUCCESS)
      << DeviceKernelLaunchErrorLogDump{cu_result,
                                        this,
                                        grid_dim_x,
                                        grid_dim_y,
                                        grid_dim_z,
                                        block_dim_x,
                                        block_dim_y,
                                        block_dim_z,
                                        compilation_result,
                                        cuda_stream_,
                                        kernel_params,
                                        kernel_params_log,
                                        optimize_block_and_grid_sizes};
}

// Assumes DeviceKernelLaunchErrorLogDump is logged on FATAL severity.
// Log any information that can help diagnose the cause of the CUDA error.
std::ostream& operator<<(std::ostream& os, DeviceKernelLaunchErrorLogDump const& ld) {
  // clang-format off
  os << CudaErrorLog{ld.cu_result}
     << "\ngrid_dim_x(" << ld.grid_dim_x
     << ") grid_dim_y(" << ld.grid_dim_y
     << ") grid_dim_z(" << ld.grid_dim_z
     << ") block_dim_x(" << ld.block_dim_x
     << ") block_dim_y(" << ld.block_dim_y
     << ") block_dim_z(" << ld.block_dim_z
     << ") qe_cuda_stream(" << static_cast<void*>(ld.qe_cuda_stream)
     << ") optimize_block_and_grid_sizes(" << ld.optimize_block_and_grid_sizes
     << ')';
  if (auto* nvidia_kernel = dynamic_cast<NvidiaKernel*>(ld.device_kernel)) {
    os << "\nfunction_ptr(" << static_cast<void*>(nvidia_kernel->function_ptr_)
       << ") module_ptr(" << static_cast<void*>(nvidia_kernel->module_ptr_)
       << ") device_id(" << nvidia_kernel->device_id_
       << ") name(" << nvidia_kernel->name_
       << ')';
  }
  return os << '\n' << "kernel_params_log(" << ld.kernel_params_log << ')'
            << '\n' << "compilation_result(" << ld.compilation_result << ')';
  // clang-format on
}
#endif

std::unique_ptr<DeviceKernel> create_device_kernel(const CompilationContext* ctx,
                                                   int device_id,
                                                   CUstream cuda_stream) {
#ifdef HAVE_CUDA
  return std::make_unique<NvidiaKernel>(ctx, device_id, cuda_stream);
#else
  return nullptr;
#endif
}
