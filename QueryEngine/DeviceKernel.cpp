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

#ifdef HAVE_CUDA
CUstream getQueryEngineCudaStreamForDevice(int device_num);

class NvidiaKernel : public DeviceKernel {
 public:
  NvidiaKernel(const CompilationContext* ctx, int device_id) : device_id_(device_id) {
    auto cuda_ctx = dynamic_cast<const GpuCompilationContext*>(ctx);
    CHECK(cuda_ctx);
    name_ = cuda_ctx->name(device_id);
    const auto native_code = cuda_ctx->getNativeCode(device_id);
    function_ptr_ = static_cast<CUfunction>(native_code.first);
    module_ptr_ = static_cast<CUmodule>(native_code.second);
  }

  void launch(unsigned int grid_dim_x,
              unsigned int grid_dim_y,
              unsigned int grid_dim_z,
              unsigned int block_dim_x,
              unsigned int block_dim_y,
              unsigned int block_dim_z,
              CompilationResult const& compilation_result,
              void** kernel_params,
              KernelParamsLog const& kernel_params_log,
              bool optimize_block_and_grid_sizes) override {
    size_t const shared_memory_size =
        compilation_result.gpu_smem_context.getSharedMemorySize();
    auto qe_cuda_stream = getQueryEngineCudaStreamForDevice(device_id_);
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
            << " and " << block_dim_x;
    checkCudaErrors(cuLaunchKernel(function_ptr_,
                                   grid_dim_x,
                                   grid_dim_y,
                                   grid_dim_z,
                                   block_dim_x,
                                   block_dim_y,
                                   block_dim_z,
                                   shared_memory_size,
                                   qe_cuda_stream,
                                   kernel_params,
                                   nullptr));
    CUresult const cu_result = cuStreamSynchronize(qe_cuda_stream);
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
                                          qe_cuda_stream,
                                          kernel_params,
                                          kernel_params_log,
                                          optimize_block_and_grid_sizes};
  }

  void initializeDynamicWatchdog(bool could_interrupt, uint64_t cycle_budget) override {
    CHECK(module_ptr_);
    auto init_start = timer_start();
    CUdeviceptr dw_cycle_budget;
    size_t dw_cycle_budget_size;

    // Translate milliseconds to device cycles
    if (device_id_ == 0) {
      LOG(INFO) << "Dynamic Watchdog budget: GPU: " << g_dynamic_watchdog_time_limit
                << "ms, " << cycle_budget << " cycles";
    }
    auto qe_cuda_stream = getQueryEngineCudaStreamForDevice(device_id_);
    checkCudaErrors(cuModuleGetGlobal(
        &dw_cycle_budget, &dw_cycle_budget_size, module_ptr_, "dw_cycle_budget"));
    CHECK_EQ(dw_cycle_budget_size, sizeof(uint64_t));
    checkCudaErrors(cuMemcpyHtoDAsync(dw_cycle_budget,
                                      reinterpret_cast<void*>(&cycle_budget),
                                      sizeof(uint64_t),
                                      qe_cuda_stream));
    checkCudaErrors(cuStreamSynchronize(qe_cuda_stream));

    CUdeviceptr dw_sm_cycle_start;
    size_t dw_sm_cycle_start_size;
    checkCudaErrors(cuModuleGetGlobal(
        &dw_sm_cycle_start, &dw_sm_cycle_start_size, module_ptr_, "dw_sm_cycle_start"));
    CHECK_EQ(dw_sm_cycle_start_size, 128 * sizeof(uint64_t));
    checkCudaErrors(cuMemsetD32Async(dw_sm_cycle_start, 0, 128 * 2, qe_cuda_stream));
    checkCudaErrors(cuStreamSynchronize(qe_cuda_stream));

    if (!could_interrupt) {
      // Executor is not marked as interrupted, make sure dynamic watchdog doesn't block
      // execution
      CUdeviceptr dw_abort;
      size_t dw_abort_size;
      checkCudaErrors(
          cuModuleGetGlobal(&dw_abort, &dw_abort_size, module_ptr_, "dw_abort"));
      CHECK_EQ(dw_abort_size, sizeof(uint32_t));
      checkCudaErrors(cuMemsetD32Async(dw_abort, 0, 1, qe_cuda_stream));
      checkCudaErrors(cuStreamSynchronize(qe_cuda_stream));
    }
    VLOG(1) << "Device " << device_id_
            << ": launchGpuCode: dynamic watchdog init: " << timer_stop(init_start)
            << " ms\n";
  }

  void initializeRuntimeInterrupter(const int device_id) override {
    CHECK(module_ptr_);
    auto init_start = timer_start();

    CUdeviceptr runtime_interrupt_flag;
    size_t runtime_interrupt_flag_size;
    checkCudaErrors(cuModuleGetGlobal(&runtime_interrupt_flag,
                                      &runtime_interrupt_flag_size,
                                      module_ptr_,
                                      "runtime_interrupt_flag"));
    CHECK_EQ(runtime_interrupt_flag_size, sizeof(uint32_t));
    auto qe_cuda_stream = getQueryEngineCudaStreamForDevice(device_id);
    checkCudaErrors(cuMemsetD32Async(runtime_interrupt_flag, 0, 1, qe_cuda_stream));
    checkCudaErrors(cuStreamSynchronize(qe_cuda_stream));

    VLOG(1) << "Device " << device_id
            << ": launchGpuCode: runtime query interrupter init: "
            << timer_stop(init_start) << " ms";
    Executor::registerActiveModule(module_ptr_, device_id);
  }

  void resetRuntimeInterrupter(const int device_id) override {
    Executor::unregisterActiveModule(device_id);
  }

  char const* name() const override { return name_.c_str(); }

 private:
  CUfunction function_ptr_;
  CUmodule module_ptr_;
  int device_id_;
  std::string name_;

  friend std::ostream& operator<<(std::ostream&, DeviceKernelLaunchErrorLogDump const&);
};

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
       << ") name_(" << nvidia_kernel->name_
       << ')';
  }
  return os << '\n' << "kernel_params_log(" << ld.kernel_params_log << ')'
            << '\n' << "compilation_result(" << ld.compilation_result << ')';
  // clang-format on
}
#endif

std::unique_ptr<DeviceKernel> create_device_kernel(const CompilationContext* ctx,
                                                   int device_id) {
#ifdef HAVE_CUDA
  return std::make_unique<NvidiaKernel>(ctx, device_id);
#else
  return nullptr;
#endif
}
