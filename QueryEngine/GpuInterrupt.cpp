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

#include "DynamicWatchdog.h"
#include "Execute.h"

void Executor::registerActiveModule(void* module, const int device_id) const {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
  CHECK_LT(device_id, max_gpu_count);
  gpu_active_modules_device_mask_ |= (1 << device_id);
  gpu_active_modules_[device_id] = module;
  VLOG(1) << "Executor " << this << ", mask 0x" << std::hex
          << gpu_active_modules_device_mask_ << ": Registered module " << module
          << " on device " << std::to_string(device_id);
#endif
}

void Executor::unregisterActiveModule(void* module, const int device_id) const {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
  CHECK_LT(device_id, max_gpu_count);
  if ((gpu_active_modules_device_mask_ & (1 << device_id)) == 0) {
    return;
  }
  CHECK_EQ(gpu_active_modules_[device_id], module);
  gpu_active_modules_device_mask_ ^= (1 << device_id);
  VLOG(1) << "Executor " << this << ", mask 0x" << std::hex
          << gpu_active_modules_device_mask_ << ": Unregistered module " << module
          << " on device " << std::to_string(device_id);
#endif
}

void Executor::interrupt(const std::string& query_session,
                         const std::string& interrupt_session) {
  const auto allow_interrupt =
      g_enable_runtime_query_interrupt || g_enable_non_kernel_time_query_interrupt;
  if (allow_interrupt) {
    bool is_running_query = false;
    {
      // here we validate the requested query session is valid (is already enrolled)
      // if not, we skip the interrupt request
      mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
      if (!checkIsQuerySessionEnrolled(query_session, session_read_lock)) {
        VLOG(1) << "Skip the interrupt request (no query has been submitted from the "
                   "given query session)";
        return;
      }
      if (checkIsQuerySessionInterrupted(query_session, session_read_lock)) {
        VLOG(1) << "Skip the interrupt request (already interrupted query session)";
        return;
      }
      // if a query is pending query, we just need to turn interrupt flag for the session
      // on (not sending interrupt signal to "RUNNING" kernel, see the below code)
      is_running_query = checkCurrentQuerySession(query_session, session_read_lock);
    }
    {
      // We have to cover interrupt request from *any* session because we don't know
      // whether the request is for the running query or pending query
      // or for non-kernel time interrupt
      // (or just false alarm that indicates unregistered session in a queue).
      // So we try to set a session has been interrupted once we confirm
      // the session has been enrolled and is not interrupted at this moment
      mapd_unique_lock<mapd_shared_mutex> session_write_lock(executor_session_mutex_);
      setQuerySessionAsInterrupted(query_session, session_write_lock);
    }
    if (!is_running_query) {
      return;
    }
    interrupted_.store(true);
  }

  bool CPU_execution_mode = true;

#ifdef HAVE_CUDA
  // The below code is basically for runtime query interrupt for GPU.
  // It is also possible that user forces to use CPU-mode even if the user has GPU(s).
  // In this case, we should not execute the code in below to avoid runtime failure
  CHECK(data_mgr_);
  auto cuda_mgr = data_mgr_->getCudaMgr();
  if (cuda_mgr && (g_enable_dynamic_watchdog || allow_interrupt)) {
    // we additionally allow sending interrupt signal for
    // `g_enable_non_kernel_time_query_interrupt` especially for CTAS/ITAS queries: data
    // population happens on CPU but select_query can be processed via GPU
    CHECK_GE(cuda_mgr->getDeviceCount(), 1);
    std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
    VLOG(1) << "Executor " << this << ": Interrupting Active Modules: mask 0x" << std::hex
            << gpu_active_modules_device_mask_;
    CUcontext old_cu_context;
    checkCudaErrors(cuCtxGetCurrent(&old_cu_context));
    for (int device_id = 0; device_id < max_gpu_count; device_id++) {
      if (gpu_active_modules_device_mask_ & (1 << device_id)) {
        void* module = gpu_active_modules_[device_id];
        auto cu_module = static_cast<CUmodule>(module);
        if (!cu_module) {
          continue;
        } else {
          VLOG(1) << "Try to interrupt the running query on GPU";
          CPU_execution_mode = false;
        }
        VLOG(1) << "Executor " << this << ": Interrupting Active Modules: mask 0x"
                << std::hex << gpu_active_modules_device_mask_ << " on device "
                << std::to_string(device_id);

        cuda_mgr->setContext(device_id);

        // Create high priority non-blocking communication stream
        CUstream cu_stream1;
        checkCudaErrors(
            cuStreamCreateWithPriority(&cu_stream1, CU_STREAM_NON_BLOCKING, 1));

        CUevent start, stop;
        cuEventCreate(&start, 0);
        cuEventCreate(&stop, 0);
        cuEventRecord(start, cu_stream1);

        if (g_enable_dynamic_watchdog) {
          CUdeviceptr dw_abort;
          size_t dw_abort_size;
          if (cuModuleGetGlobal(&dw_abort, &dw_abort_size, cu_module, "dw_abort") ==
              CUDA_SUCCESS) {
            CHECK_EQ(dw_abort_size, sizeof(uint32_t));
            int32_t abort_val = 1;
            checkCudaErrors(cuMemcpyHtoDAsync(dw_abort,
                                              reinterpret_cast<void*>(&abort_val),
                                              sizeof(int32_t),
                                              cu_stream1));

            if (device_id == 0) {
              VLOG(1) << "GPU: Async Abort submitted to Device "
                      << std::to_string(device_id);
            }
          }
        }

        if (allow_interrupt) {
          CUdeviceptr runtime_interrupt_flag;
          size_t runtime_interrupt_flag_size;
          auto status = cuModuleGetGlobal(&runtime_interrupt_flag,
                                          &runtime_interrupt_flag_size,
                                          cu_module,
                                          "runtime_interrupt_flag");
          if (status == CUDA_SUCCESS) {
            VLOG(1) << "Interrupt on GPU status: CUDA_SUCCESS";
            CHECK_EQ(runtime_interrupt_flag_size, sizeof(uint32_t));
            int32_t abort_val = 1;
            checkCudaErrors(cuMemcpyHtoDAsync(runtime_interrupt_flag,
                                              reinterpret_cast<void*>(&abort_val),
                                              sizeof(int32_t),
                                              cu_stream1));
            if (device_id == 0) {
              VLOG(1) << "GPU: Async Abort submitted to Device "
                      << std::to_string(device_id);
            }
          } else if (status == CUDA_ERROR_NOT_FOUND) {
            std::runtime_error(
                "Runtime query interrupt has failed: an interrupt flag on the GPU could "
                "not be initialized (CUDA_ERROR_CODE: CUDA_ERROR_NOT_FOUND)");
          } else {
            // if we reach here, query runtime interrupt is failed due to
            // one of the following error: CUDA_ERROR_NOT_INITIALIZED,
            // CUDA_ERROR_DEINITIALIZED. CUDA_ERROR_INVALID_CONTEXT, and
            // CUDA_ERROR_INVALID_VALUE. All those error codes are due to device failure.
            const char* error_ret_str = nullptr;
            cuGetErrorName(status, &error_ret_str);
            if (!error_ret_str) {
              error_ret_str = "UNKNOWN";
            }
            std::string error_str(error_ret_str);
            std::runtime_error(
                "Runtime interrupt has failed due to a device related issue "
                "(CUDA_ERROR_CODE: " +
                error_str + ")");
          }

          cuEventRecord(stop, cu_stream1);
          cuEventSynchronize(stop);
          float milliseconds = 0;
          cuEventElapsedTime(&milliseconds, start, stop);
          VLOG(1) << "Device " << std::to_string(device_id)
                  << ": submitted async request to abort SUCCESS: "
                  << std::to_string(milliseconds) << " ms";
          checkCudaErrors(cuStreamDestroy(cu_stream1));
        }
      }
      checkCudaErrors(cuCtxSetCurrent(old_cu_context));
    }
  }
#endif
  if (g_enable_dynamic_watchdog) {
    dynamic_watchdog_init(static_cast<unsigned>(DW_ABORT));
  }

  if (allow_interrupt && CPU_execution_mode) {
    // turn interrupt flag on for CPU mode
    VLOG(1) << "Try to interrupt the running query on CPU";
    check_interrupt_init(static_cast<unsigned>(INT_ABORT));
  }
}

void Executor::resetInterrupt() {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
#endif
  const auto allow_interrupt =
      g_enable_runtime_query_interrupt || g_enable_non_kernel_time_query_interrupt;
  if (g_enable_dynamic_watchdog) {
    dynamic_watchdog_init(static_cast<unsigned>(DW_RESET));
  } else if (allow_interrupt) {
    check_interrupt_init(static_cast<unsigned>(INT_RESET));
  }

  if (interrupted_.load()) {
    VLOG(1) << "RESET Executor " << this << " that had previously been interrupted";
    interrupted_.store(false);
  }
}
