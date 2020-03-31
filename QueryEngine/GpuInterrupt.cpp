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

void Executor::interrupt(std::string query_session, std::string interrupt_session) {
  VLOG(1) << "Receive INTERRUPT request on the Executor " << this;
  interrupted_ = true;
  if (g_enable_runtime_query_interrupt) {
    // We first check the query session as interrupted first.
    this->setQuerySessionAsInterrupted(query_session);
    // We have to cover interrupt request from *any* session because we don't know
    // whether the request is for the running query or pending query.
    // But pending query hangs on the executor until the running query is finished
    // to get the computing resources to execute the query
    // So, we just need to kill the pending query on the executor.
    if (!this->checkCurrentQuerySession(query_session)) {
      interrupted_ = false;
      return;
    }
  }

#ifdef HAVE_CUDA
  // The below code is basically for runtime query interrupt for GPU.
  // It is also possible that user forces to use CPU-mode even if the user has GPU(s).
  // In this case, we should not execute the code in below to avoid runtime failure
  // Also, checking catalog ptr is to prevent runtime failure before we assign catalog
  // i.e., send an interrupt signal in query preparation time
  if (catalog_ && !isCPUOnly()) {
    CHECK_GE(deviceCount(ExecutorDeviceType::GPU), 1);
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
        }
        VLOG(1) << "Terminating module " << module << " on device "
                << std::to_string(device_id)
                << ", gpu_active_modules_device_mask_: " << std::hex
                << std::to_string(gpu_active_modules_device_mask_);

        catalog_->getDataMgr().getCudaMgr()->setContext(device_id);

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
              LOG(INFO) << "GPU: Async Abort submitted to Device "
                        << std::to_string(device_id);
            }
          }
        }

        if (g_enable_runtime_query_interrupt) {
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
              LOG(INFO) << "GPU: Async Abort submitted to Device "
                        << std::to_string(device_id);
            }
          } else if (status == CUDA_ERROR_NOT_FOUND) {
            std::runtime_error(
                "Runtime query interruption is failed: "
                "a interrupt flag on GPU does not be initialized.");
          } else {
            // if we reach here, query runtime interrupt is failed due to
            // one of the following error: CUDA_ERROR_NOT_INITIALIZED,
            // CUDA_ERROR_DEINITIALIZED. CUDA_ERROR_INVALID_CONTEXT, and
            // CUDA_ERROR_INVALID_VALUE. All those error codes are due to device failure.
            std::runtime_error("Runtime interrupt is failed due to device-related issue");
          }

          cuEventRecord(stop, cu_stream1);
          cuEventSynchronize(stop);
          float milliseconds = 0;
          cuEventElapsedTime(&milliseconds, start, stop);
          VLOG(1) << "Device " << std::to_string(device_id)
                  << ": submitted async request to abort SUCCESS: "
                  << std::to_string(milliseconds) << " ms\n";
          checkCudaErrors(cuStreamDestroy(cu_stream1));
        }
      }
      checkCudaErrors(cuCtxSetCurrent(old_cu_context));
    }
  }
#endif
  if (g_enable_dynamic_watchdog) {
    dynamic_watchdog_init(static_cast<unsigned>(DW_ABORT));
  } else if (g_enable_runtime_query_interrupt) {
    check_interrupt_init(static_cast<unsigned>(INT_ABORT));
  }
}

void Executor::resetInterrupt() {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
#endif

  if (!interrupted_) {
    return;
  }

  if (g_enable_dynamic_watchdog) {
    dynamic_watchdog_init(static_cast<unsigned>(DW_RESET));
  } else if (g_enable_runtime_query_interrupt) {
    check_interrupt_init(static_cast<unsigned>(INT_RESET));
  }

  interrupted_ = false;
  VLOG(1) << "RESET Executor " << this << " that had previously been interrupted";
}
