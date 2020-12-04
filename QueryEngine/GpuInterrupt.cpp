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
  VLOG(1) << "Receive INTERRUPT request on the Executor " << this;
  interrupted_.store(true);
  if (g_enable_runtime_query_interrupt) {
    {
      // We first check the query session is enrolled (as running query or pending query)
      // If not, this interrupt request is a false alarm
      mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
      if (!checkIsQuerySessionEnrolled(query_session, session_read_lock)) {
        session_read_lock.unlock();
        interrupted_.store(false);
        VLOG(1) << "INTERRUPT request on the Executor " << this << " is skipped";
        return;
      }
      session_read_lock.unlock();
    }
    // We have to cover interrupt request from *any* session because we don't know
    // whether the request is for the running query or pending query.
    // So we first check the session has been interrupted
    mapd_unique_lock<mapd_shared_mutex> session_write_lock(executor_session_mutex_);
    this->setQuerySessionAsInterrupted(query_session, session_write_lock);
    session_write_lock.unlock();

    // if this request is not for running session, all we need is turning the interrupt
    // flag on for the target query session (do not need to send interrupt signal to
    // the running kernel)
    mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
    bool isRunningSession =
        this->checkCurrentQuerySession(query_session, session_read_lock);
    session_read_lock.unlock();
    if (!isRunningSession) {
      interrupted_.store(false);
      return;
    }
  }

  bool CPU_execution_mode = true;

#ifdef HAVE_CUDA
  // The below code is basically for runtime query interrupt for GPU.
  // It is also possible that user forces to use CPU-mode even if the user has GPU(s).
  // In this case, we should not execute the code in below to avoid runtime failure
  auto cuda_mgr = Catalog_Namespace::SysCatalog::instance().getDataMgr().getCudaMgr();
  if (cuda_mgr) {
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

        if (catalog_) {
          catalog_->getDataMgr().getCudaMgr()->setContext(device_id);
        } else {
          Catalog_Namespace::SysCatalog::instance().getDataMgr().getCudaMgr()->setContext(
              device_id);
        }

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
  }

  if (g_enable_runtime_query_interrupt && CPU_execution_mode) {
    // turn interrupt flag on for CPU mode
    VLOG(1) << "Try to interrupt the running query on CPU";
    check_interrupt_init(static_cast<unsigned>(INT_ABORT));
  }
}

void Executor::resetInterrupt() {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
#endif

  if (g_enable_dynamic_watchdog) {
    dynamic_watchdog_init(static_cast<unsigned>(DW_RESET));
  } else if (g_enable_runtime_query_interrupt) {
    check_interrupt_init(static_cast<unsigned>(INT_RESET));
  }

  if (g_cluster) {
    bool sessionLeft = false;
    mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
    std::string curSession = getCurrentQuerySession(session_read_lock);
    sessionLeft = checkIsQuerySessionInterrupted(curSession, session_read_lock);
    session_read_lock.unlock();
    if (curSession != "" || sessionLeft) {
      mapd_unique_lock<mapd_shared_mutex> session_write_lock(executor_session_mutex_);
      removeFromQuerySessionList(curSession, session_write_lock);
      invalidateRunningQuerySession(session_write_lock);
      session_write_lock.unlock();
    }
  }

  if (interrupted_.load()) {
    interrupted_.store(false);
  }
  VLOG(1) << "RESET Executor " << this << " that had previously been interrupted";
}
