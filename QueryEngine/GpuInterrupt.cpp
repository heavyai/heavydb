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
  VLOG(1) << "Executor " << this << ", mask 0x" << std::hex << gpu_active_modules_device_mask_ << ": Registered module "
          << module << " on device " << std::to_string(device_id);
#endif
}

void Executor::unregisterActiveModule(void* module, const int device_id) const {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
  CHECK_LT(device_id, max_gpu_count);
  if ((gpu_active_modules_device_mask_ & (1 << device_id)) == 0)
    return;
  CHECK_EQ(gpu_active_modules_[device_id], module);
  gpu_active_modules_device_mask_ ^= (1 << device_id);
  VLOG(1) << "Executor " << this << ", mask 0x" << std::hex << gpu_active_modules_device_mask_
          << ": Unregistered module " << module << " on device " << std::to_string(device_id);
#endif
}

void Executor::interrupt() {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
  VLOG(1) << "Executor " << this << ": Interrupting Active Modules: mask 0x" << std::hex
          << gpu_active_modules_device_mask_;
  CUcontext old_cu_context;
  checkCudaErrors(cuCtxGetCurrent(&old_cu_context));
  for (int device_id = 0; device_id < max_gpu_count; device_id++) {
    if (gpu_active_modules_device_mask_ & (1 << device_id)) {
      void* module = gpu_active_modules_[device_id];
      auto cu_module = static_cast<CUmodule>(module);
      if (!cu_module)
        continue;
      VLOG(1) << "Terminating module " << module << " on device " << std::to_string(device_id)
              << ", gpu_active_modules_device_mask_: " << std::hex << std::to_string(gpu_active_modules_device_mask_);

      catalog_->get_dataMgr().cudaMgr_->setContext(device_id);

      // Create high priority non-blocking communication stream
      CUstream cu_stream1;
      checkCudaErrors(cuStreamCreateWithPriority(&cu_stream1, CU_STREAM_NON_BLOCKING, 1));

      CUevent start, stop;
      cuEventCreate(&start, 0);
      cuEventCreate(&stop, 0);
      cuEventRecord(start, cu_stream1);

      CUdeviceptr dw_abort;
      size_t dw_abort_size;
      if (cuModuleGetGlobal(&dw_abort, &dw_abort_size, cu_module, "dw_abort") == CUDA_SUCCESS) {
        CHECK_EQ(dw_abort_size, sizeof(uint32_t));
        int32_t abort_val = 1;
        checkCudaErrors(cuMemcpyHtoDAsync(dw_abort, reinterpret_cast<void*>(&abort_val), sizeof(int32_t), cu_stream1));

        if (device_id == 0) {
          LOG(INFO) << "GPU: Async Abort submitted to Device " << std::to_string(device_id);
        }
      }

      cuEventRecord(stop, cu_stream1);
      cuEventSynchronize(stop);
      float milliseconds = 0;
      cuEventElapsedTime(&milliseconds, start, stop);
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": submitted async request to abort: " << std::to_string(milliseconds) << " ms\n";
      checkCudaErrors(cuStreamDestroy(cu_stream1));
    }
  }
  checkCudaErrors(cuCtxSetCurrent(old_cu_context));
#endif

  dynamic_watchdog_init(static_cast<unsigned>(DW_ABORT));

  interrupted_ = true;
  VLOG(1) << "INTERRUPT Executor " << this;
}

void Executor::resetInterrupt() {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
#endif

  if (!interrupted_)
    return;

  dynamic_watchdog_init(static_cast<unsigned>(DW_RESET));

  interrupted_ = false;
  VLOG(1) << "RESET Executor " << this << " that had previously been interrupted";
}
