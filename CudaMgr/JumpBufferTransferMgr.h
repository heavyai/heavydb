/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda.h>

namespace CudaMgr_Namespace {

class JumpBufferTransferMgr {
 public:
  JumpBufferTransferMgr(size_t device_count,
                        const std::vector<CUcontext>& device_contexts);
  ~JumpBufferTransferMgr();

  bool shouldUseForHostToDeviceTransfer(size_t num_bytes) const;

  bool shouldUseForDeviceToHostTransfer(size_t num_bytes) const;

  void startBackgroundAllocations();

  bool copyHostToDevice(int8_t* device_ptr,
                        const int8_t* host_ptr,
                        size_t num_bytes,
                        int32_t device_num,
                        CUstream cuda_stream);

  bool copyHostToDeviceFromPinnedProducer(
      int8_t* device_ptr,
      size_t num_bytes,
      int32_t device_num,
      CUstream cuda_stream,
      const std::function<void(int8_t* host_ptr, size_t num_bytes, size_t offset)>&
          producer);

  bool copyDeviceToHost(int8_t* host_ptr,
                        const int8_t* device_ptr,
                        size_t num_bytes,
                        int32_t device_num,
                        CUstream cuda_stream);

 private:
  int8_t* allocatePinnedHostMem(int32_t device_num, size_t num_bytes);
  void freePinnedHostMem(int32_t device_num, int8_t* host_ptr);

  struct QueuedJumpBuffer {
    int8_t* buffer{nullptr};
    size_t size{0};          // Total size of jump buffer
    size_t segment_size{0};  // Size of each segment (half of total buffer)
    std::atomic<bool> allocation_in_progress{false};

    std::mutex
        transfer_mutex;      // For synchronizing concurrent transfers/use of jump buffers
    std::mutex queue_mutex;  // For synchronizing producer/consumer queue operations

    std::condition_variable writer_cv;
    std::condition_variable reader_cv;

    // Track state of each segment
    bool read_allowed[2]{false, false};  // true when producer has filled segment
    bool write_allowed[2]{true,
                          true};  // true when consumer has finished copying from segment
    bool write_finished{false};   // Signals end of all data writes
    bool error_occurred{false};   // Signals error during data transfer
    std::exception_ptr transfer_exception;
    size_t segment_buffer_size[2]{0, 0};  // Actual size of each queued segment
  };

  void startBackgroundAllocation(QueuedJumpBuffer& jump_buffer, int32_t device_num);
  void joinBackgroundAllocationThreads();
  void resetTransferState(QueuedJumpBuffer& jump_buffer);
  bool ensureJumpBufferAllocated(QueuedJumpBuffer& jump_buffer, int32_t device_num);
  void recordTransferError(QueuedJumpBuffer& jump_buffer);
  std::thread createHostToDeviceProducer(QueuedJumpBuffer& jump_buffer,
                                         const int8_t* host_ptr,
                                         size_t num_bytes);
  std::thread createHostToDeviceProducer(
      QueuedJumpBuffer& jump_buffer,
      size_t num_bytes,
      const std::function<void(int8_t* host_ptr, size_t num_bytes, size_t offset)>&
          producer);
  void runHostToDeviceProducer(
      QueuedJumpBuffer& jump_buffer,
      size_t num_bytes,
      const std::function<void(int8_t* host_ptr, size_t num_bytes, size_t offset)>&
          producer);
  std::thread createHostToDeviceConsumer(QueuedJumpBuffer& jump_buffer,
                                         int8_t* device_ptr,
                                         int32_t device_num,
                                         CUstream cuda_stream);
  std::thread createDeviceToHostProducer(QueuedJumpBuffer& jump_buffer,
                                         const int8_t* device_ptr,
                                         int32_t device_num,
                                         CUstream cuda_stream,
                                         size_t num_bytes);
  std::thread createDeviceToHostConsumer(QueuedJumpBuffer& jump_buffer, int8_t* host_ptr);
  QueuedJumpBuffer* tryLockJumpBuffer(int32_t device_num);

  std::vector<std::vector<std::unique_ptr<QueuedJumpBuffer>>> jump_buffers_;
  std::vector<std::thread> allocation_threads_;

  const size_t device_count_;
  const std::vector<CUcontext>& device_contexts_;
};
}  // namespace CudaMgr_Namespace
