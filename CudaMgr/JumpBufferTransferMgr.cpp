/*
 * Copyright 2025 HEAVY.AI, Inc.
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

#include "JumpBufferTransferMgr.h"

#include "CudaMgr/CudaShared.h"
#include "Logger/Logger.h"

size_t g_jump_buffer_size{128 * 1024 * 1024};  // 128MB
size_t g_jump_buffer_parallel_copy_threads{4};

// We use a larger default minimum transfer size for D2H transfers because host memory
// copies from the pinned jump buffer are slower than copies to the pinned jump buffer
// and require larger transfers to amortize the overhead.
size_t g_jump_buffer_min_h2d_transfer_threshold{32 * 1024 * 1024};  // 32MB
size_t g_jump_buffer_min_d2h_transfer_threshold{64 * 1024 * 1024};  // 64MB

namespace CudaMgr_Namespace {
JumpBufferTransferMgr::JumpBufferTransferMgr(
    size_t device_count,
    const std::vector<CUcontext>& device_contexts)
    : device_count_(device_count), device_contexts_(device_contexts) {
  if (g_jump_buffer_size > 0) {
    CHECK_EQ(device_count_, device_contexts_.size());

    VLOG(1) << "Initializing " << device_count << " pinned memory jump buffers of size "
            << g_jump_buffer_size << " bytes";
    jump_buffers_.reserve(device_count_);
    for (size_t device_num = 0; device_num < device_count_; device_num++) {
      jump_buffers_.emplace_back(std::make_unique<QueuedJumpBuffer>());
      jump_buffers_.back()->buffer =
          allocatePinnedHostMem(device_num, g_jump_buffer_size);
      jump_buffers_.back()->size = g_jump_buffer_size;
    }

    CHECK_EQ(device_count_, jump_buffers_.size());
    VLOG(1) << "Initialized " << device_count << " pinned memory jump buffers of size "
            << g_jump_buffer_size << " bytes";
  }
}

JumpBufferTransferMgr::~JumpBufferTransferMgr() {
  if (!jump_buffers_.empty()) {
    VLOG(1) << "Freeing " << jump_buffers_.size() << " pinned memory jump buffers";
    int32_t device_num{0};
    for (auto& jump_buffer : jump_buffers_) {
      if (jump_buffer) {
        try {
          freePinnedHostMem(device_num, jump_buffer->buffer);
        } catch (const CudaErrorException& e) {
          if (e.getStatus() == CUDA_ERROR_DEINITIALIZED) {
            // Skip further attempts to free pinned host memory if the CUDA driver is
            // shutting down.
            jump_buffers_.clear();
            return;
          }
          VLOG(1) << "CUDA error when attempting to free pinned host memory: "
                  << e.what();
        }
        jump_buffer->buffer = nullptr;
        jump_buffer->size = 0;
      }
      device_num++;
    }
    VLOG(1) << "Freed " << jump_buffers_.size() << " pinned memory jump buffers";
    jump_buffers_.clear();
  }
}

bool JumpBufferTransferMgr::shouldUseForHostToDeviceTransfer(size_t num_bytes) const {
  return g_jump_buffer_size > 0 && num_bytes >= g_jump_buffer_min_h2d_transfer_threshold;
}

bool JumpBufferTransferMgr::shouldUseForDeviceToHostTransfer(size_t num_bytes) const {
  return g_jump_buffer_size > 0 && num_bytes >= g_jump_buffer_min_d2h_transfer_threshold;
}

namespace {
void parallel_copy_buffer(int8_t* dst, const int8_t* src, size_t buffer_size) {
  CHECK_GT(buffer_size, size_t(0));

  // Calculate number of threads based on segment size ratio
  const size_t segment_size = std::max(g_jump_buffer_size / 2, size_t(1));
  const size_t num_threads =
      std::min((buffer_size * g_jump_buffer_parallel_copy_threads + segment_size - 1) /
                   segment_size,
               g_jump_buffer_parallel_copy_threads);
  CHECK_GT(num_threads, size_t(0));

  size_t base_chunk_size = buffer_size / num_threads;
  size_t remainder = buffer_size % num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  size_t offset = 0;
  for (size_t i = 0; i < num_threads; ++i) {
    size_t thread_chunk_size = base_chunk_size;
    if (remainder > 0) {
      thread_chunk_size++;
      remainder--;
    }

    threads.emplace_back(
        [=]() { memcpy(dst + offset, src + offset, thread_chunk_size); });

    offset += thread_chunk_size;
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

size_t get_segment_size(size_t num_bytes) {
  // Calculate dynamic segment size based on transfer size
  constexpr size_t min_segment_size = 16 * 1024 * 1024;  // 16MB minimum segment size

  // Simply divide by 2 for double buffering
  size_t segment_size = (num_bytes + 1) / 2;  // Round up division to handle odd numbers

  // Apply constraints
  segment_size = std::max(segment_size, min_segment_size);
  segment_size = std::min(segment_size, g_jump_buffer_size / 2);

  return segment_size;
}
}  // namespace

bool JumpBufferTransferMgr::copyHostToDevice(int8_t* device_ptr,
                                             const int8_t* host_ptr,
                                             size_t num_bytes,
                                             int32_t device_num,
                                             CUstream cuda_stream) {
  CHECK_GE(device_num, 0);
  CHECK_LT(size_t(device_num), jump_buffers_.size());

  auto& jump_buffer = *jump_buffers_[device_num];
  if (jump_buffer.transfer_mutex.try_lock()) {
    try {
      jump_buffer.segment_size = get_segment_size(num_bytes);

      resetTransferState(jump_buffer);

      set_context(device_contexts_, device_num);

      auto producer = createHostToDeviceProducer(jump_buffer, host_ptr, num_bytes);
      auto consumer =
          createHostToDeviceConsumer(jump_buffer, device_ptr, device_num, cuda_stream);

      producer.join();
      consumer.join();

      jump_buffer.transfer_mutex.unlock();
      return true;
    } catch (...) {
      jump_buffer.transfer_mutex.unlock();
      throw;
    }
  } else {
    // Fall back to direct transfer if the transfer lock cannot be acquired
    return false;
  }
}

void JumpBufferTransferMgr::resetTransferState(QueuedJumpBuffer& jump_buffer) {
  std::lock_guard<std::mutex> lock(jump_buffer.queue_mutex);
  jump_buffer.read_allowed[0] = false;
  jump_buffer.read_allowed[1] = false;
  // Start with segments available for writes
  jump_buffer.write_allowed[0] = true;
  jump_buffer.write_allowed[1] = true;
  jump_buffer.write_finished = false;
  jump_buffer.error_occurred = false;
  jump_buffer.segment_buffer_size = 0;
}

std::thread JumpBufferTransferMgr::createHostToDeviceProducer(
    QueuedJumpBuffer& jump_buffer,
    const int8_t* host_ptr,
    size_t num_bytes) {
  return std::thread([&jump_buffer, host_ptr, num_bytes]() {
    size_t bytes_remaining = num_bytes;
    size_t src_offset = 0;
    int32_t current_segment = 0;

    while (bytes_remaining > 0) {
      const size_t copy_size = std::min(bytes_remaining, jump_buffer.segment_size);

      // Wait for current segment to be available
      {
        std::unique_lock<std::mutex> lock(jump_buffer.queue_mutex);
        jump_buffer.writer_cv.wait(lock, [&]() {
          return jump_buffer.error_occurred || jump_buffer.write_allowed[current_segment];
        });
        jump_buffer.write_allowed[current_segment] = false;
      }

      if (jump_buffer.error_occurred) {
        return;
      }

      try {
        parallel_copy_buffer(
            jump_buffer.buffer + (current_segment * jump_buffer.segment_size),
            host_ptr + src_offset,
            copy_size);
      } catch (...) {
        // Ensure consumer stops processing if an exception occurs.
        {
          std::lock_guard<std::mutex> lock(jump_buffer.queue_mutex);
          jump_buffer.error_occurred = true;
        }
        jump_buffer.reader_cv.notify_one();
        throw;
      }

      // Mark segment as ready to be read and specify the segment buffer size
      {
        std::lock_guard<std::mutex> lock(jump_buffer.queue_mutex);
        jump_buffer.read_allowed[current_segment] = true;
        jump_buffer.segment_buffer_size = copy_size;
      }
      jump_buffer.reader_cv.notify_one();

      bytes_remaining -= copy_size;
      src_offset += copy_size;
      current_segment = 1 - current_segment;
    }

    // Signal completion
    {
      std::lock_guard<std::mutex> lock(jump_buffer.queue_mutex);
      jump_buffer.write_finished = true;
    }
    jump_buffer.reader_cv.notify_one();
  });
}

std::thread JumpBufferTransferMgr::createHostToDeviceConsumer(
    QueuedJumpBuffer& jump_buffer,
    int8_t* device_ptr,
    int32_t device_num,
    CUstream cuda_stream) {
  return std::thread([this, &jump_buffer, device_ptr, device_num, cuda_stream]() {
    set_context(device_contexts_, device_num);

    int32_t current_segment = 0;
    size_t dst_offset = 0;

    bool continue_reading{true};
    while (continue_reading) {
      size_t copy_size;

      // Wait for the current segment to be ready for reads
      {
        std::unique_lock<std::mutex> lock(jump_buffer.queue_mutex);
        jump_buffer.reader_cv.wait(lock, [&]() {
          return jump_buffer.error_occurred ||
                 jump_buffer.read_allowed[current_segment] ||
                 (jump_buffer.write_finished && !jump_buffer.read_allowed[0] &&
                  !jump_buffer.read_allowed[1]);
        });

        if (jump_buffer.error_occurred) {
          return;
        }

        if (jump_buffer.read_allowed[current_segment]) {
          copy_size = jump_buffer.segment_buffer_size;
          jump_buffer.read_allowed[current_segment] = false;
          continue_reading = true;
        } else {
          continue_reading = false;
        }
      }

      if (continue_reading) {
        try {
          if (cuda_stream) {
            check_error(cuMemcpyHtoDAsync(
                reinterpret_cast<CUdeviceptr>(device_ptr + dst_offset),
                jump_buffer.buffer + (current_segment * jump_buffer.segment_size),
                copy_size,
                cuda_stream));
            check_error(cuStreamSynchronize(cuda_stream));
          } else {
            check_error(cuMemcpyHtoD(
                reinterpret_cast<CUdeviceptr>(device_ptr + dst_offset),
                jump_buffer.buffer + (current_segment * jump_buffer.segment_size),
                copy_size));
          }
        } catch (...) {
          // Ensure producer stops processing if an exception occurs.
          {
            std::lock_guard<std::mutex> lock(jump_buffer.queue_mutex);
            jump_buffer.error_occurred = true;
          }
          jump_buffer.writer_cv.notify_one();
          throw;
        }

        // Mark segment as available for writes
        {
          std::lock_guard<std::mutex> lock(jump_buffer.queue_mutex);
          jump_buffer.write_allowed[current_segment] = true;
        }
        jump_buffer.writer_cv.notify_one();

        dst_offset += copy_size;
        current_segment = 1 - current_segment;
      }
    }
  });
}

bool JumpBufferTransferMgr::copyDeviceToHost(int8_t* host_ptr,
                                             const int8_t* device_ptr,
                                             size_t num_bytes,
                                             int32_t device_num,
                                             CUstream cuda_stream) {
  CHECK_GE(device_num, 0);
  CHECK_LT(size_t(device_num), jump_buffers_.size());

  auto& jump_buffer = *jump_buffers_[device_num];
  if (jump_buffer.transfer_mutex.try_lock()) {
    try {
      jump_buffer.segment_size = get_segment_size(num_bytes);

      resetTransferState(jump_buffer);

      set_context(device_contexts_, device_num);

      auto producer = createDeviceToHostProducer(
          jump_buffer, device_ptr, device_num, cuda_stream, num_bytes);
      auto consumer = createDeviceToHostConsumer(jump_buffer, host_ptr);

      producer.join();
      consumer.join();

      jump_buffer.transfer_mutex.unlock();
      return true;
    } catch (...) {
      jump_buffer.transfer_mutex.unlock();
      throw;
    }
  } else {
    // Fall back to direct transfer if the transfer lock cannot be acquired
    return false;
  }
}

std::thread JumpBufferTransferMgr::createDeviceToHostProducer(
    QueuedJumpBuffer& jump_buffer,
    const int8_t* device_ptr,
    int32_t device_num,
    CUstream cuda_stream,
    size_t num_bytes) {
  return std::thread(
      [this, &jump_buffer, device_ptr, num_bytes, device_num, cuda_stream]() {
        set_context(device_contexts_, device_num);

        size_t bytes_remaining = num_bytes;
        size_t src_offset = 0;
        int32_t current_segment = 0;

        while (bytes_remaining > 0) {
          size_t copy_size = std::min(bytes_remaining, jump_buffer.segment_size);

          // Wait for current segment to be available
          {
            std::unique_lock<std::mutex> lock(jump_buffer.queue_mutex);
            jump_buffer.writer_cv.wait(lock, [&]() {
              return jump_buffer.error_occurred ||
                     jump_buffer.write_allowed[current_segment];
            });

            if (jump_buffer.error_occurred) {
              return;
            }

            jump_buffer.write_allowed[current_segment] = false;
          }

          try {
            // Copy from GPU to jump buffer
            if (cuda_stream) {
              check_error(cuMemcpyDtoHAsync(
                  jump_buffer.buffer + (current_segment * jump_buffer.segment_size),
                  reinterpret_cast<CUdeviceptr>(device_ptr + src_offset),
                  copy_size,
                  cuda_stream));
              check_error(cuStreamSynchronize(cuda_stream));
            } else {
              check_error(cuMemcpyDtoH(
                  jump_buffer.buffer + (current_segment * jump_buffer.segment_size),
                  reinterpret_cast<CUdeviceptr>(device_ptr + src_offset),
                  copy_size));
            }
          } catch (...) {
            // Ensure consumer stops processing if an exception occurs.
            {
              std::lock_guard<std::mutex> lock(jump_buffer.queue_mutex);
              jump_buffer.error_occurred = true;
            }
            jump_buffer.reader_cv.notify_one();
            throw;
          }

          // Mark segment as ready for reads
          {
            std::lock_guard<std::mutex> lock(jump_buffer.queue_mutex);
            jump_buffer.read_allowed[current_segment] = true;
            jump_buffer.segment_buffer_size = copy_size;
          }
          jump_buffer.reader_cv.notify_one();

          bytes_remaining -= copy_size;
          src_offset += copy_size;
          current_segment = 1 - current_segment;
        }

        // Signal completion
        {
          std::lock_guard<std::mutex> lock(jump_buffer.queue_mutex);
          jump_buffer.write_finished = true;
        }
        jump_buffer.reader_cv.notify_one();
      });
}

std::thread JumpBufferTransferMgr::createDeviceToHostConsumer(
    QueuedJumpBuffer& jump_buffer,
    int8_t* host_ptr) {
  // Consumer thread - writes from jump buffer to host memory
  return std::thread([&jump_buffer, host_ptr]() {
    int32_t current_segment = 0;
    size_t dst_offset = 0;

    bool continue_reading{true};
    while (continue_reading) {
      size_t copy_size;

      // Wait for current segment to be ready for reads
      {
        std::unique_lock<std::mutex> lock(jump_buffer.queue_mutex);
        jump_buffer.reader_cv.wait(lock, [&]() {
          return jump_buffer.error_occurred ||
                 jump_buffer.read_allowed[current_segment] ||
                 (jump_buffer.write_finished && !jump_buffer.read_allowed[0] &&
                  !jump_buffer.read_allowed[1]);
        });

        if (jump_buffer.error_occurred) {
          return;
        }

        if (jump_buffer.read_allowed[current_segment]) {
          copy_size = jump_buffer.segment_buffer_size;
          jump_buffer.read_allowed[current_segment] = false;
          continue_reading = true;
        } else {
          continue_reading = false;
        }
      }

      if (continue_reading) {
        try {
          // Copy from jump buffer to host memory
          parallel_copy_buffer(
              host_ptr + dst_offset,
              jump_buffer.buffer + (current_segment * jump_buffer.segment_size),
              copy_size);
        } catch (...) {
          // Ensure producer stops processing if an exception occurs.
          {
            std::lock_guard<std::mutex> lock(jump_buffer.queue_mutex);
            jump_buffer.error_occurred = true;
          }
          jump_buffer.writer_cv.notify_one();
        }

        // Mark segment as available
        {
          std::lock_guard<std::mutex> lock(jump_buffer.queue_mutex);
          jump_buffer.write_allowed[current_segment] = true;
        }
        jump_buffer.writer_cv.notify_one();

        dst_offset += copy_size;
        current_segment = 1 - current_segment;
      }
    }
  });
}

int8_t* JumpBufferTransferMgr::allocatePinnedHostMem(int32_t device_num,
                                                     size_t num_bytes) {
  set_context(device_contexts_, device_num);

  void* host_ptr;
  check_error(cuMemHostAlloc(&host_ptr, num_bytes, CU_MEMHOSTALLOC_PORTABLE));
  return reinterpret_cast<int8_t*>(host_ptr);
}

void JumpBufferTransferMgr::freePinnedHostMem(int32_t device_num, int8_t* host_ptr) {
  if (!host_ptr) {
    return;
  }

  set_context(device_contexts_, device_num);
  check_error(cuMemFreeHost(host_ptr));
}
}  // namespace CudaMgr_Namespace
