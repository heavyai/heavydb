/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DataMgr/BufferMgr/CpuBufferMgr/CpuBufferMgr.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/Allocators/ArenaAllocator.h"
#include "DataMgr/BufferMgr/Buffer.h"
#include "DataMgr/BufferMgr/CpuBufferMgr/CpuBuffer.h"
#include "DataMgr/FileMgr/FileBuffer.h"
#include "DataMgr/FileMgr/FileInfo.h"
#include "Shared/scope.h"

#include <sys/mman.h>
#include <unistd.h>

#ifdef HAVE_NVCOMP
#include <nvcomp/bitcomp.h>
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>
#ifdef HAVE_NVCOMP_GDEFLATE
#include <nvcomp/gdeflate.h>
#endif
#endif

bool g_enable_gpu_input_cpu_buffer_bypass{false};
size_t g_gpu_input_cpu_buffer_bypass_staging_buffer_bytes{64 * 1024 * 1024};
size_t g_gpu_input_cpu_buffer_bypass_reader_threads{16};
size_t g_gpu_input_compressed_batch_max_bytes{2ULL * 1024 * 1024 * 1024};
bool g_enable_gpu_input_compressed_pipeline{false};
bool g_enable_gpu_input_compressed_peer_exchange{false};
std::string g_gpu_input_cpu_buffer_bypass_mode{"staged"};

namespace Buffer_Namespace {

struct CompressedGpuInputWorkspace {
  explicit CompressedGpuInputWorkspace(AbstractBuffer* destination_buffer)
      : device_id(destination_buffer->getDeviceId()) {
    auto buffer = dynamic_cast<Buffer*>(destination_buffer);
    CHECK(buffer);
    buffer_mgr = buffer->getBufferMgr();
    CHECK(buffer_mgr);
  }

  void releaseAll() noexcept {
    if (buffer) {
      try {
        buffer_mgr->free(buffer);
      } catch (const std::exception& error) {
        LOG(ERROR) << "Failed to release compressed input workspace buffer: device="
                   << device_id << " error=" << error.what();
      } catch (...) {
        LOG(ERROR) << "Failed to release compressed input workspace buffer: device="
                   << device_id;
      }
      buffer = nullptr;
      device = nullptr;
      capacity = 0;
    }
  }

  int32_t device_id;
  BufferMgr* buffer_mgr{nullptr};
  AbstractBuffer* buffer{nullptr};
  int8_t* device{nullptr};
  size_t capacity{0};
};

namespace {

size_t checked_size_add(const size_t lhs,
                        const size_t rhs,
                        const char* const description) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    throw std::overflow_error(description);
  }
  return lhs + rhs;
}

size_t checked_size_multiply(const size_t lhs,
                             const size_t rhs,
                             const char* const description) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error(description);
  }
  return lhs * rhs;
}

bool use_mmap_gpu_input_bypass() {
  return g_gpu_input_cpu_buffer_bypass_mode == "mmap";
}

bool is_cuda_out_of_memory(const CudaMgr_Namespace::CudaErrorException& error) noexcept {
#ifdef HAVE_CUDA
  return error.getStatus() == CUDA_ERROR_OUT_OF_MEMORY;
#else
  static_cast<void>(error);
  return false;
#endif
}

#ifdef HAVE_NVCOMP
constexpr size_t kCompressedWorkspaceAlignment{256};

size_t align_workspace_offset(const size_t offset) {
  static_assert((kCompressedWorkspaceAlignment & (kCompressedWorkspaceAlignment - 1)) ==
                0);
  const auto mask = kCompressedWorkspaceAlignment - 1;
  return checked_size_add(
             offset, mask, "Compressed GPU input workspace alignment overflow") &
         ~mask;
}

struct CompressedWorkspaceLayout {
  size_t append(const size_t bytes) {
    const auto offset = align_workspace_offset(total_bytes);
    total_bytes =
        checked_size_add(offset, bytes, "Compressed GPU input workspace size overflow");
    return offset;
  }

  size_t total_bytes{0};
};

enum class CompressedPayloadLeaseRole { Storage, Producer, Consumer };

struct CompressedPayloadState {
  File_Namespace::FileBuffer* source_file_buffer{nullptr};
  int32_t source_device_id{-1};
  int8_t* source_device_ptr{nullptr};
  size_t payload_bytes{0};
  std::mutex mutex;
  std::condition_variable cv;
  bool accepting_consumers{true};
  bool ready{false};
  bool failed{false};
  size_t active_consumers{0};
};

struct CompressedPayloadLease {
  CompressedPayloadLeaseRole role{CompressedPayloadLeaseRole::Storage};
  std::shared_ptr<CompressedPayloadState> state;
};

class CompressedPayloadExchange {
 public:
  std::vector<CompressedPayloadLease> acquire(
      const std::vector<File_Namespace::FileBuffer*>& source_file_buffers,
      const std::vector<size_t>& compressed_chunk_offsets,
      int8_t* compressed_device,
      const int32_t device_id,
      CudaMgr_Namespace::CudaMgr* cuda_mgr) {
    CHECK_EQ(compressed_chunk_offsets.size(), source_file_buffers.size() + size_t{1});
    std::vector<CompressedPayloadLease> leases(source_file_buffers.size());
    std::lock_guard<std::mutex> map_lock(mutex_);
    for (size_t chunk_idx = 0; chunk_idx < source_file_buffers.size(); ++chunk_idx) {
      auto source_file_buffer = source_file_buffers[chunk_idx];
      CHECK(source_file_buffer);
      const auto payload_bytes = source_file_buffer->storageCompressedSize();
      CHECK_EQ(
          compressed_chunk_offsets[chunk_idx + 1] - compressed_chunk_offsets[chunk_idx],
          payload_bytes);

      const auto state_it = states_.find(source_file_buffer);
      if (state_it != states_.end()) {
        auto state = state_it->second;
        std::lock_guard<std::mutex> state_lock(state->mutex);
        const bool can_copy = state->source_device_id == device_id ||
                              cuda_mgr->canAccessPeer(device_id, state->source_device_id);
        if (state->accepting_consumers && !state->failed &&
            state->payload_bytes == payload_bytes && can_copy) {
          ++state->active_consumers;
          leases[chunk_idx] = {CompressedPayloadLeaseRole::Consumer, std::move(state)};
        }
        continue;
      }

      auto state = std::make_shared<CompressedPayloadState>();
      state->source_file_buffer = source_file_buffer;
      state->source_device_id = device_id;
      state->source_device_ptr = compressed_device + compressed_chunk_offsets[chunk_idx];
      state->payload_bytes = payload_bytes;
      states_.emplace(source_file_buffer, state);
      leases[chunk_idx] = {CompressedPayloadLeaseRole::Producer, std::move(state)};
    }
    return leases;
  }

  void publish(const std::vector<CompressedPayloadLease>& leases) {
    for (const auto& lease : leases) {
      if (lease.role != CompressedPayloadLeaseRole::Producer) {
        continue;
      }
      CHECK(lease.state);
      {
        std::lock_guard<std::mutex> state_lock(lease.state->mutex);
        CHECK(!lease.state->failed);
        lease.state->ready = true;
      }
      lease.state->cv.notify_all();
    }
  }

  bool waitForConsumer(const CompressedPayloadLease& lease,
                       const int8_t*& source_device_ptr,
                       int32_t& source_device_id) {
    CHECK(lease.role == CompressedPayloadLeaseRole::Consumer);
    CHECK(lease.state);
    std::unique_lock<std::mutex> state_lock(lease.state->mutex);
    lease.state->cv.wait(state_lock,
                         [&] { return lease.state->ready || lease.state->failed; });
    if (lease.state->failed) {
      return false;
    }
    source_device_ptr = lease.state->source_device_ptr;
    source_device_id = lease.state->source_device_id;
    return true;
  }

  void releaseConsumers(const std::vector<CompressedPayloadLease>& leases) noexcept {
    for (const auto& lease : leases) {
      if (lease.role != CompressedPayloadLeaseRole::Consumer || !lease.state) {
        continue;
      }
      {
        std::lock_guard<std::mutex> state_lock(lease.state->mutex);
        CHECK_GT(lease.state->active_consumers, size_t{0});
        --lease.state->active_consumers;
      }
      lease.state->cv.notify_all();
    }
  }

  void finishProducers(const std::vector<CompressedPayloadLease>& leases) noexcept {
    std::vector<std::shared_ptr<CompressedPayloadState>> producer_states;
    {
      std::lock_guard<std::mutex> map_lock(mutex_);
      for (const auto& lease : leases) {
        if (lease.role != CompressedPayloadLeaseRole::Producer || !lease.state) {
          continue;
        }
        {
          std::lock_guard<std::mutex> state_lock(lease.state->mutex);
          lease.state->accepting_consumers = false;
          if (!lease.state->ready) {
            lease.state->failed = true;
          }
        }
        const auto state_it = states_.find(lease.state->source_file_buffer);
        if (state_it != states_.end() && state_it->second == lease.state) {
          states_.erase(state_it);
        }
        producer_states.push_back(lease.state);
      }
    }

    for (const auto& state : producer_states) {
      state->cv.notify_all();
      std::unique_lock<std::mutex> state_lock(state->mutex);
      state->cv.wait(state_lock, [&] { return state->active_consumers == 0; });
    }
  }

 private:
  std::mutex mutex_;
  std::unordered_map<File_Namespace::FileBuffer*, std::shared_ptr<CompressedPayloadState>>
      states_;
};

CompressedPayloadExchange& compressed_payload_exchange() {
  static CompressedPayloadExchange exchange;
  return exchange;
}

int8_t* ensure_device_workspace(CompressedGpuInputWorkspace& workspace,
                                const size_t required_bytes) {
  if (required_bytes == 0) {
    return nullptr;
  }
  if (workspace.capacity >= required_bytes) {
    return workspace.device;
  }
  workspace.releaseAll();
  workspace.buffer = workspace.buffer_mgr->alloc(required_bytes);
  workspace.device = workspace.buffer->getMemoryPtr();
  workspace.capacity = workspace.buffer->reservedSize();
  return workspace.device;
}

#endif

struct MappedFileRegion {
  explicit MappedFileRegion(File_Namespace::FileInfo* file_info)
      : file_info(file_info)
      , read_lock(file_info->readWriteMutex_)
      , length(file_info->size()) {
    CHECK(file_info);
    CHECK(file_info->f);
    const auto fd = fileno(file_info->f);
    if (fd < 0) {
      throw std::runtime_error("Could not get file descriptor for " +
                               file_info->file_path);
    }
    base = mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
      base = nullptr;
      throw std::runtime_error("Could not mmap " + file_info->file_path + ": " +
                               strerror(errno));
    }
  }

  ~MappedFileRegion() {
    if (base) {
      munmap(base, length);
    }
  }

  const int8_t* ptr(const size_t offset) const {
    CHECK(base);
    CHECK_LE(offset, length);
    return static_cast<const int8_t*>(base) + offset;
  }

  File_Namespace::FileInfo* file_info;
  std::shared_lock<std::shared_mutex> read_lock;
  size_t length;
  void* base{nullptr};
};

struct GpuMmapCopySpan {
  File_Namespace::FileBufferReadSpan span;
  int8_t* destination_base{nullptr};
  size_t destination_size{0};
};

#ifdef HAVE_NVCOMP
struct HostReadSegment {
  File_Namespace::FileInfo* file_info{nullptr};
  size_t file_offset{0};
  size_t host_offset{0};
  size_t num_bytes{0};
};

int get_device_host_numa_node(CudaMgr_Namespace::CudaMgr* cuda_mgr,
                              const int32_t device_id) {
#if defined(__linux__) && defined(CUDA_VERSION) && CUDA_VERSION >= 12090
  int host_numa_id{-1};
  const auto status =
      cuDeviceGetAttribute(&host_numa_id,
                           CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                           cuda_mgr->getDeviceProperties(device_id)->device);
  return status == CUDA_SUCCESS ? host_numa_id : -1;
#else
  return -1;
#endif
}

void pin_current_thread_to_numa_node(const int numa_node) noexcept {
#ifdef __linux__
  if (numa_node < 0) {
    return;
  }
  try {
    std::ifstream cpulist_file("/sys/devices/system/node/node" +
                               std::to_string(numa_node) + "/cpulist");
    std::string cpulist;
    if (!cpulist_file || !std::getline(cpulist_file, cpulist)) {
      return;
    }

    cpu_set_t allowed_cpus;
    cpu_set_t target_cpus;
    CPU_ZERO(&allowed_cpus);
    CPU_ZERO(&target_cpus);
    if (sched_getaffinity(0, sizeof(allowed_cpus), &allowed_cpus) != 0) {
      return;
    }

    size_t begin = 0;
    while (begin < cpulist.size()) {
      const auto end = cpulist.find(',', begin);
      const auto token = cpulist.substr(begin, end - begin);
      const auto dash = token.find('-');
      const auto first_cpu = std::stoul(token.substr(0, dash));
      const auto last_cpu =
          dash == std::string::npos ? first_cpu : std::stoul(token.substr(dash + 1));
      for (auto cpu = first_cpu; cpu <= last_cpu && cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed_cpus)) {
          CPU_SET(cpu, &target_cpus);
        }
      }
      if (end == std::string::npos) {
        break;
      }
      begin = end + 1;
    }
    if (CPU_COUNT(&target_cpus) == 0) {
      return;
    }
    const auto status =
        pthread_setaffinity_np(pthread_self(), sizeof(target_cpus), &target_cpus);
    if (status != 0) {
      VLOG(1) << "Could not bind GPU input reader to NUMA node " << numa_node
              << ": error=" << status;
    }
  } catch (const std::exception& error) {
    VLOG(1) << "Could not resolve GPU input reader NUMA affinity: " << error.what();
  }
#else
  (void)numa_node;
#endif
}

void append_overlapping_read_segments(std::vector<HostReadSegment>& read_segments,
                                      const File_Namespace::FileBufferReadSpan& span,
                                      const size_t range_begin,
                                      const size_t range_end) {
  CHECK_LE(range_begin, range_end);
  if (span.height == 0) {
    return;
  }
  for (size_t row_idx = 0; row_idx < span.height; ++row_idx) {
    const size_t row_destination_begin = checked_size_add(
        span.destination_offset,
        checked_size_multiply(
            row_idx, span.destination_pitch, "GPU input row offset overflow"),
        "GPU input destination offset overflow");
    const size_t row_destination_end = checked_size_add(
        row_destination_begin, span.width_bytes, "GPU input row size overflow");
    const size_t read_begin = std::max(range_begin, row_destination_begin);
    const size_t read_end = std::min(range_end, row_destination_end);
    if (read_begin >= read_end) {
      continue;
    }

    const size_t row_offset = read_begin - row_destination_begin;
    const auto file_offset = checked_size_add(
        checked_size_add(
            span.file_offset,
            checked_size_multiply(
                row_idx, span.source_pitch, "GPU input source row offset overflow"),
            "GPU input source offset overflow"),
        row_offset,
        "GPU input source offset overflow");
    read_segments.push_back(
        {span.file_info, file_offset, read_begin - range_begin, read_end - read_begin});
  }
}

std::vector<HostReadSegment> collect_read_segments_for_destination_range(
    const std::vector<File_Namespace::FileBufferReadSpan>& spans,
    const size_t range_begin,
    const size_t num_bytes) {
  const size_t range_end =
      checked_size_add(range_begin, num_bytes, "GPU input read range overflow");
  std::vector<HostReadSegment> read_segments;
  for (const auto& span : spans) {
    if (span.height == 0) {
      continue;
    }
    const size_t span_begin = span.destination_offset;
    const size_t span_end = checked_size_add(
        checked_size_add(
            span.destination_offset,
            checked_size_multiply(
                span.destination_pitch, span.height - 1, "GPU input span size overflow"),
            "GPU input span offset overflow"),
        span.width_bytes,
        "GPU input span size overflow");
    if (span_end <= range_begin) {
      continue;
    }
    if (span_begin >= range_end) {
      break;
    }
    append_overlapping_read_segments(read_segments, span, range_begin, range_end);
  }
  return read_segments;
}

class HostReadWorkerPool {
 public:
  HostReadWorkerPool(const size_t num_reader_threads, const int numa_node) {
    const auto helper_thread_count =
        std::max<size_t>(num_reader_threads, size_t(1)) - size_t(1);
    workers_.reserve(helper_thread_count);
    try {
      for (size_t thread_idx = 0; thread_idx < helper_thread_count; ++thread_idx) {
        workers_.emplace_back([this, numa_node] {
          pin_current_thread_to_numa_node(numa_node);
          workerLoop();
        });
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
      }
      work_cv_.notify_all();
      for (auto& worker : workers_) {
        worker.join();
      }
      throw;
    }
  }

  ~HostReadWorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    work_cv_.notify_all();
    for (auto& worker : workers_) {
      worker.join();
    }
  }

  size_t read(int8_t* host_ptr, const std::vector<HostReadSegment>& read_segments) {
    if (read_segments.empty()) {
      return 0;
    }
    if (workers_.empty()) {
      size_t bytes_read = 0;
      for (const auto& read_segment : read_segments) {
        bytes_read = checked_size_add(
            bytes_read,
            read_segment.file_info->read(read_segment.file_offset,
                                         read_segment.num_bytes,
                                         host_ptr + read_segment.host_offset),
            "GPU input bytes-read counter overflow");
      }
      return bytes_read;
    }

    std::lock_guard<std::mutex> read_lock(read_mutex_);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      CHECK(!read_segments_);
      host_ptr_ = host_ptr;
      read_segments_ = &read_segments;
      next_segment_idx_.store(0);
      completed_participants_ = 0;
      total_bytes_read_ = 0;
      read_exception_ = nullptr;
      ++generation_;
    }
    work_cv_.notify_all();

    consumeSegments();

    std::unique_lock<std::mutex> lock(mutex_);
    completed_cv_.wait(
        lock, [&] { return completed_participants_ == workers_.size() + size_t(1); });
    const auto bytes_read = total_bytes_read_;
    const auto read_exception = read_exception_;
    host_ptr_ = nullptr;
    read_segments_ = nullptr;
    lock.unlock();
    if (read_exception) {
      std::rethrow_exception(read_exception);
    }
    return bytes_read;
  }

 private:
  void workerLoop() noexcept {
    size_t observed_generation = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait(lock, [&] { return stop_ || generation_ != observed_generation; });
        if (stop_) {
          return;
        }
        observed_generation = generation_;
      }
      consumeSegments();
    }
  }

  void consumeSegments() noexcept {
    size_t bytes_read = 0;
    std::exception_ptr read_exception;
    try {
      while (true) {
        const auto segment_idx = next_segment_idx_.fetch_add(1);
        if (segment_idx >= read_segments_->size()) {
          break;
        }
        const auto& read_segment = (*read_segments_)[segment_idx];
        bytes_read = checked_size_add(
            bytes_read,
            read_segment.file_info->read(read_segment.file_offset,
                                         read_segment.num_bytes,
                                         host_ptr_ + read_segment.host_offset),
            "GPU input bytes-read counter overflow");
      }
    } catch (...) {
      read_exception = std::current_exception();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (read_exception && !read_exception_) {
      read_exception_ = read_exception;
    }
    if (bytes_read > std::numeric_limits<size_t>::max() - total_bytes_read_) {
      if (!read_exception_) {
        read_exception_ = std::make_exception_ptr(
            std::overflow_error("GPU input bytes-read counter overflow"));
      }
    } else {
      total_bytes_read_ += bytes_read;
    }
    ++completed_participants_;
    if (completed_participants_ == workers_.size() + size_t(1)) {
      completed_cv_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex read_mutex_;
  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable completed_cv_;
  bool stop_{false};
  size_t generation_{0};
  int8_t* host_ptr_{nullptr};
  const std::vector<HostReadSegment>* read_segments_{nullptr};
  std::atomic<size_t> next_segment_idx_{0};
  size_t completed_participants_{0};
  size_t total_bytes_read_{0};
  std::exception_ptr read_exception_;
};

HostReadWorkerPool& get_host_read_worker_pool(const int32_t device_id,
                                              const size_t num_reader_threads,
                                              CudaMgr_Namespace::CudaMgr* cuda_mgr) {
  const auto normalized_reader_threads = std::max<size_t>(num_reader_threads, size_t(1));
  using PoolKey = std::pair<int32_t, size_t>;
  const PoolKey key{device_id, normalized_reader_threads};
  static std::mutex pool_mutex;
  static std::map<PoolKey, std::unique_ptr<HostReadWorkerPool>> pools;

  std::lock_guard<std::mutex> lock(pool_mutex);
  const auto pool_it = pools.find(key);
  if (pool_it != pools.end()) {
    return *pool_it->second;
  }
  auto pool = std::make_unique<HostReadWorkerPool>(
      normalized_reader_threads, get_device_host_numa_node(cuda_mgr, device_id));
  auto [inserted_it, inserted] = pools.emplace(key, std::move(pool));
  CHECK(inserted);
  return *inserted_it->second;
}

bool copy_read_spans_to_gpu_with_pinned_producer(
    const std::vector<File_Namespace::FileBufferReadSpan>& spans,
    int8_t* destination_base,
    const size_t destination_size,
    const int32_t device_id,
    CudaMgr_Namespace::CudaMgr* cuda_mgr,
    CUstream cuda_stream,
    const char* copy_label,
    size_t& copied_bytes) {
  auto& reader_pool = get_host_read_worker_pool(
      device_id, g_gpu_input_cpu_buffer_bypass_reader_threads, cuda_mgr);
  size_t produced_bytes = 0;
  const bool used_pinned_transfer = cuda_mgr->copyHostToDeviceFromPinnedProducer(
      destination_base,
      destination_size,
      device_id,
      copy_label,
      [&](int8_t* host_ptr, size_t bytes_to_copy, size_t producer_offset) {
        const auto read_segments = collect_read_segments_for_destination_range(
            spans, producer_offset, bytes_to_copy);
        const auto bytes_read = reader_pool.read(host_ptr, read_segments);
        produced_bytes = checked_size_add(
            produced_bytes, bytes_read, "GPU input produced-byte counter overflow");
        CHECK_EQ(bytes_read, bytes_to_copy);
      },
      cuda_stream);
  if (!used_pinned_transfer) {
    return false;
  }

  copied_bytes = produced_bytes;
  return true;
}
#endif

bool copy_mmap_spans_to_gpu(const std::vector<GpuMmapCopySpan>& copy_spans,
                            const int32_t device_id,
                            CudaMgr_Namespace::CudaMgr* cuda_mgr,
                            const char* copy_label,
                            size_t& copied_bytes) {
  try {
    std::unordered_map<File_Namespace::FileInfo*, std::unique_ptr<MappedFileRegion>>
        mappings;
    mappings.reserve(copy_spans.size());
    for (const auto& copy_span : copy_spans) {
      if (mappings.find(copy_span.span.file_info) == mappings.end()) {
        mappings.emplace(copy_span.span.file_info,
                         std::make_unique<MappedFileRegion>(copy_span.span.file_info));
      }
    }
    copied_bytes = 0;
    for (const auto& copy_span : copy_spans) {
      const auto& span = copy_span.span;
      if (span.height == 0) {
        continue;
      }
      auto mapping = mappings.at(span.file_info).get();
      const auto source_end = checked_size_add(
          checked_size_add(
              span.file_offset,
              checked_size_multiply(
                  span.height - 1, span.source_pitch, "GPU mmap source span overflow"),
              "GPU mmap source span overflow"),
          span.width_bytes,
          "GPU mmap source span overflow");
      const auto destination_end = checked_size_add(
          checked_size_add(span.destination_offset,
                           checked_size_multiply(span.height - 1,
                                                 span.destination_pitch,
                                                 "GPU mmap destination span overflow"),
                           "GPU mmap destination span overflow"),
          span.width_bytes,
          "GPU mmap destination span overflow");
      CHECK_LE(source_end, mapping->length);
      CHECK_LE(destination_end, copy_span.destination_size);
      const auto source = mapping->ptr(span.file_offset);
      auto destination = copy_span.destination_base + span.destination_offset;
      if (span.height == 1) {
        cuda_mgr->copyHostToDeviceDirect(
            destination, source, span.width_bytes, device_id, copy_label);
      } else {
        cuda_mgr->copyHostToDevice2DDirect(destination,
                                           span.destination_pitch,
                                           source,
                                           span.source_pitch,
                                           span.width_bytes,
                                           span.height,
                                           device_id,
                                           copy_label);
      }
      copied_bytes = checked_size_add(
          copied_bytes, span.bytes(), "GPU mmap copied-byte counter overflow");
    }
    return true;
  } catch (const std::exception& e) {
    VLOG(1) << "Falling back from mmap GPU input bypass: " << e.what();
    return false;
  }
}

bool copy_read_spans_to_gpu_with_mmap(
    const std::vector<File_Namespace::FileBufferReadSpan>& spans,
    int8_t* destination_base,
    const size_t destination_size,
    const int32_t device_id,
    CudaMgr_Namespace::CudaMgr* cuda_mgr,
    const char* copy_label,
    size_t& copied_bytes) {
  std::vector<GpuMmapCopySpan> copy_spans;
  copy_spans.reserve(spans.size());
  for (const auto& span : spans) {
    copy_spans.push_back({span, destination_base, destination_size});
  }
  return copy_mmap_spans_to_gpu(
      copy_spans, device_id, cuda_mgr, copy_label, copied_bytes);
}

bool copy_file_buffer_to_gpu_with_mmap(File_Namespace::FileBuffer* source_file_buffer,
                                       AbstractBuffer* dest_buffer,
                                       CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                       const size_t existing_dest_size,
                                       const size_t chunk_size) {
  const auto spans = source_file_buffer->getReadSpans(chunk_size - existing_dest_size,
                                                      existing_dest_size);
  size_t copied_bytes = 0;
  if (copy_read_spans_to_gpu_with_mmap(spans,
                                       dest_buffer->getMemoryPtr() + existing_dest_size,
                                       chunk_size - existing_dest_size,
                                       dest_buffer->getDeviceId(),
                                       cuda_mgr,
                                       "GpuInputMmapBypass",
                                       copied_bytes)) {
    return copied_bytes == chunk_size - existing_dest_size;
  }
  return false;
}

bool copy_file_buffers_to_gpu_with_mmap(
    const std::vector<File_Namespace::FileBuffer*>& source_file_buffers,
    const std::vector<AbstractBuffer*>& dest_buffers,
    CudaMgr_Namespace::CudaMgr* cuda_mgr) {
  CHECK_EQ(source_file_buffers.size(), dest_buffers.size());
  if (source_file_buffers.empty()) {
    return true;
  }

  const auto device_id = dest_buffers.front()->getDeviceId();
  std::vector<GpuMmapCopySpan> copy_spans;
  size_t total_bytes = 0;
  for (size_t chunk_idx = 0; chunk_idx < source_file_buffers.size(); ++chunk_idx) {
    auto source_file_buffer = source_file_buffers[chunk_idx];
    auto dest_buffer = dest_buffers[chunk_idx];
    CHECK(source_file_buffer);
    CHECK(dest_buffer);
    CHECK_EQ(dest_buffer->getDeviceId(), device_id);
    CHECK(!source_file_buffer->isStorageCompressed());
    if (dest_buffer->size() != 0) {
      VLOG(1) << "Falling back from batched mmap GPU input fetch for partial "
                 "destination chunk";
      return false;
    }

    const auto chunk_size = source_file_buffer->size();
    const auto spans = source_file_buffer->getReadSpans(chunk_size, 0);
    for (const auto& span : spans) {
      copy_spans.push_back({span, dest_buffer->getMemoryPtr(), chunk_size});
    }
    total_bytes =
        checked_size_add(total_bytes, chunk_size, "Batched GPU input size overflow");
  }

  size_t copied_bytes = 0;
  if (!copy_mmap_spans_to_gpu(
          copy_spans, device_id, cuda_mgr, "GpuInputMmapBatchBypass", copied_bytes)) {
    return false;
  }
  CHECK_EQ(copied_bytes, total_bytes);
  return true;
}

#ifdef HAVE_NVCOMP
constexpr bool kNativeStorageRawLz4SupportsNvcompDecode{false};

const char* nvcomp_status_name(const nvcompStatus_t status) {
  switch (status) {
    case nvcompSuccess:
      return "nvcompSuccess";
    case nvcompErrorInvalidValue:
      return "nvcompErrorInvalidValue";
    case nvcompErrorNotSupported:
      return "nvcompErrorNotSupported";
    case nvcompErrorCannotDecompress:
      return "nvcompErrorCannotDecompress";
    case nvcompErrorBadChecksum:
      return "nvcompErrorBadChecksum";
    case nvcompErrorOutputBufferTooSmall:
      return "nvcompErrorOutputBufferTooSmall";
    case nvcompErrorWrongHeaderLength:
      return "nvcompErrorWrongHeaderLength";
    case nvcompErrorAlignment:
      return "nvcompErrorAlignment";
    case nvcompErrorChunkSizeTooLarge:
      return "nvcompErrorChunkSizeTooLarge";
    case nvcompErrorCudaError:
      return "nvcompErrorCudaError";
    default:
      return "nvcompErrorOther";
  }
}

bool copy_compressed_file_buffer_to_gpu_with_nvcomp(
    File_Namespace::FileBuffer* source_file_buffer,
    AbstractBuffer* dest_buffer,
    CudaMgr_Namespace::CudaMgr* cuda_mgr,
    CompressedGpuInputWorkspace& workspace,
    const size_t existing_dest_size,
    const size_t chunk_size) {
  CHECK(source_file_buffer->isStorageCompressed());
  const bool use_lz4 = source_file_buffer->isLz4StorageCompressed();
  const bool use_snappy = source_file_buffer->isSnappyStorageCompressed();
  const bool use_gdeflate = source_file_buffer->isGdeflateStorageCompressed();
  const bool use_bitcomp = source_file_buffer->isBitcompStorageCompressed();
  if (!use_lz4 && !use_snappy && !use_gdeflate && !use_bitcomp) {
    return false;
  }
#ifndef HAVE_NVCOMP_GDEFLATE
  if (use_gdeflate) {
    return false;
  }
#endif
  if (use_lz4 && !kNativeStorageRawLz4SupportsNvcompDecode) {
    // Native storage compression currently writes raw CPU LZ4 blocks via
    // LZ4_compress_default.  The nvCOMP LZ4 GPU path accepts these buffers but
    // can silently produce different values, so keep the correctness-preserving
    // CPU decompression fallback until the writer records an nvCOMP-compatible
    // native bitstream format.
    return false;
  }
  if (existing_dest_size != 0 || chunk_size != source_file_buffer->size()) {
    VLOG(1) << "Falling back from compressed GPU input bypass for partial chunk fetch";
    return false;
  }

  const auto& frame_sizes = source_file_buffer->storageCompressedFrameSizes();
  const auto compressed_size = source_file_buffer->storageCompressedSize();
  const auto frame_uncompressed_size = source_file_buffer->storageCompressionFrameSize();
  if (frame_sizes.empty() || compressed_size == 0 || frame_uncompressed_size == 0) {
    return false;
  }
  size_t compressed_size_sum = 0;
  for (const auto frame_size : frame_sizes) {
    compressed_size_sum = checked_size_add(
        compressed_size_sum, frame_size, "Compressed frame table size overflow");
  }
  if (compressed_size != compressed_size_sum) {
    LOG(WARNING) << "Compressed native storage frame sizes do not sum to payload size";
    return false;
  }

  const auto device_id = dest_buffer->getDeviceId();
  CHECK_EQ(workspace.device_id, device_id);
  cuda_mgr->setContext(device_id);
  const auto transfer_stream = cuda_mgr->getDeviceTransferStream(device_id);

  const auto frame_count = frame_sizes.size();
  const auto pointer_array_bytes = checked_size_multiply(
      frame_count, sizeof(void*), "Compressed GPU input pointer array overflow");
  const auto size_array_bytes = checked_size_multiply(
      frame_count, sizeof(size_t), "Compressed GPU input size array overflow");
  const auto status_array_bytes = checked_size_multiply(
      frame_count, sizeof(nvcompStatus_t), "Compressed GPU input status array overflow");
  std::vector<const void*> compressed_ptrs(frame_count);
  std::vector<size_t> compressed_offsets(frame_count);
  std::vector<void*> output_ptrs(frame_count);
  std::vector<size_t> output_sizes(frame_count);
  size_t compressed_offset = 0;
  size_t uncompressed_offset = 0;
  for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
    if (uncompressed_offset >= chunk_size) {
      return false;
    }
    const auto compressed_end = checked_size_add(
        compressed_offset, frame_sizes[frame_idx], "Compressed frame offset overflow");
    if (compressed_end > compressed_size) {
      return false;
    }
    compressed_offsets[frame_idx] = compressed_offset;
    output_ptrs[frame_idx] = dest_buffer->getMemoryPtr() + uncompressed_offset;
    output_sizes[frame_idx] =
        std::min(frame_uncompressed_size, chunk_size - uncompressed_offset);
    compressed_offset = compressed_end;
    uncompressed_offset = checked_size_add(uncompressed_offset,
                                           output_sizes[frame_idx],
                                           "Uncompressed frame offset overflow");
  }
  CHECK_EQ(compressed_offset, compressed_size);
  CHECK_EQ(uncompressed_offset, chunk_size);

  size_t temp_bytes = 0;
  nvcompStatus_t status = nvcompSuccess;
  nvcompBatchedLZ4DecompressOpts_t lz4_decompress_opts{};
  nvcompBatchedSnappyDecompressOpts_t snappy_decompress_opts{};
  nvcompBatchedBitcompDecompressOpts_t bitcomp_decompress_opts{};
#ifdef HAVE_NVCOMP_GDEFLATE
  nvcompBatchedGdeflateDecompressOpts_t gdeflate_decompress_opts{};
#endif
  if (use_snappy) {
    snappy_decompress_opts = nvcompBatchedSnappyDecompressDefaultOpts;
    snappy_decompress_opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    status = nvcompBatchedSnappyDecompressGetTempSizeAsync(frame_count,
                                                           frame_uncompressed_size,
                                                           snappy_decompress_opts,
                                                           &temp_bytes,
                                                           chunk_size);
  } else if (use_bitcomp) {
    bitcomp_decompress_opts = nvcompBatchedBitcompDecompressDefaultOpts;
    bitcomp_decompress_opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    status = nvcompBatchedBitcompDecompressGetTempSizeAsync(frame_count,
                                                            frame_uncompressed_size,
                                                            bitcomp_decompress_opts,
                                                            &temp_bytes,
                                                            chunk_size);
  }
#ifdef HAVE_NVCOMP_GDEFLATE
  else if (use_gdeflate) {
    gdeflate_decompress_opts = nvcompBatchedGdeflateDecompressDefaultOpts;
    gdeflate_decompress_opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    status = nvcompBatchedGdeflateDecompressGetTempSizeAsync(frame_count,
                                                             frame_uncompressed_size,
                                                             gdeflate_decompress_opts,
                                                             &temp_bytes,
                                                             chunk_size);
  }
#endif
  else {
    lz4_decompress_opts = nvcompBatchedLZ4DecompressDefaultOpts;
    lz4_decompress_opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    status = nvcompBatchedLZ4DecompressGetTempSizeAsync(frame_count,
                                                        frame_uncompressed_size,
                                                        lz4_decompress_opts,
                                                        &temp_bytes,
                                                        chunk_size);
  }
  if (status != nvcompSuccess) {
    VLOG(1) << "nvCOMP compressed native storage temp-size request failed: "
            << nvcomp_status_name(status);
    return false;
  }

  CompressedWorkspaceLayout layout;
  const auto compressed_device_offset = layout.append(compressed_size);
  const auto compressed_ptrs_device_offset = layout.append(pointer_array_bytes);
  const auto input_metadata_begin = compressed_ptrs_device_offset;
  const auto compressed_sizes_device_offset = layout.append(size_array_bytes);
  const auto output_sizes_device_offset = layout.append(size_array_bytes);
  const auto output_ptrs_device_offset = layout.append(pointer_array_bytes);
  const auto input_metadata_end = layout.total_bytes;
  const auto actual_output_sizes_device_offset = layout.append(size_array_bytes);
  const auto output_metadata_begin = actual_output_sizes_device_offset;
  const auto statuses_device_offset = layout.append(status_array_bytes);
  const auto output_metadata_end = layout.total_bytes;
  const auto temp_device_offset = layout.append(temp_bytes);

  auto workspace_device = ensure_device_workspace(workspace, layout.total_bytes);
  auto compressed_device = workspace_device + compressed_device_offset;
  for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
    compressed_ptrs[frame_idx] = compressed_device + compressed_offsets[frame_idx];
  }

  bool copied_compressed_payload = false;
  const auto spans = source_file_buffer->getCompressedReadSpans();
  size_t copied_bytes = 0;
  const std::string compressed_copy_label =
      std::string("GpuInputCompressed") +
      (use_snappy ? "Snappy"
                  : (use_bitcomp ? "Bitcomp" : (use_gdeflate ? "Gdeflate" : "Lz4")));
  const auto pinned_copy_label = compressed_copy_label + "Pinned";
  copied_compressed_payload =
      copy_read_spans_to_gpu_with_pinned_producer(spans,
                                                  compressed_device,
                                                  compressed_size,
                                                  device_id,
                                                  cuda_mgr,
                                                  transfer_stream,
                                                  pinned_copy_label.c_str(),
                                                  copied_bytes);
  if (copied_compressed_payload) {
    CHECK_EQ(copied_bytes, compressed_size);
  }

  if (!copied_compressed_payload && use_mmap_gpu_input_bypass()) {
    copied_bytes = 0;
    const auto mmap_copy_label = compressed_copy_label + "Mmap";
    copied_compressed_payload = copy_read_spans_to_gpu_with_mmap(spans,
                                                                 compressed_device,
                                                                 compressed_size,
                                                                 device_id,
                                                                 cuda_mgr,
                                                                 mmap_copy_label.c_str(),
                                                                 copied_bytes);
    if (copied_compressed_payload) {
      CHECK_EQ(copied_bytes, compressed_size);
    }
  }

  if (!copied_compressed_payload) {
    std::vector<int8_t> compressed_staging(compressed_size);
    source_file_buffer->readCompressedPayloadWithReaderThreads(
        compressed_staging.data(),
        std::max<size_t>(g_gpu_input_cpu_buffer_bypass_reader_threads, size_t(1)));
    cuda_mgr->copyHostToDeviceDirect(compressed_device,
                                     compressed_staging.data(),
                                     compressed_size,
                                     device_id,
                                     compressed_copy_label,
                                     transfer_stream);
  }

  std::vector<int8_t> input_metadata(input_metadata_end - input_metadata_begin);
  const auto pack_input_metadata = [&](const size_t device_offset,
                                       const void* host_ptr,
                                       const size_t bytes) {
    CHECK_GE(device_offset, input_metadata_begin);
    CHECK_LE(checked_size_add(
                 device_offset, bytes, "Compressed GPU input metadata range overflow"),
             input_metadata_end);
    memcpy(input_metadata.data() + device_offset - input_metadata_begin, host_ptr, bytes);
  };
  pack_input_metadata(
      compressed_ptrs_device_offset, compressed_ptrs.data(), pointer_array_bytes);
  pack_input_metadata(
      compressed_sizes_device_offset, frame_sizes.data(), size_array_bytes);
  pack_input_metadata(output_sizes_device_offset, output_sizes.data(), size_array_bytes);
  pack_input_metadata(output_ptrs_device_offset, output_ptrs.data(), pointer_array_bytes);
  cuda_mgr->copyHostToDeviceDirect(workspace_device + input_metadata_begin,
                                   input_metadata.data(),
                                   input_metadata.size(),
                                   device_id,
                                   compressed_copy_label + "Metadata",
                                   transfer_stream);

  auto compressed_ptrs_device = workspace_device + compressed_ptrs_device_offset;
  auto compressed_sizes_device = workspace_device + compressed_sizes_device_offset;
  auto output_sizes_device = workspace_device + output_sizes_device_offset;
  auto output_ptrs_device = workspace_device + output_ptrs_device_offset;
  auto actual_output_sizes_device = workspace_device + actual_output_sizes_device_offset;
  auto statuses_device = workspace_device + statuses_device_offset;
  auto temp_device = temp_bytes ? workspace_device + temp_device_offset : nullptr;

  if (use_snappy) {
    status = nvcompBatchedSnappyDecompressAsync(
        reinterpret_cast<const void* const*>(compressed_ptrs_device),
        reinterpret_cast<const size_t*>(compressed_sizes_device),
        reinterpret_cast<const size_t*>(output_sizes_device),
        reinterpret_cast<size_t*>(actual_output_sizes_device),
        frame_count,
        temp_device,
        temp_bytes,
        reinterpret_cast<void* const*>(output_ptrs_device),
        snappy_decompress_opts,
        reinterpret_cast<nvcompStatus_t*>(statuses_device),
        transfer_stream);
  } else if (use_bitcomp) {
    status = nvcompBatchedBitcompDecompressAsync(
        reinterpret_cast<const void* const*>(compressed_ptrs_device),
        reinterpret_cast<const size_t*>(compressed_sizes_device),
        reinterpret_cast<const size_t*>(output_sizes_device),
        reinterpret_cast<size_t*>(actual_output_sizes_device),
        frame_count,
        temp_device,
        temp_bytes,
        reinterpret_cast<void* const*>(output_ptrs_device),
        bitcomp_decompress_opts,
        reinterpret_cast<nvcompStatus_t*>(statuses_device),
        transfer_stream);
  }
#ifdef HAVE_NVCOMP_GDEFLATE
  else if (use_gdeflate) {
    status = nvcompBatchedGdeflateDecompressAsync(
        reinterpret_cast<const void* const*>(compressed_ptrs_device),
        reinterpret_cast<const size_t*>(compressed_sizes_device),
        reinterpret_cast<const size_t*>(output_sizes_device),
        reinterpret_cast<size_t*>(actual_output_sizes_device),
        frame_count,
        temp_device,
        temp_bytes,
        reinterpret_cast<void* const*>(output_ptrs_device),
        gdeflate_decompress_opts,
        reinterpret_cast<nvcompStatus_t*>(statuses_device),
        transfer_stream);
  }
#endif
  else {
    status = nvcompBatchedLZ4DecompressAsync(
        reinterpret_cast<const void* const*>(compressed_ptrs_device),
        reinterpret_cast<const size_t*>(compressed_sizes_device),
        reinterpret_cast<const size_t*>(output_sizes_device),
        reinterpret_cast<size_t*>(actual_output_sizes_device),
        frame_count,
        temp_device,
        temp_bytes,
        reinterpret_cast<void* const*>(output_ptrs_device),
        lz4_decompress_opts,
        reinterpret_cast<nvcompStatus_t*>(statuses_device),
        transfer_stream);
  }
  if (status != nvcompSuccess) {
    VLOG(1) << "nvCOMP compressed native storage decompression launch failed: "
            << nvcomp_status_name(status);
    return false;
  }

  std::vector<int8_t> output_metadata(output_metadata_end - output_metadata_begin);
  cuda_mgr->copyDeviceToHost(output_metadata.data(),
                             workspace_device + output_metadata_begin,
                             output_metadata.size(),
                             device_id,
                             compressed_copy_label + "ResultMetadata",
                             transfer_stream);
  std::vector<nvcompStatus_t> statuses(frame_count);
  std::vector<size_t> actual_output_sizes(frame_count);
  memcpy(
      actual_output_sizes.data(),
      output_metadata.data() + actual_output_sizes_device_offset - output_metadata_begin,
      size_array_bytes);
  memcpy(statuses.data(),
         output_metadata.data() + statuses_device_offset - output_metadata_begin,
         status_array_bytes);
  for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
    if (statuses[frame_idx] != nvcompSuccess ||
        actual_output_sizes[frame_idx] != output_sizes[frame_idx]) {
      VLOG(1) << "nvCOMP compressed native storage frame " << frame_idx
              << " failed: status=" << nvcomp_status_name(statuses[frame_idx])
              << " actual=" << actual_output_sizes[frame_idx]
              << " expected=" << output_sizes[frame_idx];
      return false;
    }
  }

  return true;
}

bool copy_compressed_file_buffers_to_gpu_with_nvcomp(
    const std::vector<File_Namespace::FileBuffer*>& source_file_buffers,
    const std::vector<AbstractBuffer*>& dest_buffers,
    CudaMgr_Namespace::CudaMgr* cuda_mgr,
    CompressedGpuInputWorkspace& workspace,
    const CUstream transfer_stream,
    std::mutex& payload_transfer_mutex) {
  CHECK_EQ(source_file_buffers.size(), dest_buffers.size());
  if (source_file_buffers.empty()) {
    return true;
  }
  const bool use_lz4 = source_file_buffers.front()->isLz4StorageCompressed();
  const bool use_snappy = source_file_buffers.front()->isSnappyStorageCompressed();
  const bool use_gdeflate = source_file_buffers.front()->isGdeflateStorageCompressed();
  const bool use_bitcomp = source_file_buffers.front()->isBitcompStorageCompressed();
  const bool bitcomp_sparse =
      use_bitcomp && source_file_buffers.front()->isBitcompSparseStorageCompressed();
  const size_t bitcomp_element_width =
      use_bitcomp ? source_file_buffers.front()->storageBitcompElementWidth() : 0;
  if (!use_lz4 && !use_snappy && !use_gdeflate && !use_bitcomp) {
    return false;
  }
#ifndef HAVE_NVCOMP_GDEFLATE
  if (use_gdeflate) {
    return false;
  }
#endif
  if (use_lz4 && !kNativeStorageRawLz4SupportsNvcompDecode) {
    // See the single-buffer path above; do not batch-decode CPU LZ4 storage
    // frames through nvCOMP until the on-disk format is GPU-decodable.
    return false;
  }

  const auto device_id = dest_buffers.front()->getDeviceId();
  CHECK_EQ(workspace.device_id, device_id);
  CHECK(transfer_stream);
  cuda_mgr->setContext(device_id);
  size_t total_compressed_size = 0;
  size_t total_uncompressed_size = 0;
  size_t total_frame_count = 0;
  size_t max_frame_uncompressed_size = 0;
  std::vector<std::vector<File_Namespace::FileBufferReadSpan>>
      compressed_chunk_read_spans;
  compressed_chunk_read_spans.reserve(source_file_buffers.size());

  for (size_t chunk_idx = 0; chunk_idx < source_file_buffers.size(); ++chunk_idx) {
    auto source_file_buffer = source_file_buffers[chunk_idx];
    auto dest_buffer = dest_buffers[chunk_idx];
    CHECK(source_file_buffer);
    CHECK(dest_buffer);
    CHECK_EQ(dest_buffer->getDeviceId(), device_id);
    CHECK(source_file_buffer->isStorageCompressed());
    if (source_file_buffer->isSnappyStorageCompressed() != use_snappy ||
        source_file_buffer->isLz4StorageCompressed() != use_lz4 ||
        source_file_buffer->isGdeflateStorageCompressed() != use_gdeflate ||
        source_file_buffer->isBitcompStorageCompressed() != use_bitcomp) {
      return false;
    }
    if (use_bitcomp &&
        (source_file_buffer->isBitcompSparseStorageCompressed() != bitcomp_sparse ||
         source_file_buffer->storageBitcompElementWidth() != bitcomp_element_width)) {
      return false;
    }
    if (dest_buffer->size() != 0) {
      VLOG(1) << "Falling back from batched compressed GPU input fetch for partial "
                 "destination chunk";
      return false;
    }

    const auto& frame_sizes = source_file_buffer->storageCompressedFrameSizes();
    const auto compressed_size = source_file_buffer->storageCompressedSize();
    const auto frame_uncompressed_size =
        source_file_buffer->storageCompressionFrameSize();
    if (frame_sizes.empty() || compressed_size == 0 || frame_uncompressed_size == 0) {
      return false;
    }
    size_t compressed_size_sum = 0;
    for (const auto frame_size : frame_sizes) {
      compressed_size_sum = checked_size_add(
          compressed_size_sum, frame_size, "Compressed frame table size overflow");
    }
    if (compressed_size != compressed_size_sum) {
      LOG(WARNING) << "Compressed native storage frame sizes do not sum to payload size";
      return false;
    }

    auto spans = source_file_buffer->getCompressedReadSpans();
    for (auto& span : spans) {
      span.destination_offset = checked_size_add(span.destination_offset,
                                                 total_compressed_size,
                                                 "Compressed GPU input span overflow");
    }
    compressed_chunk_read_spans.push_back(std::move(spans));

    total_compressed_size = checked_size_add(
        total_compressed_size, compressed_size, "Batched compressed input size overflow");
    total_uncompressed_size =
        checked_size_add(total_uncompressed_size,
                         source_file_buffer->size(),
                         "Batched uncompressed input size overflow");
    total_frame_count = checked_size_add(
        total_frame_count, frame_sizes.size(), "Batched compressed frame count overflow");
    max_frame_uncompressed_size =
        std::max(max_frame_uncompressed_size, frame_uncompressed_size);
  }

  std::vector<const void*> compressed_ptrs(total_frame_count);
  std::vector<size_t> compressed_device_offsets(total_frame_count);
  std::vector<void*> output_ptrs(total_frame_count);
  std::vector<size_t> compressed_sizes(total_frame_count);
  std::vector<size_t> output_sizes(total_frame_count);
  const auto pointer_array_bytes =
      checked_size_multiply(total_frame_count,
                            sizeof(void*),
                            "Batched compressed GPU input pointer array overflow");
  const auto size_array_bytes =
      checked_size_multiply(total_frame_count,
                            sizeof(size_t),
                            "Batched compressed GPU input size array overflow");
  const auto status_array_bytes =
      checked_size_multiply(total_frame_count,
                            sizeof(nvcompStatus_t),
                            "Batched compressed GPU input status array overflow");

  size_t frame_idx = 0;
  size_t compressed_base_offset = 0;
  std::vector<size_t> compressed_chunk_offsets(source_file_buffers.size() + size_t{1});
  for (size_t chunk_idx = 0; chunk_idx < source_file_buffers.size(); ++chunk_idx) {
    compressed_chunk_offsets[chunk_idx] = compressed_base_offset;
    auto source_file_buffer = source_file_buffers[chunk_idx];
    auto dest_buffer = dest_buffers[chunk_idx];
    const auto& frame_sizes = source_file_buffer->storageCompressedFrameSizes();
    const auto frame_uncompressed_size =
        source_file_buffer->storageCompressionFrameSize();
    const auto chunk_size = source_file_buffer->size();
    size_t compressed_offset = 0;
    size_t uncompressed_offset = 0;
    for (const auto frame_compressed_size : frame_sizes) {
      if (frame_idx >= total_frame_count) {
        return false;
      }
      if (uncompressed_offset >= chunk_size) {
        return false;
      }
      const auto compressed_end =
          checked_size_add(compressed_offset,
                           frame_compressed_size,
                           "Batched compressed frame offset overflow");
      if (compressed_end > source_file_buffer->storageCompressedSize()) {
        return false;
      }
      const auto output_size =
          std::min(frame_uncompressed_size, chunk_size - uncompressed_offset);
      const auto compressed_device_offset =
          checked_size_add(compressed_base_offset,
                           compressed_offset,
                           "Batched compressed device offset overflow");
      if (compressed_device_offset >= total_compressed_size) {
        return false;
      }
      compressed_device_offsets[frame_idx] = compressed_device_offset;
      compressed_sizes[frame_idx] = frame_compressed_size;
      output_ptrs[frame_idx] = dest_buffer->getMemoryPtr() + uncompressed_offset;
      output_sizes[frame_idx] = output_size;
      compressed_offset = compressed_end;
      uncompressed_offset = checked_size_add(
          uncompressed_offset, output_size, "Batched output frame offset overflow");
      ++frame_idx;
    }
    CHECK_EQ(compressed_offset, source_file_buffer->storageCompressedSize());
    CHECK_EQ(uncompressed_offset, chunk_size);
    compressed_base_offset = checked_size_add(compressed_base_offset,
                                              source_file_buffer->storageCompressedSize(),
                                              "Batched compressed chunk offset overflow");
  }
  compressed_chunk_offsets.back() = compressed_base_offset;
  CHECK_EQ(frame_idx, total_frame_count);
  CHECK_EQ(compressed_base_offset, total_compressed_size);
  size_t temp_bytes = 0;
  nvcompStatus_t status = nvcompSuccess;
  nvcompBatchedLZ4DecompressOpts_t lz4_decompress_opts{};
  nvcompBatchedSnappyDecompressOpts_t snappy_decompress_opts{};
  nvcompBatchedBitcompDecompressOpts_t bitcomp_decompress_opts{};
#ifdef HAVE_NVCOMP_GDEFLATE
  nvcompBatchedGdeflateDecompressOpts_t gdeflate_decompress_opts{};
#endif
  if (use_snappy) {
    snappy_decompress_opts = nvcompBatchedSnappyDecompressDefaultOpts;
    snappy_decompress_opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    status = nvcompBatchedSnappyDecompressGetTempSizeAsync(total_frame_count,
                                                           max_frame_uncompressed_size,
                                                           snappy_decompress_opts,
                                                           &temp_bytes,
                                                           total_uncompressed_size);
  } else if (use_bitcomp) {
    bitcomp_decompress_opts = nvcompBatchedBitcompDecompressDefaultOpts;
    bitcomp_decompress_opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    status = nvcompBatchedBitcompDecompressGetTempSizeAsync(total_frame_count,
                                                            max_frame_uncompressed_size,
                                                            bitcomp_decompress_opts,
                                                            &temp_bytes,
                                                            total_uncompressed_size);
  }
#ifdef HAVE_NVCOMP_GDEFLATE
  else if (use_gdeflate) {
    gdeflate_decompress_opts = nvcompBatchedGdeflateDecompressDefaultOpts;
    gdeflate_decompress_opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    status = nvcompBatchedGdeflateDecompressGetTempSizeAsync(total_frame_count,
                                                             max_frame_uncompressed_size,
                                                             gdeflate_decompress_opts,
                                                             &temp_bytes,
                                                             total_uncompressed_size);
  }
#endif
  else {
    lz4_decompress_opts = nvcompBatchedLZ4DecompressDefaultOpts;
    lz4_decompress_opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    status = nvcompBatchedLZ4DecompressGetTempSizeAsync(total_frame_count,
                                                        max_frame_uncompressed_size,
                                                        lz4_decompress_opts,
                                                        &temp_bytes,
                                                        total_uncompressed_size);
  }
  if (status != nvcompSuccess) {
    VLOG(1) << "Batched nvCOMP compressed native storage temp-size request failed: "
            << nvcomp_status_name(status);
    return false;
  }

  CompressedWorkspaceLayout layout;
  const auto compressed_device_offset = layout.append(total_compressed_size);
  const auto compressed_ptrs_device_offset = layout.append(pointer_array_bytes);
  const auto input_metadata_begin = compressed_ptrs_device_offset;
  const auto compressed_sizes_device_offset = layout.append(size_array_bytes);
  const auto output_sizes_device_offset = layout.append(size_array_bytes);
  const auto output_ptrs_device_offset = layout.append(pointer_array_bytes);
  const auto input_metadata_end = layout.total_bytes;
  const auto actual_output_sizes_device_offset = layout.append(size_array_bytes);
  const auto output_metadata_begin = actual_output_sizes_device_offset;
  const auto statuses_device_offset = layout.append(status_array_bytes);
  const auto output_metadata_end = layout.total_bytes;
  const auto temp_device_offset = layout.append(temp_bytes);
  auto workspace_device = ensure_device_workspace(workspace, layout.total_bytes);
  auto compressed_device = workspace_device + compressed_device_offset;
  for (size_t i = 0; i < total_frame_count; ++i) {
    compressed_ptrs[i] = compressed_device + compressed_device_offsets[i];
  }
  const std::string compressed_copy_label =
      std::string("GpuInputCompressed") +
      (use_snappy ? "Snappy"
                  : (use_bitcomp ? "Bitcomp" : (use_gdeflate ? "Gdeflate" : "Lz4"))) +
      "Batch";

  auto& payload_exchange = compressed_payload_exchange();
  const bool use_compressed_peer_exchange = g_enable_gpu_input_compressed_peer_exchange;
  std::vector<CompressedPayloadLease> payload_leases;
  if (use_compressed_peer_exchange) {
    payload_leases = payload_exchange.acquire(source_file_buffers,
                                              compressed_chunk_offsets,
                                              compressed_device,
                                              device_id,
                                              cuda_mgr);
  }
  ScopeGuard producer_lease_guard = [&] {
    if (use_compressed_peer_exchange) {
      payload_exchange.finishProducers(payload_leases);
    }
  };
  bool consumer_leases_released{!use_compressed_peer_exchange};
  ScopeGuard consumer_lease_guard = [&] {
    if (!consumer_leases_released) {
      payload_exchange.releaseConsumers(payload_leases);
    }
  };

  const auto copy_storage_range = [&](const size_t chunk_begin, const size_t chunk_end) {
    CHECK_LT(chunk_begin, chunk_end);
    CHECK_LE(chunk_end, source_file_buffers.size());
    const auto range_device_offset = compressed_chunk_offsets[chunk_begin];
    const auto range_bytes = compressed_chunk_offsets[chunk_end] - range_device_offset;
    CHECK_GT(range_bytes, size_t{0});

    std::vector<File_Namespace::FileBufferReadSpan> range_read_spans;
    for (size_t chunk_idx = chunk_begin; chunk_idx < chunk_end; ++chunk_idx) {
      for (auto span : compressed_chunk_read_spans[chunk_idx]) {
        CHECK_GE(span.destination_offset, range_device_offset);
        span.destination_offset -= range_device_offset;
        range_read_spans.push_back(std::move(span));
      }
    }

    std::unique_lock<std::mutex> payload_transfer_lock(payload_transfer_mutex);
    bool copied_compressed_payload{false};
    size_t copied_bytes{0};
    const auto pinned_copy_label = compressed_copy_label + "Pinned";
    copied_compressed_payload = copy_read_spans_to_gpu_with_pinned_producer(
        range_read_spans,
        compressed_device + range_device_offset,
        range_bytes,
        device_id,
        cuda_mgr,
        transfer_stream,
        pinned_copy_label.c_str(),
        copied_bytes);
    if (copied_compressed_payload) {
      CHECK_EQ(copied_bytes, range_bytes);
    }

    if (!copied_compressed_payload && use_mmap_gpu_input_bypass()) {
      copied_bytes = 0;
      const auto mmap_copy_label = compressed_copy_label + "Mmap";
      copied_compressed_payload =
          copy_read_spans_to_gpu_with_mmap(range_read_spans,
                                           compressed_device + range_device_offset,
                                           range_bytes,
                                           device_id,
                                           cuda_mgr,
                                           mmap_copy_label.c_str(),
                                           copied_bytes);
      if (copied_compressed_payload) {
        CHECK_EQ(copied_bytes, range_bytes);
      }
    }

    if (!copied_compressed_payload) {
      std::vector<int8_t> compressed_staging(range_bytes);
      size_t compressed_staging_offset = 0;
      for (size_t chunk_idx = chunk_begin; chunk_idx < chunk_end; ++chunk_idx) {
        auto source_file_buffer = source_file_buffers[chunk_idx];
        source_file_buffer->readCompressedPayloadWithReaderThreads(
            compressed_staging.data() + compressed_staging_offset,
            std::max<size_t>(g_gpu_input_cpu_buffer_bypass_reader_threads, size_t(1)));
        compressed_staging_offset =
            checked_size_add(compressed_staging_offset,
                             source_file_buffer->storageCompressedSize(),
                             "Batched compressed staging offset overflow");
      }
      CHECK_EQ(compressed_staging_offset, range_bytes);
      cuda_mgr->copyHostToDeviceDirect(compressed_device + range_device_offset,
                                       compressed_staging.data(),
                                       range_bytes,
                                       device_id,
                                       compressed_copy_label,
                                       transfer_stream);
    }
  };

  if (!use_compressed_peer_exchange) {
    copy_storage_range(0, source_file_buffers.size());
  } else {
    size_t chunk_begin{0};
    while (chunk_begin < payload_leases.size()) {
      while (chunk_begin < payload_leases.size() &&
             payload_leases[chunk_begin].role == CompressedPayloadLeaseRole::Consumer) {
        ++chunk_begin;
      }
      if (chunk_begin == payload_leases.size()) {
        break;
      }
      size_t chunk_end = chunk_begin + 1;
      while (chunk_end < payload_leases.size() &&
             payload_leases[chunk_end].role != CompressedPayloadLeaseRole::Consumer) {
        ++chunk_end;
      }
      copy_storage_range(chunk_begin, chunk_end);
      chunk_begin = chunk_end;
    }
  }
  payload_exchange.publish(payload_leases);

  bool peer_copy_enqueued{false};
  std::vector<size_t> storage_fallback_chunks;
  try {
    for (size_t chunk_idx = 0; chunk_idx < payload_leases.size(); ++chunk_idx) {
      const auto& lease = payload_leases[chunk_idx];
      if (lease.role != CompressedPayloadLeaseRole::Consumer) {
        continue;
      }
      const int8_t* source_device_ptr{nullptr};
      int32_t source_device_id{-1};
      if (!payload_exchange.waitForConsumer(lease, source_device_ptr, source_device_id)) {
        storage_fallback_chunks.push_back(chunk_idx);
        continue;
      }
      const auto payload_bytes = source_file_buffers[chunk_idx]->storageCompressedSize();
      cuda_mgr->copyPeerToPeer(compressed_device + compressed_chunk_offsets[chunk_idx],
                               source_device_ptr,
                               payload_bytes,
                               device_id,
                               source_device_id,
                               compressed_copy_label + "Peer",
                               transfer_stream,
                               false);
      peer_copy_enqueued = true;
    }
    if (peer_copy_enqueued) {
      cuda_mgr->synchronizeStream(transfer_stream);
    }
  } catch (const CudaMgr_Namespace::CudaErrorException& error) {
    VLOG(1) << "Compressed GPU input peer exchange failed; using scalar fetch: "
            << error.what();
    return false;
  }
  for (const auto chunk_idx : storage_fallback_chunks) {
    copy_storage_range(chunk_idx, chunk_idx + 1);
  }
  payload_exchange.releaseConsumers(payload_leases);
  consumer_leases_released = true;
  std::vector<int8_t> input_metadata(input_metadata_end - input_metadata_begin);
  const auto pack_input_metadata = [&](const size_t device_offset,
                                       const void* host_ptr,
                                       const size_t bytes) {
    CHECK_GE(device_offset, input_metadata_begin);
    CHECK_LE(
        checked_size_add(
            device_offset, bytes, "Batched compressed GPU input metadata range overflow"),
        input_metadata_end);
    memcpy(input_metadata.data() + device_offset - input_metadata_begin, host_ptr, bytes);
  };
  pack_input_metadata(
      compressed_ptrs_device_offset, compressed_ptrs.data(), pointer_array_bytes);
  pack_input_metadata(
      compressed_sizes_device_offset, compressed_sizes.data(), size_array_bytes);
  pack_input_metadata(output_sizes_device_offset, output_sizes.data(), size_array_bytes);
  pack_input_metadata(output_ptrs_device_offset, output_ptrs.data(), pointer_array_bytes);
  cuda_mgr->copyHostToDeviceDirect(workspace_device + input_metadata_begin,
                                   input_metadata.data(),
                                   input_metadata.size(),
                                   device_id,
                                   compressed_copy_label + "Metadata",
                                   transfer_stream);
  auto compressed_ptrs_device = workspace_device + compressed_ptrs_device_offset;
  auto compressed_sizes_device = workspace_device + compressed_sizes_device_offset;
  auto output_sizes_device = workspace_device + output_sizes_device_offset;
  auto output_ptrs_device = workspace_device + output_ptrs_device_offset;
  auto actual_output_sizes_device = workspace_device + actual_output_sizes_device_offset;
  auto statuses_device = workspace_device + statuses_device_offset;
  auto temp_device = temp_bytes ? workspace_device + temp_device_offset : nullptr;

  {
    if (use_snappy) {
      status = nvcompBatchedSnappyDecompressAsync(
          reinterpret_cast<const void* const*>(compressed_ptrs_device),
          reinterpret_cast<const size_t*>(compressed_sizes_device),
          reinterpret_cast<const size_t*>(output_sizes_device),
          reinterpret_cast<size_t*>(actual_output_sizes_device),
          total_frame_count,
          temp_device,
          temp_bytes,
          reinterpret_cast<void* const*>(output_ptrs_device),
          snappy_decompress_opts,
          reinterpret_cast<nvcompStatus_t*>(statuses_device),
          transfer_stream);
    } else if (use_bitcomp) {
      status = nvcompBatchedBitcompDecompressAsync(
          reinterpret_cast<const void* const*>(compressed_ptrs_device),
          reinterpret_cast<const size_t*>(compressed_sizes_device),
          reinterpret_cast<const size_t*>(output_sizes_device),
          reinterpret_cast<size_t*>(actual_output_sizes_device),
          total_frame_count,
          temp_device,
          temp_bytes,
          reinterpret_cast<void* const*>(output_ptrs_device),
          bitcomp_decompress_opts,
          reinterpret_cast<nvcompStatus_t*>(statuses_device),
          transfer_stream);
    }
#ifdef HAVE_NVCOMP_GDEFLATE
    else if (use_gdeflate) {
      status = nvcompBatchedGdeflateDecompressAsync(
          reinterpret_cast<const void* const*>(compressed_ptrs_device),
          reinterpret_cast<const size_t*>(compressed_sizes_device),
          reinterpret_cast<const size_t*>(output_sizes_device),
          reinterpret_cast<size_t*>(actual_output_sizes_device),
          total_frame_count,
          temp_device,
          temp_bytes,
          reinterpret_cast<void* const*>(output_ptrs_device),
          gdeflate_decompress_opts,
          reinterpret_cast<nvcompStatus_t*>(statuses_device),
          transfer_stream);
    }
#endif
    else {
      status = nvcompBatchedLZ4DecompressAsync(
          reinterpret_cast<const void* const*>(compressed_ptrs_device),
          reinterpret_cast<const size_t*>(compressed_sizes_device),
          reinterpret_cast<const size_t*>(output_sizes_device),
          reinterpret_cast<size_t*>(actual_output_sizes_device),
          total_frame_count,
          temp_device,
          temp_bytes,
          reinterpret_cast<void* const*>(output_ptrs_device),
          lz4_decompress_opts,
          reinterpret_cast<nvcompStatus_t*>(statuses_device),
          transfer_stream);
    }
    if (status != nvcompSuccess) {
      VLOG(1) << "Batched nvCOMP compressed native storage decompression launch failed: "
              << nvcomp_status_name(status);
      return false;
    }
  }
  std::vector<int8_t> output_metadata(output_metadata_end - output_metadata_begin);
  {
    cuda_mgr->copyDeviceToHost(output_metadata.data(),
                               workspace_device + output_metadata_begin,
                               output_metadata.size(),
                               device_id,
                               compressed_copy_label + "ResultMetadata",
                               transfer_stream);
  }
  std::vector<nvcompStatus_t> statuses(total_frame_count);
  std::vector<size_t> actual_output_sizes(total_frame_count);
  memcpy(
      actual_output_sizes.data(),
      output_metadata.data() + actual_output_sizes_device_offset - output_metadata_begin,
      size_array_bytes);
  memcpy(statuses.data(),
         output_metadata.data() + statuses_device_offset - output_metadata_begin,
         status_array_bytes);
  for (size_t i = 0; i < total_frame_count; ++i) {
    if (statuses[i] != nvcompSuccess || actual_output_sizes[i] != output_sizes[i]) {
      VLOG(1) << "Batched nvCOMP compressed native storage frame " << i
              << " failed: status=" << nvcomp_status_name(statuses[i])
              << " actual=" << actual_output_sizes[i] << " expected=" << output_sizes[i];
      return false;
    }
  }

  return true;
}
#else
bool copy_compressed_file_buffer_to_gpu_with_nvcomp(File_Namespace::FileBuffer*,
                                                    AbstractBuffer*,
                                                    CudaMgr_Namespace::CudaMgr*,
                                                    CompressedGpuInputWorkspace&,
                                                    const size_t,
                                                    const size_t) {
  return false;
}

bool copy_compressed_file_buffers_to_gpu_with_nvcomp(
    const std::vector<File_Namespace::FileBuffer*>&,
    const std::vector<AbstractBuffer*>&,
    CudaMgr_Namespace::CudaMgr*,
    CompressedGpuInputWorkspace&,
    CUstream,
    std::mutex&) {
  return false;
}
#endif

}  // namespace

void CpuBufferMgr::fetchBuffer(const ChunkKey& key,
                               AbstractBuffer* dest_buffer,
                               const size_t num_bytes) {
  auto parent_mgr = getParentMgr();
  if (!g_enable_gpu_input_cpu_buffer_bypass || !cuda_mgr_ || !parent_mgr ||
      !dest_buffer || dest_buffer->getType() != GPU_LEVEL ||
      g_gpu_input_cpu_buffer_bypass_staging_buffer_bytes == 0 || isBufferOnDevice(key)) {
    BufferMgr::fetchBuffer(key, dest_buffer, num_bytes);
    return;
  }

  CHECK(!dest_buffer->isDirty())
      << "Aborting attempt to fetch a chunk marked dirty. Chunk inconsistency for key: "
      << show_chunk(key);

  AbstractBuffer* source_buffer = parent_mgr->getBufferIfNativeStorage(key, num_bytes);
  if (!source_buffer) {
    BufferMgr::fetchBuffer(key, dest_buffer, num_bytes);
    return;
  }
  auto source_file_buffer = dynamic_cast<File_Namespace::FileBuffer*>(source_buffer);
  if (!source_file_buffer) {
    BufferMgr::fetchBuffer(key, dest_buffer, num_bytes);
    return;
  }

  const size_t chunk_size = (num_bytes == 0) ? source_buffer->size() : num_bytes;
  CHECK_GE(source_buffer->size(), chunk_size)
      << "Attempting to fetch more bytes than a source buffer contains";

  const size_t existing_dest_size = dest_buffer->size();
  CHECK_LE(existing_dest_size, chunk_size)
      << "Destination buffer is larger than the source chunk";

  dest_buffer->reserve(chunk_size);

  if (source_file_buffer->isStorageCompressed()) {
    bool copied_compressed_payload = false;
    try {
      {
        CompressedGpuInputWorkspace compressed_workspace{dest_buffer};
        ScopeGuard compressed_workspace_guard = [&] {
          compressed_workspace.releaseAll();
        };
        copied_compressed_payload =
            copy_compressed_file_buffer_to_gpu_with_nvcomp(source_file_buffer,
                                                           dest_buffer,
                                                           cuda_mgr_,
                                                           compressed_workspace,
                                                           existing_dest_size,
                                                           chunk_size);
      }
    } catch (const OutOfMemory& error) {
      VLOG(1) << "Compressed GPU input workspace did not fit in the GPU buffer pool; "
                 "using CPU decompression for chunk "
              << show_chunk(key) << ": " << error.what();
    } catch (const CudaMgr_Namespace::CudaErrorException& error) {
      if (!is_cuda_out_of_memory(error)) {
        throw;
      }
      VLOG(1) << "Compressed GPU input scratch allocation failed; using CPU "
                 "decompression for chunk "
              << show_chunk(key) << ": " << error.what();
    }
    if (copied_compressed_payload) {
      dest_buffer->setSize(chunk_size);
      dest_buffer->syncEncoder(source_buffer);
      return;
    }
    BufferMgr::fetchBuffer(key, dest_buffer, num_bytes);
    return;
  }

  if (use_mmap_gpu_input_bypass() &&
      copy_file_buffer_to_gpu_with_mmap(
          source_file_buffer, dest_buffer, cuda_mgr_, existing_dest_size, chunk_size)) {
    dest_buffer->setSize(chunk_size);
    dest_buffer->syncEncoder(source_buffer);
    return;
  }

  auto read_source = [&](int8_t* host_ptr, size_t bytes_to_copy, size_t offset) {
    source_file_buffer->readWithReaderThreads(
        host_ptr,
        bytes_to_copy,
        offset,
        std::max<size_t>(g_gpu_input_cpu_buffer_bypass_reader_threads, size_t(1)));
  };

  const bool used_pinned_staging = cuda_mgr_->copyHostToDeviceFromPinnedProducer(
      dest_buffer->getMemoryPtr() + existing_dest_size,
      chunk_size - existing_dest_size,
      dest_buffer->getDeviceId(),
      "GpuInputCpuBufferBypass",
      [&](int8_t* host_ptr, size_t bytes_to_copy, size_t producer_offset) {
        read_source(host_ptr, bytes_to_copy, existing_dest_size + producer_offset);
      });

  if (used_pinned_staging) {
    dest_buffer->setSize(chunk_size);
    dest_buffer->syncEncoder(source_buffer);
    return;
  }

  const size_t staging_buffer_size =
      std::min(g_gpu_input_cpu_buffer_bypass_staging_buffer_bytes,
               chunk_size - existing_dest_size);
  thread_local std::vector<int8_t> staging_buffer;
  if (staging_buffer.size() < staging_buffer_size) {
    staging_buffer.resize(staging_buffer_size);
  }

  for (size_t offset = existing_dest_size; offset < chunk_size;) {
    const size_t bytes_to_copy = std::min(staging_buffer_size, chunk_size - offset);

    read_source(staging_buffer.data(), bytes_to_copy, offset);

    cuda_mgr_->copyHostToDeviceDirect(dest_buffer->getMemoryPtr() + offset,
                                      staging_buffer.data(),
                                      bytes_to_copy,
                                      dest_buffer->getDeviceId(),
                                      "GpuInputCpuBufferBypass");
    offset = checked_size_add(offset, bytes_to_copy, "GPU input copy offset overflow");
  }

  dest_buffer->setSize(chunk_size);
  dest_buffer->syncEncoder(source_buffer);
}

void CpuBufferMgr::fetchBuffers(
    const std::vector<Data_Namespace::BufferFetchRequest>& requests,
    const std::vector<AbstractBuffer*>& dest_buffers) {
  CHECK_EQ(requests.size(), dest_buffers.size());
  const auto scalar_fetch = [&](const std::vector<size_t>& request_indices) {
    for (const auto request_idx : request_indices) {
      fetchBuffer(requests[request_idx].key,
                  dest_buffers[request_idx],
                  requests[request_idx].num_bytes);
    }
  };
  const auto fallback_to_scalar_fetch = [&] {
    for (size_t i = 0; i < requests.size(); ++i) {
      fetchBuffer(requests[i].key, dest_buffers[i], requests[i].num_bytes);
    }
  };

  if (requests.empty()) {
    return;
  }

  auto parent_mgr = getParentMgr();
  if (!g_enable_gpu_input_cpu_buffer_bypass || !cuda_mgr_ || !parent_mgr ||
      g_gpu_input_cpu_buffer_bypass_staging_buffer_bytes == 0) {
    fallback_to_scalar_fetch();
    return;
  }

  const auto first_dest_buffer = dest_buffers.front();
  if (!first_dest_buffer || first_dest_buffer->getType() != GPU_LEVEL) {
    fallback_to_scalar_fetch();
    return;
  }
  const auto device_id = first_dest_buffer->getDeviceId();
  for (const auto dest_buffer : dest_buffers) {
    if (!dest_buffer || dest_buffer->getType() != GPU_LEVEL ||
        dest_buffer->getDeviceId() != device_id) {
      fallback_to_scalar_fetch();
      return;
    }
    CHECK(!dest_buffer->isDirty())
        << "Aborting attempt to fetch a chunk marked dirty. Chunk inconsistency.";
  }

  struct NativeStorageBatch {
    std::vector<size_t> request_indices;
    std::vector<AbstractBuffer*> source_buffers;
    std::vector<File_Namespace::FileBuffer*> source_file_buffers;
    std::vector<AbstractBuffer*> dest_buffers;
  };

  NativeStorageBatch snappy_batch;
  // Native Bitcomp batch plans require one algorithm and element type per launch.
  // Keep compatible columns together without falling back to one launch per column.
  using BitcompBatchKey = std::pair<bool, size_t>;
  std::map<BitcompBatchKey, NativeStorageBatch> bitcomp_batches;
  NativeStorageBatch lz4_batch;
  NativeStorageBatch gdeflate_batch;
  NativeStorageBatch uncompressed_batch;
  std::vector<size_t> scalar_request_indices;
  const auto reserve_batch_capacity = [&](NativeStorageBatch& batch) {
    batch.request_indices.reserve(requests.size());
    batch.source_buffers.reserve(requests.size());
    batch.source_file_buffers.reserve(requests.size());
    batch.dest_buffers.reserve(requests.size());
  };
  reserve_batch_capacity(snappy_batch);
  reserve_batch_capacity(lz4_batch);
  reserve_batch_capacity(gdeflate_batch);
  reserve_batch_capacity(uncompressed_batch);
  scalar_request_indices.reserve(requests.size());

  const auto add_to_batch = [](NativeStorageBatch& batch,
                               const size_t request_idx,
                               AbstractBuffer* source_buffer,
                               File_Namespace::FileBuffer* source_file_buffer,
                               AbstractBuffer* dest_buffer) {
    batch.request_indices.push_back(request_idx);
    batch.source_buffers.push_back(source_buffer);
    batch.source_file_buffers.push_back(source_file_buffer);
    batch.dest_buffers.push_back(dest_buffer);
  };

  for (size_t i = 0; i < requests.size(); ++i) {
    if (isBufferOnDevice(requests[i].key)) {
      // The CPU buffer may be newer than persistent storage, so it must remain the
      // source for this request. It should not prevent unrelated cold chunks from
      // using a batched native-storage transfer.
      scalar_request_indices.push_back(i);
      continue;
    }
    auto source_buffer =
        parent_mgr->getBufferIfNativeStorage(requests[i].key, requests[i].num_bytes);
    if (!source_buffer) {
      scalar_request_indices.push_back(i);
      continue;
    }
    auto source_file_buffer = dynamic_cast<File_Namespace::FileBuffer*>(source_buffer);
    if (!source_file_buffer) {
      scalar_request_indices.push_back(i);
      continue;
    }
    const size_t chunk_size =
        (requests[i].num_bytes == 0) ? source_buffer->size() : requests[i].num_bytes;
    if (dest_buffers[i]->size() != 0 || chunk_size != source_buffer->size()) {
      scalar_request_indices.push_back(i);
      continue;
    }
    dest_buffers[i]->reserve(chunk_size);

    if (source_file_buffer->isSnappyStorageCompressed()) {
      add_to_batch(snappy_batch, i, source_buffer, source_file_buffer, dest_buffers[i]);
    } else if (source_file_buffer->isBitcompStorageCompressed()) {
      const BitcompBatchKey batch_key{
          source_file_buffer->isBitcompSparseStorageCompressed(),
          source_file_buffer->storageBitcompElementWidth()};
      auto [batch_it, inserted] = bitcomp_batches.try_emplace(batch_key);
      if (inserted) {
        reserve_batch_capacity(batch_it->second);
      }
      add_to_batch(
          batch_it->second, i, source_buffer, source_file_buffer, dest_buffers[i]);
    } else if (source_file_buffer->isGdeflateStorageCompressed()) {
      add_to_batch(gdeflate_batch, i, source_buffer, source_file_buffer, dest_buffers[i]);
    } else if (source_file_buffer->isLz4StorageCompressed()) {
      add_to_batch(lz4_batch, i, source_buffer, source_file_buffer, dest_buffers[i]);
    } else if (!source_file_buffer->isStorageCompressed()) {
      add_to_batch(
          uncompressed_batch, i, source_buffer, source_file_buffer, dest_buffers[i]);
    } else {
      scalar_request_indices.push_back(i);
    }
  }
  const auto finish_batch = [](const NativeStorageBatch& batch) {
    CHECK_EQ(batch.source_buffers.size(), batch.dest_buffers.size());
    for (size_t i = 0; i < batch.dest_buffers.size(); ++i) {
      batch.dest_buffers[i]->setSize(batch.source_buffers[i]->size());
      batch.dest_buffers[i]->syncEncoder(batch.source_buffers[i]);
    }
  };

  const auto fetch_compressed_batch = [&](const NativeStorageBatch& batch) {
    if (batch.request_indices.empty()) {
      return;
    }
    CHECK_EQ(batch.request_indices.size(), batch.source_buffers.size());
    CHECK_EQ(batch.request_indices.size(), batch.source_file_buffers.size());
    CHECK_EQ(batch.request_indices.size(), batch.dest_buffers.size());

    const auto make_sub_batch = [&](const size_t begin, const size_t end) {
      CHECK_LT(begin, end);
      CHECK_LE(end, batch.request_indices.size());
      NativeStorageBatch sub_batch;
      const auto chunk_count = end - begin;
      sub_batch.request_indices.reserve(chunk_count);
      sub_batch.source_buffers.reserve(chunk_count);
      sub_batch.source_file_buffers.reserve(chunk_count);
      sub_batch.dest_buffers.reserve(chunk_count);
      for (size_t i = begin; i < end; ++i) {
        add_to_batch(sub_batch,
                     batch.request_indices[i],
                     batch.source_buffers[i],
                     batch.source_file_buffers[i],
                     batch.dest_buffers[i]);
      }
      return sub_batch;
    };

    const auto compressed_bytes = [&](const size_t begin, const size_t end) {
      size_t bytes{0};
      for (size_t i = begin; i < end; ++i) {
        bytes = checked_size_add(bytes,
                                 batch.source_file_buffers[i]->storageCompressedSize(),
                                 "Compressed GPU input sub-batch size overflow");
      }
      return bytes;
    };

    const auto split_by_compressed_bytes = [&](const size_t begin, const size_t end) {
      CHECK_GT(end - begin, size_t{1});
      const auto target_bytes = compressed_bytes(begin, end) / 2;
      size_t accumulated_bytes{0};
      for (size_t i = begin; i + 1 < end; ++i) {
        accumulated_bytes =
            checked_size_add(accumulated_bytes,
                             batch.source_file_buffers[i]->storageCompressedSize(),
                             "Compressed GPU input split size overflow");
        if (accumulated_bytes >= target_bytes) {
          return i + 1;
        }
      }
      return begin + (end - begin) / 2;
    };

    struct CompressedBatchRange {
      size_t begin;
      size_t end;
    };
    std::vector<CompressedBatchRange> ranges;
    const bool split_batch_budget = g_enable_gpu_input_compressed_pipeline &&
                                    g_gpu_input_compressed_batch_max_bytes > 1;
    const auto batch_limit = split_batch_budget
                                 ? g_gpu_input_compressed_batch_max_bytes / 2
                                 : g_gpu_input_compressed_batch_max_bytes;
    size_t begin{0};
    while (begin < batch.request_indices.size()) {
      size_t end{begin};
      size_t payload_bytes{0};
      while (end < batch.request_indices.size()) {
        const auto chunk_bytes = batch.source_file_buffers[end]->storageCompressedSize();
        if (end > begin && batch_limit != 0 &&
            (payload_bytes >= batch_limit || chunk_bytes > batch_limit - payload_bytes)) {
          break;
        }
        payload_bytes = checked_size_add(
            payload_bytes, chunk_bytes, "Compressed GPU input batch size overflow");
        ++end;
        if (batch_limit != 0 && payload_bytes >= batch_limit) {
          break;
        }
      }
      CHECK_GT(end, begin);
      ranges.push_back({begin, end});
      begin = end;
    }

    std::mutex payload_transfer_mutex;
    std::mutex scalar_fetch_mutex;
    const auto fetch_range = [&](auto&& self,
                                 const size_t range_begin,
                                 const size_t range_end,
                                 CompressedGpuInputWorkspace& compressed_workspace,
                                 const CUstream transfer_stream) -> void {
      auto sub_batch = make_sub_batch(range_begin, range_end);
      bool batch_fetch_succeeded{false};
      try {
        batch_fetch_succeeded =
            copy_compressed_file_buffers_to_gpu_with_nvcomp(sub_batch.source_file_buffers,
                                                            sub_batch.dest_buffers,
                                                            cuda_mgr_,
                                                            compressed_workspace,
                                                            transfer_stream,
                                                            payload_transfer_mutex);
      } catch (const OutOfMemory& error) {
        compressed_workspace.releaseAll();
        if (range_end - range_begin > 1) {
          const auto split = split_by_compressed_bytes(range_begin, range_end);
          self(self, range_begin, split, compressed_workspace, transfer_stream);
          self(self, split, range_end, compressed_workspace, transfer_stream);
          return;
        }
        VLOG(1) << "Compressed GPU input workspace did not fit in the GPU buffer pool "
                   "for one chunk; using scalar fallback: device="
                << device_id << " request=" << batch.request_indices[range_begin] << ": "
                << error.what();
      } catch (const CudaMgr_Namespace::CudaErrorException& error) {
        if (!is_cuda_out_of_memory(error)) {
          throw;
        }

        // A failed growth attempt releases the previous workspace. Explicitly clear
        // any allocation retained after failures elsewhere in the CUDA path before
        // retrying with less scratch pressure.
        compressed_workspace.releaseAll();
        if (range_end - range_begin > 1) {
          const auto split = split_by_compressed_bytes(range_begin, range_end);
          self(self, range_begin, split, compressed_workspace, transfer_stream);
          self(self, split, range_end, compressed_workspace, transfer_stream);
          return;
        }
        VLOG(1) << "Compressed GPU input scratch allocation failed for one chunk; "
                   "using scalar fallback: device="
                << device_id << " request=" << batch.request_indices[range_begin] << ": "
                << error.what();
      }

      if (batch_fetch_succeeded) {
        finish_batch(sub_batch);
      } else {
        std::lock_guard<std::mutex> scalar_fetch_lock(scalar_fetch_mutex);
        scalar_fetch(sub_batch.request_indices);
      }
    };

    const auto primary_stream = cuda_mgr_->getDeviceTransferStream(device_id);
#ifdef HAVE_NVCOMP
    const bool use_compressed_pipeline =
        g_enable_gpu_input_compressed_pipeline && ranges.size() > 1;
#else
    constexpr bool use_compressed_pipeline = false;
#endif
    if (!use_compressed_pipeline) {
      CompressedGpuInputWorkspace compressed_workspace{batch.dest_buffers.front()};
      ScopeGuard compressed_workspace_guard = [&] { compressed_workspace.releaseAll(); };
      for (const auto& range : ranges) {
        fetch_range(
            fetch_range, range.begin, range.end, compressed_workspace, primary_stream);
      }
      return;
    }

#ifdef HAVE_NVCOMP
    cuda_mgr_->setContext(device_id);
    // Keep both pipeline lanes isolated from concurrent users of the shared transfer
    // stream.
    std::vector<CUstream> transfer_streams(2, nullptr);
    ScopeGuard transfer_streams_guard = [&] {
      for (auto& transfer_stream : transfer_streams) {
        if (!transfer_stream) {
          continue;
        }
        try {
          cuda_mgr_->setContext(device_id);
          const auto status = cuStreamDestroy(transfer_stream);
          if (status != CUDA_SUCCESS && status != CUDA_ERROR_DEINITIALIZED) {
            LOG(ERROR) << "Failed to destroy compressed GPU input stream: device="
                       << device_id << " status=" << status;
          }
        } catch (const std::exception& error) {
          LOG(ERROR) << "Failed to release compressed GPU input stream: device="
                     << device_id << " error=" << error.what();
        }
        transfer_stream = nullptr;
      }
    };
    for (auto& transfer_stream : transfer_streams) {
      CudaMgr_Namespace::check_error(
          cuStreamCreate(&transfer_stream, CU_STREAM_NON_BLOCKING));
    }

    std::vector<std::unique_ptr<CompressedGpuInputWorkspace>> workspaces;
    workspaces.reserve(2);
    workspaces.emplace_back(
        std::make_unique<CompressedGpuInputWorkspace>(batch.dest_buffers.front()));
    workspaces.emplace_back(
        std::make_unique<CompressedGpuInputWorkspace>(batch.dest_buffers.front()));
    ScopeGuard workspace_guard = [&] {
      for (auto& workspace : workspaces) {
        workspace->releaseAll();
      }
    };

    std::vector<std::future<void>> in_flight(2);
    for (size_t range_idx = 0; range_idx < ranges.size(); ++range_idx) {
      const auto lane = range_idx % in_flight.size();
      if (in_flight[lane].valid()) {
        in_flight[lane].get();
      }
      const auto range = ranges[range_idx];
      in_flight[lane] = std::async(std::launch::async, [&, lane, range] {
        fetch_range(fetch_range,
                    range.begin,
                    range.end,
                    *workspaces[lane],
                    transfer_streams[lane]);
      });
    }
    for (auto& future : in_flight) {
      if (future.valid()) {
        future.get();
      }
    }
#endif
  };

  fetch_compressed_batch(snappy_batch);
  for (const auto& bitcomp_batch_entry : bitcomp_batches) {
    fetch_compressed_batch(bitcomp_batch_entry.second);
  }
  fetch_compressed_batch(gdeflate_batch);
  fetch_compressed_batch(lz4_batch);

  if (!uncompressed_batch.request_indices.empty()) {
    const bool batch_fetch_succeeded =
        use_mmap_gpu_input_bypass() &&
        copy_file_buffers_to_gpu_with_mmap(uncompressed_batch.source_file_buffers,
                                           uncompressed_batch.dest_buffers,
                                           cuda_mgr_);
    if (batch_fetch_succeeded) {
      finish_batch(uncompressed_batch);
    } else {
      scalar_fetch(uncompressed_batch.request_indices);
    }
  }

  scalar_fetch(scalar_request_indices);
}

CpuBufferMgr::~CpuBufferMgr() = default;

void CpuBufferMgr::addSlab(const size_t slab_size) {
  CHECK(allocator_);
  slabs_.resize(slabs_.size() + 1);
  try {
    slabs_.back() = reinterpret_cast<int8_t*>(allocator_->allocate(slab_size));
  } catch (std::bad_alloc&) {
    slabs_.resize(slabs_.size() - 1);
    throw FailedToCreateSlab(slab_size);
  }
  slab_segments_.resize(slab_segments_.size() + 1);
  slab_segments_[slab_segments_.size() - 1].emplace_back(0, slab_size / page_size_);
}

void CpuBufferMgr::freeAllMem() {
  CHECK(allocator_);
  initializeMem();
}

Buffer* CpuBufferMgr::createBuffer(BufferList::iterator seg_it, size_t page_size) {
  return new CpuBuffer(this, seg_it, device_id_, cuda_mgr_, page_size);
}

void CpuBufferMgr::initializeMem() {
  allocator_.reset(new DramArena(default_slab_size_ + kArenaBlockOverhead));
}

std::ostream& operator<<(std::ostream& os,
                         const CpuBufferMgr::CpuBufferMgrMemoryUsage& bm) {
  return os << "\"CPU Buffers\": {"
            << "\"Allocated MB\": " << bm.allocated / (1024. * 1024.) << ", "
            << "\"In Use MB\": " << bm.in_use / (1024. * 1024.) << "}";
}

}  // namespace Buffer_Namespace
