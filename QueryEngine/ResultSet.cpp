/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    ResultSet.cpp
 * @brief   Basic constructors and methods of the row set interface.
 *
 */

#include "ResultSet.h"
#include "CudaMgr/CudaMgr.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "DataMgr/FileMgr/FileBuffer.h"
#include "DataMgr/ForeignStorage/PassThroughBuffer.h"
#include "DataMgr/PersistentStorageMgr/PersistentStorageMgr.h"
#include "Execute.h"
#include "GpuInitGroups.h"
#include "GpuMemUtils.h"
#include "InPlaceSort.h"
#include "MurmurHash.h"
#include "OutputBufferInitialization.h"
#include "QueryEngine/QueryEngine.h"
#include "RelAlgExecutionUnit.h"
#include "RuntimeFunctions.h"
#include "Shared/Intervals.h"
#include "Shared/SqlTypesLayout.h"
#include "Shared/checked_alloc.h"
#include "Shared/likely.h"
#include "Shared/scope.h"
#include "Shared/thread_count.h"
#include "Shared/threading.h"
#include "Utils/StringLike.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cstring>
#include <functional>
#include <future>
#include <iterator>
#include <numeric>
#include <set>
#include <sstream>
#include <string_view>
#include <unordered_set>

#include <sys/mman.h>

size_t g_parallel_top_min = 100e3;
size_t g_parallel_top_max = 20e6;            // In effect only with g_enable_watchdog.
size_t g_watchdog_baseline_sort_max = 10e6;  // In effect only with g_enable_watchdog.
size_t g_streaming_topn_max = 100e3;
constexpr int64_t uninitialized_cached_row_count{-1};
constexpr size_t auto_parallel_row_count_threshold{20000UL};
constexpr size_t local_dictionary_sort_max_result_rows{100000UL};
constexpr size_t parallel_top_max_worker_count{64UL};
extern size_t g_baseline_groupby_threshold;
extern bool g_enable_result_reduction_pipeline;
extern bool g_enable_gpu_input_cpu_buffer_bypass;

namespace {

const int8_t* get_entry_target_ptr(const ResultSetStorage& storage,
                                   const size_t entry_idx,
                                   const size_t target_idx);
int8_t get_entry_target_width(const ResultSetStorage& storage, const size_t target_idx);

bool is_cuda_out_of_memory(const CudaMgr_Namespace::CudaErrorException& error) noexcept {
#ifdef HAVE_CUDA
  return error.getStatus() == CUDA_ERROR_OUT_OF_MEMORY;
#else
  static_cast<void>(error);
  return false;
#endif
}

}  // namespace

void write_int_to_buff(int8_t* const ptr, const int8_t compact_sz, const int64_t value);

DeferredLazyFetchChunk::DeferredLazyFetchChunk(const ColumnDescriptor& column_descriptor,
                                               Data_Namespace::DataMgr* data_mgr,
                                               ChunkKey chunk_key,
                                               const size_t num_bytes,
                                               const size_t num_elements)
    : DeferredLazyFetchChunk(column_descriptor,
                             data_mgr,
                             std::vector<DeferredLazyFetchChunkSource>{
                                 DeferredLazyFetchChunkSource{std::move(chunk_key),
                                                              num_bytes,
                                                              num_elements}}) {}

DeferredLazyFetchChunk::DeferredLazyFetchChunk(
    const ColumnDescriptor& column_descriptor,
    Data_Namespace::DataMgr* data_mgr,
    std::vector<DeferredLazyFetchChunkSource> sources)
    : column_descriptor_(column_descriptor), data_mgr_(data_mgr) {
  CHECK(data_mgr_);
  CHECK(!sources.empty());
  sources_.reserve(sources.size());
  for (auto& source : sources) {
    CHECK_GT(source.num_bytes, size_t(0));
    CHECK_GT(source.num_elements, size_t(0));
    if (source.num_bytes > std::numeric_limits<size_t>::max() - num_bytes_ ||
        source.num_elements > std::numeric_limits<size_t>::max() - num_elements_) {
      throw std::overflow_error("Deferred lazy fetch source size overflow");
    }
    sources_.push_back(SourceState{std::move(source.chunk_key),
                                   source.num_bytes,
                                   source.num_elements,
                                   num_bytes_,
                                   num_elements_});
    num_bytes_ += source.num_bytes;
    num_elements_ += source.num_elements;
  }
}

DeferredLazyFetchChunk::~DeferredLazyFetchChunk() {
  if (cpu_cache_future_.valid()) {
    try {
      cpu_cache_future_.get();
    } catch (const std::exception& error) {
      LOG(WARNING) << "Deferred lazy fetch CPU cache population failed: " << error.what();
    }
  }
  if (sparse_buffer_ && munmap(sparse_buffer_, num_bytes_) != 0) {
    LOG(ERROR) << "Failed to release deferred lazy fetch mapping";
  }
}

bool DeferredLazyFetchChunk::materializeFullyViaGpuLocked() const {
  if (!g_enable_gpu_input_cpu_buffer_bypass || sparse_buffer_ ||
      !data_mgr_->gpusPresent()) {
    return false;
  }
  auto* cuda_mgr = data_mgr_->getCudaMgr();
  if (!cuda_mgr || cuda_mgr->getDeviceCount() <= 0 || sources_.empty() ||
      sources_.front().chunk_key.size() < size_t(3)) {
    return false;
  }

  std::vector<Data_Namespace::BufferFetchRequest> requests;
  std::vector<ChunkMetadata> source_metadata;
  requests.reserve(sources_.size());
  source_metadata.reserve(sources_.size());
  for (const auto& source : sources_) {
    if (data_mgr_->isBufferOnDevice(source.chunk_key, Data_Namespace::CPU_LEVEL, 0)) {
      return false;
    }
    auto* file_buffer = dynamic_cast<File_Namespace::FileBuffer*>(
        data_mgr_->getPersistentStorageMgr()->getBufferIfNativeStorage(source.chunk_key,
                                                                       source.num_bytes));
    if (!file_buffer || !file_buffer->isStorageCompressed()) {
      return false;
    }
    CHECK(file_buffer->hasEncoder());
    requests.push_back({source.chunk_key, source.num_bytes});
    source_metadata.push_back(file_buffer->getEncoder()->getMetadata());
  }

  const auto column_id = get_column(sources_.front().chunk_key);
  const auto device_id = static_cast<int>(
      std::hash<int>{}(column_id) % static_cast<size_t>(cuda_mgr->getDeviceCount()));
  try {
    auto gpu_buffers =
        data_mgr_->getChunkBuffers(requests, Data_Namespace::GPU_LEVEL, device_id);
    CHECK_EQ(gpu_buffers.size(), sources_.size());
    ScopeGuard unpin_buffers([&] {
      for (auto* buffer : gpu_buffers) {
        CHECK(buffer);
        buffer->unPin();
      }
    });

    materialized_buffer_.resize(num_bytes_);
    for (size_t source_idx = 0; source_idx < sources_.size(); ++source_idx) {
      const auto& source = sources_[source_idx];
      auto* gpu_buffer = gpu_buffers[source_idx];
      CHECK(gpu_buffer);
      CHECK_EQ(gpu_buffer->getType(), Data_Namespace::GPU_LEVEL);
      CHECK_EQ(gpu_buffer->getDeviceId(), device_id);
      CHECK_GE(gpu_buffer->size(), source.num_bytes);
      cuda_mgr->copyDeviceToHost(materialized_buffer_.data() + source.byte_offset,
                                 gpu_buffer->getMemoryPtr(),
                                 source.num_bytes,
                                 device_id,
                                 "DeferredLazyFetchGpuMaterialization");
    }

    buffer_ = materialized_buffer_.data();
    CHECK(!cpu_cache_future_.valid());
    // Seed the ordinary CPU chunk cache from the decompressed host bytes while the
    // downstream query consumes this contiguous view. Destruction joins the task.
    try {
      cpu_cache_future_ = std::async(
          std::launch::async, [this, source_metadata = std::move(source_metadata)] {
            CHECK_EQ(source_metadata.size(), sources_.size());
            for (size_t source_idx = 0; source_idx < sources_.size(); ++source_idx) {
              const auto& source = sources_[source_idx];
              foreign_storage::PassThroughBuffer host_buffer(
                  materialized_buffer_.data() + source.byte_offset, source.num_bytes);
              host_buffer.setMetadata(source_metadata[source_idx]);
              try {
                auto* cpu_buffer = data_mgr_->cacheCpuChunkBuffer(
                    source.chunk_key, &host_buffer, source.num_bytes);
                CHECK(cpu_buffer);
                cpu_buffer->unPin();
              } catch (const OutOfMemory&) {
                // The current query already owns valid host bytes. A later query can
                // use the same GPU path if the CPU cache cannot retain this chunk.
              }
            }
          });
    } catch (const std::system_error& error) {
      LOG(WARNING) << "Unable to start deferred lazy fetch CPU cache population: "
                   << error.what();
    } catch (const std::bad_alloc&) {
      // Cache population is optional; the current query already has valid host bytes.
    }
    return true;
  } catch (const OutOfMemory&) {
    materialized_buffer_.clear();
    return false;
  } catch (const CudaMgr_Namespace::CudaErrorException& error) {
    if (!is_cuda_out_of_memory(error)) {
      throw;
    }
    materialized_buffer_.clear();
    return false;
  }
}

void DeferredLazyFetchChunk::materializeFullyLocked(const int8_t*& buffer_slot) const {
  if (!fully_materialized_.load(std::memory_order_relaxed)) {
    if (materializeFullyViaGpuLocked()) {
      CHECK(buffer_);
    } else if (sparse_buffer_) {
      for (auto& source : sources_) {
        auto* file_buffer = dynamic_cast<File_Namespace::FileBuffer*>(
            data_mgr_->getPersistentStorageMgr()->getBufferIfNativeStorage(
                source.chunk_key, source.num_bytes));
        CHECK(file_buffer);
        CHECK(file_buffer->isStorageCompressed());
        file_buffer->readWithReaderThreads(
            sparse_buffer_ + source.byte_offset,
            source.num_bytes,
            0,
            std::min<size_t>(source.materialized_frames.size(), 16));
        std::fill(
            source.materialized_frames.begin(), source.materialized_frames.end(), true);
      }
      buffer_ = sparse_buffer_;
    } else if (sources_.size() == size_t(1)) {
      auto& source = sources_.front();
      source.chunk = Chunk_NS::Chunk::getChunk(&column_descriptor_,
                                               data_mgr_,
                                               source.chunk_key,
                                               Data_Namespace::CPU_LEVEL,
                                               0,
                                               source.num_bytes,
                                               source.num_elements);
      CHECK(source.chunk);
      CHECK(source.chunk->getBuffer());
      buffer_ = source.chunk->getBuffer()->getMemoryPtr();
      CHECK(buffer_);
    } else {
      materialized_buffer_.resize(num_bytes_);
      std::atomic<size_t> next_source_idx{0};
      const auto worker_count = std::max<size_t>(
          1, std::min({sources_.size(), static_cast<size_t>(cpu_threads()), size_t(32)}));
      std::vector<std::future<void>> workers;
      workers.reserve(worker_count);
      for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
        workers.push_back(std::async(std::launch::async, [&] {
          while (true) {
            const auto source_idx = next_source_idx.fetch_add(1);
            if (source_idx >= sources_.size()) {
              return;
            }
            auto& source = sources_[source_idx];
            source.chunk = Chunk_NS::Chunk::getChunk(&column_descriptor_,
                                                     data_mgr_,
                                                     source.chunk_key,
                                                     Data_Namespace::CPU_LEVEL,
                                                     0,
                                                     source.num_bytes,
                                                     source.num_elements);
            CHECK(source.chunk);
            CHECK(source.chunk->getBuffer());
            const auto* source_buffer = source.chunk->getBuffer()->getMemoryPtr();
            CHECK(source_buffer);
            std::memcpy(materialized_buffer_.data() + source.byte_offset,
                        source_buffer,
                        source.num_bytes);
          }
        }));
      }
      for (auto& worker : workers) {
        worker.get();
      }
      buffer_ = materialized_buffer_.data();
      CHECK(buffer_);
    }
    fully_materialized_.store(true, std::memory_order_release);
  }
  CHECK(buffer_);
  if (!buffer_slot) {
    buffer_slot = buffer_;
  } else {
    CHECK_EQ(buffer_slot, buffer_);
  }
}

void DeferredLazyFetchChunk::materialize(const int8_t*& buffer_slot) const {
  if (fully_materialized_.load(std::memory_order_acquire)) {
    CHECK(buffer_);
    if (!buffer_slot) {
      buffer_slot = buffer_;
    } else {
      CHECK_EQ(buffer_slot, buffer_);
    }
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  materializeFullyLocked(buffer_slot);
}

bool DeferredLazyFetchChunk::materializeRowsLocked(const int64_t* local_row_indices,
                                                   const size_t row_count,
                                                   const int8_t*& buffer_slot) const {
  if (fully_materialized_.load(std::memory_order_relaxed)) {
    CHECK(buffer_);
    if (!buffer_slot) {
      buffer_slot = buffer_;
    } else {
      CHECK_EQ(buffer_slot, buffer_);
    }
    return true;
  }
  if (row_count == 0) {
    return true;
  }
  CHECK(local_row_indices);
  if (num_elements_ == 0 || num_bytes_ == 0) {
    return false;
  }

  size_t row_width = 0;
  std::vector<File_Namespace::FileBuffer*> file_buffers;
  file_buffers.reserve(sources_.size());
  for (auto& source : sources_) {
    if (source.num_bytes % source.num_elements != 0) {
      CHECK(!sparse_buffer_);
      return false;
    }
    const auto source_row_width = source.num_bytes / source.num_elements;
    if (source_row_width == 0 || (row_width != 0 && row_width != source_row_width)) {
      CHECK(!sparse_buffer_);
      return false;
    }
    row_width = source_row_width;
    auto* file_buffer = dynamic_cast<File_Namespace::FileBuffer*>(
        data_mgr_->getPersistentStorageMgr()->getBufferIfNativeStorage(source.chunk_key,
                                                                       source.num_bytes));
    if (!file_buffer || !file_buffer->isStorageCompressed()) {
      CHECK(!sparse_buffer_);
      return false;
    }
    CHECK_EQ(file_buffer->size(), source.num_bytes);
    const auto frame_size = file_buffer->storageCompressionFrameSize();
    const auto& compressed_frame_sizes = file_buffer->storageCompressedFrameSizes();
    CHECK_GT(frame_size, size_t(0));
    CHECK(!compressed_frame_sizes.empty());
    if (source.materialized_frames.empty()) {
      source.sparse_frame_size = frame_size;
      source.materialized_frames.assign(compressed_frame_sizes.size(), false);
    } else {
      CHECK_EQ(source.sparse_frame_size, frame_size);
      CHECK_EQ(source.materialized_frames.size(), compressed_frame_sizes.size());
    }
    file_buffers.push_back(file_buffer);
  }
  CHECK_GT(row_width, size_t(0));
  CHECK_EQ(num_bytes_ % row_width, size_t(0));
  CHECK_EQ(num_bytes_ / row_width, num_elements_);

  std::vector<std::vector<size_t>> required_frames(sources_.size());
  for (size_t row_idx = 0; row_idx < row_count; ++row_idx) {
    CHECK_GE(local_row_indices[row_idx], int64_t(0));
    const auto local_row_idx = static_cast<size_t>(local_row_indices[row_idx]);
    CHECK_LT(local_row_idx, num_elements_);
    auto source_it =
        std::upper_bound(sources_.begin(),
                         sources_.end(),
                         local_row_idx,
                         [](const size_t candidate_row_idx, const SourceState& source) {
                           return candidate_row_idx < source.row_offset;
                         });
    CHECK(source_it != sources_.begin());
    --source_it;
    const auto source_idx =
        static_cast<size_t>(std::distance(sources_.begin(), source_it));
    CHECK_LT(local_row_idx - source_it->row_offset, source_it->num_elements);
    const auto byte_offset = (local_row_idx - source_it->row_offset) * row_width;
    const auto byte_end = byte_offset + row_width;
    const auto first_frame_idx = byte_offset / source_it->sparse_frame_size;
    const auto last_frame_idx = (byte_end - 1) / source_it->sparse_frame_size;
    CHECK_LT(last_frame_idx, source_it->materialized_frames.size());
    for (size_t frame_idx = first_frame_idx; frame_idx <= last_frame_idx; ++frame_idx) {
      required_frames[source_idx].push_back(frame_idx);
    }
  }

  bool requires_every_frame = true;
  for (size_t source_idx = 0; source_idx < sources_.size(); ++source_idx) {
    auto& source_required_frames = required_frames[source_idx];
    std::sort(source_required_frames.begin(), source_required_frames.end());
    source_required_frames.erase(
        std::unique(source_required_frames.begin(), source_required_frames.end()),
        source_required_frames.end());
    requires_every_frame &=
        source_required_frames.size() == sources_[source_idx].materialized_frames.size();
  }
  if (requires_every_frame) {
    return false;
  }

  if (!sparse_buffer_) {
    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    mmap_flags |= MAP_NORESERVE;
#endif
    auto* mapping = mmap(nullptr, num_bytes_, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
    if (mapping == MAP_FAILED) {
      return false;
    }
    sparse_buffer_ = static_cast<int8_t*>(mapping);
    buffer_ = sparse_buffer_;
  }

  for (size_t source_idx = 0; source_idx < sources_.size(); ++source_idx) {
    auto& source = sources_[source_idx];
    auto& source_required_frames = required_frames[source_idx];
    size_t required_idx = 0;
    while (required_idx < source_required_frames.size()) {
      if (source.materialized_frames[source_required_frames[required_idx]]) {
        ++required_idx;
        continue;
      }
      const auto run_first_frame = source_required_frames[required_idx];
      auto run_last_frame = run_first_frame;
      ++required_idx;
      while (required_idx < source_required_frames.size() &&
             source_required_frames[required_idx] == run_last_frame + 1 &&
             !source.materialized_frames[source_required_frames[required_idx]]) {
        run_last_frame = source_required_frames[required_idx];
        ++required_idx;
      }
      const auto run_offset = run_first_frame * source.sparse_frame_size;
      const auto run_end =
          std::min(source.num_bytes, (run_last_frame + 1) * source.sparse_frame_size);
      file_buffers[source_idx]->readWithReaderThreads(
          sparse_buffer_ + source.byte_offset + run_offset,
          run_end - run_offset,
          run_offset,
          1);
      std::fill(source.materialized_frames.begin() + run_first_frame,
                source.materialized_frames.begin() + run_last_frame + 1,
                true);
    }
  }

  const auto all_frames_materialized =
      std::all_of(sources_.begin(), sources_.end(), [](const SourceState& source) {
        return std::all_of(source.materialized_frames.begin(),
                           source.materialized_frames.end(),
                           [](const bool is_materialized) { return is_materialized; });
      });
  if (all_frames_materialized) {
    fully_materialized_.store(true, std::memory_order_release);
  }
  if (!buffer_slot) {
    buffer_slot = buffer_;
  } else {
    CHECK_EQ(buffer_slot, buffer_);
  }
  return true;
}

void DeferredLazyFetchChunk::materializeRows(
    const std::vector<int64_t>& local_row_indices,
    const int8_t*& buffer_slot) const {
  if (fully_materialized_.load(std::memory_order_acquire)) {
    CHECK(buffer_);
    if (!buffer_slot) {
      buffer_slot = buffer_;
    } else {
      CHECK_EQ(buffer_slot, buffer_);
    }
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!materializeRowsLocked(
          local_row_indices.data(), local_row_indices.size(), buffer_slot)) {
    materializeFullyLocked(buffer_slot);
  }
}

void DeferredLazyFetchChunk::materializeRow(const int64_t local_row_idx,
                                            const int8_t*& buffer_slot) const {
  if (fully_materialized_.load(std::memory_order_acquire)) {
    CHECK(buffer_);
    if (!buffer_slot) {
      buffer_slot = buffer_;
    } else {
      CHECK_EQ(buffer_slot, buffer_);
    }
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (sparse_buffer_ && local_row_idx >= 0 &&
      static_cast<size_t>(local_row_idx) < num_elements_) {
    const auto row_idx = static_cast<size_t>(local_row_idx);
    auto source_it =
        std::upper_bound(sources_.begin(),
                         sources_.end(),
                         row_idx,
                         [](const size_t candidate_row_idx, const SourceState& source) {
                           return candidate_row_idx < source.row_offset;
                         });
    CHECK(source_it != sources_.begin());
    --source_it;
    CHECK_GT(source_it->num_elements, size_t(0));
    CHECK_EQ(source_it->num_bytes % source_it->num_elements, size_t(0));
    const auto row_width = source_it->num_bytes / source_it->num_elements;
    const auto source_row_idx = row_idx - source_it->row_offset;
    CHECK_LT(source_row_idx, source_it->num_elements);
    const auto byte_offset = source_row_idx * row_width;
    const auto first_frame_idx = byte_offset / source_it->sparse_frame_size;
    const auto last_frame_idx =
        (byte_offset + row_width - 1) / source_it->sparse_frame_size;
    CHECK_LT(last_frame_idx, source_it->materialized_frames.size());
    bool row_is_materialized = true;
    for (size_t frame_idx = first_frame_idx; frame_idx <= last_frame_idx; ++frame_idx) {
      row_is_materialized =
          row_is_materialized && source_it->materialized_frames[frame_idx];
    }
    if (row_is_materialized) {
      CHECK(buffer_);
      if (!buffer_slot) {
        buffer_slot = buffer_;
      } else {
        CHECK_EQ(buffer_slot, buffer_);
      }
      return;
    }
  }
  if (!materializeRowsLocked(&local_row_idx, 1, buffer_slot)) {
    materializeFullyLocked(buffer_slot);
  }
}

void ResultSet::keepFirstN(const size_t n) {
  invalidateCachedRowCount();
  keep_first_ = n;
}

void ResultSet::dropFirstN(const size_t n) {
  invalidateCachedRowCount();
  drop_first_ = n;
}

void ResultSet::setDeferredLazyFetchChunks(
    const DeferredLazyFetchChunks& deferred_lazy_fetch_chunks) {
  if (deferred_lazy_fetch_chunks.empty()) {
    return;
  }
  CHECK_EQ(col_buffers_.size(), size_t(1));
  CHECK_EQ(deferred_lazy_fetch_chunks.size(), col_buffers_.front().size());
  for (size_t frag_idx = 0; frag_idx < deferred_lazy_fetch_chunks.size(); ++frag_idx) {
    CHECK_EQ(deferred_lazy_fetch_chunks[frag_idx].size(),
             col_buffers_.front()[frag_idx].size());
  }
  deferred_lazy_fetch_chunks_ = {deferred_lazy_fetch_chunks};
  deferred_lazy_fetch_columns_materialized_for_all_rows_.clear();
}

void ResultSet::materializeDeferredLazyFetchColumn(
    const size_t target_logical_idx) const {
  materializeDeferredLazyFetchColumns({target_logical_idx});
}

void ResultSet::materializeDeferredLazyFetchColumns(
    const std::vector<size_t>& target_logical_indices) const {
  if (deferred_lazy_fetch_chunks_.empty()) {
    return;
  }
  std::lock_guard<std::mutex> materialization_lock(
      deferred_lazy_fetch_materialization_mutex_);

  struct DeferredChunkLocation {
    size_t storage_idx;
    size_t frag_idx;
    size_t local_col_idx;
    DeferredLazyFetchChunkPtr chunk;
  };
  std::vector<DeferredChunkLocation> chunk_locations;
  std::unordered_set<const DeferredLazyFetchChunk*> seen_chunks;
  for (const auto target_logical_idx : target_logical_indices) {
    CHECK_LT(target_logical_idx, lazy_fetch_info_.size());
    const auto& lazy_fetch_info = lazy_fetch_info_[target_logical_idx];
    if (!lazy_fetch_info.is_lazily_fetched) {
      continue;
    }
    CHECK_GE(lazy_fetch_info.local_col_id, 0);
    const auto local_col_idx = static_cast<size_t>(lazy_fetch_info.local_col_id);
    for (size_t storage_idx = 0; storage_idx < deferred_lazy_fetch_chunks_.size();
         ++storage_idx) {
      const auto& storage_chunks = deferred_lazy_fetch_chunks_[storage_idx];
      CHECK_LT(storage_idx, col_buffers_.size());
      CHECK_EQ(storage_chunks.size(), col_buffers_[storage_idx].size());
      for (size_t frag_idx = 0; frag_idx < storage_chunks.size(); ++frag_idx) {
        CHECK_LT(local_col_idx, storage_chunks[frag_idx].size());
        const auto& deferred_chunk = storage_chunks[frag_idx][local_col_idx];
        if (deferred_chunk && !col_buffers_[storage_idx][frag_idx][local_col_idx] &&
            seen_chunks.insert(deferred_chunk.get()).second) {
          chunk_locations.push_back(DeferredChunkLocation{
              storage_idx, frag_idx, local_col_idx, deferred_chunk});
        }
      }
    }
  }
  if (chunk_locations.empty()) {
    markDeferredLazyFetchColumnsMaterializedForAllRows(target_logical_indices);
    return;
  }

  std::atomic<size_t> next_chunk_idx{0};
  const auto worker_count = std::max<size_t>(
      1,
      std::min({chunk_locations.size(), static_cast<size_t>(cpu_threads()), size_t(32)}));
  std::vector<std::future<void>> workers;
  workers.reserve(worker_count);
  for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.push_back(std::async(std::launch::async, [&] {
      while (true) {
        const auto chunk_idx = next_chunk_idx.fetch_add(1);
        if (chunk_idx >= chunk_locations.size()) {
          return;
        }
        const auto& location = chunk_locations[chunk_idx];
        auto& buffer_slot =
            col_buffers_[location.storage_idx][location.frag_idx][location.local_col_idx];
        location.chunk->materialize(buffer_slot);
      }
    }));
  }
  for (auto& worker : workers) {
    worker.get();
  }
  markDeferredLazyFetchColumnsMaterializedForAllRows(target_logical_indices);
}

void ResultSet::materializeDeferredLazyFetchColumnsForOutputRows(
    const std::vector<size_t>& target_logical_indices) const {
  materializeDeferredLazyFetchColumnsForRows(target_logical_indices,
                                             DeferredLazyFetchRowSelection::OutputRows);
}

void ResultSet::materializeDeferredLazyFetchColumnsForAllRows(
    const std::vector<size_t>& target_logical_indices) const {
  materializeDeferredLazyFetchColumnsForRows(
      target_logical_indices, DeferredLazyFetchRowSelection::AllNonEmptyRows);
}

void ResultSet::materializeDeferredLazyFetchColumnsForRows(
    const std::vector<size_t>& target_logical_indices,
    const DeferredLazyFetchRowSelection row_selection) const {
  if (deferred_lazy_fetch_chunks_.empty()) {
    return;
  }
  std::lock_guard<std::mutex> materialization_lock(
      deferred_lazy_fetch_materialization_mutex_);
  const auto materializes_all_rows =
      row_selection == DeferredLazyFetchRowSelection::AllNonEmptyRows;
  if (entryCount() == 0) {
    if (materializes_all_rows) {
      markDeferredLazyFetchColumnsMaterializedForAllRows(target_logical_indices);
    }
    return;
  }

  struct DeferredChunkLocation {
    size_t storage_idx;
    size_t fragment_idx;
    size_t local_col_idx;
    DeferredLazyFetchChunkPtr chunk;
    std::vector<int64_t> local_row_indices;
  };
  struct DeferredTarget {
    size_t target_logical_idx;
    size_t local_col_idx;
  };
  std::vector<DeferredChunkLocation> chunk_locations;
  std::unordered_map<const DeferredLazyFetchChunk*, size_t> chunk_location_indices;
  std::vector<DeferredTarget> deferred_targets;
  for (const auto target_logical_idx : target_logical_indices) {
    CHECK_LT(target_logical_idx, lazy_fetch_info_.size());
    const auto& lazy_fetch_info = lazy_fetch_info_[target_logical_idx];
    if (!lazy_fetch_info.is_lazily_fetched) {
      continue;
    }
    CHECK_GE(lazy_fetch_info.local_col_id, 0);
    const auto local_col_idx = static_cast<size_t>(lazy_fetch_info.local_col_id);
    const auto has_deferred_chunk = std::any_of(
        deferred_lazy_fetch_chunks_.begin(),
        deferred_lazy_fetch_chunks_.end(),
        [local_col_idx](const auto& storage_chunks) {
          return std::any_of(storage_chunks.begin(),
                             storage_chunks.end(),
                             [local_col_idx](const auto& fragment_chunks) {
                               return local_col_idx < fragment_chunks.size() &&
                                      fragment_chunks[local_col_idx];
                             });
        });
    if (has_deferred_chunk) {
      deferred_targets.push_back(DeferredTarget{target_logical_idx, local_col_idx});
    }
  }

  const auto append_deferred_rows =
      [&](const StorageLookupResult& storage_lookup_result,
          std::vector<DeferredChunkLocation>& locations,
          std::unordered_map<const DeferredLazyFetchChunk*, size_t>& location_indices) {
        const auto* storage = storage_lookup_result.storage_ptr;
        CHECK(storage);
        for (const auto& deferred_target : deferred_targets) {
          const auto* source_row_idx_ptr =
              get_entry_target_ptr(*storage,
                                   storage_lookup_result.fixedup_entry_idx,
                                   deferred_target.target_logical_idx);
          CHECK(source_row_idx_ptr);
          const auto source_row_idx = read_int_from_buff(
              source_row_idx_ptr,
              get_entry_target_width(*storage, deferred_target.target_logical_idx));
          CHECK_GE(source_row_idx, 0);
          const auto fragment_lookup =
              resolveColumnFragment(storage_lookup_result.storage_idx,
                                    deferred_target.target_logical_idx,
                                    static_cast<int>(deferred_target.local_col_idx),
                                    source_row_idx);
          CHECK_LT(fragment_lookup.storage_idx, deferred_lazy_fetch_chunks_.size());
          const auto& storage_chunks =
              deferred_lazy_fetch_chunks_[fragment_lookup.storage_idx];
          if (storage_chunks.empty()) {
            continue;
          }
          CHECK_LT(fragment_lookup.fragment_idx, storage_chunks.size());
          CHECK_LT(deferred_target.local_col_idx,
                   storage_chunks[fragment_lookup.fragment_idx].size());
          const auto& deferred_chunk =
              storage_chunks[fragment_lookup.fragment_idx][deferred_target.local_col_idx];
          if (!deferred_chunk) {
            continue;
          }
          const auto [location_it, inserted] =
              location_indices.emplace(deferred_chunk.get(), locations.size());
          if (inserted) {
            locations.push_back(DeferredChunkLocation{fragment_lookup.storage_idx,
                                                      fragment_lookup.fragment_idx,
                                                      deferred_target.local_col_idx,
                                                      deferred_chunk,
                                                      {}});
          }
          locations[location_it->second].local_row_indices.push_back(
              fragment_lookup.local_row_idx);
        }
      };

  const auto entry_count = entryCount();
  if (materializes_all_rows && entry_count >= auto_parallel_row_count_threshold) {
    const auto worker_count = std::max<size_t>(
        1, std::min({entry_count, static_cast<size_t>(cpu_threads()), size_t(32)}));
    std::vector<std::vector<DeferredChunkLocation>> worker_locations(worker_count);
    threading::task_group workers;
    for (const auto interval : makeIntervals<size_t>(0, entry_count, worker_count)) {
      workers.run([&, interval] {
        auto& locations = worker_locations[interval.index];
        std::unordered_map<const DeferredLazyFetchChunk*, size_t> location_indices;
        for (auto logical_entry_idx = interval.begin; logical_entry_idx < interval.end;
             ++logical_entry_idx) {
          const auto storage_lookup_result = findStorage(logical_entry_idx);
          const auto* storage = storage_lookup_result.storage_ptr;
          CHECK(storage);
          if (!storage->isEmptyEntry(storage_lookup_result.fixedup_entry_idx)) {
            append_deferred_rows(storage_lookup_result, locations, location_indices);
          }
        }
      });
    }
    workers.wait();

    for (auto& locations : worker_locations) {
      for (auto& location : locations) {
        const auto [location_it, inserted] =
            chunk_location_indices.emplace(location.chunk.get(), chunk_locations.size());
        if (inserted) {
          chunk_locations.push_back(std::move(location));
        } else {
          auto& combined_location = chunk_locations[location_it->second];
          CHECK_EQ(combined_location.storage_idx, location.storage_idx);
          CHECK_EQ(combined_location.fragment_idx, location.fragment_idx);
          CHECK_EQ(combined_location.local_col_idx, location.local_col_idx);
          combined_location.local_row_indices.insert(
              combined_location.local_row_indices.end(),
              location.local_row_indices.begin(),
              location.local_row_indices.end());
        }
      }
    }
  } else {
    const auto select_output_rows =
        row_selection == DeferredLazyFetchRowSelection::OutputRows;
    auto rows_to_skip = select_output_rows ? drop_first_ : size_t(0);
    auto rows_to_materialize = select_output_rows && keep_first_
                                   ? keep_first_
                                   : std::numeric_limits<size_t>::max();
    for (size_t logical_entry_idx = 0;
         logical_entry_idx < entryCount() && rows_to_materialize > 0;
         ++logical_entry_idx) {
      const auto entry_idx = select_output_rows && !permutation_.empty()
                                 ? permutation_[logical_entry_idx]
                                 : logical_entry_idx;
      const auto storage_lookup_result = findStorage(entry_idx);
      const auto* storage = storage_lookup_result.storage_ptr;
      CHECK(storage);
      if (storage->isEmptyEntry(storage_lookup_result.fixedup_entry_idx)) {
        continue;
      }
      if (rows_to_skip > 0) {
        --rows_to_skip;
        continue;
      }
      --rows_to_materialize;
      append_deferred_rows(
          storage_lookup_result, chunk_locations, chunk_location_indices);
    }
  }
  if (chunk_locations.empty()) {
    if (materializes_all_rows) {
      markDeferredLazyFetchColumnsMaterializedForAllRows(target_logical_indices);
    }
    return;
  }

  {
    std::atomic<size_t> next_chunk_idx{0};
    const auto worker_count = std::max<size_t>(
        1,
        std::min(
            {chunk_locations.size(), static_cast<size_t>(cpu_threads()), size_t(32)}));
    std::vector<std::future<void>> workers;
    workers.reserve(worker_count);
    for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
      workers.push_back(std::async(std::launch::async, [&] {
        while (true) {
          const auto chunk_idx = next_chunk_idx.fetch_add(1);
          if (chunk_idx >= chunk_locations.size()) {
            return;
          }
          const auto& location = chunk_locations[chunk_idx];
          auto& buffer_slot = col_buffers_[location.storage_idx][location.fragment_idx]
                                          [location.local_col_idx];
          location.chunk->materializeRows(location.local_row_indices, buffer_slot);
        }
      }));
    }
    for (auto& worker : workers) {
      worker.get();
    }
  }
  if (materializes_all_rows) {
    markDeferredLazyFetchColumnsMaterializedForAllRows(target_logical_indices);
  }
}

void ResultSet::markDeferredLazyFetchColumnsMaterializedForAllRows(
    const std::vector<size_t>& target_logical_indices) const {
  if (deferred_lazy_fetch_columns_materialized_for_all_rows_.size() <
      lazy_fetch_info_.size()) {
    deferred_lazy_fetch_columns_materialized_for_all_rows_.resize(lazy_fetch_info_.size(),
                                                                  uint8_t{0});
  }
  for (const auto target_logical_idx : target_logical_indices) {
    CHECK_LT(target_logical_idx, lazy_fetch_info_.size());
    if (lazy_fetch_info_[target_logical_idx].is_lazily_fetched) {
      deferred_lazy_fetch_columns_materialized_for_all_rows_[target_logical_idx] = 1;
    }
  }
}

bool ResultSet::isDeferredLazyFetchColumnMaterializedForAllRows(
    const size_t target_logical_idx) const {
  return target_logical_idx <
             deferred_lazy_fetch_columns_materialized_for_all_rows_.size() &&
         deferred_lazy_fetch_columns_materialized_for_all_rows_[target_logical_idx];
}

void ResultSet::setLazyFetchSourceMetadata(
    const LazyFetchSourceMetadata& lazy_fetch_source_metadata) {
  lazy_fetch_source_metadata_ = lazy_fetch_source_metadata;
}

std::vector<LazyFetchSourceMetadataEntry> ResultSet::getLazyFetchSourceMetadata(
    const size_t target_logical_idx) const {
  std::vector<LazyFetchSourceMetadataEntry> metadata;
  if (lazy_fetch_source_metadata_.empty()) {
    return metadata;
  }
  CHECK_LT(target_logical_idx, lazy_fetch_info_.size());
  const auto& lazy_fetch_info = lazy_fetch_info_[target_logical_idx];
  if (!lazy_fetch_info.is_lazily_fetched) {
    return metadata;
  }
  CHECK_GE(lazy_fetch_info.local_col_id, 0);
  const auto local_col_idx = static_cast<size_t>(lazy_fetch_info.local_col_id);
  const auto metadata_it = lazy_fetch_source_metadata_.find(local_col_idx);
  if (metadata_it != lazy_fetch_source_metadata_.end()) {
    metadata = metadata_it->second;
  }
  return metadata;
}

ResultSet::ResultSet(const std::vector<TargetInfo>& targets,
                     const ExecutorDeviceType device_type,
                     const QueryMemoryDescriptor& query_mem_desc,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                     const unsigned block_size,
                     const unsigned grid_size)
    : targets_(targets)
    , device_type_(device_type)
    , device_id_(-1)
    , thread_idx_(-1)
    , query_mem_desc_(query_mem_desc)
    , crt_row_buff_idx_(0)
    , fetched_so_far_(0)
    , drop_first_(0)
    , keep_first_(0)
    , row_set_mem_owner_(row_set_mem_owner)
    , block_size_(block_size)
    , grid_size_(grid_size)
    , data_mgr_(nullptr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , for_validation_only_(false)
    , cached_row_count_(uninitialized_cached_row_count)
    , geo_return_type_(GeoReturnType::WktString)
    , cached_(false)
    , query_exec_time_(0)
    , query_plan_(EMPTY_HASHED_PLAN_DAG_KEY)
    , can_use_speculative_top_n_sort(std::nullopt) {}

ResultSet::ResultSet(const std::vector<TargetInfo>& targets,
                     const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
                     const std::vector<std::vector<const int8_t*>>& col_buffers,
                     const ColumnBufferLayouts& col_buffer_layouts,
                     const std::vector<std::vector<int64_t>>& frag_offsets,
                     const std::vector<int64_t>& consistent_frag_sizes,
                     const ExecutorDeviceType device_type,
                     const int device_id,
                     const int thread_idx,
                     const QueryMemoryDescriptor& query_mem_desc,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                     const unsigned block_size,
                     const unsigned grid_size)
    : targets_(targets)
    , device_type_(device_type)
    , device_id_(device_id)
    , thread_idx_(thread_idx)
    , query_mem_desc_(query_mem_desc)
    , crt_row_buff_idx_(0)
    , fetched_so_far_(0)
    , drop_first_(0)
    , keep_first_(0)
    , row_set_mem_owner_(row_set_mem_owner)
    , block_size_(block_size)
    , grid_size_(grid_size)
    , lazy_fetch_info_(lazy_fetch_info)
    , col_buffers_{col_buffers}
    , col_buffer_layouts_{col_buffer_layouts}
    , frag_offsets_{frag_offsets}
    , consistent_frag_sizes_{consistent_frag_sizes}
    , data_mgr_(nullptr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , for_validation_only_(false)
    , cached_row_count_(uninitialized_cached_row_count)
    , geo_return_type_(GeoReturnType::WktString)
    , cached_(false)
    , query_exec_time_(0)
    , query_plan_(EMPTY_HASHED_PLAN_DAG_KEY)
    , can_use_speculative_top_n_sort(std::nullopt) {}

ResultSet::ResultSet(const std::shared_ptr<const Analyzer::Estimator> estimator,
                     const ExecutorDeviceType device_type,
                     const int device_id,
                     Data_Namespace::DataMgr* data_mgr,
                     std::shared_ptr<CudaAllocator> device_allocator)
    : device_type_(device_type)
    , device_id_(device_id)
    , thread_idx_(-1)
    , query_mem_desc_{}
    , crt_row_buff_idx_(0)
    , estimator_(estimator)
    , data_mgr_(data_mgr)
    , cuda_allocator_(device_allocator)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , for_validation_only_(false)
    , cached_row_count_(uninitialized_cached_row_count)
    , geo_return_type_(GeoReturnType::WktString)
    , cached_(false)
    , query_exec_time_(0)
    , query_plan_(EMPTY_HASHED_PLAN_DAG_KEY)
    , can_use_speculative_top_n_sort(std::nullopt) {
  if (device_type == ExecutorDeviceType::GPU) {
    device_estimator_buffer_ = CudaAllocator::allocGpuAbstractBuffer(
        data_mgr_, estimator_->getBufferSize(), device_id_);
    cuda_allocator_->zeroDeviceMem(device_estimator_buffer_->getMemoryPtr(),
                                   estimator_->getBufferSize());
  } else {
    host_estimator_buffer_ =
        static_cast<int8_t*>(checked_calloc(estimator_->getBufferSize(), 1));
  }
}

ResultSet::ResultSet(const std::string& explanation)
    : device_type_(ExecutorDeviceType::CPU)
    , device_id_(-1)
    , thread_idx_(-1)
    , fetched_so_far_(0)
    , separate_varlen_storage_valid_(false)
    , explanation_(explanation)
    , just_explain_(true)
    , for_validation_only_(false)
    , cached_row_count_(uninitialized_cached_row_count)
    , geo_return_type_(GeoReturnType::WktString)
    , cached_(false)
    , query_exec_time_(0)
    , query_plan_(EMPTY_HASHED_PLAN_DAG_KEY)
    , can_use_speculative_top_n_sort(std::nullopt) {}

ResultSet::ResultSet(int64_t queue_time_ms,
                     int64_t render_time_ms,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
    : device_type_(ExecutorDeviceType::CPU)
    , device_id_(-1)
    , thread_idx_(-1)
    , fetched_so_far_(0)
    , row_set_mem_owner_(row_set_mem_owner)
    , timings_(QueryExecutionTimings{queue_time_ms, render_time_ms, 0, 0})
    , separate_varlen_storage_valid_(false)
    , just_explain_(true)
    , for_validation_only_(false)
    , cached_row_count_(uninitialized_cached_row_count)
    , geo_return_type_(GeoReturnType::WktString)
    , cached_(false)
    , query_exec_time_(0)
    , query_plan_(EMPTY_HASHED_PLAN_DAG_KEY)
    , can_use_speculative_top_n_sort(std::nullopt) {}

ResultSet::~ResultSet() {
  if (storage_) {
    if (!storage_->buff_is_provided_) {
      CHECK(storage_->getUnderlyingBuffer());
      free(storage_->getUnderlyingBuffer());
    }
  }
  for (auto& storage : appended_storage_) {
    if (storage && !storage->buff_is_provided_) {
      free(storage->getUnderlyingBuffer());
    }
  }
  if (host_estimator_buffer_) {
    CHECK(device_type_ == ExecutorDeviceType::CPU || device_estimator_buffer_);
    free(host_estimator_buffer_);
  }
  if (device_estimator_buffer_) {
    CHECK(data_mgr_);
    data_mgr_->free(device_estimator_buffer_);
  }
}

std::string ResultSet::summaryToString() const {
  std::ostringstream oss;
  oss << "Result Set Info" << std::endl;
  oss << "\tLayout: " << query_mem_desc_.queryDescTypeToString() << std::endl;
  oss << "\tColumns: " << colCount() << std::endl;
  oss << "\tRows: " << rowCount() << std::endl;
  oss << "\tEntry count: " << entryCount() << std::endl;
  const std::string is_empty = isEmpty() ? "True" : "False";
  oss << "\tIs empty: " << is_empty << std::endl;
  const std::string did_output_columnar = didOutputColumnar() ? "True" : "False;";
  oss << "\tColumnar: " << did_output_columnar << std::endl;
  oss << "\tLazy-fetched columns: " << getNumColumnsLazyFetched() << std::endl;
  const std::string is_direct_columnar_conversion_possible =
      isDirectColumnarConversionPossible() ? "True" : "False";
  oss << "\tDirect columnar conversion possible: "
      << is_direct_columnar_conversion_possible << std::endl;

  size_t num_columns_zero_copy_columnarizable{0};
  for (size_t target_idx = 0; target_idx < targets_.size(); target_idx++) {
    if (isZeroCopyColumnarConversionPossible(target_idx)) {
      num_columns_zero_copy_columnarizable++;
    }
  }
  oss << "\tZero-copy columnar conversion columns: "
      << num_columns_zero_copy_columnarizable << std::endl;

  oss << "\tPermutation size: " << permutation_.size() << std::endl;
  oss << "\tLimit: " << keep_first_ << std::endl;
  oss << "\tOffset: " << drop_first_ << std::endl;
  return oss.str();
}

ExecutorDeviceType ResultSet::getDeviceType() const {
  return device_type_;
}

const ResultSetStorage* ResultSet::allocateStorage() const {
  CHECK(!storage_);
  CHECK(row_set_mem_owner_);
  storage_buffer_size_bytes_ = query_mem_desc_.getBufferSizeBytes(device_type_);
  auto buff = row_set_mem_owner_->allocate(storage_buffer_size_bytes_, /*thread_idx=*/0);
  storage_.reset(
      new ResultSetStorage(targets_, query_mem_desc_, buff, /*buff_is_provided=*/true));
  return storage_.get();
}

const ResultSetStorage* ResultSet::allocateStorage(
    int8_t* buff,
    const std::vector<int64_t>& target_init_vals,
    std::shared_ptr<VarlenOutputInfo> varlen_output_info,
    const size_t provided_buffer_size_bytes) const {
  CHECK(buff);
  CHECK(!storage_);
  storage_buffer_size_bytes_ = provided_buffer_size_bytes
                                   ? provided_buffer_size_bytes
                                   : query_mem_desc_.getBufferSizeBytes(device_type_);
  storage_.reset(new ResultSetStorage(targets_, query_mem_desc_, buff, true));
  // TODO: add both to the constructor
  storage_->target_init_vals_ = target_init_vals;
  if (varlen_output_info) {
    storage_->varlen_output_info_ = varlen_output_info;
  }
  return storage_.get();
}

const ResultSetStorage* ResultSet::allocateStorage(
    const std::vector<int64_t>& target_init_vals) const {
  CHECK(!storage_);
  CHECK(row_set_mem_owner_);
  storage_buffer_size_bytes_ = query_mem_desc_.getBufferSizeBytes(device_type_);
  auto buff = row_set_mem_owner_->allocate(storage_buffer_size_bytes_, /*thread_idx=*/0);
  storage_.reset(
      new ResultSetStorage(targets_, query_mem_desc_, buff, /*buff_is_provided=*/true));
  storage_->target_init_vals_ = target_init_vals;
  return storage_.get();
}

size_t ResultSet::getCurrentRowBufferIndex() const {
  if (crt_row_buff_idx_ == 0) {
    throw std::runtime_error("current row buffer iteration index is undefined");
  }
  return crt_row_buff_idx_ - 1;
}

void ResultSet::append(ResultSet& that) {
  if (this == &that) {
    throw std::invalid_argument("Cannot append a ResultSet to itself");
  }
  const auto has_non_empty_columnar_fragments = [](const auto& columns) {
    return std::any_of(columns.begin(), columns.end(), [](const auto& fragments) {
      return !fragments.empty();
    });
  };
  const bool target_has_device_columnar_fragments =
      has_non_empty_columnar_fragments(device_columnar_fragments_);
  const bool source_has_device_columnar_fragments =
      has_non_empty_columnar_fragments(that.device_columnar_fragments_);
  const bool target_has_device_rowwise_fragments = !device_rowwise_fragments_.empty();
  const bool source_has_device_rowwise_fragments =
      !that.device_rowwise_fragments_.empty();
  const bool source_has_device_fragments =
      source_has_device_columnar_fragments || source_has_device_rowwise_fragments;
  if (!that.storage_ && that.appended_storage_.empty() && !source_has_device_fragments) {
    return;
  }

  if (separate_varlen_storage_valid_ != that.separate_varlen_storage_valid_) {
    throw std::runtime_error("Cannot append ResultSets with incompatible varlen storage");
  }

  const auto target_entry_count = query_mem_desc_.getEntryCount();
  const auto appended_entry_count = that.query_mem_desc_.getEntryCount();
  if (appended_entry_count > std::numeric_limits<size_t>::max() - target_entry_count) {
    throw std::overflow_error("ResultSet append entry count overflow");
  }
  size_t that_appended_storage_entry_count{0};
  for (const auto& storage : that.appended_storage_) {
    const auto storage_entry_count = storage ? storage->getEntryCount() : size_t(0);
    if (storage_entry_count >
        std::numeric_limits<size_t>::max() - that_appended_storage_entry_count) {
      throw std::overflow_error("ResultSet appended storage entry count overflow");
    }
    that_appended_storage_entry_count += storage_entry_count;
  }
  if (that_appended_storage_entry_count > appended_entry_count) {
    throw std::runtime_error(
        "ResultSet appended storage exceeds its declared entry count");
  }
  const bool appending_baseline_hash_results =
      query_mem_desc_.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByBaselineHash &&
      that.query_mem_desc_.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByBaselineHash;
  const bool combined_baseline_hash_dense =
      appending_baseline_hash_results &&
      (target_entry_count == size_t(0) || baseline_hash_dense_for_reduction_) &&
      (appended_entry_count == size_t(0) || that.baseline_hash_dense_for_reduction_);

  const auto device_columnar_covers_entries = [&](const ResultSet& result,
                                                  const bool has_columnar_fragments) {
    if (result.query_mem_desc_.getEntryCount() == size_t(0)) {
      return true;
    }
    if (!has_columnar_fragments || result.isTruncated() || !result.permutation_.empty()) {
      return false;
    }
    const auto query_type = result.query_mem_desc_.getQueryDescriptionType();
    if (query_type == QueryDescriptionType::GroupByPerfectHash ||
        query_type == QueryDescriptionType::GroupByBaselineHash) {
      return result.device_columnar_fragments_cover_logical_rows_;
    }
    if (query_type != QueryDescriptionType::Projection &&
        query_type != QueryDescriptionType::TableFunction) {
      return false;
    }
    if (result.device_columnar_fragments_.size() < result.targets_.size()) {
      return false;
    }
    for (size_t column_idx = 0; column_idx < result.targets_.size(); ++column_idx) {
      const auto& fragments = result.device_columnar_fragments_[column_idx];
      if (fragments.empty()) {
        return false;
      }
      size_t column_entry_count{0};
      for (const auto& fragment : fragments) {
        if (fragment.entry_count >
            std::numeric_limits<size_t>::max() - column_entry_count) {
          return false;
        }
        column_entry_count += fragment.entry_count;
      }
      if (column_entry_count != result.query_mem_desc_.getEntryCount()) {
        return false;
      }
    }
    return true;
  };
  const auto device_rowwise_covers_entries = [](const ResultSet& result,
                                                const bool has_rowwise_fragments) {
    if (result.query_mem_desc_.getEntryCount() == size_t(0)) {
      return true;
    }
    if (!has_rowwise_fragments || result.isTruncated() || !result.permutation_.empty()) {
      return false;
    }
    size_t rowwise_entry_count{0};
    for (const auto& fragment : result.device_rowwise_fragments_) {
      if (fragment.entry_count >
          std::numeric_limits<size_t>::max() - rowwise_entry_count) {
        return false;
      }
      rowwise_entry_count += fragment.entry_count;
    }
    return rowwise_entry_count == result.query_mem_desc_.getEntryCount();
  };

  const bool target_columnar_fragments_cover_entries =
      device_columnar_covers_entries(*this, target_has_device_columnar_fragments);
  const bool source_columnar_fragments_cover_entries =
      device_columnar_covers_entries(that, source_has_device_columnar_fragments);
  const bool combined_columnar_fragments_cover_entries =
      target_columnar_fragments_cover_entries && source_columnar_fragments_cover_entries;
  const bool combined_rowwise_fragments_cover_entries =
      device_rowwise_covers_entries(*this, target_has_device_rowwise_fragments) &&
      device_rowwise_covers_entries(that, source_has_device_rowwise_fragments);

  const auto has_valid_cpu_storage = [](const ResultSet& result) {
    const bool has_cpu_storage = result.storage_ || !result.appended_storage_.empty() ||
                                 result.query_mem_desc_.getEntryCount() == size_t(0);
    return has_cpu_storage &&
           result.device_columnar_cpu_storage_valid_.load(std::memory_order_acquire);
  };
  bool target_cpu_storage_valid = has_valid_cpu_storage(*this);
  bool source_cpu_storage_valid = has_valid_cpu_storage(that);
  if ((!target_cpu_storage_valid || !source_cpu_storage_valid) &&
      !combined_columnar_fragments_cover_entries &&
      !combined_rowwise_fragments_cover_entries) {
    if (!target_cpu_storage_valid) {
      if (!storage_) {
        throw std::runtime_error(
            "Cannot materialize target ResultSet before mixed device append");
      }
      materializeDeviceColumnarCpuStorageIfNeeded();
      target_cpu_storage_valid = true;
    }
    if (!source_cpu_storage_valid) {
      if (!that.storage_) {
        throw std::runtime_error(
            "Cannot materialize source ResultSet before mixed device append");
      }
      that.materializeDeviceColumnarCpuStorageIfNeeded();
      source_cpu_storage_valid = true;
    }
  }

  invalidateCachedRowCount();
  if (that.storage_) {
    const auto that_base_entry_count =
        appended_entry_count - that_appended_storage_entry_count;
    if (that.storage_->query_mem_desc_.getEntryCount() != that_base_entry_count) {
      that.storage_->updateEntryCount(that_base_entry_count);
    }
  }
  if (that.storage_) {
    appended_storage_.push_back(std::move(that.storage_));
  }
  for (auto& storage : that.appended_storage_) {
    if (storage) {
      appended_storage_.push_back(std::move(storage));
    }
  }
  that.appended_storage_.clear();
  query_mem_desc_.setEntryCount(target_entry_count + appended_entry_count);
  chunks_.insert(chunks_.end(), that.chunks_.begin(), that.chunks_.end());
  const auto target_col_buffer_storage_count = col_buffers_.size();
  const auto source_col_buffer_storage_count = that.col_buffers_.size();
  if (!deferred_lazy_fetch_chunks_.empty() || !that.deferred_lazy_fetch_chunks_.empty()) {
    CHECK_LE(deferred_lazy_fetch_chunks_.size(), target_col_buffer_storage_count);
    CHECK_LE(that.deferred_lazy_fetch_chunks_.size(), source_col_buffer_storage_count);
    deferred_lazy_fetch_chunks_.resize(target_col_buffer_storage_count);
    auto source_deferred_lazy_fetch_chunks = that.deferred_lazy_fetch_chunks_;
    source_deferred_lazy_fetch_chunks.resize(source_col_buffer_storage_count);
    deferred_lazy_fetch_chunks_.insert(deferred_lazy_fetch_chunks_.end(),
                                       source_deferred_lazy_fetch_chunks.begin(),
                                       source_deferred_lazy_fetch_chunks.end());
  }
  deferred_lazy_fetch_columns_materialized_for_all_rows_.clear();
  for (const auto& [local_col_idx, source_metadata] : that.lazy_fetch_source_metadata_) {
    auto& target_metadata = lazy_fetch_source_metadata_[local_col_idx];
    std::unordered_set<const ChunkMetadata*> seen_metadata;
    seen_metadata.reserve(target_metadata.size() + source_metadata.size());
    for (const auto& metadata : target_metadata) {
      seen_metadata.insert(metadata.chunk_metadata.get());
    }
    for (const auto& metadata : source_metadata) {
      if (metadata.chunk_metadata &&
          seen_metadata.insert(metadata.chunk_metadata.get()).second) {
        target_metadata.push_back(metadata);
      }
    }
  }
  col_buffers_.insert(
      col_buffers_.end(), that.col_buffers_.begin(), that.col_buffers_.end());
  col_buffer_layouts_.insert(col_buffer_layouts_.end(),
                             that.col_buffer_layouts_.begin(),
                             that.col_buffer_layouts_.end());
  frag_offsets_.insert(
      frag_offsets_.end(), that.frag_offsets_.begin(), that.frag_offsets_.end());
  consistent_frag_sizes_.insert(consistent_frag_sizes_.end(),
                                that.consistent_frag_sizes_.begin(),
                                that.consistent_frag_sizes_.end());
  chunk_iters_.insert(
      chunk_iters_.end(), that.chunk_iters_.begin(), that.chunk_iters_.end());
  if (separate_varlen_storage_valid_) {
    serialized_varlen_buffer_.insert(
        serialized_varlen_buffer_.end(),
        std::make_move_iterator(that.serialized_varlen_buffer_.begin()),
        std::make_move_iterator(that.serialized_varlen_buffer_.end()));
    that.serialized_varlen_buffer_.clear();
  }
  for (auto& buff : that.literal_buffers_) {
    literal_buffers_.push_back(std::move(buff));
  }
  if (!that.device_columnar_fragments_.empty()) {
    if (device_columnar_fragments_.size() < that.device_columnar_fragments_.size()) {
      device_columnar_fragments_.resize(that.device_columnar_fragments_.size());
    }
    for (size_t column_idx = 0; column_idx < that.device_columnar_fragments_.size();
         ++column_idx) {
      auto& target_fragments = device_columnar_fragments_[column_idx];
      auto& source_fragments = that.device_columnar_fragments_[column_idx];
      std::move(source_fragments.begin(),
                source_fragments.end(),
                std::back_inserter(target_fragments));
      source_fragments.clear();
    }
  }
  std::move(that.device_rowwise_fragments_.begin(),
            that.device_rowwise_fragments_.end(),
            std::back_inserter(device_rowwise_fragments_));
  that.device_rowwise_fragments_.clear();
  device_columnar_fragments_cover_logical_rows_ =
      combined_columnar_fragments_cover_entries;
  device_columnar_fragments_form_dense_cpu_rows_ = false;
  const bool target_boundary_exclusion_complete =
      !target_has_device_rowwise_fragments || target_entry_count == size_t(0) ||
      device_columnar_fragments_exclude_baseline_boundary_keys_;
  const bool source_boundary_exclusion_complete =
      !source_has_device_rowwise_fragments || appended_entry_count == size_t(0) ||
      that.device_columnar_fragments_exclude_baseline_boundary_keys_;
  device_columnar_fragments_exclude_baseline_boundary_keys_ =
      target_boundary_exclusion_complete && source_boundary_exclusion_complete &&
      (device_columnar_fragments_exclude_baseline_boundary_keys_ ||
       that.device_columnar_fragments_exclude_baseline_boundary_keys_);
  device_columnar_fragments_cover_cpu_baseline_boundary_rows_ =
      device_columnar_fragments_cover_cpu_baseline_boundary_rows_ ||
      that.device_columnar_fragments_cover_cpu_baseline_boundary_rows_;
  if (!target_cpu_storage_valid || !source_cpu_storage_valid) {
    device_columnar_cpu_storage_valid_.store(false, std::memory_order_release);
    if (storage_ && !device_rowwise_fragments_.empty()) {
      const auto device_rowwise_entry_count = std::accumulate(
          device_rowwise_fragments_.begin(),
          device_rowwise_fragments_.end(),
          size_t(0),
          [](const size_t total, const DeviceRowwiseBufferFragment& fragment) {
            CHECK(fragment.buffer);
            CHECK(fragment.owner);
            return total + fragment.entry_count;
          });
      if (device_rowwise_entry_count == query_mem_desc_.getEntryCount() &&
          appended_storage_.empty()) {
        storage_->updateEntryCount(device_rowwise_entry_count);
      } else if (!device_columnar_fragments_.empty() && appended_storage_.empty()) {
        storage_->updateEntryCount(query_mem_desc_.getEntryCount());
      }
    } else if (storage_ && appended_storage_.empty() &&
               !device_columnar_fragments_.empty()) {
      storage_->updateEntryCount(query_mem_desc_.getEntryCount());
    }
    if (query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection) {
      setCachedRowCount(query_mem_desc_.getEntryCount());
    }
  }
  if (query_mem_desc_.getQueryDescriptionType() ==
      QueryDescriptionType::GroupByBaselineHash) {
    baseline_hash_dense_for_reduction_ = combined_baseline_hash_dense;
    if (combined_baseline_hash_dense) {
      setCachedRowCount(query_mem_desc_.getEntryCount());
    }
  }
}

ResultSetPtr ResultSet::copyForCacheInsertion() {
  return copyForCache();
}

ResultSetPtr ResultSet::copyForCacheRetrieval() {
  return copyForCache();
}

ResultSetPtr ResultSet::copyForCache() {
  auto timer = DEBUG_TIMER(__func__);
  if (!storage_) {
    return nullptr;
  }
  // The ResultSet recycler is host-byte-accounted. Materialize a self-contained host
  // value and leave query-scoped device views out of the persistent cache entry.
  materializeDeviceColumnarCpuStorageIfNeeded();

  auto executor = getExecutor();
  CHECK(executor);
  ResultSetPtr copied_rs = std::make_shared<ResultSet>(targets_,
                                                       device_type_,
                                                       query_mem_desc_,
                                                       row_set_mem_owner_,
                                                       executor->blockSize(),
                                                       executor->gridSize());

  auto allocate_and_copy_storage =
      [&](const ResultSetStorage* prev_storage) -> std::unique_ptr<ResultSetStorage> {
    const auto& prev_qmd = prev_storage->query_mem_desc_;
    const auto storage_size = prev_qmd.getBufferSizeBytes(device_type_);
    auto buff = row_set_mem_owner_->allocate(storage_size, /*thread_idx=*/0);
    std::unique_ptr<ResultSetStorage> new_storage;
    new_storage.reset(new ResultSetStorage(
        prev_storage->targets_, prev_qmd, buff, /*buff_is_provided=*/true));
    new_storage->target_init_vals_ = prev_storage->target_init_vals_;
    if (prev_storage->varlen_output_info_) {
      new_storage->varlen_output_info_ = prev_storage->varlen_output_info_;
    }
    memcpy(new_storage->buff_, prev_storage->buff_, storage_size);
    new_storage->query_mem_desc_ = prev_qmd;
    return new_storage;
  };

  copied_rs->storage_ = allocate_and_copy_storage(storage_.get());
  if (!appended_storage_.empty()) {
    for (const auto& storage : appended_storage_) {
      copied_rs->appended_storage_.push_back(allocate_and_copy_storage(storage.get()));
    }
  }
  std::copy(chunks_.begin(), chunks_.end(), std::back_inserter(copied_rs->chunks_));
  std::copy(chunk_iters_.begin(),
            chunk_iters_.end(),
            std::back_inserter(copied_rs->chunk_iters_));
  std::copy(col_buffers_.begin(),
            col_buffers_.end(),
            std::back_inserter(copied_rs->col_buffers_));
  std::copy(deferred_lazy_fetch_chunks_.begin(),
            deferred_lazy_fetch_chunks_.end(),
            std::back_inserter(copied_rs->deferred_lazy_fetch_chunks_));
  copied_rs->deferred_lazy_fetch_columns_materialized_for_all_rows_ =
      deferred_lazy_fetch_columns_materialized_for_all_rows_;
  copied_rs->lazy_fetch_source_metadata_ = lazy_fetch_source_metadata_;
  std::copy(col_buffer_layouts_.begin(),
            col_buffer_layouts_.end(),
            std::back_inserter(copied_rs->col_buffer_layouts_));
  std::copy(frag_offsets_.begin(),
            frag_offsets_.end(),
            std::back_inserter(copied_rs->frag_offsets_));
  std::copy(consistent_frag_sizes_.begin(),
            consistent_frag_sizes_.end(),
            std::back_inserter(copied_rs->consistent_frag_sizes_));
  if (separate_varlen_storage_valid_) {
    std::copy(serialized_varlen_buffer_.begin(),
              serialized_varlen_buffer_.end(),
              std::back_inserter(copied_rs->serialized_varlen_buffer_));
  }
  std::copy(literal_buffers_.begin(),
            literal_buffers_.end(),
            std::back_inserter(copied_rs->literal_buffers_));
  std::copy(lazy_fetch_info_.begin(),
            lazy_fetch_info_.end(),
            std::back_inserter(copied_rs->lazy_fetch_info_));

  copied_rs->permutation_ = permutation_;
  copied_rs->drop_first_ = drop_first_;
  copied_rs->keep_first_ = keep_first_;
  copied_rs->separate_varlen_storage_valid_ = separate_varlen_storage_valid_;
  copied_rs->baseline_hash_dense_for_reduction_ = baseline_hash_dense_for_reduction_;
  copied_rs->query_exec_time_ = query_exec_time_;
  copied_rs->input_table_keys_ = input_table_keys_;
  copied_rs->target_meta_info_ = target_meta_info_;
  copied_rs->geo_return_type_ = geo_return_type_;
  copied_rs->query_plan_ = query_plan_;
  if (can_use_speculative_top_n_sort) {
    copied_rs->can_use_speculative_top_n_sort = can_use_speculative_top_n_sort;
  }

  return copied_rs;
}

namespace {

bool count_distinct_descriptors_safe_for_group_key_output(
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t target_count) {
  if (query_mem_desc.countDistinctDescriptorsLogicallyEmpty()) {
    return true;
  }
  if (target_count == 0 || query_mem_desc.targetGroupbyIndicesSize() != target_count) {
    return false;
  }
  for (size_t target_idx = 0; target_idx < target_count; ++target_idx) {
    if (query_mem_desc.getTargetGroupbyIndex(target_idx) < 0) {
      return false;
    }
  }
  return true;
}

bool can_compact_baseline_hash_for_reduction(const ResultSet& result_set) {
  if (!result_set.hasStorage()) {
    return false;
  }
  const auto& query_mem_desc = result_set.getQueryMemDesc();
  if (query_mem_desc.getQueryDescriptionType() !=
          QueryDescriptionType::GroupByBaselineHash ||
      query_mem_desc.hasKeylessHash()) {
    return false;
  }
  if (query_mem_desc.hasVarlenOutput() ||
      !count_distinct_descriptors_safe_for_group_key_output(query_mem_desc,
                                                            result_set.colCount()) ||
      query_mem_desc.getNumModeTargets() > 0) {
    return false;
  }
  if (query_mem_desc.didOutputColumnar()) {
    for (size_t slot_idx = 0; slot_idx < query_mem_desc.getBufferColSlotCount();
         ++slot_idx) {
      if (query_mem_desc.getPaddedSlotWidthBytes(slot_idx) <= 0) {
        return false;
      }
    }
  }
  for (const auto& target : result_set.getTargetInfos()) {
    if (is_distinct_target(target) || target.sql_type.is_varlen() ||
        target.sql_type.is_array() || target.sql_type.is_geometry() ||
        target.agg_kind == kAPPROX_QUANTILE || target.agg_kind == kMODE) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool compare_scalar_values(const T lhs, const T rhs, const SQLOps op) {
  switch (op) {
    case kEQ:
    case kBW_EQ:
      return lhs == rhs;
    case kNE:
      return lhs != rhs;
    case kLT:
      return lhs < rhs;
    case kGT:
      return lhs > rhs;
    case kLE:
      return lhs <= rhs;
    case kGE:
      return lhs >= rhs;
    default:
      return false;
  }
}

std::optional<int64_t> checked_scale_decimal_value(const int64_t value,
                                                   const unsigned scale) {
  const auto unsigned_factor = exp_to_scale(scale);
  if (unsigned_factor > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  const auto factor = static_cast<int64_t>(unsigned_factor);
  int64_t scaled_value{0};
  if (__builtin_mul_overflow(value, factor, &scaled_value)) {
    return std::nullopt;
  }
  return scaled_value;
}

std::optional<int64_t> literal_as_integral_value(const ResultSetEntryLiteral& literal,
                                                 const SQLTypeInfo& target_type) {
  if (literal.is_null || literal.type_info.is_fp() || target_type.is_fp()) {
    return std::nullopt;
  }
  if (literal.type_info.is_decimal()) {
    if (target_type.is_decimal()) {
      return convert_decimal_value_to_scale(
          literal.int_val, literal.type_info, target_type);
    }
    // Scale-zero decimal literals are exact integers. Calcite commonly uses this
    // representation for integral constants in aggregate filters.
    return literal.type_info.get_scale() == 0 ? std::optional<int64_t>(literal.int_val)
                                              : std::nullopt;
  }
  if (target_type.is_decimal()) {
    return checked_scale_decimal_value(literal.int_val, target_type.get_scale());
  }
  if (literal.type_info.get_type() == kBOOLEAN) {
    return literal.bool_val ? 1 : 0;
  }
  return literal.int_val;
}

std::optional<double> literal_as_fp_value(const ResultSetEntryLiteral& literal,
                                          const SQLTypeInfo& target_type) {
  if (literal.is_null) {
    return std::nullopt;
  }
  if (literal.type_info.is_fp()) {
    return literal.double_val;
  }
  if (literal.type_info.is_decimal()) {
    return literal.int_val /
           static_cast<double>(exp_to_scale(literal.type_info.get_scale()));
  }
  if (literal.type_info.get_type() == kBOOLEAN) {
    return literal.bool_val ? 1.0 : 0.0;
  }
  if (target_type.is_decimal()) {
    return literal.int_val * static_cast<double>(exp_to_scale(target_type.get_scale()));
  }
  return static_cast<double>(literal.int_val);
}

const int8_t* get_entry_target_ptr(const ResultSetStorage& storage,
                                   const size_t entry_idx,
                                   const size_t target_idx) {
  const auto& query_mem_desc = storage.getQueryMemDesc();
  const auto* const source_buffer = storage.getUnderlyingBuffer();

  if (query_mem_desc.targetGroupbyIndicesSize() > 0) {
    const auto groupby_idx = query_mem_desc.getTargetGroupbyIndex(target_idx);
    if (groupby_idx >= 0) {
      if (query_mem_desc.didOutputColumnar()) {
        const auto group_width = query_mem_desc.groupColWidth(groupby_idx);
        const auto physical_group_width =
            std::max(static_cast<size_t>(group_width), sizeof(int64_t));
        return source_buffer +
               query_mem_desc.getPrependedGroupColOffInBytes(groupby_idx) +
               entry_idx * physical_group_width;
      }
      const auto* row_ptr = row_ptr_rowwise(source_buffer, query_mem_desc, entry_idx);
      return row_ptr + groupby_idx * query_mem_desc.getEffectiveKeyWidth();
    }
  }

  const auto slot_idx = query_mem_desc.getSlotIndexForSingleSlotCol(target_idx);
  const auto slot_width = query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
  CHECK_GT(slot_width, 0);
  if (query_mem_desc.didOutputColumnar()) {
    return source_buffer + query_mem_desc.getColOffInBytes(slot_idx) +
           entry_idx * slot_width;
  }
  const auto* row_ptr = row_ptr_rowwise(source_buffer, query_mem_desc, entry_idx);
  return row_ptr + align_to_int64(get_key_bytes_rowwise(query_mem_desc)) +
         result_set::get_byteoff_of_slot(slot_idx, query_mem_desc);
}

int8_t get_entry_target_width(const ResultSetStorage& storage, const size_t target_idx) {
  const auto& query_mem_desc = storage.getQueryMemDesc();
  if (query_mem_desc.targetGroupbyIndicesSize() > 0) {
    const auto groupby_idx = query_mem_desc.getTargetGroupbyIndex(target_idx);
    if (groupby_idx >= 0) {
      return query_mem_desc.didOutputColumnar()
                 ? query_mem_desc.groupColWidth(groupby_idx)
                 : query_mem_desc.getEffectiveKeyWidth();
    }
  }
  return query_mem_desc.getPaddedSlotWidthBytes(
      query_mem_desc.getSlotIndexForSingleSlotCol(target_idx));
}

bool result_set_entry_matches(const ResultSet& result_set,
                              const ResultSetStorage& storage,
                              const size_t entry_idx,
                              const ResultSetEntryComparison& comparison) {
  const auto& targets = result_set.getTargetInfos();
  if (comparison.target_idx >= targets.size()) {
    return false;
  }
  const auto& target_info = targets[comparison.target_idx];
  if (target_info.agg_kind == kAVG || target_info.sql_type.is_varlen() ||
      target_info.sql_type.is_array() || target_info.sql_type.is_geometry()) {
    return false;
  }
  const auto* target_ptr =
      get_entry_target_ptr(storage, entry_idx, comparison.target_idx);
  const auto target_width = get_entry_target_width(storage, comparison.target_idx);
  const auto& target_type = target_info.sql_type;
  const auto target_bits = read_int_from_buff(target_ptr, target_width);
  const auto target_null_bits =
      null_val_bit_pattern(target_type, takes_float_argument(target_info));
  if (!target_type.get_notnull() && target_bits == target_null_bits) {
    return false;
  }

  if (target_type.is_fp()) {
    const auto literal = literal_as_fp_value(comparison.literal, target_type);
    if (!literal) {
      return false;
    }
    double target_value{0.0};
    if (target_type.get_type() == kFLOAT) {
      const auto float_bits = static_cast<int32_t>(target_bits);
      target_value = static_cast<double>(shared::reinterpret_bits<float>(float_bits));
    } else {
      CHECK_EQ(target_type.get_type(), kDOUBLE);
      target_value = shared::reinterpret_bits<double>(target_bits);
    }
    return compare_scalar_values(target_value, *literal, comparison.op);
  }

  const auto literal = literal_as_integral_value(comparison.literal, target_type);
  if (!literal) {
    return false;
  }
  return compare_scalar_values(target_bits, *literal, comparison.op);
}

bool result_set_entry_matches(const ResultSet& result_set,
                              const ResultSetStorage& storage,
                              const size_t entry_idx,
                              const ResultSetEntryFilter* entry_filter) {
  if (!entry_filter || entry_filter->empty()) {
    return true;
  }
  for (const auto& comparison : *entry_filter) {
    if (!result_set_entry_matches(result_set, storage, entry_idx, comparison)) {
      return false;
    }
  }
  return true;
}

struct ResultSetStorageEntryRange {
  const ResultSetStorage* storage;
  size_t start;
  size_t end;
};

std::vector<ResultSetStorageEntryRange> make_storage_entry_ranges(
    const std::vector<const ResultSetStorage*>& storages,
    const size_t source_entry_count,
    const size_t thread_count) {
  CHECK_GT(thread_count, size_t(0));
  std::vector<ResultSetStorageEntryRange> ranges;
  const auto entries_per_range = std::max<size_t>(
      1, source_entry_count / thread_count + (source_entry_count % thread_count != 0));
  for (const auto* storage : storages) {
    CHECK(storage);
    const auto storage_entry_count = storage->getEntryCount();
    for (size_t start = 0; start < storage_entry_count;) {
      const auto remaining_entries = storage_entry_count - start;
      const auto end = remaining_entries <= entries_per_range ? storage_entry_count
                                                              : start + entries_per_range;
      ranges.push_back(ResultSetStorageEntryRange{storage, start, end});
      start = end;
    }
  }
  return ranges;
}

}  // namespace

ResultSetPtr ResultSet::compactBaselineHashForReduction(
    const size_t min_compaction_entry_count,
    const ResultSetEntryFilter* entry_filter) const {
  const auto has_entry_filter = entry_filter && !entry_filter->empty();
  if (!can_compact_baseline_hash_for_reduction(*this) || !permutation_.empty() ||
      separate_varlen_storage_valid_) {
    return nullptr;
  }
  if (!device_columnar_cpu_storage_valid_.load(std::memory_order_acquire) &&
      !device_columnar_fragments_.empty()) {
    return nullptr;
  }
  materializeDeviceColumnarCpuStorageIfNeeded();

  std::vector<const ResultSetStorage*> source_storages;
  source_storages.reserve(appended_storage_.size() + 1);
  source_storages.push_back(storage_.get());
  for (const auto& storage : appended_storage_) {
    source_storages.push_back(storage.get());
  }
  size_t source_entry_count{0};
  for (const auto* source_storage : source_storages) {
    const auto storage_entry_count = source_storage->getEntryCount();
    if (storage_entry_count > std::numeric_limits<size_t>::max() - source_entry_count) {
      throw std::overflow_error("ResultSet compaction entry count overflow");
    }
    source_entry_count += storage_entry_count;
  }
  if (source_entry_count < min_compaction_entry_count) {
    return nullptr;
  }

  const auto thread_count =
      std::max<size_t>(1, std::min<size_t>(cpu_threads(), source_entry_count));
  const auto ranges =
      make_storage_entry_ranges(source_storages, source_entry_count, thread_count);
  std::vector<size_t> row_counts(ranges.size());
  threading::parallel_for(
      threading::blocked_range<size_t>(0, ranges.size()),
      [&](const threading::blocked_range<size_t>& thread_range) {
        for (size_t thread_idx = thread_range.begin(); thread_idx != thread_range.end();
             ++thread_idx) {
          const auto [source_storage, start, end] = ranges[thread_idx];
          size_t row_count{0};
          for (size_t entry_idx = start; entry_idx < end; ++entry_idx) {
            if (!source_storage->isEmptyEntry(entry_idx) &&
                result_set_entry_matches(
                    *this, *source_storage, entry_idx, entry_filter)) {
              ++row_count;
            }
          }
          row_counts[thread_idx] = row_count;
        }
      });

  std::vector<size_t> row_offsets(row_counts.size() + 1);
  std::partial_sum(row_counts.begin(), row_counts.end(), row_offsets.begin() + 1);
  const auto compact_row_count = row_offsets.back();
  if (compact_row_count == 0) {
    if (!has_entry_filter) {
      return nullptr;
    }
    auto empty_query_mem_desc = query_mem_desc_;
    empty_query_mem_desc.setEntryCount(0);
    auto empty_result_set = std::make_shared<ResultSet>(targets_,
                                                        ExecutorDeviceType::CPU,
                                                        empty_query_mem_desc,
                                                        row_set_mem_owner_,
                                                        block_size_,
                                                        grid_size_);
    empty_result_set->allocateStorage(storage_->target_init_vals_);
    empty_result_set->markBaselineHashDenseForReduction(0);
    return empty_result_set;
  }

  // Dense compaction pays for a second scan and allocation, so keep nearly dense sources
  // on the existing reducer path.
  constexpr size_t max_dense_numerator = 3;
  constexpr size_t max_dense_denominator = 4;
  if (!has_entry_filter &&
      static_cast<unsigned __int128>(compact_row_count) * max_dense_denominator >=
          static_cast<unsigned __int128>(source_entry_count) * max_dense_numerator) {
    return nullptr;
  }

  auto compact_query_mem_desc = query_mem_desc_;
  compact_query_mem_desc.setEntryCount(compact_row_count);
  auto compact_result_set = std::make_shared<ResultSet>(targets_,
                                                        ExecutorDeviceType::CPU,
                                                        compact_query_mem_desc,
                                                        row_set_mem_owner_,
                                                        block_size_,
                                                        grid_size_);
  auto compact_storage = const_cast<ResultSetStorage*>(
      compact_result_set->allocateStorage(storage_->target_init_vals_));
  auto* const compact_buffer = compact_storage->buff_;

  auto copy_entry = [&](const ResultSetStorage* source_storage,
                        const size_t source_idx,
                        const size_t compact_idx) {
    CHECK(source_storage);
    const auto& source_query_mem_desc = source_storage->getQueryMemDesc();
    auto* const source_buffer = source_storage->getUnderlyingBuffer();
    if (!source_query_mem_desc.didOutputColumnar()) {
      const auto row_size = get_row_bytes(source_query_mem_desc);
      memcpy(compact_buffer + compact_idx * row_size,
             source_buffer + source_idx * row_size,
             row_size);
      return;
    }
    for (size_t group_idx = 0; group_idx < source_query_mem_desc.getGroupbyColCount();
         ++group_idx) {
      const auto group_width = std::max<size_t>(
          static_cast<size_t>(source_query_mem_desc.groupColWidth(group_idx)),
          sizeof(int64_t));
      const auto source_offset =
          source_query_mem_desc.getPrependedGroupColOffInBytes(group_idx) +
          source_idx * group_width;
      const auto compact_offset =
          compact_query_mem_desc.getPrependedGroupColOffInBytes(group_idx) +
          compact_idx * group_width;
      memcpy(compact_buffer + compact_offset, source_buffer + source_offset, group_width);
    }
    for (size_t slot_idx = 0; slot_idx < source_query_mem_desc.getBufferColSlotCount();
         ++slot_idx) {
      const auto slot_width =
          static_cast<size_t>(source_query_mem_desc.getPaddedSlotWidthBytes(slot_idx));
      const auto source_offset =
          source_query_mem_desc.getColOffInBytes(slot_idx) + source_idx * slot_width;
      const auto compact_offset =
          compact_query_mem_desc.getColOffInBytes(slot_idx) + compact_idx * slot_width;
      memcpy(compact_buffer + compact_offset, source_buffer + source_offset, slot_width);
    }
  };

  threading::parallel_for(
      threading::blocked_range<size_t>(0, ranges.size()),
      [&](const threading::blocked_range<size_t>& thread_range) {
        for (size_t thread_idx = thread_range.begin(); thread_idx != thread_range.end();
             ++thread_idx) {
          const auto [source_storage, start, end] = ranges[thread_idx];
          const auto compact_start = row_offsets[thread_idx];
          auto compact_idx = compact_start;
          for (size_t source_idx = start; source_idx < end; ++source_idx) {
            if (!source_storage->isEmptyEntry(source_idx) &&
                result_set_entry_matches(
                    *this, *source_storage, source_idx, entry_filter)) {
              copy_entry(source_storage, source_idx, compact_idx++);
            }
          }
        }
      });

  compact_result_set->setCachedRowCount(compact_row_count);
  compact_result_set->markBaselineHashDenseForReduction(compact_row_count);
  return compact_result_set;
}

ResultSetPtr ResultSet::extractAndClearBaselineHashEntries(
    const std::vector<int64_t>& keys,
    const bool retain_device_rowwise_for_post_filter) {
  (void)retain_device_rowwise_for_post_filter;
  if (!storage_ || keys.empty() || !permutation_.empty() ||
      query_mem_desc_.getQueryDescriptionType() !=
          QueryDescriptionType::GroupByBaselineHash ||
      query_mem_desc_.hasKeylessHash() || query_mem_desc_.didOutputColumnar() ||
      query_mem_desc_.getGroupbyColCount() != size_t(1)) {
    return nullptr;
  }

#ifdef HAVE_CUDA
  if (!device_columnar_cpu_storage_valid_.load(std::memory_order_acquire) &&
      !device_rowwise_fragments_.empty() &&
      device_columnar_fragments_exclude_baseline_boundary_keys_) {
    std::vector<int64_t> sorted_keys = keys;
    std::sort(sorted_keys.begin(), sorted_keys.end());
    sorted_keys.erase(std::unique(sorted_keys.begin(), sorted_keys.end()),
                      sorted_keys.end());
    if (sorted_keys.empty()) {
      return nullptr;
    }
    const auto row_size = get_row_bytes(query_mem_desc_);
    const auto key_width = static_cast<size_t>(query_mem_desc_.getEffectiveKeyWidth());
    const auto key_bytes = sorted_keys.size() * sizeof(int64_t);
    std::vector<std::vector<int8_t>> fragment_rows;
    size_t found_row_count{0};

    for (const auto& fragment : device_rowwise_fragments_) {
      CHECK(fragment.buffer);
      CHECK(fragment.owner);
      const auto max_matching_rows = std::min(fragment.entry_count, sorted_keys.size());
      if (max_matching_rows == 0) {
        continue;
      }
      CudaAllocator scratch_allocator(fragment.owner->getDataMgr(),
                                      fragment.device_id,
                                      fragment.owner->getCudaStream());
      auto* device_keys = reinterpret_cast<int64_t*>(scratch_allocator.alloc(key_bytes));
      scratch_allocator.copyToDevice(
          device_keys, sorted_keys.data(), key_bytes, "Deferred baseline boundary keys");
      auto* matching_rows = scratch_allocator.alloc(max_matching_rows * row_size);
      auto* matching_row_count =
          reinterpret_cast<uint64_t*>(scratch_allocator.alloc(sizeof(uint64_t)));
      compact_baseline_hash_rows_matching_keys_on_device(
          const_cast<int8_t*>(fragment.buffer),
          matching_rows,
          matching_row_count,
          fragment.entry_count,
          row_size,
          key_width,
          device_keys,
          sorted_keys.size(),
          true,
          fragment.device_id,
          fragment.owner ? fragment.owner->getCudaStream() : 0);
      uint64_t host_matching_row_count{0};
      scratch_allocator.copyFromDevice(&host_matching_row_count,
                                       matching_row_count,
                                       sizeof(host_matching_row_count),
                                       "Deferred baseline boundary row count");
      CHECK_LE(static_cast<size_t>(host_matching_row_count), max_matching_rows);
      if (host_matching_row_count == 0) {
        continue;
      }
      auto& rows = fragment_rows.emplace_back(host_matching_row_count * row_size);
      scratch_allocator.copyFromDevice(
          rows.data(), matching_rows, rows.size(), "Deferred baseline boundary rows");
      found_row_count += host_matching_row_count;
    }

    if (found_row_count == 0) {
      return nullptr;
    }

    auto compact_query_mem_desc = query_mem_desc_;
    compact_query_mem_desc.setEntryCount(found_row_count);
    auto compact_result_set = std::make_shared<ResultSet>(targets_,
                                                          ExecutorDeviceType::CPU,
                                                          compact_query_mem_desc,
                                                          row_set_mem_owner_,
                                                          block_size_,
                                                          grid_size_);
    auto compact_storage = const_cast<ResultSetStorage*>(
        compact_result_set->allocateStorage(storage_->target_init_vals_));
    size_t output_row_idx{0};
    for (const auto& rows : fragment_rows) {
      CHECK_EQ(size_t(0), rows.size() % row_size);
      memcpy(
          compact_storage->buff_ + output_row_idx * row_size, rows.data(), rows.size());
      output_row_idx += rows.size() / row_size;
    }
    CHECK_EQ(output_row_idx, found_row_count);

    if (retain_device_rowwise_for_post_filter) {
      // Boundary rows have been cleared from the sparse rowwise buffer. Preserve its
      // allocation shape until the remaining disjoint rows are filtered and compacted.
      clearDeviceColumnarBufferFragments();
      invalidateCachedRowCount();
      clearBaselineHashDenseForReduction();
      compact_result_set->setCachedRowCount(found_row_count);
      compact_result_set->markBaselineHashDenseForReduction(found_row_count);
      return compact_result_set;
    }

    const bool remains_dense_for_reduction =
        device_columnar_fragments_exclude_baseline_boundary_keys_;
    if (!remains_dense_for_reduction) {
      clearDeviceColumnarBufferFragments();
    } else {
      clearDeviceRowwiseBufferFragments();
    }
    invalidateCachedRowCount();
    CHECK_GE(query_mem_desc_.getEntryCount(), found_row_count);
    auto remaining_row_count = query_mem_desc_.getEntryCount() - found_row_count;
    if (remains_dense_for_reduction && !device_columnar_fragments_.empty()) {
      size_t columnar_row_count{0};
      bool initialized_columnar_row_count{false};
      bool consistent_columnar_row_count{true};
      for (const auto& column_fragments : device_columnar_fragments_) {
        if (column_fragments.empty()) {
          consistent_columnar_row_count = false;
          break;
        }
        const auto column_row_count = std::accumulate(
            column_fragments.begin(),
            column_fragments.end(),
            size_t(0),
            [](const size_t total, const DeviceColumnarBufferFragment& fragment) {
              CHECK(fragment.buffer);
              CHECK(fragment.owner);
              return total + fragment.entry_count;
            });
        if (!initialized_columnar_row_count) {
          columnar_row_count = column_row_count;
          initialized_columnar_row_count = true;
        } else if (columnar_row_count != column_row_count) {
          consistent_columnar_row_count = false;
          break;
        }
      }
      if (consistent_columnar_row_count && initialized_columnar_row_count) {
        remaining_row_count = columnar_row_count;
      }
    }
    query_mem_desc_.setEntryCount(remaining_row_count);
    if (storage_) {
      storage_->updateEntryCount(remaining_row_count);
    }
    setCachedRowCount(remaining_row_count);
    if (remains_dense_for_reduction) {
      markBaselineHashDenseForReduction(remaining_row_count);
    } else {
      clearBaselineHashDenseForReduction();
    }
    compact_result_set->setCachedRowCount(found_row_count);
    compact_result_set->markBaselineHashDenseForReduction(found_row_count);
    return compact_result_set;
  }
#endif

  materializeDeviceColumnarCpuStorageIfNeeded();

  std::vector<ResultSetStorage*> source_storages;
  source_storages.reserve(appended_storage_.size() + 1);
  source_storages.push_back(storage_.get());
  for (auto& storage : appended_storage_) {
    source_storages.push_back(storage.get());
  }
  const auto source_entry_count =
      std::accumulate(source_storages.begin(),
                      source_storages.end(),
                      size_t(0),
                      [](const auto total, const auto* storage) {
                        return total + storage->getEntryCount();
                      });
  if (!source_entry_count) {
    return nullptr;
  }

  auto read_entry_key = [](const ResultSetStorage& source_storage,
                           const size_t entry_idx) -> std::optional<int64_t> {
    const auto& source_query_mem_desc = source_storage.getQueryMemDesc();
    auto* row_ptr = row_ptr_rowwise(
        source_storage.getUnderlyingBuffer(), source_query_mem_desc, entry_idx);
    switch (source_query_mem_desc.getEffectiveKeyWidth()) {
      case 1: {
        const auto entry_key = *reinterpret_cast<const int8_t*>(row_ptr);
        if (entry_key == EMPTY_KEY_8) {
          return std::nullopt;
        }
        return static_cast<int64_t>(entry_key);
      }
      case 2: {
        const auto entry_key = *reinterpret_cast<const int16_t*>(row_ptr);
        if (entry_key == EMPTY_KEY_16) {
          return std::nullopt;
        }
        return static_cast<int64_t>(entry_key);
      }
      case 4: {
        const auto entry_key = *reinterpret_cast<const int32_t*>(row_ptr);
        if (entry_key == EMPTY_KEY_32) {
          return std::nullopt;
        }
        return static_cast<int64_t>(entry_key);
      }
      case 8: {
        const auto entry_key = *reinterpret_cast<const int64_t*>(row_ptr);
        if (entry_key == EMPTY_KEY_64) {
          return std::nullopt;
        }
        return entry_key;
      }
      default:
        return std::nullopt;
    }
  };

  auto clear_entry_key = [](ResultSetStorage& source_storage, const size_t entry_idx) {
    const auto& source_query_mem_desc = source_storage.getQueryMemDesc();
    auto* row_ptr = row_ptr_rowwise(
        source_storage.getUnderlyingBuffer(), source_query_mem_desc, entry_idx);
    switch (source_query_mem_desc.getEffectiveKeyWidth()) {
      case 1:
        *reinterpret_cast<int8_t*>(row_ptr) = EMPTY_KEY_8;
        break;
      case 2:
        *reinterpret_cast<int16_t*>(row_ptr) = EMPTY_KEY_16;
        break;
      case 4:
        *reinterpret_cast<int32_t*>(row_ptr) = EMPTY_KEY_32;
        break;
      case 8:
        *reinterpret_cast<int64_t*>(row_ptr) = EMPTY_KEY_64;
        break;
      default:
        CHECK(false);
    }
  };

  struct FoundEntry {
    ResultSetStorage* storage;
    size_t entry_idx;
  };

  std::vector<FoundEntry> found_entries;
  const std::unordered_set<int64_t> key_set(keys.begin(), keys.end());
  auto scan_entries = [&](ResultSetStorage* source_storage,
                          const size_t start,
                          const size_t end,
                          std::vector<FoundEntry>& local_found_entries) {
    CHECK(source_storage);
    for (size_t entry_idx = start; entry_idx < end; ++entry_idx) {
      const auto entry_key = read_entry_key(*source_storage, entry_idx);
      if (entry_key && key_set.count(*entry_key)) {
        local_found_entries.push_back(FoundEntry{source_storage, entry_idx});
      }
    }
  };

  auto scan_source_storages = [&]() {
    found_entries.clear();
    if (source_entry_count < auto_parallel_row_count_threshold) {
      for (auto* source_storage : source_storages) {
        scan_entries(source_storage, 0, source_storage->getEntryCount(), found_entries);
      }
    } else {
      const auto thread_count =
          std::max<size_t>(1, std::min<size_t>(cpu_threads(), source_entry_count));
      std::vector<const ResultSetStorage*> const_source_storages;
      const_source_storages.reserve(source_storages.size());
      for (const auto* source_storage : source_storages) {
        const_source_storages.push_back(source_storage);
      }
      const auto ranges = make_storage_entry_ranges(
          const_source_storages, source_entry_count, thread_count);
      std::vector<std::vector<FoundEntry>> thread_found_entries(ranges.size());
      threading::parallel_for(
          threading::blocked_range<size_t>(0, ranges.size()),
          [&](const threading::blocked_range<size_t>& thread_range) {
            for (size_t thread_idx = thread_range.begin();
                 thread_idx != thread_range.end();
                 ++thread_idx) {
              const auto [source_storage, start, end] = ranges[thread_idx];
              scan_entries(const_cast<ResultSetStorage*>(source_storage),
                           start,
                           end,
                           thread_found_entries[thread_idx]);
            }
          });
      for (auto& local_found_entries : thread_found_entries) {
        found_entries.insert(
            found_entries.end(), local_found_entries.begin(), local_found_entries.end());
      }
    }
    std::sort(
        found_entries.begin(), found_entries.end(), [](const auto& lhs, const auto& rhs) {
          return std::tie(lhs.storage, lhs.entry_idx) <
                 std::tie(rhs.storage, rhs.entry_idx);
        });
  };

  if (baseline_hash_dense_for_reduction_ || source_storages.size() > 1) {
    scan_source_storages();
  } else {
    auto* source_storage = source_storages.front();
    const auto& source_query_mem_desc = source_storage->getQueryMemDesc();
    const auto entry_count = source_storage->getEntryCount();
    auto find_entry_for_key = [&](const int64_t key) -> std::optional<FoundEntry> {
      const auto key_width = source_query_mem_desc.getEffectiveKeyWidth();
      int8_t key8{0};
      int16_t key16{0};
      int32_t key32{0};
      const void* hash_key{nullptr};
      int hash_key_bytes{0};
      switch (key_width) {
        case 1:
          key8 = static_cast<int8_t>(key);
          if (static_cast<int64_t>(key8) != key) {
            return std::nullopt;
          }
          hash_key = &key8;
          hash_key_bytes = sizeof(key8);
          break;
        case 2:
          key16 = static_cast<int16_t>(key);
          if (static_cast<int64_t>(key16) != key) {
            return std::nullopt;
          }
          hash_key = &key16;
          hash_key_bytes = sizeof(key16);
          break;
        case 4:
          key32 = static_cast<int32_t>(key);
          if (static_cast<int64_t>(key32) != key) {
            return std::nullopt;
          }
          hash_key = &key32;
          hash_key_bytes = sizeof(key32);
          break;
        case 8:
          hash_key = &key;
          hash_key_bytes = sizeof(key);
          break;
        default:
          return std::nullopt;
      }

      const auto start_idx = MurmurHash3(hash_key, hash_key_bytes, 0) % entry_count;
      for (size_t probe_count = 0; probe_count < entry_count; ++probe_count) {
        const auto entry_idx = (start_idx + probe_count) % entry_count;
        const auto entry_key = read_entry_key(*source_storage, entry_idx);
        if (!entry_key) {
          return std::nullopt;
        }
        if (*entry_key == key) {
          return FoundEntry{source_storage, entry_idx};
        }
      }
      return std::nullopt;
    };

    std::set<std::pair<ResultSetStorage*, size_t>> unique_entries;
    for (const auto key : keys) {
      if (auto found_entry = find_entry_for_key(key)) {
        unique_entries.insert({found_entry->storage, found_entry->entry_idx});
      }
    }
    for (const auto& [source_storage, entry_idx] : unique_entries) {
      found_entries.push_back(FoundEntry{source_storage, entry_idx});
    }
    if (found_entries.size() < key_set.size()) {
      scan_source_storages();
    }
  }
  if (found_entries.empty()) {
    return nullptr;
  }

  auto compact_query_mem_desc = query_mem_desc_;
  compact_query_mem_desc.setEntryCount(found_entries.size());
  auto compact_result_set = std::make_shared<ResultSet>(targets_,
                                                        ExecutorDeviceType::CPU,
                                                        compact_query_mem_desc,
                                                        row_set_mem_owner_,
                                                        block_size_,
                                                        grid_size_);
  auto compact_storage = const_cast<ResultSetStorage*>(
      compact_result_set->allocateStorage(storage_->target_init_vals_));
  const auto row_size = get_row_bytes(compact_query_mem_desc);
  size_t compact_idx{0};
  for (const auto [source_storage, source_idx] : found_entries) {
    CHECK(source_storage);
    CHECK_EQ(row_size, get_row_bytes(source_storage->getQueryMemDesc()));
    auto* source_row_ptr = row_ptr_rowwise(source_storage->getUnderlyingBuffer(),
                                           source_storage->getQueryMemDesc(),
                                           source_idx);
    auto* compact_row_ptr =
        row_ptr_rowwise(compact_storage->buff_, compact_query_mem_desc, compact_idx++);
    memcpy(compact_row_ptr, source_row_ptr, row_size);
    clear_entry_key(*source_storage, source_idx);
  }

  invalidateCachedRowCount();
  clearBaselineHashDenseForReduction();
  clearDeviceColumnarBufferFragments();
  clearDeviceRowwiseBufferFragments();
  markDeviceColumnarCpuStorageValid();
  compact_result_set->setCachedRowCount(found_entries.size());
  compact_result_set->markBaselineHashDenseForReduction(found_entries.size());
  return compact_result_set;
}

const ResultSetStorage* ResultSet::getStorage() const {
  materializeDeviceColumnarCpuStorageIfNeeded();
  return storage_.get();
}

size_t ResultSet::colCount() const {
  return just_explain_ ? 1 : targets_.size();
}

SQLTypeInfo ResultSet::getColType(const size_t col_idx) const {
  if (just_explain_) {
    return SQLTypeInfo(kTEXT, false);
  }
  CHECK_LT(col_idx, targets_.size());
  return targets_[col_idx].agg_kind == kAVG ? SQLTypeInfo(kDOUBLE, false)
                                            : targets_[col_idx].sql_type;
}

StringDictionaryProxy* ResultSet::getStringDictionaryProxy(
    const shared::StringDictKey& dict_key) const {
  constexpr bool with_generation = true;
  return (dict_key.db_id > 0 || dict_key.dict_id == DictRef::literalsDictId)
             ? row_set_mem_owner_->getOrAddStringDictProxy(dict_key, with_generation)
             : row_set_mem_owner_->getStringDictProxy(dict_key);
}

class ResultSet::CellCallback {
  StringDictionaryProxy::IdMap const id_map_;
  int64_t const null_int_;

 public:
  CellCallback(StringDictionaryProxy::IdMap&& id_map, int64_t const null_int)
      : id_map_(std::move(id_map)), null_int_(null_int) {}
  void operator()(int8_t* const cell_ptr) const {
    using StringId = int32_t;
    StringId* const string_id_ptr = reinterpret_cast<StringId*>(cell_ptr);
    if (*string_id_ptr != null_int_) {
      *string_id_ptr = id_map_[*string_id_ptr];
    }
  }
};

void write_int_to_buff(int8_t* const ptr, const int8_t compact_sz, const int64_t value) {
  switch (compact_sz) {
    case 1:
      *reinterpret_cast<int8_t*>(ptr) = static_cast<int8_t>(value);
      return;
    case 2:
      *reinterpret_cast<int16_t*>(ptr) = static_cast<int16_t>(value);
      return;
    case 4:
      *reinterpret_cast<int32_t*>(ptr) = static_cast<int32_t>(value);
      return;
    case 8:
      *reinterpret_cast<int64_t*>(ptr) = value;
      return;
    default:
      UNREACHABLE() << "Unexpected integer width: " << static_cast<int>(compact_sz);
  }
}

// Update any dictionary-encoded targets within storage_ with the corresponding
// dictionary in the given targets parameter, if their comp_param (dictionary) differs.
// This may modify both the storage_ values and storage_ targets.
// Does not iterate through appended_storage_.
// Iterate over targets starting at index target_idx.
void ResultSet::translateDictEncodedColumns(std::vector<TargetInfo> const& targets,
                                            size_t const start_idx) {
  if (storage_) {
    CHECK_EQ(targets.size(), storage_->targets_.size());
    RowIterationState state;
    bool translated_dict_encoded_target = false;
    std::vector<size_t> materialized_lazy_targets;
    for (size_t target_idx = start_idx; target_idx < targets.size(); ++target_idx) {
      auto const& type_lhs = targets[target_idx].sql_type;
      if (type_lhs.is_dict_encoded_string()) {
        auto& type_rhs =
            const_cast<SQLTypeInfo&>(storage_->targets_[target_idx].sql_type);
        CHECK(type_rhs.is_dict_encoded_string());
        if (type_lhs.getStringDictKey() != type_rhs.getStringDictKey()) {
          materializeDeviceColumnarCpuStorageIfNeeded();
          auto* const sdp_lhs = getStringDictionaryProxy(type_lhs.getStringDictKey());
          CHECK(sdp_lhs);
          auto const* const sdp_rhs =
              getStringDictionaryProxy(type_rhs.getStringDictKey());
          CHECK(sdp_rhs);
          state.cur_target_idx_ = target_idx;
          CellCallback const translate_string_ids(sdp_lhs->transientUnion(*sdp_rhs),
                                                  inline_int_null_val(type_rhs));
          eachCellInColumn(state, translate_string_ids);
          type_rhs.set_comp_param(type_lhs.get_comp_param());
          type_rhs.setStringDictKey(type_lhs.getStringDictKey());
          for (auto& appended_storage : appended_storage_) {
            CHECK(appended_storage);
            CHECK_EQ(targets.size(), appended_storage->targets_.size());
            auto& appended_type_rhs =
                const_cast<SQLTypeInfo&>(appended_storage->targets_[target_idx].sql_type);
            CHECK(appended_type_rhs.is_dict_encoded_string());
            appended_type_rhs.set_comp_param(type_lhs.get_comp_param());
            appended_type_rhs.setStringDictKey(type_lhs.getStringDictKey());
          }
          translated_dict_encoded_target = true;
          if (target_idx < lazy_fetch_info_.size() &&
              lazy_fetch_info_[target_idx].is_lazily_fetched) {
            materialized_lazy_targets.push_back(target_idx);
          }
        }
      }
    }
    if (translated_dict_encoded_target) {
      device_columnar_fragments_.clear();
    }
    if (!materialized_lazy_targets.empty()) {
      std::vector<ColumnLazyFetchInfo> lazy_fetch_info;
      lazy_fetch_info.reserve(lazy_fetch_info_.size());
      for (size_t target_idx = 0; target_idx < lazy_fetch_info_.size(); ++target_idx) {
        const auto& info = lazy_fetch_info_[target_idx];
        const bool materialized =
            std::find(materialized_lazy_targets.begin(),
                      materialized_lazy_targets.end(),
                      target_idx) != materialized_lazy_targets.end();
        lazy_fetch_info.push_back(
            materialized ? ColumnLazyFetchInfo{false, -1, info.type, false} : info);
      }
      lazy_fetch_info_ = std::move(lazy_fetch_info);
    }
  }
}

// For each cell in column target_idx, callback func with pointer to datum.
// This currently assumes the column type is a dictionary-encoded string, but this logic
// can be generalized to other types.
void ResultSet::eachCellInColumn(RowIterationState& state, CellCallback const& func) {
  size_t const target_idx = state.cur_target_idx_;
  CHECK_LT(target_idx, lazy_fetch_info_.size());
  const auto& col_lazy_fetch = lazy_fetch_info_[target_idx];
  int const target_size = storage_->targets_[target_idx].sql_type.get_size();
  CHECK_LT(0, target_size) << storage_->targets_[target_idx].toString();

  auto translate_storage = [&](ResultSetStorage& storage, const size_t storage_idx) {
    size_t const nrows = storage.binSearchRowCount();
    for (size_t i = 0; i < nrows; ++i) {
      auto* const target_ptr =
          const_cast<int8_t*>(get_entry_target_ptr(storage, i, target_idx));
      if (col_lazy_fetch.is_lazily_fetched) {
        int64_t pos =
            read_int_from_buff(target_ptr, get_entry_target_width(storage, target_idx));
        if (pos == inline_int_null_val(storage.targets_[target_idx].sql_type)) {
          func(target_ptr);
          continue;
        }
        CHECK_GE(pos, 0);
        auto& frag_col_buffers =
            getColumnFrag(storage_idx, target_idx, col_lazy_fetch.local_col_id, pos);
        CHECK_LT(size_t(col_lazy_fetch.local_col_id), frag_col_buffers.size());
        int8_t const* const col_frag = frag_col_buffers[col_lazy_fetch.local_col_id];
        const auto string_id = read_int_from_buff(col_frag + pos * target_size,
                                                  static_cast<int8_t>(target_size));
        write_int_to_buff(target_ptr, static_cast<int8_t>(target_size), string_id);
      }
      func(target_ptr);
    }
  };

  translate_storage(*storage_, 0);
  for (size_t storage_idx = 0; storage_idx < appended_storage_.size(); ++storage_idx) {
    if (appended_storage_[storage_idx]) {
      translate_storage(*appended_storage_[storage_idx], storage_idx + 1);
    }
  }
}

namespace {

size_t get_truncated_row_count(size_t total_row_count, size_t limit, size_t offset) {
  if (total_row_count < offset) {
    return 0;
  }

  size_t total_truncated_row_count = total_row_count - offset;

  if (limit) {
    return std::min(total_truncated_row_count, limit);
  }

  return total_truncated_row_count;
}

}  // namespace

size_t ResultSet::rowCountImpl(const bool force_parallel) const {
  if (just_explain_) {
    return 1;
  }
  if (query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::TableFunction) {
    return entryCount();
  }
  if (!permutation_.empty()) {
    // keep_first_ corresponds to SQL LIMIT
    // drop_first_ corresponds to SQL OFFSET
    return get_truncated_row_count(permutation_.size(), keep_first_, drop_first_);
  }
  if (!storage_) {
    return 0;
  }
  CHECK(permutation_.empty());
  if (query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    materializeDeviceColumnarCpuStorageIfNeeded();
    return binSearchRowCount();
  }
  if (targets_.empty()) {
    return parallelRowCount();
  }

  if (force_parallel || entryCount() >= auto_parallel_row_count_threshold) {
    return parallelRowCount();
  }
  std::lock_guard<std::mutex> lock(row_iteration_mutex_);
  moveToBegin();
  size_t row_count{0};
  while (true) {
    auto crt_row = getNextRowUnlocked(false, false);
    if (crt_row.empty()) {
      break;
    }
    ++row_count;
  }
  moveToBegin();
  return row_count;
}

size_t ResultSet::rowCount(const bool force_parallel) const {
  // cached_row_count_ is atomic, so fetch it into a local variable first
  // to avoid repeat fetches
  const int64_t cached_row_count = cached_row_count_;
  if (cached_row_count != uninitialized_cached_row_count) {
    CHECK_GE(cached_row_count, 0);
    return cached_row_count;
  }
  const auto computed_row_count = rowCountImpl(force_parallel);
  setCachedRowCount(computed_row_count);
  return cached_row_count_;
}

void ResultSet::invalidateCachedRowCount() const {
  cached_row_count_ = uninitialized_cached_row_count;
}

void ResultSet::setCachedRowCount(const size_t row_count) const {
  if (row_count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("ResultSet row count exceeds the cached range");
  }
  const int64_t signed_row_count = static_cast<int64_t>(row_count);
  const int64_t old_cached_row_count = cached_row_count_.exchange(signed_row_count);
  CHECK(old_cached_row_count == uninitialized_cached_row_count ||
        old_cached_row_count == signed_row_count);
}

size_t ResultSet::binSearchRowCount() const {
  if (!storage_) {
    return 0;
  }

  size_t row_count = storage_->binSearchRowCount();
  for (auto& s : appended_storage_) {
    row_count += s->binSearchRowCount();
  }

  return get_truncated_row_count(row_count, getLimit(), drop_first_);
}

size_t ResultSet::parallelRowCount() const {
  using namespace threading;
  auto execute_parallel_row_count =
      [this, parent_thread_local_ids = logger::thread_local_ids()](
          const blocked_range<size_t>& r, size_t row_count) {
        logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
        for (size_t i = r.begin(); i < r.end(); ++i) {
          if (!isRowAtEmpty(i)) {
            ++row_count;
          }
        }
        return row_count;
      };
  const auto row_count = parallel_reduce(blocked_range<size_t>(0, entryCount()),
                                         size_t(0),
                                         execute_parallel_row_count,
                                         std::plus<size_t>());
  return get_truncated_row_count(row_count, getLimit(), drop_first_);
}

bool ResultSet::isEmpty() const {
  // To simplify this function and de-dup logic with ResultSet::rowCount()
  // (mismatches between the two were causing bugs), we modified this function
  // to simply fetch rowCount(). The potential downside of this approach is that
  // in some cases more work will need to be done, as we can't just stop at the first row.
  // Mitigating that for most cases is the following:
  // 1) rowCount() is cached, so the logic for actually computing row counts will run only
  // once
  //    per result set.
  // 2) If the cache is empty (cached_row_count_ == -1), rowCount() will use parallel
  //    methods if deemed appropriate, which in many cases could be faster for a sparse
  //    large result set that single-threaded iteration from the beginning
  // 3) Often where isEmpty() is needed, rowCount() is also needed. Since the first call
  // to rowCount()
  //    will be cached, there is no extra overhead in these cases

  return rowCount() == size_t(0);
}

bool ResultSet::definitelyHasNoRows() const {
  return (!storage_ && !estimator_ && !just_explain_) || cached_row_count_ == 0;
}

const QueryMemoryDescriptor& ResultSet::getQueryMemDesc() const {
  CHECK(storage_);
  return storage_->query_mem_desc_;
}

const std::vector<TargetInfo>& ResultSet::getTargetInfos() const {
  return targets_;
}

const std::vector<int64_t>& ResultSet::getTargetInitVals() const {
  CHECK(storage_);
  return storage_->target_init_vals_;
}

int8_t* ResultSet::getDeviceEstimatorBuffer() const {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  CHECK(device_estimator_buffer_);
  return device_estimator_buffer_->getMemoryPtr();
}

int8_t* ResultSet::getHostEstimatorBuffer() const {
  return host_estimator_buffer_;
}

void ResultSet::syncEstimatorBuffer() const {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  CHECK(!host_estimator_buffer_);
  CHECK_EQ(size_t(0), estimator_->getBufferSize() % sizeof(int64_t));
  host_estimator_buffer_ =
      static_cast<int8_t*>(checked_calloc(estimator_->getBufferSize(), 1));
  CHECK(device_estimator_buffer_);
  auto device_buffer_ptr = device_estimator_buffer_->getMemoryPtr();
  cuda_allocator_->copyFromDevice(host_estimator_buffer_,
                                  device_buffer_ptr,
                                  estimator_->getBufferSize(),
                                  "Estimator buffer");
}

void ResultSet::setQueueTime(const int64_t queue_time) {
  timings_.executor_queue_time = queue_time;
}

void ResultSet::setKernelQueueTime(const int64_t kernel_queue_time) {
  timings_.kernel_queue_time = kernel_queue_time;
}

void ResultSet::addCompilationQueueTime(const int64_t compilation_queue_time) {
  timings_.compilation_queue_time += compilation_queue_time;
}

int64_t ResultSet::getQueueTime() const {
  return timings_.executor_queue_time + timings_.kernel_queue_time +
         timings_.compilation_queue_time;
}

int64_t ResultSet::getRenderTime() const {
  return timings_.render_time;
}

void ResultSet::moveToBegin() const {
  crt_row_buff_idx_ = 0;
  fetched_so_far_ = 0;
}

bool ResultSet::isTruncated() const {
  return keep_first_ + drop_first_;
}

bool ResultSet::isExplain() const {
  return just_explain_;
}

void ResultSet::setValidationOnlyRes() {
  for_validation_only_ = true;
}

bool ResultSet::isValidationOnlyRes() const {
  return for_validation_only_;
}

int ResultSet::getDeviceId() const {
  return device_id_;
}

int ResultSet::getThreadIdx() const {
  return thread_idx_;
}

QueryMemoryDescriptor ResultSet::fixupQueryMemoryDescriptor(
    const QueryMemoryDescriptor& query_mem_desc) {
  auto query_mem_desc_copy = query_mem_desc;
  query_mem_desc_copy.resetGroupColWidths(
      std::vector<int8_t>(query_mem_desc_copy.getGroupbyColCount(), 8));
  if (query_mem_desc.didOutputColumnar()) {
    return query_mem_desc_copy;
  }
  query_mem_desc_copy.alignPaddedSlots();
  return query_mem_desc_copy;
}

void ResultSet::sort(const std::list<Analyzer::OrderEntry>& order_entries,
                     size_t top_n,
                     ExecutorDeviceType device_type,
                     Executor* executor,
                     bool need_to_initialize_device_ids_to_use) {
  auto timer = DEBUG_TIMER(__func__);

  if (!storage_) {
    return;
  }
  CHECK(!targets_.empty());
  if (need_to_initialize_device_ids_to_use) {
    // this function can be called in test suites directly, not as a part of
    // SELECT query processing
    // in such case, we mock device id selection logic to continue the process
    executor->mockDeviceIdSelectionLogicToOnlyUseSingleDevice();
  }
  materializeDeviceColumnarCpuStorageIfNeeded();
  invalidateCachedRowCount();
#ifdef HAVE_CUDA
  if (canUseFastBaselineSort(order_entries, top_n)) {
    baselineSort(order_entries, top_n, device_type, executor);
    return;
  }
#endif  // HAVE_CUDA
  if (query_mem_desc_.sortOnGpu() && appended_storage_.empty()) {
    try {
      radixSortOnGpu(order_entries);
    } catch (const OutOfMemory&) {
      LOG(WARNING) << "Out of GPU memory during sort, finish on CPU";
      radixSortOnCpu(order_entries);
    } catch (const std::bad_alloc&) {
      LOG(WARNING) << "Out of GPU memory during sort, finish on CPU";
      radixSortOnCpu(order_entries);
    }
    return;
  }
  if (hasDeferredLazyFetchChunks()) {
    std::vector<size_t> lazy_sort_target_indices;
    lazy_sort_target_indices.reserve(order_entries.size());
    for (const auto& order_entry : order_entries) {
      CHECK_GE(order_entry.tle_no, 1);
      const auto target_idx = static_cast<size_t>(order_entry.tle_no - 1);
      CHECK_LT(target_idx, lazy_fetch_info_.size());
      if (lazy_fetch_info_[target_idx].is_lazily_fetched) {
        lazy_sort_target_indices.push_back(target_idx);
      }
    }
    std::sort(lazy_sort_target_indices.begin(), lazy_sort_target_indices.end());
    lazy_sort_target_indices.erase(
        std::unique(lazy_sort_target_indices.begin(), lazy_sort_target_indices.end()),
        lazy_sort_target_indices.end());
    if (!lazy_sort_target_indices.empty()) {
      materializeDeferredLazyFetchColumnsForAllRows(lazy_sort_target_indices);
    }
  }
  // This check isn't strictly required, but allows the index buffer to be 32-bit.
  const auto sort_entry_count = entryCount();
  if (sort_entry_count > std::numeric_limits<uint32_t>::max()) {
    throw RowSortException("Sorting more than 4B elements not supported");
  }

  CHECK(permutation_.empty());

  const bool is_top_n_sort = top_n != 0;
  if (top_n && sort_entry_count > g_parallel_top_min) {
    if (g_enable_watchdog && sortRowCountForWatchdog() > g_parallel_top_max) {
      throw WatchdogException("Sorting the result would be too slow");
    }
    parallelTop(order_entries, top_n, sort_entry_count, executor);
  } else {
    if (g_enable_watchdog && sortRowCountForWatchdog() > g_watchdog_baseline_sort_max) {
      throw WatchdogException("Sorting the result would be too slow");
    }
    PermutationView pv;
    if (!top_n && sort_entry_count > g_parallel_top_min) {
      pv = parallelInitPermutationBuffer(sort_entry_count);
    } else {
      permutation_.resize(sort_entry_count);
      // PermutationView is used to share common API with parallelTop().
      pv = PermutationView(permutation_.data(), 0, permutation_.size());
      pv = initPermutationBuffer(pv, 0, permutation_.size());
    }
    if (top_n == 0) {
      if (sortWithMaterializedNumericKey(order_entries, pv)) {
        if (pv.size() < permutation_.size()) {
          permutation_.resize(pv.size());
          permutation_.shrink_to_fit();
        }
        return;
      }
      top_n = pv.size();  // top_n == 0 implies a full sort
    }
    std::optional<MaterializedSortBuffersBase::TopNDictionarySortContext>
        top_n_dictionary_sort_context;
    if (g_enable_result_reduction_pipeline) {
      if (is_top_n_sort) {
        top_n_dictionary_sort_context =
            MaterializedSortBuffersBase::TopNDictionarySortContext{pv.size(), top_n};
      }
    }
    initMaterializedSortBuffers(
        order_entries, false, pv.size(), top_n_dictionary_sort_context);
    pv = topPermutation(
        pv, top_n, createComparator(order_entries, pv, executor, false).get());
    if (pv.size() < permutation_.size()) {
      permutation_.resize(pv.size());
      permutation_.shrink_to_fit();
    }
  }
}

size_t ResultSet::sortRowCountForWatchdog() const {
  return parallelRowCount();
}

#ifdef HAVE_CUDA
void ResultSet::baselineSort(const std::list<Analyzer::OrderEntry>& order_entries,
                             const size_t top_n,
                             const ExecutorDeviceType device_type,
                             const Executor* executor) {
  auto timer = DEBUG_TIMER(__func__);
  // If we only have on GPU, it's usually faster to do multi-threaded radix sort on CPU
  if (device_type == ExecutorDeviceType::GPU && getGpuCount() > 1) {
    try {
      doBaselineSort(ExecutorDeviceType::GPU, order_entries, top_n, executor);
    } catch (...) {
      doBaselineSort(ExecutorDeviceType::CPU, order_entries, top_n, executor);
    }
  } else {
    doBaselineSort(ExecutorDeviceType::CPU, order_entries, top_n, executor);
  }
}
#endif  // HAVE_CUDA

// Append non-empty indexes i in [begin,end) from findStorage(i) to permutation.
PermutationView ResultSet::initPermutationBuffer(PermutationView permutation,
                                                 PermutationIdx const begin,
                                                 PermutationIdx const end) const {
  auto timer = DEBUG_TIMER(__func__);
  for (PermutationIdx i = begin; i < end; ++i) {
    const auto storage_lookup_result = findStorage(i);
    const auto lhs_storage = storage_lookup_result.storage_ptr;
    const auto off = storage_lookup_result.fixedup_entry_idx;
    CHECK(lhs_storage);
    if (!lhs_storage->isEmptyEntry(off)) {
      permutation.push_back(i);
    }
  }
  return permutation;
}

const Permutation& ResultSet::getPermutationBuffer() const {
  return permutation_;
}

PermutationView ResultSet::parallelInitPermutationBuffer(const size_t entry_count) {
  auto timer = DEBUG_TIMER(__func__);
  const size_t nthreads = cpu_threads();
  permutation_.resize(entry_count);
  std::vector<PermutationView> permutation_views(nthreads);

  {
    threading::task_group init_threads;
    for (auto interval :
         makeIntervals<PermutationIdx>(0, permutation_.size(), nthreads)) {
      init_threads.run([this,
                        &permutation_views,
                        interval,
                        parent_thread_local_ids = logger::thread_local_ids()] {
        logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
        PermutationView pv(permutation_.data() + interval.begin, 0, interval.size());
        permutation_views[interval.index] =
            initPermutationBuffer(pv, interval.begin, interval.end);
      });
    }
    init_threads.wait();
  }

  auto end = permutation_.begin() + permutation_views.front().size();
  for (size_t i = 1; i < nthreads; ++i) {
    std::copy(permutation_views[i].begin(), permutation_views[i].end(), end);
    end += permutation_views[i].size();
  }
  return PermutationView(permutation_.data(), end - permutation_.begin());
}

void ResultSet::parallelTop(const std::list<Analyzer::OrderEntry>& order_entries,
                            const size_t top_n,
                            const size_t entry_count,
                            const Executor* executor) {
  // Each worker contributes up to top_n rows to the final merge. Bound both the
  // scan grain and that partial-winner fan-in instead of oversubscribing small
  // ranges on high-core-count hosts.
  const auto min_entries_per_worker = std::max(g_parallel_top_min, size_t{1});
  const auto workers_for_entry_count =
      entry_count / min_entries_per_worker +
      static_cast<size_t>(entry_count % min_entries_per_worker != 0);
  const auto nthreads = std::max(size_t{1},
                                 std::min({static_cast<size_t>(cpu_threads()),
                                           parallel_top_max_worker_count,
                                           workers_for_entry_count}));

  // Split permutation_ into nthreads subranges and initialize them
  permutation_.resize(entry_count);
  std::vector<PermutationView> permutation_views(nthreads);

  // First, initialize all permutation views. We need these initialized so that
  // we can init the materialized sort buffers, which requires that the permutations
  // to be initialized

  {
    threading::task_group init_threads;
    for (auto interval :
         makeIntervals<PermutationIdx>(0, permutation_.size(), nthreads)) {
      init_threads.run([this,
                        &permutation_views,
                        interval,
                        parent_thread_local_ids = logger::thread_local_ids()] {
        logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
        PermutationView pv(permutation_.data() + interval.begin, 0, interval.size());
        permutation_views[interval.index] =
            initPermutationBuffer(pv, interval.begin, interval.end);
      });
    }
    init_threads.wait();
  }

  std::optional<MaterializedSortBuffersBase::TopNDictionarySortContext>
      top_n_dictionary_sort_context;
  if (g_enable_result_reduction_pipeline) {
    const auto candidate_count = std::accumulate(
        permutation_views.begin(),
        permutation_views.end(),
        size_t{0},
        [](const size_t count, const auto& view) { return count + view.size(); });
    top_n_dictionary_sort_context =
        MaterializedSortBuffersBase::TopNDictionarySortContext{candidate_count, top_n};
  }

  // Now that all permutation views are initialized, create shared buffers
  // These buffers will be used by all comparators generated below
  initMaterializedSortBuffers(
      order_entries, false, std::nullopt, top_n_dictionary_sort_context);

  // Perform the top-k sort on each permutation view
  {
    threading::task_group top_sort_threads;
    for (size_t i = 0; i < nthreads; ++i) {
      top_sort_threads.run([this,
                            &order_entries,
                            &permutation_views,
                            top_n,
                            executor,
                            i,
                            parent_thread_local_ids = logger::thread_local_ids()] {
        logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
        const auto comparator =
            createComparator(order_entries, permutation_views[i], executor, true);
        permutation_views[i] =
            topPermutation(permutation_views[i], top_n, comparator.get());
      });
    }
    top_sort_threads.wait();
  }

  // In case you are considering implementing a parallel reduction, note that the
  // ResultSetComparator constructor is O(N) in order to materialize some of the aggregate
  // columns as necessary to perform a comparison. This cost is why reduction is chosen to
  // be serial instead; only one more Comparator is needed below.

  // Left-copy disjoint top-sorted subranges into one contiguous range.
  // ++++....+++.....+++++...  ->  ++++++++++++............
  auto end = permutation_.begin() + permutation_views.front().size();
  for (size_t i = 1; i < nthreads; ++i) {
    std::copy(permutation_views[i].begin(), permutation_views[i].end(), end);
    end += permutation_views[i].size();
  }

  // Top sort final range.
  PermutationView pv(permutation_.data(), end - permutation_.begin());
  const auto comparator = createComparator(order_entries, pv, executor, false);
  pv = topPermutation(pv, top_n, comparator.get());
  permutation_.resize(pv.size());
  permutation_.shrink_to_fit();
}

namespace {

struct NumericSortKey {
  int64_t int_value{0};
  double fp_value{0.0};
  bool is_null{false};
  bool is_fp{false};
};

bool numeric_sort_key_less(const NumericSortKey& lhs,
                           const NumericSortKey& rhs,
                           const Analyzer::OrderEntry& order_entry) {
  if (lhs.is_null || rhs.is_null) {
    if (lhs.is_null == rhs.is_null) {
      return false;
    }
    return lhs.is_null ? order_entry.nulls_first : !order_entry.nulls_first;
  }
  if (lhs.is_fp) {
    CHECK(rhs.is_fp);
    if (lhs.fp_value == rhs.fp_value) {
      return false;
    }
    return (lhs.fp_value < rhs.fp_value) != order_entry.is_desc;
  }
  CHECK(!rhs.is_fp);
  if (lhs.int_value == rhs.int_value) {
    return false;
  }
  return (lhs.int_value < rhs.int_value) != order_entry.is_desc;
}

bool should_read_float_argument_from_32bit_slot(
    const QueryMemoryDescriptor& query_mem_desc,
    const TargetInfo& target_info,
    const size_t slot_idx,
    const bool is_col_lazy) {
  const auto padded_width =
      static_cast<size_t>(query_mem_desc.getPaddedSlotWidthBytes(slot_idx));
  const auto logical_width =
      static_cast<size_t>(query_mem_desc.getLogicalSlotWidthBytes(slot_idx));
  const bool stored_as_float = padded_width == sizeof(float) ||
                               (target_info.is_agg && logical_width == sizeof(float));
  if (!stored_as_float) {
    return false;
  }
  return query_mem_desc.didOutputColumnar() ? !is_col_lazy : true;
}

}  // namespace

bool ResultSet::sortWithMaterializedNumericKey(
    const std::list<Analyzer::OrderEntry>& order_entries,
    PermutationView permutation) const {
  if (order_entries.size() != 1 || permutation.empty()) {
    return false;
  }

  const auto& order_entry = order_entries.front();
  CHECK_GE(order_entry.tle_no, 1);
  const auto target_idx = static_cast<size_t>(order_entry.tle_no - 1);
  CHECK_LT(target_idx, targets_.size());
  const auto& target_info = targets_[target_idx];
  if (!target_info.sql_type.is_number() || is_distinct_target(target_info) ||
      target_info.agg_kind == kAPPROX_QUANTILE || target_info.agg_kind == kMODE) {
    return false;
  }

  const auto entry_ti = get_compact_type(target_info);
  if (!entry_ti.is_number()) {
    return false;
  }

  bool float_argument_input = takes_float_argument(target_info);
  if (entry_ti.get_type() == kFLOAT) {
    const auto is_col_lazy =
        !lazy_fetch_info_.empty() && lazy_fetch_info_[target_idx].is_lazily_fetched;
    const auto slot_indices = getSlotIndicesForTargetIndices();
    CHECK_LT(target_idx, slot_indices.size());
    if (should_read_float_argument_from_32bit_slot(
            query_mem_desc_, target_info, slot_indices[target_idx], is_col_lazy)) {
      float_argument_input = true;
    }
  }

  std::vector<NumericSortKey> sort_keys(permutation.size());
  const auto extract_key = [&](const auto& buffer_itr, const size_t permutation_pos) {
    const auto entry_idx = permutation[permutation_pos];
    const auto storage_lookup_result = findStorage(entry_idx);
    const auto storage = storage_lookup_result.storage_ptr;
    CHECK(storage);
    const auto value =
        buffer_itr.getColumnInternal(storage->buff_,
                                     storage_lookup_result.fixedup_entry_idx,
                                     target_idx,
                                     storage_lookup_result);

    NumericSortKey key;
    key.is_null = isNull(entry_ti, value, float_argument_input);
    if (!key.is_null) {
      if (value.isPair()) {
        key.is_fp = true;
        key.fp_value =
            pair_to_double({value.i1, value.i2}, entry_ti, float_argument_input);
      } else {
        CHECK(value.isInt());
        if (entry_ti.is_fp()) {
          key.is_fp = true;
          if (float_argument_input) {
            key.fp_value = *reinterpret_cast<const float*>(may_alias_ptr(&value.i1));
          } else {
            std::memcpy(&key.fp_value, &value.i1, sizeof(key.fp_value));
          }
        } else {
          key.int_value = value.i1;
        }
      }
    }
    sort_keys[permutation_pos] = key;
  };

  const auto fill_keys = [&](const auto& buffer_itr) {
    if (permutation.size() >= auto_parallel_row_count_threshold) {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, permutation.size()),
                        [&](const tbb::blocked_range<size_t>& range) {
                          for (size_t i = range.begin(); i != range.end(); ++i) {
                            extract_key(buffer_itr, i);
                          }
                        });
    } else {
      for (size_t i = 0; i < permutation.size(); ++i) {
        extract_key(buffer_itr, i);
      }
    }
  };

  if (query_mem_desc_.didOutputColumnar()) {
    fill_keys(ColumnWiseTargetAccessor(this));
  } else {
    fill_keys(RowWiseTargetAccessor(this));
  }

  std::vector<PermutationIdx> sorted_positions(permutation.size());
  std::iota(sorted_positions.begin(), sorted_positions.end(), PermutationIdx{0});
  const auto position_less = [&](const PermutationIdx lhs_pos,
                                 const PermutationIdx rhs_pos) {
    return numeric_sort_key_less(sort_keys[lhs_pos], sort_keys[rhs_pos], order_entry);
  };
  if (sorted_positions.size() >= g_parallel_top_min) {
    tbb::parallel_sort(sorted_positions.begin(), sorted_positions.end(), position_less);
  } else {
    std::sort(sorted_positions.begin(), sorted_positions.end(), position_less);
  }

  Permutation sorted_permutation(permutation.size());
  for (size_t i = 0; i < sorted_positions.size(); ++i) {
    sorted_permutation[i] = permutation[sorted_positions[i]];
  }
  std::copy(sorted_permutation.begin(), sorted_permutation.end(), permutation.begin());
  return true;
}

std::pair<size_t, size_t> ResultSet::getStorageIndex(const size_t entry_idx) const {
  size_t fixedup_entry_idx = entry_idx;
  auto entry_count = storage_->query_mem_desc_.getEntryCount();
  const bool is_rowwise_layout = !storage_->query_mem_desc_.didOutputColumnar();
  if (fixedup_entry_idx < entry_count) {
    return {0, fixedup_entry_idx};
  }
  fixedup_entry_idx -= entry_count;
  for (size_t i = 0; i < appended_storage_.size(); ++i) {
    const auto& desc = appended_storage_[i]->query_mem_desc_;
    CHECK_NE(is_rowwise_layout, desc.didOutputColumnar());
    entry_count = desc.getEntryCount();
    if (fixedup_entry_idx < entry_count) {
      return {i + 1, fixedup_entry_idx};
    }
    fixedup_entry_idx -= entry_count;
  }
  UNREACHABLE() << "entry_idx = " << entry_idx << ", query_mem_desc_.getEntryCount() = "
                << query_mem_desc_.getEntryCount();
  return {};
}

template struct ResultSet::MaterializedSortBuffers<ResultSet::RowWiseTargetAccessor>;
template struct ResultSet::MaterializedSortBuffers<ResultSet::ColumnWiseTargetAccessor>;
template struct ResultSet::ResultSetComparator<ResultSet::RowWiseTargetAccessor>;
template struct ResultSet::ResultSetComparator<ResultSet::ColumnWiseTargetAccessor>;

ResultSet::StorageLookupResult ResultSet::findStorage(const size_t entry_idx) const {
  auto [stg_idx, fixedup_entry_idx] = getStorageIndex(entry_idx);
  return {stg_idx ? appended_storage_[stg_idx - 1].get() : storage_.get(),
          fixedup_entry_idx,
          stg_idx};
}

namespace {
struct IsAggKind {
  std::vector<TargetInfo> const& targets_;
  SQLAgg const agg_kind_;
  IsAggKind(std::vector<TargetInfo> const& targets, SQLAgg const agg_kind)
      : targets_(targets), agg_kind_(agg_kind) {}
  bool operator()(Analyzer::OrderEntry const& order_entry) const {
    return targets_[order_entry.tle_no - 1].agg_kind == agg_kind_;
  }
};

bool is_notnull_dictionary_string_translated_null(
    const SQLTypeInfo& type_info,
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t target_idx,
    const int32_t string_id) {
  if (!type_info.is_dict_encoded_string() || !type_info.get_notnull()) {
    return false;
  }
  if (string_id == inline_int_null_value<int32_t>()) {
    return true;
  }
  const auto translated_null_key =
      query_mem_desc.getTranslatedGroupbyNullForTarget(target_idx);
  return translated_null_key && string_id == *translated_null_key;
}

size_t ceil_log2(const size_t value) {
  if (value <= 1) {
    return 0;
  }
  size_t exponent{0};
  for (auto remaining = value - 1; remaining; remaining >>= 1) {
    ++exponent;
  }
  return exponent;
}

size_t estimated_comparison_count(const size_t row_count, const size_t sorted_count) {
  const auto comparisons_per_row = std::max(size_t{1}, ceil_log2(sorted_count));
  if (row_count > std::numeric_limits<size_t>::max() / comparisons_per_row) {
    return std::numeric_limits<size_t>::max();
  }
  return row_count * comparisons_per_row;
}

size_t saturating_add(const size_t lhs, const size_t rhs) {
  return lhs > std::numeric_limits<size_t>::max() - rhs
             ? std::numeric_limits<size_t>::max()
             : lhs + rhs;
}

size_t saturating_multiply(const size_t lhs, const size_t rhs) {
  return lhs && rhs > std::numeric_limits<size_t>::max() / lhs
             ? std::numeric_limits<size_t>::max()
             : lhs * rhs;
}

bool should_extend_local_dictionary_sort(const size_t candidate_count,
                                         const size_t dictionary_entry_count,
                                         const bool global_rank_cache_complete) {
  if (!candidate_count || !dictionary_entry_count || global_rank_cache_complete) {
    return false;
  }

  const auto candidate_sort_work =
      estimated_comparison_count(candidate_count, candidate_count);
  const auto local_work = saturating_add(
      candidate_count, saturating_multiply(candidate_sort_work, size_t{2}));
  const auto global_work =
      estimated_comparison_count(dictionary_entry_count, dictionary_entry_count);
  if (local_work >= global_work) {
    return false;
  }

  using LocalStringVector = std::vector<std::pair<std::string, int32_t>>;
  using RankMap = robin_hood::unordered_flat_map<int32_t, int32_t>;
  constexpr size_t hash_container_overhead = 2 * sizeof(void*);
  constexpr size_t estimated_string_payload_bytes = sizeof(std::string);
  constexpr size_t local_bytes_per_candidate =
      sizeof(LocalStringVector::value_type) + sizeof(RankMap::value_type) +
      hash_container_overhead + estimated_string_payload_bytes;
  const auto local_bytes =
      saturating_multiply(candidate_count, local_bytes_per_candidate);
  const auto global_rank_bytes =
      saturating_multiply(dictionary_entry_count, sizeof(int32_t));
  return local_bytes < global_rank_bytes;
}
}  // namespace

bool ResultSet::MaterializedSortBuffersBase::DictionaryStringSortPermutation::operator()(
    const int32_t lhs,
    const int32_t rhs,
    const bool sort_descending,
    const bool nulls_first) const {
  if (global_permutation_) {
    return (*global_permutation_)(lhs, rhs);
  }
  if (!string_dictionary_proxy_) {
    const auto lhs_rank_it = local_string_id_to_rank_.find(lhs);
    const auto rhs_rank_it = local_string_id_to_rank_.find(rhs);
    if (lhs_rank_it == local_string_id_to_rank_.end() ||
        rhs_rank_it == local_string_id_to_rank_.end()) {
      if (lhs_rank_it == local_string_id_to_rank_.end() &&
          rhs_rank_it == local_string_id_to_rank_.end()) {
        return false;
      }
      return lhs_rank_it == local_string_id_to_rank_.end() ? nulls_first : !nulls_first;
    }
    const auto lhs_rank = lhs_rank_it->second;
    const auto rhs_rank = rhs_rank_it->second;
    if (lhs_rank == rhs_rank) {
      return false;
    }
    return (lhs_rank < rhs_rank) != sort_descending;
  }

  const auto decode_sort_string =
      [this](const int32_t string_id) -> std::optional<std::string_view> {
    if (notnull_ && (string_id == inline_int_null_value<int32_t>() ||
                     (translated_null_ && string_id == *translated_null_))) {
      return std::string_view{""};
    }
    if (!string_dictionary_proxy_->canDecodeStringId(string_id)) {
      return std::nullopt;
    }
    const auto [string_bytes, string_size] =
        string_dictionary_proxy_->getStringBytes(string_id);
    return std::string_view{string_bytes, string_size};
  };

  const auto lhs_string = decode_sort_string(lhs);
  const auto rhs_string = decode_sort_string(rhs);
  if (!lhs_string || !rhs_string) {
    if (!lhs_string && !rhs_string) {
      return false;
    }
    return !lhs_string ? nulls_first : !nulls_first;
  }
  if (*lhs_string == *rhs_string) {
    return false;
  }
  return string_lt(lhs_string->data(),
                   static_cast<int32_t>(lhs_string->size()),
                   rhs_string->data(),
                   static_cast<int32_t>(rhs_string->size())) != sort_descending;
}

template <typename BUFFER_ITERATOR_TYPE>
ResultSet::MaterializedSortBuffersBase::DictionaryStringSortPermutation
ResultSet::MaterializedSortBuffers<BUFFER_ITERATOR_TYPE>::
    materializeDictionaryEncodedSortPermutation(
        const Analyzer::OrderEntry& order_entry) const {
  const auto entry_ti = get_compact_type(result_set_->targets_[order_entry.tle_no - 1]);
  const auto string_dict_proxy =
      result_set_->getStringDictionaryProxy(entry_ti.getStringDictKey());
  const auto target_idx = static_cast<size_t>(order_entry.tle_no - 1);

  if (top_n_dictionary_sort_context_) {
    const auto candidate_count = top_n_dictionary_sort_context_->candidate_count;
    const auto effective_top_n =
        std::min(candidate_count, top_n_dictionary_sort_context_->top_n);
    const auto dictionary_entry_count = string_dict_proxy->entryCount();
    const auto direct_comparison_count =
        estimated_comparison_count(candidate_count, effective_top_n);
    const auto global_comparison_count =
        estimated_comparison_count(dictionary_entry_count, dictionary_entry_count);
    const auto local_comparison_count =
        compact_permutation_size_ &&
                *compact_permutation_size_ <= local_dictionary_sort_max_result_rows
            ? candidate_count +
                  estimated_comparison_count(candidate_count, candidate_count)
            : std::numeric_limits<size_t>::max();
    // Bounded Top-N performs O(N log K) comparisons. Compare strings lazily when
    // that is cheaper than assigning ranks to either all D dictionary entries or
    // all N compact candidates.
    if (candidate_count && effective_top_n &&
        direct_comparison_count <
            std::min(global_comparison_count, local_comparison_count)) {
      return MaterializedSortBuffersBase::DictionaryStringSortPermutation(
          string_dict_proxy,
          entry_ti.get_notnull(),
          result_set_->query_mem_desc_.getTranslatedGroupbyNullForTarget(target_idx));
    }
  }

  const bool use_extended_local_dictionary_sort =
      g_enable_stringdict_parallel_sort && !single_threaded_ &&
      compact_permutation_size_ &&
      *compact_permutation_size_ > local_dictionary_sort_max_result_rows &&
      !top_n_dictionary_sort_context_ &&
      should_extend_local_dictionary_sort(
          *compact_permutation_size_,
          string_dict_proxy->entryCount(),
          string_dict_proxy->isSortedPermutationCacheComplete());
  const bool use_local_dictionary_sort =
      compact_permutation_size_ &&
      (*compact_permutation_size_ <= local_dictionary_sort_max_result_rows ||
       use_extended_local_dictionary_sort);
  if (use_local_dictionary_sort) {
    const bool parallelize_local_sort = use_extended_local_dictionary_sort;
    const auto decode_sort_string =
        [this, &entry_ti, string_dict_proxy, target_idx](
            const int32_t string_id) -> std::optional<std::string> {
      if (is_notnull_dictionary_string_translated_null(
              entry_ti, result_set_->query_mem_desc_, target_idx, string_id)) {
        return std::string{};
      }
      if (string_id == inline_int_null_value<int32_t>() ||
          string_id == StringDictionary::INVALID_STR_ID) {
        return std::nullopt;
      }
      if (string_dict_proxy->canDecodeStringId(string_id)) {
        return string_dict_proxy->getString(string_id);
      }
      return std::nullopt;
    };
    std::vector<int32_t> local_string_ids(*compact_permutation_size_);
    const auto collect_string_ids = [&](const size_t begin, const size_t end) {
      for (size_t i = begin; i < end; ++i) {
        const auto permuted_idx = result_set_->permutation_[i];
        const auto storage_lookup_result = result_set_->findStorage(permuted_idx);
        const auto storage = storage_lookup_result.storage_ptr;
        const auto off = storage_lookup_result.fixedup_entry_idx;
        const auto value = buffer_itr_.getColumnInternal(
            storage->buff_, off, order_entry.tle_no - 1, storage_lookup_result);
        CHECK(value.isInt());
        local_string_ids[i] = static_cast<int32_t>(value.i1);
      }
    };
    if (parallelize_local_sort &&
        local_string_ids.size() >= auto_parallel_row_count_threshold) {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, local_string_ids.size()),
                        [&](const tbb::blocked_range<size_t>& range) {
                          collect_string_ids(range.begin(), range.end());
                        });
      tbb::parallel_sort(local_string_ids.begin(), local_string_ids.end());
    } else {
      collect_string_ids(0, local_string_ids.size());
      std::sort(local_string_ids.begin(), local_string_ids.end());
    }
    local_string_ids.erase(std::unique(local_string_ids.begin(), local_string_ids.end()),
                           local_string_ids.end());

    std::vector<std::pair<std::string, int32_t>> local_strings(local_string_ids.size());
    const auto decode_local_strings = [&](const size_t begin, const size_t end) {
      for (size_t i = begin; i < end; ++i) {
        const auto string_id = local_string_ids[i];
        if (auto string_value = decode_sort_string(string_id)) {
          local_strings[i] = std::make_pair(std::move(*string_value), string_id);
        } else {
          local_strings[i].second = StringDictionary::INVALID_STR_ID;
        }
      }
    };
    if (parallelize_local_sort &&
        local_strings.size() >= auto_parallel_row_count_threshold) {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, local_strings.size()),
                        [&](const tbb::blocked_range<size_t>& range) {
                          decode_local_strings(range.begin(), range.end());
                        });
    } else {
      decode_local_strings(0, local_strings.size());
    }
    local_strings.erase(std::remove_if(local_strings.begin(),
                                       local_strings.end(),
                                       [](const auto& value) {
                                         return value.second ==
                                                StringDictionary::INVALID_STR_ID;
                                       }),
                        local_strings.end());
    std::vector<int32_t>().swap(local_string_ids);

    const auto compare_local_strings = [](const auto& lhs, const auto& rhs) {
      return string_lt(lhs.first.data(),
                       static_cast<int32_t>(lhs.first.size()),
                       rhs.first.data(),
                       static_cast<int32_t>(rhs.first.size()));
    };
    if (parallelize_local_sort &&
        local_strings.size() >= auto_parallel_row_count_threshold) {
      tbb::parallel_sort(
          local_strings.begin(), local_strings.end(), compare_local_strings);
    } else {
      std::sort(local_strings.begin(), local_strings.end(), compare_local_strings);
    }

    MaterializedSortBuffersBase::DictionaryStringSortPermutation::LocalStringRankMap
        local_string_id_to_rank;
    local_string_id_to_rank.reserve(local_strings.size());
    for (size_t rank = 0; rank < local_strings.size(); ++rank) {
      CHECK_LE(rank, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
      local_string_id_to_rank.emplace(local_strings[rank].second,
                                      static_cast<int32_t>(rank));
    }
    return MaterializedSortBuffersBase::DictionaryStringSortPermutation(
        std::move(local_string_id_to_rank));
  }

  return MaterializedSortBuffersBase::DictionaryStringSortPermutation(
      string_dict_proxy->getSortedPermutation(order_entry.is_desc));
}

template <typename BUFFER_ITERATOR_TYPE>
std::vector<ResultSet::MaterializedSortBuffersBase::DictionaryStringSortPermutation>
ResultSet::MaterializedSortBuffers<
    BUFFER_ITERATOR_TYPE>::materializeDictionaryEncodedSortPermutations() const {
  // First, count the number of dictionary-encoded string columns
  size_t dictionary_encoded_str_col_count = 0;
  for (const auto& order_entry : order_entries_) {
    const auto entry_ti = get_compact_type(result_set_->targets_[order_entry.tle_no - 1]);
    if (entry_ti.is_string() && entry_ti.get_compression() == kENCODING_DICT) {
      dictionary_encoded_str_col_count++;
    }
  }

  // Reserve space for the exact number of dictionary-encoded string columns
  std::vector<MaterializedSortBuffersBase::DictionaryStringSortPermutation> permutations;
  permutations.reserve(dictionary_encoded_str_col_count);

  // Populate the vector only for dictionary-encoded string columns
  for (const auto& order_entry : order_entries_) {
    const auto entry_ti = get_compact_type(result_set_->targets_[order_entry.tle_no - 1]);
    if (entry_ti.is_string() && entry_ti.get_compression() == kENCODING_DICT) {
      permutations.push_back(materializeDictionaryEncodedSortPermutation(order_entry));
    }
  }

  return permutations;
}

template <typename BUFFER_ITERATOR_TYPE>
std::vector<std::vector<int64_t>> ResultSet::MaterializedSortBuffers<
    BUFFER_ITERATOR_TYPE>::materializeCountDistinctColumns() const {
  // First, count the number of count distinct columns
  size_t count_distinct_col_count = 0;
  for (const auto& order_entry : order_entries_) {
    if (is_distinct_target(result_set_->targets_[order_entry.tle_no - 1])) {
      count_distinct_col_count++;
    }
  }

  // Reserve space for the exact number of count distinct columns
  std::vector<std::vector<int64_t>> buffers;
  buffers.reserve(count_distinct_col_count);

  // Populate the vector only for count distinct columns
  for (const auto& order_entry : order_entries_) {
    if (is_distinct_target(result_set_->targets_[order_entry.tle_no - 1])) {
      buffers.push_back(materializeCountDistinctColumn(order_entry));
    }
  }

  return buffers;
}

template <typename BUFFER_ITERATOR_TYPE>
ResultSet::ApproxQuantileBuffers ResultSet::MaterializedSortBuffers<
    BUFFER_ITERATOR_TYPE>::materializeApproxQuantileColumns() const {
  ResultSet::ApproxQuantileBuffers approx_quantile_materialized_buffers;
  for (const auto& order_entry : order_entries_) {
    if (result_set_->targets_[order_entry.tle_no - 1].agg_kind == kAPPROX_QUANTILE) {
      approx_quantile_materialized_buffers.emplace_back(
          materializeApproxQuantileColumn(order_entry));
    }
  }
  return approx_quantile_materialized_buffers;
}

template <typename BUFFER_ITERATOR_TYPE>
ResultSet::ModeBuffers
ResultSet::MaterializedSortBuffers<BUFFER_ITERATOR_TYPE>::materializeModeColumns() const {
  ResultSet::ModeBuffers mode_buffers;
  IsAggKind const is_mode(result_set_->targets_, kMODE);
  mode_buffers.reserve(
      std::count_if(order_entries_.begin(), order_entries_.end(), is_mode));
  for (auto const& order_entry : order_entries_) {
    if (is_mode(order_entry)) {
      mode_buffers.emplace_back(materializeModeColumn(order_entry));
    }
  }
  return mode_buffers;
}

template <typename BUFFER_ITERATOR_TYPE>
std::vector<int64_t>
ResultSet::MaterializedSortBuffers<BUFFER_ITERATOR_TYPE>::materializeCountDistinctColumn(
    const Analyzer::OrderEntry& order_entry) const {
  const size_t num_storage_entries = result_set_->query_mem_desc_.getEntryCount();
  std::vector<int64_t> count_distinct_materialized_buffer(num_storage_entries);
  const CountDistinctDescriptor count_distinct_descriptor =
      result_set_->query_mem_desc_.getCountDistinctDescriptor(order_entry.tle_no - 1);
  const size_t num_non_empty_entries = result_set_->permutation_.size();

  const auto work = [&,
                     parent_thread_local_ids = logger::thread_local_ids(),
                     result_set_ = this->result_set_](const size_t start,
                                                      const size_t end) {
    logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
    for (size_t i = start; i < end; ++i) {
      const PermutationIdx permuted_idx = result_set_->permutation_[i];
      const auto storage_lookup_result = result_set_->findStorage(permuted_idx);
      const auto storage = storage_lookup_result.storage_ptr;
      const auto off = storage_lookup_result.fixedup_entry_idx;
      const auto value = buffer_itr_.getColumnInternal(
          storage->buff_, off, order_entry.tle_no - 1, storage_lookup_result);
      count_distinct_materialized_buffer[permuted_idx] =
          count_distinct_set_size(value.i1, count_distinct_descriptor);
    }
  };
  // TODO(tlm): Allow use of tbb after we determine how to easily encapsulate the choice
  // between thread pool types
  if (single_threaded_) {
    work(0, num_non_empty_entries);
  } else {
    threading::task_group thread_pool;
    for (auto interval : makeIntervals<size_t>(0, num_non_empty_entries, cpu_threads())) {
      thread_pool.run([=] { work(interval.begin, interval.end); });
    }
    thread_pool.wait();
  }
  return count_distinct_materialized_buffer;
}

double ResultSet::calculateQuantile(quantile::TDigest* const t_digest) {
  static_assert(sizeof(int64_t) == sizeof(quantile::TDigest*));
  CHECK(t_digest);
  t_digest->mergeBufferFinal();
  double const quantile = t_digest->quantile();
  return boost::math::isnan(quantile) ? NULL_DOUBLE : quantile;
}

template <typename BUFFER_ITERATOR_TYPE>
ResultSet::ApproxQuantileBuffers::value_type
ResultSet::MaterializedSortBuffers<BUFFER_ITERATOR_TYPE>::materializeApproxQuantileColumn(
    const Analyzer::OrderEntry& order_entry) const {
  ResultSet::ApproxQuantileBuffers::value_type materialized_buffer(
      result_set_->query_mem_desc_.getEntryCount());
  const size_t size = result_set_->permutation_.size();
  const auto work = [&,
                     parent_thread_local_ids = logger::thread_local_ids(),
                     result_set_ = this->result_set_](const size_t start,
                                                      const size_t end) {
    logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
    for (size_t i = start; i < end; ++i) {
      const PermutationIdx permuted_idx = result_set_->permutation_[i];
      const auto storage_lookup_result = result_set_->findStorage(permuted_idx);
      const auto storage = storage_lookup_result.storage_ptr;
      const auto off = storage_lookup_result.fixedup_entry_idx;
      const auto value = buffer_itr_.getColumnInternal(
          storage->buff_, off, order_entry.tle_no - 1, storage_lookup_result);
      materialized_buffer[permuted_idx] =
          value.i1 ? calculateQuantile(reinterpret_cast<quantile::TDigest*>(value.i1))
                   : NULL_DOUBLE;
    }
  };
  if (single_threaded_) {
    work(0, size);
  } else {
    threading::task_group thread_pool;
    for (auto interval : makeIntervals<size_t>(0, size, cpu_threads())) {
      thread_pool.run([=] { work(interval.begin, interval.end); });
    }
    thread_pool.wait();
  }
  return materialized_buffer;
}

namespace {
// i1 is from InternalTargetValue
int64_t materialize_mode(RowSetMemoryOwner const* const rsmo, int64_t const i1) {
  if (AggMode const* const agg_mode = rsmo->getAggMode(i1)) {
    if (std::optional<int64_t> const mode = agg_mode->mode()) {
      return *mode;
    }
  }
  return NULL_BIGINT;
}

using ModeBlockedRange = tbb::blocked_range<size_t>;
}  // namespace

template <typename BUFFER_ITERATOR_TYPE>
struct ResultSet::MaterializedSortBuffers<BUFFER_ITERATOR_TYPE>::ModeScatter {
  logger::ThreadLocalIds const parent_thread_local_ids_;
  ResultSet::MaterializedSortBuffers<BUFFER_ITERATOR_TYPE> const* const shared_buffers_;
  RowSetMemoryOwner const* const row_set_memory_owner_;
  Analyzer::OrderEntry const& order_entry_;
  ResultSet::ModeBuffers::value_type& materialized_buffer_;

  void operator()(ModeBlockedRange const& r) const {
    logger::LocalIdsScopeGuard lisg = parent_thread_local_ids_.setNewThreadId();
    for (size_t i = r.begin(); i != r.end(); ++i) {
      PermutationIdx const permuted_idx = shared_buffers_->result_set_->permutation_[i];
      auto const storage_lookup_result =
          shared_buffers_->result_set_->findStorage(permuted_idx);
      auto const storage = storage_lookup_result.storage_ptr;
      auto const off = storage_lookup_result.fixedup_entry_idx;
      auto const value = shared_buffers_->buffer_itr_.getColumnInternal(
          storage->buff_, off, order_entry_.tle_no - 1, storage_lookup_result);
      materialized_buffer_[permuted_idx] =
          materialize_mode(row_set_memory_owner_, value.i1);
    }
  }
};

template <typename BUFFER_ITERATOR_TYPE>
ResultSet::ModeBuffers::value_type
ResultSet::MaterializedSortBuffers<BUFFER_ITERATOR_TYPE>::materializeModeColumn(
    const Analyzer::OrderEntry& order_entry) const {
  RowSetMemoryOwner const* const rsmo = result_set_->getRowSetMemOwner().get();
  ResultSet::ModeBuffers::value_type materialized_buffer(
      result_set_->query_mem_desc_.getEntryCount());
  ModeScatter mode_scatter{
      logger::thread_local_ids(), this, rsmo, order_entry, materialized_buffer};
  if (single_threaded_) {
    mode_scatter(ModeBlockedRange(
        0, result_set_->permutation_.size()));  // Still has new thread_id.
  } else {
    tbb::parallel_for(ModeBlockedRange(0, result_set_->permutation_.size()),
                      mode_scatter);
  }
  return materialized_buffer;
}

template <typename BUFFER_ITERATOR_TYPE>
bool ResultSet::ResultSetComparator<BUFFER_ITERATOR_TYPE>::operator()(
    const PermutationIdx lhs,
    const PermutationIdx rhs) const {
  // NB: The compare function must define a strict weak ordering, otherwise
  // std::sort will trigger a segmentation fault (or corrupt memory).
  const auto lhs_storage_lookup_result = result_set_->findStorage(lhs);
  const auto rhs_storage_lookup_result = result_set_->findStorage(rhs);
  const auto lhs_storage = lhs_storage_lookup_result.storage_ptr;
  const auto rhs_storage = rhs_storage_lookup_result.storage_ptr;
  const auto fixedup_lhs = lhs_storage_lookup_result.fixedup_entry_idx;
  const auto fixedup_rhs = rhs_storage_lookup_result.fixedup_entry_idx;
  size_t materialized_count_distinct_buffer_idx{0};
  size_t materialized_approx_quantile_buffer_idx{0};
  size_t materialized_mode_buffer_idx{0};
  size_t dictionary_string_sorted_permutation_idx{0};

  for (const auto& order_entry : order_entries_) {
    CHECK_GE(order_entry.tle_no, 1);
    // lhs_entry_ti and rhs_entry_ti can differ on comp_param w/ UNION of string dicts.
    const auto& lhs_agg_info = lhs_storage->targets_[order_entry.tle_no - 1];
    const auto& rhs_agg_info = rhs_storage->targets_[order_entry.tle_no - 1];
    const auto lhs_entry_ti = get_compact_type(lhs_agg_info);
    const auto rhs_entry_ti = get_compact_type(rhs_agg_info);
    // When lhs vs rhs doesn't matter, the lhs is used. For example:
    bool float_argument_input = takes_float_argument(lhs_agg_info);
    // Need to determine if the float value has been stored as float
    // or if it has been compacted to a different (often larger 8 bytes)
    // in distributed case the floats are actually 4 bytes
    // TODO the above takes_float_argument() is widely used wonder if this problem
    // exists elsewhere
    if (lhs_entry_ti.get_type() == kFLOAT) {
      const auto target_idx = static_cast<size_t>(order_entry.tle_no - 1);
      const auto is_col_lazy =
          !result_set_->lazy_fetch_info_.empty() &&
          result_set_->lazy_fetch_info_[target_idx].is_lazily_fetched;
      const auto slot_indices = result_set_->getSlotIndicesForTargetIndices();
      CHECK_LT(target_idx, slot_indices.size());
      if (should_read_float_argument_from_32bit_slot(result_set_->query_mem_desc_,
                                                     lhs_agg_info,
                                                     slot_indices[target_idx],
                                                     is_col_lazy)) {
        float_argument_input = true;
      }
    }

    if (UNLIKELY(is_distinct_target(lhs_agg_info))) {
      CHECK_LT(materialized_count_distinct_buffer_idx,
               count_distinct_materialized_buffers_.size());

      const auto& count_distinct_materialized_buffer =
          count_distinct_materialized_buffers_[materialized_count_distinct_buffer_idx];
      const auto lhs_sz = count_distinct_materialized_buffer[lhs];
      const auto rhs_sz = count_distinct_materialized_buffer[rhs];
      ++materialized_count_distinct_buffer_idx;
      if (lhs_sz == rhs_sz) {
        continue;
      }
      return (lhs_sz < rhs_sz) != order_entry.is_desc;
    } else if (UNLIKELY(lhs_agg_info.agg_kind == kAPPROX_QUANTILE)) {
      CHECK_LT(materialized_approx_quantile_buffer_idx,
               approx_quantile_materialized_buffers_.size());
      const auto& approx_quantile_materialized_buffer =
          approx_quantile_materialized_buffers_[materialized_approx_quantile_buffer_idx];
      const auto lhs_value = approx_quantile_materialized_buffer[lhs];
      const auto rhs_value = approx_quantile_materialized_buffer[rhs];
      ++materialized_approx_quantile_buffer_idx;
      if (lhs_value == rhs_value) {
        continue;
      } else if (!lhs_entry_ti.get_notnull()) {
        if (lhs_value == NULL_DOUBLE) {
          return order_entry.nulls_first;
        } else if (rhs_value == NULL_DOUBLE) {
          return !order_entry.nulls_first;
        }
      }
      return (lhs_value < rhs_value) != order_entry.is_desc;
    } else if (UNLIKELY(lhs_agg_info.agg_kind == kMODE)) {
      CHECK_LT(materialized_mode_buffer_idx, mode_buffers_.size());
      auto const& mode_buffer = mode_buffers_[materialized_mode_buffer_idx++];
      int64_t const lhs_value = mode_buffer[lhs];
      int64_t const rhs_value = mode_buffer[rhs];
      if (lhs_value == rhs_value) {
        continue;
        // MODE(x) can only be NULL when the group is empty, since it skips null values.
      } else if (lhs_value == NULL_BIGINT) {  // NULL_BIGINT from materialize_mode()
        return order_entry.nulls_first;
      } else if (rhs_value == NULL_BIGINT) {
        return !order_entry.nulls_first;
      } else {
        return result_set_->isLessThan(lhs_entry_ti, lhs_value, rhs_value) !=
               order_entry.is_desc;
      }
    }

    const auto lhs_v = buffer_itr_.getColumnInternal(lhs_storage->buff_,
                                                     fixedup_lhs,
                                                     order_entry.tle_no - 1,
                                                     lhs_storage_lookup_result);
    const auto rhs_v = buffer_itr_.getColumnInternal(rhs_storage->buff_,
                                                     fixedup_rhs,
                                                     order_entry.tle_no - 1,
                                                     rhs_storage_lookup_result);

    if (UNLIKELY(isNull(lhs_entry_ti, lhs_v, float_argument_input) &&
                 isNull(rhs_entry_ti, rhs_v, float_argument_input))) {
      continue;
    }
    if (UNLIKELY(isNull(lhs_entry_ti, lhs_v, float_argument_input) &&
                 !isNull(rhs_entry_ti, rhs_v, float_argument_input))) {
      return order_entry.nulls_first;
    }
    if (UNLIKELY(isNull(rhs_entry_ti, rhs_v, float_argument_input) &&
                 !isNull(lhs_entry_ti, lhs_v, float_argument_input))) {
      return !order_entry.nulls_first;
    }

    if (LIKELY(lhs_v.isInt())) {
      CHECK(rhs_v.isInt());
      if (UNLIKELY(lhs_entry_ti.is_string() &&
                   lhs_entry_ti.get_compression() == kENCODING_DICT)) {
        CHECK_EQ(4, lhs_entry_ti.get_logical_size());
        CHECK(executor_);
        if (lhs_v.i1 == rhs_v.i1) {
          ++dictionary_string_sorted_permutation_idx;
          continue;
        }
        CHECK_LT(dictionary_string_sorted_permutation_idx,
                 dictionary_string_sorted_permutations_.size());
        return dictionary_string_sorted_permutations_
            [dictionary_string_sorted_permutation_idx++](static_cast<int32_t>(lhs_v.i1),
                                                         static_cast<int32_t>(rhs_v.i1),
                                                         order_entry.is_desc,
                                                         order_entry.nulls_first);
      }
      if (lhs_v.i1 == rhs_v.i1) {
        continue;
      }
      if (lhs_entry_ti.is_fp()) {
        if (float_argument_input) {
          const auto lhs_dval = *reinterpret_cast<const float*>(may_alias_ptr(&lhs_v.i1));
          const auto rhs_dval = *reinterpret_cast<const float*>(may_alias_ptr(&rhs_v.i1));
          return (lhs_dval < rhs_dval) != order_entry.is_desc;
        } else {
          const auto lhs_dval =
              *reinterpret_cast<const double*>(may_alias_ptr(&lhs_v.i1));
          const auto rhs_dval =
              *reinterpret_cast<const double*>(may_alias_ptr(&rhs_v.i1));
          return (lhs_dval < rhs_dval) != order_entry.is_desc;
        }
      }
      return (lhs_v.i1 < rhs_v.i1) != order_entry.is_desc;
    } else {
      if (lhs_v.isPair()) {
        CHECK(rhs_v.isPair());
        const auto lhs =
            pair_to_double({lhs_v.i1, lhs_v.i2}, lhs_entry_ti, float_argument_input);
        const auto rhs =
            pair_to_double({rhs_v.i1, rhs_v.i2}, rhs_entry_ti, float_argument_input);
        if (lhs == rhs) {
          continue;
        }
        return (lhs < rhs) != order_entry.is_desc;
      } else {
        CHECK(lhs_v.isStr() && rhs_v.isStr());
        const auto lhs = lhs_v.strVal();
        const auto rhs = rhs_v.strVal();
        if (lhs == rhs) {
          continue;
        }
        return (lhs < rhs) != order_entry.is_desc;
      }
    }
  }
  return false;
}

// A separate topPermutationImpl() would not be needed if the comparison operator was
// called as a virtual function. The downside would be a vtable lookup on each comparison.
// Doing the dynamic_cast() here effectively hoists the vtable lookup out of the loop.
PermutationView ResultSet::topPermutation(PermutationView permutation,
                                          const size_t top_n,
                                          const ResultSetComparatorBase* comparator) {
  if (auto* rsc = dynamic_cast<ResultSetComparator<ColumnWiseTargetAccessor> const*>(
          comparator)) {
    return topPermutationImpl(permutation, top_n, rsc);
  } else if (auto* rsc = dynamic_cast<ResultSetComparator<RowWiseTargetAccessor> const*>(
                 comparator)) {
    return topPermutationImpl(permutation, top_n, rsc);
  } else {
    UNREACHABLE();
    return {};
  }
}

// Partial sort permutation into top(least by compare) top_n elements.
// If permutation.size() <= top_n then sort entire permutation by compare.
// Return PermutationView with new size() = min(top_n, permutation.size()).
template <typename BUFFER_ITERATOR_TYPE>
PermutationView ResultSet::topPermutationImpl(
    PermutationView permutation,
    const size_t top_n,
    const ResultSet::ResultSetComparator<BUFFER_ITERATOR_TYPE>* rsc) {
  auto timer = DEBUG_TIMER(__func__);
  // sort() requires a copyable comparison operator.
  auto compare_op = [rsc](PermutationIdx a, PermutationIdx b) { return (*rsc)(a, b); };
  bool const top_k_sort = top_n < permutation.size();
  if (top_k_sort) {
    std::partial_sort(
        permutation.begin(), permutation.begin() + top_n, permutation.end(), compare_op);
    permutation.resize(top_n);
  } else {
    // non-top-k sort, need to sort an entire permutation range
    if (rsc->single_threaded_) {
      std::sort(permutation.begin(), permutation.end(), compare_op);
    } else {
      tbb::parallel_sort(permutation.begin(), permutation.end(), compare_op);
    }
  }
  return permutation;
}

void ResultSet::radixSortOnGpu(
    const std::list<Analyzer::OrderEntry>& order_entries) const {
  auto timer = DEBUG_TIMER(__func__);
  const int device_id{0};
  CHECK_GT(block_size_, 0);
  CHECK_GT(grid_size_, 0);
  std::vector<int64_t*> group_by_buffers(block_size_);
  group_by_buffers[0] = reinterpret_cast<int64_t*>(storage_->getUnderlyingBuffer());
  auto dev_group_by_buffers =
      create_dev_group_by_buffers(getCudaAllocator(),
                                  group_by_buffers,
                                  query_mem_desc_,
                                  block_size_,
                                  grid_size_,
                                  device_id,
                                  ExecutorDispatchMode::KernelPerFragment,
                                  /*num_input_rows=*/-1,
                                  /*prepend_index_buffer=*/true,
                                  /*always_init_group_by_on_host=*/true,
                                  /*use_bump_allocator=*/false,
                                  /*has_varlen_output=*/false,
                                  /*insitu_allocator*=*/nullptr);
  inplace_sort_gpu(order_entries,
                   query_mem_desc_,
                   dev_group_by_buffers,
                   getCudaAllocator(),
                   getCudaStream());
  copy_group_by_buffers_from_gpu(
      *getCudaAllocator(),
      group_by_buffers,
      query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU),
      dev_group_by_buffers.data,
      query_mem_desc_,
      block_size_,
      grid_size_,
      device_id,
      getCudaStream(),
      /*use_bump_allocator=*/false,
      /*has_varlen_output=*/false);
}

void ResultSet::radixSortOnCpu(
    const std::list<Analyzer::OrderEntry>& order_entries) const {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(!query_mem_desc_.hasKeylessHash());
  size_t max_slot_width = sizeof(int64_t);
  for (size_t slot_idx = 0; slot_idx < query_mem_desc_.getSlotCount(); ++slot_idx) {
    max_slot_width =
        std::max(max_slot_width,
                 static_cast<size_t>(query_mem_desc_.getPaddedSlotWidthBytes(slot_idx)));
  }
  std::vector<int64_t> tmp_buff(
      (query_mem_desc_.getEntryCount() * max_slot_width + sizeof(int64_t) - 1) /
      sizeof(int64_t));
  std::vector<int32_t> idx_buff(query_mem_desc_.getEntryCount());
  CHECK_EQ(size_t(1), order_entries.size());
  auto buffer_ptr = storage_->getUnderlyingBuffer();
  for (const auto& order_entry : order_entries) {
    CHECK_GE(order_entry.tle_no, 1);
    const auto target_idx = static_cast<size_t>(order_entry.tle_no - 1);
    const auto& target_slots =
        query_mem_desc_.getColSlotContext().getSlotsForCol(target_idx);
    CHECK(!target_slots.empty());
    const auto sort_slot_idx = target_slots.front();
    const auto sortkey_val_buff = reinterpret_cast<int64_t*>(
        buffer_ptr + query_mem_desc_.getColOffInBytes(sort_slot_idx));
    const auto chosen_bytes = query_mem_desc_.getPaddedSlotWidthBytes(sort_slot_idx);
    sort_groups_cpu(sortkey_val_buff,
                    &idx_buff[0],
                    query_mem_desc_.getEntryCount(),
                    order_entry.is_desc,
                    chosen_bytes);
    apply_permutation_cpu(reinterpret_cast<int64_t*>(buffer_ptr),
                          &idx_buff[0],
                          query_mem_desc_.getEntryCount(),
                          &tmp_buff[0],
                          sizeof(int64_t));
    for (size_t target_idx = 0; target_idx < query_mem_desc_.getSlotCount();
         ++target_idx) {
      if (target_idx == sort_slot_idx) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc_.getPaddedSlotWidthBytes(target_idx);
      const auto satellite_val_buff = reinterpret_cast<int64_t*>(
          buffer_ptr + query_mem_desc_.getColOffInBytes(target_idx));
      apply_permutation_cpu(satellite_val_buff,
                            &idx_buff[0],
                            query_mem_desc_.getEntryCount(),
                            &tmp_buff[0],
                            chosen_bytes);
    }
  }
}

size_t ResultSet::getLimit() const {
  return keep_first_;
}

const std::vector<std::string> ResultSet::getStringDictionaryPayloadCopy(
    const shared::StringDictKey& dict_key) const {
  const auto sdp =
      row_set_mem_owner_->getOrAddStringDictProxy(dict_key, /*with_generation=*/true);
  CHECK(sdp);
  return sdp->getDictionary()->copyStrings();
}

const ResultSet::UniqueStringsForDictEncodedTargetCol
ResultSet::getUniqueStringsForDictEncodedTargetCol(const size_t col_idx) const {
  auto unique_strings = getUniqueStringsForDictEncodedTargetCols({col_idx});
  CHECK_EQ(unique_strings.size(), size_t(1));
  return std::move(unique_strings.front());
}

std::vector<ResultSet::UniqueStringsForDictEncodedTargetCol>
ResultSet::getUniqueStringsForDictEncodedTargetCols(
    const std::vector<size_t>& col_indices) const {
  if (col_indices.empty()) {
    return {};
  }

  std::vector<SQLTypeInfo> col_types;
  col_types.reserve(col_indices.size());
  std::vector<int64_t> null_values;
  null_values.reserve(col_indices.size());
  std::vector<bool> targets_to_skip(colCount(), true);
  for (const auto col_idx : col_indices) {
    CHECK_LT(col_idx, colCount());
    const auto col_type = getColType(col_idx);
    CHECK(col_type.is_dict_encoded_type());  // Array<Text> or Text
    col_types.push_back(col_type);
    null_values.push_back(inline_fixed_encoding_null_val(
        col_type.is_array() ? col_type.get_elem_type() : col_type));
    targets_to_skip[col_idx] = false;
  }

  using StringIdsByColumn = std::vector<std::vector<int32_t>>;
  const auto collect_string_ids = [&](const size_t begin,
                                      const size_t end,
                                      StringIdsByColumn& string_ids_by_column) {
    for (size_t row_idx = begin; row_idx < end; ++row_idx) {
      const auto result_row = getRowAtNoTranslations(row_idx, targets_to_skip);
      if (result_row.empty()) {
        continue;
      }
      for (size_t selected_col_idx = 0; selected_col_idx < col_indices.size();
           ++selected_col_idx) {
        const auto result_col_idx = col_indices[selected_col_idx];
        auto& string_ids = string_ids_by_column[selected_col_idx];
        const auto null_val = null_values[selected_col_idx];
        if (const auto scalar_col_val =
                boost::get<ScalarTargetValue>(&result_row[result_col_idx])) {
          const int32_t string_id =
              static_cast<int32_t>(boost::get<int64_t>(*scalar_col_val));
          if (string_id != null_val) {
            string_ids.push_back(string_id);
          }
        } else if (const auto array_col_val =
                       boost::get<ArrayTargetValue>(&result_row[result_col_idx])) {
          if (*array_col_val) {
            for (const ScalarTargetValue& scalar : array_col_val->value()) {
              const int32_t string_id = static_cast<int32_t>(boost::get<int64_t>(scalar));
              if (string_id != null_val) {
                string_ids.push_back(string_id);
              }
            }
          }
        }
      }
    }
  };

  const size_t num_entries = entryCount();
  const bool use_parallel_collection = num_entries > 10000 && !isTruncated();
  const auto worker_count =
      use_parallel_collection
          ? std::max<size_t>(
                1, std::min<size_t>(static_cast<size_t>(cpu_threads()), num_entries))
          : size_t(1);
  std::vector<Interval<size_t>> intervals;
  intervals.reserve(worker_count);
  for (const auto& interval : makeIntervals<size_t>(0, num_entries, worker_count)) {
    intervals.push_back(interval);
  }
  std::vector<StringIdsByColumn> string_id_segments;
  string_id_segments.reserve(intervals.size());
  for (const auto& interval : intervals) {
    auto& segment = string_id_segments.emplace_back(col_indices.size());
    for (auto& string_ids : segment) {
      string_ids.reserve(interval.end - interval.begin);
    }
  }

  const auto parent_thread_local_ids = logger::thread_local_ids();
  if (intervals.size() == 1) {
    collect_string_ids(
        intervals.front().begin, intervals.front().end, string_id_segments.front());
  } else if (!intervals.empty()) {
    threading::parallel_for(
        threading::blocked_range<size_t>(0, intervals.size()),
        [&](const threading::blocked_range<size_t>& worker_range) {
          logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
          for (size_t worker_idx = worker_range.begin(); worker_idx != worker_range.end();
               ++worker_idx) {
            collect_string_ids(intervals[worker_idx].begin,
                               intervals[worker_idx].end,
                               string_id_segments[worker_idx]);
          }
        });
  }

  StringIdsByColumn string_ids_by_column(col_indices.size());
  for (size_t selected_col_idx = 0; selected_col_idx < col_indices.size();
       ++selected_col_idx) {
    auto& combined_string_ids = string_ids_by_column[selected_col_idx];
    size_t combined_size{0};
    for (const auto& segment : string_id_segments) {
      combined_size += segment[selected_col_idx].size();
    }
    combined_string_ids.reserve(combined_size);
    for (auto& segment : string_id_segments) {
      const auto& segment_string_ids = segment[selected_col_idx];
      combined_string_ids.insert(combined_string_ids.end(),
                                 segment_string_ids.begin(),
                                 segment_string_ids.end());
    }
  }

  std::vector<UniqueStringsForDictEncodedTargetCol> unique_strings;
  unique_strings.reserve(col_indices.size());
  for (size_t selected_col_idx = 0; selected_col_idx < col_indices.size();
       ++selected_col_idx) {
    auto& unique_string_ids = string_ids_by_column[selected_col_idx];
    std::sort(unique_string_ids.begin(), unique_string_ids.end());
    unique_string_ids.erase(
        std::unique(unique_string_ids.begin(), unique_string_ids.end()),
        unique_string_ids.end());

    const auto sdp = row_set_mem_owner_->getOrAddStringDictProxy(
        col_types[selected_col_idx].getStringDictKey(), /*with_generation=*/true);
    CHECK(sdp);
    auto strings = sdp->getStrings(unique_string_ids);
    unique_strings.emplace_back(std::move(unique_string_ids), std::move(strings));
  }
  return unique_strings;
}

/**
 * Determines if it is possible to directly form a ColumnarResults class from this
 * result set, bypassing the default columnarization.
 *
 * NOTE: If there exists a permutation vector (i.e., in some ORDER BY queries), it
 * becomes equivalent to the row-wise columnarization.
 */
bool ResultSet::isDirectColumnarConversionPossible() const {
  if (!g_enable_direct_columnarization) {
    return false;
  } else if (isTruncated()) {
    return false;
  } else if (query_mem_desc_.didOutputColumnar()) {
    return permutation_.empty() && (query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::Projection ||
                                    query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::TableFunction ||
                                    query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::GroupByPerfectHash ||
                                    query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::GroupByBaselineHash);
  } else {
    CHECK(!(query_mem_desc_.getQueryDescriptionType() ==
            QueryDescriptionType::TableFunction));
    return permutation_.empty() && (query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::GroupByPerfectHash ||
                                    query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::GroupByBaselineHash);
  }
}

bool ResultSet::isZeroCopyColumnarConversionPossible(size_t column_idx) const {
  return query_mem_desc_.didOutputColumnar() &&
         (query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection ||
          query_mem_desc_.getQueryDescriptionType() ==
              QueryDescriptionType::TableFunction) &&
         appended_storage_.empty() && storage_ &&
         (lazy_fetch_info_.empty() || !lazy_fetch_info_[column_idx].is_lazily_fetched) &&
         !isTruncated();
}

const int8_t* ResultSet::getColumnarBuffer(size_t column_idx) const {
  materializeDeviceColumnarCpuStorageIfNeeded();
  CHECK(isZeroCopyColumnarConversionPossible(column_idx));
  return storage_->getUnderlyingBuffer() + query_mem_desc_.getColOffInBytes(column_idx);
}

const size_t ResultSet::getColumnarBufferSize(size_t column_idx) const {
  const auto col_context = query_mem_desc_.getColSlotContext();
  const auto idx = col_context.getSlotsForCol(column_idx).front();
  return query_mem_desc_.getPaddedSlotBufferSize(idx);
  if (checkSlotUsesFlatBufferFormat(idx)) {
    return query_mem_desc_.getFlatBufferSize(idx);
  }
  const size_t padded_slot_width = static_cast<size_t>(getPaddedSlotWidthBytes(idx));
  return padded_slot_width * entryCount();
}

void ResultSet::addDeviceColumnarBufferFragment(const size_t column_idx,
                                                const int device_id,
                                                const int8_t* buffer,
                                                const size_t entry_count) {
  CHECK(buffer);
  CHECK(cuda_allocator_);
  CHECK_LT(column_idx, targets_.size());
  if (device_columnar_fragments_.size() < targets_.size()) {
    device_columnar_fragments_.resize(targets_.size());
  }
  device_columnar_fragments_[column_idx].push_back(
      DeviceColumnarBufferFragment{buffer,
                                   entry_count,
                                   device_id,
                                   cuda_allocator_,
                                   cuda_allocator_->recordReadyEvent()});
}

void ResultSet::clearDeviceColumnarBufferFragments() {
  device_columnar_fragments_.clear();
  device_columnar_fragments_cover_logical_rows_ = false;
  device_columnar_fragments_form_dense_cpu_rows_ = false;
  device_columnar_fragments_exclude_baseline_boundary_keys_ = false;
  device_columnar_fragments_cover_cpu_baseline_boundary_rows_ = false;
}

void ResultSet::markDeviceColumnarFragmentsCoverLogicalRows() const {
  device_columnar_fragments_cover_logical_rows_ = true;
}

void ResultSet::markDeviceColumnarFragmentsFormDenseCpuRows() const {
  CHECK(device_columnar_fragments_cover_logical_rows_);
  device_columnar_fragments_form_dense_cpu_rows_ = true;
}

void ResultSet::markDeviceColumnarFragmentsExcludeBaselineBoundaryKeys() const {
  device_columnar_fragments_exclude_baseline_boundary_keys_ = true;
}

void ResultSet::markDeviceColumnarFragmentsCoverCpuBaselineBoundaryRows() const {
  device_columnar_fragments_cover_cpu_baseline_boundary_rows_ = true;
}

void ResultSet::addDeviceRowwiseBufferFragment(const int device_id,
                                               const int8_t* buffer,
                                               const size_t entry_count) {
  CHECK(buffer);
  CHECK(cuda_allocator_);
  device_rowwise_fragments_.push_back(
      DeviceRowwiseBufferFragment{buffer,
                                  entry_count,
                                  device_id,
                                  cuda_allocator_,
                                  cuda_allocator_->recordReadyEvent()});
  {
    std::lock_guard<std::mutex> lock(device_columnar_cpu_storage_mutex_);
    device_columnar_cpu_storage_valid_.store(false, std::memory_order_release);
  }
  const int64_t cached_row_count = cached_row_count_;
  if (cached_row_count != uninitialized_cached_row_count) {
    CHECK_GE(cached_row_count, 0);
    CHECK_LE(static_cast<size_t>(cached_row_count), entryCount());
  }
}

void ResultSet::clearDeviceRowwiseBufferFragments() {
  device_rowwise_fragments_.clear();
}

bool ResultSet::getDeviceRowwiseBufferFragments(
    std::vector<DeviceRowwiseBufferFragment>& fragments) const {
  fragments.clear();
  if (isTruncated() || !permutation_.empty() || device_rowwise_fragments_.empty()) {
    return false;
  }
  const auto query_type = query_mem_desc_.getQueryDescriptionType();
  const bool can_expose_rowwise_group_by =
      query_type == QueryDescriptionType::GroupByBaselineHash ||
      query_type == QueryDescriptionType::GroupByPerfectHash;
  if (!can_expose_rowwise_group_by || query_mem_desc_.didOutputColumnar() ||
      (query_mem_desc_.hasKeylessHash() &&
       query_type != QueryDescriptionType::GroupByPerfectHash) ||
      query_mem_desc_.hasVarlenOutput()) {
    return false;
  }
  const auto fragment_entry_count = std::accumulate(
      device_rowwise_fragments_.begin(),
      device_rowwise_fragments_.end(),
      size_t(0),
      [](const size_t total, const DeviceRowwiseBufferFragment& fragment) {
        CHECK(fragment.buffer);
        CHECK(fragment.owner);
        return total + fragment.entry_count;
      });
  if (fragment_entry_count != entryCount()) {
    return false;
  }
  fragments = device_rowwise_fragments_;
  return true;
}

bool ResultSet::appendDeviceColumnarFragmentsFromCpuBaselineHashResult(
    const ResultSet& source) {
  if (!cuda_allocator_ || device_columnar_fragments_.empty() || source.colCount() == 0 ||
      source.colCount() != colCount() || source.query_mem_desc_.didOutputColumnar() ||
      source.query_mem_desc_.getQueryDescriptionType() !=
          QueryDescriptionType::GroupByBaselineHash ||
      source.query_mem_desc_.hasKeylessHash() ||
      source.query_mem_desc_.hasVarlenOutput()) {
    return false;
  }

  const auto row_count = source.rowCount();
  if (row_count == 0) {
    return true;
  }

  struct ColumnCopy {
    size_t column_idx;
    size_t elem_size;
    SQLTypeInfo logical_ti;
    std::vector<int8_t> host_buffer;
  };

  std::vector<ColumnCopy> columns;
  columns.reserve(source.colCount());
  for (size_t column_idx = 0; column_idx < source.colCount(); ++column_idx) {
    if (column_idx >= device_columnar_fragments_.size() ||
        device_columnar_fragments_[column_idx].empty()) {
      return false;
    }
    const auto logical_ti = get_logical_type_info(source.getColType(column_idx));
    if (logical_ti.is_varlen()) {
      return false;
    }
    const auto elem_size = logical_ti.get_size();
    if (elem_size <= 0) {
      return false;
    }
    if (row_count > std::numeric_limits<size_t>::max() / static_cast<size_t>(elem_size)) {
      return false;
    }
    columns.push_back(
        ColumnCopy{column_idx,
                   static_cast<size_t>(elem_size),
                   logical_ti,
                   std::vector<int8_t>(row_count * static_cast<size_t>(elem_size))});
  }

  size_t output_row_idx = 0;
  for (size_t entry_idx = 0; entry_idx < source.entryCount(); ++entry_idx) {
    const auto row = source.getRowAtNoTranslations(entry_idx);
    if (row.empty()) {
      continue;
    }
    if (row.size() != columns.size() || output_row_idx >= row_count) {
      return false;
    }
    for (auto& column : columns) {
      const auto* scalar = boost::get<ScalarTargetValue>(&row[column.column_idx]);
      if (!scalar) {
        return false;
      }
      auto* const destination =
          column.host_buffer.data() + output_row_idx * column.elem_size;
      if (column.logical_ti.get_type() == kFLOAT) {
        const auto* value = boost::get<float>(scalar);
        if (!value || column.elem_size != sizeof(*value)) {
          return false;
        }
        memcpy(destination, value, sizeof(*value));
      } else if (column.logical_ti.get_type() == kDOUBLE) {
        const auto* value = boost::get<double>(scalar);
        if (!value || column.elem_size != sizeof(*value)) {
          return false;
        }
        memcpy(destination, value, sizeof(*value));
      } else {
        const bool supported_integral_type =
            column.logical_ti.is_integer() || column.logical_ti.is_decimal() ||
            column.logical_ti.is_boolean() || column.logical_ti.is_time() ||
            column.logical_ti.is_timeinterval() ||
            column.logical_ti.is_dict_encoded_string();
        const auto* value = boost::get<int64_t>(scalar);
        if (!supported_integral_type || !value ||
            (column.elem_size != sizeof(int8_t) && column.elem_size != sizeof(int16_t) &&
             column.elem_size != sizeof(int32_t) &&
             column.elem_size != sizeof(int64_t))) {
          return false;
        }
        write_int_to_buff(destination, static_cast<int8_t>(column.elem_size), *value);
      }
    }
    ++output_row_idx;
  }
  if (output_row_idx != row_count) {
    return false;
  }

  const auto device_id = cuda_allocator_->getDeviceId();
  for (const auto& column : columns) {
    auto& fragments = device_columnar_fragments_[column.column_idx];
    fragments.reserve(fragments.size() + 1);
  }
  auto boundary_allocator = std::make_shared<CudaAllocator>(
      cuda_allocator_->getDataMgr(), device_id, cuda_allocator_->getCudaStream());
  std::vector<int8_t*> device_buffers;
  device_buffers.reserve(columns.size());
  for (const auto& column : columns) {
    const auto column_bytes = column.host_buffer.size();
    auto* device_buffer = boundary_allocator->alloc(column_bytes);
    boundary_allocator->copyToDevice(device_buffer,
                                     column.host_buffer.data(),
                                     column_bytes,
                                     "Boundary baseline hash device columnar fragment");
    device_buffers.push_back(device_buffer);
  }
  auto ready_event = boundary_allocator->recordReadyEvent();
  for (size_t column_idx = 0; column_idx < columns.size(); ++column_idx) {
    device_columnar_fragments_[columns[column_idx].column_idx].push_back(
        DeviceColumnarBufferFragment{device_buffers[column_idx],
                                     row_count,
                                     device_id,
                                     boundary_allocator,
                                     ready_event});
  }
  markDeviceColumnarFragmentsCoverLogicalRows();
  markDeviceColumnarFragmentsCoverCpuBaselineBoundaryRows();
  return true;
}

bool ResultSet::appendDeviceOnlyColumnarFragmentsFromCpuBaselineHashResult(
    const ResultSet& source) {
  if (query_mem_desc_.getQueryDescriptionType() !=
          QueryDescriptionType::GroupByBaselineHash ||
      query_mem_desc_.didOutputColumnar()) {
    return false;
  }
  const auto source_row_count = source.rowCount();
  if (source_row_count == 0) {
    return true;
  }
  const auto original_entry_count = entryCount();
  if (source_row_count > std::numeric_limits<size_t>::max() - original_entry_count) {
    return false;
  }
  std::vector<DeviceColumnarFragmentInfo> fragment_info;
  if (!getDeviceColumnarFragmentInfo(fragment_info)) {
    return false;
  }
  if (!canDeferDeviceColumnarCpuMaterialization()) {
    return false;
  }
  if (!appendDeviceColumnarFragmentsFromCpuBaselineHashResult(source)) {
    return false;
  }

  const auto appended_entry_count = original_entry_count + source_row_count;
  invalidateCachedRowCount();
  query_mem_desc_.setEntryCount(appended_entry_count);
  markBaselineHashDenseForReduction(appended_entry_count);
  CHECK(canDeferDeviceColumnarCpuMaterialization());
  clearDeviceRowwiseBufferFragments();
  CHECK(canDeferDeviceColumnarCpuMaterialization());
  markDeviceColumnarCpuStorageInvalid();
  return true;
}

bool ResultSet::canDeferDeviceColumnarCpuMaterialization() const {
  if (!storage_ || isTruncated() || !permutation_.empty()) {
    return false;
  }
  const auto query_type = query_mem_desc_.getQueryDescriptionType();
  const bool columnar_projection_result = query_mem_desc_.didOutputColumnar() &&
                                          query_type == QueryDescriptionType::Projection;
  const bool rowwise_projection_result = !query_mem_desc_.didOutputColumnar() &&
                                         query_type == QueryDescriptionType::Projection &&
                                         !query_mem_desc_.hasVarlenOutput();
  const bool projection_result = columnar_projection_result || rowwise_projection_result;
  const bool rowwise_group_by_result =
      !query_mem_desc_.didOutputColumnar() &&
      (query_type == QueryDescriptionType::GroupByPerfectHash ||
       query_type == QueryDescriptionType::GroupByBaselineHash) &&
      !query_mem_desc_.hasVarlenOutput();
  if (!projection_result && !rowwise_group_by_result) {
    return false;
  }
  if (columnar_projection_result && device_columnar_fragments_.empty()) {
    return false;
  }
  if (rowwise_group_by_result) {
    const int64_t cached_row_count = cached_row_count_;
    if (cached_row_count == uninitialized_cached_row_count || cached_row_count < 0 ||
        static_cast<size_t>(cached_row_count) > entryCount()) {
      return false;
    }
    if (query_type == QueryDescriptionType::GroupByBaselineHash &&
        !baseline_hash_dense_for_reduction_) {
      return false;
    }
  }
  for (size_t column_idx = 0; column_idx < targets_.size(); ++column_idx) {
    const auto column_ti = getColType(column_idx);
    if (projection_result && column_ti.is_string() &&
        !column_ti.is_dict_encoded_string()) {
      return false;
    }
    const auto logical_ti = get_logical_type_info(column_ti);
    const auto elem_size = logical_ti.get_size();
    if (elem_size <= 0 || logical_ti.is_varlen()) {
      return false;
    }
    if (!lazy_fetch_info_.empty()) {
      CHECK_LT(column_idx, lazy_fetch_info_.size());
      if (lazy_fetch_info_[column_idx].is_lazily_fetched) {
        return false;
      }
    }
  }
  if (rowwise_projection_result) {
    const int64_t cached_row_count = cached_row_count_;
    if (cached_row_count == uninitialized_cached_row_count || cached_row_count < 0 ||
        static_cast<size_t>(cached_row_count) != entryCount()) {
      return false;
    }
  }
  if ((rowwise_projection_result || rowwise_group_by_result) &&
      !device_rowwise_fragments_.empty()) {
    const auto fragment_entry_count = std::accumulate(
        device_rowwise_fragments_.begin(),
        device_rowwise_fragments_.end(),
        size_t(0),
        [](const size_t total, const DeviceRowwiseBufferFragment& fragment) {
          CHECK(fragment.buffer);
          CHECK(fragment.owner);
          return total + fragment.entry_count;
        });
    if (fragment_entry_count == entryCount()) {
      return true;
    }
  }
  if (rowwise_projection_result) {
    return false;
  }
  if (device_columnar_fragments_.empty()) {
    return false;
  }
  for (size_t column_idx = 0; column_idx < targets_.size(); ++column_idx) {
    const auto logical_ti = get_logical_type_info(getColType(column_idx));
    const auto elem_size = logical_ti.get_size();
    std::vector<DeviceColumnarBufferFragment> fragments;
    if (!getDeviceColumnarBufferFragments(
            column_idx, static_cast<size_t>(elem_size), fragments) ||
        fragments.empty()) {
      return false;
    }
  }
  return true;
}

void ResultSet::markDeviceColumnarCpuStorageInvalid() const {
  CHECK(canDeferDeviceColumnarCpuMaterialization());
  std::lock_guard<std::mutex> lock(device_columnar_cpu_storage_mutex_);
  device_columnar_cpu_storage_valid_.store(false, std::memory_order_release);
  const int64_t cached_row_count = cached_row_count_;
  if (cached_row_count == uninitialized_cached_row_count) {
    setCachedRowCount(entryCount());
  } else {
    CHECK_GE(cached_row_count, 0);
    CHECK_LE(static_cast<size_t>(cached_row_count), entryCount());
  }
}

void ResultSet::markDeviceColumnarCpuStorageValid() const {
  std::lock_guard<std::mutex> lock(device_columnar_cpu_storage_mutex_);
  device_columnar_cpu_storage_valid_.store(true, std::memory_order_release);
}

namespace {

void copy_widened_fixed_width_value(int8_t* dst,
                                    const size_t dst_width,
                                    const int8_t* src,
                                    const size_t src_width,
                                    const SQLTypeInfo& logical_ti) {
  if (dst_width == src_width) {
    memcpy(dst, src, dst_width);
    return;
  }
  CHECK(logical_ti.is_integer() || logical_ti.is_decimal() || logical_ti.is_boolean() ||
        logical_ti.is_time() || logical_ti.is_timeinterval() ||
        logical_ti.is_dict_encoded_string())
      << "Unsupported deferred group-by materialization width conversion for "
      << logical_ti.to_string();

  int64_t value{0};
  switch (src_width) {
    case 1: {
      int8_t typed_value;
      memcpy(&typed_value, src, sizeof(typed_value));
      value = typed_value;
      break;
    }
    case 2: {
      int16_t typed_value;
      memcpy(&typed_value, src, sizeof(typed_value));
      value = typed_value;
      break;
    }
    case 4: {
      int32_t typed_value;
      memcpy(&typed_value, src, sizeof(typed_value));
      value = typed_value;
      break;
    }
    case 8: {
      memcpy(&value, src, sizeof(value));
      break;
    }
    default:
      UNREACHABLE();
  }

  switch (dst_width) {
    case 1: {
      const auto typed_value = static_cast<int8_t>(value);
      memcpy(dst, &typed_value, sizeof(typed_value));
      break;
    }
    case 2: {
      const auto typed_value = static_cast<int16_t>(value);
      memcpy(dst, &typed_value, sizeof(typed_value));
      break;
    }
    case 4: {
      const auto typed_value = static_cast<int32_t>(value);
      memcpy(dst, &typed_value, sizeof(typed_value));
      break;
    }
    case 8: {
      memcpy(dst, &value, sizeof(value));
      break;
    }
    default:
      UNREACHABLE();
  }
}

}  // namespace

void ResultSet::materializeDeviceColumnarCpuStorageIfNeeded() const {
  if (device_columnar_cpu_storage_valid_.load(std::memory_order_acquire)) {
    return;
  }
  std::lock_guard<std::mutex> lock(device_columnar_cpu_storage_mutex_);
  if (device_columnar_cpu_storage_valid_.load(std::memory_order_acquire)) {
    return;
  }
  CHECK(storage_);
  CHECK(lazy_fetch_info_.empty() ||
        std::none_of(lazy_fetch_info_.begin(),
                     lazy_fetch_info_.end(),
                     [](const auto& info) { return info.is_lazily_fetched; }));
  const auto query_type = query_mem_desc_.getQueryDescriptionType();
  const bool columnar_projection_result = query_mem_desc_.didOutputColumnar() &&
                                          query_type == QueryDescriptionType::Projection;
  const bool rowwise_projection_result = !query_mem_desc_.didOutputColumnar() &&
                                         query_type == QueryDescriptionType::Projection &&
                                         !query_mem_desc_.hasVarlenOutput();
  const bool rowwise_group_by_result =
      !query_mem_desc_.didOutputColumnar() &&
      (query_type == QueryDescriptionType::GroupByPerfectHash ||
       query_type == QueryDescriptionType::GroupByBaselineHash) &&
      !query_mem_desc_.hasVarlenOutput();
  CHECK(columnar_projection_result || rowwise_projection_result ||
        rowwise_group_by_result);

  const auto device_rowwise_entry_count = std::accumulate(
      device_rowwise_fragments_.begin(),
      device_rowwise_fragments_.end(),
      size_t(0),
      [](const size_t total, const DeviceRowwiseBufferFragment& fragment) {
        CHECK(fragment.buffer);
        CHECK(fragment.owner);
        return total + fragment.entry_count;
      });

  const bool has_device_columnar_fragments =
      std::any_of(device_columnar_fragments_.begin(),
                  device_columnar_fragments_.end(),
                  [](const auto& fragments) { return !fragments.empty(); });

  size_t device_columnar_entry_count{0};
  bool device_columnar_fragments_consistent = false;
  bool device_columnar_fragments_cover_result = false;
  if (rowwise_group_by_result && has_device_columnar_fragments) {
    bool initialized_entry_count = false;
    bool consistent_column_fragments = true;
    for (size_t column_idx = 0; column_idx < targets_.size(); ++column_idx) {
      if (column_idx >= device_columnar_fragments_.size() ||
          device_columnar_fragments_[column_idx].empty()) {
        consistent_column_fragments = false;
        break;
      }
      const auto logical_ti = get_logical_type_info(getColType(column_idx));
      if (logical_ti.is_varlen() || logical_ti.get_size() <= 0) {
        consistent_column_fragments = false;
        break;
      }
      const auto column_entry_count = std::accumulate(
          device_columnar_fragments_[column_idx].begin(),
          device_columnar_fragments_[column_idx].end(),
          size_t(0),
          [](const size_t total, const DeviceColumnarBufferFragment& fragment) {
            CHECK(fragment.buffer);
            CHECK(fragment.owner);
            return total + fragment.entry_count;
          });
      if (!initialized_entry_count) {
        device_columnar_entry_count = column_entry_count;
        initialized_entry_count = true;
      } else if (column_entry_count != device_columnar_entry_count) {
        consistent_column_fragments = false;
        break;
      }
    }
    device_columnar_fragments_consistent =
        consistent_column_fragments && initialized_entry_count;
    const int64_t cached_row_count = cached_row_count_;
    const auto logical_row_count = cached_row_count == uninitialized_cached_row_count
                                       ? query_mem_desc_.getEntryCount()
                                       : static_cast<size_t>(cached_row_count);
    device_columnar_fragments_cover_result =
        device_columnar_fragments_consistent &&
        device_columnar_entry_count == logical_row_count;
  }

  const auto storage_entry_count = [&]() {
    return std::accumulate(
        appended_storage_.begin(),
        appended_storage_.end(),
        storage_ ? storage_->query_mem_desc_.getEntryCount() : size_t(0),
        [](const size_t total, const std::unique_ptr<ResultSetStorage>& storage) {
          return total + (storage ? storage->query_mem_desc_.getEntryCount() : 0);
        });
  };

  const auto resize_storage_entry_counts_to = [&](const size_t target_entry_count) {
    CHECK(storage_);
    const auto current_storage_entry_count = storage_entry_count();
    if (target_entry_count > current_storage_entry_count) {
      const auto extra_entries = target_entry_count - current_storage_entry_count;
      if (!appended_storage_.empty()) {
        auto& target_storage = appended_storage_.back();
        target_storage->updateEntryCount(target_storage->query_mem_desc_.getEntryCount() +
                                         extra_entries);
      } else {
        storage_->updateEntryCount(storage_->query_mem_desc_.getEntryCount() +
                                   extra_entries);
      }
      return;
    }

    size_t remaining_entries = target_entry_count;
    const auto primary_entries = storage_->query_mem_desc_.getEntryCount();
    const auto new_primary_entries = std::min(primary_entries, remaining_entries);
    if (new_primary_entries != primary_entries) {
      storage_->updateEntryCount(new_primary_entries);
    }
    remaining_entries -= new_primary_entries;
    for (auto& appended_storage : appended_storage_) {
      CHECK(appended_storage);
      const auto appended_entries = appended_storage->query_mem_desc_.getEntryCount();
      const auto new_appended_entries = std::min(appended_entries, remaining_entries);
      if (new_appended_entries != appended_entries) {
        appended_storage->updateEntryCount(new_appended_entries);
      }
      remaining_entries -= new_appended_entries;
    }
    CHECK_EQ(remaining_entries, size_t(0));
    while (!appended_storage_.empty() &&
           appended_storage_.back()->query_mem_desc_.getEntryCount() == size_t(0)) {
      appended_storage_.pop_back();
    }
  };

  if ((device_columnar_fragments_cover_cpu_baseline_boundary_rows_ ||
       device_columnar_fragments_form_dense_cpu_rows_) &&
      device_columnar_fragments_cover_result && device_columnar_entry_count > size_t(0)) {
    query_mem_desc_.setEntryCount(device_columnar_entry_count);
    resize_storage_entry_counts_to(device_columnar_entry_count);
    device_columnar_fragments_cover_result = true;
  }
  const auto appended_entry_count = std::accumulate(
      appended_storage_.begin(),
      appended_storage_.end(),
      size_t(0),
      [](const size_t total, const std::unique_ptr<ResultSetStorage>& storage) {
        return total + (storage ? storage->query_mem_desc_.getEntryCount() : 0);
      });
  if (!appended_storage_.empty()) {
    CHECK_GE(query_mem_desc_.getEntryCount(), appended_entry_count);
    const auto base_entry_count = query_mem_desc_.getEntryCount() - appended_entry_count;
    if (storage_->query_mem_desc_.getEntryCount() != base_entry_count) {
      storage_->updateEntryCount(base_entry_count);
    }
  } else if (appended_storage_.empty() && storage_->query_mem_desc_.getEntryCount() !=
                                              query_mem_desc_.getEntryCount()) {
    storage_->updateEntryCount(query_mem_desc_.getEntryCount());
  }
  std::vector<ResultSetStorage*> storages;
  const auto refresh_storages = [&]() {
    storages.clear();
    storages.reserve(appended_storage_.size() + 1);
    storages.push_back(storage_.get());
    for (const auto& appended_storage : appended_storage_) {
      storages.push_back(appended_storage.get());
    }
  };
  refresh_storages();

  const auto ensure_storage_capacity = [&](const size_t storage_idx) {
    CHECK_LT(storage_idx, appended_storage_.size() + 1);
    auto* storage =
        storage_idx == 0 ? storage_.get() : appended_storage_[storage_idx - 1].get();
    CHECK(storage);
    const auto required_storage_bytes =
        storage->query_mem_desc_.getBufferSizeBytes(device_type_);
    CHECK_GT(required_storage_bytes, size_t(0));
    if (storage_idx == 0) {
      if (storage_buffer_size_bytes_ >= required_storage_bytes) {
        return;
      }
      const auto storage_query_mem_desc = storage_->query_mem_desc_;
      const auto target_init_vals = storage_->target_init_vals_;
      auto varlen_output_info = storage_->varlen_output_info_;
      auto* new_buffer =
          row_set_mem_owner_->allocate(required_storage_bytes, /*thread_idx=*/0);
      storage_.reset(
          new ResultSetStorage(targets_, storage_query_mem_desc, new_buffer, true));
      storage_->target_init_vals_ = target_init_vals;
      storage_->varlen_output_info_ = varlen_output_info;
      storage_buffer_size_bytes_ = required_storage_bytes;
      return;
    }

    auto& appended_storage = appended_storage_[storage_idx - 1];
    const auto storage_query_mem_desc = appended_storage->query_mem_desc_;
    const auto target_init_vals = appended_storage->target_init_vals_;
    auto varlen_output_info = appended_storage->varlen_output_info_;
    auto* new_buffer =
        row_set_mem_owner_->allocate(required_storage_bytes, /*thread_idx=*/0);
    appended_storage.reset(
        new ResultSetStorage(targets_, storage_query_mem_desc, new_buffer, true));
    appended_storage->target_init_vals_ = target_init_vals;
    appended_storage->varlen_output_info_ = varlen_output_info;
  };

  const auto ensure_materialized_storage_capacity =
      [&](const size_t materialized_row_count) {
        size_t storage_start_row{0};
        for (size_t storage_idx = 0; storage_idx < storages.size(); ++storage_idx) {
          auto* storage = storages[storage_idx];
          CHECK(storage);
          const auto storage_entry_count = storage->query_mem_desc_.getEntryCount();
          const auto storage_end_row = storage_start_row + storage_entry_count;
          if (materialized_row_count <= storage_start_row) {
            break;
          }
          ensure_storage_capacity(storage_idx);
          if (materialized_row_count < storage_end_row) {
            break;
          }
          storage_start_row = storage_end_row;
        }
        refresh_storages();
      };

  size_t row_count{0};
  if (columnar_projection_result) {
    CHECK(!device_columnar_fragments_.empty());
    for (auto* storage : storages) {
      CHECK(storage);
      const auto storage_entry_count = storage->query_mem_desc_.getEntryCount();
      row_count += storage_entry_count;
    }
    ensure_materialized_storage_capacity(row_count);
    for (auto* storage : storages) {
      CHECK(storage);
      auto* const row_index_buffer =
          reinterpret_cast<int64_t*>(storage->getUnderlyingBuffer());
      const auto storage_entry_count = storage->query_mem_desc_.getEntryCount();
      for (size_t entry_idx = 0; entry_idx < storage_entry_count; ++entry_idx) {
        row_index_buffer[entry_idx] = static_cast<int64_t>(entry_idx);
      }
    }
    for (size_t column_idx = 0; column_idx < targets_.size(); ++column_idx) {
      const auto logical_ti = get_logical_type_info(getColType(column_idx));
      const auto elem_size = logical_ti.get_size();
      CHECK_GT(elem_size, 0);
      CHECK(!logical_ti.is_varlen());
      CHECK_LT(column_idx, device_columnar_fragments_.size());
      const auto& fragments = device_columnar_fragments_[column_idx];
      size_t fragment_idx = 0;
      size_t fragment_offset_entries = 0;
      for (auto* storage : storages) {
        const auto storage_entry_count = storage->query_mem_desc_.getEntryCount();
        auto* const column_buffer = storage->getUnderlyingBuffer() +
                                    storage->query_mem_desc_.getColOffInBytes(column_idx);
        size_t copied_entries = 0;
        while (copied_entries < storage_entry_count) {
          CHECK_LT(fragment_idx, fragments.size());
          const auto& fragment = fragments[fragment_idx];
          CHECK(fragment.buffer);
          CHECK(fragment.owner);
          CHECK_LT(fragment_offset_entries, fragment.entry_count);
          const auto fragment_entries_remaining =
              fragment.entry_count - fragment_offset_entries;
          const auto entries_to_copy =
              std::min(storage_entry_count - copied_entries, fragment_entries_remaining);
          CHECK_LE(entries_to_copy,
                   std::numeric_limits<size_t>::max() / static_cast<size_t>(elem_size));
          const auto fragment_bytes = entries_to_copy * static_cast<size_t>(elem_size);
          fragment.owner->copyFromDevice(
              column_buffer + copied_entries * static_cast<size_t>(elem_size),
              fragment.buffer + fragment_offset_entries * static_cast<size_t>(elem_size),
              fragment_bytes,
              "Deferred GPU ResultSet column materialization");
          copied_entries += entries_to_copy;
          fragment_offset_entries += entries_to_copy;
          if (fragment_offset_entries == fragment.entry_count) {
            fragment_offset_entries = 0;
            ++fragment_idx;
          }
        }
      }
      CHECK_EQ(fragment_idx, fragments.size());
      CHECK_EQ(fragment_offset_entries, size_t(0));
    }
    device_columnar_cpu_storage_valid_.store(true, std::memory_order_release);
    return;
  }

  for (auto* storage : storages) {
    CHECK(storage);
    row_count += storage->query_mem_desc_.getEntryCount();
  }
  CHECK_EQ(row_count, entryCount());
  if (device_rowwise_entry_count == row_count) {
    ensure_materialized_storage_capacity(row_count);
    const auto row_size = query_mem_desc_.getRowSize();
    size_t fragment_idx{0};
    size_t fragment_offset_rows{0};
    for (auto* storage : storages) {
      CHECK(storage);
      CHECK_EQ(row_size, storage->query_mem_desc_.getRowSize());
      auto* const storage_buffer = storage->getUnderlyingBuffer();
      const auto storage_entry_count = storage->query_mem_desc_.getEntryCount();
      size_t copied_entries{0};
      while (copied_entries < storage_entry_count) {
        CHECK_LT(fragment_idx, device_rowwise_fragments_.size());
        const auto& fragment = device_rowwise_fragments_[fragment_idx];
        CHECK(fragment.buffer);
        CHECK(fragment.owner);
        CHECK_LT(fragment_offset_rows, fragment.entry_count);
        const auto fragment_remaining = fragment.entry_count - fragment_offset_rows;
        const auto entries_to_copy =
            std::min(storage_entry_count - copied_entries, fragment_remaining);
        const auto bytes_to_copy = entries_to_copy * row_size;
        fragment.owner->copyFromDevice(storage_buffer + copied_entries * row_size,
                                       fragment.buffer + fragment_offset_rows * row_size,
                                       bytes_to_copy,
                                       "Deferred GPU group-by row materialization");
        copied_entries += entries_to_copy;
        fragment_offset_rows += entries_to_copy;
        if (fragment_offset_rows == fragment.entry_count) {
          fragment_offset_rows = 0;
          ++fragment_idx;
        }
      }
    }
    CHECK_EQ(fragment_idx, device_rowwise_fragments_.size());
  } else {
    CHECK(!device_columnar_fragments_.empty());
    CHECK(device_columnar_fragments_consistent);
    CHECK_GT(device_columnar_entry_count, size_t(0));
    CHECK_LE(device_columnar_entry_count, row_count);
    if (!device_columnar_fragments_cover_result) {
      CHECK(!appended_storage_.empty());
      CHECK(device_columnar_fragments_exclude_baseline_boundary_keys_);
    }
    const auto expected_columnar_row_count = device_columnar_entry_count;
    ensure_materialized_storage_capacity(expected_columnar_row_count);
    for (size_t column_idx = 0; column_idx < targets_.size(); ++column_idx) {
      const auto logical_ti = get_logical_type_info(getColType(column_idx));
      const auto elem_size = static_cast<size_t>(logical_ti.get_size());
      CHECK_GT(elem_size, size_t(0));
      CHECK_LT(column_idx, device_columnar_fragments_.size());
      const auto& fragments = device_columnar_fragments_[column_idx];
      CHECK(!fragments.empty());
      size_t global_row_idx{0};
      for (const auto& fragment : fragments) {
        CHECK(fragment.buffer);
        CHECK(fragment.owner);
        std::vector<int8_t> host_column(fragment.entry_count * elem_size);
        fragment.owner->copyFromDevice(host_column.data(),
                                       fragment.buffer,
                                       host_column.size(),
                                       "Deferred GPU group-by column materialization");
        for (size_t fragment_row_idx = 0; fragment_row_idx < fragment.entry_count;
             ++fragment_row_idx, ++global_row_idx) {
          size_t storage_base_row{0};
          ResultSetStorage* destination_storage{nullptr};
          size_t destination_entry_idx{0};
          for (auto* storage : storages) {
            const auto storage_entry_count = storage->query_mem_desc_.getEntryCount();
            if (global_row_idx < storage_base_row + storage_entry_count) {
              destination_storage = storage;
              destination_entry_idx = global_row_idx - storage_base_row;
              break;
            }
            storage_base_row += storage_entry_count;
          }
          CHECK(destination_storage);
          auto* const destination = const_cast<int8_t*>(get_entry_target_ptr(
              *destination_storage, destination_entry_idx, column_idx));
          const auto destination_width = static_cast<size_t>(
              get_entry_target_width(*destination_storage, column_idx));
          copy_widened_fixed_width_value(
              destination,
              destination_width,
              host_column.data() + fragment_row_idx * elem_size,
              elem_size,
              logical_ti);
        }
      }
      CHECK_EQ(global_row_idx, expected_columnar_row_count);
    }
  }
  device_columnar_cpu_storage_valid_.store(true, std::memory_order_release);
}

bool ResultSet::getDeviceColumnarBufferFragments(
    const size_t column_idx,
    const size_t elem_size,
    std::vector<DeviceColumnarBufferFragment>& fragments) const {
  fragments.clear();
  if (column_idx >= targets_.size()) {
    return false;
  }
  if (column_idx >= device_columnar_fragments_.size() ||
      device_columnar_fragments_[column_idx].empty()) {
    return false;
  }
  if (elem_size == 0) {
    return false;
  }
  if (isTruncated()) {
    return false;
  }
  if (!permutation_.empty()) {
    return false;
  }
  const auto query_type = query_mem_desc_.getQueryDescriptionType();
  const bool can_expose_group_by_device_fragments =
      query_type == QueryDescriptionType::GroupByPerfectHash ||
      query_type == QueryDescriptionType::GroupByBaselineHash;
  const bool can_expose_compact_rowwise_projection =
      query_type == QueryDescriptionType::Projection &&
      device_columnar_fragments_cover_logical_rows_;
  if (!query_mem_desc_.didOutputColumnar() && !can_expose_group_by_device_fragments &&
      !can_expose_compact_rowwise_projection) {
    return false;
  }
  if (query_type != QueryDescriptionType::Projection &&
      query_type != QueryDescriptionType::TableFunction &&
      !can_expose_group_by_device_fragments) {
    return false;
  }
  if (!lazy_fetch_info_.empty()) {
    CHECK_LT(column_idx, lazy_fetch_info_.size());
    if (lazy_fetch_info_[column_idx].is_lazily_fetched) {
      return false;
    }
  }
  const auto result_row_count = rowCount();
  if (can_expose_group_by_device_fragments) {
    if (!device_columnar_fragments_cover_logical_rows_) {
      return false;
    }
    fragments = device_columnar_fragments_[column_idx];
    const auto fragment_entry_count = std::accumulate(
        fragments.begin(),
        fragments.end(),
        size_t(0),
        [](const size_t total, const DeviceColumnarBufferFragment& fragment) {
          CHECK(fragment.buffer);
          CHECK(fragment.owner);
          return total + fragment.entry_count;
        });
    if (fragment_entry_count != result_row_count) {
      fragments.clear();
      return false;
    }
    return true;
  }
  if (result_row_count != entryCount()) {
    return false;
  }
  const auto& slots = query_mem_desc_.getColSlotContext().getSlotsForCol(column_idx);
  if (slots.size() != 1) {
    return false;
  }
  const auto slot_idx = slots.front();
  if (query_mem_desc_.checkSlotUsesFlatBufferFormat(slot_idx)) {
    return false;
  }
  const auto padded_slot_width = query_mem_desc_.getPaddedSlotWidthBytes(slot_idx);
  if (padded_slot_width <= 0 || static_cast<size_t>(padded_slot_width) != elem_size) {
    return false;
  }
  fragments = device_columnar_fragments_[column_idx];
  const auto fragment_entry_count = std::accumulate(
      fragments.begin(),
      fragments.end(),
      size_t(0),
      [](const size_t total, const DeviceColumnarBufferFragment& fragment) {
        CHECK(fragment.buffer);
        CHECK(fragment.owner);
        return total + fragment.entry_count;
      });
  if (fragment_entry_count != result_row_count) {
    fragments.clear();
    return false;
  }
  return true;
}

bool ResultSet::getDeviceColumnarFragmentInfo(
    std::vector<DeviceColumnarFragmentInfo>& fragment_info) const {
  fragment_info.clear();
  if (colCount() == 0 || isTruncated() || !permutation_.empty()) {
    return false;
  }
  const auto query_type = query_mem_desc_.getQueryDescriptionType();
  const bool can_expose_group_by_device_fragments =
      query_type == QueryDescriptionType::GroupByPerfectHash ||
      query_type == QueryDescriptionType::GroupByBaselineHash;
  if (!query_mem_desc_.didOutputColumnar() && !can_expose_group_by_device_fragments) {
    return false;
  }
  if (query_type != QueryDescriptionType::Projection &&
      query_type != QueryDescriptionType::TableFunction &&
      !can_expose_group_by_device_fragments) {
    return false;
  }
  if (can_expose_group_by_device_fragments && query_mem_desc_.hasKeylessHash() &&
      device_columnar_cpu_storage_valid_.load(std::memory_order_acquire) &&
      entryCount() <= size_t(4096) && rowCountImpl(true) != entryCount()) {
    return false;
  }

  for (size_t column_idx = 0; column_idx < colCount(); ++column_idx) {
    const auto logical_ti = get_logical_type_info(getColType(column_idx));
    const auto elem_size = logical_ti.get_size();
    if (elem_size <= 0 || logical_ti.is_varlen()) {
      fragment_info.clear();
      return false;
    }

    if (!lazy_fetch_info_.empty()) {
      CHECK_LT(column_idx, lazy_fetch_info_.size());
      if (lazy_fetch_info_[column_idx].is_lazily_fetched) {
        fragment_info.clear();
        return false;
      }
    }

    if (column_idx >= device_columnar_fragments_.size() ||
        device_columnar_fragments_[column_idx].empty()) {
      fragment_info.clear();
      return false;
    }
    const auto& column_fragments = device_columnar_fragments_[column_idx];

    if (!can_expose_group_by_device_fragments) {
      const auto& slots = query_mem_desc_.getColSlotContext().getSlotsForCol(column_idx);
      if (slots.size() != 1) {
        fragment_info.clear();
        return false;
      }
      const auto slot_idx = slots.front();
      if (query_mem_desc_.checkSlotUsesFlatBufferFormat(slot_idx)) {
        fragment_info.clear();
        return false;
      }
      const auto padded_slot_width = query_mem_desc_.getPaddedSlotWidthBytes(slot_idx);
      if (padded_slot_width <= 0 ||
          static_cast<size_t>(padded_slot_width) != static_cast<size_t>(elem_size)) {
        fragment_info.clear();
        return false;
      }
    }

    if (column_idx == 0) {
      fragment_info.reserve(column_fragments.size());
      for (const auto& fragment : column_fragments) {
        CHECK(fragment.buffer);
        CHECK(fragment.owner);
        fragment_info.push_back(
            DeviceColumnarFragmentInfo{fragment.entry_count, fragment.device_id});
      }
    } else {
      if (column_fragments.size() != fragment_info.size()) {
        fragment_info.clear();
        return false;
      }
      for (size_t fragment_idx = 0; fragment_idx < column_fragments.size();
           ++fragment_idx) {
        const auto& fragment = column_fragments[fragment_idx];
        if (fragment.entry_count != fragment_info[fragment_idx].entry_count ||
            fragment.device_id != fragment_info[fragment_idx].device_id) {
          fragment_info.clear();
          return false;
        }
      }
    }
  }
  if (can_expose_group_by_device_fragments &&
      !device_columnar_fragments_cover_logical_rows_) {
    fragment_info.clear();
    return false;
  }
  size_t fragment_entry_count{0};
  for (const auto& fragment : fragment_info) {
    if (fragment.entry_count >
        std::numeric_limits<size_t>::max() - fragment_entry_count) {
      fragment_info.clear();
      return false;
    }
    fragment_entry_count += fragment.entry_count;
  }
  const auto result_row_count =
      can_expose_group_by_device_fragments ? fragment_entry_count : rowCount();
  if (can_expose_group_by_device_fragments) {
    setCachedRowCount(result_row_count);
  }
  if (fragment_entry_count != result_row_count ||
      (!can_expose_group_by_device_fragments && result_row_count != entryCount())) {
    fragment_info.clear();
    return false;
  }

  return !fragment_info.empty();
}

bool ResultSet::getColumnarBufferFragments(
    size_t column_idx,
    size_t elem_size,
    std::vector<ColumnarBufferFragment>& fragments) const {
  materializeDeviceColumnarCpuStorageIfNeeded();
  fragments.clear();
  if (!storage_ || elem_size == 0 || column_idx >= targets_.size() || isTruncated() ||
      !permutation_.empty() || !query_mem_desc_.didOutputColumnar()) {
    return false;
  }
  const auto query_type = query_mem_desc_.getQueryDescriptionType();
  if (query_type != QueryDescriptionType::Projection &&
      query_type != QueryDescriptionType::TableFunction) {
    return false;
  }
  if (!lazy_fetch_info_.empty()) {
    CHECK_LT(column_idx, lazy_fetch_info_.size());
    if (lazy_fetch_info_[column_idx].is_lazily_fetched) {
      return false;
    }
  }
  if (rowCount() != entryCount()) {
    return false;
  }

  const auto append_storage_fragment = [&](const ResultSetStorage* storage) {
    if (!storage) {
      return true;
    }
    const auto& qmd = storage->getQueryMemDesc();
    if (!qmd.didOutputColumnar() || qmd.getQueryDescriptionType() != query_type ||
        column_idx >= qmd.getColCount()) {
      return false;
    }
    const auto& slots = qmd.getColSlotContext().getSlotsForCol(column_idx);
    if (slots.size() != 1) {
      return false;
    }
    const auto slot_idx = slots.front();
    if (qmd.checkSlotUsesFlatBufferFormat(slot_idx)) {
      return false;
    }
    const auto padded_slot_width = qmd.getPaddedSlotWidthBytes(slot_idx);
    if (padded_slot_width <= 0 || static_cast<size_t>(padded_slot_width) != elem_size) {
      return false;
    }
    const auto entry_count = qmd.getEntryCount();
    if (entry_count == 0) {
      return true;
    }
    fragments.emplace_back(
        storage->getUnderlyingBuffer() + qmd.getColOffInBytes(slot_idx), entry_count);
    return true;
  };

  if (!append_storage_fragment(storage_.get())) {
    fragments.clear();
    return false;
  }
  for (const auto& storage : appended_storage_) {
    if (!append_storage_fragment(storage.get())) {
      fragments.clear();
      return false;
    }
  }
  return true;
}

bool ResultSet::getColumnarFragmentRowCounts(std::vector<size_t>& row_counts) const {
  row_counts.clear();
  if (colCount() == 0) {
    return false;
  }

  for (size_t column_idx = 0; column_idx < colCount(); ++column_idx) {
    const auto logical_ti = get_logical_type_info(getColType(column_idx));
    const auto elem_size = logical_ti.get_size();
    if (elem_size <= 0 || logical_ti.is_varlen()) {
      row_counts.clear();
      return false;
    }

    std::vector<ColumnarBufferFragment> column_fragments;
    if (!getColumnarBufferFragments(
            column_idx, static_cast<size_t>(elem_size), column_fragments) ||
        column_fragments.empty()) {
      row_counts.clear();
      return false;
    }

    if (column_idx == 0) {
      row_counts.reserve(column_fragments.size());
      for (const auto& [buffer, entry_count] : column_fragments) {
        CHECK(buffer);
        row_counts.push_back(entry_count);
      }
    } else {
      if (column_fragments.size() != row_counts.size()) {
        row_counts.clear();
        return false;
      }
      for (size_t fragment_idx = 0; fragment_idx < column_fragments.size();
           ++fragment_idx) {
        if (column_fragments[fragment_idx].second != row_counts[fragment_idx]) {
          row_counts.clear();
          return false;
        }
      }
    }
  }

  return !row_counts.empty();
}

// returns a bitmap (and total number) of all single slot targets
std::tuple<std::vector<bool>, size_t> ResultSet::getSingleSlotTargetBitmap() const {
  std::vector<bool> target_bitmap(targets_.size(), true);
  size_t num_single_slot_targets = 0;
  for (size_t target_idx = 0; target_idx < targets_.size(); target_idx++) {
    const auto& sql_type = targets_[target_idx].sql_type;
    if (targets_[target_idx].is_agg && targets_[target_idx].agg_kind == kAVG) {
      target_bitmap[target_idx] = false;
    } else if (sql_type.is_varlen()) {
      target_bitmap[target_idx] = false;
    } else {
      num_single_slot_targets++;
    }
  }
  return std::make_tuple(std::move(target_bitmap), num_single_slot_targets);
}

/**
 * This function returns a bitmap and population count of it, where it denotes
 * all supported single-column targets suitable for direct columnarization.
 *
 * The final goal is to remove the need for such selection, but at the moment for any
 * target that doesn't qualify for direct columnarization, we use the traditional
 * result set's iteration to handle it (e.g., count distinct, approximate count distinct)
 */
std::tuple<std::vector<bool>, size_t> ResultSet::getSupportedSingleSlotTargetBitmap()
    const {
  CHECK(isDirectColumnarConversionPossible());
  auto [single_slot_targets, num_single_slot_targets] = getSingleSlotTargetBitmap();
  const auto slot_indices = getSlotIndicesForTargetIndices();

  for (size_t target_idx = 0; target_idx < single_slot_targets.size(); target_idx++) {
    const auto& target = targets_[target_idx];
    if (single_slot_targets[target_idx] &&
        (is_distinct_target(target) ||
         shared::is_any<kAPPROX_QUANTILE, kMODE>(target.agg_kind) ||
         (target.is_agg && target.agg_kind == kSAMPLE && target.sql_type == kFLOAT))) {
      single_slot_targets[target_idx] = false;
      num_single_slot_targets--;
    }
    if (!single_slot_targets[target_idx]) {
      continue;
    }
    const auto slot_width =
        query_mem_desc_.getPaddedSlotWidthBytes(slot_indices[target_idx]);
    const bool is_supported_scalar_width =
        slot_width == 1 || slot_width == 2 || slot_width == 4 || slot_width == 8;
    const bool is_groupby_key = target_idx < query_mem_desc_.targetGroupbyIndicesSize() &&
                                query_mem_desc_.getTargetGroupbyIndex(target_idx) >= 0;
    const bool is_direct_groupby_key =
        is_groupby_key && (query_mem_desc_.getQueryDescriptionType() ==
                               QueryDescriptionType::GroupByBaselineHash ||
                           query_mem_desc_.getQueryDescriptionType() ==
                               QueryDescriptionType::GroupByPerfectHash);
    if (!is_supported_scalar_width && !is_direct_groupby_key) {
      single_slot_targets[target_idx] = false;
      num_single_slot_targets--;
    }
  }
  CHECK_GE(num_single_slot_targets, size_t(0));
  return std::make_tuple(std::move(single_slot_targets), num_single_slot_targets);
}

// returns the starting slot index for all targets in the result set
std::vector<size_t> ResultSet::getSlotIndicesForTargetIndices() const {
  std::vector<size_t> slot_indices(targets_.size(), 0);
  const auto [single_slot_targets, _] = getSingleSlotTargetBitmap();
  size_t slot_index = 0;
  const bool separate_varlen_storage =
      query_mem_desc_.hasVarlenOutput() || separate_varlen_storage_valid_;
  for (size_t target_idx = 0; target_idx < targets_.size(); target_idx++) {
    slot_indices[target_idx] =
        single_slot_targets[target_idx]
            ? static_cast<size_t>(
                  query_mem_desc_.getSlotIndexForSingleSlotCol(target_idx))
            : slot_index;
    slot_index = advance_slot(slot_index, targets_[target_idx], separate_varlen_storage);
  }
  return slot_indices;
}

void ResultSet::setCudaAllocator(const Executor* executor, int device_id) {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  CHECK_EQ(device_id, device_id_);
  cuda_allocator_ = executor->getCudaAllocatorShared(device_id);
  CHECK(cuda_allocator_);
}

void ResultSet::setCudaAllocator(std::shared_ptr<CudaAllocator> cuda_allocator) {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  CHECK(cuda_allocator);
  CHECK_EQ(cuda_allocator->getDeviceId(), device_id_);
  cuda_allocator_ = std::move(cuda_allocator);
}

CudaAllocator* ResultSet::getCudaAllocator() const {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  CHECK(cuda_allocator_);
  return cuda_allocator_.get();
}

bool ResultSet::hasDeviceBufferOwnership() const {
  if (cuda_allocator_ || !device_rowwise_fragments_.empty()) {
    return true;
  }
  return std::any_of(device_columnar_fragments_.begin(),
                     device_columnar_fragments_.end(),
                     [](const auto& fragments) { return !fragments.empty(); });
}

void ResultSet::setCudaStream(const Executor* executor, int device_id) {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  cuda_stream_ = executor->getCudaStream(device_id_);
}

CUstream ResultSet::getCudaStream() const {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  return cuda_stream_;
}

void ResultSet::initMaterializedSortBuffers(
    const std::list<Analyzer::OrderEntry>& order_entries,
    bool single_threaded,
    std::optional<size_t> compact_permutation_size,
    std::optional<MaterializedSortBuffersBase::TopNDictionarySortContext>
        top_n_dictionary_sort_context) {
  // This method is not thread safe
  if (!materialized_sort_buffers_) {
    if (query_mem_desc_.didOutputColumnar()) {
      materialized_sort_buffers_ =
          std::make_unique<MaterializedSortBuffers<ColumnWiseTargetAccessor>>(
              this,
              order_entries,
              single_threaded,
              compact_permutation_size,
              top_n_dictionary_sort_context);
    } else {
      materialized_sort_buffers_ =
          std::make_unique<MaterializedSortBuffers<RowWiseTargetAccessor>>(
              this,
              order_entries,
              single_threaded,
              compact_permutation_size,
              top_n_dictionary_sort_context);
    }
  }
}

// namespace result_set

bool result_set::can_use_parallel_algorithms(const ResultSet& rows) {
  return !rows.isTruncated();
}

namespace {
struct IsDictEncodedStr {
  bool operator()(TargetInfo const& target_info) const {
    return target_info.sql_type.is_dict_encoded_string();
  }
};
}  // namespace

std::optional<size_t> result_set::first_dict_encoded_idx(
    std::vector<TargetInfo> const& targets) {
  auto const itr = std::find_if(targets.begin(), targets.end(), IsDictEncodedStr{});
  return itr == targets.end() ? std::nullopt
                              : std::make_optional<size_t>(itr - targets.begin());
}

bool result_set::use_parallel_algorithms(const ResultSet& rows) {
  return result_set::can_use_parallel_algorithms(rows) &&
         rows.entryCount() >= auto_parallel_row_count_threshold;
}
