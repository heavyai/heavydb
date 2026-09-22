/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/Allocators/CudaAllocator.h"
#include "Logger/Logger.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/JoinHashTable/HashTable.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

class RankedBitmapHashTable : public HashTable {
 public:
  static constexpr size_t kRankBlockWordCount = 8;
  static constexpr size_t kHeaderWordCount = 4;
  static constexpr size_t kHeaderBitmapPtr = 0;
  static constexpr size_t kHeaderRankBlocksPtr = 1;
  static constexpr size_t kHeaderPayloadPtrsPtr = 2;
  static constexpr size_t kHeaderPayloadChunkWords = 3;
  static constexpr size_t kHeaderCountsPtr = 4;
  static constexpr size_t kHeaderOffsetsPtr = 5;
  static constexpr size_t kOneToManyHeaderWordCount = 6;

  RankedBitmapHashTable(const ExecutorDeviceType device_type,
                        const size_t bit_count,
                        const size_t payload_count,
                        const size_t max_slab_size,
                        Data_Namespace::DataMgr* data_mgr = nullptr,
                        const int device_id = -1,
                        const HashType layout = HashType::OneToOne,
                        const size_t distinct_count = 0,
                        const bool force_segmented_layout = false,
                        const bool payload_free = false)
      : bit_count_(bit_count)
      , payload_count_(payload_free ? size_t(0) : payload_count)
      , distinct_count_(layout == HashType::OneToMany ? distinct_count : 0)
      , bitmap_word_count_(wordsForBits(bit_count))
      , rank_block_count_(blocksForWords(bitmap_word_count_))
      , total_word_count_(computeTotalWordCount(layout,
                                                bitmap_word_count_,
                                                rank_block_count_,
                                                payload_count_,
                                                distinct_count_))
      , allocated_bytes_(checkedMultiply(total_word_count_,
                                         sizeof(uint32_t),
                                         "Ranked bitmap allocation size overflow"))
      , segmented_layout_(force_segmented_layout || layout == HashType::OneToMany ||
                          allocated_bytes_ > max_slab_size)
      , layout_(layout)
      , payload_free_(payload_free)
      , data_mgr_(data_mgr)
      , device_id_(device_id) {
    CHECK(!payload_free_ || layout_ == HashType::OneToOne);
    if (device_type == ExecutorDeviceType::CPU) {
      allocateCpu(max_slab_size);
    } else {
#ifdef HAVE_CUDA
      CHECK(data_mgr_);
      CHECK_GE(device_id_, 0);
      allocateGpu(max_slab_size);
#else
      UNREACHABLE();
#endif
    }
  }

  ~RankedBitmapHashTable() override = default;

  size_t getHashTableBufferSize(const ExecutorDeviceType) const override {
    return allocated_bytes_;
  }

  int8_t* getCpuBuffer() override {
    return segmented_layout_ ? reinterpret_cast<int8_t*>(cpu_header_.data())
                             : reinterpret_cast<int8_t*>(cpu_buffer_.get());
  }

  int8_t* getGpuBuffer() const override {
#ifdef HAVE_CUDA
    if (segmented_layout_) {
      return gpu_header_buffer_ ? gpu_header_buffer_->getMemoryPtr() : nullptr;
    }
    return gpu_buffer_ ? gpu_buffer_->getMemoryPtr() : nullptr;
#else
    return nullptr;
#endif
  }

  HashType getLayout() const override {
    return layout_;
  }

  size_t getEntryCount() const override {
    return bit_count_;
  }

  size_t getEmittedKeysCount() const override {
    return payload_count_;
  }

  size_t getRowIdSize() const override {
    return sizeof(int32_t);
  }

  size_t getBitmapWordCount() const noexcept {
    return bitmap_word_count_;
  }

  size_t getRankBlockCount() const noexcept {
    return rank_block_count_;
  }

  size_t getPayloadCount() const noexcept {
    return payload_count_;
  }

  size_t getDistinctCount() const noexcept {
    return distinct_count_;
  }

  size_t getAllocatedBytes() const noexcept {
    return allocated_bytes_;
  }

  bool hasSegmentedLayout() const noexcept {
    return segmented_layout_;
  }

  bool isPayloadFree() const noexcept {
    return payload_free_;
  }

  size_t getPayloadChunkWordCount() const noexcept {
    return payload_chunk_word_count_;
  }

  size_t getPayloadChunkCount() const noexcept {
    return payload_chunk_count_;
  }

  size_t getIndexWordCount() const noexcept {
    return bitmap_word_count_ + rank_block_count_;
  }

  uint32_t* getCpuBitmap() {
    return segmented_layout_ ? cpu_index_buffer_.get() : cpu_buffer_.get();
  }

  uint32_t* getCpuRankBlocks() {
    return getCpuBitmap() + bitmap_word_count_;
  }

  uint32_t* getCpuPayload() {
    CHECK(!segmented_layout_);
    return cpu_buffer_.get() + bitmap_word_count_ + rank_block_count_;
  }

  uint32_t* getCpuCounts() {
    CHECK(layout_ == HashType::OneToMany);
    CHECK(cpu_counts_);
    return cpu_counts_.get();
  }

  uint32_t* getCpuOffsets() {
    CHECK(layout_ == HashType::OneToMany);
    CHECK(cpu_offsets_);
    return cpu_offsets_.get();
  }

  uint32_t* getCpuPayloadChunk(const size_t chunk_idx) {
    CHECK(segmented_layout_);
    CHECK_LT(chunk_idx, cpu_payload_chunks_.size());
    return cpu_payload_chunks_[chunk_idx].get();
  }

  uint64_t* getCpuPayloadPtrs() {
    CHECK(segmented_layout_);
    return cpu_payload_ptrs_.data();
  }

  size_t getPayloadChunkWordCount(const size_t chunk_idx) const {
    CHECK_LT(chunk_idx, payload_chunk_count_);
    const size_t chunk_start = chunk_idx * payload_chunk_word_count_;
    return std::min(payload_chunk_word_count_, payload_count_ - chunk_start);
  }

#ifdef HAVE_CUDA
  uint32_t* getGpuBitmap() const {
    CHECK(segmented_layout_);
    return reinterpret_cast<uint32_t*>(gpu_index_buffer_->getMemoryPtr());
  }

  uint32_t* getGpuRankBlocks() const {
    return getGpuBitmap() + bitmap_word_count_;
  }

  uint32_t* getGpuPayloadChunk(const size_t chunk_idx) const {
    CHECK(segmented_layout_);
    CHECK_LT(chunk_idx, gpu_payload_buffers_.size());
    return reinterpret_cast<uint32_t*>(gpu_payload_buffers_[chunk_idx]->getMemoryPtr());
  }

  uint64_t* getGpuPayloadPtrs() const {
    CHECK(segmented_layout_);
    return gpu_payload_ptrs_buffer_
               ? reinterpret_cast<uint64_t*>(gpu_payload_ptrs_buffer_->getMemoryPtr())
               : nullptr;
  }

  uint32_t* getGpuCounts() const {
    CHECK(layout_ == HashType::OneToMany);
    CHECK(gpu_counts_buffer_);
    return reinterpret_cast<uint32_t*>(gpu_counts_buffer_->getMemoryPtr());
  }

  uint32_t* getGpuOffsets() const {
    CHECK(layout_ == HashType::OneToMany);
    CHECK(gpu_offsets_buffer_);
    return reinterpret_cast<uint32_t*>(gpu_offsets_buffer_->getMemoryPtr());
  }

  void copyGpuHeaderToDevice(DeviceAllocator* device_allocator) const {
    CHECK(segmented_layout_);
    CHECK(device_allocator);
    if (gpu_payload_ptrs_buffer_) {
      device_allocator->copyToDevice(
          gpu_payload_ptrs_buffer_->getMemoryPtr(),
          gpu_payload_ptrs_host_.data(),
          checkedMultiply(gpu_payload_ptrs_host_.size(),
                          sizeof(uint64_t),
                          "Ranked bitmap pointer table size overflow"),
          "ranked bitmap payload pointers");
    }
    const size_t header_word_count =
        layout_ == HashType::OneToMany ? kOneToManyHeaderWordCount : kHeaderWordCount;
    device_allocator->copyToDevice(gpu_header_buffer_->getMemoryPtr(),
                                   gpu_header_host_.data(),
                                   header_word_count * sizeof(uint64_t),
                                   "ranked bitmap header");
  }
#endif

  void resizeOneToOnePayloadBuffer(const size_t payload_count,
                                   const size_t max_slab_size) {
    CHECK(segmented_layout_);
    CHECK(layout_ == HashType::OneToOne);
    CHECK(!payload_free_);
    if (payload_count == payload_count_) {
      return;
    }
    const auto old_payload_count = payload_count_;
    const auto old_total_word_count = total_word_count_;
    const auto old_allocated_bytes = allocated_bytes_;
    try {
      payload_count_ = payload_count;
      recomputeOneToOneWordCounts();
      if (data_mgr_) {
#ifdef HAVE_CUDA
        resizeGpuPayloadBuffers(max_slab_size);
#else
        UNREACHABLE();
#endif
      } else {
        resizeCpuPayloadBuffers(max_slab_size);
      }
    } catch (...) {
      payload_count_ = old_payload_count;
      total_word_count_ = old_total_word_count;
      allocated_bytes_ = old_allocated_bytes;
      throw;
    }
  }

  void allocateOneToManyBuffers(const size_t distinct_count, const size_t max_slab_size) {
    CHECK(segmented_layout_);
    CHECK(layout_ == HashType::OneToOne);
    CHECK(!payload_free_);
    CHECK_EQ(size_t(0), distinct_count_);
    const auto old_layout = layout_;
    const auto old_distinct_count = distinct_count_;
    const auto old_total_word_count = total_word_count_;
    const auto old_allocated_bytes = allocated_bytes_;
    try {
      layout_ = HashType::OneToMany;
      distinct_count_ = distinct_count;
      recomputeOneToManyWordCounts();
      if (data_mgr_) {
#ifdef HAVE_CUDA
        allocateGpuOneToManyBuffers(max_slab_size);
#else
        UNREACHABLE();
#endif
      } else {
        allocateCpuOneToManyBuffers(max_slab_size);
      }
    } catch (...) {
      layout_ = old_layout;
      distinct_count_ = old_distinct_count;
      total_word_count_ = old_total_word_count;
      allocated_bytes_ = old_allocated_bytes;
      throw;
    }
  }

  void resizeOneToManyPayloadBuffer(const size_t payload_count,
                                    const size_t max_slab_size) {
    CHECK(segmented_layout_);
    CHECK(layout_ == HashType::OneToMany);
    if (payload_count == payload_count_) {
      return;
    }
    const auto old_payload_count = payload_count_;
    const auto old_total_word_count = total_word_count_;
    const auto old_allocated_bytes = allocated_bytes_;
    try {
      payload_count_ = payload_count;
      recomputeOneToManyWordCounts();
      if (data_mgr_) {
#ifdef HAVE_CUDA
        resizeGpuPayloadBuffers(max_slab_size);
#else
        UNREACHABLE();
#endif
      } else {
        resizeCpuPayloadBuffers(max_slab_size);
      }
    } catch (...) {
      payload_count_ = old_payload_count;
      total_word_count_ = old_total_word_count;
      allocated_bytes_ = old_allocated_bytes;
      throw;
    }
  }

  static size_t wordsForBits(const size_t bit_count) {
    return bit_count == 0 ? 0 : ((bit_count - 1) / 32 + 1);
  }

  static size_t blocksForWords(const size_t word_count) {
    return word_count == 0 ? 0 : ((word_count - 1) / kRankBlockWordCount + 1);
  }

 private:
  void checkSlabSize(const size_t bytes, const size_t max_slab_size) {
    if (bytes > max_slab_size) {
      throw JoinHashTableTooBig(bytes, max_slab_size);
    }
  }

  void recomputeOneToManyWordCounts() {
    CHECK(layout_ == HashType::OneToMany);
    total_word_count_ = computeTotalWordCount(
        layout_, bitmap_word_count_, rank_block_count_, payload_count_, distinct_count_);
    allocated_bytes_ = checkedMultiply(
        total_word_count_, sizeof(uint32_t), "Ranked bitmap allocation size overflow");
  }

  void recomputeOneToOneWordCounts() {
    CHECK(layout_ == HashType::OneToOne);
    CHECK(!payload_free_);
    total_word_count_ = computeTotalWordCount(
        layout_, bitmap_word_count_, rank_block_count_, payload_count_, distinct_count_);
    allocated_bytes_ = checkedMultiply(
        total_word_count_, sizeof(uint32_t), "Ranked bitmap allocation size overflow");
  }

  void resizeCpuPayloadBuffers(const size_t max_slab_size) {
    const auto new_payload_chunk_word_count =
        std::max<size_t>(size_t(1), max_slab_size / sizeof(uint32_t));
    const auto new_payload_chunk_count =
        payload_count_ == 0 ? 0
                            : ((payload_count_ - 1) / new_payload_chunk_word_count + 1);
    std::vector<std::unique_ptr<uint32_t[]>> new_payload_chunks;
    std::vector<uint64_t> new_payload_ptrs;
    new_payload_chunks.reserve(new_payload_chunk_count);
    new_payload_ptrs.reserve(new_payload_chunk_count);
    for (size_t chunk_idx = 0; chunk_idx < new_payload_chunk_count; ++chunk_idx) {
      const size_t chunk_start = chunk_idx * new_payload_chunk_word_count;
      const size_t chunk_words =
          std::min(new_payload_chunk_word_count, payload_count_ - chunk_start);
      new_payload_chunks.emplace_back(new uint32_t[chunk_words]());
      new_payload_ptrs.push_back(
          reinterpret_cast<uint64_t>(new_payload_chunks.back().get()));
    }
    cpu_payload_chunks_.swap(new_payload_chunks);
    cpu_payload_ptrs_.swap(new_payload_ptrs);
    payload_chunk_word_count_ = new_payload_chunk_word_count;
    payload_chunk_count_ = new_payload_chunk_count;
    updateCpuHeader();
  }

#ifdef HAVE_CUDA
  using GpuBufferPtr = std::shared_ptr<Data_Namespace::AbstractBuffer>;

  GpuBufferPtr allocateGpuBuffer(const size_t num_bytes) {
    CHECK(data_mgr_);
    auto* buffer =
        CudaAllocator::allocGpuAbstractBuffer(data_mgr_, num_bytes, device_id_);
    return GpuBufferPtr(buffer, [data_mgr = data_mgr_](auto* owned_buffer) noexcept {
      try {
        data_mgr->free(owned_buffer);
      } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to release ranked bitmap GPU buffer: " << e.what();
      } catch (...) {
        LOG(ERROR) << "Failed to release ranked bitmap GPU buffer";
      }
    });
  }

  void resizeGpuPayloadBuffers(const size_t max_slab_size) {
    CHECK(data_mgr_);
    const auto new_payload_chunk_word_count =
        std::max<size_t>(size_t(1), max_slab_size / sizeof(uint32_t));
    const auto new_payload_chunk_count =
        payload_count_ == 0 ? 0
                            : ((payload_count_ - 1) / new_payload_chunk_word_count + 1);
    std::vector<GpuBufferPtr> new_payload_buffers;
    std::vector<uint64_t> new_payload_ptrs_host;
    new_payload_buffers.reserve(new_payload_chunk_count);
    new_payload_ptrs_host.reserve(new_payload_chunk_count);
    for (size_t chunk_idx = 0; chunk_idx < new_payload_chunk_count; ++chunk_idx) {
      const size_t chunk_start = chunk_idx * new_payload_chunk_word_count;
      const size_t chunk_words =
          std::min(new_payload_chunk_word_count, payload_count_ - chunk_start);
      const size_t chunk_bytes = chunk_words * sizeof(uint32_t);
      checkSlabSize(chunk_bytes, max_slab_size);
      auto payload_buffer = allocateGpuBuffer(chunk_bytes);
      new_payload_ptrs_host.push_back(
          reinterpret_cast<uint64_t>(payload_buffer->getMemoryPtr()));
      new_payload_buffers.push_back(std::move(payload_buffer));
    }
    const size_t payload_ptrs_bytes =
        checkedMultiply(new_payload_ptrs_host.size(),
                        sizeof(uint64_t),
                        "Ranked bitmap pointer table size overflow");
    GpuBufferPtr new_payload_ptrs_buffer;
    if (payload_ptrs_bytes > 0) {
      new_payload_ptrs_buffer = allocateGpuBuffer(payload_ptrs_bytes);
    }
    gpu_payload_buffers_.swap(new_payload_buffers);
    gpu_payload_ptrs_host_.swap(new_payload_ptrs_host);
    gpu_payload_ptrs_buffer_.swap(new_payload_ptrs_buffer);
    payload_chunk_word_count_ = new_payload_chunk_word_count;
    payload_chunk_count_ = new_payload_chunk_count;
    updateGpuHeader();
  }
#endif

  void allocateCpu(const size_t max_slab_size) {
    if (layout_ == HashType::OneToMany) {
      const size_t index_words = bitmap_word_count_ + rank_block_count_;
      checkSlabSize(index_words * sizeof(uint32_t), max_slab_size);
      cpu_index_buffer_.reset(new uint32_t[index_words]());
      if (distinct_count_) {
        allocateCpuOneToManyBuffers(max_slab_size);
      }
      updateCpuHeader();
      return;
    }
    if (!segmented_layout_) {
      cpu_buffer_.reset(new uint32_t[total_word_count_]());
      return;
    }

    const size_t index_words = bitmap_word_count_ + rank_block_count_;
    checkSlabSize(index_words * sizeof(uint32_t), max_slab_size);
    cpu_index_buffer_.reset(new uint32_t[index_words]());
    payload_chunk_word_count_ =
        std::max<size_t>(size_t(1), max_slab_size / sizeof(uint32_t));
    payload_chunk_count_ =
        payload_count_ == 0 ? 0 : ((payload_count_ - 1) / payload_chunk_word_count_ + 1);
    cpu_payload_chunks_.reserve(payload_chunk_count_);
    cpu_payload_ptrs_.reserve(payload_chunk_count_);
    for (size_t chunk_idx = 0; chunk_idx < payload_chunk_count_; ++chunk_idx) {
      const size_t chunk_words = getPayloadChunkWordCount(chunk_idx);
      cpu_payload_chunks_.emplace_back(new uint32_t[chunk_words]());
      cpu_payload_ptrs_.push_back(
          reinterpret_cast<uint64_t>(cpu_payload_chunks_.back().get()));
    }
    cpu_header_[kHeaderBitmapPtr] = reinterpret_cast<uint64_t>(cpu_index_buffer_.get());
    cpu_header_[kHeaderRankBlocksPtr] =
        reinterpret_cast<uint64_t>(cpu_index_buffer_.get() + bitmap_word_count_);
    cpu_header_[kHeaderPayloadPtrsPtr] =
        cpu_payload_ptrs_.empty() ? 0
                                  : reinterpret_cast<uint64_t>(cpu_payload_ptrs_.data());
    cpu_header_[kHeaderPayloadChunkWords] = payload_chunk_word_count_;
  }

#ifdef HAVE_CUDA
  void allocateGpu(const size_t max_slab_size) {
    if (layout_ == HashType::OneToMany) {
      const size_t index_words = bitmap_word_count_ + rank_block_count_;
      const size_t index_bytes = index_words * sizeof(uint32_t);
      checkSlabSize(index_bytes, max_slab_size);
      gpu_index_buffer_ = allocateGpuBuffer(index_bytes);
      if (distinct_count_) {
        allocateGpuOneToManyBuffers(max_slab_size);
      }
      updateGpuHeader();
      gpu_header_buffer_ =
          allocateGpuBuffer(kOneToManyHeaderWordCount * sizeof(uint64_t));
      return;
    }
    if (!segmented_layout_) {
      checkSlabSize(allocated_bytes_, max_slab_size);
      gpu_buffer_ = allocateGpuBuffer(allocated_bytes_);
      return;
    }

    const size_t index_words = bitmap_word_count_ + rank_block_count_;
    const size_t index_bytes = index_words * sizeof(uint32_t);
    checkSlabSize(index_bytes, max_slab_size);
    gpu_index_buffer_ = allocateGpuBuffer(index_bytes);

    payload_chunk_word_count_ =
        std::max<size_t>(size_t(1), max_slab_size / sizeof(uint32_t));
    payload_chunk_count_ =
        payload_count_ == 0 ? 0 : ((payload_count_ - 1) / payload_chunk_word_count_ + 1);
    gpu_payload_buffers_.reserve(payload_chunk_count_);
    gpu_payload_ptrs_host_.clear();
    gpu_payload_ptrs_host_.reserve(payload_chunk_count_);
    for (size_t chunk_idx = 0; chunk_idx < payload_chunk_count_; ++chunk_idx) {
      const size_t chunk_words = getPayloadChunkWordCount(chunk_idx);
      const size_t chunk_bytes = chunk_words * sizeof(uint32_t);
      checkSlabSize(chunk_bytes, max_slab_size);
      auto payload_buffer = allocateGpuBuffer(chunk_bytes);
      gpu_payload_ptrs_host_.push_back(
          reinterpret_cast<uint64_t>(payload_buffer->getMemoryPtr()));
      gpu_payload_buffers_.push_back(payload_buffer);
    }

    const size_t payload_ptrs_bytes =
        checkedMultiply(gpu_payload_ptrs_host_.size(),
                        sizeof(uint64_t),
                        "Ranked bitmap pointer table size overflow");
    if (payload_ptrs_bytes > 0) {
      gpu_payload_ptrs_buffer_ = allocateGpuBuffer(payload_ptrs_bytes);
    }

    updateGpuHeader();
    gpu_header_buffer_ = allocateGpuBuffer(kOneToManyHeaderWordCount * sizeof(uint64_t));
  }
#endif

  void allocateCpuOneToManyBuffers(const size_t max_slab_size) {
    const size_t count_bytes = distinct_count_ * sizeof(uint32_t);
    checkSlabSize(count_bytes, max_slab_size);
    auto new_counts = std::make_unique<uint32_t[]>(distinct_count_);
    auto new_offsets = std::make_unique<uint32_t[]>(distinct_count_);

    std::vector<std::unique_ptr<uint32_t[]>> new_payload_chunks;
    std::vector<uint64_t> new_payload_ptrs;
    size_t new_payload_chunk_word_count = payload_chunk_word_count_;
    size_t new_payload_chunk_count = payload_chunk_count_;
    if (cpu_payload_chunks_.empty()) {
      new_payload_chunk_word_count =
          std::max<size_t>(size_t(1), max_slab_size / sizeof(uint32_t));
      new_payload_chunk_count =
          payload_count_ == 0 ? 0
                              : ((payload_count_ - 1) / new_payload_chunk_word_count + 1);
      new_payload_chunks.reserve(new_payload_chunk_count);
      new_payload_ptrs.reserve(new_payload_chunk_count);
      for (size_t chunk_idx = 0; chunk_idx < new_payload_chunk_count; ++chunk_idx) {
        const size_t chunk_start = chunk_idx * new_payload_chunk_word_count;
        const size_t chunk_words =
            std::min(new_payload_chunk_word_count, payload_count_ - chunk_start);
        new_payload_chunks.emplace_back(new uint32_t[chunk_words]());
        new_payload_ptrs.push_back(
            reinterpret_cast<uint64_t>(new_payload_chunks.back().get()));
      }
    }
    cpu_counts_ = std::move(new_counts);
    cpu_offsets_ = std::move(new_offsets);
    if (cpu_payload_chunks_.empty()) {
      cpu_payload_chunks_.swap(new_payload_chunks);
      cpu_payload_ptrs_.swap(new_payload_ptrs);
      payload_chunk_word_count_ = new_payload_chunk_word_count;
      payload_chunk_count_ = new_payload_chunk_count;
    }
    updateCpuHeader();
  }

  void updateCpuHeader() {
    if (!segmented_layout_) {
      return;
    }
    cpu_header_[kHeaderBitmapPtr] = reinterpret_cast<uint64_t>(cpu_index_buffer_.get());
    cpu_header_[kHeaderRankBlocksPtr] =
        reinterpret_cast<uint64_t>(cpu_index_buffer_.get() + bitmap_word_count_);
    cpu_header_[kHeaderPayloadPtrsPtr] =
        cpu_payload_ptrs_.empty() ? 0
                                  : reinterpret_cast<uint64_t>(cpu_payload_ptrs_.data());
    cpu_header_[kHeaderPayloadChunkWords] = payload_chunk_word_count_;
    if (layout_ == HashType::OneToMany) {
      cpu_header_[kHeaderCountsPtr] = reinterpret_cast<uint64_t>(cpu_counts_.get());
      cpu_header_[kHeaderOffsetsPtr] = reinterpret_cast<uint64_t>(cpu_offsets_.get());
    }
  }

#ifdef HAVE_CUDA
  void allocateGpuOneToManyBuffers(const size_t max_slab_size) {
    const size_t count_bytes = distinct_count_ * sizeof(uint32_t);
    checkSlabSize(count_bytes, max_slab_size);
    auto new_counts_buffer = allocateGpuBuffer(count_bytes);
    auto new_offsets_buffer = allocateGpuBuffer(count_bytes);

    std::vector<GpuBufferPtr> new_payload_buffers;
    std::vector<uint64_t> new_payload_ptrs_host;
    size_t new_payload_chunk_word_count = payload_chunk_word_count_;
    size_t new_payload_chunk_count = payload_chunk_count_;
    if (gpu_payload_buffers_.empty()) {
      new_payload_chunk_word_count =
          std::max<size_t>(size_t(1), max_slab_size / sizeof(uint32_t));
      new_payload_chunk_count =
          payload_count_ == 0 ? 0
                              : ((payload_count_ - 1) / new_payload_chunk_word_count + 1);
      new_payload_buffers.reserve(new_payload_chunk_count);
      new_payload_ptrs_host.reserve(new_payload_chunk_count);
      for (size_t chunk_idx = 0; chunk_idx < new_payload_chunk_count; ++chunk_idx) {
        const size_t chunk_start = chunk_idx * new_payload_chunk_word_count;
        const size_t chunk_words =
            std::min(new_payload_chunk_word_count, payload_count_ - chunk_start);
        const size_t chunk_bytes = chunk_words * sizeof(uint32_t);
        checkSlabSize(chunk_bytes, max_slab_size);
        auto payload_buffer = allocateGpuBuffer(chunk_bytes);
        new_payload_ptrs_host.push_back(
            reinterpret_cast<uint64_t>(payload_buffer->getMemoryPtr()));
        new_payload_buffers.push_back(std::move(payload_buffer));
      }
    }

    GpuBufferPtr new_payload_ptrs_buffer;
    if (!gpu_payload_ptrs_buffer_) {
      const auto& payload_ptrs_host =
          gpu_payload_buffers_.empty() ? new_payload_ptrs_host : gpu_payload_ptrs_host_;
      const size_t payload_ptrs_bytes =
          checkedMultiply(payload_ptrs_host.size(),
                          sizeof(uint64_t),
                          "Ranked bitmap pointer table size overflow");
      if (payload_ptrs_bytes > 0) {
        new_payload_ptrs_buffer = allocateGpuBuffer(payload_ptrs_bytes);
      }
    }
    gpu_counts_buffer_ = std::move(new_counts_buffer);
    gpu_offsets_buffer_ = std::move(new_offsets_buffer);
    if (gpu_payload_buffers_.empty()) {
      gpu_payload_buffers_.swap(new_payload_buffers);
      gpu_payload_ptrs_host_.swap(new_payload_ptrs_host);
      payload_chunk_word_count_ = new_payload_chunk_word_count;
      payload_chunk_count_ = new_payload_chunk_count;
    }
    if (new_payload_ptrs_buffer) {
      gpu_payload_ptrs_buffer_ = std::move(new_payload_ptrs_buffer);
    }
    updateGpuHeader();
  }

  void updateGpuHeader() {
    if (!segmented_layout_) {
      return;
    }
    gpu_header_host_[kHeaderBitmapPtr] =
        reinterpret_cast<uint64_t>(gpu_index_buffer_->getMemoryPtr());
    gpu_header_host_[kHeaderRankBlocksPtr] = reinterpret_cast<uint64_t>(
        gpu_index_buffer_->getMemoryPtr() + bitmap_word_count_ * sizeof(uint32_t));
    gpu_header_host_[kHeaderPayloadPtrsPtr] =
        gpu_payload_ptrs_buffer_
            ? reinterpret_cast<uint64_t>(gpu_payload_ptrs_buffer_->getMemoryPtr())
            : 0;
    gpu_header_host_[kHeaderPayloadChunkWords] = payload_chunk_word_count_;
    if (layout_ == HashType::OneToMany) {
      gpu_header_host_[kHeaderCountsPtr] =
          gpu_counts_buffer_
              ? reinterpret_cast<uint64_t>(gpu_counts_buffer_->getMemoryPtr())
              : 0;
      gpu_header_host_[kHeaderOffsetsPtr] =
          gpu_offsets_buffer_
              ? reinterpret_cast<uint64_t>(gpu_offsets_buffer_->getMemoryPtr())
              : 0;
    }
  }
#endif

  static size_t checkedAdd(const size_t lhs,
                           const size_t rhs,
                           const char* const description) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
      throw std::overflow_error(description);
    }
    return lhs + rhs;
  }

  static size_t checkedMultiply(const size_t lhs,
                                const size_t rhs,
                                const char* const description) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
      throw std::overflow_error(description);
    }
    return lhs * rhs;
  }

  static size_t computeTotalWordCount(const HashType layout,
                                      const size_t bitmap_word_count,
                                      const size_t rank_block_count,
                                      const size_t payload_count,
                                      const size_t distinct_count) {
    auto total = checkedAdd(
        bitmap_word_count, rank_block_count, "Ranked bitmap index size overflow");
    if (layout == HashType::OneToOne) {
      return checkedAdd(total, payload_count, "Ranked bitmap payload size overflow");
    }
    if (layout != HashType::OneToMany) {
      throw std::invalid_argument("Unsupported ranked bitmap layout");
    }
    // Preserve the existing zero-distinct transition: the builder reports the
    // payload mismatch and falls back after it has computed the exact distinct count.
    if (distinct_count == 0) {
      return total;
    }
    total = checkedAdd(total, payload_count, "Ranked bitmap payload size overflow");
    return checkedAdd(
        total,
        checkedMultiply(
            distinct_count, size_t(2), "Ranked bitmap one-to-many index size overflow"),
        "Ranked bitmap one-to-many size overflow");
  }

#ifdef HAVE_CUDA
  GpuBufferPtr gpu_buffer_;
  GpuBufferPtr gpu_index_buffer_;
  GpuBufferPtr gpu_payload_ptrs_buffer_;
  GpuBufferPtr gpu_header_buffer_;
  GpuBufferPtr gpu_counts_buffer_;
  GpuBufferPtr gpu_offsets_buffer_;
  std::vector<GpuBufferPtr> gpu_payload_buffers_;
  std::vector<uint64_t> gpu_payload_ptrs_host_;
  std::array<uint64_t, kOneToManyHeaderWordCount> gpu_header_host_{{ 0, 0, 0, 0, 0, 0 }};
#endif
  std::unique_ptr<uint32_t[]> cpu_buffer_;
  std::unique_ptr<uint32_t[]> cpu_index_buffer_;
  std::unique_ptr<uint32_t[]> cpu_counts_;
  std::unique_ptr<uint32_t[]> cpu_offsets_;
  std::vector<std::unique_ptr<uint32_t[]>> cpu_payload_chunks_;
  std::vector<uint64_t> cpu_payload_ptrs_;
  std::array<uint64_t, kOneToManyHeaderWordCount> cpu_header_{{0, 0, 0, 0, 0, 0}};
  size_t bit_count_;
  size_t payload_count_;
  size_t distinct_count_;
  size_t bitmap_word_count_;
  size_t rank_block_count_;
  size_t total_word_count_;
  size_t allocated_bytes_;
  bool segmented_layout_;
  HashType layout_;
  bool payload_free_;
  size_t payload_chunk_word_count_{0};
  size_t payload_chunk_count_{0};
  Data_Namespace::DataMgr* data_mgr_;
  int device_id_;
};
