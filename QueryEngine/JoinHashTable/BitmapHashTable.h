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
#include <memory>
#include <vector>

class BitmapHashTable : public HashTable {
 public:
  static constexpr size_t kHeaderWordCount = 2;
  static constexpr size_t kHeaderBitmapPtrsPtr = 0;
  static constexpr size_t kHeaderBitmapChunkWords = 1;

  BitmapHashTable(const ExecutorDeviceType device_type,
                  const size_t bit_count,
                  const size_t max_slab_size,
                  Data_Namespace::DataMgr* data_mgr = nullptr,
                  const int device_id = -1)
      : bit_count_(bit_count)
      , byte_count_(bytesForBits(bit_count))
      , word_count_(wordsForBytes(bytesForBits(bit_count)))
      , allocated_bytes_(word_count_ * sizeof(uint32_t))
      , segmented_layout_(allocated_bytes_ > max_slab_size)
      , data_mgr_(data_mgr)
      , device_id_(device_id) {
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

  ~BitmapHashTable() override = default;

  size_t getHashTableBufferSize(const ExecutorDeviceType) const override {
    return allocated_bytes_;
  }

  int8_t* getCpuBuffer() override {
    return segmented_layout_ ? reinterpret_cast<int8_t*>(cpu_header_.data())
                             : reinterpret_cast<int8_t*>(cpu_bitmap_.get());
  }

  int8_t* getGpuBuffer() const override {
#ifdef HAVE_CUDA
    if (segmented_layout_) {
      return gpu_header_buffer_ ? gpu_header_buffer_->getMemoryPtr() : nullptr;
    }
    return gpu_bitmap_ ? gpu_bitmap_->getMemoryPtr() : nullptr;
#else
    return nullptr;
#endif
  }

  HashType getLayout() const override {
    return HashType::OneToOne;
  }

  size_t getEntryCount() const override {
    return bit_count_;
  }

  size_t getEmittedKeysCount() const override {
    return 0;
  }

  size_t getRowIdSize() const override {
    return 0;
  }

  size_t getWordCount() const noexcept {
    return word_count_;
  }

  size_t getAllocatedBytes() const noexcept {
    return allocated_bytes_;
  }

  bool hasSegmentedLayout() const noexcept {
    return segmented_layout_;
  }

  size_t getBitmapChunkWordCount() const noexcept {
    return bitmap_chunk_word_count_;
  }

  size_t getBitmapChunkCount() const noexcept {
    return bitmap_chunk_count_;
  }

  uint32_t* getCpuBitmap() {
    CHECK(!segmented_layout_);
    return cpu_bitmap_.get();
  }

  uint32_t* getCpuBitmapChunk(const size_t chunk_idx) {
    CHECK(segmented_layout_);
    CHECK_LT(chunk_idx, cpu_bitmap_chunks_.size());
    return cpu_bitmap_chunks_[chunk_idx].get();
  }

  uint64_t* getCpuBitmapPtrs() {
    CHECK(segmented_layout_);
    return cpu_bitmap_ptrs_.data();
  }

  size_t getBitmapChunkWordCount(const size_t chunk_idx) const {
    CHECK_LT(chunk_idx, bitmap_chunk_count_);
    const size_t chunk_start = chunk_idx * bitmap_chunk_word_count_;
    return std::min(bitmap_chunk_word_count_, word_count_ - chunk_start);
  }

#ifdef HAVE_CUDA
  uint32_t* getGpuBitmapChunk(const size_t chunk_idx) const {
    CHECK(segmented_layout_);
    CHECK_LT(chunk_idx, gpu_bitmap_buffers_.size());
    return reinterpret_cast<uint32_t*>(gpu_bitmap_buffers_[chunk_idx]->getMemoryPtr());
  }

  uint64_t* getGpuBitmapPtrs() const {
    CHECK(segmented_layout_);
    return reinterpret_cast<uint64_t*>(gpu_bitmap_ptrs_buffer_->getMemoryPtr());
  }

  void copyGpuHeaderToDevice(DeviceAllocator* device_allocator) const {
    CHECK(segmented_layout_);
    CHECK(device_allocator);
    device_allocator->copyToDevice(gpu_bitmap_ptrs_buffer_->getMemoryPtr(),
                                   gpu_bitmap_ptrs_host_.data(),
                                   gpu_bitmap_ptrs_host_.size() * sizeof(uint64_t),
                                   "bitmap chunk pointers");
    device_allocator->copyToDevice(gpu_header_buffer_->getMemoryPtr(),
                                   gpu_header_host_.data(),
                                   gpu_header_host_.size() * sizeof(uint64_t),
                                   "bitmap header");
  }
#endif

  static size_t bytesForBits(const size_t bit_count) {
    return bit_count == 0 ? 0 : ((bit_count - 1) / 8 + 1);
  }

  static size_t wordsForBytes(const size_t byte_count) {
    return byte_count == 0 ? 0 : ((byte_count - 1) / sizeof(uint32_t) + 1);
  }

 private:
  void checkSlabSize(const size_t bytes, const size_t max_slab_size) {
    if (bytes > max_slab_size) {
      throw JoinHashTableTooBig(bytes, max_slab_size);
    }
  }

  void allocateCpu(const size_t max_slab_size) {
    if (!segmented_layout_) {
      cpu_bitmap_.reset(new uint32_t[word_count_]());
      return;
    }

    bitmap_chunk_word_count_ =
        std::max<size_t>(size_t(1), max_slab_size / sizeof(uint32_t));
    bitmap_chunk_count_ =
        word_count_ == 0 ? 0 : ((word_count_ - 1) / bitmap_chunk_word_count_ + 1);
    allocated_bytes_ = word_count_ * sizeof(uint32_t) +
                       bitmap_chunk_count_ * sizeof(uint64_t) +
                       kHeaderWordCount * sizeof(uint64_t);
    cpu_bitmap_chunks_.reserve(bitmap_chunk_count_);
    cpu_bitmap_ptrs_.reserve(bitmap_chunk_count_);
    for (size_t chunk_idx = 0; chunk_idx < bitmap_chunk_count_; ++chunk_idx) {
      const size_t chunk_words = getBitmapChunkWordCount(chunk_idx);
      cpu_bitmap_chunks_.emplace_back(new uint32_t[chunk_words]());
      cpu_bitmap_ptrs_.push_back(
          reinterpret_cast<uint64_t>(cpu_bitmap_chunks_.back().get()));
    }
    cpu_header_[kHeaderBitmapPtrsPtr] =
        reinterpret_cast<uint64_t>(cpu_bitmap_ptrs_.data());
    cpu_header_[kHeaderBitmapChunkWords] = bitmap_chunk_word_count_;
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
        LOG(ERROR) << "Failed to release bitmap GPU buffer: " << e.what();
      } catch (...) {
        LOG(ERROR) << "Failed to release bitmap GPU buffer";
      }
    });
  }

  void allocateGpu(const size_t max_slab_size) {
    if (!segmented_layout_) {
      checkSlabSize(allocated_bytes_, max_slab_size);
      gpu_bitmap_ = allocateGpuBuffer(allocated_bytes_);
      return;
    }

    bitmap_chunk_word_count_ =
        std::max<size_t>(size_t(1), max_slab_size / sizeof(uint32_t));
    bitmap_chunk_count_ =
        word_count_ == 0 ? 0 : ((word_count_ - 1) / bitmap_chunk_word_count_ + 1);
    allocated_bytes_ = word_count_ * sizeof(uint32_t) +
                       bitmap_chunk_count_ * sizeof(uint64_t) +
                       kHeaderWordCount * sizeof(uint64_t);
    gpu_bitmap_buffers_.reserve(bitmap_chunk_count_);
    gpu_bitmap_ptrs_host_.clear();
    gpu_bitmap_ptrs_host_.reserve(bitmap_chunk_count_);
    for (size_t chunk_idx = 0; chunk_idx < bitmap_chunk_count_; ++chunk_idx) {
      const size_t chunk_words = getBitmapChunkWordCount(chunk_idx);
      const size_t chunk_bytes = chunk_words * sizeof(uint32_t);
      checkSlabSize(chunk_bytes, max_slab_size);
      auto bitmap_buffer = allocateGpuBuffer(chunk_bytes);
      gpu_bitmap_ptrs_host_.push_back(
          reinterpret_cast<uint64_t>(bitmap_buffer->getMemoryPtr()));
      gpu_bitmap_buffers_.push_back(std::move(bitmap_buffer));
    }

    const size_t bitmap_ptrs_bytes = gpu_bitmap_ptrs_host_.size() * sizeof(uint64_t);
    gpu_bitmap_ptrs_buffer_ = allocateGpuBuffer(bitmap_ptrs_bytes);
    gpu_header_host_ = std::array<uint64_t, kHeaderWordCount>{
        reinterpret_cast<uint64_t>(gpu_bitmap_ptrs_buffer_->getMemoryPtr()),
        bitmap_chunk_word_count_};
    gpu_header_buffer_ = allocateGpuBuffer(kHeaderWordCount * sizeof(uint64_t));
  }
#endif

#ifdef HAVE_CUDA
  GpuBufferPtr gpu_bitmap_;
  GpuBufferPtr gpu_bitmap_ptrs_buffer_;
  GpuBufferPtr gpu_header_buffer_;
  std::vector<GpuBufferPtr> gpu_bitmap_buffers_;
  std::vector<uint64_t> gpu_bitmap_ptrs_host_;
  std::array<uint64_t, kHeaderWordCount> gpu_header_host_{{ 0, 0 }};
#endif
  std::unique_ptr<uint32_t[]> cpu_bitmap_;
  std::vector<std::unique_ptr<uint32_t[]>> cpu_bitmap_chunks_;
  std::vector<uint64_t> cpu_bitmap_ptrs_;
  std::array<uint64_t, kHeaderWordCount> cpu_header_{{0, 0}};
  size_t bit_count_;
  size_t byte_count_;
  size_t word_count_;
  size_t allocated_bytes_;
  bool segmented_layout_;
  size_t bitmap_chunk_word_count_{0};
  size_t bitmap_chunk_count_{0};
  Data_Namespace::DataMgr* data_mgr_;
  int device_id_;
};
