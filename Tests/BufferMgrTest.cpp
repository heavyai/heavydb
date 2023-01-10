/*
 * Copyright 2023 HEAVY.AI, Inc.
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

/**
 * @file BufferMgrTest.cpp
 * @brief Unit tests for BufferMgr classes.
 */

#include <future>

#include <gtest/gtest.h>

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/AbstractBufferMgr.h"
#include "DataMgr/Allocators/ArenaAllocator.h"
#include "DataMgr/BufferMgr/CpuBufferMgr/CpuBufferMgr.h"
#include "DataMgr/BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.h"
#include "DataMgr/ForeignStorage/ForeignStorageBuffer.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "Shared/StringTransform.h"
#include "TestHelpers.h"

class UnimplementedBufferMgr : public AbstractBufferMgr {
 public:
  UnimplementedBufferMgr() : AbstractBufferMgr(0) {}

  void fetchBuffer(const ChunkKey& chunk_key,
                   AbstractBuffer* destination_buffer,
                   const size_t num_bytes) override {
    UNREACHABLE() << "Unimplemented method";
  }

  AbstractBuffer* putBuffer(const ChunkKey& chunk_key,
                            AbstractBuffer* source_buffer,
                            const size_t num_bytes) override {
    UNREACHABLE() << "Unimplemented method";
    return nullptr;
  }

  AbstractBuffer* createBuffer(const ChunkKey& chunk_key,
                               const size_t page_size,
                               const size_t initial_size) override {
    UNREACHABLE() << "Unimplemented method";
    return nullptr;
  }

  void deleteBuffer(const ChunkKey& chunk_key, const bool purge) override {
    UNREACHABLE() << "Unimplemented method";
  }

  void deleteBuffersWithPrefix(const ChunkKey& chunk_key_prefix,
                               const bool purge) override {
    UNREACHABLE() << "Unimplemented method";
  }

  AbstractBuffer* getBuffer(const ChunkKey& chunk_key, const size_t num_bytes) override {
    UNREACHABLE() << "Unimplemented method";
    return nullptr;
  }

  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunk_metadata,
                                       const ChunkKey& chunk_key_prefix) override {
    UNREACHABLE() << "Unimplemented method";
  }

  bool isBufferOnDevice(const ChunkKey& chunk_key) override {
    UNREACHABLE() << "Unimplemented method";
    return false;
  }

  std::string printSlabs() override {
    UNREACHABLE() << "Unimplemented method";
    return {};
  }

  size_t getMaxSize() override {
    UNREACHABLE() << "Unimplemented method";
    return 0;
  }

  size_t getInUseSize() override {
    UNREACHABLE() << "Unimplemented method";
    return 0;
  }

  size_t getAllocated() override {
    UNREACHABLE() << "Unimplemented method";
    return 0;
  }

  bool isAllocationCapped() override {
    UNREACHABLE() << "Unimplemented method";
    return false;
  }

  void checkpoint() override { UNREACHABLE() << "Unimplemented method"; }

  void checkpoint(const int db_id, const int tb_id) override {
    UNREACHABLE() << "Unimplemented method";
  }

  AbstractBuffer* alloc(const size_t num_bytes) override {
    UNREACHABLE() << "Unimplemented method";
    return nullptr;
  }

  void free(AbstractBuffer* buffer) override { UNREACHABLE() << "Unimplemented method"; }

  MgrType getMgrType() override {
    UNREACHABLE() << "Unimplemented method";
    return MgrType::PERSISTENT_STORAGE_MGR;
  }

  std::string getStringMgrType() override {
    UNREACHABLE() << "Unimplemented method";
    return {};
  }

  size_t getNumChunks() override {
    UNREACHABLE() << "Unimplemented method";
    return 0;
  }

  void removeTableRelatedDS(const int db_id, const int table_id) override {
    UNREACHABLE() << "Unimplemented method";
  }
};

enum class ParentMgrMethod { kNone, kPutBuffer, kFetchBuffer };

std::ostream& operator<<(std::ostream& os, ParentMgrMethod method) {
  if (method == ParentMgrMethod::kNone) {
    os << "None";
  } else if (method == ParentMgrMethod::kPutBuffer) {
    os << "PutBuffer";
  } else if (method == ParentMgrMethod::kFetchBuffer) {
    os << "FetchBuffer";
  } else {
    UNREACHABLE() << "Unexpected method: " << static_cast<int32_t>(method);
  }
  return os;
}

struct ParentMgrCallParams {
  ChunkKey chunk_key;
  AbstractBuffer* buffer;
  size_t num_bytes;

  bool operator==(const ParentMgrCallParams& other) const {
    return chunk_key == other.chunk_key && buffer == other.buffer &&
           num_bytes == other.num_bytes;
  }

  friend std::ostream& operator<<(std::ostream& os, const ParentMgrCallParams& params) {
    os << "(chunk_key: {" << join(params.chunk_key, ",") << "}, buffer: " << params.buffer
       << ", num_bytes: " << params.num_bytes << ")";
    return os;
  }
};

class MockBufferMgr : public UnimplementedBufferMgr {
 public:
  MockBufferMgr()
      : called_method_(ParentMgrMethod::kNone)
      , captured_params_({})
      , throw_foreign_storage_exception_(false)
      , reserve_twice_buffer_size_(false)
      , reserved_size_(std::nullopt)
      , skip_param_tracking_(false) {}

  void fetchBuffer(const ChunkKey& chunk_key,
                   AbstractBuffer* destination_buffer,
                   const size_t num_bytes) override {
    auto options_sum = int32_t(throw_foreign_storage_exception_) +
                       int32_t(reserve_twice_buffer_size_) +
                       int32_t(reserved_size_.has_value());
    CHECK_LE(options_sum, 1)
        << "At most one of throw_foreign_storage_exception_, reserve_twice_buffer_size_, "
           "or reserved_size_ can be set.";
    if (!skip_param_tracking_) {
      CHECK_EQ(called_method_, ParentMgrMethod::kNone);
      called_method_ = ParentMgrMethod::kFetchBuffer;
      captureParameters(chunk_key, destination_buffer, num_bytes);
    }

    if (throw_foreign_storage_exception_) {
      throw foreign_storage::ForeignStorageException(
          "Exception from mock parent buffer manager.");
    }

    CHECK(destination_buffer);
    if (reserve_twice_buffer_size_) {
      destination_buffer->reserve(destination_buffer->reservedSize() * 2);
    } else if (reserved_size_.has_value()) {
      destination_buffer->reserve(reserved_size_.value());
    } else {
      destination_buffer->reserve(num_bytes);
    }
    if (num_bytes != 0) {
      CHECK_EQ(num_bytes, buffer_content_.size());
    }
    if (destination_buffer->reservedSize() < buffer_content_.size()) {
      destination_buffer->reserve(buffer_content_.size());
    }
    destination_buffer->write(buffer_content_.data(), buffer_content_.size());
  }

  AbstractBuffer* putBuffer(const ChunkKey& chunk_key,
                            AbstractBuffer* source_buffer,
                            const size_t num_bytes) override {
    if (!skip_param_tracking_) {
      // Existing value of ParentMgrMethod::kPutBuffer is also valid because this method
      // can be called multiple times while checkpointing buffers.
      CHECK(called_method_ == ParentMgrMethod::kNone ||
            called_method_ == ParentMgrMethod::kPutBuffer)
          << "Unexpected called method: " << called_method_;
      called_method_ = ParentMgrMethod::kPutBuffer;
      captureParameters(chunk_key, source_buffer, num_bytes);
    }
    source_buffer->clearDirtyBits();
    return nullptr;
  }

  ParentMgrMethod getCalledMethod() { return called_method_; }

  const std::vector<ParentMgrCallParams>& getCapturedParams() { return captured_params_; }

  void throwForeignStorageException() { throw_foreign_storage_exception_ = true; }

  void setReserveSize(size_t reserved_size) { reserved_size_ = reserved_size; }

  void reserveTwiceBufferSize() { reserve_twice_buffer_size_ = true; }

  void skipParamTracking() { skip_param_tracking_ = true; }

  static inline std::vector<int8_t> buffer_content_{1, 2, 3, 4, 5, 6};

 private:
  void captureParameters(const ChunkKey& chunk_key,
                         AbstractBuffer* buffer,
                         const size_t num_bytes) {
    captured_params_.emplace_back(ParentMgrCallParams{chunk_key, buffer, num_bytes});
  }

  ParentMgrMethod called_method_;
  std::vector<ParentMgrCallParams> captured_params_;
  bool throw_foreign_storage_exception_;
  bool reserve_twice_buffer_size_;
  std::optional<size_t> reserved_size_;
  bool skip_param_tracking_;
};

class FailingAllocator : public DramArena {
 public:
  void* allocate(size_t size) override { throw std::bad_alloc(); }
};

#ifdef HAVE_CUDA
class MockCudaMgr : public CudaMgr_Namespace::CudaMgr {
 public:
  MockCudaMgr() : CudaMgr_Namespace::CudaMgr(1), fail_on_allocation_(false) {}

  int8_t* allocateDeviceMem(const size_t num_bytes, const int device_num) override {
    int8_t* mem_ptr{nullptr};
    if (fail_on_allocation_) {
      throw CudaMgr_Namespace::CudaErrorException(CUDA_ERROR_OUT_OF_MEMORY);
    } else {
      mem_ptr = CudaMgr_Namespace::CudaMgr::allocateDeviceMem(num_bytes, device_num);
    }
    return mem_ptr;
  }

  void setFailOnAllocation(bool fail_on_allocation) {
    fail_on_allocation_ = fail_on_allocation;
  }

 private:
  bool fail_on_allocation_;
};
#endif

class BufferMgrTest : public testing::TestWithParam<MgrType> {
 protected:
  void SetUp() override {
#ifdef HAVE_CUDA
    auto mgr_type = GetParam();
    if (mgr_type == MgrType::GPU_MGR) {
      if (!mock_cuda_mgr_) {
        mock_cuda_mgr_ = std::make_unique<MockCudaMgr>();
      }
      CHECK(mock_cuda_mgr_);
      mock_cuda_mgr_->setFailOnAllocation(false);
    }
#endif
  }

  std::unique_ptr<Buffer_Namespace::BufferMgr> createBufferMgr(
      int32_t device_id = device_id_,
      size_t max_buffer_pool_size = max_buffer_pool_size_,
      size_t min_slab_size = min_slab_size_,
      size_t max_slab_size = max_slab_size_,
      size_t page_size = page_size_) {
    auto mgr_type = GetParam();
    if (mgr_type == MgrType::CPU_MGR) {
      return std::make_unique<Buffer_Namespace::CpuBufferMgr>(device_id,
                                                              max_buffer_pool_size,
                                                              nullptr,
                                                              min_slab_size,
                                                              max_slab_size,
                                                              page_size,
                                                              &mock_parent_mgr_);
#ifdef HAVE_CUDA
    } else if (mgr_type == MgrType::GPU_MGR) {
      CHECK(mock_cuda_mgr_);
      return std::make_unique<Buffer_Namespace::GpuCudaBufferMgr>(device_id,
                                                                  max_buffer_pool_size,
                                                                  mock_cuda_mgr_.get(),
                                                                  min_slab_size,
                                                                  max_slab_size,
                                                                  page_size,
                                                                  &mock_parent_mgr_);
#endif
    } else {
      UNREACHABLE() << "Unexpected manager type: " << ToString(mgr_type);
      return nullptr;
    }
  }

  std::unique_ptr<foreign_storage::ForeignStorageBuffer> createTempBuffer(
      const std::vector<int8_t>& buffer_content,
      bool has_encoder = true) {
    auto buffer = std::make_unique<foreign_storage::ForeignStorageBuffer>();
    if (has_encoder) {
      buffer->initEncoder({kTINYINT});
    }
    buffer->append(const_cast<int8_t*>(buffer_content.data()), buffer_content.size());
    return buffer;
  }

  void setFailingMockAllocator() {
    auto mgr_type = GetParam();
    if (mgr_type == MgrType::CPU_MGR) {
      auto cpu_buffer_mgr =
          dynamic_cast<Buffer_Namespace::CpuBufferMgr*>(buffer_mgr_.get());
      CHECK(cpu_buffer_mgr);
      cpu_buffer_mgr->setAllocator(std::make_unique<FailingAllocator>());
#ifdef HAVE_CUDA
    } else if (mgr_type == MgrType::GPU_MGR) {
      CHECK(mock_cuda_mgr_);
      mock_cuda_mgr_->setFailOnAllocation(true);
#endif
    } else {
      UNREACHABLE() << "Unexpected manager type: " << ToString(mgr_type);
    }
  }

  void createPinnedBuffers(size_t buffer_count, size_t buffer_size = test_buffer_size_) {
    createBuffers(buffer_count, false, buffer_size);
  }

  void createUnpinnedBuffers(size_t buffer_count,
                             size_t buffer_size = test_buffer_size_) {
    createBuffers(buffer_count, true, buffer_size);
  }

  void createBuffers(size_t buffer_count, bool unpin_buffers, size_t buffer_size) {
    for (size_t i = 1; i <= buffer_count; i++) {
      const ChunkKey chunk_key{1, 1, 1, int32_t(i)};
      EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(chunk_key));
      auto buffer = buffer_mgr_->createBuffer(chunk_key, page_size_, buffer_size);
      EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(chunk_key));
      if (unpin_buffers) {
        buffer->unPin();
      }
    }
  }

  void assertParentMethodCalledWithParams(
      ParentMgrMethod expected_method,
      const std::vector<ParentMgrCallParams>& expected_params) {
    EXPECT_EQ(mock_parent_mgr_.getCalledMethod(), expected_method);
    EXPECT_EQ(mock_parent_mgr_.getCapturedParams(), expected_params);
  }

  void assertNoParentMethodCalled() {
    EXPECT_EQ(mock_parent_mgr_.getCalledMethod(), ParentMgrMethod::kNone);
  }

  void assertExpectedBufferAttributes(AbstractBuffer* buffer,
                                      bool has_mem_ptr = true,
                                      size_t page_count = test_buffer_size_ / page_size_,
                                      size_t reserved_size = test_buffer_size_) {
    CHECK(buffer);

    EXPECT_EQ(buffer->pageSize(), page_size_);
    EXPECT_EQ(buffer->size(), size_t(0));
    EXPECT_EQ(buffer->reservedSize(), reserved_size);
    if (has_mem_ptr) {
      EXPECT_NE(buffer->getMemoryPtr(), nullptr);
    } else {
      EXPECT_EQ(buffer->getMemoryPtr(), nullptr);
    }
    EXPECT_EQ(buffer->getPinCount(), 1);
    EXPECT_EQ(buffer->pageCount(), page_count);
    EXPECT_FALSE(buffer->isDirty());
  }

  void assertExpectedBufferMgrAttributes(size_t used_size = test_buffer_size_,
                                         size_t allocated_size = max_slab_size_,
                                         size_t num_chunks = 1,
                                         bool is_allocation_capped = false) {
    EXPECT_EQ(buffer_mgr_->getInUseSize(), used_size);
    EXPECT_EQ(buffer_mgr_->getNumChunks(), num_chunks);
    EXPECT_EQ(buffer_mgr_->size(), allocated_size / page_size_);
    EXPECT_EQ(buffer_mgr_->getAllocated(), allocated_size);
    EXPECT_EQ(buffer_mgr_->isAllocationCapped(), is_allocation_capped);
    if (allocated_size == 0) {
      EXPECT_TRUE(buffer_mgr_->getSlabSegments().empty());
    } else {
      EXPECT_FALSE(buffer_mgr_->getSlabSegments().empty());
    }
  }

  void assertEqualMetadata(AbstractBuffer* buffer_1, AbstractBuffer* buffer_2) {
    CHECK(buffer_1);
    CHECK(buffer_2);

    ASSERT_TRUE(buffer_1->hasEncoder());
    ASSERT_TRUE(buffer_2->hasEncoder());
    EXPECT_EQ(buffer_1->getSqlType(), buffer_2->getSqlType());

    auto source_chunk_metadata = std::make_shared<ChunkMetadata>();
    buffer_1->getEncoder()->getMetadata(source_chunk_metadata);

    auto chunk_metadata = std::make_shared<ChunkMetadata>();
    buffer_2->getEncoder()->getMetadata(chunk_metadata);
    EXPECT_EQ(*source_chunk_metadata, *chunk_metadata);
  }

  void assertSegmentCount(size_t expected_segment_count) {
    const auto& segments = buffer_mgr_->getSlabSegments();
    size_t segment_count{0};
    for (const auto& slab_segments : segments) {
      segment_count += slab_segments.size();
    }
    EXPECT_EQ(expected_segment_count, segment_count);
  }

  void assertSegmentAttributes(size_t slab_index,
                               size_t segment_index,
                               Buffer_Namespace::MemStatus expected_status,
                               const std::optional<ChunkKey>& expected_chunk_key = {},
                               const std::optional<size_t>& expected_size = {}) {
    auto segment_it = getSegmentAt(slab_index, segment_index);
    EXPECT_EQ(segment_it->mem_status, expected_status);

    if (expected_chunk_key.has_value()) {
      EXPECT_EQ(segment_it->chunk_key, expected_chunk_key.value());
    }

    if (expected_size.has_value()) {
      EXPECT_EQ(segment_it->num_pages, expected_size.value() / page_size_);
    }
  }

  Buffer_Namespace::BufferList::iterator getSegmentAt(size_t slab_index,
                                                      size_t segment_index) {
    auto& segments = const_cast<std::vector<Buffer_Namespace::BufferList>&>(
        buffer_mgr_->getSlabSegments());
    CHECK_GT(segments.size(), slab_index);
    CHECK_GT(segments[slab_index].size(), segment_index);
    return std::next(segments[slab_index].begin(), segment_index);
  }

  void setSegmentScores(const std::vector<std::vector<uint32_t>>& segment_scores) {
    auto& segments = const_cast<std::vector<Buffer_Namespace::BufferList>&>(
        buffer_mgr_->getSlabSegments());
    ASSERT_EQ(segments.size(), segment_scores.size());

    for (size_t slab_index = 0; slab_index < segment_scores.size(); slab_index++) {
      ASSERT_EQ(segments[slab_index].size(), segment_scores[slab_index].size());
      auto segment_it = segments[slab_index].begin();
      for (auto segment_score : segment_scores[slab_index]) {
        segment_it->last_touched = segment_score;
        std::advance(segment_it, 1);
      }
    }
  }

  std::unique_ptr<Buffer_Namespace::BufferMgr> buffer_mgr_;
#ifdef HAVE_CUDA
  static inline std::unique_ptr<MockCudaMgr> mock_cuda_mgr_;
#endif

  MockBufferMgr mock_parent_mgr_;
  static constexpr int32_t device_id_{0};
  static constexpr size_t max_buffer_pool_size_{1000};
  static constexpr size_t min_slab_size_{100};
  static constexpr size_t max_slab_size_{500};
  static constexpr size_t page_size_{10};
  static constexpr size_t test_buffer_size_{100};
  static inline const ChunkKey test_chunk_key_{1, 1, 1, 1};
  static inline const ChunkKey test_chunk_key_2_{1, 1, 1, 2};
  static inline const ChunkKey test_chunk_key_3_{1, 1, 1, 3};
};

TEST_P(BufferMgrTest, CreateBufferMgr) {
  constexpr int32_t test_device_id{5};
  constexpr size_t test_max_buffer_pool_size{5000};
  constexpr size_t test_min_slab_size{500};
  constexpr size_t test_max_slab_size{800};
  constexpr size_t test_page_size{2};

  buffer_mgr_ = createBufferMgr(test_device_id,
                                test_max_buffer_pool_size,
                                test_min_slab_size,
                                test_max_slab_size,
                                test_page_size);
  EXPECT_EQ(buffer_mgr_->getDeviceId(), test_device_id);
  EXPECT_EQ(buffer_mgr_->getMaxBufferSize(), test_max_buffer_pool_size);
  EXPECT_EQ(buffer_mgr_->getMaxBufferSize(), buffer_mgr_->getMaxSize());
  EXPECT_EQ(buffer_mgr_->getMaxSlabSize(), test_max_slab_size);
  EXPECT_EQ(buffer_mgr_->getInUseSize(), size_t(0));
  EXPECT_EQ(buffer_mgr_->getNumChunks(), size_t(0));
  EXPECT_EQ(buffer_mgr_->getPageSize(), test_page_size);
  EXPECT_EQ(buffer_mgr_->size(), size_t(0));
  EXPECT_EQ(buffer_mgr_->getAllocated(), size_t(0));
  EXPECT_FALSE(buffer_mgr_->isAllocationCapped());
  EXPECT_TRUE(buffer_mgr_->getSlabSegments().empty());
}

TEST_P(BufferMgrTest, CreateBuffer) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  assertExpectedBufferAttributes(buffer);
  assertExpectedBufferMgrAttributes();
}

TEST_P(BufferMgrTest, CreateEmptyBuffer) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, 0);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  assertExpectedBufferAttributes(buffer, false, 0, 0);
  assertExpectedBufferMgrAttributes(0, 0);
}

TEST_P(BufferMgrTest, CreateBufferNoPageSize) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, 0, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  assertExpectedBufferAttributes(buffer);
  assertExpectedBufferMgrAttributes();
}

TEST_P(BufferMgrTest, CreateBufferRequestedSizeGreaterThanMaxSlabSize) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  EXPECT_THROW(buffer_mgr_->createBuffer(test_chunk_key_, page_size_, max_slab_size_ + 1),
               TooBigForSlab);
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  assertExpectedBufferMgrAttributes(0, 0, 0);
}

TEST_P(BufferMgrTest, CreateBufferExistingSlabWithSufficientFreeSegment) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));

  auto buffer =
      buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));

  assertExpectedBufferAttributes(buffer);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);
}

TEST_P(BufferMgrTest, CreateBufferExistingSlabWithoutSufficientFreeSegment) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  // First buffer occupies almost entire slab (only one page free).
  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, max_slab_size_ - page_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  // Second buffer is too big for free segment in the first slab, so a new slab is created
  // and used.
  auto buffer =
      buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));

  assertExpectedBufferAttributes(buffer);
  assertExpectedBufferMgrAttributes(
      max_slab_size_ - page_size_ + test_buffer_size_, 2 * max_slab_size_, 2);
}

TEST_P(BufferMgrTest, CreateBufferNewSlabCreationError) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  // Create buffer that occupies entire first slab.
  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, max_slab_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  setFailingMockAllocator();
  EXPECT_THROW(
      buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_),
      OutOfMemory);
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));

  assertExpectedBufferMgrAttributes(max_slab_size_, max_slab_size_, 1, true);
}

TEST_P(BufferMgrTest, CreateBufferCannotCreateFirstSlabError) {
  buffer_mgr_ = createBufferMgr();
  setFailingMockAllocator();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  EXPECT_THROW(buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_),
               FailedToCreateFirstSlab);
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  assertExpectedBufferMgrAttributes(0, 0, 0, true);
}

TEST_P(BufferMgrTest, CreateBufferEviction) {
  constexpr int32_t test_device_id{0};
  constexpr size_t test_max_buffer_pool_size{200};
  constexpr size_t test_min_slab_size{100};
  constexpr size_t test_max_slab_size{200};

  buffer_mgr_ = createBufferMgr(test_device_id,
                                test_max_buffer_pool_size,
                                test_min_slab_size,
                                test_max_slab_size,
                                page_size_);
  createUnpinnedBuffers(2);

  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, test_max_slab_size, 2);

  auto buffer =
      buffer_mgr_->createBuffer(test_chunk_key_3_, page_size_, test_buffer_size_);
  assertExpectedBufferAttributes(buffer);

  const auto& segments = buffer_mgr_->getSlabSegments();
  ASSERT_EQ(segments.size(), size_t(1));
  ASSERT_EQ(segments[0].size(), size_t(2));
  EXPECT_EQ(segments[0].begin()->buffer, buffer);

  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, test_max_slab_size, 2);
}

TEST_P(BufferMgrTest, ClearSlabs) {
  buffer_mgr_ = createBufferMgr();
  createUnpinnedBuffers(1);

  assertSegmentCount(2);
  // First segment contains created buffer.
  assertSegmentAttributes(0, 0, Buffer_Namespace::USED);
  // Second segment is remaining free allocated memory.
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);

  buffer_mgr_->clearSlabs();

  assertExpectedBufferMgrAttributes(0, 0, 0);
}

TEST_P(BufferMgrTest, ClearSlabsPinnedBuffer) {
  buffer_mgr_ = createBufferMgr();
  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);

  assertSegmentCount(2);
  // First segment contains created buffer.
  assertSegmentAttributes(0, 0, Buffer_Namespace::USED);
  // Second segment is remaining free allocated memory.
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);

  buffer_mgr_->clearSlabs();

  assertSegmentCount(2);
  // First segment still contains created buffer due to pinning.
  assertSegmentAttributes(0, 0, Buffer_Namespace::USED);
  // Second segment is remaining free allocated memory.
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);

  assertExpectedBufferMgrAttributes();
}

TEST_P(BufferMgrTest, ClearSlabsFreeBuffer) {
  buffer_mgr_ = createBufferMgr();
  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  buffer_mgr_->deleteBuffer(test_chunk_key_);

  assertSegmentCount(1);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE);

  buffer_mgr_->clearSlabs();

  assertExpectedBufferMgrAttributes(0, 0, 0);
}

TEST_P(BufferMgrTest, DeleteBuffer) {
  buffer_mgr_ = createBufferMgr();
  createPinnedBuffers(2);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);

  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  buffer_mgr_->deleteBuffer(test_chunk_key_);
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));

  assertSegmentCount(3);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE);
  assertSegmentAttributes(0, 1, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 2, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 1);

  buffer_mgr_->deleteBuffer(test_chunk_key_2_);
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));

  assertSegmentCount(1);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(0, max_slab_size_, 0);
}

TEST_P(BufferMgrTest, DeleteBufferLeftAndRightSegmentsMerged) {
  buffer_mgr_ = createBufferMgr();
  createPinnedBuffers(3);
  assertExpectedBufferMgrAttributes(3 * test_buffer_size_, max_slab_size_, 3);

  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_3_));
  buffer_mgr_->deleteBuffer(test_chunk_key_);
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_3_));

  assertSegmentCount(4);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE);
  assertSegmentAttributes(0, 1, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 2, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 3, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);

  buffer_mgr_->deleteBuffer(test_chunk_key_3_);
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_3_));

  assertSegmentCount(3);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE);
  assertSegmentAttributes(0, 1, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 2, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 1);

  buffer_mgr_->deleteBuffer(test_chunk_key_2_);
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_3_));

  assertSegmentCount(1);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(0, max_slab_size_, 0);
}

TEST_P(BufferMgrTest, DeleteBuffersWithPrefix) {
  buffer_mgr_ = createBufferMgr();
  createUnpinnedBuffers(2);

  assertSegmentCount(3);
  assertSegmentAttributes(0, 0, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 1, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 2, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);

  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  buffer_mgr_->deleteBuffersWithPrefix({1, 1, 1});
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));

  assertSegmentCount(1);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(0, max_slab_size_, 0);
}

TEST_P(BufferMgrTest, DeleteBuffersWithPrefixPinnedBuffer) {
  buffer_mgr_ = createBufferMgr();
  createPinnedBuffers(2);

  assertSegmentCount(3);
  assertSegmentAttributes(0, 0, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 1, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 2, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);

  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  buffer_mgr_->deleteBuffersWithPrefix({1, 1, 1});
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));

  assertSegmentCount(3);
  assertSegmentAttributes(0, 0, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 1, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 2, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);
}

TEST_P(BufferMgrTest, DeleteBuffersWithPrefixNoMatchingPrefix) {
  buffer_mgr_ = createBufferMgr();
  createPinnedBuffers(1);

  assertSegmentCount(2);
  assertSegmentAttributes(0, 0, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes();

  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  buffer_mgr_->deleteBuffersWithPrefix({1, 1, 2});
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  assertSegmentCount(2);
  assertSegmentAttributes(0, 0, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes();
}

TEST_P(BufferMgrTest, AllocAndFree) {
  buffer_mgr_ = createBufferMgr();

  auto buffer = buffer_mgr_->alloc(test_buffer_size_);
  assertExpectedBufferAttributes(buffer);

  assertSegmentCount(2);
  assertSegmentAttributes(0, 0, Buffer_Namespace::USED);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes();

  buffer_mgr_->free(buffer);

  assertSegmentCount(1);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(0, max_slab_size_, 0);
}

TEST_P(BufferMgrTest, PutBufferUpdatedSourceBuffer) {
  buffer_mgr_ = createBufferMgr();

  const std::vector<int8_t> source_content{1, 2, 3, 4};
  auto source_buffer = createTempBuffer(source_content);
  source_buffer->setUpdated();
  EXPECT_TRUE(source_buffer->isDirty());

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer =
      buffer_mgr_->putBuffer(test_chunk_key_, source_buffer.get(), source_buffer->size());
  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(source_content, read_content);

  EXPECT_FALSE(source_buffer->isDirty());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(buffer, buffer_mgr_->getBuffer(test_chunk_key_));

  assertEqualMetadata(source_buffer.get(), buffer);
  // One page is used for source_content.
  assertExpectedBufferMgrAttributes(page_size_);
}

TEST_P(BufferMgrTest, PutBufferAppendedSourceBuffer) {
  buffer_mgr_ = createBufferMgr();

  const std::vector<int8_t> source_content{1, 2, 3, 4};
  auto source_buffer = createTempBuffer(source_content);
  source_buffer->setAppended();
  EXPECT_TRUE(source_buffer->isDirty());

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer =
      buffer_mgr_->putBuffer(test_chunk_key_, source_buffer.get(), source_buffer->size());

  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(source_content, read_content);

  EXPECT_FALSE(source_buffer->isDirty());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(buffer, buffer_mgr_->getBuffer(test_chunk_key_));

  assertEqualMetadata(source_buffer.get(), buffer);
  // One page is used for source_content.
  assertExpectedBufferMgrAttributes(page_size_);
}

TEST_P(BufferMgrTest, PutBufferExistingBufferUpdate) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  std::vector<int8_t> old_source_content{10, 10, 10, 10};
  buffer->write(old_source_content.data(), old_source_content.size());
  buffer->clearDirtyBits();

  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(old_source_content, read_content);

  const std::vector<int8_t> new_source_content{1, 2, 3, 4};
  auto source_buffer = createTempBuffer(new_source_content);
  source_buffer->setUpdated();
  EXPECT_TRUE(source_buffer->isDirty());

  buffer =
      buffer_mgr_->putBuffer(test_chunk_key_, source_buffer.get(), source_buffer->size());
  read_content.resize(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(new_source_content, read_content);

  EXPECT_FALSE(source_buffer->isDirty());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(buffer, buffer_mgr_->getBuffer(test_chunk_key_));

  assertEqualMetadata(source_buffer.get(), buffer);
  assertExpectedBufferMgrAttributes();
}

TEST_P(BufferMgrTest, PutBufferExistingBufferPartialUpdate) {
  buffer_mgr_ = createBufferMgr();

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  std::vector<int8_t> old_source_content{10, 10, 10, 10};
  buffer->write(old_source_content.data(), old_source_content.size());
  buffer->clearDirtyBits();
  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(old_source_content, read_content);

  const std::vector<int8_t> new_source_content{1, 2};
  auto source_buffer = createTempBuffer(new_source_content);
  source_buffer->setUpdated();
  EXPECT_TRUE(source_buffer->isDirty());

  buffer =
      buffer_mgr_->putBuffer(test_chunk_key_, source_buffer.get(), source_buffer->size());
  read_content.resize(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  const std::vector<int8_t> final_content{1, 2, 10, 10};
  EXPECT_EQ(final_content, read_content);

  EXPECT_FALSE(source_buffer->isDirty());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(buffer, buffer_mgr_->getBuffer(test_chunk_key_));

  ASSERT_TRUE(source_buffer->hasEncoder());
  ASSERT_TRUE(buffer->hasEncoder());
  EXPECT_EQ(source_buffer->getSqlType(), buffer->getSqlType());

  auto source_chunk_metadata = std::make_shared<ChunkMetadata>();
  source_buffer->getEncoder()->getMetadata(source_chunk_metadata);

  auto chunk_metadata = std::make_shared<ChunkMetadata>();
  buffer->getEncoder()->getMetadata(chunk_metadata);

  EXPECT_EQ(source_chunk_metadata->sqlType, chunk_metadata->sqlType);
  EXPECT_EQ(source_chunk_metadata->numBytes, size_t(2));
  EXPECT_EQ(chunk_metadata->numBytes, size_t(4));

  assertExpectedBufferMgrAttributes();
}

TEST_P(BufferMgrTest, PutBufferExistingBufferAppend) {
  buffer_mgr_ = createBufferMgr();

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  std::vector<int8_t> old_source_content{10, 10};
  buffer->write(old_source_content.data(), old_source_content.size());
  buffer->clearDirtyBits();
  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(old_source_content, read_content);

  const std::vector<int8_t> new_source_content{1, 2, 3, 4};
  auto source_buffer = createTempBuffer(new_source_content);
  source_buffer->setAppended();
  EXPECT_TRUE(source_buffer->isDirty());

  buffer =
      buffer_mgr_->putBuffer(test_chunk_key_, source_buffer.get(), source_buffer->size());
  read_content.resize(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  const std::vector<int8_t> final_content{10, 10, 3, 4};
  EXPECT_EQ(final_content, read_content);

  EXPECT_FALSE(source_buffer->isDirty());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(buffer, buffer_mgr_->getBuffer(test_chunk_key_));

  assertEqualMetadata(source_buffer.get(), buffer);
  assertExpectedBufferMgrAttributes();
}

TEST_P(BufferMgrTest, PutBufferZeroNumBytes) {
  buffer_mgr_ = createBufferMgr();

  const std::vector<int8_t> source_content{1, 2, 3, 4};
  auto source_buffer = createTempBuffer(source_content);
  source_buffer->setUpdated();
  EXPECT_TRUE(source_buffer->isDirty());

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer = buffer_mgr_->putBuffer(test_chunk_key_, source_buffer.get());
  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(source_content, read_content);

  EXPECT_FALSE(source_buffer->isDirty());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(buffer, buffer_mgr_->getBuffer(test_chunk_key_));

  assertEqualMetadata(source_buffer.get(), buffer);
  // One page is used for source_content.
  assertExpectedBufferMgrAttributes(page_size_);
}

TEST_P(BufferMgrTest, PutBufferSourceBufferHasNoEncoder) {
  buffer_mgr_ = createBufferMgr();

  const std::vector<int8_t> source_content{1, 2, 3, 4};
  auto source_buffer = createTempBuffer(source_content, false);
  source_buffer->setUpdated();
  EXPECT_TRUE(source_buffer->isDirty());

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer = buffer_mgr_->putBuffer(test_chunk_key_, source_buffer.get());
  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(source_content, read_content);

  EXPECT_FALSE(source_buffer->isDirty());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(buffer, buffer_mgr_->getBuffer(test_chunk_key_));

  EXPECT_FALSE(source_buffer->hasEncoder());
  EXPECT_FALSE(buffer->hasEncoder());
  EXPECT_EQ(source_buffer->getSqlType(), buffer->getSqlType());

  // One page is used for source_content.
  assertExpectedBufferMgrAttributes(page_size_);
}

TEST_P(BufferMgrTest, Checkpoint) {
  buffer_mgr_ = createBufferMgr();

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer_1 =
      buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  buffer_1->setDirty();
  EXPECT_TRUE(buffer_1->isDirty());

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  auto buffer_2 =
      buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  buffer_2->setDirty();
  EXPECT_TRUE(buffer_2->isDirty());

  auto buffer_3 = buffer_mgr_->alloc(test_buffer_size_);
  buffer_3->setDirty();
  EXPECT_TRUE(buffer_3->isDirty());

  buffer_mgr_->checkpoint();
  EXPECT_FALSE(buffer_1->isDirty());
  EXPECT_FALSE(buffer_2->isDirty());
  EXPECT_TRUE(buffer_3->isDirty());

  assertExpectedBufferMgrAttributes(3 * test_buffer_size_, max_slab_size_, 3);
  assertParentMethodCalledWithParams(ParentMgrMethod::kPutBuffer,
                                     {{test_chunk_key_, buffer_1, buffer_1->size()},
                                      {test_chunk_key_2_, buffer_2, buffer_2->size()}});
}

TEST_P(BufferMgrTest, CheckpointCleanBuffers) {
  buffer_mgr_ = createBufferMgr();

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer_1 =
      buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_FALSE(buffer_1->isDirty());

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  auto buffer_2 =
      buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  EXPECT_FALSE(buffer_2->isDirty());

  auto buffer_3 = buffer_mgr_->alloc(test_buffer_size_);
  EXPECT_FALSE(buffer_3->isDirty());

  buffer_mgr_->checkpoint();
  EXPECT_FALSE(buffer_1->isDirty());
  EXPECT_FALSE(buffer_2->isDirty());
  EXPECT_FALSE(buffer_3->isDirty());

  assertExpectedBufferMgrAttributes(3 * test_buffer_size_, max_slab_size_, 3);
  assertNoParentMethodCalled();
}

TEST_P(BufferMgrTest, TableLevelCheckpoint) {
  buffer_mgr_ = createBufferMgr();

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer_1 =
      buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  buffer_1->setDirty();
  EXPECT_TRUE(buffer_1->isDirty());

  const ChunkKey different_table_chunk_key{1, 2, 1, 1};
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(different_table_chunk_key));
  auto different_table_buffer =
      buffer_mgr_->createBuffer(different_table_chunk_key, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(different_table_chunk_key));
  different_table_buffer->setDirty();
  EXPECT_TRUE(different_table_buffer->isDirty());

  auto buffer_3 = buffer_mgr_->alloc(test_buffer_size_);
  buffer_3->setDirty();
  EXPECT_TRUE(buffer_3->isDirty());

  buffer_mgr_->checkpoint(1, 2);
  EXPECT_TRUE(buffer_1->isDirty());
  EXPECT_FALSE(different_table_buffer->isDirty());
  EXPECT_TRUE(buffer_3->isDirty());

  assertExpectedBufferMgrAttributes(3 * test_buffer_size_, max_slab_size_, 3);
  assertParentMethodCalledWithParams(ParentMgrMethod::kPutBuffer,
                                     {{different_table_chunk_key,
                                       different_table_buffer,
                                       different_table_buffer->size()}});
}

TEST_P(BufferMgrTest, TableLevelCheckpointCleanBuffers) {
  buffer_mgr_ = createBufferMgr();

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer_1 =
      buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_FALSE(buffer_1->isDirty());

  const ChunkKey different_table_chunk_key{1, 2, 1, 1};
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(different_table_chunk_key));
  auto different_table_buffer =
      buffer_mgr_->createBuffer(different_table_chunk_key, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(different_table_chunk_key));
  EXPECT_FALSE(different_table_buffer->isDirty());

  auto buffer_3 = buffer_mgr_->alloc(test_buffer_size_);
  EXPECT_FALSE(buffer_3->isDirty());

  buffer_mgr_->checkpoint(1, 2);
  EXPECT_FALSE(buffer_1->isDirty());
  EXPECT_FALSE(different_table_buffer->isDirty());
  EXPECT_FALSE(buffer_3->isDirty());

  assertExpectedBufferMgrAttributes(3 * test_buffer_size_, max_slab_size_, 3);
  assertNoParentMethodCalled();
}

TEST_P(BufferMgrTest, TableLevelCheckpointNoBuffersForTable) {
  buffer_mgr_ = createBufferMgr();

  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  auto buffer_1 =
      buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_FALSE(buffer_1->isDirty());

  const ChunkKey different_table_chunk_key{1, 2, 1, 1};
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(different_table_chunk_key));
  auto different_table_buffer =
      buffer_mgr_->createBuffer(different_table_chunk_key, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(different_table_chunk_key));
  EXPECT_FALSE(different_table_buffer->isDirty());

  auto buffer_3 = buffer_mgr_->alloc(test_buffer_size_);
  EXPECT_FALSE(buffer_3->isDirty());

  buffer_mgr_->checkpoint(1, 3);
  EXPECT_FALSE(buffer_1->isDirty());
  EXPECT_FALSE(different_table_buffer->isDirty());
  EXPECT_FALSE(buffer_3->isDirty());

  assertExpectedBufferMgrAttributes(3 * test_buffer_size_, max_slab_size_, 3);
  assertNoParentMethodCalled();
}

TEST_P(BufferMgrTest, ReserveBufferRequestedSizeLessThanReservedSize) {
  buffer_mgr_ = createBufferMgr();
  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);

  assertSegmentCount(2);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);

  auto segment_it = getSegmentAt(0, 0);
  buffer_mgr_->reserveBuffer(segment_it, 1);

  assertSegmentCount(2);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);

  assertExpectedBufferMgrAttributes();
}

TEST_P(BufferMgrTest, ReserveBufferUseNextFreeSegment) {
  buffer_mgr_ = createBufferMgr();

  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);

  assertSegmentCount(2);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 1);

  auto new_reserved_size = test_buffer_size_ + page_size_;
  auto segment_it = getSegmentAt(0, 0);
  buffer_mgr_->reserveBuffer(segment_it, new_reserved_size);

  assertSegmentCount(2);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, new_reserved_size);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(new_reserved_size, max_slab_size_, 1);
}

TEST_P(BufferMgrTest, ReserveBufferUseAllNextFreeSegment) {
  buffer_mgr_ = createBufferMgr();

  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);

  assertSegmentCount(2);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 1);

  auto segment_it = getSegmentAt(0, 0);
  buffer_mgr_->reserveBuffer(segment_it, max_slab_size_);

  assertSegmentCount(2);
  assertSegmentAttributes(0, 0, Buffer_Namespace::USED, test_chunk_key_, max_slab_size_);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE, {}, 0);
  assertExpectedBufferMgrAttributes(max_slab_size_, max_slab_size_, 1);
}

TEST_P(BufferMgrTest, ReserveBufferRequestedSizeGreaterThanMaxSlabSize) {
  buffer_mgr_ = createBufferMgr();

  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);

  assertSegmentCount(2);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 1);

  auto segment_it = getSegmentAt(0, 0);
  EXPECT_THROW(buffer_mgr_->reserveBuffer(segment_it, max_slab_size_ + 1), TooBigForSlab);

  assertSegmentCount(2);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 1);
}

TEST_P(BufferMgrTest, ReserveBufferFreeSegmentInSubsequentSlab) {
  buffer_mgr_ = createBufferMgr();
  buffer_mgr_->createBuffer(test_chunk_key_3_, page_size_, max_slab_size_);

  // Create buffer in second slab.
  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_);

  assertSegmentCount(4);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_3_, max_slab_size_);
  assertSegmentAttributes(
      1, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_);
  assertSegmentAttributes(
      1, 1, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size_);
  assertSegmentAttributes(1, 2, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(
      max_slab_size_ + (2 * test_buffer_size_), 2 * max_slab_size_, 3);

  auto slab_2_remaining_size = max_slab_size_ - (2 * test_buffer_size_);

  auto segment_it = getSegmentAt(1, 0);
  std::vector<int8_t> old_content{1, 2, 3, 4, 5};
  ASSERT_NE(segment_it->buffer, nullptr);
  segment_it->buffer->write(old_content.data(), old_content.size());
  EXPECT_EQ(segment_it->buffer->size(), old_content.size());

  auto returned_it = buffer_mgr_->reserveBuffer(segment_it, slab_2_remaining_size);

  assertSegmentCount(4);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_3_, max_slab_size_);
  assertSegmentAttributes(1, 0, Buffer_Namespace::FREE, {}, test_buffer_size_);
  assertSegmentAttributes(
      1, 1, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size_);
  assertSegmentAttributes(
      1, 2, Buffer_Namespace::USED, test_chunk_key_, slab_2_remaining_size);

  auto segment_it_2 = getSegmentAt(1, 2);

  ASSERT_EQ(segment_it_2, returned_it);
  ASSERT_NE(segment_it_2->buffer->getMemoryPtr(), nullptr);

  // Previous buffer content should be transferred to new segment.
  ASSERT_EQ(segment_it_2->buffer->size(), old_content.size());
  std::vector<int8_t> new_content(segment_it_2->buffer->size());
  segment_it_2->buffer->read(new_content.data(), segment_it_2->buffer->size());
  EXPECT_EQ(old_content, new_content);

  // Old buffer should be freed.
  ASSERT_EQ(segment_it->buffer, nullptr);

  assertExpectedBufferMgrAttributes(
      max_slab_size_ + slab_2_remaining_size + test_buffer_size_, 2 * max_slab_size_, 3);
}

TEST_P(BufferMgrTest, ReserveBufferNewSlabCreation) {
  buffer_mgr_ = createBufferMgr();
  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_);

  assertSegmentCount(3);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_);
  assertSegmentAttributes(
      0, 1, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size_);
  assertSegmentAttributes(0, 2, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);

  auto segment_it = getSegmentAt(0, 0);
  segment_it = buffer_mgr_->reserveBuffer(segment_it, max_slab_size_);

  assertSegmentCount(4);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE, {}, test_buffer_size_);
  assertSegmentAttributes(
      0, 1, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size_);
  assertSegmentAttributes(0, 2, Buffer_Namespace::FREE);
  assertSegmentAttributes(1, 0, Buffer_Namespace::USED, test_chunk_key_, max_slab_size_);
  assertExpectedBufferMgrAttributes(
      max_slab_size_ + test_buffer_size_, 2 * max_slab_size_, 2);
}

TEST_P(BufferMgrTest, ReserveBufferNewSlabCreationError) {
  buffer_mgr_ = createBufferMgr();

  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_);

  assertSegmentCount(3);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_);
  assertSegmentAttributes(
      0, 1, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size_);
  assertSegmentAttributes(0, 2, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);

  setFailingMockAllocator();

  auto segment_it = getSegmentAt(0, 0);
  EXPECT_THROW(buffer_mgr_->reserveBuffer(segment_it, max_slab_size_), OutOfMemory);

  assertSegmentCount(3);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_);
  assertSegmentAttributes(
      0, 1, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size_);
  assertSegmentAttributes(0, 2, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);
}

TEST_P(BufferMgrTest, ReserveBufferNewSlabCreationPreviouslyEmptyBuffer) {
  buffer_mgr_ = createBufferMgr();
  auto empty_buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, 0);
  buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_);

  assertSegmentCount(2);
  // No segment for empty buffer. Buffer is still contained in unsized_segs_ vector.
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size_);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 2);

  // Reservation is made using empty_buffer, since corresponding segment is not contained
  // in slab_segments_ vector.
  empty_buffer->reserve(max_slab_size_);

  assertSegmentCount(3);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size_);
  assertSegmentAttributes(0, 1, Buffer_Namespace::FREE);
  assertSegmentAttributes(1, 0, Buffer_Namespace::USED, test_chunk_key_, max_slab_size_);
  assertExpectedBufferMgrAttributes(
      max_slab_size_ + test_buffer_size_, 2 * max_slab_size_, 2);
}

TEST_P(BufferMgrTest, ReserveBufferWithBufferEviction) {
  constexpr int32_t test_device_id{0};
  constexpr size_t test_max_buffer_pool_size{200};
  constexpr size_t test_min_slab_size{100};
  constexpr size_t test_max_slab_size{200};

  buffer_mgr_ = createBufferMgr(test_device_id,
                                test_max_buffer_pool_size,
                                test_min_slab_size,
                                test_max_slab_size,
                                page_size_);

  constexpr size_t test_buffer_size = test_max_slab_size / 4;
  createUnpinnedBuffers(2, test_buffer_size);

  assertSegmentCount(3);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size);
  assertSegmentAttributes(
      0, 1, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size);
  assertSegmentAttributes(0, 2, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size, test_max_slab_size, 2);

  auto segment_it = getSegmentAt(0, 0);
  segment_it->buffer->pin();
  // Force eviction of second segment by reserving the remaining slab size (i.e. slab size
  // - pinned buffer reserved size).
  auto remaining_slab_size = test_max_slab_size - test_buffer_size;
  buffer_mgr_->reserveBuffer(segment_it, remaining_slab_size);

  assertSegmentCount(2);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE, {}, test_buffer_size);
  assertSegmentAttributes(
      0, 1, Buffer_Namespace::USED, test_chunk_key_, remaining_slab_size);
  assertExpectedBufferMgrAttributes(remaining_slab_size, test_max_slab_size, 1);
}

TEST_P(BufferMgrTest,
       ReserveBufferWithBufferEvictionContiguousUnpinnedSegmentsWithLowestScoreEvicted) {
  constexpr int32_t test_device_id{0};
  constexpr size_t test_max_buffer_pool_size{1000};
  constexpr size_t test_min_slab_size{100};
  constexpr size_t test_max_slab_size{500};

  buffer_mgr_ = createBufferMgr(test_device_id,
                                test_max_buffer_pool_size,
                                test_min_slab_size,
                                test_max_slab_size,
                                page_size_);
  createUnpinnedBuffers(10);
  setSegmentScores({
      {0, 10, 5, 10, 10},  // Slab 0
      {10, 11, 7, 8, 9}    // Slab 1
  });

  assertSegmentCount(10);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_);
  assertExpectedBufferMgrAttributes(10 * test_buffer_size_, 2 * test_max_slab_size, 10);

  auto segment_it = getSegmentAt(0, 0);
  segment_it->buffer->pin();
  buffer_mgr_->reserveBuffer(segment_it, test_buffer_size_ * 3);

  // Segments 2, 3, and 4 on slab 1 should be evicted.
  assertSegmentCount(8);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE, {}, test_buffer_size_);
  assertSegmentAttributes(
      1, 2, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size_ * 3);
  assertExpectedBufferMgrAttributes(9 * test_buffer_size_, 2 * test_max_slab_size, 7);
}

TEST_P(BufferMgrTest, ReserveBufferWithBufferEvictionLastSegmentEvicted) {
  constexpr int32_t test_device_id{0};
  constexpr size_t test_max_buffer_pool_size{200};
  constexpr size_t test_min_slab_size{100};
  constexpr size_t test_max_slab_size{150};

  buffer_mgr_ = createBufferMgr(test_device_id,
                                test_max_buffer_pool_size,
                                test_min_slab_size,
                                test_max_slab_size,
                                page_size_);

  constexpr size_t test_buffer_size = test_max_slab_size / 3;
  createUnpinnedBuffers(3, test_buffer_size);

  assertSegmentCount(3);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size);
  assertSegmentAttributes(
      0, 1, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size);
  assertSegmentAttributes(
      0, 2, Buffer_Namespace::USED, test_chunk_key_3_, test_buffer_size);
  assertExpectedBufferMgrAttributes(3 * test_buffer_size, test_max_slab_size, 3);

  auto segment_it = getSegmentAt(0, 0);
  segment_it->buffer->pin();
  buffer_mgr_->reserveBuffer(segment_it, test_buffer_size * 2);

  assertSegmentCount(2);
  assertSegmentAttributes(0, 0, Buffer_Namespace::FREE, {}, test_buffer_size);
  assertSegmentAttributes(
      0, 1, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size * 2);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size, test_max_slab_size, 1);
}

TEST_P(BufferMgrTest, ReserveBufferWithBufferEvictionNoBufferToEvict) {
  constexpr int32_t test_device_id{0};
  constexpr size_t test_max_buffer_pool_size{200};
  constexpr size_t test_min_slab_size{100};
  constexpr size_t test_max_slab_size{200};

  buffer_mgr_ = createBufferMgr(test_device_id,
                                test_max_buffer_pool_size,
                                test_min_slab_size,
                                test_max_slab_size,
                                page_size_);

  constexpr size_t test_buffer_size = test_max_slab_size / 4;
  createPinnedBuffers(3, test_buffer_size);

  assertSegmentCount(4);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size);
  assertSegmentAttributes(
      0, 1, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size);
  assertSegmentAttributes(
      0, 2, Buffer_Namespace::USED, test_chunk_key_3_, test_buffer_size);
  assertSegmentAttributes(0, 3, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(3 * test_buffer_size, test_max_slab_size, 3);

  auto segment_it = getSegmentAt(0, 0);
  EXPECT_THROW(buffer_mgr_->reserveBuffer(segment_it, test_max_slab_size / 2),
               OutOfMemory);

  assertSegmentCount(4);
  assertSegmentAttributes(
      0, 0, Buffer_Namespace::USED, test_chunk_key_, test_buffer_size);
  assertSegmentAttributes(
      0, 1, Buffer_Namespace::USED, test_chunk_key_2_, test_buffer_size);
  assertSegmentAttributes(
      0, 2, Buffer_Namespace::USED, test_chunk_key_3_, test_buffer_size);
  assertSegmentAttributes(0, 3, Buffer_Namespace::FREE);
  assertExpectedBufferMgrAttributes(3 * test_buffer_size, test_max_slab_size, 3);
}

TEST_P(BufferMgrTest, FetchBufferCacheHit) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  std::vector<int8_t> source_content{1, 2, 3, 4};
  buffer->append(source_content.data(), source_content.size());

  auto dest_buffer = std::make_unique<foreign_storage::ForeignStorageBuffer>();

  buffer_mgr_->fetchBuffer(test_chunk_key_, dest_buffer.get());
  EXPECT_EQ(buffer->size(), dest_buffer->size());

  std::vector<int8_t> dest_content(dest_buffer->size());
  dest_buffer->read(dest_content.data(), dest_buffer->size());
  EXPECT_EQ(dest_content, source_content);

  assertExpectedBufferMgrAttributes();
  assertNoParentMethodCalled();
}

TEST_P(BufferMgrTest, FetchBufferCacheHitNumBytesGreaterThanBufferSize) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 1);

  std::vector<int8_t> source_content{1, 2, 3, 4};
  buffer->append(source_content.data(), source_content.size());

  auto dest_buffer = std::make_unique<foreign_storage::ForeignStorageBuffer>();
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;

  buffer_mgr_->fetchBuffer(test_chunk_key_, dest_buffer.get(), new_source_content.size());
  EXPECT_EQ(new_source_content.size(), dest_buffer->size());

  std::vector<int8_t> dest_content(dest_buffer->size());
  dest_buffer->read(dest_content.data(), dest_buffer->size());
  EXPECT_EQ(dest_content, new_source_content);

  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 1);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, buffer, new_source_content.size()}});
}

// TODO: Should this case also delete the buffer (similar to what getBuffer does)?
TEST_P(BufferMgrTest,
       FetchBufferCacheHitNumBytesGreaterThanBufferSizeForeignStorageException) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 1);

  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  std::vector<int8_t> source_content{1, 2, 3, 4};
  buffer->append(source_content.data(), source_content.size());

  auto dest_buffer = std::make_unique<foreign_storage::ForeignStorageBuffer>();
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;
  mock_parent_mgr_.throwForeignStorageException();

  EXPECT_THROW(buffer_mgr_->fetchBuffer(
                   test_chunk_key_, dest_buffer.get(), new_source_content.size()),
               foreign_storage::ForeignStorageException);

  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 1);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, buffer, new_source_content.size()}});
}

TEST_P(BufferMgrTest, FetchBufferCacheMiss) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  auto dest_buffer = std::make_unique<foreign_storage::ForeignStorageBuffer>();
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;

  buffer_mgr_->fetchBuffer(test_chunk_key_, dest_buffer.get(), new_source_content.size());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(new_source_content.size(), dest_buffer->size());

  std::vector<int8_t> dest_content(dest_buffer->size());
  dest_buffer->read(dest_content.data(), dest_buffer->size());
  EXPECT_EQ(dest_content, new_source_content);

  // One page is used for new_source_content.
  assertExpectedBufferMgrAttributes(page_size_);
  auto segment_it = getSegmentAt(0, 0);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, segment_it->buffer, new_source_content.size()}});
}

TEST_P(BufferMgrTest, FetchBufferCacheMissForeignStorageException) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  auto dest_buffer = std::make_unique<foreign_storage::ForeignStorageBuffer>();
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;
  mock_parent_mgr_.throwForeignStorageException();

  EXPECT_THROW(buffer_mgr_->fetchBuffer(
                   test_chunk_key_, dest_buffer.get(), new_source_content.size()),
               foreign_storage::ForeignStorageException);
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  assertExpectedBufferMgrAttributes(0, max_slab_size_, 0);
  EXPECT_EQ(getSegmentAt(0, 0)->buffer, nullptr);
}

TEST_P(BufferMgrTest, GetBufferCacheHit) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  std::vector<int8_t> source_content{1, 2, 3, 4};
  buffer->append(source_content.data(), source_content.size());

  buffer = buffer_mgr_->getBuffer(test_chunk_key_);
  EXPECT_EQ(buffer->size(), source_content.size());

  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(read_content, source_content);

  assertExpectedBufferMgrAttributes();
  assertNoParentMethodCalled();
}

TEST_P(BufferMgrTest, GetBufferCacheHitNumBytesGreaterThanBufferSize) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  std::vector<int8_t> source_content{1, 2, 3, 4};
  buffer->append(source_content.data(), source_content.size());

  const auto& new_source_content = mock_parent_mgr_.buffer_content_;

  buffer = buffer_mgr_->getBuffer(test_chunk_key_, new_source_content.size());
  EXPECT_EQ(new_source_content.size(), buffer->size());

  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(read_content, new_source_content);

  assertExpectedBufferMgrAttributes();
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, buffer, new_source_content.size()}});
}

TEST_P(BufferMgrTest,
       GetBufferCacheHitNumBytesGreaterThanBufferSizeForeignStorageException) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  auto buffer = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  std::vector<int8_t> source_content{1, 2, 3, 4};
  buffer->append(source_content.data(), source_content.size());

  const auto& new_source_content = mock_parent_mgr_.buffer_content_;
  mock_parent_mgr_.throwForeignStorageException();

  EXPECT_THROW(buffer_mgr_->getBuffer(test_chunk_key_, new_source_content.size()),
               foreign_storage::ForeignStorageException);

  assertExpectedBufferMgrAttributes(test_buffer_size_, max_slab_size_, 1);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, buffer, new_source_content.size()}});
}

TEST_P(BufferMgrTest, GetBufferCacheMiss) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  const auto& new_source_content = mock_parent_mgr_.buffer_content_;

  auto buffer = buffer_mgr_->getBuffer(test_chunk_key_, new_source_content.size());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(new_source_content.size(), buffer->size());

  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(read_content, new_source_content);

  // One page is used for new_source_content.
  assertExpectedBufferMgrAttributes(page_size_);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, buffer, new_source_content.size()}});
}

TEST_P(BufferMgrTest, GetBufferCacheMissForeignStorageException) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  const auto& new_source_content = mock_parent_mgr_.buffer_content_;
  mock_parent_mgr_.throwForeignStorageException();

  EXPECT_THROW(buffer_mgr_->getBuffer(test_chunk_key_, new_source_content.size()),
               foreign_storage::ForeignStorageException);
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  assertExpectedBufferMgrAttributes(0, max_slab_size_, 0);
  EXPECT_EQ(getSegmentAt(0, 0)->buffer, nullptr);
}

// For the following test cases, "re-entrant" means that the buffer reservation call is a
// result of the parent buffer manager calling `reserve` on the buffer when
// `BufferMgr::getBuffer()` is called.
TEST_P(BufferMgrTest, GetBufferReentrantReserveExtendsToNextSegment) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  constexpr size_t test_page_size = 10;
  constexpr size_t test_buffer_size = 100;
  buffer_mgr_->createBuffer(test_chunk_key_, test_page_size, test_buffer_size);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  assertExpectedBufferMgrAttributes(test_buffer_size, max_slab_size_, 1);

  auto new_reserved_size = 2 * test_buffer_size;
  mock_parent_mgr_.setReserveSize(new_reserved_size);
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;

  auto buffer = buffer_mgr_->getBuffer(test_chunk_key_, new_source_content.size());
  EXPECT_EQ(buffer->reservedSize(), new_reserved_size);
  EXPECT_EQ(new_source_content.size(), buffer->size());

  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(read_content, new_source_content);

  assertExpectedBufferMgrAttributes(new_reserved_size, max_slab_size_, 1);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, buffer, new_source_content.size()}});
}

// TODO: This test case currently causes a crash and should likely just propagate the out
// of memory error.
TEST_P(BufferMgrTest, DISABLED_GetBufferReentrantReserveTooBigForSlab) {
  buffer_mgr_ = createBufferMgr();
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  mock_parent_mgr_.setReserveSize(max_slab_size_ + 1);
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;

  EXPECT_THROW(buffer_mgr_->getBuffer(test_chunk_key_, new_source_content.size()),
               TooBigForSlab);
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));

  assertExpectedBufferMgrAttributes();
  auto segment_it = getSegmentAt(0, 0);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, segment_it->buffer, new_source_content.size()}});
}

TEST_P(BufferMgrTest, GetBufferReentrantReserveFreeSegmentInSubsequentSlab) {
  buffer_mgr_ = createBufferMgr();
  buffer_mgr_->createBuffer(test_chunk_key_3_, page_size_, max_slab_size_);

  // Slab 2
  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_);

  assertExpectedBufferMgrAttributes(
      max_slab_size_ + (2 * test_buffer_size_), 2 * max_slab_size_, 3);

  auto new_reserved_size = max_slab_size_ - (2 * test_buffer_size_);
  mock_parent_mgr_.setReserveSize(new_reserved_size);
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;

  auto buffer = buffer_mgr_->getBuffer(test_chunk_key_, new_source_content.size());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(buffer->reservedSize(), new_reserved_size);
  EXPECT_EQ(new_source_content.size(), buffer->size());

  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(read_content, new_source_content);

  assertExpectedBufferMgrAttributes(
      max_slab_size_ + new_reserved_size + test_buffer_size_, 2 * max_slab_size_, 3);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, buffer, new_source_content.size()}});
}

TEST_P(BufferMgrTest, GetBufferReentrantReserveNewSlabCreation) {
  buffer_mgr_ = createBufferMgr();
  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);
  buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_);

  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);

  mock_parent_mgr_.setReserveSize(max_slab_size_);
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;

  auto buffer = buffer_mgr_->getBuffer(test_chunk_key_, new_source_content.size());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(buffer->reservedSize(), max_slab_size_);
  EXPECT_EQ(new_source_content.size(), buffer->size());

  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(read_content, new_source_content);

  assertExpectedBufferMgrAttributes(
      max_slab_size_ + test_buffer_size_, 2 * max_slab_size_, 2);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, buffer, new_source_content.size()}});
}

TEST_P(BufferMgrTest, GetBufferReentrantReserveNewSlabCreationError) {
  buffer_mgr_ = createBufferMgr();
  buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, test_buffer_size_);
  buffer_mgr_->createBuffer(test_chunk_key_, page_size_, test_buffer_size_);

  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);

  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  mock_parent_mgr_.setReserveSize(max_slab_size_);
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;
  setFailingMockAllocator();

  EXPECT_THROW(buffer_mgr_->getBuffer(test_chunk_key_, new_source_content.size()),
               OutOfMemory);

  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, max_slab_size_, 2);
  auto segment_it = getSegmentAt(0, 1);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, segment_it->buffer, new_source_content.size()}});
}

TEST_P(BufferMgrTest, GetBufferReentrantReserveWithEviction) {
  constexpr int32_t test_device_id{0};
  constexpr size_t test_max_buffer_pool_size{200};
  constexpr size_t test_max_slab_size = test_max_buffer_pool_size;
  constexpr size_t test_min_slab_size = {100};

  buffer_mgr_ = createBufferMgr(test_device_id,
                                test_max_buffer_pool_size,
                                test_min_slab_size,
                                test_max_slab_size,
                                page_size_);
  auto buffer_1_size = test_max_slab_size * 3 / 4;
  auto buffer_1 = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, buffer_1_size);
  buffer_1->unPin();

  auto buffer_2_size = test_max_slab_size * 1 / 4;
  auto buffer_2 = buffer_mgr_->createBuffer(test_chunk_key_2_, page_size_, buffer_2_size);
  buffer_2->unPin();
  assertExpectedBufferMgrAttributes(test_max_slab_size, test_max_slab_size, 2);

  auto new_buffer_size = test_max_slab_size / 2;
  mock_parent_mgr_.setReserveSize(new_buffer_size);
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;

  auto buffer = buffer_mgr_->getBuffer(test_chunk_key_2_, new_source_content.size());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_EQ(buffer->reservedSize(), new_buffer_size);
  EXPECT_EQ(new_source_content.size(), buffer->size());

  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(read_content, new_source_content);

  assertExpectedBufferMgrAttributes(new_buffer_size, test_max_slab_size, 1);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_2_, buffer, new_source_content.size()}});
}

TEST_P(BufferMgrTest, GetBufferReentrantReserveWithEvictionOfLastSegmentInSlab) {
  constexpr int32_t test_device_id{0};
  constexpr size_t test_max_buffer_pool_size{200};
  constexpr size_t test_max_slab_size = test_max_buffer_pool_size;
  constexpr size_t test_min_slab_size{100};

  buffer_mgr_ = createBufferMgr(test_device_id,
                                test_max_buffer_pool_size,
                                test_min_slab_size,
                                test_max_slab_size,
                                page_size_);

  auto buffer_1_size = test_max_slab_size / 4;
  auto buffer_1 = buffer_mgr_->createBuffer(test_chunk_key_, page_size_, buffer_1_size);
  buffer_1->unPin();

  auto buffer_2 = buffer_mgr_->createBuffer(
      test_chunk_key_2_, page_size_, test_max_slab_size - buffer_1_size);
  buffer_2->unPin();

  assertExpectedBufferMgrAttributes(test_max_slab_size, test_max_slab_size, 2);

  mock_parent_mgr_.setReserveSize(test_max_slab_size - buffer_1_size);
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;

  auto buffer = buffer_mgr_->getBuffer(test_chunk_key_, new_source_content.size());
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_));
  EXPECT_FALSE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));
  EXPECT_EQ(buffer->reservedSize(), test_max_slab_size - buffer_1_size);
  EXPECT_EQ(new_source_content.size(), buffer->size());

  std::vector<int8_t> read_content(buffer->size());
  buffer->read(read_content.data(), buffer->size());
  EXPECT_EQ(read_content, new_source_content);

  assertExpectedBufferMgrAttributes(
      test_max_slab_size - buffer_1_size, test_max_slab_size, 1);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_, buffer, new_source_content.size()}});
}

TEST_P(BufferMgrTest, GetBufferReentrantReserveCannotEvict) {
  constexpr int32_t test_device_id{0};
  constexpr size_t test_max_buffer_pool_size{200};
  constexpr size_t test_min_slab_size{100};
  constexpr size_t test_max_slab_size{200};

  buffer_mgr_ = createBufferMgr(test_device_id,
                                test_max_buffer_pool_size,
                                test_min_slab_size,
                                test_max_slab_size,
                                page_size_);

  createPinnedBuffers(2);
  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, test_max_slab_size, 2);

  mock_parent_mgr_.setReserveSize(test_max_slab_size);
  const auto& new_source_content = mock_parent_mgr_.buffer_content_;

  EXPECT_THROW(buffer_mgr_->getBuffer(test_chunk_key_2_, new_source_content.size()),
               OutOfMemory);
  EXPECT_TRUE(buffer_mgr_->isBufferOnDevice(test_chunk_key_2_));

  assertExpectedBufferMgrAttributes(2 * test_buffer_size_, test_max_slab_size, 2);
  auto segment_it = getSegmentAt(0, 1);
  assertParentMethodCalledWithParams(
      ParentMgrMethod::kFetchBuffer,
      {{test_chunk_key_2_, segment_it->buffer, new_source_content.size()}});
}

INSTANTIATE_TEST_SUITE_P(CpuAndGpuMgrs,
                         BufferMgrTest,
                         testing::Values(
#ifdef HAVE_CUDA
                             MgrType::GPU_MGR,
#endif
                             MgrType::CPU_MGR),
                         [](const auto& param_info) {
                           return ToString(param_info.param);
                         });

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
