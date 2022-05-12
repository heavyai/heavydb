/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @file DataMgrTest.cpp
 * @brief Test suite for the DataMgr
 *
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "Catalog/ColumnDescriptor.h"
#include "CudaMgr/CudaMgr.h"
#include "DataMgr/Allocators/ArenaAllocator.h"
#include "DataMgr/BufferMgr/CpuBufferMgr/TieredCpuBufferMgr.h"
#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/DataMgr.h"
#include "TestHelpers.h"

#ifdef ENABLE_MEMKIND
extern bool g_enable_tiered_cpu_mem;
extern size_t g_pmem_size;
extern std::string g_pmem_path;
#endif

class DataMgrTest : public testing::Test {
 public:
  void SetUp() override { resetDataMgr(); }

  void TearDown() override { boost::filesystem::remove_all(data_mgr_path_); }

  virtual void resetDataMgr(size_t num_slabs = 1) {
    boost::filesystem::remove_all(data_mgr_path_);
    system_params_.max_cpu_slab_size = slab_size_;
    system_params_.min_cpu_slab_size = slab_size_;
    system_params_.cpu_buffer_mem_bytes = slab_size_ * num_slabs;
#ifdef ENABLE_MEMKIND
    g_pmem_size = slab_size_ * num_slabs;
#endif
    std::unique_ptr<CudaMgr_Namespace::CudaMgr> cuda_mgr;
    data_mgr_ = std::make_unique<Data_Namespace::DataMgr>(data_mgr_path_,
                                                          system_params_,
                                                          std::move(cuda_mgr),
                                                          use_gpus_,
                                                          reserved_gpu_mem_,
                                                          num_reader_threads_,
                                                          disk_cache_config_);
  }

  // Writes some data to disk through the FileMgr and then reads it into a CPU buffer via
  // the Chunk interface.  The Chunk interface is used because it will keep the cpu buffer
  // pinned for the lifetime of the Chunk during which time it is not-evictable.
  std::shared_ptr<Chunk_NS::Chunk> writeChunkForKey(const ChunkKey& key) {
    auto disk_buf = data_mgr_->createChunkBuffer(key, MemoryLevel::DISK_LEVEL);
    disk_buf->append(std::vector<int8_t>{1, 2, 3, 4}.data(), 4U);
    auto cd =
        std::make_unique<ColumnDescriptor>(key[1], key[2], "temp", SQLTypeInfo{kTINYINT});
    return Chunk_NS::Chunk::getChunk(
        cd.get(), data_mgr_.get(), key, MemoryLevel::CPU_LEVEL, 0, 4, 4);
  }

 protected:
  std::string data_mgr_path_{"./data_mgr_test_dir"};
  bool use_gpus_{false};
  size_t reserved_gpu_mem_{0};
  size_t num_reader_threads_{0};
  File_Namespace::DiskCacheConfig disk_cache_config_{};
  SystemParameters system_params_{};
  // The DataMgr's default page_size is 512, so we are setting the default slab size to
  // one page for simplicity.
  size_t slab_size_{512};
  std::unique_ptr<Data_Namespace::DataMgr> data_mgr_;
};

TEST_F(DataMgrTest, ReuseWithPinnedGaps) {
  // This test is designed to catch ASAN memory leaks and therefore does not use explicit
  // assertions.
  resetDataMgr(2);
  auto chunk1 = writeChunkForKey({1, 1, 1, 1});  // pinned
  writeChunkForKey({1, 1, 1, 2});                // unpinned
  writeChunkForKey({1, 1, 1, 3});                // unpinned
}

#ifdef ENABLE_MEMKIND
// Tests for the TieredCpuBufferMgr class.
// These tests set the DataMgr to use small slabs (one page) to force situations like
// exceeding an allocator's capacity and requiring eviction of slabs from the BufferMgr.
class TieredCpuBufferMgrTest : public DataMgrTest {
 public:
  void SetUp() override {
    if (g_pmem_path == "") {
      GTEST_SKIP() << "No PMEM path specified.";
    }
    g_enable_tiered_cpu_mem = true;
    g_pmem_size = slab_size_;
    DataMgrTest::SetUp();
  }

  void TearDown() override {
    DataMgrTest::TearDown();
    g_enable_tiered_cpu_mem = false;
    g_pmem_size = 0;
  }

  void resetDataMgr(size_t num_slabs = 1) override {
    DataMgrTest::resetDataMgr(num_slabs);
    tiered_buffer_mgr_ =
        dynamic_cast<Buffer_Namespace::TieredCpuBufferMgr*>(data_mgr_->getCpuBufferMgr());
  }

  uint32_t getAllocatorTierForChunk(const Chunk_NS::Chunk* chunk) {
    auto memory_type =
        tiered_buffer_mgr_
            ->getAllocatorForSlab(
                dynamic_cast<Buffer_Namespace::Buffer*>(chunk->getBuffer())->getSlabNum())
            ->getMemoryType();
    if (memory_type == Arena::MemoryType::DRAM) {
      return 0;
    } else if (memory_type == Arena::MemoryType::PMEM) {
      return 1;
    } else {
      UNREACHABLE() << "Unknown memory type";
    }
    return 0;
  }

 protected:
  Buffer_Namespace::TieredCpuBufferMgr* tiered_buffer_mgr_;
};

TEST_F(TieredCpuBufferMgrTest, AllocateInOrder) {
  // Two buffers will each allocate a new slab, so they should use new allocators for
  // each.
  for (auto i = 0U; i < 2; ++i) {
    auto chunk = writeChunkForKey({1, 1, 1, static_cast<int32_t>(i)});
    ASSERT_EQ(getAllocatorTierForChunk(chunk.get()), i);
  }
}

TEST_F(TieredCpuBufferMgrTest, MultipleSlabsPerAllocator) {
  resetDataMgr(3);
  // Each allocator can hold 3 slabs, so first three slabs should be allocator #1 and next
  // three allocator #2.
  for (auto i = 0U; i < 6; ++i) {
    auto chunk = writeChunkForKey({1, 1, 1, static_cast<int32_t>(i)});
    ASSERT_EQ(getAllocatorTierForChunk(chunk.get()), i / 3);
  }
}

TEST_F(TieredCpuBufferMgrTest, ReuseInOrder) {
  // First two buffers will allocate new slabs, so they should use new allocators for
  // each.
  for (auto i = 0U; i < 2; ++i) {
    auto chunk = writeChunkForKey({1, 1, 1, static_cast<int32_t>(i)});
    ASSERT_EQ(getAllocatorTierForChunk(chunk.get()), i % 2);
  }
  // After the first two we run out of space, so we will start reusing slabs.
  for (auto i = 2U; i < 6; ++i) {
    auto chunk = writeChunkForKey({1, 1, 1, static_cast<int32_t>(i)});
    ASSERT_EQ(getAllocatorTierForChunk(chunk.get()), i % 2);
  }
}

// Tests that we will prioritize the allocator order when evicting/reusing slabs.
TEST_F(TieredCpuBufferMgrTest, ReuseWithPinnedGaps) {
  resetDataMgr(2);
  auto chunk1 = writeChunkForKey({1, 1, 1, 1});  // pinned
  writeChunkForKey({1, 1, 1, 2});                // unpinned
  auto chunk2 = writeChunkForKey({1, 1, 1, 3});  // pinned
  writeChunkForKey({1, 1, 1, 4});                // unpinned

  // Each tier now has one pinned and one unpinned chunk so if we allocate two more chunks
  // they should go to separate tiers.
  auto chunk3 = writeChunkForKey({1, 1, 1, 5});
  auto chunk4 = writeChunkForKey({1, 1, 1, 6});

  ASSERT_EQ(getAllocatorTierForChunk(chunk3.get()), 0U);
  ASSERT_EQ(getAllocatorTierForChunk(chunk4.get()), 1U);
}

#endif

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

#ifdef ENABLE_MEMKIND
  // We only need option processing if we are specifying a pmem path.
  namespace po = boost::program_options;
  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all tests");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()(
      "pmem-path", po::value<std::string>(&g_pmem_path), "a path to app-direct pmem");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("pmem-path")) {
    g_pmem_path = boost::any_cast<std::string>(vm["pmem-path"].value());
  }
#endif

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
