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
#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/DataMgr.h"
#include "TestHelpers.h"

extern bool g_use_cpu_mem_pool_size_for_max_cpu_slab_size;

class DataMgrTest : public testing::Test {
 public:
  void SetUp() override { resetDataMgr(); }

  void TearDown() override {
    g_use_cpu_mem_pool_size_for_max_cpu_slab_size =
        orig_use_cpu_mem_pool_size_for_max_cpu_slab_size_;
    boost::filesystem::remove_all(data_mgr_path_);
  }

  virtual void resetDataMgr(size_t num_slabs = 1) {
    system_params_.max_cpu_slab_size = slab_size_;
    system_params_.default_cpu_slab_size = system_params_.max_cpu_slab_size;
    system_params_.min_cpu_slab_size = slab_size_;
    system_params_.cpu_buffer_mem_bytes = slab_size_ * num_slabs;
    resetDataMgr(system_params_);
  }

  void resetDataMgr(const SystemParameters& sys_params) {
    boost::filesystem::remove_all(data_mgr_path_);
    std::unique_ptr<CudaMgr_Namespace::CudaMgr> cuda_mgr;
#ifdef HAVE_CUDA
    try {
      cuda_mgr = std::make_unique<CudaMgr_Namespace::CudaMgr>(system_params_.num_gpus,
                                                              system_params_.start_gpu);
    } catch (...) {
      // Tests should still run on instances without GPUs.
    }
#endif

    data_mgr_ = std::make_unique<Data_Namespace::DataMgr>(data_mgr_path_,
                                                          sys_params,
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
    auto cd = std::make_unique<ColumnDescriptor>(
        key[1], key[2], "temp", SQLTypeInfo{kTINYINT}, key[0]);
    return Chunk_NS::Chunk::getChunk(
        cd.get(), data_mgr_.get(), key, MemoryLevel::CPU_LEVEL, 0, 4, 4);
  }

 protected:
  std::string data_mgr_path_{"./data_mgr_test_dir"};
  bool use_gpus_{true};
  size_t reserved_gpu_mem_{0};
  size_t num_reader_threads_{0};
  File_Namespace::DiskCacheConfig disk_cache_config_{};
  SystemParameters system_params_{};
  // The DataMgr's default page_size is 512, so we are setting the default slab size to
  // one page for simplicity.
  size_t slab_size_{512};
  std::unique_ptr<Data_Namespace::DataMgr> data_mgr_;
  const bool orig_use_cpu_mem_pool_size_for_max_cpu_slab_size_{
      g_use_cpu_mem_pool_size_for_max_cpu_slab_size};
};

TEST_F(DataMgrTest, ReuseWithPinnedGaps) {
  // This test is designed to catch ASAN memory leaks and therefore does not use explicit
  // assertions.
  resetDataMgr(2);
  auto chunk1 = writeChunkForKey({1, 1, 1, 1});  // pinned
  writeChunkForKey({1, 1, 1, 2});                // unpinned
  writeChunkForKey({1, 1, 1, 3});                // unpinned
}

TEST_F(DataMgrTest, UseCpuMemPoolSizeForMaxCpuSlabSizeNonMultipleOfPageSize) {
  g_use_cpu_mem_pool_size_for_max_cpu_slab_size = true;
  size_t page_size = 11;
  SystemParameters sys_params;
  sys_params.buffer_page_size = page_size;
  sys_params.min_cpu_slab_size = page_size;
  sys_params.max_cpu_slab_size = page_size * 2;
  sys_params.cpu_buffer_mem_bytes = (page_size * 3) + 1;
  resetDataMgr(sys_params);
}

TEST_F(DataMgrTest, CpuAndGpuMemPoolSizeNotMultipleOfPageSize) {
  size_t page_size = 11;
  SystemParameters sys_params;
  sys_params.buffer_page_size = page_size;
  sys_params.min_cpu_slab_size = page_size;
  sys_params.max_cpu_slab_size = page_size * 2;
  sys_params.default_cpu_slab_size = page_size * 2;
  sys_params.min_gpu_slab_size = page_size;
  sys_params.max_gpu_slab_size = page_size * 2;
  sys_params.default_gpu_slab_size = page_size * 2;

  sys_params.cpu_buffer_mem_bytes = (page_size * 3) + 1;
  sys_params.gpu_buffer_mem_bytes = (page_size * 3) + 1;
  resetDataMgr(sys_params);
}

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
