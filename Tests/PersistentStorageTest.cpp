/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <future>
#include <thread>

#include <boost/filesystem.hpp>
#include "DataMgr/PersistentStorageMgr/PersistentStorageMgr.h"
#include "DataMgrTestHelpers.h"
#include "Shared/scope.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>

#include "Catalog/Catalog.h"

const std::string data_path = "./tmp/" + shared::kDataDirectoryName;
extern bool g_enable_fsi;
extern bool g_enable_gpu_input_cpu_buffer_bypass;

namespace File_Namespace {
extern bool g_enable_native_storage_compression;
extern std::string g_native_storage_compression_codec;
extern size_t g_native_storage_compression_frame_size;
}  // namespace File_Namespace

using namespace foreign_storage;
using namespace File_Namespace;
using namespace TestHelpers;

class PersistentStorageMgrTest : public testing::Test {
 protected:
  inline static const std::string cache_path_ = "./test_foreign_data_cache";
  void TearDown() override { boost::filesystem::remove_all(cache_path_); }
};

TEST_F(PersistentStorageMgrTest, DiskCache_CustomPath) {
  PersistentStorageMgr psm(data_path, 0, {cache_path_, DiskCacheLevel::fsi});
  ASSERT_EQ(psm.getDiskCache()->getCacheDirectory(), cache_path_);
}

TEST_F(PersistentStorageMgrTest, DiskCache_InitializeWithoutCache) {
  PersistentStorageMgr psm(data_path, 0, {});
  ASSERT_EQ(psm.getDiskCache(), nullptr);
}

TEST_F(PersistentStorageMgrTest, ConcurrentCompressedNativeFetchesAreIndependent) {
  const std::string test_data_path{"./persistent_storage_concurrent_read_test"};
  boost::filesystem::remove_all(test_data_path);
  ScopeGuard cleanup = [&] { boost::filesystem::remove_all(test_data_path); };

  const auto saved_compression_enabled = g_enable_native_storage_compression;
  const auto saved_compression_codec = g_native_storage_compression_codec;
  const auto saved_compression_frame_size = g_native_storage_compression_frame_size;
  const auto saved_cpu_buffer_bypass = g_enable_gpu_input_cpu_buffer_bypass;
  ScopeGuard restore_flags = [&] {
    g_enable_native_storage_compression = saved_compression_enabled;
    g_native_storage_compression_codec = saved_compression_codec;
    g_native_storage_compression_frame_size = saved_compression_frame_size;
    g_enable_gpu_input_cpu_buffer_bypass = saved_cpu_buffer_bypass;
  };

  g_enable_native_storage_compression = true;
  g_native_storage_compression_codec = "snappy";
  g_native_storage_compression_frame_size = 64 * 1024;
  g_enable_gpu_input_cpu_buffer_bypass = true;

  PersistentStorageMgr psm(test_data_path, 4, {});
  const ChunkKey chunk_key{7, 11, 13, 0};
  std::vector<int32_t> expected_values(256 * 1024);
  for (size_t i = 0; i < expected_values.size(); ++i) {
    expected_values[i] = static_cast<int32_t>(i % 251);
  }
  TestBuffer source_buffer{expected_values};
  auto* native_buffer = dynamic_cast<FileBuffer*>(
      psm.putBuffer(chunk_key, &source_buffer, source_buffer.size()));
  ASSERT_NE(native_buffer, nullptr);
  ASSERT_TRUE(native_buffer->isSnappyStorageCompressed());
  psm.checkpoint(chunk_key[CHUNK_KEY_DB_IDX], chunk_key[CHUNK_KEY_TABLE_IDX]);

  constexpr size_t reader_count = 8;
  std::atomic<size_t> ready_count{0};
  std::atomic<bool> start{false};
  std::vector<std::future<bool>> readers;
  readers.reserve(reader_count);
  for (size_t reader_idx = 0; reader_idx < reader_count; ++reader_idx) {
    readers.emplace_back(std::async(std::launch::async, [&] {
      TestBuffer destination_buffer{SQLTypeInfo{kINT}};
      ready_count.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      psm.fetchBuffer(chunk_key, &destination_buffer, source_buffer.size());
      return destination_buffer.size() == source_buffer.size() &&
             source_buffer.compare(&destination_buffer, source_buffer.size());
    }));
  }
  while (ready_count.load(std::memory_order_acquire) != reader_count) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  for (size_t reader_idx = 0; reader_idx < readers.size(); ++reader_idx) {
    EXPECT_TRUE(readers[reader_idx].get()) << "reader " << reader_idx;
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  g_enable_fsi = true;
  int err{0};

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_fsi = false;
  return err;
}
