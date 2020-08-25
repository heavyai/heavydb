/*
 * Copyright 2020 MapD Technologies, Inc.
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

#include "DBHandlerTestHelpers.h"
#include "DataMgr/ForeignStorage/ForeignStorageCache.h"
#include "DataMgr/ForeignStorage/ForeignStorageMgr.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>

#include "Catalog/Catalog.h"

const std::string data_path = "./tmp/mapd_data";
extern bool g_enable_fsi;
extern size_t foreign_cache_entry_limit;

using namespace foreign_storage;
using namespace File_Namespace;

static const std::string test_table_name = "test_cache_table";
static const std::string test_server_name = "test_cache_server";
static const std::string test_first_col_name = "col1";
static const std::string test_second_col_name = "col2";
static const std::string test_third_col_name = "col3";
static const ChunkKey chunk_key1 = {1, 1, 1, 0};
static const ChunkKey chunk_key2 = {1, 1, 2, 0};
static const ChunkKey chunk_key3 = {1, 1, 3, 0};
static const ChunkKey chunk_key_table2 = {1, 2, 1, 0};
static const ChunkKey table_prefix1 = {1, 1};

class TestBuffer : public AbstractBuffer {
 public:
  TestBuffer(const SQLTypeInfo sql_type) : AbstractBuffer(0, sql_type) {}
  TestBuffer(const std::vector<int8_t> bytes) : AbstractBuffer(0, kTINYINT) {
    write((int8_t*)bytes.data(), bytes.size());
  }
  ~TestBuffer() override {
    if (mem_ != nullptr) {
      free(mem_);
    }
  }

  void read(int8_t* const dst,
            const size_t num_bytes,
            const size_t offset = 0,
            const MemoryLevel dst_buffer_type = CPU_LEVEL,
            const int dst_device_id = -1) override {
    memcpy(dst, mem_ + offset, num_bytes);
  }

  void write(int8_t* src,
             const size_t num_bytes,
             const size_t offset = 0,
             const MemoryLevel src_buffer_type = CPU_LEVEL,
             const int src_device_id = -1) override {
    reserve(num_bytes + offset);
    memcpy(mem_ + offset, src, num_bytes);
    is_dirty_ = true;
    if (offset < size_) {
      is_updated_ = true;
    }
    if (offset + num_bytes > size_) {
      is_appended_ = true;
      size_ = offset + num_bytes;
    }
  }

  void reserve(size_t num_bytes) override {
    if (mem_ == nullptr) {
      mem_ = (int8_t*)malloc(num_bytes);
    } else {
      mem_ = (int8_t*)realloc(mem_, num_bytes);
    }
    size_ = num_bytes;
  }

  void append(int8_t* src,
              const size_t num_bytes,
              const MemoryLevel src_buffer_type,
              const int device_id) override {
    UNREACHABLE();
  }

  int8_t* getMemoryPtr() override { return mem_; }

  size_t pageCount() const override {
    UNREACHABLE();
    return 0;
  }

  size_t pageSize() const override {
    UNREACHABLE();
    return 0;
  }

  size_t size() const override { return size_; }

  size_t reservedSize() const override { return size_; }

  MemoryLevel getType() const override { return Data_Namespace::CPU_LEVEL; }

  bool compare(AbstractBuffer* buffer, size_t num_bytes) {
    int8_t left_array[num_bytes];
    int8_t right_array[num_bytes];
    read(left_array, num_bytes);
    buffer->read(right_array, num_bytes);
    if ((std::memcmp(left_array, right_array, num_bytes) == 0) &&
        (has_encoder == buffer->has_encoder)) {
      return true;
    }
    std::cerr << "buffers do not match:\n";
    for (size_t i = 0; i < num_bytes; ++i) {
      std::cerr << "a[" << i << "]: " << (int32_t)left_array[i] << " b[" << i
                << "]: " << (int32_t)right_array[i] << "\n";
    }
    return false;
  }

 protected:
  int8_t* mem_ = nullptr;
  size_t size_ = 0;
};

class ForeignStorageCacheUnitTest : public testing::Test {
 protected:
  inline static std::unique_ptr<GlobalFileMgr> gfm;
  inline static std::unique_ptr<ForeignStorageCache> cache;
  inline static std::string cache_path;
  static void SetUpTestSuite() {
    cache_path = "./tmp/mapd_data/test_foreign_data_cache";
    gfm = std::make_unique<GlobalFileMgr>(0, cache_path, 0);
    cache = std::make_unique<ForeignStorageCache>(gfm.get(), foreign_cache_entry_limit);
  }

  static void TearDownTestSuite() { boost::filesystem::remove_all(cache_path); }

  void SetUp() override { cache->clear(); }
};

TEST_F(ForeignStorageCacheUnitTest, CacheChunk) {
  ASSERT_EQ(cache->getCachedChunkIfExists(chunk_key1), nullptr);
  TestBuffer source_buffer{std::vector<int8_t>{1, 2, 3, 4}};
  cache->cacheChunk(chunk_key1, &source_buffer);
  AbstractBuffer* cached_buffer = cache->getCachedChunkIfExists(chunk_key1);
  ASSERT_NE(cached_buffer, nullptr);
  ASSERT_TRUE(source_buffer.compare(cached_buffer, 4));
}

TEST_F(ForeignStorageCacheUnitTest, CacheMetadata) {
  ASSERT_FALSE(cache->isMetadataCached(chunk_key1));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(table_prefix1));
  std::shared_ptr<ChunkMetadata> metadata =
      std::make_shared<ChunkMetadata>(kTINYINT, 0, 0, ChunkStats{});
  ChunkMetadataVector metadata_vec_source{std::make_pair(chunk_key1, metadata)};
  cache->cacheMetadataVec(metadata_vec_source);
  ASSERT_TRUE(cache->isMetadataCached(chunk_key1));
  ASSERT_TRUE(cache->hasCachedMetadataForKeyPrefix(table_prefix1));
  ChunkMetadataVector metadata_vec_cached{};
  cache->getCachedMetadataVecForKeyPrefix(metadata_vec_cached, chunk_key1);
  ASSERT_EQ(metadata_vec_cached.size(), 1U);
  ASSERT_EQ(metadata_vec_cached[0].second, metadata_vec_source[0].second);
}

TEST_F(ForeignStorageCacheUnitTest, HasCachedMetadataForKeyPrefix) {
  ASSERT_FALSE(cache->isMetadataCached(chunk_key1));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(table_prefix1));
  std::shared_ptr<ChunkMetadata> metadata =
      std::make_shared<ChunkMetadata>(kTINYINT, 0, 0, ChunkStats{});
  ChunkMetadataVector metadata_vec_source{std::make_pair(chunk_key_table2, metadata)};
  cache->cacheMetadataVec(metadata_vec_source);
  ASSERT_FALSE(cache->isMetadataCached(chunk_key1));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(table_prefix1));
  ASSERT_TRUE(cache->isMetadataCached(chunk_key_table2));
  ASSERT_TRUE(cache->hasCachedMetadataForKeyPrefix(chunk_key_table2));
}

TEST_F(ForeignStorageCacheUnitTest, GetCachedMetadataVecForKeyPrefix) {
  ASSERT_FALSE(cache->isMetadataCached(chunk_key1));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(table_prefix1));
  std::shared_ptr<ChunkMetadata> metadata =
      std::make_shared<ChunkMetadata>(kTINYINT, 0, 0, ChunkStats{});
  ChunkMetadataVector metadata_vec_source{std::make_pair(chunk_key1, metadata),
                                          std::make_pair(chunk_key2, metadata),
                                          std::make_pair(chunk_key_table2, metadata)};
  cache->cacheMetadataVec(metadata_vec_source);
  ASSERT_TRUE(cache->isMetadataCached(chunk_key1));
  ASSERT_TRUE(cache->hasCachedMetadataForKeyPrefix(table_prefix1));
  ChunkMetadataVector metadata_vec_cached{};
  cache->getCachedMetadataVecForKeyPrefix(metadata_vec_cached, table_prefix1);
  ASSERT_EQ(metadata_vec_cached.size(), 2U);
  ChunkMetadataVector col_meta_vec{};
  cache->getCachedMetadataVecForKeyPrefix(col_meta_vec, {1, 1, 1});
  ASSERT_EQ(col_meta_vec.size(), 1U);
}

TEST_F(ForeignStorageCacheUnitTest, ClearForTablePrefix) {
  TestBuffer test_buffer1{std::vector<int8_t>{1}};
  TestBuffer test_buffer2{std::vector<int8_t>{1}};
  TestBuffer test_buffer3{std::vector<int8_t>{1}};
  cache->cacheChunk(chunk_key1, &test_buffer1);
  cache->cacheChunk(chunk_key2, &test_buffer2);
  cache->cacheChunk(chunk_key_table2, &test_buffer3);
  ASSERT_EQ(cache->getNumCachedChunks(), 3U);
  ASSERT_TRUE(gfm->isBufferOnDevice(chunk_key1));
  ASSERT_TRUE(gfm->isBufferOnDevice(chunk_key2));
  ASSERT_TRUE(gfm->isBufferOnDevice(chunk_key_table2));
  std::shared_ptr<ChunkMetadata> metadata =
      std::make_shared<ChunkMetadata>(kTINYINT, 0, 0, ChunkStats{});
  ChunkMetadataVector metadata_vec_source{std::make_pair(chunk_key1, metadata),
                                          std::make_pair(chunk_key2, metadata),
                                          std::make_pair(chunk_key_table2, metadata)};
  cache->cacheMetadataVec(metadata_vec_source);
  ASSERT_EQ(cache->getNumCachedMetadata(), 3U);
  cache->clearForTablePrefix(table_prefix1);
  ASSERT_EQ(cache->getNumCachedChunks(), 1U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 1U);
  ASSERT_FALSE(gfm->isBufferOnDevice(chunk_key1));
  ASSERT_FALSE(gfm->isBufferOnDevice(chunk_key2));
  ASSERT_TRUE(gfm->isBufferOnDevice(chunk_key_table2));
}

TEST_F(ForeignStorageCacheUnitTest, Clear) {
  TestBuffer test_buffer1{std::vector<int8_t>{1}};
  TestBuffer test_buffer2{std::vector<int8_t>{1}};
  TestBuffer test_buffer3{std::vector<int8_t>{1}};
  cache->cacheChunk(chunk_key1, &test_buffer1);
  cache->cacheChunk(chunk_key2, &test_buffer2);
  cache->cacheChunk(chunk_key_table2, &test_buffer3);
  ASSERT_TRUE(gfm->isBufferOnDevice(chunk_key1));
  ASSERT_TRUE(gfm->isBufferOnDevice(chunk_key2));
  ASSERT_TRUE(gfm->isBufferOnDevice(chunk_key_table2));
  ASSERT_EQ(cache->getNumCachedChunks(), 3U);
  std::shared_ptr<ChunkMetadata> metadata =
      std::make_shared<ChunkMetadata>(kTINYINT, 0, 0, ChunkStats{});
  ChunkMetadataVector metadata_vec_source{std::make_pair(chunk_key1, metadata),
                                          std::make_pair(chunk_key2, metadata),
                                          std::make_pair(chunk_key_table2, metadata)};
  cache->cacheMetadataVec(metadata_vec_source);
  ASSERT_EQ(cache->getNumCachedMetadata(), 3U);
  cache->clear();
  ASSERT_EQ(cache->getNumCachedChunks(), 0U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 0U);
  ASSERT_FALSE(gfm->isBufferOnDevice(chunk_key1));
  ASSERT_FALSE(gfm->isBufferOnDevice(chunk_key2));
  ASSERT_FALSE(gfm->isBufferOnDevice(chunk_key_table2));
}

TEST_F(ForeignStorageCacheUnitTest, SetLimit) {
  size_t old_limit = cache->getLimit();
  size_t new_limit = 1;
  TestBuffer test_buffer1{std::vector<int8_t>{1}};
  TestBuffer test_buffer2{std::vector<int8_t>{1}};
  TestBuffer test_buffer3{std::vector<int8_t>{1}};
  cache->cacheChunk(chunk_key1, &test_buffer1);
  cache->cacheChunk(chunk_key2, &test_buffer2);
  ASSERT_EQ(cache->getNumCachedChunks(), 2U);
  cache->setLimit(new_limit);
  ASSERT_EQ(cache->getLimit(), new_limit);
  ASSERT_EQ(cache->getNumCachedChunks(), new_limit);
  cache->cacheChunk(chunk_key3, &test_buffer3);
  ASSERT_EQ(cache->getLimit(), new_limit);
  ASSERT_EQ(cache->getNumCachedChunks(), new_limit);
  cache->setLimit(old_limit);
}

TEST_F(ForeignStorageCacheUnitTest, CachePath) {
  ASSERT_EQ(cache->getCachedChunkIfExists(chunk_key1), nullptr);
  ASSERT_FALSE(gfm->isBufferOnDevice(chunk_key1));
  TestBuffer source_buffer{std::vector<int8_t>{1, 2, 3, 4}};
  cache->cacheChunk(chunk_key1, &source_buffer);
  ASSERT_TRUE(gfm->isBufferOnDevice(chunk_key1));
  ASSERT_TRUE(boost::filesystem::exists(cache_path + "/table_1_1/0." +
                                        to_string(gfm->getDefaultPageSize()) + ".mapd"));
}

class ForeignStorageCacheLRUTest : public testing::Test {};
TEST_F(ForeignStorageCacheLRUTest, Basic) {
  LRUEvictionAlgorithm lru_alg{};
  lru_alg.touchChunk(chunk_key1);
  lru_alg.touchChunk(chunk_key2);
  lru_alg.touchChunk(chunk_key3);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key1);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key2);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key3);
  ASSERT_THROW(lru_alg.evictNextChunk(), NoEntryFoundException);
}

TEST_F(ForeignStorageCacheLRUTest, AfterEvict) {
  LRUEvictionAlgorithm lru_alg{};
  lru_alg.touchChunk(chunk_key1);
  lru_alg.touchChunk(chunk_key2);
  lru_alg.touchChunk(chunk_key3);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key1);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key2);
  lru_alg.touchChunk(chunk_key2);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key3);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key2);
  ASSERT_THROW(lru_alg.evictNextChunk(), NoEntryFoundException);
}

TEST_F(ForeignStorageCacheLRUTest, RemoveChunk) {
  LRUEvictionAlgorithm lru_alg{};
  lru_alg.touchChunk(chunk_key1);
  lru_alg.touchChunk(chunk_key2);
  lru_alg.touchChunk(chunk_key3);
  lru_alg.removeChunk(chunk_key2);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key1);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key3);
  ASSERT_THROW(lru_alg.evictNextChunk(), NoEntryFoundException);
}

TEST_F(ForeignStorageCacheLRUTest, Reorder) {
  LRUEvictionAlgorithm lru_alg{};
  lru_alg.touchChunk(chunk_key1);
  lru_alg.touchChunk(chunk_key2);
  lru_alg.touchChunk(chunk_key3);
  lru_alg.touchChunk(chunk_key1);
  lru_alg.touchChunk(chunk_key2);
  lru_alg.touchChunk(chunk_key1);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key3);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key2);
  ASSERT_EQ(lru_alg.evictNextChunk(), chunk_key1);
  ASSERT_THROW(lru_alg.evictNextChunk(), NoEntryFoundException);
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
