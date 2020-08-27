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
#include "DataMgr/PersistentStorageMgr/PersistentStorageMgr.h"
#include "DataMgrTestHelpers.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>

#include "Catalog/Catalog.h"

const std::string data_path = "./tmp/mapd_data";
extern bool g_enable_fsi;

using namespace foreign_storage;
using namespace File_Namespace;
using namespace TestHelpers;

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

class ForeignStorageCacheUnitTest : public testing::Test {
 protected:
  inline static GlobalFileMgr* gfm_;
  inline static std::unique_ptr<ForeignStorageCache> cache_;
  inline static const std::string cache_path_ = "./test_foreign_data_cache";
  static void SetUpTestSuite() {
    boost::filesystem::remove_all(cache_path_);
    cache_ = std::make_unique<ForeignStorageCache>(cache_path_, 0, 1024);
    gfm_ = cache_->getGlobalFileMgr();
  }

  static void TearDownTestSuite() { boost::filesystem::remove_all(cache_path_); }

  void SetUp() override { cache_->clear(); }
};

TEST_F(ForeignStorageCacheUnitTest, CacheChunk) {
  ASSERT_EQ(cache_->getCachedChunkIfExists(chunk_key1), nullptr);
  TestBuffer source_buffer{std::vector<int8_t>{1, 2, 3, 4}};
  cache_->cacheChunk(chunk_key1, &source_buffer);
  AbstractBuffer* cached_buffer = cache_->getCachedChunkIfExists(chunk_key1);
  ASSERT_NE(cached_buffer, nullptr);
  ASSERT_TRUE(source_buffer.compare(cached_buffer, 4));
}

TEST_F(ForeignStorageCacheUnitTest, CacheMetadata) {
  ASSERT_FALSE(cache_->isMetadataCached(chunk_key1));
  ASSERT_FALSE(cache_->hasCachedMetadataForKeyPrefix(table_prefix1));
  std::shared_ptr<ChunkMetadata> metadata =
      std::make_shared<ChunkMetadata>(kTINYINT, 0, 0, ChunkStats{});
  ChunkMetadataVector metadata_vec_source{std::make_pair(chunk_key1, metadata)};
  cache_->cacheMetadataVec(metadata_vec_source);
  ASSERT_TRUE(cache_->isMetadataCached(chunk_key1));
  ASSERT_TRUE(cache_->hasCachedMetadataForKeyPrefix(table_prefix1));
  ChunkMetadataVector metadata_vec_cached{};
  cache_->getCachedMetadataVecForKeyPrefix(metadata_vec_cached, chunk_key1);
  ASSERT_EQ(metadata_vec_cached.size(), 1U);
  ASSERT_EQ(metadata_vec_cached[0].second, metadata_vec_source[0].second);
}

TEST_F(ForeignStorageCacheUnitTest, HasCachedMetadataForKeyPrefix) {
  ASSERT_FALSE(cache_->isMetadataCached(chunk_key1));
  ASSERT_FALSE(cache_->hasCachedMetadataForKeyPrefix(table_prefix1));
  std::shared_ptr<ChunkMetadata> metadata =
      std::make_shared<ChunkMetadata>(kTINYINT, 0, 0, ChunkStats{});
  ChunkMetadataVector metadata_vec_source{std::make_pair(chunk_key_table2, metadata)};
  cache_->cacheMetadataVec(metadata_vec_source);
  ASSERT_FALSE(cache_->isMetadataCached(chunk_key1));
  ASSERT_FALSE(cache_->hasCachedMetadataForKeyPrefix(table_prefix1));
  ASSERT_TRUE(cache_->isMetadataCached(chunk_key_table2));
  ASSERT_TRUE(cache_->hasCachedMetadataForKeyPrefix(chunk_key_table2));
}

TEST_F(ForeignStorageCacheUnitTest, GetCachedMetadataVecForKeyPrefix) {
  ASSERT_FALSE(cache_->isMetadataCached(chunk_key1));
  ASSERT_FALSE(cache_->hasCachedMetadataForKeyPrefix(table_prefix1));
  std::shared_ptr<ChunkMetadata> metadata =
      std::make_shared<ChunkMetadata>(kTINYINT, 0, 0, ChunkStats{});
  ChunkMetadataVector metadata_vec_source{std::make_pair(chunk_key1, metadata),
                                          std::make_pair(chunk_key2, metadata),
                                          std::make_pair(chunk_key_table2, metadata)};
  cache_->cacheMetadataVec(metadata_vec_source);
  ASSERT_TRUE(cache_->isMetadataCached(chunk_key1));
  ASSERT_TRUE(cache_->hasCachedMetadataForKeyPrefix(table_prefix1));
  ChunkMetadataVector metadata_vec_cached{};
  cache_->getCachedMetadataVecForKeyPrefix(metadata_vec_cached, table_prefix1);
  ASSERT_EQ(metadata_vec_cached.size(), 2U);
  ChunkMetadataVector col_meta_vec{};
  cache_->getCachedMetadataVecForKeyPrefix(col_meta_vec, {1, 1, 1});
  ASSERT_EQ(col_meta_vec.size(), 1U);
}

TEST_F(ForeignStorageCacheUnitTest, ClearForTablePrefix) {
  TestBuffer test_buffer1{std::vector<int8_t>{1}};
  TestBuffer test_buffer2{std::vector<int8_t>{1}};
  TestBuffer test_buffer3{std::vector<int8_t>{1}};
  cache_->cacheChunk(chunk_key1, &test_buffer1);
  cache_->cacheChunk(chunk_key2, &test_buffer2);
  cache_->cacheChunk(chunk_key_table2, &test_buffer3);
  ASSERT_EQ(cache_->getNumCachedChunks(), 3U);
  ASSERT_TRUE(gfm_->isBufferOnDevice(chunk_key1));
  ASSERT_TRUE(gfm_->isBufferOnDevice(chunk_key2));
  ASSERT_TRUE(gfm_->isBufferOnDevice(chunk_key_table2));
  std::shared_ptr<ChunkMetadata> metadata =
      std::make_shared<ChunkMetadata>(kTINYINT, 0, 0, ChunkStats{});
  ChunkMetadataVector metadata_vec_source{std::make_pair(chunk_key1, metadata),
                                          std::make_pair(chunk_key2, metadata),
                                          std::make_pair(chunk_key_table2, metadata)};
  cache_->cacheMetadataVec(metadata_vec_source);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 3U);
  cache_->clearForTablePrefix(table_prefix1);
  ASSERT_EQ(cache_->getNumCachedChunks(), 1U);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 1U);
  ASSERT_FALSE(gfm_->isBufferOnDevice(chunk_key1));
  ASSERT_FALSE(gfm_->isBufferOnDevice(chunk_key2));
  ASSERT_TRUE(gfm_->isBufferOnDevice(chunk_key_table2));
}

TEST_F(ForeignStorageCacheUnitTest, Clear) {
  TestBuffer test_buffer1{std::vector<int8_t>{1}};
  TestBuffer test_buffer2{std::vector<int8_t>{1}};
  TestBuffer test_buffer3{std::vector<int8_t>{1}};
  cache_->cacheChunk(chunk_key1, &test_buffer1);
  cache_->cacheChunk(chunk_key2, &test_buffer2);
  cache_->cacheChunk(chunk_key_table2, &test_buffer3);
  ASSERT_TRUE(gfm_->isBufferOnDevice(chunk_key1));
  ASSERT_TRUE(gfm_->isBufferOnDevice(chunk_key2));
  ASSERT_TRUE(gfm_->isBufferOnDevice(chunk_key_table2));
  ASSERT_EQ(cache_->getNumCachedChunks(), 3U);
  std::shared_ptr<ChunkMetadata> metadata =
      std::make_shared<ChunkMetadata>(kTINYINT, 0, 0, ChunkStats{});
  ChunkMetadataVector metadata_vec_source{std::make_pair(chunk_key1, metadata),
                                          std::make_pair(chunk_key2, metadata),
                                          std::make_pair(chunk_key_table2, metadata)};
  cache_->cacheMetadataVec(metadata_vec_source);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 3U);
  cache_->clear();
  ASSERT_EQ(cache_->getNumCachedChunks(), 0U);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 0U);
  ASSERT_FALSE(gfm_->isBufferOnDevice(chunk_key1));
  ASSERT_FALSE(gfm_->isBufferOnDevice(chunk_key2));
  ASSERT_FALSE(gfm_->isBufferOnDevice(chunk_key_table2));
}

TEST_F(ForeignStorageCacheUnitTest, SetLimit) {
  size_t old_limit = cache_->getLimit();
  size_t new_limit = 1;
  TestBuffer test_buffer1{std::vector<int8_t>{1}};
  TestBuffer test_buffer2{std::vector<int8_t>{1}};
  TestBuffer test_buffer3{std::vector<int8_t>{1}};
  cache_->cacheChunk(chunk_key1, &test_buffer1);
  cache_->cacheChunk(chunk_key2, &test_buffer2);
  ASSERT_EQ(cache_->getNumCachedChunks(), 2U);
  cache_->setLimit(new_limit);
  ASSERT_EQ(cache_->getLimit(), new_limit);
  ASSERT_EQ(cache_->getNumCachedChunks(), new_limit);
  cache_->cacheChunk(chunk_key3, &test_buffer3);
  ASSERT_EQ(cache_->getLimit(), new_limit);
  ASSERT_EQ(cache_->getNumCachedChunks(), new_limit);
  cache_->setLimit(old_limit);
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

class ForeignStorageCacheFileTest : public testing::Test {
 protected:
  std::string cache_path_;
  void TearDown() override { boost::filesystem::remove_all(cache_path_); }
};

TEST_F(ForeignStorageCacheFileTest, FileBlocksPath) {
  cache_path_ = "./test_foreign_data_cache";
  boost::filesystem::remove_all(cache_path_);
  ASSERT_FALSE(boost::filesystem::exists(cache_path_));
  boost::filesystem::ofstream tmp_file(cache_path_);
  tmp_file << "1";
  tmp_file.close();
  try {
    ForeignStorageCache cache{cache_path_, 0, 1024};
    FAIL() << "An exception should have been thrown for this testcase";
  } catch (std::runtime_error& e) {
    ASSERT_EQ(e.what(),
              "cache path \"" + cache_path_ +
                  "\" is not a directory.  Please specify a valid directory "
                  "with --disk_cache_path=<path>, or use the default location.");
  }
}

TEST_F(ForeignStorageCacheFileTest, ExistingDir) {
  cache_path_ = "./test_foreign_data_cache";
  boost::filesystem::remove_all(cache_path_);
  boost::filesystem::create_directory(cache_path_);
  ForeignStorageCache cache{cache_path_, 0, 1024};
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
