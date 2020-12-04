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

#include "DataMgr/ForeignStorage/ForeignStorageCache.h"
#include "DataMgr/ForeignStorage/ForeignStorageMgr.h"
#include "DataMgr/PersistentStorageMgr/PersistentStorageMgr.h"
#include "DataMgrTestHelpers.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>

#include "Catalog/Catalog.h"

extern bool g_enable_fsi;

using namespace foreign_storage;
using namespace File_Namespace;
using namespace TestHelpers;

static const std::string data_path = "./tmp/mapd_data";
static const size_t cache_default_size = 21474836480;
static const size_t cache_minimum_size = 536870912;
static const std::string test_table_name = "test_cache_table";
static const std::string test_server_name = "test_cache_server";
static const std::string test_first_col_name = "col1";
static const std::string test_second_col_name = "col2";
static const std::string test_third_col_name = "col3";
static const ChunkKey chunk_key1 = {1, 1, 1, 0};
static const ChunkKey chunk_key2 = {1, 1, 2, 0};
static const ChunkKey chunk_key3 = {1, 1, 3, 0};
static const ChunkKey chunk_key4 = {1, 1, 4, 0};
static const ChunkKey chunk_key5 = {1, 1, 5, 0};
static const ChunkKey chunk_key_table2 = {1, 2, 1, 0};
static const ChunkKey table_prefix1 = {1, 1};
static const ChunkKey table_prefix2 = {1, 2};

class ForeignStorageCacheUnitTest : public testing::Test {
 protected:
  inline static GlobalFileMgr* gfm_;
  inline static std::unique_ptr<ForeignStorageCache> cache_;
  inline static std::string cache_path_;

  template <typename T>
  struct ChunkWrapper {
    std::unique_ptr<ColumnDescriptor> cd;
    std::unique_ptr<TestBuffer> test_buf;
    std::unique_ptr<TestBuffer> index_buf;
    std::unique_ptr<DataBlockPtr> data_block;
    std::unique_ptr<std::vector<T>> data_vec;
    std::unique_ptr<Chunk_NS::Chunk> chunk;
    ChunkWrapper(const SQLTypeInfo& type, const std::vector<T>& data) {
      cd = std::make_unique<ColumnDescriptor>();
      cd->columnType = type;
      test_buf = std::make_unique<TestBuffer>(type);
      chunk = std::make_unique<Chunk_NS::Chunk>(test_buf.get(), nullptr, cd.get());
      data_block = std::make_unique<DataBlockPtr>();
      data_vec = std::make_unique<std::vector<T>>(data);
      data_block->numbersPtr = reinterpret_cast<int8_t*>(data_vec->data());
      chunk->initEncoder();
      chunk->appendData(*data_block, data_vec->size(), 0);
    }

    void cacheMetadata(const ChunkKey& chunk_key) {
      std::shared_ptr<ChunkMetadata> cached_meta = std::make_shared<ChunkMetadata>();
      chunk->getBuffer()->getEncoder()->getMetadata(cached_meta);
      cache_->cacheMetadataVec({std::make_pair(chunk_key, cached_meta)});
    }

    void cacheChunk(const ChunkKey& chunk_key) {
      auto buffer_map = cache_->getChunkBuffersForCaching({chunk_key});
      buffer_map[chunk_key]->write(chunk->getBuffer()->getMemoryPtr(),
                                   chunk->getBuffer()->size());
      buffer_map[chunk_key]->syncEncoder(chunk->getBuffer());
      cache_->cacheTableChunks({chunk_key});
    }

    void cacheMetadataThenChunk(const ChunkKey& chunk_key) {
      cacheMetadata(chunk_key);
      cacheChunk(chunk_key);
    }
  };

  template <typename T>
  std::vector<ArrayDatum> convertToArrayDatum(std::vector<std::vector<T>> data,
                                              ArrayDatum null_datum) {
    std::vector<ArrayDatum> datums;
    for (auto& array : data) {
      if (array.size()) {
        size_t num_bytes = array.size() * sizeof(T);
        datums.push_back(
            ArrayDatum(num_bytes, reinterpret_cast<int8_t*>(array.data()), false));
      } else {
        datums.push_back(null_datum);
      }
    }
    return datums;
  }

  struct CacheLimitScope {
    uint64_t old_limit;
    CacheLimitScope(const uint64_t limit) {
      old_limit = cache_->getLimit();
      cache_->setLimit(limit);
    }
    ~CacheLimitScope() { cache_->setLimit(old_limit); }
  };

  std::shared_ptr<ChunkMetadata> createMetadata(const size_t num_bytes,
                                                const size_t num_elements,
                                                const int32_t in_min,
                                                const int32_t in_max,
                                                const bool has_nulls) {
    Datum min, max;
    min.intval = in_min;
    max.intval = in_max;
    return std::make_shared<ChunkMetadata>(
        kINT, num_bytes, num_elements, ChunkStats{min, max, has_nulls});
  }

  static void assertMetadataEqual(const std::shared_ptr<ChunkMetadata> left_metadata,
                                  const std::shared_ptr<ChunkMetadata> right_metadata) {
    ASSERT_EQ(*left_metadata, *right_metadata) << left_metadata->dump() << "\n"
                                               << right_metadata->dump() << "\n";
  }

  void evictChunksKeepMetadata() {
    size_t old_limit = cache_->getLimit();
    cache_->setLimit(0);
    cache_->setLimit(old_limit);
  }

  static void reinitializeCache(std::unique_ptr<ForeignStorageCache>& cache,
                                GlobalFileMgr*& gfm,
                                const DiskCacheConfig& config) {
    cache = std::make_unique<ForeignStorageCache>(config);
    gfm = cache->getGlobalFileMgr();
  }

  static void SetUpTestSuite() {
    cache_path_ = "./tmp/mapd_data/test_foreign_data_cache";
    boost::filesystem::remove_all(cache_path_);
    reinitializeCache(cache_, gfm_, {cache_path_, DiskCacheLevel::fsi});
  }

  static void TearDownTestSuite() { boost::filesystem::remove_all(cache_path_); }

  void SetUp() override {
    cache_->clear();
    ASSERT_EQ(gfm_->getNumChunks(), 0U);
    ASSERT_EQ(cache_->getNumCachedChunks(), 0U);
    ASSERT_EQ(cache_->getNumCachedMetadata(), 0U);
  }

  void TearDown() override {}
};

namespace {
struct PrintToStringParamName {
  std::string operator()(const ::testing::TestParamInfo<SQLTypeInfo>& info) const {
    std::string str = info.param.get_type_name();
    return str.substr(0, str.find("("));
  }
};
}  // namespace

TEST_F(ForeignStorageCacheUnitTest, CacheMetadata) {
  ChunkWrapper<int32_t> chunk_wrapper{kINT, {1, 2, 3, 4}};
  chunk_wrapper.cacheMetadata(chunk_key1);
  ASSERT_EQ(cache_->getCachedChunkIfExists(chunk_key1), nullptr);

  ChunkMetadataVector metadata_vec_cached{};
  cache_->getCachedMetadataVecForKeyPrefix(metadata_vec_cached, chunk_key1);
  assertMetadataEqual(metadata_vec_cached[0].second, createMetadata(16, 4, 1, 4, false));
  ASSERT_EQ(cache_->getCachedChunkIfExists(chunk_key1), nullptr);
}

TEST_F(ForeignStorageCacheUnitTest, CacheMetadata_Empty) {
  ChunkWrapper<int32_t> chunk_wrapper{kINT, {}};
  chunk_wrapper.cacheMetadata(chunk_key1);
  ASSERT_EQ(cache_->getCachedChunkIfExists(chunk_key1), nullptr);

  ChunkMetadataVector metadata_vec_cached{};
  cache_->getCachedMetadataVecForKeyPrefix(metadata_vec_cached, chunk_key1);
  assertMetadataEqual(metadata_vec_cached[0].second,
                      createMetadata(0,
                                     0,
                                     std::numeric_limits<int32_t>::max(),
                                     std::numeric_limits<int32_t>::lowest(),
                                     false));
  ASSERT_EQ(cache_->getCachedChunkIfExists(chunk_key1), nullptr);
}

TEST_F(ForeignStorageCacheUnitTest, CacheMetadata_Update) {
  ChunkWrapper<int32_t> chunk_wrapper{kINT, {1, 2, 3, 4}};
  chunk_wrapper.cacheMetadata(chunk_key1);

  ChunkWrapper<int32_t> overwrite_wrapper{kINT, {5, 6}};
  overwrite_wrapper.cacheMetadata(chunk_key1);
  ASSERT_EQ(cache_->getCachedChunkIfExists(chunk_key1), nullptr);

  ChunkMetadataVector metadata_vec_cached{};
  cache_->getCachedMetadataVecForKeyPrefix(metadata_vec_cached, chunk_key1);
  assertMetadataEqual(metadata_vec_cached[0].second, createMetadata(8, 2, 5, 6, false));
  ASSERT_EQ(cache_->getCachedChunkIfExists(chunk_key1), nullptr);
}

TEST_F(ForeignStorageCacheUnitTest, CacheChunk) {
  ChunkWrapper<int32_t> chunk_wrapper{kINT, {1, 2, 3, 4}};
  chunk_wrapper.cacheMetadataThenChunk(chunk_key1);
  AbstractBuffer* cached_buf = cache_->getCachedChunkIfExists(chunk_key1);
  ASSERT_NE(cached_buf, nullptr);
  ASSERT_TRUE(chunk_wrapper.test_buf->compare(cached_buf, 16));
}

TEST_F(ForeignStorageCacheUnitTest, UpdateMetadata_ClearsChunk) {
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, {1}};
  ChunkWrapper<int32_t> chunk_wrapper2{kINT, {2}};
  chunk_wrapper1.cacheMetadataThenChunk(chunk_key1);
  chunk_wrapper2.cacheMetadata(chunk_key1);
  ASSERT_EQ(cache_->getCachedChunkIfExists(chunk_key1), nullptr);
}

TEST_F(ForeignStorageCacheUnitTest, UpdateMetadata_KeepChunkWithSameMeta) {
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, {1, 2, 1}};
  ChunkWrapper<int32_t> chunk_wrapper2{kINT, {1, 2, 2}};
  chunk_wrapper1.cacheMetadataThenChunk(chunk_key1);
  chunk_wrapper2.cacheMetadata(chunk_key1);
  ASSERT_NE(cache_->getCachedChunkIfExists(chunk_key1), nullptr);
}

TEST_F(ForeignStorageCacheUnitTest, UpdateMetadata_UpdatesMetadata) {
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, {1}};
  ChunkWrapper<int32_t> chunk_wrapper2{kINT, {2}};
  chunk_wrapper1.cacheMetadataThenChunk(chunk_key1);
  chunk_wrapper2.cacheMetadata(chunk_key1);
  ChunkMetadataVector metadata_vec_cached{};
  cache_->getCachedMetadataVecForKeyPrefix(metadata_vec_cached, table_prefix1);
  assertMetadataEqual(metadata_vec_cached[0].second, createMetadata(4, 1, 2, 2, false));
}

TEST_F(ForeignStorageCacheUnitTest, UpdateMetadataUsingChunk) {
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, {1}};
  ChunkWrapper<int32_t> chunk_wrapper2{kINT, {2}};
  chunk_wrapper1.cacheMetadata(chunk_key1);
  chunk_wrapper2.cacheChunk(chunk_key1);
  ChunkMetadataVector metadata_vec_cached{};
  cache_->getCachedMetadataVecForKeyPrefix(metadata_vec_cached, table_prefix1);
  assertMetadataEqual(metadata_vec_cached[0].second, createMetadata(4, 1, 2, 2, false));
}

TEST_F(ForeignStorageCacheUnitTest, CacheChunk_CacheFull) {
  CacheLimitScope scope{cache_minimum_size};
  std::vector<int32_t> full_frag_vec(32000000);
  for (size_t i = 0; i < full_frag_vec.size(); ++i) {
    full_frag_vec[i] = i;
  }
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, full_frag_vec};
  chunk_wrapper1.cacheMetadataThenChunk(chunk_key1);
  ChunkWrapper<int32_t> chunk_wrapper2{kINT, full_frag_vec};
  chunk_wrapper2.cacheMetadataThenChunk(chunk_key2);
  ChunkWrapper<int32_t> chunk_wrapper3{kINT, full_frag_vec};
  chunk_wrapper3.cacheMetadataThenChunk(chunk_key3);
  ChunkWrapper<int32_t> chunk_wrapper4{kINT, full_frag_vec};
  chunk_wrapper4.cacheMetadataThenChunk(chunk_key4);
  ChunkWrapper<int32_t> chunk_wrapper5{kINT, full_frag_vec};
  chunk_wrapper5.cacheMetadataThenChunk(chunk_key5);
  ASSERT_EQ(cache_->getNumCachedChunks(), 4U);
  ASSERT_EQ(cache_->getCachedChunkIfExists(chunk_key1), nullptr);
  ASSERT_NE(cache_->getCachedChunkIfExists(chunk_key2), nullptr);
  ASSERT_NE(cache_->getCachedChunkIfExists(chunk_key3), nullptr);
  ASSERT_NE(cache_->getCachedChunkIfExists(chunk_key4), nullptr);
  ASSERT_NE(cache_->getCachedChunkIfExists(chunk_key5), nullptr);
}

TEST_F(ForeignStorageCacheUnitTest, HasCachedMetadataForKeyPrefix) {
  ASSERT_FALSE(cache_->hasCachedMetadataForKeyPrefix(table_prefix1));
  ASSERT_FALSE(cache_->hasCachedMetadataForKeyPrefix(table_prefix2));
  ChunkWrapper<int32_t> chunk_wrapper{kINT, {1, 2, 3, 4}};
  chunk_wrapper.cacheMetadata(chunk_key1);
  ASSERT_TRUE(cache_->hasCachedMetadataForKeyPrefix(table_prefix1));
  ASSERT_FALSE(cache_->hasCachedMetadataForKeyPrefix(table_prefix2));
}

TEST_F(ForeignStorageCacheUnitTest, GetCachedMetadataVecForKeyPrefix) {
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, {1}};
  ChunkWrapper<int32_t> chunk_wrapper2{kINT, {2}};
  ChunkWrapper<int32_t> chunk_wrapper3{kINT, {3}};
  chunk_wrapper1.cacheMetadata(chunk_key1);
  chunk_wrapper2.cacheMetadata(chunk_key2);
  chunk_wrapper3.cacheMetadata(chunk_key_table2);
  ChunkMetadataVector metadata_vec_cached{};
  cache_->getCachedMetadataVecForKeyPrefix(metadata_vec_cached, table_prefix1);
  ASSERT_EQ(metadata_vec_cached.size(), 2U);
  assertMetadataEqual(metadata_vec_cached[0].second, createMetadata(4, 1, 1, 1, false));
  assertMetadataEqual(metadata_vec_cached[1].second, createMetadata(4, 1, 2, 2, false));

  ChunkMetadataVector col_meta_vec{};
  cache_->getCachedMetadataVecForKeyPrefix(col_meta_vec, {1, 1, 2});
  ASSERT_EQ(col_meta_vec.size(), 1U);
  assertMetadataEqual(col_meta_vec[0].second, createMetadata(4, 1, 2, 2, false));
}

TEST_F(ForeignStorageCacheUnitTest, ClearForTablePrefix) {
  ChunkWrapper<int8_t> chunk_wrapper1{kTINYINT, {1}};
  ChunkWrapper<int8_t> chunk_wrapper2{kTINYINT, {2}};
  ChunkWrapper<int8_t> chunk_wrapper3{kTINYINT, {3}};
  chunk_wrapper1.cacheMetadataThenChunk(chunk_key1);
  chunk_wrapper2.cacheMetadataThenChunk(chunk_key2);
  chunk_wrapper3.cacheMetadataThenChunk(chunk_key_table2);
  ASSERT_EQ(cache_->getNumCachedChunks(), 3U);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 3U);
  cache_->clearForTablePrefix(table_prefix1);
  ASSERT_EQ(cache_->getNumCachedChunks(), 1U);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 1U);
  ASSERT_FALSE(gfm_->isBufferOnDevice(chunk_key1));
  ASSERT_FALSE(gfm_->isBufferOnDevice(chunk_key2));
  ASSERT_TRUE(gfm_->isBufferOnDevice(chunk_key_table2));
}

TEST_F(ForeignStorageCacheUnitTest, Clear) {
  ChunkWrapper<int8_t> chunk_wrapper1{kTINYINT, {1}};
  ChunkWrapper<int8_t> chunk_wrapper2{kTINYINT, {2}};
  ChunkWrapper<int8_t> chunk_wrapper3{kTINYINT, {3}};
  chunk_wrapper1.cacheMetadataThenChunk(chunk_key1);
  chunk_wrapper2.cacheMetadataThenChunk(chunk_key2);
  chunk_wrapper3.cacheMetadataThenChunk(chunk_key_table2);
  ASSERT_EQ(cache_->getNumCachedChunks(), 3U);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 3U);
  cache_->clear();
  ASSERT_EQ(cache_->getNumCachedChunks(), 0U);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 0U);
  ASSERT_FALSE(gfm_->isBufferOnDevice(chunk_key1));
  ASSERT_FALSE(gfm_->isBufferOnDevice(chunk_key2));
  ASSERT_FALSE(gfm_->isBufferOnDevice(chunk_key_table2));
}

TEST_F(ForeignStorageCacheUnitTest, SetLimit) {
  std::vector<int32_t> full_frag_vec(32000000);
  for (size_t i = 0; i < full_frag_vec.size(); ++i) {
    full_frag_vec[i] = i;
  }
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, full_frag_vec};
  chunk_wrapper1.cacheMetadataThenChunk(chunk_key1);
  ChunkWrapper<int32_t> chunk_wrapper2{kINT, full_frag_vec};
  chunk_wrapper2.cacheMetadataThenChunk(chunk_key2);
  ChunkWrapper<int32_t> chunk_wrapper3{kINT, full_frag_vec};
  chunk_wrapper3.cacheMetadataThenChunk(chunk_key3);
  ChunkWrapper<int32_t> chunk_wrapper4{kINT, full_frag_vec};
  chunk_wrapper4.cacheMetadataThenChunk(chunk_key4);
  ChunkWrapper<int32_t> chunk_wrapper5{kINT, full_frag_vec};
  chunk_wrapper5.cacheMetadataThenChunk(chunk_key5);
  ASSERT_EQ(cache_->getNumCachedChunks(), 5U);
  ASSERT_EQ(cache_->getLimit(), 21474836480UL);
  CacheLimitScope scope{cache_minimum_size};
  ASSERT_EQ(cache_->getNumCachedChunks(), 4U);
  ASSERT_EQ(cache_->getLimit(), 536870912UL);
}

TEST_F(ForeignStorageCacheUnitTest, CacheTooSmall) {
  ASSERT_THROW(CacheLimitScope scope{1}, CacheTooSmallException);
}

TEST_F(ForeignStorageCacheUnitTest, ChunkTooBigForCache) {
  CacheLimitScope scope{cache_minimum_size};
  std::vector<int32_t> full_frag_vec(32000000 * 5);
  for (size_t i = 0; i < full_frag_vec.size(); ++i) {
    full_frag_vec[i] = i;
  }
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, full_frag_vec};
  chunk_wrapper1.cacheMetadataThenChunk(chunk_key1);
  ASSERT_EQ(cache_->getNumCachedChunks(), 0U);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 1U);
}

class CacheDiskStorageTest : public ForeignStorageCacheUnitTest {
 protected:
  static void SetUpTestSuite() {
    cache_path_ = "./tmp/mapd_data/test_foreign_data_cache";
  }
  static void TearDownTestSuite() {}
  void SetUp() override {
    boost::filesystem::remove_all(cache_path_);
    reinitializeCache(cache_, gfm_, {cache_path_, DiskCacheLevel::fsi});
  }
  void TearDown() override { boost::filesystem::remove_all(cache_path_); }
};

TEST_F(CacheDiskStorageTest, CachePath_CreateBaseDir) {
  ASSERT_FALSE(boost::filesystem::exists(cache_path_ + "/table_1_1"));
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, {1, 2, 3, 4}};
  chunk_wrapper1.cacheMetadata(chunk_key1);
  ASSERT_TRUE(boost::filesystem::exists(cache_path_ + "/table_1_1"));
}

TEST_F(CacheDiskStorageTest, CacheMetadata_VerifyMetadataFileCreated) {
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, {1, 2, 3, 4}};
  chunk_wrapper1.cacheMetadata(chunk_key1);
  ASSERT_TRUE(boost::filesystem::exists(cache_path_ + "/table_1_1/0.4096.mapd"));
}

TEST_F(CacheDiskStorageTest, CacheChunk_VerifyChunkFileCreated) {
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, {1, 2, 3, 4}};
  chunk_wrapper1.cacheMetadataThenChunk(chunk_key1);
  ASSERT_TRUE(boost::filesystem::exists(cache_path_ + "/table_1_1/1." +
                                        to_string(gfm_->getDefaultPageSize()) + ".mapd"));
}

TEST_F(CacheDiskStorageTest, RecoverCache_Metadata) {
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, {1, 2, 3, 4}};
  chunk_wrapper1.cacheMetadata(chunk_key1);
  reinitializeCache(cache_, gfm_, {cache_path_, DiskCacheLevel::fsi});
  ASSERT_EQ(cache_->getNumCachedMetadata(), 0U);
  ChunkMetadataVector metadata_vec_cached{};
  cache_->recoverCacheForTable(metadata_vec_cached, table_prefix1);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 1U);
  ASSERT_EQ(cache_->getNumCachedChunks(), 0U);
  assertMetadataEqual(metadata_vec_cached[0].second, createMetadata(16, 4, 1, 4, false));
}

TEST_F(CacheDiskStorageTest, RecoverCache_UpdatedMetadata) {
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, {1, 2, 3, 4}};
  chunk_wrapper1.cacheMetadata(chunk_key1);
  ChunkWrapper<int32_t> chunk_wrapper2{kINT, {5, 6}};
  chunk_wrapper2.cacheMetadata(chunk_key1);
  reinitializeCache(cache_, gfm_, {cache_path_, DiskCacheLevel::fsi});
  ChunkMetadataVector metadata_vec_cached{};
  cache_->recoverCacheForTable(metadata_vec_cached, table_prefix1);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 1U);
  assertMetadataEqual(metadata_vec_cached[0].second, createMetadata(8, 2, 5, 6, false));
}

TEST_F(CacheDiskStorageTest, RecoverCache_SingleChunk) {
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, {1, 2, 3, 4}};
  chunk_wrapper1.cacheMetadataThenChunk(chunk_key1);
  reinitializeCache(cache_, gfm_, {cache_path_, DiskCacheLevel::fsi});
  ASSERT_EQ(cache_->getNumCachedChunks(), 0U);
  ChunkMetadataVector metadata_vec_cached{};
  cache_->recoverCacheForTable(metadata_vec_cached, table_prefix1);
  ASSERT_EQ(cache_->getNumCachedChunks(), 1U);
  AbstractBuffer* cached_buf = cache_->getCachedChunkIfExists(chunk_key1);
  ASSERT_NE(cached_buf, nullptr);
  ASSERT_TRUE(chunk_wrapper1.test_buf->compare(cached_buf, 16));
}

TEST_F(CacheDiskStorageTest, RecoverCache_EvictBeforeRecovery) {
  std::vector<int32_t> full_frag_vec(32000000);
  for (size_t i = 0; i < full_frag_vec.size(); ++i) {
    full_frag_vec[i] = i;
  }
  ChunkWrapper<int32_t> chunk_wrapper1{kINT, full_frag_vec};
  chunk_wrapper1.cacheMetadataThenChunk(chunk_key1);
  ChunkWrapper<int32_t> chunk_wrapper2{kINT, full_frag_vec};
  chunk_wrapper2.cacheMetadataThenChunk(chunk_key2);
  ChunkWrapper<int32_t> chunk_wrapper3{kINT, full_frag_vec};
  chunk_wrapper3.cacheMetadataThenChunk(chunk_key3);
  ChunkWrapper<int32_t> chunk_wrapper4{kINT, full_frag_vec};
  chunk_wrapper4.cacheMetadataThenChunk(chunk_key4);
  ChunkWrapper<int32_t> chunk_wrapper5{kINT, full_frag_vec};
  chunk_wrapper5.cacheMetadataThenChunk(chunk_key5);

  // size_t chunk_size = 32000000 * sizeof(int32_t);
  { CacheLimitScope scope{cache_minimum_size}; }

  reinitializeCache(cache_, gfm_, {cache_path_, DiskCacheLevel::fsi});
  ChunkMetadataVector metadata_vec_cached{};
  cache_->recoverCacheForTable(metadata_vec_cached, table_prefix1);
  ASSERT_EQ(cache_->getNumCachedChunks(), 4U);
  AbstractBuffer* cached_buf = cache_->getCachedChunkIfExists(chunk_key5);
  ASSERT_NE(cached_buf, nullptr);
  ASSERT_NE(chunk_wrapper5.test_buf, nullptr);
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

TEST_F(ForeignStorageCacheFileTest, FileCreation) {
  cache_path_ = "./test_foreign_data_cache";
  boost::filesystem::remove_all(cache_path_);
  {
    ForeignStorageCache cache{{cache_path_, DiskCacheLevel::fsi}};
    GlobalFileMgr* gfm = cache.getGlobalFileMgr();
    ASSERT_TRUE(boost::filesystem::exists(cache_path_));
    ASSERT_FALSE(boost::filesystem::exists(cache_path_ + "/table_1_1"));
    ASSERT_EQ(cache.getCachedChunkIfExists(chunk_key1), nullptr);
    ASSERT_FALSE(gfm->isBufferOnDevice(chunk_key1));
    TestBuffer source_buffer{std::vector<int8_t>{1, 2, 3, 4}};
    source_buffer.initEncoder(kINT);
    std::shared_ptr<ChunkMetadata> cached_meta = std::make_shared<ChunkMetadata>();
    source_buffer.getEncoder()->getMetadata(cached_meta);
    cache.cacheMetadataVec({std::make_pair(chunk_key1, cached_meta)});
    auto buffer_map = cache.getChunkBuffersForCaching({chunk_key1});
    buffer_map[chunk_key1]->write(source_buffer.getMemoryPtr(), source_buffer.size());
    cache.cacheTableChunks({chunk_key1});
    ASSERT_TRUE(gfm->isBufferOnDevice(chunk_key1));
    ASSERT_TRUE(boost::filesystem::exists(cache_path_ + "/table_1_1/0.4096.mapd"));
    ASSERT_TRUE(boost::filesystem::exists(
        cache_path_ + "/table_1_1/1." + to_string(gfm->getDefaultPageSize()) + ".mapd"));
  }
  // Cache files should persist after cache is destroyed.
  ASSERT_TRUE(boost::filesystem::exists(cache_path_));
}

TEST_F(ForeignStorageCacheFileTest, CustomPath) {
  cache_path_ = "./test_foreign_data_cache";
  PersistentStorageMgr psm(data_path, 0, {cache_path_, DiskCacheLevel::fsi});
  ASSERT_EQ(psm.getDiskCache()->getGlobalFileMgr()->getBasePath(), cache_path_ + "/");
}

TEST_F(ForeignStorageCacheFileTest, InitializeSansCache) {
  cache_path_ = "./test_foreign_data_cache";
  PersistentStorageMgr psm(data_path, 0, {cache_path_});
  ASSERT_EQ(psm.getDiskCache(), nullptr);
}

TEST_F(ForeignStorageCacheFileTest, EnableCache) {
  cache_path_ = "./test_foreign_data_cache";
  PersistentStorageMgr psm(data_path, 0, {cache_path_, DiskCacheLevel::fsi});
  ASSERT_NE(psm.getDiskCache(), nullptr);
}

TEST_F(ForeignStorageCacheFileTest, FileBlocksPath) {
  cache_path_ = "./test_foreign_data_cache";
  boost::filesystem::remove_all(cache_path_);
  ASSERT_FALSE(boost::filesystem::exists(cache_path_));
  boost::filesystem::ofstream tmp_file(cache_path_);
  tmp_file << "1";
  tmp_file.close();
  try {
    ForeignStorageCache cache{{cache_path_, DiskCacheLevel::fsi}};
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
  ForeignStorageCache cache{{cache_path_, DiskCacheLevel::fsi}};
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
