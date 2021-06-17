/*
 * Copyright 2021 OmniSci, Inc.
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
 * @file CachingFileMgrTest.cpp
 * @brief Unit tests for CachingFileMgr class.
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

#include "DataMgr/FileMgr/CachingFileMgr.h"
#include "DataMgrTestHelpers.h"

constexpr char test_path[] = "./CacheTest";
namespace bf = boost::filesystem;
namespace fn = File_Namespace;

namespace {
void assert_dir_contains(const std::string& dir, const std::set<std::string>& files) {
  std::set<std::string> expected_files(files);
  for (const auto& file : bf::directory_iterator(dir)) {
    ASSERT_FALSE(files.find(file.path().filename().string()) == files.end())
        << "Unexpected file (" << file.path() << ") found in dir (" << dir << ")";
    expected_files.erase(file.path().filename().string());
  }
  if (!expected_files.empty()) {
    std::stringstream files_left;
    for (auto file : expected_files) {
      files_left << file << ", ";
    }
    FAIL() << "Could not find expected files: " << files_left.str();
  }
}
}  // namespace

class CachingFileMgrTest : public testing::Test {
 protected:
  // Keep page size small for these tests so we can hit limits more easily.
  static constexpr size_t page_size_ = 64;
  static constexpr size_t page_data_size_ =
      page_size_ - fn::FileBuffer::headerBufferOffset_;
  static constexpr size_t data_file_size_ =
      page_size_ * fn::CachingFileMgr::DEFAULT_NUM_PAGES_PER_DATA_FILE;
  static constexpr size_t meta_file_size_ =
      METADATA_PAGE_SIZE * fn::CachingFileMgr::DEFAULT_NUM_PAGES_PER_METADATA_FILE;
  static constexpr size_t cache_size_ = fn::DiskCacheConfig::DEFAULT_MAX_SIZE;
  static constexpr int32_t db_ = 1, tb_ = 1;
  const ChunkKey key_{db_, tb_, 1, 1};
  static constexpr size_t small_buffer_size_ = 4;
  std::vector<int8_t> small_buffer_ = std::vector<int8_t>(small_buffer_size_, -1);
  std::vector<int8_t> full_page_buffer_ = std::vector<int8_t>(page_data_size_, -1);
  fn::DiskCacheConfig disk_cache_config{test_path,
                                        fn::DiskCacheLevel::all,
                                        0,
                                        cache_size_,
                                        page_size_};
  std::string doc = std::string(1000, 'a');

  void SetUp() override {
    bf::remove_all(test_path);
    bf::create_directory(test_path);
  }
  void TearDown() override { bf::remove_all(test_path); }

  std::unique_ptr<fn::CachingFileMgr> initializeCFM(const size_t num_pages = 0) {
    auto cfm = std::make_unique<fn::CachingFileMgr>(disk_cache_config);
    // Override these defaults to hit limits more easily.
    cfm->setMaxNumDataFiles(1);
    cfm->setMaxNumMetadataFiles(1);
    cfm->setMaxWrapperSpace(1500);
    if (num_pages > 0) {
      writePages(*cfm, key_, num_pages);
    }
    return cfm;
  }

  void validatePageInfo(const fn::FileBuffer* buf,
                        const size_t num_pages,
                        const size_t num_page_versions,
                        const size_t num_meta_versions) {
    ASSERT_EQ(buf->getMultiPage().size(), num_pages);
    for (auto multi_page : buf->getMultiPage()) {
      ASSERT_EQ(multi_page.pageVersions.size(), num_page_versions);
    }
    ASSERT_EQ(buf->getMetadataPage().pageVersions.size(), num_meta_versions);
  }

  void writePages(fn::CachingFileMgr& cfm,
                  const ChunkKey& key,
                  size_t num_pages,
                  int8_t value = -1) {
    auto [db, tb] = get_table_prefix(key);
    TestHelpers::TestBuffer test_buf{
        std::vector<int8_t>(page_data_size_ * num_pages, value)};
    cfm.putBuffer(key, &test_buf);
    cfm.checkpoint(db, tb);
  }

  void writeMetaPages(fn::CachingFileMgr& cfm,
                      int32_t db,
                      int32_t tb,
                      int32_t num_pages) {
    for (int32_t i = 0; i < num_pages; ++i) {
      ChunkKey key{db, tb, 1, i};
      TestHelpers::TestBuffer test_buf{small_buffer_};
      cfm.putBuffer(key, &test_buf);
    }
    cfm.checkpoint(db, tb);
  }

  void writeWrapperFile(fn::CachingFileMgr& cfm,
                        const std::string& doc,
                        int32_t db,
                        int32_t tb) {
    cfm.writeWrapperFile(doc, db, tb);
  }

  void assertCachedMetadataEquals(
      const fn::CachingFileMgr& cfm,
      const std::vector<std::tuple<int32_t, int32_t, int32_t>>& metadata) {
    std::set<ChunkKey> expected_keys;
    for (auto [db, tb, num_keys] : metadata) {
      for (auto i = 0; i < num_keys; ++i) {
        expected_keys.emplace(ChunkKey{db, tb, 1, i});
      }
    }

    auto keys_with_metadata = cfm.getKeysWithMetadata();
    ASSERT_EQ(keys_with_metadata, expected_keys);
  }
};

TEST_F(CachingFileMgrTest, InitializeEmpty) {
  auto cfm = initializeCFM();
  assert_dir_contains(test_path, {});
}

// Epoch tests
TEST_F(CachingFileMgrTest, EpochSingleTable) {
  auto cfm = initializeCFM();
  // Create a buffer and add some values
  auto buf = cfm->createBuffer(key_);
  assert_dir_contains(std::string(test_path) + "/table_1_1", {"epoch_metadata"});
  ASSERT_EQ(cfm->epoch(1, 1), 1);
  buf->append(small_buffer_.data(), small_buffer_size_);
  cfm->checkpoint(1, 1);
  // Confirm epoch file exists
  ASSERT_EQ(cfm->epoch(1, 1), 2);
}

TEST_F(CachingFileMgrTest, EpochMultiTable) {
  auto cfm = initializeCFM();
  // Create a buffer and add some values
  auto buf1 = cfm->createBuffer(key_);
  auto buf2 = cfm->createBuffer({1, 2, 1, 1});
  buf1->append(small_buffer_.data(), small_buffer_size_);
  buf2->append(small_buffer_.data(), small_buffer_size_);
  cfm->checkpoint(1, 1);
  ASSERT_EQ(cfm->epoch(1, 1), 2);
  ASSERT_EQ(cfm->epoch(1, 2), 1);
  cfm->checkpoint(1, 2);
  ASSERT_EQ(cfm->epoch(1, 1), 2);
  ASSERT_EQ(cfm->epoch(1, 2), 2);
  cfm->checkpoint(1, 1);
  ASSERT_EQ(cfm->epoch(1, 1), 3);
  ASSERT_EQ(cfm->epoch(1, 2), 2);
}

// Initialization tests
TEST_F(CachingFileMgrTest, InitializeFromCheckpointedData) {
  std::vector<int8_t> read_buffer(4);
  {
    auto cfm = initializeCFM();
    auto buf = cfm->createBuffer(key_);
    buf->append(small_buffer_.data(), small_buffer_size_);
    cfm->checkpoint(1, 1);
  }
  auto cfm = initializeCFM();
  ASSERT_EQ(cfm->epoch(1, 1), 2);
  auto buffer = cfm->getBuffer(key_);
  ASSERT_EQ(buffer->pageCount(), 1U);
  ASSERT_EQ(buffer->numMetadataPages(), 1U);
  ASSERT_EQ(buffer->size(), 4U);
  buffer->read(read_buffer.data(), 4);
  ASSERT_EQ(read_buffer, small_buffer_);
}

TEST_F(CachingFileMgrTest, InitializeFromUncheckpointedData) {
  std::vector<int8_t> read_buffer(4);
  {
    auto cfm = initializeCFM();
    auto buf = cfm->createBuffer(key_);
    buf->append(small_buffer_.data(), small_buffer_size_);
  }
  auto cfm = initializeCFM();
  // When creating a new CFM, if we find an existing epoch for a table (even if there is
  // no table data) we will increment the epoch after reading it, so we will always be on
  // a new epoch (hence epoch = 1 despite never having checkpointed any data).
  ASSERT_EQ(cfm->epoch(1, 1), 1);
  ASSERT_FALSE(cfm->isBufferOnDevice(key_));
}

TEST_F(CachingFileMgrTest, InitializeFromPartiallyCheckpointedData) {
  std::vector<int8_t> overwrite_buffer{5, 6, 7, 8};
  std::vector<int8_t> read_buffer(4);
  {
    auto cfm = initializeCFM();
    auto buf = cfm->createBuffer(key_);
    buf->append(small_buffer_.data(), small_buffer_size_);
    cfm->checkpoint(1, 1);
    buf->append(overwrite_buffer.data(), 4);
  }
  auto cfm = initializeCFM();
  ASSERT_EQ(cfm->epoch(1, 1), 2);
  auto buffer = cfm->getBuffer(key_);
  ASSERT_EQ(buffer->pageCount(), 1U);
  ASSERT_EQ(buffer->numMetadataPages(), 1U);
  ASSERT_EQ(buffer->size(), 4U);
  buffer->read(read_buffer.data(), small_buffer_size_);
  ASSERT_EQ(read_buffer, small_buffer_);
}

TEST_F(CachingFileMgrTest, InitializeFromPartiallyFreedDataLastPage) {
  {
    auto temp_cfm = initializeCFM(2);
    auto buffer = temp_cfm->getBuffer(key_);
    buffer->freePage(buffer->getMultiPage().back().current().page);
  }
  auto cfm = initializeCFM();
  auto buffer = cfm->getBuffer(key_);
  ASSERT_EQ(buffer->size(), (page_data_size_)*2);
  ASSERT_EQ(buffer->pageCount(), 0U);
  ASSERT_EQ(buffer->numMetadataPages(), 1U);
}

TEST_F(CachingFileMgrTest, InitializeFromPartiallyFreedDataFirstPage) {
  {
    auto temp_cfm = initializeCFM(2);
    auto buffer = temp_cfm->getBuffer(key_);
    buffer->freePage(buffer->getMultiPage().front().current().page);
  }
  auto cfm = initializeCFM();
  auto buffer = cfm->getBuffer(key_);
  ASSERT_EQ(buffer->size(), (page_data_size_)*2);
  ASSERT_EQ(buffer->pageCount(), 0U);
  ASSERT_EQ(buffer->numMetadataPages(), 1U);
}

TEST_F(CachingFileMgrTest, InitializeFromFreedMetadata) {
  {
    auto temp_cfm = initializeCFM(2);
    auto buffer = temp_cfm->getBuffer(key_);
    buffer->freePage(buffer->getMetadataPage().current().page);
  }
  auto cfm = initializeCFM();
  ASSERT_FALSE(cfm->isBufferOnDevice(key_));
  ASSERT_EQ(cfm->getNumChunks(), 0U);
}

// Tests to make sure we only have one version of data in the CFM
TEST_F(CachingFileMgrTest, SingleVersion_SinglePage) {
  auto cfm = initializeCFM();
  auto buf = cfm->createBuffer(key_);
  buf->append(full_page_buffer_.data(), page_data_size_);
  cfm->checkpoint(db_, tb_);
  validatePageInfo(buf, 1, 1, 1);
}

TEST_F(CachingFileMgrTest, SingleVersion_TwoPages_SingleCheckpoint) {
  auto cfm = initializeCFM();
  auto buf = cfm->createBuffer(key_);
  buf->append(full_page_buffer_.data(), page_data_size_);
  buf->append(full_page_buffer_.data(), page_data_size_);
  cfm->checkpoint(db_, tb_);
  validatePageInfo(buf, 2, 1, 1);
}

TEST_F(CachingFileMgrTest, SingleVersion_TwoPages_TwoCheckpoints) {
  auto cfm = initializeCFM();
  auto buf = cfm->createBuffer(key_);
  buf->append(full_page_buffer_.data(), page_data_size_);
  cfm->checkpoint(db_, tb_);
  buf->append(full_page_buffer_.data(), page_data_size_);
  cfm->checkpoint(db_, tb_);
  validatePageInfo(buf, 2, 1, 1);
}

// Test how chunks are evicted - data is evicted on the chunk-level.
class ChunkEvictionTest : public CachingFileMgrTest {};

TEST_F(ChunkEvictionTest, SameTable) {
  auto cfm = initializeCFM();
  writePages(*cfm, {1, 1, 1, 1}, 128);
  writePages(*cfm, {1, 1, 1, 2}, 128);
  writePages(*cfm, {1, 1, 1, 3}, 128);
  ASSERT_EQ(cfm->getBuffer({1, 1, 1, 1})->pageCount(), 0U);
  ASSERT_EQ(cfm->getBuffer({1, 1, 1, 2})->pageCount(), 128U);
  ASSERT_EQ(cfm->getBuffer({1, 1, 1, 3})->pageCount(), 128U);
}

TEST_F(ChunkEvictionTest, LargeChunk) {
  auto cfm = initializeCFM();
  writePages(*cfm, {1, 1, 1, 1}, 128);
  writePages(*cfm, {1, 1, 1, 2}, 128);
  writePages(*cfm, {1, 1, 1, 3}, 256);
  writePages(*cfm, {1, 1, 1, 1}, 64);
  ASSERT_EQ(cfm->getBuffer({1, 1, 1, 1})->pageCount(), 64U);
  ASSERT_EQ(cfm->getBuffer({1, 1, 1, 2})->pageCount(), 0U);
  ASSERT_EQ(cfm->getBuffer({1, 1, 1, 3})->pageCount(), 0U);
}

TEST_F(ChunkEvictionTest, ChunkReusesPagesIfOverwritten) {
  auto cfm = initializeCFM();
  writePages(*cfm, {1, 1, 1, 1}, 256);
  writePages(*cfm, {1, 1, 1, 1}, 256, 64);
  auto buf = cfm->getBuffer({1, 1, 1, 1});
  ASSERT_EQ(buf->pageCount(), 256U);
  std::vector<int8_t> vec(4);
  buf->read(vec.data(), 4);
  ASSERT_EQ(vec, std::vector<int8_t>(4, 64));
}

TEST_F(ChunkEvictionTest, MultipleTables) {
  auto cfm = initializeCFM();
  writePages(*cfm, {1, 1, 1, 1}, 128);
  writePages(*cfm, {1, 2, 1, 1}, 128);
  writePages(*cfm, {1, 1, 1, 2}, 128);
  ASSERT_EQ(cfm->getBuffer({1, 1, 1, 1})->pageCount(), 0U);
  ASSERT_EQ(cfm->getBuffer({1, 2, 1, 1})->pageCount(), 128U);
  ASSERT_EQ(cfm->getBuffer({1, 1, 1, 2})->pageCount(), 128U);
}

// Test how metadata is evicted - metadata is evicted on the table-level.
class MetadataEvictionTest : public CachingFileMgrTest {};

TEST_F(MetadataEvictionTest, WholeTable) {
  auto cfm = initializeCFM();
  writeMetaPages(*cfm, 1, 1, fn::CachingFileMgr::DEFAULT_NUM_PAGES_PER_METADATA_FILE);
  assertCachedMetadataEquals(
      *cfm, {{1, 1, fn::CachingFileMgr::DEFAULT_NUM_PAGES_PER_METADATA_FILE}});
  writeMetaPages(*cfm, 1, 2, 1);
  assertCachedMetadataEquals(*cfm, {{1, 2, 1}});  // Other table has been evicted
}

TEST_F(MetadataEvictionTest, MultipleTable) {
  auto cfm = initializeCFM();
  writeMetaPages(*cfm, 1, 1, 1024);  // Will get evicted
  writeMetaPages(*cfm, 1, 2, 1024);
  writeMetaPages(*cfm, 1, 3, 1024);
  writeMetaPages(*cfm, 1, 4, 1024);
  writeMetaPages(*cfm, 1, 5, 1);
  assertCachedMetadataEquals(*cfm, {{1, 2, 1024}, {1, 3, 1024}, {1, 4, 1024}, {1, 5, 1}});
}

// Verifies that deleting a buffer does not remove the table from the eviction queue if
// there are additional chunks in the table.
TEST_F(MetadataEvictionTest, ChunkDeletedFromLargeTable) {
  auto cfm = initializeCFM();
  writeMetaPages(*cfm, 1, 1, 1024);
  cfm->deleteBufferIfExists({1, 1, 1, 1023});
  writeMetaPages(*cfm, 1, 2, 512);
  writeMetaPages(*cfm, 1, 3, 512);
  writeMetaPages(*cfm, 1, 4, 1024);
  writeMetaPages(*cfm, 1, 5, 1024);
  writeMetaPages(*cfm, 1, 6, 1);
  // Deletion should have made space for an extra chunk without evicting a table
  assertCachedMetadataEquals(
      *cfm,
      {{1, 1, 1023}, {1, 2, 512}, {1, 3, 512}, {1, 4, 1024}, {1, 5, 1024}, {1, 6, 1}});
  writeMetaPages(*cfm, 1, 7, 1);  // table {1,1} should now be evicted.
  assertCachedMetadataEquals(
      *cfm, {{1, 2, 512}, {1, 3, 512}, {1, 4, 1024}, {1, 5, 1024}, {1, 6, 1}, {1, 7, 1}});
}

TEST_F(MetadataEvictionTest, ChunkDeletedFromSmallTable) {
  auto cfm = initializeCFM();
  writeMetaPages(*cfm, 1, 1, 1);
  // deleting this chunk should remove the table from the eviction queue
  cfm->deleteBufferIfExists({1, 1, 1, 0});
  writeMetaPages(*cfm, 1, 2, 1024);
  writeMetaPages(*cfm, 1, 3, 1024);
  writeMetaPages(*cfm, 1, 4, 1024);
  writeMetaPages(*cfm, 1, 5, 1024);
  assertCachedMetadataEquals(*cfm,
                             {{1, 2, 1024}, {1, 3, 1024}, {1, 4, 1024}, {1, 5, 1024}});
  // Deletion should have made space for an extra chunk without evicting a table
  writeMetaPages(*cfm, 1, 6, 1);
  assertCachedMetadataEquals(*cfm, {{1, 3, 1024}, {1, 4, 1024}, {1, 5, 1024}, {1, 6, 1}});
}

class WrapperEvictionTest : public CachingFileMgrTest {};

TEST_F(WrapperEvictionTest, WrapperWritten) {
  auto cfm = initializeCFM();
  writeWrapperFile(*cfm, doc, 1, 1);
  ASSERT_TRUE(boost::filesystem::exists(cfm->getFileMgrBasePath() +
                                        "/table_1_1/wrapper_metadata.json"));
}

TEST_F(WrapperEvictionTest, WrapperWrittenAfterEvictingAnother) {
  auto cfm = initializeCFM();
  writeMetaPages(*cfm, 1, 1, 1);  // Force allocation of a meta file and data file.
  writeMetaPages(*cfm, 1, 2, 1);
  writeWrapperFile(*cfm, doc, 1, 1);
  writeWrapperFile(*cfm, doc, 1, 2);
  ASSERT_FALSE(boost::filesystem::exists(cfm->getFileMgrBasePath() +
                                         "/table_1_1/wrapper_metadata.json"));
  ASSERT_TRUE(boost::filesystem::exists(cfm->getFileMgrBasePath() +
                                        "/table_1_2/wrapper_metadata.json"));
  assertCachedMetadataEquals(*cfm, {{1, 2, 1}});
}

TEST_F(WrapperEvictionTest, WrapperWrittenAfterEvictingMultiples) {
  auto cfm = initializeCFM();
  writeMetaPages(*cfm, 1, 1, 1);  // Force allocation of a meta file and data file.
  writeMetaPages(*cfm, 1, 2, 1);
  writeMetaPages(*cfm, 1, 3, 1);
  writeWrapperFile(*cfm, doc, 1, 1);
  writeWrapperFile(*cfm, doc, 1, 2);
  writeWrapperFile(*cfm, doc, 1, 3);
  ASSERT_FALSE(boost::filesystem::exists(cfm->getFileMgrBasePath() +
                                         "/table_1_1/wrapper_metadata.json"));
  ASSERT_FALSE(boost::filesystem::exists(cfm->getFileMgrBasePath() +
                                         "/table_1_2/wrapper_metadata.json"));
  ASSERT_TRUE(boost::filesystem::exists(cfm->getFileMgrBasePath() +
                                        "/table_1_3/wrapper_metadata.json"));
  assertCachedMetadataEquals(*cfm, {{1, 3, 1}});
}

class SizeTest : public CachingFileMgrTest {};

// Currently the minimum size is deteremined by the minimum metadata size requirements.
TEST_F(SizeTest, MinMetadataSize) {
  // The cache needs to be at least big enough to create one metadata file within the
  // assigned percentage, so this is the smallest cache we can reasonably make.  Based on
  // the current file size ratios, the smallest usable cache size is based on the metadata
  // file.
  size_t min_cache_size =
      meta_file_size_ / fn::CachingFileMgr::METADATA_FILE_SPACE_PERCENTAGE;
  fn::DiskCacheConfig config{
      test_path, fn::DiskCacheLevel::all, 0, min_cache_size, page_size_};
  auto cfm = fn::CachingFileMgr(config);

  ASSERT_EQ(cfm.getMaxWrapperSize(),
            (min_cache_size * fn::CachingFileMgr::METADATA_SPACE_PERCENTAGE) -
                (min_cache_size * fn::CachingFileMgr::METADATA_FILE_SPACE_PERCENTAGE));
  ASSERT_EQ(cfm.getMaxMetaFiles(), 1U);
}

TEST_F(SizeTest, DefaultSize) {
  fn::DiskCacheConfig config{
      test_path, fn::DiskCacheLevel::all, 0, cache_size_, DEFAULT_PAGE_SIZE};
  auto cfm = fn::CachingFileMgr(config);
  ASSERT_EQ(cfm.getMaxWrapperSize(),
            (cache_size_ * fn::CachingFileMgr::METADATA_SPACE_PERCENTAGE) -
                (cache_size_ * fn::CachingFileMgr::METADATA_FILE_SPACE_PERCENTAGE));
  ASSERT_EQ(cfm.getMaxDataFiles(), 36U);
  ASSERT_EQ(cfm.getMaxMetaFiles(), 12U);
}

TEST_F(SizeTest, ChunkSpace) {
  auto cfm = initializeCFM();
  ASSERT_EQ(cfm->getChunkSpaceReservedByTable(1, 1), 0U);
  writePages(*cfm, {1, 1, 1, 1}, 2);
  ASSERT_EQ(cfm->getChunkSpaceReservedByTable(1, 1), page_size_ * 2);
}

TEST_F(SizeTest, MetaSpace) {
  auto cfm = initializeCFM();
  ASSERT_EQ(cfm->getMetadataSpaceReservedByTable(1, 1), 0U);
  writeMetaPages(*cfm, 1, 1, 2);
  ASSERT_EQ(cfm->getMetadataSpaceReservedByTable(1, 1), METADATA_PAGE_SIZE * 2U);
}

TEST_F(SizeTest, WrapperSpace) {
  auto cfm = initializeCFM();
  ASSERT_EQ(cfm->getTableFileMgrSpaceReserved(1, 1), 0U);
  writeWrapperFile(*cfm, doc, 1, 1);
  ASSERT_EQ(cfm->getTableFileMgrSpaceReserved(1, 1),
            boost::filesystem::file_size(cfm->getFileMgrBasePath() +
                                         "/table_1_1/wrapper_metadata.json") +
                Epoch::byte_size());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
