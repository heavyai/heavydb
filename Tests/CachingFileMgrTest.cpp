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
  static constexpr size_t page_size_ = 64;
  static constexpr size_t page_data_size_ =
      page_size_ - fn::FileBuffer::headerBufferOffset_;
  static constexpr int32_t db_ = 1, tb_ = 1;
  const ChunkKey key_{db_, tb_, 1, 1};
  static constexpr size_t small_buffer_size_ = 4;
  std::vector<int8_t> small_buffer_ = std::vector<int8_t>(small_buffer_size_, 255);
  std::vector<int8_t> full_page_buffer_ = std::vector<int8_t>(page_data_size_, 255);

  void SetUp() override {
    bf::remove_all(test_path);
    bf::create_directory(test_path);
  }
  void TearDown() override { bf::remove_all(test_path); }

  std::unique_ptr<fn::CachingFileMgr> initializeCFM(const size_t num_pages = 1) {
    auto cfm = std::make_unique<fn::CachingFileMgr>(test_path, 0, page_size_);
    auto buf = cfm->createBuffer(key_);
    for (size_t i = 0; i < num_pages; ++i) {
      buf->append(full_page_buffer_.data(), page_data_size_);
    }
    cfm->checkpoint(db_, tb_);
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
};

TEST_F(CachingFileMgrTest, InitializeEmpty) {
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  assert_dir_contains(test_path, {});
}

// Epoch tests
TEST_F(CachingFileMgrTest, EpochSingleTable) {
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  // Create a buffer and add some values
  auto buf = cfm.createBuffer(key_);
  assert_dir_contains(std::string(test_path) + "/table_1_1", {"epoch_metadata"});
  ASSERT_EQ(cfm.epoch(1, 1), 1);
  buf->append(small_buffer_.data(), small_buffer_size_);
  cfm.checkpoint(1, 1);
  // Confirm epoch file exists
  ASSERT_EQ(cfm.epoch(1, 1), 2);
}

TEST_F(CachingFileMgrTest, EpochMultiTable) {
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  // Create a buffer and add some values
  auto buf1 = cfm.createBuffer(key_);
  auto buf2 = cfm.createBuffer({1, 2, 1, 1});
  buf1->append(small_buffer_.data(), small_buffer_size_);
  buf2->append(small_buffer_.data(), small_buffer_size_);
  cfm.checkpoint(1, 1);
  ASSERT_EQ(cfm.epoch(1, 1), 2);
  ASSERT_EQ(cfm.epoch(1, 2), 1);
  cfm.checkpoint(1, 2);
  ASSERT_EQ(cfm.epoch(1, 1), 2);
  ASSERT_EQ(cfm.epoch(1, 2), 2);
  cfm.checkpoint(1, 1);
  ASSERT_EQ(cfm.epoch(1, 1), 3);
  ASSERT_EQ(cfm.epoch(1, 2), 2);
}

// Initialization tests
TEST_F(CachingFileMgrTest, InitializeFromCheckpointedData) {
  std::vector<int8_t> read_buffer(4);
  {
    auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
    auto buf = cfm.createBuffer(key_);
    buf->append(small_buffer_.data(), small_buffer_size_);
    cfm.checkpoint(1, 1);
  }
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  ASSERT_EQ(cfm.epoch(1, 1), 2);
  auto buffer = cfm.getBuffer(key_);
  ASSERT_EQ(buffer->pageCount(), 1U);
  ASSERT_EQ(buffer->numMetadataPages(), 1U);
  ASSERT_EQ(buffer->size(), 4U);
  buffer->read(read_buffer.data(), 4);
  ASSERT_EQ(read_buffer, small_buffer_);
}

TEST_F(CachingFileMgrTest, InitializeFromUncheckpointedData) {
  std::vector<int8_t> read_buffer(4);
  {
    auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
    auto buf = cfm.createBuffer(key_);
    buf->append(small_buffer_.data(), small_buffer_size_);
  }
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  // When creating a new CFM, if we find an existing epoch for a table (even if there is
  // no table data) we will increment the epoch after reading it, so we will always be on
  // a new epoch (hence epoch = 1 despite never having checkpointed any data).
  ASSERT_EQ(cfm.epoch(1, 1), 1);
  ASSERT_FALSE(cfm.isBufferOnDevice(key_));
}

TEST_F(CachingFileMgrTest, InitializeFromPartiallyCheckpointedData) {
  std::vector<int8_t> overwrite_buffer{5, 6, 7, 8};
  std::vector<int8_t> read_buffer(4);
  {
    auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
    auto buf = cfm.createBuffer(key_);
    buf->append(small_buffer_.data(), small_buffer_size_);
    cfm.checkpoint(1, 1);
    buf->append(overwrite_buffer.data(), 4);
  }
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  ASSERT_EQ(cfm.epoch(1, 1), 2);
  auto buffer = cfm.getBuffer(key_);
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
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  auto buffer = cfm.getBuffer(key_);
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
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  auto buffer = cfm.getBuffer(key_);
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
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  ASSERT_FALSE(cfm.isBufferOnDevice(key_));
  ASSERT_EQ(cfm.getNumChunks(), 0U);
}

// Tests to make sure we only have one version of data in the CFM
TEST_F(CachingFileMgrTest, SingleVersion_SinglePage) {
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  auto buf = cfm.createBuffer(key_);
  buf->append(small_buffer_.data(), small_buffer_size_);
  cfm.checkpoint(db_, tb_);
  validatePageInfo(buf, 1, 1, 1);
}

TEST_F(CachingFileMgrTest, SingleVersion_TwoPages_SingleCheckpoint) {
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  auto buf = cfm.createBuffer(key_);
  buf->append(full_page_buffer_.data(), page_data_size_);
  buf->append(full_page_buffer_.data(), page_data_size_);
  cfm.checkpoint(db_, tb_);
  validatePageInfo(buf, 2, 1, 1);
}

TEST_F(CachingFileMgrTest, SingleVersion_TwoPages_TwoCheckpoints) {
  auto cfm = fn::CachingFileMgr(test_path, 0, page_size_);
  auto buf = cfm.createBuffer(key_);
  buf->append(full_page_buffer_.data(), page_data_size_);
  cfm.checkpoint(db_, tb_);
  buf->append(full_page_buffer_.data(), page_data_size_);
  cfm.checkpoint(db_, tb_);
  validatePageInfo(buf, 2, 1, 1);
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
