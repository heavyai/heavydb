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
 * @file FileMgrTest.cpp
 * @brief Unit tests for FileMgr class.
 */

#include <fstream>

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "DataMgr/FileMgr/FileMgr.h"
#include "DataMgr/FileMgr/GlobalFileMgr.h"
#include "DataMgr/ForeignStorage/ArrowForeignStorage.h"
#include "DataMgrTestHelpers.h"
#include "Shared/File.h"
#include "TestHelpers.h"

namespace bf = boost::filesystem;

class FileInfoTest : public testing::Test {
 public:
  constexpr static const char* test_data_dir = "./test_dir";
  constexpr static const char* data_file_name = "./test_dir/0.64.data";
  constexpr static const char* meta_file_name = "./test_dir/1.128.data";
  constexpr static const int32_t db = 1, tb = 1, data_file_id = 0, meta_file_id = 1;
  constexpr static const size_t num_pages = 16, page_size = 64, meta_page_size = 128;

 protected:
  void SetUp() override {
    bf::remove_all(test_data_dir);
    bf::create_directory(test_data_dir);

    // Currently FileInfo has a dependency on having a parent FileMgr, so we generate them
    // here.  Other than openExistingFile() the parent FileMgr state will not affect the
    // FileInfo's method calls.  Future work is underway to remove this dependency
    // entirely (a FileInfo should not need access to a parent FileMgr).
    fsi_ = std::make_shared<ForeignStorageInterface>();
    gfm_ = std::make_unique<File_Namespace::GlobalFileMgr>(
        0, fsi_, test_data_dir, 0, page_size, meta_page_size);
    fm_ptr_ = dynamic_cast<File_Namespace::FileMgr*>(gfm_->getFileMgr(1, 1));

    auto fd = File_Namespace::create(test_data_dir, data_file_id, page_size, num_pages);
    file_info_ = std::make_unique<File_Namespace::FileInfo>(
        fm_ptr_, data_file_id, fd, page_size, num_pages);
  }

  void TearDown() override {
    file_info_ = nullptr;
    bf::remove_all(test_data_dir);
  }

  static void SetUpTestSuite() {
    File_Namespace::FileMgr::setNumPagesPerDataFile(num_pages);
    File_Namespace::FileMgr::setNumPagesPerMetadataFile(num_pages);
  }

  static void TearDownTestSute() {
    File_Namespace::FileMgr::setNumPagesPerDataFile(
        File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_DATA_FILE);
    File_Namespace::FileMgr::setNumPagesPerMetadataFile(
        File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_METADATA_FILE);
  }

  template <class T>
  std::vector<T> readFromFile(const char* file, size_t offset, size_t num_elems) {
    CHECK(!file_info_) << "File desc must be closed before we read file directly.";
    auto fd = heavyai::fopen(file, "r");
    std::vector<T> buf(num_elems);
    File_Namespace::read(
        fd, offset, num_elems * sizeof(T), reinterpret_cast<int8_t*>(buf.data()));
    fclose(fd);
    return buf;
  }

  std::vector<int32_t> getTypeInfoBufferFromSqlInfo(const SQLTypeInfo& sql_type) {
    CHECK_EQ(NUM_METADATA, 10);  // Defined in FileBuffer.h
    std::vector<int32_t> type_data(NUM_METADATA);
    type_data[0] = METADATA_VERSION;  // METADATA_VERSION is currently 0.
    type_data[1] = 1;                 // Set has_encoder.
    type_data[2] = static_cast<int32_t>(sql_type.get_type());
    type_data[3] = static_cast<int32_t>(sql_type.get_subtype());
    type_data[4] = sql_type.get_dimension();
    type_data[5] = sql_type.get_scale();
    type_data[6] = static_cast<int32_t>(sql_type.get_notnull());
    type_data[7] = static_cast<int32_t>(sql_type.get_compression());
    type_data[8] = sql_type.get_comp_param();
    type_data[9] = sql_type.get_size();
    return type_data;
  }

  std::shared_ptr<ForeignStorageInterface> fsi_;
  std::unique_ptr<File_Namespace::GlobalFileMgr> gfm_;
  File_Namespace::FileMgr* fm_ptr_;
  std::unique_ptr<File_Namespace::FileInfo> file_info_;
};

TEST_F(FileInfoTest, initNewFile) {
  file_info_->initNewFile();
  EXPECT_EQ(file_info_->numFreePages(), num_pages);
  file_info_ = nullptr;  // close file descriptor;

  ASSERT_TRUE(bf::exists(data_file_name));
  ASSERT_EQ(bf::file_size(data_file_name), num_pages * page_size);

  auto fd = heavyai::fopen(data_file_name, "r");
  int32_t header_size = 0;
  int8_t* buf = reinterpret_cast<int8_t*>(&header_size);
  for (size_t i = 0; i < page_size * num_pages; i += page_size) {
    File_Namespace::read(fd, i, sizeof(int32_t), buf);
    // Check that all pages have zero-ed headers
    ASSERT_EQ(*(reinterpret_cast<int32_t*>(buf)), 0);
  }
}

TEST_F(FileInfoTest, Size) {
  file_info_->initNewFile();

  // All pages should be included in the size.
  EXPECT_EQ(file_info_->size(), page_size * num_pages);
}

TEST_F(FileInfoTest, Available) {
  file_info_->initNewFile();

  // All pages are available by default.
  ASSERT_EQ(file_info_->available(), num_pages * page_size);

  file_info_->getFreePage();

  // One less page is now available.
  EXPECT_EQ(file_info_->available(), (num_pages - 1) * page_size);
}

TEST_F(FileInfoTest, NumFreePages) {
  file_info_->initNewFile();

  // All pages are free by default.
  ASSERT_EQ(file_info_->numFreePages(), num_pages);

  file_info_->getFreePage();

  // One less page is now free.
  EXPECT_EQ(file_info_->numFreePages(), num_pages - 1);
}

TEST_F(FileInfoTest, Used) {
  file_info_->initNewFile();

  // No bytes used if no pages used.
  ASSERT_EQ(file_info_->used(), 0U);

  file_info_->getFreePage();

  // One page is now in use.
  EXPECT_EQ(file_info_->used(), page_size);
}

TEST_F(FileInfoTest, getFreePages) {
  file_info_->initNewFile();

  // All pages should be free by default.
  std::set<size_t> free_pages;
  for (size_t i = 0; i < num_pages; ++i) {
    free_pages.emplace(i);
  }
  ASSERT_EQ(file_info_->getFreePages(), free_pages);

  auto page_id = file_info_->getFreePage();
  free_pages.erase(page_id);

  // Reserved page should be missing.
  EXPECT_EQ(file_info_->getFreePages(), free_pages);
}

TEST_F(FileInfoTest, Print) {
  file_info_->initNewFile();

  std::stringstream ss;
  ss << "File: " << data_file_id << std::endl;
  ss << "Size: " << num_pages * page_size << std::endl;
  ss << "Used: " << 0 << std::endl;
  ss << "Free: " << num_pages * page_size << std::endl;

  ASSERT_EQ(file_info_->print(), ss.str());
}

class FreePageDeferredTest : public FileInfoTest {};
TEST_F(FreePageDeferredTest, Reserved) {
  // Reserve a page.
  file_info_->initNewFile();
  auto page = file_info_->getFreePage();

  ASSERT_GT(page, -1);

  // Write to the reserved page
  std::vector<int32_t> write_buf(page_size / sizeof(int32_t), 1);
  file_info_->write(page, page_size, reinterpret_cast<const int8_t*>(write_buf.data()));

  // Free the reserved page.
  file_info_->freePageDeferred(page);

  // Page should be back in free pages.
  auto free_pages = file_info_->getFreePages();
  EXPECT_NE(free_pages.count(page), 0U);

  // Page should not have a zero-ed header.
  file_info_ = nullptr;  // close file descriptor.
  auto buf = readFromFile<int32_t>(data_file_name, page * page_size, 1);
  ASSERT_EQ(buf[0], 1);
}

TEST_F(FreePageDeferredTest, AlreadyFree) {
  // Reserve a page.
  file_info_->initNewFile();

  // Write to the reserved page
  std::vector<int32_t> write_buf(page_size / sizeof(int32_t), 1);
  file_info_->write(
      page_size, page_size, reinterpret_cast<const int8_t*>(write_buf.data()));

  // Free the reserved page.
  file_info_->freePageDeferred(1);

  // Page should be in free pages.
  auto free_pages = file_info_->getFreePages();
  EXPECT_NE(free_pages.count(1), 0U);

  // Page should not have a zero-ed header.
  file_info_ = nullptr;  // Close file desc.
  auto buf = readFromFile<int32_t>(data_file_name, page_size, 1);
  ASSERT_EQ(buf[0], 1);
}

class FreePageImmediateTest : public FileInfoTest {};
TEST_F(FreePageImmediateTest, Reserved) {
  // Reserve a page.
  file_info_->initNewFile();
  auto page = file_info_->getFreePage();

  ASSERT_GT(page, -1);

  // Write to the reserved page
  std::vector<int32_t> write_buf(page_size / sizeof(int32_t), 1);
  file_info_->write(page, page_size, reinterpret_cast<const int8_t*>(write_buf.data()));

  // Free the reserved page.
  file_info_->freePageImmediate(page);

  // Page should be back in free pages.
  auto free_pages = file_info_->getFreePages();
  EXPECT_NE(free_pages.count(page), 0U);

  // Page should have a zero-ed header.
  file_info_ = nullptr;  // close file descriptor.
  auto buf = readFromFile<int32_t>(data_file_name, page * page_size, 1);
  ASSERT_EQ(buf[0], 0);
}

TEST_F(FreePageImmediateTest, AlreadyFree) {
  // Reserve a page.
  file_info_->initNewFile();

  // Write to the reserved page
  std::vector<int32_t> write_buf(page_size / sizeof(int32_t), 1);
  file_info_->write(
      page_size, page_size, reinterpret_cast<const int8_t*>(write_buf.data()));

  // Free the reserved page.
  file_info_->freePageImmediate(1);

  // Page should be in free pages.
  auto free_pages = file_info_->getFreePages();
  EXPECT_NE(free_pages.count(1), 0U);

  // Page should have a zero-ed header.
  file_info_ = nullptr;  // Close file desc.
  auto buf = readFromFile<int32_t>(data_file_name, page_size, 1);
  ASSERT_EQ(buf[0], 0);
}

class GetFreePageTest : public FileInfoTest {};
TEST_F(GetFreePageTest, NoPagesAvailable) {
  // Verify we have no pages available.
  EXPECT_EQ(file_info_->numFreePages(), 0U);

  // Verify we don't get a page when requested.
  auto page = file_info_->getFreePage();
  EXPECT_EQ(page, -1);
  ASSERT_EQ(file_info_->numFreePages(), 0U);
}

TEST_F(GetFreePageTest, PagesAvailable) {
  // Verify we have some pages available.
  file_info_->initNewFile();
  EXPECT_GT(file_info_->numFreePages(), 0U);

  // Request free page.
  auto page = file_info_->getFreePage();

  // Verify we have gotten a page.
  ASSERT_GT(page, -1);

  // Verify that the page we got is no longer free.
  auto free_pages = file_info_->getFreePages();
  ASSERT_EQ(free_pages.count(page), 0U);
}

class FreePageTest : public FileInfoTest {};
TEST_F(FreePageTest, Delete) {
  // Initialize pages with zeroed headers.
  file_info_->initNewFile();

  // Delete page.
  file_info_->freePage(1, false, 2);

  // Verify page header has been overwritten with contingent and epoch.
  file_info_ = nullptr;  // Close file desc.
  auto buf = readFromFile<int32_t>(data_file_name, page_size + sizeof(int32_t), 2);
  EXPECT_EQ(buf[0], -1);  // delete contingent.
  ASSERT_EQ(buf[1], 2);   // epoch.
}

TEST_F(FreePageTest, Rolloff) {
  // Initialize pages with zeroed headers.
  file_info_->initNewFile();

  // Delete page.
  file_info_->freePage(1, true, 2);

  // Verify page header has been overwritten with contingent and epoch.
  file_info_ = nullptr;  // Close file desc.
  auto buf = readFromFile<int32_t>(data_file_name, page_size + sizeof(int32_t), 2);
  EXPECT_EQ(buf[0], -2);  // rolloff contingent.
  ASSERT_EQ(buf[1], 2);   // epoch.
}

TEST_F(FileInfoTest, RecoverPage) {
  // Initialize pages with zeroed headers.
  file_info_->initNewFile();

  // Delete page.
  file_info_->freePage(1, false, 2);

  // Recover page (replace the contingent and epoch with chunk key values).
  file_info_->recoverPage({1, 2}, 1);

  file_info_ = nullptr;  // Close file desc.
  auto buf = readFromFile<int32_t>(data_file_name, page_size + sizeof(int32_t), 2);
  EXPECT_EQ(buf[0], 1);  // contingent replaced with db_id.
  ASSERT_EQ(buf[1], 2);  // epoch replaced with tb_id.
}

TEST_F(FileInfoTest, Write) {
  // Write data to file
  int8_t write_buf[8]{1, 2, 3, 4, 5, 6, 7, 8};
  file_info_->write(0, 8, write_buf);

  file_info_ = nullptr;  // Close file desc.
  // Verify read from disk is what we expected to write.
  auto read_buf = readFromFile<int8_t>(data_file_name, 0, 8);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(read_buf[i], write_buf[i]);
  }
}

TEST_F(FileInfoTest, Read) {
  // Write data to file
  int8_t write_buf[8]{1, 2, 3, 4, 5, 6, 7, 8};
  file_info_->write(0, 8, write_buf);

  int8_t read_buf[8];
  file_info_->read(0, 8, read_buf);

  // Verify we have read what we expected.
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(read_buf[i], write_buf[i]);
  }
}

class OpenExistingFileTest : public FileInfoTest {
 protected:
  constexpr static const char* source_data_file =
      "../../Tests/FileMgrDataFiles/0.64.data";
  constexpr static const char* source_meta_file =
      "../../Tests/FileMgrDataFiles/1.128.data";

  void SetUp() override {
    bf::remove_all(test_data_dir);
    bf::create_directory(test_data_dir);

    // Tests need a FileMgr to access epoch data.
    fsi_ = std::make_shared<ForeignStorageInterface>();
    gfm_ = std::make_unique<File_Namespace::GlobalFileMgr>(
        0, fsi_, test_data_dir, 0, page_size, meta_page_size);

    // The last checkpointed epoch for the pre-created files is actually "2", but the
    // FileMgr will automatically increment the epoch during initialization so we need to
    // override the epoch to "1" so that we get "2" once it's done.
    // This is necessary because openExistingFile() is expected to be called as part of FM
    // initialization before we increment the epoch, so if we were callilng it normally,
    // the epoch would be read as "2", then we call openExistingFile(), then we increment.
    // But here we are pre-initializing a FM and calling the function after the epoch is
    // incremented (so as not to depend on any of the initialiation code).
    fm_ = std::make_unique<File_Namespace::FileMgr>(
        0, gfm_.get(), File_Namespace::TablePair{1, 1}, -1, 0, 1 /* epoch */);
  }

  // These methods were used to create the data files used for comparison purposes.
  void createTestingFiles(const std::string& gfm_path) const {
    CHECK_NE(gfm_path, test_data_dir)
        << "Can't create new test files in a directory that will be used.";
    bf::remove_all(gfm_path);

    // Need to setup a temporary FileMgr to create files.
    auto fsi = std::make_shared<ForeignStorageInterface>();
    auto gfm = std::make_unique<File_Namespace::GlobalFileMgr>(
        0, fsi, gfm_path, 0, page_size, meta_page_size);
    auto fm = dynamic_cast<File_Namespace::FileMgr*>(gfm->getFileMgr(1, 1));

    // Data to write.
    auto sql_info = SQLTypeInfo{kINT};
    std::vector<int32_t> int_data{1, 2, 3, 4, 5, 6, 7, 8};
    auto data_size = int_data.size() * sizeof(int32_t);

    // Normal checkpointed chunk (should be present).
    auto int_buf = fm->createBuffer({1, 1, 1, 0});
    int_buf->initEncoder(sql_info);

    // A chunk that will be written, but never checkpointed.
    auto uncheckpointed_buf = fm->createBuffer({1, 1, 1, 2});
    uncheckpointed_buf->initEncoder(sql_info);

    // Chunk that is deleted and the delete has been checkpointed (should be gone).
    auto deleted_buf = fm->createBuffer({1, 1, 1, 1});
    deleted_buf->initEncoder(sql_info);

    // Chunk that was written, then checkpointed, then deleted without checkpoint (should
    // be present).
    auto uncheckpointed_deleted_buf = fm->createBuffer({1, 1, 1, 3});
    uncheckpointed_deleted_buf->initEncoder(sql_info);

    // All the appends for the first checkpoint.
    int_buf->append(reinterpret_cast<int8_t*>(int_data.data()), data_size);
    deleted_buf->append(reinterpret_cast<int8_t*>(int_data.data()), data_size);
    uncheckpointed_deleted_buf->append(reinterpret_cast<int8_t*>(int_data.data()),
                                       data_size);
    int_buf->append(reinterpret_cast<int8_t*>(int_data.data()), data_size);
    fm->checkpoint();  // Checkpointed with epoch '2'.

    // Delete one buffer before second checkpoint.
    fm->deleteBuffer({1, 1, 1, 1});
    fm->checkpoint();  // Checkpointed with epoch '3'.

    uncheckpointed_buf->append(reinterpret_cast<int8_t*>(int_data.data()), data_size);

    fm->deleteBuffer({1, 1, 1, 3});  // Uncheckpointed delete
  }

  std::shared_ptr<ForeignStorageInterface> fsi_;
  std::unique_ptr<File_Namespace::GlobalFileMgr> gfm_;
  std::unique_ptr<File_Namespace::FileMgr> fm_;
};

TEST_F(OpenExistingFileTest, Data) {
  ASSERT_EQ(fm_->epoch(1, 1), 2) << "FM was not initialized correctly.";

  // Fetch source file.
  bf::copy(source_data_file, data_file_name);

  auto fd = heavyai::fopen(data_file_name, "r+w");
  File_Namespace::FileInfo file_info(fm_.get(), data_file_id, fd, page_size, num_pages);

  std::vector<File_Namespace::HeaderInfo> headers;
  file_info.openExistingFile(headers);

  EXPECT_EQ(file_info.numFreePages(), 13U);
  ASSERT_EQ(headers.size(), 3U);

  // TODO(Misiu): Implement HeaderInfo/Page ==() operator to simplify these comparisons.

  // Normally checkpointed buffer.
  EXPECT_EQ(headers[0].chunkKey, (ChunkKey{1, 1, 1, 0}));
  EXPECT_EQ(headers[0].pageId, 0);  // Page's ordering within buffer.
  EXPECT_EQ(headers[0].versionEpoch, 1);
  EXPECT_EQ(headers[0].page.fileId, data_file_id);
  EXPECT_EQ(headers[0].page.pageNum, 0U);  // First write.

  // Uncheckpointed deleted buffer (restored).
  EXPECT_EQ(headers[1].chunkKey, (ChunkKey{1, 1, 1, 3}));
  EXPECT_EQ(headers[1].pageId, 0);  // Page's ordering within buffer.
  EXPECT_EQ(headers[1].versionEpoch, 1);
  EXPECT_EQ(headers[1].page.fileId, data_file_id);
  EXPECT_EQ(headers[1].page.pageNum, 2U);  // Third write.

  // Second append to checkpointed buffer takes pageNum after checkpointed buffer,
  // deleted buffer, and uncheckpointed_deleted buffer (4th page).
  EXPECT_EQ(headers[2].chunkKey, (ChunkKey{1, 1, 1, 0}));
  EXPECT_EQ(headers[2].pageId, 1);  // Page's ordering within buffer.
  EXPECT_EQ(headers[2].versionEpoch, 1);
  EXPECT_EQ(headers[2].page.fileId, data_file_id);
  EXPECT_EQ(headers[2].page.pageNum, 3U);  // Fourth write.
}

TEST_F(OpenExistingFileTest, Metadata) {
  ASSERT_EQ(fm_->epoch(1, 1), 2) << "FM was not initialized correctly.";

  // Fetch source file.
  bf::copy(source_meta_file, meta_file_name);

  auto fd = heavyai::fopen(meta_file_name, "r+w");
  File_Namespace::FileInfo file_info(
      fm_.get(), meta_file_id, fd, meta_page_size, num_pages);

  std::vector<File_Namespace::HeaderInfo> headers;
  file_info.openExistingFile(headers);

  ASSERT_EQ(headers.size(), 2U);

  EXPECT_EQ(headers[0].chunkKey, (ChunkKey{1, 1, 1, 0}));
  EXPECT_EQ(headers[0].pageId, -1);
  EXPECT_EQ(headers[0].versionEpoch, 1);
  EXPECT_EQ(headers[0].page.fileId, meta_file_id);
  EXPECT_EQ(headers[0].page.pageNum, 0U);

  EXPECT_EQ(headers[1].chunkKey, (ChunkKey{1, 1, 1, 3}));
  EXPECT_EQ(headers[1].pageId, -1);
  EXPECT_EQ(headers[1].versionEpoch, 1);
  EXPECT_EQ(headers[1].page.fileId, meta_file_id);
  EXPECT_EQ(headers[1].page.pageNum, 2U);  // 2 because page 1 was deleted.
}

TEST_F(FileInfoTest, SyncToDisk) {
  // Write data to file
  int8_t write_buf[8]{1, 2, 3, 4, 5, 6, 7, 8};
  file_info_->write(0, 8, write_buf);
  EXPECT_TRUE(file_info_->isDirty);

  // Sync file descriptor to disk.
  file_info_->syncToDisk();

  // The file will clear the dirty flag if all the flushing system calls succeed.
  EXPECT_FALSE(file_info_->isDirty);

  file_info_ = nullptr;  // Close file desc.
  // Verify read from disk is what we expected to write.
  auto read_buf = readFromFile<int8_t>(data_file_name, 0, 8);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(read_buf[i], write_buf[i]);
  }
}

// TODO(Misiu): Add concurrency tests for FileInfo.

class FileMgrTest : public testing::Test {
 protected:
  inline static const std::string TEST_DATA_DIR{"./test_dir"};
  inline static const ChunkKey TEST_CHUNK_KEY{1, 1, 1, 0};

  void SetUp() override {
    initializeGlobalFileMgr();
    initializeChunk(1);
  }

  void TearDown() override { bf::remove_all(TEST_DATA_DIR); }

  void initializeGlobalFileMgr() {
    bf::remove_all(TEST_DATA_DIR);
    global_file_mgr_ = std::make_unique<File_Namespace::GlobalFileMgr>(
        0, std::make_shared<ForeignStorageInterface>(), TEST_DATA_DIR, 0);
  }

  File_Namespace::FileMgr* getFileMgr() {
    auto file_mgr = global_file_mgr_->getFileMgr(TEST_CHUNK_KEY[CHUNK_KEY_DB_IDX],
                                                 TEST_CHUNK_KEY[CHUNK_KEY_TABLE_IDX]);
    return dynamic_cast<File_Namespace::FileMgr*>(file_mgr);
  }

  void initializeChunk(int32_t value) {
    auto file_mgr = getFileMgr();
    auto buffer = file_mgr->createBuffer(TEST_CHUNK_KEY);
    buffer->initEncoder(SQLTypeInfo{kINT});
    std::vector<int32_t> data{value};
    writeData(buffer, data, 0);
    file_mgr->checkpoint();
  }

  void setMaxRollbackEpochs(const int32_t max_rollback_epochs) {
    File_Namespace::FileMgrParams file_mgr_params;
    file_mgr_params.max_rollback_epochs = max_rollback_epochs;
    global_file_mgr_->setFileMgrParams(TEST_CHUNK_KEY[CHUNK_KEY_DB_IDX],
                                       TEST_CHUNK_KEY[CHUNK_KEY_TABLE_IDX],
                                       file_mgr_params);
  }

  void compareBuffers(AbstractBuffer* left_buffer,
                      AbstractBuffer* right_buffer,
                      size_t num_bytes) {
    std::vector<int8_t> left_array(num_bytes);
    std::vector<int8_t> right_array(num_bytes);
    left_buffer->read(left_array.data(), num_bytes);
    right_buffer->read(right_array.data(), num_bytes);
    ASSERT_EQ(left_array, right_array);
    ASSERT_EQ(left_buffer->hasEncoder(), right_buffer->hasEncoder());
  }

  void compareMetadata(const std::shared_ptr<ChunkMetadata> lhs_metadata,
                       const std::shared_ptr<ChunkMetadata> rhs_metadata) {
    SQLTypeInfo lhs_sqltypeinfo = lhs_metadata->sqlType;
    SQLTypeInfo rhs_sqltypeinfo = rhs_metadata->sqlType;
    ASSERT_EQ(lhs_sqltypeinfo.get_type(), rhs_sqltypeinfo.get_type());
    ASSERT_EQ(lhs_sqltypeinfo.get_subtype(), rhs_sqltypeinfo.get_subtype());
    ASSERT_EQ(lhs_sqltypeinfo.get_dimension(), rhs_sqltypeinfo.get_dimension());
    ASSERT_EQ(lhs_sqltypeinfo.get_scale(), rhs_sqltypeinfo.get_scale());
    ASSERT_EQ(lhs_sqltypeinfo.get_notnull(), rhs_sqltypeinfo.get_notnull());
    ASSERT_EQ(lhs_sqltypeinfo.get_comp_param(), rhs_sqltypeinfo.get_comp_param());
    ASSERT_EQ(lhs_sqltypeinfo.get_size(), rhs_sqltypeinfo.get_size());

    ASSERT_EQ(lhs_metadata->numBytes, rhs_metadata->numBytes);
    ASSERT_EQ(lhs_metadata->numElements, rhs_metadata->numElements);

    ChunkStats lhs_chunk_stats = lhs_metadata->chunkStats;
    ChunkStats rhs_chunk_stats = rhs_metadata->chunkStats;
    ASSERT_EQ(lhs_chunk_stats.min.intval, rhs_chunk_stats.min.intval);
    ASSERT_EQ(lhs_chunk_stats.max.intval, rhs_chunk_stats.max.intval);
    ASSERT_EQ(lhs_chunk_stats.has_nulls, rhs_chunk_stats.has_nulls);
  }

  std::shared_ptr<ChunkMetadata> getMetadataForBuffer(AbstractBuffer* buffer) {
    const std::shared_ptr<ChunkMetadata> metadata = std::make_shared<ChunkMetadata>();
    buffer->getEncoder()->getMetadata(metadata);
    return metadata;
  }

  void compareBuffersAndMetadata(AbstractBuffer* left_buffer,
                                 AbstractBuffer* right_buffer) {
    ASSERT_TRUE(left_buffer->hasEncoder());
    ASSERT_TRUE(right_buffer->hasEncoder());
    ASSERT_TRUE(left_buffer->getEncoder());
    ASSERT_TRUE(right_buffer->getEncoder());
    ASSERT_EQ(left_buffer->size(), getMetadataForBuffer(left_buffer)->numBytes);
    ASSERT_EQ(right_buffer->size(), getMetadataForBuffer(right_buffer)->numBytes);
    compareMetadata(getMetadataForBuffer(left_buffer),
                    getMetadataForBuffer(right_buffer));
    compareBuffers(left_buffer, right_buffer, left_buffer->size());
  }

  int8_t* getDataPtr(std::vector<int32_t>& data_vector) {
    return reinterpret_cast<int8_t*>(data_vector.data());
  }

  void appendData(AbstractBuffer* data_buffer, std::vector<int32_t>& append_data) {
    writeData(data_buffer, append_data, -1);
  }

  void writeData(AbstractBuffer* data_buffer,
                 std::vector<int32_t>& write_data,
                 const size_t offset) {
    CHECK(data_buffer->hasEncoder());
    SQLTypeInfo sql_type_info = getMetadataForBuffer(data_buffer)->sqlType;
    int8_t* write_ptr = getDataPtr(write_data);
    // appendData is a misnomer, with the offset we are overwriting part of the buffer
    data_buffer->getEncoder()->appendData(
        write_ptr, write_data.size(), sql_type_info, false /*replicating*/, offset);
  }

  std::unique_ptr<File_Namespace::GlobalFileMgr> global_file_mgr_;
};

TEST_F(FileMgrTest, putBuffer_update) {
  TestHelpers::TestBuffer source_buffer{std::vector<int32_t>{1}};
  source_buffer.setUpdated();
  auto file_mgr = getFileMgr();
  AbstractBuffer* file_buffer = file_mgr->putBuffer(TEST_CHUNK_KEY, &source_buffer, 4);
  compareBuffersAndMetadata(&source_buffer, file_buffer);
  ASSERT_FALSE(source_buffer.isAppended());
  ASSERT_FALSE(source_buffer.isUpdated());
  ASSERT_FALSE(source_buffer.isDirty());
}

TEST_F(FileMgrTest, putBuffer_subwrite) {
  TestHelpers::TestBuffer source_buffer{SQLTypeInfo{kINT}};
  int8_t temp_array[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  source_buffer.write(temp_array, 8);
  auto file_mgr = getFileMgr();
  AbstractBuffer* file_buffer = file_mgr->putBuffer(TEST_CHUNK_KEY, &source_buffer, 4);
  compareBuffers(&source_buffer, file_buffer, 4);
}

TEST_F(FileMgrTest, putBuffer_exists) {
  TestHelpers::TestBuffer source_buffer{SQLTypeInfo{kINT}};
  int8_t temp_array[4] = {1, 2, 3, 4};
  source_buffer.write(temp_array, 4);
  auto file_mgr = getFileMgr();
  file_mgr->putBuffer(TEST_CHUNK_KEY, &source_buffer, 4);
  file_mgr->checkpoint();
  source_buffer.write(temp_array, 4);
  AbstractBuffer* file_buffer = file_mgr->putBuffer(TEST_CHUNK_KEY, &source_buffer, 4);
  compareBuffersAndMetadata(&source_buffer, file_buffer);
}

TEST_F(FileMgrTest, putBuffer_append) {
  TestHelpers::TestBuffer source_buffer{std::vector<int32_t>{1}};
  int8_t temp_array[4] = {1, 2, 3, 4};
  source_buffer.append(temp_array, 4);
  auto file_mgr = getFileMgr();
  AbstractBuffer* file_buffer = file_mgr->putBuffer(TEST_CHUNK_KEY, &source_buffer, 8);
  compareBuffersAndMetadata(&source_buffer, file_buffer);
}

TEST_F(FileMgrTest, put_checkpoint_get) {
  TestHelpers::TestBuffer source_buffer{std::vector<int32_t>{1}};
  std::vector<int32_t> data_v1 = {1, 2, 3, 5, 7};
  appendData(&source_buffer, data_v1);
  auto file_mgr = getFileMgr();
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  AbstractBuffer* file_buffer_put =
      file_mgr->putBuffer(TEST_CHUNK_KEY, &source_buffer, 24);
  ASSERT_TRUE(file_buffer_put->isDirty());
  ASSERT_FALSE(file_buffer_put->isUpdated());
  ASSERT_TRUE(file_buffer_put->isAppended());
  ASSERT_EQ(file_buffer_put->size(), static_cast<size_t>(24));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  file_mgr->checkpoint();
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  AbstractBuffer* file_buffer_get = file_mgr->getBuffer(TEST_CHUNK_KEY, 24);
  ASSERT_EQ(file_buffer_put, file_buffer_get);
  CHECK(!(file_buffer_get->isDirty()));
  CHECK(!(file_buffer_get->isUpdated()));
  CHECK(!(file_buffer_get->isAppended()));
  ASSERT_EQ(file_buffer_get->size(), static_cast<size_t>(24));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  compareBuffersAndMetadata(&source_buffer, file_buffer_get);
}

TEST_F(FileMgrTest, put_checkpoint_get_double_write) {
  TestHelpers::TestBuffer source_buffer{std::vector<int32_t>{1}};
  std::vector<int32_t> data_v1 = {1, 2, 3, 5, 7};
  std::vector<int32_t> data_v2 = {11, 13, 17, 19};
  appendData(&source_buffer, data_v1);
  auto file_mgr = getFileMgr();
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  file_mgr->putBuffer(TEST_CHUNK_KEY, &source_buffer, 24);
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  file_mgr->checkpoint();
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  AbstractBuffer* file_buffer = file_mgr->getBuffer(TEST_CHUNK_KEY, 24);
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  ASSERT_FALSE(file_buffer->isDirty());
  ASSERT_EQ(file_buffer->size(), static_cast<size_t>(24));
  compareBuffersAndMetadata(&source_buffer, file_buffer);
  appendData(file_buffer, data_v2);
  ASSERT_TRUE(file_buffer->isDirty());
  ASSERT_EQ(file_buffer->size(), static_cast<size_t>(40));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  appendData(file_buffer, data_v2);
  CHECK(file_buffer->isDirty());
  ASSERT_EQ(file_buffer->size(), static_cast<size_t>(56));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  file_mgr->checkpoint();
  CHECK(!(file_buffer->isDirty()));
  ASSERT_EQ(file_buffer->size(), static_cast<size_t>(56));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 3);
  appendData(&source_buffer, data_v2);
  appendData(&source_buffer, data_v2);
  compareBuffersAndMetadata(&source_buffer, file_buffer);
}

TEST_F(FileMgrTest, buffer_append_and_recovery) {
  TestHelpers::TestBuffer source_buffer{SQLTypeInfo{kINT}};
  std::vector<int32_t> initial_value{1};
  appendData(&source_buffer, initial_value);
  ASSERT_EQ(getMetadataForBuffer(&source_buffer)->numElements, static_cast<size_t>(1));

  std::vector<int32_t> data_v1 = {1, 2, 3, 5, 7};
  std::vector<int32_t> data_v2 = {11, 13, 17, 19};
  appendData(&source_buffer, data_v1);
  {
    auto file_mgr = getFileMgr();
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
    AbstractBuffer* file_buffer = file_mgr->putBuffer(TEST_CHUNK_KEY, &source_buffer, 24);
    file_mgr->checkpoint();
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
    ASSERT_EQ(getMetadataForBuffer(&source_buffer)->numElements, static_cast<size_t>(6));
    ASSERT_EQ(getMetadataForBuffer(file_buffer)->numElements, static_cast<size_t>(6));
    SCOPED_TRACE("Buffer Append and Recovery - Compare #1");
    compareBuffersAndMetadata(&source_buffer, file_buffer);

    // Now write data we will not checkpoint
    appendData(file_buffer, data_v1);
    ASSERT_EQ(file_buffer->size(), static_cast<size_t>(44));
    // Now close filemgr to test recovery
    global_file_mgr_->closeFileMgr(TEST_CHUNK_KEY[CHUNK_KEY_DB_IDX],
                                   TEST_CHUNK_KEY[CHUNK_KEY_TABLE_IDX]);
  }

  {
    auto file_mgr = getFileMgr();
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
    ChunkMetadataVector chunkMetadataVector;
    file_mgr->getChunkMetadataVecForKeyPrefix(chunkMetadataVector, TEST_CHUNK_KEY);
    ASSERT_EQ(chunkMetadataVector.size(), static_cast<size_t>(1));
    ASSERT_EQ(std::memcmp(chunkMetadataVector[0].first.data(), TEST_CHUNK_KEY.data(), 16),
              0);
    ASSERT_EQ(chunkMetadataVector[0].first, TEST_CHUNK_KEY);
    std::shared_ptr<ChunkMetadata> chunk_metadata = chunkMetadataVector[0].second;
    ASSERT_EQ(chunk_metadata->numBytes, static_cast<size_t>(24));
    ASSERT_EQ(chunk_metadata->numElements, static_cast<size_t>(6));
    AbstractBuffer* file_buffer =
        file_mgr->getBuffer(TEST_CHUNK_KEY, chunk_metadata->numBytes);
    {
      SCOPED_TRACE("Buffer Append and Recovery - Compare #2");
      compareBuffersAndMetadata(&source_buffer, file_buffer);
    }
    appendData(&source_buffer, data_v2);
    appendData(file_buffer, data_v2);

    file_mgr->checkpoint();
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 3);
    {
      SCOPED_TRACE("Buffer Append and Recovery - Compare #3");
      compareBuffersAndMetadata(&source_buffer, file_buffer);
    }
  }
}

TEST_F(FileMgrTest, buffer_update_and_recovery) {
  std::vector<int32_t> data_v1 = {
      2,
      3,
      5,
      7,
      11};  // Make first element different than 1 stored in col1 at t0 so that we can
            // ensure updates and rollbacks show a change in col[0]
  std::vector<int32_t> data_v2 = {13, 17, 19, 23};
  {
    EXPECT_EQ(global_file_mgr_->getTableEpoch(TEST_CHUNK_KEY[CHUNK_KEY_DB_IDX],
                                              TEST_CHUNK_KEY[CHUNK_KEY_TABLE_IDX]),
              std::size_t(1));
    TestHelpers::TestBuffer source_buffer{SQLTypeInfo{kINT}};
    std::vector<int32_t> initial_value{1};
    appendData(&source_buffer, initial_value);
    source_buffer.clearDirtyBits();

    auto file_mgr = getFileMgr();
    AbstractBuffer* file_buffer = file_mgr->getBuffer(TEST_CHUNK_KEY);
    ASSERT_FALSE(source_buffer.isDirty());
    ASSERT_FALSE(source_buffer.isUpdated());
    ASSERT_FALSE(source_buffer.isAppended());
    ASSERT_FALSE(file_buffer->isDirty());
    ASSERT_FALSE(file_buffer->isUpdated());
    ASSERT_FALSE(file_buffer->isAppended());
    ASSERT_EQ(file_buffer->size(), static_cast<size_t>(4));
    {
      SCOPED_TRACE("Buffer Update and Recovery - Compare #1");
      compareBuffersAndMetadata(file_buffer, &source_buffer);
    }
    writeData(file_buffer, data_v1, 0);
    ASSERT_TRUE(file_buffer->isDirty());
    ASSERT_TRUE(file_buffer->isUpdated());
    ASSERT_TRUE(file_buffer->isAppended());
    ASSERT_EQ(file_buffer->size(), static_cast<size_t>(20));
    {
      std::vector<int32_t> file_buffer_data(file_buffer->size() / sizeof(int32_t));
      file_buffer->read(reinterpret_cast<int8_t*>(file_buffer_data.data()),
                        file_buffer->size());
      ASSERT_EQ(file_buffer_data[0], 2);
      ASSERT_EQ(file_buffer_data[1], 3);
      ASSERT_EQ(file_buffer_data[2], 5);
      ASSERT_EQ(file_buffer_data[3], 7);
      ASSERT_EQ(file_buffer_data[4], 11);
      std::shared_ptr<ChunkMetadata> file_chunk_metadata =
          getMetadataForBuffer(file_buffer);
      ASSERT_EQ(file_chunk_metadata->numElements, static_cast<size_t>(5));
      ASSERT_EQ(file_chunk_metadata->numBytes, static_cast<size_t>(20));
      ASSERT_EQ(file_chunk_metadata->chunkStats.min.intval, 2);
      ASSERT_EQ(file_chunk_metadata->chunkStats.max.intval, 11);
      ASSERT_EQ(file_chunk_metadata->chunkStats.has_nulls, false);
    }
    file_mgr->checkpoint();
    EXPECT_EQ(global_file_mgr_->getTableEpoch(TEST_CHUNK_KEY[CHUNK_KEY_DB_IDX],
                                              TEST_CHUNK_KEY[CHUNK_KEY_TABLE_IDX]),
              std::size_t(2));
    ASSERT_FALSE(file_buffer->isDirty());
    ASSERT_FALSE(file_buffer->isUpdated());
    ASSERT_FALSE(file_buffer->isAppended());

    source_buffer.reset();
    file_mgr->fetchBuffer(
        TEST_CHUNK_KEY,
        &source_buffer,
        20);  // Dragons here: if we didn't unpin andy flush the data, the first value
              // will be 1, and not 2, as we only fetch the portion of data we don't have
              // from FileMgr (there's no DataMgr versioning currently, so for example,
              // for updates we just flush the in-memory buffers to get a clean start)
    ASSERT_FALSE(source_buffer.isDirty());
    ASSERT_FALSE(source_buffer.isUpdated());
    ASSERT_FALSE(source_buffer.isAppended());
    ASSERT_EQ(file_buffer->size(), static_cast<size_t>(20));
    {
      std::vector<int32_t> source_buffer_data(source_buffer.size() / sizeof(int32_t));
      source_buffer.read(reinterpret_cast<int8_t*>(source_buffer_data.data()),
                         source_buffer.size());
      ASSERT_EQ(source_buffer_data[0], 2);
      ASSERT_EQ(source_buffer_data[1], 3);
      ASSERT_EQ(source_buffer_data[2], 5);
      ASSERT_EQ(source_buffer_data[3], 7);
      ASSERT_EQ(source_buffer_data[4], 11);
      std::shared_ptr<ChunkMetadata> cpu_chunk_metadata =
          getMetadataForBuffer(&source_buffer);
      ASSERT_EQ(cpu_chunk_metadata->numElements, static_cast<size_t>(5));
      ASSERT_EQ(cpu_chunk_metadata->numBytes, static_cast<size_t>(20));
      ASSERT_EQ(cpu_chunk_metadata->chunkStats.min.intval, 2);
      ASSERT_EQ(cpu_chunk_metadata->chunkStats.max.intval, 11);
      ASSERT_EQ(cpu_chunk_metadata->chunkStats.has_nulls, false);
    }
    {
      SCOPED_TRACE("Buffer Update and Recovery - Compare #2");
      compareBuffersAndMetadata(file_buffer, &source_buffer);
    }
    // Now roll back to epoch 1
    File_Namespace::FileMgrParams file_mgr_params;
    file_mgr_params.epoch = 1;
    global_file_mgr_->setFileMgrParams(TEST_CHUNK_KEY[CHUNK_KEY_DB_IDX],
                                       TEST_CHUNK_KEY[CHUNK_KEY_TABLE_IDX],
                                       file_mgr_params);
    file_mgr = getFileMgr();
    file_buffer = file_mgr->getBuffer(TEST_CHUNK_KEY);
    ASSERT_FALSE(file_buffer->isDirty());
    ASSERT_FALSE(file_buffer->isUpdated());
    ASSERT_FALSE(file_buffer->isAppended());
    ASSERT_EQ(file_buffer->size(), static_cast<size_t>(4));
    {
      std::vector<int32_t> file_buffer_data(file_buffer->size() / sizeof(int32_t));
      file_buffer->read(reinterpret_cast<int8_t*>(file_buffer_data.data()),
                        file_buffer->size());
      ASSERT_EQ(file_buffer_data[0], 1);
      std::shared_ptr<ChunkMetadata> file_chunk_metadata =
          getMetadataForBuffer(file_buffer);
      ASSERT_EQ(file_chunk_metadata->numElements, static_cast<size_t>(1));
      ASSERT_EQ(file_chunk_metadata->numBytes, static_cast<size_t>(4));
      ASSERT_EQ(file_chunk_metadata->chunkStats.min.intval, 1);
      ASSERT_EQ(file_chunk_metadata->chunkStats.max.intval, 1);
      ASSERT_EQ(file_chunk_metadata->chunkStats.has_nulls, false);
    }
  }
}

TEST_F(FileMgrTest, capped_metadata) {
  const int rollback_ceiling = 10;
  const int num_data_writes = rollback_ceiling * 2;
  for (int max_rollback_epochs = 0; max_rollback_epochs != rollback_ceiling;
       ++max_rollback_epochs) {
    initializeGlobalFileMgr();
    setMaxRollbackEpochs(max_rollback_epochs);
    initializeChunk(1);
    const auto& capped_chunk_key = TEST_CHUNK_KEY;
    // Have one element already written to key -- epoch should be 2
    ASSERT_EQ(global_file_mgr_->getTableEpoch(capped_chunk_key[0], capped_chunk_key[1]),
              static_cast<size_t>(1));
    File_Namespace::FileMgr* file_mgr = dynamic_cast<File_Namespace::FileMgr*>(
        global_file_mgr_->getFileMgr(capped_chunk_key[0], capped_chunk_key[1]));
    // buffer inside loop
    for (int data_write = 1; data_write <= num_data_writes; ++data_write) {
      std::vector<int32_t> data;
      data.emplace_back(data_write);
      AbstractBuffer* file_buffer = global_file_mgr_->getBuffer(capped_chunk_key);
      appendData(file_buffer, data);
      global_file_mgr_->checkpoint(capped_chunk_key[0], capped_chunk_key[1]);
      ASSERT_EQ(global_file_mgr_->getTableEpoch(capped_chunk_key[0], capped_chunk_key[1]),
                static_cast<size_t>(data_write + 1));
      const size_t num_metadata_pages_expected =
          std::min(data_write + 1, max_rollback_epochs + 1);
      ASSERT_EQ(file_mgr->getNumUsedMetadataPagesForChunkKey(capped_chunk_key),
                num_metadata_pages_expected);
    }
  }
}

class DataCompactionTest : public FileMgrTest {
 protected:
  void SetUp() override {
    TearDown();
    initializeGlobalFileMgr();
  }

  void TearDown() override {
    File_Namespace::FileMgr::setNumPagesPerDataFile(
        File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_DATA_FILE);
    File_Namespace::FileMgr::setNumPagesPerMetadataFile(
        File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_METADATA_FILE);
    FileMgrTest::TearDown();
  }

  ChunkKey getChunkKey(int32_t column_id) {
    auto chunk_key = TEST_CHUNK_KEY;
    chunk_key[CHUNK_KEY_COLUMN_IDX] = column_id;
    return chunk_key;
  }

  void assertStorageStats(uint64_t metadata_file_count,
                          std::optional<uint64_t> free_metadata_page_count,
                          uint64_t data_file_count,
                          std::optional<uint64_t> free_data_page_count) {
    auto stats = global_file_mgr_->getStorageStats(TEST_CHUNK_KEY[CHUNK_KEY_DB_IDX],
                                                   TEST_CHUNK_KEY[CHUNK_KEY_TABLE_IDX]);
    EXPECT_EQ(metadata_file_count, stats.metadata_file_count);
    ASSERT_EQ(free_metadata_page_count.has_value(),
              stats.total_free_metadata_page_count.has_value());
    if (free_metadata_page_count.has_value()) {
      EXPECT_EQ(free_metadata_page_count.value(),
                stats.total_free_metadata_page_count.value());
    }

    EXPECT_EQ(data_file_count, stats.data_file_count);
    ASSERT_EQ(free_data_page_count.has_value(),
              stats.total_free_data_page_count.has_value());
    if (free_data_page_count.has_value()) {
      EXPECT_EQ(free_data_page_count.value(), stats.total_free_data_page_count.value());
    }
  }

  void assertChunkMetadata(AbstractBuffer* buffer, int32_t value) {
    auto metadata = getMetadataForBuffer(buffer);
    EXPECT_EQ(static_cast<size_t>(1), metadata->numElements);
    EXPECT_EQ(sizeof(int32_t), metadata->numBytes);
    EXPECT_FALSE(metadata->chunkStats.has_nulls);
    EXPECT_EQ(value, metadata->chunkStats.min.intval);
    EXPECT_EQ(value, metadata->chunkStats.max.intval);
  }

  void assertBufferValueAndMetadata(int32_t expected_value, int32_t column_id) {
    auto chunk_key = getChunkKey(column_id);
    auto file_mgr = getFileMgr();
    auto buffer = file_mgr->getBuffer(chunk_key, sizeof(int32_t));
    int32_t value;
    buffer->read(reinterpret_cast<int8_t*>(&value), sizeof(int32_t));
    EXPECT_EQ(expected_value, value);
    assertChunkMetadata(buffer, expected_value);
  }

  AbstractBuffer* createBuffer(int32_t column_id) {
    auto chunk_key = getChunkKey(column_id);
    auto buffer = getFileMgr()->createBuffer(chunk_key, DEFAULT_PAGE_SIZE, 0);
    buffer->initEncoder(SQLTypeInfo{kINT});
    return buffer;
  }

  void writeValue(AbstractBuffer* buffer, int32_t value) {
    std::vector<int32_t> data{value};
    writeData(buffer, data, 0);
    getFileMgr()->checkpoint();
  }

  void writeMultipleValues(AbstractBuffer* buffer, int32_t start_value, int32_t count) {
    for (int32_t i = 0; i < count; i++) {
      writeValue(buffer, start_value + i);
    }
  }

  void compactDataFiles() {
    global_file_mgr_->compactDataFiles(TEST_CHUNK_KEY[CHUNK_KEY_DB_IDX],
                                       TEST_CHUNK_KEY[CHUNK_KEY_TABLE_IDX]);
  }

  void deleteFileMgr() {
    global_file_mgr_->closeFileMgr(TEST_CHUNK_KEY[CHUNK_KEY_DB_IDX],
                                   TEST_CHUNK_KEY[CHUNK_KEY_TABLE_IDX]);
  }

  void deleteBuffer(int32_t column_id) {
    auto chunk_key = getChunkKey(column_id);
    auto file_mgr = getFileMgr();
    file_mgr->deleteBuffer(chunk_key);
    file_mgr->checkpoint();
  }
};

TEST_F(DataCompactionTest, DataFileCompaction) {
  // One page per file for the data file (metadata file default
  // configuration of 4096 pages remains the same), so each
  // write creates a new data file.
  File_Namespace::FileMgr::setNumPagesPerDataFile(1);

  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer = createBuffer(1);
  writeValue(buffer, 1);
  // One data file and one metadata file. Data file has no free pages,
  // since the only page available has been used. Metadata file has
  // default (4096) - 1 free pages
  assertStorageStats(1, 4095, 1, 0);

  writeValue(buffer, 2);
  // Second data file created for new page
  assertStorageStats(1, 4094, 2, 0);

  writeValue(buffer, 3);
  // Third data file created for new page
  assertStorageStats(1, 4093, 3, 0);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(3, 1);

  // Setting max rollback epochs to 0 should free up
  // oldest 2 pages
  setMaxRollbackEpochs(0);
  assertStorageStats(1, 4095, 3, 2);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, 1);

  // Data compaction should result in removal of the
  // 2 files with free pages
  compactDataFiles();
  assertStorageStats(1, 4095, 1, 0);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, 1);
}

TEST_F(DataCompactionTest, MetadataFileCompaction) {
  // One page per file for the metadata file (data file default
  // configuration of 256 pages remains the same), so each write
  // creates a new metadata file.
  File_Namespace::FileMgr::setNumPagesPerMetadataFile(1);

  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer = createBuffer(1);
  writeValue(buffer, 1);
  // One data file and one metadata file. Metadata file has no free pages,
  // since the only page available has been used. Data file has
  // default (256) - 1 free pages
  assertStorageStats(1, 0, 1, 255);

  writeValue(buffer, 2);
  // Second metadata file created for new page
  assertStorageStats(2, 0, 1, 254);

  writeValue(buffer, 3);
  // Third metadata file created for new page
  assertStorageStats(3, 0, 1, 253);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(3, 1);

  // Setting max rollback epochs to 0 should free up
  // oldest 2 pages
  setMaxRollbackEpochs(0);
  assertStorageStats(3, 2, 1, 255);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, 1);

  // Data compaction should result in removal of the
  // 2 files with free pages
  compactDataFiles();
  assertStorageStats(1, 0, 1, 255);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, 1);
}

TEST_F(DataCompactionTest, DataAndMetadataFileCompaction) {
  // One page per file for the data and metadata files, so each
  // write creates a new data and metadata file.
  File_Namespace::FileMgr::setNumPagesPerDataFile(1);
  File_Namespace::FileMgr::setNumPagesPerMetadataFile(1);

  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer = createBuffer(1);
  writeValue(buffer, 1);
  // One data file and one metadata file. Both files have no free pages,
  // since the only page available has been used.
  assertStorageStats(1, 0, 1, 0);

  writeValue(buffer, 2);
  // Second data and metadata file created for new page
  assertStorageStats(2, 0, 2, 0);

  writeValue(buffer, 3);
  // Third data and metadata file created for new page
  assertStorageStats(3, 0, 3, 0);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(3, 1);

  // Setting max rollback epochs to 0 should free up
  // oldest 2 pages for both files
  setMaxRollbackEpochs(0);
  assertStorageStats(3, 2, 3, 2);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, 1);

  // Data compaction should result in removal of the
  // 4 files with free pages
  compactDataFiles();
  assertStorageStats(1, 0, 1, 0);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, 1);
}

TEST_F(DataCompactionTest, MultipleChunksPerFile) {
  File_Namespace::FileMgr::setNumPagesPerDataFile(4);

  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer_1 = createBuffer(1);
  auto buffer_2 = createBuffer(2);

  writeValue(buffer_1, 1);
  // One data file and one metadata file. Data file has 3 free pages.
  // Metadata file has default (4096) - 1 free pages
  assertStorageStats(1, 4095, 1, 3);

  writeValue(buffer_2, 1);
  assertStorageStats(1, 4094, 1, 2);

  writeValue(buffer_2, 2);
  assertStorageStats(1, 4093, 1, 1);

  writeValue(buffer_2, 3);
  assertStorageStats(1, 4092, 1, 0);

  writeValue(buffer_2, 4);
  // Second data file created for new page
  assertStorageStats(1, 4091, 2, 3);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(1, 1);
  assertBufferValueAndMetadata(4, 2);

  // Setting max rollback epochs to 0 should free up
  // oldest 3 pages for column "i2"
  setMaxRollbackEpochs(0);

  assertStorageStats(1, 4094, 2, 6);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(1, 1);
  assertBufferValueAndMetadata(4, 2);

  // Data compaction should result in movement of a page from the
  // last data page file to the first data page file and deletion
  // of the last data page file
  compactDataFiles();
  assertStorageStats(1, 4094, 1, 2);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(1, 1);
  assertBufferValueAndMetadata(4, 2);
}

TEST_F(DataCompactionTest, SourceFilePagesCopiedOverMultipleDestinationFiles) {
  File_Namespace::FileMgr::setNumPagesPerDataFile(4);

  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer_1 = createBuffer(1);
  auto buffer_2 = createBuffer(2);

  // First file has 2 pages for each buffer
  writeMultipleValues(buffer_1, 1, 2);
  writeMultipleValues(buffer_2, 1, 2);
  assertStorageStats(1, 4092, 1, 0);

  // Second file has 2 pages for each buffer
  writeMultipleValues(buffer_1, 3, 2);
  writeMultipleValues(buffer_2, 3, 2);
  assertStorageStats(1, 4088, 2, 0);

  // Third file has 2 pages for each buffer
  writeMultipleValues(buffer_1, 5, 2);
  writeMultipleValues(buffer_2, 5, 2);
  assertStorageStats(1, 4084, 3, 0);

  // Fourth file has 3 pages for buffer "i" and 1 for buffer "i2"
  writeMultipleValues(buffer_1, 7, 3);
  writeMultipleValues(buffer_2, 7, 1);
  assertStorageStats(1, 4080, 4, 0);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(9, 1);
  assertBufferValueAndMetadata(7, 2);

  // Free the 7 pages used by buffer "i2" across the 4 files
  deleteBuffer(2);
  assertStorageStats(1, 4087, 4, 7);

  // Verify first buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(9, 1);

  // Files 1, 2, and 3 each have 2 free pages (out of 4 total pages per file).
  // File 4 has 1 free page. Used pages in file 1 should be copied over to
  // files 4 and 3. File 1 should then be deleted.
  compactDataFiles();
  assertStorageStats(1, 4087, 3, 3);

  // Verify first buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(9, 1);
}

TEST_F(DataCompactionTest, SingleDataAndMetadataPages) {
  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer = createBuffer(1);
  writeMultipleValues(buffer, 1, 3);
  assertStorageStats(1, 4093, 1, 253);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(3, 1);

  // Setting max rollback epochs to 0 should free up
  // oldest 2 pages
  setMaxRollbackEpochs(0);
  assertStorageStats(1, 4095, 1, 255);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, 1);

  // Data compaction should result in no changes to files
  compactDataFiles();
  assertStorageStats(1, 4095, 1, 255);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, 1);
}

TEST_F(DataCompactionTest, RecoveryFromCopyPageStatus) {
  // One page per file for the data file (metadata file default
  // configuration of 4096 pages remains the same), so each
  // write creates a new data file.
  File_Namespace::FileMgr::setNumPagesPerDataFile(1);

  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer = createBuffer(1);
  writeMultipleValues(buffer, 1, 3);
  assertStorageStats(1, 4093, 3, 0);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(3, 1);

  // Setting max rollback epochs to 0 should free up
  // oldest 2 pages
  setMaxRollbackEpochs(0);
  assertStorageStats(1, 4095, 3, 2);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, 1);

  // Creating a "pending_data_compaction_0" status file and re-initializing
  // file mgr should result in (resumption of) data compaction and remove the
  // 2 files with free pages
  auto status_file_path =
      getFileMgr()->getFilePath(File_Namespace::FileMgr::COPY_PAGES_STATUS);
  deleteFileMgr();
  std::ofstream status_file{status_file_path.string(), std::ios::out | std::ios::binary};
  status_file.close();

  getFileMgr();
  assertStorageStats(1, 4095, 1, 0);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, 1);
}

TEST_F(DataCompactionTest, RecoveryFromUpdatePageVisibiltyStatus) {
  File_Namespace::FileMgr::setNumPagesPerDataFile(4);

  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer_1 = createBuffer(1);
  auto buffer_2 = createBuffer(2);

  writeMultipleValues(buffer_1, 1, 2);
  assertStorageStats(1, 4094, 1, 2);

  writeMultipleValues(buffer_2, 1, 3);
  assertStorageStats(1, 4091, 2, 3);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(2, 1);
  assertBufferValueAndMetadata(3, 2);

  // Setting max rollback epochs to 0 should free up oldest
  // page for chunk "i1" and oldest 2 pages for chunk "i2"
  setMaxRollbackEpochs(0);
  assertStorageStats(1, 4094, 2, 6);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(2, 1);
  assertBufferValueAndMetadata(3, 2);

  // Creating a "pending_data_compaction_1" status file and re-initializing
  // file mgr should result in (resumption of) data compaction,
  // movement of a page for the last data page file to the first data
  // page file, and deletion of the last data page file
  auto status_file_path =
      getFileMgr()->getFilePath(File_Namespace::FileMgr::COPY_PAGES_STATUS);
  std::ofstream status_file{status_file_path.string(), std::ios::out | std::ios::binary};
  status_file.close();

  auto file_mgr = getFileMgr();
  auto buffer = std::make_unique<int8_t[]>(DEFAULT_PAGE_SIZE);

  // Copy last page for "i2" chunk in last data file to first data file
  int32_t source_file_id{2}, dest_file_id{0};
  auto source_file_info = file_mgr->getFileInfoForFileId(source_file_id);
  source_file_info->read(0, DEFAULT_PAGE_SIZE, buffer.get());

  size_t offset{sizeof(File_Namespace::PageHeaderSizeType)};
  auto destination_file_info = file_mgr->getFileInfoForFileId(dest_file_id);
  destination_file_info->write(offset, DEFAULT_PAGE_SIZE - offset, buffer.get() + offset);
  destination_file_info->syncToDisk();

  File_Namespace::PageHeaderSizeType int_chunk_header_size{24};
  std::vector<File_Namespace::PageMapping> page_mappings{
      {source_file_id, 0, int_chunk_header_size, dest_file_id, 0}};
  file_mgr->writePageMappingsToStatusFile(page_mappings);
  file_mgr->renameCompactionStatusFile(
      File_Namespace::FileMgr::COPY_PAGES_STATUS,
      File_Namespace::FileMgr::UPDATE_PAGE_VISIBILITY_STATUS);
  deleteFileMgr();

  getFileMgr();
  assertStorageStats(1, 4094, 1, 2);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(2, 1);
  assertBufferValueAndMetadata(3, 2);
}

TEST_F(DataCompactionTest, RecoveryFromDeleteEmptyFileStatus) {
  File_Namespace::FileMgr::setNumPagesPerDataFile(4);

  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer_1 = createBuffer(1);
  auto buffer_2 = createBuffer(2);

  writeValue(buffer_1, 1);
  // One data file and one metadata file. Data file has 3 free pages.
  // Metadata file has default (4096) - 1 free pages
  assertStorageStats(1, 4095, 1, 3);

  writeMultipleValues(buffer_2, 1, 3);
  assertStorageStats(1, 4092, 1, 0);

  writeValue(buffer_2, 4);
  // Second data file created for new page
  assertStorageStats(1, 4091, 2, 3);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(1, 1);
  assertBufferValueAndMetadata(4, 2);

  // Delete chunks for "i2"
  deleteBuffer(2);
  assertStorageStats(1, 4095, 2, 7);

  // Verify first buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(1, 1);

  // Creating a "pending_data_compaction_2" status file and re-initializing
  // file mgr should result in (resumption of) data compaction and deletion
  // of the last file that contains only free pages
  auto status_file_path =
      getFileMgr()->getFilePath(File_Namespace::FileMgr::DELETE_EMPTY_FILES_STATUS);
  deleteFileMgr();
  std::ofstream status_file{status_file_path.string(), std::ios::out | std::ios::binary};
  status_file.close();

  getFileMgr();
  assertStorageStats(1, 4095, 1, 3);

  // Verify first buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(1, 1);
}

class MaxRollbackEpochTest : public FileMgrTest {
 protected:
  void SetUp() override { initializeGlobalFileMgr(); }

  AbstractBuffer* createBuffer(size_t num_entries_per_page) {
    const auto& chunk_key = TEST_CHUNK_KEY;
    constexpr size_t reserved_header_size{32};
    auto buffer = getFileMgr()->createBuffer(
        chunk_key, reserved_header_size + (num_entries_per_page * sizeof(int32_t)), 0);
    buffer->initEncoder(SQLTypeInfo{kINT});
    return buffer;
  }

  void setMaxRollbackEpochs(int32_t max_rollback_epochs) {
    File_Namespace::FileMgrParams params;
    params.max_rollback_epochs = max_rollback_epochs;
    global_file_mgr_->setFileMgrParams(
        TEST_CHUNK_KEY[CHUNK_KEY_DB_IDX], TEST_CHUNK_KEY[CHUNK_KEY_TABLE_IDX], params);
  }

  void updateData(std::vector<int32_t>& values) {
    const auto& chunk_key = TEST_CHUNK_KEY;
    auto buffer_size = values.size() * sizeof(int32_t);
    TestHelpers::TestBuffer buffer{std::vector<int32_t>{1}};
    buffer.reserve(buffer_size);
    memcpy(buffer.getMemoryPtr(), reinterpret_cast<int8_t*>(values.data()), buffer_size);
    buffer.setSize(buffer_size);
    buffer.setUpdated();
    getFileMgr()->putBuffer(chunk_key, &buffer, buffer_size);
  }
};

TEST_F(MaxRollbackEpochTest, WriteEmptyBufferAndSingleEpochVersion) {
  setMaxRollbackEpochs(0);
  auto file_mgr = getFileMgr();

  // Creates a buffer that allows for a maximum of 2 integers per page
  auto buffer = createBuffer(2);

  std::vector<int32_t> data{1, 2, 3, 4};
  writeData(buffer, data, 0);
  file_mgr->checkpoint();

  // 2 pages should be used for the above 4 integers
  auto stats = file_mgr->getStorageStats();
  auto used_page_count =
      stats.total_data_page_count - stats.total_free_data_page_count.value();
  ASSERT_EQ(static_cast<uint64_t>(2), used_page_count);

  data = {};
  updateData(data);
  file_mgr->checkpoint();

  // Last 2 pages should be rolled-off. No page should be in use, since
  // an empty buffer was written
  stats = file_mgr->getStorageStats();
  ASSERT_EQ(stats.total_data_page_count, stats.total_free_data_page_count.value());
}

TEST_F(MaxRollbackEpochTest, WriteEmptyBufferAndMultipleEpochVersions) {
  setMaxRollbackEpochs(1);
  auto file_mgr = getFileMgr();

  // Creates a buffer that allows for a maximum of 2 integers per page
  auto buffer = createBuffer(2);

  std::vector<int32_t> data{1, 2, 3, 4};
  writeData(buffer, data, 0);
  file_mgr->checkpoint();

  // 2 pages should be used for the above 4 integers
  auto stats = file_mgr->getStorageStats();
  auto used_page_count =
      stats.total_data_page_count - stats.total_free_data_page_count.value();
  ASSERT_EQ(static_cast<uint64_t>(2), used_page_count);

  data = {};
  updateData(data);
  file_mgr->checkpoint();

  // With a max_rollback_epochs of 1, the 2 previous pages should still
  // be in use for the above 4 integers
  stats = file_mgr->getStorageStats();
  used_page_count =
      stats.total_data_page_count - stats.total_free_data_page_count.value();
  ASSERT_EQ(static_cast<uint64_t>(2), used_page_count);
}

constexpr char file_mgr_path[] = "./FileMgrTestDir";

class FileMgrUnitTest : public testing::Test {
 protected:
  static constexpr size_t page_size_ = 64;
  void SetUp() override {
    bf::remove_all(file_mgr_path);
    bf::create_directory(file_mgr_path);
  }

  void TearDown() override { bf::remove_all(file_mgr_path); }

  std::unique_ptr<File_Namespace::GlobalFileMgr> initializeGFM(
      std::shared_ptr<ForeignStorageInterface> fsi,
      size_t num_pages = 1) {
    std::vector<int8_t> write_buffer{1, 2, 3, 4};
    auto gfm = std::make_unique<File_Namespace::GlobalFileMgr>(
        0, fsi, file_mgr_path, 0, page_size_);
    auto fm = dynamic_cast<File_Namespace::FileMgr*>(gfm->getFileMgr(1, 1));
    auto buffer = fm->createBuffer({1, 1, 1, 1});
    auto page_data_size = page_size_ - buffer->reservedHeaderSize();
    for (size_t i = 0; i < page_data_size * num_pages; i += 4) {
      buffer->append(write_buffer.data(), 4);
    }
    gfm->checkpoint(1, 1);
    return gfm;
  }
};

TEST_F(FileMgrUnitTest, InitializeWithUncheckpointedFreedFirstPage) {
  auto fsi = std::make_shared<ForeignStorageInterface>();
  {
    auto temp_gfm = initializeGFM(fsi, 2);
    auto buffer =
        dynamic_cast<File_Namespace::FileBuffer*>(temp_gfm->getBuffer({1, 1, 1, 1}));
    buffer->freePage(buffer->getMultiPage().front().current().page);
  }
  File_Namespace::GlobalFileMgr gfm(0, fsi, file_mgr_path, 0, page_size_);
  auto buffer = gfm.getBuffer({1, 1, 1, 1});
  ASSERT_EQ(buffer->pageCount(), 2U);
}

TEST_F(FileMgrUnitTest, InitializeWithUncheckpointedFreedLastPage) {
  auto fsi = std::make_shared<ForeignStorageInterface>();
  {
    auto temp_gfm = initializeGFM(fsi, 2);
    auto buffer =
        dynamic_cast<File_Namespace::FileBuffer*>(temp_gfm->getBuffer({1, 1, 1, 1}));
    buffer->freePage(buffer->getMultiPage().back().current().page);
  }
  File_Namespace::GlobalFileMgr gfm(0, fsi, file_mgr_path, 0, page_size_);
  auto buffer = gfm.getBuffer({1, 1, 1, 1});
  ASSERT_EQ(buffer->pageCount(), 2U);
}

TEST_F(FileMgrUnitTest, InitializeWithUncheckpointedAppendPages) {
  auto fsi = std::make_shared<ForeignStorageInterface>();
  std::vector<int8_t> write_buffer{1, 2, 3, 4};
  {
    auto temp_gfm = initializeGFM(fsi, 1);
    auto buffer =
        dynamic_cast<File_Namespace::FileBuffer*>(temp_gfm->getBuffer({1, 1, 1, 1}));
    buffer->append(write_buffer.data(), 4);
  }
  File_Namespace::GlobalFileMgr gfm(0, fsi, file_mgr_path, 0, page_size_);
  auto buffer = dynamic_cast<File_Namespace::FileBuffer*>(gfm.getBuffer({1, 1, 1, 1}));
  ASSERT_EQ(buffer->pageCount(), 1U);
}

class RebrandMigrationTest : public FileMgrUnitTest {
 protected:
  void setFileMgrVersion(int32_t version_number) {
    const auto table_data_dir = bf::path(file_mgr_path) / "table_1_1";
    const auto filename = table_data_dir / "filemgr_version";
    std::ofstream version_file{filename.string()};
    version_file.write(reinterpret_cast<char*>(&version_number), sizeof(int32_t));
    version_file.close();
  }

  int32_t getFileMgrVersion() {
    const auto table_data_dir = bf::path(file_mgr_path) / "table_1_1";
    const auto filename = table_data_dir / "filemgr_version";
    std::ifstream version_file{filename.string()};
    int32_t version_number;
    version_file.read(reinterpret_cast<char*>(&version_number), sizeof(int32_t));
    version_file.close();
    return version_number;
  }
};

TEST_F(RebrandMigrationTest, ExistingLegacyDataFiles) {
  auto global_file_mgr = initializeGFM(std::make_shared<ForeignStorageInterface>());
  constexpr int32_t db_id{1};
  constexpr int32_t table_id{1};
  global_file_mgr->closeFileMgr(db_id, table_id);
  const auto table_data_dir = bf::path(file_mgr_path) / "table_1_1";
  const auto legacy_data_file_path =
      table_data_dir / ("0." + std::to_string(page_size_) + ".mapd");
  const auto new_data_file_path =
      table_data_dir / ("0." + std::to_string(page_size_) + ".data");
  const auto legacy_metadata_file_path =
      table_data_dir / ("1." + std::to_string(DEFAULT_METADATA_PAGE_SIZE) + ".mapd");
  const auto new_metadata_file_path =
      table_data_dir / ("1." + std::to_string(DEFAULT_METADATA_PAGE_SIZE) + ".data");

  if (bf::exists(legacy_data_file_path)) {
    bf::remove(legacy_data_file_path);
  }

  if (bf::exists(legacy_metadata_file_path)) {
    bf::remove(legacy_metadata_file_path);
  }

  ASSERT_TRUE(bf::exists(new_data_file_path));
  bf::rename(new_data_file_path, legacy_data_file_path);

  ASSERT_TRUE(bf::exists(new_metadata_file_path));
  bf::rename(new_metadata_file_path, legacy_metadata_file_path);

  setFileMgrVersion(1);
  ASSERT_EQ(getFileMgrVersion(), 1);
  // Migration should occur when the file manager is initialized.
  global_file_mgr->getFileMgr(db_id, table_id);
  ASSERT_EQ(getFileMgrVersion(), 2);

  ASSERT_TRUE(bf::exists(new_data_file_path));
  ASSERT_TRUE(bf::is_regular_file(new_data_file_path));

  ASSERT_TRUE(bf::exists(new_metadata_file_path));
  ASSERT_TRUE(bf::is_regular_file(new_metadata_file_path));

  bf::canonical(legacy_data_file_path);
  ASSERT_TRUE(bf::exists(legacy_data_file_path));
  ASSERT_TRUE(bf::is_symlink(legacy_data_file_path));

  ASSERT_TRUE(bf::exists(legacy_metadata_file_path));
  ASSERT_TRUE(bf::is_symlink(legacy_metadata_file_path));
}

TEST_F(RebrandMigrationTest, NewDataFiles) {
  initializeGFM(std::make_shared<ForeignStorageInterface>(), 1);

  const auto table_data_dir = bf::path(file_mgr_path) / "table_1_1";
  const auto legacy_data_file_path =
      table_data_dir / ("0." + std::to_string(page_size_) + ".mapd");
  const auto new_data_file_path =
      table_data_dir / ("0." + std::to_string(page_size_) + ".data");
  const auto legacy_metadata_file_path =
      table_data_dir / ("1." + std::to_string(DEFAULT_METADATA_PAGE_SIZE) + ".mapd");
  const auto new_metadata_file_path =
      table_data_dir / ("1." + std::to_string(DEFAULT_METADATA_PAGE_SIZE) + ".data");

  ASSERT_TRUE(bf::exists(new_data_file_path));
  ASSERT_TRUE(bf::is_regular_file(new_data_file_path));

  ASSERT_TRUE(bf::exists(new_metadata_file_path));
  ASSERT_TRUE(bf::is_regular_file(new_metadata_file_path));

  ASSERT_TRUE(bf::exists(legacy_data_file_path));
  ASSERT_TRUE(bf::is_symlink(legacy_data_file_path));

  ASSERT_TRUE(bf::exists(legacy_metadata_file_path));
  ASSERT_TRUE(bf::is_symlink(legacy_metadata_file_path));
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
