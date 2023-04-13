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

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>  // TODO(Misiu): Update FileMgr API to remove this.
#include <filesystem>
#include <fstream>
#include "DataMgr/FileMgr/FileMgr.h"
#include "DataMgr/FileMgr/GlobalFileMgr.h"
#include "DataMgr/ForeignStorage/ArrowForeignStorage.h"
#include "DataMgrTestHelpers.h"
#include "Shared/File.h"
#include "TestHelpers.h"

extern bool g_read_only;

namespace fs = std::filesystem;
namespace fn = File_Namespace;
namespace bf = boost::filesystem;

constexpr const char* kReadOnlyWriteError{"Error trying to write file"};
constexpr const char* kReadOnlyCreateError{"Error trying to create file"};
constexpr const char* kFileMgrPath{"./FileMgrTestDir"};
constexpr const char* kTestDataDir{"./test_dir"};
constexpr const char* kDataDir{"./test_dir/mapd_data"};
constexpr const char* kTempFile{"./test_dir/mapd_data/temp.txt"};

namespace {
struct ExpectedException : public std::runtime_error {
  ExpectedException(const std::string& msg) : std::runtime_error(msg) {}
};

// Wrapper that executes a given function with the expectation that it throws an exception
// containing specific text.
template <typename Func>
void run_and_catch(Func func, const std::string& exception_text = kReadOnlyWriteError) {
  try {
    func();
    throw ExpectedException("expected exception with text: '" + exception_text + "',");
  } catch (const ExpectedException& e) {
    // Need special handling for the exception this function throws if there are no
    // uncaught exceptions in func because we want to wrap the expected exception text in
    // the results but don't want it caught in the subsequent catch-block (hence a custom
    // exception class).
    throw;
  } catch (const std::exception& e) {
    std::string err_msg = e.what();
    if (err_msg.find(exception_text) == std::string::npos) {
      // If the caught exception does not match the intended exception text, then rethrow.
      throw;
    }
  }
}

// Execute some function while temporarily disableing read-only mode (if it was enabled).
template <typename Func>
void run_in_write_mode(Func func) {
  bool old_state = g_read_only;
  g_read_only = false;
  func();
  g_read_only = old_state;
}

void compare_buffers(AbstractBuffer* left_buffer,
                     AbstractBuffer* right_buffer,
                     size_t num_bytes) {
  std::vector<int8_t> left_array(num_bytes);
  std::vector<int8_t> right_array(num_bytes);
  left_buffer->read(left_array.data(), num_bytes);
  right_buffer->read(right_array.data(), num_bytes);
  ASSERT_EQ(left_array, right_array);
  ASSERT_EQ(left_buffer->hasEncoder(), right_buffer->hasEncoder());
}

void compare_metadata(const std::shared_ptr<ChunkMetadata> lhs_metadata,
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

std::shared_ptr<ChunkMetadata> get_metadata_for_buffer(AbstractBuffer* buffer) {
  const std::shared_ptr<ChunkMetadata> metadata = std::make_shared<ChunkMetadata>();
  buffer->getEncoder()->getMetadata(metadata);
  return metadata;
}

void compare_buffers_and_metadata(AbstractBuffer* left_buffer,
                                  AbstractBuffer* right_buffer) {
  ASSERT_TRUE(left_buffer->hasEncoder());
  ASSERT_TRUE(right_buffer->hasEncoder());
  ASSERT_TRUE(left_buffer->getEncoder());
  ASSERT_TRUE(right_buffer->getEncoder());
  ASSERT_EQ(left_buffer->size(), get_metadata_for_buffer(left_buffer)->numBytes);
  ASSERT_EQ(right_buffer->size(), get_metadata_for_buffer(right_buffer)->numBytes);
  compare_metadata(get_metadata_for_buffer(left_buffer),
                   get_metadata_for_buffer(right_buffer));
  compare_buffers(left_buffer, right_buffer, left_buffer->size());
}

int8_t* get_data_ptr(std::vector<int32_t>& data_vector) {
  return reinterpret_cast<int8_t*>(data_vector.data());
}

void write_data(AbstractBuffer* data_buffer,
                std::vector<int32_t>& write_data,
                const size_t offset) {
  CHECK(data_buffer->hasEncoder());
  auto sql_type_info = get_metadata_for_buffer(data_buffer)->sqlType;
  auto write_ptr = get_data_ptr(write_data);
  // appendData is a misnomer, with the offset we are overwriting part of the buffer
  data_buffer->getEncoder()->appendData(
      write_ptr, write_data.size(), sql_type_info, false /*replicating*/, offset);
}

void append_data(AbstractBuffer* data_buffer, std::vector<int32_t>& append_data) {
  write_data(data_buffer, append_data, -1);
}

}  // namespace

class FileInfoTest : public testing::Test {
 public:
  constexpr static const char* data_file_name = "./test_dir/0.64.data";
  constexpr static const char* meta_file_name = "./test_dir/1.128.data";
  constexpr static const int32_t db = 1, tb = 1, data_file_id = 0, meta_file_id = 1;
  constexpr static const size_t num_pages = 16, page_size = 64, meta_page_size = 128;

 protected:
  void SetUp() override {
    fs::remove_all(kTestDataDir);
    fs::create_directory(kTestDataDir);

    // Currently FileInfo has a dependency on having a parent FileMgr, so we generate them
    // here.  Other than openExistingFile() the parent FileMgr state will not affect the
    // FileInfo's method calls.  Future work is underway to remove this dependency
    // entirely (a FileInfo should not need access to a parent FileMgr).
    fsi_ = std::make_shared<ForeignStorageInterface>();
    gfm_ = std::make_unique<fn::GlobalFileMgr>(
        0, fsi_, kTestDataDir, 0, page_size, meta_page_size);
    fm_ptr_ = dynamic_cast<fn::FileMgr*>(gfm_->getFileMgr(db, tb));

    auto [fd, file_path] = fn::create(kTestDataDir, data_file_id, page_size, num_pages);
    file_info_ = std::make_unique<fn::FileInfo>(
        fm_ptr_, data_file_id, fd, page_size, num_pages, file_path);
  }

  void TearDown() override {
    file_info_ = nullptr;
    fs::remove_all(kTestDataDir);
  }

  static void SetUpTestSuite() {
    fn::FileMgr::setNumPagesPerDataFile(num_pages);
    fn::FileMgr::setNumPagesPerMetadataFile(num_pages);
  }

  static void TearDownTestSute() {
    fn::FileMgr::setNumPagesPerDataFile(fn::FileMgr::DEFAULT_NUM_PAGES_PER_DATA_FILE);
    fn::FileMgr::setNumPagesPerMetadataFile(
        fn::FileMgr::DEFAULT_NUM_PAGES_PER_METADATA_FILE);
  }

  template <class T>
  std::vector<T> readFromFile(const char* file, size_t offset, size_t num_elems) {
    CHECK(!file_info_) << "File desc must be closed before we read file directly.";
    auto fd = heavyai::fopen(file, "r");
    std::vector<T> buf(num_elems);
    fn::read(
        fd, offset, num_elems * sizeof(T), reinterpret_cast<int8_t*>(buf.data()), file);
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
  std::unique_ptr<fn::GlobalFileMgr> gfm_;
  fn::FileMgr* fm_ptr_;
  std::unique_ptr<fn::FileInfo> file_info_;
};

TEST_F(FileInfoTest, initNewFile) {
  file_info_->initNewFile();
  EXPECT_EQ(file_info_->numFreePages(), num_pages);
  file_info_ = nullptr;  // close file descriptor;

  ASSERT_TRUE(fs::exists(data_file_name));
  ASSERT_EQ(fs::file_size(data_file_name), num_pages * page_size);

  auto fd = heavyai::fopen(data_file_name, "r");
  int32_t header_size = 0;
  int8_t* buf = reinterpret_cast<int8_t*>(&header_size);
  for (size_t i = 0; i < page_size * num_pages; i += page_size) {
    fn::read(fd, i, sizeof(int32_t), buf, data_file_name);
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
    fs::remove_all(kTestDataDir);
    fs::create_directory(kTestDataDir);

    // Tests need a FileMgr to access epoch data.
    fsi_ = std::make_shared<ForeignStorageInterface>();
    gfm_ = std::make_unique<fn::GlobalFileMgr>(
        0, fsi_, kTestDataDir, 0, page_size, meta_page_size);

    // The last checkpointed epoch for the pre-created files is actually "2", but the
    // FileMgr will automatically increment the epoch during initialization so we need to
    // override the epoch to "1" so that we get "2" once it's done.
    // This is necessary because openExistingFile() is expected to be called as part of FM
    // initialization before we increment the epoch, so if we were callilng it normally,
    // the epoch would be read as "2", then we call openExistingFile(), then we increment.
    // But here we are pre-initializing a FM and calling the function after the epoch is
    // incremented (so as not to depend on any of the initialiation code).
    fm_ = std::make_unique<fn::FileMgr>(
        0, gfm_.get(), fn::TablePair{1, 1}, -1, 0, 1 /* epoch */);
  }

  // These methods were used to create the data files used for comparison purposes.
  void createTestingFiles(const std::string& gfm_path) const {
    CHECK_NE(gfm_path, kTestDataDir)
        << "Can't create new test files in a directory that will be used.";
    fs::remove_all(gfm_path);

    // Need to setup a temporary FileMgr to create files.
    auto fsi = std::make_shared<ForeignStorageInterface>();
    auto gfm = std::make_unique<fn::GlobalFileMgr>(
        0, fsi, gfm_path, 0, page_size, meta_page_size);
    auto fm = dynamic_cast<fn::FileMgr*>(gfm->getFileMgr(1, 1));

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
  std::unique_ptr<fn::GlobalFileMgr> gfm_;
  std::unique_ptr<fn::FileMgr> fm_;
};

TEST_F(OpenExistingFileTest, Data) {
  ASSERT_EQ(fm_->epoch(1, 1), 2) << "FM was not initialized correctly.";

  // Fetch source file.
  fs::copy(source_data_file, data_file_name);

  auto fd = heavyai::fopen(data_file_name, "r+w");
  fn::FileInfo file_info(
      fm_.get(), data_file_id, fd, page_size, num_pages, data_file_name);

  std::vector<fn::HeaderInfo> headers;
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
  fs::copy(source_meta_file, meta_file_name);

  auto fd = heavyai::fopen(meta_file_name, "r+w");
  fn::FileInfo file_info(
      fm_.get(), meta_file_id, fd, meta_page_size, num_pages, meta_file_name);

  std::vector<fn::HeaderInfo> headers;
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

class AbstractFileMgrTest : public testing::Test {
 public:
  static inline const std::string table_dir{std::string{kDataDir} + "/table_1_1"};
  static inline const std::string version_file_name{
      table_dir + "/" + fn::FileMgr::FILE_MGR_VERSION_FILENAME};
  static constexpr const int32_t db_id = 1, tb_id = 1, empty_tb_id = 2, data_file_id = 0,
                                 meta_file_id = 1;
  static constexpr const size_t num_pages = 16, page_size = 64, meta_page_size = 128;
  static inline const ChunkKey default_key{db_id, tb_id, 1, 0};
  static inline const ChunkKey empty_key{db_id, empty_tb_id, 1, 0};
  static inline const fn::TablePair table_pair{db_id, tb_id};

  void SetUp() override {
    fs::remove_all(kTestDataDir);
    fs::create_directory(kTestDataDir);
    fs::create_directory(kDataDir);
  }

  void TearDown() override { fs::remove_all(kTestDataDir); }
};

class FileMgrTest : public AbstractFileMgrTest {
 protected:
  void SetUp() override {
    initializeGlobalFileMgr();
    initializeChunk(1);
  }

  void initializeGlobalFileMgr() {
    fs::remove_all(kTestDataDir);
    global_file_mgr_ = std::make_unique<fn::GlobalFileMgr>(
        0, std::make_shared<ForeignStorageInterface>(), kTestDataDir, 0);
  }

  fn::FileMgr* getFileMgr() {
    auto file_mgr = global_file_mgr_->getFileMgr(db_id, tb_id);
    return dynamic_cast<fn::FileMgr*>(file_mgr);
  }

  void initializeChunk(int32_t value) {
    auto file_mgr = getFileMgr();
    auto buffer = file_mgr->createBuffer(default_key);
    buffer->initEncoder(SQLTypeInfo{kINT});
    std::vector<int32_t> data{value};
    write_data(buffer, data, 0);
    file_mgr->checkpoint();
  }

  void setMaxRollbackEpochs(const int32_t max_rollback_epochs) {
    fn::FileMgrParams file_mgr_params;
    file_mgr_params.max_rollback_epochs = max_rollback_epochs;
    global_file_mgr_->setFileMgrParams(db_id, tb_id, file_mgr_params);
  }

  std::unique_ptr<fn::GlobalFileMgr> global_file_mgr_;
};

TEST_F(FileMgrTest, putBuffer_update) {
  TestHelpers::TestBuffer source_buffer{std::vector<int32_t>{1}};
  source_buffer.setUpdated();
  auto file_mgr = getFileMgr();
  AbstractBuffer* file_buffer = file_mgr->putBuffer(default_key, &source_buffer, 4);
  compare_buffers_and_metadata(&source_buffer, file_buffer);
  ASSERT_FALSE(source_buffer.isAppended());
  ASSERT_FALSE(source_buffer.isUpdated());
  ASSERT_FALSE(source_buffer.isDirty());
}

TEST_F(FileMgrTest, putBuffer_subwrite) {
  TestHelpers::TestBuffer source_buffer{SQLTypeInfo{kINT}};
  int8_t temp_array[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  source_buffer.write(temp_array, 8);
  auto file_mgr = getFileMgr();
  AbstractBuffer* file_buffer = file_mgr->putBuffer(default_key, &source_buffer, 4);
  compare_buffers(&source_buffer, file_buffer, 4);
}

TEST_F(FileMgrTest, putBuffer_exists) {
  TestHelpers::TestBuffer source_buffer{SQLTypeInfo{kINT}};
  int8_t temp_array[4] = {1, 2, 3, 4};
  source_buffer.write(temp_array, 4);
  auto file_mgr = getFileMgr();
  file_mgr->putBuffer(default_key, &source_buffer, 4);
  file_mgr->checkpoint();
  source_buffer.write(temp_array, 4);
  AbstractBuffer* file_buffer = file_mgr->putBuffer(default_key, &source_buffer, 4);
  compare_buffers_and_metadata(&source_buffer, file_buffer);
}

TEST_F(FileMgrTest, putBuffer_append) {
  TestHelpers::TestBuffer source_buffer{std::vector<int32_t>{1}};
  int8_t temp_array[4] = {1, 2, 3, 4};
  source_buffer.append(temp_array, 4);
  auto file_mgr = getFileMgr();
  AbstractBuffer* file_buffer = file_mgr->putBuffer(default_key, &source_buffer, 8);
  compare_buffers_and_metadata(&source_buffer, file_buffer);
}

TEST_F(FileMgrTest, put_checkpoint_get) {
  TestHelpers::TestBuffer source_buffer{std::vector<int32_t>{1}};
  std::vector<int32_t> data_v1 = {1, 2, 3, 5, 7};
  append_data(&source_buffer, data_v1);
  auto file_mgr = getFileMgr();
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  AbstractBuffer* file_buffer_put = file_mgr->putBuffer(default_key, &source_buffer, 24);
  ASSERT_TRUE(file_buffer_put->isDirty());
  ASSERT_FALSE(file_buffer_put->isUpdated());
  ASSERT_TRUE(file_buffer_put->isAppended());
  ASSERT_EQ(file_buffer_put->size(), static_cast<size_t>(24));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  file_mgr->checkpoint();
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  AbstractBuffer* file_buffer_get = file_mgr->getBuffer(default_key, 24);
  ASSERT_EQ(file_buffer_put, file_buffer_get);
  CHECK(!(file_buffer_get->isDirty()));
  CHECK(!(file_buffer_get->isUpdated()));
  CHECK(!(file_buffer_get->isAppended()));
  ASSERT_EQ(file_buffer_get->size(), static_cast<size_t>(24));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  compare_buffers_and_metadata(&source_buffer, file_buffer_get);
}

TEST_F(FileMgrTest, put_checkpoint_get_double_write) {
  TestHelpers::TestBuffer source_buffer{std::vector<int32_t>{1}};
  std::vector<int32_t> data_v1 = {1, 2, 3, 5, 7};
  std::vector<int32_t> data_v2 = {11, 13, 17, 19};
  append_data(&source_buffer, data_v1);
  auto file_mgr = getFileMgr();
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  file_mgr->putBuffer(default_key, &source_buffer, 24);
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  file_mgr->checkpoint();
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  AbstractBuffer* file_buffer = file_mgr->getBuffer(default_key, 24);
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  ASSERT_FALSE(file_buffer->isDirty());
  ASSERT_EQ(file_buffer->size(), static_cast<size_t>(24));
  compare_buffers_and_metadata(&source_buffer, file_buffer);
  append_data(file_buffer, data_v2);
  ASSERT_TRUE(file_buffer->isDirty());
  ASSERT_EQ(file_buffer->size(), static_cast<size_t>(40));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  append_data(file_buffer, data_v2);
  CHECK(file_buffer->isDirty());
  ASSERT_EQ(file_buffer->size(), static_cast<size_t>(56));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  file_mgr->checkpoint();
  CHECK(!(file_buffer->isDirty()));
  ASSERT_EQ(file_buffer->size(), static_cast<size_t>(56));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 3);
  append_data(&source_buffer, data_v2);
  append_data(&source_buffer, data_v2);
  compare_buffers_and_metadata(&source_buffer, file_buffer);
}

TEST_F(FileMgrTest, buffer_append_and_recovery) {
  TestHelpers::TestBuffer source_buffer{SQLTypeInfo{kINT}};
  std::vector<int32_t> initial_value{1};
  append_data(&source_buffer, initial_value);
  ASSERT_EQ(get_metadata_for_buffer(&source_buffer)->numElements, static_cast<size_t>(1));

  std::vector<int32_t> data_v1 = {1, 2, 3, 5, 7};
  std::vector<int32_t> data_v2 = {11, 13, 17, 19};
  append_data(&source_buffer, data_v1);
  {
    auto file_mgr = getFileMgr();
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
    AbstractBuffer* file_buffer = file_mgr->putBuffer(default_key, &source_buffer, 24);
    file_mgr->checkpoint();
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
    ASSERT_EQ(get_metadata_for_buffer(&source_buffer)->numElements,
              static_cast<size_t>(6));
    ASSERT_EQ(get_metadata_for_buffer(file_buffer)->numElements, static_cast<size_t>(6));
    SCOPED_TRACE("Buffer Append and Recovery - Compare #1");
    compare_buffers_and_metadata(&source_buffer, file_buffer);

    // Now write data we will not checkpoint
    append_data(file_buffer, data_v1);
    ASSERT_EQ(file_buffer->size(), static_cast<size_t>(44));
    // Now close filemgr to test recovery
    global_file_mgr_->closeFileMgr(db_id, tb_id);
  }

  {
    auto file_mgr = getFileMgr();
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
    ChunkMetadataVector chunkMetadataVector;
    file_mgr->getChunkMetadataVecForKeyPrefix(chunkMetadataVector, default_key);
    ASSERT_EQ(chunkMetadataVector.size(), static_cast<size_t>(1));
    ASSERT_EQ(std::memcmp(chunkMetadataVector[0].first.data(), default_key.data(), 16),
              0);
    ASSERT_EQ(chunkMetadataVector[0].first, default_key);
    std::shared_ptr<ChunkMetadata> chunk_metadata = chunkMetadataVector[0].second;
    ASSERT_EQ(chunk_metadata->numBytes, static_cast<size_t>(24));
    ASSERT_EQ(chunk_metadata->numElements, static_cast<size_t>(6));
    AbstractBuffer* file_buffer =
        file_mgr->getBuffer(default_key, chunk_metadata->numBytes);
    {
      SCOPED_TRACE("Buffer Append and Recovery - Compare #2");
      compare_buffers_and_metadata(&source_buffer, file_buffer);
    }
    append_data(&source_buffer, data_v2);
    append_data(file_buffer, data_v2);

    file_mgr->checkpoint();
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 3);
    {
      SCOPED_TRACE("Buffer Append and Recovery - Compare #3");
      compare_buffers_and_metadata(&source_buffer, file_buffer);
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
    EXPECT_EQ(global_file_mgr_->getTableEpoch(db_id, tb_id), std::size_t(1));
    TestHelpers::TestBuffer source_buffer{SQLTypeInfo{kINT}};
    std::vector<int32_t> initial_value{1};
    append_data(&source_buffer, initial_value);
    source_buffer.clearDirtyBits();

    auto file_mgr = getFileMgr();
    AbstractBuffer* file_buffer = file_mgr->getBuffer(default_key);
    ASSERT_FALSE(source_buffer.isDirty());
    ASSERT_FALSE(source_buffer.isUpdated());
    ASSERT_FALSE(source_buffer.isAppended());
    ASSERT_FALSE(file_buffer->isDirty());
    ASSERT_FALSE(file_buffer->isUpdated());
    ASSERT_FALSE(file_buffer->isAppended());
    ASSERT_EQ(file_buffer->size(), static_cast<size_t>(4));
    {
      SCOPED_TRACE("Buffer Update and Recovery - Compare #1");
      compare_buffers_and_metadata(file_buffer, &source_buffer);
    }
    write_data(file_buffer, data_v1, 0);
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
          get_metadata_for_buffer(file_buffer);
      ASSERT_EQ(file_chunk_metadata->numElements, static_cast<size_t>(5));
      ASSERT_EQ(file_chunk_metadata->numBytes, static_cast<size_t>(20));
      ASSERT_EQ(file_chunk_metadata->chunkStats.min.intval, 2);
      ASSERT_EQ(file_chunk_metadata->chunkStats.max.intval, 11);
      ASSERT_EQ(file_chunk_metadata->chunkStats.has_nulls, false);
    }
    file_mgr->checkpoint();
    EXPECT_EQ(global_file_mgr_->getTableEpoch(db_id, tb_id), std::size_t(2));
    ASSERT_FALSE(file_buffer->isDirty());
    ASSERT_FALSE(file_buffer->isUpdated());
    ASSERT_FALSE(file_buffer->isAppended());

    source_buffer.reset();
    file_mgr->fetchBuffer(
        default_key,
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
          get_metadata_for_buffer(&source_buffer);
      ASSERT_EQ(cpu_chunk_metadata->numElements, static_cast<size_t>(5));
      ASSERT_EQ(cpu_chunk_metadata->numBytes, static_cast<size_t>(20));
      ASSERT_EQ(cpu_chunk_metadata->chunkStats.min.intval, 2);
      ASSERT_EQ(cpu_chunk_metadata->chunkStats.max.intval, 11);
      ASSERT_EQ(cpu_chunk_metadata->chunkStats.has_nulls, false);
    }
    {
      SCOPED_TRACE("Buffer Update and Recovery - Compare #2");
      compare_buffers_and_metadata(file_buffer, &source_buffer);
    }
    // Now roll back to epoch 1
    fn::FileMgrParams file_mgr_params;
    file_mgr_params.epoch = 1;
    global_file_mgr_->setFileMgrParams(db_id, tb_id, file_mgr_params);
    file_mgr = getFileMgr();
    file_buffer = file_mgr->getBuffer(default_key);
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
          get_metadata_for_buffer(file_buffer);
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
    const auto& capped_chunk_key = default_key;
    // Have one element already written to key -- epoch should be 2
    ASSERT_EQ(global_file_mgr_->getTableEpoch(capped_chunk_key[0], capped_chunk_key[1]),
              static_cast<size_t>(1));
    fn::FileMgr* file_mgr = dynamic_cast<fn::FileMgr*>(
        global_file_mgr_->getFileMgr(capped_chunk_key[0], capped_chunk_key[1]));
    // buffer inside loop
    for (int data_write = 1; data_write <= num_data_writes; ++data_write) {
      std::vector<int32_t> data;
      data.emplace_back(data_write);
      AbstractBuffer* file_buffer = global_file_mgr_->getBuffer(capped_chunk_key);
      append_data(file_buffer, data);
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

TEST_F(FileMgrTest, InitDoubleFileId) {
  global_file_mgr_->closeFileMgr(1, 1);
  // Illegal setup.  Data file and Metadata file have same fileId.
  std::string existing_file = std::string{kTestDataDir} + "/table_1_1/1.4096.data";
  std::string new_file = std::string{kTestDataDir} + "/table_1_1/0.4096.data";
  fs::copy(existing_file, new_file);
  ASSERT_DEATH(global_file_mgr_->getFileMgr(1, 1), "Attempting to re-open file");
}

class DataCompactionTest : public FileMgrTest {
 protected:
  void SetUp() override {
    TearDown();
    initializeGlobalFileMgr();
  }

  void TearDown() override {
    fn::FileMgr::setNumPagesPerDataFile(fn::FileMgr::DEFAULT_NUM_PAGES_PER_DATA_FILE);
    fn::FileMgr::setNumPagesPerMetadataFile(
        fn::FileMgr::DEFAULT_NUM_PAGES_PER_METADATA_FILE);
    FileMgrTest::TearDown();
  }

  ChunkKey getChunkKey(int32_t column_id) {
    auto chunk_key = default_key;
    chunk_key[CHUNK_KEY_COLUMN_IDX] = column_id;
    return chunk_key;
  }

  void assertStorageStats(uint64_t metadata_file_count,
                          std::optional<uint64_t> free_metadata_page_count,
                          uint64_t data_file_count,
                          std::optional<uint64_t> free_data_page_count) {
    auto stats = global_file_mgr_->getStorageStats(db_id, tb_id);
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
    auto metadata = get_metadata_for_buffer(buffer);
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
    write_data(buffer, data, 0);
    getFileMgr()->checkpoint();
  }

  void writeMultipleValues(AbstractBuffer* buffer, int32_t start_value, int32_t count) {
    for (int32_t i = 0; i < count; i++) {
      writeValue(buffer, start_value + i);
    }
  }

  void compactDataFiles() { global_file_mgr_->compactDataFiles(db_id, tb_id); }

  void deleteFileMgr() { global_file_mgr_->closeFileMgr(db_id, tb_id); }

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
  fn::FileMgr::setNumPagesPerDataFile(1);

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
  fn::FileMgr::setNumPagesPerMetadataFile(1);

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
  fn::FileMgr::setNumPagesPerDataFile(1);
  fn::FileMgr::setNumPagesPerMetadataFile(1);

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
  fn::FileMgr::setNumPagesPerDataFile(4);

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
  fn::FileMgr::setNumPagesPerDataFile(4);

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
  fn::FileMgr::setNumPagesPerDataFile(1);

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
  auto status_file_path = getFileMgr()->getFilePath(fn::FileMgr::COPY_PAGES_STATUS);
  deleteFileMgr();
  std::ofstream status_file{status_file_path.string(), std::ios::out | std::ios::binary};
  status_file.close();

  getFileMgr();
  assertStorageStats(1, 4095, 1, 0);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, 1);
}

TEST_F(DataCompactionTest, RecoveryFromUpdatePageVisibiltyStatus) {
  fn::FileMgr::setNumPagesPerDataFile(4);

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
  auto status_file_path = getFileMgr()->getFilePath(fn::FileMgr::COPY_PAGES_STATUS);
  std::ofstream status_file{status_file_path.string(), std::ios::out | std::ios::binary};
  status_file.close();

  auto file_mgr = getFileMgr();
  auto buffer = std::make_unique<int8_t[]>(DEFAULT_PAGE_SIZE);

  // Copy last page for "i2" chunk in last data file to first data file
  int32_t source_file_id{2}, dest_file_id{0};
  auto source_file_info = file_mgr->getFileInfoForFileId(source_file_id);
  source_file_info->read(0, DEFAULT_PAGE_SIZE, buffer.get());

  size_t offset{sizeof(fn::PageHeaderSizeType)};
  auto destination_file_info = file_mgr->getFileInfoForFileId(dest_file_id);
  destination_file_info->write(offset, DEFAULT_PAGE_SIZE - offset, buffer.get() + offset);
  destination_file_info->syncToDisk();

  fn::PageHeaderSizeType int_chunk_header_size{24};
  std::vector<fn::PageMapping> page_mappings{
      {source_file_id, 0, int_chunk_header_size, dest_file_id, 0}};
  file_mgr->writePageMappingsToStatusFile(page_mappings);
  file_mgr->renameCompactionStatusFile(fn::FileMgr::COPY_PAGES_STATUS,
                                       fn::FileMgr::UPDATE_PAGE_VISIBILITY_STATUS);
  deleteFileMgr();

  getFileMgr();
  assertStorageStats(1, 4094, 1, 2);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(2, 1);
  assertBufferValueAndMetadata(3, 2);
}

TEST_F(DataCompactionTest, RecoveryFromDeleteEmptyFileStatus) {
  fn::FileMgr::setNumPagesPerDataFile(4);

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
      getFileMgr()->getFilePath(fn::FileMgr::DELETE_EMPTY_FILES_STATUS);
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
    const auto& chunk_key = default_key;
    constexpr size_t reserved_header_size{32};
    auto buffer = getFileMgr()->createBuffer(
        chunk_key, reserved_header_size + (num_entries_per_page * sizeof(int32_t)), 0);
    buffer->initEncoder(SQLTypeInfo{kINT});
    return buffer;
  }

  void setMaxRollbackEpochs(int32_t max_rollback_epochs) {
    fn::FileMgrParams params;
    params.max_rollback_epochs = max_rollback_epochs;
    global_file_mgr_->setFileMgrParams(db_id, tb_id, params);
  }

  void updateData(std::vector<int32_t>& values) {
    const auto& chunk_key = default_key;
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
  write_data(buffer, data, 0);
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
  write_data(buffer, data, 0);
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

class FileMgrUnitTest : public testing::Test {
 protected:
  static constexpr size_t page_size_ = 64;

  void SetUp() override {
    fs::remove_all(kFileMgrPath);
    fs::create_directory(kFileMgrPath);
  }

  void TearDown() override { fs::remove_all(kFileMgrPath); }

  std::unique_ptr<fn::GlobalFileMgr> initializeGFM(
      std::shared_ptr<ForeignStorageInterface> fsi,
      size_t num_pages = 1) {
    std::vector<int8_t> write_buffer{1, 2, 3, 4};
    auto gfm = std::make_unique<fn::GlobalFileMgr>(0, fsi, kFileMgrPath, 0, page_size_);
    auto fm = dynamic_cast<fn::FileMgr*>(gfm->getFileMgr(1, 1));
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
    auto buffer = dynamic_cast<fn::FileBuffer*>(temp_gfm->getBuffer({1, 1, 1, 1}));
    buffer->freePage(buffer->getMultiPage().front().current().page);
  }
  fn::GlobalFileMgr gfm(0, fsi, kFileMgrPath, 0, page_size_);
  auto buffer = gfm.getBuffer({1, 1, 1, 1});
  ASSERT_EQ(buffer->pageCount(), 2U);
}

TEST_F(FileMgrUnitTest, InitializeWithUncheckpointedFreedLastPage) {
  auto fsi = std::make_shared<ForeignStorageInterface>();
  {
    auto temp_gfm = initializeGFM(fsi, 2);
    auto buffer = dynamic_cast<fn::FileBuffer*>(temp_gfm->getBuffer({1, 1, 1, 1}));
    buffer->freePage(buffer->getMultiPage().back().current().page);
  }
  fn::GlobalFileMgr gfm(0, fsi, kFileMgrPath, 0, page_size_);
  auto buffer = gfm.getBuffer({1, 1, 1, 1});
  ASSERT_EQ(buffer->pageCount(), 2U);
}

TEST_F(FileMgrUnitTest, InitializeWithUncheckpointedAppendPages) {
  auto fsi = std::make_shared<ForeignStorageInterface>();
  std::vector<int8_t> write_buffer{1, 2, 3, 4};
  {
    auto temp_gfm = initializeGFM(fsi, 1);
    auto buffer = dynamic_cast<fn::FileBuffer*>(temp_gfm->getBuffer({1, 1, 1, 1}));
    buffer->append(write_buffer.data(), 4);
  }
  fn::GlobalFileMgr gfm(0, fsi, kFileMgrPath, 0, page_size_);
  auto buffer = dynamic_cast<fn::FileBuffer*>(gfm.getBuffer({1, 1, 1, 1}));
  ASSERT_EQ(buffer->pageCount(), 1U);
}

class RebrandMigrationTest : public FileMgrUnitTest {
 protected:
  void setFileMgrVersion(int32_t version_number) {
    const auto table_data_dir = fs::path(kFileMgrPath) / "table_1_1";
    const auto filename = table_data_dir / "filemgr_version";
    std::ofstream version_file{filename.string()};
    version_file.write(reinterpret_cast<char*>(&version_number), sizeof(int32_t));
    version_file.close();
  }

  int32_t getFileMgrVersion() {
    const auto table_data_dir = fs::path(kFileMgrPath) / "table_1_1";
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
  const auto table_data_dir = fs::path(kFileMgrPath) / "table_1_1";
  const auto legacy_data_file_path =
      table_data_dir / ("0." + std::to_string(page_size_) + ".mapd");
  const auto new_data_file_path =
      table_data_dir / ("0." + std::to_string(page_size_) + ".data");
  const auto legacy_metadata_file_path =
      table_data_dir / ("1." + std::to_string(DEFAULT_METADATA_PAGE_SIZE) + ".mapd");
  const auto new_metadata_file_path =
      table_data_dir / ("1." + std::to_string(DEFAULT_METADATA_PAGE_SIZE) + ".data");

  if (fs::exists(legacy_data_file_path)) {
    fs::remove(legacy_data_file_path);
  }

  if (fs::exists(legacy_metadata_file_path)) {
    fs::remove(legacy_metadata_file_path);
  }

  ASSERT_TRUE(fs::exists(new_data_file_path));
  fs::rename(new_data_file_path, legacy_data_file_path);

  ASSERT_TRUE(fs::exists(new_metadata_file_path));
  fs::rename(new_metadata_file_path, legacy_metadata_file_path);

  setFileMgrVersion(1);
  ASSERT_EQ(getFileMgrVersion(), 1);
  // Migration should occur when the file manager is initialized.
  global_file_mgr->getFileMgr(db_id, table_id);
  ASSERT_EQ(getFileMgrVersion(), 2);

  ASSERT_TRUE(fs::exists(new_data_file_path));
  ASSERT_TRUE(fs::is_regular_file(new_data_file_path));

  ASSERT_TRUE(fs::exists(new_metadata_file_path));
  ASSERT_TRUE(fs::is_regular_file(new_metadata_file_path));

  ASSERT_TRUE(fs::exists(legacy_data_file_path));
  ASSERT_TRUE(fs::is_symlink(legacy_data_file_path));

  ASSERT_TRUE(fs::exists(legacy_metadata_file_path));
  ASSERT_TRUE(fs::is_symlink(legacy_metadata_file_path));
}

TEST_F(RebrandMigrationTest, NewDataFiles) {
  initializeGFM(std::make_shared<ForeignStorageInterface>(), 1);

  const auto table_data_dir = fs::path(kFileMgrPath) / "table_1_1";
  const auto legacy_data_file_path =
      table_data_dir / ("0." + std::to_string(page_size_) + ".mapd");
  const auto new_data_file_path =
      table_data_dir / ("0." + std::to_string(page_size_) + ".data");
  const auto legacy_metadata_file_path =
      table_data_dir / ("1." + std::to_string(DEFAULT_METADATA_PAGE_SIZE) + ".mapd");
  const auto new_metadata_file_path =
      table_data_dir / ("1." + std::to_string(DEFAULT_METADATA_PAGE_SIZE) + ".data");

  ASSERT_TRUE(fs::exists(new_data_file_path));
  ASSERT_TRUE(fs::is_regular_file(new_data_file_path));

  ASSERT_TRUE(fs::exists(new_metadata_file_path));
  ASSERT_TRUE(fs::is_regular_file(new_metadata_file_path));

  ASSERT_TRUE(fs::exists(legacy_data_file_path));
  ASSERT_TRUE(fs::is_symlink(legacy_data_file_path));

  ASSERT_TRUE(fs::exists(legacy_metadata_file_path));
  ASSERT_TRUE(fs::is_symlink(legacy_metadata_file_path));
}

enum class FileMgrType { FileMgr, CachingFileMgr };
std::ostream& operator<<(std::ostream& os, const FileMgrType& type) {
  if (type == FileMgrType::FileMgr) {
    os << "FileMgr";
  } else if (type == FileMgrType::CachingFileMgr) {
    os << "CachingFileMgr";
  } else {
    os << "Unknown";
  }
  return os;
}

class ReadOnlyAbstractFileMgrUnitTest : public AbstractFileMgrTest {
 public:
  void initAsCachingFileMgr() {
    fn::DiskCacheConfig config{kDataDir, fn::DiskCacheLevel::all};
    config.page_size = page_size;
    config.meta_page_size = meta_page_size;
    parent_file_mgr_ = std::make_unique<fn::CachingFileMgr>(config);
    file_mgr_ = static_cast<fn::FileMgr*>(parent_file_mgr_.get());
  }

  void initAsFileMgr() {
    parent_file_mgr_ =
        std::make_unique<fn::GlobalFileMgr>(0,
                                            std::make_shared<ForeignStorageInterface>(),
                                            kDataDir,
                                            0,
                                            page_size,
                                            meta_page_size);
    file_mgr_ =
        static_cast<fn::FileMgr*>(static_cast<fn::GlobalFileMgr*>(parent_file_mgr_.get())
                                      ->getFileMgr(db_id, tb_id));
  }

  void initMgrAsType(const FileMgrType& type) {
    if (type == FileMgrType::FileMgr) {
      initAsFileMgr();
    } else if (type == FileMgrType::CachingFileMgr) {
      initAsCachingFileMgr();
    } else {
      UNREACHABLE() << "Unknown FileMgrType";
    }
  }

  fn::FileInfo createFileInfo() {
    auto [fd, file_path] = fn::create(kDataDir, data_file_id, page_size, num_pages);
    return fn::FileInfo(file_mgr_, 0, fd, page_size, num_pages, file_path);
  }

  void putData(const ChunkKey& key = default_key,
               const std::vector<int32_t>& data = {1}) {
    if (auto cfm = dynamic_cast<fn::CachingFileMgr*>(file_mgr_)) {
      TestHelpers::TestBuffer in_buf{data};
      in_buf.clearDirtyBits();
      cfm->putBuffer(key, &in_buf);
    } else if (auto fm = dynamic_cast<fn::FileMgr*>(file_mgr_)) {
      TestHelpers::TestBuffer in_buf{data};
      fm->putBuffer(key, &in_buf);
      fm->checkpoint();
    } else {
      UNREACHABLE() << "Unknown file mgr type";
    }
  }

  void writeVersionFile(int32_t version) {
    fs::remove_all(version_file_name);
    std::ofstream version_file(version_file_name);
    version_file.write(reinterpret_cast<char*>(&version), sizeof(version));
  }

  fn::GlobalFileMgr* getGlobalFileMgr() {
    return static_cast<fn::GlobalFileMgr*>(parent_file_mgr_.get());
  }

  fn::CachingFileMgr* getCachingFileMgr() {
    return static_cast<fn::CachingFileMgr*>(parent_file_mgr_.get());
  }

  fn::FileMgr* file_mgr_;

 private:
  // This is the owner of the file_mgr_ pointer (either a CachingFileMgr or a
  // GlobalFileMgr).  It should not need to be referenced outside if initialization.
  std::unique_ptr<Data_Namespace::AbstractBufferMgr> parent_file_mgr_;
};

// Tests where CachingFileMgr/FileMgr functions will expect the same results.
class ParamReadOnlyFileMgrUnitTest : public ReadOnlyAbstractFileMgrUnitTest,
                                     public ::testing::WithParamInterface<FileMgrType> {
 public:
  void SetUp() override {
    g_read_only = false;
    ReadOnlyAbstractFileMgrUnitTest::SetUp();
    initMgrAsType(GetParam());
    g_read_only = true;
  }
};

// Right now these tests just check to make sure we don't crash.  TODO(Misiu):
// Validate the results of these tests (may require splitting some of them into
// separate test fixures if expected results vary by type).
TEST_P(ParamReadOnlyFileMgrUnitTest, getStorageStats) {
  file_mgr_->getStorageStats();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, createBuffer) {
  file_mgr_->createBuffer(default_key);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, createBufferWithData) {
  run_in_write_mode([&] { putData(); });
  ASSERT_DEATH(file_mgr_->createBuffer(default_key), "already exists");
}

TEST_P(ParamReadOnlyFileMgrUnitTest, isBufferOnDevice) {
  file_mgr_->isBufferOnDevice(default_key);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getBuffer) {
  ASSERT_DEATH(file_mgr_->getBuffer(default_key), "not exist");
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getBufferWithData) {
  run_in_write_mode([&] { putData(); });
  file_mgr_->getBuffer(default_key);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, fetchBuffer) {
  TestHelpers::TestBuffer buf{SQLTypeInfo{kINT}};
  ASSERT_DEATH(file_mgr_->fetchBuffer(default_key, &buf, 4), "not exist");
}

TEST_P(ParamReadOnlyFileMgrUnitTest, fetchBufferWithData) {
  run_in_write_mode([&] { putData(); });
  TestHelpers::TestBuffer buf{SQLTypeInfo{kINT}};
  file_mgr_->fetchBuffer(default_key, &buf, 4);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getMgrType) {
  file_mgr_->getMgrType();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getStringMgrType) {
  file_mgr_->getStringMgrType();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, printSlabs) {
  file_mgr_->printSlabs();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getMaxSize) {
  file_mgr_->getMaxSize();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getInUseSize) {
  file_mgr_->getInUseSize();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getAllocated) {
  file_mgr_->getAllocated();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, isAllocationCapped) {
  file_mgr_->isAllocationCapped();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getFileInfoForFileId) {
  // The file info map should be empty for a FileMgr with no files, so this should throw
  // with a generic out-of-range exception.
  run_and_catch([&] { file_mgr_->getFileInfoForFileId(data_file_id); }, "map::at");
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getFileInfoForFileIdWithData) {
  run_in_write_mode([&] { putData(); });
  file_mgr_->getFileInfoForFileId(data_file_id);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getMetadataForFile) {
  std::ofstream temp_file(kTempFile);
  bf::path path(kDataDir);
  bf::directory_iterator it(path);
  file_mgr_->getMetadataForFile(it);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getChunkMetadataVecForKeyPrefix) {
  ChunkMetadataVector meta_vec;
  file_mgr_->getChunkMetadataVecForKeyPrefix(meta_vec, {db_id, tb_id});
}

TEST_P(ParamReadOnlyFileMgrUnitTest, hasChunkMetadataForKeyPrefix) {
  file_mgr_->hasChunkMetadataForKeyPrefix({db_id, tb_id});
}

TEST_P(ParamReadOnlyFileMgrUnitTest, epoch) {
  file_mgr_->epoch(1, 1);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, epochFloor) {
  file_mgr_->epochFloor();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, incrementEpoch) {
  file_mgr_->incrementEpoch();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, lastCheckpointedEpoch) {
  file_mgr_->lastCheckpointedEpoch();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, resetEpochFloor) {
  file_mgr_->resetEpochFloor();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, maxRollbackEpochs) {
  file_mgr_->maxRollbackEpochs();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getNumReaderThreads) {
  file_mgr_->getNumReaderThreads();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getFileForFileId) {
  ASSERT_DEATH(file_mgr_->getFileForFileId(data_file_id), "not exist");
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getFileForFileIdWithData) {
  run_in_write_mode([&] { putData(); });
  file_mgr_->getFileForFileId(data_file_id);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getNumChunks) {
  file_mgr_->getNumChunks();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getNumUsedMetadataPagesForChunkKey) {
  run_and_catch([&] { file_mgr_->getNumUsedMetadataPagesForChunkKey(default_key); },
                "not found");
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getNumUsedMetadataPagesForChunkKeyWithData) {
  run_in_write_mode([&] { putData(); });
  file_mgr_->getNumUsedMetadataPagesForChunkKey(default_key);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, createOrMigrateTopLevelMetadataWithData) {
  run_in_write_mode([&] { file_mgr_->createOrMigrateTopLevelMetadata(); });
  file_mgr_->createOrMigrateTopLevelMetadata();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getFileMgrBasePath) {
  file_mgr_->getFileMgrBasePath();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, hasFileMgrKey) {
  file_mgr_->hasFileMgrKey();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getFileMgrKey) {
  file_mgr_->get_fileMgrKey();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getFilePath) {
  file_mgr_->getFilePath("");
}

TEST_P(ParamReadOnlyFileMgrUnitTest, failOnReadError) {
  file_mgr_->failOnReadError();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getPageSize) {
  file_mgr_->getPageSize();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, getMetadataPageSize) {
  file_mgr_->getMetadataPageSize();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, describeSelf) {
  file_mgr_->describeSelf();
}

TEST_P(ParamReadOnlyFileMgrUnitTest, setNumPagesPerDataFile) {
  file_mgr_->setNumPagesPerDataFile(256);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, setNumPagesPerMetadataFile) {
  file_mgr_->setNumPagesPerMetadataFile(4096);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, renameAndSymlinkLegacyFiles) {
  file_mgr_->renameAndSymlinkLegacyFiles(kDataDir);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, updatePageIfDeleted) {
  auto [fd, file_path] = fn::create(kDataDir, data_file_id, page_size, num_pages);
  auto file_info = fn::FileInfo(file_mgr_, 0, fd, page_size, num_pages, file_path);
  ChunkKey key{1, 1, 1, 1};
  file_mgr_->updatePageIfDeleted(&file_info, key, 0, 0, 0);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, deleteBuffer) {
  ASSERT_DEATH(file_mgr_->deleteBuffer(default_key), "not exist");
}

TEST_P(ParamReadOnlyFileMgrUnitTest, deleteBuffersWithPrefix) {
  // If there are no buffers then this does not throw.
  file_mgr_->deleteBuffersWithPrefix({1, 1});
}

TEST_P(ParamReadOnlyFileMgrUnitTest, copyPage) {
  fn::Page page1{data_file_id, 0}, page2{data_file_id, 1};
  run_and_catch([&] { file_mgr_->copyPage(page1, file_mgr_, page2, 4, 0, 0); },
                "map::at");
}

TEST_P(ParamReadOnlyFileMgrUnitTest, updatePageIfDeletedWithCheckpoint) {
  auto file_info = createFileInfo();
  ChunkKey key = default_key;
  file_mgr_->updatePageIfDeleted(&file_info, key, fn::DELETE_CONTINGENT, 0, 0);
}

TEST_P(ParamReadOnlyFileMgrUnitTest, updatePageIfDeletedWithoutCheckpoint) {
  auto file_info = createFileInfo();
  ChunkKey key = default_key;
  file_mgr_->updatePageIfDeleted(&file_info, key, fn::DELETE_CONTINGENT, -1, 0);
}

INSTANTIATE_TEST_SUITE_P(SharedReadOnlyFileMgrUnitTest,
                         ParamReadOnlyFileMgrUnitTest,
                         testing::Values(FileMgrType::FileMgr,
                                         FileMgrType::CachingFileMgr),
                         [](const auto& info) {
                           std::stringstream ss;
                           ss << info.param;
                           return ss.str();
                         });

// Tests where FileMgr functions will expect different results.
class ReadOnlyFileMgrUnitTest : public ReadOnlyAbstractFileMgrUnitTest {
 public:
  void SetUp() override {
    g_read_only = false;
    ReadOnlyAbstractFileMgrUnitTest::SetUp();
    initAsFileMgr();
    g_read_only = true;
  }
};

TEST_F(ReadOnlyFileMgrUnitTest, ConstructorEmpty) {
  // Constructor will create new files if file mgr did not exist.
  ASSERT_FALSE(fs::exists(getGlobalFileMgr()->getBasePath() + "table_1_0"));
  ASSERT_DEATH(fn::FileMgr(0, getGlobalFileMgr(), {1, 0}), kReadOnlyCreateError);
}

TEST_F(ReadOnlyFileMgrUnitTest, Constructor) {
  run_in_write_mode([&] { putData(); });
  fn::FileMgr(0, getGlobalFileMgr(), table_pair);
}

TEST_F(ReadOnlyFileMgrUnitTest, ConstructorCoreInit) {
  run_in_write_mode([&] { putData(); });
  fn::FileMgr(0, getGlobalFileMgr(), table_pair, true);
}

TEST_F(ReadOnlyFileMgrUnitTest, ConstructorCoreInitFalse) {
  run_in_write_mode([&] { putData(); });
  fn::FileMgr(0, getGlobalFileMgr(), table_pair, false);
}

TEST_F(ReadOnlyFileMgrUnitTest, ConstructorInvalidMigrateMissingVersion) {
  run_in_write_mode([&] { putData(); });
  fs::remove_all(version_file_name);
  ASSERT_DEATH(fn::FileMgr(0, getGlobalFileMgr(), table_pair, true), kReadOnlyWriteError);
}

TEST_F(ReadOnlyFileMgrUnitTest, ConstructorInvalidMigrateV0) {
  run_in_write_mode([&] { putData(); });
  writeVersionFile(0);
  ASSERT_DEATH(fn::FileMgr(0, getGlobalFileMgr(), table_pair, true), "migrate epoch");
}

TEST_F(ReadOnlyFileMgrUnitTest, ConstructorInvalidMigrateV1) {
  run_in_write_mode([&] { putData(); });
  writeVersionFile(1);
  ASSERT_DEATH(fn::FileMgr(0, getGlobalFileMgr(), table_pair, true),
               "migrate file format");
}

TEST_F(ReadOnlyFileMgrUnitTest, ConstructorPath) {
  run_in_write_mode([&] { putData(); });
  ASSERT_DEATH(fn::FileMgr(getGlobalFileMgr(), table_dir), kReadOnlyCreateError);
}

TEST_F(ReadOnlyFileMgrUnitTest, createOrMigrateTopLevelMetadata) {
  ASSERT_DEATH(file_mgr_->createOrMigrateTopLevelMetadata(), kReadOnlyWriteError);
}

TEST_F(ReadOnlyFileMgrUnitTest, freePage) {
  file_mgr_->free_page(std::make_pair<fn::FileInfo*, int32_t>(nullptr, 0));
}

TEST_F(ReadOnlyFileMgrUnitTest, writePageMappingsToStatusFile) {
  std::ofstream(file_mgr_->getFilePath(fn::FileMgr::COPY_PAGES_STATUS));
  std::vector<fn::PageMapping> page_mappings;
  ASSERT_DEATH(file_mgr_->writePageMappingsToStatusFile(page_mappings),
               kReadOnlyWriteError);
}

TEST_F(ReadOnlyFileMgrUnitTest, renameCompactionStatusFile) {
  std::ofstream(file_mgr_->getFilePath("from"));
  ASSERT_DEATH(file_mgr_->renameCompactionStatusFile("from", "to"), kReadOnlyWriteError);
}

TEST_F(ReadOnlyFileMgrUnitTest, compactFiles) {
  ASSERT_DEATH(file_mgr_->compactFiles(), "run file compaction");
}

TEST_F(ReadOnlyFileMgrUnitTest, closeRemovePhysical) {
  ASSERT_DEATH(file_mgr_->closeRemovePhysical(), "Error trying to rename file");
}

TEST_F(ReadOnlyFileMgrUnitTest, copyPageWithData) {
  run_in_write_mode([&] { putData(); });
  fn::Page page1{data_file_id, 0}, page2{data_file_id, 1};
  ASSERT_DEATH(file_mgr_->copyPage(page1, file_mgr_, page2, 4, 0, 0),
               kReadOnlyWriteError);
}

// TODO(Misiu): This interface is unused (and may not work properly for CFM).  Look into
// removing it during next FileMgr refactor.
TEST_F(ReadOnlyFileMgrUnitTest, requestFreePages) {
  // FileInfo does not exist, so we expect a create error.
  std::vector<fn::Page> pages;
  ASSERT_DEATH(file_mgr_->requestFreePages(1, page_size, pages, false),
               kReadOnlyCreateError);
}

TEST_F(ReadOnlyFileMgrUnitTest, createFile) {
  std::string path{"test_path"};
  ASSERT_DEATH(file_mgr_->createFile(path, 64), "Error trying to create file");
}

TEST_F(ReadOnlyFileMgrUnitTest, writeFile) {
  FILE* f = fn::create(kTempFile, 1);
  int8_t buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  ASSERT_DEATH(file_mgr_->writeFile(f, 0, 8, buf), kReadOnlyWriteError);
  fclose(f);
}

TEST_F(ReadOnlyFileMgrUnitTest, putBuffer) {
  TestHelpers::TestBuffer buf{std::vector<int32_t>{1}};
  buf.setUpdated();
  ASSERT_DEATH(file_mgr_->putBuffer(default_key, &buf), kReadOnlyCreateError);
}

TEST_F(ReadOnlyFileMgrUnitTest, putBufferWithData) {
  // If a file already exists we should get a different error.
  run_in_write_mode([&] { putData(); });
  TestHelpers::TestBuffer buf{std::vector<int32_t>{1}};
  buf.setUpdated();
  ASSERT_DEATH(file_mgr_->putBuffer(default_key, &buf), kReadOnlyWriteError);
}

TEST_F(ReadOnlyFileMgrUnitTest, checkpoint) {
  ASSERT_DEATH(file_mgr_->checkpoint(), kReadOnlyWriteError);
}

TEST_F(ReadOnlyFileMgrUnitTest, requestFreePage) {
  ASSERT_DEATH(file_mgr_->requestFreePage(page_size, false), kReadOnlyCreateError);
}

TEST_F(ReadOnlyFileMgrUnitTest, deleteBufferWithData) {
  run_in_write_mode([&] { putData(); });
  ASSERT_DEATH(file_mgr_->deleteBuffer(default_key), kReadOnlyWriteError);
}

TEST_F(ReadOnlyFileMgrUnitTest, deleteBuffersWithPrefixWithData) {
  run_in_write_mode([&] { putData(); });
  ASSERT_DEATH(file_mgr_->deleteBuffersWithPrefix({1, 1}), kReadOnlyWriteError);
}

// Tests where CachingFileMgr functions will expect different results.
class ReadOnlyCachingFileMgrUnitTest : public ReadOnlyAbstractFileMgrUnitTest {
 public:
  void SetUp() override {
    g_read_only = false;
    AbstractFileMgrTest::SetUp();
    initAsCachingFileMgr();
    g_read_only = true;
  }
};

TEST_F(ReadOnlyCachingFileMgrUnitTest, Constructor) {
  fn::DiskCacheConfig config{kDataDir, fn::DiskCacheLevel::all};
  config.page_size = page_size;
  config.meta_page_size = meta_page_size;
  auto cfm = fn::CachingFileMgr(config);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, createOrMigrateTopLevelMetadata) {
  file_mgr_->createOrMigrateTopLevelMetadata();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, freePage) {
  // CFM needs more setup for this tests because it dereferences the file_info.
  auto file_info = createFileInfo();
  file_mgr_->free_page(std::make_pair<fn::FileInfo*, int32_t>(&file_info, 0));
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, writePageMappingsToStatusFile) {
  std::ofstream(file_mgr_->getFilePath(fn::FileMgr::COPY_PAGES_STATUS));
  std::vector<fn::PageMapping> page_mappings;
  file_mgr_->writePageMappingsToStatusFile(page_mappings);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, renameCompactionStatusFile) {
  std::ofstream(file_mgr_->getFilePath("from"));
  file_mgr_->renameCompactionStatusFile("from", "to");
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, compactFiles) {
  file_mgr_->compactFiles();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, closeRemovePhysical) {
  file_mgr_->closeRemovePhysical();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, copyPageWithData) {
  run_in_write_mode([&] { putData(); });
  fn::Page page1{data_file_id, 0}, page2{data_file_id, 1};
  file_mgr_->copyPage(page1, file_mgr_, page2, 4, 0, 0);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, createFile) {
  std::string path{"test_path"};
  fclose(file_mgr_->createFile(path, 64));
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, writeFile) {
  FILE* f = fn::create(kTempFile, 1);
  int8_t buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  file_mgr_->writeFile(f, 0, 8, buf);
  fclose(f);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, putBuffer) {
  TestHelpers::TestBuffer buf{std::vector<int32_t>{1}};
  buf.clearDirtyBits();
  file_mgr_->putBuffer(default_key, &buf);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, checkpoint) {
  ASSERT_DEATH(file_mgr_->checkpoint(1, 1), "No data for table");
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, checkpointWithData) {
  putData();
  file_mgr_->checkpoint(1, 1);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, requestFreePage) {
  file_mgr_->requestFreePage(page_size, false);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, deleteBufferWithData) {
  run_in_write_mode([&] { putData(); });
  file_mgr_->deleteBuffer(default_key);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, deleteBuffersWithPrefixWithData) {
  run_in_write_mode([&] { putData(); });
  file_mgr_->deleteBuffersWithPrefix({1, 1});
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getMinimumSize) {
  fn::CachingFileMgr::getMinimumSize();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getMaxDataFiles) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getMaxDataFiles();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getMaxMetaFiles) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getMaxMetaFiles();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getMaxWrapperSize) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getMaxWrapperSize();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getDataFileSize) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getDataFileSize();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getMetadataFileSize) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getMetadataFileSize();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getNumDataFiles) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getNumDataFiles();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getNumMetaFiles) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getNumMetaFiles();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getAvailableSpace) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getAvailableSpace();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getAvailableWrapperSpaec) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getAvailableWrapperSpace();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getMaxDataFilesSize) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getMaxDataFilesSize();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, removeChunkKeepMetadata) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->removeChunkKeepMetadata(default_key);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, clearForTable) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->clearForTable(db_id, tb_id);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getChunkSpaceReservedByTable) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getChunkSpaceReservedByTable(db_id, tb_id);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getMetadataSpaceReservedByTable) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getMetadataSpaceReservedByTable(db_id,
                                                                               tb_id);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getTableFileMgrSpaceReserved) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getTableFileMgrSpaceReserved(db_id, tb_id);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getSpaceReservedByTable) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getSpaceReservedByTable(db_id, tb_id);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, deleteBufferIfExists) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->deleteBufferIfExists(default_key);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getNumChunksWithMetadata) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getNumChunksWithMetadata();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getNumDataChunks) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getNumDataChunks();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getChunkKeysForPrefix) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getChunkKeysForPrefix({db_id, tb_id});
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, reconstruct) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->reconstruct();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, deleteWrapperFile) {
  ASSERT_DEATH(
      static_cast<fn::CachingFileMgr*>(file_mgr_)->deleteWrapperFile(db_id, tb_id),
      "Wrapper does not exist");
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, deleteWrapperFileWithData) {
  run_in_write_mode([&] { putData(); });
  static_cast<fn::CachingFileMgr*>(file_mgr_)->deleteWrapperFile(db_id, tb_id);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, writeWrapperFile) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->writeWrapperFile("file", db_id, tb_id);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, hasWrapperFile) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->hasWrapperFile(db_id, tb_id);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getTableFileMgrPath) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getTableFileMgrPath(db_id, tb_id);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getFilesSize) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getFilesSize();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getTableFileMgrsSize) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getTableFileMgrsSize();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getBufferIfExists) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getBufferIfExists(default_key);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, dumpKeysWithMetadata) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->dumpKeysWithMetadata();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, dumpKeysWithChunkData) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->dumpKeysWithChunkData();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, dumpTableQueue) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->dumpTableQueue();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, dumpEvictionQueue) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->dumpEvictionQueue();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, dump) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->dump();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, setMaxNumDataFiles) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->setMaxNumDataFiles(1);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, setMaxNumMetadataFiles) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->setMaxNumMetadataFiles(1);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, setMaxWrapperSpace) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->setMaxWrapperSpace(4096);
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, getKeysWithMetadata) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->getKeysWithMetadata();
}

TEST_F(ReadOnlyCachingFileMgrUnitTest, setDataSizeLimit) {
  static_cast<fn::CachingFileMgr*>(file_mgr_)->setDataSizeLimit(1 << 16);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  // This line can be replaced with 'GTEST_FLAG_SET(death_test_style, "threadsafe")' once
  // we update to a version of GTEST that supports it.
  (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
