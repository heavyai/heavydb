/*
 * Copyright 2020 OmniSci, Inc.
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
#include "DBHandlerTestHelpers.h"
#include "DataMgr/FileMgr/FileMgr.h"
#include "DataMgr/FileMgr/GlobalFileMgr.h"
#include "TestHelpers.h"

class FileMgrTest : public DBHandlerTestFixture {
 protected:
  std::string table_name;
  Data_Namespace::DataMgr* dm;
  ChunkKey chunk_key;
  std::pair<int, int> file_mgr_key;
  File_Namespace::GlobalFileMgr* gfm;

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    Catalog_Namespace::Catalog* cat = &getCatalog();
    dm = &cat->getDataMgr();
    gfm = dm->getGlobalFileMgr();

    table_name = "test_table";
    sql("DROP TABLE IF EXISTS " + table_name + ";");
    sql("CREATE TABLE " + table_name + " (col1 INT)");
    sql("INSERT INTO " + table_name + " VALUES(1)");
    const TableDescriptor* td = cat->getMetadataForTable(table_name);
    const auto col_descs =
        cat->getAllColumnMetadataForTable(td->tableId, false, false, false);
    const ColumnDescriptor* cd = *(col_descs.begin());
    int db_id = cat->getCurrentDB().dbId;
    int tb_id = td->tableId;
    chunk_key = {db_id, tb_id, cd->columnId, 0};
    file_mgr_key = std::make_pair(db_id, tb_id);
  }

  void TearDown() override {
    sql("DROP TABLE " + table_name);
    DBHandlerTestFixture::TearDown();
  }

  void compareBuffers(AbstractBuffer* left_buffer,
                      AbstractBuffer* right_buffer,
                      size_t num_bytes) {
    int8_t left_array[num_bytes];
    int8_t right_array[num_bytes];
    left_buffer->read(left_array, num_bytes);
    right_buffer->read(right_array, num_bytes);
    ASSERT_EQ(std::memcmp(left_array, right_array, num_bytes), 0);
    ASSERT_EQ(left_buffer->hasEncoder(), right_buffer->hasEncoder());
  }

  void compareBuffersAndMetadata(AbstractBuffer* left_buffer,
                                 AbstractBuffer* right_buffer,
                                 size_t num_bytes) {
    int8_t left_array[num_bytes];
    int8_t right_array[num_bytes];
    left_buffer->read(left_array, num_bytes);
    right_buffer->read(right_array, num_bytes);
    ASSERT_EQ(std::memcmp(left_array, right_array, num_bytes), 0);
    ASSERT_EQ(left_buffer->hasEncoder(), right_buffer->hasEncoder());
    if (left_buffer->hasEncoder()) {
      const std::shared_ptr<ChunkMetadata> left_chunk_metadata =
          std::make_shared<ChunkMetadata>();
      const std::shared_ptr<ChunkMetadata> right_chunk_metadata =
          std::make_shared<ChunkMetadata>();
      left_buffer->getEncoder()->getMetadata(left_chunk_metadata);
      right_buffer->getEncoder()->getMetadata(right_chunk_metadata);
      ASSERT_EQ(*left_chunk_metadata, *right_chunk_metadata);
    }
  }
};

TEST_F(FileMgrTest, putBuffer_update) {
  AbstractBuffer* source_buffer =
      dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
  source_buffer->setUpdated();
  auto file_mgr =
      File_Namespace::FileMgr(0, gfm, file_mgr_key, 0, 0, gfm->getDefaultPageSize());
  AbstractBuffer* file_buffer = file_mgr.putBuffer(chunk_key, source_buffer, 4);
  compareBuffersAndMetadata(source_buffer, file_buffer, 4);
  ASSERT_FALSE(source_buffer->isAppended());
  ASSERT_FALSE(source_buffer->isUpdated());
  ASSERT_FALSE(source_buffer->isDirty());
}

TEST_F(FileMgrTest, putBuffer_subwrite) {
  AbstractBuffer* source_buffer =
      dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
  int8_t temp_array[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  source_buffer->write(temp_array, 8);
  auto file_mgr =
      File_Namespace::FileMgr(0, gfm, file_mgr_key, 0, 0, gfm->getDefaultPageSize());
  AbstractBuffer* file_buffer = file_mgr.putBuffer(chunk_key, source_buffer, 4);
  compareBuffers(source_buffer, file_buffer, 4);
}

TEST_F(FileMgrTest, putBuffer_exists) {
  AbstractBuffer* source_buffer =
      dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
  int8_t temp_array[4] = {1, 2, 3, 4};
  source_buffer->write(temp_array, 4);
  auto file_mgr =
      File_Namespace::FileMgr(0, gfm, file_mgr_key, 0, 0, gfm->getDefaultPageSize());
  file_mgr.putBuffer(chunk_key, source_buffer, 4);
  source_buffer->write(temp_array, 4);
  AbstractBuffer* file_buffer = file_mgr.putBuffer(chunk_key, source_buffer, 4);
  compareBuffersAndMetadata(source_buffer, file_buffer, 4);
}

TEST_F(FileMgrTest, putBuffer_append) {
  AbstractBuffer* source_buffer =
      dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
  int8_t temp_array[4] = {1, 2, 3, 4};
  source_buffer->append(temp_array, 4);
  auto file_mgr =
      File_Namespace::FileMgr(0, gfm, file_mgr_key, 0, 0, gfm->getDefaultPageSize());
  AbstractBuffer* file_buffer = file_mgr.putBuffer(chunk_key, source_buffer, 8);
  compareBuffersAndMetadata(source_buffer, file_buffer, 8);
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
