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

#include <fstream>

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "DBHandlerTestHelpers.h"
#include "DataMgr/FileMgr/FileMgr.h"
#include "DataMgr/FileMgr/GlobalFileMgr.h"
#include "Shared/File.h"
#include "TestHelpers.h"

#include "DataMgr/ForeignStorage/ArrowForeignStorage.h"

class FileMgrTest : public DBHandlerTestFixture {
 protected:
  std::string table_name;
  Data_Namespace::DataMgr* dm;
  ChunkKey chunk_key;
  std::pair<int, int> file_mgr_key;
  File_Namespace::GlobalFileMgr* gfm;
  Catalog_Namespace::Catalog* cat;

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    cat = &getCatalog();
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

  ChunkKey setUpCappedRollbackTable(const int32_t max_rollback_epochs) {
    Catalog_Namespace::Catalog* cat = &getCatalog();
    const std::string capped_table_name("capped_table");
    sql("DROP TABLE IF EXISTS " + capped_table_name + ";");
    std::stringstream capped_ddl;
    capped_ddl << "CREATE TABLE " << capped_table_name << " "
               << "(col1 INT) WITH (max_rollback_epochs=" << max_rollback_epochs << ")";
    sql(capped_ddl.str());
    sql("INSERT INTO " + capped_table_name + " VALUES(1)");
    const TableDescriptor* td = cat->getMetadataForTable(capped_table_name);
    const auto col_descs =
        cat->getAllColumnMetadataForTable(td->tableId, false, false, false);
    const ColumnDescriptor* cd = *(col_descs.begin());
    const int db_id = cat->getCurrentDB().dbId;
    const int tb_id = td->tableId;
    const ChunkKey capped_chunk_key = {db_id, tb_id, cd->columnId, 0};
    return capped_chunk_key;
  }

  void compareBuffers(AbstractBuffer* left_buffer,
                      AbstractBuffer* right_buffer,
                      size_t num_bytes) {
    std::vector<int8_t> left_array(num_bytes);
    std::vector<int8_t> right_array(num_bytes);
    left_buffer->read(left_array.data(), num_bytes);
    right_buffer->read(right_array.data(), num_bytes);
    ASSERT_EQ(std::memcmp(left_array.data(), right_array.data(), num_bytes), 0);
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
    CHECK(data_buffer->hasEncoder());
    SQLTypeInfo sql_type_info = getMetadataForBuffer(data_buffer)->sqlType;
    int8_t* append_ptr = getDataPtr(append_data);
    data_buffer->getEncoder()->appendData(append_ptr, append_data.size(), sql_type_info);
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
};

TEST_F(FileMgrTest, putBuffer_update) {
  AbstractBuffer* source_buffer =
      dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
  source_buffer->setUpdated();
  auto file_mgr =
      File_Namespace::FileMgr(0, gfm, file_mgr_key, -1, 0, -1, gfm->getDefaultPageSize());
  AbstractBuffer* file_buffer = file_mgr.putBuffer(chunk_key, source_buffer, 4);
  compareBuffersAndMetadata(source_buffer, file_buffer);
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
      File_Namespace::FileMgr(0, gfm, file_mgr_key, -1, 0, -1, gfm->getDefaultPageSize());
  AbstractBuffer* file_buffer = file_mgr.putBuffer(chunk_key, source_buffer, 4);
  compareBuffers(source_buffer, file_buffer, 4);
}

TEST_F(FileMgrTest, putBuffer_exists) {
  AbstractBuffer* source_buffer =
      dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
  int8_t temp_array[4] = {1, 2, 3, 4};
  source_buffer->write(temp_array, 4);
  auto file_mgr =
      File_Namespace::FileMgr(0, gfm, file_mgr_key, -1, 0, -1, gfm->getDefaultPageSize());
  file_mgr.putBuffer(chunk_key, source_buffer, 4);
  file_mgr.checkpoint();
  source_buffer->write(temp_array, 4);
  AbstractBuffer* file_buffer = file_mgr.putBuffer(chunk_key, source_buffer, 4);
  compareBuffersAndMetadata(source_buffer, file_buffer);
}

TEST_F(FileMgrTest, putBuffer_append) {
  AbstractBuffer* source_buffer =
      dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
  int8_t temp_array[4] = {1, 2, 3, 4};
  source_buffer->append(temp_array, 4);
  auto file_mgr =
      File_Namespace::FileMgr(0, gfm, file_mgr_key, -1, 0, -1, gfm->getDefaultPageSize());
  AbstractBuffer* file_buffer = file_mgr.putBuffer(chunk_key, source_buffer, 8);
  compareBuffersAndMetadata(source_buffer, file_buffer);
}

TEST_F(FileMgrTest, put_checkpoint_get) {
  AbstractBuffer* source_buffer =
      dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
  std::vector<int32_t> data_v1 = {1, 2, 3, 5, 7};
  appendData(source_buffer, data_v1);
  File_Namespace::FileMgr* file_mgr = dynamic_cast<File_Namespace::FileMgr*>(
      dm->getGlobalFileMgr()->getFileMgr(file_mgr_key.first, file_mgr_key.second));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  AbstractBuffer* file_buffer_put = file_mgr->putBuffer(chunk_key, source_buffer, 24);
  ASSERT_TRUE(file_buffer_put->isDirty());
  ASSERT_FALSE(file_buffer_put->isUpdated());
  ASSERT_TRUE(file_buffer_put->isAppended());
  ASSERT_EQ(file_buffer_put->size(), static_cast<size_t>(24));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  file_mgr->checkpoint();
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  AbstractBuffer* file_buffer_get = file_mgr->getBuffer(chunk_key, 24);
  ASSERT_EQ(file_buffer_put, file_buffer_get);
  CHECK(!(file_buffer_get->isDirty()));
  CHECK(!(file_buffer_get->isUpdated()));
  CHECK(!(file_buffer_get->isAppended()));
  ASSERT_EQ(file_buffer_get->size(), static_cast<size_t>(24));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  compareBuffersAndMetadata(source_buffer, file_buffer_get);
}

TEST_F(FileMgrTest, put_checkpoint_get_double_write) {
  AbstractBuffer* source_buffer =
      dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
  std::vector<int32_t> data_v1 = {1, 2, 3, 5, 7};
  std::vector<int32_t> data_v2 = {11, 13, 17, 19};
  appendData(source_buffer, data_v1);
  File_Namespace::FileMgr* file_mgr = dynamic_cast<File_Namespace::FileMgr*>(
      dm->getGlobalFileMgr()->getFileMgr(file_mgr_key.first, file_mgr_key.second));
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  file_mgr->putBuffer(chunk_key, source_buffer, 24);
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
  file_mgr->checkpoint();
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  AbstractBuffer* file_buffer = file_mgr->getBuffer(chunk_key, 24);
  ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
  ASSERT_FALSE(file_buffer->isDirty());
  ASSERT_EQ(file_buffer->size(), static_cast<size_t>(24));
  compareBuffersAndMetadata(source_buffer, file_buffer);
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
  appendData(source_buffer, data_v2);
  appendData(source_buffer, data_v2);
  compareBuffersAndMetadata(source_buffer, file_buffer);
}

TEST_F(FileMgrTest, buffer_append_and_recovery) {
  AbstractBuffer* source_buffer =
      dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
  ASSERT_EQ(getMetadataForBuffer(source_buffer)->numElements, static_cast<size_t>(1));
  std::vector<int32_t> data_v1 = {1, 2, 3, 5, 7};
  std::vector<int32_t> data_v2 = {11, 13, 17, 19};

  appendData(source_buffer, data_v1);
  {
    File_Namespace::FileMgr* file_mgr = dynamic_cast<File_Namespace::FileMgr*>(
        dm->getGlobalFileMgr()->getFileMgr(file_mgr_key.first, file_mgr_key.second));
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 1);
    AbstractBuffer* file_buffer = file_mgr->putBuffer(chunk_key, source_buffer, 24);
    file_mgr->checkpoint();
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
    ASSERT_EQ(getMetadataForBuffer(source_buffer)->numElements, static_cast<size_t>(6));
    ASSERT_EQ(getMetadataForBuffer(file_buffer)->numElements, static_cast<size_t>(6));
    SCOPED_TRACE("Buffer Append and Recovery - Compare #1");
    compareBuffersAndMetadata(source_buffer, file_buffer);

    // Now write data we will not checkpoint
    appendData(file_buffer, data_v1);
    ASSERT_EQ(file_buffer->size(), static_cast<size_t>(44));
    // Now close filemgr to test recovery
    cat->removeFragmenterForTable(file_mgr_key.second);
    dm->getGlobalFileMgr()->closeFileMgr(file_mgr_key.first, file_mgr_key.second);
  }

  {
    File_Namespace::FileMgr* file_mgr = dynamic_cast<File_Namespace::FileMgr*>(
        dm->getGlobalFileMgr()->getFileMgr(file_mgr_key.first, file_mgr_key.second));
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 2);
    ChunkMetadataVector chunkMetadataVector;
    file_mgr->getChunkMetadataVecForKeyPrefix(chunkMetadataVector, chunk_key);
    ASSERT_EQ(chunkMetadataVector.size(), static_cast<size_t>(1));
    ASSERT_EQ(std::memcmp(chunkMetadataVector[0].first.data(), chunk_key.data(), 16), 0);
    ASSERT_EQ(chunkMetadataVector[0].first, chunk_key);
    std::shared_ptr<ChunkMetadata> chunk_metadata = chunkMetadataVector[0].second;
    ASSERT_EQ(chunk_metadata->numBytes, static_cast<size_t>(24));
    ASSERT_EQ(chunk_metadata->numElements, static_cast<size_t>(6));
    AbstractBuffer* file_buffer =
        file_mgr->getBuffer(chunk_key, chunk_metadata->numBytes);
    {
      SCOPED_TRACE("Buffer Append and Recovery - Compare #2");
      compareBuffersAndMetadata(source_buffer, file_buffer);
    }
    appendData(source_buffer, data_v2);
    appendData(file_buffer, data_v2);

    file_mgr->checkpoint();
    ASSERT_EQ(file_mgr->lastCheckpointedEpoch(), 3);
    {
      SCOPED_TRACE("Buffer Append and Recovery - Compare #3");
      compareBuffersAndMetadata(source_buffer, file_buffer);
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
    EXPECT_EQ(dm->getTableEpoch(file_mgr_key.first, file_mgr_key.second), std::size_t(1));
    AbstractBuffer* cpu_buffer =
        dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
    AbstractBuffer* file_buffer =
        dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::DISK_LEVEL);
    ASSERT_FALSE(cpu_buffer->isDirty());
    ASSERT_FALSE(cpu_buffer->isUpdated());
    ASSERT_FALSE(cpu_buffer->isAppended());
    ASSERT_FALSE(file_buffer->isDirty());
    ASSERT_FALSE(file_buffer->isUpdated());
    ASSERT_FALSE(file_buffer->isAppended());
    ASSERT_EQ(file_buffer->size(), static_cast<size_t>(4));
    {
      SCOPED_TRACE("Buffer Update and Recovery - Compare #1");
      compareBuffersAndMetadata(file_buffer, cpu_buffer);
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
    dm->checkpoint(file_mgr_key.first, file_mgr_key.second);
    EXPECT_EQ(dm->getTableEpoch(file_mgr_key.first, file_mgr_key.second), std::size_t(2));
    ASSERT_FALSE(file_buffer->isDirty());
    ASSERT_FALSE(file_buffer->isUpdated());
    ASSERT_FALSE(file_buffer->isAppended());
    cpu_buffer->unPin();  // Neccessary as we just have a raw_ptr, so there is no way to
                          // auto un-pin the pin that getBuffer sets, normally this is
                          // handled by the Chunk class wrapper
    dm->clearMemory(Data_Namespace::MemoryLevel::CPU_LEVEL);
    cpu_buffer = dm->getChunkBuffer(
        chunk_key,
        Data_Namespace::MemoryLevel::CPU_LEVEL,
        0,
        20);  // Dragons here: if we didn't unpin andy flush the data, the first value
              // will be 1, and not 2, as we only fetch the portion of data we don't have
              // from FileMgr (there's no DataMgr versioning currently, so for example,
              // for updates we just flush the in-memory buffers to get a clean start)
    ASSERT_FALSE(cpu_buffer->isDirty());
    ASSERT_FALSE(cpu_buffer->isUpdated());
    ASSERT_FALSE(cpu_buffer->isAppended());
    ASSERT_EQ(file_buffer->size(), static_cast<size_t>(20));
    {
      std::vector<int32_t> cpu_buffer_data(cpu_buffer->size() / sizeof(int32_t));
      cpu_buffer->read(reinterpret_cast<int8_t*>(cpu_buffer_data.data()),
                       cpu_buffer->size());
      ASSERT_EQ(cpu_buffer_data[0], 2);
      ASSERT_EQ(cpu_buffer_data[1], 3);
      ASSERT_EQ(cpu_buffer_data[2], 5);
      ASSERT_EQ(cpu_buffer_data[3], 7);
      ASSERT_EQ(cpu_buffer_data[4], 11);
      std::shared_ptr<ChunkMetadata> cpu_chunk_metadata =
          getMetadataForBuffer(cpu_buffer);
      ASSERT_EQ(cpu_chunk_metadata->numElements, static_cast<size_t>(5));
      ASSERT_EQ(cpu_chunk_metadata->numBytes, static_cast<size_t>(20));
      ASSERT_EQ(cpu_chunk_metadata->chunkStats.min.intval, 2);
      ASSERT_EQ(cpu_chunk_metadata->chunkStats.max.intval, 11);
      ASSERT_EQ(cpu_chunk_metadata->chunkStats.has_nulls, false);
    }
    {
      SCOPED_TRACE("Buffer Update and Recovery - Compare #2");
      compareBuffersAndMetadata(file_buffer, cpu_buffer);
    }
    // Now roll back to epoch 1
    cat->setTableEpoch(file_mgr_key.first, file_mgr_key.second, 1);
    file_buffer = dm->getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::DISK_LEVEL);
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
    const ChunkKey capped_chunk_key = setUpCappedRollbackTable(max_rollback_epochs);
    // Have one element already written to key -- epoch should be 2
    ASSERT_EQ(dm->getTableEpoch(capped_chunk_key[0], capped_chunk_key[1]),
              static_cast<size_t>(1));
    File_Namespace::FileMgr* file_mgr = dynamic_cast<File_Namespace::FileMgr*>(
        dm->getGlobalFileMgr()->getFileMgr(capped_chunk_key[0], capped_chunk_key[1]));
    // buffer inside loop
    for (int data_write = 1; data_write <= num_data_writes; ++data_write) {
      std::vector<int32_t> data;
      data.emplace_back(data_write);
      AbstractBuffer* file_buffer =
          dm->getChunkBuffer(capped_chunk_key, Data_Namespace::MemoryLevel::DISK_LEVEL);
      appendData(file_buffer, data);
      dm->checkpoint(capped_chunk_key[0], capped_chunk_key[1]);
      ASSERT_EQ(dm->getTableEpoch(capped_chunk_key[0], capped_chunk_key[1]),
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
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists test_table;");
  }

  void TearDown() override {
    sql("drop table test_table;");
    File_Namespace::FileMgr::setNumPagesPerDataFile(
        File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_DATA_FILE);
    File_Namespace::FileMgr::setNumPagesPerMetadataFile(
        File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_METADATA_FILE);
    DBHandlerTestFixture::TearDown();
  }

  File_Namespace::FileMgr* getFileMgr() {
    auto& data_mgr = getCatalog().getDataMgr();
    auto td = getCatalog().getMetadataForTable("test_table", false);
    return dynamic_cast<File_Namespace::FileMgr*>(data_mgr.getGlobalFileMgr()->getFileMgr(
        getCatalog().getDatabaseId(), td->tableId));
  }

  ChunkKey getChunkKey(const std::string& column_name) {
    auto td = getCatalog().getMetadataForTable("test_table", false);
    auto cd = getCatalog().getMetadataForColumn(td->tableId, column_name);
    return {getCatalog().getDatabaseId(), td->tableId, cd->columnId, 0};
  }

  void assertStorageStats(uint64_t metadata_file_count,
                          std::optional<uint64_t> free_metadata_page_count,
                          uint64_t data_file_count,
                          std::optional<uint64_t> free_data_page_count) {
    auto& catalog = getCatalog();
    auto global_file_mgr = catalog.getDataMgr().getGlobalFileMgr();
    auto td = catalog.getMetadataForTable("test_table", false);
    auto stats = global_file_mgr->getStorageStats(catalog.getDatabaseId(), td->tableId);
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

  void assertBufferValueAndMetadata(int32_t expected_value,
                                    const std::string& column_name) {
    auto chunk_key = getChunkKey(column_name);
    auto file_mgr = getFileMgr();
    auto buffer = file_mgr->getBuffer(chunk_key, sizeof(int32_t));
    int32_t value;
    buffer->read(reinterpret_cast<int8_t*>(&value), sizeof(int32_t));
    EXPECT_EQ(expected_value, value);
    assertChunkMetadata(buffer, expected_value);
  }

  AbstractBuffer* createBuffer(const std::string& column_name) {
    auto chunk_key = getChunkKey(column_name);
    auto buffer = getFileMgr()->createBuffer(chunk_key, DEFAULT_PAGE_SIZE, 0);
    auto cd = getCatalog().getMetadataForColumn(chunk_key[CHUNK_KEY_TABLE_IDX],
                                                chunk_key[CHUNK_KEY_COLUMN_IDX]);
    buffer->initEncoder(cd->columnType);
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

  void setMaxRollbackEpochs(int32_t max_rollback_epochs) {
    auto td = getCatalog().getMetadataForTable("test_table", false);
    auto& data_mgr = getCatalog().getDataMgr();
    File_Namespace::FileMgrParams params;
    params.max_rollback_epochs = 0;
    data_mgr.getGlobalFileMgr()->setFileMgrParams(
        getCatalog().getDatabaseId(), td->tableId, params);
  }

  void compactDataFiles() {
    auto td = getCatalog().getMetadataForTable("test_table", false);
    auto global_file_mgr = getCatalog().getDataMgr().getGlobalFileMgr();
    global_file_mgr->compactDataFiles(getCatalog().getDatabaseId(), td->tableId);
  }

  void deleteFileMgr() {
    auto td = getCatalog().getMetadataForTable("test_table", false);
    auto global_file_mgr = getCatalog().getDataMgr().getGlobalFileMgr();
    global_file_mgr->closeFileMgr(getCatalog().getDatabaseId(), td->tableId);
  }

  void deleteBuffer(const std::string& column_name) {
    auto chunk_key = getChunkKey(column_name);
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

  sql("create table test_table (i int);");
  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer = createBuffer("i");
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
  assertBufferValueAndMetadata(3, "i");

  // Setting max rollback epochs to 0 should free up
  // oldest 2 pages
  setMaxRollbackEpochs(0);
  assertStorageStats(1, 4095, 3, 2);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, "i");

  // Data compaction should result in removal of the
  // 2 files with free pages
  compactDataFiles();
  assertStorageStats(1, 4095, 1, 0);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, "i");
}

TEST_F(DataCompactionTest, MetadataFileCompaction) {
  // One page per file for the metadata file (data file default
  // configuration of 256 pages remains the same), so each write
  // creates a new metadata file.
  File_Namespace::FileMgr::setNumPagesPerMetadataFile(1);

  sql("create table test_table (i int);");
  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer = createBuffer("i");
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
  assertBufferValueAndMetadata(3, "i");

  // Setting max rollback epochs to 0 should free up
  // oldest 2 pages
  setMaxRollbackEpochs(0);
  assertStorageStats(3, 2, 1, 255);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, "i");

  // Data compaction should result in removal of the
  // 2 files with free pages
  compactDataFiles();
  assertStorageStats(1, 0, 1, 255);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, "i");
}

TEST_F(DataCompactionTest, DataAndMetadataFileCompaction) {
  // One page per file for the data and metadata files, so each
  // write creates a new data and metadata file.
  File_Namespace::FileMgr::setNumPagesPerDataFile(1);
  File_Namespace::FileMgr::setNumPagesPerMetadataFile(1);

  sql("create table test_table (i int);");
  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer = createBuffer("i");
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
  assertBufferValueAndMetadata(3, "i");

  // Setting max rollback epochs to 0 should free up
  // oldest 2 pages for both files
  setMaxRollbackEpochs(0);
  assertStorageStats(3, 2, 3, 2);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, "i");

  // Data compaction should result in removal of the
  // 4 files with free pages
  compactDataFiles();
  assertStorageStats(1, 0, 1, 0);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, "i");
}

TEST_F(DataCompactionTest, MultipleChunksPerFile) {
  File_Namespace::FileMgr::setNumPagesPerDataFile(4);

  sql("create table test_table (i int, i2 int);");
  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer_1 = createBuffer("i");
  auto buffer_2 = createBuffer("i2");

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
  assertBufferValueAndMetadata(1, "i");
  assertBufferValueAndMetadata(4, "i2");

  // Setting max rollback epochs to 0 should free up
  // oldest 3 pages for column "i2"
  setMaxRollbackEpochs(0);

  assertStorageStats(1, 4094, 2, 6);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(1, "i");
  assertBufferValueAndMetadata(4, "i2");

  // Data compaction should result in movement of a page from the
  // last data page file to the first data page file and deletion
  // of the last data page file
  compactDataFiles();
  assertStorageStats(1, 4094, 1, 2);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(1, "i");
  assertBufferValueAndMetadata(4, "i2");
}

TEST_F(DataCompactionTest, SourceFilePagesCopiedOverMultipleDestinationFiles) {
  File_Namespace::FileMgr::setNumPagesPerDataFile(4);

  sql("create table test_table (i int, i2 int);");
  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer_1 = createBuffer("i");
  auto buffer_2 = createBuffer("i2");

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
  assertBufferValueAndMetadata(9, "i");
  assertBufferValueAndMetadata(7, "i2");

  // Free the 7 pages used by buffer "i2" across the 4 files
  deleteBuffer("i2");
  assertStorageStats(1, 4087, 4, 7);

  // Verify first buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(9, "i");

  // Files 1, 2, and 3 each have 2 free pages (out of 4 total pages per file).
  // File 4 has 1 free page. Used pages in file 1 should be copied over to
  // files 4 and 3. File 1 should then be deleted.
  compactDataFiles();
  assertStorageStats(1, 4087, 3, 3);

  // Verify first buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(9, "i");
}

TEST_F(DataCompactionTest, SingleDataAndMetadataPages) {
  sql("create table test_table (i int);");
  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer = createBuffer("i");
  writeMultipleValues(buffer, 1, 3);
  assertStorageStats(1, 4093, 1, 253);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(3, "i");

  // Setting max rollback epochs to 0 should free up
  // oldest 2 pages
  setMaxRollbackEpochs(0);
  assertStorageStats(1, 4095, 1, 255);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, "i");

  // Data compaction should result in no changes to files
  compactDataFiles();
  assertStorageStats(1, 4095, 1, 255);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, "i");
}

TEST_F(DataCompactionTest, RecoveryFromCopyPageStatus) {
  // One page per file for the data file (metadata file default
  // configuration of 4096 pages remains the same), so each
  // write creates a new data file.
  File_Namespace::FileMgr::setNumPagesPerDataFile(1);

  sql("create table test_table (i int);");
  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer = createBuffer("i");
  writeMultipleValues(buffer, 1, 3);
  assertStorageStats(1, 4093, 3, 0);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(3, "i");

  // Setting max rollback epochs to 0 should free up
  // oldest 2 pages
  setMaxRollbackEpochs(0);
  assertStorageStats(1, 4095, 3, 2);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, "i");

  // Creating a "pending_data_compaction_0" status file and re-initializing
  // file mgr should result in (resumption of) data compaction and remove the
  // 2 files with free pages
  auto status_file_path =
      getFileMgr()->getFilePath(File_Namespace::FileMgr::COPY_PAGES_STATUS);
  deleteFileMgr();
  std::ofstream status_file{status_file_path, std::ios::out | std::ios::binary};
  status_file.close();

  getFileMgr();
  assertStorageStats(1, 4095, 1, 0);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(3, "i");
}

TEST_F(DataCompactionTest, RecoveryFromUpdatePageVisibiltyStatus) {
  File_Namespace::FileMgr::setNumPagesPerDataFile(4);

  sql("create table test_table (i int, i2 int);");
  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer_1 = createBuffer("i");
  auto buffer_2 = createBuffer("i2");

  writeMultipleValues(buffer_1, 1, 2);
  assertStorageStats(1, 4094, 1, 2);

  writeMultipleValues(buffer_2, 1, 3);
  assertStorageStats(1, 4091, 2, 3);

  // Verify buffer data and metadata are set as expected
  assertBufferValueAndMetadata(2, "i");
  assertBufferValueAndMetadata(3, "i2");

  // Setting max rollback epochs to 0 should free up oldest
  // page for chunk "i1" and oldest 2 pages for chunk "i2"
  setMaxRollbackEpochs(0);
  assertStorageStats(1, 4094, 2, 6);

  // Verify buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(2, "i");
  assertBufferValueAndMetadata(3, "i2");

  // Creating a "pending_data_compaction_1" status file and re-initializing
  // file mgr should result in (resumption of) data compaction,
  // movement of a page for the last data page file to the first data
  // page file, and deletion of the last data page file
  auto status_file_path =
      getFileMgr()->getFilePath(File_Namespace::FileMgr::COPY_PAGES_STATUS);
  std::ofstream status_file{status_file_path, std::ios::out | std::ios::binary};
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
  assertBufferValueAndMetadata(2, "i");
  assertBufferValueAndMetadata(3, "i2");
}

TEST_F(DataCompactionTest, RecoveryFromDeleteEmptyFileStatus) {
  File_Namespace::FileMgr::setNumPagesPerDataFile(4);

  sql("create table test_table (i int, i2 int);");
  // No files and free pages at the beginning
  assertStorageStats(0, {}, 0, {});

  auto buffer_1 = createBuffer("i");
  auto buffer_2 = createBuffer("i2");

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
  assertBufferValueAndMetadata(1, "i");
  assertBufferValueAndMetadata(4, "i2");

  // Delete chunks for "i2"
  deleteBuffer("i2");
  assertStorageStats(1, 4095, 2, 7);

  // Verify first buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(1, "i");

  // Creating a "pending_data_compaction_2" status file and re-initializing
  // file mgr should result in (resumption of) data compaction and deletion
  // of the last file that contains only free pages
  auto status_file_path =
      getFileMgr()->getFilePath(File_Namespace::FileMgr::DELETE_EMPTY_FILES_STATUS);
  deleteFileMgr();
  std::ofstream status_file{status_file_path, std::ios::out | std::ios::binary};
  status_file.close();

  getFileMgr();
  assertStorageStats(1, 4095, 1, 3);

  // Verify first buffer data and metadata are still set as expected
  assertBufferValueAndMetadata(1, "i");
}

class MaxRollbackEpochTest : public FileMgrTest {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists test_table;");
    sql("create table test_table (i int);");
  }

  void TearDown() override {
    sql("drop table test_table;");
    DBHandlerTestFixture::TearDown();
  }

  File_Namespace::FileMgr* getFileMgr() {
    auto& data_mgr = getCatalog().getDataMgr();
    auto td = getCatalog().getMetadataForTable("test_table");
    return dynamic_cast<File_Namespace::FileMgr*>(data_mgr.getGlobalFileMgr()->getFileMgr(
        getCatalog().getDatabaseId(), td->tableId));
  }

  ChunkKey getChunkKey() {
    auto td = getCatalog().getMetadataForTable("test_table");
    auto cd = getCatalog().getMetadataForColumn(td->tableId, "i");
    return {getCatalog().getDatabaseId(), td->tableId, cd->columnId, 0};
  }

  AbstractBuffer* createBuffer(size_t num_entries_per_page) {
    auto chunk_key = getChunkKey();
    constexpr size_t reserved_header_size{32};
    auto buffer = getFileMgr()->createBuffer(
        chunk_key, reserved_header_size + (num_entries_per_page * sizeof(int32_t)), 0);
    auto cd = getCatalog().getMetadataForColumn(chunk_key[CHUNK_KEY_TABLE_IDX],
                                                chunk_key[CHUNK_KEY_COLUMN_IDX]);
    buffer->initEncoder(cd->columnType);
    return buffer;
  }

  void setMaxRollbackEpochs(int32_t max_rollback_epochs) {
    auto td = getCatalog().getMetadataForTable("test_table");
    auto& data_mgr = getCatalog().getDataMgr();
    File_Namespace::FileMgrParams params;
    params.max_rollback_epochs = max_rollback_epochs;
    data_mgr.getGlobalFileMgr()->setFileMgrParams(
        getCatalog().getDatabaseId(), td->tableId, params);
  }

  void updateData(std::vector<int32_t>& values) {
    auto& data_mgr = getCatalog().getDataMgr();
    auto chunk_key = getChunkKey();
    auto buffer_size = values.size() * sizeof(int32_t);
    AbstractBuffer* buffer =
        data_mgr.getChunkBuffer(chunk_key, Data_Namespace::MemoryLevel::CPU_LEVEL);
    buffer->reserve(buffer_size);
    memcpy(buffer->getMemoryPtr(), reinterpret_cast<int8_t*>(values.data()), buffer_size);
    buffer->setSize(buffer_size);
    buffer->setUpdated();
    getFileMgr()->putBuffer(chunk_key, buffer, buffer_size);
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

constexpr char file_mgr_path[] = "./FileMgrTest";
namespace bf = boost::filesystem;

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
  ::registerArrowForeignStorage(fsi);
  ::registerArrowCsvForeignStorage(fsi);
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
  ::registerArrowForeignStorage(fsi);
  ::registerArrowCsvForeignStorage(fsi);
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
  ::registerArrowForeignStorage(fsi);
  ::registerArrowCsvForeignStorage(fsi);
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
