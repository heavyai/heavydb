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
    int8_t left_array[num_bytes];
    int8_t right_array[num_bytes];
    left_buffer->read(left_array, num_bytes);
    right_buffer->read(right_array, num_bytes);
    ASSERT_EQ(std::memcmp(left_array, right_array, num_bytes), 0);
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
      ASSERT_EQ(file_chunk_metadata->chunkStats.min.intval,
                1);  // We don't currently narrow the metadata on a full rewrite
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
      ASSERT_EQ(cpu_chunk_metadata->chunkStats.min.intval,
                1);  // We don't currently narrow the metadata on a full rewrite
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
