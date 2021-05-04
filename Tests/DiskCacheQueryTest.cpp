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
 * @file DiskCacheQueryTest.cpp
 * @brief Test suite for queries on Disk Cache
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "DBHandlerTestHelpers.h"
#include "DataMgr/ForeignStorage/ForeignStorageCache.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

std::string test_binary_file_path;

namespace bf = boost::filesystem;
using path = bf::path;

static const std::string default_table_name = "test_table";

class TableTest : public DBHandlerTestFixture {
 protected:
  inline static Catalog_Namespace::Catalog* cat_;
  inline static foreign_storage::ForeignStorageCache* cache_;
  inline static std::string cache_path_ = to_string(BASE_PATH) + "/omnisci_disk_cache/";
  inline static PersistentStorageMgr* psm_;

  static void SetUpTestSuite() {
    DBHandlerTestFixture::createDBHandler(DiskCacheLevel::all);
    cat_ = &getCatalog();
    ASSERT_NE(cat_, nullptr);
    psm_ = cat_->getDataMgr().getPersistentStorageMgr();
    ASSERT_NE(psm_, nullptr);
    cache_ = psm_->getDiskCache();
    ASSERT_NE(cache_, nullptr);
  }

  static void TearDownTestSuite() {}

  void SetUp() override {
    sqlDropTable();
    cache_->clear();
  }

  void TearDown() override {
    sqlDropTable();
    cache_->clear();
    DBHandlerTestFixture::TearDown();
  }

  static void sqlCreateTable(const std::string& columns,
                             const std::map<std::string, std::string> options = {},
                             const std::string& table_name = default_table_name) {
    std::string query{"CREATE TABLE " + table_name};
    query += " " + columns + ";";
    sql(query);
  }

  static void sqlDropTable(const std::string& table_name = default_table_name) {
    std::string query{"DROP TABLE IF EXISTS " + table_name};
    sql(query);
  }

  static ChunkKey getChunkKeyFromTable(const Catalog_Namespace::Catalog& cat,
                                       const std::string& table_name,
                                       const ChunkKey& key_suffix) {
    const TableDescriptor* fd = cat.getMetadataForTable(table_name);
    ChunkKey key{cat.getCurrentDB().dbId, fd->tableId};
    for (auto i : key_suffix) {
      key.push_back(i);
    }
    return key;
  }

  static void resetPersistentStorageMgr(DiskCacheLevel cache_level) {
    for (auto table_it : cat_->getAllTableMetadata()) {
      cat_->removeFragmenterForTable(table_it->tableId);
    }
    cat_->getDataMgr().resetPersistentStorage(
        {psm_->getDiskCacheConfig().path, cache_level}, 0, getSystemParameters());
    psm_ = cat_->getDataMgr().getPersistentStorageMgr();
    cache_ = psm_->getDiskCache();
  }

  static void resetStorageManagerAndClearTableMemory(
      const ChunkKey& table_key,
      DiskCacheLevel cache_level = DiskCacheLevel::all) {
    // Reset cache and clear memory representations.
    resetPersistentStorageMgr(cache_level);
    cat_->getDataMgr().deleteChunksWithPrefix(table_key, MemoryLevel::CPU_LEVEL);
    cat_->getDataMgr().deleteChunksWithPrefix(table_key, MemoryLevel::GPU_LEVEL);
  }
};

TEST_F(TableTest, DISABLED_InsertWithCache) {
  sqlCreateTable("(i INTEGER) WITH (fragment_size = 1)");
  const ChunkKey key1 = getChunkKeyFromTable(*cat_, default_table_name, {1, 0});
  const ChunkKey key2 = getChunkKeyFromTable(*cat_, default_table_name, {1, 1});
  ASSERT_EQ(cache_->getCachedChunkIfExists(key1), nullptr);
  ASSERT_EQ(cache_->getCachedChunkIfExists(key2), nullptr);

  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";", {});
  sql("INSERT INTO " + default_table_name + " VALUES(1);");
  ASSERT_NE(cache_->getCachedChunkIfExists(key1), nullptr);
  ASSERT_EQ(cache_->getCachedChunkIfExists(key2), nullptr);

  sql("INSERT INTO " + default_table_name + " VALUES(2);");
  ASSERT_NE(cache_->getCachedChunkIfExists(key1), nullptr);
  ASSERT_NE(cache_->getCachedChunkIfExists(key2), nullptr);

  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";", {{i(1)}, {i(2)}});
}

TEST_F(TableTest, DeleteWithCache) {
  sqlCreateTable("(i INTEGER) WITH (fragment_size = 1)");
  sql("INSERT INTO " + default_table_name + " VALUES(1);");
  sql("INSERT INTO " + default_table_name + " VALUES(2);");
  sql("INSERT INTO " + default_table_name + " VALUES(3);");
  sql("INSERT INTO " + default_table_name + " VALUES(4);");
  sql("DELETE FROM " + default_table_name + " WHERE i = 3;");
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";",
                      {{i(1)}, {i(2)}, {i(4)}});
  sql("DELETE FROM " + default_table_name + " WHERE i = 4;");
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";", {{i(1)}, {i(2)}});
}

TEST_F(TableTest, UpdateWithCache) {
  sqlCreateTable("(i INTEGER) WITH (fragment_size = 1)");
  sql("INSERT INTO " + default_table_name + " VALUES(1);");
  sql("INSERT INTO " + default_table_name + " VALUES(2);");
  sql("INSERT INTO " + default_table_name + " VALUES(3);");
  sql("INSERT INTO " + default_table_name + " VALUES(4);");
  sql("UPDATE " + default_table_name + " SET i = 0 WHERE i >= 2;");
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";",
                      {{i(1)}, {i(0)}, {i(0)}, {i(0)}});
}

TEST_F(TableTest, RecoverCache_All) {
  sqlCreateTable("(i INTEGER)");
  sql("INSERT INTO " + default_table_name + " VALUES(1);");
  const ChunkKey table_key{cat_->getCurrentDB().dbId,
                           cat_->getMetadataForTable(default_table_name)->tableId};
  const ChunkKey key1{table_key[0], table_key[1], 1, 0};
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";", {{i(1)}});
  resetStorageManagerAndClearTableMemory(table_key, DiskCacheLevel::all);
  ASSERT_EQ(cache_->getCachedChunkIfExists(key1), nullptr);
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";", {{i(1)}});
  ASSERT_NE(cache_->getCachedChunkIfExists(key1), nullptr);
}

TEST_F(TableTest, RecoverCache_NonFSI) {
  sqlCreateTable("(i INTEGER)");
  sql("INSERT INTO " + default_table_name + " VALUES(1);");
  const ChunkKey table_key{cat_->getCurrentDB().dbId,
                           cat_->getMetadataForTable(default_table_name)->tableId};
  const ChunkKey key1{table_key[0], table_key[1], 1, 0};
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";", {{i(1)}});
  resetStorageManagerAndClearTableMemory(table_key, DiskCacheLevel::non_fsi);
  ASSERT_EQ(cache_->getCachedChunkIfExists(key1), nullptr);
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";", {{i(1)}});
  ASSERT_NE(cache_->getCachedChunkIfExists(key1), nullptr);
  sqlDropTable();
  resetStorageManagerAndClearTableMemory(table_key, DiskCacheLevel::all);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  g_enable_fsi = true;

  // get dirname of test binary
  test_binary_file_path = bf::canonical(argv[0]).parent_path().string();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  g_enable_fsi = false;
  return err;
}
