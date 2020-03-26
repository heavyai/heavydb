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
 * @file CreateAndDropTableDdlTest.cpp
 * @brief Test suite for CREATE and DROP DDL commands for tables and foreign tables
 */

#include <gtest/gtest.h>

#include "Catalog/ForeignTable.h"
#include "Catalog/TableDescriptor.h"
#include "Fragmenter/FragmentDefaultValues.h"
#include "MapDHandlerTestHelpers.h"
#include "TestHelpers.h"
#include "Utils/DdlUtils.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;

namespace {
struct ColumnAttributes {
  std::string column_name;
  bool not_null{false};
  int size{-1};
  SQLTypes type;
  SQLTypes sub_type{SQLTypes::kNULLT};
  int precision{0};
  int scale{0};
  EncodingType encoding_type{EncodingType::kENCODING_NONE};
  int encoding_size{0};
};
}  // namespace

class CreateAndDropTableDdlTest : public MapDHandlerTestFixture {
 protected:
  void SetUp() override {
    g_enable_fsi = true;
    MapDHandlerTestFixture::SetUp();
  }

  std::string getCreateTableQuery(const ddl_utils::TableType table_type,
                                  const std::string& table_name,
                                  const std::string& columns,
                                  bool if_not_exists = false) {
    std::string query{"CREATE "};
    if (table_type == ddl_utils::TableType::FOREIGN_TABLE) {
      query += "FOREIGN TABLE ";
    } else {
      query += "TABLE ";
    }
    if (if_not_exists) {
      query += "IF NOT EXISTS ";
    }
    query += table_name + columns;

    if (table_type == ddl_utils::TableType::FOREIGN_TABLE) {
      query += " SERVER omnisci_local_csv WITH (file_path = 'test_file.csv')";
    }
    query += ";";
    return query;
  }

  std::string getDropTableQuery(const ddl_utils::TableType table_type,
                                const std::string& table_name,
                                bool if_exists = false) {
    std::string query{"DROP "};
    if (table_type == ddl_utils::TableType::FOREIGN_TABLE) {
      query += "FOREIGN TABLE ";
    } else {
      query += "TABLE ";
    }

    if (if_exists) {
      query += "IF EXISTS ";
    }
    query += table_name + ";";
    return query;
  }

  void createTestUser() {
    sql("CREATE USER test_user (password = 'test_pass');");
    sql("GRANT ACCESS ON DATABASE omnisci TO test_user;");
  }

  void dropTestUser() {
    loginAdmin();
    try {
      sql("DROP USER test_user;");
    } catch (const std::exception& e) {
      // Swallow and log exceptions that may occur, since there is no "IF EXISTS" option.
      LOG(WARNING) << e.what();
    }
  }
};

class CreateTableTest : public CreateAndDropTableDdlTest,
                        public testing::WithParamInterface<ddl_utils::TableType> {
 protected:
  void SetUp() override {
    CreateAndDropTableDdlTest::SetUp();
    sql(getDropTableQuery(GetParam(), "test_table", true));
    dropTestUser();
  }

  void TearDown() override {
    g_enable_fsi = true;
    sql(getDropTableQuery(GetParam(), "test_table", true));
    dropTestUser();
    CreateAndDropTableDdlTest::TearDown();
  }

  /**
   *
   * @param td - table details returned from the catalog
   * @param table_type - table type
   * @param table_name - expected table name
   * @param column_count - expected number of columns in the table
   * @param user_id - id of user who owns the table. Default value of 0 (admin user id) is
   * used
   */
  void assertTableDetails(const TableDescriptor* td,
                          const ddl_utils::TableType table_type,
                          const std::string& table_name,
                          const int column_count,
                          const int user_id = 0) {
    EXPECT_EQ(table_name, td->tableName);
    EXPECT_EQ(Fragmenter_Namespace::FragmenterType::INSERT_ORDER, td->fragType);
    EXPECT_EQ(DEFAULT_FRAGMENT_ROWS, td->maxFragRows);
    EXPECT_EQ(DEFAULT_MAX_CHUNK_SIZE, td->maxChunkSize);
    EXPECT_EQ(DEFAULT_PAGE_SIZE, td->fragPageSize);
    EXPECT_EQ(DEFAULT_MAX_ROWS, td->maxRows);
    EXPECT_EQ(user_id, td->userId);
    EXPECT_EQ(Data_Namespace::MemoryLevel::DISK_LEVEL, td->persistenceLevel);
    EXPECT_FALSE(td->isView);
    EXPECT_EQ(0, td->nShards);
    EXPECT_EQ(0, td->shardedColumnId);
    EXPECT_EQ("[]", td->keyMetainfo);
    EXPECT_EQ("", td->fragments);
    EXPECT_EQ("", td->partitions);

    if (table_type == ddl_utils::TableType::FOREIGN_TABLE) {
      auto foreign_table = dynamic_cast<const foreign_storage::ForeignTable*>(td);
      ASSERT_NE(nullptr, foreign_table);
      EXPECT_EQ(column_count + 1, td->nColumns);  // +1 for rowid column
      EXPECT_FALSE(td->hasDeletedCol);
      EXPECT_EQ(StorageType::FOREIGN_TABLE, foreign_table->storageType);
      ASSERT_TRUE(foreign_table->options.find("FILE_PATH") !=
                  foreign_table->options.end());
      EXPECT_EQ("test_file.csv", foreign_table->options.find("FILE_PATH")->second);
      EXPECT_EQ(size_t(1), foreign_table->options.size());
      EXPECT_EQ("omnisci_local_csv", foreign_table->foreign_server->name);
    } else {
      EXPECT_EQ(column_count + 2, td->nColumns);  // +2 for rowid and $deleted$ columns
      EXPECT_TRUE(td->hasDeletedCol);
      EXPECT_TRUE(td->storageType.empty());
    }
  }

  void assertColumnDetails(const ColumnAttributes expected,
                           const ColumnDescriptor* column) {
    EXPECT_EQ(expected.column_name, column->columnName);
    EXPECT_TRUE(column->sourceName.empty());
    EXPECT_TRUE(column->chunks.empty());
    EXPECT_FALSE(column->isSystemCol);
    EXPECT_FALSE(column->isVirtualCol);
    EXPECT_TRUE(column->virtualExpr.empty());
    EXPECT_FALSE(column->isDeletedCol);

    auto& type_info = column->columnType;
    EXPECT_EQ(expected.not_null, type_info.get_notnull());
    EXPECT_EQ(expected.encoding_type, type_info.get_compression());
    EXPECT_EQ(expected.precision, type_info.get_dimension());
    EXPECT_EQ(expected.precision, type_info.get_precision());
    EXPECT_EQ(expected.precision, type_info.get_input_srid());
    EXPECT_EQ(expected.scale, type_info.get_scale());
    EXPECT_EQ(expected.scale, type_info.get_output_srid());
    EXPECT_EQ(expected.size, type_info.get_size());
    EXPECT_EQ(expected.type, type_info.get_type());
    EXPECT_EQ(expected.sub_type, type_info.get_subtype());

    // Comp param contains dictionary id for encoded strings
    if (type_info.get_compression() != kENCODING_DICT) {
      EXPECT_EQ(expected.encoding_size, type_info.get_comp_param());
    }
  }
};

TEST_P(CreateTableTest, BooleanAndNumberTypes) {
  std::string query =
      getCreateTableQuery(GetParam(),
                          "test_table",
                          "(bl BOOLEAN, bint BIGINT, bint8 BIGINT ENCODING FIXED(8), "
                          "bint16 BIGINT ENCODING FIXED(16), "
                          "bint32 BIGINT ENCODING FIXED(32), dc DECIMAL(5, 2), dc1 "
                          "DECIMAL(3), db DOUBLE, fl FLOAT, i INTEGER, "
                          "i8 INTEGER ENCODING FIXED(8), i16 INTEGER ENCODING FIXED(16), "
                          "si SMALLINT, si8 SMALLINT ENCODING FIXED(8), "
                          "ti TINYINT)");
  sql(query);

  auto& catalog = getCatalog();
  auto table = catalog.getMetadataForTable("test_table", false);
  assertTableDetails(table, GetParam(), "test_table", 15);

  auto columns = catalog.getAllColumnMetadataForTable(table->tableId, true, true, true);
  auto it = columns.begin();
  auto column = *it;
  ColumnAttributes expected_attributes{};
  expected_attributes.column_name = "bl";
  expected_attributes.size = 1;
  expected_attributes.type = kBOOLEAN;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "bint";
  expected_attributes.size = 8;
  expected_attributes.type = SQLTypes::kBIGINT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "bint8";
  expected_attributes.size = 1;
  expected_attributes.type = kBIGINT;
  expected_attributes.encoding_type = kENCODING_FIXED;
  expected_attributes.encoding_size = 8;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "bint16";
  expected_attributes.size = 2;
  expected_attributes.type = kBIGINT;
  expected_attributes.encoding_type = kENCODING_FIXED;
  expected_attributes.encoding_size = 16;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "bint32";
  expected_attributes.size = 4;
  expected_attributes.type = kBIGINT;
  expected_attributes.encoding_type = kENCODING_FIXED;
  expected_attributes.encoding_size = 32;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "dc";
  expected_attributes.size = 4;
  expected_attributes.type = kDECIMAL;
  expected_attributes.encoding_type = kENCODING_FIXED;
  expected_attributes.encoding_size = 32;
  expected_attributes.precision = 5;
  expected_attributes.scale = 2;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "dc1";
  expected_attributes.size = 2;
  expected_attributes.type = kDECIMAL;
  expected_attributes.encoding_type = kENCODING_FIXED;
  expected_attributes.encoding_size = 16;
  expected_attributes.precision = 3;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "db";
  expected_attributes.size = 8;
  expected_attributes.type = kDOUBLE;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "fl";
  expected_attributes.size = 4;
  expected_attributes.type = kFLOAT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "i";
  expected_attributes.size = 4;
  expected_attributes.type = kINT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "i8";
  expected_attributes.size = 1;
  expected_attributes.type = kINT;
  expected_attributes.encoding_type = kENCODING_FIXED;
  expected_attributes.encoding_size = 8;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "i16";
  expected_attributes.size = 2;
  expected_attributes.type = kINT;
  expected_attributes.encoding_type = kENCODING_FIXED;
  expected_attributes.encoding_size = 16;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "si";
  expected_attributes.size = 2;
  expected_attributes.type = kSMALLINT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "si8";
  expected_attributes.size = 1;
  expected_attributes.type = kSMALLINT;
  expected_attributes.encoding_type = kENCODING_FIXED;
  expected_attributes.encoding_size = 8;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "ti";
  expected_attributes.size = 1;
  expected_attributes.type = kTINYINT;
  assertColumnDetails(expected_attributes, column);
}

TEST_P(CreateTableTest, DateAndTimestampTypes) {
  std::string query = getCreateTableQuery(
      GetParam(),
      "test_table",
      "(dt DATE, dt16 DATE ENCODING FIXED(16), dt16_days DATE ENCODING DAYS(16), t TIME, "
      "t32 TIME ENCODING FIXED(32), tp0 TIMESTAMP(0), tp3 TIMESTAMP(3), tp6 "
      "TIMESTAMP(6), "
      "tp9 TIMESTAMP(9), tp32 TIMESTAMP ENCODING FIXED(32))");
  sql(query);

  auto& catalog = getCatalog();
  auto table = catalog.getMetadataForTable("test_table", false);
  assertTableDetails(table, GetParam(), "test_table", 10);

  auto columns = catalog.getAllColumnMetadataForTable(table->tableId, true, true, true);
  auto it = columns.begin();
  auto column = *it;
  ColumnAttributes expected_attributes{};
  expected_attributes.column_name = "dt";
  expected_attributes.size = 4;
  expected_attributes.type = kDATE;
  expected_attributes.encoding_type = kENCODING_DATE_IN_DAYS;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "dt16";
  expected_attributes.size = 2;
  expected_attributes.type = kDATE;
  expected_attributes.encoding_type = kENCODING_DATE_IN_DAYS;
  expected_attributes.encoding_size = 16;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "dt16_days";
  expected_attributes.size = 2;
  expected_attributes.type = kDATE;
  expected_attributes.encoding_type = kENCODING_DATE_IN_DAYS;
  expected_attributes.encoding_size = 16;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "t";
  expected_attributes.size = 8;
  expected_attributes.type = kTIME;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "t32";
  expected_attributes.size = 4;
  expected_attributes.type = kTIME;
  expected_attributes.encoding_type = kENCODING_FIXED;
  expected_attributes.encoding_size = 32;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "tp0";
  expected_attributes.size = 8;
  expected_attributes.type = kTIMESTAMP;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "tp3";
  expected_attributes.size = 8;
  expected_attributes.type = kTIMESTAMP;
  expected_attributes.precision = 3;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "tp6";
  expected_attributes.size = 8;
  expected_attributes.type = kTIMESTAMP;
  expected_attributes.precision = 6;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "tp9";
  expected_attributes.size = 8;
  expected_attributes.type = kTIMESTAMP;
  expected_attributes.precision = 9;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "tp32";
  expected_attributes.size = 4;
  expected_attributes.type = kTIMESTAMP;
  expected_attributes.encoding_type = kENCODING_FIXED;
  expected_attributes.encoding_size = 32;
  assertColumnDetails(expected_attributes, column);
}

TEST_P(CreateTableTest, TextTypes) {
  std::string query = getCreateTableQuery(
      GetParam(),
      "test_table",
      "(t TEXT ENCODING DICT, t8 TEXT ENCODING DICT(8), t16 TEXT ENCODING DICT(16), "
      "t32 TEXT ENCODING DICT(32), t_non_encoded TEXT ENCODING NONE)");
  sql(query);

  auto& catalog = getCatalog();
  auto table = catalog.getMetadataForTable("test_table", false);
  assertTableDetails(table, GetParam(), "test_table", 5);

  auto columns = catalog.getAllColumnMetadataForTable(table->tableId, true, true, true);
  auto it = columns.begin();
  auto column = *it;
  ColumnAttributes expected_attributes{};
  expected_attributes.column_name = "t";
  expected_attributes.size = 4;
  expected_attributes.type = kTEXT;
  expected_attributes.encoding_type = kENCODING_DICT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "t8";
  expected_attributes.size = 1;
  expected_attributes.type = kTEXT;
  expected_attributes.encoding_type = kENCODING_DICT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "t16";
  expected_attributes.size = 2;
  expected_attributes.type = kTEXT;
  expected_attributes.encoding_type = kENCODING_DICT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "t32";
  expected_attributes.size = 4;
  expected_attributes.type = kTEXT;
  expected_attributes.encoding_type = kENCODING_DICT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "t_non_encoded";
  expected_attributes.type = kTEXT;
  assertColumnDetails(expected_attributes, column);
}

TEST_P(CreateTableTest, GeoTypes) {
  std::string query = getCreateTableQuery(
      GetParam(),
      "test_table",
      "(ls LINESTRING, mpoly MULTIPOLYGON, p POINT, poly POLYGON, p1 GEOMETRY(POINT), "
      "p2 GEOMETRY(POINT, 4326), p3 GEOMETRY(POINT, 4326) ENCODING NONE, p4 "
      "GEOMETRY(POINT, 900913), "
      "ls1 GEOMETRY(LINESTRING, 4326) ENCODING COMPRESSED(32), ls2 GEOMETRY(LINESTRING, "
      "4326) ENCODING NONE, "
      "poly1 GEOMETRY(POLYGON, 4326) ENCODING COMPRESSED(32), mpoly1 "
      "GEOMETRY(MULTIPOLYGON, 4326))");
  sql(query);

  auto& catalog = getCatalog();
  auto table = catalog.getMetadataForTable("test_table", false);
  /**
   * LINESTRING adds 2 additional columns, MULTIPOLYGON adds 5 additional columns,
   * POLYGON adds 1 additional column, and POLYGON adds 4 additional columns when
   * expanded.
   */
  assertTableDetails(table, GetParam(), "test_table", 41);

  auto columns = catalog.getAllColumnMetadataForTable(table->tableId, true, true, true);
  auto it = columns.begin();
  auto column = *it;
  ColumnAttributes expected_attributes{};
  expected_attributes.column_name = "ls";
  expected_attributes.type = kLINESTRING;
  expected_attributes.sub_type = kGEOMETRY;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 3);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "mpoly";
  expected_attributes.type = kMULTIPOLYGON;
  expected_attributes.sub_type = kGEOMETRY;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 6);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "p";
  expected_attributes.type = kPOINT;
  expected_attributes.sub_type = kGEOMETRY;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 2);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "poly";
  expected_attributes.type = kPOLYGON;
  expected_attributes.sub_type = kGEOMETRY;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 5);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "p1";
  expected_attributes.type = kPOINT;
  expected_attributes.sub_type = kGEOMETRY;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 2);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "p2";
  expected_attributes.type = kPOINT;
  expected_attributes.sub_type = kGEOMETRY;
  expected_attributes.precision = 4326;
  expected_attributes.scale = 4326;
  expected_attributes.encoding_type = kENCODING_GEOINT;
  expected_attributes.encoding_size = 32;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 2);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "p3";
  expected_attributes.type = kPOINT;
  expected_attributes.sub_type = kGEOMETRY;
  expected_attributes.precision = 4326;
  expected_attributes.scale = 4326;
  expected_attributes.encoding_type = kENCODING_NONE;
  expected_attributes.encoding_size = 64;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 2);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "p4";
  expected_attributes.type = kPOINT;
  expected_attributes.sub_type = kGEOMETRY;
  expected_attributes.precision = 900913;
  expected_attributes.scale = 900913;
  expected_attributes.encoding_type = kENCODING_NONE;
  expected_attributes.encoding_size = 0;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 2);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "ls1";
  expected_attributes.type = kLINESTRING;
  expected_attributes.sub_type = kGEOMETRY;
  expected_attributes.precision = 4326;
  expected_attributes.scale = 4326;
  expected_attributes.encoding_type = kENCODING_GEOINT;
  expected_attributes.encoding_size = 32;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 3);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "ls2";
  expected_attributes.type = kLINESTRING;
  expected_attributes.sub_type = kGEOMETRY;
  expected_attributes.precision = 4326;
  expected_attributes.scale = 4326;
  expected_attributes.encoding_type = kENCODING_NONE;
  expected_attributes.encoding_size = 64;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 3);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "poly1";
  expected_attributes.type = kPOLYGON;
  expected_attributes.sub_type = kGEOMETRY;
  expected_attributes.precision = 4326;
  expected_attributes.scale = 4326;
  expected_attributes.encoding_type = kENCODING_GEOINT;
  expected_attributes.encoding_size = 32;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 5);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "mpoly1";
  expected_attributes.type = kMULTIPOLYGON;
  expected_attributes.sub_type = kGEOMETRY;
  expected_attributes.precision = 4326;
  expected_attributes.scale = 4326;
  expected_attributes.encoding_type = kENCODING_GEOINT;
  expected_attributes.encoding_size = 32;
  assertColumnDetails(expected_attributes, column);
}

TEST_P(CreateTableTest, ArrayTypes) {
  std::string query =
      getCreateTableQuery(GetParam(),
                          "test_table",
                          "(t TINYINT[], t2 TINYINT[1], i INTEGER[], i2 INTEGER[1], bint "
                          "BIGINT[], bint2 BIGINT[1], "
                          "txt TEXT[] ENCODING DICT(32), txt2 TEXT[1] ENCODING DICT(32), "
                          "f FLOAT[], f2 FLOAT[1], "
                          "d DOUBLE[], d2 DOUBLE[1], dc DECIMAL(18,6)[], dc2 "
                          "DECIMAL(18,6)[1], b BOOLEAN[], b2 BOOLEAN[1],"
                          "dt DATE[], dt2 DATE[1], tm TIME[], tm2 TIME[1], tp "
                          "TIMESTAMP[], tp2 TIMESTAMP[1], p POINT[],"
                          "ls LINESTRING[], poly POLYGON[], mpoly MULTIPOLYGON[])");
  sql(query);

  auto& catalog = getCatalog();
  auto table = catalog.getMetadataForTable("test_table", false);
  assertTableDetails(table, GetParam(), "test_table", 26);

  auto columns = catalog.getAllColumnMetadataForTable(table->tableId, true, true, true);
  auto it = columns.begin();
  auto column = *it;
  ColumnAttributes expected_attributes{};
  expected_attributes.column_name = "t";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kTINYINT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "t2";
  expected_attributes.size = 1;
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kTINYINT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "i";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kINT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "i2";
  expected_attributes.size = 4;
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kINT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "bint";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kBIGINT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "bint2";
  expected_attributes.size = 8;
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kBIGINT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "txt";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kTEXT;
  expected_attributes.encoding_type = kENCODING_DICT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "txt2";
  expected_attributes.size = 4;
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kTEXT;
  expected_attributes.encoding_type = kENCODING_DICT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "f";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kFLOAT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "f2";
  expected_attributes.size = 4;
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kFLOAT;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "d";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kDOUBLE;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "d2";
  expected_attributes.size = 8;
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kDOUBLE;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "dc";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kDECIMAL;
  expected_attributes.precision = 18;
  expected_attributes.scale = 6;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "dc2";
  expected_attributes.size = 8;
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kDECIMAL;
  expected_attributes.precision = 18;
  expected_attributes.scale = 6;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "b";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kBOOLEAN;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "b2";
  expected_attributes.size = 1;
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kBOOLEAN;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "dt";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kDATE;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "dt2";
  expected_attributes.size = 8;
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kDATE;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "tm";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kTIME;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "tm2";
  expected_attributes.size = 8;
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kTIME;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "tp";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kTIMESTAMP;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "tp2";
  expected_attributes.size = 8;
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kTIMESTAMP;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "p";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kGEOMETRY;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "ls";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kGEOMETRY;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "poly";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kGEOMETRY;
  assertColumnDetails(expected_attributes, column);

  std::advance(it, 1);
  column = *it;
  expected_attributes = {};
  expected_attributes.column_name = "mpoly";
  expected_attributes.type = kARRAY;
  expected_attributes.sub_type = kGEOMETRY;
  assertColumnDetails(expected_attributes, column);
}

TEST_P(CreateTableTest, FixedEncodingForNonNumberOrTimeType) {
  std::string query =
      getCreateTableQuery(GetParam(), "test_table", "(col1 POINT ENCODING FIXED(8))");
  queryAndAssertException(
      query,
      "Exception: col1: Fixed encoding is only supported for integer or time columns.");
}

TEST_P(CreateTableTest, DictEncodingNonTextType) {
  std::string query =
      getCreateTableQuery(GetParam(), "test_table", "(col1 INTEGER ENCODING DICT)");
  queryAndAssertException(query,
                          "Exception: col1: Dictionary encoding is only supported on "
                          "string or string array columns.");
}

TEST_P(CreateTableTest, CompressedEncodingNonWGS84GeoType) {
  std::string query = getCreateTableQuery(
      GetParam(), "test_table", "(col1 GEOMETRY(POINT, 900913) ENCODING COMPRESSED(32))");
  queryAndAssertException(
      query,
      "Exception: col1: COMPRESSED encoding is only supported on WGS84 geo columns.");
}

TEST_P(CreateTableTest, CompressedEncodingNon32Bit) {
  std::string query = getCreateTableQuery(
      GetParam(), "test_table", "(col1 GEOMETRY(POINT, 4326) ENCODING COMPRESSED(16))");
  queryAndAssertException(
      query, "Exception: col1: only 32-bit COMPRESSED geo encoding is supported");
}

TEST_P(CreateTableTest, DaysEncodingNonDateType) {  // Param for DECIMAL and NUMERIC
  std::string query =
      getCreateTableQuery(GetParam(), "test_table", "(col1 TIME ENCODING DAYS(16))");
  queryAndAssertException(
      query, "Exception: col1: Days encoding is only supported for DATE columns.");
}

TEST_P(CreateTableTest, NonEncodedDictArray) {
  std::string query =
      getCreateTableQuery(GetParam(), "test_table", "(col1 TEXT[] ENCODING NONE)");
  queryAndAssertException(query,
                          "Exception: col1: Array of strings must be dictionary encoded. "
                          "Specify ENCODING DICT");
}

TEST_P(CreateTableTest, FixedLengthArrayOfVarLengthType) {
  std::string query =
      getCreateTableQuery(GetParam(), "test_table", "(col1 LINESTRING[5])");
  queryAndAssertException(query, "Exception: col1: Unexpected fixed length array size");
}

TEST_P(CreateTableTest, UnsupportedTimestampPrecision) {
  std::string query =
      getCreateTableQuery(GetParam(), "test_table", "(col1 TIMESTAMP(10))");
  queryAndAssertException(
      query, "Exception: Only TIMESTAMP(n) where n = (0,3,6,9) are supported now.");
}

TEST_P(CreateTableTest, UnsupportedTimePrecision) {
  std::string query = getCreateTableQuery(GetParam(), "test_table", "(col1 TIME(1))");
  queryAndAssertException(query, "Exception: Only TIME(0) is supported now.");
}

TEST_P(CreateTableTest, NotNullColumnConstraint) {
  sql(getCreateTableQuery(GetParam(), "test_table", "(col1 INTEGER NOT NULL)"));

  auto& catalog = getCatalog();
  auto table = catalog.getMetadataForTable("test_table", false);
  auto column = catalog.getMetadataForColumn(table->tableId, "col1");

  ASSERT_TRUE(column->columnType.get_notnull());
}

TEST_P(CreateTableTest, DuplicateColumnNames) {
  std::string query =
      getCreateTableQuery(GetParam(), "test_table", "(col1 INTEGER, col1 INTEGER)");
  queryAndAssertException(query, "Exception: Column 'col1' defined more than once");
}

TEST_P(CreateTableTest, ExistingTableWithIfNotExists) {
  sql(getCreateTableQuery(GetParam(), "test_table", "(col1 INTEGER)"));
  sql(getCreateTableQuery(GetParam(), "test_table", "(col1 INTEGER)", true));

  auto& catalog = getCatalog();
  auto table = catalog.getMetadataForTable("test_table", false);
  assertTableDetails(table, GetParam(), "test_table", 1);

  auto column = catalog.getMetadataForColumn(table->tableId, "col1");
  ColumnAttributes expected_attributes{};
  expected_attributes.column_name = "col1";
  expected_attributes.size = 4;
  expected_attributes.type = kINT;
  assertColumnDetails(expected_attributes, column);
}

TEST_P(CreateTableTest, ExistingTableWithoutIfNotExists) {
  std::string query = getCreateTableQuery(GetParam(), "test_table", "(col1 INTEGER)");
  sql(query);
  queryAndAssertException(
      query, "Exception: Table or View with name \"test_table\" already exists.");
}

TEST_P(CreateTableTest, UnauthorizedUser) {
  createTestUser();
  login("test_user", "test_pass");
  std::string query = getCreateTableQuery(GetParam(), "test_table", "(col1 INTEGER)");

  if (GetParam() == ddl_utils::TableType::FOREIGN_TABLE) {
    queryAndAssertException(query,
                            "Exception: Foreign table \"test_table\" will not be "
                            "created. User has no CREATE TABLE privileges.");
  } else {
    queryAndAssertException(query,
                            "Exception: Table test_table will not be created. User has "
                            "no create privileges.");
  }
}

TEST_P(CreateTableTest, AuthorizedUser) {
  createTestUser();
  sql("GRANT CREATE TABLE ON DATABASE omnisci TO test_user;");
  login("test_user", "test_pass");

  sql(getCreateTableQuery(GetParam(), "test_table", "(col1 INTEGER)"));

  auto& catalog = getCatalog();
  auto table = catalog.getMetadataForTable("test_table", false);
  assertTableDetails(table, GetParam(), "test_table", 1, 1);

  auto column = catalog.getMetadataForColumn(table->tableId, "col1");
  ColumnAttributes expected_attributes{};
  expected_attributes.column_name = "col1";
  expected_attributes.size = 4;
  expected_attributes.type = kINT;
  assertColumnDetails(expected_attributes, column);
}

INSTANTIATE_TEST_SUITE_P(CreateAndDropTableDdlTest,
                         CreateTableTest,
                         testing::Values(ddl_utils::TableType::TABLE,
                                         ddl_utils::TableType::FOREIGN_TABLE),
                         [](const auto& param_info) {
                           return ddl_utils::table_type_enum_to_string(param_info.param);
                         });

class NegativePrecisionOrDimensionTest
    : public CreateAndDropTableDdlTest,
      public testing::WithParamInterface<std::tuple<ddl_utils::TableType, std::string>> {
};

TEST_P(NegativePrecisionOrDimensionTest, NegativePrecisionOrDimension) {
  const auto& [table_type, data_type] = GetParam();
  try {
    sql(getCreateTableQuery(table_type, "test_table", "(col1 " + data_type + "(-1))"));
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    if (table_type == ddl_utils::TableType::FOREIGN_TABLE) {
      ASSERT_TRUE(e.error_msg.find("Exception: Parse failed") != std::string::npos);
    } else {
      ASSERT_EQ("Exception: No negative number in type definition.", e.error_msg);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    CreateTableTest,
    NegativePrecisionOrDimensionTest,
    testing::Combine(testing::Values(ddl_utils::TableType::TABLE,
                                     ddl_utils::TableType::FOREIGN_TABLE),
                     testing::Values("CHAR", "VARCHAR", "DECIMAL", "NUMERIC")),
    [](const auto& param_info) {
      return ddl_utils::table_type_enum_to_string(std::get<0>(param_info.param)) + "_" +
             std::get<1>(param_info.param);
    });

class PrecisionAndScaleTest
    : public CreateAndDropTableDdlTest,
      public testing::WithParamInterface<std::tuple<ddl_utils::TableType, std::string>> {
};

TEST_P(PrecisionAndScaleTest, MaxPrecisionExceeded) {
  const auto& [table_type, data_type] = GetParam();
  std::string query =
      getCreateTableQuery(table_type, "test_table", "(col1 " + data_type + "(20))");
  queryAndAssertException(
      query, "Exception: DECIMAL and NUMERIC precision cannot be larger than 19.");
}

TEST_P(PrecisionAndScaleTest, ScaleNotLessThanPrecision) {
  const auto& [table_type, data_type] = GetParam();
  std::string query =
      getCreateTableQuery(table_type, "test_table", "(col1 " + data_type + "(10, 10))");
  queryAndAssertException(
      query, "Exception: DECIMAL and NUMERIC must have precision larger than scale.");
}

INSTANTIATE_TEST_SUITE_P(
    CreateTableTest,
    PrecisionAndScaleTest,
    testing::Combine(testing::Values(ddl_utils::TableType::TABLE,
                                     ddl_utils::TableType::FOREIGN_TABLE),
                     testing::Values("DECIMAL", "NUMERIC")),
    [](const auto& param_info) {
      return ddl_utils::table_type_enum_to_string(std::get<0>(param_info.param)) + "_" +
             std::get<1>(param_info.param);
    });

class CreateForeignTableTest : public CreateAndDropTableDdlTest {
  void SetUp() override {
    CreateAndDropTableDdlTest::SetUp();
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
  }

  void TearDown() override {
    g_enable_fsi = true;
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    CreateAndDropTableDdlTest::TearDown();
  }
};

TEST_F(CreateForeignTableTest, NonExistentServer) {
  std::string query{
      "CREATE FOREIGN TABLE test_foreign_table(col1 INTEGER) SERVER "
      "non_existent_server;"};
  queryAndAssertException(
      query,
      "Exception: Foreign server with name \"non_existent_server\" does not exist.");
}

TEST_F(CreateForeignTableTest, DefaultCsvFileServerName) {
  sql("CREATE FOREIGN TABLE test_foreign_table(col1 INTEGER) "
      "SERVER omnisci_local_csv WITH (file_path = 'test_file.csv');");
  ASSERT_NE(nullptr, getCatalog().getMetadataForTable("test_foreign_table", false));
}

TEST_F(CreateForeignTableTest, DefaultParquetFileServerName) {
  sql("CREATE FOREIGN TABLE test_foreign_table(col1 INTEGER) "
      "SERVER omnisci_local_parquet WITH (file_path = 'test_file.csv');");
  ASSERT_NE(nullptr, getCatalog().getMetadataForTable("test_foreign_table", false));
}

TEST_F(CreateForeignTableTest, FsiDisabled) {
  g_enable_fsi = false;
  std::string query = getCreateTableQuery(
      ddl_utils::TableType::FOREIGN_TABLE, "test_foreign_table", "(col1 INTEGER)");
  queryAndAssertException(query, "Syntax error at: FOREIGN");
}

class DropTableTest : public CreateAndDropTableDdlTest,
                      public testing::WithParamInterface<ddl_utils::TableType> {
 protected:
  void SetUp() override {
    CreateAndDropTableDdlTest::SetUp();
    sql(getCreateTableQuery(GetParam(), "test_table", "(col1 INTEGER)", true));
    dropTestUser();
  }

  void TearDown() override {
    dropTestUser();
    CreateAndDropTableDdlTest::TearDown();
  }
};

TEST_P(DropTableTest, ExistingTable) {
  sql(getDropTableQuery(GetParam(), "test_table"));
  ASSERT_EQ(nullptr, getCatalog().getMetadataForTable("test_table", false));
}

TEST_P(DropTableTest, NonExistingTableWithIfExists) {
  sql(getDropTableQuery(GetParam(), "test_table"));
  sql(getDropTableQuery(GetParam(), "test_table", true));

  ASSERT_EQ(nullptr, getCatalog().getMetadataForTable("test_table", false));
}

TEST_P(DropTableTest, NonExistentTableWithoutIfExists) {
  std::string query = getDropTableQuery(GetParam(), "test_table_2");
  queryAndAssertException(query, "Exception: Table/View test_table_2 does not exist.");
}

TEST_P(DropTableTest, UnauthorizedUser) {
  createTestUser();
  login("test_user", "test_pass");
  std::string query = getDropTableQuery(GetParam(), "test_table");

  if (GetParam() == ddl_utils::TableType::FOREIGN_TABLE) {
    queryAndAssertException(query,
                            "Exception: Foreign table \"test_table\" will not be "
                            "dropped. User has no DROP TABLE privileges.");
  } else {
    queryAndAssertException(query,
                            "Exception: Table test_table will not be dropped. User has "
                            "no proper privileges.");
  }

  loginAdmin();
  sql(getDropTableQuery(GetParam(), "test_table"));
}

TEST_P(DropTableTest, AuthorizedUser) {
  createTestUser();
  sql("GRANT DROP ON TABLE test_table TO test_user;");
  login("test_user", "test_pass");

  sql(getDropTableQuery(GetParam(), "test_table"));

  ASSERT_EQ(nullptr, getCatalog().getMetadataForTable("test_table", false));
}

INSTANTIATE_TEST_SUITE_P(CreateAndDropTableDdlTest,
                         DropTableTest,
                         testing::Values(ddl_utils::TableType::TABLE,
                                         ddl_utils::TableType::FOREIGN_TABLE),
                         [](const auto& param_info) {
                           if (param_info.param == ddl_utils::TableType::TABLE) {
                             return "Table";
                           }
                           if (param_info.param == ddl_utils::TableType::FOREIGN_TABLE) {
                             return "ForeignTable";
                           }
                           throw std::runtime_error{"Unexpected parameter type"};
                         });

class DropTableTypeMismatchTest : public CreateAndDropTableDdlTest {};

TEST_F(DropTableTypeMismatchTest, Table_DropCommandForOtherTableTypes) {
  sql(getCreateTableQuery(ddl_utils::TableType::TABLE, "test_table", "(col1 INTEGER)"));

  queryAndAssertException("DROP VIEW test_table;",
                          "Exception: test_table is a table. Use DROP TABLE.");
  queryAndAssertException("DROP FOREIGN TABLE test_table;",
                          "Exception: test_table is a table. Use DROP TABLE.");

  sql("DROP table test_table;");
  ASSERT_EQ(nullptr, getCatalog().getMetadataForTable("test_table", false));
}

TEST_F(DropTableTypeMismatchTest, View_DropCommandForOtherTableTypes) {
  sql(getCreateTableQuery(ddl_utils::TableType::TABLE, "test_table", "(col1 INTEGER)"));
  sql("CREATE VIEW test_view AS SELECT * FROM test_table;");

  queryAndAssertException("DROP table test_view;",
                          "Exception: test_view is a view. Use DROP VIEW.");
  queryAndAssertException("DROP FOREIGN TABLE test_view;",
                          "Exception: test_view is a view. Use DROP VIEW.");

  sql("DROP VIEW test_view;");
  ASSERT_EQ(nullptr, getCatalog().getMetadataForTable("test_view", false));
}

TEST_F(DropTableTypeMismatchTest, ForeignTable_DropCommandForOtherTableTypes) {
  sql(getCreateTableQuery(
      ddl_utils::TableType::FOREIGN_TABLE, "test_foreign_table", "(col1 INTEGER)"));

  queryAndAssertException(
      "DROP table test_foreign_table;",
      "Exception: test_foreign_table is a foreign table. Use DROP FOREIGN TABLE.");
  queryAndAssertException(
      "DROP VIEW test_foreign_table;",
      "Exception: test_foreign_table is a foreign table. Use DROP FOREIGN TABLE.");

  sql("DROP FOREIGN TABLE test_foreign_table;");
  ASSERT_EQ(nullptr, getCatalog().getMetadataForTable("test_foreign_table", false));
}

class DropForeignTableTest : public CreateAndDropTableDdlTest {
 protected:
  void SetUp() override {
    CreateAndDropTableDdlTest::SetUp();
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
  }

  void TearDown() override {
    g_enable_fsi = true;
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    CreateAndDropTableDdlTest::TearDown();
  }
};

TEST_F(DropForeignTableTest, FsiDisabled) {
  sql(getCreateTableQuery(
      ddl_utils::TableType::FOREIGN_TABLE, "test_foreign_table", "(col1 INTEGER)"));

  g_enable_fsi = false;
  queryAndAssertException(
      "DROP table test_foreign_table;",
      "Exception: test_foreign_table is a foreign table. Use DROP FOREIGN TABLE.");
}

int main(int argc, char** argv) {
  g_enable_fsi = true;
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_fsi = false;
  return err;
}
