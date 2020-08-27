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
 * @file ForeignTableDmlTest.cpp
 * @brief Test suite for DML SQL queries on foreign tables
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "DBHandlerTestHelpers.h"
#include "DataMgr/ForeignStorage/ForeignStorageCache.h"
#include "DataMgr/ForeignStorage/ForeignStorageMgr.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "Shared/geo_types.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;
std::string test_binary_file_path;

namespace bp = boost::process;
namespace bf = boost::filesystem;
using path = bf::path;

static const std::string default_table_name = "test_foreign_table";

/**
 * Helper class for creating foreign tables
 */
class ForeignTableTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override { DBHandlerTestFixture::SetUp(); }
  void TearDown() override { DBHandlerTestFixture::TearDown(); }
  static std::string getCreateForeignTableQuery(const std::string& columns,
                                                const std::string& file_name_base,
                                                const std::string& data_wrapper_type,
                                                const int table_number = 0) {
    return getCreateForeignTableQuery(
        columns, {}, file_name_base, data_wrapper_type, table_number);
  }

  static std::string getCreateForeignTableQuery(
      const std::string& columns,
      const std::map<std::string, std::string> options,
      const std::string& file_name_base,
      const std::string& data_wrapper_type,
      const int table_number = 0,
      const std::string& table_name = default_table_name,
      const std::string extension = "") {
    std::string query{"CREATE FOREIGN TABLE " + table_name};
    if (table_number) {
      query += "_" + std::to_string(table_number);
    }
    std::string filename =
        file_name_base + "." + (extension.size() > 0 ? extension : data_wrapper_type);
    query += " " + columns + " SERVER omnisci_local_" + data_wrapper_type +
             " WITH (file_path = '" + getDataFilesPath() + filename + "'";
    for (auto& [key, value] : options) {
      query += ", " + key + " = '" + value + "'";
    }
    query += ");";
    return query;
  }

  static std::string getDataFilesPath() {
    return bf::canonical(test_binary_file_path + "/../../Tests/FsiDataFiles").string() +
           "/";
  }

  static void sqlCreateForeignTable(const std::string& columns,
                                    const std::string& file_name,
                                    const std::string& data_wrapper_type,
                                    const std::map<std::string, std::string> options = {},
                                    const int table_number = 0,
                                    const std::string& table_name = default_table_name) {
    sqlDropForeignTable(table_number, table_name);
    auto query = getCreateForeignTableQuery(
        columns, options, file_name, data_wrapper_type, table_number, table_name);
    sql(query);
  }

  static void sqlDropForeignTable(const int table_number = 0,
                                  const std::string& table_name = default_table_name) {
    std::string query{"DROP FOREIGN TABLE IF EXISTS " + table_name};
    if (table_number != 0) {
      query += "_" + std::to_string(table_number);
    }
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
};

class SelectQueryTest : public ForeignTableTest {
 protected:
  void SetUp() override {
    ForeignTableTest::SetUp();
    import_export::delimited_parser::set_max_buffer_resize(max_buffer_resize_);
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table_2;");
    sql("DROP SERVER IF EXISTS test_server;");
  }

  void TearDown() override {
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table_2;");
    sql("DROP SERVER IF EXISTS test_server;");
    ForeignTableTest::TearDown();
  }

  inline static size_t max_buffer_resize_ =
      import_export::delimited_parser::get_max_buffer_resize();
};

class CacheControllingSelectQueryTest : public SelectQueryTest,
                                        public ::testing::WithParamInterface<bool> {
 public:
  inline static std::string cache_path_ = to_string(BASE_PATH) + "/omnisci_disk_cache/";
  bool starting_cache_state_;

 protected:
  void resetPersistentStorageMgr(bool cache_enabled) {
    for (auto table_it : getCatalog().getAllTableMetadata()) {
      getCatalog().removeFragmenterForTable(table_it->tableId);
    }
    getCatalog().getDataMgr().resetPersistentStorage(
        {cache_path_, cache_enabled}, 0, getSystemParameters());
  }

  void SetUp() override {
    // Disable/enable the cache as test param requires
    starting_cache_state_ =
        (getCatalog().getDataMgr().getForeignStorageMgr()->getForeignStorageCache() !=
         nullptr);
    if (starting_cache_state_ != GetParam()) {
      resetPersistentStorageMgr(GetParam());
    }
    SelectQueryTest::SetUp();
  }

  void TearDown() override {
    SelectQueryTest::TearDown();
    // Reset cache to pre-test conditions
    if (starting_cache_state_ != GetParam()) {
      resetPersistentStorageMgr(starting_cache_state_);
    }
  }
};

class DataWrapperSelectQueryTest : public SelectQueryTest,
                                   public ::testing::WithParamInterface<std::string> {};

struct DataTypeFragmentSizeAndDataWrapperParam {
  int fragment_size;
  std::string wrapper;
  std::string extension;
};

class DataTypeFragmentSizeAndDataWrapperTest
    : public SelectQueryTest,
      public testing::WithParamInterface<DataTypeFragmentSizeAndDataWrapperParam> {};

class RowGroupAndFragmentSizeSelectQueryTest
    : public SelectQueryTest,
      public ::testing::WithParamInterface<std::pair<int64_t, int64_t>> {};

namespace {
struct PrintToStringParamName {
  template <class ParamType>
  std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const {
    std::stringstream ss;
    ss << info.param;
    return ss.str();
  }

  std::string operator()(
      const ::testing::TestParamInfo<std::pair<int, std::string>>& info) const {
    std::stringstream ss;
    ss << "Fragment_size_" << info.param.first << "_Data_wrapper_" << info.param.second;
    return ss.str();
  }

  std::string operator()(
      const ::testing::TestParamInfo<DataTypeFragmentSizeAndDataWrapperParam>& info)
      const {
    std::stringstream ss;
    ss << "Fragment_size_" << info.param.fragment_size << "_Data_wrapper_"
       << info.param.wrapper << "_Extension_" << info.param.extension;

    return ss.str();
  }

  std::string operator()(
      const ::testing::TestParamInfo<std::pair<int64_t, int64_t>>& info) const {
    std::stringstream ss;
    ss << "Rowgroup_size_" << info.param.first << "_Fragment_size_" << info.param.second;
    return ss.str();
  }
  std::string operator()(
      const ::testing::TestParamInfo<std::pair<std::string, std::string>>& info) const {
    std::stringstream ss;
    ss << "File_type_" << info.param.second;
    return ss.str();
  }
  std::string operator()(const ::testing::TestParamInfo<TExecuteMode::type>& info) const {
    std::stringstream ss;
    ss << ((info.param == TExecuteMode::GPU) ? "GPU" : "CPU");
    return ss.str();
  }
};
}  // namespace

TEST_P(CacheControllingSelectQueryTest, CustomServer) {
  sql("CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv "s +
      "WITH (storage_type = 'LOCAL_FILE', base_path = '" + getDataFilesPath() + "');");
  sql("CREATE FOREIGN TABLE test_foreign_table (t TEXT, i INTEGER[]) "
      "SERVER test_server WITH (file_path = 'example_1.csv');");
  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, DefaultLocalCsvServer) {
  std::string query = "CREATE FOREIGN TABLE test_foreign_table (t TEXT, i INTEGER[]) "s +
                      "SERVER omnisci_local_csv WITH (file_path = '" +
                      getDataFilesPath() + "/example_1.csv');";
  sql(query);
  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, DefaultLocalParquetServer) {
  std::string query =
      "CREATE FOREIGN TABLE test_foreign_table (t TEXT, i INTEGER, f DOUBLE) "s +
      "SERVER omnisci_local_parquet WITH (file_path = '" + getDataFilesPath() +
      "/example_2.parquet');";
  sql(query);
  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{"a", i(1), 1.1},
                        {"aa", i(1), 1.1},
                        {"aa", i(2), 2.2},
                        {"aaa", i(1), 1.1},
                        {"aaa", i(2), 2.2},
                        {"aaa", i(3), 3.3}},
                       result);
}

// Create table with multiple fragments with file buffers less than size of a
// fragment Includes both fixed and variable length data
TEST_P(CacheControllingSelectQueryTest, MultipleDataBlocksPerFragment) {
  const auto& query =
      getCreateForeignTableQuery("(i INTEGER,  txt TEXT, txt_2 TEXT ENCODING NONE)",
                                 {{"buffer_size", "25"}, {"fragment_size", "64"}},
                                 "0_255",
                                 "csv");
  sql(query);

  // Check that data is correct
  {
    std::vector<std::vector<TargetValue>> expected_result_set;
    for (int number = 0; number < 256; number++) {
      expected_result_set.push_back(
          {i(number), std::to_string(number), std::to_string(number)});
    }
    TQueryResult result;
    sql(result, "SELECT * FROM test_foreign_table ORDER BY i;");
    assertResultSetEqual(expected_result_set, result);
  }

  // Check that WHERE statements filter numerical data correctly
  {
    std::vector<std::vector<TargetValue>> expected_result_set;
    for (int number = 128; number < 256; number++) {
      expected_result_set.push_back(
          {i(number), std::to_string(number), std::to_string(number)});
    }
    TQueryResult result;
    sql(result, "SELECT * FROM test_foreign_table  WHERE i >= 128 ORDER BY i;");
    assertResultSetEqual(expected_result_set, result);
  }
  {
    std::vector<std::vector<TargetValue>> expected_result_set;
    for (int number = 0; number < 128; number++) {
      expected_result_set.push_back(
          {i(number), std::to_string(number), std::to_string(number)});
    }
    TQueryResult result;
    sql(result, "SELECT * FROM test_foreign_table  WHERE i < 128 ORDER BY i;");
    assertResultSetEqual(expected_result_set, result);
  }
}

TEST_P(CacheControllingSelectQueryTest, ParquetGeoTypesMalformed) {
  const auto& query = getCreateForeignTableQuery(
      "(p POINT, l LINESTRING, poly POLYGON, multipoly MULTIPOLYGON)",
      "geo_types.malformed",
      "parquet");
  sql(query);

  queryAndAssertException("SELECT * FROM test_foreign_table;",
                          "Exception: Failure to import geo column 'l' in table "
                          "'test_foreign_table' for row group 0 and row 1.");
}

TEST_P(CacheControllingSelectQueryTest, ParquetGeoTypesNull) {
  const auto& query = getCreateForeignTableQuery(
      "(p POINT, l LINESTRING, poly POLYGON, multipoly MULTIPOLYGON)",
      "geo_types.null",
      "parquet");
  sql(query);

  queryAndAssertException("SELECT * FROM test_foreign_table;",
                          "Exception: Failure to import geo column 'l' in table "
                          "'test_foreign_table' for row group 0 and row 1.");
}

TEST_P(CacheControllingSelectQueryTest, ParquetNullRowgroups) {
  const auto& query =
      getCreateForeignTableQuery("(a SMALLINT, b SMALLINT)", "null_columns", "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({{i(NULL_SMALLINT),i(1)},
                        {i(NULL_SMALLINT),i(2)},
                        {i(NULL_SMALLINT),i(NULL_SMALLINT)},
                        {i(NULL_SMALLINT),i(NULL_SMALLINT)}},
                       result);
  // clang-format on
}

TEST_P(CacheControllingSelectQueryTest, CacheExists) {
  auto cache = getCatalog().getDataMgr().getForeignStorageMgr()->getForeignStorageCache();
  ASSERT_EQ((cache != nullptr), GetParam());
}

INSTANTIATE_TEST_SUITE_P(CachOnOffSelectQueryTests,
                         CacheControllingSelectQueryTest,
                         ::testing::Values(true, false),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(DataWrapperParameterizedTests,
                         DataWrapperSelectQueryTest,
                         ::testing::Values("csv", "parquet"),
                         PrintToStringParamName());

TEST_P(DataWrapperSelectQueryTest, AggregateAndGroupBy) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);

  TQueryResult result;
  sql(result, "SELECT t, avg(i), sum(f) FROM test_foreign_table group by t;");
  // clang-format off
  assertResultSetEqual({{"a", 1.0, 1.1},
                        {"aa", 1.5, 3.3},
                        {"aaa", 2.0, 6.6}},
                       result);
  // clang-format on
}

// TODO: implement for parquet when kARRAY support implemented for parquet
TEST_P(CacheControllingSelectQueryTest, Join) {
  auto query = getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1", "csv");
  sql(query);

  query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER, d DOUBLE)", "example_2", "csv", 2);
  sql(query);

  TQueryResult result;
  sql(result,
      "SELECT t1.t, t1.i, t2.i, t2.d FROM test_foreign_table AS t1 JOIN "
      "test_foreign_table_2 AS t2 ON t1.t = t2.t;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)}), i(1), 1.1},
                        {"aa", array({Null_i, i(2), i(2)}), i(1), 1.1},
                        {"aa", array({Null_i, i(2), i(2)}), i(2), 2.2},
                        {"aaa", array({i(3), Null_i, i(3)}), i(1), 1.1},
                        {"aaa", array({i(3), Null_i, i(3)}), i(2), 2.2},
                        {"aaa", array({i(3), Null_i, i(3)}), i(3), 3.3}},
                       result);
}

TEST_P(DataWrapperSelectQueryTest, Filter) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table WHERE i > 1;");
  // clang-format off
  assertResultSetEqual({{"aa", i(2), 2.2},
                        {"aaa", i(2), 2.2},
                        {"aaa", i(3), 3.3}},
                       result);
  // clang-format on
}

// TODO: implement for parquet when kARRAY support implemented for parquet
TEST_P(CacheControllingSelectQueryTest, Sort) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table ORDER BY t DESC;");
  assertResultSetEqual({{"aaa", array({i(3), Null_i, i(3)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"a", array({i(1), i(1), i(1)})}},
                       result);
}

TEST_P(DataWrapperSelectQueryTest, Update) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);
  queryAndAssertException("UPDATE test_foreign_table SET t = 'abc';",
                          "Exception: DELETE, INSERT, OR UPDATE commands are not "
                          "supported for foreign tables.");
}

TEST_P(DataWrapperSelectQueryTest, Insert) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);
  queryAndAssertException("INSERT INTO test_foreign_table VALUES('abc', null, null);",
                          "Exception: DELETE, INSERT, OR UPDATE commands are not "
                          "supported for foreign tables.");
}

TEST_P(DataWrapperSelectQueryTest, InsertIntoSelect) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);
  queryAndAssertException(
      "INSERT INTO test_foreign_table SELECT * FROM test_foreign_table;",
      "Exception: DELETE, INSERT, OR UPDATE commands are not supported for "
      "foreign "
      "tables.");
}

TEST_P(DataWrapperSelectQueryTest, Delete) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);
  queryAndAssertException("DELETE FROM test_foreign_table WHERE t = 'a';",
                          "Exception: DELETE, INSERT, OR UPDATE commands are not "
                          "supported for foreign tables.");
}

TEST_P(CacheControllingSelectQueryTest, CSV_CustomDelimiters) {
  const auto& query = getCreateForeignTableQuery(
      "(b BOOLEAN, i INTEGER, f FLOAT, t TIME, tp TIMESTAMP, d DATE, "
      "txt TEXT, txt_2 TEXT, i_arr INTEGER[], txt_arr TEXT[])",
      {{"delimiter", "|"}, {"array_delimiter", "_"}},
      "custom_delimiters",
      "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({
    {
      True, i(30000), 10.1f, "00:00:10", "1/1/2000 00:00:59", "1/1/2000", "text_1",
      "quoted text", array({i(1)}), array({"quoted text"})
    },
    {
      False, i(30500), 100.12f, "00:10:00", "6/15/2020 00:59:59", "6/15/2020", "text_2",
      "quoted text 2", array({i(1), i(2), i(3)}), array({"quoted text 2", "quoted text 3"})
    },
    {
      True, i(31000), 1000.123f, "10:00:00", "12/31/2500 23:59:59", "12/31/2500", "text_3",
      "quoted text 3", array({i(10), i(20), i(30)}), array({"quoted_text_4", "quoted_text_5"})
    }},
    result);
  // clang-format on
}

class CSVFileTypeTests
    : public SelectQueryTest,
      public ::testing::WithParamInterface<std::pair<std::string, std::string>> {};

INSTANTIATE_TEST_SUITE_P(
    CSVFileTypeParameterizedTests,
    CSVFileTypeTests,
    ::testing::Values(std::make_pair("example_1.csv", "uncompressed"),
                      std::make_pair("example_1.zip", "zip"),
                      std::make_pair("example_1_multi.zip", "multi_zip"),
                      std::make_pair("example_1_multilevel.zip", "multilevel_zip"),
                      std::make_pair("example_1.tar.gz", "tar_gz"),
                      std::make_pair("example_1_multi.tar.gz", "multi_tar_gz"),
                      std::make_pair("example_1.7z", "7z"),
                      std::make_pair("example_1.rar", "rar"),
                      std::make_pair("example_1.bz2", "bz2"),
                      std::make_pair("example_1_multi.7z", "7z_multi"),
                      std::make_pair("example_1.csv.gz", "gz"),
                      std::make_pair("example_1_dir", "dir"),
                      std::make_pair("example_1_dir_archives", "dir_archives"),
                      std::make_pair("example_1_dir_multilevel", "multilevel_dir")),
    PrintToStringParamName());

TEST_P(CSVFileTypeTests, SelectCSV) {
  std::string query = "CREATE FOREIGN TABLE test_foreign_table (t TEXT, i INTEGER[]) "s +
                      "SERVER omnisci_local_csv WITH (file_path = '" +
                      getDataFilesPath() + "/" + GetParam().first + "');";
  sql(query);
  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table  ORDER BY t;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, CsvEmptyArchive) {
  std::string query = "CREATE FOREIGN TABLE test_foreign_table (t TEXT, i INTEGER[]) "s +
                      "SERVER omnisci_local_csv WITH (file_path = '" +
                      getDataFilesPath() + "/" + "example_1_empty.zip" + "');";
  sql(query);
  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table  ORDER BY t;");
  assertResultSetEqual({}, result);
}

TEST_P(CacheControllingSelectQueryTest, CsvDirectoryBadFileExt) {
  std::string query = "CREATE FOREIGN TABLE test_foreign_table (t TEXT, i INTEGER[]) "s +
                      "SERVER omnisci_local_csv WITH (file_path = '" +
                      getDataFilesPath() + "/" + "example_1_dir_bad_ext/" + "');";
  sql(query);
  queryAndAssertException("SELECT * FROM test_foreign_table  ORDER BY t;",
                          "Exception: Invalid extention for file \"" +
                              getDataFilesPath() +
                              "example_1_dir_bad_ext/example_1c.tmp\".");
}

TEST_P(CacheControllingSelectQueryTest, CsvArchiveInvalidFile) {
  std::string query = "CREATE FOREIGN TABLE test_foreign_table (t TEXT, i INTEGER[]) "s +
                      "SERVER omnisci_local_csv WITH (file_path = '" +
                      getDataFilesPath() + "/" + "example_1_invalid_file.zip" + "');";
  sql(query);
  queryAndAssertException("SELECT * FROM test_foreign_table  ORDER BY t;",
                          "Exception: Mismatched number of logical columns: (expected 2 "
                          "columns, has 1): (random text)");
}

TEST_P(CacheControllingSelectQueryTest, CSV_CustomLineDelimiters) {
  const auto& query = getCreateForeignTableQuery("(b BOOLEAN, i INTEGER, t TEXT)",
                                                 {{"line_delimiter", "*"}},
                                                 "custom_line_delimiter",
                                                 "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({{True, i(1000), "text_1"},
                        {False, i(2000), "text_2"},
                        {True, i(3000), "text_3"}},
                       result);
  // clang-format on
}

// TODO: implement for parquet when kARRAY support implemented for parquet
// Note: only need to test array_marker and array_delimiter
TEST_P(CacheControllingSelectQueryTest, CSV_CustomMarkers) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, t2 TEXT, i INTEGER[])",
      {{"array_marker", "[]"}, {"escape", "\\"}, {"nulls", "NIL"}, {"quote", "|"}},
      "custom_markers",
      "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{Null, "quoted |text|", array({i(1)})},
                        {"text_1", "quoted text", array({i(1), i(2)})},
                        {Null, "\"quoted\" \"text\"", array({i(3), i(4), i(5)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, CSV_NoHeader) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER[])", {{"header", "false"}}, "no_header", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({i(2), i(2), i(2)})},
                        {"aaa", array({i(3), i(3), i(3)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, CSV_QuotedHeader) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "quoted_headers", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({i(2), i(2), i(2)})},
                        {"aaa", array({i(3), i(3), i(3)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, CSV_NonQuotedFields) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, t2 TEXT)", {{"quoted", "false"}}, "non_quoted", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({{"text_1", "\"text_1\""},
                        {"text_2", "\"text_2\""},
                        {"text_3", "\"text_3\""}},
                       result);
  // clang-format on
}

TEST_P(CacheControllingSelectQueryTest, WithBufferSizeOption) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER[])", {{"buffer_size", "25"}}, "example_1", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table ORDER BY t;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, WithBufferSizeLessThanRowSize) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER[])", {{"buffer_size", "10"}}, "example_1", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table ORDER BY t;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, WithMaxBufferResizeLessThanRowSize) {
  import_export::delimited_parser::set_max_buffer_resize(15);
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER[])", {{"buffer_size", "10"}}, "example_1", "csv");
  sql(query);

  queryAndAssertException(
      "SELECT * FROM test_foreign_table ORDER BY t;",
      "Exception: Unable to find an end of line character after reading 15 characters. "
      "Please ensure that the correct \"line_delimiter\" option is specified or update "
      "the \"buffer_size\" option appropriately. Row number: 2. "
      "First few characters in row: aa,{'NA', 2, 2}");
}

TEST_P(CacheControllingSelectQueryTest, ReverseLongitudeAndLatitude) {
  const auto& query = getCreateForeignTableQuery(
      "(p POINT)", {{"lonlat", "false"}}, "reversed_long_lat", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({{"POINT (1 0)"},
                        {"POINT (2 1)"},
                        {"POINT (3 2)"}},
                       result);
  // clang-format on
}

class RefreshForeignTableTest : public ForeignTableTest {
 protected:
  std::string table_1_filename = "refresh_tmp_1";
  std::string table_2_filename = "refresh_tmp_2";
  std::string table_1_name = default_table_name;
  std::string table_2_name = default_table_name + "_1";
  Catalog_Namespace::Catalog* cat;
  foreign_storage::ForeignStorageCache* cache;
  ChunkKey key_1, key_2;

  void SetUp() override {
    ForeignTableTest::SetUp();
    cat = &getCatalog();
    cache = cat->getDataMgr().getForeignStorageMgr()->getForeignStorageCache();
    cache->clear();
  }

  void TearDown() override {
    bf::remove(getDataFilesPath() + table_1_filename + ".csv");
    bf::remove(getDataFilesPath() + table_2_filename + ".csv");
    sqlDropForeignTable(0, table_1_name);
    sqlDropForeignTable(0, table_2_name);
    ForeignTableTest::TearDown();
  }

  bool isChunkAndMetadataCached(const ChunkKey& chunk_key) {
    if (cache->getCachedChunkIfExists(chunk_key) != nullptr &&
        cache->isMetadataCached(chunk_key)) {
      return true;
    }
    return false;
  }
};

class RefreshTests : public ForeignTableTest {
 protected:
  const std::string default_name = "refresh_tmp";
  std::string file_type;
  std::vector<std::string> tmp_file_names;
  std::vector<std::string> table_names;
  Catalog_Namespace::Catalog* cat;
  foreign_storage::ForeignStorageCache* cache;

  void SetUp() override {
    ForeignTableTest::SetUp();
    cat = &getCatalog();
    cache = cat->getDataMgr().getForeignStorageMgr()->getForeignStorageCache();
    cache->clear();
  }

  void TearDown() override {
    for (auto file_name : tmp_file_names) {
      bf::remove(getDataFilesPath() + file_name + "." + file_type);
    }
    for (auto table_name : table_names) {
      sqlDropForeignTable(0, table_name);
    }
    ForeignTableTest::TearDown();
  }

  bool isChunkAndMetadataCached(const ChunkKey& chunk_key) {
    if (cache->getCachedChunkIfExists(chunk_key) != nullptr &&
        cache->isMetadataCached(chunk_key)) {
      return true;
    }
    return false;
  }

  void createFilesAndTables(
      const std::vector<std::string>& file_names,
      const std::string& column_schema = "(i INTEGER)",
      const std::map<std::string, std::string>& table_options = {}) {
    for (size_t i = 0; i < file_names.size(); ++i) {
      tmp_file_names.emplace_back(default_name + std::to_string(i));
      table_names.emplace_back(default_name + std::to_string(i));
      bf::copy_file(getDataFilesPath() + file_names[i] + "." + file_type,
                    getDataFilesPath() + tmp_file_names[i] + "." + file_type,
                    bf::copy_option::overwrite_if_exists);
      sqlCreateForeignTable(
          column_schema, tmp_file_names[i], file_type, table_options, 0, table_names[i]);
    }
  }
};

class RefreshParamTests : public RefreshTests,
                          public ::testing::WithParamInterface<std::string> {
 protected:
  void SetUp() override {
    file_type = GetParam();
    RefreshTests::SetUp();
  }
};
INSTANTIATE_TEST_SUITE_P(RefreshParamTestsParameterizedTests,
                         RefreshParamTests,
                         ::testing::Values("csv", "parquet"),
                         PrintToStringParamName());

TEST_P(RefreshParamTests, SingleTable) {
  // Create initial files and tables
  createFilesAndTables({"0"});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(0)}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "1" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(0)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1)}});
}

TEST_P(RefreshParamTests, FragmentSkip) {
  // Create initial files and tables
  createFilesAndTables({"0", "1"});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + " WHERE i >= 3;", {});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ASSERT_EQ(cache->getCachedChunkIfExists(orig_key0), nullptr);
  ASSERT_TRUE(cache->isMetadataCached(orig_key0));

  sqlAndCompareResult("SELECT * FROM " + table_names[1] + " WHERE i >= 3;", {});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat, table_names[1], {1, 0});
  ASSERT_EQ(cache->getCachedChunkIfExists(orig_key1), nullptr);
  ASSERT_TRUE(cache->isMetadataCached(orig_key1));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "2" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);
  bf::copy_file(getDataFilesPath() + "3" + "." + file_type,
                getDataFilesPath() + tmp_file_names[1] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + " WHERE i >= 3;", {});
  ASSERT_EQ(cache->getCachedChunkIfExists(orig_key0), nullptr);
  ASSERT_TRUE(cache->isMetadataCached(orig_key0));

  sqlAndCompareResult("SELECT * FROM " + table_names[1] + " WHERE i >= 3;", {});
  ASSERT_EQ(cache->getCachedChunkIfExists(orig_key1), nullptr);
  ASSERT_TRUE(cache->isMetadataCached(orig_key1));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + ", " + tmp_file_names[1] + ";");

  // Compare new results
  ASSERT_EQ(cache->getCachedChunkIfExists(orig_key0), nullptr);
  ASSERT_TRUE(cache->isMetadataCached(orig_key0));
  ASSERT_EQ(cache->getCachedChunkIfExists(orig_key1), nullptr);
  ASSERT_TRUE(cache->isMetadataCached(orig_key1));
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + " WHERE i >= 3;", {});
  sqlAndCompareResult("SELECT * FROM " + table_names[1] + " WHERE i >= 3;", {{i(3)}});
  ASSERT_EQ(cache->getCachedChunkIfExists(orig_key0), nullptr);
  ASSERT_TRUE(cache->isMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
}

TEST_P(RefreshParamTests, TwoTable) {
  // Create initial files and tables
  createFilesAndTables({"0", "1"});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(0)}});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));

  sqlAndCompareResult("SELECT * FROM " + table_names[1] + ";", {{i(1)}});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat, table_names[1], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "2" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);
  bf::copy_file(getDataFilesPath() + "3" + "." + file_type,
                getDataFilesPath() + tmp_file_names[1] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(0)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));

  sqlAndCompareResult("SELECT * FROM " + table_names[1] + ";", {{i(1)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + ", " + tmp_file_names[1] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(2)}});
  sqlAndCompareResult("SELECT * FROM " + table_names[1] + ";", {{i(3)}});
}

TEST_P(RefreshParamTests, EvictTrue) {
  // Create initial files and tables
  createFilesAndTables({"0"});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(0)}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "1" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(0)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + " WITH (evict = true);");

  // Compare new results
  ASSERT_EQ(cache->getCachedChunkIfExists(orig_key), nullptr);
  ASSERT_FALSE(cache->isMetadataCached(orig_key));
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1)}});
}

TEST_P(RefreshParamTests, TwoColumn) {
  // Create initial files and tables
  createFilesAndTables({"two_col_1_2"}, "(i INTEGER, i2 INTEGER)");

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1), i(2)}});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat, table_names[0], {2, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "two_col_3_4" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1), i(2)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(3), i(4)}});
}

TEST_P(RefreshParamTests, ChangeSchema) {
  // Create initial files and tables
  createFilesAndTables({"1"});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1)}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "two_col_3_4" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  try {
    sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + ";");
    FAIL() << "An exception should have been thrown";
  } catch (const std::exception& e) {
    ASSERT_NE(strstr(e.what(), "Mismatched number of logical columns"), nullptr);
  }
}

TEST_P(RefreshParamTests, AddFrags) {
  // Create initial files and tables
  createFilesAndTables({"two_row_1_2"}, "(i INTEGER)", {{"fragment_size", "1"}});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1)}, {i(2)}});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat, table_names[0], {1, 1});
  ChunkKey orig_key2 = getChunkKeyFromTable(*cat, table_names[0], {1, 2});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "three_row_3_4_5" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1)}, {i(2)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  ASSERT_EQ(cache->getCachedChunkIfExists(orig_key2), nullptr);
  ASSERT_TRUE(cache->isMetadataCached(orig_key2));
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(3)}, {i(4)}, {i(5)}});
}

TEST_P(RefreshParamTests, SubFrags) {
  // Create initial files and tables
  createFilesAndTables({"three_row_3_4_5"}, "(i INTEGER)", {{"fragment_size", "1"}});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(3)}, {i(4)}, {i(5)}});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat, table_names[0], {1, 1});
  ChunkKey orig_key2 = getChunkKeyFromTable(*cat, table_names[0], {1, 2});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key2));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "two_row_1_2" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(3)}, {i(4)}, {i(5)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key2));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  ASSERT_EQ(cache->getCachedChunkIfExists(orig_key2), nullptr);
  ASSERT_FALSE(cache->isMetadataCached(orig_key2));
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1)}, {i(2)}});
}

TEST_P(RefreshParamTests, TwoFrags) {
  // Create initial files and tables
  createFilesAndTables({"two_row_1_2"}, "(i INTEGER)", {{"fragment_size", "1"}});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1)}, {i(2)}});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat, table_names[0], {1, 1});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "two_row_3_4" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1)}, {i(2)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(3)}, {i(4)}});
}

TEST_P(RefreshParamTests, String) {
  // Create initial files and tables
  createFilesAndTables({"a"}, "(t TEXT)");

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{"a"}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "b" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{"a"}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{"b"}});
}

class RefreshDeviceTests : public RefreshTests,
                           public ::testing::WithParamInterface<TExecuteMode::type> {
 protected:
  void SetUp() override {
    RefreshTests::SetUp();
    file_type = "csv";
  }
};
INSTANTIATE_TEST_SUITE_P(RefreshDeviceTestsParameterizedTests,
                         RefreshDeviceTests,
                         ::testing::Values(TExecuteMode::CPU, TExecuteMode::GPU),
                         PrintToStringParamName());

TEST_P(RefreshDeviceTests, Device) {
  if (!setExecuteMode(GetParam())) {
    return;
  }
  // Create initial files and tables
  createFilesAndTables({"0"});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(0)}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "1" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(0)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1)}});
}

class RefreshSyntaxTests : public RefreshTests,
                           public ::testing::WithParamInterface<std::string> {
 protected:
  void SetUp() override {
    RefreshTests::SetUp();
    file_type = "csv";
  }
};
INSTANTIATE_TEST_SUITE_P(RefreshSyntaxTestsParameterizedTests,
                         RefreshSyntaxTests,
                         ::testing::Values(" WITH (evict = false)",
                                           " WITH (EVICT = FALSE)"));

TEST_P(RefreshSyntaxTests, EvictFalse) {
  // Create initial files and tables
  createFilesAndTables({"0"});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(0)}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat, table_names[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Change underlying file
  bf::copy_file(getDataFilesPath() + "1" + "." + file_type,
                getDataFilesPath() + tmp_file_names[0] + "." + file_type,
                bf::copy_option::overwrite_if_exists);

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(0)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + tmp_file_names[0] + GetParam() + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));
  sqlAndCompareResult("SELECT * FROM " + table_names[0] + ";", {{i(1)}});
}

INSTANTIATE_TEST_SUITE_P(
    DataTypeFragmentSizeAndDataWrapperParameterizedTests,
    DataTypeFragmentSizeAndDataWrapperTest,
    ::testing::Values(
        DataTypeFragmentSizeAndDataWrapperParam{1, "csv", "csv"},
        DataTypeFragmentSizeAndDataWrapperParam{1, "csv", "zip"},
        DataTypeFragmentSizeAndDataWrapperParam{1, "parquet", "parquet"},
        DataTypeFragmentSizeAndDataWrapperParam{2, "csv", "csv"},
        DataTypeFragmentSizeAndDataWrapperParam{2, "csv", "zip"},
        DataTypeFragmentSizeAndDataWrapperParam{2, "parquet", "parquet"},
        DataTypeFragmentSizeAndDataWrapperParam{32000000, "csv", "csv"},
        DataTypeFragmentSizeAndDataWrapperParam{32000000, "csv", "zip"},
        DataTypeFragmentSizeAndDataWrapperParam{32000000, "parquet", "parquet"}),
    PrintToStringParamName());

TEST_P(DataTypeFragmentSizeAndDataWrapperTest, ScalarTypes) {
  auto& param = GetParam();
  int fragment_size = param.fragment_size;
  std::string data_wrapper_type = param.wrapper;
  std::string extension = param.extension;
  const auto& query = getCreateForeignTableQuery(
      "(b BOOLEAN, t TINYINT, s SMALLINT, i INTEGER, bi BIGINT, f FLOAT, "
      "dc DECIMAL(10, 5), tm TIME, tp TIMESTAMP, d DATE, txt TEXT, "
      "txt_2 TEXT ENCODING NONE)",
      {{"fragment_size", std::to_string(fragment_size)}},
      "scalar_types",
      data_wrapper_type,
      0,
      default_table_name,
      extension);
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({
    {
      True, i(100), i(30000), i(2000000000), i(9000000000000000000), 10.1f, 100.1234, "00:00:10",
      "1/1/2000 00:00:59", "1/1/2000", "text_1", "quoted text"
    },
    {
      False, i(110), i(30500), i(2000500000), i(9000000050000000000), 100.12f, 2.1234, "00:10:00",
    "6/15/2020 00:59:59", "6/15/2020", "text_2", "quoted text 2"
    },
    {
      True, i(120), i(31000), i(2100000000), i(9100000000000000000), 1000.123f, 100.1, "10:00:00",
      "12/31/2500 23:59:59", "12/31/2500", "text_3", "quoted text 3"
    }},
    result);
  // clang-format on
}

TEST_P(DataTypeFragmentSizeAndDataWrapperTest, ArrayTypes) {
  auto& param = GetParam();
  int fragment_size = param.fragment_size;
  std::string data_wrapper_type = param.wrapper;
  std::string extension = param.extension;
  // TODO: implement for parquet when kARRAY support implemented for parquet
  if (data_wrapper_type == "parquet") {
    GTEST_SKIP();
  }
  const auto& query = getCreateForeignTableQuery(
      "(b BOOLEAN[], t TINYINT[], s SMALLINT[], i INTEGER[], bi BIGINT[], f "
      "FLOAT[], "
      "tm "
      "TIME[], tp TIMESTAMP[], "
      "d DATE[], txt TEXT[], txt_2 TEXT[])",
      {{"fragment_size", std::to_string(fragment_size)}},
      "array_types",
      data_wrapper_type,
      0,
      default_table_name,
      extension);
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({
    {
      array({True}), array({i(50), i(100)}), array({i(30000), i(20000)}), array({i(2000000000)}),
      array({i(9000000000000000000)}), array({10.1f, 11.1f}), array({"00:00:10"}),
      array({"1/1/2000 00:00:59", "1/1/2010 00:00:59"}), array({"1/1/2000", "2/2/2000"}),
      array({"text_1"}), array({"quoted text"})
    },
    {
      array({False, True}), array({i(110)}), array({i(30500)}), array({i(2000500000)}),
      array({i(9000000050000000000)}), array({100.12f}), array({"00:10:00", "00:20:00"}),
      array({"6/15/2020 00:59:59"}), array({"6/15/2020"}),
      array({"text_2", "text_3"}), array({"quoted text 2"})
    },
    {
      array({True}), array({i(120)}), array({i(31000)}), array({i(2100000000), i(200000000)}),
      array({i(9100000000000000000), i(9200000000000000000)}), array({1000.123f}), array({"10:00:00"}),
      array({"12/31/2500 23:59:59"}), array({"12/31/2500"}),
      array({"text_4"}), array({"quoted text 3", "quoted text 4"})
    }},
    result);
  // clang-format on
}

TEST_P(DataTypeFragmentSizeAndDataWrapperTest, GeoTypes) {
  auto& param = GetParam();
  int fragment_size = param.fragment_size;
  std::string data_wrapper_type = param.wrapper;
  std::string extension = param.extension;
  const auto& query = getCreateForeignTableQuery(
      "(p POINT, l LINESTRING, poly POLYGON, multipoly MULTIPOLYGON)",
      {{"fragment_size", std::to_string(fragment_size)}},
      "geo_types",
      data_wrapper_type,
      0,
      default_table_name,
      extension);
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({
    {
      "POINT (0 0)", "LINESTRING (0 0,0 0)", "POLYGON ((0 0,1 0,0 1,1 1,0 0))",
      "MULTIPOLYGON (((0 0,1 0,0 1,0 0)))"
    },
    {
      "POINT (1 1)", "LINESTRING (1 1,2 2,3 3)", "POLYGON ((5 4,7 4,6 5,5 4))",
      "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"
    },
    {
      "POINT (2 2)", "LINESTRING (2 2,3 3)", "POLYGON ((1 1,3 1,2 3,1 1))",
      "MULTIPOLYGON (((0 0,3 0,0 3,0 0)),((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"
    }},
    result);
  // clang-format on
}

INSTANTIATE_TEST_SUITE_P(RowGroupAndFragmentSizeParameterizedTests,
                         RowGroupAndFragmentSizeSelectQueryTest,
                         ::testing::Values(std::make_pair(1, 1),
                                           std::make_pair(1, 2),
                                           std::make_pair(2, 1),
                                           std::make_pair(2, 2)),
                         PrintToStringParamName());

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, MetadataOnlyCount) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c INTEGER, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT COUNT(*) FROM test_foreign_table;");
  assertResultSetEqual({{i(6)}}, result);
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, MetadataOnlyFilter) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c INTEGER, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  {
    TQueryResult result;
    sql(result, "SELECT COUNT(*) FROM test_foreign_table WHERE a > 2;");
    assertResultSetEqual({{i(4)}}, result);
  }

  {
    TQueryResult result;
    sql(result, "SELECT COUNT(*) FROM test_foreign_table WHERE d < 0;");
    assertResultSetEqual({{i(2)}}, result);
  }
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, Join) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  auto query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c INTEGER, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);
  query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, d DOUBLE)", "example_2", "parquet", 2);
  sql(query);

  TQueryResult result;
  sql(result,
      "SELECT t1.a, t1.b, t1.c, t1.d, t2.i, t2.d FROM "
      "test_foreign_table AS t1 JOIN "
      "test_foreign_table_2 AS t2 ON t1.a = t2.i ORDER BY t1.a;");

  assertResultSetEqual({{i(1), i(3), i(6), 7.1, i(1), 1.1},
                        {i(1), i(3), i(6), 7.1, i(1), 1.1},
                        {i(1), i(3), i(6), 7.1, i(1), 1.1},
                        {i(2), i(4), i(7), 0.000591, i(2), 2.2},
                        {i(2), i(4), i(7), 0.000591, i(2), 2.2},
                        {i(3), i(5), i(8), 1.1, i(3), 3.3}},
                       result);
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, Select) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c INTEGER, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{i(1), i(3), i(6), 7.1},
                        {i(2), i(4), i(7), 0.000591},
                        {i(3), i(5), i(8), 1.1},
                        {i(4), i(6), i(9), 0.022123},
                        {i(5), i(7), i(10), -1.},
                        {i(6), i(8), i(1), -100.}},
                       result);
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, Filter) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c INTEGER, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table WHERE d < 0 ;");
  assertResultSetEqual({{i(5), i(7), i(10), -1.}, {i(6), i(8), i(1), -100.}}, result);
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, NoStatistics) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "no_stats.row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c TEXT, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual(
      {
          {i(1), i(3), "5", 7.1},
          {i(2), i(4), "stuff", NULL_DOUBLE},
          {i(3), i(5), "8", 1.1},
          {i(4), i(6), "9", 2.2123e-2},
          {i(5), i(7), "10", -1.},
          {i(6), i(8), "1", -100.},
      },
      result);
}

using namespace foreign_storage;
class ForeignStorageCacheQueryTest : public ForeignTableTest {
 protected:
  inline static const std::string table_2_filename = "example_2";
  inline static const std::string col_name1 = "col1";
  inline static const std::string col_name2 = "col2";
  inline static const std::string col_name3 = "col3";
  inline static Catalog_Namespace::Catalog* cat;
  inline static ForeignStorageCache* cache;
  inline static File_Namespace::GlobalFileMgr* gfm;
  inline static const TableDescriptor* td;
  inline static const ColumnDescriptor *cd1, *cd2, *cd3;
  inline static ChunkKey query_chunk_key1, query_chunk_key2, query_chunk_key3,
      query_table_prefix;

  static void SetUpTestSuite() {
    DBHandlerTestFixture::SetUpTestSuite();
    cat = &getCatalog();
    cache = cat->getDataMgr().getForeignStorageMgr()->getForeignStorageCache();
    gfm = cache->getGlobalFileMgr();
    sqlDropForeignTable();
  }

  static void TearDownTestSuite() { DBHandlerTestFixture::TearDownTestSuite(); }

  static void createTestTable() {
    sqlCreateForeignTable(
        "(" + col_name1 + " TEXT, " + col_name2 + " INTEGER, " + col_name3 + " DOUBLE)",
        table_2_filename,
        "csv");
    td = cat->getMetadataForTable(default_table_name);
    cd1 = cat->getMetadataForColumn(td->tableId, col_name1);
    cd2 = cat->getMetadataForColumn(td->tableId, col_name2);
    cd3 = cat->getMetadataForColumn(td->tableId, col_name3);
    query_chunk_key1 = {cat->getCurrentDB().dbId, td->tableId, cd1->columnId, 0};
    query_chunk_key2 = {cat->getCurrentDB().dbId, td->tableId, cd2->columnId, 0};
    query_chunk_key3 = {cat->getCurrentDB().dbId, td->tableId, cd3->columnId, 0};
    query_table_prefix = {cat->getCurrentDB().dbId, td->tableId};
  }

  static void sqlSelect(const std::string& columns = "*",
                        const std::string& table_name = "test_foreign_table") {
    sql("SELECT " + columns + " FROM " + table_name + ";");
  }

  void SetUp() override {
    ForeignTableTest::SetUp();
    cache->clear();
    createTestTable();
  }

  void TearDown() override {
    sqlDropForeignTable();
    ForeignTableTest::TearDown();
  }
};

TEST_F(ForeignStorageCacheQueryTest, SelectPopulateChunks) {
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_FALSE(gfm->isBufferOnDevice(query_chunk_key2));
  sqlSelect();
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_TRUE(gfm->isBufferOnDevice(query_chunk_key2));
  AbstractBuffer* buffer = cache->getCachedChunkIfExists(query_chunk_key2);
  ASSERT_NE(buffer, nullptr);
  int8_t array[4];
  buffer->read(array, 4);
  int32_t val = 1;
  ASSERT_EQ(std::memcmp(array, &val, 4), 0);
}

TEST_F(ForeignStorageCacheQueryTest, CreatePopulateMetadata) {
  sqlDropForeignTable();
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key1));
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key2));
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key3));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(query_chunk_key1));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(query_table_prefix));
  createTestTable();
  ASSERT_TRUE(cache->isMetadataCached(query_chunk_key1));
  ASSERT_TRUE(cache->isMetadataCached(query_chunk_key2));
  ASSERT_TRUE(cache->isMetadataCached(query_chunk_key3));
  ASSERT_TRUE(cache->hasCachedMetadataForKeyPrefix(query_chunk_key1));
  ASSERT_TRUE(cache->hasCachedMetadataForKeyPrefix(query_table_prefix));
}

TEST_F(ForeignStorageCacheQueryTest, CacheEvictAfterDrop) {
  sqlSelect();
  ASSERT_EQ(cache->getNumCachedChunks(), 3U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 3U);
  sqlDropForeignTable();
  ASSERT_EQ(cache->getNumCachedChunks(), 0U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 0U);
}

// LRU specific tests (default algorithm).
TEST_F(ForeignStorageCacheQueryTest, LRUEvictChunkByAlgOrderOne) {
  sqlSelect(col_name3);
  sqlSelect(col_name2);
  sqlSelect(col_name1);
  ASSERT_EQ(cache->getNumCachedChunks(), 3U);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  size_t old_limit = cache->getLimit();
  cache->setLimit(2U);
  ASSERT_FALSE(cache->getCachedChunkIfExists(query_chunk_key3));
  ASSERT_TRUE(cache->getCachedChunkIfExists(query_chunk_key2));
  ASSERT_TRUE(cache->getCachedChunkIfExists(query_chunk_key1));
  cache->setLimit(1U);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  cache->setLimit(0U);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  ASSERT_EQ(cache->getNumCachedChunks(), 0U);
  cache->setLimit(old_limit);
}

// Verify that the LRU algorithm is working if we use a different order.
TEST_F(ForeignStorageCacheQueryTest, LRUEvictChunkByAlgOrderTwo) {
  sqlSelect(col_name1);
  sqlSelect(col_name2);
  sqlSelect(col_name3);
  ASSERT_EQ(cache->getNumCachedChunks(), 3U);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  size_t old_limit = cache->getLimit();
  cache->setLimit(2U);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  cache->setLimit(1U);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  cache->setLimit(0U);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  ASSERT_EQ(cache->getNumCachedChunks(), 0U);
  cache->setLimit(old_limit);
}

TEST_F(ForeignStorageCacheQueryTest, WideLogicalColumns) {
  cache->clear();
  ASSERT_EQ(cache->getNumCachedChunks(), 0U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 0U);
  sqlDropForeignTable();
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", "example_1", "csv");
  sqlSelect();
  // Metadata and chunk size differ because the INTEGER[] logical col expands into two
  // physical columns.
  ASSERT_EQ(cache->getNumCachedChunks(), 3U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 2U);
  sqlDropForeignTable();
}

class CacheDefaultTest : public DBHandlerTestFixture {};
TEST_F(CacheDefaultTest, Path) {
  auto cat = &getCatalog();
  auto cache = cat->getDataMgr().getForeignStorageMgr()->getForeignStorageCache();
  ASSERT_EQ(cache->getGlobalFileMgr()->getBasePath(),
            to_string(BASE_PATH) + "/omnisci_disk_cache/");
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
