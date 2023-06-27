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
 * @file ForeignTableDmlTest.cpp
 * @brief Test suite for DML SQL queries on foreign tables
 */

#include <fstream>
#include <regex>
#include <string>

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "Catalog/OptionsContainer.h"
#include "Catalog/RefreshTimeCalculator.h"
#include "DBHandlerTestHelpers.h"
#include "DataMgr/ForeignStorage/DataPreview.h"
#include "DataMgr/ForeignStorage/ForeignStorageCache.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "DataMgr/ForeignStorage/RegexFileBufferParser.h"

#include "Geospatial/Types.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "Shared/StringTransform.h"
#include "Shared/SysDefinitions.h"
#include "Shared/scope.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;
extern bool g_enable_s3_fsi;
extern bool g_enable_seconds_refresh;
extern bool g_allow_s3_server_privileges;
extern std::optional<size_t> g_detect_test_sample_size;

std::string test_binary_file_path;
std::string test_temp_dir;
bool g_run_odbc{false};

namespace bp = boost::process;
namespace bf = boost::filesystem;
namespace po = boost::program_options;

// Typedefs for clarity.
using path = bf::path;
using WrapperType = std::string;
using FragmentSizeType = int32_t;
using ChunkSizeType = int64_t;
using FileNameType = std::string;
using RecoverCacheFlag = bool;
using EvictCacheFlag = bool;
using RowGroupSizeType = int32_t;
using ScheduledRefreshFlag = bool;
using AnnotationType = std::string;
using UseDsnFlag = bool;
using DsnType = std::string;
using FileExtType = std::string;
using EvictCacheString = std::string;
using ImportFlag = bool;
using NameTypePair = std::pair<std::string, std::string>;

// sets of wrappers for parametarized testing
static const std::vector<WrapperType> local_wrappers{"csv",
                                                     "parquet",
                                                     "sqlite",
                                                     "postgres",
                                                     "regex_parser"};
static const std::vector<WrapperType> file_wrappers{"csv", "parquet", "regex_parser"};
static const std::vector<WrapperType> s3_wrappers{"csv",
                                                  "parquet",
                                                  "csv_s3_select",
                                                  "regex_parser"};
static const std::vector<WrapperType> csv_s3_wrappers{"csv", "csv_s3_select"};
static const std::vector<WrapperType> odbc_wrappers{"snowflake",
                                                    "sqlite",
                                                    "postgres",
                                                    "redshift",
                                                    "bigquery"};

static const std::string default_table_name = "test_foreign_table";
static const std::string default_table_name_2 = "test_foreign_table_2";
static const std::string default_file_name = "temp_file";

static const std::string default_select = "SELECT * FROM " + default_table_name + ";";

namespace {
// Needs to be a macro since GTEST_SKIP() breaks by invoking "return".
#define SKIP_SETUP_IF_DISTRIBUTED(msg) \
  if (isDistributedMode()) {           \
    skip_teardown_ = true;             \
    GTEST_SKIP() << msg;               \
  }

#define SKIP_IF_DISTRIBUTED(msg) \
  if (isDistributedMode()) {     \
    GTEST_SKIP() << msg;         \
  }

// These need to be done as macros because GTEST_SKIP() invokes "return" in the function
// it is called in as well as setting IsSkipped().
#define SKIP_SETUP_IF_ODBC_DISABLED() \
  GTEST_SKIP() << "ODBC tests not supported with this build configuration."

bool is_regex(const std::string& wrapper_type) {
  return (wrapper_type == "regex_parser");
}

std::string wrapper_file_type(const std::string& wrapper_type) {
  return (DBHandlerTestFixture::isOdbc(wrapper_type) || wrapper_type == "regex_parser")
             ? "csv"
             : wrapper_type;
}

FileExtType wrapper_ext(const std::string& wrapper_type) {
  return "." + wrapper_file_type(wrapper_type);
}

void recursive_copy(const std::string& origin, const std::string& dest) {
  bf::create_directory(dest);
  for (bf::directory_iterator file(origin); file != bf::directory_iterator(); ++file) {
    const auto& path = file->path();
    if (bf::is_directory(path)) {
      recursive_copy(path.string(), dest + "/" + path.filename().string());
    } else {
      bf::copy_file(path.string(), dest + "/" + path.filename().string());
    }
  }
}

bool does_cache_contain_chunks(Catalog_Namespace::Catalog* cat,
                               const std::string& table_name,
                               const std::vector<std::vector<int>>& subkeys) {
  // subkey is chunkey without db, table ids
  auto td = cat->getMetadataForTable(table_name, false);
  ChunkKey table_key{cat->getCurrentDB().dbId, td->tableId};
  auto cache = cat->getDataMgr().getPersistentStorageMgr()->getDiskCache();

  for (const auto& subkey : subkeys) {
    auto chunk_key = table_key;
    chunk_key.insert(chunk_key.end(), subkey.begin(), subkey.end());
    if (cache->getCachedChunkIfExists(chunk_key) == nullptr) {
      return false;
    }
  }
  return true;
}

// compare files, adjusting for basepath
bool compare_json_files(const std::string& generated,
                        const std::string& reference,
                        const std::string& basepath) {
  std::ifstream gen_file(generated);
  std::ifstream ref_file(reference);
  // Compare each file line by line
  while (gen_file && ref_file) {
    std::string gen_line;
    std::getline(gen_file, gen_line);
    std::string ref_line;
    std::getline(ref_file, ref_line);
    boost::replace_all(gen_line, basepath, "BASEPATH/");
    boost::algorithm::trim(gen_line);
    boost::algorithm::trim(ref_line);
    if (gen_line.compare(ref_line) != 0) {
      std::cout << "Mismatched json lines \n";
      std::cout << gen_line << "\n";
      std::cout << ref_line << "\n";
      return false;
    }
  }
  if (ref_file || gen_file) {
    std::cerr << "# of lines mismatch\n";
    std::cerr << generated << " vs " << reference << "\n";
    // # of lines mismatch
    return false;
  }
  return true;
}

std::string repeat_regex(size_t repeat_count, const std::string& regex) {
  std::string repeated_regex;
  for (size_t i = 0; i < repeat_count; i++) {
    if (!repeated_regex.empty()) {
      repeated_regex += "\\s*,\\s*";
    }
    repeated_regex += regex;
  }
  return repeated_regex;
}

std::string get_line_regex(size_t column_count) {
  return repeat_regex(column_count, "\"?([^,\"]*)\"?");
}

std::string get_line_array_regex(size_t column_count) {
  return repeat_regex(column_count, "(\\{[^\\}]+\\}|NULL|)");
}

std::string get_line_geo_regex(size_t column_count) {
  return repeat_regex(column_count,
                      "\"?((?:POINT|LINESTRING|POLYGON|MULTIPOLYGON)[^\"]+|\\\\N)\"?");
}

std::string get_default_server(const std::string& data_wrapper_type) {
  std::string suffix;
  if (data_wrapper_type == "parquet") {
    suffix = "parquet";
  } else if (data_wrapper_type == "csv") {
    suffix = "delimited";
  } else if (data_wrapper_type == "regex_parser") {
    suffix = "regex_parsed";
  } else {
    UNREACHABLE() << "Unexpected default server data wrapper type: " << data_wrapper_type;
  }
  return "default_local_" + suffix;
}

std::string get_data_wrapper_name(const std::string& data_wrapper_type) {
  std::string data_wrapper;
  if (is_regex(data_wrapper_type)) {
    data_wrapper = "regex_parsed_file";
  } else if (data_wrapper_type == "csv" || data_wrapper_type == "csv_s3_select") {
    data_wrapper = "delimited_file";
  } else if (data_wrapper_type == "parquet") {
    data_wrapper = "parquet_file";
  } else if (DBHandlerTestFixture::isOdbc(data_wrapper_type)) {
    data_wrapper = "odbc";
  } else {
    UNREACHABLE() << "Unexpected data wrapper type: " << data_wrapper_type;
  }
  return data_wrapper;
}
}  // namespace

/**
 * Helper base class that creates and maintains a temporary directory
 */
class TempDirManager {
 public:
  TempDirManager() {
    bf::remove_all(test_temp_dir);
    bf::create_directory(test_temp_dir);
  }

  ~TempDirManager() { bf::remove_all(test_temp_dir); }

  static void overwriteTempDir(const std::string& source_path) {
    bf::remove_all(test_temp_dir);
    recursive_copy(source_path, test_temp_dir);
  }
};

/**
 * Helper class for creating foreign tables
 */
class ForeignTableTest : public DBHandlerTestFixture {
 protected:
  inline static const std::string DEFAULT_ODBC_SERVER_NAME_ = "temp_odbc";

  std::string wrapper_type_ = "csv";
  bool skip_teardown_ = false;

  void SetUp() override {
    if (isOdbc(wrapper_type_)) {
      SKIP_SETUP_IF_ODBC_DISABLED();
    }
    g_enable_fsi = true;
    DBHandlerTestFixture::SetUp();
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    g_enable_fsi = true;
    DBHandlerTestFixture::TearDown();
  }

  static std::string getCreateForeignTableQuery(const std::string& columns,
                                                const std::string& file_name_base,
                                                const std::string& data_wrapper_type,
                                                const int table_number = 0) {
    return getCreateForeignTableQuery(
        columns, {}, file_name_base, data_wrapper_type, table_number);
  }

  static std::string getCreateForeignTableQuery(
      const std::string& columns,
      const foreign_storage::OptionsMap& options,
      const std::string& file_name_base,
      const std::string& data_wrapper_type,
      const int table_number = 0,
      const std::string& table_name = default_table_name,
      const std::string extension = "",
      const std::string& source_dir = getDataFilesPath()) {
    std::string query{"CREATE FOREIGN TABLE " + table_name};
    if (table_number) {
      query += "_" + std::to_string(table_number);
    }

    std::string filename = file_name_base;
    if (extension == "dir") {
      filename += "_" + data_wrapper_type + "_dir";
    } else if (extension.empty()) {
      filename += "." + data_wrapper_type;
    } else {
      filename += "." + extension;
    }

    query += " " + columns + " SERVER " + get_default_server(data_wrapper_type) +
             " WITH (file_path = '" + source_dir + filename + "'";
    for (auto& [key, value] : options) {
      query += ", " + key + " = '" + value + "'";
    }

    // If this is a regex wrapper then we should skip the header.
    if (is_regex(data_wrapper_type)) {
      if (options.find("HEADER") == options.end()) {
        query += ", HEADER = 'TRUE'";
      }
    }
    query += ");";
    return query;
  }

  static void createUserMappingForDsn(const std::string& server_name,
                                      const std::string& username,
                                      const std::string& password) {
    sql("CREATE USER MAPPING FOR PUBLIC SERVER " + server_name + " WITH (username='" +
        username + "', password='" + password + "');");
  }

  static void createUserMappingForCs(const std::string& server_name,
                                     const std::map<std::string, std::string>& pairs,
                                     const std::string& connection_string_suffix = "") {
    CHECK(!pairs.empty());
    auto statement = "CREATE USER MAPPING FOR PUBLIC SERVER " + server_name +
                     " WITH (credential_string='";
    for (auto const& [key, value] : pairs) {
      statement += key + "=" + value + ";";
    }
    statement.pop_back();
    statement += connection_string_suffix + "');";
    sql(statement);
  }

  static void createUserMappingForOdbc(const std::string& server_name,
                                       const std::map<std::string, std::string>& pairs,
                                       const bool use_dsn,
                                       const std::string& connection_string_suffix = "") {
    if (use_dsn) {
      auto username_it = pairs.find("USERNAME");
      auto password_it = pairs.find("PASSWORD");
      CHECK(username_it != pairs.end());
      CHECK(password_it != pairs.end());
      createUserMappingForDsn(server_name, username_it->second, password_it->second);
    } else {
      createUserMappingForCs(server_name, pairs, connection_string_suffix);
    }
  }

  static void createUserMappingForS3(const std::string& server_name,
                                     const std::string& access_key,
                                     const std::string& secret_key,
                                     const std::string session_token = "") {
    sql("CREATE USER MAPPING FOR PUBLIC SERVER " + server_name +
        " WITH (s3_access_key='" + access_key + "', s3_secret_key='" + secret_key +
        (session_token.empty() ? "" : "', s3_session_token='" + session_token) + "');");
  }

  /**
   * Returns a query to create a foreign table.  Creates a source odbc table for odbc
   * datawrappers.
   */
  static std::string createForeignTableQuery(
      const std::vector<NameTypePair>& column_pairs,
      const std::string& src_path,
      const std::string& data_wrapper_type,
      const foreign_storage::OptionsMap options = {},
      const std::string& table_name = default_table_name,
      const std::vector<NameTypePair>& db_specific_column_pairs = {},
      const int order_by_column_index = -1) {
    std::stringstream ss;
    ss << "CREATE FOREIGN TABLE " << table_name << " (";
    ss << column_pairs_to_schema_string(column_pairs) << ") ";
    const auto& stored_column_pairs =
        (db_specific_column_pairs.empty()) ? column_pairs : db_specific_column_pairs;

    if (isOdbc(data_wrapper_type)) {
      createODBCSourceTable(table_name, stored_column_pairs, src_path, data_wrapper_type);
    } else {
      ss << "SERVER " + get_default_server(data_wrapper_type);
      ss << " WITH (file_path = '";
      ss << src_path << "'";
    }
    if (data_wrapper_type == "regex_parser") {
      if (options.find("LINE_REGEX") == options.end()) {
        ss << ", LINE_REGEX = '" + get_line_regex(column_pairs.size()) + "'";
      }
      if (options.find("HEADER") == options.end()) {
        ss << ", HEADER = 'TRUE'";
      }
    }
    for (auto& [key, value] : options) {
      ss << ", " << key << " = '" << value << "'";
    }
    ss << ");";
    return ss.str();
  }

  static std::string getDataFilesPath() {
    return bf::canonical(test_binary_file_path + "/../../Tests/FsiDataFiles").string() +
           "/";
  }

  static void sqlCreateForeignTable(const std::string& columns,
                                    const std::string& file_name,
                                    const std::string& data_wrapper_type,
                                    const foreign_storage::OptionsMap options = {},
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
    const TableDescriptor* fd = cat.getMetadataForTable(table_name, false);
    ChunkKey key{cat.getCurrentDB().dbId, fd->tableId};
    for (auto i : key_suffix) {
      key.push_back(i);
    }
    return key;
  }

  void queryAndAssertFileNotFoundException(const std::string& file_path,
                                           const std::string& query = "SELECT * FROM " +
                                                                      default_table_name +
                                                                      ";") {
    queryAndAssertException(query,
                            "File or directory \"" + file_path + "\" does not exist.");
  }

  void queryAndAssertExample2Result() {
    std::string query = "SELECT * FROM " + default_table_name + " ORDER BY t, i;";
    TQueryResult result;
    sql(result, query);
    assertResultSetEqual({{"a", i(1), 1.1},
                          {"aa", i(1), 1.1},
                          {"aa", i(2), 2.2},
                          {"aaa", i(1), 1.1},
                          {"aaa", i(2), 2.2},
                          {"aaa", i(3), 3.3}},
                         result);
  }

  void queryAndAssertExample2Count() {
    TQueryResult result;
    sql(result, "SELECT COUNT(*) FROM " + default_table_name + ";");
    assertResultSetEqual({{i(6)}}, result);
  }

  std::vector<std::vector<NullableTargetValue>> getExpectedScalarTypesResult(
      bool allow_coercion) {
    // clang-format off
    std::vector<std::vector<NullableTargetValue>> expected{
        {
         True, i(100), i(30000), i(2000000000), i(9000000000000000000), 10.1f,
         100.1234, "00:00:10", "1/1/2000 00:00:59", "1/1/2000", "text_1", "quoted text"
        },
        {
         False, i(110), i(30500), i(2000500000), i(9000000050000000000), 100.12f,
         2.1234, "00:10:00", "6/15/2020 00:59:59", "6/15/2020", "text_2", "quoted text 2"
        },
        {
         True, i(120), i(31000), i(2100000000), i(9100000000000000000),
         (wrapper_type_ == "redshift" ? 1000.12f : 1000.123f),
         100.1, "10:00:00", "12/31/2500 23:59:59", "12/31/2500", "text_3", "quoted text 3"
        },
        {
         i(NULL_BOOLEAN),
         ((!allow_coercion && wrapper_type_ == "postgres") ? i(NULL_SMALLINT) : i(NULL_TINYINT)), // TINYINT
         i(NULL_SMALLINT), i(NULL_INT), i(NULL_BIGINT),
         ((!allow_coercion && wrapper_type_ == "sqlite") ? NULL_DOUBLE : NULL_FLOAT), // FLOAT
         NULL_DOUBLE, Null, Null, Null, Null, Null
        }
      };
    // clang-format on
    return expected;
  }

  void queryAndAssertScalarTypesResult(bool allow_coercion = false) {
    sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY s;",
                        getExpectedScalarTypesResult(allow_coercion));
  }

  void queryAndAssertGeoTypesResult() {
    sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY id;",
                        getExpectedGeoTypesResult());
  }

  std::vector<std::vector<NullableTargetValue>> getExpectedGeoTypesResult() {
    // clang-format off
    return {
      {
        i(1), "POINT (0 0)", "LINESTRING (0 0,1 1)", "POLYGON ((0 0,1 0,1 1,0 1,0 0))",
        "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((2 2,3 2,2 3,2 2)))"
      },
      {
        i(2), Null, Null, Null, Null
      },
      {
        i(3), "POINT (1 1)", "LINESTRING (1 1,2 2,3 3)", "POLYGON ((5 4,7 4,6 5,5 4))",
        "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((2 2,3 2,2 3,2 2),(2.1 2.1,2.1 2.9,2.9 2.1,2.1 2.1)))"
      },
      {
        i(4), "POINT (2 2)", "LINESTRING (2 2,3 3)", "POLYGON ((1 1,3 1,2 3,1 1))",
        "MULTIPOLYGON (((5 5,8 8,5 8,5 5)),((0 0,3 0,0 3,0 0)),((11 11,10 12,10 10,11 11)))"
      },
      {
        i(5), Null, Null, Null, Null
      }
    };
    // clang-format on
  }

  void createForeignTableForGeoTypes(const std::string& data_wrapper_type,
                                     const std::string& extension,
                                     size_t fragment_size = DEFAULT_FRAGMENT_ROWS) {
    // geotypes in odbc data wrappers are currently only supported by text data types
    std::vector<NameTypePair> odbc_columns{};
    if (isOdbc(data_wrapper_type)) {
      odbc_columns = {{"id", "INT"},
                      {"p", "TEXT"},
                      {"l", "TEXT"},
                      {"poly", "TEXT"},
                      {"multipoly", "TEXT"}};
    }
    foreign_storage::OptionsMap options{{"FRAGMENT_SIZE", std::to_string(fragment_size)}};
    if (data_wrapper_type == "regex_parser") {
      options["LINE_REGEX"] = "(\\d+),\\s*" + get_line_geo_regex(4);
    }
    sql(createForeignTableQuery({{"id", "INT"},
                                 {"p", "POINT"},
                                 {"l", "LINESTRING"},
                                 {"poly", "POLYGON"},
                                 {"multipoly", "MULTIPOLYGON"}},
                                getDataFilesPath() + "geo_types_valid" + extension,
                                data_wrapper_type,
                                options,
                                default_table_name,
                                odbc_columns,
                                isOdbc(data_wrapper_type) ? true : false));
  }

  void queryAndAssertQuotedIdentifierPairs() {
    TQueryResult result;
    // clang-format off
    sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY id;",{
      {1L}
      });
    // clang-format on
  }
};

class SelectQueryTest : public ForeignTableTest {
 protected:
  void SetUp() override {
    ForeignTableTest::SetUp();
    import_export::delimited_parser::set_max_buffer_resize(max_buffer_resize_);
    sqlDropForeignTable();
    sqlDropForeignTable(0, default_table_name_2);
    sql("DROP SERVER IF EXISTS test_server;");
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    g_enable_fsi = true;
    sqlDropForeignTable();
    sqlDropForeignTable(0, default_table_name_2);
    sql("DROP SERVER IF EXISTS test_server;");
    ForeignTableTest::TearDown();
  }

  template <typename T>
  inline static std::unique_ptr<ChunkMetadata> createChunkMetadata(
      const int column_id,
      const size_t num_bytes,
      const size_t num_elements,
      const T& min,
      const T& max,
      bool has_nulls,
      const std::string& table_name) {
    auto chunk_metadata =
        createChunkMetadata(column_id, num_bytes, num_elements, has_nulls, table_name);
    if (chunk_metadata->sqlType.is_array()) {
      auto saved_sql_type = chunk_metadata->sqlType;
      chunk_metadata->sqlType = saved_sql_type.get_elem_type();
      chunk_metadata->fillChunkStats(min, max, has_nulls);
      chunk_metadata->sqlType = saved_sql_type;
    } else {
      chunk_metadata->fillChunkStats(min, max, has_nulls);
    }
    return chunk_metadata;
  }

  template <typename T>
  inline static std::unique_ptr<ChunkMetadata> createChunkMetadata(
      const int column_id,
      const size_t num_bytes,
      const size_t num_elements,
      const T& min,
      const T& max,
      bool has_nulls) {
    return createChunkMetadata(
        column_id, num_bytes, num_elements, min, max, has_nulls, default_table_name);
  }

  inline static std::unique_ptr<ChunkMetadata> createChunkMetadata(
      const int column_id,
      const size_t num_bytes,
      const size_t num_elements,
      bool has_nulls,
      const std::string& table_name) {
    auto& cat = getCatalog();
    auto foreign_table = cat.getMetadataForTable(table_name, false);
    auto column_descriptor = cat.getMetadataForColumn(foreign_table->tableId, column_id);
    auto chunk_metadata = std::make_unique<ChunkMetadata>();
    chunk_metadata->sqlType = column_descriptor->columnType;
    chunk_metadata->numElements = num_elements;
    chunk_metadata->numBytes = num_bytes;
    chunk_metadata->chunkStats.has_nulls = has_nulls;
    chunk_metadata->chunkStats.min.stringval = nullptr;
    chunk_metadata->chunkStats.max.stringval = nullptr;
    return chunk_metadata;
  }

  inline static std::unique_ptr<ChunkMetadata> createChunkMetadata(
      const int column_id,
      const size_t num_bytes,
      const size_t num_elements,
      bool has_nulls) {
    return createChunkMetadata(
        column_id, num_bytes, num_elements, has_nulls, default_table_name);
  }

  void assertExpectedChunkMetadata(
      const std::map<std::pair<int, int>, std::unique_ptr<ChunkMetadata>>&
          expected_metadata) const {
    assertExpectedChunkMetadata(expected_metadata, default_table_name);
  }

  void assertExpectedChunkMetadata(
      const std::map<std::pair<int, int>, std::unique_ptr<ChunkMetadata>>&
          expected_metadata,
      const std::string& table_name) const {
    auto& cat = getCatalog();
    auto foreign_table = cat.getMetadataForTable(table_name, false);
    if (!foreign_table) {
      throw std::runtime_error("Could not find foreign table: " + table_name);
    }
    auto fragmenter = foreign_table->fragmenter;
    if (!fragmenter) {
      throw std::runtime_error("Fragmenter does not exist for foreign table: " +
                               table_name);
    }
    std::map<std::pair<int, int>, bool> expected_metadata_found;
    for (auto& [k, v] : expected_metadata) {
      expected_metadata_found[k] = false;
    }

    auto query_info = fragmenter->getFragmentsForQuery();
    for (const auto& fragment : query_info.fragments) {
      auto& chunk_metadata_map = fragment.getChunkMetadataMapPhysical();
      for (auto& [col_id, chunk_metadata] : chunk_metadata_map) {
        auto fragment_id = fragment.fragmentId;
        auto column_id = col_id;
        auto fragment_column_ids = std::make_pair(fragment_id, column_id);
        auto expected_metadata_iter = expected_metadata.find(fragment_column_ids);
        EXPECT_NE(expected_metadata_iter, expected_metadata.end())
            << boost::format(
                   "Foreign table chunk metadata not found in expected metadata: "
                   "fragment_id: %d, column_id: %d") %
                   fragment_id % column_id;
        expected_metadata_found[fragment_column_ids] = true;
        EXPECT_EQ(*chunk_metadata, *expected_metadata_iter->second)
            << (boost::format("At fragment_id: %d, column_id: %d") % fragment_id %
                column_id)
            << " Expected: " << *expected_metadata_iter->second
            << ", Found: " << *chunk_metadata;
      }
    }

    for (auto& [k, v] : expected_metadata_found) {
      auto fragment_id = k.first;
      auto column_id = k.second;
      if (!v) {
        ASSERT_TRUE(false) << boost::format(
                                  "Expected chunk metadata not found in foreign table "
                                  "metadata: fragment_id: %d, column_id: %d") %
                                  fragment_id % column_id;
      }
    }
  }

  inline static size_t max_buffer_resize_ =
      import_export::delimited_parser::get_max_buffer_resize();
};

class CacheControllingSelectQueryBaseTest : public SelectQueryTest {
 public:
  inline static std::string cache_path_ =
      to_string(BASE_PATH) + "/" + shared::kDefaultDiskCacheDirName;
  std::optional<File_Namespace::DiskCacheLevel> starting_cache_level_;
  File_Namespace::DiskCacheLevel cache_level_;

  CacheControllingSelectQueryBaseTest(const File_Namespace::DiskCacheLevel& cache_level)
      : cache_level_(cache_level) {}

 protected:
  void resetPersistentStorageMgr(File_Namespace::DiskCacheLevel cache_level) {
    for (auto table_it : getCatalog().getAllTableMetadata()) {
      getCatalog().removeFragmenterForTable(table_it->tableId);
    }
    getCatalog().getDataMgr().resetPersistentStorage(
        {cache_path_, cache_level}, 0, getSystemParameters());
  }

  void SetUp() override {
    // Distributed mode doens't handle the resetPersistentStorageMgr appropriately as the
    // leaves don't have a way of being updated, so we skip these tests.
    SKIP_SETUP_IF_DISTRIBUTED("Test relies on disk cache");
    // Disable/enable the cache as test param requires
    starting_cache_level_ = getCatalog()
                                .getDataMgr()
                                .getPersistentStorageMgr()
                                ->getDiskCacheConfig()
                                .enabled_level;
    if (starting_cache_level_ && (*starting_cache_level_ != cache_level_)) {
      resetPersistentStorageMgr(cache_level_);
    }
    SelectQueryTest::SetUp();
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    SelectQueryTest::TearDown();
    // Reset cache to pre-test conditions
    if (starting_cache_level_ && (*starting_cache_level_ != cache_level_)) {
      resetPersistentStorageMgr(*starting_cache_level_);
    }
  }
};

class CacheControllingSelectQueryTest
    : public CacheControllingSelectQueryBaseTest,
      public ::testing::WithParamInterface<File_Namespace::DiskCacheLevel> {
 public:
  CacheControllingSelectQueryTest() : CacheControllingSelectQueryBaseTest(GetParam()) {}
};

class RecoverCacheQueryTest : public ForeignTableTest {
 public:
  inline static std::string cache_path_ =
      to_string(BASE_PATH) + "/" + shared::kDefaultDiskCacheDirName;
  Catalog_Namespace::Catalog* cat_;
  PersistentStorageMgr* psm_;
  foreign_storage::ForeignStorageCache* cache_ = nullptr;

 protected:
  void resetPersistentStorageMgr(File_Namespace::DiskCacheConfig cache_config) {
    for (auto table_it : cat_->getAllTableMetadata()) {
      cat_->removeFragmenterForTable(table_it->tableId);
    }
    cat_->getDataMgr().resetPersistentStorage(cache_config, 0, getSystemParameters());
    psm_ = cat_->getDataMgr().getPersistentStorageMgr();
    cache_ = psm_->getDiskCache();
  }

  bool isTableDatawrapperRestored(const std::string& name) {
    auto td = getCatalog().getMetadataForTable(name, false);
    ChunkKey table_key{getCatalog().getCurrentDB().dbId, td->tableId};
    return getCatalog()
        .getDataMgr()
        .getPersistentStorageMgr()
        ->getForeignStorageMgr()
        ->isDatawrapperRestored(table_key);
  }

  bool isTableDatawrapperDataOnDisk(const std::string& name) {
    auto td = getCatalog().getMetadataForTable(name, false);
    auto db_id = getCatalog().getCurrentDB().dbId;
    ChunkKey table_key{db_id, td->tableId};
    return bf::exists(getCatalog()
                          .getDataMgr()
                          .getPersistentStorageMgr()
                          ->getDiskCache()
                          ->getSerializedWrapperPath(db_id, td->tableId));
  }

  bool compareTableDatawrapperMetadataToFile(const std::string& name,
                                             const std::string& filepath) {
    auto td = getCatalog().getMetadataForTable(name, false);
    auto db_id = getCatalog().getCurrentDB().dbId;
    ChunkKey table_key{db_id, td->tableId};
    return compare_json_files(getCatalog()
                                  .getDataMgr()
                                  .getPersistentStorageMgr()
                                  ->getDiskCache()
                                  ->getSerializedWrapperPath(db_id, td->tableId),
                              filepath,
                              getDataFilesPath());
  }

  void resetStorageManagerAndClearTableMemory(const ChunkKey& table_key) {
    // Reset cache and clear memory representations.
    resetPersistentStorageMgr({cache_path_, File_Namespace::DiskCacheLevel::fsi});
    cat_->getDataMgr().deleteChunksWithPrefix(table_key, MemoryLevel::CPU_LEVEL);
    cat_->getDataMgr().deleteChunksWithPrefix(table_key, MemoryLevel::GPU_LEVEL);
  }

  std::string getWrapperMetadataPath(const std::string& prefix,
                                     const std::string& data_wrapper_type = {}) {
    std::string path = getDataFilesPath() + "/wrapper_metadata/" + prefix;
    if (!data_wrapper_type.empty()) {
      path += "_" + data_wrapper_type;
    }
    return path + ".json";
  }

  ChunkKey getTestTableKey() {
    auto& catalog = getCatalog();
    auto td = catalog.getMetadataForTable(default_table_name, false);
    CHECK(td);
    return ChunkKey{catalog.getDatabaseId(), td->tableId};
  }

  void setOldDataWrapperMetadata(const std::string& table_name,
                                 const std::string& file_name_prefix) {
    auto& catalog = getCatalog();
    auto disk_cache = catalog.getDataMgr().getPersistentStorageMgr()->getDiskCache();
    ASSERT_NE(disk_cache, nullptr);
    auto db_id = catalog.getDatabaseId();
    auto td = getCatalog().getMetadataForTable(table_name, false);
    ASSERT_NE(td, nullptr);
    auto wrapper_metadata_path = disk_cache->getSerializedWrapperPath(db_id, td->tableId);
    ASSERT_TRUE(boost::filesystem::exists(wrapper_metadata_path));

    auto file_name_suffix = is_regex(wrapper_type_) ? "csv" : wrapper_type_;
    auto prefix = (boost::filesystem::path("old") / file_name_prefix).string();
    auto old_wrapper_metadata_path = getWrapperMetadataPath(prefix, file_name_suffix);
    ASSERT_TRUE(boost::filesystem::exists(old_wrapper_metadata_path));

    // Write content from the old wrapper metadata test file, replacing "BASEPATH" with
    // the actual base path value.
    boost::filesystem::remove(wrapper_metadata_path);
    std::ofstream new_file{wrapper_metadata_path};
    std::ifstream old_file{old_wrapper_metadata_path};
    std::string line;
    while (std::getline(old_file, line)) {
      static std::regex base_path_regex{"BASEPATH"};
      new_file << std::regex_replace(line, base_path_regex, getDataFilesPath());
    }
  }

  void SetUp() override {
    ForeignTableTest::SetUp();
    cat_ = &getCatalog();
    psm_ = cat_->getDataMgr().getPersistentStorageMgr();
    cache_ = psm_->getDiskCache();
    sqlDropForeignTable();
    cache_->clear();
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    sqlDropForeignTable();
    if (cache_) {
      // Cache may not exist if SetUp() was skipped.
      cache_->clear();
    }
    ForeignTableTest::TearDown();
  }
};

class DataTypeFragmentSizeAndDataWrapperTest
    : public SelectQueryTest,
      public testing::WithParamInterface<
          std::tuple<FragmentSizeType, WrapperType, FileExtType>> {
 public:
  FragmentSizeType fragment_size_;
  FileExtType extension_;

  static std::string getTestName(
      const ::testing::TestParamInfo<std::tuple<int32_t, std::string, std::string>>&
          info) {
    auto [fragment_size, wrapper_type, file_ext] = info.param;
    std::replace(file_ext.begin(), file_ext.end(), '.', '_');
    std::stringstream ss;
    ss << "FragmentSize_" << fragment_size << "_DataWrapper_" << wrapper_type
       << "_FileExt" << file_ext;
    return ss.str();
  }

  void SetUp() override {
    std::tie(fragment_size_, wrapper_type_, extension_) = GetParam();
    SelectQueryTest::SetUp();
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    SelectQueryTest::TearDown();
  }

  std::map<std::pair<int, int>, std::unique_ptr<ChunkMetadata>>
  getExpectedScalarTypeMetadata(bool data_loaded) {
    std::map<std::pair<int, int>, std::unique_ptr<ChunkMetadata>> test_chunk_metadata_map;

    size_t num_elems = 4;

    // sqlite does not encode NULL boolean
    if (wrapper_type_ == "sqlite" && !data_loaded) {
      // sqlite does not cast boolean to integer correctly
      test_chunk_metadata_map[{0, 1}] =
          createChunkMetadata<int8_t>(1, num_elems, num_elems, 0, 0, true);
    } else {
      test_chunk_metadata_map[{0, 1}] =
          createChunkMetadata<int8_t>(1, num_elems, num_elems, 0, 1, true);
    }
    if (wrapper_type_ == "postgres") {
      // Postgres does not support tinyint
      test_chunk_metadata_map[{0, 2}] =
          createChunkMetadata<int16_t>(2, num_elems * 2, num_elems, 100, 120, true);
    } else {
      test_chunk_metadata_map[{0, 2}] =
          createChunkMetadata<int8_t>(2, num_elems, num_elems, 100, 120, true);
    }
    test_chunk_metadata_map[{0, 3}] =
        createChunkMetadata<int16_t>(3, num_elems * 2, num_elems, 30000, 31000, true);
    test_chunk_metadata_map[{0, 4}] = createChunkMetadata<int32_t>(
        4, num_elems * 4, num_elems, 2000000000, 2100000000, true);
    test_chunk_metadata_map[{0, 5}] = createChunkMetadata<int64_t>(
        5, num_elems * 8, num_elems, 9000000000000000000, 9100000000000000000, true);
    if (wrapper_type_ == "sqlite") {
      // sqlite does not support decimal or float
      test_chunk_metadata_map[{0, 6}] =
          createChunkMetadata<double>(6, num_elems * 8, num_elems, 10.1, 1000.123, true);
      test_chunk_metadata_map[{0, 7}] = createChunkMetadata<double>(
          7, num_elems * 8, num_elems, 2.1234, 100.1234, true);
    } else {
      test_chunk_metadata_map[{0, 6}] =
          createChunkMetadata<float>(6, num_elems * 4, num_elems, 10.1f, 1000.123f, true);
      test_chunk_metadata_map[{0, 7}] = createChunkMetadata<int64_t>(
          7, num_elems * 8, num_elems, 212340, 10012340, true);
    }
    test_chunk_metadata_map[{0, 8}] =
        createChunkMetadata<int64_t>(8, num_elems * 8, num_elems, 10, 10 * 60 * 60, true);
    test_chunk_metadata_map[{0, 9}] = createChunkMetadata<int64_t>(
        9, num_elems * 8, num_elems, 946684859, 16756761599, true);
    test_chunk_metadata_map[{0, 10}] = createChunkMetadata<int64_t>(
        10, num_elems * 4, num_elems, 946684800, 16756675200, true);
    // encoded string
    if (data_loaded) {
      test_chunk_metadata_map[{0, 11}] =
          createChunkMetadata<int64_t>(11, num_elems * 4, num_elems, 0, 2, true);
    } else {
      test_chunk_metadata_map[{0, 11}] = createChunkMetadata<int64_t>(
          11, num_elems * 4, num_elems, 2147483647, -2147483648, false);
    }
    // unencoded string
    if (data_loaded || wrapper_type_ == "csv" || is_regex(wrapper_type_)) {
      test_chunk_metadata_map[{0, 12}] = createChunkMetadata(12, 37, num_elems, true);
    } else {
      test_chunk_metadata_map[{0, 12}] = createChunkMetadata(12, 0, num_elems, true);
    }
    return test_chunk_metadata_map;
  }

  std::string fragmentSizeStr() { return std::to_string(fragment_size_); }
};

class RowGroupAndFragmentSizeSelectQueryTest
    : public SelectQueryTest,
      public ::testing::WithParamInterface<
          std::pair<RowGroupSizeType, FragmentSizeType>> {
 public:
  RowGroupSizeType row_group_size_;
  FragmentSizeType fragment_size_;

  void SetUp() override {
    std::tie(row_group_size_, fragment_size_) = GetParam();
    SelectQueryTest::SetUp();
  }

  std::string rowGroupSizeStr() { return std::to_string(row_group_size_); }

  std::string fragmentSizeStr() { return std::to_string(fragment_size_); }
};

TEST_P(CacheControllingSelectQueryTest, CustomServer) {
  sql("CREATE SERVER test_server FOREIGN DATA WRAPPER delimited_file "s +
      "WITH (storage_type = 'LOCAL_FILE', base_path = '" + getDataFilesPath() + "');");
  sql("CREATE FOREIGN TABLE " + default_table_name +
      " (t TEXT, i INTEGER[]) "
      "SERVER test_server WITH (file_path = 'example_1.csv');");
  TQueryResult result;
  sql(result, default_select);
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, DefaultLocalCsvServer) {
  std::string query = "CREATE FOREIGN TABLE " + default_table_name +
                      " (t TEXT, i INTEGER[]) "s +
                      "SERVER default_local_delimited WITH (file_path = '" +
                      getDataFilesPath() + "/example_1.csv');";
  sql(query);
  TQueryResult result;
  sql(result, default_select);
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, DefaultLocalParquetServer) {
  std::string query = "CREATE FOREIGN TABLE " + default_table_name +
                      " (t TEXT, i BIGINT, f DOUBLE) "s +
                      "SERVER default_local_parquet WITH (file_path = '" +
                      getDataFilesPath() + "/example_2.parquet');";
  sql(query);
  queryAndAssertExample2Result();
}

// Create table with multiple fragments with file buffers less than size of a
// fragment Includes both fixed and variable length data
TEST_P(CacheControllingSelectQueryTest, MultipleDataBlocksPerFragment) {
  const auto& query = getCreateForeignTableQuery(
      "(i INTEGER,  txt TEXT, txt_2 TEXT ENCODING NONE, txt_arr TEXT[])",
      {{"buffer_size", "25"}, {"fragment_size", "64"}},
      "0_255",
      "csv");
  sql(query);

  // Check that data is correct
  {
    std::vector<std::vector<NullableTargetValue>> expected_result_set;
    for (int number = 0; number < 256; number++) {
      expected_result_set.push_back({i(number),
                                     std::to_string(number),
                                     std::to_string(number),
                                     array({std::to_string(number)})});
    }
    TQueryResult result;
    sql(result, "SELECT * FROM " + default_table_name + " ORDER BY i;");
    assertResultSetEqual(expected_result_set, result);
  }

  // Check that WHERE statements filter numerical data correctly
  {
    std::vector<std::vector<NullableTargetValue>> expected_result_set;
    for (int number = 128; number < 256; number++) {
      expected_result_set.push_back({i(number),
                                     std::to_string(number),
                                     std::to_string(number),
                                     array({std::to_string(number)})});
    }
    TQueryResult result;
    sql(result, "SELECT * FROM " + default_table_name + "  WHERE i >= 128 ORDER BY i;");
    assertResultSetEqual(expected_result_set, result);
  }
  {
    std::vector<std::vector<NullableTargetValue>> expected_result_set;
    for (int number = 0; number < 128; number++) {
      expected_result_set.push_back({i(number),
                                     std::to_string(number),
                                     std::to_string(number),
                                     array({std::to_string(number)})});
    }
    TQueryResult result;
    sql(result, "SELECT * FROM " + default_table_name + "  WHERE i < 128 ORDER BY i;");
    assertResultSetEqual(expected_result_set, result);
  }
}

TEST_P(CacheControllingSelectQueryTest, ParquetNullRowgroups) {
  const auto& query =
      getCreateForeignTableQuery("(a SMALLINT, b SMALLINT)", "null_columns", "parquet");
  sql(query);

  TQueryResult result;
  sql(result, default_select);
  // clang-format off
  assertResultSetEqual({{i(NULL_SMALLINT),i(1)},
                        {i(NULL_SMALLINT),i(2)},
                        {i(NULL_SMALLINT),i(NULL_SMALLINT)},
                        {i(NULL_SMALLINT),i(NULL_SMALLINT)}},
                       result);
  // clang-format on
}

TEST_P(CacheControllingSelectQueryTest, CacheExists) {
  auto cache = getCatalog().getDataMgr().getPersistentStorageMgr()->getDiskCache();
  if (GetParam() == File_Namespace::DiskCacheLevel::none) {
    ASSERT_EQ(cache, nullptr);
  } else {
    ASSERT_NE(cache, nullptr);
  }
}

TEST_P(CacheControllingSelectQueryTest, RefreshDisabledCache) {
  std::string temp_file{getDataFilesPath() + "/.tmp.csv"};
#if 107400 <= BOOST_VERSION
  constexpr auto copy_options = bf::copy_options::overwrite_existing;
#else
  constexpr auto copy_options = bf::copy_option::overwrite_if_exists;
#endif
  bf::copy_file(getDataFilesPath() + "0.csv", temp_file, copy_options);
  std::string query = "CREATE FOREIGN TABLE " + default_table_name + " (i INTEGER) "s +
                      "SERVER default_local_delimited WITH (file_path = '" + temp_file +
                      "');";
  sql(query);
  TQueryResult pre_refresh_result;
  sql(pre_refresh_result, default_select);
  assertResultSetEqual({{i(0)}}, pre_refresh_result);
  bf::copy_file(getDataFilesPath() + "two_row_3_4.csv", temp_file, copy_options);

  sql("REFRESH FOREIGN TABLES " + default_table_name + ";");
  TQueryResult post_refresh_result;
  sql(post_refresh_result, default_select);
  assertResultSetEqual({{i(3)}, {i(4)}}, post_refresh_result);
  bf::remove_all(temp_file);
}

class SqliteCacheControllingSelectQueryTest : public CacheControllingSelectQueryTest {
  void SetUp() override {
    wrapper_type_ = "sqlite";
    SKIP_SETUP_IF_ODBC_DISABLED();
    CacheControllingSelectQueryTest::SetUp();
  }
};

INSTANTIATE_TEST_SUITE_P(SqliteCacheOnOffSelectQueryTests,
                         SqliteCacheControllingSelectQueryTest,
                         ::testing::Values(File_Namespace::DiskCacheLevel::none,
                                           File_Namespace::DiskCacheLevel::fsi),
                         [](const auto& info) {
                           switch (info.param) {
                             case File_Namespace::DiskCacheLevel::none:
                               return "NoCache";
                             case File_Namespace::DiskCacheLevel::fsi:
                               return "FsiCache";
                             default:
                               return "UNKNOWN";
                           }
                         });

TEST_P(SqliteCacheControllingSelectQueryTest, NoDanglingChunkBuffersOnTableRecreate) {
  auto create_table = createForeignTableQuery({{"txt", "TEXT"}, {"quoted_txt", "TEXT"}},
                                              getDataFilesPath() + "non_quoted.csv",
                                              wrapper_type_);
  auto drop_table = "DROP FOREIGN TABLE IF EXISTS " + default_table_name + ";";
  auto select_table = "SELECT * FROM " + default_table_name + ";";

  for (int i = 0; i < 2; i++) {
    sql(drop_table);
    sql(create_table);
    TQueryResult result;
    sql(result, "SELECT * FROM " + default_table_name + " ORDER BY txt;");

    // clang-format off
    assertResultSetEqual({{"text_1","text_1"},
                          {"text_2","text_2"},
                          {"text_3","text_3"}},
                        result);
    // clang-format on
  }
  sql(drop_table);
}

using CacheAndDataWrapperParamType =
    std::tuple<File_Namespace::DiskCacheLevel, WrapperType>;
class CacheAndDataWrapperControllingSelectQueryTest
    : public CacheControllingSelectQueryBaseTest,
      public ::testing::WithParamInterface<CacheAndDataWrapperParamType> {
 public:
  CacheAndDataWrapperControllingSelectQueryTest()
      : CacheControllingSelectQueryBaseTest(std::get<0>(GetParam())) {}

  void SetUp() override {
    wrapper_type_ = std::get<1>(GetParam());
    CacheControllingSelectQueryBaseTest::SetUp();
  }

  static std::string getTestName(
      const ::testing::TestParamInfo<CacheAndDataWrapperParamType>& info) {
    const auto& [cache_level, wrapper_type] = info.param;
    std::stringstream ss;
    ss << "CacheLevel_" << cache_level << "_DataWrapper_" << wrapper_type;
    return ss.str();
  }
};

TEST_P(CacheAndDataWrapperControllingSelectQueryTest, RefreshGeoTypes) {
  createForeignTableForGeoTypes(wrapper_type_, wrapper_ext(wrapper_type_));
  queryAndAssertGeoTypesResult();

  sql("REFRESH FOREIGN TABLES " + default_table_name + ";");
  queryAndAssertGeoTypesResult();
}

INSTANTIATE_TEST_SUITE_P(
    DifferentCacheLevelsAndDataWrappers,
    CacheAndDataWrapperControllingSelectQueryTest,
    ::testing::Combine(::testing::Values(File_Namespace::DiskCacheLevel::none,
                                         File_Namespace::DiskCacheLevel::fsi),
                       ::testing::ValuesIn(local_wrappers)),
    CacheAndDataWrapperControllingSelectQueryTest::getTestName);

TEST_F(SelectQueryTest, ParquetStringsAllNullPlacementPermutations) {
  const auto& query = getCreateForeignTableQuery(
      "( id INT, txt1 TEXT ENCODING NONE, txt2 TEXT ENCODING DICT (32), txt3 TEXT "
      "ENCODING DICT (16), txt4 TEXT ENCODING DICT (8))",
      "strings_with_all_null_placement_permutations",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * from " + default_table_name + " ORDER BY id;");
  // clang-format off
  assertResultSetEqual(
      {
        { i(1), "txt1", "txt1", "txt1", "txt1"},
        { i(2), "txt2", "txt2", "txt2", "txt2"},
        { i(3), "txt3", "txt3", "txt3", "txt3"},
        { i(4), Null, Null, Null, Null },
        { i(5), "txt5", "txt5", "txt5", "txt5"},
        { i(6), "txt6", "txt6", "txt6", "txt6"},
        { i(7), "txt7", "txt7", "txt7", "txt7"},
        { i(8), Null, Null, Null, Null },
        { i(9), "txt9", "txt9", "txt9", "txt9"},
        { i(10), "txt10", "txt10", "txt10", "txt10"},
        { i(11), "txt11", "txt11", "txt11", "txt11"},
        { i(12), Null, Null, Null, Null },
        { i(13), Null, Null, Null, Null },
        { i(14), Null, Null, Null, Null },
        { i(15), "txt15", "txt15", "txt15", "txt15"},
        { i(16), Null, Null, Null, Null },
        { i(17), "txt17", "txt17", "txt17", "txt17"},
        { i(18), Null, Null, Null, Null },
        { i(19), "txt19", "txt19", "txt19", "txt19"},
        { i(20), Null, Null, Null, Null },
        { i(21), Null, Null, Null, Null },
        { i(22), Null, Null, Null, Null },
        { i(23), Null, Null, Null, Null },
        { i(24), Null, Null, Null, Null }
      },
      result);
  // clang-format on
}

TEST_F(SelectQueryTest, ParquetStringDictionaryEncodedMetadataTest) {
  const auto& query = getCreateForeignTableQuery("(txt TEXT ENCODING DICT (32) )",
                                                 {{"fragment_size", "4"}},
                                                 "strings_repeating",
                                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT count(txt) from " + default_table_name + " WHERE txt = 'a';");
  assertResultSetEqual({{i(5)}}, result);
}

TEST_F(SelectQueryTest, ParquetNumericAndBooleanTypesWithAllNullPlacementPermutations) {
  const auto& query = getCreateForeignTableQuery(
      "( id INT, bool BOOLEAN, i8 TINYINT, u8 SMALLINT, i16 SMALLINT, "
      "u16 INT, i32 INT, u32 BIGINT, i64 BIGINT, f32 FLOAT, "
      "f64 DOUBLE, fixedpoint DECIMAL(10,5) )",
      "numeric_and_boolean_types_with_all_null_placement_permutations",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * from " + default_table_name + " order by id;");

  // clang-format off
  assertResultSetEqual({
   {i(1),i(True),i(100),i(100),i(23000),i(23000),i(2047483647),i(2047483647),i(9123372036854775807),(1e-4f),(1e-4),(1.123)},
   {i(2),i(False),i(-127),i(0),i(-32767),i(0),i(-2147483647),i(0),i(-9223372036854775807),(3.141592f),(3.141592653589793),(100.1)},
   {i(3),i(True),i(127),i(255),i(32767),i(65535),i(2147483647),i(4294967295),i(9223372036854775807),(1e9f),(1e19),(2.22)},
   {i(4),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
   {i(5),i(False),i(-127),i(0),i(-32767),i(0),i(-2147483647),i(0),i(-9223372036854775807),(3.141592f),(3.141592653589793),(100.1)},
   {i(6),i(True),i(127),i(255),i(32767),i(65535),i(2147483647),i(4294967295),i(9223372036854775807),(1e9f),(1e19),(2.22)},
   {i(7),i(True),i(100),i(100),i(23000),i(23000),i(2047483647),i(2047483647),i(9123372036854775807),(1e-4f),(1e-4),(1.123)},
   {i(8),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
   {i(9),i(True),i(127),i(255),i(32767),i(65535),i(2147483647),i(4294967295),i(9223372036854775807),(1e9f),(1e19),(2.22)},
   {i(10),i(True),i(100),i(100),i(23000),i(23000),i(2047483647),i(2047483647),i(9123372036854775807),(1e-4f),(1e-4),(1.123)},
   {i(11),i(False),i(-127),i(0),i(-32767),i(0),i(-2147483647),i(0),i(-9223372036854775807),(3.141592f),(3.141592653589793),(100.1)},
   {i(12),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
   {i(13),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
   {i(14),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
   {i(15),i(True),i(127),i(255),i(32767),i(65535),i(2147483647),i(4294967295),i(9223372036854775807),(1e9f),(1e19),(2.22)},
   {i(16),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
   {i(17),i(False),i(-127),i(0),i(-32767),i(0),i(-2147483647),i(0),i(-9223372036854775807),(3.141592f),(3.141592653589793),(100.1)},
   {i(18),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
   {i(19),i(True),i(100),i(100),i(23000),i(23000),i(2047483647),i(2047483647),i(9123372036854775807),(1e-4f),(1e-4),(1.123)},
   {i(20),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
   {i(21),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
   {i(22),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
   {i(23),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
   {i(24),i(NULL_TINYINT),i(NULL_TINYINT),i(NULL_SMALLINT),i(NULL_SMALLINT),i(NULL_INT),i(NULL_INT),i(NULL_BIGINT),i(NULL_BIGINT),(NULL_FLOAT),(NULL_DOUBLE),(NULL_DOUBLE)},
  },
  result);
  // clang-format on
}

TEST_F(SelectQueryTest, ParquetNumericAndBooleanTypes) {
  const auto& query = getCreateForeignTableQuery(
      "( bool BOOLEAN, i8 TINYINT, u8 SMALLINT, i16 SMALLINT, "
      "u16 INT, i32 INT, u32 BIGINT, i64 BIGINT, f32 FLOAT, "
      "f64 DOUBLE, fixedpoint DECIMAL(10,5) )",
      "numeric_and_boolean_types",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, default_select);

  // clang-format off
  assertResultSetEqual({
   {i(True),i(100),i(100),i(23000),i(23000),i(2047483647),i(2047483647),i(9123372036854775807),(1e-4f),(1e-4),(1.123)},
   {i(False),i(-127),i(0),i(-32767),i(0),i(-2147483647),i(0),i(-9223372036854775807),(3.141592f),(3.141592653589793),(100.1)},
   {i(True),i(127),i(255),i(32767),i(65535),i(2147483647),i(4294967295),i(9223372036854775807),(1e9f),(1e19),(2.22)},
  },
  result);
  // clang-format on
}

TEST_F(SelectQueryTest, ParquetFixedEncodedTypes) {
  const auto& query = getCreateForeignTableQuery(
      "( i8 BIGINT ENCODING FIXED(8), u8 BIGINT ENCODING FIXED(16),"
      " i16 BIGINT ENCODING FIXED(16), "
      "u16 BIGINT ENCODING FIXED (32), i32 BIGINT ENCODING FIXED (32),"
      "i8_2 INT ENCODING FIXED(8), u8_2 INT ENCODING FIXED(16),"
      " i16_2 INT ENCODING FIXED(16),"
      "i8_3 SMALLINT ENCODING FIXED(8) )",
      "fixed_encoded_types",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, default_select);

  // clang-format off
  assertResultSetEqual({
      {i(100),i(100),i(23000),i(23000),i(2047483647),i(100),i(100),i(23000),i(100)},
      {i(-127),i(0),i(-32767),i(0),i(-2147483647),i(-127),i(0),i(-32767),i(-127)},
      {i(127),i(255),i(32767),i(65535),i(2147483647),i(127),i(255),i(32767),i(127)}
  },
  result);
  // clang-format on
}

TEST_F(SelectQueryTest, ParquetDecimalTypeMappings) {
  const auto& query = getCreateForeignTableQuery(
      "( decimal_i32 DECIMAL(8,3), decimal_i64 DECIMAL(10,3), decimal_fbla DECIMAL(7,3), "
      "decimal_ba DECIMAL(9,3)  ) ",
      "decimal",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, default_select);

  // clang-format off
  assertResultSetEqual({
   {1.123,1.123,1.123,1.123},
   {100.100,100.100,100.100,100.100},
   {2.220,2.220,2.220,2.220},
  },
  result);
  // clang-format on
}

TEST_F(SelectQueryTest, ParquetTimestampNoEncodingInSeconds) {
  const auto& query = getCreateForeignTableQuery(
      "(ts_milli TIMESTAMP, ts_micro TIMESTAMP, ts_nano TIMESTAMP)",
      "timestamp",
      "parquet");
  sql(query);
  TQueryResult result;
  sql(result, default_select);

  assertResultSetEqual(
      {{NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
       {NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
       {NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
       {"1/1/1900 00:00:10", "1/1/1900 00:00:10", "1/1/1900 00:00:10"},
       {"1/1/2200 00:00:10", "1/1/2200 00:00:10", "1/1/2200 00:00:10"},
       {"8/25/2020 00:00:10", "8/25/2020 00:00:10", "8/25/2020 00:00:10"}},
      result);
}

TEST_F(SelectQueryTest, ParquetTimestampNoEncodingAllPrecisions) {
  const auto& query = getCreateForeignTableQuery(
      "(ts_milli TIMESTAMP (3), ts_micro TIMESTAMP (6), ts_nano TIMESTAMP (9))",
      "timestamp",
      "parquet");
  sql(query);
  TQueryResult result;
  sql(result, default_select);
  assertResultSetEqual({{NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
                        {NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
                        {NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
                        {"1/1/1900 00:00:10.123",
                         "1/1/1900 00:00:10.123456",
                         "1/1/1900 00:00:10.123456789"},
                        {"1/1/2200 00:00:10.123",
                         "1/1/2200 00:00:10.123456",
                         "1/1/2200 00:00:10.123456789"},
                        {"8/25/2020 00:00:10.123",
                         "8/25/2020 00:00:10.123456",
                         "8/25/2020 00:00:10.123456789"}},
                       result);
}

TEST_F(SelectQueryTest, ParquetTimeNoEncodingInSeconds) {
  const auto& query = getCreateForeignTableQuery(
      "(time_milli TIME, time_micro TIME, time_nano TIME)", "time", "parquet");
  sql(query);
  TQueryResult result;
  sql(result, default_select);
  assertResultSetEqual({{NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
                        {NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
                        {NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
                        {"00:00:01", "00:00:01", "00:00:01"},
                        {"00:00:00", "00:00:00", "00:00:00"},
                        {"23:59:59", "23:59:59", "23:59:59"}},
                       result);
}

TEST_F(SelectQueryTest, ParquetTimeFixedLength32EncodingInSeconds) {
  const auto& query = getCreateForeignTableQuery(
      "(time_milli TIME ENCODING FIXED(32), time_micro TIME, time_nano TIME)",
      "time",
      "parquet");
  sql(query);
  TQueryResult result;
  sql(result, default_select);
  assertResultSetEqual({{NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
                        {NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
                        {NULL_BIGINT, NULL_BIGINT, NULL_BIGINT},
                        {"00:00:01", "00:00:01", "00:00:01"},
                        {"00:00:00", "00:00:00", "00:00:00"},
                        {"23:59:59", "23:59:59", "23:59:59"}},
                       result);
}

TEST_F(SelectQueryTest, ParquetDateNoEncoding) {
  const auto& query = getCreateForeignTableQuery("(days DATE)", "date", "parquet");
  sql(query);
  TQueryResult result;
  sql(result, default_select);
  assertResultSetEqual({{NULL_BIGINT},
                        {NULL_BIGINT},
                        {NULL_BIGINT},
                        {"1/1/1900"},
                        {"1/1/2200"},
                        {"8/25/2020"}},
                       result);
}

TEST_F(SelectQueryTest, ParquetDateDays32Encoding) {
  const auto& query =
      getCreateForeignTableQuery("(days DATE ENCODING DAYS (32) )", "date", "parquet");
  sql(query);
  TQueryResult result;
  sql(result, default_select);
  assertResultSetEqual({{NULL_BIGINT},
                        {NULL_BIGINT},
                        {NULL_BIGINT},
                        {"1/1/1900"},
                        {"1/1/2200"},
                        {"8/25/2020"}},
                       result);
}

TEST_F(SelectQueryTest, DirectoryWithDifferentSchema_SameNumberOfColumns) {
  std::string query = "CREATE FOREIGN TABLE " + default_table_name + " (t TIMESTAMP) "s +
                      "SERVER default_local_parquet WITH (file_path = '" +
                      getDataFilesPath() + "/different_parquet_schemas_1');";
  sql(query);
  queryAndAssertException(default_select,
                          "Parquet file \"" + getDataFilesPath() +
                              "different_parquet_schemas_1/timestamp_millis.parquet\" "
                              "has a different schema. Please ensure that all Parquet "
                              "files use the same schema. Reference Parquet file: " +
                              getDataFilesPath() +
                              "different_parquet_schemas_1/timestamp_micros.parquet, "
                              "column name: timestamp_micros. New Parquet file: " +
                              getDataFilesPath() +
                              "different_parquet_schemas_1/timestamp_millis.parquet, "
                              "column name: timestamp_millis.");
}

TEST_F(SelectQueryTest, DirectoryWithDifferentSchema_DifferentNumberOfColumns) {
  std::string query = "CREATE FOREIGN TABLE " + default_table_name + " (i BIGINT) "s +
                      "SERVER default_local_parquet WITH (file_path = '" +
                      getDataFilesPath() + "/different_parquet_schemas_2');";
  sql(query);
  queryAndAssertException(
      default_select,
      "Parquet file \"" + getDataFilesPath() +
          "different_parquet_schemas_2/two_col_1_2.parquet\" has a different schema. "
          "Please ensure that all Parquet files use the same schema. Reference Parquet "
          "file: \"" +
          getDataFilesPath() +
          "different_parquet_schemas_2/1.parquet\" has 1 columns. New Parquet file \"" +
          getDataFilesPath() +
          "different_parquet_schemas_2/two_col_1_2.parquet\" has 2 columns.");
}

TEST_F(SelectQueryTest, SchemaMismatch_CSV_Multithreaded) {
  const auto& query = getCreateForeignTableQuery(
      "(i INTEGER,  txt TEXT, txt_2 TEXT ENCODING NONE, txt_arr TEXT[], txt_3 TEXT)",
      {{"buffer_size", "25"}},
      "0_255",
      "csv");
  sql(query);
  queryAndAssertException("SELECT * FROM " + default_table_name + " ORDER BY i;",
                          "Mismatched number of logical columns: (expected 5 "
                          "columns, has 4): in file '" +
                              getDataFilesPath() + "0_255.csv'");
}
TEST_F(SelectQueryTest, ParseError) {
  const auto& query = getCreateForeignTableQuery(
      "(i INTEGER)", {{"buffer_size", "25"}}, "1badint", "csv");
  sql(query);
  queryAndAssertException(default_select,
                          "Parsing failure \""
                          "Unable to parse -a to INTEGER"
                          "\" in row \"-a\" in file \"" +
                              getDataFilesPath() + "1badint.csv\"");
}

TEST_F(SelectQueryTest, ExistingTableWithFsiDisabled) {
  std::string query = "CREATE FOREIGN TABLE " + default_table_name + " (i INTEGER) "s +
                      "SERVER default_local_delimited WITH (file_path = '" +
                      getDataFilesPath() + "/1.csv');";
  sql(query);
  g_enable_fsi = false;
  queryAndAssertException(default_select,
                          "Query cannot be executed for foreign table because "
                          "FSI is currently disabled.");
}

INSTANTIATE_TEST_SUITE_P(CacheOnOffSelectQueryTests,
                         CacheControllingSelectQueryTest,
                         ::testing::Values(File_Namespace::DiskCacheLevel::none,
                                           File_Namespace::DiskCacheLevel::fsi),
                         [](const auto& info) {
                           switch (info.param) {
                             case File_Namespace::DiskCacheLevel::none:
                               return "NoCache";
                             case File_Namespace::DiskCacheLevel::fsi:
                               return "FsiCache";
                             default:
                               return "UNKNOWN";
                           }
                         });

class DataWrapperSelectQueryTest : public SelectQueryTest,
                                   public ::testing::WithParamInterface<WrapperType> {
 public:
  void SetUp() override {
    wrapper_type_ = GetParam();
    SelectQueryTest::SetUp();
  }
};

INSTANTIATE_TEST_SUITE_P(DataWrapperParameterizedTests,
                         DataWrapperSelectQueryTest,
                         ::testing::ValuesIn(local_wrappers),
                         [](const auto& info) { return info.param; });

TEST_P(DataWrapperSelectQueryTest, SelectCount) {
  sql(createForeignTableQuery({{"t", "TEXT"}, {"i", "INT"}, {"d", "DOUBLE"}},
                              getDataFilesPath() + "example_2" + wrapper_ext(GetParam()),
                              GetParam()));
  queryAndAssertExample2Count();
}

TEST_P(DataWrapperSelectQueryTest, Int8EmptyAndNullArrayPermutations) {
  auto wrapper_type = GetParam();
  if (isOdbc(wrapper_type)) {
    GTEST_SKIP()
        << "Sqlite does not support array types; Postgres arrays currently unsupported";
  }
  // clang-format off
  sql(createForeignTableQuery(
      {{"index", "INT"},
       {"tinyint_arr_0", "TINYINT[]"}, {"tinyint_arr_1", "TINYINT[]"},
       {"tinyint_arr_2", "TINYINT[]"}, {"tinyint_arr_3", "TINYINT[]"},
       {"tinyint_arr_4", "TINYINT[]"}, {"tinyint_arr_5", "TINYINT[]"},
       {"tinyint_arr_6", "TINYINT[]"}, {"tinyint_arr_7", "TINYINT[]"},
       {"tinyint_arr_8", "TINYINT[]"}, {"tinyint_arr_9", "TINYINT[]"},
       {"tinyint_arr_10", "TINYINT[]"}, {"tinyint_arr_11", "TINYINT[]"},
       {"tinyint_arr_12", "TINYINT[]"}, {"tinyint_arr_13", "TINYINT[]"},
       {"tinyint_arr_14", "TINYINT[]"}, {"tinyint_arr_15", "TINYINT[]"},
       {"tinyint_arr_16", "TINYINT[]"}, {"tinyint_arr_17", "TINYINT[]"},
       {"tinyint_arr_18", "TINYINT[]"}, {"tinyint_arr_19", "TINYINT[]"},
       {"tinyint_arr_20", "TINYINT[]"}, {"tinyint_arr_21", "TINYINT[]"},
       {"tinyint_arr_22", "TINYINT[]"}, {"tinyint_arr_23", "TINYINT[]"}},
      getDataFilesPath() +
      "int8_empty_and_null_array_permutations" +
      wrapper_ext(wrapper_type),
      wrapper_type));
  // clang-format on

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");

  // clang-format off
  assertResultSetEqual(
    {
      {
        i(1),Null,Null,Null,Null,Null,Null,array({i(1)}),array({i(1)}),
        array({i(1)}), array({i(1)}),array({i(1)}),array({i(1)}),array({i(NULL_TINYINT)})
        ,array({i(NULL_TINYINT)}),array({i(NULL_TINYINT)}),array({i(NULL_TINYINT)}),
        array({i(NULL_TINYINT)}),array({i(NULL_TINYINT)}),array({}),array({}),array({}),
        array({}),array({}),array({})
      },
      {
        i(2),array({i(1)}),array({i(1)}),array({i(NULL_TINYINT)}),
        array({i(NULL_TINYINT)}),array({}),array({}),Null,Null,
        array({i(NULL_TINYINT)}),array({i(NULL_TINYINT)}),array({}),array({}),Null,
        Null,array({i(1)}),array({i(1)}),array({}),array({}),Null,Null,
        array({i(1)}),array({i(1)}),array({i(NULL_TINYINT)}),array({i(NULL_TINYINT)})
      },
      {
        i(3),array({i(NULL_TINYINT)}),array({}),array({i(1)}),array({}),array({i(1)}),
        array({i(NULL_TINYINT)}),array({i(NULL_TINYINT)}),array({}),Null,array({}),
        Null,array({i(NULL_TINYINT)}),array({i(1)}),array({}),Null,array({}),
        Null,array({i(1)}),array({i(1)}),array({i(NULL_TINYINT)}),Null,
        array({i(NULL_TINYINT)}),Null,array({i(1)})
      },
      {
        i(4),array({}),array({i(NULL_TINYINT)}),array({}),array({i(1)}),
        array({i(NULL_TINYINT)}),array({i(1)}),array({}),array({i(NULL_TINYINT)}),
        array({}),Null,array({i(NULL_TINYINT)}),Null,array({}),
        array({i(1)}),array({}),Null,array({i(1)}),Null,array({i(NULL_TINYINT)}),
        array({i(1)}),array({i(NULL_TINYINT)}),Null,array({i(1)}),Null
      },
    }, result);
  // clang-format on
}

TEST_P(DataWrapperSelectQueryTest, AggregateAndGroupBy) {
  sql(createForeignTableQuery({{"t", "TEXT"}, {"i", "BIGINT"}, {"f", "DOUBLE"}},
                              getDataFilesPath() + "example_2" + wrapper_ext(GetParam()),
                              GetParam()));

  TQueryResult result;
  sql(result,
      "SELECT t, avg(i), sum(f) FROM " + default_table_name + " group by t order by t;");
  // clang-format off
  assertResultSetEqual({{"a", 1.0, 1.1},
                        {"aa", 1.5, 3.3},
                        {"aaa", 2.0, 6.6}},
                       result);
  // clang-format on
}

TEST_P(DataWrapperSelectQueryTest, FilterGreater) {
  sql(createForeignTableQuery({{"t", "TEXT"}, {"i", "BIGINT"}, {"f", "DOUBLE"}},
                              getDataFilesPath() + "example_2" + wrapper_ext(GetParam()),
                              GetParam()));

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " WHERE i > 1;");
  // clang-format off
  assertResultSetEqual({{"aa", i(2), 2.2},
                        {"aaa", i(2), 2.2},
                        {"aaa", i(3), 3.3}},
                       result);
  // clang-format on
}

TEST_P(DataWrapperSelectQueryTest, FilterLesser) {
  sql(createForeignTableQuery({{"t", "TEXT"}, {"i", "BIGINT"}, {"f", "DOUBLE"}},
                              getDataFilesPath() + "example_2" + wrapper_ext(GetParam()),
                              GetParam()));

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " WHERE i <= 2;");
  // clang-format off
  assertResultSetEqual({{"a", i(1), 1.1},
                        {"aa", i(1), 1.1},
                        {"aa", i(2), 2.2},
                        {"aaa", i(1), 1.1},
                        {"aaa", i(2), 2.2}},
                       result);
  // clang-format on
}

TEST_P(DataWrapperSelectQueryTest, Update) {
  sql(createForeignTableQuery({{"t", "TEXT"}, {"i", "BIGINT"}, {"f", "DOUBLE"}},
                              getDataFilesPath() + "example_2" + wrapper_ext(GetParam()),
                              GetParam()));
  queryAndAssertException("UPDATE " + default_table_name + " SET t = 'abc';",
                          "DELETE, INSERT, TRUNCATE, OR UPDATE commands are not "
                          "supported for foreign tables.");
}

TEST_P(DataWrapperSelectQueryTest, Insert) {
  sql(createForeignTableQuery({{"t", "TEXT"}, {"i", "BIGINT"}, {"f", "DOUBLE"}},
                              getDataFilesPath() + "example_2" + wrapper_ext(GetParam()),
                              GetParam()));
  queryAndAssertException(
      "INSERT INTO " + default_table_name + " VALUES('abc', null, null);",
      "DELETE, INSERT, TRUNCATE, OR UPDATE commands are not "
      "supported for foreign tables.");
}

TEST_P(DataWrapperSelectQueryTest, InsertIntoSelect) {
  sql(createForeignTableQuery({{"t", "TEXT"}, {"i", "BIGINT"}, {"f", "DOUBLE"}},
                              getDataFilesPath() + "example_2" + wrapper_ext(GetParam()),
                              GetParam()));
  queryAndAssertException(
      "INSERT INTO " + default_table_name + " SELECT * FROM " + default_table_name + ";",
      "DELETE, INSERT, TRUNCATE, OR UPDATE commands are not supported for "
      "foreign "
      "tables.");
}

TEST_P(DataWrapperSelectQueryTest, Delete) {
  sql(createForeignTableQuery({{"t", "TEXT"}, {"i", "BIGINT"}, {"f", "DOUBLE"}},
                              getDataFilesPath() + "example_2" + wrapper_ext(GetParam()),
                              GetParam()));
  queryAndAssertException("DELETE FROM " + default_table_name + " WHERE t = 'a';",
                          "DELETE, INSERT, TRUNCATE, OR UPDATE commands are not "
                          "supported for foreign tables.");
}

TEST_P(DataWrapperSelectQueryTest, AggregateAndGroupByNull) {
  sql(createForeignTableQuery({{"t", "TEXT"}, {"i", "INT"}},
                              getDataFilesPath() + "null_str" + wrapper_ext(GetParam()),
                              GetParam()));
  TQueryResult result;
  sql(result,
      "select t, count( * )  from " + default_table_name + " group by 1 order by 1 asc;");
  // clang-format off
  assertResultSetEqual({{"a", i(1)},
                        {"b", i(1)},
                        {"c", i(1)},
                        {Null, i(1)}},
                       result);
  // clang-format on
}

TEST_P(DataWrapperSelectQueryTest, null_values_for_non_encoded_types) {
  // This tests looks at the situation where all values in a chunk of a
  // unencoded column are null.  To facilitate this the fragment size
  // is set to 1.
  auto wrapper_type = GetParam();

  std::vector<NameTypePair> column_pairs{
      {"idx", "SMALLINT"},
      {"txt", "TEXT ENCODING NONE"},
      {"txt_encoded", "TEXT"},
      // the p columns is removed for the odbc dw test
      {"p", "GEOMETRY(POINT,4326) ENCODING NONE"},
  };

  std::string source_file_name =
      getDataFilesPath() + "null_values_for_none_encoded_fields";
  std::vector<std::vector<NullableTargetValue>> expected_results;
  if (wrapper_type == "postgres" || wrapper_type == "sqlite") {
    // remove the geo column from the source data
    // and column definitions.
    column_pairs.pop_back();
    source_file_name += ".odbc";
    expected_results = {{i(1), Null, "text_encoded"},
                        {i(2), "text_none", Null},
                        {i(3), "text_none", "text_encoded"},
                        {i(4), Null, Null}};
  } else {
    expected_results = {{i(1), Null, "text_encoded", "POINT (0 0)"},
                        {i(2), "text_none", Null, "POINT (0 0)"},
                        {i(3), "text_none", "text_encoded", Null},
                        {i(4), Null, Null, Null}};
  }

  source_file_name += wrapper_ext(wrapper_type);

  sql(createForeignTableQuery(column_pairs,
                              source_file_name,
                              wrapper_type,
                              {{"FRAGMENT_SIZE", std::to_string(1)}}));

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " order by idx;");
  assertResultSetEqual(expected_results, result);
}

TEST_P(DataWrapperSelectQueryTest, ArrayWithNullValues) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP()
        << "Sqlite does notsupport array types; Postgres arrays currently unsupported";
  }
  foreign_storage::OptionsMap options;
  if (is_regex(wrapper_type_)) {
    options["LINE_REGEX"] = "(\\d+),\\s*" + get_line_array_regex(3);
  }
  sql(createForeignTableQuery({{"index", "INTEGER"},
                               {"i1", "INTEGER[]"},
                               {"i2", "INTEGER[]"},
                               {"i3", "INTEGER[]"}},
                              getDataFilesPath() + "null_array" + wrapper_ext(GetParam()),
                              GetParam(),
                              options));
  // clang-format off
  sqlAndCompareResult("select * from " + default_table_name +
                      " order by index;",
                      {{i(1), Null, Null, array({Null_i})},
                       {i(2), Null, array({i(100)}), array({Null_i, Null_i})},
                       {i(3), array({i(100)}), array({i(200)}), array({Null_i, i(100)})}});
  // clang-format on
}

TEST_P(DataWrapperSelectQueryTest, MissingFileOnCreateTable) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  auto file_path = getDataFilesPath() + "missing_file" + wrapper_ext(GetParam());
  auto query = createForeignTableQuery({{"i", "INTEGER"}}, file_path, GetParam());
  queryAndAssertFileNotFoundException(file_path, query);
}

TEST_P(DataWrapperSelectQueryTest, MissingFileOnSelectQuery) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  auto file_path = boost::filesystem::absolute("missing_file");
  boost::filesystem::copy_file(getDataFilesPath() + "0" + wrapper_ext(GetParam()),
                               file_path,
#if 107400 <= BOOST_VERSION
                               boost::filesystem::copy_options::overwrite_existing
#else
                               boost::filesystem::copy_option::overwrite_if_exists
#endif
  );
  sql(createForeignTableQuery({{"i", "INTEGER"}}, file_path.string(), GetParam()));
  boost::filesystem::remove_all(file_path);
  queryAndAssertFileNotFoundException(file_path.string());
}

TEST_P(DataWrapperSelectQueryTest, EmptyDirectory) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  auto dir_path = boost::filesystem::absolute("empty_dir");
  boost::filesystem::create_directory(dir_path);
  sql(createForeignTableQuery({{"i", "INTEGER"}}, dir_path.string(), GetParam()));
  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + ";");
  assertResultSetEqual({}, result);
  boost::filesystem::remove_all(dir_path);
}

TEST_P(DataWrapperSelectQueryTest, RecursiveDirectory) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  sql(createForeignTableQuery(
      {{"t", "TEXT"}, {"i", "INT"}, {"d", "DOUBLE"}},
      getDataFilesPath() + "example_2_" + wrapper_file_type(GetParam()) + "_dir/",
      wrapper_type_));

  queryAndAssertExample2Result();
}

TEST_P(DataWrapperSelectQueryTest, FilePathWithLeadingSlash) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid test case for ODBC wrappers";
  }
  sql("CREATE SERVER test_server FOREIGN DATA WRAPPER " +
      get_data_wrapper_name(wrapper_type_) +
      " WITH (storage_type = 'LOCAL_FILE', base_path = '" + getDataFilesPath() + "');");
  std::string options{"file_path = '/1" + wrapper_ext(wrapper_type_) + "'"};
  if (wrapper_type_ == "regex_parser") {
    options += ", line_regex = '(\\d+)', header = 'true'";
  }
  sql("CREATE FOREIGN TABLE " + default_table_name +
      " (i INTEGER) SERVER test_server WITH (" + options + ");");
  sqlAndCompareResult(default_select, {{i(1)}});
}

TEST_P(DataWrapperSelectQueryTest, NoMatchWildcard) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  queryAndAssertException(
      createForeignTableQuery(
          {{"i", "INT"}}, getDataFilesPath() + "no_match_*", wrapper_type_),
      "File or directory \"" + getDataFilesPath() + "no_match_*" + "\" does not exist.");
}

TEST_P(DataWrapperSelectQueryTest, WildcardOnFiles) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  sql(createForeignTableQuery(
      {{"t", "TEXT"}, {"i", "INT"}, {"d", "DOUBLE"}},
      getDataFilesPath() + "example_2_" + wrapper_file_type(GetParam()) + "_dir/f*",
      wrapper_type_));
  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY t, i;");
  // clang-format off
  assertResultSetEqual({{"a", i(1), 1.1},
                        {"aa", i(1), 1.1},
                        {"aa", i(2), 2.2},
                        {"aaa", i(1), 1.1}},
                        result);
  // clang-format on
}

TEST_P(DataWrapperSelectQueryTest, WildcardOnDirectory) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  sql(createForeignTableQuery(
      {{"t", "TEXT"}, {"i", "INT"}, {"d", "DOUBLE"}},
      getDataFilesPath() + "example_2_" + wrapper_file_type(GetParam()) + "_d*",
      wrapper_type_));
  queryAndAssertExample2Result();
}

TEST_P(DataWrapperSelectQueryTest, NoMatchRegexPathFilter) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  sql(createForeignTableQuery({{"i1", "INT"}},
                              getDataFilesPath(),
                              wrapper_type_,
                              {{"REGEX_PATH_FILTER", "very?obscure?pattern"}}));
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + ";",
      "No files matched the regex file path \"very?obscure?pattern\".");
}

TEST_P(DataWrapperSelectQueryTest, RegexPathFilterOnFiles) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  sql(createForeignTableQuery(
      {{"t", "TEXT"}, {"i", "INT"}, {"d", "DOUBLE"}},
      getDataFilesPath() + "example_2_" + wrapper_file_type(GetParam()) + "_dir/",
      GetParam(),
      {{"REGEX_PATH_FILTER", ".*_dir/file.*"}}));
  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY t, i;");
  // clang-format off
  assertResultSetEqual({{"a", i(1), 1.1},
                        {"aa", i(1), 1.1},
                        {"aa", i(2), 2.2},
                        {"aaa", i(1), 1.1}},
                       result);
  // clang-format on
}

TEST_P(DataWrapperSelectQueryTest, SortedOnPathname) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  sql(createForeignTableQuery(
      {{"i", "INT"}},
      getDataFilesPath() + "sorted_dir/",
      wrapper_file_type(GetParam()),
      {{"REGEX_PATH_FILTER", ".*" + wrapper_file_type(GetParam())},
       {"FILE_SORT_ORDER_BY", "pathNAME"}}));
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";",
                      {{i(2)}, {i(1)}, {i(0)}, {i(9)}});
}

TEST_P(DataWrapperSelectQueryTest, SortedOnDateModified) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  auto source_dir =
      bf::absolute(getDataFilesPath() + "sorted_dir/" + wrapper_file_type(GetParam()));
  auto temp_dir = bf::absolute("temp_sorted_on_date_modified");
  bf::remove_all(temp_dir);
  bf::copy(source_dir, temp_dir);

  // some platforms won't copy directory contents on a directory copy
  for (auto& file : boost::filesystem::recursive_directory_iterator(source_dir)) {
    auto source_file = file.path();
    auto dest_file = temp_dir / file.path().filename();
    if (!boost::filesystem::exists(dest_file)) {
      boost::filesystem::copy(file.path(), temp_dir / file.path().filename());
    }
  }
  auto reference_time = bf::last_write_time(temp_dir);
  bf::last_write_time(temp_dir / ("zzz." + wrapper_file_type(GetParam())),
                      reference_time - 2);
  bf::last_write_time(temp_dir / ("a_21_2021-01-01." + wrapper_file_type(GetParam())),
                      reference_time - 1);
  bf::last_write_time(temp_dir / ("c_00_2021-02-15." + wrapper_file_type(GetParam())),
                      reference_time);
  bf::last_write_time(temp_dir / ("b_15_2021-12-31." + wrapper_file_type(GetParam())),
                      reference_time + 1);

  sql(createForeignTableQuery({{"i", "INT"}},
                              temp_dir.string(),
                              wrapper_file_type(GetParam()),
                              {{"FILE_SORT_ORDER_BY", "DATE_MODIFIED"}}));
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";",
                      {{i(9)}, {i(2)}, {i(0)}, {i(1)}});

  bf::remove_all(temp_dir);
}

TEST_P(DataWrapperSelectQueryTest, SortedOnRegex) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  sql(createForeignTableQuery(
      {{"i", "INT"}},
      getDataFilesPath() + "sorted_dir/" + wrapper_file_type(GetParam()),
      wrapper_file_type(GetParam()),
      {{"FILE_SORT_ORDER_BY", "REGEX"}, {"FILE_SORT_REGEX", ".*[a-z]_[0-9]([0-9])_.*"}}));
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";",
                      {{i(9)}, {i(0)}, {i(2)}, {i(1)}});
}

TEST_P(DataWrapperSelectQueryTest, SortedOnRegexDate) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  sql(createForeignTableQuery(
      {{"i", "INT"}},
      getDataFilesPath() + "sorted_dir/" + wrapper_file_type(GetParam()),
      wrapper_file_type(GetParam()),
      {{"FILE_SORT_ORDER_BY", "REGEX_DATE"},
       {"FILE_SORT_REGEX", ".*[a-z]_[0-9][0-9]_(.*)\\..*"}}));
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";",
                      {{i(9)}, {i(2)}, {i(0)}, {i(1)}});
}

TEST_P(DataWrapperSelectQueryTest, SortedOnRegexNumberAndMultiCaptureGroup) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  sql(createForeignTableQuery(
      {{"i", "INT"}},
      getDataFilesPath() + "sorted_dir/" + wrapper_file_type(GetParam()),
      wrapper_file_type(GetParam()),
      {{"FILE_SORT_ORDER_BY", "REGEX_NUMBER"},
       {"FILE_SORT_REGEX", ".*[a-z]_(.*)_(.*)-(.*)-(.*)\\..*"}}));
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";",
                      {{i(9)}, {i(0)}, {i(1)}, {i(2)}});
}

TEST_P(DataWrapperSelectQueryTest, SortedOnNonRegexWithSortRegex) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  queryAndAssertException(
      createForeignTableQuery(
          {{"i", "INT"}},
          getDataFilesPath() + "sorted_dir/" + wrapper_file_type(GetParam()),
          wrapper_file_type(GetParam()),
          {{"FILE_SORT_ORDER_BY", "DATE_MODIFIED"}, {"FILE_SORT_REGEX", "xxx"}}),
      "Option \"FILE_SORT_REGEX\" must not be set for selected option "
      "\"FILE_SORT_ORDER_BY='DATE_MODIFIED'\".");
}

TEST_P(DataWrapperSelectQueryTest, SortedOnRegexWithoutSortRegex) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP() << "Not a valid testcase for ODBC wrappers";
  }
  queryAndAssertException(createForeignTableQuery({{"i", "INT"}},
                                                  getDataFilesPath() + "sorted_dir/" +
                                                      wrapper_file_type(GetParam()),
                                                  wrapper_file_type(GetParam()),
                                                  {{"FILE_SORT_ORDER_BY", "REGEX"}}),
                          "Option \"FILE_SORT_REGEX\" must be set for selected option "
                          "\"FILE_SORT_ORDER_BY='REGEX'\".");
}

TEST_P(DataWrapperSelectQueryTest, OutOfRange) {
  sql(createForeignTableQuery(
      {{"i", "INTEGER"}, {"i2", "INTEGER"}},
      getDataFilesPath() + "out_of_range_int" + wrapper_ext(GetParam()),
      GetParam()));

  if (GetParam() == "csv" || GetParam() == "regex_parser") {
    queryAndAssertException(
        "SELECT * FROM "s + default_table_name,
        "Parsing failure \"Integer -2147483648 exceeds minimum value for "
        "nullable INTEGER(32)\" in row \"0,-2147483648\" in file \"" +
            getDataFilesPath() + "out_of_range_int.csv\"");
  } else if (GetParam() == "parquet") {
    queryAndAssertException(
        "SELECT * FROM "s + default_table_name,
        "Parquet column contains values that are outside the range of the "
        "HeavyDB column type. Consider using a wider column type. Min allowed value: "
        "-2147483647. Max allowed value: 2147483647. Encountered value: -2147483648. "
        "Error validating statistics of Parquet column 'numeric' in row group 0 of "
        "Parquet file '" +
            getDataFilesPath() + "out_of_range_int.parquet'.");
  } else {
    // Assuming ODBC for now
    queryAndAssertException(
        "SELECT * FROM "s + default_table_name,
        "ODBC column contains values that are outside the range of the "
        "database "
        "column type INTEGER. Min allowed value: -2147483647. Max allowed value: "
        "2147483647. Encountered value: -2147483648. Foreign table: test_foreign_table");
  }
}

TEST_P(DataWrapperSelectQueryTest, NullTextArray) {
  if (isOdbc(GetParam())) {
    GTEST_SKIP()
        << "Sqlite does not support array types; Postgres arrays currently unsupported";
  }
  foreign_storage::OptionsMap options;
  if (is_regex(wrapper_type_)) {
    options["LINE_REGEX"] = "(\\d+),\\s*" + get_line_array_regex(3);
  }
  sql(createForeignTableQuery(
      {{"index", "INTEGER"}, {"txt1", "TEXT[]"}, {"txt2", "TEXT[]"}, {"txt3", "TEXT[2]"}},
      getDataFilesPath() + "null_text_arrays" + wrapper_ext(GetParam()),
      GetParam(),
      options));
  // clang-format off
  sqlAndCompareResult("SELECT * FROM " + default_table_name +
                      " ORDER BY index;",
                      {{i(1), array({Null, Null}), array({Null}), Null},
                       {i(2), array({Null}), array({Null, Null}), Null},
                       {i(3), Null, array({Null, Null}), array({Null, Null})},
                       {i(4), array({Null, Null}), Null, Null}});
  // clang-format on
}

TEST_P(DataWrapperSelectQueryTest, EmptyTable) {
  sql(createForeignTableQuery({{"t", "TEXT"}},
                              getDataFilesPath() + "empty" + wrapper_ext(GetParam()),
                              GetParam()));
  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";", {});
}

class CSVFileTypeTests
    : public SelectQueryTest,
      public ::testing::WithParamInterface<std::pair<FileNameType, FileExtType>> {};

INSTANTIATE_TEST_SUITE_P(
    CSVFileTypeParameterizedTests,
    CSVFileTypeTests,
    ::testing::Values(std::make_pair("example_1.csv", "uncompressed"),
                      std::make_pair("example_1.zip", "zip"),
                      std::make_pair("example_1_newline.zip", "zip_newline"),
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
                      std::make_pair("example_1_dir_newline", "dir_newline"),
                      std::make_pair("example_1_dir_archives", "dir_archives"),
                      std::make_pair("example_1_dir_multilevel", "multilevel_dir")),
    [](const auto& info) { return "File_Type_" + info.param.second; });

TEST_P(CSVFileTypeTests, SelectCSV) {
  std::string query = "CREATE FOREIGN TABLE " + default_table_name +
                      " (t TEXT, i INTEGER[]) "s +
                      "SERVER default_local_delimited WITH (file_path = '" +
                      getDataFilesPath() + "/" + GetParam().first + "');";
  sql(query);
  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + "  ORDER BY t;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, Sort) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY t DESC;");
  assertResultSetEqual({{"aaa", array({i(3), Null_i, i(3)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"a", array({i(1), i(1), i(1)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, Join) {
  auto query = getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1", "csv");
  sql(query);

  query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER, d DOUBLE)", "example_2", "csv", 2);
  sql(query);

  TQueryResult result;
  sql(result,
      "SELECT t1.t, t1.i, t2.i, t2.d FROM " + default_table_name +
          " AS t1 JOIN "
          "" +
          default_table_name + "_2 AS t2 ON t1.t = t2.t;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)}), i(1), 1.1},
                        {"aa", array({Null_i, i(2), i(2)}), i(1), 1.1},
                        {"aa", array({Null_i, i(2), i(2)}), i(2), 2.2},
                        {"aaa", array({i(3), Null_i, i(3)}), i(1), 1.1},
                        {"aaa", array({i(3), Null_i, i(3)}), i(2), 2.2},
                        {"aaa", array({i(3), Null_i, i(3)}), i(3), 3.3}},
                       result);
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
  sql(result, default_select);
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

TEST_P(CacheControllingSelectQueryTest, CsvEmptyArchive) {
  std::string query = "CREATE FOREIGN TABLE " + default_table_name +
                      " (t TEXT, i INTEGER[]) "s +
                      "SERVER default_local_delimited WITH (file_path = '" +
                      getDataFilesPath() + "/" + "example_1_empty.zip" + "');";
  sql(query);
  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + "  ORDER BY t;");
  assertResultSetEqual({}, result);
}

TEST_P(CacheControllingSelectQueryTest, CsvArchiveInvalidFile) {
  std::string query = "CREATE FOREIGN TABLE " + default_table_name +
                      " (t TEXT, i INTEGER[]) "s +
                      "SERVER default_local_delimited WITH (file_path = '" +
                      getDataFilesPath() + "/" + "example_1_invalid_file.zip" + "');";
  sql(query);
  queryAndAssertException("SELECT * FROM " + default_table_name + "  ORDER BY t;",
                          "Mismatched number of logical columns: (expected 2 "
                          "columns, has 1): in file '" +
                              getDataFilesPath() + "example_1_invalid_file.zip'");
}

TEST_P(CacheControllingSelectQueryTest, CSV_CustomMarkers) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, t2 TEXT, i INTEGER[])",
      {{"array_marker", "[]"}, {"escape", "\\"}, {"nulls", "NIL"}, {"quote", "|"}},
      "custom_markers",
      "csv");
  sql(query);

  TQueryResult result;
  sql(result, default_select);
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
  sql(result, default_select);
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
  sql(result, default_select);
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
  sql(result, default_select);
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
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY t;");
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
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY t;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_P(CacheControllingSelectQueryTest, WithMaxBufferResizeLessThanRowSize) {
  SKIP_IF_DISTRIBUTED("Leaf nodes not affected by global variable enabling seconds");

  import_export::delimited_parser::set_max_buffer_resize(15);
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER[])", {{"buffer_size", "10"}}, "example_1", "csv");
  sql(query);

  queryAndAssertException(
      "SELECT * FROM " + default_table_name + " ORDER BY t;",
      "Unable to find an end of line character after reading 14 characters. "
      "Please ensure that the correct \"line_delimiter\" option is specified or update "
      "the \"buffer_size\" option appropriately. Row number: 2. "
      "First few characters in row: aa,{'NA', 2, 2");
}

TEST_P(CacheControllingSelectQueryTest, ReverseLongitudeAndLatitude) {
  const auto& query = getCreateForeignTableQuery(
      "(p POINT)", {{"lonlat", "false"}}, "reversed_long_lat", "csv");
  sql(query);

  TQueryResult result;
  sql(result, default_select);
  // clang-format off
  assertResultSetEqual({{"POINT (1 0)"},
                        {"POINT (2 1)"},
                        {"POINT (3 2)"}},
                       result);
  // clang-format on
}

TEST_F(SelectQueryTest, UnsupportedColumnMapping) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i BIGINT, f INTEGER)", {}, "example_2", "parquet");
  sql(query);
  queryAndAssertException(default_select,
                          "Conversion from Parquet type "
                          "\"DOUBLE\" to HeavyDB type \"INTEGER\" is "
                          "not allowed. Please use an appropriate column type. Parquet "
                          "column: double, HeavyDB column: f, Parquet file: " +
                              getDataFilesPath() + +"example_2.parquet.");
}

TEST_F(SelectQueryTest, NoStatistics) {
  const auto& query = getCreateForeignTableQuery(
      "(a BIGINT, b BIGINT, c TEXT, d DOUBLE)", {}, "no_stats", "parquet");
  sql(query);
  queryAndAssertException(
      default_select,
      "Statistics metadata is required for all row groups. Metadata is "
      "missing for row group index: 0, column index: 0, file path: " +
          getDataFilesPath() + "no_stats.parquet");
}

TEST_F(SelectQueryTest, EmptyNoStatistics) {
  const auto& query = getCreateForeignTableQuery(
      "(a BIGINT, b BIGINT, c TEXT, d DOUBLE)", {}, "empty_no_stats", "parquet");
  sql(query);
  sqlAndCompareResult(default_select, {});
}

TEST_F(SelectQueryTest, EmptyRowGroup) {
  const auto& query = getCreateForeignTableQuery(
      "(a BIGINT, b BIGINT, c TEXT, d DOUBLE)", {}, "empty_rowgroup", "parquet");
  sql(query);
  sqlAndCompareResult(default_select, {{1L, 3L, "5", 7.1}});
}

TEST_F(SelectQueryTest, RowGroupSizeLargerThanFragmentSize) {
  const auto& query = getCreateForeignTableQuery("(a BIGINT, b BIGINT, c TEXT, d DOUBLE)",
                                                 {{"fragment_size", "1"}},
                                                 "row_group_size_2",
                                                 "parquet");
  sql(query);
  queryAndAssertException(
      default_select,
      "Parquet file has a row group size that is larger than the fragment "
      "size. Please set the table fragment size to a number that is larger than the row "
      "group size. Row group index: 0, row group size: 2, fragment size: 1, file path: " +
          getDataFilesPath() + "row_group_size_2.parquet");
}

TEST_F(SelectQueryTest, DecimalIntEncoding) {
  const auto& query = getCreateForeignTableQuery(
      "(decimal_int_32 DECIMAL(9, 5), decimal_int_64 DECIMAL(15, 10))",
      {},
      "decimal_int_encoding",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, default_select);
  assertResultSetEqual({{100.1234, 100.1234}, {2.1234, 2.1234}, {100.1, 100.1}}, result);
}

TEST_F(SelectQueryTest, ByteArrayDecimalFilterAndSort) {
  const auto& query = getCreateForeignTableQuery(
      "(dc DECIMAL(4, 2))", {{"fragment_size", "3"}}, "byte_array_decimal", "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " where dc > 25 ORDER BY dc;");
  assertResultSetEqual({{25.55}, {50.11}}, result);
}

/**
 * Test for both FSI and Import tables
 */
class FsiImportSelectTest : public SelectQueryTest {
 protected:
  void SetUp() override {
    wrapper_type_ = "csv";
    SelectQueryTest::SetUp();
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    sql("DROP TABLE IF EXISTS " + default_table_name + ";");
    SelectQueryTest::TearDown();
  }

  /* Create option string from key/value pairs */
  static std::string createOptionsString(
      const foreign_storage::OptionsMap& options = {}) {
    std::stringstream opt_ss;
    for (auto& [key, value] : options) {
      if (opt_ss.str().size() > 0) {
        opt_ss << ", ";
      }
      if (atoi(value.c_str())) {
        opt_ss << key << " = " << value << "";
      } else {
        opt_ss << key << " = '" << value << "'";
      }
    }
    return opt_ss.str();
  }

  /**
   * Returns a query to create an import table.
   */
  static std::string createImportTableQuery(
      const std::string& columns,
      const foreign_storage::OptionsMap& options = {},
      const std::string& table_name = default_table_name) {
    std::stringstream ss;
    ss << "CREATE TABLE " << table_name << " ";
    ss << columns;
    if (options.size() > 0) {
      ss << " WITH (" << createOptionsString(options) << ")";
    }
    ss << ";";
    return ss.str();
  }

  /**
   * Returns a query to copy into an import table.
   */
  static std::string sqlCopyFromQuery(
      const std::string& src_path,
      const foreign_storage::OptionsMap options = {},
      const std::string& table_name = default_table_name) {
    std::stringstream ss;
    ss << "COPY " << table_name << " ";
    ss << " FROM '" << src_path << "'";
    if (options.size() > 0) {
      ss << " WITH (" << createOptionsString(options) << ")";
    }
    ss << ";";
    return ss.str();
  }

  /**
   * Create FSI/Import table and import data from files
   */
  static void sqlCreateTable(const ImportFlag import,
                             const std::string& columns,
                             const std::string& file_name,
                             const std::string& data_wrapper_type,
                             const foreign_storage::OptionsMap& table_options = {},
                             const foreign_storage::OptionsMap& copy_options = {}) {
    if (import) {
      sql(createImportTableQuery(columns, table_options));
      std::string filename = getDataFilesPath() + file_name + "." + data_wrapper_type;
      sql(sqlCopyFromQuery(getDataFilesPath() + file_name + "." + data_wrapper_type,
                           copy_options));
    } else {
      sqlCreateForeignTable(columns, file_name, data_wrapper_type, table_options);
    }
  }

  // Return last row of data file
  std::string getLastRow(const std::string& filename) {
    std::ifstream source_file(getDataFilesPath() + filename + "." + wrapper_type_);
    std::string value;
    // Return last line
    while (source_file) {
      std::getline(source_file, value);
    }
    return value;
  }
};

class FsiImportDecimalTest : public FsiImportSelectTest,
                             public ::testing::WithParamInterface<ImportFlag> {
  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    std::string table_type = GetParam() ? "TABLE" : "FOREIGN TABLE";
    sql("DROP " + table_type + " IF EXISTS " + default_table_name + ";");
    SelectQueryTest::TearDown();
  }
};

TEST_P(FsiImportDecimalTest, LongStrTruncate) {
  sqlCreateTable(GetParam(), "(d DECIMAL(18,2))", "decimal_longstr", wrapper_type_, {});
  TQueryResult result;
  sql(result, default_select);
  assertResultSetEqual(
      {
          {float(0)},
          {-0.01},
          {-10.01},
          {10.01},
          {9999999999999999.99},   // Max supported value
          {-9999999999999999.99},  // Min supported value
          {9999999999999999.99},   // Max supported value with trailing digit rounded down
          {-9999999999999999.99}   // Min supported value with trailing digit rounded down
      },
      result);
}

TEST_P(FsiImportDecimalTest, OutOfRangeBeforeDot) {
  auto import_flag = GetParam();
  std::string file_name = "decimal_out_of_range";
  sqlCreateTable(import_flag, "(d DECIMAL(18,2))", file_name, wrapper_type_, {});
  if (import_flag) {
    TQueryResult result;
    sql(result, default_select);
    // Failure will result in empty table
    assertResultSetEqual({}, result);
  } else {
    queryAndAssertException(default_select,
                            "Parsing failure \""
                            "Got out of range error during conversion from string to "
                            "DECIMAL(18,2)\" in row \"" +
                                getLastRow(file_name) + "\" in file \"" +
                                getDataFilesPath() + file_name + ".csv\"");
  }
}

TEST_P(FsiImportDecimalTest, OutOfRangeMax) {
  auto import_flag = GetParam();
  std::string file_name = "decimal_out_of_range_max";
  sqlCreateTable(import_flag, "(d DECIMAL(18,2))", file_name, wrapper_type_, {});
  if (import_flag) {
    TQueryResult result;
    sql(result, default_select);
    // Failure will result in empty table
    assertResultSetEqual({}, result);
  } else {
    queryAndAssertException(
        default_select,
        "Parsing failure \"Decimal overflow: value is greater than 10^16 max "
        "1000000000000000000 value 1000000000000000000\" in row \"" +
            getLastRow(file_name) + "\" in file \"" + getDataFilesPath() + file_name +
            ".csv\"");
  }
}

TEST_P(FsiImportDecimalTest, OutOfRangeMaxRound) {
  auto import_flag = GetParam();
  std::string file_name = "decimal_out_of_range_max_round";
  sqlCreateTable(import_flag, "(d DECIMAL(18,2))", file_name, wrapper_type_, {});
  if (import_flag) {
    TQueryResult result;
    sql(result, default_select);
    // Failure will result in empty table
    assertResultSetEqual({}, result);
  } else {
    queryAndAssertException(
        default_select,
        "Parsing failure \"Decimal overflow: value is greater than 10^16 max "
        "1000000000000000000 value 1000000000000000000\" in row \"" +
            getLastRow(file_name) + "\" in file \"" + getDataFilesPath() + file_name +
            ".csv\"");
  }
}

TEST_P(FsiImportDecimalTest, OutOfRangeMin) {
  auto import_flag = GetParam();
  std::string file_name = "decimal_out_of_range_min";
  sqlCreateTable(import_flag, "(d DECIMAL(18,2))", file_name, wrapper_type_, {});
  if (import_flag) {
    TQueryResult result;
    sql(result, default_select);
    // Failure will result in empty table
    assertResultSetEqual({}, result);
  } else {
    queryAndAssertException(
        default_select,
        "Parsing failure \"Decimal overflow: value is less than -10^16 min "
        "-1000000000000000000 value -1000000000000000000\" in row \"" +
            getLastRow(file_name) + "\" in file \"" + getDataFilesPath() + file_name +
            ".csv\"");
  }
}

TEST_P(FsiImportDecimalTest, OutOfRangeMinRound) {
  auto import_flag = GetParam();
  std::string file_name = "decimal_out_of_range_min_round";
  sqlCreateTable(import_flag, "(d DECIMAL(18,2))", file_name, wrapper_type_, {});
  if (import_flag) {
    TQueryResult result;
    sql(result, default_select);
    // Failure will result in empty table
    assertResultSetEqual({}, result);
  } else {
    queryAndAssertException(
        default_select,
        "Parsing failure \"Decimal overflow: value is less than -10^16 min "
        "-1000000000000000000 value -1000000000000000000\" in row \"" +
            getLastRow(file_name) + "\" in file \"" + getDataFilesPath() + file_name +
            ".csv\"");
  }
}

TEST_P(FsiImportDecimalTest, ImportAcceptableRejectThreshold) {
  auto import_flag = GetParam();
  if (!import_flag) {
    GTEST_SKIP() << "FSI does not support max_reject";
  }
  std::string file_name = "decimal_survival";
  sqlCreateTable(import_flag, "(d DECIMAL(18,2))", file_name, wrapper_type_, {});
  TQueryResult result;
  sql(result, default_select);
  // Only rows within bounds end up being imported
  assertResultSetEqual({{100.00}, {0.00}, {-100.00}}, result);
}

TEST_P(FsiImportDecimalTest, ImportExceedRejectThreshold) {
  auto import_flag = GetParam();
  if (!import_flag) {
    GTEST_SKIP() << "FSI does not support max_reject";
  }
  std::string file_name = "decimal_survival";
  sqlCreateTable(import_flag,
                 "(d DECIMAL(18,2))",
                 file_name,
                 wrapper_type_,
                 {},
                 {{"max_reject", "3"}});
  TQueryResult result;
  sql(result, default_select);
  // max_reject exceed, import halted
  assertResultSetEqual({}, result);
}

INSTANTIATE_TEST_SUITE_P(FsiImportDecimalParamaterizedTest,
                         FsiImportDecimalTest,
                         ::testing::Values(True, False),
                         [](const auto& info) {
                           return ((info.param) ? "Import" : "FSI");
                         });

class CsvDelimiterTest : public SelectQueryTest {};

TEST_F(CsvDelimiterTest, CSVLineDelimNewline) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i BIGINT, f DOUBLE)", {{"line_delimiter", "\\n"}}, "example_2", "csv");
  sql(query);
  queryAndAssertExample2Result();
}

TEST_F(CsvDelimiterTest, CSVDelimTab) {
  const auto& query = getCreateForeignTableQuery("(t TEXT, i BIGINT, f DOUBLE)",
                                                 {{"delimiter", "\\t"}},
                                                 "example_2",
                                                 "csv",
                                                 0,
                                                 "" + default_table_name + "",
                                                 "tsv");
  sql(query);
  queryAndAssertExample2Result();
}

TEST_F(CsvDelimiterTest, CSVFieldDelim) {
  const auto& query = getCreateForeignTableQuery("(t TEXT, i BIGINT, f DOUBLE)",
                                                 {{"delimiter", "|"}},
                                                 "example_2_field_delim",
                                                 "csv");
  sql(query);
  queryAndAssertExample2Result();
}

TEST_F(CsvDelimiterTest, CSVLineDelim) {
  const auto& query = getCreateForeignTableQuery("(t TEXT, i BIGINT, f DOUBLE)",
                                                 {{"line_delimiter", "*"}},
                                                 "example_2_line_delim",
                                                 "csv");
  sql(query);
  queryAndAssertExample2Result();
}

TEST_F(CsvDelimiterTest, CSVQuote) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i BIGINT, f DOUBLE)", {{"quote", "~"}}, "example_2_quote", "csv");
  sql(query);
  queryAndAssertExample2Result();
}

TEST_F(CsvDelimiterTest, CSVCarriageReturn) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i BIGINT, f DOUBLE)", {}, "example_2_crlf", "csv");
  sql(query);
  queryAndAssertExample2Result();
}

TEST_F(CsvDelimiterTest, CSVQuoteEscape) {
  const auto& query = getCreateForeignTableQuery("(t TEXT, i BIGINT, f DOUBLE)",
                                                 {{"quote", "a"}, {"escape", "$"}},
                                                 "example_2_quote_escape",
                                                 "csv");
  // 'a' used as quote, so "a" is a$aa in csv file
  sql(query);
  queryAndAssertExample2Result();
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
    cache = cat->getDataMgr().getPersistentStorageMgr()->getDiskCache();
    cache->clear();
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
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

class RefreshTests : public ForeignTableTest, public TempDirManager {
 protected:
  FileExtType file_ext_;
  std::vector<std::string> tmp_file_names_;
  std::vector<std::string> table_names_;
  Catalog_Namespace::Catalog* cat_;
  foreign_storage::ForeignStorageCache* cache_ = nullptr;

  void SetUp() override {
    ForeignTableTest::SetUp();
    file_ext_ = wrapper_ext(wrapper_type_);
    cat_ = &getCatalog();
    cache_ = cat_->getDataMgr().getPersistentStorageMgr()->getDiskCache();
    cache_->clear();
    sql("DROP FOREIGN TABLE IF EXISTS " + default_table_name + ";");
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    for (auto table_name : table_names_) {
      sqlDropForeignTable(0, table_name);
    }
    sqlDropForeignTable(0, default_table_name);
    ForeignTableTest::TearDown();
  }

  bool isChunkAndMetadataCached(const ChunkKey& chunk_key) {
    if (cache_->getCachedChunkIfExists(chunk_key) != nullptr &&
        cache_->isMetadataCached(chunk_key)) {
      return true;
    }
    return false;
  }

  void createFilesAndTables(const std::vector<std::string>& file_names,
                            const std::vector<NameTypePair>& column_pairs = {{"i",
                                                                              "BIGINT"}},
                            const foreign_storage::OptionsMap& table_options = {}) {
    for (size_t i = 0; i < file_names.size(); ++i) {
      tmp_file_names_.emplace_back(test_temp_dir + default_table_name +
                                   std::to_string(i) + file_ext_);
      table_names_.emplace_back(default_table_name + std::to_string(i));
      bf::copy_file(getDataFilesPath() + file_names[i] + file_ext_,
                    tmp_file_names_[i],
#if 107400 <= BOOST_VERSION
                    bf::copy_options::overwrite_existing
#else
                    bf::copy_option::overwrite_if_exists
#endif
      );
      sql("DROP FOREIGN TABLE IF EXISTS " + table_names_[i] + ";");
      sql(createForeignTableQuery(column_pairs,
                                  tmp_file_names_[i],
                                  wrapper_type_,
                                  table_options,
                                  table_names_[i]));
    }
  }

  void updateForeignSource(const std::vector<std::string>& file_names,
                           const std::vector<NameTypePair>& column_pairs = {{"i",
                                                                             "BIGINT"}},
                           const foreign_storage::OptionsMap& table_options = {}) {
    for (size_t i = 0; i < file_names.size(); ++i) {
      bf::copy_file(getDataFilesPath() + file_names[i] + file_ext_,
                    tmp_file_names_[i],
#if 107400 <= BOOST_VERSION
                    bf::copy_options::overwrite_existing
#else
                    bf::copy_option::overwrite_if_exists
#endif
      );
      if (isOdbc(wrapper_type_)) {
        // If we are in ODBC we need to recreate the ODBC table as well.
        createODBCSourceTable(
            table_names_[i], column_pairs, tmp_file_names_[i], wrapper_type_);
      }
    }
  }

  int64_t getCurrentTime() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  std::pair<int64_t, int64_t> getLastAndNextRefreshTimes(
      const std::string& table_name = "" + default_table_name + "") {
    auto table = getCatalog().getMetadataForTable(table_name, false);
    CHECK(table);
    const auto foreign_table = dynamic_cast<const foreign_storage::ForeignTable*>(table);
    CHECK(foreign_table);
    return {foreign_table->last_refresh_time, foreign_table->next_refresh_time};
  }

  void assertNullRefreshTime(int64_t refresh_time) { ASSERT_EQ(-1, refresh_time); }

  void assertRefreshTimeBetween(int64_t refresh_time,
                                int64_t start_time,
                                int64_t end_time) {
    ASSERT_GE(refresh_time, start_time);
    ASSERT_LE(refresh_time, end_time);
  }
};

TEST_F(RefreshTests, InvalidRefreshMode) {
  std::string filename = "archive_delete_file.zip";
  std::string query = "CREATE FOREIGN TABLE " + default_table_name + " (i INTEGER) "s +
                      "SERVER default_local_delimited WITH (file_path = '" +
                      getDataFilesPath() + "append_before/" + filename +
                      "', fragment_size = '1' " + ", REFRESH_UPDATE_TYPE = 'INVALID');";
  queryAndAssertException(query,
                          "Invalid value \"INVALID\" for REFRESH_UPDATE_TYPE option. "
                          "Value must be \"APPEND\" or \"ALL\".");
}

class AlteredSourceTest : public RefreshTests,
                          public ::testing::WithParamInterface<WrapperType> {
 public:
  static std::string getTestName(const ::testing::TestParamInfo<WrapperType>& info) {
    auto wrapper_type = info.param;
    std::stringstream ss;
    ss << "DataWrapper_" << wrapper_type;
    return ss.str();
  }

  void SetUp() override {
    wrapper_type_ = GetParam();
    if (wrapper_type_ == "csv" || wrapper_type_ == "parquet") {
      file_ext_ = wrapper_type_;
    } else {
      CHECK(wrapper_type_ == "regex_parser" || isOdbc(wrapper_type_));
      file_ext_ = "csv";
    }
    auto cat = &getCatalog();
    // store previous cache config
    stored_cache_config_ =
        cat->getDataMgr().getPersistentStorageMgr()->getDiskCacheConfig();
    // turn on cache for setup as required
    cat->getDataMgr().resetPersistentStorage(
        {cache_path_, File_Namespace::DiskCacheLevel::fsi}, 0, getSystemParameters());
    RefreshTests::SetUp();
    // turn off cache for test
    cat->getDataMgr().resetPersistentStorage(
        {cache_path_, File_Namespace::DiskCacheLevel::none}, 0, getSystemParameters());
  }

  void TearDown() override {
    RefreshTests::TearDown();
    auto cat = &getCatalog();
    cat->getDataMgr().resetPersistentStorage(
        stored_cache_config_, 0, getSystemParameters());
  }

 protected:
  inline static std::string cache_path_ =
      to_string(BASE_PATH) + "/" + shared::kDefaultDiskCacheDirName;

  File_Namespace::DiskCacheConfig stored_cache_config_;
};

TEST_P(AlteredSourceTest, FragmentRemoved) {
  SKIP_IF_DISTRIBUTED(
      "Test requires cache to be turned off in distributed mode, which is not an "
      "implemented configuration at this time");

  createFilesAndTables({"1"});
  sqlAndCompareResult("SELECT COUNT(i) FROM " + default_table_name + "0", {{i(1)}});
  sql("ALTER SYSTEM CLEAR CPU MEMORY");
  sql("ALTER SYSTEM CLEAR GPU MEMORY");
  updateForeignSource({"empty"});

  std::string expected_exception;

  if (wrapper_type_ == "csv" || wrapper_type_ == "regex_parser") {
    expected_exception =
        "Unexpected number of bytes while loading from foreign data source: expected "
        "2 , obtained 1 bytes. Please use the \"REFRESH FOREIGN TABLES\" command on "
        "the foreign table if data source has been updated. Foreign table: "
        "test_foreign_table0";
  } else if (wrapper_type_ == "parquet") {
    expected_exception =
        "Unable to read from foreign data source, possible cause is an unexpected "
        "change of source. Please use the \"REFRESH FOREIGN TABLES\" command on the "
        "foreign table if data source has been updated. Foreign table: "
        "test_foreign_table0";
  } else if (isOdbc(wrapper_type_)) {
    expected_exception =
        "Unexpected number of records while loading from foreign data source: "
        "expected 1 , obtained 0 records. Please use the \"REFRESH FOREIGN TABLES\" "
        "command on the foreign table if data source has been updated. Foreign table: "
        "test_foreign_table0";
  } else {
    UNREACHABLE() << "unexpected case in test";
  }
  queryAndAssertException("SELECT COUNT(i) FROM " + default_table_name + "0",
                          expected_exception);
}

INSTANTIATE_TEST_SUITE_P(FragmentRemovedForDataWrapper,
                         AlteredSourceTest,
                         testing::Values("csv", "regex_parser", "parquet", "postgres"),
                         AlteredSourceTest::getTestName);

class RefreshMetadataTypeTest : public SelectQueryTest {};
TEST_F(RefreshMetadataTypeTest, ScalarTypes) {
  const auto& query = getCreateForeignTableQuery(
      "(b BOOLEAN, t TINYINT, s SMALLINT, i INTEGER, bi BIGINT, f FLOAT, "
      "dc DECIMAL(10, 5), tm TIME, tp TIMESTAMP, d DATE, txt TEXT, "
      "txt_2 TEXT ENCODING NONE)",
      {},
      "scalar_types",
      "csv",
      0,
      default_table_name,
      "csv");
  sql(query);
  sql(default_select);
  sql("REFRESH FOREIGN TABLES " + default_table_name + ";");
  sql(default_select);
}

TEST_F(RefreshMetadataTypeTest, ArrayTypes) {
  const auto& query = getCreateForeignTableQuery(
      "(index int, b BOOLEAN[], t TINYINT[], s SMALLINT[], i INTEGER[], bi BIGINT[], f "
      "FLOAT[], "
      "tm "
      "TIME[], tp TIMESTAMP[], "
      "d DATE[], txt TEXT[], fixedpoint DECIMAL(10,5)[])",
      {},
      "array_types",
      "csv",
      0,
      default_table_name,
      "csv");
  sql(query);
  sql(default_select);
  sql("REFRESH FOREIGN TABLES " + default_table_name + ";");
  sql(default_select);
}

TEST_F(RefreshMetadataTypeTest, GeoTypes) {
  const auto& query = getCreateForeignTableQuery(
      "(index int, p POINT, l LINESTRING, poly POLYGON, multipoly MULTIPOLYGON)",
      {},
      "geo_types_valid",
      "csv",
      0,
      default_table_name,
      "csv");
  sql(query);
  sql(default_select);
  sql("REFRESH FOREIGN TABLES " + default_table_name + ";");
  sql(default_select);
}

class RefreshParamTests : public RefreshTests,
                          public ::testing::WithParamInterface<WrapperType> {
 protected:
  void SetUp() override {
    SKIP_SETUP_IF_DISTRIBUTED("Test needs local metadata");
    wrapper_type_ = GetParam();
    RefreshTests::SetUp();
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    RefreshTests::TearDown();
  }

  void assertExpectedCacheStatePostScan(ChunkKey& chunk_key) {
    bool cache_on_scan = (wrapper_type_ == "csv" || is_regex(wrapper_type_));
    if (cache_on_scan) {
      ASSERT_NE(cache_->getCachedChunkIfExists(chunk_key), nullptr);
    } else {
      ASSERT_EQ(cache_->getCachedChunkIfExists(chunk_key), nullptr);
    }
  }

  std::string getSelect(const std::string& table_name,
                        const std::string& col_name) const {
    return "SELECT * FROM " + table_name + " ORDER BY " + col_name + ";";
  }
};

INSTANTIATE_TEST_SUITE_P(RefreshParamTestsParameterizedTests,
                         RefreshParamTests,
                         ::testing::ValuesIn(local_wrappers),
                         [](const auto& info) { return info.param; });

TEST_P(RefreshParamTests, SingleTable) {
  // Create initial files and tables
  createFilesAndTables({"0"});

  // Read from table to populate cache.
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(0)}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  updateForeignSource({"1"});

  // Confirm changing file hasn't changed cached results
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(0)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + table_names_[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(1)}});
}

TEST_P(RefreshParamTests, FragmentSkip) {
  // Create initial files and tables
  createFilesAndTables({"0", "1"});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names_[0] + " WHERE i >= 3;", {});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  assertExpectedCacheStatePostScan(orig_key0);
  ASSERT_TRUE(cache_->isMetadataCached(orig_key0));

  sqlAndCompareResult("SELECT * FROM " + table_names_[1] + " WHERE i >= 3;", {});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat_, table_names_[1], {1, 0});
  assertExpectedCacheStatePostScan(orig_key1);
  ASSERT_TRUE(cache_->isMetadataCached(orig_key1));

  updateForeignSource({"2", "3"});

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names_[0] + " WHERE i >= 3;", {});
  assertExpectedCacheStatePostScan(orig_key0);
  ASSERT_TRUE(cache_->isMetadataCached(orig_key0));

  sqlAndCompareResult("SELECT * FROM " + table_names_[1] + " WHERE i >= 3;", {});
  assertExpectedCacheStatePostScan(orig_key1);
  ASSERT_TRUE(cache_->isMetadataCached(orig_key1));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + table_names_[0] + ", " + table_names_[1] + ";");

  // Compare new results
  assertExpectedCacheStatePostScan(orig_key0);
  ASSERT_TRUE(cache_->isMetadataCached(orig_key0));
  assertExpectedCacheStatePostScan(orig_key1);
  ASSERT_TRUE(cache_->isMetadataCached(orig_key1));
  sqlAndCompareResult("SELECT * FROM " + table_names_[0] + " WHERE i >= 3;", {});
  sqlAndCompareResult("SELECT * FROM " + table_names_[1] + " WHERE i >= 3;", {{i(3)}});
  assertExpectedCacheStatePostScan(orig_key0);
  ASSERT_TRUE(cache_->isMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
}

TEST_P(RefreshParamTests, TwoTable) {
  // Create initial files and tables
  createFilesAndTables({"0", "1"});

  // Read from table to populate cache.
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(0)}});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));

  sqlAndCompareResult(getSelect(table_names_[1], "i"), {{i(1)}});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat_, table_names_[1], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  updateForeignSource({"2", "3"});

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(0)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));

  sqlAndCompareResult(getSelect(table_names_[1], "i"), {{i(1)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + table_names_[0] + ", " + table_names_[1] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(2)}});
  sqlAndCompareResult(getSelect(table_names_[1], "i"), {{i(3)}});
}

TEST_P(RefreshParamTests, EvictTrue) {
  // Create initial files and tables
  createFilesAndTables({"0"});

  // Read from table to populate cache.
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(0)}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  updateForeignSource({"1"});

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(0)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  auto start_time = getCurrentTime();
  sql("REFRESH FOREIGN TABLES " + table_names_[0] + " WITH (evict = true);");
  auto end_time = getCurrentTime();

  // Compare new results
  ASSERT_EQ(cache_->getCachedChunkIfExists(orig_key), nullptr);
  ASSERT_FALSE(cache_->isMetadataCached(orig_key));
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(1)}});

  auto [last_refresh_time, next_refresh_time] =
      getLastAndNextRefreshTimes(table_names_[0]);
  assertRefreshTimeBetween(last_refresh_time, start_time, end_time);
  assertNullRefreshTime(next_refresh_time);
}

TEST_P(RefreshParamTests, TwoColumn) {
  // Create initial files and tables
  createFilesAndTables({"two_col_1_2"}, {{"i", "BIGINT"}, {"i2", "BIGINT"}});

  // Read from table to populate cache.
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(1), i(2)}});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat_, table_names_[0], {2, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  updateForeignSource({"two_col_3_4"}, {{"i", "BIGINT"}, {"i2", "BIGINT"}});

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(1), i(2)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + table_names_[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(3), i(4)}});
}

TEST_P(RefreshParamTests, ChangeSchema) {
  // Create initial files and tables
  createFilesAndTables({"1"});

  // Read from table to populate cache.
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(1)}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  updateForeignSource({"two_col_3_4"}, {{"i", "BIGINT"}, {"i2", "BIGINT"}});

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(1)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  if (isOdbc(wrapper_type_)) {
    // ODBC can handle this case fine, since it can select individual columns.
    sql("REFRESH FOREIGN TABLES " + table_names_[0] + ";");
  } else if (is_regex(wrapper_type_)) {
    // When the file changes in a way that results in a regex mismatch, the regex parser
    // wrapper should return rows with all null values.
    sql("REFRESH FOREIGN TABLES " + table_names_[0] + ";");
    sqlAndCompareResult("SELECT * FROM " + table_names_[0] + ";", {{NULL_BIGINT}});
  } else {
    try {
      sql("REFRESH FOREIGN TABLES " + table_names_[0] + ";");
      FAIL() << "An exception should have been thrown";
    } catch (const std::exception& e) {
      ASSERT_NE(strstr(e.what(), "Mismatched number of logical columns"), nullptr);
    }
  }
}

TEST_P(RefreshParamTests, AddFrags) {
  // Create initial files and tables
  createFilesAndTables({"two_row_1_2"}, {{"i", "BIGINT"}}, {{"fragment_size", "1"}});

  // Read from table to populate cache.
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(1)}, {i(2)}});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat_, table_names_[0], {1, 1});
  ChunkKey orig_key2 = getChunkKeyFromTable(*cat_, table_names_[0], {1, 2});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  updateForeignSource({"three_row_3_4_5"}, {{"i", "BIGINT"}}, {{"fragment_size", "1"}});

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(1)}, {i(2)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + table_names_[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  assertExpectedCacheStatePostScan(orig_key2);
  ASSERT_TRUE(cache_->isMetadataCached(orig_key2));
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(3)}, {i(4)}, {i(5)}});
}

TEST_P(RefreshParamTests, SubFrags) {
  // Create initial files and tables
  createFilesAndTables({"three_row_3_4_5"}, {{"i", "BIGINT"}}, {{"fragment_size", "1"}});

  // Read from table to populate cache.
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(3)}, {i(4)}, {i(5)}});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat_, table_names_[0], {1, 1});
  ChunkKey orig_key2 = getChunkKeyFromTable(*cat_, table_names_[0], {1, 2});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key2));

  updateForeignSource({"two_row_1_2"}, {{"i", "BIGINT"}}, {{"fragment_size", "1"}});

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(3)}, {i(4)}, {i(5)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key2));

  sql("REFRESH FOREIGN TABLES " + table_names_[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  ASSERT_EQ(cache_->getCachedChunkIfExists(orig_key2), nullptr);
  ASSERT_FALSE(cache_->isMetadataCached(orig_key2));
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(1)}, {i(2)}});
}

TEST_P(RefreshParamTests, TwoFrags) {
  // Create initial files and tables
  createFilesAndTables({"two_row_1_2"}, {{"i", "BIGINT"}}, {{"fragment_size", "1"}});

  // Read from table to populate cache.
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(1)}, {i(2)}});
  ChunkKey orig_key0 = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  ChunkKey orig_key1 = getChunkKeyFromTable(*cat_, table_names_[0], {1, 1});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  updateForeignSource({"two_row_3_4"}, {{"i", "BIGINT"}}, {{"fragment_size", "1"}});

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(1)}, {i(2)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + table_names_[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key0));
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key1));
  sqlAndCompareResult(getSelect(table_names_[0], "i"), {{i(3)}, {i(4)}});
}

TEST_P(RefreshParamTests, String) {
  createFilesAndTables({"a"}, {{"t", "TEXT"}});

  // Read from table to populate cache.
  sqlAndCompareResult(getSelect(table_names_[0], "t"), {{"a"}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  updateForeignSource({"b"}, {{"t", "TEXT"}});

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult(getSelect(table_names_[0], "t"), {{"a"}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + table_names_[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));
  sqlAndCompareResult(getSelect(table_names_[0], "t"), {{"b"}});
}

TEST_P(RefreshParamTests, BulkMissingRows) {
  createFilesAndTables({"three_row_3_4_5"}, {{"i", "BIGINT"}});

  sqlAndCompareResult("SELECT * FROM "s + table_names_[0] + " ORDER BY i;",
                      {{i(3)}, {i(4)}, {i(5)}});
  updateForeignSource({"two_row_1_2"}, {{"i", "BIGINT"}});
  sql("REFRESH FOREIGN TABLES " + table_names_[0] + ";");
  sqlAndCompareResult("SELECT * FROM "s + table_names_[0] + " ORDER BY i;",
                      {{i(1)}, {i(2)}});
}

class RefreshDeviceTests : public RefreshTests,
                           public ::testing::WithParamInterface<TExecuteMode::type> {};

INSTANTIATE_TEST_SUITE_P(RefreshDeviceTestsParameterizedTests,
                         RefreshDeviceTests,
                         ::testing::Values(TExecuteMode::CPU, TExecuteMode::GPU),
                         [](const auto& info) {
                           return ((info.param == TExecuteMode::GPU) ? "GPU" : "CPU");
                         });

TEST_P(RefreshDeviceTests, Device) {
  SKIP_IF_DISTRIBUTED("Test relies on local cache access");

  if (!setExecuteMode(GetParam())) {
    return;
  }
  // Create initial files and tables
  createFilesAndTables({"0"});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names_[0] + ";", {{i(0)}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Change underlying file
  updateForeignSource({"1"});

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names_[0] + ";", {{i(0)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + table_names_[0] + ";");

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));
  sqlAndCompareResult("SELECT * FROM " + table_names_[0] + ";", {{i(1)}});
}

class RefreshSyntaxTests : public RefreshTests,
                           public ::testing::WithParamInterface<EvictCacheString> {};

INSTANTIATE_TEST_SUITE_P(RefreshSyntaxTestsParameterizedTests,
                         RefreshSyntaxTests,
                         ::testing::Values(" WITH (evict = false)",
                                           " WITH (EVICT = FALSE)"));

TEST_P(RefreshSyntaxTests, EvictFalse) {
  SKIP_IF_DISTRIBUTED("Test relies on local cache access");

  // Create initial files and tables
  createFilesAndTables({"0"});

  // Read from table to populate cache.
  sqlAndCompareResult("SELECT * FROM " + table_names_[0] + ";", {{i(0)}});
  ChunkKey orig_key = getChunkKeyFromTable(*cat_, table_names_[0], {1, 0});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Change underlying file
  updateForeignSource({"1"});

  // Confirm chaning file hasn't changed cached results
  sqlAndCompareResult("SELECT * FROM " + table_names_[0] + ";", {{i(0)}});
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));

  // Refresh command
  auto start_time = getCurrentTime();
  sql("REFRESH FOREIGN TABLES " + table_names_[0] + GetParam() + ";");
  auto end_time = getCurrentTime();

  // Compare new results
  ASSERT_TRUE(isChunkAndMetadataCached(orig_key));
  sqlAndCompareResult("SELECT * FROM " + table_names_[0] + ";", {{i(1)}});

  auto [last_refresh_time, next_refresh_time] =
      getLastAndNextRefreshTimes(table_names_[0]);
  assertRefreshTimeBetween(last_refresh_time, start_time, end_time);
  assertNullRefreshTime(next_refresh_time);
}

class RefreshSyntaxErrorTests : public RefreshTests {};

TEST_F(RefreshSyntaxErrorTests, InvalidEvictValue) {
  createFilesAndTables({"0"});
  std::string query{"REFRESH FOREIGN TABLES " + table_names_[0] +
                    " WITH (evict = 'invalid');"};
  queryAndAssertException(query,
                          "Invalid value \"invalid\" provided for EVICT "
                          "option. Value must be either \"true\" or \"false\".");
}

TEST_F(RefreshSyntaxErrorTests, InvalidOption) {
  createFilesAndTables({"0"});
  std::string query{"REFRESH FOREIGN TABLES " + table_names_[0] +
                    " WITH (invalid_key = false);"};
  queryAndAssertException(query,
                          "Invalid option \"INVALID_KEY\" provided for "
                          "refresh command. Only \"EVICT\" option is supported.");
}

class AppendRefreshTestCSV : public RecoverCacheQueryTest, public TempDirManager {
 protected:
  void SetUp() override {
    RecoverCacheQueryTest::SetUp();
    sqlDropForeignTable(0, default_table_name);
    recursive_copy(getDataFilesPath() + "append_before", test_temp_dir);
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    sqlDropForeignTable(0, default_table_name);
    RecoverCacheQueryTest::TearDown();
  }
};

TEST_F(AppendRefreshTestCSV, MissingFileArchive) {
  int fragment_size = 1;
  std::string filename = "archive_delete_file.zip";

  std::string query = "CREATE FOREIGN TABLE " + default_table_name + " (i INTEGER) "s +
                      "SERVER default_local_delimited WITH (file_path = '" +
                      test_temp_dir + filename + "', fragment_size = '" +
                      std::to_string(fragment_size) +
                      "', REFRESH_UPDATE_TYPE = 'APPEND');";
  sql(query);

  std::string select = "SELECT * FROM "s + default_table_name + " ORDER BY i;";
  // Read from table
  sqlAndCompareResult(select, {{i(1)}, {i(2)}});

  // Modify files
  bf::remove_all(test_temp_dir);
  recursive_copy(getDataFilesPath() + "append_after", test_temp_dir);

  // Refresh command
  queryAndAssertException(
      "REFRESH FOREIGN TABLES " + default_table_name + ";",
      "Foreign table refreshed with APPEND mode missing archive entry "
      "\"single_file_delete_rows.csv\" from file \"archive_delete_file.zip\".");
}

class AppendRefreshBase : public RecoverCacheQueryTest, public TempDirManager {
 protected:
  const std::string table_name_ = "refresh_tmp";

  std::string file_name_;
  int32_t fragment_size_{1};
  bool recover_cache_{false};
  bool is_evict_{false};

  void SetUp() override {
    RecoverCacheQueryTest::SetUp();
    sqlDropForeignTable(0, table_name_);
    recursive_copy(getDataFilesPath() + "append_before", test_temp_dir);
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    sqlDropForeignTable(0, table_name_);
    RecoverCacheQueryTest::TearDown();
  }

  std::string evictString() {
    return (is_evict_) ? " WITH (evict = true)" : " WITH (evict = false)";
  }

  void overwriteSourceDir(const std::vector<ColumnPair>& column_pairs) {
    overwriteTempDir(getDataFilesPath() + "append_after");
    if (isOdbc(wrapper_type_)) {
      createODBCSourceTable(
          table_name_, column_pairs, test_temp_dir + file_name_, wrapper_type_);
    }
  }

  void renameSourceDir(const std::string& new_dir_name) {
    auto current_dir_path = boost::filesystem::path(test_temp_dir) / file_name_;
    if (boost::filesystem::exists(current_dir_path)) {
      boost::filesystem::remove_all(current_dir_path);
    }
    auto new_dir_path = boost::filesystem::path(test_temp_dir) / new_dir_name;
    ASSERT_TRUE(boost::filesystem::exists(new_dir_path));
    boost::filesystem::rename(new_dir_path, current_dir_path);
  }

  void renameToEmptyDir() {
    auto current_dir_path = boost::filesystem::path(test_temp_dir) / file_name_;
    if (boost::filesystem::exists(current_dir_path)) {
      boost::filesystem::remove_all(current_dir_path);
    }
    auto empty_dir_path = boost::filesystem::path(test_temp_dir) / "empty_dir";
    boost::filesystem::create_directory(empty_dir_path);
    boost::filesystem::rename(empty_dir_path, current_dir_path);
  }

  void recoverCacheIfSpecified() {
    if (recover_cache_) {
      resetPersistentStorageMgr({cache_path_, File_Namespace::DiskCacheLevel::fsi});
    }
  }

  static std::vector<std::vector<int32_t>> createSubKeys(size_t num_chunks) {
    std::vector<std::vector<int32_t>> chunk_subkeys;
    for (int32_t i = 0; i < static_cast<int>(num_chunks); ++i) {
      chunk_subkeys.push_back({1, i});
    }
    return chunk_subkeys;
  }
};

class FragmentSizesAppendRefreshTest
    : public AppendRefreshBase,
      public ::testing::WithParamInterface<
          std::tuple<FragmentSizeType, WrapperType, FileNameType, RecoverCacheFlag>> {
 public:
  static std::string getTestName(
      const ::testing::TestParamInfo<
          std::tuple<FragmentSizeType, WrapperType, FileNameType, RecoverCacheFlag>>&
          info) {
    auto [fragment_size, wrapper_type, file_name, recover_cache] = info.param;
    std::replace(file_name.begin(), file_name.end(), '.', '_');
    std::stringstream ss;
    ss << "FragmentSize_" << fragment_size << "_DataWrapper_" << wrapper_type
       << "_FileName" << file_name << (recover_cache ? "_RecoverCache" : "");
    return ss.str();
  }

 protected:
  void SetUp() override {
    std::tie(fragment_size_, wrapper_type_, file_name_, recover_cache_) = GetParam();
    AppendRefreshBase::SetUp();
  }

  void sqlCreateTestTable() {
    sql(createForeignTableQuery({{"i", "BIGINT"}},
                                test_temp_dir + file_name_,
                                wrapper_type_,
                                {{"FRAGMENT_SIZE", std::to_string(fragment_size_)},
                                 {"REFRESH_UPDATE_TYPE", "APPEND"}},
                                table_name_,
                                {},
                                0));
  }
};

/*
 * It would be too runtime intensive to run all combinations of frag sizes and refresh
 * options without much additional coverage, so we only re-test the recovery with single
 * row fragments.
 */
INSTANTIATE_TEST_SUITE_P(AppendParamaterizedTestsCsv,
                         FragmentSizesAppendRefreshTest,
                         testing::Combine(testing::Values(1, 4, 3200000),
                                          testing::Values("csv"),
                                          testing::Values("single_file.csv",
                                                          "single_file.zip",
                                                          "csv_dir_file",
                                                          "csv_dir_file_multi",
                                                          "dir_file_multi.zip"),
                                          testing::Values(false)),
                         FragmentSizesAppendRefreshTest::getTestName);
INSTANTIATE_TEST_SUITE_P(AppendParamaterizedTestsCsvRecover,
                         FragmentSizesAppendRefreshTest,
                         testing::Combine(testing::Values(1),
                                          testing::Values("csv"),
                                          testing::Values("single_file.csv",
                                                          "single_file.zip",
                                                          "csv_dir_file",
                                                          "csv_dir_file_multi",
                                                          "dir_file_multi.zip"),
                                          testing::Values(true)),
                         FragmentSizesAppendRefreshTest::getTestName);

INSTANTIATE_TEST_SUITE_P(AppendParamaterizedTestsParquet,
                         FragmentSizesAppendRefreshTest,
                         testing::Combine(testing::Values(1, 4, 3200000),
                                          testing::Values("parquet"),
                                          testing::Values("single_file.parquet",
                                                          "parquet_dir_file",
                                                          "parquet_dir_file_multi"),
                                          testing::Values(false)),
                         FragmentSizesAppendRefreshTest::getTestName);
INSTANTIATE_TEST_SUITE_P(AppendParamaterizedTestsParquetRecover,
                         FragmentSizesAppendRefreshTest,
                         testing::Combine(testing::Values(1),
                                          testing::Values("parquet"),
                                          testing::Values("single_file.parquet",
                                                          "parquet_dir_file",
                                                          "parquet_dir_file_multi"),
                                          testing::Values(true)),
                         FragmentSizesAppendRefreshTest::getTestName);

INSTANTIATE_TEST_SUITE_P(AppendParamaterizedTestsRegexParser,
                         FragmentSizesAppendRefreshTest,
                         testing::Combine(testing::Values(1, 4, 3200000),
                                          testing::Values("regex_parser"),
                                          testing::Values("single_file.csv",
                                                          "single_file.zip",
                                                          "csv_dir_file",
                                                          "csv_dir_file_multi",
                                                          "dir_file_multi.zip"),
                                          testing::Values(false)),
                         FragmentSizesAppendRefreshTest::getTestName);
INSTANTIATE_TEST_SUITE_P(AppendParamaterizedTestsRegexParserRecover,
                         FragmentSizesAppendRefreshTest,
                         testing::Combine(testing::Values(1),
                                          testing::Values("regex_parser"),
                                          testing::Values("single_file.csv",
                                                          "single_file.zip",
                                                          "csv_dir_file",
                                                          "csv_dir_file_multi",
                                                          "dir_file_multi.zip"),
                                          testing::Values(true)),
                         FragmentSizesAppendRefreshTest::getTestName);

INSTANTIATE_TEST_SUITE_P(AppendParamaterizedTestsOdbc,
                         FragmentSizesAppendRefreshTest,
                         testing::Combine(testing::Values(1, 4, 3200000),
                                          testing::Values("postgres", "sqlite"),
                                          testing::Values("single_file.csv"),
                                          testing::Values(false)),
                         FragmentSizesAppendRefreshTest::getTestName);
INSTANTIATE_TEST_SUITE_P(AppendParamaterizedTestsOdbcRecover,
                         FragmentSizesAppendRefreshTest,
                         testing::Combine(testing::Values(1),
                                          testing::Values("postgres", "sqlite"),
                                          testing::Values("single_file.csv"),
                                          testing::Values(true)),
                         FragmentSizesAppendRefreshTest::getTestName);

TEST_P(FragmentSizesAppendRefreshTest, AppendFrags) {
  SKIP_IF_DISTRIBUTED("Test relies on local cache access");

  std::string count_query = "SELECT COUNT(*) FROM "s + table_name_ + ";";
  std::string select_query = "SELECT * FROM "s + table_name_ + " ORDER BY i;";

  sqlCreateTestTable();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});

  // Overwrite source dir with append_after data and update odbc source table (if
  // necessary).
  overwriteSourceDir({{"i", "BIGINT"}});

  recoverCacheIfSpecified();

  size_t original_chunks = std::ceil(double(2) / fragment_size_);
  size_t final_chunks = std::ceil(double(5) / fragment_size_);
  auto cat = &getCatalog();

  // cache contains all original chunks
  ASSERT_TRUE(
      does_cache_contain_chunks(cat, table_name_, createSubKeys(original_chunks)));
  sqlAndCompareResult(count_query, {{i(2)}});
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");

  // Check count to ensure metadata is updated
  sqlAndCompareResult(count_query, {{i(5)}});
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});

  ASSERT_EQ(recover_cache_, isTableDatawrapperRestored(table_name_));

  // cache contains all original+new chunks
  ASSERT_TRUE(does_cache_contain_chunks(cat, table_name_, createSubKeys(final_chunks)));
}

// Test that string dictionaries are populated correctly after an append
class StringDictAppendTest
    : public AppendRefreshBase,
      public ::testing::WithParamInterface<std::tuple<FragmentSizeType,
                                                      WrapperType,
                                                      FileNameType,
                                                      RecoverCacheFlag,
                                                      EvictCacheFlag>> {
 public:
  static std::string getTestName(
      const ::testing::TestParamInfo<std::tuple<FragmentSizeType,
                                                WrapperType,
                                                FileNameType,
                                                RecoverCacheFlag,
                                                EvictCacheFlag>>& info) {
    auto [fragment_size, wrapper_type, file_name, recover_cache, is_evict] = info.param;
    std::stringstream ss;
    ss << "Fragment_Size_" << fragment_size << "_Data_Wrapper_" << wrapper_type
       << ((is_evict) ? "_evict" : "");
    return ss.str();
  }

 protected:
  const std::string table_name2_ = "refresh_tmp2";

  void SetUp() override {
    std::tie(fragment_size_, wrapper_type_, file_name_, recover_cache_, is_evict_) =
        GetParam();
    AppendRefreshBase::SetUp();
    sqlDropForeignTable(0, table_name2_);
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    sqlDropForeignTable(0, table_name2_);
    AppendRefreshBase::TearDown();
  }

  void sqlCreateTestTable(const std::string& custom_table_name) {
    sql(createForeignTableQuery({{"txt", "TEXT"}},
                                test_temp_dir + file_name_,
                                wrapper_type_,
                                {{"FRAGMENT_SIZE", std::to_string(fragment_size_)},
                                 {"REFRESH_UPDATE_TYPE", "APPEND"}},
                                custom_table_name,
                                {},
                                0));
  }
};

INSTANTIATE_TEST_SUITE_P(StringDictAppendParamaterizedTestsCsv,
                         StringDictAppendTest,
                         testing::Combine(testing::Values(2, 5, 3200000),
                                          testing::Values("csv"),
                                          testing::Values("csv_string_dir"),
                                          testing::Values(false),
                                          testing::Values(true, false)),
                         StringDictAppendTest::getTestName);

INSTANTIATE_TEST_SUITE_P(StringDictAppendParamaterizedTestsParquet,
                         StringDictAppendTest,
                         testing::Combine(testing::Values(2, 5, 3200000),
                                          testing::Values("parquet"),
                                          testing::Values("parquet_string_dir"),
                                          testing::Values(false),
                                          testing::Values(true, false)),
                         StringDictAppendTest::getTestName);

INSTANTIATE_TEST_SUITE_P(StringDictAppendParamaterizedTestsRegexParser,
                         StringDictAppendTest,
                         testing::Combine(testing::Values(2, 5, 3200000),
                                          testing::Values("regex_parser"),
                                          testing::Values("csv_string_dir"),
                                          testing::Values(false),
                                          testing::Values(true, false)),
                         StringDictAppendTest::getTestName);

// Single fragment size parameterization for recovering from disk
INSTANTIATE_TEST_SUITE_P(StringDictAppendParamaterizedTestsCsvFromDisk,
                         StringDictAppendTest,
                         testing::Combine(testing::Values(2),
                                          testing::Values("csv"),
                                          testing::Values("csv_string_dir"),
                                          testing::Values(true),
                                          testing::Values(true, false)),
                         StringDictAppendTest::getTestName);

INSTANTIATE_TEST_SUITE_P(StringDictAppendParamaterizedTestsParquetFromDisk,
                         StringDictAppendTest,
                         testing::Combine(testing::Values(2),
                                          testing::Values("parquet"),
                                          testing::Values("parquet_string_dir"),
                                          testing::Values(true),
                                          testing::Values(true, false)),
                         StringDictAppendTest::getTestName);

INSTANTIATE_TEST_SUITE_P(StringDictAppendParamaterizedTestsRegexParserFromDisk,
                         StringDictAppendTest,
                         testing::Combine(testing::Values(2),
                                          testing::Values("regex_parser"),
                                          testing::Values("csv_string_dir"),
                                          testing::Values(true),
                                          testing::Values(true, false)),
                         StringDictAppendTest::getTestName);

INSTANTIATE_TEST_SUITE_P(
    StringDictAppendParamaterizedTestsOdbcFromDisk,
    StringDictAppendTest,
    testing::Combine(testing::Values(2),
                     testing::Values("postgres", "sqlite"),
                     testing::Values("csv_string_dir/single_file.csv"),
                     testing::Values(true),
                     testing::Values(true, false)),
    StringDictAppendTest::getTestName);

TEST_P(StringDictAppendTest, AppendStringDictFilter) {
  sqlCreateTestTable(table_name_);
  sqlAndCompareResult("SELECT count(txt) from " + table_name_ + " WHERE txt = 'a';",
                      {{i(1)}});
  recoverCacheIfSpecified();
  overwriteSourceDir({{"txt", "TEXT"}});
  sql("REFRESH FOREIGN TABLES " + table_name_ + evictString() + ";");
  sqlAndCompareResult("SELECT count(txt) from " + table_name_ + " WHERE txt = 'aaaa';",
                      {{i(1)}});
}

TEST_P(StringDictAppendTest, AppendStringDictJoin) {
  foreign_storage::OptionsMap options{{"FRAGMENT_SIZE", std::to_string(fragment_size_)},
                                      {"REFRESH_UPDATE_TYPE", "APPEND"},
                                      {"PARTITIONS", "REPLICATED"}};

  std::string name_1 = table_name_;
  std::string name_2 = table_name2_;
  for (auto const& name : {name_1, name_2}) {
    sql(createForeignTableQuery({{"txt", "TEXT"}},
                                test_temp_dir + file_name_,
                                wrapper_type_,
                                options,
                                name,
                                {},
                                0));
  }

  std::string join = "SELECT t1.txt, t2.txt FROM " + name_1 + " AS t1 JOIN " + name_2 +
                     " AS t2 ON t1.txt = t2.txt ORDER BY t1.txt;";

  sqlAndCompareResult(join, {{"a", "a"}, {"aa", "aa"}, {"aaa", "aaa"}});
  recoverCacheIfSpecified();
  std::vector<ColumnPair> table_schema = {{"txt", "TEXT"}};
  overwriteSourceDir(table_schema);
  // Need an extra createODBCSourceTable() call as only the first table is handled by
  // overwriteSourceDir()
  if (isOdbc(wrapper_type_)) {
    createODBCSourceTable(
        name_2, table_schema, test_temp_dir + file_name_, wrapper_type_);
  }

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + name_1 + evictString() + ";");
  sql("REFRESH FOREIGN TABLES " + name_2 + evictString() + ";");

  sqlAndCompareResult(join,
                      {{"a", "a"},
                       {"aa", "aa"},
                       {"aaa", "aaa"},
                       {"aaaa", "aaaa"},
                       {"aaaaa", "aaaaa"},
                       {"aaaaaa", "aaaaaa"}});
}

class DataWrapperAppendRefreshTest
    : public AppendRefreshBase,
      public ::testing::WithParamInterface<std::tuple<WrapperType, RecoverCacheFlag>> {
 public:
  static std::string testParamsToString(
      const std::tuple<WrapperType, RecoverCacheFlag>& params) {
    std::stringstream ss;
    ss << "DataWrapper_" << std::get<0>(params)
       << (std::get<1>(params) ? "_RecoverCache" : "");
    return ss.str();
  }

 protected:
  void SetUp() override {
    std::tie(wrapper_type_, recover_cache_) = GetParam();
    AppendRefreshBase::SetUp();
  }

  void sqlCreateTestTable(const foreign_storage::OptionsMap& additional_options = {},
                          const std::vector<NameTypePair>& column_pairs = {{"i",
                                                                            "BIGINT"}},
                          const std::vector<NameTypePair>& odbc_columns = {}) {
    foreign_storage::OptionsMap options{{"FRAGMENT_SIZE", std::to_string(fragment_size_)},
                                        {"REFRESH_UPDATE_TYPE", "APPEND"}};
    options.insert(additional_options.begin(), additional_options.end());
    sql(createForeignTableQuery(column_pairs,
                                test_temp_dir + file_name_,
                                wrapper_type_,
                                options,
                                table_name_,
                                odbc_columns,
                                0));
  }

  void sqlCreateTestTableWithRollOffOption() {
    sqlCreateTestTable({{"ALLOW_FILE_ROLL_OFF", "TRUE"}});
  }

  std::string getRemovedFileErrorMessage(const std::string& missing_file_name) {
    return "Refresh of foreign table created with \"APPEND\" update type failed as "
           "file \"" +
           test_temp_dir + file_name_ + "/" + missing_file_name +
           wrapper_ext(wrapper_type_) + "\" was removed.";
  }

  const std::string select_query = "SELECT * FROM "s + table_name_ + " ORDER BY i;";
};

INSTANTIATE_TEST_SUITE_P(AppendParamaterizedTests,
                         DataWrapperAppendRefreshTest,
                         ::testing::Combine(::testing::ValuesIn(local_wrappers),
                                            ::testing::Values(true, false)),
                         [](const auto& info) {
                           return DataWrapperAppendRefreshTest::testParamsToString(
                               info.param);
                         });

TEST_P(DataWrapperAppendRefreshTest, AppendNothingGeo) {
  SKIP_IF_DISTRIBUTED("Test relies on local cache access");

  file_name_ = "geo_types_valid"s + wrapper_ext(wrapper_type_);
  std::vector<NameTypePair> odbc_columns{};
  if (isOdbc(wrapper_type_)) {
    odbc_columns = {{"id", "INT"},
                    {"p", "TEXT"},
                    {"l", "TEXT"},
                    {"poly", "TEXT"},
                    {"multipoly", "TEXT"}};
  }
  foreign_storage::OptionsMap additional_options;
  if (wrapper_type_ == "regex_parser") {
    additional_options["LINE_REGEX"] = "(\\d+),\\s*" + get_line_geo_regex(4);
  }
  sqlCreateTestTable(additional_options,
                     {{"i", "INT"},
                      {"p", "POINT"},
                      {"l", "LINESTRING"},
                      {"poly", "POLYGON"},
                      {"mpoly", "MULTIPOLYGON"}},
                     odbc_columns);
  sqlAndCompareResult(select_query, getExpectedGeoTypesResult());

  recoverCacheIfSpecified();
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, getExpectedGeoTypesResult());

  ASSERT_EQ(recover_cache_, isTableDatawrapperRestored(table_name_));
}

TEST_P(DataWrapperAppendRefreshTest, AppendNothing) {
  SKIP_IF_DISTRIBUTED("Test relies on local cache access");

  file_name_ = "single_file"s + wrapper_ext(wrapper_type_);
  sqlCreateTestTable();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  recoverCacheIfSpecified();
  ASSERT_EQ(cache_->getNumCachedChunks(), 2U);
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  ASSERT_EQ(cache_->getNumCachedChunks(), 2U);
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  ASSERT_EQ(recover_cache_, isTableDatawrapperRestored(table_name_));
}

TEST_P(DataWrapperAppendRefreshTest, MissingRows) {
  file_name_ = "single_file_delete_rows"s + wrapper_ext(wrapper_type_);
  sqlCreateTestTable();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});

  overwriteSourceDir({{"i", "BIGINT"}});

  // Refresh command
  if (isOdbc(wrapper_type_)) {
    auto schema_name_table_name = getOdbcTableName(table_name_, wrapper_type_);
    queryAndAssertException("REFRESH FOREIGN TABLES " + table_name_ + ";",
                            "Refresh of foreign table created with \"APPEND\" update "
                            "type failed as result set of select statement reduced in "
                            "size: \"select i from " +
                                schema_name_table_name + "\"");
  } else {
    queryAndAssertException(
        "REFRESH FOREIGN TABLES " + table_name_ + ";",
        "Refresh of foreign table created with \"APPEND\" update type failed as "
        "file reduced in size: \"" +
            test_temp_dir + file_name_ + "\"");
  }
}

TEST_P(DataWrapperAppendRefreshTest, MissingRowsEvict) {
  file_name_ = "single_file_delete_rows"s + wrapper_ext(wrapper_type_);
  sqlCreateTestTable();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  sql("REFRESH FOREIGN TABLES " + table_name_ + " WITH (evict=true); ");
  sqlAndCompareResult(select_query, {{i(1)}});
}

TEST_P(DataWrapperAppendRefreshTest, MissingFile) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_missing_file";
  sqlCreateTestTable();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  queryAndAssertException("REFRESH FOREIGN TABLES " + table_name_ + ";",
                          getRemovedFileErrorMessage("one_row_2"));
}

TEST_P(DataWrapperAppendRefreshTest, MissingFileAlterFromAppendToAll) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_missing_file";
  sqlCreateTestTable();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  sql("ALTER FOREIGN TABLE " + table_name_ + " SET (REFRESH_UPDATE_TYPE = 'ALL');");
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(1)}});
}

// This tests the use case where there are multiple files in a
// directory but an update is made to only one of the files.
TEST_P(DataWrapperAppendRefreshTest, MultifileAppendtoFile) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file_multi_bad_append";
  sqlCreateTestTable();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
}

TEST_P(DataWrapperAppendRefreshTest, SortedFiles) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This test case is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file";
  sqlCreateTestTable({{"FILE_SORT_ORDER_BY", "REGEX_NUMBER"},
                      {"FILE_SORT_REGEX", "[^-]+_[^-]+_(\\d+)_.*"}});
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});
}

TEST_P(DataWrapperAppendRefreshTest, FilteredFiles) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This test case is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file";
  sqlCreateTestTable({{"REGEX_PATH_FILTER", ".*two.*"}});
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(4)}, {i(5)}});
}

TEST_P(DataWrapperAppendRefreshTest, FileRollOffOnly) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file_roll_off";
  fragment_size_ = DEFAULT_FRAGMENT_ROWS;
  sqlCreateTestTableWithRollOffOption();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  renameSourceDir(wrapper_file_type(wrapper_type_) + "_dir_file_roll_off_only");
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(3)}, {i(4)}});
}

TEST_P(DataWrapperAppendRefreshTest, FileRollOffAppend) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file_roll_off";
  fragment_size_ = DEFAULT_FRAGMENT_ROWS;
  sqlCreateTestTableWithRollOffOption();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  renameSourceDir(wrapper_file_type(wrapper_type_) + "_dir_file_roll_off_append");
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(3)}, {i(4)}, {i(5)}});
}

TEST_P(DataWrapperAppendRefreshTest, AllFilesRolledOff) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file_roll_off";
  fragment_size_ = DEFAULT_FRAGMENT_ROWS;
  sqlCreateTestTableWithRollOffOption();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  renameToEmptyDir();
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {});
}

TEST_P(DataWrapperAppendRefreshTest, AllFilesRolledOffAndNewFiles) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file_roll_off";
  fragment_size_ = DEFAULT_FRAGMENT_ROWS;
  sqlCreateTestTableWithRollOffOption();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  renameSourceDir(wrapper_file_type(wrapper_type_) + "_dir_file_roll_off_new_files");
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(7)}, {i(8)}});
}

TEST_P(DataWrapperAppendRefreshTest, FileRollOffMultipleFragments) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file_roll_off";
  fragment_size_ = 2;
  sqlCreateTestTableWithRollOffOption();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  renameSourceDir(wrapper_file_type(wrapper_type_) + "_dir_file_roll_off_append");
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(3)}, {i(4)}, {i(5)}});
}

TEST_P(DataWrapperAppendRefreshTest, FileRollOffOptionSetAndRegularAppend) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file_multi";
  fragment_size_ = DEFAULT_FRAGMENT_ROWS;
  sqlCreateTestTableWithRollOffOption();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});
}

TEST_P(DataWrapperAppendRefreshTest, FileRollOffOptionSetAndOldestFileNotRemoved) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file_roll_off";
  sqlCreateTestTableWithRollOffOption();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  renameSourceDir(wrapper_file_type(wrapper_type_) + "_dir_middle_file_roll_off");
  queryAndAssertException("REFRESH FOREIGN TABLES " + table_name_ + ";",
                          getRemovedFileErrorMessage("2"));
}

TEST_P(DataWrapperAppendRefreshTest, AlterAddFileRollOff) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file_roll_off";
  sqlCreateTestTable();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  renameSourceDir(wrapper_file_type(wrapper_type_) + "_dir_file_roll_off_only");
  queryAndAssertException("REFRESH FOREIGN TABLES " + table_name_ + ";",
                          getRemovedFileErrorMessage("1"));
  sql("ALTER FOREIGN TABLE " + table_name_ + " SET (allow_file_roll_off = 'true');");
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(3)}, {i(4)}});
}

TEST_P(DataWrapperAppendRefreshTest, AlterRemoveFileRollOff) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file_roll_off";
  sqlCreateTestTableWithRollOffOption();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  renameSourceDir(wrapper_file_type(wrapper_type_) + "_dir_file_roll_off_only");
  sql("ALTER FOREIGN TABLE " + table_name_ + " SET (allow_file_roll_off = 'false');");
  queryAndAssertException("REFRESH FOREIGN TABLES " + table_name_ + ";",
                          getRemovedFileErrorMessage("1"));
}

TEST_P(DataWrapperAppendRefreshTest, AppendToLastFile) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_last_file_append";
  fragment_size_ = 2;
  sqlCreateTestTable();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}});
}

TEST_P(DataWrapperAppendRefreshTest, AppendToLastFileThatSpansMultipleFragments) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_last_file_append_multi_fragment";
  fragment_size_ = 2;
  sqlCreateTestTable();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}, {i(6)}});
}

TEST_P(DataWrapperAppendRefreshTest, AppendToLastFileAndNewFileAdded) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_last_file_append";
  fragment_size_ = 2;
  sqlCreateTestTable();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  renameSourceDir(wrapper_file_type(wrapper_type_) + "_dir_last_file_append_new_file");
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}});
}

TEST_P(DataWrapperAppendRefreshTest, AppendToLastFileAndRollOff) {
  if (isOdbc(wrapper_type_)) {
    GTEST_SKIP() << "This testcase is not relevant to ODBC";
  }
  file_name_ = wrapper_file_type(wrapper_type_) + "_dir_last_file_append";
  fragment_size_ = 2;
  sqlCreateTestTableWithRollOffOption();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}});
  overwriteSourceDir({{"i", "BIGINT"}});
  renameSourceDir(wrapper_file_type(wrapper_type_) + "_dir_last_file_append_roll_off");
  sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  sqlAndCompareResult(select_query, {{i(2)}, {i(3)}, {i(4)}});
}

class DistributedCacheConsistencyTest : public DataWrapperAppendRefreshTest {
 protected:
  void SetUp() override {
    if (!isDistributedMode()) {
      GTEST_SKIP() << "This test case only applies to distributed mode.";
    }
    DataWrapperAppendRefreshTest::SetUp();
    file_name_ = wrapper_file_type(wrapper_type_) + "_dir_file_roll_off";
    fragment_size_ = 2;
    switchToAdmin();
  }

  void TearDown() override {
    if (!isDistributedMode()) {
      GTEST_SKIP() << "This test case only applies to distributed mode.";
    }
    switchToAdmin();
    sql("DROP DATABASE IF EXISTS test_db;");
    DataWrapperAppendRefreshTest::TearDown();
  }

  void rollOffFileAndRefresh() {
    overwriteSourceDir({{"i", "BIGINT"}});
    renameSourceDir(wrapper_file_type(wrapper_type_) + "_dir_file_roll_off_only");
    sql("REFRESH FOREIGN TABLES " + table_name_ + ";");
  }

  void forceFirstNodeTableEviction() {
    const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
    Catalog_Namespace::SessionInfo session_info{
        nullptr, {}, ExecutorDeviceType::CPU, session_id};
    db_handler->leaf_aggregator_.forwardQueryToLeaf(
        session_info,
        "REFRESH FOREIGN TABLES " + table_name_ + " WITH (evict = 'true');",
        0);
  }

  void switchToTestDb() {
    if (test_db_session_.empty()) {
      login(
          shared::kRootUsername, shared::kDefaultRootPasswd, "test_db", test_db_session_);
    }
    ASSERT_FALSE(test_db_session_.empty());
    setSessionId(test_db_session_);
  }

  void switchToDefaultDb() { switchToAdmin(); }

  std::string test_db_session_{};
};

TEST_P(DistributedCacheConsistencyTest, FileRollOffWithEvictionOnOneLeafNode) {
  sqlCreateTestTableWithRollOffOption();
  sqlAndCompareResult(select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}});

  rollOffFileAndRefresh();
  sqlAndCompareResult(select_query, {{i(3)}, {i(4)}});

  forceFirstNodeTableEviction();
  sqlAndCompareResult(select_query, {{i(3)}, {i(4)}});
}

TEST_P(DistributedCacheConsistencyTest, FileRollOffWithEvictionOnOneLeafNodeCrossDb) {
  sql("CREATE DATABASE test_db;");
  switchToTestDb();
  sqlCreateTestTableWithRollOffOption();

  static const std::string cross_db_select_query =
      "SELECT * FROM test_db." + table_name_ + " ORDER BY i;";
  switchToDefaultDb();
  sqlAndCompareResult(cross_db_select_query, {{i(1)}, {i(2)}, {i(3)}, {i(4)}});

  switchToTestDb();
  rollOffFileAndRefresh();

  switchToDefaultDb();
  sqlAndCompareResult(cross_db_select_query, {{i(3)}, {i(4)}});

  switchToTestDb();
  forceFirstNodeTableEviction();

  switchToDefaultDb();
  queryAndAssertException(
      cross_db_select_query,
      "Table data inconsistently cached for table: refresh_tmp in catalog: test_db. "
      "Please refresh table with the cache eviction option set.");
}

INSTANTIATE_TEST_SUITE_P(FileDataWrappers,
                         DistributedCacheConsistencyTest,
                         ::testing::Combine(::testing::ValuesIn(file_wrappers),
                                            ::testing::Values(true, false)),
                         [](const auto& info) {
                           return DataWrapperAppendRefreshTest::testParamsToString(
                               info.param);
                         });

INSTANTIATE_TEST_SUITE_P(
    DataTypeFragmentSizeAndDataWrapperCsvTests,
    DataTypeFragmentSizeAndDataWrapperTest,
    ::testing::Combine(::testing::Values(1, 2, 32'000'000),
                       ::testing::Values("csv"),
                       ::testing::Values(".csv", "_csv_dir", ".zip")),
    DataTypeFragmentSizeAndDataWrapperTest::getTestName);
INSTANTIATE_TEST_SUITE_P(DataTypeFragmentSizeAndDataWrapperParquetTests,
                         DataTypeFragmentSizeAndDataWrapperTest,
                         ::testing::Combine(::testing::Values(1, 2, 32'000'000),
                                            ::testing::Values("parquet"),
                                            ::testing::Values(".parquet",
                                                              "_parquet_dir")),
                         DataTypeFragmentSizeAndDataWrapperTest::getTestName);
INSTANTIATE_TEST_SUITE_P(DataTypeFragmentSizeAndDataWrapperOdbcTests,
                         DataTypeFragmentSizeAndDataWrapperTest,
                         ::testing::Combine(::testing::Values(1, 2, 32'000'000),
                                            ::testing::Values("sqlite", "postgres"),
                                            ::testing::Values(".csv")),
                         DataTypeFragmentSizeAndDataWrapperTest::getTestName);
INSTANTIATE_TEST_SUITE_P(
    DataTypeFragmentSizeAndDataWrapperRegexParserTests,
    DataTypeFragmentSizeAndDataWrapperTest,
    ::testing::Combine(::testing::Values(1, 2, 32'000'000),
                       ::testing::Values("regex_parser"),
                       ::testing::Values(".csv", "_csv_dir", ".zip")),
    DataTypeFragmentSizeAndDataWrapperTest::getTestName);

TEST_P(DataTypeFragmentSizeAndDataWrapperTest, ScalarTypes) {
  SKIP_IF_DISTRIBUTED("Test relies on local metadata or cache access");
  // Data type changes to handle unimplemented types in ODBC
  // Note: This requires the following option to be added to the postgres entry of
  // .odbc.ini:
  //    BoolsAsChar=false
  sql(createForeignTableQuery(
      {{"b", "BOOLEAN"},
       {"t", wrapper_type_ == "postgres" ? "SMALLINT" : "TINYINT"},
       {"s", "SMALLINT"},
       {"i", "INTEGER"},
       {"bi", "BIGINT"},
       {"f", wrapper_type_ == "sqlite" ? "DOUBLE" : "FLOAT"},
       {"dc", wrapper_type_ == "sqlite" ? "DOUBLE" : "DECIMAL(10, 5)"},
       {"tm", "TIME"},
       {"tp", "TIMESTAMP"},
       {"d", "DATE"},
       {"txt", "TEXT"},
       {"txt_2", "TEXT ENCODING NONE"}},
      getDataFilesPath() + "scalar_types" + extension_,
      wrapper_type_,
      {{"FRAGMENT_SIZE", fragmentSizeStr()}}));

  // Initial select count(*) for metadata scan prior to loading
  {
    TQueryResult result;
    sql(result, "SELECT COUNT(*) FROM " + default_table_name + ";");
  }

  std::map<std::pair<int, int>, std::unique_ptr<ChunkMetadata>> test_chunk_metadata_map;

  // Compare expected metadata if we are loading all values into a single fragment
  if (fragment_size_ >= 4) {
    assertExpectedChunkMetadata(getExpectedScalarTypeMetadata(false), default_table_name);
  }
  queryAndAssertScalarTypesResult();

  if (fragment_size_ >= 4) {
    assertExpectedChunkMetadata(getExpectedScalarTypeMetadata(true), default_table_name);
  }
}

TEST_F(SelectQueryTest, CsvArrayQuotedText) {
  const auto& query = getCreateForeignTableQuery(
      "(index INT, quoted_text TEXT[])", "array_quoted_text", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");
  // clang-format off
  assertResultSetEqual({
    { i(1),array({"quoted text"}) },
    { i(2),array({"quoted text 2"}) },
    { i(3),array({"quoted text 3", "quoted text 4"}) }},
    result);
  // clang-format on
}

TEST_F(SelectQueryTest, CsvArrayEmptyText) {
  const auto& query = getCreateForeignTableQuery("(index INT, txt1 TEXT[], txt2 TEXT[])",
                                                 {{"header", "false"}},
                                                 "empty_text_arrays",
                                                 "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");
  // clang-format off
  assertResultSetEqual({
                          {(int64_t)1, array({}), array({"string 1", "string 2"})},
                          {(int64_t)2, array({"string 1", "string 2"}), array({})},
                      },
    result);
  // clang-format on
}

TEST_F(SelectQueryTest, CsvMultipleFilesWithExpectedAndMismatchedColumns) {
  auto dir_path = getDataFilesPath() + "different_cols";
  sql("CREATE FOREIGN TABLE " + default_table_name +
      " (t TEXT, i1 BIGINT) SERVER default_local_delimited WITH ("
      "file_path = '" +
      dir_path + "');");

  auto file_path = dir_path + "/3_col_a_1_2.csv";
  queryAndAssertException("SELECT * FROM " + default_table_name + ";",
                          "Mismatched number of logical columns: (expected 2 "
                          "columns, has 3): in file '" +
                              file_path + "'");

  // Second query would previously result in a crash
  queryAndAssertException("SELECT * FROM " + default_table_name + ";",
                          "Mismatched number of logical columns: (expected 2 "
                          "columns, has 3): in file '" +
                              file_path + "'");
}

TEST_F(SelectQueryTest, CsvTrimSpaces) {
  const auto& query =
      getCreateForeignTableQuery("(index INT, txt1 TEXT, txt2 TEXT[], b BOOLEAN)",
                                 {{"header", "false"}, {"trim_spaces", "true"}},
                                 "with_spaces",
                                 "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");
  // clang-format off
  assertResultSetEqual({
                          {i(1), "text1", array({"t1", "t2", "t3"}), False},
                          {i(2), "text2", array({"t10", "t20", "t30"}), True},
                      },
    result);
  // clang-format on
}

TEST_F(SelectQueryTest, CsvNoTrimSpaces) {
  const auto& query =
      getCreateForeignTableQuery("(index INT, txt1 TEXT, txt2 TEXT[], b BOOLEAN)",
                                 {{"header", "false"}, {"trim_spaces", "false"}},
                                 "with_spaces",
                                 "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");
  // clang-format off
  assertResultSetEqual({
                          {i(1), "  text1 ", array({"t1","  t2"," t3 "}), False},
                          {i(2), " text2 ", array({" t10"," t20 ","  t30 "}), True},
                      },
    result);
  // clang-format on
}

TEST_F(SelectQueryTest, ParquetRequiredColumnScalars) {
  const auto& query = getCreateForeignTableQuery(
      "(b BOOLEAN, t TINYINT, s SMALLINT, i INTEGER, bi BIGINT, f FLOAT, "
      "dc DECIMAL(10,5), tm TIME, tp TIMESTAMP, d DATE, txt TEXT, "
      "txt_2 TEXT ENCODING NONE)",
      "scalar_types_max_def_level_zero",
      "parquet");
  sql(query);

  // clang-format off
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {True, 100L, 30000L, 2000000000L, 9000000000000000000L, 10.1f, 100.1234,
        "00:00:10", "1/1/2000 00:00:59", "1/1/2000", "text_1", "quoted text"},
      {False, 110L, 30500L, 2000500000L, 9000000050000000000L, 100.12f, 2.1234,
        "00:10:00", "6/15/2020 00:59:59", "6/15/2020", "text_2", "quoted text 2"},
      {True, 120L, 31000L, 2100000000L, 9100000000000000000L, 1000.123f, 100.1,
        "10:00:00", "12/31/2500 23:59:59", "12/31/2500", "text_3", "quoted text 3"},
  };
  // clang-format on

  sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY s;",
                      expected_values);
}

TEST_F(SelectQueryTest, ParquetArrayInt8EmptyWithFixedLengthArray) {
  const auto& query = getCreateForeignTableQuery(
      "(tinyint_arr_empty TINYINT[1])", "int8_empty_array", "parquet");
  sql(query);

  queryAndAssertException(
      default_select,

      "Detected an empty array being loaded into HeavyDB column "
      "'tinyint_arr_empty' which has a fixed length array type, expecting 1 elements. "
      "Row group: 0, Parquet column: 'tinyint_arr_empty.list.item', Parquet file: '" +
          getDataFilesPath() + "int8_empty_array.parquet'");
}

TEST_F(SelectQueryTest, ParquetArrayDateTimeTypes) {
  const auto& query = getCreateForeignTableQuery(
      "(index INT, time_milli_array TIME[], time_micro_array TIME[],"
      " time_nano_array TIME[], timestamp_milli1_array TIMESTAMP[],"
      " timestamp_micro1_array TIMESTAMP[], timestamp_milli2_array TIMESTAMP(3)[],"
      " timestamp_micro2_array TIMESTAMP(6)[], date_array DATE[])",
      "array_datetime_types",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");

  // clang-format off
  assertResultSetEqual(
      {
          {
            i(1),Null, Null, Null, Null, Null, Null,
            Null, Null
          },
          {
            i(2),array({"23:59:59", "00:59:59", "12:00:00"}),
            array({"23:59:59", "00:59:59", "12:00:00"}),
            array({"23:59:59", "00:59:59", "12:00:00"}),
            array({"1871-07-06 23:59:59", "1931-03-01 00:59:59", "1900-12-29 12:00:00"}),
            array({"1871-07-06 23:59:59", "1931-03-01 00:59:59", "1900-12-29 12:00:00"}),
            array({"1871-07-06 23:59:59.123", "1931-03-01 00:59:59.123",
                   "1900-12-29 12:00:00.123"}), array({"1871-07-06 23:59:59.123456",
                   "1931-03-01 00:59:59.123456", "1900-12-29 12:00:00.123456"}),
            array({"1871-07-06", "1931-03-01", "1900-12-29"})
          },
          {
            i(3),array({"10:10:10", i(NULL_BIGINT)}), array({"10:10:10", i(NULL_BIGINT)}),
            array({"10:10:10", i(NULL_BIGINT)}),
            array({"2020-11-10 10:10:10", i(NULL_BIGINT)}),
            array({"2020-11-10 10:10:10", i(NULL_BIGINT)}),
            array({"2020-11-10 10:10:10.123", i(NULL_BIGINT)}),
            array({"2020-11-10 10:10:10.123456", i(NULL_BIGINT)}),
            array({"2020-11-10", i(NULL_BIGINT)})
          },
          {
            i(4),Null, Null, Null,
            Null, Null, Null, Null, Null
          },
          {
            i(5),array({"00:00:01"}),
            array({"00:00:01"}), array({"00:00:01"}), array({"2200-01-01 00:00:01"}),
            array({"2200-01-01 00:00:01"}), array({"2200-01-01 00:00:01.123"}),
            array({"2200-01-01 00:00:01.123456"}), array({"2200-01-01"})
          },
      }, result);
  // clang-format on
}

TEST_F(SelectQueryTest, ParquetFixedLengthArrayMalformed) {
  const auto& query = getCreateForeignTableQuery(
      "(bigint_array BIGINT[3])", "array_fixed_len_malformed", "parquet");
  sql(query);
  TQueryResult result;
  queryAndAssertException(
      default_select,
      "Detected a row with 2 elements being loaded into HeavyDB column "
      "'bigint_array' which has a fixed length array type, expecting 3 elements. Row "
      "group: 2, Parquet column: 'i64.list.item', Parquet file: '" +
          getDataFilesPath() + "array_fixed_len_malformed.parquet'");
}

TEST_F(SelectQueryTest, ParquetFixedLengthArrayDateTimeTypes) {
  const auto& query = getCreateForeignTableQuery(
      "(index INT, time_milli_array TIME[2], time_micro_array TIME[2],"
      " time_nano_array TIME[2], timestamp_milli1_array TIMESTAMP[2],"
      " timestamp_micro1_array TIMESTAMP[2], timestamp_milli2_array TIMESTAMP(3)[2],"
      " timestamp_micro2_array TIMESTAMP(6)[2], date_array DATE[2])",
      "array_fixed_len_datetime_types",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");

  // clang-format off
  assertResultSetEqual(
      {
          {
            i(1),Null, Null, Null, Null, Null, Null,
            Null, Null
          },
          {
            i(2),array({"23:59:59", "00:59:59"}),
            array({"23:59:59", "00:59:59"}),
            array({"23:59:59", "00:59:59"}),
            array({"1871-07-06 23:59:59", "1931-03-01 00:59:59"}),
            array({"1871-07-06 23:59:59", "1931-03-01 00:59:59"}),
            array({"1871-07-06 23:59:59.123", "1931-03-01 00:59:59.123"}),
            array({"1871-07-06 23:59:59.123456", "1931-03-01 00:59:59.123456"}),
            array({"1871-07-06", "1931-03-01"})
          },
          {
            i(3),array({"10:10:10", i(NULL_BIGINT)}), array({"10:10:10", i(NULL_BIGINT)}),
            array({"10:10:10", i(NULL_BIGINT)}),
            array({"2020-11-10 10:10:10", i(NULL_BIGINT)}),
            array({"2020-11-10 10:10:10", i(NULL_BIGINT)}),
            array({"2020-11-10 10:10:10.123", i(NULL_BIGINT)}),
            array({"2020-11-10 10:10:10.123456", i(NULL_BIGINT)}),
            array({"2020-11-10", i(NULL_BIGINT)})
          },
          {
            i(4),Null, Null, Null,
            Null, Null, Null, Null, Null
          },
          {
            i(5),array({"00:00:01", "12:00:00"}),
            array({"00:00:01", "12:00:00"}), array({"00:00:01", "12:00:00"}),
            array({"2200-01-01 00:00:01", "1900-12-29 12:00:00"}),
            array({"2200-01-01 00:00:01", "1900-12-29 12:00:00"}),
            array({"2200-01-01 00:00:01.123", "1900-12-29 12:00:00.123"}),
            array({"2200-01-01 00:00:01.123456", "1900-12-29 12:00:00.123456"}),
            array({"2200-01-01", "1900-12-29"})
          },
      }, result);
  // clang-format on
}

TEST_F(SelectQueryTest, ParquetNullCompressedGeoTypes) {
  const auto& query = getCreateForeignTableQuery(
      "( index INT, p GEOMETRY(POINT,4326) ENCODING COMPRESSED(32), l "
      "GEOMETRY(LINESTRING,4326) ENCODING COMPRESSED(32), poly GEOMETRY(POLYGON,4326) "
      "ENCODING COMPRESSED(32), mpoly GEOMETRY(MULTIPOLYGON,4326) ENCODING "
      "COMPRESSED(32) )",
      "geo_types_valid",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");

  // clang-format off
  assertResultSetEqual(
  {
      {
        i(1),
        "POINT (0 0)",
        "LINESTRING (0 0,0.999999940861017 0.999999982770532)",
        "POLYGON ((0 0,0.999999940861017 0.0,0.999999940861017 0.999999982770532,0.0 0.999999982770532,0 0))",
        "MULTIPOLYGON (((0 0,0.999999940861017 0.0,0.0 0.999999982770532,0 0)),((1.99999996554106 1.99999996554106,2.99999999022111 1.99999996554106,1.99999996554106 2.99999999022111,1.99999996554106 1.99999996554106)))"},
      {
        i(2), Null, Null, Null, Null
      },
      {
        i(3),
        "POINT (0.999999940861017 0.999999982770532)",
        "LINESTRING (0.999999940861017 0.999999982770532,1.99999996554106 "
        "1.99999996554106,2.99999999022111 2.99999999022111)",
        "POLYGON ((4.99999995576218 3.99999997299165,6.99999992130324 "
        "3.99999997299165,5.99999998044223 4.99999999767169,4.99999995576218 "
        "3.99999997299165))",
        "MULTIPOLYGON (((0 0,0.999999940861017 0.0,0.0 0.999999982770532,0 0)),((1.99999996554106 1.99999996554106,2.99999999022111 1.99999996554106,1.99999996554106 2.99999999022111,1.99999996554106 1.99999996554106),(2.09999992609955 2.09999996800907,2.09999992609955 2.89999998775311,2.89999994584359 2.09999996800907,2.09999992609955 2.09999996800907)))",
      },
      {
        i(4),
        "POINT (1.99999996554106 1.99999996554106)",
        "LINESTRING (1.99999996554106 1.99999996554106,2.99999999022111 "
        "2.99999999022111)",
        "POLYGON ((0.999999940861017 0.999999982770532,2.99999999022111 "
        "0.999999982770532,1.99999996554106 2.99999999022111,0.999999940861017 "
        "0.999999982770532))",
        "MULTIPOLYGON (((4.99999995576218 4.99999999767169,7.99999994598329 7.99999998789281,4.99999995576218 7.99999998789281,4.99999995576218 4.99999999767169)),((0 0,2.99999999022111 0.0,0.0 2.99999999022111,0 0)),((10.9999999362044 10.9999999781139,9.99999999534339 11.9999999608845,9.99999999534339 9.99999999534339,10.9999999362044 10.9999999781139)))",
      },
      {
        i(5), Null, Null, Null, Null
      },
  },
  result);
  // clang-format on
}

// TODO(Misiu): The has_nulls metadata stas are not consistent between data wrappers for
// geo-types.  This test should be re-written to make sure we are consistent (or at least
// igore that parts that don't matter).
TEST_F(SelectQueryTest, ParquetGeoTypesMetadata) {
  SKIP_IF_DISTRIBUTED("Test relies on local metadata or cache access");

  const auto& query = getCreateForeignTableQuery(
      "( index INT, p POINT, l LINESTRING, poly POLYGON, mpoly MULTIPOLYGON )",
      "geo_types_valid",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");

  std::map<std::pair<int, int>, std::unique_ptr<ChunkMetadata>> test_chunk_metadata_map;
  test_chunk_metadata_map[{0, 1}] = createChunkMetadata<int32_t>(1, 20, 5, 1, 5, false);
  test_chunk_metadata_map[{0, 2}] = createChunkMetadata(2, 0, 5, true);
  test_chunk_metadata_map[{0, 3}] = createChunkMetadata<int8_t>(3, 80, 5, -16, 64, false);
  test_chunk_metadata_map[{0, 4}] = createChunkMetadata(4, 0, 5, true);
  test_chunk_metadata_map[{0, 5}] = createChunkMetadata<int8_t>(5, 112, 5, -16, 64, true);
  test_chunk_metadata_map[{0, 6}] =
      createChunkMetadata<double>(6, 160, 5, 0.000000, 3.000000, true);
  test_chunk_metadata_map[{0, 7}] = createChunkMetadata(7, 0, 5, true);
  test_chunk_metadata_map[{0, 8}] = createChunkMetadata<int8_t>(8, 160, 5, -16, 64, true);
  test_chunk_metadata_map[{0, 9}] = createChunkMetadata<int32_t>(9, 12, 5, 3, 4, true);
  test_chunk_metadata_map[{0, 10}] =
      createChunkMetadata<double>(10, 160, 5, 0.000000, 7.000000, true);
  test_chunk_metadata_map[{0, 11}] = createChunkMetadata(11, 0, 5, true);
  test_chunk_metadata_map[{0, 12}] =
      createChunkMetadata<int8_t>(12, 384, 5, -52, 64, true);
  test_chunk_metadata_map[{0, 13}] = createChunkMetadata<int32_t>(13, 32, 5, 3, 3, true);
  test_chunk_metadata_map[{0, 14}] = createChunkMetadata<int32_t>(14, 28, 5, 1, 2, true);
  test_chunk_metadata_map[{0, 15}] =
      createChunkMetadata<double>(15, 160, 5, 0.000000, 12.000000, true);
  assertExpectedChunkMetadata(test_chunk_metadata_map);
}

TEST_F(SelectQueryTest, ParquetMalformedGeoPoint) {
  const auto& query =
      getCreateForeignTableQuery("( p POINT )", "geo_point_malformed", "parquet");
  sql(query);

  TQueryResult result;
  queryAndAssertException(default_select,
                          "Failed to extract valid geometry in HeavyDB column 'p'. Row "
                          "group: 0, Parquet column: 'point', Parquet file: '" +
                              getDataFilesPath() + "geo_point_malformed.parquet'");
}

TEST_F(SelectQueryTest, ParquetWrongGeoType) {
  const auto& query =
      getCreateForeignTableQuery("( p LINESTRING )", "geo_point", "parquet");
  sql(query);

  TQueryResult result;
  queryAndAssertException(
      default_select,
      "Imported geometry doesn't match the geospatial type of HeavyDB column "
      "'p'. Row group: 0, Parquet column: 'point', Parquet file: '" +
          getDataFilesPath() + "geo_point.parquet'");
}

TEST_F(SelectQueryTest, ParquetArrayUnsignedIntegerTypes) {
  const auto& query = getCreateForeignTableQuery(
      "( index INT, utinyint_array SMALLINT[], usmallint_array INT[],"
      " uint_array BIGINT[] )",
      "array_unsigned_types",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");

  // clang-format off
  assertResultSetEqual( {
        {
          i(1),array({i(1),i(2)}),array({i(1),i(2)}),array({i(1),i(2)})
        },
        {
          i(2),array({i(3),i(4),i(5)}),array({i(3),i(4),i(5)}),array({i(3),i(4),i(5)})
        },
        {
          i(3),array({i(6),i(NULL_SMALLINT)}),
          array({i(6),i(NULL_INT)}),array({i(6),i(NULL_BIGINT)})
        },
        {
          i(4),Null,Null,Null,Null
        },
        {
          i(5),array({i(7)}),array({i(7)}),array({i(7)})
        }
      }, result);
  // clang-format on
}

TEST_F(SelectQueryTest, ParquetFixedLengthArrayUnsignedIntegerTypes) {
  const auto& query = getCreateForeignTableQuery(
      "( index INT, utinyint_array SMALLINT[2], usmallint_array INT[2],"
      " uint_array BIGINT[2] )",
      "array_fixed_len_unsigned_types",
      "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");

  // clang-format off
  assertResultSetEqual( {
        {
          i(1),array({i(1),i(2)}),array({i(1),i(2)}),array({i(1),i(2)})
        },
        {
          i(2),array({i(3),i(4)}),array({i(3),i(4)}),array({i(3),i(4)})
        },
        {
          i(3),array({i(6),i(NULL_SMALLINT)}),
          array({i(6),i(NULL_INT)}),array({i(6),i(NULL_BIGINT)})
        },
        {
          i(4),Null,Null,Null,Null
        },
        {
          i(5),array({i(7),i(8)}),array({i(7),i(8)}),array({i(7),i(8)})
        }
      }, result);
  // clang-format on
}

TEST_P(DataTypeFragmentSizeAndDataWrapperTest, ArrayTypes) {
  auto [fragment_size, data_wrapper_type, extension] = GetParam();
  if (isOdbc(data_wrapper_type)) {
    GTEST_SKIP()
        << "Sqlite does notsupport array types; Postgres arrays currently unsupported";
  }
  foreign_storage::OptionsMap options{{"FRAGMENT_SIZE", std::to_string(fragment_size)}};
  if (data_wrapper_type == "regex_parser") {
    options["LINE_REGEX"] = "(\\d+),\\s*" + get_line_array_regex(11);
  }
  sql(createForeignTableQuery({{"index", "INT"},
                               {"b", "BOOLEAN[]"},
                               {"t", "TINYINT[]"},
                               {"s", "SMALLINT[]"},
                               {"i", "INTEGER[]"},
                               {"bi", "BIGINT[]"},
                               {"f", "FLOAT[]"},
                               {"tm", "TIME[]"},
                               {"tp", "TIMESTAMP[]"},
                               {"d", "DATE[]"},
                               {"txt", "TEXT[]"},
                               {"fixedpoint", "DECIMAL(10,5)[]"}},
                              getDataFilesPath() + "array_types" + extension,
                              data_wrapper_type,
                              options));

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");
  // clang-format off
  assertResultSetEqual({
    {
      i(1), array({True}), array({i(50), i(100)}), array({i(30000), i(20000)}), array({i(2000000000)}),
      array({i(9000000000000000000)}), array({10.1f, 11.1f}), array({"00:00:10"}),
      array({"1/1/2000 00:00:59", "1/1/2010 00:00:59"}), array({"1/1/2000", "2/2/2000"}),
      array({"text_1"}),array({1.23,2.34})
    },
    {
      i(2), array({False, True}), array({i(110)}), array({i(30500)}), array({i(2000500000)}),
      array({i(9000000050000000000)}), array({100.12f}), array({"00:10:00", "00:20:00"}),
      array({"6/15/2020 00:59:59"}), array({"6/15/2020"}),
      array({"text_2", "text_3"}),array({3.456,4.5,5.6})
    },
    {
      i(3), array({True}), array({i(120)}), array({i(31000)}), array({i(2100000000), i(200000000)}),
      array({i(9100000000000000000), i(9200000000000000000)}), array({1000.123f}), array({"10:00:00"}),
      array({"12/31/2500 23:59:59"}), array({"12/31/2500"}),
      array({"text_4"}),array({6.78})
    }},
    result);
  // clang-format on
}

TEST_P(DataTypeFragmentSizeAndDataWrapperTest, FixedLengthArrayTypes) {
  auto [fragment_size, data_wrapper_type, extension] = GetParam();
  if (isOdbc(data_wrapper_type)) {
    GTEST_SKIP()
        << "Sqlite does notsupport array types; Postgres arrays currently unsupported";
  }
  foreign_storage::OptionsMap options{{"FRAGMENT_SIZE", std::to_string(fragment_size)}};
  if (data_wrapper_type == "regex_parser") {
    options["LINE_REGEX"] = "(\\d+),\\s*" + get_line_array_regex(11);
  }
  sql(createForeignTableQuery({{"index", "INT"},
                               {"b", "BOOLEAN[2]"},
                               {"t", "TINYINT[2]"},
                               {"s", "SMALLINT[2]"},
                               {"i", "INTEGER[2]"},
                               {"bi", "BIGINT[2]"},
                               {"f", "FLOAT[2]"},
                               {"tm", "TIME[2]"},
                               {"tp", "TIMESTAMP[2]"},
                               {"d", "DATE[2]"},
                               {"txt", "TEXT[2]"},
                               {"fixedpoint", "DECIMAL(10,5)[2]"}},
                              getDataFilesPath() + "array_fixed_len_types" + extension,
                              data_wrapper_type,
                              options));

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY index;");

  // clang-format off
  assertResultSetEqual({
    {
      i(1), array({True,False}), array({i(50), i(100)}), array({i(30000), i(20000)}), array({i(2000000000),i(-100000)}),
      array({i(9000000000000000000),i(-9000000000000000000)}), array({10.1f, 11.1f}), array({"00:00:10","01:00:10"}),
      array({"1/1/2000 00:00:59", "1/1/2010 00:00:59"}), array({"1/1/2000", "2/2/2000"}),
      array({"text_1","text_2"}),array({1.23,2.34})
    },
    {
      i(2), array({False, True}), array({i(110),i(101)}), array({i(30500),i(10001)}), array({i(2000500000),i(-23233)}),
      array({i(9000000050000000000),i(-9200000000000000000)}), array({100.12f,2.22f}), array({"00:10:00", "00:20:00"}),
      array({"6/15/2020 00:59:59","8/22/2020 00:00:59"}), array({"6/15/2020","8/22/2020"}),
      array({"text_3", "text_4"}),array({3.456,4.5})
    },
    {
      i(3), array({True,True}), array({i(120),i(44)}), array({i(31000),i(8123)}), array({i(2100000000), i(200000000)}),
      array({i(9100000000000000000), i(9200000000000000000)}), array({1000.123f,1392.22f}), array({"10:00:00","20:00:00"}),
      array({"12/31/2500 23:59:59","1/1/2500 23:59:59"}), array({"12/31/2500","1/1/2500"}),
      array({"text_5","text_6"}),array({6.78,5.6})
    }},
    result);
}

TEST_P(DataTypeFragmentSizeAndDataWrapperTest, GeoTypes) {
  createForeignTableForGeoTypes(wrapper_type_, extension_, fragment_size_);
  queryAndAssertGeoTypesResult();
}

TEST_P(DataTypeFragmentSizeAndDataWrapperTest, GeoTypesEvictPhysicalColumn) {
  SKIP_SETUP_IF_DISTRIBUTED("Test relies on disk cache");
  
  createForeignTableForGeoTypes(wrapper_type_, extension_, fragment_size_);
  queryAndAssertGeoTypesResult();

  const auto& catalog = getCatalog();
  auto& data_mgr = catalog.getDataMgr();
  auto storage_mgr = data_mgr.getPersistentStorageMgr();
  CHECK(storage_mgr);
  auto disk_cache = storage_mgr->getDiskCache();
  CHECK(disk_cache);

  auto db_id = catalog.getDatabaseId();
  auto td = catalog.getMetadataForTable(default_table_name, false);
  CHECK(td);
  auto table_id = td->tableId;
  bool physical_column_evicted{false};
  for (auto cd : catalog.getAllColumnMetadataForTable(table_id, false, false, true)) {
    if (cd->isGeoPhyCol) {
      ChunkMetadataVector metadata_vector;
      disk_cache->getCachedMetadataVecForKeyPrefix(metadata_vector, {db_id, table_id, cd->columnId});
      for (const auto& [chunk_key, metadata] : metadata_vector) {
        disk_cache->eraseChunk(chunk_key);
        physical_column_evicted = true;
      }
      break;
    }
  }
  ASSERT_TRUE(physical_column_evicted);

  const ChunkKey table_key{db_id, table_id};
  data_mgr.deleteChunksWithPrefix(table_key, MemoryLevel::CPU_LEVEL);
  data_mgr.deleteChunksWithPrefix(table_key, MemoryLevel::GPU_LEVEL);

  queryAndAssertGeoTypesResult();
}

class PostGisSelectQueryTest : public SelectQueryTest {
  // These test is intended to run from a POSTGIS enabled database, assumes the database
  // in use has POSTGIS enabled.
 public:
  void SetUp() override {
    wrapper_type_ = "postgres";
    SKIP_SETUP_IF_ODBC_DISABLED();
    SelectQueryTest::SetUp();
  }
};

TEST_F(PostGisSelectQueryTest, GeometryFromPostGisAsText) {
  sql(createForeignTableQuery(
      {{"id", "INT"},
       {"p", "POINT"},
       {"l", "LINESTRING"},
       {"poly", "POLYGON"},
       {"multipoly", "MULTIPOLYGON"}},
      getDataFilesPath() + "geo_types_valid.csv",
      "postgres",
      {{"sql_select",
        "select id, ST_AsText(p) as p, ST_AsText(l) as l, ST_AsText(poly) as poly, "
        "ST_AsText(multipoly) as multipoly from "s +
            getOdbcTableName(default_table_name, wrapper_type_)}},
      default_table_name,
      {{"id", "INT"},
       {"p", "TEXT"},
       {"l", "TEXT"},
       {"poly", "TEXT"},
       {"multipoly", "TEXT"}},
      true));

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " ORDER BY id;");
  // clang-format off
  assertResultSetEqual({
    {
      i(1), "POINT (0 0)", "LINESTRING (0 0,1 1)", "POLYGON ((0 0,1 0,1 1,0 1,0 0))",
      "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((2 2,3 2,2 3,2 2)))"
    },
    {
      i(2), Null, Null, Null, Null
    },
    {
      i(3), "POINT (1 1)", "LINESTRING (1 1,2 2,3 3)", "POLYGON ((5 4,7 4,6 5,5 4))",
      "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((2 2,3 2,2 3,2 2),(2.1 2.1,2.1 2.9,2.9 2.1,2.1 2.1)))"
    },
    {
      i(4), "POINT (2 2)", "LINESTRING (2 2,3 3)", "POLYGON ((1 1,3 1,2 3,1 1))",
      "MULTIPOLYGON (((5 5,8 8,5 8,5 5)),((0 0,3 0,0 3,0 0)),((11 11,10 12,10 10,11 11)))"
    },
    {
      i(5), Null, Null, Null, Null
    }},
    result);
  // clang-format on
}

INSTANTIATE_TEST_SUITE_P(RowGroupAndFragmentSizeParameterizedTests,
                         RowGroupAndFragmentSizeSelectQueryTest,
                         ::testing::Values(std::make_pair(1, 1),
                                           std::make_pair(1, 2),
                                           std::make_pair(2, 2)),
                         [](const auto& info) {
                           std::stringstream ss;
                           ss << "Rowgroup_size_" << info.param.first << "_Fragment_size_"
                              << info.param.second;
                           return ss.str();
                         });

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, MetadataOnlyCount) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a BIGINT, b BIGINT, c BIGINT, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT COUNT(*) FROM " + default_table_name + ";");
  assertResultSetEqual({{i(6)}}, result);
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, MetadataOnlyFilter) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a BIGINT, b BIGINT, c BIGINT, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  {
    TQueryResult result;
    sql(result, "SELECT COUNT(*) FROM " + default_table_name + " WHERE a > 2;");
    assertResultSetEqual({{i(4)}}, result);
  }

  {
    TQueryResult result;
    sql(result, "SELECT COUNT(*) FROM " + default_table_name + " WHERE d < 0;");
    assertResultSetEqual({{i(2)}}, result);
  }
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, Join) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_1_row_group_size." << row_group_size;
  foreign_storage::OptionsMap options{{"fragment_size", std::to_string(fragment_size)}};
  foreign_storage::OptionsMap options2;
  if (isDistributedMode()) {
    options["partitions"] = "REPLICATED";
    options2["partitions"] = "REPLICATED";
  }
  auto query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER)", options, filename_stream.str(), "parquet");
  sql(query);
  query = getCreateForeignTableQuery(
      "(t TEXT, i BIGINT, d DOUBLE, idx INTEGER)", options2, "example_2_index", "csv", 2);
  sql(query);

  TQueryResult result;
  sql(result,
      "SELECT t1.t, t1.i, t2.i, t2.d FROM " + default_table_name +
          " AS t1 JOIN "
          "" +
          default_table_name + "_2 AS t2 ON t1.t = t2.t order by t2.idx;");
  assertResultSetEqual({{"a", i(1), i(1), 1.1},
                        {"aa", Null_i, i(1), 1.1},
                        {"aa", Null_i, i(2), 2.2},
                        {"aaa", i(1), i(1), 1.1},
                        {"aaa", i(1), i(2), 2.2},
                        {"aaa", i(1), i(3), 3.3}},
                       result);
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, Select) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a BIGINT, b BIGINT, c BIGINT, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "select * from " + default_table_name + " order by a;");
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
      getCreateForeignTableQuery("(a BIGINT, b BIGINT, c BIGINT, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM " + default_table_name + " WHERE d < 0 ;");
  assertResultSetEqual({{i(5), i(7), i(10), -1.}, {i(6), i(8), i(1), -100.}}, result);
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
  inline static const TableDescriptor* td;
  inline static const ColumnDescriptor *cd1, *cd2, *cd3;
  inline static ChunkKey query_chunk_key1, query_chunk_key2, query_chunk_key3,
      query_table_prefix;

  static void SetUpTestSuite() {
    DBHandlerTestFixture::SetUpTestSuite();
    cat = &getCatalog();
    cache = cat->getDataMgr().getPersistentStorageMgr()->getDiskCache();
    sqlDropForeignTable();
  }

  static void TearDownTestSuite() { DBHandlerTestFixture::TearDownTestSuite(); }

  static void createTestTable() {
    sqlCreateForeignTable(
        "(" + col_name1 + " TEXT, " + col_name2 + " INTEGER, " + col_name3 + " DOUBLE)",
        table_2_filename,
        "csv");
    td = cat->getMetadataForTable(default_table_name, false);
    cd1 = cat->getMetadataForColumn(td->tableId, col_name1);
    cd2 = cat->getMetadataForColumn(td->tableId, col_name2);
    cd3 = cat->getMetadataForColumn(td->tableId, col_name3);
    query_chunk_key1 = {cat->getCurrentDB().dbId, td->tableId, cd1->columnId, 0};
    query_chunk_key2 = {cat->getCurrentDB().dbId, td->tableId, cd2->columnId, 0};
    query_chunk_key3 = {cat->getCurrentDB().dbId, td->tableId, cd3->columnId, 0};
    query_table_prefix = {cat->getCurrentDB().dbId, td->tableId};
  }

  static void sqlSelect(const std::string& columns = "*",
                        const std::string& table_name = "" + default_table_name + "") {
    sql("SELECT " + columns + " FROM " + table_name + ";");
  }

  void SetUp() override {
    SKIP_SETUP_IF_DISTRIBUTED("Needs local cache");
    ForeignTableTest::SetUp();
    cache->clear();
    createTestTable();
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    sqlDropForeignTable();
    ForeignTableTest::TearDown();
  }

  static std::shared_ptr<ChunkMetadata> createMetadata(const SQLTypeInfo type,
                                                       const size_t num_bytes,
                                                       const size_t num_elements,
                                                       const std::string& s_min,
                                                       const std::string& s_max,
                                                       const bool has_nulls) {
    auto elem_type = type.is_array() ? type.get_elem_type() : type;
    Datum min, max;
    if (elem_type.is_string()) {
      CHECK(elem_type.get_compression() == kENCODING_DICT);
      min.intval = std::stoi(s_min);
      max.intval = std::stoi(s_max);
    } else {
      min = StringToDatum(s_min, elem_type);
      max = StringToDatum(s_max, elem_type);
    }
    return std::make_shared<ChunkMetadata>(
        type, num_bytes, num_elements, ChunkStats{min, max, has_nulls});
  }

  static void assertMetadataEqual(const std::shared_ptr<ChunkMetadata> left_metadata,
                                  const std::shared_ptr<ChunkMetadata> right_metadata) {
    ASSERT_EQ(*left_metadata, *right_metadata) << *left_metadata << "\n"
                                               << *right_metadata << "\n";
  }
};

TEST_F(ForeignStorageCacheQueryTest, CreateDoesNotPopulateMetadata) {
  sqlDropForeignTable();
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key1));
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key2));
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key3));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(query_chunk_key1));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(query_table_prefix));
  createTestTable();
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key1));
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key2));
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key3));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(query_chunk_key1));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(query_table_prefix));
}

TEST_F(ForeignStorageCacheQueryTest, CacheEvictAfterDrop) {
  sqlSelect();
  ASSERT_EQ(cache->getNumCachedChunks(), 3U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 3U);
  sqlDropForeignTable();
  ASSERT_EQ(cache->getNumCachedChunks(), 0U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 0U);
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

TEST_F(ForeignStorageCacheQueryTest, CacheWithLimitCachesWholeChunks) {
  sqlDropForeignTable();
  sqlCreateForeignTable(
      "(i INTEGER,  txt TEXT, txt_2 TEXT ENCODING NONE, txt_arr TEXT[])", "0_255", "csv");
  sqlAndCompareResult("SELECT SUM(i) FROM (SELECT i FROM " + default_table_name +
                          " ORDER BY i LIMIT 10);",
                      {{i(45)}});
  sqlAndCompareResult("SELECT SUM(i) FROM " + default_table_name + ";", {{i(32640)}});
}

TEST_F(ForeignStorageCacheQueryTest, ArrayTypes) {
  sqlDropForeignTable();
  const auto& query = getCreateForeignTableQuery(
      "(index int, b BOOLEAN[], t TINYINT[], s SMALLINT[], i INTEGER[], bi BIGINT[], f "
      "FLOAT[], "
      "tm "
      "TIME[], tp TIMESTAMP[], "
      "d DATE[], txt TEXT[], fixedpoint DECIMAL(10,5)[])",
      {},
      "array_types",
      "csv",
      0,
      default_table_name,
      "csv");
  sql(query);
  sql("SELECT COUNT(*) FROM " + default_table_name + ";");

  auto td = cat->getMetadataForTable(default_table_name, false);
  ChunkMetadataVector metadata_vec{};
  cache->getCachedMetadataVecForKeyPrefix(metadata_vec,
                                          {cat->getCurrentDB().dbId, td->tableId});

  // clang-format off
  assertMetadataEqual(metadata_vec[0].second,
                      createMetadata(kINT, 12, 3, "1", "3", false));
  // we add 8-byte padding when 1-byte type (boolean[] and tinyint[]) varlen array col
  // inserts the first row and its array value is one of the following cases:
  // 1) is null,
  // 2) has a single elem, (i.e., {true} and {1}) or
  // 3) an empty array (i.e., {})
  // and in this test dataset, only boolean[] col matches the second case
  // --> see file_1.csv where the second column has an array having a single elem: '{true}'
  // so we need to consider the additional 8-byte adding, so 'numBytes' should become 12 (=4+8) bytes instead of 4
  assertMetadataEqual(
      metadata_vec[1].second,
      createMetadata(SQLTypeInfo(kARRAY, 0, 0, false, kENCODING_NONE, 0, kBOOLEAN),
                     12, 3, "false", "true", false));
  assertMetadataEqual(
      metadata_vec[2].second,
      createMetadata(SQLTypeInfo(kARRAY, 0, 0, false, kENCODING_NONE, 0, kTINYINT),
                     4, 3, "50", "120", false));
  assertMetadataEqual(
      metadata_vec[3].second,
      createMetadata(SQLTypeInfo(kARRAY, 0, 0, false, kENCODING_NONE, 0, kSMALLINT),
                     8, 3, "20000", "31000", false));
  assertMetadataEqual(
      metadata_vec[4].second,
      createMetadata(SQLTypeInfo(kARRAY, 0, 0, false, kENCODING_NONE, 0, kINT),
                     16, 3, "200000000", "2100000000", false));
  assertMetadataEqual(
      metadata_vec[5].second,
      createMetadata(SQLTypeInfo(kARRAY, 0, 0, false, kENCODING_NONE, 0, kBIGINT),
                     32, 3, "9000000000000000000", "9200000000000000000", false));
  assertMetadataEqual(
      metadata_vec[6].second,
      createMetadata(SQLTypeInfo(kARRAY, 0, 0, false, kENCODING_NONE, 0, kFLOAT),
                     16, 3, "10.100000", "1000.122986", false));
  assertMetadataEqual(
      metadata_vec[7].second,
      createMetadata(SQLTypeInfo(kARRAY, 0, 0, false, kENCODING_NONE, 0, kTIME),
                     32, 3, "00:00:10", "10:00:00", false));
  assertMetadataEqual(
      metadata_vec[8].second,
      createMetadata(SQLTypeInfo(kARRAY, 0, 0, false, kENCODING_NONE, 0, kTIMESTAMP),
                     32, 3, "2000-01-01 00:00:59", "2500-12-31 23:59:59", false));
  assertMetadataEqual(
      metadata_vec[9].second,
      createMetadata(SQLTypeInfo(kARRAY, 0, 0, false, kENCODING_NONE, 0, kDATE),
                     32, 3, "2000-01-01", "2500-12-31", false));
  // Correct metadata for string dict is not yet generated 
  assertMetadataEqual(
      metadata_vec[11].second,
      createMetadata(SQLTypeInfo(kARRAY, 10, 5, false, kENCODING_NONE, 0, kDECIMAL),
                     48, 3, "1.23000", "6.78000", false));
  // clang-format on
}

class CacheDefaultTest : public DBHandlerTestFixture {};
TEST_F(CacheDefaultTest, Path) {
  auto cat = &getCatalog();
  auto cache = cat->getDataMgr().getPersistentStorageMgr()->getDiskCache();
  ASSERT_EQ(cache->getCacheDirectory(),
            to_string(BASE_PATH) + "/" + shared::kDefaultDiskCacheDirName);
}

TEST_F(RecoverCacheQueryTest, RecoverWithoutWrappers) {
  SKIP_IF_DISTRIBUTED("Test relies on local metadata or cache access");

  std::string query = "CREATE FOREIGN TABLE " + default_table_name +
                      " (t TEXT, i BIGINT[]) "s +
                      "SERVER default_local_delimited WITH (file_path = '" +
                      getDataFilesPath() + "/" + "example_1_dir_archives/');";
  sql(query);
  auto td = cat_->getMetadataForTable(default_table_name, false);
  ChunkKey key{cat_->getCurrentDB().dbId, td->tableId, 1, 0};
  ChunkKey table_key{cat_->getCurrentDB().dbId, td->tableId};

  sqlAndCompareResult("SELECT * FROM " + default_table_name + "  ORDER BY t;",
                      {{"a", array({i(1), i(1), i(1)})},
                       {"aa", array({NULL_BIGINT, i(2), i(2)})},
                       {"aaa", array({i(3), NULL_BIGINT, i(3)})}});

  // Reset cache and clear memory representations.
  resetStorageManagerAndClearTableMemory(table_key);
  ASSERT_FALSE(isTableDatawrapperRestored(default_table_name));

  sqlAndCompareResult("SELECT * FROM " + default_table_name + "  ORDER BY t;",
                      {{"a", array({i(1), i(1), i(1)})},
                       {"aa", array({NULL_BIGINT, i(2), i(2)})},
                       {"aaa", array({i(3), NULL_BIGINT, i(3)})}});

  ASSERT_EQ(cache_->getNumCachedChunks(), 3U);    // 2 data + 1 index chunk
  ASSERT_EQ(cache_->getNumCachedMetadata(), 2U);  // Only 2 metadata
  ASSERT_TRUE(isTableDatawrapperRestored(default_table_name));
}

class RecoverCacheTest : public RecoverCacheQueryTest,
                         public ::testing::WithParamInterface<WrapperType> {
 protected:
  std::string file_ext_;

  void SetUp() override {
    wrapper_type_ = GetParam();
    file_ext_ = wrapper_ext(wrapper_type_);
    RecoverCacheQueryTest::SetUp();
  }
};

TEST_P(RecoverCacheTest, RestoreCache) {
  SKIP_IF_DISTRIBUTED("Test relies on local metadata or cache acces");

  sql(createForeignTableQuery(
      {{"t", "TEXT"}}, getDataFilesPath() + "a" + file_ext_, wrapper_type_));

  ASSERT_EQ(cache_->getNumCachedChunks(), 0U);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 0U);

  auto td = cat_->getMetadataForTable(default_table_name, false);
  ChunkKey table_key{cat_->getCurrentDB().dbId, td->tableId};
  sqlAndCompareResult("SELECT COUNT(*) FROM " + default_table_name + ";", {{i(1)}});
  ASSERT_FALSE(isTableDatawrapperRestored(default_table_name));

  // Reset cache and clear memory representations (disk data persists).
  resetStorageManagerAndClearTableMemory(table_key);
  ASSERT_TRUE(isTableDatawrapperDataOnDisk(default_table_name));
  ASSERT_FALSE(isTableDatawrapperRestored(default_table_name));

  sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY t;", {{"a"}});

  ASSERT_EQ(cache_->getNumCachedChunks(), 1U);
  ASSERT_EQ(cache_->getNumCachedMetadata(), 1U);
  ASSERT_TRUE(isTableDatawrapperRestored(default_table_name));
}

TEST_P(RecoverCacheTest, RestoreCacheFromOldWrapperMetadata) {
  SKIP_IF_DISTRIBUTED("Test relies on local metadata or cache access");

  sql(createForeignTableQuery(
      {{"col1", "BIGINT"}}, getDataFilesPath() + "1" + file_ext_, wrapper_type_));
  sqlAndCompareResult("SELECT COUNT(*) FROM " + default_table_name + ";", {{i(1)}});

  // Reset cache and clear memory representations (disk data persists).
  resetStorageManagerAndClearTableMemory(getTestTableKey());
  ASSERT_TRUE(isTableDatawrapperDataOnDisk(default_table_name));
  ASSERT_FALSE(isTableDatawrapperRestored(default_table_name));
  setOldDataWrapperMetadata(default_table_name, "1");

  sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY col1;",
                      {{i(1)}});

  ASSERT_EQ(cache_->getNumCachedChunks(), size_t(1));
  ASSERT_EQ(cache_->getNumCachedMetadata(), size_t(1));
  ASSERT_TRUE(isTableDatawrapperRestored(default_table_name));
}

INSTANTIATE_TEST_SUITE_P(RecoverCacheParameterizedTests,
                         RecoverCacheTest,
                         ::testing::ValuesIn(local_wrappers),
                         [](const auto& param_info) { return param_info.param; });

class DataWrapperRecoverCacheQueryTest
    : public RecoverCacheQueryTest,
      public ::testing::WithParamInterface<WrapperType> {
 protected:
  std::string file_ext_;

  void SetUp() override {
    wrapper_type_ = GetParam();
    file_ext_ = wrapper_ext(wrapper_type_);
    RecoverCacheQueryTest::SetUp();
  }

  std::string getWrapperFileExt() const {
    return is_regex(wrapper_type_) ? "csv" : wrapper_type_;
  }
};

TEST_P(DataWrapperRecoverCacheQueryTest, RecoverThenPopulateDataWrappersOnDemand) {
  SKIP_IF_DISTRIBUTED("Test relies on local metadata or cache acces");
  bool cache_during_scan = (wrapper_type_ == "csv" || wrapper_type_ == "regex_parser");

  sql(createForeignTableQuery(
      {{"col1", "BIGINT"}}, getDataFilesPath() + "1" + file_ext_, wrapper_type_));

  auto td = cat_->getMetadataForTable(default_table_name, false);
  ChunkKey key{cat_->getCurrentDB().dbId, td->tableId, 1, 0};
  ChunkKey table_key{cat_->getCurrentDB().dbId, td->tableId};

  sqlAndCompareResult("SELECT COUNT(*) FROM " + default_table_name + ";", {{i(1)}});
  // Cache now has metadata only.
  ASSERT_EQ(cache_->getNumCachedMetadata(), 1U);
  ASSERT_EQ(cache_->getNumCachedChunks(), cache_during_scan ? 1U : 0U);
  ASSERT_TRUE(psm_->getForeignStorageMgr()->hasDataWrapperForChunk(key));

  // Reset cache and clear memory representations.
  resetStorageManagerAndClearTableMemory(table_key);

  ASSERT_TRUE(isTableDatawrapperDataOnDisk(default_table_name));
  if (wrapper_type_ != "csv") {
    ASSERT_TRUE(compareTableDatawrapperMetadataToFile(
        default_table_name, getWrapperMetadataPath("1", getWrapperFileExt())));
  }

  // This query should hit recovered disk data and not need to create datawrappers.
  sqlAndCompareResult("SELECT COUNT(*) FROM " + default_table_name + ";", {{i(1)}});

  ASSERT_EQ(cache_->getNumCachedMetadata(), 1U);
  ASSERT_EQ(cache_->getNumCachedChunks(), cache_during_scan ? 1U : 0U);
  ASSERT_TRUE(psm_->getForeignStorageMgr()->hasDataWrapperForChunk(key));

  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";", {{i(1)}});
  ASSERT_EQ(cache_->getNumCachedChunks(), 1U);
  sqlDropForeignTable();
}

// Check that datawrapper metadata is generated and restored correctly when appending
// data
TEST_P(DataWrapperRecoverCacheQueryTest, AppendData) {
  SKIP_IF_DISTRIBUTED("Test relies on local metadata or cache acces");

  int fragment_size = 2;
  // Create initial files and tables
  bf::remove_all(getDataFilesPath() + "append_tmp");

  recursive_copy(getDataFilesPath() + "append_before", getDataFilesPath() + "append_tmp");
  auto file_path = getDataFilesPath() + "append_tmp/single_file" + file_ext_;

  sql(createForeignTableQuery({{"i", "BIGINT"}},
                              file_path,
                              wrapper_type_,
                              {{"FRAGMENT_SIZE", std::to_string(fragment_size)},
                               {"REFRESH_UPDATE_TYPE", "APPEND"}},
                              default_table_name,
                              {},
                              0));

  auto td = cat_->getMetadataForTable(default_table_name, false);
  ChunkKey key{cat_->getCurrentDB().dbId, td->tableId, 1, 0};
  ChunkKey table_key{cat_->getCurrentDB().dbId, td->tableId};

  std::string select = "SELECT * FROM "s + default_table_name + " ORDER BY i;";
  // Read from table
  sqlAndCompareResult(select, {{i(1)}, {i(2)}});

  ASSERT_TRUE(isTableDatawrapperDataOnDisk(default_table_name));
  ASSERT_TRUE(compareTableDatawrapperMetadataToFile(
      default_table_name, getWrapperMetadataPath("append_before", getWrapperFileExt())));

  // Reset cache and clear memory representations.
  resetStorageManagerAndClearTableMemory(table_key);

  // Modify tables on disk
  bf::remove_all(getDataFilesPath() + "append_tmp");
  recursive_copy(getDataFilesPath() + "append_after", getDataFilesPath() + "append_tmp");
  // Recreate table for ODBC data wrappers
  if (isOdbc(wrapper_type_)) {
    createODBCSourceTable(
        default_table_name, {{"i", "BIGINT"}}, file_path, wrapper_type_);
  }

  // Refresh command
  sql("REFRESH FOREIGN TABLES " + default_table_name + ";");
  // Read new data
  sqlAndCompareResult(select, {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});

  // Metadata file should be updated
  ASSERT_TRUE(isTableDatawrapperDataOnDisk(default_table_name));
  ASSERT_TRUE(compareTableDatawrapperMetadataToFile(
      default_table_name, getWrapperMetadataPath("append_after", getWrapperFileExt())));

  bf::remove_all(getDataFilesPath() + "append_tmp");
  sqlDropForeignTable();
}

INSTANTIATE_TEST_SUITE_P(DataWrapperRecoverCacheQueryTest,
                         DataWrapperRecoverCacheQueryTest,
                         ::testing::ValuesIn(local_wrappers),
                         [](const auto& param_info) { return param_info.param; });

class MockDataWrapper : public foreign_storage::MockForeignDataWrapper {
 public:
  MockDataWrapper() : throw_on_metadata_scan_(false), throw_on_chunk_fetch_(false) {}

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override {
    if (throw_on_metadata_scan_) {
      throw std::runtime_error{"populateChunkMetadata mock exception"};
    } else {
      parent_data_wrapper_->populateChunkMetadata(chunk_metadata_vector);
    }
  }

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override {
    if (throw_on_chunk_fetch_) {
      throw std::runtime_error{"populateChunkBuffers mock exception"};
    } else {
      parent_data_wrapper_->populateChunkBuffers(required_buffers, optional_buffers);
    }
  }

  void setParentWrapper(
      std::shared_ptr<ForeignDataWrapper> parent_data_wrapper) override {
    parent_data_wrapper_ = parent_data_wrapper;
  }

  void unsetParentWrapper() override { parent_data_wrapper_ = nullptr; }

  void throwOnMetadataScan(bool throw_on_metadata_scan) {
    throw_on_metadata_scan_ = throw_on_metadata_scan;
  }

  void throwOnChunkFetch(bool throw_on_chunk_fetch) {
    throw_on_chunk_fetch_ = throw_on_chunk_fetch;
  }

  std::string getSerializedDataWrapper() const override {
    return parent_data_wrapper_->getSerializedDataWrapper();
  }

  void restoreDataWrapperInternals(const std::string& file_path,
                                   const ChunkMetadataVector& chunk_metadata) override {
    parent_data_wrapper_->restoreDataWrapperInternals(file_path, chunk_metadata);
  }

  bool isRestored() const override { return parent_data_wrapper_->isRestored(); }

  void validateServerOptions(const ForeignServer* foreign_server) const override {
    parent_data_wrapper_->validateServerOptions(foreign_server);
  }

  void validateTableOptions(const ForeignTable* foreign_table) const override {
    parent_data_wrapper_->validateTableOptions(foreign_table);
  }

  const std::set<std::string_view>& getSupportedTableOptions() const override {
    return supported_table_options_;
  }

  void validateUserMappingOptions(const UserMapping* user_mapping,
                                  const ForeignServer* foreign_server) const override {
    parent_data_wrapper_->validateUserMappingOptions(user_mapping, foreign_server);
  }

  void validateSchema(const std::list<ColumnDescriptor>& columns) const override {
    parent_data_wrapper_->validateSchema(columns);
  };

  ParallelismLevel getCachedParallelismLevel() const override {
    return parent_data_wrapper_->getCachedParallelismLevel();
  }

  ParallelismLevel getNonCachedParallelismLevel() const override {
    return parent_data_wrapper_->getNonCachedParallelismLevel();
  }

  const std::set<std::string_view>& getSupportedUserMappingOptions() const override {
    return supported_user_mapping_options_;
  }

 protected:
  std::shared_ptr<foreign_storage::ForeignDataWrapper> parent_data_wrapper_;
  std::atomic<bool> throw_on_metadata_scan_;
  std::atomic<bool> throw_on_chunk_fetch_;
  std::set<std::string_view> supported_table_options_;
  std::set<std::string_view> supported_user_mapping_options_;
};

class SchemaMismatchTest : public ForeignTableTest,
                           public TempDirManager,
                           public ::testing::WithParamInterface<WrapperType> {
 protected:
  FileExtType ext_;

  virtual void setTestFile(const std::string& file_name,
                           const std::string& ext,
                           const std::vector<ColumnPair>& table_schema) {
    bf::copy_file(getDataFilesPath() + file_name + ext,
                  test_temp_dir + TEMP_FILE + ext,
#if 107400 <= BOOST_VERSION
                  bf::copy_options::overwrite_existing
#else
                  bf::copy_option::overwrite_if_exists
#endif
    );
  }

  void SetUp() override {
    wrapper_type_ = GetParam();
    ext_ = wrapper_ext(wrapper_type_);
    ForeignTableTest::SetUp();
    sqlDropForeignTable();
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    sqlDropForeignTable();
    ForeignTableTest::TearDown();
  }

  inline static const std::string TEMP_FILE{default_file_name};
};

INSTANTIATE_TEST_SUITE_P(DataWrapperParameterization,
                         SchemaMismatchTest,
                         ::testing::ValuesIn(file_wrappers),
                         [](const auto& info) { return info.param; });

TEST_P(SchemaMismatchTest, FileHasTooManyColumns_Create) {
  foreign_storage::OptionsMap options;
  // Use a line regex that matches two columns
  if (is_regex(wrapper_type_)) {
    options["LINE_REGEX"] = get_line_regex(2);
  }
  sql(createForeignTableQuery({{"i", "BIGINT"}},
                              getDataFilesPath() + "two_col_1_2" + ext_,
                              wrapper_type_,
                              options));
  queryAndAssertException("SELECT COUNT(*) FROM " + default_table_name + ";",
                          "Mismatched number of logical columns: (expected 1 "
                          "columns, has 2): in file '" +
                              getDataFilesPath() + "two_col_1_2" + ext_ + "'");
}

TEST_P(SchemaMismatchTest, FileHasTooFewColumns_Create) {
  foreign_storage::OptionsMap options;
  // Use a line regex that matches only one column
  if (is_regex(wrapper_type_)) {
    options["LINE_REGEX"] = get_line_regex(1);
  }
  sql(createForeignTableQuery({{"i", "BIGINT"}, {"i2", "BIGINT"}},
                              getDataFilesPath() + "0" + ext_,
                              wrapper_type_,
                              options));
  queryAndAssertException("SELECT COUNT(*) FROM " + default_table_name + ";",
                          "Mismatched number of logical columns: (expected 2 "
                          "columns, has 1): in file '" +
                              getDataFilesPath() + "0" + ext_ + "'");
}

TEST_P(SchemaMismatchTest, FileHasTooFewColumns_Repeat) {
  foreign_storage::OptionsMap options{{"REFRESH_UPDATE_TYPE", "APPEND"}};
  // Use a line regex that matches only one column
  if (is_regex(wrapper_type_)) {
    options["LINE_REGEX"] = get_line_regex(1);
  }
  sql(createForeignTableQuery({{"i", "BIGINT"}, {"i2", "BIGINT"}},
                              getDataFilesPath() + "0" + ext_,
                              wrapper_type_,
                              options));
  queryAndAssertException("SELECT COUNT(*) FROM " + default_table_name + ";",
                          "Mismatched number of logical columns: (expected 2 "
                          "columns, has 1): in file '" +
                              getDataFilesPath() + "0" + ext_ + "'");
  queryAndAssertException("SELECT COUNT(*) FROM " + default_table_name + ";",
                          "Mismatched number of logical columns: (expected 2 "
                          "columns, has 1): in file '" +
                              getDataFilesPath() + "0" + ext_ + "'");
}

TEST_P(SchemaMismatchTest, FileHasTooManyColumns_Refresh) {
  foreign_storage::OptionsMap options;
  // Use a line regex that matches two columns
  if (is_regex(wrapper_type_)) {
    options["LINE_REGEX"] = get_line_regex(2);
  }
  setTestFile("0", ext_, {{"i", "BIGINT"}});
  sql(createForeignTableQuery(
      {{"i", "BIGINT"}}, test_temp_dir + TEMP_FILE + ext_, wrapper_type_, options));
  sql("SELECT COUNT(*) FROM " + default_table_name + ";");
  if (is_regex(wrapper_type_)) {
    // Mismatch between file content and regex should result in rows with all null values
    sqlAndCompareResult("SELECT * FROM " + default_table_name + ";", {{NULL_BIGINT}});
  }
  setTestFile("two_col_1_2", ext_, {{"i", "BIGINT"}});
  queryAndAssertException("REFRESH FOREIGN TABLES " + default_table_name + ";",
                          "Mismatched number of logical columns: (expected 1 "
                          "columns, has 2): in file '" +
                              test_temp_dir + TEMP_FILE + ext_ + "'");
}

TEST_P(SchemaMismatchTest, FileHasTooFewColumns_Refresh) {
  foreign_storage::OptionsMap options;
  // Use a line regex that matches only one column
  if (is_regex(wrapper_type_)) {
    options["LINE_REGEX"] = get_line_regex(1);
  }
  setTestFile("two_col_1_2", ext_, {{"i", "BIGINT"}, {"i2", "BIGINT"}});
  sql(createForeignTableQuery({{"i", "BIGINT"}, {"i2", "BIGINT"}},
                              test_temp_dir + TEMP_FILE + ext_,
                              wrapper_type_,
                              options));
  sql("SELECT COUNT(*) FROM " + default_table_name + ";");
  if (is_regex(wrapper_type_)) {
    // Mismatch between file content and regex should result in rows with all null values
    sqlAndCompareResult("SELECT * FROM " + default_table_name + ";",
                        {{NULL_BIGINT, NULL_BIGINT}});
  }
  setTestFile("0", ext_, {{"i", "BIGINT"}, {"i2", "BIGINT"}});
  queryAndAssertException("REFRESH FOREIGN TABLES " + default_table_name + ";",
                          "Mismatched number of logical columns: (expected 2 "
                          "columns, has 1): in file '" +
                              test_temp_dir + TEMP_FILE + ext_ + "'");
}

class AlterForeignTableRegularTableTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("DROP TABLE IF EXISTS test_table");
    sql("DROP TABLE IF EXISTS renamed_table");
  }
  void TearDown() override {
    sql("DROP TABLE IF EXISTS test_table");
    sql("DROP TABLE IF EXISTS renamed_table");
    DBHandlerTestFixture::TearDown();
  }
};

TEST_F(AlterForeignTableRegularTableTest, RenameRegularTable) {
  sql("CREATE TABLE test_table (i INTEGER);");
  queryAndAssertException("ALTER FOREIGN TABLE test_table RENAME to renamed_table;",
                          "test_table is a table. Use ALTER TABLE.");
}

class ParquetCoercionTest : public SelectQueryTest {
 protected:
  void createForeignTableWithCoercion(const std::string& coerced_type,
                                      const std::string& base_file_name) {
    const auto& query = getCreateForeignTableQuery(
        "( coerced " + coerced_type + " )", base_file_name, "parquet");
    sql(query);
  }

  std::string getCoercionException(const std::string& min_allowed_value,
                                   const std::string& max_allowed_value,
                                   const std::string& encountered_value,
                                   const std::string& base_file_name) {
    const std::string file_name = getDataFilesPath() + base_file_name + ".parquet";
    return "Parquet column contains values that are outside the range of the "
           "HeavyDB "
           "column type. Consider using a wider column type. Min allowed value: " +
           min_allowed_value +
           ". Max allowed "
           "value: " +
           max_allowed_value + ". Encountered value: " + encountered_value +
           ". Error validating statistics of Parquet column "
           "'value' in row group 0 of Parquet file "
           "'" +
           file_name + "'.";
  }
};

class ParquetCoercionTestOptionalAnnotation
    : public ParquetCoercionTest,
      public ::testing::WithParamInterface<AnnotationType> {};

INSTANTIATE_TEST_SUITE_P(OptionalAnnotationParameterizedTests,
                         ParquetCoercionTestOptionalAnnotation,
                         ::testing::Values("", "_no_annotation"));

TEST_P(ParquetCoercionTestOptionalAnnotation, Int64ToInt) {
  createForeignTableWithCoercion("INT",
                                 "ParquetCoercionTypes/coercible_int64" + GetParam());
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int64ToIntInformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_int64" + GetParam();
  createForeignTableWithCoercion("INT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException(
          "-2147483647", "2147483647", "9223372036854775807", base_file_name));
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int64ToSmallInt) {
  createForeignTableWithCoercion("SMALLINT",
                                 "ParquetCoercionTypes/coercible_int64" + GetParam());
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int64ToSmallIntInformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_int64" + GetParam();
  createForeignTableWithCoercion("SMALLINT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-32767", "32767", "9223372036854775807", base_file_name));
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int64ToTinyInt) {
  createForeignTableWithCoercion("TINYINT",
                                 "ParquetCoercionTypes/coercible_int64" + GetParam());
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int64ToTinyIntInformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_int64" + GetParam();
  createForeignTableWithCoercion("TINYINT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-127", "127", "9223372036854775807", base_file_name));
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int64ToBigIntFixedLengthEncoded32) {
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (32)",
                                 "ParquetCoercionTypes/coercible_int64" + GetParam());
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_P(ParquetCoercionTestOptionalAnnotation,
       Int64ToBigIntFixedLengthEncoded32InformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_int64" + GetParam();
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (32)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException(
          "-2147483647", "2147483647", "9223372036854775807", base_file_name));
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int64ToBigIntFixedLengthEncoded16) {
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (16)",
                                 "ParquetCoercionTypes/coercible_int64" + GetParam());
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_P(ParquetCoercionTestOptionalAnnotation,
       Int64ToBigIntFixedLengthEncoded16InformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_int64" + GetParam();
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (16)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-32767", "32767", "9223372036854775807", base_file_name));
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int64ToBigIntFixedLengthEncoded8) {
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (8)",
                                 "ParquetCoercionTypes/coercible_int64" + GetParam());
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_P(ParquetCoercionTestOptionalAnnotation,
       Int64ToBigIntFixedLengthEncoded8InformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_int64" + GetParam();
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (8)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-127", "127", "9223372036854775807", base_file_name));
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int32ToSmallInt) {
  createForeignTableWithCoercion("SMALLINT",
                                 "ParquetCoercionTypes/coercible_int32" + GetParam());
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int32ToSmallIntInformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_int32" + GetParam();
  createForeignTableWithCoercion("SMALLINT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-32767", "32767", "2147483647", base_file_name));
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int32ToTinyInt) {
  createForeignTableWithCoercion("TINYINT",
                                 "ParquetCoercionTypes/coercible_int32" + GetParam());
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int32ToTinyIntInformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_int32" + GetParam();
  createForeignTableWithCoercion("TINYINT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-127", "127", "2147483647", base_file_name));
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int32ToIntFixedLengthEncoded16) {
  createForeignTableWithCoercion("INT ENCODING FIXED (16)",
                                 "ParquetCoercionTypes/coercible_int32" + GetParam());
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_P(ParquetCoercionTestOptionalAnnotation,
       Int32ToIntFixedLengthEncoded16InformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_int32" + GetParam();
  createForeignTableWithCoercion("INT ENCODING FIXED (16)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-32767", "32767", "2147483647", base_file_name));
}

TEST_P(ParquetCoercionTestOptionalAnnotation, Int32ToIntFixedLengthEncoded8) {
  createForeignTableWithCoercion("INT ENCODING FIXED (8)",
                                 "ParquetCoercionTypes/coercible_int32" + GetParam());
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_P(ParquetCoercionTestOptionalAnnotation,
       Int32ToIntFixedLengthEncoded8InformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_int32" + GetParam();
  createForeignTableWithCoercion("INT ENCODING FIXED (8)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-127", "127", "2147483647", base_file_name));
}

TEST_F(ParquetCoercionTest, Int16ToTinyInt) {
  createForeignTableWithCoercion("TINYINT", "ParquetCoercionTypes/coercible_int16");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, Int16ToTinyIntInformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_int16";
  createForeignTableWithCoercion("TINYINT", base_file_name);
  queryAndAssertException("SELECT * FROM " + default_table_name + "",
                          getCoercionException("-127", "127", "32767", base_file_name));
}

TEST_F(ParquetCoercionTest, Int16ToSmallIntFixedLengthEncoded8) {
  createForeignTableWithCoercion("SMALLINT ENCODING FIXED (8)",
                                 "ParquetCoercionTypes/coercible_int16");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, Int16ToSmallIntFixedLengthEncoded8InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_int16";
  createForeignTableWithCoercion("SMALLINT ENCODING FIXED (8)", base_file_name);
  queryAndAssertException("SELECT * FROM " + default_table_name + "",
                          getCoercionException("-127", "127", "32767", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToBigInt) {
  createForeignTableWithCoercion("BIGINT", "ParquetCoercionTypes/coercible_uint64");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToBigIntInformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint64";
  createForeignTableWithCoercion("BIGINT", base_file_name);
  queryAndAssertException("SELECT * FROM " + default_table_name + "",
                          getCoercionException("-9223372036854775807",
                                               "9223372036854775807",
                                               "18446744073709551615",
                                               base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToInt) {
  createForeignTableWithCoercion("INT", "ParquetCoercionTypes/coercible_uint64");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToIntInformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint64";
  createForeignTableWithCoercion("INT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException(
          "-2147483647", "2147483647", "18446744073709551615", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToSmallInt) {
  createForeignTableWithCoercion("SMALLINT", "ParquetCoercionTypes/coercible_uint64");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToSmallIntInformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint64";
  createForeignTableWithCoercion("SMALLINT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-32767", "32767", "18446744073709551615", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToTinyInt) {
  createForeignTableWithCoercion("TINYINT", "ParquetCoercionTypes/coercible_uint64");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToTinyIntInformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint64";
  createForeignTableWithCoercion("TINYINT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-127", "127", "18446744073709551615", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToBigIntFixedLengthEncoded32) {
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (32)",
                                 "ParquetCoercionTypes/coercible_uint64");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToBigIntFixedLengthEncoded32InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint64";
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (32)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException(
          "-2147483647", "2147483647", "18446744073709551615", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToBigIntFixedLengthEncoded16) {
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (16)",
                                 "ParquetCoercionTypes/coercible_uint64");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToBigIntFixedLengthEncoded16InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint64";
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (16)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-32767", "32767", "18446744073709551615", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToBigIntFixedLengthEncoded8) {
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (8)",
                                 "ParquetCoercionTypes/coercible_uint64");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt64ToBigIntFixedLengthEncoded8InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint64";
  createForeignTableWithCoercion("BIGINT ENCODING FIXED (8)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-127", "127", "18446744073709551615", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt32ToInt) {
  createForeignTableWithCoercion("INT", "ParquetCoercionTypes/coercible_uint32");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt32ToIntInformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint32";
  createForeignTableWithCoercion("INT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-2147483647", "2147483647", "4294967295", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt32ToSmallInt) {
  createForeignTableWithCoercion("SMALLINT", "ParquetCoercionTypes/coercible_uint32");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt32ToSmallIntInformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint32";
  createForeignTableWithCoercion("SMALLINT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-32767", "32767", "4294967295", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt32ToTinyInt) {
  createForeignTableWithCoercion("TINYINT", "ParquetCoercionTypes/coercible_uint32");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt32ToTinyIntInformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint32";
  createForeignTableWithCoercion("TINYINT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-127", "127", "4294967295", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt32ToIntFixedLengthEncoded16) {
  createForeignTableWithCoercion("INT ENCODING FIXED (16)",
                                 "ParquetCoercionTypes/coercible_uint32");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt32ToIntFixedLengthEncoded16InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint32";
  createForeignTableWithCoercion("INT ENCODING FIXED (16)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-32767", "32767", "4294967295", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt32ToIntFixedLengthEncoded8) {
  createForeignTableWithCoercion("INT ENCODING FIXED (8)",
                                 "ParquetCoercionTypes/coercible_uint32");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt32ToIntFixedLengthEncoded8InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint32";
  createForeignTableWithCoercion("INT ENCODING FIXED (8)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-127", "127", "4294967295", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt16ToSmallInt) {
  createForeignTableWithCoercion("SMALLINT", "ParquetCoercionTypes/coercible_uint16");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt16ToSmallIntInformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint16";
  createForeignTableWithCoercion("SMALLINT", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + "",
      getCoercionException("-32767", "32767", "65535", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt16ToTinyInt) {
  createForeignTableWithCoercion("TINYINT", "ParquetCoercionTypes/coercible_uint16");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt16ToTinyIntInformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint16";
  createForeignTableWithCoercion("TINYINT", base_file_name);
  queryAndAssertException("SELECT * FROM " + default_table_name + "",
                          getCoercionException("-127", "127", "65535", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt16ToSmallIntFixedLengthEncoded8) {
  createForeignTableWithCoercion("SMALLINT ENCODING FIXED (8)",
                                 "ParquetCoercionTypes/coercible_uint16");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, UnsignedInt16ToSmallIntFixedLengthEncoded8InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint16";
  createForeignTableWithCoercion("SMALLINT ENCODING FIXED (8)", base_file_name);
  queryAndAssertException("SELECT * FROM " + default_table_name + "",
                          getCoercionException("-127", "127", "65535", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt8ToTinyIntInformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_uint8";
  createForeignTableWithCoercion("TINYINT", base_file_name);
  queryAndAssertException("SELECT * FROM " + default_table_name + "",
                          getCoercionException("-127", "127", "255", base_file_name));
}

TEST_F(ParquetCoercionTest, UnsignedInt8ToTinyInt) {
  createForeignTableWithCoercion("TINYINT", "ParquetCoercionTypes/coercible_uint8");
  sqlAndCompareResult(default_select, {{i(127)}});
}

TEST_F(ParquetCoercionTest, TimestampMilliToTimestampFixedLengthEncoded32) {
  createForeignTableWithCoercion("TIMESTAMP (0) ENCODING FIXED (32)",
                                 "ParquetCoercionTypes/coercible_timestamp_milli");
  sqlAndCompareResult(default_select, {{"2020-03-02 09:59:58"}});
}

TEST_F(ParquetCoercionTest,
       TimestampMilliToTimestampFixedLengthEncoded32InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_timestamp_milli";
  createForeignTableWithCoercion("TIMESTAMP (0) ENCODING FIXED (32)", base_file_name);
  queryAndAssertException("SELECT * FROM " + default_table_name + "",
                          getCoercionException("1901-12-13 20:45:53",
                                               "2038-01-19 03:14:07",
                                               "2038-01-19 03:14:08",
                                               base_file_name));
}

TEST_F(ParquetCoercionTest, TimestampMicroToTimestampFixedLengthEncoded32) {
  createForeignTableWithCoercion("TIMESTAMP (0) ENCODING FIXED (32)",
                                 "ParquetCoercionTypes/coercible_timestamp_micro");
  sqlAndCompareResult(default_select, {{"2020-03-02 09:59:58"}});
}

TEST_F(ParquetCoercionTest,
       TimestampMicroToTimestampFixedLengthEncoded32InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_timestamp_micro";
  createForeignTableWithCoercion("TIMESTAMP (0) ENCODING FIXED (32)", base_file_name);
  queryAndAssertException("SELECT * FROM " + default_table_name + "",
                          getCoercionException("1901-12-13 20:45:53",
                                               "2038-01-19 03:14:07",
                                               "2038-01-19 03:14:08",
                                               base_file_name));
}

TEST_F(ParquetCoercionTest, TimestampNanoToTimestampFixedLengthEncoded32) {
  createForeignTableWithCoercion("TIMESTAMP (0) ENCODING FIXED (32)",
                                 "ParquetCoercionTypes/coercible_timestamp_nano");
  sqlAndCompareResult(default_select, {{"2020-03-02 09:59:58"}});
}

TEST_F(ParquetCoercionTest, TimestampNanoToTimestampFixedLengthEncoded32InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_timestamp_nano";
  createForeignTableWithCoercion("TIMESTAMP (0) ENCODING FIXED (32)", base_file_name);
  queryAndAssertException("SELECT * FROM " + default_table_name + "",
                          getCoercionException("1901-12-13 20:45:53",
                                               "2038-01-19 03:14:07",
                                               "2038-01-19 03:14:08",
                                               base_file_name));
}

TEST_F(ParquetCoercionTest, Int64NoAnnotationToTimestampSeconds) {
  const std::string base_file_name = "ParquetCoercionTypes/coercible_int64_no_annotation";
  createForeignTableWithCoercion("TIMESTAMP (0)", base_file_name);
  sqlAndCompareResult(default_select, {{"01/01/1970 00:02:07"}});
}

TEST_F(ParquetCoercionTest, Int64NoAnnotationToTimestampMilliseconds) {
  const std::string base_file_name = "ParquetCoercionTypes/coercible_int64_no_annotation";
  createForeignTableWithCoercion("TIMESTAMP (3)", base_file_name);
  sqlAndCompareResult(default_select, {{"01/01/1970 00:00:00.127"}});
}

TEST_F(ParquetCoercionTest, Int64NoAnnotationToTimestampMicroseconds) {
  const std::string base_file_name = "ParquetCoercionTypes/coercible_int64_no_annotation";
  createForeignTableWithCoercion("TIMESTAMP (6)", base_file_name);
  sqlAndCompareResult(default_select, {{"01/01/1970 00:00:00.000127"}});
}

TEST_F(ParquetCoercionTest, Int64NoAnnotationToTimestampNanoseconds) {
  const std::string base_file_name = "ParquetCoercionTypes/coercible_int64_no_annotation";
  createForeignTableWithCoercion("TIMESTAMP (9)", base_file_name);
  sqlAndCompareResult(default_select, {{"01/01/1970 00:00:00.000000127"}});
}

TEST_F(ParquetCoercionTest, Int64NoAnnotationToTimestampFixedLengthEncoded32) {
  createForeignTableWithCoercion("TIMESTAMP (0) ENCODING FIXED (32)",
                                 "ParquetCoercionTypes/coercible_int64_no_annotation");
  sqlAndCompareResult(default_select, {{"01/01/1970 00:02:07"}});
}

TEST_F(ParquetCoercionTest,
       Int64NoAnnotationToTimestampFixedLengthEncoded32InformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_int64_no_annotation";
  createForeignTableWithCoercion("TIMESTAMP (0) ENCODING FIXED (32)", base_file_name);
  queryAndAssertException("SELECT * FROM " + default_table_name + "",
                          getCoercionException("1901-12-13 20:45:53",
                                               "2038-01-19 03:14:07",
                                               "292277026596-12-04 15:30:07",
                                               base_file_name));
}

TEST_F(ParquetCoercionTest, Int32NoAnnotationToTimestampFixedLengthEncoded32) {
  createForeignTableWithCoercion("TIMESTAMP (0) ENCODING FIXED (32)",
                                 "ParquetCoercionTypes/coercible_int32_no_annotation");
  sqlAndCompareResult(default_select, {{"01/01/1970 00:02:07"}});
}

TEST_F(ParquetCoercionTest, TimeMilliToTimeFixedLengthEncoded32) {
  createForeignTableWithCoercion("TIME ENCODING FIXED (32)",
                                 "ParquetCoercionTypes/coercible_time_milli");
  sqlAndCompareResult(default_select, {{"23:59:59"}});
}

TEST_F(ParquetCoercionTest, TimeMicroToTimeFixedLengthEncoded32) {
  createForeignTableWithCoercion("TIME ENCODING FIXED (32)",
                                 "ParquetCoercionTypes/coercible_time_micro");
  sqlAndCompareResult(default_select, {{"23:59:59"}});
}

TEST_F(ParquetCoercionTest, TimeNanoToTimeFixedLengthEncoded32) {
  createForeignTableWithCoercion("TIME ENCODING FIXED (32)",
                                 "ParquetCoercionTypes/coercible_time_nano");
  sqlAndCompareResult(default_select, {{"23:59:59"}});
}

TEST_F(ParquetCoercionTest, DateToDateFixedLengthEncoded16) {
  createForeignTableWithCoercion("DATE ENCODING DAYS (16)",
                                 "ParquetCoercionTypes/coercible_date");
  sqlAndCompareResult(default_select, {{"05/08/1970"}});
}

TEST_F(ParquetCoercionTest, DateToDateFixedLengthEncoded16InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_date";
  createForeignTableWithCoercion("DATE ENCODING DAYS (16)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name,
      getCoercionException("1880-04-15", "2059-09-18", "2149-06-06", base_file_name));
}

TEST_F(ParquetCoercionTest, TimestampMilliToDateFixedLengthEncoded16) {
  createForeignTableWithCoercion("DATE ENCODING DAYS (16)",
                                 "ParquetCoercionTypes/coercible_timestamp_milli");
  sqlAndCompareResult(default_select, {{"03/02/2020"}});
}

TEST_F(ParquetCoercionTest, TimestampMicroToDateFixedLengthEncoded16) {
  createForeignTableWithCoercion("DATE ENCODING DAYS (16)",
                                 "ParquetCoercionTypes/coercible_timestamp_micro");
  sqlAndCompareResult(default_select, {{"03/02/2020"}});
}

TEST_F(ParquetCoercionTest, TimestampNanoToDateFixedLengthEncoded16) {
  createForeignTableWithCoercion("DATE ENCODING DAYS (16)",
                                 "ParquetCoercionTypes/coercible_timestamp_nano");
  sqlAndCompareResult(default_select, {{"03/02/2020"}});
}

TEST_F(ParquetCoercionTest, TimestampMilliToDateFixedLengthEncoded32) {
  createForeignTableWithCoercion("DATE ENCODING DAYS (32)",
                                 "ParquetCoercionTypes/coercible_timestamp_milli");
  sqlAndCompareResult(default_select, {{"03/02/2020"}});
}

TEST_F(ParquetCoercionTest, TimestampMicroToDateFixedLengthEncoded32) {
  createForeignTableWithCoercion("DATE ENCODING DAYS (32)",
                                 "ParquetCoercionTypes/coercible_timestamp_micro");
  sqlAndCompareResult(default_select, {{"03/02/2020"}});
}

TEST_F(ParquetCoercionTest, TimestampNanoToDateFixedLengthEncoded32) {
  createForeignTableWithCoercion("DATE ENCODING DAYS (32)",
                                 "ParquetCoercionTypes/coercible_timestamp_nano");
  sqlAndCompareResult(default_select, {{"03/02/2020"}});
}

TEST_F(ParquetCoercionTest, TimestampMilliToDateFixedLengthEncoded16InformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_date16_as_timestamp_milli";
  createForeignTableWithCoercion("DATE ENCODING DAYS (16)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name,
      getCoercionException("1880-04-15", "2059-09-18", "2149-06-06", base_file_name));
}

TEST_F(ParquetCoercionTest, TimestampMicroToDateFixedLengthEncoded16InformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_date16_as_timestamp_micro";
  createForeignTableWithCoercion("DATE ENCODING DAYS (16)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name,
      getCoercionException("1880-04-15", "2059-09-18", "2149-06-06", base_file_name));
}

TEST_F(ParquetCoercionTest, TimestampNanoToDateFixedLengthEncoded16InformationLoss) {
  const std::string base_file_name =
      "ParquetCoercionTypes/non_coercible_date16_as_timestamp_nano";
  createForeignTableWithCoercion("DATE ENCODING DAYS (16)", base_file_name);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name,
      getCoercionException("1880-04-15", "2059-09-18", "2149-06-06", base_file_name));
}

TEST_F(SelectQueryTest, ParquetNotNullWithoutNullOutOfRange) {
  const auto& query = getCreateForeignTableQuery(
      "( int8 TINYINT NOT NULL )", "tinyint_without_null_out_of_range", "parquet");
  sql(query);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name,
      "Parquet column contains values that are outside the range of the "
      "HeavyDB "
      "column type. Consider using a wider column type. Min allowed value: -127"
      ". Max allowed "
      "value: 127. Encountered value: -128"
      ". Error validating statistics of Parquet column "
      "'tinyint' in row group 1 of Parquet file "
      "'" +
          getDataFilesPath() + "tinyint_without_null_out_of_range.parquet'.");
}

TEST_F(SelectQueryTest, ParquetNotNullWithoutNull) {
  const auto& query = getCreateForeignTableQuery(
      "( int8 TINYINT NOT NULL )", "tinyint_without_null", "parquet");
  sql(query);
  sqlAndCompareResult(default_select, {{i(127)}, {i(-127)}});
}

TEST_F(SelectQueryTest, ParquetNotNullWithNull) {
  const std::string base_file_name = "tinyint_with_null";
  const std::string file_name = getDataFilesPath() + base_file_name + ".parquet";
  const auto& query =
      getCreateForeignTableQuery("( int8 TINYINT NOT NULL )", base_file_name, "parquet");
  sql(query);
  queryAndAssertException(
      default_select,
      "A null value was detected in Parquet column 'tinyint' but HeavyDB "
      "column is set to not null in row group 1 of Parquet file '" +
          file_name + "'.");
}

TEST_F(ParquetCoercionTest, Float64ToFloat32) {
  createForeignTableWithCoercion("FLOAT", "ParquetCoercionTypes/coercible_float64");
  sqlAndCompareResult(default_select, {{1e-3f}});
}

TEST_F(ParquetCoercionTest, Float64ToFloat32InformationLoss) {
  const std::string base_file_name = "ParquetCoercionTypes/non_coercible_float64";
  createForeignTableWithCoercion("FLOAT", base_file_name);
  queryAndAssertException(
      default_select,
      getCoercionException(
          "-340282346638528859811704183484516925440.000000",
          "340282346638528859811704183484516925440.000000",
          "179769000000000006323030492138942643493033036433685336215410983289126434148906"
          "289940615299632196609445533816320312774433484859900046491141051651091672734470"
          "972759941382582304802812882753059262973637182942535982636884444611376868582636"
          "745405553206881859340916340092953230149901406738427651121855107737424232448."
          "000000",
          base_file_name));
}

class RegexParserSelectQueryTest : public SelectQueryTest,
                                   public ::testing::WithParamInterface<size_t> {
 protected:
  void SetUp() override { SelectQueryTest::SetUp(); }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    foreign_storage::RegexFileBufferParser::setMaxBufferResize(
        import_export::max_import_buffer_resize_byte_size);
    SelectQueryTest::TearDown();
  }

  void createForeignTable(const std::string& file_name,
                          size_t buffer_size,
                          bool use_line_start_regex = false,
                          const std::string& regex_path_filter = {},
                          const std::string& line_regex = {}) {
    std::string query{"CREATE FOREIGN TABLE " + default_table_name +
                      " (t TIMESTAMP, txt TEXT)"
                      " SERVER default_local_regex_parsed"
                      " WITH (file_path = '" +
                      getFilePath(file_name) + "', buffer_size = " +
                      std::to_string(buffer_size) + ", HEADER = false"};
    if (line_regex.empty()) {
      query += ", line_regex = '^([^\\s]+)\\s+((?:\\w|\\n)+)$'";
    } else {
      query += ", line_regex = '" + line_regex + "'";
    }
    if (use_line_start_regex) {
      query += ", line_start_regex = '" + getLineStartRegex() + "'";
    }
    if (!regex_path_filter.empty()) {
      query += ", regex_path_filter = '" + regex_path_filter + "'";
    }
    query += ");";
    sql(query);
  }

  std::string getLineStartRegex() {
    return "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}";
  }

  std::string getFilePath(const std::string& file_name) {
    return boost::filesystem::canonical(getDataFilesPath() + "regex_parser/" + file_name)
        .string();
  }

  std::string getLongRunningRegex() {
    const std::string base_regex{"([^,]*)"};
    const size_t repeat_count{50};
    std::string regex;
    regex.reserve(base_regex.length() * repeat_count);
    for (size_t i = 0; i < repeat_count; i++) {
      regex += base_regex;
    }
    return regex;
  }
};

TEST_P(RegexParserSelectQueryTest, SingleLines) {
  createForeignTable("single_lines.log", GetParam());
  sqlAndCompareResult(
      "SELECT * FROM " + default_table_name + " ORDER BY t;",
      {{"2/08/2021 18:11:36", "message1"}, {"3/08/2021 18:11:36", "message2"}});
}

TEST_P(RegexParserSelectQueryTest, SingleLinesWithNewLines) {
  createForeignTable("single_lines_with_new_lines.log", GetParam());
  sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY t;",
                      {{"2/08/2021 18:11:36", "message1"},
                       {"3/08/2021 18:11:36", "message2"},
                       {Null, Null},
                       {Null, Null}});
}

TEST_P(RegexParserSelectQueryTest, MultipleLines) {
  createForeignTable("multi_lines.log", GetParam(), true);
  sqlAndCompareResult(
      "SELECT * FROM " + default_table_name + " ORDER BY t;",
      {{"2/08/2021 18:11:36", "message\n1"}, {"4/08/2021 18:11:36", "message\n\n2"}});
}

TEST_P(RegexParserSelectQueryTest, MultipleLinesCompressed) {
  createForeignTable("multi_lines.log.gz", GetParam(), true);
  sqlAndCompareResult(
      "SELECT * FROM " + default_table_name + " ORDER BY t;",
      {{"2/08/2021 18:11:36", "message\n1"}, {"4/08/2021 18:11:36", "message\n\n2"}});
}

TEST_P(RegexParserSelectQueryTest, MultipleLinesWithSomeMismatches) {
  createForeignTable("multi_lines_with_mismatch.log", GetParam(), true);
  sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY t;",
                      {{"1/08/2021 18:11:36", "message\n1"},
                       {"3/08/2021 18:11:36", "message2"},
                       {Null, Null},
                       {Null, Null}});
}

TEST_P(RegexParserSelectQueryTest, MultipleLinesWithFirstLineMismatch) {
  createForeignTable("first_line_mismatch.log", GetParam(), true);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + " ORDER BY t;",
      "First line in file \"" + getFilePath("first_line_mismatch.log") +
          "\" does not match line start regex \"" + getLineStartRegex() + "\"");
}

TEST_P(RegexParserSelectQueryTest, MultipleMultiLineFiles) {
  createForeignTable("", GetParam(), true, ".*multi_lines.*");
  sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY t;",
                      {{"1/08/2021 18:11:36", "message\n1"},
                       {"2/08/2021 18:11:36", "message\n1"},
                       {"2/08/2021 18:11:36", "message\n1"},
                       {"3/08/2021 18:11:36", "message2"},
                       {"4/08/2021 18:11:36", "message\n\n2"},
                       {"4/08/2021 18:11:36", "message\n\n2"},
                       {Null, Null},
                       {Null, Null}});
}

TEST_F(RegexParserSelectQueryTest, MaxBufferResizeLessThanRowSize) {
  SKIP_IF_DISTRIBUTED("Leaf nodes not affected by global variable");
  foreign_storage::RegexFileBufferParser::setMaxBufferResize(8);
  createForeignTable("single_lines.log", 4);
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + " ORDER BY t;",
      "Unable to find an end of line character after reading 7 characters.");
}

TEST_F(RegexParserSelectQueryTest, LongRunningRegex) {
  createForeignTable("../scalar_types.csv",
                     import_export::kImportFileBufferSize,
                     false,
                     {},
                     getLongRunningRegex());
  queryAndAssertException(
      "SELECT * FROM " + default_table_name + " ORDER BY t;",
      "Parsing failure \"The complexity of matching the regular expression exceeded "
      "predefined bounds.  Try refactoring the regular expression to make each choice "
      "made by the state machine unambiguous.  This exception is thrown to prevent "
      "\"eternal\" matches that take an indefinite period time to locate.\" in row "
      "\"boolean,tiny_int,small_int,int,big_int,float,decimal,time,timestamp,date,text,"
      "quoted_text\" in file \"" +
          getDataFilesPath() + "scalar_types.csv\"");
}

INSTANTIATE_TEST_SUITE_P(DifferentBufferSizes,
                         RegexParserSelectQueryTest,
                         testing::Values(4, import_export::kImportFileBufferSize),
                         testing::PrintToStringParamName());

namespace fn = File_Namespace;
class TrackBuffersMockWrapper : public MockDataWrapper {
 public:
  TrackBuffersMockWrapper() : is_first_call_(true) {}

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override {
    parent_data_wrapper_->populateChunkBuffers(required_buffers, optional_buffers);
  }

 protected:
  // When the cache is invlolved, we will prune already cached buffers out of
  // optional_buffers, so to get accurate results for some checks we only want to compare
  // against the first call when the cache is empty.
  bool is_first_call_;
};

class KeySetMockWrapper : public TrackBuffersMockWrapper {
 public:
  KeySetMockWrapper(const std::set<ChunkKey>& data_keys) : data_keys_(data_keys) {}

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override {
    TrackBuffersMockWrapper::populateChunkBuffers(
        required_buffers, optional_buffers, delete_buffer);
    if (is_first_call_) {
      compareExpected(required_buffers, optional_buffers);
      is_first_call_ = true;
    }
  }

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override {
    TrackBuffersMockWrapper::populateChunkMetadata(chunk_metadata_vector);
  }

  virtual void compareExpected(const ChunkToBufferMap& required_buffers,
                               const ChunkToBufferMap& optional_buffers) {
    std::set<ChunkKey> keys;
    for (auto [key, buf] : optional_buffers) {
      keys.emplace(key);
    }
    for (auto [key, buf] : required_buffers) {
      keys.emplace(key);
    }
    ASSERT_EQ(keys, data_keys_);
  }

 protected:
  std::set<ChunkKey> data_keys_;
};

class CountChunksMockWrapper : public TrackBuffersMockWrapper {
 public:
  CountChunksMockWrapper(size_t num_chunks) : expected_num_chunks_(num_chunks) {}

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override {
    TrackBuffersMockWrapper::populateChunkBuffers(
        required_buffers, optional_buffers, delete_buffer);
    if (is_first_call_) {
      // optional_buffers has any cached buffers removed, so only the first call (the
      // one that populates the cache) will contain any optional buffers.
      compareExpectedNumChunks(required_buffers, optional_buffers);
      is_first_call_ = false;
    }
  }

  void compareExpectedNumChunks(const ChunkToBufferMap& required_buffers,
                                const ChunkToBufferMap& optional_buffers) const {
    std::set<ChunkKey> keys;
    for (auto [key, buf] : optional_buffers) {
      keys.emplace(key);
    }
    for (auto [key, buf] : required_buffers) {
      keys.emplace(key);
    }
    ASSERT_EQ(keys.size(), expected_num_chunks_);
  }

 private:
  size_t expected_num_chunks_ = 0;
};

class SizeLimitMockWrapper : public CountChunksMockWrapper {
 public:
  SizeLimitMockWrapper(size_t size, size_t num_chunks)
      : CountChunksMockWrapper(num_chunks), expected_size_(size) {}

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override {
    TrackBuffersMockWrapper::populateChunkBuffers(
        required_buffers, optional_buffers, delete_buffer);
    if (is_first_call_) {
      // optional_buffers has any cached buffers removed, so only the first call (the
      // one that populates the cache) will contain any optional buffers.
      compareExpectedNumChunks(required_buffers, optional_buffers);
      compareExpectedBufferSizes(required_buffers, optional_buffers);
      is_first_call_ = false;
    }
  }

  void compareExpectedBufferSizes(const ChunkToBufferMap& required_buffers,
                                  const ChunkToBufferMap& optional_buffers) const {
    size_t calculated_size = 0;
    for (auto& buffer_maps : {required_buffers, optional_buffers}) {
      for (auto& [key, buffer] : buffer_maps) {
        calculated_size += buffer->size();
      }
    }
    ASSERT_EQ(calculated_size, expected_size_);
  }

 private:
  size_t expected_size_ = 0;
};

class SameFragmentMockWrapper : public SizeLimitMockWrapper {
 public:
  SameFragmentMockWrapper(size_t size, size_t num_chunks)
      : SizeLimitMockWrapper(size, num_chunks) {}
  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override {
    assertSameFragment(required_buffers, optional_buffers);
    SizeLimitMockWrapper::populateChunkBuffers(
        required_buffers, optional_buffers, delete_buffer);
  }

 private:
  // We want to prioritize any optional chunks that share the same fragment as the
  // required buffer, so check that chunks share the same fragment id.
  void assertSameFragment(const ChunkToBufferMap& required_buffers,
                          const ChunkToBufferMap& optional_buffers) {
    auto required_fragment = required_buffers.begin()->first[CHUNK_KEY_FRAGMENT_IDX];
    for (const auto& [key, buf] : optional_buffers) {
      ASSERT_EQ(key[CHUNK_KEY_FRAGMENT_IDX], required_fragment);
    }
  }
};

class PrefetchLimitTest : public RecoverCacheQueryTest {
 protected:
  static constexpr size_t max_buffer_size_ = 1ULL << 31;  // 2GB
  static constexpr size_t char_buffer_size_ = 1U;         // Num elements.
  static constexpr size_t int_buffer_size_ = 4U;          // Num elements.
  static constexpr size_t index_buffer_size_ = 8U;        // Num elements + 1 for 0 index.
  static constexpr size_t cache_size_ = 1ULL << 34;       // 16GB

  void SetUp() override {
    SKIP_SETUP_IF_DISTRIBUTED("Cache settings are not distributed yet.");
    // TODO(Misiu): Right now we only test parquet since it prefetches multi-fragment and
    // does not prefetch for metdata scan.
    wrapper_type_ = "parquet";
    RecoverCacheQueryTest::SetUp();
    resetPersistentStorageMgr({cache_path_, fn::DiskCacheLevel::fsi, 0, cache_size_});
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    resetPersistentStorageMgr({cache_path_, fn::DiskCacheLevel::fsi});
    RecoverCacheQueryTest::TearDown();
  }

  void setMockWrapper(std::shared_ptr<MockDataWrapper> mock_wrapper,
                      const std::string& table_name) {
    auto db_id = cat_->getCurrentDB().dbId;
    auto tb_id = cat_->getMetadataForTable(table_name, false)->tableId;
    auto fsm = psm_->getForeignStorageMgr();
    auto parent_wrapper = fsm->hasDataWrapperForChunk({db_id, tb_id})
                              ? fsm->getDataWrapper({db_id, tb_id})
                              : nullptr;
    mock_wrapper->setParentWrapper(parent_wrapper);
    fsm->setDataWrapper({db_id, tb_id}, mock_wrapper);
  }

  void createTextTable() {
    sql(createForeignTableQuery({{"i", "INTEGER"},
                                 {"text_encoded", "TEXT"},
                                 {"text_unencoded", "TEXT ENCODING NONE"},
                                 {"text_array", "TEXT[]"}},
                                getDataFilesPath() + "0_9.parquet",
                                wrapper_type_,
                                {{"fragment_size", "1"}}));
    sql("SELECT COUNT(*) FROM " + default_table_name);
  }

  std::vector<std::vector<NullableTargetValue>> textTableResults() {
    std::vector<std::vector<NullableTargetValue>> expected;
    for (int number = 0; number < 10; number++) {
      expected.push_back({i(number),
                          std::to_string(number),
                          std::to_string(number),
                          array({std::to_string(number)})});
    }
    return expected;
  }
};

// If the cache is too small to prefetch all the chunks we want for the query, then we
// should only fetch the amount that can fit.
TEST_F(PrefetchLimitTest, MultiFragmentLimit) {
  // 3 full fragments (2 varlen data + 2 varlen index + 2 int chunks) + 2 int chunks from
  // fourth fragment.
  size_t size_limit =
      ((max_buffer_size_ + index_buffer_size_ + int_buffer_size_) * 2U) * 3U +
      int_buffer_size_ * 2U;
  size_t actual_size = (int_buffer_size_ + char_buffer_size_ +
                        (index_buffer_size_ + int_buffer_size_) * 2U) *
                           3U +
                       int_buffer_size_ * 2U;
  cache_->setDataSizeLimit(size_limit);
  createTextTable();

  auto mock_data_wrapper = std::make_shared<SizeLimitMockWrapper>(actual_size, 20);
  setMockWrapper(mock_data_wrapper, default_table_name);

  sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY i;",
                      textTableResults());
}

// If we extend past one fragment, then we should fill the first fragment and then start
// populating the second.
TEST_F(PrefetchLimitTest, OneAndHalfFragmentLimit) {
  // 1 varlen data chunk + 1 varlen index chunk + 2 int chunks
  size_t size_limit = max_buffer_size_ + index_buffer_size_ + int_buffer_size_ * 2U;
  size_t actual_size = char_buffer_size_ + index_buffer_size_ + int_buffer_size_ * 2U;
  cache_->setDataSizeLimit(size_limit);
  createTextTable();

  auto mock_data_wrapper = std::make_shared<SizeLimitMockWrapper>(actual_size, 4);
  setMockWrapper(mock_data_wrapper, default_table_name);

  std::vector<std::vector<NullableTargetValue>> expected;
  for (int number = 0; number < 10; number++) {
    expected.push_back({i(number), std::to_string(number)});
  }

  sqlAndCompareResult(
      "SELECT i, text_unencoded FROM " + default_table_name + " ORDER BY i;", expected);
}

// If the cache is too small to contain one full fragment, then we should partially
// populate it with chunks from that one fragment.
TEST_F(PrefetchLimitTest, PartialFragmentLimit) {
  // TODO(Misiu): This test is currently inconsistent.  Depending which column we try to
  // fetch first (which is arbitrarily determined by the QueryEngine) we will store a
  // different set of chunks (which can have different sizes).  We can fix this by calling
  // fetchBuffer() directly from the ForeignStorageMgr after setting the parallelism
  // hints. This will simulate the call that we would get from the QE but guarantee the
  // consistency for which chunk is requested first.  This format should be used to
  // rewrite all of the PrefetchLimitTests.
  GTEST_SKIP() << "Test currently disabled";

  // 1 varlen data + 1 varlen index + 2 int chunk.
  size_t size_limit = max_buffer_size_ + index_buffer_size_ + int_buffer_size_ * 2U;
  size_t actual_size = int_buffer_size_ + index_buffer_size_ + int_buffer_size_ * 2U;
  cache_->setDataSizeLimit(size_limit);
  createTextTable();

  auto mock_data_wrapper = std::make_shared<SameFragmentMockWrapper>(actual_size, 4);
  setMockWrapper(mock_data_wrapper, default_table_name);

  sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY i;",
                      textTableResults());
}

// If we don't have space for both varlen keys then we store neither.
TEST_F(PrefetchLimitTest, PartialFragmentLimitVarlen) {
  // 1 varlen data + 1 varlen index + 1 int chunk
  size_t size_limit = max_buffer_size_ + index_buffer_size_ + int_buffer_size_;
  size_t actual_size = char_buffer_size_ + index_buffer_size_;
  cache_->setDataSizeLimit(size_limit);
  createTextTable();

  // Despite having room to store a third chunk, we should only have two because we won't
  // store a varlen index chunk without it's corresponding data chunk (which we don't have
  // space for).
  auto mock_data_wrapper = std::make_shared<SameFragmentMockWrapper>(actual_size, 2);
  setMockWrapper(mock_data_wrapper, default_table_name);

  std::vector<std::vector<NullableTargetValue>> expected;
  for (int number = 0; number < 10; number++) {
    expected.push_back({std::to_string(number)});
  }

  sqlAndCompareResult("SELECT text_unencoded FROM " + default_table_name + ";", expected);
}

// This point chunk is made up of 3 physical chunks.
// Currently tries to add more than two fragments.  1 is required, the other two are
// optional. Should return one required and one optional.
TEST_F(PrefetchLimitTest, PartialFragmentLimitPoint) {
  // 2 fragments (1 varlen data + 1 varlen index + 1 empty chunk) + 1 varlen data.
  size_t size_limit = (max_buffer_size_ + index_buffer_size_) * 2U + max_buffer_size_;
  cache_->setDataSizeLimit(size_limit);

  sql(createForeignTableQuery({{"p", "POINT"}},
                              getDataFilesPath() + "GeoTypes/point.parquet",
                              wrapper_type_,
                              {{"fragment_size", "1"}}));
  sql("SELECT COUNT(*) FROM " + default_table_name);

  // Despite having room for part of a third fragment we should only store 2, because
  // caching a geo-column should be atomic.
  auto mock_data_wrapper = std::make_shared<CountChunksMockWrapper>(6);
  setMockWrapper(mock_data_wrapper, default_table_name);

  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";",
                      {{"POINT (0 0)"}, {"POINT (1 1)"}, {"POINT (2 2)"}});
}

// This linestring is composed of 5 physical chunks
TEST_F(PrefetchLimitTest, PartialFragmentLimitLinestring) {
  // 2 fragments (2 varlen data + 2 varlen index + 1 empty) + 1 varlen data.
  size_t size_limit =
      (max_buffer_size_ * 2U + index_buffer_size_ * 2U) * 2U + max_buffer_size_;
  cache_->setDataSizeLimit(size_limit);

  sql(createForeignTableQuery({{"l", "LINESTRING"}},
                              getDataFilesPath() + "GeoTypes/linestring.parquet",
                              wrapper_type_,
                              {{"fragment_size", "1"}}));
  sql("SELECT COUNT(*) FROM " + default_table_name);

  // Despite having room for part of a third fragment we should only store 2, because
  // caching a geo-column should be atomic.
  auto mock_data_wrapper = std::make_shared<CountChunksMockWrapper>(10);
  setMockWrapper(mock_data_wrapper, default_table_name);

  sqlAndCompareResult(
      "SELECT * FROM " + default_table_name + ";",
      {{"LINESTRING (0 0,0 0)"}, {"LINESTRING (1 1,2 2,3 3)"}, {"LINESTRING (2 2,3 3)"}});
}

// This polygon is composed of 8 physical chunks
TEST_F(PrefetchLimitTest, PartialFragmentLimitPolygon) {
  // 2 fragments (3 varlen data + 3 varlen index + 2 empty) + 1 varlen data.
  size_t size_limit =
      (max_buffer_size_ * 3U + index_buffer_size_ * 3U) * 2U + max_buffer_size_;
  cache_->setDataSizeLimit(size_limit);

  sql(createForeignTableQuery({{"p", "POLYGON"}},
                              getDataFilesPath() + "GeoTypes/polygon.parquet",
                              wrapper_type_,
                              {{"fragment_size", "1"}}));
  sql("SELECT COUNT(*) FROM " + default_table_name);

  // Despite having room for part of a third fragment we should only store 2, because
  // caching a geo-column should be atomic.
  auto mock_data_wrapper = std::make_shared<CountChunksMockWrapper>(14);
  setMockWrapper(mock_data_wrapper, default_table_name);

  sqlAndCompareResult("SELECT * FROM " + default_table_name + ";",
                      {{"POLYGON ((0 0,1 0,0 1,1 1,0 0))"},
                       {"POLYGON ((5 4,7 4,6 5,5 4))"},
                       {"POLYGON ((1 1,3 1,2 3,1 1))"}});
}

// This multipolygon is composed of 10 physical chunks.
TEST_F(PrefetchLimitTest, PartialFragmentLimitMultiPolygon) {
  // 2 fragments (4 varlen data + 4 varlen index + 2 empty) + 1 varlen data.
  size_t size_limit =
      (max_buffer_size_ * 4U + index_buffer_size_ * 4U) * 2U + max_buffer_size_;
  cache_->setDataSizeLimit(size_limit);

  sql(createForeignTableQuery({{"m", "MULTIPOLYGON"}},
                              getDataFilesPath() + "GeoTypes/multipolygon.parquet",
                              wrapper_type_,
                              {{"fragment_size", "1"}}));
  sql("SELECT COUNT(*) FROM " + default_table_name);

  // Despite having room for part of a third fragment we should only store 2, because
  // caching a geo-column should be atomic
  auto mock_data_wrapper = std::make_shared<CountChunksMockWrapper>(18);
  setMockWrapper(mock_data_wrapper, default_table_name);

  sqlAndCompareResult(
      "SELECT * FROM " + default_table_name + ";",
      {{"MULTIPOLYGON (((0 0,1 0,0 1,0 0)))"},
       {"MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"},
       {"MULTIPOLYGON (((0 0,3 0,0 3,0 0)),((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"}});
}

// Required buffers don't fit in cache.
TEST_F(PrefetchLimitTest, RequiredBuffersDontFit) {
  cache_->setDataSizeLimit(1);
  createTextTable();

  sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY i;",
                      textTableResults());
}

// If the cache is disabled, then we should only be prefetching one fragment's worth of
// chunks at a time, since that is the amount that needs to fit in memory for a query to
// process anyways.
TEST_F(PrefetchLimitTest, NoCacheFragmentOnly) {
  size_t actual_size = (char_buffer_size_ + index_buffer_size_ * 2U + int_buffer_size_ +
                        int_buffer_size_ * 2U);
  resetPersistentStorageMgr({cache_path_, fn::DiskCacheLevel::none, 0});
  createTextTable();

  // The non-cached parallelism level for parquet is only intra-fragment.
  auto mock_data_wrapper = std::make_shared<SameFragmentMockWrapper>(actual_size, 6);
  setMockWrapper(mock_data_wrapper, default_table_name);

  sqlAndCompareResult("SELECT * FROM " + default_table_name + " ORDER BY i;",
                      textTableResults());
}

class TableInteractionTest : public ForeignTableTest {
 protected:
  inline static const std::string table_name{"table_1"};

  void SetUp() override {
    ForeignTableTest::SetUp();
    sql("drop table if exists " + table_name + ";");
    sql("drop foreign table if exists " + default_table_name + ";");
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    sql("drop table if exists " + table_name + ";");
    sql("drop foreign table if exists " + default_table_name + ";");
    ForeignTableTest::TearDown();
  }
};

TEST_F(TableInteractionTest, SharedDictionaryDisabled) {
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", "example_1", "csv");
  queryAndAssertException(
      "create table " + table_name + " (t text, shared dictionary (t) references " +
          default_table_name + "(t));",
      "Attempting to share dictionary with foreign table " + default_table_name +
          ".  Foreign table dictionaries cannot currently be shared.");
}

class AlterOdbcClearCacheTest : public RefreshTests {
 protected:
  void SetUp() override {
    wrapper_type_ = "sqlite";
    SKIP_SETUP_IF_ODBC_DISABLED();
    RefreshTests::SetUp();
  }

  void TearDown() override {
    if (skip_teardown_) {
      return;
    }
    RefreshTests::TearDown();
  }
};

TEST_F(AlterOdbcClearCacheTest, AlterOdbcClearCache) {
  sql(createForeignTableQuery(
      {{"t", "TEXT"}, {"i", "INTEGER"}, {"d", "DOUBLE"}},
      getDataFilesPath() + "/example_2.csv",
      wrapper_type_,
      {{"SQL_SELECT", "select t, i, d from " + default_table_name},
       {"SQL_ORDER_BY", "i"}}));
  ASSERT_EQ(cache_->getNumCachedChunks(), 0U);
  sql("SELECT * FROM " + default_table_name + ";");
  ASSERT_GT(cache_->getNumCachedChunks(), 0U);
  sql("ALTER FOREIGN TABLE " + default_table_name + " SET (SQL_ORDER_BY = 'i');");
  ASSERT_EQ(cache_->getNumCachedChunks(), 0U);
}

class OdbcSkipMetadataTest : public ForeignTableTest, public TempDirManager {
  void SetUp() override {
    wrapper_type_ = "sqlite";
    ForeignTableTest::SetUp();
    sql("DROP FOREIGN TABLE IF EXISTS " + default_table_name);
    sql("DROP FOREIGN TABLE IF EXISTS " + default_table_name + "_1");
  }

  void TearDown() override {
    sql("DROP FOREIGN TABLE IF EXISTS " + default_table_name);
    sql("DROP FOREIGN TABLE IF EXISTS " + default_table_name + "_1");
    ForeignTableTest::TearDown();
  }
};

TEST_F(OdbcSkipMetadataTest, JoinOnNonDictColumn) {
  std::string t1{default_table_name};
  std::string t2{default_table_name + "_1"};
  sql(createForeignTableQuery(
      {{"t", "TEXT"}, {"i", "INTEGER"}, {"d", "DOUBLE"}},
      getDataFilesPath() + "/example_2.csv",
      wrapper_type_,
      {{"SQL_SELECT", "select t, i, d from " + t1}, {"SQL_ORDER_BY", "i"}}));
  sql(createForeignTableQuery(
      {{"t", "TEXT"}, {"i", "INTEGER"}, {"d", "DOUBLE"}},
      getDataFilesPath() + "/example_2.csv",
      wrapper_type_,
      {{"SQL_SELECT", "select t, i, d from " + t2}, {"SQL_ORDER_BY", "i"}},
      t2));
  sqlAndCompareResult("SELECT " + t1 + ".i, " + t2 + ".i from " + t1 + ", " + t2 +
                          " WHERE " + t1 + ".i = " + t2 + ".i ORDER BY " + t1 + ".i",
                      {{i(1), i(1)},
                       {i(1), i(1)},
                       {i(1), i(1)},
                       {i(1), i(1)},
                       {i(1), i(1)},
                       {i(1), i(1)},
                       {i(1), i(1)},
                       {i(1), i(1)},
                       {i(1), i(1)},
                       {i(2), i(2)},
                       {i(2), i(2)},
                       {i(2), i(2)},
                       {i(2), i(2)},
                       {i(3), i(3)}});
}

// Added to make sure we cross the ExpressionRange generaetion path for empty fsi tables.
TEST_F(OdbcSkipMetadataTest, JoinOnNonDictColumnEmptyRefresh) {
  std::string t1{default_table_name};
  std::string t2{default_table_name + "_1"};
  bf::copy_file(getDataFilesPath() + "example_2.csv", test_temp_dir + "/example_2.csv");
  sql(createForeignTableQuery(
      {{"t", "TEXT"}, {"i", "INTEGER"}, {"d", "DOUBLE"}},
      test_temp_dir + "/example_2.csv",
      wrapper_type_,
      {{"SQL_SELECT", "select t, i, d from " + t1}, {"SQL_ORDER_BY", "i"}}));
  sql(createForeignTableQuery(
      {{"t", "TEXT"}, {"i", "INTEGER"}, {"d", "DOUBLE"}},
      getDataFilesPath() + "/example_2.csv",
      wrapper_type_,
      {{"SQL_SELECT", "select t, i, d from " + t2}, {"SQL_ORDER_BY", "i"}},
      t2));
  bf::ofstream(test_temp_dir + "/empty.csv");
  createODBCSourceTable(t1,
                        {{"t", "TEXT"}, {"i", "INTEGER"}, {"d", "DOUBLE"}},
                        test_temp_dir + "/empty.csv",
                        wrapper_type_);
  sql("REFRESH FOREIGN TABLES " + t1 + " WITH (evict='true')");
  sqlAndCompareResult("SELECT " + t1 + ".i, " + t2 + ".i from " + t1 + ", " + t2 +
                          " WHERE " + t1 + ".i = " + t2 + ".i ORDER BY " + t1 + ".i",
                      {});
}

class UntouchedRefreshTest : public PrefetchLimitTest {};

TEST_F(UntouchedRefreshTest, DefaultRefresh) {
  sql(createForeignTableQuery({{"i", "INTEGER"},
                               {"text_encoded", "TEXT"},
                               {"text_unencoded", "TEXT ENCODING NONE"},
                               {"text_array", "TEXT[]"}},
                              getDataFilesPath() + "0_9.parquet",
                              wrapper_type_));
  auto mock_data_wrapper = std::make_shared<MockDataWrapper>();
  mock_data_wrapper->throwOnMetadataScan(true);
  setMockWrapper(mock_data_wrapper, default_table_name);

  sql("REFRESH FOREIGN TABLES " + default_table_name);
  queryAndAssertException("SELECT COUNT(*) FROM " + default_table_name,
                          "populateChunkMetadata mock exception");
}

TEST_F(UntouchedRefreshTest, EvictRefresh) {
  sql(createForeignTableQuery({{"i", "INTEGER"},
                               {"text_encoded", "TEXT"},
                               {"text_unencoded", "TEXT ENCODING NONE"},
                               {"text_array", "TEXT[]"}},
                              getDataFilesPath() + "0_9.parquet",
                              wrapper_type_));
  auto mock_data_wrapper = std::make_shared<MockDataWrapper>();
  mock_data_wrapper->throwOnMetadataScan(true);
  setMockWrapper(mock_data_wrapper, default_table_name);
  sql("REFRESH FOREIGN TABLES " + default_table_name + " WITH (EVICT='TRUE')");
  queryAndAssertException("SELECT COUNT(*) FROM " + default_table_name,
                          "populateChunkMetadata mock exception");
}

TEST_F(UntouchedRefreshTest, AppendRefresh) {
  sql(createForeignTableQuery({{"i", "INTEGER"},
                               {"text_encoded", "TEXT"},
                               {"text_unencoded", "TEXT ENCODING NONE"},
                               {"text_array", "TEXT[]"}},
                              getDataFilesPath() + "0_9.parquet",
                              wrapper_type_,
                              {{"REFRESH_UPDATE_TYPE", "APPEND"}}));
  auto mock_data_wrapper = std::make_shared<MockDataWrapper>();
  mock_data_wrapper->throwOnMetadataScan(true);
  setMockWrapper(mock_data_wrapper, default_table_name);

  sql("REFRESH FOREIGN TABLES " + default_table_name);
  queryAndAssertException("SELECT COUNT(*) FROM " + default_table_name,
                          "populateChunkMetadata mock exception");
}

int main(int argc, char** argv) {
  g_enable_fsi = true;
  g_enable_s3_fsi = true;
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  // get dirname of test binary
  test_binary_file_path = bf::canonical(argv[0]).parent_path().string();
  test_temp_dir = test_binary_file_path + "/fsi_temp_dir/";

  po::options_description desc("Options");
  desc.add_options()("run-odbc-tests", "Run ODBC DML tests.");
  po::variables_map vm = DBHandlerTestFixture::initTestArgs(argc, argv, desc);
  g_run_odbc = (vm.count("run-odbc-tests"));

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_fsi = false;
  return err;
}
