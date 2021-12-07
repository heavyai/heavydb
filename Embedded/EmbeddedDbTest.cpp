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

// project headers
#include "DBEngine.h"
#include "Shared/ArrowUtil.h"
#include "Shared/InlineNullValues.h"
#include "Shared/scope.h"

// boost headers
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

// std headers
#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>

// arrow headers
#include <arrow/api.h>
#include <arrow/csv/reader.h>
#include <arrow/io/file.h>

// Google Test
#include <gtest/gtest.h>

// Global variables controlling execution
extern bool g_enable_columnar_output;
extern bool g_enable_lazy_fetch;

// DBEngine Instance
static std::shared_ptr<EmbeddedDatabase::DBEngine> g_dbe;

// Input files' names
static const char* TABLE6x4_CSV_FILE =
    "../../Tests/EmbeddedDataFiles/embedded_db_test_6x4table.csv";
static const char* JOIN_TABLE_CSV_FILE =
    "../../Tests/EmbeddedDataFiles/embedded_db_test_join_table.csv";
static const char* NULLSTABLE6x4_CSV_FILE =
    "../../Tests/EmbeddedDataFiles/embedded_db_test_nulls_table.csv";

//  Content of the table stored in $TABLE6x4_CSV_FILE file
static std::array<int32_t, 6> table6x4_col_i32 = {0, 0, 0, 1, 1, 1};
static std::array<int64_t, 6> table6x4_col_i64 = {0, 0, 0, 1, 1, 1};
static std::array<int64_t, 6> table6x4_col_bi = {1, 2, 3, 4, 5, 6};
static std::array<double, 6> table6x4_col_d = {10.1, 20.2, 30.3, 40.4, 50.5, 60.6};

//  HELPERS
namespace {

// Processes command line args, return options string
std::string parse_cli_args(int argc, char* argv[]) {
  namespace po = boost::program_options;

  int calcite_port = 5555;
  bool columnar_output = true;
  std::string catalog_path = "tmp";
  std::string opt_str;

  po::options_description desc("Options");

  desc.add_options()("help,h", "Print help messages ");

  desc.add_options()("data,d",
                     po::value<std::string>(&catalog_path)->default_value(catalog_path),
                     "Directory path to OmniSci catalogs");

  desc.add_options()("calcite-port,p",
                     po::value<int>(&calcite_port)->default_value(calcite_port),
                     "Calcite port");

  desc.add_options()("enable-columnar-output,o",
                     po::value<bool>(&columnar_output)->default_value(columnar_output),
                     "Enable columnar_output");

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

    if (vm.count("help")) {
      std::cout << desc;
      std::exit(EXIT_SUCCESS);
    }
    po::notify(vm);

    opt_str = catalog_path + " --calcite-port " + std::to_string(calcite_port);
    if (columnar_output) {
      opt_str += " --enable-columnar-output";
    }
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    std::cout << desc;
    std::exit(EXIT_FAILURE);
  }

  return opt_str;
}

// Creates a table from TABLE6x4_CSV_FILE file with provided
// parameters:
//  + table_name -- name of the table
//  + fragment_size -- fragment size for the table; 0 to use the default value
void load_table6x4_csv(std::string csv_file,
                       std::string table_name,
                       size_t fragment_size) {
  if (!g_dbe) {
    throw std::runtime_error("DBEngine is not initialized.  Aborting.\n");
  }
  std::string frag_size_param =
      fragment_size > 0 ? ", fragment_size=" + std::to_string(fragment_size) : "";

  std::string create_query = "CREATE TEMPORARY TABLE " + table_name +
                             " (t TEXT, i INT, bi BIGINT, d DOUBLE) "
                             "WITH (storage_type='CSV:" +
                             csv_file + "'" + frag_size_param + ");";

  g_dbe->executeDDL(create_query);
}

// Load CSV file into a temporary table using Arrow import
void import_file(const std::string& file_name,
                 const std::string& table_name,
                 size_t fragment_size = 0) {
  arrow::io::IOContext io_context = arrow::io::default_io_context();
  auto arrow_parse_options = arrow::csv::ParseOptions::Defaults();
  auto arrow_read_options = arrow::csv::ReadOptions::Defaults();
  auto arrow_convert_options = arrow::csv::ConvertOptions::Defaults();
  std::shared_ptr<arrow::io::ReadableFile> inp;
  auto file_result = arrow::io::ReadableFile::Open(file_name);
  ARROW_THROW_NOT_OK(file_result.status());
  inp = file_result.ValueOrDie();
  auto table_reader_result = arrow::csv::TableReader::Make(
      io_context, inp, arrow_read_options, arrow_parse_options, arrow_convert_options);

  ARROW_THROW_NOT_OK(table_reader_result.status());
  auto table_reader = table_reader_result.ValueOrDie();
  std::shared_ptr<arrow::Table> arrowTable;
  auto arrow_table_result = table_reader->Read();
  ARROW_THROW_NOT_OK(arrow_table_result.status());
  arrowTable = arrow_table_result.ValueOrDie();

  g_dbe->importArrowTable(table_name, arrowTable, fragment_size);
}

template <typename TYPE>
constexpr TYPE null_builder() {
  static_assert(std::is_floating_point_v<TYPE> || std::is_integral_v<TYPE>,
                "Unsupported type");

  if constexpr (std::is_floating_point_v<TYPE>) {
    return inline_fp_null_value<TYPE>();
  } else if constexpr (std::is_integral_v<TYPE>) {
    return inline_int_null_value<TYPE>();
  }
}

template <typename TYPE, size_t len>
void compare_columns(const std::array<TYPE, len>& expected,
                     const std::shared_ptr<arrow::ChunkedArray>& actual) {
  using ArrowColType = arrow::NumericArray<typename arrow::CTypeTraits<TYPE>::ArrowType>;
  const arrow::ArrayVector& chunks = actual->chunks();

  TYPE null_val = null_builder<TYPE>();

  for (int i = 0, k = 0; i < actual->num_chunks(); i++) {
    auto chunk = chunks[i];
    auto arrow_row_array = std::static_pointer_cast<ArrowColType>(chunk);

    const TYPE* chunk_data = arrow_row_array->raw_values();
    for (int64_t j = 0; j < arrow_row_array->length(); j++, k++) {
      if (expected[k] == null_val) {
        CHECK(chunk->IsNull(j));
      } else {
        CHECK(chunk->IsValid(j));
        ASSERT_EQ(expected[k], chunk_data[j]);
      }
    }
  }
};

//  Testing helper functions for 6x4 table contained in $TABLE6x4_CSV_FILE file.
//  NOTE: load_table6x4_csv() creates table with Arrow types <dictionary, int32, int64,
//  float64> so we check that.
void test_arrow_table_conversion_table6x4(size_t fragment_size) {
  auto cursor = g_dbe->executeDML("select * from test_chunked");

  ASSERT_NE(cursor, nullptr);
  auto table = cursor->getArrowTable();

  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 4);
  ASSERT_EQ(table->num_rows(), (int64_t)6);

  auto column = table->column(1);
  size_t expected_chunk_count =
      (g_enable_lazy_fetch || !g_enable_columnar_output)
          ? 1
          : (table->num_rows() + fragment_size - 1) / fragment_size;

  size_t actual_chunks_count = column->num_chunks();
  ASSERT_EQ(actual_chunks_count, expected_chunk_count);

  ASSERT_NE(table->column(0), nullptr);
  ASSERT_NE(table->column(1), nullptr);
  ASSERT_NE(table->column(2), nullptr);
  ASSERT_NE(table->column(3), nullptr);

  auto schema = table->schema();
  ASSERT_EQ(schema->num_fields(), 4);
  ASSERT_EQ(schema->GetFieldByName("i")->type()->Equals(arrow::int32()), true);
  ASSERT_EQ(schema->GetFieldByName("bi")->type()->Equals(arrow::int64()), true);
  ASSERT_EQ(schema->GetFieldByName("d")->type()->Equals(arrow::float64()), true);

  compare_columns(table6x4_col_i32, table->column(1));
  compare_columns(table6x4_col_bi, table->column(2));
  compare_columns(table6x4_col_d, table->column(3));
}

//  Performs tests for 6x4 table contained in $TABLE6x4_CSV_FILE for various
//  values of enable_columnar_output, enable_lazy_fetch, fragment_size
void test_chunked_conversion(bool enable_columnar_output,
                             bool enable_lazy_fetch,
                             size_t fragment_size) {
  bool prev_enable_columnar_output = g_enable_columnar_output;
  bool prev_enable_lazy_fetch = g_enable_lazy_fetch;

  ScopeGuard reset = [prev_enable_columnar_output, prev_enable_lazy_fetch] {
    g_enable_columnar_output = prev_enable_columnar_output;
    g_enable_lazy_fetch = prev_enable_lazy_fetch;
  };

  g_enable_columnar_output = enable_columnar_output;
  g_enable_lazy_fetch = enable_lazy_fetch;
  test_arrow_table_conversion_table6x4(/*fragment_size=*/fragment_size);
}

}  // anonymous namespace

//  Tests `DBEngine::importArrowTable()'  functionality
//  NOTE: For $TABLE6x4_CSV_FILE file, import_file() creates a table with Arrow
//  types <dictionary, int64, int64, float64> rather than <dictionary, int32, int64,
//  float64> (the latter is obtained via SQL query loading), so we perform checks
//  accordingly
TEST(DBEngine, ImportArrowTable) {
  import_file(TABLE6x4_CSV_FILE, "loading_test", /*fragment_size=*/4);
  auto cursor = g_dbe->executeDML("select * from loading_test");
  ASSERT_NE(cursor, nullptr);
  auto table = cursor->getArrowTable();
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 4);
  ASSERT_EQ(table->num_rows(), (int64_t)6);

  std::shared_ptr<arrow::Schema> schema = table->schema();

  ASSERT_EQ(schema->num_fields(), 4);
  ASSERT_EQ(schema->GetFieldByName("i")->type()->Equals(arrow::int64()), true);
  ASSERT_EQ(schema->GetFieldByName("bi")->type()->Equals(arrow::int64()), true);
  ASSERT_EQ(schema->GetFieldByName("d")->type()->Equals(arrow::float64()), true);

  compare_columns(table6x4_col_i64, table->column(1));
  compare_columns(table6x4_col_bi, table->column(2));
  compare_columns(table6x4_col_d, table->column(3));
}

TEST(DBEngine, SelectBool) {
  GTEST_SKIP();  //  `DBEngine::importArrowTable()' incorrectly imports bools so until
                 //  that is fixed, this test will fail. Remove GTEST_SKIP(); after that 
                 //  bug is fixed.
                 //  Cf: https://github.com/intel-ai/omniscidb/issues/218#issue-1070805451 
  auto cursor = g_dbe->executeDML("SELECT * FROM join_table;");
  ASSERT_NE(cursor, nullptr);
  auto table = cursor->getArrowTable();
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 3);
  ASSERT_EQ(table->num_rows(), 2);
  compare_columns(std::array<int64_t, 2>{0, 1}, table->column(0));
  compare_columns(std::array<bool, 2>{false, true}, table->column(1));
  compare_columns(std::array<int64_t, 2>{100, 200}, table->column(2));
}

//  Tests getArrowRecordBatch() for all columns selection
TEST(DBEngine, ArrowRecordBatch_SimpleSelectAll) {
  auto cursor = g_dbe->executeDML("select * from test");
  ASSERT_NE(cursor, nullptr);
  auto rbatch = cursor->getArrowRecordBatch();
  ASSERT_NE(rbatch, nullptr);
  ASSERT_EQ(rbatch->num_columns(), 4);
  ASSERT_EQ(rbatch->num_rows(), (int64_t)6);
}

//  Tests getArrowRecordBatch() for two columns (TEXT "t", INT "i") selection
TEST(DBEngine, ArrowRecordBatch_TextIntSelect) {
  auto cursor = g_dbe->executeDML("select t, i from test");
  ASSERT_NE(cursor, nullptr);
  auto rbatch = cursor->getArrowRecordBatch();
  ASSERT_NE(rbatch, nullptr);
  ASSERT_EQ(rbatch->num_columns(), 2);
  ASSERT_EQ(rbatch->num_rows(), (int64_t)6);
}

//  Tests getArrowTable() for three columns (TEXT "t", INT "i", BIGINT "bi") selection
TEST(DBEngine, ArrowTable_TextIntBigintSelect) {
  auto cursor = g_dbe->executeDML("select t, i, bi from test");
  ASSERT_NE(cursor, nullptr);
  auto table = cursor->getArrowTable();
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 3);
  ASSERT_EQ(table->num_rows(), (int64_t)6);
}

//  Two column selection with filtering. The result is empty table.
TEST(DBEngine, ArrowTable_EmptySelection) {
  auto cursor = g_dbe->executeDML("select i, d from test where d > 1000");
  ASSERT_NE(cursor, nullptr);
  auto table = cursor->getArrowTable();
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 2);
  ASSERT_EQ(table->num_rows(), (int64_t)0);
}

//  Chunked Arrow Table Conversion Tests
TEST(DBEngine, ArrowTableChunked_Conversion1) {
  test_chunked_conversion(/*columnar_output=*/false, /*lazy_fetch=*/false, 4);
}

TEST(DBEngine, ArrowTableChunked_Conversion2) {
  test_chunked_conversion(/*columnar_output=*/true, /*lazy_fetch=*/false, 4);
}

TEST(DBEngine, ArrowTableChunked_Conversion3) {
  test_chunked_conversion(/*columnar_output=*/false, /*lazy_fetch=*/true, 4);
}

TEST(DBEngine, ArrowTableChunked_Conversion4) {
  test_chunked_conversion(/*columnar_output=*/true, /*lazy_fetch=*/true, 4);
}

//  Projection operation test (column "d")
TEST(DBEngine, ArrowTableChunked_SingleColumnConversion) {
  auto cursor = g_dbe->executeDML("select 2*d from test_chunked");
  ASSERT_NE(cursor, nullptr);
  auto table = cursor->getArrowTable();
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 1);
  ASSERT_EQ(table->num_rows(), (int64_t)6);
}

//  Tests for GROUP BY query
TEST(DBEngine, ArrowTableChunked_GROUPBY1) {
  auto cursor = g_dbe->executeDML(
      "SELECT COUNT(d),COUNT(bi),COUNT(t),i FROM test_chunked GROUP BY i;");
  ASSERT_NE(cursor, nullptr);
  auto table = cursor->getArrowTable();
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 4);
  ASSERT_EQ(table->num_rows(), (int64_t)2);

  compare_columns(std::array<int32_t, 2>{3, 3}, table->column(0));
  compare_columns(std::array<int32_t, 2>{3, 3}, table->column(1));
  compare_columns(std::array<int32_t, 2>{3, 3}, table->column(2));
  compare_columns(std::array<int32_t, 2>{0, 1}, table->column(3));
}

//  Tests for JOIN query

//  This test performs a simple JOIN test for supplied boolean
//  values of enable_columnar_output, enable_lazy_fetch
void JoinTest(bool enable_columnar_output, bool enable_lazy_fetch) {
  bool prev_enable_columnar_output = g_enable_columnar_output;
  bool prev_enable_lazy_fetch = g_enable_lazy_fetch;

  ScopeGuard reset = [prev_enable_columnar_output, prev_enable_lazy_fetch] {
    g_enable_columnar_output = prev_enable_columnar_output;
    g_enable_lazy_fetch = prev_enable_lazy_fetch;
  };

  g_enable_columnar_output = enable_columnar_output;
  g_enable_lazy_fetch = enable_lazy_fetch;

  auto cursor = g_dbe->executeDML(
      "SELECT * FROM test_chunked "
      "INNER JOIN join_table "
      "ON test_chunked.i=join_table.i;");
  ASSERT_NE(cursor, nullptr);
  auto table = cursor->getArrowTable();
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 7);
  ASSERT_EQ(table->num_rows(), (int64_t)6);

  compare_columns(table6x4_col_i32, table->column(1));
  compare_columns(table6x4_col_bi, table->column(2));
  compare_columns(table6x4_col_d, table->column(3));
  compare_columns(table6x4_col_i64, table->column(4));
  // TODO: Enable the line below after SelectBool test works OK
  // compare_columns(std::array<bool,6>{false, false, false, true, true, true},
  // table->column(5));
  compare_columns(std::array<int64_t, 6>{100, 100, 100, 200, 200, 200}, table->column(6));
}

TEST(DBEngine, ArrowTableChunked_JOIN1) {
  JoinTest(false, false);
}
TEST(DBEngine, ArrowTableChunked_JOIN2) {
  JoinTest(false, true);
}
TEST(DBEngine, ArrowTableChunked_JOIN3) {
  JoinTest(true, false);
}
TEST(DBEngine, ArrowTableChunked_JOIN4) {
  JoinTest(true, true);
}

//  Tests with NULLs
TEST(DBEngine, ArrowTableChunked_NULLS1) {
  auto cursor = g_dbe->executeDML("select * from chunked_nulls;");
  ASSERT_NE(cursor, nullptr);
  auto table = cursor->getArrowTable();
  ASSERT_NE(table, nullptr);

  ASSERT_EQ(table->num_columns(), 4);
  ASSERT_EQ(table->num_rows(), (int64_t)6);

  int32_t i32_null = null_builder<int32_t>();
  int64_t i64_null = null_builder<int64_t>();
  double f64_null = null_builder<double>();
  compare_columns(std::array<int32_t, 6>{i32_null, 0, i32_null, i32_null, 1, 1},
                  table->column(1));
  compare_columns(std::array<int64_t, 6>{i64_null, 2, 3, 4, i64_null, 6},
                  table->column(2));
  compare_columns(std::array<double, 6>{10.1, f64_null, f64_null, 40.4, 50.5, f64_null},
                  table->column(3));
}

TEST(DBEngine, ArrowTableChunked_NULLS2) {
  auto cursor = g_dbe->executeDML("select 2*i,3*bi,4*d from chunked_nulls;");
  ASSERT_NE(cursor, nullptr);
  auto table = cursor->getArrowTable();
  ASSERT_NE(table, nullptr);

  ASSERT_EQ(table->num_columns(), 3);
  ASSERT_EQ(table->num_rows(), (int64_t)6);

  int32_t i32_null = null_builder<int32_t>();
  int64_t i64_null = null_builder<int64_t>();
  double f64_null = null_builder<double>();
  compare_columns(std::array<int32_t, 6>{i32_null, 0, i32_null, i32_null, 2 * 1, 2 * 1},
                  table->column(0));
  compare_columns(std::array<int64_t, 6>{i64_null, 3 * 2, 3 * 3, 3 * 4, i64_null, 3 * 6},
                  table->column(1));
  compare_columns(
      std::array<double, 6>{4 * 10.1, f64_null, f64_null, 4 * 40.4, 4 * 50.5, f64_null},
      table->column(2));
}

int main(int argc, char* argv[]) try {
  namespace fs = std::filesystem;

  testing::InitGoogleTest(&argc, argv);

  auto options_str = parse_cli_args(argc, argv);

  auto check_file_existence = [](std::string fname) {
    if (!fs::exists(fs::path{fname})) {
      throw std::runtime_error("File not found: " + fname + ". Aborting\n");
    }
  };

  check_file_existence(TABLE6x4_CSV_FILE);
  check_file_existence(NULLSTABLE6x4_CSV_FILE);
  check_file_existence(JOIN_TABLE_CSV_FILE);

  g_dbe = EmbeddedDatabase::DBEngine::create(options_str);

  load_table6x4_csv(TABLE6x4_CSV_FILE, "test", /*fragment_size=*/0);
  load_table6x4_csv(TABLE6x4_CSV_FILE, "test_chunked", /*fragment_size=*/4);
  load_table6x4_csv(NULLSTABLE6x4_CSV_FILE, "chunked_nulls", /*fragment_size=*/3);
  import_file(JOIN_TABLE_CSV_FILE, "join_table", /*fragment_size=*/2);

  int err = RUN_ALL_TESTS();

  g_dbe.reset();
  return err;
} catch (const boost::system::system_error& e) {
  std::cout << e.what();
  LOG(ERROR) << e.what();
  return EXIT_FAILURE;
} catch (const std::exception& e) {
  std::cout << e.what();
  LOG(ERROR) << e.what();
  return EXIT_FAILURE;
}
