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

#include "ArrowSQLRunner/ArrowSQLRunner.h"

#include "QueryEngine/ArrowResultSet.h"
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

// Input files' names
static const char* TABLE6x4_CSV_FILE =
    "../../Tests/EmbeddedDataFiles/embedded_db_test_6x4table.csv";
static const char* JOIN_TABLE_CSV_FILE =
    "../../Tests/EmbeddedDataFiles/embedded_db_test_join_table.csv";
static const char* NULLSTABLE6x4_CSV_FILE =
    "../../Tests/EmbeddedDataFiles/embedded_db_test_nulls_table.csv";

//  Content of the table stored in $TABLE6x4_CSV_FILE file
static std::array<int64_t, 6> table6x4_col_i64 = {0, 0, 0, 1, 1, 1};
static std::array<int64_t, 6> table6x4_col_bi = {1, 2, 3, 4, 5, 6};
static std::array<double, 6> table6x4_col_d = {10.1, 20.2, 30.3, 40.4, 50.5, 60.6};

using namespace TestHelpers::ArrowSQLRunner;

//  HELPERS
namespace {

// Processes command line args
void parse_cli_args(int argc, char* argv[]) {
  namespace po = boost::program_options;

  po::options_description desc("Options");

  desc.add_options()("help,h", "Print help messages ");
  desc.add_options()(
      "enable-columnar-output",
      po::value<bool>(&g_enable_columnar_output)->default_value(g_enable_columnar_output),
      "Enable columnar_output");

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.set_options();  // update default values
  desc.add(log_options.get_options());

  try {
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc;
      std::exit(EXIT_SUCCESS);
    }

    logger::init(log_options);
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    std::cout << desc;
    std::exit(EXIT_FAILURE);
  }
}

void import_data() {
  getStorage()->importCsvFile(TABLE6x4_CSV_FILE, "test");
  getStorage()->importCsvFile(
      TABLE6x4_CSV_FILE, "test_chunked", ArrowStorage::TableOptions{4});
  getStorage()->importCsvFile(
      NULLSTABLE6x4_CSV_FILE, "chunked_nulls", ArrowStorage::TableOptions{3});
  getStorage()->importCsvFile(
      JOIN_TABLE_CSV_FILE, "join_table", ArrowStorage::TableOptions{2});
}

std::shared_ptr<arrow::RecordBatch> getArrowRecordBatch(const ExecutionResult& res) {
  std::vector<std::string> col_names;
  for (auto& target : res.getTargetsMeta()) {
    col_names.push_back(target.get_resname());
  }
  auto converter =
      std::make_unique<ArrowResultSetConverter>(res.getDataPtr(), col_names, -1);
  return converter->convertToArrow();
}

std::shared_ptr<arrow::Table> getArrowTable(const ExecutionResult& res) {
  std::vector<std::string> col_names;
  for (auto& target : res.getTargetsMeta()) {
    col_names.push_back(target.get_resname());
  }
  auto converter =
      std::make_unique<ArrowResultSetConverter>(res.getDataPtr(), col_names, -1);
  return converter->convertToArrowTable();
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
}

template <typename TYPE>
void compare_columns(const std::vector<TYPE>& expected,
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
}

//  Testing helper functions for 6x4 table contained in $TABLE6x4_CSV_FILE file.
//  NOTE: load_table6x4_csv() creates table with Arrow types <dictionary, int32, int64,
//  float64> so we check that.
void test_arrow_table_conversion_table6x4(size_t fragment_size) {
  auto res = runSqlQuery("select * from test_chunked;", ExecutorDeviceType::CPU, true);
  auto table = getArrowTable(res);

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
  ASSERT_EQ(schema->GetFieldByName("i")->type()->Equals(arrow::int64()), true);
  ASSERT_EQ(schema->GetFieldByName("bi")->type()->Equals(arrow::int64()), true);
  ASSERT_EQ(schema->GetFieldByName("d")->type()->Equals(arrow::float64()), true);

  compare_columns(table6x4_col_i64, table->column(1));
  compare_columns(table6x4_col_bi, table->column(2));
  compare_columns(table6x4_col_d, table->column(3));
}

//  Performs tests for 6x4 table contained in $TABLE6x4_CSV_FILE for specified
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
  test_arrow_table_conversion_table6x4(fragment_size);
}

template <typename T>
struct ConversionTraits {};

template <>
struct ConversionTraits<int8_t> {
  using arrow_type = arrow::Int8Type;
  static std::shared_ptr<arrow::DataType> getSchemaValue() { return arrow::int8(); }
};

template <>
struct ConversionTraits<uint8_t> {
  using arrow_type = arrow::UInt8Type;
  static std::shared_ptr<arrow::DataType> getSchemaValue() { return arrow::uint8(); }
};

template <>
struct ConversionTraits<int16_t> {
  using arrow_type = arrow::Int16Type;
  static std::shared_ptr<arrow::DataType> getSchemaValue() { return arrow::int16(); }
};

template <>
struct ConversionTraits<uint16_t> {
  using arrow_type = arrow::UInt16Type;
  static std::shared_ptr<arrow::DataType> getSchemaValue() { return arrow::uint16(); }
};

template <>
struct ConversionTraits<int32_t> {
  using arrow_type = arrow::Int32Type;
  static std::shared_ptr<arrow::DataType> getSchemaValue() { return arrow::int32(); }
};

template <>
struct ConversionTraits<uint32_t> {
  using arrow_type = arrow::UInt32Type;
  static std::shared_ptr<arrow::DataType> getSchemaValue() { return arrow::uint32(); }
};

template <>
struct ConversionTraits<int64_t> {
  using arrow_type = arrow::Int64Type;
  static std::shared_ptr<arrow::DataType> getSchemaValue() { return arrow::int64(); }
};

template <>
struct ConversionTraits<uint64_t> {
  using arrow_type = arrow::UInt64Type;
  static std::shared_ptr<arrow::DataType> getSchemaValue() { return arrow::uint64(); }
};

template <>
struct ConversionTraits<float> {
  using arrow_type = arrow::FloatType;
  static std::shared_ptr<arrow::DataType> getSchemaValue() { return arrow::float32(); }
};

template <>
struct ConversionTraits<double> {
  using arrow_type = arrow::DoubleType;
  static std::shared_ptr<arrow::DataType> getSchemaValue() { return arrow::float64(); }
};

template <typename TYPE>
void build_table(const std::vector<TYPE>& values,
                 size_t fragments_count = 1,
                 std::string table_name = "table1") {
  size_t fragment_size =
      fragments_count > 1
          ? size_t((values.size() + fragments_count - 1) / fragments_count)
          : 32000000;

  {
    std::shared_ptr<arrow::Array> array;
    arrow::NumericBuilder<typename ConversionTraits<TYPE>::arrow_type> builder;
    builder.Resize(values.size());
    builder.AppendValues(values);
    builder.Finish(&array);

    auto schema =
        arrow::schema({arrow::field("value", ConversionTraits<TYPE>::getSchemaValue())});
    std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {array});
    getStorage()->importArrowTable(
        table, table_name, ArrowStorage::TableOptions{fragment_size});
  }
}

template <typename TYPE>
void test_single_column_table(size_t N, size_t fragments_count) {
  std::vector<TYPE> values(N, 0);

  for (size_t i = 0; i < values.size(); i++) {
    values[i] = (i % 2) == 0 ? 0 : null_builder<TYPE>();
  }

  build_table<TYPE>(values, fragments_count, "LargeTable");

  auto res = runSqlQuery("SELECT * FROM LargeTable;", ExecutorDeviceType::CPU, true);
  auto table = getArrowTable(res);
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 1);
  ASSERT_EQ(table->num_rows(), (int64_t)values.size());
  compare_columns(values, table->column(0));
  getStorage()->dropTable("LargeTable");
}

}  // anonymous namespace

//  Tests getArrowRecordBatch() for all columns selection
TEST(ArrowRecordBatch, SimpleSelectAll) {
  auto res = runSqlQuery("select * from test;", ExecutorDeviceType::CPU, true);
  auto rbatch = getArrowRecordBatch(res);
  ASSERT_NE(rbatch, nullptr);
  ASSERT_EQ(rbatch->num_columns(), 4);
  ASSERT_EQ(rbatch->num_rows(), (int64_t)6);
}

//  Tests getArrowRecordBatch() for two columns (TEXT "t", INT "i") selection
TEST(ArrowRecordBatch, TextIntSelect) {
  auto res = runSqlQuery("select t, i from test;", ExecutorDeviceType::CPU, true);
  auto rbatch = getArrowRecordBatch(res);
  ASSERT_NE(rbatch, nullptr);
  ASSERT_EQ(rbatch->num_columns(), 2);
  ASSERT_EQ(rbatch->num_rows(), (int64_t)6);
}

//  Tests getArrowTable() for three columns (TEXT "t", INT "i", BIGINT "bi") selection
TEST(ArrowTable, TextIntBigintSelect) {
  auto res = runSqlQuery("select t, i, bi from test;", ExecutorDeviceType::CPU, true);
  auto table = getArrowTable(res);
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 3);
  ASSERT_EQ(table->num_rows(), (int64_t)6);
}

//  Two column selection with filtering. The result is empty table.
TEST(ArrowTable, EmptySelection) {
  auto res =
      runSqlQuery("select i, d from test where d > 1000;", ExecutorDeviceType::CPU, true);
  auto table = getArrowTable(res);
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 2);
  ASSERT_EQ(table->num_rows(), (int64_t)0);
}

//  Chunked Arrow Table Conversion Tests
TEST(ArrowTable, Chunked_Conversion1) {
  test_chunked_conversion(/*columnar_output=*/false, /*lazy_fetch=*/false, 4);
}

TEST(ArrowTable, Chunked_Conversion2) {
  test_chunked_conversion(/*columnar_output=*/true, /*lazy_fetch=*/false, 4);
}

TEST(ArrowTable, Chunked_Conversion3) {
  test_chunked_conversion(/*columnar_output=*/false, /*lazy_fetch=*/true, 4);
}

TEST(ArrowTable, Chunked_Conversion4) {
  test_chunked_conversion(/*columnar_output=*/true, /*lazy_fetch=*/true, 4);
}

//  Projection operation test (column "d")
TEST(ArrowTable, Chunked_SingleColumnConversion) {
  auto res = runSqlQuery("select 2*d from test_chunked;", ExecutorDeviceType::CPU, true);
  auto table = getArrowTable(res);
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 1);
  ASSERT_EQ(table->num_rows(), (int64_t)6);
}

//  Tests for GROUP BY query
TEST(ArrowTable, Chunked_GROUPBY1) {
  auto res =
      runSqlQuery("SELECT COUNT(d),COUNT(bi),COUNT(t),i FROM test_chunked GROUP BY i;",
                  ExecutorDeviceType::CPU,
                  true);
  auto table = getArrowTable(res);
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 4);
  ASSERT_EQ(table->num_rows(), (int64_t)2);

  compare_columns(std::array<int32_t, 2>{3, 3}, table->column(0));
  compare_columns(std::array<int32_t, 2>{3, 3}, table->column(1));
  compare_columns(std::array<int32_t, 2>{3, 3}, table->column(2));
  compare_columns(std::array<int64_t, 2>{0, 1}, table->column(3));
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

  auto res = runSqlQuery(
      "SELECT * FROM test_chunked INNER JOIN join_table ON test_chunked.i=join_table.i;",
      ExecutorDeviceType::CPU,
      true);
  auto table = getArrowTable(res);
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_columns(), 7);
  ASSERT_EQ(table->num_rows(), (int64_t)6);

  compare_columns(table6x4_col_i64, table->column(1));
  compare_columns(table6x4_col_bi, table->column(2));
  compare_columns(table6x4_col_d, table->column(3));
  compare_columns(table6x4_col_i64, table->column(4));
  // TODO: Enable the line below after SelectBool test works OK
  // compare_columns(std::array<bool,6>{false, false, false, true, true, true},
  // table->column(5));
  compare_columns(std::array<int64_t, 6>{100, 100, 100, 200, 200, 200}, table->column(6));
}

TEST(ArrowTable, Chunked_JOIN1) {
  JoinTest(false, false);
}
TEST(ArrowTable, Chunked_JOIN2) {
  JoinTest(false, true);
}
TEST(ArrowTable, Chunked_JOIN3) {
  JoinTest(true, false);
}
TEST(ArrowTable, Chunked_JOIN4) {
  JoinTest(true, true);
}

//  Tests with NULLs
TEST(ArrowTable, Chunked_NULLS1) {
  auto res = runSqlQuery("select * from chunked_nulls;", ExecutorDeviceType::CPU, true);
  auto table = getArrowTable(res);
  ASSERT_NE(table, nullptr);

  ASSERT_EQ(table->num_columns(), 4);
  ASSERT_EQ(table->num_rows(), (int64_t)6);

  int64_t i64_null = null_builder<int64_t>();
  double f64_null = null_builder<double>();
  compare_columns(std::array<int64_t, 6>{i64_null, 0, i64_null, i64_null, 1, 1},
                  table->column(1));
  compare_columns(std::array<int64_t, 6>{i64_null, 2, 3, 4, i64_null, 6},
                  table->column(2));
  compare_columns(std::array<double, 6>{10.1, f64_null, f64_null, 40.4, 50.5, f64_null},
                  table->column(3));
}

TEST(ArrowTable, Chunked_NULLS2) {
  auto res = runSqlQuery(
      "select 2*i,3*bi,4*d from chunked_nulls;", ExecutorDeviceType::CPU, true);
  auto table = getArrowTable(res);
  ASSERT_NE(table, nullptr);

  ASSERT_EQ(table->num_columns(), 3);
  ASSERT_EQ(table->num_rows(), (int64_t)6);

  int64_t i64_null = null_builder<int64_t>();
  double f64_null = null_builder<double>();
  compare_columns(std::array<int64_t, 6>{i64_null, 0, i64_null, i64_null, 2 * 1, 2 * 1},
                  table->column(0));
  compare_columns(std::array<int64_t, 6>{i64_null, 3 * 2, 3 * 3, 3 * 4, i64_null, 3 * 6},
                  table->column(1));
  compare_columns(
      std::array<double, 6>{4 * 10.1, f64_null, f64_null, 4 * 40.4, 4 * 50.5, f64_null},
      table->column(2));
}

//  Tests for large tables
TEST(ArrowTable, LargeTables) {
  bool prev_enable_columnar_output = g_enable_columnar_output;
  bool prev_enable_lazy_fetch = g_enable_lazy_fetch;

  ScopeGuard reset = [prev_enable_columnar_output, prev_enable_lazy_fetch] {
    g_enable_columnar_output = prev_enable_columnar_output;
    g_enable_lazy_fetch = prev_enable_lazy_fetch;
  };

  g_enable_columnar_output = true;
  g_enable_lazy_fetch = false;
  const size_t N = 500'000;
  test_single_column_table<int8_t>(N, 150);
  test_single_column_table<int16_t>(N, 150);
  test_single_column_table<int32_t>(N, 150);
  test_single_column_table<int64_t>(N, 150);
  test_single_column_table<float>(N, 150);
  test_single_column_table<double>(N, 150);
}

TEST(ArrowTable, LargeTablesRowWise) {
  bool prev_enable_columnar_output = g_enable_columnar_output;
  bool prev_enable_lazy_fetch = g_enable_lazy_fetch;

  ScopeGuard reset = [prev_enable_columnar_output, prev_enable_lazy_fetch] {
    g_enable_columnar_output = prev_enable_columnar_output;
    g_enable_lazy_fetch = prev_enable_lazy_fetch;
  };

  g_enable_columnar_output = false;
  g_enable_lazy_fetch = false;
  const size_t N = 500'000;
  test_single_column_table<int8_t>(N, 150);
  test_single_column_table<int16_t>(N, 150);
  test_single_column_table<int32_t>(N, 150);
  test_single_column_table<int64_t>(N, 150);
  test_single_column_table<float>(N, 150);
  test_single_column_table<double>(N, 150);
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    parse_cli_args(argc, argv);

    init();

    import_data();

    err = RUN_ALL_TESTS();

    reset();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  return err;
}
