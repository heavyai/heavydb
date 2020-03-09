/*
 * Copyright 2019 OmniSci, Inc.
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

#include <gtest/gtest.h>
#include <llvm/Support/Program.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <csignal>
#include <exception>
#include <limits>
#include <memory>
#include <vector>
#include "Catalog/Catalog.h"
#include "Catalog/DBObject.h"
#include "DataMgr/DataMgr.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/ResultSet.h"
#include "QueryEngine/UDFCompiler.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/Logger.h"
#include "Shared/MapDParameters.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace Catalog_Namespace;
using namespace TestHelpers;

using QR = QueryRunner::QueryRunner;

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

namespace {

std::shared_ptr<Calcite> g_calcite = nullptr;
std::string udf_file_name_base("../../Tests/Udf/udf_sample");

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type,
                                            const bool allow_loop_joins) {
  return QR::get()->runSQL(query_str, device_type, true, allow_loop_joins);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return run_multiple_agg(query_str, device_type, true);
}

TargetValue run_simple_agg(const std::string& query_str,
                           const ExecutorDeviceType device_type,
                           const bool allow_loop_joins = true) {
  auto rows = run_multiple_agg(query_str, device_type, allow_loop_joins);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  return crt_row[0];
}

std::string get_udf_filename() {
  return udf_file_name_base + ".cpp";
}

std::string get_udf_cpu_ir_filename() {
  return udf_file_name_base + "_cpu.bc";
}

std::string get_udf_gpu_ir_filename() {
  return udf_file_name_base + "_gpu.bc";
}

std::string get_udf_ast_filename() {
  return udf_file_name_base + ".ast";
}

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !QR::get()->gpusPresent();
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

inline void run_ddl_statement(const std::string& query) {
  QR::get()->runDDLStatement(query);
}

class SQLTestEnv : public ::testing::Environment {
 public:
  void SetUp() override {
    boost::filesystem::path udf_file((get_udf_filename()));
    if (!boost::filesystem::exists(udf_file)) {
      throw std::runtime_error("udf file: " + udf_file.string() + " does not exist");
    }

    UdfCompiler compiler(udf_file.string());
    auto compile_result = compiler.compileUdf();
    EXPECT_EQ(compile_result, 0);

    QR::init(BASE_PATH, compiler.getAstFileName());

    g_calcite = QR::get()->getCalcite();
  }

  void TearDown() override {
    boost::filesystem::path cpu_ir_file(get_udf_cpu_ir_filename());
    if (boost::filesystem::exists(cpu_ir_file)) {
      boost::filesystem::remove(cpu_ir_file);
    }

    boost::filesystem::path gpu_ir_file(get_udf_gpu_ir_filename());
    if (boost::filesystem::exists(gpu_ir_file)) {
      boost::filesystem::remove(gpu_ir_file);
    }

    boost::filesystem::path udf_ast_file(get_udf_ast_filename());
    if (boost::filesystem::exists(udf_ast_file)) {
      boost::filesystem::remove(udf_ast_file);
    }

    QR::reset();
  }
};
}  // namespace

class UDFCompilerTest : public ::testing::Test {
 protected:
  UDFCompilerTest() : udf_file_(boost::filesystem::path(get_udf_filename())) {
    if (!boost::filesystem::exists(udf_file_)) {
      throw std::runtime_error("udf file: " + udf_file_.string() + " does not exist");
    }

    setup_objects();
  }

  ~UDFCompilerTest() override { remove_objects(); }

  void SetUp() override {}

  void TearDown() override {}

  void setup_objects() {}

  void remove_objects() {}

  std::string getUdfFileName() const { return udf_file_.string(); }

 private:
  boost::filesystem::path udf_file_;
};

TEST_F(UDFCompilerTest, CompileTest) {
  UdfCompiler compiler(getUdfFileName());
  auto compile_result = compiler.compileUdf();

  EXPECT_EQ(compile_result, 0);
  // TODO cannot test invalid file path because the compileUdf function uses
  // LOG(FATAL) which stops the process and does not return
}

TEST_F(UDFCompilerTest, CompilerPathTest) {
  UdfCompiler compiler(getUdfFileName(), llvm::sys::findProgramByName("clang++").get());
  auto compile_result = compiler.compileUdf();

  EXPECT_EQ(compile_result, 0);
  // TODO cannot test invalid file path because the compileUdf function uses
  // LOG(FATAL) which stops the process and does not return
}

TEST_F(UDFCompilerTest, CalciteRegistration) {
  UdfCompiler compiler(getUdfFileName());
  auto compile_result = compiler.compileUdf();

  ASSERT_EQ(compile_result, 0);

  ASSERT_TRUE(g_calcite != nullptr);

  auto signature = ExtensionFunctionsWhitelist::get_udf("udf_truerange");
  ASSERT_NE(signature, nullptr);

  auto signature2 = ExtensionFunctionsWhitelist::get_udf("udf_truehigh");
  ASSERT_NE(signature2, nullptr);

  auto signature3 = ExtensionFunctionsWhitelist::get_udf("udf_truelow");
  ASSERT_NE(signature3, nullptr);

  auto signature4 = ExtensionFunctionsWhitelist::get_udf("udf_range");
  ASSERT_NE(signature4, nullptr);

  auto signature5 = ExtensionFunctionsWhitelist::get_udf("udf_range_int");
  ASSERT_NE(signature5, nullptr);

  auto signature6 = ExtensionFunctionsWhitelist::get_udf("udf_range_integer");
  ASSERT_EQ(signature6, nullptr);
}

TEST_F(UDFCompilerTest, UdfQuery) {
  UdfCompiler compiler(getUdfFileName());
  auto compile_result = compiler.compileUdf();

  ASSERT_EQ(compile_result, 0);

  run_ddl_statement("DROP TABLE IF EXISTS stocks;");
  run_ddl_statement("DROP TABLE IF EXISTS sal_emp;");
  run_ddl_statement("DROP TABLE IF EXISTS geospatial_test;");
  run_ddl_statement("DROP TABLE IF EXISTS geospatial_linestring;");
  run_ddl_statement("DROP TABLE IF EXISTS geo_poly;");
  run_ddl_statement("DROP TABLE IF EXISTS geo_mpoly;");

  run_ddl_statement(
      "CREATE TABLE stocks(symbol text, open_p int, high_p int, "
      "low_p int, close_p int, entry_d DATE);");
  run_ddl_statement(
      "CREATE TABLE geospatial_test (id INT, p POINT, "
      "gp4326 GEOMETRY(POINT,4326) ENCODING COMPRESSED(32), "
      "gp4326none GEOMETRY(POINT,4326) ENCODING NONE) ;");

  run_ddl_statement("CREATE TABLE sal_emp(name text, pay_by_quarter integer[]);");

  run_ddl_statement("CREATE TABLE geospatial_linestring (id INT, l LINESTRING)");
  run_ddl_statement("CREATE TABLE geo_poly (id INT, p POLYGON);");
  run_ddl_statement("CREATE TABLE geo_mpoly (id INT, p MULTIPOLYGON);");

  std::string insert1(
      "INSERT into stocks VALUES ('NVDA', '178', '178', '171', '173', '2019-05-07');");
  EXPECT_NO_THROW(run_multiple_agg(insert1, ExecutorDeviceType::CPU));

  std::string insert2(
      "INSERT into stocks VALUES ('NVDA', '175', '181', '174', '178', '2019-05-06');");
  EXPECT_NO_THROW(run_multiple_agg(insert2, ExecutorDeviceType::CPU));

  std::string insert3(
      "INSERT into stocks VALUES ('NVDA', '183', '184', '181', '183', '2019-05-03');");
  EXPECT_NO_THROW(run_multiple_agg(insert3, ExecutorDeviceType::CPU));

  std::string array_insert1(
      "INSERT into sal_emp VALUES ('Sarah', ARRAY[5000, 6000, 7000, 8000]);");
  EXPECT_NO_THROW(run_multiple_agg(array_insert1, ExecutorDeviceType::CPU));

  std::string array_insert2(
      "INSERT into sal_emp VALUES ('John', ARRAY[3000, 3500, 4000, 4300]);");

  EXPECT_NO_THROW(run_multiple_agg(array_insert2, ExecutorDeviceType::CPU));

  std::string array_insert3("INSERT into sal_emp VALUES ('Jim', NULL);");
  EXPECT_NO_THROW(run_multiple_agg(array_insert3, ExecutorDeviceType::CPU));

  std::string array_insert4(
      "INSERT into sal_emp VALUES ('Carla', ARRAY[7000, NULL, NULL, 9000]);");

  EXPECT_NO_THROW(run_multiple_agg(array_insert4, ExecutorDeviceType::CPU));

  std::string point_insert1(
      "INSERT into geospatial_test VALUES(0, 'POINT(55.8659449685365 "
      "-4.25072511658072)', "
      "'POINT(51.4618933852762 -0.926690306514502)', "
      "'POINT(55.9523783996701 -3.20510306395594326)');");
  EXPECT_NO_THROW(run_multiple_agg(point_insert1, ExecutorDeviceType::CPU));
  std::string linestring_insert1(
      "INSERT into geospatial_linestring VALUES(0, 'LINESTRING(1 0, 2 3, 3 4)');");
  std::string linestring_insert2(
      "INSERT into geospatial_linestring VALUES(1, 'LINESTRING(1 0, 0 1, -1 0, 0 -1, 1 "
      "0)');");

  EXPECT_NO_THROW(run_multiple_agg(linestring_insert1, ExecutorDeviceType::CPU));
  EXPECT_NO_THROW(run_multiple_agg(linestring_insert2, ExecutorDeviceType::CPU));

  std::string polygon_insert1(
      "INSERT into geo_poly VALUES(0, 'POLYGON((1 0, "
      "0 1, -1 0, 0 -1, 1 0), (0.1 0, 0 0.1, -0.1 0, 0 -0.1, 0.1 0))');");
  EXPECT_NO_THROW(run_multiple_agg(polygon_insert1, ExecutorDeviceType::CPU));

  std::string multipolygon_insert1(
      "INSERT into geo_mpoly VALUES(0, 'MULTIPOLYGON(((1 0, 0 1, -1 0, 0 -1, 1 0), "
      "(0.1 0, 0 0.1, -0.1 0, 0 -0.1, 0.1 0)), ((2 0, 0 2, -2 0, 0 -2, 2 0), "
      "(0.2 0, 0 0.2, -0.2 0, 0 -0.2, 0.2 0)))');");
  EXPECT_NO_THROW(run_multiple_agg(multipolygon_insert1, ExecutorDeviceType::CPU));

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(7,
              v<int64_t>(run_simple_agg("SELECT udf_range_int(high_p, low_p) from stocks "
                                        "where entry_d = '2019-05-06';",
                                        dt)));
    ASSERT_EQ(3,
              v<int64_t>(run_simple_agg("SELECT udf_range_int(high_p, low_p) from stocks "
                                        "where entry_d = '2019-05-03';",
                                        dt)));
    ASSERT_EQ(
        5000,
        v<int64_t>(run_simple_agg("select array_at_int32(pay_by_quarter, 0) from sal_emp "
                                  "where name = 'Sarah';",
                                  dt)));

    ASSERT_EQ(
        4300,
        v<int64_t>(run_simple_agg("select array_at_int32(pay_by_quarter, 3) from sal_emp "
                                  "where name = 'John';",
                                  dt)));

    ASSERT_EQ(
        4,
        v<int64_t>(run_simple_agg("select array_sz_int32(pay_by_quarter) from sal_emp "
                                  "where name = 'John';",
                                  dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "select array_is_null_int32(pay_by_quarter) from sal_emp "
                  "where name = 'Jim';",
                  dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "select array_is_null_int32(pay_by_quarter) from sal_emp "
                  "where name = 'John';",
                  dt)));

    ASSERT_EQ(
        std::numeric_limits<int32_t>::min(),
        v<int64_t>(run_simple_agg("select array_at_int32(pay_by_quarter, 1) from sal_emp "
                                  "where name = 'Carla';",
                                  dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "select array_at_int32_is_null(pay_by_quarter, 0) from sal_emp "
                  "where name = 'Carla';",
                  dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "select array_at_int32_is_null(pay_by_quarter, 1) from sal_emp "
                  "where name = 'Carla';",
                  dt)));

    ASSERT_DOUBLE_EQ(55.8659449685365,
                     v<double>(run_simple_agg(
                         "select point_x(p) from geospatial_test WHERE id = 0;", dt)));
    ASSERT_DOUBLE_EQ(-4.25072511658072,
                     v<double>(run_simple_agg(
                         "select point_y(p) from geospatial_test WHERE id = 0;", dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "select point_compression(p) from geospatial_test WHERE id = 0;", dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "select point_input_srid(p) from geospatial_test WHERE id = 0;", dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "select point_output_srid(p) from geospatial_test WHERE id = 0;", dt)));

    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "select point_compression(gp4326) from geospatial_test WHERE id = 0;", dt)));

    ASSERT_EQ(
        4326,
        v<int64_t>(run_simple_agg(
            "select point_input_srid(gp4326) from geospatial_test WHERE id = 0;", dt)));

    ASSERT_EQ(
        4326,
        v<int64_t>(run_simple_agg(
            "select point_output_srid(gp4326) from geospatial_test WHERE id = 0;", dt)));

    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "select point_compression(gp4326none) from geospatial_test WHERE id = 0;",
            dt)));

    ASSERT_DOUBLE_EQ(
        1.0,
        v<double>(run_simple_agg(
            "select linestring_x(l, 1) from geospatial_linestring WHERE id = 0;", dt)));
    ASSERT_DOUBLE_EQ(
        2.0,
        v<double>(run_simple_agg(
            "select linestring_x(l, 2) from geospatial_linestring WHERE id = 0;", dt)));

    ASSERT_DOUBLE_EQ(
        3.0,
        v<double>(run_simple_agg(
            "select linestring_x(l, 3) from geospatial_linestring WHERE id = 0;", dt)));
    ASSERT_DOUBLE_EQ(
        0.0,
        v<double>(run_simple_agg(
            "select linestring_y(l, 1) from geospatial_linestring WHERE id = 0;", dt)));
    ASSERT_DOUBLE_EQ(
        3.0,
        v<double>(run_simple_agg(
            "select linestring_y(l, 2) from geospatial_linestring WHERE id = 0;", dt)));

    ASSERT_DOUBLE_EQ(
        4.0,
        v<double>(run_simple_agg(
            "select linestring_y(l, 3) from geospatial_linestring WHERE id = 0;", dt)));

    ASSERT_DOUBLE_EQ(
        5.656854249492381,
        v<double>(run_simple_agg(
            "select linestring_length(l) from geospatial_linestring WHERE id = 1;", dt)));

    ASSERT_DOUBLE_EQ(1.98,
                     v<double>(run_simple_agg(
                         "select polygon_area(p) from geo_poly WHERE id = 0;", dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "select polygon_compression(p) from geo_poly WHERE id = 0;", dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "select polygon_input_srid(p) from geo_poly WHERE id = 0;", dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "select polygon_output_srid(p) from geo_poly WHERE id = 0;", dt)));

    ASSERT_NEAR(static_cast<double>(2.0 - 0.02 + 8.0 - 0.08),
                v<double>(run_simple_agg(
                    "select multipolygon_area(p) from geo_mpoly WHERE id = 0;", dt)),
                static_cast<double>(0.0001));

    ASSERT_NEAR(static_cast<double>(4 * 1.41421 + 4 * 2.82842),
                v<double>(run_simple_agg(
                    "select multipolygon_perimeter(p) from geo_mpoly WHERE id = 0;", dt)),
                static_cast<double>(0.0001));

    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "select multipolygon_compression(p) from geo_mpoly WHERE id = 0;", dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "select multipolygon_input_srid(p) from geo_mpoly WHERE id = 0;", dt)));

    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "select multipolygon_output_srid(p) from geo_mpoly WHERE id = 0;", dt)));

    {
      auto check_row_result = [](const auto& crt_row, const auto& expected) {
        compare_array(crt_row[0], expected);
      };

      const auto rows = run_multiple_agg(
          "SELECT array_ret_udf(pay_by_quarter, CAST(1.2 AS DOUBLE)) FROM sal_emp;", dt);
      ASSERT_EQ(rows->rowCount(), size_t(4));
      check_row_result(rows->getNextRow(false, false),
                       std::vector<double>{6000, 7200, 8400, 9600});
      check_row_result(rows->getNextRow(false, false),
                       std::vector<double>{3600, 4200, 4800, 5160});
      check_row_result(rows->getNextRow(false, false), std::vector<double>{});
      check_row_result(rows->getNextRow(false, false),
                       std::vector<double>{8400,
                                           inline_fp_null_value<double>(),
                                           inline_fp_null_value<double>(),
                                           10800});
    }
  }

  EXPECT_THROW(run_simple_agg("SELECT udf_range_integer(high_p, low_p) from stocks where "
                              "entry_d = '2019-05-06';",
                              ExecutorDeviceType::CPU),
               std::exception);

  run_ddl_statement("DROP TABLE stocks;");
  run_ddl_statement("DROP TABLE sal_emp;");
  run_ddl_statement("DROP TABLE geospatial_test;");
  run_ddl_statement("DROP TABLE geospatial_linestring;");
  run_ddl_statement("DROP TABLE geo_poly;");
  run_ddl_statement("DROP TABLE geo_mpoly;");
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SQLTestEnv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
