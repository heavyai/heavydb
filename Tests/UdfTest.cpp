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

#include "ArrowSQLRunner/ArrowSQLRunner.h"
#include "TestHelpers.h"

#include "DataMgr/DataMgr.h"
#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/ResultSet.h"
#include "UdfCompiler/UdfCompiler.h"

#include <gtest/gtest.h>
#include <llvm/Support/Program.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>

#include <csignal>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

using namespace TestHelpers;
using namespace TestHelpers::ArrowSQLRunner;

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

namespace {

std::string udf_file_name_base("../../Tests/Udf/udf_sample");

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
  return device_type == ExecutorDeviceType::GPU && !gpusPresent();
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

CudaMgr_Namespace::NvidiaDeviceArch init_nvidia_device_arch() {
#ifdef HAVE_CUDA
  auto cuda_mgr = std::make_unique<CudaMgr_Namespace::CudaMgr>(/*num_gpus=*/0);
  CHECK(cuda_mgr);
  return cuda_mgr->getDeviceArch();
#else
  return CudaMgr_Namespace::NvidiaDeviceArch::Kepler;
#endif
}

CudaMgr_Namespace::NvidiaDeviceArch g_device_arch = init_nvidia_device_arch();

class SQLTestEnv : public ::testing::Environment {
 public:
  void SetUp() override {
    boost::filesystem::path udf_file((get_udf_filename()));
    if (!boost::filesystem::exists(udf_file)) {
      throw std::runtime_error("udf file: " + udf_file.string() + " does not exist");
    }

    std::vector<std::string> udf_compiler_options{std::string("-D UDF_COMPILER_OPTION")};
    UdfCompiler compiler(g_device_arch, std::string(""), udf_compiler_options);
    auto compile_result = compiler.compileUdf(udf_file.string());
    Executor::addUdfIrToModule(compile_result.first, /*is_cuda_ir=*/false);
    if (!compile_result.second.empty()) {
      Executor::addUdfIrToModule(compile_result.second, /*is_cuda_ir=*/true);
    }

    init(0, compiler.getAstFileName(udf_file.string()));
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

    reset();
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
  UdfCompiler compiler(g_device_arch);
  auto [cpu_ir_file, cuda_ir_file] = compiler.compileUdf(getUdfFileName());

  EXPECT_TRUE(!cpu_ir_file.empty());
  if (gpusPresent()) {
    if (cuda_ir_file.empty()) {
      LOG(ERROR) << "Failed to compile UDF for CUDA. Skipping test due to Clang 9 / Cuda "
                    "11 dependency issues.";
    }
    // TODO: re-enable after upgrading llvm/clang in main deps
    // EXPECT_TRUE(!cuda_ir_file.empty());
  } else {
    EXPECT_TRUE(cuda_ir_file.empty());
  }
}

TEST_F(UDFCompilerTest, InvalidPath) {
  UdfCompiler compiler(g_device_arch);
  EXPECT_ANY_THROW(compiler.compileUdf(getUdfFileName() + ".invalid"));
}

TEST_F(UDFCompilerTest, CompilerOptionTest) {
  UdfCompiler compiler(g_device_arch);
  EXPECT_NO_THROW(compiler.compileUdf(getUdfFileName()));

  // This function signature is only visible via the -DUDF_COMPILER_OPTION
  // definition. This definition was passed to the UdfCompiler is Setup.
  // We had to do it there because Calcite only reads the ast definitions once
  // at startup

  auto signature = ExtensionFunctionsWhitelist::get_udf("udf_range_int2");
  ASSERT_NE(signature, nullptr);
}

TEST_F(UDFCompilerTest, CompilerPathTest) {
  UdfCompiler compiler(g_device_arch, llvm::sys::findProgramByName("clang++").get());
  EXPECT_NO_THROW(compiler.compileUdf(getUdfFileName()));
}

TEST_F(UDFCompilerTest, BadClangPath) {
  UdfCompiler compiler(g_device_arch, /*clang_path_override=*/get_udf_filename());
  EXPECT_ANY_THROW(compiler.compileUdf(getUdfFileName()));
}

TEST_F(UDFCompilerTest, CalciteRegistration) {
  UdfCompiler compiler(g_device_arch);
  EXPECT_NO_THROW(compiler.compileUdf(getUdfFileName()));

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
  UdfCompiler compiler(g_device_arch);
  EXPECT_NO_THROW(compiler.compileUdf(getUdfFileName()));

  createTable("stocks",
              {{"symbol", dictType()},
               {"open_p", SQLTypeInfo(kINT)},
               {"high_p", SQLTypeInfo(kINT)},
               {"low_p", SQLTypeInfo(kINT)},
               {"close_p", SQLTypeInfo(kINT)},
               {"entry_d", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 0, kNULLT)}});
  createTable("sal_emp", {{"name", dictType()}, {"pay_by_quarter", arrayType(kINT)}});

  insertCsvValues("stocks", "NVDA,178,178,171,173,2019-05-07");
  insertCsvValues("stocks", "NVDA,175,181,174,178,2019-05-06");
  insertCsvValues("stocks", "NVDA,183,184,181,183,2019-05-03");

  insertJsonValues(
      "sal_emp",
      "{\"name\": \"Sarah\", \"pay_by_quarter\": [5000, 6000, 7000, 8000]}\n"
      "{\"name\": \"John\", \"pay_by_quarter\": [3000, 3500, 4000, 4300]}\n"
      "{\"name\": \"Jim\", \"pay_by_quarter\": null}\n"
      "{\"name\": \"Carla\", \"pay_by_quarter\": [7000, null, null, 9000]}\n");

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

  dropTable("stocks");
  dropTable("sal_emp");
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
