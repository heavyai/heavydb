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

#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <codecvt>
#include <locale>

#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;
using namespace TestHelpers;

bool g_keep_data{false};

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !QR::get()->gpusPresent();
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, true, true);
}

class UnicodeSpecialChars : public ::testing::Test {
 public:
  void SetUp() override {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS special_chars;");
    QR::get()->runDDLStatement(
        "CREATE TABLE special_chars (x INT, str TEXT ENCODING DICT(32)) WITH "
        "(FRAGMENT_SIZE=2);");

    run_multiple_agg("INSERT INTO special_chars VALUES (1, '" +
                         UnicodeSpecialChars::getStringValue() + "');",
                     ExecutorDeviceType::CPU);
  }

  void TearDown() override {
    if (!g_keep_data) {
      QR::get()->runDDLStatement("DROP TABLE IF EXISTS special_chars;");
    }
  }

  static std::string getStringValue() {
    return std::string("妙高杉ノ原スキー場に来ています。天気良好。");
  }
};

TEST_F(UnicodeSpecialChars, Basics) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    auto rows = run_multiple_agg("SELECT count(*) FROM special_chars WHERE str = '" +
                                     UnicodeSpecialChars::getStringValue() + "';",
                                 dt);
    ASSERT_EQ(rows->rowCount(), size_t(1));
    auto crt_row = rows->getNextRow(true, true);
    ASSERT_EQ(crt_row.size(), size_t(1));
    ASSERT_EQ(v<int64_t>(crt_row[0]), int64_t(1));
  }
}

class EscapeSeqSpecialChars : public ::testing::Test {
 public:
  void SetUp() override {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS special_chars;");
    QR::get()->runDDLStatement(
        "CREATE TABLE special_chars (x INT, str TEXT ENCODING DICT(32)) WITH "
        "(FRAGMENT_SIZE=2);");

    run_multiple_agg("INSERT INTO special_chars VALUES (1, '" +
                         EscapeSeqSpecialChars::getString1() + "');",
                     ExecutorDeviceType::CPU);
    run_multiple_agg("INSERT INTO special_chars VALUES (2, '" +
                         EscapeSeqSpecialChars::getString2() + "');",
                     ExecutorDeviceType::CPU);
  }

  void TearDown() override {
    if (!g_keep_data) {
      QR::get()->runDDLStatement("DROP TABLE IF EXISTS special_chars;");
    }
  }

  static std::string getString1() {
    return std::string("\u001e");  // information separator two
  }

  static std::string getString2() {
    return std::string("\u008D");  // reverse line feed
  }
};

TEST_F(EscapeSeqSpecialChars, Basics) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    {
      auto rows = run_multiple_agg("SELECT x FROM special_chars WHERE str = '" +
                                       EscapeSeqSpecialChars::getString1() + "';",
                                   dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(true, true);
      ASSERT_EQ(crt_row.size(), size_t(1));
      ASSERT_EQ(v<int64_t>(crt_row[0]), int64_t(1));
    }
    {
      auto rows = run_multiple_agg("SELECT x FROM special_chars WHERE str = '" +
                                       EscapeSeqSpecialChars::getString2() + "';",
                                   dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(true, true);
      ASSERT_EQ(crt_row.size(), size_t(1));
      ASSERT_EQ(v<int64_t>(crt_row[0]), int64_t(2));
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;

  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all tests");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");
  desc.add_options()("keep-data",
                     "Don't drop tables at the end of the tests. Note that individual "
                     "tests may still create and drop tables. Use in combination with "
                     "--gtest_filter to preserve tables for a specific test group.");
  desc.add_options()(
      "test-help",
      "Print all BumpAllocatorTest specific options (for gtest options use `--help`).");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: BumpAllocatorTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  if (vm.count("keep-data")) {
    g_keep_data = true;
  }

  logger::init(log_options);

  QR::init(BASE_PATH);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}
