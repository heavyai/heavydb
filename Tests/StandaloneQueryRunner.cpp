/*
 * Copyright 2022 Intel Corporation.
 * Copyright 2021 OmniSci, Inc.
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
#include "QueryEngine/Execute.h"
#include "Tests/TestHelpers.h"

#include <boost/program_options.hpp>

using namespace TestHelpers;

namespace {

const size_t g_num_rows{10};
bool g_cpu_only{false};

}  // namespace

void createAndPopulateTable() {
  auto& ctx = ArrowSQLRunner::ctx();
  ArrowSQLRunner::createTable(
      "test",
      {{"x", ctx.int32(false)},
       {"w", ctx.int8()},
       {"y", ctx.int32()},
       {"z", ctx.int16()},
       {"t", ctx.int64()},
       {"b", ctx.boolean()},
       {"f", ctx.fp32()},
       {"ff", ctx.fp32()},
       {"fn", ctx.fp32()},
       {"d", ctx.fp64()},
       {"dn", ctx.fp64()},
       {"str", ctx.extDict(ctx.text(), 0)}, /* not shared here */
       {"null_str", ctx.extDict(ctx.text(), 0)},
       {"fixed_str", ctx.extDict(ctx.text(), 0, 2)},
       {"fixed_null_str", ctx.extDict(ctx.text(), 0, 2)},
       {"real_str", ctx.text()},
       {"shared_dict", ctx.extDict(ctx.text(), 0)}, /* not shared here */
       {"m", ctx.timestamp(hdk::ir::TimeUnit::kSecond)},
       {"m_3", ctx.timestamp(hdk::ir::TimeUnit::kMilli)},
       {"m_6", ctx.timestamp(hdk::ir::TimeUnit::kMicro)},
       {"m_9", ctx.timestamp(hdk::ir::TimeUnit::kNano)},
       {"n", ctx.time64(hdk::ir::TimeUnit::kSecond)},
       {"o", ctx.date32(hdk::ir::TimeUnit::kDay)},
       {"o1", ctx.date16(hdk::ir::TimeUnit::kDay)},
       {"o2", ctx.date32(hdk::ir::TimeUnit::kDay)},
       {"fx", ctx.int16()},
       {"dd", ctx.decimal64(10, 2)},
       {"dd_notnull", ctx.decimal64(10, 2, false)},
       {"ss", ctx.extDict(ctx.text(), 0)},
       {"u", ctx.int32()},
       {"ofd", ctx.int32()},
       {"ufd", ctx.int16(false)},
       {"ofq", ctx.int64()},
       {"ufq", ctx.int64(false)},
       {"smallint_nulls", ctx.int16()},
       {"bn", ctx.boolean(false)}},
      {2});
  for (size_t i = 0; i < g_num_rows; ++i) {
    ArrowSQLRunner::insertCsvValues(
        "test",
        "7,-8,42,101,1001,true,1.1,1.1,,2.2,,foo,,foo,,real_foo,foo,2014-12-13 "
        "22:23:15,2014-12-13 22:23:15.323,1999-07-11 14:02:53.874533,2006-04-26 "
        "03:49:04.607435125,15:13:14,1999-09-09,1999-09-09,1999-09-09,9,111.1,111.1,"
        "fish,,2147483647,-2147483648,,-1,32767,true");
  }
  for (size_t i = 0; i < g_num_rows / 2; ++i) {
    ArrowSQLRunner::insertCsvValues(
        "test",
        "8,-7,43,-78,1002,false,1.2,101.2,-101.2,2.4,-2002.4,bar,,bar,,real_bar,,2014-"
        "12-13 22:23:15,2014-12-13 22:23:15.323,2014-12-13 22:23:15.874533,2014-12-13 "
        "22:23:15.607435763,15:13:14,,,,,222.2,222.2,,,,-2147483647,"
        "9223372036854775807,-9223372036854775808,,false");
  }
  for (size_t i = 0; i < g_num_rows / 2; ++i) {
    ArrowSQLRunner::insertCsvValues(
        "test",
        "7,-7,43,102,1002,,1.3,1000.3,-1000.3,2.6,-220.6,baz,,,,real_baz,baz,2014-12-"
        "14 22:23:15,2014-12-14 22:23:15.750,2014-12-14 22:23:15.437321,2014-12-14 "
        "22:23:15.934567401,15:13:14,1999-09-09,1999-09-09,1999-09-09,11,333.3,333.3,"
        "boat,,1,-1,1,-9223372036854775808,1,true");
  }
}

int main(int argc, char** argv) {
  auto config = std::make_shared<Config>();

  namespace po = boost::program_options;

  po::options_description desc("Options");

  desc.add_options()("dump-ir",
                     po::value<bool>()->default_value(false)->implicit_value(true),
                     "Dump IR and PTX for all executed queries to file."
                     " Currently only supports single node tests.");
  desc.add_options()(
      "cpu-only",
      po::value<bool>(&g_cpu_only)->default_value(g_cpu_only)->implicit_value(true),
      "Force CPU only execution for all queries.");

  bool just_explain{false};
  desc.add_options()(
      "just-explain",
      po::value<bool>(&just_explain)->default_value(just_explain)->implicit_value(true),
      "Run LLVM IR explain.");

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.set_options();  // update default values
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm["dump-ir"].as<bool>()) {
    // Only log IR, PTX channels to file with no rotation size.
    log_options.channels_ = {logger::Channel::IR, logger::Channel::PTX};
    log_options.rotation_size_ = std::numeric_limits<size_t>::max();
  }

  logger::init(log_options);

  config->exec.window_func.enable = true;
  config->exec.enable_interop = false;

  ArrowSQLRunner::init(config);

  try {
    createAndPopulateTable();

    {
      const ExecutorDeviceType dt =
          g_cpu_only ? ExecutorDeviceType::CPU : ExecutorDeviceType::GPU;
      const auto device_type_str = dt == ExecutorDeviceType::CPU ? " CPU: " : " GPU: ";
      auto eo =
          ArrowSQLRunner::getExecutionOptions(/*allow_loop_joins=*/false, just_explain);
      auto co = ArrowSQLRunner::getCompilationOptions(dt);
      auto res = ArrowSQLRunner::runSqlQuery(
          R"(SELECT COUNT(*), AVG(x), SUM(y), SUM(w), AVG(t) FROM test;)", co, eo);
      if (just_explain) {
        std::cout << "Explanation for " << device_type_str << res.getExplanation()
                  << std::endl;
      } else {
        auto rows = res.getRows();
        CHECK_EQ(rows->rowCount(), size_t(1));
        auto row =
            rows->getNextRow(/*translate_strings=*/true, /*decimal_to_double=*/true);
        CHECK_EQ(row.size(), size_t(5));
        std::cout << "Result for " << device_type_str << v<int64_t>(row[0]) << " : "
                  << v<double>(row[1]) << " : " << v<int64_t>(row[2]) << " : "
                  << v<int64_t>(row[3]) << " : " << v<double>(row[4]) << std::endl;
      }
    }
    {
      const ExecutorDeviceType dt =
          g_cpu_only ? ExecutorDeviceType::CPU : ExecutorDeviceType::GPU;
      const auto device_type_str = dt == ExecutorDeviceType::CPU ? " CPU: " : " GPU: ";
      auto eo =
          ArrowSQLRunner::getExecutionOptions(/*allow_loop_joins=*/false, just_explain);
      auto co = ArrowSQLRunner::getCompilationOptions(dt);
      auto res = ArrowSQLRunner::runSqlQuery(
          R"(SELECT COUNT(*), AVG(x), SUM(y), t FROM test GROUP BY t;)", co, eo);
      if (just_explain) {
        std::cout << "Explanation for " << device_type_str << res.getExplanation()
                  << std::endl;
      } else {
        auto rows = res.getRows();
        CHECK_GE(rows->rowCount(), size_t(1));
        auto row =
            rows->getNextRow(/*translate_strings=*/true, /*decimal_to_double=*/true);
        CHECK_EQ(row.size(), size_t(4));
        std::cout << "Result for " << device_type_str << v<int64_t>(row[0]) << " : "
                  << v<double>(row[1]) << " : " << v<int64_t>(row[2]) << " : "
                  << v<int64_t>(row[3]) << std::endl;
      }
    }

  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  ArrowSQLRunner::printStats();
  ArrowSQLRunner::reset();

  return 0;
}
