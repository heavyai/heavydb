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

}  // namespace

void createAndPopulateTable() {
  ArrowSQLRunner::createTable(
      "test",
      {{"x", SQLTypeInfo(kINT, true)},
       {"w", SQLTypeInfo(kTINYINT)},
       {"y", SQLTypeInfo(kINT)},
       {"z", SQLTypeInfo(kSMALLINT)},
       {"t", SQLTypeInfo(kBIGINT)},
       {"b", SQLTypeInfo(kBOOLEAN)},
       {"f", SQLTypeInfo(kFLOAT)},
       {"ff", SQLTypeInfo(kFLOAT)},
       {"fn", SQLTypeInfo(kFLOAT)},
       {"d", SQLTypeInfo(kDOUBLE)},
       {"dn", SQLTypeInfo(kDOUBLE)},
       {"str", dictType()}, /* not shared here */
       {"null_str", dictType()},
       {"fixed_str", dictType(2)},
       {"fixed_null_str", dictType(2)},
       {"real_str", SQLTypeInfo(kTEXT)},
       {"shared_dict", dictType()}, /* not shared here */
       {"m", SQLTypeInfo(kTIMESTAMP, 0, 0)},
       {"m_3", SQLTypeInfo(kTIMESTAMP, 3, 0)},
       {"m_6", SQLTypeInfo(kTIMESTAMP, 6, 0)},
       {"m_9", SQLTypeInfo(kTIMESTAMP, 9, 0)},
       {"n", SQLTypeInfo(kTIME)},
       {"o", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 0, kNULLT)},
       {"o1", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 16, kNULLT)},
       {"o2", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 32, kNULLT)},
       {"fx", SQLTypeInfo(kSMALLINT)},
       {"dd", SQLTypeInfo(kDECIMAL, 10, 2, false)},
       {"dd_notnull", SQLTypeInfo(kDECIMAL, 10, 2, true)},
       {"ss", dictType()},
       {"u", SQLTypeInfo(kINT)},
       {"ofd", SQLTypeInfo(kINT)},
       {"ufd", SQLTypeInfo(kINT, true)},
       {"ofq", SQLTypeInfo(kBIGINT)},
       {"ufq", SQLTypeInfo(kBIGINT, true)},
       {"smallint_nulls", SQLTypeInfo(kSMALLINT)},
       {"bn", SQLTypeInfo(kBOOLEAN, true)}},
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
      const ExecutorDeviceType dt = ExecutorDeviceType::CPU;
      auto eo = ArrowSQLRunner::getExecutionOptions(/*allow_loop_joins=*/false,
                                                    /*just_explain=*/false);
      auto co = ArrowSQLRunner::getCompilationOptions(dt);
      auto res = ArrowSQLRunner::runSqlQuery(R"(SELECT COUNT(*) FROM test;)", co, eo);
      auto rows = res.getRows();
      CHECK_EQ(rows->rowCount(), size_t(1));
      auto row = rows->getNextRow(/*translate_strings=*/true, /*decimal_to_double=*/true);
      CHECK_EQ(row.size(), size_t(1));
      std::cout << "Result for CPU: " << v<int64_t>(row[0]) << std::endl;
    }
    {
      const ExecutorDeviceType dt = ExecutorDeviceType::GPU;
      auto eo = ArrowSQLRunner::getExecutionOptions(/*allow_loop_joins=*/false,
                                                    /*just_explain=*/false);
      auto co = ArrowSQLRunner::getCompilationOptions(dt);
      auto res = ArrowSQLRunner::runSqlQuery(
          R"(SELECT SAMPLE(real_str), COUNT(*) FROM test WHERE x > 7;)", co, eo);
      std::cout << "Result for GPU: " << res.getRows()->rowCount() << std::endl;
    }

  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  Executor::nukeCacheOfExecutors();

  ArrowSQLRunner::printStats();
  ArrowSQLRunner::reset();

  return 0;
}