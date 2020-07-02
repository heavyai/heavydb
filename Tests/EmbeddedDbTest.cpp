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

#include <gtest/gtest.h>
#include "Embedded/DBEngine.h"
#include <boost/program_options.hpp>

using namespace std;
using namespace EmbeddedDatabase;

DBEngine* engine = nullptr; 

int run_simple_agg(const string& query_str) {
  auto cursor = engine->executeDML(query_str);
  auto row = cursor->getNextRow();
  return row.getInt(0);
}

std::shared_ptr<arrow::RecordBatch> run_multiple_agg(const string& query_str) {
  auto cursor = engine->executeDML(query_str);
  return cursor ? cursor->getArrowRecordBatch(): nullptr;
}

TEST(BasicDbeTest, InsertDict) {
  EXPECT_NO_THROW(engine->executeDDL("DROP TABLE IF EXISTS dist5;"));
  EXPECT_NO_THROW(engine->executeDDL(
      "create table dist5 (col1 TEXT ENCODING DICT) with (partitions='replicated');"));
  EXPECT_NO_THROW(run_multiple_agg("insert into dist5 values('t1');"));
  ASSERT_EQ(1, run_simple_agg("SELECT count(*) FROM dist5;"));
  EXPECT_NO_THROW(run_multiple_agg("insert into dist5 values('t2');"));
  ASSERT_EQ(2, run_simple_agg("SELECT count(*) FROM dist5;"));
  EXPECT_NO_THROW(run_multiple_agg("insert into dist5 values('t3');"));
  ASSERT_EQ(3, run_simple_agg("SELECT count(*) FROM dist5;"));
  EXPECT_NO_THROW(engine->executeDDL("DROP TABLE IF EXISTS dist5;"));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;

  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all tests");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()("test-help",
                     "Print all EmbeddedDbTest specific options (for gtest "
                     "options use `--help`).");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: EmbeddedDbTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  logger::init(log_options);

  engine = DBEngine::create(BASE_PATH, 5555);

  int err{0};

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  engine->reset();
  return err;
}
