/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "QueryRunner.h"

#include <boost/program_options.hpp>

int main(int argc, char** argv) {
  std::string db_path;
  std::string query;
  size_t iter;

  ExecutorDeviceType device_type{ExecutorDeviceType::GPU};

  boost::program_options::options_description desc("Options");
  desc.add_options()(
      "path", boost::program_options::value<std::string>(&db_path)->required(), "Directory path to Mapd catalogs")(
      "query", boost::program_options::value<std::string>(&query)->required(), "Query")(
      "iter", boost::program_options::value<size_t>(&iter), "Number of iterations")(
      "cpu", "Run on CPU (run on GPU by default)");

  boost::program_options::positional_options_description positionalOptions;
  positionalOptions.add("path", 1);
  positionalOptions.add("query", 1);

  boost::program_options::variables_map vm;

  try {
    boost::program_options::store(
        boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
    boost::program_options::notify(vm);
  } catch (boost::program_options::error& err) {
    LOG(ERROR) << err.what();
    return 1;
  }

  if (!vm.count("iter")) {
    iter = 100;
  }

  if (vm.count("cpu")) {
    device_type = ExecutorDeviceType::CPU;
  }

  std::unique_ptr<Catalog_Namespace::SessionInfo> session(get_session(db_path.c_str()));
  for (size_t i = 0; i < iter; ++i) {
    run_multiple_agg(query, session, device_type, true, true);
  }
  return 0;
}
