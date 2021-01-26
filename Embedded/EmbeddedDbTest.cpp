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

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include "DBEngine.h"

#include <arrow/api.h>
#include <arrow/csv/reader.h>
#include <arrow/io/file.h>
#include "Shared/ArrowUtil.h"

using namespace EmbeddedDatabase;

int main(int argc, char* argv[]) {
  std::string base_path;
  int calcite_port = 5555;
  bool columnar_output = true;
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages ")(
      "data",
      po::value<std::string>(&base_path)->required(),
      "Directory path to OmniSci catalogs")(
      "calcite-port",
      po::value<int>(&calcite_port)->default_value(calcite_port),
      "Calcite port")("columnar-output",
                      po::value<bool>(&columnar_output)->default_value(columnar_output),
                      "Enable columnar_output");

  po::positional_options_description positionalOptions;
  positionalOptions.add("data", 1);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .positional(positionalOptions)
                  .run(),
              vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }

  try {
    auto opt_str = base_path + " --calcite-port " + std::to_string(calcite_port);
    if (columnar_output) {
      opt_str += "--columnar-output";
    }
    auto dbe = DBEngine::create(opt_str);
    if (dbe) {
      auto memory_pool = arrow::default_memory_pool();
      auto arrow_parse_options = arrow::csv::ParseOptions::Defaults();
      auto arrow_read_options = arrow::csv::ReadOptions::Defaults();
      auto arrow_convert_options = arrow::csv::ConvertOptions::Defaults();
      std::shared_ptr<arrow::io::ReadableFile> inp;
      auto file_result = arrow::io::ReadableFile::Open("/localdisk/artemale/test.csv");
      ARROW_THROW_NOT_OK(file_result.status());
      inp = file_result.ValueOrDie();
      auto table_reader_result = arrow::csv::TableReader::Make(memory_pool,
                                                               inp,
                                                               arrow_read_options,
                                                               arrow_parse_options,
                                                               arrow_convert_options);
      ARROW_THROW_NOT_OK(table_reader_result.status());
      auto table_reader = table_reader_result.ValueOrDie();
      std::shared_ptr<arrow::Table> arrowTable;
      auto arrow_table_result = table_reader->Read();
      ARROW_THROW_NOT_OK(arrow_table_result.status());
      arrowTable = arrow_table_result.ValueOrDie();
      dbe->importArrowTable("test", arrowTable);

      auto schema = dbe->getTableDetails("test");
      for (auto& item : schema) {
        std::cout << item.col_name << std::endl;
      }
      auto cursor = dbe->executeDML("select * from test");
      if (cursor) {
        std::cout << cursor->getRowCount() << " rows selected" << std::endl;
        std::shared_ptr<arrow::RecordBatch> rbatch = cursor->getArrowRecordBatch();
      } else {
        std::cerr << "Cursor is NULL" << std::endl;
      }
    }
  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }
  return 0;
}
