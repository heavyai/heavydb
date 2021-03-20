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
  std::string base_path, csv_path;
  int calcite_port = 5555;
  bool columnar_output = true;
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages ")(
      "data",
      po::value<std::string>(&base_path)->required(),
      "Directory path to OmniSci catalogs")(
      "csv", po::value<std::string>(&csv_path)->required(), "Directory path to CSV file")(
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
      dbe->executeDDL(std::string(R"(
CREATE TEMPORARY TABLE test (
trip_id BIGINT,
vendor_id TEXT ENCODING NONE,
pickup_datetime TIMESTAMP,
dropoff_datetime TIMESTAMP,
store_and_fwd_flag TEXT ENCODING DICT,
rate_code_id BIGINT,
pickup_longitude DOUBLE,
pickup_latitude DOUBLE,
dropoff_longitude DOUBLE,
dropoff_latitude DOUBLE,
passenger_count BIGINT,
trip_distance DOUBLE,
fare_amount DOUBLE,
extra DOUBLE,
mta_tax DOUBLE,
tip_amount DOUBLE,
tolls_amount DOUBLE,
ehail_fee DOUBLE,
improvement_surcharge DOUBLE,
total_amount DOUBLE,
payment_type TEXT ENCODING DICT,
trip_type BIGINT,
pickup TEXT ENCODING DICT,
dropoff TEXT ENCODING NONE,
cab_type TEXT ENCODING DICT,
precipitation DOUBLE,
snow_depth BIGINT,
snowfall DOUBLE,
max_temperature BIGINT,
min_temperature BIGINT,
average_wind_speed DOUBLE,
pickup_nyct2010_gid BIGINT,
pickup_ctlabel DOUBLE,
pickup_borocode BIGINT,
pickup_boroname TEXT ENCODING NONE,
pickup_ct2010 BIGINT,
pickup_boroct2010 BIGINT,
pickup_cdeligibil TEXT ENCODING DICT,
pickup_ntacode TEXT ENCODING DICT,
pickup_ntaname TEXT ENCODING DICT,
pickup_puma BIGINT,
dropoff_nyct2010_gid BIGINT,
dropoff_ctlabel DOUBLE,
dropoff_borocode BIGINT,
dropoff_boroname TEXT ENCODING NONE,
dropoff_ct2010 BIGINT,
dropoff_boroct2010 BIGINT,
dropoff_cdeligibil TEXT ENCODING NONE,
dropoff_ntacode TEXT ENCODING NONE,
dropoff_ntaname TEXT ENCODING NONE,
dropoff_puma BIGINT) WITH (storage_type='CSV:") + csv_path + std::string("', fragment_size=100);)"));
      auto schema = dbe->getTableDetails("test");
      for (auto& item : schema) {
        std::cout << item.col_name << std::endl;
      }
      auto cursor = dbe->executeDML("select count(*) from test");
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
