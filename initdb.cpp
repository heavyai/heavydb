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

#include <thrift/Thrift.h>
#include <array>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "Catalog/Catalog.h"
#include "Logger/Logger.h"
#include "OSDependent/omnisci_path.h"
#include "ThriftHandler/DBHandler.h"

#define CALCITEPORT 3279

bool g_enable_thrift_logs{false};

int main(int argc, char* argv[]) {
  std::string base_path;
  bool force = false;
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages ")(
      "data",
      po::value<std::string>(&base_path)->required(),
      "Directory path to OmniSci catalogs")("force,f",
                                            "Force overwriting of existing OmniSci "
                                            "instance");

  desc.add_options()("enable-thrift-logs",
                     po::value<bool>(&g_enable_thrift_logs)
                         ->default_value(g_enable_thrift_logs)
                         ->implicit_value(true),
                     "Enable writing messages directly from thrift to stdout/stderr.");

  logger::LogOptions log_options(argv[0]);
  desc.add(log_options.get_options());

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
    if (vm.count("force")) {
      force = true;
    }
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }

  if (!g_enable_thrift_logs) {
    apache::thrift::GlobalOutput.setOutputFunction([](const char* msg) {});
  }

  if (!boost::filesystem::exists(base_path)) {
    std::cerr << "Catalog basepath " + base_path + " does not exist.\n";
    return 1;
  }
  std::string catalogs_path = base_path + "/mapd_catalogs";
  if (boost::filesystem::exists(catalogs_path)) {
    if (force) {
      boost::filesystem::remove_all(catalogs_path);
    } else {
      std::cerr << "OmniSci catalogs already initialized at " + base_path +
                       ". Use -f to force reinitialization.\n";
      return 1;
    }
  }
  std::string data_path = base_path + "/mapd_data";
  if (boost::filesystem::exists(data_path)) {
    if (force) {
      boost::filesystem::remove_all(data_path);
    } else {
      std::cerr << "OmniSci data directory already exists at " + base_path +
                       ". Use -f to force reinitialization.\n";
      return 1;
    }
  }
  std::string export_path = base_path + "/mapd_export";
  if (boost::filesystem::exists(export_path)) {
    if (force) {
      boost::filesystem::remove_all(export_path);
    } else {
      std::cerr << "OmniSci export directory already exists at " + base_path +
                       ". Use -f to force reinitialization.\n";
      return 1;
    }
  }
  std::string disk_cache_path = base_path + "/omnisci_disk_cache";
  if (boost::filesystem::exists(disk_cache_path)) {
    if (force) {
      boost::filesystem::remove_all(disk_cache_path);
    } else {
      std::cerr << "OmniSci disk cache already exists at " + disk_cache_path +
                       ". Use -f to force reinitialization.\n";
      return 1;
    }
  }
  if (!boost::filesystem::create_directory(catalogs_path)) {
    std::cerr << "Cannot create mapd_catalogs subdirectory under " << base_path
              << std::endl;
  }
  if (!boost::filesystem::create_directory(export_path)) {
    std::cerr << "Cannot create mapd_export subdirectory under " << base_path
              << std::endl;
  }

  log_options.set_base_path(base_path);
  logger::init(log_options);

  try {
    SystemParameters sys_parms;
    auto dummy = std::make_shared<Data_Namespace::DataMgr>(
        data_path, sys_parms, nullptr, false, 0);
    auto calcite =
        std::make_shared<Calcite>(-1, CALCITEPORT, base_path, 1024, 5000, true, "");
    g_base_path = base_path;
    auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
    sys_cat.init(base_path, dummy, {}, calcite, true, false, {});

  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  Catalog_Namespace::SysCatalog::destroy();
  return 0;
}
