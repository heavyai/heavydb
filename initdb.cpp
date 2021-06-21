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

static const std::array<std::string, 3> SampleGeoFileNames{"us-states.json",
                                                           "us-counties.json",
                                                           "countries.json"};
static const std::array<std::string, 3> SampleGeoTableNames{"omnisci_states",
                                                            "omnisci_counties",
                                                            "omnisci_countries"};

bool g_enable_thrift_logs{false};

static void loadGeo(std::string base_path) {
  TSessionId session_id{};
  SystemParameters system_parameters{};
  AuthMetadata auth_metadata{};
  std::string udf_filename{};
  std::string udf_compiler_path{};
  std::vector<std::string> udf_compiler_options{};
#ifdef ENABLE_GEOS
  std::string libgeos_so_filename{};
#endif
  std::vector<LeafHostInfo> db_leaves{};
  std::vector<LeafHostInfo> string_leaves{};

  // Whitelist root path for tests by default
  ddl_utils::FilePathWhitelist::clear();
  ddl_utils::FilePathWhitelist::initialize(base_path, "[\"/\"]", "[\"/\"]");

  // Based on default values observed from starting up an OmniSci DB server.
  const bool allow_multifrag{true};
  const bool jit_debug{false};
  const bool intel_jit_profile{false};
  const bool read_only{false};
  const bool allow_loop_joins{false};
  const bool enable_rendering{false};
  const bool renderer_use_vulkan_driver{false};
  const bool enable_auto_clear_render_mem{false};
  const int render_oom_retry_threshold{0};
  const size_t render_mem_bytes{500000000};
  const size_t max_concurrent_render_sessions{500};
  const bool render_compositor_use_last_gpu{false};
  const size_t reserved_gpu_mem{134217728};
  const size_t num_reader_threads{0};
  const bool legacy_syntax{true};
  const int idle_session_duration{60};
  const int max_session_duration{43200};
  const bool enable_runtime_udf_registration{false};
  system_parameters.omnisci_server_port = -1;
  system_parameters.calcite_port = 3280;

  system_parameters.aggregator = false;
  g_leaf_count = 0;
  g_cluster = false;

  File_Namespace::DiskCacheLevel cache_level{File_Namespace::DiskCacheLevel::fsi};
  File_Namespace::DiskCacheConfig disk_cache_config{
      File_Namespace::DiskCacheConfig::getDefaultPath(std::string(base_path)),
      cache_level};

  auto db_handler = std::make_unique<DBHandler>(db_leaves,
                                                string_leaves,
                                                base_path,
                                                allow_multifrag,
                                                jit_debug,
                                                intel_jit_profile,
                                                read_only,
                                                allow_loop_joins,
                                                enable_rendering,
                                                renderer_use_vulkan_driver,
                                                enable_auto_clear_render_mem,
                                                render_oom_retry_threshold,
                                                render_mem_bytes,
                                                max_concurrent_render_sessions,
                                                reserved_gpu_mem,
                                                render_compositor_use_last_gpu,
                                                num_reader_threads,
                                                auth_metadata,
                                                system_parameters,
                                                legacy_syntax,
                                                idle_session_duration,
                                                max_session_duration,
                                                enable_runtime_udf_registration,
                                                udf_filename,
                                                udf_compiler_path,
                                                udf_compiler_options,
#ifdef ENABLE_GEOS
                                                libgeos_so_filename,
#endif
                                                disk_cache_config,
                                                false);
  db_handler->internal_connect(session_id, OMNISCI_ROOT_USER, OMNISCI_DEFAULT_DB);

  // Execute on CPU by default
  db_handler->set_execution_mode(session_id, TExecuteMode::CPU);
  TQueryResult res;

  const size_t num_samples = SampleGeoFileNames.size();
  for (size_t i = 0; i < num_samples; i++) {
    const std::string table_name = SampleGeoTableNames[i];
    const std::string file_name = SampleGeoFileNames[i];

    const auto file_path = boost::filesystem::path(
        omnisci::get_root_abs_path() + "/ThirdParty/geo_samples/" + file_name);
    if (!boost::filesystem::exists(file_path)) {
      throw std::runtime_error(
          "Unable to populate geo sample data. File does not exist: " +
          file_path.string());
    }
    std::string sql_string =
        "COPY " + table_name + " FROM '" + file_path.string() + "' WITH (GEO='true');";
    db_handler->sql_execute(res, session_id, sql_string, true, "", -1, -1);
  }
}

int main(int argc, char* argv[]) {
  std::string base_path;
  bool force = false;
  bool skip_geo = false;
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages ")(
      "data",
      po::value<std::string>(&base_path)->required(),
      "Directory path to OmniSci catalogs")(
      "force,f", "Force overwriting of existing OmniSci instance")(
      "skip-geo", "Skip inserting sample geo data");

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
    if (vm.count("skip-geo")) {
      skip_geo = true;
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

    if (!skip_geo) {
      loadGeo(base_path);
    }

  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }
  return 0;
}
