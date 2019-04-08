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

#include <array>
#include <boost/filesystem.hpp>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include "Catalog/Catalog.h"
#include "Import/Importer.h"
#include "Shared/mapdpath.h"
#include "boost/program_options.hpp"

#define CALCITEPORT 36279

static const std::array<std::string, 3> SampleGeoFileNames{"us-states.json",
                                                           "us-counties.json",
                                                           "countries.json"};
static const std::array<std::string, 3> SampleGeoTableNames{"omnisci_states",
                                                            "omnisci_counties",
                                                            "omnisci_countries"};

int main(int argc, char* argv[]) {
  std::string base_path;
  bool force = false;
  bool skip_geo = false;
  namespace po = boost::program_options;

  google::InitGoogleLogging(argv[0]);

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages ")(
      "data",
      po::value<std::string>(&base_path)->required(),
      "Directory path to OmniSci catalogs")(
      "force,f", "Force overwriting of existing OmniSci instance")(
      "skip-geo", "Skip inserting sample geo data");

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
      std::cout << "Usage: initdb [-f] <catalog path>\n";
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
  if (!boost::filesystem::create_directory(catalogs_path)) {
    std::cerr << "Cannot create mapd_catalogs subdirectory under " << base_path
              << std::endl;
  }
  if (!boost::filesystem::create_directory(export_path)) {
    std::cerr << "Cannot create mapd_export subdirectory under " << base_path
              << std::endl;
  }

  try {
    MapDParameters mapd_parms;
    auto dummy =
        std::make_shared<Data_Namespace::DataMgr>(data_path, mapd_parms, false, 0);
    auto calcite = std::make_shared<Calcite>(-1, CALCITEPORT, base_path, 1024);
    auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
    sys_cat.init(base_path, dummy, {}, calcite, true, false, false, {});

    if (!skip_geo) {
      // Add geo samples to the system database using the root user
      Catalog_Namespace::DBMetadata cur_db;
      const std::string db_name(MAPD_SYSTEM_DB);
      CHECK(sys_cat.getMetadataForDB(db_name, cur_db));
      auto cat = Catalog_Namespace::Catalog::get(
          base_path, cur_db, dummy, std::vector<LeafHostInfo>(), calcite, false);
      Catalog_Namespace::UserMetadata user;
      CHECK(sys_cat.getMetadataForUser(MAPD_ROOT_USER, user));

      Importer_NS::ImportDriver import_driver(cat, user);

      const size_t num_samples = SampleGeoFileNames.size();
      for (size_t i = 0; i < num_samples; i++) {
        const std::string table_name = SampleGeoTableNames[i];
        const std::string file_name = SampleGeoFileNames[i];

        const auto file_path = boost::filesystem::path(
            mapd_root_abs_path() + "/ThirdParty/geo_samples/" + file_name);
        if (!boost::filesystem::exists(file_path)) {
          throw std::runtime_error(
              "Unable to populate geo sample data. File does not exist: " +
              file_path.string());
        }

        import_driver.import_geo_table(file_path.string(), table_name);
      }
    }

  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }
  return 0;
}
