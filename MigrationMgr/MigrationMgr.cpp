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

#include "MigrationMgr/MigrationMgr.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/TableOptimizer.h"
#include "Shared/SysDefinitions.h"
#include "Shared/sqltypes.h"

#include "MapDRelease.h"

namespace migrations {

void MigrationMgr::migrateDateInDaysMetadata(
    const Catalog_Namespace::TableDescriptorMapById& table_descriptors_by_id,
    const int database_id,
    const Catalog_Namespace::Catalog* cat,
    SqliteConnector& sqlite) {
  std::vector<int> tables_migrated = {};
  std::unordered_map<int, std::vector<std::string>> tables_to_migrate;
  sqlite.query("BEGIN TRANSACTION");
  try {
    sqlite.query(
        "select name from sqlite_master WHERE type='table' AND "
        "name='mapd_version_history'");
    if (sqlite.getNumRows() == 0) {
      sqlite.query(
          "CREATE TABLE mapd_version_history(version integer, migration_history text "
          "unique)");
      sqlite.query(
          "CREATE TABLE mapd_date_in_days_column_migration_tmp(table_id integer primary "
          "key)");
    } else {
      sqlite.query(
          "select * from mapd_version_history where migration_history = "
          "'date_in_days_column'");
      if (sqlite.getNumRows() != 0) {
        // no need for further execution
        sqlite.query("END TRANSACTION");
        return;
      }
      LOG(INFO) << "Checking for date columns requiring metadata migration.";
      sqlite.query(
          "select name from sqlite_master where type='table' AND "
          "name='mapd_date_in_days_column_migration_tmp'");
      if (sqlite.getNumRows() != 0) {
        sqlite.query("select table_id from mapd_date_in_days_column_migration_tmp");
        if (sqlite.getNumRows() != 0) {
          for (size_t i = 0; i < sqlite.getNumRows(); i++) {
            tables_migrated.push_back(sqlite.getData<int>(i, 0));
          }
        }
      } else {
        sqlite.query(
            "CREATE TABLE mapd_date_in_days_column_migration_tmp(table_id integer "
            "primary key)");
      }
    }
    sqlite.query_with_text_params(
        "SELECT tables.tableid, tables.name, columns.name FROM mapd_tables tables, "
        "mapd_columns columns where tables.tableid = columns.tableid AND "
        "columns.coltype = ?1 AND columns.compression = ?2",
        std::vector<std::string>{
            std::to_string(static_cast<int>(SQLTypes::kDATE)),
            std::to_string(static_cast<int>(EncodingType::kENCODING_DATE_IN_DAYS))});
    if (sqlite.getNumRows() != 0) {
      for (size_t i = 0; i < sqlite.getNumRows(); i++) {
        tables_to_migrate[sqlite.getData<int>(i, 0)] = {
            sqlite.getData<std::string>(i, 1), sqlite.getData<std::string>(i, 2)};
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to complete migration on date in days column metadata: "
               << e.what();
    sqlite.query("ROLLBACK");
    throw;
  }
  sqlite.query("END TRANSACTION");

  for (auto& id_names : tables_to_migrate) {
    if (std::find(tables_migrated.begin(), tables_migrated.end(), id_names.first) ==
        tables_migrated.end()) {
      sqlite.query("BEGIN TRANSACTION");
      try {
        LOG(INFO) << "Table: " << id_names.second[0]
                  << " may suffer from issues with DATE column: " << id_names.second[1]
                  << ". Running an OPTIMIZE command to solve any issues with metadata.";

        // TODO(adb): Could have the TableOptimizer get the Executor and avoid including
        // Execute.h
        auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
        auto table_desc_itr = table_descriptors_by_id.find(id_names.first);
        if (table_desc_itr == table_descriptors_by_id.end()) {
          throw std::runtime_error("Table descriptor does not exist for table " +
                                   id_names.second[0] + " does not exist.");
        }
        auto td = table_desc_itr->second;
        TableOptimizer optimizer(td, executor.get(), *cat);
        optimizer.recomputeMetadata();

        sqlite.query_with_text_params(
            "INSERT INTO mapd_date_in_days_column_migration_tmp VALUES(?)",
            std::vector<std::string>{std::to_string(id_names.first)});
      } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to complete metadata migration on date in days column: "
                   << e.what();
        sqlite.query("ROLLBACK");
        throw;
      }
      sqlite.query("COMMIT");
    }
  }

  sqlite.query("BEGIN TRANSACTION");
  try {
    sqlite.query("DROP TABLE mapd_date_in_days_column_migration_tmp");
    sqlite.query_with_text_params(
        "INSERT INTO mapd_version_history(version, migration_history) values(?,?)",
        std::vector<std::string>{std::to_string(MAPD_VERSION), "date_in_days_column"});
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to complete migraion on date in days column: " << e.what();
    sqlite.query("ROLLBACK");
    throw;
  }
  sqlite.query("END TRANSACTION");
  LOG(INFO) << "Successfully migrated all date in days column metadata.";
}

namespace {
bool rename_and_symlink_path(const std::filesystem::path& old_path,
                             const std::filesystem::path& new_path) {
  bool file_updated{false};
  if (std::filesystem::exists(old_path)) {
    // Skip if we have already created a symlink for the old path.
    if (std::filesystem::is_symlink(old_path)) {
      if (std::filesystem::read_symlink(old_path) != new_path.filename()) {
        std::stringstream ss;
        ss << "Rebrand migration: Encountered an unexpected symlink at path: " << old_path
           << ". Symlink does not reference file: " << new_path.filename();
        throw std::runtime_error(ss.str());
      }
      if (!std::filesystem::exists(new_path)) {
        std::stringstream ss;
        ss << "Rebrand migration: Encountered symlink at legacy path: " << old_path
           << " but no corresponding file at new path: " << new_path;
        throw std::runtime_error(ss.str());
      }
    } else {
      if (std::filesystem::exists(new_path)) {
        std::stringstream ss;
        ss << "Rebrand migration: Encountered existing non-symlink files at the legacy "
              "path: "
           << old_path << " and new path: " << new_path;
        throw std::runtime_error(ss.str());
      }
      std::filesystem::rename(old_path, new_path);
      std::cout << "Rebrand migration: Renamed " << old_path << " to " << new_path
                << std::endl;
      file_updated = true;
    }
  }

  if (std::filesystem::exists(old_path)) {
    if (!std::filesystem::is_symlink(old_path)) {
      std::stringstream ss;
      ss << "Rebrand migration: An unexpected error occurred. A symlink should have been "
            "created at "
         << old_path;
      throw std::runtime_error(ss.str());
    }
    if (std::filesystem::read_symlink(old_path) != new_path.filename()) {
      std::stringstream ss;
      ss << "Rebrand migration: Encountered an unexpected symlink at path: " << old_path
         << ". Symlink does not reference file: " << new_path.filename();
      throw std::runtime_error(ss.str());
    }
  } else if (std::filesystem::exists(new_path)) {
    std::filesystem::create_symlink(new_path.filename(), old_path);
    std::cout << "Rebrand migration: Added symlink from " << old_path << " to "
              << new_path.filename() << std::endl;
    file_updated = true;
  }
  return file_updated;
}

bool rename_and_symlink_file(const std::filesystem::path& base_path,
                             const std::string& dir_name,
                             const std::string& old_file_name,
                             const std::string& new_file_name) {
  auto old_path = std::filesystem::canonical(base_path);
  auto new_path = std::filesystem::canonical(base_path);
  if (!dir_name.empty()) {
    old_path /= dir_name;
    new_path /= dir_name;
  }
  if (old_file_name.empty()) {
    throw std::runtime_error(
        "Unexpected error in rename_and_symlink_file: old_file_name is empty");
  }
  old_path /= old_file_name;

  if (new_file_name.empty()) {
    throw std::runtime_error(
        "Unexpected error in rename_and_symlink_file: new_file_name is empty");
  }
  new_path /= new_file_name;

  return rename_and_symlink_path(old_path, new_path);
}
}  // namespace

void MigrationMgr::executeRebrandMigration(const std::string& base_path) {
  bool migration_occurred{false};

  // clang-format off
  const std::map<std::string, std::string> old_to_new_dir_names {
    {"mapd_catalogs", shared::kCatalogDirectoryName},
    {"mapd_data", shared::kDataDirectoryName},
    {"mapd_log", shared::kDefaultLogDirName},
    {"mapd_export", shared::kDefaultExportDirName},
    {"mapd_import", shared::kDefaultImportDirName},
    {"omnisci_key_store", shared::kDefaultKeyStoreDirName}
  };
  // clang-format on

  const auto storage_base_path = std::filesystem::canonical(base_path);
  // Rename legacy directories (if they exist), and create symlinks from legacy directory
  // names to the new names (if they don't already exist).
  for (const auto& [old_dir_name, new_dir_name] : old_to_new_dir_names) {
    auto old_path = storage_base_path / old_dir_name;
    auto new_path = storage_base_path / new_dir_name;
    if (rename_and_symlink_path(old_path, new_path)) {
      migration_occurred = true;
    }
  }

  // Rename legacy files and create symlinks to them.
  const auto license_updated = rename_and_symlink_file(
      storage_base_path, "", "omnisci.license", shared::kDefaultLicenseFileName);
  const auto key_updated = rename_and_symlink_file(storage_base_path,
                                                   shared::kDefaultKeyStoreDirName,
                                                   "omnisci.pem",
                                                   shared::kDefaultKeyFileName);
  const auto sys_catalog_updated = rename_and_symlink_file(storage_base_path,
                                                           shared::kCatalogDirectoryName,
                                                           "omnisci_system_catalog",
                                                           shared::kSystemCatalogName);
  if (license_updated || key_updated || sys_catalog_updated) {
    migration_occurred = true;
  }

  // Delete the disk cache directory and legacy files that will no longer be used.
  const std::array<std::filesystem::path, 9> files_to_delete{
      storage_base_path / "omnisci_disk_cache",
      storage_base_path / "omnisci_server_pid.lck",
      storage_base_path / "mapd_server_pid.lck",
      storage_base_path / shared::kDefaultLogDirName / "omnisci_server.FATAL",
      storage_base_path / shared::kDefaultLogDirName / "omnisci_server.ERROR",
      storage_base_path / shared::kDefaultLogDirName / "omnisci_server.WARNING",
      storage_base_path / shared::kDefaultLogDirName / "omnisci_server.INFO",
      storage_base_path / shared::kDefaultLogDirName / "omnisci_web_server.ALL",
      storage_base_path / shared::kDefaultLogDirName / "omnisci_web_server.ACCESS"};

  for (const auto& file_path : files_to_delete) {
    if (std::filesystem::exists(file_path)) {
      std::filesystem::remove_all(file_path);
      std::cout << "Rebrand migration: Deleted file " << file_path << std::endl;
      migration_occurred = true;
    }
  }
  if (migration_occurred) {
    std::cout << "Rebrand migration completed" << std::endl;
  }
}
}  // namespace migrations
