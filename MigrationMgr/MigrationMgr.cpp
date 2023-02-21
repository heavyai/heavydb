/*
 * Copyright 2022 HEAVY.AI, Inc.
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

extern bool g_multi_instance;

namespace migrations {

void MigrationMgr::takeMigrationLock(const std::string& base_path) {
// TODO: support lock on Windows
#ifndef _WIN32
  // Only used for --multi-instance clusters.
  if (!g_multi_instance) {
    migration_enabled_ = true;
    return;
  }

  // If we already have the migration lock then do nothing.
  if (migration_mutex_) {
    return;
  }

  // Initialize the migration mutex. Will be locked until process exit.
  migration_mutex_ = std::make_unique<heavyai::DistributedSharedMutex>(
      std::filesystem::path(base_path) / shared::kLockfilesDirectoryName /
      "migration.lockfile");

  // Take an exclusive lock if we can. If we get the exclusive lock, then later it will be
  // relaxed to a shared lock, after we run migrations.
  migration_enabled_ = migration_mutex_->try_lock();
  if (!g_multi_instance && !migration_enabled_) {
    throw std::runtime_error(
        "another HeavyDB server instance is already using data directory: " + base_path);
  }

  // If we didn't get the exclusive lock, we'll wait for a shared lock instead, and we
  // won't run migrations.
  if (!migration_enabled_) {
    migration_mutex_->lock_shared();
  }
#else
  migration_enabled_ = true;
#endif  // _WIN32
}

void MigrationMgr::relaxMigrationLock() {
// TODO: support lock on Windows
#ifndef _WIN32
  // Only used for --multi-instance clusters.
  if (!g_multi_instance) {
    return;
  }

  // If we ran migrations, now relax the exclusive lock to a shared lock.
  if (migration_enabled_ && migration_mutex_) {
    migration_mutex_->convert_lock_shared();
  }
#endif  // _WIN32
}

void MigrationMgr::migrateDateInDaysMetadata(
    const Catalog_Namespace::TableDescriptorMapById& table_descriptors_by_id,
    const int database_id,
    Catalog_Namespace::Catalog* cat,
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

void MigrationMgr::dropRenderGroupColumns(
    const Catalog_Namespace::TableDescriptorMapById& table_descriptors_by_id,
    Catalog_Namespace::Catalog* cat) {
  // for this catalog...
  CHECK(cat);
  auto& catalog = *cat;

  // skip info schema catalog
  if (catalog.isInfoSchemaDb()) {
    return;
  }

  // report catalog
  LOG(INFO) << "MigrationMgr: dropRenderGroupColumns: Processing catalog '"
            << catalog.name() << "'";

  // HeavyConnect cache
  auto* heavyconnect_cache =
      catalog.getDataMgr().getPersistentStorageMgr()->getDiskCache();

  // all tables...
  for (auto itr = table_descriptors_by_id.begin(); itr != table_descriptors_by_id.end();
       itr++) {
    // the table...
    auto const* td = itr->second;
    CHECK(td);

    // skip views and temps
    if (td->isView) {
      LOG(INFO) << "MigrationMgr: dropRenderGroupColumns:  Skipping view '"
                << td->tableName << "'";
      continue;
    }
    if (table_is_temporary(td)) {
      LOG(INFO) << "MigrationMgr: dropRenderGroupColumns:  Skipping temporary table '"
                << td->tableName << "'";
      continue;
    }

    LOG(INFO) << "MigrationMgr: dropRenderGroupColumns:  Examining table '"
              << td->tableName << "'";

    // find render group columns
    auto logical_cds =
        catalog.getAllColumnMetadataForTable(td->tableId, false, false, false);
    // prepare to capture names
    std::vector<std::string> columns_to_drop;
    // iterate all columns
    for (auto itr = logical_cds.begin(); itr != logical_cds.end(); itr++) {
      auto const* cd = *itr;
      CHECK(cd);
      // poly or multipoly column?
      auto const cd_type = cd->columnType.get_type();
      if (cd_type == kPOLYGON || cd_type == kMULTIPOLYGON) {
        // next logical column
        auto const next_itr = std::next(itr);
        if (next_itr == logical_cds.end()) {
          // no next column, perhaps table already migrated
          break;
        }
        // the next column
        auto const* next_cd = *next_itr;
        CHECK(next_cd);
        // expected name?
        auto const next_name = next_cd->columnName;
        auto const expected_name = cd->columnName + "_render_group";
        if (next_name == expected_name) {
          // expected type?
          auto const next_type = next_cd->columnType.get_type();
          if (next_type == kINT) {
            // expected ID increment
            auto const next_id = next_cd->columnId;
            auto const expected_next_id =
                cd->columnId + cd->columnType.get_physical_cols() + 1;
            if (next_id == expected_next_id) {
              // report
              LOG(INFO) << "MigrationMgr: dropRenderGroupColumns:   Removing render "
                           "group column '"
                        << next_name << "'";
              // capture name
              columns_to_drop.emplace_back(next_name);
              // restart with the column after the one we identified
              itr++;
            } else {
              LOG(WARNING) << "MigrationMgr: dropRenderGroupColumns:   Expected render "
                              "group column '"
                           << next_name << "' has wrong ID (" << next_id << "/"
                           << expected_next_id << "), skipping...";
            }
          } else {
            LOG(WARNING) << "MigrationMgr: dropRenderGroupColumns:   Expected render "
                            "group column '"
                         << next_name << "' has wrong type (" << to_string(next_type)
                         << "), skipping...";
          }
        }
      }
    }

    // any to drop?
    if (columns_to_drop.size() == 0) {
      LOG(INFO)
          << "MigrationMgr: dropRenderGroupColumns:   No render group columns found";
      continue;
    }

    // drop the columns
    catalog.getSqliteConnector().query("BEGIN TRANSACTION");
    try {
      std::vector<int> column_ids;
      for (auto const& column : columns_to_drop) {
        auto const* cd = catalog.getMetadataForColumn(td->tableId, column);
        CHECK(cd);
        catalog.dropColumn(*td, *cd);
        column_ids.push_back(cd->columnId);
      }
      for (auto const* physical_td : catalog.getPhysicalTablesDescriptors(td)) {
        CHECK(physical_td);
        // getMetadataForTable() may not have been called on this table, so
        // the table may not yet have a fragmenter (which that function lazy
        // creates by default) so call it manually here to force creation of
        // the fragmenter so we can use it to actually drop the columns
        if (physical_td->fragmenter == nullptr) {
          catalog.getMetadataForTable(physical_td->tableId, true);
          CHECK(physical_td->fragmenter);
        }
        physical_td->fragmenter->dropColumns(column_ids);
      }
      catalog.rollLegacy(true);
      if (td->persistenceLevel == Data_Namespace::MemoryLevel::DISK_LEVEL) {
        catalog.resetTableEpochFloor(td->tableId);
        catalog.checkpoint(td->tableId);
      }
      catalog.getSqliteConnector().query("END TRANSACTION");
    } catch (std::exception& e) {
      catalog.setForReload(td->tableId);
      catalog.rollLegacy(false);
      catalog.getSqliteConnector().query("ROLLBACK TRANSACTION");
      LOG(ERROR) << "MigrationMgr: dropRenderGroupColumns:   Failed to drop render group "
                    "columns for table '"
                 << td->tableName << "' (" << e.what() << ")";
      // don't do anything else for this table
      continue;
    }

    // flush any HeavyConnect foreign table cache for the physical tables
    if (heavyconnect_cache) {
      for (auto const* physical_td : catalog.getPhysicalTablesDescriptors(td)) {
        CHECK(physical_td);
        LOG(INFO) << "MigrationMgr: dropRenderGroupColumns:   Flushing HeavyConnect "
                     "cache for table '"
                  << physical_td->tableName << "'";
        heavyconnect_cache->clearForTablePrefix(
            {catalog.getCurrentDB().dbId, physical_td->tableId});
      }
    }
  }
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
