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
#include <string>
#include <unordered_map>
#include <vector>

#include "Catalog/CatalogSchemaProvider.h"
#include "DataMgr/DataMgrBufferProvider.h"
#include "DataMgr/DataMgrDataProvider.h"
#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/TableOptimizer.h"
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

        auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                              &cat->getDataMgr(),
                                              cat->getDataMgr().getBufferProvider());
        auto data_provider = std::make_shared<DataMgrDataProvider>(&cat->getDataMgr());
        auto schema_provider =
            std::make_shared<Catalog_Namespace::CatalogSchemaProvider>(cat);
        executor->setSchemaProvider(schema_provider);

        auto table_desc_itr = table_descriptors_by_id.find(id_names.first);
        if (table_desc_itr == table_descriptors_by_id.end()) {
          throw std::runtime_error("Table descriptor does not exist for table " +
                                   id_names.second[0] + " does not exist.");
        }
        auto td = table_desc_itr->second;
        TableOptimizer optimizer(
            td, executor.get(), data_provider, schema_provider, *cat);
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

}  // namespace migrations
