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

/**
 * @file    Catalog.h
 * @author  Todd Mostak <todd@map-d.com>, Wei Hong <wei@map-d.com>
 * @brief   This file contains the class specification and related data structures for
 * Catalog.
 *
 * This file contains the Catalog class specification. The Catalog class is responsible
 * for storing, accessing and caching metadata for a single database. A global metadata
 * could be accessed with SysCatalog class.
 *
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <ctime>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "Calcite/Calcite.h"
#include "Catalog/ColumnDescriptor.h"
#include "Catalog/DashboardDescriptor.h"
#include "Catalog/DictDescriptor.h"
#include "Catalog/ForeignServer.h"
#include "Catalog/ForeignTable.h"
#include "Catalog/LinkDescriptor.h"
#include "Catalog/SessionInfo.h"
#include "Catalog/SysCatalog.h"
#include "Catalog/TableDescriptor.h"
#include "Catalog/TableMetadata.h"
#include "Catalog/Types.h"
#include "DataMgr/DataMgr.h"
#include "QueryEngine/CompilationOptions.h"
#include "Shared/mapd_shared_mutex.h"
#include "SqliteConnector/SqliteConnector.h"

#include "LeafHostInfo.h"

enum GetTablesType { GET_PHYSICAL_TABLES_AND_VIEWS, GET_PHYSICAL_TABLES, GET_VIEWS };

namespace Parser {

class SharedDictionaryDef;

}  // namespace Parser

class TableArchiver;

// SPI means Sequential Positional Index which is equivalent to the input index in a
// RexInput node
#define SPIMAP_MAGIC1 (std::numeric_limits<unsigned>::max() / 4)
#define SPIMAP_MAGIC2 8
#define SPIMAP_GEO_PHYSICAL_INPUT(c, i) \
  (SPIMAP_MAGIC1 + (unsigned)(SPIMAP_MAGIC2 * ((c) + 1) + (i)))

namespace File_Namespace {
struct FileMgrParams;
}
namespace Catalog_Namespace {
struct TableEpochInfo {
  int32_t table_id, table_epoch, leaf_index{-1};

  TableEpochInfo(const int32_t table_id_param, const int32_t table_epoch_param)
      : table_id(table_id_param), table_epoch(table_epoch_param) {}
  TableEpochInfo(const int32_t table_id_param,
                 const int32_t table_epoch_param,
                 const size_t leaf_index_param)
      : table_id(table_id_param)
      , table_epoch(table_epoch_param)
      , leaf_index(leaf_index_param) {}
};

/**
 * @type Catalog
 * @brief class for a per-database catalog.  also includes metadata for the
 * current database and the current user.
 */

class Catalog final {
 public:
  /**
   * @brief Constructor - takes basePath to already extant
   * data directory for writing
   * @param basePath directory path for writing catalog
   * @param dbName name of the database
   * @param fragmenter Fragmenter object
   * metadata - expects for this directory to already exist
   */
  Catalog(const std::string& basePath,
          const DBMetadata& curDB,
          std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
          const std::vector<LeafHostInfo>& string_dict_hosts,
          std::shared_ptr<Calcite> calcite,
          bool is_new_db);
  /**
   * @brief Constructor builds a hollow catalog
   * used during constructor of other catalogs
   */
  Catalog();

  /**
   * @brief Destructor - deletes all
   * ColumnDescriptor and TableDescriptor structures
   * which were allocated on the heap and writes
   * Catalog to Sqlite
   */
  ~Catalog();

  static void expandGeoColumn(const ColumnDescriptor& cd,
                              std::list<ColumnDescriptor>& columns);
  void createTable(TableDescriptor& td,
                   const std::list<ColumnDescriptor>& columns,
                   const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs,
                   bool isLogicalTable);
  void createShardedTable(
      TableDescriptor& td,
      const std::list<ColumnDescriptor>& columns,
      const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs);
  int32_t createDashboard(DashboardDescriptor& vd);
  void replaceDashboard(DashboardDescriptor& vd);
  std::string createLink(LinkDescriptor& ld, size_t min_length);
  void dropTable(const TableDescriptor* td);
  void truncateTable(const TableDescriptor* td);
  void renameTable(const TableDescriptor* td, const std::string& newTableName);
  void renameColumn(const TableDescriptor* td,
                    const ColumnDescriptor* cd,
                    const std::string& newColumnName);
  void addColumn(const TableDescriptor& td, ColumnDescriptor& cd);
  void dropColumn(const TableDescriptor& td, const ColumnDescriptor& cd);
  void removeChunks(const int table_id);
  void removeFragmenterForTable(const int table_id);

  const std::map<int, const ColumnDescriptor*> getDictionaryToColumnMapping();

  /**
   * @brief Returns a pointer to a const TableDescriptor struct matching
   * the provided tableName
   * @param tableName table specified column belongs to
   * @return pointer to const TableDescriptor object queried for or nullptr if it does not
   * exist.
   */

  const TableDescriptor* getMetadataForTable(const std::string& tableName,
                                             const bool populateFragmenter = true) const;
  const TableDescriptor* getMetadataForTableImpl(int tableId,
                                                 const bool populateFragmenter) const;
  const TableDescriptor* getMetadataForTable(int tableId,
                                             bool populateFragmenter = true) const;

  const ColumnDescriptor* getMetadataForColumn(int tableId,
                                               const std::string& colName) const;
  const ColumnDescriptor* getMetadataForColumn(int tableId, int columnId) const;
  const ColumnDescriptor* getMetadataForColumnUnlocked(int tableId, int columnId) const;

  const int getColumnIdBySpi(const int tableId, const size_t spi) const;
  const ColumnDescriptor* getMetadataForColumnBySpi(const int tableId,
                                                    const size_t spi) const;

  const DashboardDescriptor* getMetadataForDashboard(const std::string& userId,
                                                     const std::string& dashName) const;

  const DashboardDescriptor* getMetadataForDashboard(const int32_t dashboard_id) const;
  void deleteMetadataForDashboards(const std::vector<int32_t> ids,
                                   const UserMetadata& user);

  const LinkDescriptor* getMetadataForLink(const std::string& link) const;
  const LinkDescriptor* getMetadataForLink(int linkId) const;

  const foreign_storage::ForeignTable* getForeignTableUnlocked(int tableId) const;
  const foreign_storage::ForeignTable* getForeignTable(
      const std::string& tableName) const;

  const foreign_storage::ForeignTable* getForeignTable(int table_id) const;

  /**
   * @brief Returns a list of pointers to constant ColumnDescriptor structs for all the
   * columns from a particular table specified by table id
   * @param tableId table id we want the column metadata for
   * @return list of pointers to const ColumnDescriptor structs - one
   * for each and every column in the table
   *
   */
  std::list<const ColumnDescriptor*> getAllColumnMetadataForTable(
      const int tableId,
      const bool fetchSystemColumns,
      const bool fetchVirtualColumns,
      const bool fetchPhysicalColumns) const;
  /**
   * Same as above, but without first taking a catalog read lock.
   */
  std::list<const ColumnDescriptor*> getAllColumnMetadataForTableUnlocked(
      const int tableId,
      const bool fetchSystemColumns,
      const bool fetchVirtualColumns,
      const bool fetchPhysicalColumns) const;

  std::list<const TableDescriptor*> getAllTableMetadata() const;
  std::list<const DashboardDescriptor*> getAllDashboardsMetadata() const;
  const DBMetadata& getCurrentDB() const { return currentDB_; }
  Data_Namespace::DataMgr& getDataMgr() const { return *dataMgr_; }
  std::shared_ptr<Calcite> getCalciteMgr() const { return calciteMgr_; }
  const std::string& getBasePath() const { return basePath_; }

  const DictDescriptor* getMetadataForDict(int dict_ref, bool loadDict = true) const;
  const DictDescriptor* getMetadataForDictUnlocked(int dict_ref, bool loadDict) const;

  const std::vector<LeafHostInfo>& getStringDictionaryHosts() const;

  const ColumnDescriptor* getShardColumnMetadataForTable(const TableDescriptor* td) const;

  std::vector<const TableDescriptor*> getPhysicalTablesDescriptors(
      const TableDescriptor* logical_table_desc,
      bool populate_fragmenter = true) const;

  /**
   * Get names of all tables accessible to user.
   *
   * @param user - user to retrieve table names for
   * @param get_tables_type - enum indicating if tables, views or tables & views
   * should be returned
   * @return table_names - vector of table names accessible by user
   */
  std::vector<std::string> getTableNamesForUser(
      const UserMetadata& user,
      const GetTablesType get_tables_type) const;

  /**
   * Get table descriptors of all tables accessible to user.
   *
   * @param user - user to retrieve table descriptors for
   * @param get_tables_type - enum indicating if tables, views or tables & views
   * should be returned
   * @return table_descriptors - vector of table descriptors accessible by user
   */

  std::vector<TableMetadata> getTablesMetadataForUser(
      const UserMetadata& user_metadata,
      const GetTablesType get_tables_type,
      const std::string& filter_table_name) const;

  int32_t getTableEpoch(const int32_t db_id, const int32_t table_id) const;
  void setTableEpoch(const int db_id, const int table_id, const int new_epoch);
  void setMaxRollbackEpochs(const int32_t table_id, const int32_t max_rollback_epochs);

  std::vector<TableEpochInfo> getTableEpochs(const int32_t db_id,
                                             const int32_t table_id) const;
  void setTableEpochs(const int32_t db_id,
                      const std::vector<TableEpochInfo>& table_epochs);

  void setTableEpochsLogExceptions(const int32_t db_id,
                                   const std::vector<TableEpochInfo>& table_epochs);

  int getDatabaseId() const { return currentDB_.dbId; }
  SqliteConnector& getSqliteConnector() { return sqliteConnector_; }
  void roll(const bool forward);
  DictRef addDictionary(ColumnDescriptor& cd);
  void delDictionary(const ColumnDescriptor& cd);
  void getDictionary(const ColumnDescriptor& cd,
                     std::map<int, StringDictionary*>& stringDicts);

  const bool checkMetadataForDeletedRecs(const TableDescriptor* td, int column_id) const;
  const ColumnDescriptor* getDeletedColumn(const TableDescriptor* td) const;
  const ColumnDescriptor* getDeletedColumnIfRowsDeleted(const TableDescriptor* td) const;

  void setDeletedColumn(const TableDescriptor* td, const ColumnDescriptor* cd);
  void setDeletedColumnUnlocked(const TableDescriptor* td, const ColumnDescriptor* cd);
  int getLogicalTableId(const int physicalTableId) const;
  void checkpoint(const int logicalTableId) const;
  void checkpointWithAutoRollback(const int logical_table_id);
  std::string name() const { return getCurrentDB().dbName; }
  void eraseDBData();
  void eraseTablePhysicalData(const TableDescriptor* td);
  void vacuumDeletedRows(const TableDescriptor* td) const;
  void vacuumDeletedRows(const int logicalTableId) const;
  void setForReload(const int32_t tableId);

  std::vector<std::string> getTableDataDirectories(const TableDescriptor* td) const;
  std::vector<std::string> getTableDictDirectories(const TableDescriptor* td) const;
  std::string getColumnDictDirectory(const ColumnDescriptor* cd) const;
  std::string dumpSchema(const TableDescriptor* td) const;
  std::string dumpCreateTable(const TableDescriptor* td,
                              bool multiline_formatting = true,
                              bool dump_defaults = false) const;

  /**
   * Gets the DDL statement used to create a foreign table schema.
   *
   * @param if_not_exists - flag that indicates whether or not to include
   * the "IF NOT EXISTS" phrase in the DDL statement
   * @return string containing DDL statement
   */
  static const std::string getForeignTableSchema(bool if_not_exists = false);

  /**
   * Gets the DDL statement used to create a foreign server schema.
   *
   * @param if_not_exists - flag that indicates whether or not to include
   * the "IF NOT EXISTS" phrase in the DDL statement
   * @return string containing DDL statement
   */
  static const std::string getForeignServerSchema(bool if_not_exists = false);

  /**
   * Creates a new foreign server DB object.
   *
   * @param foreign_server - unique pointer to struct containing foreign server details
   * @param if_not_exists - flag indicating whether or not an attempt to create a new
   * foreign server should occur if a server with the same name already exists. An
   * exception is thrown if this flag is set to "false" and an attempt is made to create
   * a pre-existing foreign server
   */
  void createForeignServer(std::unique_ptr<foreign_storage::ForeignServer> foreign_server,
                           bool if_not_exists);

  /**
   * Gets a pointer to a struct containing foreign server details.
   *
   * @param server_name - Name of foreign server whose details will be fetched
   * @return pointer to a struct containing foreign server details. nullptr is returned if
   * no foreign server exists with the given name
   */
  const foreign_storage::ForeignServer* getForeignServer(
      const std::string& server_name) const;

  /**
   * Gets a pointer to a struct containing foreign server details fetched from storage.
   * This is mainly used for testing when asserting that expected catalog data is
   * persisted.
   *
   * @param server_name - Name of foreign server whose details will be fetched
   * @return pointer to a struct containing foreign server details. nullptr is returned if
   * no foreign server exists with the given name
   */
  const std::unique_ptr<const foreign_storage::ForeignServer> getForeignServerFromStorage(
      const std::string& server_name);

  /**
   * Gets a pointer to a struct containing foreign table details fetched from storage.
   * This is mainly used for testing when asserting that expected catalog data is
   * persisted.
   *
   * @param table_name - Name of foreign table whose details will be fetched
   * @return pointer to a struct containing foreign table details. nullptr is returned if
   * no foreign table exists with the given name
   */
  const std::unique_ptr<const foreign_storage::ForeignTable> getForeignTableFromStorage(
      int table_id);

  /**
   * Change the owner of a Foreign Server to a new owner.
   *
   * @param server_name - Name of the foreign server whose owner to change
   * @param new_owner_id - New owner's user id
   */
  void changeForeignServerOwner(const std::string& server_name, const int new_owner_id);

  /**
   * Set the data wrapper of a Foreign Server.
   *
   * @param server_name - Name of the foreign server whose data wrapper will be set
   * @param data_wrapper - Data wrapper to use
   */
  void setForeignServerDataWrapper(const std::string& server_name,
                                   const std::string& data_wrapper);
  /**
   * Set the options of a Foreign Server.
   *
   * @param server_name - Name of the foreign server whose options will be set
   * @param options - Options to set
   */
  void setForeignServerOptions(const std::string& server_name,
                               const std::string& options);
  /**
   * Rename a Foreign Server.
   *
   * @param server_name - Name of the foreign server whose name will be changed
   * @param name - New name of server
   */
  void renameForeignServer(const std::string& server_name, const std::string& name);

  /**
   * Drops/deletes a foreign server DB object.
   *
   * @param server_name - Name of foreign server that will be deleted
   */
  void dropForeignServer(const std::string& server_name);

  /**
   * Performs a query on all foreign servers accessible to user with optional filter,
   * and returns pointers toresulting server objects
   *
   * @param filters - Json Value representing SQL WHERE clause to filter results, eg.:
   * "WHERE attribute1 = value1 AND attribute2 LIKE value2", or Null Value
   *  Array of Values with attribute, value, operator, and chain specifier after first
   * entry
   * @param user - user to retrieve server names
   * @param results - results returned as a vector of pointers to
   * const foreign_storage::ForeignServer
   */
  void getForeignServersForUser(
      const rapidjson::Value* filters,
      const UserMetadata& user,
      std::vector<const foreign_storage::ForeignServer*>& results);

  /**
   * Creates default local file servers (if they don't already exist).
   */
  void createDefaultServersIfNotExists();

  /**
   * Validates that a table or view with given name does not already exist.
   * An exception is thrown if a table or view with given name already exists and
   * "if_not_exists" is false.
   *
   * @param name - Name of table or view whose existence is checked
   * @param if_not_exists - flag indicating whether or not existence of a table or view
   * with given name is an exception
   * @return true if table or view with name does not exist. Otherwise, return false
   */
  bool validateNonExistentTableOrView(const std::string& name, const bool if_not_exists);

  /**
   * Gets all the foreign tables that are pending refreshes. The list of tables
   * includes tables that are configured for scheduled refreshes with next
   * refresh timestamps that are in the past.
   *
   * @return foreign tables pending refreshes
   */
  std::vector<const TableDescriptor*> getAllForeignTablesForRefresh() const;

  /**
   * Updates the last and next (if applicable) refresh times of the foreign table
   * with the given table id.
   *
   * @param table_id - id of table to apply updates to
   */
  void updateForeignTableRefreshTimes(const int32_t table_id);

  /**
   * Set the options of a Foreign Table.
   *
   * @param table_name - Name of the foreign table whose options will be set
   * @param options - Options to set
   */
  void setForeignTableOptions(const std::string& table_name,
                              foreign_storage::OptionsMap& options_map,
                              bool clear_existing_options = true);

  void updateLeaf(const LeafHostInfo& string_dict_host);

 protected:
  void CheckAndExecuteMigrations();
  void CheckAndExecuteMigrationsPostBuildMaps();
  void updateDictionaryNames();
  void updateTableDescriptorSchema();
  void updateFixlenArrayColumns();
  void updateGeoColumns();
  void updateFrontendViewSchema();
  void updateLinkSchema();
  void updateFrontendViewAndLinkUsers();
  void updateLogicalToPhysicalTableLinkSchema();
  void updateLogicalToPhysicalTableMap(const int32_t logical_tb_id);
  void updateDictionarySchema();
  void updatePageSize();
  void updateDeletedColumnIndicator();
  void updateFrontendViewsToDashboards();
  void createFsiSchemas();
  void dropFsiSchemasAndTables();
  void recordOwnershipOfObjectsInObjectPermissions();
  void checkDateInDaysColumnMigration();
  void createDashboardSystemRoles();
  void buildMaps();
  void addTableToMap(const TableDescriptor* td,
                     const std::list<ColumnDescriptor>& columns,
                     const std::list<DictDescriptor>& dicts);
  void addReferenceToForeignDict(ColumnDescriptor& referencing_column,
                                 Parser::SharedDictionaryDef shared_dict_def,
                                 const bool persist_reference);
  bool setColumnSharedDictionary(
      ColumnDescriptor& cd,
      std::list<ColumnDescriptor>& cdd,
      std::list<DictDescriptor>& dds,
      const TableDescriptor td,
      const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs);
  void setColumnDictionary(ColumnDescriptor& cd,
                           std::list<DictDescriptor>& dds,
                           const TableDescriptor& td,
                           const bool isLogicalTable);
  void addFrontendViewToMap(DashboardDescriptor& vd);
  void addFrontendViewToMapNoLock(DashboardDescriptor& vd);
  void addLinkToMap(LinkDescriptor& ld);
  void removeTableFromMap(const std::string& tableName,
                          const int tableId,
                          const bool is_on_error = false);
  void doDropTable(const TableDescriptor* td);
  void executeDropTableSqliteQueries(const TableDescriptor* td);
  void doTruncateTable(const TableDescriptor* td);
  void renamePhysicalTable(const TableDescriptor* td, const std::string& newTableName);
  void instantiateFragmenter(TableDescriptor* td) const;
  void getAllColumnMetadataForTableImpl(const TableDescriptor* td,
                                        std::list<const ColumnDescriptor*>& colDescs,
                                        const bool fetchSystemColumns,
                                        const bool fetchVirtualColumns,
                                        const bool fetchPhysicalColumns) const;
  std::string calculateSHA1(const std::string& data);
  std::string generatePhysicalTableName(const std::string& logicalTableName,
                                        const int32_t& shardNumber);
  std::vector<DBObject> parseDashboardObjects(const std::string& view_meta,
                                              const int& user_id);
  void createOrUpdateDashboardSystemRole(const std::string& view_meta,
                                         const int32_t& user_id,
                                         const std::string& dash_role_name);

  const int getColumnIdBySpiUnlocked(const int table_id, const size_t spi) const;

  void serializeTableJsonUnlocked(const TableDescriptor* td,
                                  const std::list<ColumnDescriptor>& cds) const;
  void dropTableFromJsonUnlocked(const std::string& table_name) const;

  std::string basePath_;
  TableDescriptorMap tableDescriptorMap_;
  TableDescriptorMapById tableDescriptorMapById_;
  ColumnDescriptorMap columnDescriptorMap_;
  ColumnDescriptorMapById columnDescriptorMapById_;
  DictDescriptorMapById dictDescriptorMapByRef_;
  DashboardDescriptorMap dashboardDescriptorMap_;
  LinkDescriptorMap linkDescriptorMap_;
  LinkDescriptorMapById linkDescriptorMapById_;
  ForeignServerMap foreignServerMap_;
  ForeignServerMapById foreignServerMapById_;

  SqliteConnector sqliteConnector_;
  const DBMetadata currentDB_;
  std::shared_ptr<Data_Namespace::DataMgr> dataMgr_;

  const std::vector<LeafHostInfo> string_dict_hosts_;
  std::shared_ptr<Calcite> calciteMgr_;

  LogicalToPhysicalTableMapById logicalToPhysicalTableMapById_;
  static const std::string
      physicalTableNameTag_;  // extra component added to the name of each physical table
  int nextTempTableId_;
  int nextTempDictId_;

  // this tuple is for rolling forw/back once after ALTER ADD/DEL/MODIFY columns
  // succeeds/fails
  //	get(0) = old ColumnDescriptor*
  //	get(1) = new ColumnDescriptor*
  using ColumnDescriptorsForRoll =
      std::vector<std::pair<ColumnDescriptor*, ColumnDescriptor*>>;
  ColumnDescriptorsForRoll columnDescriptorsForRoll;

 private:
  DeletedColumnPerTableMap deletedColumnPerTable_;
  void adjustAlteredTableFiles(
      const std::string& temp_data_dir,
      const std::unordered_map<int, int>& all_column_ids_map) const;
  void renameTableDirectories(const std::string& temp_data_dir,
                              const std::vector<std::string>& target_paths,
                              const std::string& name_prefix) const;
  void buildForeignServerMap();
  void addForeignTableDetails();

  void setForeignServerProperty(const std::string& server_name,
                                const std::string& property,
                                const std::string& value);

  void setForeignTableProperty(const foreign_storage::ForeignTable* table,
                               const std::string& property,
                               const std::string& value);

  void alterPhysicalTableMetadata(const TableDescriptor* td,
                                  const TableDescriptorUpdateParams& table_update_params);
  void alterTableMetadata(const TableDescriptor* td,
                          const TableDescriptorUpdateParams& table_update_params);
  void setTableFileMgrParams(const int table_id,
                             const File_Namespace::FileMgrParams& file_mgr_params);
  bool filterTableByTypeAndUser(const TableDescriptor* td,
                                const UserMetadata& user_metadata,
                                const GetTablesType get_tables_type) const;

  TableDescriptor* getMutableMetadataForTableUnlocked(int tableId);

  /**
   * Same as createForeignServer() but without acquiring locks. This should only be called
   * from within a function/code block that already acquires appropriate locks.
   */
  void createForeignServerNoLocks(
      std::unique_ptr<foreign_storage::ForeignServer> foreign_server,
      bool if_not_exists);

  foreign_storage::ForeignTable* getForeignTableUnlocked(
      const std::string& tableName) const;

  const Catalog* getObjForLock();

 public:
  mutable std::mutex sqliteMutex_;
  mutable mapd_shared_mutex sharedMutex_;
  mutable std::atomic<std::thread::id> thread_holding_sqlite_lock;
  mutable std::atomic<std::thread::id> thread_holding_write_lock;
  // assuming that you never call into a catalog from another catalog via the same thread
  static thread_local bool thread_holds_read_lock;
  bool initialized_ = false;
};

}  // namespace Catalog_Namespace
