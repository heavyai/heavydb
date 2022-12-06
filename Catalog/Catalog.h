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

/**
 * @file    Catalog.h
 * @brief   This file contains the class specification and related data structures for
 * Catalog.
 *
 * This file contains the Catalog class specification. The Catalog class is responsible
 * for storing, accessing and caching metadata for a single database. A global metadata
 * could be accessed with SysCatalog class.
 *
 */

#pragma once

#include <array>
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
#include "Catalog/CustomExpression.h"
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
#include "OSDependent/heavyai_locks.h"
#include "QueryEngine/CompilationOptions.h"
#include "Shared/heavyai_shared_mutex.h"
#include "SqliteConnector/SqliteConnector.h"

#include "LeafHostInfo.h"

enum GetTablesType { GET_PHYSICAL_TABLES_AND_VIEWS, GET_PHYSICAL_TABLES, GET_VIEWS };

namespace lockmgr {

template <class T>
class TableLockMgrImpl;

}  // namespace lockmgr

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

static constexpr const char* USERS_SYS_TABLE_NAME{"users"};
static constexpr const char* TABLES_SYS_TABLE_NAME{"tables"};
static constexpr const char* DASHBOARDS_SYS_TABLE_NAME{"dashboards"};
static constexpr const char* DATABASES_SYS_TABLE_NAME{"databases"};
static constexpr const char* PERMISSIONS_SYS_TABLE_NAME{"permissions"};
static constexpr const char* ROLES_SYS_TABLE_NAME{"roles"};
static constexpr const char* ROLE_ASSIGNMENTS_SYS_TABLE_NAME{"role_assignments"};
static constexpr const char* MEMORY_SUMMARY_SYS_TABLE_NAME{"memory_summary"};
static constexpr const char* MEMORY_DETAILS_SYS_TABLE_NAME{"memory_details"};
static constexpr const char* STORAGE_DETAILS_SYS_TABLE_NAME{"storage_details"};
static constexpr const char* SERVER_LOGS_SYS_TABLE_NAME{"server_logs"};
static constexpr const char* REQUEST_LOGS_SYS_TABLE_NAME{"request_logs"};
static constexpr const char* WS_SERVER_LOGS_SYS_TABLE_NAME{"web_server_logs"};
static constexpr const char* WS_SERVER_ACCESS_LOGS_SYS_TABLE_NAME{
    "web_server_access_logs"};

static const std::array<std::string, 4> kAggregatorOnlySystemTables{
    DASHBOARDS_SYS_TABLE_NAME,
    REQUEST_LOGS_SYS_TABLE_NAME,
    WS_SERVER_LOGS_SYS_TABLE_NAME,
    WS_SERVER_ACCESS_LOGS_SYS_TABLE_NAME};

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
  void renameTable(std::vector<std::pair<std::string, std::string>>& names);
  void renameColumn(const TableDescriptor* td,
                    const ColumnDescriptor* cd,
                    const std::string& newColumnName);
  void addColumn(const TableDescriptor& td, ColumnDescriptor& cd);
  void dropColumn(const TableDescriptor& td, const ColumnDescriptor& cd);
  void invalidateCachesForTable(const int table_id);
  void removeFragmenterForTable(const int table_id) const;

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
  const TableDescriptor* getMetadataForTable(int tableId,
                                             bool populateFragmenter = true) const;

  std::optional<std::string> getTableName(int32_t table_id) const;
  std::optional<int32_t> getTableId(const std::string& table_name) const;

  const ColumnDescriptor* getMetadataForColumn(int tableId,
                                               const std::string& colName) const;
  const ColumnDescriptor* getMetadataForColumn(int tableId, int columnId) const;
  const std::optional<std::string> getColumnName(int table_id, int column_id) const;

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

  std::list<const TableDescriptor*> getAllTableMetadata() const;
  std::vector<TableDescriptor> getAllTableMetadataCopy() const;
  std::list<const DashboardDescriptor*> getAllDashboardsMetadata() const;
  std::vector<DashboardDescriptor> getAllDashboardsMetadataCopy() const;
  const DBMetadata& getCurrentDB() const { return currentDB_; }
  Data_Namespace::DataMgr& getDataMgr() const { return *dataMgr_; }
  std::shared_ptr<Calcite> getCalciteMgr() const { return calciteMgr_; }
  const std::string& getCatalogBasePath() const { return basePath_; }

  const DictDescriptor* getMetadataForDict(int dict_ref, bool loadDict = true) const;

  const std::vector<LeafHostInfo>& getStringDictionaryHosts() const;

  const ColumnDescriptor* getShardColumnMetadataForTable(const TableDescriptor* td) const;

  std::vector<const TableDescriptor*> getPhysicalTablesDescriptors(
      const TableDescriptor* logical_table_desc,
      bool populate_fragmenter = true) const;

  std::vector<std::pair<int32_t, int32_t>> getAllPersistedTableAndShardIds() const;

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
  void setMaxRows(const int32_t table_id, const int64_t max_rows);

  std::vector<TableEpochInfo> getTableEpochs(const int32_t db_id,
                                             const int32_t table_id) const;
  void setTableEpochs(const int32_t db_id,
                      const std::vector<TableEpochInfo>& table_epochs) const;

  void setTableEpochsLogExceptions(const int32_t db_id,
                                   const std::vector<TableEpochInfo>& table_epochs) const;

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
  int getLogicalTableId(const int physicalTableId) const;
  void checkpoint(const int logicalTableId) const;
  void checkpointWithAutoRollback(const int logical_table_id) const;
  void resetTableEpochFloor(const int logicalTableId) const;
  std::string name() const { return getCurrentDB().dbName; }
  void eraseDbMetadata();
  void eraseDbPhysicalData();
  void eraseTablePhysicalData(const TableDescriptor* td);
  void setForReload(const int32_t tableId);

  std::vector<std::string> getTableDataDirectories(const TableDescriptor* td) const;
  std::vector<std::string> getTableDictDirectories(const TableDescriptor* td) const;
  std::set<std::string> getTableDictDirectoryPaths(int32_t table_id) const;
  std::string getColumnDictDirectory(const ColumnDescriptor* cd,
                                     bool file_name_only = true) const;
  std::string dumpSchema(const TableDescriptor* td) const;
  std::string dumpCreateTable(const TableDescriptor* td,
                              bool multiline_formatting = true,
                              bool dump_defaults = false) const;
  std::optional<std::string> dumpCreateTable(int32_t table_id,
                                             bool multiline_formatting = true,
                                             bool dump_defaults = false) const;
  std::string dumpCreateServer(const std::string& name,
                               bool multiline_formatting = true) const;

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
   * @brief Get all of the foreign tables for associated with a foreign server id
   *
   * @param foreign_server_id - id of foreign server
   * @return std::vector<const ForeignTable*> a vector containing foreign tables which
   * use the foreign server
   */
  std::vector<const foreign_storage::ForeignTable*> getAllForeignTablesForForeignServer(
      const int32_t foreign_server_id);

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

  // For testing purposes only
  void setUncappedTableEpoch(const std::string& table_name);

  // Methods for accessing and updating custom expressions

  /**
   * Gets the DDL statement used to create the custom expressions table.
   *
   * @param if_not_exists - flag the indicates whether or not to include the "IF NOT
   * EXISTS" phrase in the DDL statement.
   * @return string containing DDL statement
   */
  static const std::string getCustomExpressionsSchema(bool if_not_exists = false);

  /**
   * Creates a new custom expression.
   *
   * @param custom_expression - unique pointer to struct containing custom expression
   * details.
   * @return id of created custom expression
   */
  int32_t createCustomExpression(std::unique_ptr<CustomExpression> custom_expression);

  /**
   * Gets a pointer to the custom expression object with the given id.
   *
   * @param custom_expression_id - id of custom expression to get
   * @return pointer to custom expression object. nullptr is returned if no custom
   * expression is found for the given id.
   */
  const CustomExpression* getCustomExpression(int32_t custom_expression_id) const;

  /**
   * Gets a pointer to a struct containing custom expression details fetched from storage.
   * This is intended for use in tests, when asserting that expected custom expression
   * data is persisted.
   *
   * @param custom_expression_id - id of custom expression to get
   * @return pointer to custom expression object. nullptr is returned if no custom
   * expression is found for the given id.
   */
  const std::unique_ptr<const CustomExpression> getCustomExpressionFromStorage(
      int32_t custom_expression_id);

  /**
   * Gets pointers to all the custom expression objects that the given user has access to.
   * For custom expressions that are associated with a table data source, custom
   * expressions for tables that the given user has SELECT access to are returned.
   *
   * @param user - user for which to get accessible custom expressions
   * @return pointer to custom expression objects that the given user has access to
   */
  std::vector<const CustomExpression*> getCustomExpressionsForUser(
      const UserMetadata& user) const;

  /**
   * Updates the custom expression for the given id with the given expression json string.
   *
   * @param custom_expression_id - id of custom expression to update
   * @param expression_json - expression json string to be set
   */
  void updateCustomExpression(int32_t custom_expression_id,
                              const std::string& expression_json);

  /**
   * Deletes custom expressions with the given ids.
   *
   * @param custom_expression_ids - ids of custom expressions to delete
   * @param do_soft_delete - flag indicating whether or not to do a soft delete
   */
  void deleteCustomExpressions(const std::vector<int32_t>& custom_expression_ids,
                               bool do_soft_delete);

  /**
   * Reassigns database object ownership from a set of users (old owners) to another user
   * (new owner).
   *
   * @param old_owners - users whose database object ownership will be reassigned to a new
   * user
   * @param new_owner - user who will own reassigned database objects
   */
  void reassignOwners(const std::set<std::string>& old_owners,
                      const std::string& new_owner);

  bool isInfoSchemaDb() const;

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
  void updateDefaultColumnValues();
  void updateFrontendViewsToDashboards();
  void updateCustomExpressionsSchema();
  void updateFsiSchemas();
  void renameLegacyDataWrappers();
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
                           bool is_logical_table,
                           bool use_temp_dictionary = false);
  void addFrontendViewToMap(DashboardDescriptor& vd);
  void addFrontendViewToMapNoLock(DashboardDescriptor& vd);
  void addLinkToMap(LinkDescriptor& ld);
  void removeTableFromMap(const std::string& tableName,
                          const int tableId,
                          const bool is_on_error = false);
  void eraseTableMetadata(const TableDescriptor* td);
  void executeDropTableSqliteQueries(const TableDescriptor* td);
  void doTruncateTable(const TableDescriptor* td);
  void renamePhysicalTable(const TableDescriptor* td, const std::string& newTableName);
  void renamePhysicalTable(std::vector<std::pair<std::string, std::string>>& names,
                           std::vector<int>& tableIds);
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
  CustomExpressionMapById custom_expr_map_by_id_;
  TableDictColumnsMap dict_columns_by_table_id_;

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
  void buildDictionaryMapUnlocked();
  void reloadTableMetadata(int table_id);
  template <class T>
  friend class lockmgr::TableLockMgrImpl;  // for reloadTableMetadataUnlocked()
  void reloadTableMetadataUnlocked(int table_id);
  void reloadCatalogMetadata(const std::map<int32_t, std::string>& user_name_by_user_id);
  void reloadCatalogMetadataUnlocked(
      const std::map<int32_t, std::string>& user_name_by_user_id);
  void refreshDictionaryCachesForTableUnlocked(const TableDescriptor& td);
  void buildTablesMapUnlocked();
  void buildColumnsMapUnlocked();
  void updateViewsInMapUnlocked();
  void buildDashboardsMapUnlocked(
      const std::map<int32_t, std::string>& user_name_by_user_id);
  void buildLinksMapUnlocked();
  void buildLogicalToPhysicalMapUnlocked();
  void updateForeignTablesInMapUnlocked();

  void gatherAdditionalInfo(std::vector<std::string>& additional_info,
                            std::set<std::string>& shared_dict_column_names,
                            const TableDescriptor* td) const;
  std::string quoteIfRequired(const std::string& column_name) const;
  DeletedColumnPerTableMap deletedColumnPerTable_;
  void adjustAlteredTableFiles(
      const std::string& temp_data_dir,
      const std::unordered_map<int, int>& all_column_ids_map) const;
  void renameTableDirectories(const std::string& temp_data_dir,
                              const std::vector<std::string>& target_paths,
                              const std::string& name_prefix) const;
  void buildForeignServerMapUnlocked();

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

  TableDescriptor* getMutableMetadataForTableUnlocked(int table_id) const;
  TableDescriptor* getMutableMetadataForTableUnlocked(
      const std::string& table_name) const;

  /**
   * Same as createForeignServer() but without acquiring locks. This should only be called
   * from within a function/code block that already acquires appropriate locks.
   */
  void createForeignServerNoLocks(
      std::unique_ptr<foreign_storage::ForeignServer> foreign_server,
      bool if_not_exists);

  foreign_storage::ForeignTable* getForeignTableUnlocked(
      const std::string& tableName) const;

  void removeChunks(const int table_id) const;

  void buildCustomExpressionsMapUnlocked();
  std::unique_ptr<CustomExpression> getCustomExpressionFromConnector(size_t row);

  void restoreOldOwners(
      const std::map<int32_t, std::string>& old_owners_user_name_by_id,
      const std::map<int32_t, std::vector<DBObject>>& old_owner_db_objects,
      int32_t new_owner_id);
  void restoreOldOwnersInMemory(
      const std::map<int32_t, std::string>& old_owners_user_name_by_id,
      const std::map<int32_t, std::vector<DBObject>>& old_owner_db_objects,
      int32_t new_owner_id);

  void conditionallyInitializeSystemObjects();
  void initializeSystemServers();
  void initializeSystemTables();
  void initializeUsersSystemTable();
  void initializeDatabasesSystemTable();
  void initializePermissionsSystemTable();
  void initializeRolesSystemTable();
  void initializeTablesSystemTable();
  void initializeDashboardsSystemTable();
  void initializeRoleAssignmentsSystemTable();
  void initializeMemorySummarySystemTable();
  void initializeMemoryDetailsSystemTable();
  void initializeStorageDetailsSystemTable();
  void initializeServerLogsSystemTables();
  void initializeRequestLogsSystemTables();
  void initializeWebServerLogsSystemTables();
  void initializeWebServerAccessLogsSystemTables();

  void createSystemTableServer(const std::string& server_name,
                               const std::string& data_wrapper_type,
                               const foreign_storage::OptionsMap& options = {});
  std::pair<foreign_storage::ForeignTable, std::list<ColumnDescriptor>>
  getSystemTableSchema(
      const std::string& table_name,
      const std::string& server_name,
      const std::vector<std::pair<std::string, SQLTypeInfo>>& column_type_by_name,
      bool is_in_memory_system_table);

  bool recreateSystemTableIfUpdated(foreign_storage::ForeignTable& foreign_table,
                                    const std::list<ColumnDescriptor>& columns);

  void setDeletedColumnUnlocked(const TableDescriptor* td, const ColumnDescriptor* cd);

  void addToColumnMap(ColumnDescriptor* cd);
  void removeFromColumnMap(ColumnDescriptor* cd);

  void deleteTableCatalogMetadata(
      const TableDescriptor* logical_table,
      const std::vector<const TableDescriptor*>& physical_tables);

  std::string dumpCreateTableUnlocked(const TableDescriptor* td,
                                      bool multiline_formatting,
                                      bool dump_defaults) const;

  static constexpr const char* CATALOG_SERVER_NAME{"system_catalog_server"};
  static constexpr const char* MEMORY_STATS_SERVER_NAME{"system_memory_stats_server"};
  static constexpr const char* STORAGE_STATS_SERVER_NAME{"system_storage_stats_server"};
  static constexpr const char* LOGS_SERVER_NAME{"system_logs_server"};
  static constexpr std::array<const char*, 4> INTERNAL_SERVERS{CATALOG_SERVER_NAME,
                                                               MEMORY_STATS_SERVER_NAME,
                                                               STORAGE_STATS_SERVER_NAME,
                                                               LOGS_SERVER_NAME};

 public:
  mutable std::unique_ptr<heavyai::DistributedSharedMutex> dcatalogMutex_;
  mutable std::unique_ptr<heavyai::DistributedSharedMutex> dsqliteMutex_;
  mutable std::mutex sqliteMutex_;
  mutable heavyai::shared_mutex sharedMutex_;
  mutable std::atomic<std::thread::id> thread_holding_sqlite_lock;
  mutable std::atomic<std::thread::id> thread_holding_write_lock;
  // assuming that you never call into a catalog from another catalog via the same thread
  static thread_local bool thread_holds_read_lock;
  bool initialized_ = false;
};

}  // namespace Catalog_Namespace
