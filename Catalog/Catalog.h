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

#ifndef CATALOG_H
#define CATALOG_H

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

#include "ColumnDescriptor.h"
#include "DashboardDescriptor.h"
#include "DictDescriptor.h"
#include "LinkDescriptor.h"
#include "TableDescriptor.h"

#include "../DataMgr/DataMgr.h"
#include "../QueryEngine/CompilationOptions.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "LeafHostInfo.h"

#include "../Calcite/Calcite.h"
#include "../Shared/mapd_shared_mutex.h"

#include "SessionInfo.h"
#include "SysCatalog.h"

namespace Parser {

class SharedDictionaryDef;

}  // namespace Parser

// SPI means Sequential Positional Index which is equivalent to the input index in a
// RexInput node
#define SPIMAP_MAGIC1 (std::numeric_limits<unsigned>::max() / 4)
#define SPIMAP_MAGIC2 8
#define SPIMAP_GEO_PHYSICAL_INPUT(c, i) \
  (SPIMAP_MAGIC1 + (unsigned)(SPIMAP_MAGIC2 * ((c) + 1) + (i)))

namespace Catalog_Namespace {

/**
 * @type Catalog
 * @brief class for a per-database catalog.  also includes metadata for the
 * current database and the current user.
 */

class Catalog {
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
  virtual ~Catalog();

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
  void removeChunks(const int table_id);

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
  const TableDescriptor* getMetadataForTable(int tableId) const;

  const ColumnDescriptor* getMetadataForColumn(int tableId,
                                               const std::string& colName) const;
  const ColumnDescriptor* getMetadataForColumn(int tableId, int columnId) const;

  const int getColumnIdBySpi(const int tableId, const size_t spi) const;
  const ColumnDescriptor* getMetadataForColumnBySpi(const int tableId,
                                                    const size_t spi) const;

  const DashboardDescriptor* getMetadataForDashboard(const std::string& userId,
                                                     const std::string& dashName) const;
  void deleteMetadataForDashboard(const std::string& userId, const std::string& dashName);

  const DashboardDescriptor* getMetadataForDashboard(const int32_t dashboard_id) const;
  void deleteMetadataForDashboard(const int32_t dashboard_id);

  const LinkDescriptor* getMetadataForLink(const std::string& link) const;
  const LinkDescriptor* getMetadataForLink(int linkId) const;

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
  std::list<const DashboardDescriptor*> getAllDashboardsMetadata() const;
  const DBMetadata& getCurrentDB() const { return currentDB_; }
  Data_Namespace::DataMgr& getDataMgr() const { return *dataMgr_; }
  std::shared_ptr<Calcite> getCalciteMgr() const { return calciteMgr_; }
  const std::string& getBasePath() const { return basePath_; }

  const DictDescriptor* getMetadataForDict(int dict_ref, bool loadDict = true) const;

  const std::vector<LeafHostInfo>& getStringDictionaryHosts() const;

  const ColumnDescriptor* getShardColumnMetadataForTable(const TableDescriptor* td) const;

  std::vector<const TableDescriptor*> getPhysicalTablesDescriptors(
      const TableDescriptor* logicalTableDesc) const;

  int32_t getTableEpoch(const int32_t db_id, const int32_t table_id) const;
  void setTableEpoch(const int db_id, const int table_id, const int new_epoch);
  int getDatabaseId() const { return currentDB_.dbId; }

  SqliteConnector& getSqliteConnector() { return sqliteConnector_; }
  void roll(const bool forward);
  DictRef addDictionary(ColumnDescriptor& cd);
  void delDictionary(const ColumnDescriptor& cd);
  void getDictionary(const ColumnDescriptor& cd,
                     std::map<int, StringDictionary*>& stringDicts);

  static void set(const std::string& dbName, std::shared_ptr<Catalog> cat);
  static std::shared_ptr<Catalog> get(const std::string& dbName);
  static std::shared_ptr<Catalog> get(const std::string& basePath,
                                      const DBMetadata& curDB,
                                      std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
                                      const std::vector<LeafHostInfo>& string_dict_hosts,
                                      std::shared_ptr<Calcite> calcite,
                                      bool is_new_db);
  static void remove(const std::string& dbName);

  const bool checkMetadataForDeletedRecs(int dbId, int tableId, int columnId) const;
  const ColumnDescriptor* getDeletedColumn(const TableDescriptor* td) const;
  const ColumnDescriptor* getDeletedColumnIfRowsDeleted(const TableDescriptor* td) const;

  void setDeletedColumn(const TableDescriptor* td, const ColumnDescriptor* cd);
  void setDeletedColumnUnlocked(const TableDescriptor* td, const ColumnDescriptor* cd);
  int getLogicalTableId(const int physicalTableId) const;
  void checkpoint(const int logicalTableId) const;
  std::string name() const { return getCurrentDB().dbName; }
  void eraseDBData();
  void eraseTablePhysicalData(const TableDescriptor* td);
  void optimizeTable(const TableDescriptor* td) const;

 protected:
  typedef std::map<std::string, TableDescriptor*> TableDescriptorMap;
  typedef std::map<int, TableDescriptor*> TableDescriptorMapById;
  typedef std::map<int32_t, std::vector<int32_t>> LogicalToPhysicalTableMapById;
  typedef std::tuple<int, std::string> ColumnKey;
  typedef std::map<ColumnKey, ColumnDescriptor*> ColumnDescriptorMap;
  typedef std::tuple<int, int> ColumnIdKey;
  typedef std::map<ColumnIdKey, ColumnDescriptor*> ColumnDescriptorMapById;
  typedef std::map<DictRef, std::unique_ptr<DictDescriptor>> DictDescriptorMapById;
  typedef std::map<std::string, std::shared_ptr<DashboardDescriptor>>
      DashboardDescriptorMap;
  typedef std::map<std::string, LinkDescriptor*> LinkDescriptorMap;
  typedef std::map<int, LinkDescriptor*> LinkDescriptorMapById;
  typedef std::unordered_map<const TableDescriptor*, const ColumnDescriptor*>
      DeletedColumnPerTableMap;

  void CheckAndExecuteMigrations();
  void CheckAndExecuteMigrationsPostBuildMaps();
  void updateDictionaryNames();
  void updateTableDescriptorSchema();
  void updateFrontendViewSchema();
  void updateLinkSchema();
  void updateFrontendViewAndLinkUsers();
  void updateLogicalToPhysicalTableLinkSchema();
  void updateLogicalToPhysicalTableMap(const int32_t logical_tb_id);
  void updateDictionarySchema();
  void updatePageSize();
  void updateDeletedColumnIndicator();
  void updateFrontendViewsToDashboards();
  void recordOwnershipOfObjectsInObjectPermissions();
  void checkDateInDaysColumnMigration();
  void createDashboardSystemRoles();
  void buildMaps();
  void addTableToMap(TableDescriptor& td,
                     const std::list<ColumnDescriptor>& columns,
                     const std::list<DictDescriptor>& dicts);
  void addReferenceToForeignDict(ColumnDescriptor& referencing_column,
                                 Parser::SharedDictionaryDef shared_dict_def);
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
  void removeTableFromMap(const std::string& tableName, int tableId);
  void doDropTable(const TableDescriptor* td);
  void doTruncateTable(const TableDescriptor* td);
  void renamePhysicalTable(const TableDescriptor* td, const std::string& newTableName);
  void instantiateFragmenter(TableDescriptor* td) const;
  void getAllColumnMetadataForTable(const TableDescriptor* td,
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

  std::string basePath_;
  TableDescriptorMap tableDescriptorMap_;
  TableDescriptorMapById tableDescriptorMapById_;
  ColumnDescriptorMap columnDescriptorMap_;
  ColumnDescriptorMapById columnDescriptorMapById_;
  DictDescriptorMapById dictDescriptorMapByRef_;
  DashboardDescriptorMap dashboardDescriptorMap_;
  LinkDescriptorMap linkDescriptorMap_;
  LinkDescriptorMapById linkDescriptorMapById_;
  SqliteConnector sqliteConnector_;
  DBMetadata currentDB_;
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
  static std::map<std::string, std::shared_ptr<Catalog>> mapd_cat_map_;
  DeletedColumnPerTableMap deletedColumnPerTable_;

 public:
  mutable std::mutex sqliteMutex_;
  mutable mapd_shared_mutex sharedMutex_;
  mutable std::atomic<std::thread::id> thread_holding_sqlite_lock;
  mutable std::atomic<std::thread::id> thread_holding_write_lock;
  // assuming that you never call into a catalog from another catalog via the same thread
  static thread_local bool thread_holds_read_lock;
};

}  // namespace Catalog_Namespace

#endif  // CATALOG_H
