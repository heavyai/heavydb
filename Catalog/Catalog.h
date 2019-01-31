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
 * for storing metadata about stored objects in the system (currently just relations).  At
 * this point it does not take advantage of the database storage infrastructure; this
 * likely will change in the future as the buildout continues. Although it persists the
 * metainfo on disk, at database startup it reads everything into in-memory dictionaries
 * for fast access.
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
#include "DictDescriptor.h"
#include "FrontendViewDescriptor.h"
#include "Grantee.h"
#include "LdapServer.h"
#include "LinkDescriptor.h"
#include "ObjectRoleDescriptor.h"
#include "RestServer.h"
#include "TableDescriptor.h"

#include "../DataMgr/DataMgr.h"
#include "../QueryEngine/CompilationOptions.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "LeafHostInfo.h"

#include "../Calcite/Calcite.h"
#include "../Shared/mapd_shared_mutex.h"

struct Privileges {
  bool super_;
  bool select_;
  bool insert_;
};

namespace Parser {

class SharedDictionaryDef;

}  // namespace Parser

namespace Importer_NS {
class Loader;
class TypedImportBuffer;
}  // namespace Importer_NS

// SPI means Sequential Positional Index which is equivalent to the input index in a
// RexInput node
#define SPIMAP_MAGIC1 (std::numeric_limits<unsigned>::max() / 4)
#define SPIMAP_MAGIC2 8
#define SPIMAP_GEO_PHYSICAL_INPUT(c, i) \
  (SPIMAP_MAGIC1 + (unsigned)(SPIMAP_MAGIC2 * ((c) + 1) + (i)))

namespace Catalog_Namespace {

/*
 * @type UserMetadata
 * @brief metadata for a mapd user
 */
struct UserMetadata {
  UserMetadata(int32_t u, const std::string& n, const std::string& p, bool s)
      : userId(u), userName(n), passwd_hash(p), isSuper(s), isReallySuper(s) {}
  UserMetadata() {}
  int32_t userId;
  std::string userName;
  std::string passwd_hash;
  bool isSuper;
  bool isReallySuper;
};

/*
 * @type DBMetadata
 * @brief metadata for a mapd database
 */
struct DBMetadata {
  DBMetadata() : dbId(0), dbOwner(0) {}
  int32_t dbId;
  std::string dbName;
  int32_t dbOwner;
};

/* database name for the system database */
#define MAPD_SYSTEM_DB "mapd"
/* the mapd root user */
#define MAPD_ROOT_USER "mapd"

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
  int32_t createFrontendView(FrontendViewDescriptor& vd);
  void replaceDashboard(FrontendViewDescriptor& vd);
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

  const FrontendViewDescriptor* getMetadataForFrontendView(
      const std::string& userId,
      const std::string& viewName) const;
  void deleteMetadataForFrontendView(const std::string& userId,
                                     const std::string& viewName);

  const FrontendViewDescriptor* getMetadataForDashboard(const int32_t dashboard_id) const;
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
  std::list<const FrontendViewDescriptor*> getAllFrontendViewMetadata() const;
  const DBMetadata& getCurrentDB() const { return currentDB_; }
  Data_Namespace::DataMgr& getDataMgr() const { return *dataMgr_; }
  Calcite& getCalciteMgr() const { return *calciteMgr_; }
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
  typedef std::map<std::string, std::shared_ptr<FrontendViewDescriptor>>
      FrontendViewDescriptorMap;
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
  void addFrontendViewToMap(FrontendViewDescriptor& vd);
  void addFrontendViewToMapNoLock(FrontendViewDescriptor& vd);
  void addLinkToMap(LinkDescriptor& ld);
  void removeTableFromMap(const std::string& tableName, int tableId);
  void doDropTable(const TableDescriptor* td, SqliteConnector* conn);
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
  FrontendViewDescriptorMap dashboardDescriptorMap_;
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

/*
 * @type SysCatalog
 * @brief class for the system-wide catalog, currently containing user and database
 * metadata
 */
class SysCatalog {
 public:
  void init(const std::string& basePath,
            std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
            AuthMetadata authMetadata,
            std::shared_ptr<Calcite> calcite,
            bool is_new_db,
            bool check_privileges,
            bool aggregator,
            const std::vector<LeafHostInfo>& string_dict_hosts);

  /**
   * logins (connects) a user against a database.
   *
   * throws a std::exception in all error cases! (including wrong password)
   */
  std::shared_ptr<Catalog> login(const std::string& db,
                                 std::string& username,
                                 const std::string& password,
                                 UserMetadata& user_meta,
                                 bool check_password = true);
  void createUser(const std::string& name, const std::string& passwd, bool issuper);
  void dropUser(const std::string& name);
  void alterUser(const int32_t userid, const std::string* passwd, bool* issuper);
  void grantPrivileges(const int32_t userid, const int32_t dbid, const Privileges& privs);
  bool checkPrivileges(UserMetadata& user, DBMetadata& db, const Privileges& wants_privs);
  void createDatabase(const std::string& dbname, int owner);
  void dropDatabase(const DBMetadata& db);
  bool getMetadataForUser(const std::string& name, UserMetadata& user);
  bool getMetadataForUserById(const int32_t idIn, UserMetadata& user);
  bool checkPasswordForUser(const std::string& passwd,
                            std::string& name,
                            UserMetadata& user);
  bool getMetadataForDB(const std::string& name, DBMetadata& db);
  const DBMetadata& getCurrentDB() const { return currentDB_; }
  Data_Namespace::DataMgr& getDataMgr() const { return *dataMgr_; }
  Calcite& getCalciteMgr() const { return *calciteMgr_; }
  const std::string& getBasePath() const { return basePath_; }
  SqliteConnector* getSqliteConnector() { return sqliteConnector_.get(); }
  std::list<DBMetadata> getAllDBMetadata();
  std::list<UserMetadata> getAllUserMetadata();
  /**
   * return the users associated with the given DB
   */
  std::list<UserMetadata> getAllUserMetadata(long dbId);
  void createDBObject(const UserMetadata& user,
                      const std::string& objectName,
                      DBObjectType type,
                      const Catalog_Namespace::Catalog& catalog,
                      int32_t objectId = -1);
  void grantDBObjectPrivileges(const std::string& grantee,
                               const DBObject& object,
                               const Catalog_Namespace::Catalog& catalog);
  void grantDBObjectPrivilegesBatch(const std::vector<std::string>& grantees,
                                    const std::vector<DBObject>& objects,
                                    const Catalog_Namespace::Catalog& catalog);
  void revokeDBObjectPrivileges(const std::string& grantee,
                                const DBObject& object,
                                const Catalog_Namespace::Catalog& catalog);
  void revokeDBObjectPrivilegesBatch(const std::vector<std::string>& grantees,
                                     const std::vector<DBObject>& objects,
                                     const Catalog_Namespace::Catalog& catalog);
  void revokeDBObjectPrivilegesFromAll_unsafe(DBObject object, Catalog* catalog);
  void getDBObjectPrivileges(const std::string& granteeName,
                             DBObject& object,
                             const Catalog_Namespace::Catalog& catalog) const;
  bool verifyDBObjectOwnership(const UserMetadata& user,
                               DBObject object,
                               const Catalog_Namespace::Catalog& catalog);
  void createRole(const std::string& roleName, const bool& userPrivateRole = false);
  void dropRole(const std::string& roleName);
  void grantRoleBatch(const std::vector<std::string>& roles,
                      const std::vector<std::string>& grantees);
  void grantRole(const std::string& role, const std::string& grantee);
  void revokeRoleBatch(const std::vector<std::string>& roles,
                       const std::vector<std::string>& grantees);
  void revokeRole(const std::string& role, const std::string& grantee);
  // check if the user has any permissions on all the given objects
  bool hasAnyPrivileges(const UserMetadata& user, std::vector<DBObject>& privObjects);
  // check if the user has the requested permissions on all the given objects
  bool checkPrivileges(const UserMetadata& user,
                       const std::vector<DBObject>& privObjects) const;
  bool checkPrivileges(const std::string& userName,
                       const std::vector<DBObject>& privObjects) const;
  Grantee* getGrantee(const std::string& name) const;
  Role* getRoleGrantee(const std::string& name) const;
  User* getUserGrantee(const std::string& name) const;
  std::vector<ObjectRoleDescriptor*> getMetadataForObject(int32_t dbId,
                                                          int32_t dbType,
                                                          int32_t objectId) const;
  bool isRoleGrantedToGrantee(const std::string& granteeName,
                              const std::string& roleName,
                              bool only_direct) const;
  std::vector<std::string> getRoles(bool userPrivateRole,
                                    bool isSuper,
                                    const std::string& userName);
  std::vector<std::string> getRoles(const std::string& userName, const int32_t dbId);
  bool arePrivilegesOn() const { return check_privileges_; }
  bool isAggregator() const { return aggregator_; }
  static SysCatalog& instance() {
    static SysCatalog sys_cat{};
    return sys_cat;
  }

  void populateRoleDbObjects(const std::vector<DBObject>& objects);
  std::string name() const { return MAPD_SYSTEM_DB; }
  void renameObjectsInDescriptorMap(DBObject& object,
                                    const Catalog_Namespace::Catalog& cat);
  void syncUserWithRemoteProvider(const std::string& user_name,
                                  const std::vector<std::string>& roles,
                                  bool* issuper);

 private:
  typedef std::map<std::string, Grantee*> GranteeMap;
  typedef std::multimap<std::string, ObjectRoleDescriptor*> ObjectRoleDescriptorMap;

  SysCatalog()
      : aggregator_(false)
      , sqliteMutex_()
      , sharedMutex_()
      , thread_holding_sqlite_lock(std::thread::id())
      , thread_holding_write_lock(std::thread::id()) {}
  virtual ~SysCatalog();

  void initDB();
  void buildRoleMap();
  void buildUserRoleMap();
  void buildObjectDescriptorMap();
  void checkAndExecuteMigrations();
  void createUserRoles();
  void migratePrivileges();
  void migratePrivileged_old();
  void updatePasswordsToHashes();
  void migrateDBAccessPrivileges();

  // Here go functions not wrapped into transactions (necessary for nested calls)
  void grantDefaultPrivilegesToRole_unsafe(const std::string& name, bool issuper);
  void createRole_unsafe(const std::string& roleName,
                         const bool& userPrivateRole = false);
  void dropRole_unsafe(const std::string& roleName);
  void grantRoleBatch_unsafe(const std::vector<std::string>& roles,
                             const std::vector<std::string>& grantees);
  void grantRole_unsafe(const std::string& roleName, const std::string& granteeName);
  void revokeRoleBatch_unsafe(const std::vector<std::string>& roles,
                              const std::vector<std::string>& grantees);
  void revokeRole_unsafe(const std::string& roleName, const std::string& granteeName);
  void updateObjectDescriptorMap(const std::string& roleName,
                                 DBObject& object,
                                 bool roleType,
                                 const Catalog_Namespace::Catalog& cat);
  void deleteObjectDescriptorMap(const std::string& roleName);
  void deleteObjectDescriptorMap(const std::string& roleName,
                                 DBObject& object,
                                 const Catalog_Namespace::Catalog& cat);
  void grantDBObjectPrivilegesBatch_unsafe(const std::vector<std::string>& grantees,
                                           const std::vector<DBObject>& objects,
                                           const Catalog_Namespace::Catalog& catalog);
  void grantDBObjectPrivileges_unsafe(const std::string& granteeName,
                                      const DBObject object,
                                      const Catalog_Namespace::Catalog& catalog);
  void revokeDBObjectPrivilegesBatch_unsafe(const std::vector<std::string>& grantees,
                                            const std::vector<DBObject>& objects,
                                            const Catalog_Namespace::Catalog& catalog);
  void revokeDBObjectPrivileges_unsafe(const std::string& granteeName,
                                       DBObject object,
                                       const Catalog_Namespace::Catalog& catalog);
  void grantAllOnDatabase_unsafe(const std::string& roleName,
                                 DBObject& object,
                                 const Catalog_Namespace::Catalog& catalog);
  void revokeAllOnDatabase_unsafe(const std::string& roleName,
                                  int32_t dbId,
                                  Grantee* grantee);
  bool isDashboardSystemRole(const std::string& roleName);

  template <typename F, typename... Args>
  void execInTransaction(F&& f, Args&&... args) {
    sqliteConnector_->query("BEGIN TRANSACTION");
    try {
      (this->*f)(std::forward<Args>(args)...);
    } catch (std::exception&) {
      sqliteConnector_->query("ROLLBACK TRANSACTION");
      throw;
    }
    sqliteConnector_->query("END TRANSACTION");
  }

  bool check_privileges_;
  std::string basePath_;
  GranteeMap granteeMap_;
  ObjectRoleDescriptorMap objectDescriptorMap_;
  DBMetadata currentDB_;
  std::unique_ptr<SqliteConnector> sqliteConnector_;

  std::shared_ptr<Data_Namespace::DataMgr> dataMgr_;
  std::unique_ptr<LdapServer> ldap_server_;
  std::unique_ptr<RestServer> rest_server_;
  std::shared_ptr<Calcite> calciteMgr_;
  std::vector<LeafHostInfo> string_dict_hosts_;
  bool aggregator_;

 public:
  mutable std::mutex sqliteMutex_;
  mutable mapd_shared_mutex sharedMutex_;
  mutable std::atomic<std::thread::id> thread_holding_sqlite_lock;
  mutable std::atomic<std::thread::id> thread_holding_write_lock;
  static thread_local bool thread_holds_read_lock;
};

// this class is defined to accommodate both Thrift and non-Thrift builds.
class MapDHandler {
 public:
  virtual void prepare_columnar_loader(
      const std::string& session,
      const std::string& table_name,
      size_t num_cols,
      std::unique_ptr<Importer_NS::Loader>* loader,
      std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>>* import_buffers);
  virtual ~MapDHandler() {}
};

/*
 * @type SessionInfo
 * @brief a user session
 */
class SessionInfo {
 public:
  SessionInfo(std::shared_ptr<MapDHandler> mapdHandler,
              std::shared_ptr<Catalog> cat,
              const UserMetadata& user,
              const ExecutorDeviceType t,
              const std::string& sid)
      : mapdHandler_(mapdHandler)
      , catalog_(cat)
      , currentUser_(user)
      , executor_device_type_(t)
      , session_id(sid)
      , last_used_time(time(0))
      , start_time(time(0)) {}
  SessionInfo(std::shared_ptr<Catalog> cat,
              const UserMetadata& user,
              const ExecutorDeviceType t,
              const std::string& sid)
      : SessionInfo(std::make_shared<MapDHandler>(), cat, user, t, sid) {}
  SessionInfo(const SessionInfo& s)
      : mapdHandler_(s.mapdHandler_)
      , catalog_(s.catalog_)
      , currentUser_(s.currentUser_)
      , executor_device_type_(static_cast<ExecutorDeviceType>(s.executor_device_type_))
      , session_id(s.session_id) {}
  MapDHandler* get_mapdHandler() const { return mapdHandler_.get(); };
  Catalog& getCatalog() const { return *catalog_; }
  std::shared_ptr<Catalog> get_catalog_ptr() const { return catalog_; }
  const UserMetadata& get_currentUser() const { return currentUser_; }
  const ExecutorDeviceType get_executor_device_type() const {
    return executor_device_type_;
  }
  void set_executor_device_type(ExecutorDeviceType t) { executor_device_type_ = t; }
  std::string get_session_id() const { return session_id; }
  time_t get_last_used_time() const { return last_used_time; }
  void update_last_used_time() { last_used_time = time(0); }
  void reset_superuser() { currentUser_.isSuper = currentUser_.isReallySuper; }
  void make_superuser() { currentUser_.isSuper = true; }
  bool checkDBAccessPrivileges(const DBObjectType& permissionType,
                               const AccessPrivileges& privs,
                               const std::string& objectName = "") const;
  time_t get_start_time() const { return start_time; }

  operator std::string() const;

 private:
  std::shared_ptr<MapDHandler> mapdHandler_;
  std::shared_ptr<Catalog> catalog_;
  UserMetadata currentUser_;
  std::atomic<ExecutorDeviceType> executor_device_type_;
  const std::string session_id;
  std::atomic<time_t> last_used_time;  // for cleaning up SessionInfo after client dies
  std::atomic<time_t> start_time;      // for invalidating session after tolerance period
};

}  // namespace Catalog_Namespace

std::ostream& operator<<(std::ostream& os,
                         const Catalog_Namespace::SessionInfo& session_info);

#endif  // CATALOG_H
