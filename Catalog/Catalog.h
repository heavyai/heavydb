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
 * @brief   This file contains the class specification and related data structures for Catalog.
 *
 * This file contains the Catalog class specification. The Catalog class is responsible for storing metadata
 * about stored objects in the system (currently just relations).  At this point it does not take advantage of the
 * database storage infrastructure; this likely will change in the future as the buildout continues. Although it
 * persists the metainfo on disk, at database startup it reads everything into in-memory dictionaries for fast access.
 *
 */

#ifndef CATALOG_H
#define CATALOG_H

#include <atomic>
#include <cstdint>
#include <ctime>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "ColumnDescriptor.h"
#include "DictDescriptor.h"
#include "FrontendViewDescriptor.h"
#include "LdapServer.h"
#include "LinkDescriptor.h"
#include "Role.h"
#include "TableDescriptor.h"

#include "../DataMgr/DataMgr.h"
#include "../QueryEngine/CompilationOptions.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "LeafHostInfo.h"

#include "../Calcite/Calcite.h"

struct Privileges {
  bool super_;
  bool select_;
  bool insert_;
};

namespace Parser {

class SharedDictionaryDef;

}  // namespace Parser

namespace Catalog_Namespace {

/*
 * @type UserMetadata
 * @brief metadata for a mapd user
 */
struct UserMetadata {
  UserMetadata(int32_t u, const std::string& n, const std::string& p, bool s)
      : userId(u), userName(n), passwd(p), isSuper(s) {}
  UserMetadata() {}
  int32_t userId;
  std::string userName;
  std::string passwd;
  bool isSuper;
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
  Catalog(const std::string& basePath,
          const std::string& dbname,
          std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
          const std::vector<LeafHostInfo>& string_dict_hosts,
          LdapMetadata ldapMetadata,
          bool is_initdb,
          std::shared_ptr<Calcite> calcite);

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
          std::shared_ptr<Calcite> calcite);

  /*
   builds a catalog that uses an ldap server
   */
  Catalog(const std::string& basePath,
          const DBMetadata& curDB,
          std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
          LdapMetadata ldapMetadata,
          std::shared_ptr<Calcite> calcite);

  /**
   * @brief Destructor - deletes all
   * ColumnDescriptor and TableDescriptor structures
   * which were allocated on the heap and writes
   * Catalog to Sqlite
   */
  virtual ~Catalog();

  void createTable(TableDescriptor& td,
                   const std::list<ColumnDescriptor>& columns,
                   const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs,
                   bool isLogicalTable);
  void createShardedTable(TableDescriptor& td,
                          const std::list<ColumnDescriptor>& columns,
                          const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs);
  void createFrontendView(FrontendViewDescriptor& vd);
  std::string createLink(LinkDescriptor& ld, size_t min_length);
  void dropTable(const TableDescriptor* td);
  void truncateTable(const TableDescriptor* td);
  void renameTable(const TableDescriptor* td, const std::string& newTableName);
  void renameColumn(const TableDescriptor* td, const ColumnDescriptor* cd, const std::string& newColumnName);

  void removeChunks(const int table_id);

  /**
   * @brief Returns a pointer to a const TableDescriptor struct matching
   * the provided tableName
   * @param tableName table specified column belongs to
   * @return pointer to const TableDescriptor object queried for or nullptr if it does not exist.
   */

  const TableDescriptor* getMetadataForTable(const std::string& tableName, const bool populateFragmenter = true) const;
  const TableDescriptor* getMetadataForTable(int tableId) const;

  const ColumnDescriptor* getMetadataForColumn(int tableId, const std::string& colName) const;
  const ColumnDescriptor* getMetadataForColumn(int tableId, int columnId) const;

  const FrontendViewDescriptor* getMetadataForFrontendView(const std::string& userId,
                                                           const std::string& viewName) const;
  const FrontendViewDescriptor* getMetadataForFrontendView(int viewId) const;

  void deleteMetadataForFrontendView(const std::string& userId, const std::string& viewName);

  const LinkDescriptor* getMetadataForLink(const std::string& link) const;
  const LinkDescriptor* getMetadataForLink(int linkId) const;

  /**
   * @brief Returns a list of pointers to constant ColumnDescriptor structs for all the columns from a particular table
   * specified by table id
   * @param tableId table id we want the column metadata for
   * @return list of pointers to const ColumnDescriptor structs - one
   * for each and every column in the table
   *
   */

  std::list<const ColumnDescriptor*> getAllColumnMetadataForTable(const int tableId,
                                                                  const bool fetchSystemColumns,
                                                                  const bool fetchVirtualColumns) const;

  std::list<const TableDescriptor*> getAllTableMetadata() const;
  std::list<const FrontendViewDescriptor*> getAllFrontendViewMetadata() const;
  const DBMetadata& get_currentDB() const { return currentDB_; }
  void set_currentDB(const DBMetadata& db) { currentDB_ = db; }
  Data_Namespace::DataMgr& get_dataMgr() const { return *dataMgr_; }
  Calcite& get_calciteMgr() const { return *calciteMgr_; }
  const std::string& get_basePath() const { return basePath_; }

  const DictDescriptor* getMetadataForDict(int dict_ref, bool loadDict = true) const;

  const std::vector<LeafHostInfo>& getStringDictionaryHosts() const;

  std::vector<const TableDescriptor*> getPhysicalTablesDescriptors(const TableDescriptor* logicalTableDesc) const;

  int32_t getTableEpoch(const int32_t db_id, const int32_t table_id) const;
  void setTableEpoch(const int db_id, const int table_id, const int new_epoch);
  int getDatabaseId() const { return currentDB_.dbId; }

  static void set(const std::string& dbName, std::shared_ptr<Catalog> cat);
  static std::shared_ptr<Catalog> get(const std::string& dbName);
  static void remove(const std::string& dbName);

  const ColumnDescriptor* getDeletedColumn(const TableDescriptor* td) const;

  void setDeletedColumn(const TableDescriptor* td, const ColumnDescriptor* cd);

 protected:
  typedef std::map<std::string, TableDescriptor*> TableDescriptorMap;
  typedef std::map<int, TableDescriptor*> TableDescriptorMapById;
  typedef std::map<int32_t, std::vector<int32_t>> LogicalToPhysicalTableMapById;
  typedef std::tuple<int, std::string> ColumnKey;
  typedef std::map<ColumnKey, ColumnDescriptor*> ColumnDescriptorMap;
  typedef std::tuple<int, int> ColumnIdKey;
  typedef std::map<ColumnIdKey, ColumnDescriptor*> ColumnDescriptorMapById;
  typedef std::map<DictRef, std::unique_ptr<DictDescriptor>> DictDescriptorMapById;
  typedef std::map<std::string, FrontendViewDescriptor*> FrontendViewDescriptorMap;
  typedef std::map<int, FrontendViewDescriptor*> FrontendViewDescriptorMapById;
  typedef std::map<std::string, LinkDescriptor*> LinkDescriptorMap;
  typedef std::map<int, LinkDescriptor*> LinkDescriptorMapById;

  void CheckAndExecuteMigrations();
  void updateDictionaryNames();
  void updateTableDescriptorSchema();
  void updateFrontendViewSchema();
  void updateLinkSchema();
  void updateFrontendViewAndLinkUsers();
  void updateLogicalToPhysicalTableLinkSchema();
  void updateLogicalToPhysicalTableMap(const int32_t logical_tb_id);
  void updateDictionarySchema();
  void updatePageSize();
  void buildMaps();
  void addTableToMap(TableDescriptor& td,
                     const std::list<ColumnDescriptor>& columns,
                     const std::list<DictDescriptor>& dicts);
  void addReferenceToForeignDict(ColumnDescriptor& referencing_column, Parser::SharedDictionaryDef shared_dict_def);
  bool setColumnSharedDictionary(ColumnDescriptor& cd,
                                 std::list<ColumnDescriptor>& cdd,
                                 std::list<DictDescriptor>& dds,
                                 const TableDescriptor td,
                                 const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs);
  void setColumnDictionary(ColumnDescriptor& cd,
                           std::list<DictDescriptor>& dds,
                           const TableDescriptor& td,
                           const bool isLogicalTable);
  void addFrontendViewToMap(FrontendViewDescriptor& vd);
  void addLinkToMap(LinkDescriptor& ld);
  void removeTableFromMap(const std::string& tableName, int tableId);
  void doDropTable(const TableDescriptor* td);
  void doTruncateTable(const TableDescriptor* td);
  void renamePhysicalTable(const TableDescriptor* td, const std::string& newTableName);
  void instantiateFragmenter(TableDescriptor* td) const;
  void getAllColumnMetadataForTable(const TableDescriptor* td,
                                    std::list<const ColumnDescriptor*>& colDescs,
                                    const bool fetchSystemColumns,
                                    const bool fetchVirtualColumns) const;
  std::string calculateSHA1(const std::string& data);
  std::string generatePhysicalTableName(const std::string& logicalTableName, const int32_t& shardNumber);

  std::string basePath_;
  TableDescriptorMap tableDescriptorMap_;
  TableDescriptorMapById tableDescriptorMapById_;
  ColumnDescriptorMap columnDescriptorMap_;
  ColumnDescriptorMapById columnDescriptorMapById_;
  DictDescriptorMapById dictDescriptorMapByRef_;
  FrontendViewDescriptorMap frontendViewDescriptorMap_;
  LinkDescriptorMap linkDescriptorMap_;
  LinkDescriptorMapById linkDescriptorMapById_;
  SqliteConnector sqliteConnector_;
  DBMetadata currentDB_;
  std::shared_ptr<Data_Namespace::DataMgr> dataMgr_;
  mutable std::mutex cat_mutex_;

  std::unique_ptr<LdapServer> ldap_server_;
  const std::vector<LeafHostInfo> string_dict_hosts_;
  std::shared_ptr<Calcite> calciteMgr_;

  LogicalToPhysicalTableMapById logicalToPhysicalTableMapById_;
  int logicalTableDictId_;                         // logical table DictId (used for physical tables)
  static const std::string physicalTableNameTag_;  // extra component added to the name of each physical table
  int nextTempTableId_;
  int nextTempDictId_;

 private:
  static std::map<std::string, std::shared_ptr<Catalog>> mapd_cat_map_;
  std::unordered_map<const TableDescriptor*, const ColumnDescriptor*> deletedColumnPerTable_;
};

/*
 * @type SysCatalog
 * @brief class for the system-wide catalog, currently containing user and database metadata
 */
class SysCatalog {
 public:
  void init(const std::string& basePath,
            std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
            LdapMetadata ldapMetadata,
            std::shared_ptr<Calcite> calcite,
            bool is_new_db,
            bool check_privileges);
  void initDB();
  void initObjectPrivileges();
  void createUser(const std::string& name, const std::string& passwd, bool issuper);
  void dropUser(const std::string& name);
  void alterUser(const int32_t userid, const std::string* passwd, bool* is_superp);
  void grantPrivileges(const int32_t userid, const int32_t dbid, const Privileges& privs);
  bool checkPrivileges(UserMetadata& user, DBMetadata& db, const Privileges& wants_privs);
  void createDatabase(const std::string& dbname, int owner);
  void dropDatabase(const int32_t dbid, const std::string& name, Catalog* db_cat);
  bool getMetadataForUser(const std::string& name, UserMetadata& user);
  bool checkPasswordForUser(const std::string& passwd, UserMetadata& user);
  bool getMetadataForDB(const std::string& name, DBMetadata& db);
  const DBMetadata& get_currentDB() const { return currentDB_; }
  Data_Namespace::DataMgr& get_dataMgr() const { return *dataMgr_; }
  Calcite& get_calciteMgr() const { return *calciteMgr_; }
  const std::string& get_basePath() const { return basePath_; }
  std::list<DBMetadata> getAllDBMetadata();
  std::list<UserMetadata> getAllUserMetadata();
  void createDBObject(const UserMetadata& user,
                      const std::string& objectName,
                      const Catalog_Namespace::Catalog& catalog);
  void grantDBObjectPrivileges(const std::string& roleName,
                               DBObject& object,
                               const Catalog_Namespace::Catalog& catalog);
  void revokeDBObjectPrivileges(const std::string& roleName,
                                DBObject& object,
                                const Catalog_Namespace::Catalog& catalog);
  void revokeDBObjectPrivilegesFromAllRoles_unsafe(const std::string& objectName,
                                                   const DBObjectType& objectType,
                                                   Catalog* catalog = nullptr);
  void getDBObjectPrivileges(const std::string& roleName,
                             DBObject& object,
                             const Catalog_Namespace::Catalog& catalog) const;
  bool verifyDBObjectOwnership(const UserMetadata& user, DBObject object, const Catalog_Namespace::Catalog& catalog);
  void createRole(const std::string& roleName, const bool& userPrivateRole = false);
  void dropRole(const std::string& roleName);
  void grantRole(const std::string& roleName, const std::string& userName);
  void revokeRole(const std::string& roleName, const std::string& userName);
  bool checkPrivileges(const UserMetadata& user, std::vector<DBObject>& privObjects);
  bool checkPrivileges(const std::string& userName, std::vector<DBObject>& privObjects);
  Role* getMetadataForRole(const std::string& roleName) const;
  Role* getMetadataForUserRole(int32_t userId) const;
  bool isRoleGrantedToUser(const int32_t userId, const std::string& roleName) const;
  bool hasRole(const std::string& roleName, bool userPrivateRole) const;  // true - role exists, false - otherwise
  std::vector<std::string> getRoles(bool userPrivateRole, bool isSuper, const int32_t userId);
  std::vector<std::string> getUserRoles(const int32_t userId);
  bool arePrivilegesOn() const { return check_privileges_; }

  static SysCatalog& instance() {
    static SysCatalog sys_cat;
    return sys_cat;
  }

 private:
  typedef std::map<std::string, Role*> RoleMap;
  typedef std::map<int32_t, Role*> UserRoleMap;

  SysCatalog() {}
  virtual ~SysCatalog();

  void buildRoleMap();
  void buildUserRoleMap();
  void migrateSysCatalogSchema();
  void dropUserRole(const std::string& userName);

  // Here go functions not wrapped into transactions (necessary for nested calls)
  void grantDefaultPrivilegesToRole_unsafe(const std::string& name, bool issuper);
  void createDefaultMapdRoles_unsafe();
  void createRole_unsafe(const std::string& roleName, const bool& userPrivateRole = false);
  void dropRole_unsafe(const std::string& roleName);
  void grantRole_unsafe(const std::string& roleName, const std::string& userName);
  void revokeRole_unsafe(const std::string& roleName, const std::string& userName);
  void grantDBObjectPrivileges_unsafe(const std::string& roleName,
                                      DBObject& object,
                                      const Catalog_Namespace::Catalog& catalog);
  void revokeDBObjectPrivileges_unsafe(const std::string& roleName,
                                       DBObject& object,
                                       const Catalog_Namespace::Catalog& catalog);

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
  RoleMap roleMap_;
  UserRoleMap userRoleMap_;
  DBMetadata currentDB_;
  std::unique_ptr<SqliteConnector> sqliteConnector_;
  std::shared_ptr<Data_Namespace::DataMgr> dataMgr_;
  std::unique_ptr<LdapServer> ldap_server_;
  std::shared_ptr<Calcite> calciteMgr_;
  mutable std::mutex cat_mutex_;
};

/*
 * @type SessionInfo
 * @brief a user session
 */
class SessionInfo {
 public:
  SessionInfo(std::shared_ptr<Catalog> cat,
              const UserMetadata& user,
              const ExecutorDeviceType t,
              const std::string& sid)
      : catalog_(cat), currentUser_(user), executor_device_type_(t), session_id(sid), last_used_time(time(0)) {}
  SessionInfo(const SessionInfo& s)
      : catalog_(s.catalog_),
        currentUser_(s.currentUser_),
        executor_device_type_(static_cast<ExecutorDeviceType>(s.executor_device_type_)),
        session_id(s.session_id) {}
  Catalog& get_catalog() const { return *catalog_; }
  const UserMetadata& get_currentUser() const { return currentUser_; }
  const ExecutorDeviceType get_executor_device_type() const { return executor_device_type_; }
  void set_executor_device_type(ExecutorDeviceType t) { executor_device_type_ = t; }
  std::string get_session_id() const { return session_id; }
  time_t get_last_used_time() const { return last_used_time; }
  void update_time() { last_used_time = time(0); }
  bool checkDBAccessPrivileges(const AccessPrivileges& privs) const;

 private:
  std::shared_ptr<Catalog> catalog_;
  const UserMetadata currentUser_;
  std::atomic<ExecutorDeviceType> executor_device_type_;
  const std::string session_id;
  std::atomic<time_t> last_used_time;  // for cleaning up SessionInfo after client dies
};

}  // namespace Catalog_Namespace

#endif  // CATALOG_H
