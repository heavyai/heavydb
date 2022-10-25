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
 * @file    SysCatalog.h
 * @brief   This file contains the class specification and related data structures for
 * SysCatalog.
 *
 * This file contains the SysCatalog class specification. The SysCatalog class is
 * responsible for changning, accessing and caching file with global metadata: users,
 * roles, privileges and databases.
 *
 */

#ifndef SYS_CATALOG_H
#define SYS_CATALOG_H

#include <atomic>
#include <cstdint>
#include <ctime>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tbb/concurrent_hash_map.h"

#include "Calcite/Calcite.h"
#include "DataMgr/DataMgr.h"
#include "Grantee.h"
#include "LeafHostInfo.h"
#include "MigrationMgr/MigrationMgr.h"
#include "OSDependent/heavyai_locks.h"
#include "ObjectRoleDescriptor.h"
#include "PkiServer.h"
#include "Shared/DbObjectKeys.h"
#include "Shared/SysDefinitions.h"
#include "Shared/heavyai_shared_mutex.h"
#include "SqliteConnector/SqliteConnector.h"

class Calcite;

extern std::string g_base_path;

namespace Catalog_Namespace {

/*
 * @type UserMetadata
 * @brief metadata for a db user
 */
struct UserMetadata {
  UserMetadata(int32_t u,
               const std::string& n,
               const std::string& p,
               bool s,
               int32_t d,
               bool l,
               bool t)
      : userId(u)
      , userName(n)
      , passwd_hash(p)
      , isSuper(s)
      , defaultDbId(d)
      , can_login(l)
      , is_temporary(t) {}
  UserMetadata() {}
  UserMetadata(UserMetadata const& user_meta)
      : UserMetadata(user_meta.userId,
                     user_meta.userName,
                     user_meta.passwd_hash,
                     user_meta.isSuper.load(),
                     user_meta.defaultDbId,
                     user_meta.can_login,
                     user_meta.is_temporary) {}
  UserMetadata& operator=(UserMetadata const& user_meta) {
    if (this != &user_meta) {
      userId = user_meta.userId;
      userName = user_meta.userName;
      passwd_hash = user_meta.passwd_hash;
      isSuper.store(user_meta.isSuper.load());
      defaultDbId = user_meta.defaultDbId;
      can_login = user_meta.can_login;
      is_temporary = user_meta.is_temporary;
    }
    return *this;
  }
  int32_t userId;
  std::string userName;
  std::string passwd_hash;
  std::atomic<bool> isSuper{false};
  int32_t defaultDbId{-1};
  bool can_login{true};
  bool is_temporary{false};

  // Return a string that is safe to log for the username based on --log-user-id.
  std::string userLoggable() const;
};

struct UserAlterations {
  std::optional<std::string> passwd;
  std::optional<bool> is_super;
  std::optional<std::string> default_db;
  std::optional<bool> can_login;

  bool wouldChange(UserMetadata const& user_meta) const;
  std::string toString(bool hide_password = true) const;
};

/*
 * @type DBMetadata
 * @brief metadata for a database
 */
struct DBMetadata {
  DBMetadata() : dbId(0), dbOwner(0) {}
  int32_t dbId;
  std::string dbName;
  int32_t dbOwner;
};

/*
 * @type DBSummary
 * @brief summary info for a database
 */
struct DBSummary {
  std::string dbName;
  std::string dbOwnerName;
};
using DBSummaryList = std::list<DBSummary>;

class CommonFileOperations {
 public:
  CommonFileOperations(std::string const& base_path) : base_path_(base_path) {}

  inline void removeCatalogByFullPath(std::string const& full_path);
  inline void removeCatalogByName(std::string const& name);
  inline auto duplicateAndRenameCatalog(std::string const& current_name,
                                        std::string const& new_name);
  inline auto assembleCatalogName(std::string const& name);

 private:
  std::string const& base_path_;
};

/*
 * @type SysCatalog
 * @brief class for the system-wide catalog, currently containing user and database
 * metadata
 */
class SysCatalog : private CommonFileOperations {
 public:
  void init(const std::string& basePath,
            std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
            const AuthMetadata& authMetadata,
            std::shared_ptr<Calcite> calcite,
            bool is_new_db,
            bool aggregator,
            const std::vector<LeafHostInfo>& string_dict_hosts);

  bool isInitialized() const;

  /**
   * logins (connects) a user against a database.
   *
   * throws a std::exception in all error cases! (including wrong password)
   */
  std::shared_ptr<Catalog> login(std::string& db,
                                 std::string& username,
                                 const std::string& password,
                                 UserMetadata& user_meta,
                                 bool check_password = true);
  std::shared_ptr<Catalog> switchDatabase(std::string& dbname,
                                          const std::string& username);
  UserMetadata createUser(std::string const& name,
                          UserAlterations alts,
                          bool is_temporary);
  void dropUser(const std::string& name);
  // TODO(Misiu): This method is needed only by tests and should otherwise be private and
  // accessed via friendship.
  void dropUserUnchecked(const std::string& name, const UserMetadata& user);
  UserMetadata alterUser(std::string const& name, UserAlterations alts);
  void renameUser(std::string const& old_name, std::string const& new_name);
  void createDatabase(const std::string& dbname, int owner);
  void renameDatabase(std::string const& old_name, std::string const& new_name);
  void changeDatabaseOwner(std::string const& dbname, const std::string& new_owner);
  void dropDatabase(const DBMetadata& db);
  std::optional<UserMetadata> getUser(std::string const& uname) {
    if (UserMetadata user; getMetadataForUser(uname, user)) {
      return user;
    }
    return {};
  }
  std::optional<UserMetadata> getUser(int32_t const uid) {
    if (UserMetadata user; getMetadataForUserById(uid, user)) {
      return user;
    }
    return {};
  }
  std::optional<DBMetadata> getDB(std::string const& dbname) {
    if (DBMetadata db; getMetadataForDB(dbname, db)) {
      return db;
    }
    return {};
  }
  std::optional<DBMetadata> getDB(int32_t const dbid) {
    if (DBMetadata db; getMetadataForDBById(dbid, db)) {
      return db;
    }
    return {};
  }
  bool getMetadataForUser(const std::string& name, UserMetadata& user);
  bool getMetadataForUserById(const int32_t idIn, UserMetadata& user);
  bool checkPasswordForUser(const std::string& passwd,
                            std::string& name,
                            UserMetadata& user);
  bool getMetadataForDB(const std::string& name, DBMetadata& db);
  bool getMetadataForDBById(const int32_t idIn, DBMetadata& db);
  Data_Namespace::DataMgr& getDataMgr() const { return *dataMgr_; }
  Calcite& getCalciteMgr() const { return *calciteMgr_; }
  const std::string& getCatalogBasePath() const { return basePath_; }
  SqliteConnector* getSqliteConnector() { return sqliteConnector_.get(); }
  std::list<DBMetadata> getAllDBMetadata();
  std::list<UserMetadata> getAllUserMetadata();
  /**
   * return the users associated with the given DB
   */
  std::list<UserMetadata> getAllUserMetadata(const int64_t dbId);
  DBSummaryList getDatabaseListForUser(const UserMetadata& user);
  void createDBObject(const UserMetadata& user,
                      const std::string& objectName,
                      DBObjectType type,
                      const Catalog_Namespace::Catalog& catalog,
                      int32_t objectId = -1);
  /**
   * Renames an DBObject
   *
   * @param objectName - original DBObject name
   * @param newName - new name of DBObject
   * @param type - type of DBObject
   * @param objectId - original DBObject ID
   * @param catalog - Catalog instance object exists in
   */
  void renameDBObject(const std::string& objectName,
                      const std::string& newName,
                      DBObjectType type,
                      int32_t objectId,
                      const Catalog_Namespace::Catalog& catalog);
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
  void revokeDBObjectPrivilegesFromAll(DBObject object, Catalog* catalog);
  void revokeDBObjectPrivilegesFromAll_unsafe(DBObject object, Catalog* catalog);
  void revokeDBObjectPrivilegesFromAllBatch(std::vector<DBObject>& objects,
                                            Catalog* catalog);
  void revokeDBObjectPrivilegesFromAllBatch_unsafe(std::vector<DBObject>& objects,
                                                   Catalog* catalog);
  void getDBObjectPrivileges(const std::string& granteeName,
                             DBObject& object,
                             const Catalog_Namespace::Catalog& catalog) const;
  bool verifyDBObjectOwnership(const UserMetadata& user,
                               DBObject object,
                               const Catalog_Namespace::Catalog& catalog);
  /**
   * Change ownership of a DBObject
   *
   * @param new_owner - new owner of DBObject
   * @param previous_owner - previous owner of DBObject
   * @param object - DBObject to change ownership of
   * @param catalog - Catalog instance object exists in
   * @param revoke_privileges - if true, revoke previous_owner's privileges
   */
  void changeDBObjectOwnership(const UserMetadata& new_owner,
                               const UserMetadata& previous_owner,
                               DBObject object,
                               const Catalog_Namespace::Catalog& catalog,
                               bool revoke_privileges = true);
  void createRole(const std::string& roleName,
                  const bool user_private_role,
                  const bool is_temporary = false);
  void dropRole(const std::string& roleName, const bool is_temporary = false);
  void grantRoleBatch(const std::vector<std::string>& roles,
                      const std::vector<std::string>& grantees);
  void grantRole(const std::string& role,
                 const std::string& grantee,
                 const bool is_temporary = false);
  void revokeRoleBatch(const std::vector<std::string>& roles,
                       const std::vector<std::string>& grantees);
  void revokeRole(const std::string& role,
                  const std::string& grantee,
                  const bool is_temporary = false);
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
  std::vector<ObjectRoleDescriptor> getMetadataForAllObjects() const;
  bool isRoleGrantedToGrantee(const std::string& granteeName,
                              const std::string& roleName,
                              bool only_direct) const;
  std::vector<std::string> getRoles(const std::string& user_name, bool effective = true);
  std::vector<std::string> getRoles(bool include_user_private_role,
                                    bool is_super,
                                    const std::string& user_name,
                                    bool ignore_deleted_user = false);
  std::vector<std::string> getRoles(const std::string& userName, const int32_t dbId);
  // Get all roles that have been created, even roles that have not been assigned to other
  // users or roles.
  std::set<std::string> getCreatedRoles() const;
  bool isAggregator() const { return aggregator_; }
  static SysCatalog& instance() {
    std::unique_lock lk(instance_mutex_);
    if (!instance_) {
      instance_.reset(new SysCatalog());
    }
    return *instance_;
  }

  static void destroy() {
    std::unique_lock lk(instance_mutex_);
    instance_.reset();
    migrations::MigrationMgr::destroy();
  }

  void populateRoleDbObjects(const std::vector<DBObject>& objects);
  std::string name() const { return shared::kSystemCatalogName; }
  void renameObjectsInDescriptorMap(DBObject& object,
                                    const Catalog_Namespace::Catalog& cat);
  void syncUserWithRemoteProvider(const std::string& user_name,
                                  std::vector<std::string> idp_roles,
                                  UserAlterations alts);
  std::unordered_map<std::string, std::vector<std::string>> getGranteesOfSharedDashboards(
      const std::vector<std::string>& dashboard_ids);
  void check_for_session_encryption(const std::string& pki_cert, std::string& session);
  std::vector<Catalog*> getCatalogsForAllDbs();

  std::shared_ptr<Catalog> getDummyCatalog() { return dummyCatalog_; }

  std::shared_ptr<Catalog> getCatalog(const std::string& dbName);
  std::shared_ptr<Catalog> getCatalog(const int32_t db_id);
  std::shared_ptr<Catalog> getCatalog(const DBMetadata& curDB, bool is_new_db);

  void removeCatalog(const std::string& dbName);

  virtual ~SysCatalog();

  /**
   * Reassigns database object ownership from a set of users (old owners) to another user
   * (new owner).
   *
   * @param old_owner_db_objects - map of user ids and database objects whose ownership
   * will be reassigned
   * @param new_owner_id - id of user who will own reassigned database objects
   * @param catalog - catalog for database where ownership reassignment occurred
   */
  void reassignObjectOwners(
      const std::map<int32_t, std::vector<DBObject>>& old_owner_db_objects,
      int32_t new_owner_id,
      const Catalog_Namespace::Catalog& catalog);

  bool hasExecutedMigration(const std::string& migration_name) const;

 private:
  using GranteeMap = std::map<std::string, std::unique_ptr<Grantee>>;
  using ObjectRoleDescriptorMap =
      std::multimap<std::string, std::unique_ptr<ObjectRoleDescriptor>>;

  SysCatalog();

  void initDB();
  void buildMaps(bool is_new_db = false);
  void buildMapsUnlocked(bool is_new_db = false);
  void buildRoleMapUnlocked();
  void buildUserRoleMapUnlocked();
  void buildObjectDescriptorMapUnlocked();
  void rebuildObjectMapsUnlocked();
  void checkAndExecuteMigrations();
  void importDataFromOldMapdDB();
  void createRoles();
  void fixRolesMigration();
  void addAdminUserRole();
  void migratePrivileges();
  void migratePrivileged_old();
  void updateUserSchema();
  void updatePasswordsToHashes();
  void updateBlankPasswordsToRandom();
  void updateSupportUserDeactivation();
  void migrateDBAccessPrivileges();
  void loginImpl(std::string& username,
                 const std::string& password,
                 UserMetadata& user_meta);
  bool checkPasswordForUserImpl(const std::string& passwd,
                                std::string& name,
                                UserMetadata& user);

  void checkDuplicateCaseInsensitiveDbNames() const;

  struct UpdateQuery {
    std::string query;
    std::vector<std::string> text_params;
  };
  using UpdateQueries = std::list<UpdateQuery>;
  void runUpdateQueriesAndChangeOwnership(const UserMetadata& new_owner,
                                          const UserMetadata& previous_owner,
                                          DBObject object,
                                          const Catalog_Namespace::Catalog& catalog,
                                          const UpdateQueries& update_queries,
                                          bool revoke_privileges = true);

  // Here go functions not wrapped into transactions (necessary for nested calls)
  void grantDefaultPrivilegesToRole_unsafe(const std::string& name, bool issuper);
  void createRole_unsafe(const std::string& roleName,
                         const bool userPrivateRole,
                         const bool is_temporary);
  void dropRole_unsafe(const std::string& roleName, const bool is_temporary);
  void grantRoleBatch_unsafe(const std::vector<std::string>& roles,
                             const std::vector<std::string>& grantees);
  void grantRole_unsafe(const std::string& roleName,
                        const std::string& granteeName,
                        const bool is_temporary);
  void revokeRoleBatch_unsafe(const std::vector<std::string>& roles,
                              const std::vector<std::string>& grantees);
  void revokeRole_unsafe(const std::string& roleName,
                         const std::string& granteeName,
                         const bool is_temporary);
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
  bool isDashboardSystemRole(const std::string& roleName) const;
  void updateUserRoleName(const std::string& roleName, const std::string& newName);
  void getMetadataWithDefaultDB(std::string& dbname,
                                const std::string& username,
                                Catalog_Namespace::DBMetadata& db_meta,
                                UserMetadata& user_meta);
  /**
   * For servers configured to use external authentication providers, determine whether
   * users will be allowed to fallback to local login accounts. If no external providers
   * are configured, returns true.
   */
  bool allowLocalLogin() const;

  template <typename F, typename... Args>
  void execInTransaction(F&& f, Args&&... args);

  void initializeInformationSchemaDb();
  void recordExecutedMigration(const std::string& migration_name) const;
  bool hasVersionHistoryTable() const;
  void createVersionHistoryTable() const;

  std::string basePath_;
  GranteeMap granteeMap_;
  ObjectRoleDescriptorMap objectDescriptorMap_;
  std::unique_ptr<SqliteConnector> sqliteConnector_;

  std::shared_ptr<Data_Namespace::DataMgr> dataMgr_;
  std::unique_ptr<PkiServer> pki_server_;
  const AuthMetadata* authMetadata_;
  std::shared_ptr<Calcite> calciteMgr_;
  std::vector<LeafHostInfo> string_dict_hosts_;
  bool aggregator_;
  auto yieldTransactionStreamer();

  // contains a map of all the catalog within this system
  // it is lazy loaded
  // std::map<std::string, std::shared_ptr<Catalog>> cat_map_;
  using dbid_to_cat_map = tbb::concurrent_hash_map<std::string, std::shared_ptr<Catalog>>;
  dbid_to_cat_map cat_map_;

  static std::mutex instance_mutex_;
  static std::unique_ptr<SysCatalog> instance_;

  // Flag used to indicate whether this SysCatalog instance has been initialized. This is
  // currently used in tests to ensure that QueryRunner and DBHandlerTestFixture are not
  // re-initializing SysCatalog.
  bool is_initialized_{false};

 public:
  mutable std::unique_ptr<heavyai::DistributedSharedMutex> dcatalogMutex_;
  mutable std::unique_ptr<heavyai::DistributedSharedMutex> dsqliteMutex_;
  mutable std::mutex sqliteMutex_;
  mutable heavyai::shared_mutex sharedMutex_;
  mutable std::atomic<std::thread::id> thread_holding_sqlite_lock;
  mutable std::atomic<std::thread::id> thread_holding_write_lock;
  static thread_local bool thread_holds_read_lock;
  // used by catalog when initially creating a catalog instance
  std::shared_ptr<Catalog> dummyCatalog_;
  std::unordered_map<std::string, std::shared_ptr<UserMetadata>> temporary_users_by_name_;
  std::unordered_map<int32_t, std::shared_ptr<UserMetadata>> temporary_users_by_id_;
  int32_t next_temporary_user_id_{shared::kTempUserIdRange};
};

const TableDescriptor* get_metadata_for_table(const ::shared::TableKey& table_key,
                                              bool populate_fragmenter = true);

const ColumnDescriptor* get_metadata_for_column(const ::shared::ColumnKey& column_key);
}  // namespace Catalog_Namespace

#endif  // SYS_CATALOG_H
