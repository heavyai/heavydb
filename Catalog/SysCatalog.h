/*
 * Copyright 2019 OmniSci, Inc.
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
 * @author  Todd Mostak <todd@map-d.com>, Wei Hong <wei@map-d.com>
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

#include "Grantee.h"
#include "LdapServer.h"
#include "ObjectRoleDescriptor.h"
#include "PkiServer.h"
#include "RestServer.h"

#include "../DataMgr/DataMgr.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "LeafHostInfo.h"

#include "../Calcite/Calcite.h"
#include "../Shared/mapd_shared_mutex.h"

const std::string OMNISCI_SYSTEM_CATALOG = "omnisci_system_catalog";
const std::string OMNISCI_DEFAULT_DB = "omnisci";
const std::string OMNISCI_ROOT_USER = "admin";
const int OMNISCI_ROOT_USER_ID = 0;
const std::string OMNISCI_ROOT_USER_ID_STR = "0";
const std::string OMNISCI_ROOT_PASSWD_DEFAULT = "HyperInteractive";

class Calcite;

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
               bool l)
      : userId(u)
      , userName(n)
      , passwd_hash(p)
      , isSuper(s)
      , defaultDbId(d)
      , can_login(l) {}
  UserMetadata() {}
  UserMetadata(UserMetadata const& user_meta)
      : UserMetadata(user_meta.userId,
                     user_meta.userName,
                     user_meta.passwd_hash,
                     user_meta.isSuper.load(),
                     user_meta.defaultDbId,
                     user_meta.can_login) {}
  int32_t userId;
  std::string userName;
  std::string passwd_hash;
  std::atomic<bool> isSuper{false};
  int32_t defaultDbId;
  bool can_login{true};
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
  void createUser(const std::string& name,
                  const std::string& passwd,
                  bool issuper,
                  const std::string& dbname,
                  bool can_login);
  void dropUser(const std::string& name);
  void alterUser(const int32_t userid,
                 const std::string* passwd,
                 bool* issuper,
                 const std::string* dbname,
                 bool* can_login);
  void renameUser(std::string const& old_name, std::string const& new_name);
  void createDatabase(const std::string& dbname, int owner);
  void renameDatabase(std::string const& old_name, std::string const& new_name);
  void dropDatabase(const DBMetadata& db);
  bool getMetadataForUser(const std::string& name, UserMetadata& user);
  bool getMetadataForUserById(const int32_t idIn, UserMetadata& user);
  bool checkPasswordForUser(const std::string& passwd,
                            std::string& name,
                            UserMetadata& user);
  void getMetadataWithDefaultDB(std::string& dbname,
                                const std::string& username,
                                Catalog_Namespace::DBMetadata& db_meta,
                                UserMetadata& user_meta);
  bool getMetadataForDB(const std::string& name, DBMetadata& db);
  bool getMetadataForDBById(const int32_t idIn, DBMetadata& db);
  Data_Namespace::DataMgr& getDataMgr() const { return *dataMgr_; }
  Calcite& getCalciteMgr() const { return *calciteMgr_; }
  const std::string& getBasePath() const { return basePath_; }
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
  void revokeDashboardSystemRole(const std::string roleName,
                                 const std::vector<std::string> grantees);
  bool isAggregator() const { return aggregator_; }
  static SysCatalog& instance() {
    static SysCatalog sys_cat{};
    return sys_cat;
  }

  void populateRoleDbObjects(const std::vector<DBObject>& objects);
  std::string name() const { return OMNISCI_DEFAULT_DB; }
  void renameObjectsInDescriptorMap(DBObject& object,
                                    const Catalog_Namespace::Catalog& cat);
  void syncUserWithRemoteProvider(const std::string& user_name,
                                  std::vector<std::string> idp_roles,
                                  bool* issuper);
  std::unordered_map<std::string, std::vector<std::string>> getGranteesOfSharedDashboards(
      const std::vector<std::string>& dashboard_ids);
  void check_for_session_encryption(const std::string& pki_cert, std::string& session);

 private:
  using GranteeMap = std::map<std::string, Grantee*>;
  using ObjectRoleDescriptorMap = std::multimap<std::string, ObjectRoleDescriptor*>;

  SysCatalog()
      : CommonFileOperations(basePath_)
      , aggregator_(false)
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
  void importDataFromOldMapdDB();
  void createUserRoles();
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
  void updateUserRoleName(const std::string& roleName, const std::string& newName);

  template <typename F, typename... Args>
  void execInTransaction(F&& f, Args&&... args);

  std::string basePath_;
  GranteeMap granteeMap_;
  ObjectRoleDescriptorMap objectDescriptorMap_;
  std::unique_ptr<SqliteConnector> sqliteConnector_;

  std::shared_ptr<Data_Namespace::DataMgr> dataMgr_;
  std::unique_ptr<LdapServer> ldap_server_;
  std::unique_ptr<RestServer> rest_server_;
  std::unique_ptr<PkiServer> pki_server_;
  const AuthMetadata* authMetadata_;
  std::shared_ptr<Calcite> calciteMgr_;
  std::vector<LeafHostInfo> string_dict_hosts_;
  bool aggregator_;
  auto yieldTransactionStreamer();

 public:
  mutable std::mutex sqliteMutex_;
  mutable mapd_shared_mutex sharedMutex_;
  mutable std::atomic<std::thread::id> thread_holding_sqlite_lock;
  mutable std::atomic<std::thread::id> thread_holding_write_lock;
  static thread_local bool thread_holds_read_lock;
};

}  // namespace Catalog_Namespace

#endif  // SYS_CATALOG_H
