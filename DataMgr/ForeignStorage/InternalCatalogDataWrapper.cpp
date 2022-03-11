/*
 * Copyright 2021 OmniSci, Inc.
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

#include "InternalCatalogDataWrapper.h"

#include <regex>

#include "Catalog/Catalog.h"
#include "Catalog/SysCatalog.h"
#include "FsiChunkUtils.h"
#include "FsiJsonUtils.h"
#include "ImportExport/Importer.h"
#include "Shared/StringTransform.h"
#include "Shared/SysDefinitions.h"
#include "Shared/distributed.h"

namespace foreign_storage {
InternalCatalogDataWrapper::InternalCatalogDataWrapper() : InternalSystemDataWrapper() {}

InternalCatalogDataWrapper::InternalCatalogDataWrapper(const int db_id,
                                                       const ForeignTable* foreign_table)
    : InternalSystemDataWrapper(db_id, foreign_table) {}

namespace {
void set_null(import_export::TypedImportBuffer* import_buffer) {
  import_buffer->add_value(import_buffer->getColumnDesc(), "", true, {});
}

void populate_import_buffers_for_catalog_users(
    const std::list<Catalog_Namespace::UserMetadata>& all_users,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& user : all_users) {
    if (import_buffers.find("user_id") != import_buffers.end()) {
      import_buffers["user_id"]->addInt(user.userId);
    }
    if (import_buffers.find("user_name") != import_buffers.end()) {
      import_buffers["user_name"]->addString(user.userName);
    }
    if (import_buffers.find("is_super_user") != import_buffers.end()) {
      import_buffers["is_super_user"]->addBoolean(user.isSuper);
    }
    if (import_buffers.find("default_db_id") != import_buffers.end()) {
      import_buffers["default_db_id"]->addInt(user.defaultDbId);
    }
    if (import_buffers.find("default_db_name") != import_buffers.end()) {
      if (user.defaultDbId > 0) {
        import_buffers["default_db_name"]->addString(get_db_name(user.defaultDbId));
      } else {
        set_null(import_buffers["default_db_name"]);
      }
    }
    if (import_buffers.find("can_login") != import_buffers.end()) {
      import_buffers["can_login"]->addBoolean(user.can_login);
    }
  }
}

std::string get_user_name(int32_t user_id) {
  Catalog_Namespace::UserMetadata user_metadata;
  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  if (sys_catalog.getMetadataForUserById(user_id, user_metadata)) {
    return user_metadata.userName;
  } else {
    // User has been deleted.
    return kDeletedValueIndicator;
  }
}

std::string get_table_type(const TableDescriptor& td) {
  std::string table_type{"DEFAULT"};
  if (td.isView) {
    table_type = "VIEW";
  } else if (td.isTemporaryTable()) {
    table_type = "TEMPORARY";
  } else if (td.isForeignTable()) {
    table_type = "FOREIGN";
  }
  return table_type;
}

std::string get_table_ddl(int32_t db_id, const TableDescriptor& td) {
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  CHECK(catalog);
  auto ddl = catalog->dumpCreateTable(td.tableId);
  if (!ddl.has_value()) {
    // It is possible for the table to be concurrently deleted while querying the system
    // table.
    return kDeletedValueIndicator;
  }
  return ddl.value();
}

void populate_import_buffers_for_catalog_tables(
    const std::map<int32_t, std::vector<TableDescriptor>>& tables_by_database,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& [db_id, tables] : tables_by_database) {
    for (const auto& table : tables) {
      if (import_buffers.find("database_id") != import_buffers.end()) {
        import_buffers["database_id"]->addInt(db_id);
      }
      if (import_buffers.find("database_name") != import_buffers.end()) {
        import_buffers["database_name"]->addString(get_db_name(db_id));
      }
      if (import_buffers.find("table_id") != import_buffers.end()) {
        import_buffers["table_id"]->addInt(table.tableId);
      }
      if (import_buffers.find("table_name") != import_buffers.end()) {
        import_buffers["table_name"]->addString(table.tableName);
      }
      if (import_buffers.find("owner_id") != import_buffers.end()) {
        import_buffers["owner_id"]->addInt(table.userId);
      }
      if (import_buffers.find("owner_user_name") != import_buffers.end()) {
        import_buffers["owner_user_name"]->addString(get_user_name(table.userId));
      }
      if (import_buffers.find("column_count") != import_buffers.end()) {
        import_buffers["column_count"]->addInt(table.nColumns);
      }
      if (import_buffers.find("table_type") != import_buffers.end()) {
        import_buffers["table_type"]->addString(get_table_type(table));
      }
      if (import_buffers.find("view_sql") != import_buffers.end()) {
        import_buffers["view_sql"]->addString(table.viewSQL);
      }
      if (import_buffers.find("max_fragment_size") != import_buffers.end()) {
        import_buffers["max_fragment_size"]->addInt(table.maxFragRows);
      }
      if (import_buffers.find("max_chunk_size") != import_buffers.end()) {
        import_buffers["max_chunk_size"]->addBigint(table.maxChunkSize);
      }
      if (import_buffers.find("fragment_page_size") != import_buffers.end()) {
        import_buffers["fragment_page_size"]->addInt(table.fragPageSize);
      }
      if (import_buffers.find("max_rows") != import_buffers.end()) {
        import_buffers["max_rows"]->addBigint(table.maxRows);
      }
      if (import_buffers.find("max_rollback_epochs") != import_buffers.end()) {
        import_buffers["max_rollback_epochs"]->addInt(table.maxRollbackEpochs);
      }
      if (import_buffers.find("shard_count") != import_buffers.end()) {
        import_buffers["shard_count"]->addInt(table.nShards);
      }
      if (import_buffers.find("ddl_statement") != import_buffers.end()) {
        import_buffers["ddl_statement"]->addString(get_table_ddl(db_id, table));
      }
    }
  }
}

std::vector<std::string> get_data_sources(const std::string& dashboard_metadata) {
  rapidjson::Document document;
  document.Parse(dashboard_metadata);
  if (!document.IsObject()) {
    return {};
  }
  std::string data_sources_str;
  json_utils::get_value_from_object(document, data_sources_str, "table");
  auto data_sources = split(data_sources_str, ",");
  for (auto it = data_sources.begin(); it != data_sources.end();) {
    *it = strip(*it);
    static std::regex parameter_regex{"\\$\\{.+\\}"};
    if (std::regex_match(*it, parameter_regex)) {
      // Remove custom SQL sources.
      it = data_sources.erase(it);
    } else {
      it++;
    }
  }
  return data_sources;
}

void populate_import_buffers_for_catalog_dashboards(
    const std::map<int32_t, std::vector<DashboardDescriptor>>& dashboards_by_database,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& [db_id, dashboards] : dashboards_by_database) {
    for (const auto& dashboard : dashboards) {
      if (import_buffers.find("database_id") != import_buffers.end()) {
        import_buffers["database_id"]->addInt(db_id);
      }
      if (import_buffers.find("database_name") != import_buffers.end()) {
        import_buffers["database_name"]->addString(get_db_name(db_id));
      }
      if (import_buffers.find("dashboard_id") != import_buffers.end()) {
        import_buffers["dashboard_id"]->addInt(dashboard.dashboardId);
      }
      if (import_buffers.find("dashboard_name") != import_buffers.end()) {
        import_buffers["dashboard_name"]->addString(dashboard.dashboardName);
      }
      if (import_buffers.find("owner_id") != import_buffers.end()) {
        import_buffers["owner_id"]->addInt(dashboard.userId);
      }
      if (import_buffers.find("owner_user_name") != import_buffers.end()) {
        import_buffers["owner_user_name"]->addString(get_user_name(dashboard.userId));
      }
      if (import_buffers.find("last_updated_at") != import_buffers.end()) {
        auto& buffer = import_buffers["last_updated_at"];
        import_buffers["last_updated_at"]->add_value(
            buffer->getColumnDesc(), dashboard.updateTime, false, {});
      }
      if (import_buffers.find("data_sources") != import_buffers.end()) {
        import_buffers["data_sources"]->addStringArray(
            get_data_sources(dashboard.dashboardMetadata));
      }
    }
  }
}

std::vector<std::string> get_permissions(const AccessPrivileges privileges,
                                         int32_t object_type,
                                         int32_t object_id) {
  std::vector<std::string> permissions;
  auto type = static_cast<DBObjectType>(object_type);
  if (type == DBObjectType::DatabaseDBObjectType) {
    if (privileges.hasPermission(AccessPrivileges::ALL_DATABASE.privileges)) {
      permissions.emplace_back("all");
    } else {
      if (privileges.hasPermission(AccessPrivileges::VIEW_SQL_EDITOR.privileges)) {
        permissions.emplace_back("view_sql_editor");
      }
      if (privileges.hasPermission(AccessPrivileges::ACCESS.privileges)) {
        permissions.emplace_back("access");
      }
    }
  } else if (type == DBObjectType::TableDBObjectType) {
    if (privileges.hasPermission(AccessPrivileges::ALL_TABLE.privileges)) {
      permissions.emplace_back("all");
    } else {
      std::string suffix;
      if (object_id == -1) {
        suffix = " table";
      }
      if (privileges.hasPermission(AccessPrivileges::SELECT_FROM_TABLE.privileges)) {
        permissions.emplace_back("select" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::INSERT_INTO_TABLE.privileges)) {
        permissions.emplace_back("insert" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::UPDATE_IN_TABLE.privileges)) {
        permissions.emplace_back("update" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::DELETE_FROM_TABLE.privileges)) {
        permissions.emplace_back("delete" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::TRUNCATE_TABLE.privileges)) {
        permissions.emplace_back("truncate" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::ALTER_TABLE.privileges)) {
        permissions.emplace_back("alter" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::DROP_TABLE.privileges)) {
        permissions.emplace_back("drop" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::CREATE_TABLE.privileges)) {
        permissions.emplace_back("create table");
      }
    }
  } else if (type == DBObjectType::DashboardDBObjectType) {
    if (privileges.hasPermission(AccessPrivileges::ALL_DASHBOARD.privileges)) {
      permissions.emplace_back("all");
    } else {
      std::string suffix;
      if (object_id == -1) {
        suffix = " dashboard";
      }
      if (privileges.hasPermission(AccessPrivileges::VIEW_DASHBOARD.privileges)) {
        permissions.emplace_back("view" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::EDIT_DASHBOARD.privileges)) {
        permissions.emplace_back("edit" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::DELETE_DASHBOARD.privileges)) {
        permissions.emplace_back("delete" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::CREATE_DASHBOARD.privileges)) {
        permissions.emplace_back("create dashboard");
      }
    }
  } else if (type == DBObjectType::ViewDBObjectType) {
    if (privileges.hasPermission(AccessPrivileges::ALL_VIEW.privileges)) {
      permissions.emplace_back("all");
    } else {
      std::string suffix;
      if (object_id == -1) {
        suffix = " view";
      }
      if (privileges.hasPermission(AccessPrivileges::SELECT_FROM_VIEW.privileges)) {
        permissions.emplace_back("select" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::DROP_VIEW.privileges)) {
        permissions.emplace_back("drop" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::CREATE_VIEW.privileges)) {
        permissions.emplace_back("create view");
      }
    }
  } else if (type == DBObjectType::ServerDBObjectType) {
    if (privileges.hasPermission(AccessPrivileges::ALL_SERVER.privileges)) {
      permissions.emplace_back("all");
    } else {
      std::string suffix;
      if (object_id == -1) {
        suffix = " server";
      }
      if (privileges.hasPermission(AccessPrivileges::ALTER_SERVER.privileges)) {
        permissions.emplace_back("alter" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::SERVER_USAGE.privileges)) {
        permissions.emplace_back("usage" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::DROP_SERVER.privileges)) {
        permissions.emplace_back("drop" + suffix);
      }
      if (privileges.hasPermission(AccessPrivileges::CREATE_SERVER.privileges)) {
        permissions.emplace_back("create server");
      }
    }
  } else {
    UNREACHABLE() << "Unexpected object type: " << object_type;
  }
  return permissions;
}

std::string get_object_type_str(int32_t object_type) {
  std::string object_type_str;
  auto type = static_cast<DBObjectType>(object_type);
  if (type == DBObjectType::DatabaseDBObjectType) {
    object_type_str = "database";
  } else if (type == DBObjectType::TableDBObjectType) {
    object_type_str = "table";
  } else if (type == DBObjectType::DashboardDBObjectType) {
    object_type_str = "dashboard";
  } else if (type == DBObjectType::ViewDBObjectType) {
    object_type_str = "view";
  } else if (type == DBObjectType::ServerDBObjectType) {
    object_type_str = "server";
  } else {
    UNREACHABLE() << "Unexpected object type: " << object_type;
  }
  return object_type_str;
}

void populate_import_buffers_for_catalog_permissions(
    const std::vector<ObjectRoleDescriptor>& object_permissions,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& permission : object_permissions) {
    if (import_buffers.find("role_name") != import_buffers.end()) {
      import_buffers["role_name"]->addString(permission.roleName);
    }
    if (import_buffers.find("is_user_role") != import_buffers.end()) {
      import_buffers["is_user_role"]->addBoolean(permission.roleType);
    }
    if (import_buffers.find("database_id") != import_buffers.end()) {
      import_buffers["database_id"]->addInt(permission.dbId);
    }
    if (import_buffers.find("database_name") != import_buffers.end()) {
      import_buffers["database_name"]->addString(get_db_name(permission.dbId));
    }
    if (import_buffers.find("object_name") != import_buffers.end()) {
      import_buffers["object_name"]->addString(permission.objectName);
    }
    if (import_buffers.find("object_id") != import_buffers.end()) {
      import_buffers["object_id"]->addInt(permission.objectId);
    }
    if (import_buffers.find("object_owner_id") != import_buffers.end()) {
      import_buffers["object_owner_id"]->addInt(permission.objectOwnerId);
    }
    if (import_buffers.find("object_owner_user_name") != import_buffers.end()) {
      import_buffers["object_owner_user_name"]->addString(
          get_user_name(permission.objectOwnerId));
    }
    if (import_buffers.find("object_permission_type") != import_buffers.end()) {
      import_buffers["object_permission_type"]->addString(
          get_object_type_str(permission.objectType));
    }
    if (import_buffers.find("object_permissions") != import_buffers.end()) {
      auto permissions =
          get_permissions(permission.privs, permission.objectType, permission.objectId);
      import_buffers["object_permissions"]->addStringArray(permissions);
    }
  }
}

void populate_import_buffers_for_catalog_databases(
    const std::list<Catalog_Namespace::DBMetadata>& databases,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& db : databases) {
    if (import_buffers.find("database_id") != import_buffers.end()) {
      import_buffers["database_id"]->addInt(db.dbId);
    }
    if (import_buffers.find("database_name") != import_buffers.end()) {
      import_buffers["database_name"]->addString(db.dbName);
    }
    if (import_buffers.find("owner_id") != import_buffers.end()) {
      import_buffers["owner_id"]->addInt(db.dbOwner);
    }
    if (import_buffers.find("owner_user_name") != import_buffers.end()) {
      import_buffers["owner_user_name"]->addString(get_user_name(db.dbOwner));
    }
  }
}

void populate_import_buffers_for_catalog_roles(
    const std::set<std::string>& roles,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& role : roles) {
    CHECK(import_buffers.find("role_name") != import_buffers.end());
    import_buffers["role_name"]->addString(role);
  }
}

void populate_import_buffers_for_catalog_role_assignments(
    const std::map<std::string, std::vector<std::string>>& user_names_by_role_,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& [role, user_names] : user_names_by_role_) {
    for (const auto& user_name : user_names) {
      if (import_buffers.find("role_name") != import_buffers.end()) {
        import_buffers["role_name"]->addString(role);
      }
      if (import_buffers.find("user_name") != import_buffers.end()) {
        import_buffers["user_name"]->addString(user_name);
      }
    }
  }
}

std::map<int32_t, std::vector<TableDescriptor>> get_all_tables() {
  std::map<int32_t, std::vector<TableDescriptor>> tables_by_database;
  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  for (const auto& catalog : sys_catalog.getCatalogsForAllDbs()) {
    if (catalog->name() != shared::kInfoSchemaDbName) {
      for (const auto& td : catalog->getAllTableMetadataCopy()) {
        tables_by_database[catalog->getDatabaseId()].emplace_back(td);
      }
    }
  }
  return tables_by_database;
}

std::map<int32_t, std::vector<DashboardDescriptor>> get_all_dashboards() {
  std::map<int32_t, std::vector<DashboardDescriptor>> dashboards_by_database;
  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  for (const auto& catalog : sys_catalog.getCatalogsForAllDbs()) {
    if (catalog->name() != shared::kInfoSchemaDbName) {
      for (const auto& dashboard : catalog->getAllDashboardsMetadataCopy()) {
        dashboards_by_database[catalog->getDatabaseId()].emplace_back(dashboard);
      }
    }
  }
  return dashboards_by_database;
}

std::map<std::string, std::vector<std::string>> get_all_role_assignments() {
  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  const auto& users = sys_catalog.getAllUserMetadata();
  std::map<std::string, std::vector<std::string>> user_names_by_role;
  for (const auto& user : users) {
    for (const auto& role :
         sys_catalog.getRoles(false, user.isSuper, user.userName, true)) {
      user_names_by_role[role].emplace_back(user.userName);
    }
  }
  return user_names_by_role;
}
}  // namespace

void InternalCatalogDataWrapper::initializeObjectsForTable(
    const std::string& table_name) {
  row_count_ = 0;

  // Dashboads are handled separately since they are only on the aggregator in
  // distributed.  All others are only on the first leaf.
  if (foreign_table_->tableName == Catalog_Namespace::DASHBOARDS_SYS_TABLE_NAME) {
    if (dist::is_distributed() && !dist::is_aggregator()) {
      // Only the aggregator can contain dashboards in distributed.
      return;
    }
    dashboards_by_database_.clear();
    dashboards_by_database_ = get_all_dashboards();
    for (const auto& [db_id, dashboards] : dashboards_by_database_) {
      row_count_ += dashboards.size();
    }
    return;
  }

  if (dist::is_distributed() && !dist::is_first_leaf()) {
    // For every table except dashboards, only the first leaf returns information in
    // distributed.
    return;
  }
  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  if (foreign_table_->tableName == Catalog_Namespace::USERS_SYS_TABLE_NAME) {
    users_.clear();
    users_ = sys_catalog.getAllUserMetadata();
    row_count_ = users_.size();
  } else if (foreign_table_->tableName == Catalog_Namespace::TABLES_SYS_TABLE_NAME) {
    tables_by_database_.clear();
    tables_by_database_ = get_all_tables();
    for (const auto& [db_id, tables] : tables_by_database_) {
      row_count_ += tables.size();
    }
  } else if (foreign_table_->tableName == Catalog_Namespace::PERMISSIONS_SYS_TABLE_NAME) {
    object_permissions_.clear();
    object_permissions_ = sys_catalog.getMetadataForAllObjects();
    row_count_ = object_permissions_.size();
  } else if (foreign_table_->tableName == Catalog_Namespace::DATABASES_SYS_TABLE_NAME) {
    databases_.clear();
    databases_ = sys_catalog.getAllDBMetadata();
    row_count_ = databases_.size();
  } else if (foreign_table_->tableName == Catalog_Namespace::ROLES_SYS_TABLE_NAME) {
    roles_.clear();
    roles_ = sys_catalog.getCreatedRoles();
    row_count_ = roles_.size();
  } else if (foreign_table_->tableName ==
             Catalog_Namespace::ROLE_ASSIGNMENTS_SYS_TABLE_NAME) {
    user_names_by_role_.clear();
    user_names_by_role_ = get_all_role_assignments();
    for (const auto& [role, user_names] : user_names_by_role_) {
      row_count_ += user_names.size();
    }
  } else {
    UNREACHABLE() << "Unexpected table name: " << foreign_table_->tableName;
  }
}

void InternalCatalogDataWrapper::populateChunkBuffersForTable(
    const std::string& table_name,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  if (foreign_table_->tableName == Catalog_Namespace::USERS_SYS_TABLE_NAME) {
    populate_import_buffers_for_catalog_users(users_, import_buffers);
  } else if (foreign_table_->tableName == Catalog_Namespace::TABLES_SYS_TABLE_NAME) {
    populate_import_buffers_for_catalog_tables(tables_by_database_, import_buffers);
  } else if (foreign_table_->tableName == Catalog_Namespace::DASHBOARDS_SYS_TABLE_NAME) {
    populate_import_buffers_for_catalog_dashboards(dashboards_by_database_,
                                                   import_buffers);
  } else if (foreign_table_->tableName == Catalog_Namespace::PERMISSIONS_SYS_TABLE_NAME) {
    populate_import_buffers_for_catalog_permissions(object_permissions_, import_buffers);
  } else if (foreign_table_->tableName == Catalog_Namespace::DATABASES_SYS_TABLE_NAME) {
    populate_import_buffers_for_catalog_databases(databases_, import_buffers);
  } else if (foreign_table_->tableName == Catalog_Namespace::ROLES_SYS_TABLE_NAME) {
    populate_import_buffers_for_catalog_roles(roles_, import_buffers);
  } else if (foreign_table_->tableName ==
             Catalog_Namespace::ROLE_ASSIGNMENTS_SYS_TABLE_NAME) {
    populate_import_buffers_for_catalog_role_assignments(user_names_by_role_,
                                                         import_buffers);
  } else {
    UNREACHABLE() << "Unexpected table name: " << foreign_table_->tableName;
  }
}
}  // namespace foreign_storage
