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

#include "Catalog/Catalog.h"
#include "Catalog/SysCatalog.h"
#include "ForeignTableSchema.h"
#include "FsiChunkUtils.h"
#include "ImportExport/Importer.h"
#include "TextFileBufferParser.h"

namespace foreign_storage {
InternalCatalogDataWrapper::InternalCatalogDataWrapper()
    : db_id_(-1), foreign_table_(nullptr) {}

InternalCatalogDataWrapper::InternalCatalogDataWrapper(const int db_id,
                                                       const ForeignTable* foreign_table)
    : db_id_(db_id), foreign_table_(foreign_table) {}

void InternalCatalogDataWrapper::validateServerOptions(
    const ForeignServer* foreign_server) const {}

void InternalCatalogDataWrapper::validateTableOptions(
    const ForeignTable* foreign_table) const {}

const std::set<std::string_view>& InternalCatalogDataWrapper::getSupportedTableOptions()
    const {
  static const std::set<std::string_view> supported_table_options{};
  return supported_table_options;
}

void InternalCatalogDataWrapper::validateUserMappingOptions(
    const UserMapping* user_mapping,
    const ForeignServer* foreign_server) const {}

const std::set<std::string_view>&
InternalCatalogDataWrapper::getSupportedUserMappingOptions() const {
  static const std::set<std::string_view> supported_user_mapping_options{};
  return supported_user_mapping_options;
}

namespace {
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
    if (import_buffers.find("can_login") != import_buffers.end()) {
      import_buffers["can_login"]->addBoolean(user.can_login);
    }
  }
}

void populate_import_buffers_for_catalog_tables(
    const std::map<int32_t, std::vector<TableDescriptor>>& tables_by_database,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& [db_id, tables] : tables_by_database) {
    for (const auto& table : tables) {
      if (import_buffers.find("database_id") != import_buffers.end()) {
        import_buffers["database_id"]->addInt(db_id);
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
      if (import_buffers.find("column_count") != import_buffers.end()) {
        import_buffers["column_count"]->addInt(table.nColumns);
      }
      if (import_buffers.find("is_view") != import_buffers.end()) {
        import_buffers["is_view"]->addBoolean(table.isView);
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
    }
  }
}

void populate_import_buffers_for_catalog_dashboards(
    const std::map<int32_t, std::vector<DashboardDescriptor>>& dashboards_by_database,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& [db_id, dashboards] : dashboards_by_database) {
    for (const auto& dashboard : dashboards) {
      if (import_buffers.find("database_id") != import_buffers.end()) {
        import_buffers["database_id"]->addInt(db_id);
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
      if (import_buffers.find("last_updated_at") != import_buffers.end()) {
        auto& buffer = import_buffers["last_updated_at"];
        import_buffers["last_updated_at"]->add_value(
            buffer->getColumnDesc(), dashboard.updateTime, false, {});
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
    if (import_buffers.find("object_name") != import_buffers.end()) {
      import_buffers["object_name"]->addString(permission.objectName);
    }
    if (import_buffers.find("object_id") != import_buffers.end()) {
      import_buffers["object_id"]->addInt(permission.objectId);
    }
    if (import_buffers.find("object_owner_id") != import_buffers.end()) {
      import_buffers["object_owner_id"]->addInt(permission.objectOwnerId);
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
    if (catalog->name() != INFORMATION_SCHEMA_DB) {
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
    for (const auto& dashboard : catalog->getAllDashboardsMetadataCopy()) {
      dashboards_by_database[catalog->getDatabaseId()].emplace_back(dashboard);
    }
  }
  return dashboards_by_database;
}

std::map<std::string, std::vector<std::string>> get_all_role_assignments() {
  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  const auto& users = sys_catalog.getAllUserMetadata();
  std::map<std::string, std::vector<std::string>> user_names_by_role;
  for (const auto& user : users) {
    for (const auto& role : sys_catalog.getRoles(false, user.isSuper, user.userName)) {
      user_names_by_role[role].emplace_back(user.userName);
    }
  }
  return user_names_by_role;
}

void set_default_chunk_metadata(int32_t db_id,
                                const ForeignTable* foreign_table,
                                size_t row_count,
                                ChunkMetadataVector& chunk_metadata_vector) {
  foreign_storage::ForeignTableSchema schema(db_id, foreign_table);
  for (auto column : schema.getLogicalColumns()) {
    ChunkKey chunk_key = {db_id, foreign_table->tableId, column->columnId, 0};
    if (column->columnType.is_varlen_indeed()) {
      chunk_key.emplace_back(1);
    }
    ForeignStorageBuffer empty_buffer;
    // Use default encoder metadata
    empty_buffer.initEncoder(column->columnType);
    auto chunk_metadata = empty_buffer.getEncoder()->getMetadata(column->columnType);
    chunk_metadata->numElements = row_count;
    if (!column->columnType.is_varlen_indeed()) {
      chunk_metadata->numBytes = column->columnType.get_size() * row_count;
    }
    if (column->columnType.is_array()) {
      ForeignStorageBuffer scalar_buffer;
      scalar_buffer.initEncoder(column->columnType.get_elem_type());
      auto scalar_metadata =
          scalar_buffer.getEncoder()->getMetadata(column->columnType.get_elem_type());
      chunk_metadata->chunkStats.min = scalar_metadata->chunkStats.min;
      chunk_metadata->chunkStats.max = scalar_metadata->chunkStats.max;
    }
    chunk_metadata->chunkStats.has_nulls = true;
    chunk_metadata_vector.emplace_back(chunk_key, chunk_metadata);
  }
}

void initialize_chunks(std::map<ChunkKey, Chunk_NS::Chunk>& chunks,
                       const ChunkToBufferMap& buffers,
                       size_t row_count,
                       std::set<const ColumnDescriptor*>& columns_to_parse,
                       int32_t fragment_id,
                       const Catalog_Namespace::Catalog& catalog) {
  for (auto& [chunk_key, buffer] : buffers) {
    CHECK_EQ(fragment_id, chunk_key[CHUNK_KEY_FRAGMENT_IDX]);
    const auto column = catalog.getMetadataForColumnUnlocked(
        chunk_key[CHUNK_KEY_TABLE_IDX], chunk_key[CHUNK_KEY_COLUMN_IDX]);
    if (is_varlen_index_key(chunk_key)) {
      continue;
    }
    chunks[chunk_key] = Chunk_NS::Chunk{column};
    if (column->columnType.is_varlen_indeed()) {
      CHECK(is_varlen_data_key(chunk_key));
      size_t index_offset_size{0};
      if (column->columnType.is_string()) {
        index_offset_size = sizeof(StringOffsetT);
      } else if (column->columnType.is_array()) {
        index_offset_size = sizeof(ArrayOffsetT);
      } else {
        UNREACHABLE() << "Unexpected column type: " << column->columnType.to_string();
      }
      ChunkKey index_chunk_key = chunk_key;
      index_chunk_key[CHUNK_KEY_VARLEN_IDX] = 2;
      CHECK(buffers.find(index_chunk_key) != buffers.end());
      AbstractBuffer* index_buffer = buffers.find(index_chunk_key)->second;
      index_buffer->reserve(index_offset_size * row_count + 1);
      chunks[chunk_key].setIndexBuffer(index_buffer);
    }

    if (!column->columnType.is_varlen_indeed()) {
      buffer->reserve(column->columnType.get_size() * row_count);
    }
    chunks[chunk_key].setBuffer(buffer);
    chunks[chunk_key].initEncoder();
    columns_to_parse.emplace(column);
  }
}

void initialize_import_buffers(
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>>& import_buffers,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers_map,
    const std::set<const ColumnDescriptor*>& columns_to_parse,
    const Catalog_Namespace::Catalog& catalog) {
  for (const auto column : columns_to_parse) {
    StringDictionary* string_dictionary = nullptr;
    if (column->columnType.is_dict_encoded_string() ||
        (column->columnType.is_array() && IS_STRING(column->columnType.get_subtype()) &&
         column->columnType.get_compression() == kENCODING_DICT)) {
      auto dict_descriptor =
          catalog.getMetadataForDictUnlocked(column->columnType.get_comp_param(), true);
      string_dictionary = dict_descriptor->stringDict.get();
    }
    import_buffers.emplace_back(
        std::make_unique<import_export::TypedImportBuffer>(column, string_dictionary));
    import_buffers_map[column->columnName] = import_buffers.back().get();
  }
}
}  // namespace

void InternalCatalogDataWrapper::populateChunkMetadata(
    ChunkMetadataVector& chunk_metadata_vector) {
  auto timer = DEBUG_TIMER(__func__);
  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  auto catalog = sys_catalog.getCatalog(db_id_);
  CHECK(catalog);
  CHECK_EQ(catalog->name(), INFORMATION_SCHEMA_DB);

  initializeObjectsForTable(foreign_table_->tableName);
  set_default_chunk_metadata(db_id_, foreign_table_, row_count_, chunk_metadata_vector);
}

void InternalCatalogDataWrapper::populateChunkBuffers(
    const ChunkToBufferMap& required_buffers,
    const ChunkToBufferMap& optional_buffers) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(optional_buffers.empty());

  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  auto catalog = sys_catalog.getCatalog(db_id_);
  CHECK(catalog);
  CHECK_EQ(catalog->name(), INFORMATION_SCHEMA_DB);

  auto fragment_id = required_buffers.begin()->first[CHUNK_KEY_FRAGMENT_IDX];
  CHECK_EQ(fragment_id, 0);

  std::map<ChunkKey, Chunk_NS::Chunk> chunks;
  std::set<const ColumnDescriptor*> columns_to_parse;
  initialize_chunks(
      chunks, required_buffers, row_count_, columns_to_parse, fragment_id, *catalog);

  // initialize import buffers from columns.
  std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
  std::map<std::string, import_export::TypedImportBuffer*> import_buffers_map;
  initialize_import_buffers(
      import_buffers, import_buffers_map, columns_to_parse, *catalog);
  populateChunkBuffersForTable(foreign_table_->tableName, import_buffers_map);

  auto column_id_to_data_blocks_map =
      TextFileBufferParser::convertImportBuffersToDataBlocks(import_buffers);
  for (auto& [chunk_key, chunk] : chunks) {
    auto data_block_entry =
        column_id_to_data_blocks_map.find(chunk_key[CHUNK_KEY_COLUMN_IDX]);
    CHECK(data_block_entry != column_id_to_data_blocks_map.end());
    chunk.appendData(data_block_entry->second, row_count_, 0);
    auto cd = chunk.getColumnDesc();
    if (!cd->columnType.is_varlen_indeed()) {
      CHECK(foreign_table_->fragmenter);
      auto metadata = chunk.getBuffer()->getEncoder()->getMetadata(cd->columnType);
      foreign_table_->fragmenter->updateColumnChunkMetadata(cd, fragment_id, metadata);
    }
    chunk.setBuffer(nullptr);
    chunk.setIndexBuffer(nullptr);
  }
}

void InternalCatalogDataWrapper::initializeObjectsForTable(
    const std::string& table_name) {
  row_count_ = 0;
  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  if (foreign_table_->tableName == Catalog_Namespace::USERS_SYS_TABLE_NAME) {
    users_.clear();
    users_ = sys_catalog.getAllUserMetadataUnlocked();
    row_count_ = users_.size();
  } else if (foreign_table_->tableName == Catalog_Namespace::TABLES_SYS_TABLE_NAME) {
    tables_by_database_.clear();
    tables_by_database_ = get_all_tables();
    for (const auto& [db_id, tables] : tables_by_database_) {
      row_count_ += tables.size();
    }
  } else if (foreign_table_->tableName == Catalog_Namespace::DASHBOARDS_SYS_TABLE_NAME) {
    dashboards_by_database_.clear();
    dashboards_by_database_ = get_all_dashboards();
    for (const auto& [db_id, dashboards] : dashboards_by_database_) {
      row_count_ += dashboards.size();
    }
  } else if (foreign_table_->tableName == Catalog_Namespace::PERMISSIONS_SYS_TABLE_NAME) {
    object_permissions_.clear();
    object_permissions_ = sys_catalog.getMetadataForAllObjectsUnlocked();
    row_count_ = object_permissions_.size();
  } else if (foreign_table_->tableName == Catalog_Namespace::DATABASES_SYS_TABLE_NAME) {
    databases_.clear();
    databases_ = sys_catalog.getAllDBMetadataUnlocked();
    row_count_ = databases_.size();
  } else if (foreign_table_->tableName == Catalog_Namespace::ROLES_SYS_TABLE_NAME) {
    roles_.clear();
    roles_ = sys_catalog.getCreatedRolesUnlocked();
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

std::string InternalCatalogDataWrapper::getSerializedDataWrapper() const {
  return {};
}

void InternalCatalogDataWrapper::restoreDataWrapperInternals(
    const std::string& file_path,
    const ChunkMetadataVector& chunk_metadata) {}

bool InternalCatalogDataWrapper::isRestored() const {
  return false;
}
}  // namespace foreign_storage
