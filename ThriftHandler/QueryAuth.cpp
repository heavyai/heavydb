/*
 * Copyright 2024 HEAVY.AI, Inc.
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

#include "QueryAuth.h"

#include "QueryEngine/QueryPhysicalInputsCollector.h"
#include "QueryEngine/RelAlgDag.h"
#include "Shared/DbObjectKeys.h"
#include "ThriftHandler/QueryState.h"

namespace {

std::map<std::pair<std::string, std::string>, std::set<std::string>>
get_column_set_from_rel_alg_dag(RelAlgDag* query_dag) {
  std::map<std::pair<std::string, std::string>, std::set<std::string>> columns_set;
  const auto& ra_node = query_dag->getRootNode();
  for (const auto [col_id, table_id, db_id] : get_physical_inputs(&ra_node)) {
    const auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
    CHECK(catalog);
    const auto table = catalog->getMetadataForTable(table_id, false);
    const auto spi_col_id = catalog->getColumnIdBySpi(table_id, col_id);
    auto column_name = catalog->getColumnName(table_id, spi_col_id);
    CHECK(column_name.has_value());
    auto table_name = table->tableName;
    auto db_name = catalog->name();
    columns_set[{db_name, table_name}].insert(column_name.value());
  }
  return columns_set;
}

std::vector<query_auth::CapturedColumns> convert_column_set_to_captured_columns(
    const std::map<std::pair<std::string, std::string>, std::set<std::string>>&
        column_map) {
  std::vector<query_auth::CapturedColumns> captured_columns;
  for (const auto& [db_table_pair, columns] : column_map) {
    const auto [db_name, table_name] = db_table_pair;
    auto& current_captured_columns = captured_columns.emplace_back();
    current_captured_columns.db_name = db_name;
    current_captured_columns.table_name = table_name;
    current_captured_columns.column_names =
        std::vector<std::string>{columns.begin(), columns.end()};
  }
  return captured_columns;
}

std::vector<query_auth::CapturedColumns> capture_columns(RelAlgDag* query_dag) {
  auto column_set = get_column_set_from_rel_alg_dag(query_dag);
  return convert_column_set_to_captured_columns(column_set);
}

void check_db_access(const Catalog_Namespace::SessionInfo& session_info,
                     const Catalog_Namespace::Catalog& accessed_catalog) {
  const auto db_name = accessed_catalog.name();
  DBObject db_object(db_name, DatabaseDBObjectType);
  db_object.loadKey(accessed_catalog);
  db_object.setPrivileges(AccessPrivileges::ACCESS);

  const auto& user = session_info.get_currentUser();
  if (!Catalog_Namespace::SysCatalog::instance().checkPrivileges(user, {db_object})) {
    throw std::runtime_error("Unauthorized Access: user " + user.userLoggable() +
                             " is not allowed to access database " + db_name + ".");
  }
}

bool has_any_column_privilege(const Catalog_Namespace::Catalog* catalog,
                              const TableDescriptor* table,
                              const Catalog_Namespace::SessionInfo& session_info,
                              const std::optional<AccessPrivileges>& column_privs) {
  if (!column_privs.has_value()) {
    return false;
  }
  CHECK(column_privs.value().hasAny())
      << "No valid privilege defined for column privileges on table " + table->tableName;
  for (const auto cd :
       catalog->getAllColumnMetadataForTable(table->tableId, true, false, true)) {
    std::vector<DBObject> priv_objects;
    DBObject column_object(
        table->tableName, cd->columnName, DBObjectType::ColumnDBObjectType);
    column_object.loadKey(*catalog);
    column_object.setPrivileges(column_privs.value());
    priv_objects.emplace_back(column_object);
    if (Catalog_Namespace::SysCatalog::instance().checkPrivileges(
            session_info.get_currentUser(), priv_objects)) {
      return true;
    }
  }
  return false;
}

bool is_view_in_use(const std::vector<std::vector<std::string>>& table_or_view_names,
                    const Catalog_Namespace::SessionInfo& session_info) {
  for (auto table_or_view : table_or_view_names) {
    // Calcite returns table names in the form of a {table_name, database_name} vector.
    const auto catalog =
        Catalog_Namespace::SysCatalog::instance().getCatalog(table_or_view[1]);
    CHECK(catalog);
    check_db_access(session_info, *catalog);

    const TableDescriptor* tableMeta =
        catalog->getMetadataForTable(table_or_view[0], false);

    if (!tableMeta) {
      throw std::runtime_error("unknown table or view: " + table_or_view[0]);
    }

    if (tableMeta->isView) {
      return true;
    }
  }
  return false;
}

void check_permissions_for_table(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::vector<std::vector<std::string>>& tableOrViewNames,
    AccessPrivileges tablePrivs,
    AccessPrivileges viewPrivs,
    const std::optional<AccessPrivileges>& column_privs = std::nullopt) {
  for (auto tableOrViewName : tableOrViewNames) {
    // Calcite returns table names in the form of a {table_name, database_name} vector.
    const auto catalog =
        Catalog_Namespace::SysCatalog::instance().getCatalog(tableOrViewName[1]);
    CHECK(catalog);
    check_db_access(session_info, *catalog);

    const TableDescriptor* tableMeta =
        catalog->getMetadataForTable(tableOrViewName[0], false);

    if (!tableMeta) {
      throw std::runtime_error("Unknown table or view: " + tableOrViewName[0]);
    }

    DBObjectKey key;
    key.dbId = catalog->getCurrentDB().dbId;
    key.permissionType = tableMeta->isView ? DBObjectType::ViewDBObjectType
                                           : DBObjectType::TableDBObjectType;
    key.objectId = tableMeta->tableId;
    AccessPrivileges privs = tableMeta->isView ? viewPrivs : tablePrivs;
    DBObject dbobject(key, privs, tableMeta->userId);
    std::vector<DBObject> privObjects{dbobject};

    if (!privs.hasAny()) {
      throw std::runtime_error("Operation not supported for object " +
                               tableOrViewName[0]);
    }

    if (!Catalog_Namespace::SysCatalog::instance().checkPrivileges(
            session_info.get_currentUser(), privObjects) &&
        !has_any_column_privilege(catalog.get(), tableMeta, session_info, column_privs)) {
      throw std::runtime_error("Violation of access privileges: user " +
                               session_info.get_currentUser().userLoggable() +
                               " has no proper privileges for object " +
                               tableOrViewName[0]);
    }
  }
}

void check_accessed_table_and_view_privileges(
    query_state::QueryStateProxy query_state_proxy,
    TPlanResult plan,
    const bool check_column_level_privileges) {
  AccessPrivileges NOOP;
  // check the individual tables
  auto const session_ptr = query_state_proxy->getConstSessionInfo();
  // TODO: Replace resolved tables vector with a `FullyQualifiedTableName` struct.
  check_permissions_for_table(
      *session_ptr,
      plan.primary_accessed_objects.tables_selected_from,
      AccessPrivileges::SELECT_FROM_TABLE,
      AccessPrivileges::SELECT_FROM_VIEW,
      (check_column_level_privileges
           ? std::optional<AccessPrivileges>{AccessPrivileges::SELECT_COLUMN_FROM_TABLE}
           : std::nullopt));
  check_permissions_for_table(*session_ptr,
                              plan.primary_accessed_objects.tables_inserted_into,
                              AccessPrivileges::INSERT_INTO_TABLE,
                              NOOP);
  check_permissions_for_table(*session_ptr,
                              plan.primary_accessed_objects.tables_updated_in,
                              AccessPrivileges::UPDATE_IN_TABLE,
                              NOOP);
  check_permissions_for_table(*session_ptr,
                              plan.primary_accessed_objects.tables_deleted_from,
                              AccessPrivileges::DELETE_FROM_TABLE,
                              NOOP);
}

void check_accessed_table_and_column_privileges(
    query_state::QueryStateProxy query_state_proxy,
    const TPlanResult& plan) {
  auto session_ptr = query_state_proxy->getConstSessionInfo();

  std::unique_ptr<RelAlgDag> query_dag;
  query_dag = RelAlgDagBuilder::buildDag(plan.plan_result, true);

  auto captured_columns = capture_columns(query_dag.get());
  for (const auto& captured_column : captured_columns) {
    const auto catalog =
        Catalog_Namespace::SysCatalog::instance().getCatalog(captured_column.db_name);
    CHECK(catalog);
    const auto table = catalog->getMetadataForTable(captured_column.table_name, false);
    CHECK(table);

    DBObjectKey key;
    key.dbId = catalog->getCurrentDB().dbId;
    CHECK(!table->isView);
    key.permissionType = DBObjectType::TableDBObjectType;
    key.objectId = table->tableId;
    AccessPrivileges privs = AccessPrivileges::SELECT_FROM_TABLE;
    DBObject db_object(key, privs, table->userId);
    std::vector<DBObject> priv_objects{db_object};

    auto& current_user = session_ptr->get_currentUser();
    if (!Catalog_Namespace::SysCatalog::instance().checkPrivileges(current_user,
                                                                   priv_objects)) {
      // Table level privilege failed, check column privileges
      for (const auto& column_name : captured_column.column_names) {
        const auto cd = catalog->getMetadataForColumn(table->tableId, column_name);
        if (!cd) {
          throw std::runtime_error("Encountered column " + column_name +
                                   " reported by Calcite, but column "
                                   "does not exist. Database: " +
                                   catalog->name() + ", Table: " + table->tableName);
        }
        std::vector<DBObject> priv_objects;
        DBObject column_object(
            table->tableName, cd->columnName, DBObjectType::ColumnDBObjectType);
        column_object.loadKey(*catalog);
        column_object.setPrivileges(AccessPrivileges::SELECT_COLUMN_FROM_TABLE);
        priv_objects.emplace_back(column_object);
        if (!Catalog_Namespace::SysCatalog::instance().checkPrivileges(current_user,
                                                                       priv_objects)) {
          throw std::runtime_error(
              "Violation of access privileges: user " + current_user.userLoggable() +
              " has no proper privileges for object " + table->tableName);
        }
      }
    }
  }
}

bool should_check_column_level_privileges(query_state::QueryStateProxy query_state_proxy,
                                          const TPlanResult& plan) {
  return g_enable_column_level_security &&
         !is_view_in_use(plan.primary_accessed_objects.tables_selected_from,
                         *query_state_proxy->getConstSessionInfo());
}

}  // namespace

namespace query_auth {

std::vector<CapturedColumns> capture_columns(const std::string& query_ra) {
  std::unique_ptr<RelAlgDag> query_dag = RelAlgDagBuilder::buildDag(query_ra, true);
  return ::capture_columns(query_dag.get());
}

void check_access_privileges(query_state::QueryStateProxy query_state_proxy,
                             const TPlanResult& plan) {
  const bool do_column_level_privilege_check =
      should_check_column_level_privileges(query_state_proxy, plan);

  check_accessed_table_and_view_privileges(
      query_state_proxy, plan, do_column_level_privilege_check);

  if (!do_column_level_privilege_check || !plan.is_rel_alg) {
    // Example cases that will reach this point in execution include ITAS,
    // CTAS, COPY TO.
    // These cases defer the column-level privilege check below to when the
    // projection query is processed.
    return;
  }

  check_accessed_table_and_column_privileges(query_state_proxy, plan);
}

}  // namespace query_auth
