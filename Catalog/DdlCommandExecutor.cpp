/*
 * Copyright 2020 OmniSci, Inc.
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

#include "DdlCommandExecutor.h"

#include <boost/algorithm/string/predicate.hpp>

#include "Catalog/Catalog.h"
#include "Catalog/SysCatalog.h"
#include "DataMgr/ForeignStorage/CsvDataWrapper.h"
#include "LockMgr/LockMgr.h"
#include "Parser/ParserNode.h"
#include "Shared/StringTransform.h"

extern bool g_enable_fsi;

bool DdlCommand::isDefaultServer(const std::string& server_name) {
  return boost::iequals(server_name.substr(0, 7), "omnisci");
}

namespace {
void set_headers(TQueryResult& _return, const std::vector<std::string>& headers) {
  TRowDescriptor row_descriptor;
  for (const auto& header : headers) {
    TColumnType column_type{};
    column_type.col_name = header;
    column_type.col_type.type = TDatumType::type::STR;
    row_descriptor.push_back(column_type);

    _return.row_set.columns.emplace_back();
  }
  _return.row_set.row_desc = row_descriptor;
  _return.row_set.is_columnar = true;
}

void add_row(TQueryResult& _return, const std::vector<std::string>& row) {
  for (size_t i = 0; i < row.size(); i++) {
    _return.row_set.columns[i].data.str_col.emplace_back(row[i]);
    _return.row_set.columns[i].nulls.emplace_back(false);
  }
}
}  // namespace

DdlCommandExecutor::DdlCommandExecutor(
    const std::string& ddl_statement,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : session_ptr_(session_ptr) {
  CHECK(!ddl_statement.empty());
  ddl_query_.Parse(ddl_statement);
  CHECK(ddl_query_.IsObject());
  CHECK(ddl_query_.HasMember("payload"));
  CHECK(ddl_query_["payload"].IsObject());
  const auto& payload = ddl_query_["payload"].GetObject();
  CHECK(payload.HasMember("command"));
  CHECK(payload["command"].IsString());
}

void DdlCommandExecutor::execute(TQueryResult& _return) {
  const auto& payload = ddl_query_["payload"].GetObject();
  const auto& ddl_command = std::string_view(payload["command"].GetString());
  if (ddl_command == "CREATE_SERVER") {
    CreateForeignServerCommand{payload, session_ptr_}.execute(_return);
  } else if (ddl_command == "DROP_SERVER") {
    DropForeignServerCommand{payload, session_ptr_}.execute(_return);
  } else if (ddl_command == "CREATE_FOREIGN_TABLE") {
    CreateForeignTableCommand{payload, session_ptr_}.execute(_return);
  } else if (ddl_command == "DROP_FOREIGN_TABLE") {
    DropForeignTableCommand{payload, session_ptr_}.execute(_return);
  } else if (ddl_command == "SHOW_TABLES") {
    ShowTablesCommand{payload, session_ptr_}.execute(_return);
  } else if (ddl_command == "SHOW_DATABASES") {
    ShowDatabasesCommand{payload, session_ptr_}.execute(_return);
  } else if (ddl_command == "SHOW_SERVERS") {
    ShowForeignServersCommand{payload, session_ptr_}.execute(_return);
  } else if (ddl_command == "ALTER_SERVER") {
    AlterForeignServerCommand{payload, session_ptr_}.execute(_return);
  } else {
    throw std::runtime_error("Unsupported DDL command");
  }
}

bool DdlCommandExecutor::isShowUserSessions() {
  const auto& payload = ddl_query_["payload"].GetObject();
  const auto& ddl_command = std::string_view(payload["command"].GetString());
  return (ddl_command == "SHOW_USER_SESSIONS");
}

CreateForeignServerCommand::CreateForeignServerCommand(
    const rapidjson::Value& ddl_payload,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_payload, session_ptr) {
  CHECK(ddl_payload.HasMember("serverName"));
  CHECK(ddl_payload["serverName"].IsString());
  CHECK(ddl_payload.HasMember("dataWrapper"));
  CHECK(ddl_payload["dataWrapper"].IsString());
  CHECK(ddl_payload.HasMember("options"));
  CHECK(ddl_payload["options"].IsObject());
  CHECK(ddl_payload.HasMember("ifNotExists"));
  CHECK(ddl_payload["ifNotExists"].IsBool());
}

void CreateForeignServerCommand::execute(TQueryResult& _return) {
  std::string server_name = ddl_payload_["serverName"].GetString();
  if (isDefaultServer(server_name)) {
    throw std::runtime_error{"Server names cannot start with \"omnisci\"."};
  }
  bool if_not_exists = ddl_payload_["ifNotExists"].GetBool();
  if (session_ptr_->getCatalog().getForeignServer(server_name)) {
    if (if_not_exists) {
      return;
    } else {
      throw std::runtime_error{"A foreign server with name \"" + server_name +
                               "\" already exists."};
    }
  }
  // check access privileges
  if (!session_ptr_->checkDBAccessPrivileges(DBObjectType::ServerDBObjectType,
                                             AccessPrivileges::CREATE_SERVER)) {
    throw std::runtime_error("Server " + std::string(server_name) +
                             " will not be created. User has no create privileges.");
  }

  auto& current_user = session_ptr_->get_currentUser();
  auto foreign_server = std::make_unique<foreign_storage::ForeignServer>();
  foreign_server->data_wrapper_type = to_upper(ddl_payload_["dataWrapper"].GetString());
  foreign_server->name = server_name;
  foreign_server->user_id = current_user.userId;
  foreign_server->populateOptionsMap(ddl_payload_["options"]);
  foreign_server->validate();

  auto& catalog = session_ptr_->getCatalog();
  catalog.createForeignServer(std::move(foreign_server),
                              ddl_payload_["ifNotExists"].GetBool());
  Catalog_Namespace::SysCatalog::instance().createDBObject(
      current_user, server_name, ServerDBObjectType, catalog);
}

AlterForeignServerCommand::AlterForeignServerCommand(
    const rapidjson::Value& ddl_payload,
    std::shared_ptr<const Catalog_Namespace::SessionInfo> session_ptr)
    : DdlCommand(ddl_payload, session_ptr) {
  CHECK(ddl_payload.HasMember("serverName"));
  CHECK(ddl_payload["serverName"].IsString());
  CHECK(ddl_payload.HasMember("alterType"));
  CHECK(ddl_payload["alterType"].IsString());
  if (ddl_payload["alterType"] == "SET_OPTIONS") {
    CHECK(ddl_payload.HasMember("options"));
    CHECK(ddl_payload["options"].IsObject());
  } else if (ddl_payload["alterType"] == "SET_DATA_WRAPPER") {
    CHECK(ddl_payload.HasMember("dataWrapper"));
    CHECK(ddl_payload["dataWrapper"].IsString());
  } else if (ddl_payload["alterType"] == "RENAME_SERVER") {
    CHECK(ddl_payload.HasMember("newServerName"));
    CHECK(ddl_payload["newServerName"].IsString());
  } else if (ddl_payload["alterType"] == "CHANGE_OWNER") {
    CHECK(ddl_payload.HasMember("newOwner"));
    CHECK(ddl_payload["newOwner"].IsString());
  } else {
    UNREACHABLE();  // not-implemented alterType
  }
}

void AlterForeignServerCommand::execute(TQueryResult& _return) {
  std::string server_name = ddl_payload_["serverName"].GetString();
  if (isDefaultServer(server_name)) {
    throw std::runtime_error{"OmniSci default servers cannot be altered."};
  }
  if (!session_ptr_->getCatalog().getForeignServer(server_name)) {
    throw std::runtime_error{"Foreign server with name \"" + server_name +
                             "\" does not exist and can not be altered."};
  }
  if (!hasAlterServerPrivileges()) {
    throw std::runtime_error("Server " + server_name +
                             " can not be altered. User has no ALTER SERVER privileges.");
  }
  std::string alter_type = ddl_payload_["alterType"].GetString();
  if (alter_type == "CHANGE_OWNER") {
    changeForeignServerOwner();
  } else if (alter_type == "SET_DATA_WRAPPER") {
    setForeignServerDataWrapper();
  } else if (alter_type == "SET_OPTIONS") {
    setForeignServerOptions();
  } else if (alter_type == "RENAME_SERVER") {
    renameForeignServer();
  }
}

void AlterForeignServerCommand::changeForeignServerOwner() {
  std::string server_name = ddl_payload_["serverName"].GetString();
  std::string new_owner = ddl_payload_["newOwner"].GetString();
  auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
  if (!session_ptr_->get_currentUser().isSuper) {
    throw std::runtime_error(
        "Only a super user can change a foreign server's owner. "
        "Current user is not a super-user. "
        "Foreign server with name \"" +
        server_name + "\" will not have owner changed.");
  }
  Catalog_Namespace::UserMetadata user, original_owner;
  if (!sys_cat.getMetadataForUser(new_owner, user)) {
    throw std::runtime_error("User with username \"" + new_owner + "\" does not exist. " +
                             "Foreign server with name \"" + server_name +
                             "\" can not have owner changed.");
  }
  auto& cat = session_ptr_->getCatalog();
  // get original owner metadata
  bool original_owner_exists = sys_cat.getMetadataForUserById(
      cat.getForeignServer(server_name)->user_id, original_owner);
  // update catalog
  cat.changeForeignServerOwner(server_name, user.userId);
  try {
    // update permissions
    DBObject db_object(server_name, DBObjectType::ServerDBObjectType);
    sys_cat.changeDBObjectOwnership(
        user, original_owner, db_object, cat, original_owner_exists);
  } catch (const std::runtime_error& e) {
    // update permissions failed, revert catalog update
    cat.changeForeignServerOwner(server_name, original_owner.userId);
    throw;
  }
}

void AlterForeignServerCommand::renameForeignServer() {
  std::string server_name = ddl_payload_["serverName"].GetString();
  std::string new_server_name = ddl_payload_["newServerName"].GetString();
  if (isDefaultServer(new_server_name)) {
    throw std::runtime_error{"OmniSci prefix can not be used for new name of server."};
  }
  auto& cat = session_ptr_->getCatalog();
  // check for a conflicting server
  if (cat.getForeignServer(new_server_name)) {
    throw std::runtime_error("Foreign server with name \"" + server_name +
                             "\" can not be renamed to \"" + new_server_name + "\"." +
                             "Foreign server with name \"" + new_server_name +
                             "\" exists.");
  }
  // update catalog
  cat.renameForeignServer(server_name, new_server_name);
  try {
    // migrate object privileges
    auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
    sys_cat.renameDBObject(server_name,
                           new_server_name,
                           DBObjectType::ServerDBObjectType,
                           cat.getForeignServer(new_server_name)->id,
                           cat);
  } catch (const std::runtime_error& e) {
    // permission migration failed, revert catalog update
    cat.renameForeignServer(new_server_name, server_name);
    throw;
  }
}

void AlterForeignServerCommand::setForeignServerOptions() {
  std::string server_name = ddl_payload_["serverName"].GetString();
  auto& cat = session_ptr_->getCatalog();
  // update catalog
  const auto foreign_server = cat.getForeignServer(server_name);
  foreign_storage::OptionsContainer opt;
  opt.populateOptionsMap(foreign_server->getOptionsAsJsonString());
  opt.populateOptionsMap(ddl_payload_["options"]);
  cat.setForeignServerOptions(server_name, opt.getOptionsAsJsonString());
}

void AlterForeignServerCommand::setForeignServerDataWrapper() {
  std::string server_name = ddl_payload_["serverName"].GetString();
  std::string data_wrapper = ddl_payload_["dataWrapper"].GetString();
  auto& cat = session_ptr_->getCatalog();
  // update catalog
  cat.setForeignServerDataWrapper(server_name, data_wrapper);
}

bool AlterForeignServerCommand::hasAlterServerPrivileges() {
  // TODO: implement `GRANT/REVOKE ALTER_SERVER` DDL commands
  std::string server_name = ddl_payload_["serverName"].GetString();
  return session_ptr_->checkDBAccessPrivileges(
      DBObjectType::ServerDBObjectType, AccessPrivileges::ALTER_SERVER, server_name);
}

DropForeignServerCommand::DropForeignServerCommand(
    const rapidjson::Value& ddl_payload,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_payload, session_ptr) {
  CHECK(ddl_payload.HasMember("serverName"));
  CHECK(ddl_payload["serverName"].IsString());
  CHECK(ddl_payload.HasMember("ifExists"));
  CHECK(ddl_payload["ifExists"].IsBool());
}

void DropForeignServerCommand::execute(TQueryResult& _return) {
  std::string server_name = ddl_payload_["serverName"].GetString();
  if (isDefaultServer(server_name)) {
    throw std::runtime_error{"OmniSci default servers cannot be dropped."};
  }
  bool if_exists = ddl_payload_["ifExists"].GetBool();
  if (!session_ptr_->getCatalog().getForeignServer(server_name)) {
    if (if_exists) {
      return;
    } else {
      throw std::runtime_error{"Foreign server with name \"" + server_name +
                               "\" does not exist."};
    }
  }
  // check access privileges
  if (!session_ptr_->checkDBAccessPrivileges(
          DBObjectType::ServerDBObjectType, AccessPrivileges::DROP_SERVER, server_name)) {
    throw std::runtime_error("Server " + server_name +
                             " will not be dropped. User has no DROP SERVER privileges.");
  }
  Catalog_Namespace::SysCatalog::instance().revokeDBObjectPrivilegesFromAll(
      DBObject(server_name, ServerDBObjectType), session_ptr_->get_catalog_ptr().get());
  session_ptr_->getCatalog().dropForeignServer(ddl_payload_["serverName"].GetString());
}

SQLTypes JsonColumnSqlType::getSqlType(const rapidjson::Value& data_type) {
  CHECK(data_type.IsObject());
  CHECK(data_type.HasMember("type"));
  CHECK(data_type["type"].IsString());

  std::string type = data_type["type"].GetString();
  if (boost::iequals(type, "ARRAY")) {
    CHECK(data_type.HasMember("array"));
    CHECK(data_type["array"].IsObject());

    const auto& array = data_type["array"].GetObject();
    CHECK(array.HasMember("elementType"));
    CHECK(array["elementType"].IsString());
    type = array["elementType"].GetString();
  }
  return getSqlType(type);
}

SQLTypes JsonColumnSqlType::getSqlType(const std::string& type) {
  if (boost::iequals(type, "BIGINT")) {
    return kBIGINT;
  }
  if (boost::iequals(type, "BOOLEAN")) {
    return kBOOLEAN;
  }
  if (boost::iequals(type, "DATE")) {
    return kDATE;
  }
  if (boost::iequals(type, "DECIMAL")) {
    return kDECIMAL;
  }
  if (boost::iequals(type, "DOUBLE")) {
    return kDOUBLE;
  }
  if (boost::iequals(type, "FLOAT")) {
    return kFLOAT;
  }
  if (boost::iequals(type, "INTEGER")) {
    return kINT;
  }
  if (boost::iequals(type, "LINESTRING")) {
    return kLINESTRING;
  }
  if (boost::iequals(type, "MULTIPOLYGON")) {
    return kMULTIPOLYGON;
  }
  if (boost::iequals(type, "POINT")) {
    return kPOINT;
  }
  if (boost::iequals(type, "POLYGON")) {
    return kPOLYGON;
  }
  if (boost::iequals(type, "SMALLINT")) {
    return kSMALLINT;
  }
  if (boost::iequals(type, "TEXT")) {
    return kTEXT;
  }
  if (boost::iequals(type, "TIME")) {
    return kTIME;
  }
  if (boost::iequals(type, "TIMESTAMP")) {
    return kTIMESTAMP;
  }
  if (boost::iequals(type, "TINYINT")) {
    return kTINYINT;
  }

  throw std::runtime_error{"Unsupported type \"" + type + "\" specified."};
}

int JsonColumnSqlType::getParam1(const rapidjson::Value& data_type) {
  int param1 = -1;
  CHECK(data_type.IsObject());
  if (data_type.HasMember("precision") && !data_type["precision"].IsNull()) {
    CHECK(data_type["precision"].IsInt());
    param1 = data_type["precision"].GetInt();
  } else if (auto type = getSqlType(data_type); IS_GEO(type)) {
    param1 = static_cast<int>(kGEOMETRY);
  }
  return param1;
}

int JsonColumnSqlType::getParam2(const rapidjson::Value& data_type) {
  int param2 = 0;
  CHECK(data_type.IsObject());
  if (data_type.HasMember("scale") && !data_type["scale"].IsNull()) {
    CHECK(data_type["scale"].IsInt());
    param2 = data_type["scale"].GetInt();
  } else if (auto type = getSqlType(data_type); IS_GEO(type) &&
                                                data_type.HasMember("coordinateSystem") &&
                                                !data_type["coordinateSystem"].IsNull()) {
    CHECK(data_type["coordinateSystem"].IsInt());
    param2 = data_type["coordinateSystem"].GetInt();
  }
  return param2;
}

bool JsonColumnSqlType::isArray(const rapidjson::Value& data_type) {
  CHECK(data_type.IsObject());
  CHECK(data_type.HasMember("type"));
  CHECK(data_type["type"].IsString());
  return boost::iequals(data_type["type"].GetString(), "ARRAY");
}

int JsonColumnSqlType::getArraySize(const rapidjson::Value& data_type) {
  int size = -1;
  if (isArray(data_type)) {
    CHECK(data_type.HasMember("array"));
    CHECK(data_type["array"].IsObject());

    const auto& array = data_type["array"].GetObject();
    if (array.HasMember("size") && !array["size"].IsNull()) {
      CHECK(array["size"].IsInt());
      size = array["size"].GetInt();
    }
  }
  return size;
}

std::string* JsonColumnEncoding::getEncodingName(const rapidjson::Value& data_type) {
  CHECK(data_type.IsObject());
  CHECK(data_type.HasMember("encoding"));
  CHECK(data_type["encoding"].IsObject());

  const auto& encoding = data_type["encoding"].GetObject();
  CHECK(encoding.HasMember("type"));
  CHECK(encoding["type"].IsString());
  return new std::string(encoding["type"].GetString());
}

int JsonColumnEncoding::getEncodingParam(const rapidjson::Value& data_type) {
  CHECK(data_type.IsObject());
  CHECK(data_type.HasMember("encoding"));
  CHECK(data_type["encoding"].IsObject());

  int encoding_size = 0;
  const auto& encoding = data_type["encoding"].GetObject();
  if (encoding.HasMember("size") && !encoding["size"].IsNull()) {
    CHECK(encoding["size"].IsInt());
    encoding_size = encoding["size"].GetInt();
  }
  return encoding_size;
}

CreateForeignTableCommand::CreateForeignTableCommand(
    const rapidjson::Value& ddl_payload,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_payload, session_ptr) {
  CHECK(ddl_payload.HasMember("serverName"));
  CHECK(ddl_payload["serverName"].IsString());
  CHECK(ddl_payload.HasMember("tableName"));
  CHECK(ddl_payload["tableName"].IsString());
  CHECK(ddl_payload.HasMember("ifNotExists"));
  CHECK(ddl_payload["ifNotExists"].IsBool());
  CHECK(ddl_payload.HasMember("columns"));
  CHECK(ddl_payload["columns"].IsArray());
}

void CreateForeignTableCommand::execute(TQueryResult& _return) {
  auto& catalog = session_ptr_->getCatalog();

  const std::string& table_name = ddl_payload_["tableName"].GetString();
  if (!session_ptr_->checkDBAccessPrivileges(DBObjectType::TableDBObjectType,
                                             AccessPrivileges::CREATE_TABLE)) {
    throw std::runtime_error(
        "Foreign table \"" + table_name +
        "\" will not be created. User has no CREATE TABLE privileges.");
  }

  bool if_not_exists = ddl_payload_["ifNotExists"].GetBool();
  if (!ddl_utils::validate_nonexistent_table(table_name, catalog, if_not_exists)) {
    return;
  }

  foreign_storage::ForeignTable foreign_table{};
  std::list<ColumnDescriptor> columns{};
  setColumnDetails(columns);
  setTableDetails(table_name, foreign_table, columns.size());
  catalog.createTable(foreign_table, columns, {}, true);

  // TODO (max): It's transactionally unsafe, should be fixed: we may create object w/o
  // privileges
  Catalog_Namespace::SysCatalog::instance().createDBObject(
      session_ptr_->get_currentUser(),
      foreign_table.tableName,
      TableDBObjectType,
      catalog);
}

void CreateForeignTableCommand::setTableDetails(const std::string& table_name,
                                                TableDescriptor& td,
                                                const size_t column_count) {
  ddl_utils::set_default_table_attributes(table_name, td, column_count);
  td.userId = session_ptr_->get_currentUser().userId;
  td.storageType = StorageType::FOREIGN_TABLE;
  td.hasDeletedCol = false;
  td.keyMetainfo = "[]";
  td.fragments = "";
  td.partitions = "";

  auto& foreign_table = dynamic_cast<foreign_storage::ForeignTable&>(td);
  const std::string server_name = ddl_payload_["serverName"].GetString();
  foreign_table.foreign_server = session_ptr_->getCatalog().getForeignServer(server_name);
  if (!foreign_table.foreign_server) {
    throw std::runtime_error{"Foreign server with name \"" + server_name +
                             "\" does not exist."};
  }

  if (ddl_payload_.HasMember("options") && !ddl_payload_["options"].IsNull()) {
    CHECK(ddl_payload_["options"].IsObject());
    foreign_table.populateOptionsMap(ddl_payload_["options"]);

    if (foreign_table.foreign_server->data_wrapper_type ==
        foreign_storage::DataWrapperType::CSV) {
      foreign_storage::CsvDataWrapper::validateOptions(&foreign_table);
    }
  }

  if (const auto it = foreign_table.options.find("FRAGMENT_SIZE");
      it != foreign_table.options.end()) {
    foreign_table.maxFragRows = std::stoi(it->second);
  }
}

void CreateForeignTableCommand::setColumnDetails(std::list<ColumnDescriptor>& columns) {
  std::unordered_set<std::string> column_names{};
  for (auto& column_def : ddl_payload_["columns"].GetArray()) {
    CHECK(column_def.IsObject());
    CHECK(column_def.HasMember("name"));
    CHECK(column_def["name"].IsString());
    const std::string& column_name = column_def["name"].GetString();

    CHECK(column_def.HasMember("dataType"));
    CHECK(column_def["dataType"].IsObject());

    JsonColumnSqlType sql_type{column_def["dataType"]};
    const auto& data_type = column_def["dataType"].GetObject();
    CHECK(data_type.HasMember("notNull"));
    CHECK(data_type["notNull"].IsBool());

    std::unique_ptr<JsonColumnEncoding> encoding;
    if (data_type.HasMember("encoding") && !data_type["encoding"].IsNull()) {
      CHECK(data_type["encoding"].IsObject());
      encoding = std::make_unique<JsonColumnEncoding>(column_def["dataType"]);
    }

    ColumnDescriptor cd;
    ddl_utils::validate_non_duplicate_column(column_name, column_names);
    ddl_utils::validate_non_reserved_keyword(column_name);
    ddl_utils::set_column_descriptor(
        column_name, cd, &sql_type, data_type["notNull"].GetBool(), encoding.get());
    columns.emplace_back(cd);
  }
}

DropForeignTableCommand::DropForeignTableCommand(
    const rapidjson::Value& ddl_payload,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_payload, session_ptr) {
  CHECK(ddl_payload.HasMember("tableName"));
  CHECK(ddl_payload["tableName"].IsString());
  CHECK(ddl_payload.HasMember("ifExists"));
  CHECK(ddl_payload["ifExists"].IsBool());
}

void DropForeignTableCommand::execute(TQueryResult& _return) {
  auto& catalog = session_ptr_->getCatalog();
  const std::string& table_name = ddl_payload_["tableName"].GetString();
  const TableDescriptor* td{nullptr};
  std::unique_ptr<lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>> td_with_lock;

  try {
    td_with_lock =
        std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>>(
            lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
                catalog, table_name, false));
    td = (*td_with_lock)();
  } catch (const std::runtime_error& e) {
    if (ddl_payload_["ifExists"].GetBool()) {
      return;
    } else {
      throw e;
    }
  }

  CHECK(td);
  CHECK(td_with_lock);

  if (!session_ptr_->checkDBAccessPrivileges(
          DBObjectType::TableDBObjectType, AccessPrivileges::DROP_TABLE, table_name)) {
    throw std::runtime_error(
        "Foreign table \"" + table_name +
        "\" will not be dropped. User has no DROP TABLE privileges.");
  }

  ddl_utils::validate_drop_table_type(td, ddl_utils::TableType::FOREIGN_TABLE);
  auto table_data_write_lock =
      lockmgr::TableDataLockMgr::getWriteLockForTable(catalog, table_name);
  catalog.dropTable(td);
}

ShowTablesCommand::ShowTablesCommand(
    const rapidjson::Value& ddl_payload,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_payload, session_ptr) {}

void ShowTablesCommand::execute(TQueryResult& _return) {
  // Get all table names in the same way as OmniSql \t command
  auto cat_ptr = session_ptr_->get_catalog_ptr();
  auto table_names =
      cat_ptr->getTableNamesForUser(session_ptr_->get_currentUser(), GET_PHYSICAL_TABLES);
  set_headers(_return, std::vector<std::string>{"table_name"});
  // Place table names in query result
  for (auto& table_name : table_names) {
    add_row(_return, std::vector<std::string>{table_name});
  }
}

ShowDatabasesCommand::ShowDatabasesCommand(
    const rapidjson::Value& ddl_payload,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_payload, session_ptr) {}

void ShowDatabasesCommand::execute(TQueryResult& _return) {
  const auto& user = session_ptr_->get_currentUser();
  const Catalog_Namespace::DBSummaryList db_summaries =
      Catalog_Namespace::SysCatalog::instance().getDatabaseListForUser(user);
  set_headers(_return, {"Database", "Owner"});
  for (const auto& db_summary : db_summaries) {
    add_row(_return, {db_summary.dbName, db_summary.dbOwnerName});
  }
}

ShowForeignServersCommand::ShowForeignServersCommand(
    const rapidjson::Value& ddl_payload,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_payload, session_ptr) {
  if (!g_enable_fsi) {
    throw std::runtime_error("Unsupported command: SHOW FOREIGN SERVERS");
  }
  // Verify that members are valid
  CHECK(ddl_payload.HasMember("command"));
  if (ddl_payload.HasMember("filters")) {
    CHECK(ddl_payload["filters"].IsArray());
    int num_filters = 0;
    for (auto const& filter_def : ddl_payload["filters"].GetArray()) {
      CHECK(filter_def.IsObject());
      CHECK(filter_def.HasMember("attribute"));
      CHECK(filter_def["attribute"].IsString());
      CHECK(filter_def.HasMember("value"));
      CHECK(filter_def["value"].IsString());
      CHECK(filter_def.HasMember("operation"));
      CHECK(filter_def["operation"].IsString());
      if (num_filters > 0) {
        CHECK(filter_def.HasMember("chain"));
        CHECK(filter_def["chain"].IsString());
      } else {
        CHECK(!filter_def.HasMember("chain"));
      }
      num_filters++;
    }
  }
}

void ShowForeignServersCommand::execute(TQueryResult& _return) {
  const std::vector<std::string> col_names{
      "server_name", "data_wrapper", "created_at", "options"};

  std::vector<const foreign_storage::ForeignServer*> results;
  const auto& user = session_ptr_->get_currentUser();
  if (ddl_payload_.HasMember("filters")) {
    session_ptr_->getCatalog().getForeignServersForUser(
        &ddl_payload_["filters"], user, results);
  } else {
    session_ptr_->getCatalog().getForeignServersForUser(nullptr, user, results);
  }
  set_headers(_return, col_names);

  _return.row_set.row_desc[2].col_type.type = TDatumType::type::TIMESTAMP;

  for (auto const& server_ptr : results) {
    _return.row_set.columns[0].data.str_col.emplace_back(server_ptr->name);

    _return.row_set.columns[1].data.str_col.emplace_back(server_ptr->data_wrapper_type);

    _return.row_set.columns[2].data.int_col.push_back(server_ptr->creation_time);

    _return.row_set.columns[3].data.str_col.emplace_back(
        server_ptr->getOptionsAsJsonString());

    for (size_t i = 0; i < _return.row_set.columns.size(); i++)
      _return.row_set.columns[i].nulls.emplace_back(false);
  }
}
