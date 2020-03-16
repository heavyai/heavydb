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

#include <boost/algorithm/string/predicate.hpp>

#include "Catalog//Catalog.h"
#include "DdlCommandExecutor.h"

DdlCommandExecutor::DdlCommandExecutor(
    const std::string& ddl_statement,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : ddl_statement(ddl_statement), session_ptr(session_ptr) {
  CHECK(!ddl_statement.empty());
}

void DdlCommandExecutor::execute(TQueryResult& _return) {
  rapidjson::Document ddl_query;
  ddl_query.Parse(ddl_statement);
  CHECK(ddl_query.IsObject());
  CHECK(ddl_query.HasMember("payload"));
  CHECK(ddl_query["payload"].IsObject());
  const auto& payload = ddl_query["payload"].GetObject();

  CHECK(payload.HasMember("command"));
  CHECK(payload["command"].IsString());
  const auto& ddl_command = std::string_view(payload["command"].GetString());
  if (ddl_command == "CREATE_SERVER") {
    CreateForeignServerCommand{payload, session_ptr}.execute(_return);
  } else if (ddl_command == "DROP_SERVER") {
    DropForeignServerCommand{payload, session_ptr}.execute(_return);
  } else {
    throw std::runtime_error("Unsupported DDL command");
  }
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
  std::string_view server_name = ddl_payload["serverName"].GetString();
  if (boost::iequals(server_name.substr(0, 7), "omnisci")) {
    throw std::runtime_error{"Server names cannot start with \"omnisci\"."};
  }

  // TODO: add permissions check and ownership
  auto foreign_server = std::make_unique<foreign_storage::ForeignServer>(
      foreign_storage::DataWrapper{ddl_payload["dataWrapper"].GetString()});
  foreign_server->name = server_name;
  foreign_server->populateOptionsMap(ddl_payload["options"]);
  foreign_server->validate();

  session_ptr->getCatalog().createForeignServer(std::move(foreign_server),
                                                ddl_payload["ifNotExists"].GetBool());
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
  std::string_view server_name = ddl_payload["serverName"].GetString();
  if (boost::iequals(server_name.substr(0, 7), "omnisci")) {
    throw std::runtime_error{"OmniSci default servers cannot be dropped."};
  }

  // TODO: add permissions check
  session_ptr->getCatalog().dropForeignServer(ddl_payload["serverName"].GetString(),
                                              ddl_payload["ifExists"].GetBool());
}
