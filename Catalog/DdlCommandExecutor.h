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

#pragma once

#include <string>

#include "rapidjson/document.h"

#include "Catalog/SessionInfo.h"
#include "gen-cpp/mapd_types.h"

class DdlCommand {
 public:
  DdlCommand(const rapidjson::Value& ddl_payload,
             std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
      : ddl_payload(ddl_payload), session_ptr(session_ptr) {}

  /**
   * Executes the DDL command corresponding to provided JSON payload.
   *
   * @param _return result of DDL command execution (if applicable)
   */
  virtual void execute(TQueryResult& _return) = 0;

 protected:
  const rapidjson::Value& ddl_payload;
  std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr;
};

class CreateForeignServerCommand : public DdlCommand {
 public:
  CreateForeignServerCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  void execute(TQueryResult& _return) override;
};

class DropForeignServerCommand : public DdlCommand {
 public:
  DropForeignServerCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  void execute(TQueryResult& _return) override;
};

class DdlCommandExecutor {
 public:
  DdlCommandExecutor(const std::string& ddl_statement,
                     std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  /**
   * Parses given JSON string and routes resulting payload to appropriate DDL command
   * class for execution.
   *
   * @param _return result of DDL command execution (if applicable)
   */
  void execute(TQueryResult& _return);

 private:
  const std::string& ddl_statement;
  std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr;
};
