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

// Note: avoid adding #include(s) that require thrift

#include "Catalog/ColumnDescriptor.h"
#include "Catalog/SessionInfo.h"
#include "Catalog/TableDescriptor.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "Utils/DdlUtils.h"
#include "rapidjson/document.h"

class DdlCommand {
 public:
  DdlCommand(const rapidjson::Value& ddl_payload,
             std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
      : ddl_payload_(ddl_payload), session_ptr_(session_ptr) {}

  /**
   * Executes the DDL command corresponding to provided JSON payload.
   *
   * @param _return result of DDL command execution (if applicable)
   */
  virtual ExecutionResult execute() = 0;

 protected:
  const rapidjson::Value& ddl_payload_;
  std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr_;
  bool isDefaultServer(const std::string& server_name);
};

class CreateForeignServerCommand : public DdlCommand {
 public:
  CreateForeignServerCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;
};

class AlterForeignServerCommand : public DdlCommand {
 public:
  AlterForeignServerCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;

 private:
  void changeForeignServerOwner();
  void renameForeignServer();
  void setForeignServerOptions();
  void setForeignServerDataWrapper();
  bool hasAlterServerPrivileges();
};

class DropForeignServerCommand : public DdlCommand {
 public:
  DropForeignServerCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;
};

class JsonColumnSqlType : public ddl_utils::SqlType {
 public:
  JsonColumnSqlType(const rapidjson::Value& data_type)
      : ddl_utils::SqlType(getSqlType(data_type),
                           getParam1(data_type),
                           getParam2(data_type),
                           isArray(data_type),
                           getArraySize(data_type)) {}

 private:
  static SQLTypes getSqlType(const rapidjson::Value& data_type);
  static SQLTypes getSqlType(const std::string& type);
  static int getParam1(const rapidjson::Value& data_type);
  static int getParam2(const rapidjson::Value& data_type);
  static bool isArray(const rapidjson::Value& data_type);
  static int getArraySize(const rapidjson::Value& data_type);
};

class JsonColumnEncoding : public ddl_utils::Encoding {
 public:
  JsonColumnEncoding(const rapidjson::Value& data_type)
      : ddl_utils::Encoding(getEncodingName(data_type), getEncodingParam(data_type)) {}

 private:
  static std::string* getEncodingName(const rapidjson::Value& data_type);
  static int getEncodingParam(const rapidjson::Value& data_type);
};

class CreateForeignTableCommand : public DdlCommand {
 public:
  CreateForeignTableCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;

 private:
  void setTableDetails(const std::string& table_name,
                       TableDescriptor& td,
                       const size_t column_count);
  void setColumnDetails(std::list<ColumnDescriptor>& columns);
};

class DropForeignTableCommand : public DdlCommand {
 public:
  DropForeignTableCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);
  ExecutionResult execute() override;
};

class AlterForeignTableCommand : public DdlCommand {
 public:
  AlterForeignTableCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);
  ExecutionResult execute() override;

 private:
  void alterOptions(const foreign_storage::ForeignTable* foreign_table);
  void renameTable(const foreign_storage::ForeignTable* foreign_table);
  void renameColumn(const foreign_storage::ForeignTable* foreign_table);
};

class ShowForeignServersCommand : public DdlCommand {
 public:
  ShowForeignServersCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;
};

class ShowTablesCommand : public DdlCommand {
 public:
  ShowTablesCommand(const rapidjson::Value& ddl_payload,
                    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;
};

class ShowTableDetailsCommand : public DdlCommand {
 public:
  ShowTableDetailsCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;

 private:
  std::vector<std::string> getFilteredTableNames();
};

class ShowDatabasesCommand : public DdlCommand {
 public:
  ShowDatabasesCommand(const rapidjson::Value& ddl_payload,
                       std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;
};

class ShowDiskCacheUsageCommand : public DdlCommand {
 public:
  ShowDiskCacheUsageCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;

 private:
  std::vector<std::string> getFilteredTableNames();
};

class RefreshForeignTablesCommand : public DdlCommand {
 public:
  RefreshForeignTablesCommand(
      const rapidjson::Value& ddl_payload,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;
};

enum class ExecutionLocation { ALL_NODES, AGGREGATOR_ONLY, LEAVES_ONLY };
enum class AggregationType { NONE, UNION };

struct DistributedExecutionDetails {
  ExecutionLocation execution_location;
  AggregationType aggregation_type;
};

class DdlCommandExecutor {
 public:
  DdlCommandExecutor(const std::string& ddl_statement,
                     std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  /**
   * Parses given JSON string and routes resulting payload to appropriate DDL command
   * class for execution.
   *
   * @param return ExecutionResult of DDL command execution (if applicable)
   */
  ExecutionResult execute();

  /**
   * Returns true if this command is SHOW USER SESSIONS
   */
  bool isShowUserSessions();

  /**
   * Returns true if this command is SHOW QUERIES
   */
  bool isShowQueries();

  /**
   * Returns true if this command is KILL QUERY
   */
  bool isKillQuery();

  /**
   * Returns target query session if this command is KILL QUERY
   */
  const std::string getTargetQuerySessionToKill();

  /**
   * Returns an object indicating where command execution should
   * take place and how results should be aggregated for
   * distributed setups.
   */
  DistributedExecutionDetails getDistributedExecutionDetails();

  /**
   * Returns command string, can be useful for logging, conversion
   */
  const std::string commandStr();

 private:
  rapidjson::Document ddl_query_;
  std::string ddl_command_;  // extracted from ddl_query
  std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr_;
};
