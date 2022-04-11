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

// This class should be subclassed to cache additional 'internal' data
//    useful for the implementation, but will avoid being exposed in the header
class DdlCommandData {
 public:
  DdlCommandData(const std::string& ddl_statement) {}
  virtual ~DdlCommandData() {}
  virtual std::string commandStr() = 0;
};

class DdlCommand {
 public:
  DdlCommand(const DdlCommandData& ddl_data,
             std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
      : ddl_data_(ddl_data), session_ptr_(session_ptr) {}

  /**
   * Executes the DDL command corresponding to provided JSON payload.
   *
   * @param _return result of DDL command execution (if applicable)
   */
  virtual ExecutionResult execute() = 0;

 protected:
  const DdlCommandData& ddl_data_;
  std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr_;
  bool isDefaultServer(const std::string& server_name);
};

class ShowTablesCommand : public DdlCommand {
 public:
  ShowTablesCommand(const DdlCommandData& ddl_data,
                    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;
};

class ShowTableDetailsCommand : public DdlCommand {
 public:
  ShowTableDetailsCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;

 private:
  std::vector<std::string> getFilteredTableNames();
};

class ShowDatabasesCommand : public DdlCommand {
 public:
  ShowDatabasesCommand(const DdlCommandData& ddl_data,
                       std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;
};

class ShowDiskCacheUsageCommand : public DdlCommand {
 public:
  ShowDiskCacheUsageCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;

 private:
  std::vector<std::string> getFilteredTableNames();
};

class ShowUserDetailsCommand : public DdlCommand {
 public:
  ShowUserDetailsCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;
};

class ReassignOwnedCommand : public DdlCommand {
 public:
  ReassignOwnedCommand(const DdlCommandData& ddl_data,
                       std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute() override;

 private:
  std::string new_owner_;
  std::set<std::string> old_owners_;
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
   * Returns true if this command is SHOW CREATE TABLE
   */
  bool isShowCreateTable();

  /**
   * Returns true if this command is KILL QUERY
   */
  bool isKillQuery();

  /**
   * Returns true if this command is ALTER SYSTEM CLEAR
   */
  bool isAlterSystemClear();

  /**
   * Returns which kind of caches if to clear
   * ALTER SYSTEM CLEAR
   */

  std::string returnCacheType();

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
  std::string ddl_statement_;                 // incoming ddl_statement
  std::string ddl_command_;                   // extracted from ddl_statement_
  std::unique_ptr<DdlCommandData> ddl_data_;  // container for parse data
  std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr_;
};
