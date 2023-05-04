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

#pragma once

#include <string>

// Note: avoid adding #include(s) that require thrift

#include "Catalog/AlterColumnRecovery.h"
#include "Catalog/ColumnDescriptor.h"
#include "Catalog/SessionInfo.h"
#include "Catalog/TableDescriptor.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "Utils/DdlUtils.h"

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
  virtual ExecutionResult execute(bool read_only_mode) = 0;

 protected:
  const DdlCommandData& ddl_data_;
  std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr_;
};

class CreateForeignServerCommand : public DdlCommand {
 public:
  CreateForeignServerCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class AlterForeignServerCommand : public DdlCommand {
 public:
  AlterForeignServerCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;

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
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class CreateForeignTableCommand : public DdlCommand {
 public:
  CreateForeignTableCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;

 private:
  void setTableDetails(const std::string& table_name,
                       TableDescriptor& td,
                       const std::list<ColumnDescriptor>& columns);
  void setColumnDetails(std::list<ColumnDescriptor>& columns);
};

class DropForeignTableCommand : public DdlCommand {
 public:
  DropForeignTableCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);
  ExecutionResult execute(bool read_only_mode) override;
};

class AlterTableAlterColumnCommand : public DdlCommand {
 public:
  AlterTableAlterColumnCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;

  using TypePairs = alter_column_shared::TypePairs;

 private:
  void alterColumn();

  void populateAndWriteRecoveryInfo(const TableDescriptor* td,
                                    const TypePairs& src_dst_cds);

  void cleanupRecoveryInfo(const TableDescriptor* td);

  void alterColumnTypes(const TableDescriptor* td, const TypePairs& src_dst_cds);

  void collectExpectedCatalogChanges(const TableDescriptor* td,
                                     const TypePairs& src_dst_cds);

  std::list<std::list<ColumnDescriptor>> prepareGeoColumns(const TableDescriptor* td,
                                                           const TypePairs& src_dst_cds);

  std::list<const ColumnDescriptor*> prepareColumns(const TableDescriptor* td,
                                                    const TypePairs& src_dst_cds);

  void alterColumns(const TableDescriptor* td, const TypePairs& src_dst_cds);

  void alterNonGeoColumnData(const TableDescriptor* td,
                             const std::list<const ColumnDescriptor*>& cds);

  void alterGeoColumnData(
      const TableDescriptor* td,
      const std::list<std::pair<const ColumnDescriptor*,
                                std::list<const ColumnDescriptor*>>>& geo_src_dst_cds);

  void clearInMemoryData(const TableDescriptor* td, const TypePairs& src_dst_cds);

  AlterTableAlterColumnCommandRecoveryMgr::RecoveryInfo recovery_info_;
  AlterTableAlterColumnCommandRecoveryMgr recovery_mgr_;
};

class AlterTableCommand : public DdlCommand {
 public:
  AlterTableCommand(const DdlCommandData& ddl_data,
                    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);
  ExecutionResult execute(bool read_only_mode) override;
};

class AlterForeignTableCommand : public DdlCommand {
 public:
  AlterForeignTableCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);
  ExecutionResult execute(bool read_only_mode) override;

 private:
  void alterOptions(const foreign_storage::ForeignTable& foreign_table);
  void renameTable(const foreign_storage::ForeignTable* foreign_table);
  void renameColumn(const foreign_storage::ForeignTable* foreign_table);
};

class ShowForeignServersCommand : public DdlCommand {
 public:
  ShowForeignServersCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowCreateServerCommand : public DdlCommand {
 public:
  ShowCreateServerCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;

 private:
  std::string server_;
};

class ShowTablesCommand : public DdlCommand {
 public:
  ShowTablesCommand(const DdlCommandData& ddl_data,
                    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowTableDetailsCommand : public DdlCommand {
 public:
  ShowTableDetailsCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;

 private:
  std::vector<std::string> getFilteredTableNames();
};

class ShowCreateTableCommand : public DdlCommand {
 public:
  ShowCreateTableCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowDatabasesCommand : public DdlCommand {
 public:
  ShowDatabasesCommand(const DdlCommandData& ddl_data,
                       std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowFunctionsCommand : public DdlCommand {
 public:
  ShowFunctionsCommand(const DdlCommandData& ddl_data,
                       std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowRuntimeFunctionsCommand : public DdlCommand {
 public:
  ShowRuntimeFunctionsCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowTableFunctionsCommand : public DdlCommand {
 public:
  ShowTableFunctionsCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowRuntimeTableFunctionsCommand : public DdlCommand {
 public:
  ShowRuntimeTableFunctionsCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowModelsCommand : public DdlCommand {
 public:
  ShowModelsCommand(const DdlCommandData& ddl_data,
                    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowModelDetailsCommand : public DdlCommand {
 public:
  ShowModelDetailsCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;

 private:
  std::vector<std::string> getFilteredModelNames();
};

class AbstractMLModel;
class MLModelMetadata;

class ShowModelFeatureDetailsCommand : public DdlCommand {
 public:
  ShowModelFeatureDetailsCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;

 private:
  std::vector<TargetMetaInfo> prepareLabelInfos() const;
  std::pair<std::vector<double>, std::vector<std::vector<double>>> extractExtraMetadata(
      std::shared_ptr<AbstractMLModel> model,
      std::vector<TargetMetaInfo>& label_infos) const;

  std::vector<RelLogicalValues::RowValues> prepareLogicalValues(
      const MLModelMetadata& model_metadata,
      const std::vector<std::vector<std::string>>& cat_sub_features,
      std::vector<double>& extra_metadata,
      const std::vector<std::vector<double>>& eigenvectors,
      const std::vector<int64_t>& inverse_permutations) const;
};

class EvaluateModelCommand : public DdlCommand {
 public:
  EvaluateModelCommand(const DdlCommandData& ddl_data,
                       std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowDiskCacheUsageCommand : public DdlCommand {
 public:
  ShowDiskCacheUsageCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;

 private:
  std::vector<std::string> getFilteredTableNames();
};

class ShowUserDetailsCommand : public DdlCommand {
 public:
  ShowUserDetailsCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowRolesCommand : public DdlCommand {
 public:
  ShowRolesCommand(const DdlCommandData& ddl_data,
                   std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class RefreshForeignTablesCommand : public DdlCommand {
 public:
  RefreshForeignTablesCommand(
      const DdlCommandData& ddl_data,
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class CreatePolicyCommand : public DdlCommand {
 public:
  CreatePolicyCommand(const DdlCommandData& ddl_data,
                      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class ShowPoliciesCommand : public DdlCommand {
 public:
  ShowPoliciesCommand(const DdlCommandData& ddl_data,
                      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class DropPolicyCommand : public DdlCommand {
 public:
  DropPolicyCommand(const DdlCommandData& ddl_data,
                    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;
};

class AlterDatabaseCommand : public DdlCommand {
 public:
  AlterDatabaseCommand(const DdlCommandData& ddl_data,
                       std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;

 private:
  void changeOwner();
  void rename();
};

class ReassignOwnedCommand : public DdlCommand {
 public:
  ReassignOwnedCommand(const DdlCommandData& ddl_data,
                       std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  ExecutionResult execute(bool read_only_mode) override;

 private:
  std::string new_owner_;
  std::set<std::string> old_owners_;
  bool all_;
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
  ExecutionResult execute(bool read_only_mode);

  /**
   * Returns true if this command is SHOW USER SESSIONS
   */
  bool isShowUserSessions() const;

  /**
   * Returns true if this command is SHOW QUERIES
   */
  bool isShowQueries() const;

  /**
   * Returns true if this command is KILL QUERY
   */
  bool isKillQuery() const;

  /**
   * Returns true if this command is ALTER SYSTEM CLEAR
   */
  bool isAlterSystemClear() const;

  /**
   * Returns true if this command is ALTER SESSION SET
   */
  bool isAlterSessionSet() const;

  /**
   * Returns which kind of caches to clear if ALTER SYSTEM CLEAR
   */
  std::string returnCacheType() const;

  /**
   * Returns true if this command is ALTER SYSTEM PAUSE|RESUME EXECUTOR QUEUE
   */
  bool isAlterSystemControlExecutorQueue() const;

  /**
   * Returns whether PAUSE or RESUME request
   * has been delivered to ALTER SYSTEM <CONTROL> EXECUTOR qUEUE
   */
  std::string returnQueueAction() const;

  /**
   * Returns target query session if this command is KILL QUERY
   */
  const std::string getTargetQuerySessionToKill() const;

  /**
   * Returns an object indicating where command execution should
   * take place and how results should be aggregated for
   * distributed setups.
   */
  DistributedExecutionDetails getDistributedExecutionDetails() const;

  /**
   * Returns command string, can be useful for logging, conversion
   */
  const std::string commandStr() const;

  /**
   * Returns name and value of a Session parameter
   */
  std::pair<std::string, std::string> getSessionParameter() const;

 private:
  std::string ddl_statement_;                 // incoming ddl_statement
  std::string ddl_command_;                   // extracted from ddl_statement_
  std::unique_ptr<DdlCommandData> ddl_data_;  // container for parse data
  std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr_;
};
