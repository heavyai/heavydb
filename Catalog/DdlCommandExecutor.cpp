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

#include "rapidjson/document.h"

// Note: avoid adding #include(s) that require thrift

#include "Catalog/Catalog.h"
#include "Catalog/SysCatalog.h"
#include "LockMgr/LockMgr.h"
#include "Shared/StringTransform.h"

#include "QueryEngine/Execute.h"  // Executor::getArenaBlockSize()
#include "QueryEngine/ExternalCacheInvalidators.h"
#include "QueryEngine/ResultSetBuilder.h"

extern bool g_enable_fsi;

bool DdlCommand::isDefaultServer(const std::string& server_name) {
  return boost::iequals(server_name.substr(0, 7), "omnisci");
}

namespace {
template <class LockType>
std::tuple<const TableDescriptor*,
           std::unique_ptr<lockmgr::TableSchemaLockContainer<LockType>>>
get_table_descriptor_with_lock(const Catalog_Namespace::Catalog& cat,
                               const std::string& table_name,
                               const bool populate_fragmenter) {
  const TableDescriptor* td{nullptr};
  std::unique_ptr<lockmgr::TableSchemaLockContainer<LockType>> td_with_lock =
      std::make_unique<lockmgr::TableSchemaLockContainer<LockType>>(
          lockmgr::TableSchemaLockContainer<LockType>::acquireTableDescriptor(
              cat, table_name, populate_fragmenter));
  CHECK(td_with_lock);
  td = (*td_with_lock)();
  CHECK(td);
  return std::make_tuple(td, std::move(td_with_lock));
}

struct StorageStats {
  int32_t epoch{0};
  int32_t epoch_floor{0};
  uint64_t metadata_file_count{0};
  uint64_t total_metadata_file_size{0};
  uint64_t total_metadata_page_count{0};
  std::optional<uint64_t> total_free_metadata_page_count{};
  uint64_t data_file_count{0};
  uint64_t total_data_file_size{0};
  uint64_t total_data_page_count{0};
  std::optional<uint64_t> total_free_data_page_count{};

  StorageStats() = default;
  StorageStats(const StorageStats& storage_stats) = default;
  virtual ~StorageStats() = default;
};

struct AggregratedStorageStats : public StorageStats {
  int32_t min_epoch;
  int32_t max_epoch;
  int32_t min_epoch_floor;
  int32_t max_epoch_floor;

  AggregratedStorageStats(const StorageStats& storage_stats)
      : StorageStats(storage_stats)
      , min_epoch(storage_stats.epoch)
      , max_epoch(storage_stats.epoch)
      , min_epoch_floor(storage_stats.epoch_floor)
      , max_epoch_floor(storage_stats.epoch_floor) {}

  void aggregate(const StorageStats& storage_stats) {
    metadata_file_count += storage_stats.metadata_file_count;
    total_metadata_file_size += storage_stats.total_metadata_file_size;
    total_metadata_page_count += storage_stats.total_metadata_page_count;
    if (storage_stats.total_free_metadata_page_count) {
      if (total_free_metadata_page_count) {
        total_free_metadata_page_count.value() +=
            storage_stats.total_free_metadata_page_count.value();
      } else {
        total_free_metadata_page_count = storage_stats.total_free_metadata_page_count;
      }
    }
    data_file_count += storage_stats.data_file_count;
    total_data_file_size += storage_stats.total_data_file_size;
    total_data_page_count += storage_stats.total_data_page_count;
    if (storage_stats.total_free_data_page_count) {
      if (total_free_data_page_count) {
        total_free_data_page_count.value() +=
            storage_stats.total_free_data_page_count.value();
      } else {
        total_free_data_page_count = storage_stats.total_free_data_page_count;
      }
    }
    min_epoch = std::min(min_epoch, storage_stats.epoch);
    max_epoch = std::max(max_epoch, storage_stats.epoch);
    min_epoch_floor = std::min(min_epoch_floor, storage_stats.epoch_floor);
    max_epoch_floor = std::max(max_epoch_floor, storage_stats.epoch_floor);
  }
};

AggregratedStorageStats get_agg_storage_stats(const TableDescriptor* td,
                                              const Catalog_Namespace::Catalog* catalog) {
  UNREACHABLE();
}

std::unique_ptr<RexLiteral> genLiteralStr(std::string val) {
  return std::unique_ptr<RexLiteral>(
      new RexLiteral(val, SQLTypes::kTEXT, SQLTypes::kTEXT, 0, 0, 0, 0));
}

std::unique_ptr<RexLiteral> genLiteralBigInt(int64_t val) {
  return std::unique_ptr<RexLiteral>(
      new RexLiteral(val, SQLTypes::kBIGINT, SQLTypes::kBIGINT, 0, 8, 0, 8));
}

std::unique_ptr<RexLiteral> genLiteralBoolean(bool val) {
  return std::unique_ptr<RexLiteral>(
      // new RexLiteral(val, SQLTypes::kBOOLEAN, SQLTypes::kBOOLEAN, 0, 0, 0, 0));
      new RexLiteral(
          (int64_t)(val ? 1 : 0), SQLTypes::kBIGINT, SQLTypes::kBIGINT, 0, 8, 0, 8));
}

void set_headers_with_type(
    std::vector<TargetMetaInfo>& label_infos,
    const std::vector<std::tuple<std::string, SQLTypes, bool>>& headers) {
  for (const auto& header : headers) {
    auto [_val, _type, _notnull] = header;
    if (_type == kBIGINT || _type == kTEXT || _type == kTIMESTAMP || _type == kBOOLEAN) {
      label_infos.emplace_back(_val, SQLTypeInfo(_type, _notnull));
    } else {
      UNREACHABLE() << "Unsupported type provided for header. SQL type: "
                    << to_string(_type);
    }
  }
}

void add_table_details(std::vector<RelLogicalValues::RowValues>& logical_values,
                       const TableDescriptor* logical_table,
                       const AggregratedStorageStats& agg_storage_stats) {
  logical_values.emplace_back(RelLogicalValues::RowValues{});
  logical_values.back().emplace_back(genLiteralBigInt(logical_table->tableId));
  logical_values.back().emplace_back(genLiteralStr(logical_table->tableName));
  logical_values.back().emplace_back(genLiteralBigInt(logical_table->nColumns));
  logical_values.back().emplace_back(genLiteralBoolean(false));  // sharded
  logical_values.back().emplace_back(genLiteralBigInt(0));       // nShards
  logical_values.back().emplace_back(genLiteralBigInt(logical_table->maxRows));
  logical_values.back().emplace_back(genLiteralBigInt(logical_table->maxFragRows));
  logical_values.back().emplace_back(genLiteralBigInt(logical_table->maxRollbackEpochs));
  logical_values.back().emplace_back(genLiteralBigInt(agg_storage_stats.min_epoch));
  logical_values.back().emplace_back(genLiteralBigInt(agg_storage_stats.max_epoch));
  logical_values.back().emplace_back(genLiteralBigInt(agg_storage_stats.min_epoch_floor));
  logical_values.back().emplace_back(genLiteralBigInt(agg_storage_stats.max_epoch_floor));
  logical_values.back().emplace_back(
      genLiteralBigInt(agg_storage_stats.metadata_file_count));
  logical_values.back().emplace_back(
      genLiteralBigInt(agg_storage_stats.total_metadata_file_size));
  logical_values.back().emplace_back(
      genLiteralBigInt(agg_storage_stats.total_metadata_page_count));

  if (agg_storage_stats.total_free_metadata_page_count) {
    logical_values.back().emplace_back(
        genLiteralBigInt(agg_storage_stats.total_free_metadata_page_count.value()));
  } else {
    logical_values.back().emplace_back(genLiteralBigInt(NULL_BIGINT));
  }

  logical_values.back().emplace_back(genLiteralBigInt(agg_storage_stats.data_file_count));
  logical_values.back().emplace_back(
      genLiteralBigInt(agg_storage_stats.total_data_file_size));
  logical_values.back().emplace_back(
      genLiteralBigInt(agg_storage_stats.total_data_page_count));

  if (agg_storage_stats.total_free_data_page_count) {
    logical_values.back().emplace_back(
        genLiteralBigInt(agg_storage_stats.total_free_data_page_count.value()));
  } else {
    logical_values.back().emplace_back(genLiteralBigInt(NULL_BIGINT));
  }
}

// -----------------------------------------------------------------------
// class DdlCommandDataImpl:
//
// Concrete class to cache parse data
//   Defined & Implemented here to avoid exposing rapidjson in the header file
//   Helper/access fns available to get useful pieces of cache data
// -----------------------------------------------------------------------
class DdlCommandDataImpl : public DdlCommandData {
 public:
  DdlCommandDataImpl(const std::string& ddl_statement);
  ~DdlCommandDataImpl() override;

  // The full query available for futher analysis
  const rapidjson::Value& query() const;

  // payload as extracted from the query
  const rapidjson::Value& payload() const;

  // commandStr extracted from the payload
  std::string commandStr() override;

  rapidjson::Document ddl_query;
};

DdlCommandDataImpl::DdlCommandDataImpl(const std::string& ddl_statement)
    : DdlCommandData(ddl_statement) {
  ddl_query.Parse(ddl_statement);
}

DdlCommandDataImpl::~DdlCommandDataImpl() {}

const rapidjson::Value& DdlCommandDataImpl::query() const {
  return ddl_query;
}

const rapidjson::Value& DdlCommandDataImpl::payload() const {
  CHECK(ddl_query.HasMember("payload"));
  CHECK(ddl_query["payload"].IsObject());
  return ddl_query["payload"];
}

std::string DdlCommandDataImpl::commandStr() {
  if (ddl_query.IsObject() && ddl_query.HasMember("payload") &&
      ddl_query["payload"].IsObject()) {
    auto& payload = ddl_query["payload"];
    if (payload.HasMember("command") && payload["command"].IsString()) {
      return payload["command"].GetString();
    }
  }
  return "";
}

// Helper Fn to get the payload from the abstract base class
const rapidjson::Value& extractPayload(const DdlCommandData& ddl_data) {
  const DdlCommandDataImpl* data = static_cast<const DdlCommandDataImpl*>(&ddl_data);
  return data->payload();
}

}  // namespace

DdlCommandExecutor::DdlCommandExecutor(
    const std::string& ddl_statement,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : session_ptr_(session_ptr) {
  CHECK(!ddl_statement.empty());
  ddl_statement_ = ddl_statement;

  // parse the incoming query,
  //    cache the parsed rapidjson object inside a DdlCommandDataImpl
  //    store the "abstract/base class" reference in ddl_data_
  DdlCommandDataImpl* ddl_query_data = new DdlCommandDataImpl(ddl_statement);
  ddl_data_ = std::unique_ptr<DdlCommandData>(ddl_query_data);

  VLOG(2) << "Parsing JSON DDL from Calcite: " << ddl_statement;
  auto& ddl_query = ddl_query_data->query();
  CHECK(ddl_query.IsObject()) << ddl_statement;
  CHECK(ddl_query.HasMember("payload"));
  CHECK(ddl_query["payload"].IsObject());
  const auto& payload = ddl_query["payload"].GetObject();
  CHECK(payload.HasMember("command"));
  CHECK(payload["command"].IsString());
  ddl_command_ = payload["command"].GetString();
}

ExecutionResult DdlCommandExecutor::execute() {
  ExecutionResult result;
  return result;
}

bool DdlCommandExecutor::isShowUserSessions() {
  return (ddl_command_ == "SHOW_USER_SESSIONS");
}

bool DdlCommandExecutor::isShowQueries() {
  return (ddl_command_ == "SHOW_QUERIES");
}

bool DdlCommandExecutor::isKillQuery() {
  return (ddl_command_ == "KILL_QUERY");
}

bool DdlCommandExecutor::isShowCreateTable() {
  return (ddl_command_ == "SHOW_CREATE_TABLE");
}

bool DdlCommandExecutor::isAlterSystemClear() {
  return (ddl_command_ == "ALTER_SYSTEM_CLEAR");
}

std::string DdlCommandExecutor::returnCacheType() {
  CHECK(ddl_command_ == "ALTER_SYSTEM_CLEAR");
  auto& ddl_payload = extractPayload(*ddl_data_);
  CHECK(ddl_payload.HasMember("cacheType"));
  CHECK(ddl_payload["cacheType"].IsString());
  return ddl_payload["cacheType"].GetString();
}

DistributedExecutionDetails DdlCommandExecutor::getDistributedExecutionDetails() {
  DistributedExecutionDetails execution_details;
  if (ddl_command_ == "CREATE_DATAFRAME" || ddl_command_ == "RENAME_TABLE" ||
      ddl_command_ == "ALTER_TABLE" || ddl_command_ == "CREATE_TABLE" ||
      ddl_command_ == "DROP_TABLE" || ddl_command_ == "TRUNCATE_TABLE" ||
      ddl_command_ == "DUMP_TABLE" || ddl_command_ == "RESTORE_TABLE" ||
      ddl_command_ == "OPTIMIZE_TABLE" || ddl_command_ == "CREATE_VIEW" ||
      ddl_command_ == "DROP_VIEW" || ddl_command_ == "CREATE_DB" ||
      ddl_command_ == "DROP_DB" || ddl_command_ == "RENAME_DB" ||
      ddl_command_ == "CREATE_USER" || ddl_command_ == "DROP_USER" ||
      ddl_command_ == "ALTER_USER" || ddl_command_ == "RENAME_USER" ||
      ddl_command_ == "CREATE_ROLE" || ddl_command_ == "DROP_ROLE" ||
      ddl_command_ == "GRANT_ROLE" || ddl_command_ == "REVOKE_ROLE" ||
      ddl_command_ == "REASSIGN_OWNED") {
    // group user/role/db commands
    execution_details.execution_location = ExecutionLocation::ALL_NODES;
    execution_details.aggregation_type = AggregationType::NONE;
  } else if (ddl_command_ == "GRANT_PRIVILEGE" || ddl_command_ == "REVOKE_PRIVILEGE") {
    auto& ddl_payload = extractPayload(*ddl_data_);
    CHECK(ddl_payload.HasMember("type"));
    const std::string& targetType = ddl_payload["type"].GetString();
    if (targetType == "DASHBOARD") {
      // dashboard commands should run on Aggregator alone
      execution_details.execution_location = ExecutionLocation::AGGREGATOR_ONLY;
      execution_details.aggregation_type = AggregationType::NONE;
    } else {
      execution_details.execution_location = ExecutionLocation::ALL_NODES;
      execution_details.aggregation_type = AggregationType::NONE;
    }

  } else if (ddl_command_ == "SHOW_TABLE_DETAILS") {
    execution_details.execution_location = ExecutionLocation::LEAVES_ONLY;
    execution_details.aggregation_type = AggregationType::UNION;
  } else {
    // Commands that fall here : COPY_TABLE, EXPORT_QUERY, etc.
    execution_details.execution_location = ExecutionLocation::AGGREGATOR_ONLY;
    execution_details.aggregation_type = AggregationType::NONE;
  }
  return execution_details;
}

const std::string DdlCommandExecutor::getTargetQuerySessionToKill() {
  // caller should check whether DDL indicates KillQuery request
  // i.e., use isKillQuery() before calling this function
  auto& ddl_payload = extractPayload(*ddl_data_);
  CHECK(isKillQuery());
  CHECK(ddl_payload.HasMember("querySession"));
  const std::string& query_session = ddl_payload["querySession"].GetString();
  // regex matcher for public_session: start_time{3}-session_id{4} (Example:819-4RDo)
  boost::regex session_id_regex{R"([0-9]{3}-[a-zA-Z0-9]{4})",
                                boost::regex::extended | boost::regex::icase};
  if (!boost::regex_match(query_session, session_id_regex)) {
    throw std::runtime_error(
        "Please provide the correct session ID of the query that you want to interrupt.");
  }
  return query_session;
}

const std::string DdlCommandExecutor::commandStr() {
  return ddl_command_;
}

ShowTablesCommand::ShowTablesCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {}

ExecutionResult ShowTablesCommand::execute() {
  // Get all table names in the same way as OmniSql \t command

  // label_infos -> column labels
  std::vector<std::string> labels{"table_name"};
  std::vector<TargetMetaInfo> label_infos;
  for (const auto& label : labels) {
    label_infos.emplace_back(label, SQLTypeInfo(kTEXT, true));
  }

  // Get all table names
  auto cat_ptr = session_ptr_->get_catalog_ptr();
  auto cur_user = session_ptr_->get_currentUser();
  auto table_names = cat_ptr->getTableNamesForUser(cur_user, GET_PHYSICAL_TABLES);

  // logical_values -> table data
  std::vector<RelLogicalValues::RowValues> logical_values;
  for (auto table_name : table_names) {
    logical_values.emplace_back(RelLogicalValues::RowValues{});
    logical_values.back().emplace_back(genLiteralStr(table_name));
  }

  // Create ResultSet
  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

ShowTableDetailsCommand::ShowTableDetailsCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  auto& ddl_payload = extractPayload(ddl_data_);
  if (ddl_payload.HasMember("tableNames")) {
    CHECK(ddl_payload["tableNames"].IsArray());
    for (const auto& table_name : ddl_payload["tableNames"].GetArray()) {
      CHECK(table_name.IsString());
    }
  }
}

ExecutionResult ShowTableDetailsCommand::execute() {
  const auto catalog = session_ptr_->get_catalog_ptr();
  std::vector<std::string> filtered_table_names = getFilteredTableNames();

  std::vector<TargetMetaInfo> label_infos;
  set_headers_with_type(label_infos,
                        {// { label, type, notNull }
                         {"table_id", kBIGINT, true},
                         {"table_name", kTEXT, true},
                         {"column_count", kBIGINT, true},
                         {"is_sharded_table", kBOOLEAN, true},
                         {"shard_count", kBIGINT, true},
                         {"max_rows", kBIGINT, true},
                         {"fragment_size", kBIGINT, true},
                         {"max_rollback_epochs", kBIGINT, true},
                         {"min_epoch", kBIGINT, true},
                         {"max_epoch", kBIGINT, true},
                         {"min_epoch_floor", kBIGINT, true},
                         {"max_epoch_floor", kBIGINT, true},
                         {"metadata_file_count", kBIGINT, true},
                         {"total_metadata_file_size", kBIGINT, true},
                         {"total_metadata_page_count", kBIGINT, true},
                         {"total_free_metadata_page_count", kBIGINT, false},
                         {"data_file_count", kBIGINT, true},
                         {"total_data_file_size", kBIGINT, true},
                         {"total_data_page_count", kBIGINT, true},
                         {"total_free_data_page_count", kBIGINT, false}});

  std::vector<RelLogicalValues::RowValues> logical_values;
  for (const auto& table_name : filtered_table_names) {
    auto [td, td_with_lock] =
        get_table_descriptor_with_lock<lockmgr::ReadLock>(*catalog, table_name, false);
    auto agg_storage_stats = get_agg_storage_stats(td, catalog.get());
    add_table_details(logical_values, td, agg_storage_stats);
  }

  // Create ResultSet
  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

std::vector<std::string> ShowTableDetailsCommand::getFilteredTableNames() {
  const auto catalog = session_ptr_->get_catalog_ptr();
  auto& ddl_payload = extractPayload(ddl_data_);
  auto all_table_names =
      catalog->getTableNamesForUser(session_ptr_->get_currentUser(), GET_PHYSICAL_TABLES);
  std::transform(all_table_names.begin(),
                 all_table_names.end(),
                 all_table_names.begin(),
                 [](const std::string& s) { return to_upper(s); });
  std::vector<std::string> filtered_table_names;
  if (ddl_payload.HasMember("tableNames")) {
    std::set<std::string> all_table_names_set(all_table_names.begin(),
                                              all_table_names.end());
    for (const auto& table_name_json : ddl_payload["tableNames"].GetArray()) {
      std::string table_name = table_name_json.GetString();
      if (all_table_names_set.find(to_upper(table_name)) == all_table_names_set.end()) {
        throw std::runtime_error{"Unable to show table details for table: " + table_name +
                                 ". Table does not exist."};
      }
      auto [td, td_with_lock] =
          get_table_descriptor_with_lock<lockmgr::ReadLock>(*catalog, table_name, false);
      if (td->isTemporaryTable()) {
        throw std::runtime_error{
            "SHOW TABLE DETAILS is not supported for temporary tables. Table name: " +
            table_name + "."};
      }
      filtered_table_names.emplace_back(table_name);
    }
  } else {
    for (const auto& table_name : all_table_names) {
      auto [td, td_with_lock] =
          get_table_descriptor_with_lock<lockmgr::ReadLock>(*catalog, table_name, false);
      if (td->isTemporaryTable()) {
        continue;
      }
      filtered_table_names.emplace_back(table_name);
    }
  }
  return filtered_table_names;
}

ShowDatabasesCommand::ShowDatabasesCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {}

ExecutionResult ShowDatabasesCommand::execute() {
  // label_infos -> column labels
  std::vector<std::string> labels{"Database", "Owner"};
  std::vector<TargetMetaInfo> label_infos;
  for (const auto& label : labels) {
    label_infos.emplace_back(label, SQLTypeInfo(kTEXT, true));
  }

  // Get all table names
  auto cur_user = session_ptr_->get_currentUser();
  const Catalog_Namespace::DBSummaryList db_summaries =
      Catalog_Namespace::SysCatalog::instance().getDatabaseListForUser(cur_user);

  // logical_values -> table data
  std::vector<RelLogicalValues::RowValues> logical_values;
  for (const auto& db_summary : db_summaries) {
    logical_values.emplace_back(RelLogicalValues::RowValues{});
    logical_values.back().emplace_back(genLiteralStr(db_summary.dbName));
    logical_values.back().emplace_back(genLiteralStr(db_summary.dbOwnerName));
  }

  // Create ResultSet
  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

ShowDiskCacheUsageCommand::ShowDiskCacheUsageCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  auto& ddl_payload = extractPayload(ddl_data_);
  if (ddl_payload.HasMember("tableNames")) {
    CHECK(ddl_payload["tableNames"].IsArray());
    for (auto const& tablename_def : ddl_payload["tableNames"].GetArray()) {
      CHECK(tablename_def.IsString());
    }
  }
}

std::vector<std::string> ShowDiskCacheUsageCommand::getFilteredTableNames() {
  auto table_names = session_ptr_->get_catalog_ptr()->getTableNamesForUser(
      session_ptr_->get_currentUser(), GET_PHYSICAL_TABLES);

  auto& ddl_payload = extractPayload(ddl_data_);
  if (ddl_payload.HasMember("tableNames")) {
    std::vector<std::string> filtered_names;
    for (const auto& tablename_def : ddl_payload["tableNames"].GetArray()) {
      std::string filter_name = tablename_def.GetString();
      if (std::find(table_names.begin(), table_names.end(), filter_name) !=
          table_names.end()) {
        filtered_names.emplace_back(filter_name);
      } else {
        throw std::runtime_error("Can not show disk cache usage for table: " +
                                 filter_name + ". Table does not exist.");
      }
    }
    return filtered_names;
  } else {
    return table_names;
  }
}

ExecutionResult ShowDiskCacheUsageCommand::execute() {
  UNREACHABLE();
}

ShowUserDetailsCommand::ShowUserDetailsCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  auto& ddl_payload = extractPayload(ddl_data);
  if (ddl_payload.HasMember("userNames")) {
    CHECK(ddl_payload["userNames"].IsArray());
    for (const auto& user_name : ddl_payload["userNames"].GetArray()) {
      CHECK(user_name.IsString());
    }
  }
}

ExecutionResult ShowUserDetailsCommand::execute() {
  auto& ddl_payload = extractPayload(ddl_data_);
  auto& sys_cat = Catalog_Namespace::SysCatalog::instance();

  // label_infos -> column labels
  std::vector<std::string> labels{"NAME", "ID", "IS_SUPER", "DEFAULT_DB", "CAN_LOGIN"};
  std::vector<TargetMetaInfo> label_infos;
  label_infos.emplace_back(labels[0], SQLTypeInfo(kTEXT, true));
  label_infos.emplace_back(labels[1], SQLTypeInfo(kBIGINT, true));
  label_infos.emplace_back(labels[2], SQLTypeInfo(kBOOLEAN, true));
  label_infos.emplace_back(labels[3], SQLTypeInfo(kTEXT, true));
  label_infos.emplace_back(labels[4], SQLTypeInfo(kBOOLEAN, true));
  std::vector<RelLogicalValues::RowValues> logical_values;

  Catalog_Namespace::UserMetadata self = session_ptr_->get_currentUser();
  Catalog_Namespace::DBSummaryList dbsums = sys_cat.getDatabaseListForUser(self);
  std::unordered_set<std::string> visible_databases;
  if (!self.isSuper) {
    for (const auto& dbsum : dbsums) {
      visible_databases.insert(dbsum.dbName);
    }
  }

  std::list<Catalog_Namespace::UserMetadata> user_list;
  if (ddl_payload.HasMember("userNames")) {
    for (const auto& user_name_json : ddl_payload["userNames"].GetArray()) {
      std::string user_name = user_name_json.GetString();
      Catalog_Namespace::UserMetadata user;
      if (!sys_cat.getMetadataForUser(user_name, user)) {
        throw std::runtime_error("User with username \"" + user_name +
                                 "\" does not exist. ");
      }
      user_list.emplace_back(std::move(user));
    }
  } else {
    user_list = sys_cat.getAllUserMetadata();
  }

  for (const auto& user : user_list) {
    // database
    std::string dbname;
    Catalog_Namespace::DBMetadata db;
    if (sys_cat.getMetadataForDBById(user.defaultDbId, db)) {
      if (self.isSuper.load() || visible_databases.count(db.dbName)) {
        dbname = db.dbName;
      }
    }
    if (self.isSuper.load()) {
      dbname += "(" + std::to_string(user.defaultDbId) + ")";
    }

    // logical_values -> table data
    logical_values.emplace_back(RelLogicalValues::RowValues{});
    logical_values.back().emplace_back(genLiteralStr(user.userName));
    logical_values.back().emplace_back(genLiteralBigInt(user.userId));
    logical_values.back().emplace_back(genLiteralBoolean(user.isSuper.load()));
    logical_values.back().emplace_back(genLiteralStr(dbname));
    logical_values.back().emplace_back(genLiteralBoolean(user.can_login));
  }

  // Create ResultSet
  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

ReassignOwnedCommand::ReassignOwnedCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  auto& ddl_payload = extractPayload(ddl_data_);
  CHECK(ddl_payload.HasMember("oldOwners"));
  CHECK(ddl_payload["oldOwners"].IsArray());
  for (const auto& old_owner : ddl_payload["oldOwners"].GetArray()) {
    CHECK(old_owner.IsString());
    old_owners_.emplace(old_owner.GetString());
  }
  CHECK(ddl_payload.HasMember("newOwner"));
  CHECK(ddl_payload["newOwner"].IsString());
  new_owner_ = ddl_payload["newOwner"].GetString();
}

ExecutionResult ReassignOwnedCommand::execute() {
  if (!session_ptr_->get_currentUser().isSuper) {
    throw std::runtime_error{
        "Only super users can reassign ownership of database objects."};
  }
  const auto catalog = session_ptr_->get_catalog_ptr();
  catalog->reassignOwners(old_owners_, new_owner_);
  return ExecutionResult();
}
