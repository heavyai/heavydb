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

#include "DdlCommandExecutor.h"

#include <algorithm>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "rapidjson/document.h"

// Note: avoid adding #include(s) that require thrift

#include "Catalog/Catalog.h"
#include "Catalog/SysCatalog.h"
#include "DataMgr/ForeignStorage/ForeignTableRefresh.h"
#include "LockMgr/LockMgr.h"
#include "Parser/ParserNode.h"
#include "Shared/StringTransform.h"
#include "Shared/SysDefinitions.h"

#include "Fragmenter/InsertOrderFragmenter.h"
#include "QueryEngine/Execute.h"  // Executor::getArenaBlockSize()
#include "QueryEngine/ExternalCacheInvalidators.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryEngine/ResultSetBuilder.h"

extern bool g_enable_fsi;

namespace {

void validate_alter_type_allowed(const std::string& colname,
                                 const SQLTypeInfo& src,
                                 const SQLTypeInfo& dst) {
  if (!src.is_string()) {
    throw std::runtime_error("Altering column " + colname +
                             " type not allowed. Column type must be TEXT.");
  }
}

void validate_alter_type_metadata(const Catalog_Namespace::Catalog& catalog,
                                  const TableDescriptor* td,
                                  const ColumnDescriptor& cd) {
  ChunkMetadataVector column_metadata;
  catalog.getDataMgr().getChunkMetadataVecForKeyPrefix(
      column_metadata, {catalog.getDatabaseId(), td->tableId, cd.columnId});

  const bool is_not_null = cd.columnType.get_notnull();
  // check for non nulls
  for (const auto& [key, metadata] : column_metadata) {
    if (is_not_null && metadata->chunkStats.has_nulls) {
      throw std::runtime_error("Alter column type: Column " + cd.columnName +
                               ": NULL value not allowed in NOT NULL column");
    }
  }

  if (td->nShards > 0) {
    throw std::runtime_error("Alter column type: Column " + cd.columnName +
                             ": altering a sharded table is unsupported");
  }
  // further checks on metadata can be done to prevent late exceptions
}

std::list<std::pair<const ColumnDescriptor*, std::list<const ColumnDescriptor*>>>
get_alter_column_geo_pairs_from_src_dst_pairs_phys_cds(
    const AlterTableAlterColumnCommand::AlterColumnTypePairs& src_dst_cds,
    const std::list<std::list<ColumnDescriptor>>& phys_cds) {
  std::list<std::pair<const ColumnDescriptor*, std::list<const ColumnDescriptor*>>>
      geo_src_dst_column_pairs;

  auto phys_cds_it = phys_cds.begin();
  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (dst_cd->columnType.is_geometry()) {
      geo_src_dst_column_pairs.emplace_back();
      auto& pair = geo_src_dst_column_pairs.back();
      pair.first = src_cd;

      std::list<const ColumnDescriptor*> geo_dst_cds;
      CHECK(phys_cds_it != phys_cds.end());
      auto& phy_geo_columns = *phys_cds_it;
      geo_dst_cds.push_back(dst_cd);
      for (const auto& cd : phy_geo_columns) {
        geo_dst_cds.push_back(&cd);
      }
      pair.second = geo_dst_cds;

      phys_cds_it++;
    }
  }

  return geo_src_dst_column_pairs;
}

AlterTableAlterColumnCommand::AlterColumnTypePairs
get_alter_column_pairs_from_src_dst_cds(std::list<ColumnDescriptor>& src_cds,
                                        std::list<ColumnDescriptor>& dst_cds) {
  CHECK_EQ(src_cds.size(), dst_cds.size());
  AlterTableAlterColumnCommand::AlterColumnTypePairs src_dst_column_pairs;
  auto src_cd_it = src_cds.begin();
  auto dst_cd_it = dst_cds.begin();
  for (; src_cd_it != src_cds.end(); ++src_cd_it, ++dst_cd_it) {
    src_dst_column_pairs.emplace_back(&(*src_cd_it), &(*dst_cd_it));
  }
  return src_dst_column_pairs;
}

std::pair<std::list<ColumnDescriptor>, std::list<ColumnDescriptor>>
get_alter_column_src_dst_cds(const std::list<Parser::ColumnDef>& columns,
                             Catalog_Namespace::Catalog& catalog,
                             const TableDescriptor* td) {
  std::list<ColumnDescriptor> src_cds;
  std::list<ColumnDescriptor> dst_cds;
  for (const auto& coldef : columns) {
    dst_cds.emplace_back();
    ColumnDescriptor& dst_cd = dst_cds.back();
    set_column_descriptor(dst_cd, &coldef);

    // update kENCODING_DICT column descriptors to reflect correct sizing based on comp
    // param
    if (dst_cd.columnType.is_dict_encoded_string()) {
      switch (dst_cd.columnType.get_comp_param()) {
        case 8:
          dst_cd.columnType.set_size(1);
          break;
        case 16:
          dst_cd.columnType.set_size(2);
          break;
        case 32:
          dst_cd.columnType.set_size(4);
          break;
        default:
          UNREACHABLE();
      }
    }

    auto catalog_cd = catalog.getMetadataForColumn(td->tableId, dst_cd.columnName);
    CHECK(catalog_cd);

    validate_alter_type_allowed(
        dst_cd.columnName, catalog_cd->columnType, dst_cd.columnType);

    // Set remaining values in column descriptor that must be obtained from catalog
    dst_cd.columnId = catalog_cd->columnId;
    dst_cd.tableId = catalog_cd->tableId;
    dst_cd.sourceName = catalog_cd->sourceName;
    dst_cd.chunks = catalog_cd->chunks;
    dst_cd.db_id = catalog_cd->db_id;

    // This branch handles the special case where a string dictionary column
    // type is not changed, but altering is required (for example default
    // changes)
    if (catalog_cd->columnType.is_dict_encoded_type() &&
        dst_cd.columnType.is_dict_encoded_type() &&
        ddl_utils::alter_column_utils::compare_column_descriptors(catalog_cd, &dst_cd)
            .sql_types_match) {
      dst_cd.columnType.set_comp_param(catalog_cd->columnType.get_comp_param());
      dst_cd.columnType.setStringDictKey(catalog_cd->columnType.getStringDictKey());
    }

    if (ddl_utils::alter_column_utils::compare_column_descriptors(catalog_cd, &dst_cd)
            .exact_match) {
      throw std::runtime_error("Altering column " + dst_cd.columnName +
                               " results in no change to column, please review command.");
    }

    validate_alter_type_metadata(catalog, td, dst_cd);

    // A copy of the catalog column descriptor is stored for the source because
    // the catalog may delete its version of the source descriptor in progress
    // of the alter column command
    src_cds.emplace_back();
    src_cds.back() = *catalog_cd;
  }
  return {src_cds, dst_cds};
}

template <class LockType>
std::tuple<const TableDescriptor*,
           std::unique_ptr<lockmgr::TableSchemaLockContainer<LockType>>>
get_table_descriptor_with_lock(Catalog_Namespace::Catalog& cat,
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

// There are cases where we may query a catalog for a list of table names and then perform
// some action on those tables without acquiring a lock, which means the tables could have
// been dropped in between the query and the action.  In such cases, sometimes we want to
// skip that action if the table no longer exists without throwing an error.
template <class Func>
void exec_for_tables_which_exist(const std::vector<std::string>& table_names,
                                 Catalog_Namespace::Catalog* cat_ptr,
                                 Func func) {
  for (const auto& table_name : table_names) {
    try {
      auto [td, td_with_lock] =
          get_table_descriptor_with_lock<lockmgr::ReadLock>(*cat_ptr, table_name, false);
      func(td, table_name);
    } catch (const Catalog_Namespace::TableNotFoundException& e) {
      continue;
    }
  }
}

struct AggregratedStorageStats : public File_Namespace::StorageStats {
  int32_t min_epoch;
  int32_t max_epoch;
  int32_t min_epoch_floor;
  int32_t max_epoch_floor;

  AggregratedStorageStats(const File_Namespace::StorageStats& storage_stats)
      : File_Namespace::StorageStats(storage_stats)
      , min_epoch(storage_stats.epoch)
      , max_epoch(storage_stats.epoch)
      , min_epoch_floor(storage_stats.epoch_floor)
      , max_epoch_floor(storage_stats.epoch_floor) {}

  void aggregate(const File_Namespace::StorageStats& storage_stats) {
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
  const auto global_file_mgr = catalog->getDataMgr().getGlobalFileMgr();
  std::optional<AggregratedStorageStats> agg_storage_stats;
  if (td->nShards > 0) {
    const auto physical_tables = catalog->getPhysicalTablesDescriptors(td, false);
    CHECK_EQ(static_cast<size_t>(td->nShards), physical_tables.size());

    for (const auto physical_table : physical_tables) {
      auto storage_stats = global_file_mgr->getStorageStats(catalog->getDatabaseId(),
                                                            physical_table->tableId);
      if (agg_storage_stats) {
        agg_storage_stats.value().aggregate(storage_stats);
      } else {
        agg_storage_stats = storage_stats;
      }
    }
  } else {
    agg_storage_stats =
        global_file_mgr->getStorageStats(catalog->getDatabaseId(), td->tableId);
  }
  CHECK(agg_storage_stats.has_value());
  return agg_storage_stats.value();
}

std::unique_ptr<RexLiteral> genLiteralStr(std::string val) {
  return std::unique_ptr<RexLiteral>(
      new RexLiteral(val, SQLTypes::kTEXT, SQLTypes::kTEXT, 0, 0, 0, 0));
}

std::unique_ptr<RexLiteral> genLiteralTimestamp(time_t val) {
  return std::unique_ptr<RexLiteral>(new RexLiteral(
      (int64_t)val, SQLTypes::kTIMESTAMP, SQLTypes::kTIMESTAMP, 0, 8, 0, 8));
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
  bool is_sharded_table = (logical_table->nShards > 0);
  logical_values.emplace_back(RelLogicalValues::RowValues{});
  logical_values.back().emplace_back(genLiteralBigInt(logical_table->tableId));
  logical_values.back().emplace_back(genLiteralStr(logical_table->tableName));
  logical_values.back().emplace_back(genLiteralBigInt(logical_table->nColumns));
  logical_values.back().emplace_back(genLiteralBoolean(is_sharded_table));
  logical_values.back().emplace_back(genLiteralBigInt(logical_table->nShards));
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
// class: JsonColumnSqlType
//   Defined & Implemented here to avoid exposing rapidjson in the header file
// -----------------------------------------------------------------------

/// declare this class scoped local to avoid exposing rapidjson in the header file
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

const rapidjson::Value* extractFilters(const rapidjson::Value& payload) {
  const rapidjson::Value* filters = nullptr;
  if (payload.HasMember("filters") && payload["filters"].IsArray()) {
    filters = &payload["filters"];
  }
  return filters;
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

ExecutionResult DdlCommandExecutor::execute(bool read_only_mode) {
  ExecutionResult result;

  // the following commands use parser node locking to ensure safe concurrent access
  if (ddl_command_ == "CREATE_TABLE") {
    auto create_table_stmt = Parser::CreateTableStmt(extractPayload(*ddl_data_));
    create_table_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "CREATE_VIEW") {
    auto create_view_stmt = Parser::CreateViewStmt(extractPayload(*ddl_data_));
    create_view_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "DROP_TABLE") {
    auto drop_table_stmt = Parser::DropTableStmt(extractPayload(*ddl_data_));
    drop_table_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "DROP_VIEW") {
    auto drop_view_stmt = Parser::DropViewStmt(extractPayload(*ddl_data_));
    drop_view_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "RENAME_TABLE") {
    auto rename_table_stmt = Parser::RenameTableStmt(extractPayload(*ddl_data_));
    rename_table_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "ALTER_TABLE") {
    // ALTER TABLE uses the parser node locking partially as well as the global locking
    // scheme for some cases
    return AlterTableCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "TRUNCATE_TABLE") {
    auto truncate_table_stmt = Parser::TruncateTableStmt(extractPayload(*ddl_data_));
    truncate_table_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "DUMP_TABLE") {
    auto dump_table_stmt = Parser::DumpTableStmt(extractPayload(*ddl_data_));
    dump_table_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "RESTORE_TABLE") {
    auto restore_table_stmt = Parser::RestoreTableStmt(extractPayload(*ddl_data_));
    restore_table_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "OPTIMIZE_TABLE") {
    auto optimize_table_stmt = Parser::OptimizeTableStmt(extractPayload(*ddl_data_));
    optimize_table_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "COPY_TABLE") {
    auto copy_table_stmt = Parser::CopyTableStmt(extractPayload(*ddl_data_));
    copy_table_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "EXPORT_QUERY") {
    auto export_query_stmt = Parser::ExportQueryStmt(extractPayload(*ddl_data_));
    export_query_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "CREATE_DB") {
    auto create_db_stmt = Parser::CreateDBStmt(extractPayload(*ddl_data_));
    create_db_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "DROP_DB") {
    auto drop_db_stmt = Parser::DropDBStmt(extractPayload(*ddl_data_));
    drop_db_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "CREATE_USER") {
    auto create_user_stmt = Parser::CreateUserStmt(extractPayload(*ddl_data_));
    create_user_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "DROP_USER") {
    auto drop_user_stmt = Parser::DropUserStmt(extractPayload(*ddl_data_));
    drop_user_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "ALTER_USER") {
    auto alter_user_stmt = Parser::AlterUserStmt(extractPayload(*ddl_data_));
    alter_user_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "RENAME_USER") {
    auto rename_user_stmt = Parser::RenameUserStmt(extractPayload(*ddl_data_));
    rename_user_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "CREATE_ROLE") {
    auto create_role_stmt = Parser::CreateRoleStmt(extractPayload(*ddl_data_));
    create_role_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "DROP_ROLE") {
    auto drop_role_stmt = Parser::DropRoleStmt(extractPayload(*ddl_data_));
    drop_role_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "GRANT_ROLE") {
    auto grant_role_stmt = Parser::GrantRoleStmt(extractPayload(*ddl_data_));
    grant_role_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "REVOKE_ROLE") {
    auto revoke_role_stmt = Parser::RevokeRoleStmt(extractPayload(*ddl_data_));
    revoke_role_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "GRANT_PRIVILEGE") {
    auto grant_privilege_stmt = Parser::GrantPrivilegesStmt(extractPayload(*ddl_data_));
    grant_privilege_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "REVOKE_PRIVILEGE") {
    auto revoke_privileges_stmt =
        Parser::RevokePrivilegesStmt(extractPayload(*ddl_data_));
    revoke_privileges_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "CREATE_DATAFRAME") {
    auto create_dataframe_stmt = Parser::CreateDataframeStmt(extractPayload(*ddl_data_));
    create_dataframe_stmt.execute(*session_ptr_, read_only_mode);
    return result;
  } else if (ddl_command_ == "VALIDATE_SYSTEM") {
    // VALIDATE should have been excuted in outer context before it reaches here
    UNREACHABLE();
  } else if (ddl_command_ == "REFRESH_FOREIGN_TABLES") {
    result =
        RefreshForeignTablesCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
    return result;
  } else if (ddl_command_ == "CREATE_SERVER") {
    result = CreateForeignServerCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "DROP_SERVER") {
    result = DropForeignServerCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "CREATE_FOREIGN_TABLE") {
    result = CreateForeignTableCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "DROP_FOREIGN_TABLE") {
    result = DropForeignTableCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_TABLES") {
    result = ShowTablesCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_TABLE_DETAILS") {
    result = ShowTableDetailsCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_CREATE_TABLE") {
    result = ShowCreateTableCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_DATABASES") {
    result = ShowDatabasesCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_SERVERS") {
    result = ShowForeignServersCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_CREATE_SERVER") {
    result = ShowCreateServerCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_FUNCTIONS") {
    result = ShowFunctionsCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_RUNTIME_FUNCTIONS") {
    result =
        ShowRuntimeFunctionsCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_TABLE_FUNCTIONS") {
    result = ShowTableFunctionsCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_RUNTIME_TABLE_FUNCTIONS") {
    result = ShowRuntimeTableFunctionsCommand{*ddl_data_, session_ptr_}.execute(
        read_only_mode);
  } else if (ddl_command_ == "ALTER_SERVER") {
    result = AlterForeignServerCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "ALTER_DATABASE") {
    result = AlterDatabaseCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "ALTER_FOREIGN_TABLE") {
    result = AlterForeignTableCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_DISK_CACHE_USAGE") {
    result = ShowDiskCacheUsageCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_USER_DETAILS") {
    result = ShowUserDetailsCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "SHOW_ROLES") {
    result = ShowRolesCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (ddl_command_ == "REASSIGN_OWNED") {
    result = ReassignOwnedCommand{*ddl_data_, session_ptr_}.execute(read_only_mode);
  } else {
    throw std::runtime_error("Unsupported DDL command");
  }

  return result;
}

bool DdlCommandExecutor::isShowUserSessions() const {
  return (ddl_command_ == "SHOW_USER_SESSIONS");
}

bool DdlCommandExecutor::isShowQueries() const {
  return (ddl_command_ == "SHOW_QUERIES");
}

bool DdlCommandExecutor::isKillQuery() const {
  return (ddl_command_ == "KILL_QUERY");
}

bool DdlCommandExecutor::isAlterSystemClear() const {
  return (ddl_command_ == "ALTER_SYSTEM_CLEAR");
}

bool DdlCommandExecutor::isAlterSessionSet() const {
  return (ddl_command_ == "ALTER_SESSION_SET");
}

std::pair<std::string, std::string> DdlCommandExecutor::getSessionParameter() const {
  enum SetParameterType { String_t, Numeric_t };
  static const std::unordered_map<std::string, SetParameterType>
      session_set_parameters_map = {{"EXECUTOR_DEVICE", SetParameterType::String_t},
                                    {"CURRENT_DATABASE", SetParameterType::String_t}};

  auto& ddl_payload = extractPayload(*ddl_data_);
  CHECK(ddl_payload.HasMember("sessionParameter"));
  CHECK(ddl_payload["sessionParameter"].IsString());
  CHECK(ddl_payload.HasMember("parameterValue"));
  CHECK(ddl_payload["parameterValue"].IsString());
  std::string parameter_name = to_upper(ddl_payload["sessionParameter"].GetString());
  std::string parameter_value = ddl_payload["parameterValue"].GetString();

  const auto param_it = session_set_parameters_map.find(parameter_name);
  if (param_it == session_set_parameters_map.end()) {
    throw std::runtime_error(parameter_name + " is not a settable session parameter.");
  }
  if (param_it->second == SetParameterType::Numeric_t) {
    if (!std::regex_match(parameter_value, std::regex("[(-|+)|][0-9]+"))) {
      throw std::runtime_error("The value of session parameter " + param_it->first +
                               " should be a numeric.");
    }
  }
  return {parameter_name, parameter_value};
}

std::string DdlCommandExecutor::returnCacheType() const {
  CHECK(ddl_command_ == "ALTER_SYSTEM_CLEAR");
  auto& ddl_payload = extractPayload(*ddl_data_);
  CHECK(ddl_payload.HasMember("cacheType"));
  CHECK(ddl_payload["cacheType"].IsString());
  return ddl_payload["cacheType"].GetString();
}

bool DdlCommandExecutor::isAlterSystemControlExecutorQueue() const {
  return (ddl_command_ == "ALTER_SYSTEM_CONTROL_EXECUTOR_QUEUE");
}

std::string DdlCommandExecutor::returnQueueAction() const {
  CHECK(ddl_command_ == "ALTER_SYSTEM_CONTROL_EXECUTOR_QUEUE");
  auto& ddl_payload = extractPayload(*ddl_data_);
  CHECK(ddl_payload.HasMember("queueAction"));
  CHECK(ddl_payload["queueAction"].IsString());
  return ddl_payload["queueAction"].GetString();
}

DistributedExecutionDetails DdlCommandExecutor::getDistributedExecutionDetails() const {
  DistributedExecutionDetails execution_details;
  if (ddl_command_ == "CREATE_DATAFRAME" || ddl_command_ == "RENAME_TABLE" ||
      ddl_command_ == "ALTER_TABLE" || ddl_command_ == "CREATE_TABLE" ||
      ddl_command_ == "DROP_TABLE" || ddl_command_ == "TRUNCATE_TABLE" ||
      ddl_command_ == "DUMP_TABLE" || ddl_command_ == "RESTORE_TABLE" ||
      ddl_command_ == "OPTIMIZE_TABLE" || ddl_command_ == "CREATE_VIEW" ||
      ddl_command_ == "DROP_VIEW" || ddl_command_ == "CREATE_DB" ||
      ddl_command_ == "DROP_DB" || ddl_command_ == "ALTER_DATABASE" ||
      ddl_command_ == "CREATE_USER" || ddl_command_ == "DROP_USER" ||
      ddl_command_ == "ALTER_USER" || ddl_command_ == "RENAME_USER" ||
      ddl_command_ == "CREATE_ROLE" || ddl_command_ == "DROP_ROLE" ||
      ddl_command_ == "GRANT_ROLE" || ddl_command_ == "REVOKE_ROLE" ||
      ddl_command_ == "REASSIGN_OWNED" || ddl_command_ == "CREATE_POLICY" ||
      ddl_command_ == "DROP_POLICY" || ddl_command_ == "CREATE_SERVER" ||
      ddl_command_ == "DROP_SERVER" || ddl_command_ == "CREATE_FOREIGN_TABLE" ||
      ddl_command_ == "DROP_FOREIGN_TABLE" || ddl_command_ == "CREATE_USER_MAPPING" ||
      ddl_command_ == "DROP_USER_MAPPING" || ddl_command_ == "ALTER_FOREIGN_TABLE" ||
      ddl_command_ == "ALTER_SERVER" || ddl_command_ == "REFRESH_FOREIGN_TABLES" ||
      ddl_command_ == "ALTER_SYSTEM_CLEAR") {
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

  } else if (ddl_command_ == "SHOW_TABLE_DETAILS" ||
             ddl_command_ == "SHOW_DISK_CACHE_USAGE") {
    execution_details.execution_location = ExecutionLocation::LEAVES_ONLY;
    execution_details.aggregation_type = AggregationType::UNION;
  } else {
    // Commands that fall here : COPY_TABLE, EXPORT_QUERY, etc.
    execution_details.execution_location = ExecutionLocation::AGGREGATOR_ONLY;
    execution_details.aggregation_type = AggregationType::NONE;
  }
  return execution_details;
}

const std::string DdlCommandExecutor::getTargetQuerySessionToKill() const {
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

const std::string DdlCommandExecutor::commandStr() const {
  return ddl_command_;
}

namespace {
const std::array<std::string, 3> kReservedServerPrefixes{"default", "system", "internal"};

bool is_default_server(const std::string& server_name) {
  return std::any_of(kReservedServerPrefixes.begin(),
                     kReservedServerPrefixes.end(),
                     [&server_name](const std::string& reserved_prefix) {
                       return boost::istarts_with(server_name, reserved_prefix);
                     });
}

void throw_reserved_server_prefix_exception() {
  std::string error_message{"Foreign server names cannot start with "};
  for (size_t i = 0; i < kReservedServerPrefixes.size(); i++) {
    if (i > 0) {
      error_message += ", ";
    }
    if (i == kReservedServerPrefixes.size() - 1) {
      error_message += "or ";
    }
    error_message += "\"" + kReservedServerPrefixes[i] + "\"";
  }
  error_message += ".";
  throw std::runtime_error{error_message};
}
}  // namespace

CreateForeignServerCommand::CreateForeignServerCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  if (!g_enable_fsi) {
    throw std::runtime_error("Unsupported command: CREATE FOREIGN SERVER");
  }
  auto& ddl_payload = extractPayload(ddl_data_);
  CHECK(ddl_payload.HasMember("serverName"));
  CHECK(ddl_payload["serverName"].IsString());
  CHECK(ddl_payload.HasMember("dataWrapper"));
  CHECK(ddl_payload["dataWrapper"].IsString());
  if (ddl_payload.HasMember("options")) {
    CHECK(ddl_payload["options"].IsObject());
  }
  CHECK(ddl_payload.HasMember("ifNotExists"));
  CHECK(ddl_payload["ifNotExists"].IsBool());
}

ExecutionResult CreateForeignServerCommand::execute(bool read_only_mode) {
  auto execute_write_lock = legacylockmgr::getExecuteWriteLock();

  ExecutionResult result;

  if (read_only_mode) {
    throw std::runtime_error("CREATE FOREIGN SERVER invalid in read only mode.");
  }

  auto& ddl_payload = extractPayload(ddl_data_);
  std::string server_name = ddl_payload["serverName"].GetString();
  if (is_default_server(server_name)) {
    throw_reserved_server_prefix_exception();
  }
  bool if_not_exists = ddl_payload["ifNotExists"].GetBool();
  if (session_ptr_->getCatalog().getForeignServer(server_name)) {
    if (if_not_exists) {
      return result;
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
  foreign_server->data_wrapper_type = to_upper(ddl_payload["dataWrapper"].GetString());
  foreign_server->name = server_name;
  foreign_server->user_id = current_user.userId;
  if (ddl_payload.HasMember("options")) {
    foreign_server->populateOptionsMap(ddl_payload["options"]);
  }
  foreign_server->validate();

  auto& catalog = session_ptr_->getCatalog();
  catalog.createForeignServer(std::move(foreign_server),
                              ddl_payload["ifNotExists"].GetBool());
  Catalog_Namespace::SysCatalog::instance().createDBObject(
      current_user, server_name, ServerDBObjectType, catalog);

  return result;
}

AlterDatabaseCommand::AlterDatabaseCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<const Catalog_Namespace::SessionInfo> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  auto& ddl_payload = extractPayload(ddl_data_);
  CHECK(ddl_payload.HasMember("databaseName"));
  CHECK(ddl_payload["databaseName"].IsString());
  CHECK(ddl_payload.HasMember("alterType"));
  CHECK(ddl_payload["alterType"].IsString());
  if (ddl_payload["alterType"] == "RENAME_DATABASE") {
    CHECK(ddl_payload.HasMember("newDatabaseName"));
    CHECK(ddl_payload["newDatabaseName"].IsString());
  } else if (ddl_payload["alterType"] == "CHANGE_OWNER") {
    CHECK(ddl_payload.HasMember("newOwner"));
    CHECK(ddl_payload["newOwner"].IsString());
  } else {
    UNREACHABLE();  // not-implemented alterType
  }
}

ExecutionResult AlterDatabaseCommand::execute(bool read_only_mode) {
  auto execute_write_lock = legacylockmgr::getExecuteWriteLock();

  if (read_only_mode) {
    throw std::runtime_error("ALTER DATABASE invalid in read only mode.");
  }
  auto& ddl_payload = extractPayload(ddl_data_);
  std::string databaseName = ddl_payload["databaseName"].GetString();

  auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
  Catalog_Namespace::DBMetadata db;
  if (!sys_cat.getMetadataForDB(databaseName, db)) {
    throw std::runtime_error("Database " + databaseName + " does not exists.");
  }

  std::string alter_type = ddl_payload["alterType"].GetString();
  if (alter_type == "CHANGE_OWNER") {
    changeOwner();
  } else if (alter_type == "RENAME_DATABASE") {
    rename();
  }

  return ExecutionResult();
}

void AlterDatabaseCommand::rename() {
  auto& ddl_payload = extractPayload(ddl_data_);
  std::string database_name = ddl_payload["databaseName"].GetString();
  std::string new_database_name = ddl_payload["newDatabaseName"].GetString();

  Catalog_Namespace::DBMetadata db;
  CHECK(Catalog_Namespace::SysCatalog::instance().getMetadataForDB(database_name, db));

  if (!session_ptr_->get_currentUser().isSuper &&
      session_ptr_->get_currentUser().userId != db.dbOwner) {
    throw std::runtime_error("Only a super user or the owner can rename the database.");
  }

  Catalog_Namespace::SysCatalog::instance().renameDatabase(database_name,
                                                           new_database_name);
}

void AlterDatabaseCommand::changeOwner() {
  auto& ddl_payload = extractPayload(ddl_data_);
  std::string database_name = ddl_payload["databaseName"].GetString();
  std::string new_owner = ddl_payload["newOwner"].GetString();
  auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
  if (!session_ptr_->get_currentUser().isSuper) {
    throw std::runtime_error(
        "Only a super user can change a database's owner. "
        "Current user is not a super-user. "
        "Database with name \"" +
        database_name + "\" will not have owner changed.");
  }

  sys_cat.changeDatabaseOwner(database_name, new_owner);
}

AlterForeignServerCommand::AlterForeignServerCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<const Catalog_Namespace::SessionInfo> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  if (!g_enable_fsi) {
    throw std::runtime_error("Unsupported command: ALTER FOREIGN SERVER");
  }
  auto& ddl_payload = extractPayload(ddl_data_);
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

ExecutionResult AlterForeignServerCommand::execute(bool read_only_mode) {
  auto execute_write_lock = legacylockmgr::getExecuteWriteLock();

  if (read_only_mode) {
    throw std::runtime_error("ALTER FOREIGN SERVER invalid in read only mode.");
  }
  auto& ddl_payload = extractPayload(ddl_data_);
  std::string server_name = ddl_payload["serverName"].GetString();
  if (is_default_server(server_name)) {
    throw std::runtime_error{"Default servers cannot be altered."};
  }
  if (!session_ptr_->getCatalog().getForeignServer(server_name)) {
    throw std::runtime_error{"Foreign server with name \"" + server_name +
                             "\" does not exist and can not be altered."};
  }
  if (!hasAlterServerPrivileges()) {
    throw std::runtime_error("Server " + server_name +
                             " can not be altered. User has no ALTER SERVER privileges.");
  }
  std::string alter_type = ddl_payload["alterType"].GetString();
  if (alter_type == "CHANGE_OWNER") {
    changeForeignServerOwner();
  } else if (alter_type == "SET_DATA_WRAPPER") {
    setForeignServerDataWrapper();
  } else if (alter_type == "SET_OPTIONS") {
    setForeignServerOptions();
  } else if (alter_type == "RENAME_SERVER") {
    renameForeignServer();
  }

  return ExecutionResult();
}

void AlterForeignServerCommand::changeForeignServerOwner() {
  auto& ddl_payload = extractPayload(ddl_data_);
  std::string server_name = ddl_payload["serverName"].GetString();
  std::string new_owner = ddl_payload["newOwner"].GetString();
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
  auto& ddl_payload = extractPayload(ddl_data_);
  std::string server_name = ddl_payload["serverName"].GetString();
  std::string new_server_name = ddl_payload["newServerName"].GetString();
  if (is_default_server(new_server_name)) {
    throw_reserved_server_prefix_exception();
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
  auto& ddl_payload = extractPayload(ddl_data_);
  std::string server_name = ddl_payload["serverName"].GetString();
  auto& cat = session_ptr_->getCatalog();
  // update catalog
  const auto foreign_server = cat.getForeignServer(server_name);
  foreign_storage::OptionsContainer opt;
  opt.populateOptionsMap(foreign_server->getOptionsAsJsonString());
  opt.populateOptionsMap(ddl_payload["options"]);
  cat.setForeignServerOptions(server_name, opt.getOptionsAsJsonString());
}

void AlterForeignServerCommand::setForeignServerDataWrapper() {
  auto& ddl_payload = extractPayload(ddl_data_);
  std::string server_name = ddl_payload["serverName"].GetString();
  std::string data_wrapper = ddl_payload["dataWrapper"].GetString();
  auto& cat = session_ptr_->getCatalog();
  // update catalog
  cat.setForeignServerDataWrapper(server_name, data_wrapper);
}

bool AlterForeignServerCommand::hasAlterServerPrivileges() {
  // TODO: implement `GRANT/REVOKE ALTER_SERVER` DDL commands
  auto& ddl_payload = extractPayload(ddl_data_);
  std::string server_name = ddl_payload["serverName"].GetString();
  return session_ptr_->checkDBAccessPrivileges(
      DBObjectType::ServerDBObjectType, AccessPrivileges::ALTER_SERVER, server_name);
}

DropForeignServerCommand::DropForeignServerCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  if (!g_enable_fsi) {
    throw std::runtime_error("Unsupported command: DROP FOREIGN SERVER");
  }
  auto& ddl_payload = extractPayload(ddl_data_);
  CHECK(ddl_payload.HasMember("serverName"));
  CHECK(ddl_payload["serverName"].IsString());
  CHECK(ddl_payload.HasMember("ifExists"));
  CHECK(ddl_payload["ifExists"].IsBool());
}

ExecutionResult DropForeignServerCommand::execute(bool read_only_mode) {
  auto execute_write_lock = legacylockmgr::getExecuteWriteLock();

  if (read_only_mode) {
    throw std::runtime_error("DROP FOREIGN SERVER invalid in read only mode.");
  }

  auto& ddl_payload = extractPayload(ddl_data_);
  std::string server_name = ddl_payload["serverName"].GetString();
  if (is_default_server(server_name)) {
    throw std::runtime_error{"Default servers cannot be dropped."};
  }
  bool if_exists = ddl_payload["ifExists"].GetBool();
  if (!session_ptr_->getCatalog().getForeignServer(server_name)) {
    if (if_exists) {
      return ExecutionResult();
    } else {
      throw std::runtime_error{"Foreign server with name \"" + server_name +
                               "\" can not be dropped. Server does not exist."};
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
  session_ptr_->getCatalog().dropForeignServer(ddl_payload["serverName"].GetString());

  return ExecutionResult();
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
  if (boost::iequals(type, "MULTILINESTRING")) {
    return kMULTILINESTRING;
  }
  if (boost::iequals(type, "MULTIPOLYGON")) {
    return kMULTIPOLYGON;
  }
  if (boost::iequals(type, "POINT")) {
    return kPOINT;
  }
  if (boost::iequals(type, "MULTIPOINT")) {
    return kMULTIPOINT;
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
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  if (!g_enable_fsi) {
    throw std::runtime_error("Unsupported command: CREATE FOREIGN TABLE");
  }
  auto& ddl_payload = extractPayload(ddl_data);
  CHECK(ddl_payload.HasMember("serverName"));
  CHECK(ddl_payload["serverName"].IsString());
  CHECK(ddl_payload.HasMember("tableName"));
  CHECK(ddl_payload["tableName"].IsString());
  CHECK(ddl_payload.HasMember("ifNotExists"));
  CHECK(ddl_payload["ifNotExists"].IsBool());
  CHECK(ddl_payload.HasMember("columns"));
  CHECK(ddl_payload["columns"].IsArray());
}

ExecutionResult CreateForeignTableCommand::execute(bool read_only_mode) {
  auto execute_write_lock = legacylockmgr::getExecuteWriteLock();

  auto& catalog = session_ptr_->getCatalog();
  auto& ddl_payload = extractPayload(ddl_data_);

  if (read_only_mode) {
    throw std::runtime_error("CREATE FOREIGN TABLE invalid in read only mode.");
  }

  const std::string& table_name = ddl_payload["tableName"].GetString();
  if (!session_ptr_->checkDBAccessPrivileges(DBObjectType::TableDBObjectType,
                                             AccessPrivileges::CREATE_TABLE)) {
    throw std::runtime_error(
        "Foreign table \"" + table_name +
        "\" will not be created. User has no CREATE TABLE privileges.");
  }

  bool if_not_exists = ddl_payload["ifNotExists"].GetBool();
  if (!catalog.validateNonExistentTableOrView(table_name, if_not_exists)) {
    return ExecutionResult();
  }

  foreign_storage::ForeignTable foreign_table{};
  std::list<ColumnDescriptor> columns{};
  setColumnDetails(columns);
  setTableDetails(table_name, foreign_table, columns);
  catalog.createTable(foreign_table, columns, {}, true);

  // TODO (max): It's transactionally unsafe, should be fixed: we may create object w/o
  // privileges
  Catalog_Namespace::SysCatalog::instance().createDBObject(
      session_ptr_->get_currentUser(),
      foreign_table.tableName,
      TableDBObjectType,
      catalog);

  return ExecutionResult();
}

void CreateForeignTableCommand::setTableDetails(
    const std::string& table_name,
    TableDescriptor& td,
    const std::list<ColumnDescriptor>& columns) {
  ddl_utils::set_default_table_attributes(table_name, td, columns.size());
  td.userId = session_ptr_->get_currentUser().userId;
  td.storageType = StorageType::FOREIGN_TABLE;
  td.hasDeletedCol = false;
  td.keyMetainfo = "[]";
  td.fragments = "";
  td.partitions = "";

  auto& ddl_payload = extractPayload(ddl_data_);
  auto& foreign_table = dynamic_cast<foreign_storage::ForeignTable&>(td);
  const std::string server_name = ddl_payload["serverName"].GetString();
  foreign_table.foreign_server = session_ptr_->getCatalog().getForeignServer(server_name);
  if (!foreign_table.foreign_server) {
    throw std::runtime_error{
        "Foreign Table with name \"" + table_name +
        "\" can not be created. Associated foreign server with name \"" + server_name +
        "\" does not exist."};
  }

  // check server usage privileges
  if (!is_default_server(server_name) &&
      !session_ptr_->checkDBAccessPrivileges(DBObjectType::ServerDBObjectType,
                                             AccessPrivileges::SERVER_USAGE,
                                             server_name)) {
    throw std::runtime_error(
        "Current user does not have USAGE privilege on foreign server: " + server_name);
  }

  if (ddl_payload.HasMember("options") && !ddl_payload["options"].IsNull()) {
    CHECK(ddl_payload["options"].IsObject());
    foreign_table.initializeOptions(ddl_payload["options"]);
  } else {
    // Initialize options even if none were provided to verify a legal state.
    // This is necessary because some options (like "file_path") are optional only if a
    // paired option ("base_path") exists in the server.
    foreign_table.initializeOptions();
  }
  foreign_table.validateSchema(columns);

  if (const auto it = foreign_table.options.find("FRAGMENT_SIZE");
      it != foreign_table.options.end()) {
    foreign_table.maxFragRows = std::stoi(it->second);
  }

  if (const auto it = foreign_table.options.find("MAX_CHUNK_SIZE");
      it != foreign_table.options.end()) {
    foreign_table.maxChunkSize = std::stol(it->second);
  }

  if (const auto it = foreign_table.options.find("PARTITIONS");
      it != foreign_table.options.end()) {
    foreign_table.partitions = it->second;
  }
}

void CreateForeignTableCommand::setColumnDetails(std::list<ColumnDescriptor>& columns) {
  auto& ddl_payload = extractPayload(ddl_data_);
  std::unordered_set<std::string> column_names{};
  for (auto& column_def : ddl_payload["columns"].GetArray()) {
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
    ddl_utils::set_column_descriptor(column_name,
                                     cd,
                                     &sql_type,
                                     data_type["notNull"].GetBool(),
                                     encoding.get(),
                                     nullptr);
    columns.emplace_back(cd);
  }
}

DropForeignTableCommand::DropForeignTableCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  if (!g_enable_fsi) {
    throw std::runtime_error("Unsupported command: DROP FOREIGN TABLE");
  }
  auto& ddl_payload = extractPayload(ddl_data_);
  CHECK(ddl_payload.HasMember("tableName"));
  CHECK(ddl_payload["tableName"].IsString());
  CHECK(ddl_payload.HasMember("ifExists"));
  CHECK(ddl_payload["ifExists"].IsBool());
}

ExecutionResult DropForeignTableCommand::execute(bool read_only_mode) {
  auto execute_write_lock = legacylockmgr::getExecuteWriteLock();

  auto& catalog = session_ptr_->getCatalog();
  auto& ddl_payload = extractPayload(ddl_data_);

  if (read_only_mode) {
    throw std::runtime_error("DROP FOREIGN TABLE invalid in read only mode.");
  }
  const std::string& table_name = ddl_payload["tableName"].GetString();
  const TableDescriptor* td{nullptr};
  std::unique_ptr<lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>> td_with_lock;

  try {
    td_with_lock =
        std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>>(
            lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
                catalog, table_name, false));
    CHECK(td_with_lock);
    td = (*td_with_lock)();
  } catch (const std::runtime_error& e) {
    // TODO(Misiu): This should not just swallow any exception, it should only catch
    // exceptions that stem from the table not existing.
    if (ddl_payload["ifExists"].GetBool()) {
      return ExecutionResult();
    } else {
      throw e;
    }
  }

  CHECK(td);

  if (!session_ptr_->checkDBAccessPrivileges(
          DBObjectType::TableDBObjectType, AccessPrivileges::DROP_TABLE, table_name)) {
    throw std::runtime_error(
        "Foreign table \"" + table_name +
        "\" will not be dropped. User has no DROP TABLE privileges.");
  }

  ddl_utils::validate_table_type(td, ddl_utils::TableType::FOREIGN_TABLE, "DROP");
  auto table_data_write_lock =
      lockmgr::TableDataLockMgr::getWriteLockForTable(catalog, table_name);
  catalog.dropTable(td);

  // TODO(Misiu): Implement per-table cache invalidation.
  DeleteTriggeredCacheInvalidator::invalidateCaches();

  return ExecutionResult();
}

ShowTablesCommand::ShowTablesCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {}

ExecutionResult ShowTablesCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  // Get all table names in the same way as OmniSql \t command

  // valid in read_only_mode

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

ExecutionResult ShowTableDetailsCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  const auto catalog = session_ptr_->get_catalog_ptr();
  std::vector<std::string> filtered_table_names = getFilteredTableNames();

  // valid in read_only_mode

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
  exec_for_tables_which_exist(filtered_table_names,
                              catalog.get(),
                              [&logical_values, &catalog](const TableDescriptor* td,
                                                          const std::string& table_name) {
                                auto agg_storage_stats =
                                    get_agg_storage_stats(td, catalog.get());
                                add_table_details(logical_values, td, agg_storage_stats);
                              });

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
      if (td->isForeignTable()) {
        throw std::runtime_error{
            "SHOW TABLE DETAILS is not supported for foreign tables. Table name: " +
            table_name + "."};
      }
      if (td->isTemporaryTable()) {
        throw std::runtime_error{
            "SHOW TABLE DETAILS is not supported for temporary tables. Table name: " +
            table_name + "."};
      }
      filtered_table_names.emplace_back(table_name);
    }
  } else {
    exec_for_tables_which_exist(all_table_names,
                                catalog.get(),
                                [&filtered_table_names](const TableDescriptor* td,
                                                        const std::string& table_name) {
                                  if (td->isForeignTable() || td->isTemporaryTable()) {
                                    return;
                                  }
                                  filtered_table_names.emplace_back(table_name);
                                });
  }
  return filtered_table_names;
}

ShowCreateTableCommand::ShowCreateTableCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {}

ExecutionResult ShowCreateTableCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  // valid in read_only_mode

  auto& ddl_payload = extractPayload(ddl_data_);
  CHECK(ddl_payload.HasMember("tableName"));
  CHECK(ddl_payload["tableName"].IsString());
  const std::string& table_name = ddl_payload["tableName"].GetString();

  auto& catalog = session_ptr_->getCatalog();
  auto table_read_lock =
      lockmgr::TableSchemaLockMgr::getReadLockForTable(catalog, table_name);

  const TableDescriptor* td = catalog.getMetadataForTable(table_name, false);
  if (!td) {
    throw std::runtime_error("Table/View " + table_name + " does not exist.");
  }

  DBObject dbObject(td->tableName, td->isView ? ViewDBObjectType : TableDBObjectType);
  dbObject.loadKey(catalog);
  std::vector<DBObject> privObjects = {dbObject};

  if (!Catalog_Namespace::SysCatalog::instance().hasAnyPrivileges(
          session_ptr_->get_currentUser(), privObjects)) {
    throw std::runtime_error("Table/View " + table_name + " does not exist.");
  }
  if (td->isView && !session_ptr_->get_currentUser().isSuper) {
    auto query_state = query_state::QueryState::create(session_ptr_, td->viewSQL);
    auto query_state_proxy = query_state->createQueryStateProxy();
    auto calcite_mgr = catalog.getCalciteMgr();
    const auto calciteQueryParsingOption =
        calcite_mgr->getCalciteQueryParsingOption(true, false, false);
    const auto calciteOptimizationOption = calcite_mgr->getCalciteOptimizationOption(
        false,
        g_enable_watchdog,
        {},
        Catalog_Namespace::SysCatalog::instance().isAggregator());
    auto result = calcite_mgr->process(query_state_proxy,
                                       td->viewSQL,
                                       calciteQueryParsingOption,
                                       calciteOptimizationOption);
    try {
      calcite_mgr->checkAccessedObjectsPrivileges(query_state_proxy, result);
    } catch (const std::runtime_error&) {
      throw std::runtime_error("Not enough privileges to show the view SQL");
    }
  }
  // Construct
  auto create_table_sql = catalog.dumpCreateTable(td);
  ExecutionResult result;
  result.updateResultSet(create_table_sql, ExecutionResult::SimpleResult);
  return result;
}

ShowDatabasesCommand::ShowDatabasesCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {}

ExecutionResult ShowDatabasesCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  // valid in read_only_mode

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

ShowFunctionsCommand::ShowFunctionsCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {}

ExecutionResult ShowFunctionsCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  // Get all row-wise functions
  auto& ddl_payload = extractPayload(ddl_data_);
  std::vector<TargetMetaInfo> label_infos;
  std::vector<RelLogicalValues::RowValues> logical_values;

  if (ddl_payload.HasMember("ScalarFnNames")) {
    // label_infos -> column labels
    label_infos.emplace_back("name", SQLTypeInfo(kTEXT, true));
    label_infos.emplace_back("signature", SQLTypeInfo(kTEXT, true));
    label_infos.emplace_back("CPU", SQLTypeInfo(kBOOLEAN, true));
    label_infos.emplace_back("GPU", SQLTypeInfo(kBOOLEAN, true));
    label_infos.emplace_back("Runtime", SQLTypeInfo(kBOOLEAN, true));
    for (const auto& udf_name_json : ddl_payload["ScalarFnNames"].GetArray()) {
      std::string udf_name = udf_name_json.GetString();
      std::vector<ExtensionFunction> ext_funcs =
          ExtensionFunctionsWhitelist::get_ext_funcs(udf_name);

      for (ExtensionFunction& fn : ext_funcs) {
        logical_values.emplace_back(RelLogicalValues::RowValues{});
        // Name
        logical_values.back().emplace_back(genLiteralStr(udf_name));
        // Signature
        logical_values.back().emplace_back(genLiteralStr(fn.toSignature()));
        // CPU?
        logical_values.back().emplace_back(genLiteralBoolean(fn.isCPU()));
        // GPU?
        logical_values.back().emplace_back(genLiteralBoolean(fn.isGPU()));
        // Runtime?
        logical_values.back().emplace_back(genLiteralBoolean(fn.isRuntime()));
      }
    }

  } else {
    // label_infos -> column labels
    for (const auto& label : {"Scalar UDF"}) {
      label_infos.emplace_back(label, SQLTypeInfo(kTEXT, true));
    }

    // logical_values -> table data
    for (auto name : ExtensionFunctionsWhitelist::get_udfs_name(/* is_runtime */ false)) {
      logical_values.emplace_back(RelLogicalValues::RowValues{});
      logical_values.back().emplace_back(genLiteralStr(name));
    }
  }

  // Create ResultSet
  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

ShowRuntimeFunctionsCommand::ShowRuntimeFunctionsCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {}

ExecutionResult ShowRuntimeFunctionsCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  // Get all runtime row-wise functions
  std::vector<TargetMetaInfo> label_infos;
  std::vector<RelLogicalValues::RowValues> logical_values;

  // label_infos -> column labels
  for (const auto& label : {"Runtime Scalar UDF"}) {
    label_infos.emplace_back(label, SQLTypeInfo(kTEXT, true));
  }

  // logical_values -> table data
  for (auto name : ExtensionFunctionsWhitelist::get_udfs_name(/* is_runtime */ true)) {
    logical_values.emplace_back(RelLogicalValues::RowValues{});
    logical_values.back().emplace_back(genLiteralStr(name));
  }

  // Create ResultSet
  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

ShowTableFunctionsCommand::ShowTableFunctionsCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {}

ExecutionResult ShowTableFunctionsCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  // valid in read_only_mode

  // Get all table functions
  auto& ddl_payload = extractPayload(ddl_data_);
  std::vector<TargetMetaInfo> label_infos;
  std::vector<RelLogicalValues::RowValues> logical_values;

  if (ddl_payload.HasMember("tfNames")) {
    // label_infos -> column labels
    label_infos.emplace_back("name", SQLTypeInfo(kTEXT, true));
    label_infos.emplace_back("signature", SQLTypeInfo(kTEXT, true));
    label_infos.emplace_back("input_names", SQLTypeInfo(kTEXT, true));
    label_infos.emplace_back("input_types", SQLTypeInfo(kTEXT, true));
    label_infos.emplace_back("input_arg_defaults", SQLTypeInfo(kTEXT, true));
    label_infos.emplace_back("output_names", SQLTypeInfo(kTEXT, true));
    label_infos.emplace_back("output_types", SQLTypeInfo(kTEXT, true));
    label_infos.emplace_back("CPU", SQLTypeInfo(kBOOLEAN, true));
    label_infos.emplace_back("GPU", SQLTypeInfo(kBOOLEAN, true));
    label_infos.emplace_back("Runtime", SQLTypeInfo(kBOOLEAN, true));
    label_infos.emplace_back("filter_table_transpose", SQLTypeInfo(kBOOLEAN, true));
    // logical_values -> table data
    for (const auto& tf_name_json : ddl_payload["tfNames"].GetArray()) {
      std::string tf_name = tf_name_json.GetString();
      auto tfs = table_functions::TableFunctionsFactory::get_table_funcs(tf_name);
      for (table_functions::TableFunction& tf : tfs) {
        logical_values.emplace_back(RelLogicalValues::RowValues{});
        // Name
        logical_values.back().emplace_back(genLiteralStr(tf.getName(true, false)));
        // Signature
        logical_values.back().emplace_back(genLiteralStr(
            tf.getSignature(/*include_name*/ false, /*include_output*/ true)));
        // Input argument names
        logical_values.back().emplace_back(
            genLiteralStr(tf.getArgNames(/* use_input_args */ true)));
        // Input argument types
        logical_values.back().emplace_back(
            genLiteralStr(tf.getArgTypes(/* use_input_args */ true)));
        // Input argument default values
        logical_values.back().emplace_back(genLiteralStr(tf.getInputArgsDefaultValues()));
        // Output argument names
        logical_values.back().emplace_back(
            genLiteralStr(tf.getArgNames(/* use_input_args */ false)));
        // Output argument types
        logical_values.back().emplace_back(
            genLiteralStr(tf.getArgTypes(/* use_input_args */ false)));
        // CPU?
        logical_values.back().emplace_back(genLiteralBoolean(tf.isCPU()));
        // GPU?
        logical_values.back().emplace_back(genLiteralBoolean(tf.isGPU()));
        // Runtime?
        logical_values.back().emplace_back(genLiteralBoolean(tf.isRuntime()));
        // Implements filter_table_transpose?
        logical_values.back().emplace_back(genLiteralBoolean(
            !tf.getFunctionAnnotation("filter_table_function_transpose", "").empty()));
      }
    }
  } else {
    // label_infos -> column labels
    for (const auto& label : {"Table UDF"}) {
      label_infos.emplace_back(label, SQLTypeInfo(kTEXT, true));
    }

    // logical_values -> table data
    std::unordered_set<std::string> unique_names;
    for (auto tf : table_functions::TableFunctionsFactory::get_table_funcs(
             /* is_runtime */ false)) {
      std::string name = tf.getName(true, true);
      if (unique_names.find(name) == unique_names.end()) {
        unique_names.emplace(name);
        logical_values.emplace_back(RelLogicalValues::RowValues{});
        logical_values.back().emplace_back(genLiteralStr(name));
      }
    }
  }

  // Create ResultSet
  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

ShowRuntimeTableFunctionsCommand::ShowRuntimeTableFunctionsCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {}

ExecutionResult ShowRuntimeTableFunctionsCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  // valid in read_only_mode

  // Get all runtime table functions
  std::vector<TargetMetaInfo> label_infos;
  std::vector<RelLogicalValues::RowValues> logical_values;

  // label_infos -> column labels
  for (const auto& label : {"Runtime Table UDF"}) {
    label_infos.emplace_back(label, SQLTypeInfo(kTEXT, true));
  }

  // logical_values -> table data
  std::unordered_set<std::string> unique_names;
  for (auto tf :
       table_functions::TableFunctionsFactory::get_table_funcs(/* is_runtime */ true)) {
    std::string name = tf.getName(true, true);
    if (unique_names.find(name) == unique_names.end()) {
      unique_names.emplace(name);
      logical_values.emplace_back(RelLogicalValues::RowValues{});
      logical_values.back().emplace_back(genLiteralStr(name));
    }
  }

  // Create ResultSet
  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

ShowForeignServersCommand::ShowForeignServersCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  if (!g_enable_fsi) {
    throw std::runtime_error("Unsupported command: SHOW FOREIGN SERVERS");
  }
  // Verify that members are valid
  auto& ddl_payload = extractPayload(ddl_data_);
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

ExecutionResult ShowForeignServersCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  // valid in read_only_mode

  std::vector<TargetMetaInfo> label_infos;
  auto& ddl_payload = extractPayload(ddl_data_);

  // label_infos -> column labels
  std::vector<std::string> labels{"server_name", "data_wrapper", "created_at", "options"};
  label_infos.emplace_back(labels[0], SQLTypeInfo(kTEXT, true));
  label_infos.emplace_back(labels[1], SQLTypeInfo(kTEXT, true));
  // created_at is a TIMESTAMP
  label_infos.emplace_back(labels[2], SQLTypeInfo(kTIMESTAMP, true));
  label_infos.emplace_back(labels[3], SQLTypeInfo(kTEXT, true));

  const auto& user = session_ptr_->get_currentUser();

  std::vector<const foreign_storage::ForeignServer*> results;

  session_ptr_->getCatalog().getForeignServersForUser(
      extractFilters(ddl_payload), user, results);

  // logical_values -> table data
  std::vector<RelLogicalValues::RowValues> logical_values;
  for (auto const& server_ptr : results) {
    logical_values.emplace_back(RelLogicalValues::RowValues{});
    logical_values.back().emplace_back(genLiteralStr(server_ptr->name));
    logical_values.back().emplace_back(genLiteralStr(server_ptr->data_wrapper_type));
    logical_values.back().emplace_back(genLiteralTimestamp(server_ptr->creation_time));
    logical_values.back().emplace_back(
        genLiteralStr(server_ptr->getOptionsAsJsonString()));
  }

  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

ShowCreateServerCommand::ShowCreateServerCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  if (!g_enable_fsi) {
    throw std::runtime_error("Unsupported command: SHOW CREATE SERVER");
  }
  // Verify that members are valid
  auto& payload = extractPayload(ddl_data_);
  CHECK(payload.HasMember("serverName"));
  CHECK(payload["serverName"].IsString());
  server_ = (payload["serverName"].GetString());
}

ExecutionResult ShowCreateServerCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();
  // valid in read_only_mode

  using namespace Catalog_Namespace;
  auto& catalog = session_ptr_->getCatalog();
  const auto server = catalog.getForeignServer(server_);
  if (!server) {
    throw std::runtime_error("Foreign server " + server_ + " does not exist.");
  }
  DBObject dbObject(server_, ServerDBObjectType);
  dbObject.loadKey(catalog);
  std::vector<DBObject> privObjects = {dbObject};
  if (!SysCatalog::instance().hasAnyPrivileges(session_ptr_->get_currentUser(),
                                               privObjects)) {
    throw std::runtime_error("Foreign server " + server_ + " does not exist.");
  }
  auto create_stmt = catalog.dumpCreateServer(server_);

  std::vector<std::string> labels{"create_server_sql"};
  std::vector<TargetMetaInfo> label_infos;
  label_infos.emplace_back(labels[0], SQLTypeInfo(kTEXT, true));

  std::vector<RelLogicalValues::RowValues> logical_values;
  logical_values.emplace_back(RelLogicalValues::RowValues{});
  logical_values.back().emplace_back(genLiteralStr(create_stmt));

  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

RefreshForeignTablesCommand::RefreshForeignTablesCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  if (!g_enable_fsi) {
    throw std::runtime_error("Unsupported command: REFRESH FOREIGN TABLE");
  }
  auto& ddl_payload = extractPayload(ddl_data_);
  CHECK(ddl_payload.HasMember("tableNames"));
  CHECK(ddl_payload["tableNames"].IsArray());
  for (auto const& tablename_def : ddl_payload["tableNames"].GetArray()) {
    CHECK(tablename_def.IsString());
  }
}

ExecutionResult RefreshForeignTablesCommand::execute(bool read_only_mode) {
  if (read_only_mode) {
    throw std::runtime_error("REFRESH FOREIGN TABLE invalid in read only mode.");
  }

  const auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  bool evict_cached_entries{false};
  foreign_storage::OptionsContainer opt;
  auto& ddl_payload = extractPayload(ddl_data_);
  if (ddl_payload.HasMember("options") && !ddl_payload["options"].IsNull()) {
    opt.populateOptionsMap(ddl_payload["options"]);
    for (const auto& entry : opt.options) {
      if (entry.first != "EVICT") {
        throw std::runtime_error{
            "Invalid option \"" + entry.first +
            "\" provided for refresh command. Only \"EVICT\" option is supported."};
      }
    }
    CHECK(opt.options.find("EVICT") != opt.options.end());

    if (boost::iequals(opt.options["EVICT"], "true") ||
        boost::iequals(opt.options["EVICT"], "false")) {
      if (boost::iequals(opt.options["EVICT"], "true")) {
        evict_cached_entries = true;
      }
    } else {
      throw std::runtime_error{
          "Invalid value \"" + opt.options["EVICT"] +
          "\" provided for EVICT option. Value must be either \"true\" or \"false\"."};
    }
  }

  auto& cat = session_ptr_->getCatalog();
  const auto& current_user = session_ptr_->get_currentUser();
  /* verify object ownership if not suser */
  if (!current_user.isSuper) {
    for (const auto& table_name_json : ddl_payload["tableNames"].GetArray()) {
      std::string table_name = table_name_json.GetString();
      if (!Catalog_Namespace::SysCatalog::instance().verifyDBObjectOwnership(
              current_user, DBObject(table_name, TableDBObjectType), cat)) {
        throw std::runtime_error(
            std::string("REFRESH FOREIGN TABLES failed on table \"") + table_name +
            "\". It can only be executed by super user or "
            "owner of the "
            "object.");
      }
    }
  }

  for (const auto& table_name_json : ddl_payload["tableNames"].GetArray()) {
    std::string table_name = table_name_json.GetString();
    static const std::array<std::string, 4> log_system_tables{
        Catalog_Namespace::SERVER_LOGS_SYS_TABLE_NAME,
        Catalog_Namespace::REQUEST_LOGS_SYS_TABLE_NAME,
        Catalog_Namespace::WS_SERVER_LOGS_SYS_TABLE_NAME,
        Catalog_Namespace::WS_SERVER_ACCESS_LOGS_SYS_TABLE_NAME};
    if (cat.isInfoSchemaDb() && !shared::contains(log_system_tables, table_name)) {
      throw std::runtime_error(
          "REFRESH FOREIGN TABLE can only be executed for the following tables: " +
          join(log_system_tables, ","));
    }
    foreign_storage::refresh_foreign_table(cat, table_name, evict_cached_entries);
  }

  // todo(yoonmin) : allow per-table cache invalidation for the foreign table
  UpdateTriggeredCacheInvalidator::invalidateCaches();

  return ExecutionResult();
}

void AlterTableAlterColumnCommand::clearChunk(Catalog_Namespace::Catalog* catalog,
                                              const ChunkKey& key,
                                              const MemoryLevel mem_level) {
  auto& data_mgr = catalog->getDataMgr();
  if (mem_level >= data_mgr.levelSizes_.size()) {
    return;
  }
  for (int device = 0; device < data_mgr.levelSizes_[mem_level]; ++device) {
    if (data_mgr.isBufferOnDevice(key, mem_level, device)) {
      data_mgr.deleteChunk(key, mem_level, device);
    }
  }
}

void AlterTableAlterColumnCommand::clearChunk(Catalog_Namespace::Catalog* catalog,
                                              const ChunkKey& key) {
  clearChunk(catalog, key, MemoryLevel::GPU_LEVEL);
  clearChunk(catalog, key, MemoryLevel::CPU_LEVEL);
  clearChunk(catalog, key, MemoryLevel::DISK_LEVEL);
}

void AlterTableAlterColumnCommand::clearRemainingChunks(
    const TableDescriptor* td,
    const AlterColumnTypePairs& src_dst_cds) {
  auto catalog = &session_ptr_->getCatalog();
  // for (non-geo) cases where the chunk keys change, chunks that remain with old chunk
  // key must be removed
  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (src_cd->columnType.is_varlen_indeed() != dst_cd->columnType.is_varlen_indeed()) {
      auto fragments = td->fragmenter->getFragmentsForQuery().fragments;
      for (const auto& fragment : fragments) {
        ChunkKey key = {
            catalog->getDatabaseId(), td->tableId, src_cd->columnId, fragment.fragmentId};
        if (src_cd->columnType.is_varlen_indeed()) {
          auto data_key = key;
          data_key.push_back(1);
          clearChunk(catalog, data_key);
          auto index_key = key;
          index_key.push_back(2);
          clearChunk(catalog, index_key);
        } else {  // no varlen case
          clearChunk(catalog, key);
        }
      }
    }
  }
}

AlterTableCommand::AlterTableCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<const Catalog_Namespace::SessionInfo> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  auto& ddl_payload = extractPayload(ddl_data_);

  CHECK(ddl_payload.HasMember("tableName"));
  CHECK(ddl_payload["tableName"].IsString());

  CHECK(ddl_payload.HasMember("alterType"));
  CHECK(ddl_payload["alterType"].IsString());
}

AlterTableAlterColumnCommand::AlterTableAlterColumnCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<const Catalog_Namespace::SessionInfo> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  auto& ddl_payload = extractPayload(ddl_data_);

  CHECK_EQ(std::string(ddl_payload["alterType"].GetString()), "ALTER_COLUMN");

  CHECK(ddl_payload.HasMember("alterData"));
  CHECK(ddl_payload["alterData"].IsArray());

  const auto elements = ddl_payload["alterData"].GetArray();
  for (const auto& element : elements) {
    CHECK(element.HasMember("type"));
    CHECK(element["type"].IsString());
    CHECK_EQ(std::string(element["type"].GetString()), "SQL_COLUMN_DECLARATION");

    CHECK(element.HasMember("name"));
    CHECK(element["name"].IsString());

    CHECK(element.HasMember("default"));

    CHECK(element.HasMember("nullable"));
    CHECK(element["nullable"].IsBool());

    CHECK(element.HasMember("encodingType"));
    CHECK(element.HasMember("encodingSize"));

    CHECK(element.HasMember("sqltype"));
    CHECK(element["sqltype"].IsString());
  }
}

void AlterTableAlterColumnCommand::alterColumns(const TableDescriptor* td,
                                                const AlterColumnTypePairs& src_dst_cds) {
  auto& catalog = session_ptr_->getCatalog();

  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (dst_cd->columnType.is_geometry()) {
      continue;
    }
    auto compare_result =
        ddl_utils::alter_column_utils::compare_column_descriptors(src_cd, dst_cd);
    CHECK(!compare_result.sql_types_match || !compare_result.defaults_match);

    catalog.alterColumnTypeTransactional(*dst_cd);
  }
}

std::list<const ColumnDescriptor*> AlterTableAlterColumnCommand::prepareColumns(
    const TableDescriptor* td,
    const AlterColumnTypePairs& src_dst_cds) {
  auto& catalog = session_ptr_->getCatalog();

  std::list<const ColumnDescriptor*> non_geo_cds;
  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (dst_cd->columnType.is_geometry()) {
      continue;
    }
    non_geo_cds.emplace_back(dst_cd);

    auto compare_result =
        ddl_utils::alter_column_utils::compare_column_descriptors(src_cd, dst_cd);
    if (compare_result.sql_types_match) {
      continue;
    }

    if (dst_cd->columnType.is_dict_encoded_type()) {
      catalog.addDictionaryTransactional(*dst_cd);
    }
  }

  return non_geo_cds;
}

std::list<std::list<ColumnDescriptor>> AlterTableAlterColumnCommand::prepareGeoColumns(
    const TableDescriptor* td,
    const AlterColumnTypePairs& src_dst_cds) {
  auto& catalog = session_ptr_->getCatalog();
  std::list<std::list<ColumnDescriptor>> physical_geo_columns;

  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (dst_cd->columnType.is_geometry()) {
      std::string col_name = src_cd->columnName;
      auto uuid = boost::uuids::random_generator()();
      catalog.renameColumn(td, src_cd, col_name + "_" + boost::uuids::to_string(uuid));

      physical_geo_columns.emplace_back();
      std::list<ColumnDescriptor>& phy_geo_columns = physical_geo_columns.back();
      catalog.expandGeoColumn(*dst_cd, phy_geo_columns);

      catalog.addColumnTransactional(*td, *dst_cd);

      for (auto& cd : phy_geo_columns) {
        catalog.addColumnTransactional(*td, cd);
      }
    }
  }

  return physical_geo_columns;
}

void AlterTableAlterColumnCommand::dropSourceGeoColumns(
    const TableDescriptor* td,
    const AlterColumnTypePairs& src_dst_cds) {
  auto& catalog = session_ptr_->getCatalog();
  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (!dst_cd->columnType.is_geometry()) {
      continue;
    }
    auto catalog_cd = catalog.getMetadataForColumn(src_cd->tableId, src_cd->columnId);
    catalog.dropColumnTransactional(*td, *catalog_cd);
    ChunkKey col_key{catalog.getCurrentDB().dbId, td->tableId, src_cd->columnId};
    auto& data_mgr = catalog.getDataMgr();
    data_mgr.deleteChunksWithPrefix(col_key, MemoryLevel::GPU_LEVEL);
    data_mgr.deleteChunksWithPrefix(col_key, MemoryLevel::CPU_LEVEL);
    data_mgr.deleteChunksWithPrefix(col_key, MemoryLevel::DISK_LEVEL);
  }
}

void AlterTableAlterColumnCommand::alterNonGeoColumnData(
    const TableDescriptor* td,
    const std::list<const ColumnDescriptor*>& cds) {
  if (cds.empty()) {
    return;
  }
  auto fragmenter = td->fragmenter;
  CHECK(fragmenter);
  auto io_fragmenter =
      dynamic_cast<Fragmenter_Namespace::InsertOrderFragmenter*>(fragmenter.get());
  CHECK(io_fragmenter);
  io_fragmenter->alterNonGeoColumnType(cds);
}

void AlterTableAlterColumnCommand::alterGeoColumnData(
    const TableDescriptor* td,
    const std::list<std::pair<const ColumnDescriptor*,
                              std::list<const ColumnDescriptor*>>>& geo_src_dst_cds) {
  if (geo_src_dst_cds.empty()) {
    return;
  }
  auto fragmenter = td->fragmenter;
  CHECK(fragmenter);
  auto io_fragmenter =
      dynamic_cast<Fragmenter_Namespace::InsertOrderFragmenter*>(fragmenter.get());
  CHECK(io_fragmenter);
  io_fragmenter->alterColumnGeoType(geo_src_dst_cds);
}

void AlterTableAlterColumnCommand::clearInMemoryData(
    const TableDescriptor* td,
    const AlterColumnTypePairs& src_dst_cds) {
  auto& catalog = session_ptr_->getCatalog();

  ChunkKey column_key{catalog.getCurrentDB().dbId, td->tableId, 0};
  column_key.resize(2);
  UpdateTriggeredCacheInvalidator::invalidateCachesByTable(boost::hash_value(column_key));
  column_key.resize(3);
  for (auto& [src_cd, _] : src_dst_cds) {
    column_key[2] = src_cd->columnId;
    catalog.getDataMgr().deleteChunksWithPrefix(column_key, MemoryLevel::GPU_LEVEL);
    catalog.getDataMgr().deleteChunksWithPrefix(column_key, MemoryLevel::CPU_LEVEL);
  }

  catalog.removeFragmenterForTable(td->tableId);
}

void AlterTableAlterColumnCommand::deleteDictionaries(
    const TableDescriptor* td,
    const AlterColumnTypePairs& src_dst_cds) {
  auto& catalog = session_ptr_->getCatalog();
  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (!src_cd->columnType.is_dict_encoded_type()) {
      continue;
    }
    if (!ddl_utils::alter_column_utils::compare_column_descriptors(src_cd, dst_cd)
             .sql_types_match) {
      catalog.delDictionaryTransactional(*src_cd);
    }
  }
}

void AlterTableAlterColumnCommand::checkpoint(const TableDescriptor* td,
                                              const AlterColumnTypePairs& src_dst_cds) {
  auto& catalog = session_ptr_->getCatalog();
  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (!dst_cd->columnType.is_dict_encoded_type()) {
      continue;
    }
    if (ddl_utils::alter_column_utils::compare_column_descriptors(src_cd, dst_cd)
            .sql_types_match) {
      continue;
    }
    auto string_dictionary =
        catalog.getMetadataForDict(dst_cd->columnType.get_comp_param(), true)
            ->stringDict.get();
    if (!string_dictionary->checkpoint()) {
      throw std::runtime_error("Failed to checkpoint dictionary while altering column " +
                               dst_cd->columnName + ".");
    }
  }
  catalog.checkpointWithAutoRollback(td->tableId);
}

void AlterTableAlterColumnCommand::collectExpectedCatalogChanges(
    const TableDescriptor* td,
    const AlterTableAlterColumnCommand::AlterColumnTypePairs& src_dst_cds) {
  auto& catalog = session_ptr_->getCatalog();

  auto column_id_start = catalog.getNextAddedColumnId(*td);
  auto current_column_id = column_id_start;

  // Simulate operations required for geo changes
  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (dst_cd->columnType.is_geometry()) {
      renamed_columns_.push_back(*src_cd);

      std::list<ColumnDescriptor> phy_geo_columns;
      catalog.expandGeoColumn(*dst_cd, phy_geo_columns);

      auto col_to_add = *dst_cd;
      col_to_add.tableId = td->tableId;
      col_to_add.columnId = current_column_id++;
      added_columns_.push_back(col_to_add);

      for (auto& cd : phy_geo_columns) {
        ColumnDescriptor phys_col_to_add = cd;
        phys_col_to_add.tableId = td->tableId;
        phys_col_to_add.columnId = current_column_id++;
        added_columns_.push_back(phys_col_to_add);
      }
    } else if (dst_cd->columnType.is_dict_encoded_type()) {
      if (!ddl_utils::alter_column_utils::compare_column_descriptors(src_cd, dst_cd)
               .sql_types_match) {
        updated_dict_cds_.push_back(*src_cd);
      }
    }
  }

  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (dst_cd->columnType.is_geometry()) {
      continue;
    }
    auto compare_result =
        ddl_utils::alter_column_utils::compare_column_descriptors(src_cd, dst_cd);
    CHECK(!compare_result.sql_types_match || !compare_result.defaults_match);
    altered_columns_.push_back(ColumnAltered{*src_cd, *dst_cd});
  }
}

void AlterTableAlterColumnCommand::rollback(
    const TableDescriptor* td,
    const AlterTableAlterColumnCommand::AlterColumnTypePairs& src_dst_cds) {
  auto& catalog = session_ptr_->getCatalog();
  auto cds = catalog.getAllColumnMetadataForTable(td->tableId, false, false, true);

  // Drop any columns that were added
  std::list<const ColumnDescriptor*> added_columns_to_drop;
  for (auto& added_column : added_columns_) {
    auto cd_it =
        std::find_if(cds.begin(), cds.end(), [&added_column](const ColumnDescriptor* cd) {
          return added_column.columnId == cd->columnId;
        });
    if (cd_it != cds.end()) {
      added_columns_to_drop.emplace_back(*cd_it);
    }
  }
  for (const auto& cd : added_columns_to_drop) {
    ChunkKey col_key{catalog.getCurrentDB().dbId, td->tableId, cd->columnId};
    catalog.dropColumnTransactional(*td, *cd);
    auto& data_mgr = catalog.getDataMgr();
    data_mgr.deleteChunksWithPrefix(col_key, MemoryLevel::GPU_LEVEL);
    data_mgr.deleteChunksWithPrefix(col_key, MemoryLevel::CPU_LEVEL);
  }

  // Rename any columns back to original name
  for (auto& renamed_column : renamed_columns_) {
    auto cd_it = std::find_if(
        cds.begin(), cds.end(), [&renamed_column](const ColumnDescriptor* cd) {
          return renamed_column.columnId == cd->columnId;
        });
    if (cd_it != cds.end()) {
      auto cd = *cd_it;
      auto old_name = renamed_column.columnName;
      if (cd->columnName != old_name) {
        catalog.renameColumn(td, cd, old_name);
      }
    }
  }

  // Remove any added dictionary
  for (auto& added_dict : updated_dict_cds_) {
    auto cd_it =
        std::find_if(cds.begin(), cds.end(), [&added_dict](const ColumnDescriptor* cd) {
          return added_dict.columnId == cd->columnId;
        });
    if (cd_it != cds.end()) {
      auto cd = *cd_it;

      // Find all dictionaries, delete dictionaries which are defunct
      auto dds = catalog.getAllDictionariesWithColumnInName(cd);
      for (const auto& dd : dds) {
        if (!added_dict.columnType.is_dict_encoded_type() ||
            dd->dictRef.dictId != added_dict.columnType.get_comp_param()) {
          auto temp_cd = *cd;
          temp_cd.columnType.set_comp_param(dd->dictRef.dictId);
          catalog.delDictionaryTransactional(temp_cd);
        }
      }
    }
  }

  // Undo any altered column
  for (auto& altered_column : altered_columns_) {
    auto cd_it = std::find_if(
        cds.begin(), cds.end(), [&altered_column](const ColumnDescriptor* cd) {
          return altered_column.new_cd.columnId == cd->columnId;
        });
    if (cd_it != cds.end()) {
      catalog.alterColumnTypeTransactional(altered_column.old_cd);
    }
  }
}

void AlterTableAlterColumnCommand::alterColumnTypes(
    const TableDescriptor* td,
    const AlterColumnTypePairs& src_dst_cds) {
  auto& catalog = session_ptr_->getCatalog();

  // Store information necessary to rollback changes for both data and catalog
  auto table_epochs = catalog.getTableEpochs(catalog.getDatabaseId(), td->tableId);
  collectExpectedCatalogChanges(td, src_dst_cds);

  try {
    auto physical_columns = prepareGeoColumns(td, src_dst_cds);
    auto geo_src_dst_column_pairs =
        get_alter_column_geo_pairs_from_src_dst_pairs_phys_cds(src_dst_cds,
                                                               physical_columns);

    auto non_geo_cds = prepareColumns(td, src_dst_cds);

    alterGeoColumnData(td, geo_src_dst_column_pairs);
    alterNonGeoColumnData(td, non_geo_cds);

    alterColumns(td, src_dst_cds);

    // First checkpoint is for added/altered data, rollback is possible
    checkpoint(td, src_dst_cds);

  } catch (std::exception& except) {
    catalog.setTableEpochs(catalog.getDatabaseId(), table_epochs);
    clearInMemoryData(td, src_dst_cds);
    rollback(td, src_dst_cds);
    throw std::runtime_error("Alter column type: " + std::string(except.what()));
  }

  // After the last checkpoint, the following operations are non-reversible,
  // when recovering from a crash will be required to finish
  try {
    try {
      deleteDictionaries(td, src_dst_cds);
    } catch (std::exception& except) {
      LOG(WARNING) << "Alter column type: failed to clear source dictionaries: "
                   << except.what();
      throw;
    }

    try {
      clearRemainingChunks(td, src_dst_cds);
    } catch (std::exception& except) {
      LOG(WARNING) << "Alter column type: failed to clear remaining chunks: "
                   << except.what();
      throw;
    }

    try {
      dropSourceGeoColumns(td, src_dst_cds);
    } catch (std::exception& except) {
      LOG(WARNING) << "Alter column type: failed to remove geo's source column : "
                   << except.what();
      throw;
    }

    // Second checkpoint is for removed data, rollback no longer possible
    checkpoint(td, src_dst_cds);
    catalog.resetTableEpochFloor(td->tableId);

  } catch (std::exception& except) {
    // Any exception encountered during the last steps is unexpected and will cause a
    // crash, relying on the recovery process to fix resulting issues
    LOG(FATAL) << "Alter column type: encountered fatal error during finalizing: "
               << except.what();
  }

  clearInMemoryData(td, src_dst_cds);
}

void AlterTableAlterColumnCommand::alterColumn() {
  auto& ddl_payload = extractPayload(ddl_data_);
  const auto tableName = std::string(ddl_payload["tableName"].GetString());

  auto columns = Parser::get_columns_from_json_payload("alterData", ddl_payload);

  auto& catalog = session_ptr_->getCatalog();
  const auto td_with_lock =
      lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
          catalog, tableName, true);
  const auto td = td_with_lock();

  if (!td) {
    throw std::runtime_error("Table " + tableName + " does not exist.");
  } else {
    if (td->isView) {
      throw std::runtime_error("Altering columns in a view is not supported.");
    }
    validate_table_type(td, ddl_utils::TableType::TABLE, "ALTER");
    if (table_is_temporary(td)) {
      throw std::runtime_error("Altering columns in temporary tables is not supported.");
    }
  }

  Parser::check_alter_table_privilege(*session_ptr_, td);

  for (const auto& coldef : columns) {
    const auto& column_name = *coldef.get_column_name();
    if (catalog.getMetadataForColumn(td->tableId, column_name) == nullptr) {
      throw std::runtime_error("Column " + column_name + " does not exist.");
    }
  }

  CHECK(td->fragmenter);
  if (td->sortedColumnId) {
    throw std::runtime_error(
        "Altering columns to a table is not supported when using the \"sort_column\" "
        "option.");
  }

  auto [src_cds, dst_cds] = get_alter_column_src_dst_cds(columns, catalog, td);
  alterColumnTypes(td, get_alter_column_pairs_from_src_dst_cds(src_cds, dst_cds));
}

ExecutionResult AlterTableAlterColumnCommand::execute(bool read_only_mode) {
  if (g_cluster) {
    throw std::runtime_error(
        "ALTER TABLE ALTER COLUMN is unsupported in distributed mode.");
  }

  // NOTE: read_only_mode is validated at a higher level in AlterTableCommand

  // TODO: Refactor this lock when refactoring other ALTER TABLE commands
  const auto execute_read_lock =
      heavyai::shared_lock<legacylockmgr::WrapperType<heavyai::shared_mutex>>(
          *legacylockmgr::LockMgr<heavyai::shared_mutex, bool>::getMutex(
              legacylockmgr::ExecutorOuterLock, true));

  // There are a few major cases to consider when alter column is invoked.
  // Below non variable length column is abbreviated as NVL and a variable
  // length column is abbreviated as VL.
  //
  // 1. A NVL -> NVL or VL -> VL column conversion.
  //
  // 2. A NVL -> VL or VL -> NVL column conversion.
  //
  // 3. A VL/NVL column converted to a Geo column.
  //
  // Case (1) is the simplest since chunks do not change their chunk keys.
  //
  // Case (2) requires that the chunk keys are added or removed (typically the
  // index chunk), and this requires special treatment.
  //
  // Case (3) requires temporarily renaming the source column, creating new Geo
  // columns, populating the destination Geo columns and dropping the source
  // column.

  alterColumn();
  return {};
}

ExecutionResult AlterTableCommand::execute(bool read_only_mode) {
  if (read_only_mode) {
    throw std::runtime_error("ALTER TABLE invalid in read only mode.");
  }

  auto& ddl_payload = extractPayload(ddl_data_);
  const auto tableName = std::string(ddl_payload["tableName"].GetString());

  CHECK(ddl_payload.HasMember("alterType"));
  auto type = json_str(ddl_payload["alterType"]);

  if (type == "RENAME_TABLE") {
    CHECK(ddl_payload.HasMember("newTableName"));
    auto newTableName = json_str(ddl_payload["newTableName"]);
    std::unique_ptr<Parser::DDLStmt>(
        new Parser::RenameTableStmt(new std::string(tableName),
                                    new std::string(newTableName)))
        ->execute(*session_ptr_, read_only_mode);
    return {};

  } else if (type == "RENAME_COLUMN") {
    CHECK(ddl_payload.HasMember("columnName"));
    auto columnName = json_str(ddl_payload["columnName"]);
    CHECK(ddl_payload.HasMember("newColumnName"));
    auto newColumnName = json_str(ddl_payload["newColumnName"]);
    std::unique_ptr<Parser::DDLStmt>(
        new Parser::RenameColumnStmt(new std::string(tableName),
                                     new std::string(columnName),
                                     new std::string(newColumnName)))
        ->execute(*session_ptr_, read_only_mode);
    return {};

  } else if (type == "ALTER_COLUMN") {
    return AlterTableAlterColumnCommand{ddl_data_, session_ptr_}.execute(read_only_mode);
  } else if (type == "ADD_COLUMN") {
    CHECK(ddl_payload.HasMember("columnData"));
    CHECK(ddl_payload["columnData"].IsArray());

    // New Columns go into this list
    std::list<Parser::ColumnDef*>* table_element_list_ =
        new std::list<Parser::ColumnDef*>;

    const auto elements = ddl_payload["columnData"].GetArray();
    for (const auto& element : elements) {
      CHECK(element.IsObject());
      CHECK(element.HasMember("type"));
      if (json_str(element["type"]) == "SQL_COLUMN_DECLARATION") {
        auto col_def = Parser::column_from_json(element);
        table_element_list_->emplace_back(col_def.release());
      } else {
        LOG(FATAL) << "Unsupported element type for ALTER TABLE: "
                   << element["type"].GetString();
      }
    }

    std::unique_ptr<Parser::DDLStmt>(
        new Parser::AddColumnStmt(new std::string(tableName), table_element_list_))
        ->execute(*session_ptr_, read_only_mode);
    return {};

  } else if (type == "DROP_COLUMN") {
    CHECK(ddl_payload.HasMember("columnData"));
    auto columnData = json_str(ddl_payload["columnData"]);
    // Convert columnData to std::list<std::string*>*
    //    allocate std::list<> as DropColumnStmt will delete it;
    std::list<std::string*>* cols = new std::list<std::string*>;
    std::vector<std::string> cols1;
    boost::split(cols1, columnData, boost::is_any_of(","));
    for (auto s : cols1) {
      // strip leading/trailing spaces/quotes/single quotes
      boost::algorithm::trim_if(s, boost::is_any_of(" \"'`"));
      std::string* str = new std::string(s);
      cols->emplace_back(str);
    }

    std::unique_ptr<Parser::DDLStmt>(
        new Parser::DropColumnStmt(new std::string(tableName), cols))
        ->execute(*session_ptr_, read_only_mode);
    return {};

  } else if (type == "ALTER_OPTIONS") {
    CHECK(ddl_payload.HasMember("options"));
    const auto& options = ddl_payload["options"];
    if (options.IsObject()) {
      for (auto itr = options.MemberBegin(); itr != options.MemberEnd(); ++itr) {
        std::string* option_name = new std::string(json_str(itr->name));
        Parser::Literal* literal_value;
        if (itr->value.IsString()) {
          std::string literal_string = json_str(itr->value);

          // iff this string can be converted to INT
          //   ... do so because it is necessary for AlterTableParamStmt
          std::size_t sz;
          int iVal = std::stoi(literal_string, &sz);
          if (sz == literal_string.size()) {
            literal_value = new Parser::IntLiteral(iVal);
          } else {
            literal_value = new Parser::StringLiteral(&literal_string);
          }
        } else if (itr->value.IsInt() || itr->value.IsInt64()) {
          literal_value = new Parser::IntLiteral(json_i64(itr->value));
        } else if (itr->value.IsNull()) {
          literal_value = new Parser::NullLiteral();
        } else {
          throw std::runtime_error("Unable to handle literal for " + *option_name);
        }
        CHECK(literal_value);
        Parser::NameValueAssign* nv =
            new Parser::NameValueAssign(option_name, literal_value);
        std::unique_ptr<Parser::DDLStmt>(
            new Parser::AlterTableParamStmt(new std::string(tableName), nv))
            ->execute(*session_ptr_, read_only_mode);
        return {};
      }
    } else {
      CHECK(options.IsNull());
    }
  }

  return ExecutionResult();
}

AlterForeignTableCommand::AlterForeignTableCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<const Catalog_Namespace::SessionInfo> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  if (!g_enable_fsi) {
    throw std::runtime_error("Unsupported command: ALTER FOREIGN TABLE");
  }
  auto& ddl_payload = extractPayload(ddl_data_);
  CHECK(ddl_payload.HasMember("tableName"));
  CHECK(ddl_payload["tableName"].IsString());
  CHECK(ddl_payload.HasMember("alterType"));
  CHECK(ddl_payload["alterType"].IsString());
  if (ddl_payload["alterType"] == "RENAME_TABLE") {
    CHECK(ddl_payload.HasMember("newTableName"));
    CHECK(ddl_payload["newTableName"].IsString());
  } else if (ddl_payload["alterType"] == "RENAME_COLUMN") {
    CHECK(ddl_payload.HasMember("oldColumnName"));
    CHECK(ddl_payload["oldColumnName"].IsString());
    CHECK(ddl_payload.HasMember("newColumnName"));
    CHECK(ddl_payload["newColumnName"].IsString());
  } else if (ddl_payload["alterType"] == "ALTER_OPTIONS") {
    CHECK(ddl_payload.HasMember("options"));
    CHECK(ddl_payload["options"].IsObject());
  } else {
    UNREACHABLE() << "Not a valid alter foreign table command: "
                  << ddl_payload["alterType"].GetString();
  }
}

ExecutionResult AlterForeignTableCommand::execute(bool read_only_mode) {
  if (read_only_mode) {
    throw std::runtime_error("ALTER FOREIGN TABLE invalid in read only mode.");
  }

  auto& ddl_payload = extractPayload(ddl_data_);
  std::string alter_type = ddl_payload["alterType"].GetString();

  // We only need a write lock if we are renaming a table, otherwise a read lock will do.
  heavyai::unique_lock<legacylockmgr::WrapperType<heavyai::shared_mutex>> write_lock;
  heavyai::shared_lock<legacylockmgr::WrapperType<heavyai::shared_mutex>> read_lock;
  if (alter_type == "RENAME_TABLE") {
    write_lock = legacylockmgr::getExecuteWriteLock();
  } else {
    read_lock = legacylockmgr::getExecuteReadLock();
  }

  auto& catalog = session_ptr_->getCatalog();
  const std::string& table_name = ddl_payload["tableName"].GetString();
  auto [td, td_with_lock] =
      get_table_descriptor_with_lock<lockmgr::WriteLock>(catalog, table_name, false);

  ddl_utils::validate_table_type(td, ddl_utils::TableType::FOREIGN_TABLE, "ALTER");

  if (!session_ptr_->checkDBAccessPrivileges(
          DBObjectType::TableDBObjectType, AccessPrivileges::ALTER_TABLE, table_name)) {
    throw std::runtime_error(
        "Current user does not have the privilege to alter foreign table: " + table_name);
  }

  auto table_data_write_lock =
      lockmgr::TableDataLockMgr::getWriteLockForTable(catalog, table_name);
  auto foreign_table = dynamic_cast<const foreign_storage::ForeignTable*>(td);
  CHECK(foreign_table);

  // std::string alter_type = ddl_payload["alterType"].GetString();
  if (alter_type == "RENAME_TABLE") {
    renameTable(foreign_table);
  } else if (alter_type == "RENAME_COLUMN") {
    renameColumn(foreign_table);
  } else if (alter_type == "ALTER_OPTIONS") {
    alterOptions(*foreign_table);
  }

  return ExecutionResult();
}

void AlterForeignTableCommand::renameTable(
    const foreign_storage::ForeignTable* foreign_table) {
  auto& ddl_payload = extractPayload(ddl_data_);
  auto& cat = session_ptr_->getCatalog();
  const std::string& table_name = ddl_payload["tableName"].GetString();
  const std::string& new_table_name = ddl_payload["newTableName"].GetString();
  if (cat.getForeignTable(new_table_name)) {
    throw std::runtime_error("Foreign table with name \"" + table_name +
                             "\" can not be renamed to \"" + new_table_name + "\". " +
                             "A different table with name \"" + new_table_name +
                             "\" already exists.");
  }
  cat.renameTable(foreign_table, new_table_name);
}

void AlterForeignTableCommand::renameColumn(
    const foreign_storage::ForeignTable* foreign_table) {
  auto& ddl_payload = extractPayload(ddl_data_);
  auto& cat = session_ptr_->getCatalog();
  const std::string& old_column_name = ddl_payload["oldColumnName"].GetString();
  const std::string& new_column_name = ddl_payload["newColumnName"].GetString();
  auto column = cat.getMetadataForColumn(foreign_table->tableId, old_column_name);
  if (!column) {
    throw std::runtime_error("Column with name \"" + old_column_name +
                             "\" can not be renamed to \"" + new_column_name + "\". " +
                             "Column with name \"" + old_column_name +
                             "\" does not exist.");
  }
  if (cat.getMetadataForColumn(foreign_table->tableId, new_column_name)) {
    throw std::runtime_error("Column with name \"" + old_column_name +
                             "\" can not be renamed to \"" + new_column_name + "\". " +
                             "A column with name \"" + new_column_name +
                             "\" already exists.");
  }
  cat.renameColumn(foreign_table, column, new_column_name);
}

void AlterForeignTableCommand::alterOptions(
    const foreign_storage::ForeignTable& foreign_table) {
  auto& ddl_payload = extractPayload(ddl_data_);
  const std::string& table_name = ddl_payload["tableName"].GetString();
  auto& cat = session_ptr_->getCatalog();
  auto new_options_map =
      foreign_storage::ForeignTable::createOptionsMap(ddl_payload["options"]);
  foreign_table.validateSupportedOptionKeys(new_options_map);
  foreign_table.validateAlterOptions(new_options_map);
  cat.setForeignTableOptions(table_name, new_options_map, false);
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

ExecutionResult ShowDiskCacheUsageCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  // valid in read_only_mode

  auto cat_ptr = session_ptr_->get_catalog_ptr();
  auto table_names = getFilteredTableNames();

  const auto disk_cache = cat_ptr->getDataMgr().getPersistentStorageMgr()->getDiskCache();
  if (!disk_cache) {
    throw std::runtime_error{"Disk cache not enabled.  Cannot show disk cache usage."};
  }

  // label_infos -> column labels
  std::vector<std::string> labels{"table name", "current cache size"};
  std::vector<TargetMetaInfo> label_infos;
  label_infos.emplace_back(labels[0], SQLTypeInfo(kTEXT, true));
  label_infos.emplace_back(labels[1], SQLTypeInfo(kBIGINT, true));

  std::vector<RelLogicalValues::RowValues> logical_values;

  exec_for_tables_which_exist(
      table_names,
      cat_ptr.get(),
      [&logical_values, &disk_cache, &cat_ptr](const TableDescriptor* td,
                                               const std::string& table_name) {
        auto table_cache_size =
            disk_cache->getSpaceReservedByTable(cat_ptr->getDatabaseId(), td->tableId);
        // logical_values -> table data
        logical_values.emplace_back(RelLogicalValues::RowValues{});
        logical_values.back().emplace_back(genLiteralStr(table_name));
        logical_values.back().emplace_back(genLiteralBigInt(table_cache_size));
      });

  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
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
  CHECK(ddl_payload.HasMember("all"));
  CHECK(ddl_payload["all"].IsBool());
}

ExecutionResult ShowUserDetailsCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  // valid in read_only_mode

  auto& ddl_payload = extractPayload(ddl_data_);
  auto& sys_cat = Catalog_Namespace::SysCatalog::instance();

  Catalog_Namespace::UserMetadata self = session_ptr_->get_currentUser();
  bool all = ddl_payload.HasMember("all") ? ddl_payload["all"].GetBool() : false;
  if (all && !self.isSuper) {
    throw std::runtime_error(
        "SHOW ALL USER DETAILS is only available to superusers. (Try SHOW USER "
        "DETAILS instead?)");
  }

  // label_infos -> column labels
  std::vector<TargetMetaInfo> label_infos;
  label_infos.emplace_back("NAME", SQLTypeInfo(kTEXT, true));
  label_infos.emplace_back("ID", SQLTypeInfo(kBIGINT, true));
  if (all) {
    label_infos.emplace_back("IS_SUPER", SQLTypeInfo(kBOOLEAN, true));
  }
  label_infos.emplace_back("DEFAULT_DB", SQLTypeInfo(kTEXT, true));
  if (self.isSuper) {
    label_infos.emplace_back("CAN_LOGIN", SQLTypeInfo(kBOOLEAN, true));
  }
  std::vector<RelLogicalValues::RowValues> logical_values;

  auto cat = session_ptr_->get_catalog_ptr();
  DBObject dbObject(cat->name(), DatabaseDBObjectType);
  dbObject.loadKey();
  dbObject.setPrivileges(AccessPrivileges::ACCESS);

  std::map<std::string, Catalog_Namespace::UserMetadata> user_map;
  auto user_list = !all ? sys_cat.getAllUserMetadata(cat->getDatabaseId())
                        : sys_cat.getAllUserMetadata();
  for (auto& user : user_list) {
    if (user.can_login || self.isSuper) {  // hide users who have disabled accounts
      user_map[user.userName] = user;
    }
  }

  if (ddl_payload.HasMember("userNames")) {
    std::map<std::string, Catalog_Namespace::UserMetadata> user_map2;
    for (const auto& user_name_json : ddl_payload["userNames"].GetArray()) {
      std::string user_name = user_name_json.GetString();
      auto uit = user_map.find(user_name);
      if (uit == user_map.end()) {
        throw std::runtime_error("User \"" + user_name + "\" not found. ");
      }
      user_map2[uit->first] = uit->second;
    }
    user_map = user_map2;
  }

  Catalog_Namespace::DBSummaryList dbsums = sys_cat.getDatabaseListForUser(self);
  std::unordered_set<std::string> visible_databases;
  if (!self.isSuper) {
    for (const auto& dbsum : dbsums) {
      visible_databases.insert(dbsum.dbName);
    }
  }

  for (const auto& [user_name, user] : user_map) {
    // database
    std::string dbname;
    Catalog_Namespace::DBMetadata db;
    if (sys_cat.getMetadataForDBById(user.defaultDbId, db)) {
      if (self.isSuper || visible_databases.count(db.dbName)) {
        dbname = db.dbName;
      }
    }
    if (self.isSuper) {
      dbname += "(" + std::to_string(user.defaultDbId) + ")";
    }

    // logical_values -> table data
    logical_values.emplace_back(RelLogicalValues::RowValues{});
    logical_values.back().emplace_back(genLiteralStr(user.userName));
    logical_values.back().emplace_back(genLiteralBigInt(user.userId));
    if (all) {
      logical_values.back().emplace_back(genLiteralBoolean(user.isSuper));
    }
    logical_values.back().emplace_back(genLiteralStr(dbname));
    if (self.isSuper) {
      logical_values.back().emplace_back(genLiteralBoolean(user.can_login));
    }
  }

  // Create ResultSet
  CHECK_EQ(logical_values.size(), user_map.size());
  if (logical_values.size() >= 1U) {
    CHECK_EQ(logical_values[0].size(), label_infos.size());
  }
  std::shared_ptr<ResultSet> rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

ShowRolesCommand::ShowRolesCommand(
    const DdlCommandData& ddl_data,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr)
    : DdlCommand(ddl_data, session_ptr) {
  auto& ddl_payload = extractPayload(ddl_data);
  CHECK(ddl_payload["userName"].IsString());
  CHECK(ddl_payload["effective"].IsBool());
}

ExecutionResult ShowRolesCommand::execute(bool read_only_mode) {
  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  // valid in read_only_mode

  auto& ddl_payload = extractPayload(ddl_data_);
  auto& sys_cat = Catalog_Namespace::SysCatalog::instance();

  // label_infos -> column labels
  std::vector<TargetMetaInfo> label_infos;
  std::vector<std::string> labels{"ROLES"};
  label_infos.emplace_back(labels[0], SQLTypeInfo(kTEXT, true));

  // logical_values -> table data
  std::vector<RelLogicalValues::RowValues> logical_values;
  std::vector<std::string> roles_list;
  Catalog_Namespace::UserMetadata self = session_ptr_->get_currentUser();
  std::string user_name = ddl_payload["userName"].GetString();
  bool effective = ddl_payload["effective"].GetBool();
  if (user_name.empty()) {
    user_name = self.userName;
  }
  Catalog_Namespace::UserMetadata user;
  bool is_user = sys_cat.getMetadataForUser(user_name, user);
  if (!self.isSuper) {
    if (is_user) {
      if (self.userId != user.userId) {
        throw std::runtime_error(
            "Only a superuser is authorized to request list of roles granted to another "
            "user.");
      }
    } else {
      if (!sys_cat.isRoleGrantedToGrantee(
              self.userName, user_name, /*only_direct=*/false)) {
        throw std::runtime_error(
            "Only a superuser is authorized to request list of roles granted to a role "
            "they don't have.");
      }
    }
  }
  if (user.isSuper) {
    auto s = sys_cat.getCreatedRoles();
    roles_list.insert(roles_list.end(), s.begin(), s.end());
  } else {
    roles_list = sys_cat.getRoles(user_name, effective);
  }
  for (const std::string& role_name : roles_list) {
    logical_values.emplace_back(RelLogicalValues::RowValues{});
    logical_values.back().emplace_back(genLiteralStr(role_name));
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
  CHECK(ddl_payload.HasMember("all"));
  CHECK(ddl_payload["all"].IsBool());
  CHECK(ddl_payload.HasMember("oldOwners"));
  CHECK(ddl_payload["oldOwners"].IsArray());
  for (const auto& old_owner : ddl_payload["oldOwners"].GetArray()) {
    CHECK(old_owner.IsString());
    old_owners_.emplace(old_owner.GetString());
  }
  CHECK(ddl_payload.HasMember("newOwner"));
  CHECK(ddl_payload["newOwner"].IsString());
  new_owner_ = ddl_payload["newOwner"].GetString();
  all_ = ddl_payload["all"].GetBool();
}

ExecutionResult ReassignOwnedCommand::execute(bool read_only_mode) {
  auto execute_write_lock = legacylockmgr::getExecuteWriteLock();

  if (read_only_mode) {
    throw std::runtime_error("REASSIGN OWNER invalid in read only mode.");
  }
  if (!session_ptr_->get_currentUser().isSuper) {
    throw std::runtime_error{
        "Only super users can reassign ownership of database objects."};
  }
  if (all_) {
    auto catalogs = Catalog_Namespace::SysCatalog::instance().getCatalogsForAllDbs();
    for (auto& catalog : catalogs) {
      catalog->reassignOwners(old_owners_, new_owner_);
    }
  } else {
    const auto catalog = session_ptr_->get_catalog_ptr();
    catalog->reassignOwners(old_owners_, new_owner_);
  }
  return ExecutionResult();
}
