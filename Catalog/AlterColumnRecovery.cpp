/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#include "AlterColumnRecovery.h"

#include <filesystem>

#include "Catalog/SysCatalog.h"
#include "QueryEngine/ExternalCacheInvalidators.h"
#include "Shared/JsonUtils.h"
#include "Utils/DdlUtils.h"

AlterTableAlterColumnCommandRecoveryMgr::AlterTableAlterColumnCommandRecoveryMgr(
    Catalog_Namespace::Catalog& catalog)
    : catalog_(catalog) {}

void AlterTableAlterColumnCommandRecoveryMgr::rollback(const TableDescriptor* td,
                                                       const RecoveryInfo& param) {
  auto cds = catalog_.getAllColumnMetadataForTable(td->tableId, false, false, true);

  // Drop any columns that were added
  for (auto& added_column : param.added_columns) {
    auto cd_it =
        std::find_if(cds.begin(), cds.end(), [&added_column](const ColumnDescriptor* cd) {
          return added_column.columnId == cd->columnId;
        });
    auto cd = *cd_it;
    ChunkKey col_key{catalog_.getCurrentDB().dbId, td->tableId, cd->columnId};
    catalog_.dropColumnTransactional(*td, *cd);
    auto& data_mgr = catalog_.getDataMgr();
    data_mgr.deleteChunksWithPrefix(col_key, MemoryLevel::GPU_LEVEL);
    data_mgr.deleteChunksWithPrefix(col_key, MemoryLevel::CPU_LEVEL);
    cds.erase(cd_it);
  }

  // Rename any columns back to original name
  for (auto& renamed_column : param.renamed_columns) {
    auto cd_it = std::find_if(
        cds.begin(), cds.end(), [&renamed_column](const ColumnDescriptor* cd) {
          return renamed_column.columnId == cd->columnId;
        });
    if (cd_it != cds.end()) {
      auto cd = *cd_it;
      auto old_name = renamed_column.columnName;
      if (cd->columnName != old_name) {
        catalog_.renameColumn(td, cd, old_name);
      }
    }
  }

  // Remove any added dictionary
  for (auto& added_dict : param.updated_dict_cds) {
    auto cd_it =
        std::find_if(cds.begin(), cds.end(), [&added_dict](const ColumnDescriptor* cd) {
          return added_dict.columnId == cd->columnId;
        });
    if (cd_it != cds.end()) {
      auto cd = *cd_it;

      // Find all dictionaries, delete dictionaries which are defunct
      auto dds = catalog_.getAllDictionariesWithColumnInName(cd);
      for (const auto& dd : dds) {
        if (!added_dict.columnType.is_dict_encoded_type() ||
            dd->dictRef.dictId != added_dict.columnType.get_comp_param()) {
          auto temp_cd = *cd;
          temp_cd.columnType.set_comp_param(dd->dictRef.dictId);
          temp_cd.columnType.setStringDictKey(
              {catalog_.getDatabaseId(), dd->dictRef.dictId});
          catalog_.delDictionaryTransactional(temp_cd);
        }
      }
    }
  }

  // Undo any altered column
  for (auto& altered_column : param.altered_columns) {
    auto cd_it = std::find_if(
        cds.begin(), cds.end(), [&altered_column](const ColumnDescriptor* cd) {
          return altered_column.new_cd.columnId == cd->columnId;
        });
    if (cd_it != cds.end()) {
      catalog_.alterColumnTypeTransactional(altered_column.old_cd);
    }
  }
}

void AlterTableAlterColumnCommandRecoveryMgr::cleanupDeleteDictionaries(
    const TableDescriptor* td,
    const TypePairs& src_dst_cds) {
  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (!src_cd->columnType.is_dict_encoded_type()) {
      continue;
    }
    if (!ddl_utils::alter_column_utils::compare_column_descriptors(src_cd, dst_cd)
             .sql_types_match) {
      catalog_.delDictionaryTransactional(*src_cd);
    }
  }
}

void AlterTableAlterColumnCommandRecoveryMgr::cleanupClearChunk(
    const ChunkKey& key,
    const MemoryLevel mem_level) {
  auto& data_mgr = catalog_.getDataMgr();
  if (mem_level >= data_mgr.levelSizes_.size()) {
    return;
  }
  for (int device = 0; device < data_mgr.levelSizes_[mem_level]; ++device) {
    if (data_mgr.isBufferOnDevice(key, mem_level, device)) {
      data_mgr.deleteChunk(key, mem_level, device);
    }
  }
}

void AlterTableAlterColumnCommandRecoveryMgr::cleanupClearChunk(const ChunkKey& key) {
  cleanupClearChunk(key, MemoryLevel::GPU_LEVEL);
  cleanupClearChunk(key, MemoryLevel::CPU_LEVEL);
  cleanupClearChunk(key, MemoryLevel::DISK_LEVEL);
}

void AlterTableAlterColumnCommandRecoveryMgr::cleanupClearRemainingChunks(
    const TableDescriptor* td,
    const TypePairs& src_dst_cds) {
  // for (non-geo) cases where the chunk keys change, chunks that remain with old chunk
  // key must be removed
  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (src_cd->columnType.is_varlen_indeed() != dst_cd->columnType.is_varlen_indeed()) {
      ChunkMetadataVector chunk_metadata;
      catalog_.getDataMgr().getChunkMetadataVecForKeyPrefix(
          chunk_metadata, {catalog_.getDatabaseId(), td->tableId, dst_cd->columnId});
      std::set<int> fragment_ids;
      for (const auto& [key, _] : chunk_metadata) {
        fragment_ids.insert(key[CHUNK_KEY_FRAGMENT_IDX]);
      }
      for (const auto& frag_id : fragment_ids) {
        ChunkKey key = {catalog_.getDatabaseId(), td->tableId, src_cd->columnId, frag_id};
        if (src_cd->columnType.is_varlen_indeed()) {
          auto data_key = key;
          data_key.push_back(1);
          cleanupClearChunk(data_key);
          auto index_key = key;
          index_key.push_back(2);
          cleanupClearChunk(index_key);
        } else {  // no varlen case
          cleanupClearChunk(key);
        }
      }
    }
  }
}

void AlterTableAlterColumnCommandRecoveryMgr::cleanupDropSourceGeoColumns(
    const TableDescriptor* td,
    const TypePairs& src_dst_cds) {
  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (!dst_cd->columnType.is_geometry()) {
      continue;
    }
    auto catalog_cd = catalog_.getMetadataForColumn(src_cd->tableId, src_cd->columnId);
    catalog_.dropColumnTransactional(*td, *catalog_cd);
    ChunkKey col_key{catalog_.getCurrentDB().dbId, td->tableId, src_cd->columnId};
    auto& data_mgr = catalog_.getDataMgr();
    data_mgr.deleteChunksWithPrefix(col_key, MemoryLevel::GPU_LEVEL);
    data_mgr.deleteChunksWithPrefix(col_key, MemoryLevel::CPU_LEVEL);
    data_mgr.deleteChunksWithPrefix(col_key, MemoryLevel::DISK_LEVEL);
  }
}

std::list<AlterTableAlterColumnCommandRecoveryMgr::ColumnAltered>
AlterTableAlterColumnCommandRecoveryMgr::fromPairedCds(
    const std::list<std::pair<ColumnDescriptor, ColumnDescriptor>>& altered_columns) {
  std::list<ColumnAltered> retval;
  for (const auto& [old_cd, new_cd] : altered_columns) {
    retval.emplace_back(old_cd, new_cd);
  }
  return retval;
}

std::list<std::pair<ColumnDescriptor, ColumnDescriptor>>
AlterTableAlterColumnCommandRecoveryMgr::toPairedCds(
    const std::list<ColumnAltered>& altered_columns) {
  std::list<std::pair<ColumnDescriptor, ColumnDescriptor>> retval;
  for (const auto& [old_cd, new_cd] : altered_columns) {
    retval.emplace_back(old_cd, new_cd);
  }
  return retval;
}

AlterTableAlterColumnCommandRecoveryMgr::RecoveryInfo
AlterTableAlterColumnCommandRecoveryMgr::deserializeRecoveryInformation(
    const std::string& filename) {
  RecoveryInfo param;
  auto d = json_utils::read_from_file(filename);
  CHECK(d.IsObject());

  json_utils::get_value_from_object(d, param.added_columns, "added_columns");

  std::list<std::pair<ColumnDescriptor, ColumnDescriptor>> altered_column_pairs;
  json_utils::get_value_from_object(d, altered_column_pairs, "altered_columns");
  param.altered_columns = fromPairedCds(altered_column_pairs);

  json_utils::get_value_from_object(d, param.renamed_columns, "renamed_columns");

  json_utils::get_value_from_object(d, param.updated_dict_cds, "updated_dict_cds");

  json_utils::get_value_from_object(d, param.table_epoch, "table_epoch");

  json_utils::get_value_from_object(d, param.src_dst_cds, "src_dst_cds");

  return param;
}

std::filesystem::path AlterTableAlterColumnCommandRecoveryMgr::getRecoveryPrefix(
    const std::string& base_path) {
  return std::filesystem::path(base_path) / std::filesystem::path(kRecoveryDirectoryName);
}

std::string AlterTableAlterColumnCommandRecoveryMgr::recoveryFilepath(
    const RecoveryParamFilepathInfo& filepath_info) {
  auto prefix = getRecoveryPrefix(filepath_info.base_path);

  if (filepath_info.table_name.empty()) {
    return prefix.string();
  }

  return (prefix /
          std::filesystem::path("alter_column_recovery_db_" + filepath_info.db_name +
                                "_table_" + filepath_info.table_name + ".json"))
      .string();
}

AlterTableAlterColumnCommandRecoveryMgr::RecoveryParamFilepathInfo
AlterTableAlterColumnCommandRecoveryMgr::getRecoveryFilepathInfo(const int32_t table_id) {
  RecoveryParamFilepathInfo path_info;
  path_info.base_path = catalog_.getCatalogBasePath();
  path_info.db_name = catalog_.getCurrentDB().dbName;
  path_info.table_name =
      table_id >= 0 ? catalog_.getTableName(table_id).value() : std::string{};
  return path_info;
}

void AlterTableAlterColumnCommandRecoveryMgr::readSerializedRecoveryInformation(
    RecoveryInfo& param,
    const RecoveryParamFilepathInfo& filepath_info) {
  auto filename = recoveryFilepath(filepath_info);
  std::ifstream ifs(filename);
  if (!ifs) {
    throw std::runtime_error{"Error trying to read file \"" + filename +
                             "\". The error was: " + std::strerror(errno)};
  }
  std::string json_string;
  ifs >> json_string;
  param = deserializeRecoveryInformation(json_string);
}

std::string AlterTableAlterColumnCommandRecoveryMgr::serializeRecoveryInformation(
    const RecoveryInfo& param) {
  rapidjson::Document d;
  d.SetObject();

  json_utils::add_value_to_object(
      d, param.added_columns, "added_columns", d.GetAllocator());
  json_utils::add_value_to_object(
      d, toPairedCds(param.altered_columns), "altered_columns", d.GetAllocator());
  json_utils::add_value_to_object(
      d, param.renamed_columns, "renamed_columns", d.GetAllocator());
  json_utils::add_value_to_object(
      d, param.updated_dict_cds, "updated_dict_cds", d.GetAllocator());
  json_utils::add_value_to_object(d, param.table_epoch, "table_epoch", d.GetAllocator());

  json_utils::add_value_to_object(d, param.src_dst_cds, "src_dst_cds", d.GetAllocator());

  return json_utils::write_to_string(d);
}

void AlterTableAlterColumnCommandRecoveryMgr::writeSerializedRecoveryInformation(
    const RecoveryInfo& param,
    const RecoveryParamFilepathInfo& filepath_info) {
  auto filename = recoveryFilepath(filepath_info);

  // Create crash recovery directory if non-existent
  auto prefix = getRecoveryPrefix(filepath_info.base_path).string();
  if (!std::filesystem::exists(prefix)) {
    if (!std::filesystem::create_directory(prefix)) {
      throw std::runtime_error{"Error trying to create crash recovery directory \"" +
                               prefix + "\". The error was: " + std::strerror(errno)};
    }
  }

  // Use a temporary file name to indicate file has not been written yet
  std::ofstream ofs(filename + ".tmp");
  if (!ofs) {
    throw std::runtime_error{"Error trying to create file \"" + filename +
                             "\". The error was: " + std::strerror(errno)};
  }
  ofs << serializeRecoveryInformation(param);
  // Rename to target filename to indicate file was successfully written
  std::filesystem::rename(filename + ".tmp", filename);
}

void AlterTableAlterColumnCommandRecoveryMgr::cleanup(const TableDescriptor* td,
                                                      const TypePairs& src_dst_cds) {
  try {
    cleanupDeleteDictionaries(td, src_dst_cds);
  } catch (std::exception& except) {
    LOG(WARNING) << "Alter column type: failed to clear source dictionaries: "
                 << except.what();
    throw;
  }

  try {
    cleanupClearRemainingChunks(td, src_dst_cds);
  } catch (std::exception& except) {
    LOG(WARNING) << "Alter column type: failed to clear remaining chunks: "
                 << except.what();
    throw;
  }

  try {
    cleanupDropSourceGeoColumns(td, src_dst_cds);
  } catch (std::exception& except) {
    LOG(WARNING) << "Alter column type: failed to remove geo's source column : "
                 << except.what();
    throw;
  }

  // Data is removed data, rollback no longer possible
  checkpoint(td, src_dst_cds);
  catalog_.resetTableEpochFloor(td->tableId);
}

void AlterTableAlterColumnCommandRecoveryMgr::checkpoint(const TableDescriptor* td,
                                                         const TypePairs& src_dst_cds) {
  for (auto& [src_cd, dst_cd] : src_dst_cds) {
    if (!dst_cd->columnType.is_dict_encoded_type()) {
      continue;
    }
    if (ddl_utils::alter_column_utils::compare_column_descriptors(src_cd, dst_cd)
            .sql_types_match) {
      continue;
    }
    auto string_dictionary =
        catalog_.getMetadataForDict(dst_cd->columnType.get_comp_param(), true)
            ->stringDict.get();
    if (!string_dictionary->checkpoint()) {
      throw std::runtime_error("Failed to checkpoint dictionary while altering column " +
                               dst_cd->columnName + ".");
    }
  }
  catalog_.checkpointWithAutoRollback(td->tableId);
}

std::list<std::filesystem::path>
AlterTableAlterColumnCommandRecoveryMgr::getRecoveryFiles() {
  std::list<std::filesystem::path> result;
  std::string path = recoveryFilepath(getRecoveryFilepathInfo());

  if (!std::filesystem::exists(path)) {
    return {};
  }

  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    auto entry_path = entry.path().string();
    if (entry_path.find("alter_column_recovery_db_" + catalog_.name() + "_table_") !=
            std::string::npos &&
        entry_path.find(".json") != std::string::npos) {
      if (entry_path.find(".tmp") != std::string::npos &&
          entry_path.find(".tmp") == entry_path.size() - std::string{".tmp"}.size()) {
        if (!std::filesystem::remove(entry_path)) {
          throw std::runtime_error("Failed to remove incomplete recovery file: " +
                                   entry_path);
        } else {
          LOG(INFO) << "Removing incomplete ALTER COLUMN recovery file: " + entry_path;
        }
      } else {
        result.emplace_back(entry.path());
      }
    }
  }
  return result;
}

AlterTableAlterColumnCommandRecoveryMgr::TypePairs
AlterTableAlterColumnCommandRecoveryMgr::getSrcDstCds(
    int table_id,
    std::list<std::pair<ColumnDescriptor, ColumnDescriptor>>& pairs_list) {
  // Source columns must be obtained from catalog to ensure correctness/consistency
  std::list<std::pair<const ColumnDescriptor*, ColumnDescriptor*>> result;
  for (auto& [src, dst] : pairs_list) {
    auto catalog_cd = catalog_.getMetadataForColumn(table_id, src.columnId);
    if (!catalog_cd) {
      // If column is missing in catalog, operate under the assumption it was
      // already successfully removed in cleanup, along with all related
      // components such as dictionaries
      continue;
    }
    result.emplace_back(catalog_cd, &dst);
  }
  return result;
}

void AlterTableAlterColumnCommandRecoveryMgr::recoverAlterTableAlterColumnFromFile(
    const std::string& filename) {
  AlterTableAlterColumnCommandRecoveryMgr::RecoveryInfo recovery_param;

  recovery_param = deserializeRecoveryInformation(filename);

  CHECK_GT(recovery_param.src_dst_cds.size(), 0UL);

  auto table_id = recovery_param.src_dst_cds.begin()->first.tableId;
  auto td = catalog_.getMetadataForTable(table_id, false);
  CHECK(td);
  LOG(INFO) << "Starting crash recovery for table: " << td->tableName;

  auto table_epochs = catalog_.getTableEpochs(catalog_.getDatabaseId(), td->tableId);
  CHECK_GT(table_epochs.size(), 0UL);
  auto current_first_epoch = table_epochs[0].table_epoch;

  if (current_first_epoch == recovery_param.table_epoch) {
    rollback(td, recovery_param);
  } else if (current_first_epoch == recovery_param.table_epoch + 1) {
    auto src_dst_cds = getSrcDstCds(table_id, recovery_param.src_dst_cds);
    try {
      cleanup(td, src_dst_cds);
    } catch (std::exception& except) {
      throw std::runtime_error("Alter column recovery error during cleanup: " +
                               std::string(except.what()));
    }
    ChunkKey table_key{catalog_.getCurrentDB().dbId, td->tableId};
    UpdateTriggeredCacheInvalidator::invalidateCachesByTable(
        boost::hash_value(table_key));
    catalog_.getDataMgr().deleteChunksWithPrefix(table_key, MemoryLevel::GPU_LEVEL);
    catalog_.getDataMgr().deleteChunksWithPrefix(table_key, MemoryLevel::CPU_LEVEL);

  } else {
    CHECK_EQ(current_first_epoch, recovery_param.table_epoch + 2);
    // no-op, last checkpoint reached in processing
  }
  LOG(INFO) << "Completed crash recovery for table: " << td->tableName;
}

void AlterTableAlterColumnCommandRecoveryMgr::resolveIncompleteAlterColumnCommands() {
  auto recovery_files = getRecoveryFiles();
  if (recovery_files.empty()) {
    return;
  }

  LOG(INFO) << "Starting crash recovery for tables in catalog: " << catalog_.name();
  for (const auto& filepath : recovery_files) {
    recoverAlterTableAlterColumnFromFile(filepath.string());
    std::filesystem::remove(filepath);
  }
  LOG(INFO) << "Completed crash recovery for tables in catalog: " << catalog_.name();
}

std::map<std::string, AlterTableAlterColumnCommandRecoveryMgr>
AlterTableAlterColumnCommandRecoveryMgr::createRecoveryManagersForCatalogs() {
  std::map<std::string, AlterTableAlterColumnCommandRecoveryMgr> result;

  auto& syscat = Catalog_Namespace::SysCatalog::instance();
  auto base_path = syscat.getCatalogBasePath();
  auto prefix = getRecoveryPrefix(base_path);
  if (!std::filesystem::exists(prefix)) {
    return {};
  }

  auto catalog_metadata = syscat.getAllDBMetadata();

  for (const auto& entry : std::filesystem::directory_iterator(prefix)) {
    auto entry_path = entry.path().string();

    for (const auto& db_metadata : catalog_metadata) {
      if (result.count(db_metadata.dbName)) {
        continue;
      }
      auto match_db = entry_path.find("alter_column_recovery_db_" + db_metadata.dbName);
      if (match_db == std::string::npos) {
        continue;
      }
      auto catalog = syscat.getCatalog(db_metadata.dbName);
      CHECK(catalog.get());
      result.emplace(db_metadata.dbName, *catalog);
    }
  }

  return result;
}

void AlterTableAlterColumnCommandRecoveryMgr::
    resolveIncompleteAlterColumnCommandsForAllCatalogs() {
  auto recovery_mgrs = createRecoveryManagersForCatalogs();

  for (auto& [dbname, recovery_mgr] : recovery_mgrs) {
    recovery_mgr.resolveIncompleteAlterColumnCommands();
  }
}

namespace json_utils {
// ColumnDescriptor
void set_value(rapidjson::Value& json_val,
               const ColumnDescriptor& column_desc,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetObject();
  auto default_value = column_desc.default_value;
  if (default_value.has_value()) {
    add_value_to_object(json_val, true, "has_default_value", allocator);
    add_value_to_object(
        json_val, default_value.value(), "default_value_literal", allocator);
  } else {
    add_value_to_object(json_val, false, "has_default_value", allocator);
  }
  add_value_to_object(json_val, column_desc.chunks, "chunks", allocator);
  add_value_to_object(json_val, column_desc.columnId, "column_id", allocator);
  add_value_to_object(json_val, column_desc.columnName, "column_name", allocator);
  add_value_to_object(json_val, column_desc.columnType, "column_type", allocator);
  add_value_to_object(json_val, column_desc.db_id, "db_id", allocator);
  add_value_to_object(json_val, column_desc.isDeletedCol, "is_deleted_col", allocator);
  add_value_to_object(json_val, column_desc.isGeoPhyCol, "is_geo_phy_col", allocator);
  add_value_to_object(json_val, column_desc.isSystemCol, "is_system_col", allocator);
  add_value_to_object(json_val, column_desc.isVirtualCol, "is_virtual_col", allocator);
  add_value_to_object(json_val, column_desc.sourceName, "source_name", allocator);
  add_value_to_object(json_val, column_desc.tableId, "table_id", allocator);
  add_value_to_object(json_val, column_desc.virtualExpr, "virtual_expr", allocator);
}

void get_value(const rapidjson::Value& json_val, ColumnDescriptor& column_desc) {
  CHECK(json_val.IsObject());

  bool has_default_value;
  get_value_from_object(json_val, has_default_value, "has_default_value");
  if (has_default_value) {
    std::string default_value;
    get_value_from_object(json_val, default_value, "default_value_literal");
    column_desc.default_value = default_value;
  } else {
    column_desc.default_value = std::nullopt;
  }

  get_value_from_object(json_val, column_desc.chunks, "chunks");
  get_value_from_object(json_val, column_desc.columnId, "column_id");
  get_value_from_object(json_val, column_desc.columnName, "column_name");
  get_value_from_object(json_val, column_desc.db_id, "db_id");
  get_value_from_object(json_val, column_desc.isDeletedCol, "is_deleted_col");
  get_value_from_object(json_val, column_desc.isGeoPhyCol, "is_geo_phy_col");
  get_value_from_object(json_val, column_desc.isSystemCol, "is_system_col");
  get_value_from_object(json_val, column_desc.isVirtualCol, "is_virtual_col");
  get_value_from_object(json_val, column_desc.sourceName, "source_name");
  get_value_from_object(json_val, column_desc.tableId, "table_id");
  get_value_from_object(json_val, column_desc.virtualExpr, "virtual_expr");
  get_value_from_object(json_val, column_desc.columnType, "column_type");
}
}  // namespace json_utils