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

#pragma once

#include "AlterColumnShared.h"
#include "Catalog.h"

class AlterTableAlterColumnCommandRecoveryMgr {
 public:
  AlterTableAlterColumnCommandRecoveryMgr(Catalog_Namespace::Catalog& catalog);

  using TypePairs = alter_column_shared::TypePairs;

  struct ColumnAltered {
    ColumnDescriptor old_cd;
    ColumnDescriptor new_cd;

    ColumnAltered(const ColumnDescriptor& old_cd, const ColumnDescriptor& new_cd)
        : old_cd(old_cd), new_cd(new_cd) {}
  };

  struct RecoveryInfo {
    std::list<ColumnDescriptor> renamed_columns;
    std::list<ColumnDescriptor> added_columns;
    std::list<ColumnAltered> altered_columns;
    std::list<ColumnDescriptor> updated_dict_cds;

    int32_t table_epoch;
    bool is_vacuumed;

    std::list<std::pair<ColumnDescriptor, ColumnDescriptor>> src_dst_cds;
  };

  void rollback(const TableDescriptor* td, const RecoveryInfo& param);

  void cleanup(const TableDescriptor* td, const TypePairs& src_dst_cds);

  void checkpoint(const TableDescriptor* td, const TypePairs& src_dst_cds);

  struct RecoveryParamFilepathInfo {
    std::string base_path;
    std::string db_name;
    std::string table_name;
  };

  RecoveryInfo deserializeRecoveryInformation(const std::string& filename);

  std::string serializeRecoveryInformation(const RecoveryInfo& param);

  void writeSerializedRecoveryInformation(const RecoveryInfo& param,
                                          const RecoveryParamFilepathInfo& filepath_info);

  void readSerializedRecoveryInformation(RecoveryInfo& param,
                                         const RecoveryParamFilepathInfo& filepath_info);

  std::string recoveryFilepath(const RecoveryParamFilepathInfo& filepath_info);

  RecoveryParamFilepathInfo getRecoveryFilepathInfo(const int32_t table_id = -1);

  static void resolveIncompleteAlterColumnCommandsForAllCatalogs();

  inline static const std::string kRecoveryDirectoryName = "crash_recovery";

 private:
  void cleanupDeleteDictionaries(const TableDescriptor* td, const TypePairs& src_dst_cds);

  void cleanupClearChunk(const ChunkKey& key, const MemoryLevel mem_level);

  void cleanupClearChunk(const ChunkKey& key);

  void cleanupClearRemainingChunks(const TableDescriptor* td,
                                   const TypePairs& src_dst_cds);

  void cleanupDropSourceGeoColumns(const TableDescriptor* td,
                                   const TypePairs& src_dst_cds);

  static std::list<ColumnAltered> fromPairedCds(
      const std::list<std::pair<ColumnDescriptor, ColumnDescriptor>>& altered_columns);

  std::list<std::pair<ColumnDescriptor, ColumnDescriptor>> toPairedCds(
      const std::list<ColumnAltered>& altered_columns);

  void recoverAlterTableAlterColumnFromFile(const std::string& filename);

  std::list<std::filesystem::path> getRecoveryFiles();

  TypePairs getSrcDstCds(
      int table_id,
      std::list<std::pair<ColumnDescriptor, ColumnDescriptor>>& pairs_list);

  static std::filesystem::path getRecoveryPrefix(const std::string& base_path);

  void resolveIncompleteAlterColumnCommands();

  static std::map<std::string, AlterTableAlterColumnCommandRecoveryMgr>
  createRecoveryManagersForCatalogs();

  Catalog_Namespace::Catalog& catalog_;
};

namespace json_utils {
// ColumnDescriptor
void set_value(rapidjson::Value& json_val,
               const ColumnDescriptor& column_desc,
               rapidjson::Document::AllocatorType& allocator);

void get_value(const rapidjson::Value& json_val, ColumnDescriptor& column_desc);
}  // namespace json_utils
