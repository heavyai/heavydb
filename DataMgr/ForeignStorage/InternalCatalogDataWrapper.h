/*
 * Copyright 2021 OmniSci, Inc.
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

#include <list>

#include "Catalog/DashboardDescriptor.h"
#include "Catalog/ForeignTable.h"
#include "Catalog/SysCatalog.h"
#include "DataMgr/Chunk/Chunk.h"
#include "ForeignDataWrapper.h"

namespace foreign_storage {

class InternalCatalogDataWrapper : public ForeignDataWrapper {
 public:
  InternalCatalogDataWrapper();

  InternalCatalogDataWrapper(const int db_id, const ForeignTable* foreign_table);

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers) override;

  void validateServerOptions(const ForeignServer* foreign_server) const override;

  void validateTableOptions(const ForeignTable* foreign_table) const override;

  const std::set<std::string_view>& getSupportedTableOptions() const override;

  void validateUserMappingOptions(const UserMapping* user_mapping,
                                  const ForeignServer* foreign_server) const override;

  const std::set<std::string_view>& getSupportedUserMappingOptions() const override;

  std::string getSerializedDataWrapper() const override;

  void restoreDataWrapperInternals(const std::string& file_path,
                                   const ChunkMetadataVector& chunk_metadata) override;

  bool isRestored() const override;

 private:
  void initializeObjectsForTable(const std::string& table_name);
  void populateChunkBuffersForTable(
      const std::string& table_name,
      std::map<std::string, import_export::TypedImportBuffer*>& import_buffers);

  const int db_id_;
  const ForeignTable* foreign_table_;
  std::list<Catalog_Namespace::UserMetadata> users_;
  std::map<int32_t, std::vector<TableDescriptor>> tables_by_database_;
  std::map<int32_t, std::vector<DashboardDescriptor>> dashboards_by_database_;
  std::vector<ObjectRoleDescriptor> object_permissions_;
  std::list<Catalog_Namespace::DBMetadata> databases_;
  std::set<std::string> roles_;
  std::map<std::string, std::vector<std::string>> user_names_by_role_;
  size_t row_count_{0};
};
}  // namespace foreign_storage
