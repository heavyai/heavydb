/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ForeignDataWrapper.h"

namespace import_export {
template <const bool>
class OptionallyMemoryManagedTypedImportBuffer;
using UnmanagedTypedImportBuffer = OptionallyMemoryManagedTypedImportBuffer<false>;
}  // namespace import_export

namespace foreign_storage {
constexpr const char* kDeletedValueIndicator{"<DELETED>"};

std::string get_db_name(int32_t db_id);
std::string get_table_name(int32_t db_id, int32_t table_id);

void set_node_name(
    std::map<std::string, import_export::UnmanagedTypedImportBuffer*>& import_buffers);

class InternalSystemDataWrapper : public ForeignDataWrapper {
 public:
  InternalSystemDataWrapper();

  InternalSystemDataWrapper(const int db_id, const ForeignTable* foreign_table);

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override;

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

 protected:
  virtual void initializeObjectsForTable(const std::string& table_name) = 0;
  virtual void populateChunkBuffersForTable(
      const std::string& table_name,
      std::map<std::string, import_export::UnmanagedTypedImportBuffer*>&
          import_buffers) = 0;

  const int db_id_;
  const ForeignTable* foreign_table_;
  size_t row_count_{0};
};
}  // namespace foreign_storage
