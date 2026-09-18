/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <vector>

#include "DataMgr/ForeignStorage/DataPreview.h"
#include "DataMgr/ForeignStorage/ForeignDataWrapper.h"
#include "DataMgr/ForeignStorage/ODBC/odbc_utils.h"

namespace foreign_storage {

class OdbcDataWrapper : public ForeignDataWrapper {
 public:
  OdbcDataWrapper();
  OdbcDataWrapper(const int db_id,
                  const ForeignTable* foreign_table,
                  const UserMapping* user_mapping);

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;
  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override;

  std::string getSerializedDataWrapper() const override;
  void restoreDataWrapperInternals(const std::string& file_path,
                                   const ChunkMetadataVector& chunk_metadata) override;
  bool isRestored() const override;

  OdbcConnectionInfo getDbConnectionInfo() const;
  std::string getDbSelect() const;

  void validateServerOptions(const ForeignServer* foreign_server) const override;

  void validateTableOptions(const ForeignTable* foreign_table) const override;

  const std::set<std::string_view>& getSupportedTableOptions() const override;

  const std::set<std::string> getAlterableTableOptions() const override;

  void validateUserMappingOptions(const UserMapping* user_mapping,
                                  const ForeignServer* foreign_server) const override;

  const std::set<std::string_view>& getSupportedUserMappingOptions() const override;

  /**
   * Gets the set of more specific supported datasource types.
   */
  std::set<std::string> getSupportedSubDatasourceTypes() const;

  ParallelismLevel getCachedParallelismLevel() const override { return INTER_FRAGMENT; }

  ParallelismLevel getNonCachedParallelismLevel() const override {
    return INTRA_FRAGMENT;
  }

  DataPreview getDataPreview(size_t max_row_count) const;

  inline static const std::string ODBC_DSN_KEY = "DATA_SOURCE_NAME";
  inline static const std::string ODBC_CONNECTION_KEY = "CONNECTION_STRING";
  inline static const std::string ODBC_SELECT_KEY = "SQL_SELECT";
  inline static const std::string ODBC_BUFFER_SIZE_KEY = "BUFFER_SIZE";
  inline static const std::string ODBC_ORDER_BY_KEY = "SQL_ORDER_BY";
  constexpr static size_t DEFAULT_BUFFER_SIZE = 8 * 1024 * 1024;

  // user mappings for DATA_SOURCE_NAME option
  inline static const std::string ODBC_USERNAME = "USERNAME";
  inline static const std::string ODBC_PASSWORD = "PASSWORD";
  // user mappings for CONNECTION_STRING option
  inline static const std::string ODBC_CREDENTIAL = "CREDENTIAL_STRING";

 private:
  static const std::set<std::string_view> supported_table_options_;
  static const std::set<std::string_view> supported_server_options_;
  static const std::set<std::string_view> supported_user_mapping_options_;

  void processRemoteDataSource(const int64_t buffer_byte_size,
                               ResultSetProcessor result_set_processor,
                               const OdbcSelectDescriptor& select_desc);

  std::list<Chunk_NS::Chunk> initializeGeoChunks(
      const std::map<ChunkKey, AbstractBuffer*>& required_buffers,
      const std::list<const ColumnDescriptor*>& column_descriptors,
      const int fragment_id);

  void updateGeoChunkMetadata(const std::map<ChunkKey, AbstractBuffer*>& required_buffers,
                              const std::list<Chunk_NS::Chunk>& chunks,
                              const int fragment_id,
                              std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata);

  void processGeoBuffer(const std::map<ChunkKey, AbstractBuffer*>& required_buffers,
                        const std::list<const ColumnDescriptor*>& column_descriptors,
                        const ChunkKey& key,
                        const size_t thread_count,
                        AbstractBuffer* delete_buffer);

  void processChunkbuffer(const std::map<ChunkKey, AbstractBuffer*>& required_buffers,
                          const ColumnDescriptor* column_descriptor,
                          const ChunkKey& key,
                          const size_t thread_count,
                          AbstractBuffer* delete_buffer);

  void processRemoteDataWithThreads(const ChunkKey& key,
                                    const size_t thread_count,
                                    ResultSetProcessor result_set_processor,
                                    const OdbcSelectDescriptor& select_desc);

  void appendRemoteDataToChunk(const ColumnDescriptor* column_descriptor,
                               Data_Namespace::AbstractBuffer* data_buffer,
                               Data_Namespace::AbstractBuffer* index_buffer,
                               Data_Namespace::AbstractBuffer* delete_buffer,
                               const ChunkKey& key,
                               Chunk_NS::Chunk& chunk,
                               const int thread_count,
                               const OdbcSelectDescriptor& select_desc);

  OdbcSelectDescriptor getDefaultSelectDescriptor(
      const ColumnDescriptor* column_descriptor,
      const ChunkKey& key) const;

  OdbcSelectDescriptor getMinValueSelectDescriptor(
      const ColumnDescriptor* column_descriptor,
      const ChunkKey& key) const;

  OdbcSelectDescriptor getMaxValueSelectDescriptor(
      const ColumnDescriptor* column_descriptor,
      const ChunkKey& key) const;

  void populateMinMaxChunkStats(ChunkStats& chunk_stats,
                                const ColumnDescriptor* column_descriptor,
                                const ChunkKey& key);

  const int db_id_;
  const ForeignTable* foreign_table_;

  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> chunk_metadata_map_;

  // map the starting row of the record in the remote db loaded into a
  // fragment.  Used in the limit portion of the select statement executed
  // against the remote db. The key is the fragment id
  std::map<int, int64_t> fragment_remote_db_rownumber_start_;

  std::map<int, foreign_storage::RemoteColumnDescription> remote_column_details_;

  std::mutex odbc_connection_mutex_;
  bool is_restored_{false};

  size_t total_row_count_;

  const UserMapping* user_mapping_;

  std::mutex delete_buffer_mutex_;
};

void validate_odbc_credential_options(const std::string& dsn,
                                      const std::string& connection_string,
                                      const std::string& credential_string,
                                      const std::string& username,
                                      const std::string& password,
                                      const bool is_import,
                                      const std::optional<std::string>& server_name = {});
}  // namespace foreign_storage
