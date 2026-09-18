/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "InternalLogsDataWrapper.h"
#include "ForeignStorageException.h"

size_t g_logs_system_tables_max_files_count{100};

namespace foreign_storage {
InternalLogsDataWrapper::InternalLogsDataWrapper()
    : RegexParserDataWrapper(), log_file_buffer_parser_{nullptr, -1} {}

InternalLogsDataWrapper::InternalLogsDataWrapper(const int db_id,
                                                 const ForeignTable* foreign_table)
    : RegexParserDataWrapper(db_id, foreign_table)
    , log_file_buffer_parser_{foreign_table, db_id} {}

InternalLogsDataWrapper::InternalLogsDataWrapper(const int db_id,
                                                 const ForeignTable* foreign_table,
                                                 const UserMapping* user_mapping)
    : RegexParserDataWrapper(db_id, foreign_table, user_mapping)
    , log_file_buffer_parser_{foreign_table, db_id} {}

const TextFileBufferParser& InternalLogsDataWrapper::getFileBufferParser() const {
  return log_file_buffer_parser_;
}

std::optional<size_t> InternalLogsDataWrapper::getMaxFileCount() const {
  return g_logs_system_tables_max_files_count;
}

void InternalLogsDataWrapper::populateChunkMetadata(ChunkMetadataVector& meta_vec) {
  try {
    AbstractTextFileDataWrapper::populateChunkMetadata(meta_vec);
  } catch (const shared::NoRegexFilterMatchException& e) {
  }
}

std::string InternalLogsDataWrapper::getSerializedDataWrapper() const {
  // The file reader might not exist if we threw an exception during metadata creation.
  // This is possible for some of the internal data wrappers which allow for empty
  // wrappers.
  if (!file_reader_) {
    throw IncompleteWrapperException("Wrapper has no file reader");
  }
  return AbstractTextFileDataWrapper::getSerializedDataWrapper();
}
}  // namespace foreign_storage
