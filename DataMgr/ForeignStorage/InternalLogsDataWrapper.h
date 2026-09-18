/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/ForeignStorage/LogFileBufferParser.h"
#include "RegexParserDataWrapper.h"

namespace foreign_storage {
class InternalLogsDataWrapper : public RegexParserDataWrapper {
 public:
  InternalLogsDataWrapper();

  InternalLogsDataWrapper(const int db_id, const ForeignTable* foreign_table);

  InternalLogsDataWrapper(const int db_id,
                          const ForeignTable* foreign_table,
                          const UserMapping* user_mapping);

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;

  std::string getSerializedDataWrapper() const override;

 protected:
  const TextFileBufferParser& getFileBufferParser() const override;
  std::optional<size_t> getMaxFileCount() const override;

 private:
  const LogFileBufferParser log_file_buffer_parser_;
};
}  // namespace foreign_storage
