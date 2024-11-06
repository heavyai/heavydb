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
