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

#include "InternalLogsDataWrapper.h"

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
}  // namespace foreign_storage
