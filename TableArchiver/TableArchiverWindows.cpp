/*
 * Copyright 2020 OmniSci, Inc.
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

#include "TableArchiver/TableArchiver.h"

void TableArchiver::dumpTable(const TableDescriptor* td,
                              const std::string& archive_path,
                              const std::string& compression) {
  throw std::runtime_error("Dump/restore table not yet supported on Windows.");
}

void TableArchiver::restoreTable(const Catalog_Namespace::SessionInfo& session,
                                 const TableDescriptor* td,
                                 const std::string& archive_path,
                                 const std::string& compression) {
  throw std::runtime_error("Dump/restore table not yet supported on Windows.");
}

void TableArchiver::restoreTable(const Catalog_Namespace::SessionInfo& session,
                                 const std::string& table_name,
                                 const std::string& archive_path,
                                 const std::string& compression) {
  throw std::runtime_error("Dump/restore table not yet supported on Windows.");
}