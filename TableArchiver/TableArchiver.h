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

#include <string>

#include "Catalog/Catalog.h"
#include "Catalog/SessionInfo.h"

struct TableArchiverS3Options {
  std::string s3_access_key;
  std::string s3_secret_key;
  std::string s3_session_token;
  std::string s3_region;
  std::string s3_endpoint;
};

class TableArchiver {
 public:
  TableArchiver(Catalog_Namespace::Catalog* cat) : cat_(cat){};

  void dumpTable(const TableDescriptor* td,
                 const std::string& archive_path,
                 const std::string& compression);

  void restoreTable(const Catalog_Namespace::SessionInfo& session,
                    const std::string& table_name,
                    const std::string& archive_path,
                    const std::string& compression,
                    const TableArchiverS3Options& s3_options);

 private:
  void restoreTable(const Catalog_Namespace::SessionInfo& session,
                    const TableDescriptor* td,
                    const std::string& archive_path,
                    const std::string& compression);

  Catalog_Namespace::Catalog* cat_;
};
