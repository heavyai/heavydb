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

#include "AbstractImporter.h"
#include "Catalog/TableDescriptor.h"
#include "CopyParams.h"
#include "ParserNode.h"

namespace import_export {

class ForeignDataImporter : public AbstractImporter {
 public:
  ForeignDataImporter(const std::string& file_path,
                      const CopyParams& copy_params,
                      const TableDescriptor* table);

  /*
   * Import data returning the status of the import.
   */
  ImportStatus import(const Catalog_Namespace::SessionInfo* session_info) override;

 protected:
  std::unique_ptr<Fragmenter_Namespace::InsertDataLoader::DistributedConnector>
      connector_;

 private:
  std::string file_path_;
  CopyParams copy_params_;
  const TableDescriptor* table_;
};
}  // namespace import_export
