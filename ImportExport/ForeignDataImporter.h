/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

  static void setDefaultImportPath(const std::string& base_path);

  // This parameter is publicly exposed for testing purposes only
  static int32_t proxy_foreign_table_fragment_size_;

 protected:
  std::unique_ptr<Fragmenter_Namespace::InsertDataLoader::InsertConnector> connector_;

 private:
  void finalize(const Catalog_Namespace::SessionInfo& parent_session_info,
                ImportStatus& import_status,
                const std::vector<std::pair<const ColumnDescriptor*, StringDictionary*>>&
                    string_dictionaries);

  void finalize(const Catalog_Namespace::SessionInfo& parent_session_info,
                ImportStatus& import_status,
                const int32_t table_id);

#ifdef ENABLE_IMPORT_PARQUET
  ImportStatus importParquet(const Catalog_Namespace::SessionInfo* session_info);
#endif

  ImportStatus importGeneral(const Catalog_Namespace::SessionInfo* session_info);
  ImportStatus importGeneral(const Catalog_Namespace::SessionInfo* session_info,
                             const std::string& copy_from_source,
                             const CopyParams& copy_params);
  ImportStatus importGeneralNoFinalize(const Catalog_Namespace::SessionInfo* session_info,
                                       const std::string& copy_from_source,
                                       const CopyParams& copy_params);

  ImportStatus importGeneralS3(const Catalog_Namespace::SessionInfo* session_info);

  std::string copy_from_source_;
  CopyParams copy_params_;
  const TableDescriptor* table_;
  inline static std::string default_import_path_;
};
}  // namespace import_export
