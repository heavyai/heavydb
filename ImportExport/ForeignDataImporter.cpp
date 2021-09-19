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

#include "ForeignDataImporter.h"
#include "DataMgr/ForeignStorage/ForeignDataWrapperFactory.h"
#include "DataMgr/ForeignStorage/ParquetImporter.h"
#include "Importer.h"
#include "Parser/ParserNode.h"
#include "Shared/measure.h"
#include "UserMapping.h"

namespace import_export {

ForeignDataImporter::ForeignDataImporter(const std::string& file_path,
                                         const CopyParams& copy_params,
                                         const TableDescriptor* table)
    : file_path_(file_path), copy_params_(copy_params), table_(table) {
  connector_ = std::make_unique<Parser::LocalConnector>();
}

ImportStatus ForeignDataImporter::import(
    const Catalog_Namespace::SessionInfo* session_info) {
  auto& catalog = session_info->getCatalog();

  CHECK(copy_params_.file_type == import_export::FileType::PARQUET);

  auto& current_user = session_info->get_currentUser();
  auto server = foreign_storage::ForeignDataWrapperFactory::createForeignServerProxy(
      catalog.getDatabaseId(), current_user.userId, file_path_, copy_params_);

  auto user_mapping =
      foreign_storage::ForeignDataWrapperFactory::createUserMappingProxyIfApplicable(
          catalog.getDatabaseId(),
          current_user.userId,
          file_path_,
          copy_params_,
          server.get());

  auto foreign_table =
      foreign_storage::ForeignDataWrapperFactory::createForeignTableProxy(
          catalog.getDatabaseId(), table_, file_path_, copy_params_, server.get());

  foreign_table->validateOptionValues();

  auto data_wrapper = foreign_storage::ForeignDataWrapperFactory::createForImport(
      foreign_storage::DataWrapperType::PARQUET,
      catalog.getDatabaseId(),
      foreign_table.get(),
      user_mapping.get());

  if (auto parquet_import =
          dynamic_cast<foreign_storage::ParquetImporter*>(data_wrapper.get())) {
    Fragmenter_Namespace::InsertDataLoader insert_data_loader(*connector_);
    ImportStatus import_status;  // manually update
    import_status.rows_completed = 0;
    while (true) {
      auto batch_result = parquet_import->getNextImportBatch();
      auto batch = batch_result->getInsertData();
      if (batch.numRows == 0) {
        break;
      }
      insert_data_loader.insertData(*session_info, batch);
      import_status.rows_completed += batch.numRows;
    }

    // TODO: rollback on exceeded number of errors
    connector_->checkpoint(*session_info, foreign_table->tableId);

    return import_status;
  }

  UNREACHABLE();
  return {};
}

}  // namespace import_export
