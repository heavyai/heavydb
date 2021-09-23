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

void ForeignDataImporter::finalize(
    const Catalog_Namespace::SessionInfo& parent_session_info,
    ImportStatus& import_status,
    const std::vector<std::pair<const ColumnDescriptor*, StringDictionary*> >&
        string_dictionaries) {
  if (table_->persistenceLevel ==
      Data_Namespace::MemoryLevel::DISK_LEVEL) {  // only checkpoint disk-resident
                                                  // tables
    if (!import_status.load_failed) {
      auto timer = DEBUG_TIMER("Dictionary Checkpointing");
      for (const auto& [column_desciptor, string_dictionary] : string_dictionaries) {
        if (!string_dictionary->checkpoint()) {
          LOG(ERROR) << "Checkpointing Dictionary for Column "
                     << column_desciptor->columnName << " failed.";
          import_status.load_failed = true;
          import_status.load_msg = "Dictionary checkpoint failed";
          break;
        }
      }
    }
  }
  if (import_status.load_failed) {
    connector_->rollback(parent_session_info, table_->tableId);
  } else {
    connector_->checkpoint(parent_session_info, table_->tableId);
  }
}

ImportStatus ForeignDataImporter::import(
    const Catalog_Namespace::SessionInfo* session_info) {
  auto& catalog = session_info->getCatalog();

#ifdef ENABLE_IMPORT_PARQUET
  CHECK(copy_params_.file_type == import_export::FileType::PARQUET);
#else
  UNREACHABLE() << "Unexpected method call for non-Parquet import";
#endif

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
    while (true) {
      auto batch_result = parquet_import->getNextImportBatch();
      auto batch = batch_result->getInsertData();
      if (!batch) {
        break;
      }
      insert_data_loader.insertData(*session_info, *batch);

      auto batch_import_status = batch_result->getImportStatus();
      import_status.rows_completed += batch_import_status.rows_completed;
      import_status.rows_rejected += batch_import_status.rows_rejected;
      if (import_status.rows_rejected > copy_params_.max_reject) {
        import_status.load_failed = true;
        import_status.load_msg =
            "Load was cancelled due to max reject rows being reached";
        break;
      }
    }

    if (import_status.load_failed) {
      foreign_table.reset();  // this is to avoid calling the TableDescriptor dtor after
                              // the rollback in the checkpoint below
    }

    finalize(*session_info, import_status, parquet_import->getStringDictionaries());

    return import_status;
  }

  UNREACHABLE();
  return {};
}

}  // namespace import_export
