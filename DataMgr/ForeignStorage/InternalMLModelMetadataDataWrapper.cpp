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

#include "InternalMLModelMetadataDataWrapper.h"

#include "Catalog/SysCatalog.h"
#include "ImportExport/Importer.h"
#include "QueryEngine/Execute.h"

#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/MLModel.h"

namespace foreign_storage {
InternalMLModelMetadataDataWrapper::InternalMLModelMetadataDataWrapper()
    : InternalSystemDataWrapper() {}

InternalMLModelMetadataDataWrapper::InternalMLModelMetadataDataWrapper(
    const int db_id,
    const ForeignTable* foreign_table)
    : InternalSystemDataWrapper(db_id, foreign_table) {}

namespace {
void populate_import_buffers_for_ml_model_metadata(
    const std::vector<MLModelMetadata>& ml_models_metadata,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& ml_model_metadata : ml_models_metadata) {
    if (auto itr = import_buffers.find("model_name"); itr != import_buffers.end()) {
      itr->second->addString(ml_model_metadata.getModelName());
    }
    if (auto itr = import_buffers.find("model_type"); itr != import_buffers.end()) {
      itr->second->addString(ml_model_metadata.getModelType());
    }
    if (auto itr = import_buffers.find("predicted"); itr != import_buffers.end()) {
      itr->second->addString(ml_model_metadata.getPredicted());
    }
    if (auto itr = import_buffers.find("predictors"); itr != import_buffers.end()) {
      itr->second->addStringArray(ml_model_metadata.getPredictors());
    }
    if (auto itr = import_buffers.find("training_query"); itr != import_buffers.end()) {
      itr->second->addString(ml_model_metadata.getTrainingQuery());
    }
    if (auto itr = import_buffers.find("num_logical_features");
        itr != import_buffers.end()) {
      itr->second->addBigint(ml_model_metadata.getNumLogicalFeatures());
    }
    if (auto itr = import_buffers.find("num_physical_features");
        itr != import_buffers.end()) {
      itr->second->addBigint(ml_model_metadata.getNumFeatures());
    }
    if (auto itr = import_buffers.find("num_categorical_features");
        itr != import_buffers.end()) {
      itr->second->addBigint(ml_model_metadata.getNumCategoricalFeatures());
    }
    if (auto itr = import_buffers.find("num_numeric_features");
        itr != import_buffers.end()) {
      itr->second->addBigint(ml_model_metadata.getNumLogicalFeatures() -
                             ml_model_metadata.getNumCategoricalFeatures());
    }
  }
}

}  // namespace

void InternalMLModelMetadataDataWrapper::initializeObjectsForTable(
    const std::string& table_name) {
  CHECK_EQ(foreign_table_->tableName, table_name);
  CHECK_EQ(foreign_table_->tableName, Catalog_Namespace::ML_MODEL_METADATA_SYS_TABLE_NAME)
      << "Unexpected table name: " << foreign_table_->tableName;
  ml_models_metadata_ = ml_models_.getModelMetadata();
  row_count_ = ml_models_metadata_.size();
}

void InternalMLModelMetadataDataWrapper::populateChunkBuffersForTable(
    const std::string& table_name,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  CHECK_EQ(foreign_table_->tableName, table_name);
  CHECK_EQ(foreign_table_->tableName, Catalog_Namespace::ML_MODEL_METADATA_SYS_TABLE_NAME)
      << "Unexpected table name: " << foreign_table_->tableName;
  populate_import_buffers_for_ml_model_metadata(ml_models_metadata_, import_buffers);
}

}  // namespace foreign_storage
