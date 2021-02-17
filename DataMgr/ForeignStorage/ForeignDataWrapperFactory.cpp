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

#include "ForeignDataWrapperFactory.h"

#include "CsvDataWrapper.h"
#include "DataMgr/ForeignStorage/CsvShared.h"
#include "ForeignDataWrapper.h"
#include "ParquetDataWrapper.h"

namespace foreign_storage {
std::unique_ptr<ForeignDataWrapper> ForeignDataWrapperFactory::create(
    const std::string& data_wrapper_type,
    const int db_id,
    const ForeignTable* foreign_table) {
  std::unique_ptr<ForeignDataWrapper> data_wrapper;
  if (data_wrapper_type == DataWrapperType::CSV) {
    if (Csv::validate_and_get_is_s3_select(foreign_table)) {
      UNREACHABLE();
    } else {
      data_wrapper = std::make_unique<CsvDataWrapper>(db_id, foreign_table);
    }
  } else if (data_wrapper_type == DataWrapperType::PARQUET) {
    data_wrapper = std::make_unique<ParquetDataWrapper>(db_id, foreign_table);
  } else {
    throw std::runtime_error("Unsupported data wrapper");
  }
  return data_wrapper;
}

const ForeignDataWrapper& ForeignDataWrapperFactory::createForValidation(
    const std::string& data_wrapper_type,
    const ForeignTable* foreign_table) {
  bool is_s3_select_wrapper{false};
  std::string data_wrapper_type_key{data_wrapper_type};
  constexpr const char* S3_SELECT_WRAPPER_KEY = "CSV_S3_SELECT";
  if (foreign_table && data_wrapper_type == DataWrapperType::CSV &&
      Csv::validate_and_get_is_s3_select(foreign_table)) {
    is_s3_select_wrapper = true;
    data_wrapper_type_key = S3_SELECT_WRAPPER_KEY;
  }

  if (validation_data_wrappers_.find(data_wrapper_type_key) ==
      validation_data_wrappers_.end()) {
    if (data_wrapper_type == DataWrapperType::CSV) {
      if (is_s3_select_wrapper) {
        UNREACHABLE();
      } else {
        validation_data_wrappers_[data_wrapper_type_key] =
            std::make_unique<CsvDataWrapper>();
      }
    } else if (data_wrapper_type == DataWrapperType::PARQUET) {
      validation_data_wrappers_[data_wrapper_type_key] =
          std::make_unique<ParquetDataWrapper>();
    } else {
      UNREACHABLE();
    }
  }
  CHECK(validation_data_wrappers_.find(data_wrapper_type_key) !=
        validation_data_wrappers_.end());
  return *validation_data_wrappers_[data_wrapper_type_key];
}

void ForeignDataWrapperFactory::validateDataWrapperType(
    const std::string& data_wrapper_type) {
  const auto& supported_wrapper_types = DataWrapperType::supported_data_wrapper_types;
  if (std::find(supported_wrapper_types.begin(),
                supported_wrapper_types.end(),
                data_wrapper_type) == supported_wrapper_types.end()) {
    throw std::runtime_error{"Invalid data wrapper type \"" + data_wrapper_type +
                             "\". Data wrapper type must be one of the following: " +
                             join(supported_wrapper_types, ", ") + "."};
  }
}

std::map<std::string, std::unique_ptr<ForeignDataWrapper>>
    ForeignDataWrapperFactory::validation_data_wrappers_;
}  // namespace foreign_storage
