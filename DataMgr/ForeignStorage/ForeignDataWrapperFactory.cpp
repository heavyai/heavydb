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
#include "FsiJsonUtils.h"

#include "CsvDataWrapper.h"
#include "ForeignDataWrapper.h"
#include "InternalCatalogDataWrapper.h"
#ifdef ENABLE_IMPORT_PARQUET
#include "ParquetDataWrapper.h"
#include "ParquetImporter.h"
#endif
#include "RegexParserDataWrapper.h"
#include "Catalog/os/UserMapping.h"

namespace {
std::string get_s3_bucket_from_uri(const std::string& uri) {
  const std::string bucket_prefix = "://";
  auto base_pos = uri.find(bucket_prefix);
  CHECK(base_pos != std::string::npos && base_pos < uri.length());
  auto start_pos = base_pos + bucket_prefix.length();
  auto end_pos = uri.find("/", start_pos);
  return uri.substr(start_pos, end_pos - start_pos);
}

std::string get_s3_key_from_uri(const std::string& uri) {
  const std::string bucket_prefix = "://";
  auto base_pos = uri.find(bucket_prefix);
  CHECK(base_pos != std::string::npos && base_pos < uri.length());
  auto bucket_start_pos = base_pos + bucket_prefix.length();
  auto bucket_end_pos = uri.find("/", bucket_start_pos);
  return uri.substr(bucket_end_pos + 1);
}

bool is_s3_uri(const std::string& file_path) {
  const std::string s3_prefix = "s3://";
  return file_path.find(s3_prefix) != std::string::npos;
}
}  // namespace

namespace foreign_storage {

std::unique_ptr<ForeignDataWrapper> ForeignDataWrapperFactory::createForImport(
    const std::string& data_wrapper_type,
    const int db_id,
    const ForeignTable* foreign_table,
    const UserMapping* user_mapping) {
#ifdef ENABLE_IMPORT_PARQUET
  // only supported for parquet import path currently
  CHECK(data_wrapper_type == DataWrapperType::PARQUET);
  return std::make_unique<ParquetImporter>(db_id, foreign_table, user_mapping);
#else
  return {};
#endif
}

std::unique_ptr<UserMapping>
ForeignDataWrapperFactory::createUserMappingProxyIfApplicable(
    const int db_id,
    const int user_id,
    const std::string& file_path,
    const import_export::CopyParams& copy_params,
    const ForeignServer* server) {
  if (!is_s3_uri(file_path)) {
    return {};
  }

  if (copy_params.s3_access_key.empty() && copy_params.s3_secret_key.empty() &&
      copy_params.s3_session_token.empty()) {
    return {};
  }

  rapidjson::Document d;
  d.SetObject();

  if (!copy_params.s3_access_key.empty()) {
    json_utils::add_value_to_object(d,
                                    copy_params.s3_access_key,
                                    AbstractFileStorageDataWrapper::S3_ACCESS_KEY,
                                    d.GetAllocator());
  }
  if (!copy_params.s3_secret_key.empty()) {
    json_utils::add_value_to_object(d,
                                    copy_params.s3_secret_key,
                                    AbstractFileStorageDataWrapper::S3_SECRET_KEY,
                                    d.GetAllocator());
  }
  if (!copy_params.s3_session_token.empty()) {
    json_utils::add_value_to_object(d,
                                    copy_params.s3_session_token,
                                    AbstractFileStorageDataWrapper::S3_SESSION_TOKEN,
                                    d.GetAllocator());
  }

  auto options_str = json_utils::write_to_string(d);
  auto user_mapping = std::make_unique<UserMapping>();
  user_mapping->options = PkiEncryptor::publicKeyEncrypt(options_str);
  user_mapping->foreign_server_id = -1;
  user_mapping->user_id = OMNISCI_ROOT_USER_ID;
  user_mapping->type = UserMappingType::PUBLIC;
  user_mapping->validate(server);

  return user_mapping;
}

std::unique_ptr<ForeignServer> ForeignDataWrapperFactory::createForeignServerProxy(
    const int db_id,
    const int user_id,
    const std::string& file_path,
    const import_export::CopyParams& copy_params) {
  // only supported for parquet import path currently
  CHECK(copy_params.file_type == import_export::FileType::PARQUET);

  auto foreign_server = std::make_unique<foreign_storage::ForeignServer>();

  foreign_server->id = -1;
  foreign_server->user_id = user_id;
  foreign_server->data_wrapper_type = DataWrapperType::PARQUET;
  foreign_server->name = "import_proxy_server";

  bool is_aws_s3_storage_type = is_s3_uri(file_path);
  if (is_aws_s3_storage_type) {
    throw std::runtime_error("AWS storage not supported");
  } else {
    foreign_server->options[AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY] =
        AbstractFileStorageDataWrapper::LOCAL_FILE_STORAGE_TYPE;
  }

  return foreign_server;
}

std::unique_ptr<ForeignTable> ForeignDataWrapperFactory::createForeignTableProxy(
    const int db_id,
    const TableDescriptor* table,
    const std::string& file_path,
    const import_export::CopyParams& copy_params,
    const ForeignServer* server) {
  // only supported for parquet import path currently
  CHECK(copy_params.file_type == import_export::FileType::PARQUET);

  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  auto foreign_table = std::make_unique<ForeignTable>();

  *static_cast<TableDescriptor*>(foreign_table.get()) =
      *table;  // copy table related values

  CHECK(server);
  foreign_table->foreign_server = server;

  bool is_aws_s3_storage_type = is_s3_uri(file_path);
  if (is_aws_s3_storage_type) {
    throw std::runtime_error("AWS storage not supported");
  } else {
    foreign_table->options["FILE_PATH"] = file_path;
  }

  foreign_table->initializeOptions();
  return foreign_table;
}

std::unique_ptr<ForeignDataWrapper> ForeignDataWrapperFactory::create(
    const std::string& data_wrapper_type,
    const int db_id,
    const ForeignTable* foreign_table) {
  std::unique_ptr<ForeignDataWrapper> data_wrapper;
  if (data_wrapper_type == DataWrapperType::CSV) {
    if (CsvDataWrapper::validateAndGetIsS3Select(foreign_table)) {
      UNREACHABLE();
    } else {
      data_wrapper = std::make_unique<CsvDataWrapper>(db_id, foreign_table);
    }
#ifdef ENABLE_IMPORT_PARQUET
  } else if (data_wrapper_type == DataWrapperType::PARQUET) {
    data_wrapper = std::make_unique<ParquetDataWrapper>(db_id, foreign_table);
#endif
  } else if (data_wrapper_type == DataWrapperType::REGEX_PARSER) {
    data_wrapper = std::make_unique<RegexParserDataWrapper>(db_id, foreign_table);
  } else if (data_wrapper_type == DataWrapperType::INTERNAL_CATALOG) {
    data_wrapper = std::make_unique<InternalCatalogDataWrapper>(db_id, foreign_table);
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
      CsvDataWrapper::validateAndGetIsS3Select(foreign_table)) {
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
#ifdef ENABLE_IMPORT_PARQUET
    } else if (data_wrapper_type == DataWrapperType::PARQUET) {
      validation_data_wrappers_[data_wrapper_type_key] =
          std::make_unique<ParquetDataWrapper>();
#endif
    } else if (data_wrapper_type == DataWrapperType::REGEX_PARSER) {
      validation_data_wrappers_[data_wrapper_type_key] =
          std::make_unique<RegexParserDataWrapper>();
    } else if (data_wrapper_type == DataWrapperType::INTERNAL_CATALOG) {
      validation_data_wrappers_[data_wrapper_type_key] =
          std::make_unique<InternalCatalogDataWrapper>();
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
    std::vector<std::string_view> user_facing_wrapper_types;
    for (const auto& type : supported_wrapper_types) {
      if (type != DataWrapperType::INTERNAL_CATALOG) {
        user_facing_wrapper_types.emplace_back(type);
      }
    }
    throw std::runtime_error{"Invalid data wrapper type \"" + data_wrapper_type +
                             "\". Data wrapper type must be one of the following: " +
                             join(user_facing_wrapper_types, ", ") + "."};
  }
}

std::map<std::string, std::unique_ptr<ForeignDataWrapper>>
    ForeignDataWrapperFactory::validation_data_wrappers_;
}  // namespace foreign_storage
