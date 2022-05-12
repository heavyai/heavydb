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

#include "Catalog/TableDescriptor.h"
#include "ForeignDataWrapper.h"
#include "ImportExport/CopyParams.h"

namespace foreign_storage {
class UserMapping;

bool is_s3_uri(const std::string& file_path);

/**
 * Verify if `source_type` is valid.
 */
bool is_valid_source_type(const import_export::CopyParams& copy_params);

void validate_regex_parser_options(const import_export::CopyParams& copy_params);

/**
 * @brief Create proxy fsi objects for use outside FSI
 *
 * @param copy_from_source - the source that will be copied
 * @param copy_params - CopyParams that specify parameters around use case
 * @param db_id - db id of database in use case
 * @param table - the table descriptor of the table in use case
 * @param user_id - the user id of user in use case
 *
 * @return tuple of FSI objects that can be used for particular use case (for example
 * import or detect)
 */
std::tuple<std::unique_ptr<foreign_storage::ForeignServer>,
           std::unique_ptr<foreign_storage::UserMapping>,
           std::unique_ptr<foreign_storage::ForeignTable>>
create_proxy_fsi_objects(const std::string& copy_from_source,
                         const import_export::CopyParams& copy_params,
                         const int db_id,
                         const TableDescriptor* table,
                         const int32_t user_id);

/**
 * @brief Create proxy fsi objects for use outside FSI
 * NOTE: parameters mirror function above
 */
std::tuple<std::unique_ptr<foreign_storage::ForeignServer>,
           std::unique_ptr<foreign_storage::UserMapping>,
           std::unique_ptr<foreign_storage::ForeignTable>>
create_proxy_fsi_objects(const std::string& copy_from_source,
                         const import_export::CopyParams& copy_params,
                         const TableDescriptor* table);

/**
 * @type DataWrapperType
 * @brief Encapsulates an enumeration of foreign data wrapper type strings
 */
struct DataWrapperType {
  static constexpr char const* CSV = "DELIMITED_FILE";
  static constexpr char const* PARQUET = "PARQUET_FILE";
  static constexpr char const* REGEX_PARSER = "REGEX_PARSED_FILE";
  static constexpr char const* INTERNAL_CATALOG = "INTERNAL_CATALOG";
  static constexpr char const* INTERNAL_MEMORY_STATS = "INTERNAL_MEMORY_STATS";
  static constexpr char const* INTERNAL_STORAGE_STATS = "INTERNAL_STORAGE_STATS";

  static constexpr std::array<char const*, 3> INTERNAL_DATA_WRAPPERS{
      INTERNAL_CATALOG,
      INTERNAL_MEMORY_STATS,
      INTERNAL_STORAGE_STATS};

  static constexpr std::array<std::string_view, 6> supported_data_wrapper_types{
      PARQUET,
      CSV,
      REGEX_PARSER,
      INTERNAL_CATALOG,
      INTERNAL_MEMORY_STATS,
      INTERNAL_STORAGE_STATS};
};

class ForeignDataWrapperFactory {
 public:
  /**
   * Creates an instance of a ForeignDataWrapper for the given data wrapper type using
   * provided database and foreign table details.
   */
  static std::unique_ptr<ForeignDataWrapper> create(const std::string& data_wrapper_type,
                                                    const int db_id,
                                                    const ForeignTable* foreign_table);

  static std::unique_ptr<UserMapping> createUserMappingProxyIfApplicable(
      const int db_id,
      const int user_id,
      const std::string& file_path,
      const import_export::CopyParams& copy_params,
      const ForeignServer* server);

  static std::unique_ptr<ForeignServer> createForeignServerProxy(
      const int db_id,
      const int user_id,
      const std::string& file_path,
      const import_export::CopyParams& copy_params);

  static std::unique_ptr<ForeignTable> createForeignTableProxy(
      const int db_id,
      const TableDescriptor* table,
      const std::string& file_path,
      const import_export::CopyParams& copy_params,
      const ForeignServer* server);

  /**
   * Create for the import use-case.
   */
  static std::unique_ptr<ForeignDataWrapper> createForImport(
      const std::string& data_wrapper_type,
      const int db_id,
      const ForeignTable* foreign_table,
      const UserMapping* user_mapping);

  static std::unique_ptr<ForeignDataWrapper> createForGeneralImport(
      const std::string& data_wrapper_type,
      const int db_id,
      const ForeignTable* foreign_table,
      const UserMapping* user_mapping);

  /**
   * Creates an instance (or gets an existing instance) of an immutable ForeignDataWrapper
   * to be used for validation purposes. Returned instance should not be used for any
   * stateful operations, such as fetching foreign table data/metadata.
   */
  static const ForeignDataWrapper& createForValidation(
      const std::string& data_wrapper_type,
      const ForeignTable* foreign_table = nullptr);

  /**
   * Checks that the given data wrapper type is valid.
   */
  static void validateDataWrapperType(const std::string& data_wrapper_type);

 private:
  static std::map<std::string, std::unique_ptr<ForeignDataWrapper>>
      validation_data_wrappers_;
};
}  // namespace foreign_storage
