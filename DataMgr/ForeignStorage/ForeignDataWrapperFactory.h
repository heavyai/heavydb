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

#include "Catalog/TableDescriptor.h"
#include "ForeignDataWrapper.h"
#include "ImportExport/CopyParams.h"

namespace foreign_storage {
/**
 * @type DataWrapperType
 * @brief Encapsulates an enumeration of foreign data wrapper type strings
 */
struct DataWrapperType {
  static constexpr char const* CSV = "OMNISCI_CSV";
  static constexpr char const* PARQUET = "OMNISCI_PARQUET";
  static constexpr char const* REGEX_PARSER = "OMNISCI_REGEX_PARSER";
  static constexpr char const* INTERNAL_CATALOG = "OMNISCI_INTERNAL_CATALOG";

  static constexpr std::array<std::string_view, 4> supported_data_wrapper_types{
      PARQUET,
      CSV,
      REGEX_PARSER,
      INTERNAL_CATALOG};
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
