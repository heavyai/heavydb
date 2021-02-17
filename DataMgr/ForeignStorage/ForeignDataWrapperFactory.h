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

#include "ForeignDataWrapper.h"

namespace foreign_storage {
/**
 * @type DataWrapperType
 * @brief Encapsulates an enumeration of foreign data wrapper type strings
 */
struct DataWrapperType {
  static constexpr char const* CSV = "OMNISCI_CSV";
  static constexpr char const* PARQUET = "OMNISCI_PARQUET";

  static constexpr std::array<std::string_view, 2> supported_data_wrapper_types{PARQUET,
                                                                                CSV};
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
