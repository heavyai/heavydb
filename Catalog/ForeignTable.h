/*
 * Copyright 2020 OmniSci, Inc.
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

#include "ForeignServer.h"
#include "OptionsContainer.h"
#include "TableDescriptor.h"

namespace foreign_storage {

struct ForeignTable : public TableDescriptor, public OptionsContainer {
  ForeignTable();

  ForeignTable(const int32_t id,
               const ForeignServer* server,
               const std::string& options_str,
               const int64_t last_refresh,
               const int64_t next_refresh)
      : OptionsContainer(options_str)
      , foreign_server(server)
      , last_refresh_time(last_refresh)
      , next_refresh_time(next_refresh) {
    tableId = id;
  }

  // Option keys
  static constexpr const char* FRAGMENT_SIZE_KEY = "FRAGMENT_SIZE";
  static constexpr const char* REFRESH_TIMING_TYPE_KEY = "REFRESH_TIMING_TYPE";
  static constexpr const char* REFRESH_START_DATE_TIME_KEY = "REFRESH_START_DATE_TIME";
  static constexpr const char* REFRESH_INTERVAL_KEY = "REFRESH_INTERVAL";
  static constexpr const char* REFRESH_UPDATE_TYPE_KEY = "REFRESH_UPDATE_TYPE";
  static constexpr const char* BUFFER_SIZE_KEY = "BUFFER_SIZE";
  // Option values
  static constexpr const char* ALL_REFRESH_UPDATE_TYPE = "ALL";
  static constexpr const char* APPEND_REFRESH_UPDATE_TYPE = "APPEND";
  static constexpr const char* SCHEDULE_REFRESH_TIMING_TYPE = "SCHEDULED";
  static constexpr const char* MANUAL_REFRESH_TIMING_TYPE = "MANUAL";
  static constexpr int NULL_REFRESH_TIME = -1;

  const ForeignServer* foreign_server;
  int64_t last_refresh_time{NULL_REFRESH_TIME}, next_refresh_time{NULL_REFRESH_TIME};

  inline static const std::set<const char*> supported_options{FRAGMENT_SIZE_KEY,
                                                              REFRESH_TIMING_TYPE_KEY,
                                                              REFRESH_START_DATE_TIME_KEY,
                                                              REFRESH_INTERVAL_KEY,
                                                              REFRESH_UPDATE_TYPE_KEY};

  inline static const std::set<const char*> upper_case_options{
      REFRESH_TIMING_TYPE_KEY,
      REFRESH_START_DATE_TIME_KEY,
      REFRESH_INTERVAL_KEY,
      REFRESH_UPDATE_TYPE_KEY};

  // We don't want all options to be alterable, so this contains a subset.
  inline static const std::set<const char*> alterable_options{REFRESH_TIMING_TYPE_KEY,
                                                              REFRESH_START_DATE_TIME_KEY,
                                                              REFRESH_INTERVAL_KEY,
                                                              REFRESH_UPDATE_TYPE_KEY,
                                                              BUFFER_SIZE_KEY};

  /**
    @brief Verifies the values for mapped options are valid.
   */
  void validateOptionValues() const;

  /**
    @brief Creates an empty option map for the table.  Verifies that the required
    option keys are present and that they contain legal values (as far as they can be
    checked statically).  This is necessary even on a set of empty options because some
    options may be required based on the options set in the server (e.g. file_path is
    needed if the server has no base_path).
  */
  void initializeOptions();

  /**
    @brief Creates an option map from the given json options.  Verifies that the required
    option keys are present and that they contain legal values (as far as they can be
    checked statically).
  */
  void initializeOptions(const rapidjson::Value& options);

  /**
    @brief Verifies that the options_map contains the keys required by a foreign table;
    including those specified by the table's data wrapper.
   */
  void validateSupportedOptionKeys(const OptionsMap& options_map) const;

  /**
    @brief Checks if the table is in append mode.
  */
  bool isAppendMode() const;

  /**
    @brief Creates an options map from given options.  Converts options that must be upper
    case appropriately.
   */
  static OptionsMap createOptionsMap(const rapidjson::Value& json_options);

  /**
    @brief Verifies that the given options map only contains options that can be legally
    altered.
   */
  static void validateAlterOptions(const OptionsMap& options_map);

  /**
    @brief Verifies the schema is supported by this foreign table
   */
  void validateSchema(const std::list<ColumnDescriptor>& columns) const;

 private:
  void validateDataWrapperOptions() const;
  void validateRefreshOptionValues() const;
};
}  // namespace foreign_storage
