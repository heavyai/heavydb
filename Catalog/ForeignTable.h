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

template <typename T>
bool contains(const T& set, const std::string_view element) {
  if (std::find(set.begin(), set.end(), element) == set.end()) {
    return false;
  } else {
    return true;
  }
}

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
  static constexpr const char* FILE_PATH_KEY = "FILE_PATH";
  static constexpr const char* FRAGMENT_SIZE_KEY = "FRAGMENT_SIZE";
  static constexpr const char* REFRESH_TIMING_TYPE_KEY = "REFRESH_TIMING_TYPE";
  static constexpr const char* REFRESH_START_DATE_TIME_KEY = "REFRESH_START_DATE_TIME";
  static constexpr const char* REFRESH_INTERVAL_KEY = "REFRESH_INTERVAL";
  static constexpr const char* REFRESH_UPDATE_TYPE_KEY = "REFRESH_UPDATE_TYPE";
  // Option values
  static constexpr const char* ALL_REFRESH_UPDATE_TYPE = "ALL";
  static constexpr const char* APPEND_REFRESH_UPDATE_TYPE = "APPEND";
  static constexpr const char* SCHEDULE_REFRESH_TIMING_TYPE = "SCHEDULED";
  static constexpr const char* MANUAL_REFRESH_TIMING_TYPE = "MANUAL";
  static constexpr int NULL_REFRESH_TIME = -1;

  const ForeignServer* foreign_server;
  int64_t last_refresh_time{NULL_REFRESH_TIME}, next_refresh_time{NULL_REFRESH_TIME};
  inline static const std::set<const char*> supported_options{FILE_PATH_KEY,
                                                              FRAGMENT_SIZE_KEY,
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
                                                              REFRESH_UPDATE_TYPE_KEY};

  void validateOptions() const;
  void initializeOptions(const rapidjson::Value& options);
  std::vector<std::string_view> getSupportedDataWrapperOptions() const;
  void validateSupportedOptions(const OptionsMap& options_map) const;
  bool isAppendMode() const;
  std::string getFilePath() const;

  static OptionsMap create_options_map(const rapidjson::Value& json_options);
  static void validate_alter_options(const OptionsMap& options_map);

 private:
  void validateDataWrapperOptions() const;
  void validateRefreshOptions() const;
};
}  // namespace foreign_storage
