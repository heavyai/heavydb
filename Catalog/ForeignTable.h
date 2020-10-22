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
  static constexpr const char* FRAGMENT_SIZE_KEY = "FRAGMENT_SIZE";
  static constexpr const char* REFRESH_TIMING_TYPE_KEY = "REFRESH_TIMING_TYPE";
  static constexpr const char* REFRESH_START_DATE_TIME_KEY = "REFRESH_START_DATE_TIME";
  static constexpr const char* REFRESH_INTERVAL_KEY = "REFRESH_INTERVAL";
  static constexpr const char* REFRESH_UPDATE_TYPE_KEY = "REFRESH_UPDATE_TYPE";
  static constexpr const char* ALL_REFRESH_UPDATE_TYPE = "ALL";
  static constexpr const char* APPEND_REFRESH_UPDATE_TYPE = "APPEND";
  static constexpr const char* SCHEDULE_REFRESH_TIMING_TYPE = "SCHEDULED";
  static constexpr const char* MANUAL_REFRESH_TIMING_TYPE = "MANUAL";
  static constexpr int NULL_REFRESH_TIME = -1;

  const ForeignServer* foreign_server;
  int64_t last_refresh_time{NULL_REFRESH_TIME}, next_refresh_time{NULL_REFRESH_TIME};
  static constexpr std::array<const char*, 5> supported_options{
      FRAGMENT_SIZE_KEY,
      REFRESH_TIMING_TYPE_KEY,
      REFRESH_START_DATE_TIME_KEY,
      REFRESH_INTERVAL_KEY,
      REFRESH_UPDATE_TYPE_KEY};

  void validate(const std::vector<std::string_view>& supported_data_wrapper_options);
  bool isAppendMode() const;
  std::string getFilePath() const;

 private:
  void validateRecognizedOption(
      const std::vector<std::string_view>& supported_data_wrapper_options);
};
}  // namespace foreign_storage
