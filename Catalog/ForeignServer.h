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

#include <string>
#include <unordered_map>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "Catalog/OptionsContainer.h"
#include "Shared/StringTransform.h"

namespace foreign_storage {
/**
 * @type DataWrapperType
 * @brief Encapsulates an enumeration of foreign data wrapper type strings
 */
struct DataWrapperType {
  static constexpr char const* CSV = "OMNISCI_CSV";
  static constexpr char const* PARQUET = "OMNISCI_PARQUET";

  static constexpr std::array<std::string_view, 2> supported_data_wrapper_types{CSV,
                                                                                PARQUET};
};

struct ForeignServer : public OptionsContainer {
  static constexpr std::string_view STORAGE_TYPE_KEY = "STORAGE_TYPE";
  static constexpr std::string_view BASE_PATH_KEY = "BASE_PATH";
  static constexpr std::string_view LOCAL_FILE_STORAGE_TYPE = "LOCAL_FILE";
  static constexpr std::array<std::string_view, 1> supported_storage_types{
      LOCAL_FILE_STORAGE_TYPE};
  static constexpr std::array<std::string_view, 2> all_option_keys{STORAGE_TYPE_KEY,
                                                                   BASE_PATH_KEY};

  int32_t id;
  std::string name;
  std::string data_wrapper_type;
  int32_t user_id;
  time_t creation_time;

  ForeignServer() {}

  ForeignServer(const int32_t server_id,
                const std::string& server_name,
                const std::string& data_wrapper_type,
                const std::string& options_str,
                const int32_t user_id,
                const time_t creation_time)
      : OptionsContainer(options_str)
      , id(server_id)
      , name(server_name)
      , data_wrapper_type(data_wrapper_type)
      , user_id(user_id)
      , creation_time(creation_time) {}

  ForeignServer(const std::string& server_name,
                const std::string& data_wrapper_type,
                const std::map<std::string, std::string, std::less<>>& options,
                const int32_t user_id)
      : OptionsContainer(options)
      , name(server_name)
      , data_wrapper_type(data_wrapper_type)
      , user_id(user_id) {}

  void validate() {
    validateDataWrapperType();
    validateStorageParameters();
    validateRecognizedOption();
  }

 private:
  void validateDataWrapperType() {
    const auto& supported_wrapper_types = DataWrapperType::supported_data_wrapper_types;
    if (std::find(supported_wrapper_types.begin(),
                  supported_wrapper_types.end(),
                  data_wrapper_type) == supported_wrapper_types.end()) {
      throw std::runtime_error{"Invalid data wrapper type \"" + data_wrapper_type +
                               "\". Data wrapper type must be one of the following: " +
                               join(supported_wrapper_types, ", ") + "."};
    }
  }

  void validateStorageParameters() {
    if (options.find(STORAGE_TYPE_KEY) == options.end()) {
      throw std::runtime_error{"Foreign server options must contain \"STORAGE_TYPE\"."};
    }

    if (std::find(supported_storage_types.begin(),
                  supported_storage_types.end(),
                  options.find(STORAGE_TYPE_KEY)->second) ==
        supported_storage_types.end()) {
      std::string error_message =
          "Invalid storage type value. Value must be one of the following: " +
          join(supported_storage_types, ", ") + ".";
      throw std::runtime_error{error_message};
    }
  }

  void validateRecognizedOption() {
    for (const auto& entry : options) {
      if (std::find(all_option_keys.begin(), all_option_keys.end(), entry.first) ==
          all_option_keys.end()) {
        std::string error_message =
            "Invalid option \"" + entry.first + "\"." +
            " Option must be one of the following: " + join(all_option_keys, ", ") + ".";
        throw std::runtime_error{error_message};
      }
    }
  }
};
}  // namespace foreign_storage
