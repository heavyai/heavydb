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

#include "Shared/StringTransform.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace foreign_storage {
struct DataWrapper {
  static constexpr char const* CSV_WRAPPER_NAME = "OMNISCI_CSV";
  static constexpr char const* PARQUET_WRAPPER_NAME = "OMNISCI_PARQUET";

  std::string name;

  DataWrapper(const std::string& data_wrapper_name) {
    const auto& upper_wrapper_name = to_upper(data_wrapper_name);
    if (upper_wrapper_name == CSV_WRAPPER_NAME ||
        upper_wrapper_name == PARQUET_WRAPPER_NAME) {
      name = upper_wrapper_name;
    } else {
      throw std::runtime_error{"Invalid data wrapper type \"" + data_wrapper_name +
                               "\". Data wrapper type must be one of the following: " +
                               CSV_WRAPPER_NAME + ", " + PARQUET_WRAPPER_NAME + "."};
    }
  }
};

struct ForeignServer {
  static constexpr std::string_view STORAGE_TYPE_KEY = "STORAGE_TYPE";
  static constexpr std::string_view BASE_PATH_KEY = "BASE_PATH";
  static constexpr std::string_view LOCAL_FILE_STORAGE_TYPE = "LOCAL_FILE";
  static constexpr std::array<std::string_view, 1> SUPPORTED_STORAGE_TYPES{
      LOCAL_FILE_STORAGE_TYPE};

  int id;
  std::string name;
  DataWrapper data_wrapper;
  std::map<std::string, std::string, std::less<>> options;

  ForeignServer(const DataWrapper& data_wrapper) : data_wrapper(data_wrapper) {}

  void populateOptionsMap(const rapidjson::Value& ddl_options) {
    for (const auto& member : ddl_options.GetObject()) {
      options[to_upper(member.name.GetString())] = member.value.GetString();
    }
  }

  void populateOptionsMap(const std::string& options_json) {
    rapidjson::Document options;
    options.Parse(options_json);
    populateOptionsMap(options);
  }

  std::string getOptionsAsJsonString() const {
    rapidjson::Document document;
    document.SetObject();

    for (const auto& [key, value] : options) {
      document.AddMember(rapidjson::Value().SetString(
                             key.c_str(), key.length(), document.GetAllocator()),
                         rapidjson::Value().SetString(
                             value.c_str(), value.length(), document.GetAllocator()),
                         document.GetAllocator());
    }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);
    return buffer.GetString();
  }

  void validate() {
    if (options.find(STORAGE_TYPE_KEY) == options.end()) {
      throw std::runtime_error{"Foreign server options must contain \"STORAGE_TYPE\"."};
    }
    if (options.find(BASE_PATH_KEY) == options.end()) {
      throw std::runtime_error{"Foreign server options must contain \"BASE_PATH\"."};
    }
    for (const auto& entry : options) {
      if (entry.first != STORAGE_TYPE_KEY && entry.first != BASE_PATH_KEY) {
        throw std::runtime_error{
            "Invalid option \"" + entry.first +
            "\". "
            "Option must be one of the following: STORAGE_TYPE, BASE_PATH."};
      }
    }
    if (std::find(SUPPORTED_STORAGE_TYPES.begin(),
                  SUPPORTED_STORAGE_TYPES.end(),
                  options.find(STORAGE_TYPE_KEY)->second) ==
        SUPPORTED_STORAGE_TYPES.end()) {
      std::string error_message{
          "Invalid storage type value. Value must be one of the following: "};
      error_message += join(SUPPORTED_STORAGE_TYPES, ", ");
      error_message += ".";
      throw std::runtime_error{error_message};
    }
  }
};
}