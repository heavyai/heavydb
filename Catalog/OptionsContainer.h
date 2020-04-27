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

#include "Shared/StringTransform.h"

namespace foreign_storage {
struct OptionsContainer {
  std::map<std::string, std::string, std::less<>> options;

  OptionsContainer() {}

  OptionsContainer(const std::map<std::string, std::string, std::less<>>& options)
      : options(options) {}

  OptionsContainer(const std::string& options_str) { populateOptionsMap(options_str); }

  void populateOptionsMap(const rapidjson::Value& ddl_options) {
    CHECK(ddl_options.IsObject());
    for (const auto& member : ddl_options.GetObject()) {
      options[to_upper(member.name.GetString())] = member.value.GetString();
    }
  }

  void populateOptionsMap(const std::string& options_json) {
    CHECK(!options_json.empty());
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
};
}  // namespace foreign_storage
