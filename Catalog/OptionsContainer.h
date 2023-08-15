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

#include <string>
#include <unordered_map>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "Logger/Logger.h"
#include "Shared/StringTransform.h"

namespace foreign_storage {
using OptionsMap = std::map<std::string, std::string, std::less<>>;
struct OptionsContainer {
  OptionsMap options;

  OptionsContainer() {}

  OptionsContainer(const OptionsMap& options) : options(options) {}

  OptionsContainer(const std::string& options_str) { populateOptionsMap(options_str); }

  void populateOptionsMap(OptionsMap&& options_map, bool clear = false) {
    if (clear) {
      options = options_map;
    } else {
      // The merge order here is to make sure we overwrite duplicates in options.  If we
      // used options.merge(options_map) we would preserve existing entries.
      options_map.merge(options);
      options = options_map;
    }
  }

  void populateOptionsMap(const rapidjson::Value& ddl_options, bool clear = false) {
    CHECK(ddl_options.IsObject());
    if (clear) {
      options.clear();
    }
    for (auto itr = ddl_options.MemberBegin(); itr != ddl_options.MemberEnd(); ++itr) {
      std::string key = to_upper(itr->name.GetString());
      options[key] = itr->value.GetString();
    }
  }

  void populateOptionsMap(const std::string& options_json, bool clear = false) {
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

  std::optional<std::string> getOption(const std::string_view& key) const {
    if (auto it = options.find(key); it != options.end()) {
      return it->second;
    } else {
      return {};
    }
  }

  bool getOptionAsBool(const std::string_view& key) const {
    auto option = getOption(key);
    return option.has_value() && option.value() == "TRUE";
  }
};
}  // namespace foreign_storage
