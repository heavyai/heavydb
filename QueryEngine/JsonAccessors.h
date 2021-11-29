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

/**
 * @file    JsonAccessors.h
 * @brief
 *
 * Checked json field retrieval.
 *
 */

#ifndef QUERYENGINE_JSONACCESSORS_H
#define QUERYENGINE_JSONACCESSORS_H

#include <iostream>

#include <rapidjson/document.h>
#include "Logger/Logger.h"

inline const rapidjson::Value& field(const rapidjson::Value& obj,
                                     const char field[]) noexcept {
  CHECK(obj.IsObject());
  const auto field_it = obj.FindMember(field);
  CHECK(field_it != obj.MemberEnd());
  return field_it->value;
}

inline const int64_t json_i64(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsInt64());
  return obj.GetInt64();
}

inline const std::string json_str(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsString());
  return obj.GetString();
}

inline const bool json_bool(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsBool());
  return obj.GetBool();
}

inline const double json_double(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsDouble());
  return obj.GetDouble();
}

#endif  // QUERYENGINE_JSONACCESSORS_H
