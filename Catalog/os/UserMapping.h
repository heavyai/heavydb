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

#include <map>
#include <string>
#include "Catalog/ForeignServer.h"

namespace foreign_storage {
struct UserMappingType {
  static constexpr char const* USER = "USER";
  static constexpr char const* PUBLIC = "PUBLIC";
};

struct UserMapping {
  int32_t id;
  int32_t user_id;
  int32_t foreign_server_id;
  std::string type;
  std::string options;

  UserMapping() {}

  UserMapping(const int32_t id,
              const int32_t user_id,
              const int32_t foreign_server_id,
              const std::string type,
              const std::string options)
      : id(id)
      , user_id(user_id)
      , foreign_server_id(foreign_server_id)
      , type(type)
      , options(options) {}

  std::map<std::string, std::string, std::less<>> getUnencryptedOptions() const {
    return {};
  }

  void validate(const ForeignServer* foreign_server) {}
};
}  // namespace foreign_storage
