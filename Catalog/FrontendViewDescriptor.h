/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef FRONTEND_VIEW_DESCRIPTOR_H
#define FRONTEND_VIEW_DESCRIPTOR_H

#include <cstdint>
#include <regex>
#include <string>
#include "../Shared/sqldefs.h"

/**
 * @type FrontendViewDescriptor
 * @brief specifies the content in-memory of a row in the frontend view metadata view
 *
 */

static const std::string SYSTEM_ROLE_TAG("#dash_system_role");

struct FrontendViewDescriptor {
  int32_t viewId;       /**< viewId starts at 0 for valid views. */
  std::string viewName; /**< viewName is the name of the view view -must be unique */
  std::string viewState;
  std::string imageHash;
  std::string updateTime;
  std::string viewMetadata;
  int32_t userId;
  std::string user;
  std::string viewSystemRoleName; /** Stores system role name */
};

inline std::string generate_dash_system_rolename(const std::string& id) {
  return id + SYSTEM_ROLE_TAG;
}

inline std::vector<std::string> parse_underlying_dash_objects(const std::string& meta) {
  /** Parses underlying Tables/Views */
  std::regex extract_objects_expr(".*table\":\"(.*?)\"");
  std::smatch match;
  if (std::regex_search(meta, match, extract_objects_expr)) {
    const std::string list = match[1];
    std::vector<std::string> dash_objects;
    std::regex individual_objects_expr(R"(\w+)");
    std::sregex_iterator iter(list.begin(), list.end(), individual_objects_expr);
    std::sregex_iterator end;
    while (iter != end) {
      dash_objects.push_back((*iter)[0]);
      ++iter;
    }
    return dash_objects;
  }
  return {};
}

#endif  // FRONTEND_VIEW_DESCRIPTOR
