/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <regex>
#include <string>
#include "../Shared/sqldefs.h"

/**
 * @type DashboardDescriptor
 * @brief specifies the content in-memory of a row in the dashboard
 *
 */

static const std::string SYSTEM_ROLE_TAG("#dash_system_role");

struct DashboardDescriptor {
  int32_t dashboardId;       /**< dashboardId starts at 0 for valid dashboard. */
  std::string dashboardName; /**< dashboardName is the name of the dashboard. dashboard
                                -must be unique */
  std::string dashboardState;
  std::string imageHash;
  std::string updateTime;
  std::string dashboardMetadata;
  int32_t userId;
  std::string user;
  std::string dashboardSystemRoleName; /** Stores system role name */
};

inline std::string generate_dashboard_system_rolename(const std::string& db_id,
                                                      const std::string& dash_id) {
  return db_id + "_" + dash_id + SYSTEM_ROLE_TAG;
}

inline std::vector<std::string> parse_underlying_dashboard_objects(
    const std::string& meta) {
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
