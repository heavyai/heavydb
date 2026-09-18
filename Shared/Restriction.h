/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file		Restriction.cpp
 * @brief		Structure to hold details of restrictions a given session has.
 *              Column: name of the column the restriction is on
 *              Values: vector of strings the restriction allows access to
 *
 */

#pragma once

#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include "Logger/Logger.h"

struct Restriction {
  int32_t dbId;
  int32_t tableId;
  int columnId;
  std::set<std::string> values;
  std::string deprecatedSamlColumnName;
  // TODO(sy): deprecatedSamlColumnName: For backwards-compatibility. Maybe remove in
  // OmniSciDB 6.0.

  using Key = std::tuple<int32_t, int32_t, int>;

  Restriction() : dbId(-1), tableId(-1), columnId(-1) {}

  Restriction(int32_t d, int32_t t, int c) : dbId(d), tableId(t), columnId(c) {}

  Restriction(Key key)
      : dbId(std::get<0>(key)), tableId(std::get<1>(key)), columnId(std::get<2>(key)) {}

  Restriction(int32_t d, int32_t t, int c, const std::set<std::string>& v)
      : dbId(d), tableId(t), columnId(c), values(v) {}

  Restriction(Key key, const std::set<std::string>& v)
      : dbId(std::get<0>(key))
      , tableId(std::get<1>(key))
      , columnId(std::get<2>(key))
      , values(v) {}

  Restriction::Key getKey() { return std::make_tuple(dbId, tableId, columnId); }

  friend std::ostream& operator<<(std::ostream& out, const Restriction& r) {
    std::string k;
    if (!r.deprecatedSamlColumnName.empty()) {
      k = "/" + r.deprecatedSamlColumnName;
    }
    out << "Restriction(" << r.dbId << "," << r.tableId << "," << r.columnId << k << ": ["
        << boost::algorithm::join(r.values, ",") << "])";
    return out;
  }
};

using Restrictions = std::map<Restriction::Key, Restriction>;
