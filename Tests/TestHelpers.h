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

#ifndef TEST_HELPERS_H_
#define TEST_HELPERS_H_

#include "../QueryEngine/TargetValue.h"
#include "Shared/Logger.h"

#include "LeafHostInfo.h"

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/variant.hpp>

namespace TestHelpers {

template <class T>
void compare_array(const TargetValue& r,
                   const std::vector<T>& arr,
                   const double tol = -1.) {
  auto array_tv = boost::get<ArrayTargetValue>(&r);
  CHECK(array_tv);
  if (!array_tv->is_initialized()) {
    ASSERT_EQ(size_t(0), arr.size());
    return;
  }
  const auto& scalar_tv_vector = array_tv->get();
  ASSERT_EQ(scalar_tv_vector.size(), arr.size());
  size_t ctr = 0;
  for (const ScalarTargetValue scalar_tv : scalar_tv_vector) {
    auto p = boost::get<T>(&scalar_tv);
    CHECK(p);
    if (tol < 0.) {
      ASSERT_EQ(*p, arr[ctr++]);
    } else {
      ASSERT_NEAR(*p, arr[ctr++], tol);
    }
  }
}

template <>
void compare_array(const TargetValue& r,
                   const std::vector<std::string>& arr,
                   const double tol) {
  auto array_tv = boost::get<ArrayTargetValue>(&r);
  CHECK(array_tv);
  if (!array_tv->is_initialized()) {
    ASSERT_EQ(size_t(0), arr.size());
    return;
  }
  const auto& scalar_tv_vector = array_tv->get();
  ASSERT_EQ(scalar_tv_vector.size(), arr.size());
  size_t ctr = 0;
  for (const ScalarTargetValue scalar_tv : scalar_tv_vector) {
    auto ns = boost::get<NullableString>(&scalar_tv);
    CHECK(ns);
    auto str = boost::get<std::string>(ns);
    CHECK(str);
    ASSERT_TRUE(*str == arr[ctr++]);
  }
}

template <class T>
void compare_array(const std::vector<T>& a,
                   const std::vector<T>& b,
                   const double tol = -1.) {
  CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); i++) {
    if (tol < 0.) {
      ASSERT_EQ(a[i], b[i]);
    } else {
      ASSERT_NEAR(a[i], b[i], tol);
    }
  }
}

struct GeoTargetComparator {
  static void compare(const GeoPointTargetValue& a,
                      const GeoPointTargetValue& b,
                      const double tol = -1.) {
    compare_array(*a.coords, *b.coords, tol);
  }
  static void compare(const GeoLineStringTargetValue& a,
                      const GeoLineStringTargetValue& b,
                      const double tol = -1.) {
    compare_array(*a.coords, *b.coords, tol);
  }
  static void compare(const GeoPolyTargetValue& a,
                      const GeoPolyTargetValue& b,
                      const double tol = -1.) {
    compare_array(*a.coords, *b.coords, tol);
    compare_array(*a.ring_sizes, *b.ring_sizes);
  }
  static void compare(const GeoMultiPolyTargetValue& a,
                      const GeoMultiPolyTargetValue& b,
                      const double tol = -1.) {
    compare_array(*a.coords, *b.coords, tol);
    compare_array(*a.ring_sizes, *b.ring_sizes);
    compare_array(*a.poly_rings, *b.poly_rings);
  }
};

template <class T>
T g(const TargetValue& r) {
  auto geo_r = boost::get<GeoTargetValue>(&r);
  CHECK(geo_r);
  auto p = boost::get<T>(geo_r);
  CHECK(p);
  return *p;
}

template <class T>
void compare_geo_target(const TargetValue& r,
                        const T& geo_truth_target,
                        const double tol = -1.) {
  const auto geo_value = g<T>(r);
  GeoTargetComparator::compare(geo_value, geo_truth_target, tol);
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

template <typename T>
inline std::string convert(const T& t) {
  return std::to_string(t);
}

template <std::size_t N>
inline std::string convert(const char (&t)[N]) {
  return std::string(t);
}

template <>
inline std::string convert(const std::string& t) {
  return t;
}

struct ValuesGenerator {
  ValuesGenerator(const std::string& table_name) : table_name_(table_name) {}

  template <typename... COL_ARGS>
  std::string operator()(COL_ARGS&&... args) const {
    std::vector<std::string> vals({convert(std::forward<COL_ARGS>(args))...});
    return std::string("INSERT INTO " + table_name_ + " VALUES(" +
                       boost::algorithm::join(vals, ",") + ");");
  }

  const std::string table_name_;
};

LeafHostInfo to_leaf_host_info(std::string& server_info, NodeRole role) {
  size_t pos = server_info.find(':');
  if (pos == std::string::npos) {
    throw std::runtime_error("Invalid host:port -> " + server_info);
  }

  auto host = server_info.substr(0, pos);
  auto port = server_info.substr(pos + 1);

  return LeafHostInfo(host, std::stoi(port), role);
}

std::vector<LeafHostInfo> to_leaf_host_info(std::vector<std::string>& server_infos,
                                            NodeRole role) {
  std::vector<LeafHostInfo> host_infos;

  for (auto& server_info : server_infos) {
    host_infos.push_back(to_leaf_host_info(server_info, role));
  }

  return host_infos;
}

void init_logger_stderr_only(int argc, char const* const* argv) {
  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  log_options.parse_command_line(argc, argv);
  logger::init(log_options);
}

}  // namespace TestHelpers

#endif  // TEST_HELPERS_H_
