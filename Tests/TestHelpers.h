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
#include "Logger/Logger.h"

#include "LeafHostInfo.h"

#include "arrow/api.h"

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
  for (const ScalarTargetValue& scalar_tv : scalar_tv_vector) {
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
  for (const ScalarTargetValue& scalar_tv : scalar_tv_vector) {
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
  CHECK(geo_r->is_initialized());
  return boost::get<T>(geo_r->get());
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

bool is_null_tv(const TargetValue& tv, const SQLTypeInfo& ti) {
  if (ti.get_notnull()) {
    return false;
  }
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  if (!scalar_tv) {
    CHECK(ti.is_array());
    const auto array_tv = boost::get<ArrayTargetValue>(&tv);
    CHECK(array_tv);
    return !array_tv->is_initialized();
  }
  if (boost::get<int64_t>(scalar_tv)) {
    int64_t data = *(boost::get<int64_t>(scalar_tv));
    switch (ti.get_type()) {
      case kBOOLEAN:
        return data == NULL_BOOLEAN;
      case kTINYINT:
        return data == NULL_TINYINT;
      case kSMALLINT:
        return data == NULL_SMALLINT;
      case kINT:
        return data == NULL_INT;
      case kDECIMAL:
      case kNUMERIC:
      case kBIGINT:
        return data == NULL_BIGINT;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH:
        return data == NULL_BIGINT;
      default:
        CHECK(false);
    }
  } else if (boost::get<double>(scalar_tv)) {
    double data = *(boost::get<double>(scalar_tv));
    if (ti.get_type() == kFLOAT) {
      return data == NULL_FLOAT;
    } else {
      return data == NULL_DOUBLE;
    }
  } else if (boost::get<float>(scalar_tv)) {
    CHECK_EQ(kFLOAT, ti.get_type());
    float data = *(boost::get<float>(scalar_tv));
    return data == NULL_FLOAT;
  } else if (boost::get<NullableString>(scalar_tv)) {
    auto s_n = boost::get<NullableString>(scalar_tv);
    auto s = boost::get<std::string>(s_n);
    return !s;
  }
  CHECK(false);
  return false;
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

void init_logger_stderr_only() {
  logger::LogOptions log_options(nullptr);
  log_options.max_files_ = 0;  // stderr only by default
  logger::init(log_options);
}

struct SharedDictionaryInfo {
  const std::string col;
  const std::string ref_table;
  const std::string ref_col;
};

std::string build_create_table_statement(
    const std::string& columns_definition,
    const std::string& table_name,
    const std::vector<SharedDictionaryInfo>& shared_dict_info,
    const size_t fragment_size,
    const bool use_temporary_tables,
    const bool replicated = false) {
  std::vector<std::string> shared_dict_def;
  if (shared_dict_info.size() > 0) {
    for (size_t idx = 0; idx < shared_dict_info.size(); ++idx) {
      shared_dict_def.push_back(", SHARED DICTIONARY (" + shared_dict_info[idx].col +
                                ") REFERENCES " + shared_dict_info[idx].ref_table + "(" +
                                shared_dict_info[idx].ref_col + ")");
    }
  }

  std::ostringstream with_statement_assembly;
  with_statement_assembly << "fragment_size=" << fragment_size;

  const std::string replicated_def{(!replicated) ? "" : ", PARTITIONS='REPLICATED' "};

  const std::string create_def{use_temporary_tables ? "CREATE TEMPORARY TABLE "
                                                    : "CREATE TABLE "};

  return create_def + table_name + "(" + columns_definition +
         boost::algorithm::join(shared_dict_def, "") + ") WITH (" +
         with_statement_assembly.str() + replicated_def + ");";
}

template <
    typename V,
    std::enable_if_t<!std::is_same_v<V, bool> && std::is_integral<V>::value, int> = 0>
inline V inline_null_value() {
  return inline_int_null_value<V>();
}

template <typename V, std::enable_if_t<std::is_floating_point<V>::value, int> = 0>
inline V inline_null_value() {
  return inline_fp_null_value<V>();
}

template <typename TYPE>
void compare_arrow_array(const std::vector<TYPE>& expected,
                         const std::shared_ptr<arrow::ChunkedArray>& actual) {
  ASSERT_EQ(actual->type()->ToString(),
            arrow::CTypeTraits<TYPE>::type_singleton()->ToString());
  using ArrowColType = arrow::NumericArray<typename arrow::CTypeTraits<TYPE>::ArrowType>;
  const arrow::ArrayVector& chunks = actual->chunks();

  TYPE null_val = inline_null_value<TYPE>();
  size_t compared = 0;

  for (int i = 0; i < actual->num_chunks(); i++) {
    auto chunk = chunks[i];
    auto arrow_row_array = std::static_pointer_cast<ArrowColType>(chunk);

    const TYPE* chunk_data = arrow_row_array->raw_values();
    for (int64_t j = 0; j < arrow_row_array->length(); j++, compared++) {
      if (expected[compared] == null_val) {
        CHECK(chunk->IsNull(j));
      } else {
        CHECK(chunk->IsValid(j));
        ASSERT_EQ(expected[compared], chunk_data[j]);
      }
    }
  }

  ASSERT_EQ(compared, expected.size());
}

void compare_arrow_table_impl(std::shared_ptr<arrow::Table> at, int col_idx) {}

template <typename T, typename... Ts>
void compare_arrow_table_impl(std::shared_ptr<arrow::Table> at,
                              int col_idx,
                              const std::vector<T>& expected,
                              const std::vector<Ts>... expected_rem) {
  ASSERT_LT(col_idx, at->columns().size());
  auto col = at->column(col_idx);
  compare_arrow_array(expected, at->column(col_idx));
  compare_arrow_table_impl(at, col_idx + 1, expected_rem...);
}

template <typename... Ts>
void compare_arrow_table(std::shared_ptr<arrow::Table> at,
                         const std::vector<Ts>&... expected) {
  ASSERT_EQ(at->columns().size(), sizeof...(Ts));
  compare_arrow_table_impl(at, 0, expected...);
}

}  // namespace TestHelpers

#endif  // TEST_HELPERS_H_
