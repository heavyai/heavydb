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

#include "CalciteAdapter.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>

#include "Logger/Logger.h"
#include "Shared/StringTransform.h"
#include "Shared/clean_boost_regex.hpp"

namespace {

std::string pg_shim_impl(const std::string& query) {
  auto result = query;
  {
    static const auto& unnest_expr = *new boost::regex(
        R"((\s+|,)(unnest)\s*\()", boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(unnest_expr)>);
    apply_shim(result, unnest_expr, [](std::string& result, const boost::smatch& what) {
      result.replace(what.position(), what.length(), what[1] + "PG_UNNEST(");
    });
  }
  {
    static const auto& cast_true_expr =
        *new boost::regex(R"(CAST\s*\(\s*'t'\s+AS\s+boolean\s*\))",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(cast_true_expr)>);
    apply_shim(
        result, cast_true_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(), what.length(), "true");
        });
  }
  {
    static const auto& cast_false_expr =
        *new boost::regex(R"(CAST\s*\(\s*'f'\s+AS\s+boolean\s*\))",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(cast_false_expr)>);
    apply_shim(
        result, cast_false_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(), what.length(), "false");
        });
  }
  {
    static const auto& ilike_expr = *new boost::regex(
        R"((\s+|\()((?!\()[^\s]+)\s+ilike\s+('(?:[^']+|'')+')(\s+escape(\s+('[^']+')))?)",
        boost::regex::perl | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(ilike_expr)>);
    apply_shim(result, ilike_expr, [](std::string& result, const boost::smatch& what) {
      std::string esc = what[6];
      result.replace(what.position(),
                     what.length(),
                     what[1] + "PG_ILIKE(" + what[2] + ", " + what[3] +
                         (esc.empty() ? "" : ", " + esc) + ")");
    });
  }
  {
    static const auto& regexp_expr = *new boost::regex(
        R"((\s+)([^\s]+)\s+REGEXP\s+('(?:[^']+|'')+')(\s+escape(\s+('[^']+')))?)",
        boost::regex::perl | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(regexp_expr)>);
    apply_shim(result, regexp_expr, [](std::string& result, const boost::smatch& what) {
      std::string esc = what[6];
      result.replace(what.position(),
                     what.length(),
                     what[1] + "REGEXP_LIKE(" + what[2] + ", " + what[3] +
                         (esc.empty() ? "" : ", " + esc) + ")");
    });
  }
  {
    // Comparison operator needed to distinguish from other uses of ALL (e.g. UNION ALL)
    static const auto& quant_expr =
        *new boost::regex(R"(([<=>]\s*)(any|all)\s+([^(\s|;)]+))",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(quant_expr)>);
    apply_shim(result, quant_expr, [](std::string& result, const boost::smatch& what) {
      auto const quant_fname = boost::iequals(what[2], "any") ? "PG_ANY(" : "PG_ALL(";
      result.replace(
          what.position(), what.length(), what[1] + quant_fname + what[3] + ')');
    });
  }
  {
    static const auto& immediate_cast_expr =
        *new boost::regex(R"(TIMESTAMP\(([0369])\)\s+('[^']+'))",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(immediate_cast_expr)>);
    apply_shim(
        result, immediate_cast_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(),
                         what.length(),
                         "CAST(" + what[2] + " AS TIMESTAMP(" + what[1] + "))");
        });
  }
  {
    static const auto& timestampadd_expr =
        *new boost::regex(R"(DATE(ADD|DIFF|PART|_TRUNC)\s*\(\s*(\w+)\s*,)",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(timestampadd_expr)>);
    apply_shim(
        result, timestampadd_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(
              what.position(), what.length(), "DATE" + what[1] + "('" + what[2] + "',");
        });
  }

  {
    static const auto& pg_extract_expr = *new boost::regex(
        R"(PG_EXTRACT\s*\(\s*(\w+)\s*,)", boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(pg_extract_expr)>);
    apply_shim(
        result, pg_extract_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(), what.length(), "PG_EXTRACT('" + what[1] + "',");
        });

    static const auto& extract_expr_quoted =
        *new boost::regex(R"(extract\s*\(\s*'(\w+)'\s+from\s+(.+)\))",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(extract_expr_quoted)>);
    apply_shim(
        result, extract_expr_quoted, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(),
                         what.length(),
                         "PG_EXTRACT('" + what[1] + "', " + what[2] + ")");
        });

    static const auto& extract_expr =
        *new boost::regex(R"(extract\s*\(\s*(\w+)\s+from\s+(.+)\))",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(extract_expr)>);
    apply_shim(result, extract_expr, [](std::string& result, const boost::smatch& what) {
      result.replace(what.position(),
                     what.length(),
                     "PG_EXTRACT('" + what[1] + "', " + what[2] + ")");
    });
  }

  {
    static const auto& date_trunc_expr = *new boost::regex(
        R"(([^_])date_trunc\s*)", boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(date_trunc_expr)>);
    apply_shim(
        result, date_trunc_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(), what.length(), what[1] + "PG_DATE_TRUNC");
        });
  }
  {
    static const auto& timestampadd_expr_quoted =
        *new boost::regex(R"(TIMESTAMP(ADD|DIFF)\s*\(\s*'(\w+)'\s*,)",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(timestampadd_expr_quoted)>);
    apply_shim(result,
               timestampadd_expr_quoted,
               [](std::string& result, const boost::smatch& what) {
                 result.replace(what.position(),
                                what.length(),
                                "DATE" + what[1] + "('" + what[2] + "',");
               });
    static const auto& timestampadd_expr =
        *new boost::regex(R"(TIMESTAMP(ADD|DIFF)\s*\(\s*(\w+)\s*,)",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(timestampadd_expr)>);
    apply_shim(
        result, timestampadd_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(
              what.position(), what.length(), "DATE" + what[1] + "('" + what[2] + "',");
        });
  }
  {
    static const auto& us_timestamp_cast_expr =
        *new boost::regex(R"(CAST\s*\(\s*('[^']+')\s*AS\s*TIMESTAMP\(6\)\s*\))",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(us_timestamp_cast_expr)>);
    apply_shim(result,
               us_timestamp_cast_expr,
               [](std::string& result, const boost::smatch& what) {
                 result.replace(
                     what.position(), what.length(), "usTIMESTAMP(" + what[1] + ")");
               });
  }
  {
    static const auto& ns_timestamp_cast_expr =
        *new boost::regex(R"(CAST\s*\(\s*('[^']+')\s*AS\s*TIMESTAMP\(9\)\s*\))",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(ns_timestamp_cast_expr)>);
    apply_shim(result,
               ns_timestamp_cast_expr,
               [](std::string& result, const boost::smatch& what) {
                 result.replace(
                     what.position(), what.length(), "nsTIMESTAMP(" + what[1] + ")");
               });
  }
  {
    static const auto& corr_expr = *new boost::regex(
        R"((\s+|,|\()(corr)\s*\()", boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(corr_expr)>);
    apply_shim(result, corr_expr, [](std::string& result, const boost::smatch& what) {
      result.replace(what.position(), what.length(), what[1] + "CORRELATION(");
    });
  }
  {
    try {
      // the geography regex pattern is expensive and can sometimes run out of stack space
      // on long queries. Treat it separately from the other shims.
      static const auto& cast_to_geography_expr =
          *new boost::regex(R"(CAST\s*\(\s*(((?!geography).)+)\s+AS\s+geography\s*\))",
                            boost::regex::perl | boost::regex::icase);
      static_assert(std::is_trivially_destructible_v<decltype(cast_to_geography_expr)>);
      apply_shim(result,
                 cast_to_geography_expr,
                 [](std::string& result, const boost::smatch& what) {
                   result.replace(what.position(),
                                  what.length(),
                                  "CastToGeography(" + what[1] + ")");
                 });
    } catch (const std::exception& e) {
      LOG(WARNING) << "Error apply geography cast shim: " << e.what()
                   << "\nContinuing query parse...";
    }
  }
  {
    static const auto& interval_subsecond_expr =
        *new boost::regex(R"(interval\s+([0-9]+)\s+(millisecond|microsecond|nanosecond))",
                          boost::regex::extended | boost::regex::icase);
    static_assert(std::is_trivially_destructible_v<decltype(interval_subsecond_expr)>);
    apply_shim(
        result,
        interval_subsecond_expr,
        [](std::string& result, const boost::smatch& what) {
          std::string interval_str = what[1];
          const std::string time_unit_str = to_lower(to_string(what[2]));
          static const std::array<std::pair<std::string_view, size_t>, 3> precision_map{
              std::make_pair("millisecond", 3),
              std::make_pair("microsecond", 6),
              std::make_pair("nanosecond", 9)};
          static_assert(std::is_trivially_destructible_v<decltype(precision_map)>);
          auto precision_it = std::find_if(
              precision_map.cbegin(),
              precision_map.cend(),
              [&time_unit_str](const std::pair<std::string_view, size_t>& precision) {
                return time_unit_str.compare(precision.first) == 0;
              });
          if (precision_it != precision_map.end()) {
            std::ostringstream out;
            const auto interval_time = std::strtod(interval_str.c_str(), nullptr);
            double const scale = shared::power10(precision_it->second);
            out << std::fixed << interval_time / scale;
            interval_str = out.str();
            result.replace(
                what.position(), what.length(), "interval " + interval_str + " second");
          }
        });
  }

  return result;
}

}  // namespace

std::string pg_shim(const std::string& query) {
  try {
    return pg_shim_impl(query);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Error applying shim: " << e.what() << "\nContinuing query parse...";
    // boost::regex throws an exception about the complexity of matching when
    // the wrong type of quotes are used or they're mismatched. Let the query
    // through unmodified, the parser will throw a much more informative error.
  }
  return query;
}
