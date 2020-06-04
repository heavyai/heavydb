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

#include "CalciteAdapter.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>

#include "Shared/StringTransform.h"

namespace {

std::string pg_shim_impl(const std::string& query) {
  auto result = query;
  {
    static const boost::regex unnest_expr{R"((\s+|,)(unnest)\s*\()",
                                          boost::regex::extended | boost::regex::icase};
    apply_shim(result, unnest_expr, [](std::string& result, const boost::smatch& what) {
      result.replace(what.position(), what.length(), what[1] + "PG_UNNEST(");
    });
  }
  {
    static const boost::regex cast_true_expr{
        R"(CAST\s*\(\s*'t'\s+AS\s+boolean\s*\))",
        boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, cast_true_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(), what.length(), "true");
        });
  }
  {
    static const boost::regex cast_false_expr{
        R"(CAST\s*\(\s*'f'\s+AS\s+boolean\s*\))",
        boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, cast_false_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(), what.length(), "false");
        });
  }
  {
    static const boost::regex ilike_expr{
        R"((\s+|\()((?!\()[^\s]+)\s+ilike\s+('(?:[^']+|'')+')(\s+escape(\s+('[^']+')))?)",
        boost::regex::perl | boost::regex::icase};
    apply_shim(result, ilike_expr, [](std::string& result, const boost::smatch& what) {
      std::string esc = what[6];
      result.replace(what.position(),
                     what.length(),
                     what[1] + "PG_ILIKE(" + what[2] + ", " + what[3] +
                         (esc.empty() ? "" : ", " + esc) + ")");
    });
  }
  {
    static const boost::regex regexp_expr{
        R"((\s+)([^\s]+)\s+REGEXP\s+('(?:[^']+|'')+')(\s+escape(\s+('[^']+')))?)",
        boost::regex::perl | boost::regex::icase};
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
    static const boost::regex quant_expr{R"(([<=>]\s*)(any|all)\s+([^(\s|;)]+))",
                                         boost::regex::extended | boost::regex::icase};
    apply_shim(result, quant_expr, [](std::string& result, const boost::smatch& what) {
      auto const quant_fname = boost::iequals(what[2], "any") ? "PG_ANY(" : "PG_ALL(";
      result.replace(
          what.position(), what.length(), what[1] + quant_fname + what[3] + ')');
    });
  }
  {
    static const boost::regex immediate_cast_expr{
        R"(TIMESTAMP\(([0369])\)\s+('[^']+'))",
        boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, immediate_cast_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(),
                         what.length(),
                         "CAST(" + what[2] + " AS TIMESTAMP(" + what[1] + "))");
        });
  }
  {
    static const boost::regex timestampadd_expr{
        R"(DATE(ADD|DIFF|PART|_TRUNC)\s*\(\s*(\w+)\s*,)",
        boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, timestampadd_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(
              what.position(), what.length(), "DATE" + what[1] + "('" + what[2] + "',");
        });
  }

  {
    static const boost::regex pg_extract_expr{
        R"(PG_EXTRACT\s*\(\s*(\w+)\s*,)", boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, pg_extract_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(), what.length(), "PG_EXTRACT('" + what[1] + "',");
        });

    static const boost::regex extract_expr_quoted{
        R"(extract\s*\(\s*'(\w+)'\s+from\s+(.+)\))",
        boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, extract_expr_quoted, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(),
                         what.length(),
                         "PG_EXTRACT('" + what[1] + "', " + what[2] + ")");
        });

    static const boost::regex extract_expr{R"(extract\s*\(\s*(\w+)\s+from\s+(.+)\))",
                                           boost::regex::extended | boost::regex::icase};
    apply_shim(result, extract_expr, [](std::string& result, const boost::smatch& what) {
      result.replace(what.position(),
                     what.length(),
                     "PG_EXTRACT('" + what[1] + "', " + what[2] + ")");
    });
  }

  {
    static const boost::regex date_trunc_expr{
        R"(([^_])date_trunc\s*)", boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, date_trunc_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(), what.length(), what[1] + "PG_DATE_TRUNC");
        });
  }
  {
    // timestamp add is rewritten to date add
    static const boost::regex timestampadd_expr_quoted{
        R"(TIMESTAMPADD\s*\(\s*'(\w+)'\s*,)",
        boost::regex::extended | boost::regex::icase};
    apply_shim(result,
               timestampadd_expr_quoted,
               [](std::string& result, const boost::smatch& what) {
                 result.replace(
                     what.position(), what.length(), "DATEADD('" + what[1] + "',");
               });
    static const boost::regex timestampadd_expr{
        R"(TIMESTAMPADD\s*\(\s*(\w+)\s*,)", boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, timestampadd_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(), what.length(), "DATEADD('" + what[1] + "',");
        });
  }
  {
    // timestamp diff stays as timestamp diff
    static const boost::regex timestampdiff_expr_quoted{
        R"(TIMESTAMPDIFF\s*\(\s*'(\w+)'\s*,)",
        boost::regex::extended | boost::regex::icase};
    apply_shim(result,
               timestampdiff_expr_quoted,
               [](std::string& result, const boost::smatch& what) {
                 result.replace(
                     what.position(), what.length(), "TIMESTAMPDIFF(" + what[1] + ",");
               });
  }
  {
    static const boost::regex us_timestamp_cast_expr{
        R"(CAST\s*\(\s*('[^']+')\s*AS\s*TIMESTAMP\(6\)\s*\))",
        boost::regex::extended | boost::regex::icase};
    apply_shim(result,
               us_timestamp_cast_expr,
               [](std::string& result, const boost::smatch& what) {
                 result.replace(
                     what.position(), what.length(), "usTIMESTAMP(" + what[1] + ")");
               });
  }
  {
    static const boost::regex ns_timestamp_cast_expr{
        R"(CAST\s*\(\s*('[^']+')\s*AS\s*TIMESTAMP\(9\)\s*\))",
        boost::regex::extended | boost::regex::icase};
    apply_shim(result,
               ns_timestamp_cast_expr,
               [](std::string& result, const boost::smatch& what) {
                 result.replace(
                     what.position(), what.length(), "nsTIMESTAMP(" + what[1] + ")");
               });
  }
  {
    static const boost::regex corr_expr{R"((\s+|,|\()(corr)\s*\()",
                                        boost::regex::extended | boost::regex::icase};
    apply_shim(result, corr_expr, [](std::string& result, const boost::smatch& what) {
      result.replace(what.position(), what.length(), what[1] + "CORRELATION(");
    });
  }
  {
    try {
      // the geography regex pattern is expensive and can sometimes run out of stack space
      // on long queries. Treat it separately from the other shims.
      static const boost::regex cast_to_geography_expr{
          R"(CAST\s*\(\s*(((?!geography).)+)\s+AS\s+geography\s*\))",
          boost::regex::perl | boost::regex::icase};
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
