/*
 * Copyright 2021 OmniSci, Inc.
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

#include "SQLiteComparator.h"

#include "Shared/DateTimeParser.h"

#include <gtest/gtest.h>

#ifdef _WIN32
#define timegm _mkgmtime
#endif

namespace TestHelpers {

namespace {

void checkTypeConsistency(const int ref_col_type, const SQLTypeInfo& omnisci_ti) {
  if (ref_col_type == SQLITE_NULL) {
    // TODO(alex): re-enable the check that omnisci_ti is nullable,
    //             got invalidated because of outer joins
    return;
  }
  if (omnisci_ti.is_integer()) {
    CHECK_EQ(SQLITE_INTEGER, ref_col_type);
  } else if (omnisci_ti.is_fp() || omnisci_ti.is_decimal()) {
    CHECK(ref_col_type == SQLITE_FLOAT || ref_col_type == SQLITE_INTEGER);
  } else {
    CHECK_EQ(SQLITE_TEXT, ref_col_type);
  }
}

template <class RESULT_SET>
void compare_impl(SqliteConnector& connector,
                  bool use_row_iterator,
                  const RESULT_SET* omnisci_results,
                  const std::string& sqlite_query_string,
                  const ExecutorDeviceType device_type,
                  const bool timestamp_approx,
                  const bool is_arrow = false) {
  auto const errmsg = ExecutorDeviceType::CPU == device_type
                          ? "CPU: " + sqlite_query_string
                          : "GPU: " + sqlite_query_string;
  connector.query(sqlite_query_string);
  ASSERT_EQ(connector.getNumRows(), omnisci_results->rowCount()) << errmsg;
  const int num_rows{static_cast<int>(connector.getNumRows())};
  if (omnisci_results->definitelyHasNoRows()) {
    ASSERT_EQ(0, num_rows) << errmsg;
    return;
  }
  if (!num_rows) {
    return;
  }
  CHECK_EQ(connector.getNumCols(), omnisci_results->colCount()) << errmsg;
  const int num_cols{static_cast<int>(connector.getNumCols())};
  auto row_iterator = omnisci_results->rowIterator(true, true);
  for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
    const auto crt_row =
        use_row_iterator ? *row_iterator++ : omnisci_results->getNextRow(true, true);
    CHECK(!crt_row.empty()) << errmsg;
    CHECK_EQ(static_cast<size_t>(num_cols), crt_row.size()) << errmsg;
    for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
      const auto ref_col_type = connector.columnTypes[col_idx];
      const auto omnisci_variant = crt_row[col_idx];
      const auto scalar_omnisci_variant = boost::get<ScalarTargetValue>(&omnisci_variant);
      CHECK(scalar_omnisci_variant) << errmsg;
      auto omnisci_ti = omnisci_results->getColType(col_idx);
      const auto omnisci_type = omnisci_ti.get_type();
      checkTypeConsistency(ref_col_type, omnisci_ti);
      const bool ref_is_null = connector.isNull(row_idx, col_idx);
      switch (omnisci_type) {
        case kTINYINT:
        case kSMALLINT:
        case kINT:
        case kBIGINT: {
          const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
          ASSERT_NE(nullptr, omnisci_as_int_p);
          const auto omnisci_val = *omnisci_as_int_p;
          if (ref_is_null) {
            ASSERT_EQ(inline_int_null_val(omnisci_ti), omnisci_val) << errmsg;
          } else {
            const auto ref_val = connector.getData<int64_t>(row_idx, col_idx);
            ASSERT_EQ(ref_val, omnisci_val) << errmsg;
          }
          break;
        }
        case kTEXT:
        case kCHAR:
        case kVARCHAR: {
          const auto omnisci_as_str_p =
              boost::get<NullableString>(scalar_omnisci_variant);
          ASSERT_NE(nullptr, omnisci_as_str_p) << errmsg;
          const auto omnisci_str_notnull = boost::get<std::string>(omnisci_as_str_p);
          const auto ref_val = connector.getData<std::string>(row_idx, col_idx);
          if (omnisci_str_notnull) {
            const auto omnisci_val = *omnisci_str_notnull;
            ASSERT_EQ(ref_val, omnisci_val) << errmsg;
          } else {
            // not null but no data, so val is empty string
            const auto omnisci_val = "";
            ASSERT_EQ(ref_val, omnisci_val) << errmsg;
          }
          break;
        }
        case kNUMERIC:
        case kDECIMAL:
        case kDOUBLE: {
          const auto omnisci_as_double_p = boost::get<double>(scalar_omnisci_variant);
          ASSERT_NE(nullptr, omnisci_as_double_p) << errmsg;
          const auto omnisci_val = *omnisci_as_double_p;
          if (ref_is_null) {
            ASSERT_EQ(inline_fp_null_val(SQLTypeInfo(kDOUBLE, false)), omnisci_val)
                << errmsg;
          } else {
            const auto ref_val = connector.getData<double>(row_idx, col_idx);
            if (!std::isinf(omnisci_val) || !std::isinf(ref_val) ||
                ((omnisci_val < 0) ^ (ref_val < 0))) {
              ASSERT_NEAR(ref_val, omnisci_val, EPS * std::fabs(ref_val)) << errmsg;
            }
          }
          break;
        }
        case kFLOAT: {
          const auto omnisci_as_float_p = boost::get<float>(scalar_omnisci_variant);
          ASSERT_NE(nullptr, omnisci_as_float_p) << errmsg;
          const auto omnisci_val = *omnisci_as_float_p;
          if (ref_is_null) {
            ASSERT_EQ(inline_fp_null_val(SQLTypeInfo(kFLOAT, false)), omnisci_val)
                << errmsg;
          } else {
            const auto ref_val = connector.getData<float>(row_idx, col_idx);
            if (!std::isinf(omnisci_val) || !std::isinf(ref_val) ||
                ((omnisci_val < 0) ^ (ref_val < 0))) {
              ASSERT_NEAR(ref_val, omnisci_val, EPS * std::fabs(ref_val)) << errmsg;
            }
          }
          break;
        }
        case kTIMESTAMP:
        case kDATE: {
          const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
          CHECK(omnisci_as_int_p);
          const auto omnisci_val = *omnisci_as_int_p;
          time_t nsec = 0;
          const int dimen = omnisci_ti.get_dimension();
          if (ref_is_null) {
            CHECK_EQ(inline_int_null_val(omnisci_ti), omnisci_val) << errmsg;
          } else {
            const auto ref_val = connector.getData<std::string>(row_idx, col_idx);
            auto temp_val = dateTimeParseOptional<kTIMESTAMP>(ref_val, dimen);
            if (!temp_val) {
              temp_val = dateTimeParseOptional<kDATE>(ref_val, dimen);
            }
            CHECK(temp_val) << ref_val;
            nsec = temp_val.value();
            if (timestamp_approx) {
              // approximate result give 10 second lee way
              ASSERT_NEAR(*omnisci_as_int_p, nsec, dimen > 0 ? 10 * pow(10, dimen) : 10);
            } else {
              struct tm tm_struct {
                0
              };
#ifdef _WIN32
              auto ret_code = gmtime_s(&tm_struct, &nsec);
              CHECK(ret_code == 0) << "Error code returned " << ret_code;
#else
              gmtime_r(&nsec, &tm_struct);
#endif
              if (is_arrow && omnisci_type == kDATE) {
                if (device_type == ExecutorDeviceType::CPU) {
                  ASSERT_EQ(
                      *omnisci_as_int_p,
                      DateConverters::get_epoch_days_from_seconds(timegm(&tm_struct)))
                      << errmsg;
                } else {
                  ASSERT_EQ(*omnisci_as_int_p, timegm(&tm_struct) * kMilliSecsPerSec)
                      << errmsg;
                }
              } else {
                ASSERT_EQ(*omnisci_as_int_p, dimen > 0 ? nsec : timegm(&tm_struct))
                    << errmsg;
              }
            }
          }
          break;
        }
        case kBOOLEAN: {
          const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
          CHECK(omnisci_as_int_p) << errmsg;
          const auto omnisci_val = *omnisci_as_int_p;
          if (ref_is_null) {
            CHECK_EQ(inline_int_null_val(omnisci_ti), omnisci_val) << errmsg;
          } else {
            const auto ref_val = connector.getData<std::string>(row_idx, col_idx);
            if (ref_val == "t") {
              ASSERT_EQ(1, *omnisci_as_int_p) << errmsg;
            } else {
              CHECK_EQ("f", ref_val) << errmsg;
              ASSERT_EQ(0, *omnisci_as_int_p) << errmsg;
            }
          }
          break;
        }
        case kTIME: {
          const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
          CHECK(omnisci_as_int_p) << errmsg;
          const auto omnisci_val = *omnisci_as_int_p;
          if (ref_is_null) {
            CHECK_EQ(inline_int_null_val(omnisci_ti), omnisci_val) << errmsg;
          } else {
            const auto ref_val = connector.getData<std::string>(row_idx, col_idx);
            std::vector<std::string> time_tokens;
            boost::split(time_tokens, ref_val, boost::is_any_of(":"));
            ASSERT_EQ(size_t(3), time_tokens.size()) << errmsg;
            ASSERT_EQ(boost::lexical_cast<int64_t>(time_tokens[0]) * 3600 +
                          boost::lexical_cast<int64_t>(time_tokens[1]) * 60 +
                          boost::lexical_cast<int64_t>(time_tokens[2]),
                      *omnisci_as_int_p)
                << errmsg;
          }
          break;
        }
        default:
          CHECK(false) << errmsg;
      }
    }
  }
}

}  // namespace

void SQLiteComparator::compare(ResultSetPtr omnisci_results,
                               const std::string& query_string,
                               const ExecutorDeviceType device_type) {
  compare_impl(connector_,
               use_row_iterator_,
               omnisci_results.get(),
               query_string,
               device_type,
               false);
}

void SQLiteComparator::compare_arrow_output(
    std::unique_ptr<ArrowResultSet>& arrow_omnisci_results,
    const std::string& sqlite_query_string,
    const ExecutorDeviceType device_type) {
  compare_impl(connector_,
               use_row_iterator_,
               arrow_omnisci_results.get(),
               sqlite_query_string,
               device_type,
               false,
               true);
}

// added to deal with time shift for now testing
void SQLiteComparator::compare_timstamp_approx(ResultSetPtr omnisci_results,
                                               const std::string& query_string,
                                               const ExecutorDeviceType device_type) {
  compare_impl(connector_,
               use_row_iterator_,
               omnisci_results.get(),
               query_string,
               device_type,
               true);
}

}  // namespace TestHelpers
