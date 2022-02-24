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

#include "TestHelpers.h"

#include "ArrowStorage/ArrowStorage.h"
#include "Calcite/Calcite.h"
#include "DataMgr/DataMgr.h"
#include "DistributedLoader.h"
#include "ImportExport/Importer.h"
#include "Parser/parser.h"
#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/CgenState.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExpressionRange.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "QueryEngine/ResultSetReductionJIT.h"
#include "QueryEngine/ThriftSerializers.h"
#include "Shared/DateConverters.h"
#include "Shared/StringTransform.h"
#include "Shared/scope.h"
#include "SqliteConnector/SqliteConnector.h"

#include "gen-cpp/CalciteServer.h"

#include "SchemaJson.h"

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/any.hpp>
#include <boost/program_options.hpp>

#include <cmath>
#include <cstdio>

using namespace std;
using namespace TestHelpers;

bool g_aggregator{false};

extern bool g_enable_smem_group_by;
extern bool g_allow_cpu_retry;
extern bool g_allow_query_step_cpu_retry;
extern bool g_enable_watchdog;
extern bool g_skip_intermediate_count;
extern bool g_enable_left_join_filter_hoisting;
extern bool g_from_table_reordering;
extern bool g_inf_div_by_zero;
extern bool g_null_div_by_zero;

extern size_t g_big_group_threshold;
extern unsigned g_trivial_loop_join_threshold;
extern bool g_enable_overlaps_hashjoin;
extern double g_gpu_mem_limit_percent;
extern size_t g_parallel_top_min;
extern size_t g_constrained_by_in_threshold;

extern bool g_enable_window_functions;
extern bool g_enable_calcite_view_optimize;
extern bool g_enable_bump_allocator;
extern bool g_enable_interop;
extern bool g_enable_union;

extern bool g_is_test_env;

namespace import_export {

ArrayDatum StringToArray(const std::string& s,
                         const SQLTypeInfo& ti,
                         const CopyParams& copy_params);

}  // namespace import_export

namespace {

bool g_hoist_literals{true};
bool g_use_row_iterator{true};

constexpr double EPS = 1.25e-5;

class SQLiteComparator {
 public:
  SQLiteComparator() : connector_("sqliteTestDB", "") {}

  void query(const std::string& query_string) { connector_.query(query_string); }

  void compare(ResultSetPtr omnisci_results,
               const std::string& query_string,
               const ExecutorDeviceType device_type) {
    compare_impl(omnisci_results.get(), query_string, device_type, false);
  }

  void compare_arrow_output(std::unique_ptr<ArrowResultSet>& arrow_omnisci_results,
                            const std::string& sqlite_query_string,
                            const ExecutorDeviceType device_type) {
    compare_impl(
        arrow_omnisci_results.get(), sqlite_query_string, device_type, false, true);
  }

  // added to deal with time shift for now testing
  void compare_timstamp_approx(ResultSetPtr omnisci_results,
                               const std::string& query_string,
                               const ExecutorDeviceType device_type) {
    compare_impl(omnisci_results.get(), query_string, device_type, true);
  }

 private:
  // Moved from TimeGM::parse_fractional_seconds().
  int parse_fractional_seconds(unsigned sfrac, const int ntotal, const SQLTypeInfo& ti) {
    int dimen = ti.get_dimension();
    int nfrac = log10(sfrac) + 1;
    if (ntotal - nfrac > dimen) {
      return 0;
    }
    if (ntotal >= 0 && ntotal < dimen) {
      sfrac *= pow(10, dimen - ntotal);
    } else if (ntotal > dimen) {
      sfrac /= pow(10, ntotal - dimen);
    }
    return sfrac;
  }

  template <class RESULT_SET>
  void compare_impl(const RESULT_SET* omnisci_results,
                    const std::string& sqlite_query_string,
                    const ExecutorDeviceType device_type,
                    const bool timestamp_approx,
                    const bool is_arrow = false) {
    auto const errmsg = ExecutorDeviceType::CPU == device_type
                            ? "CPU: " + sqlite_query_string
                            : "GPU: " + sqlite_query_string;
    connector_.query(sqlite_query_string);
    ASSERT_EQ(connector_.getNumRows(), omnisci_results->rowCount()) << errmsg;
    const int num_rows{static_cast<int>(connector_.getNumRows())};
    if (omnisci_results->definitelyHasNoRows()) {
      ASSERT_EQ(0, num_rows) << errmsg;
      return;
    }
    if (!num_rows) {
      return;
    }
    CHECK_EQ(connector_.getNumCols(), omnisci_results->colCount()) << errmsg;
    const int num_cols{static_cast<int>(connector_.getNumCols())};
    auto row_iterator = omnisci_results->rowIterator(true, true);
    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
      const auto crt_row =
          g_use_row_iterator ? *row_iterator++ : omnisci_results->getNextRow(true, true);
      CHECK(!crt_row.empty()) << errmsg;
      CHECK_EQ(static_cast<size_t>(num_cols), crt_row.size()) << errmsg;
      for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
        const auto ref_col_type = connector_.columnTypes[col_idx];
        const auto omnisci_variant = crt_row[col_idx];
        const auto scalar_omnisci_variant =
            boost::get<ScalarTargetValue>(&omnisci_variant);
        CHECK(scalar_omnisci_variant) << errmsg;
        auto omnisci_ti = omnisci_results->getColType(col_idx);
        const auto omnisci_type = omnisci_ti.get_type();
        checkTypeConsistency(ref_col_type, omnisci_ti);
        const bool ref_is_null = connector_.isNull(row_idx, col_idx);
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
              const auto ref_val = connector_.getData<int64_t>(row_idx, col_idx);
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
            const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
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
              const auto ref_val = connector_.getData<double>(row_idx, col_idx);
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
              const auto ref_val = connector_.getData<float>(row_idx, col_idx);
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
              struct tm tm_struct {
                0
              };
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              auto end_str =
                  strptime(ref_val.c_str(),
                           omnisci_type == kTIMESTAMP ? "%Y-%m-%d %H:%M:%S" : "%Y-%m-%d",
                           &tm_struct);
              // handle fractional seconds
              if (end_str != nullptr && *end_str != '.') {
                if (end_str) {
                  ASSERT_EQ(0, *end_str) << errmsg;
                }
                ASSERT_EQ(ref_val.size(), static_cast<size_t>(end_str - ref_val.c_str()))
                    << errmsg;
              }
              if (dimen > 0 && omnisci_type == kTIMESTAMP) {
                int fs = 0;
                if (*end_str == '.') {
                  end_str++;
                  unsigned int frac_num;
                  int ntotal;
                  sscanf(end_str, "%d%n", &frac_num, &ntotal);
                  fs = parse_fractional_seconds(frac_num, ntotal, omnisci_ti);
                  nsec = timegm(&tm_struct) * pow(10, dimen);
                  nsec += fs;
                } else if (*end_str == '\0') {
                  nsec = timegm(&tm_struct) * pow(10, dimen);
                } else {
                  CHECK(false) << errmsg;
                }
              }
              if (timestamp_approx) {
                // approximate result give 10 second lee way
                ASSERT_NEAR(*omnisci_as_int_p,
                            dimen > 0 ? nsec : timegm(&tm_struct),
                            dimen > 0 ? 10 * pow(10, dimen) : 10)
                    << errmsg;
              } else {
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
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
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
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
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

 private:
  static void checkTypeConsistency(const int ref_col_type,
                                   const SQLTypeInfo& omnisci_ti) {
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

  SqliteConnector connector_;
};

const size_t g_num_rows{10};

}  // namespace

class ExecuteTestBase {
 public:
  static constexpr int TEST_SCHEMA_ID = 1;
  static constexpr int TEST_DB_ID = (TEST_SCHEMA_ID << 24) + 1;

  static constexpr int CALCITE_PORT = 3278;

  static void init() {
    storage_ = std::make_shared<ArrowStorage>(TEST_SCHEMA_ID, "test", TEST_DB_ID);

    SystemParameters system_parameters;
    data_mgr_ = std::make_shared<DataMgr>("", system_parameters, nullptr, false);
    auto* ps_mgr = data_mgr_->getPersistentStorageMgr();
    ps_mgr->registerDataProvider(TEST_SCHEMA_ID, storage_);

    executor_ = std::make_shared<Executor>(0,
                                           data_mgr_.get(),
                                           system_parameters.cuda_block_size,
                                           system_parameters.cuda_grid_size,
                                           system_parameters.max_gpu_slab_size,
                                           "",
                                           "");

    calcite_ = std::make_shared<Calcite>(-1, CALCITE_PORT, "", 1024, 5000, true, "");
    ExtensionFunctionsWhitelist::add(calcite_->getExtensionFunctionWhitelist());
    table_functions::TableFunctionsFactory::init();
    auto udtfs = ThriftSerializers::to_thrift(
        table_functions::TableFunctionsFactory::get_table_funcs(/*is_runtime=*/false));
    std::vector<TUserDefinedFunction> udfs = {};
    calcite_->setRuntimeExtensionFunctions(udfs, udtfs, /*is_runtime=*/false);
  }

  static void createEmptyTestTable() {
    createTable("empty_test_table",
                {{"id", SQLTypeInfo(kINT)},
                 {"x", SQLTypeInfo(kBIGINT)},
                 {"y", SQLTypeInfo(kINT)},
                 {"z", SQLTypeInfo(kSMALLINT)},
                 {"t", SQLTypeInfo(kTINYINT)},
                 {"f", SQLTypeInfo(kFLOAT)},
                 {"d", SQLTypeInfo(kDOUBLE)},
                 {"b", SQLTypeInfo(kBOOLEAN)}});
    sqlite_comparator_.query("DROP TABLE IF EXISTS empty_test_table;");
    std::string create_statement(
        "CREATE TABLE empty_test_table (id int, x bigint, y int, z smallint, t "
        "tinyint, "
        "f float, d double, b boolean);");
    sqlite_comparator_.query(create_statement);
  }

  static void createTestRangesTable() {
    createTable(
        "test_ranges", {{"i", SQLTypeInfo(kINT)}, {"b", SQLTypeInfo(kBIGINT)}}, {2});
    sqlite_comparator_.query("DROP TABLE IF EXISTS test_ranges;");
    sqlite_comparator_.query("CREATE TABLE test_ranges(i INT, b BIGINT);");
    {
      const std::string insert_query{
          "INSERT INTO test_ranges VALUES(2147483647, 9223372036854775806);"};
      sqlite_comparator_.query(insert_query);
      insertCsvValues("test_ranges", "2147483647, 9223372036854775806");
    }
    {
      const std::string insert_query{
          "INSERT INTO test_ranges VALUES(-2147483647, -9223372036854775807);"};
      sqlite_comparator_.query(insert_query);
      insertCsvValues("test_ranges", "-2147483647, -9223372036854775807");
    }
  }

  static void createTestInnerTable() {
    createTable("test_inner",
                {{"x", SQLTypeInfo(kINT, true)},
                 {"y", SQLTypeInfo(kINT)},
                 {"xx", SQLTypeInfo(kSMALLINT)},
                 {"str", dictType()},
                 {"dt", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 0, kNULLT)},
                 {"dt32", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 32, kNULLT)},
                 {"dt16", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 16, kNULLT)},
                 {"ts", SQLTypeInfo(kTIMESTAMP)}},
                {2});
    sqlite_comparator_.query("DROP TABLE IF EXISTS test_inner;");
    sqlite_comparator_.query(
        "CREATE TABLE test_inner(x int not null, y int, xx smallint, str text, dt "
        "DATE, "
        "dt32 DATE, dt16 DATE, ts DATETIME);");
    {
      const std::string insert_query{
          "INSERT INTO test_inner VALUES(7, 43, 7, 'foo', '1999-09-09', '1999-09-09', "
          "'1999-09-09', '2014-12-13 22:23:15');"};
      insertCsvValues("test_inner",
                      "7,43,7,foo,1999-09-09,1999-09-09,1999-09-09,2014-12-13 22:23:15");
      sqlite_comparator_.query(insert_query);
    }
    {
      const std::string insert_query{
          "INSERT INTO test_inner VALUES(-9, 72, -9, 'bars', '2014-12-13', '2014-12-13', "
          "'2014-12-13', '1999-09-09 14:15:16');"};
      insertCsvValues(
          "test_inner",
          "-9,72,-9,bars,2014-12-13,2014-12-13,2014-12-13,1999-09-09 14:15:16");
      sqlite_comparator_.query(insert_query);
    }
  }

  static void createTestTable() {
    auto test_inner = storage_->getTableInfo(TEST_DB_ID, "test_inner");
    auto test_inner_str = storage_->getColumnInfo(*test_inner, "str");
    auto test_inner_str_type = test_inner_str->type;

    createTable("test",
                {{"x", SQLTypeInfo(kINT, true)},
                 {"w", SQLTypeInfo(kTINYINT)},
                 {"y", SQLTypeInfo(kINT)},
                 {"z", SQLTypeInfo(kSMALLINT)},
                 {"t", SQLTypeInfo(kBIGINT)},
                 {"b", SQLTypeInfo(kBOOLEAN)},
                 {"f", SQLTypeInfo(kFLOAT)},
                 {"ff", SQLTypeInfo(kFLOAT)},
                 {"fn", SQLTypeInfo(kFLOAT)},
                 {"d", SQLTypeInfo(kDOUBLE)},
                 {"dn", SQLTypeInfo(kDOUBLE)},
                 {"str", test_inner_str_type},
                 {"null_str", dictType()},
                 {"fixed_str", dictType(2)},
                 {"fixed_null_str", dictType(2)},
                 {"real_str", SQLTypeInfo(kTEXT)},
                 {"shared_dict", test_inner_str_type},
                 {"m", SQLTypeInfo(kTIMESTAMP, 0, 0)},
                 {"m_3", SQLTypeInfo(kTIMESTAMP, 3, 0)},
                 {"m_6", SQLTypeInfo(kTIMESTAMP, 6, 0)},
                 {"m_9", SQLTypeInfo(kTIMESTAMP, 9, 0)},
                 {"n", SQLTypeInfo(kTIME)},
                 {"o", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 0, kNULLT)},
                 {"o1", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 16, kNULLT)},
                 {"o2", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 32, kNULLT)},
                 {"fx", SQLTypeInfo(kSMALLINT)},
                 {"dd", SQLTypeInfo(kDECIMAL, 10, 2, false)},
                 {"dd_notnull", SQLTypeInfo(kDECIMAL, 10, 2, true)},
                 {"ss", dictType()},
                 {"u", SQLTypeInfo(kINT)},
                 {"ofd", SQLTypeInfo(kINT)},
                 {"ufd", SQLTypeInfo(kINT, true)},
                 {"ofq", SQLTypeInfo(kBIGINT)},
                 {"ufq", SQLTypeInfo(kBIGINT, true)},
                 {"smallint_nulls", SQLTypeInfo(kSMALLINT)},
                 {"bn", SQLTypeInfo(kBOOLEAN, true)}},
                {2});
    sqlite_comparator_.query("DROP TABLE IF EXISTS test;");
    sqlite_comparator_.query(
        "CREATE TABLE test(x int not null, w tinyint, y int, z smallint, t bigint, b "
        "boolean, f "
        "float, ff float, fn float, d "
        "double, dn double, str varchar(10), null_str text, fixed_str text, "
        "fixed_null_str text, real_str text, "
        "shared_dict "
        "text, m timestamp(0), m_3 timestamp(3), m_6 timestamp(6), m_9 timestamp(9), n "
        "time(0), o date, o1 date, o2 date, "
        "fx int, dd decimal(10, 2), dd_notnull decimal(10, 2) not "
        "null, ss "
        "text, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not null, "
        "smallint_nulls smallint, bn boolean not null);");

    CHECK_EQ(g_num_rows % 2, size_t(0));
    for (size_t i = 0; i < g_num_rows; ++i) {
      const std::string insert_query{
          "INSERT INTO test VALUES(7, -8, 42, 101, 1001, 't', 1.1, 1.1, null, 2.2, null, "
          "'foo', null, 'foo', null, "
          "'real_foo', 'foo',"
          "'2014-12-13 22:23:15', '2014-12-13 22:23:15.323', '1999-07-11 "
          "14:02:53.874533', "
          "'2006-04-26 "
          "03:49:04.607435125', "
          "'15:13:14', '1999-09-09', '1999-09-09', '1999-09-09', 9, 111.1, 111.1, "
          "'fish', "
          "null, "
          "2147483647, -2147483648, null, -1, 32767, 't');"};
      insertCsvValues(
          "test",
          "7,-8,42,101,1001,true,1.1,1.1,,2.2,,foo,,foo,,real_foo,foo,2014-12-13 "
          "22:23:15,2014-12-13 22:23:15.323,1999-07-11 14:02:53.874533,2006-04-26 "
          "03:49:04.607435125,15:13:14,1999-09-09,1999-09-09,1999-09-09,9,111.1,111.1,"
          "fish,,2147483647,-2147483648,,-1,32767,true");
      sqlite_comparator_.query(insert_query);
    }
    for (size_t i = 0; i < g_num_rows / 2; ++i) {
      const std::string insert_query{
          "INSERT INTO test VALUES(8, -7, 43, -78, 1002, 'f', 1.2, 101.2, -101.2, 2.4, "
          "-2002.4, 'bar', null, 'bar', null, "
          "'real_bar', NULL, '2014-12-13 22:23:15', '2014-12-13 22:23:15.323', "
          "'2014-12-13 "
          "22:23:15.874533', "
          "'2014-12-13 22:23:15.607435763', '15:13:14', NULL, NULL, NULL, NULL, 222.2, "
          "222.2, "
          "null, null, null, "
          "-2147483647, "
          "9223372036854775807, -9223372036854775808, null, 'f');"};
      insertCsvValues(
          "test",
          "8,-7,43,-78,1002,false,1.2,101.2,-101.2,2.4,-2002.4,bar,,bar,,real_bar,,2014-"
          "12-13 22:23:15,2014-12-13 22:23:15.323,2014-12-13 22:23:15.874533,2014-12-13 "
          "22:23:15.607435763,15:13:14,,,,,222.2,222.2,,,,-2147483647,"
          "9223372036854775807,-9223372036854775808,,false");
      sqlite_comparator_.query(insert_query);
    }
    for (size_t i = 0; i < g_num_rows / 2; ++i) {
      const std::string insert_query{
          "INSERT INTO test VALUES(7, -7, 43, 102, 1002, null, 1.3, 1000.3, -1000.3, "
          "2.6, "
          "-220.6, 'baz', null, null, null, "
          "'real_baz', 'baz', '2014-12-14 22:23:15', '2014-12-14 22:23:15.750', "
          "'2014-12-14 22:23:15.437321', "
          "'2014-12-14 22:23:15.934567401', '15:13:14', '1999-09-09', '1999-09-09', "
          "'1999-09-09', 11, "
          "333.3, 333.3, "
          "'boat', null, 1, "
          "-1, 1, -9223372036854775808, 1, 't');"};
      insertCsvValues(
          "test",
          "7,-7,43,102,1002,,1.3,1000.3,-1000.3,2.6,-220.6,baz,,,,real_baz,baz,2014-12-"
          "14 22:23:15,2014-12-14 22:23:15.750,2014-12-14 22:23:15.437321,2014-12-14 "
          "22:23:15.934567401,15:13:14,1999-09-09,1999-09-09,1999-09-09,11,333.3,333.3,"
          "boat,,1,-1,1,-9223372036854775808,1,true");
      sqlite_comparator_.query(insert_query);
    }
  }

  static void createTestEmptyTable() {
    auto test_inner = storage_->getTableInfo(TEST_DB_ID, "test_inner");
    auto test_inner_str = storage_->getColumnInfo(*test_inner, "str");
    auto test_inner_str_type = test_inner_str->type;
    createTable("test_empty",
                {{"x", SQLTypeInfo(kINT, true)},
                 {"w", SQLTypeInfo(kTINYINT)},
                 {"y", SQLTypeInfo(kINT)},
                 {"z", SQLTypeInfo(kSMALLINT)},
                 {"t", SQLTypeInfo(kBIGINT)},
                 {"b", SQLTypeInfo(kBOOLEAN)},
                 {"f", SQLTypeInfo(kFLOAT)},
                 {"ff", SQLTypeInfo(kFLOAT)},
                 {"fn", SQLTypeInfo(kFLOAT)},
                 {"d", SQLTypeInfo(kDOUBLE)},
                 {"dn", SQLTypeInfo(kDOUBLE)},
                 {"str", test_inner_str_type},
                 {"null_str", dictType()},
                 {"fixed_str", dictType(2)},
                 {"fixed_null_str", dictType(2)},
                 {"real_str", SQLTypeInfo(kTEXT)},
                 {"shared_dict", test_inner_str_type},
                 {"m", SQLTypeInfo(kTIMESTAMP, 0, 0)},
                 {"n", SQLTypeInfo(kTIME)},
                 {"o", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 0, kNULLT)},
                 {"o1", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 16, kNULLT)},
                 {"o2", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 32, kNULLT)},
                 {"fx", SQLTypeInfo(kSMALLINT)},
                 {"dd", SQLTypeInfo(kDECIMAL, 10, 2, false)},
                 {"dd_notnull", SQLTypeInfo(kDECIMAL, 10, 2, true)},
                 {"ss", dictType()},
                 {"u", SQLTypeInfo(kINT)},
                 {"ofd", SQLTypeInfo(kINT)},
                 {"ufd", SQLTypeInfo(kINT, true)},
                 {"ofq", SQLTypeInfo(kBIGINT)},
                 {"ufq", SQLTypeInfo(kBIGINT, true)}},
                {2});

    sqlite_comparator_.query("DROP TABLE IF EXISTS test_empty;");
    sqlite_comparator_.query(
        "CREATE TABLE test_empty(x int not null, w tinyint, y int, z smallint, t "
        "bigint, b boolean, f float, ff float, fn float, d "
        "double, dn double, str varchar(10), null_str text, fixed_str text, "
        "fixed_null_str text, real_str text, "
        "shared_dict "
        "text, m timestamp(0), n "
        "time(0), o date, o1 date, o2 date, "
        "fx int, dd decimal(10, 2), dd_notnull decimal(10, 2) not "
        "null, ss "
        "text, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not null);");
  }

  static void createTestOneRowTable() {
    auto test_inner = storage_->getTableInfo(TEST_DB_ID, "test_inner");
    auto test_inner_str = storage_->getColumnInfo(*test_inner, "str");
    auto test_inner_str_type = test_inner_str->type;
    createTable("test_one_row",
                {{"x", SQLTypeInfo(kINT, true)},
                 {"w", SQLTypeInfo(kTINYINT)},
                 {"y", SQLTypeInfo(kINT)},
                 {"z", SQLTypeInfo(kSMALLINT)},
                 {"t", SQLTypeInfo(kBIGINT)},
                 {"b", SQLTypeInfo(kBOOLEAN)},
                 {"f", SQLTypeInfo(kFLOAT)},
                 {"ff", SQLTypeInfo(kFLOAT)},
                 {"fn", SQLTypeInfo(kFLOAT)},
                 {"d", SQLTypeInfo(kDOUBLE)},
                 {"dn", SQLTypeInfo(kDOUBLE)},
                 {"str", test_inner_str_type},
                 {"null_str", dictType()},
                 {"fixed_str", dictType(2)},
                 {"fixed_null_str", dictType(2)},
                 {"real_str", SQLTypeInfo(kTEXT)},
                 {"shared_dict", test_inner_str_type},
                 {"m", SQLTypeInfo(kTIMESTAMP, 0, 0)},
                 {"n", SQLTypeInfo(kTIME)},
                 {"o", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 0, kNULLT)},
                 {"o1", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 16, kNULLT)},
                 {"o2", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 32, kNULLT)},
                 {"fx", SQLTypeInfo(kSMALLINT)},
                 {"dd", SQLTypeInfo(kDECIMAL, 10, 2, false)},
                 {"dd_notnull", SQLTypeInfo(kDECIMAL, 10, 2, true)},
                 {"ss", dictType()},
                 {"u", SQLTypeInfo(kINT)},
                 {"ofd", SQLTypeInfo(kINT)},
                 {"ufd", SQLTypeInfo(kINT, true)},
                 {"ofq", SQLTypeInfo(kBIGINT)},
                 {"ufq", SQLTypeInfo(kBIGINT, true)}},
                {2});

    sqlite_comparator_.query("DROP TABLE IF EXISTS test_one_row;");
    sqlite_comparator_.query(
        "CREATE TABLE test_one_row(x int not null, w tinyint, y int, z smallint, t "
        "bigint, b "
        "boolean, "
        "f "
        "float, ff float, fn float, d "
        "double, dn double, str varchar(10), null_str text, fixed_str text, "
        "fixed_null_str text, real_str text, "
        "shared_dict "
        "text, m timestamp(0), n "
        "time(0), o date, o1 date, o2 date, "
        "fx int, dd decimal(10, 2), dd_notnull decimal(10, 2) not "
        "null, ss "
        "text, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not null);");

    {
      insertCsvValues("test_one_row",
                      "8,-8,43,-78,1002,false,1.2,101.2,-101.2,2.4,-2002.4,bar,,bar,,"
                      "real_bar,,2014-12-13 22:23:15,15:13:14,,,,, 222.2,"
                      "222.2,,,,2147483647,9223372036854775807,-9223372036854775808");
      const std::string insert_query{
          "INSERT INTO test_one_row VALUES(8, -8, 43, -78, 1002, 'f', 1.2, 101.2, "
          "-101.2, "
          "2.4, "
          "-2002.4, 'bar', null, 'bar', null, "
          "'real_bar', NULL, '2014-12-13 22:23:15', "
          "'15:13:14', NULL, NULL, NULL, NULL, 222.2, 222.2, "
          "null, null, null, "
          "-2147483647, "
          "9223372036854775807, -9223372036854775808);"};
      sqlite_comparator_.query(insert_query);
    }
  }

  static constexpr size_t g_array_test_row_count{20};

  static void import_array_test(const std::string& table_name) {
    CHECK_EQ(size_t(0), g_array_test_row_count % 4);
    auto tinfo = storage_->getTableInfo(TEST_DB_ID, table_name);
    ASSERT(tinfo);
    auto col_infos = storage_->listColumns(*tinfo);
    /*
        auto& cat = QR::get()->getSession()->getCatalog();
        const auto td = cat.getMetadataForTable(table_name);
        CHECK(td);
        auto loader = QR::get()->getLoader(td);
        std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
        const auto col_descs =
            cat.getAllColumnMetadataForTable(td->tableId, false, false, false);
        for (const auto cd : col_descs) {
          import_buffers.emplace_back(new import_export::TypedImportBuffer(
              cd,
              cd->columnType.get_compression() == kENCODING_DICT
                  ?
       cat.getMetadataForDict(cd->columnType.get_comp_param())->stringDict.get() :
       nullptr));
        }
        import_export::CopyParams copy_params;
        copy_params.array_begin = '{';
        copy_params.array_end = '}';
    */
    std::stringstream json_ss;
    for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
      json_ss << "{";
      for (size_t col_idx = 0; col_idx < col_infos.size(); ++col_idx) {
        auto col_info = col_infos[col_idx];
        if (col_info->is_rowid) {
          continue;
        }
        if (col_idx) {
          json_ss << ", ";
        }
        json_ss << "\"" << col_info->name << "\" : ";
        const auto& ti = col_info->type;
        switch (ti.get_type()) {
          case kINT:
            json_ss << (7 + row_idx);
            break;
          case kARRAY: {
            const auto& elem_ti = ti.get_elem_type();
            std::vector<std::string> array_elems;
            switch (elem_ti.get_type()) {
              case kBOOLEAN: {
                for (size_t i = 0; i < 3; ++i) {
                  if (row_idx % 2) {
                    array_elems.emplace_back("true");
                    array_elems.emplace_back("false");
                  } else {
                    array_elems.emplace_back("false");
                    array_elems.emplace_back("true");
                  }
                }
                break;
              }
              case kTINYINT:
                for (size_t i = 0; i < 3; ++i) {
                  array_elems.push_back(std::to_string(row_idx + i + 1));
                }
                break;
              case kSMALLINT:
                for (size_t i = 0; i < 3; ++i) {
                  array_elems.push_back(std::to_string(row_idx + i + 1));
                }
                break;
              case kINT:
                for (size_t i = 0; i < 3; ++i) {
                  array_elems.push_back(std::to_string((row_idx + i + 1) * 10));
                }
                break;
              case kBIGINT:
                for (size_t i = 0; i < 3; ++i) {
                  array_elems.push_back(std::to_string((row_idx + i + 1) * 100));
                }
                break;
              case kTEXT:
                for (size_t i = 0; i < 3; ++i) {
                  std::string val(2, 'a' + row_idx + i);
                  array_elems.push_back("\""s + val + "\""s);
                }
                break;
              case kFLOAT:
                for (size_t i = 0; i < 3; ++i) {
                  array_elems.emplace_back(std::to_string(row_idx + i + 1) + "." +
                                           std::to_string(row_idx + i + 1));
                }
                break;
              case kDOUBLE:
                for (size_t i = 0; i < 3; ++i) {
                  array_elems.emplace_back(std::to_string(11 * (row_idx + i + 1)) + "." +
                                           std::to_string(row_idx + i + 1));
                }
                break;
              case kDECIMAL:
                for (size_t i = 0; i < 3; ++i) {
                  array_elems.emplace_back(std::to_string(11 * (row_idx + i + 1)) + "." +
                                           std::to_string(row_idx + i + 1));
                }
                break;
              default:
                CHECK(false);
            }
            json_ss << "[" << boost::algorithm::join(array_elems, ", ") << "]";
            break;
          }
          case kTEXT:
            json_ss << "\"real_str" << row_idx << "\"";
            break;
          default:
            CHECK(false);
        }
      }
      json_ss << "}\n";
    }
    auto json_data = json_ss.str();
    insertJsonValues(table_name, json_data);
  }

  static void createTestArrayTable() {
    createTable("array_test",
                {{"x", SQLTypeInfo(kINT, true)},
                 {"arr_i16", arrayType(kSMALLINT)},
                 {"arr_i32", arrayType(kINT)},
                 {"arr_i64", arrayType(kBIGINT)},
                 {"arr_str", arrayType(kTEXT)},
                 {"arr_float", arrayType(kFLOAT)},
                 {"arr_double", arrayType(kDOUBLE)},
                 {"arr_bool", arrayType(kBOOLEAN)},
                 {"arr_decimal", decimalArrayType(18, 6)},
                 {"real_str", SQLTypeInfo(kTEXT)},
                 {"arr3_i8", arrayType(kTINYINT, 3)},
                 {"arr3_i16", arrayType(kSMALLINT, 3)},
                 {"arr3_i32", arrayType(kINT, 3)},
                 {"arr3_i64", arrayType(kBIGINT, 3)},
                 {"arr3_float", arrayType(kFLOAT, 3)},
                 {"arr3_double", arrayType(kDOUBLE, 3)},
                 {"arr6_bool", arrayType(kBOOLEAN, 6)},
                 {"arr3_decimal", decimalArrayType(18, 6, 3)}});
    import_array_test("array_test");
  }

  static void importCoalesceColsTestTable(const int id) {
    const std::string table_name = "coalesce_cols_test_" + std::to_string(id);
    createTable(table_name,
                {{"x", SQLTypeInfo(kINT, true)},
                 {"y", SQLTypeInfo(kINT)},
                 {"str", dictType()},
                 {"dup_str", dictType()},
                 {"d", SQLTypeInfo(kDATE)},
                 {"t", SQLTypeInfo(kTIME)},
                 {"tz", SQLTypeInfo(kTIMESTAMP)},
                 {"dn", SQLTypeInfo(kDECIMAL, 5, 0, false)}},
                {id == 2 ? 2ULL : 20ULL});
    sqlite_comparator_.query("DROP TABLE IF EXISTS " + table_name + ";");
    sqlite_comparator_.query("CREATE TABLE " + table_name +
                             "(x int not null, y int, str text, dup_str text, d date, t "
                             "time, tz timestamp, dn decimal(5));");
    TestHelpers::ValuesGenerator gen(table_name);
    std::stringstream ss;
    for (int i = 0; i < 5; i++) {
      const auto insert_query = gen(i,
                                    20 - i,
                                    "'test'",
                                    "'test'",
                                    "'2018-01-01'",
                                    "'12:34:56'",
                                    "'2018-01-01 12:34:56'",
                                    i * 1.1);
      sqlite_comparator_.query(insert_query);
      ss << i << "," << (20 - i) << ",test,test,2018-01-01,12:34:56,2018-01-01 12:34:56,"
         << int(i * 1.1) << std::endl;
    }
    for (size_t i = 5; i < 10; i++) {
      const auto insert_query = gen(i,
                                    20 - i,
                                    "'test1'",
                                    "'test1'",
                                    "'2017-01-01'",
                                    "'12:34:00'",
                                    "'2017-01-01 12:34:56'",
                                    i * 1.1);
      sqlite_comparator_.query(insert_query);
      ss << i << "," << (20 - i)
         << ",test1,test1,2017-01-01,12:34:00,2017-01-01 12:34:56," << int(i * 1.1)
         << std::endl;
    }
    if (id > 0) {
      for (size_t i = 10; i < 15; i++) {
        const auto insert_query = gen(i,
                                      20 - i,
                                      "'test2'",
                                      "'test2'",
                                      "'2016-01-01'",
                                      "'12:00:56'",
                                      "'2016-01-01 12:34:56'",
                                      i * 1.1);
        sqlite_comparator_.query(insert_query);
        ss << i << "," << (20 - i)
           << ",test2,test2,2016-01-01,12:00:56,2016-01-01 12:34:56," << int(i * 1.1)
           << std::endl;
      }
    }
    if (id > 1) {
      for (size_t i = 15; i < 20; i++) {
        const auto insert_query = gen(i,
                                      20 - i,
                                      "'test3'",
                                      "'test3'",
                                      "'2015-01-01'",
                                      "'10:34:56'",
                                      "'2015-01-01 12:34:56'",
                                      i * 1.1);
        sqlite_comparator_.query(insert_query);
        ss << i << "," << (20 - i)
           << ",test3,test3,2015-01-01,10:34:56,2015-01-01 12:34:56," << int(i * 1.1)
           << std::endl;
      }
    }
    insertCsvValues(table_name, ss.str());
  }

  static void createProjTopTable() {
    createTable("proj_top", {{"str", SQLTypeInfo(kTEXT)}, {"x", SQLTypeInfo(kINT)}});
    insertCsvValues("proj_top", "a,7\nb,6\nc,5");
    sqlite_comparator_.query("DROP TABLE IF EXISTS proj_top;");
    sqlite_comparator_.query("CREATE TABLE proj_top(str TEXT, x INT);");
    sqlite_comparator_.query("INSERT INTO proj_top VALUES('a', 7);");
    sqlite_comparator_.query("INSERT INTO proj_top VALUES('b', 6);");
    sqlite_comparator_.query("INSERT INTO proj_top VALUES('c', 5);");
  }

  static void createJoinTestTable() {
    createTable("join_test",
                {{"x", SQLTypeInfo(kINT, true)},
                 {"y", SQLTypeInfo(kINT)},
                 {"str", dictType()},
                 {"dup_str", dictType()}},
                {2});
    insertCsvValues("join_test", "7,43,foo,foo\n8,,bar,foo\n9,,baz,bar");
    sqlite_comparator_.query("DROP TABLE IF EXISTS join_test;");
    sqlite_comparator_.query(
        "CREATE TABLE join_test(x int not null, y int, str text, dup_str text);");
    sqlite_comparator_.query("INSERT INTO join_test VALUES(7, 43, 'foo', 'foo');");
    sqlite_comparator_.query("INSERT INTO join_test VALUES(8, null, 'bar', 'foo');");
    sqlite_comparator_.query("INSERT INTO join_test VALUES(9, null, 'baz', 'bar');");
  }

  static void createQueryRewriteTestTable() {
    createTable(
        "query_rewrite_test", {{"x", SQLTypeInfo(kINT)}, {"str", dictType()}}, {2});
    sqlite_comparator_.query("DROP TABLE IF EXISTS query_rewrite_test;");
    sqlite_comparator_.query("CREATE TABLE query_rewrite_test(x int, str text);");
    std::stringstream ss;
    for (size_t i = 1; i <= 30; ++i) {
      for (size_t j = 1; j <= i % 2 + 1; ++j) {
        const std::string insert_query{"INSERT INTO query_rewrite_test VALUES(" +
                                       std::to_string(i) + ", 'str" + std::to_string(i) +
                                       "');"};
        sqlite_comparator_.query(insert_query);
        ss << i << ",str" << i << std::endl;
      }
    }
    insertCsvValues("query_rewrite_test", ss.str());
  }

  static void createEmpTable() {
    createTable("emp",
                {{"empno", SQLTypeInfo(kINT)},
                 {"ename", dictType()},
                 {"deptno", SQLTypeInfo(kINT)}},
                {2});
    insertCsvValues("emp", "1,Brock,10\n2,Bill,20\n3,Julia,60\n4,David,10");
    sqlite_comparator_.query("DROP TABLE IF EXISTS emp;");
    sqlite_comparator_.query(
        "CREATE TABLE emp(empno INT, ename TEXT NOT NULL, deptno INT);");
    sqlite_comparator_.query("INSERT INTO emp VALUES(1, 'Brock', 10);");
    sqlite_comparator_.query("INSERT INTO emp VALUES(2, 'Bill', 20);");
    sqlite_comparator_.query("INSERT INTO emp VALUES(3, 'Julia', 60);");
    sqlite_comparator_.query("INSERT INTO emp VALUES(4, 'David', 10);");
  }

  void createTestLotsColsTable() {
    const size_t num_columns = 50;
    const std::string table_name("test_lots_cols");
    const std::string drop_table("DROP TABLE IF EXISTS " + table_name + ";");
    sqlite_comparator_.query(drop_table);
    std::string create_query("CREATE TABLE " + table_name + "(");
    std::string insert_query1("INSERT INTO " + table_name + " VALUES (");
    std::string insert_query2(insert_query1);
    std::vector<ArrowStorage::ColumnDescription> cols;
    std::string csv1;
    std::string csv2;

    for (size_t i = 0; i < num_columns - 1; i++) {
      create_query += ("x" + std::to_string(i) + " INTEGER, ");
      insert_query1 += (std::to_string(i) + ", ");
      insert_query2 += (std::to_string(10000 + i) + ", ");
      cols.push_back({"x"s + std::to_string(i), SQLTypeInfo(kINT)});
      csv1 += std::to_string(i) + ",";
      csv2 += std::to_string(10000 + i) + ",";
    }
    create_query += "real_str TEXT";
    insert_query1 += "'real_foo');";
    insert_query2 += "'real_bar');";
    cols.push_back({"real_str", SQLTypeInfo(kTEXT)});
    csv1 += "real_foo";
    csv2 += "real_bar";

    createTable(table_name, cols, {2});
    sqlite_comparator_.query(create_query + ");");

    for (size_t i = 0; i < 10; i++) {
      insertCsvValues(table_name, i % 2 ? csv2 : csv1);
      sqlite_comparator_.query(i % 2 ? insert_query2 : insert_query1);
    }
  }

  static void createTestDateTimeTable() {
    createTable("test_date_time",
                {{"dt", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 0, kNULLT)}},
                {2});
    insertCsvValues("test_date_time", "1963-05-07\n1968-04-22\n1970-01-01\n1980-11-28");
  }

  static void createBigDecimalRangeTestTable() {
    createTable("big_decimal_range_test",
                {{"d", SQLTypeInfo(kDECIMAL, 14, 2, false)},
                 {"d1", SQLTypeInfo(kDECIMAL, 17, 11)}},
                {2});
    insertCsvValues("big_decimal_range_test",
                    "-40840124.400000,1.3\n59016609.300000,1.3\n-999999999999.99,1.3");
    sqlite_comparator_.query("DROP TABLE IF EXISTS big_decimal_range_test;");
    sqlite_comparator_.query(
        "CREATE TABLE big_decimal_range_test(d DECIMAL(14, 2), d1 DECIMAL(17,11));");
    sqlite_comparator_.query(
        "INSERT INTO big_decimal_range_test VALUES(-40840124.400000, 1.3);");
    sqlite_comparator_.query(
        "INSERT INTO big_decimal_range_test VALUES(59016609.300000, 1.3);");
    sqlite_comparator_.query(
        "INSERT INTO big_decimal_range_test VALUES(-999999999999.99, 1.3);");
  }

  static void createAndPopulateTestTables() {
    createTestInnerTable();
    createTestTable();
    createTestEmptyTable();
    createTestOneRowTable();
    createEmptyTestTable();
    createTestRangesTable();
    createTestArrayTable();
    importCoalesceColsTestTable(0);
    importCoalesceColsTestTable(1);
    importCoalesceColsTestTable(2);
    createProjTopTable();
    createJoinTestTable();
    createQueryRewriteTestTable();
    createEmpTable();
    createTestDateTimeTable();
    createBigDecimalRangeTestTable();
  }

  static void reset() {
    data_mgr_.reset();
    storage_.reset();
    executor_.reset();
    calcite_.reset();
  }

  static bool gpusPresent() { return data_mgr_->gpusPresent(); }

  static void printStats() {
    std::cout << "Total schema to JSON time: " << (schema_to_json_time_ / 1000) << "ms."
              << std::endl;
    std::cout << "Total Calcite parsing time: " << (calcite_time_ / 1000) << "ms."
              << std::endl;
    std::cout << "Total execution time: " << (execution_time_ / 1000) << "ms."
              << std::endl;
  }

 protected:
  static void createTable(
      const std::string& table_name,
      const std::vector<ArrowStorage::ColumnDescription>& columns,
      const ArrowStorage::TableOptions& options = ArrowStorage::TableOptions()) {
    storage_->createTable(table_name, columns, options);
  }

  static void dropTable(const std::string& table_name) {
    storage_->dropTable(table_name);
  }

  static void insertCsvValues(const std::string& table_name, const std::string& values) {
    ArrowStorage::CsvParseOptions parse_options;
    parse_options.header = false;
    storage_->appendCsvData(values, table_name, parse_options);
  }

  static void insertJsonValues(const std::string& table_name, const std::string& values) {
    storage_->appendJsonData(values, table_name);
  }

  static SQLTypeInfo arrayType(SQLTypes st, int size = 0) {
    SQLTypeInfo res(kARRAY, (st == kTEXT) ? kENCODING_DICT : kENCODING_NONE, 0, st);
    if (size) {
      res.set_size(size * res.get_elem_type().get_size());
    }
    return res;
  }

  static SQLTypeInfo decimalArrayType(int dimension, int scale, int size = 0) {
    SQLTypeInfo res(kARRAY, dimension, scale, false, kENCODING_NONE, 0, kDECIMAL);
    if (size) {
      res.set_size(size * res.get_elem_type().get_size());
    }
    return res;
  }

  static SQLTypeInfo dictType(int size = 4, bool notnull = false, int comp = 0) {
    SQLTypeInfo res(kTEXT, notnull, kENCODING_DICT);
    res.set_size(size);
    res.set_comp_param(comp);
    return res;
  }

  static ExecutionResult runSqlQuery(const std::string& sql,
                                     const CompilationOptions& co,
                                     const ExecutionOptions& eo) {
    std::string schema_json;
    std::string query_ra;
    ExecutionResult res;

    schema_to_json_time_ += measure<std::chrono::microseconds>::execution(
        [&]() { schema_json = schema_to_json(storage_); });

    calcite_time_ += measure<std::chrono::microseconds>::execution([&]() {
      query_ra =
          calcite_->process("admin", "test_db", pg_shim(sql), schema_json, "", {}, true)
              .plan_result;
    });

    execution_time_ += measure<std::chrono::microseconds>::execution([&]() {
      auto dag =
          std::make_unique<RelAlgDagBuilder>(query_ra, TEST_DB_ID, storage_, nullptr);
      auto ra_executor =
          RelAlgExecutor(executor_.get(), TEST_DB_ID, storage_, std::move(dag));
      res = ra_executor.executeRelAlgQuery(co, eo, false, nullptr);
    });

    return res;
  }

  static ExecutionResult runSqlQuery(const std::string& sql,
                                     ExecutorDeviceType device_type,
                                     const ExecutionOptions& eo) {
    return runSqlQuery(sql, getCompilationOptions(device_type), eo);
  }

  static ExecutionResult runSqlQuery(const std::string& sql,
                                     ExecutorDeviceType device_type,
                                     bool allow_loop_joins) {
    return runSqlQuery(sql, device_type, getExecutionOptions(allow_loop_joins));
  }

  static ExecutionOptions getExecutionOptions(bool allow_loop_joins,
                                              bool just_explain = false) {
    return {g_enable_columnar_output,
            true,
            just_explain,
            allow_loop_joins,
            false,
            false,
            false,
            false,
            10000,
            false,
            false,
            g_gpu_mem_limit_percent,
            false,
            1000};
  }

  static CompilationOptions getCompilationOptions(ExecutorDeviceType device_type) {
    auto co = CompilationOptions::defaults(device_type);
    co.hoist_literals = g_hoist_literals;
    return co;
  }

  static std::shared_ptr<ResultSet> run_multiple_agg(const string& query_str,
                                                     const ExecutorDeviceType device_type,
                                                     const bool allow_loop_joins = true) {
    return runSqlQuery(query_str, device_type, allow_loop_joins).getRows();
  }

  static TargetValue run_simple_agg(const string& query_str,
                                    const ExecutorDeviceType device_type,
                                    const bool allow_loop_joins = true) {
    auto rows = run_multiple_agg(query_str, device_type, allow_loop_joins);
    auto crt_row = rows->getNextRow(true, true);
    CHECK_EQ(size_t(1), crt_row.size()) << query_str;
    return crt_row[0];
  }

  static void c(const std::string& query_string, const ExecutorDeviceType device_type) {
    sqlite_comparator_.compare(
        run_multiple_agg(query_string, device_type), query_string, device_type);
  }

  static void c(const std::string& query_string,
                const std::string& sqlite_query_string,
                const ExecutorDeviceType device_type) {
    sqlite_comparator_.compare(
        run_multiple_agg(query_string, device_type), sqlite_query_string, device_type);
  }

  /* timestamp approximate checking for NOW() */
  static void cta(const std::string& query_string, const ExecutorDeviceType device_type) {
    sqlite_comparator_.compare_timstamp_approx(
        run_multiple_agg(query_string, device_type), query_string, device_type);
  }

  static void c_arrow(const std::string& query_string,
                      const ExecutorDeviceType device_type) {
    auto results = run_multiple_agg(query_string, device_type);
    auto arrow_omnisci_results = result_set_arrow_loopback(nullptr, results, device_type);
    sqlite_comparator_.compare_arrow_output(
        arrow_omnisci_results, query_string, device_type);
  }

  static void check_date_trunc_groups(const ResultSet& rows) {
    {
      const auto crt_row = rows.getNextRow(true, true);
      CHECK(!crt_row.empty());
      CHECK_EQ(size_t(3), crt_row.size());
      const auto sv0 = v<int64_t>(crt_row[0]);
      ASSERT_EQ(int64_t(936144000), sv0);
      const auto sv1 = boost::get<std::string>(v<NullableString>(crt_row[1]));
      ASSERT_EQ("foo", sv1);
      const auto sv2 = v<int64_t>(crt_row[2]);
      ASSERT_EQ(static_cast<int64_t>(g_num_rows), sv2);
    }
    {
      const auto crt_row = rows.getNextRow(true, true);
      CHECK(!crt_row.empty());
      CHECK_EQ(size_t(3), crt_row.size());
      const auto sv0 = v<int64_t>(crt_row[0]);
      ASSERT_EQ(inline_int_null_val(rows.getColType(0)), sv0);
      const auto sv1 = boost::get<std::string>(v<NullableString>(crt_row[1]));
      ASSERT_EQ("bar", sv1);
      const auto sv2 = v<int64_t>(crt_row[2]);
      ASSERT_EQ(static_cast<int64_t>(g_num_rows) / 2, sv2);
    }
    {
      const auto crt_row = rows.getNextRow(true, true);
      CHECK(!crt_row.empty());
      CHECK_EQ(size_t(3), crt_row.size());
      const auto sv0 = v<int64_t>(crt_row[0]);
      ASSERT_EQ(int64_t(936144000), sv0);
      const auto sv1 = boost::get<std::string>(v<NullableString>(crt_row[1]));
      ASSERT_EQ("baz", sv1);
      const auto sv2 = v<int64_t>(crt_row[2]);
      ASSERT_EQ(static_cast<int64_t>(g_num_rows) / 2, sv2);
    }
    const auto crt_row = rows.getNextRow(true, true);
    CHECK(crt_row.empty());
  }

  static void check_one_date_trunc_group(const ResultSet& rows, const int64_t ref_ts) {
    const auto crt_row = rows.getNextRow(true, true);
    ASSERT_EQ(size_t(1), crt_row.size());
    const auto actual_ts = v<int64_t>(crt_row[0]);
    ASSERT_EQ(ref_ts, actual_ts);
    const auto empty_row = rows.getNextRow(true, true);
    ASSERT_TRUE(empty_row.empty());
  }

  static void check_one_date_trunc_group_with_agg(const ResultSet& rows,
                                                  const int64_t ref_ts,
                                                  const int64_t ref_agg) {
    const auto crt_row = rows.getNextRow(true, true);
    ASSERT_EQ(size_t(2), crt_row.size());
    const auto actual_ts = v<int64_t>(crt_row[0]);
    ASSERT_EQ(ref_ts, actual_ts);
    const auto actual_agg = v<int64_t>(crt_row[1]);
    ASSERT_EQ(ref_agg, actual_agg);
    const auto empty_row = rows.getNextRow(true, true);
    ASSERT_TRUE(empty_row.empty());
  }

  // Example: "1969-12-31 23:59:59.999999" -> -1
  // The number of fractional digits must be 0, 3, 6, or 9.
  static int64_t timestampToInt64(char const* timestr, ExecutorDeviceType const dt) {
    constexpr int max = 128;
    char query[max];
    unsigned const dim = strlen(timestr) == 19 ? 0 : strlen(timestr) - 20;
    int const n = snprintf(query, max, "SELECT TIMESTAMP(%d) '%s';", dim, timestr);
    CHECK_LT(0, n);
    CHECK_LT(n, max);
    return v<int64_t>(run_simple_agg(query, dt));
  }

  static int64_t dateadd(char const* unit,
                         int const num,
                         char const* timestr,
                         ExecutorDeviceType const dt) {
    constexpr int max = 128;
    char query[max];
    unsigned const dim = strlen(timestr) == 19 ? 0 : strlen(timestr) - 20;
    int const n =
        snprintf(query,
                 max,
                 // Cast from TIMESTAMP(6) to TEXT not supported
                 // "SELECT CAST(DATEADD('%s', %d, TIMESTAMP(%d) '%s') AS TEXT);",
                 "SELECT DATEADD('%s', %d, TIMESTAMP(%d) '%s');",
                 unit,
                 num,
                 dim,
                 timestr);
    CHECK_LT(0, n);
    CHECK_LT(n, max);
    return v<int64_t>(run_simple_agg(query, dt));
  }

  static int64_t datediff(char const* unit,
                          char const* start,
                          char const* end,
                          ExecutorDeviceType const dt) {
    constexpr int max = 128;
    char query[max];
    unsigned const dim_start = strlen(start) == 19 ? 0 : strlen(start) - 20;
    unsigned const dim_end = strlen(end) == 19 ? 0 : strlen(end) - 20;
    int const n =
        snprintf(query,
                 max,
                 "SELECT DATEDIFF('%s', TIMESTAMP(%d) '%s', TIMESTAMP(%d) '%s');",
                 unit,
                 dim_start,
                 start,
                 dim_end,
                 end);
    CHECK_LT(0, n);
    CHECK_LT(n, max);
    return v<int64_t>(run_simple_agg(query, dt));
  }

  static std::string date_trunc(std::string const& unit,
                                char const* ts,
                                ExecutorDeviceType dt) {
    std::string const query =
        "SELECT CAST(DATE_TRUNC('" + unit + "', TIMESTAMP '" + ts + "') AS TEXT);";
    return boost::get<std::string>(v<NullableString>(run_simple_agg(query, dt)));
  }

  static std::shared_ptr<DataMgr> data_mgr_;
  static std::shared_ptr<ArrowStorage> storage_;
  static std::shared_ptr<Executor> executor_;
  static std::shared_ptr<Calcite> calcite_;
  static SQLiteComparator sqlite_comparator_;
  static int64_t schema_to_json_time_;
  static int64_t calcite_time_;
  static int64_t execution_time_;
};

std::shared_ptr<DataMgr> ExecuteTestBase::data_mgr_;
std::shared_ptr<ArrowStorage> ExecuteTestBase::storage_;
std::shared_ptr<Executor> ExecuteTestBase::executor_;
std::shared_ptr<Calcite> ExecuteTestBase::calcite_;
SQLiteComparator ExecuteTestBase::sqlite_comparator_;
int64_t ExecuteTestBase::schema_to_json_time_ = 0;
int64_t ExecuteTestBase::calcite_time_ = 0;
int64_t ExecuteTestBase::execution_time_ = 0;

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(ExecuteTestBase::gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

class Distributed50 : public ExecuteTestBase, public ::testing::Test {};

TEST_F(Distributed50, FailOver) {
  createTable("dist5", {{"col1", dictType()}});

  auto dt = ExecutorDeviceType::CPU;

  EXPECT_NO_THROW(insertCsvValues("dist5", "t1"));
  ASSERT_EQ(1, v<int64_t>(run_simple_agg("SELECT count(*) FROM dist5;", dt)));

  EXPECT_NO_THROW(insertCsvValues("dist5", "t2"));
  ASSERT_EQ(2, v<int64_t>(run_simple_agg("SELECT count(*) FROM dist5;", dt)));

  EXPECT_NO_THROW(insertCsvValues("dist5", "t3"));
  ASSERT_EQ(3, v<int64_t>(run_simple_agg("SELECT count(*) FROM dist5;", dt)));

  dropTable("dist5");
}

class Errors : public ExecuteTestBase, public ::testing::Test {};

TEST_F(Errors, InvalidQueries) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_ANY_THROW(run_multiple_agg(
        "SELECT * FROM test WHERE 1 = 2 AND ( 1 = 2 and 3 = 4 limit 100);", dt));
    EXPECT_ANY_THROW(run_multiple_agg("SET x = y;", dt));
  }
}

class Insert : public ExecuteTestBase, public ::testing::Test {};

TEST_F(Insert, NullArrayNullEmpty) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    createTable("table_array_empty", {{"val", arrayType(kINT)}});
    EXPECT_NO_THROW(insertJsonValues("table_array_empty", "{\"val\": []}"));
    EXPECT_NO_THROW(run_simple_agg("SELECT * from table_array_empty;", dt));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT CARDINALITY(val) from table_array_empty limit 1;", dt)));
    dropTable("table_array_empty");

    createTable("table_array_fixlen_text", {{"strings", arrayType(kTEXT, 2)}});
    EXPECT_NO_THROW(insertJsonValues("table_array_fixlen_text",
                                     R"___({"strings": null}
{"strings": []}
{"strings": [null, null]}
{"strings": ["a", "b"]})___"));
    ASSERT_EQ(
        4,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM table_array_fixlen_text;", dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM table_array_fixlen_text WHERE strings IS NULL;", dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM table_array_fixlen_text WHERE strings IS NOT NULL;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM table_array_fixlen_text WHERE strings[1] IS NOT NULL;",
            dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM table_array_fixlen_text WHERE strings[2] IS NULL;",
            dt)));
    dropTable("table_array_fixlen_text");

    createTable("table_array_with_nulls",
                {{"i", SQLTypeInfo(kSMALLINT)},
                 {"sia", arrayType(kSMALLINT)},
                 {"fa2", arrayType(kFLOAT, 2)}});

    EXPECT_NO_THROW(insertJsonValues("table_array_with_nulls",
                                     R"___({"i": 1, "sia": [1, 1], "fa2": [1.0, 1.0]}
{"i": 2, "sia": [null, 2], "fa2": [null, 2.0]}
{"i": 3, "sia": [3, null], "fa2": [3.0, null]}
{"i": 4, "sia": [null, null], "fa2": [null, null]}
{"i": 5, "sia": null, "fa2": null}
{"i": 6, "sia": [], "fa2": null}
{"i": 7, "sia": [null, null], "fa2": [null, null]})___"));

    ASSERT_EQ(1,
              v<int64_t>(
                  run_simple_agg("SELECT MIN(sia[1]) FROM table_array_with_nulls;", dt)));
    ASSERT_EQ(3,
              v<int64_t>(
                  run_simple_agg("SELECT MAX(sia[1]) FROM table_array_with_nulls;", dt)));
    ASSERT_EQ(
        5,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM table_array_with_nulls WHERE sia[2] IS NULL;", dt)));
    ASSERT_EQ(
        3.0,
        v<float>(run_simple_agg("SELECT MAX(fa2[1]) FROM table_array_with_nulls;", dt)));
    ASSERT_EQ(
        2.0,
        v<float>(run_simple_agg("SELECT MAX(fa2[2]) FROM table_array_with_nulls;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM table_array_with_nulls WHERE fa2[1] IS NOT NULL;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM table_array_with_nulls WHERE sia IS NULL;", dt)));
    ASSERT_EQ(
        5,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM table_array_with_nulls WHERE fa2 IS NOT NULL;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM table_array_with_nulls WHERE CARDINALITY(sia)=0;",
                  dt)));
    ASSERT_EQ(5,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM table_array_with_nulls WHERE CARDINALITY(sia)=2;",
                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM table_array_with_nulls WHERE CARDINALITY(sia) IS NULL;",
            dt)));

    // Simple lazy projection
    compare_array(
        run_simple_agg("SELECT sia FROM table_array_with_nulls WHERE i = 5;", dt),
        std::vector<int64_t>({}));

    // Simple non-lazy projection
    compare_array(
        run_simple_agg("SELECT sia FROM table_array_with_nulls WHERE sia IS NULL;", dt),
        std::vector<int64_t>({}));

    dropTable("table_array_with_nulls");
  }
}

TEST_F(Insert, IntArrayInsert) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    createTable("table_int_array", {{"bi", arrayType(kBIGINT)}});

    vector<std::string> vals = {"1", "33000", "650000", "1", "-7", "null", "5000000000"};
    string json;
    for (size_t ol = 0; ol < vals.size(); ol++) {
      string arr = "";
      for (size_t il = 0; il < vals.size(); il++) {
        size_t pos = (ol + il) % vals.size();
        arr.append(vals[pos]);
        if (il < (vals.size() - 1)) {
          arr.append(",");
        }
      }
      json += "{\"bi\": [" + arr + "]}\n";
    }
    EXPECT_NO_THROW(insertJsonValues("table_int_array", json));

    EXPECT_ANY_THROW(insertJsonValues("table_int_array", "{\"bi:\": [1,34,\"roof\"]}"));

    for (size_t ol = 0; ol < vals.size(); ol++) {
      string selString =
          "select sum(bi[" + std::to_string(ol + 1) + "]) from table_int_array;";
      ASSERT_EQ(5000682995, v<int64_t>(run_simple_agg(selString, dt)));
    }

    dropTable("table_int_array");
  }
}

TEST_F(Insert, DictBoundary) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    createTable("table_with_small_dict", {{"i", SQLTypeInfo(kINT)}, {"t", dictType(1)}});

    string csv;
    for (int cVal = 0; cVal < 280; cVal++) {
      csv += std::to_string(cVal) + ", \"" + std::to_string(cVal) + "\"\n";
    }
    EXPECT_NO_THROW(insertCsvValues("table_with_small_dict", csv));

    ASSERT_EQ(
        280,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM table_with_small_dict;", dt)));
    ASSERT_EQ(255,
              v<int64_t>(run_simple_agg(
                  "SELECT count(distinct t) FROM table_with_small_dict;", dt)));
    ASSERT_EQ(25,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM table_with_small_dict WHERE t IS NULL;", dt)));

    dropTable("table_with_small_dict");
  }
}

class KeyForString : public ExecuteTestBase, public ::testing::Test {};

TEST_F(KeyForString, KeyForString) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_NO_THROW(createTable("kfs",
                                {{"ts", dictType(1)},
                                 {"ss", dictType(2)},
                                 {"ws", dictType(4)},
                                 {"ns", dictType(4, true)},
                                 {"sa", arrayType(kTEXT)}}));
    insertJsonValues("kfs",
                     R"___({"ts": "0", "ss": "0", "ws": "0", "ns": "0", "sa": ["0", "0"]}
{"ts": "1", "ss": "1", "ws": "1", "ns": "1", "sa": ["1", "1"]}
{"ts": null, "ss": null, "ws": null, "ns": "2", "sa": ["2", "2"]})___");

    ASSERT_EQ(3, v<int64_t>(run_simple_agg("select count(*) from kfs;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "select count(*) from kfs where key_for_string(ts) is not null;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "select count(*) from kfs where key_for_string(ss) is not null;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "select count(*) from kfs where key_for_string(ws) is not null;", dt)));
    ASSERT_EQ(3,
              v<int64_t>(run_simple_agg(
                  "select count(*) from kfs where key_for_string(ns) is not null;", dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg(
            "select count(*) from kfs where key_for_string(sa[1]) is not null;", dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg(
            "select count(*) from kfs where key_for_string(ts) = key_for_string(ss);",
            dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg(
            "select count(*) from kfs where key_for_string(ss) = key_for_string(ws);",
            dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg(
            "select count(*) from kfs where key_for_string(ws) = key_for_string(ts);",
            dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg(
            "select count(*) from kfs where key_for_string(ws) = key_for_string(ns);",
            dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg(
            "select count(*) from kfs where key_for_string(ws) = key_for_string(sa[1]);",
            dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("select min(key_for_string(ts)) from kfs;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("select min(key_for_string(ss)) from kfs;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("select min(key_for_string(ws)) from kfs;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("select min(key_for_string(ns)) from kfs;", dt)));
    ASSERT_EQ(
        0, v<int64_t>(run_simple_agg("select min(key_for_string(sa[1])) from kfs;", dt)));
    ASSERT_EQ(
        0, v<int64_t>(run_simple_agg("select min(key_for_string(sa[2])) from kfs;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("select max(key_for_string(ts)) from kfs;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("select max(key_for_string(ss)) from kfs;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("select max(key_for_string(ws)) from kfs;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg("select max(key_for_string(ns)) from kfs;", dt)));
    ASSERT_EQ(
        2, v<int64_t>(run_simple_agg("select max(key_for_string(sa[1])) from kfs;", dt)));
    ASSERT_EQ(
        2, v<int64_t>(run_simple_agg("select max(key_for_string(sa[2])) from kfs;", dt)));
    ASSERT_EQ(
        2, v<int64_t>(run_simple_agg("select count(key_for_string(ts)) from kfs;", dt)));
    ASSERT_EQ(
        2, v<int64_t>(run_simple_agg("select count(key_for_string(ss)) from kfs;", dt)));
    ASSERT_EQ(
        2, v<int64_t>(run_simple_agg("select count(key_for_string(ws)) from kfs;", dt)));
    ASSERT_EQ(
        3, v<int64_t>(run_simple_agg("select count(key_for_string(ns)) from kfs;", dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg("select count(key_for_string(sa[1])) from kfs;", dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg("select count(key_for_string(sa[2])) from kfs;", dt)));

    EXPECT_NO_THROW(dropTable("kfs"));
  }
}

class Select : public ExecuteTestBase, public ::testing::Test {};

TEST_F(Select, NullWithAndOr) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    createTable("table_bool_test",
                {{"id", SQLTypeInfo(kINT)}, {"val", SQLTypeInfo(kBOOLEAN)}});
    insertCsvValues("table_bool_test", "1,true\n2,false\n3,");

    auto BOOLEAN_NULL_SENTINEL = inline_int_null_val(SQLTypeInfo(kBOOLEAN, false));

    ASSERT_EQ(
        BOOLEAN_NULL_SENTINEL,
        v<int64_t>(run_simple_agg(
            "SELECT CAST(NULL AS BOOLEAN) AND val from table_bool_test WHERE id = 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT CAST(NULL AS BOOLEAN) AND val from table_bool_test WHERE id = 2;",
            dt)));
    ASSERT_EQ(
        BOOLEAN_NULL_SENTINEL,
        v<int64_t>(run_simple_agg(
            "SELECT CAST(NULL AS BOOLEAN) AND val from table_bool_test WHERE id = 3;",
            dt)));
    ASSERT_EQ(
        BOOLEAN_NULL_SENTINEL,
        v<int64_t>(run_simple_agg(
            "SELECT val AND CAST(NULL AS BOOLEAN) from table_bool_test WHERE id = 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT val AND CAST(NULL AS BOOLEAN) from table_bool_test WHERE id = 2;",
            dt)));
    ASSERT_EQ(
        BOOLEAN_NULL_SENTINEL,
        v<int64_t>(run_simple_agg(
            "SELECT val AND CAST(NULL AS BOOLEAN) from table_bool_test WHERE id = 3;",
            dt)));

    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT CAST(NULL AS BOOLEAN) OR val from table_bool_test WHERE id = 1;",
            dt)));
    ASSERT_EQ(
        BOOLEAN_NULL_SENTINEL,
        v<int64_t>(run_simple_agg(
            "SELECT CAST(NULL AS BOOLEAN) OR val from table_bool_test WHERE id = 2;",
            dt)));
    ASSERT_EQ(
        BOOLEAN_NULL_SENTINEL,
        v<int64_t>(run_simple_agg(
            "SELECT CAST(NULL AS BOOLEAN) OR val from table_bool_test WHERE id = 3;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT val OR CAST(NULL AS BOOLEAN) from table_bool_test WHERE id = 1;",
            dt)));
    ASSERT_EQ(
        BOOLEAN_NULL_SENTINEL,
        v<int64_t>(run_simple_agg(
            "SELECT val OR CAST(NULL AS BOOLEAN) from table_bool_test WHERE id = 2;",
            dt)));
    ASSERT_EQ(
        BOOLEAN_NULL_SENTINEL,
        v<int64_t>(run_simple_agg(
            "SELECT val OR CAST(NULL AS BOOLEAN) from table_bool_test WHERE id = 3;",
            dt)));

    dropTable("table_bool_test");
  }
}

TEST_F(Select, NullGroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    createTable("table_null_group_by", {{"val", dictType()}});
    insertJsonValues("table_null_group_by", "{\"val\": null}");
    run_simple_agg("SELECT val FROM table_null_group_by GROUP BY val;", dt);
    dropTable("table_null_group_by");

    createTable("table_null_group_by", {{"val", SQLTypeInfo(kDOUBLE)}});
    insertJsonValues("table_null_group_by", "{\"val\": null}");
    run_simple_agg("SELECT val FROM table_null_group_by GROUP BY val;", dt);
    dropTable("table_null_group_by");
  }
}

TEST_F(Select, FilterAndSimpleAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test;", dt);
    c("SELECT COUNT(f) FROM test;", dt);
    c("SELECT COUNT(smallint_nulls), COUNT(*), COUNT(fn) FROM test;", dt);
    c("SELECT MIN(x) FROM test;", dt);
    c("SELECT MAX(x) FROM test;", dt);
    c("SELECT MIN(z) FROM test;", dt);
    c("SELECT MAX(z) FROM test;", dt);
    c("SELECT MIN(t) FROM test;", dt);
    c("SELECT MAX(t) FROM test;", dt);
    c("SELECT MIN(ff) FROM test;", dt);
    c("SELECT MIN(fn) FROM test;", dt);
    c("SELECT SUM(ff) FROM test;", dt);
    c("SELECT SUM(fn) FROM test;", dt);
    c("SELECT SUM(x + y) FROM test;", dt);
    c("SELECT SUM(x + y + z) FROM test;", dt);
    c("SELECT SUM(x + y + z + t) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 103);", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > "
      "1000 AND t < 1002;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 103);", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 102) OR (t > "
      "1000 AND t < 1003);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x <> 7;", dt);
    c("SELECT COUNT(*) FROM test WHERE z <> 102;", dt);
    c("SELECT COUNT(*) FROM test WHERE t <> 1002;", dt);
    c("SELECT COUNT(*) FROM test WHERE x + y = 49;", dt);
    c("SELECT COUNT(*) FROM test WHERE x + y + z = 150;", dt);
    c("SELECT COUNT(*) FROM test WHERE x + y + z + t = 1151;", dt);
    c("SELECT COUNT(*) FROM test WHERE CAST(x as TINYINT) + CAST(y as TINYINT) < CAST(z "
      "as TINYINT);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE CAST(y as TINYINT) / CAST(x as TINYINT) = 6", dt);
    c("SELECT SUM(x + y) FROM test WHERE x + y = 49;", dt);
    c("SELECT SUM(x + y + z) FROM test WHERE x + y = 49;", dt);
    c("SELECT SUM(x + y + z + t) FROM test WHERE x + y = 49;", dt);
    c("SELECT COUNT(*) FROM test WHERE x - y = -35;", dt);
    c("SELECT COUNT(*) FROM test WHERE x - y + z = 66;", dt);
    c("SELECT COUNT(*) FROM test WHERE x - y + z + t = 1067;", dt);
    c("SELECT COUNT(*) FROM test WHERE y - x = 35;", dt);
    c("SELECT 'Hello', 'World', 7 FROM test WHERE x <> 7;", dt);
    c("SELECT 'Total', COUNT(*) FROM test WHERE x <> 7;", dt);
    c("SELECT SUM(dd * x) FROM test;", dt);
    c("SELECT SUM(dd * y) FROM test;", dt);
    c("SELECT SUM(dd * w) FROM test;", dt);
    c("SELECT SUM(dd * z) FROM test;", dt);
    c("SELECT SUM(dd * t) FROM test;", dt);
    c("SELECT SUM(x * dd) FROM test;", dt);
    c("SELECT SUM(y * dd) FROM test;", dt);
    c("SELECT SUM(w * dd) FROM test;", dt);
    c("SELECT SUM(z * dd) FROM test;", dt);
    c("SELECT SUM(t * dd) FROM test;", dt);
    c("SELECT SUM(dd * ufd) FROM test;", dt);
    c("SELECT SUM(dd * d) FROM test;", dt);
    c("SELECT SUM(dd * dn) FROM test;", dt);
    c("SELECT SUM(x * dd_notnull) FROM test;", dt);
    c("SELECT SUM(2 * x) FROM test WHERE x = 7;", dt);
    c("SELECT SUM(2 * x + z) FROM test WHERE x = 7;", dt);
    c("SELECT SUM(x + y) FROM test WHERE x - y = -35;", dt);
    c("SELECT SUM(x + y) FROM test WHERE y - x = 35;", dt);
    c("SELECT SUM(x + y - z) FROM test WHERE y - x = 35;", dt);
    c("SELECT SUM(x * y + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT SUM(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", dt);
    c("SELECT SUM(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", dt);
    c("SELECT SUM(z) FROM test WHERE z IS NOT NULL;", dt);
    c("SELECT MIN(x * y + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT MIN(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", dt);
    c("SELECT MIN(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", dt);
    c("SELECT MAX(x * y + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT MAX(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", dt);
    c("SELECT MAX(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", dt);
    c("SELECT MIN(x) FROM test WHERE x = 7;", dt);
    c("SELECT MIN(z) FROM test WHERE z = 101;", dt);
    c("SELECT MIN(t) FROM test WHERE t = 1001;", dt);
    c("SELECT AVG(x + y) FROM test;", dt);
    c("SELECT AVG(x + y + z) FROM test;", dt);
    c("SELECT AVG(x + y + z + t) FROM test;", dt);
    c("SELECT AVG(y) FROM test WHERE x > 6 AND x < 8;", dt);
    c("SELECT AVG(y) FROM test WHERE z > 100 AND z < 102;", dt);
    c("SELECT AVG(y) FROM test WHERE t > 1000 AND t < 1002;", dt);
    c("SELECT MIN(dd) FROM test;", dt);
    c("SELECT MAX(dd) FROM test;", dt);
    c("SELECT SUM(dd) FROM test;", dt);
    c("SELECT AVG(dd) FROM test;", dt);
    c("SELECT AVG(dd) FROM test WHERE x > 6 AND x < 8;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 100;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 200;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 300;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 111.0;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 111.1;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 222.2;", dt);
    c("SELECT MAX(x + dd) FROM test;", dt);
    c("SELECT MAX(x + 2 * dd), MIN(x + 2 * dd) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > CAST(111.0 AS decimal(10, 2));", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > CAST(222.0 AS decimal(10, 2));", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > CAST(333.0 AS decimal(10, 2));", dt);
    c("SELECT MIN(dd * dd) FROM test;", dt);
    c("SELECT MAX(dd * dd) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE u IS NOT NULL;", dt);
    c("SELECT AVG(u * f) FROM test;", dt);
    c("SELECT AVG(u * d) FROM test;", dt);
    c("SELECT SUM(-y) FROM test;", dt);
    c("SELECT SUM(-z) FROM test;", dt);
    c("SELECT SUM(-t) FROM test;", dt);
    c("SELECT SUM(-dd) FROM test;", dt);
    c("SELECT SUM(-f) FROM test;", dt);
    c("SELECT SUM(-d) FROM test;", dt);
    c("SELECT SUM(dd * 0.99) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE 1<>2;", dt);
    c("SELECT COUNT(*) FROM test WHERE 1=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE 22 > 33;", dt);
    c("SELECT COUNT(*) FROM test WHERE ff < 23.0/4.0 AND 22 < 33;", dt);
    c("SELECT COUNT(*) FROM test WHERE x + 3*8/2 < 35 + y - 20/5;", dt);
    c("SELECT x + 2 * 10/4 + 3 AS expr FROM test WHERE x + 3*8/2 < 35 + y - 20/5 ORDER "
      "BY expr ASC;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE ff + 3.0*8 < 20.0/5;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y AND 0=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y AND 1=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y OR 1<1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y OR 1=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < 35 AND x < y AND 1=1 AND 0=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE 1>2 AND x < 35 AND x < y AND y < 10;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y GROUP BY x HAVING 0=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y GROUP BY x HAVING 1=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE ofq >= 0 OR ofq IS NULL;", dt);
    c("SELECT COUNT(*) AS val FROM test WHERE (test.dd = 0.5 OR test.dd = 3);", dt);
    c("SELECT MAX(dd_notnull * 1) FROM test;", dt);
    c("SELECT x, COUNT(*) AS n FROM test GROUP BY x, ufd ORDER BY x, n;", dt);
    c("SELECT MIN(x), MAX(x) FROM test WHERE real_str LIKE '%nope%';", dt);
    c("SELECT COUNT(*) FROM test WHERE (x > 7 AND y / (x - 7) < 44);", dt);
    c("SELECT x, AVG(ff) AS val FROM test GROUP BY x ORDER BY val;", dt);
    c("SELECT x, MAX(fn) as val FROM test WHERE fn IS NOT NULL GROUP BY x ORDER BY val;",
      dt);
    c("SELECT MAX(dn) FROM test WHERE dn IS NOT NULL;", dt);
    c("SELECT x, MAX(dn) as val FROM test WHERE dn IS NOT NULL GROUP BY x ORDER BY val;",
      dt);
    c("SELECT COUNT(*) as val FROM test GROUP BY x, y, ufd ORDER BY val;", dt);
    ASSERT_NEAR(
        static_cast<double>(-1000.3),
        v<double>(run_simple_agg(
            "SELECT AVG(fn) AS val FROM test GROUP BY rowid ORDER BY val LIMIT 1;", dt)),
        static_cast<double>(0.2));
    c("SELECT COUNT(*) FROM test WHERE d = 2.2", dt);
    c("SELECT COUNT(*) FROM test WHERE fx + 1 IS NULL;", dt);
    c("SELECT COUNT(ss) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE null IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE null_str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE null IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 > '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 <= '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 = '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 <> '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o >= CAST('1999-09-09' AS DATE);", dt);
    c("SELECT COUNT(*) FROM test WHERE o2 > '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o2 <= '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o2 = '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o2 <> '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 = o2;", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 <> o2;", dt);
    c("SELECT COUNT(*) FROM test WHERE b = 'f';", dt);
    c("SELECT COUNT(*) FROM test WHERE bn = 'f';", dt);
    c("SELECT COUNT(*) FROM test WHERE b = null;", dt);
    c("SELECT COUNT(*) FROM test WHERE bn = null;", dt);
    c("SELECT COUNT(*) FROM test WHERE bn = b;", dt);
    ASSERT_EQ(19,
              v<int64_t>(run_simple_agg("SELECT rowid FROM test WHERE rowid = 19;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg("SELECT MAX(rowid) - MIN(rowid) + 1 FROM test;", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) = 0;", dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) = 7;", dt)));
    ASSERT_EQ(5,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) <> 0;", dt)));
    ASSERT_EQ(20,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) <> 7;", dt)));
    c("SELECT MIN(x) FROM test WHERE x <> 7 AND x <> 8;", dt);
    c("SELECT MIN(x) FROM test WHERE z <> 101 AND z <> 102;", dt);
    c("SELECT MIN(x) FROM test WHERE t <> 1001 AND t <> 1002;", dt);
    ASSERT_NEAR(static_cast<double>(0.5),
                v<double>(run_simple_agg("SELECT STDDEV_POP(x) FROM test;", dt)),
                static_cast<double>(0.2));
    ASSERT_NEAR(static_cast<double>(0.5),
                v<double>(run_simple_agg("SELECT STDDEV_SAMP(x) FROM test;", dt)),
                static_cast<double>(0.2));
    ASSERT_NEAR(static_cast<double>(0.2),
                v<double>(run_simple_agg("SELECT VAR_POP(x) FROM test;", dt)),
                static_cast<double>(0.1));
    ASSERT_NEAR(static_cast<double>(0.2),
                v<double>(run_simple_agg("SELECT VAR_SAMP(x) FROM test;", dt)),
                static_cast<double>(0.1));
    ASSERT_NEAR(static_cast<double>(92.0),
                v<double>(run_simple_agg("SELECT STDDEV_POP(dd) FROM test;", dt)),
                static_cast<double>(2.0));
    ASSERT_NEAR(static_cast<double>(94.5),
                v<double>(run_simple_agg("SELECT STDDEV_SAMP(dd) FROM test;", dt)),
                static_cast<double>(1.0));
    ASSERT_NEAR(
        static_cast<double>(94.5),
        v<double>(run_simple_agg("SELECT POWER(((SUM(dd * dd) - SUM(dd) * SUM(dd) / "
                                 "COUNT(dd)) / (COUNT(dd) - 1)), 0.5) FROM test;",
                                 dt)),
        static_cast<double>(1.0));
    ASSERT_NEAR(static_cast<double>(8485.0),
                v<double>(run_simple_agg("SELECT VAR_POP(dd) FROM test;", dt)),
                static_cast<double>(10.0));
    ASSERT_NEAR(static_cast<double>(8932.0),
                v<double>(run_simple_agg("SELECT VAR_SAMP(dd) FROM test;", dt)),
                static_cast<double>(10.0));
    ASSERT_EQ(20,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test HAVING STDDEV_POP(x) < 1.0;", dt)));
    ASSERT_EQ(20,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test HAVING STDDEV_POP(x) * 5 < 3.0;", dt)));
    ASSERT_NEAR(
        static_cast<double>(0.65),
        v<double>(run_simple_agg("SELECT stddev(x) + VARIANCE(x) FROM test;", dt)),
        static_cast<double>(0.10));
    ASSERT_NEAR(static_cast<float>(0.5),
                v<float>(run_simple_agg("SELECT STDDEV_POP_FLOAT(x) FROM test;", dt)),
                static_cast<float>(0.2));
    ASSERT_NEAR(static_cast<float>(0.5),
                v<float>(run_simple_agg("SELECT STDDEV_SAMP_FLOAT(x) FROM test;", dt)),
                static_cast<float>(0.2));
    ASSERT_NEAR(static_cast<float>(0.2),
                v<float>(run_simple_agg("SELECT VAR_POP_FLOAT(x) FROM test;", dt)),
                static_cast<float>(0.1));
    ASSERT_NEAR(static_cast<float>(0.2),
                v<float>(run_simple_agg("SELECT VAR_SAMP_FLOAT(x) FROM test;", dt)),
                static_cast<float>(0.1));
    ASSERT_NEAR(static_cast<float>(92.0),
                v<float>(run_simple_agg("SELECT STDDEV_POP_FLOAT(dd) FROM test;", dt)),
                static_cast<float>(2.0));
    ASSERT_NEAR(static_cast<float>(94.5),
                v<float>(run_simple_agg("SELECT STDDEV_SAMP_FLOAT(dd) FROM test;", dt)),
                static_cast<float>(1.0));
    ASSERT_NEAR(
        static_cast<double>(94.5),
        v<double>(run_simple_agg("SELECT POWER(((SUM(dd * dd) - SUM(dd) * SUM(dd) / "
                                 "COUNT(dd)) / (COUNT(dd) - 1)), 0.5) FROM test;",
                                 dt)),
        static_cast<double>(1.0));
    ASSERT_NEAR(static_cast<float>(8485.0),
                v<float>(run_simple_agg("SELECT VAR_POP_FLOAT(dd) FROM test;", dt)),
                static_cast<float>(10.0));
    ASSERT_NEAR(static_cast<float>(8932.0),
                v<float>(run_simple_agg("SELECT VAR_SAMP_FLOAT(dd) FROM test;", dt)),
                static_cast<float>(10.0));
    ASSERT_EQ(20,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test HAVING STDDEV_POP_FLOAT(x) < 1.0;", dt)));
    ASSERT_EQ(
        20,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test HAVING STDDEV_POP_FLOAT(x) * 5 < 3.0;", dt)));
    ASSERT_NEAR(static_cast<float>(0.65),
                v<float>(run_simple_agg(
                    "SELECT stddev_FLOAT(x) + VARIANCE_float(x) FROM test;", dt)),
                static_cast<float>(0.10));
    ASSERT_NEAR(static_cast<double>(0.125),
                v<double>(run_simple_agg("SELECT COVAR_POP(x, y) FROM test;", dt)),
                static_cast<double>(0.001));
    ASSERT_NEAR(static_cast<float>(0.125),
                v<float>(run_simple_agg("SELECT COVAR_POP_FLOAT(x, y) FROM test;", dt)),
                static_cast<float>(0.001));
    ASSERT_NEAR(
        static_cast<double>(0.125),  // covar_pop expansion
        v<double>(run_simple_agg("SELECT avg(x * y) - avg(x) * avg(y) FROM test;", dt)),
        static_cast<double>(0.001));
    ASSERT_NEAR(static_cast<double>(0.131),
                v<double>(run_simple_agg("SELECT COVAR_SAMP(x, y) FROM test;", dt)),
                static_cast<double>(0.001));
    ASSERT_NEAR(static_cast<double>(0.131),
                v<double>(run_simple_agg("SELECT COVAR_SAMP_FLOAT(x, y) FROM test;", dt)),
                static_cast<double>(0.001));
    ASSERT_NEAR(
        static_cast<double>(0.131),  // covar_samp expansion
        v<double>(run_simple_agg(
            "SELECT ((sum(x * y) - sum(x) * avg(y)) / (count(x) - 1)) FROM test;", dt)),
        static_cast<double>(0.001));
    ASSERT_NEAR(static_cast<double>(0.58),
                v<double>(run_simple_agg("SELECT CORRELATION(x, y) FROM test;", dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<float>(0.58),
                v<float>(run_simple_agg("SELECT CORRELATION_FLOAT(x, y) FROM test;", dt)),
                static_cast<float>(0.01));
    ASSERT_NEAR(static_cast<double>(0.58),
                v<double>(run_simple_agg("SELECT CORR(x, y) FROM test;", dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<float>(0.58),
                v<float>(run_simple_agg("SELECT CORR_FLOAT(x, y) FROM test;", dt)),
                static_cast<float>(0.01));
    ASSERT_NEAR(static_cast<double>(0.33),
                v<double>(run_simple_agg("SELECT POWER(CORR(x, y), 2) FROM test;", dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(0.58),  // corr expansion
                v<double>(run_simple_agg("SELECT (avg(x * y) - avg(x) * avg(y)) /"
                                         "(stddev_pop(x) * stddev_pop(y)) FROM test;",
                                         dt)),
                static_cast<double>(0.01));
  }
}

TEST_F(Select, AggregateOnEmptyDecimalColumn) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (int p = 1; p <= 18; ++p) {
      for (int s = 0; s <= p - 1; ++s) {
        std::string tbl_name = "D" + std::to_string(p) + "_" + std::to_string(s);

        createTable(tbl_name, {{"val", SQLTypeInfo(kDECIMAL, p, s)}});
        std::string decimal_prec =
            "val DECIMAL(" + std::to_string(p) + "," + std::to_string(s) + ")";
        sqlite_comparator_.query("DROP TABLE IF EXISTS " + tbl_name + ";");
        sqlite_comparator_.query("CREATE TABLE " + tbl_name + "( " + decimal_prec + ");");

        std::string query =
            "SELECT MIN(val), MAX(val), SUM(val), AVG(val) FROM " + tbl_name + ";";
        c(query, dt);

        sqlite_comparator_.query("DROP TABLE IF EXISTS " + tbl_name + ";");
        dropTable(tbl_name);
      }
    }
  }
}

TEST_F(Select, AggregateConstantValueOnEmptyTable) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // tinyint: -126 / 126
    c("SELECT MIN(-126), MAX(-126), SUM(-126), AVG(-126), MIN(126), MAX(126), SUM(126), "
      "AVG(126) FROM "
      "empty_test_table;",
      dt);
    // smallint: -32766 / 32766
    c("SELECT MIN(-32766), MAX(-32766), SUM(-32766), AVG(-32766), MIN(32766), "
      "MAX(32766), SUM(32766), AVG(32766) "
      "FROM empty_test_table;",
      dt);
    // int: -2147483646 / 2147483646
    c("SELECT MIN(-2147483646), MAX(-2147483646), SUM(-2147483646), AVG(-2147483646), "
      "MIN(2147483646), "
      "MAX(2147483646), SUM(2147483646), AVG(2147483646) FROM empty_test_table;",
      dt);
    // bigint: -9223372036854775806 / 9223372036854775806
    c("SELECT MIN(-9223372036854775806), MAX(-9223372036854775806), "
      "AVG(-9223372036854775806),"
      "SUM(-9223372036854775806), MIN(9223372036854775806), MAX(9223372036854775806), "
      "SUM(9223372036854775806), AVG(9223372036854775806) FROM empty_test_table;",
      dt);
    // float: -1.5 / 1.5
    c("SELECT MIN(-1.5), MAX(-1.5), SUM(-1.5), AVG(-1.5), MIN(1.5), MAX(1.5), SUM(1.5), "
      "AVG(1.5) FROM "
      "empty_test_table;",
      dt);
    // double: -1.5055487897 / 1.5055487897
    c("SELECT MIN(-1.5055487897), MAX(-1.5055487897), SUM(-1.5055487897), "
      "AVG(-1.5055487897),"
      "MIN(1.5055487897), MAX(1.5055487897), SUM(1.5055487897), AVG(1.5055487897) FROM "
      "empty_test_table;",
      dt);
    // boolean: true / false
    c("SELECT MIN(true), MAX(true), MIN(false), MAX(false) FROM empty_test_table;", dt);
  }
}

TEST_F(Select, AggregateOnEmptyTable) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT AVG(x), AVG(y), AVG(z), AVG(t), AVG(f), AVG(d) FROM empty_test_table;", dt);
    c("SELECT MIN(x), MIN(y), MIN(z), MIN(t), MIN(f), MIN(d), MIN(b) FROM "
      "empty_test_table;",
      dt);
    c("SELECT MAX(x), MAX(y), MAX(z), MAX(t), MAX(f), MAX(d), MAX(b)  FROM "
      "empty_test_table;",
      dt);
    c("SELECT SUM(x), SUM(y), SUM(z), SUM(t), SUM(f), SUM(d) FROM empty_test_table;", dt);
    c("SELECT COUNT(x), COUNT(y), COUNT(z), COUNT(t), COUNT(f), COUNT(d), COUNT(b) FROM "
      "empty_test_table;",
      dt);
    // skipped fragment
    c("SELECT AVG(x), AVG(y), AVG(z), AVG(t), AVG(f), AVG(d) FROM empty_test_table "
      "WHERE id > 5;",
      dt);
    c("SELECT MIN(x), MIN(y), MIN(z), MIN(t), MIN(f), MIN(d), MIN(b) FROM "
      "empty_test_table WHERE "
      "id > 5;",
      dt);
    c("SELECT MAX(x), MAX(y), MAX(z), MAX(t), MAX(f), MAX(d), MAX(b) FROM "
      "empty_test_table WHERE "
      "id > 5;",
      dt);
    c("SELECT SUM(x), SUM(y), SUM(z), SUM(t), SUM(f), SUM(d) FROM empty_test_table WHERE "
      "id > 5;",
      dt);
    c("SELECT COUNT(x), COUNT(y), COUNT(z), COUNT(t), COUNT(f), COUNT(d), COUNT(b) FROM "
      "empty_test_table WHERE id > 5;",
      dt);
  }
}

TEST_F(Select, LimitAndOffset) {
  CHECK(g_num_rows >= 4);
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      const auto rows = run_multiple_agg("SELECT * FROM test LIMIT 5;", dt);
      ASSERT_EQ(size_t(5), rows->rowCount());
    }
    {
      const auto rows = run_multiple_agg("SELECT * FROM test LIMIT 5 OFFSET 3;", dt);
      ASSERT_EQ(size_t(5), rows->rowCount());
    }
    {
      const auto rows =
          run_multiple_agg("SELECT * FROM test WHERE x <> 8 LIMIT 3 OFFSET 1;", dt);
      ASSERT_EQ(size_t(3), rows->rowCount());
    }

    c("SELECT str FROM (SELECT str, SUM(y) as total_y FROM test GROUP BY str ORDER BY "
      "total_y DESC, "
      "str LIMIT 1);",
      dt);

    {
      const auto rows = run_multiple_agg("SELECT * FROM test LIMIT 0;", dt);
      ASSERT_EQ(size_t(0), rows->rowCount());
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT str FROM (SELECT str, SUM(y) as total_y FROM test GROUP BY str ORDER "
          "BY total_y DESC, str LIMIT 0);",
          dt);
      ASSERT_EQ(size_t(0), rows->rowCount());
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM ( SELECT * FROM test_inner LIMIT 3 ) t0 LIMIT 2", dt);
      ASSERT_EQ(size_t(2), rows->rowCount());
    }
  }
}

TEST_F(Select, FloatAndDoubleTests) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(f) FROM test;", dt);
    c("SELECT MAX(f) FROM test;", dt);
    c("SELECT AVG(f) FROM test;", dt);
    c("SELECT MIN(d) FROM test;", dt);
    c("SELECT MAX(d) FROM test;", dt);
    c("SELECT AVG(d) FROM test;", dt);
    c("SELECT SUM(f) FROM test;", dt);
    c("SELECT SUM(d) FROM test;", dt);
    c("SELECT SUM(f + d) FROM test;", dt);
    c("SELECT AVG(x * f) FROM test;", dt);
    c("SELECT AVG(z - 200) FROM test;", dt);
    c("SELECT SUM(CAST(x AS FLOAT)) FROM test;", dt);
    c("SELECT SUM(CAST(x AS FLOAT)) FROM test GROUP BY z;", dt);
    c("SELECT AVG(CAST(x AS FLOAT)) FROM test;", dt);
    c("SELECT AVG(CAST(x AS FLOAT)) FROM test GROUP BY y;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.0 AND f < 1.2;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.101 AND f < 1.299;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.201 AND f < 1.4;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.0 AND f < 1.2 AND d > 2.0 AND d < 2.4;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.0 AND f < 1.2 OR (d > 2.0 AND d < 3.0);",
      dt);
    c("SELECT SUM(x + y) FROM test WHERE f > 1.0 AND f < 1.2;", dt);
    c("SELECT SUM(x + y) FROM test WHERE d + f > 3.0 AND d + f < 4.0;", dt);
    c("SELECT SUM(f + d) FROM test WHERE x - y = -35;", dt);
    c("SELECT SUM(f + d) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT SUM(f * d + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT MIN(x), AVG(x * y), MAX(y + 7), AVG(x * f + 15), COUNT(*) FROM test WHERE "
      "x + y > 47 AND x + y < 51;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(y) > 42.0 "
      "ORDER BY n;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) > 1.09 "
      "ORDER BY n;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) > 1.09 "
      "AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(d) > 2.2 "
      "AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(d) > 2.2 "
      "AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f) + AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING "
      "AVG(f) + AVG(d) > 3.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f) + AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING "
      "AVG(f) + AVG(d) > 3.5 ORDER BY n;",
      dt);
    c("SELECT f + d AS s, x * y FROM test ORDER by s DESC;", dt);
    c("SELECT COUNT(*) AS n FROM test GROUP BY f ORDER BY n;", dt);
    c("SELECT f, COUNT(*) FROM test GROUP BY f HAVING f > 1.25;", dt);
    c("SELECT COUNT(*) AS n FROM test GROUP BY d ORDER BY n;", dt);
    c("SELECT MIN(x + y) AS n FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY f + 1, "
      "f + d ORDER BY n;",
      dt);
    c("SELECT f + d AS s FROM test GROUP BY s ORDER BY s DESC;", dt);
    c("SELECT f + 1 AS s, AVG(u * f) FROM test GROUP BY s ORDER BY s DESC;", dt);
    c("SELECT (CAST(dd AS float) * 0.5) AS key FROM test GROUP BY key ORDER BY key DESC;",
      dt);
    c("SELECT (CAST(dd AS double) * 0.5) AS key FROM test GROUP BY key ORDER BY key "
      "DESC;",
      dt);

    c("SELECT fn FROM test ORDER BY fn ASC NULLS FIRST;",
      "SELECT fn FROM test ORDER BY fn ASC;",
      dt);
    c("SELECT fn FROM test WHERE fn < 0 OR fn IS NULL ORDER BY fn ASC NULLS FIRST;",
      "SELECT fn FROM test WHERE fn < 0 OR fn IS NULL ORDER BY fn ASC;",
      dt);
    ASSERT_NEAR(static_cast<double>(1.3),
                v<double>(run_simple_agg("SELECT AVG(f) AS n FROM test WHERE x = 7 GROUP "
                                         "BY z HAVING AVG(y) + STDDEV(y) "
                                         "> 42.0 ORDER BY n + VARIANCE(y);",
                                         dt)),
                static_cast<double>(0.1));
    ASSERT_NEAR(
        static_cast<double>(92.0),
        v<double>(run_simple_agg("SELECT STDDEV_POP(dd) AS n FROM test ORDER BY n;", dt)),
        static_cast<double>(1.0));
  }
}

TEST_F(Select, FilterShortCircuit) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > "
      "1000 AND UNLIKELY(t < 1002);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > "
      "1000 AND t > 1000 AND t > 1001 "
      "AND t > 1002 AND t > 1003 AND t > 1004 AND UNLIKELY(t < 1002);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > "
      "1000 AND t > 1000 AND t > 1001 "
      "AND t > 1002 AND t > 1003 AND t > 1004 AND t > 1005 AND UNLIKELY(t < 1002);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > "
      "1000 AND t > 1000 AND t > 1001 "
      "AND t > 1002 AND t > 1003 AND UNLIKELY(t < 111) AND (str LIKE 'f__%%');",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND UNLIKELY(z < 200) AND z > 100 "
      "AND z < 102 AND t > 1000 AND "
      "t > 1000 AND t > 1001  AND UNLIKELY(t < 1111 AND t > 1100) AND (str LIKE 'f__%%') "
      "AND t > 1002 AND t > 1003;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE UNLIKELY(x IN (7, 8, 9, 10)) AND y > 42;", dt);
    c("SELECT COUNT(*) FROM test WHERE (x / 2.0 > 3.500) AND (str LIKE 's__');", dt);
    {
      std::string query(
          "SELECT COUNT(*) FROM test WHERE (MOD(x, 2) = 0) AND (str LIKE 's__') AND (x "
          "in (7));");
      const auto result = run_multiple_agg(query, dt);
      const auto row = result->getNextRow(true, true);
      ASSERT_EQ(size_t(1), row.size());
      ASSERT_EQ(int64_t(0), v<int64_t>(row[0]));
    }
  }
}

TEST_F(Select, InValues) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c(R"(SELECT x FROM test WHERE x IN (8, 9, 10, 11, 12, 13, 14) GROUP BY x ORDER BY x;)",
      dt);
    c(R"(SELECT y FROM test WHERE y IN (43, 44, 45, 46, 47, 48, 49) GROUP BY y ORDER BY y;)",
      dt);
    c(R"(SELECT t FROM test WHERE t NOT IN (NULL) GROUP BY t ORDER BY t;)", dt);
    c(R"(SELECT t FROM test WHERE t NOT IN (1001, 1003, 1005, 1007, 1009, -10) GROUP BY t ORDER BY t;)",
      dt);
    c(R"(WITH dimensionValues AS (SELECT b FROM test GROUP BY b ORDER BY b) SELECT x FROM test WHERE b in (SELECT b FROM dimensionValues) GROUP BY x ORDER BY x;)",
      dt);
  }
}

TEST_F(Select, FilterAndMultipleAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT AVG(x), AVG(y) FROM test;", dt);
    c("SELECT MIN(x), AVG(x * y), MAX(y + 7), COUNT(*) FROM test WHERE x + y > 47 AND x "
      "+ y < 51;",
      dt);
    c("SELECT str, AVG(x), COUNT(*) as xx, COUNT(*) as countval FROM test GROUP BY str "
      "ORDER BY str;",
      dt);
  }
}

TEST_F(Select, GroupBy) {
  {  // generate dataset to test count distinct rewrite
    createTable("count_distinct_rewrite", {{"v1", SQLTypeInfo(kINT)}});
    std::string csv_data;
    for (int i = 0; i < 1000000; i++) {
      csv_data += std::to_string(i) + "\n";
    }
    insertCsvValues("count_distinct_rewrite", csv_data);
  }

  std::vector<std::string> runnable_column_names = {
      "x", "w", "y", "z", "fx", "f", "d", "dd", "dd_notnull", "u", "smallint_nulls"};
  std::set<std::string> str_col_names = {
      "str", "fixed_str", "shared_dict", "null_str", "fixed_null_str"};
  // some column type can execute a subset of agg ops without an exception
  using TypeAndAvaliableAggOps = std::pair<std::string, std::vector<SQLAgg>>;
  std::vector<TypeAndAvaliableAggOps> column_names_with_available_agg_ops = {
      {"str", std::vector<SQLAgg>{SQLAgg::kSAMPLE}},
      {"fixed_str", std::vector<SQLAgg>{SQLAgg::kSAMPLE}},
      {"shared_dict", std::vector<SQLAgg>{SQLAgg::kSAMPLE}},
      {"null_str", std::vector<SQLAgg>{SQLAgg::kSAMPLE}},
      {"fixed_null_str", std::vector<SQLAgg>{SQLAgg::kSAMPLE}},
      {"b", std::vector<SQLAgg>{SQLAgg::kMIN, SQLAgg::kMAX, SQLAgg::kSAMPLE}},
      {"m", std::vector<SQLAgg>{SQLAgg::kMIN, SQLAgg::kMAX, SQLAgg::kSAMPLE}},
      {"m_3", std::vector<SQLAgg>{SQLAgg::kMIN, SQLAgg::kMAX, SQLAgg::kSAMPLE}},
      {"m_6", std::vector<SQLAgg>{SQLAgg::kMIN, SQLAgg::kMAX, SQLAgg::kSAMPLE}},
      {"m_9", std::vector<SQLAgg>{SQLAgg::kMIN, SQLAgg::kMAX, SQLAgg::kSAMPLE}},
      {"n", std::vector<SQLAgg>{SQLAgg::kMIN, SQLAgg::kMAX, SQLAgg::kSAMPLE}},
      {"o", std::vector<SQLAgg>{SQLAgg::kMIN, SQLAgg::kMAX, SQLAgg::kSAMPLE}},
      {"o1", std::vector<SQLAgg>{SQLAgg::kMIN, SQLAgg::kMAX, SQLAgg::kSAMPLE}}};
  auto get_query_str = [](SQLAgg agg_op, const std::string& col_name) {
    std::ostringstream oss;
    oss << "SELECT " << col_name << " v1, ";
    switch (agg_op) {
      case SQLAgg::kMIN:
        oss << "MIN(" << col_name << ")";
        break;
      case SQLAgg::kMAX:
        oss << "MAX(" << col_name << ")";
        break;
      case SQLAgg::kAVG:
        oss << "AVG(" << col_name << ")";
        break;
      case SQLAgg::kSAMPLE:
        oss << "SAMPLE(" << col_name << ")";
        break;
      case SQLAgg::kAPPROX_QUANTILE:
        oss << "APPROX_PERCENTILE(" << col_name << ", 0.5) ";
        break;
      default:
        CHECK(false);
        break;
    }
    oss << " v2 FROM test GROUP BY " << col_name;
    return oss.str();
  };
  auto perform_test = [&get_query_str, &str_col_names](SQLAgg agg_op,
                                                       const std::string& col_name,
                                                       ExecutorDeviceType dt) {
    std::string omnisci_cnt_query, sqlite_cnt_query, omnisci_min_query, sqlite_min_query;
    if (agg_op == SQLAgg::kAPPROX_QUANTILE || agg_op == SQLAgg::kSAMPLE) {
      omnisci_cnt_query =
          "SELECT COUNT(*) FROM (" + get_query_str(agg_op, col_name) + ")";
      omnisci_min_query = "SELECT MIN(v2) FROM (" + get_query_str(agg_op, col_name) + ")";
      // since sqlite does not support sample and approx_quantile
      // we instead use max agg op; min and avg are also possible
      sqlite_cnt_query =
          "SELECT COUNT(*) FROM (" + get_query_str(SQLAgg::kMAX, col_name) + ")";
      sqlite_min_query =
          "SELECT MIN(v2) FROM (" + get_query_str(SQLAgg::kMAX, col_name) + ")";
      if (col_name.compare("d") != 0 && col_name.compare("f") != 0 &&
          col_name.compare("fx") != 0) {
        omnisci_cnt_query += " WHERE v1 = v2";
        sqlite_cnt_query += " WHERE v1 = v2";
      }
    } else {
      omnisci_cnt_query =
          "SELECT COUNT(*) FROM (" + get_query_str(agg_op, col_name) + ")";
      omnisci_min_query = "SELECT MIN(v2) FROM (" + get_query_str(agg_op, col_name) + ")";
      if (col_name.compare("d") != 0 && col_name.compare("f") != 0 &&
          col_name.compare("fx") != 0) {
        omnisci_cnt_query += " WHERE v1 = v2";
      }
      sqlite_cnt_query = omnisci_cnt_query;
      sqlite_min_query = omnisci_min_query;
    }
    c(omnisci_cnt_query, sqlite_cnt_query, dt);
    if (!str_col_names.count(col_name)) {
      c(omnisci_min_query, sqlite_min_query, dt);
    } else {
      LOG(WARNING) << "Skipping aggregation query on string column: "
                   << omnisci_min_query;
    }
  };

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c("SELECT COUNT(*) FROM test_ranges GROUP BY i, b;", dt);
    c("SELECT i, b FROM test_ranges GROUP BY i, b;", dt);

    {
      const auto big_group_threshold = g_big_group_threshold;
      ScopeGuard reset_big_group_threshold = [&big_group_threshold] {
        g_big_group_threshold = big_group_threshold;
      };
      g_big_group_threshold = 1;
      c("SELECT d, COUNT(*) FROM test GROUP BY d ORDER BY d DESC LIMIT 10;", dt);
    }

    if (g_enable_columnar_output) {
      // TODO: Fixup the tests below when running with columnar output enabled
      continue;
    }

    c("SELECT x, y, COUNT(*) FROM test GROUP BY x, y;", dt);
    c("SELECT x, y, APPROX_COUNT_DISTINCT(str) FROM test GROUP BY x, y;",
      "SELECT x, y, COUNT(distinct str) FROM test GROUP BY x, y;",
      dt);
    c("SELECT f, ff, APPROX_COUNT_DISTINCT(str) from test group by f, ff ORDER BY f, ff;",
      "SELECT f, ff, COUNT(distinct str) FROM test GROUP BY f, ff ORDER BY f, ff;",
      dt);

    // check rewriting agg on gby col to its equivalent case-when
    // 1. check count-distinct op runs successfully
    ASSERT_NO_THROW(run_multiple_agg(
        "SELECT v1, COUNT(DISTINCT v1) FROM count_distinct_rewrite GROUP BY v1 limit "
        "1;",
        dt));
    ASSERT_NO_THROW(run_multiple_agg(
        "SELECT v1, COUNT(DISTINCT v1), CASE WHEN v1 IS NOT NULL THEN 1 ELSE 0 END "
        "FROM count_distinct_rewrite GROUP BY v1 limit 1;",
        dt));
    ASSERT_NO_THROW(
        run_multiple_agg("SELECT v1, COUNT(DISTINCT v1), APPROX_COUNT_DISTINCT(DISTINCT "
                         "v1) FROM count_distinct_rewrite GROUP BY v1 limit "
                         "1;",
                         dt));
    ASSERT_NO_THROW(run_multiple_agg(
        "SELECT v1, APPROX_COUNT_DISTINCT(v1) FROM count_distinct_rewrite GROUP BY v1 "
        "limit 1;",
        dt));
    ASSERT_NO_THROW(run_multiple_agg(
        "SELECT v1, APPROX_COUNT_DISTINCT(v1), CASE WHEN v1 IS NOT NULL THEN 1 ELSE 0 "
        "END, COUNT(DISTINCT v1) FROM count_distinct_rewrite GROUP BY v1 limit 1;",
        dt));

    // 2. remaining agg ops: avg / min / max / sample / approx_quantile
    // there are two exceptions when perform gby-agg: 1) gby fails and 2) agg fails
    // otherwise this rewriting should return the same result as the original query
    std::vector<SQLAgg> test_agg_ops = {SQLAgg::kMIN,
                                        SQLAgg::kMAX,
                                        SQLAgg::kSAMPLE,
                                        SQLAgg::kAVG,
                                        SQLAgg::kAPPROX_QUANTILE};
    for (auto& col_name : runnable_column_names) {
      for (auto& agg_op : test_agg_ops) {
        perform_test(agg_op, col_name, dt);
      }
    }
    for (TypeAndAvaliableAggOps& info : column_names_with_available_agg_ops) {
      const auto& col_name = info.first;
      const auto& agg_ops = info.second;
      for (auto& agg_op : agg_ops) {
        perform_test(agg_op, col_name, dt);
      }
    }

    // check whether we only apply case-when optimization towards count* distinct agg
    c("SELECT x, COUNT(x) FROM test GROUP BY x;", dt);
    c("SELECT x, COUNT(DISTINCT x) FROM test GROUP BY x;", dt);
    c("SELECT x, y, COUNT(x) FROM test GROUP BY x,y;", dt);
    c("SELECT x, y, COUNT(DISTINCT x) FROM test GROUP BY x,y;", dt);
  }
  dropTable("count_distinct_rewrite");
}

TEST_F(Select, ExecutePlanWithoutGroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // SQLite doesn't support NOW(), and timestamps may not be exactly equal,
    // so just test for no throw.
    EXPECT_NO_THROW(
        run_multiple_agg("SELECT COUNT(*), NOW(), CURRENT_TIME, CURRENT_DATE, "
                         "CURRENT_TIMESTAMP FROM test;",
                         dt));
    EXPECT_NO_THROW(
        run_multiple_agg("SELECT x, COUNT(*), NOW(), CURRENT_TIME, CURRENT_DATE, "
                         "CURRENT_TIMESTAMP FROM test GROUP BY x;",
                         dt));
  }
}

TEST_F(Select, FilterAndGroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x, y;", dt);
    c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x + 1, x + "
      "y;",
      dt);
    c("SELECT x, y, COUNT(*) FROM test GROUP BY x, y;", dt);
    c("SELECT x, dd, COUNT(*) FROM test GROUP BY x, dd ORDER BY x, dd;", dt);
    c("SELECT dd AS key1, COUNT(*) AS value1 FROM test GROUP BY key1 HAVING key1 IS NOT "
      "NULL ORDER BY key1, value1 "
      "DESC "
      "LIMIT 12;",
      dt);
    c("SELECT 'literal_string' AS key0 FROM test GROUP BY key0;", dt);
    c("SELECT str, MIN(y) FROM test WHERE y IS NOT NULL GROUP BY str ORDER BY str DESC;",
      dt);
    c("SELECT x, MAX(z) FROM test WHERE z IS NOT NULL GROUP BY x HAVING x > 7;", dt);
    c("SELECT CAST((dd - 0.5) * 2.0 AS int) AS key0, COUNT(*) AS val FROM test WHERE (dd "
      ">= 100.0 AND dd < 400.0) "
      "GROUP "
      "BY key0 HAVING key0 >= 0 AND key0 < 400 ORDER BY val DESC LIMIT 50 OFFSET 0;",
      dt);
    c("SELECT y, AVG(CASE WHEN x BETWEEN 6 AND 7 THEN x END) FROM test GROUP BY y ORDER "
      "BY y;",
      dt);
    c("SELECT x, AVG(u), COUNT(*) AS n FROM test GROUP BY x ORDER BY n DESC;", dt);
    c("SELECT f, ss FROM test GROUP BY f, ss ORDER BY f DESC;", dt);
    c("SELECT fx, COUNT(*) FROM test GROUP BY fx HAVING COUNT(*) > 5;", dt);
    c("SELECT fx, COUNT(*) n FROM test GROUP BY fx ORDER BY n DESC, fx IS NULL DESC;",
      dt);
    c("SELECT CASE WHEN x > 8 THEN 100000000 ELSE 42 END AS c, COUNT(*) FROM test GROUP "
      "BY c;",
      dt);
    // SQLite floors instead of rounds when casting float to int.
    c("SELECT COUNT(*) FROM test WHERE CAST((CAST(x AS FLOAT) - 1) * 0.2 AS INT) = 1;",
      dt);
    c("SELECT CAST(CAST(d/2 AS FLOAT) AS INTEGER) AS key, COUNT(*) FROM test GROUP BY "
      "key;",
      dt);
    c("SELECT x * 2 AS x2, COUNT(DISTINCT y) AS n FROM test GROUP BY x2 ORDER BY n DESC;",
      dt);
    c("SELECT x, COUNT(real_str) FROM test GROUP BY x ORDER BY x DESC;", dt);
    c("SELECT str, SUM(y - y) FROM test GROUP BY str ORDER BY str ASC;", dt);
    c("SELECT str, SUM(y - y) FROM test WHERE y - y IS NOT NULL GROUP BY str ORDER BY "
      "str ASC;",
      dt);
    c("select shared_dict,m from test where (m >= CAST('2014-12-13 22:23:15' AS "
      "TIMESTAMP(0)) and m <= "
      "CAST('2014-12-14 22:23:15' AS TIMESTAMP(0)))  and CAST(m AS TIMESTAMP(0)) BETWEEN "
      "'2014-12-14 22:23:15' AND "
      "'2014-12-13 22:23:15' group by shared_dict,m;",
      dt);
    c("SELECT x, SUM(z) FROM test WHERE z IS NOT NULL GROUP BY x ORDER BY x;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT MIN(str) FROM test GROUP BY x;", dt),
                 std::runtime_error);
  }
}

TEST_F(Select, GroupByBoundariesAndNull) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      std::string query(
          "SELECT CAST(CASE WHEN x = 7 THEN 2147483647 ELSE null END AS INTEGER) AS "
          "col0, COUNT(*) FROM test GROUP BY col0 ORDER BY col0 ASC");
      c(query + " NULLS FIRST;", query + ";", dt);
    }
    {
      std::string query(
          "SELECT smallint_nulls, COUNT(*) FROM test GROUP BY smallint_nulls ORDER BY "
          "smallint_nulls ASC");
      c(query + " NULLS FIRST;", query + ";", dt);
    }
    {
      std::string query(
          "SELECT CAST(CASE WHEN x = 7 THEN 127 ELSE null END AS TINYINT) AS col0, "
          "COUNT(*) FROM test GROUP BY col0 ORDER BY col0 ASC");
      c(query + " NULLS FIRST;", query + ";", dt);
    }
  }
}

TEST_F(Select, NestedGroupByWithFloat) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    char const* query =
        "SELECT c, x, f FROM ("
        "   SELECT x, COUNT(*) AS c, f"
        "   FROM test"
        "   GROUP BY x, f"
        " )"
        " GROUP BY c, x, f"
        " ORDER BY c, x, f;";
    c(query, dt);
  }
}

TEST_F(Select, Arrays) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // Simple lazy projection
    compare_array(run_simple_agg("SELECT arr_i16 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({2, 3, 4}));
    compare_array(run_simple_agg("SELECT arr_i32 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({20, 30, 40}));
    compare_array(run_simple_agg("SELECT arr_i64 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({200, 300, 400}));
    compare_array(run_simple_agg("SELECT arr_str FROM array_test WHERE x = 8;", dt),
                  std::vector<std::string>({"bb", "cc", "dd"}));
    compare_array(run_simple_agg("SELECT arr_float FROM array_test WHERE x = 8;", dt),
                  std::vector<float>({2.2, 3.3, 4.4}));
    compare_array(run_simple_agg("SELECT arr_double FROM array_test WHERE x = 8;", dt),
                  std::vector<double>({22.2, 33.3, 44.4}));
    compare_array(run_simple_agg("SELECT arr_bool FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({1, 0, 1, 0, 1, 0}));

    compare_array(run_simple_agg("SELECT arr3_i8 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({2, 3, 4}));
    compare_array(run_simple_agg("SELECT arr3_i16 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({2, 3, 4}));
    compare_array(run_simple_agg("SELECT arr3_i32 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({20, 30, 40}));
    compare_array(run_simple_agg("SELECT arr3_i64 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({200, 300, 400}));
    compare_array(run_simple_agg("SELECT arr3_float FROM array_test WHERE x = 8;", dt),
                  std::vector<float>({2.2, 3.3, 4.4}));
    compare_array(run_simple_agg("SELECT arr3_double FROM array_test WHERE x = 8;", dt),
                  std::vector<double>({22.2, 33.3, 44.4}));
    compare_array(run_simple_agg("SELECT arr6_bool FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({1, 0, 1, 0, 1, 0}));

    // requires punt to CPU
    compare_array(
        run_simple_agg("SELECT ARRAY[1,2,3,5] from array_test WHERE x = 8 limit 8675309;",
                       dt),
        std::vector<int64_t>({1, 2, 3, 5}));
    compare_array(
        run_simple_agg("SELECT ARRAY[2*arr3_i32[1],2*arr3_i32[2],2*arr3_i32[3]] FROM "
                       "array_test a WHERE x = 8 limit 31337;",
                       dt),
        std::vector<int64_t>({40, 60, 80}));

    // Simple non-lazy projection
    compare_array(
        run_simple_agg("SELECT arr_i16 FROM array_test WHERE arr_i16[1] = 2;", dt),
        std::vector<int64_t>({2, 3, 4}));
    compare_array(
        run_simple_agg("SELECT arr_i32 FROM array_test WHERE arr_i32[1] = 20;", dt),
        std::vector<int64_t>({20, 30, 40}));
    compare_array(
        run_simple_agg("SELECT arr_i64 FROM array_test WHERE arr_i64[1] = 200;", dt),
        std::vector<int64_t>({200, 300, 400}));
    compare_array(
        run_simple_agg("SELECT arr_str FROM array_test WHERE arr_str[1] = 'bb';", dt),
        std::vector<std::string>({"bb", "cc", "dd"}));
    // TODO(adb): Calcite is casting the column value to DOUBLE to do the comparison,
    // which results in the comparison failing. Is this desired behavior or a bug? Adding
    // the CAST below for now to test projection.
    compare_array(
        run_simple_agg(
            "SELECT arr_float FROM array_test WHERE arr_float[1] = CAST(2.2 as FLOAT);",
            dt),
        std::vector<float>({2.2, 3.3, 4.4}));
    compare_array(
        run_simple_agg("SELECT arr_double FROM array_test WHERE arr_double[1] = 22.2;",
                       dt),
        std::vector<double>({22.2, 33.3, 44.4}));
    compare_array(run_simple_agg(
                      "SELECT arr_bool FROM array_test WHERE x < 9 AND arr_bool[1];", dt),
                  std::vector<int64_t>({1, 0, 1, 0, 1, 0}));

    compare_array(
        run_simple_agg("SELECT arr3_i8 FROM array_test WHERE arr3_i8[1] = 2;", dt),
        std::vector<int64_t>({2, 3, 4}));
    compare_array(
        run_simple_agg("SELECT arr3_i16 FROM array_test WHERE arr3_i16[1] = 2;", dt),
        std::vector<int64_t>({2, 3, 4}));
    compare_array(
        run_simple_agg("SELECT arr3_i32 FROM array_test WHERE arr3_i32[1] = 20;", dt),
        std::vector<int64_t>({20, 30, 40}));
    compare_array(
        run_simple_agg("SELECT arr3_i64 FROM array_test WHERE arr3_i64[1] = 200;", dt),
        std::vector<int64_t>({200, 300, 400}));
    compare_array(
        run_simple_agg(
            "SELECT arr3_float FROM array_test WHERE arr3_float[1] = CAST(2.2 AS FLOAT);",
            dt),
        std::vector<float>({2.2, 3.3, 4.4}));
    compare_array(
        run_simple_agg("SELECT arr3_double FROM array_test WHERE arr3_double[1] = 22.2;",
                       dt),
        std::vector<double>({22.2, 33.3, 44.4}));
    compare_array(
        run_simple_agg("SELECT arr6_bool FROM array_test WHERE x < 9 AND arr6_bool[1];",
                       dt),
        std::vector<int64_t>({1, 0, 1, 0, 1, 0}));

    // throw exception when comparing full array joins
    EXPECT_THROW(run_simple_agg("SELECT COUNT(1) FROM array_test t1, array_test t2 WHERE "
                                "t1.arr_i32 < t2.arr_i32;",
                                dt),
                 std::runtime_error);

    EXPECT_THROW(run_simple_agg("SELECT COUNT(1) FROM array_test t1, array_test t2 WHERE "
                                "t1.arr_i32 <= t2.arr_i32;",
                                dt),
                 std::runtime_error);

    EXPECT_THROW(run_simple_agg("SELECT COUNT(1) FROM array_test t1, array_test t2 WHERE "
                                "t1.arr_i32 > t2.arr_i32;",
                                dt),
                 std::runtime_error);

    EXPECT_THROW(run_simple_agg("SELECT COUNT(1) FROM array_test t1, array_test t2 WHERE "
                                "t1.arr_i32 >= t2.arr_i32;",
                                dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT COUNT(1) FROM array_test t1, array_test t2 WHERE "
                                "t1.arr_i32 <> t2.arr_i32;",
                                dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT COUNT(1) FROM array_test t1, array_test t2 WHERE "
                                "t1.arr_str[1] > t2.arr_str[1];",
                                dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT COUNT(1) FROM array_test t1, array_test t2 WHERE "
                                "t1.arr_str[1] >= t2.arr_str[1];",
                                dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT COUNT(1) FROM array_test t1, array_test t2 WHERE "
                                "t1.arr_str[1] < t2.arr_str[1];",
                                dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT COUNT(1) FROM array_test t1, array_test t2 WHERE "
                                "t1.arr_str[1] <= t2.arr_str[1];",
                                dt),
                 std::runtime_error);
    EXPECT_NO_THROW(
        run_simple_agg("SELECT COUNT(1) FROM array_test t1, array_test t2 WHERE "
                       "t1.arr_str[1] <> t2.arr_str[1];",
                       dt));
  }
}

TEST_F(Select, FilterCastToDecimal) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    ASSERT_EQ(static_cast<int64_t>(5),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 7.1;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(10),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE y > 42.5;", dt)));
    ASSERT_EQ(static_cast<int64_t>(10),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE ufd > -2147483648.0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(15),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE ofd > -2147483648;", dt)));
  }
}

TEST_F(Select, FilterAndGroupByMultipleAgg) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(x + y), COUNT(*), AVG(x + 1) FROM test WHERE x + y > 47 AND x + y < 53 "
      "GROUP BY x, y;",
      dt);
    c("SELECT MIN(x + y), COUNT(*), AVG(x + 1) FROM test WHERE x + y > 47 AND x + y < 53 "
      "GROUP BY x + 1, x + y;",
      dt);
  }
}

TEST_F(Select, GroupByKeylessAndNotKeyless) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT fixed_str FROM test WHERE fixed_str = 'fish' GROUP BY fixed_str;", dt);
    c("SELECT AVG(x), fixed_str FROM test WHERE fixed_str = 'fish' GROUP BY fixed_str;",
      dt);
    c("SELECT AVG(smallint_nulls), fixed_str FROM test WHERE fixed_str = 'foo' GROUP BY "
      "fixed_str;",
      dt);
    c("SELECT null_str, AVG(smallint_nulls) FROM test GROUP BY null_str;", dt);
  }
}

TEST_F(Select, Having) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING MAX(x) > 5 ORDER BY n;",
      dt);
    c("SELECT MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING MAX(x) > 5 ORDER BY n "
      "LIMIT 1;",
      dt);
    c("SELECT MAX(y) AS n FROM test WHERE x > 7 GROUP BY z HAVING MAX(x) < 100 ORDER BY "
      "n;",
      dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 ORDER "
      "BY n;",
      dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 AND "
      "COUNT(*) > 5 ORDER BY n;",
      dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 AND "
      "COUNT(*) > 9 ORDER BY n;",
      dt);
    c("SELECT str, COUNT(*) AS n FROM test GROUP BY str HAVING str IN ('bar', 'baz') "
      "ORDER BY str;",
      dt);
    c("SELECT str, COUNT(*) AS n FROM test GROUP BY str HAVING str LIKE 'ba_' ORDER BY "
      "str;",
      dt);
    c("SELECT ss, COUNT(*) AS n FROM test GROUP BY ss HAVING ss LIKE 'bo_' ORDER BY ss;",
      dt);
    c("SELECT x, COUNT(*) FROM test WHERE x > 9 GROUP BY x HAVING x > 15;", dt);
    c("SELECT x, AVG(y), AVG(y) FROM test GROUP BY x HAVING x >= 0 ORDER BY x;", dt);
    c("SELECT AVG(y), x, AVG(y) FROM test GROUP BY x HAVING x >= 0 ORDER BY x;", dt);
    c("SELECT x, y, COUNT(*) FROM test WHERE real_str LIKE 'nope%' GROUP BY x, y HAVING "
      "x >= 0 AND x < 12 AND y >= 0 "
      "AND y < 12 ORDER BY x, y;",
      dt);
  }
}

TEST_F(Select, ConstantWidthBucketExpr) {
  createTable("wb_test",
              {{"i1", SQLTypeInfo(kTINYINT)},
               {"i2", SQLTypeInfo(kSMALLINT)},
               {"i4", SQLTypeInfo(kINT)},
               {"i8", SQLTypeInfo(kBIGINT)},
               {"f", SQLTypeInfo(kFLOAT)},
               {"d", SQLTypeInfo(kDOUBLE)},
               {"dc", SQLTypeInfo(kDECIMAL, 15, 8, false)},
               {"n", SQLTypeInfo(kDECIMAL, 15, 8, false)}});
  insertCsvValues("wb_test", ",,,,,,,");
  auto drop = "DROP TABLE IF EXISTS wb_test;";
  auto create =
      "CREATE TABLE wb_test (i1 tinyint, i2 smallint, i4 int, i8 bigint, f float, d "
      "double, dc decimal(15,8), n numeric(15,8));";
  auto insert =
      "INSERT INTO wb_test VALUES (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);";
  sqlite_comparator_.query(drop);
  sqlite_comparator_.query(create);
  sqlite_comparator_.query(insert);
  auto test_queries = [](const std::string col_name) {
    std::string omnisci_query =
        "SELECT WIDTH_BUCKET(" + col_name + ", 1, 2, 3) FROM wb_test;";
    std::string sqlite_query = "SELECT " + col_name + " FROM wb_test;";
    return std::make_pair(omnisci_query, sqlite_query);
  };
  std::vector<std::string> col_names{"i1", "i2", "i4", "i8", "d", "f", "dc", "n"};
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // in single-mode, we can see std::runtime_error exception for the below test queries
    // having unsupported or invalid arguments of the function
    // but in dist-mode, we detect Calcite SQL error instead of std::runtime_error
    // so we try to detect 'any' exception instead of the specific exception type
    EXPECT_ANY_THROW(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, 0);", dt));
    EXPECT_ANY_THROW(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, -1);", dt));
    EXPECT_ANY_THROW(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, NULL);", dt));
    EXPECT_ANY_THROW(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, 2147483649);", dt));
    EXPECT_ANY_THROW(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, -2147483649);", dt));
    EXPECT_ANY_THROW(
        run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, 9223372036854775800);", dt));
    EXPECT_ANY_THROW(
        run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, -9223372036854775800);", dt));
    EXPECT_ANY_THROW(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, 1.11112);", dt));
    EXPECT_ANY_THROW(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, 1.111121112);", dt));
    EXPECT_ANY_THROW(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, -1.11112);", dt));
    EXPECT_ANY_THROW(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, -1.111121112);", dt));
    EXPECT_ANY_THROW(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 2, 3);", dt));
    EXPECT_ANY_THROW(
        run_simple_agg("SELECT WIDTH_BUCKET(1, 2147483647, 2147483647, 3);", dt));
    EXPECT_ANY_THROW(
        run_simple_agg("SELECT WIDTH_BUCKET(1, 2147483649, 2147483649, 3);", dt));
    EXPECT_ANY_THROW(
        run_simple_agg("SELECT WIDTH_BUCKET(1, -2147483647, -2147483647, 3);", dt));
    EXPECT_ANY_THROW(
        run_simple_agg("SELECT WIDTH_BUCKET(1, -2147483649, -2147483649, 3);", dt));
    EXPECT_ANY_THROW(run_simple_agg(
        "SELECT WIDTH_BUCKET(1, 9223372036854775808, 9223372036854775808, 3);", dt));
    EXPECT_ANY_THROW(run_simple_agg(
        "SELECT WIDTH_BUCKET(1, -9223372036854775808, -9223372036854775808, 3);", dt));
    EXPECT_NO_THROW(run_simple_agg("SELECT WIDTH_BUCKET(NULL, 2, 3, 100);", dt));

    // check the correctness of the function based on postgres 12.7's result
    EXPECT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, 3, 100);", dt)));
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(2, 2, 3, 100);", dt)));
    EXPECT_EQ(int64_t(101),
              v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(3, 2, 3, 100);", dt)));
    EXPECT_EQ(int64_t(11),
              v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(2.1, 2, 3, 100);", dt)));
    EXPECT_EQ(
        int64_t(11),
        v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(2.11, 2.1, 2.2, 100);", dt)));
    EXPECT_EQ(int64_t(91),
              v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(2.1, 3, 2, 100);", dt)));
    EXPECT_EQ(
        int64_t(95),
        v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(2.156789, 3, 2.11, 100);", dt)));
    EXPECT_EQ(int64_t(26),
              v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(1, 2, -2, 100);", dt)));
    EXPECT_EQ(int64_t(48),
              v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(0.1, 2, -2, 100);", dt)));
    EXPECT_EQ(int64_t(48),
              v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(-0.1, -2, 2, 100);", dt)));
    EXPECT_EQ(int64_t(53),
              v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(-0.1, 2, -2, 100);", dt)));

    for (auto& col : col_names) {
      auto queries = test_queries(col);
      c(queries.first, queries.second, dt);
    }
  }
  dropTable("wb_test");
  sqlite_comparator_.query(drop);
}

TEST_F(Select, WidthBucketExpr) {
  createTable("wb_test",
              {{"i1", SQLTypeInfo(kTINYINT)},
               {"i2", SQLTypeInfo(kSMALLINT)},
               {"i4", SQLTypeInfo(kINT)},
               {"i8", SQLTypeInfo(kBIGINT)},
               {"f", SQLTypeInfo(kFLOAT)},
               {"d", SQLTypeInfo(kDOUBLE)},
               {"dc", SQLTypeInfo(kDECIMAL, 15, 8, false)},
               {"n", SQLTypeInfo(kDECIMAL, 15, 8, false)},
               {"i1n", SQLTypeInfo(kTINYINT)},
               {"i2n", SQLTypeInfo(kSMALLINT)},
               {"i4n", SQLTypeInfo(kINT)},
               {"i8n", SQLTypeInfo(kBIGINT)},
               {"fn", SQLTypeInfo(kFLOAT)},
               {"dn", SQLTypeInfo(kDOUBLE)},
               {"dcn", SQLTypeInfo(kDECIMAL, 15, 8, false)},
               {"nn", SQLTypeInfo(kDECIMAL, 15, 8, false)}});
  insertCsvValues("wb_test", "1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0,,,,,,,,");
  auto drop = "DROP TABLE IF EXISTS wb_test;";
  auto create_sqlite =
      "CREATE TABLE wb_test (i1n tinyint, i2n smallint, i4n int, i8n bigint, fn float, "
      "dn "
      "double, dcn decimal(15,8), nn numeric(15,8));";
  auto insert_sqlite =
      "INSERT INTO wb_test VALUES (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);";
  sqlite_comparator_.query(drop);
  sqlite_comparator_.query(create_sqlite);
  sqlite_comparator_.query(insert_sqlite);
  auto test_queries = [](const std::string col_name) {
    std::string omnisci_query =
        "SELECT WIDTH_BUCKET(" + col_name + ", i4, i4*10, i4*10) FROM wb_test;";
    std::string sqlite_query = "SELECT " + col_name + " FROM wb_test;";
    return std::make_pair(omnisci_query, sqlite_query);
  };
  std::vector<std::string> col_names{"i1", "i2", "i4", "i8", "d", "f", "dc", "n"};
  std::vector<std::string> wrong_partition_expr{"i1-1", "-1*i1", "d", "f", "dc", "n"};
  std::vector<std::string> null_col_names{
      "i1n", "i2n", "i4n", "i8n", "dn", "fn", "dcn", "nn"};
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    for (auto& col : col_names) {
      std::string q = "SELECT WIDTH_BUCKET(i1, i1, " + col + ", i1) FROM wb_test;";
      EXPECT_ANY_THROW(run_simple_agg(q, dt));
    }

    for (auto& col : col_names) {
      std::string q =
          "SELECT WIDTH_BUCKET(i1, " + col + ", " + col + ", i1) FROM wb_test;";
      EXPECT_ANY_THROW(run_simple_agg(q, dt));
    }

    for (auto& expr : wrong_partition_expr) {
      std::string q = "SELECT WIDTH_BUCKET(i1, i1*2, i1*3, " + expr + ") FROM wb_test;";
      EXPECT_ANY_THROW(run_simple_agg(q, dt)) << q;
    }

    for (size_t i = 0; i < col_names.size(); ++i) {
      auto col = col_names[i];
      auto ncol = null_col_names[i];
      std::string q1 =
          "SELECT WIDTH_BUCKET(i1, " + col + ", " + ncol + ", i1) FROM wb_test;";
      std::string q2 =
          "SELECT WIDTH_BUCKET(i1, " + ncol + ", " + col + ", i1) FROM wb_test;";
      EXPECT_ANY_THROW(run_simple_agg(q1, dt));
      EXPECT_ANY_THROW(run_simple_agg(q2, dt));
    }

    for (auto& ncol : null_col_names) {
      std::string q = "SELECT WIDTH_BUCKET(i1, i1*2, i1*3," + ncol + ") FROM wb_test;";
      EXPECT_ANY_THROW(run_simple_agg(q, dt));
    }

    EXPECT_EQ(int64_t(5),
              v<int64_t>(run_simple_agg(
                  "SELECT WIDTH_BUCKET(i1*5, i4, i4*10, i4*10) FROM wb_test;", dt)));
    EXPECT_EQ(int64_t(6),
              v<int64_t>(run_simple_agg(
                  "SELECT WIDTH_BUCKET(i1*5, i4*10, i4, i4*10) FROM wb_test;", dt)));
    EXPECT_EQ(int64_t(5),
              v<int64_t>(run_simple_agg(
                  "SELECT WIDTH_BUCKET(i1*5, i4*10, i4, i4*10) - 1 FROM wb_test;", dt)));
    EXPECT_EQ(
        int64_t(-1),
        v<int64_t>(run_simple_agg("SELECT WIDTH_BUCKET(i1*5, i4, i4*10, i4*10) - "
                                  "WIDTH_BUCKET(i1*5, i4*10, i4, i4*10) FROM wb_test;",
                                  dt)));
    EXPECT_EQ(int64_t(12),
              v<int64_t>(run_simple_agg(
                  "select width_bucket(i2+15, cast(i2*(d+1) as int), cast(i4*(n+25) as "
                  "int), cast(i8*20 as int)) from wb_test;",
                  dt)));
    EXPECT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg(
            "SELECT WIDTH_BUCKET(i1, i4, i4*10, i4*10) b FROM wb_test GROUP BY b;", dt)));
    EXPECT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_agg(
            "SELECT WIDTH_BUCKET(i1-1, i4, i4*10, i4*10) b FROM wb_test GROUP BY b;",
            dt)));
    EXPECT_EQ(
        int64_t(11),
        v<int64_t>(run_simple_agg(
            "SELECT WIDTH_BUCKET(i1+11, i4, i4*10, i4*10) b FROM wb_test GROUP BY b;",
            dt)));

    for (auto& col : null_col_names) {
      auto queries = test_queries(col);
      c(queries.first, queries.second, dt);
    }
  }
  dropTable("wb_test");
  sqlite_comparator_.query(drop);
}

TEST_F(Select, WidthBucketWithGroupBy) {
  createTable("wb_test_nullable", {{"val", SQLTypeInfo(kINT)}});
  createTable("wb_test_non_nullable", {{"val", SQLTypeInfo(kINT, true)}});
  createTable("wb_test", {{"val", SQLTypeInfo(kINT)}});
  insertCsvValues("wb_test_nullable", "1\n2\n3");
  insertJsonValues("wb_test_nullable", "{\"val\": null}");
  insertCsvValues("wb_test_non_nullable", "1\n2\n3");
  insertCsvValues("wb_test", "1\n2\n3");
  std::vector<std::string> drop_tables;
  drop_tables.push_back("DROP TABLE IF EXISTS wb_test_nullable;");
  drop_tables.push_back("DROP TABLE IF EXISTS wb_test_non_nullable;");
  drop_tables.push_back("DROP TABLE IF EXISTS wb_test;");
  std::vector<std::string> create_tables;
  create_tables.push_back("CREATE TABLE wb_test_nullable (val int);");
  create_tables.push_back("CREATE TABLE wb_test_non_nullable (val int not null);");
  create_tables.push_back("CREATE TABLE wb_test (val int);");
  std::vector<std::string> populate_tables;
  for (int i = 1; i < 4; ++i) {
    populate_tables.push_back("INSERT INTO wb_test_nullable VALUES(" + std::to_string(i) +
                              ");");
    populate_tables.push_back("INSERT INTO wb_test_non_nullable VALUES(" +
                              std::to_string(i) + ");");
    populate_tables.push_back("INSERT INTO wb_test VALUES(" + std::to_string(i) + ");");
  }
  populate_tables.push_back("INSERT INTO wb_test_nullable VALUES(null);");
  for (const auto& stmt : drop_tables) {
    sqlite_comparator_.query(stmt);
  }
  for (const auto& stmt : create_tables) {
    sqlite_comparator_.query(stmt);
  }
  for (const auto& stmt : populate_tables) {
    sqlite_comparator_.query(stmt);
  }

  auto query_gen = [&](const std::string& table_name, bool for_omnisci, bool has_filter) {
    std::ostringstream oss;
    oss << "SELECT SUM(cnt) FROM (SELECT ";
    if (for_omnisci) {
      oss << "WIDTH_BUCKET(val, 1, 3, 3) b";
    } else {
      oss << "val b";
    }
    oss << ", COUNT(1) cnt FROM ";
    oss << table_name;
    if (has_filter) {
      oss << " WHERE val < 3";
    }
    oss << " GROUP BY b);";
    return oss.str();
  };

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c(query_gen("wb_test_nullable", true, false),
      query_gen("wb_test_nullable", false, false),
      dt);
    c(query_gen("wb_test_nullable", true, true),
      query_gen("wb_test_nullable", false, true),
      dt);
    c(query_gen("wb_test_non_nullable", true, false),
      query_gen("wb_test_non_nullable", false, false),
      dt);
    c(query_gen("wb_test_non_nullable", true, true),
      query_gen("wb_test_non_nullable", false, true),
      dt);
    c(query_gen("wb_test", true, false), query_gen("wb_test", false, false), dt);
    c(query_gen("wb_test", true, true), query_gen("wb_test", false, true), dt);
  }
  for (const auto& stmt : drop_tables) {
    sqlite_comparator_.query(stmt);
  }
  dropTable("wb_test_nullable");
  dropTable("wb_test_non_nullable");
  dropTable("wb_test");
}

TEST_F(Select, WidthBucketNullability) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    {
      // no results
      auto result = run_multiple_agg(
          R"(SELECT WIDTH_BUCKET(x, 1, 20, 3) AS w, COUNT(*) AS n FROM test GROUP BY w HAVING (w IS null);)",
          dt);
      EXPECT_EQ(result->rowCount(), size_t(0));
    }

    {
      // one null row
      // no results
      auto result = run_multiple_agg(
          R"(SELECT WIDTH_BUCKET(ofd, 1, 20, 3) AS w, COUNT(*) AS n FROM test GROUP BY w HAVING (w IS null);)",
          dt);
      EXPECT_EQ(result->rowCount(), size_t(1));
    }
  }
}

TEST_F(Select, CountWithLimitAndOffset) {
  createTable("count_test", {{"val", SQLTypeInfo(kINT)}});
  insertCsvValues("count_test", "0\n1\n2\n3\n4\n5\n6\n7\n8\n9");

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    EXPECT_EQ(int64_t(10),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test);", dt)));
    EXPECT_EQ(int64_t(9),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test OFFSET 1);", dt)));
    EXPECT_EQ(int64_t(8),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test OFFSET 2);", dt)));
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 1);", dt)));
    EXPECT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 2);", dt)));
    EXPECT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 1 OFFSET 1);", dt)));
    EXPECT_EQ(
        int64_t(2),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 2 OFFSET 1);", dt)));
    EXPECT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 2 OFFSET 9);", dt)));
    EXPECT_EQ(
        int64_t(2),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 2 OFFSET 8);", dt)));
    EXPECT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 1 OFFSET 8);", dt)));

    EXPECT_EQ(int64_t(10),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test GROUP BY val);", dt)));
    EXPECT_EQ(
        int64_t(9),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test GROUP BY val OFFSET 1);",
            dt)));
    EXPECT_EQ(
        int64_t(8),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test GROUP BY val OFFSET 2);",
            dt)));
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test GROUP BY val LIMIT 1);",
                  dt)));
    EXPECT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test GROUP BY val LIMIT 2);",
                  dt)));
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT * FROM count_test "
                                        "GROUP BY val LIMIT 1 OFFSET 1);",
                                        dt)));
    EXPECT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT * FROM count_test "
                                        "GROUP BY val LIMIT 2 OFFSET 1);",
                                        dt)));
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT * FROM count_test "
                                        "GROUP BY val LIMIT 2 OFFSET 9);",
                                        dt)));
    EXPECT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT * FROM count_test "
                                        "GROUP BY val LIMIT 2 OFFSET 8);",
                                        dt)));
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT * FROM count_test "
                                        "GROUP BY val LIMIT 1 OFFSET 8);",
                                        dt)));
  }

  // now increase the data
  {
    std::stringstream ss;
    // num_sets (-1) because data started out with 1 set of (10) items
    int64_t num_sets = static_cast<int64_t>(pow(2, 16)) - 1;
    for (int i = 0; i < num_sets; i++) {
      for (int j = 0; j < 10; j++) {
        ss << j << "\n";
      }
    }
    insertCsvValues("count_test", ss.str());
  }

  int64_t size = static_cast<int64_t>(pow(2, 16) * 10);
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    EXPECT_EQ(int64_t(size),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test);", dt)));
    EXPECT_EQ(int64_t(size - 1),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test OFFSET 1);", dt)));
    EXPECT_EQ(int64_t(size - 2),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test OFFSET 2);", dt)));
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 1);", dt)));
    EXPECT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 2);", dt)));
    EXPECT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 1 OFFSET 1);", dt)));
    EXPECT_EQ(
        int64_t(2),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 2 OFFSET 1);", dt)));
    EXPECT_EQ(
        int64_t(2),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 2 OFFSET 9);", dt)));
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 2 OFFSET " +
                      std::to_string(size - 1) + ");",
                  dt)));
    EXPECT_EQ(
        int64_t(2),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 2 OFFSET 8);", dt)));
    EXPECT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT * FROM count_test LIMIT 1 OFFSET 8);", dt)));

    EXPECT_EQ(
        int64_t(size),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT rowid FROM count_test GROUP BY rowid);", dt)));
    EXPECT_EQ(int64_t(size - 1),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT rowid FROM "
                                        "count_test GROUP BY rowid OFFSET 1);",
                                        dt)));
    EXPECT_EQ(int64_t(size - 2),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT rowid FROM "
                                        "count_test GROUP BY rowid OFFSET 2);",
                                        dt)));
    EXPECT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT rowid FROM count_test GROUP BY rowid LIMIT 1);",
            dt)));
    EXPECT_EQ(
        int64_t(2),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM (SELECT rowid FROM count_test GROUP BY rowid LIMIT 2);",
            dt)));
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT rowid FROM "
                                        "count_test GROUP BY rowid LIMIT 1 OFFSET 1);",
                                        dt)));
    EXPECT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT rowid FROM "
                                        "count_test GROUP BY rowid LIMIT 2 OFFSET 1);",
                                        dt)));
    EXPECT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT rowid FROM "
                                        "count_test GROUP BY rowid LIMIT 2 OFFSET 9);",
                                        dt)));
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT rowid FROM "
                                        "count_test GROUP BY rowid LIMIT 2 OFFSET " +
                                            std::to_string(size - 1) + ");",
                                        dt)));
    EXPECT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT rowid FROM "
                                        "count_test GROUP BY rowid LIMIT 2 OFFSET 8);",
                                        dt)));
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg("SELECT count(*) FROM (SELECT rowid FROM "
                                        "count_test GROUP BY rowid LIMIT 1 OFFSET 8);",
                                        dt)));
  }
  dropTable("count_test");
}

TEST_F(Select, CountDistinct) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(distinct x) FROM test;", dt);
    c("SELECT COUNT(distinct b) FROM test;", dt);
    c("SELECT COUNT(distinct f) FROM test;", dt);
    c("SELECT COUNT(distinct d) FROM test;", dt);
    c("SELECT COUNT(distinct str) FROM test;", dt);
    c("SELECT COUNT(distinct ss) FROM test;", dt);
    c("SELECT COUNT(distinct x + 1) FROM test;", dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x) FROM test "
      "GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x + 1) FROM "
      "test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(distinct dd) AS n FROM test GROUP BY y ORDER BY n;", dt);
    c("SELECT z, str, AVG(z), COUNT(distinct z) FROM test GROUP BY z, str ORDER BY z, "
      "str;",
      dt);
    c("SELECT AVG(z), COUNT(distinct x) AS dx FROM test GROUP BY y HAVING dx > 1;", dt);
    c("SELECT z, str, COUNT(distinct f) FROM test GROUP BY z, str ORDER BY str DESC;",
      dt);
    c("SELECT COUNT(distinct x * (50000 - 1)) FROM test;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct real_str) FROM test;", dt),
                 std::runtime_error);  // Strings must be dictionary-encoded
                                       // for COUNT(DISTINCT).
  }
}

TEST_F(Select, ApproxCountDistinct) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT APPROX_COUNT_DISTINCT(x) FROM test;",
      "SELECT COUNT(distinct x) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(x) FROM test_empty;",
      "SELECT COUNT(distinct x) FROM test_empty;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(x) FROM test_one_row;",
      "SELECT COUNT(distinct x) FROM test_one_row;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(b) FROM test;",
      "SELECT COUNT(distinct b) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(f) FROM test;",
      "SELECT COUNT(distinct f) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(d) FROM test;",
      "SELECT COUNT(distinct d) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(str) FROM test;",
      "SELECT COUNT(distinct str) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(null_str) FROM test;",
      "SELECT COUNT(distinct null_str) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(ss) FROM test WHERE ss IS NOT NULL;",
      "SELECT COUNT(distinct ss) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(x + 1) FROM test;",
      "SELECT COUNT(distinct x + 1) FROM test;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x) "
      "FROM test GROUP BY y ORDER "
      "BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x) FROM test "
      "GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x + "
      "1) FROM test GROUP BY y "
      "ORDER BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x + 1) FROM "
      "test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(dd) AS n FROM test GROUP BY y ORDER BY n;",
      "SELECT COUNT(distinct dd) AS n FROM test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT z, str, AVG(z), APPROX_COUNT_DISTINCT(z) FROM test GROUP BY z, str ORDER "
      "BY z;",
      "SELECT z, str, AVG(z), COUNT(distinct z) FROM test GROUP BY z, str ORDER BY z;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(null_str) AS n FROM test GROUP BY x ORDER BY n;",
      "SELECT COUNT(distinct null_str) AS n FROM test GROUP BY x ORDER BY n;",
      dt);
    c("SELECT z, APPROX_COUNT_DISTINCT(null_str) AS n FROM test GROUP BY z ORDER BY z, "
      "n;",
      "SELECT z, COUNT(distinct null_str) AS n FROM test GROUP BY z ORDER BY z, n;",
      dt);
    c("SELECT AVG(z), APPROX_COUNT_DISTINCT(x) AS dx FROM test GROUP BY y HAVING dx > 1;",
      "SELECT AVG(z), COUNT(distinct x) AS dx FROM test GROUP BY y HAVING dx > 1;",
      dt);
    c("SELECT approx_value, exact_value FROM (SELECT APPROX_COUNT_DISTINCT(x) AS "
      "approx_value FROM test), (SELECT "
      "COUNT(distinct x) AS exact_value FROM test);",
      "SELECT approx_value, exact_value FROM (SELECT COUNT(distinct x) AS approx_value "
      "FROM test), (SELECT "
      "COUNT(distinct x) AS exact_value FROM test);",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(x, 1) FROM test;",
      "SELECT COUNT(distinct x) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(b, 10) FROM test;",
      "SELECT COUNT(distinct b) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(f, 20) FROM test;",
      "SELECT COUNT(distinct f) FROM test;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x, 1) "
      "FROM test GROUP BY y ORDER "
      "BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x) FROM test "
      "GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x + "
      "1, 1) FROM test GROUP BY y "
      "ORDER BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x + 1) FROM "
      "test GROUP BY y ORDER BY n;",
      dt);
    // Test approx_count_distinct buffer allocation with multi-slot targets
    // sqlite does not support SAMPLE, grab the first row only
    c("SELECT SAMPLE(real_str), str, APPROX_COUNT_DISTINCT(x) FROM test WHERE real_str = "
      "'real_bar' GROUP BY str;",
      "SELECT real_str, str, COUNT( distinct x) FROM test WHERE real_str = "
      "'real_bar' GROUP BY str;",
      dt);
    c("SELECT SAMPLE(real_str), str, APPROX_COUNT_DISTINCT(x) FROM test WHERE real_str = "
      "'real_foo' GROUP BY str;",
      "SELECT real_str, str, COUNT(distinct x) FROM test WHERE real_str = "
      "'real_foo' GROUP BY str, real_str;",
      dt);

    EXPECT_NO_THROW(run_multiple_agg(
        "SELECT APPROX_COUNT_DISTINCT(x), SAMPLE(real_str) FROM test GROUP BY x;", dt));
    EXPECT_THROW(
        run_multiple_agg("SELECT APPROX_COUNT_DISTINCT(real_str) FROM test;", dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT APPROX_COUNT_DISTINCT(x, 0) FROM test;", dt),
                 std::runtime_error);
  }
}

TEST_F(Select, ApproxMedianSanity) {
  auto dt = ExecutorDeviceType::CPU;
  auto approx_median = [dt](std::string const col) {
    std::string const query = "SELECT APPROX_MEDIAN(" + col + ") FROM test;";
    return v<double>(run_simple_agg(query, dt));
  };
  EXPECT_EQ(-7.5, approx_median("w"));
  EXPECT_EQ(7.0, approx_median("x"));
  EXPECT_EQ(42.5, approx_median("y"));
  EXPECT_EQ(101.0, approx_median("z"));
  EXPECT_EQ(1001.5, approx_median("t"));
  EXPECT_EQ((double(1.1f) + double(1.2f)) / 2, approx_median("f"));
  EXPECT_EQ((double(1.1f) + double(101.2f)) / 2, approx_median("ff"));
  EXPECT_EQ((double(-101.2f) + double(-1000.3f)) / 2, approx_median("fn"));
  EXPECT_EQ(2.3, approx_median("d"));
  EXPECT_EQ(-1111.5, approx_median("dn"));
  EXPECT_EQ((11110.0 / 100 + 22220.0 / 100) / 2, approx_median("dd"));
  EXPECT_EQ((11110.0 / 100 + 22220.0 / 100) / 2, approx_median("dd_notnull"));
  EXPECT_EQ(NULL_DOUBLE, approx_median("u"));
  EXPECT_EQ(2147483647.0, approx_median("ofd"));
  EXPECT_EQ(-2147483647.5, approx_median("ufd"));
  EXPECT_EQ(4611686018427387904.0, approx_median("ofq"));
  EXPECT_EQ(-4611686018427387904.5, approx_median("ufq"));
  EXPECT_EQ(32767.0, approx_median("smallint_nulls"));
}

TEST_F(Select, ApproxMedianLargeInts) {
  auto dt = ExecutorDeviceType::CPU;
  auto approx_median = [dt](std::string const col) {
    std::string const query =
        "SELECT APPROX_MEDIAN(" + col + ") FROM test_approx_median;";
    return v<double>(run_simple_agg(query, dt));
  };
  createTable("test_approx_median", {{"b", SQLTypeInfo(kBIGINT)}});
  insertCsvValues("test_approx_median", "-9223372036854775807\n9223372036854775807");
  EXPECT_EQ(0.0, approx_median("b"));
  dropTable("test_approx_median");
}

TEST_F(Select, ApproxMedianSort) {
  auto const dt = ExecutorDeviceType::CPU;
  char const* const prefix =
      "SELECT t2.x, APPROX_MEDIAN(t0.x) am FROM coalesce_cols_test_2 t2 LEFT JOIN "
      "coalesce_cols_test_0 t0 ON t2.x=t0.x GROUP BY t2.x ORDER BY am ";
  std::vector<std::string> const tests{
      "ASC NULLS FIRST", "ASC NULLS LAST", "DESC NULLS FIRST", "DESC NULLS LAST"};
  constexpr size_t NROWS = 20;
  for (size_t t = 0; t < tests.size(); ++t) {
    std::string const query = prefix + tests[t] + ", x;";
    auto rows = run_multiple_agg(query, dt);
    EXPECT_EQ(rows->colCount(), 2u) << query;
    EXPECT_EQ(rows->rowCount(), NROWS) << query;
    for (size_t i = 0; i < NROWS; ++i) {
      switch (t) {
        case 0:
          if (i < 10) {
            EXPECT_EQ(v<int64_t>(rows->getRowAt(i, 0, true)), int64_t(i) + 10)
                << query << "i=" << i;
            EXPECT_EQ(v<double>(rows->getRowAt(i, 1, true)), NULL_DOUBLE)
                << query << "i=" << i;
          } else {
            EXPECT_EQ(v<int64_t>(rows->getRowAt(i, 0, true)), int64_t(i) - 10)
                << query << "i=" << i;
            EXPECT_EQ(v<double>(rows->getRowAt(i, 1, true)), double(i) - 10)
                << query << "i=" << i;
          }
          break;
        case 1:
          EXPECT_EQ(v<int64_t>(rows->getRowAt(i, 0, true)), int64_t(i))
              << query << "i=" << i;
          if (i < 10) {
            EXPECT_EQ(v<double>(rows->getRowAt(i, 1, true)), double(i))
                << query << "i=" << i;
          } else {
            EXPECT_EQ(v<double>(rows->getRowAt(i, 1, true)), NULL_DOUBLE)
                << query << "i=" << i;
          }
          break;
        case 2:
          if (i < 10) {
            EXPECT_EQ(v<int64_t>(rows->getRowAt(i, 0, true)), int64_t(i) + 10)
                << query << "i=" << i;
            EXPECT_EQ(v<double>(rows->getRowAt(i, 1, true)), NULL_DOUBLE)
                << query << "i=" << i;
          } else {
            EXPECT_EQ(v<int64_t>(rows->getRowAt(i, 0, true)), 19 - int64_t(i))
                << query << "i=" << i;
            EXPECT_EQ(v<double>(rows->getRowAt(i, 1, true)), 19 - double(i))
                << query << "i=" << i;
          }
          break;
        case 3:
          if (i < 10) {
            EXPECT_EQ(v<int64_t>(rows->getRowAt(i, 0, true)), 9 - int64_t(i))
                << query << "i=" << i;
            EXPECT_EQ(v<double>(rows->getRowAt(i, 1, true)), 9 - double(i))
                << query << "i=" << i;
          } else {
            EXPECT_EQ(v<int64_t>(rows->getRowAt(i, 0, true)), int64_t(i))
                << query << "i=" << i;
            EXPECT_EQ(v<double>(rows->getRowAt(i, 1, true)), NULL_DOUBLE)
                << query << "i=" << i;
          }
          break;
        default:
          EXPECT_TRUE(false) << t;
      }
    }
  }
}

// APPROX_PERCENTILE is exact when the number of rows is low.
TEST_F(Select, ApproxPercentileExactValues) {
  auto const dt = ExecutorDeviceType::CPU;
  // clang-format off
  double tests[][2]{{0.0, 2.2}, {0.25, 2.2}, {0.45, 2.2}, {0.5, 2.3}, {0.55, 2.4},
                    {0.7, 2.4}, {0.75, 2.5}, {0.8, 2.6}, {1.0, 2.6}};
  // clang-format on
  for (auto test : tests) {
    std::stringstream query;
    query << "SELECT APPROX_PERCENTILE(d," << test[0] << ") FROM test;";
    EXPECT_EQ(test[1], v<double>(run_simple_agg(query.str(), dt)));
  }
}

// APPROX_QUANTILE is exact when the number of rows is low.
TEST_F(Select, ApproxQuantileExactValues) {
  auto const dt = ExecutorDeviceType::CPU;
  // clang-format off
  double tests[][2]{{0.0, 2.2}, {0.25, 2.2}, {0.45, 2.2}, {0.5, 2.3}, {0.55, 2.4},
                    {0.7, 2.4}, {0.75, 2.5}, {0.8, 2.6}, {1.0, 2.6}};
  // clang-format on
  for (auto test : tests) {
    std::stringstream query;
    query << "SELECT APPROX_QUANTILE(d," << test[0] << ") FROM test;";
    EXPECT_EQ(test[1], v<double>(run_simple_agg(query.str(), dt)));
  }
}

TEST_F(Select, ApproxPercentileMinMax) {
  auto const dt = ExecutorDeviceType::CPU;
  // clang-format off
  char const* cols[]{"w", "x", "y", "z", "t", "f", "ff", "fn", "d", "dn", "dd",
                     "dd_notnull", "u", "ofd", "ufd", "ofq", "ufq", "smallint_nulls"};
  // clang-format on
  for (std::string col : cols) {
    c("SELECT APPROX_PERCENTILE(" + col + ",0) FROM test;",
      // MIN(ofq) = -1 but MIN(CAST(ofq AS DOUBLE)) = -2^63 due to null sentinel logic
      //"SELECT CAST(MIN(" + col + ") AS DOUBLE) FROM test;",
      "SELECT MIN(CAST(" + col + " AS DOUBLE)) FROM test;",
      dt);
    c("SELECT APPROX_PERCENTILE(" + col + ",1) FROM test;",
      "SELECT CAST(MAX(" + col + ") AS DOUBLE) FROM test;",
      dt);
  }
}

TEST_F(Select, ApproxPercentileSubqueries) {
  auto const dt = ExecutorDeviceType::CPU;
  const char* query =
      "SELECT MIN(am) FROM (SELECT x, APPROX_MEDIAN(w) AS am FROM test GROUP BY x);";
  EXPECT_EQ(-8.0, v<double>(run_simple_agg(query, dt)));
  query =
      "SELECT MIN(am) FROM (SELECT x, APPROX_PERCENTILE(w,0.5) AS am FROM test GROUP "
      "BY x);";
  EXPECT_EQ(-8.0, v<double>(run_simple_agg(query, dt)));
  query = "SELECT MAX(am) FROM (SELECT x, APPROX_MEDIAN(w) AS am FROM test GROUP BY x);";
  EXPECT_EQ(-7.0, v<double>(run_simple_agg(query, dt)));
  query =
      "SELECT MAX(am) FROM (SELECT x, APPROX_PERCENTILE(w,0.5) AS am FROM test GROUP "
      "BY x);";
  EXPECT_EQ(-7.0, v<double>(run_simple_agg(query, dt)));
}

// Immerse invokes sql_validate which requires testing.
TEST_F(Select, ApproxPercentileValidate) {
  auto const dt = ExecutorDeviceType::CPU;
  auto eo = getExecutionOptions(true);
  eo.just_validate = true;
  // APPROX_MEDIAN
  char const* query = "SELECT APPROX_MEDIAN(x) FROM test;";
  std::shared_ptr<ResultSet> rows = runSqlQuery(query, dt, std::move(eo)).getRows();
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(1u, crt_row.size()) << query;
  EXPECT_EQ(NULL_DOUBLE, v<double>(crt_row[0]));
  // APPROX_PERCENTILE
  query = "SELECT APPROX_PERCENTILE(x,0.1) FROM test;";
  rows = runSqlQuery(query, dt, std::move(eo)).getRows();
  crt_row = rows->getNextRow(true, true);
  CHECK_EQ(1u, crt_row.size()) << query;
  EXPECT_EQ(NULL_DOUBLE, v<double>(crt_row[0]));
}

TEST_F(Select, ScanNoAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT * FROM test ORDER BY x ASC, y ASC;", dt);
    c("SELECT t.* FROM test t ORDER BY x ASC, y ASC;", dt);
    c("SELECT x, z, t FROM test ORDER BY x ASC, y ASC;", dt);
    c("SELECT x, y, x + 1 FROM test ORDER BY x ASC, y ASC;", dt);
    c("SELECT x + z, t FROM test WHERE x <> 7 AND y > 42;", dt);
    c("SELECT * FROM test WHERE x > 8;", dt);
    c("SELECT fx FROM test WHERE fx IS NULL;", dt);
    c("SELECT z,t,f,m,d,x,real_str,u,z,y FROM test WHERE z = -78 AND t = "
      "1002 AND x >= 8 AND y = 43 AND d > 1.0 AND f > 1.0 AND real_str = 'real_bar' "
      "ORDER BY f ASC;",
      dt);
    c("SELECT * FROM test WHERE d > 2.4 AND real_str IS NOT NULL AND fixed_str IS NULL "
      "AND z = 102 AND fn < 0 AND y = 43 AND t >= 0 AND x <> 8;",
      dt);
    c("SELECT * FROM test WHERE d > 2.4 AND real_str IS NOT NULL AND fixed_str IS NULL "
      "AND z = 102 AND fn < 0 AND y = 43 AND t >= 0 AND x = 8;",
      dt);
    c("SELECT real_str,f,fn,y,d,x,z,str,fixed_str,t,dn FROM test WHERE f IS NOT NULL AND "
      "y IS NOT NULL AND str = 'bar' AND x >= 7 AND t < 1003 AND z < 0;",
      dt);
    c("SELECT t,y,str,real_str,d,fixed_str,dn,fn,z,f,x FROM test WHERE f IS NOT NULL AND "
      "y IS NOT NULL AND str = 'baz' AND x >= 7 AND t < 1003 AND f > 1.2 LIMIT 1;",
      dt);
    c("SELECT fn,real_str,str,z,d,x,fixed_str,dn,y,t,f FROM test WHERE f < 1.4 AND "
      "real_str IS NOT NULL AND fixed_str IS NULL AND z = 102 AND dn < 0 AND y = 43;",
      dt);
    c("SELECT dn,str,y,z,fixed_str,fn,d,real_str,t,f,x FROM test WHERE z < 0 AND f < 2 "
      "AND d > 2.0 AND fn IS NOT NULL AND dn < 2000 AND str IS NOT NULL AND fixed_str = "
      "'bar' AND real_str = 'real_bar' AND t >= 1001 AND y >= 42 AND x > 7 ORDER BY z, "
      "x;",
      dt);
    c("SELECT z,f,d,str,real_str,x,dn,y,t,fn,fixed_str FROM test WHERE fn IS NULL AND dn "
      "IS NULL AND x >= 0 AND real_str = 'real_foo' ORDER BY y;",
      dt);
  }
}

TEST_F(Select, OrderBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    const auto rows = run_multiple_agg(
        "SELECT x, y, z + t, x * y AS m FROM test ORDER BY 3 desc LIMIT 5;", dt);
    CHECK_EQ(rows->rowCount(), std::min(size_t(5), static_cast<size_t>(g_num_rows)) + 0);
    CHECK_EQ(rows->colCount(), size_t(4));
    for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
      ASSERT_TRUE(v<int64_t>(rows->getRowAt(row_idx, 0, true)) == 8 ||
                  v<int64_t>(rows->getRowAt(row_idx, 0, true)) == 7);
      ASSERT_EQ(v<int64_t>(rows->getRowAt(row_idx, 1, true)), 43);
      ASSERT_EQ(v<int64_t>(rows->getRowAt(row_idx, 2, true)), 1104);
      ASSERT_TRUE(v<int64_t>(rows->getRowAt(row_idx, 3, true)) == 344 ||
                  v<int64_t>(rows->getRowAt(row_idx, 3, true)) == 301);
    }
    c("SELECT x, COUNT(distinct y) AS n FROM test GROUP BY x ORDER BY n DESC;", dt);
    c("SELECT x x1, x, COUNT(*) AS val FROM test GROUP BY x HAVING val > 5 ORDER BY val "
      "DESC LIMIT 5;",
      dt);
    c("SELECT ufd, COUNT(*) n FROM test GROUP BY ufd, str ORDER BY ufd, n;", dt);
    c("SELECT -x, COUNT(*) FROM test GROUP BY x ORDER BY x DESC;", dt);
    c("SELECT real_str FROM test WHERE real_str LIKE '%real%' ORDER BY real_str ASC;",
      dt);
    c("SELECT ss FROM test GROUP by ss ORDER BY ss ASC NULLS FIRST;",
      "SELECT ss FROM test GROUP by ss ORDER BY ss ASC;",
      dt);
    c("SELECT str, COUNT(*) n FROM test WHERE x < 0 GROUP BY str ORDER BY n DESC LIMIT "
      "5;",
      dt);
    c("SELECT x FROM test ORDER BY x LIMIT 50;", dt);
    c("SELECT x FROM test ORDER BY x LIMIT 5;", dt);
    c("SELECT x FROM test ORDER BY x ASC LIMIT 20;", dt);
    c("SELECT dd FROM test ORDER BY dd ASC LIMIT 20;", dt);
    c("SELECT f FROM test ORDER BY f ASC LIMIT 5;", dt);
    c("SELECT f FROM test ORDER BY f ASC LIMIT 20;", dt);
    c("SELECT fn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT fn as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT fn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT fn as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT dn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT dn as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT dn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT dn as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT ff as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT ff as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT ff as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT ff as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT d as k FROM test ORDER BY k ASC LIMIT 5;", dt);
    c("SELECT d as k FROM test ORDER BY k ASC LIMIT 20;", dt);
    c("SELECT dn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT dn as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT dn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT dn as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT ofq AS k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT ofq as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT ofq AS k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT ofq as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT ufq as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT ufq as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT ufq as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT ufq as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT CAST(ofd AS FLOAT) as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT CAST(ofd AS FLOAT) as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT CAST(ofd AS FLOAT) as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT CAST(ofd AS FLOAT) as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT CAST(ufd AS FLOAT) as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT CAST(ufd AS FLOAT) as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT CAST(ufd AS FLOAT) as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT CAST(ufd AS FLOAT) as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT m AS k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT m AS k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT n AS k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT n AS k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT o AS k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT o AS k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    for (std::string order : {"ASC", "DESC"}) {
      c("SELECT d, MAX(f) FROM test WHERE f IS NOT NULL GROUP BY d ORDER BY 2 " + order +
            " LIMIT "
            "1;",
        dt);
      c("SELECT d, AVG(f) FROM test WHERE f IS NOT NULL GROUP BY d ORDER BY 2 " + order +
            " LIMIT "
            "1;",
        dt);
      c("SELECT d, SUM(f) FROM test WHERE f IS NOT NULL GROUP BY d ORDER BY 2 " + order +
            " LIMIT "
            "1;",
        dt);
      c("SELECT d, MAX(f) FROM test GROUP BY d ORDER BY 2 " + order + " LIMIT 1;", dt);
      c("SELECT x, y, MAX(f) FROM test GROUP BY x, y ORDER BY 3 " + order + " LIMIT 1;",
        dt);
      c("SELECT x, y, SUM(f) FROM test WHERE f IS NOT NULL GROUP BY x, y ORDER BY 3 " +
            order + " LIMIT 1;",
        dt);
      for (std::string nulls : {" NULLS LAST", " NULLS FIRST"}) {
        char const* const prefix =
            "SELECT t2.x, t0.x FROM coalesce_cols_test_2 t2 LEFT JOIN "
            "coalesce_cols_test_0 t0 ON t2.x=t0.x ORDER BY t0.x ";
        std::string query = prefix + order + nulls + ", t2.x ASC NULLS LAST;";
        c(query, dt);
      }
    }
    c("SELECT * FROM ( SELECT x, y FROM test ORDER BY x, y ASC NULLS FIRST LIMIT 10 ) t0 "
      "LIMIT 5;",
      "SELECT * FROM ( SELECT x, y FROM test ORDER BY x, y ASC LIMIT 10 ) t0 LIMIT 5;",
      dt);
    c(R"(SELECT str, COUNT(*) FROM test GROUP BY str ORDER BY 2 DESC NULLS FIRST LIMIT 50 OFFSET 10;)",
      R"(SELECT str, COUNT(*) FROM test GROUP BY str ORDER BY 2 DESC LIMIT 50 OFFSET 10;)",
      dt);
  }
}

TEST_F(Select, VariableLengthOrderBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT real_str FROM test ORDER BY real_str;", dt);
    EXPECT_THROW(
        run_multiple_agg("SELECT arr_float FROM array_test ORDER BY arr_float;", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT arr3_i16 FROM array_test ORDER BY arr3_i16 DESC;", dt),
        std::runtime_error);
  }
}

TEST_F(Select, TopKHeap) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT str, x FROM proj_top ORDER BY x DESC LIMIT 1;", dt);
  }
}

TEST_F(Select, ComplexQueries) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) * MAX(y) - SUM(z) FROM test;", dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 "
      "AND 200 GROUP BY x, y;",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 "
      "AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x);",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 "
      "AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x) + 35;",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 "
      "AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x) + 36;",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test "
      "WHERE z BETWEEN 100 AND 200 GROUP BY a, y;",
      dt);
    c("SELECT x, y FROM (SELECT a.str AS str, b.x AS x, a.y AS y FROM test a, join_test "
      "b WHERE a.x = b.x) WHERE str = "
      "'foo' ORDER BY x LIMIT 1;",
      dt);
    const auto rows = run_multiple_agg(
        "SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test "
        "WHERE z BETWEEN 100 AND 200 GROUP BY x, y ORDER BY a DESC LIMIT 2;",
        dt);
    ASSERT_EQ(rows->rowCount(), size_t(2));
    {
      auto crt_row = rows->getNextRow(true, true);
      CHECK_EQ(size_t(2), crt_row.size());
      ASSERT_EQ(v<int64_t>(crt_row[0]), 50);
      ASSERT_EQ(v<int64_t>(crt_row[1]), -295);
    }
    {
      auto crt_row = rows->getNextRow(true, true);
      CHECK_EQ(size_t(2), crt_row.size());
      ASSERT_EQ(v<int64_t>(crt_row[0]), 49);
      ASSERT_EQ(v<int64_t>(crt_row[1]), -590);
    }
    auto empty_row = rows->getNextRow(true, true);
    CHECK(empty_row.empty());
  }
}

TEST_F(Select, MultiStepQueries) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    const auto skip_intermediate_count = g_skip_intermediate_count;
    ScopeGuard reset_skip_intermediate_count = [&skip_intermediate_count] {
      g_skip_intermediate_count = skip_intermediate_count;
    };

    c("SELECT z, (z * SUM(x)) / SUM(y) + 1 FROM test GROUP BY z ORDER BY z;", dt);
    c("SELECT z,COUNT(*), AVG(x) / SUM(y) + 1 FROM test GROUP BY z ORDER BY z;", dt);
  }
}

TEST_F(Select, GroupByPushDownFilterIntoExprRange) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    const auto rows = run_multiple_agg(
        "SELECT b, COUNT(*) AS n FROM test WHERE b GROUP BY b ORDER BY b", dt);
    ASSERT_EQ(
        size_t(1),
        rows->rowCount());  // Sqlite does not have a boolean type, so do this for now
    c("SELECT x, COUNT(*) AS n FROM test WHERE x > 7 GROUP BY x ORDER BY x", dt);
    c("SELECT y, COUNT(*) AS n FROM test WHERE y < 43 GROUP BY y ORDER BY n DESC", dt);
    c("SELECT z, COUNT(*) AS n FROM test WHERE z <= 43 AND y > 10 GROUP BY z ORDER BY n "
      "DESC",
      dt);
    c("SELECT t, SUM(y) AS sum_y FROM test WHERE t < 2000 GROUP BY t ORDER BY t DESC",
      dt);
    c("SELECT t, SUM(y) AS sum_y FROM test WHERE t < 2000 GROUP BY t ORDER BY sum_y", dt);
    c("SELECT o, COUNT(*) as n FROM test WHERE o <= '1999-09-09' GROUP BY o ORDER BY n",
      dt);
    c("SELECT t + x, AVG(x) AS avg_x FROM test WHERE z <= 50 and t < 2000 GROUP BY t + x "
      "ORDER BY avg_x DESC",
      dt);
  }
}

TEST_F(Select, GroupByExprNoFilterNoAggregate) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x + y AS a FROM test GROUP BY a ORDER BY a;", dt);
    ASSERT_EQ(8,
              v<int64_t>(run_simple_agg("SELECT TRUNCATE(x, 0) AS foo FROM test GROUP BY "
                                        "TRUNCATE(x, 0) ORDER BY foo DESC LIMIT 1;",
                                        dt)));
  }
}

TEST_F(Select, DistinctProjection) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT DISTINCT str FROM test ORDER BY str;", dt);
    c("SELECT DISTINCT(str), SUM(x) FROM test WHERE x > 7 GROUP BY str LIMIT 2;", dt);
  }
}

TEST_F(Select, Case) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE "
      "3 END) FROM test;",
      dt);
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 END) FROM test;", dt);
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE "
      "3 END) "
      "FROM test WHERE CASE WHEN y BETWEEN 42 AND 43 THEN 5 ELSE 4 END > 4;",
      dt);
    ASSERT_EQ(std::numeric_limits<int64_t>::min(),
              v<int64_t>(run_simple_agg(
                  "SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 "
                  "THEN 2 ELSE 3 END) FROM test "
                  "WHERE CASE WHEN y BETWEEN 44 AND 45 THEN 5 ELSE 4 END > 4;",
                  dt)));
    c("SELECT CASE WHEN x + y > 50 THEN 77 ELSE 88 END AS foo, COUNT(*) FROM test GROUP "
      "BY foo ORDER BY foo;",
      dt);
    ASSERT_EQ(std::numeric_limits<double>::min(),
              v<double>(run_simple_agg(
                  "SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1.1 WHEN x BETWEEN 8 AND "
                  "9 THEN 2.2 ELSE 3.3 END) FROM "
                  "test WHERE CASE WHEN y BETWEEN 44 AND 45 THEN 5.1 ELSE 3.9 END > 4;",
                  dt)));
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN 'oops 1' WHEN x BETWEEN 4 AND 6 THEN "
      "'oops 2' ELSE real_str END c "
      "FROM "
      "test ORDER BY c ASC;",
      dt);

    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN 'oops 1' WHEN x BETWEEN 4 AND 6 THEN "
      "'oops 2' ELSE str END c FROM "
      "test "
      "ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN "
      "'eight' ELSE 'ooops' END c FROM "
      "test ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN "
      "real_str ELSE 'ooops' END AS g "
      "FROM test ORDER BY g ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN str "
      "ELSE 'ooops' END c FROM test "
      "ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN "
      "'eight' ELSE 'ooops' END c FROM "
      "test ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN str WHEN x BETWEEN 7 AND 10 THEN 'eight' "
      "ELSE 'ooops' END AS g, "
      "COUNT(*) FROM test GROUP BY g ORDER BY g;",
      dt);
    c(R"(SELECT CASE WHEN x = 7 THEN 'a' WHEN x = 8 then 'b' ELSE shared_dict END FROM test GROUP BY 1 ORDER BY 1 ASC;)",
      dt);
    c(R"(SELECT CASE WHEN x = 7 THEN 'a' WHEN x = 8 then str ELSE str END FROM test GROUP BY 1 ORDER BY 1 ASC;)",
      dt);
    c(R"(SELECT CASE WHEN x = 7 THEN 'a' WHEN x = 8 then str ELSE shared_dict END FROM test GROUP BY 1 ORDER BY 1 ASC;)",
      dt);
    c(R"(SELECT COUNT(*) FROM (SELECT CASE WHEN x = 7 THEN 1 WHEN x = 8 then str ELSE shared_dict END FROM test GROUP BY 1);)",
      dt);
    c(R"(SELECT COUNT(*) FROM test WHERE (CASE WHEN x = 7 THEN str ELSE 'b' END) = shared_dict;)",
      dt);
    c(R"(SELECT COUNT(*) FROM test WHERE (CASE WHEN str = 'foo' THEN 'a' WHEN str = 'bar' THEN 'b' ELSE str END) = 'b';)",
      dt);
    c(R"(SELECT str, count(*) FROM test WHERE (CASE WHEN str = 'foo' THEN 'a' WHEN str = 'bar' THEN 'b' ELSE str END) = 'b' GROUP BY str;)",
      dt);
    c(R"(SELECT COUNT(*) FROM test WHERE (CASE WHEN fixed_str = 'foo' THEN 'a' WHEN fixed_str is NULL THEN 'b' ELSE str END) = 'z';)",
      dt);
    {
      const auto watchdog_state = g_enable_watchdog;
      g_enable_watchdog = true;
      ScopeGuard reset_Watchdog_state = [&watchdog_state] {
        g_enable_watchdog = watchdog_state;
      };
      EXPECT_ANY_THROW(run_multiple_agg(
          R"(SELECT CASE WHEN x = 7 THEN 'a' WHEN x = 8 then str ELSE fixed_str END FROM test;)",
          dt));  // Cast from dictionary-encoded string to none-encoded would
                 // be slow
      g_enable_watchdog = false;
      // casts not yet supported in distributed mode
      c(R"(SELECT CASE WHEN x = 7 THEN 'a' WHEN x = 8 then str ELSE fixed_str END FROM test ORDER BY 1;)",
        dt);
      c(R"(SELECT CASE WHEN str = 'foo' THEN real_str WHEN str = 'bar' THEN 'b' ELSE null_str END FROM test ORDER BY 1)",
        dt);
      EXPECT_ANY_THROW(run_multiple_agg(
          R"(SELECT CASE WHEN str = 'foo' THEN real_str WHEN str = 'bar' THEN 'b' ELSE null_str END case_col, sum(x) FROM test GROUP BY case_col;)",
          dt));  // cannot group by none encoded string columns
    }
    c("SELECT y AS key0, SUM(CASE WHEN x > 7 THEN x / (x - 7) ELSE 99 END) FROM test "
      "GROUP BY key0 ORDER BY key0;",
      dt);
    ASSERT_NO_THROW(run_multiple_agg(
        "SELECT y AS key0, CASE WHEN y > 7 THEN STDDEV(x) ELSE 99 END FROM test "
        "GROUP BY y ORDER BY y;",
        dt,
        false));
    ASSERT_NO_THROW(run_multiple_agg(
        "SELECT y AS key0, CASE WHEN y > 7 THEN 1 ELSE STDDEV(x) END FROM test "
        "GROUP BY y ORDER BY y;",
        dt,
        false));
    c("SELECT CASE WHEN str IN ('str1', 'str3', 'str8') THEN 'foo' WHEN str IN ('str2', "
      "'str4', 'str9') THEN 'bar' "
      "ELSE 'baz' END AS bucketed_str, COUNT(*) AS n FROM query_rewrite_test GROUP BY "
      "bucketed_str ORDER BY n "
      "DESC;",
      dt);
    c("SELECT CASE WHEN y > 40 THEN x END c, x FROM test ORDER BY c ASC;", dt);
    c("SELECT COUNT(CASE WHEN str = 'foo' THEN 1 END) FROM test;", dt);
    c("SELECT COUNT(CASE WHEN str = 'foo' THEN 1 ELSE NULL END) FROM test;", dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN y ELSE y END AS foobar FROM test ORDER BY "
      "foobar DESC;",
      dt);
    c("SELECT x, AVG(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x "
      "ORDER BY x;",
      dt);
    c("SELECT x, SUM(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x "
      "ORDER BY x;",
      dt);
    c("SELECT x, COUNT(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x "
      "ORDER BY x;",
      dt);
    c("SELECT CASE WHEN x > 8 THEN 'oops' ELSE 'ok' END FROM test LIMIT 1;", dt);
    c("SELECT CASE WHEN x < 9 THEN 'ok' ELSE 'oops' END FROM test LIMIT 1;", dt);
    c("SELECT CASE WHEN str IN ('foo', 'bar') THEN str END key1, COUNT(*) FROM test "
      "GROUP BY str HAVING key1 IS NOT "
      "NULL ORDER BY key1;",
      dt);

    c("SELECT CASE WHEN str IN ('foo') THEN 'FOO' WHEN str IN ('bar') THEN 'BAR' ELSE "
      "'BAZ' END AS g, COUNT(*) "
      "FROM test GROUP BY g ORDER BY g DESC;",
      dt);
    c("SELECT x, COUNT(case when y = 42 then 1 else 0 end) AS n1, COUNT(*) AS n2 FROM "
      "test GROUP BY x ORDER BY n2 "
      "DESC;",
      dt);
    c("SELECT CASE WHEN test.str = 'foo' THEN 'foo' ELSE test.str END AS g FROM test "
      "GROUP BY g ORDER BY g ASC;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE CASE WHEN x > 8 THEN 'oops' END = 'oops' OR CASE "
      "WHEN x > 8 THEN 'oops' END = 'oops';",
      dt);
    ASSERT_EQ(
        int64_t(1418428800),
        v<int64_t>(run_simple_agg(
            "SELECT CASE WHEN 1 > 0 THEN DATE_TRUNC(day, m) ELSE DATE_TRUNC(year, m) END "
            "AS date_bin FROM test GROUP BY date_bin;",
            dt)));
    ASSERT_EQ(
        int64_t(1388534400),
        v<int64_t>(run_simple_agg(
            "SELECT CASE WHEN 1 < 0 THEN DATE_TRUNC(day, m) ELSE DATE_TRUNC(year, m) END "
            "AS date_bin FROM test GROUP BY date_bin;",
            dt)));
    c("SELECT COUNT(CASE WHEN str IN ('foo', 'bar') THEN 'foo_bar' END) from test;", dt);
    ASSERT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg(
            "SELECT MIN(CASE WHEN x BETWEEN 7 AND 8 THEN true ELSE false END) FROM test;",
            dt)));
    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_agg(
            "SELECT MIN(CASE WHEN x BETWEEN 6 AND 7 THEN true ELSE false END) FROM test;",
            dt)));
    c("SELECT CASE WHEN test.str in ('boo', 'simple', 'case', 'not', 'much', 'to', "
      "'see', 'foo_in_case', 'foo', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', "
      "'k', 'l') THEN 'foo_in_case' ELSE test.str END AS g FROM test GROUP BY g ORDER BY "
      "g ASC;",
      dt);
    c("SELECT CASE WHEN shared_dict is null THEN 'hello' ELSE 'world' END key0, count(*) "
      "val FROM test GROUP BY key0 ORDER BY val;",
      dt);
    c("WITH distinct_x AS (SELECT x FROM test GROUP BY x) SELECT SUM(CASE WHEN x = 7 "
      "THEN -32767 ELSE -1 END) FROM distinct_x",
      dt);
    ASSERT_NO_THROW(run_multiple_agg(
        "WITH distinct_x AS (SELECT x FROM test GROUP BY x) SELECT CASE WHEN x = 7 "
        "THEN STDDEV(x) ELSE -1 END FROM distinct_x GROUP BY x;",
        dt,
        false));
    ASSERT_NO_THROW(run_multiple_agg(
        "WITH distinct_x AS (SELECT x FROM test GROUP BY x) SELECT CASE WHEN x = 7 "
        "THEN -32767 ELSE STDDEV(x) END FROM distinct_x GROUP BY x;",
        dt,
        false));
    ASSERT_NO_THROW(run_multiple_agg(
        "WITH distinct_x AS (SELECT x FROM test GROUP BY x) SELECT CASE WHEN x = 7 "
        "THEN -32767 ELSE STDDEV(x) END as V FROM distinct_x GROUP BY x ORDER BY V;",
        dt,
        false));
    ASSERT_NO_THROW(run_multiple_agg(
        "WITH distinct_x AS (SELECT x FROM test GROUP BY x) SELECT CASE WHEN x = 7 "
        "THEN STDDEV(x) ELSE -1 END as V FROM distinct_x GROUP BY x ORDER BY V;",
        dt,
        false));
    c("WITH distinct_x AS (SELECT x FROM test GROUP BY x) SELECT AVG(CASE WHEN x = 7 "
      "THEN -32767 ELSE -1 END) FROM distinct_x",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN '1' WHEN x BETWEEN 8 AND 10 THEN '2' ELSE "
      "real_str END AS c FROM test WHERE y IN (43) ORDER BY c ASC;",
      dt);
    c("SELECT ROUND(a.numerator / a.denominator, 2) FROM (SELECT SUM(CASE WHEN y > 42 "
      "THEN 1.0 ELSE 0.0 END) as numerator, SUM(CASE WHEN dd > 0 THEN 1 ELSE -1 END) as "
      "denominator, y FROM test GROUP BY y ORDER BY y) a",
      dt);
    c("SELECT ROUND((numerator / denominator) * 100, 2) FROM (SELECT "
      "SUM(CASE WHEN a.x > 0 THEN "
      "1 ELSE 0 END) as numerator, SUM(CASE WHEN a.dd < 0 "
      "THEN 0.5 ELSE -0.5 END) as denominator "
      "FROM test a, test_inner b where a.x = b.x) test_sub",
      dt);
    EXPECT_EQ(
        int64_t(-1),
        v<int64_t>(run_simple_agg("SELECT ROUND(numerator / denominator, 2) FROM (SELECT "
                                  "SUM(CASE WHEN a.x > 0 THEN "
                                  "1 ELSE 0 END) as numerator, SUM(CASE WHEN a.dd < 0 "
                                  "THEN 1 ELSE -1 END) as denominator "
                                  "FROM test a, test_inner b where a.x = b.x) test_sub",
                                  dt)));
    EXPECT_EQ(
        double(100),
        v<double>(run_simple_agg(
            "SELECT CEIL((a.numerator / a.denominator) * 100) as c FROM (SELECT SUM(CASE "
            "WHEN "
            "y > 42 "
            "THEN 1.0 ELSE 0.0 END) as numerator, SUM(CASE WHEN dd > 0 THEN 1 ELSE "
            "-1 END) as "
            "denominator, y FROM test GROUP BY y ORDER BY y) a GROUP BY c HAVING c > 0",
            dt)));

    const auto constrained_by_in_threshold_state = g_constrained_by_in_threshold;
    g_constrained_by_in_threshold = 0;
    ScopeGuard reset_constrained_by_in_threshold = [&constrained_by_in_threshold_state] {
      g_constrained_by_in_threshold = constrained_by_in_threshold_state;
    };
    c("SELECT fixed_str AS key0, str as key1, count(*) as val FROM test WHERE "
      "((fixed_str IN (SELECT fixed_str FROM test GROUP BY fixed_str))) GROUP BY key0, "
      "key1 ORDER BY val desc;",
      dt);
  }
}

TEST_F(Select, Strings) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c("SELECT str, COUNT(*) FROM test GROUP BY str HAVING COUNT(*) > 5 ORDER BY str;",
      dt);
    c("SELECT str, COUNT(*) FROM test WHERE str = 'bar' GROUP BY str HAVING COUNT(*) > 4 "
      "ORDER BY str;",
      dt);
    c("SELECT str, COUNT(*) FROM test WHERE str = 'bar' GROUP BY str HAVING COUNT(*) > 5 "
      "ORDER BY str;",
      dt);
    c("SELECT str, COUNT(*) FROM test where str IS NOT NULL GROUP BY str ORDER BY str;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE 'ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%eal_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%ba%';", dt);
    c("SELECT * FROM test WHERE str LIKE '%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE str LIKE 'f%%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE str LIKE 'f%\%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE ss LIKE 'f%\%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE str LIKE '@f%%' ESCAPE '@' ORDER BY x ASC, y ASC;", dt);
    c(R"(SELECT COUNT(*) FROM test WHERE real_str LIKE '%foo' OR real_str LIKE '%"bar"';)",
      dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE 'ba_' or str LIKE 'fo_';", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str > 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE str > 'fo';", dt);
    c("SELECT COUNT(*) FROM test WHERE str >= 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' < str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'fo' < str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <= str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str = 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' = str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str <> 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <> str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str = 'foo' OR str = 'bar';", dt);
    // The following tests throw Cast from dictionary-encoded string to none-encoded not
    // supported for distributed queries in distributed mode
    c("SELECT COUNT(*) FROM test WHERE str = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str <> str;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss <> str;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss = str;", dt);
    c("SELECT COUNT(*) FROM test WHERE LENGTH(str) = 3;", dt);
    c("SELECT fixed_str, COUNT(*) FROM test GROUP BY fixed_str HAVING COUNT(*) > 5 ORDER "
      "BY fixed_str;",
      dt);
    c("SELECT fixed_str, COUNT(*) FROM test WHERE fixed_str = 'bar' GROUP BY fixed_str "
      "HAVING COUNT(*) > 4 ORDER BY "
      "fixed_str;",
      dt);
    c("SELECT COUNT(*) FROM emp WHERE ename LIKE 'D%%' OR ename = 'Julia';", dt);
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(str) = 3;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str ILIKE 'f%%';", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE (str ILIKE 'f%%');", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE ( str ILIKE 'f%%' );", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str ILIKE 'McDonald''s';", dt)));
    ASSERT_EQ("foo",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT str FROM test WHERE REGEXP_LIKE(str, '^f.?.+');", dt))));
    ASSERT_EQ("bar",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT str FROM test WHERE REGEXP_LIKE(str, '^[a-z]+r$');", dt))));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP '.*';", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP '...';", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP '.+.+.+';", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP '.?.?.?';", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or str REGEXP 'fo.';",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE "
                                        "REGEXP_LIKE(str, 'ba.') or str REGEXP 'fo.?';",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP "
                                        "'ba.' or REGEXP_LIKE(str, 'fo.+');",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.+';", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(str, '.?ba.*');", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE "
                                  "REGEXP_LIKE(str,'ba.') or REGEXP_LIKE(str, 'fo.+');",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP "
                                        "'ba.' or REGEXP_LIKE(str, 'fo.+');",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE "
                                        "REGEXP_LIKE(str, 'ba.') or str REGEXP 'fo.?';",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or str REGEXP 'fo.';",
                  dt)));
    EXPECT_ANY_THROW(run_simple_agg("SELECT LENGTH(NULL) FROM test;", dt));
  }
}

TEST_F(Select, SharedDictionary) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c("SELECT shared_dict, COUNT(*) FROM test GROUP BY shared_dict HAVING COUNT(*) > 5 "
      "ORDER BY shared_dict;",
      dt);
    c("SELECT shared_dict, COUNT(*) FROM test WHERE shared_dict = 'bar' GROUP BY "
      "shared_dict HAVING COUNT(*) > 4 ORDER "
      "BY shared_dict;",
      dt);
    c("SELECT shared_dict, COUNT(*) FROM test WHERE shared_dict = 'bar' GROUP BY "
      "shared_dict HAVING COUNT(*) > 5 ORDER "
      "BY shared_dict;",
      dt);
    c("SELECT shared_dict, COUNT(*) FROM test where shared_dict IS NOT NULL GROUP BY "
      "shared_dict ORDER BY shared_dict;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict LIKE '%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict LIKE 'ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict LIKE '%eal_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict LIKE '%ba%';", dt);
    c("SELECT * FROM test WHERE shared_dict LIKE '%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE shared_dict LIKE 'f%%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE shared_dict LIKE 'f%\%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE ss LIKE 'f%\%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE shared_dict LIKE '@f%%' ESCAPE '@' ORDER BY x ASC, y "
      "ASC;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict LIKE 'ba_' or shared_dict LIKE 'fo_';",
      dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict = 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' = shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict <> 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <> shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict = 'foo' OR shared_dict = 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict <> shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'fo';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict >= 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' < shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'fo' < shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <= shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE LENGTH(shared_dict) = 3;", dt);

    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(shared_dict) = 3;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict ILIKE 'f%%';", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE (shared_dict ILIKE 'f%%');", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE ( shared_dict ILIKE 'f%%' );", dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE shared_dict ILIKE 'McDonald''s';", dt)));

    ASSERT_EQ(
        "foo",
        boost::get<std::string>(v<NullableString>(run_simple_agg(
            "SELECT shared_dict FROM test WHERE REGEXP_LIKE(shared_dict, '^f.?.+');",
            dt))));
    ASSERT_EQ(
        "baz",
        boost::get<std::string>(v<NullableString>(run_simple_agg(
            "SELECT shared_dict FROM test WHERE REGEXP_LIKE(shared_dict, '^[a-z]+z$');",
            dt))));

    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '.*';", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '...';", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '.+.+.+';", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '.?.?.?';", dt)));

    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict "
                                        "REGEXP 'ba.' or shared_dict REGEXP 'fo.';",
                                        dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict, 'ba.') or "
                  "shared_dict REGEXP 'fo.?';",
                  dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict REGEXP "
                                  "'ba.' or REGEXP_LIKE(shared_dict, 'fo.+');",
                                  dt)));
    ASSERT_EQ(5,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP 'ba.+';", dt)));
    ASSERT_EQ(
        5,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict, '.?ba.*');", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict,'ba.') or "
                  "REGEXP_LIKE(shared_dict, 'fo.+');",
                  dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict REGEXP "
                                  "'ba.' or REGEXP_LIKE(shared_dict, 'fo.+');",
                                  dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict, 'ba.') or "
                  "shared_dict REGEXP 'fo.?';",
                  dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict "
                                        "REGEXP 'ba.' or shared_dict REGEXP 'fo.';",
                                        dt)));
  }
}

TEST_F(Select, StringCompare) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE str = 'ba';", dt);
    c("SELECT COUNT(*) FROM test WHERE str <> 'ba';", dt);

    c("SELECT COUNT(*) FROM test WHERE shared_dict < 'ba';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict < 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict < 'baf';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict < 'baz';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict < 'bbz';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict < 'foo';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict < 'foon';", dt);

    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'ba';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'baf';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'baz';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'bbz';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'foo';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'foon';", dt);

    c("SELECT COUNT(*) FROM test WHERE real_str <= 'ba';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <= 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <= 'baf';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <= 'baz';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <= 'bbz';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <= 'foo';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <= 'foon';", dt);

    c("SELECT COUNT(*) FROM test WHERE real_str >= 'ba';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str >= 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str >= 'baf';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str >= 'baz';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str >= 'bbz';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str >= 'foo';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str >= 'foon';", dt);

    c("SELECT COUNT(*) FROM test WHERE real_str <= 'äâ';", dt);

    c("SELECT COUNT(*) FROM test WHERE 'ba' < shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' < shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'ba' > shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' > shared_dict;", dt);

    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test, test_inner WHERE "
                                  "test.shared_dict < test_inner.str",
                                  dt),
                 std::runtime_error);
  }
}

TEST_F(Select, StringsNoneEncoding) {
  createTestLotsColsTable();
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE '%eal_bar';", dt);
    c("SELECT * FROM test_lots_cols WHERE real_str LIKE '%' ORDER BY x0 ASC;", dt);
    c("SELECT * FROM test WHERE real_str LIKE '%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_f%%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_f%\%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_@f%%' ESCAPE '@' ORDER BY x ASC, y "
      "ASC;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_ba_' or real_str LIKE "
      "'real_fo_';",
      dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str > 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str > 'real_fo';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str >= 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' < real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_fo' < real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' <= real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str = 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <> 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' <> real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str = 'real_foo' OR real_str = 'real_bar';",
      dt);
    c("SELECT COUNT(*) FROM test WHERE real_str = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <> real_str;", dt);
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE real_str ILIKE 'rEaL_f%%';", dt)));
    c("SELECT COUNT(*) FROM test WHERE LENGTH(real_str) = 8;", dt);
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(real_str) = 8;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(real_str,'real_.*.*.*');", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_ba.*';", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE real_str REGEXP '.*';", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_f.*.*';", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_f.+\%';", dt)));
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test WHERE real_str LIKE str;", dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(real_str, str);", dt),
                 std::runtime_error);
  }
  dropTable("test_lots_cols");
}

TEST_F(Select, TimeSyntaxCheck) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    ASSERT_EQ(1325376000LL,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(year, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1325376000LL,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC('year', CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1325376000LL,
              v<int64_t>(run_simple_agg("SELECT PG_DATE_TRUNC(year, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1325376000LL,
              v<int64_t>(run_simple_agg("SELECT PG_DATE_TRUNC('year', CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(2007,
              v<int64_t>(run_simple_agg("SELECT PG_EXTRACT('year', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(2007,
              v<int64_t>(run_simple_agg("SELECT PG_EXTRACT(YEAR, CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(2007,
              v<int64_t>(run_simple_agg("SELECT extract('year' from CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(2007,
              v<int64_t>(run_simple_agg("SELECT extract(year from CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(2007,
              v<int64_t>(run_simple_agg("SELECT DATEPART('year', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(2007,
              v<int64_t>(run_simple_agg("SELECT DATEPART(YEAR, CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg("SELECT DATEDIFF('year', CAST('2006-01-07 00:00:00' as "
                                  "TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                                  "TIMESTAMP)) FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg("SELECT DATEDIFF(YEAR, CAST('2006-01-07 00:00:00' as "
                                  "TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                                  "TIMESTAMP)) FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('day', 1, CAST('2017-05-31' AS DATE)) "
                                  "= TIMESTAMP '2017-06-01 0:00:00' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD(DAY, 1, CAST('2017-05-31' AS DATE)) "
                                  "= TIMESTAMP '2017-06-01 0:00:00' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPADD('year', 1, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
            "'2009-02-28 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPADD(YEAR, 1, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
            "'2009-02-28 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        128885,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPDIFF('minute', TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
            "'2003-05-01 12:05:55') FROM test LIMIT 1;",
            dt)));
    ASSERT_EQ(
        128885,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPDIFF(minute, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
            "'2003-05-01 12:05:55') FROM test LIMIT 1;",
            dt)));
    ASSERT_EQ(128885,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF('sql_tsi_minute', "
                                        "TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
                                        "'2003-05-01 12:05:55') FROM test LIMIT 1;",
                                        dt)));
    ASSERT_EQ(128885,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(sql_tsi_minute, TIMESTAMP "
                                        "'2003-02-01 0:00:00', TIMESTAMP "
                                        "'2003-05-01 12:05:55') FROM test LIMIT 1;",
                                        dt)));
  }
}

TEST_F(Select, Time) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // check DATE Formats
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows + g_num_rows / 2),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) > o;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows + g_num_rows / 2),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE CAST('10/09/1999' AS DATE) > o;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('10/09/99' AS DATE) > o;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('10-Sep-99' AS DATE) > o;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('9/10/99' AS DATE) > o;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows + g_num_rows / 2),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE CAST('31/Oct/2013' AS DATE) > o;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('10/31/13' AS DATE) > o;", dt)));
    // check TIME FORMATS
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) > n;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('151315' AS TIME) > n;", dt)));

    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows + g_num_rows / 2),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) > o;", dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) <= o;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) > n;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) <= n;", dt)));
    cta("SELECT DATETIME('NOW') FROM test limit 1;", dt);
    EXPECT_ANY_THROW(run_simple_agg("SELECT DATETIME(NULL) FROM test LIMIT 1;", dt));
    // these next tests work because all dates are before now 2015-12-8 17:00:00
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m < NOW();", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE o IS NULL OR o < CURRENT_DATE;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE o IS NULL OR o < CURRENT_DATE();", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m < CURRENT_TIMESTAMP;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m < CURRENT_TIMESTAMP();", dt)));
    ASSERT_TRUE(v<int64_t>(
        run_simple_agg("SELECT CURRENT_DATE = CAST(CURRENT_TIMESTAMP AS DATE);", dt)));
    ASSERT_TRUE(
        v<int64_t>(run_simple_agg("SELECT DATEADD('day', -1, CURRENT_TIMESTAMP) < "
                                  "CURRENT_DATE AND CURRENT_DATE <= CURRENT_TIMESTAMP;",
                                  dt)));
    ASSERT_TRUE(v<int64_t>(run_simple_agg(
        "SELECT CAST(CURRENT_DATE AS TIMESTAMP) <= CURRENT_TIMESTAMP;", dt)));
    ASSERT_TRUE(v<int64_t>(run_simple_agg(
        "SELECT EXTRACT(YEAR FROM CURRENT_DATE) = EXTRACT(YEAR FROM CURRENT_TIMESTAMP)"
        " AND EXTRACT(MONTH FROM CURRENT_DATE) = EXTRACT(MONTH FROM CURRENT_TIMESTAMP)"
        " AND EXTRACT(DAY FROM CURRENT_DATE) = EXTRACT(DAY FROM CURRENT_TIMESTAMP)"
        " AND EXTRACT(HOUR FROM CURRENT_DATE) = 0"
        " AND EXTRACT(MINUTE FROM CURRENT_DATE) = 0"
        " AND EXTRACT(SECOND FROM CURRENT_DATE) = 0;",
        dt)));
    ASSERT_TRUE(v<int64_t>(run_simple_agg(
        "SELECT EXTRACT(HOUR FROM CURRENT_TIME()) = EXTRACT(HOUR FROM CURRENT_TIMESTAMP)"
        " AND EXTRACT(MINUTE FROM CURRENT_TIME) = EXTRACT(MINUTE FROM CURRENT_TIMESTAMP)"
        " AND EXTRACT(SECOND FROM CURRENT_TIME) = EXTRACT(SECOND FROM CURRENT_TIMESTAMP)"
        ";",
        dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m > timestamp(0) '2014-12-13T000000';",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST(o AS "
                                        "TIMESTAMP) > timestamp(0) '1999-09-08T160000';",
                                        dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST(o AS "
                                        "TIMESTAMP) > timestamp(0) '1999-09-10T160000';",
                                        dt)));
    ASSERT_EQ(14185957950LL,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(EPOCH FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(14185152000LL,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(DATEEPOCH FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(20140,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(YEAR FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(120,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(MONTH FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(140,
              v<int64_t>(
                  run_simple_agg("SELECT MAX(EXTRACT(DAY FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(
        22,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM m)) FROM test;", dt)));
    ASSERT_EQ(
        23,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM m)) FROM test;", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM m)) FROM test;", dt)));
    ASSERT_EQ(
        6, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOW FROM m)) FROM test;", dt)));
    ASSERT_EQ(
        348,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOY FROM m)) FROM test;", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM n)) FROM test;", dt)));
    ASSERT_EQ(
        13,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM n)) FROM test;", dt)));
    ASSERT_EQ(
        14,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM n)) FROM test;", dt)));
    ASSERT_EQ(
        1999,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(YEAR FROM o)) FROM test;", dt)));
    ASSERT_EQ(
        9,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MONTH FROM o)) FROM test;", dt)));
    ASSERT_EQ(
        9, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DAY FROM o)) FROM test;", dt)));
    ASSERT_EQ(4,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(DOW FROM o) FROM test WHERE o IS NOT NULL;", dt)));
    ASSERT_EQ(252,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(DOY FROM o) FROM test WHERE o IS NOT NULL;", dt)));
    ASSERT_EQ(
        936835200LL,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(EPOCH FROM o)) FROM test;", dt)));
    ASSERT_EQ(936835200LL,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(DATEEPOCH FROM o)) FROM test;", dt)));
    ASSERT_EQ(52LL,
              v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK FROM CAST('2012-01-01 "
                                        "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(
        1LL,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK_SUNDAY FROM CAST('2012-01-01 "
                                  "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                  dt)));
    ASSERT_EQ(1LL,
              v<int64_t>(
                  run_simple_agg("SELECT MAX(EXTRACT(WEEK_SATURDAY FROM CAST('2012-01-01 "
                                 "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                 dt)));
    ASSERT_EQ(10LL,
              v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK FROM CAST('2008-03-03 "
                                        "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(
        10LL,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK_SUNDAY FROM CAST('2008-03-03 "
                                  "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                  dt)));
    ASSERT_EQ(10LL,
              v<int64_t>(
                  run_simple_agg("SELECT MAX(EXTRACT(WEEK_SATURDAY FROM CAST('2008-03-03 "
                                 "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                 dt)));
    // Monday
    ASSERT_EQ(1LL,
              v<int64_t>(run_simple_agg("SELECT EXTRACT(DOW FROM CAST('2008-03-03 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    // Monday
    ASSERT_EQ(1LL,
              v<int64_t>(run_simple_agg("SELECT EXTRACT(ISODOW FROM CAST('2008-03-03 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    // Sunday
    ASSERT_EQ(0LL,
              v<int64_t>(run_simple_agg("SELECT EXTRACT(DOW FROM CAST('2008-03-02 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    // Sunday
    ASSERT_EQ(7LL,
              v<int64_t>(run_simple_agg("SELECT EXTRACT(ISODOW FROM CAST('2008-03-02 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(15000000000LL,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(nanosecond from m) FROM test limit 1;", dt)));
    ASSERT_EQ(15000000LL,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(microsecond from m) FROM test limit 1;", dt)));
    ASSERT_EQ(15000LL,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(millisecond from m) FROM test limit 1;", dt)));
    ASSERT_EQ(56000000000LL,
              v<int64_t>(run_simple_agg("SELECT EXTRACT(nanosecond from TIMESTAMP(0) "
                                        "'1999-03-14 23:34:56') FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(56000000LL,
              v<int64_t>(run_simple_agg("SELECT EXTRACT(microsecond from TIMESTAMP(0) "
                                        "'1999-03-14 23:34:56') FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(56000LL,
              v<int64_t>(run_simple_agg("SELECT EXTRACT(millisecond from TIMESTAMP(0) "
                                        "'1999-03-14 23:34:56') FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(2005,
              v<int64_t>(run_simple_agg("select EXTRACT(year from TIMESTAMP '2005-12-31 "
                                        "23:59:59') from test limit 1;",
                                        dt)));
    ASSERT_EQ(1997,
              v<int64_t>(run_simple_agg("select EXTRACT(year from TIMESTAMP '1997-01-01 "
                                        "23:59:59') from test limit 1;",
                                        dt)));
    ASSERT_EQ(2006,
              v<int64_t>(run_simple_agg("select EXTRACT(year from TIMESTAMP '2006-01-01 "
                                        "00:0:00') from test limit 1;",
                                        dt)));
    ASSERT_EQ(2014,
              v<int64_t>(run_simple_agg("select EXTRACT(year from TIMESTAMP '2014-01-01 "
                                        "00:00:00') from test limit 1;",
                                        dt)));

    // test DATE format processing
    ASSERT_EQ(1434844800LL,
              v<int64_t>(run_simple_agg(
                  "select CAST('2015-06-21' AS DATE) FROM test limit 1;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows + g_num_rows / 2),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE o < CAST('06/21/2015' AS DATE);", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE o < CAST('21-Jun-15' AS DATE);", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows + g_num_rows / 2),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE o < CAST('21/Jun/2015' AS DATE);", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows + g_num_rows / 2),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE o < CAST('1434844800' AS DATE);", dt)));

    // test different input formats
    // added new format for customer
    ASSERT_EQ(
        1434896116LL,
        v<int64_t>(run_simple_agg(
            "select CAST('2015-06-21 14:15:16' AS timestamp) FROM test limit 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('2015-06-21:141516' AS TIMESTAMP);",
                                        dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= CAST('21-JUN-15 "
                                  "2.15.16.12345 PM' AS TIMESTAMP);",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= CAST('21-JUN-15 "
                                  "2.15.16.12345 AM' AS TIMESTAMP);",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('21-JUN-15 2:15:16 AM' AS TIMESTAMP);",
                                        dt)));

    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('06/21/2015 14:15:16' AS TIMESTAMP);",
                                        dt)));

    // Support ISO date offset format
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                  "CAST('21/Aug/2015:12:13:14 -0600' AS TIMESTAMP);",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('2015-08-21T12:13:14 -0600' AS TIMESTAMP);",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('21-Aug-15 12:13:14 -0600' AS TIMESTAMP);",
                                        dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                  "CAST('21/Aug/2015:13:13:14 -0500' AS TIMESTAMP);",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('2015-08-21T18:13:14' AS TIMESTAMP);",
                                        dt)));
    // add test for quarterday behaviour
    ASSERT_EQ(1LL,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T04:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(1LL,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T00:00:00' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(2LL,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T08:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(3LL,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T14:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(4LL,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T23:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(1440115200LL,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T04:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(1440136800LL,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T08:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(1440158400LL,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T13:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(1440180000LL,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T23:59:59' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(2007,
              v<int64_t>(run_simple_agg("SELECT DATEPART('year', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(2007,
              v<int64_t>(run_simple_agg("SELECT DATEPART('yyyy', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        2007,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('yy', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(4,
              v<int64_t>(run_simple_agg("SELECT DATEPART('quarter', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        4,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('qq', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        4,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('q', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(10,
              v<int64_t>(run_simple_agg("SELECT DATEPART('month', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        10,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('mm', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        10,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('m', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(303,
              v<int64_t>(run_simple_agg("SELECT DATEPART('dayofyear', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        303,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('dy', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        303,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('y', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('day', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('dd', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('d', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(12,
              v<int64_t>(run_simple_agg("SELECT DATEPART('hour', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('hh', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg("SELECT DATEPART('minute', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('mi', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('n', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(32,
              v<int64_t>(run_simple_agg("SELECT DATEPART('second', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        32,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('ss', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        32,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('s', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        32,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('s', TIMESTAMP '2007-10-30 12:15:32') FROM test;", dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg("SELECT DATEDIFF('year', CAST('2006-01-07 00:00:00' as "
                                  "TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                                  "TIMESTAMP)) FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        36,
        v<int64_t>(run_simple_agg("SELECT DATEDIFF('month', CAST('2006-01-07 00:00:00' "
                                  "as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                                  "TIMESTAMP)) FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1096,
        v<int64_t>(run_simple_agg("SELECT DATEDIFF('day', CAST('2006-01-07 00:00:00' as "
                                  "TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                                  "TIMESTAMP)) FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg("SELECT DATEDIFF('quarter', CAST('2006-01-07 00:00:00' "
                                  "as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                                  "TIMESTAMP)) FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('day', DATE '2009-2-28', DATE "
                                        "'2009-03-01') FROM test LIMIT 1;",
                                        dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('day', DATE '2008-2-28', DATE "
                                        "'2008-03-01') FROM test LIMIT 1;",
                                        dt)));
    ASSERT_EQ(-425,
              v<int64_t>(run_simple_agg("select DATEDIFF('day', DATE '1971-03-02', DATE "
                                        "'1970-01-01') from test limit 1;",
                                        dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('day', o, o + INTERVAL '1' DAY) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg("SELECT count(*) from test where DATEDIFF('day', "
                                        "CAST (m AS DATE), o) < -5570;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('second', m, TIMESTAMP(0) "
                                        "'2014-12-13 22:23:16') FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('millisecond', m, TIMESTAMP(0) "
                                        "'2014-12-13 22:23:16') FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(44000000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('microsecond', m, TIMESTAMP(0) "
                                        "'2014-12-13 22:23:59') FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(34000000000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('nanosecond', m, TIMESTAMP(0) "
                                        "'2014-12-13 22:23:49') FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-1000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('millisecond', TIMESTAMP(0) "
                                        "'2014-12-13 22:23:16', m) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-44000000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('microsecond', TIMESTAMP(0) "
                                        "'2014-12-13 22:23:59', m) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-34000000000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('nanosecond', TIMESTAMP(0) "
                                        "'2014-12-13 22:23:49', m) FROM test limit 1;",
                                        dt)));
    // DATEADD tests
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('day', 1, CAST('2017-05-31' AS DATE)) "
                                  "= TIMESTAMP '2017-06-01 0:00:00' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('day', 2, DATE '2017-05-31') = "
                                  "TIMESTAMP '2017-06-02 0:00:00' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('day', -1, CAST('2017-05-31' AS DATE)) "
                                  "= TIMESTAMP '2017-05-30 0:00:00' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('day', -2, DATE '2017-05-31') = "
                                  "TIMESTAMP '2017-05-29 0:00:00' from test limit 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('hour', 1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                  "'2017-05-31 2:11:11' from test limit 1;",
                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('hour', 10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 11:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('hour', -1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 0:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('hour', -10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-30 15:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('minute', 1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:12:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('minute', 10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:21:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('minute', -1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:10:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('minute', -10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:01:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('second', 1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:11:12' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('second', 10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:11:21' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('second', -1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:11:10' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('second', -10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:11:01' from test limit 1;",
            dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('month', 1, DATE '2017-01-10') = TIMESTAMP "
                  "'2017-02-10 0:00:00' from test limit 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('month', 10, DATE '2017-01-10') = TIMESTAMP "
                  "'2017-11-10 0:00:00' from test limit 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('month', 1, DATE '2009-01-30') = TIMESTAMP "
                  "'2009-02-28 0:00:00' from test limit 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('month', 1, DATE '2008-01-30') = TIMESTAMP "
                  "'2008-02-29 0:00:00' from test limit 1;",
                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('month', 1, TIMESTAMP '2009-01-30 1:11:11') = TIMESTAMP "
            "'2009-02-28 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('month', -1, TIMESTAMP '2009-03-30 1:11:11') = TIMESTAMP "
            "'2009-02-28 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('month', -4, TIMESTAMP '2009-03-30 1:11:11') = TIMESTAMP "
            "'2008-11-30 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('month', 5, TIMESTAMP '2009-01-31 1:11:11') = TIMESTAMP "
            "'2009-6-30 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('year', 1, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
                  "'2009-02-28 1:11:11' from test limit 1;",
                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPADD(YEAR, 1, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
            "'2009-02-28 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPADD(YEAR, -8, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
            "'2000-02-29 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPADD(YEAR, -8, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
            "'2000-02-29 1:11:11' from test limit 1;",
            dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT m = TIMESTAMP '2014-12-13 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', 1, m) = TIMESTAMP "
                                        "'2014-12-14 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', -1, m) = TIMESTAMP "
                                        "'2014-12-12 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', 1, m) = TIMESTAMP "
                                        "'2014-12-14 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', -1, m) = TIMESTAMP "
                                        "'2014-12-12 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(
                  run_simple_agg("SELECT o = DATE '1999-09-09' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', 1, o) = TIMESTAMP "
                                        "'1999-09-10 0:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', -3, o) = TIMESTAMP "
                                        "'1999-09-06 0:00:00' from test limit 1;",
                                        dt)));
    /* DATE ADD subseconds to default timestamp(0) */
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('millisecond', 1000, m) = TIMESTAMP "
                                  "'2014-12-13 22:23:16' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('microsecond', 1000000, m) = TIMESTAMP "
                                  "'2014-12-13 22:23:16' from test limit 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('nanosecond', 1000000000, m) = TIMESTAMP "
                  "'2014-12-13 22:23:16' from test limit 1;",
                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('millisecond', 5123, m) = TIMESTAMP "
                                  "'2014-12-13 22:23:20' from test limit 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('microsecond', 86400000000, m) = TIMESTAMP "
                  "'2014-12-14 22:23:15' from test limit 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('nanosecond', 86400000000123, m) = TIMESTAMP "
                  "'2014-12-14 22:23:15' from test limit 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('weekday', -3, o) = TIMESTAMP "
                                        "'1999-09-06 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('decade', 3, o) = TIMESTAMP "
                                        "'2029-09-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('week', 1, o) = TIMESTAMP "
                                        "'1999-09-16 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, 1, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-03 1:23:45' "
                                  "FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, -1, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-01 1:23:45' "
                                  "FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, 15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-17 1:23:45' "
                                  "FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, -15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-02-15 1:23:45' "
                                  "FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, 1, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-02 2:23:45' "
                                  "FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, -1, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-02 0:23:45' "
                                  "FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, 15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-02 16:23:45' "
                                  "FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, -15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-01 10:23:45' "
                                  "FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(MINUTE, 15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-02 1:38:45' "
                                  "FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(
                  run_simple_agg("SELECT TIMESTAMPADD(MINUTE, -15, TIMESTAMP '2009-03-02 "
                                 "1:23:45') = TIMESTAMP '2009-03-02 1:08:45' "
                                 "FROM test LIMIT 1;",
                                 dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(SECOND, 15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-02 1:24:00' "
                                  "FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(
                  run_simple_agg("SELECT TIMESTAMPADD(SECOND, -15, TIMESTAMP '2009-03-02 "
                                 "1:23:45') = TIMESTAMP '2009-03-02 1:23:30' "
                                 "FROM test LIMIT 1;",
                                 dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, 1, m) = TIMESTAMP '2014-12-14 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, -1, m) = TIMESTAMP '2014-12-12 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, 15, m) = TIMESTAMP '2014-12-28 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, -15, m) = TIMESTAMP '2014-11-28 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, 1, m) = TIMESTAMP '2014-12-13 23:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, -1, m) = TIMESTAMP '2014-12-13 21:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, 15, m) = TIMESTAMP '2014-12-14 13:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, -15, m) = TIMESTAMP '2014-12-13 7:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MINUTE, 15, m) = TIMESTAMP '2014-12-13 22:38:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MINUTE, -15, m) = TIMESTAMP '2014-12-13 22:08:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SECOND, 15, m) = TIMESTAMP '2014-12-13 22:23:30' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SECOND, -15, m) = TIMESTAMP '2014-12-13 22:23:00' "
                  "FROM test LIMIT 1;",
                  dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MONTH, 1, m) = TIMESTAMP '2015-01-13 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MONTH, -1, m) = TIMESTAMP '2014-11-13 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MONTH, 5, m) = TIMESTAMP '2015-05-13 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, -5, m) = TIMESTAMP '2014-12-08 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(YEAR, 1, m) = TIMESTAMP '2015-12-13 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(YEAR, -1, m) = TIMESTAMP '2013-12-13 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(YEAR, 5, m) = TIMESTAMP '2019-12-13 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(YEAR, -5, m) = TIMESTAMP '2009-12-13 22:23:15' "
                  "FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg("select count(*) from test where TIMESTAMPADD(YEAR, "
                                  "15, CAST(o AS TIMESTAMP)) > m;",
                                  dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("select count(*) from test where TIMESTAMPADD(YEAR, "
                                  "16, CAST(o AS TIMESTAMP)) > m;",
                                  dt)));

    ASSERT_EQ(
        128885,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPDIFF(minute, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
            "'2003-05-01 12:05:55') FROM test LIMIT 1;",
            dt)));
    ASSERT_EQ(2148,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPDIFF(hour, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
                  "'2003-05-01 12:05:55') FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(89,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPDIFF(day, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
                  "'2003-05-01 12:05:55') FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(3,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPDIFF(month, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
                  "'2003-05-01 12:05:55') FROM test LIMIT 1;",
                  dt)));
    ASSERT_EQ(
        -3,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPDIFF(month, TIMESTAMP '2003-05-01 12:05:55', TIMESTAMP "
            "'2003-02-01 0:00:00') FROM test LIMIT 1;",
            dt)));
    ASSERT_EQ(
        5,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPDIFF(month, m, m + INTERVAL '5' MONTH) FROM test LIMIT 1;",
            dt)));
    ASSERT_EQ(
        -5,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPDIFF(month, m, m - INTERVAL '5' MONTH) FROM test LIMIT 1;",
            dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("select count(*) from test where TIMESTAMPDIFF(YEAR, "
                                  "m, CAST(o AS TIMESTAMP)) < 0;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(year, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM test LIMIT 1;",
                                        dt)));
    ASSERT_EQ(14,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(month, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM test LIMIT 1;",
                                        dt)));
    ASSERT_EQ(426,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(day, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM test LIMIT 1;",
                                        dt)));
    ASSERT_EQ(60,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(week, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM test LIMIT 1;",
                                        dt)));
    ASSERT_EQ(
        60,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(week_sunday, DATE '2018-01-02', "
                                  "DATE '2019-03-04') FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        60,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(week_saturday, DATE "
                                  "'2018-01-02', DATE '2019-03-04') FROM test LIMIT 1;",
                                  dt)));
    ASSERT_EQ(613440,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(minute, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM test LIMIT 1;",
                                        dt)));
    ASSERT_EQ(10224,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(hour, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM test LIMIT 1;",
                                        dt)));
    ASSERT_EQ(36806400,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(second, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM test LIMIT 1;",
                                        dt)));

    ASSERT_EQ(
        1418428800LL,
        v<int64_t>(run_simple_agg("SELECT CAST(m AS date) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(1336435200LL,
              v<int64_t>(run_simple_agg("SELECT CAST(CAST('2012-05-08 20:15:12' AS "
                                        "TIMESTAMP) AS DATE) FROM test LIMIT 1;",
                                        dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test GROUP BY CAST(m AS date);", dt)));
    const auto rows = run_multiple_agg(
        "SELECT DATE_TRUNC(month, CAST(o AS TIMESTAMP(0))) AS key0, str AS key1, "
        "COUNT(*) AS val FROM test GROUP BY "
        "key0, key1 ORDER BY val DESC, key1;",
        dt);
    check_date_trunc_groups(*rows);
    const auto one_row = run_multiple_agg(
        "SELECT DATE_TRUNC(year, CASE WHEN str = 'foo' THEN m END) d FROM test GROUP BY "
        "d "
        "HAVING d IS NOT NULL;",
        dt);
    check_one_date_trunc_group(*one_row, 1388534400);
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test where "
                                        "DATE '2017-05-30' = DATE '2017-05-31' OR "
                                        "DATE '2017-05-31' = DATE '2017-05-30';",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test where "
                                        "EXTRACT(DOW from TIMESTAMPADD(HOUR, -5, "
                                        "TIMESTAMP '2017-05-31 1:11:11')) = 1 OR "
                                        "EXTRACT(DOW from TIMESTAMPADD(HOUR, -5, "
                                        "TIMESTAMP '2017-05-31 1:11:11')) = 2;",
                                        dt)));

    std::vector<std::tuple<std::string, int64_t, int64_t>> date_trunc_queries{
        /*TIMESTAMP(0) */
        std::make_tuple("year, m", 1388534400LL, 20),
        std::make_tuple("month, m", 1417392000LL, 20),
        std::make_tuple("day, m", 1418428800LL, 15),
        std::make_tuple("hour, m", 1418508000LL, 15),
        std::make_tuple("minute, m", 1418509380LL, 15),
        std::make_tuple("second, m", 1418509395LL, 15),
        std::make_tuple("millennium, m", 978307200LL, 20),
        std::make_tuple("century, m", 978307200LL, 20),
        std::make_tuple("decade, m", 1262304000LL, 20),
        std::make_tuple("week, m", 1417996800LL, 20),
        std::make_tuple("week_sunday, m", 1417910400LL, 15),
        std::make_tuple("week_saturday, m", 1418428800LL, 20),
        std::make_tuple("nanosecond, m", 1418509395LL, 15),
        std::make_tuple("microsecond, m", 1418509395LL, 15),
        std::make_tuple("millisecond, m", 1418509395LL, 15),
        /* TIMESTAMP(3) */
        std::make_tuple("year, m_3", 1388534400000LL, 20),
        std::make_tuple("month, m_3", 1417392000000LL, 20),
        std::make_tuple("day, m_3", 1418428800000LL, 15),
        std::make_tuple("hour, m_3", 1418508000000LL, 15),
        std::make_tuple("minute, m_3", 1418509380000LL, 15),
        std::make_tuple("second, m_3", 1418509395000LL, 15),
        std::make_tuple("millennium, m_3", 978307200000LL, 20),
        std::make_tuple("century, m_3", 978307200000LL, 20),
        std::make_tuple("decade, m_3", 1262304000000LL, 20),
        std::make_tuple("week, m_3", 1417996800000LL, 20),
        std::make_tuple("week_sunday, m_3", 1417910400000LL, 15),
        std::make_tuple("week_saturday, m_3", 1418428800000LL, 20),
        std::make_tuple("nanosecond, m_3", 1418509395323LL, 15),
        std::make_tuple("microsecond, m_3", 1418509395323LL, 15),
        std::make_tuple("millisecond, m_3", 1418509395323LL, 15),
        /* TIMESTAMP(6) */
        std::make_tuple("year, m_6", 915148800000000LL, 10),
        std::make_tuple("month, m_6", 930787200000000LL, 10),
        std::make_tuple("day, m_6", 931651200000000LL, 10),
        std::make_tuple("hour, m_6", 931701600000000LL, 10),
        /* std::make_tuple("minute, m_6", 931701720000000LL, 10), // Exception with sort
           watchdog */
        std::make_tuple("second, m_6", 931701773000000LL, 10),
        std::make_tuple("millennium, m_6", -30578688000000000LL, 10),
        std::make_tuple("century, m_6", -2177452800000000LL, 10),
        std::make_tuple("decade, m_6", 631152000000000LL, 10),
        std::make_tuple("week, m_6", 931132800000000LL, 10),
        std::make_tuple("week_sunday, m_6", 931651200000000LL, 10),
        std::make_tuple("week_saturday, m_6", 931564800000000LL, 10),
        std::make_tuple("nanosecond, m_6", 931701773874533LL, 10),
        std::make_tuple("microsecond, m_6", 931701773874533LL, 10),
        std::make_tuple("millisecond, m_6", 931701773874000LL, 10),
        /* TIMESTAMP(9) */
        std::make_tuple("year, m_9", 1136073600000000000LL, 10),
        std::make_tuple("month, m_9", 1143849600000000000LL, 10),
        std::make_tuple("day, m_9", 1146009600000000000LL, 10),
        std::make_tuple("hour, m_9", 1146020400000000000LL, 10),
        /* std::make_tuple("minute, m_9", 1146023340000000000LL, 10), // Exception with
           sort watchdog */
        std::make_tuple("second, m_9", 1146023344000000000LL, 10),
        std::make_tuple("millennium, m_9", 978307200000000000LL, 20),
        std::make_tuple("century, m_9", 978307200000000000LL, 20),
        std::make_tuple("decade, m_9", 946684800000000000LL, 10),
        std::make_tuple("week, m_9", 1145836800000000000LL, 10),
        std::make_tuple("week_sunday, m_9", 1145750400000000000LL, 10),
        std::make_tuple("week_saturday, m_9", 1145664000000000000LL, 10),
        std::make_tuple("nanosecond, m_9", 1146023344607435125LL, 10),
        std::make_tuple("microsecond, m_9", 1146023344607435000LL, 10),
        std::make_tuple("millisecond, m_9", 1146023344607000000LL, 10)};
    for (auto& query : date_trunc_queries) {
      const auto one_row = run_multiple_agg(
          "SELECT date_trunc(" + std::get<0>(query) +
              ") as key0,COUNT(*) AS val FROM test group by key0 order by key0 "
              "limit 1;",
          dt);
      check_one_date_trunc_group_with_agg(
          *one_row, std::get<1>(query), std::get<2>(query));
    }
    // Compressed DATE - limits test
    ASSERT_EQ(4708022400LL,
              v<int64_t>(run_simple_agg(
                  "select CAST('2119-03-12' AS DATE) FROM test limit 1;", dt)));
    ASSERT_EQ(7998912000LL,
              v<int64_t>(run_simple_agg("select CAST(CAST('2223-06-24 23:13:57' AS "
                                        "TIMESTAMP) AS DATE) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('year', 411, o) = TIMESTAMP "
                                        "'2410-09-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('year', -399, o) = TIMESTAMP "
                                        "'1600-09-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', 6132, o) = TIMESTAMP "
                                        "'2510-09-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', -1100, o) = TIMESTAMP "
                                        "'1908-01-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', 312456, o) = TIMESTAMP "
                                        "'2855-03-01 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', -23674, o) = TIMESTAMP "
                                        "'1934-11-15 00:00:00' from test limit 1 ;",
                                        dt)));
    ASSERT_EQ(
        -302,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('year', DATE '2302-04-21', o) from test limit 1;", dt)));
    ASSERT_EQ(
        501,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('year', o, DATE '2501-04-21') from test limit 1;", dt)));
    ASSERT_EQ(
        -4895,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('month', DATE '2407-09-01', o) from test limit 1;", dt)));
    ASSERT_EQ(
        3817,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('month', o, DATE '2317-11-01') from test limit 1;", dt)));
    ASSERT_EQ(
        -86972,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('day', DATE '2237-10-23', o) from test limit 1;", dt)));
    ASSERT_EQ(
        86972,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('day', o, DATE '2237-10-23') from test limit 1;", dt)));
    ASSERT_EQ(
        2617,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('year', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('month', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        23,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('day', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('hour', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('minute', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('second', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        6,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('weekday', CAST ('2011-12-31' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(365,
              v<int64_t>(run_simple_agg("SELECT DATEPART('dayofyear', CAST ('2011-12-31' "
                                        "as DATE)) from test limit 1;",
                                        dt)));
    // Compressed DATE - limits test
    ASSERT_EQ(4708022400LL,
              v<int64_t>(run_simple_agg(
                  "select CAST('2119-03-12' AS DATE) FROM test limit 1;", dt)));
    ASSERT_EQ(7998912000LL,
              v<int64_t>(run_simple_agg("select CAST(CAST('2223-06-24 23:13:57' AS "
                                        "TIMESTAMP) AS DATE) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('year', 411, o) = TIMESTAMP "
                                        "'2410-09-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('year', -399, o) = TIMESTAMP "
                                        "'1600-09-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', 6132, o) = TIMESTAMP "
                                        "'2510-09-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', -1100, o) = TIMESTAMP "
                                        "'1908-01-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', 312456, o) = TIMESTAMP "
                                        "'2855-03-01 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', -23674, o) = TIMESTAMP "
                                        "'1934-11-15 00:00:00' from test limit 1 ;",
                                        dt)));
    ASSERT_EQ(
        -302,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('year', DATE '2302-04-21', o) from test limit 1;", dt)));
    ASSERT_EQ(
        501,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('year', o, DATE '2501-04-21') from test limit 1;", dt)));
    ASSERT_EQ(
        -4895,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('month', DATE '2407-09-01', o) from test limit 1;", dt)));
    ASSERT_EQ(
        3817,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('month', o, DATE '2317-11-01') from test limit 1;", dt)));
    ASSERT_EQ(
        -86972,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('day', DATE '2237-10-23', o) from test limit 1;", dt)));
    ASSERT_EQ(
        86972,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('day', o, DATE '2237-10-23') from test limit 1;", dt)));
    ASSERT_EQ(
        2617,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('year', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('month', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        23,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('day', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('hour', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('minute', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('second', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    /* Compressed Date ColumnarResults fetch tests*/
    ASSERT_EQ(1999,
              v<int64_t>(run_simple_agg("select yr from (SELECT EXTRACT(year from o) as "
                                        "yr, o from test order by x) limit 1;",
                                        dt)));
    ASSERT_EQ(936835200,
              v<int64_t>(run_simple_agg("select dy from (SELECT DATE_TRUNC(day, o) as "
                                        "dy, o from test order by x) limit 1;",
                                        dt)));
    ASSERT_EQ(936921600,
              v<int64_t>(run_simple_agg("select dy from (SELECT DATEADD('day', 1, o) as "
                                        "dy, o from test order by x) limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "select dy from (SELECT DATEDIFF('day', o, DATE '1999-09-10') as dy, o "
                  "from test order by x) limit 1;",
                  dt)));

    // range tests
    ASSERT_EQ(
        1417392000,
        v<int64_t>(run_simple_agg("SELECT date_trunc(month, m) as key0 FROM "
                                  "test WHERE (m >= TIMESTAMP(3) '1970-01-01 "
                                  "00:00:00.000') GROUP BY key0 ORDER BY key0 LIMIT 1;",
                                  dt)));
  }
}

TEST_F(Select, DateTruncate) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    ASSERT_EQ(1325376000LL,
              v<int64_t>(run_simple_agg(
                  R"(SELECT DATE_TRUNC(year, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
                  dt)));
    ASSERT_EQ(
        1335830400LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(month, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        1336435200LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(day, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)", dt)));
    ASSERT_EQ(1336507200LL,
              v<int64_t>(run_simple_agg(
                  R"(SELECT DATE_TRUNC(hour, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
                  dt)));
    ASSERT_EQ(
        1336508112LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(second, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        978307200LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(millennium, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        978307200LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(century, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        1262304000LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(decade, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        1336508112LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(millisecond, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        1336508112LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(microsecond, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(1336348800LL,
              v<int64_t>(run_simple_agg(
                  R"(SELECT DATE_TRUNC(week, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
                  dt)));
    ASSERT_EQ(
        1336348800LL - 24 * 3600,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(week_sunday, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        1336348800LL - 48 * 3600,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(week_saturday, CAST('2012-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(-2114380800LL,
              v<int64_t>(run_simple_agg(
                  R"(SELECT DATE_TRUNC(year, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
                  dt)));
    ASSERT_EQ(
        -2104012800LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(month, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        -2103408000LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(day, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)", dt)));
    ASSERT_EQ(-2103336000LL,
              v<int64_t>(run_simple_agg(
                  R"(SELECT DATE_TRUNC(hour, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
                  dt)));
    ASSERT_EQ(
        -2103335088LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(second, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        -30578688000LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(millennium, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        -2177452800LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(century, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        -2208988800LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(decade, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        -2103335088LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(millisecond, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        -2103335088LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(microsecond, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(-2103753600LL,
              v<int64_t>(run_simple_agg(
                  R"(SELECT DATE_TRUNC(week, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
                  dt)));
    ASSERT_EQ(
        -2103753600L - 24 * 3600,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(week_sunday, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        -2103753600L - 48 * 3600,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(week_saturday, CAST('1903-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        0LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(decade, CAST('1972-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    ASSERT_EQ(
        946684800LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(decade, CAST('2000-05-08 20:15:12' AS TIMESTAMP));)",
            dt)));
    // test QUARTER
    ASSERT_EQ(
        4,
        v<int64_t>(run_simple_agg(
            R"(SELECT EXTRACT(quarter FROM CAST('2008-11-27 12:12:12' AS timestamp));)",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            R"(SELECT EXTRACT(quarter FROM CAST('2008-03-21 12:12:12' AS timestamp));)",
            dt)));
    ASSERT_EQ(
        1199145600LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(quarter, CAST('2008-03-21 12:12:12' AS timestamp));)",
            dt)));
    ASSERT_EQ(
        1230768000LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(quarter, CAST('2009-03-21 12:12:12' AS timestamp));)",
            dt)));
    ASSERT_EQ(
        1254355200LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(quarter, CAST('2009-11-21 12:12:12' AS timestamp));)",
            dt)));
    ASSERT_EQ(
        946684800LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(quarter, CAST('2000-03-21 12:12:12' AS timestamp));)",
            dt)));
    ASSERT_EQ(
        -2208988800LL,
        v<int64_t>(run_simple_agg(
            R"(SELECT DATE_TRUNC(quarter, CAST('1900-03-21 12:12:12' AS timestamp));)",
            dt)));

    // Correctness tests for pre-epoch, epoch, and post-epoch dates
    auto check_epoch_result = [](const auto& result,
                                 const std::vector<int64_t>& expected) {
      EXPECT_EQ(result->rowCount(), expected.size());
      for (size_t i = 0; i < expected.size(); i++) {
        auto row = result->getNextRow(false, false);
        EXPECT_EQ(row.size(), size_t(1));
        EXPECT_EQ(expected[i], v<int64_t>(row[0]));
      }
    };

    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM dt) FROM test_date_time ORDER BY dt;)", dt),
        {-210038400, -53481600, 0, 344217600});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('year', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-220924800, -63158400, 0, 315532800});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('quarter', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-213148800, -55296000, 0, 339206400});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('month', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-210556800, -55296000, 0, 341884800});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('day', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-210038400, -53481600, 0, 344217600});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('hour', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-210038400, -53481600, 0, 344217600});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('minute', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-210038400, -53481600, 0, 344217600});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('second', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-210038400, -53481600, 0, 344217600});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('millennium', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-30578688000, -30578688000, -30578688000, -30578688000});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('century', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-2177452800, -2177452800, -2177452800, -2177452800});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('decade', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-315619200, -315619200, 0, 315532800});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('week', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-210124800, -53481600, -259200, 343872000});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('week_sunday', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-210211200, -53568000, -345600, 343785600});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('week_saturday', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-210297600, -53654400, -432000, 343699200});
    check_epoch_result(
        run_multiple_agg(
            R"(SELECT EXTRACT('epoch' FROM date_trunc('quarter', dt)) FROM test_date_time ORDER BY dt;)",
            dt),
        {-213148800, -55296000, 0, 339206400});
  }
}

TEST_F(Select, ExtractEpoch) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // Test EXTRACT(epoch) for high-precision timestamps when read from a table.
    ASSERT_TRUE(v<int64_t>(run_simple_agg(
        "SELECT MIN(DATEDIFF('second', DATE '1970-01-01', dt) = EXTRACT('epoch' FROM "
        "CAST(dt AS TIMESTAMP(0)))) FROM test_date_time;",
        dt)));
    ASSERT_TRUE(v<int64_t>(run_simple_agg(
        "SELECT MIN(DATEDIFF('second', DATE '1970-01-01', dt) = EXTRACT('epoch' FROM "
        "CAST(dt AS TIMESTAMP(3)))) FROM test_date_time;",
        dt)));
    ASSERT_TRUE(v<int64_t>(run_simple_agg(
        "SELECT MIN(DATEDIFF('second', DATE '1970-01-01', dt) = EXTRACT('epoch' FROM "
        "CAST(dt AS TIMESTAMP(6)))) FROM test_date_time;",
        dt)));
    ASSERT_TRUE(v<int64_t>(run_simple_agg(
        "SELECT MIN(DATEDIFF('second', DATE '1970-01-01', dt) = EXTRACT('epoch' FROM "
        "CAST(dt AS TIMESTAMP(9)))) FROM test_date_time;",
        dt)));

    // Test EXTRACT(epoch) for constant high-precision timestamps.
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg(
            "SELECT EXTRACT('epoch' FROM TIMESTAMP(0) '1970-01-01 00:00:03');", dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg(
            "SELECT EXTRACT('epoch' FROM TIMESTAMP(3) '1970-01-01 00:00:03.123');", dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg(
            "SELECT EXTRACT('epoch' FROM TIMESTAMP(6) '1970-01-01 00:00:03.123456');",
            dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg(
            "SELECT EXTRACT('epoch' FROM TIMESTAMP(9) '1970-01-01 00:00:03.123456789');",
            dt)));

    ASSERT_EQ(
        -3,
        v<int64_t>(run_simple_agg(
            "SELECT EXTRACT('epoch' FROM TIMESTAMP(0) '1969-12-31 23:59:57');", dt)));
    ASSERT_EQ(
        -3,
        v<int64_t>(run_simple_agg(
            "SELECT EXTRACT('epoch' FROM TIMESTAMP(3) '1969-12-31 23:59:57.123');", dt)));
    ASSERT_EQ(
        -3,
        v<int64_t>(run_simple_agg(
            "SELECT EXTRACT('epoch' FROM TIMESTAMP(6) '1969-12-31 23:59:57.123456');",
            dt)));
    ASSERT_EQ(
        -3,
        v<int64_t>(run_simple_agg(
            "SELECT EXTRACT('epoch' FROM TIMESTAMP(9) '1969-12-31 23:59:57.123456789');",
            dt)));
  }
}

TEST_F(Select, DateTruncate2) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    ASSERT_EQ("1900-01-01 12:34:59", date_trunc("SECOND", "1900-01-01 12:34:59", dt));
    ASSERT_EQ("1900-01-01 12:35:00", date_trunc("SECOND", "1900-01-01 12:35:00", dt));
    ASSERT_EQ("3900-01-01 12:34:59", date_trunc("SECOND", "3900-01-01 12:34:59", dt));
    ASSERT_EQ("3900-01-01 12:35:00", date_trunc("SECOND", "3900-01-01 12:35:00", dt));

    ASSERT_EQ("1900-01-01 12:34:00", date_trunc("MINUTE", "1900-01-01 12:34:59", dt));
    ASSERT_EQ("1900-01-01 12:35:00", date_trunc("MINUTE", "1900-01-01 12:35:00", dt));
    ASSERT_EQ("3900-01-01 12:34:00", date_trunc("MINUTE", "3900-01-01 12:34:59", dt));
    ASSERT_EQ("3900-01-01 12:35:00", date_trunc("MINUTE", "3900-01-01 12:35:00", dt));

    ASSERT_EQ("1900-01-01 12:00:00", date_trunc("HOUR", "1900-01-01 12:59:59", dt));
    ASSERT_EQ("1900-01-01 13:00:00", date_trunc("HOUR", "1900-01-01 13:00:00", dt));
    ASSERT_EQ("3900-01-01 12:00:00", date_trunc("HOUR", "3900-01-01 12:59:59", dt));
    ASSERT_EQ("3900-01-01 13:00:00", date_trunc("HOUR", "3900-01-01 13:00:00", dt));

    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("QUARTERDAY", "1900-01-01 00:00:00", dt));
    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("QUARTERDAY", "1900-01-01 05:59:59", dt));
    ASSERT_EQ("1900-01-01 06:00:00", date_trunc("QUARTERDAY", "1900-01-01 06:00:00", dt));
    ASSERT_EQ("1900-01-01 06:00:00", date_trunc("QUARTERDAY", "1900-01-01 11:59:59", dt));
    ASSERT_EQ("1900-01-01 12:00:00", date_trunc("QUARTERDAY", "1900-01-01 12:00:00", dt));
    ASSERT_EQ("1900-01-01 12:00:00", date_trunc("QUARTERDAY", "1900-01-01 17:59:59", dt));
    ASSERT_EQ("1900-01-01 18:00:00", date_trunc("QUARTERDAY", "1900-01-01 18:00:00", dt));
    ASSERT_EQ("1900-01-01 18:00:00", date_trunc("QUARTERDAY", "1900-01-01 23:59:59", dt));
    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("QUARTERDAY", "3900-01-01 00:00:00", dt));
    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("QUARTERDAY", "3900-01-01 05:59:59", dt));
    ASSERT_EQ("3900-01-01 06:00:00", date_trunc("QUARTERDAY", "3900-01-01 06:00:00", dt));
    ASSERT_EQ("3900-01-01 06:00:00", date_trunc("QUARTERDAY", "3900-01-01 11:59:59", dt));
    ASSERT_EQ("3900-01-01 12:00:00", date_trunc("QUARTERDAY", "3900-01-01 12:00:00", dt));
    ASSERT_EQ("3900-01-01 12:00:00", date_trunc("QUARTERDAY", "3900-01-01 17:59:59", dt));
    ASSERT_EQ("3900-01-01 18:00:00", date_trunc("QUARTERDAY", "3900-01-01 18:00:00", dt));
    ASSERT_EQ("3900-01-01 18:00:00", date_trunc("QUARTERDAY", "3900-01-01 23:59:59", dt));

    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("DAY", "1900-01-01 00:00:00", dt));
    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("DAY", "1900-01-01 23:59:59", dt));
    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("DAY", "3900-01-01 00:00:00", dt));
    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("DAY", "3900-01-01 23:59:59", dt));

    // 1900-01-01 is a Monday (= start of "WEEK").
    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("WEEK", "1900-01-01 00:00:00", dt));
    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("WEEK", "1900-01-07 23:59:59", dt));
    ASSERT_EQ("1900-01-08 00:00:00", date_trunc("WEEK", "1900-01-08 00:00:00", dt));
    ASSERT_EQ("1900-01-08 00:00:00", date_trunc("WEEK", "1900-01-14 23:59:59", dt));
    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("WEEK", "3900-01-01 00:00:00", dt));
    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("WEEK", "3900-01-07 23:59:59", dt));
    ASSERT_EQ("3900-01-08 00:00:00", date_trunc("WEEK", "3900-01-08 00:00:00", dt));
    ASSERT_EQ("3900-01-08 00:00:00", date_trunc("WEEK", "3900-01-14 23:59:59", dt));

    // 1899-12-31 is a Sunday (= start of "WEEK_SUNDAY").
    ASSERT_EQ("1899-12-31 00:00:00",
              date_trunc("WEEK_SUNDAY", "1899-12-31 00:00:00", dt));
    ASSERT_EQ("1899-12-31 00:00:00",
              date_trunc("WEEK_SUNDAY", "1900-01-06 23:59:59", dt));
    ASSERT_EQ("1900-01-07 00:00:00",
              date_trunc("WEEK_SUNDAY", "1900-01-07 00:00:00", dt));
    ASSERT_EQ("1900-01-07 00:00:00",
              date_trunc("WEEK_SUNDAY", "1900-01-13 23:59:59", dt));
    ASSERT_EQ("3899-12-31 00:00:00",
              date_trunc("WEEK_SUNDAY", "3899-12-31 00:00:00", dt));
    ASSERT_EQ("3899-12-31 00:00:00",
              date_trunc("WEEK_SUNDAY", "3900-01-06 23:59:59", dt));
    ASSERT_EQ("3900-01-07 00:00:00",
              date_trunc("WEEK_SUNDAY", "3900-01-07 00:00:00", dt));
    ASSERT_EQ("3900-01-07 00:00:00",
              date_trunc("WEEK_SUNDAY", "3900-01-13 23:59:59", dt));

    // 1899-12-30 is a Saturday (= start of "WEEK_SATURDAY").
    ASSERT_EQ("1899-12-30 00:00:00",
              date_trunc("WEEK_SATURDAY", "1899-12-30 00:00:00", dt));
    ASSERT_EQ("1899-12-30 00:00:00",
              date_trunc("WEEK_SATURDAY", "1900-01-05 23:59:59", dt));
    ASSERT_EQ("1900-01-06 00:00:00",
              date_trunc("WEEK_SATURDAY", "1900-01-06 00:00:00", dt));
    ASSERT_EQ("1900-01-06 00:00:00",
              date_trunc("WEEK_SATURDAY", "1900-01-12 23:59:59", dt));
    ASSERT_EQ("3899-12-30 00:00:00",
              date_trunc("WEEK_SATURDAY", "3899-12-30 00:00:00", dt));
    ASSERT_EQ("3899-12-30 00:00:00",
              date_trunc("WEEK_SATURDAY", "3900-01-05 23:59:59", dt));
    ASSERT_EQ("3900-01-06 00:00:00",
              date_trunc("WEEK_SATURDAY", "3900-01-06 00:00:00", dt));
    ASSERT_EQ("3900-01-06 00:00:00",
              date_trunc("WEEK_SATURDAY", "3900-01-12 23:59:59", dt));

    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("MONTH", "1900-01-01 00:00:00", dt));
    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("MONTH", "1900-01-31 23:59:59", dt));
    ASSERT_EQ("1900-02-01 00:00:00", date_trunc("MONTH", "1900-02-01 00:00:00", dt));
    ASSERT_EQ("1900-02-01 00:00:00", date_trunc("MONTH", "1900-02-28 23:59:59", dt));
    ASSERT_EQ("1900-03-01 00:00:00", date_trunc("MONTH", "1900-03-01 00:00:00", dt));
    ASSERT_EQ("1900-03-01 00:00:00", date_trunc("MONTH", "1900-03-31 23:59:59", dt));
    ASSERT_EQ("1900-04-01 00:00:00", date_trunc("MONTH", "1900-04-01 00:00:00", dt));
    ASSERT_EQ("1900-04-01 00:00:00", date_trunc("MONTH", "1900-04-30 23:59:59", dt));
    ASSERT_EQ("1900-05-01 00:00:00", date_trunc("MONTH", "1900-05-01 00:00:00", dt));
    ASSERT_EQ("1900-05-01 00:00:00", date_trunc("MONTH", "1900-05-31 23:59:59", dt));
    ASSERT_EQ("1900-06-01 00:00:00", date_trunc("MONTH", "1900-06-01 00:00:00", dt));
    ASSERT_EQ("1900-06-01 00:00:00", date_trunc("MONTH", "1900-06-30 23:59:59", dt));
    ASSERT_EQ("1900-07-01 00:00:00", date_trunc("MONTH", "1900-07-01 00:00:00", dt));
    ASSERT_EQ("1900-07-01 00:00:00", date_trunc("MONTH", "1900-07-31 23:59:59", dt));
    ASSERT_EQ("1900-08-01 00:00:00", date_trunc("MONTH", "1900-08-01 00:00:00", dt));
    ASSERT_EQ("1900-08-01 00:00:00", date_trunc("MONTH", "1900-08-31 23:59:59", dt));
    ASSERT_EQ("1900-09-01 00:00:00", date_trunc("MONTH", "1900-09-01 00:00:00", dt));
    ASSERT_EQ("1900-09-01 00:00:00", date_trunc("MONTH", "1900-09-30 23:59:59", dt));
    ASSERT_EQ("1900-10-01 00:00:00", date_trunc("MONTH", "1900-10-01 00:00:00", dt));
    ASSERT_EQ("1900-10-01 00:00:00", date_trunc("MONTH", "1900-10-31 23:59:59", dt));
    ASSERT_EQ("1900-11-01 00:00:00", date_trunc("MONTH", "1900-11-01 00:00:00", dt));
    ASSERT_EQ("1900-11-01 00:00:00", date_trunc("MONTH", "1900-11-30 23:59:59", dt));
    ASSERT_EQ("1900-12-01 00:00:00", date_trunc("MONTH", "1900-12-01 00:00:00", dt));
    ASSERT_EQ("1900-12-01 00:00:00", date_trunc("MONTH", "1900-12-31 23:59:59", dt));

    ASSERT_EQ("2000-01-01 00:00:00", date_trunc("MONTH", "2000-01-01 00:00:00", dt));
    ASSERT_EQ("2000-01-01 00:00:00", date_trunc("MONTH", "2000-01-31 23:59:59", dt));
    ASSERT_EQ("2000-02-01 00:00:00", date_trunc("MONTH", "2000-02-01 00:00:00", dt));
    ASSERT_EQ("2000-02-01 00:00:00", date_trunc("MONTH", "2000-02-29 23:59:59", dt));
    ASSERT_EQ("2000-03-01 00:00:00", date_trunc("MONTH", "2000-03-01 00:00:00", dt));
    ASSERT_EQ("2000-03-01 00:00:00", date_trunc("MONTH", "2000-03-31 23:59:59", dt));
    ASSERT_EQ("2000-04-01 00:00:00", date_trunc("MONTH", "2000-04-01 00:00:00", dt));
    ASSERT_EQ("2000-04-01 00:00:00", date_trunc("MONTH", "2000-04-30 23:59:59", dt));
    ASSERT_EQ("2000-05-01 00:00:00", date_trunc("MONTH", "2000-05-01 00:00:00", dt));
    ASSERT_EQ("2000-05-01 00:00:00", date_trunc("MONTH", "2000-05-31 23:59:59", dt));
    ASSERT_EQ("2000-06-01 00:00:00", date_trunc("MONTH", "2000-06-01 00:00:00", dt));
    ASSERT_EQ("2000-06-01 00:00:00", date_trunc("MONTH", "2000-06-30 23:59:59", dt));
    ASSERT_EQ("2000-07-01 00:00:00", date_trunc("MONTH", "2000-07-01 00:00:00", dt));
    ASSERT_EQ("2000-07-01 00:00:00", date_trunc("MONTH", "2000-07-31 23:59:59", dt));
    ASSERT_EQ("2000-08-01 00:00:00", date_trunc("MONTH", "2000-08-01 00:00:00", dt));
    ASSERT_EQ("2000-08-01 00:00:00", date_trunc("MONTH", "2000-08-31 23:59:59", dt));
    ASSERT_EQ("2000-09-01 00:00:00", date_trunc("MONTH", "2000-09-01 00:00:00", dt));
    ASSERT_EQ("2000-09-01 00:00:00", date_trunc("MONTH", "2000-09-30 23:59:59", dt));
    ASSERT_EQ("2000-10-01 00:00:00", date_trunc("MONTH", "2000-10-01 00:00:00", dt));
    ASSERT_EQ("2000-10-01 00:00:00", date_trunc("MONTH", "2000-10-31 23:59:59", dt));
    ASSERT_EQ("2000-11-01 00:00:00", date_trunc("MONTH", "2000-11-01 00:00:00", dt));
    ASSERT_EQ("2000-11-01 00:00:00", date_trunc("MONTH", "2000-11-30 23:59:59", dt));
    ASSERT_EQ("2000-12-01 00:00:00", date_trunc("MONTH", "2000-12-01 00:00:00", dt));
    ASSERT_EQ("2000-12-01 00:00:00", date_trunc("MONTH", "2000-12-31 23:59:59", dt));

    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("MONTH", "3900-01-01 00:00:00", dt));
    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("MONTH", "3900-01-31 23:59:59", dt));
    ASSERT_EQ("3900-02-01 00:00:00", date_trunc("MONTH", "3900-02-01 00:00:00", dt));
    ASSERT_EQ("3900-02-01 00:00:00", date_trunc("MONTH", "3900-02-28 23:59:59", dt));
    ASSERT_EQ("3900-03-01 00:00:00", date_trunc("MONTH", "3900-03-01 00:00:00", dt));
    ASSERT_EQ("3900-03-01 00:00:00", date_trunc("MONTH", "3900-03-31 23:59:59", dt));
    ASSERT_EQ("3900-04-01 00:00:00", date_trunc("MONTH", "3900-04-01 00:00:00", dt));
    ASSERT_EQ("3900-04-01 00:00:00", date_trunc("MONTH", "3900-04-30 23:59:59", dt));
    ASSERT_EQ("3900-05-01 00:00:00", date_trunc("MONTH", "3900-05-01 00:00:00", dt));
    ASSERT_EQ("3900-05-01 00:00:00", date_trunc("MONTH", "3900-05-31 23:59:59", dt));
    ASSERT_EQ("3900-06-01 00:00:00", date_trunc("MONTH", "3900-06-01 00:00:00", dt));
    ASSERT_EQ("3900-06-01 00:00:00", date_trunc("MONTH", "3900-06-30 23:59:59", dt));
    ASSERT_EQ("3900-07-01 00:00:00", date_trunc("MONTH", "3900-07-01 00:00:00", dt));
    ASSERT_EQ("3900-07-01 00:00:00", date_trunc("MONTH", "3900-07-31 23:59:59", dt));
    ASSERT_EQ("3900-08-01 00:00:00", date_trunc("MONTH", "3900-08-01 00:00:00", dt));
    ASSERT_EQ("3900-08-01 00:00:00", date_trunc("MONTH", "3900-08-31 23:59:59", dt));
    ASSERT_EQ("3900-09-01 00:00:00", date_trunc("MONTH", "3900-09-01 00:00:00", dt));
    ASSERT_EQ("3900-09-01 00:00:00", date_trunc("MONTH", "3900-09-30 23:59:59", dt));
    ASSERT_EQ("3900-10-01 00:00:00", date_trunc("MONTH", "3900-10-01 00:00:00", dt));
    ASSERT_EQ("3900-10-01 00:00:00", date_trunc("MONTH", "3900-10-31 23:59:59", dt));
    ASSERT_EQ("3900-11-01 00:00:00", date_trunc("MONTH", "3900-11-01 00:00:00", dt));
    ASSERT_EQ("3900-11-01 00:00:00", date_trunc("MONTH", "3900-11-30 23:59:59", dt));
    ASSERT_EQ("3900-12-01 00:00:00", date_trunc("MONTH", "3900-12-01 00:00:00", dt));
    ASSERT_EQ("3900-12-01 00:00:00", date_trunc("MONTH", "3900-12-31 23:59:59", dt));

    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("QUARTER", "1900-01-01 00:00:00", dt));
    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("QUARTER", "1900-03-31 23:59:59", dt));
    ASSERT_EQ("1900-04-01 00:00:00", date_trunc("QUARTER", "1900-04-01 00:00:00", dt));
    ASSERT_EQ("1900-04-01 00:00:00", date_trunc("QUARTER", "1900-06-30 23:59:59", dt));
    ASSERT_EQ("1900-07-01 00:00:00", date_trunc("QUARTER", "1900-07-01 00:00:00", dt));
    ASSERT_EQ("1900-07-01 00:00:00", date_trunc("QUARTER", "1900-09-30 23:59:59", dt));
    ASSERT_EQ("1900-10-01 00:00:00", date_trunc("QUARTER", "1900-10-01 00:00:00", dt));
    ASSERT_EQ("1900-10-01 00:00:00", date_trunc("QUARTER", "1900-12-31 23:59:59", dt));

    ASSERT_EQ("2000-01-01 00:00:00", date_trunc("QUARTER", "2000-01-01 00:00:00", dt));
    ASSERT_EQ("2000-01-01 00:00:00", date_trunc("QUARTER", "2000-03-31 23:59:59", dt));
    ASSERT_EQ("2000-04-01 00:00:00", date_trunc("QUARTER", "2000-04-01 00:00:00", dt));
    ASSERT_EQ("2000-04-01 00:00:00", date_trunc("QUARTER", "2000-06-30 23:59:59", dt));
    ASSERT_EQ("2000-07-01 00:00:00", date_trunc("QUARTER", "2000-07-01 00:00:00", dt));
    ASSERT_EQ("2000-07-01 00:00:00", date_trunc("QUARTER", "2000-09-30 23:59:59", dt));
    ASSERT_EQ("2000-10-01 00:00:00", date_trunc("QUARTER", "2000-10-01 00:00:00", dt));
    ASSERT_EQ("2000-10-01 00:00:00", date_trunc("QUARTER", "2000-12-31 23:59:59", dt));

    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("QUARTER", "3900-01-01 00:00:00", dt));
    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("QUARTER", "3900-03-31 23:59:59", dt));
    ASSERT_EQ("3900-04-01 00:00:00", date_trunc("QUARTER", "3900-04-01 00:00:00", dt));
    ASSERT_EQ("3900-04-01 00:00:00", date_trunc("QUARTER", "3900-06-30 23:59:59", dt));
    ASSERT_EQ("3900-07-01 00:00:00", date_trunc("QUARTER", "3900-07-01 00:00:00", dt));
    ASSERT_EQ("3900-07-01 00:00:00", date_trunc("QUARTER", "3900-09-30 23:59:59", dt));
    ASSERT_EQ("3900-10-01 00:00:00", date_trunc("QUARTER", "3900-10-01 00:00:00", dt));
    ASSERT_EQ("3900-10-01 00:00:00", date_trunc("QUARTER", "3900-12-31 23:59:59", dt));

    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("YEAR", "1900-01-01 00:00:00", dt));
    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("YEAR", "1900-12-31 23:59:59", dt));
    ASSERT_EQ("1901-01-01 00:00:00", date_trunc("YEAR", "1901-01-01 00:00:00", dt));
    ASSERT_EQ("1901-01-01 00:00:00", date_trunc("YEAR", "1901-12-31 23:59:59", dt));

    ASSERT_EQ("2000-01-01 00:00:00", date_trunc("YEAR", "2000-01-01 00:00:00", dt));
    ASSERT_EQ("2000-01-01 00:00:00", date_trunc("YEAR", "2000-12-31 23:59:59", dt));
    ASSERT_EQ("2001-01-01 00:00:00", date_trunc("YEAR", "2001-01-01 00:00:00", dt));
    ASSERT_EQ("2001-01-01 00:00:00", date_trunc("YEAR", "2001-12-31 23:59:59", dt));

    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("YEAR", "3900-01-01 00:00:00", dt));
    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("YEAR", "3900-12-31 23:59:59", dt));
    ASSERT_EQ("3901-01-01 00:00:00", date_trunc("YEAR", "3901-01-01 00:00:00", dt));
    ASSERT_EQ("3901-01-01 00:00:00", date_trunc("YEAR", "3901-12-31 23:59:59", dt));

    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("DECADE", "1900-01-01 00:00:00", dt));
    ASSERT_EQ("1900-01-01 00:00:00", date_trunc("DECADE", "1909-12-31 23:59:59", dt));
    ASSERT_EQ("1910-01-01 00:00:00", date_trunc("DECADE", "1910-01-01 00:00:00", dt));
    ASSERT_EQ("1910-01-01 00:00:00", date_trunc("DECADE", "1919-12-31 23:59:59", dt));

    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("DECADE", "3900-01-01 00:00:00", dt));
    ASSERT_EQ("3900-01-01 00:00:00", date_trunc("DECADE", "3909-12-31 23:59:59", dt));
    ASSERT_EQ("3910-01-01 00:00:00", date_trunc("DECADE", "3910-01-01 00:00:00", dt));
    ASSERT_EQ("3910-01-01 00:00:00", date_trunc("DECADE", "3919-12-31 23:59:59", dt));

    ASSERT_EQ("1801-01-01 00:00:00", date_trunc("CENTURY", "1801-01-01 00:00:00", dt));
    ASSERT_EQ("1801-01-01 00:00:00", date_trunc("CENTURY", "1900-12-31 23:59:59", dt));
    ASSERT_EQ("1901-01-01 00:00:00", date_trunc("CENTURY", "1901-01-01 00:00:00", dt));
    ASSERT_EQ("1901-01-01 00:00:00", date_trunc("CENTURY", "2000-12-31 23:59:59", dt));
    ASSERT_EQ("2001-01-01 00:00:00", date_trunc("CENTURY", "2001-01-01 00:00:00", dt));
    ASSERT_EQ("2001-01-01 00:00:00", date_trunc("CENTURY", "2100-12-31 23:59:59", dt));
    ASSERT_EQ("3901-01-01 00:00:00", date_trunc("CENTURY", "3901-01-01 00:00:00", dt));
    ASSERT_EQ("3901-01-01 00:00:00", date_trunc("CENTURY", "4000-12-31 23:59:59", dt));

    ASSERT_EQ("0001-01-01 00:00:00", date_trunc("MILLENNIUM", "0001-01-01 00:00:00", dt));
    ASSERT_EQ("0001-01-01 00:00:00", date_trunc("MILLENNIUM", "1000-12-31 23:59:59", dt));
    ASSERT_EQ("1001-01-01 00:00:00", date_trunc("MILLENNIUM", "1001-01-01 00:00:00", dt));
    ASSERT_EQ("1001-01-01 00:00:00", date_trunc("MILLENNIUM", "1900-12-31 23:59:59", dt));
    ASSERT_EQ("1001-01-01 00:00:00", date_trunc("MILLENNIUM", "1901-01-01 00:00:00", dt));
    ASSERT_EQ("1001-01-01 00:00:00", date_trunc("MILLENNIUM", "2000-12-31 23:59:59", dt));
    ASSERT_EQ("2001-01-01 00:00:00", date_trunc("MILLENNIUM", "2001-01-01 00:00:00", dt));
    ASSERT_EQ("2001-01-01 00:00:00", date_trunc("MILLENNIUM", "3000-12-31 23:59:59", dt));
    ASSERT_EQ("3001-01-01 00:00:00", date_trunc("MILLENNIUM", "3001-01-01 00:00:00", dt));
    ASSERT_EQ("3001-01-01 00:00:00", date_trunc("MILLENNIUM", "4000-12-31 23:59:59", dt));
    ASSERT_EQ("4001-01-01 00:00:00", date_trunc("MILLENNIUM", "4001-01-01 00:00:00", dt));
    ASSERT_EQ("4001-01-01 00:00:00", date_trunc("MILLENNIUM", "5000-12-31 23:59:59", dt));
  }
}

TEST_F(Select, TimeRedux) {
  // The time tests need a general cleanup. Collect tests found from specific bugs here so
  // we don't accidentally remove them
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    EXPECT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM test WHERE o = (DATE '1999-09-01') OR CAST(o AS TIMESTAMP) = (TIMESTAMP '1999-09-09 00:00:00.000');)",
            dt)));
  }
}

TEST_F(Select, In) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE x IN (7, 8);", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (9, 10);", dt);
    c("SELECT COUNT(*) FROM test WHERE z IN (101, 102);", dt);
    c("SELECT COUNT(*) FROM test WHERE z IN (201, 202);", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IN ('real_foo', 'real_bar');", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IN ('real_foo', 'real_bar', 'real_baz', "
      "'foo');",
      dt);
    c("SELECT COUNT(*) FROM test WHERE str IN ('foo', 'bar', 'real_foo');", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, "
      "14, 15, 16, 17, 18, 19, 20);",
      dt);
  }
}

TEST_F(Select, DivByZero) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT x / 0 FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT 1 / 0 FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct x / 0) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT f / 0. FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT d / 0. FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT f / (f - f) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY y / (x - x);", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY z, y / (x - x);", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY MOD(y , (x - x));", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg(
            "SELECT SUM(x) / SUM(CASE WHEN str = 'none' THEN y ELSE 0 END) FROM test;",
            dt),
        std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT COUNT(*) FROM test WHERE y / (x - x) = 0;", dt),
                 std::runtime_error);
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE x = x OR  y / (x - x) = y;", dt)));
  }
}

TEST_F(Select, ReturnNullFromDivByZero) {
  g_null_div_by_zero = true;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x / 0 FROM test;", dt);
    c("SELECT 1 / 0 FROM test;", dt);
    c("SELECT f / 0. FROM test;", dt);
    c("SELECT d / 0. FROM test;", dt);
    c("SELECT f / (f - f) FROM test;", dt);
    c("SELECT COUNT(*) FROM test GROUP BY y / (x - x);", dt);
    c("SELECT COUNT(*) n FROM test GROUP BY z, y / (x - x) ORDER BY n ASC;", dt);
    c("SELECT SUM(x) / SUM(CASE WHEN str = 'none' THEN y ELSE 0 END) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE y / (x - x) = 0;", dt);
    c("SELECT COUNT(*) FROM test WHERE x = x OR  y / (x - x) = y;", dt);
  }
}

TEST_F(Select, ReturnInfFromDivByZero) {
  g_inf_div_by_zero = true;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT f / 0. FROM test;", "SELECT 2e308 FROM test;", dt);
    c("SELECT d / 0. FROM test;", "SELECT 2e308 FROM test;", dt);
    c("SELECT -f / 0. FROM test;", "SELECT -2e308 FROM test;", dt);
    c("SELECT -d / 0. FROM test;", "SELECT -2e308 FROM test;", dt);
    c("SELECT f / (f - f) FROM test;", "SELECT 2e308 FROM test;", dt);
    c("SELECT (f - f) / 0. FROM test;", dt);
  }
}

TEST_F(Select, ConstantFolding) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT 1 + 2 FROM test limit 1;", dt);
    c("SELECT 1 + 2.3 FROM test limit 1;", dt);
    c("SELECT 2.3 + 1 FROM test limit 1;", dt);
    c("SELECT 2 * 3 FROM test limit 1;", dt);
    c("SELECT 604 * 575 FROM test limit 1;", dt);
    c("SELECT 604 * (75 + 500) FROM test limit 1;", dt);
    c("SELECT 604 * (5 * 115) FROM test limit 1;", dt);
    c("SELECT 100000 + (1 - 604 * 575) FROM test limit 1;", dt);
    c("SELECT 1 + 604 * 575 FROM test limit 1;", dt);
    c("SELECT 2 + (1 - 604 * 575) FROM test limit 1;", dt);
    c("SELECT t + 604 * 575 FROM test limit 1;", dt);  // mul is folded in BIGINT
    c("SELECT z + 604 * 575 FROM test limit 1;", dt);
    c("SELECT 9.1 + 2.9999999999 FROM test limit 1;", dt);
    c("SELECT -9.1 - 2.9999999999 FROM test limit 1;", dt);
    c("SELECT -(9.1 + 99.22) FROM test limit 1;", dt);
    c("SELECT 3/2 FROM test limit 1;", dt);
    c("SELECT 3/2.0 FROM test limit 1;", dt);
    c("SELECT 11.1 * 2.22 FROM test limit 1;", dt);
    c("SELECT 1.01 * 1.00001 FROM test limit 1;", dt);
    c("SELECT 11.1 * 2.222222222 FROM test limit 1;", dt);
    c("SELECT 9.99 * 9999.9 FROM test limit 1;", dt);
    c("SELECT 9.22337203685477 * 9.223 FROM test limit 1;", dt);
    c("SELECT 3.0+8 from test limit 1;", dt);
    c("SELECT 3.0*8 from test limit 1;", dt);
    c("SELECT 1.79769e+308 * 0.1 FROM test limit 1;", dt);
    c("SELECT COUNT(*) FROM test WHERE 3.0+8 < 30;", dt);
    c("SELECT COUNT(*) FROM test WHERE 3.0*8 > 30.01;", dt);
    c("SELECT COUNT(*) FROM test WHERE 3.0*8 > 30.0001;", dt);
    c("SELECT COUNT(*) FROM test WHERE ff + 3.0*8 < 60.0/2;", dt);
    c("SELECT COUNT(*) FROM test WHERE t > 0 AND t = t;", dt);
    c("SELECT COUNT(*) FROM test WHERE t > 0 AND t <> t;", dt);
    c("SELECT COUNT(*) FROM test WHERE t > 0 OR t = t;", dt);
    c("SELECT COUNT(*) FROM test WHERE t > 0 OR t <> t;", dt);
    c("SELECT COUNT(*) FROM test where (604=575) OR (33.0<>12 AND 2.0001e+4>20000.9) "
      "OR (NOT t>=t OR f<>f OR (x=x AND x-x=0));",
      dt);
  }
}

TEST_F(Select, OverflowAndUnderFlow) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE z + 32600 > 0;", dt);
    c("SELECT COUNT(*) FROM test WHERE z + 32666 > 0;", dt);
    c("SELECT COUNT(*) FROM test WHERE -32670 - z < 0;", dt);
    c("SELECT COUNT(*) FROM test WHERE (z + 16333) * 2 > 0;", dt);
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test WHERE x + 2147483640 > 0;", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test WHERE -x - 2147483642 < 0;", dt),
        std::runtime_error);
    c("SELECT COUNT(*) FROM test WHERE t + 9223372036854774000 > 0;", dt);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT COUNT(*) FROM test WHERE t + 9223372036854775000 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT COUNT(*) FROM test WHERE -t - 9223372036854775000 < 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofd + x - 2 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT COUNT(*) FROM test WHERE ufd * 3 - ofd * 1024 < -2;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofd * 2 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofq + 1 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg(
            "SELECT COUNT(*) FROM test WHERE -ufq - 9223372036854775000 > 0;", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test WHERE -92233720368547758 - ofq <= 0;",
                         dt),
        std::runtime_error);
    c("SELECT cast((z - -32666) * 0.000190 as int) as key0, COUNT(*) AS val FROM test "
      "WHERE (z >= -32666 AND z < 31496) GROUP BY key0 HAVING key0 >= 0 AND key0 < 12 "
      "ORDER BY val DESC LIMIT 50 OFFSET 0;",
      dt);
    EXPECT_THROW(run_multiple_agg("SELECT dd * 2000000000000000 FROM test LIMIT 5;", dt),
                 std::runtime_error);
    c("SELECT dd * 200000000000000 FROM test ORDER BY dd ASC LIMIT 5;",
      dt);  // overflow avoided through decimal mul optimization
    c("SELECT COUNT(*) FROM test WHERE dd + 2.0000000000000009 > 110.0;",
      dt);  // no overflow in the cast
    EXPECT_THROW(
        run_multiple_agg(
            "SELECT COUNT(*) FROM test WHERE dd + 2.00000000000000099 > 110.0;", dt),
        std::runtime_error);  // overflow in the cast due to higher precision
    c("SELECT dd / 2.00000009 FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // dividend still fits after cast and division upscaling
    EXPECT_THROW(run_multiple_agg("SELECT dd / 2.000000099 FROM test LIMIT 1;", dt),
                 std::runtime_error);  // dividend overflows after cast and division
                                       // upscaling due to higher precision
    c("SELECT (dd - 40.6364668888) / 2 FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // decimal div by const optimization avoids overflow
    c("SELECT (dd - 40.6364668888) / x FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // decimal div by int cast optimization avoids overflow
    c("SELECT (dd - 40.63646688) / dd FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // dividend still fits after upscaling from cast and division
    EXPECT_THROW(run_multiple_agg("select (dd-40.6364668888)/dd from test limit 1;", dt),
                 std::runtime_error);  // dividend overflows on upscaling on a slightly
                                       // higher precision, test detection
    EXPECT_THROW(run_multiple_agg("SELECT CAST(x * 10000 AS SMALLINT) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(y * 1000 AS SMALLINT) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(x * -10000 AS SMALLINT) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(y * -1000 AS SMALLINT) FROM test;", dt),
                 std::runtime_error);
    c("SELECT cast((cast(z as int) - -32666) *0.000190 as int) as key0, "
      "COUNT(*) AS val FROM test WHERE (z >= -32666 AND z < 31496) "
      "GROUP BY key0 HAVING key0 >= 0 AND key0 < 12 ORDER BY val "
      "DESC LIMIT 50 OFFSET 0;",
      dt);
    c("select -1 * dd as expr from test order by expr asc;", dt);
    c("select dd * -1 as expr from test order by expr asc;", dt);
    c("select (dd - 1000000111.10) * dd as expr from test order by expr asc;", dt);
    c("select dd * (dd - 1000000111.10) as expr from test order by expr asc;", dt);
    // avoiding overflows in decimal compares against higher precision literals:
    // truncate literals based on the other side's precision, e.g. for d which is
    // DECIMAL(14,2)
    c("select count(*) from big_decimal_range_test where (d >  4.955357142857142);",
      dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d >= 4.955357142857142);",
      dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d <  4.955357142857142);",
      dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d <= 4.955357142857142);",
      dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d >= 4.950357142857142);",
      dt);  // compare with 4.951
    c("select count(*) from big_decimal_range_test where (d <  4.950357142857142);",
      dt);  // compare with 4.951
    c("select count(*) from big_decimal_range_test where (d < 59016609.300000056);",
      dt);  // compare with 59016609.301
    c("select count(*) from test where (t*123456 > 9681668.33071388567);",
      dt);  // compare with 9681668.3
    c("select count(*) from test where (x*12345678 < 9681668.33071388567);",
      dt);  // compare with 9681668.3
    c("select count(*) from test where (z*12345678 < 9681668.33071388567);",
      dt);  // compare with 9681668.3
    c("select count(*) from test where dd <= 111.222;", dt);
    c("select count(*) from test where dd >= -15264923.533545015;", dt);
    // avoiding overflows with constant folding and pushed down casts
    c("select count(*) + (604*575) from test;", dt);
    c("select count(*) - (604*575) from test;", dt);
    c("select count(*) * (604*575) from test;", dt);
    c("select (604*575) / count(*) from test;", dt);
    c("select (400604+575) / count(*) from test;", dt);
    c("select cast(count(*) as DOUBLE) + (604*575) from test;", dt);
    c("select cast(count(*) as DOUBLE) - (604*575) from test;", dt);
    c("select cast(count(*) as DOUBLE) * (604*575) from test;", dt);
    c("select (604*575) / cast(count(*) as DOUBLE) from test;", dt);
    c("select (12345-123456789012345) / cast(count(*) as DOUBLE) from test;", dt);
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg("SELECT COUNT(CAST(EXTRACT(QUARTER FROM CAST(NULL AS "
                                  "TIMESTAMP)) AS BIGINT) - 1) FROM test;",
                                  dt)));
  }
}

TEST_F(Select, DetectOverflowedLiteralBuf) {
  // constructing literal buf to trigger overflow takes too much time
  // so we mimic the literal buffer collection during codegen
  std::vector<CgenState::LiteralValue> literals;
  size_t literal_bytes{0};
  auto getOrAddLiteral = [&literals, &literal_bytes](const std::string& val) {
    const CgenState::LiteralValue var_val(val);
    literals.emplace_back(val);
    const auto lit_bytes = CgenState::literalBytes(var_val);
    literal_bytes = CgenState::addAligned(literal_bytes, lit_bytes);
    return literal_bytes - lit_bytes;
  };

  // add unique string literals until we detect the overflow
  // note that we only consider unique literals so we don't need to
  // lookup the existing literal buffer offset when adding the literal
  auto perform_test = [getOrAddLiteral]() {
    checked_int16_t checked_lit_off{-1};
    int added_literals = 0;
    try {
      for (; added_literals < 100000; ++added_literals) {
        checked_lit_off = getOrAddLiteral(std::to_string(added_literals));
      }
    } catch (const std::range_error& e) {
      throw TooManyLiterals();
    }
  };
  EXPECT_THROW(perform_test(), TooManyLiterals);
}

TEST_F(Select, BooleanColumn) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE bn;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE b;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE NOT bn;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows + g_num_rows / 2),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x < 8 AND bn;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE x < 8 AND NOT bn;", dt)));
    ASSERT_EQ(5,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 7 OR false;", dt)));
    ASSERT_EQ(7,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(x) FROM test WHERE b = CAST('t' AS boolean);", dt)));
    ASSERT_EQ(static_cast<int64_t>(3 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  " SELECT SUM(2 *(CASE when x = 7 then 1 else 0 END)) FROM test;", dt)));
    c("SELECT COUNT(*) AS n FROM test GROUP BY x = 7, b ORDER BY n;", dt);
  }
}

TEST_F(Select, UnsupportedCast) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT CAST(x AS VARCHAR) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(f AS VARCHAR) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(d AS VARCHAR) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(f AS DECIMAL) FROM test;", dt),
                 std::runtime_error);
  }
}

TEST_F(Select, CastFromLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(2.3 AS TINYINT) FROM test;", dt);
    c("SELECT CAST(2.3 AS SMALLINT) FROM test;", dt);
    c("SELECT CAST(2.3 AS INT) FROM test;", dt);
    c("SELECT CAST(2.3 AS BIGINT) FROM test;", dt);
    c("SELECT CAST(2.3 AS FLOAT) FROM test;", dt);
    c("SELECT CAST(2.3 AS DOUBLE) FROM test;", dt);
    c("SELECT CAST(2.3 AS DECIMAL(2, 1)) FROM test;", dt);
    c("SELECT CAST(2.3 AS NUMERIC(2, 1)) FROM test;", dt);
    c("SELECT CAST(CAST(10 AS float) / CAST(3600 as float) AS float) FROM test LIMIT 1;",
      dt);
    c("SELECT CAST(CAST(10 AS double) / CAST(3600 as double) AS double) FROM test LIMIT "
      "1;",
      dt);
    c("SELECT z from test where z = -78;", dt);
  }
}

TEST_F(Select, CastFromNull) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(NULL AS TINYINT) FROM test;", dt);
    c("SELECT CAST(NULL AS SMALLINT) FROM test;", dt);
    c("SELECT CAST(NULL AS INT) FROM test;", dt);
    c("SELECT CAST(NULL AS BIGINT) FROM test;", dt);
    c("SELECT CAST(NULL AS FLOAT) FROM test;", dt);
    c("SELECT CAST(NULL AS DOUBLE) FROM test;", dt);
    c("SELECT CAST(NULL AS DECIMAL) FROM test;", dt);
    c("SELECT CAST(NULL AS NUMERIC) FROM test;", dt);
  }
}

TEST_F(Select, CastFromNull2) {
  createTable("cast_from_null2",
              {{"d", SQLTypeInfo(kDOUBLE)}, {"dd", SQLTypeInfo(kDECIMAL, 8, 2, false)}});
  insertCsvValues("cast_from_null2", "1.0,");
  sqlite_comparator_.query("DROP TABLE IF EXISTS cast_from_null2;");
  sqlite_comparator_.query("CREATE TABLE cast_from_null2 (d DOUBLE, dd DECIMAL(8,2));");
  sqlite_comparator_.query("INSERT INTO cast_from_null2 VALUES (1.0, NULL);");
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT d * dd FROM cast_from_null2;", dt);
  }
  dropTable("cast_from_null2");
}

TEST_F(Select, CastRound) {
  auto const run = [](char const* n, char const* type, ExecutorDeviceType const dt) {
    return run_simple_agg(std::string("SELECT CAST(") + n + " AS " + type + ");", dt);
  };
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_EQ(127, v<int64_t>(run("127.4999999999999999", "TINYINT", dt)));
    EXPECT_ANY_THROW(run("127.5", "TINYINT", dt));  // overflow
    EXPECT_EQ(-128, v<int64_t>(run("-128.4999999999999999", "TINYINT", dt)));
    EXPECT_ANY_THROW(run("-128.5", "TINYINT", dt));  // overflow

    EXPECT_EQ(32767, v<int64_t>(run("32767.49999999999999", "SMALLINT", dt)));
    EXPECT_ANY_THROW(run("32767.5", "SMALLINT", dt));  // overflow
    EXPECT_EQ(-32768, v<int64_t>(run("-32768.49999999999999", "SMALLINT", dt)));
    EXPECT_ANY_THROW(run("-32768.5", "SMALLINT", dt));  // overflow

    EXPECT_EQ(2147483647, v<int64_t>(run("2147483647.499999999", "INT", dt)));
    EXPECT_ANY_THROW(run("2147483647.5", "INT", dt));  // overflow
    EXPECT_EQ(-2147483648, v<int64_t>(run("-2147483648.499999999", "INT", dt)));
    EXPECT_ANY_THROW(run("-2147483648.5", "INT", dt));  // overflow

    EXPECT_EQ(std::numeric_limits<int64_t>::max(),
              v<int64_t>(run("9223372036854775807.", "BIGINT", dt)));
    EXPECT_ANY_THROW(run("9223372036854775807.0", "BIGINT", dt));  // out of range
    EXPECT_ANY_THROW(run("9223372036854775807.5", "BIGINT", dt));  // out of range
    EXPECT_EQ(std::numeric_limits<int64_t>::min(),
              v<int64_t>(run("-9223372036854775808.", "BIGINT", dt)));
    EXPECT_ANY_THROW(run("-9223372036854775808.0", "BIGINT", dt));  // out of range
    EXPECT_ANY_THROW(run("-9223372036854775808.5", "BIGINT", dt));  // out of range

    EXPECT_EQ(1e18f, v<float>(run("999999999999999999", "FLOAT", dt)));
    EXPECT_EQ(1e10f, v<float>(run("9999999999.99999999", "FLOAT", dt)));
    EXPECT_EQ(-1e18f, v<float>(run("-999999999999999999", "FLOAT", dt)));
    EXPECT_EQ(-1e10f, v<float>(run("-9999999999.99999999", "FLOAT", dt)));

    EXPECT_EQ(1e18, v<double>(run("999999999999999999", "DOUBLE", dt)));
    EXPECT_EQ(1e10, v<double>(run("9999999999.99999999", "DOUBLE", dt)));
    EXPECT_EQ(-1e18, v<double>(run("-999999999999999999", "DOUBLE", dt)));
    EXPECT_EQ(-1e10, v<double>(run("-9999999999.99999999", "DOUBLE", dt)));

    EXPECT_ANY_THROW(run("9223372036854775808e0", "BIGINT", dt));  // overflow
    EXPECT_ANY_THROW(run("9223372036854775807e0", "BIGINT", dt));  // overflow
    EXPECT_ANY_THROW(run("9223372036854775296e0", "BIGINT", dt));  // overflow
    // RHS = Largest integer that doesn't overflow when cast to DOUBLE to BIGINT.
    // LHS = Largest double value less than std::numeric_limits<int64_t>::max().
    EXPECT_EQ(9223372036854774784ll,
              v<int64_t>(run("9223372036854775295e0", "BIGINT", dt)));
    EXPECT_EQ(std::numeric_limits<int64_t>::min(),
              v<int64_t>(run("-9223372036854775808e0", "BIGINT", dt)));
    /* These results may be platform-dependent so are not included in tests.
    EXPECT_EQ(std::numeric_limits<int64_t>::min(),
              v<int64_t>(run("-9223372036854776959e0", "BIGINT", dt)));
    EXPECT_ANY_THROW(run("-9223372036854776960e0", "BIGINT", dt));  // overflow
    */

    // Apply BIGINT tests to DECIMAL
    EXPECT_ANY_THROW(run("9223372036854775808e0", "DECIMAL", dt));  // overflow
    EXPECT_ANY_THROW(run("9223372036854775807e0", "DECIMAL", dt));  // overflow
    EXPECT_ANY_THROW(run("9223372036854775296e0", "DECIMAL", dt));  // overflow
    EXPECT_EQ(9223372036854774784.0,
              v<double>(run("9223372036854775295e0", "DECIMAL", dt)));
    EXPECT_EQ(static_cast<double>(std::numeric_limits<int64_t>::min()),
              v<double>(run("-9223372036854775808e0", "DECIMAL", dt)));

    EXPECT_ANY_THROW(run("2147483647.5e0", "INT", dt));  // overflow
    EXPECT_EQ(2147483647, v<int64_t>(run("2147483647.4999e0", "BIGINT", dt)));
    EXPECT_EQ(std::numeric_limits<int32_t>::min(),
              v<int64_t>(run("-2147483648.4999e0", "INT", dt)));
    EXPECT_ANY_THROW(run("-2147483648.5e0", "INT", dt));  // overflow

    EXPECT_ANY_THROW(run("32767.5e0", "SMALLINT", dt));  // overflow
    EXPECT_EQ(32767, v<int64_t>(run("32767.4999e0", "SMALLINT", dt)));
    EXPECT_EQ(-32768, v<int64_t>(run("-32768.4999e0", "SMALLINT", dt)));
    EXPECT_ANY_THROW(run("-32768.5e0", "SMALLINT", dt));  // overflow

    EXPECT_ANY_THROW(run("127.5e0", "TINYINT", dt));  // overflow
    EXPECT_EQ(127, v<int64_t>(run("127.4999e0", "TINYINT", dt)));
    EXPECT_EQ(-128, v<int64_t>(run("-128.4999e0", "TINYINT", dt)));
    EXPECT_ANY_THROW(run("-128.5e0", "TINYINT", dt));  // overflow

    EXPECT_TRUE(
        v<int64_t>(run_simple_agg("SELECT '292277026596-12-04 15:30:07' = "
                                  "CAST(9223372036854775807 AS TIMESTAMP(0));",
                                  dt)));
    EXPECT_TRUE(
        v<int64_t>(run_simple_agg("SELECT '292278994-08-17 07:12:55.807' = "
                                  "CAST(9223372036854775807 AS TIMESTAMP(3));",
                                  dt)));
    EXPECT_TRUE(v<int64_t>(
        run_simple_agg("SELECT CAST('294247-01-10 04:00:54.775807' AS TIMESTAMP(6)) = "
                       "CAST(9223372036854775807 AS TIMESTAMP(6));",
                       dt)));
    EXPECT_TRUE(v<int64_t>(
        run_simple_agg("SELECT CAST('2262-04-11 23:47:16.854775807' AS TIMESTAMP(9)) = "
                       "CAST(9223372036854775807 AS TIMESTAMP(9));",
                       dt)));
  }
}

TEST_F(Select, CastRoundNullable) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_EQ(
        20,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE ROUND(f+0.2) = CAST(f+0.2 AS INT);", dt)));
    EXPECT_EQ(
        10,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE ROUND(fn-0.2) = CAST(fn-0.2 AS INT);", dt)));
    EXPECT_EQ(10,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST(fn AS INT) IS NULL;", dt)));
    EXPECT_EQ(11,
              v<int64_t>(run_simple_agg("SELECT CAST(CAST(x AS FLOAT) * 1.6 AS INT) AS "
                                        "key0 FROM test GROUP BY key0 ORDER BY key0;",
                                        dt)));
  }
}

TEST_F(Select, ExtensionFunctionsTypeMatching) {
  createTable("extension_func_type_match_test",
              {
                  {"tinyint_type", SQLTypeInfo(kTINYINT)},
                  {"smallint_type", SQLTypeInfo(kSMALLINT)},
                  {"int_type", SQLTypeInfo(kTINYINT)},
                  {"bigint_type", SQLTypeInfo(kBIGINT)},
                  {"float_type", SQLTypeInfo(kFLOAT)},
                  {"double_type", SQLTypeInfo(kDOUBLE)},
                  {"decimal_7_type", SQLTypeInfo(kDECIMAL, 7, 1, false)},
                  {"decimal_8_type", SQLTypeInfo(kDECIMAL, 8, 1, false)},
              });
  insertCsvValues("extension_func_type_match_test", "10,10,10,10,10.0,10.0,10.0,10.0");
  const double float_result = 2.302585124969482;  // log(10) result using the fp32 version
                                                  // of the log extension function
  const double double_result =
      2.302585092994046;  // log(10) result using the fp64 version of the log extension
                          // function
  constexpr double RESULT_EPS =
      1.0e-8;  // Sufficient to differentiate fp32 and fp64 results
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      ASSERT_NEAR(
          double_result, v<double>(run_simple_agg("SELECT log(10);", dt)), RESULT_EPS);
    }

    {
      ASSERT_NEAR(
          double_result, v<double>(run_simple_agg("SELECT log(10.0);", dt)), RESULT_EPS);
    }

    {
      ASSERT_NEAR(double_result,
                  v<double>(run_simple_agg("SELECT log(CAST(10.0 AS FLOAT));", dt)),
                  RESULT_EPS);
    }

    {
      ASSERT_NEAR(
          float_result,
          v<double>(run_simple_agg(
              "SELECT log(tinyint_type) FROM extension_func_type_match_test;", dt)),
          RESULT_EPS);
    }

    {
      ASSERT_NEAR(
          float_result,
          v<double>(run_simple_agg(
              "SELECT log(smallint_type) FROM extension_func_type_match_test;", dt)),
          RESULT_EPS);
    }

    {
      ASSERT_NEAR(float_result,
                  v<double>(run_simple_agg(
                      "SELECT log(int_type) FROM extension_func_type_match_test;", dt)),
                  RESULT_EPS);
    }

    {
      ASSERT_NEAR(
          double_result,
          v<double>(run_simple_agg(
              "SELECT log(bigint_type) FROM extension_func_type_match_test;", dt)),
          RESULT_EPS);
    }

    {
      ASSERT_NEAR(float_result,
                  v<double>(run_simple_agg(
                      "SELECT log(float_type) FROM extension_func_type_match_test;", dt)),
                  RESULT_EPS);
    }

    {
      ASSERT_NEAR(
          double_result,
          v<double>(run_simple_agg(
              "SELECT log(double_type) FROM extension_func_type_match_test;", dt)),
          RESULT_EPS);
    }

    {
      ASSERT_NEAR(
          float_result,
          v<double>(run_simple_agg(
              "SELECT log(decimal_7_type) FROM extension_func_type_match_test;", dt)),
          RESULT_EPS);
    }

    {
      ASSERT_NEAR(
          double_result,
          v<double>(run_simple_agg(
              "SELECT log(decimal_8_type) FROM extension_func_type_match_test;", dt)),
          RESULT_EPS);
    }
  }
  dropTable("extension_func_type_match_test");
}

TEST_F(Select, CastDecimalToDecimal) {
  createTable("decimal_to_decimal_test",
              {{"id", SQLTypeInfo(kINT)}, {"val", SQLTypeInfo(kDECIMAL, 10, 5, false)}});
  insertCsvValues("decimal_to_decimal_test",
                  "1,456.78956\n2,456.12345\n-1,-456.78956\n-2,-456.12345");

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    ASSERT_NEAR(456.78956,
                v<double>(run_simple_agg(
                    "SELECT val FROM decimal_to_decimal_test WHERE id = 1;", dt)),
                456.78956 * EPS);
    ASSERT_NEAR(-456.78956,
                v<double>(run_simple_agg(
                    "SELECT val FROM decimal_to_decimal_test WHERE id = -1;", dt)),
                456.78956 * EPS);
    ASSERT_NEAR(456.12345,
                v<double>(run_simple_agg(
                    "SELECT val FROM decimal_to_decimal_test WHERE id = 2;", dt)),
                EPS);
    ASSERT_NEAR(-456.12345,
                v<double>(run_simple_agg(
                    "SELECT val FROM decimal_to_decimal_test WHERE id = -2;", dt)),
                456.12345 * EPS);

    ASSERT_NEAR(456.7896,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,4)) FROM "
                                         "decimal_to_decimal_test WHERE id = 1;",
                                         dt)),
                456.7896 * EPS);
    ASSERT_NEAR(-456.7896,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,4)) FROM "
                                         "decimal_to_decimal_test WHERE id = -1;",
                                         dt)),
                456.7896 * EPS);
    ASSERT_NEAR(456.123,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,4)) FROM "
                                         "decimal_to_decimal_test WHERE id = 2;",
                                         dt)),
                456.123 * EPS);
    ASSERT_NEAR(-456.123,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,4)) FROM "
                                         "decimal_to_decimal_test WHERE id = -2;",
                                         dt)),
                456.123 * EPS);

    ASSERT_NEAR(456.790,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,3)) FROM "
                                         "decimal_to_decimal_test WHERE id = 1;",
                                         dt)),
                456.790 * EPS);
    ASSERT_NEAR(-456.790,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,3)) FROM "
                                         "decimal_to_decimal_test WHERE id = -1;",
                                         dt)),
                456.790 * EPS);
    ASSERT_NEAR(456.1234,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,3)) FROM "
                                         "decimal_to_decimal_test WHERE id = 2;",
                                         dt)),
                456.1234 * EPS);
    ASSERT_NEAR(-456.1234,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,3)) FROM "
                                         "decimal_to_decimal_test WHERE id = -2;",
                                         dt)),
                456.1234 * EPS);

    ASSERT_NEAR(456.79,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,2)) FROM "
                                         "decimal_to_decimal_test WHERE id = 1;",
                                         dt)),
                456.79 * EPS);
    ASSERT_NEAR(-456.79,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,2)) FROM "
                                         "decimal_to_decimal_test WHERE id = -1;",
                                         dt)),
                456.79 * EPS);
    ASSERT_NEAR(456.12,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,2)) FROM "
                                         "decimal_to_decimal_test WHERE id = 2;",
                                         dt)),
                456.12 * EPS);
    ASSERT_NEAR(-456.12,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,2)) FROM "
                                         "decimal_to_decimal_test WHERE id = -2;",
                                         dt)),
                456.12 * EPS);

    ASSERT_NEAR(456.8,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,1)) FROM "
                                         "decimal_to_decimal_test WHERE id = 1;",
                                         dt)),
                456.8 * EPS);
    ASSERT_NEAR(-456.8,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,1)) FROM "
                                         "decimal_to_decimal_test WHERE id = -1;",
                                         dt)),
                456.8 * EPS);
    ASSERT_NEAR(456.1,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,1)) FROM "
                                         "decimal_to_decimal_test WHERE id = 2;",
                                         dt)),
                456.1 * EPS);
    ASSERT_NEAR(-456.1,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,1)) FROM "
                                         "decimal_to_decimal_test WHERE id = -2;",
                                         dt)),
                456.1 * EPS);
    ASSERT_NEAR(457,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,0)) FROM "
                                         "decimal_to_decimal_test WHERE id = 1;",
                                         dt)),
                457 * EPS);
    ASSERT_NEAR(-457,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,0)) FROM "
                                         "decimal_to_decimal_test WHERE id = -1;",
                                         dt)),
                457 * EPS);
    ASSERT_NEAR(456,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,0)) FROM "
                                         "decimal_to_decimal_test WHERE id = 2;",
                                         dt)),
                456 * EPS);
    ASSERT_NEAR(-456,
                v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,0)) FROM "
                                         "decimal_to_decimal_test WHERE id = -2;",
                                         dt)),
                456 * EPS);

    ASSERT_EQ(457,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(val AS BIGINT) FROM decimal_to_decimal_test WHERE id = 1;",
                  dt)));
    ASSERT_EQ(
        -457,
        v<int64_t>(run_simple_agg(
            "SELECT CAST(val AS BIGINT) FROM decimal_to_decimal_test WHERE id = -1;",
            dt)));
    ASSERT_EQ(456,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(val AS BIGINT) FROM decimal_to_decimal_test WHERE id = 2;",
                  dt)));
    ASSERT_EQ(
        -456,
        v<int64_t>(run_simple_agg(
            "SELECT CAST(val AS BIGINT) FROM decimal_to_decimal_test WHERE id = -2;",
            dt)));
  }
  dropTable("decimal_to_decimal_test");
}

TEST_F(Select, ColumnWidths) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT DISTINCT x FROM test_inner ORDER BY x;", dt);
    c("SELECT DISTINCT y FROM test_inner ORDER BY y;", dt);
    c("SELECT DISTINCT xx FROM test_inner ORDER BY xx;", dt);
    c("SELECT x, xx, y FROM test_inner GROUP BY x, xx, y ORDER BY x, xx, y;", dt);
    c("SELECT x, xx, y FROM test_inner GROUP BY x, xx, y ORDER BY x, xx, y;", dt);
    c("SELECT DISTINCT str from test_inner ORDER BY str;", dt);
    c("SELECT DISTINCT t FROM test ORDER BY t;", dt);
    c("SELECT DISTINCT t, z FROM test GROUP BY t, z ORDER BY t, z;", dt);
    c("SELECT fn from test where fn < -100.7 ORDER BY fn;", dt);
    c("SELECT fixed_str, SUM(f)/SUM(t)  FROM test WHERE fixed_str IN ('foo','bar') GROUP "
      "BY fixed_str ORDER BY "
      "fixed_str;",
      dt);
  }
}

TEST_F(Select, TimeInterval) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(
        60 * 60 * 1000LL,
        v<int64_t>(run_simple_agg("SELECT INTERVAL '1' HOUR FROM test LIMIT 1;", dt)));
    ASSERT_EQ(
        24 * 60 * 60 * 1000LL,
        v<int64_t>(run_simple_agg("SELECT INTERVAL '1' DAY FROM test LIMIT 1;", dt)));
    ASSERT_EQ(1LL,
              v<int64_t>(run_simple_agg(
                  "SELECT (INTERVAL '1' YEAR)/12 FROM test order by o LIMIT 1;", dt)));
    ASSERT_EQ(
        1LL,
        v<int64_t>(run_simple_agg(
            "SELECT INTERVAL '1' MONTH FROM test group by m order by m LIMIT 1;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE INTERVAL '1' MONTH < INTERVAL '2' MONTH;",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE INTERVAL '1' DAY < INTERVAL '2' DAY;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test GROUP BY INTERVAL '1' DAY;", dt)));
    ASSERT_EQ(3 * 60 * 60 * 1000LL,
              v<int64_t>(
                  run_simple_agg("SELECT 3 * INTERVAL '1' HOUR FROM test LIMIT 1;", dt)));
    ASSERT_EQ(3 * 60 * 60 * 1000LL,
              v<int64_t>(
                  run_simple_agg("SELECT INTERVAL '1' HOUR * 3 FROM test LIMIT 1;", dt)));
    ASSERT_EQ(7LL,
              v<int64_t>(run_simple_agg(
                  "SELECT INTERVAL '1' MONTH * x FROM test WHERE x <> 8 LIMIT 1;", dt)));
    ASSERT_EQ(7LL,
              v<int64_t>(run_simple_agg(
                  "SELECT x * INTERVAL '1' MONTH FROM test WHERE x <> 8 LIMIT 1;", dt)));
    ASSERT_EQ(42LL,
              v<int64_t>(run_simple_agg(
                  "SELECT INTERVAL '1' MONTH * y FROM test WHERE y <> 43 LIMIT 1;", dt)));
    ASSERT_EQ(42LL,
              v<int64_t>(run_simple_agg(
                  "SELECT y * INTERVAL '1' MONTH FROM test WHERE y <> 43 LIMIT 1;", dt)));
    ASSERT_EQ(
        1002LL,
        v<int64_t>(run_simple_agg(
            "SELECT INTERVAL '1' MONTH * t FROM test WHERE t <> 1001 LIMIT 1;", dt)));
    ASSERT_EQ(
        1002LL,
        v<int64_t>(run_simple_agg(
            "SELECT t * INTERVAL '1' MONTH FROM test WHERE t <> 1001 LIMIT 1;", dt)));
    ASSERT_EQ(
        3LL,
        v<int64_t>(run_simple_agg(
            "SELECT INTERVAL '1' MONTH + INTERVAL '2' MONTH FROM test LIMIT 1;", dt)));
    ASSERT_EQ(
        1388534400LL,
        v<int64_t>(run_simple_agg("SELECT CAST(m AS date) + CAST(TRUNCATE(-1 * "
                                  "(EXTRACT(DOY FROM m) - 1), 0) AS INTEGER) * INTERVAL "
                                  "'1' DAY AS g FROM test GROUP BY g;",
                                  dt)));
    ASSERT_EQ(
        1417392000LL,
        v<int64_t>(run_simple_agg("SELECT CAST(m AS date) + CAST(TRUNCATE(-1 * "
                                  "(EXTRACT(DAY FROM m) - 1), 0) AS INTEGER) * INTERVAL "
                                  "'1' DAY AS g FROM test GROUP BY g;",
                                  dt)));
    ASSERT_EQ(1418508000LL,
              v<int64_t>(run_simple_agg("SELECT CAST(m AS date) + EXTRACT(HOUR FROM m) * "
                                        "INTERVAL '1' HOUR AS g FROM test GROUP BY g;",
                                        dt)));
    ASSERT_EQ(
        1388534400LL,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(SQL_TSI_DAY, CAST(TRUNCATE(-1 * "
                                  "(EXTRACT(DOY from m) - 1), 0) AS INTEGER), "
                                  "CAST(m AS DATE)) AS g FROM test GROUP BY g;",
                                  dt)));
    ASSERT_EQ(
        1417392000LL,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(SQL_TSI_DAY, CAST(TRUNCATE(-1 * "
                                  "(EXTRACT(DAY from m) - 1), 0) AS INTEGER), "
                                  "CAST(m AS DATE)) AS g FROM test GROUP BY g;",
                                  dt)));
    ASSERT_EQ(1418508000LL,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SQL_TSI_HOUR, EXTRACT(HOUR from "
                  "m), CAST(m AS DATE)) AS g FROM test GROUP BY g order by g;",
                  dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' + INTERVAL '1' YEAR) = "
                                        "DATE '2009-01-31' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' + INTERVAL '5' YEAR) = "
                                        "DATE '2013-01-31' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' - INTERVAL '1' YEAR) = "
                                        "DATE '2007-01-31' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' - INTERVAL '4' YEAR) = "
                                        "DATE '2004-01-31' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' + INTERVAL '1' MONTH) "
                                        "= DATE '2008-02-29' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' + INTERVAL '5' MONTH) "
                                        "= DATE '2008-06-30' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' - INTERVAL '1' MONTH) "
                                        "= DATE '2007-12-31' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' - INTERVAL '4' MONTH) "
                                        "= DATE '2007-09-30' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-2-28' + INTERVAL '1' DAY) = "
                                        "DATE '2008-02-29' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2009-2-28' + INTERVAL '1' DAY) = "
                                        "DATE '2009-03-01' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-2-28' + INTERVAL '4' DAY) = "
                                        "DATE '2008-03-03' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2009-2-28' + INTERVAL '4' DAY) = "
                                        "DATE '2009-03-04' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-03-01' - INTERVAL '1' DAY) = "
                                        "DATE '2008-02-29' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2009-03-01' - INTERVAL '1' DAY) = "
                                        "DATE '2009-02-28' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-03-03' - INTERVAL '4' DAY) = "
                                        "DATE '2008-02-28' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2009-03-04' - INTERVAL '4' DAY) = "
                                        "DATE '2009-02-28' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT m = TIMESTAMP '2014-12-13 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '1' SECOND) = TIMESTAMP "
                                        "'2014-12-13 22:23:16' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '1' MINUTE) = TIMESTAMP "
                                        "'2014-12-13 22:24:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '1' HOUR) = TIMESTAMP "
                                        "'2014-12-13 23:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '2' DAY) = TIMESTAMP "
                                        "'2014-12-15 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '1' MONTH) = TIMESTAMP "
                                        "'2015-01-13 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '1' YEAR) = TIMESTAMP "
                                        "'2015-12-13 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT (m - 5 * INTERVAL '1' SECOND) = TIMESTAMP "
                                  "'2014-12-13 22:23:10' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT (m - x * INTERVAL '1' MINUTE) = TIMESTAMP "
                                  "'2014-12-13 22:16:15' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT (m - 2 * x * INTERVAL '1' HOUR) = TIMESTAMP "
                                  "'2014-12-13 8:23:15' from test limit 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m - x * INTERVAL '1' DAY) = TIMESTAMP "
                                        "'2014-12-06 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m - x * INTERVAL '1' MONTH) = TIMESTAMP "
                                        "'2014-05-13 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m - x * INTERVAL '1' YEAR) = TIMESTAMP "
                                        "'2007-12-13 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m - INTERVAL '5' DAY + INTERVAL '2' HOUR - x * INTERVAL '2' "
                  "SECOND) +"
                  "(x - 1) * INTERVAL '1' MONTH - x * INTERVAL '10' YEAR = "
                  "TIMESTAMP '1945-06-09 00:23:01' from test limit 1;",
                  dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "select count(*) from test where m < CAST (o AS TIMESTAMP) + INTERVAL '10' "
            "YEAR AND m > CAST(o AS TIMESTAMP) - INTERVAL '10' YEAR;",
            dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            "select count(*) from test where m < CAST (o AS TIMESTAMP) + INTERVAL '16' "
            "YEAR AND m > CAST(o AS TIMESTAMP) - INTERVAL '16' YEAR;",
            dt)));

    ASSERT_EQ(1,
              v<int64_t>(
                  run_simple_agg("SELECT o = DATE '1999-09-09' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (o + INTERVAL '10' DAY) = DATE '1999-09-19' from test limit 1;",
                  dt)));
  }
}

TEST_F(Select, LogicalValues) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // empty logical values
    c("SELECT 1 + 2;", dt);
    c("SELECT 1 * 2.1;", dt);
    c("SELECT 'alex', 'omnisci';", dt);
    c("SELECT COALESCE(5, NULL, 4);", dt);
    c("SELECT abs(-5) AS tmp;", dt);

    EXPECT_EQ(6, v<double>(run_simple_agg("SELECT ceil(5.556) AS tmp;", dt)));
    EXPECT_EQ(5, v<double>(run_simple_agg("SELECT floor(5.556) AS tmp;", dt)));

    // values
    c("SELECT * FROM (VALUES(1,2,3));", dt);
    c("SELECT * FROM (VALUES(1, NULL, 3));", dt);
    c("SELECT * FROM (VALUES(1, 2), (3, NULL));", dt);
    c("SELECT * FROM (SELECT * FROM (VALUES (1,2) , (3,4)) t1) t0 LIMIT 5;", dt);

    {
      auto rows = run_multiple_agg("SELECT * FROM (VALUES(1, 2, 3)) as t(x, y, z);", dt);
      EXPECT_EQ(rows->rowCount(), size_t(1));
      const auto row = rows->getNextRow(false, false);
      EXPECT_EQ(1, v<int64_t>(row[0]));
      EXPECT_EQ(2, v<int64_t>(row[1]));
      EXPECT_EQ(3, v<int64_t>(row[2]));
    }
    {
      auto rows = run_multiple_agg(
          "SELECT x, COUNT(y) FROM (VALUES(1, 1), (2, 2), (NULL, NULL), (3, 3)) as t(x, "
          "y) GROUP BY x;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(4));
      {
        const auto row = rows->getNextRow(false, false);
        EXPECT_EQ(1, v<int64_t>(row[0]));
        EXPECT_EQ(1, v<int64_t>(row[1]));
      }
      {
        const auto row = rows->getNextRow(false, false);
        EXPECT_EQ(2, v<int64_t>(row[0]));
        EXPECT_EQ(1, v<int64_t>(row[1]));
      }
      {
        const auto row = rows->getNextRow(false, false);
        EXPECT_EQ(3, v<int64_t>(row[0]));
        EXPECT_EQ(1, v<int64_t>(row[1]));
      }
      {
        const auto row = rows->getNextRow(false, false);
        EXPECT_EQ(inline_int_null_val(SQLTypeInfo(kINT, false)), v<int64_t>(row[0]));
        EXPECT_EQ(0, v<int64_t>(row[1]));
      }
    }
    {
      auto rows = run_multiple_agg(
          "SELECT SUM(x), AVG(y), MIN(z) FROM (VALUES(1, 2, 3)) as t(x, y, z);", dt);
      EXPECT_EQ(rows->rowCount(), size_t(1));
      const auto row = rows->getNextRow(false, false);
      EXPECT_EQ(1, v<int64_t>(row[0]));
      EXPECT_EQ(2, v<double>(row[1]));
      EXPECT_EQ(3, v<int64_t>(row[2]));
    }
    {
      auto rows = run_multiple_agg("SELECT * FROM (VALUES(1, 2, 3),(4, 5, 6));", dt);
      EXPECT_EQ(rows->rowCount(), size_t(2));
      {
        const auto row = rows->getNextRow(false, false);
        EXPECT_EQ(1, v<int64_t>(row[0]));
        EXPECT_EQ(2, v<int64_t>(row[1]));
        EXPECT_EQ(3, v<int64_t>(row[2]));
      }
      {
        const auto row = rows->getNextRow(false, false);
        EXPECT_EQ(4, v<int64_t>(row[0]));
        EXPECT_EQ(5, v<int64_t>(row[1]));
        EXPECT_EQ(6, v<int64_t>(row[2]));
      }
    }
    {
      auto rows = run_multiple_agg(
          "SELECT SUM(x), AVG(y), MIN(z) FROM (VALUES(1, 2, 3),(4, 5, 6)) as t(x, y, z);",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(1));
      const auto row = rows->getNextRow(false, false);
      EXPECT_EQ(5, v<int64_t>(row[0]));
      ASSERT_NEAR(3.5, v<double>(row[1]), double(0.01));
      EXPECT_EQ(3, v<int64_t>(row[2]));
    }
    EXPECT_ANY_THROW(run_simple_agg("SELECT * FROM (VALUES(1, 'test'));", dt));

    EXPECT_ANY_THROW(run_simple_agg("SELECT (1,2);", dt));

    {
      auto eo = getExecutionOptions(false, true);
      auto co = getCompilationOptions(dt);
      co.hoist_literals = true;
      const auto query_explain_result = runSqlQuery("SELECT 1+2;", co, eo);
      const auto explain_result = query_explain_result.getRows();
      EXPECT_EQ(size_t(1), explain_result->rowCount());
      const auto crt_row = explain_result->getNextRow(true, true);
      EXPECT_EQ(size_t(1), crt_row.size());
      const auto explain_str = boost::get<std::string>(v<NullableString>(crt_row[0]));
      EXPECT_TRUE(explain_str.find("IR for the ") == 0);
    }
  }
}

TEST_F(Select, UnsupportedNodes) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // MAT No longer throws a logicalValues gets a regular parse error'
    // EXPECT_THROW(run_multiple_agg("SELECT *;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT x, COUNT(*) FROM test GROUP BY ROLLUP(x);", dt),
                 std::runtime_error);
  }
}

TEST_F(Select, UnsupportedMultipleArgAggregate) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct x, y) FROM test;", dt),
                 std::runtime_error);
  }
}

TEST_F(Select, ArrayUnnest) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    unsigned power10 = 1;
    for (const unsigned int_width : {16, 32, 64}) {
      auto result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr_i" + std::to_string(int_width) +
                               ") AS a FROM array_test GROUP BY a ORDER BY a DESC;",
                           dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows->rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count + 2) * power10,
                v<int64_t>(result_rows->getRowAt(0, 1, true)));
      ASSERT_EQ(1,
                v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 0, true)));
      ASSERT_EQ(power10,
                v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 1, true)));

      auto fixed_result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr3_i" + std::to_string(int_width) +
                               ") AS a FROM array_test GROUP BY a ORDER BY a DESC;",
                           dt);
      ASSERT_EQ(g_array_test_row_count + 2, fixed_result_rows->rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count + 2) * power10,
                v<int64_t>(fixed_result_rows->getRowAt(0, 1, true)));
      ASSERT_EQ(
          1,
          v<int64_t>(fixed_result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(fixed_result_rows->getRowAt(0, 0, true)));
      ASSERT_EQ(
          power10,
          v<int64_t>(fixed_result_rows->getRowAt(g_array_test_row_count + 1, 1, true)));

      power10 *= 10;
    }
    for (const std::string float_type : {"float", "double"}) {
      auto result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr_" + float_type +
                               ") AS a FROM array_test GROUP BY a ORDER BY a DESC;",
                           dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows->rowCount());
      ASSERT_EQ(1,
                v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 0, true)));

      auto fixed_result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr3_" + float_type +
                               ") AS a FROM array_test GROUP BY a ORDER BY a DESC;",
                           dt);
      ASSERT_EQ(g_array_test_row_count + 2, fixed_result_rows->rowCount());
      ASSERT_EQ(
          1,
          v<int64_t>(fixed_result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(fixed_result_rows->getRowAt(0, 0, true)));
    }
    {
      auto result_rows = run_multiple_agg(
          "SELECT COUNT(*), UNNEST(arr_str) AS a FROM array_test GROUP BY a ORDER BY a "
          "DESC;",
          dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows->rowCount());
      ASSERT_EQ(1,
                v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 0, true)));
    }
    {
      auto result_rows = run_multiple_agg(
          "SELECT COUNT(*), UNNEST(arr_bool) AS a FROM array_test GROUP BY a ORDER BY a "
          "DESC;",
          dt);
      ASSERT_EQ(size_t(2), result_rows->rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count * 3),
                v<int64_t>(result_rows->getRowAt(0, 0, true)));
      ASSERT_EQ(int64_t(g_array_test_row_count * 3),
                v<int64_t>(result_rows->getRowAt(1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 1, true)));
      ASSERT_EQ(0, v<int64_t>(result_rows->getRowAt(1, 1, true)));

      auto fixed_result_rows = run_multiple_agg(
          "SELECT COUNT(*), UNNEST(arr6_bool) AS a FROM array_test GROUP BY a ORDER BY a "
          "DESC;",
          dt);
      ASSERT_EQ(size_t(2), fixed_result_rows->rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count * 3),
                v<int64_t>(fixed_result_rows->getRowAt(0, 0, true)));
      ASSERT_EQ(int64_t(g_array_test_row_count * 3),
                v<int64_t>(fixed_result_rows->getRowAt(1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(fixed_result_rows->getRowAt(0, 1, true)));
      ASSERT_EQ(0, v<int64_t>(fixed_result_rows->getRowAt(1, 1, true)));
    }

    // unnest groupby, force estimator run
    const auto big_group_threshold = g_big_group_threshold;
    ScopeGuard reset_big_group_threshold = [&big_group_threshold] {
      // this sets the "has estimation" parameter to false for baseline hash groupby of
      // small tables, forcing the estimator to run
      g_big_group_threshold = big_group_threshold;
    };
    g_big_group_threshold = 1;

    EXPECT_EQ(
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM (SELECT  unnest(arr_str), unnest(arr_float) FROM array_test GROUP BY 1, 2);)",
            dt)),
        int64_t(104));
  }
}

TEST_F(Select, ArrayIndex) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
      ASSERT_EQ(1,
                v<int64_t>(run_simple_agg(
                    "SELECT COUNT(*) FROM array_test WHERE arr_i32[2] = " +
                        std::to_string(10 * (row_idx + 2)) +
                        " AND x = " + std::to_string(7 + row_idx) +
                        " AND arr3_i32[2] = " + std::to_string(10 * (row_idx + 2)) +
                        " AND real_str LIKE 'real_str" + std::to_string(row_idx) + "';",
                    dt)));
      ASSERT_EQ(0,
                v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE "
                                          "arr_i32[4] > 0 OR arr_i32[4] <= 0 OR "
                                          "arr3_i32[4] > 0 OR arr3_i32[4] <= 0;",
                                          dt)));
      ASSERT_EQ(0,
                v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE "
                                          "arr_i32[0] > 0 OR arr_i32[0] <= 0 OR "
                                          "arr3_i32[0] > 0 OR arr3_i32[0] <= 0;",
                                          dt)));
    }
    for (size_t i = 1; i <= 6; ++i) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count / 2),
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_bool[" +
                                        std::to_string(i) +
                                        "] AND "
                                        "arr6_bool[" +
                                        std::to_string(i) + "];",
                                    dt)));
    }
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE arr_bool[7];", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE arr6_bool[7];", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE arr_bool[0];", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE arr6_bool[0];", dt)));
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE NOT "
                                        "(arr_i16[7] > 0 AND arr_i16[7] <= 0 AND "
                                        "arr3_i16[7] > 0 AND arr3_i16[7] <= 0);",
                                        dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE NOT "
                                        "(arr_i16[2] > 0 AND arr_i16[2] <= 0 AND "
                                        "arr3_i16[2] > 0 AND arr3_i16[2] <= 0);",
                                        dt)));
  }
}

TEST_F(Select, ArrayCountDistinct) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (const unsigned int_width : {16, 32, 64}) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count + 2),
          v<int64_t>(run_simple_agg("SELECT COUNT(distinct arr_i" +
                                        std::to_string(int_width) + ") FROM array_test;",
                                    dt)));
      auto result_rows =
          run_multiple_agg("SELECT COUNT(distinct arr_i" + std::to_string(int_width) +
                               ") FROM array_test GROUP BY x;",
                           dt);
      ASSERT_EQ(g_array_test_row_count, result_rows->rowCount());
      for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
        ASSERT_EQ(3, v<int64_t>(result_rows->getRowAt(row_idx, 0, true)));
      }

      ASSERT_EQ(
          int64_t(g_array_test_row_count + 2),
          v<int64_t>(run_simple_agg("SELECT COUNT(distinct arr3_i" +
                                        std::to_string(int_width) + ") FROM array_test;",
                                    dt)));
      auto fixed_result_rows =
          run_multiple_agg("SELECT COUNT(distinct arr3_i" + std::to_string(int_width) +
                               ") FROM array_test GROUP BY x;",
                           dt);
      ASSERT_EQ(g_array_test_row_count, fixed_result_rows->rowCount());
      for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
        ASSERT_EQ(3, v<int64_t>(fixed_result_rows->getRowAt(row_idx, 0, true)));
      }
    }
    for (const std::string float_type : {"float", "double"}) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count + 2),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(distinct arr_" + float_type + ") FROM array_test;", dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count + 2),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(distinct arr3_" + float_type + ") FROM array_test;", dt)));
    }
    ASSERT_EQ(int64_t(g_array_test_row_count + 2),
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(distinct arr_str) FROM array_test;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(distinct arr_bool) FROM array_test;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(distinct arr6_bool) FROM array_test;", dt)));
  }
}

TEST_F(Select, ArrayAnyAndAll) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    unsigned power10 = 1;
    for (const unsigned int_width : {16, 32, 64}) {
      ASSERT_EQ(
          2,
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE " +
                                        std::to_string(2 * power10) + " = ANY arr_i" +
                                        std::to_string(int_width) + " AND " +
                                        std::to_string(2 * power10) + " = ANY arr3_i" +
                                        std::to_string(int_width) + ";",
                                    dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count) - 2,
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE " +
                                        std::to_string(2 * power10) + " < ALL arr_i" +
                                        std::to_string(int_width) + " AND " +
                                        std::to_string(2 * power10) + " < ALL arr3_i" +
                                        std::to_string(int_width) + ";",
                                    dt)));
      power10 *= 10;
    }
    for (const std::string float_type : {"float", "double", "decimal"}) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(*) FROM array_test WHERE 1 < ANY arr_" + float_type + ";",
              dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(*) FROM array_test WHERE 2 < ANY arr_" + float_type + ";",
              dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(*) FROM array_test WHERE 0 < ALL arr_" + float_type + ";",
              dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(*) FROM array_test WHERE 0 < ALL arr3_" + float_type + ";",
              dt)));
    }
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE x - 5 = ANY arr_i16;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE x - 5 = ANY arr3_i16;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'aa' = ANY arr_str;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'bb' = ANY arr_str;", dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM array_test WHERE CAST('t' AS boolean) = ANY arr_bool;",
            dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM array_test WHERE CAST('t' AS boolean) = ANY arr6_bool;",
            dt)));
    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM array_test WHERE CAST('t' AS boolean) = ALL arr_bool;",
            dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count - 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'bb' < ALL arr_str;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count - 1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'bb' <= ALL arr_str;", dt)));
    ASSERT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'bb' > ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'bb' >= ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE  real_str = ANY arr_str;", dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM array_test WHERE  real_str <> ANY arr_str;", dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count - 1),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM array_test WHERE (NOT ('aa' = ANY arr_str));", dt)));
    // these two test just confirm that the regex does not mess with other similar
    // patterns
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) as SMALL FROM array_test;", dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) as COMPANY FROM array_test;", dt)));
  }
}

TEST_F(Select, ArrayUnsupported) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT MIN(arr_i64) FROM array_test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT MIN(arr3_i64) FROM array_test;", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg(
            "SELECT UNNEST(arr_str), COUNT(*) cc FROM array_test GROUP BY arr_str;", dt),
        std::runtime_error);
  }
}

int main(int argc, char** argv) {
  g_is_test_env = true;

  std::cout << "Starting ExecuteTest" << std::endl;

  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all test");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()("disable-literal-hoisting", "Disable literal hoisting");
  desc.add_options()("from-table-reordering",
                     po::value<bool>(&g_from_table_reordering)
                         ->default_value(g_from_table_reordering)
                         ->implicit_value(true),
                     "Enable automatic table reordering in FROM clause");
  desc.add_options()("bigint-count",
                     po::value<bool>(&g_bigint_count)
                         ->default_value(g_bigint_count)
                         ->implicit_value(false),
                     "Use 64-bit count");
  desc.add_options()("disable-shared-mem-group-by",
                     po::value<bool>(&g_enable_smem_group_by)
                         ->default_value(g_enable_smem_group_by)
                         ->implicit_value(false),
                     "Enable/disable using GPU shared memory for GROUP BY.");
  desc.add_options()("enable-columnar-output",
                     po::value<bool>(&g_enable_columnar_output)
                         ->default_value(g_enable_columnar_output)
                         ->implicit_value(true),
                     "Enable/disable using columnar output format.");
  desc.add_options()("enable-bump-allocator",
                     po::value<bool>(&g_enable_bump_allocator)
                         ->default_value(g_enable_bump_allocator)
                         ->implicit_value(true),
                     "Enable the bump allocator for projection queries on GPU.");
  desc.add_options()("dump-ir",
                     po::value<bool>()->default_value(false)->implicit_value(true),
                     "Dump IR and PTX for all executed queries to file."
                     " Currently only supports single node tests.");
  desc.add_options()("use-disk-cache",
                     "Use the disk cache for all tables with minimum size settings.");

  desc.add_options()(
      "test-help",
      "Print all ExecuteTest specific options (for gtest options use `--help`).");

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.set_options();  // update default values
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: ExecuteTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  if (vm["dump-ir"].as<bool>()) {
    // Only log IR, PTX channels to file with no rotation size.
    log_options.channels_ = {logger::Channel::IR, logger::Channel::PTX};
    log_options.rotation_size_ = std::numeric_limits<size_t>::max();
  }

  logger::init(log_options);

  if (vm.count("disable-literal-hoisting")) {
    g_hoist_literals = false;
  }

  g_enable_window_functions = true;
  g_enable_interop = false;

  File_Namespace::DiskCacheConfig disk_cache_config{};
  if (vm.count("use-disk-cache")) {
    disk_cache_config = File_Namespace::DiskCacheConfig{
        File_Namespace::DiskCacheConfig::getDefaultPath(std::string(BASE_PATH)),
        File_Namespace::DiskCacheLevel::all};
  }

  ExecuteTestBase::init();

  int err{0};
  try {
    ExecuteTestBase::createAndPopulateTestTables();
    if (!err) {
      err = RUN_ALL_TESTS();
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  Executor::nukeCacheOfExecutors();
  ResultSetReductionJIT::clearCache();
  ExecuteTestBase::reset();

  ExecuteTestBase::printStats();

  return err;
}
