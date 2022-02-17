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
  }

  static void reset() {
    data_mgr_.reset();
    storage_.reset();
    executor_.reset();
    calcite_.reset();
  }

 protected:
  void createTable(
      const std::string& table_name,
      const std::vector<ArrowStorage::ColumnDescription>& columns,
      const ArrowStorage::TableOptions& options = ArrowStorage::TableOptions()) {
    storage_->createTable(table_name, columns, options);
  }

  void dropTable(const std::string& table_name) { storage_->dropTable(table_name); }

  void insertValues(const std::string& table_name, const std::string& values) {
    ArrowStorage::CsvParseOptions parse_options;
    parse_options.header = false;
    storage_->appendCsvData(values, table_name, parse_options);
  }

  ExecutionResult runSqlQuery(const std::string& sql,
                              ExecutorDeviceType device_type,
                              bool allow_loop_joins) {
    auto schema_json = schema_to_json(storage_);
    const auto query_ra =
        calcite_->process("admin", "test_db", pg_shim(sql), schema_json).plan_result;
    auto dag =
        std::make_unique<RelAlgDagBuilder>(query_ra, TEST_DB_ID, storage_, nullptr);
    auto ra_executor =
        RelAlgExecutor(executor_.get(), TEST_DB_ID, storage_, std::move(dag));

    return ra_executor.executeRelAlgQuery(getCompilationOptions(device_type),
                                          getExecutionOptions(allow_loop_joins),
                                          false,
                                          nullptr);
  }

  ExecutionOptions getExecutionOptions(bool allow_loop_joins, bool just_explain = false) {
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

  CompilationOptions getCompilationOptions(ExecutorDeviceType device_type) {
    auto co = CompilationOptions::defaults(device_type);
    co.hoist_literals = g_hoist_literals;
    return co;
  }

  std::shared_ptr<ResultSet> run_multiple_agg(const string& query_str,
                                              const ExecutorDeviceType device_type,
                                              const bool allow_loop_joins = true) {
    return runSqlQuery(query_str, device_type, allow_loop_joins).getRows();
  }

  TargetValue run_simple_agg(const string& query_str,
                             const ExecutorDeviceType device_type,
                             const bool allow_loop_joins = true) {
    auto rows = run_multiple_agg(query_str, device_type, allow_loop_joins);
    auto crt_row = rows->getNextRow(true, true);
    CHECK_EQ(size_t(1), crt_row.size()) << query_str;
    return crt_row[0];
  }

  void c(const std::string& query_string, const ExecutorDeviceType device_type) {
    sqlite_comparator_.compare(
        run_multiple_agg(query_string, device_type), query_string, device_type);
  }

  void c(const std::string& query_string,
         const std::string& sqlite_query_string,
         const ExecutorDeviceType device_type) {
    sqlite_comparator_.compare(
        run_multiple_agg(query_string, device_type), sqlite_query_string, device_type);
  }

  /* timestamp approximate checking for NOW() */
  void cta(const std::string& query_string, const ExecutorDeviceType device_type) {
    sqlite_comparator_.compare_timstamp_approx(
        run_multiple_agg(query_string, device_type), query_string, device_type);
  }

  void c_arrow(const std::string& query_string, const ExecutorDeviceType device_type) {
    auto results = run_multiple_agg(query_string, device_type);
    auto arrow_omnisci_results = result_set_arrow_loopback(nullptr, results, device_type);
    sqlite_comparator_.compare_arrow_output(
        arrow_omnisci_results, query_string, device_type);
  }

  static std::shared_ptr<DataMgr> data_mgr_;
  static std::shared_ptr<ArrowStorage> storage_;
  static std::shared_ptr<Executor> executor_;
  static std::shared_ptr<Calcite> calcite_;
  static SQLiteComparator sqlite_comparator_;
};

std::shared_ptr<DataMgr> ExecuteTestBase::data_mgr_;
std::shared_ptr<ArrowStorage> ExecuteTestBase::storage_;
std::shared_ptr<Executor> ExecuteTestBase::executor_;
std::shared_ptr<Calcite> ExecuteTestBase::calcite_;
SQLiteComparator ExecuteTestBase::sqlite_comparator_;

class Distributed50 : public ExecuteTestBase, public ::testing::Test {};

TEST_F(Distributed50, FailOver) {
  createTable("dist5", {{"col1", SQLTypeInfo(kTEXT, false, kENCODING_DICT)}});

  auto dt = ExecutorDeviceType::CPU;

  EXPECT_NO_THROW(insertValues("dist5", "t1"));
  ASSERT_EQ(1, v<int64_t>(run_simple_agg("SELECT count(*) FROM dist5;", dt)));

  EXPECT_NO_THROW(insertValues("dist5", "t2"));
  ASSERT_EQ(2, v<int64_t>(run_simple_agg("SELECT count(*) FROM dist5;", dt)));

  EXPECT_NO_THROW(insertValues("dist5", "t3"));
  ASSERT_EQ(3, v<int64_t>(run_simple_agg("SELECT count(*) FROM dist5;", dt)));

  dropTable("dist5");
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
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  Executor::nukeCacheOfExecutors();
  ResultSetReductionJIT::clearCache();
  ExecuteTestBase::reset();

  return err;
}
