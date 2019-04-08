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

#include <csignal>
#include <cstdlib>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include "boost/filesystem.hpp"

#include "Catalog/Catalog.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Import/Importer.h"
#include "Parser/parser.h"
#include "QueryEngine/ResultSet.h"
#include "QueryEngine/TableOptimizer.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/MapDParameters.h"
#include "Shared/UpdelRoll.h"
#include "Shared/measure.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#define CALCITEPORT 6279

using namespace Catalog_Namespace;

namespace {
struct UpdelTestConfig {
  static bool showMeasuredTime;
  static bool enableVarUpdelPerfTest;
  static bool enableFixUpdelPerfTest;
  static int64_t fixNumRows;
  static int64_t varNumRows;
  static std::string fixFile;
  static std::string varFile;
  static std::string sequence;
};

constexpr static int varNumRowsByDefault = 8;
constexpr static int fixNumRowsByDefault = 100;

bool UpdelTestConfig::showMeasuredTime = false;
bool UpdelTestConfig::enableVarUpdelPerfTest = false;
bool UpdelTestConfig::enableFixUpdelPerfTest = false;
int64_t UpdelTestConfig::fixNumRows = fixNumRowsByDefault;
int64_t UpdelTestConfig::varNumRows = varNumRowsByDefault;
std::string UpdelTestConfig::fixFile = "trip_data_b.txt";
std::string UpdelTestConfig::varFile = "varlen.txt";
std::string UpdelTestConfig::sequence = "rate_code_id";
}  // namespace

namespace {
struct ScalarTargetValueExtractor : public boost::static_visitor<std::string> {
  result_type operator()(void*) const { return std::string("null"); }
  result_type operator()(std::string const& rhs) const { return rhs; }
  template <typename T>
  result_type operator()(T const& rhs) const {
    return std::to_string(rhs);
  }
  template <typename... VARIANT_ARGS>
  result_type operator()(boost::variant<VARIANT_ARGS...> const& rhs) const {
    return boost::apply_visitor(ScalarTargetValueExtractor(), rhs);
  }
};
}  // namespace
// namespace
namespace {

std::unique_ptr<SessionInfo> gsession;
bool g_hoist_literals{true};

std::shared_ptr<Calcite> g_calcite = nullptr;

void calcite_shutdown_handler() {
  if (g_calcite) {
    g_calcite->close_calcite_server();
  }
}

void mapd_signal_handler(int signal_number) {
  LOG(ERROR) << "Interrupt signal (" << signal_number << ") received.";
  calcite_shutdown_handler();
  // shut down logging force a flush
  google::ShutdownGoogleLogging();
  // terminate program
  if (signal_number == SIGTERM) {
    std::exit(EXIT_SUCCESS);
  } else {
    std::exit(signal_number);
  }
}

void register_signal_handler() {
  std::signal(SIGTERM, mapd_signal_handler);
  std::signal(SIGSEGV, mapd_signal_handler);
  std::signal(SIGABRT, mapd_signal_handler);
}

inline void run_ddl_statement(const std::string& input_str) {
  QueryRunner::run_ddl_statement(input_str, gsession);
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

std::shared_ptr<ResultSet> run_query(const std::string& query_str) {
  return QueryRunner::run_multiple_agg(
      query_str, gsession, ExecutorDeviceType::CPU, g_hoist_literals, true);
}

bool compare_agg(const std::string& table,
                 const std::string& column,
                 const int64_t cnt,
                 const double avg) {
  std::string query_str = "SELECT COUNT(*), AVG(" + column + ") FROM " + table + ";";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(2), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  auto r_avg = v<double>(crt_row[1]);
  // VLOG(1) << "r_avg: " << std::to_string(r_avg) << ", avg: " << std::to_string(avg);
  return r_cnt == cnt && fabs(r_avg - avg) < 1E-6;
}

template <typename T>
void update_prepare_offsets_values(const int64_t cnt,
                                   const int step,
                                   const T val,
                                   std::vector<uint64_t>& fragOffsets,
                                   std::vector<ScalarTargetValue>& rhsValues) {
  for (int64_t i = 0; i < cnt; i += step) {
    fragOffsets.push_back(i);
  }
  rhsValues.push_back(ScalarTargetValue(val));
}

template <typename T>
void update_common(const std::string& table,
                   const std::string& column,
                   const int64_t cnt,
                   const int step,
                   const T& val,
                   const SQLTypeInfo& rhsType,
                   const bool commit = true) {
  UpdelRoll updelRoll;
  std::vector<uint64_t> fragOffsets;
  std::vector<ScalarTargetValue> rhsValues;
  update_prepare_offsets_values<T>(cnt, step, val, fragOffsets, rhsValues);
  Fragmenter_Namespace::InsertOrderFragmenter::updateColumn(
      &gsession->getCatalog(),
      table,
      column,
      0,  // 1st frag since we have only 100 rows
      fragOffsets,
      rhsValues,
      rhsType,
      Data_Namespace::MemoryLevel::CPU_LEVEL,
      updelRoll);
  if (commit) {
    updelRoll.commitUpdate();
  }
}

bool update_a_numeric_column(const std::string& table,
                             const std::string& column,
                             const int64_t cnt,
                             const int step,
                             const double val,
                             const double avg,
                             const bool commit = true,
                             const bool by_string = false) {
  if (by_string) {
    update_common<std::string>(
        table, column, cnt, step, std::to_string(val), SQLTypeInfo(), commit);
  } else {
    update_common<double>(table, column, cnt, step, val, SQLTypeInfo(), commit);
  }
  return compare_agg(table, column, cnt, avg);
}

template <typename T>
bool nullize_a_fixed_encoded_column(const std::string& table,
                                    const std::string& column,
                                    const int64_t cnt) {
  update_common<int64_t>(
      table, column, cnt, 1, inline_int_null_value<T>(), SQLTypeInfo());
  std::string query_str =
      "SELECT count() FROM " + table + " WHERE " + column + " IS NULL;";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  return r_cnt == cnt;
}

using ResultRow = std::vector<TargetValue>;
using ResultRows = std::vector<ResultRow>;

const ResultRows get_some_rows(const std::string& table, const int limit = 10) {
  ResultRows result_rows;
  if (!UpdelTestConfig::enableVarUpdelPerfTest) {
    std::string query_str =
        "SELECT * FROM " + table + (limit ? " LIMIT " + std::to_string(limit) : "") + ";";
    const auto rows = run_query(query_str);
    // subtle here is we can't simply return ResultSet because it won't survive
    // any rollback which wipes out all chunk data/metadata of the table, crashing
    // UpdelRoll destructor that needs to free dirty chunks. So we need to copy
    // the result rows out before the rollback ...
    const auto nrow = rows->rowCount();
    const auto ncol = rows->colCount();
    for (std::remove_const<decltype(nrow)>::type r = 0; r < nrow; ++r) {
      // translate string, or encoded strings won't be equal w/ boost ==
      const auto row = rows->getNextRow(true, false);
      std::vector<TargetValue> result_row;
      for (std::remove_const<decltype(ncol)>::type c = 0; c < ncol; ++c) {
        result_row.emplace_back(row[c]);
      }
      result_rows.emplace_back(result_row);
    }
  }
  return result_rows;
}

template <typename T>
TargetValue targetValue(std::true_type, const std::vector<T>& vals) {
  return std::vector<ScalarTargetValue>(vals.begin(), vals.end());
}

template <typename T>
TargetValue targetValue(std::false_type, const T& val) {
  return ScalarTargetValue(val);
}

const std::string dumpv(const TargetValue& tv) {
  std::ostringstream os;
  if (const auto svp = boost::get<ScalarTargetValue>(&tv)) {
    os << *svp;
  } else if (const auto svp = boost::get<std::vector<ScalarTargetValue>>(&tv)) {
    os << boost::algorithm::join(
        *svp | boost::adaptors::transformed(ScalarTargetValueExtractor()), ",");
  }
  return os.str();
}

inline bool is_equal(const TargetValue& lhs, const TargetValue& rhs) {
  CHECK(!(boost::get<GeoTargetValue>(&lhs) || boost::get<GeoTargetValue>(&rhs)));
  if (lhs.which() == rhs.which()) {
    const auto l = boost::get<ScalarTargetValue>(&lhs);
    const auto r = boost::get<ScalarTargetValue>(&rhs);
    return l && r ? *l == *r
                  : *boost::get<std::vector<ScalarTargetValue>>(&lhs) ==
                        *boost::get<std::vector<ScalarTargetValue>>(&rhs);
  } else {
    return false;
  }
}

bool compare_row(const std::string& table,
                 const std::string& column,
                 const ResultRow& oldr,
                 const ResultRow& newr,
                 const TargetValue& val,
                 const bool commit) {
  const auto cat = &gsession->getCatalog();
  const auto td = cat->getMetadataForTable(table);
  const auto cdl = cat->getAllColumnMetadataForTable(td->tableId, false, false, false);
  const auto cds = std::vector<const ColumnDescriptor*>(cdl.begin(), cdl.end());
  const auto noldc = oldr.size();
  const auto nnewc = newr.size();
  const auto is_geophy_included = cds.size() < noldc;
  CHECK_EQ(noldc, nnewc);
  for (std::remove_const<decltype(noldc)>::type i = 0, c = 0; c < noldc;
       c += is_geophy_included ? cds[i]->columnType.get_physical_cols() + 1 : 1, ++i) {
    if (!is_equal(newr[c], (commit && cds[i]->columnName == column ? val : oldr[c]))) {
      VLOG(1) << cds[i]->columnName << ": " << dumpv(newr[c]) << " - "
              << dumpv((commit && cds[i]->columnName == column ? val : oldr[c]));
      return false;
    }
  }
  return true;
}

bool update_a_encoded_string_column(const std::string& table,
                                    const std::string& column,
                                    const int64_t cnt,
                                    const int step,
                                    const std::string& val,
                                    const bool commit = true) {
  update_common<const std::string>(table, column, cnt, step, val, SQLTypeInfo(), commit);
  // count updated string
  std::string query_str =
      "SELECT count() FROM " + table + " WHERE " + column + " = '" + val + "';";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  return r_cnt == (commit ? cnt / step : 0);
}

#define update_a_datetime_column update_a_encoded_string_column

bool update_a_boolean_column(const std::string& table,
                             const std::string& column,
                             const int64_t cnt,
                             const int step,
                             const bool val,
                             const bool commit = true) {
  update_common<const std::string>(
      table, column, cnt, step, val ? "T" : "F", SQLTypeInfo(), commit);
  // count updated bools
  std::string query_str =
      "SELECT count() FROM " + table + " WHERE " + (val ? "" : " NOT ") + column + ";";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  return r_cnt == (commit ? cnt / step : 0);
}

template <typename T>
bool update_column_from_decimal(const std::string& table,
                                const std::string& column,
                                const int64_t cnt,
                                const int64_t rhsDecimal,
                                const SQLTypeInfo& rhsType,
                                const SQLTypeInfo& lhsType,
                                const double max_loss,
                                const bool commit = true) {
  update_common<int64_t>(table, column, cnt, 1, rhsDecimal, rhsType, commit);
  std::string query_str = "SELECT DISTINCT " + column + " FROM " + table + ";";
  auto rows = run_query(query_str);
  CHECK_EQ(size_t(1), rows->rowCount());
  auto crt_row = rows->getNextRow(true, false);  // no decimal_to_double convert
  auto l_decimal = v<T>(crt_row[0]);
  int64_t r_decimal = rhsDecimal;

  if (lhsType.is_decimal()) {
    l_decimal = convert_decimal_value_to_scale(l_decimal, lhsType, rhsType);
  } else {
    l_decimal *= pow(10, rhsType.get_scale());
  }

  auto r_loss = fabs(l_decimal - r_decimal) / r_decimal;
  if (!(r_loss <= max_loss)) {
    VLOG(1) << "l_decimal: " << l_decimal << ", r_decimal: " << r_decimal
            << ", r_loss: " << r_loss << ", max_loss: " << max_loss;
  }
  return r_loss <= max_loss;
}

void import_table_file(const std::string& table, const std::string& file) {
  std::string query_str = std::string("COPY " + table + " FROM '") +
                          "../../Tests/Import/datafiles/" + file +
                          "' WITH (header='true');";

  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  if (parser.parse(query_str, parse_trees, last_parsed)) {
    throw std::runtime_error("Failed to parse: " + query_str);
  }
  CHECK_EQ(parse_trees.size(), size_t(1));

  const auto& stmt = parse_trees.front();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
  if (!ddl) {
    throw std::runtime_error("Not a DDLStmt: " + query_str);
  }
  ddl->execute(*gsession);
}

bool prepare_table_for_delete(const std::string& table = "trips",
                              const std::string& column = UpdelTestConfig::sequence,
                              const int64_t cnt = UpdelTestConfig::fixNumRows) {
  UpdelRoll updelRoll;
  std::vector<uint64_t> fragOffsets;
  std::vector<ScalarTargetValue> rhsValues;
  for (int64_t i = 0; i < cnt; ++i) {
    fragOffsets.push_back(i);
    rhsValues.emplace_back(ScalarTargetValue(i));
  }
  auto ms = measure<>::execution([&]() {
    Fragmenter_Namespace::InsertOrderFragmenter::updateColumn(
        &gsession->getCatalog(),
        table,
        column,
        0,  // 1st frag since we have only 100 rows
        fragOffsets,
        rhsValues,
        SQLTypeInfo(kBIGINT, false),
        Data_Namespace::MemoryLevel::CPU_LEVEL,
        updelRoll);
  });
  if (UpdelTestConfig::showMeasuredTime) {
    VLOG(2) << "time on update " << cnt << " rows:" << ms << " ms";
  }
  ms = measure<>::execution([&]() { updelRoll.commitUpdate(); });
  if (UpdelTestConfig::showMeasuredTime) {
    VLOG(2) << "time on commit:" << ms << " ms";
  }
  return compare_agg(table, column, cnt, (0 + cnt - 1) * cnt / 2. / cnt);
}

bool check_row_count_with_string(const std::string& table,
                                 const std::string& column,
                                 const int64_t cnt,
                                 const std::string& val) {
  std::string query_str =
      "SELECT count() FROM " + table + " WHERE " + column + " = '" + val + "';";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  if (r_cnt == cnt) {
    return true;
  }
  VLOG(1) << "r_cnt: " << std::to_string(r_cnt) << ", cnt: " << std::to_string(cnt);
  return false;
}

bool delete_and_immediately_vacuum_rows(const std::string& table,
                                        const std::string& column,
                                        const std::string& deleted_column,
                                        const int64_t nall,
                                        const int dcnt,
                                        const int start,
                                        const int step) {
  // set a column to "traceable" values
  if (false == prepare_table_for_delete(table, column, nall)) {
    return false;
  }

  // pre calc expected count and sum of the traceable values
  UpdelRoll updelRoll;
  std::vector<uint64_t> fragOffsets;
  std::vector<ScalarTargetValue> rhsValues;
  rhsValues.emplace_back(int64_t{1});
  int64_t sum = 0, cnt = 0;
  for (int d = 0, i = start; d < dcnt; ++d, i += step) {
    fragOffsets.push_back(i);
    sum += i;
    cnt += 1;
  }

  // delete and vacuum rows supposedly immediately
  auto cat = &gsession->getCatalog();
  auto td = cat->getMetadataForTable(table);
  auto cd = cat->getMetadataForColumn(td->tableId, deleted_column);
  const_cast<ColumnDescriptor*>(cd)->isDeletedCol = true;
  auto ms = measure<>::execution([&]() {
    td->fragmenter->updateColumn(cat,
                                 td,
                                 cd,
                                 0,  // 1st frag since we have only 100 rows
                                 fragOffsets,
                                 rhsValues,
                                 SQLTypeInfo(kBOOLEAN, false),
                                 Data_Namespace::MemoryLevel::CPU_LEVEL,
                                 updelRoll);
  });
  if (UpdelTestConfig::showMeasuredTime) {
    VLOG(2) << "time on delete & vacuum " << dcnt << " of " << nall << " rows:" << ms
            << " ms";
  }
  ms = measure<>::execution([&]() { updelRoll.commitUpdate(); });
  if (UpdelTestConfig::showMeasuredTime) {
    VLOG(2) << "time on commit:" << ms << " ms";
  }
  cnt = nall - cnt;
  // check varlen column vacuumed
  if (false == check_row_count_with_string(
                   table, "hack_license", cnt, "BA96DE419E711691B9445D6A6307C170")) {
    return false;
  }
  return compare_agg(
      table, column, cnt, ((0 + nall - 1) * nall / 2. - sum) / (cnt ? cnt : 1));
}

bool delete_and_vacuum_varlen_rows(const std::string& table,
                                   const int nall,
                                   const int ndel,
                                   const int start,
                                   const int step,
                                   const bool manual_vacuum) {
  const auto old_rows = get_some_rows(table, nall);

  std::vector<uint64_t> fragOffsets;
  for (int d = 0, i = start; d < ndel && i < nall; ++d, i += step) {
    fragOffsets.push_back(i);
  }

  // delete and vacuum rows
  auto cond = std::string("rowid >= ") + std::to_string(start) + " and mod(rowid-" +
              std::to_string(start) + "," + std::to_string(step) + ")=0 and rowid < " +
              std::to_string(start) + "+" + std::to_string(step) + "*" +
              std::to_string(ndel);
  auto ms = measure<>::execution([&]() {
    ASSERT_NO_THROW(run_query("delete from " + table + " where " + cond + ";"););
  });
  if (UpdelTestConfig::showMeasuredTime) {
    VLOG(2) << "time on delete " << (manual_vacuum ? "" : "& vacuum ")
            << fragOffsets.size() << " rows:" << ms << " ms";
  }

  if (manual_vacuum) {
    ms = measure<>::execution([&]() {
      auto cat = &gsession->getCatalog();
      const auto td = cat->getMetadataForTable(table,
                                               /*populateFragmenter=*/true);
      auto executor = Executor::getExecutor(cat->getCurrentDB().dbId);
      TableOptimizer optimizer(td, executor.get(), *cat);
      optimizer.vacuumDeletedRows();
    });
    if (UpdelTestConfig::showMeasuredTime) {
      VLOG(2) << "time on vacuum:" << ms << " ms";
    }
  }

  const auto new_rows = get_some_rows(table, nall);
  if (!UpdelTestConfig::enableVarUpdelPerfTest) {
    for (int oi = 0, ni = 0; oi < nall; ++oi) {
      if (fragOffsets.end() == std::find(fragOffsets.begin(), fragOffsets.end(), oi)) {
        if (!compare_row(table,
                         "",
                         old_rows[oi],
                         new_rows[ni++],
                         ScalarTargetValue(int64_t{1}),
                         false)) {
          return false;
        }
      }
    }
  }
  return true;
}

class SQLTestEnv : public ::testing::Environment {
 public:
  virtual void SetUp() {
    boost::filesystem::path base_path{BASE_PATH};
    CHECK(boost::filesystem::exists(base_path));
    auto system_db_file = base_path / "mapd_catalogs" / MAPD_SYSTEM_DB;
    auto data_dir = base_path / "mapd_data";
    UserMetadata user;
    DBMetadata db;

    register_signal_handler();
    google::InstallFailureFunction(&calcite_shutdown_handler);

    g_calcite = std::make_shared<Calcite>(-1, CALCITEPORT, base_path.string(), 1024);
    MapDParameters mapd_parms;
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(
        data_dir.string(), mapd_parms, false, 0);
    // if no catalog create one
    auto& sys_cat = SysCatalog::instance();
    sys_cat.init(base_path.string(),
                 dataMgr,
                 {},
                 g_calcite,
                 !boost::filesystem::exists(system_db_file),
                 false,
                 mapd_parms.aggregator,
                 {});
    CHECK(sys_cat.getMetadataForUser(MAPD_ROOT_USER, user));
    // if no user create one
    if (!sys_cat.getMetadataForUser("gtest", user)) {
      sys_cat.createUser("gtest", "test!test!", false);
      CHECK(sys_cat.getMetadataForUser("gtest", user));
    }
    // if no db create one
    if (!sys_cat.getMetadataForDB("gtest_db", db)) {
      sys_cat.createDatabase("gtest_db", user.userId);
      CHECK(sys_cat.getMetadataForDB("gtest_db", db));
    }
    gsession.reset(new SessionInfo(std::make_shared<Catalog>(base_path.string(),
                                                             db,
                                                             dataMgr,
                                                             std::vector<LeafHostInfo>{},
                                                             g_calcite,
                                                             false),
                                   user,
                                   ExecutorDeviceType::GPU,
                                   ""));
  }
};

// don't use R"()" format; somehow it causes many blank lines
// to be output on console. how come?
const std::string create_varlen_table1 =
    "CREATE TABLE varlen("
    "ti tinyint,"
    "si smallint,"
    "ii int,"
    "bi bigint,"
    "ff float,"
    "fd double,"
    "de decimal(5,2),"
    "ts timestamp,"
    "ns text encoding none,"
    "es text encoding dict(16),"
    // move these fixed array ahead of geo columns to bypass issue#2008
    // for this unit test to continue independently :)
    "faii int[2],"
    "fadc decimal(5,2)[2],"
    "fatx text[2],";
const std::string create_varlen_table2 =
    "pt point,"
    "ls linestring,"
    "pg polygon,"
    "mp multipolygon,";
const std::string create_varlen_table3 =
    "ati tinyint[],"
    "asi smallint[],"
    "aii int[],"
    "abi bigint[],"
    "aff float[],"
    "afd double[],"
    "adc decimal(5,2)[],"
    "atx text[],"
    "ats timestamp[]"
    ")";

const std::string create_table_trips =
    "	CREATE TABLE trips ("
    "			medallion               TEXT ENCODING DICT,"
    "			hack_license            TEXT ENCODING DICT,"
    "			vendor_id               TEXT ENCODING DICT,"
    "			rate_code_id            SMALLINT ENCODING FIXED(8),"
    "			store_and_fwd_flag      TEXT ENCODING DICT,"
    "			pickup_datetime         TIMESTAMP,"
    "			dropoff_datetime        TIMESTAMP,"
    "			passenger_count         INTEGER ENCODING FIXED(16),"
    "			trip_time_in_secs       INTEGER,"
    "			trip_distance           FLOAT,"
    "			pickup_longitude        DECIMAL(14,7),"
    "			pickup_latitude         DECIMAL(14,7),"
    "			dropoff_longitude       DOUBLE,"
    "			dropoff_latitude        DECIMAL(18,5),"
    "			deleted                 BOOLEAN"
    "			)";

void init_table_data(const std::string& table = "trips",
                     const std::string& create_table_cmd = create_table_trips,
                     const std::string& file = UpdelTestConfig::fixFile) {
  run_ddl_statement("drop table if exists " + table + ";");
  run_ddl_statement(create_table_cmd + ";");
  if (file.size()) {
    auto ms = measure<>::execution([&]() { import_table_file(table, file); });
    if (UpdelTestConfig::showMeasuredTime) {
      VLOG(2) << "time on import: " << ms << " ms";
    }
  }
}

template <int N = 0>
class RowVacuumTestWithVarlenAndArraysN : public ::testing::Test {
 protected:
  virtual void SetUp() {
    auto create_varlen_table =
        create_varlen_table1 +
        (UpdelTestConfig::enableVarUpdelPerfTest ? "" : create_varlen_table2) +
        create_varlen_table3;
    // test >1 fragments?
    create_varlen_table +=
        "WITH (FRAGMENT_SIZE = " + std::to_string(N ? N : 32'000'000) + ")";
    ASSERT_NO_THROW(
        init_table_data("varlen", create_varlen_table, UpdelTestConfig::varFile););
    // immediate vacuum?
    Fragmenter_Namespace::FragmentInfo::setUnconditionalVacuum(N == 0);
  }

  virtual void TearDown() {
    Fragmenter_Namespace::FragmentInfo::setUnconditionalVacuum(false);
    ASSERT_NO_THROW(run_ddl_statement("drop table varlen;"););
  }
};

using ManualRowVacuumTestWithVarlenAndArrays =
    RowVacuumTestWithVarlenAndArraysN<varNumRowsByDefault / 2>;
TEST_F(ManualRowVacuumTestWithVarlenAndArrays, Vacuum_Half_First) {
  EXPECT_TRUE(delete_and_vacuum_varlen_rows("varlen",
                                            UpdelTestConfig::varNumRows,
                                            UpdelTestConfig::varNumRows / 2,
                                            0,
                                            1,
                                            true));
}
TEST_F(ManualRowVacuumTestWithVarlenAndArrays, Vacuum_Half_Second) {
  EXPECT_TRUE(delete_and_vacuum_varlen_rows("varlen",
                                            UpdelTestConfig::varNumRows,
                                            UpdelTestConfig::varNumRows / 2,
                                            UpdelTestConfig::varNumRows / 2,
                                            1,
                                            true));
}
TEST_F(ManualRowVacuumTestWithVarlenAndArrays, Vacuum_Interleaved_2) {
  EXPECT_TRUE(delete_and_vacuum_varlen_rows("varlen",
                                            UpdelTestConfig::varNumRows,
                                            UpdelTestConfig::varNumRows / 2,
                                            0,
                                            2,
                                            true));
}
TEST_F(ManualRowVacuumTestWithVarlenAndArrays, Vacuum_Interleaved_3) {
  EXPECT_TRUE(delete_and_vacuum_varlen_rows("varlen",
                                            UpdelTestConfig::varNumRows,
                                            UpdelTestConfig::varNumRows / 3,
                                            0,
                                            3,
                                            true));
}

// make class backward compatible
using RowVacuumTestWithVarlenAndArrays = RowVacuumTestWithVarlenAndArraysN<0>;
TEST_F(RowVacuumTestWithVarlenAndArrays, Vacuum_Half_First) {
  EXPECT_TRUE(delete_and_vacuum_varlen_rows("varlen",
                                            UpdelTestConfig::varNumRows,
                                            UpdelTestConfig::varNumRows / 2,
                                            0,
                                            1,
                                            false));
}
TEST_F(RowVacuumTestWithVarlenAndArrays, Vacuum_Half_Second) {
  EXPECT_TRUE(delete_and_vacuum_varlen_rows("varlen",
                                            UpdelTestConfig::varNumRows,
                                            UpdelTestConfig::varNumRows / 2,
                                            UpdelTestConfig::varNumRows / 2,
                                            1,
                                            false));
}
TEST_F(RowVacuumTestWithVarlenAndArrays, Vacuum_Interleaved_2) {
  EXPECT_TRUE(delete_and_vacuum_varlen_rows("varlen",
                                            UpdelTestConfig::varNumRows,
                                            UpdelTestConfig::varNumRows / 2,
                                            0,
                                            2,
                                            false));
}
TEST_F(RowVacuumTestWithVarlenAndArrays, Vacuum_Interleaved_3) {
  EXPECT_TRUE(delete_and_vacuum_varlen_rows("varlen",
                                            UpdelTestConfig::varNumRows,
                                            UpdelTestConfig::varNumRows / 3,
                                            0,
                                            3,
                                            false));
}

class RowVacuumTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    ASSERT_NO_THROW(init_table_data(););
    Fragmenter_Namespace::FragmentInfo::setUnconditionalVacuum(true);
  }

  virtual void TearDown() {
    Fragmenter_Namespace::FragmentInfo::setUnconditionalVacuum(false);
    ASSERT_NO_THROW(run_ddl_statement("drop table trips;"););
  }
};

TEST_F(RowVacuumTest, Vacuum_Half_First) {
  EXPECT_TRUE(delete_and_immediately_vacuum_rows("trips",
                                                 UpdelTestConfig::sequence,
                                                 "deleted",
                                                 UpdelTestConfig::fixNumRows,
                                                 UpdelTestConfig::fixNumRows / 2,
                                                 0,
                                                 1));
}
TEST_F(RowVacuumTest, Vacuum_Half_Second) {
  EXPECT_TRUE(delete_and_immediately_vacuum_rows("trips",
                                                 UpdelTestConfig::sequence,
                                                 "deleted",
                                                 UpdelTestConfig::fixNumRows,
                                                 UpdelTestConfig::fixNumRows / 2,
                                                 UpdelTestConfig::fixNumRows / 2,
                                                 1));
}
TEST_F(RowVacuumTest, Vacuum_Interleaved_2) {
  EXPECT_TRUE(delete_and_immediately_vacuum_rows("trips",
                                                 UpdelTestConfig::sequence,
                                                 "deleted",
                                                 UpdelTestConfig::fixNumRows,
                                                 UpdelTestConfig::fixNumRows / 2,
                                                 0,
                                                 2));
}
TEST_F(RowVacuumTest, Vacuum_Interleaved_4) {
  EXPECT_TRUE(delete_and_immediately_vacuum_rows("trips",
                                                 UpdelTestConfig::sequence,
                                                 "deleted",
                                                 UpdelTestConfig::fixNumRows,
                                                 UpdelTestConfig::fixNumRows / 4,
                                                 0,
                                                 4));
}

// It is currently not possible to do select query on temp table w/o
// MapDHandler. Perhaps in the future a thrift rpc like `push_table_details`
// can be added to enable such queries here...
#if 0
const char* create_temp_table = "CREATE TEMPORARY TABLE temp(i int) WITH (vacuum='delayed');";

class UpdateStorageTest_Temp : public ::testing::Test {
 protected:
  virtual void SetUp() {
    ASSERT_NO_THROW(init_table_data("temp", create_temp_table, ""););
    EXPECT_NO_THROW(run_query("insert into temp values(1)"););
    EXPECT_NO_THROW(run_query("insert into temp values(1)"););
  }

  virtual void TearDown() { ASSERT_NO_THROW(run_ddl_statement("drop table temp;");); }
};

TEST_F(UpdateStorageTest_Temp, Update_temp) {
  EXPECT_TRUE(update_a_numeric_column("temp", "i", 2, 1, 99, 99));
}

TEST_F(UpdateStorageTest_Temp, Update_temp_rollback) {
  EXPECT_TRUE(update_a_numeric_column("temp", "i", 2, 1, 99, 1, true));
}
#endif

class UpdateStorageTest : public ::testing::Test {
 protected:
  virtual void SetUp() { ASSERT_NO_THROW(init_table_data();); }

  virtual void TearDown() { ASSERT_NO_THROW(run_ddl_statement("drop table trips;");); }
};

TEST_F(UpdateStorageTest, All_fixed_encoded_smallint_rate_code_id_null_8) {
  EXPECT_TRUE(nullize_a_fixed_encoded_column<int8_t>(
      "trips", "rate_code_id", UpdelTestConfig::fixNumRows));
}
TEST_F(UpdateStorageTest, All_fixed_encoded_smallint_rate_code_id_null_16) {
  EXPECT_TRUE(nullize_a_fixed_encoded_column<int16_t>(
      "trips", "rate_code_id", UpdelTestConfig::fixNumRows));
}
TEST_F(UpdateStorageTest, All_fixed_encoded_smallint_rate_code_id_null_32_throw) {
  EXPECT_THROW(nullize_a_fixed_encoded_column<int32_t>(
                   "trips", "rate_code_id", UpdelTestConfig::fixNumRows),
               std::runtime_error);
}
TEST_F(UpdateStorageTest, All_fixed_encoded_smallint_rate_code_id_null_64_throw) {
  EXPECT_THROW(nullize_a_fixed_encoded_column<int64_t>(
                   "trips", "rate_code_id", UpdelTestConfig::fixNumRows),
               std::runtime_error);
}

TEST_F(UpdateStorageTest, All_fixed_encoded_integer_passenger_count_null_16) {
  EXPECT_TRUE(nullize_a_fixed_encoded_column<int16_t>(
      "trips", "passenger_count", UpdelTestConfig::fixNumRows));
}
TEST_F(UpdateStorageTest, All_fixed_encoded_integer_passenger_count_null_32) {
  EXPECT_TRUE(nullize_a_fixed_encoded_column<int32_t>(
      "trips", "passenger_count", UpdelTestConfig::fixNumRows));
}
TEST_F(UpdateStorageTest, All_fixed_encoded_integer_passenger_count_null_64_throw) {
  EXPECT_THROW(nullize_a_fixed_encoded_column<int64_t>(
                   "trips", "passenger_count", UpdelTestConfig::fixNumRows),
               std::runtime_error);
}

TEST_F(UpdateStorageTest, All_integer_trip_time_in_secs_null_32) {
  EXPECT_TRUE(nullize_a_fixed_encoded_column<int32_t>(
      "trips", "trip_time_in_secs", UpdelTestConfig::fixNumRows));
}
TEST_F(UpdateStorageTest, All_integer_trip_time_in_secs_null_64_throw) {
  EXPECT_THROW(nullize_a_fixed_encoded_column<int64_t>(
                   "trips", "trip_time_in_secs", UpdelTestConfig::fixNumRows),
               std::runtime_error);
}

TEST_F(UpdateStorageTest, All_fixed_encoded_smallint_rate_code_id_throw) {
  EXPECT_TRUE(update_a_numeric_column(
      "trips", "rate_code_id", UpdelTestConfig::fixNumRows, 1, 44 * 2, 44 * 2.0));
  EXPECT_THROW(update_a_numeric_column(
                   "trips", "rate_code_id", UpdelTestConfig::fixNumRows, 1, +257, +257),
               std::runtime_error);
  EXPECT_THROW(update_a_numeric_column(
                   "trips", "rate_code_id", UpdelTestConfig::fixNumRows, 1, -256, -256),
               std::runtime_error);
}

#define SQLTypeInfo_dropoff_latitude SQLTypeInfo(kDECIMAL, 19, 5, false)
TEST_F(UpdateStorageTest, All_RHS_decimal_10_0_LHS_decimal_19_5) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "dropoff_latitude",
                                                  UpdelTestConfig::fixNumRows,
                                                  1234506789,
                                                  SQLTypeInfo(kDECIMAL, 10, 0, false),
                                                  SQLTypeInfo_dropoff_latitude,
                                                  0));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_10_2_LHS_decimal_19_5) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "dropoff_latitude",
                                                  UpdelTestConfig::fixNumRows,
                                                  1234506789,
                                                  SQLTypeInfo(kDECIMAL, 10, 2, false),
                                                  SQLTypeInfo_dropoff_latitude,
                                                  0));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_15_2_LHS_decimal_19_5) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "dropoff_latitude",
                                                  UpdelTestConfig::fixNumRows,
                                                  123456789012345,
                                                  SQLTypeInfo(kDECIMAL, 15, 2, false),
                                                  SQLTypeInfo_dropoff_latitude,
                                                  0));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_17_2_LHS_decimal_19_5_throw) {
  EXPECT_THROW(update_column_from_decimal<int64_t>("trips",
                                                   "dropoff_latitude",
                                                   UpdelTestConfig::fixNumRows,
                                                   12345678901234567,
                                                   SQLTypeInfo(kDECIMAL, 17, 2, false),
                                                   SQLTypeInfo_dropoff_latitude,
                                                   0),
               std::runtime_error);
}

#define SQLTypeInfo_dropoff_longitude SQLTypeInfo(kDOUBLE, false)
TEST_F(UpdateStorageTest, All_RHS_decimal_10_0_LHS_double) {
  EXPECT_TRUE(update_column_from_decimal<double>("trips",
                                                 "dropoff_longitude",
                                                 UpdelTestConfig::fixNumRows,
                                                 1234506789,
                                                 SQLTypeInfo(kDECIMAL, 10, 0, false),
                                                 SQLTypeInfo_dropoff_longitude,
                                                 1E-15));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_10_2_LHS_double) {
  EXPECT_TRUE(update_column_from_decimal<double>("trips",
                                                 "dropoff_longitude",
                                                 UpdelTestConfig::fixNumRows,
                                                 1234506789,
                                                 SQLTypeInfo(kDECIMAL, 10, 2, false),
                                                 SQLTypeInfo_dropoff_longitude,
                                                 1E-15));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_15_2_LHS_double) {
  EXPECT_TRUE(update_column_from_decimal<double>("trips",
                                                 "dropoff_longitude",
                                                 UpdelTestConfig::fixNumRows,
                                                 123456789012345,
                                                 SQLTypeInfo(kDECIMAL, 15, 2, false),
                                                 SQLTypeInfo_dropoff_longitude,
                                                 1E-15));
}

#define SQLTypeInfo_trip_time_in_secs SQLTypeInfo(kINT, false)
TEST_F(UpdateStorageTest, All_RHS_decimal_10_0_LHS_integer) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "trip_time_in_secs",
                                                  UpdelTestConfig::fixNumRows,
                                                  1234506789,
                                                  SQLTypeInfo(kDECIMAL, 10, 0, false),
                                                  SQLTypeInfo_trip_time_in_secs,
                                                  1));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_10_2_LHS_integer) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "trip_time_in_secs",
                                                  UpdelTestConfig::fixNumRows,
                                                  1234506789,
                                                  SQLTypeInfo(kDECIMAL, 10, 2, false),
                                                  SQLTypeInfo_trip_time_in_secs,
                                                  1));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_15_5_LHS_integer) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "trip_time_in_secs",
                                                  UpdelTestConfig::fixNumRows,
                                                  123456789012345,
                                                  SQLTypeInfo(kDECIMAL, 15, 5, false),
                                                  SQLTypeInfo_trip_time_in_secs,
                                                  1));
}

TEST_F(UpdateStorageTest, All_fixed_encoded_integer_passenger_count_x1_throw) {
  EXPECT_TRUE(update_a_numeric_column(
      "trips", "passenger_count", UpdelTestConfig::fixNumRows, 1, 4 * 2, 4 * 2.0));
  EXPECT_THROW(
      update_a_numeric_column(
          "trips", "passenger_count", UpdelTestConfig::fixNumRows, 1, +65537, +65537),
      std::runtime_error);
  EXPECT_THROW(
      update_a_numeric_column(
          "trips", "passenger_count", UpdelTestConfig::fixNumRows, 1, -65536, -65536),
      std::runtime_error);
}
TEST_F(UpdateStorageTest, All_fixed_encoded_integer_passenger_count_x1_rollback) {
  EXPECT_TRUE(update_a_numeric_column(
      "trips", "passenger_count", UpdelTestConfig::fixNumRows, 1, 4 * 2, 4 * 1.0, false));
}

TEST_F(UpdateStorageTest, Half_fixed_encoded_integer_passenger_count_x2) {
  EXPECT_TRUE(update_a_numeric_column(
      "trips", "passenger_count", UpdelTestConfig::fixNumRows, 2, 4 * 2, 4. * 1.5));
}
TEST_F(UpdateStorageTest, Half_fixed_encoded_integer_passenger_count_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips",
                                      "passenger_count",
                                      UpdelTestConfig::fixNumRows,
                                      2,
                                      4 * 2,
                                      4. * 1.0,
                                      false));
}

TEST_F(UpdateStorageTest, All_int_trip_time_in_secs_x2) {
  EXPECT_TRUE(update_a_numeric_column(
      "trips", "trip_time_in_secs", UpdelTestConfig::fixNumRows, 1, 382 * 2, 382 * 2.0));
}
TEST_F(UpdateStorageTest, All_int_trip_time_in_secs_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips",
                                      "trip_time_in_secs",
                                      UpdelTestConfig::fixNumRows,
                                      1,
                                      382 * 2,
                                      382 * 1.0,
                                      false));
}

TEST_F(UpdateStorageTest, Half_int_trip_time_in_secs_x2) {
  EXPECT_TRUE(update_a_numeric_column(
      "trips", "trip_time_in_secs", UpdelTestConfig::fixNumRows, 2, 382 * 2, 382. * 1.5));
}
TEST_F(UpdateStorageTest, Half_int_trip_time_in_secs_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips",
                                      "trip_time_in_secs",
                                      UpdelTestConfig::fixNumRows,
                                      2,
                                      382 * 2,
                                      382. * 1.0,
                                      false));
}

TEST_F(UpdateStorageTest, All_float_trip_distance_x2) {
  EXPECT_TRUE(update_a_numeric_column(
      "trips", "trip_distance", UpdelTestConfig::fixNumRows, 1, 1 * 2, 1 * 2.0));
}
TEST_F(UpdateStorageTest, All_float_trip_distance_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column(
      "trips", "trip_distance", UpdelTestConfig::fixNumRows, 1, 1 * 2, 1 * 1.0, false));
}

TEST_F(UpdateStorageTest, Half_float_trip_distance_x2) {
  EXPECT_TRUE(update_a_numeric_column(
      "trips", "trip_distance", UpdelTestConfig::fixNumRows, 2, 1 * 2, 1. * 1.5));
}
TEST_F(UpdateStorageTest, Half_float_trip_distance_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column(
      "trips", "trip_distance", UpdelTestConfig::fixNumRows, 2, 1 * 2, 1. * 1.0, false));
}

TEST_F(UpdateStorageTest, All_decimal_pickup_longitude_x2) {
  EXPECT_TRUE(update_a_numeric_column("trips",
                                      "pickup_longitude",
                                      UpdelTestConfig::fixNumRows,
                                      1,
                                      -73.978165 * 2,
                                      -73.978165 * 2.0));
}
TEST_F(UpdateStorageTest, All_decimal_pickup_longitude_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips",
                                      "pickup_longitude",
                                      UpdelTestConfig::fixNumRows,
                                      1,
                                      -73.978165 * 2,
                                      -73.978165 * 1.0,
                                      false));
}

TEST_F(UpdateStorageTest, Half_decimal_pickup_longitude_x2) {
  EXPECT_TRUE(update_a_numeric_column("trips",
                                      "pickup_longitude",
                                      UpdelTestConfig::fixNumRows,
                                      2,
                                      -73.978165 * 2,
                                      -73.978165 * 1.5));
}
TEST_F(UpdateStorageTest, Half_decimal_pickup_longitude_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips",
                                      "pickup_longitude",
                                      UpdelTestConfig::fixNumRows,
                                      2,
                                      -73.978165 * 2,
                                      -73.978165 * 1.0,
                                      false));
}
TEST_F(UpdateStorageTest, Half_decimal_pickup_longitude_x2_by_string) {
  EXPECT_TRUE(update_a_numeric_column("trips",
                                      "pickup_longitude",
                                      UpdelTestConfig::fixNumRows,
                                      2,
                                      -73.978165 * 2,
                                      -73.978165 * 1.5,
                                      true,
                                      true));
}
TEST_F(UpdateStorageTest, Half_decimal_pickup_longitude_x2_rollback_by_string) {
  EXPECT_TRUE(update_a_numeric_column("trips",
                                      "pickup_longitude",
                                      UpdelTestConfig::fixNumRows,
                                      2,
                                      -73.978165 * 2,
                                      -73.978165 * 1.0,
                                      false,
                                      true));
}

TEST_F(UpdateStorageTest, All_string_vendor_id) {
  EXPECT_TRUE(update_a_encoded_string_column(
      "trips", "vendor_id", UpdelTestConfig::fixNumRows, 1, "abcxyz"));
}
TEST_F(UpdateStorageTest, All_string_vendor_id_rollback) {
  EXPECT_TRUE(update_a_encoded_string_column(
      "trips", "vendor_id", UpdelTestConfig::fixNumRows, 1, "abcxyz", false));
}

TEST_F(UpdateStorageTest, Half_string_vendor_id) {
  EXPECT_TRUE(update_a_encoded_string_column(
      "trips", "vendor_id", UpdelTestConfig::fixNumRows, 2, "xyzabc"));
}
TEST_F(UpdateStorageTest, Half_string_vendor_id_rollback) {
  EXPECT_TRUE(update_a_encoded_string_column(
      "trips", "vendor_id", UpdelTestConfig::fixNumRows, 2, "xyzabc", false));
}

TEST_F(UpdateStorageTest, All_boolean_deleted) {
  EXPECT_TRUE(
      update_a_boolean_column("trips", "deleted", UpdelTestConfig::fixNumRows, 1, true));
}
TEST_F(UpdateStorageTest, All_boolean_deleted_rollback) {
  EXPECT_TRUE(update_a_boolean_column(
      "trips", "deleted", UpdelTestConfig::fixNumRows, 1, true, false));
}

TEST_F(UpdateStorageTest, Half_boolean_deleted) {
  EXPECT_TRUE(
      update_a_boolean_column("trips", "deleted", UpdelTestConfig::fixNumRows, 2, true));
}
TEST_F(UpdateStorageTest, Half_boolean_deleted_rollback) {
  EXPECT_TRUE(update_a_boolean_column(
      "trips", "deleted", UpdelTestConfig::fixNumRows, 2, true, false));
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SQLTestEnv);

  // the data files for perf tests are too big to check in, so perf tests
  // are done privately in someone's dev host. prog option seems a overkill.
  auto var_file = getenv("updel_var_file");  // used to be "varlen.big"
  auto var_nrow = getenv("updel_var_nrow");  // used to be 1'000'000
  auto fix_file = getenv("updel_fix_file");  // used to be S3 file "trip_data_32m.tgz"
  auto fix_nrow = getenv("updel_fix_nrow");  // used to be 32'000'000
  if (var_file && *var_file && var_nrow && *var_nrow) {
    UpdelTestConfig::enableVarUpdelPerfTest = true;
    UpdelTestConfig::varFile = var_file;
    UpdelTestConfig::varNumRows = atoi(var_nrow);
  }
  if (fix_file && *fix_file && fix_nrow && *fix_nrow) {
    UpdelTestConfig::enableFixUpdelPerfTest = true;
    UpdelTestConfig::fixFile = fix_file;
    UpdelTestConfig::fixNumRows = atoi(fix_nrow);
  }
  UpdelTestConfig::showMeasuredTime =
      UpdelTestConfig::enableVarUpdelPerfTest || UpdelTestConfig::enableFixUpdelPerfTest;

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
