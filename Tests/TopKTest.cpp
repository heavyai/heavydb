
/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ArrowSQLRunner/ArrowSQLRunner.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>

#ifdef _WIN32
#define timegm _mkgmtime
#endif

using namespace std;
using namespace TestHelpers;
using namespace TestHelpers::ArrowSQLRunner;

namespace {

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !gpusPresent();
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

}  // namespace

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

namespace {

void create_and_populate_tables() {
  createTable("tdata",
              {{"id", SQLTypeInfo(kSMALLINT)},
               {"b", SQLTypeInfo(kBOOLEAN)},
               {"i", SQLTypeInfo(kINT)},
               {"bi", SQLTypeInfo(kBIGINT)},
               {"n", SQLTypeInfo(kDECIMAL, 10, 2, false)},
               {"f", SQLTypeInfo(kFLOAT)},
               {"t", dictType()},
               {"tt", SQLTypeInfo(kTIME)},
               {"d", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 32, kNULLT)},
               {"ts", SQLTypeInfo(kTIMESTAMP, 0, 0)},
               {"vc", dictType()}});
  insertCsvValues(
      "tdata",
      R"___(1,true,23,2349923,111.1,1.1,SFO,15:13:14,1999-09-09,2014-12-13 22:23:15,paris
2,false,,-973273,7263.11,87.1,,20:05:00,2017-12-12,2017-12-12 20:05:00,toronto
3,false,702,87395,333.5,,YVR,11:11:11,2010-01-01,2010-01-02 04:11:45,vancouver
4,,864,100001,,9.9,SJC,,2015-05-05,2010-05-05 05:15:55,london
5,false,333,112233,99.9,9.9,ABQ,22:22:22,2015-05-05,2010-05-05 05:15:55,new york
6,true,-3,18,765.8,2.2,YYZ,00:00:01,,2009-01-08 12:13:14,
7,false,-9873,3789,789.3,4.7,DCA,11:22:33,2001-02-03,2005-04-03 15:16:17,rio de janerio
8,true,12,4321,83.9,1.2,DXB,21:20:10,,2007-12-01 23:22:21,dubai
9,true,48,,83.9,1.2,BWI,09:08:07,2001-09-11,,washington
10,false,99,777,77.7,7.7,LLBG,07:07:07,2017-07-07,2017-07-07 07:07:07,Tel Aviv)___");

  run_sqlite_query("DROP TABLE IF EXISTS tdata;");
  run_sqlite_query(
      "CREATE TABLE tdata (id SMALLINT, b BOOLEAN, i INT, bi BIGINT, n DECIMAL(10, 2), "
      "f FLOAT, t TEXT, tt TIME, d DATE, ts TIMESTAMP, vc VARCHAR(15));");

  // Insert data into the table
  run_sqlite_query(
      "INSERT INTO tdata VALUES(1, 't', 23, 2349923, 111.1, 1.1, 'SFO', '15:13:14', "
      "'1999-09-09', '2014-12-13 22:23:15', 'paris');");
  run_sqlite_query(
      "INSERT INTO tdata VALUES(2, 'f', null, -973273, 7263.11, 87.1, null, "
      "'20:05:00', '2017-12-12', '2017-12-12 20:05:00', 'toronto');");
  run_sqlite_query(
      "INSERT INTO tdata VALUES(3, 'f', 702, 87395, 333.5, null, 'YVR', '11:11:11', "
      "'2010-01-01', '2010-01-02 04:11:45', 'vancouver');");
  run_sqlite_query(
      "INSERT INTO tdata VALUES(4, null, 864, 100001, null, 9.9, 'SJC', null, "
      "'2015-05-05', '2010-05-05 05:15:55', 'london');");
  run_sqlite_query(
      "INSERT INTO tdata VALUES(5, 'f', 333, 112233, 99.9, 9.9, 'ABQ', '22:22:22', "
      "'2015-05-05', '2010-05-05 05:15:55', 'new york');");
  run_sqlite_query(
      "INSERT INTO tdata VALUES(6, 't', -3, 18, 765.8, 2.2, 'YYZ', '00:00:01', null, "
      "'2009-01-08 12:13:14', null);");
  run_sqlite_query(
      "INSERT INTO tdata VALUES(7, 'f', -9873, 3789, 789.3, 4.7, 'DCA', '11:22:33', "
      "'2001-02-03', '2005-04-03 15:16:17', 'rio de janerio');");
  run_sqlite_query(
      "INSERT INTO tdata VALUES(8, 't', 12, 4321, 83.9, 1.2, 'DXB', '21:20:10', null, "
      "'2007-12-01 23:22:21', 'dubai');");
  run_sqlite_query(
      "INSERT INTO tdata VALUES(9, 't', 48, null, 83.9, 1.2, 'BWI', '09:08:07', "
      "'2001-09-11', null, 'washington');");
  run_sqlite_query(
      "INSERT INTO tdata VALUES(10, 'f', 99, 777, 77.7, 7.7, 'LLBG', '07:07:07', "
      "'2017-07-07', '2017-07-07 07:07:07', 'Tel Aviv');");
}

void drop_tables() {
  dropTable("tdata");
  run_sqlite_query("DROP TABLE tdata;");
}
}  // namespace

TEST(Select, TopK_LIMIT_AscendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i NULLS FIRST LIMIT 5;",
      "SELECT i FROM tdata ORDER BY i ASC LIMIT 5;",
      dt);
    c("SELECT b FROM tdata ORDER BY b NULLS FIRST LIMIT 5;",
      "SELECT b FROM tdata ORDER BY b LIMIT 5;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi NULLS FIRST LIMIT 5;",
      "SELECT bi FROM tdata ORDER BY bi LIMIT 5;",
      dt);
    c("SELECT n FROM tdata ORDER BY n NULLS FIRST LIMIT 5;",
      "SELECT n FROM tdata ORDER BY n LIMIT 5;",
      dt);
    c("SELECT f FROM tdata ORDER BY f NULLS FIRST LIMIT 5;",
      "SELECT f FROM tdata ORDER BY f LIMIT 5;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt NULLS FIRST LIMIT 5;",
      "SELECT tt FROM tdata ORDER BY tt LIMIT 5;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts NULLS FIRST LIMIT 5;",
      "SELECT ts FROM tdata ORDER BY ts LIMIT 5;",
      dt);
    c("SELECT d FROM tdata ORDER BY d NULLS FIRST LIMIT 5;",
      "SELECT d FROM tdata ORDER BY d LIMIT 5;",
      dt);
  }
}

TEST(Select, TopK_LIMIT_DescendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5;",
      "SELECT i FROM tdata ORDER BY i DESC LIMIT 5;",
      dt);
    c("SELECT b FROM tdata ORDER BY b DESC NULLS LAST LIMIT 5;",
      "SELECT b FROM tdata ORDER BY b DESC LIMIT 5;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi DESC NULLS LAST LIMIT 5;",
      "SELECT bi FROM tdata ORDER BY bi DESC LIMIT 5;",
      dt);
    c("SELECT n FROM tdata ORDER BY n DESC NULLS LAST LIMIT 5;",
      "SELECT n FROM tdata ORDER BY n DESC LIMIT 5;",
      dt);
    c("SELECT f FROM tdata ORDER BY f DESC NULLS LAST LIMIT 5;",
      "SELECT f FROM tdata ORDER BY f DESC LIMIT 5;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt DESC NULLS LAST LIMIT 5;",
      "SELECT tt FROM tdata ORDER BY tt DESC LIMIT 5;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts DESC NULLS LAST LIMIT 5;",
      "SELECT ts FROM tdata ORDER BY ts DESC LIMIT 5;",
      dt);
    c("SELECT d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 5;",
      "SELECT d FROM tdata ORDER BY d DESC LIMIT 5;",
      dt);
  }
}

TEST(Select, TopK_LIMIT_GreaterThan_TotalOfDataRows_AscendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT b FROM tdata ORDER BY b NULLS FIRST LIMIT 11;",
      "SELECT b FROM tdata ORDER BY b LIMIT 11;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi NULLS FIRST LIMIT 11;",
      "SELECT bi FROM tdata ORDER BY bi LIMIT 11;",
      dt);
    c("SELECT n FROM tdata ORDER BY n NULLS FIRST LIMIT 11;",
      "SELECT n FROM tdata ORDER BY n LIMIT 11;",
      dt);
    c("SELECT f FROM tdata ORDER BY f NULLS FIRST LIMIT 11;",
      "SELECT f FROM tdata ORDER BY f LIMIT 11;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt NULLS FIRST LIMIT 11;",
      "SELECT tt FROM tdata ORDER BY tt LIMIT 11;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts NULLS FIRST LIMIT 11;",
      "SELECT ts FROM tdata ORDER BY ts LIMIT 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY d NULLS FIRST LIMIT 11;",
      "SELECT d FROM tdata ORDER BY d LIMIT 11;",
      dt);
  }
}

TEST(Select, TopK_LIMIT_GreaterThan_TotalOfDataRows_DescendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i DESC NULLS LAST LIMIT 11;",
      "SELECT i FROM tdata ORDER BY i DESC LIMIT 11;",
      dt);
    c("SELECT b FROM tdata ORDER BY b DESC NULLS LAST LIMIT 11;",
      "SELECT b FROM tdata ORDER BY b DESC LIMIT 11;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi DESC NULLS LAST LIMIT 11;",
      "SELECT bi FROM tdata ORDER BY bi DESC LIMIT 11;",
      dt);
    c("SELECT n FROM tdata ORDER BY n DESC NULLS LAST LIMIT 11;",
      "SELECT n FROM tdata ORDER BY n DESC LIMIT 11;",
      dt);
    c("SELECT f FROM tdata ORDER BY f DESC NULLS LAST LIMIT 11;",
      "SELECT f FROM tdata ORDER BY f DESC LIMIT 11;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt DESC NULLS LAST LIMIT 11;",
      "SELECT tt FROM tdata ORDER BY tt DESC LIMIT 11;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts DESC NULLS LAST LIMIT 11;",
      "SELECT ts FROM tdata ORDER BY ts DESC LIMIT 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 11;",
      "SELECT d FROM tdata ORDER BY d DESC LIMIT 11;",
      dt);
  }
}

TEST(Select, TopK_LIMIT_OFFSET_TopHalf_AscendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT i FROM tdata ORDER BY i LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT b FROM tdata ORDER BY b NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT b FROM tdata ORDER BY b LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT bi FROM tdata ORDER BY bi LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT n FROM tdata ORDER BY n NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT n FROM tdata ORDER BY n LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT f FROM tdata ORDER BY f NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT f FROM tdata ORDER BY f LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT tt FROM tdata ORDER BY tt LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT ts FROM tdata ORDER BY ts LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT d FROM tdata ORDER BY d NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT d FROM tdata ORDER BY d LIMIT 5 OFFSET 0;",
      dt);
  }
}

TEST(Select, TopK_LIMIT_OFFSET_TopHalf_DescendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT i FROM tdata ORDER BY i DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT b FROM tdata ORDER BY b DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT b FROM tdata ORDER BY b DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT bi FROM tdata ORDER BY bi DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT n FROM tdata ORDER BY n DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT n FROM tdata ORDER BY n DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT f FROM tdata ORDER BY f DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT f FROM tdata ORDER BY f DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT tt FROM tdata ORDER BY tt DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT ts FROM tdata ORDER BY ts DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT d FROM tdata ORDER BY d DESC LIMIT 5 OFFSET 0;xxx",
      dt);
  }
}

TEST(Select, DISABLED_TopK_LIMIT_OFFSET_BottomHalf_AscendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT i FROM tdata ORDER BY i LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT b FROM tdata ORDER BY b NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT b FROM tdata ORDER BY b LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT bi FROM tdata ORDER BY bi LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT n FROM tdata ORDER BY n NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT n FROM tdata ORDER BY n LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT f FROM tdata ORDER BY f NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT f FROM tdata ORDER BY f LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT tt FROM tdata ORDER BY tt LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT ts FROM tdata ORDER BY ts LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT d FROM tdata ORDER BY d NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT d FROM tdata ORDER BY d LIMIT 5 OFFSET 5;",
      dt);
  }
}

TEST(Select, DISABLED_TopK_LIMIT_OFFSET_BottomHalf_DescendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT i FROM tdata ORDER BY i DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT b FROM tdata ORDER BY b DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT b FROM tdata ORDER BY b DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT bi FROM tdata ORDER BY bi DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT n FROM tdata ORDER BY n DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT n FROM tdata ORDER BY n DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT f FROM tdata ORDER BY f DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT f FROM tdata ORDER BY f DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT tt FROM tdata ORDER BY tt DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT ts FROM tdata ORDER BY ts DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT d FROM tdata ORDER BY d DESC LIMIT 5 OFFSET 5;",
      dt);
  }
}

TEST(Select, DISABLED_TopK_LIMIT_OFFSET_GreaterThan_TotalOfDataRows_AscendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT i FROM tdata ORDER BY i LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT b FROM tdata ORDER BY b NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT b FROM tdata ORDER BY b LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT bi FROM tdata ORDER BY bi LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT n FROM tdata ORDER BY n NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT n FROM tdata ORDER BY n LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT f FROM tdata ORDER BY f NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT f FROM tdata ORDER BY f LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT tt FROM tdata ORDER BY tt LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT ts FROM tdata ORDER BY ts LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY d NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY d LIMIT 5 OFFSET 11;",
      dt);
  }
}

TEST(Select, DISABLED_TopK_LIMIT_OFFSET_GreaterThan_TotalOfDataRows_DescendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT i FROM tdata ORDER BY i DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT b FROM tdata ORDER BY b DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT b FROM tdata ORDER BY b DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT bi FROM tdata ORDER BY bi DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT n FROM tdata ORDER BY n DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT n FROM tdata ORDER BY n DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT f FROM tdata ORDER BY f DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT f FROM tdata ORDER BY f DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT tt FROM tdata ORDER BY tt DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT ts FROM tdata ORDER BY ts DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY d DESC LIMIT 5 OFFSET 11;",
      dt);
  }
}

TEST(Select, DISABLED_TopK_LIMIT_OFFSET_DifferentOrders) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i,d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY d DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT i,d FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY i DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT i,d FROM tdata ORDER BY i,d DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY i,d DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY i DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY i,d DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY i,d DESC LIMIT 5 OFFSET 11;",
      dt);

    c("SELECT i,d FROM tdata ORDER BY d NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY d LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT i,d FROM tdata ORDER BY i NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY i LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT i,d FROM tdata ORDER BY i,d NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY i,d LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY i NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY i LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY i,d NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY i,d LIMIT 5 OFFSET 11;",
      dt);
  }
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  init();

  int err{0};

  try {
    create_and_populate_tables();
    err = RUN_ALL_TESTS();
    drop_tables();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    err = EINVAL;
  }

  reset();
  return err;
}
