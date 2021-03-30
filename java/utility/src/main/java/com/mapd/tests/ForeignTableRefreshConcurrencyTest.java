/*
 * Copyright 2021 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mapd.tests;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

public class ForeignTableRefreshConcurrencyTest {
  final static Logger logger =
          LoggerFactory.getLogger(ForeignTableRefreshConcurrencyTest.class);

  public static void main(String[] args) throws Exception {
    ForeignTableRefreshConcurrencyTest test = new ForeignTableRefreshConcurrencyTest();
    test.testConcurrency();
  }

  private Path getAbsolutePath(String path) {
    Path path_obj = Paths.get(path).toAbsolutePath();
    assert Files.exists(path_obj);
    return path_obj;
  }

  // This test creates mulitple competing threads, some of which run selects
  // with joins on a table while others refresh a foreign table. The test
  // should resolve without exception as the cache being used during the join
  // exectution should be safely invalidated during the foreign table refreshes.
  private void runTest(String db,
          String userName,
          String userPassword,
          int num_foreign_refresh_threads,
          int num_table_join_threads,
          int num_runs) throws Exception {
    ArrayList<Exception> exceptions = new ArrayList<Exception>();

    Thread[] foreign_table_refresh_threads = new Thread[num_foreign_refresh_threads];
    Thread[] table_join_threads = new Thread[num_table_join_threads];

    for (int i = 0; i < num_foreign_refresh_threads; ++i) {
      final int tid = i;
      foreign_table_refresh_threads[tid] = new Thread(new Runnable() {
        @Override
        public void run() {
          final String thread_name = "F[" + tid + "]";
          try {
            logger.info("Starting foreign table refresh thread " + thread_name);
            MapdTestClient user = MapdTestClient.getClient(
                    "localhost", 6274, db, userName, userPassword);
            for (int irun = 0; irun < num_runs; ++irun) {
              runSqlAsUser(
                      "SELECT * FROM test_foreign_table_" + tid + ";", user, thread_name);
              runSqlAsUser("REFRESH FOREIGN TABLES test_foreign_table_" + tid + ";",
                      user,
                      thread_name);
            }
            logger.info("Finished foreign table refresh " + thread_name);
          } catch (Exception e) {
            logger.error("Foreign table refresh " + thread_name
                            + " Caught Exception: " + e.getMessage(),
                    e);
            exceptions.add(e);
          }
        }
      });
      foreign_table_refresh_threads[tid].start();
    }

    for (int i = 0; i < num_table_join_threads; ++i) {
      final int tid = i;
      table_join_threads[tid] = new Thread(new Runnable() {
        @Override
        public void run() {
          final String thread_name = "T[" + tid + "]";
          try {
            logger.info("Starting table join " + thread_name);
            MapdTestClient user = MapdTestClient.getClient(
                    "localhost", 6274, db, userName, userPassword);
            for (int irun = 0; irun < num_runs; ++irun) {
              runSqlAsUser("SELECT * FROM test_table_" + tid
                              + "_left AS l JOIN test_table_" + tid
                              + "_right AS r ON l.id = r.id;",
                      user,
                      thread_name);
              Thread.sleep(200); // sleep 200 milliseconds between queries to allow more
                                 // contention between foreign table refreshes
            }
            logger.info("Finished table join thread[0]");
          } catch (Exception e) {
            logger.error(
                    "Table join " + thread_name + " Caught Exception: " + e.getMessage(),
                    e);
            exceptions.add(e);
          }
        }
      });
      table_join_threads[tid].start();
    }

    for (Thread t : foreign_table_refresh_threads) {
      t.join();
    }
    for (Thread t : table_join_threads) {
      t.join();
    }

    for (Exception e : exceptions) {
      if (null != e) {
        logger.error("Exception: " + e.getMessage(), e);
        throw e;
      }
    }
  }

  public void runSqlAsUser(String sql, MapdTestClient user, String logPrefix)
          throws Exception {
    logger.info(logPrefix + " " + sql);
    user.runSql(sql);
  }

  private void createForeignTestTable(MapdTestClient dba, String foreign_table_name)
          throws Exception {
    dba.runSql("CREATE FOREIGN TABLE " + foreign_table_name + " "
            + "(b BOOLEAN, t TINYINT, s SMALLINT, i INTEGER, bi BIGINT, f FLOAT, "
            + "dc DECIMAL(10, 5), tm TIME, tp TIMESTAMP, d DATE, txt TEXT, "
            + "txt_2 TEXT ENCODING NONE) "
            + "SERVER test_server WITH "
            + "(file_path = 'scalar_types.csv', "
            + "FRAGMENT_SIZE = 2);");
  }

  private void createTestTable(MapdTestClient dba, String table_name, Path copy_from_path)
          throws Exception {
    dba.runSql("CREATE TABLE " + table_name
            + " (id INTEGER, str TEXT ENCODING DICT(32), x DOUBLE, y BIGINT) WITH (FRAGMENT_SIZE=1)");
    dba.runSql("COPY " + table_name + " FROM '" + copy_from_path.toString()
            + "' WITH (header='false');");
  }

  private void createTestTables(
          MapdTestClient dba, String table_name, Path copy_from_path) throws Exception {
    createTestTable(dba, table_name + "_left", copy_from_path);
    createTestTable(dba, table_name + "_right", copy_from_path);
  }

  public void testConcurrency() throws Exception {
    logger.info("ForeignTableRefreshConcurrencyTest()");

    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "omnisci", "admin", "HyperInteractive");

    // initialize
    su.runSql("DROP DATABASE IF EXISTS db1;");
    su.runSql("CREATE DATABASE db1;");

    // create tables for use
    final int num_foreign_refresh_threads = 2;
    final int num_table_join_threads = 2;
    Path table_import_path = getAbsolutePath(
            "../java/utility/src/main/java/com/mapd/tests/data/simple_test.csv");
    Path foreign_server_path = getAbsolutePath("../Tests/FsiDataFiles/");
    MapdTestClient dba = MapdTestClient.getClient(
            "localhost", 6274, "db1", "admin", "HyperInteractive");
    dba.runSql("CREATE SERVER test_server "
            + "FOREIGN DATA WRAPPER omnisci_csv WITH (storage_type = 'LOCAL_FILE', "
            + "base_path = '" + foreign_server_path.toString() + "');");
    for (int i = 0; i < num_foreign_refresh_threads; ++i) {
      createForeignTestTable(dba, "test_foreign_table_" + i);
    }
    for (int i = 0; i < num_table_join_threads; ++i) {
      createTestTables(dba, "test_table_" + i, table_import_path);
    }

    runTest("db1",
            "admin",
            "HyperInteractive",
            num_foreign_refresh_threads,
            num_table_join_threads,
            25);

    // cleanup
    for (int i = 0; i < num_foreign_refresh_threads; ++i) {
      dba.runSql("DROP FOREIGN TABLE test_foreign_table_" + i + ";");
    }
    for (int i = 0; i < num_table_join_threads; ++i) {
      dba.runSql("DROP TABLE test_table_" + i + "_left ;");
      dba.runSql("DROP TABLE test_table_" + i + "_right ;");
    }
    dba.runSql("DROP SERVER test_server;");
    su.runSql("DROP DATABASE db1;");

    logger.info("ForeignTableRefreshConcurrencyTest() done");
  }
}
