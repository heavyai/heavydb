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

import com.omnisci.thrift.server.TOmniSciException;
import com.omnisci.thrift.server.TQueryInfo;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class RuntimeInterruptConcurrencyTest {
  final static Logger logger =
          LoggerFactory.getLogger(RuntimeInterruptConcurrencyTest.class);

  public static void main(String[] args) throws Exception {
    RuntimeInterruptConcurrencyTest test = new RuntimeInterruptConcurrencyTest();
    test.testConcurrency();
  }

  private Path getAbsolutePath(String path) {
    Path path_obj = Paths.get(path).toAbsolutePath();
    assert Files.exists(path_obj);
    return path_obj;
  }

  private void cleanupUserAndDB(MapdTestClient su) {
    try {
      su.runSql("DROP DATABASE db1;");
      su.runSql("DROP USER u0;");
      su.runSql("DROP USER u1;");
      su.runSql("DROP USER u2;");
      su.runSql("DROP USER u3;");
      su.runSql("DROP USER u4;");
      su.runSql("DROP USER interrupter;");
      su.runSql("DROP TABLE IF EXISTS test_large;");
      su.runSql("DROP TABLE IF EXISTS test_small;");
      su.runSql("DROP TABLE IF EXISTS test_geo;");
    } catch (Exception e) {
      logger.error(
              "Get exception while cleanup db, tables and users: " + e.getMessage(), e);
    }
  }

  private void runTest(
          String db, String dbaUser, String dbaPassword, String dbUser, String dbPassword)
          throws Exception {
    int num_threads = 4;
    int num_runs = 5;
    final String large_table = "test_large";
    final String small_table = "test_small";
    final String geo_table = "test_geo";
    Exception[] exceptions = new Exception[num_threads];
    String loop_join_query =
            "SELECT /*+ cpu_mode */ COUNT(1) FROM test_large T1, test_large T2;";
    String hash_join_query =
            "SELECT /*+ cpu_mode */ COUNT(1) FROM test_large T1, test_small T2 WHERE T1.x = T2.x;";
    String gby_query =
            "SELECT /*+ cpu_mode */ x, count(1) FROM test_large T1 GROUP BY x;";
    Path large_table_path =
            getAbsolutePath("../java/utility/src/main/java/com/mapd/tests/data/1M.csv");
    Path small_table_path =
            getAbsolutePath("../java/utility/src/main/java/com/mapd/tests/data/1K.csv");
    try {
      MapdTestClient dba =
              MapdTestClient.getClient("localhost", 6274, db, dbaUser, dbaPassword);
      dba.runSql("CREATE TABLE " + large_table + "(x int not null);");
      dba.runSql("CREATE TABLE " + small_table + "(x int not null);");
      dba.runSql("CREATE TABLE " + geo_table
              + "(trip DOUBLE, omnisci_geo GEOMETRY(POINT, 4326) ENCODING NONE);");

      File large_data = new File(large_table_path.toString());
      try (BufferedWriter writer = new BufferedWriter(new FileWriter(large_data))) {
        for (int i = 0; i < 1000000; i++) {
          writer.write(i + "\n");
        }
      } catch (IOException e) {
        e.printStackTrace();
      }

      File small_data = new File(small_table_path.toString());
      try (BufferedWriter writer = new BufferedWriter(new FileWriter(small_data))) {
        for (int i = 0; i < 1000; i++) {
          writer.write(i + "\n");
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
      dba.runSql("COPY " + large_table + " FROM '" + large_table_path.toString()
              + "' WITH (header='false');");
      dba.runSql("COPY " + large_table + " FROM '" + small_table_path.toString()
              + "' WITH (header='false');");
    } catch (Exception e) {
      logger.error("[" + Thread.currentThread().getId() + "]"
                      + " Caught Exception: " + e.getMessage(),
              e);
      exceptions[0] = e;
    }

    ArrayList<Thread> queryThreads = new ArrayList<>();
    ArrayList<Thread> interrupterThreads = new ArrayList<>();
    for (int i = 0; i < num_threads; i++) {
      logger.info("Starting " + i);
      final int threadId = i;
      final String user_name = "u".concat(Integer.toString(threadId));
      for (int r = 0; r < num_runs; r++) {
        if (i < 2) {
          String[] queries = {hash_join_query, gby_query, loop_join_query};
          boolean interrupted = false;
          Thread interrupt_case_t = new Thread(new Runnable() {
            @Override
            public void run() {
              long tid = Thread.currentThread().getId();
              String logPrefix = "[" + tid + "]";
              try {
                MapdTestClient user = MapdTestClient.getClient(
                        "localhost", 6274, db, user_name, "password");
                MapdTestClient interrupter = MapdTestClient.getClient(
                        "localhost", 6274, db, "interrupter", "password");
                for (int r = 0; r < num_runs; r++) {
                  boolean interrupted = false;
                  for (int q = 0; q < 3; q++) {
                    if (q == 2) {
                      Thread interrupt_thread = new Thread(new Runnable() {
                        @Override
                        public void run() {
                          // try to interrupt
                          boolean foundRunningQuery = false;
                          while (!foundRunningQuery) {
                            try {
                              List<TQueryInfo> queryInfos =
                                      interrupter.get_queries_info();
                              for (TQueryInfo queryInfo : queryInfos) {
                                if (queryInfo.query_session_id.equals(user.sessionId)
                                        && queryInfo.current_status.equals(
                                                "RUNNING_QUERY_KERNEL")) {
                                  foundRunningQuery = true;
                                  interrupter.runSql("KILL QUERY '"
                                          + queryInfo.query_public_session_id + "';");
                                  break;
                                }
                              }
                              Thread.sleep(100);
                            } catch (Exception e) {
                              logger.error(
                                      logPrefix + " Caught Exception: " + e.getMessage(),
                                      e);
                              exceptions[threadId] = e;
                            }
                          }
                        }
                      });
                      interrupt_thread.start();
                      interrupterThreads.add(interrupt_thread);
                    }
                    try {
                      logger.info(logPrefix + "Run SELECT query: " + queries[q]);
                      user.runSql(queries[q]);
                    } catch (Exception e3) {
                      if (e3 instanceof TOmniSciException) {
                        TOmniSciException ee = (TOmniSciException) e3;
                        if (q == 2 && ee.error_msg.contains("ERR_INTERRUPTED")) {
                          interrupted = true;
                          logger.info(
                                  logPrefix + "Select query issued has been interrupted");
                        }
                      } else {
                        logger.error(
                                logPrefix + " Caught Exception: " + e3.getMessage(), e3);
                        exceptions[threadId] = e3;
                      }
                    }
                  }
                  assert interrupted;
                }
              } catch (Exception e) {
                logger.error(logPrefix + " Caught Exception: " + e.getMessage(), e);
                exceptions[threadId] = e;
              }
            }
          });
          interrupt_case_t.start();
          queryThreads.add(interrupt_case_t);
        } else {
          boolean interrupted = false;
          Thread importer_case_t = new Thread(new Runnable() {
            @Override
            public void run() {
              long tid = Thread.currentThread().getId();
              String logPrefix = "[" + tid + "]";
              try {
                MapdTestClient user = MapdTestClient.getClient(
                        "localhost", 6274, db, user_name, "password");
                MapdTestClient interrupter = MapdTestClient.getClient(
                        "localhost", 6274, db, "interrupter", "password");
                for (int r = 0; r < num_runs; r++) {
                  boolean interrupted = false;
                  Thread interrupt_thread = new Thread(new Runnable() {
                    @Override
                    public void run() {
                      // try to interrupt
                      boolean foundRunningQuery = false;
                      while (!foundRunningQuery) {
                        try {
                          List<TQueryInfo> queryInfos = interrupter.get_queries_info();
                          for (TQueryInfo queryInfo : queryInfos) {
                            if (queryInfo.query_session_id.equals(user.sessionId)
                                    && queryInfo.current_status.equals(
                                            "RUNNING_IMPORTER")) {
                              foundRunningQuery = true;
                              interrupter.runSql("KILL QUERY '"
                                      + queryInfo.query_public_session_id + "';");
                              break;
                            }
                          }
                          Thread.sleep(100);
                        } catch (Exception e) {
                          logger.error(
                                  logPrefix + " Caught Exception: " + e.getMessage(), e);
                          exceptions[threadId] = e;
                        }
                      }
                    }
                  });
                  interrupt_thread.start();
                  interrupterThreads.add(interrupt_thread);
                  try {
                    Path geo_table_path = getAbsolutePath(
                            "../Tests/Import/datafiles/interrupt_table_gdal.geojson");
                    user.runSql("COPY " + geo_table + " FROM '"
                            + geo_table_path.toString() + "' WITH (geo='true');");
                    logger.info(logPrefix + "Run Import query");
                  } catch (Exception e3) {
                    if (e3 instanceof TOmniSciException) {
                      TOmniSciException ee = (TOmniSciException) e3;
                      if (ee.error_msg.contains("error code 10")) {
                        interrupted = true;
                        logger.info(logPrefix + "Import query has been interrupted");
                      }
                    } else {
                      logger.error(
                              logPrefix + " Caught Exception: " + e3.getMessage(), e3);
                      exceptions[threadId] = e3;
                    }
                  }
                  assert interrupted;
                }
              } catch (Exception e) {
                logger.error(logPrefix + " Caught Exception: " + e.getMessage(), e);
                exceptions[threadId] = e;
              }
            }
          });
          importer_case_t.start();
          queryThreads.add(importer_case_t);
        }
      }
    }

    for (Thread t : queryThreads) {
      t.join();
    }
    for (Thread t : interrupterThreads) {
      t.join();
    }

    MapdTestClient dba =
            MapdTestClient.getClient("localhost", 6274, db, dbaUser, dbaPassword);
    dba.runSql("DROP TABLE " + large_table + ";");
    dba.runSql("DROP TABLE " + small_table + ";");
    dba.runSql("DROP TABLE " + geo_table + ";");
    File large_data = new File(large_table_path.toString());
    File small_data = new File(small_table_path.toString());
    if (large_data.exists()) {
      large_data.delete();
    }
    if (small_data.exists()) {
      small_data.delete();
    }
  }

  public void testConcurrency() throws Exception {
    logger.info("RuntimeInterruptConcurrencyTest()");

    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "omnisci", "admin", "HyperInteractive");
    cleanupUserAndDB(su);
    su.runSql("CREATE DATABASE db1;");
    su.runSql("CREATE USER u0 (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER u1 (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER u2 (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER u3 (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER u4 (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER interrupter (password = 'password', is_super = 'true');");
    su.runSql("GRANT ALL on DATABASE db1 TO u0;");
    su.runSql("GRANT ALL on DATABASE db1 TO u1;");
    su.runSql("GRANT ALL on DATABASE db1 TO u2;");
    su.runSql("GRANT ALL on DATABASE db1 TO u3;");
    su.runSql("GRANT ALL on DATABASE db1 TO u4;");
    su.runSql("GRANT ALL on DATABASE db1 TO interrupter;");
    runTest("db1", "admin", "HyperInteractive", "admin", "HyperInteractive");
    cleanupUserAndDB(su);
    logger.info("RuntimeInterruptConcurrencyTest() done");
  }
}
