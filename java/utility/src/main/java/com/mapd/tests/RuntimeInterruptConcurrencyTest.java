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

import org.checkerframework.checker.units.qual.A;
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
import java.util.HashSet;
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

  private MapdTestClient getClient(String db, String username) {
    try {
      return MapdTestClient.getClient("localhost", 6274, db, username, "password");
    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
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
    int num_threads = 5;
    int INTERRUPTER_TID = num_threads - 1;
    int num_runs = 5;
    final String large_table = "test_large";
    final String small_table = "test_small";
    final String geo_table = "test_geo";
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
    Path geojson_table_path = getAbsolutePath(
            "../java/utility/src/main/java/com/mapd/tests/data/geogdal.geojson");
    try {
      MapdTestClient dba =
              MapdTestClient.getClient("localhost", 6274, db, dbaUser, dbaPassword);
      dba.runSql("CREATE TABLE " + large_table + "(x int not null);");
      dba.runSql("CREATE TABLE " + small_table + "(x int not null);");
      dba.runSql("CREATE TABLE " + geo_table
              + "(trip DOUBLE, pt GEOMETRY(POINT, 4326) ENCODING NONE);");

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

      File geojson_data = new File(geojson_table_path.toString());
      ArrayList<String> geojson_header = new ArrayList<>();
      ArrayList<String> geojson_footer = new ArrayList<>();
      ArrayList<String> geojson_feature = new ArrayList<>();
      geojson_header.add("{");
      geojson_header.add("\"type\": \"FeatureCollection\",");
      geojson_header.add("\"name\": \"geospatial_point\",");
      geojson_header.add(
              "\"crs\": { \"type\": \"name\", \"properties\": { \"name\": \"urn:ogc:def:crs:OGC:1.3:CRS84\" } },");
      geojson_header.add("\"features\": [");
      geojson_footer.add(
              "{ \"type\": \"Feature\", \"properties\": { \"trip\": 10.0 }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [ 10.0, 9.0 ] } }");
      geojson_footer.add("]");
      geojson_footer.add("}");
      geojson_feature.add(
              "{ \"type\": \"Feature\", \"properties\": { \"trip\": 0.0 }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [ 0.0, 1.0 ] } },");
      geojson_feature.add(
              "{ \"type\": \"Feature\", \"properties\": { \"trip\": 1.0 }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [ 1.0, 2.0 ] } },");
      geojson_feature.add(
              "{ \"type\": \"Feature\", \"properties\": { \"trip\": 2.0 }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [ 2.0, 3.0 ] } },");
      geojson_feature.add(
              "{ \"type\": \"Feature\", \"properties\": { \"trip\": 3.0 }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [ 3.0, 4.0 ] } },");
      geojson_feature.add(
              "{ \"type\": \"Feature\", \"properties\": { \"trip\": 4.0 }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [ 4.0, 5.0 ] } },");
      geojson_feature.add(
              "{ \"type\": \"Feature\", \"properties\": { \"trip\": 5.0 }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [ 5.0, 6.0 ] } },");
      geojson_feature.add(
              "{ \"type\": \"Feature\", \"properties\": { \"trip\": 6.0 }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [ 6.0, 7.0 ] } },");
      geojson_feature.add(
              "{ \"type\": \"Feature\", \"properties\": { \"trip\": 7.0 }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [ 7.0, 8.0 ] } },");
      geojson_feature.add(
              "{ \"type\": \"Feature\", \"properties\": { \"trip\": 8.0 }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [ 8.0, 9.0 ] } },");
      geojson_feature.add(
              "{ \"type\": \"Feature\", \"properties\": { \"trip\": 9.0 }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [ 9.0, 0.0 ] } },");
      try (BufferedWriter writer = new BufferedWriter(new FileWriter(geojson_data))) {
        for (String str : geojson_header) {
          writer.write(str + "\n");
        }
        for (int i = 0; i < 1000; i++) {
          for (String str : geojson_feature) {
            writer.write(str + "\n");
          }
        }
        for (String str : geojson_footer) {
          writer.write(str + "\n");
        }
      } catch (IOException e) {
        e.printStackTrace();
      }

      dba.runSql("COPY " + large_table + " FROM '" + large_table_path.toString()
              + "' WITH (header='false');");
      dba.runSql("COPY " + small_table + " FROM '" + small_table_path.toString()
              + "' WITH (header='false');");
      dba.runSql("COPY " + geo_table + " FROM '" + geojson_table_path.toString()
              + "' WITH (header='false', geo='true');");
    } catch (Exception e) {
      logger.error("[" + Thread.currentThread().getId() + "]"
                      + " Caught Exception: " + e.getMessage(),
              e);
    }

    ArrayList<Thread> queryThreads = new ArrayList<>();
    ArrayList<Thread> interrupterThreads = new ArrayList<>();
    Thread query_interrupter = new Thread(new Runnable() {
      @Override
      public void run() {
        // try to interrupt
        int tid = INTERRUPTER_TID;
        String logPrefix = "[" + tid + "]";
        MapdTestClient interrupter = getClient(db, "interrupter");
        int check_empty_session_queue = 0;
        while (true) {
          try {
            List<TQueryInfo> queryInfos = interrupter.get_queries_info();
            boolean found_target_query = false;
            for (TQueryInfo queryInfo : queryInfos) {
              String session_id = queryInfo.query_public_session_id;
              boolean select_query =
                      queryInfo.current_status.equals("RUNNING_QUERY_KERNEL");
              boolean import_query = queryInfo.current_status.equals("RUNNING_IMPORTER");
              boolean can_interrupt = false;
              if (import_query
                      || (select_query
                              && queryInfo.query_str.compareTo(loop_join_query) == 0)) {
                can_interrupt = true;
              }
              if (can_interrupt) {
                interrupter.runSql("KILL QUERY '" + session_id + "';");
                check_empty_session_queue = 0;
                found_target_query = true;
              }
            }
            if (!found_target_query || queryInfos.isEmpty()) {
              ++check_empty_session_queue;
            }
            if (check_empty_session_queue > 20) {
              break;
            }
            Thread.sleep(1000);
          } catch (Exception e) {
            logger.error(logPrefix + " Caught Exception: " + e.getMessage(), e);
          }
        }
      }
    });
    query_interrupter.start();
    interrupterThreads.add(query_interrupter);

    for (int i = 0; i < num_runs; i++) {
      logger.info("Starting run-" + i);
      for (int r = 0; r < num_threads; r++) {
        final int tid = r;
        final String logPrefix = "[" + tid + "]";
        final String user_name = "u".concat(Integer.toString(tid));
        if (r < num_threads - 2) {
          String[] queries = {hash_join_query, gby_query, loop_join_query};
          Thread select_query_runner = new Thread(new Runnable() {
            @Override
            public void run() {
              logger.info("Starting thread-" + tid);
              final MapdTestClient user = getClient(db, user_name);
              for (int k = 0; k < 5; k++) {
                boolean interrupted = false;
                for (int q = 0; q < 3; q++) {
                  try {
                    logger.info(logPrefix + " Run SELECT query: " + queries[q]);
                    user.runSql(queries[q]);
                  } catch (Exception e2) {
                    if (e2 instanceof TOmniSciException) {
                      TOmniSciException ee = (TOmniSciException) e2;
                      if (q == 2 && ee.error_msg.contains("ERR_INTERRUPTED")) {
                        interrupted = true;
                        logger.info(
                                logPrefix + " Select query issued has been interrupted");
                      }
                    } else {
                      logger.error(
                              logPrefix + " Caught Exception: " + e2.getMessage(), e2);
                    }
                  }
                }
                assert interrupted;
              }
            }
          });
          select_query_runner.start();
          queryThreads.add(select_query_runner);
        } else {
          Thread import_query_runner = new Thread(new Runnable() {
            @Override
            public void run() {
              logger.info("Starting thread-" + tid);
              final MapdTestClient user = getClient(db, user_name);
              for (int k = 0; k < 2; k++) {
                boolean interrupted = false;
                try {
                  Path geo_table_path = getAbsolutePath(
                          "../Tests/Import/datafiles/interrupt_table_gdal.geojson");
                  user.runSql("COPY " + geo_table + " FROM '" + geo_table_path.toString()
                          + "' WITH (geo='true');");
                  logger.info(logPrefix + " Run Import query");
                } catch (Exception e2) {
                  if (e2 instanceof TOmniSciException) {
                    TOmniSciException ee = (TOmniSciException) e2;
                    if (ee.error_msg.contains("error code 10")) {
                      interrupted = true;
                      logger.info(logPrefix + " Import query has been interrupted");
                    }
                  } else {
                    logger.error(logPrefix + " Caught Exception: " + e2.getMessage(), e2);
                  }
                }
                assert interrupted;
              }
            }
          });
          import_query_runner.start();
          queryThreads.add(import_query_runner);
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
    File geojson_data = new File(geojson_table_path.toString());
    if (large_data.exists()) {
      large_data.delete();
    }
    if (small_data.exists()) {
      small_data.delete();
    }
    if (geojson_data.exists()) {
      geojson_data.delete();
    }
  }

  public void testConcurrency() throws Exception {
    logger.info("RuntimeInterruptConcurrencyTest()");

    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "heavyai", "admin", "HyperInteractive");
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
