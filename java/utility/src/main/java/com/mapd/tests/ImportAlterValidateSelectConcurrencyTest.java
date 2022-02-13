/*
 * Copyright 2020 OmniSci, Inc.
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

import com.omnisci.thrift.server.TColumnType;
import com.omnisci.thrift.server.TCopyParams;
import com.omnisci.thrift.server.TCreateParams;
import com.omnisci.thrift.server.TImportHeaderRow;
import com.omnisci.thrift.server.TOmniSciException;
import com.omnisci.thrift.server.TSourceType;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CyclicBarrier;

public class ImportAlterValidateSelectConcurrencyTest {
  final static String csvTableName = "import_test_mixed_varlen";
  final static String geoTableName = "geospatial";

  String csv_file_path;
  String geo_file_path;

  final static Logger logger =
          LoggerFactory.getLogger(ImportAlterValidateSelectConcurrencyTest.class);

  public ImportAlterValidateSelectConcurrencyTest(
          String csv_file_path, String geo_file_path) {
    this.geo_file_path = geo_file_path;
    this.csv_file_path = csv_file_path;
  }

  public static void main(String[] args) throws Exception {
    // Command Line Args:
    //  0: CSV file to import (absolute path accessible by server)
    //  1: Geo file to import (absolute path accessible by server)
    assert args.length == 2;
    ImportAlterValidateSelectConcurrencyTest test =
            new ImportAlterValidateSelectConcurrencyTest(args[0], args[1]);
    test.testConcurrency();
  }

  private void runTest(
          String db, String dbaUser, String dbaPassword, String dbUser, String dbPassword)
          throws Exception {
    int num_threads = 4;
    final int runs = 25;
    final int fragment_size = 10;
    Exception exceptions[] = new Exception[num_threads];

    final CyclicBarrier barrier = new CyclicBarrier(num_threads, new Runnable() {
      public void run() {
        try {
          MapdTestClient dba =
                  MapdTestClient.getClient("localhost", 6274, db, dbaUser, dbaPassword);
          dba.runSql("CREATE TABLE " + csvTableName
                  + "(pt GEOMETRY(POINT), ls GEOMETRY(LINESTRING), faii INTEGER[2], fadc DECIMAL(5, 2)[2], fatx TEXT[] ENCODING DICT(32), fatx2 TEXT[2] ENCODING DICT(32)) WITH(FRAGMENT_SIZE = "
                  + fragment_size + ")");

          dba.runSql("CREATE TABLE " + geoTableName
                  + "( trip INT, mpoly MULTIPOLYGON ) WITH(FRAGMENT_SIZE = "
                  + fragment_size + ")");

        } catch (Exception e) {
          logger.error("[" + Thread.currentThread().getId() + "]"
                          + " Caught Exception: " + e.getMessage(),
                  e);
          exceptions[0] = e;
        }
      }
    });

    ArrayList<Thread> threads = new ArrayList<>();
    for (int i = 0; i < num_threads; i++) {
      logger.info("Starting " + i);
      final int threadId = i;

      Thread t = new Thread(new Runnable() {
        @Override
        public void run() {
          long tid = Thread.currentThread().getId();
          String logPrefix = "[" + tid + "]";
          String sql = "";

          TCopyParams copy_params = new TCopyParams();
          copy_params.has_header = TImportHeaderRow.NO_HEADER;
          copy_params.delimiter = ",";
          copy_params.null_str = "\\N";
          copy_params.quoted = true;
          copy_params.quote = "\"";
          copy_params.escape = "\"";
          copy_params.line_delim = "\n";
          copy_params.array_delim = ",";
          copy_params.array_begin = "{";
          copy_params.array_end = "}";
          copy_params.threads = 0;

          TCopyParams geo_copy_params = new TCopyParams();
          geo_copy_params.delimiter = ",";
          geo_copy_params.null_str = "\\N";
          geo_copy_params.quoted = true;
          geo_copy_params.quote = "\"";
          geo_copy_params.escape = "\"";
          geo_copy_params.line_delim = "\n";
          geo_copy_params.array_delim = ",";
          geo_copy_params.array_begin = "{";
          geo_copy_params.array_end = "}";
          geo_copy_params.threads = 0;
          geo_copy_params.source_type = TSourceType.GEO_FILE;

          try {
            barrier.await();

            MapdTestClient user =
                    MapdTestClient.getClient("localhost", 6274, db, dbUser, dbPassword);

            if (threadId % 2 == 0) {
              logger.info(logPrefix + " IMPORT TABLE");
              user.import_table(csvTableName, csv_file_path, copy_params);
              if (threadId == 0) {
                loadTable(user, logPrefix);
              } else {
                loadTableBinaryColumnar(user, logPrefix);
                sql = "COPY " + csvTableName + " FROM '" + csv_file_path
                        + "' WITH (header = 'false');";
                logAndRunSql(sql, user, logPrefix);
              }
            }

            sql = "DELETE FROM " + csvTableName + " WHERE fatx2 IS NULL;";
            logAndRunSql(sql, user, logPrefix);

            sql = "SELECT COUNT(*) FROM " + csvTableName + ";";
            logAndRunSql(sql, user, logPrefix);

            if (threadId == 1) {
              Thread.sleep(5000); // Ensure import is launched
              sql = "ALTER TABLE " + csvTableName + " DROP COLUMN faii;";
              logAndRunSql(sql, user, logPrefix);
            }

            if (threadId % 2 == 1) {
              getTableDetails(user, logPrefix);
            } else {
              getTablesMetadata(user, logPrefix);
            }

            sql = "SELECT * FROM " + geoTableName + ";";
            logger.info(logPrefix + " VALIDATE " + sql);
            final String validateSql = sql;
            // Concurrent request to drop table may have occurred when this query is
            // executed. Ignore the error response in this case.
            ignoreMissingTable(() -> user.sqlValidate(validateSql), geoTableName);

            final String alterSql = "ALTER TABLE " + geoTableName + " SET max_rows = 10;";
            // Concurrent request to drop table may have occurred when this query is
            // executed. Ignore the error response in this case.
            ignoreMissingTable(
                    () -> logAndRunSql(alterSql, user, logPrefix), geoTableName);

            if (threadId == 3) {
              logger.info(logPrefix + " IMPORT GEO TABLE");
              // Concurrent request to drop table may have occurred when this query is
              // executed. Ignore the error response in this case.
              ignoreMissingTable(()
                                         -> user.import_geo_table(geoTableName,
                                                 geo_file_path,
                                                 geo_copy_params,
                                                 new java.util.ArrayList<TColumnType>(),
                                                 new TCreateParams()),
                      geoTableName);
              loadTableBinaryColumnarPolys(user, logPrefix);
            }

            final String selectSql = "SELECT * FROM " + geoTableName + " LIMIT 2;";
            // Concurrent request to drop table may have occurred when this query is
            // executed. Ignore the error response in this case.
            ignoreMissingTable(
                    () -> logAndRunSql(selectSql, user, logPrefix), geoTableName);

            sql = "SELECT * FROM " + csvTableName + ";";
            logger.info(logPrefix + " VALIDATE " + sql);
            user.sqlValidate(sql);

            sql = "ALTER TABLE " + csvTableName + " SET max_rollback_epochs = 0;";
            logAndRunSql(sql, user, logPrefix);

            sql = "COPY (SELECT * FROM  " + csvTableName + ") TO 'test_export.csv';";
            logAndRunSql(sql, user, logPrefix);

            for (int i = 0; i < 5; i++) {
              final String insertSql = "INSERT INTO " + geoTableName + " VALUES (" + i
                      + ", 'MULTIPOLYGON(((0 0, 1 1, 2 2)))');";
              // Concurrent request to drop table may have occurred when this query is
              // executed. Ignore the error response in this case.
              ignoreMissingTable(
                      () -> logAndRunSql(insertSql, user, logPrefix), geoTableName);
            }

            sql = "COPY (SELECT * FROM  " + csvTableName + ") TO 'test_export.csv';";
            logAndRunSql(sql, user, logPrefix);

            sql = "TRUNCATE TABLE " + csvTableName + ";";
            logAndRunSql(sql, user, logPrefix);

            sql = "SELECT COUNT(*) FROM " + csvTableName + ";";
            logger.info(logPrefix + " VALIDATE " + sql);
            user.sqlValidate(sql);

            if (threadId == 0) {
              Thread.sleep(5000); // Ensure import is launched
              sql = "DROP TABLE " + geoTableName + ";";
              logAndRunSql(sql, user, logPrefix);
            }
          } catch (Exception e) {
            logger.error(logPrefix + " Caught Exception: " + e.getMessage(), e);
            exceptions[threadId] = e;
          }
        }
      });
      t.start();
      threads.add(t);
    }

    for (Thread t : threads) {
      t.join();
    }

    MapdTestClient dba =
            MapdTestClient.getClient("localhost", 6274, db, dbaUser, dbaPassword);
    dba.runSql("DROP TABLE " + csvTableName + ";");

    for (Exception e : exceptions) {
      if (null != e) {
        logger.error("Exception: " + e.getMessage(), e);
        throw e;
      }
    }
  }

  public void testConcurrency() throws Exception {
    logger.info("ImportAlterValidateSelectConcurrencyTest()");
    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "omnisci", "admin", "HyperInteractive");
    try {
      su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
      su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");

      su.runSql("GRANT CREATE on DATABASE omnisci TO bob;");

      su.runSql("CREATE DATABASE db1;");
      su.runSql("GRANT CREATE on DATABASE db1 TO bob;");
      su.runSql("GRANT CREATE VIEW on DATABASE db1 TO bob;");
      su.runSql("GRANT DROP on DATABASE db1 TO bob;");
      su.runSql("GRANT DROP VIEW on DATABASE db1 TO bob;");

      runTest("db1", "admin", "HyperInteractive", "admin", "HyperInteractive");
      // TODO: run some tests as bob
    } finally {
      su.runSql("DROP DATABASE IF EXISTS db1;");
      su.runSql("DROP USER IF EXISTS bob;");
      su.runSql("DROP USER IF EXISTS dba;");
    }

    logger.info("ImportAlterValidateSelectConcurrencyTest() done");
  }

  @FunctionalInterface
  private interface VoidFunction {
    void call() throws Exception;
  }

  private void ignoreMissingTable(final VoidFunction function, final String tableName)
          throws Exception {
    try {
      function.call();
    } catch (TOmniSciException e) {
      if (e.error_msg.matches("(Table/View\\s+" + tableName
                  + ".+does not exist|.+Object\\s+'" + tableName + "'\\s+not found)")) {
        logger.info("Ignoring missing table error: " + e.error_msg);
      } else {
        throw e;
      }
    }
  }

  private void logAndRunSql(String sql, MapdTestClient user, String logPrefix)
          throws Exception {
    logger.info(logPrefix + " " + sql);
    user.runSql(sql);
  }

  private void loadTable(MapdTestClient user, String logPrefix) throws Exception {
    logger.info(logPrefix + " Calling load_table API");
    List<List<String>> rows = new ArrayList<>();
    for (int i = 0; i < 5; i++) {
      rows.add(Arrays.asList("point(0 0)",
              "linestring(0 0,1 1)",
              "{1,1}",
              "{1.11,1.11}",
              "{\"1\",\"1\"}",
              "{\"1\",\"1\"}"));
    }
    user.load_table(csvTableName, rows, new ArrayList<>());
  }

  private void loadTableBinaryColumnar(MapdTestClient user, String logPrefix)
          throws Exception {
    logger.info(logPrefix + " Calling load_table_binary_columnar API");
    List<List<Object>> columns = new ArrayList<>();
    for (int i = 0; i < 3; i++) {
      columns.add(new ArrayList<>());
    }
    for (int i = 0; i < 5; i++) {
      columns.get(0).add(Arrays.asList(Long.valueOf(1), Long.valueOf(1)));
      columns.get(1).add(Arrays.asList("1", "1"));
      columns.get(2).add(Arrays.asList("1", "1"));
    }
    user.load_table_binary_columnar(
            csvTableName, columns, Arrays.asList("faii", "fatx", "fatx2"));
  }

  private void loadTableBinaryColumnarPolys(MapdTestClient user, String logPrefix)
          throws Exception {
    logger.info(logPrefix + " Calling load_table_binary_columnar_polys API");
    List<List<Object>> columns = new ArrayList<>();
    for (int i = 0; i < 2; i++) {
      columns.add(new ArrayList<>());
    }
    for (int i = 0; i < 5; i++) {
      columns.get(0).add(Long.valueOf(i));
      columns.get(1).add("MULTIPOLYGON(((0 0,0 9,9 9,9 0),(2 2,1 1,3 3)))");
    }
    user.load_table_binary_columnar_polys(geoTableName, columns, new ArrayList<>());
  }

  private void getTableDetails(MapdTestClient user, String logPrefix) throws Exception {
    logger.info(logPrefix + " Calling get_table_details API");
    user.get_table_details(csvTableName);
    logger.info(logPrefix + " Calling get_table_details_for_database API");
    // Concurrent request to drop table may have occurred when this query is
    // executed. Ignore the error response in this case.
    ignoreMissingTable(
            ()
                    -> user.get_table_details_for_database(geoTableName, "omnisci"),
            geoTableName);
  }

  private void getTablesMetadata(MapdTestClient user, String logPrefix) throws Exception {
    logger.info(logPrefix + " Calling get_tables_meta API");
    user.get_tables_meta();
  }
}
