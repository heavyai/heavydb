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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.concurrent.CyclicBarrier;

public class ImportAlterValidateSelectConcurrencyTest {
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
    final String csvTableName = "import_test_mixed_varlen";
    final String geoTableName = "geospatial";
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
                  + "( trip INT, omnisci_geo MULTIPOLYGON ) WITH(FRAGMENT_SIZE = "
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

          try {
            barrier.await();

            MapdTestClient user =
                    MapdTestClient.getClient("localhost", 6274, db, dbUser, dbPassword);

            if (tid % 2 == 0) {
              logger.info(logPrefix + " IMPORT TABLE");
              user.import_table(csvTableName, csv_file_path, copy_params);
            }

            sql = "DELETE FROM " + csvTableName + " WHERE fatx2 IS NULL;";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

            sql = "SELECT COUNT(*) FROM " + csvTableName + ";";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

            if (threadId == 1) {
              Thread.sleep(5000); // Ensure import is launched
              sql = "ALTER TABLE " + csvTableName + " DROP COLUMN faii;";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);
            }

            // TODO(adb): add get_table_details once thread safe

            sql = "SELECT * FROM " + geoTableName + ";";
            logger.info(logPrefix + " VALIDATE " + sql);
            user.sqlValidate(sql);

            if (threadId == 3) {
              logger.info(logPrefix + " IMPORT GEO TABLE");
              user.import_geo_table(geoTableName,
                      geo_file_path,
                      copy_params,
                      new java.util.ArrayList<TColumnType>(),
                      new TCreateParams());
            }

            sql = "SELECT * FROM " + geoTableName + " LIMIT 2;";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

            sql = "SELECT * FROM " + csvTableName + ";";
            logger.info(logPrefix + " VALIDATE " + sql);
            user.sqlValidate(sql);

            for (int i = 0; i < 5; i++) {
              sql = "INSERT INTO " + geoTableName + " VALUES (" + i
                      + ", 'MULTIPOLYGON(((0 0, 1 1, 2 2)))');";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);
            }

            sql = "TRUNCATE TABLE " + csvTableName + ";";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

            sql = "SELECT COUNT(*) FROM " + csvTableName + ";";
            logger.info(logPrefix + " VALIDATE " + sql);
            user.sqlValidate(sql);

            if (threadId == 0) {
              Thread.sleep(5000); // Ensure import is launched
              sql = "DROP TABLE " + geoTableName + ";";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);
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

    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER dba;");

    logger.info("ImportAlterValidateSelectConcurrencyTest() done");
  }
}
