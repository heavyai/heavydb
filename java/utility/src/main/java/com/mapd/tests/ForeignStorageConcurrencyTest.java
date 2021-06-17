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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.CyclicBarrier;

public class ForeignStorageConcurrencyTest {
  final static Logger logger =
          LoggerFactory.getLogger(ForeignStorageConcurrencyTest.class);

  public static void main(String[] args) throws Exception {
    ForeignStorageConcurrencyTest test = new ForeignStorageConcurrencyTest();
    test.testConcurrency();
  }

  // This test creates 4 threads that all perform a series of common projections on a
  // table while attempting to modify different columns at different points
  // simultaneously.  The test should resolve without exception as each thread waits on
  // another to finish the alteration until it can complete it's own function.
  private void runTest(String db,
          String adminName,
          String adminPassword,
          String userName,
          String userPassword) throws Exception {
    final int num_threads = 4;
    ArrayList<Exception> exceptions = new ArrayList<Exception>();

    // We use a barrier here to synchronize the start of each competing thread and make
    // sure there is a race condition.  The barrier ensures no thread will run until they
    // are all created and waiting for the barrier to complete.
    final CyclicBarrier barrier = new CyclicBarrier(num_threads, new Runnable() {
      public void run() {
        try {
          logger.info("Barrier acquired.");
          MapdTestClient dba = MapdTestClient.getClient(
                  "localhost", 6274, db, adminName, adminPassword);

          dba.runSql("CREATE SERVER test_server "
                  + "FOREIGN DATA WRAPPER omnisci_csv WITH (storage_type = 'LOCAL_FILE', "
                  + "base_path = '" + System.getProperty("user.dir") + "');");

          dba.runSql("CREATE FOREIGN TABLE test_table "
                  + "(b BOOLEAN, t TINYINT, s SMALLINT, i INTEGER, bi BIGINT, f FLOAT, "
                  + "dc DECIMAL(10, 5), tm TIME, tp TIMESTAMP, d DATE, txt TEXT, "
                  + "txt_2 TEXT ENCODING NONE) "
                  + "SERVER test_server WITH "
                  + "(file_path = '../Tests/FsiDataFiles/scalar_types.csv', "
                  + "FRAGMENT_SIZE = 10);");
          logger.info("Barrier released.");
        } catch (Exception e) {
          logger.error("Barrier Caught Exception: " + e.getMessage(), e);
          exceptions.add(e);
        }
      }
    });

    // Each thread performs the same projections with an alter command targeting a
    // different column injected at a different point.
    Thread[] threads = new Thread[num_threads];

    threads[0] = new Thread(new Runnable() {
      @Override
      public void run() {
        try {
          logger.info("Starting thread[0]");
          MapdTestClient user =
                  MapdTestClient.getClient("localhost", 6274, db, userName, userPassword);
          barrier.await();
          runSqlAsUser(
                  "ALTER FOREIGN TABLE test_table RENAME COLUMN t TO tint", user, "0");
          runSqlAsUser("SELECT * FROM test_table LIMIT 2", user, "0");
          runSqlAsUser("SELECT * FROM test_table WHERE txt = 'quoted text'", user, "0");
          runSqlValidateAsUser("SELECT COUNT(*) FROM test_table LIMIT 2", user, "0");
          logger.info("Finished thread[0]");
        } catch (Exception e) {
          logger.error("Thread[0] Caught Exception: " + e.getMessage(), e);
          exceptions.add(e);
        }
      }
    });
    threads[0].start();

    threads[1] = new Thread(new Runnable() {
      @Override
      public void run() {
        try {
          logger.info("Starting thread[1]");
          MapdTestClient user =
                  MapdTestClient.getClient("localhost", 6274, db, userName, userPassword);
          barrier.await();
          runSqlAsUser("SELECT * FROM test_table LIMIT 2", user, "1");
          runSqlAsUser(
                  "ALTER FOREIGN TABLE test_table RENAME COLUMN s TO sint", user, "1");
          runSqlAsUser("SELECT * FROM test_table WHERE txt = 'quoted text'", user, "1");
          runSqlValidateAsUser("SELECT COUNT(*) FROM test_table LIMIT 2", user, "1");
          logger.info("Finished thread[1]");
        } catch (Exception e) {
          logger.error("Thread[1] Caught Exception: " + e.getMessage(), e);
          exceptions.add(e);
        }
      }
    });
    threads[1].start();

    threads[2] = new Thread(new Runnable() {
      @Override
      public void run() {
        try {
          logger.info("Starting thread[2]");
          MapdTestClient user =
                  MapdTestClient.getClient("localhost", 6274, db, userName, userPassword);
          barrier.await();
          runSqlAsUser("SELECT * FROM test_table LIMIT 2", user, "2");
          runSqlAsUser("SELECT * FROM test_table WHERE txt = 'quoted text'", user, "2");
          runSqlAsUser(
                  "ALTER FOREIGN TABLE test_table RENAME COLUMN i TO iint", user, "2");
          runSqlValidateAsUser("SELECT COUNT(*) FROM test_table LIMIT 2", user, "2");
          logger.info("Finished thread[2]");
        } catch (Exception e) {
          logger.error("Thread[0] Caught Exception: " + e.getMessage(), e);
          exceptions.add(e);
        }
      }
    });
    threads[2].start();

    threads[3] = new Thread(new Runnable() {
      @Override
      public void run() {
        try {
          logger.info("Starting thread[3]");
          MapdTestClient user =
                  MapdTestClient.getClient("localhost", 6274, db, userName, userPassword);
          barrier.await();
          runSqlAsUser("SELECT * FROM test_table LIMIT 2", user, "3");
          runSqlAsUser("SELECT * FROM test_table WHERE txt = 'quoted text'", user, "3");
          runSqlValidateAsUser("SELECT COUNT(*) FROM test_table LIMIT 2", user, "3");
          runSqlAsUser(
                  "ALTER FOREIGN TABLE test_table RENAME COLUMN b TO bint", user, "3");
          logger.info("Finished thread[3]");
        } catch (Exception e) {
          logger.error("Thread[0] Caught Exception: " + e.getMessage(), e);
          exceptions.add(e);
        }
      }
    });
    threads[3].start();

    for (Thread t : threads) {
      t.join();
    }

    MapdTestClient dba =
            MapdTestClient.getClient("localhost", 6274, db, adminName, adminPassword);
    dba.runSql("DROP FOREIGN TABLE test_table;");
    dba.runSql("DROP SERVER test_server;");

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

  public void runSqlValidateAsUser(String sql, MapdTestClient user, String logPrefix)
          throws Exception {
    logger.info(logPrefix + " " + sql);
    user.sqlValidate(sql);
  }

  public void testConcurrency() throws Exception {
    logger.info("ForeignStorageConcurrencyTest()");

    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "omnisci", "admin", "HyperInteractive");

    // Initialize.
    su.runSql("DROP DATABASE IF EXISTS db1;");
    su.runSql("CREATE DATABASE db1;");

    runTest("db1", "admin", "HyperInteractive", "admin", "HyperInteractive");

    // Cleanup.
    su.runSql("DROP DATABASE db1;");

    logger.info("ForeignStorageConcurrencyTest() done");
  }
}
