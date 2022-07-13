/*
 * Copyright 2015 The Apache Software Foundation.
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

public class EagainConcurrencyTest {
  final static Logger logger = LoggerFactory.getLogger(EagainConcurrencyTest.class);

  public static void main(String[] args) throws Exception {
    EagainConcurrencyTest test = new EagainConcurrencyTest();
    test.testCatalogConcurrency();
  }

  private void run_test(HeavyDBTestClient dba,
          String db,
          String dbUser,
          String dbPassword,
          String prefix,
          int max) throws Exception {
    String tableName = "table_" + prefix + "_";
    String viewName = "view_" + prefix + "_";
    long tid = Thread.currentThread().getId();

    logger.info("[" + tid + "]"
            + "FIRST USER CONNECT");
    HeavyDBTestClient first_user =
            HeavyDBTestClient.getClient("localhost", 6274, db, dbUser, dbPassword);

    logger.info("[" + tid + "]"
            + "CREATE " + tableName);
    first_user.runSql("CREATE TABLE " + tableName + " (id integer);");

    logger.info("[" + tid + "]"
            + "CREATE VIEW " + viewName);
    first_user.runSql(
            "CREATE VIEW " + viewName + " AS (SELECT id * 2 FROM " + tableName + ");");

    logger.info("[" + tid + "]"
            + "FIRST USER DISCONNECT");
    first_user.disconnect();

    for (int i = 0; i < max; i++) {
      logger.info("[" + tid + "]"
              + "USER CONNECT");
      HeavyDBTestClient user =
              HeavyDBTestClient.getClient("localhost", 6274, db, dbUser, dbPassword);

      logger.info("[" + tid + "]"
              + "INSERT INTO " + tableName);
      user.runSql("INSERT INTO " + tableName + " VALUES (" + i + ");");

      user.get_status();
      user.get_server_status();
      user.get_hardware_info();

      logger.info("[" + tid + "]"
              + "SELECT FROM " + tableName);
      user.runSql("SELECT * FROM " + tableName + ";");

      user.get_memory("cpu");
      user.get_dashboards();
      //       user.get_tables_meta(); // TODO(adb): re-enable after fixing up catalog/db
      //       locking

      logger.info("[" + tid + "]"
              + "USER DISCONNECT");
      user.disconnect();
    }

    logger.info("[" + tid + "]"
            + "DROP " + viewName);
    dba.runSql("DROP VIEW " + viewName + ";");

    logger.info("[" + tid + "]"
            + "DROP " + tableName);
    dba.runSql("DROP TABLE " + tableName + ";");
  }

  private void runTest(
          String db, String dbaUser, String dbaPassword, String dbUser, String dbPassword)
          throws Exception {
    int num_threads = 5;
    final int runs = 25;
    Exception exceptions[] = new Exception[num_threads];

    ArrayList<Thread> threads = new ArrayList<>();
    for (int i = 0; i < num_threads; i++) {
      logger.info("Starting " + i);
      final String prefix = "for_bob_" + i + "_";
      final int threadId = i;
      Thread t = new Thread(new Runnable() {
        @Override
        public void run() {
          try {
            HeavyDBTestClient dba = HeavyDBTestClient.getClient(
                    "localhost", 6274, db, dbaUser, dbaPassword);
            run_test(dba, db, dbUser, dbPassword, prefix, runs);
          } catch (Exception e) {
            logger.error("[" + Thread.currentThread().getId() + "]"
                            + "Caught Exception: " + e.getMessage(),
                    e);
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

    for (Exception e : exceptions) {
      if (null != e) {
        logger.error("Exception: " + e.getMessage(), e);
        throw new Exception(e.getMessage(), e);
      }
    }
  }

  public void testCatalogConcurrency() throws Exception {
    logger.info("testCatalogConcurrency()");

    HeavyDBTestClient su = HeavyDBTestClient.getClient(
            "localhost", 6274, "heavyai", "admin", "HyperInteractive");
    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");

    su.runSql("GRANT CREATE on DATABASE heavyai TO bob;");
    su.runSql("GRANT CREATE VIEW on DATABASE heavyai TO bob;");
    su.runSql("GRANT CREATE DASHBOARD on DATABASE heavyai TO bob;");

    su.runSql("GRANT DROP on DATABASE heavyai TO bob;");
    su.runSql("GRANT DROP VIEW on DATABASE heavyai TO bob;");
    su.runSql("GRANT DELETE DASHBOARD on DATABASE heavyai TO bob;");

    su.runSql("CREATE DATABASE db1;");

    su.runSql("GRANT CREATE on DATABASE db1 TO bob;");
    su.runSql("GRANT CREATE VIEW on DATABASE db1 TO bob;");
    su.runSql("GRANT CREATE DASHBOARD on DATABASE db1 TO bob;");

    su.runSql("GRANT DROP on DATABASE db1 TO bob;");
    su.runSql("GRANT DROP VIEW on DATABASE db1 TO bob;");
    su.runSql("GRANT DELETE DASHBOARD on DATABASE db1 TO bob;");

    su.runSql("GRANT ACCESS on database heavyai TO dba;");
    su.runSql("GRANT ACCESS on database heavyai TO bob;");
    su.runSql("GRANT ACCESS on database db1 TO dba;");
    su.runSql("GRANT ACCESS on database db1 TO bob;");

    runTest("db1", "admin", "HyperInteractive", "admin", "HyperInteractive");
    runTest("db1", "admin", "HyperInteractive", "dba", "password");
    runTest("db1", "admin", "HyperInteractive", "bob", "password");
    runTest("db1", "dba", "password", "admin", "HyperInteractive");
    runTest("db1", "dba", "password", "bob", "password");

    runTest("heavyai", "admin", "HyperInteractive", "admin", "HyperInteractive");
    runTest("heavyai", "admin", "HyperInteractive", "dba", "password");
    runTest("heavyai", "admin", "HyperInteractive", "bob", "password");
    runTest("heavyai", "dba", "password", "admin", "HyperInteractive");
    runTest("heavyai", "dba", "password", "bob", "password");

    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER dba;");

    logger.info("testCatalogConcurrency() done");
  }
}
