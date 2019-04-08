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

import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CatalogConcurrencyTest {
  final static Logger logger = LoggerFactory.getLogger(CatalogConcurrencyTest.class);

  public static void main(String[] args) throws Exception {
    CatalogConcurrencyTest test = new CatalogConcurrencyTest();
    test.testCatalogConcurrency();
  }

  private void run_test(MapdTestClient dba, MapdTestClient user, String prefix, int max)
          throws Exception {
    for (int i = 0; i < max; i++) {
      String tableName = "table_" + prefix + "_" + i;
      String viewName = "view_" + prefix + "_" + i;
      String dashName = "dash_" + prefix + "_" + i;
      long tid = Thread.currentThread().getId();

      logger.info("[" + tid + "]"
              + "CREATE " + tableName);
      user.runSql("CREATE TABLE " + tableName + " (id integer);");
      MapdAsserts.assertEqual(true, null != dba.get_table_details(tableName));
      dba.runSql("GRANT SELECT ON TABLE " + tableName + " TO bob;");

      logger.info("[" + tid + "]"
              + "CREATE " + viewName);
      user.runSql("CREATE VIEW " + viewName + " AS SELECT * FROM " + tableName + ";");
      MapdAsserts.assertEqual(true, null != dba.get_table_details(viewName));
      dba.runSql("GRANT SELECT ON VIEW " + viewName + " TO bob;");

      logger.info("[" + tid + "]"
              + "CREATE " + dashName);
      int dash_id = user.create_dashboard(dashName);
      MapdAsserts.assertEqual(true, null != dba.get_dashboard(dash_id));
      dba.runSql("GRANT VIEW ON DASHBOARD " + dash_id + " TO bob;");

      dba.runSql("REVOKE VIEW ON DASHBOARD " + dash_id + " FROM bob;");
      dba.runSql("REVOKE SELECT ON VIEW " + viewName + " FROM bob;");
      dba.runSql("REVOKE SELECT ON TABLE " + tableName + " FROM bob;");

      logger.info("[" + tid + "]"
              + "DROP " + dashName);
      dba.delete_dashboard(dash_id);
      logger.info("[" + tid + "]"
              + "DROP " + viewName);
      dba.runSql("DROP VIEW " + viewName + ";");
      logger.info("[" + tid + "]"
              + "DROP " + tableName);
      dba.runSql("DROP TABLE " + tableName + ";");
    }
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
            MapdTestClient dba =
                    MapdTestClient.getClient("localhost", 6274, db, dbaUser, dbaPassword);
            MapdTestClient user =
                    MapdTestClient.getClient("localhost", 6274, db, dbUser, dbPassword);
            run_test(dba, user, prefix, runs);
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

    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "mapd", "mapd", "HyperInteractive");
    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");

    su.runSql("GRANT CREATE on DATABASE mapd TO bob;");
    su.runSql("GRANT CREATE VIEW on DATABASE mapd TO bob;");
    su.runSql("GRANT CREATE DASHBOARD on DATABASE mapd TO bob;");

    su.runSql("GRANT DROP on DATABASE mapd TO bob;");
    su.runSql("GRANT DROP VIEW on DATABASE mapd TO bob;");
    su.runSql("GRANT DELETE DASHBOARD on DATABASE mapd TO bob;");

    su.runSql("CREATE DATABASE db1;");

    su.runSql("GRANT CREATE on DATABASE db1 TO bob;");
    su.runSql("GRANT CREATE VIEW on DATABASE db1 TO bob;");
    su.runSql("GRANT CREATE DASHBOARD on DATABASE db1 TO bob;");

    su.runSql("GRANT DROP on DATABASE db1 TO bob;");
    su.runSql("GRANT DROP VIEW on DATABASE db1 TO bob;");
    su.runSql("GRANT DELETE DASHBOARD on DATABASE db1 TO bob;");

    su.runSql("GRANT ACCESS on database mapd TO dba;");
    su.runSql("GRANT ACCESS on database mapd TO bob;");
    su.runSql("GRANT ACCESS on database db1 TO dba;");
    su.runSql("GRANT ACCESS on database db1 TO bob;");

    runTest("db1", "mapd", "HyperInteractive", "mapd", "HyperInteractive");
    runTest("db1", "mapd", "HyperInteractive", "dba", "password");
    runTest("db1", "mapd", "HyperInteractive", "bob", "password");
    runTest("db1", "dba", "password", "mapd", "HyperInteractive");
    runTest("db1", "dba", "password", "bob", "password");

    runTest("mapd", "mapd", "HyperInteractive", "mapd", "HyperInteractive");
    runTest("mapd", "mapd", "HyperInteractive", "dba", "password");
    runTest("mapd", "mapd", "HyperInteractive", "bob", "password");
    runTest("mapd", "dba", "password", "mapd", "HyperInteractive");
    runTest("mapd", "dba", "password", "bob", "password");

    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER dba;");
  }
}
