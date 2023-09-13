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

import java.util.*;

import ai.heavy.thrift.server.*;

public class CatalogConcurrencyTest {
  final static String defPwd = "HyperInteractive", local = "localhost", defDb = "heavyai",
                      admin = "admin";
  final static int port = 6274;
  final static Logger logger = LoggerFactory.getLogger(CatalogConcurrencyTest.class);

  public static void main(String[] args) throws Exception {
    CatalogConcurrencyTest test = new CatalogConcurrencyTest();
    test.testCatalogConcurrency();
  }

  private void run_test(HeavyDBTestClient dba,
          HeavyDBTestClient user,
          String prefix,
          int max,
          List<Integer> dashboardIds) throws Exception {
    final String sharedTableName = "table_shared";
    for (int i = 0; i < max; i++) {
      final long tid = Thread.currentThread().getId();
      final String threadPrefix = "[" + tid + "] ",
                   tableName = "table_" + prefix + "_" + i,
                   viewName = "view_" + prefix + "_" + i,
                   dashName = "dash_" + prefix + "_" + i;

      // Modify the fixed id dashboards in parallel.
      for (int id : dashboardIds) {
        TDashboard board = dba.get_dashboard(id);
        logger.info("REPLACE DASHBOARD id (" + id + ") " + board.dashboard_name);
        dba.replace_dashboard(board.dashboard_id, board.dashboard_name + "_", admin);
      }

      logger.info(threadPrefix + "CREATE TABLE " + tableName);
      user.runSql("CREATE TABLE " + tableName + " (id text);");
      HeavyDBAsserts.assertEqual(true, null != dba.get_table_details(tableName));
      logger.info(threadPrefix + "INSERT INTO " + tableName);
      user.runSql("INSERT INTO " + tableName + " VALUES(1);");
      dba.runSql("GRANT SELECT ON TABLE " + tableName + " TO bob;");

      logger.info(threadPrefix + "CREATE VIEW " + viewName);
      user.runSql("CREATE VIEW " + viewName + " AS SELECT * FROM " + tableName + ";");
      HeavyDBAsserts.assertEqual(true, null != dba.get_table_details(viewName));
      dba.runSql("GRANT SELECT ON VIEW " + viewName + " TO bob;");

      logger.info(threadPrefix + "CREATE DASHBOARD " + dashName);
      int dash_id = user.create_dashboard(dashName);
      HeavyDBAsserts.assertEqual(true, null != dba.get_dashboard(dash_id));
      dba.runSql("GRANT VIEW ON DASHBOARD " + dash_id + " TO bob;");

      dba.runSql("REVOKE VIEW ON DASHBOARD " + dash_id + " FROM bob;");
      dba.runSql("REVOKE SELECT ON VIEW " + viewName + " FROM bob;");
      dba.runSql("REVOKE SELECT ON TABLE " + tableName + " FROM bob;");

      logger.info(threadPrefix + "DELETE DASHBOARD " + dashName);
      dba.delete_dashboard(dash_id);
      logger.info(threadPrefix + "DROP VIEW " + viewName);
      dba.runSql("DROP VIEW " + viewName + ";");
      logger.info(threadPrefix + "DROP TABLE " + tableName);
      dba.runSql("DROP TABLE " + tableName + ";");

      logger.info(threadPrefix + "CREATE IF NOT EXISTS " + sharedTableName);
      dba.runSql("CREATE TABLE IF NOT EXISTS " + sharedTableName + " (id INTEGER);");

      logger.info(threadPrefix + "DROP IF EXISTS " + sharedTableName);
      dba.runSql("DROP TABLE IF EXISTS " + sharedTableName + ";");
    }
  }

  private void runTest(String db,
          String dbaUser,
          String dbaPassword,
          String dbUser,
          String dbPassword,
          List<Integer> dashboardIds) throws Exception {
    final int num_threads = 5, runs = 25;
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
            HeavyDBTestClient dba =
                    HeavyDBTestClient.getClient(local, port, db, dbaUser, dbaPassword);
            HeavyDBTestClient user =
                    HeavyDBTestClient.getClient(local, port, db, dbUser, dbPassword);
            run_test(dba, user, prefix, runs, dashboardIds);
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

    HeavyDBTestClient su = HeavyDBTestClient.getClient(local, port, defDb, admin, defPwd);

    su.runSql("DROP USER IF EXISTS bob;");
    su.runSql("DROP USER IF EXISTS dba;");
    su.runSql("DROP DATABASE IF EXISTS db1;");

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

    // We create a series of dashboards with fixed ids to be modified in parallel.
    HeavyDBTestClient dba =
            HeavyDBTestClient.getClient(local, port, "db1", admin, defPwd);
    for (TDashboard board : dba.get_dashboards()) {
      logger.info("DROP DASHBOARD " + board.dashboard_name);
      dba.delete_dashboard(board.dashboard_id);
    }
    ArrayList<Integer> dashboardIds = new ArrayList<>();
    for (int i = 0; i < 5; ++i) {
      String dashName = "dash_" + i;
      logger.info("CREATE DASHBOARD " + dashName);
      dashboardIds.add(dba.create_dashboard(dashName));
    }
    HeavyDBAsserts.assertEqual(5, dba.get_dashboards().size());

    runTest("db1", admin, defPwd, admin, defPwd, dashboardIds);
    runTest("db1", admin, defPwd, "dba", "password", dashboardIds);
    runTest("db1", admin, defPwd, "bob", "password", dashboardIds);
    runTest("db1", "dba", "password", admin, defPwd, dashboardIds);
    runTest("db1", "dba", "password", "bob", "password", dashboardIds);

    runTest(defDb, admin, defPwd, admin, defPwd, dashboardIds);
    runTest(defDb, admin, defPwd, "dba", "password", dashboardIds);
    runTest(defDb, admin, defPwd, "bob", "password", dashboardIds);
    runTest(defDb, "dba", "password", admin, defPwd, dashboardIds);
    runTest(defDb, "dba", "password", "bob", "password", dashboardIds);

    for (TDashboard board : dba.get_dashboards()) {
      logger.info("DROP DASHBOARD " + board.dashboard_name);
      dba.delete_dashboard(board.dashboard_id);
    }
    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER dba;");

    logger.info("testCatalogConcurrency() done");
  }
}
