/*
 * Copyright 2022 HEAVY.AI, Inc.
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
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ThreadLocalRandom;

public class SystemTableConcurrencyTest {
  final static Logger logger = LoggerFactory.getLogger(SystemTableConcurrencyTest.class);

  public static void main(String[] args) throws Exception {
    SystemTableConcurrencyTest test = new SystemTableConcurrencyTest();
    test.testConcurrency();
    test.testMemorySystemTablesConcurrency();
  }

  public void testConcurrency() throws Exception {
    logger.info("SystemTableConcurrencyTest()");
    runTest();
    logger.info("SystemTableConcurrencyTest() done");
  }

  private void runTest() throws Exception {
    List<ThreadDbQueries> queriesPerThread = new ArrayList<ThreadDbQueries>(Arrays.asList(
            new ThreadDbQueries("heavyai",
                    Arrays.asList("CREATE USER user_1 (password = 'HyperInteractive');",
                            "ALTER USER user_1 (password = 'HyperInteractive2');",
                            "DROP USER user_1;",
                            "CREATE USER user_2 (password = 'HyperInteractive');",
                            "ALTER USER user_2 (password = 'HyperInteractive2');",
                            "DROP USER user_2;")),
            new ThreadDbQueries("heavyai",
                    Arrays.asList("CREATE USER user_3 (password = 'HyperInteractive');",
                            "GRANT SELECT ON DATABASE heavyai TO user_3;",
                            "REVOKE SELECT ON DATABASE heavyai FROM user_3;",
                            "GRANT CREATE ON DATABASE heavyai TO user_3;",
                            "REVOKE CREATE ON DATABASE heavyai FROM user_3;",
                            "DROP USER user_3;")),
            new ThreadDbQueries("heavyai",
                    Arrays.asList("CREATE DATABASE db_1;",
                            "CREATE DATABASE db_2;",
                            "DROP DATABASE db_1;",
                            "DROP DATABASE db_2;")),
            new ThreadDbQueries("heavyai",
                    Arrays.asList("CREATE ROLE role_1;",
                            "CREATE ROLE role_2;",
                            "DROP ROLE role_1;",
                            "DROP ROLE role_2;")),
            new ThreadDbQueries("heavyai",
                    Arrays.asList("CREATE TABLE table_1 (i INTEGER, t TEXT);",
                            "CREATE TABLE table_2 (i INTEGER[], pt POINT);",
                            "INSERT INTO table_1 VALUES (1, 'abc');",
                            "INSERT INTO table_2 VALUES ({1,2}, 'POINT (1 1)');",
                            "SELECT AVG(i) FROM table_1;",
                            "CREATE VIEW view_1 AS SELECT * FROM table_1;",
                            "COMMENT ON TABLE table_1 IS 'this is test table#1';",
                            "COMMENT ON TABLE table_2 IS 'this is test table#2';",
                            "COMMENT ON COLUMN table_1.t IS 'this is a test column in test table#1';",
                            "COMMENT ON COLUMN table_2.pt IS 'this is a test column in test table#2';",
                            "SELECT * FROM view_1;",
                            "DROP VIEW view_1;",
                            "DROP TABLE table_1;",
                            "DROP TABLE table_2;")),
            new ThreadDbQueries("heavyai",
                    Arrays.asList("CREATE USER user_4 (password = 'HyperInteractive');",
                            "CREATE USER user_5 (password = 'HyperInteractive');",
                            "CREATE ROLE role_3;",
                            "CREATE ROLE role_4;",
                            "GRANT role_3, role_4 TO user_4, user_5;",
                            "REVOKE role_3 FROM user_5;",
                            "REVOKE role_4 FROM user_4, user_5;",
                            "REVOKE role_3 FROM user_4;",
                            "DROP USER user_4;",
                            "DROP USER user_5;",
                            "DROP ROLE role_3;",
                            "DROP ROLE role_4;"))));
    final List<String> systemTableQueries = Arrays.asList("SELECT * FROM users;",
            "SELECT * FROM permissions;",
            "SELECT * FROM databases;",
            "SELECT * FROM roles;",
            "SELECT * FROM tables;",
            "SELECT * FROM columns;",
            "SELECT * FROM role_assignments;",
            "SELECT * FROM dashboards;",
            "SELECT * FROM memory_summary;",
            "SELECT * FROM memory_details;",
            "SELECT * FROM storage_details;");

    for (int i = 0; i < systemTableQueries.size(); i++) {
      // Run queries for the same system table in parallel.
      final int parallelQueryCount = 5;
      for (int j = 0; j < parallelQueryCount; j++) {
        queriesPerThread.add(new ThreadDbQueries(
                "information_schema", Arrays.asList(systemTableQueries.get(i))));
      }
    }

    final int num_threads = queriesPerThread.size()
            + 1; // +1 for dashboard creation/update thread, which is created separately.
    Exception[] exceptions = new Exception[num_threads];
    Thread[] threads = new Thread[num_threads];

    // Use a barrier here to synchronize the start of each competing thread and make
    // sure there is a race condition. The barrier ensures no thread will run until they
    // are all created and waiting for the barrier to complete.
    final CyclicBarrier barrier = new CyclicBarrier(
            num_threads, () -> { logger.info("Barrier acquired. Starting queries..."); });

    threads[0] = new Thread(() -> {
      try {
        logger.info("Starting thread[0]");
        HeavyDBTestClient user = HeavyDBTestClient.getClient(
                "localhost", 6274, "heavyai", "admin", "HyperInteractive");
        barrier.await();
        logger.info("0 create dashboard \"dashboard_1\"");
        int dashboardId = user.create_dashboard("dashboard_1");
        logger.info("0 get dashboard " + dashboardId);
        user.get_dashboard(dashboardId);
        logger.info("0 replace dashboard " + dashboardId);
        user.replace_dashboard(dashboardId, "dashboard_2", "admin");
        logger.info("0 delete dashboard " + dashboardId);
        user.delete_dashboard(dashboardId);
        logger.info("Finished thread[0]");
      } catch (Exception e) {
        logger.error("Thread[0] Caught Exception: " + e.getMessage(), e);
        exceptions[0] = e;
      }
    });
    threads[0].start();

    for (int i = 0; i < queriesPerThread.size(); i++) {
      final ThreadDbQueries threadQueries = queriesPerThread.get(i);
      final int threadId = i + 1;
      threads[threadId] = new Thread(() -> {
        try {
          logger.info("Starting thread[" + threadId + "]");
          HeavyDBTestClient user = HeavyDBTestClient.getClient(
                  "localhost", 6274, threadQueries.database, "admin", "HyperInteractive");
          barrier.await();

          final int repeatQueryCount = 10;
          for (int irepeat = 0; irepeat < repeatQueryCount; ++irepeat) {
            for (final String query : threadQueries.queries) {
              runSqlAsUser(query, user, threadId);
              Thread.sleep(ThreadLocalRandom.current().nextInt(
                      0, 1000 + 1)); // Sleep up to one second randomly to allow random
                                     // interleaving of queries between threads
            }
          }
          logger.info("Finished thread[" + threadId + "]");
        } catch (Exception e) {
          logger.error("Thread[" + threadId + "] Caught Exception: " + e.getMessage(), e);
          exceptions[threadId] = e;
        }
      });
      threads[threadId].start();
    }

    for (Thread t : threads) {
      t.join();
    }

    for (Exception e : exceptions) {
      if (e != null) {
        logger.error("Exception: " + e.getMessage(), e);
        throw e;
      }
    }
  }

  public void testMemorySystemTablesConcurrency() throws Exception {
    logger.info("Starting Memory System Tables Concurrency Test");

    Thread[] threads = new Thread[2];
    final int memorySummaryThreadId = 0, memoryDetailsThreadId = 1;
    threads[memorySummaryThreadId] = new Thread(
            () -> { runSystemTableQueries("memory_summary", memorySummaryThreadId); });

    threads[memoryDetailsThreadId] = new Thread(
            () -> { runSystemTableQueries("memory_details", memoryDetailsThreadId); });

    for (Thread thread : threads) {
      thread.start();
    }

    for (Thread thread : threads) {
      thread.join();
    }

    logger.info("Completed Memory System Tables Concurrency Test");
  }

  private void runSystemTableQueries(final String tableName, final int threadId) {
    try {
      logger.info("Starting thread[" + threadId + "]");
      HeavyDBTestClient user = HeavyDBTestClient.getClient(
              "localhost", 6274, "information_schema", "admin", "HyperInteractive");

      final int repeatQueryCount = 50;
      for (int i = 0; i < repeatQueryCount; i++) {
        runSqlAsUser(String.format("SELECT * FROM %s;", tableName), user, threadId);
      }
      logger.info("Finished thread[" + threadId + "]");
    } catch (Exception e) {
      logger.error("Thread[" + threadId + "] Caught Exception: " + e.getMessage(), e);
      throw new RuntimeException(e);
    }
  }

  private void runSqlAsUser(String sql, HeavyDBTestClient user, int threadId)
          throws Exception {
    logger.info(threadId + " " + sql);
    user.runSql(sql);
  }

  private class ThreadDbQueries {
    public ThreadDbQueries(final String database, final List<String> queries) {
      this.database = database;
      this.queries = queries;
    }

    public final String database;
    public final List<String> queries;
  }
}
