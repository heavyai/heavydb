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

import java.util.ArrayList;
import java.util.concurrent.CyclicBarrier;

public class SystemTableConcurrencyTest {
  final static Logger logger = LoggerFactory.getLogger(SystemTableConcurrencyTest.class);

  public static void main(String[] args) throws Exception {
    SystemTableConcurrencyTest test = new SystemTableConcurrencyTest();
    test.testConcurrency();
  }

  public void testConcurrency() throws Exception {
    logger.info("SystemTableConcurrencyTest()");
    runTest();
    logger.info("SystemTableConcurrencyTest() done");
  }

  private void runTest() throws Exception {
    final int num_threads = 8;
    ArrayList<Exception> exceptions = new ArrayList<Exception>();

    // Use a barrier here to synchronize the start of each competing thread and make
    // sure there is a race condition. The barrier ensures no thread will run until they
    // are all created and waiting for the barrier to complete.
    final CyclicBarrier barrier = new CyclicBarrier(
            num_threads, () -> { logger.info("Barrier acquired. Starting queries..."); });

    Thread[] threads = new Thread[num_threads];
    threads[0] = new Thread(() -> {
      try {
        logger.info("Starting thread[0]");
        MapdTestClient user = MapdTestClient.getClient(
                "localhost", 6274, "omnisci", "admin", "HyperInteractive");
        barrier.await();
        runSqlAsUser("CREATE USER user_1 (password = 'HyperInteractive');", user, "0");
        runSqlAsUser("ALTER USER user_1 (password = 'HyperInteractive2');", user, "0");
        runSqlAsUser("DROP USER user_1;", user, "0");
        runSqlAsUser("CREATE USER user_2 (password = 'HyperInteractive');", user, "0");
        runSqlAsUser("ALTER USER user_2 (password = 'HyperInteractive2');", user, "0");
        runSqlAsUser("DROP USER user_2;", user, "0");
        logger.info("Finished thread[0]");
      } catch (Exception e) {
        logger.error("Thread[0] Caught Exception: " + e.getMessage(), e);
        exceptions.add(e);
      }
    });
    threads[0].start();

    threads[1] = new Thread(() -> {
      try {
        logger.info("Starting thread[1]");
        MapdTestClient user = MapdTestClient.getClient(
                "localhost", 6274, "omnisci", "admin", "HyperInteractive");
        barrier.await();
        runSqlAsUser("CREATE USER user_3 (password = 'HyperInteractive');", user, "1");
        runSqlAsUser("GRANT SELECT ON DATABASE omnisci TO user_3;", user, "1");
        runSqlAsUser("REVOKE SELECT ON DATABASE omnisci FROM user_3;", user, "1");
        runSqlAsUser("GRANT CREATE ON DATABASE omnisci TO user_3;", user, "1");
        runSqlAsUser("REVOKE CREATE ON DATABASE omnisci FROM user_3;", user, "1");
        runSqlAsUser("DROP USER user_3;", user, "1");
        logger.info("Finished thread[1]");
      } catch (Exception e) {
        logger.error("Thread[1] Caught Exception: " + e.getMessage(), e);
        exceptions.add(e);
      }
    });
    threads[1].start();

    threads[2] = new Thread(() -> {
      try {
        logger.info("Starting thread[2]");
        MapdTestClient user = MapdTestClient.getClient(
                "localhost", 6274, "omnisci", "admin", "HyperInteractive");
        barrier.await();
        runSqlAsUser("CREATE DATABASE db_1;", user, "2");
        runSqlAsUser("CREATE DATABASE db_2;", user, "2");
        runSqlAsUser("DROP DATABASE db_1;", user, "2");
        runSqlAsUser("DROP DATABASE db_2;", user, "2");
        logger.info("Finished thread[2]");
      } catch (Exception e) {
        logger.error("Thread[2] Caught Exception: " + e.getMessage(), e);
        exceptions.add(e);
      }
    });
    threads[2].start();

    threads[3] = new Thread(() -> {
      try {
        logger.info("Starting thread[3]");
        MapdTestClient user = MapdTestClient.getClient(
                "localhost", 6274, "omnisci", "admin", "HyperInteractive");
        barrier.await();
        runSqlAsUser("CREATE ROLE role_1;", user, "3");
        runSqlAsUser("CREATE ROLE role_2;", user, "3");
        runSqlAsUser("DROP ROLE role_1;", user, "3");
        runSqlAsUser("DROP ROLE role_2;", user, "3");
        logger.info("Finished thread[3]");
      } catch (Exception e) {
        logger.error("Thread[3] Caught Exception: " + e.getMessage(), e);
        exceptions.add(e);
      }
    });
    threads[3].start();

    threads[4] = new Thread(() -> {
      try {
        logger.info("Starting thread[4]");
        MapdTestClient user = MapdTestClient.getClient(
                "localhost", 6274, "information_schema", "admin", "HyperInteractive");
        barrier.await();
        runSqlAsUser("SELECT * FROM users;", user, "4");
        logger.info("Finished thread[4]");
      } catch (Exception e) {
        logger.error("Thread[4] Caught Exception: " + e.getMessage(), e);
        exceptions.add(e);
      }
    });
    threads[4].start();

    threads[5] = new Thread(() -> {
      try {
        logger.info("Starting thread[5]");
        MapdTestClient user = MapdTestClient.getClient(
                "localhost", 6274, "information_schema", "admin", "HyperInteractive");
        barrier.await();
        runSqlAsUser("SELECT * FROM permissions;", user, "5");
        logger.info("Finished thread[5]");
      } catch (Exception e) {
        logger.error("Thread[5] Caught Exception: " + e.getMessage(), e);
        exceptions.add(e);
      }
    });
    threads[5].start();

    threads[6] = new Thread(() -> {
      try {
        logger.info("Starting thread[6]");
        MapdTestClient user = MapdTestClient.getClient(
                "localhost", 6274, "information_schema", "admin", "HyperInteractive");
        barrier.await();
        runSqlAsUser("SELECT * FROM databases;", user, "6");
        logger.info("Finished thread[6]");
      } catch (Exception e) {
        logger.error("Thread[6] Caught Exception: " + e.getMessage(), e);
        exceptions.add(e);
      }
    });
    threads[6].start();

    threads[7] = new Thread(() -> {
      try {
        logger.info("Starting thread[7]");
        MapdTestClient user = MapdTestClient.getClient(
                "localhost", 6274, "information_schema", "admin", "HyperInteractive");
        barrier.await();
        runSqlAsUser("SELECT * FROM roles;", user, "7");
        logger.info("Finished thread[7]");
      } catch (Exception e) {
        logger.error("Thread[7] Caught Exception: " + e.getMessage(), e);
        exceptions.add(e);
      }
    });
    threads[7].start();

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

  private void runSqlAsUser(String sql, MapdTestClient user, String logPrefix)
          throws Exception {
    logger.info(logPrefix + " " + sql);
    user.runSql(sql);
  }
}
