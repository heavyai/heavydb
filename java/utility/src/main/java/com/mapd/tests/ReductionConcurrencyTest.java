/*
 * Copyright 2022 OmniSci, Inc.
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

public class ReductionConcurrencyTest {
  final static Logger logger = LoggerFactory.getLogger(ReductionConcurrencyTest.class);

  final static String[] text_values = {"foo",
          "bar",
          "hello",
          "world",
          "a",
          "b",
          "c",
          "d",
          "e",
          "f",
          "g",
          "h",
          "i",
          "j",
          "k",
          "l",
          "m",
          "n",
          "o",
          "p"};

  public static void main(String[] args) throws Exception {
    ReductionConcurrencyTest test = new ReductionConcurrencyTest();
    test.testConcurrency();
  }

  private void runTest(String db,
          String dbaUser,
          String dbaPassword,
          String dbUser,
          String dbPassword,
          int numRows,
          int fragmentSize) throws Exception {
    int num_threads = 5;
    final int runs = 5;
    Exception exceptions[] = new Exception[num_threads];

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

          for (int runId = 0; runId < runs; runId++) {
            final String tableName = dbaUser + "_" + threadId + "_" + runId;

            try {
              MapdTestClient user =
                      MapdTestClient.getClient("localhost", 6274, db, dbUser, dbPassword);
              user.runSql("DROP TABLE IF EXISTS " + tableName + ";");
              user.runSql("CREATE TABLE " + tableName
                      + "(x BIGINT, y INTEGER, z SMALLINT, a TINYINT, f FLOAT, d DOUBLE, deci DECIMAL(18,6), str TEXT ENCODING NONE) WITH (FRAGMENT_SIZE = "
                      + fragmentSize + ")");

              for (int i = 0; i < numRows; i++) {
                final String integer_val = Integer.toString(i);
                final String small_val = Integer.toString(i % 128);
                final String fp_val = Double.toString(i * 1.1);
                final String deci_val = Double.toString(i + 0.01);
                final String str_val = "'" + text_values[i % text_values.length] + "'";
                final String values_string = String.join(" , ",
                        integer_val,
                        integer_val,
                        small_val,
                        small_val,
                        fp_val,
                        fp_val,
                        deci_val,
                        str_val);
                user.runSql("INSERT INTO " + tableName + " VALUES "
                        + "(" + values_string + ")");
              }

              Random rand = new Random(tid);

              sql = "SELECT * FROM " + tableName + " LIMIT 2;";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "SELECT x, sum(1) FROM " + tableName + " GROUP BY 1;";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "SELECT x, y, sum(1) FROM " + tableName + " GROUP BY 1, 2;";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "SELECT x, y, z, sum(1) FROM " + tableName + " GROUP BY 1, 2, 3;";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "SELECT x, y, avg(z), sum(1) FROM " + tableName + " GROUP BY 1, 2;";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "SELECT x, y, max(z), sum(1) FROM " + tableName + " GROUP BY 1, 2;";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "SELECT x, y, min(z), sum(1) FROM " + tableName + " GROUP BY 1, 2;";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "SELECT * FROM " + tableName + " WHERE str = '"
                      + text_values[rand.nextInt(text_values.length)] + "';";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

            } catch (Exception e) {
              logger.error(logPrefix + " Caught Exception: " + e.getMessage(), e);
              exceptions[threadId] = e;
            }
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
        throw e;
      }
    }
  }

  public void testConcurrency() throws Exception {
    logger.info("ReductionConcurrencyTest()");

    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "omnisci", "admin", "HyperInteractive");
    su.runSql("DROP USER IF EXISTS dba;");
    su.runSql("DROP USER IF EXISTS bob;");
    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");

    su.runSql("GRANT CREATE on DATABASE omnisci TO bob;");

    su.runSql("DROP DATABASE IF EXISTS db1;");
    su.runSql("CREATE DATABASE db1;");

    su.runSql("GRANT CREATE on DATABASE db1 TO bob;");
    su.runSql("GRANT CREATE VIEW on DATABASE db1 TO bob;");
    su.runSql("GRANT DROP on DATABASE db1 TO bob;");
    su.runSql("GRANT DROP VIEW on DATABASE db1 TO bob;");

    // run the test multiple times to make sure about the test results
    for (int i = 0; i < 3; ++i) {
      // test 1. reduction by interpreter
      runTest("db1", "admin", "HyperInteractive", "admin", "HyperInteractive", 10, 3);
      // test 2. reduction by codegen
      runTest("db1", "admin", "HyperInteractive", "admin", "HyperInteractive", 30, 3);
    }

    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER dba;");

    logger.info("ReductionConcurrencyTest() done");
  }
}
