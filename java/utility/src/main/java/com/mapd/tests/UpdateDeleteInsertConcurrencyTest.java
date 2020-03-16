/*
 * Copyright 2018 OmniSci, Inc.
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

public class UpdateDeleteInsertConcurrencyTest {
  final static Logger logger =
          LoggerFactory.getLogger(UpdateDeleteInsertConcurrencyTest.class);

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
    UpdateDeleteInsertConcurrencyTest test = new UpdateDeleteInsertConcurrencyTest();
    test.testUpdateDeleteInsertConcurrency();
  }

  private void runTest(String db,
          String dbaUser,
          String dbaPassword,
          String dbUser,
          String dbPassword,
          Boolean concurrentInserts) throws Exception {
    int num_threads = 5;
    final int runs = 25;
    final int num_rows = 1000;
    final int fragment_size = 10;
    final String tableName = "test";
    Exception exceptions[] = new Exception[num_threads];

    final CyclicBarrier barrier = new CyclicBarrier(num_threads, new Runnable() {
      public void run() {
        try {
          MapdTestClient dba =
                  MapdTestClient.getClient("localhost", 6274, db, dbaUser, dbaPassword);
          dba.runSql("CREATE TABLE " + tableName
                  + "(x BIGINT, y INTEGER, z SMALLINT, a TINYINT, f FLOAT, d DOUBLE, deci DECIMAL(18,6), str TEXT ENCODING NONE) WITH (FRAGMENT_SIZE = "
                  + fragment_size + ")");
          if (!concurrentInserts) {
            for (int i = 0; i < num_rows; i++) {
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
              dba.runSql("INSERT INTO " + tableName + " VALUES "
                      + "(" + values_string + ")");
            }
          }
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

          try {
            barrier.await();

            MapdTestClient user =
                    MapdTestClient.getClient("localhost", 6274, db, dbUser, dbPassword);

            if (concurrentInserts) {
              for (int i = 0; i < num_rows / num_threads; i++) {
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
            }

            Random rand = new Random(tid);

            sql = "DELETE FROM " + tableName + " WHERE x = " + (tid * 2) + ";";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

            sql = "DELETE FROM " + tableName + " WHERE y = " + rand.nextInt(num_rows)
                    + ";";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

            sql = "SELECT COUNT(*) FROM " + tableName + " WHERE x > " + (tid * 2) + ";";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

            sql = "DELETE FROM " + tableName + " WHERE str = '"
                    + text_values[rand.nextInt(text_values.length)] + "';";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

            sql = "SELECT * FROM " + tableName + " WHERE str = '"
                    + text_values[rand.nextInt(text_values.length)] + "';";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

            sql = "DELETE FROM " + tableName + " WHERE d <  " + rand.nextInt(num_rows / 4)
                    + ";";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

            sql = "INSERT INTO " + tableName + " VALUES "
                    + "(" + tid + "," + tid + "," + tid + "," + tid + "," + tid + ","
                    + tid + "," + tid + "," + (tid % 2 == 0 ? "'mapd'" : "'omnisci'")
                    + ");";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

            sql = "DELETE FROM " + tableName + " WHERE z = " + tid + ";";
            logger.info(logPrefix + " " + sql);
            user.runSql(sql);

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
    dba.runSql("DROP TABLE " + tableName + ";");

    for (Exception e : exceptions) {
      if (null != e) {
        logger.error("Exception: " + e.getMessage(), e);
        throw e;
      }
    }
  }

  public void testUpdateDeleteInsertConcurrency() throws Exception {
    logger.info("testUpdateDeleteInsertConcurrency()");

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

    runTest("db1",
            "admin",
            "HyperInteractive",
            "admin",
            "HyperInteractive",
            /* concurrentInserts= */ false);
    runTest("db1",
            "admin",
            "HyperInteractive",
            "admin",
            "HyperInteractive",
            /* concurrentInserts= */ true);
    // TODO: run some tests as bob

    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER dba;");

    logger.info("testUpdateDeleteInsertConcurrency() done");
  }
}
