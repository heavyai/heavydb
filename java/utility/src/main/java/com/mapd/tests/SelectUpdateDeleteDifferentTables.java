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
import java.util.Random;

public class SelectUpdateDeleteDifferentTables {
  final static Logger logger =
          LoggerFactory.getLogger(SelectUpdateDeleteDifferentTables.class);

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
    SelectUpdateDeleteDifferentTables test = new SelectUpdateDeleteDifferentTables();
    test.testConcurrency();
  }

  private void runTest(
          String db, String dbaUser, String dbaPassword, String dbUser, String dbPassword)
          throws Exception {
    int num_threads = 5;
    final int runs = 5;
    final int num_rows = 400;
    final int fragment_size = 25;
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
              HeavyDBTestClient user = HeavyDBTestClient.getClient(
                      "localhost", 6274, db, dbUser, dbPassword);

              user.runSql("CREATE TABLE " + tableName
                      + "(x BIGINT, y INTEGER, z SMALLINT, a TINYINT, f FLOAT, d DOUBLE, deci DECIMAL(18,6), str TEXT ENCODING NONE) WITH (FRAGMENT_SIZE = "
                      + fragment_size + ")");

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
                user.runSql("INSERT INTO " + tableName + " VALUES "
                        + "(" + values_string + ")");
              }

              Random rand = new Random(tid);

              sql = "ALTER TABLE " + tableName + " ADD COLUMN zz TEXT ENCODING DICT(8);";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              // TODO(adb): add get_table_details once thread safe

              sql = "SELECT * FROM " + tableName + " LIMIT 2;";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "DELETE FROM " + tableName + " WHERE y = " + rand.nextInt(num_rows)
                      + ";";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "ALTER TABLE " + tableName + " DROP COLUMN x;";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "SELECT * FROM " + tableName + " WHERE str = '"
                      + text_values[rand.nextInt(text_values.length)] + "';";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "SELECT COUNT(*) FROM " + tableName + ";";
              logger.info(logPrefix + " VALIDATE " + sql);
              user.sqlValidate(sql);

              final String tableRename = tableName + "_rename";
              sql = "ALTER TABLE " + tableName + " RENAME TO " + tableRename;
              user.runSql(sql);

              sql = "TRUNCATE TABLE " + tableRename + ";";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "INSERT INTO " + tableRename + " VALUES "
                      + "(" + tid + "," + tid + "," + tid + "," + tid + "," + tid + ","
                      + tid + "," + tid + "," + (tid % 2 == 0 ? "'value_1'" : "'value_2'")
                      + ");";
              logger.info(logPrefix + " " + sql);
              user.runSql(sql);

              sql = "DROP TABLE " + tableRename + ";";
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
    logger.info("SelectUpdateDeleteDifferentTables()");

    HeavyDBTestClient su = HeavyDBTestClient.getClient(
            "localhost", 6274, "heavyai", "admin", "HyperInteractive");
    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");

    su.runSql("GRANT CREATE on DATABASE heavyai TO bob;");

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

    logger.info("SelectUpdateDeleteDifferentTables() done");
  }
}
