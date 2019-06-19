/*
 * Copyright 2019 OmniSci, Inc.
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

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

public class SelectCopyFromDeleteConcurrencyTest {
  final static Logger logger =
          LoggerFactory.getLogger(SelectCopyFromDeleteConcurrencyTest.class);

  Path input_file_path_;

  public SelectCopyFromDeleteConcurrencyTest(Path path) {
    input_file_path_ = path;
  }

  public static void main(String[] args) throws Exception {
    // TODO: take as arg
    String path_str = "java/utility/src/main/java/com/mapd/tests/data/simple_test.csv";

    Path path = Paths.get(path_str).toAbsolutePath();
    assert Files.exists(path);

    SelectCopyFromDeleteConcurrencyTest test =
            new SelectCopyFromDeleteConcurrencyTest(path);
    test.testSelecyCopyFromConcurrency(/*shard_count=*/0);
    // TODO: take as arg
    int shard_count = 4; // expected configuration: 2 leaves, 2 GPUs per leaf
    test.testSelecyCopyFromConcurrency(shard_count);
  }

  private void run_test(MapdTestClient dba,
          MapdTestClient user,
          String prefix,
          Path filepath,
          int shard_count,
          int runs) throws Exception {
    String table_name = "table_" + prefix + "_";
    long tid = Thread.currentThread().getId();
    if (shard_count > 0) {
      logger.info("[" + tid + "] "
              + "CREATE " + table_name + " WITH " + shard_count + " SHARDS");
      user.runSql("CREATE TABLE " + table_name
              + " (id INTEGER, str TEXT ENCODING DICT(32), x DOUBLE, y BIGINT, SHARD KEY(id)) WITH (FRAGMENT_SIZE=1, SHARD_COUNT="
              + shard_count + ")");
    } else {
      logger.info("[" + tid + "] "
              + "CREATE " + table_name);
      user.runSql("CREATE TABLE " + table_name
              + " (id INTEGER, str TEXT ENCODING DICT(32), x DOUBLE, y BIGINT) WITH (FRAGMENT_SIZE=1)");
    }

    for (int i = 0; i < runs; i++) {
      logger.info("[" + tid + "] "
              + "SELECT 1 " + table_name);
      user.runSql("SELECT id, str FROM " + table_name + " WHERE x < 5.0 LIMIT 1;");

      logger.info("[" + tid + "] "
              + "COPY 1 " + table_name);
      user.runSql("COPY " + table_name + " FROM '" + filepath.toString()
              + "' WITH (header='false');");

      logger.info("[" + tid + "] "
              + "SELECT 2 " + table_name);
      user.runSql("SELECT COUNT(*) FROM " + table_name + " WHERE x = (SELECT MIN(x) FROM "
              + table_name + ");");

      logger.info("[" + tid + "] "
              + "DELETE 1 " + table_name);
      user.runSql("DELETE FROM " + table_name + " WHERE x = (SELECT MAX(x) FROM "
              + table_name + ");");

      logger.info("[" + tid + "] "
              + "SELECT 2 " + table_name);
      user.runSql("COPY " + table_name + " FROM '" + filepath.toString()
              + "' WITH (header='false');");

      logger.info("[" + tid + "] "
              + "TRUNCATE 1 " + table_name);
      user.runSql("TRUNCATE TABLE " + table_name);

      if (tid % 4 == 0) {
        logger.info("[" + tid + "] "
                + "COPY 3 " + table_name);
        dba.runSql("COPY " + table_name + " FROM '" + filepath.toString()
                + "' WITH (header='false');");
      } else {
        logger.info("[" + tid + "] "
                + "SELECT 3 " + table_name);
        user.runSql("SELECT COUNT(*) FROM " + table_name
                + " WHERE x = (SELECT MIN(x) FROM " + table_name + ");");
      }
    }

    logger.info("[" + tid + "] "
            + "DROP TABLE " + table_name);
    dba.runSql("DROP TABLE " + table_name + ";");
  }

  private void runTest(int num_threads, int shard_count) throws Exception {
    final int runs = 25;
    Exception exceptions[] = new Exception[num_threads];

    ArrayList<Thread> threads = new ArrayList<>();
    for (int i = 0; i < num_threads; i++) {
      logger.info("Starting " + i);
      final int threadId = i;

      Thread t = new Thread(new Runnable() {
        @Override
        public void run() {
          try {
            final String username = threadId % 2 == 0 ? "alice" : "bob";
            MapdTestClient dba = MapdTestClient.getClient(
                    "localhost", 6274, "omnisci", "admin", "HyperInteractive");
            MapdTestClient user = MapdTestClient.getClient(
                    "localhost", 6274, "omnisci", username, "password");
            final String prefix = "for_" + username + "_" + threadId + "_";

            run_test(dba, user, prefix, input_file_path_, shard_count, runs);

          } catch (Exception e) {
            logger.error("[" + Thread.currentThread().getId() + "] "
                    + "Caught Exception: " + e.getMessage());
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

  public void testSelecyCopyFromConcurrency(int shard_count) throws Exception {
    logger.info("testSelectCopyFromConcurrency()");

    logger.info("Using import file: " + input_file_path_.toString());

    // Use the default database for now
    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "omnisci", "admin", "HyperInteractive");
    su.runSql("CREATE USER alice (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");

    su.runSql("GRANT CREATE on DATABASE omnisci TO alice;");
    su.runSql("GRANT CREATE on DATABASE omnisci TO bob;");

    su.runSql("GRANT CREATE VIEW on DATABASE omnisci TO alice;");
    su.runSql("GRANT CREATE VIEW on DATABASE omnisci TO bob;");

    su.runSql("GRANT DROP VIEW on DATABASE omnisci TO alice;");
    su.runSql("GRANT DROP VIEW on DATABASE omnisci TO bob;");

    su.runSql("GRANT ACCESS on database omnisci TO alice;");
    su.runSql("GRANT ACCESS on database omnisci TO bob;");

    final int num_threads = 5;
    runTest(num_threads, shard_count);

    su.runSql("DROP USER alice;");
    su.runSql("DROP USER bob;");

    logger.info("Pass!");
  }
}
