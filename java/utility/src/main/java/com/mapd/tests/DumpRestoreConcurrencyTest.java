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

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CyclicBarrier;

public class DumpRestoreConcurrencyTest {
  final static Logger logger = LoggerFactory.getLogger(DumpRestoreConcurrencyTest.class);

  public static void main(String[] args) throws Exception {
    DumpRestoreConcurrencyTest test = new DumpRestoreConcurrencyTest();
    test.testConcurrency();
  }

  public void testConcurrency() throws Exception {
    logger.info("DumpRestoreConcurrencyTest()");
    runTest();
    logger.info("DumpRestoreConcurrencyTest() done");
  }

  private void deleteFileIfExist(String file) {
    try {
      Files.deleteIfExists((new File(file)).toPath());
    } catch (Exception e) {
      logger.error("While Deleting Archives Caught Exception: " + e.getMessage(), e);
    }
  }

  // This is the file path utilized by `DumpRestoreTest`, used here for consistency
  static String tar_ball_path_prefix = "/tmp/_Orz__";

  List<String> getDumpRestoreQueries(String table_identifier) {
    return Arrays.asList("DROP TABLE IF EXISTS " + table_identifier + ";",
            "DROP TABLE IF EXISTS restored_" + table_identifier + ";",
            "CREATE TABLE " + table_identifier + " (v INT);",
            "INSERT INTO " + table_identifier + " VALUES (1),(2);",
            "DUMP TABLE " + table_identifier + " TO '" + tar_ball_path_prefix + "_"
                    + table_identifier + "';",
            "RESTORE TABLE restored_" + table_identifier + " FROM '"
                    + tar_ball_path_prefix + "_" + table_identifier + "';");
  }

  private void runTest() throws Exception {
    List<ThreadDbDumpRestoreQueries> queriesPerThread =
            new ArrayList<ThreadDbDumpRestoreQueries>(Arrays.asList(
                    new ThreadDbDumpRestoreQueries(
                            tar_ball_path_prefix + "_aa", getDumpRestoreQueries("aa")),
                    new ThreadDbDumpRestoreQueries(
                            tar_ball_path_prefix + "_bb", getDumpRestoreQueries("bb")),
                    new ThreadDbDumpRestoreQueries(
                            tar_ball_path_prefix + "_cc", getDumpRestoreQueries("cc")),
                    new ThreadDbDumpRestoreQueries(
                            tar_ball_path_prefix + "_dd", getDumpRestoreQueries("dd")),
                    new ThreadDbDumpRestoreQueries(
                            tar_ball_path_prefix + "_ee", getDumpRestoreQueries("ee"))));

    final int num_threads = queriesPerThread.size();
    Exception[] exceptions = new Exception[num_threads];
    Thread[] threads = new Thread[num_threads];

    // Use a barrier here to synchronize the start of each competing thread and make
    // sure there is a race condition. The barrier ensures no thread will run until they
    // are all created and waiting for the barrier to complete.
    final CyclicBarrier barrier = new CyclicBarrier(
            num_threads, () -> { logger.info("Barrier acquired. Starting queries..."); });

    for (int i = 0; i < queriesPerThread.size(); i++) {
      final ThreadDbDumpRestoreQueries threadQueries = queriesPerThread.get(i);
      final int threadId = i;
      threads[threadId] = new Thread(() -> {
        try {
          logger.info("Starting thread[" + threadId + "]");
          HeavyDBTestClient user = HeavyDBTestClient.getClient(
                  "localhost", 6274, "heavyai", "admin", "HyperInteractive");
          deleteFileIfExist(threadQueries.archive);
          barrier.await();
          for (final String query : threadQueries.queries) {
            runSqlAsUser(query, user, threadId);
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

    // clean up archive files
    for (int i = 0; i < queriesPerThread.size(); i++) {
      final ThreadDbDumpRestoreQueries threadQueries = queriesPerThread.get(i);
      deleteFileIfExist(threadQueries.archive);
    }

    for (Exception e : exceptions) {
      if (e != null) {
        logger.error("Exception: " + e.getMessage(), e);
        throw e;
      }
    }
  }

  private void runSqlAsUser(String sql, HeavyDBTestClient user, int threadId)
          throws Exception {
    logger.info(threadId + " " + sql);
    user.runSql(sql);
  }

  private class ThreadDbDumpRestoreQueries {
    public ThreadDbDumpRestoreQueries(
            final String archive_file, final List<String> queries) {
      this.archive = archive_file;
      this.queries = queries;
    }

    public final String archive;
    public final List<String> queries;
  }
}
