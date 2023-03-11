/*
 * Copyright 2023 HEAVY.AI, Inc.
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
import java.lang.Integer;
import java.lang.String;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CyclicBarrier;

public class DdlConcurrencyTest extends ConcurrencyTest {
  static File scratchDir;
  static String scratchDirPath;

  public static void main(String[] args) throws Exception {
    configure();
    scratchDir = new File("./scratch");
    scratchDirPath = scratchDir.getCanonicalPath();
    DdlConcurrencyTest test = new DdlConcurrencyTest();
    test.testConcurrency();
  }

  // Configure testing based on command line parameteres.
  public static void configure() {
    logger = getLogger();
    testName = getTestName();
    ports = getPorts();
    numThreads = getNumThreads();
    numIterations = getNumIterations();
    enableHeavyConnect = getEnableHeavyConnect();
    enableMonitorThread = getEnableMonitorThread();
    logger.info(getConfig());
  }

  public static Logger getLogger() {
    return LoggerFactory.getLogger(DdlConcurrencyTest.class);
  }

  public static String getTestName() {
    return "DdlConcurrencyTest";
  }

  // Full list of tests to perform.
  public List<SqlCommandThread[]> createTestThreads() {
    logger.info("Initializing test threads...");
    ArrayList<SqlCommandThread[]> tests = new ArrayList<>();
    tests.add(makeTests("ShowTable", "SHOW TABLES;"));
    tests.add(makeTests("ShowTableDetails", "SHOW TABLE DETAILS;"));
    tests.add(makeTests("ShowCreateTable", "SHOW CREATE TABLE test_table;"));
    tests.add(makeTests("ShowServers", "SHOW SERVERS;"));
    tests.add(makeTests("ShowCreateServer", "SHOW CREATE SERVER test_server;"));
    tests.add(makeTests("ShowFunctions", "SHOW FUNCTIONS;"));
    tests.add(makeTests("ShowRuntimeFunctions", "SHOW RUNTIME FUNCTIONS;"));
    tests.add(makeTests("ShowRuntimeTF", "SHOW RUNTIME TABLE FUNCTIONS;"));
    tests.add(makeTests("AlterDatabase", "ALTER DATABASE " + db + " OWNER TO admin;"));
    tests.add(makeTests("ShowUserDetails", "SHOW USER DETAILS;"));
    tests.add(makeTests("ShowRoles", "SHOW ROLES"));
    tests.add(makeTests("ReassignOwned", "REASSIGN OWNED BY test_user TO admin;"));
    tests.add(makeTests("CreatePolicy",
            "CREATE POLICY ON COLUMN test_table.i TO test_user VALUES (1);",
            "already exists"));
    tests.add(makeTests("DropPolicy",
            "DROP POLICY ON COLUMN test_table.i FROM test_user;",
            "not found"));
    tests.add(makeTests("ShowPolicy", "SHOW POLICIES test_user;"));
    tests.add(makeTests("ShowSupportedDataSources", "SHOW POLICIES test_user;"));
    tests.add(makeTests(
            "CreateTable", "CREATE TABLE IF NOT EXISTS test_table_<threadId> (i int);"));
    tests.add(makeTests("DropTable", "DROP TABLE IF EXISTS test_table_<threadId>;"));
    tests.add(makeTests("InsertTable", "INSERT INTO test_table (i) VALUES (2);"));
    tests.add(makeTests("UpdateTable", "UPDATE test_table SET i = 3 WHERE i = 2;"));
    tests.add(makeTests("AddColumn",
            "ALTER TABLE test_table ADD (x int DEFAULT 1);",
            "already exists"));
    tests.add(makeTests(
            "DropColumn", "ALTER TABLE test_table DROP COLUMN x;", "does not exist"));
    tests.add(makeTests("RenameTable",
            "ALTER TABLE test_table_<threadId> RENAME TO altered_table_<threadId>;",
            Arrays.asList("not exist", "Attempted to overwrite")));
    tests.add(makeTests("DumpTable",
            "DUMP TABLE test_table to '" + scratchDirPath + "/dump.gz';",
            "exists"));
    tests.add(makeTests("RestoreTable",
            "RESTORE TABLE restore_table_<threadId> from '" + scratchDirPath
                    + "/dump.gz';",
            Arrays.asList("does not exist", "exists"),
            "DROP TABLE IF EXISTS restore_table_<threadId>;"));
    tests.add(makeTests("TruncateTable", "TRUNCATE TABLE test_table;"));
    tests.add(makeTests("OptimizeTable", "OPTIMIZE TABLE test_table;"));
    tests.add(
            makeTests("CopyFrom", "COPY test_table FROM '../Tests/FsiDataFiles/1.csv';"));
    tests.add(makeTests("CopyTo",
            "COPY (SELECT * FROM test_table) TO '" + scratchDirPath + "/copy.csv';",
            "exists"));
    tests.add(makeTests("CreateDB",
            "CREATE DATABASE IF NOT EXISTS test_db_<threadId>;",
            Collections.emptyList(),
            "DROP DATABASE IF EXISTS test_db_<threadId>;"));
    tests.add(makeTests("DropDB", "DROP DATABASE IF EXISTS test_db_<threadId>;"));
    tests.add(makeTests("CreateUser",
            "CREATE USER test_user_<threadId> (password = 'pass');",
            "exists",
            "DROP USER IF EXISTS test_user_<threadId>;"));
    tests.add(makeTests("DropUser", "DROP USER IF EXISTS test_user_<threadId>;"));
    tests.add(makeTests("AlterUser", "ALTER USER test_user (password = 'pass');"));
    tests.add(makeTests("RenameUser",
            "ALTER USER test_user_<threadId> RENAME TO altered_user_<threadId>;",
            Arrays.asList("doesn't exist", "already exists", "does not exist"),
            "DROP USER IF EXISTS altered_user_<threadId>;"));
    tests.add(makeTests("CreateRole",
            "CREATE ROLE test_role_<threadId>;",
            "already exists",
            "DROP ROLE IF EXISTS test_role_<threadId>;"));
    tests.add(makeTests("DropRole", "DROP ROLE IF EXISTS test_role_<threadId>;"));
    tests.add(makeTests(
            "GrantRole", "GRANT test_role_<threadId> TO test_user;", "does not exist"));
    tests.add(makeTests("RevokeRole",
            "REVOKE test_role_<threadId> FROM test_user;",
            Arrays.asList("does not exist", "have not been granted")));
    tests.add(makeTests("ValidateSystem", "Validate"));
    tests.add(makeTests("CreateView",
            "CREATE VIEW IF NOT EXISTS test_view_<threadId> AS (SELECT * FROM test_table);"));
    tests.add(makeTests("DropView", "DROP VIEW IF EXISTS test_view_<threadId>;"));
    if (enableHeavyConnect) {
      tests.add(makeTests("CreateServer",
              "CREATE SERVER IF NOT EXISTS test_server_<threadId> FOREIGN DATA WRAPPER "
                      + "delimited_file WITH (storage_type = 'LOCAL_FILE', base_path = '"
                      + System.getProperty("user.dir") + "');"));
      tests.add(makeTests("DropServer", "DROP SERVER IF EXISTS test_server_<threadId>;"));
      tests.add(makeTests("AlterServer",
              "ALTER SERVER test_server SET (s3_bucket = 'diff_bucket');"));
      tests.add(makeTests("CreateForeignTable",
              "CREATE FOREIGN TABLE IF NOT EXISTS test_foreign_table_<threadId> "
                      + "(b BOOLEAN, t TINYINT, s SMALLINT, i INTEGER, bi BIGINT, f FLOAT, "
                      + "dc DECIMAL(10, 5), tm TIME, tp TIMESTAMP, d DATE, txt TEXT, "
                      + "txt_2 TEXT ENCODING NONE) "
                      + "SERVER default_local_delimited WITH "
                      + "(file_path = '../Tests/FsiDataFiles/scalar_types.csv', "
                      + "FRAGMENT_SIZE = 10);"));
      tests.add(makeTests("DropForeignTable",
              "DROP FOREIGN TABLE IF EXISTS test_foreign_table_<threadId>;"));
      tests.add(makeTests("CreateUserMapping",
              "CREATE USER MAPPING IF NOT EXISTS FOR PUBLIC SERVER test_server WITH "
                      + "(S3_ACCESS_KEY = 'test_key', S3_SECRET_KEY = 'test_key');"));
      tests.add(makeTests("DropUserMapping",
              "DROP USER MAPPING IF EXISTS FOR PUBLIC SERVER test_server;"));
      tests.add(makeTests("AlterForeignTable",
              "ALTER FOREIGN TABLE test_foreign_table SET (refresh_update_type = 'APPEND');"));
      tests.add(makeTests("ShowDiskCacheUsage", "SHOW DISK CACHE USAGE;"));
      tests.add(makeTests("RefreshForeignTable",
              "REFRESH FOREIGN TABLES test_foreign_table_<threadId>;",
              "does not exist"));
    }
    logger.info("Initialized test threads.");
    return tests;
  }

  // Some tests use custom queries that include the threadID.  Here we do a quick
  // find/replace of the <threadId> symbol with the actual thread id.
  public static List<String> customizeQueriesByThreadId(
          final List<String> queries, int threadId) {
    List<String> customQueries = new ArrayList<String>();
    for (String query : queries) {
      customQueries.add(query.replaceAll("<threadId>", Integer.toString(threadId)));
    }
    return customQueries;
  }

  // Initializes an array of threads that perform the same test, customized by thread id.
  public final SqlCommandThread[] makeTests(final String threadName,
          final List<String> queries,
          final List<String> expectedExceptions,
          final List<String> cleanUpQueries,
          int numThreads) {
    SqlCommandThread[] threadGroup = new SqlCommandThread[numThreads];
    for (int threadId = 0; threadId < numThreads; ++threadId) {
      List<String> customQueries = customizeQueriesByThreadId(queries, threadId);
      List<String> customCleanUps = customizeQueriesByThreadId(cleanUpQueries, threadId);
      threadGroup[threadId] = new SqlCommandThread(
              threadName, customQueries, threadId, expectedExceptions, customCleanUps);
    }
    return threadGroup;
  }

  // Syntactic sugar for common cases.
  public final SqlCommandThread[] makeTests(final String threadName, final String query) {
    return makeTests(threadName,
            Arrays.asList(query),
            Collections.emptyList(),
            Collections.emptyList(),
            numThreads);
  }

  // Syntactic sugar for common cases.
  public final SqlCommandThread[] makeTests(
          final String threadName, final String query, final String exception) {
    return makeTests(threadName,
            Arrays.asList(query),
            Arrays.asList(exception),
            Collections.emptyList(),
            numThreads);
  }

  // Syntactic sugar for common cases.
  public final SqlCommandThread[] makeTests(
          final String threadName, final String query, final List<String> exceptions) {
    return makeTests(threadName,
            Arrays.asList(query),
            exceptions,
            Collections.emptyList(),
            numThreads);
  }

  // Syntactic sugar for common cases.
  public final SqlCommandThread[] makeTests(final String threadName,
          final String query,
          final List<String> exceptions,
          final String cleanUpQueries) {
    return makeTests(threadName,
            Arrays.asList(query),
            exceptions,
            Arrays.asList(cleanUpQueries),
            numThreads);
  }

  // Syntactic sugar for common cases.
  public final SqlCommandThread[] makeTests(final String threadName,
          final String query,
          final String exceptions,
          final String cleanUpQueries) {
    return makeTests(threadName,
            Arrays.asList(query),
            Arrays.asList(exceptions),
            Arrays.asList(cleanUpQueries),
            numThreads);
  }

  @Override
  public void setUpTests() throws Exception {
    logger.info("Starting Setup...");
    scratchDir.mkdirs();
    HeavyDBTestClient heavyAdmin = getAdminClient(defaultDb);
    runAndLog(heavyAdmin, "DROP DATABASE IF EXISTS " + db + ";");
    runAndLog(heavyAdmin, "CREATE DATABASE " + db + ";");
    HeavyDBTestClient dbAdmin = getAdminClient(db);
    runAndLog(dbAdmin, "DROP USER IF EXISTS test_user");
    runAndLog(dbAdmin, "CREATE USER test_user (password = 'pass');");
    runAndLog(dbAdmin, "CREATE TABLE test_table (i int);");
    if (enableHeavyConnect) {
      runAndLog(dbAdmin,
              "CREATE SERVER test_server FOREIGN DATA WRAPPER delimited_file "
                      + "WITH (storage_type = 'AWS_S3', s3_bucket = 'test_bucket', "
                      + "aws_region = 'test_region');");
      runAndLog(dbAdmin,
              "CREATE FOREIGN TABLE IF NOT EXISTS test_foreign_table"
                      + " (b BOOLEAN, t TINYINT, s SMALLINT, i INTEGER, bi BIGINT, f FLOAT, "
                      + "dc DECIMAL(10, 5), tm TIME, tp TIMESTAMP, d DATE, txt TEXT, "
                      + "txt_2 TEXT ENCODING NONE) "
                      + "SERVER default_local_delimited WITH "
                      + "(file_path = '../Tests/FsiDataFiles/scalar_types.csv', "
                      + "FRAGMENT_SIZE = 10);");
    }
    logger.info("Finished Setup.");
  }

  @Override
  public void cleanUpTests(final List<SqlCommandThread[]> tests) throws Exception {
    logger.info("Starting cleanup...");
    HeavyDBTestClient heavyAdmin = getAdminClient(defaultDb);
    runAndLog(heavyAdmin, "DROP USER test_user;");
    runAndLog(heavyAdmin, "DROP DATABASE " + db + ";");
    // Some tests require (and can specify) custom cleanup.
    for (SqlCommandThread[] testGroup : tests) {
      for (SqlCommandThread test : testGroup) {
        for (String query : test.cleanUpQueries) {
          runAndLog(heavyAdmin, query);
        }
      }
    }
    deleteDirectory(scratchDir);
    logger.info("Finished cleanup.");
  }

  @Override
  public void runTests(final List<SqlCommandThread[]> tests) throws Exception {
    logger.info("starting runTests.");

    // Initialize threads, but block them from executing until all threads are ready.
    int numTests = 0;
    for (SqlCommandThread threadGroup[] : tests) {
      numTests += threadGroup.length;
    }
    barrier = createBarrier(numTests);

    for (SqlCommandThread threadGroup[] : tests) {
      for (SqlCommandThread thread : threadGroup) {
        thread.start();
      }
    }

    logger.info("Waiting for threads to sync...");

    if (enableMonitorThread) {
      new MonitoringThread(tests, numThreads).start();
    }

    for (SqlCommandThread threadGroup[] : tests) {
      for (SqlCommandThread thread : threadGroup) {
        thread.join();
      }
    }

    printErrors(exceptionTexts);

    logger.info("Finished runTests.");
  }
}
