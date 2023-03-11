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
import java.lang.RuntimeException;
import java.lang.String;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CyclicBarrier;

import ai.heavy.thrift.server.TDBException;

public abstract class ConcurrencyTest {
  public final static String db = "TestDB", userName = "admin",
                             password = "HyperInteractive", localhost = "localhost",
                             defaultDb = "heavyai";
  public final static int defaultPort = 6274, defaultNumThreads = 5,
                          defaultNumIterations = 5;
  public final static boolean defaultEnableHeavyConnect = true,
                              defaultEnableMonitorThread = false;

  public static int numThreads = defaultNumThreads, numIterations = defaultNumIterations;
  public static boolean enableHeavyConnect = defaultEnableHeavyConnect,
                        enableMonitorThread = defaultEnableMonitorThread;

  static Random rand = new Random();

  // These need to be initialized in the derived class.
  public static Logger logger = null;
  public static String testName = "";
  public static int[] ports = null;

  public CyclicBarrier barrier; // Sync point for threads.
  public ArrayList<String> exceptionTexts = new ArrayList<String>();

  public static boolean deleteDirectory(File dir) {
    File[] files = dir.listFiles();
    if (files != null) {
      for (File file : files) {
        deleteDirectory(file);
      }
    }
    return dir.delete();
  }

  // Optionally specify the set of ports on which heavyDB servers are running (can run
  // multiple servers for Horizontal Scaling testing).  Threads will be randomly assigned
  // to ports.
  public static int[] getPorts() {
    String portString = System.getProperty("PORT_LIST");
    int[] retPorts;
    if (portString != null) {
      List<String> portList = Arrays.asList(portString.split(","));
      if (portList.size() > 0) {
        retPorts = new int[portList.size()];
        for (int i = 0; i < portList.size(); ++i) {
          retPorts[i] = Integer.parseInt(portList.get(i));
        }
        return retPorts;
      }
    }
    retPorts = new int[1];
    retPorts[0] = defaultPort;
    return retPorts;
  }

  // Optionally specify the number of threads teach test class will run.
  public static int getNumThreads() {
    String propertyString = System.getProperty("NUM_THREADS");
    if (propertyString == null) {
      return defaultNumThreads;
    } else {
      return Integer.parseInt(propertyString);
    }
  }

  // Optionally specify the number of iterations each thread will run.
  public static int getNumIterations() {
    String propertyString = System.getProperty("NUM_ITERATIONS");
    if (propertyString == null) {
      return defaultNumIterations;
    } else {
      return Integer.parseInt(propertyString);
    }
  }

  // Optionally controll if HeavyConnect testing is enabled.
  public static boolean getEnableHeavyConnect() {
    String propertyString = System.getProperty("ENABLE_HEAVY_CONNECT");
    if (propertyString == null) {
      return defaultEnableHeavyConnect;
    } else {
      return Boolean.parseBoolean(propertyString);
    }
  }

  // Optionally enable a monitoring thread that will periodically display how many threads
  // are still running (useful for debugging).
  public static boolean getEnableMonitorThread() {
    String propertyString = System.getProperty("ENABLE_MONITOR_THREAD");
    if (propertyString == null) {
      return defaultEnableMonitorThread;
    } else {
      return Boolean.parseBoolean(propertyString);
    }
  }

  // Dump the configuration.
  public static String getConfig() {
    String log = "Config for " + testName + ":\n{\n  ports = {\n";
    for (int port : ports) {
      log += "    " + port + "\n";
    }
    log += "  }\n  num_threads = " + numThreads + "\n  num_iterations = " + numIterations
            + "\n  enable_heavy_connect = " + enableHeavyConnect
            + "\n  enable_monitor_thread = " + enableMonitorThread + "\n}";
    return log;
  }

  // Barrier to synchronize all threads.
  public static CyclicBarrier createBarrier(int numThreadsToWait) {
    return new CyclicBarrier(numThreadsToWait, new Runnable() {
      @Override
      public void run() {
        logger.info("Threads Synched");
      }
    });
  }

  // Print all errors that have accumulated.
  public static void printErrors(ArrayList<String> exceptionTexts) {
    if (exceptionTexts.size() > 0) {
      String errors = "\n";
      for (String s : exceptionTexts) {
        errors += s + "\n";
      }
      logger.error("Found exceptions:" + errors);
    }
  }

  public static int getRandomPort() {
    return ports[rand.nextInt(ports.length)];
  }

  public abstract List<SqlCommandThread[]> createTestThreads();
  public abstract void runTests(final List<SqlCommandThread[]> tests) throws Exception;
  public abstract void setUpTests() throws Exception;
  public abstract void cleanUpTests(final List<SqlCommandThread[]> tests)
          throws Exception;

  // Performs test setup, running of all tests, and teardown.
  public void testConcurrency() throws Exception {
    if (testName.equals("")) {
      throw new RuntimeException("Derived test has not set a test name");
    }
    if (logger == null) {
      throw new RuntimeException("Derived test has not initialized logger");
    }
    logger.info(testName + "()");
    final List<SqlCommandThread[]> tests = createTestThreads();
    setUpTests();
    runTests(tests);
    cleanUpTests(tests);
    logger.info(testName + "() done");
  }

  public HeavyDBTestClient getAdminClient(String db) throws Exception {
    return HeavyDBTestClient.getClient(localhost, ports[0], db, userName, password);
  }

  public void runAndLog(HeavyDBTestClient client, String sql) throws Exception {
    client.runSql(sql);
    logger.info("  " + sql);
  }

  // Optional monitoring thread that will report how many threads are still running once a
  // minute (useful when debugging).
  public class MonitoringThread extends Thread {
    List<SqlCommandThread[]> monitoredThreads;
    int numThreads;

    MonitoringThread(List<SqlCommandThread[]> monitoredThreads, int numThreads) {
      this.monitoredThreads = monitoredThreads;
      this.numThreads = numThreads;
    }

    @Override
    public void run() {
      int finishedThreads = 0;
      int totalThreads = monitoredThreads.size() * numThreads;
      while (finishedThreads < totalThreads) {
        finishedThreads = 0;
        for (SqlCommandThread threadGroup[] : monitoredThreads) {
          for (SqlCommandThread thread : threadGroup) {
            if (thread.getState() == Thread.State.TERMINATED) {
              finishedThreads++;
            }
          }
        }
        logger.info("Threads running: " + (totalThreads - finishedThreads));
        try {
          Thread.sleep(60000);
        } catch (Exception e) {
          logger.error("Monitoring thread: " + e.getMessage());
        }
      }
    }
  }

  public class SqlCommandThread extends Thread {
    final List<String> queries, expectedExceptionTexts, cleanUpQueries;
    final String threadName;
    final int port, iterations, threadId;

    SqlCommandThread(final String threadName,
            final List<String> queries,
            int threadId,
            int port,
            int iterations,
            final List<String> exceptions,
            final List<String> cleanUpQueries) {
      this.queries = queries;
      this.port = port;
      this.iterations = iterations;
      this.threadId = threadId;
      this.threadName = threadName + "[" + port + "][" + threadId + "]";
      this.expectedExceptionTexts = exceptions;
      this.cleanUpQueries = cleanUpQueries;
    }

    SqlCommandThread(final String threadName,
            final List<String> queries,
            int threadId,
            final List<String> exceptions,
            final List<String> cleanUpQueries) {
      this(threadName,
              queries,
              threadId,
              getRandomPort(),
              numIterations,
              exceptions,
              cleanUpQueries);
    }

    @Override
    public void run() {
      logger.info("  Starting: " + threadName);
      try {
        HeavyDBTestClient user =
                HeavyDBTestClient.getClient(localhost, port, db, userName, password);
        barrier.await(); // Synch point.
        for (int iteration = 0; iteration < iterations; ++iteration) {
          for (String query : queries) {
            logger.info("  " + threadName + "[" + iteration + "]: " + query);
            try {
              user.runSql(query);
            } catch (TDBException e) {
              boolean foundExpected = false;
              for (String exceptionText : expectedExceptionTexts) {
                if (e.error_msg.contains(exceptionText)) {
                  foundExpected = true;
                }
              }
              if (!foundExpected) {
                throw e;
              }
            }
          }
        }
      } catch (TDBException e) {
        logger.error("  " + threadName + ": caught exception - " + e.error_msg);
        exceptionTexts.add(threadName + ": " + e.error_msg);
      } catch (Exception e) {
        logger.error("  " + threadName + ": caught exception - " + e.getMessage());
        exceptionTexts.add(threadName + ": " + e.getMessage());
      }
      logger.info("  Finished: " + threadName);
    }
  }
}
