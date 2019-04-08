/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mapd.bench;

// STEP 1. Import required packages
import com.omnisci.jdbc.OmniSciStatement;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class BenchmarkCloud {
  final static Logger logger = LoggerFactory.getLogger(BenchmarkCloud.class);

  static final String QUERY_RESULT_MACHINE = "bencher";

  // JDBC driver name and database URL
  static final String DB_URL = "jdbc:omnisci:localhost:6274:mapd";
  static final String JDBC_DRIVER = "com.omnisci.jdbc.OmniSciDriver";

  // Database credentials
  static final String USER = "mapd";
  static final String PASS = "";

  // Database credentials
  static final String RESULTS_USER = "mapd";
  static final String RESULTS_PASS = "";

  private String driver;
  private String url;
  private String iUser;
  private String queryResultMachine;
  private String iPasswd;
  private String iResultsUser;
  private String iResultsPasswd;
  private String rid;
  private String rTimestamp;
  private String tableName;
  private String label;
  private String gpuCount;
  private String targetDBVersion;
  Connection bencherCon;
  private List<String> LResult = new ArrayList<String>();

  private String headDescriptor =
          "%3s, %8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s";
  private String header2 = String.format(headDescriptor,
          "QRY",
          "T-Avg",
          "T-Min",
          "T-Max",
          "T-85%",
          "E-Avg",
          "E-Min",
          "E-Max",
          "E-85%",
          "E-25%",
          "E-StdD",
          "J-Avg",
          "J-Min",
          "J-Max",
          "J-85%",
          "I-Avg",
          "I-Min",
          "I-Max",
          "I-85%",
          "F-Exec",
          "F-jdbc",
          "F-iter",
          "ITER",
          "Total",
          "Account");
  private String lineDescriptor =
          "%3s, %8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8d,%8d,%8d,%8d,%8d,%8d";
  private String insertDescriptor =
          "('%s','%s','%s','%s','%s',%s,'%s','%s',%d,'%s', %8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8d,%8d,%8d,%8d,%8d,%8d, '%s')";

  public static void main(String[] args) {
    BenchmarkCloud bm = new BenchmarkCloud();
    bm.doWork(args, 1);
  }

  void doWork(String[] args, int query) {
    // Grab parameters from args
    // parm0 number of iterations per query
    // parm1 file containing sql queries {contains quoted query, expected result
    // count]
    // parm2 table name
    // parm3 run label
    // parm4 gpu count
    // parm5 optional query and result machine
    // parm6 optional DB URL
    // parm7 optional JDBC Driver class name
    // parm8 optional user
    // parm9 optional passwd
    // parm10 optional query results db user
    // parm11 optional query results db passwd
    int iterations = Integer.valueOf(args[0]);
    logger.debug("Iterations per query is " + iterations);

    String queryFile = args[1];
    tableName = args[2];
    label = args[3];
    gpuCount = args[4];

    // int expectedResults = Integer.valueOf(args[2]);
    queryResultMachine = (args.length > 5) ? args[5] : QUERY_RESULT_MACHINE;
    url = (args.length > 6) ? args[6] : DB_URL;
    driver = (args.length > 7) ? args[7] : JDBC_DRIVER;

    iUser = (args.length > 8) ? args[8] : USER;
    iPasswd = (args.length > 9) ? args[9] : PASS;

    iResultsUser = (args.length > 10) ? args[10] : RESULTS_USER;
    iResultsPasswd = (args.length > 11) ? args[11] : RESULTS_PASS;

    // register the driver
    try {
      // Register JDBC driver
      Class.forName(driver);
    } catch (ClassNotFoundException ex) {
      logger.error("Could not load class " + driver + " " + ex.getMessage());
      System.exit(1);
    }

    UUID uuid = UUID.randomUUID();
    rid = uuid.toString();
    java.util.Date date = new java.util.Date();
    Timestamp t = new Timestamp(date.getTime());
    rTimestamp = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss").format(t);

    System.out.println("run id is " + rid + " date is " + rTimestamp);

    // read from query file and execute queries
    String sCurrentLine;
    List<String> resultArray = new ArrayList();
    Map<String, String> queryIDMap = new LinkedHashMap();
    BufferedReader br;
    try {
      br = new BufferedReader(new FileReader(queryFile));

      while ((sCurrentLine = br.readLine()) != null) {
        queryIDMap.put(sCurrentLine, null);
      }
      br.close();

    } catch (FileNotFoundException ex) {
      logger.error("Could not find file " + queryFile + " " + ex.getMessage());
      System.exit(2);
    } catch (IOException ex) {
      logger.error("IO Exeception " + ex.getMessage());
      System.exit(3);
    }

    bencherCon = getConnection("jdbc:omnisci:" + queryResultMachine + ":6274:mapd",
            iResultsUser,
            iResultsPasswd);

    getQueries(queryIDMap, bencherCon, tableName);

    runQueries(resultArray, queryIDMap, iterations);

    // if all completed ok store the results
    storeResults();

    // All done dump out results
    System.out.println(header2);
    for (String s : resultArray) {
      System.out.println(s);
    }
  }

  Connection getConnection(String url, String iUser, String iPasswd) {
    // Open a connection
    logger.debug("Connecting to database url :" + url);
    try {
      Connection conn = DriverManager.getConnection(url, iUser, iPasswd);

      targetDBVersion = conn.getMetaData().getDatabaseProductVersion();
      logger.debug("Target DB version is " + targetDBVersion);

      return conn;
    } catch (SQLException ex) {
      logger.error(
              "Exception making connection to " + url + " text is " + ex.getMessage());
      System.exit(2);
    }
    return null;
  }

  String executeQuery(Connection conn1, String qid, String sql, int iterations) {
    Statement stmt = null;
    Connection conn = getConnection(url, iUser, iPasswd);

    Long firstExecute = 0l;
    Long firstJdbc = 0l;
    Long firstIterate = 0l;

    DescriptiveStatistics statsExecute = new DescriptiveStatistics();
    DescriptiveStatistics statsJdbc = new DescriptiveStatistics();
    DescriptiveStatistics statsIterate = new DescriptiveStatistics();
    DescriptiveStatistics statsTotal = new DescriptiveStatistics();

    long totalTime = 0;
    int resultCount = 0;
    try {
      long startTime = System.currentTimeMillis();
      for (int loop = 0; loop < iterations; loop++) {
        // Execute a query
        stmt = conn.createStatement();

        long timer = System.currentTimeMillis();
        if (loop == 0) {
          System.out.println(String.format("Query Id is %s : query is '%s'", qid, sql));
        }
        ResultSet rs = stmt.executeQuery(sql);

        long executeTime = 0;
        long jdbcTime = 0;

        // gather internal execute time for OmniSci as we are interested in that
        if (driver.equals(JDBC_DRIVER)) {
          executeTime = ((OmniSciStatement) stmt).getQueryInternalExecuteTime();
          jdbcTime = (System.currentTimeMillis() - timer) - executeTime;
        } else {
          jdbcTime = (System.currentTimeMillis() - timer);
          executeTime = 0;
        }
        // this is fake to get our intenal execute time.
        logger.debug("Query Timeout/AKA internal Execution Time was "
                + stmt.getQueryTimeout() + " ms Elapsed time in JVM space was "
                + (System.currentTimeMillis() - timer) + "ms");

        timer = System.currentTimeMillis();
        // Extract data from result set
        resultCount = 0;
        while (rs.next()) {
          Object obj = rs.getObject(1);
          if (obj != null && obj.equals(statsExecute)) {
            logger.info("Impossible");
          }
          resultCount++;
        }
        long iterateTime = (System.currentTimeMillis() - timer);

        // if (resultCount != expected) {
        // logger.error("Expect " + expected + " actual " + resultCount + " for
        // query " + sql);
        // // don't run anymore
        // break;
        // }
        if (loop == 0) {
          firstJdbc = jdbcTime;
          firstExecute = executeTime;
          firstIterate = iterateTime;

        } else {
          statsJdbc.addValue(jdbcTime);
          statsExecute.addValue(executeTime);
          statsIterate.addValue(iterateTime);
          statsTotal.addValue(jdbcTime + executeTime + iterateTime);
        }

        // Clean-up environment
        rs.close();
        stmt.close();
      }
      totalTime = System.currentTimeMillis() - startTime;
      conn.close();
    } catch (SQLException se) {
      // Handle errors for JDBC
      se.printStackTrace();
      System.exit(4);
    } catch (Exception e) {
      // Handle errors for Class.forName
      e.printStackTrace();
      System.exit(3);
    } finally {
      // finally block used to close resources
      try {
        if (stmt != null) {
          stmt.close();
        }
      } catch (SQLException se2) {
      } // nothing we can do
      try {
        if (conn != null) {
          conn.close();
        }
      } catch (SQLException se) {
        se.printStackTrace();
        System.exit(6);
      } // end finally try
    } // end try

    // write it to the db here as well
    String insertPart = String.format(insertDescriptor,
            this.rid,
            this.rTimestamp,
            url,
            this.driver,
            label,
            gpuCount,
            this.tableName,
            qid,
            resultCount,
            "",
            statsTotal.getMean(),
            statsTotal.getMin(),
            statsTotal.getMax(),
            statsTotal.getPercentile(85),
            statsExecute.getMean(),
            statsExecute.getMin(),
            statsExecute.getMax(),
            statsExecute.getPercentile(85),
            statsExecute.getPercentile(25),
            statsExecute.getStandardDeviation(),
            statsJdbc.getMean(),
            statsJdbc.getMin(),
            statsJdbc.getMax(),
            statsJdbc.getPercentile(85),
            statsIterate.getMean(),
            statsIterate.getMin(),
            statsIterate.getMax(),
            statsIterate.getPercentile(85),
            firstExecute,
            firstJdbc,
            firstIterate,
            iterations,
            totalTime,
            (long) statsTotal.getSum() + firstExecute + firstJdbc + firstIterate,
            targetDBVersion);

    LResult.add("Insert into results values " + insertPart);

    return String.format(lineDescriptor,
            qid,
            statsTotal.getMean(),
            statsTotal.getMin(),
            statsTotal.getMax(),
            statsTotal.getPercentile(85),
            statsExecute.getMean(),
            statsExecute.getMin(),
            statsExecute.getMax(),
            statsExecute.getPercentile(85),
            statsExecute.getPercentile(25),
            statsExecute.getStandardDeviation(),
            statsJdbc.getMean(),
            statsJdbc.getMin(),
            statsJdbc.getMax(),
            statsJdbc.getPercentile(85),
            statsIterate.getMean(),
            statsIterate.getMin(),
            statsIterate.getMax(),
            statsIterate.getPercentile(85),
            firstExecute,
            firstJdbc,
            firstIterate,
            iterations,
            totalTime,
            (long) statsTotal.getSum() + firstExecute + firstJdbc + firstIterate);
  }

  private void getQueries(
          Map<String, String> queryIDMap, Connection benderCon, String tableName) {
    for (Map.Entry<String, String> entry : queryIDMap.entrySet()) {
      String key = entry.getKey();
      String value = entry.getValue();

      Statement stmt = null;
      try {
        stmt = benderCon.createStatement();
      } catch (SQLException ex) {
        logger.error("Exception creating statement text is " + ex.getMessage());
        System.exit(2);
      }
      String sql =
              String.format("Select query_text from queries where query_id = '%s'", key);
      ResultSet rs = null;
      try {
        rs = stmt.executeQuery(sql);
      } catch (SQLException ex) {
        logger.error("Exception running query " + sql + " text is " + ex.getMessage());
        System.exit(2);
      }
      int resultCount = 0;
      try {
        while (rs.next()) {
          String qString = rs.getString(1);
          qString = qString.replaceAll("##TAB##", tableName);
          // System.out.println(String.format("Query Id is %s : query is '%s'", key,
          // qString));
          queryIDMap.put(key, qString);
          resultCount++;
        }
      } catch (SQLException ex) {
        logger.error("Exception making next call text is " + ex.getMessage());
        System.exit(2);
      }
      if (resultCount > 1) {
        System.out.println("multiple values for queryId " + key);
      }
    }
  }

  private void runQueries(
          List<String> resultArray, Map<String, String> queryIDMap, int iterations) {
    Connection conn = getConnection(url, iUser, iPasswd);
    for (Map.Entry<String, String> entry : queryIDMap.entrySet()) {
      String id = entry.getKey();
      String query = entry.getValue();

      resultArray.add(executeQuery(conn, id, query, iterations));
    }
  }

  private void storeResults() {
    for (String insertPart : LResult) {
      Statement sin;
      try {
        sin = bencherCon.createStatement();
        sin.execute(insertPart);
      } catch (SQLException ex) {
        logger.error("Exception performing insert '" + insertPart + "' text is "
                + ex.getMessage());
        System.exit(2);
      }
    }
  }
}
