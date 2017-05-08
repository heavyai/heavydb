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

//STEP 1. Import required packages
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class Benchmark {

  final static Logger logger = LoggerFactory.getLogger(Benchmark.class);

  // JDBC driver name and database URL
  static final String JDBC_DRIVER = "com.mapd.jdbc.MapDDriver";
  static final String DB_URL = "jdbc:mapd:localhost:9091:mapd";

  //  Database credentials
  static final String USER = "mapd";
  static final String PASS = "HyperInteractive";

  private String driver;
  private String url;
  private String iUser;
  private String iPasswd;

  private String headDescriptor = "%3s, %8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s,%8s";
  private String header2 = String.format(headDescriptor, "QRY", "T-Avg", "T-Min", "T-Max", "T-85%",
          "E-Avg", "E-Min", "E-Max", "E-85%","E-25%", "E-StdD",
          "J-Avg", "J-Min", "J-Max", "J-85%",
          "I-Avg", "I-Min", "I-Max", "I-85%",
          "F-Exec", "F-jdbc", "F-iter", "ITER", "Total", "Account");
  private String lineDescriptor = "Q%02d, %8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8d,%8d,%8d,%8d,%8d,%8d";

  public static void main(String[] args) {
    Benchmark bm = new Benchmark();
    bm.doWork(args, 1);
  }

  void doWork(String[] args, int query) {

    //Grab parameters from args
    // parm0 number of iterations per query
    // parm1 file containing sql queries {contains quoted query, expected result count]
    // parm2 optional JDBC Driver class name
    // parm3 optional DB URL
    // parm4 optionsl user
    // parm5 optional passwd
    int iterations = Integer.valueOf(args[0]);
    logger.debug("Iterations per query is " + iterations);


    String queryFile = args[1];

    //int expectedResults = Integer.valueOf(args[2]);
    driver = (args.length > 2) ? args[2] : JDBC_DRIVER;
    url = (args.length > 3) ? args[3] : DB_URL;
    iUser = (args.length > 4) ? args[4] : USER;
    iPasswd = (args.length > 5) ? args[5] : PASS;

    //register the driver
    try {
      //Register JDBC driver
      Class.forName(driver);
    } catch (ClassNotFoundException ex) {
      logger.error("Could not load class " + driver + " " + ex.getMessage());
      System.exit(1);
    }

    // read from query file and execute queries
    String sCurrentLine;
    List<String> resultArray = new ArrayList();
    BufferedReader br;
    try {
      br = new BufferedReader(new FileReader(queryFile));
      int qCount = 1;

      while ((sCurrentLine = br.readLine()) != null) {

        int expected=0;
        String sqlQuery = null;
        // find the last comma and then grab the rest as sql
        for (int i = sCurrentLine.length(); i > 0; i--){
          if (sCurrentLine.charAt(i-1) == ',') {
            // found the comma
            expected = Integer.valueOf(sCurrentLine.substring(i).trim());
            sqlQuery = sCurrentLine.substring(0, i-1).trim().substring(1);
            break;
          }
        }
        // remove final "
        sqlQuery = sqlQuery.substring(0, sqlQuery.length()-1);

        System.out.println(String.format("Q%02d %s", qCount, sqlQuery));

        resultArray.add(executeQuery(sqlQuery, expected, iterations, qCount));

        qCount++;
      }
    } catch (FileNotFoundException ex) {
      logger.error("Could not find file " + queryFile + " " + ex.getMessage());
      System.exit(2);
    } catch (IOException ex) {
      logger.error("IO Exeception " + ex.getMessage());
      System.exit(3);
    }

    // All done dump out results
    System.out.println(header2);
    for(String s: resultArray){
      System.out.println(s);
    }

  }

  String executeQuery(String sql, int expected, int iterations, int queryNum) {
    Connection conn = null;
    Statement stmt = null;

    Long firstExecute = 0l;
    Long firstJdbc = 0l;
    Long firstIterate = 0l;

    DescriptiveStatistics statsExecute = new DescriptiveStatistics();
    DescriptiveStatistics statsJdbc = new DescriptiveStatistics();
    DescriptiveStatistics statsIterate = new DescriptiveStatistics();
    DescriptiveStatistics statsTotal = new DescriptiveStatistics();

    long totalTime = 0;

    try {
      //Open a connection
      logger.debug("Connecting to database url :" + url);
      conn = DriverManager.getConnection(url, iUser, iPasswd);

      long startTime = System.currentTimeMillis();
      for (int loop = 0; loop < iterations; loop++) {

        //Execute a query
        stmt = conn.createStatement();

        long timer = System.currentTimeMillis();
        ResultSet rs = stmt.executeQuery(sql);

        long executeTime = 0;
        long jdbcTime = 0;

        // gather internal execute time for MapD as we are interested in that
        if (driver.equals(JDBC_DRIVER)){
          executeTime = stmt.getQueryTimeout();
          jdbcTime = (System.currentTimeMillis() - timer) - executeTime;
        } else {
          jdbcTime = (System.currentTimeMillis() - timer);
          executeTime = 0;
        }
        // this is fake to get our intenal execute time.
        logger.debug("Query Timeout/AKA internal Execution Time was " + stmt.getQueryTimeout() + " ms Elapsed time in JVM space was " + (System.currentTimeMillis() - timer) + "ms");

        timer = System.currentTimeMillis();
        //Extract data from result set
        int resultCount = 0;
        while (rs.next()) {
          Object obj = rs.getObject(1);
          if (obj != null && obj.equals(statsExecute)) {
            logger.info("Impossible");
          }
          resultCount++;
        }
        long iterateTime = (System.currentTimeMillis() - timer);

        if (resultCount != expected) {
          logger.error("Expect " + expected + " actual " + resultCount + " for query " + sql);
          // don't run anymore
          break;
        }

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

        //Clean-up environment
        rs.close();
        stmt.close();
      }
      totalTime = System.currentTimeMillis() - startTime;
      conn.close();
    } catch (SQLException se) {
      //Handle errors for JDBC
      se.printStackTrace();
    } catch (Exception e) {
      //Handle errors for Class.forName
      e.printStackTrace();
    } finally {
      //finally block used to close resources
      try {
        if (stmt != null) {
          stmt.close();
        }
      } catch (SQLException se2) {
      }// nothing we can do
      try {
        if (conn != null) {
          conn.close();
        }
      } catch (SQLException se) {
        se.printStackTrace();
      }//end finally try
    }//end try

    return String.format(lineDescriptor,
            queryNum,
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
            (long) statsTotal.getSum()+ firstExecute + firstJdbc + firstIterate);

  }
}
