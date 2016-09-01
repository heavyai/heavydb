package com.mapd.jdbc;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
//STEP 1. Import required packages
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.logging.Level;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class BenchmarkCloud {

  final static Logger logger = LoggerFactory.getLogger(BenchmarkCloud.class);

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
  private String rid;
  private String rTimestamp;
  private String tableName;
  private String label;
  private String gpuCount;
  Connection benderCon;

  private String headDescriptor = "%3s  %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s";
  private String header2 = "Q       Total                             Execute                                                  JDBC                             Iterate                               First";
  private String header1 = String.format(headDescriptor, "", "Avg", "Min", "Max", "85%",
          "Avg", "Min", "Max", "85%", "25%", "StdD",
          "Avg", "Min", "Max", "85%",
          "Avg", "Min", "Max", "85%",
          "Exec", "jdbc", "iter", "IT", "Total", "Acc");
  private String lineDescriptor = "%3s, %8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8d,%8d,%8d,%8d,%8d,%8d";
  private String insertDescriptor = "('%s','%s','%s','%s','%s',%s,'%s','%s',%d,'%s', %8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8d,%8d,%8d,%8d,%8d,%8d)";

  public static void main(String[] args) {
    BenchmarkCloud bm = new BenchmarkCloud();
    bm.doWork(args, 1);
  }

  void doWork(String[] args, int query) {

    //Grab parameters from args
    // parm0 number of iterations per query
    // parm1 file containing sql queries {contains quoted query, expected result count]
    // parm2 table name
    // parm3 run label
    // parm4 optional gpu count
    // parm5 optional DB URL
    // parm6 optional JDBC Driver class name
    // parm7 optionsl user
    // parm8 optional passwd
    int iterations = Integer.valueOf(args[0]);
    logger.debug("Iterations per query is " + iterations);

    String queryFile = args[1];
    tableName = args[2];
    label = args[3];
    gpuCount = args[4];

    //int expectedResults = Integer.valueOf(args[2]);
    url = (args.length > 5) ? args[5] : DB_URL;
    driver = (args.length > 6) ? args[6] : JDBC_DRIVER;

    iUser = (args.length > 7) ? args[7] : USER;
    iPasswd = (args.length > 8) ? args[8] : PASS;

    //register the driver
    try {
      //Register JDBC driver
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

    benderCon = getConnection("jdbc:mapd:bencher:9091:mapd", "mapd", "HyperInteractive");

    getQueries(queryIDMap, benderCon, tableName);

    runQueries(resultArray, queryIDMap, iterations);

    // All done dump out results
    System.out.println(header1);
    System.out.println(header2);
    for (String s : resultArray) {
      System.out.println(s);
    }

  }

  Connection getConnection(String url, String iUser, String iPasswd) {
//Open a connection
    logger.debug("Connecting to database url :" + url);
    try {
      return DriverManager.getConnection(url, iUser, iPasswd);
    } catch (SQLException ex) {
      logger.error("Exception making connection to" + url + " text is " + ex.getMessage());
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

        //Execute a query
        stmt = conn.createStatement();

        long timer = System.currentTimeMillis();
        ResultSet rs = stmt.executeQuery(sql);

        long executeTime = 0;
        long jdbcTime = 0;

        // gather internal execute time for MapD as we are interested in that
        if (driver.equals(JDBC_DRIVER)) {
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
        resultCount = 0;
        while (rs.next()) {
          Object obj = rs.getObject(1);
          if (obj != null && obj.equals(statsExecute)) {
            logger.info("Impossible");
          }
          resultCount++;
        }
        long iterateTime = (System.currentTimeMillis() - timer);

//        if (resultCount != expected) {
//          logger.error("Expect " + expected + " actual " + resultCount + " for query " + sql);
//          // don't run anymore
//          break;
//        }
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

    // write it to the db here as well
    String insertPart = String.format(insertDescriptor,
            this.rid, this.rTimestamp, url, this.driver, label, gpuCount, this.tableName,
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
            (long) statsTotal.getSum() + firstExecute + firstJdbc + firstIterate);

    System.out.println("Insert into results values " + insertPart);

    Statement sin;
    try {
      sin = benderCon.createStatement();
      sin.execute("Insert into results values " + insertPart);
    } catch (SQLException ex) {
      logger.error("Exception performing insert '" + "Insert into results values " + insertPart + "' text is " + ex.getMessage());
      System.exit(2);
    }

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

  private void getQueries(Map<String, String> queryIDMap, Connection benderCon, String tableName) {

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
      String sql = String.format("Select query_text from queries where query_id = '%s'", key);
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
          System.out.println(String.format("Query Id is %s : query is '%s'", key, qString));
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

  private void runQueries(List<String> resultArray, Map<String, String> queryIDMap, int iterations) {
    Connection conn = getConnection(url, iUser, iPasswd);
    for (Map.Entry<String, String> entry : queryIDMap.entrySet()) {
      String id = entry.getKey();
      String query = entry.getValue();

      resultArray.add(executeQuery(conn, id, query, iterations));
    }
  }
}
