package com.mapd.jdbc;



/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


//STEP 1. Import required packages

import java.sql.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FirstTry {

  final static Logger logger = LoggerFactory.getLogger(FirstTry.class);

  // JDBC driver name and database URL
  static final String JDBC_DRIVER = "com.mapd.jdbc.MapDDriver";
  static final String DB_URL = "jdbc:mapd:localhost:9091:mapd";

  //  Database credentials
  static final String USER = "mapd";
  static final String PASS = "HyperInteractive";

  public static void main(String[] args) {
    Connection conn = null;
    Statement stmt = null;
    try {
      //STEP 2: Register JDBC driver
      Class.forName(JDBC_DRIVER);

      //STEP 3: Open a connection
      logger.info("Connecting to database...");
      conn = DriverManager.getConnection(DB_URL, USER, PASS);

      //STEP 4: Execute a query
      logger.info("Creating statement...");
      stmt = conn.createStatement();
      String sql;
      sql = "SELECT uniquecarrier from flights limit 5";
      ResultSet rs = stmt.executeQuery(sql);

      //STEP 5: Extract data from result set
      while (rs.next()) {
        //Retrieve by column name
        //int id  = rs.getInt("id");
        //int age = rs.getInt("age");
        String uniquecarrier = rs.getString("uniquecarrier");
        //String last = rs.getString("last");

        //Display values
        logger.info("uniquecarrier: " + uniquecarrier);
        //logger.info(", Age: " + age);
        //logger.info(", First: " + first);
        //logger.info(", Last: " + last);
      }
      logger.info("Doing prepared statement");

      PreparedStatement ps = null;
      sql = "SELECT uniquecarrier from flights limit 5";
      ps = conn.prepareStatement(sql);

      rs = ps.executeQuery();

      //STEP 5: Extract data from result set
      while (rs.next()) {
        //Retrieve by column name
        //int id  = rs.getInt("id");
        //int age = rs.getInt("age");
        String uniquecarrier = rs.getString("uniquecarrier");
        //String last = rs.getString("last");

        //Display values
        logger.info("PS uniquecarrier: " + uniquecarrier);
        //logger.info(", Age: " + age);
        //logger.info(", First: " + first);
        //logger.info(", Last: " + last);
      }

      DatabaseMetaData dbmeta = conn.getMetaData();

      rs = dbmeta.getSchemas();
      while (rs.next()) {
        String schema = rs.getString("TABLE_SCHEM");
        logger.info("TABLE_SCHEM: " + schema);
      }

      rs = dbmeta.getTableTypes();
      while (rs.next()) {
        String schema = rs.getString(1);
        logger.info("TYPE: " + schema);
      }

      //STEP 6: Clean-up environment
      rs.close();
      stmt.close();
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
    logger.info("Goodbye!");
  }//end main
}//end FirstExample
