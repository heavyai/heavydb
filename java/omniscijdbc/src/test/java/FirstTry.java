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

// STEP 1. Import required packages

import java.math.BigDecimal;
import java.sql.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FirstTry {
  final static Logger logger = LoggerFactory.getLogger(FirstTry.class);

  // JDBC driver name and database URL
  static final String JDBC_DRIVER = "com.omnisci.jdbc.OmniSciDriver";
  static final String DB_URL = "jdbc:omnisci:localhost:6273:mapd:http";

  //  Database credentials
  static final String USER = "mapd";
  static final String PASS = "HyperInteractive";

  public static void main(String[] args) {
    Connection conn = null;
    Statement stmt = null;
    try {
      // STEP 2: Register JDBC driver
      Class.forName(JDBC_DRIVER);

      // STEP 3: Open a connection
      logger.info("Connecting to database...");
      conn = DriverManager.getConnection(DB_URL, USER, PASS);

      String sql;

      //      logger.info("Doing prepared statement");
      //
      PreparedStatement ps = null;
      ResultSet rs = null;
      //      sql = "INSERT INTO alltypes (b1, s1, i1, b2, f1, d1, c1, v1, t1, t2, t3, d2)
      //      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"; ps =
      //      conn.prepareStatement(sql);
      //
      //      ps.setBoolean(1, true);
      //      ps.setShort(2, (short)1);
      //      ps.setInt(3, 20);
      //      ps.setInt(4, 400);
      //      ps.setFloat(5, (float)4000.04);
      //      ps.setBigDecimal(6, new BigDecimal(12.2));
      //      ps.setString(7, "String1");
      //      ps.setString(8, "String2");
      //      ps.setString(9, "String3");
      //      ps.setTime(10, new Time(0));
      //      ps.setTimestamp(11, new Timestamp(0));
      //      ps.setDate(12, new Date(0));
      //

      //    ResultSet rs = ps.executeQuery();

      // STEP 4: Execute a query
      logger.info("Creating statement...");
      stmt = conn.createStatement();

      sql = "SELECT uniquecarrier from flights_2008_7M limit 5";
      rs = stmt.executeQuery(sql);

      // STEP 5: Extract data from result set
      while (rs.next()) {
        // Retrieve by column name
        // int id  = rs.getInt("id");
        // int age = rs.getInt("age");
        String uniquecarrier = rs.getString("uniquecarrier");
        // String last = rs.getString("last");

        // Display values
        logger.info("uniquecarrier: " + uniquecarrier);
        // logger.info(", Age: " + age);
        // logger.info(", First: " + first);
        // logger.info(", Last: " + last);
      }
      logger.info("Doing prepared statement");

      ps = null;
      sql = "SELECT uniquecarrier from flights limit 5";
      ps = conn.prepareStatement(sql);

      rs = ps.executeQuery();

      // STEP 5: Extract data from result set
      while (rs.next()) {
        // Retrieve by column name
        // int id  = rs.getInt("id");
        // int age = rs.getInt("age");
        String uniquecarrier = rs.getString("uniquecarrier");
        // String last = rs.getString("last");

        // Display values
        logger.info("PS uniquecarrier: " + uniquecarrier);
        // logger.info(", Age: " + age);
        // logger.info(", First: " + first);
        // logger.info(", Last: " + last);
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

      // STEP 6: Clean-up environment
      rs.close();
      stmt.close();
      conn.close();
    } catch (SQLException se) {
      // Handle errors for JDBC
      se.printStackTrace();
    } catch (Exception e) {
      // Handle errors for Class.forName
      e.printStackTrace();
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
      } // end finally try
    } // end try
    logger.info("Goodbye!");
  } // end main
} // end FirstExample
