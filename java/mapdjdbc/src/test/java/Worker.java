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

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class Worker implements Runnable {

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(Worker.class);
  // JDBC driver name and database URL
  static final String JDBC_DRIVER = "com.mapd.jdbc.MapDDriver";
  static final String DB_URL = "jdbc:mapd:localhost:9091:mapd";

  //  Database credentials
  static final String USER = "mapd";
  static final String PASS = "HyperInteractive";

  private Connection conn = null;
  private Statement stmt = null;

  Random r = new Random();

  int threadId;
  int numOfTables;

  Worker(int ti, int nt) {
    threadId = ti;
    numOfTables = nt;

    try {
      //STEP 2: Register JDBC driver
      Class.forName(JDBC_DRIVER);

      //STEP 3: Open a connection
      MAPDLOGGER.info(threadId + ": Connecting to database...");
      conn = DriverManager.getConnection(DB_URL, USER, PASS);
      stmt = conn.createStatement();

      for (int i = 0; i < numOfTables; i++) {
        //doQuery(conn, "drop table t" + i, stmt);
        doQuery(conn, "create  table if not exists t" + i + " (newn int)", stmt);
      }
    } catch (SQLException se) {
      //Handle errors for JDBC
      se.printStackTrace();

    } catch (ClassNotFoundException ex) {
      ex.printStackTrace();
    }//end try
  }

  public void run() {
    try {
      int tab = this.threadId;
      for (int i = 0; i < 100; i++) {
        doInsertQuery(conn, stmt,tab);
        doSelectQuery(conn, stmt, tab);
        tab++;
        if (tab >= this.numOfTables){
          tab = 0;
        }
      }

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
  }//end main

  private void doQuery(Connection conn, String sql, Statement stmt) throws SQLException {
    ResultSet rs = null;
    try {
       rs = stmt.executeQuery(sql);
    } catch (SQLException se) {
      if (se.getMessage().contains("does not exist")) {
        MAPDLOGGER.info(threadId + ": table not present");
        return;
      } else {
        throw se;
      }
    }
    while (rs.next()) {
      int sum = rs.getInt(1);

      MAPDLOGGER.info(threadId + ": result " + sum);
    }
    rs.close();
  }

  private void doInsertQuery(Connection conn, Statement stmt, int tab) throws SQLException {

    doQuery(conn, "insert into t"+tab+" values (1)", stmt);

  }

  private void doSelectQuery(Connection conn, Statement stmt, int tab) throws SQLException {
    doQuery(conn, ("select sum(newn) from t"+tab), stmt);
  }

}//end FirstExample
