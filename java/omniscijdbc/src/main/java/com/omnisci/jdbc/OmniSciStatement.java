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
package com.omnisci.jdbc;

import com.mapd.thrift.server.MapD;
import com.mapd.thrift.server.TMapDException;
import com.mapd.thrift.server.TQueryResult;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.SQLWarning;
import java.util.regex.Pattern;
import org.apache.thrift.TException;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public class OmniSciStatement implements java.sql.Statement {
  final static org.slf4j.Logger logger = LoggerFactory.getLogger(OmniSciStatement.class);
  public SQLWarning rootWarning = null;

  private String session;
  private MapD.Client client;
  private ResultSet currentRS = null;
  private TQueryResult sqlResult = null;
  private int maxRows = 100000; // add limit to unlimited queries
  private boolean escapeProcessing = false;

  OmniSciStatement(String tsession, MapD.Client tclient) {
    session = tsession;
    client = tclient;
  }

  @Override
  public ResultSet executeQuery(String sql)
          throws SQLException { // logger.debug("Entered");
    if (maxRows > 0) {
      // add limit to sql call if it doesn't already have one and is a select
      String[] tokens = sql.toLowerCase().split(" ", 3);
      if (tokens[0].equals("select")) {
        if (sql.toLowerCase().contains("limit")) {
          // do nothing
        } else {
          sql = sql + " LIMIT " + maxRows;
          logger.debug("Added LIMIT of " + maxRows);
        }
      }
    }
    logger.debug("sql is :'" + sql + "'");
    String afterFnSQL = fnReplace(sql);
    logger.debug("afterFnSQL is :'" + afterFnSQL + "'");
    try {
      sqlResult = client.sql_execute(session, afterFnSQL + ";", true, null, -1, -1);
    } catch (TMapDException ex) {
      throw new SQLException("Query failed : " + ex.getError_msg());
    } catch (TException ex) {
      throw new SQLException("Query failed : " + ex.toString());
    }

    currentRS = new OmniSciResultSet(sqlResult, sql);
    return currentRS;
  }

  @Override
  public int executeUpdate(String sql) throws SQLException { // logger.debug("Entered");
    try {
      // remove " characters if it is a CREATE statement
      if (sql.trim().substring(0, 6).compareToIgnoreCase("CREATE") == 0) {
        sql = sql.replace('"', ' ');
      }
      sqlResult = client.sql_execute(session, sql + ";", true, null, -1, -1);
    } catch (TMapDException ex) {
      throw new SQLException(
              "Query failed : " + ex.getError_msg() + " sql was '" + sql + "'");
    } catch (TException ex) {
      throw new SQLException("Query failed : " + ex.toString());
    }

    return sqlResult.row_set.columns.size();
  }

  @Override
  public void close() throws SQLException { // logger.debug("Entered");

    // clean up after -- nothing to do
  }

  @Override
  public int getMaxFieldSize() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setMaxFieldSize(int max) throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getMaxRows() throws SQLException { // logger.debug("Entered");
    return maxRows;
  }

  @Override
  public void setMaxRows(int max) throws SQLException { // logger.debug("Entered");
    maxRows = max;
  }

  @Override
  public void setEscapeProcessing(boolean enable)
          throws SQLException { // logger.debug("Entered");
    escapeProcessing = enable;
  }

  @Override
  public int getQueryTimeout() throws SQLException { // logger.debug("Entered");
    return 0;
  }

  // used by benchmarking to get internal execution times
  public int getQueryInternalExecuteTime()
          throws SQLException { // logger.debug("Entered");
    return (int) sqlResult.execution_time_ms;
  }

  @Override
  public void setQueryTimeout(int seconds)
          throws SQLException { // logger.debug("Entered");
    SQLWarning warning = new SQLWarning(
            "Query timeouts are not supported.  Substituting a value of zero.");
    if (rootWarning == null) {
      rootWarning = warning;
    } else {
      rootWarning.setNextWarning(warning);
    }
  }

  @Override
  public void cancel() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public SQLWarning getWarnings() throws SQLException { // logger.debug("Entered");
    return (rootWarning);
  }

  @Override
  public void clearWarnings() throws SQLException { // logger.debug("Entered");
    rootWarning = null;
  }

  @Override
  public void setCursorName(String name) throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql) throws SQLException { // logger.debug("Entered");
    ResultSet rs = executeQuery(sql);
    if (rs != null) {
      return true;
    } else {
      return false;
    }
  }

  @Override
  public ResultSet getResultSet() throws SQLException { // logger.debug("Entered");
    return currentRS;
  }

  @Override
  public int getUpdateCount() throws SQLException { // logger.debug("Entered");
    // TODO MAT fix update count
    return 0;
  }

  @Override
  public boolean getMoreResults() throws SQLException { // logger.debug("Entered");
    // TODO MAT this needs to be fixed for complex queries
    return false;
  }

  @Override
  public void setFetchDirection(int direction)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getFetchDirection() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setFetchSize(int rows) throws SQLException { // logger.debug("Entered");
    SQLWarning warning = new SQLWarning(
            "Query FetchSize are not supported.  Substituting a value of zero.");
    if (rootWarning == null) {
      rootWarning = warning;
    } else {
      rootWarning.setNextWarning(warning);
    }
  }

  @Override
  public int getFetchSize() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetConcurrency() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetType() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void addBatch(String sql) throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void clearBatch() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int[] executeBatch() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Connection getConnection() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean getMoreResults(int current)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSet getGeneratedKeys() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, int autoGeneratedKeys)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, int[] columnIndexes)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, String[] columnNames)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, int autoGeneratedKeys)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, int[] columnIndexes)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, String[] columnNames)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetHoldability() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isClosed() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setPoolable(boolean poolable)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isPoolable() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void closeOnCompletion() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isCloseOnCompletion() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public <T> T unwrap(Class<T> iface) throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isWrapperFor(Class<?> iface)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  private static final Pattern QUARTER = Pattern.compile(
          "\\sQUARTER\\(([^\\{]*?)", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
  private static final Pattern DAYOFYEAR = Pattern.compile(
          "\\sDAYOFYEAR\\(([^\\{]*?)", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
  private static final Pattern DAYOFWEEK = Pattern.compile(
          "\\sDAYOFWEEK\\(([^\\{]*?)", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
  private static final Pattern WEEK = Pattern.compile(
          "\\sWEEK\\(([^\\{]*?)", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
  private static final Pattern QUARTER_TRUNC = Pattern.compile(
          "\\(\\(\\(CAST\\(([^\\(]*?) AS DATE\\) \\+  FLOOR\\(\\(\\-1 \\* \\( EXTRACT\\(DAY FROM .*?\\) \\- 1\\)\\)\\) \\* INTERVAL '1' DAY\\) \\+  FLOOR\\(\\(\\-1 \\* \\( EXTRACT\\(MONTH FROM .*?\\) \\- 1\\)\\)\\) \\* INTERVAL '1' MONTH\\) \\+  FLOOR\\(\\(3 \\* \\( FLOOR\\( EXTRACT\\(QUARTER FROM .*?\\)\\) - 1\\)\\)\\) \\* INTERVAL '1' MONTH\\)",
          Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
  private static final Pattern MONTH_TRUNC = Pattern.compile(
          "\\(CAST\\(([^\\(]*?) AS DATE\\) \\+  FLOOR\\(\\(\\-1 \\* \\( EXTRACT\\(DAY FROM .*?\\) \\- 1\\)\\)\\) \\* INTERVAL '1' DAY\\)",
          Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
  private static final Pattern YEAR_TRUNC = Pattern.compile(
          "\\(\\(CAST\\(([^\\(]*?) AS DATE\\) \\+  FLOOR\\(\\(\\-1 \\* \\( EXTRACT\\(DAY FROM .*?\\) \\- 1\\)\\)\\) \\* INTERVAL '1' DAY\\) \\+  FLOOR\\(\\(\\-1\\ \\* \\( EXTRACT\\(MONTH FROM .*?\\) \\- 1\\)\\)\\) \\* INTERVAL '1' MONTH\\)",
          Pattern.DOTALL | Pattern.CASE_INSENSITIVE);

  private static final Pattern MINUTE_TRUNC = Pattern.compile(
          "\\(\\(CAST\\(([^\\(]*?) AS DATE\\) \\+  EXTRACT\\(HOUR FROM .*?\\) \\* INTERVAL '1' HOUR\\) \\+  EXTRACT\\(MINUTE FROM .*?\\) \\* INTERVAL '1' MINUTE\\)",
          Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
  private static final Pattern SECOND_TRUNC = Pattern.compile(
          "\\(\\(\\(CAST\\(([^\\(]*?) AS DATE\\) \\+  EXTRACT\\(HOUR FROM .*?\\) \\* INTERVAL '1' HOUR\\) \\+  EXTRACT\\(MINUTE FROM .*?\\) \\* INTERVAL '1' MINUTE\\) \\+  EXTRACT\\(SECOND FROM .*?\\) \\* INTERVAL '1' SECOND\\)",
          Pattern.DOTALL | Pattern.CASE_INSENSITIVE);

  private static final Pattern YEAR1_TRUNC = Pattern.compile(
          "\\(CAST\\(([^\\(]*?) AS DATE\\) \\+  FLOOR\\(\\(\\-1 \\* \\( EXTRACT\\(DOY FROM .*?\\) \\- 1\\)\\)\\) \\* INTERVAL '1' DAY\\)",
          Pattern.DOTALL | Pattern.CASE_INSENSITIVE);

  private static final Pattern QUARTER1_TRUNC = Pattern.compile(
          "\\(\\(CAST\\(([^\\(]*?) AS DATE\\) \\+  FLOOR\\(\\(\\-1 \\* \\( EXTRACT\\(DOY FROM .*?\\) \\- 1\\)\\)\\) \\* INTERVAL '1' DAY\\) \\+  FLOOR\\(\\(3 \\* \\( FLOOR\\( EXTRACT\\(QUARTER FROM .*?\\)\\) \\- 1\\)\\)\\) \\* INTERVAL '1' MONTH\\)",
          Pattern.DOTALL | Pattern.CASE_INSENSITIVE);

  private static final Pattern WEEK_TRUNC = Pattern.compile(
          "\\(CAST\\(([^\\(]*?) AS DATE\\) \\+ \\(\\-1 \\* \\( EXTRACT\\(ISODOW FROM .*?\\) \\- 1\\)\\) \\* INTERVAL '1' DAY\\)",
          Pattern.DOTALL | Pattern.CASE_INSENSITIVE);

  public static String fnReplace(String sql) {
    // need to iterate as each reduction of string opens up a anew match
    String start;
    do {
      start = sql;
      sql = QUARTER.matcher(sql).replaceAll(" EXTRACT(QUARTER FROM $1");
    } while (!sql.equals(start));

    do {
      start = sql;
      sql = DAYOFYEAR.matcher(sql).replaceAll(" EXTRACT(DOY FROM $1");
    } while (!sql.equals(start));

    do {
      start = sql;
      sql = DAYOFWEEK.matcher(sql).replaceAll(" EXTRACT(ISODOW FROM $1");
    } while (!sql.equals(start));

    do {
      start = sql;
      sql = WEEK.matcher(sql).replaceAll(" EXTRACT(WEEK FROM $1");
    } while (!sql.equals(start));

    // Order is important here, do not shuffle without checking
    sql = QUARTER_TRUNC.matcher(sql).replaceAll(" DATE_TRUNC(QUARTER, $1)");
    sql = YEAR_TRUNC.matcher(sql).replaceAll(" DATE_TRUNC(YEAR, $1)");
    sql = SECOND_TRUNC.matcher(sql).replaceAll(" DATE_TRUNC(SECOND, $1)");
    sql = QUARTER1_TRUNC.matcher(sql).replaceAll(" DATE_TRUNC(QUARTER, $1)");
    sql = MONTH_TRUNC.matcher(sql).replaceAll(" DATE_TRUNC(MONTH, $1)");
    sql = MINUTE_TRUNC.matcher(sql).replaceAll(" DATE_TRUNC(MINUTE, $1)");
    sql = YEAR1_TRUNC.matcher(sql).replaceAll(" DATE_TRUNC(YEAR, $1)");
    sql = WEEK_TRUNC.matcher(sql).replaceAll(" DATE_TRUNC(WEEK, $1)");

    do {
      start = sql;
      sql = QUARTER.matcher(sql).replaceAll(" EXTRACT(QUARTER FROM $1");
    } while (!sql.equals(start));

    do {
      start = sql;
      sql = DAYOFYEAR.matcher(sql).replaceAll(" EXTRACT(DOY FROM $1");
    } while (!sql.equals(start));

    do {
      start = sql;
      sql = DAYOFWEEK.matcher(sql).replaceAll(" EXTRACT(ISODOW FROM $1");
    } while (!sql.equals(start));

    do {
      start = sql;
      sql = WEEK.matcher(sql).replaceAll(" EXTRACT(WEEK FROM $1");
    } while (!sql.equals(start));

    return sql;
  }
}
