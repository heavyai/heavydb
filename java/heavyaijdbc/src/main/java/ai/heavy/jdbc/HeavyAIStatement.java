/*
 * Copyright 2022 HEAVY.AI, Inc.
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
package ai.heavy.jdbc;

import org.apache.thrift.TException;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.SQLWarning;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ai.heavy.thrift.server.Heavy;
import ai.heavy.thrift.server.TDBException;
import ai.heavy.thrift.server.TQueryResult;

public class HeavyAIStatement implements java.sql.Statement {
  final static org.slf4j.Logger logger = LoggerFactory.getLogger(HeavyAIStatement.class);
  public SQLWarning rootWarning = null;

  private String session;
  private Heavy.Client client;
  private HeavyAIConnection connection;
  private ResultSet currentRS = null;
  private TQueryResult sqlResult = null;
  private int maxRows; // add limit to unlimited queries
  private boolean escapeProcessing = false;
  private boolean isClosed = false;

  HeavyAIStatement(String tsession, HeavyAIConnection tconnection) {
    session = tsession;
    connection = tconnection;
    client = connection.client;
    maxRows = Integer.parseInt(connection.cP.getProperty(Options.max_rows));
  }

  static Pattern top_pattern =
          Pattern.compile("select top\\s+([0-9]+)\\s+", Pattern.CASE_INSENSITIVE);

  @Override
  public ResultSet executeQuery(String sql)
          throws SQLException { // logger.debug("Entered");
    checkClosed();
    // @TODO: we can and probably should use "first_n" parameter of the
    // sql_execute()
    // endpoint to force the limit on the query, instead of rewriting it here.
    if (maxRows >= 0) {
      // add limit to sql call if it doesn't already have one and is a select
      String[] tokens = sql.toLowerCase().split(" ", 3);
      if (tokens[0].equals("select")) {
        if (sql.toLowerCase().contains("limit")) {
          // do nothing -
        } else {
          // Some applications add TOP <number> to limit the
          // select statement rather than limit. Remove TOP and keep
          // the number it used as the limit.
          Matcher matcher = top_pattern.matcher(sql);
          // Take "select TOP nnnn <rest ot sql>" and translate to select <reset of sql:
          // limit nnnn
          if (matcher.find()) {
            maxRows = Integer.parseInt(matcher.group(1));
            sql = top_pattern.matcher(sql).replaceAll("select ");
          }

          sql = sql + " LIMIT " + maxRows;
          logger.debug("Added LIMIT of " + maxRows);
        }
      }
    }

    logger.debug("Before HeavyAIEscapeParser [" + sql + "]");
    // The order of these to SQL re-writes is important.
    // EscapeParse needs to come first.
    String afterEscapeParseSQL = HeavyAIEscapeParser.parse(sql);
    String afterSimpleParse = simplisticDateTransform(afterEscapeParseSQL);
    logger.debug("After HeavyAIEscapeParser [" + afterSimpleParse + "]");
    try {
      sqlResult = client.sql_execute(session, afterSimpleParse + ";", true, null, -1, -1);
    } catch (TDBException ex) {
      throw new SQLException(
              "Query failed : " + HeavyAIExceptionText.getExceptionDetail(ex));
    } catch (TException ex) {
      throw new SQLException(
              "Query failed : " + HeavyAIExceptionText.getExceptionDetail(ex));
    }

    currentRS = new HeavyAIResultSet(sqlResult, sql);
    return currentRS;
  }

  @Override
  public void cancel() throws SQLException { // logger.debug("Entered");
    checkClosed();
    HeavyAIConnection alternate_connection = null;
    try {
      alternate_connection = connection.getAlternateConnection();
      // Note alternate_connection shares a session with original connection
      alternate_connection.client.interrupt(session, session);
    } catch (TDBException ttE) {
      throw new SQLException("Thrift transport connection failed - "
                      + HeavyAIExceptionText.getExceptionDetail(ttE),
              ttE);
    } catch (TException tE) {
      throw new SQLException(
              "Thrift failed - " + HeavyAIExceptionText.getExceptionDetail(tE), tE);
    } finally {
      // Note closeConnection only closes the underlying thrft connection
      // not the logical db session connection
      alternate_connection.closeConnection();
    }
  }

  @Override
  public int executeUpdate(String sql) throws SQLException { // logger.debug("Entered");
    checkClosed();
    try {
      // remove " characters if it is a CREATE statement
      if (sql.trim().substring(0, 6).compareToIgnoreCase("CREATE") == 0) {
        sql = sql.replace('"', ' ');
      }
      sqlResult = client.sql_execute(session, sql + ";", true, null, -1, -1);
    } catch (TDBException ex) {
      throw new SQLException("Query failed :  sql was '" + sql + "' "
                      + HeavyAIExceptionText.getExceptionDetail(ex),
              ex);
    } catch (TException ex) {
      throw new SQLException(
              "Query failed : " + HeavyAIExceptionText.getExceptionDetail(ex), ex);
    }

    // TODO: OmniSciDB supports updates, inserts and deletes, but
    // there is no way to get number of affected rows at the moment
    return -1;
  }

  @Override
  public void close() throws SQLException { // logger.debug("Entered");
    if (currentRS != null) {
      currentRS.close();
    }
    isClosed = true;
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
    checkClosed();
    return currentRS;
  }

  @Override
  public int getUpdateCount() throws SQLException { // logger.debug("Entered");
    checkClosed();
    // TODO: OmniSciDB supports updates, inserts and deletes, but
    // there is no way to get number of affected rows at the moment
    return -1;
  }

  @Override
  public boolean getMoreResults() throws SQLException { // logger.debug("Entered");
    checkClosed();
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
    return ResultSet.FETCH_FORWARD;
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
    return connection;
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
    return isClosed;
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

  /*
   * CURRENTDATE should match CURRENT_DATE
   * and CURRENT_DATE() where the two strings are 'joined' to either white space,
   * punctuation or some kind of brackets. if they are joined to
   * any alpha numeric For example 'CURRENT_TIME)' is okay while a string
   * like CURRENT_DATE_NOW isn't
   *
   * Note we've include the non standard version with parenthesis to align with
   * third
   * party software.
   *
   * Breaking down the components of the pattern
   * (?<![\\w.]) The pattern can not be preceded by any word character or a '.'
   * (?:\\(\\))? pattern can end in zero or one '()' - note non capture group
   * (?![\\w.]) the pattern can not be followed by a word character or a '.'
   * Note - word characters include '_'
   */
  ;
  private static final Pattern CURRENTDATE =
          Pattern.compile("(?<![\\w.])CURRENT_DATE(?:\\(\\))?(?![\\w.])",
                  Pattern.DOTALL | Pattern.CASE_INSENSITIVE);

  public static String simplisticDateTransform(String sql) {
    // need to iterate as each reduction of string opens up a anew match
    String start;
    do {
      // Example transform - select quarter(val) from table;
      // will become select extract(quarter from val) from table;
      // will also replace all CURRENT_TIME and CURRENT_DATE with a call to now().
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

    do {
      start = sql;
      sql = CURRENTDATE.matcher(sql).replaceAll(" cast(now() as date) ");
    } while (!sql.equals(start));

    return sql;
  }

  private void checkClosed() throws SQLException {
    if (isClosed) {
      throw new SQLException("Statement is closed.");
    }
  }
}
