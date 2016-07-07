/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mapd.jdbc;

import com.mapd.thrift.server.MapD;
import com.mapd.thrift.server.TQueryResult;
import com.mapd.thrift.server.ThriftException;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.SQLWarning;
import org.apache.thrift.TException;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public class MapDStatement implements java.sql.Statement {

  final static org.slf4j.Logger logger = LoggerFactory.getLogger(MapDStatement.class);
  private int session;
  private MapD.Client client;
  private ResultSet currentRS = null;
  private TQueryResult sqlResult = null;
  private int maxRows=100; // add limit to unlimited queries

  MapDStatement(int tsession, MapD.Client tclient) {
    session = tsession;
    client = tclient;
  }

  @Override
  public ResultSet executeQuery(String sql) throws SQLException { //logger.debug("Entered");
    if (maxRows > 0) {
      // add limit to sql call if it doesn't already have one
      if (sql.toLowerCase().contains(" limit ")) {
        // do nothing
      } else {
        sql = sql + " LIMIT " + maxRows;
        logger.info("Added LIMIT of "+ maxRows);
      }
    }
    logger.debug("sql is :'" + sql + "'");
    try {
      sqlResult = client.sql_execute(session, sql + ";", true, null);
    } catch (ThriftException ex) {
      throw new SQLException("Query failed : " + ex.getError_msg());
    } catch (TException ex) {
      throw new SQLException("Query failed : " + ex.toString());
    }

    currentRS = new MapDResultSet(sqlResult, sql);
    return currentRS;
  }

  @Override
  public int executeUpdate(String sql) throws SQLException { //logger.debug("Entered");
    try {
      sqlResult = client.sql_execute(session, sql + ";", true, null);
    } catch (ThriftException ex) {
      throw new SQLException("Query failed : " + ex.getError_msg());
    } catch (TException ex) {
      throw new SQLException("Query failed : " + ex.toString());
    }

    return sqlResult.row_set.columns.size();
  }

  @Override
  public void close() throws SQLException { //logger.debug("Entered");

    // clean up after -- nothing to do
  }

  @Override
  public int getMaxFieldSize() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void setMaxFieldSize(int max) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public int getMaxRows() throws SQLException { //logger.debug("Entered");
    return maxRows;
  }

  @Override
  public void setMaxRows(int max) throws SQLException { //logger.debug("Entered");
    maxRows = max;
  }

  @Override
  public void setEscapeProcessing(boolean enable) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public int getQueryTimeout() throws SQLException { //logger.debug("Entered");
    //TODO MAT have overloaded this option to allow for internal time comparisson
    return (int)sqlResult.execution_time_ms;
  }

  @Override
  public void setQueryTimeout(int seconds) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void cancel() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SQLWarning getWarnings() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void clearWarnings() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void setCursorName(String name) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean execute(String sql) throws SQLException { //logger.debug("Entered");
    ResultSet rs = executeQuery(sql);
    if (rs != null){
      return true;
    } else {
      return false;
    }
  }

  @Override
  public ResultSet getResultSet() throws SQLException { //logger.debug("Entered");
    return currentRS;
  }

  @Override
  public int getUpdateCount() throws SQLException { //logger.debug("Entered");
    //TODO MAT fix update count
    return 0;
  }

  @Override
  public boolean getMoreResults() throws SQLException { //logger.debug("Entered");
    //TODO MAT this needs to be fixed for complex queries
    return false;
  }

  @Override
  public void setFetchDirection(int direction) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public int getFetchDirection() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void setFetchSize(int rows) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public int getFetchSize() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public int getResultSetConcurrency() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public int getResultSetType() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void addBatch(String sql) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void clearBatch() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public int[] executeBatch() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public Connection getConnection() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean getMoreResults(int current) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getGeneratedKeys() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public int executeUpdate(String sql, int autoGeneratedKeys) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public int executeUpdate(String sql, int[] columnIndexes) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public int executeUpdate(String sql, String[] columnNames) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean execute(String sql, int autoGeneratedKeys) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean execute(String sql, int[] columnIndexes) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean execute(String sql, String[] columnNames) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public int getResultSetHoldability() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean isClosed() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void setPoolable(boolean poolable) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean isPoolable() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void closeOnCompletion() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean isCloseOnCompletion() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public <T> T unwrap(Class<T> iface) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean isWrapperFor(Class<?> iface) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

}
