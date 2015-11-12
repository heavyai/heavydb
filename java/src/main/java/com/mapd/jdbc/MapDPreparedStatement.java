/*
 *  Some cool MapD header
 */
package com.mapd.jdbc;

import com.mapd.thrift.server.MapD;
import java.io.InputStream;
import java.io.Reader;
import java.math.BigDecimal;
import java.net.URL;
import java.sql.Array;
import java.sql.Blob;
import java.sql.Clob;
import java.sql.Connection;
import java.sql.Date;
import java.sql.NClob;
import java.sql.ParameterMetaData;
import java.sql.PreparedStatement;
import java.sql.Ref;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.RowId;
import java.sql.SQLException;
import java.sql.SQLWarning;
import java.sql.SQLXML;
import java.sql.Time;
import java.sql.Timestamp;
import java.util.Calendar;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
class MapDPreparedStatement implements PreparedStatement {

  final static Logger logger = LoggerFactory.getLogger(MapDPreparedStatement.class);
  
  private String currentSQL;
  private int session;
  private MapD.Client client;
  private MapDStatement stmt;

  MapDPreparedStatement(String sql, int session, MapD.Client client) {
    currentSQL = sql;
    this.client = client;
    this.session = session;
  }

  @Override
  public ResultSet executeQuery()  throws SQLException { //logger.debug("Entered");
    stmt = new MapDStatement(session, client);
    return stmt.executeQuery(currentSQL);
  }

  @Override
  public int executeUpdate()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setNull(int parameterIndex, int sqlType)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setBoolean(int parameterIndex, boolean x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setByte(int parameterIndex, byte x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setShort(int parameterIndex, short x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setInt(int parameterIndex, int x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setLong(int parameterIndex, long x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setFloat(int parameterIndex, float x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setDouble(int parameterIndex, double x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setBigDecimal(int parameterIndex, BigDecimal x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setString(int parameterIndex, String x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setBytes(int parameterIndex, byte[] x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setDate(int parameterIndex, Date x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setTime(int parameterIndex, Time x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setTimestamp(int parameterIndex, Timestamp x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x, int length)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setUnicodeStream(int parameterIndex, InputStream x, int length)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x, int length)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void clearParameters()  throws SQLException { //logger.debug("Entered");
    //TODO MAT we will actually need to do something here one day
  }

  @Override
  public void setObject(int parameterIndex, Object x, int targetSqlType)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setObject(int parameterIndex, Object x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public boolean execute()  throws SQLException { //logger.debug("Entered");
    stmt = new MapDStatement(session, client);
    return stmt.execute(currentSQL);
  }

  @Override
  public void addBatch()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader, int length)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setRef(int parameterIndex, Ref x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setBlob(int parameterIndex, Blob x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setClob(int parameterIndex, Clob x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setArray(int parameterIndex, Array x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public ResultSetMetaData getMetaData()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setDate(int parameterIndex, Date x, Calendar cal)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setTime(int parameterIndex, Time x, Calendar cal)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setTimestamp(int parameterIndex, Timestamp x, Calendar cal)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setNull(int parameterIndex, int sqlType, String typeName)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setURL(int parameterIndex, URL x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public ParameterMetaData getParameterMetaData()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setRowId(int parameterIndex, RowId x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setNString(int parameterIndex, String value)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setNCharacterStream(int parameterIndex, Reader value, long length)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setNClob(int parameterIndex, NClob value)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setClob(int parameterIndex, Reader reader, long length)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setBlob(int parameterIndex, InputStream inputStream, long length)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setNClob(int parameterIndex, Reader reader, long length)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setSQLXML(int parameterIndex, SQLXML xmlObject)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setObject(int parameterIndex, Object x, int targetSqlType, int scaleOrLength)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x, long length)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x, long length)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader, long length)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setNCharacterStream(int parameterIndex, Reader value)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setClob(int parameterIndex, Reader reader)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setBlob(int parameterIndex, InputStream inputStream)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setNClob(int parameterIndex, Reader reader)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public ResultSet executeQuery(String sql)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int executeUpdate(String sql)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void close()  throws SQLException { //logger.debug("Entered");
    //TODO MAT probably more needed here
    stmt.close();
  }

  @Override
  public int getMaxFieldSize()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setMaxFieldSize(int max)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int getMaxRows()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setMaxRows(int max)  throws SQLException { //logger.debug("Entered");
    logger.info("SetMaxRows to "+max);
    stmt.setMaxRows(max);
  }

  @Override
  public void setEscapeProcessing(boolean enable)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int getQueryTimeout()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setQueryTimeout(int seconds)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void cancel()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public SQLWarning getWarnings()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void clearWarnings()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setCursorName(String name)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public boolean execute(String sql)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public ResultSet getResultSet()  throws SQLException { //logger.debug("Entered");
    return stmt.getResultSet();
  }

  @Override
  public int getUpdateCount()  throws SQLException { //logger.debug("Entered");
    //TODO MAT this needs to chnag when updates are added
    return 0;
  }

  @Override
  public boolean getMoreResults()  throws SQLException { //logger.debug("Entered");
    return stmt.getMoreResults();
  }

  @Override
  public void setFetchDirection(int direction)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int getFetchDirection()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setFetchSize(int rows)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int getFetchSize()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int getResultSetConcurrency()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int getResultSetType()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void addBatch(String sql)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void clearBatch()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int[] executeBatch()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public Connection getConnection()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public boolean getMoreResults(int current)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public ResultSet getGeneratedKeys()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int executeUpdate(String sql, int autoGeneratedKeys)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int executeUpdate(String sql, int[] columnIndexes)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int executeUpdate(String sql, String[] columnNames)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public boolean execute(String sql, int autoGeneratedKeys)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public boolean execute(String sql, int[] columnIndexes)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public boolean execute(String sql, String[] columnNames)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public int getResultSetHoldability()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public boolean isClosed()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void setPoolable(boolean poolable)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public boolean isPoolable()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public void closeOnCompletion()  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public boolean isCloseOnCompletion()  throws SQLException { //logger.debug("Entered");    
    logger.info("Entered");
    logger.info("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public <T> T unwrap(Class<T> iface)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

  @Override
  public boolean isWrapperFor(Class<?> iface)  throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet, line:"+new Throwable().getStackTrace()[0].getLineNumber());
  }

}
