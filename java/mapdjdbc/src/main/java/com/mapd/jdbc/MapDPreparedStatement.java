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

package com.mapd.jdbc;

import com.mapd.thrift.server.MapD;
import com.mapd.thrift.server.TStringRow;
import com.mapd.thrift.server.TStringValue;
import com.mapd.thrift.server.TMapDException;
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
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.thrift.TException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
class MapDPreparedStatement implements PreparedStatement {

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDPreparedStatement.class);

  private String currentSQL;
  private String insertTableName;
  private int parmCount = 0;
  private String brokenSQL[];
  private String parmRep[];
  private int repCount;
  private String session;
  private MapD.Client client;
  private MapDStatement stmt = null;
  private StringBuffer modQuery;
  private boolean isInsert = false;
  private boolean isNewBatch = true;
  private boolean isParmString[] = null;
  private List<TStringRow> rows = null;
  private String warnings = null;
  private static final Pattern REGEX_PATTERN = Pattern.compile("(?i) INTO (\\w+)");

  MapDPreparedStatement(String sql, String session, MapD.Client client) {
    currentSQL = sql;
    this.client = client;
    this.session = session;
    this.stmt = new MapDStatement(session, client);
    MAPDLOGGER.debug("Prepared statement is " + currentSQL);
    //TODO in real life this needs to check if the ? isinside quotes before we assume it a parameter
    brokenSQL = currentSQL.split("\\?");
    parmCount = brokenSQL.length - 1;
    parmRep = new String[parmCount];
    isParmString = new boolean[parmCount];
    repCount = 0;
    modQuery = new StringBuffer(currentSQL.length() * 5);
    if (currentSQL.toUpperCase().contains("INSERT ")) {
      isInsert = true;
      Matcher matcher = REGEX_PATTERN.matcher(currentSQL);
      while (matcher.find()) {
        insertTableName = matcher.group(1);
        MAPDLOGGER.debug("Table name for insert is '" + insertTableName + "'");
      }
    }
  }

  private String getQuery() {
    String qsql;
    //put string together if required
    if (parmCount > 0) {
      if (repCount != parmCount) {
        throw new UnsupportedOperationException("Incorrect number of replace parameters for prepared statement "
                + currentSQL + " has only " + repCount + " parameters");
      }
      for (int i = 0; i < repCount; i++) {
        modQuery.append(brokenSQL[i]);
        if (isParmString[i]) {
          modQuery.append("'").append(parmRep[i]).append("'");
        } else {
          modQuery.append(parmRep[i]);
        }
      }
      modQuery.append(brokenSQL[parmCount]);
      qsql = modQuery.toString();
    } else {
      qsql = currentSQL;
    }

    qsql = qsql.replace(" WHERE 1=0", " LIMIT 1 ");
    MAPDLOGGER.debug("Query is now " + qsql);
    return qsql;
  }

  @Override
  public ResultSet executeQuery() throws SQLException { //logger.debug("Entered");
    if (isNewBatch) {
      String qsql = getQuery();
      return stmt.executeQuery(qsql);
    }
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNull(int parameterIndex, int sqlType) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBoolean(int parameterIndex, boolean x) throws SQLException { //logger.debug("Entered");
    parmRep[parameterIndex - 1] = x ? "t" : "f";
    isParmString[parameterIndex - 1] = true;
    repCount++;
  }

  @Override
  public void setByte(int parameterIndex, byte x) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setShort(int parameterIndex, short x) throws SQLException { //logger.debug("Entered");
    parmRep[parameterIndex - 1] = Short.toString(x);
    repCount++;
  }

  @Override
  public void setInt(int parameterIndex, int x) throws SQLException { //logger.debug("Entered");
    parmRep[parameterIndex - 1] = Integer.toString(x);
    repCount++;
  }

  @Override
  public void setLong(int parameterIndex, long x) throws SQLException { //logger.debug("Entered");
    parmRep[parameterIndex - 1] = Long.toString(x);
    repCount++;
  }

  @Override
  public void setFloat(int parameterIndex, float x) throws SQLException { //logger.debug("Entered");
    parmRep[parameterIndex - 1] = Float.toString(x);
    repCount++;
  }

  @Override
  public void setDouble(int parameterIndex, double x) throws SQLException { //logger.debug("Entered");
    parmRep[parameterIndex - 1] = Double.toString(x);
    repCount++;
  }

  @Override
  public void setBigDecimal(int parameterIndex, BigDecimal x) throws SQLException { //logger.debug("Entered");
    parmRep[parameterIndex - 1] = x.toString();
    repCount++;
  }

  @Override
  public void setString(int parameterIndex, String x) throws SQLException { //logger.debug("Entered");
    parmRep[parameterIndex - 1] = x;
    isParmString[parameterIndex - 1] = true;
    repCount++;
  }

  @Override
  public void setBytes(int parameterIndex, byte[] x) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setDate(int parameterIndex, Date x) throws SQLException { //logger.debug("Entered");
    parmRep[parameterIndex - 1] = x.toString();
    isParmString[parameterIndex - 1] = true;
    repCount++;
  }

  @Override
  public void setTime(int parameterIndex, Time x) throws SQLException { //logger.debug("Entered");
    parmRep[parameterIndex - 1] = x.toString();
    isParmString[parameterIndex - 1] = true;
    repCount++;
  }

  @Override
  public void setTimestamp(int parameterIndex, Timestamp x) throws SQLException { //logger.debug("Entered");
    parmRep[parameterIndex - 1] = x.toString(); //new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(x);
    isParmString[parameterIndex - 1] = true;
    repCount++;
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x, int length) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setUnicodeStream(int parameterIndex, InputStream x, int length) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x, int length) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void clearParameters() throws SQLException { //logger.debug("Entered");
    //TODO MAT we will actually need to do something here one day
  }

  @Override
  public void setObject(int parameterIndex, Object x, int targetSqlType) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setObject(int parameterIndex, Object x) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute() throws SQLException { //logger.debug("Entered");
    String tQuery = getQuery();
    return stmt.execute(tQuery);
  }

  @Override
  public void addBatch() throws SQLException { //logger.debug("Entered");
    if (isInsert) {
      // take the values and use stream inserter to add them
      if (isNewBatch) {
        rows = new ArrayList(5000);
        isNewBatch = false;
      }
      // add data to stream

      TStringRow tsr = new TStringRow();
      for (int i = 0; i < parmCount; i++) {
        // place string in rows array
        TStringValue tsv = new TStringValue();
        tsv.str_val = this.parmRep[i];
        if (parmRep[i].length() == 0) {
          tsv.is_null = true;
        } else {
          tsv.is_null = false;
        }
        tsr.addToCols(tsv);
      }
      rows.add(tsr);
    } else {
      throw new UnsupportedOperationException("addBatch only supported for insert, line:" + new Throwable().
              getStackTrace()[0].getLineNumber());
    }
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader, int length) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setRef(int parameterIndex, Ref x) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBlob(int parameterIndex, Blob x) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setClob(int parameterIndex, Clob x) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setArray(int parameterIndex, Array x) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSetMetaData getMetaData() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setDate(int parameterIndex, Date x, Calendar cal) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setTime(int parameterIndex, Time x, Calendar cal) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setTimestamp(int parameterIndex, Timestamp x, Calendar cal) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNull(int parameterIndex, int sqlType, String typeName) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setURL(int parameterIndex, URL x) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public ParameterMetaData getParameterMetaData() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setRowId(int parameterIndex, RowId x) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNString(int parameterIndex, String value) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNCharacterStream(int parameterIndex, Reader value, long length) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNClob(int parameterIndex, NClob value) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setClob(int parameterIndex, Reader reader, long length) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBlob(int parameterIndex, InputStream inputStream, long length) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNClob(int parameterIndex, Reader reader, long length) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setSQLXML(int parameterIndex, SQLXML xmlObject) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setObject(int parameterIndex, Object x, int targetSqlType, int scaleOrLength) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x, long length) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x, long length) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader, long length) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNCharacterStream(int parameterIndex, Reader value) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setClob(int parameterIndex, Reader reader) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBlob(int parameterIndex, InputStream inputStream) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNClob(int parameterIndex, Reader reader) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSet executeQuery(String sql) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void close() throws SQLException { //logger.debug("Entered");
    if (stmt != null) {
      //TODO MAT probably more needed here
      stmt.close();
      stmt = null;
    }
  }

  @Override
  public int getMaxFieldSize() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setMaxFieldSize(int max) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getMaxRows() throws SQLException { //logger.debug("Entered");
    return stmt.getMaxRows();
  }

  @Override
  public void setMaxRows(int max) throws SQLException { //logger.debug("Entered");
    MAPDLOGGER.debug("SetMaxRows to " + max);
    stmt.setMaxRows(max);
  }

  @Override
  public void setEscapeProcessing(boolean enable) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getQueryTimeout() throws SQLException { //logger.debug("Entered");
    return stmt.getQueryTimeout();
  }

  @Override
  public void setQueryTimeout(int seconds) throws SQLException { //logger.debug("Entered");
    MAPDLOGGER.debug("SetQueryTimeout to " + seconds);
    stmt.setQueryTimeout(seconds);
  }

  @Override
  public void cancel() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public SQLWarning getWarnings() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void clearWarnings() throws SQLException { //logger.debug("Entered");
    warnings = null;
  }

  @Override
  public void setCursorName(String name) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName()
    );
  }

  @Override
  public ResultSet getResultSet() throws SQLException { //logger.debug("Entered");
    return stmt.getResultSet();
  }

  @Override
  public int getUpdateCount() throws SQLException { //logger.debug("Entered");
    //TODO MAT this needs to change when updates are added
    return 0;
  }

  @Override
  public boolean getMoreResults() throws SQLException { //logger.debug("Entered");
    return stmt.getMoreResults();
  }

  @Override
  public void setFetchDirection(int direction) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getFetchDirection() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setFetchSize(int rows) throws SQLException { //logger.debug("Entered");
    //TODO we need to chnage the model to allow smaller select chunks at the moment you get everything
  }

  @Override
  public int getFetchSize() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetConcurrency() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetType() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void addBatch(String sql) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void clearBatch() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int[] executeBatch() throws SQLException { //logger.debug("Entered");
    int ret[] = null;
    if (rows != null) {
      try {
        // send the batch
        client.load_table(session, insertTableName, rows);
      } catch (TMapDException ex) {
        throw new SQLException("addBatch failed : " + ex.getError_msg());
      } catch (TException ex) {
        throw new SQLException("addBatch failed : " + ex.toString());
      }
      ret = new int[rows.size()];
      rows.clear();
    }
    return ret;
  }

  @Override
  public Connection getConnection() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean getMoreResults(int current) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSet getGeneratedKeys() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, int autoGeneratedKeys) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, int[] columnIndexes) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, String[] columnNames) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, int autoGeneratedKeys) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, int[] columnIndexes) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, String[] columnNames) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetHoldability() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isClosed() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setPoolable(boolean poolable) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isPoolable() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void closeOnCompletion() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isCloseOnCompletion() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public <T> T unwrap(Class<T> iface) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isWrapperFor(Class<?> iface) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }
}
