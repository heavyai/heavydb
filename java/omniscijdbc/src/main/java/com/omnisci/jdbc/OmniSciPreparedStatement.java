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
class OmniSciPreparedStatement implements PreparedStatement {
  final static Logger MAPDLOGGER =
          LoggerFactory.getLogger(OmniSciPreparedStatement.class);
  public SQLWarning rootWarning = null;

  private String currentSQL;
  private String insertTableName;
  private int parmCount = 0;
  private String brokenSQL[];
  private String parmRep[];
  private boolean parmIsNull[];
  private int repCount;
  private String session;
  private MapD.Client client;
  private OmniSciStatement stmt = null;
  private boolean isInsert = false;
  private boolean isNewBatch = true;
  private boolean[] parmIsString = null;
  private List<TStringRow> rows = null;
  private static final Pattern REGEX_PATTERN = Pattern.compile("(?i)\\s+INTO\\s+(\\w+)");

  OmniSciPreparedStatement(String sql, String session, MapD.Client client) {
    MAPDLOGGER.debug("Entered");
    currentSQL = sql;
    this.client = client;
    this.session = session;
    this.stmt = new OmniSciStatement(session, client);
    MAPDLOGGER.debug("Prepared statement is " + currentSQL);
    // TODO in real life this needs to check if the ? is inside quotes before we assume it
    // a parameter
    brokenSQL = currentSQL.split("\\?", -1);
    parmCount = brokenSQL.length - 1;
    parmRep = new String[parmCount];
    parmIsNull = new boolean[parmCount];
    parmIsString = new boolean[parmCount];
    repCount = 0;
    if (currentSQL.toUpperCase().contains("INSERT ")) {
      // remove double quotes required for queries generated with " around all names like
      // kafka connect
      currentSQL = currentSQL.replaceAll("\"", " ");
      MAPDLOGGER.debug("Insert Prepared statement is " + currentSQL);
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
    // put string together if required
    if (parmCount > 0) {
      if (repCount != parmCount) {
        throw new UnsupportedOperationException(
                "Incorrect number of replace parameters for prepared statement "
                + currentSQL + " has only " + repCount + " parameters");
      }
      StringBuilder modQuery = new StringBuilder(currentSQL.length() * 5);
      for (int i = 0; i < repCount; i++) {
        modQuery.append(brokenSQL[i]);
        if (parmIsNull[i]) {
          modQuery.append("NULL");
        } else {
          if (parmIsString[i]) {
            modQuery.append("'").append(parmRep[i]).append("'");
          } else {
            modQuery.append(parmRep[i]);
          }
        }
      }
      modQuery.append(brokenSQL[parmCount]);
      qsql = modQuery.toString();
    } else {
      qsql = currentSQL;
    }

    qsql = qsql.replace(" WHERE 1=0", " LIMIT 1 ");
    MAPDLOGGER.debug("Query is now " + qsql);
    repCount = 0; // reset the parameters
    return qsql;
  }

  @Override
  public ResultSet executeQuery() throws SQLException {
    if (isNewBatch) {
      String qsql = getQuery();
      MAPDLOGGER.debug("executeQuery, sql=" + qsql);
      return stmt.executeQuery(qsql);
    }
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate() throws SQLException {
    MAPDLOGGER.debug("Entered");
    executeQuery();
    return 1;
  }

  @Override
  public void setNull(int parameterIndex, int sqlType) throws SQLException {
    MAPDLOGGER.debug("Entered");
    parmIsNull[parameterIndex - 1] = true;
    repCount++;
  }

  @Override
  public void setBoolean(int parameterIndex, boolean x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = x ? "true" : "false";
    parmIsString[parameterIndex - 1] = false;
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setByte(int parameterIndex, byte x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setShort(int parameterIndex, short x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = Short.toString(x);
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setInt(int parameterIndex, int x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = Integer.toString(x);
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setLong(int parameterIndex, long x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = Long.toString(x);
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setFloat(int parameterIndex, float x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = Float.toString(x);
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setDouble(int parameterIndex, double x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = Double.toString(x);
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setBigDecimal(int parameterIndex, BigDecimal x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = x.toString();
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setString(int parameterIndex, String x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    // add extra ' if there are any in string
    x = x.replaceAll("'", "''");
    parmRep[parameterIndex - 1] = x;
    parmIsString[parameterIndex - 1] = true;
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setBytes(int parameterIndex, byte[] x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setDate(int parameterIndex, Date x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = x.toString();
    parmIsString[parameterIndex - 1] = true;
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setTime(int parameterIndex, Time x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = x.toString();
    parmIsString[parameterIndex - 1] = true;
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setTimestamp(int parameterIndex, Timestamp x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] =
            x.toString(); // new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(x);
    parmIsString[parameterIndex - 1] = true;
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x, int length)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setUnicodeStream(int parameterIndex, InputStream x, int length)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x, int length)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void clearParameters() throws SQLException {
    MAPDLOGGER.debug("Entered");
    // TODO MAT we will actually need to do something here one day
  }

  @Override
  public void setObject(int parameterIndex, Object x, int targetSqlType)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setObject(int parameterIndex, Object x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute() throws SQLException {
    MAPDLOGGER.debug("Entered");
    String tQuery = getQuery();
    return stmt.execute(tQuery);
  }

  @Override
  public void addBatch() throws SQLException {
    MAPDLOGGER.debug("Entered");
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
        if (parmIsNull[i]) {
          tsv.is_null = true;
        } else {
          tsv.is_null = false;
        }
        tsr.addToCols(tsv);
      }
      rows.add(tsr);
      MAPDLOGGER.debug("addBatch, rows=" + rows.size());
    } else {
      throw new UnsupportedOperationException("addBatch only supported for insert, line:"
              + new Throwable().getStackTrace()[0].getLineNumber());
    }
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader, int length)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setRef(int parameterIndex, Ref x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBlob(int parameterIndex, Blob x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setClob(int parameterIndex, Clob x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setArray(int parameterIndex, Array x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSetMetaData getMetaData() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setDate(int parameterIndex, Date x, Calendar cal) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setTime(int parameterIndex, Time x, Calendar cal) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setTimestamp(int parameterIndex, Timestamp x, Calendar cal)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNull(int parameterIndex, int sqlType, String typeName)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setURL(int parameterIndex, URL x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public ParameterMetaData getParameterMetaData() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setRowId(int parameterIndex, RowId x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNString(int parameterIndex, String value) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNCharacterStream(int parameterIndex, Reader value, long length)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNClob(int parameterIndex, NClob value) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setClob(int parameterIndex, Reader reader, long length)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBlob(int parameterIndex, InputStream inputStream, long length)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNClob(int parameterIndex, Reader reader, long length)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setSQLXML(int parameterIndex, SQLXML xmlObject) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setObject(
          int parameterIndex, Object x, int targetSqlType, int scaleOrLength)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x, long length)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x, long length)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader, long length)
          throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNCharacterStream(int parameterIndex, Reader value) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setClob(int parameterIndex, Reader reader) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBlob(int parameterIndex, InputStream inputStream) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNClob(int parameterIndex, Reader reader) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSet executeQuery(String sql) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void close() throws SQLException {
    MAPDLOGGER.debug("close");
    if (stmt != null) {
      // TODO MAT probably more needed here
      stmt.close();
      stmt = null;
    }
  }

  @Override
  public int getMaxFieldSize() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setMaxFieldSize(int max) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getMaxRows() throws SQLException {
    MAPDLOGGER.debug("Entered");
    return stmt.getMaxRows();
  }

  @Override
  public void setMaxRows(int max) throws SQLException {
    MAPDLOGGER.debug("Entered");
    MAPDLOGGER.debug("SetMaxRows to " + max);
    stmt.setMaxRows(max);
  }

  @Override
  public void setEscapeProcessing(boolean enable) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getQueryTimeout() throws SQLException {
    MAPDLOGGER.debug("Entered");
    return 0;
  }

  @Override
  public void setQueryTimeout(int seconds) throws SQLException {
    MAPDLOGGER.debug("Entered");
    SQLWarning warning = new SQLWarning(
            "Query timeouts are not supported.  Substituting a value of zero.");
    if (rootWarning == null)
      rootWarning = warning;
    else
      rootWarning.setNextWarning(warning);
  }

  @Override
  public void cancel() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public SQLWarning getWarnings() throws SQLException {
    MAPDLOGGER.debug("Entered");
    return rootWarning;
  }

  @Override
  public void clearWarnings() throws SQLException {
    MAPDLOGGER.debug("Entered");
    rootWarning = null;
  }

  @Override
  public void setCursorName(String name) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSet getResultSet() throws SQLException {
    MAPDLOGGER.debug("Entered");
    return stmt.getResultSet();
  }

  @Override
  public int getUpdateCount() throws SQLException {
    MAPDLOGGER.debug("Entered");
    // TODO MAT this needs to change when updates are added
    return 0;
  }

  @Override
  public boolean getMoreResults() throws SQLException {
    MAPDLOGGER.debug("Entered");
    return stmt.getMoreResults();
  }

  @Override
  public void setFetchDirection(int direction) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getFetchDirection() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setFetchSize(int rows) throws SQLException {
    MAPDLOGGER.debug("Entered");
    // TODO we need to chnage the model to allow smaller select chunks at the moment you
    // get everything
  }

  @Override
  public int getFetchSize() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetConcurrency() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetType() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void addBatch(String sql) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void clearBatch() throws SQLException {
    MAPDLOGGER.debug("Entered");
    if (rows != null) {
      rows.clear();
    }
  }

  @Override
  public int[] executeBatch() throws SQLException {
    int ret[] = null;
    if (rows != null) {
      MAPDLOGGER.debug("executeBatch, rows=" + rows.size());
      try {
        // send the batch
        client.load_table(session, insertTableName, rows);
      } catch (TMapDException ex) {
        throw new SQLException("executeBatch failed: " + ex.getError_msg());
      } catch (TException ex) {
        throw new SQLException("executeBatch failed: " + ex.toString());
      }
      ret = new int[rows.size()];
      for (int i = 0; i < rows.size(); i++) {
        ret[i] = 1;
      }
      clearBatch();
    }
    return ret;
  }

  @Override
  public Connection getConnection() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean getMoreResults(int current) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSet getGeneratedKeys() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, int autoGeneratedKeys) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, int[] columnIndexes) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, String[] columnNames) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, int autoGeneratedKeys) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, int[] columnIndexes) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, String[] columnNames) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetHoldability() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isClosed() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setPoolable(boolean poolable) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isPoolable() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void closeOnCompletion() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isCloseOnCompletion() throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public <T> T unwrap(Class<T> iface) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isWrapperFor(Class<?> iface) throws SQLException {
    MAPDLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }
}
