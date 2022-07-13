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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
import java.sql.Types;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ai.heavy.thrift.server.Heavy;
import ai.heavy.thrift.server.TColumnType;
import ai.heavy.thrift.server.TDBException;
import ai.heavy.thrift.server.TStringRow;
import ai.heavy.thrift.server.TStringValue;
import ai.heavy.thrift.server.TTableDetails;

class HeavyAIPreparedStatement implements PreparedStatement {
  final static Logger HEAVYDBLOGGER =
          LoggerFactory.getLogger(HeavyAIPreparedStatement.class);
  public SQLWarning rootWarning = null;

  private String currentSQL;
  private String insertTableName;
  private int parmCount = 0;
  private String brokenSQL[];
  private String parmRep[];
  private boolean parmIsNull[];
  private String listOfFields[];
  private int repCount;
  private String session;
  private Heavy.Client client;
  private HeavyAIStatement stmt = null;
  private boolean isInsert = false;
  private boolean isNewBatch = true;
  private boolean[] parmIsString = null;
  private List<TStringRow> rows = null;
  private static final Pattern REGEX_PATTERN = Pattern.compile("(?i)\\s+INTO\\s+(\\w+)");
  private static final Pattern REGEX_LOF_PATTERN = Pattern.compile(
          "(?i)\\s*insert\\s+into\\s+[\\w:\\.]+\\s*\\(([\\w:\\s:\\,:\\']+)\\)[\\w:\\s]+\\(");
  // this regex ignores all multi- and single-line comments and whitespaces at the
  // beginning of a query and checks if the first meaningful word is SELECT
  private static final Pattern REGEX_IS_SELECT_PATTERN =
          Pattern.compile("^(?:\\s|--.*?\\R|/\\*[\\S\\s]*?\\*/|\\s*)*\\s*select[\\S\\s]*",
                  Pattern.CASE_INSENSITIVE | Pattern.MULTILINE);
  private boolean isClosed = false;

  HeavyAIPreparedStatement(String sql, String session, HeavyAIConnection connection) {
    HEAVYDBLOGGER.debug("Entered");
    currentSQL = sql;
    this.client = connection.client;
    this.session = session;
    this.stmt = new HeavyAIStatement(session, connection);
    HEAVYDBLOGGER.debug("Prepared statement is " + currentSQL);
    // TODO in real life this needs to check if the ? is inside quotes before we
    // assume it
    // a parameter
    brokenSQL = currentSQL.split("\\?", -1);
    parmCount = brokenSQL.length - 1;
    parmRep = new String[parmCount];
    parmIsNull = new boolean[parmCount];
    parmIsString = new boolean[parmCount];
    repCount = 0;
    if (currentSQL.toUpperCase().contains("INSERT ")) {
      // remove double quotes required for queries generated with " around all names
      // like
      // kafka connect
      currentSQL = currentSQL.replaceAll("\"", " ");
      HEAVYDBLOGGER.debug("Insert Prepared statement is " + currentSQL);
      isInsert = true;
      Matcher matcher = REGEX_PATTERN.matcher(currentSQL);
      while (matcher.find()) {
        insertTableName = matcher.group(1);
        HEAVYDBLOGGER.debug("Table name for insert is '" + insertTableName + "'");
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
    HEAVYDBLOGGER.debug("Query is now " + qsql);
    repCount = 0; // reset the parameters
    return qsql;
  }

  private boolean isSelect() {
    Matcher matcher = REGEX_IS_SELECT_PATTERN.matcher(currentSQL);
    return matcher.matches();
  }

  @Override
  public ResultSet executeQuery() throws SQLException {
    if (isNewBatch) {
      String qsql = getQuery();
      HEAVYDBLOGGER.debug("executeQuery, sql=" + qsql);
      return stmt.executeQuery(qsql);
    }
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    executeQuery();
    // TODO: OmniSciDB supports updates, inserts and deletes, but
    // there is no way to get number of affected rows at the moment
    return -1;
  }

  @Override
  public void setNull(int parameterIndex, int sqlType) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmIsNull[parameterIndex - 1] = true;
    repCount++;
  }

  @Override
  public void setBoolean(int parameterIndex, boolean x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = x ? "true" : "false";
    parmIsString[parameterIndex - 1] = false;
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setByte(int parameterIndex, byte x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setShort(int parameterIndex, short x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = Short.toString(x);
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setInt(int parameterIndex, int x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = Integer.toString(x);
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setLong(int parameterIndex, long x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = Long.toString(x);
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setFloat(int parameterIndex, float x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = Float.toString(x);
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setDouble(int parameterIndex, double x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = Double.toString(x);
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setBigDecimal(int parameterIndex, BigDecimal x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = x.toString();
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setString(int parameterIndex, String x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    // add extra ' if there are any in string
    x = x.replaceAll("'", "''");
    parmRep[parameterIndex - 1] = x;
    parmIsString[parameterIndex - 1] = true;
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setBytes(int parameterIndex, byte[] x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setDate(int parameterIndex, Date x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = x.toString();
    parmIsString[parameterIndex - 1] = true;
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setTime(int parameterIndex, Time x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = x.toString();
    parmIsString[parameterIndex - 1] = true;
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setTimestamp(int parameterIndex, Timestamp x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] =
            x.toString(); // new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(x);
    parmIsString[parameterIndex - 1] = true;
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x, int length)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setUnicodeStream(int parameterIndex, InputStream x, int length)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x, int length)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void clearParameters() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    // TODO MAT we will actually need to do something here one day
  }

  @Override
  public void setObject(int parameterIndex, Object x, int targetSqlType)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setObject(int parameterIndex, Object x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    String tQuery = getQuery();
    return stmt.execute(tQuery);
  }

  @Override
  public void addBatch() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    if (isInsert) {
      // take the values and use stream inserter to add them
      if (isNewBatch) {
        // check for columns names
        Matcher matcher = REGEX_LOF_PATTERN.matcher(currentSQL);
        if (matcher.find()) {
          listOfFields = matcher.group(1).trim().split("\\s*,+\\s*,*\\s*");
          if (listOfFields.length != parmCount) {
            throw new SQLException("Too many or too few values");
          } else if (Arrays.stream(listOfFields).distinct().toArray().length
                  != listOfFields.length) {
            throw new SQLException("Duplicated column name");
          }
          List<String> listOfColumns = new ArrayList<String>();
          try {
            TTableDetails tableDetails =
                    client.get_table_details(session, insertTableName);
            for (TColumnType column : tableDetails.row_desc) {
              listOfColumns.add(column.col_name.toLowerCase());
            }
          } catch (TException ex) {
            throw new SQLException(ex.toString());
          }
          for (String paramName : listOfFields) {
            if (listOfColumns.indexOf(paramName) == -1) {
              throw new SQLException(
                      "Column " + paramName.toLowerCase() + " does not exist");
            }
          }
        } else {
          listOfFields = new String[0];
        }

        rows = new ArrayList(5000);
        isNewBatch = false;
      }
      // add data to stream
      TStringRow tsr = new TStringRow();
      for (int i = 0; i < parmCount; i++) {
        // place string in rows array
        TStringValue tsv = new TStringValue();
        if (parmIsNull[i]) {
          tsv.is_null = true;
        } else {
          tsv.str_val = this.parmRep[i];
          tsv.is_null = false;
        }
        tsr.addToCols(tsv);
      }
      rows.add(tsr);
      HEAVYDBLOGGER.debug("addBatch, rows=" + rows.size());
    } else {
      throw new UnsupportedOperationException("addBatch only supported for insert, line:"
              + new Throwable().getStackTrace()[0].getLineNumber());
    }
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader, int length)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setRef(int parameterIndex, Ref x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBlob(int parameterIndex, Blob x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setClob(int parameterIndex, Clob x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setArray(int parameterIndex, Array x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    parmRep[parameterIndex - 1] = x.toString();
    parmIsNull[parameterIndex - 1] = false;
    repCount++;
  }

  @Override
  public ResultSetMetaData getMetaData() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    if (!isSelect()) {
      return null;
    }
    if (stmt.getResultSet() != null) {
      return stmt.getResultSet().getMetaData();
    }
    PreparedStatement ps = null;
    try {
      ps = new HeavyAIPreparedStatement(
              currentSQL, session, (HeavyAIConnection) getConnection());
      ps.setMaxRows(0);
      for (int i = 1; i <= this.parmCount; ++i) {
        ps.setNull(i, Types.NULL);
      }
      ResultSet rs = ps.executeQuery();
      if (rs != null) {
        return rs.getMetaData();
      } else {
        return null;
      }
    } finally {
      if (ps != null) {
        ps.close();
      }
    }
  }

  @Override
  public void setDate(int parameterIndex, Date x, Calendar cal) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setTime(int parameterIndex, Time x, Calendar cal) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setTimestamp(int parameterIndex, Timestamp x, Calendar cal)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNull(int parameterIndex, int sqlType, String typeName)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setURL(int parameterIndex, URL x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public ParameterMetaData getParameterMetaData() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setRowId(int parameterIndex, RowId x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNString(int parameterIndex, String value) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNCharacterStream(int parameterIndex, Reader value, long length)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNClob(int parameterIndex, NClob value) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setClob(int parameterIndex, Reader reader, long length)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBlob(int parameterIndex, InputStream inputStream, long length)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNClob(int parameterIndex, Reader reader, long length)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setSQLXML(int parameterIndex, SQLXML xmlObject) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setObject(
          int parameterIndex, Object x, int targetSqlType, int scaleOrLength)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x, long length)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x, long length)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader, long length)
          throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setAsciiStream(int parameterIndex, InputStream x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBinaryStream(int parameterIndex, InputStream x) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setCharacterStream(int parameterIndex, Reader reader) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNCharacterStream(int parameterIndex, Reader value) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setClob(int parameterIndex, Reader reader) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setBlob(int parameterIndex, InputStream inputStream) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNClob(int parameterIndex, Reader reader) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSet executeQuery(String sql) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void close() throws SQLException {
    HEAVYDBLOGGER.debug("close");
    if (stmt != null) {
      // TODO MAT probably more needed here
      stmt.close();
      stmt = null;
    }
    isClosed = true;
  }

  @Override
  public int getMaxFieldSize() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setMaxFieldSize(int max) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getMaxRows() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    return stmt.getMaxRows();
  }

  @Override
  public void setMaxRows(int max) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    HEAVYDBLOGGER.debug("SetMaxRows to " + max);
    stmt.setMaxRows(max);
  }

  @Override
  public void setEscapeProcessing(boolean enable) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getQueryTimeout() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    return 0;
  }

  @Override
  public void setQueryTimeout(int seconds) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    SQLWarning warning = new SQLWarning(
            "Query timeouts are not supported.  Substituting a value of zero.");
    if (rootWarning == null)
      rootWarning = warning;
    else
      rootWarning.setNextWarning(warning);
  }

  @Override
  public void cancel() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public SQLWarning getWarnings() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    return rootWarning;
  }

  @Override
  public void clearWarnings() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    rootWarning = null;
  }

  @Override
  public void setCursorName(String name) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSet getResultSet() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    return stmt.getResultSet();
  }

  @Override
  public int getUpdateCount() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    return stmt.getUpdateCount();
  }

  @Override
  public boolean getMoreResults() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    return stmt.getMoreResults();
  }

  @Override
  public void setFetchDirection(int direction) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getFetchDirection() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    return ResultSet.FETCH_FORWARD;
  }

  @Override
  public void setFetchSize(int rows) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    // TODO we need to chnage the model to allow smaller select chunks at the moment
    // you
    // get everything
  }

  @Override
  public int getFetchSize() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetConcurrency() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetType() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void addBatch(String sql) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void clearBatch() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    if (rows != null) {
      rows.clear();
    }
  }

  @Override
  public int[] executeBatch() throws SQLException {
    checkClosed();
    int ret[] = null;
    if (rows != null) {
      HEAVYDBLOGGER.debug("executeBatch, rows=" + rows.size());
      try {
        // send the batch
        client.load_table(session, insertTableName, rows, Arrays.asList(listOfFields));
      } catch (TDBException ex) {
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
    HEAVYDBLOGGER.debug("Entered");
    return stmt.getConnection();
  }

  @Override
  public boolean getMoreResults(int current) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSet getGeneratedKeys() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, int autoGeneratedKeys) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, int[] columnIndexes) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int executeUpdate(String sql, String[] columnNames) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, int autoGeneratedKeys) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, int[] columnIndexes) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean execute(String sql, String[] columnNames) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getResultSetHoldability() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isClosed() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    return isClosed;
  }

  @Override
  public void setPoolable(boolean poolable) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isPoolable() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void closeOnCompletion() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isCloseOnCompletion() throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public <T> T unwrap(Class<T> iface) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isWrapperFor(Class<?> iface) throws SQLException {
    HEAVYDBLOGGER.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  private void checkClosed() throws SQLException {
    if (isClosed) {
      throw new SQLException("PreparedStatement is closed.");
    }
  }
}
