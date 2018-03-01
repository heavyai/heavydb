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

import com.mapd.thrift.server.TColumnType;
import com.mapd.thrift.server.TDatumType;
import com.mapd.thrift.server.TQueryResult;
import com.mapd.thrift.server.TRowSet;
import java.io.InputStream;
import java.io.Reader;
import java.math.BigDecimal;
import java.net.URL;
import java.sql.Array;
import java.sql.Blob;
import java.sql.Clob;
import java.sql.Date;
import java.sql.NClob;
import java.sql.Ref;
import java.sql.ResultSetMetaData;
import java.sql.RowId;
import java.sql.SQLException;
import java.sql.SQLWarning;
import java.sql.SQLXML;
import java.sql.Statement;
import java.sql.Time;
import java.sql.Timestamp;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
//Useful debug string
//System.out.println("Entered " + " line:" + new Throwable().getStackTrace()[0].getLineNumber() +
//" class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" +
//new Throwable().getStackTrace()[0].getMethodName());
class MapDResultSet implements java.sql.ResultSet {

  final static Logger logger = LoggerFactory.getLogger(MapDResultSet.class);
  private TQueryResult sqlResult = null;
  private int offset = -1;
  private int numOfRecords = 0;
  private String sql;
  private TRowSet rowSet = null;
  private List<TColumnType> rowDesc;
  private boolean wasNull = false;
  private Map<String, Integer> columnMap;
  private int fetchSize = 0;
  private SQLWarning warnings = null;

  public MapDResultSet(TQueryResult tsqlResult, String sql) throws SQLException { //logger.debug("Entered "+ sql );
    sqlResult = tsqlResult;
    offset = -1;
    this.sql = sql;
    rowSet = sqlResult.getRow_set();
    rowDesc = rowSet.getRow_desc();

    // in the case of a create (maybe insert) nothing is returned in these field
    if (rowDesc.isEmpty()) {
      numOfRecords = 0;
      return;
    }

    rowDesc.get(0).getCol_name();

    columnMap = new HashMap();
    int current = 1;
    for (final TColumnType colType : rowDesc) {
      columnMap.put(colType.getCol_name(), current);
      current++;
    }
    if (rowSet.columns.isEmpty()) {
      numOfRecords = 0;
    } else {
      numOfRecords = rowSet.getColumns().get(0).getNullsSize();
    }

    logger.debug("number of records is " + numOfRecords);
    //logger.debug("Record is "+ sqlResult.toString());

  }

  MapDResultSet() {
    numOfRecords = 0;
  }

  @Override
  public boolean next() throws SQLException { //logger.debug("Entered "+ sql );

    // do real work
    offset++;
    if (offset < numOfRecords) {
      return true;
    }
    return false;
  }

  @Override
  public void close() throws SQLException { //logger.debug("Entered "+ sql );
    // clean up the result object
    this.rowDesc = null;
    this.rowSet = null;
    this.sqlResult = null;
  }

  @Override
  public boolean wasNull() throws SQLException { //logger.debug("Entered "+ sql );
    return wasNull;
  }

  @Override
  public String getString(int columnIndex) throws SQLException {
    //System.out.println("Entered " + " line:" + new Throwable().getStackTrace()[0].getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().getStackTrace()[0].getMethodName());
    //logger.info("Dump result columns "+rowSet.columns.toString());
    //logger.info("Dump column:offset "+ columnIndex + ":" +offset);
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return null;
    } else {
      wasNull = false;
      TDatumType type = sqlResult.row_set.row_desc.get(columnIndex - 1).col_type.type;

      if (type == TDatumType.STR) {
        return rowSet.columns.get(columnIndex - 1).data.str_col.get(offset);
      } else {
        return (String) getStringInternal(columnIndex);
      }
    }
  }

  private String getStringInternal(int columnIndex) throws SQLException {
    TDatumType type = sqlResult.row_set.row_desc.get(columnIndex - 1).col_type.type;

    switch (type) {
      case SMALLINT:
      case INT:
      case BIGINT:
        return String.valueOf(getInt(columnIndex));
      case FLOAT:
        return String.valueOf(getFloat(columnIndex));
      case DECIMAL:
        return String.valueOf(getFloat(columnIndex));
      case DOUBLE:
        return String.valueOf(getDouble(columnIndex));
      case STR:
        return getString(columnIndex);
      case TIME:
        return getTime(columnIndex).toString();
      case TIMESTAMP:
        return getTimestamp(columnIndex).toString();
      case DATE:
        return getDate(columnIndex).toString();
      case BOOL:
        return getBoolean(columnIndex) ? "1" : "0";
      default:
        throw new AssertionError(type.name());
    }
  }

  @Override
  public boolean getBoolean(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return false;
    } else {
      // assume column is str already for now
      wasNull = false;
      if (rowSet.columns.get(columnIndex - 1).data.int_col.get(offset) == 0) {
        return false;
      } else {
        return true;
      }
    }
  }

  @Override
  public byte getByte(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public short getShort(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    //logger.info("Dump result columns "+rowSet.columns.toString());
    //logger.info("Dump column:offset "+ columnIndex + ":" +offset);
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return 0;
    } else {
      // assume column is str already for now
      wasNull = false;
      Long lObj = rowSet.columns.get(columnIndex - 1).data.int_col.get(offset);
      return lObj.shortValue();
    }
  }

  @Override
  public int getInt(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return 0;
    } else {
      // assume column is str already for now
      wasNull = false;
      Long lObj = rowSet.columns.get(columnIndex - 1).data.int_col.get(offset);
      return lObj.intValue();
    }
  }

  @Override
  public long getLong(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return 0;
    } else {
      // assume column is str already for now
      wasNull = false;
      return rowSet.columns.get(columnIndex - 1).data.int_col.get(offset);
    }
  }

  @Override
  public float getFloat(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return 0;
    } else {
      // assume column is str already for now
      wasNull = false;
      return rowSet.columns.get(columnIndex - 1).data.real_col.get(offset).floatValue();
    }
  }

  @Override
  public double getDouble(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return 0;
    } else {
      // assume column is str already for now
      wasNull = false;
      TDatumType type = sqlResult.row_set.row_desc.get(columnIndex - 1).col_type.type;

      if (type == TDatumType.DOUBLE) {
        return rowSet.columns.get(columnIndex - 1).data.real_col.get(offset);
      } else {
        return getDoubleInternal(columnIndex);
      }
    }
  }

  private double getDoubleInternal(int columnIndex) throws SQLException {
    TDatumType type = sqlResult.row_set.row_desc.get(columnIndex - 1).col_type.type;

    switch (type) {
      case SMALLINT:
      case INT:
      case BIGINT:
        return (double) getInt(columnIndex);
      case FLOAT:
        return (double) getFloat(columnIndex);
      case DECIMAL:
        return (double) getFloat(columnIndex);
      case DOUBLE:
        return getDouble(columnIndex);
      case STR:
        return Double.valueOf(getString(columnIndex));
      case TIME:
        return (double) getTime(columnIndex).getTime();
      case TIMESTAMP:
        return (double) getTimestamp(columnIndex).getTime();
      case DATE:
        return (double) getDate(columnIndex).getTime();
      case BOOL:
        return (double) (getBoolean(columnIndex) ? 1 : 0);
      default:
        throw new AssertionError(type.name());
    }
  }

  @Override
  public BigDecimal getBigDecimal(int columnIndex, int scale) throws SQLException { //logger.debug("Entered "+ sql );
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return null;
    } else {
      // assume column is str already for now
      wasNull = false;
      return new BigDecimal(rowSet.columns.get(columnIndex - 1).data.real_col.get(offset));
    }
  }

  @Override
  public byte[] getBytes(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Date getDate(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    //logger.info("Dump result columns "+rowSet.columns.get(columnIndex-1).data.toString());
    //logger.info("Dump offset::" +offset);
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return null;
    } else {
      // assume column is str already for now
      wasNull = false;
      long val = rowSet.columns.get(columnIndex - 1).data.int_col.get(offset);
      Date d = new Date(val * 1000);
      return d;
    }
  }

  @Override
  public Time getTime(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return null;
    } else {
      // assume column is str already for now
      wasNull = false;
      long val = rowSet.columns.get(columnIndex - 1).data.int_col.get(offset);
      return new Time(val * 1000);
    }
  }

  @Override
  public Timestamp getTimestamp(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return null;
    } else {
      // assume column is str already for now
      wasNull = false;
      long val = rowSet.columns.get(columnIndex - 1).data.int_col.get(offset);
      return new Timestamp(val * 1000);
    }
  }

  @Override
  public InputStream getAsciiStream(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public InputStream getUnicodeStream(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public InputStream getBinaryStream(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public String getString(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    Integer colNum = columnMap.get(columnLabel);
    if (colNum == null) {
      throw new SQLException("Could not find column " + columnLabel);
    }
    return getString(colNum);
  }

  @Override
  public boolean getBoolean(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    Integer colNum = columnMap.get(columnLabel);
    if (colNum == null) {
      throw new SQLException("Could not find column " + columnLabel);
    }
    return getBoolean(colNum);
  }

  @Override
  public byte getByte(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public short getShort(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    Integer colNum = columnMap.get(columnLabel);
    if (colNum == null) {
      throw new SQLException("Could not find column " + columnLabel);
    }
    return getShort(colNum);
  }

  @Override
  public int getInt(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    Integer colNum = columnMap.get(columnLabel);
    if (colNum == null) {
      throw new SQLException("Could not find column " + columnLabel);
    }
    return getInt(colNum);
  }

  @Override
  public long getLong(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    Integer colNum = columnMap.get(columnLabel);
    if (colNum == null) {
      throw new SQLException("Could not find column " + columnLabel);
    }
    return getLong(colNum);
  }

  @Override
  public float getFloat(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    Integer colNum = columnMap.get(columnLabel);
    if (colNum == null) {
      throw new SQLException("Could not find column " + columnLabel);
    }
    return getFloat(colNum);
  }

  @Override
  public double getDouble(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    Integer colNum = columnMap.get(columnLabel);
    if (colNum == null) {
      throw new SQLException("Could not find column " + columnLabel);
    }
    return getDouble(colNum);
  }

  @Override
  public BigDecimal getBigDecimal(String columnLabel, int scale) throws SQLException { //logger.debug("Entered "+ sql );
    Integer colNum = columnMap.get(columnLabel);
    if (colNum == null) {
      throw new SQLException("Could not find column " + columnLabel);
    }
    return getBigDecimal(colNum);
  }

  @Override
  public byte[] getBytes(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Date getDate(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    Integer colNum = columnMap.get(columnLabel);
    if (colNum == null) {
      throw new SQLException("Could not find column " + columnLabel);
    }
    return getDate(colNum);
  }

  @Override
  public Time getTime(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    Integer colNum = columnMap.get(columnLabel);
    if (colNum == null) {
      throw new SQLException("Could not find column " + columnLabel);
    }
    return getTime(colNum);
  }

  @Override
  public Timestamp getTimestamp(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    Integer colNum = columnMap.get(columnLabel);
    if (colNum == null) {
      throw new SQLException("Could not find column " + columnLabel);
    }
    return getTimestamp(colNum);
  }

  @Override
  public InputStream getAsciiStream(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public InputStream getUnicodeStream(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public InputStream getBinaryStream(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public SQLWarning getWarnings() throws SQLException { //logger.debug("Entered "+ sql );
    return warnings;
  }

  @Override
  public void clearWarnings() throws SQLException { //logger.debug("Entered "+ sql );
    warnings = null;
  }

  @Override
  public String getCursorName() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSetMetaData getMetaData() throws SQLException { //logger.debug("Entered "+ sql );
    return new MapDResultSetMetaData(sqlResult, sql);
  }

  @Override
  public Object getObject(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    //logger.info("Dump result column "+rowSet.columns.get(columnIndex-1));
    //logger.info("Dump column:offset "+ columnIndex + ":" +offset);
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return null;
    } else {
      wasNull = false;
      // check type
      switch (rowDesc.get(columnIndex - 1).col_type.type) {
        case SMALLINT:
        case INT:
        case BIGINT:
        case BOOL:
        case TIME:
        case TIMESTAMP:
        case DATE:
          return this.rowSet.columns.get(columnIndex - 1).data.int_col.get(offset);
        case FLOAT:
        case DECIMAL:
        case DOUBLE:
          return this.rowSet.columns.get(columnIndex - 1).data.real_col.get(offset);
        case STR:
          return this.rowSet.columns.get(columnIndex - 1).data.str_col.get(offset);
        default:
          throw new AssertionError(rowDesc.get(columnIndex - 1).col_type.type.name());
      }
    }
  }

  @Override
  public Object getObject(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    return getObject(columnMap.get(columnLabel));
  }

  @Override
  public int findColumn(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Reader getCharacterStream(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Reader getCharacterStream(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public BigDecimal getBigDecimal(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    if (rowSet.columns.get(columnIndex - 1).nulls.get(offset)) {
      wasNull = true;
      return null;
    } else {
      // assume column is str already for now
      wasNull = false;
      return new BigDecimal(rowSet.columns.get(columnIndex - 1).data.real_col.get(offset));
    }
  }

  @Override
  public BigDecimal getBigDecimal(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    return getBigDecimal(columnMap.get(columnLabel));
  }

  @Override
  public boolean isBeforeFirst() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isAfterLast() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isFirst() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isLast() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void beforeFirst() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void afterLast() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean first() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean last() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getRow() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean absolute(int row) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean relative(int rows) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean previous() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setFetchDirection(int direction) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getFetchDirection() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setFetchSize(int rows) throws SQLException { //logger.debug("Entered "+ sql );
    fetchSize = rows;
  }

  @Override
  public int getFetchSize() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getType() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getConcurrency() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean rowUpdated() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean rowInserted() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean rowDeleted() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNull(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBoolean(int columnIndex, boolean x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateByte(int columnIndex, byte x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateShort(int columnIndex, short x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateInt(int columnIndex, int x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateLong(int columnIndex, long x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateFloat(int columnIndex, float x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateDouble(int columnIndex, double x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBigDecimal(int columnIndex, BigDecimal x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateString(int columnIndex, String x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBytes(int columnIndex, byte[] x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateDate(int columnIndex, Date x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateTime(int columnIndex, Time x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateTimestamp(int columnIndex, Timestamp x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateAsciiStream(int columnIndex, InputStream x, int length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBinaryStream(int columnIndex, InputStream x, int length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateCharacterStream(int columnIndex, Reader x, int length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateObject(int columnIndex, Object x, int scaleOrLength) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateObject(int columnIndex, Object x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNull(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBoolean(String columnLabel, boolean x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateByte(String columnLabel, byte x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateShort(String columnLabel, short x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateInt(String columnLabel, int x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateLong(String columnLabel, long x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateFloat(String columnLabel, float x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateDouble(String columnLabel, double x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBigDecimal(String columnLabel, BigDecimal x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateString(String columnLabel, String x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBytes(String columnLabel, byte[] x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateDate(String columnLabel, Date x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateTime(String columnLabel, Time x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateTimestamp(String columnLabel, Timestamp x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateAsciiStream(String columnLabel, InputStream x, int length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBinaryStream(String columnLabel, InputStream x, int length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateCharacterStream(String columnLabel, Reader reader, int length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateObject(String columnLabel, Object x, int scaleOrLength) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateObject(String columnLabel, Object x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void insertRow() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateRow() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void deleteRow() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void refreshRow() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void cancelRowUpdates() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void moveToInsertRow() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void moveToCurrentRow() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Statement getStatement() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Object getObject(int columnIndex, Map<String, Class<?>> map) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Ref getRef(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Blob getBlob(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Clob getClob(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Array getArray(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Object getObject(String columnLabel, Map<String, Class<?>> map) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Ref getRef(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Blob getBlob(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Clob getClob(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Array getArray(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Date getDate(int columnIndex, Calendar cal) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Date getDate(String columnLabel, Calendar cal) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Time getTime(int columnIndex, Calendar cal) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Time getTime(String columnLabel, Calendar cal) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Timestamp getTimestamp(int columnIndex, Calendar cal) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Timestamp getTimestamp(String columnLabel, Calendar cal) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public URL getURL(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public URL getURL(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateRef(int columnIndex, Ref x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateRef(String columnLabel, Ref x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBlob(int columnIndex, Blob x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBlob(String columnLabel, Blob x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateClob(int columnIndex, Clob x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateClob(String columnLabel, Clob x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateArray(int columnIndex, Array x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateArray(String columnLabel, Array x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public RowId getRowId(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public RowId getRowId(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateRowId(int columnIndex, RowId x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateRowId(String columnLabel, RowId x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getHoldability() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isClosed() throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNString(int columnIndex, String nString) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNString(String columnLabel, String nString) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNClob(int columnIndex, NClob nClob) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNClob(String columnLabel, NClob nClob) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public NClob getNClob(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public NClob getNClob(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public SQLXML getSQLXML(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public SQLXML getSQLXML(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateSQLXML(int columnIndex, SQLXML xmlObject) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateSQLXML(String columnLabel, SQLXML xmlObject) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public String getNString(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public String getNString(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Reader getNCharacterStream(int columnIndex) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Reader getNCharacterStream(String columnLabel) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNCharacterStream(int columnIndex, Reader x, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNCharacterStream(String columnLabel, Reader reader, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateAsciiStream(int columnIndex, InputStream x, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBinaryStream(int columnIndex, InputStream x, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateCharacterStream(int columnIndex, Reader x, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateAsciiStream(String columnLabel, InputStream x, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBinaryStream(String columnLabel, InputStream x, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateCharacterStream(String columnLabel, Reader reader, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBlob(int columnIndex, InputStream inputStream, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBlob(String columnLabel, InputStream inputStream, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateClob(int columnIndex, Reader reader, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateClob(String columnLabel, Reader reader, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNClob(int columnIndex, Reader reader, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNClob(String columnLabel, Reader reader, long length) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNCharacterStream(int columnIndex, Reader x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNCharacterStream(String columnLabel, Reader reader) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateAsciiStream(int columnIndex, InputStream x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBinaryStream(int columnIndex, InputStream x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateCharacterStream(int columnIndex, Reader x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateAsciiStream(String columnLabel, InputStream x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBinaryStream(String columnLabel, InputStream x) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateCharacterStream(String columnLabel, Reader reader) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBlob(int columnIndex, InputStream inputStream) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateBlob(String columnLabel, InputStream inputStream) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateClob(int columnIndex, Reader reader) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateClob(String columnLabel, Reader reader) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNClob(int columnIndex, Reader reader) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void updateNClob(String columnLabel, Reader reader) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public <T> T getObject(int columnIndex, Class<T> type) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public <T> T getObject(String columnLabel, Class<T> type) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public <T> T unwrap(Class<T> iface) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isWrapperFor(Class<?> iface) throws SQLException { //logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

}
