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

import com.mapd.thrift.server.TDatumType;
import com.mapd.thrift.server.TQueryResult;
import java.sql.DatabaseMetaData;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
class MapDResultSetMetaData implements ResultSetMetaData {

  final static Logger logger = LoggerFactory.getLogger(MapDResultSetMetaData.class);
  final TQueryResult sqlResult;
  final String sql;

  public MapDResultSetMetaData(TQueryResult sqlResult, String sql) {
    this.sqlResult = sqlResult;
    this.sql = sql;
  }

  @Override
  public int getColumnCount() throws SQLException { //logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.size();
  }

  @Override
  public boolean isAutoIncrement(int column) throws SQLException { //logger.debug("Entered "+ sql );
    //logger.debug("returning false");
    return false;
  }

  @Override
  public boolean isCaseSensitive(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return true;
  }

  @Override
  public boolean isSearchable(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return false;
  }

  @Override
  public boolean isCurrency(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return false;
  }

  @Override
  public int isNullable(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.get(column - 1).col_type.nullable
            ? DatabaseMetaData.columnNullable
            : DatabaseMetaData.columnNoNulls;
  }

  @Override
  public boolean isSigned(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return true;
  }

  @Override
  public int getColumnDisplaySize(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return 100;
  }

  @Override
  public String getColumnLabel(int column) throws SQLException { //logger.debug("Entered "+ sql );
    //logger.debug("ColumnLabel is "+ sqlResult.row_set.row_desc.get(column -1).col_name);
    return sqlResult.row_set.row_desc.get(column - 1).col_name;
  }

  @Override
  public String getColumnName(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.get(column - 1).getCol_name();
  }

  @Override
  public String getSchemaName(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return null;
  }

  @Override
  public int getPrecision(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.get(column - 1).col_type.precision;
  }

  @Override
  public int getScale(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.get(column - 1).col_type.scale;
  }

  @Override
  public String getTableName(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return "tableName??";
  }

  @Override
  public String getCatalogName(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return null;
  }

  @Override
  public int getColumnType(int column) throws SQLException { //logger.debug("Entered "+ sql );
    TDatumType type = sqlResult.row_set.row_desc.get(column - 1).col_type.type;

    return MapDType.toJava(type);
  }

  @Override
  public String getColumnTypeName(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.get(column - 1).col_type.type.name();
  }

  @Override
  public boolean isReadOnly(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return true;
  }

  @Override
  public boolean isWritable(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return false;
  }

  @Override
  public boolean isDefinitelyWritable(int column) throws SQLException { //logger.debug("Entered "+ sql );
    return false;
  }

  @Override
  public String getColumnClassName(int column) throws SQLException { //logger.debug("Entered "+ sql );
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
